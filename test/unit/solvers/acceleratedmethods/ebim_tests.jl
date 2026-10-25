# Unit tests for
#   src/solvers/acceleratedmethods/ebim.jl   (_ebim_lin1_deriv, _ebim_const0_deriv, the DLP / CFIE / composite matrix derivatives of
#                                             construct_matrices(::ExpandedBIMSolver, pts, k), EBIMTaylorCache, _construct_matrices! /
#                                             construct_matrices(solver, cache, k), _solve via solve (direct and Taylor panel),
#                                             solve_wavenumber, solve_spectrum)
#   src/spectra/spectralutils.jl             (compute_spectrum(::ExpandedBIMSolver, ...) window, centre and merge handling)
# overlap_and_merge_ebim! and _local_gap are covered in spectra/spectralutils_tests.jl and not repeated here; constructor validation is
# covered in solvers/solverconstructors_tests.jl. The ExpandedBIMSolver in the current source has no `_argmin_finite` helper: solve_wavenumber
# solve_wavenumber selects the nearest finite candidate and skips non-finite ones, so the NaN handling is tested through solve_wavenumber.
# The symmetry-reduced Taylor panel is checked on a rectangle sector; the full composite Taylor panel is left to integration coverage.
#
# Matrix derivatives. The kernel derivatives of the DLP / CFIE / composite Fredholm matrices are compared with central finite differences
# of the ordinary matrix A(k) = construct_matrices(kernel, pts, k) on small boundary discretizations (a circle, a rectangle sector and an
# annulus; N = 20 to 56) at k = 3. The Taylor panel is compared with the direct derivatives through the identities
#       coeffs[:,:,1] = A(k0),   coeffs[:,:,2] = A'(k0),   coeffs[:,:,3] = A''(k0) / 2.
#
# Synthetic nonlinear eigenproblem. A test-local kernel `eb_Problem <: SweepBIMSolver` implements the extension interface of the sweep BIM
# branch (`construct_matrices`, `boundary_matrix_size`, `_bim_numeric_type`, `evaluate_points`), the direct EBIM hook
# `_ebim_construct_matrices` and the analytic Taylor hooks `_taylor_cache` / `_fredholm_taylor!`, so ExpandedBIMSolver runs unchanged and no
# billiard is solved. The Fredholm-like matrix is
#
#       A(k) = Q diag(d_m(k)) Q',    d_m(k) = (k - kappa_m)(1 + c k^2),
#
# with a fixed orthogonal Q (dense, so the roots are not visible in the matrix entries) and simple real roots kappa_m. The derivatives
# A' = Q diag(d_m') Q' and A'' = Q diag(d_m'') Q' are known in closed form and, since d_m is a cubic, a Taylor panel of degree >= 3 is exact.
# The generalized eigenproblem A0 v = lambda A1 v is diagonal in the columns of Q with lambda_m = d_m / d_m', and the left and right
# eigenvectors coincide (u = v = Q[:,m]), so the second-order EBIM estimate of root m is, in closed form,
#
#       eps1 = -d_m/d_m',   eps2 = -(eps1^2/2) d_m''/d_m',   kEBIM = k0 + eps1 + eps2,   t = |eps1 + eps2|,
#
# and the Krylov iteration returns the roots with the largest |d_m'/d_m|, i.e. the smallest |lambda_m|.
#
# Scripted kernel. A second test-local kernel `eb_Scripted` wraps a function (k, nlevels) -> (ks, ts) that replaces the local EBIM solve, so
# that the window, centre and merge handling of solve_wavenumber, solve_spectrum and compute_spectrum is tested on hand-built candidate lists.
# Helper definitions use the prefix `eb_`.

using LinearAlgebra: Diagonal, qr, norm, diag, I
using Random: MersenneTwister
using SpecialFunctions: besselj, besselh
using QuantumBilliards: construct_matrices, _bim_numeric_type, TaylorWorkspace, EBIMTaylorCache, _construct_matrices!, _solve,
                        _ebim_lin1_deriv, _ebim_const0_deriv, boundary_matrix_size, k_range_for_states

# Roots, polynomial strength and orthogonal Q of the synthetic problem.
struct eb_Problem <: QuantumBilliards.SweepBIMSolver
    kappa::Vector{Float64}
    c::Float64
    Q::Matrix{Float64}
    symmetry::Nothing
end
struct eb_Cache
    solver::eb_Problem
end
const EB_KAPPA = [1.2, 1.9, 2.5, 2.85, 3.0, 3.3, 3.6, 4.1, 4.7, 5.3, 6.0, 6.8]
const EB_N = length(EB_KAPPA)
const EB_C = 0.02
eb_problem() = eb_Problem(EB_KAPPA, EB_C, Matrix(qr(randn(MersenneTwister(13), EB_N, EB_N)).Q), nothing)

eb_circle = CircleBilliard(1.0)
const EB_PTS = evaluate_points(DoubleLayerPotentialSolver(2.0, eb_circle; min_pts=EB_N), eb_circle, 2.0)   # EBIMTaylorCache takes its matrix size from length(pts)

# d_m and its first two derivatives, and the three matrices.
eb_d(s::eb_Problem, m::Int, k) = (k - s.kappa[m]) * (1 + s.c * k^2)
eb_d1(s::eb_Problem, m::Int, k) = 3s.c * k^2 - 2s.c * s.kappa[m] * k + 1
eb_d2(s::eb_Problem, m::Int, k) = 6s.c * k - 2s.c * s.kappa[m]
eb_diagmatrix(s::eb_Problem, f, k) = ComplexF64.(s.Q * Diagonal([f(s, m, k) for m in eachindex(s.kappa)]) * s.Q')
eb_matrix(s::eb_Problem, k) = eb_diagmatrix(s, eb_d, k)
eb_dmatrix(s::eb_Problem, k) = eb_diagmatrix(s, eb_d1, k)
eb_ddmatrix(s::eb_Problem, k) = eb_diagmatrix(s, eb_d2, k)

# Closed-form second-order EBIM estimate of root m from the expansion center k0 and the magnitude of its correction.
function eb_predict(s::eb_Problem, m::Int, k0)
    ε1 = -eb_d(s, m, k0) / eb_d1(s, m, k0)
    ε2 = -(ε1^2 / 2) * eb_d2(s, m, k0) / eb_d1(s, m, k0)
    return k0 + ε1 + ε2, abs(ε1 + ε2)
end

# The roots the local solve is expected to return: the nkeep roots with the smallest |lambda_m|, sorted by their estimate.
function eb_expected(s::eb_Problem, k0, nkeep::Int)
    order = sortperm([abs(eb_d1(s, m, k0) / eb_d(s, m, k0)) for m in eachindex(s.kappa)]; rev=true)[1:nkeep]
    pred = [eb_predict(s, m, k0) for m in order]
    p = sortperm(real.(first.(pred)))
    return first.(pred)[p], last.(pred)[p], order[p]
end

# Kernel interface of the sweep BIM branch, the direct EBIM hook and the Taylor hooks.
QuantumBilliards._bim_numeric_type(::eb_Problem) = Float64
QuantumBilliards.boundary_matrix_size(s::eb_Problem, ::BoundaryPoints) = length(s.kappa)
QuantumBilliards.construct_matrices(s::eb_Problem, ::BoundaryPoints, k; multithreaded::Bool=true) = eb_matrix(s, k)
QuantumBilliards.evaluate_points(::eb_Problem, ::AbsBilliard, k) = EB_PTS
QuantumBilliards._ebim_construct_matrices(s::eb_Problem, ::BoundaryPoints, k; multithreaded::Bool=true) = (eb_matrix(s, k), eb_dmatrix(s, k), eb_ddmatrix(s, k))

# Normalized Taylor coefficients a[l+1] = A_ij^(l)(k)/l! of the Fredholm matrix entry, exact for l <= 3 and zero beyond.
QuantumBilliards._taylor_cache(s::eb_Problem, pts) = eb_Cache(s)
function QuantumBilliards._fredholm_taylor!(cache::eb_Cache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int)
    s = cache.solver; c = s.c
    fill!(w.a, 0)
    for m in eachindex(s.kappa)
        κ = s.kappa[m]; q = s.Q[i, m] * s.Q[j, m]
        w.a[1] += q * (c * k^3 - c * κ * k^2 + k - κ)
        p >= 1 && (w.a[2] += q * (3c * k^2 - 2c * κ * k + 1))
        p >= 2 && (w.a[3] += q * (3c * k - c * κ))
        p >= 3 && (w.a[4] += q * c)
    end
    return w.a
end

eb_solver(; kwargs...) = ExpandedBIMSolver(eb_problem(); kwargs...)

# Scripted kernel: `f(k, nlevels) -> (ks, ts)` replaces the local EBIM solve.
struct eb_Scripted{F} <: QuantumBilliards.SweepBIMSolver
    f::F
    symmetry::Nothing
end
QuantumBilliards._bim_numeric_type(::eb_Scripted) = Float64
QuantumBilliards.boundary_matrix_size(::eb_Scripted, ::BoundaryPoints) = 0
QuantumBilliards.evaluate_points(::eb_Scripted, ::AbsBilliard, k) = EB_PTS
QuantumBilliards.solve(s::ExpandedBIMSolver{Float64,<:eb_Scripted}, ::BoundaryPoints, k, nlevels::Int; multithreaded::Bool=true) = s.kernel.f(k, nlevels)
eb_scripted(f; kwargs...) = ExpandedBIMSolver(eb_Scripted(f, nothing); kwargs...)

# Central finite differences of the ordinary Fredholm matrix of the wrapped kernel.
eb_fd1(kernel, pts, k, h) = (construct_matrices(kernel, pts, k + h) - construct_matrices(kernel, pts, k - h)) / (2h)
eb_fd2(kernel, pts, k, h) = (construct_matrices(kernel, pts, k + h) - 2 * construct_matrices(kernel, pts, k) + construct_matrices(kernel, pts, k - h)) / h^2
eb_relerr(X, Y) = norm(X - Y) / norm(Y)

# Small real discretizations used by the matrix tests: kernel, boundary points and the expansion center.
const EB_K0 = 3.0
eb_dlp_circle() = DoubleLayerPotentialSolver(3.0, eb_circle; min_pts=20)
eb_cfie_circle() = CombinedFieldIntegralEquationSolver(3.0, eb_circle; min_pts=20)
eb_rect = RectangleBilliard(1.0, 0.6)
eb_cfie_sector() = CombinedFieldIntegralEquationSolver(5.0, eb_rect; min_pts=32, sym_characters=(-1, 1))
eb_annulus = AnnularBilliard(0.3, 0.1)
eb_composite() = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, eb_annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, eb_annulus; min_pts=24))

@testset "synthetic problem: roots, derivatives and Taylor hook" begin
    s = eb_problem()
    @test length(EB_PTS) == EB_N                                             # the token discretization matches the matrix size
    @test all(m -> norm(eb_matrix(s, s.kappa[m]) * s.Q[:, m]) < 1e-13, 1:EB_N)
    @test norm(eb_matrix(s, 2.0) * s.Q[:, 4]) > 0.1                          # away from a root the residual is not small
    h = 1e-4
    @test eb_dmatrix(s, 3.2) ≈ (eb_matrix(s, 3.2 + h) - eb_matrix(s, 3.2 - h)) / 2h atol = 1e-7
    @test eb_ddmatrix(s, 3.2) ≈ (eb_dmatrix(s, 3.2 + h) - eb_dmatrix(s, 3.2 - h)) / 2h atol = 1e-7
    cache = QuantumBilliards._taylor_cache(s, EB_PTS)
    w = TaylorWorkspace(5)
    δ = 0.37 - 0.2im
    for (i, j) in ((1, 1), (3, 7), (12, 5))
        a = QuantumBilliards._fredholm_taylor!(cache, w, 3.2, 5, i, j)
        @test sum(a[l+1] * δ^l for l in 0:5) ≈ eb_matrix(s, 3.2 + δ)[i, j]
        @test a[5] == 0 && a[6] == 0                                         # the cubic stops at degree 3
    end
end

@testset "eb_predict: closed-form EBIM estimate is third-order accurate" begin
    s = eb_problem()
    err(δ) = abs(eb_predict(s, 5, 3.0 + δ)[1] - 3.0)                       # root kappa_5 = 3.0
    @test eb_predict(s, 5, 3.0) == (3.0, 0.0)                                # at a root the correction vanishes
    @test err(0.1) / err(0.05) > 6                                           # error ~ delta^3: halving delta divides it by about 8
    @test err(0.05) / err(0.025) > 6
    @test err(0.05) < 0.05 / 100                                             # far below the first-order error ~ delta
end

@testset "_ebim_lin1_deriv: value, first and second k-derivative of a k Z1(kr)/r" begin
    r, a, h = 0.7, 0.9, 1e-3
    invr = inv(r)

    @testset "Bessel J (real prefactor)" begin
        f(k) = a * k * besselj(1, k * r) / r
        k = 2.3
        val, dval, ddval = _ebim_lin1_deriv(a * k, r, invr, besselj(0, k * r), besselj(1, k * r), k)
        @test val ≈ f(k)
        @test dval ≈ (f(k + h) - f(k - h)) / 2h rtol = 1e-6
        @test ddval ≈ (f(k + h) - 2f(k) + f(k - h)) / h^2 rtol = 1e-5
    end

    @testset "Hankel H1 (complex prefactor)" begin
        f(k) = im * k / 2 * 0.4 * besselh(1, 1, k * r) / r                   # alpha = i k inn / 2 with inn = 0.4
        k = 3.1
        val, dval, ddval = _ebim_lin1_deriv(im * k / 2 * 0.4, r, invr, besselh(0, 1, k * r), besselh(1, 1, k * r), k)
        @test val ≈ f(k)
        @test dval ≈ (f(k + h) - f(k - h)) / 2h rtol = 1e-6
        @test ddval ≈ (f(k + h) - 2f(k) + f(k - h)) / h^2 rtol = 1e-5
    end
end

@testset "_ebim_const0_deriv: value, first and second k-derivative of beta Z0(kr)" begin
    r, β, h = 0.7, -0.35, 1e-3

    @testset "Bessel J" begin
        f(k) = β * besselj(0, k * r)
        k = 2.3
        val, dval, ddval = _ebim_const0_deriv(β, r, k, besselj(0, k * r), besselj(1, k * r))
        @test val ≈ f(k)
        @test dval ≈ (f(k + h) - f(k - h)) / 2h rtol = 1e-6
        @test ddval ≈ (f(k + h) - 2f(k) + f(k - h)) / h^2 rtol = 1e-5
    end

    @testset "Hankel H0 (complex prefactor)" begin
        βc = 0.5im * 1.3
        f(k) = βc * besselh(0, 1, k * r)
        k = 3.1
        val, dval, ddval = _ebim_const0_deriv(βc, r, k, besselh(0, 1, k * r), besselh(1, 1, k * r))
        @test val ≈ f(k)
        @test dval ≈ (f(k + h) - f(k - h)) / 2h rtol = 1e-6
        @test ddval ≈ (f(k + h) - 2f(k) + f(k - h)) / h^2 rtol = 1e-5
    end
end

@testset "construct_matrices(::ExpandedBIMSolver, pts, k): DLP on a circle" begin
    kernel = eb_dlp_circle()
    solver = ExpandedBIMSolver(kernel)
    pts = evaluate_points(solver, eb_circle, EB_K0)
    A, dA, ddA = construct_matrices(solver, pts, EB_K0; multithreaded=false)
    N = length(pts)

    @testset "shapes and value equal the ordinary Fredholm matrix" begin
        @test size(A) == size(dA) == size(ddA) == (N, N)
        @test A isa Matrix{ComplexF64} && dA isa Matrix{ComplexF64} && ddA isa Matrix{ComplexF64}
        @test A ≈ construct_matrices(kernel, pts, EB_K0; multithreaded=false) atol = 1e-12
    end

    @testset "first and second derivative equal central finite differences" begin
        @test eb_relerr(dA, eb_fd1(kernel, pts, EB_K0, 1e-3)) < 1e-5
        @test eb_relerr(ddA, eb_fd2(kernel, pts, EB_K0, 1e-3)) < 1e-5
    end

    @testset "the geometric diagonal is independent of k" begin
        @test all(iszero, diag(dA)) && all(iszero, diag(ddA))
    end

    @testset "threaded equals serial" begin
        At, dAt, ddAt = construct_matrices(solver, pts, EB_K0; multithreaded=true)
        @test At ≈ A atol = 1e-13
        @test dAt ≈ dA atol = 1e-13
        @test ddAt ≈ ddA atol = 1e-13
    end
end

@testset "construct_matrices(::ExpandedBIMSolver, pts, k): CFIE on a circle" begin
    kernel = eb_cfie_circle()
    solver = ExpandedBIMSolver(kernel)
    pts = evaluate_points(solver, eb_circle, EB_K0)
    A, dA, ddA = construct_matrices(solver, pts, EB_K0; multithreaded=false)

    @testset "value equals the ordinary Fredholm matrix" begin
        @test A ≈ construct_matrices(kernel, pts, EB_K0; multithreaded=false) atol = 1e-12
    end

    @testset "first and second derivative equal central finite differences (including the Kress diagonal)" begin
        @test eb_relerr(dA, eb_fd1(kernel, pts, EB_K0, 1e-3)) < 1e-5
        @test eb_relerr(ddA, eb_fd2(kernel, pts, EB_K0, 1e-3)) < 1e-5
        @test norm(diag(dA)) > 1e-3                                          # the diagonal of the CFIE derivative is not zero
    end

    @testset "threaded equals serial" begin
        At, dAt, ddAt = construct_matrices(solver, pts, EB_K0; multithreaded=true)
        @test At ≈ A atol = 1e-13
        @test dAt ≈ dA atol = 1e-13
        @test ddAt ≈ ddA atol = 1e-13
    end
end

@testset "construct_matrices(::ExpandedBIMSolver, pts, k): symmetry-reduced CFIE on a rectangle sector" begin
    kernel = eb_cfie_sector()
    solver = ExpandedBIMSolver(kernel)
    pts = evaluate_points(solver, eb_rect, EB_K0)
    A, dA, ddA = construct_matrices(solver, pts, EB_K0; multithreaded=false)

    @test size(A) == (length(pts) ÷ 4, length(pts) ÷ 4)                      # X and Y reflection fold the boundary onto one quarter
    @test A ≈ construct_matrices(kernel, pts, EB_K0; multithreaded=false) atol = 1e-12
    @test eb_relerr(dA, eb_fd1(kernel, pts, EB_K0, 1e-3)) < 1e-5
    @test eb_relerr(ddA, eb_fd2(kernel, pts, EB_K0, 1e-3)) < 1e-5
end

@testset "construct_matrices(::ExpandedBIMSolver, pts, k): composite annulus with a DLP outer and a CFIE hole component" begin
    kernel = eb_composite()
    solver = ExpandedBIMSolver(kernel)
    pts = evaluate_points(solver, eb_annulus, EB_K0)
    A, dA, ddA = construct_matrices(solver, pts, EB_K0; multithreaded=false)

    @test size(A) == (56, 56)                                                # 32 outer + 24 hole nodes
    @test A ≈ construct_matrices(kernel, pts, EB_K0; multithreaded=false) atol = 1e-12
    @test eb_relerr(dA, eb_fd1(kernel, pts, EB_K0, 1e-3)) < 1e-5            # includes the smooth cross-component blocks
    @test eb_relerr(ddA, eb_fd2(kernel, pts, EB_K0, 1e-3)) < 1e-5
end

@testset "EBIMTaylorCache: construction" begin
    s = eb_problem()

    @testset "synthetic coefficients are the normalized Taylor coefficients" begin
        solver = eb_solver()
        cache = EBIMTaylorCache(s, EB_PTS, 3.0, 5; multithreaded=false)
        @test cache isa EBIMTaylorCache
        @test (cache.k0, cache.degree) == (3.0, 5)
        @test size(cache.coeffs) == (EB_N, EB_N, 6)
        @test cache.coeffs[:, :, 1] ≈ eb_matrix(s, 3.0) atol = 1e-12
        @test cache.coeffs[:, :, 2] ≈ eb_dmatrix(s, 3.0) atol = 1e-12
        @test cache.coeffs[:, :, 3] ≈ eb_ddmatrix(s, 3.0) / 2 atol = 1e-12
        @test cache.coeffs[:, :, 4] ≈ s.c * (s.Q * s.Q') atol = 1e-12        # d_m''' / 3! = c for every m
        @test iszero(cache.coeffs[:, :, 5:6])
    end

    @testset "multithreaded and serial construction agree" begin
        c1 = EBIMTaylorCache(s, EB_PTS, 3.0, 4; multithreaded=false)
        c2 = EBIMTaylorCache(s, EB_PTS, 3.0, 4; multithreaded=true)
        @test c1.coeffs ≈ c2.coeffs atol = 1e-14
    end

    @testset "degrees below 2 are rejected" begin
        @test_throws "at least 2" EBIMTaylorCache(s, EB_PTS, 3.0, 1)
        @test_throws ArgumentError EBIMTaylorCache(s, EB_PTS, 3.0, 0)
    end

    @testset "real DLP kernel: coefficients equal the direct matrix and its derivatives" begin
        kernel = eb_dlp_circle()
        solver = ExpandedBIMSolver(kernel)
        pts = evaluate_points(solver, eb_circle, EB_K0)
        A, dA, ddA = construct_matrices(solver, pts, EB_K0; multithreaded=false)
        cache = EBIMTaylorCache(kernel, pts, EB_K0, 8; multithreaded=false)
        @test eb_relerr(cache.coeffs[:, :, 1], A) < 1e-10
        @test eb_relerr(cache.coeffs[:, :, 2], dA) < 1e-10
        @test eb_relerr(cache.coeffs[:, :, 3], ddA / 2) < 1e-10
    end

    @testset "real symmetry-reduced CFIE kernel: orbit-folded coefficients equal the reduced direct matrices" begin
        kernel = eb_cfie_sector()
        solver = ExpandedBIMSolver(kernel)
        pts = evaluate_points(solver, eb_rect, EB_K0)
        A, dA, ddA = construct_matrices(solver, pts, EB_K0; multithreaded=false)
        cache = EBIMTaylorCache(kernel, pts, EB_K0, 8; multithreaded=false)
        @test size(cache.coeffs) == (size(A, 1), size(A, 2), 9)
        @test eb_relerr(cache.coeffs[:, :, 1], A) < 1e-10
        @test eb_relerr(cache.coeffs[:, :, 2], dA) < 1e-10
        @test eb_relerr(cache.coeffs[:, :, 3], ddA / 2) < 1e-10
    end
end

@testset "_construct_matrices! / construct_matrices(solver, cache, k): Horner evaluation" begin
    s = eb_problem()
    solver = eb_solver(taylor_radius=0.5)
    cache = EBIMTaylorCache(s, EB_PTS, 3.0, 5; multithreaded=false)

    @testset "exact degree reproduces A, A' and A'' inside the radius" begin
        for k in (3.0, 3.2, 2.7, 3.5)
            A, dA, ddA = construct_matrices(solver, cache, k)
            @test A ≈ eb_matrix(s, k) atol = 1e-11
            @test dA ≈ eb_dmatrix(s, k) atol = 1e-11
            @test ddA ≈ eb_ddmatrix(s, k) atol = 1e-11
        end
    end

    @testset "the in-place method fills and returns its buffers" begin
        A = Matrix{ComplexF64}(undef, EB_N, EB_N); dA = similar(A); ddA = similar(A)
        out = _construct_matrices!(solver, A, dA, ddA, cache, 3.3)
        @test out[1] === A && out[2] === dA && out[3] === ddA
        @test A ≈ eb_matrix(s, 3.3) atol = 1e-11
    end

    @testset "a wavenumber outside the Taylor radius is rejected" begin
        @test_throws "outside the Taylor radius" construct_matrices(solver, cache, 3.6)
        @test_throws ArgumentError construct_matrices(solver, cache, 2.4)
        @test_throws ArgumentError construct_matrices(eb_solver(taylor_radius=0.1), cache, 3.2)
    end

    @testset "a truncated degree misses exactly the cubic term c delta^3 I" begin
        cache2 = EBIMTaylorCache(s, EB_PTS, 3.0, 2; multithreaded=false)
        A, _, _ = construct_matrices(solver, cache2, 3.4)
        @test eb_matrix(s, 3.4) - A ≈ s.c * 0.4^3 * I atol = 1e-12
    end

    @testset "real kernel: the Taylor panel agrees with the direct derivatives away from its center" begin
        kernel = eb_cfie_circle()
        esolver = ExpandedBIMSolver(kernel; taylor_degree=16)
        pts = evaluate_points(esolver, eb_circle, EB_K0)
        pcache = EBIMTaylorCache(kernel, pts, EB_K0, 16; multithreaded=false)
        A, dA, ddA = construct_matrices(esolver, pcache, EB_K0 + 0.1)
        Ad, dAd, ddAd = construct_matrices(esolver, pts, EB_K0 + 0.1; multithreaded=false)
        @test eb_relerr(A, Ad) < 1e-9
        @test eb_relerr(dA, dAd) < 1e-9
        @test eb_relerr(ddA, ddAd) < 1e-9
    end
end

@testset "_solve: local second-order EBIM problem" begin
    s = eb_problem()
    solver = eb_solver()
    k0, nlevels = 3.1, 3

    @testset "estimates and corrections equal the closed form for the nearest roots" begin
        ks, ts = _solve(solver, eb_matrix(s, k0), eb_dmatrix(s, k0), eb_ddmatrix(s, k0), k0, nlevels)
        kexp, texp, _ = eb_expected(s, k0, nlevels + 5)
        @test ks isa Vector{ComplexF64} && ts isa Vector{Float64}
        @test length(ks) == length(ts) == nlevels + 5                        # nlevels + 5 candidates are kept
        p = sortperm(real.(ks))
        @test real.(ks[p]) ≈ real.(kexp) atol = 1e-8
        @test all(<(1e-8), abs.(imag.(ks)))                                  # real problem, real roots
        @test ts[p] ≈ texp atol = 1e-8
        @test all(q -> ts[q] ≈ abs(ks[q] - k0), eachindex(ks))               # t is the magnitude of the total correction
    end

    @testset "the nearest roots are well resolved" begin
        ks, _ = _solve(solver, eb_matrix(s, k0), eb_dmatrix(s, k0), eb_ddmatrix(s, k0), k0, nlevels)
        for κ in (3.0, 3.3)
            @test minimum(abs.(real.(ks) .- κ)) < 1e-3
        end
    end

    @testset "at a root the correction vanishes" begin
        ks, ts = _solve(solver, eb_matrix(s, 3.0), eb_dmatrix(s, 3.0), eb_ddmatrix(s, 3.0), 3.0, nlevels)
        q = argmin(abs.(real.(ks) .- 3.0))
        @test real(ks[q]) ≈ 3.0 atol = 1e-8
        @test ts[q] < 1e-6
    end

    @testset "the number of candidates is capped by the matrix size" begin
        ks, ts = _solve(solver, eb_matrix(s, k0), eb_dmatrix(s, k0), eb_ddmatrix(s, k0), k0, EB_N)
        @test length(ks) == length(ts) == EB_N - 1                           # nev = min(nlevels + 5, N - 1) = N - 1
    end

    @testset "matrices smaller than 2 x 2 give empty outputs" begin
        ks, ts = _solve(solver, ComplexF64[2.0;;], ComplexF64[1.0;;], ComplexF64[0.0;;], 1.0, 1)
        @test ks isa Vector{ComplexF64} && isempty(ks)
        @test ts isa Vector{Float64} && isempty(ts)
    end

    @testset "A is overwritten by its LU factorization" begin
        A = eb_matrix(s, k0); A0 = copy(A)
        _solve(solver, A, eb_dmatrix(s, k0), eb_ddmatrix(s, k0), k0, nlevels)
        @test A != A0
    end
end

@testset "solve(::ExpandedBIMSolver, ...): direct matrices and Taylor panel" begin
    s = eb_problem()
    k0, nlevels = 3.1, 3
    kexp, texp, _ = eb_expected(s, k0, nlevels + 5)

    @testset "direct construction returns the closed-form estimates" begin
        ks, ts = solve(eb_solver(), EB_PTS, k0, nlevels; multithreaded=false)
        p = sortperm(real.(ks))
        @test real.(ks[p]) ≈ real.(kexp) atol = 1e-8
        @test ts[p] ≈ texp atol = 1e-8
    end

    @testset "a Taylor panel centered elsewhere gives the same estimates" begin
        solver = eb_solver(taylor_radius=0.5)
        cache = EBIMTaylorCache(s, EB_PTS, 3.0, 5; multithreaded=false)
        ks, ts = solve(solver, cache, k0, nlevels)
        p = sortperm(real.(ks))
        @test real.(ks[p]) ≈ real.(kexp) atol = 1e-8
        @test ts[p] ≈ texp atol = 1e-8
    end

    @testset "a center outside the panel radius is rejected" begin
        cache = EBIMTaylorCache(s, EB_PTS, 3.0, 5; multithreaded=false)
        @test_throws ArgumentError solve(eb_solver(taylor_radius=0.05), cache, k0, nlevels)
    end

    @testset "real CFIE kernel: the nearest estimate lies near the circle level" begin
        # Circle (radius 1): the CFIE kernel has the Dirichlet levels j_{0,1} = 2.4048256 and j_{1,1} = 3.8317060 as roots.
        solver = ExpandedBIMSolver(eb_cfie_circle())
        pts = evaluate_points(solver, eb_circle, 2.4)
        ks, ts = solve(solver, pts, 2.4, 1; multithreaded=false)
        q = argmin(abs.(real.(ks) .- 2.4))
        @test real(ks[q]) ≈ 2.404825557695773 atol = 1e-4                    # j_{0,1}, analytic
        @test ts[q] < 0.1
    end
end

@testset "solve_wavenumber: candidate selection on hand-built inputs" begin
    k, dk = 10.2, 1.0

    @testset "the candidate with the smallest |Re(k_i) - k| is returned with its tension" begin
        solver = eb_scripted((k0, nl) -> (ComplexF64[9.0, 10.15 + 1e-9im, 11.0], [0.1, 2e-3, 0.3]))
        kb, tb = solve_wavenumber(solver, eb_circle, k, dk; multithreaded=false)
        @test kb == 10.15 + 1e-9im
        @test tb == 2e-3
    end

    @testset "an empty candidate list gives (nothing, nothing)" begin
        solver = eb_scripted((k0, nl) -> (ComplexF64[], Float64[]))
        @test solve_wavenumber(solver, eb_circle, k, dk; multithreaded=false) === (nothing, nothing)
    end

    @testset "the requested number of local levels is the larger of nlevels and the Weyl count" begin
        seen = Int[]
        solver = eb_scripted((k0, nl) -> (push!(seen, nl); (ComplexF64[10.2], [1e-3])))
        solve_wavenumber(solver, eb_circle, k, dk; multithreaded=false)             # circle: k dk = 10.2 -> ceil = 11 > 5
        solve_wavenumber(solver, eb_circle, k, dk; multithreaded=false, nlevels=20)
        solve_wavenumber(solver, eb_circle, 3.0, 0.1; multithreaded=false)          # Weyl count 0.3 -> ceil = 1 < 5
        @test seen == [11, 20, 5]
    end

    @testset "the solve is performed at the center k" begin
        centers = Float64[]
        solver = eb_scripted((k0, nl) -> (push!(centers, k0); (ComplexF64[k0], [1e-3])))
        solve_wavenumber(solver, eb_circle, 7.25, 0.5; multithreaded=false)
        @test centers == [7.25]
    end

    @testset "a NaN candidate does not hide the genuine nearest candidate" begin
        # Non-finite candidates are skipped; the nearest finite candidate 10.15 is returned.
        solver = eb_scripted((k0, nl) -> (ComplexF64[NaN, 10.15, 9.0], [1e-3, 2e-3, 0.1]))
        kb, tb = solve_wavenumber(solver, eb_circle, k, dk; multithreaded=false)
        @test kb == 10.15 + 0im
        @test tb == 2e-3
    end

    @testset "all candidates NaN gives (nothing, nothing)" begin
        solver = eb_scripted((k0, nl) -> (ComplexF64[NaN, NaN], [1e-3, 2e-3]))
        @test solve_wavenumber(solver, eb_circle, k, dk; multithreaded=false) === (nothing, nothing)
    end
end

@testset "solve_spectrum: window handling on hand-built inputs" begin
    k, dk = 10.2, 0.5
    ks_in = ComplexF64[9.6, 9.7, 10.2, 10.7, 10.8, 10.2 + 5e-7im]
    ts_in = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3]

    @testset "candidates within |Re(k_i) - k| <= dk are kept in order, the others dropped" begin
        solver = eb_scripted((k0, nl) -> (ks_in, ts_in))
        ks, ts = solve_spectrum(solver, eb_circle, k, dk; multithreaded=false)
        @test ks == ComplexF64[9.7, 10.2, 10.7, 10.2 + 5e-7im]               # 9.6 and 10.8 lie 0.6 away; 9.7 and 10.7 exactly dk away
        @test ts == [2e-3, 3e-3, 4e-3, 6e-3]
    end

    @testset "the window is closed: candidates exactly dk from the center are kept" begin
        solver = eb_scripted((k0, nl) -> (ComplexF64[0.5, 1.5, 0.25, 1.75], [1e-3, 1e-3, 1e-3, 1e-3]))
        ks, _ = solve_spectrum(solver, eb_circle, 1.0, 0.5; multithreaded=false)
        @test ks == ComplexF64[0.5, 1.5]                                     # binary-exact distances 0.5, 0.5, 0.75, 0.75
    end

    @testset "NaN and Inf candidates are dropped" begin
        solver = eb_scripted((k0, nl) -> (ComplexF64[NaN, Inf, -Inf, 10.2, complex(NaN, 0.0)], [1e-3, 1e-3, 1e-3, 2e-3, 1e-3]))
        ks, ts = solve_spectrum(solver, eb_circle, k, dk; multithreaded=false)
        @test ks == ComplexF64[10.2]
        @test ts == [2e-3]
    end

    @testset "an empty candidate list gives empty outputs" begin
        solver = eb_scripted((k0, nl) -> (ComplexF64[], Float64[]))
        ks, ts = solve_spectrum(solver, eb_circle, k, dk; multithreaded=false)
        @test isempty(ks) && isempty(ts)
    end

    @testset "nlevels is the larger of nlevels and the Weyl count" begin
        seen = Int[]
        solver = eb_scripted((k0, nl) -> (push!(seen, nl); (ComplexF64[], Float64[])))
        solve_spectrum(solver, eb_circle, 10.2, 1.0; multithreaded=false)           # Weyl count 10.2 -> 11
        solve_spectrum(solver, eb_circle, 2.0, 0.1; multithreaded=false, nlevels=2)  # Weyl count 0.2 -> 1 < 2
        @test seen == [11, 2]
    end

    @testset "real synthetic kernel: all roots inside the window are returned" begin
        s = eb_problem()
        solver = eb_solver()
        ks, ts = solve_spectrum(solver, eb_circle, 3.1, 0.22; multithreaded=false, nlevels=3)
        @test all(q -> abs(real(ks[q]) - 3.1) <= 0.22, eachindex(ks))
        kexp, _, _ = eb_expected(s, 3.1, 8)
        inwin = filter(x -> abs(real(x) - 3.1) <= 0.22, kexp)
        @test sort(real.(ks)) ≈ sort(real.(inwin)) atol = 1e-8
        @test length(ks) == length(ts) >= 2                                  # kappa = 3.0 and 3.3 are inside
    end
end

@testset "compute_spectrum(::ExpandedBIMSolver): centers, window filter and merge on hand-built inputs" begin
    dk = 0.25
    centers = Float64[]
    nls = Int[]
    candidates = Dict(
        1.0 => (ComplexF64[1.1, 1.24, 1.2, 0.7, NaN, 0.95], [1e-3, 2e-3, 3e-3, 1e-4, 1e-4, 1e-3]),
        1.25 => (ComplexF64[1.1000004, 1.2400003, 1.45], [5e-4, 4e-3, 1e-3]),
        1.5 => (ComplexF64[1.4500002, 1.75], [2e-4, 1e-3]))
    f = (k0, nl) -> (push!(centers, k0); push!(nls, nl); candidates[k0])
    solver = eb_scripted(f; use_taylor=false)
    data = compute_spectrum(solver, eb_circle, 1.0, 1.5; dk=k -> dk, ten_tol=nothing, multithreaded=false, show_progress=false)

    @testset "expansion centers are k1, k1 + dk, ... up to k2 with the Weyl local level count" begin
        @test centers == [1.0, 1.25, 1.5]
        @test nls == [1, 1, 1]                                               # circle: k dk = 0.25, 0.3125, 0.375 -> ceil = 1
    end

    @testset "returns SpectralData with the solver and billiard attached" begin
        @test data isa SpectralData
        @test data.billiard === eb_circle && data.solver === solver
    end

    @testset "candidates outside the local window, outside [k1, k2] or non-finite are dropped, duplicates are merged" begin
        @test data.k == [1.1000004, 1.2, 1.24, 1.4500002]                    # 0.7 (outside window), 0.95 (< k1), 1.75 (> k2), NaN dropped
        @test data.ten == [5e-4, 3e-3, 2e-3, 2e-4]                           # the merged duplicate keeps the lower-tension candidate
        @test data.control == [true, false, true, true]                      # a merged cluster is flagged, the singleton 1.2 is not
        @test all(iszero, data.k_im)
    end

    @testset "the output is sorted by the real part" begin
        @test issorted(data.k)
    end

    @testset "ten_tol drops candidates with a larger correction before merging" begin
        filtered = compute_spectrum(solver, eb_circle, 1.0, 1.5; dk=k -> dk, ten_tol=2.5e-3, multithreaded=false, show_progress=false)
        @test all(filtered.ten .<= 2.5e-3)
        @test filtered.k == [1.1000004, 1.24, 1.4500002]                     # 1.2 (3e-3) is discarded
        default = compute_spectrum(solver, eb_circle, 1.0, 1.5; dk=k -> dk, multithreaded=false, show_progress=false)
        @test all(default.ten .<= 5e-4)
        @test default.k == [1.1000004, 1.4500002]
    end

    @testset "invalid arguments" begin
        @test_throws "require k1 < k2" compute_spectrum(solver, eb_circle, 1.5, 1.0; show_progress=false)
        @test_throws "dk(k) must be positive" compute_spectrum(solver, eb_circle, 1.0, 1.5; dk=k -> 0.0, show_progress=false)
    end

    @testset "no candidate in the interval returns an empty spectrum" begin
        empty_solver = eb_scripted((k0, nl) -> (ComplexF64[], Float64[]); use_taylor=false)
        data = compute_spectrum(empty_solver, eb_circle, 1.0, 1.5; dk=k -> dk, show_progress=false)
        @test length(data) == 0
        outside = eb_scripted((k0, nl) -> (ComplexF64[k0 + 0.2], [1e-4]); use_taylor=false)   # every candidate above k2
        data = compute_spectrum(outside, eb_circle, 1.0, 1.0 + 1e-9; dk=k -> dk, show_progress=false)
        @test length(data) == 0
    end

    @testset "the number-of-states interface uses the Weyl-law wavenumber range" begin
        g = eb_scripted((k0, nl) -> (ComplexF64[k0], [1e-4]); use_taylor=false)
        k1, k2 = k_range_for_states(eb_circle, 5, 12; fundamental=false)
        byk = compute_spectrum(g, eb_circle, k1, k2; dk=k -> 0.05, multithreaded=false, show_progress=false)
        byn = compute_spectrum(g, eb_circle, 5, 12; dk=k -> 0.05, multithreaded=false, show_progress=false)
        @test length(byk) > 0
        @test byn.k == byk.k && byn.ten == byk.ten && byn.control == byk.control
    end
end

@testset "compute_spectrum(::ExpandedBIMSolver): synthetic problem, direct and Taylor-panel paths" begin
    s = eb_problem()
    k1, k2, dk = 2.7, 3.4, 0.05
    roots = [2.85, 3.0, 3.3]                                                 # simple roots inside [k1, k2], all at least 0.15 from the edges
    ms = [4, 5, 6]
    # The returned level of a root comes from an expansion center at most about 1.2 dk away; its error is therefore bounded by the
    # closed-form estimate error at that distance.
    bound(m) = 1.5 * max(abs(eb_predict(s, m, s.kappa[m] + 1.2dk)[1] - s.kappa[m]), abs(eb_predict(s, m, s.kappa[m] - 1.2dk)[1] - s.kappa[m]))

    function check(data)
        @test issorted(data.k)
        @test all(k -> k1 <= k <= k2, data.k)
        for (κ, m) in zip(roots, ms)
            @test minimum(abs.(data.k .- κ)) <= bound(m)
        end
        @test all(<(1e-8), abs.(data.k_im))
    end

    @testset "direct matrix construction" begin
        data = compute_spectrum(eb_solver(use_taylor=false), eb_circle, k1, k2; dk=k -> dk, multithreaded=false, show_progress=false)
        check(data)
    end

    @testset "Taylor panels (several panels: 2 taylor_radius is shorter than the interval)" begin
        data = compute_spectrum(eb_solver(use_taylor=true, taylor_degree=5, taylor_radius=0.1), eb_circle, k1, k2; dk=k -> dk, multithreaded=false, show_progress=false)
        check(data)
    end
end
