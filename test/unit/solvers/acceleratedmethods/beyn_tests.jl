# Unit tests for
#   src/solvers/acceleratedmethods/beyn.jl   (construct_matrices(::BeynSolver, ...) contour moments, _beyn_projected_solve,
#                                             _beyn_solve_core via solve / solve_vectors, _beyn_imag_k_check,
#                                             validate_beyn_taylor, _tune_beyn_taylor_degree!, solve_wavenumber, solve_spectrum)
# Weyl window planning (plan_weyl_windows, beyn_disks_from_windows, beyn_buffer_matrices) is covered in
# spectra/spectralutils_tests.jl and not repeated here; constructor validation is covered in solvers/solverconstructors_tests.jl.
#
# Synthetic nonlinear eigenproblem. A test-local kernel `bn_Problem <: SweepBIMSolver` implements the extension interface of the
# sweep BIM branch (`construct_matrices`, `boundary_matrix_size`, `_bim_numeric_type`, `evaluate_points` and the Taylor contour
# hook), so BeynSolver runs unchanged and no billiard is solved. The Fredholm-like matrix is
#
#       A(k) = Q diag(d_m(k)) Q',    d_m(k) = (k - kappa_m)(1 + c k^2),
#
# with a fixed orthogonal Q (dense, so the roots are not visible in the matrix entries). Its eigenvalues are exactly
#   * the real roots kappa_m (kappa_3 = kappa_4 is a semisimple double root, eigenvectors Q[:,3], Q[:,4]),
#   * the two extra roots k = +-i/sqrt(c), far from every contour used below.
# Since d_m is a cubic, the Taylor expansion is exact from degree 3 on; the contour moments follow from residues,
#       (1/2 pi i) oint 1/d_m(z) dz   = 1/(1 + c kappa_m^2)       if kappa_m is inside the contour, 0 otherwise,
#       (1/2 pi i) oint z/d_m(z) dz   = kappa_m/(1 + c kappa_m^2) if kappa_m is inside the contour, 0 otherwise,
# so A0 = Q diag(...) Q' V and A1 = Q diag(...) Q' V are known in closed form for the probing matrix V.
# The kernel only receives a `BoundaryPoints` token (one tiny circle discretization, never used numerically).
# Helper definitions use the prefix `bn_`.

using LinearAlgebra: Diagonal, qr, norm, svdvals, opnorm
using Random: MersenneTwister
using QuantumBilliards: construct_matrices, _beyn_projected_solve, _beyn_solve_core, _beyn_imag_k_check,
                        validate_beyn_taylor, _tune_beyn_taylor_degree!, _bim_numeric_type

# Roots, polynomial strength and orthogonal Q of the synthetic problem.
struct bn_Problem <: QuantumBilliards.SweepBIMSolver
    kappa::Vector{Float64}
    c::Float64
    Q::Matrix{Float64}
    symmetry::Nothing
end
const BN_KAPPA = [1.5, 2.2, 2.7, 3.0, 3.0, 3.3, 3.8, 4.6, 6.0]
const BN_N = length(BN_KAPPA)
const BN_C = 0.2
bn_problem() = bn_Problem(BN_KAPPA, BN_C, Matrix(qr(randn(MersenneTwister(7), BN_N, BN_N)).Q), nothing)

bn_circle = CircleBilliard(1.0)
const BN_PTS = evaluate_points(DoubleLayerPotentialSolver(2.0, bn_circle; min_pts=16), bn_circle, 2.0)

# Kernel interface of the sweep BIM branch.
bn_d(s::bn_Problem, m::Int, k) = (k - s.kappa[m]) * (1 + s.c * k^2)
bn_matrix(s::bn_Problem, k) = ComplexF64.(s.Q * Diagonal([bn_d(s, m, k) for m in eachindex(s.kappa)]) * s.Q')
QuantumBilliards._bim_numeric_type(::bn_Problem) = Float64
QuantumBilliards.boundary_matrix_size(s::bn_Problem, ::BoundaryPoints) = length(s.kappa)
QuantumBilliards.construct_matrices(s::bn_Problem, ::BoundaryPoints, k; multithreaded::Bool=true) = bn_matrix(s, k)
QuantumBilliards.evaluate_points(::bn_Problem, ::AbsBilliard, k) = BN_PTS

# Normalized Taylor coefficients a[l+1][m] (l = 0..3) of d_m(k0 + delta), from d_m(k) = c k^3 - c kappa k^2 + k - kappa.
function bn_taylor_coeffs(s::bn_Problem, k0::Float64)
    c = s.c
    return [[c * k0^3 - c * κ * k0^2 + k0 - κ for κ in s.kappa],
            [3c * k0^2 - 2c * κ * k0 + 1 for κ in s.kappa],
            [3c * k0 - c * κ for κ in s.kappa],
            fill(c, length(s.kappa))]
end

# Taylor contour hook used by BeynSolver(use_taylor=true): truncation at degree p, so p < 3 is deliberately inexact.
function QuantumBilliards._construct_matrices_multi_k_taylor(s::bn_Problem, ::BoundaryPoints{Float64}, k0::Float64, R::Float64,
                                                           nq::Int, p::Int; multithreaded::Bool=true)
    a = bn_taylor_coeffs(s, k0)
    zs = ComplexF64.(k0 .+ R .* cis.(2π .* (0:nq-1) ./ nq))
    return [ComplexF64.(s.Q * Diagonal([sum(a[l+1][m] * (z - k0)^l for l in 0:min(p, 3)) for m in eachindex(s.kappa)]) * s.Q')
            for z in zs]
end

bn_solver(; kwargs...) = BeynSolver(bn_problem(); r=6, nq=64, kwargs...)

# Residue prediction of the contour moments for the probing matrix V.
function bn_moments(s::bn_Problem, k0, R, V)
    inside = abs.(s.kappa .- k0) .< R
    A0 = s.Q * Diagonal(inside ./ (1 .+ s.c .* s.kappa .^ 2)) * s.Q' * V
    A1 = s.Q * Diagonal(inside .* s.kappa ./ (1 .+ s.c .* s.kappa .^ 2)) * s.Q' * V
    return A0, A1
end

# Contour used by most tests: center 3, radius 0.5, containing kappa = 2.7, 3.0 (double), 3.3; the nearest excluded roots
# 2.2 and 3.8 lie 1.6 radii from the center.
const BN_K0, BN_DK = 3.0, 1.0
const BN_INSIDE = [2.7, 3.0, 3.0, 3.3]

@testset "synthetic problem: roots and eigenvectors" begin
    s = bn_problem()
    @test all(m -> norm(bn_matrix(s, s.kappa[m]) * s.Q[:, m]) < 1e-13, 1:BN_N)
    @test norm(bn_matrix(s, 2.0) * s.Q[:, 3]) > 0.1                       # away from a root the residual is not small
    @test all(m -> isapprox(abs(bn_d(s, m, im / sqrt(s.c))), 0; atol=1e-12), 1:BN_N)    # the extra roots +-i/sqrt(c)
    a = bn_taylor_coeffs(s, 3.0)
    z = 3.4 + 0.1im
    @test all(m -> sum(a[l+1][m] * (z - 3.0)^l for l in 0:3) ≈ bn_d(s, m, z), 1:BN_N)  # the cubic is exact at degree 3
end

@testset "construct_matrices(::BeynSolver): contour moments" begin
    s = bn_problem()
    V = randn(MersenneTwister(0), ComplexF64, BN_N, 6)                       # the default probing matrix of the solver

    @testset "Taylor backend reproduces the residue moments" begin
        solver = bn_solver(use_taylor=true, taylor_degree=3)
        A0, A1 = construct_matrices(solver, BN_PTS, BN_K0, 0.5; multithreaded=false)
        A0e, A1e = bn_moments(s, BN_K0, 0.5, V)
        @test size(A0) == size(A1) == (BN_N, 6)
        @test A0 ≈ A0e atol = 1e-8
        @test A1 ≈ A1e atol = 1e-8
    end

    @testset "direct backend agrees with the Taylor backend" begin
        A0d, A1d = construct_matrices(bn_solver(use_taylor=false), BN_PTS, BN_K0, 0.5; multithreaded=false)
        A0t, A1t = construct_matrices(bn_solver(use_taylor=true, taylor_degree=3), BN_PTS, BN_K0, 0.5; multithreaded=false)
        @test A0d ≈ A0t atol = 1e-10
        @test A1d ≈ A1t atol = 1e-10
    end

    @testset "numerical rank counts multiplicity" begin
        A0, _ = construct_matrices(bn_solver(use_taylor=false), BN_PTS, BN_K0, 0.5; multithreaded=false)
        σ = svdvals(A0)
        @test count(>=(1e-8), σ) == 4                                        # 2.7, 3.0, 3.0, 3.3
        @test σ[4] > 1e-3 > σ[5]
    end

    @testset "rank-saturated probe grows the probing rank" begin
        solver = BeynSolver(bn_problem(); r=3, nq=64, use_taylor=false)      # rank 4 > r = 3
        A0, A1 = construct_matrices(solver, BN_PTS, BN_K0, 0.5; multithreaded=false)
        @test size(A0, 2) == size(A1, 2) == 6                                # r is raised by r until the rank is resolved
        @test count(>=(solver.svd_tol), svdvals(A0)) == 4
    end

    @testset "rank-saturation at the maximum probe rank throws" begin
        solver = BeynSolver(bn_problem(); r=6, nq=64, use_taylor=false)
        @test_throws ArgumentError construct_matrices(solver, BN_PTS, 3.75, 2.6; multithreaded=false)   # all nine roots inside
    end

    @testset "an empty contour has numerical rank zero" begin
        solver = bn_solver(use_taylor=false)
        A0, _ = construct_matrices(solver, BN_PTS, 5.2, 0.2; multithreaded=false)
        @test norm(A0) < 1e-8
    end
end

@testset "_beyn_projected_solve: roots inside the contour" begin
    for use_taylor in (true, false)
        solver = bn_solver(use_taylor=use_taylor, taylor_degree=3)
        ks, X = _beyn_projected_solve(solver, BN_PTS, BN_K0, BN_DK; multithreaded=false)
        order = sortperm(real.(ks))
        @testset "use_taylor = $use_taylor" begin
            @test ks isa Vector{ComplexF64}
            @test length(ks) == 4
            @test real.(ks[order]) ≈ BN_INSIDE atol = 1e-7
            @test all(<(1e-7), abs.(imag.(ks)))
            @test size(X) == (BN_N, 4)
        end
    end

    @testset "roots outside the contour are excluded" begin
        ks, _ = _beyn_projected_solve(bn_solver(use_taylor=false), BN_PTS, BN_K0, BN_DK; multithreaded=false)
        @test all(k -> abs(k - BN_K0) <= 0.5, ks)
        @test !any(k -> abs(k - 2.2) < 0.1 || abs(k - 3.8) < 0.1, ks)
    end

    @testset "contour radius is dk / 2 and the contour moves with k0" begin
        ks, _ = _beyn_projected_solve(bn_solver(use_taylor=false), BN_PTS, 3.0, 0.5; multithreaded=false)   # radius 0.25
        @test sort(real.(ks)) ≈ [3.0, 3.0] atol = 1e-6
        ks, _ = _beyn_projected_solve(bn_solver(use_taylor=false), BN_PTS, 4.2, 1.0; multithreaded=false)   # radius 0.5: 3.8 and 4.6
        @test sort(real.(ks)) ≈ [3.8, 4.6] atol = 1e-6
    end

    @testset "a contour without roots returns empty outputs" begin
        ks, X = _beyn_projected_solve(bn_solver(use_taylor=false), BN_PTS, 5.2, 0.4; multithreaded=false)
        @test isempty(ks)
        @test size(X) == (BN_N, 0)
    end

    @testset "svd_tol larger than every singular value leaves rank zero" begin
        ks, X = _beyn_projected_solve(bn_solver(use_taylor=false, svd_tol=1e3), BN_PTS, BN_K0, BN_DK; multithreaded=false)
        @test isempty(ks)
        @test size(X) == (BN_N, 0)
    end

    @testset "svd_tol = 0 can never resolve the rank" begin
        @test_throws ArgumentError _beyn_projected_solve(bn_solver(use_taylor=false, svd_tol=0.0), BN_PTS, BN_K0, BN_DK; multithreaded=false)
    end
end

@testset "solve_vectors / solve: validated roots and eigenvectors" begin
    s = bn_problem()
    solver = bn_solver(use_taylor=true, taylor_degree=3)
    ks, res, X = solve_vectors(solver, BN_PTS, BN_K0, BN_DK; multithreaded=false)
    order = sortperm(real.(ks))
    ks, res, X = ks[order], res[order], X[:, order]

    @testset "roots, residual vector and shapes" begin
        @test length(ks) == length(res) == 4
        @test size(X) == (BN_N, 4)
        @test real.(ks) ≈ BN_INSIDE atol = 1e-7
        @test all(<(solver.res_tol), res)
        @test eltype(res) == Float64
    end

    @testset "eigenvector residuals" begin
        for j in 1:4
            x = X[:, j]
            @test norm(bn_matrix(s, ks[j]) * x) / norm(x) < 1e-7
            @test res[j] ≈ norm(bn_matrix(s, ks[j]) * x) atol = 1e-12              # reported residual is the one just computed
        end
    end

    @testset "simple roots recover the known eigenvectors" begin
        for (j, m) in ((1, 3), (4, 6))                                             # kappa = 2.7 -> Q[:,3], kappa = 3.3 -> Q[:,6]
            @test abs(s.Q[:, m]' * X[:, j]) / norm(X[:, j]) ≈ 1 atol = 1e-6
        end
    end

    @testset "double root keeps multiplicity two" begin
        @test sum(k -> abs(real(k) - 3.0) < 1e-6, ks) == 2
        idx = findall(k -> abs(real(k) - 3.0) < 1e-6, ks)
        Xd = X[:, idx] ./ norm.(eachcol(X[:, idx]))'
        @test svdvals(Xd)[2] > 1e-3                                                 # two linearly independent eigenvectors
        P = s.Q[:, 4:5] * s.Q[:, 4:5]'                                              # projector on span(Q[:,4], Q[:,5])
        @test norm(Xd - P * Xd) < 1e-6                                              # both lie in the eigenspace of kappa = 3
    end

    @testset "solve returns the first two outputs of solve_vectors" begin
        ks2, res2 = solve(solver, BN_PTS, BN_K0, BN_DK; multithreaded=false)
        @test ks2 isa Vector{ComplexF64}
        @test sort(real.(ks2)) ≈ real.(ks) atol = 1e-10
        @test sort(res2) ≈ sort(res) atol = 1e-12
    end

    @testset "empty window" begin
        ks0, res0, X0 = solve_vectors(solver, BN_PTS, 5.2, 0.4; multithreaded=false)
        @test isempty(ks0) && isempty(res0)
        @test size(X0) == (BN_N, 0)
    end
end

@testset "_beyn_solve_core: res_tol and auto_discard_spurious" begin
    @testset "res_tol = 0 rejects every candidate" begin
        ks, res, X = _beyn_solve_core(bn_solver(use_taylor=false, res_tol=0.0), BN_PTS, BN_K0, BN_DK; multithreaded=false)
        @test isempty(ks) && isempty(res)
        @test size(X) == (BN_N, 0)
    end

    @testset "auto_discard_spurious = false skips the residual check" begin
        ks, res, X = _beyn_solve_core(bn_solver(use_taylor=false, res_tol=0.0, auto_discard_spurious=false), BN_PTS, BN_K0, BN_DK; multithreaded=false)
        @test length(ks) == 4                                                       # same candidates as the projected solve
        @test all(isnan, res)
        @test size(X) == (BN_N, 4)
    end

    @testset "coarse quadrature: kept candidates are exactly those below res_tol" begin
        # nq = 24 trapezoid nodes leak the outside roots 2.2 and 3.8 into the moments at the 1e-5 level (svd_tol = 1e-3 hides
        # that rank), so the four projected roots are slightly inaccurate and have different residuals
        coarse(; kwargs...) = bn_solver(; use_taylor=false, nq=24, svd_tol=1e-3, kwargs...)
        ks_all, X_all = _beyn_projected_solve(coarse(), BN_PTS, BN_K0, BN_DK; multithreaded=false)
        s = bn_problem()
        rall = [norm(bn_matrix(s, ks_all[j]) * X_all[:, j]) for j in eachindex(ks_all)]
        @test length(ks_all) == 4
        @test minimum(rall) < maximum(rall)
        tol = sqrt(minimum(rall) * maximum(rall))                                   # separates the best from the worst candidate
        ks, res, X = _beyn_solve_core(coarse(res_tol=tol), BN_PTS, BN_K0, BN_DK; multithreaded=false)
        @test 0 < length(ks) < length(ks_all)
        @test length(ks) == count(<(tol), rall)
        @test all(<(tol), res)
        @test size(X, 2) == length(ks)
        @test all(k -> any(==(k), ks_all), ks)                                      # discarding never alters the surviving roots
    end
end

# Candidate bookkeeping for _beyn_imag_k_check. A candidate is (k, m): the root estimate k paired with the vector Q[:,m], whose
# residual is exactly |d_m(k)| since Q[:,m] is a unit eigenvector of the matrix A(k) with eigenvalue d_m(k).
function bn_candidates(s::bn_Problem, cands)
    ks = ComplexF64[k for (k, _) in cands]
    X = hcat([s.Q[:, m] for (_, m) in cands]...)
    return ks, ComplexF64.(X)
end

@testset "_beyn_imag_k_check: global imaginary-k screening" begin
    s = bn_problem()
    # window 1: two good roots (imag 1e-5, 2e-5) and one fake far from the real axis (imag 0.3, residual about 0.9)
    # window 2: one good root (imag 3e-5) and one wrong root with the smallest imaginary part (1e-7, residual about 1.9)
    ks1, X1 = bn_candidates(s, [(2.7 + 1e-5im, 3), (3.0 + 2e-5im, 4), (2.7 + 0.3im, 3)])
    ks2, X2 = bn_candidates(s, [(3.3 + 3e-5im, 6), (3.8 + 1e-7im, 6)])
    ks_all, X_all, pts_all = [ks1, ks2], [X1, X2], [BN_PTS, BN_PTS]
    # processing order by decreasing |imag k|: fake (w1,3), good (w2,1), good (w1,2), good (w1,1), wrong (w2,2)

    @testset "large pad checks every candidate and drops the failing ones" begin
        solver = bn_solver(use_taylor=false, res_tol=1e-2)
        idx, res = _beyn_imag_k_check(solver, ks_all, X_all, pts_all; pad=10, group_size=1, multithreaded=false)
        @test idx == [[1, 2], [1]]
        @test all(isfinite, vcat(res...))
        @test all(<(1e-2), vcat(res...))
        @test res[1][1] ≈ abs(bn_d(s, 3, ks1[1])) atol = 1e-12
    end

    @testset "pad consecutive good roots end the check; later roots are kept unchecked" begin
        solver = bn_solver(use_taylor=false, res_tol=1e-2)
        idx, res = _beyn_imag_k_check(solver, ks_all, X_all, pts_all; pad=2, group_size=1, multithreaded=false)
        @test idx == [[1, 2], [1, 2]]                                       # the wrong root (w2,2) escapes the check
        @test isnan(res[1][1]) && isfinite(res[1][2])                      # (w1,1) unchecked, (w1,2) checked
        @test isfinite(res[2][1]) && isnan(res[2][2])                      # (w2,1) checked, (w2,2) unchecked
    end

    @testset "a failing root resets the good-root streak" begin
        solver = bn_solver(use_taylor=false, res_tol=1e-2)
        # pad = 3: the fake fails first, so only three good roots in a row (w2,1 w1,2 w1,1) end the check before the wrong root
        idx, res = _beyn_imag_k_check(solver, ks_all, X_all, pts_all; pad=3, group_size=1, multithreaded=false)
        @test idx == [[1, 2], [1, 2]]
        @test isnan(res[2][2])
        # pad = 4 cannot be reached before the wrong root, which is then checked and dropped
        idx, _ = _beyn_imag_k_check(solver, ks_all, X_all, pts_all; pad=4, group_size=1, multithreaded=false)
        @test idx == [[1, 2], [1]]
    end

    @testset "auto_discard_spurious = false only resets the streak" begin
        solver = bn_solver(use_taylor=false, res_tol=1e-2, auto_discard_spurious=false)
        idx, res = _beyn_imag_k_check(solver, ks_all, X_all, pts_all; pad=10, group_size=1, multithreaded=false)
        @test idx == [[1, 2, 3], [1, 2]]
        @test all(isfinite, vcat(res...))
        @test res[1][3] > 1e-2 && res[2][2] > 1e-2                          # failing residuals are reported, not discarded
    end

    @testset "pad and group_size default to the solver fields" begin
        solver = bn_solver(use_taylor=false, res_tol=1e-2, imag_k_pad=10, imag_k_group_size=1)
        idx, _ = _beyn_imag_k_check(solver, ks_all, X_all, pts_all; multithreaded=false)
        @test idx == [[1, 2], [1]]
    end

    @testset "no candidates" begin
        solver = bn_solver(use_taylor=false)
        empty_X = zeros(ComplexF64, BN_N, 0)
        idx, res = _beyn_imag_k_check(solver, [ComplexF64[], ComplexF64[]], [empty_X, empty_X], pts_all; multithreaded=false)
        @test idx == [Int[], Int[]]
        @test res == [Float64[], Float64[]]
    end
end

# Degree-2 Taylor matrix of the synthetic problem about k0 at a real k, to check the single-sample error of validate_beyn_taylor.
function bn_taylor_matrix_p2(s::bn_Problem, k0, k)
    a = bn_taylor_coeffs(s, k0)
    δ = k - k0
    return ComplexF64.(s.Q * Diagonal([a[1][m] + a[2][m] * δ + a[3][m] * δ^2 for m in eachindex(s.kappa)]) * s.Q')
end

@testset "validate_beyn_taylor" begin
    s = bn_problem()

    @testset "exact Taylor degree gives round-off error" begin
        @test validate_beyn_taylor(s, BN_PTS, 3.0, 0.5, 3) < 1e-13
        @test validate_beyn_taylor(s, BN_PTS, 3.0, 0.5, 6) < 1e-13
    end

    @testset "truncated degree gives a visible, monotonically smaller error" begin
        e1 = validate_beyn_taylor(s, BN_PTS, 3.0, 0.5, 1)
        e2 = validate_beyn_taylor(s, BN_PTS, 3.0, 0.5, 2)
        @test e1 > 1e-3
        @test e2 > 1e-3
        @test e2 < e1
    end

    @testset "error grows with the contour radius for a truncated degree" begin
        @test validate_beyn_taylor(s, BN_PTS, 3.0, 0.2, 2) < validate_beyn_taylor(s, BN_PTS, 3.0, 0.8, 2)
    end

    @testset "return type and nsample" begin
        @test validate_beyn_taylor(s, BN_PTS, 3.0, 0.5, 3; nsample=3, multithreaded=false) isa Float64
        @test validate_beyn_taylor(s, BN_PTS, 3.0, 0.5, 2; nsample=1) ≈ norm(bn_matrix(s, 3.5) - bn_taylor_matrix_p2(s, 3.0, 3.5)) / norm(bn_matrix(s, 3.5))
    end
end

@testset "_tune_beyn_taylor_degree!" begin
    @testset "too high a starting degree is lowered to the smallest accurate one" begin
        solver = bn_solver(taylor_degree=9, taylor_tol=1e-12)
        p = _tune_beyn_taylor_degree!(solver, [BN_PTS], [3.0 + 0im], [0.5]; multithreaded=false)
        @test p == 3                                                         # the cubic is exact from degree 3 on, degree 2 is not
        @test solver.taylor_degree == 3
    end

    @testset "too low a starting degree is raised" begin
        solver = bn_solver(taylor_degree=1, taylor_tol=1e-12)
        @test _tune_beyn_taylor_degree!(solver, [BN_PTS], [3.0 + 0im], [0.5]; multithreaded=false) == 3
        @test solver.taylor_degree == 3
    end

    @testset "a start at the accurate degree is kept" begin
        solver = bn_solver(taylor_degree=3, taylor_tol=1e-12)
        @test _tune_beyn_taylor_degree!(solver, [BN_PTS], [3.0 + 0im], [0.5]; multithreaded=false) == 3
    end

    @testset "the window with the largest radius is calibrated" begin
        # equal Taylor accuracy is reached at degree 3 on every window; the call must not fail with several windows
        solver = bn_solver(taylor_degree=5, taylor_tol=1e-12)
        p = _tune_beyn_taylor_degree!(solver, [BN_PTS, BN_PTS], [3.0 + 0im, 4.0 + 0im], [0.2, 0.5]; multithreaded=false)
        @test p == 3
    end

    @testset "a loose tolerance accepts a lower degree" begin
        solver = bn_solver(taylor_degree=6, taylor_tol=1e-1)
        p = _tune_beyn_taylor_degree!(solver, [BN_PTS], [3.0 + 0im], [0.5]; multithreaded=false)
        @test 2 <= p < 3
        @test p == solver.taylor_degree
    end
end

@testset "solve_wavenumber / solve_spectrum on the synthetic kernel" begin
    billiard = bn_circle                                                     # only handed through to evaluate_points
    solver = bn_solver(use_taylor=false)

    @testset "solve_wavenumber returns the root closest to k" begin
        k, res = solve_wavenumber(solver, billiard, 3.05, 1.0; multithreaded=false)
        @test real(k) ≈ 3.0 atol = 1e-6
        @test res < solver.res_tol
        k, _ = solve_wavenumber(solver, billiard, 3.25, 0.4; multithreaded=false)   # radius 0.2: only 3.3 (and not 3.0) inside
        @test real(k) ≈ 3.3 atol = 1e-6
    end

    @testset "solve_wavenumber with no root in the window returns (nothing, nothing)" begin
        @test solve_wavenumber(solver, billiard, 5.2, 0.4; multithreaded=false) === (nothing, nothing)
    end

    @testset "solve_spectrum returns every validated root" begin
        ks, res = solve_spectrum(solver, billiard, BN_K0, BN_DK; multithreaded=false)
        @test sort(real.(ks)) ≈ BN_INSIDE atol = 1e-6
        @test length(res) == 4 && all(<(solver.res_tol), res)
    end
end
