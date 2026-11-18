# Unit tests for
#   src/solvers/acceleratedmethods/cork.jl   (taylor_to_cheb!, build_cork_polynomial / build_B_full, CORKPolynomial,
#                                             evaluate_cork_polynomial, validate_polynomial, get_A0, cache_block!,
#                                             compact_project!, compact_block_qr!, pack! / unpack!, init_cork, compute_tail! /
#                                             promote_tail! / extend!, ritz_roots, reconstruct_cork_eigenvectors, adaptive_cork,
#                                             _cork_solve_core via solve / solve_vectors, solve_wavenumber, solve_spectrum, cork_left_vectors, cork_left_residuals,
#                                             _cork_left_vectors, side = :left,
#                                             _cork_warn_if_incomplete)
# The symmetry-reduced polynomial (build_B_reduced) needs real symmetry orbits and is left to integration coverage; constructor
# validation is covered in solvers/solverconstructors_tests.jl.
#
# Synthetic nonlinear eigenproblem. A test-local kernel `ck_Problem <: SweepBIMSolver` implements the extension interface of the
# sweep BIM branch (`construct_matrices`, `boundary_matrix_size`, `_bim_numeric_type`, `evaluate_points`) plus the analytic Taylor
# hooks `_taylor_cache` / `_fredholm_taylor!`, so CORKSolver runs unchanged and no billiard is solved. The Fredholm-like matrix is
#
#       A(k) = Q diag(d_m(k)) Q',    d_m(k) = (k - kappa_m)(1 + c k^2),
#
# with a fixed orthogonal Q (dense, so the roots are not visible in the matrix entries). Its eigenvalues are exactly
#   * the real roots kappa_m (kappa_12 = kappa_13 is a semisimple double root, eigenvectors Q[:,12], Q[:,13]),
#   * the extra roots k = +-i/sqrt(c), far from every window used below.
# Since d_m is a cubic, the degree-3 Chebyshev polynomial on any interval reproduces A(k) exactly (up to round-off), and the
# coefficient of T_3 is c*Delta^3/4 times the identity; every higher coefficient vanishes. The kernel only receives a `BoundaryPoints`
# token (one tiny circle discretization, never used numerically).
#
# Window used by most tests: center 3.1, dk = 0.6 (requested interval [2.8, 3.4], guarded polynomial half-width 0.315). It holds
# kappa = 2.85, 3.0 (double), 3.3. The nearest excluded real roots are 1.8 and 4.4; the remaining roots are spread over 0 .. 6.2 so the
# Krylov iteration converges within the matrix size N = 24 (a convergence tolerance of 1e-6 is needed for the block size 2).
# Helper definitions use the prefix `ck_`.
#
# Convergence criterion. `ritz_roots` returns a third output `diskρ`, the residual indicators of every Ritz value with |t| <= 1 + edge_tol
# (no |Im k| restriction); `adaptive_cork` accepts a Krylov dimension only when every element of `diskρ` is <= res_tol and the accepted
# set is stable. An empty accepted set is a valid result once the disk has converged, so the Weyl-count keyword `expected` no longer exists.
# The two former Weyl-guard testsets of `solve / solve_vectors` (growth of a too small `mstart` with `expected = 4.0`; empty window with
# `expected < 0.5` versus `expected = 3.0`) pinned behaviour that was removed on purpose. They are REPLACED, not weakened, by: an empty window
# returns an empty spectrum without any estimate; a too small `mstart` still yields all four roots; an unconverged Ritz value inside the
# disk blocks acceptance (`adaptive_cork`, `ritz_roots`); the `ritz_roots` residual output; `res_tol = 0` exhausts the Krylov dimension.
# With block size 1 the second copy of the double root is round-off dependent, so the two testsets that use it compare distinct roots.

using LinearAlgebra: Diagonal, qr, norm, svdvals, lu, I, tril
using Random: MersenneTwister
using QuantumBilliards: construct_matrices, _bim_numeric_type, TaylorWorkspace, power_to_cheb, taylor_to_cheb!, CORKPolynomial,
                        build_cork_polynomial, evaluate_cork_polynomial, validate_polynomial, cork_validation_references, get_A0, cache_block!,
                        compact_project!, compact_block_qr!, pack!, unpack!, init_cork, compute_tail!, promote_tail!, extend!,
                        ritz_roots, reconstruct_cork_eigenvectors, adaptive_cork, _cork_solve_core, _cork_solve, _cork_kernel_name,
                        _cork_warn_if_incomplete, cork_krylov_dim, CORK_STEPS_PER_DEGREE

# Roots, polynomial strength and orthogonal Q of the synthetic problem; the cache is the token returned by `_taylor_cache`.
struct ck_Problem <: QuantumBilliards.SweepBIMSolver
    kappa::Vector{Float64}
    c::Float64
    Q::Matrix{Float64}
    symmetry::Nothing
end
struct ck_Cache
    N::Int
    solver::ck_Problem
end
const CK_KAPPA = vcat(collect(0.0:0.2:1.8), [2.85, 3.0, 3.0, 3.3], collect(4.4:0.2:6.2))
const CK_N = length(CK_KAPPA)
const CK_C = 0.02
ck_problem() = ck_Problem(CK_KAPPA, CK_C, Matrix(qr(randn(MersenneTwister(11), CK_N, CK_N)).Q), nothing)

ck_circle = CircleBilliard(1.0)
const CK_PTS = evaluate_points(DoubleLayerPotentialSolver(2.0, ck_circle; min_pts=16), ck_circle, 2.0)

# Kernel interface of the sweep BIM branch.
ck_d(s::ck_Problem, m::Int, k) = (k - s.kappa[m]) * (1 + s.c * k^2)
ck_matrix(s::ck_Problem, k) = ComplexF64.(s.Q * Diagonal([ck_d(s, m, k) for m in eachindex(s.kappa)]) * s.Q')
QuantumBilliards._bim_numeric_type(::ck_Problem) = Float64
QuantumBilliards.boundary_matrix_size(s::ck_Problem, ::BoundaryPoints) = length(s.kappa)
QuantumBilliards.construct_matrices(s::ck_Problem, ::BoundaryPoints, k; multithreaded::Bool=true) = ck_matrix(s, k)
QuantumBilliards.evaluate_points(::ck_Problem, ::AbsBilliard, k) = CK_PTS

# Analytic Taylor hook of the Fredholm matrix entry A_ij about k: normalized coefficients a[l+1] = A_ij^(l)(k)/l!, exact for l <= 3
# and zero beyond (d_m is a cubic: c k^3 - c kappa k^2 + k - kappa).
QuantumBilliards._taylor_cache(s::ck_Problem, pts) = ck_Cache(length(s.kappa), s)
function QuantumBilliards._fredholm_taylor!(cache::ck_Cache, w::TaylorWorkspace, k::Float64, p::Int, i::Int, j::Int)
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

# Window, guarded polynomial half-width and expected roots.
const CK_K0, CK_DK, CK_GUARD = 3.1, 0.6, 0.05
const CK_DELTA = CK_DK / 2
const CK_DPOLY = CK_DELTA * (1 + CK_GUARD)
const CK_INSIDE = [2.85, 3.0, 3.0, 3.3]
const CK_INSIDE_IDX = 11:14                                                  # columns of Q belonging to CK_INSIDE
const CK_DOUBLE_IDX = 12:13

# Block size 2 resolves the double root; the loose residual tolerance lets the Krylov iteration stop below the matrix size.
ck_solver(; kwargs...) = CORKSolver(ck_problem(); merge((p=3, block_size=2, mstart=12, mstep=2, res_tol=1e-6), NamedTuple(kwargs))...)

# Degree-p Chebyshev polynomial of the synthetic problem about k0 with half-width Δ, and the LU factorization of P(0).
ck_polynomial(p::Int=3; k0=CK_K0, Δ=CK_DPOLY) = build_cork_polynomial(ck_problem(), CK_PTS, k0, Δ, p; multithreaded=false)[1]
ck_factor(P::CORKPolynomial) = lu(get_A0(P.B, P.N, P.p))

# CORK state extended to the compact Krylov dimension m (a normalized tail is pending).
function ck_state(m::Int; b::Int=2, p::Int=3)
    P = ck_polynomial(p); F = ck_factor(P)
    S = init_cork(P.B, p, P.N; block_size=b)
    extend!(S, P.B, F, m)
    return P, F, S
end

# Compact basis G[1:r,:,1:n] packed to an (r p) x n matrix, so that orthonormality is the Euclidean one.
function ck_packed(G::Array{ComplexF64,3}, r::Int, n::Int)
    Gf = zeros(ComplexF64, r * size(G, 2), n)
    pack!(Gf, G, r, size(G, 2), n)
    return Gf
end

# Chebyshev polynomial T_j(t) = cos(j acos t) on [-1, 1].
ck_T(j::Int, t::Real) = cos(j * acos(t))

# Roots sorted by real part with values closer than `tol` collapsed to the first one (a multiple root counted once).
function ck_distinct(ks::AbstractVector{<:Complex}; tol::Float64=1e-6)
    out = ComplexF64[]
    for k in sort(ks; by=real)
        (isempty(out) || abs(k - out[end]) > tol) && push!(out, k)
    end
    return out
end

# Whether the multiset `a` is contained in the multiset `b` (exact equality of the entries).
function ck_submultiset(a::AbstractVector, b::AbstractVector)
    rest = collect(b)
    for x in a
        i = findfirst(==(x), rest)
        i === nothing && return false
        deleteat!(rest, i)
    end
    return true
end

@testset "synthetic problem: roots, eigenvectors and Taylor hook" begin
    s = ck_problem()
    @test all(m -> norm(ck_matrix(s, s.kappa[m]) * s.Q[:, m]) < 1e-13, 1:CK_N)
    @test norm(ck_matrix(s, 2.0) * s.Q[:, CK_INSIDE_IDX[1]]) > 0.1           # away from a root the residual is not small
    @test all(m -> isapprox(abs(ck_d(s, m, im / sqrt(s.c))), 0; atol=1e-10), 1:CK_N)   # the extra roots +-i/sqrt(c)
    cache = QuantumBilliards._taylor_cache(s, CK_PTS)
    w = TaylorWorkspace(5)
    δ = 0.37 - 0.2im
    for (i, j) in ((1, 1), (3, 7), (24, 5))
        a = QuantumBilliards._fredholm_taylor!(cache, w, 3.2, 5, i, j)
        @test sum(a[l+1] * δ^l for l in 0:5) ≈ ck_matrix(s, 3.2 + δ)[i, j]
        @test a[5] == 0 && a[6] == 0                                         # the cubic stops at degree 3
    end
end

@testset "taylor_to_cheb!: Taylor to Chebyshev coefficients" begin
    Δ = 0.4
    C = power_to_cheb(3, Δ)

    @testset "monomials in δ = Δt" begin
        β = fill(9.9 + 0im, 4)
        @test taylor_to_cheb!(β, ComplexF64[1, 0, 0, 0], C) === β            # in place, returns its first argument
        @test β ≈ ComplexF64[1, 0, 0, 0]                                     # 1 = T0
        taylor_to_cheb!(β, ComplexF64[0, 1, 0, 0], C)
        @test β ≈ ComplexF64[0, Δ, 0, 0]                                     # δ = Δ T1
        taylor_to_cheb!(β, ComplexF64[0, 0, 1, 0], C)
        @test β ≈ ComplexF64[Δ^2 / 2, 0, Δ^2 / 2, 0]                         # δ² = Δ²(T0 + T2)/2
        taylor_to_cheb!(β, ComplexF64[0, 0, 0, 1], C)
        @test β ≈ ComplexF64[0, 3Δ^3 / 4, 0, Δ^3 / 4]                        # δ³ = Δ³(3T1 + T3)/4
    end

    @testset "a general polynomial is reproduced at an interior point" begin
        a = ComplexF64[0.3 + 0.1im, -1.2, 0.5im, 2.0 - 1.0im]
        β = taylor_to_cheb!(zeros(ComplexF64, 4), a, C)
        @test β ≈ C * a
        t = 0.7
        @test sum(β[j+1] * ck_T(j, t) for j in 0:3) ≈ sum(a[l+1] * (Δ * t)^l for l in 0:3)
    end
end

@testset "build_cork_polynomial / CORKPolynomial / evaluate_cork_polynomial" begin
    s = ck_problem()
    P, tbuild = build_cork_polynomial(s, CK_PTS, CK_K0, CK_DPOLY, 3; multithreaded=false)

    @testset "fields and shapes" begin
        @test P isa CORKPolynomial
        @test (P.k0, P.Δ, P.p, P.N) == (CK_K0, CK_DPOLY, 3, CK_N)
        @test size(P.B) == (4 * CK_N, CK_N)
        @test tbuild isa Float64 && tbuild >= 0
    end

    @testset "exact degree reproduces the matrix and has the analytic leading coefficient" begin
        for t in (-1.0, -0.3, 0.0, 0.8, 1.0, 1.7)                            # also outside [-1, 1]: the cubic is exact everywhere
            @test evaluate_cork_polynomial(P, t) ≈ ck_matrix(s, CK_K0 + CK_DPOLY * t) atol = 1e-12
        end
        @test P.B[3CK_N+1:4CK_N, :] ≈ CK_C * CK_DPOLY^3 / 4 * Matrix(1.0I, CK_N, CK_N) atol = 1e-14   # coefficient of T_3
    end

    @testset "degrees above the exact one carry vanishing coefficients" begin
        P6 = ck_polynomial(6)
        @test size(P6.B) == (7 * CK_N, CK_N)
        @test norm(P6.B[4CK_N+1:end, :]) < 1e-13
        @test P6.B[1:4CK_N, :] ≈ P.B atol = 1e-13
    end

    @testset "a truncated degree is visibly inexact" begin
        P2 = ck_polynomial(2)
        @test norm(evaluate_cork_polynomial(P2, 1.0) - ck_matrix(s, CK_K0 + CK_DPOLY)) > 1e-5
    end

    @testset "multithreaded and serial assembly agree" begin
        Pt = build_cork_polynomial(s, CK_PTS, CK_K0, CK_DPOLY, 3; multithreaded=true)[1]
        @test Pt.B ≈ P.B atol = 1e-14
    end

    @testset "p = 0 keeps only A(k0) and the result is a copy" begin
        P0 = ck_polynomial(0)
        @test size(P0.B) == (CK_N, CK_N)
        A = evaluate_cork_polynomial(P0, 0.9)
        @test A ≈ ck_matrix(s, CK_K0) atol = 1e-12
        A[1, 1] += 1
        @test evaluate_cork_polynomial(P0, 0.9)[1, 1] ≈ ck_matrix(s, CK_K0)[1, 1] atol = 1e-12
    end

    @testset "evaluation by the three-term recurrence on hand-made coefficients" begin
        B = ComplexF64[1 0; 0 1;  2 0; 0 2;  3 0; 0 3]                       # B0 = I, B1 = 2I, B2 = 3I
        Ph = CORKPolynomial(B, 0.0, 1.0, 2, 2)
        t = 0.3
        @test evaluate_cork_polynomial(Ph, t) ≈ (1 + 2t + 3 * (2t^2 - 1)) * Matrix(1.0I, 2, 2)
        P1 = CORKPolynomial(B[1:4, :], 0.0, 1.0, 1, 2)
        @test evaluate_cork_polynomial(P1, -0.6) ≈ (1 - 1.2) * Matrix(1.0I, 2, 2)
    end
end

@testset "validate_polynomial" begin
    s = ck_problem()

    @testset "exact degree gives round-off error" begin
        @test validate_polynomial(s, CK_PTS, ck_polynomial(3); multithreaded=false) < 1e-13
        @test validate_polynomial(s, CK_PTS, ck_polynomial(6); multithreaded=false) < 1e-13
    end

    @testset "truncated degree gives a visible error that is the worst over the samples" begin
        P2 = ck_polynomial(2)
        err = validate_polynomial(s, CK_PTS, P2; nsample=3, multithreaded=false)
        @test err isa Float64
        @test err > 1e-6
        errs = [norm(evaluate_cork_polynomial(P2, t) - ck_matrix(s, CK_K0 + CK_DPOLY * t)) / norm(ck_matrix(s, CK_K0 + CK_DPOLY * t)) for t in (-1.0, 0.0, 1.0)]
        @test err ≈ maximum(errs)
        @test validate_polynomial(s, CK_PTS, P2; nsample=2, multithreaded=false) ≈ max(errs[1], errs[3]) atol = 1e-12   # samples t = -1, 1
    end
end

@testset "cork_validation_references and the references keyword of validate_polynomial" begin
    @testset "synthetic kernel (no reusable geometry)" begin
        s = ck_problem()
        refs = cork_validation_references(s, CK_PTS, CK_K0, CK_DPOLY; nsample=5, multithreaded=false)
        @test length(refs) == 5
        for (r, t) in zip(refs, range(-1.0, 1.0, length=5))
            @test r == ck_matrix(s, CK_K0 + CK_DPOLY * t)
        end
        @test length(cork_validation_references(s, CK_PTS, CK_K0, CK_DPOLY; multithreaded=false)) == 5   # default nsample = 5
        @test length(cork_validation_references(s, CK_PTS, CK_K0, CK_DPOLY; nsample=3, multithreaded=false)) == 3

        P2 = ck_polynomial(2)
        for ns in (2, 3, 5)
            refs = cork_validation_references(s, CK_PTS, P2.k0, P2.Δ; nsample=ns, multithreaded=false)
            @test validate_polynomial(s, CK_PTS, P2; nsample=ns, multithreaded=false, references=refs) ==
                  validate_polynomial(s, CK_PTS, P2; nsample=ns, multithreaded=false)
        end
    end

    @testset "DLP on a circle: references equal construct_matrices at k0 + Δ t" begin
        circle = CircleBilliard(1.0)
        solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=24)
        k0, Δ, nsample = 3.0, 0.1, 4
        pts = evaluate_points(solver, circle, k0 + Δ)
        refs = cork_validation_references(solver, pts, k0, Δ; nsample, multithreaded=false)
        @test length(refs) == nsample
        for (r, t) in zip(refs, range(-1.0, 1.0, length=nsample))
            @test r == construct_matrices(solver, pts, k0 + Δ * t; multithreaded=false)
            @test size(r) == (boundary_matrix_size(solver, pts), boundary_matrix_size(solver, pts))
        end

        P, _ = build_cork_polynomial(solver, pts, k0, Δ, 8; multithreaded=false)
        e_direct = validate_polynomial(solver, pts, P; nsample, multithreaded=false)
        e_refs = validate_polynomial(solver, pts, P; nsample, multithreaded=false, references=refs)
        @test e_refs == e_direct
        @test isfinite(e_refs)
    end
end

@testset "get_A0: shift matrix P(0)" begin
    @testset "even and odd degrees on hand-made coefficients" begin
        B = ComplexF64[1 0; 0 1;  5 5; 5 5;  2 0; 0 3;  7 7; 7 7;  4 0; 0 1]   # B0 = I, B1, B2 = diag(2,3), B3, B4 = diag(4,1)
        @test get_A0(B, 2, 4) ≈ ComplexF64[1-2+4 0; 0 1-3+1]                    # B0 - B2 + B4
        @test get_A0(B[1:8, :], 2, 3) ≈ ComplexF64[1-2 0; 0 1-3]                # odd coefficients never contribute
        @test get_A0(B[1:2, :], 2, 0) ≈ Matrix(1.0I, 2, 2)
    end

    @testset "equals the polynomial at t = 0 and the matrix at the center" begin
        P = ck_polynomial(3)
        @test get_A0(P.B, P.N, 3) ≈ evaluate_cork_polynomial(P, 0.0)
        @test get_A0(P.B, P.N, 3) ≈ ck_matrix(ck_problem(), CK_K0) atol = 1e-12
        P4 = ck_polynomial(4)
        @test get_A0(P4.B, P4.N, 4) ≈ evaluate_cork_polynomial(P4, 0.0)
    end
end

@testset "cache_block!: coefficient actions" begin
    N, p, rmax = 6, 2, 6
    rng = MersenneTwister(3)
    B = randn(rng, ComplexF64, (p + 1) * N, N)
    U = randn(rng, ComplexF64, N, rmax)

    @testset "fills exactly the requested columns" begin
        W = zeros(ComplexF64, (p + 1) * N, rmax)
        @test cache_block!(W, B, U, N, p, 2, 4) === nothing
        for j in 0:p
            rows = j * N + 1:(j + 1) * N
            @test W[rows, 2:4] ≈ B[rows, :] * U[:, 2:4]
        end
        @test iszero(W[:, 1]) && iszero(W[:, 5:end])
    end

    @testset "an empty column range does nothing" begin
        W = zeros(ComplexF64, (p + 1) * N, rmax)
        @test cache_block!(W, B, U, N, p, 4, 3) === nothing
        @test iszero(W)
    end
end

@testset "pack! / unpack!" begin
    r, p, b = 4, 3, 2
    Z = randn(MersenneTwister(5), ComplexF64, r, p, b)
    Zf = zeros(ComplexF64, r * p, b)
    pack!(Zf, Z, r, p, b)
    @test Zf[(2 - 1) * r + 3, 2] == Z[3, 2, 2]                               # row (j-1) r + i holds Z[i,j,c]
    @test Zf[1:r, 1] == Z[:, 1, 1]
    Z2 = zeros(ComplexF64, r, p, b)
    unpack!(Z2, Zf, r, p, b)
    @test Z2 == Z                                                           # round trip
end

@testset "compact_block_qr!: thin Householder QR and numerical rank" begin
    rng = MersenneTwister(7)

    @testset "full rank block" begin
        Z0 = randn(rng, ComplexF64, 12, 3)
        Z = copy(Z0)
        rank, R = compact_block_qr!(Z, 3)
        @test rank == 3
        @test size(R) == (3, 3) && iszero(tril(R, -1))                       # upper triangular
        @test Z' * Z ≈ Matrix(1.0I, 3, 3) atol = 1e-13                       # Z now holds the orthonormal factor
        @test Z * R ≈ Z0 atol = 1e-13
    end

    @testset "linearly dependent column lowers the rank" begin
        Z = randn(rng, ComplexF64, 12, 3)
        Z[:, 3] = Z[:, 1] + 2 * Z[:, 2]
        rank, _ = compact_block_qr!(Z, 3)
        @test rank == 2
    end
end

@testset "compact_project!: two-pass Gram-Schmidt against the compact basis" begin
    r, p, n, b = 5, 3, 4, 2
    rng = MersenneTwister(9)
    Gf = Matrix(qr(randn(rng, ComplexF64, r * p, n)).Q)                      # orthonormal packed basis
    G = zeros(ComplexF64, r, p, n + b)
    unpack!(G, Gf, r, p, n)
    Z = randn(rng, ComplexF64, r, p, b); Z0 = copy(Z)
    H1 = zeros(ComplexF64, n, b); H2 = zeros(ComplexF64, n, b)
    @test compact_project!(Z, G, r, n, p, H1, H2) === nothing
    Zf = zeros(ComplexF64, r * p, b); pack!(Zf, Z, r, p, b)
    Z0f = zeros(ComplexF64, r * p, b); pack!(Z0f, Z0, r, p, b)
    @test norm(Gf' * Zf) < 1e-13                                             # orthogonal to the basis
    @test H1 ≈ Gf' * Z0f atol = 1e-13                                        # total projection coefficients
    @test Zf + Gf * H1 ≈ Z0f atol = 1e-13                                    # Z0 = projected part + orthogonalized remainder
end

@testset "cork_krylov_dim: maximum compact Krylov dimension" begin
    @test CORK_STEPS_PER_DEGREE == 4

    @testset "large N: (N ÷ b) b" begin
        for (N, p, b, mstep) in ((1000, 12, 10, 50), (500, 6, 5, 10), (1234, 3, 7, 14))
            @test cork_krylov_dim(N, p, b, mstep) == (N ÷ b) * b
        end
    end

    @testset "small N: extended to 4 p b + mstep" begin
        for (N, p, b, mstep) in ((50, 12, 10, 50), (24, 3, 2, 2), (30, 6, 5, 10), (1, 1, 1, 1))
            @test cork_krylov_dim(N, p, b, mstep) == CORK_STEPS_PER_DEGREE * p * b + mstep
            @test cork_krylov_dim(N, p, b, mstep) > (N ÷ b) * b
        end
        @test cork_krylov_dim(24, 3, 2, 2) == 26
        @test cork_krylov_dim(50, 12, 10, 50) == 530
    end

    @testset "the crossover is at (N ÷ b) b = 4 p b + mstep" begin
        p, b, mstep = 3, 2, 2
        Nc = CORK_STEPS_PER_DEGREE * p * b + mstep                            # 26
        @test cork_krylov_dim(Nc, p, b, mstep) == Nc
        @test cork_krylov_dim(Nc - 1, p, b, mstep) == Nc                      # (25 ÷ 2) 2 = 24 < 26
        @test cork_krylov_dim(Nc + 2, p, b, mstep) == Nc + 2
    end

    @testset "divisible by the block size and monotone in N" begin
        for (p, b, mstep) in ((3, 2, 2), (12, 10, 50), (4, 4, 8), (5, 3, 6))
            dims = [cork_krylov_dim(N, p, b, mstep) for N in 1:600]
            @test all(d -> d % b == 0, dims)
            @test issorted(dims)
            @test all(d -> d >= CORK_STEPS_PER_DEGREE * p * b + mstep, dims)
            @test all(N -> dims[N] >= (N ÷ b) * b, 1:600)
        end
    end

    @testset "integer result for any integer types" begin
        @test cork_krylov_dim(Int32(100), Int16(3), UInt8(2), Int8(2)) === 100
        @test cork_krylov_dim(24, 3, 2, 2) isa Int
    end
end

@testset "init_cork: initial compact block-Arnoldi state" begin
    P = ck_polynomial(3)

    @testset "dimension checks" begin
        B = zeros(ComplexF64, 4 * 4, 4)
        @test_throws "exceeds physical matrix dimension" init_cork(zeros(ComplexF64, 6 * 4, 4), 5, 4; block_size=2)     # p > N
        @test_throws "smaller than block size" init_cork(B, 3, 4; block_size=5)                                       # N < b
        @test_throws "smaller than block size" init_cork(zeros(ComplexF64, 3 * 12, 12), 2, 12; block_size=5)           # p^2 < b
    end

    @testset "state layout" begin
        S = init_cork(P.B, 3, CK_N; block_size=2)
        @test S isa QuantumBilliards.CORKState
        @test (S.r, S.n, S.b, S.p, S.N, S.rmax) == (3, 2, 2, 3, CK_N, CK_N)
        @test !S.pending && S.napply == 0
        @test size(S.Hb) == (CK_N + 2, CK_N)                                 # mdim = floor(N / b) b
        @test size(S.G) == (CK_N, 3, CK_N + 2)
        @test size(S.W) == (4 * CK_N, CK_N)
    end

    @testset "physical basis, cached actions and starting block" begin
        S = init_cork(P.B, 3, CK_N; block_size=2)
        U = S.U[:, 1:3]
        @test U' * U ≈ Matrix(1.0I, 3, 3) atol = 1e-13
        for j in 0:3
            rows = j * CK_N + 1:(j + 1) * CK_N
            @test S.W[rows, 1:3] ≈ P.B[rows, :] * U
        end
        Gf = ck_packed(S.G, 3, 2)
        @test Gf' * Gf ≈ Matrix(1.0I, 2, 2) atol = 1e-13                     # orthonormal starting block
    end

    @testset "the start is deterministic" begin
        S1 = init_cork(P.B, 3, CK_N; block_size=2); S2 = init_cork(P.B, 3, CK_N; block_size=2)
        @test S1.U == S2.U && S1.G == S2.G
    end
end

@testset "init_cork: explicit mdim" begin
    P = ck_polynomial(3)

    @testset "Hessenberg and basis sizes follow mdim" begin
        for (b, mdim) in ((2, 40), (2, 26), (2, 12), (1, 30), (4, 28))
            S = init_cork(P.B, 3, CK_N; block_size=b, mdim=mdim)
            @test size(S.Hb) == (mdim + b, mdim)
            @test size(S.G) == (CK_N, 3, mdim + b)
            @test size(S.H1) == (mdim + b, b) && size(S.H2) == (mdim + b, b)
            @test S.b == b && S.n == b && S.N == CK_N && S.rmax == CK_N
        end
    end

    @testset "the default is (N ÷ b) b" begin
        for b in (1, 2, 5)
            S = init_cork(P.B, 3, CK_N; block_size=b)
            @test size(S.Hb, 2) == (CK_N ÷ b) * b
            @test size(init_cork(P.B, 3, CK_N; block_size=b, mdim=(CK_N ÷ b) * b).Hb) == size(S.Hb)
        end
    end

    @testset "mdim must be divisible by the block size" begin
        @test_throws "must be divisible by block size" init_cork(P.B, 3, CK_N; block_size=2, mdim=25)
        @test_throws "mdim=7 must be divisible by block size b=2" init_cork(P.B, 3, CK_N; block_size=2, mdim=7)
        @test_throws "must be divisible by block size" init_cork(P.B, 3, CK_N; block_size=4, mdim=26)
    end

    @testset "mdim above N: the state can be extended past N" begin
        mdim = CK_N + 8
        S = init_cork(P.B, 3, CK_N; block_size=2, mdim=mdim)
        F = ck_factor(P)
        extend!(S, P.B, F, CK_N)
        @test S.n == CK_N
        @test_throws "exceeds available dimension" extend!(S, P.B, F, mdim + 2)
    end
end

@testset "compute_tail! / promote_tail! / extend!: Arnoldi bookkeeping" begin
    @testset "compute_tail! produces one pending orthonormal block and is idempotent" begin
        P = ck_polynomial(3); F = ck_factor(P)
        S = init_cork(P.B, 3, CK_N; block_size=2)
        @test compute_tail!(S, P.B, F) === nothing
        @test S.pending && S.napply == 1 && S.n == 2
        @test S.r > 3                                                        # the inverse action enlarges the physical basis
        Gf = ck_packed(S.G, S.r, S.n)
        Tf = zeros(ComplexF64, S.r * 3, 2); pack!(Tf, S.pendingG, S.r, 3, 2)
        @test Tf' * Tf ≈ Matrix(1.0I, 2, 2) atol = 1e-12                     # normalized tail
        @test norm(Gf' * Tf) < 1e-12                                         # orthogonal to the active basis
        compute_tail!(S, P.B, F)
        @test S.napply == 1                                                  # a pending tail is not recomputed
    end

    @testset "promote_tail! increases the active dimension by b" begin
        P = ck_polynomial(3); F = ck_factor(P)
        S = init_cork(P.B, 3, CK_N; block_size=2)
        compute_tail!(S, P.B, F)
        promote_tail!(S)
        @test S.n == 4 && !S.pending
        @test_throws "No pending tail" promote_tail!(S)
    end

    @testset "extend!: orthonormal basis and block-Hessenberg projection" begin
        _, _, S = ck_state(8)
        @test S.n == 8 && S.pending && S.napply == 4
        Gf = ck_packed(S.G, S.r, 8)
        @test Gf' * Gf ≈ Matrix(1.0I, 8, 8) atol = 1e-11
        Tf = zeros(ComplexF64, S.r * 3, 2); pack!(Tf, S.pendingG, S.r, 3, 2)
        @test norm(Gf' * Tf) < 1e-11
        Hb = S.Hb[1:10, 1:8]
        @test all(iszero, [Hb[i, j] for j in 1:8 for i in j+3:10])           # zero below the first block sub-diagonal
        @test all(i -> abs(Hb[i, i]) > 0 || abs(Hb[i+2, i]) > 0, 1:6)       # the sub-diagonal blocks carry the QR factors
    end

    @testset "extend! argument checks" begin
        P = ck_polynomial(3); F = ck_factor(P)
        S = init_cork(P.B, 3, CK_N; block_size=2)
        @test_throws "divisible by block size" extend!(S, P.B, F, 5)
        @test_throws "exceeds available dimension" extend!(S, P.B, F, CK_N + 2)
        extend!(S, P.B, F, 6)
        @test_throws "Cannot shrink" extend!(S, P.B, F, 4)
    end

    @testset "Reached full CORK Krylov dimension" begin
        _, _, S = ck_state(CK_N)                                             # m = mdim, tail pending
        @test S.n == CK_N && S.pending
        @test_throws "Reached full CORK Krylov dimension m=$(CK_N) for matrix size N=$(CK_N)" promote_tail!(S)
    end

    @testset "the Chebyshev action needs p >= 2" begin
        P1 = ck_polynomial(1)
        S = init_cork(P1.B, 1, CK_N; block_size=1)
        @test_throws "requires p >= 2" compute_tail!(S, P1.B, ck_factor(P1))
    end
end

@testset "ritz_roots: roots of the projected inverse linearization" begin
    P, _, S = ck_state(CK_N)
    roots, V, diskρ = ritz_roots(S, CK_K0, CK_DPOLY, CK_N)
    inwin = findall(r -> 2.8 <= r[1] <= 3.4, roots)

    @testset "requested window holds the known roots with small indicators" begin
        @test length(inwin) == 4
        @test [r[1] for r in roots[inwin]] ≈ CK_INSIDE atol = 1e-6
        @test all(r -> abs(r[2]) < 1e-6, roots[inwin])
        @test all(r -> r[3] < 1e-6, roots[inwin])
    end

    @testset "ordering, vector shape and guarded interval" begin
        @test issorted(roots; by=r -> (r[1], r[2]))
        @test size(V) == (CK_N, length(roots))
        @test all(r -> abs(r[1] - CK_K0) <= CK_DPOLY * (1 + 1e-8) + 1e-12, roots)    # Re t within [-1 - edge_tol, 1 + edge_tol]
    end

    @testset "looser tolerances can only add roots" begin
        roots_wide, V_wide = ritz_roots(S, CK_K0, CK_DPOLY, CK_N; edge_tol=0.2, imag_search_tol=1e-2)
        @test length(roots_wide) >= length(roots)
        @test size(V_wide, 2) == length(roots_wide)
        @test all(r -> any(q -> q == r, roots_wide), roots)
    end

    @testset "third output: residual indicators of every Ritz value in the window disk" begin
        @test diskρ isa Vector{Float64}
        @test all(>=(0), diskρ)
        @test length(diskρ) >= count(r -> abs(r[1] - CK_K0) <= CK_DPOLY, roots)       # the real roots of the window lie in the disk
        @test ck_submultiset([r[3] for r in roots], diskρ)                         # the residuals of the strip roots are among them
        @test count(<(1e-6), diskρ) >= length(inwin)                               # converged state (m = N): window roots have small residuals
        @test all(<(1e-6), diskρ)                                                  # and nothing else is in the disk of this problem
    end

    @testset "the disk residuals do not depend on the imaginary strip" begin
        _, _, diskρ0 = ritz_roots(S, CK_K0, CK_DPOLY, CK_N; imag_search_tol=0.0)
        @test sort(diskρ0) ≈ sort(diskρ)
        _, _, diskρ_wide = ritz_roots(S, CK_K0, CK_DPOLY, CK_N; edge_tol=0.2)
        @test length(diskρ_wide) >= length(diskρ)                                  # a larger disk can only add Ritz values
        @test ck_submultiset(diskρ, diskρ_wide)
    end

    @testset "an unconverged state has an empty strip but a visible disk" begin
        _, _, S8 = ck_state(8)
        roots8, V8, diskρ8 = ritz_roots(S8, CK_K0, CK_DPOLY, 8)
        @test isempty(roots8) && size(V8, 2) == 0
        @test !isempty(diskρ8) && maximum(diskρ8) > 1e-6
    end

    @testset "state checks" begin
        @test_throws "State dimension" ritz_roots(S, CK_K0, CK_DPOLY, CK_N - 2)
        S0 = init_cork(P.B, 3, CK_N; block_size=2)
        @test_throws "Arnoldi tail missing" ritz_roots(S0, CK_K0, CK_DPOLY, 2)
    end
end

@testset "reconstruct_cork_eigenvectors: Ritz vectors to Fredholm vectors" begin
    s = ck_problem()
    _, _, S = ck_state(CK_N)
    roots, V, _ = ritz_roots(S, CK_K0, CK_DPOLY, CK_N)
    inwin = findall(r -> 2.8 <= r[1] <= 3.4, roots)
    Ψ = reconstruct_cork_eigenvectors(S, Matrix{ComplexF64}(V[:, inwin]), CK_N)
    ks = [r[1] for r in roots[inwin]]

    @testset "shape and normalization" begin
        @test size(Ψ) == (CK_N, 4)
        @test all(j -> norm(Ψ[:, j]) ≈ 1, 1:4)
    end

    @testset "eigenvector residuals" begin
        @test all(j -> norm(ck_matrix(s, ks[j]) * Ψ[:, j]) < 1e-5, 1:4)
    end

    @testset "simple roots recover the known eigenvectors" begin
        for (j, m) in ((1, 11), (4, 14))                                     # kappa = 2.85 -> Q[:,11], kappa = 3.3 -> Q[:,14]
            @test abs(s.Q[:, m]' * Ψ[:, j]) ≈ 1 atol = 1e-5
        end
    end

    @testset "double root keeps multiplicity two and stays in its eigenspace" begin
        Ψd = Ψ[:, 2:3]
        @test svdvals(Ψd)[2] > 1e-3                                          # two linearly independent eigenvectors
        P = s.Q[:, CK_DOUBLE_IDX] * s.Q[:, CK_DOUBLE_IDX]'                   # projector on span(Q[:,12], Q[:,13])
        @test norm(Ψd - P * Ψd) < 1e-5
    end

    @testset "empty input and dimension mismatch" begin
        @test size(reconstruct_cork_eigenvectors(S, zeros(ComplexF64, CK_N, 0), CK_N)) == (CK_N, 0)
        @test_throws DimensionMismatch reconstruct_cork_eigenvectors(S, zeros(ComplexF64, CK_N - 2, 1), CK_N)
    end
end

@testset "adaptive_cork: convergence loop" begin
    P = ck_polynomial(3); F = ck_factor(P)
    kw = (k0=CK_K0, Δ=CK_DELTA, Δpoly=CK_DPOLY, block_size=2, mstart=12, mstep=2, stable_checks=1, res_tol=1e-6)

    @testset "accepted roots, requested roots and edges" begin
        S, ks, Ψ, requested, allroots, edges, edge_good, m = adaptive_cork(P.B, F, 3, CK_N; kw...)
        @test S isa QuantumBilliards.CORKState && S.n == m
        @test 12 <= m <= CK_N && m % 2 == 0
        @test Ψ === nothing                                                  # eigenvectors = false
        @test length(ks) == 4
        @test [x[1] for x in ks] ≈ CK_INSIDE atol = 1e-6
        @test all(x -> abs(x[2]) <= 1e-7 && x[3] <= 1e-6, ks)                # imag_tol and res_tol defaults
        @test requested == [x for x in allroots if 2.8 <= x[1] <= 3.4]
        @test edges == (first(requested), last(requested))
        @test edge_good == (true, true)
    end

    @testset "eigenvectors = true reconstructs one vector per accepted root" begin
        _, ks, Ψ, = adaptive_cork(P.B, F, 3, CK_N; kw..., eigenvectors=true)
        @test size(Ψ) == (CK_N, length(ks))
        @test all(j -> norm(Ψ[:, j]) ≈ 1, 1:length(ks))
    end

    @testset "a window without roots converges to an empty spectrum" begin
        Pe = ck_polynomial(3; k0=4.0, Δ=0.2 * (1 + CK_GUARD))
        _, ks, Ψ, requested, _, edges, edge_good, _ = adaptive_cork(Pe.B, ck_factor(Pe), 3, CK_N; k0=4.0, Δ=0.2, Δpoly=0.2 * (1 + CK_GUARD), block_size=2, mstart=12, mstep=2, eigenvectors=true)
        @test isempty(ks) && isempty(requested)
        @test size(Ψ) == (CK_N, 0)
        @test edges == (nothing, nothing)
        @test edge_good == (false, false)
    end

    @testset "block-size divisibility of mstart and mstep" begin
        @test_throws "mstart must be divisible" adaptive_cork(P.B, F, 3, CK_N; kw..., mstart=11)
        @test_throws "mstep must be divisible" adaptive_cork(P.B, F, 3, CK_N; kw..., mstep=3)
    end

    @testset "an unreachable residual tolerance exhausts the Krylov dimension" begin
        @test_throws "Reached full CORK Krylov dimension m=$(CK_N) for matrix size N=$(CK_N) without a stable requested spectrum" adaptive_cork(P.B, F, 3, CK_N; kw..., res_tol=0.0)
        @test_throws "inside the window disk" adaptive_cork(P.B, F, 3, CK_N; kw..., res_tol=0.0)      # the message names the unconverged disk Ritz values
    end

    @testset "an unconverged Ritz value inside the disk blocks acceptance" begin
        # Replaces the Weyl guard. Block size 2 with mstart = 6: at m = 6 and m = 8 the strip |Im k| <= imag_search_tol holds no Ritz root
        # (the requested set is empty at both dimensions, so the accepted set is trivially stable), yet the disk |t| <= 1 + edge_tol holds
        # Ritz values with residuals far above res_tol (observed: 2 values, largest 0.98 at m = 6 and 0.15 at m = 8). They carry a
        # non-negligible imaginary part, so they are invisible in the strip. The iteration must not accept the empty spectrum there.
        kw6 = (; kw..., mstart=6)
        for m in (6, 8)
            _, _, Sm = ck_state(m)
            roots_m, _, diskρ_m = ritz_roots(Sm, CK_K0, CK_DPOLY, m)
            @test isempty(roots_m)
            @test !isempty(diskρ_m) && maximum(diskρ_m) > kw6.res_tol
        end
        _, ks, _, requested, _, _, edge_good, m = adaptive_cork(P.B, F, 3, CK_N; kw6...)
        @test m > 8                                                          # not accepted as an empty spectrum at m = 6, 8
        @test [x[1] for x in ks] ≈ CK_INSIDE atol = 1e-6
        @test length(requested) == 4 && edge_good == (true, true)
    end

    @testset "mdim above the default caps the Krylov dimension and is accepted" begin
        mdim = CK_N + 8                                                      # 32 > N: the compact space of dimension r p allows it
        S, ks, _, _, _, _, _, m = adaptive_cork(P.B, F, 3, CK_N; kw..., mdim=mdim)
        @test size(S.Hb) == (mdim + 2, mdim)
        @test [x[1] for x in ks] ≈ CK_INSIDE atol = 1e-6
        @test m <= CK_N                                                      # unchanged result: converged before reaching N
        @test_throws "Reached full CORK Krylov dimension m=$(mdim) for matrix size N=$(CK_N)" adaptive_cork(P.B, F, 3, CK_N; kw..., mdim=mdim, res_tol=0.0)
    end

    @testset "mdim must be divisible by the block size" begin
        @test_throws "must be divisible by block size" adaptive_cork(P.B, F, 3, CK_N; kw..., mdim=25)
    end

    @testset "a Krylov dimension counts as stable only after a converged previous evaluation" begin
        # Window [0.8, 1.0] of the synthetic problem, block size 1, mstart = 6, mstep = 1, res_tol = 1e-6: the disk holds an
        # unconverged Ritz value up to m = 11 (observed) and is converged from m = 12 on. With stable_checks = 1 the first accepted
        # dimension needs a converged disk at m and at m - mstep, so it cannot be the first converged dimension m = 12.
        k0, Δ = 0.9, 0.1
        Δpoly = Δ * (1 + CK_GUARD)
        Pw = ck_polynomial(3; k0, Δ=Δpoly); Fw = ck_factor(Pw)
        kww = (k0=k0, Δ=Δ, Δpoly=Δpoly, block_size=1, mstart=6, mstep=1, stable_checks=1, res_tol=1e-6)
        disk_max(m) = begin
            S = init_cork(Pw.B, 3, CK_N; block_size=1)
            extend!(S, Pw.B, Fw, m)
            maximum(ritz_roots(S, k0, Δpoly, m)[3]; init=0.0)
        end
        @test disk_max(11) > kww.res_tol && disk_max(12) <= kww.res_tol      # the first converged evaluation is m = 12
        _, ks, _, _, _, _, _, m = adaptive_cork(Pw.B, Fw, 3, CK_N; kww...)
        @test m > kww.mstart
        @test disk_max(m) <= kww.res_tol
        @test disk_max(m - kww.mstep) <= kww.res_tol                         # previous evaluation was converged as well
        @test m >= 13
        @test all(x -> abs(x[1] - 1.0) < 1e-6 || abs(x[1] - 0.8) < 1e-6, ks) && !isempty(ks)
    end

    @testset "the accepted dimension always follows a converged evaluation (stable_checks = 1)" begin
        for (bs, ms, mst) in ((2, 6, 2), (2, 8, 2), (1, 4, 1), (1, 8, 2))
            kwb = (; kw..., block_size=bs, mstart=ms, mstep=mst)
            _, ks, _, _, _, _, _, m = adaptive_cork(P.B, F, 3, CK_N; kwb...)
            @test m > ms
            for mm in (m, m - mst)
                S = init_cork(P.B, 3, CK_N; block_size=bs)
                extend!(S, P.B, F, mm)
                @test maximum(ritz_roots(S, CK_K0, CK_DPOLY, mm)[3]; init=0.0) <= kwb.res_tol
            end
            bs == 2 && @test [x[1] for x in ks] ≈ CK_INSIDE atol = 1e-6     # block size 1 misses the second copy of the double root
        end
    end
end

@testset "solve / solve_vectors: accepted roots and eigenvectors" begin
    s = ck_problem()
    solver = ck_solver()
    ks, res, Ψ = solve_vectors(solver, CK_PTS, CK_K0, CK_DK; multithreaded=false)

    @testset "roots, residual indicators and shapes" begin
        @test ks isa Vector{ComplexF64} && res isa Vector{Float64}
        @test length(ks) == length(res) == 4
        @test real.(ks) ≈ CK_INSIDE atol = 1e-6
        @test all(<(1e-6), abs.(imag.(ks)))
        @test all(<=(solver.res_tol), res)
        @test size(Ψ) == (CK_N, 4)
    end

    @testset "eigenvector residuals" begin
        for j in 1:4
            @test norm(ck_matrix(s, ks[j]) * Ψ[:, j]) < 1e-5
        end
    end

    @testset "double root keeps multiplicity two" begin
        @test sum(k -> abs(real(k) - 3.0) < 1e-6, ks) == 2
        @test svdvals(Ψ[:, 2:3])[2] > 1e-3
        P = s.Q[:, CK_DOUBLE_IDX] * s.Q[:, CK_DOUBLE_IDX]'
        @test norm(Ψ[:, 2:3] - P * Ψ[:, 2:3]) < 1e-5
    end

    @testset "solve returns the first two outputs of solve_vectors" begin
        ks2, res2 = solve(solver, CK_PTS, CK_K0, CK_DK; multithreaded=false)
        @test ks2 ≈ ks && res2 ≈ res
    end

    @testset "a larger polynomial degree represents the same operator" begin
        # With block size 1 a single starting vector sees only one copy of the double root in exact arithmetic; a second copy is
        # generated by round-off and, as every Ritz value inside the disk must converge, it may or may not be returned. The test therefore
        # compares the distinct roots; it still asserts that the three distinct roots are found.
        solver1 = ck_solver(block_size=1)                                             # block size 1 converges within N also for p = 5
        ks3, _ = solve(solver1, CK_PTS, CK_K0, CK_DK; multithreaded=false)
        ks5, _ = solve(solver1, CK_PTS, CK_K0, CK_DK; multithreaded=false, p=5)
        d3, d5 = ck_distinct(ks3), ck_distinct(ks5)
        @test real.(d5) ≈ [2.85, 3.0, 3.3] atol = 1e-6
        @test d5 ≈ d3 atol = 1e-6
    end

    @testset "a window without roots returns empty outputs" begin
        ks0, res0, Ψ0 = solve_vectors(solver, CK_PTS, 4.0, 0.4; multithreaded=false)
        @test isempty(ks0) && isempty(res0)
        @test size(Ψ0) == (CK_N, 0)
    end

    @testset "multiplicity is limited by the block size" begin
        # A single starting vector sees one copy of the double root; a second copy, when present, is a round-off artefact and is
        # not asserted (see ck_distinct). The three distinct roots must be found.
        ks1, _ = solve(ck_solver(block_size=1, mstart=10, res_tol=1e-10), CK_PTS, CK_K0, CK_DK; multithreaded=false)
        @test real.(ck_distinct(ks1)) ≈ [2.85, 3.0, 3.3] atol = 1e-6
    end

    # Replaces the Weyl-guard testset "a too small initial Krylov dimension does not yield an empty spectrum when levels are expected".
    @testset "a too small initial Krylov dimension still yields all roots" begin
        # mstart = 6 leaves no Ritz root in the strip at m = 6 and m = 8, but unconverged Ritz values inside the window disk keep the
        # Krylov dimension growing until the four roots appear (no `expected` level count is involved).
        ks_small, _ = _cork_solve(ck_solver(mstart=6), CK_PTS, CK_K0, CK_DK; multithreaded=false)
        @test length(ks_small) == 4
        @test real.(ks_small) ≈ [2.85, 3.0, 3.0, 3.3] atol = 1e-6
    end

    # Replaces the Weyl-guard testset "an empty window with expected < 0.5 still returns an empty spectrum" (and its expected = 3.0 throw).
    @testset "an empty window returns an empty spectrum without a Weyl estimate" begin
        ks_empty, res_empty = _cork_solve(ck_solver(), CK_PTS, 4.0, 0.4; multithreaded=false)
        @test isempty(ks_empty) && isempty(res_empty)
    end
end

@testset "_cork_solve_core: pipeline outputs and polynomial validation" begin
    @testset "returns the polynomial, the state and the convergence data" begin
        P, S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal = _cork_solve_core(ck_solver(), CK_PTS, CK_K0, CK_DK; multithreaded=false)
        @test (P.k0, P.p, P.N) == (CK_K0, 3, CK_N)
        @test P.Δ ≈ CK_DPOLY                                                 # guarded half-width (1 + guard) dk / 2
        @test S.n == mfinal
        @test length(ks) == 4 && Ψ === nothing
        @test length(requested) == 4 && length(allroots) >= 4
        @test edge_good == (true, true)
    end

    @testset "eigenvectors keyword overrides solver.states" begin
        _, _, ks, Ψ = _cork_solve_core(ck_solver(), CK_PTS, CK_K0, CK_DK; multithreaded=false, eigenvectors=true)
        @test size(Ψ) == (CK_N, length(ks))
    end

end

@testset "solve_wavenumber / solve_spectrum on the synthetic kernel" begin
    billiard = ck_circle                                                     # only handed through to evaluate_points
    solver = ck_solver()

    @testset "evaluate_points delegates to the wrapped kernel" begin
        @test evaluate_points(solver, billiard, CK_K0) === CK_PTS
    end

    @testset "solve_wavenumber returns the root closest to k" begin
        k, res = solve_wavenumber(solver, billiard, CK_K0, CK_DK; multithreaded=false)
        @test real(k) ≈ 3.0 atol = 1e-6                                      # 3.0 is 0.1 from the center, 3.3 is 0.2
        @test res <= solver.res_tol
        k, _ = solve_wavenumber(solver, billiard, 3.25, 0.3; multithreaded=false)   # [3.1, 3.4]: only 3.3 inside
        @test real(k) ≈ 3.3 atol = 1e-6
    end

    @testset "solve_wavenumber with no root in the window returns (nothing, nothing)" begin
        # no Ritz value of the disk is left unconverged, so the empty Ritz set is accepted
        @test solve_wavenumber(solver, billiard, 4.0, 0.1; multithreaded=false) === (nothing, nothing)
    end

    @testset "solve_spectrum returns every accepted root" begin
        ks, res = solve_spectrum(solver, billiard, CK_K0, CK_DK; multithreaded=false)
        @test sort(real.(ks)) ≈ CK_INSIDE atol = 1e-6
        @test length(res) == 4 && all(<=(solver.res_tol), res)
    end

    @testset "kernel name used by the debug diagnostics" begin
        @test _cork_kernel_name(DoubleLayerPotentialSolver(2.0, billiard; min_pts=16)) == "DLP"
        @test _cork_kernel_name(ck_problem()) == string(ck_Problem)
    end
end

# The synthetic kernel has no symmetry, so the Weyl count is that of the full unit circle (area/perimeter terms of `state_at_k`).
# Helper definitions use the prefix `ck_`.
ck_weyl(k1, k2) = QuantumBilliards.state_at_k(k2, ck_circle; fundamental_only=false) - QuantumBilliards.state_at_k(k1, ck_circle; fundamental_only=false)
ck_tolerance(expected) = max(0.3 * expected, 3.0)

@testset "_cork_warn_if_incomplete: Weyl-law completeness warning" begin
    solver = ck_solver()
    k1, k2 = 10.0, 20.0
    expected = ck_weyl(k1, k2)                                               # well above the absolute floor of 3

    @testset "a count equal to the Weyl count does not warn and returns nothing" begin
        @test expected > 3 / 0.3
        @test @test_logs(_cork_warn_if_incomplete(solver, ck_circle, round(Int, expected), k1, k2)) === nothing
    end

    @testset "a deficit below max(0.3 expected, 3) does not warn" begin
        nfound = ceil(Int, expected - ck_tolerance(expected))                # deficit at most the tolerance
        @test expected - nfound <= ck_tolerance(expected)
        @test_logs _cork_warn_if_incomplete(solver, ck_circle, nfound, k1, k2)
    end

    @testset "a deficit above max(0.3 expected, 3) warns" begin
        nfound = floor(Int, expected - ck_tolerance(expected)) - 1
        @test expected - nfound > ck_tolerance(expected)
        @test_logs (:warn, r"Weyl law predicts") _cork_warn_if_incomplete(solver, ck_circle, nfound, k1, k2)
    end

    @testset "no levels found in a window with many expected levels warns and names the window" begin
        @test @test_logs((:warn, r"CORK returned 0 levels in \[10\.0, 20\.0\] but the Weyl law predicts about .*probably incomplete"),
                         _cork_warn_if_incomplete(solver, ck_circle, 0, k1, k2)) === nothing
    end

    @testset "a window with fewer than 3 expected levels never warns, also for an empty result" begin
        @test ck_weyl(3.0, 3.5) < 3
        @test @test_logs(_cork_warn_if_incomplete(solver, ck_circle, 0, 3.0, 3.5)) === nothing
    end

    @testset "more levels found than expected does not warn" begin
        @test_logs _cork_warn_if_incomplete(solver, ck_circle, round(Int, 2 * expected), k1, k2)
    end
end

# Left null vectors of the polynomial: A(k) = Q diag(d(k)) Q' has the left null vector Q[:,m] at the root kappa_m (up to phase);
# the semisimple double root kappa = 3.0 has the left null space Q[:,12:13].
using QuantumBilliards: cork_left_vectors, cork_left_residuals, _cork_left_vectors

@testset "evaluate_cork_polynomial: complex argument" begin
    P = ck_polynomial(); s = ck_problem()
    @testset "complex t with zero imaginary part equals real t" begin
        for t in (-0.7, 0.0, 0.31)
            @test evaluate_cork_polynomial(P, ComplexF64(t)) ≈ evaluate_cork_polynomial(P, t) atol = 1e-13
        end
    end
    @testset "matches A(k) at a complex k (the cubic is reproduced exactly)" begin
        for k in (3.05 + 0.04im, 2.9 - 0.1im)
            t = (k - P.k0) / P.Δ
            @test evaluate_cork_polynomial(P, t) ≈ ck_matrix(s, k) atol = 1e-10
        end
    end
end

@testset "cork_left_vectors / cork_left_residuals" begin
    P = ck_polynomial(); s = ck_problem()
    ks = ComplexF64.(CK_INSIDE)
    Y = cork_left_vectors(P, ks)

    @testset "shape, unit norm and left null property" begin
        @test Y isa Matrix{ComplexF64} && size(Y) == (CK_N, 4)
        @test all(j -> norm(Y[:, j]) ≈ 1, 1:4)
        for j in 1:4
            t = (ks[j] - P.k0) / P.Δ
            @test norm(Y[:, j]' * evaluate_cork_polynomial(P, t)) < 1e-8
        end
    end

    @testset "simple roots align with the columns of Q" begin
        for (j, m) in ((1, 11), (4, 14))
            @test abs(s.Q[:, m]' * Y[:, j]) ≈ 1 atol = 1e-8
        end
    end

    @testset "degenerate pair is orthonormal and spans Q[:,12:13]" begin
        Yd = Y[:, 2:3]
        @test Yd' * Yd ≈ Matrix(I, 2, 2) atol = 1e-8
        Qd = s.Q[:, CK_DOUBLE_IDX]
        @test norm(Yd - Qd * (Qd' * Yd)) < 1e-8
    end

    @testset "empty root list" begin
        @test size(cork_left_vectors(P, ComplexF64[])) == (CK_N, 0)
    end

    @testset "residuals: small for correct vectors, large for a wrong vector" begin
        r = cork_left_residuals(P, ks, Y, 1:4)
        @test r isa Vector{Float64} && length(r) == 4
        @test maximum(r) < 1e-8
        Yw = copy(Y); Yw[:, 1] = ComplexF64.(s.Q[:, 1])
        @test cork_left_residuals(P, ks, Yw, [1])[1] > 1e-3
        @test length(cork_left_residuals(P, ks, Y, [2, 4])) == 2
    end
end

@testset "_cork_solve: side = :left" begin
    s = ck_problem(); solver = ck_solver()
    ksR, tsR, ΨR = _cork_solve(solver, CK_PTS, CK_K0, CK_DK; multithreaded=false, vectors=true)
    ksL, tsL, ΨL = _cork_solve(solver, CK_PTS, CK_K0, CK_DK; multithreaded=false, vectors=true, side=:left)

    @testset "same roots as the right run, left vectors as Ψ" begin
        @test ksL ≈ ksR
        @test size(ΨL) == size(ΨR) == (CK_N, 4)
        @test all(j -> norm(ΨL[:, j]) ≈ 1, 1:4)
        for j in 1:4
            @test norm(ΨL[:, j]' * ck_matrix(s, ksL[j])) < 1e-5
        end
    end

    @testset "left vectors span the same invariant subspace as Q[:,11:14]" begin
        Qi = s.Q[:, CK_INSIDE_IDX]
        @test norm(ΨL - Qi * (Qi' * ΨL)) < 1e-5
    end

    @testset "argument validation" begin
        @test_throws ArgumentError _cork_solve(solver, CK_PTS, CK_K0, CK_DK; multithreaded=false, side=:left)
        @test_throws ArgumentError _cork_solve(solver, CK_PTS, CK_K0, CK_DK; multithreaded=false, vectors=true, side=:middle)
    end

    @testset "_cork_left_vectors: empty root list" begin
        P = ck_polynomial()
        @test size(_cork_left_vectors(solver, CK_PTS, P, NTuple{3,Float64}[]; multithreaded=false)) == (CK_N, 0)
    end
end
# The fallback of `_cork_left_vectors` (polynomial residual above tolerance, recomputation on the kernel matrix through
# `_posthoc_left_vectors`) is not tested here: it needs a polynomial inconsistent with the kernel, which the exact synthetic problem cannot provide.
