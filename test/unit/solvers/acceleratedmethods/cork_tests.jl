# Unit tests for
#   src/solvers/acceleratedmethods/cork.jl   (taylor_to_cheb!, build_cork_polynomial / build_B_full, CORKPolynomial,
#                                             evaluate_cork_polynomial, validate_polynomial, get_A0, cache_block!,
#                                             compact_project!, compact_block_qr!, pack! / unpack!, init_cork, compute_tail! /
#                                             promote_tail! / extend!, ritz_roots, reconstruct_cork_eigenvectors, adaptive_cork,
#                                             _cork_solve_core via solve / solve_vectors, solve_wavenumber, solve_spectrum)
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

using LinearAlgebra: Diagonal, qr, norm, svdvals, lu, I, tril
using Random: MersenneTwister
using QuantumBilliards: construct_matrices, _bim_numeric_type, TaylorWorkspace, power_to_cheb, taylor_to_cheb!, CORKPolynomial,
                        build_cork_polynomial, evaluate_cork_polynomial, validate_polynomial, get_A0, cache_block!,
                        compact_project!, compact_block_qr!, pack!, unpack!, init_cork, compute_tail!, promote_tail!, extend!,
                        ritz_roots, reconstruct_cork_eigenvectors, adaptive_cork, _cork_solve_core, _cork_kernel_name

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
ck_solver(; kwargs...) = CORKSolver(ck_problem(); merge((p=3, b=2, mstart=12, mstep=2, res_tol=1e-6), NamedTuple(kwargs))...)

# Degree-p Chebyshev polynomial of the synthetic problem about k0 with half-width Δ, and the LU factorization of P(0).
ck_polynomial(p::Int=3; k0=CK_K0, Δ=CK_DPOLY) = build_cork_polynomial(ck_problem(), CK_PTS, k0, Δ, p; multithreaded=false)[1]
ck_factor(P::CORKPolynomial) = lu(get_A0(P.B, P.N, P.p))

# CORK state extended to the compact Krylov dimension m (a normalized tail is pending).
function ck_state(m::Int; b::Int=2, p::Int=3)
    P = ck_polynomial(p); F = ck_factor(P)
    S = init_cork(P.B, p, P.N; b=b)
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
        t = cache_block!(W, B, U, N, p, 2, 4)
        @test t isa Float64 && t >= 0
        for j in 0:p
            rows = j * N + 1:(j + 1) * N
            @test W[rows, 2:4] ≈ B[rows, :] * U[:, 2:4]
        end
        @test iszero(W[:, 1]) && iszero(W[:, 5:end])
    end

    @testset "an empty column range does nothing" begin
        W = zeros(ComplexF64, (p + 1) * N, rmax)
        @test cache_block!(W, B, U, N, p, 4, 3) == 0.0
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

@testset "init_cork: initial compact block-Arnoldi state" begin
    P = ck_polynomial(3)

    @testset "dimension checks" begin
        B = zeros(ComplexF64, 4 * 4, 4)
        @test_throws "exceeds physical matrix dimension" init_cork(zeros(ComplexF64, 6 * 4, 4), 5, 4; b=2)     # p > N
        @test_throws "smaller than block size" init_cork(B, 3, 4; b=5)                                       # N < b
        @test_throws "smaller than block size" init_cork(zeros(ComplexF64, 3 * 12, 12), 2, 12; b=5)           # p^2 < b
    end

    @testset "state layout" begin
        S = init_cork(P.B, 3, CK_N; b=2)
        @test S isa QuantumBilliards.CORKState
        @test (S.r, S.n, S.b, S.p, S.N, S.rmax) == (3, 2, 2, 3, CK_N, CK_N)
        @test !S.pending && S.napply == 0
        @test size(S.Hb) == (CK_N + 2, CK_N)                                 # mdim = floor(N / b) b
        @test size(S.G) == (CK_N, 3, CK_N + 2)
        @test size(S.W) == (4 * CK_N, CK_N)
    end

    @testset "physical basis, cached actions and starting block" begin
        S = init_cork(P.B, 3, CK_N; b=2)
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
        S1 = init_cork(P.B, 3, CK_N; b=2); S2 = init_cork(P.B, 3, CK_N; b=2)
        @test S1.U == S2.U && S1.G == S2.G
    end
end

@testset "compute_tail! / promote_tail! / extend!: Arnoldi bookkeeping" begin
    @testset "compute_tail! produces one pending orthonormal block and is idempotent" begin
        P = ck_polynomial(3); F = ck_factor(P)
        S = init_cork(P.B, 3, CK_N; b=2)
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
        S = init_cork(P.B, 3, CK_N; b=2)
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
        S = init_cork(P.B, 3, CK_N; b=2)
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
        S = init_cork(P1.B, 1, CK_N; b=1)
        @test_throws "requires p >= 2" compute_tail!(S, P1.B, ck_factor(P1))
    end
end

@testset "ritz_roots: roots of the projected inverse linearization" begin
    P, _, S = ck_state(CK_N)
    roots, V = ritz_roots(S, CK_K0, CK_DPOLY, CK_N)
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

    @testset "state checks" begin
        @test_throws "State dimension" ritz_roots(S, CK_K0, CK_DPOLY, CK_N - 2)
        S0 = init_cork(P.B, 3, CK_N; b=2)
        @test_throws "Arnoldi tail missing" ritz_roots(S0, CK_K0, CK_DPOLY, 2)
    end
end

@testset "reconstruct_cork_eigenvectors: Ritz vectors to Fredholm vectors" begin
    s = ck_problem()
    _, _, S = ck_state(CK_N)
    roots, V = ritz_roots(S, CK_K0, CK_DPOLY, CK_N)
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
    kw = (k0=CK_K0, Δ=CK_DELTA, Δpoly=CK_DPOLY, b=2, mstart=12, mstep=2, stable_checks=1, res_tol=1e-6)

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
        _, ks, Ψ, requested, _, edges, edge_good, _ = adaptive_cork(Pe.B, ck_factor(Pe), 3, CK_N; k0=4.0, Δ=0.2, Δpoly=0.2 * (1 + CK_GUARD), b=2, mstart=12, mstep=2, eigenvectors=true)
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
        solver1 = ck_solver(b=1)                                             # block size 1 converges within N also for p = 5
        ks3, _ = solve(solver1, CK_PTS, CK_K0, CK_DK; multithreaded=false)
        ks5, _ = solve(solver1, CK_PTS, CK_K0, CK_DK; multithreaded=false, p=5)
        @test real.(ks5) ≈ [2.85, 3.0, 3.3] atol = 1e-6
        @test ks5 ≈ ks3 atol = 1e-6
    end

    @testset "a window without roots returns empty outputs" begin
        ks0, res0, Ψ0 = solve_vectors(solver, CK_PTS, 4.0, 0.4; multithreaded=false)
        @test isempty(ks0) && isempty(res0)
        @test size(Ψ0) == (CK_N, 0)
    end

    @testset "multiplicity is limited by the block size" begin
        ks1, _ = solve(ck_solver(b=1, mstart=10, res_tol=1e-10), CK_PTS, CK_K0, CK_DK; multithreaded=false)
        @test real.(ks1) ≈ [2.85, 3.0, 3.3] atol = 1e-6                      # a single starting vector sees one copy of the double root
    end

    @testset "a too small initial Krylov dimension does not yield an empty spectrum when levels are expected" begin
        # mstart = 6 leaves no Ritz root in the window at m = 6 and m = 8; with the expected level count the empty
        # spectrum is not accepted and the Krylov dimension keeps growing until the four roots appear.
        ks_small, _ = solve(ck_solver(mstart=6), CK_PTS, CK_K0, CK_DK; multithreaded=false, expected=4.0)
        @test length(ks_small) == 4
        @test real.(ks_small) ≈ [2.85, 3.0, 3.0, 3.3] atol = 1e-6
    end

    @testset "an empty window with expected < 0.5 still returns an empty spectrum" begin
        ks_empty, _ = solve(ck_solver(), CK_PTS, 4.0, 0.4; multithreaded=false, expected=0.1)
        @test isempty(ks_empty)
        @test_throws "Reached full CORK Krylov dimension" solve(ck_solver(), CK_PTS, 4.0, 0.4; multithreaded=false, expected=3.0)
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

    @testset "eigenvectors keyword overrides the solver field" begin
        _, _, ks, Ψ = _cork_solve_core(ck_solver(), CK_PTS, CK_K0, CK_DK; multithreaded=false, eigenvectors=true)
        @test size(Ψ) == (CK_N, length(ks))
    end

    @testset "validate = true accepts an exact polynomial and rejects a truncated one" begin
        @test length(solve(ck_solver(validate=true), CK_PTS, CK_K0, CK_DK; multithreaded=false)[1]) == 4
        @test_throws ArgumentError solve(ck_solver(validate=true), CK_PTS, CK_K0, CK_DK; multithreaded=false, p=2)
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
        # the Weyl estimate of the unit circle for dk = 0.1 is below 0.5, so the empty Ritz set is accepted
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
