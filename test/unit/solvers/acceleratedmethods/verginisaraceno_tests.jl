# Unit tests for
#   src/solvers/acceleratedmethods/verginisaraceno.jl   (VerginiSaracenoSolver: evaluate_points, construct_matrices with and
#                                                        without a basis cache, sm_results and the solve* wrappers on tiny pencils)
# The Vergini-Saraceno method builds F = B' W B and Fk = B' W dB/dk + (dB/dk)' W B with W = diag(w_vs), w_vs = ds / (r . n), both
# multiplied by the number of basis symmetries nsym = length(basis.symmetries) + 1. Fk is therefore the k-derivative of F at
# fixed weights. The generalized eigenvalues mu of the pair give the estimates k_s = k - 2/mu + 2/(k mu^2), tension 2 (2/mu)^2.
# Checks, all without a search for eigenvalues:
#   * evaluate_points: only xy and w_vs are populated, w_vs = ds / (r . n) in closed form on the first-quadrant rectangle and
#     the quarter circle, point counts and scaling factors, the Float32 type;
#   * construct_matrices on a toy basis f_i = i k (x + 2y) whose F and Fk are known in closed form, on RealPlaneWaves against the
#     dense formulas, Fk = dF/dk by central differences, symmetry factors, symmetry and sign properties, threaded = serial, and
#     the cached method against the direct one at k0 and nearby, with its argument check;
#   * sm_results at hand-computed values and its limits;
#   * solve with precomputed diagonal pencils (mu known exactly, window filter, ordering) and the one-function toy basis, for
#     which k_s = k / 2 and tension 2 k^2 in closed form, solve_vectors on that same one-function system, and the cached path.
# Helper definitions use the prefix `vs_`.

using LinearAlgebra: Diagonal, issymmetric, eigvals, Symmetric, norm
using StaticArrays: SVector
using QuantumBilliards: sm_results, basis_matrix, dk_matrix, basis_cache, _determine_bp_sizes

# Toy basis f_i(x, y) = i k (x + 2y), so df_i/dk = i (x + 2y).
struct vs_ToyBasis{S} <: AbsBasis
    dim::Int
    symmetries::S
end
vs_ToyBasis(dim::Int) = vs_ToyBasis(dim, nothing)
QuantumBilliards.basis_fun(::vs_ToyBasis, indices::AbstractArray, k::Real, pts::AbstractArray; multithreaded::Bool=true) =
    [i * k * (p[1] + 2p[2]) for p in pts, i in indices]
QuantumBilliards.dk_fun(::vs_ToyBasis, indices::AbstractArray, k::Real, pts::AbstractArray; multithreaded::Bool=true) =
    [i * (p[1] + 2p[2]) for p in pts, i in indices]

# The 2 x 1.4 rectangle, whose fundamental boundary is the right edge x = 1 (length 0.7, r.n = 1) followed by the top edge
# y = 0.7 (length 1.0, r.n = 0.7).
vs_rect() = RectangleBilliard(1.0, 0.7)

# sum over the fundamental boundary of w_vs (x + 2y)^2, exact for the Gauss-Legendre nodes (quadratic integrands).
const vs_S = (2.4^3 - 1) / 6 + (2.4^3 - 1.4^3) / (3 * 0.7)

# Fundamental boundary points of the rectangle at k and their Vergini-Saraceno weights.
vs_points(k=20.0; scaling=3.0) = evaluate_points(VerginiSaracenoSolver(2.0, scaling), vs_rect(), k)

@testset "VerginiSaracenoSolver evaluate_points: rectangle" begin
    rect = vs_rect()
    solver = VerginiSaracenoSolver(2.0, 3.0)
    k = 20.0
    pts = evaluate_points(solver, rect, k)
    Ns = _determine_bp_sizes(get_boundary_curves(rect), solver.pts_scaling_factor, k)
    right, top = 1:Ns[1], Ns[1]+1:sum(Ns)

    @testset "returned fields" begin
        @test pts isa BoundaryPoints{Float64}
        @test length(pts) == sum(Ns) > 40
        @test length(pts.xy) == length(pts.w_vs) == length(pts)
        @test isempty(pts.normal) && isempty(pts.ds) && isempty(pts.s)      # only the positions and w_vs are stored
        @test isempty(pts.w_dm) && isempty(pts.xy_int)
    end

    @testset "nodes lie on the two fundamental edges, in curve order" begin
        @test all(p -> p[1] == 1.0 && 0.0 < p[2] < 0.7, pts.xy[right])
        @test all(p -> p[2] == 0.7 && 0.0 < p[1] < 1.0, pts.xy[top])
    end

    @testset "w_vs = ds / (r . n)" begin
        @test all(>(0), pts.w_vs)
        @test sum(pts.w_vs[right]) ≈ 0.7 / 1.0                              # r.n = 1 on the right edge, length 0.7
        @test sum(pts.w_vs[top]) ≈ 1.0 / 0.7                                # r.n = 0.7 on the top edge, length 1.0
        @test sum(pts.w_vs) ≈ 0.7 + 1.0 / 0.7
    end

    @testset "the weight sums do not depend on k" begin
        for kk in (6.0, 11.0, 35.0)
            @test sum(evaluate_points(solver, rect, kk).w_vs) ≈ 0.7 + 1.0 / 0.7
        end
    end

    @testset "w_vs is consistent with the decomposition-method quadrature" begin
        dm = evaluate_points(DecompositionMethodSolver(2.0, 3.0), rect, k)
        @test dm.xy == pts.xy                                               # same boundary nodes
        @test pts.w_vs ≈ dm.ds ./ [sum(p .* n) for (p, n) in zip(dm.xy, dm.normal)]
    end
end

@testset "VerginiSaracenoSolver evaluate_points: curved boundary" begin
    circle = CircleBilliard(1.0)
    pts = evaluate_points(VerginiSaracenoSolver(2.0, 3.0), circle, 12.0)
    @test all(p -> norm(p) ≈ 1.0, pts.xy)
    @test pts.w_vs ≈ evaluate_points(DecompositionMethodSolver(2.0, 3.0), circle, 12.0).ds   # r.n = 1 on the unit circle
    @test sum(pts.w_vs) ≈ π / 2                                             # the quarter arc
end

@testset "VerginiSaracenoSolver evaluate_points: sampling rules" begin
    rect = vs_rect()
    curves = get_boundary_curves(rect)

    @testset "the number of points follows the scaling factors" begin
        small = length(vs_points(20.0))
        @test small == sum(_determine_bp_sizes(curves, [3.0, 3.0], 20.0))
        @test abs(length(vs_points(40.0)) - 2small) <= 2                    # linear in k up to rounding
        @test abs(length(vs_points(20.0; scaling=6.0)) - 2small) <= 2       # linear in the point scaling factor
    end

    @testset "per-curve scaling factors" begin
        pts = evaluate_points(VerginiSaracenoSolver(2.0, [3.0, 6.0]), rect, 20.0)
        @test length(pts) == sum(_determine_bp_sizes(curves, [3.0, 6.0], 20.0))
    end

    @testset "at least 20 points per curve" begin
        @test length(vs_points(1.0; scaling=0.01)) == 40
    end

    @testset "sampler choice changes the nodes but not the integrals" begin
        samplers = AbsSampler[LinearNodes(), LinearNodes()]
        pts = evaluate_points(VerginiSaracenoSolver(2.0, 3.0, samplers), rect, 20.0)
        @test sum(pts.w_vs) ≈ 0.7 + 1.0 / 0.7
    end

    @testset "Float32 solver" begin
        pts = evaluate_points(VerginiSaracenoSolver(2f0, 3f0), rect, 20f0)
        @test pts isa BoundaryPoints{Float32}
        @test sum(pts.w_vs) ≈ Float32(0.7 + 1.0 / 0.7)
    end

    @testset "argument validation" begin
        solver = VerginiSaracenoSolver(2.0, 3.0)
        @test_throws MethodError evaluate_points(solver, 1.0, 20.0)         # not a billiard
        @test_throws MethodError evaluate_points(1.0, rect, 20.0)           # not a solver
    end
end

@testset "VerginiSaracenoSolver construct_matrices: toy basis in closed form" begin
    solver = VerginiSaracenoSolver(2.0, 3.0)
    k = 7.0
    pts = vs_points(k)
    v = [1.0, 2.0, 3.0]                                                     # f_i has the amplitude i

    @testset "F and Fk of the toy basis" begin
        F, Fk = construct_matrices(solver, vs_ToyBasis(3), pts, k)
        # F_ab = a b k^2 S and Fk_ab = 2 a b k S, S = sum w_vs (x + 2y)^2
        @test size(F) == size(Fk) == (3, 3)
        @test F ≈ k^2 * vs_S .* (v * v')
        @test Fk ≈ 2k * vs_S .* (v * v')
        @test issymmetric(F) && issymmetric(Fk)
        @test Fk ≈ (2 / k) .* F                                             # Fk = dF/dk for F proportional to k^2
    end

    @testset "F grows like k^2 and Fk like k" begin
        F1, Fk1 = construct_matrices(solver, vs_ToyBasis(3), pts, 7.0)
        F2, Fk2 = construct_matrices(solver, vs_ToyBasis(3), pts, 14.0)
        @test F2 ≈ 4 .* F1
        @test Fk2 ≈ 2 .* Fk1
    end

    @testset "the symmetry count scales both matrices" begin
        F1, Fk1 = construct_matrices(solver, vs_ToyBasis(3), pts, k)
        F2, Fk2 = construct_matrices(solver, vs_ToyBasis(3, [XAxisReflection()]), pts, k)   # nsym = 2
        @test F2 ≈ 2 .* F1 && Fk2 ≈ 2 .* Fk1
        four = vs_ToyBasis(3, [XAxisReflection(), YAxisReflection(), XYAxisReflection()])      # nsym = 4
        F4, Fk4 = construct_matrices(solver, four, pts, k)
        @test F4 ≈ 4 .* F1 && Fk4 ≈ 4 .* Fk1
    end

    @testset "the basis dimension sets the matrix size" begin
        for dim in (1, 5)
            F, Fk = construct_matrices(solver, vs_ToyBasis(dim), pts, k)
            @test size(F) == size(Fk) == (dim, dim)
        end
    end

    @testset "the weights enter linearly" begin
        doubled = BoundaryPoints(pts.xy; w_vs=2 .* pts.w_vs)
        F, Fk = construct_matrices(solver, vs_ToyBasis(3), pts, k)
        F2, Fk2 = construct_matrices(solver, vs_ToyBasis(3), doubled, k)
        @test F2 ≈ 2 .* F && Fk2 ≈ 2 .* Fk
    end

    @testset "multithreaded keyword" begin
        Fs, Fks = construct_matrices(solver, vs_ToyBasis(3), pts, k; multithreaded=false)
        Ft, Fkt = construct_matrices(solver, vs_ToyBasis(3), pts, k; multithreaded=true)
        @test Fs ≈ Ft && Fks ≈ Fkt
    end
end

@testset "VerginiSaracenoSolver construct_matrices: RealPlaneWaves on the rectangle" begin
    solver = VerginiSaracenoSolver(2.0, 3.0)
    k = 6.0
    pts = vs_points(k)
    basis = RealPlaneWaves(8; sym_x=-1, sym_y=-1)                           # odd-odd, D2: three registered reflections
    nsym = length(basis.symmetries) + 1
    F, Fk = construct_matrices(solver, basis, pts, k)
    B = basis_matrix(basis, k, pts.xy)
    dB = dk_matrix(basis, k, pts.xy)
    W = Diagonal(pts.w_vs)

    @testset "shapes and symmetry" begin
        @test nsym == 4
        @test size(F) == size(Fk) == (basis.dim, basis.dim)
        @test issymmetric(F) && issymmetric(Fk)
        @test eltype(F) == Float64
    end

    @testset "dense formulas" begin
        @test F ≈ nsym .* (B' * W * B) rtol = 1e-10
        @test Fk ≈ nsym .* (B' * W * dB + dB' * W * B) rtol = 1e-10
    end

    @testset "F is positive semi-definite for positive weights" begin
        @test minimum(eigvals(Symmetric(F))) > -1e-10 * maximum(abs, F)
    end

    @testset "Fk is the k-derivative of F at fixed weights" begin
        h = 1e-5
        Fp, _ = construct_matrices(solver, basis, pts, k + h)
        Fm, _ = construct_matrices(solver, basis, pts, k - h)
        @test (Fp - Fm) / (2h) ≈ Fk rtol = 1e-6 atol = 1e-6 * maximum(abs, Fk)
    end

    @testset "threaded equals serial" begin
        Fs, Fks = construct_matrices(solver, basis, pts, k; multithreaded=false)
        @test Fs ≈ F && Fks ≈ Fk
    end

    @testset "a basis without symmetries has nsym = 1" begin
        plain = RealPlaneWaves(8)
        Fp, Fkp = construct_matrices(solver, plain, pts, k)
        Bp = basis_matrix(plain, k, pts.xy)
        dBp = dk_matrix(plain, k, pts.xy)
        @test Fp ≈ Bp' * W * Bp rtol = 1e-10
        @test Fkp ≈ Bp' * W * dBp + dBp' * W * Bp rtol = 1e-10
    end
end

@testset "VerginiSaracenoSolver construct_matrices: cached basis" begin
    solver = VerginiSaracenoSolver(2.0, 3.0)
    k0 = 6.0
    pts = vs_points(k0)
    basis = RealPlaneWaves(8; sym_x=-1, sym_y=-1)
    cache = basis_cache(basis, k0, pts.xy)

    @testset "at the expansion wavenumber the cache reproduces the direct matrices" begin
        F, Fk = construct_matrices(solver, basis, pts, k0)
        Fc, Fkc = construct_matrices(solver, basis, pts, cache, k0)
        @test Fc ≈ F rtol = 1e-9
        @test Fkc ≈ Fk rtol = 1e-9
        @test issymmetric(Fc) && issymmetric(Fkc)
    end

    @testset "close to the expansion wavenumber" begin
        k = k0 + 0.02
        F, Fk = construct_matrices(solver, basis, pts, k)
        Fc, Fkc = construct_matrices(solver, basis, pts, cache, k)
        @test Fc ≈ F rtol = 1e-6
        @test Fkc ≈ Fk rtol = 1e-6
    end

    @testset "threaded equals serial" begin
        Fs, Fks = construct_matrices(solver, basis, pts, cache, k0; multithreaded=false)
        Ft, Fkt = construct_matrices(solver, basis, pts, cache, k0; multithreaded=true)
        @test Fs ≈ Ft && Fks ≈ Fkt
    end

    @testset "the weights must match the cached points" begin
        shorter = BoundaryPoints(pts.xy[1:end-1]; w_vs=pts.w_vs[1:end-1])
        @test_throws DimensionMismatch construct_matrices(solver, basis, shorter, cache, k0)
    end
end

@testset "sm_results" begin
    @testset "hand-computed values" begin
        ks, ten = sm_results([2.0, -4.0], 10.0)
        @test ks ≈ [10 - 1 + 2 / (10 * 4), 10 + 0.5 + 2 / (10 * 16)]        # k - 2/mu + 2/(k mu^2)
        @test ten ≈ [2.0 * 1.0^2, 2.0 * 0.5^2]                              # 2 (2/mu)^2
    end

    @testset "limits and symmetry" begin
        ks, ten = sm_results([1e6, -1e6], 10.0)
        @test all(≈(10.0; atol=1e-5), ks)                                   # the correction vanishes for large |mu|
        @test all(<(1e-10), ten)
        mu = [0.7, 3.0, 25.0]
        kp, tp = sm_results(mu, 9.0)
        km, tm = sm_results(-mu, 9.0)
        @test tp ≈ tm                                                       # the tension is even in mu
        @test kp .+ km ≈ 2 * 9.0 .+ 4 ./ (9.0 .* mu .^ 2)                   # the linear shifts cancel, the quadratic ones add
    end

    @testset "output shape" begin
        ks, ten = sm_results(Float64[], 10.0)
        @test isempty(ks) && isempty(ten)
        ks, ten = sm_results(ones(5), 10.0)
        @test length(ks) == length(ten) == 5
    end
end

@testset "VerginiSaracenoSolver solve on tiny systems" begin
    solver = VerginiSaracenoSolver(2.0, 3.0)

    @testset "diagonal pencil: mu known exactly, window filter and ordering" begin
        # generalized_eigvals(F, Fk) are the eigenvalues of Fk x = mu F x, here the diagonal of Fk
        F = Matrix(Diagonal(ones(3)))
        Fk = Matrix(Diagonal([2.0, -4.0, 1000.0]))
        k = 10.0
        # sorted by k_s the estimates belong to mu = 2, 1000, -4
        ks_expected = [10 - 2 / 2 + 2 / (10 * 4), 10 - 2 / 1000 + 2 / (10 * 1e6), 10 + 2 / 4 + 2 / (10 * 16)]
        ten_expected = [2 * (2 / 2)^2, 2 * (2 / 1000)^2, 2 * (2 / 4)^2]
        ks, ten = solve(solver, F, Fk, k, 5.0)                              # a wide window keeps every candidate
        @test length(ks) == length(ten) == 3
        @test ks ≈ ks_expected
        @test ten ≈ ten_expected

        ks, ten = solve(solver, F, Fk, k, 0.6)                              # |k_s - k| < 0.6 drops mu = 2 (|k_s - k| = 0.95)
        @test length(ks) == 2
        @test ks ≈ ks_expected[2:3]
        @test ten ≈ ten_expected[2:3]

        ks, ten = solve(solver, F, Fk, k, 1e-4)                             # nothing is that close
        @test isempty(ks) && isempty(ten)
    end

    @testset "scaling F by a constant rescales mu and moves every estimate" begin
        F = Matrix(Diagonal([1.0, 1.0]))
        Fk = Matrix(Diagonal([2.0, -4.0]))
        ks1, _ = solve(solver, F, Fk, 10.0, 100.0)
        ks2, _ = solve(solver, 2 .* F, Fk, 10.0, 100.0)                     # mu -> mu / 2
        @test ks2 ≈ sort(sm_results([1.0, -2.0], 10.0)[1])
        @test !(ks1 ≈ ks2)
    end

    @testset "one-function toy basis: k_s = k / 2 and tension 2 k^2" begin
        # F = k^2 S and Fk = 2 k S give mu = 2 / k, so k - 2/mu + 2/(k mu^2) = k/2 and 2 (2/mu)^2 = 2 k^2
        basis = vs_ToyBasis(1)
        for k in (4.0, 9.0)
            pts = vs_points(k)
            ks, ten = solve(solver, basis, pts, k, k)                       # the window must exceed k / 2
            @test ks ≈ [k / 2]
            @test ten ≈ [2k^2]
            @test isempty(solve(solver, basis, pts, k, k / 2 - 0.1)[1])     # |k_s - k| = k / 2 is outside the window
        end
        F, Fk = construct_matrices(solver, basis, vs_points(4.0), 4.0)
        @test solve(solver, F, Fk, 4.0, 4.0) == solve(solver, basis, vs_points(4.0), 4.0, 4.0)
    end

    @testset "solve_vectors on the one-function toy basis" begin
        basis = vs_ToyBasis(1)
        k = 4.0
        pts = vs_points(k)
        F, _ = construct_matrices(solver, basis, pts, k)
        ks, ten, X = solve_vectors(solver, basis, pts, k, k)
        @test ks ≈ [k / 2] && ten ≈ [2k^2]
        @test size(X) == (1, 1)
        @test (X' * F * X)[1] ≈ ten[1]                                      # X is scaled so that X' F X = diag(ten)
        ks0, ten0, X0 = solve_vectors(solver, basis, pts, k, 0.1)           # nothing in the window
        @test isempty(ks0) && isempty(ten0) && size(X0) == (1, 0)
    end

    @testset "argument validation" begin
        pts = vs_points(4.0)
        @test_throws MethodError solve(solver, vs_ToyBasis(1), pts, 4.0)    # the window dk is required
        @test_throws MethodError solve(solver, 1.0, pts, 4.0, 1.0)          # not a basis
    end
end

@testset "VerginiSaracenoSolver solve with a basis cache" begin
    solver = VerginiSaracenoSolver(2.0, 3.0)
    k0 = 6.0
    pts = vs_points(k0)
    basis = RealPlaneWaves(4; sym_x=-1, sym_y=-1)
    cache = basis_cache(basis, k0, pts.xy)

    @testset "solve agrees with the direct method at k0" begin
        ks, ten = solve(solver, basis, pts, k0, 100.0)
        ksc, tenc = solve(solver, basis, pts, cache, k0, 100.0)
        @test length(ksc) == length(ks) == length(tenc) == length(ten)
        @test ksc ≈ ks rtol = 1e-6
        @test tenc ≈ ten rtol = 1e-4
    end

    @testset "solve_vectors agrees with the direct method at k0" begin
        ks, ten, X = solve_vectors(solver, basis, pts, k0, 100.0)
        ksc, tenc, Xc = solve_vectors(solver, basis, pts, cache, k0, 100.0)
        @test size(Xc) == size(X)
        @test ksc ≈ ks rtol = 1e-6
        @test abs.(Xc) ≈ abs.(X) rtol = 1e-4
    end
end
