# Unit tests for
#   src/solvers/sweepmethods/decompositionmethod.jl   (DecompositionMethodSolver: evaluate_points, construct_matrices,
#                                                      solve from precomputed F and G)
# F = B' W B with W = diag(ds) and G = Bn' Wn Bn with Wn = diag(w_dm), w_dm = ds ((x - c0).n) / (2 k^2), both divided over the
# number of basis symmetries nsym = length(basis.symmetries) + 1. Checks, all without an eigen solve:
#   * evaluate_points: Rellich identity sum(ds (x - c0).n) = 2 A (area of the fundamental domain) on the first-quadrant
#     rectangle, whose axis edges drop out, and = pi/2 on the quarter unit circle; unit normals; w_dm k^2 independent of k;
#     point counts per curve and the effect of the scaling factors and the sampler;
#   * construct_matrices on a toy basis f_i = i k (x + 2y) whose F and G are known in closed form (Gauss-Legendre quadrature
#     is exact for the quadratic integrands), on RealPlaneWaves against the dense formulas, symmetry factors, positive
#     semi-definiteness for a Rellich origin inside the domain, threaded = serial;
#   * the 1x1 pencil of the one-function toy basis, where the tension is exactly F/G (the code solves G x = mu F x, t = 1/mu_max).
# Helper definitions use the prefix `dm_`.

using LinearAlgebra: Diagonal, Symmetric, eigvals, issymmetric, dot, norm
using StaticArrays: SVector
using QuantumBilliards: basis_and_gradient_matrices, basis_matrix, _determine_bp_sizes

# Toy basis f_i(x, y) = i k (x + 2y): gradient (i k, 2 i k), so no wave equation, but exact algebra for F and G.
struct dm_ToyBasis{S} <: AbsBasis
    dim::Int
    symmetries::S
end
dm_ToyBasis(dim::Int) = dm_ToyBasis(dim, nothing)
QuantumBilliards.basis_and_gradient(b::dm_ToyBasis, indices::AbstractArray, k::Real, pts::AbstractArray; multithreaded::Bool=true) =
    ([i * k * (p[1] + 2p[2]) for p in pts, i in indices],
     [i * k for p in pts, i in indices],
     [2i * k for p in pts, i in indices])

# The rectangle used throughout: RectangleBilliard(1.0, 0.7) is the 2 x 1.4 rectangle, its fundamental boundary the right edge
# x = 1 (length 0.7) followed by the top edge y = 0.7 (length 1.0).
dm_rect() = RectangleBilliard(1.0, 0.7)

@testset "DecompositionMethodSolver evaluate_points: rectangle fundamental boundary" begin
    rect = dm_rect()
    solver = DecompositionMethodSolver(2.0, 3.0)
    k = 20.0
    pts = evaluate_points(solver, rect, k)

    @testset "returned fields" begin
        @test pts isa BoundaryPoints{Float64}
        N = length(pts)
        @test N > 40
        @test length(pts.xy) == length(pts.normal) == length(pts.ds) == length(pts.w_dm) == N
        @test isempty(pts.xy_int)                                           # interior points belong to the PSM
        @test isempty(pts.w_vs)                                             # Vergini-Saraceno weights are not set
    end

    @testset "points lie on the two fundamental edges, in curve order" begin
        Ns = _determine_bp_sizes(get_boundary_curves(rect), solver.pts_scaling_factor, k)
        @test length(Ns) == 2 && sum(Ns) == length(pts)
        right = 1:Ns[1]
        top = Ns[1]+1:sum(Ns)
        @test all(p -> p[1] == 1.0 && 0.0 < p[2] < 0.7, pts.xy[right])
        @test all(p -> p[2] == 0.7 && 0.0 < p[1] < 1.0, pts.xy[top])
        @test all(n -> n == SVector(1.0, 0.0), pts.normal[right])
        @test all(n -> n == SVector(0.0, 1.0), pts.normal[top])
        @test all(n -> norm(n) ≈ 1.0, pts.normal)
    end

    @testset "quadrature weights" begin
        @test all(>(0), pts.ds)
        @test sum(pts.ds) ≈ 1.7                                             # fundamental boundary length 0.7 + 1.0
        Ns = _determine_bp_sizes(get_boundary_curves(rect), solver.pts_scaling_factor, k)
        @test sum(pts.ds[1:Ns[1]]) ≈ 0.7
        @test sum(pts.ds[Ns[1]+1:end]) ≈ 1.0
    end

    @testset "Rellich weights w_dm = ds (x - c0).n / (2 k^2)" begin
        Ns = _determine_bp_sizes(get_boundary_curves(rect), solver.pts_scaling_factor, k)
        right, top = 1:Ns[1], Ns[1]+1:sum(Ns)
        @test pts.w_dm[right] ≈ pts.ds[right] .* 1.0 ./ (2k^2)              # (x - 0).n = 1 on the right edge
        @test pts.w_dm[top] ≈ pts.ds[top] .* 0.7 ./ (2k^2)                  # (x - 0).n = 0.7 on the top edge
        @test all(>(0), pts.w_dm)
        @test sum(pts.w_dm) * k^2 ≈ 0.7                                     # Rellich identity: sum ds (x.n) / 2 = area of the quadrant
    end

    @testset "w_dm k^2 does not depend on k" begin
        for kk in (6.0, 11.0, 35.0)
            p = evaluate_points(solver, rect, kk)
            @test sum(p.w_dm) * kk^2 ≈ 0.7
            @test sum(p.ds) ≈ 1.7
        end
    end
end

@testset "DecompositionMethodSolver evaluate_points: Rellich origin" begin
    rect = dm_rect()
    k = 20.0

    @testset "shifted origin" begin
        c0 = SVector(0.5, 0.2)
        solver = DecompositionMethodSolver(2.0, 3.0; rellich_origin=c0)
        pts = evaluate_points(solver, rect, k)
        # sum (x - c0).n ds / 2 = A - c0 . (sum n ds) / 2 with sum n ds = (0.7, 1.0)
        @test sum(pts.w_dm) * k^2 ≈ 0.7 - (c0[1] * 0.7 + c0[2] * 1.0) / 2
        @test pts.w_dm ≈ pts.ds .* [dot(p - c0, n) for (p, n) in zip(pts.xy, pts.normal)] ./ (2k^2)
        @test sum(pts.ds) ≈ 1.7                                             # the quadrature itself does not depend on the origin
    end

    @testset "origin on a boundary edge drops that edge" begin
        solver = DecompositionMethodSolver(2.0, 3.0; rellich_origin=SVector(1.0, 0.0))
        pts = evaluate_points(solver, rect, k)
        Ns = _determine_bp_sizes(get_boundary_curves(rect), solver.pts_scaling_factor, k)
        @test all(iszero, pts.w_dm[1:Ns[1]])                                # (x - c0).n = 0 on the whole right edge x = 1
        @test all(>(0), pts.w_dm[Ns[1]+1:end])
    end

    @testset "an origin outside the domain gives negative weights" begin
        solver = DecompositionMethodSolver(2.0, 3.0; rellich_origin=SVector(2.0, 0.0))
        pts = evaluate_points(solver, rect, k)
        Ns = _determine_bp_sizes(get_boundary_curves(rect), solver.pts_scaling_factor, k)
        @test all(<(0), pts.w_dm[1:Ns[1]])                                  # (x - c0).n = -1 on the right edge
    end
end

@testset "DecompositionMethodSolver evaluate_points: curved boundary" begin
    circle = CircleBilliard(1.0)
    solver = DecompositionMethodSolver(2.0, 3.0)
    k = 12.0
    pts = evaluate_points(solver, circle, k)
    @test length(get_boundary_curves(circle)) == 1
    @test all(p -> norm(p) ≈ 1.0, pts.xy)
    @test pts.normal ≈ pts.xy                                               # outward unit normal of the unit circle
    @test sum(pts.ds) ≈ π / 2                                               # the quarter arc
    @test pts.w_dm ≈ pts.ds ./ (2k^2)                                       # (x - 0).n = 1 on the unit circle
    @test sum(pts.w_dm) * k^2 ≈ π / 4
    @test all(p -> p[1] > 0 && p[2] > 0, pts.xy)                            # first quadrant
end

@testset "DecompositionMethodSolver evaluate_points: sampling rules" begin
    rect = dm_rect()
    curves = get_boundary_curves(rect)

    @testset "the number of points follows the scaling factors" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        @test length(evaluate_points(solver, rect, 20.0)) == sum(_determine_bp_sizes(curves, [3.0, 3.0], 20.0))
        small = length(evaluate_points(solver, rect, 20.0))
        large = length(evaluate_points(solver, rect, 40.0))
        @test abs(large - 2small) <= 2                                      # linear in k up to rounding
        double = DecompositionMethodSolver(2.0, 6.0)
        @test abs(length(evaluate_points(double, rect, 20.0)) - 2small) <= 2     # linear in the point scaling factor
    end

    @testset "per-curve scaling factors" begin
        solver = DecompositionMethodSolver(2.0, [3.0, 6.0])
        pts = evaluate_points(solver, rect, 20.0)
        Ns = _determine_bp_sizes(curves, [3.0, 6.0], 20.0)
        @test length(pts) == sum(Ns)
        @test Ns[2] / Ns[1] ≈ 6.0 / 3.0 * 1.0 / 0.7 rtol = 0.05             # edge lengths 1.0 and 0.7
        @test sum(pts.ds) ≈ 1.7
    end

    @testset "at least 20 points per curve" begin
        pts = evaluate_points(DecompositionMethodSolver(2.0, 0.01), rect, 1.0)
        @test length(pts) == 40
    end

    @testset "a missing scaling factor is padded with the smallest one" begin
        solver = DecompositionMethodSolver(2.0, 5.0)                          # one factor for two curves
        pts = evaluate_points(solver, rect, 20.0)
        @test length(pts) == sum(_determine_bp_sizes(curves, [5.0, 5.0], 20.0))
    end

    @testset "sampler choice changes the nodes but not the integrals" begin
        samplers = AbsSampler[LinearNodes(), LinearNodes()]
        solver = DecompositionMethodSolver(2.0, 3.0, samplers)
        pts = evaluate_points(solver, rect, 20.0)
        @test sum(pts.ds) ≈ 1.7
        @test sum(pts.w_dm) * 20.0^2 ≈ 0.7
    end

    @testset "Float32 solver" begin
        solver = DecompositionMethodSolver(2f0, 3f0)
        pts = evaluate_points(solver, rect, 20f0)
        @test pts isa BoundaryPoints{Float32}
        @test sum(pts.ds) ≈ 1.7f0
    end

    @testset "argument validation" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        @test_throws MethodError evaluate_points(solver, 1.0, 20.0)         # not a billiard
        @test_throws MethodError evaluate_points(1.0, rect, 20.0)           # not a solver
    end
end

@testset "DecompositionMethodSolver construct_matrices: toy basis in closed form" begin
    rect = dm_rect()
    k = 7.0
    v = [1.0, 2.0, 3.0]                                                     # f_i has the amplitude i

    @testset "F and G of the toy basis" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        pts = evaluate_points(solver, rect, k)
        F, G = construct_matrices(solver, dm_ToyBasis(3), pts, k)
        # F_ab = a b k^2 int (x + 2y)^2 ds = a b k^2 (int_0^.7 (1 + 2y)^2 dy + int_0^1 (x + 1.4)^2 dx) = a b k^2 (2.4^3 - 1)/6 + (2.4^3 - 1.4^3)/3
        cF = k^2 * ((2.4^3 - 1) / 6 + (2.4^3 - 1.4^3) / 3)
        # G_ab = a b k^2 sum w_dm (nx + 2 ny)^2 = a b (0.7 * 1 * 1 + 1.0 * 0.7 * 4) / 2
        cG = (0.7 * 1.0 * 1.0 + 1.0 * 0.7 * 4.0) / 2
        @test size(F) == size(G) == (3, 3)
        @test F ≈ cF .* (v * v')
        @test G ≈ cG .* (v * v')
        @test issymmetric(F) && issymmetric(G)
    end

    @testset "shifted Rellich origin only changes G" begin
        c0 = SVector(0.5, 0.2)
        solver = DecompositionMethodSolver(2.0, 3.0; rellich_origin=c0)
        pts = evaluate_points(solver, rect, k)
        F, G = construct_matrices(solver, dm_ToyBasis(3), pts, k)
        F0, _ = construct_matrices(DecompositionMethodSolver(2.0, 3.0), dm_ToyBasis(3), evaluate_points(DecompositionMethodSolver(2.0, 3.0), rect, k), k)
        cG = (0.7 * (1.0 - 0.5) * 1.0 + 1.0 * (0.7 - 0.2) * 4.0) / 2
        @test F ≈ F0
        @test G ≈ cG .* (v * v')
    end

    @testset "F grows like k^2, G does not depend on k for this basis" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        F1, G1 = construct_matrices(solver, dm_ToyBasis(3), evaluate_points(solver, rect, 7.0), 7.0)
        F2, G2 = construct_matrices(solver, dm_ToyBasis(3), evaluate_points(solver, rect, 14.0), 14.0)
        @test F2 ≈ 4 .* F1
        @test G2 ≈ G1
    end

    @testset "the symmetry count scales both matrices" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        pts = evaluate_points(solver, rect, k)
        F1, G1 = construct_matrices(solver, dm_ToyBasis(3), pts, k)
        two = dm_ToyBasis(3, [XAxisReflection()])                           # nsym = 2
        F2, G2 = construct_matrices(solver, two, pts, k)
        @test F2 ≈ 2 .* F1
        @test G2 ≈ 2 .* G1
        four = dm_ToyBasis(3, [XAxisReflection(), YAxisReflection(), XYAxisReflection()])   # nsym = 4
        F4, G4 = construct_matrices(solver, four, pts, k)
        @test F4 ≈ 4 .* F1
        @test G4 ≈ 4 .* G1
    end

    @testset "the basis dimension sets the matrix size" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        pts = evaluate_points(solver, rect, k)
        F, G = construct_matrices(solver, dm_ToyBasis(1), pts, k)
        @test size(F) == size(G) == (1, 1)
        F, G = construct_matrices(solver, dm_ToyBasis(5), pts, k)
        @test size(F) == size(G) == (5, 5)
    end

    @testset "multithreaded keyword" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        pts = evaluate_points(solver, rect, k)
        Fs, Gs = construct_matrices(solver, dm_ToyBasis(3), pts, k; multithreaded=false)
        Ft, Gt = construct_matrices(solver, dm_ToyBasis(3), pts, k; multithreaded=true)
        @test Fs ≈ Ft && Gs ≈ Gt
    end
end

@testset "DecompositionMethodSolver construct_matrices: RealPlaneWaves on the rectangle" begin
    rect = dm_rect()
    k = 6.0
    solver = DecompositionMethodSolver(2.0, 3.0)
    pts = evaluate_points(solver, rect, k)
    basis = RealPlaneWaves(8; sym_x=-1, sym_y=-1)                           # odd-odd, D2: three registered reflections
    F, G = construct_matrices(solver, basis, pts, k)
    nsym = length(basis.symmetries) + 1
    B, dX, dY = basis_and_gradient_matrices(basis, k, pts.xy)
    Bn = Diagonal(first.(pts.normal)) * dX + Diagonal(last.(pts.normal)) * dY

    @testset "shapes and symmetry" begin
        @test nsym == 4
        @test size(F) == size(G) == (basis.dim, basis.dim)
        @test issymmetric(F) && issymmetric(G)
        @test eltype(F) == Float64
    end

    @testset "dense formulas" begin
        @test F ≈ nsym .* (B' * Diagonal(pts.ds) * B) rtol = 1e-10
        @test G ≈ nsym .* (Bn' * Diagonal(pts.w_dm) * Bn) rtol = 1e-10
    end

    @testset "quadratic forms" begin
        x = [0.3 * (-1)^j * j for j in 1:basis.dim]
        @test x' * F * x ≈ nsym * sum(pts.ds .* (B * x) .^ 2) rtol = 1e-10
        @test x' * G * x ≈ nsym * sum(pts.w_dm .* (Bn * x) .^ 2) rtol = 1e-10
    end

    @testset "positive semi-definite for a Rellich origin inside the domain" begin
        scale_F = maximum(abs, F)
        scale_G = maximum(abs, G)
        @test minimum(eigvals(Symmetric(F))) > -1e-10 * scale_F
        @test minimum(eigvals(Symmetric(G))) > -1e-10 * scale_G
    end

    @testset "threaded equals serial" begin
        Fs, Gs = construct_matrices(solver, basis, pts, k; multithreaded=false)
        @test Fs ≈ F && Gs ≈ G
    end

    @testset "a basis without symmetries has nsym = 1" begin
        plain = RealPlaneWaves(8)
        Fp, Gp = construct_matrices(solver, plain, pts, k)
        Bp = basis_matrix(plain, k, pts.xy)
        @test isnothing(plain.symmetries) || length(plain.symmetries) == 0
        @test Fp ≈ Bp' * Diagonal(pts.ds) * Bp rtol = 1e-10
    end
end

@testset "DecompositionMethodSolver solve from F and G" begin
    rect = dm_rect()
    k = 7.0
    solver = DecompositionMethodSolver(2.0, 3.0)
    pts = evaluate_points(solver, rect, k)

    @testset "one-function basis: the tension is exactly F / G" begin
        # generalized_eigvals(F, G) returns the eigenvalues mu of G x = mu F x (F is the whitened matrix), so for a single basis
        # function mu = G / F and t = 1 / mu = F / G, the boundary norm over the Rellich-weighted normal-derivative norm.
        # (The header comment and docstring of decompositionmethod.jl state F x = lambda G x, t = G / F: inverted w.r.t. the code.)
        F, G = construct_matrices(solver, dm_ToyBasis(1), pts, k)
        t = solve(solver, F, G)
        @test t ≈ F[1, 1] / G[1, 1]
        cF = k^2 * ((2.4^3 - 1) / 6 + (2.4^3 - 1.4^3) / 3)
        cG = (0.7 * 1.0 + 1.0 * 0.7 * 4.0) / 2
        @test t ≈ cF / cG
        @test solve(solver, dm_ToyBasis(1), pts, k) ≈ t                     # the basis method assembles the same pencil
    end

    @testset "the tension is linear in F and inverse-linear in G" begin
        F, G = construct_matrices(solver, dm_ToyBasis(1), pts, k)
        @test solve(solver, 4 .* F, G) ≈ 4 * solve(solver, F, G)
        @test solve(solver, F, 4 .* G) ≈ solve(solver, F, G) / 4
    end
end
