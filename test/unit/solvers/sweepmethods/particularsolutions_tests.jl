# Unit tests for
#   src/solvers/sweepmethods/particularsolutions.jl   (ParticularSolutionsMethodSolver: evaluate_points, construct_matrices,
#                                                      _numerical_rank_from_F and the tension on tiny systems)
# The particular solutions method compares the boundary matrix B (rows: fundamental boundary nodes) with the interior matrix
# B_int (rows: random interior points), both of the basis at k, and takes t = min_c ||B c|| / ||B_int c||. Checks, all without
# a search for eigenvalues:
#   * evaluate_points: the fields that are populated (xy, normal, s, ds, xy_int), the cumulative arc length s, the interior
#     point count max(min_int_pts, round(k L int_scale / 2 pi)) with L the fundamental boundary length, interior points inside
#     the fundamental domain, the dependence on the scaling factors, and the Float32 type;
#   * construct_matrices on a toy basis f_i = i k (x + 2y) on hand-chosen points (closed form), the plain basis matrices of
#     RealPlaneWaves, no scaling by the symmetry count, threaded = serial;
#   * _numerical_rank_from_F on diagonal matrices of known pivoted-QR rank;
#   * the tension of a one-function basis, which is ||B|| / ||B_int|| exactly, its invariances (independent of k for the toy
#     basis, invariant under duplicating all points), the agreement of the full GSVD, the rank-reduced path and solve_vect, and
#     the degenerate-rank fallbacks (Inf).
# Helper definitions use the prefix `pm_`.

using LinearAlgebra: norm, qr, ColumnNorm, Diagonal
using StaticArrays: SVector
using Random: seed!
using QuantumBilliards: solve_full, solve_with_rank_reduction, solve_vect, _determine_bp_sizes, _numerical_rank_from_F

# Toy basis f_i(x, y) = i k (x + 2y): no wave equation, but exact algebra for B and B_int.
struct pm_ToyBasis{S} <: AbsBasis
    dim::Int
    symmetries::S
end
pm_ToyBasis(dim::Int) = pm_ToyBasis(dim, nothing)
QuantumBilliards.basis_fun(::pm_ToyBasis, indices::AbstractArray, k::Real, pts::AbstractArray; multithreaded::Bool=true) =
    [i * k * (p[1] + 2p[2]) for p in pts, i in indices]

# Hand-chosen points: three boundary and four interior nodes of the first-quadrant rectangle.
pm_boundary() = SVector{2,Float64}[SVector(1.0, 0.1), SVector(1.0, 0.4), SVector(0.5, 0.7)]
pm_interior() = SVector{2,Float64}[SVector(0.2, 0.2), SVector(0.4, 0.3), SVector(0.6, 0.5), SVector(0.8, 0.1)]
pm_points() = BoundaryPoints(pm_boundary(); xy_int=pm_interior())

# The 2 x 1.4 rectangle, whose fundamental boundary is the right edge x = 1 (length 0.7) followed by the top edge y = 0.7 (length 1.0).
pm_rect() = RectangleBilliard(1.0, 0.7)

@testset "ParticularSolutionsMethodSolver evaluate_points: rectangle" begin
    rect = pm_rect()
    solver = ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=10)
    k = 20.0
    seed!(1)
    pts = evaluate_points(solver, rect, k)
    Ns = _determine_bp_sizes(get_boundary_curves(rect), solver.pts_scaling_factor, k)
    right, top = 1:Ns[1], Ns[1]+1:sum(Ns)

    @testset "returned fields" begin
        @test pts isa BoundaryPoints{Float64}
        N = length(pts)
        @test N == sum(Ns) && N > 40
        @test length(pts.xy) == length(pts.normal) == length(pts.s) == length(pts.ds) == N
        @test !isempty(pts.xy_int)
        @test isempty(pts.w_dm)                                             # Rellich weights belong to the decomposition method
        @test isempty(pts.w_vs)                                             # Vergini-Saraceno weights are not set
    end

    @testset "boundary nodes, normals and quadrature weights" begin
        @test all(p -> p[1] == 1.0 && 0.0 < p[2] < 0.7, pts.xy[right])
        @test all(p -> p[2] == 0.7 && 0.0 < p[1] < 1.0, pts.xy[top])
        @test all(n -> n == SVector(1.0, 0.0), pts.normal[right])
        @test all(n -> n == SVector(0.0, 1.0), pts.normal[top])
        @test all(>(0), pts.ds)
        @test sum(pts.ds[right]) ≈ 0.7
        @test sum(pts.ds[top]) ≈ 1.0
    end

    @testset "s is the cumulative arc length along the fundamental boundary" begin
        @test issorted(pts.s; lt=<=)                                        # strictly increasing over both curves
        @test all(s -> 0.0 < s < 0.7, pts.s[right])
        @test all(s -> 0.7 < s < 1.7, pts.s[top])
        @test pts.s[right] ≈ [p[2] for p in pts.xy[right]]                  # the right edge runs upward from the origin
        @test pts.s[top] ≈ [0.7 + (1.0 - p[1]) for p in pts.xy[top]]        # the top edge continues from x = 1 down to x = 0
    end

    @testset "interior points lie in the fundamental domain" begin
        @test all(p -> 0.0 < p[1] < 1.0 && 0.0 < p[2] < 0.7, pts.xy_int)
        @test all(p -> is_inside(rect, p), pts.xy_int)
        @test eltype(pts.xy_int) == SVector{2,Float64}
    end

    @testset "interior point count max(min_int_pts, round(k L int_scale / 2 pi))" begin
        @test length(pts.xy_int) == max(10, round(Int, k * 1.7 * 2.0 / (2π))) == 11
        @test length(evaluate_points(solver, rect, 40.0).xy_int) == round(Int, 40.0 * 1.7 * 2.0 / (2π))
        dense = ParticularSolutionsMethodSolver(2.0, 3.0, 8.0; min_int_pts=10)
        @test length(evaluate_points(dense, rect, 40.0).xy_int) == round(Int, 40.0 * 1.7 * 8.0 / (2π))
        @test length(evaluate_points(solver, rect, 1.0).xy_int) == 10       # the floor min_int_pts
        defaults = ParticularSolutionsMethodSolver(2.0, 3.0, 2.0)
        @test length(evaluate_points(defaults, rect, 20.0).xy_int) == 500   # the default floor
    end

    @testset "the interior points are random, the boundary nodes are not" begin
        seed!(2)
        again = evaluate_points(solver, rect, k)
        @test again.xy == pts.xy && again.s == pts.s && again.ds == pts.ds
        @test again.xy_int != pts.xy_int
        seed!(3); first_draw = evaluate_points(solver, rect, k).xy_int
        seed!(3); second_draw = evaluate_points(solver, rect, k).xy_int
        @test first_draw == second_draw                                     # reproducible through the global RNG
    end
end

@testset "ParticularSolutionsMethodSolver evaluate_points: curved boundary" begin
    circle = CircleBilliard(1.0)
    solver = ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=10)
    pts = evaluate_points(solver, circle, 10.0)
    @test all(p -> norm(p) ≈ 1.0, pts.xy)
    @test pts.normal ≈ pts.xy                                               # outward unit normal of the unit circle
    @test sum(pts.ds) ≈ π / 2                                               # the quarter arc
    @test pts.s[end] < π / 2 && pts.s[1] > 0.0
    @test issorted(pts.s; lt=<=)
    @test all(p -> p[1] > 0 && p[2] > 0 && norm(p) < 1.0, pts.xy_int)       # inside the quarter disk
    @test length(pts.xy_int) == max(10, round(Int, 10.0 * (π / 2) * 2.0 / (2π))) == 10
end

@testset "ParticularSolutionsMethodSolver evaluate_points: sampling rules" begin
    rect = pm_rect()
    curves = get_boundary_curves(rect)

    @testset "the number of boundary points follows the scaling factors" begin
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=10)
        small = length(evaluate_points(solver, rect, 20.0))
        @test small == sum(_determine_bp_sizes(curves, [3.0, 3.0], 20.0))
        large = length(evaluate_points(solver, rect, 40.0))
        @test abs(large - 2small) <= 2                                      # linear in k up to rounding
        double = ParticularSolutionsMethodSolver(2.0, 6.0, 2.0; min_int_pts=10)
        @test abs(length(evaluate_points(double, rect, 20.0)) - 2small) <= 2
    end

    @testset "per-curve scaling factors" begin
        solver = ParticularSolutionsMethodSolver(2.0, [3.0, 6.0], 2.0; min_int_pts=10)
        pts = evaluate_points(solver, rect, 20.0)
        Ns = _determine_bp_sizes(curves, [3.0, 6.0], 20.0)
        @test length(pts) == sum(Ns)
        @test sum(pts.ds) ≈ 1.7
    end

    @testset "at least 20 points per curve" begin
        pts = evaluate_points(ParticularSolutionsMethodSolver(2.0, 0.01, 2.0; min_int_pts=10), rect, 1.0)
        @test length(pts) == 40
    end

    @testset "sampler choice changes the nodes but not the boundary integrals" begin
        samplers = AbsSampler[LinearNodes(), LinearNodes()]
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 2.0, samplers; min_int_pts=10)
        pts = evaluate_points(solver, rect, 20.0)
        @test sum(pts.ds) ≈ 1.7
        @test length(pts.xy_int) == 11
    end

    @testset "Float32 solver" begin
        solver = ParticularSolutionsMethodSolver(2f0, 3f0, 2f0; min_int_pts=10)
        pts = evaluate_points(solver, rect, 20f0)
        @test pts isa BoundaryPoints{Float32}
        @test eltype(pts.xy_int) == SVector{2,Float32}
        @test sum(pts.ds) ≈ 1.7f0
    end

    @testset "argument validation" begin
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=10)
        @test_throws MethodError evaluate_points(solver, 1.0, 20.0)         # not a billiard
        @test_throws MethodError evaluate_points(1.0, rect, 20.0)           # not a solver
        @test_throws MethodError construct_matrices(solver, 1.0, pm_points(), 3.0)   # not a basis
    end
end

@testset "ParticularSolutionsMethodSolver construct_matrices" begin
    solver = ParticularSolutionsMethodSolver(2.0, 3.0, 2.0)
    pts = pm_points()
    k = 3.0

    @testset "toy basis in closed form" begin
        B, C = construct_matrices(solver, pm_ToyBasis(3), pts, k)
        v = [1.0, 2.0, 3.0]                                                 # f_i has the amplitude i
        g_boundary = [k * (p[1] + 2p[2]) for p in pm_boundary()]
        g_interior = [k * (p[1] + 2p[2]) for p in pm_interior()]
        @test size(B) == (3, 3) && size(C) == (4, 3)
        @test B ≈ g_boundary * v'
        @test C ≈ g_interior * v'
    end

    @testset "the rows follow the points, the columns the basis functions" begin
        for dim in (1, 2, 5)
            B, C = construct_matrices(solver, pm_ToyBasis(dim), pts, k)
            @test size(B) == (length(pts.xy), dim)
            @test size(C) == (length(pts.xy_int), dim)
        end
    end

    @testset "no scaling by the symmetry count" begin
        B1, C1 = construct_matrices(solver, pm_ToyBasis(3), pts, k)
        B4, C4 = construct_matrices(solver, pm_ToyBasis(3, [XAxisReflection(), YAxisReflection(), XYAxisReflection()]), pts, k)
        @test B4 == B1 && C4 == C1                                          # unlike F and G of the decomposition method
    end

    @testset "RealPlaneWaves: the plain basis matrices at the two point sets" begin
        basis = RealPlaneWaves(6; sym_x=-1, sym_y=-1)
        rect = pm_rect()
        seed!(4)
        full = evaluate_points(ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=30), rect, 4.0)
        B, C = construct_matrices(solver, basis, full, 4.0)
        @test B == basis_matrix(basis, 4.0, full.xy)
        @test C == basis_matrix(basis, 4.0, full.xy_int)
        @test size(B) == (length(full), 6) && size(C) == (length(full.xy_int), 6)
        @test eltype(B) == Float64
    end

    @testset "threaded equals serial" begin
        basis = RealPlaneWaves(6)
        Bs, Cs = construct_matrices(solver, basis, pts, k; multithreaded=false)
        Bt, Ct = construct_matrices(solver, basis, pts, k; multithreaded=true)
        @test Bs ≈ Bt && Cs ≈ Ct
    end
end

@testset "_numerical_rank_from_F" begin
    @testset "diagonal matrices of known pivoted-QR rank" begin
        F = qr(Matrix(Diagonal([1.0, 1e-3, 1e-8])), ColumnNorm())
        @test _numerical_rank_from_F(F, 1e-12) == 3
        @test _numerical_rank_from_F(F, 1e-5) == 2
        @test _numerical_rank_from_F(F, 1e-2) == 1
        @test _numerical_rank_from_F(F, 10.0) == 0                          # the tolerance is relative to |R11|
    end

    @testset "the rank is read from the last diagonal entry above the threshold" begin
        F = qr(Matrix(Diagonal([4.0, 2.0, 1.0, 0.5])), ColumnNorm())
        @test _numerical_rank_from_F(F, 0.1) == 4                           # 0.5 > 0.1 * 4
        @test _numerical_rank_from_F(F, 0.2) == 3                           # 0.5 < 0.8 <= 1.0
        @test _numerical_rank_from_F(F, 0.4) == 2                           # 1.0 < 1.6 <= 2.0
        @test _numerical_rank_from_F(F, 0.6) == 1                           # 2.0 < 2.4 <= 4.0
    end

    @testset "zero and empty matrices" begin
        @test _numerical_rank_from_F(qr(zeros(3, 3), ColumnNorm()), 1e-10) == 0
        @test _numerical_rank_from_F(qr(zeros(0, 0), ColumnNorm()), 1e-10) == 0
    end

    @testset "rank-one matrix has one well-determined direction" begin
        F = qr([1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 9.0], ColumnNorm())
        @test _numerical_rank_from_F(F, 1e-10) == 1
    end
end

@testset "ParticularSolutionsMethodSolver tension on tiny systems" begin
    solver = ParticularSolutionsMethodSolver(2.0, 3.0, 2.0)
    pts = pm_points()
    k = 3.0

    @testset "one-function basis: t = ||B|| / ||B_int||" begin
        basis = pm_ToyBasis(1)
        B, C = construct_matrices(solver, basis, pts, k)
        expected = norm(B) / norm(C)
        @test solve_full(solver, basis, pts, k) ≈ expected
        @test solve_with_rank_reduction(solver, basis, pts, k) ≈ expected
        @test solve(solver, basis, pts, k) ≈ expected                       # rank reduction is the default
        @test solve(solver, basis, pts, k; use_rank_reduction=false) ≈ expected
        # closed form of the toy norms
        gb = sum(abs2, (p[1] + 2p[2] for p in pm_boundary()))
        gi = sum(abs2, (p[1] + 2p[2] for p in pm_interior()))
        @test expected ≈ sqrt(gb / gi)
    end

    @testset "the toy tension does not depend on k or on the amplitude of the functions" begin
        t3 = solve_full(solver, pm_ToyBasis(1), pts, 3.0)
        @test solve_full(solver, pm_ToyBasis(1), pts, 11.0) ≈ t3
        @test solve_with_rank_reduction(solver, pm_ToyBasis(1), pts, 11.0) ≈ t3
        # more functions of the same shape i k (x + 2y) add no new direction
        @test solve_with_rank_reduction(solver, pm_ToyBasis(3), pts, 3.0) ≈ t3
    end

    @testset "the tension is invariant under duplicating all boundary and interior points" begin
        basis = RealPlaneWaves(6; sym_x=-1, sym_y=-1)
        seed!(5)
        base = evaluate_points(ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=200), pm_rect(), 5.0)
        twice = BoundaryPoints(vcat(base.xy, base.xy); xy_int=vcat(base.xy_int, base.xy_int))
        @test solve_full(solver, basis, twice, 5.0) ≈ solve_full(solver, basis, base, 5.0) rtol = 1e-8
        @test solve_with_rank_reduction(solver, basis, twice, 5.0) ≈ solve_with_rank_reduction(solver, basis, base, 5.0) rtol = 1e-8
    end

    @testset "full GSVD and rank-reduced paths agree, and the tension is non-negative" begin
        basis = RealPlaneWaves(6; sym_x=-1, sym_y=-1)
        seed!(6)
        pts6 = evaluate_points(ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=200), pm_rect(), 5.0)
        tf = solve_full(solver, basis, pts6, 5.0)
        tr = solve_with_rank_reduction(solver, basis, pts6, 5.0)
        @test tf >= 0 && tr >= 0
        @test tr ≈ tf rtol = 1e-6
    end

    @testset "solve_vect returns the minimizing coefficients" begin
        basis = pm_ToyBasis(1)
        B, C = construct_matrices(solver, basis, pts, k)
        t, x = solve_vect(solver, basis, pts, k)
        @test t ≈ solve_full(solver, basis, pts, k)
        @test length(x) == 1
        @test norm(C * x) ≈ 1.0                                             # the coefficients are normalized on the interior
        @test norm(B * x) ≈ t                                               # and the boundary norm is the tension

        rpw = RealPlaneWaves(6; sym_x=-1, sym_y=-1)
        seed!(7)
        pts6 = evaluate_points(ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=200), pm_rect(), 5.0)
        B6, C6 = construct_matrices(solver, rpw, pts6, 5.0)
        t6, x6 = solve_vect(solver, rpw, pts6, 5.0)
        @test length(x6) == 6
        @test t6 ≈ solve_with_rank_reduction(solver, rpw, pts6, 5.0) rtol = 1e-8
        @test norm(B6 * x6) / norm(C6 * x6) ≈ t6 rtol = 1e-8
    end

    @testset "degenerate numerical rank gives Inf" begin
        basis = pm_ToyBasis(1)
        @test solve_with_rank_reduction(solver, basis, pts, k; tol=10.0) == Inf
        @test solve(solver, basis, pts, k; tol=10.0) == Inf
        t, x = solve_vect(solver, basis, pts, k; tol=10.0)
        @test t == Inf
        @test x == zeros(1)
    end

    @testset "threaded equals serial" begin
        basis = RealPlaneWaves(6)
        seed!(8)
        pts6 = evaluate_points(ParticularSolutionsMethodSolver(2.0, 3.0, 2.0; min_int_pts=200), pm_rect(), 5.0)
        @test solve_full(solver, basis, pts6, 5.0; multithreaded=false) ≈ solve_full(solver, basis, pts6, 5.0; multithreaded=true)
    end
end
