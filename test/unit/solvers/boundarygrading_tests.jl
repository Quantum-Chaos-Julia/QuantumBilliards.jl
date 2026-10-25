# Unit tests for
#   src/solvers/sweepmethods/boundarygrading.jl   (BoundaryGrading, SmoothPeriodicGrading, CornerGrading,
#                                                  GlobalCornerGrading, _bim_numeric_type)
#   src/solvers/sweepmethods/dlp.jl, cfie.jl       (the `evaluate_points` grading branches and
#                                                  _is_nontrivial_dlp_grading)
# The grading maps themselves (kress_graded_nodes_data, multi_kress_graded_nodes_data) belong to BilliardGeometry;
# here only their effect on the BoundaryPoints returned by `evaluate_points` is checked: monotone, periodic
# parametrization, quadrature identities, clustering at corners, and agreement between the DLP and CFIE kernels.
# No eigen solves and no matrix assembly.

using QuantumBilliards: _bim_numeric_type, _is_nontrivial_dlp_grading

# Number of cyclic local minima of `v` below `thr` (counts the clusters of a graded node distribution).
function bg_count_minima(v::AbstractVector, thr)
    n = length(v)
    return count(i -> v[i] < thr && v[i] <= v[mod1(i - 1, n)] && v[i] < v[mod1(i + 1, n)], 1:n)
end

@testset "BoundaryGrading type hierarchy" begin
    @test isabstracttype(BoundaryGrading)
    @test SmoothPeriodicGrading <: BoundaryGrading
    @test CornerGrading <: BoundaryGrading
    @test GlobalCornerGrading <: BoundaryGrading
    @test isconcretetype(SmoothPeriodicGrading)
    @test isconcretetype(CornerGrading{Float64})
    @test isconcretetype(GlobalCornerGrading{Float64})
    @test SmoothPeriodicGrading() isa SmoothPeriodicGrading
    @test fieldcount(SmoothPeriodicGrading) == 0
end

@testset "CornerGrading keyword constructor" begin
    g = CornerGrading()
    @test g isa CornerGrading{Float64}
    @test g.kressq == 2
    @test g.min_t_spacing == 1e-12

    g = CornerGrading(kressq=4, min_t_spacing=1e-9)
    @test g.kressq == 4
    @test g.min_t_spacing == 1e-9

    @test CornerGrading(kressq=3).min_t_spacing == 1e-12               # keywords are independent
    @test CornerGrading(min_t_spacing=1e-6).kressq == 2
    @test CornerGrading(min_t_spacing=1f-6) isa CornerGrading{Float32}
    @test_throws TypeError CornerGrading(kressq=2.5)                  # kressq must be an Int
    @test_throws MethodError CornerGrading(order=2)                     # unknown keyword
end

@testset "GlobalCornerGrading keyword constructor" begin
    g = GlobalCornerGrading()
    @test g isa GlobalCornerGrading{Float64}
    @test g.kressq == 2
    @test g.min_t_spacing == 1e-12

    g = GlobalCornerGrading(kressq=4, min_t_spacing=1f-6)
    @test g isa GlobalCornerGrading{Float32}
    @test g.kressq == 4
    @test g.min_t_spacing == 1f-6

    @test GlobalCornerGrading(kressq=3).kressq == 3
    @test_throws TypeError GlobalCornerGrading(kressq=2.5)
    @test_throws MethodError GlobalCornerGrading(order=2)
end

@testset "_bim_numeric_type" begin
    circle = CircleBilliard(1.0)
    rect = RectangleBilliard(1.0, 0.5)

    @testset "sweep kernels carry the scaling-factor type" begin
        @test _bim_numeric_type(DoubleLayerPotentialSolver(10.0, circle)) === Float64
        @test _bim_numeric_type(DoubleLayerPotentialSolver(10f0, circle)) === Float32
        @test _bim_numeric_type(CombinedFieldIntegralEquationSolver(10.0, rect)) === Float64
        @test _bim_numeric_type(CombinedFieldIntegralEquationSolver(10f0, rect)) === Float32
        @test @inferred(_bim_numeric_type(CombinedFieldIntegralEquationSolver(10.0, circle))) === Float64
    end

    @testset "accelerated wrappers inherit the kernel type" begin
        kernel64 = CombinedFieldIntegralEquationSolver(10.0, circle)
        kernel32 = DoubleLayerPotentialSolver(10f0, circle)
        @test _bim_numeric_type(BeynSolver(kernel64)) === Float64
        @test _bim_numeric_type(BeynSolver(kernel32)) === Float32
        @test _bim_numeric_type(ExpandedBIMSolver(kernel64)) === Float64
        @test _bim_numeric_type(ExpandedBIMSolver(kernel32)) === Float32
        @test _bim_numeric_type(CORKSolver(kernel64)) === Float64
        @test_throws ArgumentError CORKSolver(kernel32)                 # CORK is Float64 only
    end

    @testset "no method for non-solver arguments" begin
        @test_throws MethodError _bim_numeric_type(1.0)
        @test_throws MethodError _bim_numeric_type(SmoothPeriodicGrading())
    end
end

@testset "evaluate_points with SmoothPeriodicGrading" begin
    R = 1.0
    circle = CircleBilliard(R)
    L = 2π*R

    @testset "node count follows N = max(min_pts, round(k L scale / 2π)), rounded up to even" begin
        scale = 10.0
        solver = CombinedFieldIntegralEquationSolver(scale, circle; min_pts=64)
        @test length(evaluate_points(solver, circle, 5.0)) == 64          # k L scale / 2π = 50 < min_pts
        @test length(evaluate_points(solver, circle, 20.0)) == 200        # = k R scale
        @test length(evaluate_points(solver, circle, 50.0)) == 500
        @test length(evaluate_points(solver, circle, 20.1)) == 202        # 201 rounded up to even
        @test iseven(length(evaluate_points(solver, circle, 33.3)))
    end

    solver = CombinedFieldIntegralEquationSolver(10.0, circle; min_pts=64, grading=SmoothPeriodicGrading())
    pts = evaluate_points(solver, circle, 5.0)
    N = length(pts)
    h = 2π/N

    @testset "uniform midpoint periodic mesh" begin
        @test pts isa BoundaryPoints{Float64}
        @test pts.ts ≈ [2π*(j - 0.5)/N for j in 1:N]
        @test pts.tphys == pts.ts                                        # no reparametrization
        @test all(==(1.0), pts.ws_der)
        @test all(≈(h), pts.ws)
        @test all(0 .< pts.ts .< 2π)
    end

    @testset "geometry of the unit circle" begin
        @test all(isapprox.(hypot.(getindex.(pts.xy, 1), getindex.(pts.xy, 2)), R; atol=1e-12))
        @test sum(pts.ds) ≈ L
        @test all(isapprox.(pts.ds, L/N; atol=1e-12))
        @test all(isapprox.(hypot.(getindex.(pts.tangent, 1), getindex.(pts.tangent, 2)), L/(2π); atol=1e-12))
        @test issorted(pts.s) && all(0 .<= pts.s .<= L)
    end

    @testset "not a nontrivial grading" begin
        @test !_is_nontrivial_dlp_grading(pts)
    end

    @testset "Float32 solver gives Float32 points" begin
        solver32 = CombinedFieldIntegralEquationSolver(10f0, circle; min_pts=64)
        pts32 = evaluate_points(solver32, circle, 5f0)
        @test pts32 isa BoundaryPoints{Float32}
        @test eltype(pts32.tphys) == Float32
    end
end

@testset "evaluate_points with GlobalCornerGrading on a rectangle" begin
    rect = RectangleBilliard(1.0, 0.5)
    L = sum(c.length for c in full_boundary(rect))
    solver(q) = CombinedFieldIntegralEquationSolver(10.0, rect; min_pts=256, grading=GlobalCornerGrading(kressq=q))
    pts2 = evaluate_points(solver(2), rect, 5.0)
    pts3 = evaluate_points(solver(3), rect, 5.0)
    N = length(pts3)
    h = 2π/N

    @testset "monotone periodic parametrization" begin
        for pts in (pts2, pts3)
            @test length(pts) == 256
            @test all(diff(pts.ts) .> 0)
            @test all(diff(pts.tphys) .> 0)                                # monotone map t(σ)
            @test pts.tphys[1] > 0 && pts.tphys[end] < 2π
            @test pts.tphys[1] + 2π > pts.tphys[end]                       # t(σ + 2π) = t(σ) + 2π stays ordered
            @test all(pts.ws_der .> 0)                                    # dt/dσ > 0
            @test all(≈(h), pts.ws)                                        # uniform weights in σ
        end
    end

    @testset "quadrature identities" begin
        # ∫ dt/dσ dσ = 2π and ∮ ds = L; the periodic mesh integrates the graded map spectrally,
        # the residual is the O(h^q)-type error of a finite-regularity map
        @test isapprox(sum(pts3.ws .* pts3.ws_der), 2π; rtol=1e-4)
        @test isapprox(sum(pts3.ds), L; rtol=1e-4)
        @test isapprox(sum(pts2.ds), L; rtol=1e-3)
        # constant arclength speed |γ'(t)| = L/(2π) along straight edges, so |dγ/dσ| = (L/2π) dt/dσ
        speed = hypot.(getindex.(pts3.tangent, 1), getindex.(pts3.tangent, 2))
        @test all(isapprox.(speed ./ pts3.ws_der, L/(2π); rtol=1e-10))
    end

    @testset "nodes cluster at the four corners" begin
        @test _is_nontrivial_dlp_grading(pts3)
        @test bg_count_minima(pts3.ws_der, 0.1) == 4
        @test bg_count_minima(pts2.ws_der, 0.1) == 4
        @test minimum(pts3.ws_der) < 0.05                                  # strong clustering
        @test minimum(pts3.ws_der) < minimum(pts2.ws_der)                  # higher kressq clusters more
        @test maximum(pts3.ws_der) > 1                                     # compensated away from the corners
    end

    @testset "straight edges have collinear first and second derivatives" begin
        cross = [pts3.tangent[i][1]*pts3.tangent_2[i][2] - pts3.tangent[i][2]*pts3.tangent_2[i][1] for i in 1:N]
        scale = maximum(hypot.(getindex.(pts3.tangent, 1), getindex.(pts3.tangent, 2)) .*
                        hypot.(getindex.(pts3.tangent_2, 1), getindex.(pts3.tangent_2, 2)))
        @test maximum(abs, cross) <= 1e-10*scale
    end

    @testset "unsatisfiable min_t_spacing is reported" begin
        # the Kress order is reduced until the nodes are min_t_spacing apart; q = 1 is the identity map,
        # so a spacing of 1 (larger than the uniform spacing) cannot be met and the construction errors
        bad = CombinedFieldIntegralEquationSolver(10.0, rect; min_pts=64, grading=GlobalCornerGrading(min_t_spacing=1.0))
        @test_throws ErrorException evaluate_points(bad, rect, 5.0)
    end

    @testset "Float32" begin
        solver32 = CombinedFieldIntegralEquationSolver(10f0, rect; min_pts=64, grading=GlobalCornerGrading())
        pts32 = evaluate_points(solver32, rect, 5f0)
        @test pts32 isa BoundaryPoints{Float32}
        @test all(diff(pts32.tphys) .> 0)
    end
end

@testset "GlobalCornerGrading without corners reduces to SmoothPeriodicGrading" begin
    circle = CircleBilliard(1.0)
    smooth = CombinedFieldIntegralEquationSolver(10.0, circle; min_pts=64, grading=SmoothPeriodicGrading())
    graded = CombinedFieldIntegralEquationSolver(10.0, circle; min_pts=64, grading=GlobalCornerGrading())
    ps = evaluate_points(smooth, circle, 5.0)
    pg = evaluate_points(graded, circle, 5.0)
    @test pg.xy == ps.xy
    @test pg.tangent == ps.tangent
    @test pg.tphys == ps.tphys
    @test all(==(1.0), pg.ws_der)
    @test !_is_nontrivial_dlp_grading(pg)

    polar = PolarBilliard([0.0, 0.0, 0.0, 0.05])                         # smooth single curve, no corners
    pp = evaluate_points(CombinedFieldIntegralEquationSolver(10.0, polar; min_pts=64, grading=GlobalCornerGrading()), polar, 5.0)
    @test all(==(1.0), pp.ws_der)
end

@testset "evaluate_points with CornerGrading" begin
    polar = PolarBilliard([0.0, 0.0, 0.0, 0.05])
    crv = only(full_boundary(polar))
    solver(q) = CombinedFieldIntegralEquationSolver(10.0, polar; min_pts=256, grading=CornerGrading(kressq=q))
    pts = evaluate_points(solver(3), polar, 5.0)
    N = length(pts)

    @testset "single closed curve, parametrized on [0, 1)" begin
        @test N == 256
        @test all(diff(pts.tphys) .> 0)
        @test 0 < pts.tphys[1] && pts.tphys[end] < 1
        @test all(pts.ws_der .> 0)
        @test _is_nontrivial_dlp_grading(pts)
    end

    @testset "quadrature identities" begin
        @test isapprox(sum(pts.ws .* pts.ws_der), 2π; rtol=1e-4)
        @test isapprox(sum(pts.ds), crv.length; rtol=1e-4)
    end

    @testset "clustering at the corner t = 0 ≡ 1" begin
        @test argmin(pts.ws_der) in (1, N)
        @test minimum(pts.ws_der) < 0.05
        @test minimum(pts.ws_der) < minimum(evaluate_points(solver(2), polar, 5.0).ws_der)
    end

    @testset "composite boundaries are rejected" begin
        rect = RectangleBilliard(1.0, 0.5)
        bad = CombinedFieldIntegralEquationSolver(10.0, rect; min_pts=64, grading=CornerGrading())
        @test_throws ErrorException evaluate_points(bad, rect, 5.0)
        circle = CircleBilliard(1.0)                                     # four arcs
        bad = CombinedFieldIntegralEquationSolver(10.0, circle; min_pts=64, grading=CornerGrading())
        @test_throws ErrorException evaluate_points(bad, circle, 5.0)
    end
end

@testset "DLP and CFIE share the boundary discretization" begin
    rect = RectangleBilliard(1.0, 0.5)
    circle = CircleBilliard(1.0)
    for_billiard(b, grading) = (
        evaluate_points(DoubleLayerPotentialSolver(10.0, b; min_pts=64, grading=grading), b, 5.0),
        evaluate_points(CombinedFieldIntegralEquationSolver(10.0, b; min_pts=64, grading=grading), b, 5.0))

    @testset "GlobalCornerGrading on a rectangle" begin
        pd, pc = for_billiard(rect, GlobalCornerGrading())
        @test pd.xy == pc.xy
        @test pd.tangent == pc.tangent
        @test pd.tangent_2 == pc.tangent_2
        @test pd.ws_der == pc.ws_der
        @test pd.ds == pc.ds
    end

    @testset "SmoothPeriodicGrading on a circle" begin
        pd, pc = for_billiard(circle, SmoothPeriodicGrading())
        @test pd.xy == pc.xy
        @test pd.tangent == pc.tangent
        @test pd.ws_der == pc.ws_der
    end
end
