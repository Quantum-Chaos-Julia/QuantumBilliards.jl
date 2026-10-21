# Unit tests for
#   src/solvers/boundarypoints.jl      (BoundaryPoints, length/isempty, boundary_matrix_size, boundary_s,
#                                       component_offsets, points_in_billiard, random_interior_points,
#                                       _determine_bp_sizes, get_boundary_curves_with_ignored, boundary_coords)
#   src/solvers/boundarygeomcache.jl   (flatten_boundary_components, flatten_boundary_ds)
# No eigen solves: every expected value is analytic (circle parametrization, rectangle edges) or hand-built.
# Sampler definitions themselves (LinearNodes, GaussLegendreNodes, FourierNodes) belong to BilliardGeometry;
# here they are only exercised through boundary_coords.

using LinearAlgebra: norm, dot
using Random
using StaticArrays: SVector
using QuantumBilliards: boundary_coords, get_boundary_curves_with_ignored, random_interior_points,
    _determine_bp_sizes, make_veech_right_triangle_and_basis

# Circle of radius R parametrized by the angle, sampled at midpoint nodes θ_j = 2π(j-1/2)/N.
# Returns the raw ingredients of the parametrized BoundaryPoints constructor.
function bpt_circle_data(::Type{T}, N::Int, R) where {T<:Real}
    h = T(2π)/N
    θ = T[(j - T(1)/2)*h for j in 1:N]
    Rt = T(R)
    xy = [SVector{2,T}(Rt*cos(t), Rt*sin(t)) for t in θ]
    tangent = [SVector{2,T}(-Rt*sin(t), Rt*cos(t)) for t in θ]
    tangent_2 = [SVector{2,T}(-Rt*cos(t), -Rt*sin(t)) for t in θ]
    z = SVector{2,T}(zero(T), zero(T))
    return (; xy, tangent, tangent_2, ts=θ, tphys=copy(θ), ws=fill(h, N), ws_der=ones(T, N),
            s=Rt .* θ, ds=fill(Rt*h, N), z, h, R=Rt)
end

@testset "BoundaryPoints keyword constructor" begin
    xy = [SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-1.0, 0.0)]

    @testset "defaults are empty or neutral" begin
        bp = BoundaryPoints(xy)
        @test bp isa BoundaryPoints{Float64}
        @test bp.xy == xy
        @test isempty(bp.normal) && isempty(bp.kappa) && isempty(bp.s) && isempty(bp.ds)
        @test isempty(bp.rdotn) && isempty(bp.w_vs) && isempty(bp.w_dm) && isempty(bp.xy_int)
        @test isempty(bp.w) && isempty(bp.w_n) && isempty(bp.curvature)
        @test isempty(bp.tangent) && isempty(bp.tangent_2)
        @test isempty(bp.ts) && isempty(bp.tphys) && isempty(bp.ws) && isempty(bp.ws_der)
        @test bp.shift_x == 0.0 && bp.shift_y == 0.0
        @test bp.compid == 1
        @test bp.is_periodic === true
        @test bp.xL == SVector(0.0, 0.0) && bp.xR == SVector(0.0, 0.0)
        @test bp.tL == SVector(0.0, 0.0) && bp.tR == SVector(0.0, 0.0)
    end

    @testset "supplied keywords are stored" begin
        normal = [SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-1.0, 0.0)]
        bp = BoundaryPoints(xy; normal=normal, s=[0.0, 1.0, 2.0], ds=fill(0.5, 3),
                            w_vs=fill(0.1, 3), w_dm=fill(0.2, 3), kappa=fill(1.0, 3),
                            shift_x=0.25, shift_y=-0.5, compid=3, is_periodic=false,
                            xL=SVector(1.0, 2.0), xR=SVector(3.0, 4.0))
        @test bp.normal == normal
        @test bp.s == [0.0, 1.0, 2.0]
        @test bp.ds == fill(0.5, 3)
        @test bp.w_vs == fill(0.1, 3)
        @test bp.w_dm == fill(0.2, 3)
        @test bp.kappa == fill(1.0, 3)
        @test bp.shift_x == 0.25 && bp.shift_y == -0.5
        @test bp.compid == 3
        @test bp.is_periodic === false
        @test bp.xL == SVector(1.0, 2.0) && bp.xR == SVector(3.0, 4.0)
    end

    @testset "type follows the point type" begin
        @test @inferred(BoundaryPoints(xy)) isa BoundaryPoints{Float64}
        xy32 = SVector{2,Float32}[SVector(1f0, 0f0), SVector(0f0, 1f0)]
        bp32 = BoundaryPoints(xy32)
        @test bp32 isa BoundaryPoints{Float32}
        @test bp32.shift_x isa Float32
        @test eltype(bp32.normal) == SVector{2,Float32}
        @test bp32.xL isa SVector{2,Float32}
    end
end

@testset "BoundaryPoints length validation" begin
    xy = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(2.0, 0.0)]
    bad = [1.0, 2.0]
    bad_vec = [SVector(1.0, 0.0)]

    @test_throws "Length of normal (1) must match xy (3)" BoundaryPoints(xy; normal=bad_vec)
    @test_throws "Length of s (2) must match xy (3)" BoundaryPoints(xy; s=bad)
    @test_throws "Length of ds (2) must match xy (3)" BoundaryPoints(xy; ds=bad)
    @test_throws "Length of rdotn" BoundaryPoints(xy; rdotn=bad)
    @test_throws "Length of w_vs" BoundaryPoints(xy; w_vs=bad)
    @test_throws "Length of w_dm" BoundaryPoints(xy; w_dm=bad)
    @test_throws "Length of w (2)" BoundaryPoints(xy; w=bad)
    @test_throws "Length of w_n" BoundaryPoints(xy; w_n=bad)
    @test_throws "Length of curvature" BoundaryPoints(xy; curvature=bad)
    @test_throws "Length of tangent (1)" BoundaryPoints(xy; tangent=bad_vec)
    @test_throws "Length of tangent_2" BoundaryPoints(xy; tangent_2=bad_vec)
    @test_throws "Length of ts" BoundaryPoints(xy; ts=bad)
    @test_throws "Length of tphys" BoundaryPoints(xy; tphys=bad)
    @test_throws "Length of ws (2)" BoundaryPoints(xy; ws=bad)
    @test_throws "Length of ws_der" BoundaryPoints(xy; ws_der=bad)
    @test_throws ErrorException BoundaryPoints(xy; normal=bad_vec)

    # empty fields and fields of matching length are accepted
    @test BoundaryPoints(xy; ds=zeros(3), w=zeros(3), ts=zeros(3)) isa BoundaryPoints{Float64}
    @test BoundaryPoints(SVector{2,Float64}[]) isa BoundaryPoints{Float64}
end

@testset "BoundaryPoints parametrized constructor" begin
    @testset "normal from tangent on a circle" begin
        d = bpt_circle_data(Float64, 32, 2.0)
        bp = BoundaryPoints(d.xy, d.tangent, d.tangent_2, d.ts, d.tphys, d.ws, d.ws_der,
                            d.s, d.ds, 1, true, d.z, d.z, d.z, d.z)
        @test bp isa BoundaryPoints{Float64}
        @test length(bp) == 32
        @test length(bp.normal) == 32
        # n = (t_y, -t_x)/|t| is the outward radial direction for a counter-clockwise circle
        @test all(isapprox.(bp.normal, [p ./ norm(p) for p in d.xy]; atol=1e-12))
        @test all(isapprox.(norm.(bp.normal), 1.0; atol=1e-12))
        @test all(abs(dot(bp.normal[i], d.tangent[i])) < 1e-12 for i in 1:32)
    end

    @testset "normal ignores the speed of the parametrization" begin
        d = bpt_circle_data(Float64, 16, 1.0)
        scaled = [7.5 * t for t in d.tangent]
        bp = BoundaryPoints(d.xy, scaled, d.tangent_2, d.ts, d.tphys, d.ws, d.ws_der,
                            d.s, d.ds, 1, true, d.z, d.z, d.z, d.z)
        @test all(isapprox.(norm.(bp.normal), 1.0; atol=1e-12))
        @test all(isapprox.(bp.normal, [p ./ norm(p) for p in d.xy]; atol=1e-12))
    end

    @testset "input data are stored unchanged" begin
        d = bpt_circle_data(Float64, 8, 1.5)
        xL = SVector(1.0, 2.0); xR = SVector(3.0, 4.0); tL = SVector(0.0, 1.0); tR = SVector(1.0, 0.0)
        bp = BoundaryPoints(d.xy, d.tangent, d.tangent_2, d.ts, d.tphys, d.ws, d.ws_der,
                            d.s, d.ds, 2, false, xL, xR, tL, tR)
        @test bp.xy == d.xy
        @test bp.tangent == d.tangent && bp.tangent_2 == d.tangent_2
        @test bp.ts == d.ts && bp.tphys == d.tphys
        @test bp.ws == d.ws && bp.ws_der == d.ws_der
        @test bp.s == d.s && bp.ds == d.ds
        @test bp.compid == 2
        @test bp.is_periodic === false
        @test bp.xL == xL && bp.xR == xR && bp.tL == tL && bp.tR == tR
        # fields not set by this constructor stay empty
        @test isempty(bp.w) && isempty(bp.w_n) && isempty(bp.curvature) && isempty(bp.w_vs) && isempty(bp.w_dm)
    end

    @testset "Float32" begin
        d = bpt_circle_data(Float32, 16, 1.0)
        bp = BoundaryPoints(d.xy, d.tangent, d.tangent_2, d.ts, d.tphys, d.ws, d.ws_der,
                            d.s, d.ds, 1, true, d.z, d.z, d.z, d.z)
        @test bp isa BoundaryPoints{Float32}
        @test eltype(bp.normal) == SVector{2,Float32}
        @test all(isapprox.(norm.(bp.normal), 1f0; atol=1f-5))
    end

    @testset "inferred" begin
        d = bpt_circle_data(Float64, 8, 1.0)
        @test @inferred(BoundaryPoints(d.xy, d.tangent, d.tangent_2, d.ts, d.tphys, d.ws, d.ws_der,
                                       d.s, d.ds, 1, true, d.z, d.z, d.z, d.z)) isa BoundaryPoints{Float64}
    end

    @testset "length validation is applied" begin
        d = bpt_circle_data(Float64, 8, 1.0)
        @test_throws ErrorException BoundaryPoints(d.xy, d.tangent, d.tangent_2, d.ts[1:4], d.tphys, d.ws, d.ws_der,
                                                   d.s, d.ds, 1, true, d.z, d.z, d.z, d.z)
        @test_throws ErrorException BoundaryPoints(d.xy, d.tangent, d.tangent_2, d.ts, d.tphys, d.ws, d.ws_der,
                                                   d.s, d.ds[1:3], 1, true, d.z, d.z, d.z, d.z)
    end
end

@testset "length and isempty of BoundaryPoints" begin
    xy = [SVector(0.0, 0.0), SVector(1.0, 0.0), SVector(2.0, 0.0)]
    @test length(BoundaryPoints(xy)) == 3
    @test !isempty(BoundaryPoints(xy))
    empty_bp = BoundaryPoints(SVector{2,Float64}[])
    @test length(empty_bp) == 0
    @test isempty(empty_bp)
    @test @inferred(length(BoundaryPoints(xy))) == 3
end

@testset "boundary_matrix_size" begin
    bp1 = BoundaryPoints([SVector(Float64(i), 0.0) for i in 1:3])
    bp2 = BoundaryPoints([SVector(Float64(i), 1.0) for i in 1:5])
    @test boundary_matrix_size(bp1) == 3
    @test boundary_matrix_size([bp1, bp2]) == 8
    # SOURCE BUG (boundary_matrix_size(::Vector{BoundaryPoints})): `sum(length, pts)` throws on an empty vector
    # (no `init`), unlike boundary_s / component_offsets which handle the empty case.
    @test_broken boundary_matrix_size(BoundaryPoints{Float64}[]) == 0
    @test @inferred(boundary_matrix_size([bp1, bp2])) == 8
end

@testset "component_offsets" begin
    bp1 = BoundaryPoints([SVector(Float64(i), 0.0) for i in 1:3])
    bp2 = BoundaryPoints([SVector(Float64(i), 1.0) for i in 1:5])
    bp3 = BoundaryPoints([SVector(Float64(i), 2.0) for i in 1:2])
    @test component_offsets(bp1) == [1, 4]
    @test component_offsets([bp1, bp2]) == [1, 4, 9]
    @test component_offsets([bp1, bp2, bp3]) == [1, 4, 9, 11]
    @test component_offsets(BoundaryPoints{Float64}[]) == [1]
    # consecutive offsets differ by the component sizes
    @test diff(component_offsets([bp1, bp2, bp3])) == [3, 5, 2]
    @test @inferred(component_offsets([bp1, bp2])) == [1, 4, 9]
end

@testset "boundary_s" begin
    bp1 = BoundaryPoints([SVector(Float64(i), 0.0) for i in 1:3]; s=[0.0, 1.0, 2.0], ds=fill(1.0, 3))
    bp2 = BoundaryPoints([SVector(Float64(i), 1.0) for i in 1:5]; s=[0.0, 1.0, 2.0, 3.0, 4.0], ds=fill(1.0, 5))
    bp3 = BoundaryPoints([SVector(Float64(i), 2.0) for i in 1:2]; s=[0.0, 0.5], ds=fill(0.5, 2))

    @test boundary_s(bp1) == [0.0, 1.0, 2.0]
    @test boundary_s(BoundaryPoints{Float64}[]) == Float64[]

    s12 = boundary_s([bp1, bp2])
    @test length(s12) == 8
    # the second component is shifted by the total ds of the first (3.0)
    @test s12[1:3] == bp1.s
    @test s12[4:8] ≈ bp2.s .+ 3.0

    s123 = boundary_s([bp1, bp2, bp3])
    @test s123[9:10] ≈ bp3.s .+ 8.0            # shift = 3.0 + 5.0
    @test issorted(s123)
    @test @inferred(boundary_s([bp1, bp2])) isa Vector{Float64}
end

@testset "points_in_billiard" begin
    # RectangleBilliard(1.0, 0.5): fundamental domain [0,1]x[0,0.5]
    b = RectangleBilliard(1.0, 0.5)
    inside = SVector(0.5, 0.25)
    far = SVector(1e3, 1e3)
    @test points_in_billiard([inside], b) == [true]
    @test points_in_billiard([far], b) == [false]
    @test points_in_billiard([inside, far, SVector(0.1, 0.1), SVector(-0.5, 0.2)], b) == [true, false, true, false]
    @test isempty(points_in_billiard(SVector{2,Float64}[], b))

    veech, _ = make_veech_right_triangle_and_basis(4)
    @test all(points_in_billiard(random_interior_points(veech, 5), veech))
    @test points_in_billiard([far], veech) == [false]
end

@testset "random_interior_points" begin
    Random.seed!(20240607)
    b = RectangleBilliard(1.0, 0.5)
    pts = random_interior_points(b, 25)
    @test length(pts) == 25
    @test pts isa Vector{SVector{2,Float64}}
    @test all(points_in_billiard(pts, b))
    @test all(p -> 0.0 <= p[1] <= 1.0 && 0.0 <= p[2] <= 0.5, pts)
    @test isempty(random_interior_points(b, 0))
    @test length(random_interior_points(b, 3; grd=200)) == 3
end

@testset "_determine_bp_sizes" begin
    b = RectangleBilliard(1.0, 0.5)
    curves = get_boundary_curves(b)         # lengths 0.5 and 1.0
    bs = [3.0, 3.0]

    # lower bound of 20 points per curve
    @test _determine_bp_sizes(curves, bs, 0.1) == [20, 20]
    @test _determine_bp_sizes(curves, bs, 0.1) isa Vector{Int}

    # the point count grows linearly with k (up to integer rounding) and with the scaling factor
    k = 400.0
    N1 = _determine_bp_sizes(curves, bs, k)
    N2 = _determine_bp_sizes(curves, bs, 2k)
    @test all(abs.(N2 .- 2 .* N1) .<= 1)
    @test all(N1 .> 20)
    Nb = _determine_bp_sizes(curves, 2 .* bs, k)
    @test all(abs.(Nb .- 2 .* N1) .<= 1)

    # the longer curve (1.0) gets about twice the points of the shorter one (0.5)
    @test abs(N1[2] - 2*N1[1]) <= 1
end

@testset "get_boundary_curves_with_ignored" begin
    @testset "billiard without ignored curves" begin
        b = RectangleBilliard(1.0, 0.5)
        with_ignored = get_boundary_curves_with_ignored(b)
        regular = get_boundary_curves(b)
        @test length(with_ignored) == length(regular) == 2
        @test sort([c.length for c in with_ignored]) ≈ [0.5, 1.0]
    end

    @testset "billiard with QuantumSolverIgnore curves" begin
        veech, _ = make_veech_right_triangle_and_basis(4)
        with_ignored = get_boundary_curves_with_ignored(veech)
        regular = get_boundary_curves(veech)
        @test length(regular) == 1
        @test length(with_ignored) == 3
        @test count(c -> c.bc isa BilliardGeometry.QuantumSolverIgnore, with_ignored) == 2
        # right isosceles triangle with legs 1: perimeter 2 + sqrt(2)
        @test sum(c.length for c in with_ignored) ≈ 2 + sqrt(2)
    end
end

@testset "boundary_coords with explicit samplers" begin
    b = RectangleBilliard(1.0, 0.5)
    curves = get_boundary_curves_with_ignored(b)
    center = SVector(0.5, 0.25)
    samplers = AbsSampler[GaussLegendreNodes(), LinearNodes()]
    Ns = [6, 9]
    bp = boundary_coords(b, samplers, Ns)

    @test bp isa BoundaryPoints{Float64}
    @test length(bp) == sum(Ns)
    @test length(bp.normal) == length(bp.s) == length(bp.ds) == length(bp)
    @test isempty(bp.tangent) && isempty(bp.ts)

    offs = [0, Ns[1], Ns[1] + Ns[2]]
    for i in 1:2
        rng = (offs[i]+1):offs[i+1]
        t, _ = sample_points(samplers[i], Ns[i])
        tang = BilliardGeometry.tangent(curves[i], t)
        @test bp.xy[rng] ≈ BilliardGeometry.curve(curves[i], t)
        # both rules integrate the constant speed of a straight edge exactly
        @test sum(bp.ds[rng]) ≈ curves[i].length
        @test all(bp.ds[rng] .> 0)
        for (j, idx) in enumerate(rng)
            nhat = bp.normal[idx] / norm(bp.normal[idx])
            @test abs(dot(nhat, tang[j] / norm(tang[j]))) < 1e-12         # normal is orthogonal to the tangent
            @test dot(nhat, bp.xy[idx] - center) > 0                      # and points out of the convex domain
        end
    end

    # arc length is increasing and continuous across the curve junction
    @test issorted(bp.s)
    @test all(bp.s[(offs[2]+1):offs[3]] .> curves[1].length)
    @test bp.s[end] < curves[1].length + curves[2].length

    # SOURCE BUG (boundary_coords): `normal .= normal ./ norm(normal)` divides by the 2-norm of the whole
    # vector of normals instead of normalizing each one, so stored normals have length 1/sqrt(N_curve).
    # Compare `normal ./ norm.(normal)` used in evaluate_points(::DecompositionMethodSolver, ...).
    @test_broken all(isapprox.(norm.(bp.normal), 1.0; atol=1e-12))
end

@testset "boundary_coords normals on a circular arc" begin
    b = CircleBilliard(1.0)
    curves = get_boundary_curves_with_ignored(b)
    bp = boundary_coords(b, AbsSampler[GaussLegendreNodes() for _ in curves], fill(12, length(curves)))
    @test length(bp) == 12 * length(curves)
    # origin-centered circle: outward normal direction is the radial direction
    @test all(isapprox.([n / norm(n) for n in bp.normal], [p / norm(p) for p in bp.xy]; atol=1e-10))
    @test all(isapprox.(norm.(bp.xy), 1.0; atol=1e-12))
    @test sum(bp.ds) ≈ sum(c.length for c in curves)
end

@testset "boundary_coords argument errors" begin
    b = RectangleBilliard(1.0, 0.5)             # two boundary curves
    @test_throws ArgumentError boundary_coords(b, AbsSampler[GaussLegendreNodes()], [4, 4])
    @test_throws ArgumentError boundary_coords(b, AbsSampler[GaussLegendreNodes(), LinearNodes()], [4])
    @test_throws ArgumentError boundary_coords(b, AbsSampler[GaussLegendreNodes(), LinearNodes()], [4, 4, 4])
end

@testset "boundary_coords with FourierNodes" begin
    b = StadiumBilliard(0.5)
    curves = get_boundary_curves_with_ignored(b)
    lengths = [c.length for c in curves]
    M = 40
    fs = FourierNodes(lengths)
    bp = boundary_coords(b, fs, M)
    ts, _ = sample_points(fs, M)

    @test bp isa BoundaryPoints{Float64}
    @test length(bp) == sum(length, ts)
    # one node per lattice step; the count equals M up to one extra node per curve
    @test M <= length(bp) <= M + length(curves)
    @test bp.xy ≈ vcat((BilliardGeometry.curve(curves[i], ts[i]) for i in eachindex(curves))...)
    @test all(bp.ds .> 0)
    @test issorted(bp.s)
    # the lattice spacing in physical length is sum(lengths)/M, so the total is off by at most one step per curve
    @test isapprox(sum(bp.ds), sum(lengths); atol=length(curves) * sum(lengths) / M)
    off = 0
    for i in eachindex(curves)
        rng = (off+1):(off+length(ts[i]))
        tang = BilliardGeometry.tangent(curves[i], ts[i])
        for (j, idx) in enumerate(rng)
            nhat = bp.normal[idx] / norm(bp.normal[idx])
            @test abs(dot(nhat, tang[j] / norm(tang[j]))) < 1e-10
        end
        off += length(ts[i])
    end
end

@testset "flatten_boundary_components" begin
    xy1 = [SVector(0.0, 0.0), SVector(1.0, 0.0)]
    xy2 = [SVector(0.0, 1.0), SVector(1.0, 1.0), SVector(2.0, 1.0)]
    n1 = [SVector(0.0, -1.0), SVector(0.0, -1.0)]
    n2 = [SVector(0.0, 1.0), SVector(0.0, 1.0), SVector(0.0, 1.0)]
    bp1 = BoundaryPoints(xy1; normal=n1, ds=fill(0.5, 2))
    bp2 = BoundaryPoints(xy2; normal=n2, ds=fill(0.25, 3))
    flat = flatten_boundary_components([bp1, bp2])

    @test keys(flat) == (:x, :y, :nx, :ny, :ds, :offs)
    @test flat.x == [0.0, 1.0, 0.0, 1.0, 2.0]
    @test flat.y == [0.0, 0.0, 1.0, 1.0, 1.0]
    @test flat.nx == zeros(5)
    @test flat.ny == [-1.0, -1.0, 1.0, 1.0, 1.0]
    @test flat.ds == [0.5, 0.5, 0.25, 0.25, 0.25]
    @test flat.offs == [1, 3, 6]
    @test flat.offs == component_offsets([bp1, bp2])

    # normal components come from the stored normals
    d = bpt_circle_data(Float64, 16, 2.0)
    circ = BoundaryPoints(d.xy, d.tangent, d.tangent_2, d.ts, d.tphys, d.ws, d.ws_der,
                          d.s, d.ds, 1, true, d.z, d.z, d.z, d.z)
    fc = flatten_boundary_components([circ])
    @test fc.nx ≈ fc.x ./ 2.0
    @test fc.ny ≈ fc.y ./ 2.0
    @test @inferred(flatten_boundary_components([bp1, bp2])) isa NamedTuple

    # a component without normals cannot be flattened
    no_normals = BoundaryPoints(xy1; ds=fill(0.5, 2))
    @test_throws ArgumentError flatten_boundary_components([bp1, no_normals])
end

@testset "flatten_boundary_ds" begin
    bp1 = BoundaryPoints([SVector(0.0, 0.0), SVector(1.0, 0.0)]; ds=fill(0.5, 2))
    bp2 = BoundaryPoints([SVector(0.0, 1.0), SVector(1.0, 1.0), SVector(2.0, 1.0)]; ds=fill(0.25, 3))
    @test flatten_boundary_ds([bp1, bp2]) == [0.5, 0.5, 0.25, 0.25, 0.25]
    @test flatten_boundary_ds([bp2]) == fill(0.25, 3)
    @test flatten_boundary_ds([bp1, bp2]) == flatten_boundary_components([BoundaryPoints(bp1.xy; normal=fill(SVector(0.0, 1.0), 2), ds=bp1.ds), BoundaryPoints(bp2.xy; normal=fill(SVector(0.0, 1.0), 3), ds=bp2.ds)]).ds
    @test sum(flatten_boundary_ds([bp1, bp2])) ≈ sum(bp1.ds) + sum(bp2.ds)
    @test @inferred(flatten_boundary_ds([bp1, bp2])) isa Vector{Float64}
    # SOURCE BUG (flatten_boundary_ds): inherits the empty-vector failure of boundary_matrix_size.
    @test_broken isempty(flatten_boundary_ds(BoundaryPoints{Float64}[]))
end
