# Unit tests for
#   src/solvers/sweepmethods/compositebim.jl   (CompositeBIMSolver: the composite helpers _group_boundary_by_domain_id,
#                                               _group_is_hole, _merge_composite_points (through evaluate_points),
#                                               _composite_offsets, _composite_component_slice, _composite_global_to_local,
#                                               _flip_component_orientation, _combine_composite_orbits,
#                                               _composite_symmetry_orbits, and evaluate_points, boundary_matrix_size,
#                                               construct_matrices with _composite_fredholm_full! / _composite_fredholm_reduced!)
#   src/solvers/taylor/bim_kernels_recurrence.jl   (_construct_matrices_multi_k_taylor for composite solvers)
# Matrix level only: no eigen solve, no tension, no search for wavenumbers. Billiards: AnnularBilliard(0.3, 0.1) (unit circle
# with an eccentric circular hole of radius 0.3 around (0.1, 0), no symmetry) and SinaiBilliard(1.0, 0.25) (square with a
# central circular hole, D2 symmetry). The checks are
#   * the grouping of curves by domain_id (first occurrence order, order inside a group) and the hole detection;
#   * evaluate_points: component sizes from the component solvers, contiguous blocks, per-point component labels, nodes and
#     weights of every block equal to the standalone solver on the same circle, outward normals of the domain (the hole normals
#     point into the hole), per-component grading;
#   * the block structure of the full matrix: the diagonal blocks equal the single-component DLP / CFIE matrices (for the hole
#     block through the orientation flip D -> -D, S -> S, i.e. 2 I - 2 A_DLP + A_CFIE of the standalone circle) and the
#     off-diagonal blocks equal the smooth Nystrom kernels written out from the merged nodes;
#   * the symmetry reduced matrix against the orbit-summed columns of the full matrix, with the orbits assembled block by
#     block from BilliardGeometry.symmetry_index_orbits;
#   * the direct matrix against the analytic Taylor matrix, boundary_matrix_size, threaded = serial, argument validation.
# Helper definitions use the prefix `cb_`.

using LinearAlgebra: norm, diag
using SpecialFunctions: besselh
using StaticArrays: SVector
using QuantumBilliards: _group_boundary_by_domain_id, _group_is_hole, _composite_offsets, _composite_component_slice,
                        _composite_global_to_local, _flip_component_orientation, _combine_composite_orbits,
                        _composite_symmetry_orbits, _construct_matrices_multi_k_taylor, _is_nontrivial_dlp_grading

# Orbit sum of the columns of the full composite matrix. `locals[a]` is the orbit map of component a (built on the component's
# own node count); the maps are shifted into the concatenated indexing by hand, E[r, c] = sum over the nodes j of the orbit c
# of phase[j] A[fund[r], j].
function cb_orbit_sum(Afull, offs, locals)
    fund = Int[]
    orbit_of = Int[]
    phase = ComplexF64[]
    moff = 0
    for (a, o) in enumerate(locals)
        append!(fund, o.fundamental_indices .+ (offs[a] - 1))
        append!(orbit_of, o.orbit_of .+ moff)
        append!(phase, o.phase)
        moff += fundamental_size(o)
    end
    E = zeros(ComplexF64, length(fund), length(fund))
    for j in eachindex(orbit_of), r in eachindex(fund)
        E[r, orbit_of[j]] += phase[j] * Afull[fund[r], j]
    end
    return E
end

# True if for every orbit the physical nodes `xy` of its images are exactly the images of the fundamental node under `maps`.
function cb_orbit_matches(xy, orbits, maps)
    ng = orbit_size(orbits)
    length(maps) == ng || return false
    for a in 1:fundamental_size(orbits)
        x = xy[orbits.fundamental_indices[a]]
        images = [xy[orbits.fund_to_full[g, a]] for g in 1:ng]
        all(f -> any(q -> norm(q - f(x)) < 1e-10, images), maps) || return false
    end
    return true
end

# Smooth Nystrom kernels between a target node i and a source node j of different components, from the merged nodes:
#   D = w_j (i k / 2) (t_y dx - t_x dy) H1(k r) / r,    S = w_j (i / 2) H0(k r) |t|,    (dx, dy) = x_i - x_j.
function cb_cross_dlp(pts, k, i, j)
    dx, dy = pts.xy[i] - pts.xy[j]
    r = hypot(dx, dy)
    tx, ty = pts.tangent[j]
    return pts.ws[j] * im * k / 2 * (ty * dx - tx * dy) * besselh(1, 1, k * r) / r
end
function cb_cross_slp(pts, k, i, j)
    r = norm(pts.xy[i] - pts.xy[j])
    return pts.ws[j] * im / 2 * besselh(0, 1, k * r) * norm(pts.tangent[j])
end

@testset "_group_boundary_by_domain_id" begin
    annulus = AnnularBilliard(0.3, 0.1)
    sinai = SinaiBilliard(1.0, 0.25)

    @testset "annulus: outer circle and hole" begin
        comp = full_boundary(annulus)
        groups = _group_boundary_by_domain_id(comp)
        @test length(groups) == 2
        @test length.(groups) == [1, 1]
        @test [g[1].domain_id for g in groups] == [1, 2]
        @test groups[1][1] === comp[1] && groups[2][1] === comp[2]
    end

    @testset "Sinai: the curves of a component keep their order" begin
        comp = full_boundary(sinai)
        groups = _group_boundary_by_domain_id(comp)
        @test length.(groups) == [8, 4]
        @test groups[1] == [c for c in comp if c.domain_id == 1]
        @test groups[2] == [c for c in comp if c.domain_id == 2]
        @test sum(c.length for c in groups[2]) ≈ 2π * 0.25              # the hole is a full circle of radius 0.25
        @test sum(c.length for c in groups[1]) ≈ 8.0                    # the outer boundary is the square of side 2
    end

    @testset "fundamental domain of the Sinai billiard" begin
        groups = _group_boundary_by_domain_id(get_boundary_curves(sinai))
        @test length.(groups) == [2, 1]
    end

    @testset "components are ordered by first occurrence" begin
        groups = _group_boundary_by_domain_id(reverse(full_boundary(annulus)))
        @test [g[1].domain_id for g in groups] == [2, 1]
    end
end

@testset "_group_is_hole" begin
    annulus = AnnularBilliard(0.3, 0.1)
    sinai = SinaiBilliard(1.0, 0.25)

    @testset "orientation +1 is an outer boundary, -1 a hole" begin
        groups = _group_boundary_by_domain_id(get_boundary_curves(annulus))
        @test _group_is_hole(groups[1]) == false
        @test _group_is_hole(groups[2]) == true
    end

    @testset "Sinai fundamental boundary" begin
        groups = _group_boundary_by_domain_id(get_boundary_curves(sinai))
        @test _group_is_hole(groups[1]) == false
        @test _group_is_hole(groups[2]) == true
    end

    @testset "invalid groups are rejected" begin
        outer, hole = get_boundary_curves(annulus)
        @test_throws ArgumentError _group_is_hole(AbsCurve[outer, hole])                 # mixed orientations in one component
        bad = CircleSegment(1.0, 2π, 0.0, (0.0, 0.0); orientation=2)
        @test_throws ArgumentError _group_is_hole(AbsCurve[bad])                         # orientation must be +-1
    end
end

@testset "_composite_global_to_local" begin
    g2c, g2l = _composite_global_to_local([1, 4, 6])
    @test g2c == [1, 1, 1, 2, 2]
    @test g2l == [1, 2, 3, 1, 2]
    g2c, g2l = _composite_global_to_local([1, 3, 4, 8])
    @test g2c == [1, 1, 2, 3, 3, 3, 3]
    @test g2l == [1, 2, 1, 1, 2, 3, 4]
end

@testset "evaluate_points: annulus with a DLP outer and a CFIE hole component" begin
    annulus = AnnularBilliard(0.3, 0.1)
    k = 3.0
    outer_solver = DoubleLayerPotentialSolver(5.0, annulus; min_pts=32)
    hole_solver = CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24)
    solver = CompositeBIMSolver(outer_solver, hole_solver)
    pts = evaluate_points(solver, annulus, k)
    outer, hole = full_boundary(annulus)
    c, ρ = hole.center, hole.radius
    rng1, rng2 = 1:32, 33:56

    @testset "sizes, blocks and component labels" begin
        @test pts isa BoundaryPoints{Float64}
        @test length(pts) == 56 == 32 + 24                              # the component solvers fix the block sizes
        @test _composite_offsets(pts, 2) == [1, 33, 57]
        @test pts.w_dm == [fill(1.0, 32); fill(2.0, 24)]                # private per-point component label
        @test boundary_matrix_size(solver, pts) == 56 == boundary_matrix_size(pts)
        @test hole.radius == 0.3 && hole.center == SVector(0.1, 0.0)
    end

    @testset "nodes and weights equal the standalone solvers on the same circles" begin
        circle = CircleBilliard(1.0)
        ref = evaluate_points(DoubleLayerPotentialSolver(5.0, circle; min_pts=32), circle, k)
        @test pts.xy[rng1] ≈ ref.xy
        @test pts.ws[rng1] == ref.ws && pts.ds[rng1] ≈ ref.ds && pts.ts[rng1] == ref.ts
        θ = [(j - 1 / 2) * 2π / 24 for j in 1:24]
        @test pts.xy[rng2] ≈ [c + ρ * SVector(cos(t), sin(t)) for t in θ]    # the hole nodes run counterclockwise from angle 0
        @test all(≈(2π / 24), pts.ws[rng2])
        @test all(≈(ρ * 2π / 24), pts.ds[rng2])
    end

    @testset "outward normals of the domain" begin
        @test pts.normal[rng1] ≈ pts.xy[rng1]                           # outer circle: radial outward
        @test pts.normal[rng2] ≈ [-(x - c) / ρ for x in pts.xy[rng2]]   # hole: pointing into the hole
    end

    @testset "the hole tangent is the flipped orientation, the node order is kept" begin
        raw = evaluate_points(CombinedFieldIntegralEquationSolver(5.0, CircleBilliard(0.3); min_pts=24), CircleBilliard(0.3), k)
        @test pts.tangent[rng2] ≈ -raw.tangent
        @test pts.tangent_2[rng2] ≈ raw.tangent_2
        @test pts.xy[rng2] ≈ raw.xy .+ [c]
    end

    @testset "Float32 solver gives Float32 points" begin
        s32 = CompositeBIMSolver(DoubleLayerPotentialSolver(5f0, annulus; min_pts=32), DoubleLayerPotentialSolver(5f0, annulus; min_pts=24))
        pts32 = evaluate_points(s32, annulus, 3f0)
        @test pts32 isa BoundaryPoints{Float32}
        @test length(pts32) == 56
    end
end

@testset "evaluate_points: Sinai billiard with a corner graded outer component" begin
    sinai = SinaiBilliard(1.0, 0.25)
    k = 3.0
    outer_solver = DoubleLayerPotentialSolver(5.0, sinai; min_pts=64, grading=GlobalCornerGrading())
    hole_solver = CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=24)
    solver = CompositeBIMSolver(outer_solver, hole_solver)
    pts = evaluate_points(solver, sinai, k)
    offs = _composite_offsets(pts, 2)
    n1 = offs[2] - 1

    @test n1 == 64 && length(pts) == 64 + 24
    @test _is_nontrivial_dlp_grading(_composite_component_slice(pts, offs[1]:offs[2]-1, 1))     # the square is graded
    @test !_is_nontrivial_dlp_grading(_composite_component_slice(pts, offs[2]:offs[3]-1, 2))    # the circular hole is not
    @test all(==(1.0), pts.ws_der[n1+1:end])
    @test sum(pts.ws[1:n1]) ≈ 2π && sum(pts.ws[n1+1:end]) ≈ 2π
    @test sum(pts.ds[n1+1:end]) ≈ 2π * 0.25
    @test sum(pts.ds[1:n1]) ≈ 8.0 rtol = 1e-2
end

@testset "composite helpers on merged points" begin
    annulus = AnnularBilliard(0.3, 0.1)
    solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24))
    pts = evaluate_points(solver, annulus, 3.0)

    @testset "_composite_offsets" begin
        @test _composite_offsets(pts, 2) == [1, 33, 57]
        @test_throws ErrorException _composite_offsets(pts, 1)           # more labels than components
        @test_throws ErrorException _composite_offsets(pts, 3)           # fewer labels than components
        bad = deepcopy(pts)
        bad.w_dm[5] = 2.0
        bad.w_dm[6] = 1.0
        @test_throws ErrorException _composite_offsets(bad, 2)           # labels are not contiguous
        raw = evaluate_points(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), annulus, 3.0)
        @test_throws ErrorException _composite_offsets(raw, 2)           # no component bookkeeping at all
    end

    @testset "_composite_component_slice" begin
        part = _composite_component_slice(pts, 33:56, 2)
        @test part isa BoundaryPoints{Float64}
        @test length(part) == 24 && part.compid == 2
        @test part.xy == pts.xy[33:56] && part.tangent == pts.tangent[33:56] && part.tangent_2 == pts.tangent_2[33:56]
        @test part.ts == pts.ts[33:56] && part.tphys == pts.tphys[33:56] && part.ws == pts.ws[33:56] && part.ws_der == pts.ws_der[33:56]
        @test part.s == pts.s[33:56] && part.ds == pts.ds[33:56]
    end

    @testset "_flip_component_orientation" begin
        part = _composite_component_slice(pts, 1:32, 1)
        flipped = _flip_component_orientation(part)
        @test flipped.tangent == -part.tangent
        @test flipped.tangent_2 == part.tangent_2                       # the second derivative keeps its sign
        @test flipped.xy == part.xy && flipped.ts == part.ts && flipped.ws == part.ws && flipped.ds == part.ds
        @test _flip_component_orientation(flipped).tangent == part.tangent
    end
end

@testset "construct_matrices: annulus block structure" begin
    annulus = AnnularBilliard(0.3, 0.1)
    k = 3.0
    outer_solver = DoubleLayerPotentialSolver(5.0, annulus; min_pts=32)
    hole_solver = CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24)
    solver = CompositeBIMSolver(outer_solver, hole_solver)
    pts = evaluate_points(solver, annulus, k)
    A = construct_matrices(solver, pts, k)
    rng1, rng2 = 1:32, 33:56

    @test A isa Matrix{ComplexF64}
    @test size(A) == (56, 56)

    @testset "diagonal blocks equal the single-component matrices on the same nodes" begin
        @test A[rng1, rng1] == construct_matrices(outer_solver, _composite_component_slice(pts, rng1, 1), k)
        @test A[rng2, rng2] == construct_matrices(hole_solver, _composite_component_slice(pts, rng2, 2), k)
    end

    @testset "outer block equals the DLP matrix of the unit circle" begin
        circle = CircleBilliard(1.0)
        s = DoubleLayerPotentialSolver(5.0, circle; min_pts=32)
        @test A[rng1, rng1] ≈ construct_matrices(s, evaluate_points(s, circle, k), k) atol = 1e-12
    end

    @testset "hole block: outward normal of the domain flips D and keeps S" begin
        circle = CircleBilliard(0.3)
        cfie = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=24)
        dlp = DoubleLayerPotentialSolver(5.0, circle; min_pts=24)
        raw = evaluate_points(cfie, circle, k)
        A_cfie = construct_matrices(cfie, raw, k)                       # I - D - i k S
        A_dlp = construct_matrices(dlp, raw, k)                         # I - D
        expected = 2 * one(A_cfie) - 2 * A_dlp + A_cfie                 # I + D - i k S
        @test isapprox(A[rng2, rng2], expected; atol=1e-12, rtol=0)
        @test norm(A[rng2, rng2] - A_cfie) > 1e-2                       # not the unflipped matrix
    end

    @testset "off-diagonal blocks are the smooth Nystrom kernels" begin
        # source on the hole (CFIE): -(D + i k S), source on the outer circle (DLP): -D
        @test all(isapprox(A[i, j], -(cb_cross_dlp(pts, k, i, j) + im * k * cb_cross_slp(pts, k, i, j)); atol=1e-13) for i in rng1, j in rng2)
        @test all(isapprox(A[i, j], -cb_cross_dlp(pts, k, i, j); atol=1e-13) for i in rng2, j in rng1)
        @test maximum(abs.(A[rng1, rng2])) > 1e-3 && maximum(abs.(A[rng2, rng1])) > 1e-3
    end

    @testset "complex wavenumber" begin
        kc = 2.5 + 0.4im
        Ac = construct_matrices(solver, pts, kc)
        @test all(isapprox(Ac[i, j], -(cb_cross_dlp(pts, kc, i, j) + im * kc * cb_cross_slp(pts, kc, i, j)); atol=1e-12) for i in rng1, j in rng2)
        @test Ac[rng2, rng2] == construct_matrices(hole_solver, _composite_component_slice(pts, rng2, 2), kc)
    end
end

@testset "construct_matrices: Sinai billiard without symmetry, corner graded outer component" begin
    sinai = SinaiBilliard(1.0, 0.25)
    k = 3.0
    outer_solver = CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=64, grading=GlobalCornerGrading())
    hole_solver = DoubleLayerPotentialSolver(5.0, sinai; min_pts=24)
    solver = CompositeBIMSolver(outer_solver, hole_solver)
    pts = evaluate_points(solver, sinai, k)
    offs = _composite_offsets(pts, 2)
    rng1, rng2 = offs[1]:offs[2]-1, offs[2]:offs[3]-1
    A = construct_matrices(solver, pts, k)

    @test size(A) == (88, 88)
    @test A[rng1, rng1] == construct_matrices(outer_solver, _composite_component_slice(pts, rng1, 1), k)
    @test A[rng2, rng2] == construct_matrices(hole_solver, _composite_component_slice(pts, rng2, 2), k)
    @test all(isapprox(A[i, j], -cb_cross_dlp(pts, k, i, j); atol=1e-13) for i in rng1, j in rng2)      # DLP source on the hole
    @test all(isapprox(A[i, j], -(cb_cross_dlp(pts, k, i, j) + im * k * cb_cross_slp(pts, k, i, j)); atol=1e-13) for i in rng2, j in rng1)
end

@testset "reduced matrix equals orbit-summed full matrix: Sinai billiard, XY reflection" begin
    sinai = SinaiBilliard(1.0, 0.25)
    k = 3.0
    full = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, sinai; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=24))
    red = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, sinai; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(-1, 1)),
                             CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=24, symmetry=XYAxisReflection(), sym_characters=(-1, 1)))
    pts = evaluate_points(red, sinai, k)
    offs = _composite_offsets(pts, 2)
    locals = [symmetry_index_orbits(Float64, sinai, offs[a+1] - offs[a], XYAxisReflection(), -1.0 + 0im, 1.0 + 0im) for a in 1:2]
    A_red = construct_matrices(red, pts, k)
    maps = (x -> x, x -> SVector(x[1], -x[2]), x -> SVector(-x[1], x[2]), x -> -x)

    @test evaluate_points(full, sinai, k).xy == pts.xy                  # the reduced solver samples the whole boundary
    @test boundary_matrix_size(red, pts) == sum(fundamental_size, locals) == length(pts) ÷ 4
    @test size(A_red) == (length(pts) ÷ 4, length(pts) ÷ 4)
    @test cb_orbit_matches(pts.xy[offs[1]:offs[2]-1], locals[1], maps)
    @test cb_orbit_matches(pts.xy[offs[2]:offs[3]-1], locals[2], maps)
    @test isapprox(A_red, cb_orbit_sum(construct_matrices(full, pts, k), offs, locals); atol=1e-12, rtol=0)

    @testset "_composite_symmetry_orbits and _combine_composite_orbits" begin
        orbits = _composite_symmetry_orbits(Float64, red, pts)
        combined = _combine_composite_orbits(Float64, locals)
        @test full_size(orbits) == length(pts) && fundamental_size(orbits) == length(pts) ÷ 4 && orbit_size(orbits) == 4
        @test orbits.fundamental_indices == combined.fundamental_indices
        @test orbits.orbit_of == combined.orbit_of
        @test orbits.phase == combined.phase
        @test orbits.fund_to_full == combined.fund_to_full && orbits.fund_to_scale == combined.fund_to_scale
        @test orbits.orbit_of[offs[2]:end] == locals[2].orbit_of .+ fundamental_size(locals[1])      # blocks are shifted, not mixed
        @test orbits.fundamental_indices[fundamental_size(locals[1])+1:end] == locals[2].fundamental_indices .+ (offs[2] - 1)
    end
end

@testset "reduced matrix equals orbit-summed full matrix: Sinai billiard, shared CompositeReflection, corner graded outer component" begin
    sinai = SinaiBilliard(1.0, 0.25)
    k = 3.0
    inferred = DoubleLayerPotentialSolver(5.0, sinai; sym_characters=(1, -1))      # only used to obtain the inferred (Y, X) symmetry
    sym, chars = inferred.symmetry, inferred.sym_characters
    full = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=64, grading=GlobalCornerGrading()),
                              CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=24))
    red = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=64, grading=GlobalCornerGrading(), symmetry=sym, sym_characters=chars),
                             CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=24, symmetry=sym, sym_characters=chars))
    pts = evaluate_points(red, sinai, k)
    offs = _composite_offsets(pts, 2)
    locals = [symmetry_index_orbits(Float64, sinai, offs[a+1] - offs[a], CompositeReflection([YAxisReflection(), XAxisReflection()]), ComplexF64[1, -1]) for a in 1:2]

    @test red.symmetry isa CompositeReflection && red.symmetry === sym
    @test _is_nontrivial_dlp_grading(_composite_component_slice(pts, offs[1]:offs[2]-1, 1))
    @test boundary_matrix_size(red, pts) == length(pts) ÷ 4
    @test isapprox(construct_matrices(red, pts, k), cb_orbit_sum(construct_matrices(full, pts, k), offs, locals); atol=1e-12, rtol=0)
end

# Source finding: component solvers that each infer a multi-generator reflection symmetry from `sym_characters` hold two distinct
# but equal CompositeReflection objects, and CompositeBIMSolver compares them with `==`, which falls back to `===` because
# BilliardGeometry defines no `==` for CompositeReflection. The constructor therefore rejects solvers that are equal.
@testset "CompositeBIMSolver accepts separately inferred, equal CompositeReflection symmetries" begin
    sinai = SinaiBilliard(1.0, 0.25)
    a = DoubleLayerPotentialSolver(5.0, sinai; sym_characters=(1, -1))
    b = DoubleLayerPotentialSolver(5.0, sinai; sym_characters=(1, -1))
    @test a.sym_characters == b.sym_characters
    @test a.symmetry.reflections == b.symmetry.reflections
    @test_broken a.symmetry == b.symmetry
    @test_broken (CompositeBIMSolver(a, b); true)
end

@testset "construct_matrices: direct matrix against the Taylor matrix" begin
    k0 = 3.0
    ks = [k0 + 0.02, k0 - 0.01, k0 + 0.01im]

    @testset "annulus, full matrix" begin
        annulus = AnnularBilliard(0.3, 0.1)
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24))
        pts = evaluate_points(solver, annulus, k0)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        @test length(T) == 3
        @test all(M -> M isa Matrix{ComplexF64} && size(M) == (56, 56), T)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[2], construct_matrices(solver, pts, ks[2]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)     # complex wavenumber

        T0 = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 0)
        @test all(M -> isapprox(M, construct_matrices(solver, pts, k0); atol=1e-12, rtol=0), T0)
        @test norm(T0[1] - construct_matrices(solver, pts, ks[1])) > 1e-4           # the truncation is visible
    end

    @testset "Sinai billiard, XY reflection reduced" begin
        sinai = SinaiBilliard(1.0, 0.25)
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, sinai; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(-1, 1)),
                                    CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=24, symmetry=XYAxisReflection(), sym_characters=(-1, 1)))
        pts = evaluate_points(solver, sinai, k0)
        m = boundary_matrix_size(solver, pts)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        @test all(M -> size(M) == (m, m), T)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)
    end
end

@testset "boundary_matrix_size" begin
    sinai = SinaiBilliard(1.0, 0.25)
    annulus = AnnularBilliard(0.3, 0.1)
    k = 3.0

    @testset "no symmetry: number of nodes of all components" begin
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), DoubleLayerPotentialSolver(5.0, annulus; min_pts=24))
        pts = evaluate_points(solver, annulus, k)
        @test boundary_matrix_size(solver, pts) == length(pts) == 56
    end

    @testset "XY reflection: a quarter of every component" begin
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, sinai; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(1, 1)),
                                    DoubleLayerPotentialSolver(5.0, sinai; min_pts=24, symmetry=XYAxisReflection(), sym_characters=(1, 1)))
        pts = evaluate_points(solver, sinai, k)
        offs = _composite_offsets(pts, 2)
        @test boundary_matrix_size(solver, pts) == (offs[2] - 1) ÷ 4 + (offs[3] - offs[2]) ÷ 4
        @test size(construct_matrices(solver, pts, k)) == (boundary_matrix_size(solver, pts), boundary_matrix_size(solver, pts))
    end
end

@testset "threaded assembly equals serial assembly" begin
    k = 3.0

    @testset "annulus, full matrix (block and cross loops threaded)" begin
        annulus = AnnularBilliard(0.3, 0.1)
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=40), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=40))
        pts = evaluate_points(solver, annulus, k)
        @test length(pts) == 80
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end

    @testset "Sinai billiard, XY reflection reduced matrix" begin
        sinai = SinaiBilliard(1.0, 0.25)
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, sinai; min_pts=64, symmetry=XYAxisReflection(), sym_characters=(-1, 1)),
                                    CombinedFieldIntegralEquationSolver(5.0, sinai; min_pts=64, symmetry=XYAxisReflection(), sym_characters=(-1, 1)))
        pts = evaluate_points(solver, sinai, k)
        @test boundary_matrix_size(solver, pts) >= 32                   # large enough for the threaded loop to be used
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end
end

@testset "argument validation" begin
    annulus = AnnularBilliard(0.3, 0.1)
    sinai = SinaiBilliard(1.0, 0.25)
    k = 3.0
    dlp = DoubleLayerPotentialSolver(5.0, annulus; min_pts=32)
    cfie = CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24)

    @testset "the number of component solvers must equal the number of boundary components" begin
        @test_throws ArgumentError evaluate_points(CompositeBIMSolver(dlp), annulus, k)
        @test_throws ArgumentError evaluate_points(CompositeBIMSolver(dlp, cfie, cfie), annulus, k)
        @test_throws ArgumentError evaluate_points(CompositeBIMSolver(dlp, cfie), CircleBilliard(1.0), k)
        @test length(evaluate_points(CompositeBIMSolver(dlp, cfie), annulus, k)) == 56
        @test length(evaluate_points(CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, sinai; min_pts=32), cfie), sinai, k)) > 0
    end

    @testset "construct_matrices requires the points of evaluate_points(::CompositeBIMSolver, ...)" begin
        solver = CompositeBIMSolver(dlp, cfie)
        @test_throws ErrorException construct_matrices(solver, evaluate_points(dlp, annulus, k), k)
        @test_throws ErrorException construct_matrices(CompositeBIMSolver(dlp, cfie, cfie), evaluate_points(solver, annulus, k), k)
    end

    @testset "orbit maps" begin
        solver = CompositeBIMSolver(dlp, cfie)
        pts = evaluate_points(solver, annulus, k)
        @test_throws ArgumentError _composite_symmetry_orbits(Float64, solver, pts)              # no symmetry active
        @test_throws ArgumentError _combine_composite_orbits(Float64, SymmetryOrbitMap{Float64}[])
        o2 = symmetry_index_orbits(Float64, sinai, 8, YAxisReflection(), 1.0 + 0im)
        o4 = symmetry_index_orbits(Float64, sinai, 8, XYAxisReflection(), 1.0 + 0im, 1.0 + 0im)
        @test_throws ArgumentError _combine_composite_orbits(Float64, [o2, o4])                  # different orbit sizes
        @test fundamental_size(_combine_composite_orbits(Float64, [o2, o2])) == 8
    end
end
