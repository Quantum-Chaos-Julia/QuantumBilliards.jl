# Unit tests for the solver-agnostic helpers of
#   src/solvers/sweepmethods/sweepmethods.jl   (_fold_boundary, _infer_bim_symmetry, symmetrize_layer_density,
#                                               _bim_normal_derivative for DLP kernels, _bim_dlp_cross_kernel_entry,
#                                               _cfie_derivative!, and the symmetric-holes plumbing _orbit_character_args,
#                                               _fold_group, _bim_node_multiple, _bim_round_nodes)
# _fold_boundary adapts the exact symmetry-orbit map of BilliardGeometry (symmetry_index_orbits) to the character storage of
# the BIM solvers: scalar characters for single reflections, one Vector for CompositeReflection and one integer sector for
# NFoldRotation. _infer_bim_symmetry pairs `sym_characters` with the minimal generators of the billiard.
# symmetrize_layer_density expands a reduced density to the full boundary through the defining orbit identity
# full[g(q)] = chi_g * reduced[q]. The orbit maps themselves are tested in BilliardGeometry.jl; here only the dispatch, the
# argument checks and the identities that tie the helpers together are checked. The generic solve_wavenumber and k_sweep of
# this file solve eigenproblems and belong to the integration tier; solve_state and the CFIE normal derivative are covered
# with the DLP and CFIE kernels.
# No eigen solves. Helper definitions use the prefix `sw_`.
# _orbit_character_args turns the stored character tuple into the arguments that follow the symmetry in symmetry_index_orbits /
# group_symmetry_orbits; _fold_group is the same fold for one boundary group of a billiard with holes (group 0 is the outer boundary,
# so on a billiard without holes it equals _fold_boundary). _bim_node_multiple and _bim_round_nodes are the node-count rules:
# reflections about the axes 0, 1/4, 1/2, 3/4 need N divisible by 4 (4, 4, 8, 8 for X/Y, Diagonal/AntiDiagonal as in
# symmetry_node_multiple), a rotation by its order, a composite by the lcm, a generic axis by nothing (1). The hole billiards are
# the fixtures of test/holes_fixtures.jl (quarter square with D2, G' = D2 for four discs placed by D2, G' = {X} for a quarter-disc hole).

using SpecialFunctions: hankelh1
using StaticArrays: SVector
using QuantumBilliards: _fold_boundary, _infer_bim_symmetry, symmetrize_layer_density, _bim_normal_derivative,
                        _bim_dlp_cross_kernel_entry, _cfie_derivative!, _orbit_character_args, _fold_group,
                        _bim_node_multiple, _bim_round_nodes, _bim_solver_node_multiple

# Whether two SymmetryOrbitMaps agree in every field.
sw_same_orbits(a, b) = a.fundamental_indices == b.fundamental_indices && a.orbit_of == b.orbit_of && a.phase == b.phase &&
                       a.full_size == b.full_size && a.fundamental_size == b.fundamental_size &&
                       a.fund_to_full == b.fund_to_full && a.fund_to_scale == b.fund_to_scale

# The defining orbit identity: every group image of a representative carries its irrep factor times the representative value.
function sw_orbit_identity_holds(orbits, full, reduced)
    ng, nf = size(orbits.fund_to_full)
    return all(full[orbits.fund_to_full[g, b]] ≈ orbits.fund_to_scale[g, b] * reduced[b] for g in 1:ng, b in 1:nf)
end

@testset "_fold_boundary" begin
    rect = RectangleBilliard(1.0, 0.7)
    c3 = C3Billiard(0.02)

    @testset "single reflection: scalar character is converted to Complex{T}" begin
        orbits = _fold_boundary(Float64, rect, 64, XAxisReflection(), (-1,))
        @test sw_same_orbits(orbits, BilliardGeometry.symmetry_index_orbits(Float64, rect, 64, XAxisReflection(), -1.0 + 0.0im))
        @test orbits.full_size == 64 && orbits.fundamental_size == 32       # a reflection pairs the nodes
        @test sort(real.(unique(orbits.phase))) == [-1.0, 1.0]
        @test all(iszero, imag.(orbits.phase))
        @test size(orbits.fund_to_full) == (2, 32)
        @test _fold_boundary(Float64, rect, 64, XAxisReflection(), (1.0,)).phase == ones(ComplexF64, 64)   # trivial character
    end

    @testset "single reflection: node count must be a multiple of 4" begin
        @test_throws ArgumentError _fold_boundary(Float64, rect, 30, XAxisReflection(), (1,))
        @test_throws ArgumentError _fold_boundary(Float64, rect, 30, YAxisReflection(), (1,))
    end

    @testset "XYAxisReflection takes two characters" begin
        orbits = _fold_boundary(Float64, rect, 64, XYAxisReflection(), (-1, 1))
        @test sw_same_orbits(orbits, BilliardGeometry.symmetry_index_orbits(Float64, rect, 64, XYAxisReflection(), -1.0 + 0.0im, 1.0 + 0.0im))
        @test orbits.fundamental_size == 16                                 # group of order 4
        @test size(orbits.fund_to_full) == (4, 16)
        @test_throws MethodError _fold_boundary(Float64, rect, 64, XYAxisReflection(), (1, 1, 1))
    end

    @testset "CompositeReflection takes its characters as one vector" begin
        solver = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, -1))
        sym, chars = solver.symmetry, solver.sym_characters
        orbits = _fold_boundary(Float64, rect, 64, sym, chars)
        @test sw_same_orbits(orbits, BilliardGeometry.symmetry_index_orbits(Float64, rect, 64, sym, ComplexF64[chars...]))
        @test orbits.fundamental_size == 16
        @test sort(unique(real.(orbits.phase))) == [-1.0, 1.0]
        @test_throws DimensionMismatch _fold_boundary(Float64, rect, 64, sym, (1,))          # one character per reflection
    end

    @testset "NFoldRotation takes an integer sector" begin
        solver = DoubleLayerPotentialSolver(10.0, c3; sym_characters=(1,))
        rot = solver.symmetry
        @test rot isa NFoldRotation
        N = 60
        orbits = _fold_boundary(Float64, c3, N, rot, (1,))
        @test sw_same_orbits(orbits, BilliardGeometry.symmetry_index_orbits(Float64, c3, N, rot, 1))
        @test orbits.fundamental_size == N ÷ 3
        @test size(orbits.fund_to_full) == (3, N ÷ 3)
        @test sort(unique(round.(orbits.phase; digits=12)), by=z -> (real(z), imag(z))) ≈ sort([1.0, cis(2π / 3), cis(-2π / 3)], by=z -> (real(z), imag(z)))

        @test sw_same_orbits(_fold_boundary(Float64, c3, N, rot, ()),                       # empty character is the trivial sector
                             BilliardGeometry.symmetry_index_orbits(Float64, c3, N, rot))
        @test _fold_boundary(Float64, c3, N, rot, ()).phase == ones(ComplexF64, N)
        @test sw_same_orbits(_fold_boundary(Float64, c3, N, rot, (0,)), _fold_boundary(Float64, c3, N, rot, ()))
        @test !(_fold_boundary(Float64, c3, N, rot, (2,)).phase ≈ _fold_boundary(Float64, c3, N, rot, (1,)).phase)
    end

    @testset "NFoldRotation: validation" begin
        rot = DoubleLayerPotentialSolver(10.0, c3; sym_characters=(1,)).symmetry
        @test_throws ArgumentError _fold_boundary(Float64, c3, 60, rot, (1, 1))              # exactly one sector
        @test_throws ArgumentError _fold_boundary(Float64, c3, 60, rot, (1.0,))              # the sector must be an Int
        @test_throws ArgumentError _fold_boundary(Float64, c3, 62, rot, (1,))                # N must be a multiple of the order
    end

    @testset "Float32 node maps" begin
        orbits = _fold_boundary(Float32, rect, 64, XAxisReflection(), (-1,))
        @test orbits isa BilliardGeometry.SymmetryOrbitMap{Float32}
        @test eltype(orbits.phase) == ComplexF32
    end
end

@testset "_orbit_character_args" begin
    @testset "ordinary symmetries: scalars converted to Complex{T}" begin
        args = _orbit_character_args(Float64, XAxisReflection(), (-1,))
        @test args == (-1.0 + 0.0im,)
        @test args isa Tuple{ComplexF64}
        args2 = _orbit_character_args(Float64, XYAxisReflection(), (-1, 1))
        @test args2 == (-1.0 + 0.0im, 1.0 + 0.0im)
        @test args2 isa Tuple{ComplexF64,ComplexF64}
        @test _orbit_character_args(Float32, YAxisReflection(), (1,)) isa Tuple{ComplexF32}
        @test _orbit_character_args(Float64, XAxisReflection(), ()) == ()
    end

    @testset "CompositeReflection: one Vector of characters" begin
        sym = CompositeReflection(YAxisReflection(), XAxisReflection())
        args = _orbit_character_args(Float64, sym, (1, -1))
        @test length(args) == 1
        @test args[1] isa Vector{ComplexF64}
        @test args[1] == ComplexF64[1, -1]
        @test _orbit_character_args(Float32, sym, (1, -1))[1] isa Vector{ComplexF32}
    end

    @testset "NFoldRotation: () or (sector,)" begin
        rot = NFoldRotation(3, 1)
        @test _orbit_character_args(Float64, rot, ()) == ()
        @test _orbit_character_args(Float64, rot, (2,)) == (2,)
        @test _orbit_character_args(Float64, rot, (0,)) == (0,)
        @test only(_orbit_character_args(Float32, rot, (1,))) isa Int
        @test_throws ArgumentError _orbit_character_args(Float64, rot, (1.0,))         # the sector must be an Int
        @test_throws ArgumentError _orbit_character_args(Float64, rot, (1.0 + 0im,))
        @test_throws ArgumentError _orbit_character_args(Float64, rot, (1, 2))         # more than one rotation character
        @test_throws "exactly one sector" _orbit_character_args(Float64, rot, (1, 2))
        @test_throws "must be an Int" _orbit_character_args(Float64, rot, (1.0,))
    end
end

@testset "_fold_boundary (equals symmetry_index_orbits)" begin
    rect = RectangleBilliard(1.0, 0.7)
    c3 = C3Billiard(0.02)
    for N in (32, 64)
        @test sw_same_orbits(_fold_boundary(Float64, rect, N, YAxisReflection(), (1,)),
                             BilliardGeometry.symmetry_index_orbits(Float64, rect, N, YAxisReflection(), 1.0 + 0.0im))
        @test sw_same_orbits(_fold_boundary(Float64, rect, N, XYAxisReflection(), (1, -1)),
                             BilliardGeometry.symmetry_index_orbits(Float64, rect, N, XYAxisReflection(), 1.0 + 0.0im, -1.0 + 0.0im))
        comp = CompositeReflection(YAxisReflection(), XAxisReflection())
        @test sw_same_orbits(_fold_boundary(Float64, rect, N, comp, (-1, 1)),
                             BilliardGeometry.symmetry_index_orbits(Float64, rect, N, comp, ComplexF64[-1, 1]))
    end
    @test sw_same_orbits(_fold_boundary(Float64, c3, 60, NFoldRotation(3, 1), (2,)),
                         BilliardGeometry.symmetry_index_orbits(Float64, c3, 60, NFoldRotation(3, 1), 2))
end

@testset "_fold_group" begin
    rect = RectangleBilliard(1.0, 0.7)
    c3 = C3Billiard(0.02)

    @testset "group 0 of a billiard without holes equals _fold_boundary" begin
        N = 64
        comp = CompositeReflection(YAxisReflection(), XAxisReflection())
        @test sw_same_orbits(_fold_group(Float64, rect, 0, N, XAxisReflection(), (-1,)), _fold_boundary(Float64, rect, N, XAxisReflection(), (-1,)))
        @test sw_same_orbits(_fold_group(Float64, rect, 0, N, XYAxisReflection(), (-1, 1)), _fold_boundary(Float64, rect, N, XYAxisReflection(), (-1, 1)))
        @test sw_same_orbits(_fold_group(Float64, rect, 0, N, comp, (1, -1)), _fold_boundary(Float64, rect, N, comp, (1, -1)))
        @test sw_same_orbits(_fold_group(Float64, c3, 0, 60, NFoldRotation(3, 1), (1,)), _fold_boundary(Float64, c3, 60, NFoldRotation(3, 1), (1,)))
        @test sw_same_orbits(_fold_group(Float64, c3, 0, 60, NFoldRotation(3, 1), ()), _fold_boundary(Float64, c3, 60, NFoldRotation(3, 1), ()))
    end

    @testset "hole group of four discs placed by D2" begin
        b = hf_symmetrized_disc_billiard()
        comp = CompositeReflection(YAxisReflection(), XAxisReflection())
        orbits = _fold_group(Float64, b, 1, 24, comp, (1, -1))
        @test sw_same_orbits(orbits, BilliardGeometry.group_symmetry_orbits(Float64, b, 1, 24, comp, ComplexF64[1, -1]))
        @test full_size(orbits) == 24 && orbit_size(orbits) == 4 && fundamental_size(orbits) == 6
        # the outer boundary is group 0, folded by the same group
        outer = _fold_group(Float64, b, 0, 32, comp, (-1, -1))
        @test orbit_size(outer) == 4 && fundamental_size(outer) == 8
        # a single generator of G'
        x = _fold_group(Float64, b, 1, 24, XAxisReflection(), (-1,))
        @test orbit_size(x) == 2 && fundamental_size(x) == 12
        @test sort(unique(real.(x.phase))) == [-1.0, 1.0]
    end

    @testset "quarter-disc hole: G' = {X}" begin
        b = hf_quarter_disc_billiard()
        orbits = _fold_group(Float64, b, 1, 20, XAxisReflection(), (-1,))
        @test orbit_size(orbits) == 2 && fundamental_size(orbits) == 10 && full_size(orbits) == 20
        @test_throws ArgumentError _fold_group(Float64, b, 1, 20, YAxisReflection(), (1,))      # Y is not in G'
        @test_throws "G'" _fold_group(Float64, b, 1, 20, YAxisReflection(), (1,))
    end
end

@testset "_bim_node_multiple" begin
    @test _bim_node_multiple(nothing) == 1
    @test _bim_node_multiple(XAxisReflection()) == 4
    @test _bim_node_multiple(YAxisReflection()) == 4
    @test _bim_node_multiple(DiagonalReflection()) == 8
    @test _bim_node_multiple(AntiDiagonalReflection()) == 8
    @test _bim_node_multiple(XYAxisReflection()) == 4
    # a mirror at a generic angle imposes nothing here (the composite path supplies the block count)
    @test _bim_node_multiple(AxisReflection{1 // 6}()) == 1
    # a composite takes the lcm of its reflections
    @test _bim_node_multiple(CompositeReflection(YAxisReflection(), XAxisReflection())) == 4
    @test _bim_node_multiple(CompositeReflection(YAxisReflection(), DiagonalReflection())) == 8
    @test _bim_node_multiple(CompositeReflection(AxisReflection{1 // 6}(), XAxisReflection())) == 4
    @test _bim_node_multiple(CompositeReflection(AxisReflection{1 // 6}(), AxisReflection{1 // 3}())) == 1
    # a rotation needs N divisible by its order
    @test _bim_node_multiple(NFoldRotation(3, 1)) == 3
    @test _bim_node_multiple(NFoldRotation(4, 1)) == 4
    @test _bim_node_multiple(NFoldRotation(6, 5)) == 6
    @test (@inferred _bim_node_multiple(XAxisReflection())) isa Int
    @test (@inferred _bim_node_multiple(nothing)) isa Int
end

@testset "_bim_solver_node_multiple" begin
    rect = RectangleBilliard(1.0, 0.7)
    stadium = StadiumBilliard(1.0)
    c3 = C3Billiard(0.02)
    # no symmetry: no constraint
    @test _bim_solver_node_multiple(DoubleLayerPotentialSolver(5.0, rect; min_pts = 33)) == 1
    # Composite (Y, X) on the rectangle (D2, three registered elements): lcm(4, 3 + 1) = 4
    solver = DoubleLayerPotentialSolver(5.0, rect; min_pts = 33, sym_characters = (1, 1))
    @test _bim_solver_node_multiple(solver) == 4
    @test _bim_solver_node_multiple(solver) == lcm(_bim_node_multiple(solver.symmetry), length(rect.symmetries) + 1)
    # single reflection on a hole-free billiard: lcm(4, number of blocks)
    solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts = 33, symmetry = YAxisReflection(), sym_characters = (1,))
    @test _bim_solver_node_multiple(solver) == lcm(4, length(stadium.symmetries) + 1)
    # rotation: lcm(3, 3)
    solver = DoubleLayerPotentialSolver(5.0, c3; min_pts = 33, sym_characters = (1,))
    @test _bim_solver_node_multiple(solver) == lcm(3, length(c3.symmetries) + 1) == 3
    @test (@inferred _bim_solver_node_multiple(solver)) isa Int
end

@testset "_bim_round_nodes" begin
    # rounds up to a multiple of `needed`
    @test _bim_round_nodes(30, 4, nothing) == 32
    @test _bim_round_nodes(32, 4, nothing) == 32
    @test _bim_round_nodes(33, 4, nothing) == 36
    @test _bim_round_nodes(1, 7, nothing) == 7
    @test _bim_round_nodes(30, 1, nothing) == 30
    @test _bim_round_nodes(15, 6, nothing) == 18
    # a forced node count is returned as given, whatever N and needed
    @test _bim_round_nodes(30, 4, 50) == 50
    @test _bim_round_nodes(30, 4, 17) == 17
    @test _bim_round_nodes(100, 8, 24) == 24
    @test (@inferred _bim_round_nodes(30, 4, nothing)) isa Int
    @test (@inferred _bim_round_nodes(30, 4, 50)) isa Int
end

@testset "_infer_bim_symmetry" begin
    rect = RectangleBilliard(1.0, 0.7)
    c3 = C3Billiard(0.02)

    @testset "no characters: no symmetry" begin
        @test _infer_bim_symmetry(rect, ()) === (nothing, ())
        @test _infer_bim_symmetry(c3, ()) === (nothing, ())
    end

    @testset "reflections are wrapped in CompositeReflection" begin
        symmetry, chars = _infer_bim_symmetry(rect, (1, -1))
        gens = BilliardGeometry.get_symmetries(rect)
        @test symmetry isa CompositeReflection
        @test length(symmetry.reflections) == length(gens) == 2
        @test chars isa Tuple{ComplexF64,ComplexF64}
        # characters are paired with get_symmetries(rect) = (YAxisReflection, XAxisReflection) in order
        @test gens[1] isa YAxisReflection && gens[2] isa XAxisReflection
        ix = findfirst(r -> r isa XAxisReflection, symmetry.reflections)
        iy = findfirst(r -> r isa YAxisReflection, symmetry.reflections)
        @test chars[iy] == 1.0 && chars[ix] == -1.0
        symmetry2, chars2 = _infer_bim_symmetry(rect, (-1, 1))
        ix2 = findfirst(r -> r isa XAxisReflection, symmetry2.reflections)
        iy2 = findfirst(r -> r isa YAxisReflection, symmetry2.reflections)
        @test chars2[iy2] == -1.0 && chars2[ix2] == 1.0
    end

    @testset "a lone rotation generator is returned directly" begin
        symmetry, chars = _infer_bim_symmetry(c3, (1,))
        @test symmetry isa NFoldRotation
        @test symmetry.order == 3
        @test chars === (1,)                                                # the integer sector is kept as given
        @test _infer_bim_symmetry(c3, (0,))[2] === (0,)
        @test _infer_bim_symmetry(c3, (2,))[2] === (2,)
    end

    @testset "validation" begin
        @test_throws ArgumentError _infer_bim_symmetry(rect, (1,))                           # one character per generator
        @test_throws ArgumentError _infer_bim_symmetry(rect, (1, 1, 1))
        @test_throws ArgumentError _infer_bim_symmetry(rect, (2, 1))                         # reflection characters are +-1
        @test_throws ArgumentError _infer_bim_symmetry(c3, (1, 1))
        @test_throws ArgumentError _infer_bim_symmetry(c3, (3,))                             # sector outside 0:order-1
        @test_throws MethodError _infer_bim_symmetry(rect, (1.0, 1.0))                       # characters must be integers
    end
end

@testset "symmetrize_layer_density" begin
    rect = RectangleBilliard(1.0, 0.7)
    c3 = C3Billiard(0.02)

    @testset "reflection sector: orbit identity and shapes" begin
        solver = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, -1))
        pts = evaluate_points(solver, rect, 10.0)
        N = length(pts)
        orbits = _fold_boundary(Float64, rect, N, solver.symmetry, solver.sym_characters)
        reduced = collect(1.0:orbits.fundamental_size)
        full = symmetrize_layer_density(solver, reduced, pts, rect)
        @test length(full) == N
        @test boundary_matrix_size(solver, pts) == orbits.fundamental_size == N ÷ 4
        @test full isa Vector{ComplexF64}
        @test full == orbits.phase .* reduced[orbits.orbit_of]
        @test sw_orbit_identity_holds(orbits, full, reduced)
        @test full[orbits.fundamental_indices] == reduced                    # representatives keep their values
    end

    @testset "reflection sector: an already full density is returned unchanged" begin
        solver = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, 1))
        pts = evaluate_points(solver, rect, 10.0)
        full = ComplexF64.(1:length(pts))
        @test symmetrize_layer_density(solver, full, pts, rect) === full
    end

    @testset "rotation sector: orbit identity with the cube roots of unity" begin
        solver = DoubleLayerPotentialSolver(10.0, c3; sym_characters=(1,))
        pts = evaluate_points(solver, c3, 10.0)
        N = length(pts)
        orbits = _fold_boundary(Float64, c3, N, solver.symmetry, solver.sym_characters)
        reduced = ComplexF64[b + 0.5im * b for b in 1:orbits.fundamental_size]
        full = symmetrize_layer_density(solver, reduced, pts, c3)
        @test length(full) == N && orbits.fundamental_size == N ÷ 3
        @test sw_orbit_identity_holds(orbits, full, reduced)
        @test full ≈ orbits.phase .* reduced[orbits.orbit_of]
        @test all(abs.(orbits.phase) .≈ 1.0)
    end

    @testset "element types: the result is complex and at least of the solver precision" begin
        solver = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, 1))
        pts = evaluate_points(solver, rect, 10.0)
        nred = length(pts) ÷ 4
        @test symmetrize_layer_density(solver, ones(nred), pts, rect) isa Vector{ComplexF64}
        @test symmetrize_layer_density(solver, ones(Float32, nred), pts, rect) isa Vector{ComplexF64}
        @test symmetrize_layer_density(solver, ones(ComplexF64, nred), pts, rect) isa Vector{ComplexF64}
    end

    @testset "length validation" begin
        solver = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, 1))
        pts = evaluate_points(solver, rect, 10.0)
        nred = length(pts) ÷ 4
        @test_throws DimensionMismatch symmetrize_layer_density(solver, ones(nred - 1), pts, rect)
        @test_throws DimensionMismatch symmetrize_layer_density(solver, ones(nred + 1), pts, rect)
        @test_throws DimensionMismatch symmetrize_layer_density(solver, ones(length(pts) + 1), pts, rect)

        plain = DoubleLayerPotentialSolver(10.0, rect)
        plain_pts = evaluate_points(plain, rect, 10.0)
        @test symmetrize_layer_density(plain, ones(length(plain_pts)), plain_pts, rect) == ones(length(plain_pts))
        @test_throws DimensionMismatch symmetrize_layer_density(plain, ones(nred), plain_pts, rect)   # nothing to expand without a symmetry
    end
end

@testset "_bim_normal_derivative (DLP)" begin
    rect = RectangleBilliard(1.0, 0.7)

    @testset "without symmetry: conj(lvec) / ds on the whole boundary" begin
        solver = DoubleLayerPotentialSolver(10.0, rect)
        pts = evaluate_points(solver, rect, 10.0)
        lvec = ComplexF64[(1 + 2im) * j for j in 1:length(pts)]
        u = _bim_normal_derivative(solver, pts, lvec, rect)
        @test length(u) == length(pts)
        @test u == conj.(lvec) ./ pts.ds
        @test _bim_normal_derivative(solver, pts, 2 .* lvec, rect) ≈ 2 .* u         # linear (antilinear in the conjugate) in lvec
    end

    @testset "with symmetry: weights of the orbit representatives" begin
        solver = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, -1))
        pts = evaluate_points(solver, rect, 10.0)
        orbits = _fold_boundary(Float64, rect, length(pts), solver.symmetry, solver.sym_characters)
        lvec = ComplexF64[(1 - 1im) * j for j in 1:orbits.fundamental_size]
        u = _bim_normal_derivative(solver, pts, lvec, rect)
        @test length(u) == orbits.fundamental_size
        @test u == conj.(lvec) ./ pts.ds[orbits.fundamental_indices]
        # on a symmetric billiard the weights of all images of one node coincide, so any orbit member gives the same u
        @test all(pts.ds[orbits.fund_to_full[g, b]] ≈ pts.ds[orbits.fund_to_full[1, b]] for g in 1:4, b in 1:orbits.fundamental_size)
    end
end

@testset "_bim_dlp_cross_kernel_entry" begin
    circle = CircleBilliard(1.0)
    solver = DoubleLayerPotentialSolver(10.0, circle)
    pts = evaluate_points(solver, circle, 4.0)

    @testset "equals the closed-form double-layer entry" begin
        x, y, k, j = 3.0, 4.0, 2.0, 5
        xj, yj = pts.xy[j]
        tx, ty = pts.tangent[j]
        dx, dy = x - xj, y - yj
        r = hypot(dx, dy)
        expected = pts.ws[j] * im * k / 2 * (ty * dx - tx * dy) * hankelh1(1, k * r) / r
        @test _bim_dlp_cross_kernel_entry(pts, x, y, k, j) ≈ expected
        @test _bim_dlp_cross_kernel_entry(pts, x, y, k, j) isa ComplexF64
    end

    @testset "linear in the quadrature weight of the source node" begin
        j = 7
        doubled = BoundaryPoints(pts.xy; normal=pts.normal, s=pts.s, ds=pts.ds, tangent=pts.tangent,
                                 tangent_2=pts.tangent_2, ts=pts.ts, tphys=pts.tphys, ws=2 .* pts.ws, ws_der=pts.ws_der)
        a = _bim_dlp_cross_kernel_entry(pts, 2.5, -1.5, 3.0, j)
        @test _bim_dlp_cross_kernel_entry(doubled, 2.5, -1.5, 3.0, j) ≈ 2a
        @test abs(a) > 0
    end

    @testset "reversing the source tangent flips the sign" begin
        # (ty dx - tx dy) is odd in the tangent: the kernel is the normal derivative along n = (ty, -tx)/|t|
        j = 3
        xj, yj = pts.xy[j]
        tx, ty = pts.tangent[j]
        flipped = BoundaryPoints(pts.xy; normal=pts.normal, s=pts.s, ds=pts.ds, tangent=[-t for t in pts.tangent],
                                 tangent_2=pts.tangent_2, ts=pts.ts, tphys=pts.tphys, ws=pts.ws, ws_der=pts.ws_der)
        @test _bim_dlp_cross_kernel_entry(flipped, 3.0, 4.0, 2.0, j) ≈ -_bim_dlp_cross_kernel_entry(pts, 3.0, 4.0, 2.0, j)
    end

    @testset "complex wavenumber is accepted" begin
        v = _bim_dlp_cross_kernel_entry(pts, 3.0, 4.0, 2.0 + 0.1im, 5)
        @test v isa ComplexF64
        @test isfinite(v)
    end
end

@testset "_cfie_derivative!" begin
    N = 16
    t = [2π * (j - 0.5) / N for j in 1:N]
    du = zeros(ComplexF64, N)

    @testset "differentiates the Fourier modes exactly" begin
        for q in (1, 3, -2, 7)
            u = ComplexF64.(cis.(q .* t))
            out = _cfie_derivative!(du, u)
            @test out === du
            @test du ≈ im * q .* u atol = 1e-12
        end
    end

    @testset "constants have zero derivative" begin
        _cfie_derivative!(du, ones(ComplexF64, N))
        @test maximum(abs, du) < 1e-12
    end

    @testset "the Nyquist mode of an even grid is dropped" begin
        u = ComplexF64.(cis.(8 .* t))                                       # q = N/2
        _cfie_derivative!(du, u)
        @test maximum(abs, du) < 1e-12
    end

    @testset "odd grids keep the highest mode" begin
        n = 15
        tn = [2π * (j - 0.5) / n for j in 1:n]
        u = ComplexF64.(cis.(7 .* tn))
        dn = zeros(ComplexF64, n)
        _cfie_derivative!(dn, u)
        @test dn ≈ 7im .* u atol = 1e-11
    end

    @testset "linearity and the derivative of sin" begin
        a, b = 0.7, -1.3
        u1 = ComplexF64.(cos.(2 .* t))
        u2 = ComplexF64.(sin.(3 .* t))
        d1, d2, d12 = zeros(ComplexF64, N), zeros(ComplexF64, N), zeros(ComplexF64, N)
        _cfie_derivative!(d1, u1)
        _cfie_derivative!(d2, u2)
        _cfie_derivative!(d12, a .* u1 .+ b .* u2)
        @test d12 ≈ a .* d1 .+ b .* d2 atol = 1e-12
        @test real.(d1) ≈ -2 .* sin.(2 .* t) atol = 1e-12
        @test real.(d2) ≈ 3 .* cos.(3 .* t) atol = 1e-12
    end

    @testset "single precision" begin
        t32 = Float32[2π * (j - 0.5) / N for j in 1:N]
        u = ComplexF32.(cis.(3 .* t32))
        d32 = zeros(ComplexF32, N)
        _cfie_derivative!(d32, u)
        @test d32 isa Vector{ComplexF32}
        @test d32 ≈ 3im .* u atol = 1e-4
    end
end
