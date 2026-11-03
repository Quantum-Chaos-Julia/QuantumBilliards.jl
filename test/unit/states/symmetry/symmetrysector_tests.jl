# Unit tests for
#   src/states/symmetry/symmetrysector.jl   (SymmetrySector, symmetry_sector (pair form and positional form),
#                                            _sector_character, _check_sector_billiard, _resolve_bim_symmetry,
#                                            _match_rotation_sector, _rotation_subgroup_order, and the reflection / rotation
#                                            resolution helpers)
# Billiards with holes: the sector reads the symmetry group G' of the billiard (BilliardGeometry.get_symmetry_group), not the outer
# registry G. A generator registered in G but not in G' raises an ArgumentError naming G'; rotation sectors range over 0:d-1 with d
# the order of the rotation subgroup of G'. The hole fixtures are the D2 quarter-square billiard with a hole that keeps only X
# (the D2 disc about (0.5, 0)) or nothing (a DomainHole); a registry {NFoldRotation(6,2), NFoldRotation(6,4)} (C3 inside C6) checks
# the subgroup sector range and the folded phases chi_s(g^m) = exp(2 pi i s m / 6).
# The symmetry generators, the registry and the orbit maps belong to BilliardGeometry; here they are only used as
# inputs (RectangleBilliard: D2 reflections, StarBilliard: cyclic rotations, a circular PolarBilliard: a single
# diagonal reflection, TriangleBilliard: no symmetry) and the characters recovered from them are compared with the
# closed forms chi(g) = +-1 and chi_s(g^m) = exp(2 pi i s m / N).
# No eigen solves, no matrix assembly. The only cross-package check is the final testset, which compares the orbit
# folding of BilliardGeometry with the plane-wave pattern count of RealPlaneWaves for the same sector.
# Helper functions use the prefix `ss_`.

using StaticArrays: SVector
using QuantumBilliards: _sector_character, _match_rotation_sector, _resolve_bim_symmetry, _check_sector_billiard, _rotation_subgroup_order

# Vector of generator => character pairs in the exact element type consumed by the vector method of
# `_resolve_bim_symmetry`.
ss_gens(ps...) = Pair{BilliardGeometry.AbsSymmetry,ComplexF64}[ps...]

# Map from reflection generator type to its character for a resolved `(CompositeReflection, characters)` pair.
ss_char_by_type(gen::CompositeReflection, chars) = Dict(typeof(r) => c for (r, c) in zip(gen.reflections, chars))

# Unit circle whose registry holds the diagonal reflection alone (no shipped billiard registers it); only the registry is
# inspected by the sector resolution, the shape is irrelevant.
function ss_diagonal_circle()
    seg = FourierCoeffPolarSegment(Float64[]; shift_angle = pi / 4)
    dom = BilliardGeometry.PolarDomain{Float64}([seg], [curve(seg, 0.0)], 1)
    return PolarBilliard{Float64}(dom, register_symmetries(DiagonalReflection()))
end

# Whether the cyclic sector `s` of `billiard` resolves without error to exactly the sector index `s`.
function ss_resolves_to_sector(billiard, s)
    try
        _, chars = _resolve_bim_symmetry(billiard, symmetry_sector(billiard, NFoldRotation => s))
        return chars == (s,)
    catch err
        err isa ArgumentError || rethrow()
        return false
    end
end

@testset "SymmetrySector struct" begin
    rect = RectangleBilliard(1.0, 0.6)
    chars = Dict{Int,ComplexF64}(1 => 1, 3 => -1)
    sec = SymmetrySector(rect, chars)
    @test sec isa SymmetrySector{RectangleBilliard{Float64}}
    @test fieldnames(SymmetrySector) == (:billiard, :characters, :degeneracy)
    @test sec.degeneracy == 1                                              # two-argument constructor defaults to a 1D irrep
    @test sec.billiard === rect
    @test sec.characters === chars
    @test !isabstracttype(typeof(sec))
    @test isconcretetype(typeof(sec))
    @test SymmetrySector{RectangleBilliard{Float64}} <: SymmetrySector
end

@testset "_sector_character - reflection generators" begin
    @test _sector_character(XAxisReflection(1), 1) == 1
    @test _sector_character(XAxisReflection(1), -1) == -1
    @test _sector_character(YAxisReflection(2), 1) == 1
    @test _sector_character(YAxisReflection(2), -1) == -1
    @test _sector_character(XYAxisReflection(3), -1) == -1
    @test _sector_character(DiagonalReflection(1), -1) == -1
    @test _sector_character(AntiDiagonalReflection(1), 1) == 1
    @test _sector_character(CompositeReflection(XAxisReflection(1)), -1) == -1

    @testset "return type" begin
        @test @inferred(_sector_character(XAxisReflection(1), -1)) isa ComplexF64
        @test _sector_character(XAxisReflection(1), Int8(-1)) === ComplexF64(-1)
        @test imag(_sector_character(YAxisReflection(1), -1)) == 0
    end

    @testset "invalid characters" begin
        @test_throws ArgumentError _sector_character(XAxisReflection(1), 0)
        @test_throws ArgumentError _sector_character(XAxisReflection(1), 2)
        @test_throws ArgumentError _sector_character(YAxisReflection(1), -2)
        @test_throws ArgumentError _sector_character(DiagonalReflection(1), 3)
        @test_throws MethodError _sector_character(XAxisReflection(1), 1.0)     # characters must be integers
        @test_throws MethodError _sector_character(XAxisReflection(1), 1 + 0im)
        err = try _sector_character(XAxisReflection(1), 2) catch e; e end
        @test occursin("±1", err.msg) && occursin("2", err.msg)
    end
end

@testset "_sector_character - rotation generators" begin
    @testset "m = 1, N = 4: powers of i" begin
        g = NFoldRotation(4, 1, 1)
        @test _sector_character(g, 0) ≈ 1
        @test _sector_character(g, 1) ≈ im
        @test _sector_character(g, 2) ≈ -1
        @test _sector_character(g, 3) ≈ -im
    end

    @testset "higher powers" begin
        # chi_s(g^m) = chi_s(g)^m is a homomorphism on the cyclic group.
        N = 5
        g1 = NFoldRotation(N, 1, 1)
        for m in 1:N-1, s in 0:N-1
            gm = NFoldRotation(N, m, m)
            @test _sector_character(gm, s) ≈ _sector_character(g1, s)^m atol = 1e-13
        end
        @test _sector_character(NFoldRotation(4, 2, 2), 1) ≈ -1 atol = 1e-13
        @test _sector_character(NFoldRotation(4, 3, 3), 1) ≈ -im atol = 1e-13
    end

    @testset "unit modulus and trivial sector" begin
        @test all(s -> abs(_sector_character(NFoldRotation(7, 3, 3), s)) ≈ 1, 0:6)
        @test _sector_character(NFoldRotation(7, 3, 3), 0) == 1
        @test _sector_character(NFoldRotation(7, 7, 7), 4) == 1                 # m is reduced mod N, so g^7 is the identity
    end

    @testset "return type" begin
        @test @inferred(_sector_character(NFoldRotation(3, 1, 1), 2)) isa ComplexF64
    end

    @testset "invalid sectors" begin
        @test_throws ArgumentError _sector_character(NFoldRotation(4, 1, 1), -1)
        @test_throws ArgumentError _sector_character(NFoldRotation(4, 1, 1), 4)
        @test_throws ArgumentError _sector_character(NFoldRotation(3, 2, 2), 3)
        @test_throws MethodError _sector_character(NFoldRotation(4, 1, 1), 1.0)
        err = try _sector_character(NFoldRotation(4, 1, 1), 4) catch e; e end
        @test occursin("0:3", err.msg)
    end
end

@testset "_match_rotation_sector" begin
    @testset "round trip with _sector_character" begin
        N = 5                                                                     # prime, so s -> s*m mod N is injective for every m
        for m in 1:N-1
            g = NFoldRotation(N, m, m)
            @test [_match_rotation_sector(g, _sector_character(g, s)) for s in 0:N-1] == collect(0:N-1)
        end
    end

    @testset "m = 1 closed forms" begin
        g = NFoldRotation(4, 1, 1)
        @test _match_rotation_sector(g, ComplexF64(1)) == 0
        @test _match_rotation_sector(g, ComplexF64(im)) == 1
        @test _match_rotation_sector(g, ComplexF64(-1)) == 2
        @test _match_rotation_sector(g, ComplexF64(-im)) == 3
    end

    @testset "m not coprime to N: search returns the smallest matching sector" begin
        # chi_s(g^2) for N = 4 is exp(i pi s), so s = 0, 2 give +1 and s = 1, 3 give -1.
        g = NFoldRotation(4, 2, 2)
        @test _match_rotation_sector(g, ComplexF64(1)) == 0
        @test _match_rotation_sector(g, ComplexF64(-1)) == 1
        @test_throws ArgumentError _match_rotation_sector(g, ComplexF64(im))   # i is not a character value of g^2
    end

    @testset "tolerance" begin
        g = NFoldRotation(3, 1, 1)
        chi = _sector_character(g, 1)
        @test _match_rotation_sector(g, chi * (1 + 1e-12)) == 1                # within atol = 1e-9
        @test_throws ArgumentError _match_rotation_sector(g, chi * (1 + 1e-6))
    end

    @testset "invalid characters" begin
        g = NFoldRotation(4, 1, 1)
        @test_throws ArgumentError _match_rotation_sector(g, ComplexF64(0.5))
        @test_throws ArgumentError _match_rotation_sector(g, ComplexF64(0))
        @test_throws ArgumentError _match_rotation_sector(g, ComplexF64(2))     # right phase, wrong modulus
        @test_throws ArgumentError _match_rotation_sector(g, cis(2pi / 5))      # N = 5 phase for an N = 4 group
        err = try _match_rotation_sector(g, ComplexF64(0.5)) catch e; e end
        @test occursin("order=4", err.msg) && occursin("m=1", err.msg)
    end

    @testset "return type" begin
        @test @inferred(_match_rotation_sector(NFoldRotation(3, 1, 1), ComplexF64(1))) isa Int
    end
end

@testset "symmetry_sector pairs - reflection billiard (Rectangle, D2)" begin
    rect = RectangleBilliard(1.0, 0.6)
    # RectangleBilliard registers (sym_id 1, 2, 3) = (YAxisReflection, XYAxisReflection, XAxisReflection).

    @testset "both axes" begin
        sec = symmetry_sector(rect, XAxisReflection => -1, YAxisReflection => 1)
        @test sec isa SymmetrySector{RectangleBilliard{Float64}}
        @test sec.billiard === rect
        @test Set(keys(sec.characters)) == Set([1, 3])
        @test sec.characters[3] == -1                                          # XAxisReflection
        @test sec.characters[1] == 1                                           # YAxisReflection
        @test @inferred(symmetry_sector(rect, XAxisReflection => -1)) isa SymmetrySector
    end

    @testset "order of pairs is irrelevant" begin
        a = symmetry_sector(rect, XAxisReflection => -1, YAxisReflection => 1)
        b = symmetry_sector(rect, YAxisReflection => 1, XAxisReflection => -1)
        @test a.characters == b.characters
    end

    @testset "single generator" begin
        sec = symmetry_sector(rect, YAxisReflection => -1)
        @test sec.characters == Dict{Int,ComplexF64}(1 => -1)
        sec = symmetry_sector(rect, XAxisReflection => 1)
        @test sec.characters == Dict{Int,ComplexF64}(3 => 1)
        sec = symmetry_sector(rect, XYAxisReflection => -1)
        @test sec.characters == Dict{Int,ComplexF64}(2 => -1)
    end

    @testset "empty choice" begin
        sec = symmetry_sector(rect)
        @test isempty(sec.characters)
        @test sec.billiard === rect
    end

    @testset "abstract generator type selects every matching generator" begin
        sec = symmetry_sector(rect, BilliardGeometry.AbsReflection => -1)
        @test Set(keys(sec.characters)) == Set([1, 2, 3])
        @test all(==(-1), values(sec.characters))
    end

    @testset "all four D2 sectors" begin
        for (cx, cy) in ((1, 1), (-1, 1), (1, -1), (-1, -1))
            sec = symmetry_sector(rect, XAxisReflection => cx, YAxisReflection => cy)
            @test sec.characters[3] == cx
            @test sec.characters[1] == cy
        end
    end
end

@testset "symmetry_sector pairs - rotation billiard (Star, D3)" begin
    star = StarBilliard(1.0, 0.3, 3)
    # StarBilliard registers Dn_symmetry(3): the rotations NFoldRotation(3, m) have sym_id = 2m = 2, 4 (odd ids are mirrors).
    @testset "sector s gives chi_s(g^m) = exp(2 pi i s m / 3)" begin
        for s in 0:2
            sec = symmetry_sector(star, NFoldRotation => s)
            @test sec.billiard === star
            @test Set(keys(sec.characters)) == Set([2, 4])
            @test sec.characters[2] ≈ cis(2pi * s * 1 / 3) atol = 1e-13
            @test sec.characters[4] ≈ cis(2pi * s * 2 / 3) atol = 1e-13
            @test sec.characters[4] ≈ sec.characters[2]^2 atol = 1e-13
        end
        @test symmetry_sector(star, NFoldRotation => 0).characters == Dict{Int,ComplexF64}(2 => 1, 4 => 1)
    end

    @testset "abstract rotation type" begin
        sec = symmetry_sector(star, BilliardGeometry.AbsRotation => 1)
        @test sec.characters == symmetry_sector(star, NFoldRotation => 1).characters
    end

    @testset "invalid sector" begin
        @test_throws ArgumentError symmetry_sector(star, NFoldRotation => 3)
        @test_throws ArgumentError symmetry_sector(star, NFoldRotation => -1)
    end
end

@testset "symmetry_sector pairs - single diagonal reflection" begin
    sq = ss_diagonal_circle()
    sec = symmetry_sector(sq, DiagonalReflection => -1)
    @test sec.characters == Dict{Int,ComplexF64}(1 => -1)
    @test_throws ArgumentError symmetry_sector(sq, XAxisReflection => 1)
end

@testset "symmetry_sector pairs - error paths" begin
    rect = RectangleBilliard(1.0, 0.6)
    star = StarBilliard(1.0, 0.3, 3)
    tri = TriangleBilliard(pi / 2, pi / 3)

    @testset "generator not registered by the billiard" begin
        @test_throws ArgumentError symmetry_sector(tri, XAxisReflection => 1)
        @test_throws ArgumentError symmetry_sector(rect, NFoldRotation => 1)
        @test_throws ArgumentError symmetry_sector(rect, DiagonalReflection => 1)
        @test_throws ArgumentError symmetry_sector(star, DiagonalReflection => 1)   # the D3 mirrors are not the diagonal
        err = try symmetry_sector(tri, XAxisReflection => 1) catch e; e end
        @test occursin("XAxisReflection", err.msg)
    end

    @testset "invalid reflection characters" begin
        @test_throws ArgumentError symmetry_sector(rect, XAxisReflection => 0)
        @test_throws ArgumentError symmetry_sector(rect, XAxisReflection => 2)
        @test_throws ArgumentError symmetry_sector(rect, XAxisReflection => 1, YAxisReflection => -3)
        @test_throws MethodError symmetry_sector(rect, XAxisReflection => 1.0)
    end

    @testset "duplicate generator types" begin
        @test_throws ArgumentError symmetry_sector(rect, XAxisReflection => 1, XAxisReflection => -1)
        @test_throws ArgumentError symmetry_sector(rect, XAxisReflection => 1, XAxisReflection => 1)
        err = try symmetry_sector(rect, XAxisReflection => 1, XAxisReflection => -1) catch e; e end
        @test occursin("duplicate", err.msg)
    end

    @testset "an invalid later pair is not masked by a valid first pair" begin
        # The valid first pair must not mask the invalid second one.
        @test_throws ArgumentError symmetry_sector(rect, YAxisReflection => 1, XAxisReflection => 5)
    end
end

@testset "symmetry_sector positional form" begin
    rect = RectangleBilliard(1.0, 0.6)
    star = StarBilliard(1.0, 0.3, 3)
    tri = TriangleBilliard(pi / 2, pi / 3)

    @testset "D2: characters follow get_symmetries order (YAxis, XAxis)" begin
        @test typeof.(BilliardGeometry.get_symmetries(rect)) == (YAxisReflection, XAxisReflection)
        sec = symmetry_sector(rect, -1, 1)
        @test sec.billiard === rect
        @test sec.characters == symmetry_sector(rect, YAxisReflection => -1, XAxisReflection => 1).characters
        @test sec.characters[1] == -1
        @test sec.characters[3] == 1
        @test symmetry_sector(rect, 1, -1).characters == symmetry_sector(rect, YAxisReflection => 1, XAxisReflection => -1).characters
    end

    @testset "dihedral: one mirror character per generator" begin
        # D3 = Dn_symmetry(3): two mirror generators; the positional form assigns characters to the mirrors only
        gens = BilliardGeometry.get_symmetries(star)
        @test length(gens) == 2
        @test all(g -> g isa AxisReflection, gens)
        for c in (1, -1)
            sec = symmetry_sector(star, c, c)
            @test sec.billiard === star
            @test sec.characters == symmetry_sector(star, typeof(gens[1]) => c, typeof(gens[2]) => c).characters
            @test Set(keys(sec.characters)) == Set(g.sym_id for g in gens)
            @test all(==(c), values(sec.characters))
        end
    end

    @testset "wrong number of characters" begin
        @test_throws ArgumentError symmetry_sector(rect, 1)                    # D2 needs two
        @test_throws ArgumentError symmetry_sector(rect, 1, 1, 1)
        @test symmetry_sector(star, 1).degeneracy == 2                         # D3: a single integer is the 2D irrep E_1, not an error
        @test_throws ArgumentError symmetry_sector(star, 1, 1, 1)
        @test_throws ArgumentError symmetry_sector(tri, 1)                     # no generators at all
        err = try symmetry_sector(rect, 1) catch e; e end
        @test occursin("2 expected", err.msg) && occursin("received 1", err.msg)
    end

    @testset "invalid values are forwarded to the pair form" begin
        @test_throws ArgumentError symmetry_sector(rect, 0, 1)
        @test_throws ArgumentError symmetry_sector(rect, 1, 2)
        @test_throws ArgumentError symmetry_sector(star, 0, 1)
    end
end

@testset "symmetry_sector single integer and tuple forms" begin
    sinai = SinaiBilliard(1.0, 0.3)                                            # D4: two mirror generators, d = 4
    star = StarBilliard(1.0, 0.3, 3)                                           # D3
    c3 = C3Billiard(0.02)                                                      # one generator (rotation)

    @testset "single integer on D_d, d >= 3: irrep2d sector" begin
        for (b, k) in ((sinai, 1), (star, 1))
            sec = symmetry_sector(b, k)
            ref = symmetry_sector(b; irrep2d = k)
            @test sec.degeneracy == 2
            @test sec.characters == ref.characters
            @test sec.degeneracy == ref.degeneracy
        end
    end

    @testset "single integer on one-generator group: 1D sector" begin
        sec = symmetry_sector(c3, 1)
        @test sec.degeneracy == 1
        @test sec.characters == symmetry_sector(c3, NFoldRotation => 1).characters
    end

    @testset "two integers on D4: 1D sector" begin
        @test symmetry_sector(sinai, 1, -1).degeneracy == 1
    end

    @testset "tuple of integers equals splatted form" begin
        a = symmetry_sector(sinai, (1, -1))
        b = symmetry_sector(sinai, 1, -1)
        @test a.characters == b.characters
        @test a.degeneracy == b.degeneracy
        @test a.billiard === b.billiard
    end
end

@testset "_check_sector_billiard" begin
    rect = RectangleBilliard(1.0, 0.6)
    sec = symmetry_sector(rect, XAxisReflection => -1, YAxisReflection => 1)
    @test _check_sector_billiard(rect, sec) === nothing
    @test_throws ArgumentError _check_sector_billiard(RectangleBilliard(1.0, 0.6), sec)   # geometrically identical, different instance
    @test_throws ArgumentError _check_sector_billiard(StarBilliard(1.0, 0.3, 3), sec)
    err = try _check_sector_billiard(RectangleBilliard(1.0, 0.6), sec) catch e; e end
    @test occursin("different billiard instance", err.msg)
end

@testset "_resolve_bim_symmetry - empty sector" begin
    rect = RectangleBilliard(1.0, 0.6)
    tri = TriangleBilliard(pi / 2, pi / 3)
    @test _resolve_bim_symmetry(rect, symmetry_sector(rect)) == (nothing, ())
    @test _resolve_bim_symmetry(tri, symmetry_sector(tri)) == (nothing, ())
end

@testset "_resolve_bim_symmetry - reflection sectors (Rectangle)" begin
    rect = RectangleBilliard(1.0, 0.6)

    @testset "both axes" begin
        sec = symmetry_sector(rect, XAxisReflection => -1, YAxisReflection => 1)
        gen, chars = _resolve_bim_symmetry(rect, sec)
        @test gen isa CompositeReflection
        @test length(gen.reflections) == 2
        @test Set(typeof.(gen.reflections)) == Set([XAxisReflection, YAxisReflection])
        @test chars isa NTuple{2,ComplexF64}
        # The tuple is aligned with `gen.reflections` regardless of Dict iteration order.
        byt = ss_char_by_type(gen, chars)
        @test byt[XAxisReflection] == -1
        @test byt[YAxisReflection] == 1
        # The resolved generators are the ones registered by the billiard (same sym_id).
        @test Set(r.sym_id for r in gen.reflections) == Set([1, 3])
    end

    @testset "each sign combination of the D2 sector" begin
        for (cx, cy) in ((1, 1), (-1, 1), (1, -1), (-1, -1))
            sec = symmetry_sector(rect, XAxisReflection => cx, YAxisReflection => cy)
            gen, chars = _resolve_bim_symmetry(rect, sec)
            byt = ss_char_by_type(gen, chars)
            @test byt[XAxisReflection] == cx
            @test byt[YAxisReflection] == cy
        end
    end

    @testset "single generator" begin
        gen, chars = _resolve_bim_symmetry(rect, symmetry_sector(rect, YAxisReflection => -1))
        @test gen isa CompositeReflection
        @test length(gen.reflections) == 1
        @test gen.reflections[1] isa YAxisReflection
        @test chars == (-1.0 + 0.0im,)

        gen, chars = _resolve_bim_symmetry(rect, symmetry_sector(rect, XAxisReflection => 1))
        @test gen.reflections[1] isa XAxisReflection
        @test chars == (1.0 + 0.0im,)
    end

    @testset "lone XYAxisReflection is ambiguous" begin
        sec = symmetry_sector(rect, XYAxisReflection => -1)
        @test_throws ArgumentError _resolve_bim_symmetry(rect, sec)
        err = try _resolve_bim_symmetry(rect, sec) catch e; e end
        @test occursin("XYAxisReflection", err.msg)
    end

    @testset "positional sector resolves like the pair sector" begin
        gen_a, ch_a = _resolve_bim_symmetry(rect, symmetry_sector(rect, -1, 1))
        byt = ss_char_by_type(gen_a, ch_a)
        @test byt[YAxisReflection] == -1
        @test byt[XAxisReflection] == 1
    end
end

@testset "_resolve_bim_symmetry - diagonal reflection" begin
    sq = ss_diagonal_circle()
    gen, chars = _resolve_bim_symmetry(sq, symmetry_sector(sq, DiagonalReflection => -1))
    @test gen isa CompositeReflection
    @test length(gen.reflections) == 1
    @test gen.reflections[1] isa DiagonalReflection
    @test chars == (-1.0 + 0.0im,)
end

@testset "_resolve_bim_symmetry - rotation sectors (Star, D3 and D4)" begin
    star = StarBilliard(1.0, 0.3, 3)
    for s in 0:2
        sec = symmetry_sector(star, NFoldRotation => s)
        gen, chars = _resolve_bim_symmetry(star, sec)
        @test gen isa NFoldRotation
        @test gen.order == 3
        @test chars == (s,)                                                    # the sector index is recovered exactly
    end

    @testset "resolved generator is a registered element" begin
        gen, _ = _resolve_bim_symmetry(star, symmetry_sector(star, NFoldRotation => 1))
        @test gen.sym_id in (2, 4)
        @test BilliardGeometry.symmetry_of(star.symmetries, gen.sym_id) === gen
    end

    @testset "C4: sector is recovered from characters of all registered powers" begin
        star4 = StarBilliard(1.0, 0.3, 4)
        for s in 0:1
            gen, chars = _resolve_bim_symmetry(star4, symmetry_sector(star4, NFoldRotation => s))
            @test gen.order == 4
            @test chars == (s,)
        end
    end

    # For an even order N the element g^(N/2) has the same character for sectors s and s + N/2, so the sector is
    # recovered jointly from all registered elements.
    @testset "C4: sectors s = N/2..N-1 resolve" begin
        star4 = StarBilliard(1.0, 0.3, 4)
        for s in 2:3
            @test ss_resolves_to_sector(star4, s)
        end
    end

    # The element g^(N/2) of C4 alone only fixes the character on the rotation subgroup it generates ({e, g^2}, order 2), so
    # the sector range is 0:d-1 with d = 2 and the character is resolved uniquely.
    @testset "single element g^(N/2): resolved by the subgroup order" begin
        for (char, sector) in ((1, 0), (-1, 1))
            gens = ss_gens(NFoldRotation(4, 2, 1) => char)
            gen, chars = _resolve_bim_symmetry(gens)
            @test gen === gens[1].first
            @test chars == (sector,)
            # the sector reproduces the requested character on the element: chi_s(g) = exp(2 pi i s/d), d = 2
            @test cis(2pi * sector * 1 / 2) ≈ char atol = 1e-13
        end
        # a character that is not a value of the subgroup of order 2 is still rejected
        @test_throws ArgumentError _resolve_bim_symmetry(ss_gens(NFoldRotation(4, 2, 1) => im))
        @test_throws ArgumentError _resolve_bim_symmetry(ss_gens(NFoldRotation(4, 2, 1) => ComplexF64(0.5)))
    end

    @testset "elements of a rotation subgroup with inconsistent characters are rejected" begin
        # C3 inside C6: chi_s(g^2) = exp(2 pi i s/3) and chi_s(g^4) = exp(4 pi i s/3). The pair (exp(2 pi i/3), exp(2 pi i/3)) would
        # need s = 1 for g^2 and s = 2 for g^4.
        gens = ss_gens(NFoldRotation(6, 2, 1) => cis(2pi / 3), NFoldRotation(6, 4, 2) => cis(2pi / 3))
        @test_throws ArgumentError _resolve_bim_symmetry(gens)
        err = try _resolve_bim_symmetry(gens) catch e; e end
        @test err isa ArgumentError && occursin("Inconsistent rotation sector", err.msg)
    end
end

@testset "_resolve_bim_symmetry - generator-pair vector method" begin
    @testset "reflections go to CompositeReflection" begin
        gen, chars = _resolve_bim_symmetry(ss_gens(XAxisReflection(3) => -1, YAxisReflection(1) => 1))
        @test gen isa CompositeReflection
        @test length(gen.reflections) == 2
        @test chars == (-1.0 + 0.0im, 1.0 + 0.0im)                             # same order as the input pairs
    end

    @testset "rotations go to the rotation-sector path" begin
        gens = ss_gens(NFoldRotation(3, 1, 1) => cis(2pi / 3), NFoldRotation(3, 2, 2) => cis(4pi / 3))
        gen, chars = _resolve_bim_symmetry(gens)
        @test gen === gens[1].first
        @test chars == (1,)
    end

    @testset "rotation generator is the m = 1 element whatever the order of the pairs" begin
        gens = ss_gens(NFoldRotation(3, 2, 2) => cis(4pi / 3), NFoldRotation(3, 1, 1) => cis(2pi / 3))
        gen, chars = _resolve_bim_symmetry(gens)
        @test gen === gens[2].first
        @test chars == (1,)
    end

    @testset "sector is converted to the convention of a generator with m > 1" begin
        # sector 1 means chi(g^2) = exp(4 pi i / 3); relative to g^2 itself that is the sector 2 (chi = exp(2 pi i 2 / 3))
        gens = ss_gens(NFoldRotation(3, 2, 2) => cis(4pi / 3))
        gen, chars = _resolve_bim_symmetry(gens)
        @test gen === gens[1].first
        @test chars == (2,)
    end

    @testset "mixed reflection and rotation is rejected" begin
        gens = ss_gens(XAxisReflection(1) => 1, NFoldRotation(3, 1, 2) => 1)
        @test_throws ArgumentError _resolve_bim_symmetry(gens)
        err = try _resolve_bim_symmetry(gens) catch e; e end
        @test occursin("mixing reflection and rotation", err.msg)
    end

    @testset "rotation orders must agree" begin
        gens = ss_gens(NFoldRotation(3, 1, 1) => 1, NFoldRotation(4, 1, 2) => 1)
        @test_throws ArgumentError _resolve_bim_symmetry(gens)
    end

    @testset "rotation characters must define one sector" begin
        # chi_1(g) = exp(2 pi i / 3) but chi_1(g^2) would be exp(4 pi i / 3); pairing with chi_2(g^2) is inconsistent.
        gens = ss_gens(NFoldRotation(3, 1, 1) => cis(2pi / 3), NFoldRotation(3, 2, 2) => cis(2pi * 2 * 2 / 3))
        @test_throws ArgumentError _resolve_bim_symmetry(gens)
        err = try _resolve_bim_symmetry(gens) catch e; e end
        @test occursin("Inconsistent rotation sector", err.msg)
    end

    @testset "rotation character that is not an irrep value" begin
        gens = ss_gens(NFoldRotation(3, 1, 1) => ComplexF64(0.5))
        @test_throws ArgumentError _resolve_bim_symmetry(gens)
    end
end

@testset "_resolve_bim_symmetry - sector from a different billiard instance" begin
    rect = RectangleBilliard(1.0, 0.6)
    other = RectangleBilliard(1.0, 0.6)
    sec = symmetry_sector(rect, XAxisReflection => -1, YAxisReflection => 1)
    @test_throws ArgumentError _resolve_bim_symmetry(other, sec)
    @test_throws ArgumentError _resolve_bim_symmetry(RectangleBilliard(2.0, 1.0), sec)
    @test_throws ArgumentError _resolve_bim_symmetry(StarBilliard(1.0, 0.3, 3), sec)
    # An empty sector is checked before it is short-circuited.
    @test_throws ArgumentError _resolve_bim_symmetry(other, symmetry_sector(rect))
    # The original instance still resolves.
    @test first(_resolve_bim_symmetry(rect, sec)) isa CompositeReflection
end

# Cross-package check: a `SymmetrySector` of a D2 billiard, resolved through `_resolve_bim_symmetry` and fed to
# `BilliardGeometry.symmetry_index_orbits`, folds the boundary into orbits whose size equals the number of
# plane-wave parity patterns that the `RealPlaneWaves` basis of the same sector does NOT collapse:
# orbit_size * (#patterns) == |G|, where |G| = 4 for the full D2 group. Reference values are group orders, not pinned numbers.
@testset "SymmetrySector - orbit folding agrees with RealPlaneWaves (Rectangle, D2)" begin
    billiard = RectangleBilliard(1.0, 0.6)
    N = 40

    @testset "full D2 sector (XAxis, YAxis)" begin
        sector = symmetry_sector(billiard, XAxisReflection => -1, YAxisReflection => 1)
        generator, characters = _resolve_bim_symmetry(billiard, sector)
        @test generator isa CompositeReflection

        orbits = symmetry_index_orbits(Float64, billiard, N, generator, ComplexF64.(collect(characters)))
        # Full D2 reduction: the boundary folds into 4-element orbits, i.e. a quarter of the full boundary.
        @test orbit_size(orbits) == 4
        @test fundamental_size(orbits) == N ÷ 4

        basis = RealPlaneWaves(8, billiard, sector)
        patterns = unique(collect(zip(basis.parity_x, basis.parity_y)))
        # RealPlaneWaves collapses the same full D2 sector to a single quadrant (one cos/sin pattern).
        @test length(patterns) == 1
        @test orbit_size(orbits) == 4 ÷ length(patterns)
        @test basis.dim == 8 * length(patterns)
    end

    @testset "orbit size does not depend on the sign choice" begin
        for (cx, cy) in ((1, 1), (-1, 1), (1, -1), (-1, -1))
            sector = symmetry_sector(billiard, XAxisReflection => cx, YAxisReflection => cy)
            generator, characters = _resolve_bim_symmetry(billiard, sector)
            orbits = symmetry_index_orbits(Float64, billiard, N, generator, ComplexF64.(collect(characters)))
            @test orbit_size(orbits) == 4
            @test fundamental_size(orbits) == N ÷ 4
            basis = RealPlaneWaves(8, billiard, sector)
            @test length(unique(collect(zip(basis.parity_x, basis.parity_y)))) == 1
        end
    end

    @testset "single YAxisReflection sector: half group, two plane-wave patterns" begin
        sector = symmetry_sector(billiard, YAxisReflection => -1)
        generator, characters = _resolve_bim_symmetry(billiard, sector)
        orbits = symmetry_index_orbits(Float64, billiard, N, generator, ComplexF64.(collect(characters)))
        @test orbit_size(orbits) == 2
        @test fundamental_size(orbits) == N ÷ 2

        basis = RealPlaneWaves(8, billiard, sector)
        patterns = unique(collect(zip(basis.parity_x, basis.parity_y)))
        @test length(patterns) == 2
        @test orbit_size(orbits) == 4 ÷ length(patterns)
        @test basis.dim == 8 * length(patterns)
    end

    @testset "no symmetry: trivial resolution" begin
        sector = symmetry_sector(billiard)
        @test _resolve_bim_symmetry(billiard, sector) == (nothing, ())
    end

    @testset "node count must be compatible with the group order" begin
        sector = symmetry_sector(billiard, XAxisReflection => 1, YAxisReflection => 1)
        generator, characters = _resolve_bim_symmetry(billiard, sector)
        @test_throws ArgumentError symmetry_index_orbits(Float64, billiard, 42, generator, ComplexF64.(collect(characters)))
    end

    @testset "sector from another instance is rejected" begin
        sector = symmetry_sector(billiard, XAxisReflection => -1, YAxisReflection => 1)
        @test_throws ArgumentError _resolve_bim_symmetry(RectangleBilliard(1.0, 0.6), sector)
    end
end

@testset "SymmetrySector - orbit folding for a rotation sector (Star, D3)" begin
    star = StarBilliard(1.0, 0.3, 3)
    N = 30
    for s in 0:2
        sector = symmetry_sector(star, NFoldRotation => s)
        generator, characters = _resolve_bim_symmetry(star, sector)
        orbits = symmetry_index_orbits(Float64, star, N, generator, characters[1])
        @test orbit_size(orbits) == 3
        @test fundamental_size(orbits) == N ÷ 3
    end
end

################################################################################
######################## symmetry group G' of billiards with holes #############
################################################################################

# Minimal billiard holding a MultiplyConnectedDomain and a registry (the production hole billiards have an empty registry).
struct ss_HoleBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_domain::MultiplyConnectedDomain{T}
    symmetries::SymmetryRegistry
end

ss_disc_domain(R, x0, y0, id) = SimpleDomain{Float64}(AbsCurve[CircleSegment(R, 2π, 0.0, [x0, y0]; domain_id = id, segment_id = 1)],
                                                      SVector{2,Float64}[SVector(x0 + R, y0)], id)

# Quarter square [0,1]^2 of RectangleBilliard(1.0, 1.0) (registry Y, XY, X = sym_ids 1, 2, 3) with one hole object.
ss_hole_billiard(hole, reg = D2_symmetry()) =
    ss_HoleBilliard{Float64}(MultiplyConnectedDomain(RectangleBilliard(1.0, 1.0).fundamental_domain, [hole], 1), reg)

# Disc of radius 0.1 about (0.5, 0) with its own D2 group: only X keeps it in place, G' = {e, XAxisReflection(3)}.
function ss_x_disc_billiard()
    piece = CircleWedge(0.1, pi / 2, 0.0, 0.5, 0.0, 2; bcs = AbsBoundaryCondition[SpecularReflection(), SymmetryWall(1, 2), SymmetryWall(3, 4)])
    return ss_hole_billiard(SymmetricHole(piece, D2_symmetry(), SVector(0.5, 0.0)))
end
# Four discs placed by D2: G' = D2.
ss_four_discs_billiard() = ss_hole_billiard(SymmetrizedHoles(DomainHole(ss_disc_domain(0.1, 0.4, 0.2, 5)), D2_symmetry()))
# Asymmetric disc: G' = {e}.
ss_plain_disc_billiard() = ss_hole_billiard(DomainHole(ss_disc_domain(0.2, 0.3, 0.4, 5)))

# Unit circle whose registry holds the given symmetries (only the registry is inspected by the sector resolution).
function ss_registry_circle(syms::AbsSymmetry...)
    seg = FourierCoeffPolarSegment(Float64[]; shift_angle = pi / 4)
    dom = BilliardGeometry.PolarDomain{Float64}([seg], [curve(seg, 0.0)], 1)
    return PolarBilliard{Float64}(dom, register_symmetries(syms...))
end

@testset "_rotation_subgroup_order" begin
    # the identity and every orientation-preserving element
    @test _rotation_subgroup_order(register_symmetries()) == 1
    @test _rotation_subgroup_order(register_symmetries(XAxisReflection())) == 1
    @test _rotation_subgroup_order(D2_symmetry()) == 2                  # {e, XY}
    @test _rotation_subgroup_order(D4_symmetry()) == 4                  # {e, R1, XY, R3}
    @test _rotation_subgroup_order(Cn_symmetry(6)) == 6
    @test _rotation_subgroup_order(Cn_symmetry(1)) == 1
    @test _rotation_subgroup_order(register_symmetries(NFoldRotation(6, 2), NFoldRotation(6, 4))) == 3
    # a billiard registry and a G' registry alike
    @test _rotation_subgroup_order(RectangleBilliard(1.0, 0.6).symmetries) == 2
    @test _rotation_subgroup_order(get_symmetry_group(ss_x_disc_billiard())) == 1      # G' = {e, X}
    @test _rotation_subgroup_order(get_symmetry_group(ss_four_discs_billiard())) == 2  # G' = D2
end

@testset "symmetry_sector reads G' (billiards with holes)" begin
    b2 = ss_x_disc_billiard()
    @test length(b2.symmetries) == 3
    @test length(get_symmetry_group(b2)) == 1
    # X is in G'
    sec = symmetry_sector(b2, XAxisReflection => -1)
    @test sec isa SymmetrySector
    @test sec.billiard === b2
    @test sec.characters == Dict{Int,ComplexF64}(3 => -1)
    # Y and XY are registered by the billiard but broken by the hole: the error names G'
    err = try symmetry_sector(b2, YAxisReflection => 1) catch e; e end
    @test err isa ArgumentError
    @test err.msg == "YAxisReflection is registered by the billiard but not in its symmetry group G' = {e, XAxisReflection(3)}; a hole breaks it"
    @test_throws "G'" symmetry_sector(b2, YAxisReflection => 1)
    @test_throws "G'" symmetry_sector(b2, XYAxisReflection => 1)
    @test_throws "G'" symmetry_sector(b2, XAxisReflection => 1, YAxisReflection => 1)
    # a type not registered at all keeps the old message (no G' mention)
    err = try symmetry_sector(b2, DiagonalReflection => 1) catch e; e end
    @test err isa ArgumentError && occursin("no registered DiagonalReflection", err.msg) && !occursin("G'", err.msg)
    # the abstract type selects only the elements of G'
    @test symmetry_sector(b2, BilliardGeometry.AbsReflection => 1).characters == Dict{Int,ComplexF64}(3 => 1)

    # a DomainHole breaks all of G: G' = {e}
    plain = ss_plain_disc_billiard()
    @test_throws "G' = {e}" symmetry_sector(plain, XAxisReflection => 1)
    @test_throws ArgumentError symmetry_sector(plain, YAxisReflection => 1)
    @test isempty(symmetry_sector(plain).characters)
    @test_throws ArgumentError symmetry_sector(plain, 1)                # no generators in G'

    # all of D2 survives four discs placed by D2
    four = ss_four_discs_billiard()
    @test symmetry_sector(four, XAxisReflection => -1, YAxisReflection => 1).characters == Dict{Int,ComplexF64}(3 => -1, 1 => 1)

    # positional form: one character per generator of G'
    @test symmetry_sector(b2, -1).characters == Dict{Int,ComplexF64}(3 => -1)
    @test_throws ArgumentError symmetry_sector(b2, 1, 1)                # G' has a single generator
    @test symmetry_sector(four, -1, 1).characters == symmetry_sector(four, YAxisReflection => -1, XAxisReflection => 1).characters

    # billiards without holes are unchanged
    rect = RectangleBilliard(1.0, 0.6)
    @test symmetry_sector(rect, YAxisReflection => 1).characters == Dict{Int,ComplexF64}(1 => 1)
end

@testset "_resolve_bim_symmetry reads G' (billiards with holes)" begin
    b2 = ss_x_disc_billiard()
    gen, chars = _resolve_bim_symmetry(b2, symmetry_sector(b2, XAxisReflection => -1))
    @test gen isa CompositeReflection
    @test only(gen.reflections) isa XAxisReflection
    @test only(gen.reflections).sym_id == 3                              # the element of G', same sym_id as in the registry
    @test chars == (-1.0 + 0.0im,)

    four = ss_four_discs_billiard()
    gen4, chars4 = _resolve_bim_symmetry(four, symmetry_sector(four, XAxisReflection => -1, YAxisReflection => 1))
    @test gen4 isa CompositeReflection
    @test Set(typeof.(gen4.reflections)) == Set([XAxisReflection, YAxisReflection])
    @test ss_char_by_type(gen4, chars4) == Dict(XAxisReflection => -1, YAxisReflection => 1)

    # empty sector
    @test _resolve_bim_symmetry(b2, symmetry_sector(b2)) == (nothing, ())
end

@testset "symmetry_sector rotation subgroup (C3 inside C6)" begin
    b = ss_registry_circle(NFoldRotation(6, 2), NFoldRotation(6, 4))
    reg = b.symmetries
    @test _rotation_subgroup_order(reg) == 3
    # sectors range over 0:d-1 with d = 3 the order of the rotation subgroup, not 0:5
    for s in 0:2
        sec = symmetry_sector(b, NFoldRotation => s)
        @test sec.characters[reg[1].sym_id] ≈ cis(2pi * s * 2 / 6) atol = 1e-13
        @test sec.characters[reg[2].sym_id] ≈ cis(2pi * s * 4 / 6) atol = 1e-13
    end
    @test_throws ArgumentError symmetry_sector(b, NFoldRotation => 3)
    @test_throws ArgumentError symmetry_sector(b, NFoldRotation => -1)
    @test_throws "0:2" symmetry_sector(b, NFoldRotation => 3)

    # every sector resolves to the generator of the subgroup (m = 2) and the sector itself
    for s in 0:2
        gen, chars = _resolve_bim_symmetry(b, symmetry_sector(b, NFoldRotation => s))
        @test gen isa NFoldRotation && gen.order == 6 && gen.m == 2
        @test chars == (s,)
        # fold: the image of a node under g^l carries chi_s(g^(2l)) = exp(2 pi i s (2l) / 6), and g^2 (l = 2) is the element (6, 4)
        orbits = symmetry_index_orbits(Float64, b, 12, gen, chars[1])
        @test orbit_size(orbits) == 3
        @test fundamental_size(orbits) == 4
        @test sort(vec(orbits.fund_to_full)) == 1:12
        @test orbits.fund_to_scale[:, 1] ≈ [cis(2pi * s * 2l / 6) for l in 0:2] atol = 1e-13
        sec = symmetry_sector(b, NFoldRotation => s)
        @test orbits.fund_to_scale[3, 1] ≈ sec.characters[reg[2].sym_id] atol = 1e-13
        @test orbits.fund_to_scale[2, 1] ≈ sec.characters[reg[1].sym_id] atol = 1e-13
    end

    # the positional form takes the single generator of the subgroup
    @test symmetry_sector(b, 2).characters == symmetry_sector(b, NFoldRotation => 2).characters
    @test_throws ArgumentError symmetry_sector(b, 3)
end

# Billiards (constructed outside @test_logs: some constructors log an @info of their own) with the expected
# (group label, order, characters, n_2d) of G'.
ss_irrep_cases() = [
    ("Rectangle", RectangleBilliard(1.0, 0.6), "D_2", 4, Set([(1, 1), (1, -1), (-1, 1), (-1, -1)]), 0),
    ("Circle", CircleBilliard(1.0), "D_2", 4, Set([(1, 1), (1, -1), (-1, 1), (-1, -1)]), 0),
    ("Stadium", StadiumBilliard(0.5), "D_2", 4, Set([(1, 1), (1, -1), (-1, 1), (-1, -1)]), 0),
    ("Prosen", ProsenBilliard(0.3), "D_2", 4, Set([(1, 1), (1, -1), (-1, 1), (-1, -1)]), 0),
    ("Limacon", LimaconBilliard(0.3), "D_1", 2, Set([(1,), (-1,)]), 0),
    ("Mushroom", MushroomBilliard(0.5, 1.0), "D_1", 2, Set([(1,), (-1,)]), 0),
    ("Annular", AnnularBilliard(0.3, 0.0), "D_1", 2, Set([(1,), (-1,)]), 0),
    ("C3", C3Billiard(0.1), "C_3", 3, Set([(0,), (1,), (2,)]), 0),
    ("Julia", JuliaBilliard(0.3, 0.6), "D_3", 6, Set([(1, 1), (-1, -1)]), 1),
    ("Star5", StarBilliard(1.0, 0.2, 5), "D_5", 10, Set([(1, 1), (-1, -1)]), 2),
    ("Sinai", SinaiBilliard(1.0, 0.3), "D_4", 8, Set([(1, 1), (1, -1), (-1, 1), (-1, -1)]), 1),
    ("Triangle", TriangleBilliard(1.0, 1.5), "trivial", 1, Set([()]), 0),
]

@testset "symmetry_irreps" begin
    for (name, b, group, order, chars, n2d) in ss_irrep_cases()
        @testset "$name" begin
            info = symmetry_irreps(b)
            @test info.group == group
            @test info.order == order
            @test Set(info.characters) == chars
            @test length(info.characters) == length(chars)        # no duplicates
            @test info.n_2d == n2d
            @test length(info.generators) == (group == "trivial" ? 0 : length(first(info.characters)))
            # every returned character is accepted by symmetry_sector
            for c in info.characters
                @test symmetry_sector(b, c...) isa SymmetrySector
            end
        end
    end

    @testset "|G'| = sum of squared irrep dimensions" begin
        for (_, b, _, _, _, _) in ss_irrep_cases()
            info = symmetry_irreps(b)
            @test info.order == length(info.characters) + 4 * info.n_2d
        end
    end
end

@testset "log_symmetries" begin
    rect = RectangleBilliard(1.0, 0.6)
    @test_logs (:info, r"D_2 of order 4") @test log_symmetries(rect) === nothing

    star = StarBilliard(1.0, 0.2, 5)
    @test_logs (:info, r"D_5 of order 10") @test log_symmetries(star) === nothing

    sinai = SinaiBilliard(1.0, 0.3)
    @test_logs (:info, r"D_4 of order 8") @test log_symmetries(sinai) === nothing

    tri = TriangleBilliard(1.0, 1.5)
    @test_logs (:info, r"no symmetry") @test log_symmetries(tri) === nothing
end

@testset "symmetry_sector irrep2d" begin
    star = StarBilliard(1.0, 0.2, 5)
    sinai = SinaiBilliard(1.0, 0.3)

    @testset "degeneracy and rotation characters cis(2 pi k m / N)" begin
        for (b, N, ks) in ((star, 5, 1:2), (sinai, 4, 1:1), (JuliaBilliard(0.3, 0.6), 3, 1:1))
            rots = filter(g -> g isa NFoldRotation, BilliardGeometry.get_symmetry_group(b))
            for k in ks
                sec = symmetry_sector(b; irrep2d = k)
                @test sec isa SymmetrySector
                @test sec.billiard === b
                @test sec.degeneracy == 2
                @test symmetry_degeneracy(sec) == 2
                @test Set(keys(sec.characters)) == Set(g.sym_id for g in rots)
                for g in rots
                    @test g.order == N
                    @test sec.characters[g.sym_id] ≈ cis(2pi * k * g.m / N) atol = 1e-13
                end
            end
        end
    end

    @testset "symmetry_degeneracy of 1D sectors" begin
        rect = RectangleBilliard(1.0, 0.6)
        @test symmetry_degeneracy(symmetry_sector(rect, XAxisReflection => 1)) == 1
        @test symmetry_degeneracy(symmetry_sector(rect)) == 1
        @test symmetry_degeneracy(symmetry_sector(star, NFoldRotation => 1)) == 1
    end

    @testset "invalid k" begin
        @test_throws ArgumentError symmetry_sector(star; irrep2d = 0)
        @test_throws ArgumentError symmetry_sector(star; irrep2d = 3)          # D_5 has E_1, E_2 only
        @test_throws ArgumentError symmetry_sector(sinai; irrep2d = 2)         # D_4 has E_1 only
        @test_throws ArgumentError symmetry_sector(sinai; irrep2d = -1)
        err = try symmetry_sector(star; irrep2d = 3) catch e; e end
        @test occursin("1:2", err.msg)
    end

    @testset "G' is not D_d with d >= 3" begin
        @test_throws ArgumentError symmetry_sector(RectangleBilliard(1.0, 0.6); irrep2d = 1)   # D_2
        @test_throws ArgumentError symmetry_sector(LimaconBilliard(0.3); irrep2d = 1)          # D_1
        @test_throws ArgumentError symmetry_sector(C3Billiard(0.1); irrep2d = 1)               # C_3, no mirrors
        @test_throws ArgumentError symmetry_sector(TriangleBilliard(1.0, 1.5); irrep2d = 1)    # trivial
        err = try symmetry_sector(C3Billiard(0.1); irrep2d = 1) catch e; e end
        @test occursin("dihedral", err.msg)
    end

    @testset "cannot be combined with Pair choices" begin
        @test_throws ArgumentError symmetry_sector(star, NFoldRotation => 1; irrep2d = 1)
        @test_throws ArgumentError symmetry_sector(sinai, XAxisReflection => 1; irrep2d = 1)
        err = try symmetry_sector(star, NFoldRotation => 1; irrep2d = 1) catch e; e end
        @test occursin("irrep2d", err.msg)
    end
end

@testset "symmetry_irreps - 2D irrep fields" begin
    @test symmetry_irreps(StarBilliard(1.0, 0.2, 5)).irreps_2d == [1, 2]
    @test symmetry_irreps(StarBilliard(1.0, 0.2, 5)).n_2d == 2
    @test symmetry_irreps(SinaiBilliard(1.0, 0.3)).irreps_2d == [1]
    @test symmetry_irreps(JuliaBilliard(0.3, 0.6)).irreps_2d == [1]
    for b in (RectangleBilliard(1.0, 0.6), C3Billiard(0.1), TriangleBilliard(1.0, 1.5), LimaconBilliard(0.3))
        info = symmetry_irreps(b)
        @test info.irreps_2d == Int[]
        @test info.n_2d == 0
    end
    # every advertised k is accepted by symmetry_sector
    for b in (StarBilliard(1.0, 0.2, 5), SinaiBilliard(1.0, 0.3))
        @test all(k -> symmetry_degeneracy(symmetry_sector(b; irrep2d = k)) == 2, symmetry_irreps(b).irreps_2d)
    end
end

@testset "symmetry_partner" begin
    sinai = SinaiBilliard(1.0, 0.3)
    star = StarBilliard(1.0, 0.2, 5)
    f(pt) = pt[1] + 2 * pt[2] + pt[1] * pt[2]
    pts = [SVector(0.3, 0.1), SVector(-0.2, 0.7), SVector(0.5, -0.4)]
    for b in (sinai, star)
        group = BilliardGeometry.get_symmetry_group(b)
        sigma = group[findfirst(g -> g isa BilliardGeometry.AbsReflection, group)]
        g = symmetry_partner(f, b)
        @test all(p -> isapprox(g(p), f(BilliardGeometry.apply_symmetry(sigma, p)); atol = 1e-13), pts)
        # the mirror is an involution, so partner of the partner is f
        @test all(p -> isapprox(symmetry_partner(g, b)(p), f(p); atol = 1e-13), pts)
        @test any(p -> !isapprox(g(p), f(p); atol = 1e-6), pts)               # partner differs from f
    end
    # constant function is invariant
    @test symmetry_partner(pt -> 3.0, sinai)(pts[1]) == 3.0
    @test_throws ArgumentError symmetry_partner(f, C3Billiard(0.1))          # no mirror
    @test_throws ArgumentError symmetry_partner(f, TriangleBilliard(1.0, 1.5))
end

@testset "_resolve_bim_symmetry - irrep2d" begin
    sinai = SinaiBilliard(1.0, 0.3)
    gen, chars = _resolve_bim_symmetry(sinai, symmetry_sector(sinai; irrep2d = 1))
    @test gen isa NFoldRotation
    @test gen.order == 4
    @test gen.m == 1
    @test chars == (1,)

    star = StarBilliard(1.0, 0.2, 5)
    for k in 1:2
        gen, chars = _resolve_bim_symmetry(star, symmetry_sector(star; irrep2d = k))
        @test gen isa NFoldRotation && gen.order == 5 && gen.m == 1
        @test chars == (k,)
    end
end

@testset "irrep2d sectors are rejected by basis solvers" begin
    sinai = SinaiBilliard(1.0, 0.3)
    sec = symmetry_sector(sinai; irrep2d = 1)
    @test_throws ArgumentError FourierBessel(10, sinai, sec)
    @test_throws ArgumentError RealPlaneWaves(10, sinai, sec)
end
