# Unit tests for
#   src/states/symmetry/symmetrysector.jl   (SymmetrySector, symmetry_sector (pair form and positional form),
#                                            _sector_character, _check_sector_billiard, _resolve_bim_symmetry,
#                                            _match_rotation_sector, and the reflection / rotation resolution helpers)
# The symmetry generators, the registry and the orbit maps belong to BilliardGeometry; here they are only used as
# inputs (RectangleBilliard: D2 reflections, StarBilliard: cyclic rotations, a circular PolarBilliard: a single
# diagonal reflection, TriangleBilliard: no symmetry) and the characters recovered from them are compared with the
# closed forms chi(g) = +-1 and chi_s(g^m) = exp(2 pi i s m / N).
# No eigen solves, no matrix assembly. The only cross-package check is the final testset, which compares the orbit
# folding of BilliardGeometry with the plane-wave pattern count of RealPlaneWaves for the same sector.
# Helper functions use the prefix `ss_`.

using QuantumBilliards: _sector_character, _match_rotation_sector, _resolve_bim_symmetry, _check_sector_billiard

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
        _, chars = _resolve_bim_symmetry(billiard, symmetry_sector(billiard, s))
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
    @test fieldnames(SymmetrySector) == (:billiard, :characters)
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

@testset "symmetry_sector pairs - rotation billiard (Star, C3)" begin
    star = StarBilliard(1.0, 0.3, 3)
    # StarBilliard registers NFoldRotation(3, m) with sym_id = m = 1, 2.
    @testset "sector s gives chi_s(g^m) = exp(2 pi i s m / 3)" begin
        for s in 0:2
            sec = symmetry_sector(star, NFoldRotation => s)
            @test sec.billiard === star
            @test Set(keys(sec.characters)) == Set([1, 2])
            @test sec.characters[1] ≈ cis(2pi * s * 1 / 3) atol = 1e-13
            @test sec.characters[2] ≈ cis(2pi * s * 2 / 3) atol = 1e-13
            @test sec.characters[2] ≈ sec.characters[1]^2 atol = 1e-13
        end
        @test symmetry_sector(star, NFoldRotation => 0).characters == Dict{Int,ComplexF64}(1 => 1, 2 => 1)
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
        @test_throws ArgumentError symmetry_sector(star, XAxisReflection => 1)
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

    @testset "cyclic: one integer sector" begin
        for s in 0:2
            @test symmetry_sector(star, s).characters == symmetry_sector(star, NFoldRotation => s).characters
        end
    end

    @testset "wrong number of characters" begin
        @test_throws ArgumentError symmetry_sector(rect, 1)                    # D2 needs two
        @test_throws ArgumentError symmetry_sector(rect, 1, 1, 1)
        @test_throws ArgumentError symmetry_sector(star, 0, 0)                 # C3 needs one
        @test_throws ArgumentError symmetry_sector(tri, 1)                     # no generators at all
        err = try symmetry_sector(rect, 1) catch e; e end
        @test occursin("2 expected", err.msg) && occursin("received 1", err.msg)
    end

    @testset "invalid values are forwarded to the pair form" begin
        @test_throws ArgumentError symmetry_sector(rect, 0, 1)
        @test_throws ArgumentError symmetry_sector(rect, 1, 2)
        @test_throws ArgumentError symmetry_sector(star, 3)
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

@testset "_resolve_bim_symmetry - rotation sectors (Star, C3 and C4)" begin
    star = StarBilliard(1.0, 0.3, 3)
    for s in 0:2
        sec = symmetry_sector(star, NFoldRotation => s)
        gen, chars = _resolve_bim_symmetry(star, sec)
        @test gen isa NFoldRotation
        @test gen.order == 3
        @test chars == (s,)                                                    # the sector index is recovered exactly
    end

    @testset "resolved generator is a registered element" begin
        gen, _ = _resolve_bim_symmetry(star, symmetry_sector(star, 1))
        @test gen.sym_id in (1, 2)
        @test BilliardGeometry.symmetry_of(star.symmetries, gen.sym_id) === gen
    end

    @testset "C4: sector is recovered from characters of all registered powers" begin
        star4 = StarBilliard(1.0, 0.3, 4)
        for s in 0:1
            gen, chars = _resolve_bim_symmetry(star4, symmetry_sector(star4, s))
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

    @testset "ambiguous sector from a single element g^(N/2)" begin
        gens = ss_gens(NFoldRotation(4, 2, 1) => -1)
        err = try _resolve_bim_symmetry(gens) catch e; e end
        @test err isa ArgumentError && occursin("Ambiguous rotation sector", err.msg)
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

@testset "SymmetrySector - orbit folding for a cyclic sector (Star, C3)" begin
    star = StarBilliard(1.0, 0.3, 3)
    N = 30
    for s in 0:2
        sector = symmetry_sector(star, s)
        generator, characters = _resolve_bim_symmetry(star, sector)
        orbits = symmetry_index_orbits(Float64, star, N, generator, characters[1])
        @test orbit_size(orbits) == 3
        @test fundamental_size(orbits) == N ÷ 3
    end
end
