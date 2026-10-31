# Unit tests for src/spectra/unfolding.jl
#   corner_correction, weyl_law, k_at_state, state_at_k, spectral_density, k_range_for_states
# Reference billiard: RectangleBilliard(a, b) is the FULL 2a x 2b rectangle (D2 symmetric),
#   full:        A = 4ab, L = 4(a+b), four right-angle corners  ->  N(k) = (A k^2 - L k)/(4pi) + 1/4
#   fundamental: area / 4, boundary curves reported by `get_boundary_curves`
# No eigen-solves: every expected value is analytic or a consistency identity.
#
# Billiards with holes and a nonempty registry (B4): _has_reduced_holes, _weyl_area_length, _weyl_angles, _weyl_inputs and the four accessors
#   k_at_state, state_at_k, spectral_density, k_range_for_states built on them. Reference billiards are local minimal types: the quarter square
#   [0,1]^2 of RectangleBilliard(1.0, 1.0) (D2 registry, full region [-1,1]^2, area 4, perimeter 8, four right-angle corners) with circular
#   holes of radius r given in the full frame. With G' the symmetry group of the holes (|G'| = length(get_symmetry_group) + 1) the
#   fundamental = true inputs are A = A_full/|G'|, L = L_full/|G'| and every |G'|-th corner angle of the sorted full-boundary angles;
#   fundamental = false and billiards without holes or with an empty registry follow the old code path. Helpers use the prefix `uf_`.

using StaticArrays: SVector
using QuantumBilliards: k_at_state, corner_correction, _has_reduced_holes, _weyl_area_length, _weyl_angles, _weyl_inputs

@testset "corner_correction" begin
    # per-corner term (pi^2 - c^2)/(24 pi c)
    @test corner_correction([pi / 2, pi / 2, pi / 2, pi / 2]) ≈ 1 / 4   # rectangle: 4 * 1/16
    @test corner_correction([pi / 3, pi / 3, pi / 3]) ≈ 1 / 3           # equilateral triangle: 3 * 1/9
    @test corner_correction([pi]) ≈ 0 atol = 1e-15                      # a flat angle contributes nothing
    @test corner_correction([pi / 2]) ≈ corner_correction([pi / 2, pi / 2]) / 2
    @test corner_correction([pi / 4]) > corner_correction([pi / 2])     # sharper corners contribute more
    @test corner_correction(Float64[]) == 0
end

@testset "weyl_law" begin
    A, L = 3.0, 7.0

    @testset "closed form without corners" begin
        for k in (0.5, 4.0, 20.0)
            @test weyl_law(k, A, L) ≈ (A * k^2 - L * k) / (4pi)
        end
        @test weyl_law(0.0, A, L) == 0.0
        @test weyl_law(L / A, A, L) ≈ 0 atol = 1e-14                    # root of the leading terms
        @test @inferred(weyl_law(4.0, A, L)) isa Float64
    end

    @testset "corner correction is an additive constant" begin
        angles = [pi / 2, pi / 2, pi / 2, pi / 2]
        @test weyl_law(10.0, A, L, angles) ≈ weyl_law(10.0, A, L) + 1 / 4
        @test weyl_law(10.0, A, L, angles) - weyl_law(5.0, A, L, angles) ≈ weyl_law(10.0, A, L) - weyl_law(5.0, A, L)
        @test @inferred(weyl_law(10.0, A, L, angles)) isa Float64
    end

    @testset "vectorised over k" begin
        ks = [2.0, 5.0, 11.0]
        @test weyl_law(ks, A, L) ≈ [weyl_law(k, A, L) for k in ks]
        @test weyl_law(ks, A, L, [pi / 3, pi / 3, pi / 3]) ≈ weyl_law(ks, A, L) .+ 1 / 3
    end

    @testset "monotone beyond the minimum" begin
        ks = range(L / (2A) + 0.1, 50.0; length=40)
        @test issorted(weyl_law(collect(ks), A, L))
    end
end

@testset "k_at_state" begin
    A, L = 3.0, 7.0

    @testset "closed form without corners" begin
        for N in (1, 10, 250)
            @test k_at_state(N, A, L) ≈ (L + sqrt(L^2 + 16pi * A * N)) / (2A)
        end
        @test k_at_state(0, A, L) ≈ L / A                               # N = 0 at the nonzero root
        @test @inferred(k_at_state(10, A, L)) isa Float64
    end

    @testset "inverts weyl_law" begin
        for N in (1.0, 10.0, 250.0, 1e4)
            @test weyl_law(k_at_state(N, A, L), A, L) ≈ N
        end
        for k in (3.0, 9.0, 40.0)
            @test k_at_state(weyl_law(k, A, L), A, L) ≈ k
        end
    end

    @testset "inverts weyl_law with corner correction" begin
        angles = [pi / 2, pi / 2, pi / 2, pi / 2]
        for N in (1.0, 10.0, 250.0)
            @test weyl_law(k_at_state(N, A, L, angles), A, L, angles) ≈ N
        end
        @test k_at_state(10, A, L, angles) < k_at_state(10, A, L)       # N = weyl(k) + c: a positive c lowers the k reaching N
        @test @inferred(k_at_state(10, A, L, angles)) isa Float64
    end

    @testset "strictly increasing in state" begin
        ks = [k_at_state(N, A, L) for N in 1:50]
        @test all(diff(ks) .> 0)
    end

    @testset "billiard method matches rectangle closed form" begin
        a, b = 1.0, 0.5
        r = RectangleBilliard(a, b)
        Afull, Lfull = 4a * b, 4 * (a + b)
        N = 40
        @test k_at_state(N, r; fundamental=false) ≈ k_at_state(N, Afull, Lfull, fill(pi / 2, 4))
        @test weyl_law(k_at_state(N, r; fundamental=false), Afull, Lfull, fill(pi / 2, 4)) ≈ N
        @test k_at_state(N, r) ≈ k_at_state(N, fundamental_area(r), sum(c.length for c in get_boundary_curves(r)), corner_angles(r))
        @test k_at_state(N, r) > k_at_state(N, r; fundamental=false)    # smaller area -> larger k for the same N
    end
end

@testset "state_at_k" begin
    A, L = 3.0, 7.0
    angles = [pi / 2, pi / 2, pi / 2, pi / 2]

    @testset "area/length methods are weyl_law" begin
        @test state_at_k(12.0, A, L) == weyl_law(12.0, A, L)
        @test state_at_k(12.0, A, L, angles) == weyl_law(12.0, A, L, angles)
        @test @inferred(state_at_k(12.0, A, L)) isa Float64
    end

    @testset "round trip with k_at_state" begin
        for N in (5.0, 80.0, 1e3)
            @test state_at_k(k_at_state(N, A, L, angles), A, L, angles) ≈ N
        end
        for k in (4.0, 15.0, 60.0)
            @test k_at_state(state_at_k(k, A, L, angles), A, L, angles) ≈ k
        end
    end

    @testset "billiard method on a rectangle" begin
        a, b = 1.0, 0.5
        r = RectangleBilliard(a, b)
        Afull, Lfull = 4a * b, 4 * (a + b)
        k = 25.0
        @test state_at_k(k, r; fundamental=false) ≈ (Afull * k^2 - Lfull * k) / (4pi) + 1 / 4
        @test state_at_k(k, r) ≈ state_at_k(k, fundamental_area(r), sum(c.length for c in get_boundary_curves(r)), corner_angles(r))
        for N in (10, 100)
            @test state_at_k(k_at_state(N, r), r) ≈ N
            @test state_at_k(k_at_state(N, r; fundamental=false), r; fundamental=false) ≈ N
        end
    end
end

@testset "spectral_density" begin
    A, L = 3.0, 7.0

    @testset "closed form" begin
        for k in (4.0, 10.0, 40.0)
            @test spectral_density(k, A, L) ≈ (A * k - L / 2) / (2pi)
        end
        @test spectral_density(L / (2A), A, L) ≈ 0 atol = 1e-14
        @test @inferred(spectral_density(10.0, A, L)) isa Float64
    end

    @testset "derivative of weyl_law" begin
        h = 1e-3    # weyl_law is quadratic in k, so the central difference is exact up to rounding
        for k in (4.0, 10.0, 40.0)
            fd = (weyl_law(k + h, A, L) - weyl_law(k - h, A, L)) / (2h)
            @test spectral_density(k, A, L) ≈ fd rtol = 1e-8
        end
    end

    @testset "derivative of weyl_law with corner correction" begin
        angles = [pi / 2, pi / 2, pi / 2, pi / 2]
        h = 1e-3
        k = 12.0
        fd = (weyl_law(k + h, A, L, angles) - weyl_law(k - h, A, L, angles)) / (2h)
        @test spectral_density(k, A, L) ≈ fd rtol = 1e-8
    end

    @testset "billiard method on a rectangle" begin
        r = RectangleBilliard(1.0, 0.5)
        h = 1e-3
        k = 30.0
        for fundamental in (true, false)
            fd = (state_at_k(k + h, r; fundamental) - state_at_k(k - h, r; fundamental)) / (2h)
            @test spectral_density(k, r; fundamental) ≈ fd rtol = 1e-8
        end
        @test spectral_density(k, r; fundamental=false) ≈ (4 * 1.0 * 0.5 * k - 4 * (1.0 + 0.5) / 2) / (2pi) rtol = 1e-12
        @test spectral_density(k, r; fundamental=false) > spectral_density(k, r)
    end

    @testset "mean density integrates to the state count" begin
        r = RectangleBilliard(1.0, 0.5)
        k1, k2 = 20.0, 30.0
        n = 2000
        ks = range(k1, k2; length=n + 1)
        integral = sum((spectral_density(ks[i], r) + spectral_density(ks[i+1], r)) / 2 * step(ks) for i in 1:n)
        @test integral ≈ state_at_k(k2, r) - state_at_k(k1, r) rtol = 1e-8
    end
end

@testset "k_range_for_states" begin
    r = RectangleBilliard(1.0, 0.5)

    @testset "endpoints are k_at_state" begin
        k1, k2 = k_range_for_states(r, 10, 100)
        @test k1 == k_at_state(10, r)
        @test k2 == k_at_state(100, r)
        @test k1 < k2
        @test state_at_k(k1, r) ≈ 10
        @test state_at_k(k2, r) ≈ 100
    end

    @testset "fundamental keyword" begin
        k1, k2 = k_range_for_states(r, 10, 100; fundamental=false)
        @test (k1, k2) == (k_at_state(10, r; fundamental=false), k_at_state(100, r; fundamental=false))
        kf1, kf2 = k_range_for_states(r, 10, 100)
        @test k1 != kf1 && k2 != kf2
    end

    @testset "degenerate and reversed ranges" begin
        k1, k2 = k_range_for_states(r, 25, 25)
        @test k1 == k2
        k1, k2 = k_range_for_states(r, 100, 10)
        @test k1 > k2
    end

    @testset "argument types" begin
        @test_throws MethodError k_range_for_states(r, 1.5, 10)
        @test @inferred(k_range_for_states(r, 10, 100)) isa Tuple{Float64,Float64}
        @test k_range_for_states(r, 10, 100) isa Tuple{Float64,Float64}
    end
end

################################################################################
############ Weyl inputs of billiards with holes and a nonempty registry ########
################################################################################

# Minimal billiard (the area routines need the element type as a parameter): MultiplyConnectedDomain plus registry.
struct uf_HoleBilliard{T} <: AbsBilliard where {T<:Real}
    fundamental_domain::MultiplyConnectedDomain{T}
    symmetries::SymmetryRegistry
end

# Full counterclockwise disc of radius R about (x0, y0) as a simple domain with the given domain id.
uf_disc_domain(R, x0, y0, id) = SimpleDomain{Float64}(AbsCurve[CircleSegment(R, 2π, 0.0, [x0, y0]; domain_id=id, segment_id=1)],
                                                      SVector{2,Float64}[SVector(x0 + R, y0)], id)

# Quarter square [0,1]^2 of RectangleBilliard(1.0, 1.0) (full frame [-1,1]^2) with one hole object, given registry (D2 by default).
uf_hole_billiard(hole, reg=D2_symmetry()) = uf_HoleBilliard{Float64}(MultiplyConnectedDomain(RectangleBilliard(1.0, 1.0).fundamental_domain, [hole], 1), reg)

# Four discs of radius 0.1 placed by D2 about (+-0.4, +-0.2): G' = D2, |G'| = 4.
uf_four_discs() = uf_hole_billiard(SymmetrizedHoles(DomainHole(uf_disc_domain(0.1, 0.4, 0.2, 5)), D2_symmetry()))
# Disc of radius 0.1 about (0.5, 0) with its own D2 group: only X keeps it in place, G' = {X}, |G'| = 2.
function uf_x_disc()
    piece = CircleWedge(0.1, pi / 2, 0.0, 0.5, 0.0, 2; bcs=AbsBoundaryCondition[SpecularReflection(), SymmetryWall(1, 2), SymmetryWall(3, 4)])
    return uf_hole_billiard(SymmetricHole(piece, D2_symmetry(), SVector(0.5, 0.0)))
end
# Asymmetric disc of radius 0.2 about (0.3, 0.4): G' = {e}, |G'| = 1.
uf_plain_disc() = uf_hole_billiard(DomainHole(uf_disc_domain(0.2, 0.3, 0.4, 5)))

# Origin-centered Sinai and annulus are symmetric since B7 (registry D4 resp. {X}); a nonzero center keeps the unreduced construction with an
# empty registry. Sinai(1, 0.3): full area 4 - pi 0.09, full perimeter 8 + 2 pi 0.3.
uf_sinai_shifted() = SinaiBilliard(1.0, 0.3; center=[0.2, 0.1])
uf_annulus_shifted() = AnnularBilliard(0.3, 0.2; center=[0.5, 0.1])

# Full area, full perimeter and full corner angles: outer square [-1,1]^2 (area 4, perimeter 8, four right angles) minus the discs.
const uf_RTOL = 1e-10

@testset "_has_reduced_holes" begin
    # holes and a nonempty registry
    @test _has_reduced_holes(uf_four_discs())
    @test _has_reduced_holes(uf_x_disc())
    @test _has_reduced_holes(uf_plain_disc())          # also for G' = {e}: the registry of the outer region is nonempty
    # empty registry: the old code path
    @test !_has_reduced_holes(uf_hole_billiard(DomainHole(uf_disc_domain(0.2, 0.3, 0.4, 5)), register_symmetries()))
    @test !_has_reduced_holes(uf_sinai_shifted())
    @test !_has_reduced_holes(uf_annulus_shifted())
    # origin-centered Sinai (D4) and annulus ({X}) are reduced since B7
    @test _has_reduced_holes(SinaiBilliard(1.0, 0.3))
    @test _has_reduced_holes(AnnularBilliard(0.3, 0.2))
    # billiards without holes
    @test !_has_reduced_holes(CircleBilliard(1.0))
    @test !_has_reduced_holes(RectangleBilliard(1.0, 0.5))
end

@testset "_weyl_area_length" begin
    # four discs, G' = D2 (order 4): the full values 4 - 4 pi r^2 and 8 + 4 (2 pi r), divided by 4 for the fundamental one
    b = uf_four_discs()
    A_full, L_full = 4 - 4 * pi * 0.01, 8 + 4 * 2pi * 0.1
    A, L = _weyl_area_length(b, false)
    @test A ≈ A_full rtol = uf_RTOL
    @test L ≈ L_full rtol = uf_RTOL
    A, L = _weyl_area_length(b, true)
    @test A ≈ A_full / 4 rtol = uf_RTOL
    @test L ≈ L_full / 4 rtol = uf_RTOL
    @test A ≈ fundamental_area(b) rtol = uf_RTOL
    @test L isa typeof(A)

    # the disc about (0.5, 0) with G' = {X} (order 2): the full values 4 - pi r^2 and 8 + 2 pi r, halved
    b2 = uf_x_disc()
    @test length(get_symmetry_group(b2)) == 1
    A2, L2 = _weyl_area_length(b2, true)
    @test A2 ≈ (4 - pi * 0.01) / 2 rtol = uf_RTOL
    @test L2 ≈ (8 + 2pi * 0.1) / 2 rtol = uf_RTOL

    # G' = {e}: the fundamental values are the full ones
    b1 = uf_plain_disc()
    A1, L1 = _weyl_area_length(b1, true)
    @test A1 ≈ 4 - pi * 0.04 rtol = uf_RTOL
    @test L1 ≈ 8 + 2pi * 0.2 rtol = uf_RTOL
    @test (A1, L1) == _weyl_area_length(b1, false)

    # no holes, no change: area and boundary of the fundamental domain
    c = CircleBilliard(1.0)
    Ac, Lc = _weyl_area_length(c, true)
    @test Ac == fundamental_area(c)
    @test Lc == sum(crv.length for crv in get_boundary_curves(c))
    Ac_full, Lc_full = _weyl_area_length(c, false)
    @test Ac_full ≈ pi rtol = uf_RTOL
    @test Lc_full == sum(crv.length for crv in full_boundary(c))
    # empty registry with holes (center-shifted Sinai): the full billiard is the fundamental one
    s = uf_sinai_shifted()
    As, Ls = _weyl_area_length(s, true)
    @test As ≈ 4 - pi * 0.09 rtol = uf_RTOL
    @test Ls ≈ 8 + 2pi * 0.3 rtol = uf_RTOL
    @test (As, Ls) == _weyl_area_length(s, false)
    sa = uf_annulus_shifted()
    @test _weyl_area_length(sa, true)[1] ≈ pi * (1 - 0.09) rtol = uf_RTOL
    @test _weyl_area_length(sa, true)[2] ≈ 2pi * (1 + 0.3) rtol = uf_RTOL
    @test _weyl_area_length(sa, true) == _weyl_area_length(sa, false)
    # origin-centered Sinai (G' = D4, order 8): the full values are 4 - pi R^2 and 8 + 2 pi R, the fundamental ones an eighth of them
    d4 = SinaiBilliard(1.0, 0.3)
    A8, L8 = _weyl_area_length(d4, true)
    A8_full, L8_full = _weyl_area_length(d4, false)
    @test A8_full ≈ 4 - pi * 0.09 rtol = uf_RTOL
    @test L8_full ≈ 8 + 2pi * 0.3 rtol = uf_RTOL
    @test A8 ≈ A8_full / 8 rtol = uf_RTOL
    @test L8 ≈ L8_full / 8 rtol = uf_RTOL
    @test A8 ≈ fundamental_area(d4) rtol = uf_RTOL
    # origin-centered annulus (G' = {X}, order 2)
    an = AnnularBilliard(0.3, 0.2)
    A2a, L2a = _weyl_area_length(an, true)
    @test _weyl_area_length(an, false)[1] ≈ pi * (1 - 0.09) rtol = uf_RTOL
    @test _weyl_area_length(an, false)[2] ≈ 2pi * (1 + 0.3) rtol = uf_RTOL
    @test A2a ≈ pi * (1 - 0.09) / 2 rtol = uf_RTOL
    @test L2a ≈ 2pi * (1 + 0.3) / 2 rtol = uf_RTOL
end

@testset "_weyl_angles" begin
    # fundamental = false: the four right angles of the outer square, holes have no corners
    @test _weyl_angles(uf_four_discs(), false) ≈ fill(pi / 2, 4)
    # fundamental = true: one corner per orbit of G', that is every |G'|-th of the sorted full-boundary angles
    @test _weyl_angles(uf_four_discs(), true) ≈ [pi / 2]          # |G'| = 4
    @test _weyl_angles(uf_x_disc(), true) ≈ fill(pi / 2, 2)       # |G'| = 2
    @test _weyl_angles(uf_plain_disc(), true) ≈ fill(pi / 2, 4)   # |G'| = 1
    @test length(_weyl_angles(uf_four_discs(), true)) == cld(length(_weyl_angles(uf_four_discs(), false)), 4)
    # no holes or empty registry: the corner angles of the billiard itself
    @test _weyl_angles(CircleBilliard(1.0), true) == corner_angles(CircleBilliard(1.0))
    @test _weyl_angles(RectangleBilliard(1.0, 0.5), false) == corner_angles(RectangleBilliard(1.0, 0.5); fundamental=false)
    @test _weyl_angles(uf_sinai_shifted(), true) == corner_angles(uf_sinai_shifted())
    @test _weyl_angles(uf_sinai_shifted(), true) ≈ fill(pi / 2, 4)
    # origin-centered Sinai (G' = D4, |G'| = 8): one corner per orbit of the four right angles of the outer square, fundamental or not
    @test _weyl_angles(SinaiBilliard(1.0, 0.3), false) ≈ fill(pi / 2, 4)
    @test _weyl_angles(SinaiBilliard(1.0, 0.3), true) ≈ [pi / 2]
    # annuli have no corners
    @test isempty(_weyl_angles(AnnularBilliard(0.3, 0.2), true))
    @test isempty(_weyl_angles(uf_annulus_shifted(), true))
end

@testset "_weyl_inputs" begin
    # the triple of _weyl_area_length and _weyl_angles
    for b in (uf_four_discs(), uf_x_disc(), uf_plain_disc()), fundamental in (true, false)
        A, L, angles = _weyl_inputs(b, fundamental)
        @test (A, L) == _weyl_area_length(b, fundamental)
        @test angles == _weyl_angles(b, fundamental)
    end
    # four discs: the reduced inputs are the full ones divided by |G'| = 4, with one corner instead of four
    b = uf_four_discs()
    Af, Lf, anglesf = _weyl_inputs(b, false)
    @test Af ≈ 4 - 4 * pi * 0.01 rtol = uf_RTOL
    @test Lf ≈ 8 + 4 * 2pi * 0.1 rtol = uf_RTOL
    @test anglesf ≈ fill(pi / 2, 4)
    A, L, angles = _weyl_inputs(b, true)
    @test A ≈ Af / 4 rtol = uf_RTOL
    @test L ≈ Lf / 4 rtol = uf_RTOL
    @test angles ≈ [pi / 2]
    # G' = {e}: identical to the unreduced inputs
    b1 = uf_plain_disc()
    @test _weyl_inputs(b1, true) == _weyl_inputs(b1, false)
    # billiards without holes, and the empty-registry (center-shifted) Sinai billiard, keep the old inputs
    c = CircleBilliard(1.0)
    @test _weyl_inputs(c, true) == (fundamental_area(c), sum(crv.length for crv in get_boundary_curves(c)), corner_angles(c))
    s = uf_sinai_shifted()
    @test _weyl_inputs(s, true) == (fundamental_area(s), sum(crv.length for crv in get_boundary_curves(s)), corner_angles(s))
    @test _weyl_inputs(s, false)[1] ≈ 4 - pi * 0.09 rtol = uf_RTOL
    # origin-centered Sinai (G' = D4): the reduced inputs are the full ones divided by 8, with one corner
    d4 = SinaiBilliard(1.0, 0.3)
    Af, Lf, anglesf = _weyl_inputs(d4, false)
    A8, L8, angles8 = _weyl_inputs(d4, true)
    @test Af ≈ 4 - pi * 0.09 rtol = uf_RTOL
    @test Lf ≈ 8 + 2pi * 0.3 rtol = uf_RTOL
    @test anglesf ≈ fill(pi / 2, 4)
    @test A8 ≈ Af / 8 rtol = uf_RTOL
    @test L8 ≈ Lf / 8 rtol = uf_RTOL
    @test angles8 ≈ [pi / 2]
end

@testset "Weyl accessors (billiards with holes)" begin
    b = uf_four_discs()
    A_full, L_full = 4 - 4 * pi * 0.01, 8 + 4 * 2pi * 0.1
    A, L, angles = A_full / 4, L_full / 4, [pi / 2]

    @testset "k_at_state" begin
        for N in (10, 50)
            @test k_at_state(N, b) ≈ k_at_state(N, A, L, angles) rtol = 1e-9
            @test k_at_state(N, b; fundamental=false) ≈ k_at_state(N, A_full, L_full, fill(pi / 2, 4)) rtol = 1e-9
        end
        # the fundamental domain is smaller: more wavenumber for the same state number
        @test k_at_state(50, b) > k_at_state(50, b; fundamental=false)
        @test all(diff([k_at_state(N, b) for N in 1:30]) .> 0)
    end

    @testset "state_at_k" begin
        k = 25.0
        @test state_at_k(k, b) ≈ state_at_k(k, A, L, angles) rtol = 1e-9
        @test state_at_k(k, b; fundamental=false) ≈ state_at_k(k, A_full, L_full, fill(pi / 2, 4)) rtol = 1e-9
        # the leading two terms scale as 1/|G'|: the reduced count is a quarter of the full one up to the corner constants
        smooth(kk, b_) = weyl_law(kk, _weyl_area_length(b_, true)...)
        @test smooth(k, b) ≈ weyl_law(k, A_full, L_full) / 4 rtol = 1e-9
        # round trip with k_at_state
        for N in (10.0, 100.0)
            @test state_at_k(k_at_state(N, b), b) ≈ N rtol = 1e-9
            @test state_at_k(k_at_state(N, b; fundamental=false), b; fundamental=false) ≈ N rtol = 1e-9
        end
    end

    @testset "spectral_density" begin
        k = 25.0
        @test spectral_density(k, b) ≈ spectral_density(k, A, L) rtol = 1e-9
        @test spectral_density(k, b; fundamental=false) ≈ spectral_density(k, A_full, L_full) rtol = 1e-9
        # (A k - L/2) scales exactly as 1/|G'| = 1/4
        @test spectral_density(k, b; fundamental=false) ≈ 4 * spectral_density(k, b) rtol = 1e-9
        h = 1e-3
        for fundamental in (true, false)
            fd = (state_at_k(k + h, b; fundamental) - state_at_k(k - h, b; fundamental)) / (2h)
            @test spectral_density(k, b; fundamental) ≈ fd rtol = 1e-8
        end
    end

    @testset "k_range_for_states" begin
        k1, k2 = k_range_for_states(b, 10, 100)
        @test (k1, k2) == (k_at_state(10, b), k_at_state(100, b))
        @test k1 < k2
        @test (k_range_for_states(b, 10, 100; fundamental=false)) == (k_at_state(10, b; fundamental=false), k_at_state(100, b; fundamental=false))
        # the second billiard (G' = {X}) uses its own |G'| = 2
        b2 = uf_x_disc()
        A2, L2 = (4 - pi * 0.01) / 2, (8 + 2pi * 0.1) / 2
        @test k_at_state(40, b2) ≈ k_at_state(40, A2, L2, fill(pi / 2, 2)) rtol = 1e-9
    end

    @testset "unchanged for billiards without holes" begin
        c = CircleBilliard(1.0)
        @test k_at_state(50, c) == k_at_state(50, fundamental_area(c), sum(crv.length for crv in get_boundary_curves(c)), corner_angles(c))
        @test state_at_k(20.0, c) == state_at_k(20.0, fundamental_area(c), sum(crv.length for crv in get_boundary_curves(c)), corner_angles(c))
        @test spectral_density(20.0, c) == spectral_density(20.0, fundamental_area(c), sum(crv.length for crv in get_boundary_curves(c)))
        s = uf_sinai_shifted()
        @test k_at_state(50, s) == k_at_state(50, fundamental_area(s), sum(crv.length for crv in get_boundary_curves(s)), corner_angles(s))
        @test k_at_state(50, s) == k_at_state(50, s; fundamental=false)    # empty registry: fundamental and full coincide
    end

    @testset "origin-centered Sinai (G' = D4)" begin
        d4 = SinaiBilliard(1.0, 0.3)
        A_full, L_full = 4 - pi * 0.09, 8 + 2pi * 0.3
        @test k_at_state(50, d4) ≈ k_at_state(50, A_full / 8, L_full / 8, [pi / 2]) rtol = 1e-9
        @test k_at_state(50, d4; fundamental=false) ≈ k_at_state(50, A_full, L_full, fill(pi / 2, 4)) rtol = 1e-9
        @test k_at_state(50, d4) > k_at_state(50, d4; fundamental=false)
    end
end
