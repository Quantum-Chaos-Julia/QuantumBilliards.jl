# Integration tests for the trivial-group fallback of symmetric holes (Phase 2): a billiard whose holes break the registry has
# G' = {e}; its unreduced path (symmetry = nothing, the whole boundary) must work and give the physical spectrum, and every reduced request
# must be refused with an ArgumentError naming G' (a silent fallback is rejected by design: reduced and unreduced solves return different
# spectra). Two cases:
#   - D2 quarter square with an off-axis DomainHole (hf_domain_hole_billiard): reference is the same disc in a full-frame billiard with an
#     empty registry (independent construction).
#   - C3Billiard(0.2) with a disc declared as a C5 SymmetricHole (G' = {e}, since C3 is not a subgroup of C5): the disc is the same
#     geometry as in the C6 case of c3_rotation_hole_spectra.jl, so the spectra must coincide with the unreduced C6 spectrum.
# Kernel and windows: composite CFIE as in the other files of the group; [4, 9] for the square, [8, 16] for the C3 billiard.

# solver: Beyn contour integral
# kernel/basis: composite CFIE (outer square graded), ppw=10, min_pts=96
# billiard: D2 quarter square with an off-axis DomainHole (disc of radius 0.1 about (0.4, 0.2)), G' = {e}
# symmetry: None on a billiard with a registry
# reference: independent construction, the same disc in the full square [-1,1]^2 with an empty registry
# functions to test: compute_spectrum (unreduced path with G' = {e}), symmetry_sector (G' error)
@testset "Beyn (CFIE composite) - D2 square with DomainHole - G' = {e} fallback matches full-frame billiard" begin
    b = sh_billiard(:domain_hole_d2) do
        hf_domain_hole_billiard()
    end
    plain = sh_billiard(:domain_hole_plain) do
        sh_full_square_billiard(DomainHole[DomainHole(hf_circle_domain(0.1, 0.4, 0.2, 5))])
    end
    @test !isempty(b.symmetries) && isempty(get_symmetry_group(b))
    k1, k2 = 4.0, 9.0
    data = sh_spectrum(:domain_hole_d2, b, k1, k2)
    ref_data = sh_spectrum(:domain_hole_plain, plain, k1, k2)
    @test sh_quality_ok(data) && sh_quality_ok(ref_data)
    ks, ref = sh_levels(data), sh_levels(ref_data)
    @test !isempty(ref)
    @test length(ks) == length(ref)
    @test length(ks) == length(ref) && all(isapprox.(ks, ref; atol=SPECTRUM_ATOL_SECTOR))

    @test_throws ArgumentError symmetry_sector(b, XAxisReflection => 1)
    @test_throws "G'" symmetry_sector(b, YAxisReflection => 1)
end

# solver: Beyn contour integral
# kernel/basis: composite CFIE (SmoothPeriodicGrading), ppw=10, min_pts=96
# billiard: C3Billiard(0.2) with a disc hole declared as a C5 SymmetricHole (G' = {e}) against the C6 declaration (G' = C3)
# symmetry: None for both
# reference: cross-method, the unreduced Beyn spectrum of the C6 declaration (same geometry)
# functions to test: compute_spectrum (unreduced path with G' = {e} and a rotational registry), symmetry_sector (G' error)
@testset "Beyn (CFIE composite) - C3 with C5 disc hole - G' = {e} fallback matches the C6 declaration" begin
    b5 = sh_billiard(:c3_c5_disc) do
        sh_c3_disc_billiard(5, 0.15)
    end
    b6 = sh_c3_c6_billiard()
    @test !isempty(b5.symmetries) && isempty(get_symmetry_group(b5))
    data5 = sh_c3_spectrum(:c3_c5_disc, b5)
    data6 = sh_c3_spectrum(:c3_c6_disc, b6)
    @test sh_quality_ok(data5) && sh_quality_ok(data6)
    ks5, ks6 = sh_levels(data5), sh_levels(data6)
    @test !isempty(ks6)
    @test length(ks5) == length(ks6)
    @test length(ks5) == length(ks6) && all(isapprox.(ks5, ks6; atol=SPECTRUM_ATOL_SECTOR))

    @test_throws ArgumentError symmetry_sector(b5, NFoldRotation => 0)
    @test_throws "G'" symmetry_sector(b5, NFoldRotation => 1)
end
