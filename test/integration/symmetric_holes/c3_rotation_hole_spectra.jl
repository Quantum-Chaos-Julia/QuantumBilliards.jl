# Integration test for a rotational symmetric hole (symmetric holes, Phase 2): C3Billiard(0.2) (outer sector with SymmetryWall radii, registry
# Cn_symmetry(3)) with a disc of radius 0.15 about the origin declared as a C6 SymmetricHole (wedge of angle pi/3 and Cn_symmetry(6)).
# C3 is a subgroup of C6, so G' = C3. The rotation sector s in 0:2 of the generator NFoldRotation(3, 1) has character
# exp(2 pi i s / 3); sector 0 is the non-degenerate irrep A, sectors 1 and 2 are the complex-conjugate pair E that time reversal makes
# degenerate (as in symmetry/rotation_spectra.jl): each level of sectors 1 and 2 is a doublet of the unreduced spectrum and the
# sector-0 levels occur once. Reference: cross-method, the unreduced Beyn spectrum of the same kernel.
# Kernel: composite CFIE, SmoothPeriodicGrading on the smooth outer and hole loops, ppw 10, min_pts 96 per component. The disc is
# the same geometry for every n, see trivial_group_fallback.jl. Window [8, 16]: the first level of the billiard with the hole is near 9.7
# (the hole raises the levels; the hole-free C3Billiard(0.2) window of rotation_spectra.jl is [3, 12]).

# solver: Beyn contour integral
# kernel/basis: composite CFIE (SmoothPeriodicGrading), ppw=10, min_pts=96
# billiard: C3Billiard(0.2) with a disc hole declared as a C6 SymmetricHole (G' = C3)
# symmetry: C3 rotation sectors s = 0, 1, 2 against None
# reference: cross-method, unreduced Beyn spectrum of the same kernel; sector 0 levels single, sectors 1 and 2 doublets
# functions to test: compute_spectrum, symmetry_sector (NFoldRotation sector with a rotational SymmetricHole), unreduced path
@testset "Beyn (CFIE composite) - C3 with C6 disc hole - union of rotation sectors equals unreduced spectrum" begin
    b = sh_c3_c6_billiard()
    @test length(get_symmetry_group(b)) == 2                  # C3 without the identity
    full_data = sh_c3_spectrum(:c3_c6_disc, b)
    full = sh_levels(full_data)
    @test length(full) >= 6
    @test sh_quality_ok(full_data)

    sectors = [sh_c3_spectrum(:c3_c6_disc, b, NFoldRotation => s) for s in 0:2]
    @test all(sh_quality_ok, sectors)
    @test all(d -> !isempty(d.k), sectors)
    s0, s1, s2 = (sh_levels(d) for d in sectors)
    @test sh_union_matches([s0, s1, s2], full; atol=SPECTRUM_ATOL_SECTOR)
    # time reversal: sectors 1 and 2 have the same levels, which are doublets of the unreduced spectrum; sector 0 levels are single
    @test length(s1) == length(s2)
    @test length(s1) == length(s2) && all(isapprox.(s1, s2; atol=SPECTRUM_ATOL_SECTOR))
    @test all(k -> count(abs.(full .- k) .< SPECTRUM_ATOL_SECTOR) == 2, s1)
    @test all(k -> count(abs.(full .- k) .< SPECTRUM_ATOL_SECTOR) == 1, s0)
end
