# Integration tests for symmetry-reduced Beyn spectra of the Sinai billiard (symmetric holes, Phase 2): SinaiBilliard(1.0, 0.25) has the
# D4 registry of the outer square and a SymmetricHole (disc of radius 0.25) with G' = D4. The unreduced spectrum (symmetry = nothing,
# the whole boundary) is the reference of every sector; a second, independent reference is the Phase 1 construction, the translated
# billiard SinaiBilliard(1.0, 0.25; center=[0.2, 0.1]) that keeps the empty registry and a DomainHole (spectrum is translation invariant).
# Kernel: composite CFIE, GlobalCornerGrading(kressq=2) on the outer square, ppw 10, min_pts 96 per component (calibrated in
# scratchpad/sinai_reduced_check.jl). The window [4, 9] has its edges away from levels (nearest 4.54 and 8.81, unreduced spectrum).
# D4 facts used: the 1D irreps are fixed by the characters (chi_Y, chi_Diagonal) of the generators of CompositeReflection(Y, Diagonal),
# X and AntiDiagonal then carry chi_Y and chi_Diagonal; the doubly degenerate E levels belong to no 1D irrep. The D2 subgroup {X, Y, XY}
# of D4 has four 1D irreps that contain every level of D4 (E levels split into the (+,-) and (-,+) sectors); each D2 sector chi_X = chi_Y
# is the union of the two D4 irreps with that chi_Y, and the sectors (+,-) and (-,+) are isospectral through the diagonal reflection.

# solver: Beyn contour integral
# kernel/basis: composite CFIE (outer square graded, disc hole), ppw=10, min_pts=96
# billiard: Sinai (SinaiBilliard(1.0, 0.25))
# symmetry: the four D2-subgroup sectors (chi_X, chi_Y) = (+-1, +-1) of D4, against None
# reference: cross-method, the unreduced Beyn spectrum of the same kernel; the union of the four sectors contains every level
# functions to test: compute_spectrum, symmetry_sector (reduced CompositeBIMSolver with a SymmetricHole)
@testset "Beyn (CFIE composite) - Sinai - union of D2 sectors equals unreduced spectrum" begin
    b = sh_sinai()
    full_data = sh_sinai_spectrum()
    full = sh_levels(full_data)
    @test length(full) > 10
    @test sh_quality_ok(full_data)

    sectors = Dict((cx, cy) => sh_sinai_spectrum(XAxisReflection => cx, YAxisReflection => cy) for cx in (1, -1), cy in (1, -1))
    @test all(sh_quality_ok, values(sectors))
    @test sh_union_matches([sh_levels(d) for d in values(sectors)], full; atol=SPECTRUM_ATOL_SECTOR)
    @test all(d -> !isempty(d.k), values(sectors))

    # the diagonal reflection exchanges x and y, so the sectors (+,-) and (-,+) have the same levels
    ks_pm, ks_mp = sh_levels(sectors[(1, -1)]), sh_levels(sectors[(-1, 1)])
    @test length(ks_pm) == length(ks_mp)
    @test length(ks_pm) == length(ks_mp) && all(isapprox.(ks_pm, ks_mp; atol=SPECTRUM_ATOL_SECTOR))
end

# solver: Beyn contour integral
# kernel/basis: composite CFIE (outer square graded, disc hole), ppw=10, min_pts=96
# billiard: Sinai (SinaiBilliard(1.0, 0.25))
# symmetry: the four 1D irreps of D4 (chi_Y, chi_Diagonal) = (+-1, +-1) (A1 = (+,+), A2 = (-,-), B1 and B2 mixed) against None
# reference: cross-method; D4 irrep levels are unreduced levels and the remaining unreduced levels are the doubly degenerate E levels;
#   the D2 sectors (+,+) and (-,-) are unions of the D4 irreps with chi_Y = +1 and -1, the D2 sector (+,-) holds one copy of each E level
# functions to test: compute_spectrum, symmetry_sector (DiagonalReflection and YAxisReflection characters of a D4 registry)
@testset "Beyn (CFIE composite) - Sinai - D4 irreps A1 A2 B1 B2 and E doublets" begin
    b = sh_sinai()
    full = sh_levels(sh_sinai_spectrum())
    irreps = Dict((cd, cy) => sh_sinai_spectrum(DiagonalReflection => cd, YAxisReflection => cy) for cd in (1, -1), cy in (1, -1))
    @test all(sh_quality_ok, values(irreps))
    @test all(d -> !isempty(d.k), values(irreps))
    irrep_levels = Dict(key => sh_levels(d) for (key, d) in irreps)

    # every irrep level is an unreduced level; the rest of the unreduced spectrum is doubly degenerate (the 2D irrep E)
    all_irrep = sort(reduce(vcat, values(irrep_levels)))
    worst, left = sh_match_levels(all_irrep, full)
    @test worst < SPECTRUM_ATOL_SECTOR
    @test !isempty(left) && sh_are_doublets(left, SPECTRUM_ATOL_SECTOR)

    # the D2 sector (chi_X = chi_Y = c) is the union of the two D4 irreps with chi_Y = c
    for c in (1, -1)
        d2 = sh_levels(sh_sinai_spectrum(XAxisReflection => c, YAxisReflection => c))
        @test sh_union_matches([irrep_levels[(1, c)], irrep_levels[(-1, c)]], d2; atol=SPECTRUM_ATOL_SECTOR)
    end
    # the E levels (one copy each) are the D2 sector (+,-)
    e_single = left[1:2:end]
    d2_mixed = sh_levels(sh_sinai_spectrum(XAxisReflection => 1, YAxisReflection => -1))
    @test length(d2_mixed) == length(e_single)
    @test length(d2_mixed) == length(e_single) && all(isapprox.(d2_mixed, e_single; atol=SPECTRUM_ATOL_SECTOR))
end

# solver: Beyn contour integral
# kernel/basis: composite CFIE (outer square graded, disc hole), ppw=10, min_pts=96
# billiard: Sinai (SinaiBilliard(1.0, 0.25) against the translated SinaiBilliard(1.0, 0.25; center=[0.2, 0.1]), Phase 1 construction)
# symmetry: None (unreduced on the D4 registry) and the D2 sectors against the translated billiard without registry
# reference: independent construction, the translated billiard (full square outer region, DomainHole, empty registry)
# functions to test: compute_spectrum (unreduced path of a billiard with a SymmetricHole), symmetry_sector
@testset "Beyn (CFIE composite) - Sinai - symmetric construction vs translated Phase 1 construction" begin
    b = sh_sinai()
    translated = sh_billiard(:sinai_translated) do
        SinaiBilliard(1.0, 0.25; center=[0.2, 0.1])
    end
    @test isempty(translated.symmetries)
    ref_data = sh_spectrum(:sinai_translated, translated, SINAI_WINDOW...)
    ref = sh_levels(ref_data)
    @test sh_quality_ok(ref_data)

    full = sh_levels(sh_sinai_spectrum())
    @test length(full) == length(ref)
    @test length(full) == length(ref) && all(isapprox.(full, ref; atol=SPECTRUM_ATOL_SECTOR))
    sectors = [sh_levels(sh_sinai_spectrum(XAxisReflection => cx, YAxisReflection => cy)) for cx in (1, -1), cy in (1, -1)]
    @test sh_union_matches(sectors, ref; atol=SPECTRUM_ATOL_SECTOR)
end
