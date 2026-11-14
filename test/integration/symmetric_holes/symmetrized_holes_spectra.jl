# Integration tests for SymmetrizedHoles (symmetric holes, Phase 2): four discs of radius 0.1 placed by D2_symmetry() about (0.4, 0.2)
# in the quarter-square billiard of RectangleBilliard(1.0, 1.0) (full frame [-1,1]^2), G' = D2 (holes_fixtures.jl,
# hf_symmetrized_disc_billiard). Two references: the unreduced Beyn spectrum of the same kernel for the sectors (cross-method), and the
# explicit full-frame billiard with four DomainHoles at (+-0.4, +-0.2) and an empty registry (Phase 1 construction) for the placement.
# Kernel: composite CFIE, GlobalCornerGrading(kressq=2) on the outer square, ppw 10, min_pts 96 per component; the four copies share
# one hole component solver (one hole object), the explicit billiard has four. Window [4, 9]: nearest levels to the edges (observed) 4.18
# and 8.94.

@requires "integration/symmetric_holes/helpers.jl"

# solver: Beyn contour integral
# kernel/basis: composite CFIE (outer square graded, hole copies share one solver), ppw=10, min_pts=96
# billiard: D2 quarter square with SymmetrizedHoles(DomainHole(disc), D2_symmetry()), four hole loops in one group
# symmetry: the four D2 sectors (chi_Y, chi_X) = (+-1, +-1) against None
# reference: cross-method, unreduced Beyn spectrum of the same kernel
# functions to test: compute_spectrum, symmetry_sector (reduced composite solver with hole copies; node counts forced equal across copies)
@testset "Beyn (CFIE composite) - SymmetrizedHoles - union of D2 sectors equals unreduced spectrum" begin
    b = sh_billiard(:symmetrized_discs) do
        hf_symmetrized_disc_billiard()
    end
    @test length(get_symmetry_group(b)) == 3
    k1, k2 = 4.0, 9.0
    full_data = sh_spectrum(:symmetrized_discs, b, k1, k2)
    full = sh_levels(full_data)
    @test length(full) > 10
    @test sh_quality_ok(full_data)

    sectors = [sh_spectrum(:symmetrized_discs, b, k1, k2, YAxisReflection => cy, XAxisReflection => cx) for cy in (1, -1), cx in (1, -1)]
    @test all(sh_quality_ok, sectors)
    @test all(d -> !isempty(d.k), sectors)
    @test sh_union_matches([sh_levels(d) for d in sectors], full; atol=SPECTRUM_ATOL_SECTOR)
end

# solver: Beyn contour integral
# kernel/basis: composite CFIE, ppw=10, min_pts=96 (one hole component per DomainHole in the explicit billiard)
# billiard: SymmetrizedHoles placement against four explicit DomainHoles on the full square [-1,1]^2 (empty registry)
# symmetry: None for both
# reference: independent construction, explicit DomainHoles at (+-0.4, +-0.2), radius 0.1
# functions to test: compute_spectrum (unreduced path with hole copies), SymmetrizedHoles placement
@testset "Beyn (CFIE composite) - SymmetrizedHoles placement vs explicit DomainHoles - identical spectra" begin
    placed = sh_billiard(:symmetrized_discs) do
        hf_symmetrized_disc_billiard()
    end
    explicit = sh_billiard(:explicit_four_discs) do
        sh_full_square_billiard(DomainHole[DomainHole(hf_circle_domain(0.1, x, y, id)) for (id, (x, y)) in
                                          zip(5:8, ((0.4, 0.2), (-0.4, 0.2), (-0.4, -0.2), (0.4, -0.2)))])
    end
    @test isempty(explicit.symmetries)
    k1, k2 = 4.0, 9.0
    ref_data = sh_spectrum(:explicit_four_discs, explicit, k1, k2; nholes=4)
    ref = sh_levels(ref_data)
    @test !isempty(ref) && sh_quality_ok(ref_data)

    full = sh_levels(sh_spectrum(:symmetrized_discs, placed, k1, k2))
    @test length(full) == length(ref)
    @test length(full) == length(ref) && all(isapprox.(full, ref; atol=SPECTRUM_ATOL_SECTOR))
end
