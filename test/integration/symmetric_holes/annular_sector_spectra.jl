# Integration tests for symmetry-reduced Beyn spectra of the annular billiard with a hole on the x axis (symmetric holes, Phase 2).
# With the default center the registry is {XAxisReflection} (outer region: upper half disc, hole: SymmetricHole of two half-disc pieces
# about (shift, 0)), so G' = {X} for every shift. Two references:
#   - concentric (shift 0): analytic, the roots of J_m(k Ro) Y_m(k Ri) - Y_m(k Ro) J_m(k Ri) of annulus_spectrum (common.jl). The mode
#     cos(m phi) f(r) is even under y -> -y, sin(m phi) f(r) odd, so the sector chi_X = +1 holds every level once (m = 0 and m >= 1)
#     and the sector chi_X = -1 the levels with m >= 1 once; levels with m >= 1 have multiplicity 2 in the unreduced spectrum.
#   - eccentric (shift 0.2): independent construction, the translated billiard AnnularBilliard(0.3, 0.2; center=[0.5, 0.1]) (full outer
#     disc, DomainHole, empty registry); the union of the two X sectors must equal its spectrum.
# Kernel: composite CFIE with SmoothPeriodicGrading on the smooth outer loop and the hole loop, ppw 10, min_pts 96 per component.
# Windows: concentric [4, 12] as in bim_sweep/compositebim_holes.jl; eccentric [4, 11.6], the nearest levels (observed) are 11.46 and 11.79.

const ANNULAR_GRADING = SmoothPeriodicGrading()

# Distinct levels of the concentric annulus spectrum and their multiplicity (the generator repeats the same float for m >= 1).
function _an_levels_with_multiplicity(ks::Vector{Float64})
    uniq = unique(ks)
    return uniq, [count(==(k), ks) for k in uniq]
end

# solver: Beyn contour integral
# kernel/basis: composite CFIE (SmoothPeriodicGrading), ppw=10, min_pts=96
# billiard: Annular, concentric (AnnularBilliard(0.3, 0.0)), inner radius 0.3, outer radius 1
# symmetry: G' = {X}, sectors chi_X = +1 and -1
# reference: analytic, annulus_spectrum: chi_X = +1 the distinct levels, chi_X = -1 the levels of multiplicity 2
# functions to test: compute_spectrum, symmetry_sector (reduced composite solver with a SymmetricHole of two half-disc pieces)
@testset "Beyn (CFIE composite) - Annular concentric - X sectors vs Bessel roots" begin
    b = sh_billiard(:annular_concentric) do
        AnnularBilliard(0.3, 0.0)
    end
    @test length(get_symmetry_group(b)) == 1
    k1, k2 = 4.0, 12.0
    ks_ref = annulus_spectrum(0.3, 1.0, k1, k2)
    uniq, mult = _an_levels_with_multiplicity(ks_ref)
    even_ref = uniq
    odd_ref = uniq[mult .== 2]

    even_data = sh_spectrum(:annular_concentric, b, k1, k2, XAxisReflection => 1; grading=ANNULAR_GRADING, hole_grading=ANNULAR_GRADING)
    odd_data = sh_spectrum(:annular_concentric, b, k1, k2, XAxisReflection => -1; grading=ANNULAR_GRADING, hole_grading=ANNULAR_GRADING)
    even_ks, odd_ks = sh_levels(even_data), sh_levels(odd_data)

    @test length(even_ks) == length(even_ref)
    @test length(even_ks) == length(even_ref) && all(isapprox.(even_ks, even_ref; atol=SPECTRUM_ATOL_BIM))
    @test length(odd_ks) == length(odd_ref)
    @test length(odd_ks) == length(odd_ref) && all(isapprox.(odd_ks, odd_ref; atol=SPECTRUM_ATOL_BIM))
    @test sh_quality_ok(even_data) && sh_quality_ok(odd_data)
    # together the two sectors are the whole spectrum with multiplicities
    @test length(even_ks) + length(odd_ks) == length(ks_ref)
end

# solver: Beyn contour integral
# kernel/basis: composite CFIE (SmoothPeriodicGrading), ppw=10, min_pts=96
# billiard: Annular, hole displaced along the x axis (AnnularBilliard(0.3, 0.2)) against the translated Phase 1 construction
# symmetry: G' = {X}, sectors chi_X = +1 and -1, against None
# reference: independent construction, AnnularBilliard(0.3, 0.2; center=[0.5, 0.1]) (empty registry, DomainHole), and the unreduced path
# functions to test: compute_spectrum, symmetry_sector (reduced composite solver, shifted SymmetricHole), unreduced path with a registry
@testset "Beyn (CFIE composite) - Annular eccentric - X sectors vs translated Phase 1 construction" begin
    b = sh_billiard(:annular_eccentric) do
        AnnularBilliard(0.3, 0.2)
    end
    translated = sh_billiard(:annular_eccentric_translated) do
        AnnularBilliard(0.3, 0.2; center=[0.5, 0.1])
    end
    @test isempty(translated.symmetries) && length(get_symmetry_group(b)) == 1
    k1, k2 = 4.0, 11.6
    kw = (; grading=ANNULAR_GRADING, hole_grading=ANNULAR_GRADING)
    ref_data = sh_spectrum(:annular_eccentric_translated, translated, k1, k2; kw...)
    ref = sh_levels(ref_data)
    @test !isempty(ref) && sh_quality_ok(ref_data)

    full_data = sh_spectrum(:annular_eccentric, b, k1, k2; kw...)
    @test length(sh_levels(full_data)) == length(ref)
    @test length(sh_levels(full_data)) == length(ref) && all(isapprox.(sh_levels(full_data), ref; atol=SPECTRUM_ATOL_SECTOR))

    sectors = [sh_spectrum(:annular_eccentric, b, k1, k2, XAxisReflection => c; kw...) for c in (1, -1)]
    @test all(sh_quality_ok, sectors)
    @test all(d -> !isempty(d.k), sectors)
    @test sh_union_matches([sh_levels(d) for d in sectors], ref; atol=SPECTRUM_ATOL_SECTOR)
end
