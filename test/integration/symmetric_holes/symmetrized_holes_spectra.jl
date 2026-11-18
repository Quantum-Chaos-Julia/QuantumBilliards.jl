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

# solver: Beyn contour integral, states = :boundary
# kernel/basis: composite CFIE (outer square graded, hole copies share one solver), ppw=10, min_pts=96, symmetry-reduced to the D2 sector (+1, +1)
# billiard: D2 quarter square with SymmetrizedHoles(DomainHole(disc), D2_symmetry())
# symmetry: D2 sector (chi_Y, chi_X) = (1, 1); u is expanded with the orbit phases over the four hole copies and the outer loop
# reference: solve_state u on the stored pts (same kernel and sector; compared for levels below k = 7.5), levels of the states = :none run of the same sector (sh_spectrum cache)
# functions to test: compute_spectrum (states = :boundary, composite reduced kernel with hole copies), _bim_normal_derivative
@testset "Beyn (CFIE composite) - SymmetrizedHoles - D2 sector (1, 1) states = :boundary stores u equal to solve_state" begin
    b = sh_billiard(:symmetrized_discs) do
        hf_symmetrized_disc_billiard()
    end
    k1, k2 = 4.0, 9.0
    choices = (YAxisReflection => 1, XAxisReflection => 1)
    ref = sh_spectrum(:symmetrized_discs, b, k1, k2, choices...)
    sector = symmetry_sector(b, choices...)
    kernel = sh_composite_kernel(sector)
    solver = BeynSolver(kernel; use_chebyshev=false, states=:boundary)
    data = compute_spectrum(solver, b, k1, k2; multithreaded=false, show_progress=false)
    @test !isempty(data.k)
    @test length(data) == length(ref.k)
    @test all(isapprox.(sh_levels(data), sh_levels(ref); atol=SPECTRUM_ATOL_SECTOR))
    @test sh_quality_ok(data)
    @test boundary_states_wellformed(data)
    res, idx = stored_u_residuals(data, kernel, b)
    @test !isempty(idx)
    low = real.(data.k)[idx] .< 7.5                                           # u agreement degrades with k at ppw=10 (2e-3 at k = 8.9)
    @test count(low) >= 3
    @test all(res[low] .< BOUNDARY_U_TOL_BIM)
end
