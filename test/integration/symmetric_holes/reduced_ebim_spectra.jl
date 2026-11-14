# Integration test for ExpandedBIMSolver on a symmetry-reduced composite kernel with a symmetric hole (Phase 2): the sector
# (chi_X, chi_Y) = (+1, +1) of the Sinai billiard (D4 registry, SymmetricHole). The reference is the Beyn spectrum of the same sector
# (cross-method, see sinai_sector_spectra.jl). EBIM returns spurious candidates (tension above 2) and no count is asserted: as in
# accelerated/ebim_spectra.jl, every reference level must be matched by a confident level and every confident level must lie at a
# reference level. `accept_tol = nothing` and `min_pts = 200` are needed: with the default EBIM tension filter (5e-4) no level is
# returned (not specific to the reduction, observed also with the unreduced kernel).
# The reduced CORKSolver is deliberately not tested: reduced CORK fails with "Reached full CORK Krylov dimension" even for a hole-free
# rectangle (known bug in src/, see the report of the Julia Test Writer; unreduced CORK works).

@requires "integration/symmetric_holes/helpers.jl"

# solver: Expanded BIM
# kernel/basis: composite CFIE (outer square graded, disc hole), ppw=10, min_pts=200
# billiard: Sinai (SinaiBilliard(1.0, 0.25))
# symmetry: D2 sector (chi_X, chi_Y) = (+1, +1) of D4
# reference: cross-method, reduced Beyn spectrum of the same sector in [4, 9] (4 levels)
# functions to test: compute_spectrum (ExpandedBIMSolver with a reduced composite kernel)
@testset "EBIM (CFIE composite) - Sinai - D2 sector (+1, +1) vs reduced Beyn" begin
    b = sh_sinai()
    choices = (XAxisReflection => 1, YAxisReflection => 1)
    ref = sh_levels(sh_sinai_spectrum(choices...))
    @test !isempty(ref)

    sec = symmetry_sector(b, choices...)
    solver = ExpandedBIMSolver(sh_composite_kernel(sec; min_pts=200))
    data = compute_spectrum(solver, b, SINAI_WINDOW...; accept_tol=nothing, multithreaded=true, show_progress=false)
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ref)
    @test all(i -> minimum(abs.(ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
end
