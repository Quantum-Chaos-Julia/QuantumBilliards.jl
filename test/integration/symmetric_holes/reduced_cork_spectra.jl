# Integration test for CORKSolver on a symmetry-reduced composite kernel with a symmetric hole (Phase 2): the sector
# (chi_X, chi_Y) = (+1, +1) of the Sinai billiard (D4 registry, SymmetricHole). The reference is the Beyn spectrum of the same sector
# (cross-method, see sinai_sector_spectra.jl), whose window is SINAI_WINDOW = [4, 9] with 4 levels.
#
# The reduced matrix is small (N = 100 at `min_pts = 200` for the composite kernel) and the Weyl count of a window of
# `_cork_intervals` is a fraction of a level, so the test checks that `compute_spectrum` returns every reduced level and does not
# abort on windows that hold none (steps 1 and 2 of scratchpad/cork-reduced-convergence-plan.md). Settings (min_pts = 200,
# nlevels = 4, kressq = 2, ppw = 10) were calibrated: nlevels = 2, 4, 8 and min_pts = 150, 200, 300 return the four levels with
# worst |Im k| 1e-9 (min_pts = 96 gives 2.8e-8 and one candidate is marked unverified). The window edges are those of
# `_cork_intervals` (fixed by the Weyl layout, not shifted). The CORK levels differ from the reduced Beyn reference (min_pts = 96 in
# helpers.jl) by 2.1e-7 at most, 50x below SPECTRUM_ATOL_SECTOR. reduced_ebim_spectra.jl uses the same kernel settings.

@requires "integration/symmetric_holes/helpers.jl"

# solver: CORK
# kernel/basis: composite CFIE (outer square GlobalCornerGrading(kressq=2), disc hole), ppw=10, min_pts=200, nlevels=4
# billiard: Sinai (SinaiBilliard(1.0, 0.25))
# symmetry: D2 sector (chi_X, chi_Y) = (+1, +1) of D4
# reference: cross-method, reduced Beyn spectrum of the same sector in [4, 9] (4 levels)
# functions to test: compute_spectrum (CORKSolver with a reduced composite kernel)
@testset "CORK (CFIE composite) - Sinai - D2 sector (+1, +1) vs reduced Beyn" begin
    b = sh_sinai()
    choices = (XAxisReflection => 1, YAxisReflection => 1)
    ref = sh_levels(sh_sinai_spectrum(choices...))
    @test length(ref) == 4

    sec = symmetry_sector(b, choices...)
    solver = CORKSolver(sh_composite_kernel(sec; min_pts=200); nlevels=4)
    data = compute_spectrum(solver, b, SINAI_WINDOW...; multithreaded=false, show_progress=false)
    ks = sort(real.(data.k))
    @test length(ks) == length(ref)
    @test all(isapprox.(ks, ref; atol=SPECTRUM_ATOL_SECTOR))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL_HOLES)
end

# solver: CORK, states = :boundary
# kernel/basis: composite CFIE (outer square GlobalCornerGrading(kressq=2), disc hole), ppw=10, min_pts=200, nlevels=4
# billiard: Sinai (SinaiBilliard(1.0, 0.25))
# symmetry: D2 sector (chi_X, chi_Y) = (+1, +1) of D4; u is expanded over the orbit copies of the outer loop and the hole
# reference: solve_state u on the stored pts (same kernel and sector); reduced Beyn levels of the same sector in [4, 9]
# functions to test: compute_spectrum (CORKSolver, states = :boundary with a reduced composite kernel), _cork_left_vectors, _boundary_states
@testset "CORK (CFIE composite) - Sinai - D2 sector (+1, +1) states = :boundary stores u equal to solve_state" begin
    b = sh_sinai()
    choices = (XAxisReflection => 1, YAxisReflection => 1)
    ref = sh_levels(sh_sinai_spectrum(choices...))
    sec = symmetry_sector(b, choices...)
    kernel = sh_composite_kernel(sec; min_pts=200)
    solver = CORKSolver(kernel; nlevels=4, states=:boundary)
    data = compute_spectrum(solver, b, SINAI_WINDOW...; multithreaded=false, show_progress=false)
    ks = sort(real.(data.k))
    @test length(ks) == length(ref)
    @test all(isapprox.(ks, ref; atol=SPECTRUM_ATOL_SECTOR))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL_HOLES)
    @test boundary_states_wellformed(data)
    res, idx = stored_u_residuals(data, kernel, b)
    @test length(idx) == length(ref)
    @test all(res .< BOUNDARY_U_TOL_BIM)
end
