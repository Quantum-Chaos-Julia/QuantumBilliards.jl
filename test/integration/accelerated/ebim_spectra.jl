# Integration tests for ExpandedBIMSolver (src/solvers/acceleratedmethods/ebim.jl) through `compute_spectrum`.
# EBIM returns spurious candidates (tension above 2) and duplicates of genuine levels near window edges
# (finding F1 of scratchpad/test-suite-redesign-plan.md), so no count or duplicate assertions are made. Instead:
#   1. every reference level is matched by a returned level with tension < TENSION_TOL_EBIM within SPECTRUM_ATOL_EBIM;
#   2. no returned level with tension < TENSION_TOL_EBIM lies further than SPECTRUM_ATOL_EBIM from a reference level.

# solver: Expanded BIM
# kernel/basis: DLP, GlobalCornerGrading, ppw=5
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762 (it is the only level in [5.5, 6.7])
# functions to test: compute_spectrum
@testset "EBIM (DLP) - Full Triangle - Ground State" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    kernel = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
    solver = ExpandedBIMSolver(kernel)
    k1, k2 = 5.5, 6.7
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    ks_ref = [6.065090986269762]
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
    @test all(i -> minimum(abs.(ks_ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
end

# solver: Expanded BIM
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, j_{0,1} = 2.4048 (window [2, 3] holds it alone; the next level 3.8317 lies outside)
# functions to test: compute_spectrum
@testset "EBIM (CFIE) - Circle - Ground State" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    solver = ExpandedBIMSolver(kernel)
    k1, k2 = 2.0, 3.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    ks_ref = circle_spectrum(1.0, k1, k2)
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test length(ks_ref) == 1
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
    @test all(i -> minimum(abs.(ks_ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
end

# solver: Expanded BIM
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; the window [3, 10] holds 16 levels
# functions to test: compute_spectrum
@testset "EBIM (CFIE) - Rectangle - Spectrum k in [3, 10]" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    solver = ExpandedBIMSolver(kernel)
    k1, k2 = 3.0, 10.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    ks_ref = rectangle_spectrum(2.0, 1.4, k1, k2)
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test length(ks_ref) == 16
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
    @test all(i -> minimum(abs.(ks_ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
end

# solver: Expanded BIM
# kernel/basis: DLP, GlobalCornerGrading, ppw=5 (a plain DLP kernel has a rank-deficient A'(k), so the generalized
#   eigenproblem (A, A') reports NaN eigenvalues; selecting the nearest candidate with a plain argmin used to lock onto one)
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762
# functions to test: solve_wavenumber (NaN regression ported from the old chebyshevtests.jl; the old `_argmin_finite` is
#   now inlined in solve_wavenumber)
@testset "EBIM (DLP) - NaN regression - Full Triangle" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    kernel = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
    solver = ExpandedBIMSolver(kernel)
    k_ref = 6.065090986269762
    k, t = solve_wavenumber(solver, billiard, k_ref, 0.05; multithreaded=false)
    @test k !== nothing
    @test !isnan(real(k))
    @test !isnan(t)
    @test isapprox(real(k), k_ref; atol=SPECTRUM_ATOL_EBIM)
    @test t < TENSION_TOL_EBIM
end
