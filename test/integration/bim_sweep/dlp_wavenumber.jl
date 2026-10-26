# Integration tests for DoubleLayerPotentialSolver (src/solvers/sweepmethods/dlp.jl).
# Sweep solver: `solve_wavenumber` only. Parameters from the calibration log of
# scratchpad/test-suite-redesign-plan.md. Each window `k0 +- dk` contains exactly one level.

# solver: Double Layer Potential
# kernel/basis: DLP, GlobalCornerGrading
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), full rectangle 2.0 x 1.4)
# symmetry: None
# reference: analytic, pi*sqrt(1/2.0^2 + 1/1.4^2)
# functions to test: solve_wavenumber
@testset "DLP - Rectangle - Ground State" begin
    billiard = RectangleBilliard(1.0, 0.7)
    solver = DoubleLayerPotentialSolver(10.0, billiard; grading=GlobalCornerGrading())
    k_ref = rectangle_spectrum(2.0, 1.4, 0.0, 4.0)[1]
    k, t = solve_wavenumber(solver, billiard, k_ref + 0.03, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end

# solver: Double Layer Potential
# kernel/basis: DLP, SmoothPeriodicGrading
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, Bessel zero j_{0,2}/R (the neighbouring levels 5.1356 and 6.3802 lie outside the window)
# functions to test: solve_wavenumber
@testset "DLP - Circle - Second J0 State" begin
    billiard = CircleBilliard(1.0)
    solver = DoubleLayerPotentialSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    k_ref = circle_spectrum(1.0, 5.4, 5.6)[1]
    k, t = solve_wavenumber(solver, billiard, 5.52, 0.2; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end

# solver: Double Layer Potential
# kernel/basis: DLP, GlobalCornerGrading
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762 (tension 3e-15, ppw 6 and 10 agree; it is the only level in [5.5, 6.7]).
#   The old solvertests.jl values 6.0650909 / 6.0650910 are NOT used.
# functions to test: solve_wavenumber
@testset "DLP - Full Triangle - Ground State" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    solver = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
    k_ref = 6.065090986269762
    k, t = solve_wavenumber(solver, billiard, 6.1, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end
