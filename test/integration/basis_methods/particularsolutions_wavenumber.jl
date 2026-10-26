# Integration tests for ParticularSolutionsMethodSolver (src/solvers/sweepmethods/particularsolutions.jl).
# Sweep solver: `solve_wavenumber` only. Parameters from the calibration log of
# scratchpad/test-suite-redesign-plan.md.

# solver: Particular Solutions Method
# basis: Fourier-Bessel
# billiard: Circle, radius 1
# symmetry: even-even
# reference: analytic, Bessel zero j_{0,1}
# functions to test: solve_wavenumber
@testset "PSM (FourierBessel) - Circle, even-even sector - First J0 State" begin
    billiard = CircleBilliard(1.0)
    basis = FourierBessel(20; sym_x=1, sym_y=1)
    solver = ParticularSolutionsMethodSolver(2.0, 20.0, 20.0; min_dim=20, min_pts=100, min_int_pts=100)
    k_ref = circle_spectrum(1.0, 2.0, 3.0)[1]
    k, t = solve_wavenumber(solver, basis, billiard, 2.45, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BASIS)
    @test t < TENSION_TOL_BASIS
end

# solver: Particular Solutions Method
# basis: Corner Adapted Fourier-Bessel
# billiard: Triangle, right isosceles with legs 1 (make_triangle_and_basis(pi/2, 1.0))
# symmetry: None
# reference: analytic, pi*sqrt(5)
# functions to test: solve_wavenumber
@testset "PSM (CornerAdaptedFourierBessel) - Triangle pi/2 - Ground State" begin
    billiard, basis = QuantumBilliards.make_triangle_and_basis(pi / 2, 1.0)
    solver = ParticularSolutionsMethodSolver(2.0, 20.0, 20.0; min_dim=12, min_pts=64, min_int_pts=64)
    k_ref = pi * sqrt(5)
    k, t = solve_wavenumber(solver, basis, billiard, k_ref, 0.04; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BASIS)
    @test t < TENSION_TOL_BASIS
end

# solver: Particular Solutions Method
# basis: Corner Adapted Fourier-Bessel
# billiard: Veech right triangle (make_veech_right_triangle_and_basis(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762 (full triangle, tension 3e-15, ppw 6 and 10 agree; agrees with DM).
#   The old solvertests.jl value 6.065082959967892 is 8e-6 off and is NOT used.
# functions to test: solve_wavenumber
@testset "PSM (CornerAdaptedFourierBessel) - Veech Triangle - Ground State" begin
    billiard, basis = QuantumBilliards.make_veech_right_triangle_and_basis(5)
    solver = ParticularSolutionsMethodSolver(2.0, 5.0, 2.0)
    k_ref = 6.065090986269762
    k, t = solve_wavenumber(solver, basis, billiard, 6.1, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BASIS)
    @test t < TENSION_TOL_BASIS
end

# solver: Particular Solutions Method
# basis: Real Plane Waves
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), full rectangle 2.0 x 1.4)
# symmetry: even-even
# reference: analytic, pi*sqrt(1/2.0^2 + 1/1.4^2)
# functions to test: solve_wavenumber
@testset "PSM (RealPlaneWaves) - Rectangle, even-even sector - Ground State" begin
    billiard = RectangleBilliard(1.0, 0.7)
    basis = RealPlaneWaves(20; sym_x=1, sym_y=1)
    solver = ParticularSolutionsMethodSolver(2.0, 20.0, 20.0; min_dim=20, min_pts=100, min_int_pts=100)
    k_ref = rectangle_spectrum(2.0, 1.4, 0.0, 4.0)[1]
    k, t = solve_wavenumber(solver, basis, billiard, k_ref + 0.05, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BASIS)
    @test t < TENSION_TOL_BASIS
end
