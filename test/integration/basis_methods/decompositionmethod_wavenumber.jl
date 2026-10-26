# Integration tests for DecompositionMethodSolver (src/solvers/sweepmethods/decompositionmethod.jl).
# Sweep solver: `solve_wavenumber` only. One known level per testset, `k0` within `dk` of it,
# window containing exactly one level. Parameters from the calibration log of
# scratchpad/test-suite-redesign-plan.md.

# solver: Decomposition Method
# basis: Real Plane Waves
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), full rectangle 2.0 x 1.4)
# symmetry: even-even
# reference: analytic, pi*sqrt(1/2.0^2 + 1/1.4^2)
# functions to test: solve_wavenumber
@testset "DM (RealPlaneWaves) - Rectangle, even-even sector - Ground State" begin
    billiard = RectangleBilliard(1.0, 0.7)
    basis = RealPlaneWaves(20; sym_x=1, sym_y=1)
    solver = DecompositionMethodSolver(2.0, 20.0; min_dim=20, min_pts=100)
    k_ref = rectangle_spectrum(2.0, 1.4, 0.0, 4.0)[1]
    k, t = solve_wavenumber(solver, basis, billiard, k_ref + 0.05, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BASIS)
    @test t < TENSION_TOL_BASIS
end

# solver: Decomposition Method
# basis: Corner Adapted Fourier-Bessel
# billiard: Triangle, right isosceles with legs 1 (make_triangle_and_basis(pi/2, 1.0))
# symmetry: None
# reference: analytic, pi*sqrt(5)
# functions to test: solve_wavenumber
@testset "DM (CornerAdaptedFourierBessel) - Triangle pi/2 - Ground State" begin
    billiard, basis = QuantumBilliards.make_triangle_and_basis(pi / 2, 1.0)
    solver = DecompositionMethodSolver(2.0, 20.0; min_dim=12, min_pts=64)
    k_ref = pi * sqrt(5)
    k, t = solve_wavenumber(solver, basis, billiard, k_ref, 0.04; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BASIS)
    @test t < TENSION_TOL_BASIS
end

# solver: Decomposition Method
# basis: Fourier-Bessel
# billiard: Circle, radius 1
# symmetry: even-even
# reference: analytic, Bessel zero j_{0,1}
# functions to test: solve_wavenumber
@testset "DM (FourierBessel) - Circle, even-even sector - First J0 State" begin
    billiard = CircleBilliard(1.0)
    basis = FourierBessel(20; sym_x=1, sym_y=1)
    solver = DecompositionMethodSolver(2.0, 20.0; min_dim=20, min_pts=100)
    k_ref = circle_spectrum(1.0, 2.0, 3.0)[1]
    k, t = solve_wavenumber(solver, basis, billiard, 2.45, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BASIS)
    @test t < TENSION_TOL_BASIS
end
