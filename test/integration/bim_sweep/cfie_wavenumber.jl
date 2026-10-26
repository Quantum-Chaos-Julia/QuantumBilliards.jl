# Integration tests for CombinedFieldIntegralEquationSolver (src/solvers/sweepmethods/cfie.jl).
# Sweep solver: `solve_wavenumber` only. Parameters from the calibration log of
# scratchpad/test-suite-redesign-plan.md. Each window `k0 +- dk` contains exactly one level.

# Cross-method references computed inside a testset (Beyn on the same kernel) are cached here.
const CFIE_SWEEP_REFERENCES = Dict{Symbol,Float64}()

# solver: Combined Field Integral Equation
# kernel/basis: CFIE, SmoothPeriodicGrading
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, Bessel zero j_{0,1}/R
# functions to test: solve_wavenumber
@testset "CFIE - Circle - Ground State" begin
    billiard = CircleBilliard(1.0)
    solver = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    k_ref = circle_spectrum(1.0, 2.0, 3.0)[1]
    k, t = solve_wavenumber(solver, billiard, 2.4, 0.2; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end

# solver: Combined Field Integral Equation
# kernel/basis: CFIE, SmoothPeriodicGrading
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, Bessel zero j_{0,2}/R (the neighbouring levels 5.1356 and 6.3802 lie outside the window)
# functions to test: solve_wavenumber
@testset "CFIE - Circle - Second J0 State" begin
    billiard = CircleBilliard(1.0)
    solver = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    k_ref = circle_spectrum(1.0, 5.4, 5.6)[1]
    k, t = solve_wavenumber(solver, billiard, 5.52, 0.2; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end

# solver: Combined Field Integral Equation
# kernel/basis: CFIE, GlobalCornerGrading
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762 (tension 3e-15, ppw 6 and 10 agree; it is the only level in [5.5, 6.7]).
#   The old solvertests.jl values 6.0650909 / 6.0650910 are NOT used.
# functions to test: solve_wavenumber
@testset "CFIE - Full Triangle - Ground State" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    solver = CombinedFieldIntegralEquationSolver(5.0, billiard; grading=GlobalCornerGrading())
    k_ref = 6.065090986269762
    k, t = solve_wavenumber(solver, billiard, 6.1, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end

# solver: Combined Field Integral Equation
# kernel/basis: CFIE, GlobalCornerGrading(kressq=8), min_pts=400 (ppw is irrelevant at this k; the level is
#   discretization limited and kressq=2 converges slowly)
# billiard: Mushroom (MushroomBilliard(0.5, 0.7)), full boundary
# symmetry: None
# reference: cross-method, Beyn (Taylor backend) on the same kernel, window [3.0, 3.3] containing the ground state only
#   (computed in the testset, cached in CFIE_SWEEP_REFERENCES). Observed ground state about 3.14603.
# functions to test: solve_wavenumber, compute_spectrum
@testset "CFIE - Mushroom - Ground State" begin
    billiard = MushroomBilliard(0.5, 0.7)
    solver = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading(kressq=8), min_pts=400)
    k_ref = get!(CFIE_SWEEP_REFERENCES, :mushroom_ground) do
        k1, k2 = 3.0, 3.3
        beyn = BeynSolver(solver; m=6, nq=24, r=12, Rmax=(k2 - k1) / 2 + 0.02, svd_tol=1e-10, res_tol=2e-5,
            taylor_degree=6, use_chebyshev=false, imag_k_check=false, eigenvectors=false)
        data = compute_spectrum(beyn, billiard, k1, k2; multithreaded=true, show_progress=false)
        @test length(data.k) == 1
        @test all(data.ten .< HIGH_K_TENSION_TOL)
        real(data.k[1])
    end
    k, t = solve_wavenumber(solver, billiard, 3.15, 0.1; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end
