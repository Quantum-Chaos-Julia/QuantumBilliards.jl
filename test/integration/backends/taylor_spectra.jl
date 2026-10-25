# Integration tests for the Taylor matrix backend of BeynSolver (src/solvers/acceleratedmethods/beyn.jl) through
# `compute_spectrum`. The Taylor backend (`use_taylor=true`) is compared level by level with the direct backend
# (`use_taylor=false, use_chebyshev=false`) and both are checked against the analytic reference.

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10; Taylor backend vs direct backend
# billiard: Circle, radius 1
# symmetry: None
# reference: level-by-level agreement of the two backends, each also checked against analytic Bessel zeros;
#   the window [2, 8] holds 12 levels incl. the degenerate pairs j_{1,1}, j_{2,1}, j_{3,1}, j_{1,2}, j_{4,1}
# functions to test: compute_spectrum
@testset "Beyn (CFIE), Taylor vs direct - Circle - Spectrum k in [2, 8]" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    solver_taylor = BeynSolver(kernel; use_taylor=true, use_chebyshev=false)
    solver_direct = BeynSolver(kernel; use_taylor=false, use_chebyshev=false)
    k1, k2 = 2.0, 8.0
    data_taylor = compute_spectrum(solver_taylor, billiard, k1, k2; multithreaded=false, show_progress=false)
    data_direct = compute_spectrum(solver_direct, billiard, k1, k2; multithreaded=false, show_progress=false)
    ks_ref = circle_spectrum(1.0, k1, k2)
    ks_taylor = sort(real.(data_taylor.k))
    ks_direct = sort(real.(data_direct.k))
    @test length(ks_ref) == 12
    @test length(ks_taylor) == length(ks_ref)
    @test length(ks_direct) == length(ks_ref)
    # level-by-level agreement of the two backends
    @test all(isapprox.(ks_taylor, ks_direct; atol=SPECTRUM_ATOL_BIM))
    # each backend against the analytic spectrum
    @test all(isapprox.(ks_taylor, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(isapprox.(ks_direct, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data_taylor.ten .< TENSION_TOL_ACCEL)
    @test all(data_direct.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data_taylor.k_im) .< IMAG_TOL)
    @test all(abs.(data_direct.k_im) .< IMAG_TOL)
    # degenerate pair j_{1,1} appears exactly twice in each backend
    j11 = circle_spectrum(1.0, 3.8, 3.9)[1]
    @test count(k -> abs(k - j11) < SPECTRUM_ATOL_BIM, ks_taylor) == 2
    @test count(k -> abs(k - j11) < SPECTRUM_ATOL_BIM, ks_direct) == 2
end
