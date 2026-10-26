# Integration tests for BeynSolver (src/solvers/acceleratedmethods/beyn.jl) through `compute_spectrum`.
# Parameters from the calibration log of scratchpad/test-suite-redesign-plan.md. Window edges are placed away
# from levels so that the level counts are stable. Checks per "Accelerated-solver testset shape": count equal to
# the reference, levels within SPECTRUM_ATOL_BIM, tension below TENSION_TOL_ACCEL, |Im k| below IMAG_TOL, no
# duplicates (for degenerate levels: exactly the analytic multiplicity).

# solver: Beyn contour integral
# kernel/basis: DLP, GlobalCornerGrading, ppw=5 (Taylor backend, the BeynSolver default)
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762 (tension 3e-15, ppw 6 and 10 agree; it is the only level in [5.5, 6.7]).
# functions to test: compute_spectrum
@testset "Beyn (DLP) - Full Triangle - Ground State" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    kernel = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 5.5, 6.7
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    k_ref = 6.065090986269762
    @test length(data.k) == 1
    @test isapprox(real(data.k[1]), k_ref; atol=SPECTRUM_ATOL_BIM)
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic; the window [3.0, 3.5] lies between j_{0,1}=2.4048 and j_{1,1}=3.8317 and holds no level
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Circle - empty window returns an empty spectrum" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    solver = BeynSolver(kernel; use_chebyshev=false, eigenvectors=true)
    @test isempty(circle_spectrum(1.0, 3.0, 3.5))
    data = compute_spectrum(solver, billiard, 3.0, 3.5; multithreaded=true, show_progress=false)
    @test length(data) == 0
    @test isempty(data.k) && isempty(data.ten) && isempty(data.k_im) && isempty(data.control)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, Bessel zeros j_{m,n}; the window [2, 8] holds 12 levels incl. the degenerate pairs
#   j_{1,1}=3.8317, j_{2,1}=5.1356, j_{3,1}=6.3802, j_{1,2}=7.0156, j_{4,1}=7.5883 (multiplicity 2 each)
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Circle - Spectrum incl. degenerate J1 pair" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 2.0, 8.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    ks_ref = circle_spectrum(1.0, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    # multiplicity: j_{1,1} appears exactly twice, j_{0,1} exactly once
    j11 = circle_spectrum(1.0, 3.8, 3.9)[1]
    j01 = circle_spectrum(1.0, 2.3, 2.5)[1]
    @test count(k -> abs(k - j11) < SPECTRUM_ATOL_BIM, ks) == 2
    @test count(k -> abs(k - j01) < SPECTRUM_ATOL_BIM, ks) == 1
    # no spurious clusters: every reference level is matched with its analytic multiplicity
    for kd in unique(round.(ks_ref; digits=8))
        mult = count(k -> abs(k - kd) < 1e-8, ks_ref)
        @test count(k -> abs(k - kd) < SPECTRUM_ATOL_BIM, ks) == mult
    end
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic, pi*sqrt(m^2/2^2 + n^2/1.4^2); the window [3, 10] holds 16 levels
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Rectangle - Spectrum k in [3, 10]" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 3.0, 10.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    ks_ref = rectangle_spectrum(2.0, 1.4, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    @test all(diff(ks) .> SPECTRUM_ATOL_BIM)
end

# solver: Beyn contour integral (composite, one CFIE component per boundary domain)
# kernel/basis: CFIE, SmoothPeriodicGrading, min_pts=64, ppw=10
# billiard: Annular (AnnularBilliard(0.3, 0.0)), concentric, inner radius 0.3, outer radius 1
# symmetry: None
# reference: analytic roots of the annulus Bessel determinant (incl. multiplicity 2 for m > 0); window [3, 8] holds 9 levels
# functions to test: compute_spectrum
@testset "Beyn (CFIE composite) - Annular - Spectrum k in [3, 8]" begin
    billiard = AnnularBilliard(0.3, 0.0)
    component() = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading(), min_pts=64)
    kernel = CompositeBIMSolver(component(), component())
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 3.0, 8.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    ks_ref = annulus_spectrum(0.3, 1.0, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    for kd in unique(round.(ks_ref; digits=8))
        mult = count(k -> abs(k - kd) < 1e-8, ks_ref)
        @test count(k -> abs(k - kd) < SPECTRUM_ATOL_BIM, ks) == mult
    end
end
