# Integration tests for CORKSolver (src/solvers/acceleratedmethods/cork.jl) through `compute_spectrum` and `solve_spectrum`.
# CORK needs `nlevels` of the order of (or above) the number of levels in the window, and enough boundary points
# (finding F2 of scratchpad/test-suite-redesign-plan.md): with the defaults it throws, and too small a boundary
# discretization raises "Reached full CORK Krylov dimension". With nlevels = 10, 20 and 40 on the rectangle [3, 10]
# (ppw 20, min_pts 800) `compute_spectrum` was observed to return all 16 levels, so silent drops through
# `compute_spectrum` are no longer reproducible. The direct `solve_spectrum` path uses a fixed polynomial degree over the
# whole window and drops levels in a wide window; it now warns when the count falls well below the Weyl prediction.
# The testsets use the calibrated settings and assert the level count to catch regressions.

# solver: CORK
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=20, min_pts=800, nlevels=40
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, Bessel zeros; the window [3, 8] holds 11 levels incl. the degenerate pairs
#   j_{1,1}=3.8317, j_{2,1}=5.1356, j_{3,1}=6.3802, j_{1,2}=7.0156, j_{4,1}=7.5883 (multiplicity 2 each) and j_{0,2}=5.5201
# functions to test: compute_spectrum
@testset "CORK (CFIE) - Circle - Spectrum incl. degenerate pair" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(20.0, billiard; grading=SmoothPeriodicGrading(), min_pts=800)
    solver = CORKSolver(kernel; nlevels=40)
    k1, k2 = 3.0, 8.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    ks_ref = circle_spectrum(1.0, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    j11 = circle_spectrum(1.0, 3.8, 3.9)[1]
    @test count(k -> abs(k - j11) < SPECTRUM_ATOL_BIM, ks) == 2
    for kd in unique(round.(ks_ref; digits=8))
        mult = count(k -> abs(k - kd) < 1e-8, ks_ref)
        @test count(k -> abs(k - kd) < SPECTRUM_ATOL_BIM, ks) == mult
    end
end

# solver: CORK
# kernel/basis: CFIE, GlobalCornerGrading, ppw=20, min_pts=800, nlevels=40
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; the window [3, 10] holds 16 levels
# functions to test: compute_spectrum
@testset "CORK (CFIE) - Rectangle - Spectrum k in [3, 10]" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(20.0, billiard; grading=GlobalCornerGrading(), min_pts=800)
    solver = CORKSolver(kernel; nlevels=40)
    k1, k2 = 3.0, 10.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    ks_ref = rectangle_spectrum(2.0, 1.4, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    @test all(diff(ks) .> SPECTRUM_ATOL_BIM)
end

# solver: CORK
# kernel/basis: CFIE, GlobalCornerGrading, ppw=20, min_pts=800, nlevels=20, fixed polynomial degree p = 12
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; the window [3, 10] holds 16 levels (Weyl count about 16.5). The direct call with window width 7 uses
#   one polynomial of fixed degree and was observed to return only 9 of them, which must raise the completeness warning.
# functions to test: solve_spectrum, _cork_warn_if_incomplete
@testset "CORK (CFIE) - Rectangle - wide direct solve_spectrum warns about an incomplete spectrum" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(20.0, billiard; grading=GlobalCornerGrading(), min_pts=800)
    solver = CORKSolver(kernel; nlevels=20)
    ks, _ = @test_logs (:warn, r"Weyl law predicts") solve_spectrum(solver, billiard, 6.5, 7.0; multithreaded=false)
    ks_ref = rectangle_spectrum(2.0, 1.4, 3.0, 10.0)
    @test length(ks_ref) == 16
    @test length(ks) < length(ks_ref)
end

# solver: CORK
# kernel/basis: CFIE, GlobalCornerGrading, ppw=20, min_pts=800, nlevels=20
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; a narrow window [5.4, 6.6] is resolved by the fixed polynomial degree, so the direct call returns all
#   reference levels and the completeness check stays silent
# functions to test: solve_spectrum, _cork_warn_if_incomplete
@testset "CORK (CFIE) - Rectangle - narrow direct solve_spectrum returns the reference levels without a warning" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(20.0, billiard; grading=GlobalCornerGrading(), min_pts=800)
    solver = CORKSolver(kernel; nlevels=20)
    ks, _ = @test_logs solve_spectrum(solver, billiard, 6.0, 1.2; multithreaded=false)
    ks_ref = rectangle_spectrum(2.0, 1.4, 5.4, 6.6)
    ks = sort(real.(ks))
    @test !isempty(ks_ref)
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
end
