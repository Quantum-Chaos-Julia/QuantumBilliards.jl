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
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
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
# note: taylor_tol = 1e-10 instead of the default 3e-10. With the default the tuned degree is p = 13 (validation error
#   about 1.5e-10), and with the OpenBLAS backend the Ritz residual of the level k = 5.2194 in the window [4.2, 5.4] stagnates
#   above res_tol, so adaptive_cork reaches the full Krylov dimension and throws. p = 14 (error about 1.5e-11) converges.
#   The cause of the residual stagnation is not yet identified.
# functions to test: compute_spectrum
@testset "CORK (CFIE) - Rectangle - Spectrum k in [3, 10]" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(20.0, billiard; grading=GlobalCornerGrading(), min_pts=800)
    solver = CORKSolver(kernel; nlevels=40, taylor_tol=1e-10)
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
    ks, _ = @test_logs (:warn, r"Weyl law predicts") solve_spectrum(solver, billiard, 6.5, 7.0; multithreaded=true)
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
    ks, _ = @test_logs solve_spectrum(solver, billiard, 6.0, 1.2; multithreaded=true)
    ks_ref = rectangle_spectrum(2.0, 1.4, 5.4, 6.6)
    ks = sort(real.(ks))
    @test !isempty(ks_ref)
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
end

# Boundary functions stored in the spectrum (`states = :boundary`, Step 11 of scratchpad/boundary-function-spectrum-plan.md): compute_spectrum runs the
# usual right Krylov solve for the roots and takes the left null vectors from the Chebyshev polynomial (cork_left_vectors, validated on a sample and
# recomputed on the kernel matrix on failure); the stored u is the Rellich-normalized y ./ ds (no layer density). References: solve_state u at the same
# level on the stored boundary points (compared after the least-squares complex scaling, tolerance BOUNDARY_U_TOL_BIM; worst observed 6.5e-12) and the
# analytic levels. Window [4.2, 6.0] of the rectangle holds the levels 4.7544, 5.2194, 5.4783.
# A DLP kernel (same settings) was tried and not kept: the right Krylov solve for the roots does not converge on it (full Krylov dimension reached),
# independently of the stored states.

# solver: CORK, states = :boundary
# kernel/basis: CFIE, GlobalCornerGrading, ppw=20, min_pts=800, nlevels=20, taylor_tol=1e-10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: solve_state u on the stored pts; analytic levels (3 levels in [4.2, 6.0])
# functions to test: compute_spectrum (CORKSolver, states = :boundary), _cork_left_vectors, cork_left_vectors, _boundary_states
@testset "CORK (CFIE) - Rectangle - states = :boundary stores u equal to solve_state, window [4.2, 6.0]" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(20.0, billiard; grading=GlobalCornerGrading(), min_pts=800)
    solver = CORKSolver(kernel; nlevels=20, taylor_tol=1e-10, states=:boundary)
    data = compute_spectrum(solver, billiard, 4.2, 6.0; multithreaded=true, show_progress=false)
    ks_ref = rectangle_spectrum(2.0, 1.4, 4.2, 6.0)
    ks = sort(real.(data.k))
    @test length(ks_ref) == 3
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    @test boundary_states_wellformed(data)
    res, idx = stored_u_residuals(data, kernel, billiard)
    @test length(idx) == 3
    @test all(res .< BOUNDARY_U_TOL_BIM)
end

# solver: CORK, states = :boundary
# kernel/basis: composite CFIE (one SmoothPeriodicGrading component per boundary domain), min_pts=64, ppw=10, nlevels=20
# billiard: Annular (AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])), inner radius 0.3, outer radius 1
# symmetry: None
# reference: analytic annulus levels (4.4124 simple, 4.7058 x2, 5.4702 x2 in [4.2, 5.8]); solve_state u on the stored pts for the simple level
#   (degenerate levels are skipped by stored_u_residuals: one arbitrary vector of the eigenspace)
# functions to test: compute_spectrum (CORKSolver, states = :boundary with a composite kernel), _cork_left_vectors, _boundary_states
@testset "CORK (CFIE composite) - Annular - states = :boundary stores u equal to solve_state, window [4.2, 5.8]" begin
    billiard = AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])
    component() = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading(), min_pts=64)
    kernel = CompositeBIMSolver(component(), component())
    solver = CORKSolver(kernel; nlevels=20, states=:boundary)
    data = compute_spectrum(solver, billiard, 4.2, 5.8; multithreaded=true, show_progress=false)
    ks_ref = annulus_spectrum(0.3, 1.0, 4.2, 5.8)
    ks = sort(real.(data.k))
    @test length(ks_ref) == 5
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    @test boundary_states_wellformed(data)
    res, idx = stored_u_residuals(data, kernel, billiard)
    @test length(idx) == 1
    @test all(res .< BOUNDARY_U_TOL_BIM)
end
