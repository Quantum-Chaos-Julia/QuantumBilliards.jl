# Integration tests for the `compute_spectrum` drivers (src/spectra/spectralutils.jl): the wavenumber-range method
# `compute_spectrum(solver, billiard, k1, k2)` (accelerated basis solvers additionally take the spacing `dk`) and the
# Weyl state-index method `compute_spectrum(solver, billiard, N1, N2)`, which only converts `(N1, N2)` through
# `k_range_for_states(billiard, N1, N2; fundamental)` (`fundamental = true` for symmetry-reduced kernels and for basis
# solvers, `false` for the unreduced BIM kernels used here) and calls the k-range method.
# Window edges are checked to lie away from levels, so level counts are stable. Rectangle references are analytic.
# RectangleBilliard(1.0, 0.7) is the full 2.0 x 1.4 rectangle (area 2.8; fundamental domain: first quadrant, area 0.7).
# The N-range window (3, 8) of the unreduced rectangle is [4.9313, 7.2356] and holds 5 levels, nearest level to an
# edge 0.18 away.
#
# Findings reflected here: VS needs a spectral window width dk well below the spacing-controlled validity of the scaling
# method (the N-range default N_expect = 3 gives dk about 1.3 at k = 21 and returns spurious levels with tension above 1,
# so the VS testset passes N_expect = 0.35); EBIM returns spurious levels and edge duplicates (finding F1), so no counts;
# CORK needs nlevels = 40, ppw = 20, min_pts = 800 (finding F2).

# Level-by-level check against the analytic Rectangle levels: count and positions.
_drv_rectangle_levels(k1, k2) = rectangle_spectrum(2.0, 1.4, k1, k2)

# solver: Vergini-Saraceno
# kernel/basis: Real Plane Waves, even-even sector (sym_x = 1, sym_y = 1), scaling factors (5.0, 10.0)
# billiard: Rectangle (RectangleBilliard(1.0, 0.7)), fundamental domain
# symmetry: even-even
# reference: analytic, rectangle_spectrum(2.0, 1.4, ...; parity_x=1, parity_y=1); window [20, 23] holds 8 levels
#   (dk = 0.2), Weyl states (20, 26) give [20.083, 22.749] with 7 levels (N_expect = 0.35 gives dk about 0.2)
# functions to test: compute_spectrum (k-range with dk, N-range)
@testset "VS (RealPlaneWaves) - Rectangle - compute_spectrum k-range vs N-range" begin
    billiard = RectangleBilliard(1.0, 0.7)
    basis = RealPlaneWaves(20; sym_x=1, sym_y=1)
    solver = VerginiSaracenoSolver(5.0, 10.0; eigenvectors=false)
    k1, k2 = 20.0, 23.0
    data_k = compute_spectrum(solver, basis, billiard, k1, k2, 0.2; multithreaded=false, show_progress=false)
    ref_k = rectangle_spectrum(2.0, 1.4, k1, k2; parity_x=1, parity_y=1)
    ks_k = sort(real.(data_k.k))
    @test length(ref_k) == 8
    @test length(ks_k) == length(ref_k)
    @test all(isapprox.(ks_k, ref_k; atol=SPECTRUM_ATOL_VS))
    @test all(data_k.ten .< TENSION_TOL_VS)
    N1, N2 = 20, 26
    kN1, kN2 = k_range_for_states(billiard, N1, N2)
    data_N = compute_spectrum(solver, basis, billiard, N1, N2; N_expect=0.35, multithreaded=false, show_progress=false)
    ref_N = rectangle_spectrum(2.0, 1.4, kN1, kN2; parity_x=1, parity_y=1)
    ks_N = sort(real.(data_N.k))
    @test all(k -> kN1 <= k <= kN2, ks_N)
    @test length(ref_N) == 7
    @test length(ks_N) == length(ref_N)
    @test all(isapprox.(ks_N, ref_N; atol=SPECTRUM_ATOL_VS))
    @test all(data_N.ten .< TENSION_TOL_VS)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; Weyl states (3, 8) give [4.9313, 7.2356] with 5 levels
# functions to test: compute_spectrum (k-range, N-range)
@testset "Beyn (CFIE) - Rectangle - compute_spectrum k-range vs N-range" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    solver = BeynSolver(kernel; use_chebyshev=false, eigenvectors=false)
    k1, k2 = k_range_for_states(billiard, 3, 8; fundamental=false)
    ref = _drv_rectangle_levels(k1, k2)
    data_k = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    data_N = compute_spectrum(solver, billiard, 3, 8; multithreaded=false, show_progress=false)
    @test length(ref) == 5
    for data in (data_k, data_N)
        ks = sort(real.(data.k))
        @test length(ks) == length(ref)
        @test all(isapprox.(ks, ref; atol=SPECTRUM_ATOL_BIM))
        @test all(data.ten .< TENSION_TOL_ACCEL)
        @test all(abs.(data.k_im) .< IMAG_TOL)
    end
    @test sort(real.(data_N.k)) ≈ sort(real.(data_k.k)) atol = SPECTRUM_ATOL_BIM
end

# solver: CORK
# kernel/basis: CFIE, GlobalCornerGrading, ppw=20, min_pts=800, nlevels=40
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; Weyl states (3, 8) give [4.9313, 7.2356] with 5 levels
# functions to test: compute_spectrum (k-range, N-range)
@testset "CORK (CFIE) - Rectangle - compute_spectrum k-range vs N-range" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(20.0, billiard; grading=GlobalCornerGrading(), min_pts=800)
    solver = CORKSolver(kernel; nlevels=40)
    k1, k2 = k_range_for_states(billiard, 3, 8; fundamental=false)
    ref = _drv_rectangle_levels(k1, k2)
    data_k = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    data_N = compute_spectrum(solver, billiard, 3, 8; multithreaded=false, show_progress=false)
    @test length(ref) == 5
    for data in (data_k, data_N)
        ks = sort(real.(data.k))
        @test length(ks) == length(ref)
        @test all(isapprox.(ks, ref; atol=SPECTRUM_ATOL_BIM))
        @test all(data.ten .< TENSION_TOL_ACCEL)
        @test all(abs.(data.k_im) .< IMAG_TOL)
    end
    @test sort(real.(data_N.k)) ≈ sort(real.(data_k.k)) atol = SPECTRUM_ATOL_BIM
end

# solver: Expanded BIM
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; Weyl states (3, 8) give [4.9313, 7.2356] with 5 levels. EBIM returns spurious levels and
#   duplicates (finding F1), so only (1) every reference level is matched by a confident level (tension < TENSION_TOL_EBIM)
#   and (2) every confident level matches a reference level, are asserted, for both the k-range and the N-range method.
# functions to test: compute_spectrum (k-range, N-range)
@testset "EBIM (CFIE) - Rectangle - compute_spectrum k-range vs N-range" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    solver = ExpandedBIMSolver(kernel)
    k1, k2 = k_range_for_states(billiard, 3, 8; fundamental=false)
    ref = _drv_rectangle_levels(k1, k2)
    data_k = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    data_N = compute_spectrum(solver, billiard, 3, 8; multithreaded=false, show_progress=false)
    @test length(ref) == 5
    for data in (data_k, data_N)
        ks = real.(data.k)
        confident = findall(<(TENSION_TOL_EBIM), data.ten)
        @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ref)
        @test all(i -> minimum(abs.(ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
    end
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10, m = 3 states per window
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; window [3, 10] holds 16 levels and is split by `plan_weyl_windows` into 8 adjacent windows of at
#   most m = 3 Weyl states (`weyl_window_count`), so levels near window boundaries must be neither lost nor duplicated
# functions to test: compute_spectrum (Weyl-window merge), plan_weyl_windows, weyl_window_count
@testset "Beyn (CFIE) - Rectangle - window merge over several Weyl windows" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    solver = BeynSolver(kernel; m=3, use_chebyshev=false, eigenvectors=false)
    k1, k2 = 3.0, 10.0
    windows = QuantumBilliards.plan_weyl_windows(billiard, k1, k2; m=solver.m, Rmax=solver.Rmax, fundamental=false)
    counts = [QuantumBilliards.weyl_window_count(billiard, kl, kr - kl; fundamental=false) for (kl, kr) in windows]
    @test length(windows) > 1
    @test windows[1][1] == k1 && windows[end][2] == k2
    @test all(i -> windows[i][2] == windows[i+1][1], 1:length(windows)-1)
    @test all(counts .<= solver.m + 1e-9)
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    ref = rectangle_spectrum(2.0, 1.4, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ref)
    @test all(isapprox.(ks, ref; atol=SPECTRUM_ATOL_BIM))
    @test all(diff(ks) .> SPECTRUM_ATOL_BIM)
    @test all(data.ten .< TENSION_TOL_ACCEL)
    # the test is meaningful only if some level lies within 0.1 of an interior window boundary
    @test any(kw -> any(abs.(ref .- kw) .< 0.1), [w[2] for w in windows[1:end-1]])
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=8, cheap Beyn settings
# billiard: Stadium (StadiumBilliard(0.5)), smooth up to the junctions (no corner correction)
# symmetry: None
# reference: Weyl law N(k) = (A k^2 - L k) / (4 pi) with A the area and L the boundary length; the level count in
#   [3, 9] is compared with N(9) - N(3) = 25.5, observed 24 levels. The fluctuation of the counting function of a
#   chaotic billiard is of order one level, so the margin is 3 levels.
# functions to test: compute_spectrum, weyl_law
@testset "Beyn (CFIE) - Stadium - level count vs weyl_law" begin
    billiard = StadiumBilliard(0.5)
    kernel = CombinedFieldIntegralEquationSolver(8.0, billiard; grading=GlobalCornerGrading(kressq=2))
    solver = BeynSolver(kernel; m=6, nq=16, r=16, Rmax=0.5, svd_tol=1e-10, res_tol=2e-5,
        taylor_degree=10, use_chebyshev=false, imag_k_check=false, eigenvectors=false)
    k1, k2 = 3.0, 9.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    A = BilliardGeometry.area(billiard)
    L = sum(crv.length for crv in full_boundary(billiard))
    n_weyl = QuantumBilliards.weyl_law(k2, A, L) - QuantumBilliards.weyl_law(k1, A, L)
    @test abs(length(data.k) - n_weyl) < 3
    @test all(data.ten .< TENSION_TOL_BIM)
    @test all(diff(sort(real.(data.k))) .> SPECTRUM_ATOL_BASIS)
end
