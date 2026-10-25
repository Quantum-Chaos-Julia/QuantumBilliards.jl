# Unit tests for
#   src/solvers/chebyshev/dlp.jl   (_dlp_fredholm_full_cheb!, _dlp_kernel_entry_cheb, _dlp_fredholm_reduced_cheb!,
#                                   _dlp_kernel_entry_with_derivatives_cheb, _dlp_fredholm_full_multi_k_cheb!,
#                                   _dlp_fredholm_reduced_multi_k_cheb!)
#   src/solvers/chebyshev/core.jl  (through the Chebyshev radial evaluation used by every matrix entry)
#   src/solvers/chebyshev/optimalpanelization.jl (tune_dlp_cheb_plans, _cheb_geom_rminmax: plan interval taken from the geom cache)
#   src/solvers/chebyshev/bessels.jl (ChebRadialLookupCache on a real boundary geometry)
#   src/solvers/acceleratedmethods/beyn.jl (_construct_matrices_multi_k_cheb, the DLP and CompositeBIMSolver methods; tune_cheb_config)
# Matrix level only: no eigen solve, no search for wavenumbers. The direct matrix construct_matrices(solver, pts, k) is the reference, since the
# Chebyshev path only replaces the radial J1 and H1 evaluations of the Kress-split kernel. The checks are
#   * Chebyshev = direct to the configured interpolation tolerance, on a rectangle with corner grading GlobalCornerGrading(kressq=2) and on the
#     smooth unit circle, for real and complex wavenumbers; the diagonal and the straight-edge entries are k independent and must agree;
#   * a coarse manual plan is visibly less accurate than the tuned one; radii below the plan interval use the direct fallback;
#   * the lookup cache reproduces panel_t on the geometry radii and the plan interval is the padded interval of the geom cache;
#   * the symmetry reduced matrix equals the direct reduced matrix and the orbit-summed full Chebyshev matrix;
#   * the multi-wavenumber variants (used by Beyn) agree with the single-wavenumber ones; threaded = serial;
#   * derivative kernel entries against the direct EBIM derivative entries;
#   * error paths: CompositeBIMSolver, Float32 points, empty wavenumber list, inconsistent buffer counts.
# Helper definitions use the prefix `dch_`.

using LinearAlgebra: norm, diag
using QuantumBilliards: ChebyshevConfig, ChebRadialLookupCache, panel_t, hankel_z_chebyshev_cutoff, boundary_geom_cache, kress_R!, _fold_boundary,
                        _is_nontrivial_dlp_grading, _cheb_geom_rminmax, tune_dlp_cheb_plans, tune_cfie_cheb_plans, tune_cheb_config,
                        _dlp_fredholm_full_cheb!, _dlp_fredholm_reduced_cheb!, _dlp_kernel_entry_cheb, _dlp_kernel_entry,
                        _dlp_kernel_entry_with_derivatives_cheb, _dlp_kernel_entry_with_derivatives,
                        _dlp_fredholm_full_multi_k_cheb!, _dlp_fredholm_reduced_multi_k_cheb!, _construct_matrices_multi_k_cheb

# Small tuned configuration: the tuner starts from these panel counts and degrees and grows them until the sampled error is below tol.
dch_config(; kw...) = ChebyshevConfig(; n_panels_h=50, M_h=8, n_panels_j=20, M_j=8, tol=1e-10, sampling_points=400, max_iter=30, kw...)

# Manual configuration reproducing the parameters a tuner selected: no tuning, identical plans.
dch_manual(cu) = ChebyshevConfig(; n_panels_h=cu.n_panels_h, M_h=cu.M_h, n_panels_j=cu.n_panels_j, M_j=cu.M_j, tol=cu.tol,
                                 sampling_points=cu.sampling_points, param_strategy=:manual)

# Geometry cache and Kress matrix of the nodes, as built by the Chebyshev matrix constructors.
function dch_geometry(pts)
    N = length(pts)
    G = boundary_geom_cache(pts, _is_nontrivial_dlp_grading(pts))
    Rmat = zeros(N, N)
    kress_R!(Rmat)
    return G, Rmat
end

# Tuned DLP plans on the radial interval of the geom cache, and the lookup cache of the first wavenumber.
function dch_plans(G, zj, cfg)
    rmin, rmax = _cheb_geom_rminmax(G, zj)
    plans1, plansj1, cfg_used = tune_dlp_cheb_plans(rmin, rmax, zj, cfg)
    return plans1, plansj1, ChebRadialLookupCache(G, plans1[1], plansj1[1]), cfg_used
end

# Orbit sum of the columns of a full matrix: E[a, b] = sum over the nodes j of orbit b of phase[j] A[fund[a], j].
function dch_orbit_sum(Afull, orbits)
    m = fundamental_size(orbits)
    fund = orbits.fundamental_indices
    E = zeros(ComplexF64, m, m)
    for j in eachindex(orbits.orbit_of), a in 1:m
        E[a, orbits.orbit_of[j]] += orbits.phase[j] * Afull[fund[a], j]
    end
    return E
end

# Edge of the rectangle RectangleBilliard(1.0, 0.6) (x = 1, x = -1, y = 0.6, y = -0.6) carrying a node.
dch_edge(p) = abs(p[1] - 1) < 1e-9 ? 1 : abs(p[1] + 1) < 1e-9 ? 2 : abs(p[2] - 0.6) < 1e-9 ? 3 : 4

@testset "Chebyshev DLP matrix: rectangle with corner grading" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    kc = 2.8 + 0.1im
    zj = ComplexF64[k, kc]
    cfg = dch_config()
    solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=48, grading=GlobalCornerGrading(kressq=2))
    pts = evaluate_points(solver, rect, k)
    N = length(pts)
    G, Rmat = dch_geometry(pts)
    plans1, plansj1, C, cfg_used = dch_plans(G, zj, cfg)
    Ad = construct_matrices(solver, pts, k)
    Adc = construct_matrices(solver, pts, kc)

    @testset "setup is graded and tuned" begin
        @test N == 48
        @test _is_nontrivial_dlp_grading(pts)
        @test length(plans1) == length(plansj1) == 2
        @test cfg_used.tol == cfg.tol
        @test cfg_used.n_panels_h >= cfg.n_panels_h && cfg_used.n_panels_j >= cfg.n_panels_j
    end

    @testset "matrix agrees with the direct matrix, real k" begin
        F = zeros(ComplexF64, N, N)
        @test _dlp_fredholm_full_cheb!(F, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1]) === F
        @test maximum(abs, F - Ad) < 10 * cfg.tol
        @test norm(F - Ad) / norm(Ad) < 10 * cfg.tol
        @test maximum(abs, F - Ad) > 0                                  # an interpolation, not a copy of the direct matrix
    end

    @testset "matrix agrees with the direct matrix, complex k" begin
        F = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(F, pts, Rmat, G, C, kc, plans1[2], plansj1[2])
        @test maximum(abs, F - Adc) < 10 * cfg.tol
        @test maximum(abs, F - Ad) > 1e-3                               # the two wavenumbers really give different matrices
    end

    @testset "diagonal and straight-edge entries do not depend on the interpolation" begin
        F = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(F, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1])
        edge = dch_edge.(pts.xy)
        same = [edge[i] == edge[j] && i != j for i in 1:N, j in 1:N]
        @test isapprox(diag(F), diag(Ad); atol=1e-14, rtol=0)
        @test all(≈(1; atol=1e-12), diag(F))                            # zero curvature of the straight edges
        @test maximum(abs, F[same]) < 1e-12                             # n_j . (x_i - x_j) = 0 along one edge
        @test maximum(abs, F[edge .!= edge']) > 1e-3
    end

    @testset "kernel entries: A = I - D and equal to the direct kernel entry" begin
        F = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(F, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1])
        pairs = [(1, 1), (1, 2), (2, 1), (5, 30), (30, 5), (17, 17), (N, 1), (1, N), (12, 40)]
        for (i, j) in pairs
            e = _dlp_kernel_entry_cheb(pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1], i, j)
            @test F[i, j] ≈ (i == j ? 1 : 0) - e atol = 1e-13
            @test abs(e - _dlp_kernel_entry(pts, Rmat, G, k, i, j)) < 10 * cfg.tol
        end
    end

    @testset "a coarse manual plan is less accurate than the tuned plan" begin
        coarse_cfg = ChebyshevConfig(; n_panels_h=6, M_h=4, n_panels_j=4, M_j=4, param_strategy=:manual)
        rmin, rmax = _cheb_geom_rminmax(G, zj)
        c1, cj1, _ = tune_dlp_cheb_plans(rmin, rmax, zj, coarse_cfg)
        @test c1[1].npanels == 6 && cj1[1].npanels == 4                 # manual: the plans are built exactly as configured
        Cc = ChebRadialLookupCache(G, c1[1], cj1[1])
        F = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(F, pts, Rmat, G, Cc, ComplexF64(k), c1[1], cj1[1])
        coarse_err = maximum(abs, F - Ad)
        Ft = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(Ft, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1])
        @test coarse_err > 1e-6
        @test coarse_err > 1e3 * maximum(abs, Ft - Ad)
    end

    @testset "radii below the plan interval fall back to direct evaluation" begin
        _, rmax = _cheb_geom_rminmax(G, zj)
        high1, highj1, _ = tune_dlp_cheb_plans(0.5, rmax, zj, dch_manual(cfg_used))
        Ch = ChebRadialLookupCache(G, high1[1], highj1[1])
        offdiag = [(i, j) for i in 1:N for j in 1:N if i != j]
        @test any(ij -> Ch.pidx_h[ij...] == 0, offdiag)                 # many graded pairs are closer than 0.5
        @test any(ij -> Ch.pidx_h[ij...] != 0, offdiag)
        F = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(F, pts, Rmat, G, Ch, ComplexF64(k), high1[1], highj1[1])
        @test maximum(abs, F - Ad) < 10 * cfg.tol
    end

    @testset "threaded assembly equals serial assembly" begin
        Ft = zeros(ComplexF64, N, N)
        Fs = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(Ft, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1]; multithreaded=true)
        _dlp_fredholm_full_cheb!(Fs, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1]; multithreaded=false)
        @test Ft == Fs
        Cs = ChebRadialLookupCache(G, plans1[1], plansj1[1]; multithreaded=false)
        offdiag = [(i, j) for i in 1:N for j in 1:N if i != j]
        @test all(ij -> Cs.pidx_h[ij...] == C.pidx_h[ij...] && Cs.t_h[ij...] == C.t_h[ij...] &&
                        Cs.pidx_j[ij...] == C.pidx_j[ij...] && Cs.t_j[ij...] == C.t_j[ij...], offdiag)
    end

    @testset "lookup cache and plan interval come from the geom cache" begin
        rmin, rmax = _cheb_geom_rminmax(G, zj)
        offdiag = [(i, j) for i in 1:N for j in 1:N if i != j]
        @test plans1[1].rmin == rmin && plans1[1].rmax == rmax
        @test plansj1[1].rmin == rmin && plansj1[1].rmax == rmax
        @test plans1[2].rmin == rmin && plansj1[2].rmax == rmax         # all wavenumbers share one panelization
        @test maximum(G.R[i, j] for (i, j) in offdiag) < rmax           # the padded interval contains the largest distance
        @test all(ij -> (C.pidx_h[ij...], C.t_h[ij...]) == panel_t(plans1[1], G.R[ij...]), offdiag)
        @test all(ij -> (C.pidx_j[ij...], C.t_j[ij...]) == panel_t(plansj1[1], G.R[ij...]), offdiag)
        @test all(ij -> (C.pidx_h[ij...] == 0) == (G.R[ij...] < rmin), offdiag)
        @test all(ij -> C.pidx_h[ij...] == C.pidx_h[reverse(ij)...] && C.t_j[ij...] == C.t_j[reverse(ij)...], offdiag)
        @test all(ij -> -1 <= C.t_h[ij...] <= 1, offdiag)
    end

    @testset "derivative kernel entries against the direct EBIM entries" begin
        manual = dch_manual(cfg_used)
        rmin, rmax = _cheb_geom_rminmax(G, ComplexF64[k])
        p0, p1, pj0, pj1, _ = tune_cfie_cheb_plans(rmin, rmax, ComplexF64[k], manual)     # H0, H1, J0, J1 plans
        Cd = ChebRadialLookupCache(G, p1[1], pj1[1])
        for (i, j) in [(1, 1), (1, 2), (2, 1), (5, 30), (30, 5), (12, 40), (N, 3)]
            v, dv, ddv = _dlp_kernel_entry_with_derivatives_cheb(pts, Rmat, G, Cd, ComplexF64(k), p0[1], p1[1], pj0[1], pj1[1], i, j)
            vd, dvd, ddvd = _dlp_kernel_entry_with_derivatives(pts, Rmat, G, k, i, j)
            @test abs(v - vd) < 1e-9
            @test abs(dv - dvd) < 1e-8
            @test abs(ddv - ddvd) < 1e-7
            @test abs(v - _dlp_kernel_entry_cheb(pts, Rmat, G, Cd, ComplexF64(k), p1[1], pj1[1], i, j)) < 1e-12
        end
        v, dv, ddv = _dlp_kernel_entry_with_derivatives_cheb(pts, Rmat, G, Cd, ComplexF64(k), p0[1], p1[1], pj0[1], pj1[1], 7, 7)
        @test dv == 0 && ddv == 0                                       # the diagonal self term does not depend on k
    end

    @testset "multi-wavenumber matrices (Beyn path) equal the single-wavenumber matrices" begin
        Fs = [Matrix{ComplexF64}(undef, N, N) for _ in zj]
        @test _dlp_fredholm_full_multi_k_cheb!(Fs, pts, Rmat, G, C, zj, plans1, plansj1) === Fs
        F1 = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(F1, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1])
        F2 = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(F2, pts, Rmat, G, C, kc, plans1[2], plansj1[2])
        @test isapprox(Fs[1], F1; atol=1e-12, rtol=0)
        @test isapprox(Fs[2], F2; atol=1e-12, rtol=0)

        Fm = _construct_matrices_multi_k_cheb(solver, pts, zj, cfg)
        @test length(Fm) == 2 && all(M -> M isa Matrix{ComplexF64} && size(M) == (N, N), Fm)
        @test maximum(abs, Fm[1] - Ad) < 10 * cfg.tol
        @test maximum(abs, Fm[2] - Adc) < 10 * cfg.tol
        Fm_serial = _construct_matrices_multi_k_cheb(solver, pts, zj, cfg; multithreaded=false)
        @test all(q -> Fm_serial[q] == Fm[q], 1:2)
    end

    @testset "inconsistent buffer counts are rejected" begin
        Fs = [Matrix{ComplexF64}(undef, N, N)]
        @test_throws AssertionError _dlp_fredholm_full_multi_k_cheb!(Fs, pts, Rmat, G, C, zj, plans1, plansj1)
        @test_throws AssertionError _dlp_fredholm_full_multi_k_cheb!([similar(Fs[1]), similar(Fs[1])], pts, Rmat, G, C, zj, plans1[1:1], plansj1)
    end
end

@testset "Chebyshev DLP matrix: smooth boundary (unit circle)" begin
    circle = CircleBilliard(1.0)
    k = 3.0
    kc = 2.8 + 0.1im
    zj = ComplexF64[k, kc]
    cfg = dch_config()
    solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=64)
    pts = evaluate_points(solver, circle, k)
    N = length(pts)
    G, Rmat = dch_geometry(pts)
    plans1, plansj1, C, cfg_used = dch_plans(G, zj, cfg)
    Ad = construct_matrices(solver, pts, k)

    @testset "no grading, full matrix against the direct matrix" begin
        @test N == 64
        @test !_is_nontrivial_dlp_grading(pts)
        F = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(F, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1])
        @test maximum(abs, F - Ad) < 10 * cfg.tol
        Fc = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(Fc, pts, Rmat, G, C, kc, plans1[2], plansj1[2])
        @test maximum(abs, Fc - construct_matrices(solver, pts, kc)) < 10 * cfg.tol
        @test all(≈(1 + 1 / N), diag(F))                                # curvature term w kappa = h / 2 pi
        @test F[2:end, 2:end] ≈ F[1:end-1, 1:end-1] atol = 1e-9         # circulant up to the interpolation error
    end

    @testset "the plan interval covers the nearest and the farthest node pair" begin
        rmin, rmax = _cheb_geom_rminmax(G, zj)
        d = 2 * sin(π / N)                                              # nearest chord of the N equally spaced nodes
        @test rmax ≈ 1.05 * 2                                           # antipodal nodes exist for even N: longest chord is the diameter, padded by 5 %
        @test rmin ≈ max(0.95 * d, hankel_z_chebyshev_cutoff / maximum(abs, zj))   # nearest chord padded by 5 %, bounded below by the cutoff
        @test plans1[1].rmin == rmin && plans1[1].rmax == rmax
        @test cfg_used.tol == cfg.tol
    end

    @testset "multi-wavenumber matrices (Beyn path) against the direct matrices" begin
        Fm = _construct_matrices_multi_k_cheb(solver, pts, zj, cfg)
        @test maximum(abs, Fm[1] - Ad) < 10 * cfg.tol
        @test maximum(abs, Fm[2] - construct_matrices(solver, pts, kc)) < 10 * cfg.tol
        Fmm = _construct_matrices_multi_k_cheb(solver, pts, ComplexF64[k], cfg; multithreaded=false)
        @test length(Fmm) == 1
        @test maximum(abs, Fmm[1] - Ad) < 10 * cfg.tol
    end

    @testset "symmetry reduced matrix, Y reflection (reduced size 32 takes the threaded branch)" begin
        ssym = DoubleLayerPotentialSolver(5.0, circle; min_pts=64, symmetry=YAxisReflection(), sym_characters=(1,))
        ptss = evaluate_points(ssym, circle, k)
        @test ptss.xy == pts.xy                                         # the whole boundary is discretized, so the plans are reusable
        orbits = _fold_boundary(Float64, circle, N, ssym.symmetry, ssym.sym_characters)
        m = fundamental_size(orbits)
        @test m == N ÷ 2 == 32
        Ar = construct_matrices(ssym, ptss, k)
        Fr = zeros(ComplexF64, m, m)
        @test _dlp_fredholm_reduced_cheb!(Fr, ptss, Rmat, G, C, orbits, ComplexF64(k), plans1[1], plansj1[1]) === Fr
        @test maximum(abs, Fr - Ar) < 10 * cfg.tol

        Ff = zeros(ComplexF64, N, N)
        _dlp_fredholm_full_cheb!(Ff, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1])
        @test isapprox(Fr, dch_orbit_sum(Ff, orbits); atol=1e-11, rtol=0)

        Frs = zeros(ComplexF64, m, m)
        _dlp_fredholm_reduced_cheb!(Frs, ptss, Rmat, G, C, orbits, ComplexF64(k), plans1[1], plansj1[1]; multithreaded=false)
        @test Fr == Frs                                                 # threaded = serial

        Fs = [Matrix{ComplexF64}(undef, m, m) for _ in zj]
        @test _dlp_fredholm_reduced_multi_k_cheb!(Fs, ptss, Rmat, G, C, orbits, zj, plans1, plansj1) === Fs
        @test isapprox(Fs[1], Fr; atol=1e-12, rtol=0)
        @test maximum(abs, Fs[2] - construct_matrices(ssym, ptss, 2.8 + 0.1im)) < 10 * cfg.tol

        Fm = _construct_matrices_multi_k_cheb(ssym, ptss, zj, cfg)      # dispatches to the reduced assembly
        @test all(M -> size(M) == (m, m), Fm)
        @test maximum(abs, Fm[1] - Ar) < 10 * cfg.tol
    end
end

@testset "Chebyshev DLP matrix: symmetry reduced rectangle with corner grading" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    cfg = dch_config()
    solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=48, grading=GlobalCornerGrading(kressq=2), symmetry=XYAxisReflection(), sym_characters=(1, 1))
    pts = evaluate_points(solver, rect, k)
    N = length(pts)
    G, Rmat = dch_geometry(pts)
    plans1, plansj1, C, _ = dch_plans(G, ComplexF64[k], cfg)
    orbits = _fold_boundary(Float64, rect, N, solver.symmetry, solver.sym_characters)
    m = fundamental_size(orbits)

    @test _is_nontrivial_dlp_grading(pts)
    @test m == N ÷ 4 == 12
    Fr = zeros(ComplexF64, m, m)
    _dlp_fredholm_reduced_cheb!(Fr, pts, Rmat, G, C, orbits, ComplexF64(k), plans1[1], plansj1[1])
    @test maximum(abs, Fr - construct_matrices(solver, pts, k)) < 10 * cfg.tol
    Ff = zeros(ComplexF64, N, N)
    _dlp_fredholm_full_cheb!(Ff, pts, Rmat, G, C, ComplexF64(k), plans1[1], plansj1[1])
    @test isapprox(Fr, dch_orbit_sum(Ff, orbits); atol=1e-11, rtol=0)
    Fm = _construct_matrices_multi_k_cheb(solver, pts, ComplexF64[k], cfg)
    @test size(Fm[1]) == (m, m) && maximum(abs, Fm[1] - Fr) < 10 * cfg.tol
end

@testset "Chebyshev DLP matrix: error paths" begin
    cfg = dch_config()
    k = 3.0

    @testset "CompositeBIMSolver has no Chebyshev matrix construction" begin
        annulus = AnnularBilliard(0.3, 0.1)
        comp = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), DoubleLayerPotentialSolver(5.0, annulus; min_pts=24))
        pts = evaluate_points(comp, annulus, k)
        @test_throws ErrorException _construct_matrices_multi_k_cheb(comp, pts, ComplexF64[k], cfg)
        @test tune_cheb_config(comp, pts, k, cfg) === cfg                # no plan to tune: the configuration is passed through
        @test size(construct_matrices(comp, pts, k)) == (56, 56)         # the direct composite matrix is still available
    end

    @testset "Float32 points are rejected" begin
        circle = CircleBilliard(1.0)
        solver32 = DoubleLayerPotentialSolver(5f0, circle; min_pts=16)
        pts32 = evaluate_points(solver32, circle, 3f0)
        @test pts32 isa BoundaryPoints{Float32}
        @test_throws ErrorException _construct_matrices_multi_k_cheb(solver32, pts32, ComplexF64[k], cfg)
    end

    @testset "empty wavenumber list and unusable intervals" begin
        @test_throws ArgumentError tune_dlp_cheb_plans(0.1, 2.0, ComplexF64[], cfg)
        circle = CircleBilliard(1.0)
        solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=16)
        pts = evaluate_points(solver, circle, k)
        G, _ = dch_geometry(pts)
        @test_throws ArgumentError _cheb_geom_rminmax(G, ComplexF64[k]; pad=(10.0, 1.0))     # rmin = 10 x nearest chord exceeds rmax
    end
end
