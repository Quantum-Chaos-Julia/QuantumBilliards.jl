# Unit tests for
#   src/solvers/chebyshev/cfie.jl  (_cfie_fredholm_full_cheb!, _cfie_kernel_entry_cheb, _cfie_fredholm_reduced_cheb!,
#                                   _cfie_kernel_entry_with_derivatives_cheb, _cfie_fredholm_full_multi_k_cheb!,
#                                   _cfie_fredholm_reduced_multi_k_cheb!)
#   src/solvers/chebyshev/optimalpanelization.jl (tune_cfie_cheb_plans: H0, H1, J0, J1 plans on the geom cache radial interval)
#   src/solvers/acceleratedmethods/beyn.jl (_construct_matrices_multi_k_cheb for the CFIE and CompositeBIMSolver kernels; tune_cheb_config)
#   src/solvers/acceleratedmethods/chebyshevconfig.jl (the note that CompositeBIMSolver has no Chebyshev path)
# Matrix level only: no eigen solve, no search for wavenumbers. The direct matrix construct_matrices(solver, pts, k) of the combined field
# operator A = I - (D + i k S) is the reference; the Chebyshev path only replaces the radial J0, J1, H0 and H1 evaluations of the Kress split
# kernels. The checks are
#   * Chebyshev = direct to the configured interpolation tolerance, on a rectangle with corner grading GlobalCornerGrading(kressq=2) and on the
#     smooth unit circle, for real and complex wavenumbers; kernel entries satisfy A = I - K and equal the direct kernel entries;
#   * the diagonal (Kress limit with log k and the Euler constant) does not involve the interpolation;
#   * a coarse manual plan is visibly less accurate; radii below the plan interval fall back to direct evaluation;
#   * the symmetry reduced matrix equals the direct reduced matrix and the orbit-summed full Chebyshev matrix;
#   * the multi-wavenumber variants (Beyn path) agree with the single-wavenumber ones; threaded = serial;
#   * derivative kernel entries against the direct EBIM derivative entries;
#   * the annulus (CompositeBIMSolver) has no Chebyshev matrix construction; Float32 points are rejected.
# Helper definitions use the prefix `cch_`.

using LinearAlgebra: norm, diag
using QuantumBilliards: ChebyshevConfig, ChebRadialLookupCache, boundary_geom_cache, kress_R!, _fold_boundary,
                        _is_nontrivial_dlp_grading, _cheb_geom_rminmax, tune_cfie_cheb_plans, tune_cheb_config,
                        _cfie_fredholm_full_cheb!, _cfie_fredholm_reduced_cheb!, _cfie_kernel_entry_cheb, _cfie_kernel_entry,
                        _cfie_kernel_entry_with_derivatives_cheb, _cfie_kernel_entry_with_derivatives,
                        _cfie_fredholm_full_multi_k_cheb!, _cfie_fredholm_reduced_multi_k_cheb!, _construct_matrices_multi_k_cheb

# Small tuned configuration: the tuner starts from these panel counts and degrees and grows them until the sampled error is below tol.
cch_config(; kw...) = ChebyshevConfig(; n_panels_h=50, M_h=8, n_panels_j=20, M_j=8, tol=1e-10, sampling_points=400, max_iter=30, kw...)

# Manual configuration reproducing the parameters a tuner selected: no tuning, identical plans.
cch_manual(cu) = ChebyshevConfig(; n_panels_h=cu.n_panels_h, M_h=cu.M_h, n_panels_j=cu.n_panels_j, M_j=cu.M_j, tol=cu.tol,
                                 sampling_points=cu.sampling_points, param_strategy=:manual)

# Geometry cache and Kress matrix of the nodes, as built by the Chebyshev matrix constructors.
function cch_geometry(pts)
    N = length(pts)
    G = boundary_geom_cache(pts, _is_nontrivial_dlp_grading(pts))
    Rmat = zeros(N, N)
    kress_R!(Rmat)
    return G, Rmat
end

# Tuned CFIE plans on the radial interval of the geom cache, and the lookup cache of the first wavenumber.
function cch_plans(G, zj, cfg)
    rmin, rmax = _cheb_geom_rminmax(G, zj)
    plans0, plans1, plansj0, plansj1, cfg_used = tune_cfie_cheb_plans(rmin, rmax, zj, cfg)
    return plans0, plans1, plansj0, plansj1, ChebRadialLookupCache(G, plans1[1], plansj1[1]), cfg_used
end

# Orbit sum of the columns of a full matrix: E[a, b] = sum over the nodes j of orbit b of phase[j] A[fund[a], j].
function cch_orbit_sum(Afull, orbits)
    m = fundamental_size(orbits)
    fund = orbits.fundamental_indices
    E = zeros(ComplexF64, m, m)
    for j in eachindex(orbits.orbit_of), a in 1:m
        E[a, orbits.orbit_of[j]] += orbits.phase[j] * Afull[fund[a], j]
    end
    return E
end

@testset "Chebyshev CFIE matrix: rectangle with corner grading" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    kc = 2.8 + 0.1im
    zj = ComplexF64[k, kc]
    cfg = cch_config()
    solver = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=48, grading=GlobalCornerGrading(kressq=2))
    pts = evaluate_points(solver, rect, k)
    N = length(pts)
    G, Rmat = cch_geometry(pts)
    plans0, plans1, plansj0, plansj1, C, cfg_used = cch_plans(G, zj, cfg)
    Ad = construct_matrices(solver, pts, k)
    Adc = construct_matrices(solver, pts, kc)

    @testset "setup is graded and tuned" begin
        @test N == 48
        @test _is_nontrivial_dlp_grading(pts)
        @test length(plans0) == length(plans1) == length(plansj0) == length(plansj1) == 2
        @test plans0[1].ν == 0 && plans1[1].ν == 1 && plansj0[1].ν == 0 && plansj1[1].ν == 1
        @test cfg_used.tol == cfg.tol
    end

    @testset "matrix agrees with the direct matrix, real k" begin
        F = zeros(ComplexF64, N, N)
        @test _cfie_fredholm_full_cheb!(F, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1]) === F
        @test maximum(abs, F - Ad) < 10 * cfg.tol
        @test norm(F - Ad) / norm(Ad) < 10 * cfg.tol
        @test maximum(abs, F - Ad) > 0                                  # an interpolation, not a copy of the direct matrix
    end

    @testset "matrix agrees with the direct matrix, complex k" begin
        F = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(F, pts, Rmat, G, C, kc, plans0[2], plans1[2], plansj0[2], plansj1[2])
        @test maximum(abs, F - Adc) < 10 * cfg.tol
        @test maximum(abs, F - Ad) > 1e-3
    end

    @testset "diagonal does not involve the interpolation" begin
        F = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(F, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1])
        @test isapprox(diag(F), diag(Ad); atol=1e-13, rtol=0)
        @test maximum(abs.(diag(F) .- 1)) > 1e-2                        # unlike the DLP of a straight edge (diagonal 1) the single layer self term contributes
    end

    @testset "kernel entries: A = I - K and equal to the direct kernel entry" begin
        F = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(F, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1])
        for (i, j) in [(1, 1), (1, 2), (2, 1), (5, 30), (30, 5), (17, 17), (N, 1), (1, N), (12, 40)]
            e = _cfie_kernel_entry_cheb(pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1], i, j)
            @test F[i, j] ≈ (i == j ? 1 : 0) - e atol = 1e-13
            @test abs(e - _cfie_kernel_entry(pts, Rmat, G, k, i, j)) < 10 * cfg.tol
        end
    end

    @testset "a coarse manual plan is less accurate than the tuned plan" begin
        coarse_cfg = ChebyshevConfig(; n_panels_h=6, M_h=4, n_panels_j=4, M_j=4, param_strategy=:manual)
        rmin, rmax = _cheb_geom_rminmax(G, zj)
        c0, c1, cj0, cj1, _ = tune_cfie_cheb_plans(rmin, rmax, zj, coarse_cfg)
        @test c1[1].npanels == 6 && cj1[1].npanels == 4
        Cc = ChebRadialLookupCache(G, c1[1], cj1[1])
        F = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(F, pts, Rmat, G, Cc, ComplexF64(k), c0[1], c1[1], cj0[1], cj1[1])
        Ft = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(Ft, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1])
        @test maximum(abs, F - Ad) > 1e-6
        @test maximum(abs, F - Ad) > 1e3 * maximum(abs, Ft - Ad)
    end

    @testset "radii below the plan interval fall back to direct evaluation" begin
        _, rmax = _cheb_geom_rminmax(G, zj)
        h0, h1, j0, j1, _ = tune_cfie_cheb_plans(0.5, rmax, zj, cch_manual(cfg_used))
        Ch = ChebRadialLookupCache(G, h1[1], j1[1])
        offdiag = [(i, j) for i in 1:N for j in 1:N if i != j]
        @test any(ij -> Ch.pidx_h[ij...] == 0, offdiag) && any(ij -> Ch.pidx_h[ij...] != 0, offdiag)
        F = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(F, pts, Rmat, G, Ch, ComplexF64(k), h0[1], h1[1], j0[1], j1[1])
        @test maximum(abs, F - Ad) < 10 * cfg.tol
    end

    @testset "threaded assembly equals serial assembly" begin
        Ft = zeros(ComplexF64, N, N)
        Fs = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(Ft, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1]; multithreaded=true)
        _cfie_fredholm_full_cheb!(Fs, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1]; multithreaded=false)
        @test Ft == Fs
    end

    @testset "derivative kernel entries against the direct EBIM entries" begin
        Cd = C
        for (i, j) in [(1, 1), (1, 2), (2, 1), (5, 30), (30, 5), (12, 40), (N, 3)]
            v, dv, ddv = _cfie_kernel_entry_with_derivatives_cheb(pts, Rmat, G, Cd, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1], i, j)
            vd, dvd, ddvd = _cfie_kernel_entry_with_derivatives(pts, Rmat, G, k, i, j)
            @test abs(v - vd) < 1e-9
            @test abs(dv - dvd) < 1e-8
            @test abs(ddv - ddvd) < 1e-7
            @test abs(v - _cfie_kernel_entry_cheb(pts, Rmat, G, Cd, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1], i, j)) < 1e-12
        end
    end

    @testset "multi-wavenumber matrices (Beyn path) equal the single-wavenumber matrices" begin
        Fs = [Matrix{ComplexF64}(undef, N, N) for _ in zj]
        @test _cfie_fredholm_full_multi_k_cheb!(Fs, pts, Rmat, G, C, zj, plans0, plans1, plansj0, plansj1) === Fs
        F1 = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(F1, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1])
        F2 = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(F2, pts, Rmat, G, C, kc, plans0[2], plans1[2], plansj0[2], plansj1[2])
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
        @test_throws AssertionError _cfie_fredholm_full_multi_k_cheb!(Fs, pts, Rmat, G, C, zj, plans0, plans1, plansj0, plansj1)
    end
end

@testset "Chebyshev CFIE matrix: smooth boundary (unit circle)" begin
    circle = CircleBilliard(1.0)
    k = 3.0
    kc = 2.8 + 0.1im
    zj = ComplexF64[k, kc]
    cfg = cch_config()
    solver = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=64)
    pts = evaluate_points(solver, circle, k)
    N = length(pts)
    G, Rmat = cch_geometry(pts)
    plans0, plans1, plansj0, plansj1, C, _ = cch_plans(G, zj, cfg)
    Ad = construct_matrices(solver, pts, k)

    @testset "no grading, full matrix against the direct matrix" begin
        @test N == 64
        @test !_is_nontrivial_dlp_grading(pts)
        F = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(F, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1])
        @test maximum(abs, F - Ad) < 10 * cfg.tol
        Fc = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(Fc, pts, Rmat, G, C, kc, plans0[2], plans1[2], plansj0[2], plansj1[2])
        @test maximum(abs, Fc - construct_matrices(solver, pts, kc)) < 10 * cfg.tol
        @test F[2:end, 2:end] ≈ F[1:end-1, 1:end-1] atol = 1e-9         # circulant up to the interpolation error
        @test F ≈ transpose(F) atol = 1e-9
    end

    @testset "multi-wavenumber matrices (Beyn path) against the direct matrices" begin
        Fm = _construct_matrices_multi_k_cheb(solver, pts, zj, cfg)
        @test maximum(abs, Fm[1] - Ad) < 10 * cfg.tol
        @test maximum(abs, Fm[2] - construct_matrices(solver, pts, kc)) < 10 * cfg.tol
    end

    @testset "symmetry reduced matrix, Y reflection (reduced size 32 takes the threaded branch)" begin
        ssym = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=64, symmetry=YAxisReflection(), sym_characters=(1,))
        ptss = evaluate_points(ssym, circle, k)
        @test ptss.xy == pts.xy
        orbits = _fold_boundary(Float64, circle, N, ssym.symmetry, ssym.sym_characters)
        m = fundamental_size(orbits)
        @test m == 32
        Ar = construct_matrices(ssym, ptss, k)
        Fr = zeros(ComplexF64, m, m)
        @test _cfie_fredholm_reduced_cheb!(Fr, ptss, Rmat, G, C, orbits, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1]) === Fr
        @test maximum(abs, Fr - Ar) < 10 * cfg.tol

        Ff = zeros(ComplexF64, N, N)
        _cfie_fredholm_full_cheb!(Ff, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1])
        @test isapprox(Fr, cch_orbit_sum(Ff, orbits); atol=1e-11, rtol=0)

        Frs = zeros(ComplexF64, m, m)
        _cfie_fredholm_reduced_cheb!(Frs, ptss, Rmat, G, C, orbits, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1]; multithreaded=false)
        @test Fr == Frs                                                 # threaded = serial

        Fs = [Matrix{ComplexF64}(undef, m, m) for _ in zj]
        @test _cfie_fredholm_reduced_multi_k_cheb!(Fs, ptss, Rmat, G, C, orbits, zj, plans0, plans1, plansj0, plansj1) === Fs
        @test isapprox(Fs[1], Fr; atol=1e-12, rtol=0)
        @test maximum(abs, Fs[2] - construct_matrices(ssym, ptss, kc)) < 10 * cfg.tol

        Fm = _construct_matrices_multi_k_cheb(ssym, ptss, zj, cfg)      # dispatches to the reduced assembly
        @test all(M -> size(M) == (m, m), Fm)
        @test maximum(abs, Fm[1] - Ar) < 10 * cfg.tol
    end
end

@testset "Chebyshev CFIE matrix: symmetry reduced rectangle with corner grading" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    cfg = cch_config()
    solver = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=48, grading=GlobalCornerGrading(kressq=2), symmetry=XYAxisReflection(), sym_characters=(1, 1))
    pts = evaluate_points(solver, rect, k)
    N = length(pts)
    G, Rmat = cch_geometry(pts)
    plans0, plans1, plansj0, plansj1, C, _ = cch_plans(G, ComplexF64[k], cfg)
    orbits = _fold_boundary(Float64, rect, N, solver.symmetry, solver.sym_characters)
    m = fundamental_size(orbits)

    @test _is_nontrivial_dlp_grading(pts)
    @test m == N ÷ 4 == 12
    Fr = zeros(ComplexF64, m, m)
    _cfie_fredholm_reduced_cheb!(Fr, pts, Rmat, G, C, orbits, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1])
    @test maximum(abs, Fr - construct_matrices(solver, pts, k)) < 10 * cfg.tol
    Ff = zeros(ComplexF64, N, N)
    _cfie_fredholm_full_cheb!(Ff, pts, Rmat, G, C, ComplexF64(k), plans0[1], plans1[1], plansj0[1], plansj1[1])
    @test isapprox(Fr, cch_orbit_sum(Ff, orbits); atol=1e-11, rtol=0)
    Fm = _construct_matrices_multi_k_cheb(solver, pts, ComplexF64[k], cfg)
    @test size(Fm[1]) == (m, m) && maximum(abs, Fm[1] - Fr) < 10 * cfg.tol
end

@testset "Chebyshev CFIE matrix: annulus composite and error paths" begin
    cfg = cch_config()
    k = 3.0

    @testset "CompositeBIMSolver (annulus) has no Chebyshev matrix construction" begin
        annulus = AnnularBilliard(0.3, 0.1)
        comp = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24))
        pts = evaluate_points(comp, annulus, k)
        @test_throws ErrorException _construct_matrices_multi_k_cheb(comp, pts, ComplexF64[k], cfg)
        @test tune_cheb_config(comp, pts, k, cfg) === cfg                # no plan to tune: the configuration is passed through
        A = construct_matrices(comp, pts, k)                             # the direct composite matrix is the supported path
        @test size(A) == (56, 56) && A isa Matrix{ComplexF64}
    end

    @testset "Float32 points are rejected" begin
        circle = CircleBilliard(1.0)
        solver32 = CombinedFieldIntegralEquationSolver(5f0, circle; min_pts=16)
        pts32 = evaluate_points(solver32, circle, 3f0)
        @test pts32 isa BoundaryPoints{Float32}
        @test_throws ErrorException _construct_matrices_multi_k_cheb(solver32, pts32, ComplexF64[k], cfg)
    end

    @testset "empty wavenumber list" begin
        @test_throws ArgumentError tune_cfie_cheb_plans(0.1, 2.0, ComplexF64[], cfg)
    end
end
