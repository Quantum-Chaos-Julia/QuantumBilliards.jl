# Unit tests for
#   src/solvers/taylor/bim_kernels_recurrence.jl   (_taylor_cache, _kernel_taylor! for the DLP and CFIE kernels, same component and smooth
#                                                   cross component, _fredholm_taylor!, _construct_matrices_multi_k_taylor_full/_reduced,
#                                                   _construct_matrices_multi_k_taylor_derivatives)
# Matrix and kernel-entry level only: no eigen solve. The kernel coefficients a_n = K^(n)(k) / n! of one entry are compared with
#   * closed forms built from SpecialFunctions: with Z in {J, H}, d/dk [k Z1(k r)] = k r Z0(k r) and d/dk [k Z0(k r)] = Z0(k r) - k r Z1(k r),
#     hence the derivatives (k Z1, k r Z0, r Z0 - k r^2 Z1) and (k Z0, Z0 - k r Z1, -r Z1 - k r^2 Z0) of orders 0, 1, 2, inserted into the
#     Kress split K = R L1 + w L2 (+ i k (R M1 + w M2) for the CFIE); the smooth cross-component entries have no Kress split;
#   * the direct kernel entry (n = 0), the direct EBIM derivative entries (n = 1, 2) and the direct entry at a shifted wavenumber (Taylor sum);
#   * the closed form of the logarithmic CFIE diagonal, K_ii = w kappa + i k (c - a log k), a = w s / pi, whose coefficients are
#     a_n = i a (-1)^(n-1) / (n (n - 1) k^(n-1)) for n >= 2.
# Further checks: the prefix property (the first p + 1 coefficients do not depend on the degree), A = I - K for _fredholm_taylor!, the Taylor
# matrix error decreases with the degree (single component and annulus composite), the derivative matrices against central differences of the
# direct matrix, and argument/type errors. The plain agreement "direct = Taylor at degree 10" is covered in dlp_tests.jl, cfie_tests.jl and
# compositebim_tests.jl and is not repeated. Helper definitions use the prefix `tk_`.

using LinearAlgebra: norm
using SpecialFunctions: besselj, besselh
using QuantumBilliards: TaylorWorkspace, _taylor_cache, _kernel_taylor!, _fredholm_taylor!, _construct_matrices_multi_k_taylor,
                        _construct_matrices_multi_k_taylor_full, _construct_matrices_multi_k_taylor_reduced,
                        _construct_matrices_multi_k_taylor_derivatives, boundary_geom_cache, kress_R!, _is_nontrivial_dlp_grading,
                        _dlp_kernel_entry, _cfie_kernel_entry, _dlp_kernel_entry_with_derivatives, _cfie_kernel_entry_with_derivatives

# Coefficients (orders 0, 1, 2) of one raw kernel entry, copied out of the reused workspace.
function tk_coeffs(cache, k, p, i, j)
    return copy(_kernel_taylor!(cache, TaylorWorkspace(p), k, p, i, j))
end

# Derivatives of orders 0, 1, 2 of k Z1(k r) and k Z0(k r) for given values Z0 = Z0(k r), Z1 = Z1(k r).
tk_dZ1(Z0, Z1, k, r) = (k * Z1, k * r * Z0, r * Z0 - k * r^2 * Z1)
tk_dZ0(Z0, Z1, k, r) = (k * Z0, Z0 - k * r * Z1, -r * Z1 - k * r^2 * Z0)

# Coefficients (orders 0, 1, 2) of an off-diagonal same-component Kress entry from the closed forms. `cfie` adds i k (R M1 + w M2).
function tk_kress_coeffs(pts, Rmat, G, k, i, j; cfie::Bool=false)
    r, invr, lt, inn = G.R[i, j], G.invR[i, j], G.logterm[i, j], G.inner[i, j]
    j0, j1, h0, h1 = besselj(0, k * r), besselj(1, k * r), besselh(0, 1, k * r), besselh(1, 1, k * r)
    dJ1, dH1 = tk_dZ1(j0, j1, k, r), tk_dZ1(h0, h1, k, r)
    dJ0, dH0 = tk_dZ0(j0, j1, k, r), tk_dZ0(h0, h1, k, r)
    return map(1:3) do n
        L1 = -inn * invr / (2π) * dJ1[n]
        L2 = im / 2 * inn * invr * dH1[n] - L1 * lt
        v = Rmat[i, j] * L1 + pts.ws[j] * L2
        if cfie
            sj = G.speed[j]
            ikM1 = -im / (2π) * sj * dJ0[n]
            ikM2 = -sj / 2 * dH0[n] + im / (2π) * sj * lt * dJ0[n]
            v += Rmat[i, j] * ikM1 + pts.ws[j] * ikM2
        end
        v / factorial(n - 1)
    end
end

# Coefficients (orders 0, 1, 2) of a smooth cross-component entry from the merged nodes: target i, source j.
function tk_cross_coeffs(pts, k, i, j; cfie::Bool)
    dx, dy = pts.xy[i] - pts.xy[j]
    r = hypot(dx, dy)
    tx, ty = pts.tangent[j]
    inn = ty * dx - tx * dy
    h0, h1 = besselh(0, 1, k * r), besselh(1, 1, k * r)
    dH1, dH0 = tk_dZ1(h0, h1, k, r), tk_dZ0(h0, h1, k, r)
    cD = pts.ws[j] * im / 2 * inn / r
    cS = cfie ? -pts.ws[j] * hypot(tx, ty) / 2 : 0.0
    return [(cD * dH1[n] + cS * dH0[n]) / factorial(n - 1) for n in 1:3]
end

# Taylor polynomial of the stored coefficients at the shift δ.
tk_poly(a, δ) = sum(a[l+1] * δ^l for l in 0:length(a)-1)

@testset "_taylor_cache" begin
    @testset "single component: geometry cache and Kress matrix of the nodes" begin
        circle = CircleBilliard(1.0)
        solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=24)
        pts = evaluate_points(solver, circle, 3.0)
        cache = _taylor_cache(solver, pts)
        R = zeros(24, 24)
        kress_R!(R)
        @test cache.solver === solver && cache.pts === pts
        @test cache.N == 24
        @test cache.Rmat == R
        @test size(cache.G.R) == (24, 24)
        @test cache.G.R[3, 7] ≈ norm(pts.xy[3] - pts.xy[7])
    end

    @testset "composite: component blocks and global to local maps" begin
        annulus = AnnularBilliard(0.3, 0.1)
        comp = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24))
        pts = evaluate_points(comp, annulus, 3.0)
        cache = _taylor_cache(comp, pts)
        @test cache.N == 56 && cache.solver === comp
        @test length(cache.comp_pts) == 2 && length(cache.Gs) == 2 && length(cache.Rmats) == 2
        @test length.(cache.comp_pts) == (32, 24)
        @test size(cache.Rmats[1]) == (32, 32) && size(cache.Rmats[2]) == (24, 24)
        @test cache.g2c == [fill(1, 32); fill(2, 24)]
        @test cache.g2l == [1:32; 1:24]
        @test cache.comp_pts[2].xy == pts.xy[33:56]
    end
end

@testset "_kernel_taylor!: DLP, circle (smooth)" begin
    circle = CircleBilliard(1.0)
    k = 3.0
    solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=24)
    pts = evaluate_points(solver, circle, k)
    cache = _taylor_cache(solver, pts)
    G, Rmat = cache.G, cache.Rmat
    pairs = [(1, 2), (2, 1), (1, 13), (5, 20), (24, 1)]

    @testset "off-diagonal coefficients of order 0, 1, 2 against the closed forms" begin
        for (i, j) in pairs
            a = tk_coeffs(cache, k, 8, i, j)
            @test length(a) == 9
            @test a[1:3] ≈ tk_kress_coeffs(pts, Rmat, G, k, i, j) rtol = 1e-10 atol = 1e-13
        end
    end

    @testset "order 0 is the direct kernel entry, orders 1 and 2 the direct EBIM derivatives" begin
        for (i, j) in vcat(pairs, [(7, 7)])
            a = tk_coeffs(cache, k, 4, i, j)
            v, dv, ddv = _dlp_kernel_entry_with_derivatives(pts, Rmat, G, k, i, j)
            @test a[1] ≈ _dlp_kernel_entry(pts, Rmat, G, k, i, j) atol = 1e-13
            @test a[1] ≈ v atol = 1e-13
            @test a[2] ≈ dv atol = 1e-12
            @test a[3] ≈ ddv / 2 atol = 1e-12
        end
    end

    @testset "diagonal: only the k independent curvature term" begin
        a = tk_coeffs(cache, k, 6, 9, 9)
        @test a[1] ≈ pts.ws[9] * G.kappa[9]
        @test all(iszero, a[2:end])
    end

    @testset "prefix property: the first coefficients do not depend on the degree" begin
        a_low = tk_coeffs(cache, k, 3, 4, 17)
        a_mid = tk_coeffs(cache, k, 9, 4, 17)
        a_high = tk_coeffs(cache, k, 14, 4, 17)
        @test length(a_low) == 4 && length(a_mid) == 10 && length(a_high) == 15
        @test a_mid[1:4] ≈ a_low rtol = 1e-13
        @test a_high[1:10] ≈ a_mid rtol = 1e-13
    end

    @testset "degree 0 is the entry itself" begin
        a = tk_coeffs(cache, k, 0, 4, 17)
        @test length(a) == 1
        @test a[1] ≈ _dlp_kernel_entry(pts, Rmat, G, k, 4, 17) atol = 1e-13
    end

    @testset "the workspace can be reused for different entries" begin
        w = TaylorWorkspace(6)
        a1 = copy(_kernel_taylor!(cache, w, k, 6, 4, 17))
        _kernel_taylor!(cache, w, k, 6, 9, 9)                           # diagonal entry resets all higher coefficients
        @test all(iszero, w.a[2:end])
        @test copy(_kernel_taylor!(cache, w, k, 6, 4, 17)) == a1
        @test _kernel_taylor!(cache, w, k, 6, 4, 17) === w.a            # the workspace vector itself is returned
    end

    @testset "the Taylor polynomial reproduces the entry at shifted wavenumbers" begin
        a = tk_coeffs(cache, k, 12, 4, 17)
        for δ in (0.05, -0.2, 0.3, 0.1 + 0.1im)
            @test tk_poly(a, δ) ≈ _dlp_kernel_entry(pts, Rmat, G, k + δ, 4, 17) atol = 1e-10
        end
    end
end

@testset "_kernel_taylor!: DLP, rectangle with corner grading" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading(kressq=2))
    pts = evaluate_points(solver, rect, k)
    cache = _taylor_cache(solver, pts)
    G, Rmat = cache.G, cache.Rmat

    @test _is_nontrivial_dlp_grading(pts)
    for (i, j) in [(1, 2), (3, 20), (20, 3), (31, 5), (16, 32)]       # includes close pairs next to the corners
        a = tk_coeffs(cache, k, 8, i, j)
        @test a[1:3] ≈ tk_kress_coeffs(pts, Rmat, G, k, i, j) rtol = 1e-10 atol = 1e-13
        @test a[1] ≈ _dlp_kernel_entry(pts, Rmat, G, k, i, j) atol = 1e-13
    end
    a = tk_coeffs(cache, k, 12, 3, 20)
    @test tk_poly(a, 0.2) ≈ _dlp_kernel_entry(pts, Rmat, G, k + 0.2, 3, 20) atol = 1e-10
    @test tk_coeffs(cache, k, 5, 8, 8)[1] ≈ pts.ws[8] * G.kappa[8]    # straight edge: the curvature vanishes at every node
end

@testset "_kernel_taylor!: CFIE, circle" begin
    circle = CircleBilliard(1.0)
    k = 3.0
    solver = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=24)
    pts = evaluate_points(solver, circle, k)
    cache = _taylor_cache(solver, pts)
    G, Rmat = cache.G, cache.Rmat

    @testset "off-diagonal coefficients of order 0, 1, 2 against the closed forms" begin
        for (i, j) in [(1, 2), (2, 1), (1, 13), (5, 20), (24, 1)]
            a = tk_coeffs(cache, k, 8, i, j)
            @test a[1:3] ≈ tk_kress_coeffs(pts, Rmat, G, k, i, j; cfie=true) rtol = 1e-10 atol = 1e-13
        end
    end

    @testset "order 0 is the direct kernel entry, orders 1 and 2 the direct EBIM derivatives" begin
        for (i, j) in [(1, 2), (2, 1), (5, 20), (7, 7)]
            a = tk_coeffs(cache, k, 4, i, j)
            v, dv, ddv = _cfie_kernel_entry_with_derivatives(pts, Rmat, G, k, i, j)
            @test a[1] ≈ _cfie_kernel_entry(pts, Rmat, G, k, i, j) atol = 1e-12
            @test a[1] ≈ v atol = 1e-12
            @test a[2] ≈ dv atol = 1e-11
            @test a[3] ≈ ddv / 2 atol = 1e-11
        end
    end

    @testset "logarithmic diagonal: closed form of the coefficients" begin
        i = 6
        a = tk_coeffs(cache, k, 8, i, i)
        alog = pts.ws[i] * G.speed[i] / π
        for n in 2:8
            @test a[n+1] ≈ im * alog * (-1)^(n - 1) / (n * (n - 1) * k^(n - 1)) rtol = 1e-12
        end
        @test a[1] ≈ _cfie_kernel_entry(pts, Rmat, G, k, i, i)
        @test imag(a[3]) ≈ -alog / (2k) rtol = 1e-12                    # n = 2: purely imaginary
        @test real(a[3]) ≈ 0 atol = 1e-14
    end

    @testset "the Taylor polynomial reproduces the entries at shifted wavenumbers, diagonal included" begin
        for (i, j) in [(4, 17), (6, 6)]
            a = tk_coeffs(cache, k, 14, i, j)
            for δ in (0.05, -0.2, 0.3, 0.1 + 0.1im)
                @test tk_poly(a, δ) ≈ _cfie_kernel_entry(pts, Rmat, G, k + δ, i, j) atol = 1e-9
            end
        end
    end

    @testset "prefix property" begin
        @test tk_coeffs(cache, k, 9, 6, 6)[1:5] ≈ tk_coeffs(cache, k, 4, 6, 6) rtol = 1e-13      # diagonal
        @test tk_coeffs(cache, k, 9, 4, 17)[1:5] ≈ tk_coeffs(cache, k, 4, 4, 17) rtol = 1e-13    # off-diagonal
    end

    @testset "degree 1 truncates the diagonal series" begin
        a = tk_coeffs(cache, k, 1, 6, 6)
        @test length(a) == 2
        @test a[2] ≈ _cfie_kernel_entry_with_derivatives(pts, Rmat, G, k, 6, 6)[2] atol = 1e-12
    end
end

@testset "_kernel_taylor!: annulus composite, same and cross component entries" begin
    annulus = AnnularBilliard(0.3, 0.1)
    k = 3.0
    comp = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24))
    pts = evaluate_points(comp, annulus, k)
    cache = _taylor_cache(comp, pts)
    outer_idx, hole_idx = 1:32, 33:56

    @testset "cross component entries: closed form of the smooth kernels, orders 0, 1, 2" begin
        for (i, j) in [(3, 33 + 5), (30, 33 + 20)]                      # source on the CFIE hole: D + i k S
            @test tk_coeffs(cache, k, 8, i, j)[1:3] ≈ tk_cross_coeffs(pts, k, i, j; cfie=true) rtol = 1e-10 atol = 1e-13
        end
        for (i, j) in [(33 + 5, 3), (33 + 20, 30)]                      # source on the DLP outer circle: D only
            @test tk_coeffs(cache, k, 8, i, j)[1:3] ≈ tk_cross_coeffs(pts, k, i, j; cfie=false) rtol = 1e-10 atol = 1e-13
        end
    end

    @testset "cross component entries are smooth: no log singularity in k, polynomial reproduces the entry" begin
        i, j = 3, 33 + 5
        a = tk_coeffs(cache, k, 12, i, j)
        for δ in (0.2, -0.3)
            @test tk_poly(a, δ) ≈ tk_cross_coeffs(pts, k + δ, i, j; cfie=true)[1] atol = 1e-10
        end
    end

    @testset "same component entries use the component Kress kernels" begin
        G1, R1 = cache.Gs[1], cache.Rmats[1]
        G2, R2 = cache.Gs[2], cache.Rmats[2]
        @test tk_coeffs(cache, k, 8, 4, 19)[1:3] ≈ tk_kress_coeffs(cache.comp_pts[1], R1, G1, k, 4, 19) rtol = 1e-10 atol = 1e-13
        @test tk_coeffs(cache, k, 8, 33 + 3, 33 + 15)[1:3] ≈ tk_kress_coeffs(cache.comp_pts[2], R2, G2, k, 3, 15; cfie=true) rtol = 1e-10 atol = 1e-13
        @test tk_coeffs(cache, k, 6, 10, 10)[2:end] == zeros(ComplexF64, 6)                  # DLP diagonal: k independent
        @test any(!iszero, tk_coeffs(cache, k, 6, 33 + 10, 33 + 10)[2:end])                  # CFIE diagonal: i k S_ii depends on k
    end

    @testset "blocks of the matrix: global indices map onto the component kernels" begin
        for (gi, gj, ca, ia, jb) in [(4, 19, 1, 4, 19), (33 + 3, 33 + 15, 2, 3, 15)]
            @test tk_coeffs(cache, k, 5, gi, gj) ≈ tk_coeffs(_taylor_cache(comp.component_solvers[ca], cache.comp_pts[ca]), k, 5, ia, jb) rtol = 1e-12
        end
    end
end

@testset "_fredholm_taylor!: A = I - K" begin
    circle = CircleBilliard(1.0)
    k = 3.0
    solver = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=24)
    pts = evaluate_points(solver, circle, k)
    cache = _taylor_cache(solver, pts)
    w = TaylorWorkspace(6)

    for (i, j) in [(4, 17), (6, 6)]
        K = tk_coeffs(cache, k, 6, i, j)
        A = copy(_fredholm_taylor!(cache, w, k, 6, i, j))
        @test A[2:end] == -K[2:end]
        @test A[1] ≈ (i == j ? 1 : 0) - K[1]
    end
    @test _fredholm_taylor!(cache, w, k, 6, 4, 17) === w.a
end

@testset "Taylor matrix error decreases with the degree" begin
    k0 = 3.0
    ks = [k0 + 0.3]
    degrees = (1, 3, 5, 7)

    @testset "DLP, unit circle" begin
        circle = CircleBilliard(1.0)
        solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=24)
        pts = evaluate_points(solver, circle, k0)
        A = construct_matrices(solver, pts, ks[1])
        errs = [maximum(abs, _construct_matrices_multi_k_taylor(solver, pts, k0, ks, p)[1] - A) for p in degrees]
        @test issorted(errs; rev=true) && allunique(errs)
        @test errs[1] > 100 * errs[end]
        @test maximum(abs, _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 14)[1] - A) < 1e-10
    end

    @testset "CFIE, unit circle (logarithmic diagonal converges geometrically with ratio delta / k)" begin
        circle = CircleBilliard(1.0)
        solver = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=24)
        pts = evaluate_points(solver, circle, k0)
        A = construct_matrices(solver, pts, ks[1])
        errs = [maximum(abs, _construct_matrices_multi_k_taylor(solver, pts, k0, ks, p)[1] - A) for p in degrees]
        @test issorted(errs; rev=true) && allunique(errs)
        @test errs[1] > 100 * errs[end]
        @test maximum(abs, _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 16)[1] - A) < 1e-9
    end

    @testset "annulus composite with a DLP outer and a CFIE hole component" begin
        annulus = AnnularBilliard(0.3, 0.1)
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24))
        pts = evaluate_points(solver, annulus, k0)
        A = construct_matrices(solver, pts, ks[1])
        errs = [maximum(abs, _construct_matrices_multi_k_taylor(solver, pts, k0, ks, p)[1] - A) for p in degrees]
        @test issorted(errs; rev=true) && allunique(errs)
        @test errs[1] > 100 * errs[end]
        @test maximum(abs, _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 16)[1] - A) < 1e-9
    end
end

@testset "Taylor derivative matrices" begin
    k0 = 3.0
    ks = [3.0, 3.05]
    h = 1e-3

    @testset "CFIE circle, full: A equals the Taylor matrix, A' and A'' equal central differences of the direct matrix" begin
        circle = CircleBilliard(1.0)
        solver = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=24)
        pts = evaluate_points(solver, circle, k0)
        A, dA, ddA = _construct_matrices_multi_k_taylor_derivatives(solver, pts, k0, ks, 12)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 12)
        @test length(A) == length(dA) == length(ddA) == 2
        for q in 1:2
            Ap, A0, Am = (construct_matrices(solver, pts, ks[q] + s) for s in (h, 0.0, -h))
            @test A[q] ≈ T[q] atol = 1e-12
            @test dA[q] ≈ (Ap - Am) / (2h) atol = 1e-5
            @test ddA[q] ≈ (Ap - 2A0 + Am) / h^2 atol = 1e-4
        end
    end

    @testset "DLP rectangle, XY reflection reduced, corner graded" begin
        rect = RectangleBilliard(1.0, 0.6)
        solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading(kressq=2), symmetry=XYAxisReflection(), sym_characters=(1, 1))
        pts = evaluate_points(solver, rect, k0)
        m = boundary_matrix_size(solver, pts)
        A, dA, ddA = _construct_matrices_multi_k_taylor_derivatives(solver, pts, k0, ks, 12)
        @test all(M -> size(M) == (m, m), vcat(A, dA, ddA))
        Ap, A0, Am = (construct_matrices(solver, pts, ks[2] + s) for s in (h, 0.0, -h))
        @test A[2] ≈ A0 atol = 1e-10
        @test dA[2] ≈ (Ap - Am) / (2h) atol = 1e-5
        @test ddA[2] ≈ (Ap - 2A0 + Am) / h^2 atol = 1e-4
    end

    @testset "annulus composite, full" begin
        annulus = AnnularBilliard(0.3, 0.1)
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(5.0, annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24))
        pts = evaluate_points(solver, annulus, k0)
        A, dA, ddA = _construct_matrices_multi_k_taylor_derivatives(solver, pts, k0, ks[1:1], 12)
        Ap, A0, Am = (construct_matrices(solver, pts, k0 + s) for s in (h, 0.0, -h))
        @test A[1] ≈ A0 atol = 1e-10
        @test dA[1] ≈ (Ap - Am) / (2h) atol = 1e-5
        @test ddA[1] ≈ (Ap - 2A0 + Am) / h^2 atol = 1e-4
    end
end

@testset "Taylor matrix construction: argument and type errors" begin
    circle = CircleBilliard(1.0)
    k0 = 3.0
    full = DoubleLayerPotentialSolver(5.0, circle; min_pts=24)
    sym = DoubleLayerPotentialSolver(5.0, circle; min_pts=24, symmetry=YAxisReflection(), sym_characters=(1,))
    pts = evaluate_points(full, circle, k0)

    @testset "full and reduced builders check the symmetry of the solver" begin
        @test_throws ArgumentError _construct_matrices_multi_k_taylor_full(sym, pts, k0, [k0], 4)
        @test_throws ArgumentError _construct_matrices_multi_k_taylor_reduced(full, pts, k0, [k0], 4)
        @test size(_construct_matrices_multi_k_taylor(sym, pts, k0, [k0], 4)[1]) == (12, 12)       # the dispatcher picks the reduced builder
        @test size(_construct_matrices_multi_k_taylor(full, pts, k0, [k0], 4)[1]) == (24, 24)
    end

    @testset "empty wavenumber list gives no matrices" begin
        @test isempty(_construct_matrices_multi_k_taylor(full, pts, k0, Float64[], 4))
    end

    @testset "Float32 solvers and points are not supported (Float64 only)" begin
        solver32 = DoubleLayerPotentialSolver(5f0, circle; min_pts=24)
        pts32 = evaluate_points(solver32, circle, 3f0)
        @test pts32 isa BoundaryPoints{Float32}
        @test_throws MethodError _construct_matrices_multi_k_taylor(solver32, pts32, k0, [k0], 4)
        @test_throws MethodError _construct_matrices_multi_k_taylor_derivatives(solver32, pts32, k0, [k0], 4)
    end
end
