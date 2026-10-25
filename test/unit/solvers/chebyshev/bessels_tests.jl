# Unit tests for
#   src/solvers/chebyshev/bessels.jl   (plan_h, plan_j, ChebHankelTableH/ChebJTable builders, panel lookup,
#                                       eval_h, eval_j, multi-wavenumber evaluators, small-argument Hankel series,
#                                       ChebRadialLookupCache)
#   src/solvers/chebyshev/core.jl      (_chebfit!, _breaks_uniform, _cheb_clenshaw, _cheb_clenshaw_col,
#                                       _cheb_nthreads_buf)
# All expected values are SpecialFunctions.besselh/besselj evaluations, exact Chebyshev identities
# T_m(t) = cos(m acos t), or analytic panel geometry. Plans are tiny (a handful of panels, no BIM matrices).
# eval_h's mid-range cutoff fidelity (|z| < hankel_z_chebyshev_cutoff inside the panelized interval) is checked for eval_h only:
# it evaluates directly below the cutoff even when pidx != 0 (the multi-k variants still only fall back for pidx == 0).

using SpecialFunctions
using QuantumBilliards: plan_h, plan_j, panel_t, panel_and_geom, eval_h, eval_j, eval_h_multi_ks!, eval_j_multi_ks!,
                        h0_h1_multi_ks_at_r!, h1_j1_multi_ks_at_r!, h0_h1_j0_j1_multi_ks_at_r!, ChebRadialLookupCache,
                        ChebHankelPlanH, ChebJPlan, ChebHankelTableH, ChebJTable, BoundaryGeomCache,
                        hankel_z_chebyshev_cutoff, hankel_z_chebyshev_cutoff_small_z,
                        _build_table_h!, _build_table_j!, _find_panel, _find_panel_uniform,
                        _small_h0_series, _small_h1_series,
                        _chebfit!, _breaks_uniform, _cheb_clenshaw, _cheb_clenshaw_col, _cheb_nthreads_buf

# Chebyshev sum Σ c[m+1] T_m(t) from T_m(t) = cos(m acos t), valid for |t| ≤ 1.
bs_chebsum(c, t) = sum(c[m+1]*cos(m*acos(t)) for m in 0:length(c)-1)

# Largest error of eval_h / eval_j against the direct special function, scaled by max(1, |ref|).
function bs_max_error_h(pl::ChebHankelPlanH, rs)
    maximum(rs) do r
        p, t = panel_t(pl, r)
        ref = besselh(pl.ν, pl.κ, pl.k*r)
        abs(eval_h(pl, p, t, r) - ref)/max(1.0, abs(ref))
    end
end
function bs_max_error_j(pl::ChebJPlan, rs)
    maximum(rs) do r
        p, t = panel_t(pl, r)
        ref = besselj(pl.ν, pl.k*r)
        abs(eval_j(pl, p, t, r) - ref)/max(1.0, abs(ref))
    end
end

# Wavenumbers, interval and resolution shared by the plan tests: z = kr stays in [2, 12] so that no
# near-origin fallback is involved, and each panel is resolved to roundoff by degree-16 interpolation.
const BS_KS = ComplexF64[4.0, 3.0 + 0.4im]
const BS_RMIN, BS_RMAX, BS_NP, BS_M = 0.5, 3.0, 8, 16

@testset "_chebfit!" begin
    @testset "recovers the coefficients of a degree-M Chebyshev series" begin
        M = 6
        ctrue = ComplexF64[0.3 + 0.1im, 1.2, -0.4im, -0.5, 0.0, 0.25, 0.8 - 0.2im]
        f = ComplexF64[bs_chebsum(ctrue, cospi(j/M)) for j in 0:M]     # samples at Chebyshev-Lobatto nodes
        c = zeros(ComplexF64, M + 1)
        @test _chebfit!(c, f) === c                                    # in-place, returns the output vector
        @test c ≈ ctrue atol=1e-13
    end

    @testset "constant and linear functions" begin
        @test _chebfit!(zeros(ComplexF64, 5), fill(ComplexF64(2.5), 5)) ≈ ComplexF64[2.5, 0, 0, 0, 0] atol=1e-14
        # M = 1: nodes t = ±1, f = c₀ + c₁t
        c = _chebfit!(zeros(ComplexF64, 2), ComplexF64[3.0 + 1.0, 3.0 - 1.0])
        @test c ≈ ComplexF64[3.0, 1.0]
    end

    @testset "type stability" begin
        f = ComplexF64[cospi(j/4) for j in 0:4]
        @test @inferred(_chebfit!(zeros(ComplexF64, 5), f)) isa Vector{ComplexF64}
    end
end

@testset "_breaks_uniform" begin
    b = _breaks_uniform(0.5, 2.0, 5)
    @test b isa Vector{Float64}
    @test length(b) == 6
    @test b[1] == 0.5 && b[end] == 2.0                             # endpoints exact
    @test issorted(b; lt=<=)                                       # strictly increasing
    @test all(isapprox.(diff(b), 0.3; atol=1e-14))                 # uniform spacing (rmax - rmin)/np
    @test _breaks_uniform(1.0, 4.0, 1) == [1.0, 4.0]
    @test @inferred(_breaks_uniform(0.5, 2.0, 5)) isa Vector{Float64}
end

@testset "_cheb_clenshaw" begin
    @testset "matches the direct Chebyshev sum for every degree incl. unrolled M = 4:10" begin
        for M in 0:12
            c = ComplexF64[cis(0.9*m)/(m + 1) for m in 0:M]
            for t in (-1.0, -0.3, 0.0, 0.77, 1.0)
                @test _cheb_clenshaw(c, t) ≈ bs_chebsum(c, t) atol=1e-13
            end
        end
    end

    @testset "endpoint values T_m(1) = 1 and T_m(-1) = (-1)^m" begin
        c = ComplexF64[cis(0.9*m)/(m + 1) for m in 0:7]
        @test _cheb_clenshaw(c, 1.0) ≈ sum(c) atol=1e-14
        @test _cheb_clenshaw(c, -1.0) ≈ sum((-1)^m*c[m+1] for m in 0:7) atol=1e-14
    end

    @testset "accepts a vector view and is type stable" begin
        C = ComplexF64[cis(0.5*(m + 3*j)) for m in 0:6, j in 1:2]
        @test _cheb_clenshaw(view(C, :, 2), 0.4) ≈ bs_chebsum(C[:, 2], 0.4) atol=1e-13
        @test @inferred(_cheb_clenshaw(C[:, 1], 0.4)) isa ComplexF64
    end
end

@testset "_cheb_clenshaw_col" begin
    for M in 0:12
        C = ComplexF64[cis(0.5*(m + 3*j))/(m + 1) for m in 0:M, j in 1:3]
        for j in 1:3, t in (-1.0, -0.3, 0.77, 1.0)
            @test _cheb_clenshaw_col(C, j, M, t) ≈ bs_chebsum(C[:, j], t) atol=1e-13
        end
    end
    C = ComplexF64[cis(0.5*(m + j)) for m in 0:6, j in 1:2]
    @test @inferred(_cheb_clenshaw_col(C, 2, 6, 0.4)) isa ComplexF64
end

@testset "_cheb_nthreads_buf" begin
    @test _cheb_nthreads_buf() == Threads.maxthreadid()
    @test _cheb_nthreads_buf() >= Threads.nthreads()
    @test @inferred(_cheb_nthreads_buf()) isa Int
end

@testset "_small_h0_series and _small_h1_series" begin
    zs = (1e-4, 5e-4, hankel_z_chebyshev_cutoff_small_z, 1e-2)
    @testset "H₀ series agrees with besselh(0,1,z) for small real and complex z" begin
        for z in zs
            @test _small_h0_series(z) ≈ besselh(0, 1, z) rtol=1e-12
            zc = ComplexF64(z, 0.5z)
            @test _small_h0_series(zc) ≈ besselh(0, 1, zc) rtol=1e-12
        end
    end

    @testset "H₁ series agrees with besselh(1,1,z) for small real and complex z" begin
        for z in zs
            @test _small_h1_series(z) ≈ besselh(1, 1, z) rtol=1e-12
            zc = ComplexF64(z, 0.5z)
            @test _small_h1_series(zc) ≈ besselh(1, 1, zc) rtol=1e-12
        end
    end

    @testset "leading behavior and dispatch" begin
        z = 1e-5
        @test imag(_small_h0_series(z)) ≈ (2/π)*(log(z/2) + MathConstants.eulergamma) rtol=1e-8   # Y₀(z) ≈ (2/π)(ln(z/2)+γ)
        @test _small_h1_series(z) ≈ -im*2/(π*z) rtol=1e-4                                          # H₁ ≈ -2i/(πz)
        @test _small_h0_series(1f-3) ≈ _small_h0_series(ComplexF64(Float32(1e-3)))                 # generic Number method promotes
        @test @inferred(_small_h0_series(1e-3 + 0im)) isa ComplexF64
        @test @inferred(_small_h1_series(1e-3)) isa ComplexF64
    end
end

@testset "_build_table_h!" begin
    @testset "table interpolates H_ν^(κ)(kr) on its panel" begin
        k = ComplexF64(4.0, 0.3); a, b, M = 1.0, 1.2, 12
        for ν in (0, 1, 2), κ in (1, 2)
            tbl = _build_table_h!(ν, κ, k, a, b; M=M)
            @test tbl isa ChebHankelTableH
            @test (tbl.a, tbl.b, tbl.M, tbl.ν, tbl.κ) == (a, b, M, ν, κ)
            @test length(tbl.c) == M + 1
            for t in range(-1, 1; length=7)
                r = ((b + a) + (b - a)*t)/2
                @test _cheb_clenshaw(tbl.c, t) ≈ besselh(ν, κ, k*r) rtol=1e-11
            end
        end
    end

    @testset "argument errors" begin
        k = ComplexF64(4.0)
        @test_throws AssertionError _build_table_h!(0, 1, k, 0.0, 1.0)     # singular at the origin: a > 0 required
        @test_throws AssertionError _build_table_h!(0, 1, k, -0.1, 1.0)
        @test_throws AssertionError _build_table_h!(0, 1, k, 1.0, 1.0)     # empty panel
        @test_throws AssertionError _build_table_h!(0, 1, k, 2.0, 1.0)
    end
end

@testset "_build_table_j!" begin
    @testset "table interpolates J_ν(kr) on its panel, including a panel starting at r = 0" begin
        k = ComplexF64(4.0, 0.3); M = 12
        for (a, b) in ((0.0, 0.25), (1.0, 1.2)), ν in (0, 1, 3)
            tbl = _build_table_j!(ν, k, a, b; M=M)
            @test tbl isa ChebJTable
            @test (tbl.a, tbl.b, tbl.M, tbl.ν) == (a, b, M, ν)
            @test length(tbl.c) == M + 1
            for t in range(-1, 1; length=7)
                r = ((b + a) + (b - a)*t)/2
                @test _cheb_clenshaw(tbl.c, t) ≈ besselj(ν, k*r) rtol=1e-11 atol=1e-14
            end
        end
    end

    @testset "argument errors" begin
        k = ComplexF64(4.0)
        @test_throws AssertionError _build_table_j!(0, k, -0.1, 1.0)
        @test_throws AssertionError _build_table_j!(0, k, 1.0, 1.0)
        @test_throws AssertionError _build_table_j!(0, k, 2.0, 1.0)
    end
end

@testset "plan_h" begin
    @testset "panel structure covers [rmin, rmax] contiguously" begin
        pl = plan_h(1, 1, BS_KS[2], BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M)
        @test pl isa ChebHankelPlanH
        @test (pl.k, pl.ν, pl.κ, pl.rmin, pl.rmax, pl.npanels) == (BS_KS[2], 1, 1, BS_RMIN, BS_RMAX, BS_NP)
        @test pl.dr ≈ (BS_RMAX - BS_RMIN)/BS_NP
        @test pl.invdr ≈ inv(pl.dr)
        @test length(pl.panels) == BS_NP
        @test pl.panels[1].a == BS_RMIN && pl.panels[end].b == BS_RMAX
        @test all(pl.panels[i].b == pl.panels[i+1].a for i in 1:BS_NP-1)
        @test all(P.b > P.a && P.M == BS_M && P.ν == 1 && P.κ == 1 for P in pl.panels)
    end

    @testset "approximates H_ν^(κ)(kr) over the whole interval for real and complex k" begin
        rs = range(BS_RMIN, BS_RMAX; length=61)
        for ν in (0, 1, 2), k in BS_KS
            pl = plan_h(ν, 1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M)
            @test bs_max_error_h(pl, rs) < 1e-10
        end
        @test bs_max_error_h(plan_h(1, 2, BS_KS[1], BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M), rs) < 1e-10   # second kind
    end

    @testset "default panel count and degree" begin
        pl = plan_h(0, 1, BS_KS[1], BS_RMIN, BS_RMAX)
        @test pl.npanels == 64 && all(P.M == 16 for P in pl.panels)
        @test @inferred(plan_h(0, 1, BS_KS[1], BS_RMIN, BS_RMAX; npanels=4, M=8)) isa ChebHankelPlanH
    end

    @testset "argument errors" begin
        @test_throws AssertionError plan_h(0, 1, BS_KS[1], 0.0, 1.0)       # rmin > 0 required
        @test_throws AssertionError plan_h(0, 1, BS_KS[1], -1.0, 1.0)
        @test_throws AssertionError plan_h(0, 1, BS_KS[1], 1.0, 1.0)       # rmax > rmin required
        @test_throws AssertionError plan_h(0, 1, BS_KS[1], 2.0, 1.0)
    end
end

@testset "plan_j" begin
    @testset "panel structure covers [rmin, rmax] contiguously, rmin = 0 allowed" begin
        pl = plan_j(1, BS_KS[2], 0.0, BS_RMAX; npanels=BS_NP, M=BS_M)
        @test pl isa ChebJPlan
        @test (pl.k, pl.ν, pl.rmin, pl.rmax, pl.npanels) == (BS_KS[2], 1, 0.0, BS_RMAX, BS_NP)
        @test pl.dr ≈ BS_RMAX/BS_NP
        @test pl.invdr ≈ inv(pl.dr)
        @test pl.panels[1].a == 0.0 && pl.panels[end].b == BS_RMAX
        @test all(pl.panels[i].b == pl.panels[i+1].a for i in 1:BS_NP-1)
        @test all(P.b > P.a && P.M == BS_M && P.ν == 1 for P in pl.panels)
    end

    @testset "approximates J_ν(kr) over the whole interval for real and complex k" begin
        rs = range(0.0, BS_RMAX; length=61)
        for ν in (0, 1, 2), k in BS_KS
            pl = plan_j(ν, k, 0.0, BS_RMAX; npanels=BS_NP, M=BS_M)
            @test bs_max_error_j(pl, rs) < 1e-10
        end
    end

    @testset "default panel count and degree, type stability" begin
        pl = plan_j(0, BS_KS[1], 0.0, BS_RMAX)
        @test pl.npanels == 64 && all(P.M == 16 for P in pl.panels)
        @test @inferred(plan_j(0, BS_KS[1], 0.0, BS_RMAX; npanels=4, M=8)) isa ChebJPlan
    end

    @testset "argument errors" begin
        @test_throws AssertionError plan_j(0, BS_KS[1], -0.1, 1.0)         # rmin ≥ 0 required
        @test_throws AssertionError plan_j(0, BS_KS[1], 1.0, 1.0)
        @test_throws AssertionError plan_j(0, BS_KS[1], 2.0, 1.0)
    end
end

@testset "_find_panel_uniform and _find_panel" begin
    pl = plan_j(0, ComplexF64(2.0), 1.0, 3.0; npanels=4, M=6)             # panels of width 0.5
    @testset "interior radii map to the containing panel" begin
        @test _find_panel_uniform(pl, 1.0) == 1
        @test _find_panel_uniform(pl, 1.49) == 1
        @test _find_panel_uniform(pl, 1.51) == 2
        @test _find_panel_uniform(pl, 2.2) == 3
        @test _find_panel_uniform(pl, 2.99) == 4
    end
    @testset "radii outside [rmin, rmax] are clamped" begin
        @test _find_panel_uniform(pl, 3.0) == 4
        @test _find_panel_uniform(pl, 100.0) == 4
        @test _find_panel_uniform(pl, 0.2) == 1
    end
    @testset "_find_panel delegates, Hankel and Bessel plans, type stability" begin
        ph = plan_h(0, 1, ComplexF64(2.0), 1.0, 3.0; npanels=4, M=6)
        @test all(_find_panel(pl, r) == _find_panel_uniform(pl, r) for r in (0.5, 1.2, 2.2, 5.0))
        @test all(_find_panel(ph, r) == _find_panel(pl, r) for r in (0.5, 1.2, 2.2, 5.0))
        @test @inferred(_find_panel(pl, 1.7)) isa Int
    end
end

@testset "panel_t" begin
    pl = plan_h(1, 1, BS_KS[1], BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M)
    @testset "radii below rmin report panel zero" begin
        @test panel_t(pl, 0.25) == (Int32(0), 0.0)
        @test panel_t(pl, prevfloat(BS_RMIN)) == (Int32(0), 0.0)
    end
    @testset "t is the affine coordinate of r inside its panel" begin
        for r in range(BS_RMIN, BS_RMAX; length=23)
            p, t = panel_t(pl, r)
            P = pl.panels[p]
            @test 1 <= p <= BS_NP
            @test P.a - 1e-12 <= r <= P.b + 1e-12
            @test abs(t) <= 1 + 1e-12
            @test r ≈ (P.a + P.b)/2 + t*(P.b - P.a)/2 atol=1e-13
        end
        @test panel_t(pl, BS_RMIN) == (Int32(1), -1.0)
        p, t = panel_t(pl, BS_RMAX)                                    # rmax belongs to the last panel, t = +1
        @test p == BS_NP
        @test t ≈ 1.0 atol=1e-14
    end
    @testset "Bessel plans and type stability" begin
        pj = plan_j(0, BS_KS[1], 0.0, BS_RMAX; npanels=BS_NP, M=BS_M)
        @test panel_t(pj, 0.0) == (Int32(1), -1.0)                     # J plans may include r = 0
        @test @inferred(panel_t(pl, 1.3)) isa Tuple{Int32,Float64}
        @test @inferred(panel_t(pj, 1.3)) isa Tuple{Int32,Float64}
    end
end

@testset "panel_and_geom" begin
    pl = plan_h(0, 1, ComplexF64(2.0), 1.0, 3.0; npanels=4, M=6)         # dr = 0.5, centers 1.25, 1.75, 2.25, 2.75
    rvec = [0.1, 1.0, 1.25, 1.8, 3.0]
    pidx, t, invsqrt = panel_and_geom(pl, rvec)

    @test (pidx, t, invsqrt) isa Tuple{Vector{Int32},Vector{Float64},Vector{Float64}}
    @test pidx == Int32[0, 1, 1, 2, 4]                                  # r < rmin -> 0; rmax clamped to the last panel
    @test t[1] == 0.0                                                   # fallback region carries t = 0
    @test t[2] ≈ -1.0 atol=1e-14                                        # left edge of panel 1
    @test t[3] ≈ 0.0 atol=1e-14                                         # center of panel 1
    @test t[4] ≈ 2*(1.8 - 1.75)/0.5 atol=1e-14                          # t = 2(r - center)/dr
    @test t[5] ≈ 1.0 atol=1e-14                                         # right edge of last panel
    @test invsqrt ≈ inv.(sqrt.(rvec))

    @testset "agrees with panel_t pointwise" begin
        rs = collect(range(0.9, 3.0; length=17))
        pidx_s, t_s, _ = panel_and_geom(pl, rs)
        for (i, r) in enumerate(rs)
            p, τ = panel_t(pl, r)
            @test pidx_s[i] == p
            @test t_s[i] ≈ τ atol=1e-13
        end
    end

    @testset "Bessel plans share the panel geometry, type stability" begin
        pj = plan_j(0, ComplexF64(2.0), 1.0, 3.0; npanels=4, M=6)
        @test panel_and_geom(pj, rvec)[1] == pidx
        @test @inferred(panel_and_geom(pl, rvec)) isa Tuple{Vector{Int32},Vector{Float64},Vector{Float64}}
    end
end

@testset "eval_h" begin
    pl_h(ν, k) = plan_h(ν, 1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M)

    @testset "panelized region: Clenshaw evaluation matches besselh" begin
        rs = range(BS_RMIN, BS_RMAX; length=41)
        for ν in (0, 1, 2), k in BS_KS
            @test bs_max_error_h(pl_h(ν, k), rs) < 1e-10
        end
    end

    @testset "radii below rmin with |z| < small-z cutoff use the explicit series" begin
        k = BS_KS[1]; r = 1e-4                                          # |z| = 4e-4
        @test abs(k*r) < hankel_z_chebyshev_cutoff_small_z
        @test eval_h(pl_h(0, k), Int32(0), 0.0, r) ≈ besselh(0, 1, k*r) rtol=1e-12
        @test eval_h(pl_h(1, k), Int32(0), 0.0, r) ≈ besselh(1, 1, k*r) rtol=1e-12
        @test eval_h(pl_h(0, k), Int32(0), 0.0, r) == _small_h0_series(k*r)
        @test eval_h(pl_h(1, k), Int32(0), 0.0, r) == _small_h1_series(k*r)
    end

    @testset "radii below rmin with larger |z| and orders ≥ 2 use direct evaluation" begin
        k = BS_KS[2]; r = 0.1                                           # |z| = 0.3
        @test abs(k*r) > hankel_z_chebyshev_cutoff_small_z
        for ν in (0, 1, 2)
            @test eval_h(pl_h(ν, k), Int32(0), 0.0, r) == besselh(ν, 1, k*r)
        end
        r = 1e-4                                                        # ν = 2 never uses a series, even for tiny |z|
        @test eval_h(pl_h(2, k), Int32(0), 0.0, r) == besselh(2, 1, k*r)
    end

    @testset "second-kind plans use κ = 2 in the fallback" begin
        k = BS_KS[1]; r = 0.1
        pl = plan_h(1, 2, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M)
        @test eval_h(pl, Int32(0), 0.0, r) == besselh(1, 2, k*r)
    end

    @testset "mid-range cutoff fidelity" begin
        # Plan for a small |k| sharing an rmin that was floored for a much larger |k|: part of the panelized
        # interval then has |z| < hankel_z_chebyshev_cutoff although pidx != 0.
        k_big, k_small, rmax = 60.0, 2.0, 5.0
        rmin = hankel_z_chebyshev_cutoff/k_big
        @test k_small*rmin < hankel_z_chebyshev_cutoff
        pl = plan_h(1, 1, ComplexF64(k_small), rmin, rmax; npanels=200, M=8)

        pidx, t = panel_t(pl, rmin)
        @test pidx != 0                                                 # r = rmin lies inside the panelized domain
        @test eval_h(pl, pidx, t, rmin) ≈ besselh(1, 1, k_small*rmin) rtol=1e-9    # interpolation is exact at a node

        # Interior radii with |z| < cutoff: the documented behavior is direct evaluation.
        rs = range(rmin, hankel_z_chebyshev_cutoff/k_small; length=200)
        err = maximum(rs) do r
            p, τ = panel_t(pl, r)
            ref = besselh(1, 1, k_small*r)
            abs(eval_h(pl, p, τ, r) - ref)/abs(ref)
        end
        @test err < 1e-9
    end

    @testset "type stability" begin
        pl = pl_h(1, BS_KS[1])
        @test @inferred(eval_h(pl, Int32(3), 0.1, 1.0)) isa ComplexF64
        @test @inferred(eval_h(pl, Int32(0), 0.0, 0.1)) isa ComplexF64
    end
end

@testset "eval_j" begin
    pl_j(ν, k) = plan_j(ν, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M)

    @testset "panelized region: Clenshaw evaluation matches besselj" begin
        rs = range(BS_RMIN, BS_RMAX; length=41)
        for ν in (0, 1, 3), k in BS_KS
            @test bs_max_error_j(pl_j(ν, k), rs) < 1e-10
        end
    end

    @testset "radii below rmin are evaluated directly" begin
        for ν in (0, 1, 3), r in (0.0, 1e-4, 0.3)
            @test eval_j(pl_j(ν, BS_KS[2]), Int32(0), 0.0, r) == besselj(ν, BS_KS[2]*r)
        end
        @test eval_j(pl_j(0, BS_KS[1]), Int32(0), 0.0, 0.0) == 1.0 + 0.0im    # J₀(0) = 1
        @test eval_j(pl_j(1, BS_KS[1]), Int32(0), 0.0, 0.0) == 0.0 + 0.0im    # J₁(0) = 0
    end

    @testset "plan starting at r = 0 is exact at the origin node" begin
        pl = plan_j(0, BS_KS[1], 0.0, BS_RMAX; npanels=BS_NP, M=BS_M)
        p, t = panel_t(pl, 0.0)
        @test eval_j(pl, p, t, 0.0) ≈ 1.0 atol=1e-12
    end

    @testset "type stability" begin
        pl = pl_j(1, BS_KS[1])
        @test @inferred(eval_j(pl, Int32(3), 0.1, 1.0)) isa ComplexF64
        @test @inferred(eval_j(pl, Int32(0), 0.0, 0.1)) isa ComplexF64
    end
end

@testset "eval_h_multi_ks!" begin
    plans = [plan_h(1, 1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
    out = zeros(ComplexF64, length(BS_KS))

    @testset "panelized radius: one value per wavenumber" begin
        r = 1.234; p, t = panel_t(plans[1], r)
        @test eval_h_multi_ks!(out, plans, r, p, t) === nothing
        @test all(abs(out[m] - besselh(1, 1, BS_KS[m]*r)) < 1e-10 for m in eachindex(BS_KS))
        @test out == [eval_h(plans[m], p, t, r) for m in eachindex(plans)]
    end

    @testset "below-plan radii use the per-plan fallback" begin
        for r in (1e-4, 0.1)                                            # series regime and direct regime
            eval_h_multi_ks!(out, plans, r, Int32(0), 0.0)
            @test out == [eval_h(plans[m], Int32(0), 0.0, r) for m in eachindex(plans)]
            @test all(out[m] ≈ besselh(1, 1, BS_KS[m]*r) for m in eachindex(BS_KS))
        end
    end

    @testset "order-zero plans and second-order plans use the matching fallback" begin
        for ν in (0, 2)
            pls = [plan_h(ν, 1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
            eval_h_multi_ks!(out, pls, 0.1, Int32(0), 0.0)
            @test all(out[m] == besselh(ν, 1, BS_KS[m]*0.1) for m in eachindex(BS_KS))
        end
    end
end

@testset "eval_j_multi_ks!" begin
    plans = [plan_j(0, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
    out = zeros(ComplexF64, length(BS_KS))

    @testset "panelized radius" begin
        r = 1.234; p, t = panel_t(plans[1], r)
        @test eval_j_multi_ks!(out, plans, p, t, r) === nothing
        @test all(abs(out[m] - besselj(0, BS_KS[m]*r)) < 1e-10 for m in eachindex(BS_KS))
        @test out == [eval_j(plans[m], p, t, r) for m in eachindex(plans)]
    end

    @testset "below-plan radius is evaluated directly" begin
        r = 0.2
        eval_j_multi_ks!(out, plans, Int32(0), 0.0, r)
        @test out == [besselj(0, BS_KS[m]*r) for m in eachindex(BS_KS)]
    end
end

@testset "h0_h1_multi_ks_at_r!" begin
    plans0 = [plan_h(0, 1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
    plans1 = [plan_h(1, 1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
    h0 = zeros(ComplexF64, length(BS_KS)); h1 = zeros(ComplexF64, length(BS_KS))

    @testset "panelized radius" begin
        r = 2.1; p, t = panel_t(plans0[1], r)
        @test h0_h1_multi_ks_at_r!(h0, h1, plans0, plans1, p, t, r) === nothing
        @test all(abs(h0[m] - besselh(0, 1, BS_KS[m]*r)) < 1e-10 for m in eachindex(BS_KS))
        @test all(abs(h1[m] - besselh(1, 1, BS_KS[m]*r)) < 1e-10 for m in eachindex(BS_KS))
    end

    @testset "below-plan radii: small-z series and direct evaluation" begin
        for r in (1e-4, 0.1)
            h0_h1_multi_ks_at_r!(h0, h1, plans0, plans1, Int32(0), 0.0, r)
            @test all(h0[m] ≈ besselh(0, 1, BS_KS[m]*r) for m in eachindex(BS_KS))
            @test all(h1[m] ≈ besselh(1, 1, BS_KS[m]*r) for m in eachindex(BS_KS))
        end
    end
end

@testset "h1_j1_multi_ks_at_r!" begin
    plans1 = [plan_h(1, 1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
    plansj1 = [plan_j(1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
    h1 = zeros(ComplexF64, length(BS_KS)); j1 = zeros(ComplexF64, length(BS_KS))

    @testset "panelized radius" begin
        r = 2.1; ph, th = panel_t(plans1[1], r); pj, tj = panel_t(plansj1[1], r)
        @test h1_j1_multi_ks_at_r!(h1, j1, plans1, plansj1, ph, th, pj, tj, r) === nothing
        @test all(abs(h1[m] - besselh(1, 1, BS_KS[m]*r)) < 1e-10 for m in eachindex(BS_KS))
        @test all(abs(j1[m] - besselj(1, BS_KS[m]*r)) < 1e-10 for m in eachindex(BS_KS))
    end

    @testset "below-plan radius" begin
        r = 0.1
        h1_j1_multi_ks_at_r!(h1, j1, plans1, plansj1, Int32(0), 0.0, Int32(0), 0.0, r)
        @test all(h1[m] ≈ besselh(1, 1, BS_KS[m]*r) for m in eachindex(BS_KS))
        @test all(j1[m] ≈ besselj(1, BS_KS[m]*r) for m in eachindex(BS_KS))
    end
end

@testset "h0_h1_j0_j1_multi_ks_at_r!" begin
    mk_h(ν) = [plan_h(ν, 1, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
    mk_j(ν) = [plan_j(ν, k, BS_RMIN, BS_RMAX; npanels=BS_NP, M=BS_M) for k in BS_KS]
    plans0, plans1, plansj0, plansj1 = mk_h(0), mk_h(1), mk_j(0), mk_j(1)
    n = length(BS_KS)
    h0, h1, j0, j1 = (zeros(ComplexF64, n) for _ in 1:4)

    @testset "panelized radius fills all four outputs" begin
        r = 2.1; ph, th = panel_t(plans0[1], r); pj, tj = panel_t(plansj0[1], r)
        @test h0_h1_j0_j1_multi_ks_at_r!(h0, h1, j0, j1, plans0, plans1, plansj0, plansj1, ph, th, pj, tj, r) === nothing
        for m in 1:n
            @test abs(h0[m] - besselh(0, 1, BS_KS[m]*r)) < 1e-10
            @test abs(h1[m] - besselh(1, 1, BS_KS[m]*r)) < 1e-10
            @test abs(j0[m] - besselj(0, BS_KS[m]*r)) < 1e-10
            @test abs(j1[m] - besselj(1, BS_KS[m]*r)) < 1e-10
        end
    end

    @testset "below-plan radius fills all four outputs" begin
        r = 0.1
        h0_h1_j0_j1_multi_ks_at_r!(h0, h1, j0, j1, plans0, plans1, plansj0, plansj1, Int32(0), 0.0, Int32(0), 0.0, r)
        for m in 1:n
            @test h0[m] ≈ besselh(0, 1, BS_KS[m]*r)
            @test h1[m] ≈ besselh(1, 1, BS_KS[m]*r)
            @test j0[m] ≈ besselj(0, BS_KS[m]*r)
            @test j1[m] ≈ besselj(1, BS_KS[m]*r)
        end
    end
end

@testset "ChebRadialLookupCache" begin
    # Synthetic distance matrix: only G.R is read by the constructor.
    N = 34
    R = [i == j ? 1.0 : 0.6 + 0.05*abs(i - j) for i in 1:N, j in 1:N]  # off-diagonal distances in [0.65, 2.25]
    Z = zeros(N, N)
    G = BoundaryGeomCache(R, Z, Z, Z, ones(N), zeros(N), Float64[])
    ph = plan_h(1, 1, BS_KS[1], 0.5, 2.5; npanels=BS_NP, M=BS_M)
    pj = plan_j(1, BS_KS[1], 0.8, 2.5; npanels=BS_NP, M=BS_M)       # rmin above the shortest distances

    for mt in (true, false)
        @testset "multithreaded = $mt" begin
            cache = ChebRadialLookupCache(G, ph, pj; multithreaded=mt)
            @test cache isa ChebRadialLookupCache
            @test size(cache.pidx_h) == size(cache.t_h) == size(cache.pidx_j) == size(cache.t_j) == (N, N)
            @test eltype(cache.pidx_h) == Int32 && eltype(cache.t_h) == Float64
            ok = true
            for j in 1:N, i in 1:N
                i == j && continue
                r = R[i, j]
                ok &= (cache.pidx_h[i, j], cache.t_h[i, j]) == panel_t(ph, r)
                ok &= (cache.pidx_j[i, j], cache.t_j[i, j]) == panel_t(pj, r)
            end
            @test ok
            @test all(cache.pidx_h[i, j] == cache.pidx_h[j, i] && cache.t_h[i, j] == cache.t_h[j, i] for i in 1:N for j in i+1:N)   # symmetric off the diagonal (diagonal is left uninitialized)
            @test any(==(Int32(0)), cache.pidx_j) && all(!=(Int32(0)), (cache.pidx_h[i, j] for i in 1:N, j in 1:N if i != j))
        end
    end
end
