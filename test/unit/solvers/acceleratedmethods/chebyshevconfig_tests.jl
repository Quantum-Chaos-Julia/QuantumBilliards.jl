# Unit tests for
#   src/solvers/acceleratedmethods/chebyshevconfig.jl   (ChebyshevConfig struct and constructor)
#   src/solvers/chebyshev/optimalpanelization.jl        (tune_dlp_cheb_plans, tune_cfie_cheb_plans: public tuning entry
#                                                        points consuming a ChebyshevConfig)
# Constructor defaults are the ones documented in the docstring. The tuning entry points are checked against direct
# SpecialFunctions.besselh / besselj evaluation on the configured validation grid. Systems are tiny: two wavenumbers
# and a short radial interval; no BIM matrices or eigen-solves.

using SpecialFunctions
using QuantumBilliards: ChebyshevConfig, ChebHankelPlanH, ChebJPlan, panel_t, eval_h, eval_j,
                        tune_dlp_cheb_plans, tune_cfie_cheb_plans

const CC_KS = ComplexF64[2.0, 3.0 + 0.2im]
const CC_RMIN, CC_RMAX = 0.4, 2.0
const CC_TOL = 1e-8
# Unresolved starting parameters so the tuner has to refine; validation grid matches sampling_points.
cc_cfg(; kwargs...) = ChebyshevConfig(; n_panels_h=4, M_h=4, n_panels_j=2, M_j=4, tol=CC_TOL, sampling_points=101, kwargs...)
const CC_RS = range(CC_RMIN, CC_RMAX; length=101)

cc_err_h(pl::ChebHankelPlanH) = maximum(CC_RS) do r
    p, t = panel_t(pl, r)
    abs(eval_h(pl, p, t, r) - besselh(pl.ν, pl.κ, pl.k*r))
end
cc_err_j(pl::ChebJPlan) = maximum(CC_RS) do r
    p, t = panel_t(pl, r)
    abs(eval_j(pl, p, t, r) - besselj(pl.ν, pl.k*r))
end

@testset "ChebyshevConfig defaults and field storage" begin
    @testset "default constructor uses the documented values" begin
        cfg = ChebyshevConfig()
        @test cfg isa ChebyshevConfig{Float64}
        @test cfg.n_panels_h == 10000 && cfg.M_h == 5
        @test cfg.n_panels_j == 5000 && cfg.M_j == 5
        @test cfg.tol == 1e-12 && cfg.max_iter == 20 && cfg.sampling_points == 10_000
        @test cfg.grow_panels == 1.5 && cfg.grow_M == 2
        @test cfg.param_strategy === :global
    end

    @testset "keyword overrides are stored in the matching fields" begin
        cfg = ChebyshevConfig(; n_panels_h=11, M_h=6, n_panels_j=12, M_j=7, tol=1e-6, max_iter=3,
                              sampling_points=50, grow_panels=2.0, grow_M=4, param_strategy=:segment)
        @test (cfg.n_panels_h, cfg.M_h, cfg.n_panels_j, cfg.M_j) == (11, 6, 12, 7)
        @test (cfg.tol, cfg.max_iter, cfg.sampling_points, cfg.grow_panels, cfg.grow_M) == (1e-6, 3, 50, 2.0, 4)
        @test cfg.param_strategy === :segment
        @test @inferred(ChebyshevConfig(Float64; tol=1e-6)) isa ChebyshevConfig{Float64}
    end

    @testset "numeric type parameter controls tol and grow_panels storage" begin
        cfg32 = ChebyshevConfig(Float32; tol=1e-6, grow_panels=2)
        @test cfg32 isa ChebyshevConfig{Float32}
        @test typeof(cfg32.tol) == Float32 && cfg32.tol == 1f-6
        @test typeof(cfg32.grow_panels) == Float32 && cfg32.grow_panels == 2f0
        @test cfg32.n_panels_h isa Int && cfg32.M_h isa Int && cfg32.max_iter isa Int       # counts stay Int
        cfg_int = ChebyshevConfig(Float64; tol=1, grow_panels=2)                            # any Real is accepted and converted
        @test cfg_int.tol === 1.0 && cfg_int.grow_panels === 2.0
    end

    @testset "all three strategies are accepted" begin
        for s in (:global, :segment, :manual)
            @test ChebyshevConfig(; param_strategy=s).param_strategy === s
        end
    end

    @testset "config is immutable" begin
        cfg = ChebyshevConfig()
        @test_throws ErrorException (cfg.M_h = 9)
    end
end

@testset "ChebyshevConfig argument validation" begin
    @testset "unknown param_strategy is rejected" begin
        @test_throws ArgumentError ChebyshevConfig(; param_strategy=:auto)
        @test_throws ArgumentError ChebyshevConfig(; param_strategy=:Global)
        @test_throws ArgumentError ChebyshevConfig(Float32; param_strategy=:none)
        err = try ChebyshevConfig(; param_strategy=:auto); nothing catch e; e end
        @test occursin("param_strategy", sprint(showerror, err)) && occursin("auto", sprint(showerror, err))
    end

    @testset "keyword type errors" begin
        @test_throws TypeError ChebyshevConfig(; n_panels_h=1.5)          # typed keywords
        @test_throws TypeError ChebyshevConfig(; M_h=2.5)
        @test_throws TypeError ChebyshevConfig(; max_iter="20")
        @test_throws TypeError ChebyshevConfig(; param_strategy="global")
        @test_throws MethodError ChebyshevConfig(; unknown_option=1)
    end

    @testset "non-real numeric type parameter is rejected" begin
        @test_throws MethodError ChebyshevConfig(ComplexF64)
        @test_throws MethodError ChebyshevConfig(String)
    end

    @testset "nonpositive resolution or tolerance is rejected" begin
        # SOURCE GAP (ChebyshevConfig constructor): only param_strategy is validated; zero panel counts or degrees
        # (division by zero in _breaks_uniform / _chebfit!) and a nonpositive tol are silently accepted.
        rejected(; kwargs...) = try ChebyshevConfig(; kwargs...); false catch e; e isa ArgumentError end
        @test_broken rejected(n_panels_h=0)
        @test_broken rejected(n_panels_j=0)
        @test_broken rejected(M_h=0)
        @test_broken rejected(tol=-1e-8)
    end
end

@testset "tune_dlp_cheb_plans" begin
    @testset "returns H₁ and J₁ plans that meet tol against direct evaluation" begin
        cfg = cc_cfg()
        ph1, pj1, cfg_used = tune_dlp_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cfg)
        @test ph1 isa Vector{ChebHankelPlanH} && pj1 isa Vector{ChebJPlan}
        @test length(ph1) == length(CC_KS) && length(pj1) == length(CC_KS)
        @test [pl.k for pl in ph1] == CC_KS && [pl.k for pl in pj1] == CC_KS
        @test all(pl.ν == 1 && pl.κ == 1 for pl in ph1) && all(pl.ν == 1 for pl in pj1)
        @test all(cc_err_h(pl) < CC_TOL for pl in ph1)
        @test all(cc_err_j(pl) < CC_TOL for pl in pj1)
    end

    @testset "cfg_used holds the tuned resolution and preserves the other settings" begin
        cfg = cc_cfg()
        ph1, pj1, cu = tune_dlp_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cfg)
        @test cu isa ChebyshevConfig{Float64}
        @test cu.n_panels_h >= cfg.n_panels_h && cu.M_h >= cfg.M_h
        @test cu.n_panels_j >= cfg.n_panels_j && cu.M_j >= cfg.M_j
        @test all(pl.npanels == cu.n_panels_h && all(P.M == cu.M_h for P in pl.panels) for pl in ph1)
        @test all(pl.npanels == cu.n_panels_j && all(P.M == cu.M_j for P in pl.panels) for pl in pj1)
        @test (cu.tol, cu.max_iter, cu.sampling_points, cu.grow_panels, cu.grow_M, cu.param_strategy) ==
              (cfg.tol, cfg.max_iter, cfg.sampling_points, cfg.grow_panels, cfg.grow_M, cfg.param_strategy)
    end

    @testset ":manual strategy uses the supplied resolution untouched" begin
        cfg = cc_cfg(param_strategy=:manual)
        ph1, pj1, cu = tune_dlp_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cfg)
        @test cu == cfg
        @test all(pl.npanels == 4 && all(P.M == 4 for P in pl.panels) for pl in ph1)
        @test all(pl.npanels == 2 && all(P.M == 4 for P in pl.panels) for pl in pj1)
    end

    @testset "Float32 configuration keeps its numeric type" begin
        cfg = ChebyshevConfig(Float32; n_panels_h=4, M_h=6, n_panels_j=2, M_j=6, param_strategy=:manual)
        _, _, cu = tune_dlp_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cfg)
        @test cu isa ChebyshevConfig{Float32}
    end

    @testset "argument errors" begin
        @test_throws ArgumentError tune_dlp_cheb_plans(CC_RMIN, CC_RMAX, ComplexF64[], cc_cfg())
    end
end

@testset "tune_cfie_cheb_plans" begin
    @testset "returns H₀, H₁, J₀, J₁ plans that meet tol against direct evaluation" begin
        cfg = cc_cfg()
        ph0, ph1, pj0, pj1, cfg_used = tune_cfie_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cfg)
        @test ph0 isa Vector{ChebHankelPlanH} && ph1 isa Vector{ChebHankelPlanH}
        @test pj0 isa Vector{ChebJPlan} && pj1 isa Vector{ChebJPlan}
        @test all(length(v) == length(CC_KS) for v in (ph0, ph1, pj0, pj1))
        @test all(pl.ν == 0 for pl in ph0) && all(pl.ν == 1 for pl in ph1)
        @test all(pl.ν == 0 for pl in pj0) && all(pl.ν == 1 for pl in pj1)
        @test all(cc_err_h(pl) < CC_TOL for pl in vcat(ph0, ph1))
        @test all(cc_err_j(pl) < CC_TOL for pl in vcat(pj0, pj1))
        @test cfg_used isa ChebyshevConfig{Float64}
    end

    @testset "orders of one family share the tuned resolution" begin
        ph0, ph1, pj0, pj1, cu = tune_cfie_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cc_cfg())
        @test all(pl.npanels == cu.n_panels_h && all(P.M == cu.M_h for P in pl.panels) for pl in vcat(ph0, ph1))
        @test all(pl.npanels == cu.n_panels_j && all(P.M == cu.M_j for P in pl.panels) for pl in vcat(pj0, pj1))
    end

    @testset "CFIE resolution is at least that required by the DLP orders" begin
        # CFIE tunes (H₀, H₁) jointly, DLP only H₁, so the CFIE family error dominates the DLP one.
        cfg = cc_cfg()
        _, _, cu_dlp = tune_dlp_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cfg)
        _, _, _, _, cu_cfie = tune_cfie_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cfg)
        @test cu_cfie.n_panels_h >= cu_dlp.n_panels_h && cu_cfie.M_h >= cu_dlp.M_h     # growth path is monotone in both
    end

    @testset ":manual strategy uses the supplied resolution untouched" begin
        cfg = cc_cfg(param_strategy=:manual)
        ph0, ph1, pj0, pj1, cu = tune_cfie_cheb_plans(CC_RMIN, CC_RMAX, CC_KS, cfg)
        @test cu == cfg
        @test all(pl.npanels == 4 && all(P.M == 4 for P in pl.panels) for pl in vcat(ph0, ph1))
        @test all(pl.npanels == 2 && all(P.M == 4 for P in pl.panels) for pl in vcat(pj0, pj1))
    end

    @testset "argument errors" begin
        @test_throws ArgumentError tune_cfie_cheb_plans(CC_RMIN, CC_RMAX, ComplexF64[], cc_cfg())
    end
end
