# Unit tests for
#   src/solvers/chebyshev/optimalpanelization.jl
#     _cheb_h_error, _cheb_j_error, _cheb_h_family_error, _cheb_j_family_error,
#     _cheb_h_batch_errors!, _cheb_j_batch_errors!, _cheb_grow_params,
#     _tune_h_params, _tune_j_params, _build_h_plans, _build_j_plans, _tune_cheb_plans
# (_cheb_geom_rminmax is covered in boundarygeomcache_tests.jl; tune_dlp_cheb_plans / tune_cfie_cheb_plans are covered
# in acceleratedmethods/chebyshevconfig_tests.jl.)
# Reference values are direct SpecialFunctions.besselh / besselj evaluations. Systems are tiny: two wavenumbers,
# an interval with |k| r >= 0.2, and at most a few hundred validation radii. No BIM matrices or eigen-solves.

using SpecialFunctions
using QuantumBilliards: ChebyshevConfig, ChebHankelPlanH, ChebJPlan, plan_h, plan_j, panel_t, eval_h, eval_j,
                        _cheb_h_error, _cheb_j_error, _cheb_h_family_error, _cheb_j_family_error,
                        _cheb_h_batch_errors!, _cheb_j_batch_errors!, _cheb_grow_params,
                        _tune_h_params, _tune_j_params, _build_h_plans, _build_j_plans, _tune_cheb_plans

const OP_KS = ComplexF64[2.0, 3.0 + 0.2im]            # |k| rmin >= 0.2 for rmin = OP_RMIN
const OP_RMIN, OP_RMAX = 0.4, 2.0
const OP_RS = range(OP_RMIN, OP_RMAX; length=101)

# Independent reference error of a plan on the grid rs: max |plan(r) - direct(r)|.
function op_err_h(pl::ChebHankelPlanH, rs)
    maximum(rs) do r
        p, t = panel_t(pl, r)
        abs(eval_h(pl, p, t, r) - besselh(pl.ν, pl.κ, pl.k*r))
    end
end
function op_err_j(pl::ChebJPlan, rs)
    maximum(rs) do r
        p, t = panel_t(pl, r)
        abs(eval_j(pl, p, t, r) - besselj(pl.ν, pl.k*r))
    end
end

# Resolution too low to reach OP_TOL initially, so the tuner must refine.
op_cfg(; kwargs...) = ChebyshevConfig(; n_panels_h=4, M_h=4, n_panels_j=2, M_j=4, tol=1e-8, sampling_points=101, kwargs...)
const OP_TOL = 1e-8

@testset "_cheb_h_error" begin
    @testset "matches an independent maximum over the validation grid" begin
        pl = plan_h(1, 1, OP_KS[1], OP_RMIN, OP_RMAX; npanels=4, M=6)
        @test _cheb_h_error(pl, OP_RS) ≈ op_err_h(pl, OP_RS) rtol=1e-12
        @test _cheb_h_error(pl, collect(OP_RS)) ≈ op_err_h(pl, OP_RS) rtol=1e-12          # Vector grid
        @test _cheb_h_error(pl, [OP_RS[17]]) ≈ op_err_h(pl, [OP_RS[17]]) rtol=1e-12       # single radius
        @test @inferred(_cheb_h_error(pl, OP_RS)) isa Float64
    end

    @testset "error is tiny for a resolved plan, large for an unresolved one" begin
        fine = plan_h(1, 1, OP_KS[2], OP_RMIN, OP_RMAX; npanels=16, M=16)
        coarse = plan_h(1, 1, OP_KS[2], OP_RMIN, OP_RMAX; npanels=1, M=4)
        @test _cheb_h_error(fine, OP_RS) < 1e-10
        @test _cheb_h_error(coarse, OP_RS) > 1e-3
    end

    @testset "panel refinement reduces the error" begin
        errs = [_cheb_h_error(plan_h(0, 1, OP_KS[1], OP_RMIN, OP_RMAX; npanels=np, M=6), OP_RS) for np in (2, 4, 8)]
        @test errs[1] > errs[2] > errs[3]
    end

    @testset "error vanishes on the Lobatto nodes of the plan" begin
        pl = plan_h(0, 1, OP_KS[1], OP_RMIN, OP_RMAX; npanels=3, M=5)
        nodes = [((P.b + P.a) + (P.b - P.a)*cospi(j/5))/2 for P in pl.panels for j in 0:5]
        @test _cheb_h_error(pl, nodes) < 1e-12
    end
end

@testset "_cheb_j_error" begin
    @testset "matches an independent maximum over the validation grid" begin
        pl = plan_j(1, OP_KS[1], OP_RMIN, OP_RMAX; npanels=2, M=5)
        @test _cheb_j_error(pl, OP_RS) ≈ op_err_j(pl, OP_RS) rtol=1e-12
        @test _cheb_j_error(pl, collect(OP_RS)) ≈ op_err_j(pl, OP_RS) rtol=1e-12
        @test _cheb_j_error(pl, [OP_RS[17]]) ≈ op_err_j(pl, [OP_RS[17]]) rtol=1e-12
        @test @inferred(_cheb_j_error(pl, OP_RS)) isa Float64
    end

    @testset "error is tiny for a resolved plan, large for an unresolved one" begin
        fine = plan_j(0, OP_KS[2], OP_RMIN, OP_RMAX; npanels=8, M=14)
        coarse = plan_j(0, OP_KS[2], OP_RMIN, OP_RMAX; npanels=1, M=3)
        @test _cheb_j_error(fine, OP_RS) < 1e-10
        @test _cheb_j_error(coarse, OP_RS) > 1e-3
    end

    @testset "panel refinement reduces the error" begin
        errs = [_cheb_j_error(plan_j(0, OP_KS[1], OP_RMIN, OP_RMAX; npanels=np, M=5), OP_RS) for np in (1, 2, 4)]
        @test errs[1] > errs[2] > errs[3]
    end
end

@testset "_cheb_h_family_error" begin
    ks = OP_KS[1]; np, M = 4, 6
    e0 = _cheb_h_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, (0,))
    e1 = _cheb_h_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, (1,))
    @test e0 ≈ op_err_h(plan_h(0, 1, ks, OP_RMIN, OP_RMAX; npanels=np, M=M), OP_RS) rtol=1e-12
    @test e1 ≈ op_err_h(plan_h(1, 1, ks, OP_RMIN, OP_RMAX; npanels=np, M=M), OP_RS) rtol=1e-12
    @test _cheb_h_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, (0, 1)) ≈ max(e0, e1) rtol=1e-12   # worst order
    @test _cheb_h_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, ()) == 0.0                         # no orders
    @test @inferred(_cheb_h_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, (0, 1))) isa Float64
end

@testset "_cheb_j_family_error" begin
    ks = OP_KS[1]; np, M = 2, 5
    e0 = _cheb_j_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, (0,))
    e1 = _cheb_j_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, (1,))
    @test e0 ≈ op_err_j(plan_j(0, ks, OP_RMIN, OP_RMAX; npanels=np, M=M), OP_RS) rtol=1e-12
    @test e1 ≈ op_err_j(plan_j(1, ks, OP_RMIN, OP_RMAX; npanels=np, M=M), OP_RS) rtol=1e-12
    @test _cheb_j_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, (0, 1)) ≈ max(e0, e1) rtol=1e-12
    @test _cheb_j_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, ()) == 0.0
    @test @inferred(_cheb_j_family_error(ks, OP_RMIN, OP_RMAX, np, M, OP_RS, (0, 1))) isa Float64
end

@testset "_cheb_h_batch_errors!" begin
    errors = zeros(length(OP_KS))
    @test _cheb_h_batch_errors!(errors, OP_KS, OP_RMIN, OP_RMAX, 4, 6, OP_RS, (0, 1)) === errors
    for j in eachindex(OP_KS)
        @test errors[j] ≈ _cheb_h_family_error(OP_KS[j], OP_RMIN, OP_RMAX, 4, 6, OP_RS, (0, 1)) rtol=1e-12
    end
    @test all(>(0), errors)
end

@testset "_cheb_j_batch_errors!" begin
    errors = zeros(length(OP_KS))
    @test _cheb_j_batch_errors!(errors, OP_KS, OP_RMIN, OP_RMAX, 2, 5, OP_RS, (0, 1)) === errors
    for j in eachindex(OP_KS)
        @test errors[j] ≈ _cheb_j_family_error(OP_KS[j], OP_RMIN, OP_RMAX, 2, 5, OP_RS, (0, 1)) rtol=1e-12
    end
    @test all(>(0), errors)
end

@testset "_cheb_grow_params" begin
    cfg = op_cfg()                                                   # grow_panels = 1.5, grow_M = 2
    @testset "panel refinement except every fifth step" begin
        @test _cheb_grow_params(4, 4, 1, cfg) == (6, 4)
        @test _cheb_grow_params(5, 4, 2, cfg) == (8, 4)              # ceil(1.5 * 5)
        @test _cheb_grow_params(7, 4, 4, cfg) == (11, 4)
        @test _cheb_grow_params(100, 8, 6, cfg) == (150, 8)
    end
    @testset "polynomial-degree increase on every fifth step" begin
        @test _cheb_grow_params(4, 4, 5, cfg) == (4, 6)
        @test _cheb_grow_params(9, 6, 10, cfg) == (9, 8)
    end
    @testset "respects configured growth factors, type stability" begin
        cfg2 = ChebyshevConfig(; grow_panels=2.0, grow_M=3)
        @test _cheb_grow_params(4, 4, 1, cfg2) == (8, 4)
        @test _cheb_grow_params(4, 4, 5, cfg2) == (4, 7)
        @test @inferred(_cheb_grow_params(4, 4, 1, cfg)) isa Tuple{Int,Int}
    end
end

@testset "_tune_h_params" begin
    @testset "refines an unresolved start until every wavenumber meets tol" begin
        cfg = op_cfg()
        nh, Mh = _tune_h_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg, (0, 1))
        @test nh >= cfg.n_panels_h && Mh >= cfg.M_h
        @test (nh, Mh) != (cfg.n_panels_h, cfg.M_h)
        for k in OP_KS, ν in (0, 1)
            @test op_err_h(plan_h(ν, 1, k, OP_RMIN, OP_RMAX; npanels=nh, M=Mh), OP_RS) < OP_TOL
        end
    end

    @testset "an already resolved start is returned unchanged" begin
        cfg = op_cfg(); cfg_good = ChebyshevConfig(; n_panels_h=16, M_h=16, tol=OP_TOL, sampling_points=101)
        @test _tune_h_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg_good, (0, 1)) == (16, 16)
    end

    @testset "no orders and :manual strategy skip the tuning" begin
        cfg = op_cfg()
        @test _tune_h_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg, ()) == (4, 4)
        cfg_manual = op_cfg(param_strategy=:manual)
        @test _tune_h_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg_manual, (0, 1)) == (4, 4)    # unresolved but untouched
    end

    @testset ":segment strategy tunes like :global" begin
        @test _tune_h_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, op_cfg(param_strategy=:segment), (1,)) ==
              _tune_h_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, op_cfg(param_strategy=:global), (1,))
    end

    @testset "exhausting max_iter warns and returns the grown parameters" begin
        cfg = ChebyshevConfig(; n_panels_h=2, M_h=4, tol=1e-14, max_iter=1, sampling_points=101)
        local nh, Mh
        @test_logs (:warn, "Hankel Chebyshev tuning did not reach the requested tolerance") begin
            nh, Mh = _tune_h_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg, (1,))
        end
        @test (nh, Mh) == _cheb_grow_params(2, 4, 1, cfg)            # exactly one growth step was applied
    end
end

@testset "_tune_j_params" begin
    @testset "refines an unresolved start until every wavenumber meets tol" begin
        cfg = op_cfg()
        nj, Mj = _tune_j_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg, (0, 1))
        @test nj >= cfg.n_panels_j && Mj >= cfg.M_j
        @test (nj, Mj) != (cfg.n_panels_j, cfg.M_j)
        for k in OP_KS, ν in (0, 1)
            @test op_err_j(plan_j(ν, k, OP_RMIN, OP_RMAX; npanels=nj, M=Mj), OP_RS) < OP_TOL
        end
    end

    @testset "an already resolved start is returned unchanged" begin
        cfg_good = ChebyshevConfig(; n_panels_j=8, M_j=14, tol=OP_TOL, sampling_points=101)
        @test _tune_j_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg_good, (0, 1)) == (8, 14)
    end

    @testset "no orders and :manual strategy skip the tuning" begin
        @test _tune_j_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, op_cfg(), ()) == (2, 4)
        @test _tune_j_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, op_cfg(param_strategy=:manual), (0, 1)) == (2, 4)
    end

    @testset "exhausting max_iter warns and returns the grown parameters" begin
        cfg = ChebyshevConfig(; n_panels_j=1, M_j=3, tol=1e-14, max_iter=1, sampling_points=101)
        local nj, Mj
        @test_logs (:warn, "Bessel-J Chebyshev tuning did not reach the requested tolerance") begin
            nj, Mj = _tune_j_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg, (0,))
        end
        @test (nj, Mj) == _cheb_grow_params(1, 3, 1, cfg)
    end
end

@testset "_build_h_plans" begin
    for ν in (0, 1)
        plans = _build_h_plans(ν, OP_KS, OP_RMIN, OP_RMAX, 8, 12)
        @test plans isa Vector{ChebHankelPlanH}
        @test length(plans) == length(OP_KS)
        @test [pl.k for pl in plans] == OP_KS                        # order of wavenumbers preserved
        @test all(pl.ν == ν && pl.κ == 1 && pl.npanels == 8 && pl.rmin == OP_RMIN && pl.rmax == OP_RMAX for pl in plans)
        @test all(P.M == 12 for pl in plans for P in pl.panels)
        @test all(op_err_h(pl, OP_RS) < 1e-9 for pl in plans)
    end
end

@testset "_build_j_plans" begin
    for ν in (0, 1)
        plans = _build_j_plans(ν, OP_KS, OP_RMIN, OP_RMAX, 6, 12)
        @test plans isa Vector{ChebJPlan}
        @test length(plans) == length(OP_KS)
        @test [pl.k for pl in plans] == OP_KS
        @test all(pl.ν == ν && pl.npanels == 6 && pl.rmin == OP_RMIN && pl.rmax == OP_RMAX for pl in plans)
        @test all(P.M == 12 for pl in plans for P in pl.panels)
        @test all(op_err_j(pl, OP_RS) < 1e-9 for pl in plans)
    end
end

@testset "_tune_cheb_plans" begin
    @testset "requested orders are built, others are nothing" begin
        hp, jp, cfg_used = _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, op_cfg(); h_orders=(1,), j_orders=(0,))
        @test hp.h0 === nothing && jp.j1 === nothing
        @test hp.h1 isa Vector{ChebHankelPlanH} && jp.j0 isa Vector{ChebJPlan}
        @test all(pl.ν == 1 for pl in hp.h1) && all(pl.ν == 0 for pl in jp.j0)
        @test keys(hp) == (:h0, :h1) && keys(jp) == (:j0, :j1)
        @test cfg_used isa ChebyshevConfig{Float64}

        hp, jp, _ = _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, op_cfg(param_strategy=:manual))     # no orders at all
        @test hp.h0 === nothing && hp.h1 === nothing && jp.j0 === nothing && jp.j1 === nothing
    end

    @testset "tuned plans meet tol on the validation grid and cover [rmin, rmax]" begin
        cfg = op_cfg()
        hp, jp, cfg_used = _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, cfg; h_orders=(0, 1), j_orders=(0, 1))
        for pl in vcat(hp.h0, hp.h1)
            @test op_err_h(pl, OP_RS) < OP_TOL
            @test pl.panels[1].a == OP_RMIN && pl.panels[end].b == OP_RMAX
            breaks = vcat([P.a for P in pl.panels], pl.panels[end].b)
            @test issorted(breaks; lt=<=)                            # strictly increasing breakpoints
            @test pl.npanels == cfg_used.n_panels_h && all(P.M == cfg_used.M_h for P in pl.panels)
        end
        for pl in vcat(jp.j0, jp.j1)
            @test op_err_j(pl, OP_RS) < OP_TOL
            @test pl.panels[1].a == OP_RMIN && pl.panels[end].b == OP_RMAX
            @test pl.npanels == cfg_used.n_panels_j && all(P.M == cfg_used.M_j for P in pl.panels)
        end
        @test hp.h0 !== hp.h1                                        # distinct plan sets per order
    end

    @testset "cfg_used carries the tuned parameters and every other setting unchanged" begin
        cfg = op_cfg(max_iter=12, grow_panels=1.5, grow_M=2)
        _, _, cu = _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, cfg; h_orders=(1,), j_orders=(1,))
        @test (cu.n_panels_h, cu.M_h) == _tune_h_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg, (1,))
        @test (cu.n_panels_j, cu.M_j) == _tune_j_params(OP_RMIN, OP_RMAX, OP_KS, OP_RS, cfg, (1,))
        @test (cu.tol, cu.max_iter, cu.sampling_points, cu.grow_panels, cu.grow_M, cu.param_strategy) ==
              (cfg.tol, cfg.max_iter, cfg.sampling_points, cfg.grow_panels, cfg.grow_M, cfg.param_strategy)
    end

    @testset ":manual strategy builds plans directly from cfg without tuning" begin
        cfg = op_cfg(param_strategy=:manual)
        hp, jp, cu = _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, cfg; h_orders=(0,), j_orders=(1,))
        @test cu == cfg
        @test all(pl.npanels == 4 && all(P.M == 4 for P in pl.panels) for pl in hp.h0)
        @test all(pl.npanels == 2 && all(P.M == 4 for P in pl.panels) for pl in jp.j1)
        @test op_err_h(hp.h0[1], OP_RS) > OP_TOL                     # unresolved, left as requested
    end

    @testset "argument errors" begin
        cfg = op_cfg()
        @test_throws ArgumentError _tune_cheb_plans(OP_RMIN, OP_RMAX, ComplexF64[], cfg; h_orders=(1,))
        @test_throws ArgumentError _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, cfg; h_orders=(2,))
        @test_throws ArgumentError _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, cfg; h_orders=(0, 2))
        @test_throws ArgumentError _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, cfg; j_orders=(-1,))
        @test_throws ArgumentError _tune_cheb_plans(OP_RMIN, OP_RMAX, OP_KS, cfg; j_orders=(0, 3))
    end
end
