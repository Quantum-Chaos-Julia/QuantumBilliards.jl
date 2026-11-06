# Unit tests for src/solvers/computationplan.jl:
#   expected_states, WindowPlan, ComputationPlan and the Base.show methods of ComputationPlan.
# `with_params` is covered in solvers/solverconstructors_tests.jl. Plans are built by hand; no eigen solves.
# Helper definitions use the prefix `cp_`.

using QuantumBilliards: expected_states, WindowPlan, ComputationPlan, MemoryEstimate

cp_show(plan) = sprint(show, MIME("text/plain"), plan)

@testset "expected_states" begin
    rect = RectangleBilliard(1.0, 0.7)
    stadium = StadiumBilliard(0.5)

    @testset "equals state_at_k difference" begin
        for b in (rect, stadium)
            @test expected_states(b, 20.0, 40.0) ≈ state_at_k(40.0, b) - state_at_k(20.0, b)
        end
        @test expected_states(rect, 20.0, 40.0) isa Float64
        @test expected_states(rect, 20, 40) ≈ expected_states(rect, 20.0, 40.0)
    end

    @testset "clamped at zero when k2 < k1" begin
        @test expected_states(rect, 40.0, 20.0) == 0.0
        @test expected_states(stadium, 40.0, 20.0) == 0.0
        @test expected_states(rect, 30.0, 30.0) == 0.0
    end

    @testset "fundamental_only on a symmetric billiard" begin
        full = expected_states(stadium, 20.0, 40.0)
        fund = expected_states(stadium, 20.0, 40.0; fundamental_only=true)
        @test fund ≈ state_at_k(40.0, stadium; fundamental_only=true) - state_at_k(20.0, stadium; fundamental_only=true)
        @test fund > 0
        @test fund < full
        @test expected_states(stadium, 40.0, 20.0; fundamental_only=true) == 0.0
    end
end

@testset "WindowPlan" begin
    w = WindowPlan(10.0, 0.5, 3.2, 40, 80)
    @test w isa WindowPlan{Float64}
    @test (w.kcenter, w.halfwidth, w.expected_states, w.dim, w.matrix_dim) == (10.0, 0.5, 3.2, 40, 80)
    @test fieldtype(typeof(w), :kcenter) === Float64
    @test fieldtype(typeof(w), :halfwidth) === Float64
    @test fieldtype(typeof(w), :expected_states) === Float64
    @test fieldtype(typeof(w), :dim) === Int
    @test fieldtype(typeof(w), :matrix_dim) === Int
    @test isconcretetype(typeof(w))
    @test @inferred(WindowPlan(10.0, 0.5, 3.2, 40, 80)) isa WindowPlan{Float64}
    @test WindowPlan(10f0, 0.5f0, 3.2, 40, 80) isa WindowPlan{Float32}
    @test isconcretetype(WindowPlan{Float32})
end

@testset "ComputationPlan" begin
    solver = BeynSolver(DoubleLayerPotentialSolver(10.0, CircleBilliard(1.0)))
    mem = MemoryEstimate(2.0^30, 2.0^29)
    mkplan(windows, notes=String[]) = ComputationPlan(solver, 10.0, 12.0, 7.25, 3.0, 4.0, windows, mem, 4, notes)

    @testset "constructor and field types" begin
        ws = [WindowPlan(10.5, 0.5, 2.0, 40, 80), WindowPlan(11.5, 0.5, 3.0, 50, 100)]
        plan = mkplan(ws)
        @test plan isa ComputationPlan{Float64,typeof(solver)}
        @test isconcretetype(typeof(plan))
        @test plan.solver === solver
        @test (plan.k1, plan.k2) == (10.0, 12.0)
        @test plan.expected_total_states == 7.25
        @test (plan.density_k1, plan.density_k2) == (3.0, 4.0)
        @test plan.windows === ws
        @test plan.memory === mem
        @test plan.nthreads == 4
        @test plan.notes == String[]
        @test fieldtype(typeof(plan), :windows) === Vector{WindowPlan{Float64}}
        @test fieldtype(typeof(plan), :memory) === MemoryEstimate
        @test fieldtype(typeof(plan), :notes) === Vector{String}
        @test fieldtype(typeof(plan), :nthreads) === Int
        @test @inferred(mkplan(ws)) isa ComputationPlan
        @test_throws MethodError ComputationPlan(solver, 10.0, 12.0, 7.25, 3.0, 4.0, WindowPlan{Float32}[], mem, 4, String[])
    end

    @testset "show text/plain: odd window count" begin
        ws = [WindowPlan(10.5, 0.5, 5.0, 40, 80), WindowPlan(11.0, 0.5, 1.0, 60, 120), WindowPlan(11.5, 0.5, 3.0, 50, 100)]
        out = cp_show(mkplan(ws))
        @test occursin("ComputationPlan for BeynSolver", out)
        @test occursin("[10.0, 12.0]", out)
        @test occursin("windows         : 3", out)
        @test occursin("min 1.0, median 3.0, max 5.0", out)
        @test occursin("dim             : 40 to 60", out)
        @test occursin("matrix dim      : 80 to 120", out)
        @test occursin("threads         : 4", out)
        @test !occursin("note", out)
    end

    @testset "show text/plain: even window count" begin
        ws = [WindowPlan(10.5, 0.5, 4.0, 40, 80), WindowPlan(11.0, 0.5, 1.0, 60, 120),
              WindowPlan(11.5, 0.5, 2.0, 50, 100), WindowPlan(12.0, 0.5, 8.0, 50, 100)]
        out = cp_show(mkplan(ws))
        @test occursin("windows         : 4", out)
        @test occursin("min 1.0, median 3.0, max 8.0", out)   # median of (1,2,4,8) = (2+4)/2
    end

    @testset "show text/plain: zero windows" begin
        out = cp_show(mkplan(WindowPlan{Float64}[]))
        @test occursin("windows         : 0", out)
        @test !occursin("states/window", out)
        @test !occursin("matrix dim", out)
        @test occursin("threads         : 4", out)
    end

    @testset "show text/plain: notes" begin
        out = cp_show(mkplan(WindowPlan{Float64}[], ["first adjustment", "second warning"]))
        @test occursin("note            : first adjustment", out)
        @test occursin("note            : second warning", out)
        @test count("note            :", out) == 2
    end

    @testset "show compact" begin
        plan = mkplan([WindowPlan(10.5, 0.5, 2.0, 40, 80)])
        out = sprint(show, plan)
        @test out == "ComputationPlan(BeynSolver, 1 windows, 7.2 states)"
        @test !occursin('\n', out)
    end
end

# plan_computation (Beyn and CORK), _beyn_intervals, _cork_intervals, _bim_plan. Plans only; no eigen solves.
using QuantumBilliards: plan_computation, plan_weyl_windows, k_range_for_states, _beyn_intervals, _cork_intervals, _bim_plan

@testset "plan_computation" begin
    rect = RectangleBilliard(1.0, 0.5)
    kernel = DoubleLayerPotentialSolver(4.0, rect)
    beyn = BeynSolver(kernel; nlevels=20)
    cork = CORKSolver(kernel; nlevels=20)

    @testset "_beyn_intervals matches plan_weyl_windows" begin
        iv = _beyn_intervals(beyn, rect, 20.0, 40.0)
        ref = plan_weyl_windows(rect, 20.0, 40.0; nlevels=20, Rmax=beyn.Rmax)
        @test iv == ref
        @test !isempty(iv)
        @test iv[1][1] ≈ 20.0
        @test iv[end][2] ≈ 40.0
        @test_throws ArgumentError _beyn_intervals(beyn, rect, 40.0, 20.0)
    end

    @testset "_cork_intervals" begin
        iv = _cork_intervals(cork, rect, 20.0, 40.0)
        ref = plan_weyl_windows(rect, 20.0, 40.0; nlevels=20, Rmax=cork.Rmax)
        @test length(iv) == length(ref)
        @test iv[1:end-1] == ref[1:end-1]
        @test iv[end][1] == ref[end][1]
        prev_width = iv[end-1][2] - iv[end-1][1]
        @test iv[end][2] - iv[end][1] >= prev_width - 1e-12       # last window is not narrower than the previous one
        @test_throws ArgumentError _cork_intervals(cork, rect, 40.0, 20.0)
    end

    @testset "Beyn plan structure" begin
        plan = plan_computation(beyn, rect, 20.0, 40.0)
        ref = plan_weyl_windows(rect, 20.0, 40.0; nlevels=20, Rmax=beyn.Rmax)
        @test plan isa ComputationPlan
        @test (plan.k1, plan.k2) == (20.0, 40.0)
        @test length(plan.windows) == length(ref)
        @test sum(w.expected_states for w in plan.windows) ≈ plan.expected_total_states
        @test plan.expected_total_states ≈ expected_states(rect, 20.0, 40.0)
        @test all(w.dim > 0 && w.matrix_dim > 0 for w in plan.windows)
        @test all(w.matrix_dim == plan.windows[1].matrix_dim for w in plan.windows)
        @test all(w.dim == plan.windows[1].dim for w in plan.windows)
        @test all(w.halfwidth > 0 for w in plan.windows)
        @test plan.nthreads == Threads.nthreads()
        @test plan.density_k1 ≈ spectral_density(20.0, rect)
        @test plan.density_k2 ≈ spectral_density(40.0, rect)
    end

    @testset "Beyn tune keeps input solver unchanged" begin
        deg = beyn.taylor_degree
        plan = plan_computation(beyn, rect, 20.0, 40.0)
        @test beyn.taylor_degree == deg
        @test plan.solver isa BeynSolver
        @test plan.solver.kernel === beyn.kernel
        @test plan.solver.taylor_degree > 0
        plan0 = plan_computation(beyn, rect, 20.0, 40.0; tune=false)
        @test plan0.solver === beyn
        @test !any(occursin("taylor", n) for n in plan0.notes)
    end

    @testset "N1,N2 form equals k_range_for_states" begin
        for s in (beyn, cork)
            k1, k2 = k_range_for_states(rect, 10, 60)
            p = plan_computation(s, rect, 10, 60)
            @test p.k1 ≈ k1
            @test p.k2 ≈ k2
            pk = plan_computation(s, rect, k1, k2)
            @test length(pk.windows) == length(p.windows)
        end
    end

    @testset "CORK plan structure" begin
        plan = plan_computation(cork, rect, 20.0, 40.0)
        iv = _cork_intervals(cork, rect, 20.0, 40.0)
        @test length(plan.windows) == length(iv)
        # the last window may be widened beyond k2, so the window sum can exceed the total
        @test sum(w.expected_states for w in plan.windows) >= plan.expected_total_states - 1e-8
        @test sum(w.expected_states for w in plan.windows) ≈ sum(expected_states(rect, a, b) for (a, b) in iv)
        ws = plan.windows
        for i in 1:length(ws)-1
            @test ws[i].kcenter + ws[i].halfwidth ≈ ws[i+1].kcenter - ws[i+1].halfwidth
        end
        @test any(occursin("polynomial degree", n) for n in plan.notes)
        @test plan.solver === cork
        @test all(w.dim > 0 && w.matrix_dim > 0 for w in ws)
    end

    @testset "max_memory note" begin
        for s in (beyn, cork)
            tight = plan_computation(s, rect, 20.0, 40.0; max_memory=1.0)
            @test any(occursin("max_memory", n) for n in tight.notes)
            loose = plan_computation(s, rect, 20.0, 40.0; max_memory=1e15)
            @test !any(occursin("max_memory", n) for n in loose.notes)
        end
    end

    @testset "_bim_plan direct call" begin
        iv = _beyn_intervals(beyn, rect, 20.0, 40.0)
        plan = _bim_plan(beyn, rect, iv, 20.0, 40.0, 1e15, String[])
        @test plan isa ComputationPlan{Float64}
        @test length(plan.windows) == length(iv)
        @test plan.notes == String[]
        @test plan.solver === beyn
        @test plan.windows[1].kcenter ≈ (iv[1][1] + iv[1][2]) / 2
        @test plan.windows[1].halfwidth ≈ (iv[1][2] - iv[1][1]) / 2
        note = _bim_plan(beyn, rect, iv, 20.0, 40.0, 1.0, ["pre"])
        @test note.notes[1] == "pre"
        @test length(note.notes) == 2
    end

    @testset "empty interval throws" begin
        for s in (beyn, cork)
            @test_throws ArgumentError plan_computation(s, rect, 40.0, 20.0)
            @test_throws ArgumentError plan_computation(s, rect, 30.0, 30.0)
        end
    end
end

# plan_computation (AcceleratedBasisSolver, ExpandedBIMSolver), _vs_panels, _ebim_windows. Plans only; no eigen solves.
using QuantumBilliards: _vs_panels, _ebim_windows, _assemble_plan, sweep_windows, weyl_window_count

@testset "_vs_panels" begin
    centers = collect(0.0:0.1:3.0)
    R = 0.25
    starts, stops = _vs_panels(centers, R)
    @test length(starts) == length(stops)
    @test starts[1] == 1
    @test stops[end] == length(centers)
    @test all(starts[i+1] == stops[i] + 1 for i in 1:length(starts)-1)
    @test all(starts .<= stops)
    @test all(centers[stops[i]] - centers[starts[i]] <= 2R + 1e-12 for i in eachindex(starts))
    @test length(starts) > 1

    @testset "single center and tiny radius" begin
        @test _vs_panels([1.0], 0.5) == ([1], [1])
        s, e = _vs_panels(centers, 1e-6)       # every center is its own panel
        @test s == e == collect(1:length(centers))
        s, e = _vs_panels(centers, 10.0)       # one panel for everything
        @test (s, e) == ([1], [length(centers)])
    end
end

@testset "plan_computation accelerated" begin
    rect = RectangleBilliard(1.0, 0.5)
    vs = VerginiSaracenoSolver(2.0, 5.0)
    basis = RealPlaneWaves(6; use_taylor=false)
    L = CompositeCurve(get_boundary_curves(rect)).length
    vsdim(s, k) = max(s.min_dim, round(Int, L * k * s.dim_scaling_factor / (2 * pi)))

    @testset "window layout equals sweep_windows" begin
        plan = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05)
        centers, widths = sweep_windows(20.0, 22.0, 0.05)
        @test plan isa ComputationPlan
        @test plan.solver === vs
        @test length(plan.windows) == length(centers)
        @test [w.kcenter for w in plan.windows] == centers
        @test [w.halfwidth for w in plan.windows] == widths
        @test all(w.expected_states ≈ weyl_window_count(rect, w.kcenter - w.halfwidth, 2 * w.halfwidth) for w in plan.windows)
        @test sum(w.expected_states for w in plan.windows) > 0
        @test (plan.k1, plan.k2) == (20.0, 22.0)
        @test plan.notes == String[]
    end

    @testset "dim and matrix_dim" begin
        plan = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05)
        @test all(w.dim == vsdim(vs, w.kcenter) for w in plan.windows)
        @test all(w.matrix_dim == length(QuantumBilliards.evaluate_points(vs, rect, w.kcenter).w_vs) for w in plan.windows)
        vs2 = with_params(vs; min_dim=10)             # make dim depend on k
        plan2 = plan_computation(vs2, basis, rect, 20.0, 22.0, 0.05)
        @test all(w.dim == vsdim(vs2, w.kcenter) for w in plan2.windows)
        @test plan2.windows[end].dim >= plan2.windows[1].dim
    end

    @testset "dk as Function and solver unchanged" begin
        dkf = k -> 0.05 * k^(-1/3)
        plan = plan_computation(vs, basis, rect, 20.0, 22.0, dkf)
        centers, widths = sweep_windows(20.0, 22.0, dkf)
        @test [w.kcenter for w in plan.windows] == centers
        @test [w.halfwidth for w in plan.windows] == widths
        @test plan.solver === vs
        @test plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; tune=false).solver === vs
    end

    @testset "k2 <= k1 throws" begin
        @test_throws ArgumentError plan_computation(vs, basis, rect, 22.0, 20.0, 0.05)
        @test_throws ArgumentError plan_computation(vs, basis, rect, 20.0, 20.0, 0.05)
    end

    @testset "max_memory note" begin
        tight = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; max_memory=1.0)
        @test any(occursin("max_memory", n) for n in tight.notes)
        loose = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; max_memory=1e15)
        @test !any(occursin("max_memory", n) for n in loose.notes)
    end

    @testset "N1,N2 form" begin
        fo = basis.symmetries !== nothing
        k1, k2 = k_range_for_states(rect, 10, 30; fundamental_only=fo)
        plan = plan_computation(vs, basis, rect, 10, 30; nlevels=3)
        @test plan.k1 ≈ k1
        @test plan.k2 ≈ k2
        @test !isempty(plan.windows)
        @test any(occursin("nlevels", n) for n in plan.notes)
        plan_small = plan_computation(vs, basis, rect, 10, 30; nlevels=0.35)
        @test !any(occursin("nlevels", n) for n in plan_small.notes)
        @test length(plan_small.windows) > length(plan.windows)
        @test plan_small.k1 ≈ k1
    end

    @testset "Taylor panels" begin
        R = 0.5
        tbasis = RealPlaneWaves(6; use_taylor=true, taylor_radius=R)
        @test QuantumBilliards.uses_taylor(tbasis)
        vs2 = with_params(vs; min_dim=10)
        plan = plan_computation(vs2, tbasis, rect, 20.0, 22.0, 0.05)
        centers, _ = sweep_windows(20.0, 22.0, 0.05)
        @test [w.kcenter for w in plan.windows] == centers
        starts, stops = _vs_panels(centers, R)
        @test length(starts) > 1
        for (i1, i2) in zip(starts, stops)
            last = plan.windows[i2]
            @test all(plan.windows[i].dim == last.dim for i in i1:i2)
            @test all(plan.windows[i].matrix_dim == last.matrix_dim for i in i1:i2)
            @test last.dim == vsdim(vs2, centers[i2])
        end
        plain = plan_computation(vs2, basis, rect, 20.0, 22.0, 0.05)
        @test all(plan.windows[i].dim >= plain.windows[i].dim for i in eachindex(centers))
    end
end

@testset "plan_computation ExpandedBIMSolver" begin
    rect = RectangleBilliard(1.0, 0.5)
    eb = ExpandedBIMSolver(DoubleLayerPotentialSolver(4.0, rect))
    dkf = k -> 0.05 * k^(-1/3)

    @testset "_ebim_windows" begin
        centers, dks, nl = _ebim_windows(eb, rect, 20.0, 22.0, dkf)
        @test centers[1] == 20.0
        @test all(centers .<= 22.0)
        @test centers[end] + dks[end] > 22.0
        @test all(diff(centers) .≈ dks[1:end-1])
        @test dks == dkf.(centers)
        @test all(nl .>= 1)
        @test nl == [max(1, ceil(Int, weyl_window_count(rect, k - d, 2d))) for (k, d) in zip(centers, dks)]
        @test_throws ArgumentError _ebim_windows(eb, rect, 22.0, 20.0, dkf)
        @test_throws ArgumentError _ebim_windows(eb, rect, 20.0, 20.0, dkf)
        @test_throws ArgumentError _ebim_windows(eb, rect, 20.0, 22.0, k -> 0.0)
        @test_throws ArgumentError _ebim_windows(eb, rect, 20.0, 22.0, k -> -0.1)
    end

    @testset "plan structure" begin
        plan = plan_computation(eb, rect, 20.0, 22.0)
        centers, dks, _ = _ebim_windows(eb, rect, 20.0, 22.0, dkf)
        @test plan isa ComputationPlan
        @test plan.solver === eb
        @test length(plan.windows) == length(centers)
        @test [w.kcenter for w in plan.windows] == centers
        @test [w.halfwidth for w in plan.windows] ≈ dkf.(centers)
        for w in plan.windows
            @test w.expected_states isa Float64
            @test w.expected_states ≈ weyl_window_count(rect, w.kcenter - w.halfwidth, 2 * w.halfwidth)
        end
        @test all(w.dim == w.matrix_dim for w in plan.windows)
        @test all(w.dim > 0 for w in plan.windows)
        @test plan.expected_total_states ≈ expected_states(rect, 20.0, 22.0)
        @test plan.notes == String[]
        @test plan_computation(eb, rect, 20.0, 22.0; tune=false).solver === eb
    end

    @testset "custom dk" begin
        plan = plan_computation(eb, rect, 20.0, 22.0; dk=k -> 0.2)
        @test all(w.halfwidth == 0.2 for w in plan.windows)
        @test length(plan.windows) < length(plan_computation(eb, rect, 20.0, 22.0).windows)
        @test_throws ArgumentError plan_computation(eb, rect, 20.0, 22.0; dk=k -> 0.0)
    end

    @testset "N1,N2 form" begin
        k1, k2 = k_range_for_states(rect, 10, 30)
        plan = plan_computation(eb, rect, 10, 30)
        @test plan.k1 ≈ k1
        @test plan.k2 ≈ k2
        @test length(plan.windows) == length(_ebim_windows(eb, rect, k1, k2, dkf)[1])
        @test plan.solver === eb
    end

    @testset "k2 <= k1 throws" begin
        @test_throws ArgumentError plan_computation(eb, rect, 22.0, 20.0)
    end

    @testset "max_memory note" begin
        tight = plan_computation(eb, rect, 20.0, 22.0; max_memory=1.0)
        @test any(occursin("max_memory", n) for n in tight.notes)
        loose = plan_computation(eb, rect, 20.0, 22.0; max_memory=1e15)
        @test !any(occursin("max_memory", n) for n in loose.notes)
    end
end

@testset "_assemble_plan" begin
    rect = RectangleBilliard(1.0, 0.5)
    beyn = BeynSolver(DoubleLayerPotentialSolver(4.0, rect))
    ws = [WindowPlan(21.0, 1.0, 2.0, 40, 80)]
    mem = MemoryEstimate(2.0^30, 2.0^29)
    plan = _assemble_plan(beyn, rect, 20.0, 22.0, ws, mem, 1e15, String[]; fundamental_only=false)
    @test plan isa ComputationPlan{Float64}
    @test plan.windows === ws
    @test plan.memory === mem
    @test plan.expected_total_states ≈ expected_states(rect, 20.0, 22.0)
    @test plan.density_k1 ≈ spectral_density(20.0, rect)
    @test plan.density_k2 ≈ spectral_density(22.0, rect)
    @test plan.nthreads == Threads.nthreads()
    @test isempty(plan.notes)
    tight = _assemble_plan(beyn, rect, 20.0, 22.0, ws, mem, 1.0, String[]; fundamental_only=false)
    @test length(tight.notes) == 1 && occursin("max_memory", tight.notes[1])
end
