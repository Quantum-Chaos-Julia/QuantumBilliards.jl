# Unit tests for src/solvers/computationplan.jl:
#   expected_states, WindowPlan, ComputationPlan and the Base.show methods of ComputationPlan.
# `with_params` is covered in solvers/solverconstructors_tests.jl. Plans are built by hand; no eigen solves.
# Helper definitions use the prefix `cp_`.

using QuantumBilliards: expected_states, WindowPlan, ComputationPlan, MemoryEstimate, default_memory_limit

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
    circle = CircleBilliard(1.0)
    solver = BeynSolver(DoubleLayerPotentialSolver(10.0, circle))
    mem = MemoryEstimate(2.0^30, 2.0^29)
    mkplan(windows, notes=String[]) = ComputationPlan(solver, circle, nothing, 10.0, 12.0, nothing, 0.5, 1e15, 7.25, 3.0, 4.0, windows, mem, 4, notes)

    @testset "constructor and field types" begin
        ws = [WindowPlan(10.5, 0.5, 2.0, 40, 80), WindowPlan(11.5, 0.5, 3.0, 50, 100)]
        plan = mkplan(ws)
        @test plan isa ComputationPlan{Float64,typeof(solver)}
        @test isconcretetype(typeof(plan))
        @test plan.solver === solver
        @test plan.billiard === circle
        @test plan.basis === nothing
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
        @test_throws MethodError ComputationPlan(solver, circle, nothing, 10.0, 12.0, nothing, 0.5, 1e15, 7.25, 3.0, 4.0, WindowPlan{Float32}[], mem, 4, String[])
    end

    @testset "dk, window_overlap and max_memory fields" begin
        ws = [WindowPlan(10.5, 0.5, 2.0, 40, 80)]
        plan = mkplan(ws)
        @test plan.dk === nothing
        @test plan.window_overlap === 0.5
        @test plan.max_memory === 1e15
        @test plan isa ComputationPlan{Float64,typeof(solver),typeof(circle),Nothing,Nothing}
        @test fieldtype(typeof(plan), :dk) === Nothing
        @test fieldtype(typeof(plan), :window_overlap) === Float64
        @test fieldtype(typeof(plan), :max_memory) === Float64
        @test fieldnames(ComputationPlan)[6:8] == (:dk, :window_overlap, :max_memory)

        # dk is stored as given; the type parameter D follows it
        dkf = k -> 0.1 * k^(-1/3)
        plan_f = ComputationPlan(solver, circle, nothing, 10.0, 12.0, dkf, 0.25, 2.0^30, 7.25, 3.0, 4.0, ws, mem, 4, String[])
        @test plan_f.dk === dkf
        @test plan_f isa ComputationPlan{Float64,typeof(solver),typeof(circle),Nothing,typeof(dkf)}
        @test isconcretetype(typeof(plan_f))
        @test (plan_f.window_overlap, plan_f.max_memory) === (0.25, 2.0^30)
        plan_r = ComputationPlan(solver, circle, nothing, 10.0, 12.0, 0.05, 0.0, 2.0^30, 7.25, 3.0, 4.0, ws, mem, 4, String[])
        @test plan_r.dk === 0.05
        @test plan_r isa ComputationPlan{Float64,typeof(solver),typeof(circle),Nothing,Float64}

        # the default constructor does not convert: Int overlap or memory limit is rejected
        @test_throws MethodError ComputationPlan(solver, circle, nothing, 10.0, 12.0, nothing, 1, 10^9, 7.25, 3.0, 4.0, ws, mem, 4, String[])
    end

    @testset "show text/plain: odd window count" begin
        ws = [WindowPlan(10.5, 0.5, 5.0, 40, 80), WindowPlan(11.0, 0.5, 1.0, 60, 120), WindowPlan(11.5, 0.5, 3.0, 50, 100)]
        out = cp_show(mkplan(ws))
        @test occursin("ComputationPlan for BeynSolver", out)
        @test occursin("[10.0, 12.0]", out)
        for section in ("Solver", "Spectrum", "Windows", "Discretization", "Memory (upper-bound estimate)", "Resources")
            @test occursin("\n" * section, out)
        end
        @test occursin("  count           : 3", out)
        @test occursin("min 1.0, median 3.0, max 5.0", out)
        @test occursin("  boundary pts    : 40 to 60  (full boundary)", out)
        @test occursin("  matrix dim      : 80 to 120", out)
        @test occursin("  julia threads   : 4", out)
        @test occursin("  blas threads ", out)
        @test !occursin("Notes", out)
        @test occursin("  working peak    : ", out)
        @test occursin("  retained        : ", out)
        @test occursin("<- peak", out)
    end

    @testset "show text/plain: even window count" begin
        ws = [WindowPlan(10.5, 0.5, 4.0, 40, 80), WindowPlan(11.0, 0.5, 1.0, 60, 120),
              WindowPlan(11.5, 0.5, 2.0, 50, 100), WindowPlan(12.0, 0.5, 8.0, 50, 100)]
        out = cp_show(mkplan(ws))
        @test occursin("  count           : 4", out)
        @test occursin("min 1.0, median 3.0, max 8.0", out)   # median of (1,2,4,8) = (2+4)/2
    end

    @testset "show text/plain: zero windows" begin
        out = cp_show(mkplan(WindowPlan{Float64}[]))
        @test occursin("  count           : 0", out)
        @test !occursin("states/window", out)
        @test !occursin("matrix dim", out)
        @test !occursin("Discretization", out)
        @test occursin("Memory (upper-bound estimate)", out)
        @test occursin("  working         : ", out)       # no stage breakdown without windows
        @test occursin("  retained        : ", out)
        @test !occursin("working peak", out)
        @test !occursin("<- peak", out)
        @test occursin("  julia threads   : 4", out)
    end

    @testset "show text/plain: notes" begin
        out = cp_show(mkplan(WindowPlan{Float64}[], ["first adjustment", "second warning"]))
        @test occursin("\nNotes\n  - first adjustment", out)
        @test occursin("\n  - second warning", out)
        @test count("\n  - ", out) == 2
        @test endswith(out, "second warning")
    end

    @testset "show text/plain: overlap line" begin
        ws = [WindowPlan(10.5, 0.5, 2.0, 40, 80), WindowPlan(11.5, 0.5, 3.0, 50, 100)]
        # dk === nothing (Beyn, CORK): no overlap line
        @test !occursin("overlap", cp_show(mkplan(ws)))
        # dk set (sweep methods): overlap line in the Windows section
        plan_f = ComputationPlan(solver, circle, nothing, 10.0, 12.0, k -> 0.1, 0.25, 1e15, 7.25, 3.0, 4.0, ws, mem, 4, String[])
        out = cp_show(plan_f)
        @test occursin("  overlap         : 0.25", out)
        @test count("overlap", out) == 1
        windows_section = split(split(out, "\nWindows\n")[2], "\nDiscretization")[1]
        @test occursin("  overlap         : 0.25", windows_section)
        plan_r = ComputationPlan(solver, circle, nothing, 10.0, 12.0, 0.05, 0.0, 1e15, 7.25, 3.0, 4.0, ws, mem, 4, String[])
        @test occursin("  overlap         : 0.0\n", cp_show(plan_r))
        # zero windows: no Windows details, so no overlap line even with dk set
        plan_empty = ComputationPlan(solver, circle, nothing, 10.0, 12.0, 0.05, 0.5, 1e15, 7.25, 3.0, 4.0, WindowPlan{Float64}[], mem, 4, String[])
        @test !occursin("overlap", cp_show(plan_empty))
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

    @testset "stored dk, window_overlap and max_memory" begin
        for s in (beyn, cork)
            plan = plan_computation(s, rect, 20.0, 40.0; tune=false)
            @test plan.dk === nothing
            @test plan.window_overlap === 0.5
            @test plan.max_memory === default_memory_limit()
            @test plan.max_memory isa Float64
            @test fieldtype(typeof(plan), :dk) === Nothing
            @test !occursin("overlap", cp_show(plan))

            custom = plan_computation(s, rect, 20.0, 40.0; tune=false, max_memory=2^35)
            @test custom.max_memory === Float64(2^35)
            @test custom.dk === nothing

            # N1,N2 form and tuned plan forward max_memory too
            @test plan_computation(s, rect, 10, 60; max_memory=1e12).max_memory === 1e12
            @test plan_computation(s, rect, 20.0, 40.0; max_memory=1e12).max_memory === 1e12
            # stored even when it triggers the note
            @test plan_computation(s, rect, 20.0, 40.0; max_memory=1.0).max_memory === 1.0
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

    @testset "window_overlap" begin
        default = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05)
        explicit = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=0.5)
        @test [w.kcenter for w in explicit.windows] == [w.kcenter for w in default.windows]
        @test [w.halfwidth for w in explicit.windows] == [w.halfwidth for w in default.windows]

        for o in (0.0, 0.25)
            plan = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=o)
            centers, widths = sweep_windows(20.0, 22.0, 0.05; window_overlap=o)
            @test [w.kcenter for w in plan.windows] == centers
            @test [w.halfwidth for w in plan.windows] == widths
            @test all(w.halfwidth == 0.05 for w in plan.windows)
            @test all(w.dim == vsdim(vs, w.kcenter) for w in plan.windows)
        end

        zero_overlap = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=0.0)
        quarter = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=0.25)
        @test length(zero_overlap.windows) < length(quarter.windows) < length(default.windows)
        # o = 0: windows are spaced 2*halfwidth apart and adjacent
        kc = [w.kcenter for w in zero_overlap.windows]
        @test all(diff(kc) .≈ 2 * 0.05)
        @test kc[end] + zero_overlap.windows[end].halfwidth >= 22.0
        # same interval, same expected total states
        @test zero_overlap.expected_total_states ≈ default.expected_total_states
        @test zero_overlap.memory == QuantumBilliards.estimate_memory(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=0.0)

        @test_throws ArgumentError plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=0.6)
        @test_throws ArgumentError plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=-0.1)
        @test_throws ArgumentError plan_computation(vs, basis, rect, 10, 30; window_overlap=0.6)
    end

    @testset "stored dk, window_overlap and max_memory" begin
        # constant dk
        plan = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05)
        @test plan.dk === 0.05
        @test plan.window_overlap === 0.5
        @test plan.max_memory === default_memory_limit()
        @test plan isa ComputationPlan{Float64,typeof(vs),typeof(rect),typeof(basis),Float64}
        @test occursin("  overlap         : 0.5", cp_show(plan))

        # function dk
        dkf = k -> 0.05 * k^(-1/3)
        plan_f = plan_computation(vs, basis, rect, 20.0, 22.0, dkf)
        @test plan_f.dk === dkf
        @test fieldtype(typeof(plan_f), :dk) === typeof(dkf)

        # custom window_overlap and max_memory
        plan_c = plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=0.25, max_memory=2^35)
        @test plan_c.window_overlap === 0.25
        @test plan_c.max_memory === Float64(2^35)
        @test plan_c.dk === 0.05
        @test occursin("  overlap         : 0.25", cp_show(plan_c))
        # integer overlap is stored as Float64
        @test plan_computation(vs, basis, rect, 20.0, 22.0, 0.05; window_overlap=0).window_overlap === 0.0

        # N1,N2 form stores the resolved dk function and the keywords
        fo = basis.symmetries !== nothing
        plan_n = plan_computation(vs, basis, rect, 10, 30; nlevels=0.35, window_overlap=0.25, max_memory=1e12)
        @test plan_n.dk isa Function
        @test plan_n.dk(15.0) ≈ 0.35 / (2 * spectral_density(15.0, rect; fundamental_only=fo))
        @test plan_n.window_overlap === 0.25
        @test plan_n.max_memory === 1e12
        @test plan_computation(vs, basis, rect, 10, 30).dk isa Function
        @test plan_computation(vs, basis, rect, 10, 30).window_overlap === 0.5
    end

    @testset "N1,N2 nlevels is levels in the full window" begin
        fo = basis.symmetries !== nothing
        dens(k) = spectral_density(k, rect; fundamental_only=fo)
        for nl in (0.35, 0.7)
            plan = plan_computation(vs, basis, rect, 10, 30; nlevels=nl)
            # default overlap: widths[i] = dk(centers[i]) exactly, half-width = nlevels / (2 density)
            @test all(isapprox(w.halfwidth, nl / (2 * dens(w.kcenter)); rtol=1e-10) for w in plan.windows)
            # a full window holds about nlevels mean levels (area-only Weyl count vs density with perimeter term: ~20% at k ~ 10)
            @test all(isapprox(w.expected_states, nl; rtol=0.3) for w in plan.windows)
        end
        p035 = plan_computation(vs, basis, rect, 10, 30; nlevels=0.35)
        p070 = plan_computation(vs, basis, rect, 10, 30; nlevels=0.7)
        @test p070.windows[1].halfwidth ≈ 2 * p035.windows[1].halfwidth
        @test length(p070.windows) < length(p035.windows)

        # first window is centered at k1 with half-width nlevels / (2 density(k1))
        @test p035.windows[1].kcenter ≈ p035.k1

        # default nlevels is 6
        pdef = plan_computation(vs, basis, rect, 10, 30)
        p6 = plan_computation(vs, basis, rect, 10, 30; nlevels=6)
        @test [w.kcenter for w in pdef.windows] == [w.kcenter for w in p6.windows]
        @test [w.halfwidth for w in pdef.windows] == [w.halfwidth for w in p6.windows]
        @test pdef.windows[1].halfwidth ≈ 6 / (2 * dens(pdef.k1))

        # window_overlap is forwarded in the N1,N2 form
        p_o0 = plan_computation(vs, basis, rect, 10, 30; nlevels=0.35, window_overlap=0.0)
        @test length(p_o0.windows) < length(p035.windows)
        @test p_o0.windows[1].halfwidth ≈ p035.windows[1].halfwidth
        @test all(isapprox(w.halfwidth, 0.35 / (2 * dens(w.kcenter)); rtol=0.05) for w in p_o0.windows)
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

    @testset "window_overlap" begin
        default = plan_computation(eb, rect, 20.0, 22.0)
        explicit = plan_computation(eb, rect, 20.0, 22.0; window_overlap=0.5)
        @test [w.kcenter for w in explicit.windows] == [w.kcenter for w in default.windows]

        # _ebim_windows: inclusive layout, overlap property, forwarded keyword
        for o in (0.0, 0.25, 0.5)
            centers, dks, nl = _ebim_windows(eb, rect, 20.0, 22.0, dkf; window_overlap=o)
            @test centers[1] == 20.0
            @test centers[end] + dks[end] >= 22.0
            @test all(nl .>= 1)
            for i in 2:length(centers)
                @test (centers[i-1] + dks[i-1]) - (centers[i] - dks[i]) ≈ 2 * o * dks[i] atol = 1e-12
            end
        end
        c_inc, _, _ = _ebim_windows(eb, rect, 20.0, 22.0, k -> 0.5; window_overlap=0.0)
        @test c_inc == [20.0, 21.0, 22.0]
        # exact edge: the EBIM layout adds one more window than sweep_windows
        @test length(first(_ebim_windows(eb, rect, 20.0, 21.0, k -> 1.0; window_overlap=0.0))) ==
              length(first(sweep_windows(20.0, 21.0, k -> 1.0; window_overlap=0.0))) + 1
        @test_throws ArgumentError _ebim_windows(eb, rect, 20.0, 22.0, dkf; window_overlap=0.6)

        # plan windows
        zero_overlap = plan_computation(eb, rect, 20.0, 22.0; window_overlap=0.0)
        quarter = plan_computation(eb, rect, 20.0, 22.0; window_overlap=0.25)
        centers0, dks0, _ = _ebim_windows(eb, rect, 20.0, 22.0, dkf; window_overlap=0.0)
        @test [w.kcenter for w in zero_overlap.windows] == centers0
        @test [w.halfwidth for w in zero_overlap.windows] ≈ dks0
        @test length(zero_overlap.windows) < length(quarter.windows) < length(default.windows)
        @test zero_overlap.expected_total_states ≈ default.expected_total_states
        @test all(w.dim == w.matrix_dim && w.dim > 0 for w in zero_overlap.windows)
        for w in zero_overlap.windows
            @test w.expected_states ≈ weyl_window_count(rect, w.kcenter - w.halfwidth, 2 * w.halfwidth)
        end

        # constant dk: spacing 2*(1-o)*dk
        plan = plan_computation(eb, rect, 20.0, 22.0; dk=k -> 0.2, window_overlap=0.25)
        @test all(diff([w.kcenter for w in plan.windows]) .≈ 2 * (1 - 0.25) * 0.2)

        # memory estimate follows the same layout
        @test zero_overlap.memory == QuantumBilliards.estimate_memory(eb, zero_overlap.windows, zero_overlap.expected_total_states)
        @test zero_overlap.memory == QuantumBilliards.estimate_memory(eb, rect, 20.0, 22.0; dk=dkf, window_overlap=0.0)

        @test_throws ArgumentError plan_computation(eb, rect, 20.0, 22.0; window_overlap=0.6)
        @test_throws ArgumentError plan_computation(eb, rect, 20.0, 22.0; window_overlap=-0.1)
        @test_throws ArgumentError plan_computation(eb, rect, 10, 30; window_overlap=0.6)
    end

    @testset "stored dk, window_overlap and max_memory" begin
        # default dk is stored as the resolved Function
        plan = plan_computation(eb, rect, 20.0, 22.0)
        @test plan.dk isa Function
        @test plan.dk(21.0) ≈ dkf(21.0)
        @test plan.window_overlap === 0.5
        @test plan.max_memory === default_memory_limit()
        @test occursin("  overlap         : 0.5", cp_show(plan))

        # custom dk, window_overlap, max_memory
        mydk = k -> 0.2
        plan_c = plan_computation(eb, rect, 20.0, 22.0; dk=mydk, window_overlap=0.25, max_memory=2^35)
        @test plan_c.dk === mydk
        @test plan_c.window_overlap === 0.25
        @test plan_c.max_memory === Float64(2^35)
        @test occursin("  overlap         : 0.25", cp_show(plan_c))
        @test plan_computation(eb, rect, 20.0, 22.0; window_overlap=0).window_overlap === 0.0

        # N1,N2 without nlevels keeps the default dk and forwards keywords
        plan_n = plan_computation(eb, rect, 10, 30; window_overlap=0.1, max_memory=1e12)
        @test plan_n.dk isa Function
        @test plan_n.dk(15.0) ≈ dkf(15.0)
        @test plan_n.window_overlap === 0.1
        @test plan_n.max_memory === 1e12

        # N1,N2 with nlevels: dk = nlevels / (2 * spectral_density)
        for nl in (1.0, 2.5)
            p = plan_computation(eb, rect, 10, 30; nlevels=nl, window_overlap=0.25, max_memory=1e12)
            @test p.dk isa Function
            for k in (10.0, 17.3, 25.0)
                @test p.dk(k) ≈ nl / (2 * spectral_density(k, rect))
            end
            @test p.window_overlap === 0.25
            @test p.max_memory === 1e12
            # with overlap != 0.5 the local half-width only follows dk approximately (see the window_overlap testset)
            @test all(isapprox(w.halfwidth, p.dk(w.kcenter); rtol=0.05) for w in p.windows)
            @test all(isapprox(w.halfwidth, nl / (2 * spectral_density(w.kcenter, rect)); rtol=1e-10) for w in plan_computation(eb, rect, 10, 30; nlevels=nl).windows)
        end
        # nlevels=nothing is the same as omitting it
        @test plan_computation(eb, rect, 10, 30; nlevels=nothing).dk(15.0) ≈ dkf(15.0)
    end

    @testset "N1,N2 nlevels is levels in the full window" begin
        dens(k) = spectral_density(k, rect)
        for nl in (1.0, 2.0)
            plan = plan_computation(eb, rect, 10, 30; nlevels=nl)
            # default overlap: widths[i] = dk(centers[i]), half-width = nlevels / (2 density)
            @test all(isapprox(w.halfwidth, nl / (2 * dens(w.kcenter)); rtol=1e-10) for w in plan.windows)
            @test all(isapprox(w.expected_states, nl; rtol=0.3) for w in plan.windows)
        end
        p1 = plan_computation(eb, rect, 10, 30; nlevels=1.0)
        p2 = plan_computation(eb, rect, 10, 30; nlevels=2.0)
        @test p2.windows[1].halfwidth ≈ 2 * p1.windows[1].halfwidth
        @test length(p2.windows) < length(p1.windows)
        p1_o0 = plan_computation(eb, rect, 10, 30; nlevels=1.0, window_overlap=0.0)
        @test length(p1_o0.windows) < length(p1.windows)
        @test p1_o0.windows[1].halfwidth ≈ p1.windows[1].halfwidth
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

    @testset "dk, window_overlap and max_memory" begin
        @test plan.dk === nothing
        @test plan.window_overlap === 0.5
        @test plan.max_memory === 1e15
        @test tight.max_memory === 1.0
        dkf = k -> 0.1
        p = _assemble_plan(beyn, rect, 20.0, 22.0, ws, mem, 2^30, String[]; fundamental_only=false, dk=dkf, window_overlap=0.25)
        @test p.dk === dkf
        @test p.window_overlap === 0.25
        @test p.max_memory === Float64(2^30)
        @test fieldtype(typeof(p), :dk) === typeof(dkf)
        p0 = _assemble_plan(beyn, rect, 20.0, 22.0, ws, mem, 1e15, String[]; fundamental_only=false, dk=0.05, window_overlap=0)
        @test p0.dk === 0.05
        @test p0.window_overlap === 0.0
    end
end

# Memory estimates: _working_bytes, _persistent_bytes (src/solvers/memoryestimates.jl), estimate_memory, window-plan helpers
# (src/solvers/computationplan.jl) and the method existence of compute_spectrum(plan, ...) (src/spectra/spectralutils.jl).
# Plans and estimates only; no eigen solves, compute_spectrum is not called.
using QuantumBilliards: estimate_memory, total_bytes, _working_bytes, _persistent_bytes, _enforce_memory_limit,
                        _bim_window_plans, _ebim_window_plans, _vs_window_plans, _single_window

@testset "memory estimate formulas" begin
    rect = RectangleBilliard(1.0, 0.5)
    kernel = DoubleLayerPotentialSolver(4.0, rect)
    beyn = BeynSolver(kernel; nlevels=20)
    cork = CORKSolver(kernel; nlevels=20)
    eb = ExpandedBIMSolver(kernel)
    vs = VerginiSaracenoSolver(2.0, 5.0)
    pw = RealPlaneWaves(6; use_taylor=false)

    @testset "_working_bytes BIM solvers" begin
        for s in (beyn, cork, eb)
            w = _working_bytes(s, 100.0, 50.0)
            @test w isa Float64
            @test w > 0
            @test _working_bytes(s, 200.0, 50.0) > w        # grows with Nf
            @test _working_bytes(s, 100.0, 100.0) > w       # grows with M
            @test _working_bytes(s, 100, 50) == w           # integer sizes accepted
        end
    end

    @testset "_persistent_bytes BIM solvers" begin
        @test _persistent_bytes(eb, 3, 10.0, 100.0, 50.0) == 0.0
        for s in (beyn, cork)
            p = _persistent_bytes(s, 3, 10.0, 100.0, 50.0)
            @test p >= 0
            @test _persistent_bytes(s, 6, 10.0, 100.0, 50.0) >= p
            @test _persistent_bytes(s, 3, 20.0, 100.0, 50.0) >= p
        end
        @test _persistent_bytes(with_params(beyn; eigenvectors=true), 3, 10.0, 100.0, 50.0) > _persistent_bytes(with_params(beyn; eigenvectors=false, imag_k_check=false), 3, 10.0, 100.0, 50.0)
        @test _persistent_bytes(with_params(cork; eigenvectors=false), 3, 10.0, 100.0, 50.0) == 0.0
        @test _persistent_bytes(with_params(cork; eigenvectors=true), 3, 10.0, 100.0, 50.0) > 0.0
    end

    @testset "VerginiSaraceno formulas" begin
        w = _working_bytes(vs, pw, 100.0, 50.0)
        @test w > 0
        @test _working_bytes(vs, pw, 200.0, 50.0) > w
        @test _working_bytes(vs, pw, 100.0, 100.0) > w
        tpw = RealPlaneWaves(6; use_taylor=true, taylor_radius=0.5)
        @test _working_bytes(vs, tpw, 100.0, 50.0) > w     # Taylor tensors add to the working set
        @test _persistent_bytes(with_params(vs; eigenvectors=false), pw, 10.0, 100.0) == 0.0
        @test _persistent_bytes(with_params(vs; eigenvectors=true), pw, 10.0, 100.0) == 8.0 * 10.0 * 100.0
    end

    @testset "_enforce_memory_limit forwards kwargs" begin
        dkf = k -> 0.05 * k^(-1/3)
        @test _enforce_memory_limit(Inf, "EBIM", eb, rect, 20.0, 22.0; dk=dkf) === nothing
        @test _enforce_memory_limit(1e15, "EBIM", eb, rect, 20.0, 22.0; dk=dkf) === nothing
        @test_throws ErrorException _enforce_memory_limit(1.0, "EBIM", eb, rect, 20.0, 22.0; dk=dkf)
        @test_throws ErrorException _enforce_memory_limit(1.0, "Beyn", beyn, rect, 20.0, 22.0)
    end
end

@testset "estimate_memory from windows" begin
    rect = RectangleBilliard(1.0, 0.5)
    kernel = DoubleLayerPotentialSolver(4.0, rect)
    beyn = BeynSolver(kernel; nlevels=20)
    cork = CORKSolver(kernel; nlevels=20)
    eb = ExpandedBIMSolver(kernel)
    vs = VerginiSaracenoSolver(2.0, 5.0)
    pw = RealPlaneWaves(6; use_taylor=false)
    small = WindowPlan(21.0, 1.0, 2.0, 40, 80)
    big = WindowPlan(23.0, 1.0, 2.0, 80, 160)
    ws = WindowPlan{Float64}[]

    @testset "BIM solvers" begin
        for s in (beyn, cork, eb)
            e_small = estimate_memory(s, [small], 2.0)
            e_big = estimate_memory(s, [big], 2.0)
            @test e_small isa MemoryEstimate
            @test e_big.working > e_small.working                              # monotone in window dims
            @test e_small.working ≈ _working_bytes(s, 40.0, 80.0)
            e_mix = estimate_memory(s, [small, big, small], 6.0)
            @test e_mix.working >= e_big.working                                # max over windows, window count only enters through nw
            @test e_mix.working == _working_bytes(s, 80.0, 160.0; nw=3)
            @test e_mix.working == estimate_memory(s, [big, small, small], 6.0).working
            @test e_mix.persistent ≈ _persistent_bytes(s, 3, 6.0, 80.0, 160.0)  # totals / max dims
            e0 = estimate_memory(s, ws, 0.0)
            @test e0.working == 0.0
            @test e0.persistent == 0.0
            @test total_bytes(e_mix) == e_mix.working + e_mix.persistent
        end
    end

    @testset "VerginiSaraceno" begin
        e_small = estimate_memory(vs, pw, [small], 2.0)
        e_big = estimate_memory(vs, pw, [big], 2.0)
        @test e_big.working > e_small.working
        @test e_small.working ≈ _working_bytes(vs, pw, 40.0, 80.0)
        e_mix = estimate_memory(vs, pw, [small, big, small], 6.0)
        @test e_mix.working == e_big.working
        @test e_mix.persistent ≈ _persistent_bytes(vs, pw, 6.0, 80.0)
        e0 = estimate_memory(vs, pw, ws, 0.0)
        @test e0.working == 0.0
        @test e0.persistent == 0.0
    end

    @testset "window helpers" begin
        iv = _beyn_intervals(beyn, rect, 20.0, 40.0)
        bw = _bim_window_plans(beyn, rect, iv)
        @test bw isa Vector{WindowPlan{Float64}}
        @test length(bw) == length(iv)
        @test [w.kcenter for w in bw] ≈ [(a + b) / 2 for (a, b) in iv]
        @test [w.halfwidth for w in bw] ≈ [(b - a) / 2 for (a, b) in iv]
        @test all(w.dim > 0 && w.matrix_dim > 0 for w in bw)

        dkf = k -> 0.05 * k^(-1/3)
        ew = _ebim_window_plans(eb, rect, 20.0, 22.0, dkf)
        centers, dks, _ = _ebim_windows(eb, rect, 20.0, 22.0, dkf)
        @test [w.kcenter for w in ew] == centers
        @test [w.halfwidth for w in ew] == dks
        @test all(w.dim > 0 && w.matrix_dim > 0 for w in ew)

        vw = _vs_window_plans(vs, pw, rect, 20.0, 22.0, 0.05)
        vcenters, vwidths = sweep_windows(20.0, 22.0, 0.05)
        @test [w.kcenter for w in vw] == vcenters
        @test [w.halfwidth for w in vw] == vwidths
        @test all(w.dim > 0 && w.matrix_dim > 0 for w in vw)
        @test _vs_window_plans(vs, pw, rect, 20.0, 22.0, k -> 0.05) isa Vector{WindowPlan{Float64}}

        sw = _single_window(eb, rect, 20.0, 22.0)
        @test length(sw) == 1
        @test sw[1].kcenter == 22.0
        @test sw[1].halfwidth == 0.0
        @test sw[1].expected_states == 0.0
        @test sw[1].dim > 0 && sw[1].matrix_dim > 0
        sv = _single_window(vs, rect, 20.0, 22.0; vs_basis=pw)
        @test length(sv) == 1
        @test sv[1].kcenter == 22.0
        @test sv[1].dim > 0 && sv[1].matrix_dim > 0
    end
end

@testset "estimate_memory billiard forms and plan.memory" begin
    rect = RectangleBilliard(1.0, 0.5)
    kernel = DoubleLayerPotentialSolver(4.0, rect)
    beyn = BeynSolver(kernel; nlevels=20)
    cork = CORKSolver(kernel; nlevels=20)
    eb = ExpandedBIMSolver(kernel)
    vs = VerginiSaracenoSolver(2.0, 5.0)
    pw = RealPlaneWaves(6; use_taylor=false)
    dkf = k -> 0.05 * k^(-1/3)

    @testset "Beyn and CORK" begin
        for s in (beyn, cork)
            plan = plan_computation(s, rect, 20.0, 40.0; tune=false)
            @test plan.memory == estimate_memory(s, plan.windows, plan.expected_total_states)
            @test plan.memory == estimate_memory(s, rect, 20.0, 40.0)
            tuned = plan_computation(s, rect, 20.0, 40.0)
            @test tuned.memory == estimate_memory(tuned.solver, tuned.windows, tuned.expected_total_states)
            @test tuned.memory == estimate_memory(tuned.solver, rect, 20.0, 40.0)
        end
        # the largest window dominates the working set
        plan = plan_computation(beyn, rect, 20.0, 40.0; tune=false)
        wmax = argmax(w -> _working_bytes(beyn, Float64(w.dim), Float64(w.matrix_dim); nw=length(plan.windows)), plan.windows)
        @test plan.memory.working == _working_bytes(beyn, Float64(wmax.dim), Float64(wmax.matrix_dim); nw=length(plan.windows))
    end

    @testset "ExpandedBIMSolver" begin
        plan = plan_computation(eb, rect, 20.0, 22.0; tune=false)
        @test plan.memory == estimate_memory(eb, plan.windows, plan.expected_total_states)
        @test plan.memory == estimate_memory(eb, rect, 20.0, 22.0; dk=dkf)
        single = estimate_memory(eb, rect, 20.0, 22.0)                        # dk = nothing -> single window at k2
        @test single == estimate_memory(eb, _single_window(eb, rect, 20.0, 22.0), expected_states(rect, 20.0, 22.0))
        @test single.working > 0
        @test estimate_memory(eb, rect, 20.0, 22.0; dk=nothing) == single
    end

    @testset "VerginiSaracenoSolver" begin
        plan = plan_computation(vs, pw, rect, 20.0, 22.0, 0.05; tune=false)
        @test plan.memory == estimate_memory(vs, pw, plan.windows, plan.expected_total_states)
        @test plan.memory == estimate_memory(vs, pw, rect, 20.0, 22.0, 0.05)
        single = estimate_memory(vs, pw, rect, 20.0, 22.0)                    # dk = nothing -> single window at k2
        @test single == estimate_memory(vs, pw, _single_window(vs, rect, 20.0, 22.0; vs_basis=pw), expected_states(rect, 20.0, 22.0))
        @test single.working > 0
        @test estimate_memory(vs, pw, rect, 20.0, 22.0, nothing) == single
    end
end

# Per-solver memory breakdown (src/solvers/memoryestimates.jl) and its use in the ComputationPlan memory section.
using QuantumBilliards: _memory_stages, _memory_retained, _stage_peak, _format_bytes, _plan_memory_items

cp_stage_labels(stages) = first.(stages)
cp_valid_items(items) = all(!isempty(first(x)) && isfinite(last(x)) && last(x) > 0 for x in items)

@testset "_memory_stages and _memory_retained" begin
    rect = RectangleBilliard(1.0, 0.5)
    kernel = DoubleLayerPotentialSolver(4.0, rect)
    beyn = BeynSolver(kernel; nlevels=20)
    cork = CORKSolver(kernel; nlevels=20)
    eb = ExpandedBIMSolver(kernel)
    vs = VerginiSaracenoSolver(2.0, 5.0)
    pw = RealPlaneWaves(6; use_taylor=false)
    tpw = RealPlaneWaves(6; use_taylor=true, taylor_radius=0.5)
    Nf, M, ns = 200.0, 100.0, 10.0

    @testset "stage and retained entries are well formed" begin
        for s in (beyn, cork, eb)
            stages = _memory_stages(s, Nf, M)
            @test stages isa Vector{Pair{String,Float64}}
            @test !isempty(stages)
            @test cp_valid_items(stages)
            retained = _memory_retained(s, 3, ns, Nf, M)
            @test retained isa Vector{Pair{String,Float64}}
            @test cp_valid_items(retained)
        end
        for b in (pw, tpw)
            stages = _memory_stages(vs, b, Nf, M)
            @test stages isa Vector{Pair{String,Float64}}
            @test !isempty(stages)
            @test cp_valid_items(stages)
            @test cp_valid_items(_memory_retained(with_params(vs; eigenvectors=true), b, ns, Nf))
        end
    end

    @testset "_stage_peak" begin
        @test _stage_peak(Pair{String,Float64}[]) == 0.0
        @test _stage_peak(["a" => 1.0, "b" => 5.0, "c" => 3.0]) == 5.0
        for s in (beyn, cork, eb)
            @test _stage_peak(_memory_stages(s, Nf, M)) == _working_bytes(s, Nf, M)
            @test _stage_peak(_memory_stages(s, Nf, M; nw=4)) == _working_bytes(s, Nf, M; nw=4)
        end
        for b in (pw, tpw)
            @test _stage_peak(_memory_stages(vs, b, Nf, M)) == _working_bytes(vs, b, Nf, M)
        end
    end

    @testset "retained sum equals _persistent_bytes" begin
        for s in (beyn, cork, eb)
            @test sum(last, _memory_retained(s, 3, ns, Nf, M); init=0.0) == _persistent_bytes(s, 3, ns, Nf, M)
        end
        for b in (pw, tpw), ev in (true, false)
            v = with_params(vs; eigenvectors=ev)
            @test sum(last, _memory_retained(v, b, ns, Nf); init=0.0) == _persistent_bytes(v, b, ns, Nf)
        end
    end

    @testset "Beyn stages" begin
        taylor = with_params(beyn; use_taylor=true, use_chebyshev=false)
        direct = with_params(beyn; use_taylor=false, use_chebyshev=false)
        lt = cp_stage_labels(_memory_stages(taylor, Nf, M))
        ld = cp_stage_labels(_memory_stages(direct, Nf, M))
        @test lt != ld
        @test any(startswith("degree tuning"), lt)
        @test any(startswith("contour matrices"), lt)
        @test any(startswith("streamed contour sweep"), ld)
        @test !any(startswith("degree tuning"), ld)
        @test !any(startswith("streamed contour sweep"), lt)

        cheb = with_params(beyn; use_taylor=false, use_chebyshev=true)
        lc = cp_stage_labels(_memory_stages(cheb, Nf, M))
        @test any(startswith("contour matrices"), lc)
        @test !any(startswith("degree tuning"), lc)
        @test !any(startswith("streamed contour sweep"), lc)

        for s in (taylor, direct, cheb)
            on = cp_stage_labels(_memory_stages(with_params(s; imag_k_check=true), Nf, M))
            off = cp_stage_labels(_memory_stages(with_params(s; imag_k_check=false), Nf, M))
            @test any(startswith("imaginary-k check"), on)
            @test !any(startswith("imaginary-k check"), off)
            @test length(on) == length(off) + 1
        end
    end

    @testset "Beyn imaginary-k stage grows with window count" begin
        s = with_params(beyn; imag_k_check=true)
        imag(nw) = last(only(filter(p -> startswith(first(p), "imaginary-k check"), _memory_stages(s, Nf, M; nw=nw))))
        @test imag(4) > imag(1)
        @test imag(8) > imag(4)
        @test _working_bytes(s, Nf, M; nw=1000) > _working_bytes(s, Nf, M; nw=1)
        off = with_params(beyn; imag_k_check=false)
        @test _working_bytes(off, Nf, M; nw=1000) == _working_bytes(off, Nf, M; nw=1)
        # the other stages do not depend on nw
        other(nw) = [p for p in _memory_stages(s, Nf, M; nw=nw) if !startswith(first(p), "imaginary-k check")]
        @test other(1) == other(7)
    end

    @testset "Beyn retained" begin
        base = with_params(beyn; imag_k_check=false, eigenvectors=false)
        r0 = _memory_retained(base, 3, ns, Nf, M)
        @test length(r0) == 1                                     # boundary points are always retained
        r_imag = _memory_retained(with_params(base; imag_k_check=true), 3, ns, Nf, M)
        r_vec = _memory_retained(with_params(base; eigenvectors=true), 3, ns, Nf, M)
        @test length(r_imag) == 2
        @test length(r_vec) == 3
        @test sum(last, r_vec) > sum(last, r_imag) > sum(last, r0)
        @test sum(last, _memory_retained(base, 6, ns, Nf, M)) > sum(last, r0)    # boundary points scale with the window count
    end

    @testset "CORK" begin
        labels = cp_stage_labels(_memory_stages(cork, Nf, M))
        @test length(labels) == 2
        @test any(startswith("polynomial tuning"), labels)
        @test any(startswith("Krylov solve"), labels)
        @test isempty(_memory_retained(with_params(cork; eigenvectors=false), 3, ns, Nf, M))
        @test !isempty(_memory_retained(with_params(cork; eigenvectors=true), 3, ns, Nf, M))
    end

    @testset "ExpandedBIMSolver" begin
        taylor = with_params(eb; use_taylor=true)
        direct = with_params(eb; use_taylor=false)
        lt = cp_stage_labels(_memory_stages(taylor, Nf, M))
        ld = cp_stage_labels(_memory_stages(direct, Nf, M))
        @test length(lt) == 2
        @test any(startswith("panel assembly"), lt)
        @test any(startswith("Horner evaluation"), lt)
        @test length(ld) == 1
        @test startswith(only(ld), "direct assembly")
        @test isempty(_memory_retained(taylor, 3, ns, Nf, M))
        @test isempty(_memory_retained(direct, 3, ns, Nf, M))
    end

    @testset "VerginiSaraceno" begin
        lp = cp_stage_labels(_memory_stages(vs, pw, Nf, M))
        lt = cp_stage_labels(_memory_stages(vs, tpw, Nf, M))
        @test length(lp) == 1
        @test startswith(only(lp), "window solve")
        @test length(lt) == 2
        @test any(startswith("Taylor validation"), lt)
        @test any(startswith("panel solve"), lt)
        @test isempty(_memory_retained(with_params(vs; eigenvectors=false), pw, ns, Nf))
        @test isempty(_memory_retained(with_params(vs; eigenvectors=false), tpw, ns, Nf))
        for b in (pw, tpw)
            r = _memory_retained(with_params(vs; eigenvectors=true), b, ns, Nf)
            @test length(r) == 1
            @test startswith(only(r).first, "eigenstates")
        end
    end

    @testset "working set is monotone in Nf and M" begin
        for s in (beyn, cork, eb, with_params(beyn; use_taylor=false), with_params(eb; use_taylor=false))
            w = _working_bytes(s, Nf, M)
            @test _working_bytes(s, 2Nf, M) >= w
            @test _working_bytes(s, Nf, 2M) > w
            @test _working_bytes(s, 4Nf, 4M) > w
        end
        for b in (pw, tpw)
            w = _working_bytes(vs, b, Nf, M)
            @test _working_bytes(vs, b, 2Nf, M) > w
            @test _working_bytes(vs, b, Nf, 2M) > w
        end
    end
end

@testset "_format_bytes" begin
    @test _format_bytes(0.0) == "0 B"
    @test _format_bytes(512.0) == "512 B"
    @test _format_bytes(2.0^10 - 1) == "1023 B"
    @test _format_bytes(2.0^10) == "1.0 KiB"
    @test _format_bytes(1.5 * 2.0^10) == "1.5 KiB"
    @test _format_bytes(2.0^20) == "1.0 MiB"
    @test _format_bytes(2.5 * 2.0^20) == "2.5 MiB"
    @test _format_bytes(2.0^30) == "1.0 GiB"
    @test _format_bytes(3.25 * 2.0^30) == "3.25 GiB"
    @test _format_bytes(2.0^40) == "1024.0 GiB"                # no unit above GiB
    @test _format_bytes(2^20) == "1.0 MiB"                      # integer input
    @test _format_bytes(2.0^20 - 2.0^10) isa String
    @test endswith(_format_bytes(2.0^20 - 2.0^10), " KiB")
    @test endswith(_format_bytes(2.0^30 - 2.0^20), " MiB")
    @test sprint(show, MemoryEstimate(2.0^30, 2.0^20)) == "MemoryEstimate(working = 1.0 GiB, persistent = 1.0 MiB)"
end

@testset "_plan_memory_items and the show memory section" begin
    rect = RectangleBilliard(1.0, 0.5)
    kernel = DoubleLayerPotentialSolver(4.0, rect)
    beyn = BeynSolver(kernel; nlevels=20)
    cork = CORKSolver(kernel; nlevels=20)
    eb = ExpandedBIMSolver(kernel)
    vs = VerginiSaracenoSolver(2.0, 5.0)
    pw = RealPlaneWaves(6; use_taylor=false)
    tpw = RealPlaneWaves(6; use_taylor=true, taylor_radius=0.5)

    function check_memory_plan(plan)
        stages, retained = _plan_memory_items(plan)
        @test !isempty(plan.windows)
        @test !isempty(stages)
        @test cp_valid_items(stages)
        @test cp_valid_items(retained)
        @test _stage_peak(stages) == plan.memory.working
        @test sum(last, retained; init=0.0) ≈ plan.memory.persistent
        out = cp_show(plan)
        @test occursin("Memory (upper-bound estimate)", out)
        @test occursin("<- peak", out)
        @test occursin("working peak", out)
        @test occursin("  total           : ", out)
        @test occursin("GC slack", out)
        @test count("<- peak", out) >= 1
        for (label, _) in stages
            @test occursin(label, out)
        end
        if isempty(retained)
            @test !occursin("retained until the sweep ends", out)
        else
            @test occursin("retained until the sweep ends", out)
            for (label, _) in retained
                @test occursin(label, out)
            end
        end
        # the peak marker sits on a stage line that holds the largest stage
        peak_lines = filter(l -> occursin("<- peak", l), split(out, '\n'))
        @test all(any(occursin(first(st), l) for st in stages if last(st) == _stage_peak(stages)) for l in peak_lines)
        return stages, retained
    end

    @testset "Beyn" begin
        for s in (beyn, with_params(beyn; use_taylor=false), with_params(beyn; eigenvectors=false, imag_k_check=false))
            plan = plan_computation(s, rect, 20.0, 40.0; tune=false)
            @test length(plan.windows) > 1
            stages, retained = check_memory_plan(plan)
            @test plan.memory.working == _working_bytes(plan.solver, Float64(maximum(w -> w.dim, plan.windows)), Float64(maximum(w -> w.matrix_dim, plan.windows)); nw=length(plan.windows))
            @test !isempty(retained)                      # boundary points are always retained
        end
        plan = plan_computation(beyn, rect, 20.0, 40.0; tune=false)
        stages, _ = _plan_memory_items(plan)
        @test any(startswith("imaginary-k check"), cp_stage_labels(stages))
    end

    @testset "CORK" begin
        for ev in (true, false)
            plan = plan_computation(with_params(cork; eigenvectors=ev), rect, 20.0, 40.0; tune=false)
            _, retained = check_memory_plan(plan)
            @test isempty(retained) == !ev
        end
    end

    @testset "ExpandedBIMSolver" begin
        for s in (eb, with_params(eb; use_taylor=false))
            plan = plan_computation(s, rect, 20.0, 22.0; tune=false)
            _, retained = check_memory_plan(plan)
            @test isempty(retained)
            @test plan.memory.persistent == 0.0
        end
    end

    @testset "VerginiSaraceno" begin
        for b in (pw, tpw)
            plan = plan_computation(vs, b, rect, 20.0, 22.0, 0.05; tune=false)
            stages, _ = check_memory_plan(plan)
            @test plan.basis === b
            @test (length(stages) == 2) == QuantumBilliards.uses_taylor(b)
            for ev in (true, false)
                p = plan_computation(with_params(vs; eigenvectors=ev), b, rect, 20.0, 22.0, 0.05; tune=false)
                _, retained = check_memory_plan(p)
                @test isempty(retained) == !ev
            end
        end
    end

    @testset "empty window list" begin
        plan = ComputationPlan(beyn, rect, nothing, 20.0, 40.0, nothing, 0.5, 1e15, 5.0, 1.0, 2.0, WindowPlan{Float64}[], MemoryEstimate(2.0^20, 2.0^10), 1, String[])
        out = cp_show(plan)
        @test occursin("Memory (upper-bound estimate)", out)
        @test occursin("  working         : 1.0 MiB", out)
        @test occursin("  retained        : 1.0 KiB", out)
        @test !occursin("<- peak", out)
        @test !occursin("working peak", out)
    end
end

@testset "compute_spectrum(plan, ...) methods" begin
    rect = RectangleBilliard(1.0, 0.5)
    kernel = DoubleLayerPotentialSolver(4.0, rect)
    pw = RealPlaneWaves(6; use_taylor=false)
    vs = VerginiSaracenoSolver(2.0, 5.0)
    for s in (BeynSolver(kernel; nlevels=20), CORKSolver(kernel; nlevels=20), ExpandedBIMSolver(kernel))
        PT = typeof(plan_computation(s, rect, 20.0, 22.0; tune=false))
        @test hasmethod(compute_spectrum, Tuple{PT})
    end
    PV = typeof(plan_computation(vs, pw, rect, 20.0, 22.0, 0.05; tune=false))
    @test hasmethod(compute_spectrum, Tuple{PV})
    @test !hasmethod(compute_spectrum, Tuple{PV,typeof(pw),typeof(rect),Float64})
end

# Tiny end-to-end check: compute_spectrum(plan) forwards exactly the stored inputs.
@testset "compute_spectrum(plan) equals direct call" begin
    rect = RectangleBilliard(1.0, 0.5)
    vs = VerginiSaracenoSolver(2.0, 5.0)
    pw = RealPlaneWaves(6; use_taylor=false)
    k1, k2, dk = 20.0, 20.4, 0.1
    plan = plan_computation(vs, pw, rect, k1, k2, dk; window_overlap=0.25, tune=false)
    via_plan = compute_spectrum(plan; show_progress=false, multithreaded=false)
    direct = compute_spectrum(vs, pw, rect, k1, k2, dk; window_overlap=0.25, show_progress=false, multithreaded=false)
    @test via_plan isa SpectralData
    @test length(direct.k) > 0
    @test via_plan.k == direct.k

    # keyword override of window_overlap changes the window layout, hence is forwarded
    plan_default = plan_computation(vs, pw, rect, k1, k2, dk; tune=false)
    over = compute_spectrum(plan_default; window_overlap=0.25, show_progress=false, multithreaded=false)
    @test over.k == direct.k

    # stored max_memory is enforced, and a keyword overrides it
    tight = plan_computation(vs, pw, rect, k1, k2, dk; tune=false, max_memory=1.0)
    @test_throws ErrorException compute_spectrum(tight; show_progress=false)
    @test_throws ErrorException compute_spectrum(plan_default; max_memory=1.0, show_progress=false)
end
