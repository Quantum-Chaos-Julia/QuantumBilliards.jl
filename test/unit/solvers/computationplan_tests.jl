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
