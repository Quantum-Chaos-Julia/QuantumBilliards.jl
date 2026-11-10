using QuantumBilliards, BilliardGeometry
using CairoMakie, QBPlotting, QCPlotting, BilliardGeometryPlotting, LaTeXStrings
using DataFrames, Arrow
using TimerOutputs,StaticArrays
try_MKL!()

gamma = pi*(sqrt(5)+1)/4
chi = (1+sqrt(2))/2
triangle = TriangleBilliard(gamma, chi)

triangle_vs, basis = make_triangle_and_basis(gamma, chi)
triangle_vs
basis
ppw = 8.0
kernel = DoubleLayerPotentialSolver(
    ppw,
    triangle;
    grading = GlobalCornerGrading(kressq = 4, min_t_spacing = 1e-14)
)

kernel_cork = DoubleLayerPotentialSolver(16.0, triangle; grading = GlobalCornerGrading(kressq = 4, min_t_spacing = 1e-14))
vs_solver = VerginiSaracenoSolver(4.0, ppw; eigenvectors=true)

beyn_cheb   = BeynSolver(kernel; use_taylor=false, use_chebyshev=true, eigenvectors=true)
beyn_taylor = BeynSolver(kernel; use_taylor=true,  use_chebyshev=false, eigenvectors=true)
beyn_direct = BeynSolver(kernel; use_taylor=false, use_chebyshev=false, eigenvectors=true)
cork        = CORKSolver(kernel_cork; eigenvectors=true)
ebim_taylor = ExpandedBIMSolver(kernel; use_taylor=true)
ebim_direct = ExpandedBIMSolver(kernel; use_taylor=false)

cases = [
    "Vergini-Saraceno" => (N1, N2) -> plan_computation(vs_solver, basis, triangle_vs, N1, N2; nlevels = 2.0),
    "Beyn-Chebyshev"   => (N1, N2) -> plan_computation(beyn_cheb, triangle, N1, N2; Rmax = 0.5, nlevels = 25),
    "Beyn-Taylor"      => (N1, N2) -> plan_computation(beyn_taylor, triangle, N1, N2; Rmax = 0.5, nlevels = 25),
    "Beyn-Direct"      => (N1, N2) -> plan_computation(beyn_direct, triangle, N1, N2; Rmax = 0.5, nlevels = 25),
    "CORK"             => (N1, N2) -> plan_computation(cork, triangle, N1, N2; Rmax = 0.4, nlevels = 25),
    "EBIM-Taylor"      => (N1, N2) -> plan_computation(ebim_taylor, triangle, N1, N2; nlevels = 2.0),
    "EBIM-Direct"      => (N1, N2) -> plan_computation(ebim_direct, triangle, N1, N2; nlevels = 2.0),
]


#ENV["JULIA_DEBUG"] = "QuantumBilliards"
TimerOutputs.enable_debug_timings(QuantumBilliards)

logpath = joinpath(@__DIR__, "triangle_benchmark_timers.log")

function run_case(io, name, N1, N2, make_plan)
    header = "\n=== $name, states $N1-$N2 ==="
    println(header)
    println(io, header)
    try
        plan = make_plan(N1, N2)
        plan_text = sprint(show, MIME("text/plain"), plan)
        println(plan_text)
        println(io, plan_text)
        flush(io)
        reset_timer!()
        spect = compute_spectrum(plan; multithreaded=true)
        msg = "States found: $(length(spect))"
        println(msg)
        println(io, msg)
        print_timer(; allocations=true)
        print_timer(io; allocations=true)
    catch e
        e isa InterruptException && rethrow()
        msg = sprint(showerror, e, catch_backtrace())
        println(stderr, msg)
        println(io, "Error: ", msg)
    end
    flush(io)
end

open(logpath, "w") do io
    for (N1, N2) in [(1000, 1050), (5000, 5050), (10000, 10050), (50000, 50050)]
        for (name, make_plan) in cases
            run_case(io, name, N1, N2, make_plan)
        end
    end
end
