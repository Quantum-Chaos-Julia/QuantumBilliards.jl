using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

function report(name, data, ref)
    ks = sort(data.k)
    println("== ", name)
    println("   count=", length(ks), " ref=", length(ref))
    if length(ks) == length(ref)
        println("   max|dk|=", maximum(abs.(ks .- ref)), "  max ten=", maximum(data.ten), "  max|k_im|=", maximum(abs.(data.k_im)))
    else
        println("   k   =", round.(ks, digits=5))
        println("   ref =", round.(ref, digits=5))
    end
end

rect = RectangleBilliard(1.0, 0.7)
k1, k2 = 3.0, 10.0
ref = rectangle_spectrum(2.0, 1.4, k1, k2)
kernel = CombinedFieldIntegralEquationSolver(10.0, rect; grading=GlobalCornerGrading())
solver = BeynSolver(kernel; use_chebyshev=false)
t = @elapsed data = compute_spectrum(solver, rect, k1, k2; multithreaded=false, show_progress=false)
println("Beyn rect time=", t)
report("Beyn CFIE rect [3,10]", data, ref)

circ = CircleBilliard(1.0)
k1, k2 = 2.0, 8.0
ref = circle_spectrum(1.0, k1, k2)
kernel = CombinedFieldIntegralEquationSolver(10.0, circ; grading=SmoothPeriodicGrading())
solver = BeynSolver(kernel; use_chebyshev=false)
t = @elapsed data = compute_spectrum(solver, circ, k1, k2; multithreaded=false, show_progress=false)
println("Beyn circle time=", t)
report("Beyn CFIE circle [2,8]", data, ref)
