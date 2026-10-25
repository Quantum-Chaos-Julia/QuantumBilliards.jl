using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

rect = RectangleBilliard(1.0, 0.7)
circ = CircleBilliard(1.0)

function table(data, ref)
    p = sortperm(data.k)
    for i in p
        d = minimum(abs.(ref .- data.k[i]))
        println("   k=", round(data.k[i], digits=7), "  ten=", round(data.ten[i], sigdigits=3), "  ctrl=", data.control[i], "  dist_to_ref=", round(d, sigdigits=3))
    end
end

println("== EBIM rect: levels with tension and match to ref")
k1, k2 = 3.0, 10.0
ref = rectangle_spectrum(2.0, 1.4, k1, k2)
d = compute_spectrum(ExpandedBIMSolver(CombinedFieldIntegralEquationSolver(10.0, rect; grading=GlobalCornerGrading())), rect, k1, k2; multithreaded=false, show_progress=false)
table(d, ref)

println("== EBIM circle")
ref = circle_spectrum(1.0, 2.0, 8.0)
d = compute_spectrum(ExpandedBIMSolver(CombinedFieldIntegralEquationSolver(10.0, circ; grading=SmoothPeriodicGrading())), circ, 2.0, 8.0; multithreaded=false, show_progress=false)
table(d, ref)

for nl in (10, 20, 40)
    println("== CORK rect nlevels=$nl")
    ref = rectangle_spectrum(2.0, 1.4, 3.0, 10.0)
    try
        t = @elapsed d = compute_spectrum(CORKSolver(CombinedFieldIntegralEquationSolver(10.0, rect; grading=GlobalCornerGrading()); nlevels=nl), rect, 3.0, 10.0; multithreaded=false, show_progress=false)
        println("   time=", round(t, digits=2), " count=", length(d.k), " ref=", length(ref))
        length(d.k) == length(ref) && println("   max|dk|=", maximum(abs.(sort(d.k) .- ref)), " ten=", maximum(d.ten), " k_im=", maximum(abs.(d.k_im)))
    catch e
        println("   FAILED: ", sprint(showerror, e)[1:min(end, 300)])
    end
end
println("== CORK circle nlevels=10")
try
    ref = circle_spectrum(1.0, 2.0, 8.0)
    t = @elapsed d = compute_spectrum(CORKSolver(CombinedFieldIntegralEquationSolver(10.0, circ; grading=SmoothPeriodicGrading()); nlevels=10), circ, 2.0, 8.0; multithreaded=false, show_progress=false)
    println("   time=", round(t, digits=2), " count=", length(d.k), " ref=", length(ref))
    length(d.k) == length(ref) && println("   max|dk|=", maximum(abs.(sort(d.k) .- ref)), " ten=", maximum(d.ten), " k_im=", maximum(abs.(d.k_im)))
catch e
    println("   FAILED: ", sprint(showerror, e)[1:min(end, 300)])
end
