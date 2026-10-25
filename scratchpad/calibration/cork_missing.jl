using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

rect = RectangleBilliard(1.0, 0.7)
circ = CircleBilliard(1.0)

function compare(d, ref)
    ks = sort(real.(d.k))
    println("   got=", round.(ks, digits=5))
    println("   ref=", round.(ref, digits=5))
    println("   missing=", round.([r for r in ref if minimum(abs.(ks .- r)) > 1e-4], digits=5))
    println("   extra  =", round.([k for k in ks if minimum(abs.(ref .- k)) > 1e-4], digits=5))
    println("   max|k_im|=", maximum(abs.(imag.(d.k))), " ten max=", maximum(d.ten))
end

kern = CombinedFieldIntegralEquationSolver(20.0, rect; grading=GlobalCornerGrading(), min_pts=800)
for (k1, k2, nl) in ((3.0, 10.0, 20), (3.0, 6.0, 20), (6.0, 10.0, 20), (3.0, 10.0, 40))
    println("CORK rect [$k1,$k2] nlevels=$nl")
    ref = rectangle_spectrum(2.0, 1.4, k1, k2)
    try
        d = compute_spectrum(CORKSolver(kern; nlevels=nl), rect, k1, k2; multithreaded=false, show_progress=false)
        compare(d, ref)
    catch e
        println("   FAILED ", sprint(showerror, e)[1:200])
    end
end
println("windows plan:")
println(plan_weyl_windows(rect, 3.0, 10.0; m=20, Rmax=0.6, fundamental=false))
println(plan_weyl_windows(rect, 3.0, 10.0; m=10, Rmax=0.6, fundamental=false))
