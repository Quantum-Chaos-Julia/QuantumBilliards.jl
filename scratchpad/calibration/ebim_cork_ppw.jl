using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

rect = RectangleBilliard(1.0, 0.7)
circ = CircleBilliard(1.0)

function summarize(tag, d, ref; tenmax=0.1)
    keep = findall(<(tenmax), d.ten)
    ks = sort(d.k[keep])
    nsp = length(d.k) - length(keep)
    line = "   $tag: total=$(length(d.k)) kept=$(length(ks)) spurious(ten>=$tenmax)=$nsp ref=$(length(ref))"
    if length(ks) == length(ref)
        line *= " max|dk|=" * string(round(maximum(abs.(ks .- ref)), sigdigits=3)) * " max_ten_kept=" * string(round(maximum(d.ten[keep]), sigdigits=3))
    end
    println(line)
end

function try_run(f)
    try
        t = @elapsed r = f()
        println("   time=", round(t, digits=2))
        return r
    catch e
        println("   FAILED: ", sprint(showerror, e)[1:min(end, 250)])
        return nothing
    end
end

println("##### EBIM, rect [3,10], varying ppw / min_pts")
ref = rectangle_spectrum(2.0, 1.4, 3.0, 10.0)
for (ppw, mp) in ((10.0, 200), (20.0, 400), (30.0, 600))
    println("ppw=$ppw min_pts=$mp")
    kern = CombinedFieldIntegralEquationSolver(ppw, rect; grading=GlobalCornerGrading(), min_pts=mp)
    d = try_run(() -> compute_spectrum(ExpandedBIMSolver(kern), rect, 3.0, 10.0; multithreaded=false, show_progress=false))
    d === nothing || summarize("EBIM", d, ref)
end

println("##### EBIM, circle [2,8]")
ref = circle_spectrum(1.0, 2.0, 8.0)
for (ppw, mp) in ((10.0, 200), (20.0, 400))
    println("ppw=$ppw min_pts=$mp")
    kern = CombinedFieldIntegralEquationSolver(ppw, circ; grading=SmoothPeriodicGrading(), min_pts=mp)
    d = try_run(() -> compute_spectrum(ExpandedBIMSolver(kern), circ, 2.0, 8.0; multithreaded=false, show_progress=false))
    d === nothing || summarize("EBIM", d, ref)
end

println("##### CORK, rect [3,10]")
ref = rectangle_spectrum(2.0, 1.4, 3.0, 10.0)
for (ppw, mp, nl) in ((10.0, 400, 10), (20.0, 800, 10), (20.0, 800, 20), (40.0, 1000, 20))
    println("ppw=$ppw min_pts=$mp nlevels=$nl")
    kern = CombinedFieldIntegralEquationSolver(ppw, rect; grading=GlobalCornerGrading(), min_pts=mp)
    d = try_run(() -> compute_spectrum(CORKSolver(kern; nlevels=nl), rect, 3.0, 10.0; multithreaded=false, show_progress=false))
    d === nothing || summarize("CORK", d, ref; tenmax=1e-3)
end
