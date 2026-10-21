using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

function run_high(name, billiard, kern, window, ref_fun; nq=24, r=16, m=6, taylor_degree=14, Rmax=nothing, quiet=false)
    k1, k2 = window
    Rm = Rmax === nothing ? (k2 - k1) / 2 + 0.01 : Rmax
    solver = BeynSolver(kern; m, nq, r, Rmax=Rm, use_taylor=true, use_chebyshev=false, taylor_degree, imag_k_check=false, eigenvectors=false, res_tol=2e-5, svd_tol=1e-10)
    try
        t = @elapsed data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
        ks = sort(real.(data.k))
        s = "$name nq=$nq r=$r m=$m deg=$taylor_degree window=($(round(k1,digits=4)),$(round(k2,digits=4))): time=$(round(t,digits=2)) count=$(length(ks))"
        if ref_fun !== nothing
            ref = ref_fun(k1, k2)
            s *= " ref=$(length(ref))"
            length(ref) == length(ks) && (s *= " max|dk|=$(round(maximum(abs.(ks .- ref)), sigdigits=3))")
        end
        s *= " max_ten=$(round(maximum(data.ten), sigdigits=3))"
        quiet || println(s)
        return s, ks
    catch e
        println("$name: FAILED ", sprint(showerror, e)[1:min(end, 200)])
        return nothing, nothing
    end
end

rect = RectangleBilliard(1.0, 0.8)
kern = CombinedFieldIntegralEquationSolver(8.0, rect; grading=GlobalCornerGrading(kressq=2), min_pts=100, eps=1e-10)
win = high_k_window(3.2)
rr = (a, b) -> rectangle_spectrum(2.0, 1.6, a, b)
run_high("warmup", rect, kern, win, rr; quiet=true)
println("#### rect ppw=8: nq / r / degree sweep")
for nq in (12, 16, 24)
    run_high("rect", rect, kern, win, rr; nq)
end
for r in (6, 8, 12)
    run_high("rect", rect, kern, win, rr; nq=16, r)
end
for deg in (8, 10, 12)
    run_high("rect", rect, kern, win, rr; nq=16, r=8, taylor_degree=deg)
end

println("#### circle, explicit window (100.19, 100.25), 10 levels")
circ = CircleBilliard(1.0)
cw = (100.19, 100.25)
cr = (a, b) -> circle_spectrum(1.0, a, b)
for ppw in (4.0, 6.0, 8.0)
    kc = CombinedFieldIntegralEquationSolver(ppw, circ; grading=SmoothPeriodicGrading(), min_pts=64, eps=1e-10)
    println("ppw=$ppw")
    run_high("circle", circ, kc, cw, cr; nq=16, r=16)
end

println("#### stadium convergence")
stad = StadiumBilliard(0.3)
sw = high_k_window(pi + 1.2)
prev = Ref{Any}(nothing)
for ppw in (6.0, 8.0, 12.0)
    ks = CombinedFieldIntegralEquationSolver(ppw, stad; grading=SmoothPeriodicGrading(), min_pts=64, eps=1e-10)
    s, k = run_high("stadium ppw=$ppw", stad, ks, sw, nothing; nq=16, r=12)
    if k !== nothing && prev[] !== nothing && length(prev[]) == length(k)
        println("   vs previous: max|dk|=", maximum(abs.(k .- prev[])))
    end
    k === nothing || (prev[] = k)
end
