using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

function run_high(name, billiard, kern, window, ref_fun; nq=16, r=8, m=6, taylor_degree=8, Rmax=nothing)
    k1, k2 = window
    Rm = Rmax === nothing ? (k2 - k1) / 2 + 0.01 : Rmax
    solver = BeynSolver(kern; m, nq, r, Rmax=Rm, use_taylor=true, use_chebyshev=false, taylor_degree, imag_k_check=false, eigenvectors=false, res_tol=2e-5, svd_tol=1e-10)
    try
        t = @elapsed data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
        ks = sort(real.(data.k))
        s = "$name nq=$nq r=$r deg=$taylor_degree: time=$(round(t,digits=2)) count=$(length(ks))"
        if ref_fun !== nothing
            ref = ref_fun(k1, k2); s *= " ref=$(length(ref))"
            length(ref) == length(ks) && (s *= " max|dk|=$(round(maximum(abs.(ks .- ref)), sigdigits=3))")
        end
        println(s * " max_ten=$(round(maximum(data.ten), sigdigits=3))")
    catch e
        println("$name nq=$nq r=$r deg=$taylor_degree: FAILED ", sprint(showerror, e)[1:min(end, 200)])
    end
end

rect = RectangleBilliard(1.0, 0.8)
kern = CombinedFieldIntegralEquationSolver(8.0, rect; grading=GlobalCornerGrading(kressq=2), min_pts=100, eps=1e-10)
win = high_k_window(3.2); rr = (a, b) -> rectangle_spectrum(2.0, 1.6, a, b)
run_high("warmup", rect, kern, win, rr)
println("#### rect ppw=8 low degree / nq")
for deg in (4, 5, 6, 7)
    run_high("rect", rect, kern, win, rr; taylor_degree=deg)
end
for nq in (8, 10)
    run_high("rect", rect, kern, win, rr; nq, r=8, taylor_degree=8)
end

println("#### rect wider window (nlevels=10), deg 8 vs 12")
win2 = high_k_window(3.2; nlevels=10)
for deg in (8, 12)
    run_high("rect w=$(round(win2[2]-win2[1],digits=3))", rect, kern, win2, rr; taylor_degree=deg, r=16, m=12)
end

println("#### annular composite")
ann = AnnularBilliard(0.3, 0.0)
mk(p) = CompositeBIMSolver((CombinedFieldIntegralEquationSolver(p, ann; grading=SmoothPeriodicGrading(), min_pts=64, eps=1e-10) for _ in 1:2)...)
aw = high_k_window(pi * (1 - 0.09)); ar = (a, b) -> annulus_spectrum(0.3, 1.0, a, b)
for (ppw, deg) in ((4.0, 8), (6.0, 6), (6.0, 8))
    run_high("annular ppw=$ppw", ann, mk(ppw), aw, ar; taylor_degree=deg)
end

println("#### sinai (corners + hole), composite, convergence")
sin_b = SinaiBilliard(1.0, 0.25)
Asin = 4 - pi * 0.25^2
swin = high_k_window(Asin)
nc = length(QuantumBilliards._group_boundary_by_domain_id(full_boundary(sin_b)))
println("components=", nc)
for ppw in (8.0, 12.0)
    ks = CompositeBIMSolver((CombinedFieldIntegralEquationSolver(ppw, sin_b; grading=GlobalCornerGrading(kressq=2), min_pts=64, eps=1e-10) for _ in 1:nc)...)
    run_high("sinai ppw=$ppw", sin_b, ks, swin, nothing)
end
