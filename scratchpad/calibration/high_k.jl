using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

function run_high(name, billiard, mkkernel, A, ref_fun; ppw, min_pts=100, nq=24, r=16, m=6, taylor_degree=14, Rmax=nothing, k=HIGH_K, nlevels=HIGH_K_LEVELS)
    k1, k2 = high_k_window(A; k, nlevels)
    Rm = Rmax === nothing ? (k2 - k1) / 2 + 0.01 : Rmax
    kern = mkkernel(ppw, min_pts)
    solver = BeynSolver(kern; m, nq, r, Rmax=Rm, use_taylor=true, use_chebyshev=false, taylor_degree, imag_k_check=false, eigenvectors=false, res_tol=2e-5, svd_tol=1e-10)
    try
        t = @elapsed data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
        ks = sort(real.(data.k))
        ref = ref_fun === nothing ? nothing : ref_fun(k1, k2)
        s = "$name ppw=$ppw min_pts=$min_pts nq=$nq r=$r m=$m deg=$taylor_degree window=($(round(k1,digits=4)),$(round(k2,digits=4))): time=$(round(t,digits=2)) count=$(length(ks))"
        if ref !== nothing
            s *= " ref=$(length(ref))"
            length(ref) == length(ks) && (s *= " max|dk|=$(round(maximum(abs.(ks .- ref)), sigdigits=3))")
        end
        s *= " max_ten=$(round(maximum(data.ten), sigdigits=3))"
        println(s)
        return ks
    catch e
        println("$name ppw=$ppw: FAILED ", sprint(showerror, e)[1:min(end, 200)])
        return nothing
    end
end

println("#### circle (smooth, analytic)")
circ = CircleBilliard(1.0)
for ppw in (4.0, 6.0, 8.0, 12.0)
    run_high("circle", circ, (p, mp) -> CombinedFieldIntegralEquationSolver(p, circ; grading=SmoothPeriodicGrading(), min_pts=mp, eps=1e-10), pi, (a, b) -> circle_spectrum(1.0, a, b); ppw)
end

println("#### rectangle 2x1.6 (corners, analytic)")
rect = RectangleBilliard(1.0, 0.8)
for ppw in (6.0, 8.0, 12.0, 16.0)
    run_high("rect", rect, (p, mp) -> CombinedFieldIntegralEquationSolver(p, rect; grading=GlobalCornerGrading(kressq=2), min_pts=mp, eps=1e-10), 3.2, (a, b) -> rectangle_spectrum(2.0, 1.6, a, b); ppw)
end

println("#### annular composite TAYLOR (analytic)")
ann = AnnularBilliard(0.3, 0.0)
Aann = pi * (1 - 0.09)
for ppw in (6.0, 8.0, 12.0)
    mk(p, mp) = CompositeBIMSolver((CombinedFieldIntegralEquationSolver(p, ann; grading=SmoothPeriodicGrading(), min_pts=mp, eps=1e-10) for _ in 1:2)...)
    run_high("annular", ann, mk, Aann, (a, b) -> annulus_spectrum(0.3, 1.0, a, b); ppw, min_pts=64)
end

println("#### stadium (smooth, convergence)")
stad = StadiumBilliard(0.3)
Astad = pi + 1.2
prev = nothing
for ppw in (6.0, 8.0, 12.0)
    ks = run_high("stadium", stad, (p, mp) -> CombinedFieldIntegralEquationSolver(p, stad; grading=SmoothPeriodicGrading(), min_pts=mp, eps=1e-10), Astad, nothing; ppw)
    if ks !== nothing && prev !== nothing && length(prev) == length(ks)
        println("   vs previous ppw: max|dk|=", maximum(abs.(ks .- prev)))
    end
    ks === nothing || (prev = ks)
end
