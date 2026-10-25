using Test, QuantumBilliards, BilliardGeometry
include(joinpath(@__DIR__, "..", "..", "test", "integration", "common.jl"))

macro probe(name, ex)
    quote
        println("== ", $(esc(name)))
        try
            t = @elapsed r = $(esc(ex))
            println("   time=", round(t, digits=2))
            r
        catch e
            println("   FAILED: ", sprint(showerror, e)[1:min(end, 400)])
            nothing
        end
    end
end

function report(data, ref)
    ks = sort(data.k)
    println("   count=", length(ks), " ref=", length(ref))
    if length(ks) == length(ref)
        println("   max|dk|=", maximum(abs.(ks .- ref)), "  max ten=", maximum(data.ten), "  max|k_im|=", maximum(abs.(data.k_im)))
    else
        println("   k   =", round.(ks, digits=5))
        println("   ref =", round.(ref, digits=5))
    end
end

rect = RectangleBilliard(1.0, 0.7)
circ = CircleBilliard(1.0)

# EBIM
k1, k2 = 3.0, 10.0
ref = rectangle_spectrum(2.0, 1.4, k1, k2)
d = @probe "EBIM CFIE rect [3,10]" compute_spectrum(ExpandedBIMSolver(CombinedFieldIntegralEquationSolver(10.0, rect; grading=GlobalCornerGrading())), rect, k1, k2; multithreaded=false, show_progress=false)
d === nothing || report(d, ref)
k1, k2 = 2.0, 8.0
ref = circle_spectrum(1.0, k1, k2)
d = @probe "EBIM CFIE circle [2,8]" compute_spectrum(ExpandedBIMSolver(CombinedFieldIntegralEquationSolver(10.0, circ; grading=SmoothPeriodicGrading())), circ, k1, k2; multithreaded=false, show_progress=false)
d === nothing || report(d, ref)

# CORK
k1, k2 = 3.0, 10.0
ref = rectangle_spectrum(2.0, 1.4, k1, k2)
d = @probe "CORK CFIE rect [3,10]" compute_spectrum(CORKSolver(CombinedFieldIntegralEquationSolver(10.0, rect; grading=GlobalCornerGrading())), rect, k1, k2; multithreaded=false, show_progress=false)
d === nothing || report(d, ref)
k1, k2 = 2.0, 8.0
ref = circle_spectrum(1.0, k1, k2)
d = @probe "CORK CFIE circle [2,8]" compute_spectrum(CORKSolver(CombinedFieldIntegralEquationSolver(10.0, circ; grading=SmoothPeriodicGrading())), circ, k1, k2; multithreaded=false, show_progress=false)
d === nothing || report(d, ref)

# Sweep BIM, solve_wavenumber
kref = rectangle_spectrum(2.0, 1.4, 0.0, 6.0)[1]
for (nm, ctor) in (("DLP", DoubleLayerPotentialSolver), ("CFIE", CombinedFieldIntegralEquationSolver))
    s = ctor(10.0, rect; grading=GlobalCornerGrading())
    r = @probe "$nm rect ground (kref=$kref)" solve_wavenumber(s, rect, kref + 0.03, 0.1)
    r === nothing || println("   |dk|=", abs(r[1] - kref), " ten=", r[2])
    s = ctor(10.0, circ; grading=SmoothPeriodicGrading())
    r = @probe "$nm circle j01" solve_wavenumber(s, circ, 2.4, 0.2)
    r === nothing || println("   |dk|=", abs(r[1] - 2.404825557695773), " ten=", r[2])
    r = @probe "$nm circle j02" solve_wavenumber(s, circ, 5.52, 0.2)
    r === nothing || println("   |dk|=", abs(r[1] - 5.520078110286311), " ten=", r[2])
end

# Composite
ann = AnnularBilliard(0.3, 0.0)
mk() = CombinedFieldIntegralEquationSolver(10.0, ann; grading=SmoothPeriodicGrading(), min_pts=64)
comp = CompositeBIMSolver(mk(), mk())
aref = annulus_spectrum(0.3, 1.0, 0.0, 6.0)
println("annulus ref first: ", aref[1:3])
r = @probe "Composite CFIE annulus ground" solve_wavenumber(comp, ann, aref[1] + 0.03, 0.1)
r === nothing || println("   |dk|=", abs(r[1] - aref[1]), " ten=", r[2])
d = @probe "Beyn composite direct annulus [3,8]" compute_spectrum(BeynSolver(comp; use_taylor=false, use_chebyshev=false), ann, 3.0, 8.0; multithreaded=false, show_progress=false)
d === nothing || report(d, annulus_spectrum(0.3, 1.0, 3.0, 8.0))
d = @probe "Beyn composite TAYLOR annulus [3,8]" compute_spectrum(BeynSolver(comp; use_taylor=true, use_chebyshev=false), ann, 3.0, 8.0; multithreaded=false, show_progress=false)
d === nothing || report(d, annulus_spectrum(0.3, 1.0, 3.0, 8.0))
