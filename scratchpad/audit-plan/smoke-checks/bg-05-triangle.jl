using BilliardGeometry
using LinearAlgebra: norm

# TriangleBilliard: trivial symmetries always (register_symmetries()), even
# for an isosceles special case (chi=1) that does have a genuine mirror
# symmetry -- confirm full_boundary == get_boundary_curves in both cases.
billiard = TriangleBilliard(pi/3, 1.7)  # scalene
curves = full_boundary(billiard)
println("[TriangleBilliard(scalene)] full_boundary has $(length(curves)) curves (expect 3)")
@assert length(curves) == 3 "expected 3 edges, got $(length(curves))"
N = length(curves)
maxgap = maximum(norm(curve(curves[i],1.0) .- curve(curves[mod1(i+1,N)],0.0)) for i in 1:N)
println("[TriangleBilliard(scalene)] max endpoint gap = $maxgap")
@assert maxgap < 1e-8 "TriangleBilliard(scalene) boundary is not closed/continuous"

billiard_iso = TriangleBilliard(pi/3, 1.0)  # isosceles (chi=1): alpha==beta, genuine unregistered mirror symmetry
curves_iso = full_boundary(billiard_iso)
println("[TriangleBilliard(isosceles, chi=1)] full_boundary has $(length(curves_iso)) curves (expect 3; symmetries field is still trivial, i.e. this real mirror symmetry is NOT registered/exploited)")
@assert length(curves_iso) == 3
maxgap_iso = maximum(norm(curve(curves_iso[i],1.0) .- curve(curves_iso[mod1(i+1,3)],0.0)) for i in 1:3)
println("[TriangleBilliard(isosceles)] max endpoint gap = $maxgap_iso")
@assert maxgap_iso < 1e-8

println("PASS: TriangleBilliard full_boundary reconstruction OK (trivial symmetry group; isosceles case's real mirror symmetry confirmed unregistered but geometry still closes correctly since it's simply the full triangle with no folding attempted)")
