using BilliardGeometry
using LinearAlgebra: norm
using StaticArrays

# PolygonBilliard: trivial symmetries by default (register_symmetries());
# full_boundary should equal get_boundary_curves (no images appended).
vertices = SVector{2,Float64}[(0.0,0.0), (2.0,0.0), (2.0,1.0), (1.0,1.5), (0.0,1.0)]
billiard = PolygonBilliard(vertices)
curves = full_boundary(billiard)
println("[PolygonBilliard] full_boundary has $(length(curves)) curves (expect $(length(vertices)): no symmetry registered)")
@assert length(curves) == length(vertices) "expected $(length(vertices)) edges, got $(length(curves))"

N = length(curves)
maxgap = 0.0
for i in 1:N
    c = curves[i]
    cnext = curves[mod1(i+1, N)]
    gap = norm(curve(c,1.0) .- curve(cnext,0.0))
    global maxgap = max(maxgap, gap)
end
println("[PolygonBilliard] max endpoint gap = $maxgap")
@assert maxgap < 1e-8 "PolygonBilliard's boundary is not closed/continuous"

println("PASS: PolygonBilliard full_boundary reconstruction OK (trivial symmetry group)")
