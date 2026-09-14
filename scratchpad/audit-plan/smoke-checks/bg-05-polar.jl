using BilliardGeometry
using LinearAlgebra: norm

# PolarBilliard: trivial symmetries (register_symmetries()); fundamental
# curve is already the full closed 2π Fourier-coefficient curve, so
# full_boundary == get_boundary_curves == [segment] (no images appended).
coef = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
billiard = PolarBilliard(coef)
curves = full_boundary(billiard)
println("[PolarBilliard] full_boundary has $(length(curves)) curves (expect 1: no symmetry, already a full closed curve)")
@assert length(curves) == 1 "expected exactly 1 curve, got $(length(curves))"

c = curves[1]
p0 = curve(c, 0.0)
p1 = curve(c, 1.0)
gap = norm(p0 .- p1)
println("[PolarBilliard] self-closure gap (t=0 vs t=1) = $gap")
@assert gap < 1e-8 "PolarBilliard's fundamental curve is not closed"

println("PASS: PolarBilliard full_boundary reconstruction OK (trivial symmetry group, already closed)")
