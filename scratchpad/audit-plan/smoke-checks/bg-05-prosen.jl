using BilliardGeometry
using LinearAlgebra: norm
using StaticArrays

function check_full_boundary(billiard; n_per_curve=200, tol=1e-8, label="billiard")
    curves = full_boundary(billiard)
    println("[$label] full_boundary has $(length(curves)) curves")

    N = length(curves)
    maxgap = 0.0
    for i in 1:N
        c = curves[i]
        cnext = curves[mod1(i+1, N)]
        p_end = curve(c, 1.0)
        p_next_start = curve(cnext, 0.0)
        gap = norm(p_end .- p_next_start)
        maxgap = max(maxgap, gap)
    end
    println("[$label] max endpoint gap across all junctions (incl. closure) = $maxgap")
    closed = maxgap < tol
    println("[$label] closed & continuous: ", closed)

    allpts = SVector{2,Float64}[]
    for c in curves
        ts = collect(range(0.0, 1.0, length=n_per_curve))
        pts = curve(c, ts)
        append!(allpts, [SVector{2,Float64}(p) for p in pts])
    end
    cx = sum(p[1] for p in allpts) / length(allpts)
    cy = sum(p[2] for p in allpts) / length(allpts)
    angles = [atan(p[2]-cy, p[1]-cx) for p in allpts]
    total_turn = 0.0
    for i in 1:length(angles)-1
        dtheta = angles[i+1] - angles[i]
        while dtheta > pi; dtheta -= 2pi; end
        while dtheta < -pi; dtheta += 2pi; end
        total_turn += dtheta
    end
    dtheta = angles[1] - angles[end]
    while dtheta > pi; dtheta -= 2pi; end
    while dtheta < -pi; dtheta += 2pi; end
    total_turn += dtheta
    println("[$label] total winding = $(total_turn/(2pi)) turns (expect ≈ 1.0)")
    winding_ok = abs(abs(total_turn) - 2pi) < 0.05
    println("[$label] winding ≈ 1 turn (non-self-overlapping, no gaps): ", winding_ok)

    return closed, winding_ok, length(curves)
end

# ProsenBilliard: D2 symmetry, r(phi)=1+a*cos(4phi), quadrant fundamental domain.
a = 0.1
billiard = ProsenBilliard(a)
closed, winding_ok, ncurves = check_full_boundary(billiard; label="ProsenBilliard")
@assert ncurves == 4 "expected 4 image arcs (quadrants), got $ncurves"
@assert closed "full_boundary not closed/continuous for ProsenBilliard"
@assert winding_ok "full_boundary self-overlaps or has gaps for ProsenBilliard"

println("PASS: ProsenBilliard full_boundary reconstruction OK")
