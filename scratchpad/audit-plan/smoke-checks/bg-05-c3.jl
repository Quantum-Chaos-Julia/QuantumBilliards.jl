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

# C3Billiard: Z3 rotational symmetry (Cn_symmetry(3): NFoldRotation sym_id 1,2).
# full_boundary uses apply_symmetry (has a method for NFoldRotation) via
# _apply_symmetry_to_curve, NOT apply_symmetry_pb -- different code path from
# pb_sectors/pb_coords, checked separately below.
billiard = C3Billiard(0.3)
closed, winding_ok, ncurves = check_full_boundary(billiard; label="C3Billiard")
@assert ncurves == 3 "expected 3 image arcs (one per rotational copy), got $ncurves"
@assert closed "full_boundary not closed/continuous for C3Billiard"
@assert winding_ok "full_boundary self-overlaps or has gaps for C3Billiard"
println("PASS: C3Billiard full_boundary reconstruction OK")

println()
println("--- checking pb_sectors/pb_coords (Husimi-plotting code path, uses apply_symmetry_pb) ---")
try
    sectors = pb_sectors(billiard)
    println("pb_sectors(C3Billiard) succeeded: ", sectors)
catch e
    println("CONFIRMED CRASH: pb_sectors(C3Billiard) errored with: ", sprint(showerror, e))
end
