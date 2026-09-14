# Shared, copy-pasted (not `include`d — each smoke-check script must be
# self-contained) helper used by every bg-05-*.jl script in this directory:
# samples `full_boundary(billiard)` densely and checks
#   (a) continuity  — each curve's t=1 endpoint matches the next curve's t=0 endpoint
#   (b) closure     — the last curve's t=1 endpoint matches the first curve's t=0 endpoint
#   (c) winding ≈ 2π — the sampled point cloud winds around its centroid exactly
#                       once (catches doubled/overlapping symmetry images, which
#                       would wind twice, and gaps/missing images, which wind a
#                       fractional amount).
# Not `include`d by the other scripts on purpose (self-containment requirement);
# kept here only as the canonical copy this was authored against.
using BilliardGeometry

function check_full_boundary(billiard; n_per_curve=200, tol=1e-8, label="billiard")
    curves = full_boundary(billiard)
    println("[$label] full_boundary has $(length(curves)) curves")

    # continuity + closure
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

    # winding number about centroid
    allpts = SVector{2,Float64}[]
    for c in curves
        ts = range(0.0, 1.0, length=n_per_curve)
        pts = curve(c, collect(ts))
        append!(allpts, [SVector{2,Float64}(p) for p in pts])
    end
    cx = sum(p[1] for p in allpts) / length(allpts)
    cy = sum(p[2] for p in allpts) / length(allpts)
    angles = [atan(p[2]-cy, p[1]-cx) for p in allpts]
    total_turn = 0.0
    for i in 1:length(angles)-1
        dtheta = angles[i+1] - angles[i]
        while dtheta > pi
            dtheta -= 2pi
        end
        while dtheta < -pi
            dtheta += 2pi
        end
        total_turn += dtheta
    end
    # close the loop back to the first point too
    dtheta = angles[1] - angles[end]
    while dtheta > pi
        dtheta -= 2pi
    end
    while dtheta < -pi
        dtheta += 2pi
    end
    total_turn += dtheta
    println("[$label] total winding = $(total_turn/(2pi)) turns (expect ≈ 1.0)")
    winding_ok = abs(abs(total_turn) - 2pi) < 0.05
    println("[$label] winding ≈ 1 turn (non-self-overlapping, no gaps): ", winding_ok)

    return closed, winding_ok, length(curves)
end
