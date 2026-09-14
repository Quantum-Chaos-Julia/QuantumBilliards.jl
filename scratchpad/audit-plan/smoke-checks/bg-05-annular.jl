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
    return closed, length(curves)
end

# AnnularBilliard: trivial symmetries (register_symmetries()), multiply
# connected (two disjoint curves: outer + inner circle, different domain_id).
# full_boundary/get_boundary_curves' single-chain "closed & continuous" check
# does NOT apply naturally here -- there are two separate closed loops, so we
# check each curve individually (a CircleSegment with arc_angle=2π is already
# closed on its own) rather than chaining across curves like the other scripts.
R_outer, R_inner = 2.0, 1.0
billiard = AnnularBilliard(R_outer, R_inner)
curves = full_boundary(billiard)
println("[AnnularBilliard] full_boundary has $(length(curves)) curves (expect 2: no symmetry registered, so == get_boundary_curves)")
@assert length(curves) == 2 "expected exactly 2 curves (outer+inner circle, no symmetry images), got $(length(curves))"

for (i,c) in enumerate(curves)
    p0 = curve(c, 0.0)
    p1 = curve(c, 1.0)
    gap = norm(p0 .- p1)
    println("[AnnularBilliard] curve $i (domain_id=$(c.domain_id)) self-closure gap (t=0 vs t=1, full 2π arc) = $gap")
    @assert gap < 1e-8 "curve $i is not individually closed"
end

println("PASS: AnnularBilliard full_boundary reconstruction OK (2 independently-closed rings; no symmetry to fold, consistent with register_symmetries() being trivial)")
