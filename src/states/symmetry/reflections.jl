"""
    apply_symmetries_to_wavefunction(Psi, x_grid::AbstractVector, y_grid::AbstractVector, symmetries::Vector{<:BilliardGeometry.AbsReflection}, sym_qnumbers::Vector{T}) where {T<:Real} → (full_Psi, full_x, full_y)

Unfold a wavefunction from a reflection-reduced fundamental-domain grid onto
the corresponding full billiard grid.

`Psi` is assumed to be sampled on the `x > 0`, `y > 0` fundamental quadrant,
with its first dimension corresponding to `x_grid` and its second dimension to
`y_grid`. Each registered `XAxisReflection` or `YAxisReflection` generates the
corresponding reflected copy, multiplied by its parity `±1` from
`sym_qnumbers`. With both axis reflections present, the four quadrants are
assembled in the order implied by

    Q1:  x > 0, y > 0    base
    Q2:  x < 0, y > 0    Y-reflected
    Q3:  x < 0, y < 0    XY-reflected
    Q4:  x > 0, y < 0    X-reflected.

If only one axis reflection is present, the grid is unfolded across that axis.
If no supported axis reflection is present, the input wavefunction and grids
are returned unchanged.

## Arguments
* `Psi`: Wavefunction values on the fundamental-domain grid.
* `x_grid::AbstractVector`: Coordinates associated with the first dimension of `Psi`.
* `y_grid::AbstractVector`: Coordinates associated with the second dimension of `Psi`.
* `symmetries::Vector{<:BilliardGeometry.AbsReflection}`: Reflection symmetries used for unfolding.
* `sym_qnumbers::Vector{T}`: Parities `±1` associated with the entries of `symmetries`.

## Returns
* `full_Psi`: Wavefunction values on the unfolded grid.
* `full_x::Vector`: Unfolded `x` coordinates.
* `full_y::Vector`: Unfolded `y` coordinates.
"""
function apply_symmetries_to_wavefunction(Psi, x_grid, y_grid, symmetries::Vector{<:BilliardGeometry.AbsReflection}, sym_qnumbers::Vector{T}) where T<:Real
    has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
    has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
    get_qnum(::Type{S}) where S = sym_qnumbers[findfirst(s -> s isa S, symmetries)]
    if has_x && has_y
        pX = get_qnum(BilliardGeometry.XAxisReflection)
        pY = get_qnum(BilliardGeometry.YAxisReflection)
        # dim1=x, dim2=y, Q1 is base (x>0, y>0)
        Psi_Y  = reverse(pY .* Psi,            dims=1)    # Q2: flip x
        Psi_X  = reverse(pX .* Psi,            dims=2)    # Q4: flip y  
        Psi_XY = reverse(pX .* pY .* Psi, dims=(1,2))     # Q3: flip both
        # hcat joins along dim2 (y): left=y<0, right=y>0
        # vcat joins along dim1 (x): top=x<0, bottom=x>0
        left  = vcat(Psi_XY, Psi_X)   # x: [-x;+x], y<0:  Q3 on top, Q4 on bottom
        right = vcat(Psi_Y,  Psi)     # x: [-x;+x], y>0:  Q2 on top, Q1 on bottom
        full_Psi = hcat(left, right)  # join y halves: [y<0 | y>0]
        full_x = vcat(-reverse(x_grid), x_grid)
        full_y = vcat(-reverse(y_grid), y_grid)
        return full_Psi, full_x, full_y
    elseif has_y
        # YAxisReflection: mirrors x (dim1)
        p = get_qnum(BilliardGeometry.YAxisReflection)
        Psi_ref = reverse(p .* Psi, dims=1)
        return hcat(Psi_ref, Psi), vcat(-reverse(x_grid), x_grid), y_grid
    elseif has_x
        # XAxisReflection: mirrors y (dim2)
        p = get_qnum(BilliardGeometry.XAxisReflection)
        Psi_ref = reverse(p .* Psi, dims=2)
        return vcat(Psi_ref, Psi), x_grid, vcat(-reverse(y_grid), y_grid)
    else
        return Psi, x_grid, y_grid
    end
end


"""
    apply_symmetries_to_boundary_function(u::AbstractVector{U}, symmetries::Vector{<:BilliardGeometry.AbsReflection}, sym_qnumbers::Vector{T}) where {U<:Number,T<:Real} → full_u

Unfold a boundary function from a reflection-reduced fundamental boundary
onto the corresponding full physical boundary.

Each reflected copy is multiplied by the parity associated with its reflection.
Copies produced by a single axis reflection are also reversed so that their
ordering follows the counter-clockwise boundary traversal used by
[`apply_symmetries_to_boundary_points`](@ref). With both axis reflections
present, the copies are appended in the order

    base → Y-reflected → XY-reflected → X-reflected,

where the `Y`- and `X`-reflected copies are reversed and the `XY`-reflected
copy preserves the original ordering. If no reflections are supplied, `u` is
returned unchanged.

## Arguments
* `u::AbstractVector{U}`: Boundary-function values on the fundamental boundary.
* `symmetries::Vector{<:BilliardGeometry.AbsReflection}`: Reflection symmetries used for unfolding.
* `sym_qnumbers::Vector{T}`: Parities `±1` associated with the entries of `symmetries`.

## Returns
* `full_u::Vector{U}`: Boundary function unfolded onto the full physical boundary.
"""
function apply_symmetries_to_boundary_function(u::AbstractVector{U}, symmetries::Vector{<:BilliardGeometry.AbsReflection}, sym_qnumbers::Vector{T}) where {U<:Number, T<:Real}
    isempty(symmetries) && return u
    base_u = copy(u)
    full_u = copy(u)
    has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
    has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
    if has_x && has_y
        pY = sym_qnumbers[findfirst(s -> s isa BilliardGeometry.YAxisReflection, symmetries)]
        pX = sym_qnumbers[findfirst(s -> s isa BilliardGeometry.XAxisReflection, symmetries)]
        # CCW order must match apply_symmetries_to_boundary_points exactly:
        # Q1(base) → Q2(Y-reflect, reversed) → Q3(XY-reflect) → Q4(X-reflect, reversed)
        uY  =  pY      .* reverse(base_u)  # Q2: reversed
        uXY = (pX * pY) .*        base_u   # Q3: not reversed
        uX  =  pX      .* reverse(base_u)  # Q4: reversed
        append!(full_u, uY)
        append!(full_u, uXY)
        append!(full_u, uX)
    elseif has_y
        pY = sym_qnumbers[findfirst(s -> s isa BilliardGeometry.YAxisReflection, symmetries)]
        append!(full_u, pY .* reverse(base_u))
    elseif has_x
        pX = sym_qnumbers[findfirst(s -> s isa BilliardGeometry.XAxisReflection, symmetries)]
        append!(full_u, pX .* reverse(base_u))
    end
    return full_u
end


"""
    apply_symmetries_to_boundary_points(pts::BoundaryPoints{T}, symmetries::Vector{BilliardGeometry.AbsReflection}, billiard::Bi) where {Bi<:AbsBilliard,T<:Real} → full_pts::BoundaryPoints{T}

Unfold boundary points from a reflection-reduced fundamental boundary onto the
corresponding full physical boundary.

Positions and normals are transformed with [`apply_symmetry`](@ref), while
point ordering and quadrature elements are reversed whenever required to
preserve a continuous counter-clockwise traversal. With both axis reflections
present, copies are appended in the order

    base → Y-reflected → XY-reflected → X-reflected,

with the `Y`- and `X`-reflected copies reversed and the `XY`-reflected copy
preserving the original ordering. With a single axis reflection, only its
reflected and reversed copy is appended.

The cumulative boundary coordinate `s` is reconstructed from the unfolded
`ds`. Other optional [`BoundaryPoints`](@ref) fields are not propagated. If no
reflections are supplied, `pts` is returned unchanged.

## Arguments
* `pts::BoundaryPoints{T}`: Boundary points sampled on the fundamental boundary.
* `symmetries::Vector{BilliardGeometry.AbsReflection}`: Reflection symmetries used for unfolding.
* `billiard::Bi`: Associated billiard; retained for interface consistency.

## Returns
* `full_pts::BoundaryPoints{T}`: Boundary points, normals, quadrature elements, and cumulative boundary coordinate on the full physical boundary.
"""
function apply_symmetries_to_boundary_points(pts::BoundaryPoints{T}, symmetries::Vector{BilliardGeometry.AbsReflection}, billiard::Bi) where {Bi<:AbsBilliard, T<:Real}
    isempty(symmetries) && return pts
    bxy    = pts.xy
    bn     = pts.normal
    bds    = pts.ds
    has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
    has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
    copies = 1 + has_x + has_y + (has_x & has_y)
    full_xy     = copy(bxy)
    full_normal = copy(bn)
    full_ds     = copy(bds)
    sizehint!(full_xy,     length(bxy) * copies)
    sizehint!(full_normal, length(bn)  * copies)
    sizehint!(full_ds,     length(bds) * copies)
    get_sym(::Type{S}) where S = symmetries[findfirst(s -> s isa S, symmetries)]
    @inline function push_reflection!(sym::BilliardGeometry.AbsReflection, reverse_orientation::Bool)
        rxy = apply_symmetry(sym, bxy)
        rn  = apply_symmetry(sym, bn)
        if reverse_orientation
            append!(full_xy,     reverse(rxy))
            append!(full_normal, reverse(rn))
            append!(full_ds,     reverse(bds))
        else
            append!(full_xy,     rxy)
            append!(full_normal, rn)
            append!(full_ds,     bds)
        end
        return nothing
    end
    if has_x && has_y
        # CCW order: Q1(base) → Q2(Y-reflect) → Q3(XY-reflect) → Q4(X-reflect)
        push_reflection!(get_sym(BilliardGeometry.YAxisReflection),  true)  # Q1→Q2: reverse
        push_reflection!(get_sym(BilliardGeometry.XYAxisReflection), false) # Q2→Q3: preserve
        push_reflection!(get_sym(BilliardGeometry.XAxisReflection),  true)  # Q3→Q4: reverse
    elseif has_y
        # CCW order: Q1(base) → Q2(Y-reflect, reversed)
        push_reflection!(get_sym(BilliardGeometry.YAxisReflection), true)
    elseif has_x
        # CCW order: Q1(base) → Q4(X-reflect, reversed)
        push_reflection!(get_sym(BilliardGeometry.XAxisReflection), true)
    end
    full_s = cumsum(full_ds)
    return BoundaryPoints(full_xy; normal=full_normal, s=full_s, ds=full_ds)
end