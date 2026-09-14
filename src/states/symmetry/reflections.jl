"""
    apply_symmetries_to_wavefunction(Psi, x_grid::AbstractVector, y_grid::AbstractVector, symmetries::Vector{<:BilliardGeometry.AbsReflection}, sym_qnumbers::Vector{T}) where {T<:Real} → (full_Psi, full_x::Vector, full_y::Vector)

Unfolds a wavefunction `Psi`, computed on the fundamental domain grid
`(x_grid, y_grid)`, onto the full billiard by reflecting it across each axis
of reflection symmetry in `symmetries`, weighted by the corresponding
parities `sym_qnumbers`.

## Description
`Psi` is assumed to be evaluated on the `x > 0`, `y > 0` quadrant (relative to
the reflection axes). For each `XAxisReflection`/`YAxisReflection` in
`symmetries`, the corresponding quantum number in `sym_qnumbers` gives the
parity `p = \\pm 1` under that reflection, and the missing quadrant(s) are
obtained by reversing `Psi` along the appropriate grid dimension and
multiplying by `p`. If both symmetries are present, all four quadrants are
assembled (`Q1` = base, `Q2` = `Y`-reflected, `Q3` = both reflected, `Q4` =
`X`-reflected); if only one is present, only that reflection is applied; if
`symmetries` contains neither, `Psi` and the grids are returned unchanged.

## Arguments
* `Psi`: The wavefunction values on the fundamental domain grid `(x_grid, y_grid)`.
* `x_grid`: The `x` coordinates of the fundamental domain grid.
* `y_grid`: The `y` coordinates of the fundamental domain grid.
* `symmetries`: The reflection symmetries of the basis, e.g. `XAxisReflection`, `YAxisReflection`.
* `sym_qnumbers`: The parity quantum number (`\\pm 1`) associated with each entry of `symmetries`.

## Returns
*  `full_Psi` : The wavefunction unfolded onto the full domain covered by the applicable reflections.
*  `full_x` : The `x` coordinates of the unfolded grid.
*  `full_y` : The `y` coordinates of the unfolded grid.
"""
function apply_symmetries_to_wavefunction(
    Psi, x_grid, y_grid,
    symmetries::Vector{<:BilliardGeometry.AbsReflection},
    sym_qnumbers::Vector{T}
) where T<:Real

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
    apply_symmetries_to_boundary_function(u::AbstractVector{U}, symmetries::Vector{<:BilliardGeometry.AbsReflection}, sym_qnumbers::Vector{T}) where {U<:Number, T<:Real} → full_u::Vector{U}

Unfolds a boundary function `u`, computed on the fundamental-domain arc of the
boundary, onto the full boundary by appending its reflected copies for each
reflection symmetry in `symmetries`, weighted by the corresponding parities
`sym_qnumbers`.

## Description
Returns `u` unchanged if `symmetries` is empty. Otherwise, for each
`XAxisReflection`/`YAxisReflection` present, a reflected and (for a
counter-clockwise-consistent boundary traversal) order-reversed copy of `u`,
scaled by its parity from `sym_qnumbers`, is appended. If both symmetries are
present, three additional copies are appended in the order `Y`-reflected,
`XY`-reflected (not reversed), `X`-reflected, matching the point ordering
produced by [`apply_symmetries_to_boundary_points`](@ref).

## Arguments
* `u`: The boundary function values on the fundamental-domain arc.
* `symmetries`: The reflection symmetries of the basis, e.g. `XAxisReflection`, `YAxisReflection`.
* `sym_qnumbers`: The parity quantum number (`\\pm 1`) associated with each entry of `symmetries`.

## Returns
*  `full_u` : `u` with its symmetry-reflected copies appended, covering the full boundary.
"""
function apply_symmetries_to_boundary_function(
    u::AbstractVector{U},
    symmetries::Vector{<:BilliardGeometry.AbsReflection},
    sym_qnumbers::Vector{T}
) where {U<:Number, T<:Real}

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
    apply_symmetries_to_boundary_points(pts::BoundaryPoints{T}, symmetries::Vector{BilliardGeometry.AbsReflection}, billiard::Bi) where {Bi<:AbsBilliard, T<:Real} → full_pts::BoundaryPoints{T}

Unfolds the boundary points `pts`, sampled on the fundamental-domain arc of
`billiard`, onto the full boundary by appending reflected copies of the
points, normals and arc-length elements for each reflection symmetry in
`symmetries`.

## Description
Returns `pts` unchanged if `symmetries` is empty. Otherwise, the positions
`xy` and normals `normal` are reflected with `apply_symmetry` for each
applicable `XAxisReflection`/`YAxisReflection`/`XYAxisReflection`, and
appended (with the arc-length element `ds`) in an order consistent with a
continuous counter-clockwise traversal of the full boundary; reflections that
reverse the boundary orientation also reverse the order of the appended
points. If both `XAxisReflection` and `YAxisReflection` are present, three
quadrants are appended in the order `Y`-reflected (reversed), `XY`-reflected
(preserved), `X`-reflected (reversed); if only one is present, only that
reflected copy is appended. The arc-length coordinate `s` of the returned
points is recomputed as the cumulative sum of the full `ds`, and all other
fields of [`BoundaryPoints`](@ref) are left empty.

## Arguments
* `pts`: The [`BoundaryPoints`](@ref) sampled on the fundamental-domain arc.
* `symmetries`: The reflection symmetries of the basis, e.g. `XAxisReflection`, `YAxisReflection`.
* `billiard`: The billiard the points are sampled on (unused beyond dispatch, kept for interface consistency).

## Returns
*  `full_pts` : A new [`BoundaryPoints`](@ref) with the symmetry-unfolded points, normals, arc-length elements and cumulative arc-length coordinate `s`.
"""
function apply_symmetries_to_boundary_points(
    pts::BoundaryPoints{T},
    symmetries::Vector{BilliardGeometry.AbsReflection},
    billiard::Bi
) where {Bi<:AbsBilliard, T<:Real}

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