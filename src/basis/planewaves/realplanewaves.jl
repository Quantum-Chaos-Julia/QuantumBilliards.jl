"""
RealPlaneWaves{T,Sa} <: AbsBasis

`RealPlaneWaves` is a concrete basis type representing real plane waves with
optional reflection symmetries.

## Description
Each basis function has the separable form

```math
f(x,y) = F_x(kx)\\,F_y(ky),
```

where \$F_x,F_y \\in \\{\\cos,\\sin\\}\$ are selected per basis function by the
`parity_x` and `parity_y` fields (`+1` selects `cos`, `-1` selects `sin`), and
the propagation direction is set by `angles` (quadrant ordering
`(+x,+y), (+x,-y), (-x,+y), (-x,-y)`).

Each entry of `symmetries` has a corresponding quantum number at the same
index in `sym_qnumbers`. When both an x- and a y-axis reflection are present,
the symmetry order is `[YAxisReflection, XYAxisReflection, XAxisReflection]`
with quantum numbers `[sym_x, sym_x*sym_y, sym_y]`. The quantum numbers
restrict which quadrants/parities are used:
- No symmetries: all 4 quadrants, 4 patterns `(cos,cos), (cos,sin), (sin,cos), (sin,sin)`.
- X-axis reflection (`sym_y`): 2 quadrants with fixed y-parity; `sym_y = +1` gives `(cos,cos), (sin,cos)` (upper half-plane, y>0), `sym_y = -1` gives `(cos,sin), (sin,sin)` (lower half-plane, y<0).
- Y-axis reflection (`sym_x`): 2 quadrants with fixed x-parity; `sym_x = +1` gives `(cos,cos), (cos,sin)` (right half-plane, x>0), `sym_x = -1` gives `(sin,cos), (sin,sin)` (left half-plane, x<0).
- XY-axis reflection (both `sym_x` and `sym_y`): 1 quadrant, selected by `(sym_y, sym_x)`: `(+1,+1) → (cos,cos)` (quadrant I), `(+1,-1) → (sin,cos)` (quadrant II), `(-1,+1) → (cos,sin)` (quadrant IV), `(-1,-1) → (sin,sin)` (quadrant III).

## Attributes
* `dim`: Number of distinct sampled angles (the effective basis dimension is `dim` times the number of parity patterns).
* `symmetries`: Reflection symmetries applied to the basis, or `nothing`.
* `sym_qnumbers`: Quantum number for each entry of `symmetries`, or `nothing`.
* `angle_arc`: Angular range over which directions are sampled.
* `angle_shift`: Angular offset applied to the sampled directions.
* `angles`: Propagation angles of the plane waves.
* `parity_x`: Parity selector for the `x` factor of each basis function (`+1` = `cos`, `-1` = `sin`).
* `parity_y`: Parity selector for the `y` factor of each basis function (`+1` = `cos`, `-1` = `sin`).
* `sampler`: Sampling strategy used to sample the angles.

## API
The following functions can be evaluated for this type:
- [`resize_basis`](@ref)
- [`basis_fun`](@ref)
- [`gradient`](@ref)
- [`basis_and_gradient`](@ref)
"""
struct RealPlaneWaves{T,Sa} <: AbsBasis where {T<:Real, Sa<:AbsSampler}
    dim::Int64
    symmetries::Union{Vector{BilliardGeometry.AbsReflection}, Nothing}
    sym_qnumbers::Union{Vector{T}, Nothing}
    angle_arc::T
    angle_shift::T
    angles::Vector{T}
    parity_x::Vector{Int64}
    parity_y::Vector{Int64}
    sampler::Sa
end

"""
    parity_pattern(symmetries, sym_qnumbers) → (parity_x, parity_y)::Tuple{Vector{Int},Vector{Int}}

Determine the cos/sin parity pattern for each quadrant selected by the given
reflection `symmetries` and their quantum numbers `sym_qnumbers`.

## Description
Each symmetry has a corresponding quantum number at the same index. Depending
on which combination of [`BilliardGeometry.XAxisReflection`](@ref),
[`BilliardGeometry.YAxisReflection`](@ref) and
[`BilliardGeometry.XYAxisReflection`](@ref) is present, the resulting pattern
selects 1, 2 or 4 quadrants, as documented on [`RealPlaneWaves`](@ref).

## Arguments
* `symmetries`: Reflection symmetries, or `nothing` for no symmetry restriction.
* `sym_qnumbers`: Quantum number for each entry of `symmetries` (same length as `symmetries`), or `nothing`.

## Returns
*  `(parity_x, parity_y)` : Parity patterns where `1` selects `cos` and `-1` selects `sin` for each direction.
"""
@inline function parity_pattern(::Nothing, ::Nothing)
    # No symmetries: use all four quadrants in order (+x,+y), (+x,-y), (-x,+y), (-x,-y)
    return Int[1, 1, -1, -1], Int[1, -1, 1, -1]
end

@inline function parity_pattern(symmetries::Vector{BG}, 
                                sym_qnumbers::Vector{T}) where {T<:Real, BG<:BilliardGeometry.AbsReflection}
    # Check which symmetries are present
    has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
    has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
    has_xy = any(s -> s isa BilliardGeometry.XYAxisReflection, symmetries)
    
    if has_xy
        # XY-axis reflection: single quadrant
        # Find the XY quantum number (should be the product sym_x * sym_y)
        xy_idx = findfirst(s -> s isa BilliardGeometry.XYAxisReflection, symmetries)
        x_idx = findfirst(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
        y_idx = findfirst(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
        
        x_par = Int(sym_qnumbers[x_idx])
        y_par = Int(sym_qnumbers[y_idx])
        
        return Int[x_par], Int[y_par]
    elseif has_x && !has_y
        # Only X-axis reflection: 2 quadrants with fixed y-parity
        x_idx = findfirst(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
        y_par = Int(sym_qnumbers[x_idx])
        return Int[1, -1], Int[y_par, y_par]
    elseif has_y && !has_x
        # Only Y-axis reflection: 2 quadrants with fixed x-parity
        y_idx = findfirst(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
        x_par = Int(sym_qnumbers[y_idx])
        return Int[x_par, x_par], Int[1, -1]
    else
        # Fallback to no symmetries
        return parity_pattern(nothing, nothing)
    end
end

"""
    infer_quantum_numbers(symmetries) → sym_qnumbers::Union{Vector{Float64},Nothing}

Infer default quantum numbers from a symmetry list, for backward compatibility.
All quantum numbers default to `+1` (even parity).

## Arguments
* `symmetries`: Reflection symmetries, or `nothing`.

## Returns
*  `sym_qnumbers` : `nothing` if `symmetries` is empty or `nothing`, otherwise a vector of `+1.0` with one entry per symmetry.
"""
function infer_quantum_numbers(symmetries::Vector{BG}) where {BG<:BilliardGeometry.AbsReflection}
    isempty(symmetries) && return nothing
    # Default all quantum numbers to +1 (even parity)
    return ones(Float64, length(symmetries))
end

@inline infer_quantum_numbers(::Nothing) = nothing

"""
    RealPlaneWaves(dim::Int, symmetries::Union{Vector{BG},Nothing}, sym_qnumbers::Union{Vector{T},Nothing}; angle_arc = π, angle_shift = 0.0, sampler = LinearNodes()) where {T<:Real, BG<:BilliardGeometry.AbsReflection} → basis::RealPlaneWaves

Construct a [`RealPlaneWaves`](@ref) basis of dimension `dim` for the given
`symmetries` and their quantum numbers `sym_qnumbers`.

## Arguments
* `dim`: Number of distinct angles to sample.
* `symmetries`: Reflection symmetries, or `nothing` for no symmetry restriction.
* `sym_qnumbers`: Quantum number for each entry of `symmetries` (must have the same length), or `nothing`.

## Keyword arguments
*  `angle_arc::Real = π` : Angular range over which directions are sampled.
*  `angle_shift::Real = 0.0` : Angular offset applied to the sampled directions.
*  `sampler::AbsSampler = LinearNodes()` : Sampling strategy used to sample the angles.

## Returns
*  `basis` : A [`RealPlaneWaves`](@ref) basis with the given symmetries and quantum numbers.
"""
function RealPlaneWaves(dim::Int, 
                       symmetries::Union{Vector{BG}, Nothing}, 
                       sym_qnumbers::Union{Vector{T}, Nothing}; 
                       angle_arc=π, angle_shift=0.0, 
                       sampler=LinearNodes()) where {T<:Real, BG<:BilliardGeometry.AbsReflection}
    # Validate that symmetries and sym_qnumbers have matching lengths
    if !isnothing(symmetries) && !isnothing(sym_qnumbers)
        @assert length(symmetries) == length(sym_qnumbers) "symmetries and sym_qnumbers must have the same length"
    end
    
    # Get parity pattern from symmetries and quantum numbers
    par_x, par_y = parity_pattern(symmetries, sym_qnumbers)
    pl = length(par_x)
    eff_dim = dim * pl
    
    # Sample angles from the sampler
    t, dt = sample_points(sampler, dim)
    
    # Preallocate and fill arrays more efficiently
    angles = Vector{eltype(t)}(undef, eff_dim)
    parity_x = Vector{Int}(undef, eff_dim)
    parity_y = Vector{Int}(undef, eff_dim)
    
    # Fill arrays using vectorized operations
    @inbounds for i in 1:dim
        angle = t[i] * angle_arc + angle_shift
        base_idx = (i-1) * pl
        for j in 1:pl
            idx = base_idx + j
            angles[idx] = angle
            parity_x[idx] = par_x[j]
            parity_y[idx] = par_y[j]
        end
    end
    
    Sa = typeof(sampler)
    
    return RealPlaneWaves{eltype(angles), Sa}(eff_dim, symmetries, sym_qnumbers, 
                                              angle_arc, angle_shift, angles, 
                                              parity_x, parity_y, sampler)
end

"""
    RealPlaneWaves(dim::Int, billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector; angle_arc=nothing, angle_shift=nothing, sampler = LinearNodes()) → basis::RealPlaneWaves

Construct a [`RealPlaneWaves`](@ref) basis of dimension `dim` from a
billiard's registered geometric symmetry group and a chosen
[`SymmetrySector`](@ref) representation, instead of building `symmetries`/
`sym_qnumbers` by hand.

## Description
Only `BilliardGeometry.XAxisReflection`/`YAxisReflection` generators of
`billiard.symmetries` are consulted (matching the reflection axes
[`RealPlaneWaves`](@ref) natively supports); `sector` must supply a real
`±1` character for each one it constrains. If both an `x`- and a `y`-axis
reflection are constrained by `sector` and `billiard.symmetries` also
registers the corresponding `XYAxisReflection`, its character is derived
automatically as the product of the two axis characters (matching
[`RealPlaneWaves`](@ref)`(dim; sym_x, sym_y)`'s existing convention) — the
caller only ever chooses characters for the two axis reflections.

## Arguments
* `dim`: Number of distinct angles to sample.
* `billiard`: The billiard `sector`'s `sym_id`s are resolved against.
* `sector`: The requested [`SymmetrySector`](@ref) representation.

## Keyword arguments
*  `angle_arc::Union{Real,Nothing} = nothing` : Angular range to sample; auto-adjusted based on the symmetries if not given.
*  `angle_shift::Union{Real,Nothing} = nothing` : Angular offset; auto-adjusted based on the symmetries if not given.
*  `sampler::AbsSampler = LinearNodes()` : Sampling strategy used to sample the angles.

## Returns
*  `basis` : A [`RealPlaneWaves`](@ref) basis whose symmetries/quantum numbers are derived from `billiard` and `sector`.
"""
function RealPlaneWaves(dim::Int, billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector;
                       angle_arc::Union{Real,Nothing}=nothing, angle_shift::Union{Real,Nothing}=nothing,
                       sampler=LinearNodes())
    x_sym = findfirst(s -> s isa BilliardGeometry.XAxisReflection && haskey(sector.characters, s.sym_id), billiard.symmetries)
    y_sym = findfirst(s -> s isa BilliardGeometry.YAxisReflection && haskey(sector.characters, s.sym_id), billiard.symmetries)
    sym_x = isnothing(y_sym) ? nothing : Int(real(sector.characters[billiard.symmetries[y_sym].sym_id]))
    sym_y = isnothing(x_sym) ? nothing : Int(real(sector.characters[billiard.symmetries[x_sym].sym_id]))
    return RealPlaneWaves(dim; sym_x, sym_y, angle_arc, angle_shift, sampler)
end


"""
    RealPlaneWaves(dim::Int; sym_x::Union{Int,Nothing} = nothing, sym_y::Union{Int,Nothing} = nothing, angle_arc::Union{Real,Nothing} = nothing, angle_shift::Union{Real,Nothing} = nothing, sampler = LinearNodes()) → basis::RealPlaneWaves

Main constructor for [`RealPlaneWaves`](@ref), specifying symmetries through the
quantum numbers `sym_x` and `sym_y`.

## Description
The quantum numbers determine which quadrants and cos/sin patterns are used:
- `sym_x = nothing, sym_y = nothing`: all 4 quadrants, 4 combinations (`angle_arc = π`, `angle_shift = 0`).
- `sym_x = nothing, sym_y = ±1`: 2 quadrants (x-axis reflection) (`angle_arc = π`, `angle_shift = 0`).
- `sym_x = ±1, sym_y = nothing`: 2 quadrants (y-axis reflection) (`angle_arc = π`, `angle_shift = -π/2`).
- `sym_x = ±1, sym_y = ±1`: 1 quadrant (xy-axis reflection) (`angle_arc = π/2`, `angle_shift = 0`).

When both symmetries are present, the symmetry order is
`[YAxisReflection, XYAxisReflection, XAxisReflection]` with quantum numbers
`[sym_x, sym_x*sym_y, sym_y]`.

## Arguments
* `dim`: Number of distinct angles to sample.

## Keyword arguments
*  `sym_x::Union{Int,Nothing} = nothing` : Quantum number for the y-axis reflection (`±1` for even/odd parity, `nothing` for no symmetry).
*  `sym_y::Union{Int,Nothing} = nothing` : Quantum number for the x-axis reflection (`±1` for even/odd parity, `nothing` for no symmetry).
*  `angle_arc::Union{Real,Nothing} = nothing` : Angular range to sample; auto-adjusted based on the symmetries if not given.
*  `angle_shift::Union{Real,Nothing} = nothing` : Angular offset; auto-adjusted based on the symmetries if not given.
*  `sampler::AbsSampler = LinearNodes()` : Sampling strategy used to sample the angles.

## Returns
*  `basis` : A [`RealPlaneWaves`](@ref) basis with symmetries and quantum numbers derived from `sym_x` and `sym_y`.
"""
function RealPlaneWaves(dim::Int; sym_x::Union{Int,Nothing}=nothing, sym_y::Union{Int,Nothing}=nothing,
                       angle_arc::Union{Real,Nothing}=nothing, angle_shift::Union{Real,Nothing}=nothing, 
                       sampler=LinearNodes())
    # Build symmetries vector and quantum numbers based on sym_x and sym_y
    # Automatically adjust angle_arc and angle_shift based on symmetries if not provided
    
    if isnothing(sym_x) && isnothing(sym_y)
        # No symmetries - fast path
        symmetries = nothing
        sym_qnumbers = nothing
        arc = isnothing(angle_arc) ? π : angle_arc
        shift = isnothing(angle_shift) ? 0.0 : angle_shift
        
    elseif !isnothing(sym_x) && !isnothing(sym_y)
        # Both symmetries: YAxisReflection, XYAxisReflection, XAxisReflection (in that order)
        # Single quadrant → arc = π/2
        symmetries = BilliardGeometry.AbsReflection[
            BilliardGeometry.YAxisReflection(),
            BilliardGeometry.XYAxisReflection(),
            BilliardGeometry.XAxisReflection()
        ]
        # Quantum numbers: [sym_x, sym_x*sym_y, sym_y]
        sym_qnumbers = Float64[Float64(sym_x), Float64(sym_x * sym_y), Float64(sym_y)]
        arc = isnothing(angle_arc) ? π/2 : angle_arc
        shift = isnothing(angle_shift) ? 0.0 : angle_shift
        
    elseif !isnothing(sym_y)
        # Only x-axis reflection (reflects about x-axis, constrains y-parity)
        # 2 quadrants (upper or lower half-plane) → arc = π, shift = 0
        symmetries = BilliardGeometry.AbsReflection[BilliardGeometry.XAxisReflection()]
        sym_qnumbers = Float64[Float64(sym_y)]
        arc = isnothing(angle_arc) ? π : angle_arc
        shift = isnothing(angle_shift) ? 0.0 : angle_shift
        
    else  # !isnothing(sym_x)
        # Only y-axis reflection (reflects about y-axis, constrains x-parity)
        # 2 quadrants (right or left half-plane) → arc = π, shift = -π/2
        symmetries = BilliardGeometry.AbsReflection[BilliardGeometry.YAxisReflection()]
        sym_qnumbers = Float64[Float64(sym_x)]
        arc = isnothing(angle_arc) ? π : angle_arc
        shift = isnothing(angle_shift) ? -π/2 : angle_shift
    end
    
    # Call the main constructor
    return RealPlaneWaves(dim, symmetries, sym_qnumbers; 
                         angle_arc=arc, angle_shift=shift, sampler=sampler)
end

"""
    RealPlaneWaves(dim::Int, symmetries::Union{Vector{BG},Nothing}; angle_arc = π, angle_shift = 0.0, sampler = LinearNodes()) where {BG<:BilliardGeometry.AbsReflection} → basis::RealPlaneWaves

Construct a [`RealPlaneWaves`](@ref) basis of dimension `dim` for the given
`symmetries`, inferring default quantum numbers (all `+1`, even parity) via
[`infer_quantum_numbers`](@ref).

## Arguments
* `dim`: Number of distinct angles to sample.
* `symmetries`: Reflection symmetries, or `nothing` for no symmetry restriction.

## Keyword arguments
*  `angle_arc::Real = π` : Angular range over which directions are sampled.
*  `angle_shift::Real = 0.0` : Angular offset applied to the sampled directions.
*  `sampler::AbsSampler = LinearNodes()` : Sampling strategy used to sample the angles.

## Returns
*  `basis` : A [`RealPlaneWaves`](@ref) basis with the given symmetries and default (`+1`) quantum numbers.
"""
function RealPlaneWaves(dim::Int, 
                       symmetries::Union{Vector{BG}, Nothing}; 
                       angle_arc=π, angle_shift=0.0, 
                       sampler=LinearNodes()) where {BG<:BilliardGeometry.AbsReflection}
    sym_qnumbers = infer_quantum_numbers(symmetries)
    return RealPlaneWaves(dim, symmetries, sym_qnumbers; 
                         angle_arc=angle_arc, angle_shift=angle_shift, sampler=sampler)
end

"""
    resize_basis(basis::RealPlaneWaves, billiard::AbsBilliard, dim::Int, k) → basis_new::RealPlaneWaves

Return a [`RealPlaneWaves`](@ref) basis resized to dimension `dim`, preserving
symmetries, quantum numbers, and sampling parameters.

## Arguments
* `basis`: The basis to resize.
* `billiard`: Billiard the basis is defined on (unused, kept for interface consistency with other basis types).
* `dim`: Target dimension.
* `k`: Wavenumber (unused, kept for interface consistency with other basis types).

## Returns
*  `basis_new` : A new [`RealPlaneWaves`](@ref) basis of dimension `dim` with the same symmetries, quantum numbers, angular range/offset, and sampler as `basis`.
"""
@inline function resize_basis(basis::RealPlaneWaves, billiard::AbsBilliard, dim::Int, k)
    return RealPlaneWaves(dim, basis.symmetries, basis.sym_qnumbers; 
                         angle_arc=basis.angle_arc, 
                         angle_shift=basis.angle_shift, 
                         sampler=basis.sampler)
end

# Helper functions for cos/sin pattern
# parity = 1 → cos, parity = -1 → sin
@inline _cos(arg) = cos(arg)
@inline _sin(arg) = sin(arg)
@inline _rpw_fun(par::Int) = par == 1 ? _cos : _sin
@inline _drpw_fun(par::Int) = par == 1 ? (x -> -sin(x)) : _cos


"""
    basis_fun(basis::RealPlaneWaves, i::Int, k::T, pts::AbstractArray) where {T<:Real} → out::Vector{T}

Evaluate the `i`-th real plane wave basis function at wavenumber `k` on the
points `pts`.

## Arguments
* `basis`: The [`RealPlaneWaves`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the basis function is evaluated.

## Returns
*  `out` : Column `i` of the basis matrix, evaluated at the input points.
"""
@inline function basis_fun(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    parx=basis.parity_x[i]
    pary=basis.parity_y[i]
    vx=cos(basis.angles[i])
    vy=sin(basis.angles[i])
    fx=_rpw_fun(parx)
    fy=_rpw_fun(pary)
    M=length(pts)
    out=Vector{T}(undef,M)
    @inbounds @simd for j=1:M
        x=pts[j][1]
        y=pts[j][2]
        out[j]=fx(k*vx*x)*fy(k*vy*y)
    end
    return out
end

"""
    basis_fun(basis::RealPlaneWaves, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → B::Matrix{T}

Evaluate the real plane wave basis functions with the given `indices` at
wavenumber `k` on the points `pts`.

## Arguments
* `basis`: The [`RealPlaneWaves`](@ref) basis.
* `indices`: Indices of the basis functions to evaluate.
* `k`: Wavenumber.
* `pts`: Points at which the basis functions are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `B` : Basis matrix of size `(length(pts), length(indices))`.
"""
@inline function basis_fun(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts)
    N=length(indices)
    B=Matrix{T}(undef,M,N)
    @use_threads multithreading=multithreaded for c in 1:N
        idx=indices[c]
        parx=basis.parity_x[idx]
        pary=basis.parity_y[idx]
        vx=cos(basis.angles[idx])
        vy=sin(basis.angles[idx])
        fx=_rpw_fun(parx)
        fy=_rpw_fun(pary)
        col=@view B[:,c]
        @inbounds @simd for j=1:M
            x=pts[j][1]
            y=pts[j][2]
            col[j]=fx(k*vx*x)*fy(k*vy*y)
        end
    end
    return B
end

"""
    gradient(basis::RealPlaneWaves, i::Int, k::T, pts::AbstractArray) where {T<:Real} → (dx, dy)::Tuple{Vector{T},Vector{T}}

Evaluate the gradient with respect to `x` and `y` of the `i`-th real plane wave
basis function on the points `pts`.

## Arguments
* `basis`: The [`RealPlaneWaves`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the gradient is evaluated.

## Returns
*  `(dx, dy)` : Vectors with the `x` and `y` components of the gradient of basis function `i` at the input points.
"""
function gradient(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    parx=basis.parity_x[i]
    pary=basis.parity_y[i]
    vx=cos(basis.angles[i])
    vy=sin(basis.angles[i])
    fx=_rpw_fun(parx)
    fy=_rpw_fun(pary)
    dfx=_drpw_fun(parx)
    dfy=_drpw_fun(pary)
    M=length(pts)
    dx=Vector{T}(undef,M)
    dy=Vector{T}(undef,M)
    @inbounds @simd for j=1:M
        x=pts[j][1]
        y=pts[j][2]
        ax=k*vx*x
        ay=k*vy*y
        bx=fx(ax)
        by=fy(ay)
        dx[j]=k*vx*dfx(ax)*by
        dy[j]=bx*k*vy*dfy(ay)
    end
    return dx,dy
end

"""
    gradient(basis::RealPlaneWaves, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → (dB_dx, dB_dy)::Tuple{Matrix{T},Matrix{T}}

Evaluate the gradient with respect to `x` and `y` of the real plane wave basis
functions with the given `indices` on the points `pts`.

## Arguments
* `basis`: The [`RealPlaneWaves`](@ref) basis.
* `indices`: Indices of the basis functions to differentiate.
* `k`: Wavenumber.
* `pts`: Points at which the gradients are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `(dB_dx, dB_dy)` : Matrices with the `x` and `y` components of the gradients, each of size `(length(pts), length(indices))`.
"""
function gradient(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts); N=length(indices)
    dBdx=Matrix{T}(undef,M,N)
    dBdy=Matrix{T}(undef,M,N)
    @use_threads multithreading=multithreaded for c in 1:N
        idx=indices[c]
        parx=basis.parity_x[idx]
        pary=basis.parity_y[idx]
        vx=cos(basis.angles[idx])
        vy=sin(basis.angles[idx])
        fx=_rpw_fun(parx)
        fy=_rpw_fun(pary)
        dfx=_drpw_fun(parx)
        dfy=_drpw_fun(pary)
        cx=@view dBdx[:,c]
        cy=@view dBdy[:,c]
        @inbounds @simd for j=1:M
            x=pts[j][1]
            y=pts[j][2]
            ax=k*vx*x
            ay=k*vy*y
            bx=fx(ax)
            by=fy(ay)
            cx[j]=k*vx*dfx(ax)*by
            cy[j]=bx*k*vy*dfy(ay)
        end
    end
    return dBdx,dBdy
end

"""
    basis_and_gradient(basis::RealPlaneWaves, i::Int, k::T, pts::AbstractArray) where {T<:Real} → (bf, dx, dy)::Tuple{Vector{T},Vector{T},Vector{T}}

Evaluate both the `i`-th real plane wave basis function and its gradient with
respect to `x` and `y` on the points `pts`.

## Description
Combines [`basis_fun`](@ref) and [`gradient`](@ref) in a single pass over the
points, avoiding redundant evaluation of the cos/sin factors.

## Arguments
* `basis`: The [`RealPlaneWaves`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the basis function and its gradient are evaluated.

## Returns
*  `(bf, dx, dy)` : Basis function values and the `x` and `y` components of its gradient at the input points.
"""
function basis_and_gradient(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    parx=basis.parity_x[i]
    pary=basis.parity_y[i]
    vx=cos(basis.angles[i])
    vy=sin(basis.angles[i])
    fx=_rpw_fun(parx)
    fy=_rpw_fun(pary)
    dfx=_drpw_fun(parx)
    dfy=_drpw_fun(pary)
    M=length(pts)
    bf=Vector{T}(undef,M)
    dx=Vector{T}(undef,M)
    dy=Vector{T}(undef,M)
    @inbounds @simd for j=1:M
        x=pts[j][1]
        y=pts[j][2]
        ax=k*vx*x
        ay=k*vy*y
        bx=fx(ax)
        by=fy(ay)
        bf[j]=bx*by
        dx[j]=k*vx*dfx(ax)*by
        dy[j]=bx*k*vy*dfy(ay)
    end
    return bf,dx,dy
end

"""
    basis_and_gradient(basis::RealPlaneWaves, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → (B, dB_dx, dB_dy)::Tuple{Matrix{T},Matrix{T},Matrix{T}}

Evaluate both the real plane wave basis functions with the given `indices` and
their gradients with respect to `x` and `y` on the points `pts`.

## Description
Combines [`basis_fun`](@ref) and [`gradient`](@ref) column-by-column,
optionally in parallel across threads.

## Arguments
* `basis`: The [`RealPlaneWaves`](@ref) basis.
* `indices`: Indices of the basis functions to evaluate.
* `k`: Wavenumber.
* `pts`: Points at which the basis functions and gradients are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `(B, dB_dx, dB_dy)` : Basis matrix and the `x` and `y` components of its gradients, each of size `(length(pts), length(indices))`.
"""
function basis_and_gradient(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts); N=length(indices)
    B=Matrix{T}(undef,M,N)
    dBdx=Matrix{T}(undef,M,N)
    dBdy=Matrix{T}(undef,M,N)
    @use_threads multithreading=multithreaded for c in 1:N
        idx=indices[c]
        parx=basis.parity_x[idx]
        pary=basis.parity_y[idx]
        vx=cos(basis.angles[idx])
        vy=sin(basis.angles[idx])
        fx=_rpw_fun(parx)
        fy=_rpw_fun(pary)
        dfx=_drpw_fun(parx)
        dfy=_drpw_fun(pary)
        col=@view B[:,c]
        cx=@view dBdx[:,c]
        cy=@view dBdy[:,c]
        @inbounds @simd for j=1:M
            x=pts[j][1]
            y=pts[j][2]
            ax=k*vx*x
            ay=k*vy*y
            bx=fx(ax)
            by=fy(ay)
            col[j]=bx*by
            cx[j]=k*vx*dfx(ax)*by
            cy[j]=bx*k*vy*dfy(ay)
        end
    end
    return B,dBdx,dBdy
end

"""
    dk_fun(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}

Constructs the k-gradient of the basis matrix wrt k for column i.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `i::Int`: The column index of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `dk::Vector{T}`: Vector representing the column of dB/dk for the index i.
"""
@inline function dk_fun(basis::RealPlaneWaves,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    parx=basis.parity_x[i]
    pary=basis.parity_y[i]
    vx=cos(basis.angles[i])
    vy=sin(basis.angles[i])
    fx=_rpw_fun(parx)
    fy=_rpw_fun(pary)
    dfx=_drpw_fun(parx)
    dfy=_drpw_fun(pary)
    M=length(pts)
    dk=Vector{T}(undef,M)
    @inbounds @simd for j=1:M
        x=pts[j][1]
        y=pts[j][2]
        ax=k*vx*x
        ay=k*vy*y
        bx=fx(ax)
        by=fy(ay)
        dk[j]=vx*x*dfx(ax)*by + bx*vy*y*dfy(ay)
    end
    return dk
end
    
"""
    dk_fun(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}

Constructs the k-gradient of the basis matrix.

# Arguments
- `basis::RealPlaneWaves`: Struct containing all the info to compute the matrix.
- `indices::AbstractArray`: The column indexes of the matrix.
- `k::T`: Wavenumber to construct matrix at.
- `pts::AbstractArray`: Vector of xy points on the boundary.
- `multithreaded::Bool=true`: If the matrix construction per columns is multithreaded.

# Returns
- `dB_dk::Matrix{T}`: matrix representing dB/dk.
"""
@inline function dk_fun(basis::RealPlaneWaves,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    M=length(pts)
    N=length(indices)
    dBdk=Matrix{T}(undef,M,N)
    @use_threads multithreading=multithreaded for c in 1:N
        idx=indices[c]
        parx=basis.parity_x[idx]
        pary=basis.parity_y[idx]
        vx=cos(basis.angles[idx])
        vy=sin(basis.angles[idx])
        fx=_rpw_fun(parx)
        fy=_rpw_fun(pary)
        dfx=_drpw_fun(parx)
        dfy=_drpw_fun(pary)
        col=@view dBdk[:,c]
        @inbounds @simd for j=1:M
            x=pts[j][1]
            y=pts[j][2]
            ax=k*vx*x
            ay=k*vy*y
            bx=fx(ax)
            by=fy(ay)
            col[j]=vx*x*dfx(ax)*by + bx*vy*y*dfy(ay)
        end
    end
    return dBdk
end