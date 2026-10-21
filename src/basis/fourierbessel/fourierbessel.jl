using Bessels
using CoordinateTransformations, Rotations, StaticArrays

"""
    fb_term(j::Int, trig::Integer, k::T, r::T, phi::T) where {T<:Real} → f::T

Evaluate a single Fourier-Bessel term at wavenumber `k` and polar coordinates
`(r, phi)`.

## Description
The term combines the Bessel function of the first kind of integer order `j`
with an angular factor selected by `trig`:

```math
f(r,\\varphi) = J_{j}(kr)\\,T(j\\varphi), \\qquad T = \\cos \\;(\\texttt{trig} > 0), \\quad T = \\sin \\;(\\texttt{trig} < 0).
```

## Arguments
* `j`: Integer angular order.
* `trig`: Angular factor selector (`+1` = `cos`, `-1` = `sin`).
* `k`: Wavenumber.
* `r`: Radial coordinate.
* `phi`: Angular coordinate.

## Returns
*  `f` : Value of the Fourier-Bessel term.
"""
@inline function fb_term(j::Int, trig::Integer, k::T, r::T, phi::T) where {T<:Real}
    return Jv(j, k * r) * (trig > 0 ? cos(j * phi) : sin(j * phi))
end

"""
    fb_term_dk(j::Int, trig::Integer, k::T, r::T, phi::T) where {T<:Real} → df::T

Evaluate the derivative with respect to the wavenumber `k` of the
Fourier-Bessel term [`fb_term`](@ref).

## Description
Differentiating \$f(r,\\varphi) = J_{j}(kr)\\,T(j\\varphi)\$ with respect to `k`
via the chain rule gives:

```math
\\frac{\\partial f}{\\partial k} = r\\,J_{j}'(kr)\\,T(j\\varphi).
```

## Arguments
* `j`: Integer angular order.
* `trig`: Angular factor selector (`+1` = `cos`, `-1` = `sin`).
* `k`: Wavenumber.
* `r`: Radial coordinate.
* `phi`: Angular coordinate.

## Returns
*  `df` : Value of \$\\partial f/\\partial k\$.
"""
@inline function fb_term_dk(j::Int, trig::Integer, k::T, r::T, phi::T) where {T<:Real}
    return r * _fb_Jvp(j, k * r) * (trig > 0 ? cos(j * phi) : sin(j * phi))
end

# J_j'(x) for integer j ≥ 0, using J_0' = -J_1 to avoid negative orders.
@inline function _fb_Jvp(j::Int, x::T) where {T<:Real}
    j == 0 && return -Jv(1, x)
    return (Jv(j - 1, x) - Jv(j + 1, x)) / 2
end

# f, ∂ₓf, ∂ᵧf for f = J_j(kr) T(jφ), with the r == 0 guard of the polar Jacobian.
@inline function _fb_term_and_gradient(j::Int, trig::Integer, k::T, r::T, φ::T) where {T<:Real}
    x = k * r
    jv = Jv(j, x)
    dj = _fb_Jvp(j, x)
    s, c = sincos(j * φ)
    ang = trig > 0 ? c : s
    dang = trig > 0 ? -j * s : j * c
    sφ, cφ = sincos(φ)
    invr = r == 0 ? zero(T) : inv(r)
    fr = k * dj * ang
    fφ = jv * dang
    return jv * ang, cφ * fr - sφ * invr * fφ, sφ * fr + cφ * invr * fφ
end

"""
    fb_modes(sym_x::Union{Int,Nothing}, sym_y::Union{Int,Nothing}, dim::Int) → (orders, trig)::Tuple{Vector{Int},Vector{Int8}}

Generate the first `dim` Fourier-Bessel modes `(j, T)` of the symmetry sector
labelled by the reflection characters `sym_x` and `sym_y`, ordered by
increasing angular order `j` (`cos` before `sin` for equal `j`).

## Description
Under the reflection \$y\\to-y\$ (\$\\varphi\\to-\\varphi\$), \$\\cos(j\\varphi)\$ has
parity \$+1\$ and \$\\sin(j\\varphi)\$ has parity \$-1\$. Under \$x\\to-x\$
(\$\\varphi\\to\\pi-\\varphi\$), \$\\cos(j\\varphi)\$ has parity \$(-1)^j\$ and
\$\\sin(j\\varphi)\$ has parity \$(-1)^{j+1}\$. A mode belongs to the sector if its
parities agree with every character that is not `nothing`.

## Arguments
* `sym_x`: Character of the y-axis reflection (\$x\\to-x\$), `±1`, or `nothing` for no constraint.
* `sym_y`: Character of the x-axis reflection (\$y\\to-y\$), `±1`, or `nothing` for no constraint.
* `dim`: Number of modes to generate.

## Returns
*  `(orders, trig)` : Integer angular orders `j` and angular factor selectors (`+1` = `cos`, `-1` = `sin`).
"""
function fb_modes(sym_x::Union{Int,Nothing}, sym_y::Union{Int,Nothing}, dim::Int)
    orders = Vector{Int}(undef, dim)
    trig = Vector{Int8}(undef, dim)
    n = 0
    j = 0
    while n < dim
        sign_j = iseven(j) ? 1 : -1
        for τ in (Int8(1), Int8(-1))
            n < dim || break
            (τ < 0 && j == 0) && continue
            parity_y = Int(τ)
            parity_x = τ > 0 ? sign_j : -sign_j
            (isnothing(sym_y) || sym_y == parity_y) || continue
            (isnothing(sym_x) || sym_x == parity_x) || continue
            n += 1
            orders[n] = j
            trig[n] = τ
        end
        j += 1
    end
    return orders, trig
end

"""
    FourierBessel{T} <: AbsBasis

`FourierBessel` is a concrete basis type representing Fourier-Bessel functions
of integer angular order centered at a chosen origin, with optional reflection
symmetry adaptation and optional Taylor acceleration in the wavenumber.

## Description
Each basis function has the form

```math
\\phi_c(r,\\varphi) = J_{j_c}(kr)\\,T_c(j_c\\varphi), \\qquad T_c \\in \\{\\cos,\\sin\\},
```

in local polar coordinates `(r, φ)` around the origin, where the integer order
`j_c` is stored in `orders[c]` and the angular factor is selected by `trig[c]`
(`+1` = `cos`, `-1` = `sin`). The basis is complete for domains star-shaped
with respect to the origin.

Symmetry reduction is performed by selecting only the modes with the correct
parity under the reflections \$y\\to-y\$ (character `sym_y`) and \$x\\to-x\$
(character `sym_x`):

| `sym_y` | `sym_x` | modes |
|:--|:--|:--|
| `+1` | `+1` | `cos`, `j = 0,2,4,…` |
| `+1` | `-1` | `cos`, `j = 1,3,5,…` |
| `-1` | `+1` | `sin`, `j = 1,3,5,…` |
| `-1` | `-1` | `sin`, `j = 2,4,6,…` |
| `+1` | none | `cos`, `j = 0,1,2,…` |
| `-1` | none | `sin`, `j = 1,2,3,…` |
| none | `+1` | `cos` even `j` and `sin` odd `j` |
| none | `-1` | `cos` odd `j` and `sin` even `j ≥ 2` |
| none | none | `cos`, `j ≥ 0` and `sin`, `j ≥ 1` |

Each entry of `symmetries` has a corresponding quantum number at the same
index in `sym_qnumbers`. When both an x- and a y-axis reflection are present,
the symmetry order is `[YAxisReflection, XYAxisReflection, XAxisReflection]`
with quantum numbers `[sym_x, sym_x*sym_y, sym_y]`. Symmetry-adapted bases
require the origin at `(0, 0)` and no rotation.

When `use_taylor = true`, compatible accelerated solvers may construct a
[`FourierBesselTaylorCache`](@ref) and reuse it for basis evaluations within
`taylor_radius` of its expansion center. If `gradients = true`, caches
additionally store the angular data required for Cartesian spatial gradients.

## Attributes
* `cs`: Local [`PolarCS`](@ref) coordinate system centered at the origin of the basis.
* `dim`: Number of basis functions.
* `symmetries`: Reflection symmetries applied to the basis, or `nothing`.
* `sym_qnumbers`: Quantum number for each entry of `symmetries`, or `nothing`.
* `orders`: Integer angular order `j` of each basis function.
* `trig`: Angular factor of each basis function (`+1` = `cos`, `-1` = `sin`).
* `use_taylor`: Whether compatible accelerated solvers may use the analytic Taylor representation of the basis.
* `taylor_degree`: Degree of the analytic Taylor expansion in the wavenumber.
* `taylor_radius`: Maximum distance from a Taylor expansion center over which the cached representation is reused.
* `taylor_tol`: Tolerance used to determine the validity of the Taylor expansion.
* `gradients`: Whether Taylor caches additionally store data required for Cartesian spatial-gradient evaluation.

## API
The following functions can be evaluated for this type:
- [`resize_basis`](@ref)
- [`basis_fun`](@ref)
- [`dk_fun`](@ref)
- [`gradient`](@ref)
- [`basis_and_gradient`](@ref)
- [`FourierBesselTaylorCache`](@ref)
"""
struct FourierBessel{T<:Real} <: AbsBasis
    cs::PolarCS{T}
    dim::Int
    symmetries::Union{Vector{BilliardGeometry.AbsReflection},Nothing}
    sym_qnumbers::Union{Vector{T},Nothing}
    orders::Vector{Int}
    trig::Vector{Int8}
    use_taylor::Bool
    taylor_degree::Int
    taylor_radius::T
    taylor_tol::T
    gradients::Bool
end

"""
    FourierBessel(dim::Int; sym_x = nothing, sym_y = nothing, origin = SVector(0.0, 0.0), rot_angle = 0.0, use_taylor::Bool = true, taylor_degree::Int = 16, taylor_radius::Real = 0.5, taylor_tol::Real = 1e-11, gradients::Bool = false) → basis::FourierBessel

Main constructor for [`FourierBessel`](@ref), specifying the symmetry sector
through the reflection characters `sym_x` and `sym_y`.

## Description
The modes are selected by [`fb_modes`](@ref). When both characters are given,
the symmetry order is `[YAxisReflection, XYAxisReflection, XAxisReflection]`
with quantum numbers `[sym_x, sym_x*sym_y, sym_y]`. Any non-`nothing`
character requires `origin == (0, 0)` and `rot_angle == 0`, since the axis
reflections act about the global origin.

## Arguments
* `dim`: Number of basis functions.

## Keyword Arguments
* `sym_x::Union{Int,Nothing} = nothing`: Character of the y-axis reflection (\$x\\to-x\$), `±1` or `nothing` for no symmetry.
* `sym_y::Union{Int,Nothing} = nothing`: Character of the x-axis reflection (\$y\\to-y\$), `±1` or `nothing` for no symmetry.
* `origin::SVector{2,<:Real} = SVector(0.0, 0.0)`: Origin of the local [`PolarCS`](@ref).
* `rot_angle::Real = 0.0`: Rotation angle of the local coordinate system.
* `use_taylor::Bool = true`: Whether compatible accelerated solvers may use the analytic Taylor representation of the basis.
* `taylor_degree::Int = 16`: Degree of the analytic Taylor expansion in the wavenumber.
* `taylor_radius::Real = 0.5`: Maximum distance from a Taylor expansion center over which the cached representation is reused.
* `taylor_tol::Real = 1e-11`: Tolerance used to determine the validity of the Taylor expansion.
* `gradients::Bool = false`: Whether Taylor caches additionally store data required for Cartesian spatial-gradient evaluation.

## Returns
* `basis`: A [`FourierBessel`](@ref) basis of the requested symmetry sector.
"""
function FourierBessel(dim::Int; sym_x::Union{Int,Nothing}=nothing, sym_y::Union{Int,Nothing}=nothing,
                       origin::SVector{2,<:Real}=SVector(0.0, 0.0), rot_angle::Real=0.0,
                       use_taylor::Bool=true, taylor_degree::Int=16, taylor_radius::Real=0.5,
                       taylor_tol::Real=1e-11, gradients::Bool=false)
    dim >= 1 || throw(ArgumentError("dim must be at least 1"))
    taylor_degree >= 1 || throw(ArgumentError("taylor_degree must be at least 1"))
    taylor_radius > 0 || throw(ArgumentError("taylor_radius must be positive"))
    taylor_tol > 0 || throw(ArgumentError("taylor_tol must be positive"))
    for s in (sym_x, sym_y)
        (isnothing(s) || abs(s) == 1) || throw(ArgumentError("symmetry characters must be ±1 or nothing"))
    end
    T = promote_type(eltype(origin), typeof(rot_angle), typeof(taylor_radius), typeof(taylor_tol))
    if !(isnothing(sym_x) && isnothing(sym_y)) && !(iszero(origin[1]) && iszero(origin[2]) && iszero(rot_angle))
        throw(ArgumentError("symmetry-adapted FourierBessel requires origin == (0, 0) and rot_angle == 0"))
    end
    if isnothing(sym_x) && isnothing(sym_y)
        symmetries = nothing
        sym_qnumbers = nothing
    elseif !isnothing(sym_x) && !isnothing(sym_y)
        symmetries = BilliardGeometry.AbsReflection[
            BilliardGeometry.YAxisReflection(),
            BilliardGeometry.XYAxisReflection(),
            BilliardGeometry.XAxisReflection()
        ]
        sym_qnumbers = T[T(sym_x), T(sym_x * sym_y), T(sym_y)]
    elseif !isnothing(sym_y)
        symmetries = BilliardGeometry.AbsReflection[BilliardGeometry.XAxisReflection()]
        sym_qnumbers = T[T(sym_y)]
    else
        symmetries = BilliardGeometry.AbsReflection[BilliardGeometry.YAxisReflection()]
        sym_qnumbers = T[T(sym_x)]
    end
    orders, trig = fb_modes(sym_x, sym_y, dim)
    cs = PolarCS(SVector{2,T}(origin), T(rot_angle))
    return FourierBessel{T}(cs, dim, symmetries, sym_qnumbers, orders, trig, use_taylor, taylor_degree, T(taylor_radius), T(taylor_tol), gradients)
end

# Reflection characters (sym_x, sym_y) encoded by a symmetries/sym_qnumbers pair.
function _fb_axis_characters(symmetries::Union{Vector{<:BilliardGeometry.AbsReflection},Nothing}, sym_qnumbers::Union{Vector{<:Real},Nothing})
    isnothing(symmetries) && return nothing, nothing
    isnothing(sym_qnumbers) && throw(ArgumentError("sym_qnumbers must be given together with symmetries"))
    length(symmetries) == length(sym_qnumbers) || throw(ArgumentError("symmetries and sym_qnumbers must have the same length"))
    sym_x = nothing
    sym_y = nothing
    has_xy = false
    for (s, q) in zip(symmetries, sym_qnumbers)
        if s isa BilliardGeometry.YAxisReflection
            sym_x = Int(q)
        elseif s isa BilliardGeometry.XAxisReflection
            sym_y = Int(q)
        elseif s isa BilliardGeometry.XYAxisReflection
            has_xy = true
        else
            throw(ArgumentError("FourierBessel supports only XAxisReflection, YAxisReflection and XYAxisReflection symmetries, got $(typeof(s))"))
        end
    end
    if has_xy && (isnothing(sym_x) || isnothing(sym_y))
        throw(ArgumentError("XYAxisReflection requires both XAxisReflection and YAxisReflection"))
    end
    return sym_x, sym_y
end

"""
    FourierBessel(dim::Int, symmetries::Union{Vector{<:BilliardGeometry.AbsReflection},Nothing}, sym_qnumbers::Union{Vector{<:Real},Nothing}; kwargs...) → basis::FourierBessel

Construct a [`FourierBessel`](@ref) basis from a reflection `symmetries` vector
and its `sym_qnumbers`, as stored by an existing basis.

## Description
The characters `sym_x` and `sym_y` are read from the `YAxisReflection` and
`XAxisReflection` entries and the basis is built by the keyword constructor.
Any other symmetry raises an `ArgumentError`.

## Arguments
* `dim`: Number of basis functions.
* `symmetries`: Reflection symmetries, or `nothing` for no symmetry restriction.
* `sym_qnumbers`: Quantum number for each entry of `symmetries`, or `nothing`.

## Keyword Arguments
* `kwargs...`: Forwarded to the keyword constructor (`origin`, `rot_angle`, `use_taylor`, `taylor_degree`, `taylor_radius`, `taylor_tol`, `gradients`).

## Returns
* `basis`: A [`FourierBessel`](@ref) basis with the requested symmetries.
"""
function FourierBessel(dim::Int, symmetries::Union{Vector{<:BilliardGeometry.AbsReflection},Nothing}, sym_qnumbers::Union{Vector{<:Real},Nothing}; kwargs...)
    sym_x, sym_y = _fb_axis_characters(symmetries, sym_qnumbers)
    return FourierBessel(dim; sym_x, sym_y, kwargs...)
end

"""
    FourierBessel(dim::Int, billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector; kwargs...) → basis::FourierBessel

Construct a [`FourierBessel`](@ref) basis of dimension `dim` from a billiard's
registered geometric symmetry group and a chosen [`SymmetrySector`](@ref).

## Description
Only `BilliardGeometry.XAxisReflection`/`YAxisReflection` generators of
`billiard.symmetries` are consulted. `sector` must supply a real `±1`
character for each one it constrains. Any other constrained generator (e.g. an
`NFoldRotation`) raises an `ArgumentError` rather than being silently ignored.

## Arguments
* `dim`: Number of basis functions.
* `billiard`: The billiard `sector`'s `sym_id`s are resolved against. Must be the exact billiard instance `sector` was built for.
* `sector`: The requested [`SymmetrySector`](@ref) representation.

## Keyword Arguments
* `kwargs...`: Forwarded to the keyword constructor (`origin`, `rot_angle`, `use_taylor`, `taylor_degree`, `taylor_radius`, `taylor_tol`, `gradients`).

## Returns
* `basis`: A [`FourierBessel`](@ref) basis whose symmetries are derived from `billiard` and `sector`.
"""
function FourierBessel(dim::Int, billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector; kwargs...)
    _check_sector_billiard(billiard, sector)
    x_sym = findfirst(s -> s isa BilliardGeometry.XAxisReflection && haskey(sector.characters, s.sym_id), billiard.symmetries)
    y_sym = findfirst(s -> s isa BilliardGeometry.YAxisReflection && haskey(sector.characters, s.sym_id), billiard.symmetries)
    sym_x = isnothing(y_sym) ? nothing : Int(real(sector.characters[billiard.symmetries[y_sym].sym_id]))
    sym_y = isnothing(x_sym) ? nothing : Int(real(sector.characters[billiard.symmetries[x_sym].sym_id]))
    consumed = Set{Int}()
    isnothing(x_sym) || push!(consumed, billiard.symmetries[x_sym].sym_id)
    isnothing(y_sym) || push!(consumed, billiard.symmetries[y_sym].sym_id)
    unhandled = setdiff(keys(sector.characters), consumed)
    isempty(unhandled) || throw(ArgumentError(
        "FourierBessel only supports XAxisReflection/YAxisReflection symmetry " *
        "sectors; sector constrains sym_id(s) $(collect(unhandled)) that " *
        "FourierBessel has no representation for."))
    return FourierBessel(dim; sym_x, sym_y, kwargs...)
end

"""
    resize_basis(basis::FourierBessel, billiard::Bi, dim::Int, k) where {Bi<:AbsBilliard} → basis_new::FourierBessel

Return a [`FourierBessel`](@ref) basis resized to dimension `dim`, reusing
`basis` unchanged if it already has the requested dimension.

## Arguments
* `basis`: The basis to resize.
* `billiard`: Billiard the basis is defined on (unused, kept for interface consistency with other basis types).
* `dim`: Target dimension.
* `k`: Wavenumber (unused, kept for interface consistency with other basis types).

## Returns
*  `basis_new` : `basis` itself if `basis.dim == dim`, otherwise a new basis with the first `dim` modes of the same symmetry sector, origin and Taylor settings.
"""
function resize_basis(basis::FourierBessel, billiard::Bi, dim::Int, k) where {Bi<:AbsBilliard}
    basis.dim == dim && return basis
    return FourierBessel(dim, basis.symmetries, basis.sym_qnumbers; origin=basis.cs.origin, rot_angle=basis.cs.rot_angle,
                         use_taylor=basis.use_taylor, taylor_degree=basis.taylor_degree, taylor_radius=basis.taylor_radius,
                         taylor_tol=basis.taylor_tol, gradients=basis.gradients)
end

# Local polar coordinates of `pts` about the origin of `basis`.
@inline function _fb_polar_coords(basis::FourierBessel{T}, pts::AbstractArray) where {T<:Real}
    M = length(pts)
    r = Vector{T}(undef, M)
    φ = Vector{T}(undef, M)
    _polar_coords!(r, φ, basis.cs.local_map, pts, zero(T))
    return r, φ
end

"""
    basis_fun(basis::FourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real} → out::Vector{T}

Evaluate the `i`-th Fourier-Bessel basis function at wavenumber `k` on the
points `pts`.

## Arguments
* `basis`: The [`FourierBessel`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the basis function is evaluated.

## Returns
*  `out` : Values of the basis function at the input points.
"""
function basis_fun(basis::FourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real}
    r, φ = _fb_polar_coords(basis, pts)
    M = length(pts)
    j = basis.orders[i]
    τ = basis.trig[i]
    out = Vector{T}(undef, M)
    @inbounds for p in 1:M
        out[p] = fb_term(j, τ, k, r[p], φ[p])
    end
    return out
end

"""
    basis_fun(basis::FourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → B::Matrix{T}

Evaluate the Fourier-Bessel basis functions with the given `indices` at
wavenumber `k` on the points `pts`.

## Arguments
* `basis`: The [`FourierBessel`](@ref) basis.
* `indices`: Indices of the basis functions to evaluate.
* `k`: Wavenumber.
* `pts`: Points at which the basis functions are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `B` : Basis matrix of size `(length(pts), length(indices))`.
"""
function basis_fun(basis::FourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool=true) where {T<:Real}
    r, φ = _fb_polar_coords(basis, pts)
    M = length(pts)
    N = length(indices)
    B = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c in 1:N
        j = basis.orders[indices[c]]
        τ = basis.trig[indices[c]]
        col = @view B[:, c]
        @inbounds for p in 1:M
            col[p] = fb_term(j, τ, k, r[p], φ[p])
        end
    end
    return B
end

"""
    dk_fun(basis::FourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real} → dk::Vector{T}

Evaluate the derivative with respect to the wavenumber `k` of the `i`-th
Fourier-Bessel basis function on the points `pts`.

## Arguments
* `basis`: The [`FourierBessel`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the derivative is evaluated.

## Returns
*  `dk` : Column `i` of \$\\partial B/\\partial k\$.
"""
function dk_fun(basis::FourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real}
    r, φ = _fb_polar_coords(basis, pts)
    M = length(pts)
    j = basis.orders[i]
    τ = basis.trig[i]
    dk = Vector{T}(undef, M)
    @inbounds for p in 1:M
        dk[p] = fb_term_dk(j, τ, k, r[p], φ[p])
    end
    return dk
end

"""
    dk_fun(basis::FourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → dB_dk::Matrix{T}

Evaluate the derivative with respect to the wavenumber `k` of the
Fourier-Bessel basis functions with the given `indices` on the points `pts`.

## Arguments
* `basis`: The [`FourierBessel`](@ref) basis.
* `indices`: Indices of the basis functions to differentiate.
* `k`: Wavenumber.
* `pts`: Points at which the derivatives are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `dB_dk` : Wavenumber derivative of the basis matrix, of size `(length(pts), length(indices))`.
"""
function dk_fun(basis::FourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool=true) where {T<:Real}
    r, φ = _fb_polar_coords(basis, pts)
    M = length(pts)
    N = length(indices)
    dB_dk = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c in 1:N
        j = basis.orders[indices[c]]
        τ = basis.trig[indices[c]]
        col = @view dB_dk[:, c]
        @inbounds for p in 1:M
            col[p] = fb_term_dk(j, τ, k, r[p], φ[p])
        end
    end
    return dB_dk
end

"""
    gradient(basis::FourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real} → (dx, dy)::Tuple{Vector{T},Vector{T}}

Evaluate the gradient with respect to the Cartesian coordinates `x` and `y` of
the `i`-th Fourier-Bessel basis function on the points `pts`.

## Description
Using the chain rule with the polar-to-Cartesian Jacobian,

```math
\\partial_x f = \\cos\\varphi\\,\\partial_r f - \\frac{\\sin\\varphi}{r}\\,\\partial_\\varphi f, \\qquad
\\partial_y f = \\sin\\varphi\\,\\partial_r f + \\frac{\\cos\\varphi}{r}\\,\\partial_\\varphi f,
```

with \$\\partial_r f = k J_{j}'(kr)\\,T(j\\varphi)\$ and
\$\\partial_\\varphi f = J_{j}(kr)\\,\\partial_\\varphi T(j\\varphi)\$. Points at
the origin are assigned a zero gradient.

## Arguments
* `basis`: The [`FourierBessel`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the gradient is evaluated.

## Returns
*  `(dx, dy)` : Vectors with the `x` and `y` components of the gradient at the input points.
"""
function gradient(basis::FourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real}
    r, φ = _fb_polar_coords(basis, pts)
    M = length(pts)
    j = basis.orders[i]
    τ = basis.trig[i]
    dx = Vector{T}(undef, M)
    dy = Vector{T}(undef, M)
    @inbounds for p in 1:M
        _, dx[p], dy[p] = _fb_term_and_gradient(j, τ, k, r[p], φ[p])
    end
    return dx, dy
end

"""
    gradient(basis::FourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → (dB_dx, dB_dy)::Tuple{Matrix{T},Matrix{T}}

Evaluate the gradient with respect to the Cartesian coordinates `x` and `y` of
the Fourier-Bessel basis functions with the given `indices` on the points `pts`.

## Arguments
* `basis`: The [`FourierBessel`](@ref) basis.
* `indices`: Indices of the basis functions to differentiate.
* `k`: Wavenumber.
* `pts`: Points at which the gradients are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `(dB_dx, dB_dy)` : Matrices with the `x` and `y` components of the gradients, each of size `(length(pts), length(indices))`.
"""
function gradient(basis::FourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool=true) where {T<:Real}
    r, φ = _fb_polar_coords(basis, pts)
    M = length(pts)
    N = length(indices)
    dB_dx = Matrix{T}(undef, M, N)
    dB_dy = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c in 1:N
        j = basis.orders[indices[c]]
        τ = basis.trig[indices[c]]
        cx = @view dB_dx[:, c]
        cy = @view dB_dy[:, c]
        @inbounds for p in 1:M
            _, cx[p], cy[p] = _fb_term_and_gradient(j, τ, k, r[p], φ[p])
        end
    end
    return dB_dx, dB_dy
end

"""
    basis_and_gradient(basis::FourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real} → (bf, dx, dy)::Tuple{Vector{T},Vector{T},Vector{T}}

Evaluate both the `i`-th Fourier-Bessel basis function and its gradient with
respect to `x` and `y` on the points `pts`.

## Description
Combines [`basis_fun`](@ref) and [`gradient`](@ref) in a single pass over the
points, avoiding redundant coordinate transformations and Bessel function
evaluations.

## Arguments
* `basis`: The [`FourierBessel`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the basis function and its gradient are evaluated.

## Returns
*  `(bf, dx, dy)` : Basis function values and the `x` and `y` components of its gradient at the input points.
"""
function basis_and_gradient(basis::FourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real}
    r, φ = _fb_polar_coords(basis, pts)
    M = length(pts)
    j = basis.orders[i]
    τ = basis.trig[i]
    bf = Vector{T}(undef, M)
    dx = Vector{T}(undef, M)
    dy = Vector{T}(undef, M)
    @inbounds for p in 1:M
        bf[p], dx[p], dy[p] = _fb_term_and_gradient(j, τ, k, r[p], φ[p])
    end
    return bf, dx, dy
end

"""
    basis_and_gradient(basis::FourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → (B, dB_dx, dB_dy)::Tuple{Matrix{T},Matrix{T},Matrix{T}}

Evaluate both the Fourier-Bessel basis functions with the given `indices` and
their gradients with respect to `x` and `y` on the points `pts`.

## Arguments
* `basis`: The [`FourierBessel`](@ref) basis.
* `indices`: Indices of the basis functions to evaluate.
* `k`: Wavenumber.
* `pts`: Points at which the basis functions and gradients are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `(B, dB_dx, dB_dy)` : Basis matrix and the `x` and `y` components of its gradients, each of size `(length(pts), length(indices))`.
"""
function basis_and_gradient(basis::FourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool=true) where {T<:Real}
    r, φ = _fb_polar_coords(basis, pts)
    M = length(pts)
    N = length(indices)
    B = Matrix{T}(undef, M, N)
    dB_dx = Matrix{T}(undef, M, N)
    dB_dy = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c in 1:N
        j = basis.orders[indices[c]]
        τ = basis.trig[indices[c]]
        bc = @view B[:, c]
        cx = @view dB_dx[:, c]
        cy = @view dB_dy[:, c]
        @inbounds for p in 1:M
            bc[p], cx[p], cy[p] = _fb_term_and_gradient(j, τ, k, r[p], φ[p])
        end
    end
    return B, dB_dx, dB_dy
end
