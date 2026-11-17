################################################################################
# SAMPLED FUNCTION OBJECTS
#
# `wavefunction`, `boundary_function` and `husimi_function` return one of the
# three objects below. Each stores the sampled values together with the
# one-dimensional coordinate vectors and the wavenumber `k` of the state it was
# sampled from, so that it is self-describing. A single object iterates as the
# tuple the corresponding function returned before the objects existed, so
# `Psi, x, y = wavefunction(state, billiard)` keeps working.
################################################################################

function _check_wavenumber(k::Real)
    (isfinite(k) && k > zero(k)) || throw(ArgumentError("wavenumber k must be finite and positive, got $k"))
    return nothing
end

"""
    Wavefunction{T<:Real,V<:Number}

Wavefunction `ψ(x, y)` sampled on a Cartesian grid.

## Attributes
* `psi::Matrix{V}`: Values `psi[ix, iy]` at `(x[ix], y[iy])`. Points excluded by an interior mask hold zero or `NaN`, as requested at the call of [`wavefunction`](@ref).
* `x::Vector{T}`: Grid coordinates along the first dimension of `psi`.
* `y::Vector{T}`: Grid coordinates along the second dimension of `psi`.
* `k::T`: Wavenumber of the state the wavefunction was sampled from.

Iterating a `Wavefunction` yields `(psi, x, y)`. Wavefunctions of one batch that share a grid share the coordinate vectors `x` and `y`.

## Constructor
    Wavefunction(psi::Matrix{V}, x::Vector{T}, y::Vector{T}, k::T)

Throws a `DimensionMismatch` unless `size(psi) == (length(x), length(y))` and an `ArgumentError` unless `k` is finite and positive.
"""
struct Wavefunction{T<:Real,V<:Number}
    psi::Matrix{V}
    x::Vector{T}
    y::Vector{T}
    k::T
    function Wavefunction(psi::Matrix{V}, x::Vector{T}, y::Vector{T}, k::T) where {T<:Real,V<:Number}
        size(psi) == (length(x), length(y)) || throw(DimensionMismatch("size(psi) = $(size(psi)) does not match (length(x), length(y)) = ($(length(x)), $(length(y)))"))
        _check_wavenumber(k)
        return new{T,V}(psi, x, y, k)
    end
end

"""
    BoundaryFunction{T<:Real,U<:Number}

Boundary normal derivative `u(s) = ∂ₙψ` of an eigenstate sampled on a boundary discretization.

## Attributes
* `u::Vector{U}`: Values of `∂ₙψ` at the boundary nodes.
* `s::Vector{T}`: Arclength coordinate of the nodes; continuous over the concatenation of all boundary loops (outer loop first, then the hole loops).
* `ds::Vector{T}`: Quadrature weights of the nodes. They are not uniform for graded boundary-integral grids.
* `k::T`: Wavenumber of the state.
* `norm::T`: Rellich value of the state, `NaN` when the state stores none.
* `L::T`: Period of `s` on the outer loop: the outer perimeter of the billiard where it is available and `sum(ds[outer])` otherwise.
* `outer::UnitRange{Int}`: Indices of the nodes of the outer loop (`1:length(u)` for a billiard without holes).

Iterating a `BoundaryFunction` yields `(u, s, norm)`.

## Constructor
    BoundaryFunction(u::Vector{U}, s::Vector{T}, ds::Vector{T}, k::T, norm::T, L::T, outer::UnitRange{Int})

Throws a `DimensionMismatch` unless `u`, `s` and `ds` have equal length and `outer` lies inside `eachindex(u)`, and an `ArgumentError` unless `k` is finite and positive.
"""
struct BoundaryFunction{T<:Real,U<:Number}
    u::Vector{U}
    s::Vector{T}
    ds::Vector{T}
    k::T
    norm::T
    L::T
    outer::UnitRange{Int}
    function BoundaryFunction(u::Vector{U}, s::Vector{T}, ds::Vector{T}, k::T, norm::T, L::T, outer::UnitRange{Int}) where {T<:Real,U<:Number}
        length(u) == length(s) == length(ds) || throw(DimensionMismatch("u, s and ds must have equal length, got $(length(u)), $(length(s)) and $(length(ds))"))
        checkbounds(Bool, u, outer) || throw(DimensionMismatch("outer = $outer is outside 1:$(length(u))"))
        _check_wavenumber(k)
        return new{T,U}(u, s, ds, k, norm, L, outer)
    end
end

"""
    HusimiFunction{T<:Real}

Poincaré-Husimi function `H(q, p)` on a phase-space grid.

## Attributes
* `H::Matrix{T}`: Normalized density `H[iq, ip]` at `(qs[iq], ps[ip])`, with `sum(H) = 1`.
* `qs::Vector{T}`: Boundary-position grid (arclength on the outer loop).
* `ps::Vector{T}`: Tangential-momentum grid in units of `k`.
* `k::T`: Wavenumber of the state.

Iterating a `HusimiFunction` yields `(H, qs, ps)`.

## Constructor
    HusimiFunction(H::Matrix{T}, qs::Vector{T}, ps::Vector{T}, k::T)

Throws a `DimensionMismatch` unless `size(H) == (length(qs), length(ps))` and an `ArgumentError` unless `k` is finite and positive.
"""
struct HusimiFunction{T<:Real}
    H::Matrix{T}
    qs::Vector{T}
    ps::Vector{T}
    k::T
    function HusimiFunction(H::Matrix{T}, qs::Vector{T}, ps::Vector{T}, k::T) where {T<:Real}
        size(H) == (length(qs), length(ps)) || throw(DimensionMismatch("size(H) = $(size(H)) does not match (length(qs), length(ps)) = ($(length(qs)), $(length(ps)))"))
        _check_wavenumber(k)
        return new{T}(H, qs, ps, k)
    end
end

Base.length(::Wavefunction) = 3
Base.iterate(wf::Wavefunction, i::Int = 1) = i == 1 ? (wf.psi, 2) : i == 2 ? (wf.x, 3) : i == 3 ? (wf.y, 4) : nothing

Base.length(::BoundaryFunction) = 3
Base.iterate(bf::BoundaryFunction, i::Int = 1) = i == 1 ? (bf.u, 2) : i == 2 ? (bf.s, 3) : i == 3 ? (bf.norm, 4) : nothing

Base.length(::HusimiFunction) = 3
Base.iterate(hf::HusimiFunction, i::Int = 1) = i == 1 ? (hf.H, 2) : i == 2 ? (hf.qs, 3) : i == 3 ? (hf.ps, 4) : nothing

Base.show(io::IO, wf::Wavefunction{T,V}) where {T,V} = print(io, "Wavefunction{", T, ",", V, "}(k = ", wf.k, ", ", size(wf.psi, 1), "×", size(wf.psi, 2), " grid)")
Base.show(io::IO, bf::BoundaryFunction{T,U}) where {T,U} = print(io, "BoundaryFunction{", T, ",", U, "}(k = ", bf.k, ", ", length(bf.u), " nodes)")
Base.show(io::IO, hf::HusimiFunction{T}) where {T} = print(io, "HusimiFunction{", T, "}(k = ", hf.k, ", ", size(hf.H, 1), "×", size(hf.H, 2), " grid)")
