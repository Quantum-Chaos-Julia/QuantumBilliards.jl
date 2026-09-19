"""
    BasisState{K,T,Ba} <: AbsState

`BasisState` is a concrete type representing a single, unmixed basis function
of `basis` as a stationary state, useful for inspecting or plotting individual
basis functions with the same `AbsState` interface as [`BasisEigenstate`](@ref).

## Description
The expansion coefficient vector `vec` is a unit vector with a `1` in
position `idx` and zeros elsewhere, so that the resulting state is exactly
the `idx`-th basis function of `basis` evaluated at wavenumber `k`.

## Attributes
* `k`: The wavenumber at which the basis function is evaluated.
* `k_basis`: The wavenumber at which `basis` was evaluated to obtain `vec` (equal to `k`).
* `vec`: Expansion coefficients in `basis`, a unit vector with a `1` in position `idx`.
* `idx`: Index of the represented basis function.
* `dim`: Dimension of `basis` (and of `vec`).
* `eps`: Numerical precision threshold, given by `set_precision`.
* `basis`: The basis (`Ba<:AbsBasis`) the state is expressed in.

## API
The following functions can be evaluated for this type:
- [`wavefunction`](@ref)
"""
struct BasisState{K,T,Ba} <: AbsState 
    k::K
    k_basis::K
    vec::Vector{T}
    idx::Int64
    dim::Int64
    eps::T
    basis::Ba
end

"""
    BasisState(basis, k, i) → state::BasisState

Construct a [`BasisState`](@ref) representing the `i`-th basis function of
`basis` at wavenumber `k`.

## Arguments
* `basis`: The basis the state is expressed in; its dimension `basis.dim` sets the length of `vec`.
* `k`: The wavenumber at which the basis function is evaluated.
* `i`: Index of the basis function to represent.

## Returns
*  `state` : A new [`BasisState`](@ref) with a unit coefficient vector having `1` at index `i`.
"""
function BasisState(basis, k, i)  
    dim = basis.dim
    typ = typeof(k)
    eps = set_precision(k)
    vec = zeros(typ,dim)
    vec[i] = one(typ)
    return BasisState(k,k, vec, i, dim, eps, basis)
end
