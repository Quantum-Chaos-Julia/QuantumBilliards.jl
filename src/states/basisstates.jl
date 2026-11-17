"""
    BasisState(basis, k, i) → state::BasisEigenstate

Construct the [`BasisEigenstate`](@ref) representing the single, unmixed `i`-th
basis function of `basis` at wavenumber `k`, useful for inspecting or plotting
individual basis functions through the same interface as eigenstates.

## Description
The expansion coefficient vector `vec` of the returned state is a unit vector
with a `1` in position `i` and zeros elsewhere, `k_basis = k` and the tension
is `ten = 0`.

## Arguments
* `basis`: The basis the state is expressed in; its dimension `basis.dim` sets the length of `vec`.
* `k`: The wavenumber at which the basis function is evaluated.
* `i`: Index of the basis function to represent; a `BoundsError` is thrown outside `1:basis.dim`.

## Returns
* `state::BasisEigenstate`: State with a unit coefficient vector having `1` at index `i`.
"""
function BasisState(basis, k::K, i::Integer) where {K<:Real}
    vec = zeros(K, basis.dim)
    vec[i] = one(K)
    return BasisEigenstate(k, vec, zero(K), basis)
end
