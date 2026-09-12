

corner_correction(corner_angles) =  sum([(pi^2 - c^2)/(24*pi*c) for c in corner_angles])

weyl_law(k,A,L) =  @. (A * k^2 - L * k)/(4*pi)
weyl_law(k,A,L,corner_angles) =  weyl_law(k,A,L) .+ corner_correction(corner_angles)


function k_at_state(state, A, L)
    a = A
    b = -L
    c = -state*4*pi
    dis = sqrt(b^2-4*a*c)
    return (-b+dis)/(2*a)
end

function k_at_state(state, A, L, corner_angles)
    a = A
    b = -L
    c = (corner_correction(corner_angles)-state)*4*pi 
    dis = sqrt(b^2-4*a*c)
    return (-b+dis)/(2*a)
end

# Total boundary arc length of a collection of curves (as returned by
# `get_boundary_curves`/`full_boundary`), shared by every billiard-level
# Weyl-law function below.
_boundary_length(curves) = sum(crv.length for crv in curves)

"""
    k_at_state(state, billiard::AbsBilliard; fundamental::Bool=true) → k::Real

Wavenumber at which Weyl's law state-counting function (including the
corner correction) reaches `state`.

## Keyword Arguments
* `fundamental::Bool = true`: When `true` (default), uses the billiard's
  fundamental-domain boundary (`get_boundary_curves`), matching what an
  `AcceleratedBasisSolver`/`AcceleratedBIMSolver` actually diagonalizes over;
  when `false`, uses the complete physical boundary (`full_boundary`).
"""
function k_at_state(state, billiard::AbsBilliard; fundamental::Bool=true)
    curves = fundamental ? get_boundary_curves(billiard) : full_boundary(billiard)
    A = fundamental ? fundamental_area(billiard) : area(billiard)
    L = _boundary_length(curves)
    angles = corner_angles(billiard; fundamental)
    return isempty(angles) ? k_at_state(state, A, L) : k_at_state(state, A, L, angles)
end

"""
    state_at_k(k, A, L) → N::Real
    state_at_k(k, A, L, corner_angles) → N::Real
    state_at_k(k, billiard::AbsBilliard; fundamental::Bool=true) → N::Real

Weyl's law state-counting function evaluated at wavenumber `k` — the inverse
direction of [`k_at_state`](@ref) — provided under this name for symmetry
with it (equivalent to [`weyl_law`](@ref)).

## Keyword Arguments
* `fundamental::Bool = true`: See [`k_at_state`](@ref).
"""
state_at_k(k, A, L) = weyl_law(k, A, L)
state_at_k(k, A, L, corner_angles) = weyl_law(k, A, L, corner_angles)

function state_at_k(k, billiard::AbsBilliard; fundamental::Bool=true)
    curves = fundamental ? get_boundary_curves(billiard) : full_boundary(billiard)
    A = fundamental ? fundamental_area(billiard) : area(billiard)
    L = _boundary_length(curves)
    angles = corner_angles(billiard; fundamental)
    return isempty(angles) ? state_at_k(k, A, L) : state_at_k(k, A, L, angles)
end

"""
    spectral_density(k, A, L) → dNdk::Real
    spectral_density(k, billiard::AbsBilliard; fundamental::Bool=true) → dNdk::Real

Derivative `dN/dk` of Weyl's law state-counting function,
`(A*k - L/2)/(2π)` — the local mean density of states at wavenumber `k`,
used to size adaptive wavenumber steps in [`compute_spectrum`](@ref).

## Keyword Arguments
* `fundamental::Bool = true`: See [`k_at_state`](@ref).
"""
spectral_density(k, A, L) = (A*k - L/2)/(2*pi)

function spectral_density(k, billiard::AbsBilliard; fundamental::Bool=true)
    curves = fundamental ? get_boundary_curves(billiard) : full_boundary(billiard)
    A = fundamental ? fundamental_area(billiard) : area(billiard)
    L = _boundary_length(curves)
    return spectral_density(k, A, L)
end

"""
    k_range_for_states(billiard::AbsBilliard, N1::Int, N2::Int; fundamental::Bool=true) → (k1,k2)

Wavenumber range `(k1,k2)` bracketing states `N1` to `N2` of the billiard's
Weyl-law state-counting function, via two [`k_at_state`](@ref) calls.

## Keyword Arguments
* `fundamental::Bool = true`: See [`k_at_state`](@ref).
"""
function k_range_for_states(billiard::AbsBilliard, N1::Int, N2::Int; fundamental::Bool=true)
    return k_at_state(N1, billiard; fundamental), k_at_state(N2, billiard; fundamental)
end
