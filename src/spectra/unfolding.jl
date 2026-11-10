

corner_correction(corner_angles) =  sum([(pi^2 - c^2)/(24*pi*c) for c in corner_angles])

"""
    weyl_law(k, A, L) → N
    weyl_law(k, A, L, corner_angles) → N

Smooth Weyl state-counting function `N(k) = (A*k^2 - L*k)/(4π)` of a Dirichlet
billiard of area `A` and boundary length `L`. The second method adds the
constant corner correction `Σ (π² - c²)/(24π c)` over the interior corner angles `c`.

## Arguments
* `k`: Wavenumber or array of wavenumbers.
* `A`: Billiard area.
* `L`: Boundary length.
* `corner_angles`: Interior corner angles.

## Returns
* `N`: Expected number of levels below `k`, with the shape of `k`.
"""
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

# A billiard with holes and a nonempty outer registry has no fundamental-frame boundary; its reduced Weyl
# inputs come from the complete boundary divided by the order of its symmetry group G'.
_has_reduced_holes(billiard::AbsBilliard) = genus(billiard) > 0 && !isempty(billiard.symmetries)

# Length of the `SymmetryWall` cuts of the outer boundary of the fundamental frame.
_outer_wall_length(billiard::AbsBilliard, ::Type{T}) where {T<:Real} =
    sum(crv.length for crv in first(geometric_boundary_components(billiard)) if crv.bc isa SymmetryWall; init=zero(T))

# Length of the walls of a hole's own pieces that lie on mirror lines or cuts of the billiard group G'
# (`keys` are the `symmetry_key`s of G'); every such chord is seen from both adjacent pieces.
_hole_group_wall_length(::AbsDomainHole, keys, ::Type{T}) where {T<:Real} = zero(T)
function _hole_group_wall_length(h::SymmetricHole, keys, ::Type{T}) where {T<:Real}
    return sum(crv.length for crv in get_all_curves(h)
               if crv.bc isa SymmetryWall && BilliardGeometry.symmetry_key(symmetry_of(h.symmetries, crv.bc.sym_id)) in keys; init=zero(T))
end

function _hole_group_wall_length(billiard::AbsBilliard, ::Type{T}) where {T<:Real}
    keys = Set(BilliardGeometry.symmetry_key(g) for g in get_symmetry_group(billiard))
    return sum(_hole_group_wall_length(leaf, keys, T) for h in billiard.fundamental_domain.holes for leaf in hole_leaves(h); init=zero(T))
end

# Length of the outer `QuantumSolverIgnore` curves of the fundamental frame. They are Dirichlet walls that the
# basis satisfies automatically, so they belong to the Weyl boundary length although solvers skip them.
_ignored_curves(domain::AbsSimpleDomain) = domain.boundary
_ignored_curves(domain::AbsCompositeDomain) = AbsCurve[crv for sd in domain.subdomains for crv in sd.boundary]
_ignored_curves(domain::AbsMultiplyConnectedDomain) = AbsCurve[crv for sd in domain.subdomains for crv in sd.boundary]
_ignored_length(billiard::AbsBilliard, ::Type{T}) where {T<:Real} =
    sum(crv.length for crv in _ignored_curves(billiard.fundamental_domain) if crv.bc isa QuantumSolverIgnore; init=zero(T))

# Area `A` and boundary length `L` of the Weyl law. With `fundamental_only` the domain is the desymmetrized
# fundamental domain with Dirichlet conditions on the symmetry walls, so the wall cuts add to `L`. For holes
# with a nonempty outer registry the tessellation identity gives
# L = (L_full - chords of hole walls on G' axes)/|G'| + outer wall length, where |G'| counts the identity.
function _weyl_area_length(billiard::AbsBilliard, fundamental_only::Bool)
    A = fundamental_only ? fundamental_area(billiard) : area(billiard)
    T = typeof(A)
    ignored = _ignored_length(billiard, T)
    if !fundamental_only
        L = _boundary_length(full_boundary(billiard)) + (length(billiard.symmetries) + 1) * ignored
    elseif _has_reduced_holes(billiard) && isempty(get_symmetry_group(billiard))
        L = _boundary_length(full_boundary(billiard)) + ignored
    elseif _has_reduced_holes(billiard)
        n = length(get_symmetry_group(billiard)) + 1
        L = (_boundary_length(full_boundary(billiard)) - _hole_group_wall_length(billiard, T)) / n + _outer_wall_length(billiard, T) + ignored
    else
        L = _boundary_length(get_boundary_curves(billiard)) + _outer_wall_length(billiard, T) + ignored
    end
    return A, L::T
end

# One corner per orbit of G' (corners lie in free orbits; corners where a symmetry wall meets the boundary
# are omitted, an O(1) constant of N(k)): the sorted corner angles of the complete boundary, every |G'|-th.
function _weyl_angles(billiard::AbsBilliard, fundamental_only::Bool)
    if fundamental_only && _has_reduced_holes(billiard)
        angles = sort(corner_angles(billiard; fundamental=false))
        return angles[1:(length(get_symmetry_group(billiard)) + 1):end]
    end
    return corner_angles(billiard; fundamental=fundamental_only)
end

# Area, boundary length and corner angles of the Weyl law of `billiard`.
function _weyl_inputs(billiard::AbsBilliard, fundamental_only::Bool)
    A, L = _weyl_area_length(billiard, fundamental_only)
    return A, L, _weyl_angles(billiard, fundamental_only)
end

"""
    k_at_state(state, billiard::AbsBilliard; fundamental_only::Bool=false) → k::Real

Wavenumber at which Weyl's law state-counting function (including the
corner correction) reaches `state`.

## Keyword Arguments
* `fundamental_only::Bool = false`: When `false` (default), the Weyl law of the
  complete billiard (`area`, `full_boundary`, all corners). When `true`, the
  Weyl law of the desymmetrized fundamental domain with Dirichlet conditions on
  every symmetry wall (`fundamental_area`, physical boundary plus the
  `SymmetryWall` cuts, corners where a cut meets the boundary included); this is
  the all-Dirichlet sector. For billiards with holes and a nonempty symmetry
  registry the wall term accounts for hole chords on the symmetry axes, while
  the corner correction omits the cut/hole junction corners (an O(1) constant).
"""
function k_at_state(state, billiard::AbsBilliard; fundamental_only::Bool=false)
    A, L, angles = _weyl_inputs(billiard, fundamental_only)
    return isempty(angles) ? k_at_state(state, A, L) : k_at_state(state, A, L, angles)
end

"""
    state_at_k(k, A, L) → N::Real
    state_at_k(k, A, L, corner_angles) → N::Real
    state_at_k(k, billiard::AbsBilliard; fundamental_only::Bool=false) → N::Real

Weyl's law state-counting function evaluated at wavenumber `k` — the inverse
direction of [`k_at_state`](@ref) — provided under this name for symmetry
with it (equivalent to [`weyl_law`](@ref)).

## Keyword Arguments
* `fundamental_only::Bool = false`: See [`k_at_state`](@ref).
"""
state_at_k(k, A, L) = weyl_law(k, A, L)
state_at_k(k, A, L, corner_angles) = weyl_law(k, A, L, corner_angles)

function state_at_k(k, billiard::AbsBilliard; fundamental_only::Bool=false)
    A, L, angles = _weyl_inputs(billiard, fundamental_only)
    return isempty(angles) ? state_at_k(k, A, L) : state_at_k(k, A, L, angles)
end

"""
    spectral_density(k, A, L) → dNdk::Real
    spectral_density(k, billiard::AbsBilliard; fundamental_only::Bool=false) → dNdk::Real

Derivative `dN/dk` of Weyl's law state-counting function,
`(A*k - L/2)/(2π)` — the local mean density of states at wavenumber `k`,
used to size adaptive wavenumber steps in [`compute_spectrum`](@ref).

## Keyword Arguments
* `fundamental_only::Bool = false`: See [`k_at_state`](@ref).
"""
spectral_density(k, A, L) = (A*k - L/2)/(2*pi)

function spectral_density(k, billiard::AbsBilliard; fundamental_only::Bool=false)
    A, L = _weyl_area_length(billiard, fundamental_only)
    return spectral_density(k, A, L)
end

"""
    k_range_for_states(billiard::AbsBilliard, N1::Int, N2::Int; fundamental_only::Bool=false) → (k1,k2)

Wavenumber range `(k1,k2)` bracketing states `N1` to `N2` of the billiard's
Weyl-law state-counting function, via two [`k_at_state`](@ref) calls.

## Keyword Arguments
* `fundamental_only::Bool = false`: See [`k_at_state`](@ref).
"""
function k_range_for_states(billiard::AbsBilliard, N1::Int, N2::Int; fundamental_only::Bool=false)
    return k_at_state(N1, billiard; fundamental_only), k_at_state(N2, billiard; fundamental_only)
end
