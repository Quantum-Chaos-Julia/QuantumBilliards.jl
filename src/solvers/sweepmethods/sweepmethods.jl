"""
    solve_wavenumber(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k, dk; multithreaded::Bool = true)

Finds the wavenumber `k0` within `[k - dk/2, k + dk/2]` via `Optim.optimize` that minimizes the tension
computed by the sweep `solver`, together with the minimal tension `t0`.

## Arguments
* `solver::SweepBasisSolver`: The sweep solver used to solve for the tension at each wavenumber.
* `basis::AbsBasis`: The basis used to approximate the eigenstate.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `k::Real`: The center of the wavenumber search window.
* `dk::Real`: Width of the wavenumber search window, `[k - dk/2, k + dk/2]`.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `k0::Real`: The wavenumber minimizing the tension within the search window.
* `t0::Real`: The minimal tension found at `k0`.
"""
function solve_wavenumber(solver::SweepBasisSolver,basis::AbsBasis, billiard::AbsBilliard, k, dk; multithreaded=true)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    new_basis = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    function f(k)
        return solve(solver,new_basis,pts,k;multithreaded)
    end
    res =  optimize(f, k-0.5*dk, k+0.5*dk)
    k0,t0 = res.minimizer, res.minimum
    return k0, t0
end

"""
    k_sweep(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, ks; multithreaded::Bool = true, show_progress::Bool = false)

Computes the tension of the sweep `solver` at every wavenumber in `ks`, using a
single basis resized to the largest wavenumber in `ks`.

## Arguments
* `solver::SweepBasisSolver`: The sweep solver used to solve for the tension at each wavenumber.
* `basis::AbsBasis`: The basis used to approximate the eigenstate.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `ks::AbstractVector`: Wavenumbers at which the tension is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.
* `show_progress::Bool = false`: Whether a progress bar is displayed during the sweep.

## Returns
* `res::AbstractVector`: Tensions corresponding to the wavenumbers in `ks`.
"""
function k_sweep(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, ks; multithreaded=true, show_progress=false)
    k = maximum(ks)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    new_basis = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    res = similar(ks)
    @maybe_showprogress show_progress for (i,k) in enumerate(ks)
        res[i] = solve(solver,new_basis,pts,k; multithreaded)
    end
    return res
end

"""
    _bim_geometry(solver::SweepBIMSolver, pts::BoundaryPoints)

Wavenumber-independent data reused across calls of `construct_matrices(solver, pts, k, geom)`.
Solvers without reusable geometry return `nothing`, for which `construct_matrices` falls back to the
three-argument method.
"""
_bim_geometry(::SweepBIMSolver, pts) = nothing
construct_matrices(solver::SweepBIMSolver, pts, k, ::Nothing; multithreaded::Bool=true) = construct_matrices(solver, pts, k; multithreaded)

"""
    solve_wavenumber(solver::SweepBIMSolver, billiard::AbsBilliard, k, dk; multithreaded::Bool = true, pts::Union{Nothing,BoundaryPoints} = nothing) → (k0::Real, t0::Real)

Finds the wavenumber `k0` within `[k - dk/2, k + dk/2]` via `Optim.optimize` that minimizes the tension
computed by the boundary-integral sweep `solver`, together with the minimal tension `t0`.

## Arguments
* `solver::SweepBIMSolver`: The boundary-integral sweep solver used to solve for the tension at each wavenumber.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `k::Real`: The center of the wavenumber search window.
* `dk::Real`: Width of the wavenumber search window, `[k - dk/2, k + dk/2]`.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.
* `pts::Union{Nothing,BoundaryPoints} = nothing`: Optional precomputed boundary discretization; when `nothing`, one is generated at `k`.

## Returns
* `k0::Real`: The wavenumber minimizing the tension within the search window.
* `t0::Real`: The minimal tension found at `k0`.
"""
function solve_wavenumber(solver::SweepBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true, pts::Union{Nothing,BoundaryPoints}=nothing) where {Bi<:AbsBilliard}
    pts_ = pts===nothing ? evaluate_points(solver, billiard, k) : pts
    function f(k)
        return solve(solver, pts_, k; multithreaded)
    end
    res = optimize(f, k-0.5*dk, k+0.5*dk)
    k0, t0 = res.minimizer, res.minimum
    return k0, t0
end

"""
    k_sweep(solver::SweepBIMSolver, billiard::AbsBilliard, ks; multithreaded::Bool = true)

Computes the tension of the boundary-integral sweep `solver` at every
wavenumber in `ks`, using a single boundary discretization sized for the
largest wavenumber in `ks`.

## Arguments
* `solver::SweepBIMSolver`: The boundary-integral sweep solver used to solve for the tension at each wavenumber.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `ks::AbstractVector`: Wavenumbers at which the tension is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.
* `show_progress::Bool = false`: Whether a progress bar is displayed during the sweep.

## Returns
* `res::AbstractVector`: Tensions corresponding to the wavenumbers in `ks`.
"""
function k_sweep(solver::SweepBIMSolver, billiard::Bi, ks; multithreaded::Bool=true, show_progress::Bool=false) where {Bi<:AbsBilliard}
    k = maximum(ks)
    pts = evaluate_points(solver, billiard, k)
    res = similar(ks)
    @maybe_showprogress show_progress for (i,k) in enumerate(ks)
        res[i] = solve(solver, pts, k; multithreaded)
    end
    return res
end

"""
    _fold_boundary(::Type{T}, billiard::Bi, N::Int, symmetry::AbsSymmetry, character::Tuple) where {T<:Real,Bi<:AbsBilliard} → orbits::SymmetryOrbitMap{T}

Construct the exact symmetry-orbit map used to fold or expand one invariant
physical boundary component.

## Description
The orbit map is derived algebraically from the physical-boundary
reconstruction encoded by `billiard.symmetries`. No boundary coordinates are
used and no floating-point point matching is performed.

The stored solver character is converted to the representation expected by
[`BilliardGeometry.symmetry_index_orbits`](@ref). Ordinary symmetry types
receive character entries as separate scalar arguments, while
[`CompositeReflection`](@ref) receives them as a single
`Vector{Complex{T}}`.

## Arguments
* `T::Type{<:Real}`: Real scalar type used by the boundary discretization.
* `billiard::Bi`: Billiard defining the physical symmetry reconstruction.
* `N::Int`: Number of nodes on the complete invariant physical component.
* `symmetry::AbsSymmetry`: Active boundary symmetry.
* `character::Tuple`: Characters of the selected symmetry irrep.

## Returns
* `orbits::SymmetryOrbitMap{T}`: Exact symmetry-orbit map.
"""
_fold_boundary(::Type{T}, billiard::Bi, N::Int, symmetry::AbsSymmetry, character::Tuple) where {T<:Real,Bi<:AbsBilliard} = BilliardGeometry.symmetry_index_orbits(T, billiard, N, symmetry, _orbit_character_args(T, symmetry, character)...)

# Arguments that follow the symmetry in `symmetry_index_orbits` and `group_symmetry_orbits`.
# Ordinary symmetries take the characters as separate scalars, `CompositeReflection` as one
# vector, and `NFoldRotation` the integer Cₙ sector.
_orbit_character_args(::Type{T}, ::AbsSymmetry, character::Tuple) where {T<:Real} = Complex{T}.(character)
_orbit_character_args(::Type{T}, ::CompositeReflection, character::Tuple) where {T<:Real} = (Complex{T}[character...],)

function _orbit_character_args(::Type{T}, ::NFoldRotation, character::Tuple) where {T<:Real}
    isempty(character) && return ()
    length(character) == 1 || throw(ArgumentError("NFoldRotation requires exactly one sector"))
    sector = only(character)
    sector isa Int || throw(ArgumentError("NFoldRotation sector must be an Int"))
    return (sector,)
end

# Orbit map of boundary group `group` (0 outer boundary, `i` the `i`-th hole object) acted on by the symmetry group of the billiard.
_fold_group(::Type{T}, billiard::Bi, group::Int, N::Int, symmetry::AbsSymmetry, character::Tuple) where {T<:Real,Bi<:AbsBilliard} = BilliardGeometry.group_symmetry_orbits(T, billiard, group, N, symmetry, _orbit_character_args(T, symmetry, character)...)

# Node-count multiple that the boundary discretization of a solver with `symmetry` must satisfy on its own.
# Reflections about axes other than the four special ones need no multiple here: the composite path
# supplies the number of reconstruction blocks of the boundary group through `multiple`.
_bim_node_multiple(::Nothing) = 1
_bim_node_multiple(symmetry::AbsSymmetry) = symmetry_node_multiple(symmetry)
_bim_node_multiple(symmetry::AxisReflection{A}) where {A} = A in (0//1, 1//4, 1//2, 3//4) ? symmetry_node_multiple(symmetry) : 1
_bim_node_multiple(symmetry::CompositeReflection) = foldl(lcm, (_bim_node_multiple(ref) for ref in symmetry.reflections); init = 1)

# Node multiple of a solver: its symmetry's own multiple and the number of reconstruction blocks of the outer boundary.
_bim_solver_node_multiple(solver) = isnothing(solver.symmetry) ? 1 : lcm(_bim_node_multiple(solver.symmetry), length(solver.billiard.symmetries) + 1)

# Round `N` up to a multiple of `needed`, or return the forced count `N_override`.
@inline _bim_round_nodes(N::Int, needed::Int, ::Nothing) = cld(N, needed) * needed
@inline _bim_round_nodes(::Int, ::Int, N_override::Int) = N_override

# Number of boundary nodes of a DLP/CFIE discretization of a component of length `Ltot` at wavenumber `k`.
# A globally graded discretization with a symmetry does not force an even count.
@inline function _bim_node_count(solver, Ltot::Real, k::T; graded::Bool=false, multiple::Int=1, N_override::Union{Nothing,Int}=nothing) where {T<:Real}
    N = max(solver.min_pts, round(Int, k*Ltot*solver.pts_scaling_factor[1]/(2*T(pi))))
    even = graded && solver.symmetry !== nothing ? 1 : 2
    return _bim_round_nodes(N, lcm(even, _bim_solver_node_multiple(solver), multiple), N_override)
end


# Infer the discrete symmetry used to fold the full-boundary Fredholm operator
# directly from `billiard`'s minimal registered symmetry generators
# (`BilliardGeometry.get_symmetries`), given only the per-generator characters
# supplied through a BIM solver's `sym_characters` keyword. This lets DLP/CFIE
# constructors accept `sym_characters` alone instead of requiring an explicit
# `symmetry` object built by hand. A lone rotation generator is returned
# directly (so the `NFoldRotation`-specific `_fold_boundary` method applies);
# one or more reflection generators are wrapped in `CompositeReflection`. When
# `sym_characters` is empty, `(nothing, sym_characters)` is returned unchanged.
function _infer_bim_symmetry(billiard::Bi, sym_characters::Tuple) where {Bi<:AbsBilliard}
    isempty(sym_characters) && return nothing, sym_characters
    sector = symmetry_sector(billiard, sym_characters...)
    return _resolve_bim_symmetry(billiard, sector)
end

################################################################################
# BIM STATE CONSTRUCTION
#
# The native BIM layer density and the physical boundary normal derivative
# u = ∂ₙψ are distinct boundary quantities.
#
# For every kernel (DLP, CFIE, composite), grading and symmetry sector, the
# discrete operator is built from Nyström and Kress blocks whose weighted
# transpose is the adjoint operator, so u is recovered from the left singular
# vector l of the Fredholm matrix:
#
#     u_raw = W⁻¹ l,     W = diag(ds),
#
# expanded to the complete boundary with the orbit phases (no conjugation).
#
#     vec = full-boundary layer density
#     u   = full-boundary Rellich-normalized ∂ₙψ
################################################################################

"""
    _boundary_orbits(solver::AbsBIMSolver, pts::BoundaryPoints{T}, billiard::Bi) where {T<:Real,Bi<:AbsBilliard} → SymmetryOrbitMap{T}

Symmetry orbit map of the complete boundary `pts` for a solver with an active
symmetry: the construction-derived map of a single closed curve
(`_fold_boundary`) or the combined map of a `CompositeBIMSolver`
(`_composite_symmetry_orbits`).
"""
function _boundary_orbits(solver::AbsBIMSolver, pts::BoundaryPoints{T}, billiard::Bi) where {T<:Real,Bi<:AbsBilliard}
    solver.symmetry === nothing && throw(ArgumentError("_boundary_orbits requires a nontrivial symmetry"))
    return solver isa CompositeBIMSolver ? _composite_symmetry_orbits(T, solver, pts) : _fold_boundary(T, billiard, length(pts), solver.symmetry, solver.sym_characters)
end

"""
    symmetrize_layer_density(solver::AbsBIMSolver, layer_density::AbstractVector{N}, pts::BoundaryPoints{T}, billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}

Expand a symmetry-reduced boundary density onto the complete physical boundary.

If no symmetry is active, `layer_density` must already have the same length as
`pts`. When symmetry reduction is active, a fundamental-domain density is
expanded using the exact [`BilliardGeometry.SymmetryOrbitMap`](@ref) defined
by the billiard's physical-boundary reconstruction, `solver.symmetry`, and
`solver.sym_characters`. An already full-length density is returned unchanged.

For a `CompositeBIMSolver`, every physical boundary component is required to
be individually invariant under the active symmetry. Symmetries that exchange
distinct physical components are currently unsupported.

## Arguments
* `solver::AbsBIMSolver`: BIM solver defining the active symmetry and character.
* `layer_density::AbstractVector{N}`: Boundary density on either the reduced or complete physical boundary.
* `pts::BoundaryPoints{T}`: Complete physical-boundary discretization.
* `billiard::Bi`: Billiard defining the physical symmetry reconstruction.

## Returns
* `full_density::Vector`: Boundary density on the complete physical boundary.
"""
function symmetrize_layer_density(solver::AbsBIMSolver, layer_density::AbstractVector{N}, pts::BoundaryPoints{T}, billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull = length(pts)
    length(layer_density) == Nfull && return layer_density
    solver.symmetry === nothing && throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    orbits = _boundary_orbits(solver, pts, billiard)
    Nred = fundamental_size(orbits)
    length(layer_density) == Nred || throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S = promote_type(N, Complex{T})
    full_data = Vector{S}(undef, Nfull)
    @inbounds for q in 1:Nfull
        full_data[q] = orbits.phase[q] * layer_density[orbits.orbit_of[q]]
    end
    return full_data
end

"""
    _bim_normal_derivative(solver::AbsBIMSolver, pts::BoundaryPoints{T}, lvec::AbstractVector, billiard::Bi) where {T<:Real,Bi<:AbsBilliard}

Recover the unnormalized physical boundary normal derivative `u = ∂ₙψ` on the
complete boundary from the left singular vector `lvec` of a Fredholm matrix.

The Nyström/Kress discretization has the weighted transpose
`A_adj = W⁻¹ Aᵀ W`, `W = diag(ds)`, as its adjoint, so for every kernel (DLP,
CFIE, composite), grading and symmetry sector

    u_raw = W⁻¹ lvec,

without conjugation. For a symmetry-reduced solve, `lvec ./ ds` at the
fundamental orbit nodes is expanded with the orbit phases by
[`symmetrize_layer_density`](@ref).

## Arguments
* `solver::AbsBIMSolver`: BIM solver defining the active symmetry and character.
* `pts::BoundaryPoints{T}`: Complete physical-boundary discretization.
* `lvec::AbstractVector`: Left singular vector of the (reduced) Fredholm matrix.
* `billiard::Bi`: Billiard defining the physical symmetry reconstruction.

## Returns
* `u_raw::Vector`: Unnormalized physical boundary normal derivative on the complete boundary.
"""
function _bim_normal_derivative(solver::AbsBIMSolver, pts::BoundaryPoints{T}, lvec::AbstractVector, billiard::Bi) where {T<:Real,Bi<:AbsBilliard}
    solver.symmetry === nothing && return lvec ./ pts.ds
    fund = _boundary_orbits(solver, pts, billiard).fundamental_indices
    return symmetrize_layer_density(solver, lvec ./ pts.ds[fund], pts, billiard)
end

"""
    solve_state(solver::Union{DLP,CFIE}, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool = true) where {T<:Real,Bi<:AbsBilliard} → (ten::Real, vec::Vector{Complex{T}}, u::Vector, bnd_norm::T)

Solve the eigenproblem of a DLP, CFIE or composite solver at `k` and recover
both the native layer density and the physical boundary normal derivative
`u = ∂ₙψ`.

The Fredholm matrix `A(k)` is assembled once and its smallest singular triplet
is computed with a single `KrylovKit.svdsolve` call. The right singular vector
gives the native layer density; the left singular vector gives
`u_raw = W⁻¹ l` through `_bim_normal_derivative`. Both are expanded onto the
complete physical boundary when symmetry reduction is active, and `u` is
normalized with the Rellich identity. Thus `vec`, `u` and `pts` refer to the
same discretization.

## Arguments
- `solver::Union{DLP,CFIE}`: Solver used to construct the Fredholm matrix.
- `pts::BoundaryPoints{T}`: Complete physical-boundary discretization.
- `k`: Eigenvalue wave number.
- `billiard::Bi`: Associated billiard.

## Keyword Arguments
- `multithreaded::Bool = true`: Use multithreaded Fredholm-matrix construction.

## Returns
- `ten::Real`: Smallest singular value of the Fredholm matrix.
- `vec::Vector{Complex{T}}`: Layer density on the complete physical boundary.
- `u::Vector`: Rellich-normalized physical boundary normal derivative `∂ₙψ` on the complete physical boundary.
- `bnd_norm::T`: Rellich norm of the unnormalized physical boundary derivative.
"""
function solve_state(solver::Union{DLP,CFIE}, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool = true) where {T<:Real,Bi<:AbsBilliard}
    kT = T(k)
    A = construct_matrices(solver, pts, kT; multithreaded)
    @blas_1 vals, lvecs, rvecs, _ = KrylovKit.svdsolve(A, 5, :SR)
    ten = vals[1]
    vec = symmetrize_layer_density(solver, Vector{Complex{T}}(rvecs[1]), pts, billiard)
    u = _bim_normal_derivative(solver, pts, lvecs[1], billiard)
    bnd_norm = _rellich(pts, u, kT)
    u ./= sqrt(bnd_norm)
    return ten, vec, u, bnd_norm
end

"""
    _bim_dlp_cross_kernel_entry(pts::BoundaryPoints{T}, x::T, y::T, k::Union{T,Complex{T}}, j::Int) where {T<:Real} → value::Complex{T}

Evaluates one smooth double-layer Nyström entry from source node `j` on one
boundary component to a target point `(x,y)` on another component.

## Arguments
* `pts::BoundaryPoints{T}`: Source-component boundary discretization.
* `x::T`: Target x-coordinate.
* `y::T`: Target y-coordinate.
* `k::Union{T,Complex{T}}`: Helmholtz wavenumber.
* `j::Int`: Source-node index.

## Returns
* `value::Complex{T}`: Smooth cross-component double-layer entry.
"""
@inline function _bim_dlp_cross_kernel_entry(pts::BoundaryPoints{T}, x::T, y::T, k::Union{T,Complex{T}}, j::Int) where {T<:Real}
    xj, yj = pts.xy[j]; dx = x-xj; dy = y-yj; r = hypot(dx,dy)
    tx, ty = pts.tangent[j]
    return pts.ws[j]*im*k/2*(ty*dx-tx*dy)*_bim_hankelh1(1,k*r)/r
end
