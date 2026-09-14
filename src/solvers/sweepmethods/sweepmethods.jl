include("decompositionmethod.jl")
include("particularsolutions.jl")
include("boundarygrading.jl")
include("dlp.jl")
include("cfie.jl")
include("compositebim.jl")

"""
    solve_wavenumber(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k, dk; multithreaded::Bool = true) → (k0::Real, t0::Real)

Finds the wavenumber `k0` within `[k - dk/2, k + dk/2]` that minimizes the tension
computed by the sweep `solver`, together with the minimal tension `t0`.

## Description
The basis dimension is scaled with the boundary length and wavenumber via
`solver.dim_scaling_factor` (bounded below by `solver.min_dim`), the basis is
resized to this dimension with [`resize_basis`](@ref), boundary points are
generated once with [`evaluate_points`](@ref), and the tension
`solve(solver, new_basis, pts, k; multithreaded)` is minimized over `k` in the
search window with `Optim.optimize`.

## Arguments
* `solver`: The [`SweepBasisSolver`](@ref) used to solve for the tension at each wavenumber.
* `basis`: The basis used to approximate the eigenstate.
* `billiard`: The billiard whose boundary is discretized.
* `k`: The center of the wavenumber search window.
* `dk`: Width of the wavenumber search window, `[k - dk/2, k + dk/2]`.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `k0`: The wavenumber minimizing the tension within the search window.
* `t0`: The minimal tension found at `k0`.
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
    k_sweep(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, ks; multithreaded::Bool = true) → res::Vector

Computes the tension of the sweep `solver` at every wavenumber in `ks`, using a
single basis resized to the largest wavenumber in `ks`.

## Description
The basis dimension is scaled with the boundary length and `maximum(ks)` via
`solver.dim_scaling_factor` (bounded below by `solver.min_dim`), the basis is
resized to this dimension with [`resize_basis`](@ref), boundary points are
generated once with [`evaluate_points`](@ref), and [`solve`](@ref) is called for
every wavenumber in `ks`.

## Arguments
* `solver`: The [`SweepBasisSolver`](@ref) used to solve for the tension at each wavenumber.
* `basis`: The basis used to approximate the eigenstate.
* `billiard`: The billiard whose boundary is discretized.
* `ks`: Vector (or range) of wavenumbers at which the tension is evaluated.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `res`: Vector of tensions, one for each wavenumber in `ks`.
"""
function k_sweep(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, ks; multithreaded=true)
    k = maximum(ks)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    new_basis = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    res = similar(ks)
    for (i,k) in enumerate(ks)
        res[i] = solve(solver,new_basis,pts,k; multithreaded)
    end
    return res
end

"""
    solve_wavenumber(solver::SweepBIMSolver, billiard::AbsBilliard, k, dk; multithreaded::Bool = true, pts::Union{Nothing,BoundaryPoints} = nothing) → (k0::Real, t0::Real)

Finds the wavenumber `k0` within `[k - dk/2, k + dk/2]` that minimizes the tension
computed by the boundary-integral sweep `solver`, together with the minimal
tension `t0`.

## Description
Boundary points are generated once with [`evaluate_points`](@ref) (unless a
precomputed discretization is passed via `pts`, letting a caller that refines
many nearby wavenumbers, e.g. [`compute_spectrum_refined`](@ref), reuse a
single discretization instead of paying for `evaluate_points` on every call),
and the tension `solve(solver, pts, k; multithreaded)` is minimized over `k`
in the search window with `Optim.optimize`, exactly as
[`solve_wavenumber(::SweepBasisSolver, ...)`](@ref) does for basis-expansion
solvers, but without a basis to resize.

## Arguments
* `solver`: The [`SweepBIMSolver`](@ref) used to solve for the tension at each wavenumber.
* `billiard`: The billiard whose boundary is discretized.
* `k`: The center of the wavenumber search window.
* `dk`: Width of the wavenumber search window, `[k - dk/2, k + dk/2]`.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.
* `pts::Union{Nothing,BoundaryPoints} = nothing`: Optional precomputed boundary discretization (from `evaluate_points(solver, billiard, k')` at some `k'>=k`); when `nothing`, one is generated at `k`.

## Returns
* `k0`: The wavenumber minimizing the tension within the search window.
* `t0`: The minimal tension found at `k0`.
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
    k_sweep(solver::SweepBIMSolver, billiard::AbsBilliard, ks; multithreaded::Bool = true) → res::Vector

Computes the tension of the boundary-integral sweep `solver` at every
wavenumber in `ks`, using a single boundary discretization sized for the
largest wavenumber in `ks`.

## Description
Boundary points are generated once with [`evaluate_points`](@ref), sized for
`maximum(ks)`, and [`solve`](@ref) is called for every wavenumber in `ks`,
exactly as [`k_sweep(::SweepBasisSolver, ...)`](@ref) does for basis-expansion
solvers, but without a basis to resize.

## Arguments
* `solver`: The [`SweepBIMSolver`](@ref) used to solve for the tension at each wavenumber.
* `billiard`: The billiard whose boundary is discretized.
* `ks`: Vector (or range) of wavenumbers at which the tension is evaluated.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `res`: Vector of tensions, one for each wavenumber in `ks`.
"""
function k_sweep(solver::SweepBIMSolver, billiard::Bi, ks; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    k = maximum(ks)
    pts = evaluate_points(solver, billiard, k)
    res = similar(ks)
    for (i,k) in enumerate(ks)
        res[i] = solve(solver, pts, k; multithreaded)
    end
    return res
end

"""
    _fold_boundary(::Type{T}, xy, symmetry::AbsSymmetry, character::Tuple) where {T<:Real} → orbits::SymmetryOrbitMap{T}

Bridges a BIM solver's stored `(symmetry, character::Tuple)` pair to
[`symmetry_index_orbits`](@ref)'s per-generator-type trailing-argument
shape: every symmetry type except [`CompositeReflection`](@ref) takes `N`
separate scalar character arguments (splat-compatible with a stored
`Tuple`), while `CompositeReflection` takes a single `Vector{Complex{T}}`
argument instead. Exists solely because that underlying dispatch table is
genuinely non-uniform across symmetry types (Step 16 of the migration
plan), not as a speculative convenience layer.
"""
_fold_boundary(::Type{T}, xy, symmetry::AbsSymmetry, character::Tuple) where {T<:Real} =
    symmetry_index_orbits(T, xy, symmetry, character...)
_fold_boundary(::Type{T}, xy, symmetry::CompositeReflection, character::Tuple) where {T<:Real} =
    symmetry_index_orbits(T, xy, symmetry, Complex{T}[character...])

"""
    symmetrize_layer_density(solver::AbsBIMSolver, layer_density::AbstractVector, pts::BoundaryPoints, billiard::AbsBilliard) → full_density::Vector

Expands a symmetry-reduced boundary density onto the complete physical
boundary using `solver.symmetry`/`solver.character`'s
[`BilliardGeometry.SymmetryOrbitMap`](@ref).
Generic across every [`AbsBIMSolver`](@ref) (`DoubleLayerPotentialSolver`,
`CombinedFieldIntegralEquationSolver`, `CompositeBIMSolver`, ...): the
folding depends only on `solver.symmetry`, never on the specific Fredholm
kernel.

## Description
`pts` must already be the discretization of the *complete* physical boundary
(`full_boundary(billiard)` when `solver.symmetry !== nothing`, as produced by
[`evaluate_points`](@ref)). If `layer_density` already has full-boundary
length, it is returned unchanged; otherwise it is expanded from the
fundamental-domain length via `orbits.orbit_of`/`orbits.phase`.

## Arguments
* `solver`: The [`AbsBIMSolver`](@ref) whose `symmetry` (if any) defines the folding.
* `layer_density`: The boundary density, either already full-length or fundamental-domain length.
* `pts`: The complete-physical-boundary discretization corresponding to the full-length output.
* `billiard`: The billiard the boundary belongs to (unused beyond dispatch parity with `-develop`, retained for API stability).

## Returns
*  `full_density` : `layer_density` expanded (or left unchanged) to the complete physical boundary.
"""
function symmetrize_layer_density(solver::AbsBIMSolver, layer_density::AbstractVector{N}, pts::BoundaryPoints{T}, billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull = length(pts)
    length(layer_density) == Nfull && return layer_density
    solver.symmetry === nothing && throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    orbits = _fold_boundary(T, pts.xy, solver.symmetry, solver.character)
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
    _bim_normal_derivative(solver::AbsBIMSolver, pts::BoundaryPoints{T}, lvec::AbstractVector) where {T<:Real} → u_raw::Vector

Recovers the (unnormalized, not-yet-symmetrized) physical boundary normal
derivative `∂ₙψ` directly from the smallest *left* singular vector `lvec` of
a `SweepBIMSolver`'s primal Fredholm matrix `A(k)` (see
[`construct_matrices`](@ref)), without ever assembling a second (adjoint)
matrix or running a second Krylov solve.

## Description
Every Nyström-discretized boundary-integral operator in this package is
built from a diagonal quadrature weight `W = diag(ds)`. The physical
normal derivative is the (near-)null right singular vector of the
*weighted-transpose* adjoint operator `A_adj(k) = W⁻¹A(k)ᵀW` (a *bilinear*
transpose, not a conjugate transpose). Writing the primal SVD as
`A = UΣV*`, transposing gives `Aᵀ = V̄ΣŪ*`, i.e. `Aᵀ`'s right singular
vectors are exactly the *conjugates* of `A`'s left singular vectors
(`Aᵀ ū_j = σ_j v̄_j`, obtained by conjugating the standard relation
`A*u_j = σ_j v_j`). Consequently `y = W⁻¹ū_L` satisfies `A_adj(k) y =
W⁻¹Aᵀ(Wy) = W⁻¹Aᵀū_L = 0` exactly whenever `Aᵀū_L = 0` exactly (i.e. at a
true eigenvalue), and is an equally good approximation away from it as the
tension `A`'s own smallest singular value provides — the same accuracy
[`solve_vect`](@ref)'s density already carries. This lets
[`solve_state`](@ref) recover `u_L` (the primal problem's left singular
vector) from the *same* `KrylovKit.svdsolve` call already computing the
tension/density, instead of a separate adjoint-matrix assembly and solve.

Defined generically here (not per concrete solver) because this reciprocity
is a property of the underlying Helmholtz Nyström discretization shared by
every `AbsBIMSolver`, not of any one kernel. A future solver whose quadrature
weighting does not follow the plain `W = diag(ds)` convention should add its
own `_bim_normal_derivative` method instead of relying on this default.

## Returns
*  `u_raw` : The raw (fundamental-domain-length if `solver.symmetry !== nothing`) `∂ₙψ`, not yet symmetry-expanded or Rellich-normalized.
"""
function _bim_normal_derivative(solver::AbsBIMSolver, pts::BoundaryPoints{T}, lvec::AbstractVector) where {T<:Real}
    idx = solver.symmetry === nothing ? (1:length(lvec)) : _fold_boundary(T, pts.xy, solver.symmetry, solver.character).fundamental_indices
    return conj.(lvec) ./ pts.ds[idx]
end

"""
    _bim_grid_scale(solver::AbsBIMSolver) → scale::Real

Default boundary-oversampling scale factor used by `b = :auto` in
[`wavefunction(state::BIMEigenstate)`](@ref), namely
`solver.pts_scaling_factor[1]`. [`CompositeBIMSolver`](@ref) has no
`pts_scaling_factor` field of its own and must add its own method once its
`construct_matrices` is implemented (see the migration plan, Step 7).
"""
_bim_grid_scale(solver::AbsBIMSolver) = solver.pts_scaling_factor[1]

"""
    solve_state(solver::SweepBIMSolver, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool = true) where {T<:Real,Bi<:AbsBilliard} → (ten::Real, vec::Vector, u::Vector, bnd_norm::Real)

Solves the boundary-integral eigenvalue problem at wavenumber `k`, returning
everything a [`BIMEigenstate`](@ref) needs — the tension, the primal
boundary density, and the Rellich-normalized physical boundary normal
derivative `u = ∂ₙψ` on the complete physical boundary — from a *single*
`construct_matrices`/`KrylovKit.svdsolve` call, generic over every
[`SweepBIMSolver`](@ref).

## Description
`A(k)` is assembled once with [`construct_matrices`](@ref); its smallest
singular triplet is computed with one `KrylovKit.svdsolve(A, 1, :SR)` call,
giving the tension `ten`, the primal density `vec` (its right singular
vector, matching [`solve_vect`](@ref)) and, from the *same* solve, the left
singular vector used by [`_bim_normal_derivative`](@ref) to recover `∂ₙψ`
without a second matrix assembly or Krylov solve. The raw density is
expanded onto the complete physical boundary with
[`symmetrize_layer_density`](@ref) and Rellich-normalized with
[`_rellich`](@ref).

Used by [`compute_eigenstate(::SweepBIMSolver, ...)`](@ref) to populate
`BIMEigenstate`'s `vec`/`ten`/`pts`/`u`/`bnd_norm` fields up front, so
[`boundary_function`](@ref)/[`momentum_function`](@ref)/
[`husimi_function`](@ref)/[`wavefunction`](@ref) never need to re-solve
anything for a `BIMEigenstate`.

## Returns
* `ten`: The tension (smallest singular value of `A(k)`).
* `vec`: The primal boundary density (fundamental-domain length if `solver.symmetry !== nothing`).
* `u`: The Rellich-normalized `∂ₙψ` on the complete physical boundary.
* `bnd_norm`: The Rellich-identity value ([`_rellich`](@ref)) of the raw density *before* rescaling; only meaningful as the `u = u_raw/√bnd_norm` scale factor, since the raw Krylov singular vector's own norm convention carries no physical meaning by itself.
"""
function solve_state(solver::SweepBIMSolver, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    kT = T(k)
    A = construct_matrices(solver, pts, kT; multithreaded)
    @blas_1 vals, lvecs, rvecs, _ = KrylovKit.svdsolve(A, 1, :SR)
    ten = vals[1]
    vec = Vector{Complex{T}}(rvecs[1])
    u = symmetrize_layer_density(solver, _bim_normal_derivative(solver, pts, lvecs[1]), pts, billiard)
    bnd_norm = _rellich(pts, u, kT)
    u = u ./ sqrt(bnd_norm)
    return ten, vec, u, bnd_norm
end
