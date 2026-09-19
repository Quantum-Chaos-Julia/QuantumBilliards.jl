"""
    BasisEigenstate{K,T,S,Bi,Ba} <: AbsState

`BasisEigenstate` is a concrete type representing a numerically computed eigenstate
of a quantum billiard at a given wavenumber.

## Description
Eigenstates are produced by [`compute_eigenstate`](@ref), which combines a
sweep or accelerated solver, a basis and a billiard geometry to obtain the
expansion coefficients `vec` in `basis` and an estimate of the tension `ten`,
quantifying how well the boundary condition is satisfied. Coefficients
smaller in magnitude than the numerical precision `eps` (given by
`set_precision`) are set to zero when the state is constructed.

## Attributes
* `k`: The wavenumber of the eigenstate, as refined by the solver.
* `k_basis`: The wavenumber at which `basis` was evaluated to obtain `vec` (may differ slightly from `k` for accelerated solvers).
* `vec`: Expansion coefficients of the eigenstate in `basis`.
* `ten`: Tension of the solution, measuring the residual boundary condition violation.
* `dim`: Dimension of `vec` (and of `basis`).
* `eps`: Numerical precision threshold below which coefficients of `vec` are treated as zero.
* `solver`: The solver (`S<:AbsBasisSolver`) used to compute the eigenstate.
* `basis`: The basis (`Ba<:AbsBasis`), resized/evaluated at `k_basis`, in which `vec` is expressed.
* `billiard`: The billiard (`Bi<:AbsBilliard`) the eigenstate is defined on.

## API
The following functions can be evaluated for this type:
- [`compute_eigenstate`](@ref)
- [`boundary_function`](@ref)
- [`momentum_function`](@ref)
- [`wavefunction`](@ref)
- [`husimi_function`](@ref)
"""
struct BasisEigenstate{K,T,S,Bi,Ba} <: AbsState
    k::K
    k_basis::K
    vec::Vector{K}
    ten::T
    dim::Int64
    eps::T
    solver::S
    basis::Ba
    billiard::Bi
end

"""
    BasisEigenstate(k, vec, ten, solver, basis, billiard) → state::BasisEigenstate

Construct an [`BasisEigenstate`](@ref) with `k_basis` set equal to `k`, filtering
out negligible coefficients of `vec`.

## Description
The numerical precision `eps` is obtained from `set_precision` applied to
`vec[1]`. If `vec` has real entries, any entry with absolute value not
exceeding `eps` is replaced by zero; a complex-valued `vec` is left
unfiltered.

## Arguments
* `k`: The wavenumber of the eigenstate.
* `vec`: Expansion coefficients of the eigenstate in `basis`.
* `ten`: Tension of the solution.
* `solver`: The solver used to compute the eigenstate.
* `basis`: The basis in which `vec` is expressed.
* `billiard`: The billiard the eigenstate is defined on.

## Returns
*  `state` : A new [`BasisEigenstate`](@ref) with `k_basis = k` and filtered coefficients.
"""
function BasisEigenstate(k, vec, ten, solver, basis, billiard)  
    eps = set_precision(vec[1])
    if eltype(vec) <: Real
        filtered_vec = eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else 
        filtered_vec = vec
    end
    return BasisEigenstate(k, k, filtered_vec,ten, length(vec), eps, solver, basis, billiard)
end

"""
    BasisEigenstate(k, k_basis, vec, ten, solver, basis, billiard) → state::BasisEigenstate

Construct an [`BasisEigenstate`](@ref) allowing the refined wavenumber `k` and the
basis-evaluation wavenumber `k_basis` to differ, filtering out negligible
coefficients of `vec`.

## Description
Behaves as [`BasisEigenstate(k, vec, ten, solver, basis, billiard)`](@ref BasisEigenstate),
except that `k_basis` is taken as given instead of being set equal to `k`.
This is used by accelerated solvers, where `basis` is evaluated at a fixed
scaling wavenumber `k_basis` while the eigenstate itself is refined to a
nearby wavenumber `k`.

## Arguments
* `k`: The (refined) wavenumber of the eigenstate.
* `k_basis`: The wavenumber at which `basis` was evaluated to obtain `vec`.
* `vec`: Expansion coefficients of the eigenstate in `basis`.
* `ten`: Tension of the solution.
* `solver`: The solver used to compute the eigenstate.
* `basis`: The basis in which `vec` is expressed.
* `billiard`: The billiard the eigenstate is defined on.

## Returns
*  `state` : A new [`BasisEigenstate`](@ref) with filtered coefficients.
"""
function BasisEigenstate(k, k_basis, vec, ten, solver, basis, billiard)  
    eps = set_precision(vec[1])
    if eltype(vec) <: Real
        filtered_vec = eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else 
        filtered_vec = vec
    end
    return BasisEigenstate(k, k_basis, filtered_vec, ten, length(vec), eps, solver, basis, billiard)
end

"""
    compute_eigenstate(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k; multithreaded::Bool = true) → state::BasisEigenstate

Computes the [`BasisEigenstate`](@ref) of `billiard` at wavenumber `k` using a
sweep-method `solver` (e.g. `DecompositionMethodSolver`).

## Description
The basis dimension is set to
`dim = max(solver.min_dim, round(Int, L*k*solver.dim_scaling_factor/(2*pi)))`,
with `L` the total boundary length, and `basis` is resized to this dimension
with `resize_basis`. Boundary points are sampled with `evaluate_points`, and
the generalized eigenvalue problem is solved at `k` with `solve_vect` to
obtain the tension `ten` and coefficient vector `vec`.

## Arguments
* `solver`: The `SweepBasisSolver` used to solve the eigenvalue problem.
* `basis`: The basis used to approximate the eigenstate.
* `billiard`: The billiard the eigenstate is computed on.
* `k`: The wavenumber at which the eigenstate is computed.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded.

## Returns
*  `state` : The computed [`BasisEigenstate`](@ref) at wavenumber `k`.
"""
function compute_eigenstate(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard,k; multithreaded = true)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis,billiard, dim, k)
    pts = evaluate_points(solver, billiard, k)
    ten, vec = solve_vect(solver, basis_new, pts, k; multithreaded)
    return BasisEigenstate(k, vec, ten, solver, basis_new, billiard)
end

"""
    compute_eigenstate(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k; dk::Real = 0.1, multithreaded::Bool = true) → state::BasisEigenstate

Computes the [`BasisEigenstate`](@ref) of `billiard` closest to wavenumber `k`
using an accelerated `solver` (e.g. `VerginiSaracenoSolver`).

## Description
The basis dimension is set to
`dim = max(solver.min_dim, round(Int, L*k*solver.dim_scaling_factor/(2*pi)))`,
with `L` the total boundary length, and `basis` is resized to this dimension
with `resize_basis`. Boundary points are sampled with `evaluate_points`, and
`solve_vectors` is used to find all candidate wavenumbers `ks`, tensions
`tens` and eigenvectors `X` within `dk` of `k`. The candidate `k_state`
closest to `k` is selected and used to build the resulting [`BasisEigenstate`](@ref),
whose `k_basis` is set to the requested `k` (the wavenumber at which `basis`
was evaluated).

## Arguments
* `solver`: The `AcceleratedBasisSolver` used to solve the eigenvalue problem.
* `basis`: The basis used to approximate the eigenstate.
* `billiard`: The billiard the eigenstate is computed on.
* `k`: The target wavenumber around which the eigenstate is searched for.

## Keyword arguments
*  `dk::Real = 0.1` : Half-width of the wavenumber window around `k` within which candidate eigenstates are searched.
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded.

## Returns
*  `state` : The computed [`BasisEigenstate`](@ref) closest to wavenumber `k`.
"""
function compute_eigenstate(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k; dk = 0.1, multithreaded = true)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    ks, tens, X = solve_vectors(solver,basis_new, pts, k, dk; multithreaded)
    idx = findmin(abs.(ks.-k))[2]
    k_state = ks[idx]
    ten = tens[idx]
    vec = X[:,idx]
    return BasisEigenstate(k_state, k, vec, ten, solver, basis_new, billiard)
end

"""
    BIMEigenstate{K,T,S,Bi} <: AbsState

`BIMEigenstate` is a concrete type representing a numerically computed eigenstate
of a quantum billiard, obtained from a boundary-integral-method (BIM) solver.

## Description
Eigenstates are produced by [`compute_eigenstate`](@ref), which combines a
[`SweepBIMSolver`](@ref) and a billiard geometry to obtain the boundary
density `vec` at the located wavenumber and an estimate of the tension `ten`.
Unlike [`BasisEigenstate`](@ref), there is no basis expansion: `vec` is the
boundary density itself, sampled at the boundary discretization points (the
fundamental-domain points only, if `solver.symmetry !== nothing`).

!!! note "Primal density vs. `∂ₙψ`"
    `vec` is the *primal* boundary density obtained by [`solve_vect`](@ref)
    and is **not** the physical boundary normal derivative `∂ₙψ`. Unlike an
    earlier version of this type, `BIMEigenstate` now also stores the
    Rellich-normalized `∂ₙψ` directly (`u`, on the complete physical
    boundary `pts`), computed once by [`solve_state`](@ref) alongside
    `vec`/`ten` from the *same* Krylov singular-value solve — see
    [`_bim_normal_derivative`](@ref) for why this is possible without a
    second matrix assembly or solve. This makes [`boundary_function`](@ref),
    [`momentum_function`](@ref), [`wavefunction`](@ref) and
    [`husimi_function`](@ref) simple field reads for *any*
    [`SweepBIMSolver`](@ref), not just [`DoubleLayerPotentialSolver`](@ref).

## Attributes
* `k`: The wavenumber of the eigenstate, as refined by the solver. Stored with the same (generally complex) element type `K` as `vec`, since the boundary density is complex-valued; the imaginary part is always zero.
* `k_basis`: Set equal to `k` (no separate basis-evaluation wavenumber for BIM solvers).
* `vec`: Boundary density values at the boundary discretization points.
* `ten`: Tension of the solution, measuring the residual boundary condition violation.
* `dim`: Dimension of `vec`.
* `eps`: Numerical precision threshold below which coefficients of `vec` are treated as zero.
* `solver`: The solver (`S<:SweepBIMSolver`) used to compute the eigenstate.
* `billiard`: The billiard (`Bi<:AbsBilliard`) the eigenstate is defined on.
* `pts`: The complete physical boundary discretization (same `pts` the solve used) that `u` is sampled at.
* `u`: The Rellich-normalized physical boundary normal derivative `∂ₙψ` on `pts`, from [`solve_state`](@ref).
* `bnd_norm`: The Rellich-identity value ([`_rellich`](@ref)) of the *raw* density before rescaling to `u`; only meaningful as the scale factor `u = u_raw/√bnd_norm` applies, since the raw density's own norm convention is an arbitrary artifact of the underlying Krylov singular-vector solve, not a physical quantity.

## API
The following functions can be evaluated for this type:
- [`compute_eigenstate`](@ref)
- [`boundary_function`](@ref)
- [`momentum_function`](@ref)
- [`wavefunction`](@ref)
- [`husimi_function`](@ref)
"""
struct BIMEigenstate{K,T,S,Bi} <: AbsState
    k::K
    k_basis::K
    vec::Vector{K}
    ten::T
    dim::Int64
    eps::T
    solver::S
    billiard::Bi
    pts::BoundaryPoints{T}
    u::Vector{K}
    bnd_norm::T
end

"""
    BIMEigenstate(k, vec, ten, solver, billiard, pts, u, bnd_norm) → state::BIMEigenstate

Construct a [`BIMEigenstate`](@ref) with `k_basis` set equal to `k`, filtering
out negligible coefficients of `vec` (see [`BasisEigenstate`](@ref)).

## Arguments
* `k`: The wavenumber of the eigenstate.
* `vec`: Boundary density values at the boundary discretization points.
* `ten`: Tension of the solution.
* `solver`: The [`SweepBIMSolver`](@ref) used to compute the eigenstate.
* `billiard`: The billiard the eigenstate is defined on.
* `pts`: The complete physical boundary discretization `u` is sampled at.
* `u`: The Rellich-normalized physical boundary normal derivative `∂ₙψ`.
* `bnd_norm`: The Rellich-identity normalization applied to `u`.

## Returns
*  `state` : A new [`BIMEigenstate`](@ref) with `k_basis = k` and filtered coefficients.
"""
function BIMEigenstate(k, vec, ten, solver, billiard, pts, u, bnd_norm)
    K = eltype(vec)
    kK = K(k)
    eps = set_precision(real(vec[1]))
    if eltype(vec) <: Real
        filtered_vec = eltype(vec).([abs(v)>eps ? v : zero(vec[1]) for v in vec])
    else
        filtered_vec = vec
    end
    return BIMEigenstate(kK, kK, filtered_vec, ten, length(vec), eps, solver, billiard, pts, u, bnd_norm)
end

"""
    compute_eigenstate(solver::SweepBIMSolver, billiard::AbsBilliard, k; multithreaded::Bool = true) → state::BIMEigenstate

Computes the [`BIMEigenstate`](@ref) of `billiard` at wavenumber `k` using a
boundary-integral sweep `solver` (e.g. [`DoubleLayerPotentialSolver`](@ref)).

## Description
Boundary points are sampled with `evaluate_points`, and the boundary-integral
Fredholm problem is solved at `k` with [`solve_state`](@ref), which returns
the tension `ten`, boundary density `vec` *and* the Rellich-normalized
physical boundary normal derivative `u` — all from a single Krylov solve —
so the resulting `BIMEigenstate` already carries everything
[`boundary_function`](@ref)/[`wavefunction`](@ref)/[`husimi_function`](@ref)
need, without re-solving anything later.

## Arguments
* `solver`: The [`SweepBIMSolver`](@ref) used to solve the boundary-integral eigenvalue problem.
* `billiard`: The billiard the eigenstate is computed on.
* `k`: The wavenumber at which the eigenstate is computed.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded.

## Returns
*  `state` : The computed [`BIMEigenstate`](@ref) at wavenumber `k`.
"""
function compute_eigenstate(solver::SweepBIMSolver, billiard::AbsBilliard, k; multithreaded=true)
    pts = evaluate_points(solver, billiard, k)
    ten, vec, u, bnd_norm = solve_state(solver, pts, k, billiard; multithreaded)
    return BIMEigenstate(k, vec, ten, solver, billiard, pts, u, bnd_norm)
end

"""
    compute_eigenstate(solver::AcceleratedBIMSolver, billiard::AbsBilliard, k; dk::Real = 0.1, multithreaded::Bool = true) → state::BIMEigenstate

Computes the [`BIMEigenstate`](@ref) of `billiard` closest to wavenumber `k`
using an accelerated boundary-integral `solver` (e.g. [`BeynSolver`](@ref),
[`ExpandedBIMSolver`](@ref)).

## Description
`solve_wavenumber(solver, billiard, k, dk; multithreaded)` locates the
refined root `k0` nearest `k` using the accelerated root-finding strategy
(contour integral / local Taylor expansion). The eigenstate itself is then
obtained by re-solving the wrapped `solver.kernel::SweepBIMSolver` at `k0`
via [`compute_eigenstate(::SweepBIMSolver, ...)`](@ref), exactly as at a true
eigenvalue the kernel's own Fredholm nullspace solve recovers the same
boundary density/normal-derivative `∂ₙψ` regardless of which root-finding
strategy located `k0`. This mirrors
[`compute_eigenstate(::AcceleratedBasisSolver, ...)`](@ref): once a
wavenumber candidate is found, the resulting `BIMEigenstate` is a plain
[`SweepBIMSolver`](@ref) state, so [`boundary_function`](@ref),
[`momentum_function`](@ref), [`wavefunction`](@ref) and
[`husimi_function`](@ref) all work on it unchanged.

## Arguments
* `solver`: The [`AcceleratedBIMSolver`](@ref) used to locate the wavenumber.
* `billiard`: The billiard the eigenstate is computed on.
* `k`: The target wavenumber around which the eigenstate is searched for.

## Keyword arguments
*  `dk::Real = 0.1` : Half-width of the wavenumber window around `k` within which the root is searched, passed to `solve_wavenumber`.
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded.

## Returns
*  `state` : The computed [`BIMEigenstate`](@ref) closest to wavenumber `k`.
"""
function compute_eigenstate(solver::AcceleratedBIMSolver, billiard::AbsBilliard, k; dk=0.1, multithreaded=true)
    k0, t0 = solve_wavenumber(solver, billiard, k, dk; multithreaded)
    return compute_eigenstate(solver.kernel, billiard, k0; multithreaded)
end