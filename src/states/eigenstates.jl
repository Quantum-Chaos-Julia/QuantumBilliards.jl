"""
    BasisEigenstate{K,T,Ba} <: AbsState

`BasisEigenstate` is a concrete type representing a numerically computed eigenstate
of a quantum billiard at a given wavenumber.

## Description
Eigenstates are produced by [`compute_eigenstate`](@ref), which combines a
sweep or accelerated solver, a basis and a billiard geometry to obtain the
expansion coefficients `vec` in `basis` and an estimate of the tension `ten`,
quantifying how well the boundary condition is satisfied. Coefficients
smaller in magnitude than the numerical precision (given by `set_precision`)
are set to zero when the state is constructed. The enclosing
[`SpectralData`](@ref) (returned by both [`compute_spectrum`](@ref) and
[`compute_eigenstate`](@ref)) supplies the `billiard` context this state was
computed on; `BasisEigenstate` itself does not store it.

## Attributes
* `k::K`: The wavenumber of the eigenstate, as refined by the solver.
* `k_basis::K`: The wavenumber at which `basis` was evaluated to obtain `vec` (will differ slightly from `k` for accelerated solvers).
* `vec::Vector{K}`: Expansion coefficients of the eigenstate in `basis`.
* `ten::T`: Tension of the solution, measuring the residual boundary condition violation.
* `basis::Ba`: The basis (`Ba<:AbsBasis`), resized/evaluated at `k_basis`, in which `vec` is expressed.

## API
The following functions can be evaluated for this type:
- [`compute_eigenstate`](@ref)
- [`boundary_function`](@ref)
- [`momentum_function`](@ref)
- [`wavefunction`](@ref)
- [`husimi_function`](@ref)
"""
struct BasisEigenstate{K,T,Ba} <: AbsState
    k::K
    k_basis::K
    vec::Vector{K}
    ten::T
    basis::Ba
end

"""
    BasisEigenstate(k::K, vec::Vector{K}, ten::T, basis::Ba) where {K<:Number,T<:Real,Ba<:AbsBasis}

Construct a [`BasisEigenstate`](@ref) with `k_basis` set equal to `k`, filtering
out negligible coefficients of `vec`.

## Description
The numerical precision is obtained from `set_precision` applied to `vec[1]`.
If `vec` has real entries, any entry with absolute value not exceeding this
precision is replaced by zero; a complex-valued `vec` is left unfiltered.

## Arguments
* `k::K`: The wavenumber of the eigenstate.
* `vec::Vector{K}`: Expansion coefficients of the eigenstate in `basis`.
* `ten::T`: Tension of the solution.
* `basis::Ba`: The basis in which `vec` is expressed.

## Returns
* `state::BasisEigenstate`: A new [`BasisEigenstate`](@ref) with `k_basis = k` and filtered coefficients.
"""
function BasisEigenstate(k::K, vec::Vector{K}, ten::T, basis::Ba) where {K<:Number,T<:Real,Ba<:AbsBasis}
    eps = T(set_precision(vec[1]))
    if K <: Real
        filtered_vec = K[abs(v) > eps ? v : zero(K) for v in vec]
    else
        filtered_vec = vec
    end
    return BasisEigenstate{K,T,Ba}(k, k, filtered_vec, ten, basis)
end

"""
    BasisEigenstate(k::K, k_basis::K, vec::Vector{K}, ten::T, basis::Ba) where {K<:Number,T<:Real,Ba<:AbsBasis}

Construct a [`BasisEigenstate`](@ref) allowing the refined wavenumber `k` and the
basis-evaluation wavenumber `k_basis` to differ, filtering out negligible
coefficients of `vec`.

## Description
Behaves as [`BasisEigenstate(k, vec, ten, basis)`](@ref BasisEigenstate),
except that `k_basis` is taken as given instead of being set equal to `k`.
This is used by accelerated solvers, where `basis` is evaluated at a fixed
scaling wavenumber `k_basis` while the eigenstate itself is refined to a
nearby wavenumber `k`.

## Arguments
* `k::K`: The refined wavenumber of the eigenstate.
* `k_basis::K`: The wavenumber at which `basis` was evaluated to obtain `vec`.
* `vec::Vector{K}`: Expansion coefficients of the eigenstate in `basis`.
* `ten::T`: Tension of the solution.
* `basis::Ba`: The basis in which `vec` is expressed.

## Returns
* `state::BasisEigenstate`: A new [`BasisEigenstate`](@ref) with the specified `k_basis` and filtered coefficients.
"""
function BasisEigenstate(k::K, k_basis::K, vec::Vector{K}, ten::T, basis::Ba) where {K<:Number,T<:Real,Ba<:AbsBasis}
    eps = T(set_precision(vec[1]))
    if K <: Real
        filtered_vec = K[abs(v) > eps ? v : zero(K) for v in vec]
    else
        filtered_vec = vec
    end
    return BasisEigenstate{K,T,Ba}(k, k_basis, filtered_vec, ten, basis)
end

"""
    compute_eigenstate(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k; multithreaded::Bool=true) → data::SpectralData

Computes the [`BasisEigenstate`](@ref) of `billiard` at wavenumber `k` using a
sweep-method `solver` (e.g. `DecompositionMethodSolver`), wrapped in a length-1
[`SpectralData`](@ref).

## Description
The basis dimension is set to
`dim = max(solver.min_dim, round(Int, L*k*solver.dim_scaling_factor/(2*pi)))`,
with `L` the total boundary length, and `basis` is resized to this dimension
with `resize_basis`. Boundary points are sampled with `evaluate_points`, and
the generalized eigenvalue problem is solved at `k` with `solve_vect` to
obtain the tension `ten` and coefficient vector `vec`.

## Arguments
* `solver::SweepBasisSolver`: The sweep basis solver used to solve the eigenvalue problem.
* `basis::AbsBasis`: The basis used to approximate the eigenstate.
* `billiard::AbsBilliard`: The billiard the eigenstate is computed on.
* `k`: The wavenumber at which the eigenstate is computed.

## Keyword Arguments
* `multithreaded::Bool=true`: Whether the matrix construction is multithreaded.

## Returns
* `data::SpectralData`: Length-1 [`SpectralData`](@ref) wrapping the computed [`BasisEigenstate`](@ref) at wavenumber `k`.
"""
function compute_eigenstate(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k; multithreaded::Bool=true)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim, round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis, billiard, dim, k)
    pts = evaluate_points(solver, billiard, k)
    ten, vec = solve_vect(solver, basis_new, pts, k; multithreaded)
    state = BasisEigenstate(k, vec, ten, basis_new)
    return SpectralData(billiard, solver, [state.k], [ten], [false]; states=[state])
end

"""
    compute_eigenstate(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k; dk::Real=0.1, multithreaded::Bool=true) → data::SpectralData

Computes the [`BasisEigenstate`](@ref) of `billiard` closest to wavenumber `k`
using an accelerated `solver` (e.g. `VerginiSaracenoSolver`), wrapped in a
length-1 [`SpectralData`](@ref).

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
* `solver::AcceleratedBasisSolver`: The accelerated basis solver used to solve the eigenvalue problem.
* `basis::AbsBasis`: The basis used to approximate the eigenstate.
* `billiard::AbsBilliard`: The billiard the eigenstate is computed on.
* `k`: The target wavenumber around which the eigenstate is searched for.

## Keyword Arguments
* `dk::Real=0.1`: Half-width of the wavenumber window around `k` within which candidate eigenstates are searched.
* `multithreaded::Bool=true`: Whether the matrix construction is multithreaded.

## Returns
* `data::SpectralData`: Length-1 [`SpectralData`](@ref) wrapping the computed [`BasisEigenstate`](@ref) closest to wavenumber `k`.
"""
function compute_eigenstate(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k; dk::Real=0.1, multithreaded::Bool=true)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim, round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new = resize_basis(basis, billiard, dim, k)
    pts = evaluate_points(solver, billiard, k)
    ks, tens, X = solve_vectors(solver, basis_new, pts, k, dk; multithreaded)
    idx = findmin(abs.(ks .- k))[2]
    k_state = ks[idx]; ten = tens[idx]; vec = X[:,idx]
    state = BasisEigenstate(k_state, k, vec, ten, basis_new)
    return SpectralData(billiard, solver, [k_state], [ten], [false]; states=[state])
end

"""
    BIMEigenstate{K,T} <: AbsState

Represents a numerically computed eigenstate of a quantum billiard obtained
from a boundary-integral-method solver.

## Description
A `BIMEigenstate` stores the wavenumber and residual together with the boundary
data available from the computation. The layer density `vec` and physical
boundary normal derivative `u = ∂ₙψ` are optional and stored independently.
The enclosing [`SpectralData`](@ref) (returned by both [`compute_spectrum`](@ref)
and [`compute_eigenstate`](@ref)) supplies the `billiard`/`solver` context this
state was computed on; `BIMEigenstate` itself does not store either.

A complete state obtained through [`solve_state`](@ref) contains both the layer
density and the Rellich-normalized boundary normal derivative. Accelerated BIM
solvers may instead construct partial states directly from data already
produced by the spectral solve. For example, [`BeynSolver`](@ref) and
[`CORKSolver`](@ref) can retain their computed layer densities while leaving
`u` and `bnd_norm` equal to `nothing`. This avoids an additional boundary
nullspace solve solely for constructing the state.

When present, `vec` and `u` are associated with the boundary discretization
stored in `pts`.

## Attributes
* `k::T`: Real part of the wavenumber of the eigenstate.
* `k_im::T`: Imaginary part / numerical residual of the wavenumber. This is a
  numerical leakage diagnostic from boundary-integral root-finding, not a
  physical imaginary wavenumber — this codebase has no leaky/open-billiard
  resonance support.
* `vec::Union{Nothing,Vector{K}}`: Layer density on `pts`, or `nothing` when unavailable.
* `ten::T`: Tension or residual associated with the state.
* `pts::BoundaryPoints{T}`: Boundary discretization associated with the stored boundary data.
* `u::Union{Nothing,Vector{K}}`: Rellich-normalized physical boundary normal derivative `∂ₙψ` on `pts`, or `nothing` when unavailable.
* `bnd_norm::Union{Nothing,T}`: Rellich normalization of the unnormalized boundary normal derivative, or `nothing` when unavailable.

## API
The following functions can be evaluated when the boundary representation they
require is available:
- [`boundary_function`](@ref)
- [`momentum_function`](@ref)
- [`wavefunction`](@ref)
- [`husimi_function`](@ref)
"""
struct BIMEigenstate{K,T} <: AbsState
    k::T
    k_im::T
    vec::Union{Nothing,Vector{K}}
    ten::T
    pts::BoundaryPoints{T}
    u::Union{Nothing,Vector{K}}
    bnd_norm::Union{Nothing,T}
end

"""
    BIMEigenstate(k::Number, vec::Union{Nothing,Vector{K}}, ten::T, pts::BoundaryPoints{T}; u::Union{Nothing,Vector{K}}=nothing, bnd_norm::Union{Nothing,T}=nothing) where {K<:Number,T<:Real}

Construct a [`BIMEigenstate`](@ref) from boundary data already available from
a BIM computation, splitting `k` into its real part and its imaginary residual
`k_im`.

## Arguments
* `k::Number`: Wavenumber of the eigenstate (real or complex).
* `vec::Union{Nothing,Vector{K}}`: Layer density on `pts`, or `nothing`.
* `ten::T`: Tension or residual associated with the state.
* `pts::BoundaryPoints{T}`: Boundary discretization associated with the stored boundary data.

## Keyword Arguments
* `u::Union{Nothing,Vector{K}}=nothing`: Rellich-normalized physical boundary normal derivative `∂ₙψ`, or `nothing`.
* `bnd_norm::Union{Nothing,T}=nothing`: Rellich normalization associated with `u`, or `nothing`.

## Returns
* `state::BIMEigenstate`: Constructed BIM eigenstate.
"""
function BIMEigenstate(k::Number, vec::Union{Nothing,Vector{K}}, ten::T, pts::BoundaryPoints{T}; u::Union{Nothing,Vector{K}}=nothing, bnd_norm::Union{Nothing,T}=nothing) where {K<:Number,T<:Real}
    return BIMEigenstate{K,T}(T(real(k)), T(imag(k)), vec, ten, pts, u, bnd_norm)
end

"""
    compute_eigenstate(solver::SweepBIMSolver, billiard::AbsBilliard, k; multithreaded::Bool=true) → data::SpectralData

Compute a complete [`BIMEigenstate`](@ref) at wavenumber `k` using a
boundary-integral sweep solver, wrapped in a length-1 [`SpectralData`](@ref).

## Description
Boundary points are constructed with [`evaluate_points`](@ref), after which
[`solve_state`](@ref) computes the tension, layer density and physical boundary
normal derivative from the boundary-integral problem. The boundary normal
derivative is Rellich normalized and stored together with its original
normalization factor.

Unlike partial states produced directly by accelerated spectral solvers, the
returned state therefore contains both `vec` and `u`.

## Arguments
* `solver::SweepBIMSolver`: Boundary-integral solver.
* `billiard::AbsBilliard`: Billiard on which the eigenstate is computed.
* `k`: Wavenumber at which the state is computed.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded boundary-matrix construction.

## Returns
* `data::SpectralData`: Length-1 [`SpectralData`](@ref) wrapping the complete BIM eigenstate containing both the layer density and physical boundary normal derivative.
"""
function compute_eigenstate(solver::SweepBIMSolver, billiard::AbsBilliard, k; multithreaded::Bool=true)
    pts = evaluate_points(solver, billiard, k)
    ten, vec, u, bnd_norm = solve_state(solver, pts, k, billiard; multithreaded)
    state = BIMEigenstate(k, vec, ten, pts; u, bnd_norm)
    return SpectralData(billiard, solver, [state.k], [ten], [false]; states=[state])
end

"""
    compute_eigenstate(solver::AcceleratedBIMSolver, billiard::AbsBilliard, k; dk::Real=0.1, multithreaded::Bool=true) → data::SpectralData

Compute a complete [`BIMEigenstate`](@ref) associated with the accelerated BIM
eigenvalue nearest `k`, wrapped in a length-1 [`SpectralData`](@ref).

## Description
The accelerated solver first locates the eigenvalue nearest `k` through
[`solve_wavenumber`](@ref). The wrapped [`SweepBIMSolver`](@ref) is then used
to compute the complete state at the refined wavenumber, including both the
layer density and Rellich-normalized physical boundary normal derivative.

This single-state convenience pathway is distinct from accelerated
[`compute_spectrum`](@ref) calculations. Spectrum calculations retain whatever
eigenvector information is already produced by the accelerated solve and do
not perform an additional boundary nullspace solve merely to populate missing
state fields.

## Arguments
* `solver::AcceleratedBIMSolver`: Accelerated BIM solver used to locate the eigenvalue.
* `billiard::AbsBilliard`: Billiard on which the eigenstate is computed.
* `k`: Target wavenumber.
    
## Keyword Arguments
* `dk::Real=0.1`: Spectral search width passed to [`solve_wavenumber`](@ref).
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
* `data::SpectralData`: Length-1 [`SpectralData`](@ref) wrapping the complete BIM eigenstate at the located wavenumber.
"""
function compute_eigenstate(solver::AcceleratedBIMSolver, billiard::AbsBilliard, k; dk::Real=0.1, multithreaded::Bool=true)
    k0, _ = solve_wavenumber(solver, billiard, k, dk; multithreaded)
    return compute_eigenstate(solver.kernel, billiard, k0; multithreaded)
end