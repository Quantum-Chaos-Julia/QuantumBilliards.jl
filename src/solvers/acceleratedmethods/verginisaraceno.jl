################################################################################
# VERGINI-SARACENO SCALING METHOD
#
# This file implements the Vergini-Saraceno scaling method for computing many
# quantum-billiard eigenvalues near a single scaling wavenumber k₀ from one
# generalized eigenvalue problem.
#
# For a basis {ϕₙ(k)} satisfying the Helmholtz equation in the billiard, define
# the boundary matrices
#
#                         F = Bᵀ W B,
#
#                  Fₖ = Bᵀ W Bₖ + Bₖᵀ W B,
#
# where B contains the basis functions evaluated on the boundary, Bₖ their
# wavenumber derivatives, and W contains the Vergini-Saraceno boundary weights
#
#                         wᵢ = dsᵢ/(rᵢ·nᵢ).
#
# The local spectrum around k₀ is obtained from the generalized eigenproblem
#
#                         F x = μ Fₖ x.
#
# Each generalized eigenvalue μ gives the scaling estimate
#
#                  kVS = k₀ - 2/μ + 2/(k₀ μ²),
#
# with the associated tension estimate
#
#                         t = 2(2/μ)².
#
# A single generalized diagonalization therefore produces multiple eigenvalue
# estimates around k₀. Candidates are subsequently restricted to the requested
# window |kVS-k₀|<Δk.
################################################################################

"""
    VerginiSaracenoSolver{T} <: AcceleratedBasisSolver

`VerginiSaracenoSolver` is a concrete [`AcceleratedBasisSolver`](@ref) implementing the
Vergini–Saraceno scaling method for computing quantum billiard spectra.

## Description
The method constructs the matrices `F` and `Fk` (see [`construct_matrices`](@ref))
from a boundary quadrature with weights `w_vs` (see [`evaluate_points`](@ref)) and
solves the generalized eigenvalue problem `F * x = λ * Fk * x` to extract, in a
single diagonalization, every eigenvalue lying within a window around the target
wavenumber `k` (see [`solve`](@ref), [`sm_results`](@ref)).

## Attributes
* `dim_scaling_factor`: Scaling factor used to determine the basis dimension from the boundary length and wavenumber.
* `pts_scaling_factor`: Vector of scaling factors, one per fundamental boundary curve, used to determine the number of boundary sampling points.
* `sampler`: Vector of samplers, one per fundamental boundary curve, used to generate boundary points.
* `eps`: Relative tolerance used to filter small eigenvalues in the generalized eigenvalue decomposition.
* `min_dim`: Minimum basis dimension.
* `min_pts`: Minimum number of boundary sampling points.
* `eigenvectors::Bool`: Whether `compute_spectrum` retains the basis eigenvectors returned by the scaling solve.

## API
The following functions can be evaluated for this type:
- [`evaluate_points`](@ref)
- [`construct_matrices`](@ref)
- [`solve`](@ref)
- [`solve_vectors`](@ref)
- [`solve_wavenumber`](@ref)
- [`solve_spectrum`](@ref)
"""
mutable struct VerginiSaracenoSolver{T} <: AcceleratedBasisSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
    eigenvectors::Bool
end

"""
    VerginiSaracenoSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}; min_dim::Int=100, min_pts::Int=500, eigenvectors::Bool=true) where {T<:Real}

Construct a [`VerginiSaracenoSolver`](@ref) with a single
[`GaussLegendreNodes`](@ref) sampler shared by every fundamental boundary
curve.

## Arguments
* `dim_scaling_factor::T`: Scaling factor used to determine the basis dimension from the boundary length and wavenumber.
* `pts_scaling_factor::Union{T,Vector{T}}`: Boundary-point scaling factor, or one factor per fundamental boundary curve.

## Keyword Arguments
* `min_dim::Int=100`: Minimum basis dimension.
* `min_pts::Int=500`: Minimum number of boundary sampling points.
* `eigenvectors::Bool`: Whether `compute_spectrum` retains the basis eigenvectors returned by the scaling solve.

## Returns
* `VerginiSaracenoSolver{T}`: Configured solver.
"""
function VerginiSaracenoSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}; min_dim::Int=100, min_pts::Int=500, eigenvectors::Bool=true) where {T<:Real}
    d = dim_scaling_factor; bs = pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor; sampler = [GaussLegendreNodes()]
    return VerginiSaracenoSolver(d, bs, sampler, eps(T), min_dim, min_pts, eigenvectors)
end

"""
    VerginiSaracenoSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector{AbsSampler}; min_dim::Int=100, min_pts::Int=500, eigenvectors::Bool=true) where {T<:Real}

Construct a [`VerginiSaracenoSolver`](@ref) with a user-supplied sampler for
each fundamental boundary curve.

## Arguments
* `dim_scaling_factor::T`: Scaling factor used to determine the basis dimension from the boundary length and wavenumber.
* `pts_scaling_factor::Union{T,Vector{T}}`: Boundary-point scaling factor, or one factor per fundamental boundary curve.
* `samplers::Vector{AbsSampler}`: Boundary samplers.

## Keyword Arguments
* `min_dim::Int=100`: Minimum basis dimension.
* `min_pts::Int=500`: Minimum number of boundary sampling points.
* `eigenvectors::Bool`: Whether `compute_spectrum` retains the basis eigenvectors returned by the scaling solve.

## Returns
* `VerginiSaracenoSolver{T}`: Configured solver.
"""
function VerginiSaracenoSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector{AbsSampler}; min_dim::Int=100, min_pts::Int=500, eigenvectors::Bool=true) where {T<:Real}
    d = dim_scaling_factor; bs = pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    return VerginiSaracenoSolver(d, bs, samplers, eps(T), min_dim, min_pts, eigenvectors)
end

"""
    evaluate_points(solver::VerginiSaracenoSolver, billiard::Bi, k) where {Bi<:AbsBilliard} → pts::BoundaryPoints

Samples the fundamental boundary of `billiard` and computes the quadrature
weights required by the Vergini–Saraceno scaling method.

## Description
Each fundamental boundary curve is sampled in its native parameter `t`. For
quadrature nodes `tᵢ` with parameter-space weights `dtᵢ`, the physical
arclength weights are

`dsᵢ = |r'(tᵢ)| dtᵢ`.

The outward unit normal is obtained by normalizing the gradient of the domain
function pointwise. The Vergini–Saraceno weights are then

`w_vsᵢ = dsᵢ / (r(tᵢ) ⋅ nᵢ)`.

This construction includes the parametrization Jacobian explicitly and is
therefore valid for arbitrary regular parametrizations; it does not assume
that the curve parameter is proportional to arclength.

## Arguments
* `solver::VerginiSaracenoSolver`: Solver defining the boundary sampling density.
* `billiard::Bi`: Billiard whose fundamental boundary is sampled.
* `k`: Wavenumber used to determine the number of boundary sampling points.

## Returns
* `pts::BoundaryPoints`: Boundary points with `xy` and `w_vs` populated.
"""
function evaluate_points(solver::VerginiSaracenoSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    bs, samplers = adjust_scaling_and_samplers(solver, billiard)
    curves = get_boundary_curves(billiard)
    T = eltype(solver.pts_scaling_factor)
    Ns = _determine_bp_sizes(curves, bs, k)
    M = length(Ns)
    xy_all = Vector{Vector{SVector{2,T}}}(undef, M)
    w_all = Vector{Vector{T}}(undef, M)
    for i in eachindex(curves)
        crv = curves[i]; sampler = samplers[i]
        t, dt = sample_points(sampler, Ns[i])
        xy = curve(crv, t)
        ds = norm.(tangent(crv, t)) .* dt
        normal = domain_gradient_vector(crv, xy)
        normal ./= norm.(normal)
        rn = dot.(xy, normal)
        w = ds ./ rn
        xy_all[i] = xy; w_all[i] = w
    end
    return BoundaryPoints(vcat(xy_all...); w_vs=vcat(w_all...))
end

"""
    construct_matrices(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool = true) where {Ba<:AbsBasis} → (F::Matrix, Fk::Matrix)

Constructs the Vergini–Saraceno matrices `F` and `Fk` used to compute the
generalized eigenvalue problem `F * x = λ * Fk * x` for eigenstates near
wavenumber `k`.

## Description
`F = B' * W * B` and `Fk = B' * W * dB/dk + (dB/dk)' * W * B`, where `B` is the
[`basis_matrix`](@ref) at `k`, `dB/dk` is the [`dk_matrix`](@ref), and `W` is the
diagonal quadrature weight matrix built from `pts.w_vs`, normalized by the number
of basis symmetries `nsym`. Both matrices are assembled with BLAS `syrk!`/`syr2k!`
rank-k updates on the upper triangle, which is then mirrored to the lower
triangle, to minimize memory allocations.

## Arguments
* `solver`: The [`VerginiSaracenoSolver`](@ref) whose matrices are constructed.
* `basis`: The basis used to evaluate `B` and `dB/dk`.
* `pts`: The [`BoundaryPoints`](@ref) with sampled boundary points and Vergini–Saraceno quadrature weights `w_vs`.
* `k`: The wavenumber at which the basis and its `k`-derivative are evaluated.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `F`: The `F = B' * W * B` matrix.
* `Fk`: The `Fk = B' * W * dB/dk + (dB/dk)' * W * B` matrix.
"""
function construct_matrices(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k; 
                            multithreaded = true) where {Ba<:AbsBasis}    
    @timeit_debug "construct_matrices" begin
        xy = pts.xy
        w = pts.w_vs
        N = basis.dim
        M = length(xy)
        nsym = one(eltype(w)) * (isnothing(basis.symmetries) ? 1 : length(basis.symmetries) + 1)
        
        @debug "Matrix construction started" N M k nsym
        
        # Compute basis matrix G
        @timeit_debug "basis_matrix" begin
            G = basis_matrix(basis, k, xy; multithreaded)
        end
        @debug "Basis matrix computed" size=size(G) 
        
        # Compute derivative matrix dG
        @timeit_debug "dk_matrix" begin
            dG = dk_matrix(basis, k, xy; multithreaded)
        end
        @debug "Derivative matrix computed" size=size(dG)
        
        # Compute F = G' * W * G
        @timeit_debug "compute_F" begin
            _scale_rows_sqrtw!(G, w, nsym)
            F = Matrix{eltype(G)}(undef, N, N)
            @blas_multi MAX_BLAS_THREADS BLAS.syrk!('U', 'T', one(eltype(G)), G, zero(eltype(G)), F)
            _symmetrize_from_upper!(F)
        end
        @debug "F computed" size=size(F)
        
        # Compute Fk = G' * W * dG + dG' * W * G
        @timeit_debug "compute_Fk" begin
            _scale_rows_sqrtw!(dG, w, nsym)
            Fk = Matrix{eltype(G)}(undef, N, N)
            @blas_multi_then_1 MAX_BLAS_THREADS BLAS.syr2k!('U', 'T', one(eltype(G)), G, dG, zero(eltype(G)), Fk)
            _symmetrize_from_upper!(Fk)
        end
        @debug "Fk computed" size=size(Fk) 
        
        return F, Fk
    end
end

"""
    sm_results(mu, k) → (ks::Vector, ten::Vector)

Converts the generalized eigenvalues `mu` of the Vergini–Saraceno problem at
scaling wavenumber `k` into estimated wavenumbers `ks` and their tensions `ten`.

## Description
The first-order Vergini–Saraceno wavenumber correction is
```math
k_s = k - \\frac{2}{\\mu} + \\frac{2}{k\\mu^2},
```
and the tension is defined as `ten = 2 * (2 / mu)^2`.

## Arguments
* `mu`: Vector of generalized eigenvalues from the Vergini–Saraceno generalized eigenvalue problem.
* `k`: The scaling wavenumber at which the eigenvalue problem was constructed.

## Returns
* `ks`: Vector of estimated wavenumbers.
* `ten`: Vector of tensions associated with `ks`.
"""
function sm_results(mu,k)
    ks = k .- 2 ./mu .+ 2/k ./(mu.^2) 
    ten = 2.0 .*(2.0 ./ mu).^2
    return ks, ten
end

"""
    solve(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k, dk; multithreaded::Bool = true) where {Ba<:AbsBasis} → (ks::Vector, ten::Vector)

Solves the Vergini–Saraceno generalized eigenvalue problem for `basis` on the
boundary points `pts`, returning all estimated wavenumbers within `dk` of `k` and
their tensions, sorted by wavenumber.

## Description
The matrices `F` and `Fk` are built with [`construct_matrices`](@ref), the
generalized eigenvalues `mu` are computed with [`generalized_eigvals`](@ref)
(truncated using `solver.eps`), and converted to wavenumbers and tensions with
[`sm_results`](@ref). Only the candidates satisfying `abs(ks - k) < dk` are kept.

## Arguments
* `solver`: The [`VerginiSaracenoSolver`](@ref) used to solve the eigenvalue problem.
* `basis`: The basis used to approximate the eigenstates.
* `pts`: The [`BoundaryPoints`](@ref) with sampled boundary points and quadrature weights `w_vs`.
* `k`: The scaling wavenumber around which the eigenvalue problem is constructed.
* `dk`: Half-width of the wavenumber window around `k` used to filter candidates.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `ks`: Sorted vector of estimated wavenumbers within `dk` of `k`.
* `ten`: Vector of tensions associated with `ks`.
"""
function solve(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k, dk; multithreaded = true) where {Ba<:AbsBasis}
    F, Fk = construct_matrices(solver, basis, pts, k; multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mu = generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks, ten = sm_results(mu,k)
    idx = abs.(ks.-k) .< dk
    ks = ks[idx]
    ten = ten[idx]
    p = sortperm(ks)
    return ks[p], ten[p]
end

"""
    solve(solver::VerginiSaracenoSolver, F, Fk, k, dk) → (ks::Vector, ten::Vector)

Solves the Vergini–Saraceno generalized eigenvalue problem directly from
precomputed matrices `F` and `Fk` (see [`construct_matrices`](@ref)), instead of
constructing them from a basis and boundary points. See [`solve`](@ref) for
details.

## Arguments
* `solver`: The [`VerginiSaracenoSolver`](@ref) used to solve the eigenvalue problem.
* `F`: Precomputed `F` matrix, see [`construct_matrices`](@ref).
* `Fk`: Precomputed `Fk` matrix, see [`construct_matrices`](@ref).
* `k`: The scaling wavenumber around which the eigenvalue problem is constructed.
* `dk`: Half-width of the wavenumber window around `k` used to filter candidates.

## Returns
* `ks`: Sorted vector of estimated wavenumbers within `dk` of `k`.
* `ten`: Vector of tensions associated with `ks`.
"""
function solve(solver::VerginiSaracenoSolver,F,Fk, k, dk)
    #F, Fk = construct_matrices(solver, basis, pts, k)
    @blas_multi_then_1 MAX_BLAS_THREADS mu = generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks, ten = sm_results(mu,k)
    idx = abs.(ks.-k) .< dk
    ks = ks[idx]
    ten = ten[idx]
    p = sortperm(ks)
    return ks[p], ten[p]
end

"""
    solve_vectors(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k, dk; multithreaded::Bool = true) where {Ba<:AbsBasis} → (ks::Vector, ten::Vector, X::Matrix)

Solves the Vergini–Saraceno generalized eigenvalue problem for `basis` on the
boundary points `pts`, returning the estimated wavenumbers, tensions and
eigenvectors (expressed in the original basis) for all candidates within `dk` of
`k`, sorted by wavenumber.

## Description
The matrices `F` and `Fk` are built with [`construct_matrices`](@ref), the
generalized eigenproblem is solved with [`generalized_eigen`](@ref) (truncated
using `solver.eps`) to obtain eigenvalues `mu`, eigenvectors `Z` in the reduced
space and the change-of-basis matrix `C`. Wavenumbers and tensions are computed
with [`sm_results`](@ref), candidates with `abs(ks - k) < dk` are kept, and the
eigenvectors are transformed back into the original basis as `X = C * Z`,
rescaled by `sqrt.(ten)`.

## Arguments
* `solver`: The [`VerginiSaracenoSolver`](@ref) used to solve the eigenvalue problem.
* `basis`: The basis used to approximate the eigenstates.
* `pts`: The [`BoundaryPoints`](@ref) with sampled boundary points and quadrature weights `w_vs`.
* `k`: The scaling wavenumber around which the eigenvalue problem is constructed.
* `dk`: Half-width of the wavenumber window around `k` used to filter candidates.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `ks`: Sorted vector of estimated wavenumbers within `dk` of `k`.
* `ten`: Vector of tensions associated with `ks`.
* `X`: Matrix whose columns are the eigenvectors expressed in the original basis, scaled by `sqrt.(ten)`.
"""
function solve_vectors(solver::VerginiSaracenoSolver, basis::Ba, pts::BoundaryPoints, k, dk; multithreaded = true) where {Ba<:AbsBasis}
    F, Fk = construct_matrices(solver, basis, pts, k; multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mu, Z, C = generalized_eigen(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks, ten = sm_results(mu,k)
    idx = abs.(ks.-k) .< dk
    ks = ks[idx]
    ten = ten[idx]
    Z = Z[:,idx]
    X = C*Z #transform into original basis 
    X = (sqrt.(ten))' .* X
    p = sortperm(ks)
    return  ks[p], ten[p], X[:,p]
end

