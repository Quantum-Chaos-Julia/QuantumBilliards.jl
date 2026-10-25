################################################################################
# PARTICULAR SOLUTIONS METHOD (PSM)
#
# This file implements the particular solutions method for computing quantum
# billiard eigenvalues by searching for Helmholtz solutions that are small on
# the boundary while remaining nontrivial in the interior.
#
# For a basis {ϕₙ(k)} satisfying the Helmholtz equation inside the billiard,
# define the boundary and interior collocation matrices
#
#                                 Bᵢₙ = ϕₙ(xᵢ),
#
#                             (Bint)ⱼₙ = ϕₙ(yⱼ),
#
# where {xᵢ} are boundary points and {yⱼ} are interior points. For a coefficient
# vector c, the corresponding trial solution has boundary and interior samples
#
#                               Bc,    Bint c.
#
# The PSM tension at fixed wavenumber k is the minimum relative boundary norm
#
#                                        ||Bc||₂
#                           t(k) = min  --------,
#                                   c  ||Bint c||₂
#
# and is therefore the smallest generalized singular value of the matrix pair
#
#                              GSVD(B, Bint).
#
# Near a Dirichlet eigenvalue, a nontrivial Helmholtz solution can become small
# on the boundary while retaining finite interior norm, producing a sharp
# minimum of t(k). The spectrum is obtained by sweeping or minimizing this
# tension as a function of k.
################################################################################

"""
    ParticularSolutionsMethodSolver{T} <: SweepBasisSolver

`ParticularSolutionsMethodSolver` is a concrete [`SweepBasisSolver`](@ref)
implementing the particular solutions method (PSM) for computing quantum
billiard spectra by sweeping over individual wavenumbers.

## Description
For a fixed wavenumber `k`, the method constructs the basis matrices `B`
(boundary) and `B_int` (interior) (see [`construct_matrices`](@ref)) from a
boundary quadrature and a set of random interior points (see
[`evaluate_points`](@ref)), and defines the tension from the smallest singular
value of `B` normalized against `B_int` (see [`solve`](@ref)). A sequence of
tensions over a range of wavenumbers is minimized/scanned by
[`solve_wavenumber`](@ref) or [`k_sweep`](@ref), inherited from
[`SweepBasisSolver`](@ref), to locate the eigenvalues of the billiard.

## Attributes
* `dim_scaling_factor`: Scaling factor used to determine the basis dimension from the boundary length and wavenumber.
* `pts_scaling_factor`: Vector of scaling factors, one per fundamental boundary curve, used to determine the number of boundary sampling points.
* `int_pts_scaling_factor`: Scaling factor used to determine the number of interior sampling points.
* `sampler`: Vector of samplers, one per fundamental boundary curve, used to generate boundary points.
* `eps`: Relative tolerance used to filter small singular values.
* `min_dim`: Minimum basis dimension.
* `min_pts`: Minimum number of boundary sampling points.
* `min_int_pts`: Minimum number of interior sampling points.

## API
The following functions can be evaluated for this type:
- [`evaluate_points`](@ref)
- [`construct_matrices`](@ref)
- [`solve_full`](@ref)
- [`solve_with_rank_reduction`](@ref)
- [`solve`](@ref)
- [`solve_vect`](@ref)
- [`solve_wavenumber`](@ref)
- [`k_sweep`](@ref)

`solve_wavenumber` and `k_sweep` are inherited for free from the shared
[`SweepBasisSolver`](@ref) generics.
"""
struct ParticularSolutionsMethodSolver{T} <: SweepBasisSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    int_pts_scaling_factor::T
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
    min_int_pts::Int64
end

"""
    ParticularSolutionsMethodSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, int_pts_scaling_factor::T; min_dim::Int = 100, min_pts::Int = 500, min_int_pts::Int = 500) where {T<:Real} → solver::ParticularSolutionsMethodSolver{T}

Constructs a [`ParticularSolutionsMethodSolver`](@ref) with a single
`GaussLegendreNodes` sampler shared by every fundamental boundary curve.

## Arguments
* `dim_scaling_factor`: Scaling factor used to determine the basis dimension.
* `pts_scaling_factor`: Scaling factor, or vector thereof (one per fundamental boundary curve), used to determine the number of boundary sampling points.
* `int_pts_scaling_factor`: Scaling factor used to determine the number of interior sampling points.

## Keyword arguments
* `min_dim::Int = 100`: Minimum basis dimension.
* `min_pts::Int = 500`: Minimum number of boundary sampling points.
* `min_int_pts::Int = 500`: Minimum number of interior sampling points.

## Returns
* `solver`: A [`ParticularSolutionsMethodSolver{T}`](@ref) instance.
"""
function ParticularSolutionsMethodSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}},
                                    int_pts_scaling_factor::T;
                                    min_dim::Int=100, min_pts::Int=500, min_int_pts::Int=500) where {T<:Real}
    d = dim_scaling_factor
    bs = pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    sampler = [GaussLegendreNodes()]
    return ParticularSolutionsMethodSolver(d, bs, int_pts_scaling_factor, sampler, eps(T), min_dim, min_pts, min_int_pts)
end

"""
    ParticularSolutionsMethodSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, int_pts_scaling_factor::T, samplers::Vector{<:AbsSampler}; min_dim::Int = 100, min_pts::Int = 500, min_int_pts::Int = 500) where {T<:Real} → solver::ParticularSolutionsMethodSolver{T}

Constructs a [`ParticularSolutionsMethodSolver`](@ref) with a user-supplied sampler
for each fundamental boundary curve.

## Arguments
* `dim_scaling_factor`: Scaling factor used to determine the basis dimension.
* `pts_scaling_factor`: Scaling factor, or vector thereof (one per fundamental boundary curve), used to determine the number of boundary sampling points.
* `int_pts_scaling_factor`: Scaling factor used to determine the number of interior sampling points.
* `samplers`: Vector of samplers, one per fundamental boundary curve.

## Keyword arguments
* `min_dim::Int = 100`: Minimum basis dimension.
* `min_pts::Int = 500`: Minimum number of boundary sampling points.
* `min_int_pts::Int = 500`: Minimum number of interior sampling points.

## Returns
* `solver`: A [`ParticularSolutionsMethodSolver{T}`](@ref) instance.
"""
function ParticularSolutionsMethodSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}},
                                    int_pts_scaling_factor::T, samplers::Vector{<:AbsSampler};
                                    min_dim::Int=100, min_pts::Int=500, min_int_pts::Int=500) where {T<:Real}
    d = dim_scaling_factor
    bs = pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    return ParticularSolutionsMethodSolver(d, bs, int_pts_scaling_factor, samplers, eps(T), min_dim, min_pts, min_int_pts)
end

"""
    evaluate_points(solver::ParticularSolutionsMethodSolver, billiard::Bi, k) where {Bi<:AbsBilliard} → pts::BoundaryPoints

Samples the fundamental boundary and billiard interior required by the particular solutions method.

## Description
Each fundamental boundary curve is sampled with its associated sampler. For
parameter-space quadrature nodes `tᵢ` and weights `dtᵢ`, the physical boundary
quadrature weights are `dsᵢ = |r'(tᵢ)|dtᵢ`, so arbitrary regular
parametrizations are supported without assuming constant-speed curves. The
stored coordinate `s` is the cumulative physical arc length along the
fundamental boundary.

## Arguments
* `solver::ParticularSolutionsMethodSolver`: Solver defining the boundary and interior sampling densities.
* `billiard::Bi`: Billiard whose fundamental boundary and interior are sampled.
* `k`: Wavenumber used to determine the number of sampling points.

## Returns
* `pts::BoundaryPoints`: Boundary points with `xy`, `normal`, `s`, `ds` and `xy_int` populated.
"""
function evaluate_points(solver::ParticularSolutionsMethodSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    bs, samplers = adjust_scaling_and_samplers(solver, billiard); curves = get_boundary_curves(billiard); T = eltype(solver.pts_scaling_factor)
    Ns = _determine_bp_sizes(curves, bs, k); M = length(Ns)
    xy_all = Vector{Vector{SVector{2,T}}}(undef, M); normal_all = Vector{Vector{SVector{2,T}}}(undef, M)
    s_all = Vector{Vector{T}}(undef, M); ds_all = Vector{Vector{T}}(undef, M); L0 = zero(T)
    for i in eachindex(curves)
        crv = curves[i]; sampler = samplers[i]; t, dt = sample_points(sampler, Ns[i])
        xy = curve(crv, t); tan = tangent(crv, t); ds = norm.(tan).*dt
        normal = domain_gradient_vector(crv, xy); normal .= normal./norm.(normal)
        xy_all[i] = xy; normal_all[i] = normal; s_all[i] = arc_length(crv, t).+L0; ds_all[i] = ds
        L0 += crv.length
    end
    M_int = max(solver.min_int_pts, round(Int, k*L0*solver.int_pts_scaling_factor/(2pi)))
    xy_int = random_interior_points(billiard, M_int)
    return BoundaryPoints(vcat(xy_all...); normal = vcat(normal_all...), s = vcat(s_all...), ds = vcat(ds_all...), xy_int = xy_int)
end

"""
    construct_matrices(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool = true) where {Ba<:AbsBasis} → (B::Matrix, B_int::Matrix)

Constructs the boundary basis matrix `B` and the interior basis matrix
`B_int` at wavenumber `k`.

## Arguments
* `solver`: The [`ParticularSolutionsMethodSolver`](@ref) whose matrices are constructed.
* `basis`: The basis used to evaluate `B` and `B_int`.
* `pts`: The [`BoundaryPoints`](@ref) with sampled boundary points `xy` and interior points `xy_int`.
* `k`: The wavenumber at which the basis is evaluated.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `B`: Basis matrix evaluated at the boundary points `pts.xy`.
* `B_int`: Basis matrix evaluated at the interior points `pts.xy_int`.
"""
function construct_matrices(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true) where {Ba<:AbsBasis}
    pts_bd = pts.xy
    pts_int = pts.xy_int
    @blas_1 begin
        B = basis_matrix(basis, k, pts_bd; multithreaded=multithreaded)
        B_int = basis_matrix(basis, k, pts_int; multithreaded=multithreaded)
    end
    return B, B_int
end

"""
    solve_full(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool = true) where {Ba<:AbsBasis} → t::Real

Computes the particular solutions method tension at wavenumber `k` from the
full (non rank-reduced) singular value decomposition of `B`/`B_int`.

## Description
The tension is the smallest generalized singular value of the pencil
`(B, B_int)`, computed directly with `svdvals(B, B_int)`. This is more
expensive but more numerically robust than [`solve_with_rank_reduction`](@ref)
for small-to-moderate basis dimensions.

## Arguments
* `solver`: The [`ParticularSolutionsMethodSolver`](@ref) used to solve the eigenvalue problem.
* `basis`: The basis used to approximate the eigenstate.
* `pts`: The [`BoundaryPoints`](@ref) with sampled boundary/interior points.
* `k`: The wavenumber at which the tension is evaluated.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `t`: The tension at wavenumber `k`, the smallest generalized singular value of `(B, B_int)`.
"""
function solve_full(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true) where {Ba<:AbsBasis}
    B, B_int = construct_matrices(solver, basis, pts, k; multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS solution=svdvals(B,B_int)
    return minimum(solution)
end

@inline function _numerical_rank_from_F(F, tol::Real)
    A = F.factors
    n = min(size(A,1), size(A,2))
    n == 0 && return 0
    t = tol*abs(@inbounds A[1,1])
    @inbounds for i in n:-1:1
        if abs(A[i,i]) > t
            return i
        end
    end
    return 0
end

"""
    solve_with_rank_reduction(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool = true, tol::Real = 1e-10) where {Ba<:AbsBasis} → t::Real

Computes the particular solutions method tension at wavenumber `k` using a
rank-reduced singular value decomposition of `B`/`B_int` for improved
performance at large basis dimension.

## Description
`B_int` is factorized with a rank-revealing pivoted QR, `B_int * P = Q * R`.
Only the numerically well-determined leading `r × r` block of `R` (with `r`
determined by [`_numerical_rank_from_F`](@ref) and `tol`) is used to reduce `B`
via a stable triangular solve, after which the tension is the square root of
the smallest eigenvalue of the small `r × r` matrix `B_reduced' * B_reduced`.
This avoids ever forming the full generalized SVD of `(B, B_int)`.

## Arguments
* `solver`: The [`ParticularSolutionsMethodSolver`](@ref) used to solve the eigenvalue problem.
* `basis`: The basis used to approximate the eigenstate.
* `pts`: The [`BoundaryPoints`](@ref) with sampled boundary/interior points.
* `k`: The wavenumber at which the tension is evaluated.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.
* `tol::Real = 1e-10`: Relative tolerance used to determine the numerical rank of `B_int`'s QR factorization. Should increase with `k` since `R[1,1]` decreases with `k`.

## Returns
* `t`: The tension at wavenumber `k`.
"""
function solve_with_rank_reduction(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true, tol=1e-10) where {Ba<:AbsBasis}
    B, C = construct_matrices(solver, basis, pts, k; multithreaded=multithreaded)
    T = eltype(B)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        F = qr!(C, ColumnNorm()) # rank-revealing QR with column pivoting: C*P = Q*R, overwrite C since it is no longer needed
        r = _numerical_rank_from_F(F, tol) # numerical rank r from the packed factors (no copy)
        r == 0 && return Inf # degenerate fallback
        Rview = @views UpperTriangular(view(F.factors, 1:r, 1:r)) # well-determined r x r block of R (no copy)
        piv = F.p # permutation vector piv such that C[:,piv] = Q*R
        B = @views B[:,piv[1:r]]/Rview # Br = B[:,piv[1:r]] * R^{-1} via stable triangular solve, overwrite B
        r = size(B, 2)
        B_sq = Matrix{T}(undef, r, r)
        BLAS.syrk!('U','T',one(T),B,zero(T),B_sq) # B_sq[u ∈ upper] = B'*B
        _symmetrize_from_upper!(B_sq)
        return sqrt(abs(eigmin(Symmetric(B_sq)))) # smallest singular value of B via eigmin(B'B)
    end
end

"""
    solve(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool = true, use_rank_reduction::Bool = true, tol::Real = 1e-10) where {Ba<:AbsBasis} → t::Real

Computes the particular solutions method tension at wavenumber `k` for `basis`
on the boundary/interior points `pts`.

## Description
Dispatches to [`solve_with_rank_reduction`](@ref) by default, which is
substantially faster at large basis dimension; set `use_rank_reduction = false`
to instead use the more robust, full-SVD [`solve_full`](@ref).

## Arguments
* `solver`: The [`ParticularSolutionsMethodSolver`](@ref) used to solve the eigenvalue problem.
* `basis`: The basis used to approximate the eigenstate.
* `pts`: The [`BoundaryPoints`](@ref) with sampled boundary/interior points.
* `k`: The wavenumber at which the tension is evaluated.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.
* `use_rank_reduction::Bool = true`: If true, uses [`solve_with_rank_reduction`](@ref); otherwise uses [`solve_full`](@ref).
* `tol::Real = 1e-10`: Rank tolerance passed to [`solve_with_rank_reduction`](@ref).

## Returns
* `t`: The tension at wavenumber `k`.
"""
function solve(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true, use_rank_reduction::Bool=true, tol=1e-10) where {Ba<:AbsBasis}
    if use_rank_reduction
        return solve_with_rank_reduction(solver, basis, pts, k; multithreaded=multithreaded, tol=tol)
    else
        return solve_full(solver, basis, pts, k; multithreaded=multithreaded)
    end
end

"""
    solve_vect(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool = true, tol::Real = 1e-10) where {Ba<:AbsBasis} → (t::Real, x::Vector)

Computes the particular solutions method tension and the corresponding
eigenvector (expressed in the original basis) at wavenumber `k`.

## Description
Same rank-revealing QR reduction of `B_int` as
[`solve_with_rank_reduction`](@ref), but additionally recovers the minimizing
coefficient vector: the reduced matrix `B_reduced` is factorized with a full
SVD, and its right singular vector for the smallest singular value is
back-substituted through the QR triangular factor and un-pivoted to give the
coefficient vector `x` in the original basis ordering expected by
[`BasisEigenstate`](@ref).

## Arguments
* `solver`: The [`ParticularSolutionsMethodSolver`](@ref) used to solve the eigenvalue problem.
* `basis`: The basis used to approximate the eigenstate.
* `pts`: The [`BoundaryPoints`](@ref) with sampled boundary/interior points.
* `k`: The wavenumber at which the eigenstate is evaluated.

## Keyword arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.
* `tol::Real = 1e-10`: Relative tolerance used to determine the numerical rank of `B_int`'s QR factorization.

## Returns
* `t`: The tension at wavenumber `k`.
* `x`: The eigenvector expressed in the original basis coefficient ordering.
"""
function solve_vect(solver::ParticularSolutionsMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true, tol=1e-10) where {Ba<:AbsBasis}
    B, C = construct_matrices(solver, basis, pts, k; multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        T = eltype(B)
        F = qr(C, ColumnNorm()) # rank-revealing QR with column pivoting: C*P = Q*R
        R = UpperTriangular(F.R)
        piv = F.p # permutation vector piv such that C[:,piv] = Q*R
        r = findlast(i -> abs(R[i,i]) > tol*abs(R[1,1]), 1:min(size(R)...)) # numerical rank
        isnothing(r) && return (Inf, zeros(T, size(B,2))) # degenerate fallback
        Rr = R[1:r,1:r] # well-determined r x r block
        Br = B[:,piv[1:r]]/Rr # Br = B[:,piv[1:r]] * Rr^{-1} via stable triangular solve
        _,S,Vt = LAPACK.gesvd!('A','A',Br) # SVD(Br) = U*Diag(S)*Vt
        idx = findmin(S)[2]
        mu = S[idx]
        u_mu = Vt[idx,:]
        y = real.(u_mu)
        chat = zeros(T, size(B,2))
        chat[piv[1:r]] = Rr\y # back-substitute: c[piv[1:r]] = Rr^{-1} y, rest are zero
        return mu, chat
    end
end
