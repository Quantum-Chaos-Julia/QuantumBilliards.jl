################################################################################
# MEMORY ESTIMATES FOR compute_spectrum
################################################################################
#
# Upper-bound estimates of the resident memory needed by the `compute_spectrum`
# pipelines, evaluated before any expensive work starts. All estimates are
# sized at `k2` (largest boundary/basis dimension of the sweep).
#
# Notation: `Nf` = full boundary points, `M` = Fredholm matrix dimension after
# symmetry reduction, `nq` = Beyn contour nodes, `p` = polynomial/Taylor degree.
# One complex `M×M` matrix is `16M²` bytes. The geometry cache plus Kress
# matrix of the Taylor/direct BIM assembly hold about 5 real `Nf×Nf` arrays.

const _CPLX_BYTES = 16.0
const _REAL_BYTES = 8.0
const _BIM_GEOM_UNITS = 2.5          # 5 real Nf² arrays expressed in complex Nf² units
const _PTS_BYTES_PER_POINT = 200.0   # BoundaryPoints storage per boundary node
const _MEMORY_SLACK = 1.25           # allowance for garbage not yet collected

"""
    MemoryEstimate

Estimated memory of a `compute_spectrum` run in bytes.

## Fields
* `working::Float64`: Peak transient working set of a single window/panel solve.
* `persistent::Float64`: Results retained until the end of the sweep (eigenvectors, boundary points).
"""
struct MemoryEstimate
    working::Float64
    persistent::Float64
end

"""
    total_bytes(est::MemoryEstimate) → Float64

Total estimated bytes, `est.working + est.persistent`.
"""
total_bytes(est::MemoryEstimate) = est.working + est.persistent

function Base.show(io::IO, est::MemoryEstimate)
    print(io, "MemoryEstimate(working = ", round(est.working / 2^30; digits=2), " GiB, persistent = ", round(est.persistent / 2^30; digits=2), " GiB)")
end

"""
    default_memory_limit() → Float64

Default `max_memory` of `compute_spectrum`: 90% of the physical RAM, in bytes.
"""
default_memory_limit() = 0.9 * Float64(Sys.total_memory())

_expected_states(billiard, k1, k2, fundamental_only::Bool) = max(0.0, Float64(state_at_k(k2, billiard; fundamental_only)) - Float64(state_at_k(k1, billiard; fundamental_only)))

# Full boundary size and symmetry-reduced matrix size at the largest wavenumber.
function _bim_sizes(solver, billiard, k2)
    pts = evaluate_points(solver, billiard, k2)
    return Float64(length(pts.xy)), Float64(boundary_matrix_size(solver.kernel, pts))
end

"""
    estimate_memory(solver, [basis,] billiard, k1, k2) → MemoryEstimate

Estimate the peak memory of `compute_spectrum(solver, [basis,] billiard, k1, k2)`
without running the solve. The estimate is sized at `k2` and is meant as an
order-of-magnitude upper bound, not an exact figure.

## Arguments
* `solver`: A [`VerginiSaracenoSolver`](@ref) (with `basis`), [`BeynSolver`](@ref), [`ExpandedBIMSolver`](@ref) or [`CORKSolver`](@ref).
* `basis`: Basis, only for [`VerginiSaracenoSolver`](@ref).
* `billiard`: Billiard geometry.
* `k1`, `k2`: Wavenumber interval.

## Returns
* `MemoryEstimate`: Peak working set and retained results in bytes.
"""
function estimate_memory(solver::VerginiSaracenoSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = Float64(max(solver.min_dim, round(Int, L * k2 * solver.dim_scaling_factor / (2 * pi))))
    M = Float64(length(evaluate_points(solver, billiard, k2).w_vs))
    p = uses_taylor(basis) ? Float64(basis.taylor_degree) : -1.0
    taylor_units = p >= 0 ? p + 2 : 0.0     # coefficient tensor + angular factors
    working = _REAL_BYTES * ((taylor_units + 6) * M * dim + 10 * dim^2)
    ns = _expected_states(billiard, k1, k2, basis.symmetries !== nothing)
    persistent = solver.eigenvectors ? _REAL_BYTES * ns * dim : 0.0
    return MemoryEstimate(working, persistent)
end

function estimate_memory(solver::BeynSolver, billiard::AbsBilliard, k1, k2)
    fundamental_only = solver.kernel.symmetry !== nothing
    Nf, M = _bim_sizes(solver, billiard, k2)
    r = Float64(min(solver.r, round(Int, M)))
    nw = length(plan_weyl_windows(billiard, Float64(k1), Float64(k2); nlevels=solver.nlevels, Rmax=solver.Rmax, fundamental_only))
    ns = _expected_states(billiard, k1, k2, fundamental_only)
    streaming = !solver.use_taylor && !solver.use_chebyshev     # direct backend keeps O(1) contour matrices
    working = _CPLX_BYTES * ((streaming ? 3 : solver.nq) * M^2 + _BIM_GEOM_UNITS * Nf^2 + 6 * M * r)
    persistent = _PTS_BYTES_PER_POINT * nw * Nf
    (solver.imag_k_check || solver.eigenvectors) && (persistent += _CPLX_BYTES * ns * M)
    solver.eigenvectors && (persistent += _CPLX_BYTES * ns * Nf)
    return MemoryEstimate(working, persistent)
end

function estimate_memory(solver::CORKSolver, billiard::AbsBilliard, k1, k2)
    fundamental_only = solver.kernel.symmetry !== nothing
    Nf, M = _bim_sizes(solver, billiard, k2)
    p = Float64(max(solver.p, ceil(Int, sqrt(solver.block_size))))
    nw = length(plan_weyl_windows(billiard, Float64(k1), Float64(k2); nlevels=solver.nlevels, Rmax=solver.Rmax, fundamental_only))
    ns = _expected_states(billiard, k1, k2, fundamental_only)
    build = (p + 1) * M^2 + _BIM_GEOM_UNITS * Nf^2 + 2 * M^2     # polynomial B + geometry + validation matrices
    krylov = (3p + 5) * M^2                                      # B, P(0) + LU, U, W, G
    working = _CPLX_BYTES * max(build, krylov)
    persistent = 0.0
    if solver.eigenvectors
        persistent += _CPLX_BYTES * ns * Nf + _PTS_BYTES_PER_POINT * nw * Nf
    end
    return MemoryEstimate(working, persistent)
end

function estimate_memory(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2)
    Nf, M = _bim_sizes(solver, billiard, k2)
    coeff_units = solver.use_taylor ? solver.taylor_degree + 1.0 : 0.0
    working = _CPLX_BYTES * ((coeff_units + 6) * M^2 + _BIM_GEOM_UNITS * Nf^2)
    return MemoryEstimate(working, 0.0)
end

# Throw before any heavy work if the estimate exceeds `max_memory` (bytes; `Inf` disables).
function _enforce_memory_limit(max_memory::Real, name::AbstractString, args...)
    isfinite(max_memory) || return nothing
    est = estimate_memory(args...)
    need = _MEMORY_SLACK * total_bytes(est)
    @debug "$name memory estimate" maxlog=1 working_GiB=est.working/2^30 persistent_GiB=est.persistent/2^30 need_GiB=need/2^30 limit_GiB=max_memory/2^30
    need <= max_memory || error("$name: estimated peak memory $(round(need / 2^30; digits=1)) GiB (working $(round(est.working / 2^30; digits=1)) GiB + retained $(round(est.persistent / 2^30; digits=1)) GiB, incl. GC slack) exceeds the limit of $(round(max_memory / 2^30; digits=1)) GiB. Compute the range in smaller chunks, use `eigenvectors=false`, or lower `nq`/`nlevels`/`taylor_degree`; pass `max_memory=Inf` to disable this check.")
    return nothing
end
