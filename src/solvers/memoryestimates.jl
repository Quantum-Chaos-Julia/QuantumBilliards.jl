################################################################################
# MEMORY ESTIMATES FOR compute_spectrum
################################################################################
#
# Upper-bound estimates of the resident memory needed by the `compute_spectrum`
# pipelines, evaluated before any expensive work starts. The working set is the
# maximum over the windows of the layout `plan_computation` reports.
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

# Per-window working set and retained results, given the full boundary size `Nf`
# (or basis size `dim`), the matrix dimension `M` and the Weyl state count `ns`.
function _working_bytes(solver::VerginiSaracenoSolver, basis::AbsBasis, dim::Real, M::Real)
    taylor_units = uses_taylor(basis) ? basis.taylor_degree + 2.0 : 0.0     # coefficient tensor + angular factors
    return _REAL_BYTES * ((taylor_units + 6) * M * dim + 10 * dim^2)
end

_persistent_bytes(solver::VerginiSaracenoSolver, ::AbsBasis, ns::Real, dim::Real) = solver.eigenvectors ? _REAL_BYTES * ns * dim : 0.0

function _working_bytes(solver::BeynSolver, Nf::Real, M::Real)
    r = Float64(min(solver.r, round(Int, M)))
    streaming = !solver.use_taylor && !solver.use_chebyshev     # direct backend keeps O(1) contour matrices
    return _CPLX_BYTES * ((streaming ? 3 : solver.nq) * M^2 + _BIM_GEOM_UNITS * Nf^2 + 6 * M * r)
end

function _persistent_bytes(solver::BeynSolver, nw::Integer, ns::Real, Nf::Real, M::Real)
    persistent = _PTS_BYTES_PER_POINT * nw * Nf
    (solver.imag_k_check || solver.eigenvectors) && (persistent += _CPLX_BYTES * ns * M)
    solver.eigenvectors && (persistent += _CPLX_BYTES * ns * Nf)
    return persistent
end

function _working_bytes(solver::CORKSolver, Nf::Real, M::Real)
    p = Float64(max(solver.p, ceil(Int, sqrt(solver.block_size))))
    build = (p + 1) * M^2 + _BIM_GEOM_UNITS * Nf^2 + 2 * M^2     # polynomial B + geometry + validation matrices
    krylov = (3p + 5) * M^2                                      # B, P(0) + LU, U, W, G
    return _CPLX_BYTES * max(build, krylov)
end

_persistent_bytes(solver::CORKSolver, nw::Integer, ns::Real, Nf::Real, M::Real) = solver.eigenvectors ? _CPLX_BYTES * ns * Nf + _PTS_BYTES_PER_POINT * nw * Nf : 0.0

function _working_bytes(solver::ExpandedBIMSolver, Nf::Real, M::Real)
    coeff_units = solver.use_taylor ? solver.taylor_degree + 1.0 : 0.0
    return _CPLX_BYTES * ((coeff_units + 6) * M^2 + _BIM_GEOM_UNITS * Nf^2)
end

_persistent_bytes(::ExpandedBIMSolver, nw::Integer, ns::Real, Nf::Real, M::Real) = 0.0

# Throw before any heavy work if the estimate exceeds `max_memory` (bytes; `Inf` disables).
function _enforce_memory_limit(max_memory::Real, name::AbstractString, args...; kwargs...)
    isfinite(max_memory) || return nothing
    est = estimate_memory(args...; kwargs...)
    need = _MEMORY_SLACK * total_bytes(est)
    @debug "$name memory estimate" maxlog=1 working_GiB=est.working/2^30 persistent_GiB=est.persistent/2^30 need_GiB=need/2^30 limit_GiB=max_memory/2^30
    need <= max_memory || error("$name: estimated peak memory $(round(need / 2^30; digits=1)) GiB (working $(round(est.working / 2^30; digits=1)) GiB + retained $(round(est.persistent / 2^30; digits=1)) GiB, incl. GC slack) exceeds the limit of $(round(max_memory / 2^30; digits=1)) GiB. Compute the range in smaller chunks, use `eigenvectors=false`, or lower `nq`/`nlevels`/`taylor_degree`; pass `max_memory=Inf` to disable this check.")
    return nothing
end
