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
    print(io, "MemoryEstimate(working = ", _format_bytes(est.working), ", persistent = ", _format_bytes(est.persistent), ")")
end

"""
    default_memory_limit() → Float64

Default `max_memory` of `compute_spectrum`: 90% of the physical RAM, in bytes.
"""
default_memory_limit() = 0.9 * Float64(Sys.total_memory())

# Per-solver breakdown, given the full boundary size `Nf` (or basis size `dim`), the
# matrix dimension `M`, the window count `nw` and the Weyl state count `ns`.
#
# `_memory_stages` lists the transient allocations that are alive together at one step of
# the pipeline, as `label => bytes`; the working set is the largest stage.
# `_memory_retained` lists the results that stay allocated until the sweep ends.
_stage_peak(stages) = isempty(stages) ? 0.0 : maximum(last, stages)
_geometry_bytes(Nf::Real) = _CPLX_BYTES * _BIM_GEOM_UNITS * Nf^2
_cheb_lookup_bytes(Nf::Real) = 24.0 * Nf^2           # Int32 + Float64 panel index and coordinate, for Hankel and Bessel-J

function _memory_stages(solver::VerginiSaracenoSolver, basis::AbsBasis, dim::Real, M::Real)
    eigensolve = _REAL_BYTES * 10 * dim^2                # F, Fk and the generalized eigensolve workspace
    uses_taylor(basis) || return Pair{String,Float64}["window solve (G, dG, F, Fk, eigensolve)" => _REAL_BYTES * 2 * M * dim + eigensolve]
    p = Float64(basis.taylor_degree)
    return Pair{String,Float64}[
        "Taylor validation, once (cache + 4 test matrices)" => _REAL_BYTES * (p + 5) * M * dim,
        "panel solve (Taylor cache, G, dG, F, Fk, eigensolve)" => _REAL_BYTES * (p + 3) * M * dim + eigensolve,
    ]
end

function _memory_retained(solver::VerginiSaracenoSolver, ::AbsBasis, ns::Real, dim::Real)
    solver.eigenvectors || return Pair{String,Float64}[]
    return Pair{String,Float64}["eigenstates (states x basis dim)" => _REAL_BYTES * ns * dim]
end

function _memory_stages(solver::BeynSolver, Nf::Real, M::Real; nw::Integer=1)
    r = Float64(min(solver.r, round(Int, M)))
    geom = _geometry_bytes(Nf)
    moments = _CPLX_BYTES * 6 * M * r                    # probes, contour moments, SVD and projection blocks
    stages = Pair{String,Float64}[]
    if solver.use_taylor
        push!(stages, "degree tuning (10 test matrices + geometry)" => _CPLX_BYTES * 10 * M^2 + geom)
        push!(stages, "contour matrices ($(solver.nq) x M^2, geometry, moments)" => _CPLX_BYTES * solver.nq * M^2 + geom + moments)
    elseif solver.use_chebyshev
        push!(stages, "contour matrices ($(solver.nq) x M^2, geometry, lookup, moments)" => _CPLX_BYTES * solver.nq * M^2 + geom + _cheb_lookup_bytes(Nf) + moments)
    else
        push!(stages, "streamed contour sweep (1 matrix + LU, geometry, moments)" => _CPLX_BYTES * 3 * M^2 + geom + moments)
    end
    if solver.imag_k_check
        group = Float64(min(solver.imag_k_group_size, round(Int, r)))
        push!(stages, "imaginary-k check ($(Int(group)) matrices, $nw geometry caches)" => _CPLX_BYTES * group * M^2 + nw * geom + _cheb_lookup_bytes(Nf))
    end
    return stages
end

function _memory_retained(solver::BeynSolver, nw::Integer, ns::Real, Nf::Real, M::Real)
    retained = Pair{String,Float64}["boundary points ($nw windows)" => _PTS_BYTES_PER_POINT * nw * Nf]
    (solver.imag_k_check || solver.eigenvectors) && push!(retained, "layer densities, reduced (states x M)" => _CPLX_BYTES * ns * M)
    solver.eigenvectors && push!(retained, "eigenstates, full boundary (states x Nf)" => _CPLX_BYTES * ns * Nf)
    return retained
end

function _memory_stages(solver::CORKSolver, Nf::Real, M::Real; nw::Integer=1)
    p = Float64(max(solver.p, ceil(Int, sqrt(solver.block_size))))
    return Pair{String,Float64}[
        "polynomial tuning (5 references, 2 candidates, geometry)" => _CPLX_BYTES * (2p + 8) * M^2 + _geometry_bytes(Nf),
        "Krylov solve (B, W, G, Hessenberg, P(0) + LU, U)" => _CPLX_BYTES * (3p + 6) * M^2,
    ]
end

function _memory_retained(solver::CORKSolver, nw::Integer, ns::Real, Nf::Real, M::Real)
    solver.eigenvectors || return Pair{String,Float64}[]
    return Pair{String,Float64}[
        "boundary points ($nw windows)" => _PTS_BYTES_PER_POINT * nw * Nf,
        "eigenstates, full boundary (states x Nf)" => _CPLX_BYTES * ns * Nf,
    ]
end

function _memory_stages(solver::ExpandedBIMSolver, Nf::Real, M::Real; nw::Integer=1)
    solver.use_taylor || return Pair{String,Float64}["direct assembly (A, A', A'', geometry)" => _CPLX_BYTES * 3 * M^2 + _geometry_bytes(Nf)]
    coeffs = _CPLX_BYTES * (solver.taylor_degree + 1.0) * M^2
    return Pair{String,Float64}[
        "panel assembly (Taylor coefficients, geometry)" => coeffs + _geometry_bytes(Nf),
        "Horner evaluation and local solve (coefficients, A, A', A'')" => coeffs + _CPLX_BYTES * 3 * M^2,
    ]
end

_memory_retained(::ExpandedBIMSolver, nw::Integer, ns::Real, Nf::Real, M::Real) = Pair{String,Float64}[]

_working_bytes(solver::VerginiSaracenoSolver, basis::AbsBasis, dim::Real, M::Real) = _stage_peak(_memory_stages(solver, basis, dim, M))
_working_bytes(solver::Union{BeynSolver,CORKSolver,ExpandedBIMSolver}, Nf::Real, M::Real; nw::Integer=1) = _stage_peak(_memory_stages(solver, Nf, M; nw))

_persistent_bytes(solver::VerginiSaracenoSolver, basis::AbsBasis, ns::Real, dim::Real) = sum(last, _memory_retained(solver, basis, ns, dim); init=0.0)
_persistent_bytes(solver::Union{BeynSolver,CORKSolver,ExpandedBIMSolver}, nw::Integer, ns::Real, Nf::Real, M::Real) = sum(last, _memory_retained(solver, nw, ns, Nf, M); init=0.0)

_format_bytes(b::Real) = b >= 2^30 ? string(round(b / 2^30; digits=2), " GiB") : b >= 2^20 ? string(round(b / 2^20; digits=1), " MiB") : b >= 2^10 ? string(round(b / 2^10; digits=1), " KiB") : string(round(Int, b), " B")

# Throw before any heavy work if the estimate exceeds `max_memory` (bytes; `Inf` disables).
function _enforce_memory_limit(max_memory::Real, name::AbstractString, args...; kwargs...)
    isfinite(max_memory) || return nothing
    est = estimate_memory(args...; kwargs...)
    need = _MEMORY_SLACK * total_bytes(est)
    @debug "$name memory estimate" maxlog=1 working_GiB=est.working/2^30 persistent_GiB=est.persistent/2^30 need_GiB=need/2^30 limit_GiB=max_memory/2^30
    need <= max_memory || error("$name: estimated peak memory $(round(need / 2^30; digits=1)) GiB (working $(round(est.working / 2^30; digits=1)) GiB + retained $(round(est.persistent / 2^30; digits=1)) GiB, incl. GC slack) exceeds the limit of $(round(max_memory / 2^30; digits=1)) GiB. Compute the range in smaller chunks, use `eigenvectors=false`, or lower `nq`/`nlevels`/`taylor_degree`; pass `max_memory=Inf` to disable this check.")
    return nothing
end
