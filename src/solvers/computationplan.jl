"""
    with_params(solver; kwargs...) -> typeof(solver)

Return a shallow copy of the immutable `solver` with the named fields replaced
by the given values. Unspecified fields, including the kernel and any basis,
are shared with the original, which is never modified.

## Arguments
* `solver`: Solver value whose type has a positional field constructor.

## Keyword Arguments
* `kwargs...`: Field names and replacement values.

## Returns
* A new solver of the same family with the replaced fields.
"""
function with_params(solver::S; kwargs...) where {S}
    names = fieldnames(S)
    for k in keys(kwargs)
        k in names || throw(ArgumentError("$(nameof(S)) has no field `$k`"))
    end
    vals = ntuple(i -> get(kwargs, names[i], getfield(solver, names[i])), length(names))
    return S.name.wrapper(vals...)
end

"""
    expected_states(billiard::AbsBilliard, k1, k2; fundamental_only::Bool=false) -> Float64

Weyl-law estimate of the number of levels with `k1 ≤ k ≤ k2`, clamped at zero.

## Arguments
* `billiard::AbsBilliard`: Billiard geometry.
* `k1`, `k2`: Interval bounds.

## Keyword Arguments
* `fundamental_only::Bool=false`: Count only the fundamental domain (symmetry-reduced).

## Returns
* `Float64`: Expected number of states in the interval.
"""
expected_states(billiard::AbsBilliard, k1, k2; fundamental_only::Bool=false) = max(0.0, Float64(state_at_k(k2, billiard; fundamental_only)) - Float64(state_at_k(k1, billiard; fundamental_only)))

"""
    WindowPlan{T<:Real}

Resolved description of one spectral window or panel.

## Fields
* `kcenter::T`: Window center.
* `halfwidth::T`: Window half-width.
* `expected_states::Float64`: Weyl estimate of the levels in the window.
* `dim::Int`: Basis size, or number of full boundary points for BIM solvers.
* `matrix_dim::Int`: Dimension of the matrix handled in the window.
"""
struct WindowPlan{T<:Real}
    kcenter::T
    halfwidth::T
    expected_states::Float64
    dim::Int
    matrix_dim::Int
end

"""
    ComputationPlan{T<:Real,S}

Side-effect-free description of a `compute_spectrum` run.

## Fields
* `solver::S`: Solver with resolved parameters.
* `k1::T`, `k2::T`: Requested interval.
* `expected_total_states::Float64`: Weyl estimate for the whole interval.
* `density_k1::Float64`, `density_k2::Float64`: Mean density of states at the interval ends.
* `windows::Vector{WindowPlan{T}}`: Window layout.
* `memory::MemoryEstimate`: Memory estimate.
* `nthreads::Int`: Julia threads available.
* `notes::Vector{String}`: Parameter adjustments and warnings.
"""
struct ComputationPlan{T<:Real,S}
    solver::S
    k1::T
    k2::T
    expected_total_states::Float64
    density_k1::Float64
    density_k2::Float64
    windows::Vector{WindowPlan{T}}
    memory::MemoryEstimate
    nthreads::Int
    notes::Vector{String}
end

function Base.show(io::IO, ::MIME"text/plain", plan::ComputationPlan)
    w = plan.windows; n = length(w)
    println(io, "ComputationPlan for ", nameof(typeof(plan.solver)))
    println(io, "  interval        : [", plan.k1, ", ", plan.k2, "]")
    println(io, "  expected states : ", round(plan.expected_total_states; digits=1), "  (density ", round(plan.density_k1; digits=2), " to ", round(plan.density_k2; digits=2), ")")
    println(io, "  windows         : ", n)
    if n > 0
        s = sort!([x.expected_states for x in w]); d = [x.dim for x in w]; m = [x.matrix_dim for x in w]
        smed = isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
        println(io, "  states/window   : min ", round(first(s); digits=1), ", median ", round(smed; digits=1), ", max ", round(last(s); digits=1))
        println(io, "  dim             : ", minimum(d), " to ", maximum(d))
        println(io, "  matrix dim      : ", minimum(m), " to ", maximum(m))
    end
    println(io, "  memory          : ", plan.memory)
    print(io, "  threads         : ", plan.nthreads)
    for note in plan.notes
        print(io, "\n  note            : ", note)
    end
end

Base.show(io::IO, plan::ComputationPlan) = print(io, "ComputationPlan(", nameof(typeof(plan.solver)), ", ", length(plan.windows), " windows, ", round(plan.expected_total_states; digits=1), " states)")

# Weyl-sized window layout shared by `compute_spectrum` and `plan_computation`.
function _beyn_intervals(solver::BeynSolver, billiard::AbsBilliard, k1, k2)
    T = _bim_numeric_type(solver)
    intervals = plan_weyl_windows(billiard, T(k1), T(k2); nlevels=solver.nlevels, Rmax=solver.Rmax, fundamental_only=solver.kernel.symmetry !== nothing)
    isempty(intervals) && throw(ArgumentError("Spectrum interval [$k1,$k2] contains no Weyl windows"))
    return intervals
end

function _cork_intervals(solver::CORKSolver, billiard::AbsBilliard, k1, k2)
    T = _bim_numeric_type(solver)
    intervals = plan_weyl_windows(billiard, T(k1), T(k2); nlevels=solver.nlevels, Rmax=solver.Rmax, fundamental_only=solver.kernel.symmetry !== nothing)
    isempty(intervals) && throw(ArgumentError("Spectrum interval [$k1,$k2] contains no Weyl windows"))
    if length(intervals) > 1
        ap, bp = intervals[end-1]; a, b = intervals[end]; dkprev = bp - ap
        b - a < dkprev && (intervals[end] = (a, a + dkprev))
    end
    return intervals
end

function _bim_plan(solver, billiard::AbsBilliard, intervals::Vector{Tuple{T,T}}, k1, k2, max_memory::Real, notes::Vector{String}) where {T<:Real}
    fundamental_only = solver.kernel.symmetry !== nothing
    windows = @with_kress_q_floor map(intervals) do (a, b)
        kc = (a + b) / 2
        pts = evaluate_points(solver, billiard, kc)
        WindowPlan{T}(kc, (b - a) / 2, expected_states(billiard, a, b; fundamental_only), length(pts.xy), boundary_matrix_size(solver.kernel, pts))
    end
    memory = estimate_memory(solver, billiard, k1, k2)
    need = _MEMORY_SLACK * total_bytes(memory)
    need > max_memory && push!(notes, "estimated peak memory $(round(need / 2^30; digits=1)) GiB exceeds max_memory $(round(max_memory / 2^30; digits=1)) GiB; compute_spectrum would throw")
    return ComputationPlan{T,typeof(solver)}(solver, T(k1), T(k2), expected_states(billiard, k1, k2; fundamental_only), Float64(spectral_density(k1, billiard; fundamental_only)), Float64(spectral_density(k2, billiard; fundamental_only)), windows, memory, Threads.nthreads(), notes)
end

"""
    plan_computation(solver::BeynSolver, billiard, k1, k2; tune=true, max_memory=default_memory_limit()) -> ComputationPlan
    plan_computation(solver::BeynSolver, billiard, N1::Int, N2::Int; kwargs...) -> ComputationPlan

Describe the `compute_spectrum(solver, billiard, k1, k2)` run without any
eigen-solve: Weyl-sized contour windows, expected states, boundary and matrix
dimensions per window, and a memory estimate. The window layout is produced by
the same function `compute_spectrum` uses. The solver is never modified.

With `tune=true` and `solver.use_taylor`, the Taylor degree is validated on the
largest contour only (one window of boundary-point evaluation and matrix
construction) and the plan stores a solver with that degree; `compute_spectrum`
re-tunes on all windows. The `N1,N2` form converts Weyl state indices to a
wavenumber interval.

## Arguments
* `solver::BeynSolver`: Beyn contour-integral eigensolver.
* `billiard::AbsBilliard`: Billiard geometry.
* `k1`, `k2`: Wavenumber interval, or `N1`, `N2` Weyl-law state indices.

## Keyword Arguments
* `tune::Bool=true`: Validate the Taylor degree on the largest window.
* `max_memory::Real=default_memory_limit()`: Memory limit in bytes; exceeding it is reported in `plan.notes`.

## Returns
* [`ComputationPlan`](@ref): Resolved windows, memory estimate and notes.
"""
function plan_computation(solver::BeynSolver, billiard::AbsBilliard, k1, k2; tune::Bool=true, max_memory::Real=default_memory_limit())
    intervals = _beyn_intervals(solver, billiard, k1, k2)
    notes = String[]
    if tune && solver.use_taylor
        k0, R = beyn_disks_from_windows(intervals); _, i = findmax(R)
        p = @with_kress_q_floor _tune_beyn_taylor_degree(solver, [evaluate_points(solver, billiard, real(k0[i]))], [k0[i]], [R[i]])
        p != solver.taylor_degree && push!(notes, "taylor_degree tuned from $(solver.taylor_degree) to $p on the largest window")
        solver = with_params(solver; taylor_degree=p)
    end
    return _bim_plan(solver, billiard, intervals, k1, k2, max_memory, notes)
end

function plan_computation(solver::BeynSolver, billiard::AbsBilliard, N1::Int, N2::Int; kwargs...)
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental_only=solver.kernel.symmetry !== nothing)
    return plan_computation(solver, billiard, k1, k2; kwargs...)
end

"""
    plan_computation(solver::CORKSolver, billiard, k1, k2; tune=true, max_memory=default_memory_limit()) -> ComputationPlan
    plan_computation(solver::CORKSolver, billiard, N1::Int, N2::Int; kwargs...) -> ComputationPlan

Describe the `compute_spectrum(solver, billiard, k1, k2)` run without any
eigen-solve: Weyl-sized windows (including the width adjustment of the last
window), expected states, boundary and matrix dimensions per window, and a
memory estimate. The solver is never modified.

The polynomial degree is tuned by `compute_spectrum` from matrix builds, which
the plan does not perform; the plan reports the solver as given and the memory
estimate uses `solver.p`.

## Arguments
* `solver::CORKSolver`: Chebyshev-CORK boundary-integral eigensolver.
* `billiard::AbsBilliard`: Billiard geometry.
* `k1`, `k2`: Wavenumber interval, or `N1`, `N2` Weyl-law state indices.

## Keyword Arguments
* `tune::Bool=true`: Accepted for a uniform interface; has no effect for CORK.
* `max_memory::Real=default_memory_limit()`: Memory limit in bytes; exceeding it is reported in `plan.notes`.

## Returns
* [`ComputationPlan`](@ref): Resolved windows, memory estimate and notes.
"""
function plan_computation(solver::CORKSolver, billiard::AbsBilliard, k1, k2; tune::Bool=true, max_memory::Real=default_memory_limit())
    intervals = _cork_intervals(solver, billiard, k1, k2)
    notes = ["polynomial degree p is tuned at run time; the plan uses p=$(solver.p)"]
    return _bim_plan(solver, billiard, intervals, k1, k2, max_memory, notes)
end

function plan_computation(solver::CORKSolver, billiard::AbsBilliard, N1::Int, N2::Int; kwargs...)
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental_only=solver.kernel.symmetry !== nothing)
    return plan_computation(solver, billiard, k1, k2; kwargs...)
end
