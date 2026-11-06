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

function _assemble_plan(solver, billiard::AbsBilliard, k1, k2, windows::Vector{WindowPlan{T}}, memory::MemoryEstimate, max_memory::Real, notes::Vector{String}; fundamental_only::Bool) where {T<:Real}
    need = _MEMORY_SLACK * total_bytes(memory)
    need > max_memory && push!(notes, "estimated peak memory $(round(need / 2^30; digits=1)) GiB exceeds max_memory $(round(max_memory / 2^30; digits=1)) GiB; compute_spectrum would throw")
    return ComputationPlan{T,typeof(solver)}(solver, T(k1), T(k2), expected_states(billiard, k1, k2; fundamental_only), Float64(spectral_density(k1, billiard; fundamental_only)), Float64(spectral_density(k2, billiard; fundamental_only)), windows, memory, Threads.nthreads(), notes)
end

function _bim_window_plans(solver, billiard::AbsBilliard, intervals::Vector{Tuple{T,T}}) where {T<:Real}
    fundamental_only = solver.kernel.symmetry !== nothing
    return @with_kress_q_floor map(intervals) do (a, b)
        kc = (a + b) / 2
        pts = evaluate_points(solver, billiard, kc)
        WindowPlan{T}(kc, (b - a) / 2, expected_states(billiard, a, b; fundamental_only), length(pts.xy), boundary_matrix_size(solver.kernel, pts))
    end
end

function _bim_plan(solver, billiard::AbsBilliard, intervals::Vector{Tuple{T,T}}, k1, k2, max_memory::Real, notes::Vector{String}) where {T<:Real}
    fundamental_only = solver.kernel.symmetry !== nothing
    windows = _bim_window_plans(solver, billiard, intervals)
    memory = estimate_memory(solver, windows, expected_states(billiard, k1, k2; fundamental_only))
    return _assemble_plan(solver, billiard, k1, k2, windows, memory, max_memory, notes; fundamental_only)
end

function _ebim_window_plans(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2, dk::Function)
    fundamental_only = solver.kernel.symmetry !== nothing
    centers, dks, _ = _ebim_windows(solver, billiard, k1, k2, dk)
    T = eltype(centers)
    return @with_kress_q_floor map(eachindex(centers)) do i
        pts = evaluate_points(solver, billiard, centers[i])
        WindowPlan{T}(centers[i], dks[i], Float64(weyl_window_count(billiard, centers[i] - dks[i], 2 * dks[i]; fundamental_only)), length(pts.xy), boundary_matrix_size(solver.kernel, pts))
    end
end

function _vs_window_plans(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Real,Function})
    fundamental_only = basis.symmetries !== nothing
    centers, widths = sweep_windows(k1, k2, dk)
    T = eltype(centers)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    vsdim(k) = max(solver.min_dim, round(Int, L * k * solver.dim_scaling_factor / (2 * pi)))
    kdim = copy(centers)
    if uses_taylor(basis)
        R = T(basis.taylor_radius)
        R > zero(T) || throw(ArgumentError("taylor_radius must be positive"))
        for (i1, i2) in zip(_vs_panels(centers, R)...)
            kdim[i1:i2] .= centers[i2]
        end
    end
    return map(eachindex(centers)) do i
        k0, Δk = centers[i], widths[i]
        WindowPlan{T}(k0, Δk, Float64(weyl_window_count(billiard, k0 - Δk, 2Δk; fundamental_only)), vsdim(kdim[i]), length(evaluate_points(solver, billiard, kdim[i]).w_vs))
    end
end

function _single_window(solver, billiard::AbsBilliard, k1, k2; vs_basis=nothing)
    T = promote_type(typeof(float(k1)), typeof(float(k2)))
    k = T(k2)
    if vs_basis === nothing
        pts = @with_kress_q_floor evaluate_points(solver, billiard, k)
        return [WindowPlan{T}(k, zero(T), 0.0, length(pts.xy), boundary_matrix_size(solver.kernel, pts))]
    end
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim, round(Int, L * k * solver.dim_scaling_factor / (2 * pi)))
    return [WindowPlan{T}(k, zero(T), 0.0, dim, length(evaluate_points(solver, billiard, k).w_vs))]
end

"""
    estimate_memory(solver, [basis,] windows::AbstractVector{<:WindowPlan}, expected_total_states) → MemoryEstimate
    estimate_memory(solver, [basis,] billiard, k1, k2[, dk]) → MemoryEstimate

Estimate the peak memory of a `compute_spectrum` run without running the solve.
The working set is the maximum over the windows, each sized by its own basis or
boundary dimension `dim` and matrix dimension `matrix_dim`; retained results are
sized by the total expected states and the largest dimensions. The estimate is an
order-of-magnitude upper bound, not an exact figure.

The `billiard` form builds the same window layout as [`plan_computation`](@ref).
For [`VerginiSaracenoSolver`](@ref) and [`ExpandedBIMSolver`](@ref), omitting `dk`
sizes a single window at `k2`.

## Arguments
* `solver`: A [`VerginiSaracenoSolver`](@ref) (with `basis`), [`BeynSolver`](@ref), [`ExpandedBIMSolver`](@ref) or [`CORKSolver`](@ref).
* `basis`: Basis, only for [`VerginiSaracenoSolver`](@ref).
* `windows`: Window layout, for example `plan.windows`.
* `expected_total_states`: Weyl estimate of the levels in the interval.
* `billiard`: Billiard geometry.
* `k1`, `k2`: Wavenumber interval.
* `dk`: Spacing of scaling or expansion centers, for the sweep methods.

## Returns
* `MemoryEstimate`: Peak working set and retained results in bytes.
"""
function estimate_memory(solver::Union{BeynSolver,CORKSolver,ExpandedBIMSolver}, windows::AbstractVector{<:WindowPlan}, expected_total_states::Real)
    working = mapreduce(w -> _working_bytes(solver, Float64(w.dim), Float64(w.matrix_dim)), max, windows; init=0.0)
    Nf = mapreduce(w -> Float64(w.dim), max, windows; init=0.0)
    M = mapreduce(w -> Float64(w.matrix_dim), max, windows; init=0.0)
    return MemoryEstimate(working, _persistent_bytes(solver, length(windows), Float64(expected_total_states), Nf, M))
end

function estimate_memory(solver::VerginiSaracenoSolver, basis::AbsBasis, windows::AbstractVector{<:WindowPlan}, expected_total_states::Real)
    working = mapreduce(w -> _working_bytes(solver, basis, Float64(w.dim), Float64(w.matrix_dim)), max, windows; init=0.0)
    dim = mapreduce(w -> Float64(w.dim), max, windows; init=0.0)
    return MemoryEstimate(working, _persistent_bytes(solver, basis, Float64(expected_total_states), dim))
end

function estimate_memory(solver::BeynSolver, billiard::AbsBilliard, k1, k2)
    windows = _bim_window_plans(solver, billiard, _beyn_intervals(solver, billiard, k1, k2))
    return estimate_memory(solver, windows, expected_states(billiard, k1, k2; fundamental_only=solver.kernel.symmetry !== nothing))
end

function estimate_memory(solver::CORKSolver, billiard::AbsBilliard, k1, k2)
    windows = _bim_window_plans(solver, billiard, _cork_intervals(solver, billiard, k1, k2))
    return estimate_memory(solver, windows, expected_states(billiard, k1, k2; fundamental_only=solver.kernel.symmetry !== nothing))
end

function estimate_memory(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2; dk::Union{Nothing,Function}=nothing)
    windows = dk === nothing ? _single_window(solver, billiard, k1, k2) : _ebim_window_plans(solver, billiard, k1, k2, dk)
    return estimate_memory(solver, windows, expected_states(billiard, k1, k2; fundamental_only=solver.kernel.symmetry !== nothing))
end

function estimate_memory(solver::VerginiSaracenoSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Nothing,Real,Function}=nothing)
    windows = dk === nothing ? _single_window(solver, billiard, k1, k2; vs_basis=basis) : _vs_window_plans(solver, basis, billiard, k1, k2, dk)
    return estimate_memory(solver, basis, windows, expected_states(billiard, k1, k2; fundamental_only=basis.symmetries !== nothing))
end

# Group consecutive scaling centers into panels spanning at most `2R`; returns first and last center index per panel.
function _vs_panels(centers::AbstractVector, R::Real)
    panel_start = Int[]; panel_stop = Int[]; i = 1
    while i <= length(centers)
        j = max(i, searchsortedlast(centers, centers[i] + 2R))
        push!(panel_start, i); push!(panel_stop, j)
        i = j + 1
    end
    return panel_start, panel_stop
end

# EBIM expansion centers, local half-widths and requested level counts.
function _ebim_windows(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2, dk::Function)
    T = _bim_numeric_type(solver); k1T = T(k1); k2T = T(k2)
    k1T < k2T || throw(ArgumentError("require k1 < k2"))
    fundamental_only = solver.kernel.symmetry !== nothing
    centers = T[]; dks = T[]; nlevels = Int[]
    k = k1T
    while k <= k2T
        Δk = T(dk(k))
        Δk > zero(T) || throw(ArgumentError("dk(k) must be positive"))
        push!(centers, k); push!(dks, Δk)
        push!(nlevels, max(1, ceil(Int, weyl_window_count(billiard, k - Δk, 2 * Δk; fundamental_only))))
        k += Δk
    end
    return centers, dks, nlevels
end

"""
    plan_computation(solver::ExpandedBIMSolver, billiard, k1, k2; dk=(k -> 0.05*k^(-1/3)), max_memory=default_memory_limit(), tune=true) -> ComputationPlan
    plan_computation(solver::ExpandedBIMSolver, billiard, N1::Int, N2::Int; kwargs...) -> ComputationPlan

Describe the `compute_spectrum(solver, billiard, k1, k2; dk)` run without any
eigen-solve. One window per expansion center, with half-width `dk(k)`; the
expected states of a window are the Weyl count in `[k-dk, k+dk]`. The centers
are produced by the function `compute_spectrum` uses. The solver is never
modified, and `tune` has no effect.

## Arguments
* `solver::ExpandedBIMSolver`: Expanded boundary-integral eigensolver.
* `billiard::AbsBilliard`: Billiard geometry.
* `k1`, `k2`: Wavenumber interval, or `N1`, `N2` Weyl-law state indices.

## Keyword Arguments
* `dk::Function=(k -> 0.05*k^(-1/3))`: Spacing between expansion centers.
* `max_memory::Real=default_memory_limit()`: Memory limit in bytes; exceeding it is reported in `plan.notes`.
* `tune::Bool=true`: Accepted for a uniform interface.

## Returns
* [`ComputationPlan`](@ref): Resolved windows, memory estimate and notes.
"""
function plan_computation(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2; dk::Function=(k -> 0.05 * k^(-1/3)), tune::Bool=true, max_memory::Real=default_memory_limit())
    fundamental_only = solver.kernel.symmetry !== nothing
    windows = _ebim_window_plans(solver, billiard, k1, k2, dk)
    memory = estimate_memory(solver, windows, expected_states(billiard, k1, k2; fundamental_only))
    return _assemble_plan(solver, billiard, k1, k2, windows, memory, max_memory, String[]; fundamental_only)
end

function plan_computation(solver::ExpandedBIMSolver, billiard::AbsBilliard, N1::Int, N2::Int; kwargs...)
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental_only=solver.kernel.symmetry !== nothing)
    return plan_computation(solver, billiard, k1, k2; kwargs...)
end

"""
    plan_computation(solver::AcceleratedBasisSolver, basis, billiard, k1, k2, dk; max_memory=default_memory_limit(), tune=true) -> ComputationPlan
    plan_computation(solver::AcceleratedBasisSolver, basis, billiard, N1::Int, N2::Int; nlevels=3, kwargs...) -> ComputationPlan

Describe the `compute_spectrum(solver, basis, billiard, k1, k2, dk)` run without
any eigen-solve. One window per scaling center from [`sweep_windows`](@ref),
with the local half-width as `halfwidth` and the Weyl count in
`[k-halfwidth, k+halfwidth]` as expected states. The basis dimension is
`max(min_dim, round(L*k*dim_scaling_factor/2π))`. For a basis with
`use_taylor`, centers are grouped into the same panels `compute_spectrum`
uses, and `dim` and `matrix_dim` are those of the largest center in the panel.
The solver is never modified, and `tune` has no effect. Adaptive window
halving driven by `accept_tol` cannot be predicted and is not included.

## Arguments
* `solver::AcceleratedBasisSolver`: Accelerated basis eigensolver.
* `basis::AbsBasis`: Basis.
* `billiard::AbsBilliard`: Billiard geometry.
* `k1`, `k2`: Wavenumber interval, or `N1`, `N2` Weyl-law state indices.
* `dk`: Constant or function spacing of scaling centers.

## Keyword Arguments
* `nlevels::Real=3`: Levels per window for the `N1,N2` form; a note warns when it is at least 1.
* `max_memory::Real=default_memory_limit()`: Memory limit in bytes; exceeding it is reported in `plan.notes`.
* `tune::Bool=true`: Accepted for a uniform interface.

## Returns
* [`ComputationPlan`](@ref): Resolved windows, memory estimate and notes.
"""
function plan_computation(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Real,Function}; tune::Bool=true, max_memory::Real=default_memory_limit())
    fundamental_only = basis.symmetries !== nothing
    windows = _vs_window_plans(solver, basis, billiard, k1, k2, dk)
    memory = estimate_memory(solver, basis, windows, expected_states(billiard, k1, k2; fundamental_only))
    return _assemble_plan(solver, billiard, k1, k2, windows, memory, max_memory, String[]; fundamental_only)
end

function plan_computation(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, N1::Int, N2::Int; nlevels::Real=3, kwargs...)
    fundamental_only = basis.symmetries !== nothing
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental_only)
    plan = plan_computation(solver, basis, billiard, k1, k2, k -> nlevels / spectral_density(k, billiard; fundamental_only); kwargs...)
    nlevels >= 1 && push!(plan.notes, "nlevels=$nlevels >= 1 gives windows wider than one mean level spacing; the accelerated basis method loses accuracy, use nlevels below 1 unless k is large")
    return plan
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
