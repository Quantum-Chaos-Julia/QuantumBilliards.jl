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
    return S(vals...)
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
    ComputationPlan{T<:Real,S,B,Ba,D}

Side-effect-free description of a `compute_spectrum` run that also holds every
input of that run, so `compute_spectrum(plan)` needs no further arguments.

## Fields
* `solver::S`: Solver with resolved parameters.
* `billiard::B`: Billiard geometry.
* `basis::Ba`: Basis for basis methods, `nothing` for boundary-integral methods.
* `k1::T`, `k2::T`: Requested interval.
* `dk::D`: Window half-width (constant or function of `k`) for the sweep methods, `nothing` for Beyn and CORK.
* `window_overlap::Float64`: Overlap of neighbouring sweep windows as a fraction of the window width.
* `max_memory::Float64`: Memory limit in bytes enforced by `compute_spectrum(plan)`.
* `expected_total_states::Float64`: Weyl estimate for the whole interval.
* `density_k1::Float64`, `density_k2::Float64`: Mean density of states at the interval ends.
* `windows::Vector{WindowPlan{T}}`: Window layout.
* `memory::MemoryEstimate`: Memory estimate.
* `nthreads::Int`: Julia threads available.
* `notes::Vector{String}`: Parameter adjustments and warnings.
"""
struct ComputationPlan{T<:Real,S,B<:AbsBilliard,Ba,D}
    solver::S
    billiard::B
    basis::Ba
    k1::T
    k2::T
    dk::D
    window_overlap::Float64
    max_memory::Float64
    expected_total_states::Float64
    density_k1::Float64
    density_k2::Float64
    windows::Vector{WindowPlan{T}}
    memory::MemoryEstimate
    nthreads::Int
    notes::Vector{String}
end

_plan_symmetry_label(::Nothing) = "none (full domain)"
_plan_symmetry_label(sym) = "fundamental domain, " * string(nameof(typeof(sym)))

_plan_range(x) = minimum(x) == maximum(x) ? string(minimum(x)) : string(minimum(x), " to ", maximum(x))

_plan_kernel(solver::AcceleratedBasisSolver) = nothing
_plan_kernel(solver) = string(nameof(typeof(solver.kernel)))

_plan_backend(::AcceleratedBasisSolver) = nothing
_plan_backend(s::BeynSolver) = s.use_chebyshev ? "Chebyshev interpolation, degrees $(s.cheb_config.M_h)/$(s.cheb_config.M_j) (Hankel/Bessel-J)" : s.use_taylor ? "Taylor-Horner, degree $(s.taylor_degree)" : "direct"
_plan_backend(s::CORKSolver) = "Chebyshev CORK, degree $(s.p)"
_plan_backend(s::ExpandedBIMSolver) = s.use_taylor ? "Taylor panels, degree $(s.taylor_degree)" : "direct derivatives"

# Stage and retained-result breakdown behind `plan.memory`, sized by the window with the largest working set.
function _plan_memory_items(plan::ComputationPlan)
    w = plan.windows; ns = Float64(plan.expected_total_states)
    if plan.solver isa AcceleratedBasisSolver
        big = argmax(x -> _working_bytes(plan.solver, plan.basis, Float64(x.dim), Float64(x.matrix_dim)), w)
        dim = maximum(x -> Float64(x.dim), w)
        return _memory_stages(plan.solver, plan.basis, Float64(big.dim), Float64(big.matrix_dim)), _memory_retained(plan.solver, plan.basis, ns, dim)
    end
    nw = length(w)
    big = argmax(x -> _working_bytes(plan.solver, Float64(x.dim), Float64(x.matrix_dim); nw), w)
    Nf = maximum(x -> Float64(x.dim), w); M = maximum(x -> Float64(x.matrix_dim), w)
    return _memory_stages(plan.solver, Float64(big.dim), Float64(big.matrix_dim); nw), _memory_retained(plan.solver, nw, ns, Nf, M)
end

function _write_plan_memory!(io::IO, plan::ComputationPlan)
    println(io, "Memory (upper-bound estimate)")
    if isempty(plan.windows)
        println(io, "  working         : ", _format_bytes(plan.memory.working))
        println(io, "  retained        : ", _format_bytes(plan.memory.persistent))
        return nothing
    end
    stages, retained = _plan_memory_items(plan)
    peak = _stage_peak(stages)
    width = maximum(length(first(x)) for x in Iterators.flatten((stages, retained)))
    println(io, "  working stages (alive at once; largest window)")
    for (label, bytes) in stages
        println(io, "    ", rpad(label, width), " : ", lpad(_format_bytes(bytes), 10), bytes == peak ? "  <- peak" : "")
    end
    if !isempty(retained)
        println(io, "  retained until the sweep ends")
        for (label, bytes) in retained
            println(io, "    ", rpad(label, width), " : ", lpad(_format_bytes(bytes), 10))
        end
    end
    println(io, "  working peak    : ", _format_bytes(plan.memory.working))
    println(io, "  retained        : ", _format_bytes(plan.memory.persistent))
    println(io, "  total           : ", _format_bytes(total_bytes(plan.memory)), "  (", _format_bytes(_MEMORY_SLACK * total_bytes(plan.memory)), " with GC slack)")
end

function Base.show(io::IO, ::MIME"text/plain", plan::ComputationPlan)
    w = plan.windows; n = length(w)
    is_basis = plan.solver isa AcceleratedBasisSolver
    println(io, "ComputationPlan for ", nameof(typeof(plan.solver)))
    println(io, "Solver")
    kernel = _plan_kernel(plan.solver)
    kernel === nothing || println(io, "  kernel          : ", kernel)
    plan.basis === nothing || println(io, "  basis           : ", nameof(typeof(plan.basis)))
    println(io, "  billiard        : ", nameof(typeof(plan.billiard)))
    backend = _plan_backend(plan.solver)
    backend === nothing || println(io, "  backend         : ", backend)
    is_basis || println(io, "  symmetry        : ", _plan_symmetry_label(plan.solver.kernel.symmetry))
    println(io, "Spectrum")
    println(io, "  interval        : [", plan.k1, ", ", plan.k2, "]")
    fundamental_only = is_basis ? plan.basis.symmetries !== nothing : plan.solver.kernel.symmetry !== nothing
    N1 = Float64(state_at_k(plan.k1, plan.billiard; fundamental_only)); N2 = Float64(state_at_k(plan.k2, plan.billiard; fundamental_only))
    println(io, "  state numbers   : ", round(Int, N1), " to ", round(Int, N2) )
    println(io, "  expected states : ", round(plan.expected_total_states; digits=1), "  (density ", round(plan.density_k1; digits=2), " to ", round(plan.density_k2; digits=2), ")")
    if n > 0
        s = sort!([x.expected_states for x in w]); d = [x.dim for x in w]; m = [x.matrix_dim for x in w]; h = [x.halfwidth for x in w]
        smed = isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
        println(io, "Windows")
        println(io, "  count           : ", n)
        println(io, "  states/window   : min ", round(first(s); digits=1), ", median ", round(smed; digits=1), ", max ", round(last(s); digits=1))
        any(>(0), h) && println(io, "  half-width      : ", round(minimum(h); sigdigits=3), " to ", round(maximum(h); sigdigits=3))
        plan.dk === nothing || println(io, "  overlap         : ", plan.window_overlap)
        println(io, "Discretization")
        if is_basis
            println(io, "  basis dim       : ", _plan_range(d), "  (square matrices, basis x basis)")
            println(io, "  boundary pts    : ", _plan_range(m))
        else
            println(io, "  boundary pts    : ", _plan_range(d), "  (full boundary)")
            println(io, "  matrix dim      : ", _plan_range(m), "  (after symmetry reduction)")
        end
    else
        println(io, "Windows")
        println(io, "  count           : 0")
    end
    _write_plan_memory!(io, plan)
    println(io, "Resources")
    println(io, "  julia threads   : ", plan.nthreads)
    print(io, "  blas threads    : ", LinearAlgebra.BLAS.get_num_threads(), " now, up to ", MAX_BLAS_THREADS, " in heavy kernels")
    if !isempty(plan.notes)
        print(io, "\nNotes")
        for note in plan.notes
            print(io, "\n  - ", note)
        end
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

function _assemble_plan(solver, billiard::AbsBilliard, k1, k2, windows::Vector{WindowPlan{T}}, memory::MemoryEstimate, max_memory::Real, notes::Vector{String}; fundamental_only::Bool, basis=nothing, dk=nothing, window_overlap::Real=0.5) where {T<:Real}
    need = _MEMORY_SLACK * total_bytes(memory)
    need > max_memory && push!(notes, "estimated peak memory $(round(need / 2^30; digits=1)) GiB exceeds max_memory $(round(max_memory / 2^30; digits=1)) GiB; compute_spectrum would throw")
    return ComputationPlan{T,typeof(solver),typeof(billiard),typeof(basis),typeof(dk)}(solver, billiard, basis, T(k1), T(k2), dk, Float64(window_overlap), Float64(max_memory), expected_states(billiard, k1, k2; fundamental_only), Float64(spectral_density(k1, billiard; fundamental_only)), Float64(spectral_density(k2, billiard; fundamental_only)), windows, memory, Threads.nthreads(), notes)
end

# Boundary node count and matrix dimension at `k`. For DLP/CFIE kernels with smooth or global corner grading the
# node count follows from the boundary length, so no boundary points are built; other kernels evaluate them.
function _bim_dims_fn(solver, billiard::AbsBilliard)
    kernel = solver.kernel
    return _bim_dims_fn(solver, kernel, kernel isa Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver} ? kernel.grading : nothing, billiard)
end

function _bim_dims_fn(solver, kernel, ::Any, billiard::AbsBilliard)
    return function (k)
        pts = evaluate_points(solver, billiard, k)
        return length(pts.xy), boundary_matrix_size(kernel, pts)
    end
end

function _bim_dims_fn(solver, kernel, grading::Union{SmoothPeriodicGrading,GlobalCornerGrading}, billiard::AbsBilliard)
    _require_no_holes(kernel, billiard)
    T = _bim_numeric_type(kernel)
    comp = full_boundary(billiard)
    isempty(comp) && error("Boundary cannot be empty.")
    _, _, Ltot = component_lengths(comp)
    graded = grading isa GlobalCornerGrading && !isempty(BilliardGeometry._component_corner_locations(T, comp))
    folded = Dict{Int,Int}()
    return function (k)
        N = _bim_node_count(kernel, Ltot, T(k); graded)
        kernel.symmetry === nothing && return N, N
        M = get!(folded, N) do
            fundamental_size(_fold_boundary(T, kernel.billiard, N, kernel.symmetry, kernel.sym_characters))
        end
        return N, M
    end
end

function _bim_window_plans(solver, billiard::AbsBilliard, intervals::Vector{Tuple{T,T}}) where {T<:Real}
    fundamental_only = solver.kernel.symmetry !== nothing
    dims = _bim_dims_fn(solver, billiard)
    return @with_kress_q_floor map(intervals) do (a, b)
        kc = (a + b) / 2
        N, M = dims(kc)
        WindowPlan{T}(kc, (b - a) / 2, expected_states(billiard, a, b; fundamental_only), N, M)
    end
end

function _bim_plan(solver, billiard::AbsBilliard, intervals::Vector{Tuple{T,T}}, k1, k2, max_memory::Real, notes::Vector{String}) where {T<:Real}
    fundamental_only = solver.kernel.symmetry !== nothing
    windows = _bim_window_plans(solver, billiard, intervals)
    memory = estimate_memory(solver, windows, expected_states(billiard, k1, k2; fundamental_only))
    return _assemble_plan(solver, billiard, k1, k2, windows, memory, max_memory, notes; fundamental_only)
end

function _ebim_window_plans(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2, dk::Function; window_overlap::Real=0.5)
    fundamental_only = solver.kernel.symmetry !== nothing
    centers, dks, _ = _ebim_windows(solver, billiard, k1, k2, dk; window_overlap)
    T = eltype(centers)
    dims = _bim_dims_fn(solver, billiard)
    return @with_kress_q_floor map(eachindex(centers)) do i
        N, M = dims(centers[i])
        WindowPlan{T}(centers[i], dks[i], Float64(weyl_window_count(billiard, centers[i] - dks[i], 2 * dks[i]; fundamental_only)), N, M)
    end
end

function _vs_window_plans(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Real,Function}; window_overlap::Real=0.5)
    fundamental_only = basis.symmetries !== nothing
    centers, widths = sweep_windows(k1, k2, dk; window_overlap)
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
        N, M = @with_kress_q_floor _bim_dims_fn(solver, billiard)(k)
        return [WindowPlan{T}(k, zero(T), 0.0, N, M)]
    end
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim, round(Int, L * k * solver.dim_scaling_factor / (2 * pi)))
    return [WindowPlan{T}(k, zero(T), 0.0, dim, length(evaluate_points(solver, billiard, k).w_vs))]
end

"""
    estimate_memory(solver, [basis,] windows::AbstractVector{<:WindowPlan}, expected_total_states) → MemoryEstimate
    estimate_memory(solver, [basis,] billiard, k1, k2[, dk]; window_overlap=0.5) → MemoryEstimate

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
* `dk`: Window half-width, for the sweep methods. Positional for [`VerginiSaracenoSolver`](@ref) (constant or function of `k`); keyword `dk` (function of `k`) for [`ExpandedBIMSolver`](@ref).
* `window_overlap::Real=0.5`: Overlap of neighbouring sweep windows as a fraction of the window width; ignored when `dk` is omitted.

## Returns
* `MemoryEstimate`: Peak working set and retained results in bytes.
"""
function estimate_memory(solver::Union{BeynSolver,CORKSolver,ExpandedBIMSolver}, windows::AbstractVector{<:WindowPlan}, expected_total_states::Real)
    working = mapreduce(w -> _working_bytes(solver, Float64(w.dim), Float64(w.matrix_dim); nw=length(windows)), max, windows; init=0.0)
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

function estimate_memory(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2; dk::Union{Nothing,Function}=nothing, window_overlap::Real=0.5)
    windows = dk === nothing ? _single_window(solver, billiard, k1, k2) : _ebim_window_plans(solver, billiard, k1, k2, dk; window_overlap)
    return estimate_memory(solver, windows, expected_states(billiard, k1, k2; fundamental_only=solver.kernel.symmetry !== nothing))
end

function estimate_memory(solver::VerginiSaracenoSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Nothing,Real,Function}=nothing; window_overlap::Real=0.5)
    windows = dk === nothing ? _single_window(solver, billiard, k1, k2; vs_basis=basis) : _vs_window_plans(solver, basis, billiard, k1, k2, dk; window_overlap)
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
function _ebim_windows(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2, dk::Function; window_overlap::Real=0.5)
    T = _bim_numeric_type(solver)
    fundamental_only = solver.kernel.symmetry !== nothing
    centers, dks = _sweep_windows(T(k1), T(k2), dk, window_overlap, true)
    nlevels = [max(1, ceil(Int, weyl_window_count(billiard, k - Δk, 2 * Δk; fundamental_only))) for (k, Δk) in zip(centers, dks)]
    return centers, dks, nlevels
end

"""
    plan_computation(solver::ExpandedBIMSolver, billiard, k1, k2; dk=(k -> 0.05*k^(-1/3)), window_overlap=0.5, max_memory=default_memory_limit(), tune=true) -> ComputationPlan
    plan_computation(solver::ExpandedBIMSolver, billiard, N1::Int, N2::Int; nlevels=nothing, kwargs...) -> ComputationPlan

Describe the `compute_spectrum(solver, billiard, k1, k2; dk, window_overlap)` run without any
eigen-solve. One window per expansion center, with half-width `dk(k)` and
neighbouring windows overlapping by the fraction `window_overlap` of the window
width; the expected states of a window are the Weyl count in `[k-dk, k+dk]`.
The centers are produced by the function `compute_spectrum` uses. The solver is
never modified, and `tune` has no effect.

## Arguments
* `solver::ExpandedBIMSolver`: Expanded boundary-integral eigensolver.
* `billiard::AbsBilliard`: Billiard geometry.
* `k1`, `k2`: Wavenumber interval, or `N1`, `N2` Weyl-law state indices.

## Keyword Arguments
* `dk::Function=(k -> 0.05*k^(-1/3))`: Window half-width.
* `nlevels::Union{Nothing,Real}=nothing`: `N1,N2` form only. If given, overrides `dk` with `k -> nlevels / (2*spectral_density(k, billiard))`, i.e. each window contains `nlevels` mean levels.
* `window_overlap::Real=0.5`: Overlap of neighbouring windows as a fraction of the window width, in `[0, 0.5]`.
* `max_memory::Real=default_memory_limit()`: Memory limit in bytes; exceeding it is reported in `plan.notes`.
* `tune::Bool=true`: Accepted for a uniform interface.

## Returns
* [`ComputationPlan`](@ref): Resolved windows, memory estimate and notes.
"""
function plan_computation(solver::ExpandedBIMSolver, billiard::AbsBilliard, k1, k2; dk::Function=(k -> 0.05 * k^(-1/3)), window_overlap::Real=0.5, tune::Bool=true, max_memory::Real=default_memory_limit())
    fundamental_only = solver.kernel.symmetry !== nothing
    windows = _ebim_window_plans(solver, billiard, k1, k2, dk; window_overlap)
    memory = estimate_memory(solver, windows, expected_states(billiard, k1, k2; fundamental_only))
    return _assemble_plan(solver, billiard, k1, k2, windows, memory, max_memory, String[]; fundamental_only, dk, window_overlap)
end

function plan_computation(solver::ExpandedBIMSolver, billiard::AbsBilliard, N1::Int, N2::Int; nlevels::Union{Nothing,Real}=nothing, kwargs...)
    fundamental_only = solver.kernel.symmetry !== nothing
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental_only)
    nlevels === nothing || return plan_computation(solver, billiard, k1, k2; kwargs..., dk = k -> nlevels / (2 * spectral_density(k, billiard; fundamental_only)))
    return plan_computation(solver, billiard, k1, k2; kwargs...)
end

"""
    plan_computation(solver::AcceleratedBasisSolver, basis, billiard, k1, k2, dk; window_overlap=0.5, max_memory=default_memory_limit(), tune=true) -> ComputationPlan
    plan_computation(solver::AcceleratedBasisSolver, basis, billiard, N1::Int, N2::Int; nlevels=6, kwargs...) -> ComputationPlan

Describe the `compute_spectrum(solver, basis, billiard, k1, k2, dk)` run without
any eigen-solve. One window per scaling center from [`sweep_windows`](@ref),
with the local half-width as `halfwidth`, neighbouring windows overlapping by
the fraction `window_overlap` of the window width, and the Weyl count in
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
* `dk`: Constant or function window half-width.

## Keyword Arguments
* `nlevels::Real=6`: Levels in the full window for the `N1,N2` form, i.e. the half-width is `nlevels/(2*spectral_density(k, billiard))`; a note warns when it is at least 1.
* `window_overlap::Real=0.5`: Overlap of neighbouring windows as a fraction of the window width, in `[0, 0.5]`.
* `max_memory::Real=default_memory_limit()`: Memory limit in bytes; exceeding it is reported in `plan.notes`.
* `tune::Bool=true`: Accepted for a uniform interface.

## Returns
* [`ComputationPlan`](@ref): Resolved windows, memory estimate and notes.
"""
function plan_computation(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Real,Function}; window_overlap::Real=0.5, tune::Bool=true, max_memory::Real=default_memory_limit())
    fundamental_only = basis.symmetries !== nothing
    windows = _vs_window_plans(solver, basis, billiard, k1, k2, dk; window_overlap)
    memory = estimate_memory(solver, basis, windows, expected_states(billiard, k1, k2; fundamental_only))
    return _assemble_plan(solver, billiard, k1, k2, windows, memory, max_memory, String[]; fundamental_only, basis, dk, window_overlap)
end

function plan_computation(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, N1::Int, N2::Int; nlevels::Real=6, kwargs...)
    fundamental_only = basis.symmetries !== nothing
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental_only)
    plan = plan_computation(solver, basis, billiard, k1, k2, k -> nlevels / (2 * spectral_density(k, billiard; fundamental_only)); kwargs...)
    nlevels >= 1 && push!(plan.notes, "nlevels=$nlevels >= 1 gives windows wider than one mean level spacing; the accelerated basis method loses accuracy, use nlevels below 1 unless k is large")
    return plan
end

"""
    plan_computation(solver::BeynSolver, billiard, k1, k2; nlevels=nothing, Rmax=nothing, tune=true, max_memory=default_memory_limit()) -> ComputationPlan
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
* `nlevels::Union{Nothing,Real}=nothing`: Overrides `solver.nlevels` (target levels per Weyl window, may be fractional); the plan stores a solver with that value. Each window is `min(Weyl width for nlevels levels, 2*solver.Rmax)` wide, so `nlevels` only changes the layout while the Weyl width is below `2*Rmax`.
* `Rmax::Union{Nothing,Real}=nothing`: Overrides `solver.Rmax` (maximum contour radius, half the maximum window width); the plan stores a solver with that value.
* `tune::Bool=true`: Validate the Taylor degree on the largest window.
* `max_memory::Real=default_memory_limit()`: Memory limit in bytes; exceeding it is reported in `plan.notes`.

## Returns
* [`ComputationPlan`](@ref): Resolved windows, memory estimate and notes.
"""
function plan_computation(solver::BeynSolver, billiard::AbsBilliard, k1, k2; nlevels::Union{Nothing,Real}=nothing, Rmax::Union{Nothing,Real}=nothing, tune::Bool=true, max_memory::Real=default_memory_limit())
    notes = String[]
    if nlevels !== nothing && nlevels != solver.nlevels
        push!(notes, "nlevels overridden from $(solver.nlevels) to $nlevels; the plan stores a solver with that value")
        solver = with_params(solver; nlevels=_bim_numeric_type(solver)(nlevels))
    end
    if Rmax !== nothing && Rmax != solver.Rmax
        push!(notes, "Rmax overridden from $(solver.Rmax) to $Rmax; the plan stores a solver with that value")
        solver = with_params(solver; Rmax=_bim_numeric_type(solver)(Rmax))
    end
    intervals = _beyn_intervals(solver, billiard, k1, k2)
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
    plan_computation(solver::CORKSolver, billiard, k1, k2; nlevels=nothing, Rmax=nothing, tune=true, max_memory=default_memory_limit()) -> ComputationPlan
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
* `nlevels::Union{Nothing,Real}=nothing`: Overrides `solver.nlevels` (target levels per Weyl window, may be fractional) and rescales `mstart` as the constructor default does; the plan stores a solver with these values. Each window is `min(Weyl width for nlevels levels, 2*solver.Rmax)` wide, so `nlevels` only changes the layout while the Weyl width is below `2*Rmax`.
* `Rmax::Union{Nothing,Real}=nothing`: Overrides `solver.Rmax` (maximum requested half-width of a window); the plan stores a solver with that value.
* `tune::Bool=true`: Accepted for a uniform interface; has no effect for CORK.
* `max_memory::Real=default_memory_limit()`: Memory limit in bytes; exceeding it is reported in `plan.notes`.

## Returns
* [`ComputationPlan`](@ref): Resolved windows, memory estimate and notes.
"""
function plan_computation(solver::CORKSolver, billiard::AbsBilliard, k1, k2; nlevels::Union{Nothing,Real}=nothing, Rmax::Union{Nothing,Real}=nothing, tune::Bool=true, max_memory::Real=default_memory_limit())
    notes = ["polynomial degree p is tuned at run time; the plan uses p=$(solver.p)"]
    if nlevels !== nothing && nlevels != solver.nlevels
        push!(notes, "nlevels overridden from $(solver.nlevels) to $nlevels and mstart rescaled; the plan stores a solver with these values")
        solver = with_params(solver; nlevels=_bim_numeric_type(solver)(nlevels), mstart=solver.block_size * ceil(Int, 5.0nlevels / solver.block_size))
    end
    if Rmax !== nothing && Rmax != solver.Rmax
        push!(notes, "Rmax overridden from $(solver.Rmax) to $Rmax; the plan stores a solver with that value")
        solver = with_params(solver; Rmax=_bim_numeric_type(solver)(Rmax))
    end
    intervals = _cork_intervals(solver, billiard, k1, k2)
    return _bim_plan(solver, billiard, intervals, k1, k2, max_memory, notes)
end

function plan_computation(solver::CORKSolver, billiard::AbsBilliard, N1::Int, N2::Int; kwargs...)
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental_only=solver.kernel.symmetry !== nothing)
    return plan_computation(solver, billiard, k1, k2; kwargs...)
end
