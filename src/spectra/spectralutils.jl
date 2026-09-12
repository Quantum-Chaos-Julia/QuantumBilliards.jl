######################################
#Needs total rework
######################################

function is_equal(x::T, dx::T, y::T, dy::T) :: Bool where {T<:Real}
    # Define the intervals
    x_lower=x-dx
    x_upper=x+dx
    y_lower=y-dy
    y_upper=y+dy
    # Check if the intervals overlap
    return max(x_lower,y_lower) <= min(x_upper,y_upper)
end


function match_wavenumbers(ks_l,ts_l,ks_r,ts_r)
    #vectors ks_l and_ks_r must be sorted
    i = j = 1 #counting index
    control = Vector{Bool}()#control bits
    ks = Vector{eltype(ks_l)}()#final wavenumbers
    ts = Vector{eltype(ts_l)}()#final tensions
    while i <= length(ks_l) && j <= length(ks_r)
        x, dx = ks_l[i], ts_l[i]
        y, dy = ks_r[j], ts_r[j]
        if  is_equal(x,dx,y,dy) #check equality with errorbars
            i += 1 
            j += 1
            if dx < dy
                push!(ks, x)
                push!(ts, dx)
                push!(control, true)
            else
                push!(ks, y)
                push!(ts, dy)
                push!(control, true)
            end
        elseif x < y
            i += 1
            push!(ks, x)
            push!(ts, dx)
            push!(control, false)
        else 
            j += 1
            push!(ks, y)
            push!(ts, dy)
            push!(control, false)
        end
    end
    return ks, ts, control 
end

function overlap_and_merge!(k_left, ten_left, k_right, ten_right, control_left, kl, kr; tol=1e-3)
    #check if intervals are empty 
    if isempty(k_left)
        #println("Left interval is empty.")
        append!(k_left, k_right)
        append!(ten_left, ten_right)
        append!(control_left, [false for i in 1:length(k_right)])
        return nothing #return short circuits further evaluation
    end

    #if right is empty just skip the mergeing
    if isempty(k_right)
        #println("Right interval is empty.")
        return nothing
    end
    
    #find overlaps in interval [k1,k2]
    idx_l = k_left .> (kl-tol) .&& k_left .< (kr+tol)
    idx_r = k_right .> (kl-tol) .&& k_right .< (kr+tol)
    
    ks_l,ts_l,ks_r,ts_r = k_left[idx_l], ten_left[idx_l], k_right[idx_r], ten_right[idx_r]
    #check if wavnumbers match in overlap interval
    ks, ts, control = match_wavenumbers(ks_l,ts_l,ks_r,ts_r)
    #println("left: $ks_l")
    #println("right: $ks_r")
    #println("overlaping: $ks")
    #i_l = idx_l[1]
    #i_r = idx_r[end]+1
    deleteat!(k_left, idx_l)
    append!(k_left, ks)
    deleteat!(ten_left, idx_l)
    append!(ten_left, ts)
    deleteat!(control_left, idx_l)
    append!(control_left, control)

    fl = findlast(idx_r)
    idx_last = isnothing(fl) ? 1 : fl + 1
    append!(k_left, k_right[idx_last:end])
    append!(ten_left, ten_right[idx_last:end])
    append!(control_left, [false for i in idx_last:length(k_right)])
end

"""
    compute_spectrum(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Real,Function}; tol=1e-4, multithreaded=true) → SpectralData

Computes every [`AcceleratedBasisSolver`](@ref) eigenvalue candidate and its
tension over the whole wavenumber range `[k1,k2]`, by sliding a window of
half-width `dk` from `k1` to `k2` and resolving overlaps between consecutive
windows with [`overlap_and_merge!`](@ref).

## Arguments
* `solver::AcceleratedBasisSolver`: The solver performing each windowed [`solve_spectrum`](@ref) diagonalization.
* `basis::AbsBasis`: The basis used to approximate the eigenstates.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `k1`: Lower wavenumber bound.
* `k2`: Upper wavenumber bound.
* `dk::Union{Real,Function}`: Window half-width, either a constant step or a function of the current wavenumber (e.g. sized adaptively via [`spectral_density`](@ref)).

## Keyword Arguments
* `tol::Real = 1e-4`: Tolerance forwarded to [`overlap_and_merge!`](@ref).
* `multithreaded::Bool = true`: Forwarded to each window's [`solve_spectrum`](@ref) call.

## Returns
* `data::SpectralData`: Every retained `(k,ten)` pair across all windows, sorted by `k`.
"""
function compute_spectrum(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Real,Function}; tol=1e-4, multithreaded=true)
    k0 = k1
    Δk = dk isa Function ? dk(k0) : dk
    #initial computation
    k_res, ten_res = solve_spectrum(solver, basis, billiard, k0, Δk+tol; multithreaded)
    control = [false for i in 1:length(k_res)]
    while k0 < k2
        k0_prev = k0
        k0 += Δk
        Δk = dk isa Function ? dk(k0) : dk
        k_new, ten_new = solve_spectrum(solver, basis, billiard, k0, Δk+tol; multithreaded)
        overlap_and_merge!(k_res, ten_res, k_new, ten_new, control, k0_prev, k0; tol=tol)

    end
    return _finalize_spectrum(k_res, ten_res, control)
end

"""
    SpectralData{T}

Stores a merged real spectrum (wavenumbers, tensions, and merge-control
flags), together with the retained wavenumber range.

## Attributes
* `k::Vector{T}`: Retained wavenumbers.
* `ten::Vector{T}`: Tension (or tension-like quality measure) associated with each `k`.
* `control::Vector{Bool}`: `true` for states selected while resolving an overlap between neighboring windows.
* `k_min::T`: Minimum retained wavenumber.
* `k_max::T`: Maximum retained wavenumber.
* `ten2::Union{Nothing,Vector{T}}`: Optional secondary tension/residual measure (e.g. [`BeynSolver`](@ref)'s normalized nonlinear residual `‖A(λ)φ‖`, alongside the primary `ten`); `nothing` when a solver reports only a single quality measure.
"""
struct SpectralData{T}
    k::Vector{T}
    ten::Vector{T}
    control::Vector{Bool}
    k_min::T
    k_max::T
    ten2::Union{Nothing,Vector{T}}
end

"""
    SpectralData(k::Vector{T}, ten::Vector{T}, control::Vector{Bool}; ten2::Union{Nothing,Vector{T}} = nothing) where {T<:Real} → SpectralData{T}

Constructs [`SpectralData`](@ref), caching the minimum and maximum retained
wavenumbers.

## Keyword Arguments
* `ten2::Union{Nothing,Vector{T}} = nothing`: Optional secondary tension/residual measure, see [`SpectralData`](@ref).
"""
function SpectralData(k::Vector{T}, ten::Vector{T}, control::Vector{Bool}; ten2::Union{Nothing,Vector{T}}=nothing) where {T<:Real}
    isempty(k) && throw(ArgumentError("Cannot construct SpectralData from an empty spectrum"))
    ten2===nothing || length(ten2)==length(k) || throw(DimensionMismatch("ten2 must have the same length as k"))
    return SpectralData(k, ten, control, minimum(k), maximum(k), ten2)
end

"""
    _finalize_spectrum(ks::Vector{T}, ts::Vector{T}, control::Vector{Bool}; ten2::Union{Nothing,Vector{T}} = nothing) where {T<:Real} → SpectralData{T}

Sorts `ks`/`ts`/`control` (and `ten2`, if given) by wavenumber and constructs
[`SpectralData`](@ref), throwing `ArgumentError` if `ks` is empty. Shared tail
used by every `compute_spectrum` method that constructs a `SpectralData`.

## Description
Safe to call from a single thread after a solver's own parallel region has
already written into disjoint preallocated slots (one window/segment per
index) — this is how e.g. [`BeynSolver`](@ref)/[`ExpandedBIMSolver`](@ref)
already operate, so no concurrent access to `SpectralData` construction
itself ever occurs; `_finalize_spectrum` does not add any new thread-safety
mechanism, it only removes duplicated sort/construct/empty-check tails.
"""
function _finalize_spectrum(ks::Vector{T}, ts::Vector{T}, control::Vector{Bool}; ten2::Union{Nothing,Vector{T}}=nothing) where {T<:Real}
    isempty(ks) && throw(ArgumentError("compute_spectrum found no candidates in the requested range"))
    p = sortperm(ks)
    return SpectralData(ks[p], ts[p], control[p]; ten2 = ten2===nothing ? nothing : ten2[p])
end

function merge_spectra(s1, s2; tol=1e-4)
    first = interval(s1.k_min-tol/2, s1.k_max+tol/2)
    second = interval(s2.k_min-tol/2, s2.k_max+tol/2)
    overlap = intersect_interval(first, second)  #this is the overlap interval
    
    idx_1 = [in_interval(k, overlap) for k in s1.k]
    idx_2 = [in_interval(k, overlap) for k in s2.k]

    ks1 = s1.k[idx_1]
    ts1 = s1.ten[idx_1]
    ks2 = s2.k[idx_2]
    ts2 = s2.ten[idx_2]

    ks_ov, ts_ov, cont_ov = match_wavenumbers(ks1,ts1,ks2,ts2)
    
    ks = append!(s1.k[.~idx_1],ks_ov)
    ts = append!(s1.ten[.~idx_1],ts_ov)
    control = append!(s1.control[.~idx_1],cont_ov)

    append!(ks, s2.k[.~idx_2])
    append!(ts, s2.ten[.~idx_2])
    append!(control, s2.control[.~idx_2])

    p = sortperm(ks) 
    return SpectralData(ks[p], ts[p], control[p])
end

"""
    compute_spectrum(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, N1::Int, N2::Int; N_expect::Real=1, tol=1e-4, multithreaded=true) → SpectralData

Computes every [`AcceleratedBasisSolver`](@ref) eigenvalue candidate and its
tension for states `N1` to `N2` of the billiard's Weyl-law state-counting
function.

## Description
The wavenumber range is obtained with [`k_range_for_states`](@ref), and the
sliding-window half-width is sized adaptively via [`spectral_density`](@ref)
so that each window expects `N_expect` states, before delegating to the
`k1,k2,dk` method.

## Arguments
* `solver::AcceleratedBasisSolver`: The solver performing each windowed [`solve_spectrum`](@ref) diagonalization.
* `basis::AbsBasis`: The basis used to approximate the eigenstates.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `N1::Int`, `N2::Int`: Range of states to cover.

## Keyword Arguments
* `N_expect::Real = 1`: Target number of states per sliding window.
* `tol::Real = 1e-4`, `multithreaded::Bool = true`: Forwarded to the `k1,k2,dk` method.

## Returns
* `data::SpectralData`: Every retained `(k,ten)` pair across `[k1,k2]`, sorted by `k`.
"""
function compute_spectrum(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, N1::Int, N2::Int; N_expect::Real=1, tol=1e-4, multithreaded=true)
    k1, k2 = k_range_for_states(billiard, N1, N2)
    dk = k -> N_expect / spectral_density(k, billiard)
    return compute_spectrum(solver, basis, billiard, k1, k2, dk; tol, multithreaded)
end

################################################################################
################### ACCELERATED-BIM-SPECIFIC SPECTRUM MERGING ################
################################################################################

# Score used by [`overlap_and_merge_ebim!`](@ref) to pick the best candidate
# within a cluster of near-duplicate roots: lower is better, favoring roots
# with small imaginary part (real physical roots) and small tension.
@inline function _ebim_scoring_logic(k::Number, t::Real)
    return log10(abs(imag(k))+eps(Float64))+log10(abs(Float64(t))+eps(Float64))
end

# Median of the positive consecutive gaps of `xs` in a `±w`-index window
# around `i`, used by [`overlap_and_merge_ebim!`](@ref) to set a
# spacing-adaptive clustering tolerance. Implemented by hand (rather than
# `using Statistics: median`) since `w` is small (a handful of points), to
# avoid adding a new package dependency for one call site.
function _local_gap(xs::AbstractVector{T}, i::Int; w::Int=4) where {T<:Real}
    i1 = max(1, i-w)
    i2 = min(length(xs), i+w)
    gaps = filter(>(zero(T)), diff(@view xs[i1:i2]))
    isempty(gaps) && return T(Inf)
    sort!(gaps)
    n = length(gaps)
    return isodd(n) ? gaps[(n+1)÷2] : (gaps[n÷2]+gaps[n÷2+1])/2
end

"""
    overlap_and_merge_ebim!(k_left::Vector{K}, ten_left::Vector{T}, k_right::Vector{K}, ten_right::Vector{T}, control_left::Vector{Bool}; tol::T = T(1e-5), spacing_frac::T = T(0.02), tolmax::T = T(5e-3), local_window::Int = 4) where {K<:Number,T<:Real} → nothing

Merges duplicate root candidates produced by overlapping [`ExpandedBIMSolver`](@ref)
local-correction windows, in place into `k_left`/`ten_left`/`control_left`.

## Description
Unlike [`overlap_and_merge!`](@ref) (which resolves overlap only between two
adjacent, explicitly-bounded windows using matched error bars), EBIM roots
come from many small, densely-spaced, unbounded local corrections, so nearby
roots are instead clustered directly by an adaptive tolerance based on the
local median spectral spacing (see [`_local_gap`](@ref)): consecutive sorted
roots whose gap is below `clamp(spacing_frac*local_gap, tol, tolmax)` are
merged into one cluster. Within each cluster, the candidate minimizing
`log10(|Im k|+eps) + log10(|tension|+eps)` (see [`_ebim_scoring_logic`](@ref))
is retained — i.e. the most-real, lowest-tension representative.

## Arguments
* `k_left::Vector{K}`: Accumulated roots, modified in place.
* `ten_left::Vector{T}`: Accumulated tensions, modified in place.
* `k_right::Vector{K}`: New roots to merge in.
* `ten_right::Vector{T}`: Tensions of the new roots.
* `control_left::Vector{Bool}`: Merge-control flags, modified in place (`true` marks a cluster that had more than one candidate).

## Keyword Arguments
* `tol::T = T(1e-5)`: Minimum clustering tolerance.
* `spacing_frac::T = T(0.02)`: Fraction of the local spectral spacing used for clustering.
* `tolmax::T = T(5e-3)`: Maximum clustering tolerance.
* `local_window::Int = 4`: Half-width (in points) of the local-spacing window.

## Returns
* `nothing`.
"""
function overlap_and_merge_ebim!(k_left::Vector{K}, ten_left::Vector{T}, k_right::Vector{K}, ten_right::Vector{T}, control_left::Vector{Bool}; tol::T=T(1e-5), spacing_frac::T=T(0.02), tolmax::T=T(5e-3), local_window::Int=4) where {K<:Number,T<:Real}
    isempty(k_right) && return nothing
    append!(k_left, k_right)
    append!(ten_left, ten_right)
    append!(control_left, fill(false, length(k_right)))
    p = sortperm(real.(k_left))
    k_all = k_left[p]
    ten_all = ten_left[p]
    ctrl_all = control_left[p]
    xs = real.(k_all)
    new_k = K[]
    new_t = T[]
    new_c = Bool[]
    i = 1
    while i<=length(k_all)
        j = i
        while j<length(k_all)
            gap = xs[j+1]-xs[j]
            lgap = min(_local_gap(xs, j; w=local_window), _local_gap(xs, j+1; w=local_window))
            local_tol = min(tolmax, max(tol, spacing_frac*T(lgap)))
            gap<=local_tol || break
            j += 1
        end
        block = i:j
        best = block[argmin(_ebim_scoring_logic.(k_all[block], ten_all[block]))]
        push!(new_k, k_all[best])
        push!(new_t, ten_all[best])
        push!(new_c, any(ctrl_all[block]) || length(block)>1)
        i = j+1
    end
    empty!(k_left); empty!(ten_left); empty!(control_left)
    append!(k_left, new_k); append!(ten_left, new_t); append!(control_left, new_c)
    return nothing
end

"""
    compute_spectrum(solver::BeynSolver, billiard::Bi, k1, k2; Rmax::Real = 1.0, multithreaded::Bool = true, multithreaded_windows::Bool = true) where {Bi<:AbsBilliard} → SpectralData

Computes every [`BeynSolver`](@ref) eigenvalue candidate and its tension over
the whole wavenumber range `[k1,k2]`, by covering the range with consecutive
Weyl-balanced contour windows and concatenating each window's already-filtered
[`solve`](@ref) result.

## Description
`[k1,k2]` is covered with [`plan_weyl_windows`](@ref) (using `solver.m` as the
target eigenvalue count per window and `Rmax` as the maximum contour radius),
converted to contour centers/radii with [`beyn_disks_from_windows`](@ref).
Boundary points for every window are generated up front (optionally
multithreaded across windows via `multithreaded_windows`, since each
window's own [`evaluate_points`](@ref) call is independent — unlike the
per-contour-node loop inside [`construct_matrices`](@ref), which is
deliberately left single-threaded to avoid nesting `Threads.@threads`
regions). Each window is then solved sequentially with [`solve`](@ref) (which
already performs contour-containment and residual filtering internally,
using `solver`'s own `svd_tol`/`res_tol`/`auto_discard_spurious` fields), and
every retained `(k,ten)` pair is concatenated across windows.

Since [`plan_weyl_windows`](@ref) produces contiguous, non-overlapping
windows by construction, no fuzzy overlap-resolution merge (unlike
[`overlap_and_merge!`](@ref)/[`overlap_and_merge_ebim!`](@ref)) is performed
here, matching `-develop`'s own `solve_spectrum_beyn` behavior; a genuine
root sitting exactly at a window boundary could in principle be found (or
missed) by at most one neighboring window, which is a negligible-probability
edge case for Weyl-balanced windows in practice.

## Arguments
* `solver::BeynSolver`: The [`BeynSolver`](@ref) whose `m`/`nq`/`r`/`svd_tol`/`res_tol`/`auto_discard_spurious` fields configure every window.
* `billiard::Bi`: The billiard whose boundary is discretized.
* `k1`: Lower wavenumber bound.
* `k2`: Upper wavenumber bound.

## Keyword Arguments
* `Rmax::Real = 1.0`: Maximum contour radius (caps the Weyl window width at `2*Rmax`).
* `multithreaded::Bool = true`: Enable multithreaded boundary-matrix construction within each window.
* `multithreaded_windows::Bool = true`: Enable multithreading across the independent per-window boundary-point evaluations.

## Returns
* `data::SpectralData{T}`: Every retained `(k,ten)` pair across all windows, sorted by `k` (`T` is the solver's own numeric type, from `_bim_numeric_type(solver)`); `control` is all `false` (no windows were merged), and `ten2` is `nothing`.
"""
function compute_spectrum(solver::BeynSolver, billiard::Bi, k1, k2; Rmax::Real=1.0, multithreaded::Bool=true, multithreaded_windows::Bool=true) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver)
    fundamental = solver.kernel.symmetry!==nothing
    intervals = plan_weyl_windows(billiard, T(k1), T(k2); m=solver.m, Rmax=Rmax, fundamental=fundamental)
    isempty(intervals) && throw(ArgumentError("Spectrum interval [$k1,$k2] contains no Weyl windows"))
    k0, R = beyn_disks_from_windows(intervals)
    nw = length(k0)
    pts_type = typeof(evaluate_points(solver, billiard, T(k1)))
    all_pts = Vector{pts_type}(undef, nw)
    @use_threads multithreading=multithreaded_windows for i in 1:nw
        all_pts[i] = evaluate_points(solver, billiard, real(k0[i]))
    end
    ks_win = Vector{Vector{T}}(undef, nw)
    tens_win = Vector{Vector{T}}(undef, nw)
    @inbounds for i in 1:nw
        ks_win[i], tens_win[i] = solve(solver, all_pts[i], k0[i], 2*R[i]; multithreaded)
    end
    ks = reduce(vcat, ks_win)
    tens = reduce(vcat, tens_win)
    control = fill(false, length(ks))
    return _finalize_spectrum(ks, tens, control)
end

"""
    compute_spectrum(solver::BeynSolver, billiard::Bi, N1::Int, N2::Int; kwargs...) where {Bi<:AbsBilliard} → SpectralData

Computes every [`BeynSolver`](@ref) eigenvalue candidate and its tension for
states `N1` to `N2` of the billiard's Weyl-law state-counting function.

## Description
The wavenumber range is obtained with [`k_range_for_states`](@ref) (using the
same fundamental/full-boundary convention as `solver.kernel.symmetry`, see
[`k_range_for_states`](@ref)), then delegated to the `k1,k2` method.

## Arguments
* `solver::BeynSolver`: The [`BeynSolver`](@ref) whose fields configure every window.
* `billiard::Bi`: The billiard whose boundary is discretized.
* `N1::Int`, `N2::Int`: Range of states to cover.

## Keyword Arguments
* `kwargs...`: Forwarded to the `k1,k2` method (`Rmax`, `multithreaded`, `multithreaded_windows`).

## Returns
* `data::SpectralData`: Every retained `(k,ten)` pair across `[k1,k2]`, sorted by `k`.
"""
function compute_spectrum(solver::BeynSolver, billiard::Bi, N1::Int, N2::Int; kwargs...) where {Bi<:AbsBilliard}
    fundamental = solver.kernel.symmetry!==nothing
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental)
    return compute_spectrum(solver, billiard, k1, k2; kwargs...)
end

"""
    compute_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k1, k2; dk::Function = (k -> 0.05*k^(-1/3)), tol = 1e-5, spacing_frac = 0.02, tolmax = 5e-3, local_window::Int = 4, seg_reuse_frac = 0.95, multithreaded::Bool = true) where {Bi<:AbsBilliard} → SpectralData

Computes every [`ExpandedBIMSolver`](@ref) locally-corrected root over the
whole wavenumber range `[k1,k2]`, by locally correcting a dense adaptive grid
of trial wavenumbers and merging the (densely overlapping) results with
[`overlap_and_merge_ebim!`](@ref).

## Description
An adaptive grid of trial wavenumbers is built by repeatedly stepping
`k += dk(k)` from `k1` until `k2` is reached (`dk` defaults to a Weyl-motivated
shrinking step `0.05*k^(-1/3)`, matching `-develop`). Consecutive trial
wavenumbers are grouped into segments that reuse the same
[`evaluate_points`](@ref) discretization as long as they stay within
`seg_reuse_frac` of the segment's starting `k` (boundary quadrature/geometry
caches change slowly with `k`, so this avoids needlessly re-deriving them for
every single local correction — the segment's discretization is sized for
its largest `k`, safely covering every smaller `k` in the segment, exactly as
[`k_sweep`](@ref) sizes a single discretization for `maximum(ks)`). Each trial
`k` is corrected independently with [`solve`](@ref) (`dk(k)` from the grid is
not otherwise used by the local expansion itself). Since neighboring trial
points routinely converge to the very same physical root, every corrected
root is folded into a running merged set with [`overlap_and_merge_ebim!`](@ref)
(tolerance-based clustering, not the window-boundary-based
[`overlap_and_merge!`](@ref) used for basis/Beyn sweeps), and finally
restricted back to `[k1,k2]`.

## Arguments
* `solver::ExpandedBIMSolver`: The [`ExpandedBIMSolver`](@ref) performing each local correction.
* `billiard::Bi`: The billiard whose boundary is discretized.
* `k1`: Lower wavenumber bound.
* `k2`: Upper wavenumber bound.

## Keyword Arguments
* `dk::Function = (k -> 0.05*k^(-1/3))`: Adaptive trial-wavenumber step as a function of `k`.
* `tol = 1e-5`, `spacing_frac = 0.02`, `tolmax = 5e-3`, `local_window::Int = 4`: Forwarded to [`overlap_and_merge_ebim!`](@ref).
* `seg_reuse_frac = 0.95`: Trial wavenumbers `k` reuse the current segment's boundary discretization while `k<=k_seg_start/seg_reuse_frac`.
* `multithreaded::Bool = true`: Enable multithreaded boundary-matrix construction within each local correction.

## Returns
* `data::SpectralData{T}`: Every merged corrected root inside `[k1,k2]`, sorted by `k` (`T` is the solver's own numeric type, from `_bim_numeric_type(solver)`); `ten2` is `nothing`.
"""
function compute_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k1, k2; dk::Function=(k->0.05*k^(-1/3)), tol=1e-5, spacing_frac=0.02, tolmax=5e-3, local_window::Int=4, seg_reuse_frac=0.95, multithreaded::Bool=true) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver)
    k1T, k2T = T(k1), T(k2)
    tolT, spacing_fracT, tolmaxT, seg_reuse_fracT = T(tol), T(spacing_frac), T(tolmax), T(seg_reuse_frac)
    k1T<k2T || throw(ArgumentError("require k1<k2"))
    0<seg_reuse_fracT<=1 || throw(ArgumentError("seg_reuse_frac must satisfy 0<seg_reuse_frac<=1"))
    ks_grid = T[]
    k = k1T
    while k<k2T
        Δk = T(dk(k))
        Δk>0 || throw(ArgumentError("dk(k) must be positive; received dk($k)=$Δk"))
        push!(ks_grid, k)
        k += Δk
    end
    n = length(ks_grid)
    n==0 && throw(ArgumentError("Spectrum interval [$k1,$k2] contains no correction points"))
    ks_corr = Vector{T}(undef, n)
    tens_corr = Vector{T}(undef, n)
    seg_first = 1
    pts = evaluate_points(solver, billiard, ks_grid[1])
    cheb_override = nothing
    if solver.use_chebyshev
        if solver.cheb_config.param_strategy===:manual
            cheb_override = solver.cheb_config
        elseif solver.cheb_config.param_strategy===:global
            cheb_override = _tune_ebim_cheb_config(solver, evaluate_points(solver, billiard, ks_grid[end]), ks_grid[end])
        elseif solver.cheb_config.param_strategy===:segment
            cheb_override = _tune_ebim_cheb_config(solver, pts, ks_grid[1])
        end
    end
    while seg_first<=n
        seg_last = seg_first
        while seg_last<n && ks_grid[seg_last+1]<=ks_grid[seg_first]/seg_reuse_fracT
            seg_last += 1
        end
        if seg_last!=seg_first
            pts = evaluate_points(solver, billiard, ks_grid[seg_last])
            solver.use_chebyshev && solver.cheb_config.param_strategy===:segment && (cheb_override = _tune_ebim_cheb_config(solver, pts, ks_grid[seg_last]))
        end
        @inbounds for i in seg_first:seg_last
            ks_corr[i], tens_corr[i] = solve(solver, pts, ks_grid[i]; multithreaded, cheb_override)
        end
        seg_first = seg_last+1
    end
    ks_all = T[]
    tens_all = T[]
    control = Bool[]
    @inbounds for i in 1:n
        overlap_and_merge_ebim!(ks_all, tens_all, T[ks_corr[i]], T[tens_corr[i]], control; tol=tolT, spacing_frac=spacing_fracT, tolmax=tolmaxT, local_window)
    end
    keep = k1T.<=ks_all.<=k2T
    ks_f = ks_all[keep]
    tens_f = tens_all[keep]
    control_f = control[keep]
    return _finalize_spectrum(ks_f, tens_f, control_f)
end

"""
    compute_spectrum(solver::ExpandedBIMSolver, billiard::Bi, N1::Int, N2::Int; kwargs...) where {Bi<:AbsBilliard} → SpectralData

Computes every [`ExpandedBIMSolver`](@ref) locally-corrected root for states
`N1` to `N2` of the billiard's Weyl-law state-counting function.

## Description
The wavenumber range is obtained with [`k_range_for_states`](@ref) (using the
same fundamental/full-boundary convention as `solver.kernel.symmetry`, see
[`k_range_for_states`](@ref)), then delegated to the `k1,k2` method.

## Arguments
* `solver::ExpandedBIMSolver`: The [`ExpandedBIMSolver`](@ref) performing each local correction.
* `billiard::Bi`: The billiard whose boundary is discretized.
* `N1::Int`, `N2::Int`: Range of states to cover.

## Keyword Arguments
* `kwargs...`: Forwarded to the `k1,k2` method (`dk`, `tol`, `spacing_frac`, `tolmax`, `local_window`, `seg_reuse_frac`, `multithreaded`).

## Returns
* `data::SpectralData{T}`: Every merged corrected root inside `[k1,k2]`, sorted by `k`.
"""
function compute_spectrum(solver::ExpandedBIMSolver, billiard::Bi, N1::Int, N2::Int; kwargs...) where {Bi<:AbsBilliard}
    fundamental = solver.kernel.symmetry!==nothing
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental)
    return compute_spectrum(solver, billiard, k1, k2; kwargs...)
end