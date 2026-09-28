"""
    SpectralData{K,T,S}

Stores a computed spectrum together with its imaginary wavenumbers, quality
measures, merge-control flags and, optionally, the corresponding eigenstates.

## Attributes
* `k::Vector{K}`: Retained wavenumbers. Real or complex according to the solver.
* `im_k::Vector{T}`: Imaginary part of each retained wavenumber.
* `ten::Vector{T}`: Primary tension or residual associated with each wavenumber.
* `control::Vector{Bool}`: Merge or validation-control flag associated with each wavenumber.
* `k_min::T`: Minimum retained real wavenumber.
* `k_max::T`: Maximum retained real wavenumber.
* `ten2::Union{Nothing,Vector{T}}`: Optional secondary tension or residual measure.
* `states::S`: Corresponding eigenstates as a concrete `Vector{<:AbsState}`, or `nothing`.
"""
struct SpectralData{K<:Number,T<:Real,S<:Union{Nothing,Vector{<:AbsState}}}
    k::Vector{K}
    im_k::Vector{T}
    ten::Vector{T}
    control::Vector{Bool}
    k_min::T
    k_max::T
    ten2::Union{Nothing,Vector{T}}
    states::S
end

"""
    SpectralData(k::Vector{K}, ten::Vector{T}, control::Vector{Bool}; ten2::Union{Nothing,Vector{T}}=nothing, states::S=nothing) where {K<:Number,T<:Real,S<:Union{Nothing,Vector{<:AbsState}}}

Constructs [`SpectralData`](@ref), caching the imaginary parts and minimum and
maximum retained real wavenumbers.

## Arguments
* `k::Vector{K}`: Retained wavenumbers.
* `ten::Vector{T}`: Primary tension or residual associated with each wavenumber.
* `control::Vector{Bool}`: Merge or validation-control flags.

## Keyword Arguments
* `ten2::Union{Nothing,Vector{T}}=nothing`: Optional secondary tension or residual measure.
* `states::S=nothing`: Corresponding eigenstates, or `nothing`.

## Returns
* `data::SpectralData{K,T,S}`: Constructed spectral data.
"""
function SpectralData(k::Vector{K}, ten::Vector{T}, control::Vector{Bool}; ten2::Union{Nothing,Vector{T}}=nothing, states::S=nothing) where {K<:Number,T<:Real,S<:Union{Nothing,Vector{<:AbsState}}}
    im_k = T.(imag.(k)); imin = argmin(real.(k)); imax = argmax(real.(k))
    return SpectralData{K,T,S}(k, im_k, ten, control, real(k[imin]), real(k[imax]), ten2, states)
end

################################################################################
# INTERNAL HELPERS
################################################################################

# Sort all aligned spectral quantities by Re(k).
function _finalize_spectrum(ks::Vector{K}, ts::Vector{T}, control::Vector{Bool}; ten2::Union{Nothing,Vector{T}}=nothing, states::S=nothing) where {K<:Number,T<:Real,S<:Union{Nothing,Vector{<:AbsState}}}
    isempty(ks) && throw(ArgumentError("compute_spectrum found no candidates in the requested range"))
    p = sortperm(ks; by=real)
    return SpectralData(ks[p], ts[p], control[p]; ten2=ten2===nothing ? nothing : ten2[p], states=states===nothing ? nothing : states[p])
end

# Test whether two real-wavenumber uncertainty intervals overlap.
@inline is_equal(x::Number, dx::Real, y::Number, dy::Real) = max(real(x)-dx, real(y)-dy)<=min(real(x)+dx, real(y)+dy)

# Match two sorted spectra, retaining the lower-tension representative.
function match_wavenumbers(ks_l::Vector{K}, ts_l::Vector{T}, ks_r::Vector{K}, ts_r::Vector{T}; states_l::SL=nothing, states_r::SR=nothing) where {K<:Number,T<:Real,SL<:Union{Nothing,Vector{<:AbsState}},SR<:Union{Nothing,Vector{<:AbsState}}}
    i = j = 1; ks = K[]; ts = T[]; control = Bool[]; states = states_l===nothing ? nothing : similar(states_l, 0)
    while i<=length(ks_l) && j<=length(ks_r)
        x, dx = ks_l[i], ts_l[i]; y, dy = ks_r[j], ts_r[j]
        if is_equal(x, dx, y, dy)
            if dx<=dy
                push!(ks, x); push!(ts, dx); states!==nothing && push!(states, states_l[i])
            else
                push!(ks, y); push!(ts, dy); states!==nothing && push!(states, states_r[j])
            end
            push!(control, true); i += 1; j += 1
        elseif real(x)<real(y)
            push!(ks, x); push!(ts, dx); push!(control, false); states!==nothing && push!(states, states_l[i]); i += 1
        else
            push!(ks, y); push!(ts, dy); push!(control, false); states!==nothing && push!(states, states_r[j]); j += 1
        end
    end
    while i<=length(ks_l)
        push!(ks, ks_l[i]); push!(ts, ts_l[i]); push!(control, false); states!==nothing && push!(states, states_l[i]); i += 1
    end
    while j<=length(ks_r)
        push!(ks, ks_r[j]); push!(ts, ts_r[j]); push!(control, false); states!==nothing && push!(states, states_r[j]); j += 1
    end
    return ks, ts, control, states
end

# Merge a new overlapping spectral window into the accumulated spectrum.
function overlap_and_merge!(k_left::Vector{K}, ten_left::Vector{T}, k_right::Vector{K}, ten_right::Vector{T}, control_left::Vector{Bool}, kl, kr; tol::Real=1e-3, states_left::SL=nothing, states_right::SR=nothing) where {K<:Number,T<:Real,SL<:Union{Nothing,Vector{<:AbsState}},SR<:Union{Nothing,Vector{<:AbsState}}}
    if isempty(k_left)
        append!(k_left, k_right); append!(ten_left, ten_right); append!(control_left, fill(false, length(k_right)))
        states_left!==nothing && append!(states_left, states_right)
        return nothing
    end
    isempty(k_right) && return nothing
    idx_l = (real.(k_left).>kl-tol).&(real.(k_left).<kr+tol)
    idx_r = (real.(k_right).>kl-tol).&(real.(k_right).<kr+tol)
    sl = states_left===nothing ? nothing : states_left[idx_l]; sr = states_right===nothing ? nothing : states_right[idx_r]
    ks, ts, control, states = match_wavenumbers(k_left[idx_l], ten_left[idx_l], k_right[idx_r], ten_right[idx_r]; states_l=sl, states_r=sr)
    deleteat!(k_left, idx_l); append!(k_left, ks)
    deleteat!(ten_left, idx_l); append!(ten_left, ts)
    deleteat!(control_left, idx_l); append!(control_left, control)
    states_left!==nothing && (deleteat!(states_left, idx_l); append!(states_left, states))
    last_overlap = findlast(idx_r); first_tail = isnothing(last_overlap) ? 1 : last_overlap+1
    append!(k_left, k_right[first_tail:end]); append!(ten_left, ten_right[first_tail:end]); append!(control_left, fill(false, length(k_right)-first_tail+1))
    states_left!==nothing && append!(states_left, states_right[first_tail:end])
    return nothing
end

################################################################################
# ACCELERATED BASIS SOLVERS
################################################################################

"""
    compute_spectrum(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Real,Function}; tol::Real = 1e-4, multithreaded::Bool = true, show_progress::Bool = true) → data::SpectralData

Compute the spectrum of `billiard` over the wavenumber interval `[k1, k2]`
using an [`AcceleratedBasisSolver`](@ref).

## Description
The requested interval is divided into overlapping local spectral windows with
scaling centers determined by `dk`. At each center the local spectrum is
computed and overlapping results are merged with
[`overlap_and_merge!`](@ref).

For a [`CornerAdaptedFourierBessel`](@ref) basis with `use_taylor = true`,
consecutive scaling centers are grouped into panels covered by
`basis.taylor_radius`. A single [`CornerAdaptedTaylorCache`](@ref) is
constructed for each panel and reused by all scaling solves in that panel.

Before the spectral sweep, the configured Taylor degree and radius are checked
at the largest scaling wavenumber against direct basis evaluation. The cached
route is used only if both the basis matrix and its wavenumber derivative
satisfy `basis.taylor_tol`.

For each Taylor panel, the basis dimension and boundary discretization are
determined from the largest scaling center in the panel.

## Arguments
* `solver`: [`AcceleratedBasisSolver`](@ref) used to compute the local spectra.
* `basis`: Basis used to represent the eigenstates.
* `billiard`: Billiard whose spectrum is computed.
* `k1`: Lower bound of the requested wavenumber interval.
* `k2`: Upper bound of the requested wavenumber interval.
* `dk`: Spacing between consecutive scaling centers, either a positive constant or a function of the current center.

## Keyword Arguments
* `tol::Real = 1e-4`: Tolerance used when extending local windows and merging overlapping spectra.
* `multithreaded::Bool = true`: Whether basis evaluations are multithreaded.
* `show_progress::Bool = true`: Whether to display the progress bar.

## Returns
* `data`: [`SpectralData`](@ref) containing the merged spectrum and, when requested by the solver, the associated eigenstates.
"""
function compute_spectrum(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k1, k2, dk::Union{Real,Function}; tol::Real = 1e-4, multithreaded::Bool = true, show_progress::Bool = true)
    T = promote_type(typeof(k1), typeof(k2))
    k1T, k2T = T(k1), T(k2)
    k1T < k2T || throw(ArgumentError("require k1 < k2"))
    centers = T[]
    widths = T[]
    k0 = k1T
    while k0 < k2T
        Δk = T(dk isa Function ? dk(k0) : dk)
        Δk > 0 || throw(ArgumentError("dk must be positive; received dk($k0) = $Δk"))
        push!(centers, k0)
        push!(widths, Δk)
        k0 += Δk
    end
    ks = T[]
    ts = T[]
    control = Bool[]
    states = AbsState[]
    L = CompositeCurve(get_boundary_curves(billiard)).length
    use_taylor = (basis isa CornerAdaptedFourierBessel || basis isa RealPlaneWaves) && basis.use_taylor
    if use_taylor
        R = T(basis.taylor_radius)
        R > zero(T) || throw(ArgumentError("taylor_radius must be positive"))
        kcheck = centers[end]
        dimcheck = max(solver.min_dim, round(Int, L * kcheck * solver.dim_scaling_factor / (2 * pi)))
        basis_check = resize_basis(basis, billiard, dimcheck, kcheck)
        pts_check = evaluate_points(solver, billiard, kcheck)
        cache_check = basis_cache(basis_check, kcheck, pts_check.xy; multithreaded)
        keval = max(k1T, kcheck - R)
        B, dB = basis_and_dk_matrices(cache_check, keval; multithreaded)
        B_ref = basis_matrix(basis_check, keval, pts_check.xy; multithreaded)
        dB_ref = dk_matrix(basis_check, keval, pts_check.xy; multithreaded)
        err_B = norm(B - B_ref) / max(norm(B_ref), eps(T))
        err_dB = norm(dB - dB_ref) / max(norm(dB_ref), eps(T))
        max(err_B, err_dB) <= basis.taylor_tol || throw(ArgumentError("Taylor representation with degree $(basis.taylor_degree) and radius $(basis.taylor_radius) fails tolerance $(basis.taylor_tol) for expansion center k0 = $kcheck at evaluation point k = $keval: basis error = $err_B, derivative error = $err_dB"))
        panel_start = Int[]
        panel_stop = Int[]
        i = 1
        while i <= length(centers)
            j = max(i, searchsortedlast(centers, centers[i] + 2R))
            push!(panel_start, i)
            push!(panel_stop, j)
            i = j + 1
        end
        @maybe_showprogress show_progress for p in eachindex(panel_start)
            i1 = panel_start[p]
            i2 = panel_stop[p]
            kc = (centers[i1] + centers[i2]) / 2
            kmax = centers[i2]
            dim = max(solver.min_dim, round(Int, L * kmax * solver.dim_scaling_factor / (2 * pi)))
            basis_new = resize_basis(basis, billiard, dim, kmax)
            pts = evaluate_points(solver, billiard, kmax)
            cache = basis_cache(basis_new, kc, pts.xy; multithreaded)
            for i in i1:i2
                k0, Δk = centers[i], widths[i]
                if solver.eigenvectors
                    ki, ti, X = solve_vectors(solver, basis_new, pts, cache, k0, Δk + tol; multithreaded)
                    si = AbsState[BasisEigenstate(ki[j], k0, X[:, j], ti[j], solver, basis_new, billiard) for j in eachindex(ki)]
                    if i == 1
                        append!(ks, ki)
                        append!(ts, ti)
                        append!(control, fill(false, length(ki)))
                        append!(states, si)
                    else
                        overlap_and_merge!(ks, ts, ki, ti, control, centers[i - 1], k0; tol, states_left = states, states_right = si)
                    end
                else
                    ki, ti = solve(solver, basis_new, pts, cache, k0, Δk + tol; multithreaded)
                    if i == 1
                        append!(ks, ki)
                        append!(ts, ti)
                        append!(control, fill(false, length(ki)))
                    else
                        overlap_and_merge!(ks, ts, ki, ti, control, centers[i - 1], k0; tol)
                    end
                end
            end
        end
    else
        @maybe_showprogress show_progress for i in eachindex(centers)
            k0, Δk = centers[i], widths[i]
            dim = max(solver.min_dim, round(Int, L * k0 * solver.dim_scaling_factor / (2 * pi)))
            basis_new = resize_basis(basis, billiard, dim, k0)
            pts = evaluate_points(solver, billiard, k0)
            if solver.eigenvectors
                ki, ti, X = solve_vectors(solver, basis_new, pts, k0, Δk + tol; multithreaded)
                si = AbsState[BasisEigenstate(ki[j], k0, X[:, j], ti[j], solver, basis_new, billiard) for j in eachindex(ki)]
                if i == 1
                    append!(ks, ki)
                    append!(ts, ti)
                    append!(control, fill(false, length(ki)))
                    append!(states, si)
                else
                    overlap_and_merge!(ks, ts, ki, ti, control, centers[i - 1], k0; tol, states_left = states, states_right = si)
                end
            else
                ki, ti = solve(solver, basis_new, pts, k0, Δk + tol; multithreaded)
                if i == 1
                    append!(ks, ki)
                    append!(ts, ti)
                    append!(control, fill(false, length(ki)))
                else
                    overlap_and_merge!(ks, ts, ki, ti, control, centers[i - 1], k0; tol)
                end
            end
        end
    end
    keep = (k1T .<= real.(ks)) .& (real.(ks) .<= k2T)
    return _finalize_spectrum(ks[keep], ts[keep], control[keep]; states = solver.eigenvectors ? states[keep] : nothing)
end

"""
    compute_spectrum(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, N1::Int, N2::Int; N_expect::Real=3, tol::Real=1e-4, multithreaded::Bool=true, show_progress::Bool=true) → SpectralData

Computes the accelerated-basis spectrum corresponding approximately to
Weyl-law state indices `N1` through `N2`. The local spectral window spacing is
chosen from the Weyl spectral density.

## Arguments
* `solver::AcceleratedBasisSolver`: Accelerated basis eigensolver.
* `basis::AbsBasis`: Basis used to represent the eigenstates.
* `billiard::AbsBilliard`: Billiard geometry.
* `N1::Int`: First requested Weyl-law state index.
* `N2::Int`: Last requested Weyl-law state index.

## Keyword Arguments
* `N_expect::Real=3`: Expected number of eigenvalues per local spectral window.
* `tol::Real=1e-4`: Additional overlap tolerance used when solving and matching neighboring spectral windows.
* `multithreaded::Bool=true`: Enable multithreaded matrix construction and eigensolution where supported.
* `show_progress::Bool=true`: Display a progress bar during the spectral sweep.

## Returns
* `data::SpectralData`: Merged and sorted spectrum over the wavenumber range corresponding to state indices `N1:N2`, including stored [`BasisEigenstate`](@ref)s when `solver.eigenvectors=true`.
"""
function compute_spectrum(solver::AcceleratedBasisSolver, basis::AbsBasis, billiard::AbsBilliard, N1::Int, N2::Int; N_expect::Real=3, tol::Real=1e-4, multithreaded::Bool=true, show_progress::Bool=true)
    k1, k2 = k_range_for_states(billiard, N1, N2)
    dk = k -> N_expect/spectral_density(k, billiard)
    return compute_spectrum(solver, basis, billiard, k1, k2, dk; tol, multithreaded, show_progress)
end

################################################################################
# BEYN
################################################################################

"""
    compute_spectrum(solver::BeynSolver, billiard::Bi, k1, k2; multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard} → SpectralData

Computes the Beyn spectrum over the requested wavenumber interval using
Weyl-sized contour windows.

When `solver.eigenvectors=true`, each retained eigenvalue stores a
[`BIMEigenstate`](@ref) containing the layer density produced by the Beyn
solve. The density is expanded with [`symmetrize_layer_density`](@ref) before
storage, so both `state.vec` and `state.pts` refer to the complete physical
boundary.

When `solver.imag_k_check=true`, projected roots and layer densities are
retained temporarily and subjected to the global imaginary-wavenumber
residual check before the final spectrum is constructed.

## Arguments
* `solver::BeynSolver`: Beyn contour-integral eigensolver.
* `billiard::Bi`: Billiard geometry.
* `k1`: Lower bound of the requested real-wavenumber interval.
* `k2`: Upper bound of the requested real-wavenumber interval.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded boundary-integral matrix construction.
* `show_progress::Bool=true`: Display a progress bar during the contour sweep.

## Returns
* `data::SpectralData`: Sorted complex spectrum, including full-boundary [`BIMEigenstate`](@ref)s when `solver.eigenvectors=true`.
"""
function compute_spectrum(solver::BeynSolver, billiard::Bi, k1, k2; multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver); fundamental = solver.kernel.symmetry!==nothing
    intervals = plan_weyl_windows(billiard, T(k1), T(k2); m=solver.m, Rmax=solver.Rmax, fundamental)
    isempty(intervals) && throw(ArgumentError("Spectrum interval [$k1,$k2] contains no Weyl windows"))
    k0, R = beyn_disks_from_windows(intervals); nw = length(k0)
    pts_type = typeof(evaluate_points(solver, billiard, T(k1))); pts = Vector{pts_type}(undef, nw)
    for i in 1:nw
        pts[i] = evaluate_points(solver, billiard, real(k0[i]))
    end
    if solver.use_taylor
        _tune_beyn_taylor_degree!(solver, pts, k0, R; multithreaded)
    end
    cheb_config = solver.cheb_config
    if (solver.use_chebyshev || solver.imag_k_check) && solver.cheb_config.param_strategy!==:manual # always do cheb config since imag_k_check requires it anyway even if taylor
        cheb_config = tune_cheb_config(solver.kernel, pts[end], real(k0[end])+R[end], solver.cheb_config)
    end
    if !solver.imag_k_check
        ks_win = Vector{Vector{Complex{T}}}(undef, nw); ts_win = Vector{Vector{T}}(undef, nw)
        if solver.eigenvectors
            X_win = Vector{Matrix{Complex{T}}}(undef, nw)
            @maybe_showprogress show_progress for i in 1:nw
                ks_win[i], ts_win[i], X_win[i] = solve_vectors(solver, pts[i], k0[i], 2R[i]; multithreaded, cheb_config)
            end
            states_win = [[BIMEigenstate(ks_win[i][j], symmetrize_layer_density(solver.kernel, X_win[i][:,j], pts[i], billiard), ts_win[i][j], solver.kernel, billiard, pts[i]) for j in eachindex(ks_win[i])] for i in 1:nw]
            ks = reduce(vcat, ks_win); ts = reduce(vcat, ts_win); states = reduce(vcat, states_win)
            keep = (T(k1).<=real.(ks)).&(real.(ks).<=T(k2))
            return _finalize_spectrum(ks[keep], ts[keep], fill(false, count(keep)); states=states[keep])
        end
        @maybe_showprogress show_progress for i in 1:nw
            ks_win[i], ts_win[i] = solve(solver, pts[i], k0[i], 2R[i]; multithreaded, cheb_config)
        end
        ks = reduce(vcat, ks_win); ts = reduce(vcat, ts_win); keep = (T(k1).<=real.(ks)).&(real.(ks).<=T(k2))
        return _finalize_spectrum(ks[keep], ts[keep], fill(false, count(keep)))
    end
    ks_win = Vector{Vector{Complex{T}}}(undef, nw); X_win = Vector{Matrix{Complex{T}}}(undef, nw)
    @maybe_showprogress show_progress for i in 1:nw
        ks_win[i], X_win[i] = _beyn_projected_solve(solver, pts[i], k0[i], 2R[i]; multithreaded, cheb_config)
    end
    idx_keep, residuals = _beyn_imag_k_check(solver, ks_win, X_win, pts; multithreaded, cheb_config)
    n = sum(length, idx_keep); ks = Vector{Complex{T}}(undef, n); ts = Vector{T}(undef, n); control = Vector{Bool}(undef, n)
    if solver.eigenvectors
        entries = Tuple{Int,Int}[]; p = 0
        for i in 1:nw, q in eachindex(idx_keep[i])
            j = idx_keep[i][q]; p += 1
            ks[p] = ks_win[i][j]; control[p] = isnan(residuals[i][q]); ts[p] = control[p] ? abs(imag(ks[p])) : residuals[i][q]
            push!(entries, (i,j))
        end
        states = [BIMEigenstate(ks[q], symmetrize_layer_density(solver.kernel, X_win[i][:,j], pts[i], billiard), ts[q], solver.kernel, billiard, pts[i]) for (q,(i,j)) in enumerate(entries)]
        keep = (T(k1).<=real.(ks)).&(real.(ks).<=T(k2))
        return _finalize_spectrum(ks[keep], ts[keep], control[keep]; states=states[keep])
    end
    p = 0
    for i in 1:nw, q in eachindex(idx_keep[i])
        j = idx_keep[i][q]; p += 1
        ks[p] = ks_win[i][j]; control[p] = isnan(residuals[i][q]); ts[p] = control[p] ? abs(imag(ks[p])) : residuals[i][q]
    end
    keep = (T(k1).<=real.(ks)).&(real.(ks).<=T(k2))
    return _finalize_spectrum(ks[keep], ts[keep], control[keep])
end

"""
    compute_spectrum(solver::BeynSolver, billiard::Bi, N1::Int, N2::Int; multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard} → SpectralData

Computes the Beyn spectrum over the wavenumber interval corresponding
approximately to Weyl-law state indices `N1` through `N2`.

## Arguments
* `solver::BeynSolver`: Beyn contour-integral eigensolver.
* `billiard::Bi`: Billiard geometry.
* `N1::Int`: First requested Weyl-law state index.
* `N2::Int`: Last requested Weyl-law state index.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded boundary-integral matrix construction.
* `show_progress::Bool=true`: Display a progress bar during the contour sweep.

## Returns
* `data::SpectralData`: Sorted complex spectrum over the corresponding Weyl-law wavenumber range, including full-boundary [`BIMEigenstate`](@ref)s when `solver.eigenvectors=true`.
"""
function compute_spectrum(solver::BeynSolver, billiard::Bi, N1::Int, N2::Int; multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard}
    fundamental = solver.kernel.symmetry!==nothing
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental)
    return compute_spectrum(solver, billiard, k1, k2; multithreaded, show_progress)
end

################################################################################
# EXPANDED BIM
################################################################################

# Score competing EBIM roots by imaginary part and correction tension.
@inline _ebim_scoring_logic(k::Number, t::T) where {T<:Real} = log10(abs(imag(k))+eps(T))+log10(abs(t)+eps(T))

# Compute the local median real-wavenumber spacing around index i.
function _local_gap(xs::AbstractVector{T}, i::Int; w::Int=4) where {T<:Real}
    i1 = max(1, i-w); i2 = min(length(xs), i+w); gaps = filter(>(zero(T)), diff(@view xs[i1:i2]))
    isempty(gaps) && return T(Inf)
    sort!(gaps); n = length(gaps)
    return isodd(n) ? gaps[(n+1)÷2] : (gaps[n÷2]+gaps[n÷2+1])/2
end

# Cluster and merge duplicate EBIM corrections.
function overlap_and_merge_ebim!(k_left::Vector{K}, ten_left::Vector{T}, k_right::Vector{K}, ten_right::Vector{T}, control_left::Vector{Bool}; tol::T=T(1e-5), spacing_frac::T=T(0.02), tolmax::T=T(5e-3), local_window::Int=4, states_left::SL=nothing, states_right::SR=nothing) where {K<:Number,T<:Real,SL<:Union{Nothing,Vector{<:AbsState}},SR<:Union{Nothing,Vector{<:AbsState}}}
    isempty(k_right) && return nothing
    append!(k_left, k_right); append!(ten_left, ten_right); append!(control_left, fill(false, length(k_right))); states_left!==nothing && append!(states_left, states_right)
    p = sortperm(k_left; by=real); k_all = k_left[p]; ten_all = ten_left[p]; ctrl_all = control_left[p]; states_all = states_left===nothing ? nothing : states_left[p]
    xs = real.(k_all); new_k = K[]; new_t = T[]; new_c = Bool[]; new_states = states_left===nothing ? nothing : similar(states_left, 0); i = 1
    while i<=length(k_all)
        j = i
        while j<length(k_all)
            lgap = min(_local_gap(xs, j; w=local_window), _local_gap(xs, j+1; w=local_window))
            xs[j+1]-xs[j]<=min(tolmax, max(tol, spacing_frac*T(lgap))) || break
            j += 1
        end
        block = i:j; best = block[argmin(_ebim_scoring_logic.(k_all[block], ten_all[block]))]
        push!(new_k, k_all[best]); push!(new_t, ten_all[best]); push!(new_c, any(ctrl_all[block]) || length(block)>1); new_states!==nothing && push!(new_states, states_all[best])
        i = j+1
    end
    empty!(k_left); append!(k_left, new_k); empty!(ten_left); append!(ten_left, new_t); empty!(control_left); append!(control_left, new_c)
    states_left!==nothing && (empty!(states_left); append!(states_left, new_states))
    return nothing
end

@inline function weyl_window_count(billiard::Bi, k::T, dk::T; fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    A = fundamental ? fundamental_area(billiard) : area(billiard)
    return A*((k+dk)^2-k^2)/(T(4pi))
end

"""
    compute_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k1, k2; dk::Function=(k -> 0.05), tol=1e-5, spacing_frac=0.02, tolmax=5e-3, local_window::Int=4, multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard}

Compute the EBIM spectrum in `[k1,k2]`.

Local EBIM expansion centers are separated according to `dk(k)`. At each
center, the number of requested local eigenvalues is estimated from the Weyl
law. Candidates are retained within the corresponding local spectral window
and duplicate estimates from overlapping windows are merged.

If `solver.use_taylor` is enabled, consecutive EBIM centers spanning at most
`2solver.taylor_radius` are grouped into Taylor panels. One analytic Fredholm
Taylor expansion is constructed at the midpoint of each panel and reused for
all EBIM centers in that panel. Otherwise, `A`, `A'`, and `A''` are assembled
directly at every expansion center.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `billiard::Bi`: Billiard geometry.
* `k1`: Lower wavenumber bound.
* `k2`: Upper wavenumber bound.

## Keyword Arguments
* `dk::Function = (k -> 0.05)`: Local EBIM spectral half-width and center spacing.
* `tol = 1e-5`: Absolute tolerance used when merging duplicate eigenvalues.
* `spacing_frac = 0.02`: Local-spacing fraction used by duplicate detection.
* `tolmax = 5e-3`: Maximum duplicate-merging tolerance.
* `local_window::Int = 4`: Number of neighboring spacings used by the local duplicate criterion.
* `multithreaded::Bool = true`: Enable multithreaded matrix construction.
* `show_progress::Bool = true`: Display the spectrum-sweep progress indicator.

## Returns
* `data::SpectralData`: Merged EBIM spectrum and associated correction magnitudes.
"""
function compute_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k1, k2; dk::Function = (k -> 0.05), tol = 1e-5, spacing_frac = 0.02, tolmax = 5e-3, local_window::Int = 4, multithreaded::Bool = true, show_progress::Bool = true) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver)
    k1T = T(k1)
    k2T = T(k2)
    k1T < k2T || throw(ArgumentError("require k1 < k2"))
    fundamental = solver.kernel.symmetry !== nothing
    centers = T[]
    dks = T[]
    nlevels = Int[]
    k = k1T
    while k <= k2T
        Δk = T(dk(k))
        Δk > zero(T) || throw(ArgumentError("dk(k) must be positive"))
        push!(centers, k)
        push!(dks, Δk)
        nweyl = ceil(Int, weyl_window_count(billiard, k - Δk, 2 * Δk; fundamental))
        push!(nlevels, max(1, nweyl))
        k += Δk
    end
    isempty(centers) && return SpectralData(T[], T[])
    ks = Complex{T}[]
    ts = T[]
    if !solver.use_taylor
        @maybe_showprogress show_progress for i = eachindex(centers)
            ki = centers[i]
            pts = evaluate_points(solver, billiard, ki)
            ksi, tsi = solve(solver, pts, ki, nlevels[i]; multithreaded)
            @inbounds for q = eachindex(ksi)
                if abs(real(ksi[q]) - ki) <= dks[i]
                    push!(ks, ksi[q])
                    push!(ts, tsi[q])
                end
            end
        end
    else
        T === Float64 || throw(ArgumentError("analytic EBIM Taylor acceleration currently requires Float64"))
        R = T(solver.taylor_radius)
        p = solver.taylor_degree
        np = length(centers)
        panel_start = 1
        progress = show_progress ? Progress(np; desc = "EBIM spectrum: ") : nothing
        while panel_start <= np
            panel_stop = panel_start
            panel_limit = centers[panel_start] + 2R
            while panel_stop < np && centers[panel_stop + 1] <= panel_limit
                panel_stop += 1
            end
            kc = T(0.5) * (centers[panel_start] + centers[panel_stop])
            kmax_panel = min(k2T, kc + R)
            pts = evaluate_points(solver, billiard, kmax_panel)
            cache = EBIMTaylorCache(solver.kernel, pts, Float64(kc), p; multithreaded)
            N = size(cache.coeffs, 1)
            A = Matrix{ComplexF64}(undef, N, N)
            dA = similar(A)
            ddA = similar(A)
            @inbounds for i = panel_start:panel_stop
                ki = centers[i]
                _construct_matrices!(solver, A, dA, ddA, cache, ki)
                ksi, tsi = _solve(solver, A, dA, ddA, ki, nlevels[i])
                for q = eachindex(ksi)
                    if abs(real(ksi[q]) - ki) <= dks[i]
                        push!(ks, ksi[q])
                        push!(ts, tsi[q])
                    end
                end
            end
            show_progress && update!(progress, panel_stop)
            panel_start = panel_stop + 1
        end
    end
    isempty(ks) && return SpectralData(T[], T[])
    order = sortperm(real.(ks))
    ks = ks[order]
    ts = ts[order]
    kreal = T.(real.(ks))
    tvals = T.(ts)
    keep = (kreal .>= k1T) .& (kreal .<= k2T)
    kreal = kreal[keep]
    tvals = tvals[keep]
    isempty(kreal) && return SpectralData(T[], T[])
    kout = T[]
    tout = T[]
    overlap_and_merge_ebim!(kout, tout, kreal, tvals; tol = tol, spacing_frac = spacing_frac, tolmax = tolmax, local_window = local_window)
    return SpectralData(kout, tout)
end

"""
    compute_spectrum(solver::ExpandedBIMSolver, billiard::Bi, N1::Int, N2::Int; dk::Function=(k->0.05*k^(-1/3)), tol=1e-5, spacing_frac=0.02, tolmax=5e-3, local_window::Int=4, seg_reuse_frac=0.95, multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard} → SpectralData

Computes the expanded-BIM spectrum over the wavenumber interval corresponding
approximately to Weyl-law state indices `N1` through `N2`.

## Arguments
* `solver::ExpandedBIMSolver`: Expanded boundary-integral eigensolver.
* `billiard::Bi`: Billiard geometry.
* `N1::Int`: First requested Weyl-law state index.
* `N2::Int`: Last requested Weyl-law state index.

## Keyword Arguments
* `dk::Function=(k->0.05*k^(-1/3))`: Adaptive spacing between expansion wavenumbers.
* `tol=1e-5`: Minimum absolute separation used when clustering corrected roots.
* `spacing_frac=0.02`: Fraction of the local spectral spacing used as an adaptive clustering threshold.
* `tolmax=5e-3`: Maximum clustering threshold.
* `local_window::Int=4`: Number of neighboring spacings on each side used to estimate the local spectral spacing.
* `seg_reuse_frac=0.95`: Controls how far a boundary discretization may be reused before it is regenerated.
* `multithreaded::Bool=true`: Enable multithreaded boundary-integral matrix construction.
* `show_progress::Bool=true`: Display a progress bar during the correction sweep.

## Returns
* `data::SpectralData`: Merged and sorted complex spectrum over the corresponding Weyl-law wavenumber range.
"""
function compute_spectrum(solver::ExpandedBIMSolver, billiard::Bi, N1::Int, N2::Int; dk::Function=(k->0.05*k^(-1/3)), tol=1e-5, spacing_frac=0.02, tolmax=5e-3, local_window::Int=4, seg_reuse_frac=0.95, multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard}
    fundamental = solver.kernel.symmetry!==nothing
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental)
    return compute_spectrum(solver, billiard, k1, k2; dk, tol, spacing_frac, tolmax, local_window, seg_reuse_frac, multithreaded, show_progress)
end

################################################################################
# CORK
################################################################################

"""
    compute_spectrum(solver::CORKSolver, billiard::Bi, k1, k2; multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard} → SpectralData

Computes the complete CORK spectrum over the requested wavenumber interval
using adjacent Weyl-sized spectral windows.

Each requested interval is independently edge-certified and neighboring
windows use deterministic ownership. When `solver.eigenvectors=true`, each
retained root stores a [`BIMEigenstate`](@ref) containing the CORK layer
density. The density is expanded with [`symmetrize_layer_density`](@ref) before
storage, so both `state.vec` and `state.pts` refer to the complete physical
boundary.

## Arguments
* `solver::CORKSolver`: Chebyshev-CORK boundary-integral eigensolver.
* `billiard::Bi`: Billiard geometry.
* `k1`: Lower bound of the requested real-wavenumber interval.
* `k2`: Upper bound of the requested real-wavenumber interval.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded polynomial and boundary-integral matrix construction.
* `show_progress::Bool=true`: Display a progress bar during the spectral sweep.

## Returns
* `data::SpectralData`: Complete sorted complex spectrum over `k1 ≤ Re(k) ≤ k2`, including full-boundary [`BIMEigenstate`](@ref)s when `solver.eigenvectors=true`.
"""
function compute_spectrum(solver::CORKSolver, billiard::Bi, k1, k2; multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver); k1T = T(k1); k2T = T(k2); k1T<k2T || throw(ArgumentError("require k1<k2"))
    fundamental = solver.kernel.symmetry!==nothing
    intervals = plan_weyl_windows(billiard, k1T, k2T; m=solver.nlevels, Rmax=solver.Rmax, fundamental)
    isempty(intervals) && throw(ArgumentError("Spectrum interval [$k1,$k2] contains no Weyl windows"))
    if length(intervals)>1
        ap, bp = intervals[end-1]; a, b = intervals[end]; dkprev = bp-ap
        b-a<dkprev && (intervals[end] = (a, a+dkprev))
    end
    a, b = intervals[1]; k0 = (a+b)/2; Δpoly = (1+solver.guard)*(b-a)/2; pts = evaluate_points(solver, billiard, k0)
    pmin = max(2, ceil(Int, sqrt(solver.b))); p = max(solver.p, pmin)
    @time "Polynomial Validation..." begin
        P, _ = build_cork_polynomial(solver.kernel, pts, Float64(k0), Float64(Δpoly), p; multithreaded)
        err = validate_polynomial(solver.kernel, pts, P; multithreaded)
        if err>solver.taylor_tol
            while err>solver.taylor_tol
                p += 1
                P, _ = build_cork_polynomial(solver.kernel, pts, Float64(k0), Float64(Δpoly), p; multithreaded)
                err = validate_polynomial(solver.kernel, pts, P; multithreaded)
            end
        else
            while p>pmin
                P1, _ = build_cork_polynomial(solver.kernel, pts, Float64(k0), Float64(Δpoly), p-1; multithreaded)
                err1 = validate_polynomial(solver.kernel, pts, P1; multithreaded)
                err1>solver.taylor_tol && break
                p -= 1; err = err1
            end
        end
        solver.verbose && @printf("CORK polynomial degree: %d -> %d, validation error = %.3e, tolerance = %.3e\n", solver.p, p, err, solver.taylor_tol)
    end
    ks = Complex{T}[]; ts = T[]
    if solver.eigenvectors
        states = nothing
        @maybe_showprogress show_progress for i in eachindex(intervals)
            a, b = intervals[i]; k0 = (a+b)/2; dk = b-a; pts = evaluate_points(solver, billiard, k0)
            ki, ti, X = solve_vectors(solver, pts, k0, dk; multithreaded, p)
            keep = findall(j -> begin
                x = real(ki[j])
                a<=x && (i==length(intervals) ? x<=k2T : x<b) && k1T<=x
            end, eachindex(ki))
            isempty(keep) && continue
            si = [BIMEigenstate(ki[j], symmetrize_layer_density(solver.kernel, X[:,j], pts, billiard), ti[j], solver.kernel, billiard, pts) for j in keep]
            states===nothing ? (states = si) : append!(states, si)
            append!(ks, Complex{T}.(ki[keep])); append!(ts, T.(ti[keep]))
        end
        states===nothing && throw(ArgumentError("compute_spectrum found no candidates in the requested range"))
        return _finalize_spectrum(ks, ts, fill(false, length(ks)); states)
    end
    @maybe_showprogress show_progress for i in eachindex(intervals)
        a, b = intervals[i]; k0 = (a+b)/2; dk = b-a; pts = evaluate_points(solver, billiard, k0)
        ki, ti = solve(solver, pts, k0, dk; multithreaded, p)
        keep = findall(j -> begin
            x = real(ki[j])
            a<=x && (i==length(intervals) ? x<=k2T : x<b) && k1T<=x
        end, eachindex(ki))
        append!(ks, Complex{T}.(ki[keep])); append!(ts, T.(ti[keep]))
    end
    return _finalize_spectrum(ks, ts, fill(false, length(ks)))
end

"""
    compute_spectrum(solver::CORKSolver, billiard::Bi, N1::Int, N2::Int; multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard} → SpectralData

Computes the complete CORK spectrum over the wavenumber interval corresponding
approximately to Weyl-law state indices `N1` through `N2`.

## Arguments
* `solver::CORKSolver`: Chebyshev-CORK boundary-integral eigensolver.
* `billiard::Bi`: Billiard geometry.
* `N1::Int`: First requested Weyl-law state index.
* `N2::Int`: Last requested Weyl-law state index.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded polynomial and boundary-integral matrix construction.
* `show_progress::Bool=true`: Display a progress bar during the spectral sweep.

## Returns
* `data::SpectralData`: Complete sorted complex spectrum over the corresponding Weyl-law wavenumber range, including full-boundary [`BIMEigenstate`](@ref)s when `solver.eigenvectors=true`.
"""
function compute_spectrum(solver::CORKSolver, billiard::Bi, N1::Int, N2::Int; multithreaded::Bool=true, show_progress::Bool=true) where {Bi<:AbsBilliard}
    fundamental = solver.kernel.symmetry!==nothing
    k1, k2 = k_range_for_states(billiard, N1, N2; fundamental)
    return compute_spectrum(solver, billiard, k1, k2; multithreaded, show_progress)
end