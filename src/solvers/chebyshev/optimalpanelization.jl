################################################################################
# ADAPTIVE CHEBYSHEV PLAN TUNING FOR CYLINDRICAL FUNCTIONS
#
# The accelerated BIM kernels evaluate cylindrical functions Hν⁽¹⁾(kr) and
# Jν(kr) repeatedly over a radial interval r ∈ [rmin, rmax]. Direct evaluation
# is comparatively expensive, so `plan_h` and `plan_j` construct piecewise
# Chebyshev approximations that can subsequently be evaluated efficiently by
# `eval_h` and `eval_j`.
#
# Hankel and Bessel-J approximations use independent parameter pairs
#
#     Hankel:   (n_panels_h, M_h)
#     Bessel-J: (n_panels_j, M_j),
#
# where `n_panels_*` is the number of radial panels and `M_*` controls the
# polynomial order of the approximation on each panel. All requested orders
# within one family share the same parameter pair.
#
# AUTOMATIC TUNING
#
# For each requested family:
#
#   1. Start from the wavenumber k with largest |k|.
#   2. Construct the requested plan(s) only for this representative k.
#   3. Measure the interpolation error at `cfg.sampling_points` radii.
#   4. Refine the interpolation parameters until this k satisfies `cfg.tol`.
#   5. Validate the resulting parameters against every requested k.
#   6. If another k fails, continue from the current parameters using the
#      worst failing k as the new representative.
#   7. Once every k passes, construct the retained plans exactly once.
#
# This avoids rebuilding plans for every k at every tuning iteration. In the
# common case where the largest-|k| member is the most demanding one, tuning
# requires only a single-k search, one complete validation sweep, and one
# final construction of the retained plans.
#
# `_tune_cheb_plans` is the common low-level interface. Callers specify only
# the cylindrical functions they require through `h_orders` and `j_orders`.
# For wavefunction reconstruction:
#
#     DLP layer density:       H₁⁽¹⁾
#     CFIE layer density:      H₀⁽¹⁾, H₁⁽¹⁾
#     SLP from physical ∂ₙψ:   H₀⁽¹⁾
#
# BIM matrix assembly additionally requires the corresponding Jν plans.
################################################################################
# RADIAL INTERPOLATION INTERVAL
################################################################################

"""
    _cheb_geom_rminmax(G::BoundaryGeomCache{T}, ks::AbstractVector{ComplexF64}; pad::Tuple{Float64,Float64} = (0.95, 1.05)) where {T<:Real}

Determine the radial interpolation interval used by the Chebyshev cylindrical-function plans.

## Arguments
- `G::BoundaryGeomCache{T}`: Cached boundary geometry containing the pairwise distance matrix `G.R`.
- `ks::AbstractVector{ComplexF64}`: Wavenumbers for which interpolation plans will be constructed.

## Keyword Arguments
- `pad::Tuple{Float64,Float64}`: Multiplicative padding applied to the minimum and maximum nonzero pairwise distances.

## Returns
- `Tuple{Float64,Float64}`: Padded interpolation interval `(rmin, rmax)`.

The lower endpoint is bounded below by
`hankel_z_chebyshev_cutoff / maximum(abs, ks)`. Distances below this threshold
are handled by the direct or small-argument fallback of the evaluator.
"""
function _cheb_geom_rminmax(G::BoundaryGeomCache{T}, ks::AbstractVector{ComplexF64}; pad::Tuple{Float64,Float64} = (0.95, 1.05)) where {T<:Real}
    Rm = G.R; n = size(Rm, 1); rmin = Inf; rmax = 0.0
    @inbounds for j in 1:n, i in 1:n
        i == j && continue
        r = Float64(Rm[i, j])
        r < rmin && (rmin = r)
        r > rmax && (rmax = r)
    end
    isfinite(rmin) && rmax > 0.0 || throw(ArgumentError("Unable to determine a nonzero Chebyshev radial interpolation interval"))
    rmin *= pad[1]; rmax *= pad[2]
    rmin = max(rmin, hankel_z_chebyshev_cutoff / maximum(abs, ks))
    rmin < rmax || throw(ArgumentError("Empty Chebyshev radial interpolation interval: rmin=$rmin, rmax=$rmax"))
    return rmin, rmax
end

"""
    _cheb_h_error(plan::ChebHankelPlanH, rs)

Compute the maximum absolute interpolation error of a Hankel Chebyshev plan.

## Arguments
- `plan::ChebHankelPlanH`: Hankel Chebyshev interpolation plan.
- `rs`: Radial validation grid.

## Returns
- `Float64`: Maximum absolute error over `rs`.

The reference values are evaluated directly with `SpecialFunctions.besselh`.
The radial loop is multithreaded using one local maximum per thread.
"""
function _cheb_h_error(plan::ChebHankelPlanH, rs)
    nr = length(rs); nt = min(Threads.nthreads(), nr); errors = zeros(Float64, nt)
    Threads.@threads :static for t in 1:nt
        lo = fld((t - 1) * nr, nt) + 1; hi = fld(t * nr, nt); e = 0.0
        @inbounds for i in lo:hi
            r = rs[i]; p, τ = panel_t(plan, r)
            e = max(e, abs(eval_h(plan, p, τ, r) - SpecialFunctions.besselh(plan.ν, plan.κ, plan.k * r)))
        end
        errors[t] = e
    end
    return maximum(errors)
end

"""
    _cheb_j_error(plan::ChebJPlan, rs)

Compute the maximum absolute interpolation error of a Bessel-J Chebyshev plan.

## Arguments
- `plan::ChebJPlan`: Bessel-J Chebyshev interpolation plan.
- `rs`: Radial validation grid.

## Returns
- `Float64`: Maximum absolute error over `rs`.

The reference values are evaluated directly with `SpecialFunctions.besselj`.
The radial loop is multithreaded using one local maximum per thread.
"""
function _cheb_j_error(plan::ChebJPlan, rs)
    nr = length(rs); nt = min(Threads.nthreads(), nr); errors = zeros(Float64, nt)
    Threads.@threads :static for t in 1:nt
        lo = fld((t - 1) * nr, nt) + 1; hi = fld(t * nr, nt); e = 0.0
        @inbounds for i in lo:hi
            r = rs[i]; p, τ = panel_t(plan, r)
            e = max(e, abs(eval_j(plan, p, τ, r) - SpecialFunctions.besselj(plan.ν, plan.k * r)))
        end
        errors[t] = e
    end
    return maximum(errors)
end

################################################################################
# FAMILY ERRORS
# H₀/H₁ share the same Hankel interpolation parameters and J₀/J₁ share the
# same Bessel-J parameters. The family error is therefore the largest error
# among all requested orders.
################################################################################

function _cheb_h_family_error(k::ComplexF64, rmin::Float64, rmax::Float64, npanels::Int, M::Int, rs, orders::Tuple)
    e = 0.0
    for ν in orders
        plan = plan_h(ν, 1, k, rmin, rmax; npanels, M)
        e = max(e, _cheb_h_error(plan, rs))
    end
    return e
end

function _cheb_j_family_error(k::ComplexF64, rmin::Float64, rmax::Float64, npanels::Int, M::Int, rs, orders::Tuple)
    e = 0.0
    for ν in orders
        plan = plan_j(ν, k, rmin, rmax; npanels, M)
        e = max(e, _cheb_j_error(plan, rs))
    end
    return e
end

function _cheb_h_batch_errors!(errors::Vector{Float64}, ks::AbstractVector{ComplexF64}, rmin::Float64, rmax::Float64, npanels::Int, M::Int, rs, orders::Tuple)
    @inbounds for j in eachindex(ks)
        errors[j] = _cheb_h_family_error(ks[j], rmin, rmax, npanels, M, rs, orders)
    end
    return errors
end

function _cheb_j_batch_errors!(errors::Vector{Float64}, ks::AbstractVector{ComplexF64}, rmin::Float64, rmax::Float64, npanels::Int, M::Int, rs, orders::Tuple)
    @inbounds for j in eachindex(ks)
        errors[j] = _cheb_j_family_error(ks[j], rmin, rmax, npanels, M, rs, orders)
    end
    return errors
end

################################################################################
# PARAMETER GROWTH AND LOW-LEVEL FUNCTIONS
# Panel refinement is preferred because it shortens the radial interval
# represented by each polynomial. Every fifth parameter-growth step increases
# the polynomial order instead, balancing h- and p-refinement.
################################################################################

@inline function _cheb_grow_params(n::Int, M::Int, it::Int, cfg::ChebyshevConfig)
    return it % 5 == 0 ? (n, M + cfg.grow_M) : (ceil(Int, cfg.grow_panels * n), M)
end

function _tune_h_params(rmin::Float64, rmax::Float64, ks::AbstractVector{ComplexF64}, rs, cfg::ChebyshevConfig, orders::Tuple)
    nh, Mh = cfg.n_panels_h, cfg.M_h
    isempty(orders) && return nh, Mh
    cfg.param_strategy === :manual && return nh, Mh
    tol = Float64(cfg.tol); errors = Vector{Float64}(undef, length(ks)); _, jrep = findmax(abs, ks); it = 0
    while true
        e = _cheb_h_family_error(ks[jrep], rmin, rmax, nh, Mh, rs, orders)
        while e >= tol && it < cfg.max_iter
            it += 1
            nh, Mh = _cheb_grow_params(nh, Mh, it, cfg)
            e = _cheb_h_family_error(ks[jrep], rmin, rmax, nh, Mh, rs, orders)
        end
        _cheb_h_batch_errors!(errors, ks, rmin, rmax, nh, Mh, rs, orders)
        emax, jmax = findmax(errors)
        emax < tol && return nh, Mh
        if it >= cfg.max_iter
            @warn "Hankel Chebyshev tuning did not reach the requested tolerance" tol emax nh Mh orders
            return nh, Mh
        end
        jrep = jmax
    end
end

function _tune_j_params(rmin::Float64, rmax::Float64, ks::AbstractVector{ComplexF64}, rs, cfg::ChebyshevConfig, orders::Tuple)
    nj, Mj = cfg.n_panels_j, cfg.M_j
    isempty(orders) && return nj, Mj
    cfg.param_strategy === :manual && return nj, Mj
    tol = Float64(cfg.tol); errors = Vector{Float64}(undef, length(ks)); _, jrep = findmax(abs, ks); it = 0
    while true
        e = _cheb_j_family_error(ks[jrep], rmin, rmax, nj, Mj, rs, orders)
        while e >= tol && it < cfg.max_iter
            it += 1
            nj, Mj = _cheb_grow_params(nj, Mj, it, cfg)
            e = _cheb_j_family_error(ks[jrep], rmin, rmax, nj, Mj, rs, orders)
        end
        _cheb_j_batch_errors!(errors, ks, rmin, rmax, nj, Mj, rs, orders)
        emax, jmax = findmax(errors)
        emax < tol && return nj, Mj
        if it >= cfg.max_iter
            @warn "Bessel-J Chebyshev tuning did not reach the requested tolerance" tol emax nj Mj orders
            return nj, Mj
        end
        jrep = jmax
    end
end

function _build_h_plans(ν::Int, ks::AbstractVector{ComplexF64}, rmin::Float64, rmax::Float64, npanels::Int, M::Int)
    plans = Vector{ChebHankelPlanH}(undef, length(ks))
    Threads.@threads for j in eachindex(ks)
        @inbounds plans[j] = plan_h(ν, 1, ks[j], rmin, rmax; npanels, M)
    end
    return plans
end

function _build_j_plans(ν::Int, ks::AbstractVector{ComplexF64}, rmin::Float64, rmax::Float64, npanels::Int, M::Int)
    plans = Vector{ChebJPlan}(undef, length(ks))
    Threads.@threads for j in eachindex(ks)
        @inbounds plans[j] = plan_j(ν, ks[j], rmin, rmax; npanels, M)
    end
    return plans
end

"""
    _tune_cheb_plans(rmin::Float64, rmax::Float64, ks::AbstractVector{ComplexF64}, cfg::ChebyshevConfig{T}; h_orders::Tuple = (), j_orders::Tuple = ()) where {T<:Real}

Tune interpolation parameters and construct the requested cylindrical-function Chebyshev plans.

## Arguments
- `rmin::Float64`: Lower endpoint of the radial interpolation interval.
- `rmax::Float64`: Upper endpoint of the radial interpolation interval.
- `ks::AbstractVector{ComplexF64}`: Wavenumbers for which plans are required.
- `cfg::ChebyshevConfig{T}`: Chebyshev interpolation and tuning configuration.

## Keyword Arguments
- `h_orders::Tuple`: Hankel orders to construct. Supported orders are `0` and `1`.
- `j_orders::Tuple`: Bessel-J orders to construct. Supported orders are `0` and `1`.

## Returns
A three-tuple `(hplans, jplans, cfg_used)` where:
- `hplans` is the named tuple `(h0, h1)`. Unrequested orders are `nothing`.
- `jplans` is the named tuple `(j0, j1)`. Unrequested orders are `nothing`.
- `cfg_used::ChebyshevConfig{T}` contains the interpolation parameters selected by the tuner.

For `cfg.param_strategy === :manual`, parameter tuning and validation are
skipped and the requested plans are constructed directly from `cfg`.
"""
function _tune_cheb_plans(rmin::Float64, rmax::Float64, ks::AbstractVector{ComplexF64}, cfg::ChebyshevConfig{T}; h_orders::Tuple = (), j_orders::Tuple = ()) where {T<:Real}
    isempty(ks) && throw(ArgumentError("ks must be nonempty"))
    all(ν -> ν == 0 || ν == 1, h_orders) || throw(ArgumentError("h_orders supports only orders 0 and 1"))
    all(ν -> ν == 0 || ν == 1, j_orders) || throw(ArgumentError("j_orders supports only orders 0 and 1"))
    if cfg.param_strategy === :manual
        nh, Mh = cfg.n_panels_h, cfg.M_h; nj, Mj = cfg.n_panels_j, cfg.M_j
    else
        rs = range(rmin, rmax; length = cfg.sampling_points)
        nh, Mh = _tune_h_params(rmin, rmax, ks, rs, cfg, h_orders)
        nj, Mj = _tune_j_params(rmin, rmax, ks, rs, cfg, j_orders)
    end
    h0 = 0 in h_orders ? _build_h_plans(0, ks, rmin, rmax, nh, Mh) : nothing
    h1 = 1 in h_orders ? _build_h_plans(1, ks, rmin, rmax, nh, Mh) : nothing
    j0 = 0 in j_orders ? _build_j_plans(0, ks, rmin, rmax, nj, Mj) : nothing
    j1 = 1 in j_orders ? _build_j_plans(1, ks, rmin, rmax, nj, Mj) : nothing
    cfg_used = ChebyshevConfig(T; n_panels_h = nh, M_h = Mh, n_panels_j = nj, M_j = Mj, tol = cfg.tol, max_iter = cfg.max_iter, sampling_points = cfg.sampling_points, grow_panels = cfg.grow_panels, grow_M = cfg.grow_M, param_strategy = cfg.param_strategy)
    return (; h0, h1), (; j0, j1), cfg_used
end

"""
    tune_dlp_cheb_plans(rmin::Float64, rmax::Float64, ks::AbstractVector{ComplexF64}, cfg::ChebyshevConfig{T}) where {T<:Real}

Tune and construct the Chebyshev plans required by the accelerated DLP kernel.

## Arguments
- `rmin::Float64`: Lower endpoint of the radial interpolation interval.
- `rmax::Float64`: Upper endpoint of the radial interpolation interval.
- `ks::AbstractVector{ComplexF64}`: Wavenumbers for which plans are required.
- `cfg::ChebyshevConfig{T}`: Chebyshev interpolation and tuning configuration.

## Returns
A three-tuple `(plans1, plansj1, cfg_used)` where:
- `plans1::Vector{ChebHankelPlanH}` contains the `H₁⁽¹⁾` plans.
- `plansj1::Vector{ChebJPlan}` contains the `J₁` plans.
- `cfg_used::ChebyshevConfig{T}` contains the interpolation parameters actually used.
"""
function tune_dlp_cheb_plans(rmin::Float64, rmax::Float64, ks::AbstractVector{ComplexF64}, cfg::ChebyshevConfig{T}) where {T<:Real}
    hp, jp, cfg_used = _tune_cheb_plans(rmin, rmax, ks, cfg; h_orders = (1,), j_orders = (1,))
    return hp.h1, jp.j1, cfg_used
end

"""
    tune_cfie_cheb_plans(rmin::Float64, rmax::Float64, ks::AbstractVector{ComplexF64}, cfg::ChebyshevConfig{T}) where {T<:Real}

Tune and construct the Chebyshev plans required by the accelerated CFIE kernel.

## Arguments
- `rmin::Float64`: Lower endpoint of the radial interpolation interval.
- `rmax::Float64`: Upper endpoint of the radial interpolation interval.
- `ks::AbstractVector{ComplexF64}`: Wavenumbers for which plans are required.
- `cfg::ChebyshevConfig{T}`: Chebyshev interpolation and tuning configuration.

## Returns
A five-tuple `(plans0, plans1, plansj0, plansj1, cfg_used)` where:
- `plans0::Vector{ChebHankelPlanH}` contains the `H₀⁽¹⁾` plans.
- `plans1::Vector{ChebHankelPlanH}` contains the `H₁⁽¹⁾` plans.
- `plansj0::Vector{ChebJPlan}` contains the `J₀` plans.
- `plansj1::Vector{ChebJPlan}` contains the `J₁` plans.
- `cfg_used::ChebyshevConfig{T}` contains the interpolation parameters actually used.
"""
function tune_cfie_cheb_plans(rmin::Float64, rmax::Float64, ks::AbstractVector{ComplexF64}, cfg::ChebyshevConfig{T}) where {T<:Real}
    hp, jp, cfg_used = _tune_cheb_plans(rmin, rmax, ks, cfg; h_orders = (0, 1), j_orders = (0, 1))
    return hp.h0, hp.h1, jp.j0, jp.j1, cfg_used
end