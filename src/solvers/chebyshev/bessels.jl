#############################################################################
# Piecewise-Chebyshev special-function core for Helmholtz boundary integral
# operators in 2D, at fixed complex wavenumber k:
#
#     H_ν^(1)(k r),   J_ν(k r),   r >= 0.
#
# Adapted from
# QuantumBilliards-develop/src/chebyshev/chebyshev_bessels.jl, restricted to
# the complex-k route (every wavenumber this migration's BeynSolver/
# ExpandedBIMSolver present to construct_matrices is Complex{Float64}, see
# `_bim_widen_k`). The SLP wavefunction-reconstruction plan
# (`SLPWavefunctionChebPlan`, see the bottom of this file) is real-k only and
# used solely by `ϕ_slp` (states/wavefunctions.jl); main's unified
# Green's-function reconstruction covers both DLP and CFIE eigenstates with
# this single plan, so `-develop`'s separate `CFIEWavefunctionChebPlan` has
# no counterpart here (see the note above `SLPWavefunctionChebPlan`).
#
# See core.jl (`_chebfit!`, `_cheb_clenshaw`, `_breaks_uniform`) for the
# generic Chebyshev machinery this file builds on.
#############################################################################

const γ_cheb = MathConstants.eulergamma
const hankel_z_chebyshev_cutoff_small_z = 0.001
const hankel_z_chebyshev_cutoff = 0.2

struct ChebHankelTableH
    a::Float64
    b::Float64
    M::Int
    ν::Int
    κ::Int
    c::Vector{ComplexF64}
end

struct ChebJTable
    a::Float64
    b::Float64
    M::Int
    ν::Int
    c::Vector{ComplexF64}
end

# Build a Chebyshev table on panel [a,b] for the unscaled Hankel H_ν^(κ)(k r), k complex.
function _build_table_h!(ν::Int, κ::Int, k::ComplexF64, a::Float64, b::Float64; M::Int=16)::ChebHankelTableH
    @assert a>0 && b>a "a=$(a), b=$(b)"
    f1 = Vector{ComplexF64}(undef, M+1)
    @inbounds for j in 0:M
        t = cospi(j/M)
        r = ((b+a)+(b-a)*t)/2
        z = k*r
        f1[j+1] = SpecialFunctions.besselh(ν, κ, z)
    end
    c = Vector{ComplexF64}(undef, M+1)
    _chebfit!(c, f1)
    return ChebHankelTableH(a, b, M, ν, κ, c)
end

function _build_table_j!(ν::Int, k::ComplexF64, a::Float64, b::Float64; M::Int=16)::ChebJTable
    @assert a>=0 && b>a "a=$(a), b=$(b)"
    f1 = Vector{ComplexF64}(undef, M+1)
    @inbounds for j in 0:M
        t = cospi(j/M)
        r = ((b+a)+(b-a)*t)/2
        z = k*r
        f1[j+1] = SpecialFunctions.besselj(ν, z)
    end
    c = Vector{ComplexF64}(undef, M+1)
    _chebfit!(c, f1)
    return ChebJTable(a, b, M, ν, c)
end

struct ChebHankelPlanH
    k::ComplexF64
    ν::Int
    κ::Int
    panels::Vector{ChebHankelTableH}
    rmin::Float64
    rmax::Float64
    dr::Float64
    invdr::Float64
    npanels::Int
end

struct ChebJPlan
    k::ComplexF64
    ν::Int
    panels::Vector{ChebJTable}
    rmin::Float64
    rmax::Float64
    dr::Float64
    invdr::Float64
    npanels::Int
end

"""
    plan_h(ν, κ, k, rmin, rmax; npanels = 64, M = 16) → ChebHankelPlanH

Build a piecewise-Chebyshev plan for the unscaled Hankel `H_ν^(κ)(k r)` over
`r ∈ [rmin,rmax]` at fixed complex `k`, split into `npanels` uniform panels of
Chebyshev degree `M`.
"""
function plan_h(ν::Int, κ::Int, k::ComplexF64, rmin::Float64, rmax::Float64; npanels::Int=64, M::Int=16)::ChebHankelPlanH
    @assert rmin>0 && rmax>rmin
    br = _breaks_uniform(rmin, rmax, npanels)
    panels = Vector{ChebHankelTableH}(undef, npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i] = _build_table_h!(ν, κ, k, br[i], br[i+1]; M=M)
    end
    dr = (rmax-rmin)/npanels
    return ChebHankelPlanH(k, ν, κ, panels, rmin, rmax, dr, inv(dr), npanels)
end

"""
    plan_j(ν, k, rmin, rmax; npanels = 64, M = 16) → ChebJPlan

Build a piecewise-Chebyshev plan for the Bessel function `J_ν(k r)` over
`r ∈ [rmin,rmax]` at fixed complex `k`.
"""
function plan_j(ν::Int, k::ComplexF64, rmin::Float64, rmax::Float64; npanels::Int=64, M::Int=16)::ChebJPlan
    @assert rmin>=0 && rmax>rmin
    br = _breaks_uniform(rmin, rmax, npanels)
    panels = Vector{ChebJTable}(undef, npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i] = _build_table_j!(ν, k, br[i], br[i+1]; M=M)
    end
    dr = (rmax-rmin)/npanels
    return ChebJPlan(k, ν, panels, rmin, rmax, dr, inv(dr), npanels)
end

# O(1) uniform-panel lookup: p = floor((r-rmin)/dr)+1, clamped to [1,npanels].
@inline function _find_panel_uniform(pl::Union{ChebHankelPlanH,ChebJPlan}, r::Float64)::Int
    p = Int(floor((r-pl.rmin)*pl.invdr))+1
    return ifelse(p<1, 1, ifelse(p>pl.npanels, pl.npanels, p))
end
@inline _find_panel(pl::Union{ChebHankelPlanH,ChebJPlan}, r::Float64)::Int = _find_panel_uniform(pl, r)

"""
    panel_and_geom(pl, rvec) → (pidx, t, invsqrt)

Single-pass, threaded precompute of the panel index, mapped Chebyshev
coordinate, and `1/√r` for every radius in `rvec`. `pidx[i]==0` signals `r`
below the plan's `rmin` (near-zero fallback, see `eval_h`/`eval_j`).
"""
function panel_and_geom(pl::Union{ChebHankelPlanH,ChebJPlan}, rvec::AbstractVector{Float64})::Tuple{Vector{Int32},Vector{Float64},Vector{Float64}}
    n = length(rvec)
    pidx = Vector{Int32}(undef, n)
    t = Vector{Float64}(undef, n)
    invsqrt = Vector{Float64}(undef, n)
    rmin = pl.rmin
    dr = pl.dr
    invdr = pl.invdr
    np = pl.npanels
    @inbounds Threads.@threads for i in eachindex(rvec)
        r = rvec[i]
        if r<rmin
            pidx[i] = Int32(0)
            t[i] = 0.0
        else
            p = Int(floor((r-rmin)*invdr))+1
            p = ifelse(p<1, 1, ifelse(p>np, np, p))
            pidx[i] = Int32(p)
            center = rmin+(p-0.5)*dr
            t[i] = 2*(r-center)*invdr
        end
        invsqrt[i] = inv(sqrt(r))
    end
    return pidx, t, invsqrt
end

# Locate the radial panel and Chebyshev coordinate for a single r (pidx=0 signals r<rmin, near-zero fallback).
@inline function panel_t(pl::Union{ChebHankelPlanH,ChebJPlan}, r::Float64)
    if r<pl.rmin
        return Int32(0), 0.0
    end
    p = _find_panel(pl, r)
    P = pl.panels[p]
    return Int32(p), (2*r-(P.b+P.a))/(P.b-P.a)
end

##################################################################
############## NEAR 0 EXPANSIONS FOR H0 AND H1 ###################
##################################################################

@inline function _small_h0_series(z::ComplexF64)
    zz = z*z
    P = 2123366400+zz*(-530841600+zz*(33177600+zz*(-921600+zz*(14400+zz*(-144+zz)))))
    Q = 10616832000+zz*(-995328000+zz*(33792000+zz*(-600000+zz*(6576+zz*(-49)))))
    return (((10*pi+20*im*γ_cheb)*P+im*zz*Q)/(21233664000*pi))+(im*P/(1061683200*pi))*log(z/2)
end
@inline _small_h0_series(z::T) where {T<:Number} = _small_h0_series(ComplexF64(z))

@inline function _small_h1_series(z::ComplexF64)
    zz = z*z
    A = -4161798144000+
        zz*(1040449536000*(-1+2*γ_cheb-1im*pi)+
        zz*(-65028096000*(-5+4*γ_cheb-2im*pi)+
        zz*(1806336000*(-10+6*γ_cheb-3im*pi)+
        zz*(-9408000*(-47+24*γ_cheb-12im*pi)+
        zz*(47040*(-131+60*γ_cheb-30im*pi)+
        zz*(-784*(-71+30*γ_cheb-15im*pi)+
        zz*(-353+140*γ_cheb-70im*pi)))))))
    R = 14863564800+zz*(-1857945600+zz*(77414400+zz*(-1612800+zz*(20160+zz*(-168+zz)))))
    return (im*A/(2080899072000*pi*z))+(im*z*R/(14863564800*pi))*log(z/2)
end
@inline _small_h1_series(z::T) where {T<:Number} = _small_h1_series(ComplexF64(z))

##################################################################
###################### EVALUATION FUNCTIONS ######################
##################################################################

# Scalar evaluation of H_ν^(κ)(k r) at one (pidx,t,r) triple. Below
# `pl.rmin` (`pidx==0`) OR whenever `|k*r|` is still inside the mid-range
# `hankel_z_chebyshev_cutoff` band (guards against a batch-tuned `rmin`
# floored for a *different*, larger `|k|` in the same multi-k plan set —
# e.g. Beyn's contour nodes span a range of `|k|`, but every node's plan
# shares one `rmin` sized for the largest node — leaving smaller-|k| nodes
# with `|k*rmin|<hankel_z_chebyshev_cutoff` even though `r>=rmin`; see
# `h0_h1_multi_ks_at_r!` below and `_cheb_geom_rminmax` in
# optimalpanelization.jl), direct/series evaluation is used instead of the
# Chebyshev panel fit, matching `-develop`'s combined H₀/H₁ evaluators.
@inline function eval_h(pl::ChebHankelPlanH, pidx::Int32, t::Float64, r::Float64)
    z = pl.k*r
    if pidx==0 || abs(z)<hankel_z_chebyshev_cutoff
        if pl.ν==0
            return abs(z)<hankel_z_chebyshev_cutoff_small_z ? _small_h0_series(z) : SpecialFunctions.besselh(0, pl.κ, z)
        elseif pl.ν==1
            return abs(z)<hankel_z_chebyshev_cutoff_small_z ? _small_h1_series(z) : SpecialFunctions.besselh(1, pl.κ, z)
        else
            return SpecialFunctions.besselh(pl.ν, pl.κ, z)
        end
    end
    return _cheb_clenshaw(pl.panels[pidx].c, t)
end

# Scalar evaluation of J_ν(k r) at one (pidx,t) pair (J is regular at r=0, no small-z fallback needed within the plan's range).
@inline function eval_j(pl::ChebJPlan, pidx::Int32, t::Float64, r::Float64)
    pidx==0 && return SpecialFunctions.besselj(pl.ν, pl.k*r)
    return _cheb_clenshaw(pl.panels[pidx].c, t)
end

"""
    eval_h_multi_ks!(out, plans, r, pidx, t)

Evaluate `H_ν^(κ)(k_m r)` for the same radius `r` (already mapped to
`(pidx,t)`) across every plan `plans[m]` (one per wavenumber), writing into
`out` in place.
"""
function eval_h_multi_ks!(out::AbstractVector{ComplexF64}, plans::AbstractVector{ChebHankelPlanH}, r::Float64, pidx::Int32, t::Float64)
    @inbounds for m in eachindex(plans)
        plan_m = plans[m]
        z = plan_m.k*r
        if pidx==0 || abs(z)<hankel_z_chebyshev_cutoff
            if plan_m.ν==0
                out[m] = abs(z)<hankel_z_chebyshev_cutoff_small_z ? _small_h0_series(z) : SpecialFunctions.besselh(0, plan_m.κ, z)
            elseif plan_m.ν==1
                out[m] = abs(z)<hankel_z_chebyshev_cutoff_small_z ? _small_h1_series(z) : SpecialFunctions.besselh(1, plan_m.κ, z)
            else
                out[m] = SpecialFunctions.besselh(plan_m.ν, plan_m.κ, z)
            end
        else
            out[m] = _cheb_clenshaw(plan_m.panels[pidx].c, t)
        end
    end
    return nothing
end

"""
    eval_j_multi_ks!(out, plans, pidx, t, r)

Evaluate `J_ν(k_m r)` for the same radius (already mapped to `(pidx,t)`)
across every plan `plans[m]`, writing into `out` in place. Below the plan's
`rmin` (`pidx==0`, see `panel_t`), falls back to direct `SpecialFunctions`
evaluation using the physical distance `r`, matching the scalar `eval_j`'s
own near-zero fallback (`J` is regular at `r=0`, so no small-argument series
is needed).
"""
@inline function eval_j_multi_ks!(out::AbstractVector{ComplexF64}, plans::AbstractVector{ChebJPlan}, pidx::Int32, t::Float64, r::Float64)
    @inbounds for m in eachindex(plans)
        pl = plans[m]
        out[m] = pidx==0 ? SpecialFunctions.besselj(pl.ν, pl.k*r) : _cheb_clenshaw(pl.panels[pidx].c, t)
    end
    return nothing
end

"""
    h0_h1_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, pidx, t, r)

Evaluate `H₀^(1)` and `H₁^(1)` for every wavenumber at one fixed distance,
writing the results in place.
"""
@inline function h0_h1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64}, h1vals::AbstractVector{ComplexF64}, plans0::AbstractVector{ChebHankelPlanH}, plans1::AbstractVector{ChebHankelPlanH}, pidx::Int32, t::Float64, r::Float64)
    @inbounds for m in eachindex(plans0)
        z = plans0[m].k*r
        az = abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h0vals[m] = _small_h0_series(z)
            h1vals[m] = _small_h1_series(z)
        elseif az<hankel_z_chebyshev_cutoff || pidx==0
            h0vals[m] = SpecialFunctions.besselh(0, 1, z)
            h1vals[m] = SpecialFunctions.besselh(1, 1, z)
        else
            h0vals[m] = _cheb_clenshaw(plans0[m].panels[pidx].c, t)
            h1vals[m] = _cheb_clenshaw(plans1[m].panels[pidx].c, t)
        end
    end
    return nothing
end

"""
    h1_j1_multi_ks_at_r!(h1vals, j1vals, plans1, plansj1, pidx_h, t_h, pidx_j, t_j, r)

Evaluate `H₁^(1)` and `J₁` for every wavenumber at one fixed distance,
writing the results in place. Used by the value-only (Beyn) DLP Chebyshev
assembly.
"""
@inline function h1_j1_multi_ks_at_r!(h1vals::AbstractVector{ComplexF64}, j1vals::AbstractVector{ComplexF64}, plans1::AbstractVector{ChebHankelPlanH}, plansj1::AbstractVector{ChebJPlan}, pidx_h::Int32, t_h::Float64, pidx_j::Int32, t_j::Float64, r::Float64)
    eval_h_multi_ks!(h1vals, plans1, r, pidx_h, t_h)
    eval_j_multi_ks!(j1vals, plansj1, pidx_j, t_j, r)
    return nothing
end

"""
    h0_h1_j0_j1_multi_ks_at_r!(h0vals, h1vals, j0vals, j1vals, plans0, plans1, plansj0, plansj1, pidx_h, t_h, pidx_j, t_j, r)

Evaluate `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁` for every wavenumber at one fixed
distance, writing the results in place. Used by the value-only (Beyn) CFIE
Chebyshev assembly.
"""
@inline function h0_h1_j0_j1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64}, h1vals::AbstractVector{ComplexF64}, j0vals::AbstractVector{ComplexF64}, j1vals::AbstractVector{ComplexF64}, plans0::AbstractVector{ChebHankelPlanH}, plans1::AbstractVector{ChebHankelPlanH}, plansj0::AbstractVector{ChebJPlan}, plansj1::AbstractVector{ChebJPlan}, pidx_h::Int32, t_h::Float64, pidx_j::Int32, t_j::Float64, r::Float64)
    h0_h1_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, pidx_h, t_h, r)
    eval_j_multi_ks!(j0vals, plansj0, pidx_j, t_j, r)
    eval_j_multi_ks!(j1vals, plansj1, pidx_j, t_j, r)
    return nothing
end
