################################################################################
# PIECEWISE-CHEBYSHEV BESSEL AND HANKEL EVALUATION
#
# This file implements fast repeated evaluation of the radial special functions
#
#                    H_ν^(κ)(k r),        J_ν(k r),
#
# at fixed complex wavenumber k over a prescribed radial interval. The interval
#
#                         r ∈ [rmin,rmax]
#
# is divided into uniform panels [a_p,b_p]. On each panel the radial function
# is interpolated at Chebyshev-Lobatto nodes and represented by a degree-M
# Chebyshev expansion
#
#                    f(r) ≈ Σ_{m=0}^M c_m T_m(t),
#
# where the physical radius is mapped to the local Chebyshev coordinate
#
#                  t = (2r-a_p-b_p)/(b_p-a_p) ∈ [-1,1].
#
# Once the tables have been constructed, evaluation consists of an O(1) panel
# lookup followed by Clenshaw evaluation of the corresponding polynomial.
#
# Hankel functions require additional treatment near the origin because
# H_ν^(κ)(z) is singular there and is poorly suited to polynomial interpolation
# across very small arguments. For H₀^(1) and H₁^(1), sufficiently small |z|
# is evaluated with explicit local series. An intermediate small-|z| region is
# evaluated directly with SpecialFunctions, while the piecewise-Chebyshev
# representation is used only outside this near-zero region. Bessel J_ν is
# regular at the origin and therefore requires only direct evaluation below
# the tabulated radial interval.
################################################################################

const γ_cheb = MathConstants.eulergamma
const hankel_z_chebyshev_cutoff_small_z = 0.001
const hankel_z_chebyshev_cutoff = 0.2

# Store one degree-M Chebyshev approximation of H_ν^(κ)(kr) on the radial
# panel [a,b] at the fixed complex wavenumber used to construct the enclosing
# plan. `c` contains the Chebyshev coefficients consumed by `_cheb_clenshaw`.
struct ChebHankelTableH
    a::Float64
    b::Float64
    M::Int
    ν::Int
    κ::Int
    c::Vector{ComplexF64}
end

# Store one degree-M Chebyshev approximation of J_ν(kr) on the radial panel
# [a,b]. Unlike the Hankel table, the interval is allowed to begin at r=0
# because J_ν(z) is regular at the origin.
struct ChebJTable
    a::Float64
    b::Float64
    M::Int
    ν::Int
    c::Vector{ComplexF64}
end

# Construct the Chebyshev interpolant of H_ν^(κ)(kr) on one radial panel
# [a,b]. The function is sampled at the M+1 Chebyshev-Lobatto nodes
#
#                         t_j = cos(πj/M),
#
# mapped affinely to r∈[a,b]. `_chebfit!` converts these nodal values to the
# coefficients used by Clenshaw evaluation. Hankel panels require a>0 because
# the function is singular at the origin; sufficiently small arguments are
# handled separately by the evaluation routines.
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

# Construct the Chebyshev interpolant of J_ν(kr) on one radial panel [a,b].
# Because J_ν is regular at the origin, the first panel may begin at a=0.
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

# Store the complete piecewise-Chebyshev approximation of H_ν^(κ)(kr) at one
# fixed complex wavenumber. The radial interval is divided into `npanels`
# uniform panels, each carrying its own Chebyshev table. `dr` and `invdr`
# permit constant-time conversion from a physical radius to its panel index.
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

# Store the complete piecewise-Chebyshev approximation of J_ν(kr) at one fixed
# complex wavenumber. The uniform panel geometry is stored explicitly so that
# repeated radial evaluations require only constant-time panel lookup followed
# by Clenshaw evaluation.
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

"""
    plan_h(ν::Int, κ::Int, k::ComplexF64, rmin::Float64, rmax::Float64; npanels::Int=64, M::Int=16)

Construct a piecewise-Chebyshev evaluation plan for `H_ν^(κ)(k r)` on `r ∈ [rmin,rmax]`.

The radial interval is divided into `npanels` uniform panels. On each panel,
the Hankel function is sampled at `M+1` Chebyshev-Lobatto nodes and converted
to a degree-`M` Chebyshev expansion. Individual panels are constructed in
parallel.

The tabulated interval must satisfy `rmin > 0`. Hankel evaluation below the
tabulated interval is handled separately by the near-zero/direct fallbacks in
[`eval_h`](@ref) and the corresponding multi-wavenumber routines.

## Arguments
* `ν::Int`: Order of the Hankel function.
* `κ::Int`: Hankel kind.
* `k::ComplexF64`: Fixed complex wavenumber.
* `rmin::Float64`: Lower radius of the tabulated interval.
* `rmax::Float64`: Upper radius of the tabulated interval.

## Keyword Arguments
* `npanels::Int = 64`: Number of uniform radial panels.
* `M::Int = 16`: Chebyshev polynomial degree on each panel.

## Returns
* `plan::ChebHankelPlanH`: Piecewise-Chebyshev Hankel evaluation plan.
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
    plan_j(ν::Int, k::ComplexF64, rmin::Float64, rmax::Float64; npanels::Int=64, M::Int=16)

Construct a piecewise-Chebyshev evaluation plan for `J_ν(k r)` on `r ∈ [rmin,rmax]`.

The radial interval is divided into `npanels` uniform panels. On each panel,
the Bessel function is sampled at `M+1` Chebyshev-Lobatto nodes and converted
to a degree-`M` Chebyshev expansion. Individual panels are constructed in
parallel.

Because `J_ν` is regular at the origin, `rmin` may be zero.

## Arguments
* `ν::Int`: Order of the Bessel function.
* `k::ComplexF64`: Fixed complex wavenumber.
* `rmin::Float64`: Lower radius of the tabulated interval.
* `rmax::Float64`: Upper radius of the tabulated interval.

## Keyword Arguments
* `npanels::Int = 64`: Number of uniform radial panels.
* `M::Int = 16`: Chebyshev polynomial degree on each panel.

## Returns
* `plan::ChebJPlan`: Piecewise-Chebyshev Bessel evaluation plan.
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
    panel_and_geom(pl::Union{ChebHankelPlanH,ChebJPlan}, rvec::AbstractVector{Float64})

Precompute panel indices, local Chebyshev coordinates, and inverse square roots
for a vector of physical radii.

For every `r ≥ pl.rmin`, the containing uniform panel is found and the radius
is mapped to its local Chebyshev coordinate

    t = 2(r-rcenter)/dr.

For `r < pl.rmin`, the panel index is set to zero, which signals that the
evaluation routines must use their direct or near-zero fallback rather than a
Chebyshev table. The geometric quantity `1/√r` is computed simultaneously for
reuse by callers that require it.

## Arguments
* `pl::Union{ChebHankelPlanH,ChebJPlan}`: Piecewise-Chebyshev radial evaluation plan.
* `rvec::AbstractVector{Float64}`: Physical radii to preprocess.

## Returns
* `pidx::Vector{Int32}`: Panel index for each radius, with zero indicating `r < pl.rmin`.
* `t::Vector{Float64}`: Local Chebyshev coordinate associated with each radius.
* `invsqrt::Vector{Float64}`: Values `1/√r` for the input radii.
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

# Locate one physical radius in a piecewise-Chebyshev plan and compute its local
# coordinate t∈[-1,1]. A returned panel index of zero signals r<rmin and tells
# the caller to bypass polynomial interpolation. Otherwise the coordinate is
# computed from the actual stored panel endpoints, making this helper consistent
# with the panel tables themselves.
@inline function panel_t(pl::Union{ChebHankelPlanH,ChebJPlan}, r::Float64)
    if r<pl.rmin
        return Int32(0), 0.0
    end
    p = _find_panel(pl, r)
    P = pl.panels[p]
    return Int32(p), (2*r-(P.b+P.a))/(P.b-P.a)
end

# Evaluate H_ν^(κ)(kr) from a precomputed plan at one physical radius. The
# evaluation uses three regimes determined by z=kr:
# 1. For H₀^(1) and H₁^(1) at extremely small |z|, use the explicit local
#    series to resolve the singular small-argument behavior.
# 2. If the radius lies below the tabulated interval (`pidx==0`) or |z| remains
#    below the Chebyshev cutoff, evaluate the special function directly.
# 3. Otherwise evaluate the stored panel polynomial with Clenshaw's algorithm.
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
    eval_h_multi_ks!(out::AbstractVector{ComplexF64}, plans::AbstractVector{ChebHankelPlanH}, r::Float64, pidx::Int32, t::Float64)

Evaluate `H_ν^(κ)(k_m r)` at one physical radius for a collection of
wavenumber-dependent Hankel plans.

## Arguments
* `out::AbstractVector{ComplexF64}`: Output storage, with one entry per plan.
* `plans::AbstractVector{ChebHankelPlanH}`: Hankel plans associated with the wavenumbers `k_m`.
* `r::Float64`: Common physical radius.
* `pidx::Int32`: Precomputed common radial panel index, with zero indicating the below-plan fallback region.
* `t::Float64`: Precomputed local Chebyshev coordinate.

## Returns
* `nothing`: Results are written to `out` in place.
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
    eval_j_multi_ks!(out::AbstractVector{ComplexF64}, plans::AbstractVector{ChebJPlan}, pidx::Int32, t::Float64, r::Float64)

Evaluate `J_ν(k_m r)` at one physical radius for a collection of
wavenumber-dependent Bessel plans.

## Arguments
* `out::AbstractVector{ComplexF64}`: Output storage, with one entry per plan.
* `plans::AbstractVector{ChebJPlan}`: Bessel plans associated with the wavenumbers `k_m`.
* `pidx::Int32`: Precomputed common radial panel index, with zero indicating the below-plan fallback region.
* `t::Float64`: Precomputed local Chebyshev coordinate.
* `r::Float64`: Common physical radius.

## Returns
* `nothing`: Results are written to `out` in place.
"""
@inline function eval_j_multi_ks!(out::AbstractVector{ComplexF64}, plans::AbstractVector{ChebJPlan}, pidx::Int32, t::Float64, r::Float64)
    @inbounds for m in eachindex(plans)
        pl = plans[m]
        out[m] = pidx==0 ? SpecialFunctions.besselj(pl.ν, pl.k*r) : _cheb_clenshaw(pl.panels[pidx].c, t)
    end
    return nothing
end

"""
    h0_h1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64}, h1vals::AbstractVector{ComplexF64}, plans0::AbstractVector{ChebHankelPlanH}, plans1::AbstractVector{ChebHankelPlanH}, pidx::Int32, t::Float64, r::Float64)

Evaluate `H₀^(1)(k_m r)` and `H₁^(1)(k_m r)` for all supplied wavenumbers at
one fixed physical radius.

## Arguments
* `h0vals::AbstractVector{ComplexF64}`: Output storage for `H₀^(1)(k_m r)`.
* `h1vals::AbstractVector{ComplexF64}`: Output storage for `H₁^(1)(k_m r)`.
* `plans0::AbstractVector{ChebHankelPlanH}`: Order-zero Hankel plans.
* `plans1::AbstractVector{ChebHankelPlanH}`: Order-one Hankel plans.
* `pidx::Int32`: Precomputed common radial panel index.
* `t::Float64`: Precomputed local Chebyshev coordinate.
* `r::Float64`: Common physical radius.

## Returns
* `nothing`: Results are written to `h0vals` and `h1vals` in place.
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
    h1_j1_multi_ks_at_r!(h1vals::AbstractVector{ComplexF64}, j1vals::AbstractVector{ComplexF64}, plans1::AbstractVector{ChebHankelPlanH}, plansj1::AbstractVector{ChebJPlan}, pidx_h::Int32, t_h::Float64, pidx_j::Int32, t_j::Float64, r::Float64)

Evaluate `H₁^(1)(k_m r)` and `J₁(k_m r)` for all supplied wavenumbers at one
fixed physical radius.

## Arguments
* `h1vals::AbstractVector{ComplexF64}`: Output storage for `H₁^(1)(k_m r)`.
* `j1vals::AbstractVector{ComplexF64}`: Output storage for `J₁(k_m r)`.
* `plans1::AbstractVector{ChebHankelPlanH}`: Order-one Hankel plans.
* `plansj1::AbstractVector{ChebJPlan}`: Order-one Bessel plans.
* `pidx_h::Int32`: Hankel-plan panel index.
* `t_h::Float64`: Hankel-plan local Chebyshev coordinate.
* `pidx_j::Int32`: Bessel-plan panel index.
* `t_j::Float64`: Bessel-plan local Chebyshev coordinate.
* `r::Float64`: Common physical radius.

## Returns
* `nothing`: Results are written to `h1vals` and `j1vals` in place.
"""
@inline function h1_j1_multi_ks_at_r!(h1vals::AbstractVector{ComplexF64}, j1vals::AbstractVector{ComplexF64}, plans1::AbstractVector{ChebHankelPlanH}, plansj1::AbstractVector{ChebJPlan}, pidx_h::Int32, t_h::Float64, pidx_j::Int32, t_j::Float64, r::Float64)
    eval_h_multi_ks!(h1vals, plans1, r, pidx_h, t_h)
    eval_j_multi_ks!(j1vals, plansj1, pidx_j, t_j, r)
    return nothing
end

"""
    h0_h1_j0_j1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64}, h1vals::AbstractVector{ComplexF64}, j0vals::AbstractVector{ComplexF64}, j1vals::AbstractVector{ComplexF64}, plans0::AbstractVector{ChebHankelPlanH}, plans1::AbstractVector{ChebHankelPlanH}, plansj0::AbstractVector{ChebJPlan}, plansj1::AbstractVector{ChebJPlan}, pidx_h::Int32, t_h::Float64, pidx_j::Int32, t_j::Float64, r::Float64)

Evaluate `H₀^(1)(k_m r)`, `H₁^(1)(k_m r)`, `J₀(k_m r)`, and `J₁(k_m r)`
for all supplied wavenumbers at one fixed physical radius.

## Arguments
* `h0vals::AbstractVector{ComplexF64}`: Output storage for `H₀^(1)(k_m r)`.
* `h1vals::AbstractVector{ComplexF64}`: Output storage for `H₁^(1)(k_m r)`.
* `j0vals::AbstractVector{ComplexF64}`: Output storage for `J₀(k_m r)`.
* `j1vals::AbstractVector{ComplexF64}`: Output storage for `J₁(k_m r)`.
* `plans0::AbstractVector{ChebHankelPlanH}`: Order-zero Hankel plans.
* `plans1::AbstractVector{ChebHankelPlanH}`: Order-one Hankel plans.
* `plansj0::AbstractVector{ChebJPlan}`: Order-zero Bessel plans.
* `plansj1::AbstractVector{ChebJPlan}`: Order-one Bessel plans.
* `pidx_h::Int32`: Hankel-plan panel index.
* `t_h::Float64`: Hankel-plan local Chebyshev coordinate.
* `pidx_j::Int32`: Bessel-plan panel index.
* `t_j::Float64`: Bessel-plan local Chebyshev coordinate.
* `r::Float64`: Common physical radius.

## Returns
* `nothing`: Results are written to the four supplied output vectors in place.
"""
@inline function h0_h1_j0_j1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64}, h1vals::AbstractVector{ComplexF64}, j0vals::AbstractVector{ComplexF64}, j1vals::AbstractVector{ComplexF64}, plans0::AbstractVector{ChebHankelPlanH}, plans1::AbstractVector{ChebHankelPlanH}, plansj0::AbstractVector{ChebJPlan}, plansj1::AbstractVector{ChebJPlan}, pidx_h::Int32, t_h::Float64, pidx_j::Int32, t_j::Float64, r::Float64)
    h0_h1_multi_ks_at_r!(h0vals, h1vals, plans0, plans1, pidx_h, t_h, r)
    eval_j_multi_ks!(j0vals, plansj0, pidx_j, t_j, r)
    eval_j_multi_ks!(j1vals, plansj1, pidx_j, t_j, r)
    return nothing
end

struct ChebRadialLookupCache
    pidx_h::Matrix{Int32}
    t_h::Matrix{Float64}
    pidx_j::Matrix{Int32}
    t_j::Matrix{Float64}
end

function ChebRadialLookupCache(G::BoundaryGeomCache, plan_h::ChebHankelPlanH, plan_j::ChebJPlan; multithreaded::Bool = true)
    N = size(G.R, 1)
    pidx_h = Matrix{Int32}(undef, N, N); t_h = Matrix{Float64}(undef, N, N)
    pidx_j = Matrix{Int32}(undef, N, N); t_j = Matrix{Float64}(undef, N, N)
    @use_threads multithreading = (multithreaded && N >= 32) for j in 2:N
        @inbounds for i in 1:j-1
            r = Float64(G.R[i, j])
            ph, th = panel_t(plan_h, r); pj, tj = panel_t(plan_j, r)
            pidx_h[i, j] = ph; pidx_h[j, i] = ph; t_h[i, j] = th; t_h[j, i] = th
            pidx_j[i, j] = pj; pidx_j[j, i] = pj; t_j[i, j] = tj; t_j[j, i] = tj
        end
    end
    return ChebRadialLookupCache(pidx_h, t_h, pidx_j, t_j)
end
