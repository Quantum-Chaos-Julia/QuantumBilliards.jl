#############################################################################
# Piecewise-Chebyshev special-function core for Helmholtz boundary integral
# operators in 2D.
#
# This file provides reusable plan construction and evaluation routines for
#
#     H_ν^(κ)(k r),   J_ν(k r),
#
# where r≥0 is a geometric distance and k may be real or complex.
#
# The special-function layer is separated from geometry and matrix assembly:
#
#     geometry             -> distances, panel lookup, local coordinates,
#     special functions    -> Hankel/Bessel Chebyshev interpolation,
#     operator assembly    -> DLP, DLP-Kress, CFIE-Kress, Alpert, etc.
#
# ---------------------------------------------------------------------------
# CHEBYSHEV REPRESENTATIONS
# ---------------------------------------------------------------------------
#
# Two plan families are used:
#
# 1. ChebHankelPlanH
#
#        H_ν^(κ)(k r),
#
#    an unscaled piecewise-Chebyshev representation of the physical Hankel
#    function. It supports real and complex k and is shared by the ordinary
#    DLP, DLP-Kress, CFIE-Kress, and Alpert pathways.
#
# 2. ChebJPlan
#
#        J_ν(k r),
#
#    an unscaled piecewise-Chebyshev representation of the Bessel-J function.
#    These plans are needed by the Kress logarithmic split. For complex k,
#    J_ν(k r) must be evaluated independently and cannot in general be replaced
#    by real(H_ν^(1)(k r)).
#
# Each radial interval [rmin,rmax] is divided into uniform panels. On every
# panel [a,b], the special function is interpolated at Chebyshev-Lobatto nodes
#
#     t_j=cos(πj/M),   j=0,...,M,
#
# mapped to
#
#     r_j=(a+b)/2+(b-a)t_j/2.
#
# Evaluation uses the stored Chebyshev coefficients and Clenshaw recurrence.
#
# ---------------------------------------------------------------------------
# SMALL-ARGUMENT STRATEGY
# ---------------------------------------------------------------------------
#
# Hankel functions are singular at z=k r=0, so very small arguments are not
# handled solely by polynomial interpolation.
#
# The evaluation hierarchy is
#
#     |z| < hankel_z_chebyshev_cutoff_small_z
#         -> explicit small-z H₀/H₁ series,
#
#     hankel_z_chebyshev_cutoff_small_z ≤ |z|
#       < hankel_z_chebyshev_cutoff
#         -> direct special-function evaluation,
#
#     otherwise
#         -> Chebyshev interpolation.
#
# Higher-level geometry caches may additionally use
#
#     pidx==0
#
# to indicate that a distance lies below the interpolated Hankel interval.
# Such entries are evaluated by the direct/low-z fallback rather than by a
# Chebyshev panel.
#
# This is particularly useful for quadratures such as Alpert correction rules,
# where some auxiliary off-grid distances can become very small.
#
# ---------------------------------------------------------------------------
# PANELIZATION AND GEOMETRIC PRECOMPUTATION
# ---------------------------------------------------------------------------
#
# For an interpolated radius r, the geometry layer stores
#
#     pidx    panel index,
#     t       local Chebyshev coordinate in [-1,1].
#
# The helpers
#
#     panel_indices
#     precompute_geom
#     panel_and_geom
#
# precompute these k-independent quantities. Uniform panels allow O(1) lookup
#
#     p=floor((r-rmin)/dr)+1.
#
# For compatibility with existing matrix-assembly code, `precompute_geom` and
# `panel_and_geom` continue to return
#
#     invsqrt=1/sqrt(r)
#
# in addition to the panel data. The current unscaled Hankel representation does
# not require this factor, but retaining it avoids changing established caller
# interfaces in the Kress and related Chebyshev backends.
#
# ---------------------------------------------------------------------------
# MULTI-k EVALUATION
# ---------------------------------------------------------------------------
#
# Boundary-integral assembly frequently evaluates one fixed distance r for many
# wavenumbers k_m. The multi-k evaluators therefore fill preallocated buffers
# without allocation inside the matrix-entry loops.
#
# Core evaluators include
#
#     eval_h_multi_ks!
#     eval_j_multi_ks!
#     h0_h1_multi_ks_at_r!
#     h0_h1_j0_j1_multi_ks_at_r!
#     h1_j1_multi_ks_at_r!
#     h1_multi_ks_at_r!
#
# The required special functions depend on the operator:
#
#     ordinary DLP value          H₁
#     ordinary DLP derivatives    H₀,H₁ (and currently H₂ by recurrence)
#     DLP-Kress value             H₁,J₁
#     DLP-Kress derivatives       H₀,H₁,J₀,J₁
#     CFIE-Kress                  H₀,H₁,J₀,J₁.
#
# For the ordinary DLP value path, `h1_multi_ks_at_r!` and `h1_at_r` are thin
# wrappers around the same unscaled Hankel evaluator used elsewhere.
#
# ---------------------------------------------------------------------------
# THREADING / PERFORMANCE
# ---------------------------------------------------------------------------
#
# Chebyshev tables are constructed independently panel-by-panel and therefore
# use threading. Vectorized geometry and evaluation routines are also threaded
# where appropriate.
#
# The dominant costs are generally controlled by
#
#     number of radial panels,
#     Chebyshev degree per panel,
#     number of special-function families,
#     number of matrix-entry evaluations.
#
# The low-z switching branch is negligible compared with direct Hankel/Bessel
# evaluation or a Clenshaw recurrence.
#
# ---------------------------------------------------------------------------
# PUBLIC / SEMI-PUBLIC API
# ---------------------------------------------------------------------------
#
# Plan builders
#     plan_h
#     plan_j
#
# Geometry helpers
#     _find_panel
#     panel_indices
#     precompute_geom
#     panel_and_geom
#
# Scalar evaluators
#     eval_h! / eval_h
#     eval_j! / eval_j
#     h1_at_r
#
# Multi-k evaluators
#     eval_h_multi_ks!
#     eval_j_multi_ks!
#     h0_h1_multi_ks_at_r!
#     h0_h1_j0_j1_multi_ks_at_r!
#     h1_j1_multi_ks_at_r!
#     h1_multi_ks_at_r!
#     h0_h1_h2_multi_ks_at_r!
#
# Low-z support
#     _small_h0_series
#     _small_h1_series
#############################################################################

const γ=MathConstants.eulergamma
const hankel_z_chebyshev_cutoff_small_z=0.001 # for z=k*r below this we use the small-argument series expansions for the Hankel functions instead of the Chebyshev evaluation, since the Chebyshev approximation is not accurate near zero due to the singularity. This is a bit hacky but it works and is fast since we only need to evaluate a few terms in the series expansion for small z. We can afford to be conservative here since this only affects a small portion of the domain near r=0, and we want to ensure high accuracy there.
const hankel_z_chebyshev_cutoff=0.2 # this is the region between [hankel_z_chebyshev_cutoff_small_z, hankel_z_chebyshev_cutoff] where we switch from the small-argument series expansions to the direct evaluation since only 12th degree small z poly.
#TODO Differentiation of Chebyshev polynomials to get other orders of hankels. Is this even good to do for performance, how much does it really matter?

struct ChebHankelTableH
    a::Float64 # start of panel 
    b::Float64 # end of panel
    M::Int # order of the chebyshev polynomial for panel [a,b]
    ν::Int # degree of the Hankel type function
    κ::Int # order/type of the Hankel function 
    c::Vector{ComplexF64} 
end

struct ChebJTable
    a::Float64 # start of panel 
    b::Float64 # end of panel
    M::Int # order of the chebyshev polynomial for panel [a,b]
    ν::Int # degree of the J
    c::Vector{ComplexF64} 
end

# =============================================================================
# Construct a Chebyshev table on a single panel [a,b] for the unscaled Hankel:
#   Hν(z) = H_ν^(κ)(z),  with  z = k*r and k real.
#
# We approximate the function Fν(r) = H_ν^(κ)(k r) on Chebyshev nodes t_j = cos(π j / M), j=0..M, mapped to r_j ∈ [a,b].
#
# Unlike the complex-k scaled route, this real-k path stores the unscaled Hankel
# values directly.
#
# Inputs
#   ν :: Int                 # Hankel order (typically 0 or 1)
#   κ :: Int                 # Hankel type (1 or 2 etc.)
#   k :: Float64             # fixed real wavenumber
#   a,b :: Float64           # panel endpoints, 0 < a < b
#   M :: Int                 # Chebyshev degree (stores M+1 coeffs)
#
# Output
#   ChebHankelTableH(a,b,M,ν,c) where c are the Chebyshev coeffs of Fν.
# =============================================================================
function _build_table_h!(ν::Int,κ::Int,k::Float64,a::Float64,b::Float64;M::Int=16)::ChebHankelTableH
    @assert a>0 && b>a "a=$(a), b=$(b)" # sanity check to keep the interval bounded above 0 and to not be degenerate/reversed
    f1=Vector{ComplexF64}(undef,M+1) # preallocate the vector storing function evalutions at chebyshev nodes
    @inbounds for j in 0:M
        t=cospi(j/M) # chebyshev node in [-1,1]
        r=((b+a)+(b-a)*t)/2 # affine map to [a,b] so we are in the correct sector
        z=k*r # argument for Hankel in local coordinates
        f1[j+1]=ComplexF64(Bessels.besselh(ν,κ,z)) # unscaled real-argument Hankel from Bessels.jl
    end
    c=Vector{ComplexF64}(undef,M+1) # preallocate chebyshev coeffs
    _chebfit!(c,f1) # fit the chebyshev coeffs to the chebyshev node evaluations
    return ChebHankelTableH(a,b,M,ν,κ,c) # construct the table for that particular panel with local cheby polynomial 
end

# multiple dispatch version of above with complex k. Maybe redundant since we have the scaled version but this one does not need the exp mul in the end and therefore faster. Other one is LEGACY made for original formulation for Beyn's method.
function _build_table_h!(ν::Int,κ::Int,k::ComplexF64,a::Float64,b::Float64;M::Int=16)::ChebHankelTableH
    @assert a>0 && b>a "a=$(a), b=$(b)" # sanity check to keep the interval bounded above 0 and to not be degenerate/reversed
    f1=Vector{ComplexF64}(undef,M+1) # preallocate the vector storing function evalutions at chebyshev nodes
    @inbounds for j in 0:M
        t=cospi(j/M) # chebyshev node in [-1,1]
        r=((b+a)+(b-a)*t)/2 # affine map to [a,b] so we are in the correct sector
        z=k*r # argument for Hankel in local coordinates
        f1[j+1]=SpecialFunctions.besselh(ν,κ,z) # unscaled complex-argument Hankel from AMOS. We use this for complex k since Bessels.jl doesn't support complex arguments and AMOS is more stable for complex arguments. Just dont use this here with large im part for z !!! 
        #FIXME It is hacky since for _build_table_h1x! we used the scaled version primarily for Beyn's method since we did not know the size of the imaginary part, but this seems not safe
    end
    c=Vector{ComplexF64}(undef,M+1) # preallocate chebyshev coeffs
    _chebfit!(c,f1) # fit the chebyshev coeffs to the chebyshev node evaluations
    return ChebHankelTableH(a,b,M,ν,κ,c) # construct the table for that particular panel with local cheby polynomial 
end

function _build_table_j!(ν::Int,k::Float64,a::Float64,b::Float64;M::Int=16)::ChebJTable
    @assert a>=0 && b>a "a=$(a), b=$(b)" 
    f1=Vector{ComplexF64}(undef,M+1)
    @inbounds for j in 0:M
        t=cospi(j/M) 
        r=((b+a)+(b-a)*t)/2 
        z=k*r 
        f1[j+1]=Bessels.besselj(ν,z) 
    end
    c=Vector{ComplexF64}(undef,M+1) 
    _chebfit!(c,f1) 
    return ChebJTable(a,b,M,ν,c)
end

function _build_table_j!(ν::Int,k::ComplexF64,a::Float64,b::Float64;M::Int=16)::ChebJTable
    @assert a>=0 && b>a "a=$(a), b=$(b)" 
    f1=Vector{ComplexF64}(undef,M+1)
    @inbounds for j in 0:M
        t=cospi(j/M) 
        r=((b+a)+(b-a)*t)/2 
        z=k*r 
        f1[j+1]=SpecialFunctions.besselj(ν,z)
    end
    c=Vector{ComplexF64}(undef,M+1) 
    _chebfit!(c,f1) 
    return ChebJTable(a,b,M,ν,c)
end

struct ChebHankelPlanH
    k::Union{Float64,ComplexF64}
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
    k::Union{Float64,ComplexF64}
    ν::Int
    panels::Vector{ChebJTable}
    rmin::Float64
    rmax::Float64
    dr::Float64
    invdr::Float64
    npanels::Int
end

# =============================================================================
# Build a piecewise-Chebyshev plan for the unscaled Hankel
#   H_ν^(1)(k r)
# over r ∈ [rmin,rmax], with k real.
#
# The interval is split into `npanels` uniformly.
# Each panel stores M+1 Chebyshev coefficients (degree M) for the direct,
# unscaled Hankel values. This path is intended for real-k applications such as
# CFIE/BIM, where the exponential scaling used in the complex-k route is not
# needed.
#
# Inputs
#   ν :: Int
#       # Hankel order (typically 0 for SLP or 1 for DLP)
#   κ :: Int
#       # Hankel type (1 for H^(1), 2 for H^(2), etc.)
#   k :: Float64
#       # fixed real wavenumber
#   rmin,rmax :: Float64
#       # 0 < rmin < rmax
#   npanels :: Int
#       # number of panels
#   M :: Int order of the Chebyshev polynomial for each panel
#
# Output
#   ChebHankelPlanH(κ,k,ν,panels,...) containing the panelized Chebyshev tables.
# =============================================================================
function plan_h(ν::Int,κ::Int,k::Union{Float64,ComplexF64},rmin::Float64,rmax::Float64;npanels::Int=64,M::Int=16)::ChebHankelPlanH
    @assert rmin>0 && rmax>rmin
    br=_breaks_uniform(rmin,rmax,npanels) 
    panels=Vector{ChebHankelTableH}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_h!(ν,κ,k,br[i],br[i+1];M=M)
    end
    dr=(rmax-rmin)/npanels
    invdr=inv(dr)
    return ChebHankelPlanH(k,ν,κ,panels,rmin,rmax,dr,invdr,npanels)
end

# =============================================================================
# Build a piecewise-Chebyshev plan for the Bessel function J_ν(k r) over r ∈ [rmin,rmax].
#
# The interval is split into `npanels` uniformly. 
# Each panel stores M+1 Chebyshev coefficients (degree M) for the Bessel function values.
# This is intended for use in the CFIE when evaluating the interior Dirichlet Green’s function, which involves J_ν(k r).
# 
# Inputs
#   ν :: Int
#       # Bessel order need both 0 and 1 for CFIE
#   k :: Union{Float64,ComplexF64}
#       # fixed wavenumber (real or complex)
#   rmin,rmax :: Float64
#       # 0 < rmin < rmax
#   npanels :: Int
#       # number of panels
#   M :: Int order of the Chebyshev polynomial for each panel

# Output
#   ChebJPlan(k,ν,panels,...) containing the panelized Chebyshev tables for J_ν(k r).
# =============================================================================
function plan_j(ν::Int,k::Union{Float64,ComplexF64},rmin::Float64,rmax::Float64;npanels::Int=64,M::Int=16)::ChebJPlan
    @assert rmin>=0 && rmax>rmin
    br=_breaks_uniform(rmin,rmax,npanels) 
    panels=Vector{ChebJTable}(undef,npanels)
    @inbounds Threads.@threads for i in 1:npanels
        panels[i]=_build_table_j!(ν,k,br[i],br[i+1];M=M)
    end
    dr=(rmax-rmin)/npanels 
    invdr=inv(dr)
    return ChebJPlan(k,ν,panels,rmin,rmax,dr,invdr,npanels)
end

# LEGACY SINCE WE NOW HAVE CUTOFFS FOR SMALL Z
# =============================================================================
# Locate the panel index p such that panels[p].a ≤ r ≤ panels[p].b.
#
# This is a scalar binary search used to identify which Chebyshev panel
# contains the given radius `r`.
#
# Inputs
#   panels :: Union{Vector{ChebHankelTableH},Vector{ChebJTable}}   # vector of panel structs
#   r      :: Float64                      # query radius (must be positive)
#
# Output
#   Int                                 # index p such that r ∈ [panels[p].a, panels[p].b]
#
# Behavior
#   - If r lies within the covered domain, return its panel index.
#   - If r is outside the plan’s total range, an error is thrown.
#
# Implementation details
#   - O(log Np) comparisons.
#   - @inbounds avoids bounds checks for speed.
#   - Typical Np = 50–300, so this function is negligible cost compared to Hankel evaluation.
# =============================================================================
@inline function _find_panel_binary(panels::Union{Vector{ChebHankelTableH},Vector{ChebJTable}},r::Float64)::Int
    lo=1;hi=length(panels)
    @inbounds while lo≤hi
        mid=(lo+hi)>>>1
        P=panels[mid]
        if r<P.a
            hi=mid-1
        elseif r>P.b
            lo=mid+1
        else
            return mid
        end
    end
    error("r=$r outside plan range")
end

# =============================================================================
# Fast panel lookup for uniformly graded Chebyshev plans.
#
# Panels are equally spaced over [rmin, rmax], so the panel index can be obtained in O(1) time using a
# direct arithmetic mapping instead of a binary search.
#
# The mapping is:
#   p = floor((r - rmin) / dr) + 1
# where dr = (rmax - rmin) / npanels.
#
# Inputs
#   pl :: Union{ChebHankelPlanH,ChebJPlan}
#          # plan with uniform panel spacing; must contain fields rmin, invdr, npanels
#   r  :: Float64
#          # query radius (must lie within [pl.rmin, pl.rmax])
#
# Output
#   Int
#          # panel index p such that r lies in panel p
#
# Behavior
#   - Returns a clamped index in [1, pl.npanels].
#   - No bounds error is thrown; values outside range are projected.
#
# Implementation details
#   - O(1) cost: one multiply, one floor, and clamping.
# =============================================================================

@inline function _find_panel_uniform(pl::Union{ChebHankelPlanH,ChebJPlan},r::Float64)::Int
    p=Int(floor((r-pl.rmin)*pl.invdr))+1
    return ifelse(p<1,1,ifelse(p>pl.npanels,pl.npanels,p))
end

# =============================================================================
# Unified panel lookup for Chebyshev plans.
#
#  - O(1) arithmetic lookup for uniform panels
#
# This function provides a consistent interface for locating the panel index
# p such that r ∈ [panels[p].a, panels[p].b].
#
# Inputs
#   pl :: Union{ChebHankelPlanH,ChebJPlan}
#          # Chebyshev plan containing panels
#   r  :: Float64
#          # query radius (must lie within plan range)
#
# Output
#   Int
#   # panel index p corresponding to r
# =============================================================================
@inline function _find_panel(pl::Union{ChebHankelPlanH,ChebJPlan},r::Float64)::Int
    return _find_panel_uniform(pl,r)
end

# =============================================================================
# Vectorized panel search: determine for each r[i] ∈ rvec which panel
# [a,b] of the Chebyshev plan `pl` contains it.
#
# This creates an Int32 vector of panel indices to enable later vectorized
# evaluation of H₁.
#
# Inputs
#   pl    :: Union{ChebHankelPlanH,ChebJPlan}
#             # plan containing `panels::Union{Vector{ChebHankelTableH},Vector{ChebJTable}` 
#   rvec  :: AbstractVector{Float64}
#             # radii (each must satisfy pl.panels[1].a ≤ r ≤ pl.panels[end].b)
#
# Output
#   pidx  :: Vector{Int32}
#             # pidx[i] = index of panel that contains rvec[i]
# =============================================================================
function panel_indices(pl::Union{ChebHankelPlanH,ChebJPlan},rvec::AbstractVector{Float64})::Vector{Int32}
    p=similar(rvec,Int32)
    @inbounds Threads.@threads for i in eachindex(rvec)
        p[i]=Int32(_find_panel(pl,rvec[i]))
    end
    return p
end

# =============================================================================
# Precompute Chebyshev coordinates and normalization factors for the given
# radii rvec, to enable fast, k-independent evaluation of H₁.
#
# Inputs
#   pl     :: Union{ChebHankelPlanH,ChebJPlan}  # plan containing panels [a,b]
#   rvec   :: AbstractVector{Float64}  # radii (must lie in total range of plan)
#   pidx   :: AbstractVector{Int32}    # panel indices for each r[i]
#
# Outputs
#   (t, invsqrt)
#       t        :: Vector{Float64}     # mapped Chebyshev coordinates
#       invsqrt  :: Vector{Float64}     # 1/√r for normalization
# =============================================================================
function precompute_geom(pl::Union{ChebHankelPlanH,ChebJPlan},rvec::AbstractVector{Float64},pidx::AbstractVector{Int32})::Tuple{Vector{Float64},Vector{Float64}}
    @assert length(rvec)==length(pidx)
    n=length(rvec)
    t=Vector{Float64}(undef,n)
    invsqrt=Vector{Float64}(undef, n)
    pans=pl.panels
    @inbounds Threads.@threads for i in eachindex(rvec)
        rr=rvec[i]
        P=pans[pidx[i]]
        t[i]=(2*rr-(P.b+P.a))/(P.b-P.a) # aff_map_inv
        invsqrt[i]=inv(sqrt(rr))
    end
    return t, invsqrt
end

# =============================================================================
# Single-pass, threaded precompute of:
#   - pidx[i]   : panel index for rvec[i]
#   - t[i]      : mapped Chebyshev coordinate in [-1,1] on that panel
#   - invsqrt[i]: 1 / √(rvec[i])
#
# This merges the work of `panel_indices` and `precompute_geom` to avoid
# an extra pass over rvec and redundant memory traffic.
#
# Inputs
#   pl   :: Union{ChebHankelPlanH,ChebJPlan}  # Chebyshev plan with panels
#   rvec :: Vector{Float64}  # radii (must lie in [panels[1].a, panels[end].b])
#
# Outputs
#   pidx    :: Vector{Int32}
#   t       :: Vector{Float64}
#   invsqrt :: Vector{Float64}
#
# Notes
#   - Uses uniform O(1) arithmetic lookup per element.
#   - Forces no allocations inside the threaded loop.
# =============================================================================
function panel_and_geom(pl::Union{ChebHankelPlanH,ChebJPlan},rvec::AbstractVector{Float64})::Tuple{Vector{Int32},Vector{Float64},Vector{Float64}}
    n=length(rvec)
    pidx=Vector{Int32}(undef,n)
    t=Vector{Float64}(undef,n)
    invsqrt=Vector{Float64}(undef,n)
    rmin=pl.rmin
    dr=pl.dr
    invdr=pl.invdr
    np=pl.npanels
    @inbounds Threads.@threads for i in eachindex(rvec)
        r=rvec[i]
        p=Int(floor((r-rmin)*invdr))+1
        p=ifelse(p<1,1,ifelse(p>np,np,p))
        pidx[i]=Int32(p)
        center=rmin+(p-0.5)*dr
        t[i]=2*(r-center)*invdr
        invsqrt[i]=inv(sqrt(r))
    end
    return pidx,t,invsqrt
end

##################################################################
############## NEAR 0 EXPANSIONS FOR H0 AND H1 ###################
##################################################################

# For small arguments, the Hankel functions can be approximated by their series expansions. These are used to handle near-singular cases where the argument z = k*r is close to zero, which can cause numerical instability in the chebyshev evaluation of the Hankel functions.
# Up to O(z^12) for both, hopefully with defaul cutoff 1e-3 this is good enough for near machine precision.

@inline function _small_h0_series(z::ComplexF64)
    zz=z*z
    P=2123366400+zz*(-530841600+zz*(33177600+zz*(-921600+zz*(14400+zz*(-144+zz)))))
    Q=10616832000+zz*(-995328000+zz*(33792000+zz*(-600000+zz*(6576+zz*(-49)))))
    return (((10*pi+20*im*γ)*P+im*zz*Q)/(21233664000*pi))+(im*P/(1061683200*pi))*log(z/2)
end

@inline function _small_h0_series(z::T) where T<:Number
    return _small_h0_series(ComplexF64(z))
end

@inline function _small_h1_series(z::ComplexF64)
    zz=z*z
    A=-4161798144000+
    zz*(1040449536000*(-1+2*γ-1im*pi)+
    zz*(-65028096000*(-5+4*γ-2im*pi)+
    zz*(1806336000*(-10+6*γ-3im*pi)+
    zz*(-9408000*(-47+24*γ-12im*pi)+
    zz*(47040*(-131+60*γ-30im*pi)+
    zz*(-784*(-71+30*γ-15im*pi)+
    zz*(-353+140*γ-70im*pi)))))))
    R=14863564800+zz*(-1857945600+zz*(77414400+zz*(-1612800+zz*(20160+zz*(-168+zz)))))
    return (im*A/(2080899072000*pi*z))+(im*z*R/(14863564800*pi))*log(z/2)
end

@inline function _small_h1_series(z::T) where T<:Number
    return _small_h1_series(ComplexF64(z))
end

##################################################################
###################### EVALUATION FUNCTIONS ######################
##################################################################

# =============================================================================
# Fast evaluation of the unscaled Hankel for the real-k plan:
#   H_ν^(κ)(k r),  with k real.
#
# Each panel stores Chebyshev coefficients of the direct unscaled Hankel
# values on that interval. No exponential scaling and no 1/sqrt(r)
# normalization are used in this real-k route.
#
# Inputs
#   H1   :: AbstractVector{ComplexF64}
#           # output
#   pl   :: ChebHankelPlanH
#           # plan at fixed real k
#   r    :: AbstractVector{Float64}
#           # radii (unused; kept for API compatibility)
#   pidx :: AbstractVector{Int32}
#           # per-point panel index
#   t    :: AbstractVector{Float64}
#           # per-point Chebyshev coordinate
#
# Output
#   Fills H1 in place with H_ν^(κ)(k r_i).
# =============================================================================
function eval_h!(H1::AbstractVector{ComplexF64},pl::ChebHankelPlanH,r::AbstractVector{Float64},pidx::AbstractVector{Int32},t::AbstractVector{Float64})
    pans=pl.panels
    k=ComplexF64(pl.k)
    ν=pl.ν
    @inbounds Threads.@threads for i in eachindex(r)
        z=k*r[i]
        if pidx[i]==0
            if ν==0
                if abs(z)<hankel_z_chebyshev_cutoff_small_z
                    H1[i]=_small_h0_series(z)
                else
                    H1[i]=SpecialFunctions.besselh(0,pl.κ,z)
                end
            elseif ν==1
                if abs(z)<hankel_z_chebyshev_cutoff_small_z
                    H1[i]=_small_h1_series(z)
                else
                    H1[i]=SpecialFunctions.besselh(1,pl.κ,z)
                end
            else
                H1[i]=SpecialFunctions.besselh(ν,pl.κ,z)
            end
        else
            T=pans[pidx[i]]
            H1[i]=_cheb_clenshaw(T.c,t[i])
        end
    end
    return nothing
end

# =============================================================================
# Fast evaluation of the Bessel function J_ν(k r) for the same many radii r across
# for a single plan.
#
# Inputs
#   J     :: AbstractVector{ComplexF64}
#           # output vector (length = length(plans))   
#   pl    :: ChebJPlan
#           # Chebyshev plan for J_ν(k r) at fixed k
#   r     :: AbstractVector{Float64}
#           # radii (unused; kept for API compatibility)
#   pidx  :: AbstractVector{Int32}
#           # per-point panel index
#   t     :: AbstractVector{Float64}
#           # per-point Chebyshev coordinate
# Output
#   Fills J in place with J_ν(k r_i) for each plan.
# =============================================================================
function eval_j!(J::AbstractVector{ComplexF64},pl::ChebJPlan,r::AbstractVector{Float64},pidx::AbstractVector{Int32},t::AbstractVector{Float64})
    pans=pl.panels
    ν=pl.ν
    k=ComplexF64(pl.k)
    @inbounds Threads.@threads for i in eachindex(r)
        if pidx[i]==0
            J[i]=SpecialFunctions.besselj(ν,k*r[i])
        else
            T=pans[pidx[i]]
            J[i]=_cheb_clenshaw(T.c,t[i])
        end
    end
    return nothing
end

# =============================================================================
# Evaluate the unscaled Hankel
#   H_ν^(κ)(k r),  with k real,
# at a single point using the real-k Chebyshev plan.
#
# Inputs
#   pl   :: ChebHankelPlanH
#           # precomputed Chebyshev plan for fixed real k
#   pidx :: Int32
#           # panel index such that r ∈ [a,b]
#   t    :: Float64
#           # mapped Chebyshev coordinate in [-1,1]
#   r    :: Float64
#           # radius for this evaluation point - for small z asymptotics
# Output
#   ComplexF64
#           # approximation of H_ν^(κ)(k r)
# =============================================================================
@inline function eval_h(pl::ChebHankelPlanH,pidx::Int32,t::Float64,r::Float64)
    z=ComplexF64(pl.k)*r
    if pidx==0
        if pl.ν==0
            if abs(z)<hankel_z_chebyshev_cutoff_small_z
                return _small_h0_series(z)
            else
                return SpecialFunctions.besselh(0,pl.κ,z)
            end
        elseif pl.ν==1
            if abs(z)<hankel_z_chebyshev_cutoff_small_z
                return _small_h1_series(z)
            else
                return SpecialFunctions.besselh(1,pl.κ,z)
            end
        else
            return SpecialFunctions.besselh(pl.ν,pl.κ,z)
        end
    end
    return _cheb_clenshaw(pl.panels[pidx].c,t)
end

# =============================================================================
# Evaluate the Bessel function J_ν(k r) for a single point using the Chebyshev plan for J.
#
# Inputs
#   pl   :: ChebJPlan
#           # precomputed Chebyshev plan for J_ν(k r) at fixed k
#   pidx :: Int32
#           # panel index such that r ∈ [a,b]
#   t    :: Float64
#           # mapped Chebyshev coordinate in [-1,1]
# Output
#   ComplexF64
#           # approximation of J_ν(k r)
# =============================================================================
@inline function eval_j(pl::ChebJPlan,pidx::Int32,t::Float64,r::Float64)
    return _cheb_clenshaw(pl.panels[pidx].c,t)
end

# =============================================================================
# Evaluate the unscaled Hankel H_ν^(κ)(k r) for the same radius r across
# multiple real wavenumbers (one per plan).
#
# Each plan stores the direct unscaled Hankel coefficients on the same panel
# partition, so the same `pidx` and `t` may be reused across all plans.
#
# Inputs
#   out  :: AbstractVector{ComplexF64}
#           # length == length(plans)
#   plans:: AbstractVector{ChebHankelPlanH}
#           # real-k Chebyshev plans
#   r    :: Float64
#           # radius for this evaluation point (unused; kept for API symmetry)
#   pidx :: Int32
#           # panel index for r
#   t    :: Float64
#           # Chebyshev coordinate for r
#
# Output
#   out[m] = H_ν^(κ)(k_m r) for m = 1..length(plans).
# =============================================================================
function eval_h_multi_ks!(out::AbstractVector{ComplexF64},plans::AbstractVector{ChebHankelPlanH},r::Float64,pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans)
        plan_m=plans[m]
        z=ComplexF64(plan_m.k)*r
        if pidx==0
            if plan_m.ν==0
                if abs(z)<hankel_z_chebyshev_cutoff_small_z
                    out[m]=_small_h0_series(z)
                else
                    out[m]=SpecialFunctions.besselh(0,plan_m.κ,z)
                end
            elseif plan_m.ν==1
                if abs(z)<hankel_z_chebyshev_cutoff_small_z
                    out[m]=_small_h1_series(z)
                else
                    out[m]=SpecialFunctions.besselh(1,plan_m.κ,z)
                end
            else
                out[m]=SpecialFunctions.besselh(plan_m.ν,plan_m.κ,z)
            end
        else
            out[m]=_cheb_clenshaw(plan_m.panels[pidx].c,t)
        end
    end
    return nothing
end

# =============================================================================
# Evaluate the Bessel function J_ν(k r) for the same radius r across multiple
# wavenumbers (one per plan).
#
# Inputs
#   out   :: AbstractVector{ComplexF64} : Output buffer.
#   plans :: AbstractVector{ChebJPlan} : Collection of Chebyshev plans for J_ν(k_m r), one plan per wavenumber.
#   pidx  :: Int32 : Panel index for the J plans, determined by r and the first plan's panelization.
#   t     :: Float64 : Mapped Chebyshev coordinate for r in the identified panel.
#
# Output
#   Fills: out[m] = J_ν(k_m r) for all stored wavenumbers.
# =============================================================================
@inline function eval_j_multi_ks!(out::AbstractVector{ComplexF64},plans::AbstractVector{ChebJPlan},pidx::Int32,t::Float64)
    @inbounds for m in eachindex(plans)
        out[m]=_cheb_clenshaw(plans[m].panels[pidx].c,t)
    end
    return nothing
end

# Locate the radial panel and Chebyshev coordinate for both the H and J plans. Used primarily in the default dlp chebyhsev construction.
@inline function panel_t(pl::Union{ChebHankelPlanH,ChebJPlan},r::Float64)
    if r<pl.rmin
        return Int32(0),0.0
    end
    p=_find_panel(pl,r)
    P=pl.panels[p]
    return Int32(p),(2*r-(P.b+P.a))/(P.b-P.a)
end

############################################################
##################### HIGH LEVEL API #######################
############################################################

"""
    h0_h1_j0_j1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},j0vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},plansj0::AbstractVector{ChebJPlan},plansj1::AbstractVector{ChebJPlan},pidx_h::Int32,t_h::Float64,r::Float64)

Evaluate `H₀^(1)`, `H₁^(1)`, `J₀`, and `J₁` for all wavenumbers at one fixed
distance panel/location, writing the results in place.

This function is the small inner evaluator used by the multi-`k` same-component
CFIE-Kress assembly. The geometric pair `(i,j)` has already been mapped to:
- a Chebyshev panel index `pidx`,
- a local coordinate `t ∈ [-1,1]`.

For each wavenumber `k_m`, the function interpolates:
- `H₀^(1)(k_m r)`
- `H₁^(1)(k_m r)`
- `J₀(k_m r)`
- `J₁(k_m r)`

by evaluating the corresponding Chebyshev expansions with Clenshaw recurrence.
For real wavenumbers one may sometimes identify `J_n` with `real(H_n^(1))`,
but for complex `k` that is not valid. The Kress split formulas genuinely
require `J₀` and `J₁`, so they are interpolated separately.

# Arguments
- `h0vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₀^(1)` values.
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `j0vals::AbstractVector{ComplexF64}`:
  Output vector for the `J₀` values.
- `j1vals::AbstractVector{ComplexF64}`:
  Output vector for the `J₁` values.
- `plans0::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₀^(1)`.
- `plans1::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `plansj0::AbstractVector{ChebJPlan}`:
  Chebyshev plans for `J₀`.
- `plansj1::AbstractVector{ChebJPlan}`:
  Chebyshev plans for `J₁`.
- `pidx_h::Int32`:
  Panel index containing the current distance for Hankel functions.
- `t_h::Float64`:
  Local Chebyshev coordinate in that panel for Hankel functions.
- `pidx_j::Int32`:
  Panel index containing the current distance for Bessel functions.
- `t_j::Float64`:
  Local Chebyshev coordinate in that panel for Bessel functions.
- `r::Float64`:
  Physical radius at which all values are evaluated.

# Returns
- `nothing`
"""
@inline function h0_h1_j0_j1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},j0vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},plansj0::AbstractVector{ChebJPlan},plansj1::AbstractVector{ChebJPlan},pidx_h::Int32,t_h::Float64,pidx_j::Int32,t_j::Float64,r::Float64)
    h0_h1_multi_ks_at_r!(h0vals,h1vals,plans0,plans1,pidx_h,t_h,r)
    eval_j_multi_ks!(j0vals,plansj0,pidx_j,t_j)
    eval_j_multi_ks!(j1vals,plansj1,pidx_j,t_j)
    return nothing
end

"""
    h0_h1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64,r::Float64)

Evaluate `H₀^(1)` and `H₁^(1)` for all wavenumbers at one fixed distance
panel/location, writing the results in place.

This is the reduced special-function evaluator used in off-component CFIE-Kress
blocks, where the kernel is smooth and no Kress logarithmic split is needed.
Since the smooth inter-component assembly uses only the Hankel terms, the Bessel
`J₀/J₁` values are not required.

# Arguments
- `h0vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₀^(1)` values.
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `plans0::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₀^(1)`.
- `plans1::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `pidx::Int32`:
  Panel index for the active distance.
- `t::Float64`:
  Local Chebyshev coordinate in that panel.

# Returns
- `nothing`
"""
@inline function h0_h1_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64,r::Float64)
    @inbounds for m in eachindex(plans0)
        z=ComplexF64(plans0[m].k)*r
        az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h0vals[m]=_small_h0_series(z)
            h1vals[m]=_small_h1_series(z)
        elseif az<hankel_z_chebyshev_cutoff || pidx==0
            h0vals[m]=SpecialFunctions.besselh(0,1,z)
            h1vals[m]=SpecialFunctions.besselh(1,1,z)
        else
            h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
            h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
        end
    end
    return nothing
end

"""
        h1_j1_multi_ks_at_r!(h1vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},plans1::AbstractVector{ChebHankelPlanH},plansj1::AbstractVector{ChebJPlan},pidx_h::Int32,t_h::Float64,r::Float64)

Evaluate `H₁^(1)` and `J₁` for all wavenumbers at one fixed distance panel/location, writing the results in place.
This is the reduced special-function evaluator used in off-component CFIE-Kress blocks, where the kernel is smooth and no Kress logarithmic split is needed. Since the smooth inter-component assembly uses only the Hankel terms, the Bessel `J₀` values are not required.

# Arguments
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `j1vals::AbstractVector{ComplexF64}`:
  Output vector for the `J₁` values.
- `plans1::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `plansj1::AbstractVector{ChebJPlan}`:
  Chebyshev plans for `J₁`.
- `pidx_h::Int32`:
  Hankel panel index for the active distance.
- `t_h::Float64`:
  Hankel local Chebyshev coordinate.
- `pidx_j::Int32`:
  J panel index for the active distance.
- `t_j::Float64`:
  J local Chebyshev coordinate.
- `r::Float64`:
  Distance at which to evaluate the functions.

# Returns
- `nothing`
"""
@inline function h1_j1_multi_ks_at_r!(h1vals::AbstractVector{ComplexF64},j1vals::AbstractVector{ComplexF64},plans1::AbstractVector{ChebHankelPlanH},plansj1::AbstractVector{ChebJPlan},pidx_h::Int32,t_h::Float64,pidx_j::Int32,t_j::Float64,r::Float64)
    eval_h_multi_ks!(h1vals,plans1,r,pidx_h,t_h)
    eval_j_multi_ks!(j1vals,plansj1,pidx_j,t_j)
    return nothing
end

"""
    h0_h1_h2_at_r(plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,pidx::Int32,t::Float64,r::Float64)

Evaluate `H₀^(1)`, `H₁^(1)`, and `H₂^(1)` at one fixed distance for a single
wavenumber, returning the values directly.

This is the small scalar evaluator used by the single-`k` derivative DLP
Chebyshev pathway. The geometric pair `(i,j)` has already been mapped to:
- a Chebyshev panel index `pidx`,
- a local coordinate `t ∈ [-1,1]`,
- and the physical distance `r`.

For the interpolated regime, the function evaluates:
- `H₀^(1)(k r)` from `plan0`,
- `H₁^(1)(k r)` from `plan1`,

and then recovers `H₂^(1)(k r)` by the recurrence

    H₂^(1)(z) = (2/z) H₁^(1)(z) - H₀^(1)(z).

Near zero, the function switches to the small-argument series or direct special-
function evaluation.

# Arguments
- `plan0::ChebHankelPlanH`:
  Chebyshev plan for `H₀^(1)`.
- `plan1::ChebHankelPlanH`:
  Chebyshev plan for `H₁^(1)`.
- `pidx::Int32`:
  Panel index containing the current distance. If `pidx==0`, the evaluation is
  taken from the near-zero patch instead of the Chebyshev panels.
- `t::Float64`:
  Local Chebyshev coordinate in the active panel.
- `r::Float64`:
  Distance at which to evaluate the Hankel functions.

# Returns
- `(H0,H1,H2)::Tuple{ComplexF64,ComplexF64,ComplexF64}`
"""
@inline function h0_h1_h2_at_r(plan0::ChebHankelPlanH,plan1::ChebHankelPlanH,pidx::Int32,t::Float64,r::Float64)
    z=ComplexF64(plan0.k)*r
    az=abs(z)
    if az<hankel_z_chebyshev_cutoff_small_z
        H0=_small_h0_series(z)
        H1=_small_h1_series(z)
        H2=SpecialFunctions.besselh(2,1,z)
    elseif az<hankel_z_chebyshev_cutoff||pidx==0
        H0=SpecialFunctions.besselh(0,1,z)
        H1=SpecialFunctions.besselh(1,1,z)
        H2=SpecialFunctions.besselh(2,1,z)
    else
        H0=_cheb_clenshaw(plan0.panels[pidx].c,t)
        H1=_cheb_clenshaw(plan1.panels[pidx].c,t)
        H2=(2/z)*H1-H0
    end
    return H0,H1,H2
end

"""
    h0_h1_h2_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},h2vals::AbstractVector{ComplexF64},
    plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64,r::Float64)

Evaluate `H₀^(1)`, `H₁^(1)`, and `H₂^(1)` for all wavenumbers at one fixed
distance panel/location, writing the results in place.

This is the reduced special-function evaluator used by the multi-`k` derivative
DLP Chebyshev assembly. The geometric pair `(i,j)` has already been mapped to:
- a Chebyshev panel index `pidx`,
- a local coordinate `t ∈ [-1,1]`,
- and the physical distance `r`.

For each wavenumber `k_m`, the function evaluates:
- `H₀^(1)(k_m r)`,
- `H₁^(1)(k_m r)`,

and then recovers `H₂^(1)(k_m r)` by the recurrence

    H₂^(1)(z) = (2/z) H₁^(1)(z) - H₀^(1)(z),

except in the small-argument regime, where it switches to series or direct
evaluation to avoid loss of accuracy.

# Arguments
- `h0vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₀^(1)` values.
- `h1vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₁^(1)` values.
- `h2vals::AbstractVector{ComplexF64}`:
  Output vector for the `H₂^(1)` values.
- `plans0::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₀^(1)`.
- `plans1::AbstractVector{ChebHankelPlanH}`:
  Chebyshev plans for `H₁^(1)`.
- `pidx::Int32`:
  Panel index containing the current distance. If `pidx==0`, the evaluation is
  taken from the near-zero patch instead of the Chebyshev panels.
- `t::Float64`:
  Local Chebyshev coordinate in the active panel.
- `r::Float64`:
  Distance at which to evaluate the Hankel functions.

# Returns
- `nothing`
"""
@inline function h0_h1_h2_multi_ks_at_r!(h0vals::AbstractVector{ComplexF64},h1vals::AbstractVector{ComplexF64},h2vals::AbstractVector{ComplexF64},
    plans0::AbstractVector{ChebHankelPlanH},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64,r::Float64)
    @inbounds for m in eachindex(plans0)
        z=ComplexF64(plans0[m].k)*r
        az=abs(z)
        if az<hankel_z_chebyshev_cutoff_small_z
            h0vals[m]=_small_h0_series(z)
            h1vals[m]=_small_h1_series(z)
            h2vals[m]=SpecialFunctions.besselh(2,1,z)
        elseif az<hankel_z_chebyshev_cutoff||pidx==0
            h0vals[m]=SpecialFunctions.besselh(0,1,z)
            h1vals[m]=SpecialFunctions.besselh(1,1,z)
            h2vals[m]=SpecialFunctions.besselh(2,1,z)
        else
            h0vals[m]=_cheb_clenshaw(plans0[m].panels[pidx].c,t)
            h1vals[m]=_cheb_clenshaw(plans1[m].panels[pidx].c,t)
            h2vals[m]=(2/z)*h1vals[m]-h0vals[m]
        end
    end
    return nothing
end

"""
    h1_multi_ks_at_r!(h1vals,plans1,pidx,t,r)

Evaluate `H₁^(1)(k_m r)` for all stored wavenumbers at one fixed distance.

This is the value-only special-function evaluator used by the standard DLP
Chebyshev assembly.

## Arguments
* `h1vals`: Output buffer for `H₁^(1)` values.
* `plans1`: Order-one outgoing Hankel plans.
* `pidx`: Hankel panel index, with `0` selecting the direct/low-z fallback.
* `t`: Local Chebyshev coordinate.
* `r`: Physical distance.

## Returns
`nothing`.
"""
@inline function h1_multi_ks_at_r!(h1vals::AbstractVector{ComplexF64},plans1::AbstractVector{ChebHankelPlanH},pidx::Int32,t::Float64,r::Float64)
    eval_h_multi_ks!(h1vals,plans1,r,pidx,t)
    return nothing
end

"""
    h1_at_r(plan1,pidx,t,r)

Evaluate the unscaled outgoing Hankel function `H₁^(1)(k r)` at one fixed
distance using an order-one Chebyshev plan.

## Arguments
* `plan1`: Order-one outgoing Hankel plan.
* `pidx`: Hankel panel index, with `0` selecting the direct/low-z fallback.
* `t`: Local Chebyshev coordinate.
* `r`: Physical distance.

## Returns
`H₁^(1)(k r)`.
"""
@inline function h1_at_r(plan1::ChebHankelPlanH,pidx::Int32,t::Float64,r::Float64)
    return eval_h(plan1,pidx,t,r)
end

##################################################################
################## CFIE WAVEFUNCTION EVALUATION ##################
##################################################################

# Chebyhsev interpolation plan for SLP wavefunction reconstruction from DLP type kernels
struct SLPWavefunctionChebPlan
    plan::ChebHankelPlanH
end

@inline function _eval_y0_slp_cheb(pl::SLPWavefunctionChebPlan,k::T,r::T) where {T<:Real}
    z=Float64(k*r)
    if z<hankel_z_chebyshev_cutoff
        return T(Bessels.bessely0(z))
    end
    pidx=Int32(_find_panel(pl.plan,Float64(r)))
    P=pl.plan.panels[pidx]
    t=(2*Float64(r)-(P.b+P.a))/(P.b-P.a)
     # plan stores H₀⁽¹⁾(k r) = J₀(k r) + iY₀(k r)
    return T(imag(_cheb_clenshaw(P.c,t)))
end

struct DLPWavefunctionChebPlan
    h1::ChebHankelPlanH
end

@inline function _eval_h1_dlp_cheb(pl::DLPWavefunctionChebPlan,r::T) where {T<:Real}
    rf=Float64(r)
    z=ComplexF64(pl.h1.k)*rf
    az=abs(z)
    if az<hankel_z_chebyshev_cutoff_small_z
        return _small_h1_series(z)
    elseif az<hankel_z_chebyshev_cutoff || rf<pl.h1.rmin || rf>pl.h1.rmax
        return SpecialFunctions.besselh(1,1,z)
    end
    pidx,t=panel_t(pl.h1,rf)
    return h1_at_r(pl.h1,pidx,t,rf)
end

#    CFIEWavefunctionChebPlan
#
# Chebyshev interpolation plan for CFIE wavefunction reconstruction.
#
# Stores one piecewise-Chebyshev plan for H₀^(1)(k r) and one for
# H₁^(1)(k r) on the same radial interval. The plans are used only for
# postprocessing/eigenfunction plotting, not for solving the boundary problem.
struct CFIEWavefunctionChebPlan
    h0::ChebHankelPlanH
    h1::ChebHankelPlanH
end

#    _eval_h0h1_cfie_cheb(pl::CFIEWavefunctionChebPlan,r::T)
#
# Evaluate (H₀^(1)(k r), H₁^(1)(k r)) using the CFIE wavefunction Chebyshev plan.
# If r lies outside the interpolation interval, the function falls back to direct
# special-function evaluation, with the small-argument Hankel series used near
# k*r = 0. The H₀ and H₁ plans have identical radial panelization.
@inline function _eval_h0h1_cfie_cheb(pl::CFIEWavefunctionChebPlan,r::T) where {T<:Real}
    k=pl.h0.k
    rf=Float64(r)
    z=ComplexF64(k)*rf
    az=abs(z)
    if az<hankel_z_chebyshev_cutoff_small_z
        return _small_h0_series(z),_small_h1_series(z)
    elseif az<hankel_z_chebyshev_cutoff || rf<pl.h0.rmin || rf>pl.h0.rmax
        return SpecialFunctions.besselh(0,1,z),SpecialFunctions.besselh(1,1,z)
    end
    pidx=Int32(_find_panel(pl.h0,rf))
    P=pl.h0.panels[pidx]
    t=(2*rf-(P.b+P.a))/(P.b-P.a)
    return _cheb_clenshaw(P.c,t),_cheb_clenshaw(pl.h1.panels[pidx].c,t)
end