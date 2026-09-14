
"""
    antisym_vec(x::AbstractVector) → v::Vector

Antisymmetrically extends a one-sided vector `x` (assumed sorted with
`x[1]` closest to zero) to negative values, returning `[-reverse(x[2:end]);
x]`.

## Description
This is used to build symmetric grids of coherent-state evaluation points (in
`q` or `p`) around zero from a one-sided grid `x`, by mirroring and negating
all but the first entry and prepending the result to `x`.

## Arguments
* `x`: The one-sided vector to extend, typically non-negative and increasing.

## Returns
*  `v` : The antisymmetrically extended vector `[-reverse(x[2:end]); x]`, of length `2*length(x) - 1`.
"""
function antisym_vec(x)
    v = reverse(-x[2:end])
    return append!(v,x)
end

"""
    husimi_function(k, u::AbstractVector, s::AbstractVector, L::Real; c::Real = 10.0, w::Real = 7.0) → (H::Matrix, qs::Vector, ps::Vector)

Computes the boundary Husimi function of the normal-derivative boundary
function `u(s)` (sampled at equally spaced arc-length points `s`) at
wavenumber `k`, on a boundary of total length `L`, returning the Husimi
density `H` on a grid of arc-length coordinates `qs` and momenta `ps`.

## Description
`u` is projected onto (approximately) minimal-uncertainty coherent-state
wavepackets of Gaussian width `sig = 1/sqrt(k)` in the arc-length coordinate,
truncated to `w` widths (`x = s[s .<= w*sig]`) and evaluated with `c` points
per width along the momentum direction. Because the boundary is periodic with
period `L`, each coherent state is periodized by adding its two nearest
periodic images at `s ± L` (`gauss_l`, `gauss_r`) before contraction with
`u`, giving the overlap

```math
h(q,p) = \\sum_{s} u(s)\\, \\big[g(s-q) + g(s-q+L) + g(s-q-L)\\big]\\, e^{i k p (s-q)},
```

with `g(\\cdot) = \\exp(-k\\, (\\cdot)^2/2)` the Gaussian envelope. The Husimi
density is `H(q,p) = a\\, |h(q,p)|^2`, with normalization constant
`a = 1/(2\\pi\\sqrt{\\pi k})` (not normalized so that `H` integrates to `1`).
Momenta are sampled on `ps ∈ [0,1]` in steps of `sig/c` and then mirrored to
`[-1,1]` via [`antisym_vec`](@ref), and arc-length points `qs` are subsampled
from `s` with the same step `sig/c`.

## Arguments
* `k`: The wavenumber of the eigenstate.
* `u`: The (real) boundary normal-derivative function, typically from [`boundary_function`](@ref).
* `s`: The arc-length coordinates of `u`, assumed equally spaced.
* `L`: The total length of the (periodized) boundary.

## Keyword arguments
*  `c::Real = 10.0` : Number of coherent-state evaluation points per Gaussian width `sig`, controlling the resolution of `qs` and `ps`.
*  `w::Real = 7.0` : Truncation width of the coherent-state Gaussian envelope, in units of `sig`.

## Returns
*  `H` : The Husimi density on the grid `(qs, ps)`.
*  `qs` : Arc-length coordinates at which `H` is sampled.
*  `ps` : Momenta (in units of `k`, ranging over `[-1,1]`) at which `H` is sampled.
"""
function husimi_function(k,u,s,L; c = 10.0, w = 7.0)
    #c density of points in coherent state peak, w width in units of sigma
    #L is the boundary length for periodization
    #compute coherrent state weights
    N = length(s)
    sig = one(k)/sqrt(k) #width of the gaussian
    x = s[s.<=w*sig]
    x = x .- x[1] #shift so the window starts at physical offset 0, matching
                  #the CircularVector slice below (s[1] is generally > 0 for
                  #a midpoint-quadrature grid, not 0); without this shift `x`
                  #and the physical sample offsets of `u_w` disagree by s[1],
                  #which introduces a k*p*s[1] phase error that leaks
                  #spurious signal into large |p| (worse at high k)
    idx = length(x) #do not change order here
    x = antisym_vec(x)
    a = one(k)/(2*pi*sqrt(pi*k)) #normalization factor in this version Hsimi is not noramlized to 1
    ds = (x[end]-x[1])/length(x) #integration weigth
    uc = CircularVector(u) #allows circular indexing
    gauss = @. exp(-k/2*x^2)*ds
    gauss_l = @. exp(-k/2*(x+L)^2)*ds
    gauss_r = @. exp(-k/2*(x-L)^2)*ds
    #construct evaluation points in p coordinate
    ps = collect(range(0.0,1.0,step = sig/c))
    #construct evaluation points in q coordinate
    q_stride = length(s[s.<=sig/c])
    q_idx = collect(1:q_stride:N)
    if q_idx[end] != N
        push!(q_idx,N) #add last point, only if not already covered by the stride
    end
    qs = s[q_idx]
    #println(length(qs))
    H = zeros(typeof(k),length(qs),length(ps))
    for i in eachindex(ps)   
        cs = @. exp(im*ps[i]*k*x)*gauss + exp(im*ps[i]*k*(x+L))*gauss_l + exp(im*ps[i]*k*(x-L))*gauss_r#imag part of coherent state
        for j in eachindex(q_idx)
            u_w = uc[q_idx[j]-idx+1:q_idx[j]+idx-1] #window with relevant values of u
            h = sum(cs.*u_w)
            #hi = sum(ci.*u_w)
            H[j,i] = a*abs2(h)
        end
    end

    ps = antisym_vec(ps)
    H_ref = reverse(H[:, 2:end]; dims=2)
    H = hcat(H_ref,H)
     
    return H, qs, ps    
end

"""
    husimi_function(state::S; b::Real = 5.0, c::Real = 10.0, w::Real = 7.0, multithreaded::Bool = true, full_p::Bool = false) where {S<:AbsState} → (H::Matrix, qs::Vector, ps::Vector)

Computes the boundary Husimi function of an eigenstate directly from `state`,
by combining [`_basis_boundary_function_pts`](@ref) and the automatically
dispatching [`husimi_function(k, u, s, ds, L)`](@ref husimi_function) (uses
the fast uniform-arc-length stencil when valid, otherwise a general
quadrature that also works for non-uniformly-spaced boundary grids).

## Arguments
* `state`: The eigenstate for which the boundary Husimi function is computed.

## Keyword arguments
*  `b::Real = 5.0` : Oversampling factor controlling the boundary point density (see [`boundary_function`](@ref)).
*  `c::Real = 10.0` : Number of coherent-state evaluation points per Gaussian width, passed to [`husimi_function(k, u, s, ds, L)`](@ref husimi_function).
*  `w::Real = 7.0` : Truncation width of the coherent-state Gaussian envelope, passed to [`husimi_function(k, u, s, ds, L)`](@ref husimi_function).
*  `multithreaded::Bool = true` : Whether the underlying gradient matrix construction is multithreaded.
*  `full_p::Bool = false` : Passed to [`husimi_function(k, u, s, ds, L)`](@ref husimi_function).

## Returns
*  `H` : The Husimi density on the grid `(qs, ps)`.
*  `qs` : Arc-length coordinates at which `H` is sampled.
*  `ps` : Momenta at which `H` is sampled.
"""
function husimi_function(state::S; b=5.0, c=10.0, w=7.0, multithreaded=true, full_p=false) where {S<:AbsState}
    L = sum(crv.length for crv in full_boundary(state.billiard))
    k = state.k
    u, pts, norm = _basis_boundary_function_pts(state; b, multithreaded)
    return husimi_function(k,u,pts.s,pts.ds,L; c=c, w=w, full_p=full_p)
end

# Checks (to within `rtol`) whether `s` is uniformly spaced and `ds` is
# uniform, i.e. whether the fast sliding-window `husimi_function(k,u,s,L)`
# kernel's assumption of a uniform arc-length grid actually holds. A BIM
# solver using `GlobalCornerGrading` intentionally clusters nodes near
# corners, so `s` is not uniformly spaced there. Used by both the generic
# `AbsState` and `BIMEigenstate` `husimi_function` methods below to pick
# between the fast uniform stencil and the general non-uniform quadrature.
@inline function _bim_boundary_uniformly_spaced(s::AbstractVector{T}, ds::AbstractVector{T}; rtol::Real=1e-6) where {T<:Real}
    N = length(s)
    N < 3 && return true
    Δs = s[2]-s[1]
    return all(isapprox(s[i+1]-s[i], Δs; rtol=rtol) for i in 2:N-1) && all(isapprox(ds[i], ds[1]; rtol=rtol) for i in 2:N)
end

"""
    husimi_function(k, u::AbstractVector, s::AbstractVector, ds::AbstractVector, L::Real, qs::AbstractVector, ps::AbstractVector; w::Real = 7.0, full_p::Bool = false) → (H::Matrix, qs::Vector, ps::Vector)

General non-uniform-arclength boundary Husimi quadrature, evaluated directly
on the prescribed grids `qs`/`ps` from the physical arc-length coordinates
`s` and quadrature weights `ds` (which need not be uniformly spaced).

## Description
Unlike [`husimi_function(k, u, s, L)`](@ref husimi_function), which requires
`s` to be an (approximately) equally spaced grid so that the coherent-state
window can be sliced by a fixed number of neighbouring samples, this method
makes no such assumption: for each `q ∈ qs`, the physical samples within the
Gaussian truncation window `[q - w/√k, q + w/√k]` are located directly with
`searchsortedfirst`/`searchsortedlast` on the arc-length-periodized boundary
(`s .- L`, `s`, `s .+ L`, handling the two nearest periodic images the same
way [`husimi_function(k, u, s, L)`](@ref husimi_function) does), and
contracted against the true per-sample weight `ds`,

```math
h(q,p) = \\sum_{j:\\,|s_j-q|<w/\\sqrt{k}} u_j\\, e^{-k(s_j-q)^2/2}\\, \\mathrm{d}s_j\\, e^{i k p (s_j - q)},
```

with the same normalization `H(q,p) = a\\,|h(q,p)|^2`,
`a = 1/(2\\pi\\sqrt{\\pi k})`, as [`husimi_function(k, u, s, L)`](@ref husimi_function)
(not normalized so that `H` integrates to `1`), so results from the two
methods are directly comparable in scale. If `full_p=false` (default), only
`ps .>= 0` is evaluated explicitly and the negative-momentum half is
reconstructed by reflection (`H(q,-p) ≈ H(q,p)`, exact only up to `u`'s
global complex phase); set `full_p=true` to evaluate the full signed `ps`
grid explicitly instead (needed when `u` is not real up to a global phase).

## Arguments
* `k`: The wavenumber of the eigenstate.
* `u`: The (possibly complex) boundary normal-derivative function values at `s`.
* `s`: The physical arc-length coordinates of `u`, need not be uniformly spaced.
* `ds`: The physical quadrature weights corresponding to `u`/`s`.
* `L`: The total length of the (periodized) boundary.
* `qs`: The boundary-position coordinates at which to evaluate `H`.
* `ps`: The momenta to evaluate explicitly (only `ps .>= 0` needed if `full_p=false`).

## Keyword arguments
*  `w::Real = 7.0` : Truncation width of the coherent-state Gaussian envelope, in units of `1/√k`.
*  `full_p::Bool = false` : Whether `ps` already spans the full signed momentum interval.

## Returns
*  `H` : The Husimi density on the grid `(qs, ps)`, of size `(length(qs), length(ps))` (or `(length(qs), 2*count(ps.>=0)-1)` if `full_p=false`).
*  `qs` : The input boundary-position grid.
*  `ps` : The full signed momentum grid corresponding to `H`.
"""
function husimi_function(k::T, u::AbstractVector{Num}, s::AbstractVector{T}, ds::AbstractVector{T}, L::T, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    N = length(s)
    N == length(ds) == length(u) || throw(DimensionMismatch("s, ds and u must have equal length"))
    s_ext = vcat(s .- L, s, s .+ L)
    u_ext = vcat(u, u, u)
    ds_ext = vcat(ds, ds, ds)
    nx = length(qs)
    ny = length(ps)
    Hp = zeros(T, ny, nx)
    a = one(T)/(2*T(pi)*sqrt(T(pi)*k))
    width = T(w)/sqrt(k)
    c_re = Vector{T}(undef, 0)
    c_im = Vector{T}(undef, 0)
    si = Vector{T}(undef, 0)
    @inbounds for iq in 1:nx
        q = qs[iq]
        lo = searchsortedfirst(s_ext, q-width)
        hi = searchsortedlast(s_ext, q+width)
        W = max(0, hi-lo+1)
        if length(c_re) < W
            resize!(c_re, W)
            resize!(c_im, W)
            resize!(si, W)
        end
        @inbounds for t in 0:W-1
            j = lo+t
            sdiff = s_ext[j]-q
            si[t+1] = sdiff
            wt = exp(-T(0.5)*k*sdiff*sdiff)*ds_ext[j]
            uj = u_ext[j]
            if uj isa Real
                c_re[t+1] = wt*uj
                c_im[t+1] = zero(T)
            else
                c_re[t+1] = wt*real(uj)
                c_im[t+1] = wt*imag(uj)
            end
        end
        @inbounds for ip in 1:ny
            kp = k*ps[ip]
            sracc = zero(T)
            siacc = zero(T)
            @inbounds @simd for t in 1:W
                θ = kp*si[t]
                s_,c_ = sincos(θ)
                re = c_re[t]
                im_ = c_im[t]
                # (re+i*im_)*(cos θ + i sin θ), matching husimi_function(k,u,s,L)'s e^{+iθ} convention
                sracc += re*c_ - im_*s_
                siacc += re*s_ + im_*c_
            end
            Hp[ip,iq] = a*(sracc*sracc+siacc*siacc)
        end
    end
    if full_p
        H = permutedims(Hp)
        ps_out = collect(ps)
    else
        H = permutedims(vcat(reverse(Hp[2:end,:]; dims=1), Hp))
        ps_out = vcat(-reverse(ps[2:end]), ps)
    end
    return H, collect(qs), ps_out
end

"""
    husimi_function(k, u::AbstractVector, s::AbstractVector, ds::AbstractVector, L::Real; c::Real = 10.0, w::Real = 7.0, full_p::Bool = false) → (H::Matrix, qs::Vector, ps::Vector)

Automatically gridded boundary Husimi function, valid whether or not `s` is
uniformly spaced in arc-length.

## Description
If `s`/`ds` are (to within tolerance, see [`_bim_boundary_uniformly_spaced`](@ref))
uniformly spaced and `full_p=false`, delegates to the fast sliding-window
stencil [`husimi_function(k, u, s, L)`](@ref husimi_function). Otherwise, a
boundary-position grid `qs` (uniform on `[0,L)`, independent of `s`'s own
spacing) and momentum grid `ps` are built with the same `c`/`w`-controlled
sampling density as the fast stencil, and
[`husimi_function(k, u, s, ds, L, qs, ps)`](@ref husimi_function) is used to
evaluate `H` by direct physical quadrature.

## Arguments
* `k`: The wavenumber of the eigenstate.
* `u`: The (possibly complex) boundary normal-derivative function values at `s`.
* `s`: The physical arc-length coordinates of `u`.
* `ds`: The physical quadrature weights corresponding to `u`/`s`.
* `L`: The total length of the (periodized) boundary.

## Keyword arguments
*  `c::Real = 10.0` : Phase-space sampling density in units of the coherent-state width `1/√k`.
*  `w::Real = 7.0` : Truncation width of the coherent-state Gaussian envelope, in units of `1/√k`.
*  `full_p::Bool = false` : Whether to evaluate both signs of momentum explicitly (forces the general quadrature even on a uniform grid).

## Returns
*  `H` : The Husimi density on the returned grid `(qs, ps)`.
*  `qs` : Boundary-position coordinates at which `H` is sampled.
*  `ps` : Momenta at which `H` is sampled.
"""
function husimi_function(k::T, u::AbstractVector{Num}, s::AbstractVector{T}, ds::AbstractVector{T}, L::T; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    if !full_p && _bim_boundary_uniformly_spaced(s, ds)
        return husimi_function(k, u, s, L; c=c, w=w)
    end
    sig = one(T)/sqrt(k)
    nq = max(2, ceil(Int, L*T(c)/sig))
    np = max(1, ceil(Int, T(c)/sig))
    qs = collect(range(zero(T), L; length=nq+1))[1:end-1]
    ps = full_p ? collect(range(-one(T), one(T); length=2np+1)) : collect(range(zero(T), one(T); length=np+1))
    return husimi_function(k, u, s, ds, L, qs, ps; w=w, full_p=full_p)
end

"""
    husimi_function(state::BIMEigenstate{K,T,S,Bi}; b::Real = 5.0, c::Real = 10.0, w::Real = 7.0, multithreaded::Bool = true, full_p::Bool = false) where {K,T,S<:SweepBIMSolver,Bi} → (H::Matrix, qs::Vector, ps::Vector)

Computes the boundary Husimi function of a [`BIMEigenstate`](@ref) computed
with any [`SweepBIMSolver`](@ref), by combining the precomputed
`state.u`/`state.pts` (see [`compute_eigenstate`](@ref)/[`solve_state`](@ref))
with the automatically dispatching
[`husimi_function(k, u, s, ds, L)`](@ref husimi_function).

## Description
`state.pts` carries both `s` and `ds` for the complete physical boundary
(populated by `solve_state`, see [`BIMEigenstate`](@ref)), so a
[`GlobalCornerGrading`](@ref) solver's corner-clustered, non-uniformly-spaced
grid is handled correctly via the general quadrature instead of only
triggering a warning: [`husimi_function(k, u, s, ds, L)`](@ref husimi_function)
detects the non-uniform case itself and switches kernels automatically.

## Keyword arguments
*  `b::Real = 5.0` : Unused, accepted for interface compatibility (see [`boundary_function(state::BIMEigenstate)`](@ref)).
*  `c::Real = 10.0`, `w::Real = 7.0` : Passed to [`husimi_function(k, u, s, ds, L)`](@ref husimi_function).
*  `multithreaded::Bool = true` : Unused, accepted for interface compatibility; `state.u` is already computed by [`compute_eigenstate`](@ref).
*  `full_p::Bool = false` : Passed to [`husimi_function(k, u, s, ds, L)`](@ref husimi_function).
"""
function husimi_function(state::BIMEigenstate{K,T,S,Bi}; b=5.0, c=10.0, w=7.0, multithreaded=true, full_p=false) where {K,T,S<:SweepBIMSolver,Bi}
    pts = state.pts
    L = sum(pts.ds)
    return husimi_function(real(state.k), state.u, pts.s, pts.ds, L; c=c, w=w, full_p=full_p)
end

