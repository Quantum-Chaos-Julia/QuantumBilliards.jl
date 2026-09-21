################################################################################
# POINCARE-HUSIMI FUNCTIONS
#
# For a Dirichlet eigenstate, let u(s) = ∂ₙψ(s) denote the physical normal
# derivative of the wavefunction on the boundary, parametrized by physical
# arclength s ∈ [0,L). The Poincare-Husimi amplitude is the projection onto
# a periodized Gaussian coherent state,
#
#   h(q,p) = ∫∂Ω u(s) exp[-k(s-q)²/2] exp[-ikp(s-q)] ds,
#
# where q is the boundary position and p ∈ [-1,1] is the tangential momentum
# in units of the total momentum k. The Gaussian width is σ = 1/√k and is
# numerically truncated to |s-q| ≤ wσ.
# Matrix-valued Poincare-Husimi densities are normalized to Σᵢⱼ Hᵢⱼ = 1.
#
# If full_p=false, only p ≥ 0 is evaluated and the negative half is obtained
# by reflection. This requires u(s) to be real up to a global phase and is the
# default since mostly we use reflections (1d irreps). For genuinely complex
# boundary functions (such as some 2d irreps arising from NFoldRotation
# symmetry) use full_p=true.
################################################################################

# Extend a one-sided vector antisymmetrically without duplicating its first entry.
function antisym_vec(x::AbstractVector{T}) where {T}
    return vcat(reverse(-x[2:end]),x)
end

# Check whether s and ds define a sufficiently uniform physical-arclength grid.
function _husimi_uniform_arclength_grid(s::AbstractVector{T}, ds::AbstractVector{T}; rtol::Real=sqrt(eps(T))) where {T<:Real}
    N = length(s)
    Δs = s[2]-s[1]
    atol_s = T(rtol)*max(one(T),abs(Δs))
    atol_ds = T(rtol)*max(one(T),abs(ds[1]))
    @inbounds for i in 2:N-1
        isapprox(s[i+1]-s[i],Δs;rtol=rtol,atol=atol_s) || return false
    end
    @inbounds for i in 2:N
        isapprox(ds[i],ds[1];rtol=rtol,atol=atol_ds) || return false
    end
    return true
end

# Construct the symmetric relative-arclength stencil for the uniform evaluator.
function _husimi_symmetric_window(s::AbstractVector{T}, ds::AbstractVector{T}, width::Real) where {T<:Real}
    N = length(s)
    Δs = s[2]-s[1]
    nside = min(N,1+floor(Int,T(width)/Δs))
    x = Vector{T}(undef,nside)
    dx = Vector{T}(undef,nside)
    s0 = s[1]
    @inbounds for i in 1:nside
        x[i] = s[i]-s0
        dx[i] = ds[i]
    end
    idx = nside
    x = antisym_vec(x)
    dx = vcat(reverse(dx[2:end]),dx)
    return x,dx,idx
end

# Normalize a discrete Poincare-Husimi density
@inline function _normalize_husimi!(H::AbstractMatrix{T}) where {T<:Real}
    H ./= sum(H)
    return H
end

################################################################################
# GENERAL NONUNIFORM PHYSICAL-ARCLENGTH QUADRATURE
################################################################################

"""
    husimi_function(k::T, s::AbstractVector{T}, ds::AbstractVector{T}, u::AbstractVector{Num}, L::T, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}

Evaluate the Poincare-Husimi function on prescribed phase-space grids using
general physical-arclength quadrature.
The returned Poincare-Husimi density is normalized so that `sum(H)=1`.

## Arguments
* `k::T`: Eigenwavenumber.
* `s::AbstractVector{T}`: Physical boundary arclength coordinates.
* `ds::AbstractVector{T}`: Physical boundary quadrature weights.
* `u::AbstractVector{Num}`: Physical boundary normal derivative `∂ₙψ`.
* `L::T`: Total physical boundary length.
* `qs::AbstractVector{T}`: Boundary-position grid.
* `ps::AbstractVector{T}`: Momentum values evaluated explicitly.

## Keyword Arguments
* `w::Real=7.0`: Gaussian truncation radius in units of `σ=1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `H::Matrix{T}`: Normalized Poincare-Husimi density indexed as `H[q,p]`.
* `qs::Vector{T}`: Boundary-position grid.
* `ps::Vector{T}`: Full signed momentum grid.
"""
function husimi_function(k::T, s::AbstractVector{T}, ds::AbstractVector{T}, u::AbstractVector{Num}, L::T, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    # For each boundary position `q`, the coherent-state amplitude is approximated by
    # `h(q,p) = Σⱼ uⱼ exp[-k(sⱼ-q)²/2] exp[-ikp(sⱼ-q)] dsⱼ`,
    # with only points satisfying `|sⱼ-q| ≤ w/√k` retained. Periodicity is handled
    # by extending the boundary coordinates and data by one copy in either direction.
    # For fixed `q`, the Gaussian envelope, quadrature weights and boundary function
    # are combined once and reused for every momentum. This supports arbitrary
    # physical-arclength discretizations.
    full_p || (iszero(first(ps)) && all(p->p>=zero(T),ps)) || throw(ArgumentError("full_p=false requires a nonnegative momentum grid starting at p=0"))
    width = T(w)/sqrt(k)
    s_ext = vcat(s.-L,s,s.+L)
    ds_ext = vcat(ds,ds,ds)
    u_ext = vcat(u,u,u)
    nq = length(qs); np = length(ps)
    Hp = zeros(T,np,nq)
    offsets = Vector{T}(undef,0)
    c_re = Vector{T}(undef,0)
    c_im = Vector{T}(undef,0)
    @inbounds for iq in 1:nq
        q = mod(qs[iq],L)+L
        lo = searchsortedfirst(s_ext,q-width)
        hi = searchsortedlast(s_ext,q+width)
        W = max(0,hi-lo+1)
        if length(offsets)<W
            resize!(offsets,W); resize!(c_re,W); resize!(c_im,W)
        end
        for t in 1:W
            j = lo+t-1
            d = s_ext[j]-q
            wt = exp(-T(0.5)*k*d*d)*ds_ext[j]
            offsets[t] = d
            c_re[t] = wt*real(u_ext[j])
            c_im[t] = wt*imag(u_ext[j])
        end
        for ip in 1:np
            kp = k*ps[ip]
            sracc = zero(T); siacc = zero(T)
            for t in 1:W
                sn,cs = sincos(kp*offsets[t])
                re = c_re[t]; im = c_im[t]
                sracc += re*cs+im*sn
                siacc += im*cs-re*sn
            end
            Hp[ip,iq] = sracc*sracc+siacc*siacc
        end
    end
    if full_p
        H = permutedims(Hp)
        ps_out = collect(ps)
    else
        H = permutedims(vcat(reverse(Hp[2:end,:];dims=1),Hp))
        ps_out = antisym_vec(ps)
    end
    _normalize_husimi!(H)
    return H,collect(qs),ps_out
end

################################################################################
# FAST UNIFORM-ARCLENGTH QUADRATURE
################################################################################

"""
    _husimi_uniform_arclength(k::T, s::AbstractVector{T}, ds::AbstractVector{T}, u::AbstractVector{Num}, L::T; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}

Evaluate the Poincare-Husimi function using a translating coherent-state
stencil on a uniform physical-arclength boundary grid.
The returned Poincare-Husimi density is normalized so that `sum(H)=1`.

## Arguments
* `k::T`: Eigenwavenumber.
* `s::AbstractVector{T}`: Uniform physical boundary arclength coordinates.
* `ds::AbstractVector{T}`: Uniform physical boundary quadrature weights.
* `u::AbstractVector{Num}`: Physical boundary normal derivative `∂ₙψ`.
* `L::T`: Total physical boundary length.

## Keyword Arguments
* `c::Real=10.0`: Phase-space sampling density in units of `σ=1/√k`.
* `w::Real=7.0`: Gaussian truncation radius in units of `σ`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `H::Matrix{T}`: Normalized Poincare-Husimi density.
* `qs::Vector{T}`: Sampled boundary positions.
* `ps::Vector{T}`: Full signed momentum grid.
"""
function _husimi_uniform_arclength(k::T, s::AbstractVector{T}, ds::AbstractVector{T}, u::AbstractVector{Num}, L::T; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    # The relative coordinates and quadrature weights inside the truncated
    # coherent-state window are independent of its center `q`. The Gaussian factor
    # `exp(-kx²/2)` is constructed once. For each momentum `p`, the Fourier factor
    # `exp(-ikpx)` produces a fixed complex stencil which is translated around the periodic boundary.
    # The coherent-state width is `σ=1/√k`. Approximately `c` samples per `σ` are
    # used in boundary position, with comparable momentum resolution.
    N = length(s)
    sig = inv(sqrt(k))
    x,dx,idx = _husimi_symmetric_window(s,ds,T(w)*sig)
    W = length(x)
    gauss = Vector{T}(undef,W)
    @inbounds @simd for t in 1:W
        gauss[t] = exp(-T(0.5)*k*x[t]*x[t])*dx[t]
    end
    np = max(1,ceil(Int,T(c)*sqrt(k)))
    ps_eval = full_p ? collect(range(-one(T),one(T);length=2np+1)) : collect(range(zero(T),one(T);length=np+1))
    Δs = s[2]-s[1]
    q_stride = max(1,round(Int,(sig/T(c))/Δs))
    q_idx = collect(1:q_stride:N)
    qs = collect(s[q_idx])
    nq = length(qs)
    H = zeros(T,nq,length(ps_eval))
    uc = CircularVector(u)
    cs_re = Vector{T}(undef,W)
    cs_im = Vector{T}(undef,W)
    @inbounds for ip in eachindex(ps_eval)
        kp = k*ps_eval[ip]
        for t in 1:W
            sn,cs = sincos(kp*x[t])
            cs_re[t] = gauss[t]*cs
            cs_im[t] = -gauss[t]*sn
        end
        for iq in 1:nq
            j = q_idx[iq]
            sracc = zero(T); siacc = zero(T)
            for t in 1:W
                uj = uc[j-idx+t]
                ur = real(uj); ui = imag(uj)
                cr = cs_re[t]; ci = cs_im[t]
                sracc += cr*ur-ci*ui
                siacc += cr*ui+ci*ur
            end
            H[iq,ip] = sracc*sracc+siacc*siacc
        end
    end
    if full_p
        ps_out = ps_eval
    else
        H = hcat(reverse(H[:,2:end];dims=2),H)
        ps_out = antisym_vec(ps_eval)
    end
    _normalize_husimi!(H)
    return H,qs,ps_out
end

"""
    husimi_function(k::T, s::AbstractVector{T}, ds::AbstractVector{T}, u::AbstractVector{Num}, L::T; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}

Compute an automatically gridded Poincare-Husimi function.

## Arguments
* `k::T`: Eigenwavenumber.
* `s::AbstractVector{T}`: Physical boundary arclength coordinates.
* `ds::AbstractVector{T}`: Physical boundary quadrature weights.
* `u::AbstractVector{Num}`: Physical boundary normal derivative `∂ₙψ`.
* `L::T`: Total physical boundary length.

## Keyword Arguments
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `H::Matrix{T}`: Normalized Poincare-Husimi density.
* `qs::Vector{T}`: Boundary-position grid.
* `ps::Vector{T}`: Full signed momentum grid.
"""
function husimi_function(k::T, s::AbstractVector{T}, ds::AbstractVector{T}, u::AbstractVector{Num}, L::T; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    sig = inv(sqrt(k))
    if _husimi_uniform_arclength_grid(s,ds) && T(w)*sig<L/T(2)
        return _husimi_uniform_arclength(k,s,ds,u,L;c=c,w=w,full_p=full_p)
    end
    nq = max(2,ceil(Int,L*T(c)/sig))
    np = max(1,ceil(Int,T(c)/sig))
    qs = collect(range(zero(T),L;length=nq+1))[1:end-1]
    ps = full_p ? collect(range(-one(T),one(T);length=2np+1)) : collect(range(zero(T),one(T);length=np+1))
    return husimi_function(k,s,ds,u,L,qs,ps;w=w,full_p=full_p)
end

################################################################################
# BOUNDARYPOINTS API
################################################################################

"""
    husimi_function(k::T, pts::BoundaryPoints{T}, u::AbstractVector{Num}; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}

Compute the Poincare-Husimi function from a physical boundary discretization.

## Arguments
* `k::T`: Eigenwavenumber.
* `pts::BoundaryPoints{T}`: Complete physical boundary discretization.
* `u::AbstractVector{Num}`: Physical boundary normal derivative `∂ₙψ`.

## Keyword Arguments
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `H::Matrix{T}`: Normalized Poincare-Husimi density.
* `qs::Vector{T}`: Boundary-position grid.
* `ps::Vector{T}`: Full signed momentum grid.
"""
function husimi_function(k::T, pts::BoundaryPoints{T}, u::AbstractVector{Num}; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    L = sum(pts.ds)
    return husimi_function(k,pts.s,pts.ds,u,L;c=c,w=w,full_p=full_p)
end

"""
    husimi_function(k::T, pts::BoundaryPoints{T}, u::AbstractVector{Num}, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}

Compute the Poincare-Husimi function on prescribed phase-space grids.

## Arguments
* `k::T`: Eigenwavenumber.
* `pts::BoundaryPoints{T}`: Complete physical boundary discretization.
* `u::AbstractVector{Num}`: Physical boundary normal derivative `∂ₙψ`.
* `qs::AbstractVector{T}`: Boundary-position grid.
* `ps::AbstractVector{T}`: Momentum values evaluated explicitly.

## Keyword Arguments
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `H::Matrix{T}`: Normalized Poincare-Husimi density.
* `qs::Vector{T}`: Boundary-position grid.
* `ps::Vector{T}`: Full signed momentum grid.
"""
function husimi_function(k::T, pts::BoundaryPoints{T}, u::AbstractVector{Num}, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    L = sum(pts.ds)
    return husimi_function(k,pts.s,pts.ds,u,L,qs,ps;w=w,full_p=full_p)
end

################################################################################
# EIGENSTATE API
################################################################################

"""
    husimi_function(state::BIMEigenstate{K,T,S,Bi}; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {K,T<:Real,S<:SweepBIMSolver,Bi}

Compute the Poincare-Husimi function of a BIM eigenstate.

If the physical boundary normal derivative `state.u` is available, it is used
directly. Otherwise it is reconstructed at `state.k` with [`solve_state`](@ref)
using the boundary discretization stored in `state.pts`.

## Arguments
* `state::BIMEigenstate`: BIM eigenstate.

## Keyword Arguments
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `H::Matrix{T}`: Normalized Poincare-Husimi density.
* `qs::Vector{T}`: Boundary-position grid.
* `ps::Vector{T}`: Full signed momentum grid.
"""
function husimi_function(state::BIMEigenstate{K,T,S,Bi}; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {K,T<:Real,S<:SweepBIMSolver,Bi}
    k=T(real(state.k))
    u=state.u
    if u===nothing
        _,_,u,_=solve_state(state.solver,state.pts,k,state.billiard)
    end
    L=T(sum(crv.length for crv in full_boundary(state.billiard)))
    return husimi_function(k,state.pts.s,state.pts.ds,u,L;c=c,w=w,full_p=full_p)
end

"""
    husimi_function(state::S; b::Real=5.0, c::Real=10.0, w::Real=7.0, multithreaded::Bool=true, full_p::Bool=false) where {S<:AbsState}

Compute the Poincare-Husimi function of a basis eigenstate.

## Arguments
* `state::S`: Eigenstate represented in a basis.

## Keyword Arguments
* `b::Real=5.0`: Boundary-function oversampling factor.
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `multithreaded::Bool=true`: Whether boundary-function construction is multithreaded.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `H::Matrix{T}`: Normalized Poincare-Husimi density.
* `qs::Vector{T}`: Boundary-position grid.
* `ps::Vector{T}`: Full signed momentum grid.
"""
function husimi_function(state::S; b::Real=5.0, c::Real=10.0, w::Real=7.0, multithreaded::Bool=true, full_p::Bool=false) where {S<:AbsState}
    u,pts,_ = _basis_boundary_function_pts(state;b=b,multithreaded=multithreaded)
    T = eltype(pts.ds)
    k = T(real(state.k))
    L = T(sum(crv.length for crv in full_boundary(state.billiard)))
    return husimi_function(k,pts.s,pts.ds,u,L;c=c,w=w,full_p=full_p)
end

################################################################################
# MULTIPLE EIGENSTATES
################################################################################

"""
    husimi_function(states::AbstractVector{<:BIMEigenstate{K,T,S,Bi}}; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {K,T<:Real,S<:SweepBIMSolver,Bi}

Compute Poincare-Husimi functions for several BIM eigenstates.

Each state uses its natural automatically generated phase-space grid.

## Arguments
* `states::AbstractVector{<:BIMEigenstate}`: BIM eigenstates.

## Keyword Arguments
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `Hs::Vector{Matrix{T}}`: Poincare-Husimi matrices.
* `ps::Vector{Vector{T}}`: Corresponding momentum grids.
* `qs::Vector{Vector{T}}`: Corresponding boundary-position grids.
"""
function husimi_function(states::AbstractVector{<:BIMEigenstate{K,T,S,Bi}};c::Real=10.0,w::Real=7.0,full_p::Bool=false) where {K,T<:Real,S<:SweepBIMSolver,Bi}
    n = length(states)
    Hs = Vector{Matrix{T}}(undef,n)
    ps_return = Vector{Vector{T}}(undef,n)
    qs_return = Vector{Vector{T}}(undef,n)
    ok = Vector{Bool}(undef,n); fill!(ok,true)
    pbar = Progress(n;desc="Constructing Poincare-Husimi matrices, N=$n")
    progress_lock = ReentrantLock()
    Threads.@threads for i in eachindex(states)
        try
            H,qs,ps = husimi_function(states[i];c=c,w=w,full_p=full_p)
            Hs[i] = H; ps_return[i] = ps; qs_return[i] = qs
        catch e
            @debug "Poincare-Husimi fail at k=$(states[i].k)" exception=(e,catch_backtrace())
            ok[i] = false
        end
        lock(progress_lock) do
            next!(pbar)
        end
    end
    return Hs[ok],ps_return[ok],qs_return[ok]
end

"""
    husimi_function(states::AbstractVector{S}; b::Real=5.0, c::Real=10.0, w::Real=7.0, multithreaded::Bool=true, full_p::Bool=false) where {S<:AbsState}

Compute Poincare-Husimi functions for several basis eigenstates.

## Arguments
* `states::AbstractVector{S}`: Eigenstates represented in a basis.

## Keyword Arguments
* `b::Real=5.0`: Boundary-function oversampling factor.
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `multithreaded::Bool=true`: Whether each boundary-function construction is multithreaded.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false.

## Returns
* `Hs::Vector{Matrix{T}}`: Poincare-Husimi matrices.
* `ps::Vector{Vector{T}}`: Corresponding momentum grids.
* `qs::Vector{Vector{T}}`: Corresponding boundary-position grids.
"""
function husimi_function(states::AbstractVector{S}; b::Real=5.0, c::Real=10.0, w::Real=7.0, multithreaded::Bool=true, full_p::Bool=false) where {S<:AbsState}
    n = length(states)
    T = typeof(real(first(states).k))
    Hs = Vector{Matrix{T}}(undef,n)
    ps_return = Vector{Vector{T}}(undef,n)
    qs_return = Vector{Vector{T}}(undef,n)
    ok = Vector{Bool}(undef,n); fill!(ok,true)
    pbar = Progress(n;desc="Constructing Poincare-Husimi matrices, N=$n")
    @inbounds for i in eachindex(states)
        try
            H,qs,ps = husimi_function(states[i];b=b,c=c,w=w,multithreaded=multithreaded,full_p=full_p)
            Hs[i] = H; ps_return[i] = ps; qs_return[i] = qs
        catch e
            @debug "Poincare-Husimi fail at k=$(states[i].k)" exception=(e,catch_backtrace())
            ok[i] = false
        end
        next!(pbar)
    end
    return Hs[ok],ps_return[ok],qs_return[ok]
end