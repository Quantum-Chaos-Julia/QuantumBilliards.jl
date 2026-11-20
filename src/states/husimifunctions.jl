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
#
# Methods that auto-generate their own (q,p) grids detect a genuinely complex
# u(s) and silently escalate to full_p=true, since a p -> -p reflection would
# otherwise be mathematically wrong. Methods that take explicit qs/ps grids
# instead raise an ArgumentError in that situation, since the caller-supplied
# grid shape cannot be safely changed.
################################################################################

"""
    antisym_vec(x::AbstractVector) → AbstractVector

Extend the one-sided vector `x` antisymmetrically to `[-x[end], …, -x[2], x[1], …, x[end]]`,
without duplicating the first entry.
"""
function antisym_vec(x::AbstractVector{T}) where {T}
    return vcat(reverse(-x[2:end]),x)
end

# Normalize a discrete Poincare-Husimi density
@inline function _normalize_husimi!(H::AbstractMatrix{T}) where {T<:Real}
    H ./= sum(H)
    return H
end

# Check whether `u` is real up to a single global complex phase, i.e. whether
# `u = exp(iθ)v` for some real vector `v`. This is required for `full_p=false`
# to be valid, since the `p -> -p` reflection symmetry used there relies on it.
function _is_global_phase_real(u::AbstractVector{<:Number}; rtol::Real=1e-6)
    eltype(u) <: Real && return true
    uj = argmax(abs2,u)
    iszero(uj) && return true
    v = u.*(conj(uj)/abs(uj))
    return maximum(abs,imag.(v)) <= rtol*maximum(abs,real.(v))
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
* [`HusimiFunction`](@ref) with `H` the normalized Poincare-Husimi density indexed as `H[q,p]`, `qs` the boundary-position grid, `ps` the full signed momentum grid and `k` the eigenwavenumber. It iterates as `(H, qs, ps)`.
"""
function husimi_function(k::T, s::AbstractVector{T}, ds::AbstractVector{T}, u::AbstractVector{Num}, L::T, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    # The coherent-state amplitude is
    # `h(q,p) = Σⱼ uⱼ exp[-k(sⱼ-q)²/2] exp[-ikp(sⱼ-q)] dsⱼ`,
    # with only points satisfying `|sⱼ-q| ≤ w/√k` retained. Only `|h|²` is used, so the
    # factor `exp(ikpq)` drops out and the momentum dependence of the window sum factorizes:
    # `|h(q,p)|² = |Σⱼ gⱼ(q) Fⱼₚ|²` with `gⱼ = exp[-k(sⱼ-q)²/2]` and `Fⱼₚ = uⱼ dsⱼ exp(-ikp sⱼ)`.
    # `F` costs one `sincos` per node and momentum; the window sum is then multiply-adds only.
    # Periodicity is handled by the boundary copies `sⱼ+mL`, `|m| ≤ max(1, ⌈w/(√k L)⌉)`, each
    # contributing a constant phase `exp(-ikpmL)` to its run of window nodes; `q` is reduced to `[0, L)`.
    # This supports arbitrary physical-arclength discretizations, including zero-weight nodes.
    if !full_p
        (iszero(first(ps)) && all(p->p>=zero(T),ps)) || throw(ArgumentError("full_p=false requires a nonnegative momentum grid starting at p=0"))
        _is_global_phase_real(u) || throw(ArgumentError("full_p=false requires u to be real up to a global phase; pass full_p=true for genuinely complex boundary functions"))
    end
    width = T(w)/sqrt(k)
    ncopies = max(1,ceil(Int,width/L))
    nimg = 2ncopies+1
    N = length(s); nq = length(qs); np = length(ps)
    F = Matrix{Complex{T}}(undef,N,np)
    img_phase = Matrix{Complex{T}}(undef,nimg,np)
    @inbounds for ip in 1:np
        kp = k*ps[ip]
        for j in 1:N
            sn,cs = sincos(kp*s[j])
            F[j,ip] = Complex{T}(u[j]*ds[j])*Complex(cs,-sn)
        end
        for mi in 1:nimg
            sn,cs = sincos(kp*(mi-ncopies-1)*L)
            img_phase[mi,ip] = Complex(cs,-sn)
        end
    end
    Hp = zeros(T,np,nq)
    g = Vector{T}(undef,0)
    run_lo = Vector{Int}(undef,nimg)
    run_hi = Vector{Int}(undef,nimg)
    run_start = Vector{Int}(undef,nimg)
    run_img = Vector{Int}(undef,nimg)
    @inbounds for iq in 1:nq
        q = mod(qs[iq],L)
        nruns = 0
        W = 0
        for mi in 1:nimg
            off = (mi-ncopies-1)*L
            lo = searchsortedfirst(s,q-width-off)
            hi = searchsortedlast(s,q+width-off)
            hi<lo && continue
            nruns += 1
            run_lo[nruns] = lo; run_hi[nruns] = hi; run_start[nruns] = W-lo+1; run_img[nruns] = mi
            length(g)<W+hi-lo+1 && resize!(g,W+hi-lo+1)
            for j in lo:hi
                d = s[j]+off-q
                g[W+j-lo+1] = exp(-T(0.5)*k*d*d)
            end
            W += hi-lo+1
        end
        for ip in 1:np
            sracc = zero(T); siacc = zero(T)
            for r in 1:nruns
                sr = zero(T); si = zero(T)
                off = run_start[r]
                for j in run_lo[r]:run_hi[r]
                    gj = g[off+j]
                    f = F[j,ip]
                    sr += gj*real(f)
                    si += gj*imag(f)
                end
                ph = img_phase[run_img[r],ip]
                sracc += sr*real(ph)-si*imag(ph)
                siacc += sr*imag(ph)+si*real(ph)
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
    return HusimiFunction(H,collect(qs),ps_out,k)
end

################################################################################
# AUTOMATIC PHASE-SPACE GRID
################################################################################

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
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. Automatically escalated to `true` if `u` is not real up to a global phase.

## Returns
* [`HusimiFunction`](@ref) with the normalized density `H`, the boundary-position grid `qs`, the full signed momentum grid `ps` and `k`. It iterates as `(H, qs, ps)`.
"""
function husimi_function(k::T, s::AbstractVector{T}, ds::AbstractVector{T}, u::AbstractVector{Num}, L::T; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    full_p = full_p || !_is_global_phase_real(u)
    sig = inv(sqrt(k))
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
* [`HusimiFunction`](@ref) with the normalized density `H`, the boundary-position grid `qs`, the full signed momentum grid `ps` and `k`.
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
* [`HusimiFunction`](@ref) with the normalized density `H`, the boundary-position grid `qs`, the full signed momentum grid `ps` and `k`.
"""
function husimi_function(k::T, pts::BoundaryPoints{T}, u::AbstractVector{Num}, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real,Num<:Number}
    L = sum(pts.ds)
    return husimi_function(k,pts.s,pts.ds,u,L,qs,ps;w=w,full_p=full_p)
end

################################################################################
# BOUNDARYFUNCTION API
################################################################################

"""
    husimi_function(bf::BoundaryFunction; c::Real=10.0, w::Real=7.0, full_p::Bool=false) → HusimiFunction

Compute the Poincare-Husimi function of the boundary function `bf` on its outer
loop, with `k = bf.k` and period `bf.L`, on an automatically generated
phase-space grid (see [`husimi_function(k, s, ds, u, L)`](@ref)).

## Keyword Arguments
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. Automatically escalated to `true` if `bf.u` is not real up to a global phase.
"""
function husimi_function(bf::BoundaryFunction; c::Real=10.0, w::Real=7.0, full_p::Bool=false)
    return husimi_function(bf.k,_outer_loop(bf.s,bf.outer),_outer_loop(bf.ds,bf.outer),_outer_loop(bf.u,bf.outer),bf.L;c=c,w=w,full_p=full_p)
end

"""
    husimi_function(bf::BoundaryFunction, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real} → HusimiFunction

Compute the Poincare-Husimi function of the boundary function `bf` on its outer
loop on the prescribed phase-space grids `qs` and `ps`.

## Keyword Arguments
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. `full_p=false` requires a nonnegative `ps` starting at `0` and `bf.u` real up to a global phase.
"""
function husimi_function(bf::BoundaryFunction{T}, qs::AbstractVector{T}, ps::AbstractVector{T}; w::Real=7.0, full_p::Bool=false) where {T<:Real}
    return husimi_function(bf.k,_outer_loop(bf.s,bf.outer),_outer_loop(bf.ds,bf.outer),_outer_loop(bf.u,bf.outer),bf.L,qs,ps;w=w,full_p=full_p)
end

################################################################################
# EIGENSTATE API
################################################################################

"""
    husimi_function(state::BIMEigenstate, billiard::AbsBilliard, solver::AbsBIMSolver; c::Real=10.0, w::Real=7.0, full_p::Bool=false)

Compute the Poincare-Husimi function of a BIM eigenstate.

If the physical boundary normal derivative `state.u` is available, it is used
directly. Otherwise it is reconstructed at `state.k` with [`solve_state`](@ref)
using the boundary discretization stored in `state.pts`, dispatching on
`_bim_kernel(solver)`.

For a billiard with holes the Husimi function lives on the outer boundary
component only: `qs` runs over the outer boundary arclength `[0,L_outer)` and
the hole loops are excluded.

## Arguments
* `state::BIMEigenstate`: BIM eigenstate.
* `billiard::AbsBilliard`: Billiard the state was computed on.
* `solver::AbsBIMSolver`: Solver the state was computed with.

## Keyword Arguments
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* [`HusimiFunction`](@ref) with `k = state.k`; it iterates as `(H, qs, ps)`.
"""
function husimi_function(state::BIMEigenstate, billiard::AbsBilliard, solver::AbsBIMSolver; c::Real=10.0, w::Real=7.0, full_p::Bool=false)
    return husimi_function(boundary_function(state, billiard, solver); c, w, full_p)
end

"""
    husimi_function(state::BIMEigenstate, billiard::AbsBilliard; c::Real=10.0, w::Real=7.0, full_p::Bool=false)

Solver-free Husimi function of a BIM eigenstate with stored `u`, on the outer boundary loop.
Throws an `ArgumentError` when `state.u === nothing`. Keywords as in the solver-taking method.

## Returns
* [`HusimiFunction`](@ref) with `k = state.k`.
"""
function husimi_function(state::BIMEigenstate, billiard::AbsBilliard; c::Real=10.0, w::Real=7.0, full_p::Bool=false)
    return husimi_function(boundary_function(state, billiard); c, w, full_p)
end

"""
    husimi_function(state::BasisEigenstate, billiard::AbsBilliard; ppw::Real=5.0, c::Real=10.0, w::Real=7.0, multithreaded::Bool=true, full_p::Bool=false)

Compute the Poincare-Husimi function of a basis eigenstate.

## Arguments
* `state::BasisEigenstate`: Eigenstate represented in a basis.
* `billiard::AbsBilliard`: Billiard the state was computed on.

## Keyword Arguments
* `ppw::Real=5.0`: Boundary-function oversampling factor.
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `multithreaded::Bool=true`: Whether boundary-function construction is multithreaded.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* [`HusimiFunction`](@ref) with `k = state.k`; it iterates as `(H, qs, ps)`.
"""
function husimi_function(state::BasisEigenstate, billiard::AbsBilliard; ppw::Real=5.0, c::Real=10.0, w::Real=7.0, multithreaded::Bool=true, full_p::Bool=false)
    return husimi_function(boundary_function(state, billiard; ppw, multithreaded); c, w, full_p)
end

################################################################################
# MULTIPLE EIGENSTATES
################################################################################

"""
    husimi_function(states::AbstractVector{<:BIMEigenstate{K,T}}, billiard::AbsBilliard, solver::AbsBIMSolver; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {K,T<:Real}

Compute Poincare-Husimi functions for several BIM eigenstates.

Each state uses its natural automatically generated phase-space grid.

## Arguments
* `states::AbstractVector{<:BIMEigenstate}`: BIM eigenstates.
* `billiard::AbsBilliard`: Billiard the states were computed on.
* `solver::AbsBIMSolver`: Solver the states were computed with.

## Keyword Arguments
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false. 

## Returns
* `Vector{HusimiFunction}`: One [`HusimiFunction`](@ref) per state; states whose computation fails are dropped (logged with `@debug`), so the result may be shorter than `states`.
"""
function husimi_function(states::AbstractVector{<:BIMEigenstate{K,T}}, billiard::AbsBilliard, solver::AbsBIMSolver; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {K,T<:Real}
    return _collect_functions(HusimiFunction{T}, states; multithreaded=true) do state
        husimi_function(state, billiard, solver; c, w, full_p)
    end
end

"""
    husimi_function(states::AbstractVector{<:BIMEigenstate}, billiard::AbsBilliard; c::Real=10.0, w::Real=7.0, full_p::Bool=false) → Vector{HusimiFunction}

Solver-free Husimi functions of several states that store `u`; see the single-state method.
"""
function husimi_function(states::AbstractVector{<:BIMEigenstate{K,T}}, billiard::AbsBilliard; c::Real=10.0, w::Real=7.0, full_p::Bool=false) where {K,T<:Real}
    return _collect_functions(HusimiFunction{T}, states; multithreaded=true) do state
        husimi_function(state, billiard; c, w, full_p)
    end
end

"""
    husimi_function(states::AbstractVector{<:BasisEigenstate}, billiard::AbsBilliard; ppw::Real=5.0, c::Real=10.0, w::Real=7.0, multithreaded::Bool=true, full_p::Bool=false)

Compute Poincare-Husimi functions for several basis eigenstates.

## Arguments
* `states::AbstractVector{<:BasisEigenstate}`: Eigenstates represented in a basis.
* `billiard::AbsBilliard`: Billiard the states were computed on.

## Keyword Arguments
* `ppw::Real=5.0`: Boundary-function oversampling factor.
* `c::Real=10.0`: Phase-space sampling density.
* `w::Real=7.0`: Gaussian truncation radius in units of `1/√k`.
* `multithreaded::Bool=true`: Whether each boundary-function construction is multithreaded.
* `full_p::Bool=false`: Whether both momentum signs are evaluated explicitly. For 1d irreps (u real-valued up to a global complex phase), where we have p -> -p symmetry this should be false.

## Returns
* `Vector{HusimiFunction}`: One [`HusimiFunction`](@ref) per state; states whose computation fails are dropped (logged with `@debug`), so the result may be shorter than `states`.
"""
function husimi_function(states::AbstractVector{<:BasisEigenstate}, billiard::AbsBilliard; ppw::Real=5.0, c::Real=10.0, w::Real=7.0, multithreaded::Bool=true, full_p::Bool=false)
    T = typeof(real(first(states).k))
    bfs = boundary_function(states, billiard; ppw, multithreaded)
    return _collect_functions(HusimiFunction{T}, bfs) do bf
        husimi_function(bf; c, w, full_p)
    end
end

################################################################################
# SpectralData API
################################################################################

"""
    husimi_function(data::SpectralData; kwargs...) → Vector{HusimiFunction}

Compute the [`HusimiFunction`](@ref) of every eigenstate stored in `data`.
"""
function husimi_function(data::SpectralData; kwargs...)
    return _apply_state(husimi_function, _require_states(data), data.billiard, data.solver; kwargs...)
end

"""
    husimi_function(data::SpectralData, i::Integer; kwargs...) → HusimiFunction

Compute the [`HusimiFunction`](@ref) of the `i`-th eigenstate stored in `data`.
"""
function husimi_function(data::SpectralData, i::Integer; kwargs...)
    return _apply_state(husimi_function, _require_states(data)[i], data.billiard, data.solver; kwargs...)
end

################################################################################
# SHARED STATE DISPATCH
#
# The `SpectralData` entry points of `wavefunction`, `boundary_function`,
# `momentum_function` and `husimi_function` forward one keyword set to every
# state kind. A keyword valid only for the other kind is dropped by
# `_foreign_kwargs`; names outside both kinds still raise a `MethodError`.
################################################################################

_drop_kwargs(kwargs, ignored::Tuple{Vararg{Symbol}}) = (; (p for p in pairs(kwargs) if !(first(p) in ignored))...)

_foreign_kwargs(f, ::Type{<:AbsState}) = ()
_foreign_kwargs(::typeof(wavefunction), ::Type{<:BIMEigenstate}) = (:memory_limit, :multithreaded)
_foreign_kwargs(::typeof(wavefunction), ::Type{<:BasisEigenstate}) = (:min_chunk, :use_chebyshev, :show_progress, :cheb_config)
_foreign_kwargs(::typeof(boundary_function), ::Type{<:BIMEigenstate}) = (:ppw,)
_foreign_kwargs(::typeof(momentum_function), ::Type{<:BIMEigenstate}) = (:ppw,)
_foreign_kwargs(::typeof(husimi_function), ::Type{<:BIMEigenstate}) = (:ppw, :multithreaded)

# Apply `f` to a state (or vector of states) of either kind: BIM states take `(billiard, solver)`, basis states
# take `billiard` only.
function _apply_state(f, state_or_states::Union{BIMEigenstate,AbstractVector{<:BIMEigenstate}}, billiard::AbsBilliard, solver::AbsBIMSolver; kwargs...)
    kw = _drop_kwargs(kwargs, _foreign_kwargs(f, BIMEigenstate))
    return f(state_or_states, billiard, solver; kw...)
end

# `u`-states of a solver without a boundary kernel (basis solvers with `states = :boundary`) are read solver-free.
function _apply_state(f, state_or_states::Union{BIMEigenstate,AbstractVector{<:BIMEigenstate}}, billiard::AbsBilliard, solver; kwargs...)
    kw = _drop_kwargs(kwargs, (_foreign_kwargs(f, BIMEigenstate)..., :multithreaded))
    return f(state_or_states, billiard; kw...)
end

function _apply_state(f, state_or_states::Union{BasisEigenstate,AbstractVector{<:BasisEigenstate}}, billiard::AbsBilliard, solver; kwargs...)
    kw = _drop_kwargs(kwargs, _foreign_kwargs(f, BasisEigenstate))
    return f(state_or_states, billiard; kw...)
end

# Collect `f(state)` over `states` as `Vector{F}`, dropping states for which `f` throws (logged with `@debug`).
function _collect_functions(f, ::Type{F}, states::AbstractVector; multithreaded::Bool=false) where {F}
    n = length(states)
    out = Vector{F}(undef, n)
    ok = fill(true, n)
    pbar = Progress(n; desc="Constructing $(nameof(F)) objects, N=$n")
    progress_lock = ReentrantLock()
    function run!(i)
        try
            out[i] = f(states[i])
        catch e
            @debug "$(nameof(F)) fail at k=$(states[i].k)" exception=(e, catch_backtrace())
            ok[i] = false
        end
        lock(progress_lock) do
            next!(pbar)
        end
    end
    if multithreaded
        Threads.@threads for i in eachindex(states)
            run!(i)
        end
    else
        for i in eachindex(states)
            run!(i)
        end
    end
    return out[ok]
end