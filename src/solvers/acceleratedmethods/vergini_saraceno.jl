abstract type AbsScalingMethod<:AcceleratedSolver end

"""
    VerginiSaracenoSolver{T}<:AbsScalingMethod where {T<:Real}

Concrete scaling-method solver implementing the Vergini-Saraceno method for
real quantum-billiard spectra.

The method evaluates a real basis on the physical fundamental boundary,
constructs the real symmetric matrices

F = Gᵀ W G, Fk = Gᵀ W dG + dGᵀ W G.

and solves the generalized eigenvalue problem associated with the scaling
method.

The Vergini-Saraceno boundary weight

wᵢ = dsᵢ/((xᵢ-c₀) ⋅ nᵢ)

is stored in `BoundaryPoints.w`.

## Attributes
* `dim_scaling_factor::T`: Scaling factor used to determine the real basis dimension.
* `pts_scaling_factor::Vector{T}`: Boundary-point scaling factors.
* `sampler::Vector`: Samplers used on the physical fundamental boundary.
* `scaling_origin::SVector{2,T}`: c₀: Origin used for scaling the boundary points.
* `eps::T`: Relative tolerance used in the generalized eigensolver.
* `min_dim::Int64`: Minimum basis dimension.
* `min_pts::Int64`: Minimum number of boundary sampling points.
"""
struct VerginiSaracenoSolver{T}<:AbsScalingMethod where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
    scaling_origin::SVector{2,T}
end

"""
    VerginiSaracenoSolver(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}};min_dim::Int=100,min_pts::Int=500) where {T<:Real} → VerginiSaracenoSolver{T}

Construct a [`VerginiSaracenoSolver`](@ref) using `GaussLegendreNodes()` for
boundary sampling.

## Arguments
* `dim_scaling_factor::T`: Basis-dimension scaling factor.
* `pts_scaling_factor::Union{T,Vector{T}}`: Boundary-point scaling factor, or one factor per fundamental-boundary curve.

## Keyword Arguments
* `min_dim::Int=100`: Minimum basis dimension.
* `min_pts::Int=500`: Minimum number of boundary sampling points.
* `scaling_origin::SVector{2,T}=SVector{2,T}(zero(T),zero(T))`: Origin used for scaling the boundary points.

## Returns
* `solver::VerginiSaracenoSolver{T}`: Constructed solver.
"""
function VerginiSaracenoSolver(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}};min_dim::Int=100,min_pts::Int=500,scaling_origin::SVector{2,T}=SVector{2,T}(zero(T),zero(T))) where {T<:Real}
    bs=pts_scaling_factor isa T ? T[pts_scaling_factor] : pts_scaling_factor
    return VerginiSaracenoSolver(dim_scaling_factor,bs,[BilliardGeometry.GaussLegendreNodes()],eps(T),min_dim,min_pts,scaling_origin)
end

"""
    VerginiSaracenoSolver(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}},samplers::Vector{Sam};min_dim::Int=100,min_pts::Int=500) where {T<:Real,Sam<:AbsSampler} → VerginiSaracenoSolver{T}

Construct a [`VerginiSaracenoSolver`](@ref) with explicitly supplied boundary
samplers.

## Arguments
* `dim_scaling_factor::T`: Basis-dimension scaling factor.
* `pts_scaling_factor::Union{T,Vector{T}}`: Boundary-point scaling factor, or one factor per fundamental-boundary curve.
* `samplers::Vector{Sam}`: Boundary samplers.

## Keyword Arguments
* `min_dim::Int=100`: Minimum basis dimension.
* `min_pts::Int=500`: Minimum number of boundary sampling points.
* `scaling_origin::SVector{2,T}=SVector{2,T}(zero(T),zero(T))`: Origin used for scaling the boundary points.

## Returns
* `solver::VerginiSaracenoSolver{T}`: Constructed solver.
"""
function VerginiSaracenoSolver(dim_scaling_factor::T,pts_scaling_factor::Union{T,Vector{T}},samplers::Vector{Sam};min_dim::Int=100,min_pts::Int=500,scaling_origin::SVector{2,T}=SVector{2,T}(zero(T),zero(T))) where {T<:Real,Sam<:BilliardGeometry.AbsSampler}
    bs=pts_scaling_factor isa T ? T[pts_scaling_factor] : pts_scaling_factor
    return VerginiSaracenoSolver(dim_scaling_factor,bs,samplers,eps(T),min_dim,min_pts,scaling_origin)
end

"""
    evaluate_points(solver::VerginiSaracenoSolver,billiard::Bi,k::T) where {Bi<:BilliardGeometry.AbsBilliard,T<:Real} → BoundaryPoints

Sample the physical fundamental boundary used by the Vergini-Saraceno solver.

The Vergini-Saraceno weight is

    wᵢ = dsᵢ/((xᵢ-c₀)⋅nᵢ),

where `c₀=solver.scaling_origin` is the same dilation origin used by the
basis wavenumber derivative entering `Fk`.

## Arguments
* `solver::VerginiSaracenoSolver`: Scaling-method solver.
* `billiard::Bi`: Billiard geometry.
* `k::T`: Real scaling wavenumber controlling the boundary-point density.

## Returns
* `pts::BoundaryPoints`: Fundamental-boundary discretization containing `xy`,
  `normal`, `s`, `ds` and the VS weights `w`.
"""
function evaluate_points(solver::VerginiSaracenoSolver,billiard::Bi,k::T) where {Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    bs,samplers=adjust_scaling_and_samplers(solver,billiard)
    curves=BilliardGeometry.get_boundary_curves(billiard)
    W=eltype(solver.pts_scaling_factor)
    Ns=_determine_bp_sizes(curves,bs,k)
    M=length(Ns)
    xy_all=Vector{Vector{SVector{2,W}}}(undef,M)
    normal_all=Vector{Vector{SVector{2,W}}}(undef,M)
    s_all=Vector{Vector{W}}(undef,M)
    ds_all=Vector{Vector{W}}(undef,M)
    w_all=Vector{Vector{W}}(undef,M)
    c0=solver.scaling_origin
    L0=zero(W)
    @inbounds for i in eachindex(curves)
        xy,normal,s,ds=boundary_coords(curves[i],samplers[i],Ns[i])
        rn=Vector{W}(undef,length(xy))
        @inbounds @simd for j in eachindex(xy)
            rn[j]=dot(xy[j]-c0,normal[j])
        end
        tol_rn=sqrt(eps(W))
        rnmin=minimum(rn)
        rnmin>=-tol_rn||throw(ArgumentError("VS scaling origin $c0 is outside the star kernel: minimum (x-c0)⋅n=$rnmin"))
        xy_all[i]=xy
        normal_all[i]=normal
        s_all[i]=s.+L0
        ds_all[i]=ds
        w_all[i]=ds./rn
        L0+=curves[i].length
    end
    return BoundaryPoints(vcat(xy_all...);normal=vcat(normal_all...),s=vcat(s_all...),ds=vcat(ds_all...),w=vcat(w_all...))
end

# internal check so that the scaling origin of the basis and the solver match. This is important since the dk_fun uses the scaling origin of the basis to compute the derivative with respect to k
@inline function check_basis_origin(solver::VerginiSaracenoSolver,basis::AbsBasis)
    T=eltype(solver.scaling_origin)
    c0=basis_origin_check(basis,T)
    solver.scaling_origin==c0||throw(ArgumentError("inconsistent VS scaling origins: solver=$(solver.scaling_origin), basis=$c0"))
    return nothing
end

"""
    construct_matrices(solver::VerginiSaracenoSolver,basis::Ba,pts::BoundaryPoints,k::T;multithreaded::Bool=true) where {Ba<:AbsBasis,T<:Real} → Tuple

Construct the real Vergini-Saraceno matrices

F = Gᵀ W G, Fk = Gᵀ W dG + dGᵀ W G.

The diagonal weight matrix `W` is not formed explicitly. Instead the rows of
`G` and `G_k` are scaled by `sqrt(w)` and the symmetric products are assembled
with BLAS `syrk!` and `syr2k!`.

Historically this code multiplied all fundamental-boundary weights by a scalar
`nsym` equal to the number of reflected symmetry copies. That factor multiplies
both `F` and `Fk` by the same constant and therefore cancels exactly from the
generalized eigenvalues. Symmetry is already encoded in the adapted basis, so
the legacy `nsym` factor is no longer part of the method. The value `1` is
passed only to preserve the existing `_scale_rows_sqrtw!` helper interface.

## Arguments
* `solver::VerginiSaracenoSolver`: Scaling-method solver.
* `basis::Ba`: Real basis used to evaluate `G` and `G_k`.
* `pts::BoundaryPoints`: Fundamental-boundary points with VS weights stored in `pts.w`.
* `k::T`: Real scaling wavenumber.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded basis evaluation.

## Returns
* `F::Matrix`: Real weighted boundary matrix.
* `Fk::Matrix`: Real wavenumber derivative matrix.
"""
function construct_matrices(solver::VerginiSaracenoSolver,basis::Ba,pts::BoundaryPoints,k::T;multithreaded::Bool=true) where {Ba<:AbsBasis,T<:Real}
    @timeit_debug "construct_matrices" begin
        xy=pts.xy
        w=pts.w
        N=basis.dim
        M=length(xy)
        @debug "Matrix construction started" N M k
        @timeit_debug "basis_matrix" begin
            G=basis_matrix(basis,k,xy;multithreaded)
        end
        @debug "Basis matrix computed" size=size(G)
        @timeit_debug "dk_matrix" begin
            dG=dk_matrix(basis,k,xy;multithreaded)
        end
        @debug "Derivative matrix computed" size=size(dG)
        # Legacy code used nsym=2 or 4 for reflected copies. This is a common
        # scalar factor in F and Fk and cancels from the generalized spectrum.
        # Keep multiplier=1 only because _scale_rows_sqrtw! currently accepts it.
        weight_scale=one(eltype(w))
        @timeit_debug "compute_F" begin
            _scale_rows_sqrtw!(G,w,weight_scale)
            F=Matrix{eltype(G)}(undef,N,N)
            @blas_multi MAX_BLAS_THREADS BLAS.syrk!('U','T',one(eltype(G)),G,zero(eltype(G)),F)
            _symmetrize_from_upper!(F)
        end
        @debug "F computed" size=size(F)
        @timeit_debug "compute_Fk" begin
            _scale_rows_sqrtw!(dG,w,weight_scale)
            Fk=Matrix{eltype(G)}(undef,N,N)
            @blas_multi_then_1 MAX_BLAS_THREADS BLAS.syr2k!('U','T',one(eltype(G)),G,dG,zero(eltype(G)),Fk)
            _symmetrize_from_upper!(Fk)
        end
        @debug "Fk computed" size=size(Fk)
        return F,Fk
    end
end

"""
    sm_results(mu::AbstractVector{T},k::T) where {T<:Real} → Tuple

Convert real generalized eigenvalues `mu` obtained at real scaling wavenumber
`k` into Vergini-Saraceno wavenumber estimates and tensions,

k_j = k - 2/μ_j + 2/(k μ_j²),
t_j = 2(2/μ_j)².

## Returns
* `ks::Vector{T}`: Estimated real wavenumbers.
* `ten::Vector{T}`: Corresponding real tensions.
"""
function sm_results(mu::AbstractVector{T},k::T) where {T<:Real}
    ks=k .-2 ./mu .+2/k ./(mu .^2)
    ten=2 .*(2 ./mu) .^2
    return ks,ten
end

"""
    sm_window_results(mu::AbstractVector{T},k::T,dk::T) where {T<:Real} → Tuple

Convert generalized Vergini-Saraceno eigenvalues into local wavenumber
estimates, retaining only the asymptotic branches whose leading-order shift

    |2/μ| < dk

already lies inside the scaling window. This prevents small-|μ| branches from
producing false local roots through cancellation between the first- and
second-order terms in

    k_j = k - 2/μ_j + 2/(k μ_j²).
"""
function sm_window_results(mu::AbstractVector{T},k::T,dk::T) where {T<:Real}
    idx=findall(i->isfinite(mu[i])&&abs(2/mu[i])<dk,eachindex(mu))
    isempty(idx)&&return T[],T[],idx
    muf=mu[idx]
    ks,ten=sm_results(muf,k)
    keep=isfinite.(ks).&(abs.(ks.-k).<dk)
    return ks[keep],ten[keep],idx[keep]
end

"""
    solve(solver::VerginiSaracenoSolver,basis::Ba,pts::BoundaryPoints,k::T,dk::T;multithreaded::Bool=true) where {Ba<:AbsBasis,T<:Real} → Tuple

Solve one real Vergini-Saraceno scaling window centered at `k`.

Only candidates satisfying `abs(ks-k)<dk` are retained and the accepted
wavenumbers are returned in ascending order.

## Returns
* `ks::Vector{T}`: Sorted accepted real wavenumbers.
* `ten::Vector{T}`: Corresponding tensions.
"""
function solve(solver::VerginiSaracenoSolver,basis::Ba,pts::BoundaryPoints,k::T,dk::T;multithreaded::Bool=true) where {Ba<:AbsBasis,T<:Real}
    F,Fk=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mu=generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks,ten,_=sm_window_results(mu,k,dk)
    p=sortperm(ks)
    return ks[p],ten[p]
end

"""

    solve(solver::VerginiSaracenoSolver,F::AbstractMatrix{T},Fk::AbstractMatrix{T},k::T,dk::T) where {T<:Real} → Tuple

Solve one real Vergini-Saraceno scaling window from preassembled matrices `F` and `Fk`.
Only generalized-eigenvalue branches satisfying `abs(2/mu)<dk` are passed to
the second-order scaling formula. The refined estimate must then also satisfy
`abs(ks-k)<dk`.
## Returns
* `ks::Vector{T}`: Sorted accepted real wavenumbers.
* `ten::Vector{T}`: Corresponding tensions.
"""

function solve(solver::VerginiSaracenoSolver,F::AbstractMatrix{T},Fk::AbstractMatrix{T},k::T,dk::T) where {T<:Real}
    @blas_multi_then_1 MAX_BLAS_THREADS mu=generalized_eigvals(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks,ten,_=sm_window_results(mu,k,dk)
    p=sortperm(ks)
    return ks[p],ten[p]
end

"""
    solve_vectors(solver::VerginiSaracenoSolver,basis::Ba,pts::BoundaryPoints,k::T,dk::T;multithreaded::Bool=true) where {Ba<:AbsBasis,T<:Real} → Tuple

Solve one real Vergini-Saraceno scaling window and retain the real
basis-expansion coefficient vectors of the accepted states.

## Returns
* `ks::Vector{T}`: Sorted accepted real wavenumbers.
* `ten::Vector{T}`: Corresponding tensions.
* `X::Matrix{T}`: Real basis coefficient matrix, one state per column.
"""
function solve_vectors(solver::VerginiSaracenoSolver,basis::Ba,pts::BoundaryPoints,k::T,dk::T;multithreaded::Bool=true) where {Ba<:AbsBasis,T<:Real}
    F,Fk=construct_matrices(solver,basis,pts,k;multithreaded=multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mu,Z,C=generalized_eigen(Symmetric(F),Symmetric(Fk);eps=solver.eps)
    ks,ten,idx=sm_window_results(mu,k,dk)
    Z=Z[:,idx]
    X=C*Z
    X=(sqrt.(ten))'.*X
    p=sortperm(ks)
    return ks[p],ten[p],X[:,p]
end

"""
    is_equal(x::T,dx::T,y::T,dy::T) where {T<:Real} → Bool

Return whether the intervals `[x-dx,x+dx]` and `[y-dy,y+dy]` overlap.
"""
@inline function is_equal(x::T,dx::T,y::T,dy::T) where {T<:Real}
    return max(x-dx,y-dy)<=min(x+dx,y+dy)
end

"""
    match_wavenumbers(ks_l::Vector{T},ts_l::Vector{T},ks_r::Vector{T},ts_r::Vector{T}) where {T<:Real} → Tuple

Merge two sorted real spectral lists. When two candidates overlap according to
[`is_equal`](@ref), retain the candidate with the smaller tension.

## Returns
* `ks::Vector{T}`: Merged wavenumbers.
* `ts::Vector{T}`: Corresponding tensions.
* `control::Vector{Bool}`: `true` for entries created by overlap resolution.
"""
function match_wavenumbers(ks_l::Vector{T},ts_l::Vector{T},ks_r::Vector{T},ts_r::Vector{T}) where {T<:Real}
    i=1
    j=1
    ks=T[]
    ts=T[]
    control=Bool[]
    while i<=length(ks_l)&&j<=length(ks_r)
        x,dx=ks_l[i],ts_l[i]
        y,dy=ks_r[j],ts_r[j]
        if is_equal(x,dx,y,dy)
            if dx<dy
                push!(ks,x)
                push!(ts,dx)
            else
                push!(ks,y)
                push!(ts,dy)
            end
            push!(control,true)
            i+=1
            j+=1
        elseif x<y
            push!(ks,x)
            push!(ts,dx)
            push!(control,false)
            i+=1
        else
            push!(ks,y)
            push!(ts,dy)
            push!(control,false)
            j+=1
        end
    end
    while i<=length(ks_l)
        push!(ks,ks_l[i])
        push!(ts,ts_l[i])
        push!(control,false)
        i+=1
    end
    while j<=length(ks_r)
        push!(ks,ks_r[j])
        push!(ts,ts_r[j])
        push!(control,false)
        j+=1
    end
    return ks,ts,control
end

"""
    match_wavenumbers_with_X(ks_l::Vector{T},ts_l::Vector{T},X_l::Vector{Vector{T}},ks_r::Vector{T},ts_r::Vector{T},X_r::Vector{Vector{T}}) where {T<:Real} → Tuple

Merge two sorted real state lists exactly as [`match_wavenumbers`](@ref), while
carrying the corresponding real basis coefficient vectors.

## Returns
* `ks::Vector{T}`: Merged wavenumbers.
* `ts::Vector{T}`: Corresponding tensions.
* `Xs::Vector{Vector{T}}`: Coefficient vector associated with each retained state.
* `control::Vector{Bool}`: `true` for entries created by overlap resolution.
"""
function match_wavenumbers_with_X(ks_l::Vector{T},ts_l::Vector{T},X_l::Vector{Vector{T}},ks_r::Vector{T},ts_r::Vector{T},X_r::Vector{Vector{T}}) where {T<:Real}
    i=1
    j=1
    ks=T[]
    ts=T[]
    Xs=Vector{Vector{T}}()
    control=Bool[]
    while i<=length(ks_l)&&j<=length(ks_r)
        x,dx,Xx=ks_l[i],ts_l[i],X_l[i]
        y,dy,Xy=ks_r[j],ts_r[j],X_r[j]
        if is_equal(x,dx,y,dy)
            if dx<dy
                push!(ks,x)
                push!(ts,dx)
                push!(Xs,Xx)
            else
                push!(ks,y)
                push!(ts,dy)
                push!(Xs,Xy)
            end
            push!(control,true)
            i+=1
            j+=1
        elseif x<y
            push!(ks,x)
            push!(ts,dx)
            push!(Xs,Xx)
            push!(control,false)
            i+=1
        else
            push!(ks,y)
            push!(ts,dy)
            push!(Xs,Xy)
            push!(control,false)
            j+=1
        end
    end
    while i<=length(ks_l)
        push!(ks,ks_l[i])
        push!(ts,ts_l[i])
        push!(Xs,X_l[i])
        push!(control,false)
        i+=1
    end
    while j<=length(ks_r)
        push!(ks,ks_r[j])
        push!(ts,ts_r[j])
        push!(Xs,X_r[j])
        push!(control,false)
        j+=1
    end
    return ks,ts,Xs,control
end

"""
    overlap_and_merge!(k_left::Vector{T},ten_left::Vector{T},k_right::Vector{T},ten_right::Vector{T},control_left::Vector{Bool},kl::T,kr::T;tol=1e-3) where {T<:Real} → Nothing

Merge a right real spectral window into the accumulated left result over the
overlap interval `[kl-tol,kr+tol]`.
"""
function overlap_and_merge!(k_left::Vector{T},ten_left::Vector{T},k_right::Vector{T},ten_right::Vector{T},control_left::Vector{Bool},kl::T,kr::T;tol=1e-3) where {T<:Real}
    if isempty(k_left)
        append!(k_left,k_right)
        append!(ten_left,ten_right)
        append!(control_left,fill(false,length(k_right)))
        return nothing
    end
    isempty(k_right)&&return nothing
    idx_l=(k_left.>(kl-tol)).&(k_left.<(kr+tol))
    idx_r=(k_right.>(kl-tol)).&(k_right.<(kr+tol))
    ks_l=k_left[idx_l]
    ts_l=ten_left[idx_l]
    ks_r=k_right[idx_r]
    ts_r=ten_right[idx_r]
    ks,ts,control=match_wavenumbers(ks_l,ts_l,ks_r,ts_r)
    del_l=findall(idx_l)
    deleteat!(k_left,del_l)
    deleteat!(ten_left,del_l)
    deleteat!(control_left,del_l)
    append!(k_left,ks)
    append!(ten_left,ts)
    append!(control_left,control)
    fl=findlast(idx_r)
    idx_last=isnothing(fl) ? 1 : fl+1
    append!(k_left,k_right[idx_last:end])
    append!(ten_left,ten_right[idx_last:end])
    append!(control_left,fill(false,max(0,length(k_right)-idx_last+1)))
    return nothing
end

"""
    overlap_and_merge_state!(k_left::Vector{T},ten_left::Vector{T},X_left::Vector{Vector{T}},k_right::Vector{T},ten_right::Vector{T},X_right::Vector{Vector{T}},control_left::Vector{Bool},kl::T,kr::T;tol=1e-3) where {T<:Real} → Nothing

State-carrying version of [`overlap_and_merge!`](@ref).
"""
function overlap_and_merge_state!(k_left::Vector{T},ten_left::Vector{T},X_left::Vector{Vector{T}},k_right::Vector{T},ten_right::Vector{T},X_right::Vector{Vector{T}},control_left::Vector{Bool},kl::T,kr::T;tol=1e-3) where {T<:Real}
    if isempty(k_left)
        append!(k_left,k_right)
        append!(ten_left,ten_right)
        append!(X_left,X_right)
        append!(control_left,fill(false,length(k_right)))
        return nothing
    end
    isempty(k_right)&&return nothing
    idx_l=(k_left.>(kl-tol)).&(k_left.<(kr+tol))
    idx_r=(k_right.>(kl-tol)).&(k_right.<(kr+tol))
    ks_l=k_left[idx_l]
    ts_l=ten_left[idx_l]
    Xs_l=X_left[idx_l]
    ks_r=k_right[idx_r]
    ts_r=ten_right[idx_r]
    Xs_r=X_right[idx_r]
    ks,ts,Xs,control=match_wavenumbers_with_X(ks_l,ts_l,Xs_l,ks_r,ts_r,Xs_r)
    del_l=findall(idx_l)
    deleteat!(k_left,del_l)
    deleteat!(ten_left,del_l)
    deleteat!(X_left,del_l)
    deleteat!(control_left,del_l)
    append!(k_left,ks)
    append!(ten_left,ts)
    append!(X_left,Xs)
    append!(control_left,control)
    fl=findlast(idx_r)
    idx_last=isnothing(fl) ? 1 : fl+1
    append!(k_left,k_right[idx_last:end])
    append!(ten_left,ten_right[idx_last:end])
    append!(X_left,X_right[idx_last:end])
    append!(control_left,fill(false,max(0,length(k_right)-idx_last+1)))
    return nothing
end

"""
    SpectralData{T}

Store a merged real spectrum without basis coefficient vectors.

## Attributes
* `ks::Vector{T}`: Retained real wavenumbers.
* `tens::Vector{T}`: Corresponding tensions.
* `control::Vector{Bool}`: `true` for states selected while resolving an overlap.
* `k_min::T`: Minimum retained wavenumber.
* `k_max::T`: Maximum retained wavenumber.
"""
struct SpectralData{T}
    ks::Vector{T}
    tens::Vector{T}
    control::Vector{Bool}
    k_min::T
    k_max::T
end

"""
    SpectralData(ks::Vector{T},tens::Vector{T},control::Vector{Bool}) where {T<:Real} → SpectralData{T}

Construct [`SpectralData`](@ref) and cache the minimum and maximum retained
wavenumbers.
"""
function SpectralData(k::Vector{T},ten::Vector{T},control::Vector{Bool}) where {T<:Real}
    isempty(k)&&throw(ArgumentError("Cannot construct SpectralData from an empty spectrum"))
    return SpectralData(k,ten,control,minimum(k),maximum(k))
end

"""
    StateData{K,T}<:AbsState

Store a merged real Vergini-Saraceno spectrum together with the real
basis-expansion coefficients of every retained state.

The entries of `ks`, `X`, `tens` and `control` have a one-to-one
correspondence.

## Attributes
* `ks::Vector{K}`: Retained real wavenumbers.
* `X::Vector{Vector{T}}`: Real basis-expansion coefficient vector for each wavenumber.
* `tens::Vector{T}`: Tension associated with each wavenumber.
* `control::Vector{Bool}`: `true` for states selected while resolving an overlap.
"""
struct StateData{K,T}<:AbsState
    ks::Vector{K}
    X::Vector{Vector{T}}
    tens::Vector{T}
    control::Vector{Bool}
end

"""
    solve_state_data_bundle(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k::T,dk::T;multithreaded::Bool=true) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real} → StateData

Solve one real Vergini-Saraceno window centered at `k` and retain the real
basis-expansion coefficients of every accepted state.

The `control` vector is initialized to `false`; overlap flags are set when
neighboring windows are merged.
"""
function solve_state_data_bundle(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k::T,dk::T;multithreaded::Bool=true) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    L=sum(crv.length for crv in BilliardGeometry.get_boundary_curves(billiard))
    dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    ks,tens,X=solve_vectors(solver,basis_new,pts,k,dk;multithreaded=multithreaded)
    X_vectors=[Vector(col) for col in eachcol(X)]
    return StateData(ks,X_vectors,tens,fill(false,length(ks)))
end

"""
    compute_eigenstate(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k::T;dk::T=T(0.1),multithreaded::Bool=true) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real} → Eigenstate

Compute the real Vergini-Saraceno states in a window around `k` and return the
state whose refined wavenumber is closest to the requested value.

## Keyword Arguments
* `dk::T=T(0.1)`: Half-width of the scaling window.
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
"""
function compute_eigenstate(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k::T;dk::T=T(0.1),multithreaded::Bool=true) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    L=sum([crv.length for crv in BilliardGeometry.get_boundary_curves(billiard)])
    dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
    basis_new=resize_basis(basis,billiard,dim,k)
    pts=evaluate_points(solver,billiard,k)
    ks,tens,X=solve_vectors(solver,basis_new,pts,k,dk;multithreaded=multithreaded)
    isempty(ks)&&throw(ArgumentError("No Vergini-Saraceno state found within dk=$dk of k=$k"))
    idx=argmin(abs.(ks.-k))
    return Eigenstate(ks[idx],k,X[:,idx],tens[idx],basis_new,billiard)
end

########## LOW-LEVEL SPECTRUM COMPUTATION ##########

function _compute_spectrum(::Val{false},solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k_vals::Vector{T},dk_vals::Vector{T};tol::T,multithreaded_matrices::Bool,multithreaded_ks::Bool) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    isempty(k_vals)&&throw(ArgumentError("Spectrum interval contains no scaling windows"))
    all_ks=Vector{Vector{T}}(undef,length(k_vals))
    all_tens=Vector{Vector{T}}(undef,length(k_vals))
    L=sum([crv.length for crv in BilliardGeometry.get_boundary_curves(billiard)])
    p=Progress(length(k_vals),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(k_vals)
        k=k_vals[i]
        dim=max(solver.min_dim,round(Int,L*k*solver.dim_scaling_factor/(2*pi)))
        basis_new=resize_basis(basis,billiard,dim,k)
        pts=evaluate_points(solver,billiard,k)
        all_ks[i],all_tens[i]=solve(solver,basis_new,pts,k,dk_vals[i]+tol;multithreaded=multithreaded_matrices)
        next!(p)
    end
    k_res=all_ks[1]
    ten_res=all_tens[1]
    control=fill(false,length(k_res))
    @inbounds for i in 2:length(k_vals)
        overlap_and_merge!(k_res,ten_res,all_ks[i],all_tens[i],control,k_vals[i-1],k_vals[i];tol=tol)
    end
    return SpectralData(k_res,ten_res,control)
end

function _compute_spectrum(::Val{true},solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k_vals::Vector{T},dk_vals::Vector{T};tol::T,multithreaded_matrices::Bool,multithreaded_ks::Bool) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    isempty(k_vals)&&throw(ArgumentError("Spectrum interval contains no scaling windows"))
    all_states=Vector{StateData{T,T}}(undef,length(k_vals))
    p=Progress(length(k_vals),1)
    @use_threads multithreading=multithreaded_ks for i in eachindex(k_vals)
        all_states[i]=solve_state_data_bundle(solver,basis,billiard,k_vals[i],dk_vals[i]+tol;multithreaded=multithreaded_matrices)
        next!(p)
    end
    state_res=all_states[1]
    @inbounds for i in 2:length(k_vals)
        overlap_and_merge_state!(state_res.ks,state_res.tens,state_res.X,all_states[i].ks,all_states[i].tens,all_states[i].X,state_res.control,k_vals[i-1],k_vals[i];tol=tol)
    end
    return state_res
end

####################################################

"""
    compute_spectrum(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k1::T,k2::T;vectors::Bool=false,tol::T=T(1e-4),N_expect::Real=1,dk_threshold::T=T(0.05),multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real}

Compute the real Vergini-Saraceno spectrum over `[k1,k2]` using adaptively
spaced scaling windows.

The window spacing is estimated from the leading Weyl density of the physical
fundamental domain, dN/dk = Ak/(2π) - L/(4π).

If `vectors=false`, return [`SpectralData`](@ref). If `vectors=true`, retain
the real basis-expansion coefficients and return [`StateData`](@ref).

## Keyword Arguments
* `vectors::Bool=false`: Retain real basis-expansion coefficient vectors.
* `tol::T=T(1e-4)`: Extra overlap half-width used when solving and merging neighboring windows.
* `N_expect::Real=1`: Approximate number of states between neighboring scaling centers.
* `dk_threshold::T=T(0.05)`: Maximum distance between neighboring scaling centers.
* `multithreaded_matrices::Bool=true`: Enable multithreaded matrix construction within each window.
* `multithreaded_ks::Bool=false`: Enable multithreading over independent scaling windows.

## Returns
* `data::SpectralData`: Merged spectrum if `vectors=false`.
* `data::StateData`: Merged spectrum with real basis coefficients if `vectors=true`.
"""
function compute_spectrum(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k1::T,k2::T;vectors::Bool=false,tol::T=T(1e-4),N_expect::Real=1,dk_threshold::T=T(0.05),multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    curves=BilliardGeometry.get_boundary_curves(billiard)
    A=area(curves)
    L=sum(crv.length for crv in curves)
    k_vals=T[]
    dk_vals=T[]
    k0=k1
    while k0<k2
        density=A*k0/(2*pi)-L/(4*pi)
        dk=min(abs(T(N_expect)/density),dk_threshold)
        push!(k_vals,k0)
        push!(dk_vals,dk)
        k0+=dk
    end
    return _compute_spectrum(Val(vectors),solver,basis,billiard,k_vals,dk_vals;tol=tol,multithreaded_matrices=multithreaded_matrices,multithreaded_ks=multithreaded_ks)
end

"""
    compute_spectrum(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k1::T,k2::T,dk::T;vectors::Bool=false,tol::T=T(1e-4),multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real}

Compute the real Vergini-Saraceno spectrum over `[k1,k2]` using fixed
scaling-center spacing `dk`.

## Returns
* `data::SpectralData`: Merged spectrum if `vectors=false`.
* `data::StateData`: Merged spectrum with real basis coefficients if `vectors=true`.
"""
function compute_spectrum(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,k1::T,k2::T,dk::T;vectors::Bool=false,tol::T=T(1e-4),multithreaded_matrices::Bool=true,multithreaded_ks::Bool=false) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    k_vals=collect(range(k1,k2;step=dk))
    dk_vals=fill(dk,length(k_vals))
    return _compute_spectrum(Val(vectors),solver,basis,billiard,k_vals,dk_vals;tol=tol,multithreaded_matrices=multithreaded_matrices,multithreaded_ks=multithreaded_ks)
end

"""
    compute_spectrum(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,N1::Int,N2::Int;kwargs...) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard}

Compute the real Vergini-Saraceno spectrum between Weyl counting indices `N1`
and `N2`.

The state indices are converted to wavenumbers using [`k_at_state`](@ref) and
the area and perimeter of the physical fundamental domain. If `billiard`
provides an `angles` property, the constant Weyl corner correction is included.

All keyword arguments are forwarded to the wavenumber-range
[`compute_spectrum`](@ref).

## Returns
* `data::SpectralData`: Spectrum if `vectors=false`.
* `data::StateData`: Spectrum with real basis coefficients if `vectors=true`.
"""
function compute_spectrum(solver::VerginiSaracenoSolver,basis::Ba,billiard::Bi,N1::Int,N2::Int;kwargs...) where {Ba<:AbsBasis,Bi<:BilliardGeometry.AbsBilliard}
    curves=BilliardGeometry.get_boundary_curves(billiard)
    A=area(curves)
    L=sum(crv.length for crv in curves)
    if hasproperty(billiard,:angles)
        k1=k_at_state(N1,A,L,billiard.angles)
        k2=k_at_state(N2,A,L,billiard.angles)
    else
        k1=k_at_state(N1,A,L)
        k2=k_at_state(N2,A,L)
    end
    return compute_spectrum(solver,basis,billiard,k1,k2;kwargs...)
end
