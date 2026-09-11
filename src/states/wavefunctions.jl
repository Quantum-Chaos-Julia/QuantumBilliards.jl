function rectify_grid(grid)
    T=eltype(grid)
    if grid[1]<=zero(T)<=grid[end]
        idx=argmin(abs.(grid))
        new_grid=grid.-grid[idx]
        return new_grid[new_grid.>zero(T)]
    end
    return grid
end

# helper to determine based on the fundamental_domain kwarg if the points are inside the billiard or not
@inline function inside_mask(billiard::Bi,pts;fundamental_domain::Bool=true) where {Bi<:BilliardGeometry.AbsBilliard}
    fundamental_domain&&return BilliardGeometry.is_inside(billiard,pts)
    comps=_boundary_components(billiard.full_boundary)
    isempty(comps)&&return fill(false,length(pts))
    outer=comps[1]
    mask=[all(BilliardGeometry.is_inside(crv,pt) for crv in outer) for pt in pts]
    @inbounds for comp in comps[2:end]
        for i in eachindex(pts)
            mask[i]&&all(BilliardGeometry.is_inside(crv,pts[i]) for crv in comp)&&(mask[i]=false)
        end
    end
    return mask
end

###########################################################################
###################### SLP CHEBYSHEV H0 TUNER ############################
###########################################################################

"""
    chebyshev_params_slp(solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},zj::AbstractVector{Complex{T}};npanels_h_init::Int=4000,M_h_init::Int=5,tol::Real=1e-8,sampling_points::Int=5000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,verbose::Bool=false) where {T<:Real} → Tuple

Determine suitable `H₀⁽¹⁾` Chebyshev parameters for ordinary-BIM SLP
wavefunction reconstruction.

The radial bounds are obtained from `pts`, which is taken from the state with
the largest wavenumber in the batch. The common interpolation interval also
satisfies the small-argument Hankel cutoff for every supplied wavenumber.

## Arguments
* `solver::BoundaryIntegralMethod`: Ordinary boundary-integral solver.
* `pts::BoundaryPoints{T}`: Boundary discretization used to determine radial bounds.
* `zj::AbstractVector{Complex{T}}`: Wavenumbers for which `H₀⁽¹⁾` plans are required.

## Keyword Arguments
* `npanels_h_init::Int=4000`: Initial number of radial panels.
* `M_h_init::Int=5`: Initial Chebyshev degree per panel.
* `tol::Real=1e-8`: Maximum absolute interpolation error.
* `sampling_points::Int=5000`: Number of validation radii.
* `max_iter::Int=20`: Maximum number of refinement iterations.
* `grow_panels::Real=1.5`: Multiplicative panel-count growth factor.
* `grow_M::Int=2`: Additive Chebyshev-degree increment.
* `verbose::Bool=false`: Print tuning diagnostics.

## Returns
* `n::Int`: Final number of radial panels.
* `M::Int`: Final Chebyshev degree.
* `plans::Vector{ChebHankelPlanH}`: `H₀⁽¹⁾` plans, one per wavenumber.
* `errs::Vector{Float64}`: Maximum validation error for each wavenumber.
"""
function chebyshev_params_slp(solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},zj::AbstractVector{Complex{T}};npanels_h_init::Int=4000,M_h_init::Int=5,tol::Real=1e-8,sampling_points::Int=5000,max_iter::Int=20,grow_panels::Real=1.5,grow_M::Int=2,verbose::Bool=false) where {T<:Real}
    rmin_raw,rmax=estimate_rmin_rmax(pts,solver.symmetry)
    rmin=max(Float64(rmin_raw),maximum(hankel_z_chebyshev_cutoff./abs.(zj)))
    rmax=Float64(rmax)
    rmin<rmax||throw(ArgumentError("Empty SLP Hankel interpolation interval: rmin=$rmin, rmax=$rmax"))
    rs=range(rmin,rmax;length=sampling_points)
    n=npanels_h_init
    M=M_h_init
    plans=Vector{ChebHankelPlanH}(undef,length(zj))
    errs=fill(Inf,length(zj))
    for it in 1:max_iter
        Threads.@threads for j in eachindex(zj)
            p=plan_h(0,1,ComplexF64(zj[j]),rmin,rmax;npanels=n,M=M)
            plans[j]=p
            e=0.0
            @inbounds for r in rs
                ip,t=panel_t(p,r)
                e=max(e,abs(eval_h(p,ip,t,r)-SpecialFunctions.besselh(0,1,ComplexF64(zj[j])*r)))
            end
            errs[j]=e
        end
        verbose&&@info "Worst SLP H0 | n M" maximum(errs) n M
        all(<(tol),errs)&&return n,M,plans,errs
        it%5==0 ? (M+=grow_M) : (n=ceil(Int,grow_panels*n))
    end
    @warn "SLP H0 Chebyshev tuning did not reach tol=$tol after $max_iter iterations."
    return n,M,plans,errs
end
# just helpers, never touch again
@inline function _slp_chebyshev_plans(solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},zj::AbstractVector{Complex{T}};tol::Real=1e-8,verbose::Bool=false) where {T<:Real}
    return chebyshev_params_slp(solver,pts,zj;tol=tol,verbose=verbose)
end
@inline function _slp_chebyshev_plans(solver::Union{DLP_kress,DLP_kress_global_corners},pts::BoundaryPoints{T},zj::AbstractVector{Complex{T}};tol::Real=1e-8,verbose::Bool=false) where {T<:Real}
    n,M,_,_,plans0,_,_,_,errs,_,_,_=chebyshev_params(solver,pts,zj;tol=tol,verbose=verbose)
    return n,M,plans0,errs
end
@inline function _cfie_wavefunction_chebyshev_plans(solver::CFIE,pts::Vector{BoundaryPoints{T}},zj::AbstractVector{Complex{T}};tol::Real=1e-8,verbose::Bool=false) where {T<:Real}
    n,M,_,_,plans0,plans1,_,_,err0,err1,_,_=chebyshev_params(solver,pts,zj;tol=tol,verbose=verbose)
    plans=[CFIEWavefunctionChebPlan(plans0[i],plans1[i]) for i in eachindex(zj)]
    return n,M,plans,err0,err1
end

###########################################################################
######################## DLP WAVEFUNCTIONS ################################
###########################################################################

"""
    ϕ_slp(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector;float32_bessel::Bool=true,use_chebyshev::Bool=false,cheb::Union{SLPWavefunctionChebPlan,Nothing}=nothing) where {T<:Real}

Evaluate the SLP reconstruction of a Dirichlet eigenfunction at `(x,y)`,

    ψ(x,y) = (1/4)∫∂Ω Y₀(k|x-q|)u(q)ds_q,

where `u=∂ₙψ` is the boundary normal derivative. The overall sign is
irrelevant for an eigenfunction.

## Arguments
* `x::T`: Evaluation x-coordinate.
* `y::T`: Evaluation y-coordinate.
* `k::T`: Wavenumber.
* `bd::BoundaryPoints{T}`: Boundary discretization.
* `u::AbstractVector`: Boundary normal derivative.

## Keyword Arguments
* `float32_bessel::Bool=true`: Evaluate `Y₀` using Float32 arithmetic.
* `use_chebyshev::Bool=false`: Use the Chebyshev kernel approximation.
* `cheb::Union{SLPWavefunctionChebPlan,Nothing}=nothing`: Chebyshev plan.

## Returns
* `ψ`: Reconstructed wavefunction value.
"""
@inline function ϕ_slp(x::T,y::T,k::T,bd::BoundaryPoints{T},u::AbstractVector;float32_bessel::Bool=true,use_chebyshev::Bool=false,cheb::Union{SLPWavefunctionChebPlan,Nothing}=nothing) where {T<:Real}
    xy=bd.xy
    ds=bd.ds
    S=eltype(u)
    acc=zero(S)
    @inbounds @fastmath for j in eachindex(u)
        p=xy[j]
        dx=x-p[1]
        dy=y-p[2]
        r2=muladd(dx,dx,dy*dy)
        r2==zero(T)&&continue # only guard exact coincidence with a source node
        r=sqrt(r2)
        # Y0(k*r) is either evaluated from the H0 Chebyshev plan or directly.
        y0=if use_chebyshev
            _eval_y0_slp_cheb(cheb,k,r)
        elseif float32_bessel
            T(Bessels.bessely0(Float32(k*r)))
        else
            Bessels.bessely0(k*r)
        end
        # Physical boundary quadrature:
        # ψ(x) = (1/4) Σ_j Y0(k|x-q_j|) u_j ds_j.
        acc+=(y0*ds[j])*u[j]
    end
    return acc*T(0.25)
end

"""
    wavefunctions(solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners},ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_bdPoints::Vector{<:BoundaryPoints{T}},billiard::Bi;b::Union{Real,Symbol}=:auto,inside_only::Bool=true,fundamental::Bool=true,MIN_CHUNK::Int=4096,use_float_32::Bool=true,use_chebyshev::Bool=true,tol_cheb::Real=1e-8,cheb_verbose::Bool=true) where {Bi<:BilliardGeometry.AbsBilliard,T<:Real} → Tuple

Reconstruct a batch of DLP eigenfunctions on a common Cartesian grid.

The Chebyshev radial interval is determined from the boundary discretization
belonging to the largest wavenumber in `ks`, and the resulting common interval
is used to construct one `H₀⁽¹⁾` plan for every wavenumber in the batch.

## Arguments
* `solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners}`: DLP solver.
* `ks::Vector{T}`: Wavenumbers.
* `vec_us::Vector{<:AbstractVector}`: Boundary normal derivatives.
* `vec_bdPoints::Vector{<:BoundaryPoints{T}}`: Boundary discretizations, one per state.
* `billiard::Bi`: Billiard geometry.

## Keyword Arguments
* `b::Union{Real,Symbol}=:auto`: Cartesian grid density scaling.
* `inside_only::Bool=true`: Evaluate only inside the billiard.
* `MIN_CHUNK::Int=4096`: Minimum number of spatial points per thread chunk.
* `use_float_32::Bool=true`: Use Float32 direct `Y₀` evaluation when Chebyshev interpolation is disabled.
* `use_chebyshev::Bool=true`: Use Chebyshev-interpolated `H₀⁽¹⁾`.
* `tol_cheb::Real=1e-8`: Chebyshev interpolation tolerance.
* `cheb_verbose::Bool=true`: Print Chebyshev tuning diagnostics.
* `fundamental_domain::Bool=true`: Use the fundamental domain for inside/outside masking.

## Returns
* `Psi2ds::Vector{Matrix}`: Reconstructed wavefunction matrices.
* `x_grid::Vector{T}`: Common x grid.
* `y_grid::Vector{T}`: Common y grid.
"""
function wavefunctions(solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners},ks::Vector{T},vec_us::Vector{<:AbstractVector},vec_bdPoints::Vector{<:BoundaryPoints{T}},billiard::Bi;b::Union{Real,Symbol}=:auto,inside_only::Bool=true,MIN_CHUNK::Int=4096,use_float_32::Bool=true,use_chebyshev::Bool=true,tol_cheb::Real=1e-8,cheb_verbose::Bool=true,fundamental_domain::Bool=true) where {Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    k_max,idx_max=findmax(ks)
    comps=_boundary_components(billiard.full_boundary)[1]
    L=sum(c.length for c in comps)
    b=b==:auto ? (typeof(solver.pts_scaling_factor)<:Real ? solver.pts_scaling_factor : solver.pts_scaling_factor[1]) : b
    xlim,ylim=boundary_limits(comps;grd=max(1000,round(Int,k_max*L*b/(2*pi))))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,k_max*dx*b/(2*pi)),512)
    ny=max(round(Int,k_max*dy*b/(2*pi)),512)
    x_grid=collect(T,range(xlim...,nx))
    y_grid=collect(T,range(ylim...,ny))
    pts=[SVector(x,y) for y in y_grid for x in x_grid]
    pts_mask=inside_only ? inside_mask(billiard,pts;fundamental_domain=fundamental_domain) : fill(true,length(pts))
    pts_masked_indices=findall(pts_mask)
    if use_chebyshev
        zj=Complex{T}.(ks)
        cheb_npanels,cheb_M,plans0,errs=_slp_chebyshev_plans(solver,vec_bdPoints[idx_max],zj;tol=tol_cheb,verbose=cheb_verbose)
        cheb_plans=[SLPWavefunctionChebPlan(p) for p in plans0]
        @info "Using SLP wavefunction Cheb" cheb_npanels cheb_M max_err=maximum(errs)
    else
        cheb_plans=fill(nothing,length(ks))
    end
    S=eltype(vec_us[1])<:Real ? T : Complex{T}
    Psi2ds=Vector{Matrix{S}}(undef,length(ks))
    NT=Threads.nthreads()
    nmask=length(pts_masked_indices)
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    q,r=divrem(nmask,NT_eff)
    progress=Progress(length(ks),desc="Constructing wavefunction matrices...")
    @inbounds for i in eachindex(ks)
        Psi_flat=zeros(S,nx*ny)
        k=ks[i]
        bd=vec_bdPoints[i]
        u=vec_us[i]
        Threads.@threads :static for t in 1:NT_eff
            lo=(t-1)*q+min(t-1,r)+1
            hi=lo+q-1+(t<=r ? 1 : 0)
            for jj in lo:hi
                idx=pts_masked_indices[jj]
                ix=((idx-1)%nx)+1
                iy=((idx-1)÷nx)+1
                Psi_flat[idx]=ϕ_slp(x_grid[ix],y_grid[iy],k,bd,u;float32_bessel=use_float_32,use_chebyshev=use_chebyshev,cheb=use_chebyshev ? cheb_plans[i] : nothing)
            end
        end
        Psi2ds[i]=reshape(Psi_flat,nx,ny)
        next!(progress)
    end
    return Psi2ds,x_grid,y_grid
end

###########################################################################
######################## CFIE WAVEFUNCTIONS ###############################
###########################################################################

struct CFIEWavefunctionCache{T<:Real}
    x::Vector{T}      # source-point x coordinates
    y::Vector{T}      # source-point y coordinates
    tx::Vector{T}     # x component of γ'(t)
    ty::Vector{T}     # y component of γ'(t)
    sj::Vector{T}     # |γ'(t)|, converting parameter measure to arclength
    w::Vector{T}      # parameter-space quadrature weights dt
    hmin::T           # minimum ds size
end

function CFIEWavefunctionCache(pts::Vector{BoundaryPoints{T}}) where {T<:Real}
    N=sum(length,pts)
    x=Vector{T}(undef,N);y=Vector{T}(undef,N);tx=Vector{T}(undef,N);ty=Vector{T}(undef,N);sj=Vector{T}(undef,N);w=Vector{T}(undef,N)
    hmin=typemax(T)
    g=1
    @inbounds for p in pts
        for j in eachindex(p.xy)
            q=p.xy[j];t=p.tangent[j]
            x[g]=q[1];y[g]=q[2];tx[g]=t[1];ty[g]=t[2];sj[g]=hypot(t[1],t[2]);w[g]=p.ws[j]
            hmin=min(hmin,p.ds[j])
            g+=1
        end
    end
    return CFIEWavefunctionCache(x,y,tx,ty,sj,w,hmin)
end

"""
    ϕ_cfie(xp::T,yp::T,k::T,cache::CFIEWavefunctionCache{T},μ::AbstractVector{Complex{T}};float32_bessel::Bool=false,use_chebyshev::Bool=false,cheb::Union{CFIEWavefunctionChebPlan,Nothing}=nothing) where {T<:Real} → Complex{T}

Evaluate the interior CFIE wavefunction at `(xp,yp)`.

The implementation uses the doubled CFIE operators,

    ψ ∝ -(D + i*k*S)μ,

where `D` and `S` denote the doubled operators used internally. The common
factor is irrelevant after wavefunction normalization.

## Arguments
* `xp::T`: Evaluation x-coordinate.
* `yp::T`: Evaluation y-coordinate.
* `k::T`: Wavenumber.
* `cache::CFIEWavefunctionCache{T}`: CFIE geometry cache.
* `μ::AbstractVector{Complex{T}}`: CFIE layer density.

## Keyword Arguments
* `float32_bessel::Bool=false`: Evaluate Hankel functions in Float32.
* `use_chebyshev::Bool=false`: Use Chebyshev-interpolated Hankel functions.
* `cheb::Union{CFIEWavefunctionChebPlan,Nothing}=nothing`: Paired `H₀⁽¹⁾`/`H₁⁽¹⁾` Chebyshev plan.

## Returns
* `ψ::Complex{T}`: Reconstructed wavefunction value.
"""
@inline function ϕ_cfie(xp::T,yp::T,k::T,cache::CFIEWavefunctionCache{T},μ::AbstractVector{Complex{T}};float32_bessel::Bool=false,use_chebyshev::Bool=false,cheb::Union{CFIEWavefunctionChebPlan,Nothing}=nothing) where {T<:Real}
    x=cache.x
    y=cache.y
    tx=cache.tx
    ty=cache.ty
    sj=cache.sj
    w=cache.w
    N=length(x)
    tol2=(cache.hmin)^2
    @inbounds for j in 1:N
        dx=xp-x[j];dy=yp-y[j]
        muladd(dx,dx,dy*dy)<=tol2&&return zero(Complex{T})
    end
    ψr=zero(T)
    ψi=zero(T)
    # Constants:
    # dterm = (i*k/2) * inn * H1 / r
    # sterm = (i/2) * H0 * sj
    # contribution = -(w*μ) * (dterm + i*k*sterm)
    #
    # Since i*k*sterm = i*k*(i/2)H0*sj = -(k/2)H0*sj,
    # the doubled CFIE kernel is
    # K = (i*k/2)*inn*H1/r - (k/2)*H0*sj.
    khalf=k*T(0.5)
    @inbounds @fastmath for j in 1:N
        dx=xp-x[j]
        dy=yp-y[j]
        r2=muladd(dx,dx,dy*dy)
        r=sqrt(r2)
        invr=inv(r)
        inn=muladd(ty[j],dx,-tx[j]*dy) # ty*dx - tx*dy
        if use_chebyshev
            h0,h1=_eval_h0h1_cfie_cheb(cheb,r)
        elseif float32_bessel
            zf=Float32(k*r)
            h0=Complex{T}(Bessels.hankelh1(0,zf))
            h1=Complex{T}(Bessels.hankelh1(1,zf))
        else
            z=k*r
            h0=Bessels.hankelh1(0,z)
            h1=Bessels.hankelh1(1,z)
        end
        # Let A=(k/2)*inn/r and B=(k/2)*sj.
        # Then K=i*A*h1-B*h0.
        A=khalf*inn*invr
        B=khalf*sj[j]
        # Real and imaginary parts of K:
        # i*A*h1 = -A*Im(h1) + i*A*Re(h1).
        Kr=muladd(-A,imag(h1),-B*real(h0))
        Ki=muladd(A,real(h1),-B*imag(h0))
        # Multiply by the parameter-space quadrature weight and complex density.
        μj=μ[j]
        wr=w[j]*real(μj)
        wi=w[j]*imag(μj)
        # contribution = -(wr+i*wi)*(Kr+i*Ki)
        ψr-=wr*Kr-wi*Ki
        ψi-=wr*Ki+wi*Kr
    end
    return Complex{T}(ψr,ψi)
end

"""
    wavefunctions(solver::CFIE,ks::Vector{T},vec_μ::Vector{<:AbstractVector{Complex{T}}},vec_pts::AbstractVector{<:Vector{BoundaryPoints{T}}},billiard::Bi;b::Union{Real,Symbol}=:auto,inside_only::Bool=true,MIN_CHUNK::Int=4096,float32_bessel::Bool=true,use_chebyshev::Bool=true,tol_cheb::Real=1e-8,cheb_verbose::Bool=true) where {Bi<:BilliardGeometry.AbsBilliard,T<:Real} → Tuple

Reconstruct a batch of CFIE eigenfunctions on a common Cartesian grid.

The Chebyshev radial bounds are obtained once from `vec_pts[idx_max]`, where
`idx_max` is the state with the largest wavenumber. Existing
[`chebyshev_params`](@ref) then constructs `H₀⁽¹⁾` and `H₁⁽¹⁾` plans for every
wavenumber on that common interval.

## Arguments
* `solver::CFIE`: CFIE Kress solver.
* `ks::Vector{T}`: Wavenumbers.
* `vec_μ::Vector{<:AbstractVector{Complex{T}}}`: CFIE layer densities.
* `vec_pts::Vector{<:BoundaryPoints{T}}`: Boundary discretizations, one per state.
* `billiard::Bi`: Billiard geometry.

## Keyword Arguments
* `b::Union{Real,Symbol}=:auto`: Cartesian grid density scaling.
* `inside_only::Bool=true`: Evaluate only inside the billiard.
* `MIN_CHUNK::Int=4096`: Minimum number of spatial points per thread chunk.
* `float32_bessel::Bool=true`: Use Float32 direct Hankel evaluation when Chebyshev interpolation is disabled.
* `use_chebyshev::Bool=true`: Use Chebyshev-interpolated Hankel functions.
* `tol_cheb::Real=1e-8`: Chebyshev interpolation tolerance.
* `cheb_verbose::Bool=true`: Print Chebyshev tuning diagnostics.
* `fundamental_domain::Bool=true`: Use the fundamental domain for inside/outside masking.

## Returns
* `Psi2ds::Vector{Matrix{Complex{T}}}`: Normalized complex wavefunctions.
* `x_grid::Vector{T}`: Common x grid.
* `y_grid::Vector{T}`: Common y grid.
"""
function wavefunctions(solver::CFIE,ks::Vector{T},vec_μ,vec_pts::AbstractVector{<:Vector{BoundaryPoints{T}}},billiard::Bi;b::Union{Real,Symbol}=:auto,inside_only::Bool=true,MIN_CHUNK::Int=4096,float32_bessel::Bool=true,use_chebyshev::Bool=true,tol_cheb::Real=1e-8,cheb_verbose::Bool=true,fundamental_domain::Bool=true) where {Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    kmax,idx_max=findmax(ks)
    outer_bdry=_boundary_components(billiard.full_boundary)[1]
    L=sum(c.length for c in outer_bdry)
    b=b==:auto ? (typeof(solver.pts_scaling_factor)<:Real ? solver.pts_scaling_factor : solver.pts_scaling_factor[1]) : b
    xlim,ylim=boundary_limits(outer_bdry;grd=max(1000,round(Int,kmax*L*b/(2*pi))))
    dx=xlim[2]-xlim[1]
    dy=ylim[2]-ylim[1]
    nx=max(round(Int,kmax*dx*b/(2*pi)),512)
    ny=max(round(Int,kmax*dy*b/(2*pi)),512)
    x_grid=collect(T,range(xlim[1],xlim[2],length=nx))
    y_grid=collect(T,range(ylim[1],ylim[2],length=ny))
    pts=[SVector(x,y) for y in y_grid for x in x_grid]
    pts_mask=inside_only ? inside_mask(billiard,pts;fundamental_domain=fundamental_domain) : fill(true,length(pts))
    pts_masked_indices=findall(pts_mask)
    nmask=length(pts_masked_indices)
    NT=Threads.nthreads()
    NT_eff=max(1,min(NT,cld(nmask,MIN_CHUNK)))
    nstates=length(ks)
    Psi2ds=Vector{Matrix{Complex{T}}}(undef,nstates)
    caches=Vector{CFIEWavefunctionCache{T}}(undef,nstates)
    μ_full=Vector{Vector{Complex{T}}}(undef,nstates)
    @inbounds for i in 1:nstates
        ptsi,μi=symmetrize_layer_density(solver,vec_μ[i],vec_pts[i],billiard)
        caches[i]=CFIEWavefunctionCache(ptsi)
        μ_full[i]=Complex{T}.(μi)
    end
    if use_chebyshev
        zj=Complex{T}.(ks)
        nh,Mh,cheb_plans,errH0,errH1=_cfie_wavefunction_chebyshev_plans(solver,vec_pts[idx_max],zj;tol=tol_cheb,verbose=cheb_verbose)
        @info "Using CFIE wavefunction Cheb" nh Mh max_err0=maximum(errH0) max_err1=maximum(errH1)
    else
        cheb_plans=fill(nothing,nstates)
    end
    Psi_flat=zeros(Complex{T},nx*ny)
    progress=Progress(nstates,desc="Constructing CFIE wavefunction matrices...")
    q,r=divrem(nmask,NT_eff)
    @inbounds for i in eachindex(ks)
        k=ks[i]
        cache=caches[i]
        μ=μ_full[i]
        fill!(Psi_flat,zero(Complex{T}))
        Threads.@threads :static for t in 1:NT_eff
            lo=(t-1)*q+min(t-1,r)+1
            hi=lo+q-1+(t<=r ? 1 : 0)
            for jj in lo:hi
                idx=pts_masked_indices[jj]
                p=pts[idx]
                Psi_flat[idx]=ϕ_cfie(p[1],p[2],k,cache,μ;float32_bessel=float32_bessel,use_chebyshev=use_chebyshev,cheb=use_chebyshev ? cheb_plans[i] : nothing)
            end
        end
        nrm=sqrt(sum(abs2,@view Psi_flat[pts_masked_indices]))
        nrm>zero(T)&&(Psi_flat./=nrm)
        Psi2ds[i]=copy(reshape(Psi_flat,nx,ny))
        next!(progress)
    end
    return Psi2ds,x_grid,y_grid
end

###########################################################################
################### BASIS WAVEFUNCTION CONSTRUCTION #######################
###########################################################################

"""
    compute_psi(state::S,x_grid::AbstractVector,y_grid::AbstractVector;inside_only::Bool=true,memory_limit::Real=10.0e9,multithreaded::Bool=true) where {S<:AbsState} → Vector

Evaluate a basis-expanded state on a Cartesian grid.

The full basis matrix is used when its estimated memory footprint is below
`memory_limit`; otherwise the wavefunction is accumulated one basis function
at a time.

## Arguments
* `state::S`: Basis-expanded state.
* `x_grid::AbstractVector`: Cartesian x grid.
* `y_grid::AbstractVector`: Cartesian y grid.

## Keyword Arguments
* `inside_only::Bool=true`: Evaluate only inside the billiard.
* `memory_limit::Real=10.0e9`: Maximum basis-matrix memory before switching to direct accumulation.
* `multithreaded::Bool=true`: Enable threaded basis-matrix construction.

## Returns
* `Psi::Vector`: Flattened wavefunction with x varying fastest.
"""
function compute_psi(state::S,x_grid,y_grid;inside_only=true,memory_limit=10.0e9,multithreaded=true) where {S<:AbsState}
    let vec=state.vec,k=state.k,basis=state.basis,billiard=state.billiard,eps=state.eps
        sz=length(x_grid)*length(y_grid)
        pts=collect(SVector(x,y) for y in y_grid for x in x_grid)
        if inside_only
            #pts_mask=is_inside(billiard,pts)
            pts_mask=inside_mask(billiard,pts,fundamental_domain=true)
            pts=pts[pts_mask]
        end
        n_pts=length(pts)
        type=eltype(vec)
        memory=sizeof(type)*basis.dim*n_pts
        Psi=zeros(type,sz)
        if memory<memory_limit
            B=basis_matrix(basis,k,pts;multithreaded)
            Psi_pts=B*vec
            if inside_only
                Psi[pts_mask].=Psi_pts
            else
                Psi.=Psi_pts
            end
        else
            println("Warning: memory limit of $(Base.format_bytes(memory_limit)) exceeded $(Base.format_bytes(memory)).")
            if inside_only
                for i in eachindex(vec)
                    if abs(vec[i])>eps
                        Psi[pts_mask].+=vec[i].*basis_fun(basis,i,k,pts)
                    end
                end
            else
                for i in eachindex(vec)
                    if abs(vec[i])>eps
                        Psi.+=vec[i].*basis_fun(basis,i,k,pts)
                    end
                end
            end
        end
        if inside_only
            Psi[.!pts_mask].=convert(type,NaN)
        end
        return Psi
    end
end

# XAxisReflection reflects y -> -y; BilliardGeometry.YAxisReflection reflects x -> -x.
@inline _has_x_reduction(symmetry)=symmetry isa Union{BilliardGeometry.YAxisReflection,BilliardGeometry.XYAxisReflection}
@inline _has_y_reduction(symmetry)=symmetry isa Union{BilliardGeometry.XAxisReflection,BilliardGeometry.XYAxisReflection}

"""
    wavefunction(state::S;b::Real=5.0,inside_only::Bool=true,fundamental_domain::Bool=true,memory_limit::Real=10.0e9,multithreaded::Bool=true) where {S<:AbsState} → Tuple

Construct the wavefunction of a basis-expanded state on a Cartesian grid.

For reflection-adapted bases, `fundamental_domain=true` restricts the
corresponding Cartesian direction to the positive half-grid. With
`fundamental_domain=false`, the basis is evaluated directly on the full grid.

## Arguments
* `state::S`: Basis-expanded state.

## Keyword Arguments
* `b::Real=5.0`: Cartesian grid sampling density.
* `inside_only::Bool=true`: Evaluate only inside the billiard.
* `fundamental_domain::Bool=true`: Restrict the grid to the symmetry-reduced fundamental domain.
* `memory_limit::Real=10.0e9`: Memory limit passed to [`compute_psi`](@ref).
* `multithreaded::Bool=true`: Enable threaded basis-matrix construction.

## Returns
* `Psi2d::Matrix`: Wavefunction matrix.
* `x_grid::Vector`: Cartesian x grid.
* `y_grid::Vector`: Cartesian y grid.
"""
function wavefunction(state::S;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9,multithreaded=true) where {S<:AbsState}
    let k=state.k,billiard=state.billiard,symmetry=state.basis.symmetries
        T=typeof(real(k))
        V=eltype(state.vec)
        L=CompositeCurve(get_boundary_curves(billiard)).length
        xlim,ylim=boundary_limits(get_boundary_curves(billiard);grd=max(1000,round(Int,k*L*b/(2*pi))))
        dx=xlim[2]-xlim[1]
        dy=ylim[2]-ylim[1]
        nx=max(round(Int,k*dx*b/(2*pi)),512)
        ny=max(round(Int,k*dy*b/(2*pi)),512)
        x_grid=collect(T,range(xlim...,nx))
        y_grid=collect(T,range(ylim...,ny))
        if fundamental_domain
            if _has_x_reduction(symmetry);x_grid=rectify_grid(x_grid);nx=length(x_grid);end
            if _has_y_reduction(symmetry);y_grid=rectify_grid(y_grid);ny=length(y_grid);end
        end
        Psi::Vector{V}=compute_psi(state,x_grid,y_grid;inside_only=inside_only,memory_limit=memory_limit,multithreaded=multithreaded)
        Psi2d::Matrix{V}=reshape(Psi,(nx,ny))
        return Psi2d,x_grid,y_grid
    end
end

"""
    wavefunction(state::BasisState;xlim::Tuple=(-2.0,2.0),ylim::Tuple=(-2.0,2.0),b::Real=5.0) → Tuple

Evaluate a single basis function on a Cartesian grid.

## Arguments
* `state::BasisState`: Basis state.

## Keyword Arguments
* `xlim::Tuple=(-2.0,2.0)`: Cartesian x limits.
* `ylim::Tuple=(-2.0,2.0)`: Cartesian y limits.
* `b::Real=5.0`: Grid sampling density.

## Returns
* `Psi2d::Matrix`: Basis-function values.
* `x_grid::Vector`: Cartesian x grid.
* `y_grid::Vector`: Cartesian y grid.
"""
function wavefunction(state::BasisState;xlim=(-2.0,2.0),ylim=(-2.0,2.0),b=5.0)
    let k=state.k,basis=state.basis
        T=typeof(real(k))
        V=eltype(state.vec)
        dx=xlim[2]-xlim[1]
        dy=ylim[2]-ylim[1]
        nx=max(round(Int,k*dx*b/(2*pi)),512)
        ny=max(round(Int,k*dy*b/(2*pi)),512)
        x_grid=collect(T,range(xlim...,nx))
        y_grid=collect(T,range(ylim...,ny))
        pts_grid=[SVector(x,y) for y in y_grid for x in x_grid]
        Psi::Vector{V}=basis_fun(basis,state.idx,k,pts_grid)
        Psi2d::Matrix{V}=reshape(Psi,(nx,ny))
        return Psi2d,x_grid,y_grid
    end
end

"""
    wavefunctions(state_data::StateData,billiard::Bi,basis::Ba;b::Real=5.0,inside_only::Bool=true,fundamental_domain::Bool=true,memory_limit::Real=10.0e9,multithreaded::Bool=true) where {Bi<:BilliardGeometry.AbsBilliard,Ba<:AbsBasis} → Tuple

Construct wavefunctions for all states stored in `state_data`. This comes from the `compute_spectrum` function from the `VerginiSaracenoSolver()`.

## Arguments
* `state_data::StateData`: Stored eigenstates.
* `billiard::Bi`: Billiard geometry.
* `basis::Ba`: Basis used to construct the stored states.

## Keyword Arguments
* `b::Real=5.0`: Cartesian grid sampling density.
* `inside_only::Bool=true`: Evaluate only inside the billiard.
* `fundamental_domain::Bool=true`: Restrict to the symmetry-reduced fundamental domain.
* `memory_limit::Real=10.0e9`: Memory limit passed to [`compute_psi`](@ref).
* `multithreaded::Bool=true`: Enable threaded basis-matrix construction.

## Returns
* `ks`: Wavenumbers.
* `Psi2ds`: Wavefunction matrices.
* `x_grids`: Cartesian x grids.
* `y_grids`: Cartesian y grids.
"""
function wavefunctions(state_data::StateData,billiard::Bi,basis::Ba;b=5.0,inside_only=true,fundamental_domain=true,memory_limit=10.0e9,multithreaded=true) where {Bi<:BilliardGeometry.AbsBilliard,Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    V=eltype(X[1])
    T=eltype(ks)
    Psi2ds=Vector{Matrix{V}}(undef,length(ks))
    x_grids=Vector{Vector{T}}(undef,length(ks))
    y_grids=Vector{Vector{T}}(undef,length(ks))
    progress=Progress(length(ks);desc="Constructing wavefunctions...")
    for i in eachindex(ks)
        vec=X[i]
        dim=rescale_dimension(basis,length(vec))
        new_basis=resize_basis(basis,billiard,dim,ks[i])
        state=Eigenstate(ks[i],ks[i],vec,tens[i],new_basis,billiard)
        Psi2d,x_grid,y_grid=wavefunction(state;b=b,inside_only=inside_only,fundamental_domain=fundamental_domain,memory_limit=memory_limit,multithreaded=multithreaded)
        Psi2ds[i]=Psi2d
        x_grids[i]=x_grid
        y_grids[i]=y_grid
        next!(progress)
    end
    return ks,Psi2ds,x_grids,y_grids
end