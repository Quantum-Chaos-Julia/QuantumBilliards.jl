"""
    regularize!(u::AbstractVector{T}) where {T<:Number} → Nothing

Regularize a boundary function in place by replacing `NaN` entries with the
average of their two neighboring boundary values.
The boundary data are interpreted periodically, so the neighbors of the first
entry are the last and second entries, while the neighbors of the last entry
are the penultimate and first entries.

## Arguments
* `u::AbstractVector{T}`: Boundary-function values to regularize.

## Returns
* `nothing`: `u` is modified in place.
"""
@inline function regularize!(u::AbstractVector{T}) where {T<:Number}
    idx=findall(isnan,u)
    N=length(u)
    for i in idx
        im=i==1 ? N : i-1
        ip=i==N ? 1 : i+1
        u[i]=(u[ip]+u[im])/2
    end
    return nothing
end

"""
    _rellich(pts::BoundaryPoints{T},u::AbstractVector{N},k::T) where {T<:Real,N<:Number} → T

Compute the Rellich normalization integral for a Dirichlet eigenfunction from
its boundary normal derivative.

For a Dirichlet eigenfunction the Rellich identity gives

    ∫_Ω |ψ|² dx = (1/(2k²)) ∫_∂Ω (x ⋅ n)|∂ₙψ|² ds.

The physical boundary quadrature weights are taken directly from `pts.ds`.

## Arguments
* `pts::BoundaryPoints{T}`: Boundary points containing positions, outward normals and physical quadrature weights.
* `u::AbstractVector{N}`: Boundary normal derivative values `∂ₙψ`.
* `k::T`: Wavenumber.

## Returns
* `norm::T`: Approximation of the interior norm `∫Ω|ψ|²dx`.
"""
function _rellich(pts::BoundaryPoints{T},u::AbstractVector{N},k::T) where {T<:Real,N<:Number}
    acc=zero(T)
    @inbounds @simd for i in eachindex(u)
        n=pts.normal[i]
        xy=pts.xy[i]
        w=(n[1]*xy[1]+n[2]*xy[2])*pts.ds[i]
        acc+=w*abs2(u[i])
    end
    return acc/(2*k^2)
end

"""
    _rellich(pts::Vector{BoundaryPoints{T}},u::AbstractVector{N},k::T) where {T<:Real,N<:Number} → T

Compute the Rellich normalization integral over all connected boundary
components,

    ∫Ω|ψ|²dx=(1/(2k²))∮∂Ω (x⋅n)|∂ₙψ|²ds.

The supplied `pts` may be a filtered postprocessing discretization. Only the
retained physical quadrature nodes contribute.

## Arguments
* `pts::Vector{BoundaryPoints{T}}`: Connected physical boundary components.
* `u::AbstractVector{N}`: Boundary normal derivative in flattened component ordering.
* `k::T`: Helmholtz wavenumber.

## Returns
* `norm::T`: Approximation of the interior `L²` norm.
"""
function _rellich(pts::Vector{BoundaryPoints{T}},u::AbstractVector{N},k::T) where {T<:Real,N<:Number}
    length(u)==boundary_matrix_size(pts)||throw(DimensionMismatch("Boundary function length does not match boundary discretization"))
    acc=zero(T);g=1
    @inbounds for p in pts
        for j in 1:length(p)
            acc+=dot(p.xy[j],p.normal[j])*p.ds[j]*abs2(u[g])
            g+=1
        end
    end
    return acc/(2*k^2)
end

###########################################################################
################ BOUNDARY FUNCTION FOR BASIS TYPE SOLVERS #################
###########################################################################

"""
    boundary_function(state::Eigenstate;b::Real=5.0) → Tuple

Construct the Rellich-normalized boundary normal derivative of a basis-expanded
eigenstate.

The symmetry-adapted basis is evaluated directly on the physical fundamental
boundary and the basis gradients are contracted with the outward normal,

    u(s) = ∂ₙψ(s) = nₓ(s)∂ₓψ(s) + nᵧ(s)∂ᵧψ(s).

It is then fully symmetry-expanded to the complete physical boundary.

This method is intended for basis-type solvers such as
`VerginiSaracenoSolver`, `DecompositionMethodSolver` and
`ParticularSolutionsMethod`.

## Note
The sampler used is [`FourierNodes`](@ref FourierNodes) so that we have uniformly spaced nodes on each boundary curve. This is important for the momentum function, which uses FFTs to compute the Fourier coefficients of the boundary function and also Poincare-Husimi function construction via the [`husimi_function`](@ref husimi_function). The sliding window approach is applicable only in this scenario.

## Arguments
* `state::Eigenstate`: Basis-expanded eigenstate.

## Keyword Arguments
* `b::Real=5.0`: Boundary sampling density in points per de Broglie wavelength.

## Returns
* `u::Vector`: Rellich-normalized boundary normal derivative.
* `pts::BoundaryPoints`: Full-boundary discretization corresponding to `u`.
"""
function boundary_function(state::Eigenstate;b::Real=5.0)
    vec=state.vec
    k=state.k
    k_basis=state.k_basis
    new_basis=state.basis
    billiard=state.billiard
    T=eltype(vec)
    boundary=BilliardGeometry.get_all_curves(billiard)
    crv_lengths=[crv.length for crv in boundary]
    sampler=FourierNodes([2,3,5],crv_lengths) 
    L=CompositeCurve(boundary).length
    N=max(round(Int,k*L*b/(2π)),512)
    pts=boundary_coords(billiard,sampler,N)
    @blas_1 dX,dY=gradient_matrices(new_basis,k,pts.xy) # ∂xϕ, ∂yϕ evaluated on pts.xy 
    M=size(dX,1)
    tX=Vector{T}(undef,M) # tX = (∂xϕ)(x_i), always real since the basis is real.
    tY=Vector{T}(undef,M) # tY = (∂yϕ)(x_i), always real since the basis is real.
    u=Vector{T}(undef,M) # u  = ∂nϕ(x_i), always real since the basis is real and the normal is real.
    # 2 GEMVs into empty then fuse normal-combination: ∂_n ϕ = nx ∂_x ϕ + ny ∂_y ϕ
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(tX,dX,vec) # tX = dX*vec
        mul!(tY,dY,vec) # tY = dY*vec
    end
    @fastmath @inbounds @simd for i in 1:M # fuse u = nx.*tX .+ ny.*tY in one loop
        n=pts.normal[i]
        u[i]=muladd(n[2],tY[i],n[1]*tX[i]) # u = n_x tX + n_y tY via muladd
    end
    regularize!(u)
    #pts=apply_symmetries_to_boundary_points(pts,new_basis.symmetries)
    #u=apply_symmetries_to_boundary_function(u,new_basis.symmetries)
    nrlz=_rellich(pts,u,k) # Rellich boundary norm: ∫ |u|^2 (n·x) ds / (2k^2) no temps
    @blas_1 return u./sqrt(nrlz),pts
end

"""
    boundary_function(state_data::StateData,billiard::Bi,basis::Ba;b::Real=5.0) where {Bi<:AbsBilliard,Ba<:AbsBasis} → Tuple

Construct boundary functions and [`BoundaryPoints`](@ref) discretizations for
all states stored in `state_data`. This struct comes from `VerginiSaracenoSolver`'s `compute_spectrum` method.

For every stored state, the basis is resized to the corresponding wavenumber,
an [`Eigenstate`](@ref) is constructed, and the single-state
[`boundary_function`](@ref) method is used.

## Arguments
* `state_data::StateData`: Stored wavenumbers, tensions and basis coefficients.
* `billiard::Bi`: Billiard geometry.
* `basis::Ba`: Basis used for the stored states.

## Keyword Arguments
* `b::Real=5.0`: Boundary sampling density.

## Returns
* `ks`: Wavenumbers for successfully constructed states.
* `us_all`: Boundary functions corresponding to `ks`.
* `pts_all`: Boundary discretizations corresponding to `ks`.
"""
function boundary_function(state_data::StateData,billiard::Bi,basis::Ba;b::Real=5.0) where {Bi<:AbsBilliard,Ba<:AbsBasis}
    ks=state_data.ks
    tens=state_data.tens
    X=state_data.X
    us_all=Vector{Vector{eltype(ks)}}(undef,length(ks))
    pts_all=Vector{BoundaryPoints{eltype(ks)}}(undef,length(ks))
    valid_indices=fill(true,length(ks))
    progress=Progress(length(ks);desc="Constructing the u(s)...")
    for i in eachindex(ks)
        try
            vec=X[i]
            dim=length(vec)
            dim=rescale_dimension(basis,dim)
            new_basis=resize_basis(basis,billiard,dim,ks[i])
            state=Eigenstate(ks[i],vec,tens[i],new_basis,billiard)
            u,pts=boundary_function(state;b=b)
            us_all[i]=u
            pts_all[i]=pts
        catch e
            println("Error while constructing the u(s) for k = $(ks[i]): $e")
            valid_indices[i]=false
        end
        next!(progress)
    end
    ks=ks[valid_indices]
    us_all=us_all[valid_indices]
    pts_all=pts_all[valid_indices]
    return ks,us_all,pts_all
end

#########################################################################################
################################ MOMENTUM FUNCTION ######################################
#########################################################################################

"""
    momentum_function(u::AbstractVector{U},s::AbstractVector{T},ds::AbstractVector{T};rtol::Real=100eps(T)) where {U<:Number,T<:Real} → Tuple

Compute the one-sided boundary momentum distribution.

## Arguments
* `u::AbstractVector{U}`: Boundary-function values.
* `s::AbstractVector{T}`: Physical arclength coordinates.
* `ds::AbstractVector{T}`: Physical quadrature weights.

## Keyword Arguments
* `rtol::Real=100eps(T)`: Relative tolerance used to detect uniform spacing.

## Returns
* `power`: One-sided boundary momentum weight `|c_m|²`.
* `ks`: Corresponding angular boundary wavenumbers.
"""
function momentum_function(u::AbstractVector{U},s::AbstractVector{T},ds::AbstractVector{T};rtol::Real=100*eps(T)) where {U<:Number,T<:Real}
    N=length(u)
    N==length(s)==length(ds)||throw(DimensionMismatch("u, s and ds must have equal length"))
    # Check whether the physical arclength nodes are uniformly spaced.
    # If so, the Fourier coefficients can be computed with FFTW.
    Δs=s[2]-s[1]
    uniform=all(isapprox(s[i+1]-s[i],Δs;rtol=rtol,atol=eps(T)*max(one(T),abs(Δs))) for i in 2:N-1)
    if uniform && U<:Real
        # Uniform real data: use the one-sided real FFT. Division by N gives
        # the same Fourier-integral normalization as the nonuniform branch.
        fu=FFTW.rfft(u)./N
        ks=FFTW.rfftfreq(N,inv(Δs)).*(2*pi)
        power=abs2.(fu)
        # Negative modes have equal power; double positive modes except Nyquist.
        lastdouble=iseven(N) ? length(power)-1 : length(power)
        lastdouble>=2 && (power[2:lastdouble].*=2)
        return power,ks
    end
    # Nonuniform arclength nodes: evaluate c_m=(1/L)∫u(s)exp(-ik_m*s)ds
    # directly with the physical quadrature weights ds.
    L=sum(ds)
    M=div(N,2) # same number of nonnegative modes as rfft
    ks=T.(2*pi/L.*(0:M))
    power=Vector{T}(undef,M+1)
    @fastmath @inbounds for m in 0:M
        km=ks[m+1]
        acc=zero(Complex{T})
        @simd for j in eachindex(u)
            acc+=u[j]*cis(-km*s[j])*ds[j]
        end
        power[m+1]=abs2(acc/L)
    end
    # For real u, negative modes contain the same power as positive modes.
    # Double positive frequencies, except the Nyquist mode for even N.
    if U<:Real
        lastdouble=iseven(N) ? M : M+1
        lastdouble>=2 && (power[2:lastdouble].*=2)
    end
    return power,ks
end

function momentum_function(u::AbstractVector{U},pts::BoundaryPoints{T};kwargs...) where {U<:Number,T<:Real}
    return momentum_function(u,pts.s,pts.ds;kwargs...)
end

"""
    momentum_function(state::S;b::Real = 5.0) where {S<:AbsState} → (power::Vector, ks::Vector)

Computes the momentum-space representation of an eigenstate's boundary
function directly from `state`, by combining [`boundary_function`](@ref) and
[`momentum_function(u, s)`](@ref momentum_function).

## Arguments
* `state`: The eigenstate for which the boundary momentum function is computed.

## Keyword arguments
* `b::Real = 5.0`: Oversampling factor passed to [`boundary_function`](@ref)
  controlling the boundary point density.

## Returns
* `power`: The normalized boundary momentum weight `|c_m|²`.
* `ks`: The angular wavenumbers corresponding to `power`.
"""
function momentum_function(state::Eigenstate;b::Real=5.0)
    u,pts=boundary_function(state;b=b)
    return momentum_function(u,pts)
end

#########################################################################################################
################### BoundaryIntegralMethod, DLP_kress and DLP_kress_global_corners ######################
#########################################################################################################

#######################################################################
###################### SYMMETRY ORBIT EXPANSION #######################
#######################################################################

"""
    symmetrize_layer_density(solver::BoundaryIntegralMethod,layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Expand symmetry-reduced boundary data onto the complete physical boundary.

`pts` already represents the full physical boundary. If `layer_density` already
has full-boundary length, it is returned unchanged. Otherwise the active
symmetry orbit map is used to expand the reduced data.
"""
function symmetrize_layer_density(solver::BoundaryIntegralMethod,layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull=length(pts)
    length(layer_density)==Nfull&&return pts,layer_density
    isnothing(solver.symmetry)&&throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    orbits=symmetry_index_orbits(T,pts,solver.symmetry)
    Nred=fundamental_size(orbits)
    length(layer_density)==Nred||throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S=promote_type(N,Complex{T})
    full_data=Vector{S}(undef,Nfull)
    @inbounds for q in 1:Nfull
        full_data[q]=orbits.full_to_scale[q]*layer_density[orbits.full_to_fund[q]]
    end
    return pts,full_data
end

"""
    symmetrize_layer_density(solver::BoundaryIntegralMethod,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Batch version of [`symmetrize_layer_density`](@ref) for `BoundaryIntegralMethod`.
"""
function symmetrize_layer_density(solver::BoundaryIntegralMethod,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(layer_density))
    @use_threads multithreading=multithreaded for i in eachindex(layer_density)
        pts_all[i],us_all[i]=symmetrize_layer_density(solver,layer_density[i],pts[i],billiard)
    end
    return pts_all,us_all
end

"""
    symmetrize_layer_density(solver::DLP,layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Expand symmetry-reduced DLP-Kress boundary data onto the complete physical
boundary. Full-length input is returned silently unchanged.
"""
function symmetrize_layer_density(solver::DLP,layer_density::AbstractVector{N},pts::BoundaryPoints{T},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull=length(pts)
    length(layer_density)==Nfull&&return pts,layer_density
    isnothing(solver.symmetry)&&throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    orbits=symmetry_index_orbits(T,pts,solver.symmetry)
    Nred=fundamental_size(orbits)
    length(layer_density)==Nred||throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S=promote_type(N,Complex{T})
    full_data=Vector{S}(undef,Nfull)
    @inbounds for q in 1:Nfull
        full_data[q]=orbits.full_to_scale[q]*layer_density[orbits.full_to_fund[q]]
    end
    return pts,full_data
end

"""
    symmetrize_layer_density(solver::DLP,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Batch version of DLP-Kress symmetry expansion.
"""
function symmetrize_layer_density(solver::DLP,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(layer_density))
    @use_threads multithreading=multithreaded for i in eachindex(layer_density)
        pts_all[i],us_all[i]=symmetrize_layer_density(solver,layer_density[i],pts[i],billiard)
    end
    return pts_all,us_all
end

# Internal workspace overload used when the CFIE workspace already exists.
function symmetrize_layer_density(solver::CFIE,layer_density::AbstractVector{N},pts::Vector{BoundaryPoints{T}},ws::CFIEKressWorkspace{T}) where {N<:Number,T<:Real}
    Nfull=ws.Ntot
    length(layer_density)==Nfull&&return pts,layer_density
    isnothing(solver.symmetry)&&throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    orbits=ws.orbits
    isnothing(orbits)&&throw(ArgumentError("CFIE symmetry expansion requires an active symmetry orbit map"))
    Nred=fundamental_size(orbits)
    length(layer_density)==Nred||throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S=promote_type(N,Complex{T})
    full_data=Vector{S}(undef,Nfull)
    @inbounds for g in 1:Nfull
        full_data[g]=orbits.full_to_scale[g]*layer_density[orbits.full_to_fund[g]]
    end
    return pts,full_data
end

"""
    symmetrize_layer_density(solver::CFIE,layer_density::AbstractVector{N},pts::Vector{BoundaryPoints{T}},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard} → Tuple

Expand a symmetry-reduced CFIE-Kress layer density onto the complete physical
boundary. `pts` contains one `BoundaryPoints` object per connected boundary
component, while `layer_density` is stored in the corresponding flattened
global boundary ordering.

Full-length input is returned unchanged.
"""
function symmetrize_layer_density(solver::CFIE,layer_density::AbstractVector{N},pts::Vector{BoundaryPoints{T}},billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    ws=build_cfie_kress_workspace(solver,pts)
    return symmetrize_layer_density(solver,layer_density,pts,ws)
end

"""
    symmetrize_layer_density(solver::CFIE,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:Vector{BoundaryPoints{T}}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}

Batch version of CFIE-Kress symmetry expansion.
"""
function symmetrize_layer_density(solver::CFIE,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:Vector{BoundaryPoints{T}}},billiard::Bi;multithreaded::Bool=true) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(layer_density))
    @use_threads multithreading=multithreaded for i in eachindex(layer_density)
        pts_all[i],us_all[i]=symmetrize_layer_density(solver,layer_density[i],pts[i],billiard)
    end
    return pts_all,us_all
end

#######################################################################
######################## RELLICH NORMALIZATION ########################
#######################################################################

"""
    boundary_function(solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard} → Tuple

Construct the physical Dirichlet boundary normal derivative from the nullspace
of the weighted-transpose BIM Fredholm matrix, expand it to the full physical
boundary when symmetry reduction is active, and Rellich-normalize it.

## Arguments
* `solver::BoundaryIntegralMethod`: Boundary integral solver configuration.
* `pts::BoundaryPoints{T}`: Full physical boundary discretization used by the BIM solver.
* `k::T`: Eigenwavenumber at which the adjoint Fredholm nullspace is computed.
* `billiard::Bi`: Billiard geometry used when expanding symmetry-reduced boundary data.

## Returns
* `pts::BoundaryPoints{T}`: Full physical boundary discretization corresponding to the returned boundary function.
* `u::Vector`: Rellich-normalized physical boundary normal derivative `∂ₙψ`.
"""
function boundary_function(solver::BoundaryIntegralMethod,pts::BoundaryPoints{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    orbits=_dlp_symmetry_orbits(solver,pts)
    n=_dlp_matrix_dim(pts,orbits)
    A=Matrix{Complex{T}}(undef,n,n)
    D=similar(A)
    @blas_1 adjoint_fredholm_matrix!(A,D,pts,nothing,k;multithreaded=true)
    _,u,_=smallest_nullvec_krylov!(A;nev=1,tol=1e-12,maxiter=2000,krylovdim=40)
    pts,u=symmetrize_layer_density(solver,u,pts,billiard)
    nrlz=_rellich(pts,u,k)
    return pts,u./sqrt(nrlz)
end

"""
    boundary_function(solver::BoundaryIntegralMethod,pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard} → Tuple

Construct Rellich-normalized physical BIM boundary normal derivatives for a
batch of eigenstates from the nullspaces of the corresponding
weighted-transpose Fredholm matrix.

## Arguments
* `solver::BoundaryIntegralMethod`: Boundary integral solver configuration.
* `pts::AbstractVector{<:BoundaryPoints{T}}`: Boundary discretizations corresponding to the states.
* `billiard::Bi`: Billiard geometry used when expanding symmetry-reduced boundary functions.
* `ks::AbstractVector{T}`: Eigenwavenumbers corresponding to the boundary discretizations.

## Keyword Arguments
* `multithreaded::Bool=true`: Whether to process different eigenstates in parallel.

## Returns
* `pts_all::Vector`: Full physical boundary discretizations corresponding to the returned boundary functions.
* `us_all::Vector{Vector}`: Rellich-normalized physical boundary normal derivatives `∂ₙψ` for all states.
"""
function boundary_function(solver::BoundaryIntegralMethod,pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    length(pts)==length(ks)||throw(DimensionMismatch("pts and ks must have equal length"))
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(pts))
    @use_threads multithreading=multithreaded for i in eachindex(pts)
        orbits=_dlp_symmetry_orbits(solver,pts[i])
        n=_dlp_matrix_dim(pts[i],orbits)
        A=Matrix{Complex{T}}(undef,n,n)
        D=similar(A)
        @blas_1 adjoint_fredholm_matrix!(A,D,pts[i],nothing,ks[i];multithreaded=false)
        _,u,_=smallest_nullvec_krylov!(A;nev=1,tol=1e-12,maxiter=2000,krylovdim=40)
        pts_i,u=symmetrize_layer_density(solver,u,pts[i],billiard)
        nrlz=_rellich(pts_i,u,ks[i])
        pts_all[i]=pts_i
        us_all[i]=u./sqrt(nrlz)
    end
    return pts_all,us_all
end

"""
    boundary_function(solver::DLP,pts::BoundaryPoints{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard} → Tuple

Construct the physical Dirichlet boundary normal derivative from the nullspace
of the weighted-transpose DLP-Kress Fredholm matrix, expand it to the full
physical boundary when symmetry reduction is active, and Rellich-normalize it.

## Arguments
* `solver::DLP`: Smooth or globally graded DLP-Kress solver.
* `pts::BoundaryPoints{T}`: Full physical boundary discretization used by the DLP-Kress solver.
* `billiard::Bi`: Billiard geometry used when expanding symmetry-reduced boundary data.
* `k::T`: Eigenwavenumber at which the adjoint Fredholm nullspace is computed.

## Returns
* `pts::BoundaryPoints{T}`: Full physical boundary discretization corresponding to the returned boundary function.
* `u::Vector`: Rellich-normalized physical boundary normal derivative `∂ₙψ`.
"""
function boundary_function(solver::DLP,pts::BoundaryPoints{T},billiard::Bi,k::T) where {T<:Real,Bi<:AbsBilliard}
    ws=build_dlp_kress_workspace(solver,pts)
    n=_workspace_dim(ws)
    A=Matrix{Complex{T}}(undef,n,n)
    D=similar(A)
    @blas_1 adjoint_fredholm_matrix!(A,D,solver,pts,ws,k;multithreaded=true)
    _,u,_=smallest_nullvec_krylov!(A;nev=1,tol=1e-12,maxiter=2000,krylovdim=40)
    pts,u=symmetrize_layer_density(solver,u,pts,billiard)
    nrlz=_rellich(pts,u,k)
    return pts,u./sqrt(nrlz)
end

"""
    boundary_function(solver::DLP,pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard} → Tuple

Construct Rellich-normalized physical DLP-Kress boundary normal derivatives for
a batch of eigenstates from the nullspaces of the corresponding
weighted-transpose Fredholm matrix.

Each state is processed independently. When `multithreaded=true`, threading is
performed over states and individual matrix assemblies are kept single-threaded
to avoid nested threading.

## Arguments
* `solver::DLP`: Smooth or globally graded DLP-Kress solver.
* `layer_density::AbstractVector{<:AbstractVector{N}}`: Primal DLP layer densities associated with the eigenstates.
* `pts::AbstractVector{<:BoundaryPoints{T}}`: Boundary discretizations corresponding to the states.
* `billiard::Bi`: Billiard geometry used when expanding symmetry-reduced boundary functions.
* `ks::AbstractVector{T}`: Eigenwavenumbers corresponding to the boundary discretizations.

## Keyword Arguments
* `multithreaded::Bool=true`: Whether to process different eigenstates in parallel.

## Returns
* `pts_all::Vector`: Full physical boundary discretizations corresponding to the returned boundary functions.
* `us_all::Vector{Vector}`: Rellich-normalized physical boundary normal derivatives `∂ₙψ` for all states.
"""
function boundary_function(solver::DLP,pts::AbstractVector{<:BoundaryPoints{T}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    length(pts)==length(ks)||throw(DimensionMismatch("pts and ks must have equal length"))
    pts_all=Vector{typeof(pts[1])}(undef,length(pts))
    us_all=Vector{Vector}(undef,length(pts))
    @use_threads multithreading=multithreaded for i in eachindex(pts)
        ws=build_dlp_kress_workspace(solver,pts[i])
        n=_workspace_dim(ws)
        A=Matrix{Complex{T}}(undef,n,n)
        D=similar(A)
        @blas_1 adjoint_fredholm_matrix!(A,D,solver,pts[i],ws,ks[i];multithreaded=false)
        _,u,_=smallest_nullvec_krylov!(A;nev=1,tol=1e-12,maxiter=2000,krylovdim=40)
        pts_i,u=symmetrize_layer_density(solver,u,pts[i],billiard)
        nrlz=_rellich(pts_i,u,ks[i])
        pts_all[i]=pts_i
        us_all[i]=u./sqrt(nrlz)
    end
    return pts_all,us_all
end

"""
    _cfie_slp_action(pts::Vector{BoundaryPoints{T}},σ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real} → Vector{Complex{T}}

Apply the Helmholtz single-layer operator to a full CFIE boundary density.
For same-component interactions, periodic Kress logarithmic product integration
is used. Cross-component interactions are smooth and are integrated with the
ordinary physical arc-length quadrature weights.

The operator represented is

    Sσ(x)=∫_∂Ω Φ_k(x,y)σ(y)ds_y.

## Arguments
* `pts::Vector{BoundaryPoints{T}}`: Full connected boundary components.
* `σ::AbstractVector{Complex{T}}`: Density in flattened full-boundary ordering.
* `ws::CFIEKressWorkspace{T}`: Precomputed CFIE-Kress workspace.
* `k::T`: Helmholtz wavenumber.

## Returns
* `Sσ::Vector{Complex{T}}`: Single-layer action in flattened full-boundary ordering.
"""
function _cfie_slp_action(pts::Vector{BoundaryPoints{T}},σ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},
k::T) where {T<:Real}
    length(σ)==ws.Ntot||throw(DimensionMismatch("Density has length $(length(σ)); expected $(ws.Ntot)"))
    out=zeros(Complex{T},ws.Ntot)
    @inbounds for a in eachindex(pts)
        pa=pts[a];Ga=ws.Gs[a];ra=ws.offs[a]:ws.offs[a+1]-1;Na=length(pa)
        for i in 1:Na
            gi=ra[i];acc=zero(Complex{T})
            for j in 1:Na
                gj=ra[j];sj=Ga.speed[j]
                if i==j
                    m1=-inv_two_pi*sj
                    m2=((Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*sj^2))*sj
                else
                    r=Ga.R[i,j];h0=H(0,k*r)
                    m1=-inv_two_pi*real(h0)*sj
                    m2=Complex{T}(0,one(T)/2)*h0*sj-m1*Ga.logterm[i,j]
                end
                acc+=(ws.Rmat[gi,gj]*m1+pa.ws[j]*m2)*σ[gj]
            end
            x=pa.xy[i]
            for b in eachindex(pts)
                b==a&&continue
                pb=pts[b];rb=ws.offs[b]:ws.offs[b+1]-1
                for j in 1:length(pb)
                    acc+=Φ_helmholtz(k,norm(x-pb.xy[j]))*σ[rb[j]]*pb.ds[j]
                end
            end
            out[gi]=acc
        end
    end
    return out
end

"""
    _cfie_slp_dsigma_action(pts::Vector{BoundaryPoints{T}},dσμ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real} → Vector{Complex{T}}

Apply the first Maue single-layer term after analytically cancelling the source
Kress Jacobian.

For

    ∂ₛμ=(1/J)∂σμ,
    ds=J dσ,

the single-layer contribution satisfies

    S(∂ₛμ)
    =∫_∂Ω Φ_k(x,y)∂ₛμ(y)ds_y
    =∫₀²π Φ_k(x,γ(σ))∂σμ(σ)dσ.

Therefore `dσμ` stores `∂σμ` directly and no division by the graded source
speed is performed.

Same-component interactions use the same periodic Kress logarithmic splitting
as [`_cfie_slp_action`](@ref), but with the physical source Jacobian removed.
Cross-component interactions use the uniform computational quadrature weights
`pts[b].ws`.

## Arguments
* `pts::Vector{BoundaryPoints{T}}`: Full connected boundary components.
* `dσμ::AbstractVector{Complex{T}}`: Computational-coordinate derivative `∂σμ`.
* `ws::CFIEKressWorkspace{T}`: Precomputed CFIE-Kress workspace.
* `k::T`: Helmholtz wavenumber.

## Returns
* `Sdμ::Vector{Complex{T}}`: Values of `S(∂ₛμ)` on the complete graded target grid.
"""
function _cfie_slp_dsigma_action(pts::Vector{BoundaryPoints{T}},dσμ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real}
    length(dσμ)==ws.Ntot||throw(DimensionMismatch("Density has length $(length(dσμ)); expected $(ws.Ntot)"))
    out=zeros(Complex{T},ws.Ntot)
    @inbounds for a in eachindex(pts)
        pa=pts[a];Ga=ws.Gs[a];ra=ws.offs[a]:ws.offs[a+1]-1;Na=length(pa)
        for i in 1:Na
            gi=ra[i];acc=zero(Complex{T})
            for j in 1:Na
                gj=ra[j];sj=Ga.speed[j]
                if i==j
                    m1=-inv_two_pi
                    m2=(Complex{T}(0,one(T)/2)-euler_over_pi)-inv_two_pi*log((k^2/4)*sj^2)
                else
                    r=Ga.R[i,j];h0=H(0,k*r)
                    m1=-inv_two_pi*real(h0)
                    m2=Complex{T}(0,one(T)/2)*h0-m1*Ga.logterm[i,j]
                end
                acc+=(ws.Rmat[gi,gj]*m1+pa.ws[j]*m2)*dσμ[gj]
            end
            x=pa.xy[i]
            for b in eachindex(pts)
                b==a&&continue
                pb=pts[b];rb=ws.offs[b]:ws.offs[b+1]-1
                for j in 1:length(pb)
                    acc+=Φ_helmholtz(k,norm(x-pb.xy[j]))*dσμ[rb[j]]*pb.ws[j]
                end
            end
            out[gi]=acc
        end
    end
    return out
end

"""
    _cfie_dlp_matrix!(D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real} → AbstractMatrix{Complex{T}}

Assemble the CFIE operator on the complete physical boundary.
Same-component blocks use periodic Kress logarithmic product integration.
Interactions between different connected components are smooth and use ordinary
Nyström quadrature.

## Arguments
* `D::AbstractMatrix{Complex{T}}`: Preallocated full destination matrix.
* `pts::Vector{BoundaryPoints{T}}`: Full connected boundary components.
* `ws::CFIEKressWorkspace{T}`: Precomputed CFIE-Kress workspace.
* `k::T`: Helmholtz wavenumber.

## Keyword Arguments
* `multithreaded::Bool=true`: Whether sufficiently large assembly loops are threaded.

## Returns
* `D::AbstractMatrix{Complex{T}}`: Assembled full double-layer matrix.
"""
function _cfie_dlp_matrix!(D::AbstractMatrix{Complex{T}},pts::Vector{BoundaryPoints{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    size(D)==(ws.Ntot,ws.Ntot)||throw(DimensionMismatch("D must be $(ws.Ntot)×$(ws.Ntot)"))
    fill!(D,zero(Complex{T}))
    α1=-k*inv_two_pi;α2=Complex{T}(0,k/2)
    for a in eachindex(pts)
        pa=pts[a];Ga=ws.Gs[a];ra=ws.offs[a]:ws.offs[a+1]-1;Na=length(pa)
        @inbounds for i in 1:Na
            D[ra[i],ra[i]]=Complex{T}(pa.ws[i]*Ga.kappa[i],zero(T))
        end
        @use_threads multithreading=(multithreaded&&Na>=32) for j in 2:Na
            gj=ra[j];wj=pa.ws[j]
            @inbounds for i in 1:j-1
                gi=ra[i];wi=pa.ws[i];r=Ga.R[i,j];ir=Ga.invR[i,j]
                lt=Ga.logterm[i,j];h1=H(1,k*r);j1=real(h1)
                l1=α1*Ga.inner[i,j]*j1*ir
                D[gi,gj]=ws.Rmat[gi,gj]*l1+wj*(α2*Ga.inner[i,j]*h1*ir-l1*lt)
                l1=α1*Ga.inner[j,i]*j1*ir
                D[gj,gi]=ws.Rmat[gj,gi]*l1+wi*(α2*Ga.inner[j,i]*h1*ir-l1*lt)
            end
        end
    end
    for a in eachindex(pts),b in eachindex(pts)
        a==b&&continue
        pa=pts[a];pb=pts[b]
        ra=ws.offs[a]:ws.offs[a+1]-1;rb=ws.offs[b]:ws.offs[b+1]-1
        @use_threads multithreading=(multithreaded&&length(pa)>=16) for i in 1:length(pa)
            x=pa.xy[i];gi=ra[i]
            @inbounds for j in 1:length(pb)
                y=pb.xy[j];t=pb.tangent[j]
                dx=x[1]-y[1];dy=x[2]-y[2];r2=muladd(dx,dx,dy*dy)
                r2<=eps(T)^2&&continue
                r=sqrt(r2)
                D[gi,rb[j]]=pb.ws[j]*α2*(t[2]*dx-t[1]*dy)*H(1,k*r)/r
            end
        end
    end
    return D
end

"""
    _cfie_adjoint_dlp_action(pts::Vector{BoundaryPoints{T}},μ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real} → Vector{Complex{T}}

Apply the discrete adjoint double-layer action through the weighted-transpose
identity

    K'ₕ=W⁻¹DₕᵀW,

where `W=diag(ds)` contains physical arc-length quadrature weights.

This is a bilinear transpose consistent with the real Fredholm transpose
convention used for the boundary normal derivative; no complex conjugation is
applied.

## Arguments
* `pts::Vector{BoundaryPoints{T}}`: Full connected boundary components.
* `μ::AbstractVector{Complex{T}}`: Full CFIE layer density.
* `ws::CFIEKressWorkspace{T}`: Precomputed CFIE-Kress workspace.
* `k::T`: Helmholtz wavenumber.

## Keyword Arguments
* `multithreaded::Bool=true`: Whether DLP assembly is threaded.

## Returns
* `Kpμ::Vector{Complex{T}}`: Discrete adjoint DLP action.
"""
function _cfie_adjoint_dlp_action(pts::Vector{BoundaryPoints{T}},μ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T;multithreaded::Bool=true) where {T<:Real}
    length(μ)==ws.Ntot||throw(DimensionMismatch("Density has length $(length(μ)); expected $(ws.Ntot)"))
    D=Matrix{Complex{T}}(undef,ws.Ntot,ws.Ntot)
    _cfie_dlp_matrix!(D,pts,ws,k;multithreaded=multithreaded)
    ds=flatten_boundary_ds(pts)
    return (transpose(D)*(ds.*μ))./ds
end

"""
    _cfie_maue_action(pts::Vector{BoundaryPoints{T}},μ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real} → Vector{Complex{T}}

Apply the hypersingular Helmholtz operator using Maue regularization,

    Nμ=∂ₛS(∂ₛμ)+k² n⋅S(nμ).

This implementation is intended for trivially graded/ungraded periodic
boundary components. Tangential derivatives are evaluated in the
uniform periodic parameter.

For an ungraded parametrization,

    ∂ₛ=(1/|γσ|)∂σ.

The source derivative is represented as `∂σμ`; inside the first single-layer
term the source Jacobian cancels analytically,

    S(∂ₛμ)=∫ Φ_k(x,γ(σ))∂σμ(σ)dσ.

## Arguments
* `pts::Vector{BoundaryPoints{T}}`: Complete periodic boundary components.
* `μ::AbstractVector{Complex{T}}`: Full CFIE layer density.
* `ws::CFIEKressWorkspace{T}`: Precomputed CFIE-Kress workspace.
* `k::T`: Helmholtz wavenumber.

## Returns
* `Nμ::Vector{Complex{T}}`: Maue hypersingular action on the complete boundary grid.
"""
function _cfie_maue_action(pts::Vector{BoundaryPoints{T}},μ::AbstractVector{Complex{T}},ws::CFIEKressWorkspace{T},k::T) where {T<:Real}
    length(μ)==ws.Ntot||throw(DimensionMismatch("Density has length $(length(μ)); expected $(ws.Ntot)"))
    any(_is_nontrivial_grading(p) for p in pts)&&throw(ArgumentError("CFIE Maue reconstruction is not implemented for nontrivially Kress-graded boundaries due to unresolved issues with tangential differentiation on the graded grid."))
    offs=ws.offs
    dσμ=Vector{Complex{T}}(undef,ws.Ntot);σx=similar(dσμ);σy=similar(dσμ)
    @inbounds for a in eachindex(pts)
        p=pts[a];r=offs[a]:offs[a+1]-1;N=length(p)
        F=FFTW.fft(@view μ[r])
        m=iseven(N) ? vcat(0:N÷2-1,0,-N÷2+1:-1) : vcat(0:(N-1)÷2,-(N-1)÷2:-1)
        d=FFTW.ifft((im.*T.(m)).*F)
        for j in 1:N
            g=r[j];n=p.normal[j]
            dσμ[g]=d[j];σx[g]=n[1]*μ[g];σy[g]=n[2]*μ[g]
        end
    end
    Sdμ=_cfie_slp_dsigma_action(pts,dσμ,ws,k)
    Sx=_cfie_slp_action(pts,σx,ws,k);Sy=_cfie_slp_action(pts,σy,ws,k)
    out=Vector{Complex{T}}(undef,ws.Ntot)
    @inbounds for a in eachindex(pts)
        p=pts[a];r=offs[a]:offs[a+1]-1;N=length(p)
        F=FFTW.fft(@view Sdμ[r])
        m=iseven(N) ? vcat(0:N÷2-1,0,-N÷2+1:-1) : vcat(0:(N-1)÷2,-(N-1)÷2:-1)
        d=FFTW.ifft((im.*T.(m)).*F)
        for i in 1:N
            g=r[i];n=p.normal[i]
            out[g]=d[i]/norm(p.tangent[i])+k^2*(n[1]*Sx[g]+n[2]*Sy[g])
        end
    end
    return out
end

"""
    boundary_function(solver::CFIE,layer_density::AbstractVector{N},pts::Vector{BoundaryPoints{T}},billiard::Bi,k::T;multithreaded::Bool=true) where {T<:Real,N<:Number,Bi<:AbsBilliard} → Tuple{Vector{BoundaryPoints{T}},Vector{Complex{T}}}

Recover the physical Dirichlet boundary normal derivative from a CFIE layer
density and Rellich-normalize it.

The reduced CFIE density is first symmetry-expanded to the complete physical
boundary. The boundary normal derivative is reconstructed from

    u=-Nμ-i k(I+K')μ,

where `Nμ` is evaluated through Maue regularization and `K'μ` through the
weighted-transpose DLP action.

CFIE boundary-function reconstruction is currently implemented only for
trivially graded/ungraded periodic boundary discretizations. If any connected
boundary component has nontrivial Kress grading, an `ArgumentError` is thrown
because the tangential differentiation required by the Maue representation is
not yet implemented consistently for such grids.

## Arguments
* `solver::CFIE`: CFIE-Kress solver.
* `layer_density::AbstractVector{N}`: Reduced or full CFIE layer density.
* `pts::Vector{BoundaryPoints{T}}`: Boundary discretization associated with the layer density.
* `billiard::Bi`: Billiard geometry.
* `k::T`: Eigenwavenumber.

## Keyword Arguments
* `multithreaded::Bool=true`: Whether DLP assembly used for the adjoint action is threaded.

## Returns
* `pts::Vector{BoundaryPoints{T}}`: Complete symmetry-expanded physical boundary discretization.
* `u::Vector{Complex{T}}`: Rellich-normalized physical boundary normal derivative.
"""
function boundary_function(solver::CFIE,layer_density::AbstractVector{N},pts::Vector{BoundaryPoints{T}},billiard::Bi,k::T;multithreaded::Bool=true) where {T<:Real,N<:Number,Bi<:AbsBilliard}
    pts,μ=symmetrize_layer_density(solver,layer_density,pts,billiard);μ=Complex{T}.(μ)
    any(_is_nontrivial_grading(p) for p in pts)&&throw(ArgumentError("CFIE boundary-function reconstruction is not implemented for nontrivially Kress-graded boundaries due to unresolved issues with tangential differentiation on the graded grid."))
    ws=build_cfie_kress_workspace(solver,pts)
    Nμ=_cfie_maue_action(pts,μ,ws,k)
    Kpμ=_cfie_adjoint_dlp_action(pts,μ,ws,k;multithreaded=multithreaded)
    u=-Nμ-Complex{T}(0,k).*(μ+Kpμ)
    nrlz=_rellich(pts,u,k)
    nrlz>zero(T)||throw(ArgumentError("Non-positive Rellich norm $nrlz"))
    return pts,u./sqrt(nrlz)
end

"""
    boundary_function(solver::CFIE,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:Vector{BoundaryPoints{T}}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,N<:Number,Bi<:AbsBilliard} → Tuple

Batch CFIE boundary-function reconstruction.

Each state is reconstructed on its complete symmetry-expanded physical boundary
grid. No boundary nodes are removed.

## Arguments
* `solver::CFIE`: CFIE-Kress solver.
* `layer_density::AbstractVector{<:AbstractVector{N}}`: CFIE densities.
* `pts::AbstractVector{<:Vector{BoundaryPoints{T}}}`: Boundary discretizations.
* `billiard::Bi`: Billiard geometry.
* `ks::AbstractVector{T}`: Eigenwavenumbers.

## Keyword Arguments
* `multithreaded::Bool=true`: Whether different eigenstates are processed in parallel.

## Returns
* `pts_all::Vector`: Complete physical boundary grids for all states.
* `us_all::Vector{Vector{Complex{T}}}`: Rellich-normalized boundary functions.
"""
function boundary_function(solver::CFIE,layer_density::AbstractVector{<:AbstractVector{N}},pts::AbstractVector{<:Vector{BoundaryPoints{T}}},billiard::Bi,ks::AbstractVector{T};multithreaded::Bool=true) where {T<:Real,N<:Number,Bi<:AbsBilliard}
    length(layer_density)==length(pts)==length(ks)||throw(DimensionMismatch("layer_density, pts and ks must have equal length"))
    pts_all=Vector{typeof(pts[1])}(undef,length(ks));us_all=Vector{Vector{Complex{T}}}(undef,length(ks))
    @use_threads multithreading=multithreaded for i in eachindex(ks)
        pts_all[i],us_all[i]=boundary_function(solver,layer_density[i],pts[i],billiard,ks[i];multithreaded=false)
    end
    return pts_all,us_all
end