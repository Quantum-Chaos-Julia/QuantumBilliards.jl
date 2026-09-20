############## HELPER FUNCTIONS #############
function pad_limits(xlim, ylim; padding::Real = 0.01)
    return (xlim[1] - padding, xlim[2] + padding), (ylim[1] - padding, ylim[2] + padding)
end

function rectify_grid(grid::AbstractVector)
    T = eltype(grid)
    if grid[1] <= zero(T) <= grid[end]
        idx = argmin(abs.(grid))
        new_grid = grid .- grid[idx]
        return new_grid[new_grid .> zero(T)]
    end
    return grid
end

function boundary_limits(curves; grd::Int = 1000, padding::Real = 0.01)
    x_bnd = Vector{Any}(); y_bnd = Vector{Any}()
    for crv in curves
        L = crv.length
        N_bnd = max(512, round(Int, grd / L))
        t = range(0.0, 1.0, N_bnd)[1:end-1]
        pts = curve(crv, t)
        append!(x_bnd, getindex.(pts, 1)); append!(y_bnd, getindex.(pts, 2))
    end
    x_bnd[end] = x_bnd[1]; y_bnd[end] = y_bnd[1]
    xlim = extrema(x_bnd); ylim = extrema(y_bnd)
    return pad_limits(xlim, ylim; padding = padding)
end
###########################################

################################################################################
# BIM WAVEFUNCTION RECONSTRUCTION
#
# Dirichlet eigenfunctions are reconstructed from either the BIM layer
# density or the boundary normal derivative u = ∂ₙψ.
#
# DLP:
#     ψ(x) = ∫∂Ω ∂ₙq G_k(x,q) μ(q) ds_q
#
# CFIE:
#     ψ(x) ∝ -(D_k + i k S_k)μ
#
# SLP from the physical boundary function:
#     ψ(x) = 1/4 ∫∂Ω Y₀(k|x-q|) u(q) ds_q
#
# Chebyshev acceleration uses ChebHankelPlanH directly:
#     u/SLP: H₀⁽¹⁾, taking Im H₀⁽¹⁾ = Y₀,
#     DLP:   H₁⁽¹⁾,
#     CFIE:  H₀⁽¹⁾ and H₁⁽¹⁾.
################################################################################

"""
    _wavefunction_grid(kmax::T, solver::SweepBIMSolver, billiard::Bi, b::Real; inside_only::Bool = true)

Construct the Cartesian grid used for BIM wavefunction reconstruction.
For boundary length `L` and sampling factor `b`, the Cartesian grid contains
approximately `b` points per wavelength,

    Δx, Δy ≈ 2π/(b kmax),

with at least 512 points along each coordinate.

## Arguments
- `kmax::T`: Largest wave number represented on the grid.
- `solver::SweepBIMSolver`: BIM solver determining the physical boundary.
- `billiard::Bi`: Billiard geometry.
- `b::Real`: Approximate number of grid points per wavelength.

## Kwargs
- `inside_only::Bool = true`: Evaluate only points inside the billiard.

## Returns
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
- `pts::Vector{SVector{2,T}}`: Flattened Cartesian grid points.
- `indices::Vector{Int}`: Indices of points selected for evaluation.
- `nx::Int`: Number of x-grid points.
- `ny::Int`: Number of y-grid points.
"""
@inline function _wavefunction_grid(kmax::T, solver::SweepBIMSolver, billiard::Bi, b::Real; inside_only::Bool = true) where {T<:Real,Bi<:BilliardGeometry.AbsBilliard}
    comp = solver.symmetry === nothing ? get_boundary_curves(billiard) : full_boundary(billiard)
    L = sum(crv.length for crv in comp)
    xlim, ylim = boundary_limits(comp; grd = max(1000, round(Int, kmax * L * b / (2pi))))
    dx = xlim[2] - xlim[1]; dy = ylim[2] - ylim[1]
    nx = max(round(Int, kmax * dx * b / (2pi)), 512); ny = max(round(Int, kmax * dy * b / (2pi)), 512)
    x_grid = collect(T, range(xlim[1], xlim[2]; length = nx)); y_grid = collect(T, range(ylim[1], ylim[2]; length = ny))
    pts = [SVector{2,T}(x, y) for y in y_grid for x in x_grid]
    mask = inside_only ? is_inside(billiard, pts) : trues(length(pts))
    return x_grid, y_grid, pts, findall(mask), nx, ny
end

################################################################################
# BESSEL / HANKEL EVALUATION
################################################################################

@inline function _y0(r, k, cheb::ChebHankelPlanH, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r)
    pidx, t = panel_t(cheb, rf)
    return T(imag(eval_h(cheb, pidx, t, rf)))
end

@inline _y0(r, k, cheb, ::Val{:f32}, ::Type{T}) where {T<:Real} = T(Bessels.bessely0(Float32(k * r)))
@inline _y0(r, k, cheb, ::Val{:direct}, ::Type{T}) where {T<:Real} = T(Bessels.bessely0(k * r))

@inline function _h1(r, k, cheb::ChebHankelPlanH, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r)
    pidx, t = panel_t(cheb, rf)
    return Complex{T}(eval_h(cheb, pidx, t, rf))
end

@inline _h1(r, k, cheb, ::Val{:f32}, ::Type{T}) where {T<:Real} = Complex{T}(Bessels.hankelh1(1, Float32(k * r)))
@inline _h1(r, k, cheb, ::Val{:direct}, ::Type{T}) where {T<:Real} = Complex{T}(Bessels.hankelh1(1, k * r))

@inline function _h01(r, k, cheb::Tuple{ChebHankelPlanH,ChebHankelPlanH}, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r)
    pidx, t = panel_t(cheb[1], rf)
    return Complex{T}(eval_h(cheb[1], pidx, t, rf)), Complex{T}(eval_h(cheb[2], pidx, t, rf))
end

@inline function _h01(r, k, cheb, ::Val{:f32}, ::Type{T}) where {T<:Real}
    z = Float32(k * r)
    return Complex{T}(Bessels.hankelh1(0, z)), Complex{T}(Bessels.hankelh1(1, z))
end

@inline function _h01(r, k, cheb, ::Val{:direct}, ::Type{T}) where {T<:Real}
    z = k * r
    return Complex{T}(Bessels.hankelh1(0, z)), Complex{T}(Bessels.hankelh1(1, z))
end

################################################################################
# POTENTIAL EVALUATION
################################################################################

"""
    ϕ_slp(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector, cheb, mode::Val)

Reconstruct a Dirichlet eigenfunction from its physical boundary normal
derivative `u = ∂ₙψ`:

    ψ(x) = 1/4 ∫∂Ω Y₀(k|x-q|) u(q) ds_q.

## Arguments
- `x::T`, `y::T`: Evaluation coordinates.
- `k::T`: Wave number.
- `bd::BoundaryPoints{T}`: Full physical-boundary discretization.
- `u::AbstractVector`: Physical boundary normal derivative `∂ₙψ`.
- `cheb`: Chebyshev `H₀⁽¹⁾` plan, or `nothing`.
- `mode::Val`: Bessel evaluation mode.

## Returns
- `ψ::S`: Reconstructed wavefunction value, where
  `S = promote_type(T, eltype(u))`.
"""
@inline function ϕ_slp(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector, cheb, mode::Val) where {T<:Real}
    xy = bd.xy; ds = bd.ds
    S = promote_type(T, eltype(u)); acc = zero(S)
    @inbounds for j in eachindex(u)
        p = xy[j]
        dx = x - p[1]; dy = y - p[2]
        r2 = muladd(dx, dx, dy * dy)
        r2 == zero(T) && continue
        r = sqrt(r2)
        acc += _y0(r, k, cheb, mode, T) * ds[j] * u[j]
    end
    return acc * T(0.25)
end

"""
    ϕ_dlp(x::T, y::T, k::T, bd::BoundaryPoints{T}, μ::AbstractVector, cheb, mode::Val)

Evaluate the double-layer representation

    ψ(x) = ∫∂Ω ∂ₙq G_k(x,q) μ(q) ds_q,

with outgoing Helmholtz Green function

    G_k(x,q) = (i/4) H₀⁽¹⁾(k|x-q|).

## Arguments
- `x::T`, `y::T`: Evaluation coordinates.
- `k::T`: Wave number.
- `bd::BoundaryPoints{T}`: Full physical-boundary discretization.
- `μ::AbstractVector`: DLP layer density.
- `cheb`: Chebyshev `H₁⁽¹⁾` plan, or `nothing`.
- `mode::Val`: Hankel evaluation mode.

## Returns
- `ψ::S`: Reconstructed wavefunction value, where
  `S = promote_type(eltype(μ), Complex{T})`.
"""
@inline function ϕ_dlp(x::T, y::T, k::T, bd::BoundaryPoints{T}, μ::AbstractVector, cheb, mode::Val) where {T<:Real}
    xy = bd.xy; normal = bd.normal; ds = bd.ds
    S = promote_type(eltype(μ), Complex{T}); acc = zero(S); kquarter = k * T(0.25)
    @inbounds for j in eachindex(μ)
        p = xy[j]; n = normal[j]
        dx = x - p[1]; dy = y - p[2]
        r2 = muladd(dx, dx, dy * dy)
        r2 == zero(T) && continue
        r = sqrt(r2)
        inn = muladd(dx, n[1], dy * n[2])
        acc += (im * kquarter) * _h1(r, k, cheb, mode, T) * (inn / r) * μ[j] * ds[j]
    end
    return acc
end

"""
    ϕ_cfie(x::T, y::T, k::T, bd::BoundaryPoints{T}, μ::AbstractVector, cheb, mode::Val)

Evaluate the combined-field representation of a Dirichlet eigenfunction.

With the doubled-operator convention used by the CFIE discretization,

    ψ(x) ∝ -(D_k + i k S_k) μ,

so both `H₀⁽¹⁾` and `H₁⁽¹⁾` are required. The omitted overall constant is
irrelevant for an eigenfunction.

## Arguments
- `x::T`, `y::T`: Evaluation coordinates.
- `k::T`: Wave number.
- `bd::BoundaryPoints{T}`: Full physical-boundary discretization.
- `μ::AbstractVector`: CFIE layer density.
- `cheb`: Tuple containing Chebyshev `H₀⁽¹⁾` and `H₁⁽¹⁾` plans, or `nothing`.
- `mode::Val`: Hankel evaluation mode.

## Returns
- `ψ::S`: Reconstructed wavefunction value, where
  `S = promote_type(eltype(μ), Complex{T})`.
"""
@inline function ϕ_cfie(x::T, y::T, k::T, bd::BoundaryPoints{T}, μ::AbstractVector, cheb, mode::Val) where {T<:Real}
    xy = bd.xy; tangent = bd.tangent; ws = bd.ws
    S = promote_type(eltype(μ), Complex{T}); acc = zero(S); khalf = k * T(0.5)
    @inbounds for j in eachindex(μ)
        p = xy[j]; t = tangent[j]
        dx = x - p[1]; dy = y - p[2]
        r2 = muladd(dx, dx, dy * dy)
        r2 == zero(T) && continue
        r = sqrt(r2)
        inn = muladd(t[2], dx, -t[1] * dy)
        sj = hypot(t[1], t[2])
        h0, h1 = _h01(r, k, cheb, mode, T)
        A = khalf * inn / r; B = khalf * sj
        acc -= ws[j] * μ[j] * (im * A * h1 - B * h0)
    end
    return acc
end

################################################################################
# BIM WAVEFUNCTION API
################################################################################

"""
    wavefunction(solver::SweepBIMSolver, ks::Vector{T}, vec_u, vec_pts::Vector{<:BoundaryPoints{T}}, billiard::Bi; layer_density::Bool = true, b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, float32_bessel::Bool = true, use_chebyshev::Bool = true, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T))

Reconstruct one or more BIM eigenfunctions on a common Cartesian grid.

For `layer_density = true`, the native BIM representation is used,

    DLP:   ψ = D_k μ,
    CFIE:  ψ ∝ -(D_k + i k S_k) μ.

For `layer_density = false`, `vec_u` is interpreted as the physical boundary
normal derivative `u = ∂ₙψ` and

    ψ(x) = 1/4 ∫∂Ω Y₀(k|x-q|) u(q) ds_q.

The Cartesian grid is constructed once from the largest wave number in `ks`.
The boundary data must already represent the complete physical boundary.

## Arguments
- `solver::SweepBIMSolver`: BIM solver associated with the states.
- `ks::Vector{T}`: Wave numbers.
- `vec_u`: Native layer densities or physical boundary normal derivatives.
- `vec_pts::Vector{<:BoundaryPoints{T}}`: Corresponding full-boundary discretizations.
- `billiard::Bi`: Billiard geometry.

## Keyword Arguments
- `layer_density::Bool = true`: Use the native BIM density rather than `u = ∂ₙψ`.
- `b::Union{Real,Symbol} = :auto`: Grid points per wavelength; `:auto` uses `_bim_grid_scale(solver)`.
- `inside_only::Bool = true`: Evaluate only inside the billiard.
- `MIN_CHUNK::Int = 4096`: Minimum spatial workload per active thread.
- `float32_bessel::Bool = true`: Use Float32 Bessel/Hankel evaluation when Chebyshev acceleration is disabled.
- `use_chebyshev::Bool = true`: Use Chebyshev-accelerated radial functions.
- `show_progress::Bool = true`: Display progress for individual eigenstates.
- `cheb_config::ChebyshevConfig = ChebyshevConfig(T)`: Chebyshev tuning configuration.

## Returns
- `Psi2ds::Vector{Matrix{V}}`: Reconstructed wavefunctions.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(solver::SweepBIMSolver, ks::Vector{T}, vec_u, vec_pts::Vector{<:BoundaryPoints{T}}, billiard::Bi; layer_density::Bool=true, b::Union{Real,Symbol}=:auto, inside_only::Bool=true, MIN_CHUNK::Int=4096, float32_bessel::Bool=true, use_chebyshev::Bool=true, show_progress::Bool=true, cheb_config::ChebyshevConfig=ChebyshevConfig(T)) where {Bi<:BilliardGeometry.AbsBilliard,T<:Real}
    kmax = maximum(ks); bval = b === :auto ? T(_bim_grid_scale(solver)) : T(b)
    x_grid,y_grid,pts,indices,nx,ny = _wavefunction_grid(kmax,solver,billiard,bval;inside_only)
    representation = layer_density ? Val(:density) : Val(:boundary)
    mode = use_chebyshev ? Val(:cheb) : float32_bessel ? Val(:f32) : Val(:direct)
    Psi2ds = _wavefunctions(solver,ks,vec_u,vec_pts,x_grid,y_grid,pts,indices,nx,ny,representation,mode;MIN_CHUNK,show_progress,cheb_config)
    return Psi2ds,x_grid,y_grid
end

# Native DLP layer density μ: reconstruct with the double-layer potential.
function _wavefunctions(solver::DLP, ks::Vector{T}, vec_u, vec_pts::Vector{<:BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:density}, mode::Val{M}; MIN_CHUNK::Int=4096, show_progress::Bool=true, cheb_config::ChebyshevConfig=ChebyshevConfig(T)) where {T<:Real,M}
    return _wavefunctions_impl(solver,ks,vec_u,vec_pts,x_grid,y_grid,pts,indices,nx,ny,Val(:dlp),mode,ϕ_dlp;MIN_CHUNK,show_progress,cheb_config)
end

# Native CFIE layer density μ: reconstruct with the combined-field potential.
function _wavefunctions(solver::CFIE, ks::Vector{T}, vec_u, vec_pts::Vector{<:BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:density}, mode::Val{M}; MIN_CHUNK::Int=4096, show_progress::Bool=true, cheb_config::ChebyshevConfig=ChebyshevConfig(T)) where {T<:Real,M}
    return _wavefunctions_impl(solver,ks,vec_u,vec_pts,x_grid,y_grid,pts,indices,nx,ny,Val(:cfie),mode,ϕ_cfie;MIN_CHUNK,show_progress,cheb_config)
end

# Physical boundary derivative u=∂ₙψ: SLP reconstruction is independent of the original BIM formulation.
function _wavefunctions(solver::SweepBIMSolver, ks::Vector{T}, vec_u, vec_pts::Vector{<:BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:boundary}, mode::Val{M}; MIN_CHUNK::Int=4096, show_progress::Bool=true, cheb_config::ChebyshevConfig=ChebyshevConfig(T)) where {T<:Real,M}
    return _wavefunctions_impl(solver,ks,vec_u,vec_pts,x_grid,y_grid,pts,indices,nx,ny,Val(:slp),mode,ϕ_slp;MIN_CHUNK,show_progress,cheb_config)
end

"""
    _wavefunctions_impl(solver::SweepBIMSolver, ks::Vector{T}, vec_u, vec_pts::Vector{<:BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, representation::Val{R}, mode::Val{M}, ϕ::F; MIN_CHUNK::Int = 4096, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T))

Evaluate BIM wavefunctions on a preconstructed Cartesian grid.

The Cartesian grid and its active point indices are supplied by the caller so
that grid construction and geometric masking can be shared across multiple
reconstruction batches. The layer-potential evaluator is selected by dispatch
before entering the threaded evaluation kernel.

## Arguments
- `solver::SweepBIMSolver`: BIM solver.
- `ks::Vector{T}`: Wave numbers.
- `vec_u`: Boundary vectors.
- `vec_pts::Vector{<:BoundaryPoints{T}}`: Full-boundary discretizations.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
- `pts`: Flattened Cartesian grid points.
- `indices`: Grid-point indices at which the wavefunctions are evaluated.
- `nx::Int`: Number of x-grid points.
- `ny::Int`: Number of y-grid points.
- `representation::Val{R}`: `Val(:dlp)`, `Val(:cfie)`, or `Val(:slp)`.
- `mode::Val{M}`: `Val(:cheb)`, `Val(:f32)`, or `Val(:direct)`.
- `ϕ::F`: Specialized layer-potential evaluator.

## Keyword Arguments
- `MIN_CHUNK::Int = 4096`: Minimum spatial workload per active thread.
- `show_progress::Bool = true`: Display progress for individual eigenstates.
- `cheb_config::ChebyshevConfig = ChebyshevConfig(T)`: Chebyshev tuning configuration.

## Returns
- `Psi2ds::Vector{Matrix{V}}`: Reconstructed wavefunctions.
"""
function _wavefunctions_impl(solver::SweepBIMSolver, ks::Vector{T}, vec_u, vec_pts::Vector{<:BoundaryPoints{T}}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, representation::Val{R}, mode::Val{M}, ϕ::F; MIN_CHUNK::Int=4096, show_progress::Bool=true, cheb_config::ChebyshevConfig=ChebyshevConfig(T)) where {T<:Real,R,M,F}
    _,idx_max = findmax(ks)
    if M === :cheb
        ks_cheb = ComplexF64.(ks); bd = vec_pts[idx_max]
        xmin = Float64(x_grid[1]); xmax = Float64(x_grid[end]); ymin = Float64(y_grid[1]); ymax = Float64(y_grid[end]); rmax = 0.0
        @inbounds for p in bd.xy
            rmax = max(rmax,hypot(xmin-p[1],ymin-p[2]),hypot(xmin-p[1],ymax-p[2]),hypot(xmax-p[1],ymin-p[2]),hypot(xmax-p[1],ymax-p[2]))
        end
        rmax *= 1.05
        rmin = hankel_z_chebyshev_cutoff/maximum(abs,ks_cheb)
        if R === :cfie
            plans0,plans1,_,_,_ = tune_cfie_cheb_plans(rmin,rmax,ks_cheb,cheb_config)
            cheb_plans = [(plans0[i],plans1[i]) for i in eachindex(ks)]
        elseif R === :dlp
            plans1,_,_ = tune_dlp_cheb_plans(rmin,rmax,ks_cheb,cheb_config)
            cheb_plans = plans1
        else
            plans0,_,_,_,_ = tune_cfie_cheb_plans(rmin,rmax,ks_cheb,cheb_config)
            cheb_plans = plans0
        end
    else
        cheb_plans = fill(nothing,length(ks))
    end
    V = R === :slp ? promote_type(T,eltype(vec_u[1])) : promote_type(eltype(vec_u[1]),Complex{T})
    Psi2ds = Vector{Matrix{V}}(undef,length(ks))
    nmask = length(indices); NT_eff = max(1,min(Threads.nthreads(),cld(nmask,MIN_CHUNK))); q,r = divrem(nmask,NT_eff)
    @maybe_showprogress show_progress for i in eachindex(ks)
        Psi = zeros(V,nx*ny); k = ks[i]; u = vec_u[i]; bd = vec_pts[i]; cheb = cheb_plans[i]
        Threads.@threads :static for t in 1:NT_eff
            lo = (t-1)*q+min(t-1,r)+1; hi = lo+q-1+(t<=r ? 1 : 0)
            @inbounds for jj in lo:hi
                idx = indices[jj]; p = pts[idx]
                Psi[idx] = ϕ(p[1],p[2],k,bd,u,cheb,mode)
            end
        end
        Psi2ds[i] = reshape(Psi,nx,ny)
    end
    return Psi2ds
end

"""
    wavefunction(states::AbstractVector{<:BIMEigenstate}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, float32_bessel::Bool = true, use_chebyshev::Bool = true, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(eltype(first(states).pts.ds), tol = 1e-10))

Reconstruct a collection of BIM eigenstates on one common Cartesian grid.

## Arguments
- `states::AbstractVector{<:BIMEigenstate}`: BIM eigenstates to reconstruct.

## Keyword Arguments
- `b::Union{Real,Symbol} = :auto`: Grid points per wavelength; `:auto` uses the solver grid scale.
- `inside_only::Bool = true`: Evaluate only inside the billiard.
- `MIN_CHUNK::Int = 4096`: Minimum spatial workload per active thread.
- `float32_bessel::Bool = true`: Use Float32 Bessel/Hankel evaluation when Chebyshev acceleration is disabled.
- `use_chebyshev::Bool = true`: Use Chebyshev-accelerated radial functions.
- `show_progress::Bool = true`: Display progress for individual eigenstates.
- `cheb_config::ChebyshevConfig`: Chebyshev tuning configuration.

## Returns
- `Psi2ds::Vector{Matrix{V}}`: Reconstructed wavefunctions in the same order as `states`.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(states::AbstractVector{<:BIMEigenstate}; b::Union{Real,Symbol}=:auto, inside_only::Bool=true, MIN_CHUNK::Int=4096, float32_bessel::Bool=true, use_chebyshev::Bool=true, show_progress::Bool=true, cheb_config::ChebyshevConfig=ChebyshevConfig(eltype(first(states).pts.ds),tol=1e-10))
    s0 = first(states); T = eltype(s0.pts.ds)
    ks = T[real(s.k) for s in states]; kmax = maximum(ks)
    bval = b === :auto ? T(_bim_grid_scale(s0.solver)) : T(b)
    x_grid,y_grid,pts,indices,nx,ny = _wavefunction_grid(kmax,s0.solver,s0.billiard,bval;inside_only)
    layer_density = all(s.vec !== nothing for s in states)
    vec_u = layer_density ? [s.vec for s in states] : [s.u for s in states]
    any(isnothing,vec_u) && error("BIMEigenstates must provide the same boundary representation")
    vec_pts = [s.pts for s in states]
    representation = layer_density ? Val(:density) : Val(:boundary)
    mode = use_chebyshev ? Val(:cheb) : float32_bessel ? Val(:f32) : Val(:direct)
    Psi2ds = _wavefunctions(s0.solver,ks,vec_u,vec_pts,x_grid,y_grid,pts,indices,nx,ny,representation,mode;MIN_CHUNK,show_progress,cheb_config)
    return Psi2ds,x_grid,y_grid
end

"""
    wavefunction(state::BIMEigenstate; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, float32_bessel::Bool = true, use_chebyshev::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(eltype(state.pts.ds), tol = 1e-10))

Reconstruct the interior wavefunction of a BIM eigenstate.

If `state.vec` is available, the native layer-potential representation is used,

    DLP:   ψ = D_k μ,
    CFIE:  ψ ∝ -(D_k + i k S_k) μ.

Otherwise the physical boundary normal derivative `state.u = ∂ₙψ` is used,

    ψ(x) = 1/4 ∫∂Ω Y₀(k|x-q|) u(q) ds_q.

Both representations are assumed to already cover the complete physical
boundary.

## Arguments
- `state::BIMEigenstate`: BIM eigenstate to reconstruct.

## Keyword Arguments
- `b::Union{Real,Symbol} = :auto`: Grid points per wavelength.
- `inside_only::Bool = true`: Evaluate only inside the billiard.
- `MIN_CHUNK::Int = 4096`: Minimum spatial workload per active thread.
- `float32_bessel::Bool = true`: Use Float32 radial functions when Chebyshev acceleration is disabled.
- `use_chebyshev::Bool = true`: Use Chebyshev-accelerated radial functions.
- `cheb_config::ChebyshevConfig`: Chebyshev tuning configuration.

## Returns
- `Psi::Matrix{V}`: Reconstructed wavefunction.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(state::BIMEigenstate; b::Union{Real,Symbol}=:auto, inside_only::Bool=true, MIN_CHUNK::Int=4096, float32_bessel::Bool=true, use_chebyshev::Bool=true, cheb_config::ChebyshevConfig=ChebyshevConfig(eltype(state.pts.ds),tol=1e-10))
    T = eltype(state.pts.ds); ks = T[real(state.k)]
    kwargs = (;b,inside_only,MIN_CHUNK,float32_bessel,use_chebyshev,show_progress=false,cheb_config)
    if state.vec !== nothing
        Psi,x_grid,y_grid = wavefunction(state.solver,ks,[state.vec],[state.pts],state.billiard;layer_density=true,kwargs...)
    elseif state.u !== nothing
        Psi,x_grid,y_grid = wavefunction(state.solver,ks,[state.u],[state.pts],state.billiard;layer_density=false,kwargs...)
    else
        error("BIMEigenstate has neither vec nor u")
    end
    return Psi[1],x_grid,y_grid
end

################################################################################
# BASIS WAVEFUNCTION RECONSTRUCTION  ψ(x) = Σⱼ cⱼ φⱼ(k,x),
################################################################################

"""
    compute_psi(state::S, x_grid::AbstractVector, y_grid::AbstractVector; inside_only::Bool = true, memory_limit::Real = 10.0e9, multithreaded::Bool = true) where {S<:AbsState}

Evaluate the wavefunction of `state` on a Cartesian grid.

The basis expansion ψ(x) = Σⱼ cⱼ φⱼ(k,x)
is evaluated through basis-matrix multiplication ψ = B(k)c.

To bound memory usage, the evaluation points are divided into chunks such that
each basis matrix occupies at most approximately `memory_limit`. Each chunk is
constructed with `basis_matrix`, so the standard filtering of numerically small
basis-matrix elements is applied before multiplication by the state vector.

If `inside_only = true`, only points inside the billiard are evaluated and
exterior values are set to `NaN`.

## Arguments
- `state::S`: Eigenstate whose basis expansion is evaluated.
- `x_grid::AbstractVector`: Cartesian x coordinates.
- `y_grid::AbstractVector`: Cartesian y coordinates.

## Kwargs
- `inside_only::Bool = true`: Evaluate only inside the billiard and set
  exterior values to `NaN`.
- `memory_limit::Real = 10.0e9`: Approximate maximum memory in bytes allocated
  to each basis-matrix chunk.
- `multithreaded::Bool = true`: Use multithreaded basis-matrix construction.

## Returns
- `Psi::Vector{T}`: Flattened wavefunction values with x varying fastest.
"""
function compute_psi(state::S, x_grid, y_grid; inside_only = true, memory_limit = 10.0e9, multithreaded = true) where {S<:AbsState}
    vec = state.vec; k = state.k_basis; basis = state.basis; billiard = state.billiard
    T = eltype(vec)
    pts = [SVector(x, y) for y in y_grid for x in x_grid]
    mask = inside_only ? is_inside(billiard, pts) : trues(length(pts))
    idx = findall(mask); pts_eval = pts[idx]
    Psi = inside_only ? fill(convert(T, NaN), length(pts)) : zeros(T, length(pts))
    chunk_size = max(1, floor(Int, 0.8 * memory_limit / (sizeof(T) * basis.dim)))
    for lo in 1:chunk_size:length(pts_eval)
        hi = min(lo + chunk_size - 1, length(pts_eval))
        ids = lo:hi
        B = basis_matrix(basis, k, pts_eval[ids]; multithreaded)
        Psi[idx[ids]] .= B * vec
    end
    return Psi
end

"""
    wavefunction(state::S; b::Real = 5.0, inside_only::Bool = true, fundamental_domain::Bool = true, memory_limit::Real = 10.0e9, multithreaded::Bool = true) where {S<:AbsState}

Compute a basis-expanded eigenstate on a Cartesian grid.
If the basis carries reflection symmetries, the corresponding coordinate
grids are reduced with [`rectify_grid`](@ref). For
`fundamental_domain = false`, the result is subsequently unfolded onto the
complete billiard with `apply_symmetries_to_wavefunction`.

## Arguments
- `state::S`: Eigenstate to evaluate.

## Kwargs
- `b::Real = 5.0`: Approximate grid points per wavelength.
- `inside_only::Bool = true`: Evaluate only inside the billiard.
- `fundamental_domain::Bool = true`: Return the symmetry-reduced fundamental
  domain rather than the unfolded wavefunction.
- `memory_limit::Real = 10.0e9`: Maximum memory in bytes for the full basis
  matrix.
- `multithreaded::Bool = true`: Use multithreaded basis-matrix construction.

## Returns
- `Psi2d::Matrix{T}`: Wavefunction values on the Cartesian grid.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(state::S; b = 5.0, inside_only = true, fundamental_domain = true, memory_limit = 10.0e9, multithreaded = true) where {S<:AbsState}
    let k = state.k, billiard = state.billiard, symmetries = state.basis.symmetries
        type = eltype(state.vec)
        L = CompositeCurve(get_boundary_curves(billiard)).length
        xlim, ylim = boundary_limits(get_boundary_curves(billiard); grd = max(1000, round(Int, k * L * b / (2 * pi))))
        dx = xlim[2] - xlim[1]; dy = ylim[2] - ylim[1]
        nx = max(round(Int, k * dx * b / (2 * pi)), 512); ny = max(round(Int, k * dy * b / (2 * pi)), 512)
        x_grid::Vector{type} = collect(type, range(xlim..., nx)); y_grid::Vector{type} = collect(type, range(ylim..., ny))
        if ~isnothing(symmetries)
            has_x = any(s -> s isa BilliardGeometry.XAxisReflection, symmetries)
            has_y = any(s -> s isa BilliardGeometry.YAxisReflection, symmetries)
            if has_x
                x_grid = rectify_grid(x_grid)
                nx = length(x_grid)
            end
            if has_y
                y_grid = rectify_grid(y_grid)
                ny = length(y_grid)
            end
        end
        Psi::Vector{type} = compute_psi(state, x_grid, y_grid; inside_only, memory_limit, multithreaded)
        Psi2d::Array{type,2} = reshape(Psi, (nx, ny))
        if ~fundamental_domain
            if ~isnothing(symmetries)
                Psi2d, x_grid, y_grid = apply_symmetries_to_wavefunction(Psi2d, x_grid, y_grid, symmetries, state.basis.sym_qnumbers)
            end
        end
        return Psi2d, x_grid, y_grid
    end
end

"""
    wavefunction(state::BasisState; xlim::Tuple = (-2.0, 2.0), ylim::Tuple = (-2.0, 2.0), b::Real = 5.0)

Evaluate a single basis function on a Cartesian grid.
## Arguments
- `state::BasisState`: Basis state to evaluate.

## Kwargs
- `xlim::Tuple = (-2.0, 2.0)`: Cartesian x limits.
- `ylim::Tuple = (-2.0, 2.0)`: Cartesian y limits.
- `b::Real = 5.0`: Approximate grid points per wavelength.

## Returns
- `Psi2d::Matrix{T}`: Basis-function values on the Cartesian grid.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(state::BasisState; xlim = (-2.0, 2.0), ylim = (-2.0, 2.0), b = 5.0)
    let k = state.k, basis = state.basis
        type = eltype(state.vec)
        dx = xlim[2] - xlim[1]; dy = ylim[2] - ylim[1]
        nx = max(round(Int, k * dx * b / (2 * pi)), 512); ny = max(round(Int, k * dy * b / (2 * pi)), 512)
        x_grid::Vector{type} = collect(type, range(xlim..., nx)); y_grid::Vector{type} = collect(type, range(ylim..., ny))
        pts_grid = [SVector(x, y) for y in y_grid for x in x_grid]
        Psi::Vector{type} = basis_fun(basis, state.idx, k, pts_grid)
        Psi2d::Array{type,2} = reshape(Psi, (nx, ny))
        return Psi2d, x_grid, y_grid
    end
end