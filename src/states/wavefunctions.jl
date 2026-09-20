################################################################################
# BIM WAVEFUNCTION RECONSTRUCTION
#
# Dirichlet eigenfunctions are reconstructed from either the native BIM layer
# density or the physical boundary normal derivative u = ∂ₙψ:
#
#     DLP density:       ψ = D_k μ
#     CFIE density:      ψ ∝ -(D_k + i k S_k) μ
#     physical ∂ₙψ:      ψ = 1/4 ∫∂Ω Y₀(k|x-q|) u(q) ds_q
#
# Chebyshev acceleration requires only Hankel functions:
#
#     DLP:    H₁⁽¹⁾
#     CFIE:   H₀⁽¹⁾, H₁⁽¹⁾
#     SLP:    H₀⁽¹⁾ through Y₀ = Im H₀⁽¹⁾.
#
# Direct and Chebyshev evaluation use separate kernels so that all types in the
# threaded spatial loop remain concrete.
################################################################################

############## HELPER FUNCTIONS #############

function pad_limits(xlim, ylim; padding::Real = 0.01)
    return (xlim[1] - padding, xlim[2] + padding), (ylim[1] - padding, ylim[2] + padding)
end

function rectify_grid(grid::AbstractVector{T}) where {T<:Real}
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
    return pad_limits(xlim, ylim; padding)
end

################################################################################
# CARTESIAN GRID
################################################################################

function _wavefunction_grid(k::T, solver::SweepBIMSolver, billiard::Bi, b::T; inside_only::Bool = true) where {T<:Real,Bi<:BilliardGeometry.AbsBilliard}
    curves = billiard.full_boundary
    xlim, ylim = boundary_limits(curves)
    dx = T(2pi) / (b * k)
    nx = max(2, ceil(Int, (xlim[2] - xlim[1]) / dx) + 1)
    ny = max(2, ceil(Int, (ylim[2] - ylim[1]) / dx) + 1)
    x_grid = collect(range(T(xlim[1]), T(xlim[2]), length = nx))
    y_grid = collect(range(T(ylim[1]), T(ylim[2]), length = ny))
    pts = vec([SVector{2,T}(x, y) for x in x_grid, y in y_grid])
    if inside_only
        mask = BitVector(undef, length(pts))
        Threads.@threads for i in eachindex(pts)
            mask[i] = BilliardGeometry.is_inside(billiard, pts[i])
        end
        indices = findall(mask)
    else
        indices = collect(eachindex(pts))
    end
    return x_grid, y_grid, pts, indices, nx, ny
end

################################################################################
# CYLINDRICAL-FUNCTION EVALUATION
################################################################################

@inline function _y0(r::T, k::T, cheb::ChebHankelPlanH, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r)
    pidx, t = panel_t(cheb, rf)
    return T(imag(eval_h(cheb, pidx, t, rf)))
end

@inline _y0(r::T, k::T, ::Nothing, ::Val{:direct}, ::Type{T}) where {T<:Real} = T(Bessels.bessely0(k * r))

@inline function _h1(r::T, k::T, cheb::ChebHankelPlanH, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r)
    pidx, t = panel_t(cheb, rf)
    return Complex{T}(eval_h(cheb, pidx, t, rf))
end

@inline _h1(r::T, k::T, ::Nothing, ::Val{:direct}, ::Type{T}) where {T<:Real} = Complex{T}(Bessels.hankelh1(1, k * r))

@inline function _h01(r::T, k::T, cheb::Tuple{ChebHankelPlanH,ChebHankelPlanH}, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r)
    pidx, t = panel_t(cheb[1], rf)
    return Complex{T}(eval_h(cheb[1], pidx, t, rf)), Complex{T}(eval_h(cheb[2], pidx, t, rf))
end

@inline function _h01(r::T, k::T, ::Nothing, ::Val{:direct}, ::Type{T}) where {T<:Real}
    z = k * r
    return Complex{T}(Bessels.hankelh1(0, z)), Complex{T}(Bessels.hankelh1(1, z))
end

@inline function ϕ_slp(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{K}, cheb, mode::Val) where {T<:Real,K<:Number}
    xy = bd.xy; ds = bd.ds
    S = promote_type(T, K); acc = zero(S)
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

@inline function ϕ_dlp(x::T, y::T, k::T, bd::BoundaryPoints{T}, μ::AbstractVector{K}, cheb, mode::Val) where {T<:Real,K<:Number}
    xy = bd.xy; normal = bd.normal; ds = bd.ds
    S = promote_type(K, Complex{T}); acc = zero(S); kquarter = k * T(0.25)
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

@inline function ϕ_cfie(x::T, y::T, k::T, bd::BoundaryPoints{T}, μ::AbstractVector{K}, cheb, mode::Val) where {T<:Real,K<:Number}
    xy = bd.xy; tangent = bd.tangent; ws = bd.ws
    S = promote_type(K, Complex{T}); acc = zero(S); khalf = k * T(0.5)
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
# LOW-LEVEL WAVEFUNCTION API
################################################################################

"""
    wavefunction(solver::SweepBIMSolver, ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, billiard::Bi; layer_density::Bool = true, b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {T<:Real,K<:Number,P<:BoundaryPoints,Bi<:BilliardGeometry.AbsBilliard}

Reconstruct BIM eigenfunctions on a common Cartesian grid.

## Arguments
- `solver::SweepBIMSolver`: BIM solver associated with the states.
- `ks::Vector{T}`: Real wave numbers of the eigenstates.
- `vec_u::Vector{Vector{K}}`: Native layer densities or physical boundary normal derivatives.
- `vec_pts::Vector{P}`: Full-boundary discretizations corresponding to `ks`.
- `billiard::Bi`: Billiard geometry on which the wavefunctions are reconstructed.

## Keyword Arguments
- `layer_density::Bool = true`: Interpret `vec_u` as native BIM layer densities if `true`, or physical boundary normal derivatives `u = ∂ₙψ` if `false`.
- `b::Union{Real,Symbol} = :auto`: Grid points per wavelength; `:auto` uses `_bim_grid_scale(solver)`.
- `inside_only::Bool = true`: Evaluate the wavefunctions only at points inside the billiard.
- `MIN_CHUNK::Int = 4096`: Minimum number of spatial points assigned to an active thread.
- `use_chebyshev::Bool = true`: Use Chebyshev-accelerated cylindrical-function evaluation.
- `show_progress::Bool = true`: Display progress over the eigenstates.
- `cheb_config::ChebyshevConfig = ChebyshevConfig(T)`: Chebyshev tuning configuration.

## Returns
- `Psi2ds::Vector{Matrix}`: Reconstructed wavefunctions in the same order as `ks`.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(solver::SweepBIMSolver, ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, billiard::Bi; layer_density::Bool = true, b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {T<:Real,K<:Number,P<:BoundaryPoints,Bi<:BilliardGeometry.AbsBilliard}
    length(ks) == length(vec_u) == length(vec_pts) || throw(DimensionMismatch("ks, vec_u, and vec_pts must have equal length"))
    isempty(ks) && throw(ArgumentError("ks must be nonempty"))
    kmax = maximum(ks); bval = b === :auto ? T(_bim_grid_scale(solver)) : T(b)
    x_grid, y_grid, pts, indices, nx, ny = _wavefunction_grid(kmax, solver, billiard, bval; inside_only)
    representation = layer_density ? Val(:density) : Val(:boundary)
    mode = use_chebyshev ? Val(:cheb) : Val(:direct)
    Psi2ds = _wavefunctions(solver, ks, vec_u, vec_pts, x_grid, y_grid, pts, indices, nx, ny, representation, mode; MIN_CHUNK, show_progress, cheb_config)
    return Psi2ds, x_grid, y_grid
end

################################################################################
# CHEBYSHEV INTERVAL
################################################################################

function _wavefunction_cheb_interval(ks::Vector{T}, vec_pts::Vector{P}, x_grid::Vector{T}, y_grid::Vector{T}) where {T<:Real,P<:BoundaryPoints}
    xmin = Float64(x_grid[1]); xmax = Float64(x_grid[end]); ymin = Float64(y_grid[1]); ymax = Float64(y_grid[end]); rmax = 0.0
    @inbounds for bd in vec_pts, p in bd.xy
        px = Float64(p[1]); py = Float64(p[2])
        rmax = max(rmax, hypot(xmin - px, ymin - py), hypot(xmin - px, ymax - py), hypot(xmax - px, ymin - py), hypot(xmax - px, ymax - py))
    end
    ks_cheb = ComplexF64.(ks)
    rmin = hankel_z_chebyshev_cutoff / maximum(abs, ks_cheb)
    return ks_cheb, rmin, 1.05 * rmax
end

################################################################################
# REPRESENTATION DISPATCH
################################################################################

# Native DLP layer density μ: reconstruct with the double-layer potential.
function _wavefunctions(solver::DLP, ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:density}, ::Val{:cheb}; MIN_CHUNK::Int = 4096, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {T<:Real,K<:Number,P<:BoundaryPoints}
    ks_cheb, rmin, rmax = _wavefunction_cheb_interval(ks, vec_pts, x_grid, y_grid)
    hp, _, _ = _tune_cheb_plans(rmin, rmax, ks_cheb, cheb_config; h_orders = (1,))
    V = promote_type(K, Complex{T})
    return _wavefunctions_cheb(ks, vec_u, vec_pts, pts, indices, nx, ny, hp.h1, V, ϕ_dlp; MIN_CHUNK, show_progress)
end

function _wavefunctions(solver::DLP, ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:density}, ::Val{:direct}; MIN_CHUNK::Int = 4096, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {T<:Real,K<:Number,P<:BoundaryPoints}
    V = promote_type(K, Complex{T})
    return _wavefunctions_direct(ks, vec_u, vec_pts, pts, indices, nx, ny, V, ϕ_dlp; MIN_CHUNK, show_progress)
end

# Native CFIE layer density μ: reconstruct with the combined-field potential.
function _wavefunctions(solver::CFIE, ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:density}, ::Val{:cheb}; MIN_CHUNK::Int = 4096, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {T<:Real,K<:Number,P<:BoundaryPoints}
    ks_cheb, rmin, rmax = _wavefunction_cheb_interval(ks, vec_pts, x_grid, y_grid)
    hp, _, _ = _tune_cheb_plans(rmin, rmax, ks_cheb, cheb_config; h_orders = (0, 1))
    plans = collect(zip(hp.h0, hp.h1))
    V = promote_type(K, Complex{T})
    return _wavefunctions_cheb(ks, vec_u, vec_pts, pts, indices, nx, ny, plans, V, ϕ_cfie; MIN_CHUNK, show_progress)
end

function _wavefunctions(solver::CFIE, ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:density}, ::Val{:direct}; MIN_CHUNK::Int = 4096, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {T<:Real,K<:Number,P<:BoundaryPoints}
    V = promote_type(K, Complex{T})
    return _wavefunctions_direct(ks, vec_u, vec_pts, pts, indices, nx, ny, V, ϕ_cfie; MIN_CHUNK, show_progress)
end

# Physical boundary derivative u = ∂ₙψ: SLP reconstruction is independent of the original BIM formulation.
function _wavefunctions(solver::SweepBIMSolver, ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:boundary}, ::Val{:cheb}; MIN_CHUNK::Int = 4096, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {T<:Real,K<:Number,P<:BoundaryPoints}
    ks_cheb, rmin, rmax = _wavefunction_cheb_interval(ks, vec_pts, x_grid, y_grid)
    hp, _, _ = _tune_cheb_plans(rmin, rmax, ks_cheb, cheb_config; h_orders = (0,))
    V = promote_type(T, K)
    return _wavefunctions_cheb(ks, vec_u, vec_pts, pts, indices, nx, ny, hp.h0, V, ϕ_slp; MIN_CHUNK, show_progress)
end

function _wavefunctions(solver::SweepBIMSolver, ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, x_grid::Vector{T}, y_grid::Vector{T}, pts, indices, nx::Int, ny::Int, ::Val{:boundary}, ::Val{:direct}; MIN_CHUNK::Int = 4096, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {T<:Real,K<:Number,P<:BoundaryPoints}
    V = promote_type(T, K)
    return _wavefunctions_direct(ks, vec_u, vec_pts, pts, indices, nx, ny, V, ϕ_slp; MIN_CHUNK, show_progress)
end

################################################################################
# SPECIALIZED SPATIAL KERNELS
################################################################################

function _wavefunctions_direct(ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, pts, indices, nx::Int, ny::Int, ::Type{V}, ϕ::F; MIN_CHUNK::Int = 4096, show_progress::Bool = true) where {T<:Real,K<:Number,P<:BoundaryPoints,V<:Number,F}
    Psi2ds = Vector{Matrix{V}}(undef, length(ks))
    nmask = length(indices); NT_eff = max(1, min(Threads.nthreads(), cld(nmask, MIN_CHUNK))); q, r = divrem(nmask, NT_eff)
    @maybe_showprogress show_progress for i in eachindex(ks)
        Psi = zeros(V, nx * ny); k = ks[i]; u = vec_u[i]; bd = vec_pts[i]
        Threads.@threads :static for t in 1:NT_eff
            lo = (t - 1) * q + min(t - 1, r) + 1; hi = lo + q - 1 + (t <= r ? 1 : 0)
            @inbounds for jj in lo:hi
                idx = indices[jj]; p = pts[idx]
                Psi[idx] = ϕ(p[1], p[2], k, bd, u, nothing, Val(:direct))
            end
        end
        Psi2ds[i] = reshape(Psi, nx, ny)
    end
    return Psi2ds
end

function _wavefunctions_cheb(ks::Vector{T}, vec_u::Vector{Vector{K}}, vec_pts::Vector{P}, pts, indices, nx::Int, ny::Int, cheb_plans::Vector{C}, ::Type{V}, ϕ::F; MIN_CHUNK::Int = 4096, show_progress::Bool = true) where {T<:Real,K<:Number,P<:BoundaryPoints,C,V<:Number,F}
    Psi2ds = Vector{Matrix{V}}(undef, length(ks))
    nmask = length(indices); NT_eff = max(1, min(Threads.nthreads(), cld(nmask, MIN_CHUNK))); q, r = divrem(nmask, NT_eff)
    @maybe_showprogress show_progress for i in eachindex(ks)
        Psi = zeros(V, nx * ny); k = ks[i]; u = vec_u[i]; bd = vec_pts[i]; cheb = cheb_plans[i]
        Threads.@threads :static for t in 1:NT_eff
            lo = (t - 1) * q + min(t - 1, r) + 1; hi = lo + q - 1 + (t <= r ? 1 : 0)
            @inbounds for jj in lo:hi
                idx = indices[jj]; p = pts[idx]
                Psi[idx] = ϕ(p[1], p[2], k, bd, u, cheb, Val(:cheb))
            end
        end
        Psi2ds[i] = reshape(Psi, nx, ny)
    end
    return Psi2ds
end

################################################################################
# BIMEIGENSTATE API
################################################################################

"""
    wavefunction(states::AbstractVector{<:BIMEigenstate{K,T}}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {K<:Number,T<:Real}

Reconstruct a collection of BIM eigenstates on one common Cartesian grid.
Native layer densities are used when available for every state. Otherwise all
states must provide the physical boundary normal derivative `u = ∂ₙψ`.

## Arguments
- `states::AbstractVector{<:BIMEigenstate{K,T}}`: BIM eigenstates to reconstruct.

## Keyword Arguments
- `b::Union{Real,Symbol} = :auto`: Grid points per wavelength; `:auto` uses the solver default.
- `inside_only::Bool = true`: Evaluate the wavefunctions only at points inside the billiard.
- `MIN_CHUNK::Int = 4096`: Minimum number of spatial points assigned to an active thread.
- `use_chebyshev::Bool = true`: Use Chebyshev-accelerated cylindrical-function evaluation.
- `show_progress::Bool = true`: Display progress over the eigenstates.
- `cheb_config::ChebyshevConfig = ChebyshevConfig(T)`: Chebyshev tuning configuration.

## Returns
- `Psi2ds::Vector{Matrix}`: Reconstructed wavefunctions in the same order as `states`.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(states::AbstractVector{<:BIMEigenstate{K,T}}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {K<:Number,T<:Real}
    isempty(states) && throw(ArgumentError("states must be nonempty"))
    s0 = first(states); P = typeof(s0.pts); n = length(states)
    ks = Vector{T}(undef, n); vec_u = Vector{Vector{K}}(undef, n); vec_pts = Vector{P}(undef, n)
    layer_density = all(s.vec !== nothing for s in states)
    layer_density || all(s.u !== nothing for s in states) || error("BIMEigenstates must provide the same boundary representation")
    @inbounds for i in eachindex(states)
        s = states[i]
        ks[i] = real(s.k)
        vec_u[i] = layer_density ? (s.vec::Vector{K}) : (s.u::Vector{K})
        vec_pts[i] = s.pts
    end
    return wavefunction(s0.solver, ks, vec_u, vec_pts, s0.billiard; layer_density, b, inside_only, MIN_CHUNK, use_chebyshev, show_progress, cheb_config)
end

"""
    wavefunction(state::BIMEigenstate{K,T}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {K<:Number,T<:Real}

Reconstruct the interior wavefunction of one BIM eigenstate.
The native layer density `state.vec` is used when available. Otherwise the
physical boundary normal derivative `state.u = ∂ₙψ` is used.

## Arguments
- `state::BIMEigenstate{K,T}`: BIM eigenstate to reconstruct.

## Keyword Arguments
- `b::Union{Real,Symbol} = :auto`: Grid points per wavelength; `:auto` uses the solver default.
- `inside_only::Bool = true`: Evaluate the wavefunction only at points inside the billiard.
- `MIN_CHUNK::Int = 4096`: Minimum number of spatial points assigned to an active thread.
- `use_chebyshev::Bool = true`: Use Chebyshev-accelerated cylindrical-function evaluation.
- `cheb_config::ChebyshevConfig = ChebyshevConfig(T)`: Chebyshev tuning configuration.

## Returns
- `Psi::Matrix`: Reconstructed wavefunction.
- `x_grid::Vector{T}`: Cartesian x coordinates.
- `y_grid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(state::BIMEigenstate{K,T}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {K<:Number,T<:Real}
    P = typeof(state.pts); ks = T[real(state.k)]; vec_pts = P[state.pts]
    if state.vec !== nothing
        vec_u = Vector{K}[state.vec::Vector{K}]
        Psi, x_grid, y_grid = wavefunction(state.solver, ks, vec_u, vec_pts, state.billiard; layer_density = true, b, inside_only, MIN_CHUNK, use_chebyshev, show_progress = false, cheb_config)
    elseif state.u !== nothing
        vec_u = Vector{K}[state.u::Vector{K}]
        Psi, x_grid, y_grid = wavefunction(state.solver, ks, vec_u, vec_pts, state.billiard; layer_density = false, b, inside_only, MIN_CHUNK, use_chebyshev, show_progress = false, cheb_config)
    else
        error("BIMEigenstate has neither vec nor u")
    end
    return Psi[1], x_grid, y_grid
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