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

function _wavefunction_grid(k::T, billiard::Bi, b::T; inside_only::Bool = true) where {T<:Real,Bi<:AbsBilliard}
    xlim, ylim = boundary_limits(full_boundary(billiard)); dx = T(2pi) / (b * k)
    nx = max(2, ceil(Int, (xlim[2] - xlim[1]) / dx) + 1); ny = max(2, ceil(Int, (ylim[2] - ylim[1]) / dx) + 1)
    xgrid = collect(range(T(xlim[1]), T(xlim[2]), length = nx)); ygrid = collect(range(T(ylim[1]), T(ylim[2]), length = ny))
    pts = vec([SVector{2,T}(x, y) for x in xgrid, y in ygrid])
    if inside_only
        mask = BitVector(undef, length(pts))
        Threads.@threads for i in eachindex(pts)
            mask[i] = BilliardGeometry.is_inside(billiard, pts[i])
        end
        indices = findall(mask)
    else
        indices = collect(eachindex(pts))
    end
    return xgrid, ygrid, pts, indices, nx, ny
end

@inline function _y0(r::T, k::T, cheb::ChebHankelPlanH, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r); p, t = panel_t(cheb, rf)
    return T(imag(eval_h(cheb, p, t, rf)))
end

@inline _y0(r::T, k::T, ::Nothing, ::Val{:direct}, ::Type{T}) where {T<:Real} = T(Bessels.bessely0(k * r))

@inline function _h1(r::T, k::T, cheb::ChebHankelPlanH, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r); p, t = panel_t(cheb, rf)
    return Complex{T}(eval_h(cheb, p, t, rf))
end

@inline _h1(r::T, k::T, ::Nothing, ::Val{:direct}, ::Type{T}) where {T<:Real} = Complex{T}(Bessels.hankelh1(1, k * r))

@inline function _h01(r::T, k::T, cheb::Tuple{ChebHankelPlanH,ChebHankelPlanH}, ::Val{:cheb}, ::Type{T}) where {T<:Real}
    rf = Float64(r); p, t = panel_t(cheb[1], rf)
    return Complex{T}(eval_h(cheb[1], p, t, rf)), Complex{T}(eval_h(cheb[2], p, t, rf))
end

@inline function _h01(r::T, k::T, ::Nothing, ::Val{:direct}, ::Type{T}) where {T<:Real}
    z = k * r
    return Complex{T}(Bessels.hankelh1(0, z)), Complex{T}(Bessels.hankelh1(1, z))
end

@inline function ϕ_slp(x::T, y::T, k::T, bd::BoundaryPoints{T}, u::AbstractVector{K}, cheb, mode::Val) where {T<:Real,K<:Number}
    acc = zero(promote_type(T, K))
    @inbounds for j in eachindex(u)
        p = bd.xy[j]; dx = x - p[1]; dy = y - p[2]; r2 = muladd(dx, dx, dy * dy)
        r2 == zero(T) && continue
        r = sqrt(r2); acc += _y0(r, k, cheb, mode, T) * bd.ds[j] * u[j]
    end
    return acc * T(0.25)
end

@inline function ϕ_dlp(x::T, y::T, k::T, bd::BoundaryPoints{T}, μ::AbstractVector{K}, cheb, mode::Val) where {T<:Real,K<:Number}
    acc = zero(promote_type(K, Complex{T})); k4 = k * T(0.25)
    @inbounds for j in eachindex(μ)
        p = bd.xy[j]; n = bd.normal[j]; dx = x - p[1]; dy = y - p[2]; r2 = muladd(dx, dx, dy * dy)
        r2 == zero(T) && continue
        r = sqrt(r2); inn = muladd(dx, n[1], dy * n[2])
        acc += (im * k4) * _h1(r, k, cheb, mode, T) * (inn / r) * μ[j] * bd.ds[j]
    end
    return acc
end

@inline function ϕ_cfie(x::T, y::T, k::T, bd::BoundaryPoints{T}, μ::AbstractVector{K}, cheb, mode::Val) where {T<:Real,K<:Number}
    acc = zero(promote_type(K, Complex{T})); k2 = k * T(0.5)
    @inbounds for j in eachindex(μ)
        p = bd.xy[j]; t = bd.tangent[j]; dx = x - p[1]; dy = y - p[2]; r2 = muladd(dx, dx, dy * dy)
        r2 == zero(T) && continue
        r = sqrt(r2); h0, h1 = _h01(r, k, cheb, mode, T)
        A = k2 * muladd(t[2], dx, -t[1] * dy) / r; B = k2 * hypot(t[1], t[2])
        acc -= bd.ws[j] * μ[j] * (im * A * h1 - B * h0)
    end
    return acc
end

function _cheb_interval(ks, bds, xgrid, ygrid)
    xmin = Float64(first(xgrid)); xmax = Float64(last(xgrid)); ymin = Float64(first(ygrid)); ymax = Float64(last(ygrid)); rmax = 0.0
    @inbounds for bd in bds, p in bd.xy
        x = Float64(p[1]); y = Float64(p[2])
        rmax = max(rmax, hypot(xmin - x, ymin - y), hypot(xmin - x, ymax - y), hypot(xmax - x, ymin - y), hypot(xmax - x, ymax - y))
    end
    kc = ComplexF64.(ks)
    return kc, hankel_z_chebyshev_cutoff / maximum(abs, kc), 1.05 * rmax
end

function _density_kernel(solver::DLP, ks, bds, xgrid, ygrid, cfg)
    kc, rmin, rmax = _cheb_interval(ks, bds, xgrid, ygrid)
    hp, _, _ = _tune_cheb_plans(rmin, rmax, kc, cfg; h_orders = (1,))
    return hp.h1, ϕ_dlp
end

function _density_kernel(solver::CFIE, ks, bds, xgrid, ygrid, cfg)
    kc, rmin, rmax = _cheb_interval(ks, bds, xgrid, ygrid)
    hp, _, _ = _tune_cheb_plans(rmin, rmax, kc, cfg; h_orders = (0, 1))
    return collect(zip(hp.h0, hp.h1)), ϕ_cfie
end

function _evaluate_wavefunctions(ks::Vector{T}, us::Vector{Vector{K}}, bds::Vector{P}, pts, indices, nx::Int, ny::Int, plans, ϕ::F, mode::M; MIN_CHUNK::Int = 4096, show_progress::Bool = true) where {T<:Real,K<:Number,P<:BoundaryPoints,F,M<:Val}
    V = ϕ === ϕ_slp ? promote_type(T, K) : promote_type(K, Complex{T})
    out = Vector{Matrix{V}}(undef, length(ks)); n = length(indices)
    nt = max(1, min(Threads.nthreads(), cld(n, MIN_CHUNK))); q, r = divrem(n, nt)
    @maybe_showprogress show_progress for i in eachindex(ks)
        ψ = zeros(V, nx * ny); k = ks[i]; u = us[i]; bd = bds[i]; plan = plans === nothing ? nothing : plans[i]
        Threads.@threads :static for t in 1:nt
            lo = (t - 1) * q + min(t - 1, r) + 1; hi = lo + q - 1 + (t <= r)
            @inbounds for j in lo:hi
                idx = indices[j]; p = pts[idx]
                ψ[idx] = ϕ(p[1], p[2], k, bd, u, plan, mode)
            end
        end
        out[i] = reshape(ψ, nx, ny)
    end
    return out
end

"""
    wavefunction(states::AbstractVector{<:BIMEigenstate{K,T}}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {K<:Number,T<:Real}

Reconstruct BIM eigenstates on a common Cartesian grid.

## Arguments
- `states::AbstractVector{<:BIMEigenstate{K,T}}`: BIM eigenstates.

## Keyword Arguments
- `b::Union{Real,Symbol} = :auto`: Grid points per wavelength.
- `inside_only::Bool = true`: Evaluate only inside the billiard.
- `MIN_CHUNK::Int = 4096`: Minimum spatial points per active thread.
- `use_chebyshev::Bool = true`: Use Chebyshev-accelerated Hankel evaluation.
- `show_progress::Bool = true`: Display reconstruction progress.
- `cheb_config::ChebyshevConfig = ChebyshevConfig(T)`: Chebyshev configuration.

## Returns
- `Psi2ds::Vector{Matrix}`: Reconstructed wavefunctions.
- `xgrid::Vector{T}`: Cartesian x coordinates.
- `ygrid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(states::AbstractVector{<:BIMEigenstate{K,T}}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, show_progress::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {K<:Number,T<:Real}
    isempty(states) && throw(ArgumentError("states must be nonempty"))
    s0 = first(states); n = length(states); P = typeof(s0.pts)
    ks = T[real(s.k) for s in states]; bds = P[s.pts for s in states]
    density = all(s.vec !== nothing for s in states)
    density || all(s.u !== nothing for s in states) || error("states must provide the same boundary representation")
    us = density ? Vector{K}[s.vec::Vector{K} for s in states] : Vector{K}[s.u::Vector{K} for s in states]
    bval = b === :auto ? T(_bim_grid_scale(s0.solver)) : T(b)
    xgrid, ygrid, pts, indices, nx, ny = _wavefunction_grid(maximum(ks), s0.billiard, bval; inside_only)
    if density
        if use_chebyshev
            plans, ϕ = _density_kernel(s0.solver, ks, bds, xgrid, ygrid, cheb_config)
            Psi = _evaluate_wavefunctions(ks, us, bds, pts, indices, nx, ny, plans, ϕ, Val(:cheb); MIN_CHUNK, show_progress)
        else
            ϕ = s0.solver isa DLP ? ϕ_dlp : ϕ_cfie
            Psi = _evaluate_wavefunctions(ks, us, bds, pts, indices, nx, ny, nothing, ϕ, Val(:direct); MIN_CHUNK, show_progress)
        end
    else
        if use_chebyshev
            kc, rmin, rmax = _cheb_interval(ks, bds, xgrid, ygrid)
            hp, _, _ = _tune_cheb_plans(rmin, rmax, kc, cheb_config; h_orders = (0,))
            Psi = _evaluate_wavefunctions(ks, us, bds, pts, indices, nx, ny, hp.h0, ϕ_slp, Val(:cheb); MIN_CHUNK, show_progress)
        else
            Psi = _evaluate_wavefunctions(ks, us, bds, pts, indices, nx, ny, nothing, ϕ_slp, Val(:direct); MIN_CHUNK, show_progress)
        end
    end
    return Psi, xgrid, ygrid
end

"""
    wavefunction(state::BIMEigenstate{K,T}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {K<:Number,T<:Real}

Reconstruct one BIM eigenstate.

## Arguments
- `state::BIMEigenstate{K,T}`: BIM eigenstate.

## Keyword Arguments
- `b::Union{Real,Symbol} = :auto`: Grid points per wavelength.
- `inside_only::Bool = true`: Evaluate only inside the billiard.
- `MIN_CHUNK::Int = 4096`: Minimum spatial points per active thread.
- `use_chebyshev::Bool = true`: Use Chebyshev-accelerated Hankel evaluation.
- `cheb_config::ChebyshevConfig = ChebyshevConfig(T)`: Chebyshev configuration.

## Returns
- `Psi::Matrix`: Reconstructed wavefunction.
- `xgrid::Vector{T}`: Cartesian x coordinates.
- `ygrid::Vector{T}`: Cartesian y coordinates.
"""
function wavefunction(state::BIMEigenstate{K,T}; b::Union{Real,Symbol} = :auto, inside_only::Bool = true, MIN_CHUNK::Int = 4096, use_chebyshev::Bool = true, cheb_config::ChebyshevConfig = ChebyshevConfig(T)) where {K<:Number,T<:Real}
    Psi, xgrid, ygrid = wavefunction([state]; b, inside_only, MIN_CHUNK, use_chebyshev, show_progress = false, cheb_config)
    return Psi[1], xgrid, ygrid
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