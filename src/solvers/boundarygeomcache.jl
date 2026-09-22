"""
    BoundaryPanelArrays{T}

Stores one-dimensional coordinate arrays extracted from a parametrized
[`BoundaryPoints`](@ref) discretization.

## Attributes
* `X`, `Y`: Cartesian coordinates of the boundary nodes.
* `dX`, `dY`: Cartesian components of the parametrization derivative (`pts.tangent`).
* `speed`: Parametrization speed `|γ'(t)|` at each node.
"""
struct BoundaryPanelArrays{T<:Real}
    X::Vector{T}
    Y::Vector{T}
    dX::Vector{T}
    dY::Vector{T}
    speed::Vector{T}
end

"""
    _boundary_panel_arrays_cache(pts::BoundaryPoints{T}) where T<:Real → cache::BoundaryPanelArrays{T}

Extracts coordinate, tangent and speed arrays from a parametrized boundary
discretization.
"""
@inline function _boundary_panel_arrays_cache(pts::BoundaryPoints{T}) where T<:Real
    X = getindex.(pts.xy, 1)
    Y = getindex.(pts.xy, 2)
    dX = getindex.(pts.tangent, 1)
    dY = getindex.(pts.tangent, 2)
    speed = @. hypot(dX, dY)
    return BoundaryPanelArrays(X, Y, dX, dY, speed)
end

"""
    component_normals(pts::BoundaryPoints{T}) where T<:Real → (nx, ny, speed)

Returns the Cartesian components of the stored outward normals together with
the parametrization speed `|γ'(t)|` (equal to one when tangent data are not
stored).
"""
function component_normals(pts::BoundaryPoints{T}) where T<:Real
    length(pts.normal) == length(pts) || throw(ArgumentError("BoundaryPoints does not contain normal data"))
    nx = getindex.(pts.normal, 1)
    ny = getindex.(pts.normal, 2)
    if length(pts.tangent) == length(pts)
        tx = getindex.(pts.tangent, 1)
        ty = getindex.(pts.tangent, 2)
        speed = @. hypot(tx, ty)
    else
        speed = ones(T, length(pts))
    end
    return nx, ny, speed
end

"""
    flatten_boundary_components(comps::Vector{BoundaryPoints{T}}) where T<:Real → (; x, y, nx, ny, ds, offs)

Flattens multiple boundary components into contiguous coordinate, normal and
quadrature arrays, returning a named tuple with `x`, `y`, `nx`, `ny`, `ds` and
the component offsets `offs` (see [`component_offsets`](@ref)).
"""
function flatten_boundary_components(comps::Vector{BoundaryPoints{T}}) where T<:Real
    N = sum(length, comps)
    x = Vector{T}(undef, N)
    y = Vector{T}(undef, N)
    nx = Vector{T}(undef, N)
    ny = Vector{T}(undef, N)
    ds = Vector{T}(undef, N)
    offs = component_offsets(comps)
    p = 1
    @inbounds for c in comps
        cnx, cny, _ = component_normals(c)
        for j in eachindex(c.xy)
            q = c.xy[j]
            x[p] = q[1]
            y[p] = q[2]
            nx[p] = cnx[j]
            ny[p] = cny[j]
            ds[p] = c.ds[j]
            p += 1
        end
    end
    return (; x, y, nx, ny, ds, offs)
end

"""
    flatten_boundary_ds(comps::Vector{BoundaryPoints{T}}) where T<:Real → ds::Vector{T}

Concatenates the arc-length quadrature elements of all boundary components
into a single contiguous vector.
"""
function flatten_boundary_ds(comps::Vector{BoundaryPoints{T}}) where T<:Real
    ds = Vector{T}(undef, boundary_matrix_size(comps))
    p = 1
    @inbounds for c in comps
        n = length(c.ds)
        ds[p:p+n-1] .= c.ds
        p += n
    end
    return ds
end

"""
    BoundaryGeomCache{T}

Stores pairwise and local geometric quantities reused during boundary-integral
matrix assembly.

## Attributes
* `R`: Pairwise distances between boundary nodes (diagonal set to `one(T)`).
* `invR`: Pairwise inverse distances, with zero diagonal.
* `inner`: Tangential interaction term between source tangents and point differences.
* `logterm`: Periodic logarithmic kernel used in Kress splitting.
* `speed`: Parametrization speed at each boundary node.
* `kappa`: Scaled curvature term used in diagonal kernel limits.
* `original_ts`: Copy of the computational parameter nodes for corner-graded Kress discretizations (empty unless requested).
"""
struct BoundaryGeomCache{T<:Real}
    R::Matrix{T}
    invR::Matrix{T}
    inner::Matrix{T}
    logterm::Matrix{T}
    speed::Vector{T}
    kappa::Vector{T}
    original_ts::Vector{T}
end

"""
    boundary_geom_cache(pts::BoundaryPoints{T}, corner_kress::Bool=false) where T<:Real → cache::BoundaryGeomCache{T}

Constructs the geometric cache associated with a parametrized boundary
discretization `pts` (which must have `tangent`, `tangent_2` and `ts`
populated): precomputes pairwise distances, inverse distances, tangential
interaction factors, the periodic logarithmic Kress kernel, parametrization
speeds and the scaled curvature entering diagonal kernel limits.

If `corner_kress=true`, `pts.ts` is copied into `original_ts` for use by the
corner-graded Kress construction.
"""
function boundary_geom_cache(pts::BoundaryPoints{T}, corner_kress::Bool=false) where {T<:Real}
    N = length(pts)
    length(pts.tangent)==N || throw(ArgumentError("BoundaryPoints does not contain tangent data"))
    length(pts.tangent_2)==N || throw(ArgumentError("BoundaryPoints does not contain tangent_2 data"))
    length(pts.ts)==N || throw(ArgumentError("BoundaryPoints does not contain ts data"))
    ts = pts.ts
    X = getindex.(pts.xy, 1); Y = getindex.(pts.xy, 2)
    dX = getindex.(pts.tangent, 1); dY = getindex.(pts.tangent, 2)
    ddX = getindex.(pts.tangent_2, 1); ddY = getindex.(pts.tangent_2, 2)
    R = Matrix{T}(undef, N, N); invR = Matrix{T}(undef, N, N)
    inner = Matrix{T}(undef, N, N); logterm = Matrix{T}(undef, N, N)
    @inbounds for i in 1:N
        R[i,i] = one(T); invR[i,i] = zero(T)
        inner[i,i] = zero(T); logterm[i,i] = zero(T)
    end
    Threads.@threads for j in 2:N
        xj = X[j]; yj = Y[j]; dxj = dX[j]; dyj = dY[j]; tj = ts[j]
        @inbounds for i in 1:j-1
            Δx = X[i]-xj; Δy = Y[i]-yj
            r = hypot(Δx, Δy); rinv = inv(r)
            R[i,j] = r; R[j,i] = r
            invR[i,j] = rinv; invR[j,i] = rinv
            inner[i,j] = dyj*Δx-dxj*Δy
            inner[j,i] = -dY[i]*Δx+dX[i]*Δy
            Δt = ts[i]-tj
            lt = log(4*sin(Δt/2)^2)
            logterm[i,j] = lt; logterm[j,i] = lt
        end
    end
    speed = @. hypot(dX, dY)
    κnum = @. -(dX*ddY-dY*ddX)
    κden = @. dX*dX+dY*dY
    kappa = @. κnum/(2*T(pi)*κden)
    original_ts = corner_kress ? copy(ts) : T[]
    return QuantumBilliards.BoundaryGeomCache(R, invR, inner, logterm, speed, kappa, original_ts)
end
#= REF IMPLEMENTATION
function boundary_geom_cache(pts::BoundaryPoints{T}, corner_kress::Bool=false) where T<:Real
    N = length(pts)
    length(pts.tangent) == N || throw(ArgumentError("BoundaryPoints does not contain tangent data"))
    length(pts.tangent_2) == N || throw(ArgumentError("BoundaryPoints does not contain tangent_2 data"))
    length(pts.ts) == N || throw(ArgumentError("BoundaryPoints does not contain ts data"))
    ts = pts.ts
    X = getindex.(pts.xy, 1)
    Y = getindex.(pts.xy, 2)
    dX = getindex.(pts.tangent, 1)
    dY = getindex.(pts.tangent, 2)
    ddX = getindex.(pts.tangent_2, 1)
    ddY = getindex.(pts.tangent_2, 2)
    ΔX = @. X - X'
    ΔY = @. Y - Y'
    R = hypot.(ΔX, ΔY)
    R[diagind(R)] .= one(T)
    invR = inv.(R)
    invR[diagind(invR)] .= zero(T)
    dX_row = reshape(dX, 1, N)
    dY_row = reshape(dY, 1, N)
    inner = @. dY_row*ΔX - dX_row*ΔY
    original_ts = corner_kress ? copy(ts) : T[]
    ΔT = ts .- ts'
    logterm = log.(4 .*sin.(ΔT./2).^2)
    logterm[diagind(logterm)] .= zero(T)
    speed = @. hypot(dX, dY)
    κnum = -(dX.*ddY .- dY.*ddX)
    κden = dX.^2 .+ dY.^2
    kappa = (inv(2*T(pi))).*(κnum./κden)
    return BoundaryGeomCache(R, invR, inner, logterm, speed, kappa, original_ts)
end
=#

################################################################################
############### COMPLEX-SAFE HANKEL/BESSEL-J KERNEL EVALUATION ###############
################################################################################

# Dispatch helpers shared by every BIM kernel assembly (DLP/CFIE/Composite,
# see dlp.jl/cfie.jl/compositebim.jl) and by BeynSolver's complex-contour
# evaluation (see acceleratedmethods/beyn.jl). Bessels.jl is faster but only
# supports real arguments; SpecialFunctions.jl (AMOS) is used for the
# complex-k contour nodes Beyn needs. Dispatch is resolved at compile time on
# the concrete (real-vs-complex) type of `z`, so real-k sweep solves keep
# using the fast Bessels.jl path unchanged.
@inline _bim_hankelh1(ν::Int, z::Real) = Bessels.hankelh1(ν, z)
@inline _bim_hankelh1(ν::Int, z::Complex) = SpecialFunctions.besselh(ν, 1, z)

# Returns besselj(ν,z) given the already-computed hankelh1(ν,z)=h. For real
# z, H=J+iY so J=real(H) (saves a second special-function call on the DLP/CFIE
# hot path); for complex z that identity does not hold, so besselj is
# evaluated directly.
@inline _bim_besselj(::Int, ::Real, h::Complex) = real(h)
@inline _bim_besselj(ν::Int, z::Complex, ::Complex) = SpecialFunctions.besselj(ν, z)

# Widens a wavenumber `k` to the numeric type `T` used by a SweepBIMSolver,
# preserving whether it is real (ordinary sweep solve) or complex (Beyn
# contour node) instead of forcing `T(k)`, which would throw for a genuinely
# complex `k`.
@inline _bim_widen_k(::Type{T}, k::Real) where {T<:Real} = T(k)
@inline _bim_widen_k(::Type{T}, k::Complex) where {T<:Real} = Complex{T}(k)
