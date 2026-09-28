################################################################################
# REAL PLANE-WAVE TAYLOR CACHE
#
# For a real plane-wave basis function
#
#                   ϕ(k;x,y) = Fx(k vx x) Fy(k vy y),
#
# where Fx,Fy ∈ {cos,sin}, define
#
#                   a = vx x,       b = vy y,
#                   q₋ = a - b,     q₊ = a + b.
#
# Product-to-sum identities express every parity combination as a linear
# combination of sin(kq₋), cos(kq₋), sin(kq₊), and cos(kq₊). Each component
# satisfies
#
#                         y''(k) + q² y(k) = 0.
#
# For the normalized Taylor expansion
#
#                   y(k₀ + δ) = Σₙ aₙ δⁿ,
#
# the coefficients therefore obey the exact two-step recurrence
#
#                  aₙ₊₂ = -q² aₙ / [(n+2)(n+1)].
#
# The cache stores the Taylor coefficients of the complete basis functions.
# Once constructed, the basis and its k-derivative are evaluated using the
# common Horner routines without further trigonometric evaluations.
################################################################################

"""
    RealPlaneWaveTaylorCache{T} <: BasisCache

Cached Taylor representation of a [`RealPlaneWaves`](@ref) basis about a
reference wavenumber.

## Attributes
* `k0`: Reference wavenumber of the Taylor expansion.
* `degree`: Degree of the Taylor expansion.
* `coeffs`: Normalized Taylor coefficients with dimensions `(degree + 1, number of points, basis dimension)`.
"""
struct RealPlaneWaveTaylorCache{T<:Real} <: BasisCache
    k0::T
    degree::Int
    coeffs::Array{T,3}
end

"""
    _real_plane_wave_taylor_coefficients!(C::Array{T,3}, j::Int, c::Int, x::T, y::T, vx::T, vy::T, parx::Int, pary::Int, k0::T, p::Int) where {T<:Real}

Construct the normalized Taylor coefficients of one real plane-wave basis
function at one spatial point.

## Arguments
* `C`: Taylor coefficient tensor.
* `j`: Spatial-point index.
* `c`: Basis-function index.
* `x`: x coordinate of the spatial point.
* `y`: y coordinate of the spatial point.
* `vx`: x component of the propagation direction.
* `vy`: y component of the propagation direction.
* `parx`: x parity selector (`+1` = cosine, `-1` = sine).
* `pary`: y parity selector (`+1` = cosine, `-1` = sine).
* `k0`: Taylor expansion center.
* `p`: Taylor degree.

## Returns
* `nothing`: The coefficients are written in-place to `C[:, j, c]`.
"""
@inline function _real_plane_wave_taylor_coefficients!(C::Array{T,3}, j::Int, c::Int, x::T, y::T, vx::T, vy::T, parx::Int, pary::Int, k0::T, p::Int) where {T<:Real}
    a = vx * x
    b = vy * y
    qm = a - b
    qp = a + b
    sm, cm = sincos(k0 * qm)
    sp, cp = sincos(k0 * qp)
    if parx == 1 && pary == 1
        m0, p0 = cm, cp
        m1, p1 = -qm * sm, -qp * sp
        σm, σp = one(T), one(T)
    elseif parx == -1 && pary == -1
        m0, p0 = cm, cp
        m1, p1 = -qm * sm, -qp * sp
        σm, σp = one(T), -one(T)
    elseif parx == -1 && pary == 1
        m0, p0 = sm, sp
        m1, p1 = qm * cm, qp * cp
        σm, σp = one(T), one(T)
    else
        m0, p0 = sm, sp
        m1, p1 = qm * cm, qp * cp
        σm, σp = -one(T), one(T)
    end
    C[1, j, c] = (σm * m0 + σp * p0) / 2
    p == 0 && return nothing
    C[2, j, c] = (σm * m1 + σp * p1) / 2
    qm2 = qm * qm
    qp2 = qp * qp
    me, pe = m0, p0
    mo, po = m1, p1
    @inbounds for n = 0:p - 2
        den = T((n + 2) * (n + 1))
        if iseven(n)
            me = -qm2 * me / den
            pe = -qp2 * pe / den
            C[n + 3, j, c] = (σm * me + σp * pe) / 2
        else
            mo = -qm2 * mo / den
            po = -qp2 * po / den
            C[n + 3, j, c] = (σm * mo + σp * po) / 2
        end
    end
    return nothing
end

"""
    RealPlaneWaveTaylorCache(basis::RealPlaneWaves{T}, k0::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → cache::RealPlaneWaveTaylorCache{T}

Construct a Taylor cache for a [`RealPlaneWaves`](@ref) basis.

## Arguments
* `basis`: Basis for which the cache is constructed.
* `k0`: Reference wavenumber of the Taylor expansion.
* `pts`: Points at which the basis is represented.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether cache construction is multithreaded across basis functions.

## Returns
* `cache`: [`RealPlaneWaveTaylorCache`](@ref) containing the normalized Taylor coefficients.
"""
function RealPlaneWaveTaylorCache(basis::RealPlaneWaves{T}, k0::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real}
    p = basis.taylor_degree
    M = length(pts)
    N = basis.dim
    coeffs = Array{T,3}(undef, p + 1, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        vx = cos(basis.angles[c])
        vy = sin(basis.angles[c])
        parx = basis.parity_x[c]
        pary = basis.parity_y[c]
        @inbounds for j = 1:M
            _real_plane_wave_taylor_coefficients!(coeffs, j, c, T(pts[j][1]), T(pts[j][2]), vx, vy, parx, pary, k0, p)
        end
    end
    return RealPlaneWaveTaylorCache(k0, p, coeffs)
end

"""
    basis_cache(basis::RealPlaneWaves, k0, pts; multithreaded::Bool = true) → cache::RealPlaneWaveTaylorCache

Construct the Taylor cache associated with a [`RealPlaneWaves`](@ref) basis.

## Arguments
* `basis`: Basis for which the cache is constructed.
* `k0`: Reference wavenumber of the Taylor expansion.
* `pts`: Points at which the basis is represented.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether cache construction is multithreaded.

## Returns
* `cache`: [`RealPlaneWaveTaylorCache`](@ref) associated with `basis`.
"""
@inline function basis_cache(basis::RealPlaneWaves, k0, pts; multithreaded::Bool = true)
    return RealPlaneWaveTaylorCache(basis, k0, pts; multithreaded)
end

"""
    basis_fun(cache::RealPlaneWaveTaylorCache{T}, i::Int, k::T) where {T<:Real} → out::Vector{T}

Evaluate one cached real plane-wave basis function at `k`.

## Arguments
* `cache`: [`RealPlaneWaveTaylorCache`](@ref).
* `i`: Basis-function index.
* `k`: Evaluation wavenumber.

## Returns
* `out`: Values of basis function `i` at the cached spatial points.
"""
@inline function basis_fun(cache::RealPlaneWaveTaylorCache{T}, i::Int, k::T) where {T<:Real}
    M = size(cache.coeffs, 2)
    δ = k - cache.k0
    out = Vector{T}(undef, M)
    @inbounds @simd for j = 1:M
        out[j] = _horner(cache.coeffs, j, i, δ)
    end
    return filter_matrix!(out)
end

"""
    basis_fun(cache::RealPlaneWaveTaylorCache{T}, indices::AbstractArray, k::T; multithreaded::Bool = true) where {T<:Real} → B::Matrix{T}

Evaluate selected cached real plane-wave basis functions at `k`.

## Arguments
* `cache`: [`RealPlaneWaveTaylorCache`](@ref).
* `indices`: Basis-function indices to evaluate.
* `k`: Evaluation wavenumber.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether evaluation is multithreaded across columns.

## Returns
* `B`: Basis matrix with one column for each requested basis-function index.
"""
function basis_fun(cache::RealPlaneWaveTaylorCache{T}, indices::AbstractArray, k::T; multithreaded::Bool = true) where {T<:Real}
    M = size(cache.coeffs, 2)
    N = length(indices)
    δ = k - cache.k0
    B = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        idx = indices[c]
        @inbounds @simd for j = 1:M
            B[j, c] = _horner(cache.coeffs, j, idx, δ)
        end
    end
    return filter_matrix!(B)
end

"""
    basis_matrix(cache::RealPlaneWaveTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real} → B::Matrix{T}

Evaluate the complete cached real plane-wave basis at `k`.

## Arguments
* `cache`: [`RealPlaneWaveTaylorCache`](@ref).
* `k`: Evaluation wavenumber.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether evaluation is multithreaded across columns.

## Returns
* `B`: Complete basis matrix evaluated at `k`.
"""
function basis_matrix(cache::RealPlaneWaveTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real}
    M = size(cache.coeffs, 2)
    N = size(cache.coeffs, 3)
    δ = k - cache.k0
    B = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        @inbounds @simd for j = 1:M
            B[j, c] = _horner(cache.coeffs, j, c, δ)
        end
    end
    return filter_matrix!(B)
end

"""
    dk_fun(cache::RealPlaneWaveTaylorCache{T}, i::Int, k::T) where {T<:Real} → dk::Vector{T}

Evaluate the wavenumber derivative of one cached real plane-wave basis
function.

## Arguments
* `cache`: [`RealPlaneWaveTaylorCache`](@ref).
* `i`: Basis-function index.
* `k`: Evaluation wavenumber.

## Returns
* `dk`: Wavenumber derivative of basis function `i`.
"""
@inline function dk_fun(cache::RealPlaneWaveTaylorCache{T}, i::Int, k::T) where {T<:Real}
    M = size(cache.coeffs, 2)
    δ = k - cache.k0
    dk = Vector{T}(undef, M)
    @inbounds @simd for j = 1:M
        _, dk[j] = _horner_with_derivative(cache.coeffs, j, i, δ)
    end
    return filter_matrix!(dk)
end

"""
    dk_fun(cache::RealPlaneWaveTaylorCache{T}, indices::AbstractArray, k::T; multithreaded::Bool = true) where {T<:Real} → dB_dk::Matrix{T}

Evaluate the wavenumber derivatives of selected cached real plane-wave basis
functions.

## Arguments
* `cache`: [`RealPlaneWaveTaylorCache`](@ref).
* `indices`: Basis-function indices to evaluate.
* `k`: Evaluation wavenumber.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether evaluation is multithreaded across columns.

## Returns
* `dB_dk`: Wavenumber derivative matrix for the requested basis functions.
"""
function dk_fun(cache::RealPlaneWaveTaylorCache{T}, indices::AbstractArray, k::T; multithreaded::Bool = true) where {T<:Real}
    M = size(cache.coeffs, 2)
    N = length(indices)
    δ = k - cache.k0
    dB_dk = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        idx = indices[c]
        @inbounds @simd for j = 1:M
            _, dB_dk[j, c] = _horner_with_derivative(cache.coeffs, j, idx, δ)
        end
    end
    return filter_matrix!(dB_dk)
end

"""
    dk_matrix(cache::RealPlaneWaveTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real} → dB_dk::Matrix{T}

Evaluate the wavenumber derivative of the complete cached real plane-wave
basis.

## Arguments
* `cache`: [`RealPlaneWaveTaylorCache`](@ref).
* `k`: Evaluation wavenumber.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether evaluation is multithreaded across columns.

## Returns
* `dB_dk`: Wavenumber derivative of the complete basis matrix.
"""
function dk_matrix(cache::RealPlaneWaveTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real}
    M = size(cache.coeffs, 2)
    N = size(cache.coeffs, 3)
    δ = k - cache.k0
    dB_dk = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        @inbounds @simd for j = 1:M
            _, dB_dk[j, c] = _horner_with_derivative(cache.coeffs, j, c, δ)
        end
    end
    return filter_matrix!(dB_dk)
end

"""
    basis_and_dk_matrices(cache::RealPlaneWaveTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real} → (B, dB_dk)

Evaluate the complete cached real plane-wave basis and its wavenumber
derivative simultaneously.

## Arguments
* `cache`: [`RealPlaneWaveTaylorCache`](@ref).
* `k`: Evaluation wavenumber.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether evaluation is multithreaded across columns.

## Returns
* `B`: Complete basis matrix evaluated at `k`.
* `dB_dk`: Wavenumber derivative of the complete basis matrix evaluated at `k`.
"""
function basis_and_dk_matrices(cache::RealPlaneWaveTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real}
    M = size(cache.coeffs, 2)
    N = size(cache.coeffs, 3)
    δ = k - cache.k0
    B = Matrix{T}(undef, M, N)
    dB_dk = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        @inbounds @simd for j = 1:M
            B[j, c], dB_dk[j, c] = _horner_with_derivative(cache.coeffs, j, c, δ)
        end
    end
    return filter_matrix!(B), filter_matrix!(dB_dk)
end