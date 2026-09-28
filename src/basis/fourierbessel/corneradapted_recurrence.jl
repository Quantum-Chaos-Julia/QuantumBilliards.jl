################################################################################
# ANALYTIC TAYLOR REPRESENTATION OF CORNER-ADAPTED FOURIER-BESSEL BASES
#
# For a corner-adapted Fourier-Bessel radial function
#
#                         y(k) = J_m(kr),
#
# expand about a reference wavenumber k₀,
#
#                  y(k₀ + δ) = Σₙ₌₀ᵖ aₙ δⁿ.
#
# Since J_m satisfies Bessel's equation,
#
#                  k²y'' + ky' + (k²r² - m²)y = 0,
#
# the normalized Taylor coefficients satisfy
#
#     a₀ = J_m(k₀r),
#
#     a₁ = rJ'_m(k₀r)
#        = (m/k₀)J_m(k₀r) - rJ_{m+1}(k₀r),
#
# and, for n ≥ 0,
#
#     aₙ₊₂ =
#       -[k₀(n+1)(2n+1)aₙ₊₁
#         +(n²+(k₀r)²-m²)aₙ
#         +2k₀r²aₙ₋₁
#         +r²aₙ₋₂]
#        /[k₀²(n+2)(n+1)].
#
# Only J_m(k₀r) and J_{m+1}(k₀r) are evaluated directly. Values and
# wavenumber derivatives at nearby k are then obtained by Horner evaluation.
#
# The cache always stores the radial coordinates, Taylor coefficients and
# angular sine factors required by accelerated basis solvers. Additional
# angular data required for Cartesian spatial gradients are constructed only
# when `gradients = true`.
################################################################################

"""
    CornerAdaptedTaylorCache{T<:Real}

Cache for the analytic Taylor representation of a
[`CornerAdaptedFourierBessel`](@ref) basis about a reference wavenumber.

## Description
For each spatial point and basis function, the cache stores the normalized
Taylor coefficients of the radial factor `J_m(kr)` about `k0` together with
the angular factor `sin(mφ)`.

The coefficient tensor is indexed as `coeffs[l + 1, j, c]`, where `l` is the
Taylor order, `j` the spatial point and `c` the basis-function index.

The geometric quantities required for Cartesian spatial gradients are optional
and are constructed only when the cache is created with `gradients = true`.

## Attributes
* `k0`: Reference wavenumber of the Taylor expansion.
* `degree`: Degree of the Taylor expansion.
* `r`: Radial coordinates of the evaluation points.
* `coeffs`: Normalized Taylor coefficients of `J_m(kr)`.
* `sin_mφ`: Angular factors `sin(mφ)` for every point and basis function.
* `sinφ`: Values of `sin(φ)` when spatial-gradient data are requested, otherwise `nothing`.
* `cosφ`: Values of `cos(φ)` when spatial-gradient data are requested, otherwise `nothing`.
* `cos_mφ`: Angular factors `cos(mφ)` when spatial-gradient data are requested, otherwise `nothing`.
"""
struct CornerAdaptedTaylorCache{T<:Real} <: BasisCache
    k0::T
    degree::Int
    r::Vector{T}
    coeffs::Array{T,3}
    sin_mφ::Matrix{T}
    sinφ::Union{Nothing,Vector{T}}
    cosφ::Union{Nothing,Vector{T}}
    cos_mφ::Union{Nothing,Matrix{T}}
end

################################################################################
# TAYLOR COEFFICIENTS
################################################################################

# Construct the normalized Taylor coefficients of J_m(kr) about k = k₀.
@inline function _besselj_taylor_coefficients!(C::Array{T,3}, j::Int, c::Int, m::T, k0::T, r::T, p::Int) where {T<:Real}
    z = k0 * r
    jm = T(Bessels.besselj(m, z))
    C[1, j, c] = jm
    p == 0 && return nothing
    jp = T(Bessels.besselj(m + one(T), z))
    C[2, j, c] = (m / k0) * jm - r * jp
    k02 = k0 * k0
    r2 = r * r
    @inbounds for n = 0:p - 2
        an = C[n + 1, j, c]
        anp1 = C[n + 2, j, c]
        anm1 = n >= 1 ? C[n, j, c] : zero(T)
        anm2 = n >= 2 ? C[n - 1, j, c] : zero(T)
        num = k0 * (n + 1) * (2n + 1) * anp1 + (n * n + k02 * r2 - m * m) * an + 2k0 * r2 * anm1 + r2 * anm2
        C[n + 3, j, c] = -num / (k02 * (n + 2) * (n + 1))
    end
    return nothing
end

################################################################################
# CACHE CONSTRUCTION
################################################################################

"""
    CornerAdaptedTaylorCache(basis::CornerAdaptedFourierBessel{T}, k0::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → cache::CornerAdaptedTaylorCache{T}

Construct an analytic Taylor representation of a
[`CornerAdaptedFourierBessel`](@ref) basis about the reference wavenumber `k0`.

## Description
The Taylor degree is taken from `basis.taylor_degree`. For each spatial point
and basis function, the normalized Taylor coefficients of `J_m(kr)` are
constructed analytically from Bessel's differential equation. Only the first
two coefficients require direct Bessel-function evaluations.

The angular factors `sin(mφ)` required for basis evaluation are always cached.
If `gradients = true`, the additional factors `sin(φ)`, `cos(φ)` and
`cos(mφ)` required for Cartesian spatial gradients are also stored.

## Arguments
* `basis`: [`CornerAdaptedFourierBessel`](@ref) basis to expand.
* `k0`: Reference wavenumber of the Taylor expansion.
* `pts`: Spatial points at which the basis will be evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded cache construction across basis functions.

## Returns
* `cache`: [`CornerAdaptedTaylorCache`](@ref) containing the analytic Taylor representation of the basis.
"""
function CornerAdaptedTaylorCache(basis::CornerAdaptedFourierBessel{T}, k0::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real}
    p = basis.taylor_degree
    M = length(pts)
    N = basis.dim
    r = Vector{T}(undef, M)
    φ = Vector{T}(undef, M)
    coeffs = Array{T,3}(undef, p + 1, M, N)
    sin_mφ = Matrix{T}(undef, M, N)
    sinφ = basis.gradients ? Vector{T}(undef, M) : nothing
    cosφ = basis.gradients ? Vector{T}(undef, M) : nothing
    cos_mφ = basis.gradients ? Matrix{T}(undef, M, N) : nothing
    _polar_coords!(r, φ, basis.cs.local_map, pts, basis.rotation_angle_discontinuity)
    if basis.gradients
        @inbounds @simd for j = 1:M
            sinφ[j], cosφ[j] = sincos(φ[j])
        end
    end
    @use_threads multithreading=multithreaded for c = 1:N
        m = basis.nu * c
        @inbounds for j = 1:M
            if basis.gradients
                sin_mφ[j, c], cos_mφ[j, c] = sincos(m * φ[j])
            else
                sin_mφ[j, c] = sin(m * φ[j])
            end
            _besselj_taylor_coefficients!(coeffs, j, c, m, k0, r[j], p)
        end
    end
    return CornerAdaptedTaylorCache(k0, p, r, coeffs, sin_mφ, sinφ, cosφ, cos_mφ)
end

"""
    basis_cache(basis::CornerAdaptedFourierBessel, k0, pts; multithreaded::Bool = true) → cache::CornerAdaptedTaylorCache

Construct the Taylor cache associated with a [`CornerAdaptedFourierBessel`](@ref) basis.

## Arguments
* `basis`: Basis for which the cache is constructed.
* `k0`: Reference wavenumber of the Taylor expansion.
* `pts`: Points at which the basis is represented.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether cache construction is multithreaded.

## Returns
* `cache`: [`CornerAdaptedTaylorCache`](@ref) associated with `basis`.
"""
@inline function basis_cache(basis::CornerAdaptedFourierBessel, k0, pts; multithreaded::Bool = true)
    return CornerAdaptedTaylorCache(basis, k0, pts; multithreaded)
end


################################################################################
# BASIS MATRIX
################################################################################

"""
    basis_matrix(cache::CornerAdaptedTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real} → B::Matrix{T}

Evaluate a cached corner-adapted Fourier-Bessel basis matrix at wavenumber `k`.

## Description
The radial Taylor series stored in `cache` is evaluated at
`δ = k - cache.k0` using Horner's rule and multiplied by the cached angular
factor `sin(mφ)`.

## Arguments
* `cache`: [`CornerAdaptedTaylorCache`](@ref) containing the Taylor coefficients and angular factors.
* `k`: Wavenumber at which the basis is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded evaluation across basis functions.

## Returns
* `B`: Basis matrix of size `(M, N)`.
"""
function basis_matrix(cache::CornerAdaptedTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real}
    M = size(cache.coeffs, 2)
    N = size(cache.coeffs, 3)
    δ = k - cache.k0
    B = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        @inbounds @simd for j = 1:M
            B[j, c] = _horner(cache.coeffs, j, c, δ) * cache.sin_mφ[j, c]
        end
    end
    return filter_matrix!(B)
end

################################################################################
# WAVENUMBER DERIVATIVE
################################################################################

"""
    dk_matrix(cache::CornerAdaptedTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real} → dB_dk::Matrix{T}

Evaluate the wavenumber derivative of a cached corner-adapted Fourier-Bessel
basis matrix at wavenumber `k`.

## Description
The radial Taylor series stored in `cache` is differentiated and evaluated at
`δ = k - cache.k0` using simultaneous Horner evaluation. The resulting radial
derivative is multiplied by the cached angular factor `sin(mφ)`.

## Arguments
* `cache`: [`CornerAdaptedTaylorCache`](@ref) containing the Taylor coefficients and angular factors.
* `k`: Wavenumber at which the derivative is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded evaluation across basis functions.

## Returns
* `dB_dk`: Wavenumber derivative of the basis matrix, of size `(M, N)`.
"""
function dk_matrix(cache::CornerAdaptedTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real}
    M = size(cache.coeffs, 2)
    N = size(cache.coeffs, 3)
    δ = k - cache.k0
    dB_dk = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        @inbounds @simd for j = 1:M
            _, dy = _horner_with_derivative(cache.coeffs, j, c, δ)
            dB_dk[j, c] = dy * cache.sin_mφ[j, c]
        end
    end
    return filter_matrix!(dB_dk)
end

################################################################################
# BASIS + WAVENUMBER DERIVATIVE
################################################################################

"""
    basis_and_dk_matrices(cache::CornerAdaptedTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real} → (B, dB_dk)::Tuple{Matrix{T},Matrix{T}}

Evaluate a cached corner-adapted Fourier-Bessel basis matrix and its wavenumber
derivative simultaneously at wavenumber `k`.

## Description
For each point and basis function, the stored radial Taylor series and its
first derivative are evaluated simultaneously using Horner's rule. The common
angular factor `sin(mφ)` is then applied to both quantities.

This is the preferred evaluation path when both the basis matrix and its
wavenumber derivative are required, since both are obtained in a single
traversal of the Taylor coefficients.

## Arguments
* `cache`: [`CornerAdaptedTaylorCache`](@ref) containing the Taylor coefficients and angular factors.
* `k`: Wavenumber at which the basis and its derivative are evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded evaluation across basis functions.

## Returns
* `(B, dB_dk)`: Basis matrix and its wavenumber derivative, each of size `(M, N)`.
"""
function basis_and_dk_matrices(cache::CornerAdaptedTaylorCache{T}, k::T; multithreaded::Bool = true) where {T<:Real}
    M = size(cache.coeffs, 2)
    N = size(cache.coeffs, 3)
    δ = k - cache.k0
    B = Matrix{T}(undef, M, N)
    dB_dk = Matrix{T}(undef, M, N)
    @use_threads multithreading=multithreaded for c = 1:N
        @inbounds @simd for j = 1:M
            y, dy = _horner_with_derivative(cache.coeffs, j, c, δ)
            s = cache.sin_mφ[j, c]
            B[j, c] = y * s
            dB_dk[j, c] = dy * s
        end
    end
    return filter_matrix!(B), filter_matrix!(dB_dk)
end

################################################################################
# SPATIAL GRADIENTS
################################################################################

# Verify that a Taylor cache contains the optional spatial-gradient data.
@inline function _require_gradient_cache(cache::CornerAdaptedTaylorCache)
    cache.sinφ === nothing && throw(ArgumentError("spatial-gradient data are not available; construct the cache from a basis with gradients = true"))
    return nothing
end

"""
    gradient_matrices(cache::CornerAdaptedTaylorCache{T}, basis::CornerAdaptedFourierBessel{T}, k::T; multithreaded::Bool = true) where {T<:Real} → (dB_dx, dB_dy)::Tuple{Matrix{T},Matrix{T}}

Evaluate the Cartesian spatial derivatives of a cached corner-adapted
Fourier-Bessel basis at wavenumber `k`.

## Description
The radial value `J_m(kr)` and its wavenumber derivative are evaluated
simultaneously from the cached Taylor series. For `r > 0`, the radial spatial
derivative satisfies \$\\partial_r J_m(kr) = (k/r)\\partial_k J_m(kr)\$.
Together with the cached angular derivative, this is transformed into the
Cartesian `x` and `y` derivatives.

The cache must have been constructed with `gradients = true`.

## Arguments
* `cache`: [`CornerAdaptedTaylorCache`](@ref) containing the Taylor coefficients and spatial-gradient data.
* `basis`: [`CornerAdaptedFourierBessel`](@ref) basis associated with the cache.
* `k`: Wavenumber at which the spatial gradients are evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded evaluation across basis functions.

## Returns
* `(dB_dx, dB_dy)`: Cartesian `x` and `y` derivatives of the basis matrix, each of size `(M, N)`.
"""
function gradient_matrices(cache::CornerAdaptedTaylorCache{T}, basis::CornerAdaptedFourierBessel{T}, k::T; multithreaded::Bool = true) where {T<:Real}
    _require_gradient_cache(cache)
    M = size(cache.coeffs, 2)
    N = size(cache.coeffs, 3)
    N == basis.dim || throw(DimensionMismatch("cache and basis dimensions do not agree"))
    δ = k - cache.k0
    dB_dx = Matrix{T}(undef, M, N)
    dB_dy = Matrix{T}(undef, M, N)
    sinφ = cache.sinφ::Vector{T}
    cosφ = cache.cosφ::Vector{T}
    cos_mφ = cache.cos_mφ::Matrix{T}
    @use_threads multithreading=multithreaded for c = 1:N
        m = basis.nu * c
        @inbounds for j = 1:M
            r = cache.r[j]
            y, dy = _horner_with_derivative(cache.coeffs, j, c, δ)
            if r == zero(T)
                dB_dx[j, c] = zero(T)
                dB_dy[j, c] = zero(T)
            else
                invr = inv(r)
                fr = k * dy * invr * cache.sin_mφ[j, c]
                fφ = m * y * cos_mφ[j, c]
                dB_dx[j, c] = cosφ[j] * fr - sinφ[j] * invr * fφ
                dB_dy[j, c] = sinφ[j] * fr + cosφ[j] * invr * fφ
            end
        end
    end
    return filter_matrix!(dB_dx), filter_matrix!(dB_dy)
end

################################################################################
# BASIS + SPATIAL GRADIENTS
################################################################################

"""
    basis_and_gradient_matrices(cache::CornerAdaptedTaylorCache{T}, basis::CornerAdaptedFourierBessel{T}, k::T; multithreaded::Bool = true) where {T<:Real} → (B, dB_dx, dB_dy)::Tuple{Matrix{T},Matrix{T},Matrix{T}}

Evaluate a cached corner-adapted Fourier-Bessel basis matrix and its Cartesian
spatial derivatives simultaneously at wavenumber `k`.

## Description
The radial Taylor series and its first wavenumber derivative are evaluated
simultaneously using Horner's rule. The cached angular factors are then used to
construct both the basis matrix and its Cartesian spatial derivatives without
additional Bessel-function evaluations.

For `r > 0`, the radial spatial derivative is obtained from
\$\\partial_r J_m(kr) = (k/r)\\partial_k J_m(kr)\$.

The cache must have been constructed with `gradients = true`.

## Arguments
* `cache`: [`CornerAdaptedTaylorCache`](@ref) containing the Taylor coefficients and spatial-gradient data.
* `basis`: [`CornerAdaptedFourierBessel`](@ref) basis associated with the cache.
* `k`: Wavenumber at which the basis and spatial gradients are evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded evaluation across basis functions.

## Returns
* `(B, dB_dx, dB_dy)`: Basis matrix and its Cartesian `x` and `y` derivatives, each of size `(M, N)`.
"""
function basis_and_gradient_matrices(cache::CornerAdaptedTaylorCache{T}, basis::CornerAdaptedFourierBessel{T}, k::T; multithreaded::Bool = true) where {T<:Real}
    _require_gradient_cache(cache)
    M = size(cache.coeffs, 2)
    N = size(cache.coeffs, 3)
    N == basis.dim || throw(DimensionMismatch("cache and basis dimensions do not agree"))
    δ = k - cache.k0
    B = Matrix{T}(undef, M, N)
    dB_dx = Matrix{T}(undef, M, N)
    dB_dy = Matrix{T}(undef, M, N)
    sinφ = cache.sinφ::Vector{T}
    cosφ = cache.cosφ::Vector{T}
    cos_mφ = cache.cos_mφ::Matrix{T}
    @use_threads multithreading=multithreaded for c = 1:N
        m = basis.nu * c
        @inbounds for j = 1:M
            r = cache.r[j]
            y, dy = _horner_with_derivative(cache.coeffs, j, c, δ)
            s = cache.sin_mφ[j, c]
            B[j, c] = y * s
            if r == zero(T)
                dB_dx[j, c] = zero(T)
                dB_dy[j, c] = zero(T)
            else
                invr = inv(r)
                fr = k * dy * invr * s
                fφ = m * y * cos_mφ[j, c]
                dB_dx[j, c] = cosφ[j] * fr - sinφ[j] * invr * fφ
                dB_dy[j, c] = sinφ[j] * fr + cosφ[j] * invr * fφ
            end
        end
    end
    return filter_matrix!(B), filter_matrix!(dB_dx), filter_matrix!(dB_dy)
end