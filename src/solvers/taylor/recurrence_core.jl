################################################################################
# ANALYTIC TAYLOR REPRESENTATION OF BIM FREDHOLM OPERATORS
#
# This file provides the shared analytic wavenumber-expansion backend for
# CORK, Beyn, and expanded-BIM eigensolvers.
#
# For the Fredholm nonlinear eigenvalue problem
#
#                              A(k)u = 0,
#
# we construct normalized Taylor coefficients about k₀,
#
#                 A(k₀+δ) = Σₗ₌₀ᵖ Aₗδˡ + O(δᵖ⁺¹),
#                 Aₗ = A⁽ˡ⁾(k₀)/l!.
#
# The coefficients are generated analytically from Bessel/Hankel differential
# equations. Special functions are evaluated only for the initial recurrence
# values at k₀; all higher coefficients require scalar arithmetic.
#
# The same representation is consumed in three ways:
#
#   CORK: Taylor coefficients are converted to Chebyshev coefficients on a
#         guarded real interval.
#
#   Beyn: the Taylor polynomial is evaluated by Horner's rule at complex
#         contour nodes.
#
#   EBIM: the polynomial and its first two derivatives are evaluated
#         simultaneously by Horner's rule at nearby real wavenumbers.
#
# Supported Fredholm operators are
#
#   DLP:        A(k) = I - D(k),
#   CFIE:       A(k) = I - [D(k) + ikS(k)],
#   Composite:  componentwise DLP/CFIE with smooth cross-component blocks.
################################################################################

# Evaluate a Taylor polynomial by Horner's rule at a given point δ.
@inline function _horner(a::Vector{ComplexF64}, δ::Number)
    v::ComplexF64 = a[end]
    @inbounds for l = length(a) - 1:-1:1
        v = v * δ + a[l]
    end
    return v
end

# Evaluate a column j of Taylor coefficients C[:, j] by Horner's rule at a given point δ. 
#This avoids constructing a SubArray view in the symmetry-reduced path.
@inline function _horner(C::Matrix{ComplexF64}, j::Int, δ::Number)
    v::ComplexF64 = C[end, j]
    @inbounds for l = size(C, 1) - 1:-1:1
        v = v * δ + C[l, j]
    end
    return v
end

# Evaluate a Taylor polynomial and its first derivative simultaneously by
# Horner's rule at a point δ.
@inline function _horner_with_derivative(a::Vector{ComplexF64}, δ::Number)
    v::ComplexF64 = a[end]
    dv::ComplexF64 = zero(ComplexF64)
    @inbounds for l = length(a) - 1:-1:1
        dv = dv * δ + v
        v = v * δ + a[l]
    end
    return v, dv
end

# Evaluate a column j of Taylor coefficients C[:, j] and its first derivative
# simultaneously by Horner's rule at a point δ.
@inline function _horner_with_derivative(C::Matrix{ComplexF64}, j::Int, δ::Number)
    v::ComplexF64 = C[end, j]
    dv::ComplexF64 = zero(ComplexF64)
    @inbounds for l = size(C, 1) - 1:-1:1
        dv = dv * δ + v
        v = v * δ + C[l, j]
    end
    return v, dv
end

# Evaluate the Taylor coefficients C[:, j, c] and their first derivative
# simultaneously by Horner's rule at a real point δ.
@inline function _horner_with_derivative(C::Array{T,3}, j::Int, c::Int, δ::T) where {T<:Real}
    v::T = C[end, j, c]
    dv::T = zero(T)
    @inbounds for l = size(C, 1) - 1:-1:1
        dv = muladd(dv, δ, v)
        v = muladd(v, δ, C[l, j, c])
    end
    return v, dv
end

# Evaluate a Taylor polynomial and its first two derivatives simultaneously by Horner's rule
# at a point δ.
@inline function _horner_with_12_derivatives(a::Vector{ComplexF64}, δ::Number)
    v::ComplexF64 = a[end]
    dv::ComplexF64 = zero(ComplexF64)
    ddv::ComplexF64 = zero(ComplexF64)
    @inbounds for l = length(a) - 1:-1:1
        ddv = ddv * δ + 2 * dv
        dv = dv * δ + v
        v = v * δ + a[l]
    end
    return v, dv, ddv
end

# Evaluate a column j of Taylor coefficients C[:, j] and its first two derivatives
# simultaneously by Horner's rule at a point δ. This avoids constructing a SubArray view
# in the symmetry-reduced path.
@inline function _horner_with_12_derivatives(C::Matrix{ComplexF64}, j::Int, δ::Number)
    v::ComplexF64 = C[end, j]
    dv::ComplexF64 = zero(ComplexF64)
    ddv::ComplexF64 = zero(ComplexF64)
    @inbounds for l = size(C, 1) - 1:-1:1
        ddv = ddv * δ + 2 * dv
        dv = dv * δ + v
        v = v * δ + C[l, j]
    end
    return v, dv, ddv
end

# Evaluate the Taylor coefficients C[:, i, j] and their first two derivatives
# simultaneously by Horner's rule at a real point δ.
@inline function _horner_with_12_derivatives(C::Array{ComplexF64,3}, i::Int, j::Int, δ::Real)
    v::ComplexF64 = C[end, i, j]
    dv::ComplexF64 = zero(ComplexF64)
    ddv::ComplexF64 = zero(ComplexF64)
    @inbounds for l = size(C, 1) - 1:-1:1
        ddv = ddv * δ + 2 * dv
        dv = dv * δ + v
        v = v * δ + C[l, i, j]
    end
    return v, dv, ddv
end

"""
    power_to_cheb(p::Int, Δ::Float64) -> Matrix{Float64}

Construct the transformation from Taylor coefficients in
`δ=k-k₀` to Chebyshev coefficients in the scaled coordinate `t=δ/Δ`.
The matrix is defined by

    Δˡtˡ = Σⱼ C[j+1,l+1]Tⱼ(t),

so if

    F(k₀+δ) = Σₗ Fₗδˡ,

then

    F(k₀+Δt) = Σⱼ BⱼTⱼ(t),    Bⱼ = Σₗ C[j+1,l+1]Fₗ.

The unscaled power-to-Chebyshev columns are generated recursively from

    tT₀ = T₁,
    tTⱼ = (Tⱼ₋₁+Tⱼ₊₁)/2,

and column `l+1` is subsequently scaled by `Δˡ`.

## Arguments
- `p::Int`: Maximum polynomial degree.
- `Δ::Float64`: Half-width used to scale the Taylor variable according to
  `δ=Δt`.

## Returns
- `Matrix{Float64}`: The `(p+1)×(p+1)` transformation matrix `C` mapping
  normalized Taylor/power coefficients to Chebyshev coefficients.
"""
function power_to_cheb(p::Int, Δ::Float64)::Matrix{Float64}
    C = zeros(Float64, p + 1, p + 1); C[1,1] = 1.0
    for l = 0:p-1
        C[2,l+2] += C[1,l+1]
        @inbounds for j = 1:l
            c = C[j+1,l+1] / 2
            C[j,l+2] += c; C[j+2,l+2] += c
        end
    end
    s = 1.0
    @inbounds for l = 0:p
        C[:,l+1] .*= s; s *= Δ
    end
    return C
end

################################################################################
# RADIAL TAYLOR RECURRENCES
# Bessel/Hankel factors are obtained from their differential equations directly. 
# For f(k)=kZ₁(kr), where Zν is Jν or Hν⁽¹⁾, the order-one Bessel equation gives
#
#                         f'' - f'/k + r²f = 0.
#
# With f(k+δ)=Σₙfₙδⁿ, fₙ=f⁽ⁿ⁾(k)/n!, coefficient matching gives
#
#   fₙ₊₂ = -[(n+1)(n-1)fₙ₊₁ + kr²fₙ + r²fₙ₋₁]/[k(n+2)(n+1)],
#
# where f₋₁=0. Since d[zZ₁(z)]/dz=zZ₀(z),
#
#                     f₀=kZ₁(kr),    f₁=krZ₀(kr).
#
# Similarly, y(k)=Z₀(kr) satisfies y''+y'/k+r²y=0 and hence
#
#   yₙ₊₂ = -[(n+1)²yₙ₊₁ + kr²yₙ + r²yₙ₋₁]/[k(n+2)(n+1)],
#
# with y₋₁=0, y₀=Z₀(kr), y₁=-rZ₁(kr). Finally, if f(k)=ky(k),
#
#                     f₀=ky₀,       fₙ=kyₙ+yₙ₋₁, n≥1.
################################################################################

"""
    radial1_taylor!(f::Vector{ComplexF64}, k::Float64, r2::Float64, p::Int, f0::Union{ComplexF64,Float64}, f1::Union{ComplexF64,Float64}) -> Vector{ComplexF64}

Compute the normalized Taylor coefficients `f[n+1]=f⁽ⁿ⁾(k)/n!` of
`f(k)=kZ₁(kr)`, where `Z₁` may be `J₁` or `H₁⁽¹⁾`. The Bessel equation gives

f'' - f'/k + r²f = 0,

and therefore

fₙ₊₂ = -[(n+1)(n-1)fₙ₊₁ + kr²fₙ + r²fₙ₋₁]/[k(n+2)(n+1)],

with `f₋₁=0`. For Bessel/Hankel functions the initial values are
`f₀=kZ₁(kr)` and `f₁=krZ₀(kr)`, following from
`d[zZ₁(z)]/dz=zZ₀(z)`. After the initial special-function evaluations, all
coefficients through degree `p` require only `O(p)` scalar arithmetic.

## Arguments
- `f::Vector{ComplexF64}`: Preallocated output vector of length at least
  `p+1`.
- `k::Float64`: Expansion point in wavenumber.
- `r2::Float64`: Squared radial distance `r²`.
- `p::Int`: Maximum Taylor degree.
- `f0::Union{ComplexF64,Float64}`: Zeroth normalized Taylor coefficient
  `f₀=kZ₁(kr)`.
- `f1::Union{ComplexF64,Float64}`: First normalized Taylor coefficient
  `f₁=krZ₀(kr)`.

## Returns
- `Vector{ComplexF64}`: The mutated vector `f` containing
  `f[n+1]=f⁽ⁿ⁾(k)/n!` for `n=0,...,p`.
"""
@inline function radial1_taylor!(f::Vector{ComplexF64}, k::Float64, r2::Float64, p::Int, f0::Union{ComplexF64,Float64}, f1::Union{ComplexF64,Float64})::Vector{ComplexF64}
    f[1]=f0; p==0 && return f; f[2]=f1
    kinv=inv(k); r2k=r2*kinv
    @inbounds for n=0:p-2
        d=inv(Float64((n+1)*(n+2)))
        fm1=n==0 ? 0.0im : f[n]
        f[n+3]=-(n-1)*kinv/(n+2)*f[n+2]-r2*d*f[n+1]-r2k*d*fm1
    end
    return f
end

"""
    radial0_taylor!(y::Vector{ComplexF64}, k::Float64, r2::Float64, p::Int, y0::Union{ComplexF64,Float64}, y1::Union{ComplexF64,Float64}) -> Vector{ComplexF64}

Compute the normalized Taylor coefficients `y[n+1]=y⁽ⁿ⁾(k)/n!` of
`y(k)=Z₀(kr)`, where `Z₀` may be `J₀` or `H₀⁽¹⁾`. The order-zero Bessel
equation implies

y'' + y'/k + r²y = 0,

so

yₙ₊₂ = -[(n+1)²yₙ₊₁ + kr²yₙ + r²yₙ₋₁]/[k(n+2)(n+1)],

with `y₋₁=0`. The natural initial values are `y₀=Z₀(kr)` and
`y₁=-rZ₁(kr)` because `Z₀'(z)=-Z₁(z)`.

## Arguments
- `y::Vector{ComplexF64}`: Preallocated output vector of length at least
  `p+1`.
- `k::Float64`: Expansion point in wavenumber.
- `r2::Float64`: Squared radial distance `r²`.
- `p::Int`: Maximum Taylor degree.
- `y0::Union{ComplexF64,Float64}`: Zeroth normalized Taylor coefficient
  `y₀=Z₀(kr)`.
- `y1::Union{ComplexF64,Float64}`: First normalized Taylor coefficient
  `y₁=-rZ₁(kr)`.

## Returns
- `Vector{ComplexF64}`: The mutated vector `y` containing
  `y[n+1]=y⁽ⁿ⁾(k)/n!` for `n=0,...,p`.
"""
@inline function radial0_taylor!(y::Vector{ComplexF64}, k::Float64, r2::Float64, p::Int, y0::Union{ComplexF64,Float64}, y1::Union{ComplexF64,Float64})::Vector{ComplexF64}
    y[1]=y0; p==0 && return y; y[2]=y1
    kinv=inv(k); r2k=r2*kinv
    @inbounds for n=0:p-2
        d=inv(Float64((n+1)*(n+2)))
        ym1=n==0 ? 0.0im : y[n]
        y[n+3]=-(n+1)*kinv/(n+2)*y[n+2]-r2*d*y[n+1]-r2k*d*ym1
    end
    return y
end

"""
    radial0_k_taylor!(f::Vector{ComplexF64}, y::Vector{ComplexF64}, k::Float64, r2::Float64, p::Int, y0::Union{ComplexF64,Float64}, y1::Union{ComplexF64,Float64}) -> Vector{ComplexF64}

Compute normalized Taylor coefficients of `f(k)=kZ₀(kr)`. First
[`radial0_taylor!`](@ref) constructs `y(k+δ)=Σₙyₙδⁿ`. Since

(k+δ)y(k+δ) = ky₀ + Σₙ₌₁∞(kyₙ+yₙ₋₁)δⁿ,

the desired coefficients are `f₀=ky₀` and `fₙ=kyₙ+yₙ₋₁` for `n≥1`.
This supplies the `kJ₀(kr)` and `kH₀⁽¹⁾(kr)` series required by the CFIE.

## Arguments
- `f::Vector{ComplexF64}`: Preallocated output vector for the Taylor
  coefficients of `kZ₀(kr)`.
- `y::Vector{ComplexF64}`: Preallocated scratch vector used for the Taylor
  coefficients of `Z₀(kr)`.
- `k::Float64`: Expansion point in wavenumber.
- `r2::Float64`: Squared radial distance `r²`.
- `p::Int`: Maximum Taylor degree.
- `y0::Union{ComplexF64,Float64}`: Zeroth coefficient `Z₀(kr)`.
- `y1::Union{ComplexF64,Float64}`: First coefficient `-rZ₁(kr)`.

## Returns
- `Vector{ComplexF64}`: The mutated vector `f` containing the normalized
  Taylor coefficients of `kZ₀(kr)` through degree `p`.
"""
@inline function radial0_k_taylor!(f::Vector{ComplexF64}, y::Vector{ComplexF64}, k::Float64, r2::Float64, p::Int, y0::Union{ComplexF64,Float64}, y1::Union{ComplexF64,Float64})::Vector{ComplexF64}
    radial0_taylor!(y,k,r2,p,y0,y1); f[1]=k*y[1]
    @inbounds @simd for n=1:p
        f[n+1]=k*y[n+1]+y[n]
    end
    return f
end

"""
    TaylorWorkspace

Reusable degree-`p` scratch storage for analytic DLP/CFIE Taylor assembly.

`H1` and `J1` store normalized Taylor coefficients of `kH₁⁽¹⁾(kr)` and
`kJ₁(kr)`. `H0` and `J0` store coefficients of `kH₀⁽¹⁾(kr)` and
`kJ₀(kr)`, while `yH0` and `yJ0` are scratch buffers for the corresponding
unweighted order-zero series `H₀⁽¹⁾(kr)` and `J₀(kr)`.

`a` contains the Taylor coefficients of the current kernel entry, `tmp` is
used for symmetry-orbit accumulation, and `β` contains the transformed
Chebyshev coefficients. Matrix assembly allocates one workspace per Julia
thread so that source-target entries can be evaluated without shared scratch
storage.

## Arguments
The fields are populated by [`TaylorWorkspace(p::Int)`](@ref):
- `a::Vector{ComplexF64}`: Current raw kernel Taylor coefficients.
- `tmp::Vector{ComplexF64}`: Temporary accumulation buffer.
- `β::Vector{ComplexF64}`: Chebyshev coefficient buffer.
- `H1::Vector{ComplexF64}`: Taylor coefficients of `kH₁⁽¹⁾`.
- `J1::Vector{ComplexF64}`: Taylor coefficients of `kJ₁`.
- `H0::Vector{ComplexF64}`: Taylor coefficients of `kH₀⁽¹⁾`.
- `J0::Vector{ComplexF64}`: Taylor coefficients of `kJ₀`.
- `yH0::Vector{ComplexF64}`: Unweighted `H₀⁽¹⁾` Taylor workspace.
- `yJ0::Vector{ComplexF64}`: Unweighted `J₀` Taylor workspace.

## Returns
- `TaylorWorkspace`: Reusable scratch storage for analytic Taylor and
  Taylor-to-Chebyshev assembly.
"""
struct TaylorWorkspace
    a::Vector{ComplexF64}
    tmp::Vector{ComplexF64}
    β::Vector{ComplexF64}
    H1::Vector{ComplexF64}
    J1::Vector{ComplexF64}
    H0::Vector{ComplexF64}
    J0::Vector{ComplexF64}
    yH0::Vector{ComplexF64}
    yJ0::Vector{ComplexF64}
end

"""
    TaylorWorkspace(p::Int) -> TaylorWorkspace

Allocate the nine length-`p+1` scratch vectors required by the analytic
Taylor kernels. The workspace contains no geometry-dependent data and can be
reused for every source-target pair at fixed polynomial degree.

## Arguments
- `p::Int`: Maximum Taylor polynomial degree.

## Returns
- `TaylorWorkspace`: Workspace containing nine zero-initialized
  `Vector{ComplexF64}` buffers of length `p+1`.
"""
function TaylorWorkspace(p::Int)::TaylorWorkspace
    z() = zeros(ComplexF64, p + 1)
    return TaylorWorkspace(z(), z(), z(), z(), z(), z(), z(), z(), z())
end