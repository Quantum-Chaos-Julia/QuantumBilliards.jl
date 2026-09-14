using Bessels
using CoordinateTransformations, Rotations, StaticArrays

#TODO Replace independant sin and cos calls with sincos
#TODO Steer's reccurence formula for besselj might provide speedup, already in Bessels but we have to compute redundant terms.
"""
    Jv(nu, r) → J

Evaluate the Bessel function of the first kind of order `nu` at radius `r`, using
[`Bessels.besselj`](@ref).

## Arguments
* `nu`: Order of the Bessel function.
* `r`: Radial argument at which the Bessel function is evaluated.

## Returns
*  `J` : Value of the Bessel function \$J_{\\nu}(r)\$.
"""
Jv(nu,r)=Bessels.besselj(nu,r)

"""
    ca_fb(nu, k::T, r::T, phi::T) where {T<:Real} → f::T

Evaluate a single corner-adapted Fourier-Bessel term at wavenumber `k` and polar
coordinates `(r, phi)`.

## Description
The term combines the Bessel function of the first kind with an angular sine
factor enforcing the Dirichlet condition at the corner edges:

```math
f(r,\\varphi) = J_{\\nu}(kr)\\,\\sin(\\nu\\varphi).
```

## Arguments
* `nu`: Angular order of the term (typically `basis.nu * i` for basis index `i`).
* `k`: Wavenumber.
* `r`: Radial coordinate.
* `phi`: Angular coordinate.

## Returns
*  `f` : Value of the corner-adapted Fourier-Bessel term.
"""
function ca_fb(nu,k::T,r::T,phi::T) where {T<:Real}
    return Jv(nu,k*r)*sin(nu*phi) 
end

"""
    Jvp(nu, r::T) where {T<:Real} → dJ::T

Evaluate the derivative of the Bessel function of the first kind \$J_{\\nu}\$ with
respect to its argument, at `r`.

## Description
Uses the standard recurrence relation for Bessel function derivatives:

```math
J_{\\nu}'(r) = \\tfrac{1}{2}\\left(J_{\\nu-1}(r) - J_{\\nu+1}(r)\\right).
```

## Arguments
* `nu`: Order of the Bessel function.
* `r`: Argument at which the derivative is evaluated.

## Returns
*  `dJ` : Value of \$J_{\\nu}'(r)\$.
"""
function Jvp(nu,r::T) where {T<:Real}
    j_minus=Jv(nu-one(T),r)
    j_plus=Jv(nu+one(T),r)
    return 0.5*(j_minus-j_plus)
end

"""
    ca_fb_dk(nu, k, r, phi) → df

Evaluate the derivative with respect to the wavenumber `k` of the corner-adapted
Fourier-Bessel term [`ca_fb`](@ref).

## Description
Differentiating \$f(r,\\varphi) = J_{\\nu}(kr)\\sin(\\nu\\varphi)\$ with respect
to `k` via the chain rule gives:

```math
\\frac{\\partial f}{\\partial k} = r\\,J_{\\nu}'(kr)\\,\\sin(\\nu\\varphi).
```

## Arguments
* `nu`: Angular order of the term.
* `k`: Wavenumber.
* `r`: Radial coordinate.
* `phi`: Angular coordinate.

## Returns
*  `df` : Value of \$\\partial f/\\partial k\$.
"""
ca_fb_dk(nu,k,r,phi)=r*Jvp(nu,k*r)*sin(nu*phi)

"""
CornerAdaptedFourierBessel{T,Sy} <: AbsBasis

`CornerAdaptedFourierBessel` is a concrete basis type representing corner-adapted
Fourier-Bessel functions for boundary value problems in domains with sharp
corners.

## Description
The basis functions are constructed in a local polar coordinate system centered
at a corner, with angular order fixed by the corner opening angle so that the
Dirichlet boundary condition is satisfied exactly on both edges meeting at the
corner. This follows the corner-adapted approach of Betcke & Trefethen,
"Reviving the Method of Particular Solutions".

## Attributes
* `cs`: Local [`PolarCS`](@ref) coordinate system centered at the corner.
* `dim`: Number of basis functions.
* `corner_angle`: Opening angle of the corner.
* `nu`: Angular order constant, with term order equal to `nu * i` for basis index `i`.
* `symmetries`: Optional vector of symmetries applied to the basis, or `nothing`.
* `rotation_angle_discontinuity`: Angle at which the local angular coordinate wraps, used to avoid branch-cut artifacts.

## API
The following functions can be evaluated for this type:
- [`resize_basis`](@ref)
- [`basis_fun`](@ref)
- [`dk_fun`](@ref)
- [`gradient`](@ref)
- [`basis_and_gradient`](@ref)
"""
struct CornerAdaptedFourierBessel{T,Sy} <: AbsBasis where  {T<:Real,Sy<:Union{AbsSymmetry,Nothing}}
    cs::PolarCS{T}
    dim::Int64 #using concrete type
    corner_angle::T
    nu::T #order constant, order=nu*i
    symmetries::Union{Vector{Any},Nothing}
    rotation_angle_discontinuity::T
end

"""
    CornerAdaptedFourierBessel(dim::Int64, corner_angle::T, origin::SVector{2,T}, rot_angle::T; rotation_angle_discontinuity = zero(T)) where {T<:Real} → basis::CornerAdaptedFourierBessel

Construct a [`CornerAdaptedFourierBessel`](@ref) basis of dimension `dim`
adapted to a corner with opening angle `corner_angle`, located at `origin` and
rotated by `rot_angle`. No symmetries are attached.

## Description
The angular order `nu = pi/corner_angle` of each basis function is fixed by the
corner angle so that the Dirichlet boundary condition is satisfied exactly on
both edges meeting at the corner. See Betcke & Trefethen, "Reviving the Method
of Particular Solutions", for background.

## Arguments
* `dim`: Number of basis functions.
* `corner_angle`: Opening angle of the corner.
* `origin`: Position of the corner, used as the origin of the local [`PolarCS`](@ref).
* `rot_angle`: Rotation angle of the local coordinate system.

## Keyword arguments
*  `rotation_angle_discontinuity::T = zero(T)` : Angle at which the local angular coordinate wraps, used to avoid branch-cut artifacts.

## Returns
*  `basis` : A [`CornerAdaptedFourierBessel`](@ref) basis with no attached symmetries.
"""
function CornerAdaptedFourierBessel(dim::Int64,corner_angle::T,origin::SVector{2,T},rot_angle::T;rotation_angle_discontinuity=zero(T)) where {T<:Real}
    cs=PolarCS(origin,rot_angle)
    nu=pi/corner_angle
    return CornerAdaptedFourierBessel{T,Nothing}(cs,dim,corner_angle,nu,nothing,rotation_angle_discontinuity)
end

"""
    CornerAdaptedFourierBessel(dim::Int64, corner_angle::T, cs::CoordinateSystem, symmetry::Union{Vector{Any},Nothing}; rotation_angle_discontinuity = zero(T)) where {T<:Real} → basis::CornerAdaptedFourierBessel

Construct a [`CornerAdaptedFourierBessel`](@ref) basis of dimension `dim`
adapted to a corner with opening angle `corner_angle`, using an existing
coordinate system `cs` and attaching the given `symmetry`.

## Arguments
* `dim`: Number of basis functions.
* `corner_angle`: Opening angle of the corner.
* `cs`: Local coordinate system centered at the corner.
* `symmetry`: Vector of symmetries to attach to the basis, or `nothing`.

## Keyword arguments
*  `rotation_angle_discontinuity::T = zero(T)` : Angle at which the local angular coordinate wraps, used to avoid branch-cut artifacts.

## Returns
*  `basis` : A [`CornerAdaptedFourierBessel`](@ref) basis using the given coordinate system and symmetries.
"""
function CornerAdaptedFourierBessel(dim::Int64,corner_angle::T,cs::CoordinateSystem,symmetry::Union{Vector{Any},Nothing};rotation_angle_discontinuity=zero(T)) where {T<:Real}
    nu=pi/corner_angle
    return CornerAdaptedFourierBessel{T,Nothing}(cs,dim,corner_angle,nu,symmetry,rotation_angle_discontinuity)
end

"""
    CornerAdaptedFourierBessel(dim::Int64, corner_angle::T, origin::SVector{2,T}, rot_angle::T, symmetry::Union{Vector{Any},Nothing}; rotation_angle_discontinuity = zero(T)) where {T<:Real} → basis::CornerAdaptedFourierBessel

Construct a [`CornerAdaptedFourierBessel`](@ref) basis of dimension `dim`
adapted to a corner with opening angle `corner_angle`, located at `origin` and
rotated by `rot_angle`, attaching the given `symmetry`.

## Arguments
* `dim`: Number of basis functions.
* `corner_angle`: Opening angle of the corner.
* `origin`: Position of the corner, used as the origin of the local [`PolarCS`](@ref).
* `rot_angle`: Rotation angle of the local coordinate system.
* `symmetry`: Vector of symmetries to attach to the basis, or `nothing`.

## Keyword arguments
*  `rotation_angle_discontinuity::T = zero(T)` : Angle at which the local angular coordinate wraps, used to avoid branch-cut artifacts.

## Returns
*  `basis` : A [`CornerAdaptedFourierBessel`](@ref) basis with the given origin, rotation, and symmetries.
"""
function CornerAdaptedFourierBessel(dim::Int64,corner_angle::T,origin::SVector{2,T},rot_angle::T,symmetry::Union{Vector{Any},Nothing};rotation_angle_discontinuity=zero(T)) where {T<:Real}
    cs=PolarCS(origin,rot_angle)
    nu=pi/corner_angle
    return CornerAdaptedFourierBessel{T,Nothing}(cs,dim,corner_angle,nu,symmetry,rotation_angle_discontinuity)
end

"""
    toFloat32(basis::CornerAdaptedFourierBessel) → basis32::CornerAdaptedFourierBessel

Convert a [`CornerAdaptedFourierBessel`](@ref) basis to use `Float32` precision.

## Arguments
* `basis`: The basis to convert.

## Returns
*  `basis32` : A new basis with `dim`, `corner_angle`, and coordinate system fields converted to `Float32`.
"""
toFloat32(basis::CornerAdaptedFourierBessel) = CornerAdaptedFourierBessel(basis.dim,Float32(basis.corner_angle),Float32.(basis.cs.origin),Float32(basis.cs.rot_angle))

"""
    resize_basis(basis::CornerAdaptedFourierBessel, billiard::Bi, dim::Int, k) where {Bi<:AbsBilliard} → basis_new::CornerAdaptedFourierBessel

Return a [`CornerAdaptedFourierBessel`](@ref) basis resized to dimension `dim`,
reusing `basis` unchanged if it already has the requested dimension.

## Arguments
* `basis`: The basis to resize.
* `billiard`: Billiard the basis is defined on (unused, kept for interface consistency with other basis types).
* `dim`: Target dimension.
* `k`: Wavenumber (unused, kept for interface consistency with other basis types).

## Returns
*  `basis_new` : `basis` itself if `basis.dim == dim`, otherwise a new basis with dimension `dim` and the same corner angle, coordinate system, symmetries and rotation angle discontinuity.
"""
function resize_basis(basis::CornerAdaptedFourierBessel,billiard::Bi,dim::Int,k) where {Bi<:AbsBilliard}
    if basis.dim==dim
        return basis
    else
        return CornerAdaptedFourierBessel(dim,basis.corner_angle,basis.cs,basis.symmetries;rotation_angle_discontinuity=basis.rotation_angle_discontinuity)
    end
end

"""
    basis_fun(basis::CornerAdaptedFourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real} → out::Vector{T}

Evaluate the `i`-th corner-adapted Fourier-Bessel basis function at wavenumber
`k` on the points `pts`.

## Description
The points are mapped to local polar coordinates `(r, phi)` of the basis's
corner, and the term [`ca_fb`](@ref) is evaluated with angular order `nu * i`.

## Arguments
* `basis`: The [`CornerAdaptedFourierBessel`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the basis function is evaluated.

## Returns
*  `out` : Values of the basis function at the input points.
"""
@inline function basis_fun(basis::CornerAdaptedFourierBessel{T},i::Int,k::T,pts::AbstractArray) where {T<:Real}
    pm=basis.cs.local_map
    ν=basis.nu
    M=length(pts)
    r=Vector{T}(undef,M)
    φ=Vector{T}(undef,M)
    _polar_coords!(r,φ,pm,pts,basis.rotation_angle_discontinuity)
    out=Vector{T}(undef,M)
    m=ν*i
    @inbounds @simd for j=1:M
        out[j]=ca_fb(m,k,r[j],φ[j]); end
    return out
end

"""
    basis_fun(basis::CornerAdaptedFourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → B::Matrix{T}

Evaluate the corner-adapted Fourier-Bessel basis functions with the given
`indices` at wavenumber `k` on the points `pts`.

## Description
The points are mapped to local polar coordinates `(r, phi)` of the basis's
corner once, and the term [`ca_fb`](@ref) is evaluated with angular order
`nu * indices[c]` for each column `c`, optionally in parallel across threads.

## Arguments
* `basis`: The [`CornerAdaptedFourierBessel`](@ref) basis.
* `indices`: Indices of the basis functions to evaluate.
* `k`: Wavenumber.
* `pts`: Points at which the basis functions are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `B` : Basis matrix of size `(length(pts), length(indices))`.
"""
@inline function basis_fun(basis::CornerAdaptedFourierBessel{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    pm=basis.cs.local_map
    ν=basis.nu
    M=length(pts)
    N=length(indices)
    r=Vector{T}(undef,M)
    φ=Vector{T}(undef,M)
    _polar_coords!(r,φ,pm,pts,basis.rotation_angle_discontinuity)
    B=Matrix{T}(undef,M,N)
    @use_threads multithreading=multithreaded for c=1:N
        m=ν*indices[c]
        col=@view B[:,c]
        @inbounds @simd for j=1:M
            col[j]=ca_fb(m,k,r[j],φ[j])
        end
    end
    return B
end

"""
    dk_fun(basis::CornerAdaptedFourierBessel{T}, i::Int, k::T, pts::AbstractArray) where {T<:Real} → dk::Vector{T}

Evaluate the derivative with respect to the wavenumber `k` of the `i`-th
corner-adapted Fourier-Bessel basis function on the points `pts`.

## Description
The points are mapped to local polar coordinates `(r, phi)` of the basis's
corner, and the term [`ca_fb_dk`](@ref) is evaluated with angular order `nu * i`.

## Arguments
* `basis`: The [`CornerAdaptedFourierBessel`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the derivative is evaluated.

## Returns
*  `dk` : Column `i` of \$\\partial B/\\partial k\$, the wavenumber derivative of the basis matrix.
"""
@inline function dk_fun(basis::CornerAdaptedFourierBessel{T},i::Int,k::T,pts::AbstractArray) where {T<:Real}
    pm=basis.cs.local_map
    ν=basis.nu
    m=ν*i
    M=length(pts)
    r=Vector{T}(undef,M)
    φ=Vector{T}(undef,M)
    _polar_coords!(r,φ,pm,pts,basis.rotation_angle_discontinuity)
    dk=Vector{T}(undef,M)
    @inbounds @simd for j in 1:M
        dk[j]=r[j]*Jvp(m,k*r[j])*sin(m*φ[j])
    end
    return dk
end
    
"""
    dk_fun(basis::CornerAdaptedFourierBessel{T}, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → dB_dk::Matrix{T}

Evaluate the derivative with respect to the wavenumber `k` of the corner-adapted
Fourier-Bessel basis functions with the given `indices` on the points `pts`.

## Description
The points are mapped to local polar coordinates `(r, phi)` of the basis's
corner once, and the term [`ca_fb_dk`](@ref) is evaluated with angular order
`nu * indices[c]` for each column `c`, optionally in parallel across threads.

## Arguments
* `basis`: The [`CornerAdaptedFourierBessel`](@ref) basis.
* `indices`: Indices of the basis functions to differentiate.
* `k`: Wavenumber.
* `pts`: Points at which the derivatives are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `dB_dk` : Wavenumber derivative of the basis matrix, of size `(length(pts), length(indices))`.
"""
@inline function dk_fun(basis::CornerAdaptedFourierBessel{T},indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    pm=basis.cs.local_map
    ν=basis.nu
    M=length(pts)
    N=length(indices)
    r=Vector{T}(undef,M)
    φ=Vector{T}(undef,M)
    _polar_coords!(r,φ,pm,pts,basis.rotation_angle_discontinuity)
    dB_dk=Matrix{T}(undef,M,N)
    @use_threads multithreading=multithreaded for c in 1:N
        m=ν*indices[c]
        col=@view dB_dk[:,c]
        @inbounds @simd for j in 1:M
            col[j]=r[j]*Jvp(m,k*r[j])*sin(m*φ[j])
        end
    end
    return dB_dk
end

"""
    gradient(basis::CornerAdaptedFourierBessel, i::Int, k::T, pts::AbstractArray) where {T<:Real} → (dx, dy)::Tuple{Vector{T},Vector{T}}

Evaluate the gradient with respect to the Cartesian coordinates `x` and `y` of
the `i`-th corner-adapted Fourier-Bessel basis function on the points `pts`.

## Description
The points are mapped to local Cartesian, then polar, coordinates `(r, phi)` of
the basis's corner. Using the chain rule with the polar-to-Cartesian Jacobian,
the gradient components are:

```math
\\partial_x f = \\cos\\varphi\\,\\partial_r f - \\frac{\\sin\\varphi}{r}\\,\\partial_\\varphi f, \\qquad
\\partial_y f = \\sin\\varphi\\,\\partial_r f + \\frac{\\cos\\varphi}{r}\\,\\partial_\\varphi f,
```

with \$\\partial_r f = k J_{\\nu}'(kr)\\sin(\\nu\\varphi)\$ and
\$\\partial_\\varphi f = \\nu J_{\\nu}(kr)\\cos(\\nu\\varphi)\$.

## Arguments
* `basis`: The [`CornerAdaptedFourierBessel`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the gradient is evaluated.

## Returns
*  `(dx, dy)` : Vectors with the `x` and `y` components of the gradient of basis function `i` at the input points.
"""
function gradient(basis::CornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    pm=basis.cs.local_map
    ν=basis.nu
    m=ν*i
    M=length(pts)
    r=Vector{T}(undef,M)
    φ=Vector{T}(undef,M)
    _polar_coords!(r,φ,pm,pts,basis.rotation_angle_discontinuity)
    dx=Vector{T}(undef,M)
    dy=Vector{T}(undef,M)
    @inbounds for j in 1:M
        rj=r[j]
        invr=(rj==0 ? zero(T) : inv(rj))
        sφ=sin(φ[j])
        cφ=cos(φ[j])
        jv=Jv(m,k*rj)
        dj=Jvp(m,k*rj)
        fr=k*dj*sin(m*φ[j])
        fφ=m*jv*cos(m*φ[j])
        dx[j]=cφ*fr-sφ*invr*fφ
        dy[j]=sφ*fr+cφ*invr*fφ
    end
    return dx,dy
end

"""
    gradient(basis::CornerAdaptedFourierBessel, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → (dB_dx, dB_dy)::Tuple{Matrix{T},Matrix{T}}

Evaluate the gradient with respect to the Cartesian coordinates `x` and `y` of
the corner-adapted Fourier-Bessel basis functions with the given `indices` on
the points `pts`.

## Description
As in [`gradient`](@ref) for a single index, but evaluated column-by-column for
each index in `indices`, optionally in parallel across threads.

## Arguments
* `basis`: The [`CornerAdaptedFourierBessel`](@ref) basis.
* `indices`: Indices of the basis functions to differentiate.
* `k`: Wavenumber.
* `pts`: Points at which the gradients are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `(dB_dx, dB_dy)` : Matrices with the `x` and `y` components of the gradients, each of size `(length(pts), length(indices))`.
"""
function gradient(basis::CornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    pm=basis.cs.local_map
    ν=basis.nu
    M=length(pts)
    N=length(indices)
    r=Vector{T}(undef,M)
    φ=Vector{T}(undef,M)
    _polar_coords!(r,φ,pm,pts,basis.rotation_angle_discontinuity)
    dB_dx=Matrix{T}(undef,M,N); dB_dy=Matrix{T}(undef,M,N)
    @use_threads multithreading=multithreaded for c in 1:N
        m=ν*indices[c]
        cx=@view dB_dx[:,c]
        cy=@view dB_dy[:,c]
        @inbounds for j in 1:M
            rj=r[j]
            invr=(rj==0 ? zero(T) : inv(rj))
            sφ=sin(φ[j])
            cφ=cos(φ[j])
            jv=Jv(m,k*rj)
            dj=Jvp(m,k*rj)
            fr=k*dj*sin(m*φ[j])
            fφ=m*jv*cos(m*φ[j])
            cx[j]=cφ*fr-sφ*invr*fφ
            cy[j]=sφ*fr+cφ*invr*fφ
        end
    end
    return dB_dx,dB_dy
end

"""
    basis_and_gradient(basis::CornerAdaptedFourierBessel, i::Int, k::T, pts::AbstractArray) where {T<:Real} → (bf, dx, dy)::Tuple{Vector{T},Vector{T},Vector{T}}

Evaluate both the `i`-th corner-adapted Fourier-Bessel basis function and its
gradient with respect to `x` and `y` on the points `pts`.

## Description
Combines [`basis_fun`](@ref) and [`gradient`](@ref) in a single pass over the
points, avoiding redundant coordinate transformations and Bessel function
evaluations.

## Arguments
* `basis`: The [`CornerAdaptedFourierBessel`](@ref) basis.
* `i`: Index of the basis function.
* `k`: Wavenumber.
* `pts`: Points at which the basis function and its gradient are evaluated.

## Returns
*  `(bf, dx, dy)` : Basis function values and the `x` and `y` components of its gradient at the input points.
"""
function basis_and_gradient(basis::CornerAdaptedFourierBessel,i::Int,k::T,pts::AbstractArray) where {T<:Real}
    pm=basis.cs.local_map
    ν=basis.nu
    m=ν*i
    M=length(pts)
    r=Vector{T}(undef,M)
    φ=Vector{T}(undef,M)
    _polar_coords!(r,φ,pm,pts,basis.rotation_angle_discontinuity)
    bf=Vector{T}(undef,M)
    dx=Vector{T}(undef,M)
    dy=Vector{T}(undef,M)
    @inbounds for j in 1:M
        rj=r[j]
        invr=(rj==0 ? zero(T) : inv(rj))
        sφ=sin(φ[j])
        cφ=cos(φ[j])
        jv=Jv(m,k*rj)
        dj=Jvp(m,k*rj)
        s=sin(m*φ[j])
        c=cos(m*φ[j])
        bf[j]=jv*s
        fr=k*dj*s
        fφ=m*jv*c
        dx[j]=cφ*fr-sφ*invr*fφ
        dy[j]=sφ*fr+cφ*invr*fφ
    end
    return bf,dx,dy
end

"""
    basis_and_gradient(basis::CornerAdaptedFourierBessel, indices::AbstractArray, k::T, pts::AbstractArray; multithreaded::Bool = true) where {T<:Real} → (B, dB_dx, dB_dy)::Tuple{Matrix{T},Matrix{T},Matrix{T}}

Evaluate both the corner-adapted Fourier-Bessel basis functions with the given
`indices` and their gradients with respect to `x` and `y` on the points `pts`.

## Description
Combines [`basis_fun`](@ref) and [`gradient`](@ref) column-by-column,
optionally in parallel across threads, avoiding redundant coordinate
transformations and Bessel function evaluations.

## Arguments
* `basis`: The [`CornerAdaptedFourierBessel`](@ref) basis.
* `indices`: Indices of the basis functions to evaluate.
* `k`: Wavenumber.
* `pts`: Points at which the basis functions and gradients are evaluated.

## Keyword arguments
*  `multithreaded::Bool = true` : Whether the matrix construction is multithreaded across columns.

## Returns
*  `(B, dB_dx, dB_dy)` : Basis matrix and the `x` and `y` components of its gradients, each of size `(length(pts), length(indices))`.
"""
function basis_and_gradient(basis::CornerAdaptedFourierBessel,indices::AbstractArray,k::T,pts::AbstractArray;multithreaded::Bool=true) where {T<:Real}
    pm=basis.cs.local_map
    ν=basis.nu
    M=length(pts)
    N=length(indices)
    r=Vector{T}(undef,M)
    φ=Vector{T}(undef,M)
    _polar_coords!(r,φ,pm,pts,basis.rotation_angle_discontinuity)
    B=Matrix{T}(undef,M,N)
    dB_dx=Matrix{T}(undef,M,N)
    dB_dy=Matrix{T}(undef,M,N)
    @use_threads multithreading=multithreaded for c in 1:N
        m=ν*indices[c]
        bc=@view B[:,c]
        cx=@view dB_dx[:,c]
        cy=@view dB_dy[:,c]
        @inbounds for j in 1:M
            rj=r[j]
            invr=(rj==0 ? zero(T) : inv(rj))
            sφ=sin(φ[j])
            cφ=cos(φ[j])
            jv=Jv(m,k*rj)
            dj=Jvp(m,k*rj)
            s=sin(m*φ[j])
            c=cos(m*φ[j])
            bc[j]=jv*s
            fr=k*dj*s
            fφ=m*jv*c
            cx[j]=cφ*fr-sφ*invr*fφ
            cy[j]=sφ*fr+cφ*invr*fφ
        end
    end
    return B,dB_dx,dB_dy
end