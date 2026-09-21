"""
    BoundaryPoints{T<:Real}

Container for sampled boundary geometry, quadrature data, and solver-specific
boundary discretization data.

`BoundaryPoints` is shared by basis and boundary-integral solvers. The common
geometric fields store boundary positions, normals, arc-length coordinates,
and quadrature elements, while additional fields provide the parametrization
and weights required by particular numerical methods. Fields not required by
a given discretization are left empty or set to neutral scalar values.

All populated node-wise vector fields are required to have the same length as
`xy`.

## Attributes
* `xy::Vector{SVector{2,T}}`: Boundary points in Cartesian coordinates.
* `normal::Vector{SVector{2,T}}`: Outward unit normals at the boundary points.
* `kappa::Vector{T}`: Curvature data used by basis-method discretizations.
* `s::Vector{T}`: Arc-length coordinates along the boundary.
* `ds::Vector{T}`: Physical arc-length quadrature elements.
* `rdotn::Vector{T}`: Values of the geometric factor `r ⋅ n`.
* `w_vs::Vector{T}`: Vergini-Saraceno quadrature weights.
* `w_dm::Vector{T}`: Decomposition-method quadrature weights.
* `xy_int::Vector{SVector{2,T}}`: Associated interior sampling points.
* `w::Vector{T}`: Boundary-integral quadrature weights.
* `w_n::Vector{T}`: Additional boundary-integral normal weights.
* `curvature::Vector{T}`: Curvature data used by boundary-integral discretizations.
* `shift_x::T`: Horizontal parametrization or symmetry shift.
* `shift_y::T`: Vertical parametrization or symmetry shift.
* `tangent::Vector{SVector{2,T}}`: First derivatives of the boundary parametrization.
* `tangent_2::Vector{SVector{2,T}}`: Second derivatives of the boundary parametrization.
* `ts::Vector{T}`: Computational parameter values.
* `tphys::Vector{T}`: Corresponding physical boundary parameter values.
* `ws::Vector{T}`: Quadrature weights in the computational parametrization.
* `ws_der::Vector{T}`: Derivatives associated with the grading transformation.
* `compid::Int`: Connected boundary-component index.
* `is_periodic::Bool`: Whether the represented component is periodically parametrized.
* `xL::SVector{2,T}`: Left endpoint for a nonperiodic component.
* `xR::SVector{2,T}`: Right endpoint for a nonperiodic component.
* `tL::SVector{2,T}`: Endpoint tangent associated with `xL`.
* `tR::SVector{2,T}`: Endpoint tangent associated with `xR`.

## API
* [`boundary_coords`](@ref)
* [`boundary_matrix_size`](@ref)
* [`boundary_s`](@ref)
* [`component_offsets`](@ref)
* `Base.length`
* `Base.isempty`
"""
struct BoundaryPoints{T<:Real}
    xy::Vector{SVector{2,T}}
    normal::Vector{SVector{2,T}}
    kappa::Vector{T}
    s::Vector{T}
    ds::Vector{T}
    rdotn::Vector{T}
    w_vs::Vector{T}
    w_dm::Vector{T}
    xy_int::Vector{SVector{2,T}}
    w::Vector{T}
    w_n::Vector{T}
    curvature::Vector{T}
    shift_x::T
    shift_y::T
    tangent::Vector{SVector{2,T}}
    tangent_2::Vector{SVector{2,T}}
    ts::Vector{T}
    tphys::Vector{T}
    ws::Vector{T}
    ws_der::Vector{T}
    compid::Int
    is_periodic::Bool
    xL::SVector{2,T}
    xR::SVector{2,T}
    tL::SVector{2,T}
    tR::SVector{2,T}

    # Inner constructor with validation
    function BoundaryPoints{T}(xy, normal, kappa, s, ds, rdotn, w_vs, w_dm, xy_int, w, w_n, curvature, shift_x, shift_y, tangent, tangent_2, ts, tphys, ws, ws_der, compid, is_periodic, xL, xR, tL, tR) where T<:Real
        n = length(xy)
        # Validate that non-empty vectors have consistent lengths
        for (name, vec) in [(:normal, normal), (:s, s), (:ds, ds), (:rdotn, rdotn), (:w_vs, w_vs), (:w_dm, w_dm), (:w, w), (:w_n, w_n), (:curvature, curvature), (:tangent, tangent), (:tangent_2, tangent_2), (:ts, ts), (:tphys, tphys), (:ws, ws), (:ws_der, ws_der)]
            if !isempty(vec) && length(vec) != n
                error("Length of $name ($(length(vec))) must match xy ($n)")
            end
        end
        new{T}(xy, normal, kappa, s, ds, rdotn, w_vs, w_dm, xy_int, w, w_n, curvature, shift_x, shift_y, tangent, tangent_2, ts, tphys, ws, ws_der, compid, is_periodic, xL, xR, tL, tR)
    end
end

"""
    BoundaryPoints(xy::Vector{SVector{2,T}}; kwargs...) where {T<:Real} → bp::BoundaryPoints{T}

Construct a boundary discretization from Cartesian boundary points, inferring
the scalar type `T` from `xy`.

All additional geometric, quadrature, parametrization, and solver-specific data
are supplied through keyword arguments. Unspecified vector fields are empty and
unspecified scalar fields use their neutral defaults.

## Arguments
* `xy::Vector{SVector{2,T}}`: Boundary points in Cartesian coordinates.

## Keyword Arguments
* `normal::Vector{SVector{2,T}} = SVector{2,T}[]`: Outward unit normals.
* `kappa::Vector{T} = T[]`: Curvature data used by basis methods.
* `s::Vector{T} = T[]`: Arc-length coordinates.
* `ds::Vector{T} = T[]`: Physical arc-length quadrature elements.
* `rdotn::Vector{T} = T[]`: Values of `r ⋅ n`.
* `w_vs::Vector{T} = T[]`: Vergini-Saraceno quadrature weights.
* `w_dm::Vector{T} = T[]`: Decomposition-method quadrature weights.
* `xy_int::Vector{SVector{2,T}} = SVector{2,T}[]`: Associated interior sampling points.
* `w::Vector{T} = T[]`: Boundary-integral quadrature weights.
* `w_n::Vector{T} = T[]`: Additional boundary-integral normal weights.
* `curvature::Vector{T} = T[]`: Curvature data used by boundary-integral discretizations.
* `shift_x::T = zero(T)`: Horizontal parametrization or symmetry shift.
* `shift_y::T = zero(T)`: Vertical parametrization or symmetry shift.
* `tangent::Vector{SVector{2,T}} = SVector{2,T}[]`: First parametrization derivatives.
* `tangent_2::Vector{SVector{2,T}} = SVector{2,T}[]`: Second parametrization derivatives.
* `ts::Vector{T} = T[]`: Computational parameter values.
* `tphys::Vector{T} = T[]`: Physical boundary parameter values.
* `ws::Vector{T} = T[]`: Computational quadrature weights.
* `ws_der::Vector{T} = T[]`: Derivatives associated with the grading transformation.
* `compid::Int = 1`: Connected boundary-component index.
* `is_periodic::Bool = true`: Whether the component is periodically parametrized.
* `xL::SVector{2,T} = zero`: Left endpoint of a nonperiodic component.
* `xR::SVector{2,T} = zero`: Right endpoint of a nonperiodic component.
* `tL::SVector{2,T} = zero`: Endpoint tangent associated with `xL`.
* `tR::SVector{2,T} = zero`: Endpoint tangent associated with `xR`.

## Returns
* `bp::BoundaryPoints{T}`: Boundary discretization containing the supplied data.
"""
function BoundaryPoints(xy::Vector{SVector{2,T}}; normal=SVector{2,T}[], kappa=T[], s=T[], ds=T[], rdotn=T[], w_vs=T[], w_dm=T[], xy_int=SVector{2,T}[], w=T[], w_n=T[], curvature=T[], shift_x=zero(T), shift_y=zero(T), tangent=SVector{2,T}[], tangent_2=SVector{2,T}[], ts=T[], tphys=T[], ws=T[], ws_der=T[], compid=1, is_periodic=true, xL=SVector{2,T}(zero(T),zero(T)), xR=SVector{2,T}(zero(T),zero(T)), tL=SVector{2,T}(zero(T),zero(T)), tR=SVector{2,T}(zero(T),zero(T))) where T<:Real
    return BoundaryPoints{T}(xy, normal, kappa, s, ds, rdotn, w_vs, w_dm, xy_int, w, w_n, curvature, shift_x, shift_y, tangent, tangent_2, ts, tphys, ws, ws_der, compid, is_periodic, xL, xR, tL, tR)
end

"""
    BoundaryPoints(xy::Vector{SVector{2,T}}, tangent::Vector{SVector{2,T}}, tangent_2::Vector{SVector{2,T}}, ts::Vector{T}, tphys::Vector{T}, ws::Vector{T}, ws_der::Vector{T}, s::Vector{T}, ds::Vector{T}, compid::Int, is_periodic::Bool, xL::SVector{2,T}, xR::SVector{2,T}, tL::SVector{2,T}, tR::SVector{2,T}) where {T<:Real} → bp::BoundaryPoints{T}

Construct a parametrized boundary discretization for boundary-integral
quadrature.

The outward unit normal at each node is computed from the parametrization
tangent `t = (t_x,t_y)` as

    n = (t_y,-t_x)/|t|.

## Arguments
* `xy::Vector{SVector{2,T}}`: Boundary points in Cartesian coordinates.
* `tangent::Vector{SVector{2,T}}`: First parametrization derivatives.
* `tangent_2::Vector{SVector{2,T}}`: Second parametrization derivatives.
* `ts::Vector{T}`: Computational parameter values.
* `tphys::Vector{T}`: Corresponding physical boundary parameter values.
* `ws::Vector{T}`: Computational quadrature weights.
* `ws_der::Vector{T}`: Derivatives associated with the grading transformation.
* `s::Vector{T}`: Arc-length coordinates.
* `ds::Vector{T}`: Physical arc-length quadrature elements.
* `compid::Int`: Connected boundary-component index.
* `is_periodic::Bool`: Whether the component is periodically parametrized.
* `xL::SVector{2,T}`: Left endpoint of a nonperiodic component.
* `xR::SVector{2,T}`: Right endpoint of a nonperiodic component.
* `tL::SVector{2,T}`: Endpoint tangent associated with `xL`.
* `tR::SVector{2,T}`: Endpoint tangent associated with `xR`.

## Returns
* `bp::BoundaryPoints{T}`: Parametrized boundary discretization with computed outward normals.
"""
function BoundaryPoints(xy::Vector{SVector{2,T}}, tangent::Vector{SVector{2,T}}, tangent_2::Vector{SVector{2,T}}, ts::Vector{T}, tphys::Vector{T}, ws::Vector{T}, ws_der::Vector{T}, s::Vector{T}, ds::Vector{T}, compid::Int, is_periodic::Bool, xL::SVector{2,T}, xR::SVector{2,T}, tL::SVector{2,T}, tR::SVector{2,T}) where T<:Real
    n = length(xy)
    normal = Vector{SVector{2,T}}(undef, n)
    @inbounds for i in eachindex(tangent)
        tx, ty = tangent[i]
        sp = hypot(tx, ty)
        normal[i] = SVector{2,T}(ty/sp, -tx/sp)
    end
    return BoundaryPoints(xy; normal=normal, s=s, ds=ds, tangent=tangent, tangent_2=tangent_2, ts=ts, tphys=tphys, ws=ws, ws_der=ws_der, compid=compid, is_periodic=is_periodic, xL=xL, xR=xR, tL=tL, tR=tR)
end

"""
    length(bp::BoundaryPoints) → n::Int

Returns the number of sampled boundary points, `n = length(bp.xy)`.
"""
Base.length(bp::BoundaryPoints) = length(bp.xy)

"""
    isempty(bp::BoundaryPoints) → flag::Bool

Returns `true` if `bp` contains no boundary points, i.e. `isempty(bp.xy)`.
"""
Base.isempty(bp::BoundaryPoints) = isempty(bp.xy)

"""
    boundary_matrix_size(pts::BoundaryPoints) → N::Int

Returns the number of boundary degrees of freedom represented by `pts`.
"""
@inline boundary_matrix_size(pts::BoundaryPoints) = length(pts.xy)

"""
    boundary_matrix_size(pts::Vector{BoundaryPoints{T}}) where T<:Real → N::Int

Returns the total number of boundary degrees of freedom over all boundary
components in `pts` (used by [`CompositeBIMSolver`](@ref) and composite
`GlobalCornerGrading` boundaries).
"""
function boundary_matrix_size(pts::Vector{BoundaryPoints{T}}) where T<:Real
    return sum(length, pts)
end

"""
    boundary_s(pts::BoundaryPoints) → s::Vector

Returns the stored arc-length coordinates of the boundary discretization.
"""
@inline boundary_s(pts::BoundaryPoints) = pts.s

"""
    boundary_s(pts::Vector{BoundaryPoints{T}}) where T<:Real → s::Vector{T}

Returns continuous arc-length coordinates for a vector of boundary components,
shifting each component's local arc-length by the accumulated length of the
preceding components so that the result is continuous over the concatenated
boundary.
"""
function boundary_s(pts::Vector{BoundaryPoints{T}}) where T<:Real
    isempty(pts) && return T[]
    s = T[]
    sizehint!(s, sum(length(p.s) for p in pts))
    soff = zero(T)
    for p in pts
        append!(s, p.s .+ soff)
        soff += sum(p.ds)
    end
    return s
end

"""
    component_offsets(pts::Vector{BoundaryPoints{T}}) where T<:Real → offs::Vector{Int}

Returns the starting indices of the boundary components in the flattened
boundary discretization: for components with `N₁,N₂,...` points, the offsets
are `[1, 1+N₁, 1+N₁+N₂, ...]`, so the points of component `a` occupy
`offs[a]:offs[a+1]-1`.
"""
function component_offsets(pts::Vector{BoundaryPoints{T}}) where T<:Real
    offs = Vector{Int}(undef, length(pts)+1)
    offs[1] = 1
    @inbounds for i in eachindex(pts)
        offs[i+1] = offs[i] + length(pts[i])
    end
    return offs
end

"""
    component_offsets(pts::BoundaryPoints) → offs::Vector{Int}

Returns the component offsets for a single boundary component, `[1, length(pts)+1]`.
"""
@inline component_offsets(pts::BoundaryPoints) = [1, length(pts)+1]

"""
    points_in_billiard(pts, billiard) → mask

Returns the interior-membership mask of `pts` with respect to `billiard`,
delegating directly to [`is_inside`](@ref).
"""
@inline points_in_billiard(pts, billiard) = is_inside(billiard, pts)

function _determine_bp_sizes(curves, bs, k)
    Ns = Vector{Int64}(undef,length(curves)) # store the data to indexwise access. This needs to be this way b/c we dont know beforehand which curves are real and which are abstract. Use sizehint! to give an idea as to not need to resize b/c it could the that real and abstract curves and intermingled
    @inbounds for i in eachindex(curves) # make an initial size calculation of the resulting vectors
        crv=curves[i]
        Ns[i] =max(20,round(Int,k*crv.length*bs[i]/2*pi))
    end
    return Ns
end


"""
    boundary_coords(billiard::Bi, samplers::Vector{AbsSampler}, Ns::Vector{Int64}) where {Bi<:AbsBilliard} → bp::BoundaryPoints

Sample the full boundary of `billiard` using independently specified samplers
and point counts for each boundary curve. Curves marked with `QuantumSolverIgnore` are included. 

## Arguments
* `billiard::Bi`: Billiard whose full boundary is sampled.
* `samplers::Vector{AbsSampler}`: Sampler assigned to each boundary curve.
* `Ns::Vector{Int64}`: Number of points assigned to each boundary curve.

## Returns
* `bp::BoundaryPoints`: Boundary discretization containing positions, outward normals, arc-length coordinates, and physical quadrature elements.
"""
function boundary_coords(billiard::Bi, samplers::Vector{AbsSampler}, Ns::Vector{Int64}) where {Bi<:AbsBilliard}
    curves = get_boundary_curves_with_ignored(billiard); T = typeof(curves[1].length); M = length(curves)
    length(samplers) == M || throw(ArgumentError("Expected $M samplers, received $(length(samplers))"))
    length(Ns) == M || throw(ArgumentError("Expected $M point counts, received $(length(Ns))"))
    xy_all = Vector{Vector{SVector{2,T}}}(undef, M); normal_all = Vector{Vector{SVector{2,T}}}(undef, M)
    s_all = Vector{Vector{T}}(undef, M); ds_all = Vector{Vector{T}}(undef, M)
    L0 = zero(T)
    for i in eachindex(curves)
        crv = curves[i]; t, dt = sample_points(samplers[i], Ns[i])
        xy = curve(crv, t); ds = norm.(tangent(crv, t)) .* dt
        normal = domain_gradient_vector(crv, xy); normal .= normal ./ norm(normal)
        xy_all[i] = xy; normal_all[i] = normal; s_all[i] = arc_length(crv, t) .+ L0; ds_all[i] = ds
        L0 += crv.length
    end
    return BoundaryPoints(vcat(xy_all...); normal = vcat(normal_all...), s = vcat(s_all...), ds = vcat(ds_all...))
end

"""
    boundary_coords(billiard::Bi, fourier_sampler::FourierNodes, M) where {Bi<:AbsBilliard} → bp::BoundaryPoints

Samples the full boundary of `billiard` (including curves marked with
`QuantumSolverIgnore`) using a single `FourierNodes` sampler that
distributes a total of `M` points across all boundary curves. Otherwise behaves
like [`boundary_coords(::AbsBilliard, ::Vector{AbsSampler}, ::Vector{Int64})`](@ref).

## Arguments
* `billiard`: The billiard whose boundary is sampled.
* `fourier_sampler`: The `FourierNodes` sampler used to distribute the `M` points among all boundary curves.
* `M`: Total number of sample points to generate over the whole boundary.

## Returns
* `bp`: A [`BoundaryPoints`](@ref) instance with the `xy`, `normal`, `s` and `ds` fields populated.
"""
function boundary_coords(billiard::Bi, fourier_sampler::FourierNodes, M) where {Bi<:AbsBilliard}
    curves = get_boundary_curves_with_ignored(billiard); T = typeof(curves[1].length); n_curves = length(curves)
    ts, dts = sample_points(fourier_sampler, M)
    xy_all = Vector{Vector{SVector{2,T}}}(undef, n_curves); normal_all = Vector{Vector{SVector{2,T}}}(undef, n_curves)
    s_all = Vector{Vector{T}}(undef, n_curves); ds_all = Vector{Vector{T}}(undef, n_curves)
    L0 = zero(T)
    for i in eachindex(curves)
        crv = curves[i]; L = crv.length; t = ts[i]; dt = dts[i]
        xy = curve(crv, t); ds = norm.(tangent(crv, t)) .* dt
        normal = domain_gradient_vector(crv, xy); normal .= normal ./ norm(normal)
        xy_all[i] = xy; normal_all[i] = normal; s_all[i] = arc_length(crv, t) .+ L0; ds_all[i] = ds
        L0 += L
    end
    return BoundaryPoints(vcat(xy_all...); normal = vcat(normal_all...), s = vcat(s_all...), ds = vcat(ds_all...))
end

"""
    get_boundary_curves_with_ignored(domain::D) where D<:AbsSimpleDomain → boundary::Vector{AbsCurve}

Returns the connected boundary curves of `domain` used for full-boundary sampling,
including both `SpecularReflection` curves and curves marked with
`QuantumSolverIgnore`.

## Description
This differs from `get_boundary_curves`, which only retains `SpecularReflection`
curves used for constructing the solver matrices: `QuantumSolverIgnore` curves are
excluded there but are needed here so that quantities such as arc length and
boundary points, via [`boundary_coords`](@ref), can be computed over the entire
physical boundary of the domain.

## Arguments
* `domain`: A simple domain whose boundary curves are collected.

## Returns
* `boundary`: The connected vector of boundary curves.
"""
function _connected_physical_or_ignored_curves(boundary::Vector{AbsCurve})
    is_outer(crv) = (typeof(crv.bc) <: SpecularReflection || typeof(crv.bc) <: QuantumSolverIgnore)
    physical = filter(is_outer, boundary)
    return connect_curves(physical)
end

function get_boundary_curves_with_ignored(domain::D) where D<:AbsSimpleDomain
    return _connected_physical_or_ignored_curves(domain.boundary)
end


"""
    get_boundary_curves_with_ignored(domain::D) where D<:AbsMultiplyConnectedDomain → boundary::Vector{AbsCurve}

Returns the connected boundary curves of a multiply connected domain (e.g.
[`AnnularBilliard`](@ref)'s fundamental domain), including both
`SpecularReflection` and `QuantumSolverIgnore` curves. See
[`get_boundary_curves_with_ignored`](@ref) for details.

## Arguments
* `domain`: A multiply connected domain whose boundary curves are collected.

## Returns
* `boundary`: The connected vector of boundary curves.
"""
function get_boundary_curves_with_ignored(domain::D) where D<:AbsMultiplyConnectedDomain
    return _connected_physical_or_ignored_curves(domain.boundary)
end


"""
    get_boundary_curves_with_ignored(composite_domain::D) where D<:AbsCompositeDomain → boundary::Vector{AbsCurve}

Returns the connected boundary curves gathered over all subdomains of
`composite_domain`, including both `SpecularReflection` and `QuantumSolverIgnore`
curves. See [`get_boundary_curves_with_ignored`](@ref) for details.

## Arguments
* `composite_domain`: A composite domain whose subdomains' boundary curves are collected.

## Returns
* `boundary`: The connected vector of boundary curves.
"""
function get_boundary_curves_with_ignored(composite_domain::D) where D<:AbsCompositeDomain
    boundary = Vector{AbsCurve}()
    for domain in composite_domain.subdomains
        subboundary = get_boundary_curves(domain)
        append!(boundary,subboundary)
    end
    return connect_curves(boundary)
end

"""
    get_boundary_curves_with_ignored(billiard::B) where B<:AbsBilliard → boundary::Vector{AbsCurve}

Returns the connected boundary curves of the fundamental domain of `billiard`,
including both `SpecularReflection` and `QuantumSolverIgnore` curves. See
[`get_boundary_curves_with_ignored`](@ref) for details.

## Arguments
* `billiard`: The billiard whose fundamental domain's boundary curves are collected.

## Returns
* `boundary`: The connected vector of boundary curves.
"""
function get_boundary_curves_with_ignored(billiard::B) where B<:AbsBilliard
    return get_boundary_curves_with_ignored(billiard.fundamental_domain)
end

"""
    random_interior_points(billiard::Bi, N::Int; grd::Int = 1000) where {Bi<:AbsBilliard} → pts::Vector{SVector{2,T}}

Generates `N` points sampled uniformly (by rejection) from the interior of
`billiard`, for use e.g. by [`ParticularSolutionsMethod`](@ref)'s
[`evaluate_points`](@ref) methods.

## Arguments
* `billiard`: The billiard whose interior is sampled.
* `N`: The number of interior points to generate.

## Keyword arguments
* `grd::Int = 1000`: Target sampling density used to determine the bounding box, see [`boundary_limits`](@ref).

## Returns
* `pts`: A `Vector{SVector{2,T}}` of `N` interior points.
"""
function random_interior_points(billiard::Bi, N::Int; grd::Int=1000) where {Bi<:AbsBilliard}
    xlim, ylim = boundary_limits(get_boundary_curves(billiard); grd=grd)
    dx = xlim[2] - xlim[1]
    dy = ylim[2] - ylim[1]
    T = typeof(dx)
    pts = Vector{SVector{2,T}}(undef, N)
    n = 0
    while n < N
        x = dx*rand() + xlim[1]
        y = dy*rand() + ylim[1]
        pt = SVector(x, y)
        if is_inside(billiard, pt)
            n += 1
            pts[n] = pt
        end
    end
    return pts
end

"""
    estimate_rmin_rmax(pts::BoundaryPoints{T}, symmetry) where {T<:Real} → (rmin, rmax)

Estimates the minimum nonzero and maximum pairwise interaction radii between
the boundary points `pts`, used to tune the radial interpolation interval of
Chebyshev-accelerated Hankel/Bessel evaluation.

## Arguments
* `pts`: The boundary discretization whose pairwise interaction radii are estimated.
* `symmetry`: The discrete symmetry folding `pts` onto a fundamental domain, or `nothing`.

## Returns
* `(rmin, rmax)`: The minimum nonzero and maximum interaction radii found.
"""
function estimate_rmin_rmax(pts::BoundaryPoints{T}, ::Nothing) where {T<:Real}
    N = length(pts)
    xy = pts.xy
    tol2 = eps(T)^2
    rmin = Inf
    rmax = 0.0
    @inbounds for j in 2:N
        xj = xy[j]
        for i in 1:j-1
            xi = xy[i]
            dx = xi[1]-xj[1]
            dy = xi[2]-xj[2]
            d2 = muladd(dx, dx, dy*dy)
            d2 <= tol2 && continue
            r = sqrt(Float64(d2))
            rmin = min(rmin, r)
            rmax = max(rmax, r)
        end
    end
    isfinite(rmin) && rmax>0.0 || throw(ArgumentError("Unable to determine radial interval"))
    return rmin, rmax
end

function estimate_rmin_rmax(pts::BoundaryPoints{T}, symmetry) where {T<:Real}
    isnothing(symmetry) && return estimate_rmin_rmax(pts, nothing)
    orbits = symmetry_index_orbits(T, pts.xy, symmetry)
    fund = orbits.fundamental_indices
    m = fundamental_size(orbits)
    ng = orbit_size(orbits)
    xy = pts.xy
    tol2 = eps(T)^2
    rmin = Inf
    rmax = 0.0
    @inbounds for b in 1:m
        j = fund[b]
        for a in 1:m
            i = fund[a]
            xi = xy[i]
            for l in 1:ng
                q = orbits.fund_to_full[l,b]
                l==1 && i==j && continue
                xq = xy[q]
                dx = xi[1]-xq[1]
                dy = xi[2]-xq[2]
                d2 = muladd(dx, dx, dy*dy)
                d2 <= tol2 && continue
                r = sqrt(Float64(d2))
                rmin = min(rmin, r)
                rmax = max(rmax, r)
            end
        end
    end
    isfinite(rmin) && rmax>0.0 || throw(ArgumentError("Unable to determine radial interval"))
    return rmin, rmax
end