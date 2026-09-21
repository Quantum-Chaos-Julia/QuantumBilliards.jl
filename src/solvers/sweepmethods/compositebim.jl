################################################################################
# COMPOSITE BOUNDARY INTEGRAL METHOD
#
# Each connected component is assigned its own DLP or CFIE component solver, so
# different components may use different boundary parametrizations, grading
# strategies, and integral formulations while remaining coupled through one
# Fredholm operator.
#
# Let the physical boundary be the disjoint union
#
#                       ∂Ω = Γ₁ ∪ Γ₂ ∪ ... ∪ Γₙ.
#
# Writing μ=(μ₁,...,μₙ) for the component layer densities, the composite
# Fredholm equation has the block form
#
#                            A(k) μ = 0,
#
# where the diagonal block A_aa is determined by the solver assigned to Γₐ:
#
#                 A_aa(k) = I - D_aa(k)                 (DLP),
#
#                 A_aa(k) = I - (D_aa(k)+ikS_aa(k))     (CFIE).
#
# For a≠b, source and target points lie on distinct connected components and
# the corresponding kernels are smooth. The off-diagonal blocks therefore use
# direct Nyström quadrature,
#
#                 A_ab(k) = -D_ab(k)                    (DLP source),
#
#                 A_ab(k) = -(D_ab(k)+ikS_ab(k))         (CFIE source).
#
# The source component determines whether a cross-component block contributes
# only the double-layer kernel or the combined double-/single-layer kernel.
#
# BOUNDARY DISCRETIZATION
# Each connected component retains the boundary discretization and Kress
# quadrature scheme of its assigned component solver. Thus all same-component
# interactions are evaluated with the corresponding DLP or CFIE Kress
# quadrature scheme, including any smooth, single-corner, or global corner
# grading of that component. Interactions between distinct components are
# nonsingular and are evaluated directly with ordinary Nyström quadrature.
################################################################################

"""
    CompositeBIMSolver{T,CS,Sy,Ch} <: CFIE

Boundary-integral eigensolver for geometries with multiple connected physical
boundary components.

`CompositeBIMSolver` assigns one [`DoubleLayerPotentialSolver`](@ref) or
[`CombinedFieldIntegralEquationSolver`](@ref) to each connected boundary
component and assembles the resulting discretizations into one globally
coupled Fredholm operator.

Each diagonal block retains the Kress quadrature scheme and grading strategy
of its assigned component solver. Interactions between distinct connected
components are smooth and are evaluated directly with ordinary Nyström
quadrature. A source component using DLP contributes a double-layer
cross-component kernel, whereas a source component using CFIE contributes the
combined `D(k)+ikS(k)` kernel.

Hole components are identified from their boundary orientation and reversed
before discretization so that all component normals follow the physical
outward-normal convention. An optional common discrete symmetry reduces the
resulting full-boundary composite Fredholm operator to a selected symmetry
sector.

## Attributes
* `component_solvers::CS`: Tuple containing one DLP or CFIE solver for each connected physical boundary component.
* `symmetry::Sy`: Optional discrete symmetry shared by all component solvers.
* `character::Ch`: Character tuple selecting the common representation of `symmetry`.

## API
[`evaluate_points`](@ref), [`boundary_matrix_size`](@ref),
[`construct_matrices`](@ref), [`solve`](@ref), [`solve_vect`](@ref),
[`solve_wavenumber`](@ref), and [`k_sweep`](@ref).
"""
struct CompositeBIMSolver{T<:Real,CS<:Tuple,Sy<:Union{AbsSymmetry,Nothing},Ch<:Tuple} <: CFIE
    component_solvers::CS
    symmetry::Sy
    character::Ch
end

"""
    CompositeBIMSolver(component_solvers::SweepBIMSolver...)

Construct a composite boundary-integral solver from one component solver per
connected physical boundary component.

The supplied component solvers may independently use DLP or CFIE formulations
and their own boundary grading strategies. All component solvers must,
however, use the same discrete symmetry and character because symmetry
reduction is applied to the globally coupled full-boundary operator.

## Arguments
* `component_solvers::SweepBIMSolver...`: Component solvers assigned in the same order as the connected boundary components returned during boundary decomposition.

## Returns
* `solver::CompositeBIMSolver`: Configured composite boundary-integral solver.
"""
function CompositeBIMSolver(component_solvers::Vararg{SweepBIMSolver})
    isempty(component_solvers) && throw(ArgumentError("CompositeBIMSolver requires at least one component solver"))
    symmetry = component_solvers[1].symmetry
    character = component_solvers[1].character
    all(cs -> cs.symmetry == symmetry && cs.character == character, component_solvers) || throw(ArgumentError("All component solvers passed to CompositeBIMSolver must share the same symmetry and character"))
    T = _bim_numeric_type(component_solvers[1])
    return CompositeBIMSolver{T,typeof(component_solvers),typeof(symmetry),typeof(character)}(component_solvers, symmetry, character)
end

_bim_numeric_type(solver::CompositeBIMSolver{T}) where {T} = T
# for grid size, mostly legacy now
_bim_grid_scale(solver::CompositeBIMSolver) = solver.component_solvers[1].pts_scaling_factor[1]

# Group a flat boundary curve list into connected components according
# to each curve's `domain_id`. The first occurrence of each domain identifier
# determines component order, while the original curve order is preserved
# inside each component. This delegates to BilliardGeometry's own grouping
# convention so that the composite solver uses the same component decomposition
# as the underlying domain representation.
_group_boundary_by_domain_id(comp::Vector) = BilliardGeometry._group_curves_by_domain_id(comp)

# Determine whether a connected boundary component represents a hole from the
# orientation carried by its curves. Orientation +1 denotes an ordinary outer
# boundary and orientation -1 denotes a hole. Every curve belonging to one
# connected component must have the same orientation; mixed orientations are
# rejected.
function _group_is_hole(group::Vector)
    o = group[1].orientation
    all(c.orientation == o for c in group) || throw(ArgumentError("Curves within one connected boundary component must share the same `orientation` field (found a mix); mixed-orientation components are not supported"))
    o == 1 && return false
    o == -1 && return true
    throw(ArgumentError("Curve `orientation` must be ±1; found $o"))
end

# Dispatch boundary sampling of one connected component to its assigned DLP or
# CFIE solver. Each component therefore retains exactly the grading strategy,
# resolution rule, and Kress-compatible boundary parametrization implemented by
# its native solver rather than introducing a separate composite discretization.
_composite_component_points(cs::DoubleLayerPotentialSolver, group::Vector, k::T) where {T<:Real} = _dlp_evaluate_points(cs, cs.grading, group, k)
_composite_component_points(cs::CombinedFieldIntegralEquationSolver, group::Vector, k::T) where {T<:Real} = _cfie_evaluate_points(cs, cs.grading, group, k)

# Merge the independently discretized connected components into one flat
# BoundaryPoints object representing the complete physical boundary. All
# geometric and quadrature arrays are concatenated component by component,
# while `boundary_s` constructs a globally continuous boundary coordinate for
# downstream full-boundary operations such as boundary phase-space representations.
#
# The merged tangent determines the outward normal through the common
# orientation convention n=(t_y,-t_x)/|t|. Hole components have already 
# had their oriented tangent data flipped without
# changing their node ordering, so the same formula produces the physical
# outward normal on both outer and inner boundaries while preserving the
# canonical indexing required by symmetry reduction.
#
# The component number of every point is stored privately in `w_dm`, which is
# otherwise unused by BIM discretizations. This bookkeeping allows later matrix
# assembly to reconstruct the original block decomposition from the single
# BoundaryPoints object without changing the generic SweepBIMSolver interface.
# Points belonging to each component remain contiguous in the merged arrays.
function _merge_composite_points(comp_pts::Vector{BoundaryPoints{T}}) where {T<:Real}
    N = boundary_matrix_size(comp_pts)
    xy = Vector{SVector{2,T}}(undef, N)
    tangent = Vector{SVector{2,T}}(undef, N)
    tangent_2 = Vector{SVector{2,T}}(undef, N)
    ts = Vector{T}(undef, N)
    tphys = Vector{T}(undef, N)
    ws = Vector{T}(undef, N)
    ws_der = Vector{T}(undef, N)
    ds = Vector{T}(undef, N)
    compidx = Vector{T}(undef, N)
    s = boundary_s(comp_pts)
    p = 1
    @inbounds for (a, comp) in enumerate(comp_pts)
        n = length(comp)
        rng = p:p+n-1
        xy[rng] .= comp.xy
        tangent[rng] .= comp.tangent
        tangent_2[rng] .= comp.tangent_2
        ts[rng] .= comp.ts
        tphys[rng] .= comp.tphys
        ws[rng] .= comp.ws
        ws_der[rng] .= comp.ws_der
        ds[rng] .= comp.ds
        compidx[rng] .= T(a)
        p += n
    end
    normal = Vector{SVector{2,T}}(undef, N)
    @inbounds for i in 1:N
        tx, ty = tangent[i]
        sp = hypot(tx, ty)
        normal[i] = SVector{2,T}(ty/sp, -tx/sp)
    end
    return BoundaryPoints(xy; normal=normal, s=s, ds=ds, tangent=tangent, tangent_2=tangent_2, ts=ts, tphys=tphys, ws=ws, ws_der=ws_der, w_dm=compidx, compid=1, is_periodic=true)
end

# Recover the contiguous component-block offsets from the per-point component
# indices stored in `pts.w_dm` by `_merge_composite_points`. For component
# sizes N₁,...,Nₙ, the returned vector has the form
#
#                    [1, 1+N₁, 1+N₁+N₂, ..., N+1].
#
# The scan simultaneously verifies that component labels occur consecutively
# and in the expected order.
function _composite_offsets(pts::BoundaryPoints{T}, nc::Int) where {T<:Real}
    N = length(pts)
    length(pts.w_dm) == N || error("CompositeBIMSolver.construct_matrices requires pts to originate from evaluate_points(::CompositeBIMSolver, ...) (missing per-point component bookkeeping)")
    offs = Vector{Int}(undef, nc+1)
    offs[1] = 1
    a = 1
    @inbounds for i in 1:N
        cid = round(Int, pts.w_dm[i])
        if cid != a
            cid == a+1 || error("pts.w_dm component indices are inconsistent with $nc component solvers")
            offs[a+1] = i
            a += 1
        end
    end
    a == nc || error("pts.w_dm component indices are inconsistent with $nc component solvers")
    offs[nc+1] = N+1
    return offs
end

# Reconstruct the BoundaryPoints representation of one connected component from
# its contiguous range in the merged full-boundary discretization. The original
# parameter coordinates, derivatives, Kress quadrature weights, physical
# arclength weights, and geometry are preserved. The reconstructed component
# can therefore be passed directly to `boundary_geom_cache`, `kress_R!`, and
# the native DLP/CFIE same-component kernel-entry routines.
function _composite_component_slice(pts::BoundaryPoints{T}, rng::UnitRange{Int}, compid::Int) where {T<:Real}
    z = SVector{2,T}(zero(T), zero(T))
    return BoundaryPoints(pts.xy[rng], pts.tangent[rng], pts.tangent_2[rng], pts.ts[rng], pts.tphys[rng], pts.ws[rng], pts.ws_der[rng], pts.s[rng], pts.ds[rng], compid, true, z, z, z, z)
end

# Build maps from each global full-boundary node index to its connected
# component index and component-local node index. If global node g belongs to
# component a at local position j, the returned arrays satisfy
#
#                         g2c[g] = a,
#                         g2l[g] = j.
#
# These maps are required by symmetry reduction because a symmetry orbit is
# expressed in global boundary indices, whereas same- and cross-component
# kernel evaluations require component-local indices.
function _composite_global_to_local(offs::Vector{Int})
    Ntot = offs[end]-1
    g2c = Vector{Int}(undef, Ntot)
    g2l = Vector{Int}(undef, Ntot)
    @inbounds for a in 1:length(offs)-1
        off = offs[a]
        for j in 1:(offs[a+1]-offs[a])
            g2c[off+j-1] = a
            g2l[off+j-1] = j
        end
    end
    return g2c, g2l
end

"""
    _combine_composite_orbits(::Type{T}, local_orbits::Vector{SymmetryOrbitMap{T}}) where {T<:Real} → orbits::SymmetryOrbitMap{T}

Combine exact component-local symmetry orbit maps into one global composite-boundary orbit map.
Each CompositeBIMSolver boundary block is folded independently with the
existing exact symmetry-index machinery. The resulting component-local full
and reduced indices are shifted into the concatenated composite indexing and
assembled into one ordinary SymmetryOrbitMap.

## Arguments
* `T`: Real scalar type of the boundary discretization.
* `local_orbits::Vector{SymmetryOrbitMap{T}}`: Exact orbit map of each CompositeBIMSolver boundary block.

## Returns
* `orbits::SymmetryOrbitMap{T}`: Exact global orbit map for the concatenated composite boundary.
"""
function _combine_composite_orbits(::Type{T}, local_orbits::Vector{SymmetryOrbitMap{T}}) where {T<:Real}
    isempty(local_orbits) && throw(ArgumentError("Cannot combine an empty collection of symmetry orbit maps"))
    ng = orbit_size(local_orbits[1])
    all(o -> orbit_size(o)==ng,local_orbits) || throw(ArgumentError("All CompositeBIMSolver boundary blocks must have the same symmetry-orbit size"))
    N = sum(full_size,local_orbits); m = sum(fundamental_size,local_orbits)
    fundamental_indices = Vector{Int}(undef,m); orbit_of = Vector{Int}(undef,N); phase = Vector{Complex{T}}(undef,N)
    fund_to_full = Matrix{Int}(undef,ng,m); fund_to_scale = Matrix{Complex{T}}(undef,ng,m)
    full_off = 0; fund_off = 0
    @inbounds for o in local_orbits
        Na = full_size(o); ma = fundamental_size(o)
        frng = full_off+1:full_off+Na; mrng = fund_off+1:fund_off+ma
        fundamental_indices[mrng] .= o.fundamental_indices .+ full_off
        orbit_of[frng] .= o.orbit_of .+ fund_off
        phase[frng] .= o.phase
        fund_to_full[:,mrng] .= o.fund_to_full .+ full_off
        fund_to_scale[:,mrng] .= o.fund_to_scale
        full_off += Na; fund_off += ma
    end
    return SymmetryOrbitMap{T}(fundamental_indices,orbit_of,phase,N,m,fund_to_full,fund_to_scale)
end

"""
    _composite_symmetry_orbits(::Type{T}, solver::CompositeBIMSolver, pts::BoundaryPoints{T}) where {T<:Real} → orbits::SymmetryOrbitMap{T}

Construct the exact symmetry orbit map of a CompositeBIMSolver boundary.

The merged boundary is separated into the contiguous blocks created by
`_merge_composite_points`. Each block is folded using `_fold_boundary`,
 so every exact integer symmetry permutation is evaluated with that block's 
own periodic node count rather than the total node count of the concatenated 
composite boundary. The component-local orbit maps are then combined
algebraically into one global SymmetryOrbitMap. 

This construction requires each CompositeBIMSolver boundary block to be
invariant under the selected symmetry. Symmetry-related disconnected
components represented inside one block require a separate Composite-specific
component-permutation extension.

No geometric point matching or floating-point tolerances are used.

## Arguments
* `T`: Real scalar type of the boundary discretization.
* `solver::CompositeBIMSolver`: Composite solver defining the common symmetry and character.
* `pts::BoundaryPoints{T}`: Merged boundary discretization produced by `evaluate_points`.

## Returns
* `orbits::SymmetryOrbitMap{T}`: Exact global symmetry orbit map.
"""
function _composite_symmetry_orbits(::Type{T}, solver::CompositeBIMSolver, pts::BoundaryPoints{T}) where {T<:Real}
    solver.symmetry === nothing && throw(ArgumentError("_composite_symmetry_orbits requires a nontrivial symmetry"))
    nc = length(solver.component_solvers); offs = _composite_offsets(pts,nc)
    local_orbits = Vector{SymmetryOrbitMap{T}}(undef,nc)
    @inbounds for a in 1:nc
        rng = offs[a]:offs[a+1]-1
        local_orbits[a] = _fold_boundary(T,@view(pts.xy[rng]),solver.symmetry,solver.character)
    end
    return _combine_composite_orbits(T,local_orbits)
end

# Dispatch a same-component kernel entry to the solver assigned to that
# connected component. A DLP source component returns D_ij, whereas a CFIE
# source component returns (D+ikS)_ij.
# These routines return the boundary kernel itself rather than the complete
# Fredholm entry; the identity contribution and overall minus sign are applied
# by the composite matrix assembly.
@inline _composite_component_kernel_entry(::DoubleLayerPotentialSolver, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real} = _dlp_kernel_entry(pts, Rmat, G, k, i, j)
@inline _composite_component_kernel_entry(::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real} = _cfie_kernel_entry(pts, Rmat, G, k, i, j)

# Evaluate a smooth double-layer interaction from a source node on one
# connected component to a target point on a different component. Since
# distinct connected components cannot contain the same boundary point
# no logarithmic splitting or Kress singular quadrature is required.
@inline function _composite_cross_kernel_entry(::DoubleLayerPotentialSolver, pb::BoundaryPoints{T}, xi::T, yi::T, k::Union{T,Complex{T}}, j::Int) where {T<:Real}
    xj, yj = pb.xy[j]
    dx = xi-xj
    dy = yi-yj
    r = hypot(dx, dy)
    invr = inv(r)
    tx, ty = pb.tangent[j]
    inn = ty*dx - tx*dy
    h1 = _bim_hankelh1(1, k*r)
    return pb.ws[j]*im*k/2*inn*h1*invr
end

# Evaluate a smooth combined-field interaction from a CFIE source component to
# a target point on a different connected component. As for the DLP cross
# interaction, source and target cannot coincide, so the kernels are evaluated
# directly without Kress logarithmic splitting.
#
# The double-layer contribution is
#
#            D = ws_j (ik/2) inner H₁⁽¹⁾(kr)/r,
#
# while the single-layer contribution uses the source speed |γ'(t_j)|,
#
#            S = ws_j (i/2) H₀⁽¹⁾(kr) |γ'(t_j)|.
@inline function _composite_cross_kernel_entry(::CombinedFieldIntegralEquationSolver, pb::BoundaryPoints{T}, xi::T, yi::T, k::Union{T,Complex{T}}, j::Int) where {T<:Real}
    xj, yj = pb.xy[j]
    dx = xi-xj
    dy = yi-yj
    r = hypot(dx, dy)
    invr = inv(r)
    tx, ty = pb.tangent[j]
    inn = ty*dx - tx*dy
    sj = hypot(tx, ty)
    ik = im*k
    h0 = _bim_hankelh1(0, k*r)
    h1 = _bim_hankelh1(1, k*r)
    dval = pb.ws[j]*im*k/2*inn*h1*invr
    sval = pb.ws[j]*Complex{T}(0, one(T)/2)*h0*sj
    return dval + ik*sval
end

# Assemble the full, unfolded composite Fredholm matrix in connected-component
# block form. Each diagonal block corresponds to interactions whose source and
# target lie on the same component. These blocks reuse the assigned component
# solver's native Kress quadrature scheme exactly, including its DLP or CFIE
# kernel, grading, logarithmic splitting, and analytic diagonal limits.
#
# For two distinct components a≠b, the corresponding off-diagonal block is
# smooth and is assembled by direct Nyström quadrature. The solver assigned to
# the source component b determines the operator in that block: a DLP source
# contributes D_ab, while a CFIE source contributes D_ab+ikS_ab.
#
# Consequently the globally coupled matrix has the block structure
#
#                 A_aa = I - K_aa,
#                 A_ab =   - K_ab,       a ≠ b,
function _composite_fredholm_full!(A::AbstractMatrix{Complex{T}}, solver::CompositeBIMSolver, comp_pts::Vector{BoundaryPoints{T}}, Gs::Vector{BoundaryGeomCache{T}}, Rmats::Vector{Matrix{T}}, offs::Vector{Int}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    fill!(A, zero(Complex{T}))
    nc = length(comp_pts)
    @inbounds for a in 1:nc
        cs = solver.component_solvers[a]
        pa = comp_pts[a]
        Ga = Gs[a]
        Ra = Rmats[a]
        Na = length(pa)
        off = offs[a]
        for i in 1:Na
            gi = off+i-1
            A[gi,gi] = one(Complex{T}) - _composite_component_kernel_entry(cs, pa, Ra, Ga, k, i, i)
        end
        @use_threads multithreading=(multithreaded && Na>=32) for j in 2:Na
            gj = off+j-1
            @inbounds for i in 1:j-1
                gi = off+i-1
                A[gi,gj] = -_composite_component_kernel_entry(cs, pa, Ra, Ga, k, i, j)
                A[gj,gi] = -_composite_component_kernel_entry(cs, pa, Ra, Ga, k, j, i)
            end
        end
    end
    for b in 1:nc
        csb = solver.component_solvers[b]
        pb = comp_pts[b]
        offb = offs[b]
        Nb = length(pb)
        for a in 1:nc
            a == b && continue
            pa = comp_pts[a]
            offa = offs[a]
            Na = length(pa)
            @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                gi = offa+i-1
                xi, yi = pa.xy[i]
                @inbounds for j in 1:Nb
                    gj = offb+j-1
                    A[gi,gj] = -_composite_cross_kernel_entry(csb, pb, xi, yi, k, j)
                end
            end
        end
    end
    return A
end

# Assemble the symmetry-reduced composite Fredholm matrix by folding the
# complete physical-boundary operator over source symmetry orbits. Symmetry
# acts only after the full composite discretization has been defined, so each
# orbit may contain source nodes belonging to different connected components.
#
# For fundamental target index fund[a] and source orbit b,
#
#        A_ab = δ_ab - Σ_{j: orbit_of[j]=b} phase[j] K_{fund[a],j},
#
# where K is evaluated according to the connected components containing the
# target and source nodes. If both nodes belong to the same component, the
# assigned component solver's Kress-discretized DLP or CFIE kernel is used.
# Otherwise the corresponding smooth cross-component kernel is evaluated by
# ordinary Nyström quadrature.
function _composite_fredholm_reduced!(A::AbstractMatrix{Complex{T}}, solver::CompositeBIMSolver, comp_pts::Vector{BoundaryPoints{T}}, Gs::Vector{BoundaryGeomCache{T}}, Rmats::Vector{Matrix{T}}, offs::Vector{Int}, g2c::Vector{Int}, g2l::Vector{Int}, orbits::SymmetryOrbitMap{T}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    m = fundamental_size(orbits)
    N = length(orbits)
    fund = orbits.fundamental_indices
    orbit_of = orbits.orbit_of
    phase = orbits.phase
    images = [Int[] for _ in 1:m]
    @inbounds for j in 1:N
        push!(images[orbit_of[j]], j)
    end
    fill!(A, zero(Complex{T}))
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        @inbounds for a in 1:m
            gi = fund[a]
            ca = g2c[gi]
            ia = g2l[gi]
            acc = zero(Complex{T})
            for gj in images[b]
                cb = g2c[gj]
                jb = g2l[gj]
                ph = phase[gj]
                if ca == cb
                    cs = solver.component_solvers[ca]
                    acc += ph*_composite_component_kernel_entry(cs, comp_pts[ca], Rmats[ca], Gs[ca], k, ia, jb)
                else
                    csb = solver.component_solvers[cb]
                    xi, yi = comp_pts[ca].xy[ia]
                    acc += ph*_composite_cross_kernel_entry(csb, comp_pts[cb], xi, yi, k, jb)
                end
            end
            A[a,b] = -acc
        end
        A[b,b] += one(Complex{T})
    end
    return A
end

"""
    _flip_component_orientation(pts::BoundaryPoints{T}) where {T<:Real} → pts_flipped::BoundaryPoints{T}

Flips the orientation of one boundary component without changing its node ordering.

## Description
For multiply connected geometries, the outer boundary and hole boundaries must
carry opposite orientations. The boundary nodes and their parameter ordering
are preserved exactly so that symmetry index maps remain unchanged.

The first parametrization derivative changes sign under orientation reversal,
while the second derivative retains its sign. All scalar parameter, grading,
arc-length, and quadrature data remain in their original ordering.

The returned [`BoundaryPoints`](@ref) object reconstructs its normal and
curvature data from the flipped tangent orientation.

## Arguments
* `pts::BoundaryPoints{T}`: Boundary component whose orientation is flipped.

## Returns
* `pts_flipped::BoundaryPoints{T}`: New boundary discretization with opposite orientation and unchanged node indexing.
"""
function _flip_component_orientation(pts::BoundaryPoints{T}) where {T<:Real}
    tangent = -pts.tangent
    tL = -pts.tL; tR = -pts.tR
    return BoundaryPoints(pts.xy,tangent,pts.tangent_2,pts.ts,pts.tphys,pts.ws,pts.ws_der,pts.s,pts.ds,pts.compid,pts.is_periodic,pts.xL,pts.xR,tL,tR)
end

"""
    evaluate_points(solver::CompositeBIMSolver, billiard::Bi, k) where {Bi<:AbsBilliard}

Construct the complete composite boundary discretization of `billiard`.

The physical boundary is separated into connected components and each
component is sampled with its assigned DLP or CFIE component solver. Components
with orientation `-1` are interpreted as holes and their orientation is flipped
after sampling without changing the boundary-node ordering. This preserves the
canonical indexing required by exact symmetry index maps while giving hole
boundaries the opposite normal orientation required by the boundary-integral
formulation. For multiply connected fundamental domains, the number of component 
solvers must equal the number of connected boundary components. 

## Arguments
* `solver::CompositeBIMSolver`: Composite solver containing one DLP or CFIE solver per connected boundary component.
* `billiard::Bi`: Billiard whose physical boundary is discretized.
* `k`: Wavenumber used by the component solvers to determine their boundary resolutions.

## Returns
* `pts::BoundaryPoints`: Merged complete physical-boundary discretization containing all component geometry, quadrature data, and internal component bookkeeping.
"""
function evaluate_points(solver::CompositeBIMSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver); kT = T(k)
    base = get_boundary_curves(billiard)
    comp = solver.symmetry===nothing ? base : full_boundary(billiard)
    isempty(comp) && error("Boundary cannot be empty.")
    groups = _group_boundary_by_domain_id(comp); base_groups = _group_boundary_by_domain_id(base)
    nc = length(solver.component_solvers)
    length(groups)==nc || throw(ArgumentError("Billiard boundary has $(length(groups)) component group(s) but CompositeBIMSolver has $nc component solver(s)"))
    base_by_id = Dict(first(g).domain_id=>g for g in base_groups)
    comp_pts = Vector{BoundaryPoints{T}}(undef,nc)
    @inbounds for a in 1:nc
        grp = groups[a]; id = first(grp).domain_id
        haskey(base_by_id,id) || throw(ArgumentError("Expanded Composite boundary contains domain_id=$id which is absent from the fundamental boundary"))
        p = _composite_component_points(solver.component_solvers[a],grp,kT)
        comp_pts[a] = _group_is_hole(base_by_id[id]) ? _flip_component_orientation(p) : p
    end
    return _merge_composite_points(comp_pts)
end

# Returns the dimension of the assembled composite Fredholm matrix, accounting for any symmetry-orbit folding onto a fundamental domain.
function boundary_matrix_size(solver::CompositeBIMSolver, pts::BoundaryPoints)
    solver.symmetry === nothing && return boundary_matrix_size(pts)
    T = _bim_numeric_type(solver)
    return fundamental_size(_composite_symmetry_orbits(T,solver,pts))
end

"""
    construct_matrices(solver::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}

Assemble the globally coupled composite Fredholm matrix at wavenumber `k`.

The merged boundary discretization is first separated into its original
connected-component blocks. For every component, the geometry cache and Kress
quadrature matrix are reconstructed from that component's own discretization.

Same-component blocks retain the complete Kress quadrature scheme of their
assigned DLP or CFIE solver. Cross-component blocks are nonsingular and are
evaluated directly with ordinary Nyström quadrature. A DLP source component
contributes its double-layer kernel, while a CFIE source component contributes
the combined `D(k)+ikS(k)` kernel.

If `solver.symmetry` is present, this complete composite operator is folded
over source symmetry orbits with the prescribed character phases.

## Arguments
* `solver::CompositeBIMSolver{T}`: Composite solver defining the component integral formulations and common symmetry sector.
* `pts::BoundaryPoints{T}`: Merged complete physical-boundary discretization produced by [`evaluate_points`](@ref).
* `k`: Real or complex wavenumber at which the composite Fredholm operator is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.

## Returns
* `A::Matrix{Complex{T}}`: Full or symmetry-reduced globally coupled composite Fredholm matrix.
"""
function construct_matrices(solver::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}
    @timeit_debug "construct_matrices" begin
        kT = _bim_widen_k(T,k)
        nc = length(solver.component_solvers); N = length(pts)
        @debug "Composite BIM matrix construction started" N kT nc symmetry = solver.symmetry character = solver.character
        offs = _composite_offsets(pts,nc)
        comp_pts = [_composite_component_slice(pts,offs[a]:offs[a+1]-1,a) for a in 1:nc]
        Gs = Vector{BoundaryGeomCache{T}}(undef, nc); Rmats = Vector{Matrix{T}}(undef,nc)
        @timeit_debug "boundary_geom_cache" begin
            @inbounds for a in 1:nc
                graded = _is_nontrivial_dlp_grading(comp_pts[a])
                Gs[a] = boundary_geom_cache(comp_pts[a], graded)
                Na = length(comp_pts[a]); Ra = zeros(T, Na, Na)
                kress_R!(Ra)
                Rmats[a] = Ra
            end
        end
        if solver.symmetry === nothing
            A = Matrix{Complex{T}}(undef, N, N)
            @timeit_debug "fredholm_assembly" begin
                _composite_fredholm_full!(A, solver, comp_pts, Gs, Rmats, offs, kT; multithreaded)
            end
            @debug "Composite Fredholm matrix assembled" size=size(A)
            return A
        else
            @timeit_debug "symmetry_orbits" begin
                orbits = _composite_symmetry_orbits(T, solver, pts)
            end
            m = fundamental_size(orbits); g2c, g2l = _composite_global_to_local(offs)
            A = Matrix{Complex{T}}(undef, m, m)
            @timeit_debug "reduced_fredholm_assembly" begin
                _composite_fredholm_reduced!(A, solver, comp_pts, Gs, Rmats, offs, g2c, g2l, orbits, kT; multithreaded)
            end
            @debug "Symmetry-reduced composite Fredholm matrix assembled" fundamental_size = m
            return A
        end
    end
end

"""
    solve(solver::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true, use_krylov::Bool=true) where {T<:Real}

Compute the composite BIM tension at wavenumber `k`:

    t(k) = σmin(A(k)).

With `use_krylov = true`, the smallest singular value is computed iteratively
with KrylovKit. Otherwise, all singular values are computed with a dense SVD.

## Arguments
* `solver::CompositeBIMSolver{T}`: Composite solver defining the component discretizations and common symmetry sector.
* `pts::BoundaryPoints{T}`: Merged complete physical-boundary discretization.
* `k`: Wavenumber at which the tension is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.
* `use_krylov::Bool = true`: Compute only the smallest singular value iteratively instead of using a full dense SVD.

## Returns
* `t::Real`: Smallest singular value `σmin(A(k))` of the globally coupled composite Fredholm operator.
"""
function solve(solver::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true, use_krylov::Bool=true) where {T<:Real}
    A = construct_matrices(solver, pts, k; multithreaded)
    if use_krylov
        @blas_1 vals, _, _, _ = KrylovKit.svdsolve(A, 1, :SR)
        return vals[1]
    else
        @blas_multi_then_1 MAX_BLAS_THREADS s = svdvals(A)
        return s[end]
    end
end

"""
    solve_vect(solver::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}

Compute the composite BIM tension and corresponding layer density `x` at
wavenumber `k`.

At a Dirichlet eigenvalue the vector approximates a null vector of the
globally coupled Fredholm operator,

    A(k)x ≈ 0.

## Arguments
* `solver::CompositeBIMSolver{T}`: Composite solver defining the component discretizations and common symmetry sector.
* `pts::BoundaryPoints{T}`: Merged complete physical-boundary discretization.
* `k`: Wavenumber at which the layer density is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.

## Returns
* `t::Real`: Smallest singular value `σmin(A(k))` of the globally coupled composite Fredholm operator.
* `x::Vector{Complex{T}}`: Right singular vector representing the discrete component-wise layer density associated with `t`.
"""
function solve_vect(solver::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}
    A = construct_matrices(solver, pts, k; multithreaded)
    @blas_1 vals, _, rvecs, _ = KrylovKit.svdsolve(A, 1, :SR)
    return vals[1], Vector{Complex{T}}(rvecs[1])
end
