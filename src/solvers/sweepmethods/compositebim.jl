"""
    CompositeBIMSolver{T,CS,Sy} <: CFIE

`CompositeBIMSolver` is a concrete [`SweepBIMSolver`](@ref) for multiply
connected geometries whose connected boundary components require different
boundary-integral discretizations.

## Description
`CompositeBIMSolver` assigns one existing [`SweepBIMSolver`](@ref) component
solver (a [`DoubleLayerPotentialSolver`](@ref) or
[`CombinedFieldIntegralEquationSolver`](@ref)) to each connected physical
boundary component, assembling one globally coupled Fredholm operator. The
component solvers control only the discretization and same-component
Kress/grading quadrature of their own boundary component; inter-component
interactions are evaluated with ordinary Nyström quadrature. The first
component solver is interpreted as the outer boundary; the remaining
component solvers (`2:end`) are interpreted as holes and are
orientation-reversed after discretization.

## Attributes
* `component_solvers`: Tuple with one [`SweepBIMSolver`](@ref) per connected boundary component, outer boundary first.
* `symmetry`: Optional discrete symmetry shared by every component solver.
* `character`: Tuple of one-dimensional irrep character(s) requested for `symmetry` (trivial representation, `()`, by default), shared by every component solver.

## API
The following functions can be evaluated for this type:
- [`evaluate_points`](@ref)
- [`construct_matrices`](@ref)
- [`solve`](@ref)
- [`solve_vect`](@ref)
- [`solve_wavenumber`](@ref)
- [`k_sweep`](@ref)

!!! note "Migration status"
    API scaffold only (struct, constructor and method signatures), generalizing
    `QuantumBilliards-develop`'s `CFIE_kress_composite_solver` to both the
    [`DoubleLayerPotentialSolver`](@ref) and
    [`CombinedFieldIntegralEquationSolver`](@ref) families. The matrix-assembly
    bodies are not yet implemented; every method below raises an `error` until
    Step 2 of the migration plan lands.
"""
struct CompositeBIMSolver{T<:Real,CS<:Tuple,Sy<:Union{AbsSymmetry,Nothing},Ch<:Tuple} <: CFIE
    component_solvers::CS
    symmetry::Sy
    character::Ch
end

"""
    CompositeBIMSolver(component_solvers::SweepBIMSolver...) → solver::CompositeBIMSolver

Constructs a [`CompositeBIMSolver`](@ref) from one component solver per
connected boundary component, outer boundary first.

## Arguments
* `component_solvers`: One [`SweepBIMSolver`](@ref) instance per connected boundary component. Every component solver must share the same `symmetry` and `character`.

## Returns
* `solver`: A [`CompositeBIMSolver`](@ref) instance.
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

"""
    _bim_grid_scale(solver::CompositeBIMSolver) → scale::Real

`CompositeBIMSolver` has no `pts_scaling_factor` field of its own (see the
[`_bim_grid_scale`](@ref) generic fallback docstring); the outer boundary's
own scale (`component_solvers[1]`) is used as the default oversampling factor
for `wavefunction(state; b=:auto)`.
"""
_bim_grid_scale(solver::CompositeBIMSolver) = solver.component_solvers[1].pts_scaling_factor[1]

################################################################################
############### PRIVATE HELPERS: CONNECTED-COMPONENT BOOKKEEPING ##############
################################################################################

# Groups a flat physical-boundary curve list into connected components by
# curve `domain_id` (the same field `SimpleDomain.id`/each curve's
# `domain_id` already carries for multiply connected geometries), preserving
# first-seen domain_id order and within-group curve order. For every billiard
# currently in the package (all simply connected, uniform default
# `domain_id=1`), this returns a single group containing every curve,
# matching `length(component_solvers)==1`. Delegates to
# `BilliardGeometry`'s own domain_id grouping algorithm (used internally by
# `boundary_components`) rather than duplicating it; this fallback remains
# necessary (instead of always calling `boundary_components(billiard)`)
# because `comp` here may be `full_boundary(billiard)` (symmetry images
# included), which `boundary_components` does not account for.
_group_boundary_by_domain_id(comp::Vector) = BilliardGeometry._group_curves_by_domain_id(comp)

# Whether a connected boundary component group is a HOLE, using the curve's
# own `orientation` field as the sole signal -- the same `orientation=-1`
# marker `BilliardGeometry.MultiplyConnectedDomain`/`AnnularBilliard` already
# use for the "inside outer AND outside every hole" `is_inside` semantics --
# rather than the component's position in `groups` (component 1 is not
# assumed to be the outer boundary just because it comes first: two disjoint
# `orientation=1` boundaries, e.g. Step 7's two-copies smoke test, are both
# correctly treated as independent outer boundaries and neither is reversed).
# Every curve within one connected component must share the same
# orientation; a mixed-orientation component is rejected rather than
# silently guessed at.
function _group_is_hole(group::Vector)
    o = group[1].orientation
    all(c.orientation == o for c in group) || throw(ArgumentError("Curves within one connected boundary component must share the same `orientation` field (found a mix); mixed-orientation components are not supported"))
    o == 1 && return false
    o == -1 && return true
    throw(ArgumentError("Curve `orientation` must be ±1; found $o"))
end

# Dispatches boundary sampling of one connected component's curve group to the
# assigned component solver's own private per-component evaluate-points
# helper (unchanged from Steps 3/6), reused directly rather than duplicated.
_composite_component_points(cs::DoubleLayerPotentialSolver, group::Vector, k::T) where {T<:Real} = _dlp_evaluate_points(cs, cs.grading, group, k)
_composite_component_points(cs::CombinedFieldIntegralEquationSolver, group::Vector, k::T) where {T<:Real} = _cfie_evaluate_points(cs, cs.grading, group, k)

# Concatenates the per-component `BoundaryPoints` into one flat
# `BoundaryPoints`, so that the generic `SweepBIMSolver` infrastructure
# (`solve_state`/`_bim_normal_derivative`/`symmetrize_layer_density` in
# sweepmethods.jl, `BIMEigenstate.pts::BoundaryPoints{T}`) works for
# `CompositeBIMSolver` with no changes there. Arc length `s` is made globally
# continuous across components (via `boundary_s`) since downstream
# consumers (`_rellich`, `husimi_function`, `boundary_function`) integrate
# over the *entire* physical boundary. The per-point component index is
# additionally stashed in the otherwise BIM-unused `w_dm` field (reserved
# for the decomposition method, never populated by any BIM solver) purely as
# a private bookkeeping channel so `construct_matrices` can recover the
# per-component block structure from the single merged `BoundaryPoints` it
# receives — see `_composite_offsets`.
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
    return BoundaryPoints(xy; normal=normal, s=s, ds=ds, tangent=tangent, tangent_2=tangent_2,
                          ts=ts, tphys=tphys, ws=ws, ws_der=ws_der, w_dm=compidx, compid=1, is_periodic=true)
end

# Recovers the `[1, 1+N₁, 1+N₁+N₂, ...]` block offsets from a merged
# `BoundaryPoints`' privately-stashed `w_dm` per-point component index (see
# `_merge_composite_points`). Points of a given component are contiguous by
# construction, so a single linear scan suffices.
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

# Reconstructs one component's own `BoundaryPoints` (needed by
# `boundary_geom_cache`/`kress_R!`/the DLP/CFIE kernel-entry helpers) from a
# contiguous index range of the merged `BoundaryPoints`.
function _composite_component_slice(pts::BoundaryPoints{T}, rng::UnitRange{Int}, compid::Int) where {T<:Real}
    z = SVector{2,T}(zero(T), zero(T))
    return BoundaryPoints(pts.xy[rng], pts.tangent[rng], pts.tangent_2[rng], pts.ts[rng], pts.tphys[rng],
                          pts.ws[rng], pts.ws_der[rng], pts.s[rng], pts.ds[rng], compid, true, z, z, z, z)
end

# Global boundary index → (component index, component-local index) maps.
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

################################################################################
################## PRIVATE HELPERS: FREDHOLM MATRIX ASSEMBLY ##################
################################################################################

# Dispatches the same-component Kress-corrected kernel entry (the raw D(k)
# or D(k)+ikS(k) value, not yet subtracted from the identity) to the
# assigned component solver's own kernel-entry helper (Steps 3/6, reused
# unchanged).
@inline _composite_component_kernel_entry(::DoubleLayerPotentialSolver, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real} = _dlp_kernel_entry(pts, Rmat, G, k, i, j)
@inline _composite_component_kernel_entry(::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real} = _cfie_kernel_entry(pts, Rmat, G, k, i, j)

# Smooth (no Kress log-splitting needed: source and target never coincide
# across different connected components) cross-component double-layer
# kernel entry between an observation point (xi,yi) in some component `a`
# and source node `j` of component `b`, dispatched on `b`'s own solver
# kernel type. Matches the "cross-block" term of `-develop`'s
# `CFIE_kress_composite_solver` reference assembly, specialized to a pure
# double layer for a `DoubleLayerPotentialSolver` source component.
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

# Same as above, combined-field (D(k)+ikS(k)) cross-component kernel entry
# for a `CombinedFieldIntegralEquationSolver` source component.
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

# Full (unfolded) composite Fredholm matrix: same-component diagonal blocks
# reuse the Kress-corrected DLP/CFIE kernels unchanged; cross-component
# blocks use the smooth kernel above (no singular splitting needed).
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

# Symmetry-reduced composite Fredholm matrix, folding the complete
# discrete full-boundary composite kernel over each source symmetry orbit
# (mirrors `_dlp_fredholm_reduced!`/`_cfie_fredholm_reduced!`'s image-list
# folding, generalized to same-/cross-component kernel dispatch).
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
    evaluate_points(solver::CompositeBIMSolver, billiard::Bi, k) where {Bi<:AbsBilliard} → pts::BoundaryPoints

Samples every connected boundary component of `billiard` with its assigned
component solver, concatenating the results (holes orientation-reversed) into
one composite [`BoundaryPoints`](@ref) discretization.

## Description
Connected boundary components are identified from the physical boundary
curves' `domain_id` (the same field distinguishing subdomains of a
multiply connected [`BilliardGeometry.AbsCompositeDomain`](@ref)), grouping
in first-seen order. Whether a component is treated as a hole is decided by
its curves' own `orientation` field (`orientation=-1`, the same marker
[`BilliardGeometry.MultiplyConnectedDomain`](@ref)/[`BilliardGeometry.AnnularBilliard`](@ref)
already use for the "inside outer AND outside every hole" `is_inside`
semantics), not by component position — a component whose curves all have
`orientation=1` is discretized as-is (an outer/independent physical
boundary) regardless of whether it comes first or last in `groups`, while a
component whose curves all have `orientation=-1` (a hole) has its curves
reversed (both order and per-curve parametrization, via
`BilliardGeometry._reverse_curve`) before discretization, so that the
outward normal at every hole boundary point off `pts` points into the hole
rather than into the physical domain (matching the sign convention already
used by [`BilliardGeometry.full_boundary`](@ref) for orientation-reversing
symmetry images). Every curve within one connected component must share the
same `orientation` (see [`_group_is_hole`](@ref)).
"""
function evaluate_points(solver::CompositeBIMSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver)
    kT = T(k)
    comp = solver.symmetry === nothing ? get_boundary_curves(billiard) : full_boundary(billiard)
    isempty(comp) && error("Boundary cannot be empty.")
    nc = length(solver.component_solvers)
    domain = billiard.fundamental_domain
    if domain isa AbsMultiplyConnectedDomain
        nc == 1 + genus(domain) || throw(ArgumentError("Billiard's fundamental domain has genus $(genus(domain)) (requires $(1+genus(domain)) component solver(s)) but CompositeBIMSolver has $nc component solver(s)"))
    end
    groups = solver.symmetry === nothing && domain isa AbsMultiplyConnectedDomain ? boundary_components(billiard) : _group_boundary_by_domain_id(comp)
    length(groups) == nc || throw(ArgumentError("Billiard boundary has $(length(groups)) connected component(s) (grouped by curve domain_id) but CompositeBIMSolver has $nc component solver(s)"))
    comp_pts = Vector{BoundaryPoints{T}}(undef, nc)
    @inbounds for a in 1:nc
        grp = groups[a]
        group = _group_is_hole(grp) ? [BilliardGeometry._reverse_curve(c) for c in reverse(grp)] : grp
        comp_pts[a] = _composite_component_points(solver.component_solvers[a], group, kT)
    end
    return _merge_composite_points(comp_pts)
end


"""
    boundary_matrix_size(solver::CompositeBIMSolver, pts::BoundaryPoints) → N::Int

Returns the dimension of the assembled composite Fredholm matrix, accounting
for any symmetry-orbit folding onto a fundamental domain.
"""
function boundary_matrix_size(solver::CompositeBIMSolver, pts::BoundaryPoints)
    solver.symmetry === nothing && return boundary_matrix_size(pts)
    T = _bim_numeric_type(solver)
    # Orbit membership/fundamental_size depends only on the symmetry group's
    # permutation structure, never on the requested irrep character, so the
    # trivial-representation call here is intentional (Step 16.1 audit).
    return fundamental_size(symmetry_index_orbits(T, pts.xy, solver.symmetry))
end

"""
    construct_matrices(solver::CompositeBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool = true) → A::Matrix{Complex}

Assembles the globally coupled composite Fredholm matrix `A(k)`.

## Description
`pts` (as produced by [`evaluate_points`](@ref)) is split back into its
per-component discretizations via the block offsets recovered from its
per-point component bookkeeping (see `_composite_offsets`). Each component's
own [`BoundaryGeomCache`](@ref)/Kress correction matrix is built exactly as
in [`DoubleLayerPotentialSolver`](@ref)/[`CombinedFieldIntegralEquationSolver`](@ref);
same-component blocks reuse those solvers' Kress-corrected kernels unchanged,
while cross-component blocks use the smooth (non-singular) kernel between
different components' nodes.
"""
function construct_matrices(solver::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}
    @timeit_debug "construct_matrices" begin
        kT = _bim_widen_k(T, k)
        nc = length(solver.component_solvers)
        N = length(pts)
        @debug "Composite BIM matrix construction started" N kT nc symmetry=solver.symmetry character=solver.character
        offs = _composite_offsets(pts, nc)
        comp_pts = [_composite_component_slice(pts, offs[a]:offs[a+1]-1, a) for a in 1:nc]
        Gs = Vector{BoundaryGeomCache{T}}(undef, nc)
        Rmats = Vector{Matrix{T}}(undef, nc)
        @timeit_debug "boundary_geom_cache" begin
            @inbounds for a in 1:nc
                graded = _is_nontrivial_dlp_grading(comp_pts[a])
                Gs[a] = boundary_geom_cache(comp_pts[a], graded)
                Na = length(comp_pts[a])
                Ra = zeros(T, Na, Na)
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
                orbits = _fold_boundary(T, pts.xy, solver.symmetry, solver.character)
            end
            m = fundamental_size(orbits)
            g2c, g2l = _composite_global_to_local(offs)
            A = Matrix{Complex{T}}(undef, m, m)
            @timeit_debug "reduced_fredholm_assembly" begin
                _composite_fredholm_reduced!(A, solver, comp_pts, Gs, Rmats, offs, g2c, g2l, orbits, kT; multithreaded)
            end
            @debug "Symmetry-reduced composite Fredholm matrix assembled" fundamental_size=m
            return A
        end
    end
end

"""
    solve(solver::CompositeBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool = true, use_krylov::Bool = true) → t::Real

Computes the composite tension at wavenumber `k`.
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
    solve_vect(solver::CompositeBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool = true) → (t::Real, x::Vector)

Computes the composite tension and the associated boundary density eigenvector
at wavenumber `k`.
"""
function solve_vect(solver::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}
    A = construct_matrices(solver, pts, k; multithreaded)
    @blas_1 vals, _, rvecs, _ = KrylovKit.svdsolve(A, 1, :SR)
    return vals[1], Vector{Complex{T}}(rvecs[1])
end
