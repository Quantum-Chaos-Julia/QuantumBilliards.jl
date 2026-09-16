"""
    DoubleLayerPotentialSolver{T,G,Sy} <: SweepBIMSolver

`DoubleLayerPotentialSolver` is a concrete [`SweepBIMSolver`](@ref) implementing
the (optionally Kress-corrected) direct boundary integral method for computing
quantum billiard spectra from the Helmholtz double-layer Fredholm operator.

## Description
The assembled Fredholm operator is

    A(k) = I - D(k),

where `D(k)` denotes the Nyström discretization of the interior Helmholtz
double-layer operator. The tension at a fixed wavenumber `k` is a function of
the smallest singular value / nullspace residual of `A(k)` (see
[`construct_matrices`](@ref), [`solve`](@ref)). The boundary discretization
strategy (uniform periodic vs. Kress-graded around corners) is controlled by
`grading`, see [`BoundaryGrading`](@ref); an optional discrete `symmetry`
folds the discretization onto a fundamental domain via a
[`BilliardGeometry.SymmetryOrbitMap`](@ref).

## Attributes
* `pts_scaling_factor`: Vector of scaling factors, one per fundamental boundary curve, used to determine the number of boundary sampling points.
* `min_pts`: Minimum number of boundary sampling points per component.
* `grading`: [`BoundaryGrading`](@ref) strategy used to discretize the boundary.
* `symmetry`: Optional `AbsSymmetry` used to fold the discretization onto a fundamental domain.
* `character`: Tuple of one-dimensional irrep character(s) requested for `symmetry` (trivial representation, `()`, by default), passed as the trailing `character` argument(s) of [`BilliardGeometry.symmetry_index_orbits`](@ref) via [`_fold_boundary`](@ref).
* `eps`: Relative tolerance used to determine the tension from the smallest singular value / nullspace residual.

## API
The following functions can be evaluated for this type:
- [`evaluate_points`](@ref)
- [`boundary_matrix_size`](@ref)
- [`construct_matrices`](@ref)
- [`solve`](@ref)
- [`solve_vect`](@ref)
- [`solve_wavenumber`](@ref)
- [`k_sweep`](@ref)
"""
struct DoubleLayerPotentialSolver{T<:Real,G<:BoundaryGrading,Sy<:Union{AbsSymmetry,Nothing},Ch<:Tuple} <: SweepBIMSolver
    pts_scaling_factor::Vector{T}
    min_pts::Int64
    grading::G
    symmetry::Sy
    character::Ch
    eps::T
end

"""
    DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}}; min_pts::Int = 200, grading::BoundaryGrading = SmoothPeriodicGrading(), symmetry::Union{Nothing,AbsSymmetry} = nothing, character::Tuple = (), eps::T = T(1e-15)) where {T<:Real} → solver::DoubleLayerPotentialSolver{T}

Constructs a [`DoubleLayerPotentialSolver`](@ref).

## Arguments
* `pts_scaling_factor`: Scaling factor, or vector thereof (one per fundamental boundary curve), used to determine the number of boundary sampling points.

## Keyword arguments
* `min_pts::Int = 200`: Minimum number of boundary sampling points per component.
* `grading::BoundaryGrading = SmoothPeriodicGrading()`: Boundary discretization/grading strategy, see [`BoundaryGrading`](@ref).
* `symmetry::Union{Nothing,AbsSymmetry} = nothing`: Optional discrete symmetry used to fold the discretization onto a fundamental domain.
* `character::Tuple = ()`: One-dimensional irrep character(s) requested for `symmetry` (trivial representation by default); unvalidated against `symmetry`, see the `(pts_scaling_factor, billiard, sector::SymmetrySector)` constructor overload below for a validated alternative.
* `eps::T = T(1e-15)`: Relative tolerance used to determine the tension.

## Returns
* `solver`: A [`DoubleLayerPotentialSolver{T}`](@ref) instance.
"""
function DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}}; min_pts::Int=200,
                                     grading::BoundaryGrading=SmoothPeriodicGrading(),
                                     symmetry::Union{Nothing,AbsSymmetry}=nothing,
                                     character::Tuple=(),
                                     eps::T=T(1e-15)) where {T<:Real}
    bs = pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    return DoubleLayerPotentialSolver{T,typeof(grading),typeof(symmetry),typeof(character)}(bs, min_pts, grading, symmetry, character, eps)
end

"""
    DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector; kwargs...) where {T<:Real} → solver::DoubleLayerPotentialSolver{T}

Constructs a [`DoubleLayerPotentialSolver`](@ref) whose `symmetry`/`character`
are resolved from a validated [`SymmetrySector`](@ref) built against
`billiard` ([`_resolve_bim_symmetry`](@ref)), instead of the bare
`symmetry=`/`character=` keywords above. Mirrors
[`RealPlaneWaves`](@ref)`(dim, billiard, sector)`'s recommended, validated
construction pattern (Step 15 of the migration plan). Supports every
representation `RealPlaneWaves` supports for reflection symmetries, plus
`NFoldRotation` sectors, which `RealPlaneWaves` does not. The one
irreducible restriction: a lone `XYAxisReflection` character cannot
determine a unique representation of a `D2` group on its own — specify the
individual axis reflections instead (e.g. `symmetry_sector(billiard,
XAxisReflection=>χx, YAxisReflection=>χy)`); see
[`_resolve_bim_symmetry`](@ref) for why.
"""
function DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector; kwargs...) where {T<:Real}
    generator, character = _resolve_bim_symmetry(billiard, sector)
    return DoubleLayerPotentialSolver(pts_scaling_factor; symmetry=generator, character=character, kwargs...)
end

_bim_numeric_type(::DoubleLayerPotentialSolver{T}) where {T} = T

################################################################################
################### PRIVATE HELPERS: BOUNDARY EVALUATION #####################
################################################################################

# Maps a global periodic component parameter t ∈ [0,2π) to arc length along
# the composite boundary component `comp` (proportional parametrization).
function _dlp_composite_arclength(comp::Vector, t::T) where {T<:Real}
    _, cum, Ltot = component_lengths(comp)
    twopi = 2*T(pi)
    τ = mod(t, twopi)
    target = Ltot*τ/twopi
    offset = zero(T)
    @inbounds for j in eachindex(comp)
        Lj = T(comp[j].length)
        if target < offset+Lj || j == lastindex(comp)
            u = clamp((target-offset)/Lj, zero(T), one(T))
            return offset + arc_length(comp[j], u)
        end
        offset += Lj
    end
    return Ltot
 end

# Ungraded periodic Nyström discretization of a smooth (single-curve or
# smooth-composite) boundary component, sampled at Kress midpoint nodes
# σ_j = 2π(j-1/2)/N. Used directly by SmoothPeriodicGrading, and as the
# fallback for GlobalCornerGrading when no true corners are detected.
function _dlp_evaluate_points(solver::DoubleLayerPotentialSolver, ::SmoothPeriodicGrading, comp::Vector, k::T) where {T<:Real}
    twopi = 2*T(pi)
    _, _, Ltot = component_lengths(comp)
    scale = solver.pts_scaling_factor[1]
    N = max(solver.min_pts, round(Int, k*Ltot*scale/twopi))
    needed = solver.symmetry === nothing ? 2 : lcm(2, symmetry_node_multiple(solver.symmetry))
    N = cld(N, needed)*needed
    h = twopi/T(N)
    ts = T[s_mid(j, N) for j in 1:N]
    tphys = copy(ts)
    xy = Vector{SVector{2,T}}(undef, N)
    tangent_1st = Vector{SVector{2,T}}(undef, N)
    tangent_2nd = Vector{SVector{2,T}}(undef, N)
    s = Vector{T}(undef, N)
    ds = Vector{T}(undef, N)
    @inbounds for i in 1:N
        q, γt, γtt = BilliardGeometry._eval_composite_geom_global_t(T, comp, tphys[i])
        xy[i] = q
        tangent_1st[i] = γt
        tangent_2nd[i] = γtt
        s[i] = _dlp_composite_arclength(comp, tphys[i])
        ds[i] = hypot(γt[1], γt[2])*h
    end
    ws = fill(h, N)
    ws_der = ones(T, N)
    z = SVector{2,T}(zero(T), zero(T))
    return BoundaryPoints(xy, tangent_1st, tangent_2nd, ts, tphys, ws, ws_der, s, ds, 1, true, z, z, z, z)
end

# Globally Kress-graded Nyström discretization of a piecewise-smooth boundary
# component with true corners at the (already detected) global periodic
# parameter locations `corners`.
function _dlp_evaluate_points_graded(solver::DoubleLayerPotentialSolver, grading::GlobalCornerGrading, comp::Vector, k::T, corners::Vector{T}) where {T<:Real}
    twopi = 2*T(pi)
    _, _, Ltot = component_lengths(comp)
    scale = solver.pts_scaling_factor[1]
    N = max(solver.min_pts, round(Int, k*Ltot*scale/twopi))
    needed = solver.symmetry === nothing ? 2 : symmetry_node_multiple(solver.symmetry)
    N = cld(N, needed)*needed
    σ, tmap, jac, jac2, _ = multi_kress_graded_nodes_data(T, N, corners; q=grading.kressq, minsep_tol=T(grading.min_t_spacing))
    tphys = tmap
    xy = Vector{SVector{2,T}}(undef, N)
    tangent_1st = Vector{SVector{2,T}}(undef, N)
    tangent_2nd = Vector{SVector{2,T}}(undef, N)
    s = Vector{T}(undef, N)
    ds = Vector{T}(undef, N)
    h = twopi/T(N)
    @inbounds for i in 1:N
        q, γt, γtt = BilliardGeometry._eval_composite_geom_global_t(T, comp, tphys[i])
        tangent_1st[i] = γt*jac[i]
        tangent_2nd[i] = γtt*(jac[i]^2) + γt*jac2[i]
        xy[i] = q
        s[i] = _dlp_composite_arclength(comp, tphys[i])
        v = tangent_1st[i]
        ds[i] = hypot(v[1], v[2])*h
    end
    ws = fill(h, N)
    ws_der = jac
    z = SVector{2,T}(zero(T), zero(T))
    return BoundaryPoints(xy, tangent_1st, tangent_2nd, σ, tphys, ws, ws_der, s, ds, 1, true, z, z, z, z)
end

# GlobalCornerGrading: detect true corners and grade globally around them;
# falls back to the ungraded smooth discretization when none are detected.
function _dlp_evaluate_points(solver::DoubleLayerPotentialSolver, grading::GlobalCornerGrading, comp::Vector, k::T) where {T<:Real}
    corners = BilliardGeometry._component_corner_locations(T, comp)
    isempty(corners) && return _dlp_evaluate_points(solver, SmoothPeriodicGrading(), comp, k)
    return _dlp_evaluate_points_graded(solver, grading, comp, k, corners)
end

################################################################################
################## PRIVATE HELPERS: FREDHOLM MATRIX ASSEMBLY ##################
################################################################################

# `true` when `pts` carries a nontrivial (Kress-graded) reparametrization
# Jacobian, i.e. `pts.ws_der` deviates from all ones.
@inline function _is_nontrivial_dlp_grading(pts::BoundaryPoints{T}) where {T<:Real}
    length(pts.ws_der) == length(pts) || return false
    return maximum(abs.(pts.ws_der .- one(T))) > sqrt(eps(T))
end

# Full (unfolded) Kress-corrected Nyström Fredholm matrix F(k) = I - D(k).
# Off-diagonal entries: D[i,j] = Rmat[i,j]*l1 + ws[j]*l2, with
# l1 = -(k/2π)*inner[i,j]*J1(k r)/r, l2 = (ik/2)*inner[i,j]*H1(k r)/r - l1*logterm[i,j].
function _dlp_fredholm_full!(F::AbstractMatrix{Complex{T}}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    invtwopi = inv(2*T(pi))
    αL1 = -k*invtwopi
    αL2 = im*k/2
    N = length(pts)
    fill!(F, zero(Complex{T}))
    @inbounds for i in 1:N
        F[i,i] = one(Complex{T}) - Complex{T}(pts.ws[i]*G.kappa[i], zero(T))
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        @inbounds for i in 1:j-1
            r = G.R[i,j]
            invr = G.invR[i,j]
            lt = G.logterm[i,j]
            inn_ij = G.inner[i,j]
            inn_ji = G.inner[j,i]
            h1 = _bim_hankelh1(1, k*r)
            j1 = _bim_besselj(1, k*r, h1)
            l1_ij = αL1*inn_ij*j1*invr
            l2_ij = αL2*inn_ij*h1*invr - l1_ij*lt
            F[i,j] = -(Rmat[i,j]*l1_ij + pts.ws[j]*l2_ij)
            l1_ji = αL1*inn_ji*j1*invr
            l2_ji = αL2*inn_ji*h1*invr - l1_ji*lt
            F[j,i] = -(Rmat[j,i]*l1_ji + pts.ws[i]*l2_ji)
        end
    end
    return F
end

# Single Kress-corrected DLP kernel entry D[i,j] at full-boundary indices
# (i,j), used by the symmetry-reduced assembly below (no i/j-pair sharing of
# the Hankel evaluation is possible there, exactly as in the reference
# reduced assembly).
@inline function _dlp_kernel_entry(pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real}
    i == j && return Complex{T}(pts.ws[i]*G.kappa[i], zero(T))
    invtwopi = inv(2*T(pi))
    r = G.R[i,j]
    invr = G.invR[i,j]
    lt = G.logterm[i,j]
    inn = G.inner[i,j]
    h1 = _bim_hankelh1(1, k*r)
    j1 = _bim_besselj(1, k*r, h1)
    l1 = -k*invtwopi*inn*j1*invr
    l2 = im*k/2*inn*h1*invr - l1*lt
    return Rmat[i,j]*l1 + pts.ws[j]*l2
end

# Symmetry-reduced Kress-corrected Fredholm matrix, folding the complete
# discrete full-boundary Kress operator over each source symmetry orbit:
# Fred[a,b] = δ_{ab} - Σ_{j: orbit_of[j]=b} phase[j]*D[fund[a],j].
function _dlp_fredholm_reduced!(F::AbstractMatrix{Complex{T}}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, orbits::SymmetryOrbitMap{T}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    m = fundamental_size(orbits)
    N = length(orbits)
    fund = orbits.fundamental_indices
    orbit_of = orbits.orbit_of
    phase = orbits.phase
    images = [Int[] for _ in 1:m]
    @inbounds for j in 1:N
        push!(images[orbit_of[j]], j)
    end
    fill!(F, zero(Complex{T}))
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        @inbounds for a in 1:m
            i = fund[a]
            acc = zero(Complex{T})
            for j in images[b]
                acc += phase[j]*_dlp_kernel_entry(pts, Rmat, G, k, i, j)
            end
            F[a,b] = -acc
        end
        F[b,b] += one(Complex{T})
    end
    return F
end

"""
    evaluate_points(solver::DoubleLayerPotentialSolver, billiard::Bi, k) where {Bi<:AbsBilliard} → pts::BoundaryPoints

Samples the boundary of `billiard` according to `solver.grading`, producing the
boundary discretization needed to assemble the double-layer Fredholm matrix in
[`construct_matrices`](@ref).

## Description
When `solver.symmetry === nothing`, only the fundamental domain's own
physical boundary ([`get_boundary_curves`](@ref)) is discretized — this
already is the complete physical boundary in that case. When a `symmetry` is
set, the *complete* physical boundary ([`full_boundary`](@ref)) is
discretized instead, since [`symmetry_index_orbits`](@ref) folds a full
periodic boundary sampling onto the fundamental domain by exact index
permutation and therefore needs every symmetry image present in `pts`.
"""
function evaluate_points(solver::DoubleLayerPotentialSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver)
    comp = full_boundary(billiard)
    isempty(comp) && error("Boundary cannot be empty.")
    return _dlp_evaluate_points(solver, solver.grading, comp, T(k))
end

"""
    boundary_matrix_size(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints) → N::Int

Returns the dimension of the assembled Fredholm matrix, accounting for any
symmetry-orbit folding onto a fundamental domain.
"""
function boundary_matrix_size(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints)
    solver.symmetry === nothing && return boundary_matrix_size(pts)
    T = _bim_numeric_type(solver)
    # Orbit membership/fundamental_size depends only on the symmetry group's
    # permutation structure, never on the requested irrep character, so the
    # trivial-representation call here is intentional (Step 16.1 audit).
    return fundamental_size(symmetry_index_orbits(T, pts.xy, solver.symmetry))
end

"""
    construct_matrices(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool = true) → A::Matrix{Complex}

Assembles the double-layer Fredholm matrix `A(k) = I - D(k)`.
"""
function construct_matrices(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)
    @timeit_debug "construct_matrices" begin
        T = _bim_numeric_type(solver)
        kT = _bim_widen_k(T, k)
        N = length(pts)
        @debug "DLP matrix construction started" N kT symmetry=solver.symmetry character=solver.character
        graded = _is_nontrivial_dlp_grading(pts)
        @timeit_debug "boundary_geom_cache" begin
            G = boundary_geom_cache(pts, graded)
        end
        Rmat = zeros(T, N, N)
        @timeit_debug "kress_R" begin
            kress_R!(Rmat)
        end
        @debug "Geometry cache and Kress matrix built" N graded
        if solver.symmetry === nothing
            A = Matrix{Complex{T}}(undef, N, N)
            @timeit_debug "fredholm_assembly" begin
                _dlp_fredholm_full!(A, pts, Rmat, G, kT; multithreaded)
            end
            @debug "DLP Fredholm matrix assembled" size=size(A)
            return A
        else
            @timeit_debug "symmetry_orbits" begin
                orbits = _fold_boundary(T, pts.xy, solver.symmetry, solver.character)
            end
            m = fundamental_size(orbits)
            A = Matrix{Complex{T}}(undef, m, m)
            @timeit_debug "reduced_fredholm_assembly" begin
                _dlp_fredholm_reduced!(A, pts, Rmat, G, orbits, kT; multithreaded)
            end
            @debug "Symmetry-reduced DLP Fredholm matrix assembled" fundamental_size=m
            return A
        end
    end
end

"""
    solve(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool = true, use_krylov::Bool = true) → t::Real

Computes the double-layer tension at wavenumber `k`, defined from the smallest
singular value / Krylov nullspace residual of `A(k)` (see
[`construct_matrices`](@ref)).
"""
function solve(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true, use_krylov::Bool=true)
    T = _bim_numeric_type(solver)
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
    solve_vect(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool = true) → (t::Real, x::Vector)

Computes the double-layer tension and the associated boundary density
eigenvector at wavenumber `k`.
"""
function solve_vect(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)
    T = _bim_numeric_type(solver)
    A = construct_matrices(solver, pts, k; multithreaded)
    @blas_1 vals, _, rvecs, _ = KrylovKit.svdsolve(A, 1, :SR)
    return vals[1], Vector{Complex{T}}(rvecs[1])
end

# `symmetrize_layer_density`/`solve_state`/`_bim_normal_derivative` (the
# boundary-normal-derivative and BIMEigenstate support shared by every
# SweepBIMSolver) live in sweepmethods.jl, not here: none of that logic is
# specific to the double-layer kernel (see sweepmethods.jl's docstrings for
# why the same weighted-transpose reciprocity applies generically).