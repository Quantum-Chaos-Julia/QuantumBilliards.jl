"""
    CombinedFieldIntegralEquationSolver{T,G,Sy,Ch} <: CFIE

`CombinedFieldIntegralEquationSolver` is a concrete [`SweepBIMSolver`](@ref)
implementing the (optionally Kress-corrected) combined-field boundary integral
method for computing quantum billiard spectra from the coupled Helmholtz
double-layer/single-layer Fredholm operator.

## Description
The assembled Fredholm operator is

    A(k) = I - (D(k) + i k S(k)),

where `D(k)` is the Nyström discretization of the Helmholtz double-layer
operator and `S(k)` of the single-layer operator. The combined-field
coupling removes the spurious interior-nullspace problem that a pure
double-layer formulation ([`DoubleLayerPotentialSolver`](@ref)) can suffer on
some geometries. The tension at a fixed wavenumber `k` is a function of the
smallest singular value / nullspace residual of `A(k)` (see
[`construct_matrices`](@ref), [`solve`](@ref)). The boundary discretization
strategy is controlled by `grading`, see [`BoundaryGrading`](@ref)
(`CFIE` additionally supports [`CornerGrading`](@ref), a single-curve
parametric-corner variant that [`DoubleLayerPotentialSolver`](@ref) does not
need); an optional discrete `symmetry` folds the discretization onto a
fundamental domain via a [`BilliardGeometry.SymmetryOrbitMap`](@ref).

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
struct CombinedFieldIntegralEquationSolver{T<:Real,G<:BoundaryGrading,Sy<:Union{AbsSymmetry,Nothing},Ch<:Tuple} <: CFIE
    pts_scaling_factor::Vector{T}
    min_pts::Int64
    grading::G
    symmetry::Sy
    character::Ch
    eps::T
end

"""
    CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}}; min_pts::Int = 200, grading::BoundaryGrading = SmoothPeriodicGrading(), symmetry::Union{Nothing,AbsSymmetry} = nothing, character::Tuple = (), eps::T = T(1e-15)) where {T<:Real} → solver::CombinedFieldIntegralEquationSolver{T}

Constructs a [`CombinedFieldIntegralEquationSolver`](@ref).

## Arguments
* `pts_scaling_factor`: Scaling factor, or vector thereof (one per fundamental boundary curve), used to determine the number of boundary sampling points.

## Keyword arguments
* `min_pts::Int = 200`: Minimum number of boundary sampling points per component.
* `grading::BoundaryGrading = SmoothPeriodicGrading()`: Boundary discretization/grading strategy, see [`BoundaryGrading`](@ref).
* `symmetry::Union{Nothing,AbsSymmetry} = nothing`: Optional discrete symmetry used to fold the discretization onto a fundamental domain.
* `character::Tuple = ()`: One-dimensional irrep character(s) requested for `symmetry` (trivial representation by default); unvalidated against `symmetry`, see the `(pts_scaling_factor, billiard, sector::SymmetrySector)` constructor overload below for a validated alternative.
* `eps::T = T(1e-15)`: Relative tolerance used to determine the tension.

## Returns
* `solver`: A [`CombinedFieldIntegralEquationSolver{T}`](@ref) instance.
"""
function CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}}; min_pts::Int=200,
                                              grading::BoundaryGrading=SmoothPeriodicGrading(),
                                              symmetry::Union{Nothing,AbsSymmetry}=nothing,
                                              character::Tuple=(),
                                              eps::T=T(1e-15)) where {T<:Real}
    bs = pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    return CombinedFieldIntegralEquationSolver{T,typeof(grading),typeof(symmetry),typeof(character)}(bs, min_pts, grading, symmetry, character, eps)
end

"""
    CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector; kwargs...) where {T<:Real} → solver::CombinedFieldIntegralEquationSolver{T}

Constructs a [`CombinedFieldIntegralEquationSolver`](@ref) whose
`symmetry`/`character` are resolved from a validated [`SymmetrySector`](@ref)
built against `billiard` ([`_resolve_bim_symmetry`](@ref)), mirroring
[`DoubleLayerPotentialSolver`](@ref)`(pts_scaling_factor, billiard, sector)`.
See that constructor's docstring for the one irreducible restriction (a lone
`XYAxisReflection` character).
"""
function CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::BilliardGeometry.AbsBilliard, sector::SymmetrySector; kwargs...) where {T<:Real}
    generator, character = _resolve_bim_symmetry(billiard, sector)
    return CombinedFieldIntegralEquationSolver(pts_scaling_factor; symmetry=generator, character=character, kwargs...)
end

_bim_numeric_type(::CombinedFieldIntegralEquationSolver{T}) where {T} = T

################################################################################
################### PRIVATE HELPERS: BOUNDARY EVALUATION #####################
################################################################################

# Ungraded periodic Nyström discretization of a smooth (single-curve or
# smooth-composite) boundary component, sampled at Kress midpoint nodes
# σ_j = 2π(j-1/2)/N. Used directly by SmoothPeriodicGrading, and as the
# fallback for GlobalCornerGrading when no true corners are detected.
# Shares `_dlp_composite_arclength` (dlp.jl) since this is a purely geometric
# operation independent of the Fredholm kernel.
function _cfie_evaluate_points(solver::CombinedFieldIntegralEquationSolver, ::SmoothPeriodicGrading, comp::Vector, k::T) where {T<:Real}
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

# CornerGrading: single closed curve (comp of length 1) whose own [0,1)
# parametrization has a known corner at t=0, Kress-graded via a fixed
# single-corner map (no corner detection needed, unlike GlobalCornerGrading).
function _cfie_evaluate_points(solver::CombinedFieldIntegralEquationSolver, grading::CornerGrading, comp::Vector, k::T) where {T<:Real}
    length(comp) == 1 || error("CornerGrading requires the boundary to be represented by a single closed curve with its own parametric corner; use GlobalCornerGrading for composite/piecewise-smooth boundaries.")
    crv = comp[1]
    twopi = 2*T(pi)
    L = T(crv.length)
    scale = solver.pts_scaling_factor[1]
    N = max(solver.min_pts, round(Int, k*L*scale/twopi))
    needed = solver.symmetry === nothing ? 2 : lcm(2, symmetry_node_multiple(solver.symmetry))
    N = cld(N, needed)*needed
    σ, tmap, jac, jac2, _ = kress_graded_nodes_data(T, N; q=grading.kressq, minsep_tol=T(grading.min_t_spacing))
    tphys = tmap./twopi
    xy = curve(crv, tphys)
    γu = tangent(crv, tphys)
    γuu = tangent_2(crv, tphys)
    tangent_1st = Vector{SVector{2,T}}(undef, N)
    tangent_2nd = Vector{SVector{2,T}}(undef, N)
    @inbounds for i in 1:N
        a = jac[i]/twopi
        b = jac2[i]/twopi
        tangent_1st[i] = γu[i]*a
        tangent_2nd[i] = γuu[i]*a^2 + γu[i]*b
    end
    s = arc_length(crv, tphys)
    h = twopi/T(N)
    ds = Vector{T}(undef, N)
    @inbounds for i in 1:N
        v = tangent_1st[i]
        ds[i] = hypot(v[1], v[2])*h
    end
    ws = fill(h, N)
    z = SVector{2,T}(zero(T), zero(T))
    return BoundaryPoints(xy, tangent_1st, tangent_2nd, σ, tphys, ws, jac, s, ds, 1, true, z, z, z, z)
end

# Globally Kress-graded Nyström discretization of a piecewise-smooth boundary
# component with true corners at the (already detected) global periodic
# parameter locations `corners`.
function _cfie_evaluate_points_graded(solver::CombinedFieldIntegralEquationSolver, grading::GlobalCornerGrading, comp::Vector, k::T, corners::Vector{T}) where {T<:Real}
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
function _cfie_evaluate_points(solver::CombinedFieldIntegralEquationSolver, grading::GlobalCornerGrading, comp::Vector, k::T) where {T<:Real}
    corners = BilliardGeometry._component_corner_locations(T, comp)
    isempty(corners) && return _cfie_evaluate_points(solver, SmoothPeriodicGrading(), comp, k)
    return _cfie_evaluate_points_graded(solver, grading, comp, k, corners)
end

################################################################################
################## PRIVATE HELPERS: FREDHOLM MATRIX ASSEMBLY ##################
################################################################################

# Full (unfolded) Kress-corrected Nyström CFIE Fredholm matrix
# A(k) = I - (D(k) + ikS(k)). The D(k) piece is identical to the DLP kernel
# (see `_dlp_fredholm_full!` in dlp.jl); the additional S(k) single-layer
# term uses the H(0,·)/H(1,·) Hankel pair with its own Kress logarithmic
# splitting (diagonal self-term includes the Euler-gamma correction).
function _cfie_fredholm_full!(F::AbstractMatrix{Complex{T}}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    invtwopi = inv(2*T(pi))
    αL1 = -k*invtwopi
    αL2 = im*k/2
    αM1 = -invtwopi
    αM2 = Complex{T}(0, one(T)/2)
    ik = im*k
    euler_over_pi = T(Base.MathConstants.eulergamma)/T(pi)
    N = length(pts)
    fill!(F, zero(Complex{T}))
    @inbounds for i in 1:N
        si = G.speed[i]
        wi = pts.ws[i]
        dval = Complex{T}(wi*G.kappa[i], zero(T))
        m1 = αM1*si
        m2 = ((Complex{T}(0, one(T)/2) - euler_over_pi) - invtwopi*log((k^2/4)*si^2))*si
        sval = Complex{T}(Rmat[i,i]*m1, zero(T)) + wi*m2
        F[i,i] = one(Complex{T}) - (dval + ik*sval)
    end
    @use_threads multithreading=(multithreaded && N>=32) for j in 2:N
        sj = G.speed[j]
        wj = pts.ws[j]
        @inbounds for i in 1:j-1
            si = G.speed[i]
            wi = pts.ws[i]
            r = G.R[i,j]
            invr = G.invR[i,j]
            lt = G.logterm[i,j]
            inn_ij = G.inner[i,j]
            inn_ji = G.inner[j,i]
            h0 = _bim_hankelh1(0, k*r)
            h1 = _bim_hankelh1(1, k*r)
            j0 = _bim_besselj(0, k*r, h0)
            j1 = _bim_besselj(1, k*r, h1)
            l1_ij = αL1*inn_ij*j1*invr
            l2_ij = αL2*inn_ij*h1*invr - l1_ij*lt
            dval_ij = Rmat[i,j]*l1_ij + wj*l2_ij
            m1_ij = αM1*j0*sj
            m2_ij = αM2*h0*sj - m1_ij*lt
            sval_ij = Rmat[i,j]*m1_ij + wj*m2_ij
            F[i,j] = -(dval_ij + ik*sval_ij)
            l1_ji = αL1*inn_ji*j1*invr
            l2_ji = αL2*inn_ji*h1*invr - l1_ji*lt
            dval_ji = Rmat[j,i]*l1_ji + wi*l2_ji
            m1_ji = αM1*j0*si
            m2_ji = αM2*h0*si - m1_ji*lt
            sval_ji = Rmat[j,i]*m1_ji + wi*m2_ji
            F[j,i] = -(dval_ji + ik*sval_ji)
        end
    end
    return F
end

# Single Kress-corrected D(k)+ikS(k) kernel entry at full-boundary indices
# (i,j), used by the symmetry-reduced assembly below (mirrors
# `_dlp_kernel_entry` in dlp.jl but also carries the S(k) term).
@inline function _cfie_kernel_entry(pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real}
    invtwopi = inv(2*T(pi))
    ik = im*k
    if i == j
        si = G.speed[i]
        wi = pts.ws[i]
        dval = Complex{T}(wi*G.kappa[i], zero(T))
        euler_over_pi = T(Base.MathConstants.eulergamma)/T(pi)
        m1 = -invtwopi*si
        m2 = ((Complex{T}(0, one(T)/2) - euler_over_pi) - invtwopi*log((k^2/4)*si^2))*si
        sval = Complex{T}(Rmat[i,i]*m1, zero(T)) + wi*m2
        return dval + ik*sval
    end
    r = G.R[i,j]
    invr = G.invR[i,j]
    lt = G.logterm[i,j]
    inn = G.inner[i,j]
    sj = G.speed[j]
    wj = pts.ws[j]
    h0 = _bim_hankelh1(0, k*r)
    h1 = _bim_hankelh1(1, k*r)
    j0 = _bim_besselj(0, k*r, h0)
    j1 = _bim_besselj(1, k*r, h1)
    αL1 = -k*invtwopi
    αL2 = im*k/2
    αM1 = -invtwopi
    αM2 = Complex{T}(0, one(T)/2)
    l1 = αL1*inn*j1*invr
    l2 = αL2*inn*h1*invr - l1*lt
    dval = Rmat[i,j]*l1 + wj*l2
    m1 = αM1*j0*sj
    m2 = αM2*h0*sj - m1*lt
    sval = Rmat[i,j]*m1 + wj*m2
    return dval + ik*sval
end

# Symmetry-reduced Kress-corrected CFIE Fredholm matrix, folding the complete
# discrete full-boundary kernel over each source symmetry orbit:
# Fred[a,b] = δ_{ab} - Σ_{j: orbit_of[j]=b} phase[j]*(D+ikS)[fund[a],j].
function _cfie_fredholm_reduced!(F::AbstractMatrix{Complex{T}}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, orbits::SymmetryOrbitMap{T}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
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
                acc += phase[j]*_cfie_kernel_entry(pts, Rmat, G, k, i, j)
            end
            F[a,b] = -acc
        end
        F[b,b] += one(Complex{T})
    end
    return F
end

"""
    evaluate_points(solver::CombinedFieldIntegralEquationSolver, billiard::Bi, k) where {Bi<:AbsBilliard} → pts::BoundaryPoints

Samples the boundary of `billiard` according to `solver.grading`, producing the
boundary discretization needed to assemble the combined-field Fredholm matrix
in [`construct_matrices`](@ref).

## Description
When `solver.symmetry === nothing`, only the fundamental domain's own
physical boundary ([`get_boundary_curves`](@ref)) is discretized — this
already is the complete physical boundary in that case. When a `symmetry` is
set, the *complete* physical boundary ([`full_boundary`](@ref)) is
discretized instead, since [`symmetry_index_orbits`](@ref) folds a full
periodic boundary sampling onto the fundamental domain by exact index
permutation and therefore needs every symmetry image present in `pts`.
"""
function evaluate_points(solver::CombinedFieldIntegralEquationSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver)
    comp = full_boundary(billiard)
    isempty(comp) && error("Boundary cannot be empty.")
    return _cfie_evaluate_points(solver, solver.grading, comp, T(k))
end

"""
    boundary_matrix_size(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints) → N::Int

Returns the dimension of the assembled Fredholm matrix, accounting for any
symmetry-orbit folding onto a fundamental domain.
"""
function boundary_matrix_size(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints)
    solver.symmetry === nothing && return boundary_matrix_size(pts)
    T = _bim_numeric_type(solver)
    # Orbit membership/fundamental_size depends only on the symmetry group's
    # permutation structure, never on the requested irrep character, so the
    # trivial-representation call here is intentional (Step 16.1 audit).
    return fundamental_size(symmetry_index_orbits(T, pts.xy, solver.symmetry))
end

"""
    construct_matrices(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool = true) → A::Matrix{Complex}

Assembles the combined-field Fredholm matrix `A(k) = I - (D(k) + i k S(k))`.
"""
function construct_matrices(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)
    @timeit_debug "construct_matrices" begin
        T = _bim_numeric_type(solver)
        kT = _bim_widen_k(T, k)
        N = length(pts)
        @debug "CFIE matrix construction started" N kT symmetry=solver.symmetry character=solver.character
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
                _cfie_fredholm_full!(A, pts, Rmat, G, kT; multithreaded)
            end
            @debug "CFIE Fredholm matrix assembled" size=size(A)
            return A
        else
            @timeit_debug "symmetry_orbits" begin
                orbits = _fold_boundary(T, pts.xy, solver.symmetry, solver.character)
            end
            m = fundamental_size(orbits)
            A = Matrix{Complex{T}}(undef, m, m)
            @timeit_debug "reduced_fredholm_assembly" begin
                _cfie_fredholm_reduced!(A, pts, Rmat, G, orbits, kT; multithreaded)
            end
            @debug "Symmetry-reduced CFIE Fredholm matrix assembled" fundamental_size=m
            return A
        end
    end
end

"""
    solve(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool = true, use_krylov::Bool = true) → t::Real

Computes the combined-field tension at wavenumber `k`, defined from the
smallest singular value / Krylov nullspace residual of `A(k)` (see
[`construct_matrices`](@ref)).
"""
function solve(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true, use_krylov::Bool=true)
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
    solve_vect(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool = true) → (t::Real, x::Vector)

Computes the combined-field tension and the associated boundary density
eigenvector at wavenumber `k`.
"""
function solve_vect(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)
    T = _bim_numeric_type(solver)
    A = construct_matrices(solver, pts, k; multithreaded)
    @blas_1 vals, _, rvecs, _ = KrylovKit.svdsolve(A, 1, :SR)
    return vals[1], Vector{Complex{T}}(rvecs[1])
end

# `symmetrize_layer_density`/`solve_state`/`_bim_normal_derivative` (the
# boundary-normal-derivative and BIMEigenstate support shared by every
# SweepBIMSolver) live in sweepmethods.jl, not here: none of that logic is
# specific to the combined-field kernel (see sweepmethods.jl's docstrings for
# why the same weighted-transpose reciprocity applies generically).