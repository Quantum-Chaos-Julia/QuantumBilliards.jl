################################################################################
# DOUBLE-LAYER POTENTIAL BOUNDARY INTEGRAL METHOD (DLP)
#
# This file implements the direct boundary integral method for Dirichlet
# quantum billiards using the interior Helmholtz double-layer potential.
#
# For the two-dimensional Helmholtz equation
#
#                         (Δ + k²)ψ = 0
#
# in a billiard Ω with Dirichlet boundary condition ψ|∂Ω = 0, represent the
# interior wavefunction as a double-layer potential
#
#                 ψ(x) = ∫∂Ω ∂n(y)Gk(x,y) μ(y) ds(y),
#
# where μ is the layer density and
#
#                    Gk(x,y) = (i/4) H₀⁽¹⁾(k|x-y|)
#
# is the outgoing free-space Helmholtz Green function. Taking the interior
# boundary limit gives a homogeneous Fredholm equation of the second kind,
#
#                           A(k) μ = 0,
#
# with the normalization convention used here written as
#
#                           A(k) = I - D(k).
#
# The billiard eigenvalues are therefore the wavenumbers for which A(k) becomes
# singular. Numerically, the DLP tension is the smallest singular value
#
#                         t(k) = σmin(A(k)),
#
# which develops minima approaching zero at the Dirichlet eigenvalues.
#
# BOUNDARY DISCRETIZATION
# All boundary discretizations use the Kress quadrature scheme for the
# Helmholtz double-layer kernel. Smooth closed boundary components use the
# original periodic parametrization sampled on a midpoint grid. For
# piecewise-smooth boundaries, this periodic parametrization is additionally
# graded around true geometric corners using the Kress grading map. Thus the
# grading strategy changes the boundary parametrization, while the underlying
# Kress quadrature scheme is used in both cases.
################################################################################

"""
    DoubleLayerPotentialSolver{T,G,Sy,Bi,Ch} <: DLP

Boundary-integral eigensolver based on the interior Helmholtz double-layer
potential.

`DoubleLayerPotentialSolver` discretizes the Fredholm operator

    A(k) = I - D(k),

where `D(k)` is the interior Helmholtz double-layer boundary operator in the
normalization used by this implementation. Dirichlet eigenvalues correspond
to wavenumbers for which `A(k)` becomes singular.

All boundary discretizations use the Kress quadrature scheme for the
DLP Helmholtz kernel. [`SmoothPeriodicGrading`](@ref)
applies it on the original periodic boundary parametrization, while
[`GlobalCornerGrading`](@ref) additionally applies a global Kress grading map
around geometric corners. An optional discrete symmetry reduces the resulting
full-boundary Fredholm operator to a selected symmetry sector.

## Attributes
* `pts_scaling_factor::Vector{T}`: Boundary-point scaling factors used to determine the discretization size.
* `min_pts::Int64`: Minimum number of boundary points.
* `grading::G`: [`BoundaryGrading`](@ref) strategy controlling the periodic boundary parametrization used by the Kress quadrature scheme.
* `symmetry::Sy`: Optional discrete symmetry used to reduce the Fredholm operator.
* `character::Ch`: Character tuple selecting the representation of `symmetry`.
* `eps::T`: Relative numerical tolerance associated with the solver.

## API
[`evaluate_points`](@ref),
[`boundary_matrix_size`](@ref), [`construct_matrices`](@ref), [`solve`](@ref),
[`solve_vect`](@ref), [`solve_wavenumber`](@ref), and [`k_sweep`](@ref).
"""
struct DoubleLayerPotentialSolver{T<:Real,G<:BoundaryGrading,Sy<:Union{AbsSymmetry,Nothing},Bi<:AbsBilliard,Ch<:Tuple} <: DLP
    pts_scaling_factor::Vector{T}
    billiard::Bi
    min_pts::Int64
    grading::G
    symmetry::Sy
    character::Ch
    eps::T
end

"""
    DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}}; min_pts::Int=200, grading::BoundaryGrading=SmoothPeriodicGrading(), symmetry::Union{Nothing,AbsSymmetry}=nothing, character::Tuple=(), eps::T=T(1e-15)) where {T<:Real}

Construct a double-layer potential boundary-integral solver.

All boundary discretizations use the Kress quadrature scheme. `grading`
controls the periodic boundary parametrization on which this quadrature is
applied: [`SmoothPeriodicGrading`](@ref) uses the original parametrization,
while [`GlobalCornerGrading`](@ref) additionally applies a global Kress grading
map around geometric corners.

## Arguments
* `pts_scaling_factor::Union{T,Vector{T}}`: Boundary-point scaling factor or collection of scaling factors used to determine the discretization size.
* `billiard::Bi`: Billiard defining the available discrete symmetries.

## Keyword Arguments
* `min_pts::Int = 200`: Minimum number of boundary sampling points.
* `grading::BoundaryGrading = SmoothPeriodicGrading()`: Boundary parametrization strategy used with the Kress quadrature scheme.
* `symmetry::Union{Nothing,AbsSymmetry} = nothing`: Optional discrete symmetry used to reduce the full-boundary Fredholm operator.
* `character::Tuple = ()`: Character tuple selecting the requested representation of `symmetry`.
* `eps::T = T(1e-15)`: Relative numerical tolerance associated with the solver.

## Returns
* `solver::DoubleLayerPotentialSolver{T,typeof(grading),typeof(symmetry),typeof(billiard),typeof(character)}`: Configured DLP solver.
"""
function DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::Bi; min_pts::Int=200, grading::BoundaryGrading=SmoothPeriodicGrading(), symmetry::Union{Nothing,AbsSymmetry}=nothing, character::Tuple=(), eps::T=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    bs = pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    return DoubleLayerPotentialSolver{T,typeof(grading),typeof(symmetry),typeof(billiard),typeof(character)}(bs, billiard, min_pts, grading, symmetry, character, eps)
end

"""
    DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}}, sector::SymmetrySector; kwargs...) where {T<:Real}

Construct a double-layer potential solver in a validated symmetry sector of
`billiard`.

The symmetry generator and character are resolved from `sector` using the
symmetry registry of `billiard`. This provides a validated alternative to
specifying the `symmetry` and `character` keywords directly. The resulting
solver discretizes the complete physical boundary with the Kress quadrature
scheme and folds the full-boundary Fredholm operator onto the requested
symmetry sector.

## Arguments
* `pts_scaling_factor::Union{T,Vector{T}}`: Boundary-point scaling factor or collection of scaling factors used to determine the discretization size.
* `sector::SymmetrySector`: Validated symmetry sector used for the Fredholm reduction.

## Keyword Arguments
* `kwargs...`: Additional keyword arguments forwarded to [`DoubleLayerPotentialSolver`](@ref).

## Returns
* `solver::DoubleLayerPotentialSolver`: Configured DLP solver in the requested symmetry sector.
"""
function DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}}, sector::SymmetrySector; kwargs...) where {T<:Real}
    generator, character = _resolve_bim_symmetry(sector.billiard, sector)
    return DoubleLayerPotentialSolver(pts_scaling_factor, sector.billiard; symmetry=generator, character=character, kwargs...)
end

_bim_numeric_type(::DoubleLayerPotentialSolver{T}) where {T} = T

# Convert the global periodic parameter t ∈ [0,2π) of a composite boundary
# component into physical arc length measured from the beginning of `comp`.
# The global parameter distributes [0,2π) proportionally to the physical
# lengths of the constituent curves. After locating the curve containing `t`,
# its local parameter is converted to physical arc length with `arc_length`.
# This quantity is used only as the physical boundary coordinate `s`; the
# Nyström quadrature weights are constructed separately from the parametrized
# tangent and the periodic parameter step.
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

# Construct the ungraded periodic boundary discretization used by the Kress
# quadrature scheme. The global periodic parameter is sampled at midpoint nodes
#
#                            σ_j = 2π(j-1/2)/N,
#
# with constant parameter weight h=2π/N. Since no grading map is applied,
# tphys=σ, dtphys/dσ=1, and `ws_der` is identically one. Physical arclength
# weights are nevertheless computed from the actual parametrization ds_j = |γ'(σ_j)| h,
# so the discretization does not require a constant-speed boundary
# parametrization. First and second parameter derivatives are retained because
# they enter the geometry cache and the diagonal limits of the Kress-split
# kernel. The node count is rounded to a symmetry-compatible multiple whenever
# symmetry reduction is requested.
# This is the discretization selected by SmoothPeriodicGrading and is also the
# fallback for GlobalCornerGrading when the boundary contains no true corners (stadium).
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

# Construct the globally graded boundary discretization used by the Kress
# quadrature scheme for a piecewise-smooth component with known true corners.
# The uniform midpoint variable σ is mapped to the physical periodic parameter
#
#                           t = t(σ)
#
# by `multi_kress_graded_nodes_data`. The map fixes every supplied corner and
# makes dt/dσ small near it, thereby clustering quadrature nodes around the
# geometric singularities while retaining a globally periodic discretization.
# Geometry is first evaluated with respect to the physical parameter t and then
# transformed to σ by the chain rule,
#
#                 dγ/dσ  = γ_t dt/dσ,
#                 d²γ/dσ² = γ_tt (dt/dσ)² + γ_t d²t/dσ².
#
# These transformed derivatives are stored because the Kress kernel splitting
# and its diagonal limits must be expressed in the actual quadrature parameter
# σ. The physical arclength weights are correspondingly
#
#                       ds_j = |dγ/dσ| h,
#
# with h=2π/N. `ws` stores the uniform σ-space quadrature weights and `ws_der`
# stores dt/dσ.
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

# Select the globally graded boundary discretization for GlobalCornerGrading.
# True geometric corners are detected in the global periodic parametrization of
# the complete component. If corners are present, the Kress grading map is
# constructed around those locations; otherwise the method delegates to the
# ungraded periodic discretization. Both branches subsequently use the same
# Kress quadrature scheme for the DLP kernel—the only
# distinction is whether the periodic boundary parametrization is graded.
function _dlp_evaluate_points(solver::DoubleLayerPotentialSolver, grading::GlobalCornerGrading, comp::Vector, k::T) where {T<:Real}
    corners = BilliardGeometry._component_corner_locations(T, comp)
    isempty(corners) && return _dlp_evaluate_points(solver, SmoothPeriodicGrading(), comp, k)
    return _dlp_evaluate_points_graded(solver, grading, comp, k, corners)
end

# Determine whether the boundary discretization carries a nontrivial global
# grading map. `ws_der` stores dt/dσ, so an ungraded periodic parametrization
# has ws_der≡1, whereas GlobalCornerGrading produces a nonconstant Jacobian.
# The tolerance avoids treating roundoff-level deviations from unity as an
# actual reparametrization. This flag is passed to the geometry-cache
# construction so that the Kress kernel splitting uses the appropriate geometry.
@inline function _is_nontrivial_dlp_grading(pts::BoundaryPoints{T}) where {T<:Real}
    length(pts.ws_der) == length(pts) || return false
    return maximum(abs.(pts.ws_der .- one(T))) > sqrt(eps(T))
end

# Assemble the full, unfolded DLP Fredholm matrix A(k)=I-D(k) using the Kress
# quadrature scheme. The double-layer kernel is written
# as a periodic logarithmic part plus a smooth remainder. For i≠j the discrete
# kernel entry has the form
#
#                  D_ij = R_ij L1_ij + ws_j L2_ij,
#
# where R_ij is the universal Kress product-quadrature matrix and
#
#        L1_ij = -(k/2π) inner_ij J₁(k r_ij)/r_ij,
#        L2_ij =  (ik/2) inner_ij H₁⁽¹⁾(k r_ij)/r_ij - L1_ij logterm_ij.
#
# `BoundaryGeomCache` supplies distances, geometric inner products, curvature,
# and the logarithmic factor logterm_ij
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

## Evaluate one discrete DLP kernel entry D_ij under the Kress quadrature scheme.
# This is the entry-level counterpart of `_dlp_fredholm_full!` and is used when
# symmetry reduction prevents the convenient pairwise full-matrix assembly.
# For i≠j the same logarithmic splitting is used,
#
#                  D_ij = R_ij L1_ij + ws_j L2_ij,
#
# with the Bessel/Hankel factors evaluated at r_ij. For i=j the analytic
# diagonal self-limit is returned directly.
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

# Assemble the symmetry-reduced Fredholm matrix from the complete physical
# boundary discretization. The Kress quadrature scheme is first understood on
# the full periodic boundary; symmetry reduction is then performed algebraically
# by folding all source indices belonging to the same symmetry orbit.
# For fundamental target index fund[a] and source orbit b,
#
#        A_ab = δ_ab - Σ_{j: orbit_of[j]=b} phase[j] D_{fund[a],j},
#
# where `phase[j]` is the irrep character factor associated with the symmetry image
# containing source node j.
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
    evaluate_points(solver::DoubleLayerPotentialSolver, billiard::Bi, k) where {Bi<:AbsBilliard}

Construct the boundary discretization used by the DLP Fredholm operator.

All discretizations are intended for the Kress quadrature scheme used during
Fredholm matrix assembly. [`SmoothPeriodicGrading`](@ref) samples the original
periodic boundary parametrization, whereas [`GlobalCornerGrading`](@ref)
reparametrizes the boundary with a global Kress grading map that clusters
nodes around detected geometric corners. If no corners are detected, the
latter reduces to the smooth periodic discretization.

When a symmetry sector is present, the complete physical boundary is still
discretized. The full Kress Nyström operator is subsequently folded over
symmetry orbits during matrix assembly.

## Arguments
* `solver::DoubleLayerPotentialSolver`: DLP solver defining the boundary discretization.
* `billiard::Bi`: Billiard whose physical boundary is discretized.
* `k`: Wavenumber used to determine the boundary resolution.

## Returns
* `pts::BoundaryPoints`: Boundary discretization containing the geometry and quadrature data required by the Kress quadrature scheme.
"""
function evaluate_points(solver::DoubleLayerPotentialSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver)
    comp = full_boundary(billiard)
    isempty(comp) && error("Boundary cannot be empty.")
    return _dlp_evaluate_points(solver, solver.grading, comp, T(k))
end

"""
    boundary_matrix_size(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints) → N::Int

Return the dimension of the assembled DLP Fredholm matrix, including exact
algebraic symmetry reduction when a symmetry sector is active.

## Arguments
* `solver::DoubleLayerPotentialSolver`: DLP solver defining the symmetry sector.
* `pts::BoundaryPoints`: Complete physical-boundary discretization.

## Returns
* `N::Int`: Dimension of the assembled Fredholm matrix.
"""
function boundary_matrix_size(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints)
    solver.symmetry === nothing && return boundary_matrix_size(pts)
    T = _bim_numeric_type(solver)
    orbits = _fold_boundary(T, solver.billiard, length(pts), solver.symmetry, solver.character)
    return fundamental_size(orbits)
end

"""
    construct_matrices(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)

Assemble the DLP Fredholm matrix `A(k) = I - D(k)` using the Kress quadrature scheme.

The Helmholtz double-layer kernel is analytically
split into its logarithmic singular part and a smooth remainder and
discretized with Kress product quadrature. The same quadrature scheme is used
for both smooth and globally graded boundary parametrizations.

If `solver.symmetry` is present, the complete full-boundary Kress Nyström
operator is folded over source symmetry orbits with the prescribed character
phases.

## Arguments
* `solver::DoubleLayerPotentialSolver`: DLP solver defining the discretization and symmetry sector.
* `pts::BoundaryPoints`: Full physical-boundary discretization.
* `k`: Real or complex wavenumber at which the Fredholm operator is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.

## Returns
* `A::Matrix{Complex{T}}`: Full or symmetry-reduced Fredholm matrix `A(k) = I - D(k)`.
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
                orbits = _fold_boundary(T, solver.billiard, N, solver.symmetry, solver.character)
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
    solve(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true, use_krylov::Bool=true)

Compute the DLP tension at wavenumber `k`.

The Fredholm matrix `A(k) = I - D(k)` is assembled with
[`construct_matrices`](@ref) using the Kress quadrature scheme. The tension is
defined as its smallest singular value,

    t(k) = σmin(A(k)).

With `use_krylov = true`, the smallest singular value is computed iteratively
with KrylovKit. Otherwise, all singular values are computed with a dense SVD.

## Arguments
* `solver::DoubleLayerPotentialSolver`: DLP solver defining the boundary discretization and symmetry sector.
* `pts::BoundaryPoints`: Boundary discretization containing the geometry and quadrature data required by the Kress quadrature scheme.
* `k`: Wavenumber at which the tension is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.
* `use_krylov::Bool = true`: Compute only the smallest singular value iteratively instead of using a full dense SVD.

## Returns
* `t::Real`: Smallest singular value `σmin(A(k))` of the Kress-discretized Fredholm operator.
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
    solve_vect(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)

Compute the DLP tension and corresponding layer density at wavenumber `k`.

The Fredholm matrix `A(k) = I - D(k)` is assembled with
[`construct_matrices`](@ref) using the Kress quadrature scheme. Its smallest
singular triplet is then computed with KrylovKit. The returned vector is the
right singular vector associated with the smallest singular value and
represents the discrete double-layer density on the assembled boundary
degrees of freedom.

At a Dirichlet eigenvalue this vector approximates a null vector of the
Kress-discretized Fredholm operator,

    A(k)x ≈ 0.

## Arguments
* `solver::DoubleLayerPotentialSolver`: DLP solver defining the boundary discretization and symmetry sector.
* `pts::BoundaryPoints`: Boundary discretization containing the geometry and quadrature data required by the Kress quadrature scheme.
* `k`: Wavenumber at which the layer density is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.

## Returns
* `t::Real`: Smallest singular value `σmin(A(k))` of the Kress-discretized Fredholm operator.
* `x::Vector{Complex{T}}`: Right singular vector representing the discrete double-layer density associated with `t`.
"""
function solve_vect(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)
    T = _bim_numeric_type(solver)
    A = construct_matrices(solver, pts, k; multithreaded)
    @blas_1 vals, _, rvecs, _ = KrylovKit.svdsolve(A, 1, :SR)
    return vals[1], Vector{Complex{T}}(rvecs[1])
end