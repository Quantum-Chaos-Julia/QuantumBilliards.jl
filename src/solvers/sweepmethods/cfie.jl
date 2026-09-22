################################################################################
# COMBINED-FIELD INTEGRAL EQUATION BOUNDARY INTEGRAL METHOD (CFIE)
#
# This file implements the combined-field boundary integral method for
# Dirichlet quantum billiards using a linear combination of the interior
# Helmholtz double- and single-layer boundary operators.
#
# For the two-dimensional Helmholtz equation
#
#                         (Δ + k²)ψ = 0
#
# in a billiard Ω with Dirichlet boundary condition ψ|∂Ω = 0, the formulation
# combines the double-layer operator D(k) with the single-layer operator S(k).
# With the normalization and coupling convention used here:
#
#                         A(k) μ = 0,
#
# where
#
#                   A(k) = I - (D(k) + i k S(k)).
#
# Numerically, the CFIE tension is the smallest singular value
#
#                         t(k) = σmin(A(k)),
#
# which develops minima approaching zero at the Dirichlet eigenvalues.
#
# BOUNDARY DISCRETIZATION
# All boundary discretizations use the Kress quadrature scheme for both the
# double- and single-layer Helmholtz kernels. Smooth closed boundary components
# use the original periodic parametrization sampled on a midpoint grid.
# CornerGrading applies a single-corner Kress grading map to a closed curve
# whose parametric corner is known at the periodic endpoint, while
# GlobalCornerGrading detects true geometric corners and applies a global
# Kress grading map around them. Thus the grading strategy changes the boundary
# parametrization, while the underlying Kress quadrature scheme is used in all
# cases.
################################################################################

"""
    CombinedFieldIntegralEquationSolver{T,G,Sy,Bi,Ch} <: CFIE

Boundary-integral eigensolver based on the Helmholtz combined-field integral
equation.

`CombinedFieldIntegralEquationSolver` discretizes the Fredholm operator

    A(k) = I - (D(k) + i k S(k)),

where `D(k)` and `S(k)` are the Helmholtz double- and single-layer boundary
operators in the normalization used by this implementation. Dirichlet
eigenvalues correspond to wavenumbers for which `A(k)` becomes singular.

All boundary discretizations use the Kress quadrature scheme for both kernel
contributions. [`SmoothPeriodicGrading`](@ref) uses the original periodic
boundary parametrization, [`CornerGrading`](@ref) applies a single-corner
Kress grading map to a closed curve with a known parametric corner, and
[`GlobalCornerGrading`](@ref) applies a global Kress grading map around
detected geometric corners. An optional discrete symmetry reduces the
resulting full-boundary Fredholm operator to a selected symmetry sector.

## Attributes
* `pts_scaling_factor::Vector{T}`: Boundary-point scaling factors used to determine the discretization size.
* `billiard::Bi`: Billiard defining the available discrete symmetries.
* `min_pts::Int64`: Minimum number of boundary sampling points.
* `grading::G`: [`BoundaryGrading`](@ref) strategy controlling the periodic boundary parametrization used by the Kress quadrature scheme.
* `symmetry::Sy`: Optional discrete symmetry used to reduce the full-boundary Fredholm operator.
* `character::Ch`: Character tuple selecting the representation of `symmetry`.
* `eps::T`: Relative numerical tolerance associated with the solver.

## API
[`evaluate_points`](@ref), [`boundary_matrix_size`](@ref),
[`construct_matrices`](@ref), [`solve`](@ref), [`solve_vect`](@ref),
[`solve_wavenumber`](@ref), and [`k_sweep`](@ref).
"""
struct CombinedFieldIntegralEquationSolver{T<:Real,G<:BoundaryGrading,Sy<:Union{AbsSymmetry,Nothing},Bi<:AbsBilliard,Ch<:Tuple} <: CFIE
    pts_scaling_factor::Vector{T}
    billiard::Bi
    min_pts::Int64
    grading::G
    symmetry::Sy
    character::Ch
    eps::T
end

"""
    CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::Bi; min_pts::Int=200, grading::BoundaryGrading=SmoothPeriodicGrading(), symmetry::Union{Nothing,AbsSymmetry}=nothing, character::Tuple=(), eps::T=T(1e-15)) where {T<:Real, Bi<:AbsBilliard}

Construct a combined-field boundary-integral solver.
All boundary discretizations use the Kress quadrature scheme for the
double- and single-layer kernels. `grading` controls the periodic boundary
parametrization on which this quadrature is applied.

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
* `solver::CombinedFieldIntegralEquationSolver`: Configured CFIE solver in the requested symmetry sector.
"""
function CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::Bi; min_pts::Int=200, grading::BoundaryGrading=SmoothPeriodicGrading(), symmetry::Union{Nothing,AbsSymmetry}=nothing, character::Tuple=(), eps::T=T(1e-15)) where {T<:Real,Bi<:AbsBilliard}
    bs = pts_scaling_factor isa T ? [pts_scaling_factor] : pts_scaling_factor
    return CombinedFieldIntegralEquationSolver{T,typeof(grading),typeof(symmetry),typeof(billiard),typeof(character)}(bs, billiard, min_pts, grading, symmetry, character, eps)
end

"""
    CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::Bi, sector::SymmetrySector; kwargs...) where {T<:Real,Bi<:AbsBilliard}

Construct a combined-field boundary-integral solver in a validated symmetry
sector of `billiard`.

The symmetry generator and character are resolved from `sector` using the
symmetry registry of `billiard`. This provides a validated alternative to
specifying the `symmetry` and `character` keywords directly. The resulting
solver discretizes the complete physical boundary with the Kress quadrature
scheme and folds the full-boundary Fredholm operator onto the requested
symmetry sector.

## Arguments
* `pts_scaling_factor::Union{T,Vector{T}}`: Boundary-point scaling factor or collection of scaling factors used to determine the discretization size.
* `billiard::BilliardGeometry.AbsBilliard`: Billiard defining the available discrete symmetries.
* `sector::SymmetrySector`: Validated symmetry sector used for the Fredholm reduction.

## Keyword Arguments
* `kwargs...`: Additional keyword arguments forwarded to [`CombinedFieldIntegralEquationSolver`](@ref).

## Returns
* `solver::CombinedFieldIntegralEquationSolver{T,typeof(grading),typeof(symmetry),typeof(billiard),typeof(character)}`: Configured CFIE solver in the requested symmetry sector.
"""
function CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}}, billiard::Bi, sector::SymmetrySector; kwargs...) where {T<:Real,Bi<:AbsBilliard}
    generator, character = _resolve_bim_symmetry(billiard, sector)
    return CombinedFieldIntegralEquationSolver(pts_scaling_factor, billiard; symmetry=generator, character=character, kwargs...)
end

_bim_numeric_type(::CombinedFieldIntegralEquationSolver{T}) where {T} = T

# Construct the ungraded periodic boundary discretization used by the Kress
# quadrature scheme. The global periodic parameter is sampled at midpoint nodes
#
#                           σ_j = 2π(j-1/2)/N,
#
# with constant parameter weight h=2π/N. Since no grading map is applied,
# tphys=σ, dtphys/dσ=1, and `ws_der` is identically one. Physical arclength
# weights are computed from the actual parametrization,
#
#                           ds_j = |γ'(σ_j)| h,
#
# so no constant-speed parametrization is assumed. First and second parameter
# derivatives are retained because they enter the geometry cache and the
# diagonal limits of both Kress-split CFIE kernels. Physical arc length along
# a composite component is obtained with `_dlp_composite_arclength`, which is
# purely geometric and is shared with the DLP implementation.
#
# The node count is rounded to a symmetry-compatible multiple whenever symmetry
# reduction is requested. This discretization is selected directly by
# SmoothPeriodicGrading and is also the fallback for GlobalCornerGrading when
# no true geometric corners are detected. The absence of grading does not
# disable Kress quadrature: the resulting points are used by the same Kress
# quadrature scheme during CFIE Fredholm matrix assembly.
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

# Construct the single-corner graded boundary discretization used by the Kress
# quadrature scheme. CornerGrading requires the complete boundary component to
# be represented by one closed curve whose own parameter u∈[0,1) has its known
# corner at the periodic endpoint u=0≡1. Unlike GlobalCornerGrading, no
# geometric corner detection is performed.
#
# A uniform midpoint variable σ∈[0,2π) is mapped through the single-corner
# Kress grading map. Since the curve itself is parametrized by u∈[0,1), the
# mapped physical parameter is
#
#                           u = t(σ)/(2π).
#
# If `jac=dt/dσ` and `jac2=d²t/dσ²`, the derivatives of the physical curve
# with respect to the quadrature variable σ are obtained by the chain rule,
#
#                   dγ/dσ = γ_u jac/(2π),
#       d²γ/dσ² = γ_uu (jac/(2π))² + γ_u jac2/(2π).
#
# These transformed derivatives are stored because the Kress kernel splitting
# and diagonal limits must be expressed in the active quadrature parameter.
# Physical arclength weights are then ds_j=|dγ/dσ|h with h=2π/N, while `ws`
# stores the uniform σ-space weights and `jac` stores the grading Jacobian.
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

# Construct the globally graded boundary discretization used by the Kress
# quadrature scheme for a piecewise-smooth component with known true corners.
# The uniform midpoint variable σ is mapped to the physical global periodic
#
#                           t = t(σ)
#
# by `multi_kress_graded_nodes_data`. The map fixes the supplied corner
# locations and makes dt/dσ small near them, clustering quadrature nodes around
# the geometric singularities while retaining a globally periodic
# discretization.
#
# Geometry is evaluated first with respect to the physical parameter t and
# transformed to σ using
#
#                   dγ/dσ = γ_t dt/dσ,
#        d²γ/dσ² = γ_tt (dt/dσ)² + γ_t d²t/dσ².
#
# The physical arclength weights are consequently ds_j = |dγ/dσ| h,
# with h=2π/N. `ws` stores the uniform σ-space quadrature weights and `ws_der`
# stores dt/dσ. The grading changes only the periodic parametrization; both
# the double- and single-layer contributions are subsequently discretized with
# the same Kress quadrature scheme as in the smooth case.
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

# top-level _evaluate_points function for CFIE, see comments above
function _cfie_evaluate_points(solver::CombinedFieldIntegralEquationSolver, grading::GlobalCornerGrading, comp::Vector, k::T) where {T<:Real}
    corners = BilliardGeometry._component_corner_locations(T, comp)
    isempty(corners) && return _cfie_evaluate_points(solver, SmoothPeriodicGrading(), comp, k)
    return _cfie_evaluate_points_graded(solver, grading, comp, k, corners)
end

################################################################################
################## PRIVATE HELPERS: FREDHOLM MATRIX ASSEMBLY ##################
################################################################################

# Assemble the full, unfolded CFIE Fredholm matrix
#
#                       A(k) = I - (D(k) + ikS(k))
#
# using the Kress quadrature scheme for both boundary operators. The
# double-layer contribution uses the same logarithmic splitting as the DLP
# implementation. For i≠j,
#
#                  D_ij = R_ij L1_ij + ws_j L2_ij,
#
# with
#
#        L1_ij = -(k/2π) inner_ij J₁(k r_ij)/r_ij,
#        L2_ij =  (ik/2) inner_ij H₁⁽¹⁾(k r_ij)/r_ij
#                  - L1_ij logterm_ij.
#
# The single-layer contribution is split analogously,
#
#                  S_ij = R_ij M1_ij + ws_j M2_ij,
#
# where
#
#        M1_ij = -(1/2π) J₀(k r_ij) speed_j,
#        M2_ij =  (i/2) H₀⁽¹⁾(k r_ij) speed_j
#                  - M1_ij logterm_ij.
#
# `BoundaryGeomCache` supplies the pairwise distances, geometric inner
# products, source speeds, curvature, and logarithmic factors in the active
# periodic parametrization. On the diagonal, both D and S are replaced by
# their analytic self-limits; the single-layer limit contains the logarithmic
# k- and speed-dependent term together with the Euler-Mascheroni contribution.
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

# Evaluate one discrete combined-field kernel entry
#
#                         (D + ikS)_ij
#
# under the Kress quadrature scheme. This is the entry-level counterpart of
# `_cfie_fredholm_full!` and is used by the symmetry-reduced assembly, where
# arbitrary source images must be accumulated individually.
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

# Assemble the symmetry-reduced CFIE Fredholm matrix from the complete physical
# boundary discretization. The Kress quadrature scheme is defined on the full
# periodic boundary for both D and S, after which symmetry reduction is
# performed algebraically by folding source indices belonging to the same
# symmetry orbit.
#
# For fundamental target index fund[a] and source orbit b,
#
#       A_ab = δ_ab - Σ_{j: orbit_of[j]=b} phase[j] (D+ikS)_{fund[a],j},
#
# where `phase[j]` is the irrep character factor associated with the symmetry
# image containing source node j. 
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
    evaluate_points(solver::CombinedFieldIntegralEquationSolver, billiard::Bi, k) where {Bi<:AbsBilliard}

Construct the boundary discretization used by the CFIE Fredholm operator.

All discretizations are intended for the Kress quadrature scheme used for both
the double- and single-layer kernels during Fredholm matrix assembly.
[`SmoothPeriodicGrading`](@ref) samples the original periodic boundary
parametrization, [`CornerGrading`](@ref) applies a single-corner Kress grading
map to a closed curve with a known parametric corner, and
[`GlobalCornerGrading`](@ref) applies a global Kress grading map around
detected geometric corners.

The complete physical boundary is always discretized. When a symmetry sector
is present, the resulting full-boundary Fredholm operator is subsequently
folded over symmetry orbits during matrix assembly.

## Arguments
* `solver::CombinedFieldIntegralEquationSolver`: CFIE solver defining the boundary discretization.
* `billiard::Bi`: Billiard whose complete physical boundary is discretized.
* `k`: Wavenumber used to determine the boundary resolution.

## Returns
* `pts::BoundaryPoints`: Boundary discretization containing the geometry and quadrature data required by the Kress quadrature scheme.
"""
function evaluate_points(solver::CombinedFieldIntegralEquationSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver)
    comp = full_boundary(billiard)
    isempty(comp) && error("Boundary cannot be empty.")
    return _cfie_evaluate_points(solver, solver.grading, comp, T(k))
end

"""
    boundary_matrix_size(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints) → N::Int

Return the dimension of the assembled CFIE Fredholm matrix, including exact
algebraic symmetry reduction when a symmetry sector is active.

## Arguments
* `solver::CombinedFieldIntegralEquationSolver`: CFIE solver defining the symmetry sector.
* `pts::BoundaryPoints`: Complete physical-boundary discretization.

## Returns
* `N::Int`: Dimension of the assembled Fredholm matrix.
"""
function boundary_matrix_size(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints)
    solver.symmetry === nothing && return boundary_matrix_size(pts)
    T = _bim_numeric_type(solver)
    orbits = _fold_boundary(T, solver.billiard, length(pts), solver.symmetry, solver.character)
    return fundamental_size(orbits)
end

"""
    construct_matrices(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)

Assemble the CFIE Fredholm matrix `A(k) = I - (D(k) + i k S(k))` using the
Kress quadrature scheme.

The Helmholtz double- and single-layer kernels are analytically split into
periodic logarithmic parts and smooth remainders and discretized with the same
Kress quadrature scheme. The construction applies to smooth, single-corner
graded, and globally corner-graded boundary parametrizations.

If `solver.symmetry` is present, the complete full-boundary Fredholm operator
is folded over source symmetry orbits with the prescribed character phases.

## Arguments
* `solver::CombinedFieldIntegralEquationSolver`: CFIE solver defining the discretization and symmetry sector.
* `pts::BoundaryPoints`: Complete physical-boundary discretization.
* `k`: Real or complex wavenumber at which the Fredholm operator is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.

## Returns
* `A::Matrix{Complex{T}}`: Full or symmetry-reduced Fredholm matrix `A(k) = I - (D(k) + i k S(k))`.
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
                orbits = _fold_boundary(T, solver.billiard, N, solver.symmetry, solver.character)
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
    solve(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true, use_krylov::Bool=true)

Compute the CFIE tension at wavenumber `k`.

The Fredholm matrix

    A(k) = I - (D(k) + i k S(k))

is assembled with [`construct_matrices`](@ref) using the Kress quadrature
scheme. The tension is defined as its smallest singular value,

    t(k) = σmin(A(k)).

With `use_krylov = true`, the smallest singular value is computed iteratively
with KrylovKit. Otherwise, all singular values are computed with a dense SVD.

## Arguments
* `solver::CombinedFieldIntegralEquationSolver`: CFIE solver defining the boundary discretization and symmetry sector.
* `pts::BoundaryPoints`: Boundary discretization containing the geometry and quadrature data required by the Kress quadrature scheme.
* `k`: Wavenumber at which the tension is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.
* `use_krylov::Bool = true`: Compute only the smallest singular value iteratively instead of using a full dense SVD.

## Returns
* `t::Real`: Smallest singular value `σmin(A(k))` of the Kress-discretized CFIE Fredholm operator.
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
    solve_vect(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)

Compute the CFIE tension and corresponding layer density at wavenumber `k`.

The Fredholm matrix

    A(k) = I - (D(k) + i k S(k))

is assembled with [`construct_matrices`](@ref) using the Kress quadrature
scheme. Its smallest singular triplet is then computed with KrylovKit. The
returned vector is the right singular vector associated with the smallest
singular value and represents the discrete combined-field layer density on
the assembled boundary degrees of freedom.

At a Dirichlet eigenvalue this vector approximates a null vector of the
Kress-discretized Fredholm operator,

    A(k)x ≈ 0.

## Arguments
* `solver::CombinedFieldIntegralEquationSolver`: CFIE solver defining the boundary discretization and symmetry sector.
* `pts::BoundaryPoints`: Boundary discretization containing the geometry and quadrature data required by the Kress quadrature scheme.
* `k`: Wavenumber at which the layer density is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Fredholm matrix assembly.

## Returns
* `t::Real`: Smallest singular value `σmin(A(k))` of the Kress-discretized CFIE Fredholm operator.
* `x::Vector{Complex{T}}`: Right singular vector representing the discrete combined-field layer density associated with `t`.
"""
function solve_vect(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)
    T = _bim_numeric_type(solver)
    A = construct_matrices(solver, pts, k; multithreaded)
    @blas_1 vals, _, rvecs, _ = KrylovKit.svdsolve(A, 1, :SR)
    return vals[1], Vector{Complex{T}}(rvecs[1])
end