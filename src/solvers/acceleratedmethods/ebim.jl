################################################################################
# EXPANDED BOUNDARY INTEGRAL METHOD (EBIM)
#
# This file implements a local second-order eigensolver for boundary-integral
# nonlinear eigenvalue problems
#
#                              A(k)v = 0,
#
# where A(k) is the Fredholm matrix produced by a DLP, CFIE, or composite BIM
# discretization. Around an expansion center k₀, EBIM uses the local Taylor
# expansion
#
#   A(k₀ + ε) = A₀ + ε A₁ + (ε²/2) A₂ + O(ε³),
#
# with
#
#   A₀ = A(k₀),    A₁ = A'(k₀),    A₂ = A''(k₀).
#
# The leading-order eigenvalue displacements from k₀ are obtained from the
# generalized eigenproblem
#
#                         A₀v = λ A₁v,
#
# for which
#
#                              ε₁ = -λ.
#
# Thus each generalized eigenvalue λ gives a first-order approximation
#
#                              k ≈ k₀ - λ
#
# to an eigenvalue of the original nonlinear problem near k₀.
#
# For a corresponding left generalized eigenvector u, the quadratic term in
# the Taylor expansion gives the second-order correction
#
#                 ε₂ = -(ε₁²/2) (uᴴ A₂ v)/(uᴴ A₁ v),
#
# and hence
#
#                         kEBIM = k₀ + ε₁ + ε₂.
#
# EBIM can therefore recover several eigenvalues surrounding a single
# expansion center. It is a local spectral method: the useful range around k₀
# is controlled by the accuracy of the truncated Taylor expansion rather than
# by a contour or an explicitly prescribed search interval.
#
# NUMERICAL SOLUTION
# ------------------
# The generalized problem is evaluated through
#
#                         A₀⁻¹ A₁ v = μv,
#
# where μ = 1/λ. Eigenvalues with largest |μ| correspond to the smallest |λ|
# and therefore to eigenvalues of the nonlinear problem closest to k₀.
#
# A₀ is factorized once, and KrylovKit computes the required left and right
# eigenvectors without explicitly forming A₀⁻¹A₁. The requested number of
# local levels determines how many of these dominant eigenpairs are retained
# and corrected to second order.
################################################################################

"""
    ExpandedBIMSolver{T,K} <: AcceleratedBIMSolver

Local second-order eigensolver for boundary-integral nonlinear eigenvalue
problems.

`ExpandedBIMSolver` wraps a [`SweepBIMSolver`](@ref) and computes eigenvalues
near an expansion center `k₀` from the Fredholm matrix `A(k₀)` and its first
two wavenumber derivatives. A single expansion can produce multiple nearby
eigenvalues.

The first-order eigenvalue displacements are obtained from a generalized
eigenproblem involving `A(k₀)` and `A'(k₀)`, after which `A''(k₀)` supplies a
second-order correction. The matrix derivatives are assembled analytically
from the underlying DLP, CFIE, or composite BIM kernel.

Unlike the contour-based [`BeynSolver`](@ref), EBIM is a local spectral method:
it finds eigenvalues around an expansion center rather than all eigenvalues
enclosed by a contour.

## Attributes
* `kernel::K`: Wrapped [`SweepBIMSolver`](@ref) defining the boundary-integral operator.
* `use_chebyshev::Bool`: Whether supported special-function evaluations use Chebyshev acceleration.
* `cheb_config::ChebyshevConfig{T}`: Configuration of the Chebyshev interpolation and tuning procedure.
* `tol::T`: Convergence tolerance used by the Krylov eigensolver.
* `maxiter::Int`: Maximum number of Krylov iterations.
* `krylovdim::Int`: Minimum Krylov subspace dimension used for the local generalized eigensolve.

## API
The principal operations are [`evaluate_points`](@ref),
[`construct_matrices`](@ref), [`solve`](@ref), [`solve_wavenumber`](@ref),
[`solve_spectrum`](@ref), and [`compute_spectrum`](@ref).
"""
struct ExpandedBIMSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K
    use_chebyshev::Bool
    cheb_config::ChebyshevConfig{T}
    tol::T
    maxiter::Int
    krylovdim::Int
end

"""
    ExpandedBIMSolver(kernel::K; use_chebyshev::Bool=true, n_panels_h::Int=15000, M_h::Int=5, n_panels_j::Int=10000, M_j::Int=5, cheb_config::Union{Nothing,ChebyshevConfig}=nothing, tol::Real=1e-12, maxiter::Int=5000, krylovdim::Int=40) where {K<:SweepBIMSolver}

Construct an [`ExpandedBIMSolver`](@ref) for the supplied boundary-integral
kernel.

When `cheb_config` is not supplied, a [`ChebyshevConfig`](@ref) is constructed
from the specified Hankel and Bessel-J panel counts and polynomial degrees.
The Krylov parameters control the iterative eigensolve used to obtain the
local EBIM corrections.

## Arguments
* `kernel::K`: [`SweepBIMSolver`](@ref) defining the Fredholm operator to expand.

## Keyword Arguments
* `use_chebyshev::Bool = true`: Use Chebyshev-accelerated special-function evaluation when supported.
* `n_panels_h::Int = 15000`: Initial Hankel-function Chebyshev panel count, ignored when `cheb_config` is supplied.
* `M_h::Int = 5`: Hankel-function Chebyshev polynomial degree, ignored when `cheb_config` is supplied.
* `n_panels_j::Int = 10000`: Initial Bessel-J-function Chebyshev panel count, ignored when `cheb_config` is supplied.
* `M_j::Int = 5`: Bessel-J-function Chebyshev polynomial degree, ignored when `cheb_config` is supplied.
* `cheb_config::Union{Nothing,ChebyshevConfig} = nothing`: Optional preconstructed Chebyshev configuration.
* `tol::Real = 1e-12`: Convergence tolerance passed to the Krylov eigensolver.
* `maxiter::Int = 5000`: Maximum number of Krylov iterations.
* `krylovdim::Int = 40`: Minimum Krylov subspace dimension used by the eigensolver.

## Returns
* `solver::ExpandedBIMSolver`: Configured EBIM solver.
"""
function ExpandedBIMSolver(kernel::K; use_chebyshev::Bool = true, n_panels_h::Int = 15000, M_h::Int = 5, n_panels_j::Int = 10000, M_j::Int = 5, cheb_config::Union{Nothing,ChebyshevConfig} = nothing, tol::Real = 1e-12, maxiter::Int = 5000, krylovdim::Int = 40) where {K<:SweepBIMSolver}
    T = _bim_numeric_type(kernel)
    cfg = cheb_config === nothing ? ChebyshevConfig(T; n_panels_h, M_h, n_panels_j, M_j) : cheb_config
    return ExpandedBIMSolver{T,K}(kernel, use_chebyshev, cfg, T(tol), maxiter, krylovdim)
end

_bim_numeric_type(::ExpandedBIMSolver{T}) where {T} = T

mutable struct EBIMCache{G,R,O}
    G::G
    Rmat::R
    orbits::O
    cheb_lookup::Union{Nothing,ChebRadialLookupCache}
end

function EBIMCache(cs::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}, pts::BoundaryPoints{T}) where {T<:Real}
    N = length(pts)
    G = boundary_geom_cache(pts, _is_nontrivial_dlp_grading(pts))
    Rmat = zeros(T, N, N); kress_R!(Rmat)
    orbits = cs.symmetry === nothing ? nothing : _fold_boundary(T, cs.billiard, N, cs.symmetry, cs.character)
    return EBIMCache(G, Rmat, orbits, nothing)
end

################################################################################
###################### DERIVATIVE-OF-HANKEL-KERNEL HELPERS ###################
################################################################################

# Evaluate a kernel term proportional to k*Z₁(kr)/r together with its first
# two wavenumber derivatives. `α` contains the complete linear-in-k prefactor,
# while `z0` and `z1` are the already evaluated Z₀(kr) and Z₁(kr). Using
# Z₁'(z)=Z₀(z)-Z₁(z)/z gives
#
#   value = α*r⁻¹*Z₁(kr),
#   ∂ₖvalue = α*Z₀(kr),
#   ∂ₖ²value = (α/k)*Z₀(kr)-α*r*Z₁(kr).
#
# The radial factors introduced by differentiation cancel the explicit r⁻¹.
@inline function _ebim_lin1_deriv(α, r::T, invr::T, z0, z1, k) where {T<:Real}
    val = α*invr*z1
    dval = α*z0
    ddval = (α/k)*z0-α*r*z1
    return val, dval, ddval
end

# Evaluate a kernel term proportional to Z₀(kr) with a k-independent
# prefactor together with its first two wavenumber derivatives. Using
# Z₀'(z)=-Z₁(z) and Z₁'(z)=Z₀(z)-Z₁(z)/z gives
#
#   value = β*Z₀(kr),
#   ∂ₖvalue = -β*r*Z₁(kr),
#   ∂ₖ²value = -β*r²*Z₀(kr)+(β*r/k)*Z₁(kr).
@inline function _ebim_const0_deriv(β, r::T, k, z0, z1) where {T<:Real}
    val = β*z0
    dval = -β*r*z1
    ddval = -β*r*r*z0+(β*r/k)*z1
    return val, dval, ddval
end

################################################################################
######################## DLP KERNEL WITH DERIVATIVES ##########################
################################################################################

# Evaluate one Kress DLP kernel entry and its first two wavenumber
# derivatives. The returned values correspond to the discretized double-layer
# kernel D(k), not the complete Fredholm operator A(k)=I-D(k). The diagonal
# geometric self-term is independent of k and therefore has zero first and
# second derivatives.
@inline function _dlp_kernel_entry_with_derivatives(pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real}
    if i==j
        return Complex{T}(pts.ws[i]*G.kappa[i], zero(T)), zero(Complex{T}), zero(Complex{T})
    end
    invtwopi = inv(2*T(pi))
    r = G.R[i,j]
    invr = G.invR[i,j]
    lt = G.logterm[i,j]
    inn = G.inner[i,j]
    h1 = _bim_hankelh1(1, k*r)
    j1 = _bim_besselj(1, k*r, h1)
    h0 = _bim_hankelh1(0, k*r)
    j0 = _bim_besselj(0, k*r, h0)
    αL1 = -k*invtwopi
    αL2 = im*k/2
    l1, dl1, ddl1 = _ebim_lin1_deriv(αL1*inn, r, invr, j0, j1, k)
    l2a, dl2a, ddl2a = _ebim_lin1_deriv(αL2*inn, r, invr, h0, h1, k)
    l2 = l2a-l1*lt
    dl2 = dl2a-dl1*lt
    ddl2 = ddl2a-ddl1*lt
    val = Rmat[i,j]*l1+pts.ws[j]*l2
    dval = Rmat[i,j]*dl1+pts.ws[j]*dl2
    ddval = Rmat[i,j]*ddl1+pts.ws[j]*ddl2
    return val, dval, ddval
end

################################################################################
######################## CFIE KERNEL WITH DERIVATIVES #########################
################################################################################

# Evaluate one Kress CFIE kernel entry D(k)+ikS(k) and its first two
# wavenumber derivatives. Both the explicit ik factor and the k-dependence of
# the single- and double-layer kernels are differentiated analytically. On the
# diagonal, the derivatives of the logarithmic Kress self-term are included
# explicitly.
@inline function _cfie_kernel_entry_with_derivatives(pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real}
    invtwopi = inv(2*T(pi))
    ik = im*k
    if i==j
        si = G.speed[i]
        wi = pts.ws[i]
        Dval = Complex{T}(wi*G.kappa[i], zero(T))
        euler_over_pi = T(Base.MathConstants.eulergamma)/T(pi)
        m1 = -invtwopi*si
        m2 = ((Complex{T}(0, one(T)/2)-euler_over_pi)-invtwopi*log((k^2/4)*si^2))*si
        Sval = Complex{T}(Rmat[i,i]*m1, zero(T))+wi*m2
        val = Dval+ik*Sval
        dm2 = -si/(T(pi)*k)
        ddm2 = si/(T(pi)*k^2)
        dSval = wi*dm2
        ddSval = wi*ddm2
        dval = im*Sval+ik*dSval
        ddval = 2*im*dSval+ik*ddSval
        return val, dval, ddval
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
    l1, dl1, ddl1 = _ebim_lin1_deriv(αL1*inn, r, invr, j0, j1, k)
    l2a, dl2a, ddl2a = _ebim_lin1_deriv(αL2*inn, r, invr, h0, h1, k)
    l2 = l2a-l1*lt
    dl2 = dl2a-dl1*lt
    ddl2 = ddl2a-ddl1*lt
    Dval = Rmat[i,j]*l1+wj*l2
    dDval = Rmat[i,j]*dl1+wj*dl2
    ddDval = Rmat[i,j]*ddl1+wj*ddl2
    m1, dm1, ddm1 = _ebim_const0_deriv(αM1*sj, r, k, j0, j1)
    m2a, dm2a, ddm2a = _ebim_const0_deriv(αM2*sj, r, k, h0, h1)
    m2 = m2a-m1*lt
    dm2 = dm2a-dm1*lt
    ddm2 = ddm2a-ddm1*lt
    Sval = Rmat[i,j]*m1+wj*m2
    dSval = Rmat[i,j]*dm1+wj*dm2
    ddSval = Rmat[i,j]*ddm1+wj*ddm2
    val = Dval+ik*Sval
    dval = dDval+im*Sval+ik*dSval
    ddval = ddDval+2*im*dSval+ik*ddSval
    return val, dval, ddval
end

################################################################################
####################### COMPOSITE KERNEL WITH DERIVATIVES #####################
################################################################################

# For a clean interface, we define a generic function that dispatches to the appropriate kernel entry with derivatives based on the solver type.
@inline _composite_component_kernel_entry_with_derivatives(::DoubleLayerPotentialSolver, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real} = _dlp_kernel_entry_with_derivatives(pts, Rmat, G, k, i, j)
@inline _composite_component_kernel_entry_with_derivatives(::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}, i::Int, j::Int) where {T<:Real} = _cfie_kernel_entry_with_derivatives(pts, Rmat, G, k, i, j)

# Evaluate a smooth DLP interaction between two distinct boundary components of a multiply connected domain, together with its first two wavenumber
# derivatives. The target `(xi,yi)` lies on one component and `pb.xy[j]` on another, so the kernel is nonsingular and requires no Kress quadrature scheme.
@inline function _composite_cross_kernel_entry_with_derivatives(::DoubleLayerPotentialSolver, pb::BoundaryPoints{T}, xi::T, yi::T, k::Union{T,Complex{T}}, j::Int) where {T<:Real}
    xj, yj = pb.xy[j]
    dx = xi-xj
    dy = yi-yj
    r = hypot(dx, dy)
    invr = inv(r)
    tx, ty = pb.tangent[j]
    inn = ty*dx-tx*dy
    h1 = _bim_hankelh1(1, k*r)
    h0 = _bim_hankelh1(0, k*r)
    αL2 = im*k/2
    a, da, dda = _ebim_lin1_deriv(αL2*inn, r, invr, h0, h1, k)
    return pb.ws[j]*a, pb.ws[j]*da, pb.ws[j]*dda
end

# Evaluate a smooth CFIE interaction between two distinct boundary components of a multiply connected domain, together with its first two wavenumber
# derivatives. The target `(xi,yi)` lies on one component and `pb.xy[j]` on another, so the kernel is nonsingular and requires no Kress quadrature scheme.
@inline function _composite_cross_kernel_entry_with_derivatives(::CombinedFieldIntegralEquationSolver, pb::BoundaryPoints{T}, xi::T, yi::T, k::Union{T,Complex{T}}, j::Int) where {T<:Real}
    xj, yj = pb.xy[j]
    dx = xi-xj
    dy = yi-yj
    r = hypot(dx, dy)
    invr = inv(r)
    tx, ty = pb.tangent[j]
    inn = ty*dx-tx*dy
    sj = hypot(tx, ty)
    ik = im*k
    h0 = _bim_hankelh1(0, k*r)
    h1 = _bim_hankelh1(1, k*r)
    αL2 = im*k/2
    αM2 = Complex{T}(0, one(T)/2)
    Dval, dDval, ddDval = _ebim_lin1_deriv(αL2*inn, r, invr, h0, h1, k)
    Dval *= pb.ws[j]; dDval *= pb.ws[j]; ddDval *= pb.ws[j]
    Sval, dSval, ddSval = _ebim_const0_deriv(αM2*sj, r, k, h0, h1)
    Sval *= pb.ws[j]; dSval *= pb.ws[j]; ddSval *= pb.ws[j]
    val = Dval+ik*Sval
    dval = dDval+im*Sval+ik*dSval
    ddval = ddDval+2*im*dSval+ik*ddSval
    return val, dval, ddval
end

################################################################################
######################### FREDHOLM ASSEMBLY WITH DERIVATIVES ##################
################################################################################

# Assemble the full-boundary Fredholm matrix A(k)=I-K(k) and its first two wavenumber derivatives in place. `entry_fn` evaluates either the DLP kernel
# D(k) or the CFIE kernel D(k)+ikS(k), together with its first two derivatives.
function _ebim_fredholm_full_with_derivatives!(entry_fn, A::AbstractMatrix{Complex{T}}, dA::AbstractMatrix{Complex{T}}, ddA::AbstractMatrix{Complex{T}}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    N = length(pts)
    @use_threads multithreading=multithreaded for j in 1:N
        @inbounds for i in 1:N
            val, dval, ddval = entry_fn(pts, Rmat, G, k, i, j)
            Iij = i==j ? one(Complex{T}) : zero(Complex{T})
            A[i,j] = Iij-val
            dA[i,j] = -dval
            ddA[i,j] = -ddval
        end
    end
    return A, dA, ddA
end

# Assemble the symmetry-reduced Fredholm matrix A(k) and its first two wavenumber derivatives in place. The full-boundary kernel contributions are
# folded over each source symmetry orbit using the corresponding character phases, exactly as in the ordinary symmetry-reduced DLP and CFIE assembly.
function _ebim_fredholm_reduced_with_derivatives!(entry_fn, A::AbstractMatrix{Complex{T}}, dA::AbstractMatrix{Complex{T}}, ddA::AbstractMatrix{Complex{T}}, pts::BoundaryPoints{T}, Rmat::AbstractMatrix{T}, G::BoundaryGeomCache{T}, orbits::SymmetryOrbitMap{T}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    m = fundamental_size(orbits)
    N = length(orbits)
    fund = orbits.fundamental_indices
    orbit_of = orbits.orbit_of
    phase = orbits.phase
    images = [Int[] for _ in 1:m]
    @inbounds for j in 1:N
        push!(images[orbit_of[j]], j)
    end
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        @inbounds for a in 1:m
            i = fund[a]
            acc = zero(Complex{T})
            dacc = zero(Complex{T})
            ddacc = zero(Complex{T})
            for j in images[b]
                val, dval, ddval = entry_fn(pts, Rmat, G, k, i, j)
                acc += phase[j]*val
                dacc += phase[j]*dval
                ddacc += phase[j]*ddval
            end
            A[a,b] = -acc
            dA[a,b] = -dacc
            ddA[a,b] = -ddacc
        end
        A[b,b] += one(Complex{T})
    end
    return A, dA, ddA
end

# Assemble the full Fredholm matrix and its first two wavenumber derivatives for a multiply connected domain. Same-component blocks use the DLP or CFIE
# kernel and Kress quadrature scheme of their component solver, while cross-component blocks are smooth and use direct quadrature.
function _composite_fredholm_full_with_derivatives!(A::AbstractMatrix{Complex{T}}, dA::AbstractMatrix{Complex{T}}, ddA::AbstractMatrix{Complex{T}}, solver::CompositeBIMSolver, comp_pts::Vector{BoundaryPoints{T}}, Gs::Vector{BoundaryGeomCache{T}}, Rmats::Vector{Matrix{T}}, offs::Vector{Int}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    nc = length(comp_pts)
    @inbounds for a in 1:nc
        cs = solver.component_solvers[a]
        pa = comp_pts[a]
        Ga = Gs[a]
        Ra = Rmats[a]
        Na = length(pa)
        off = offs[a]
        @use_threads multithreading=(multithreaded && Na>=32) for j in 1:Na
            gj = off+j-1
            @inbounds for i in 1:Na
                gi = off+i-1
                val, dval, ddval = _composite_component_kernel_entry_with_derivatives(cs, pa, Ra, Ga, k, i, j)
                Iij = i==j ? one(Complex{T}) : zero(Complex{T})
                A[gi,gj] = Iij-val
                dA[gi,gj] = -dval
                ddA[gi,gj] = -ddval
            end
        end
    end
    for b in 1:nc
        csb = solver.component_solvers[b]
        pb = comp_pts[b]
        offb = offs[b]
        Nb = length(pb)
        for a in 1:nc
            a==b && continue
            pa = comp_pts[a]
            offa = offs[a]
            Na = length(pa)
            @use_threads multithreading=(multithreaded && Na>=16) for i in 1:Na
                gi = offa+i-1
                xi, yi = pa.xy[i]
                @inbounds for j in 1:Nb
                    gj = offb+j-1
                    val, dval, ddval = _composite_cross_kernel_entry_with_derivatives(csb, pb, xi, yi, k, j)
                    A[gi,gj] = -val
                    dA[gi,gj] = -dval
                    ddA[gi,gj] = -ddval
                end
            end
        end
    end
    return A, dA, ddA
end

# Assemble the symmetry-reduced Fredholm matrix and its first two wavenumber derivatives for a multiply connected domain. Same-component and smooth
# cross-component interactions are evaluated with their respective kernels and folded over the source symmetry orbits using the prescribed character phases.
function _composite_fredholm_reduced_with_derivatives!(A::AbstractMatrix{Complex{T}}, dA::AbstractMatrix{Complex{T}}, ddA::AbstractMatrix{Complex{T}}, solver::CompositeBIMSolver, comp_pts::Vector{BoundaryPoints{T}}, Gs::Vector{BoundaryGeomCache{T}}, Rmats::Vector{Matrix{T}}, offs::Vector{Int}, g2c::Vector{Int}, g2l::Vector{Int}, orbits::SymmetryOrbitMap{T}, k::Union{T,Complex{T}}; multithreaded::Bool=true) where {T<:Real}
    m = fundamental_size(orbits)
    N = length(orbits)
    fund = orbits.fundamental_indices
    orbit_of = orbits.orbit_of
    phase = orbits.phase
    images = [Int[] for _ in 1:m]
    @inbounds for j in 1:N
        push!(images[orbit_of[j]], j)
    end
    @use_threads multithreading=(multithreaded && m>=32) for b in 1:m
        @inbounds for a in 1:m
            gi = fund[a]
            ca = g2c[gi]
            ia = g2l[gi]
            acc = zero(Complex{T})
            dacc = zero(Complex{T})
            ddacc = zero(Complex{T})
            for gj in images[b]
                cb = g2c[gj]
                jb = g2l[gj]
                ph = phase[gj]
                if ca==cb
                    cs = solver.component_solvers[ca]
                    val, dval, ddval = _composite_component_kernel_entry_with_derivatives(cs, comp_pts[ca], Rmats[ca], Gs[ca], k, ia, jb)
                else
                    csb = solver.component_solvers[cb]
                    xi, yi = comp_pts[ca].xy[ia]
                    val, dval, ddval = _composite_cross_kernel_entry_with_derivatives(csb, comp_pts[cb], xi, yi, k, jb)
                end
                acc += ph*val
                dacc += ph*dval
                ddacc += ph*ddval
            end
            A[a,b] = -acc
            dA[a,b] = -dacc
            ddA[a,b] = -ddacc
        end
        A[b,b] += one(Complex{T})
    end
    return A, dA, ddA
end

################################################################################
################### PER-KERNEL construct_matrices DISPATCH ###################
################################################################################

function _ebim_construct_matrices(cs::DoubleLayerPotentialSolver, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}
    kT = _bim_widen_k(T, k)
    N = length(pts)
    graded = _is_nontrivial_dlp_grading(pts)
    G = boundary_geom_cache(pts, graded)
    Rmat = zeros(T, N, N)
    kress_R!(Rmat)
    if cs.symmetry===nothing
        A = Matrix{Complex{T}}(undef, N, N)
        dA = similar(A)
        ddA = similar(A)
        _ebim_fredholm_full_with_derivatives!(_dlp_kernel_entry_with_derivatives, A, dA, ddA, pts, Rmat, G, kT; multithreaded)
        return A, dA, ddA
    end
    orbits = _fold_boundary(T, cs.billiard, N, cs.symmetry, cs.character)
    m = fundamental_size(orbits)
    A = Matrix{Complex{T}}(undef, m, m)
    dA = similar(A)
    ddA = similar(A)
    _ebim_fredholm_reduced_with_derivatives!(_dlp_kernel_entry_with_derivatives, A, dA, ddA, pts, Rmat, G, orbits, kT; multithreaded)
    return A, dA, ddA
end

function _ebim_construct_matrices(cs::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}
    kT = _bim_widen_k(T, k)
    N = length(pts)
    graded = _is_nontrivial_dlp_grading(pts)
    G = boundary_geom_cache(pts, graded)
    Rmat = zeros(T, N, N)
    kress_R!(Rmat)
    if cs.symmetry===nothing
        A = Matrix{Complex{T}}(undef, N, N)
        dA = similar(A)
        ddA = similar(A)
        _ebim_fredholm_full_with_derivatives!(_cfie_kernel_entry_with_derivatives, A, dA, ddA, pts, Rmat, G, kT; multithreaded)
        return A, dA, ddA
    end
    orbits = _fold_boundary(T, cs.billiard, N, cs.symmetry, cs.character)
    m = fundamental_size(orbits)
    A = Matrix{Complex{T}}(undef, m, m)
    dA = similar(A)
    ddA = similar(A)
    _ebim_fredholm_reduced_with_derivatives!(_cfie_kernel_entry_with_derivatives, A, dA, ddA, pts, Rmat, G, orbits, kT; multithreaded)
    return A, dA, ddA
end

function _ebim_construct_matrices(cs::CompositeBIMSolver{T}, pts::BoundaryPoints{T}, k; multithreaded::Bool=true) where {T<:Real}
    kT = _bim_widen_k(T, k)
    nc = length(cs.component_solvers)
    N = length(pts)
    offs = _composite_offsets(pts, nc)
    comp_pts = [_composite_component_slice(pts, offs[a]:offs[a+1]-1, a) for a in 1:nc]
    Gs = Vector{BoundaryGeomCache{T}}(undef, nc)
    Rmats = Vector{Matrix{T}}(undef, nc)
    @inbounds for a in 1:nc
        graded = _is_nontrivial_dlp_grading(comp_pts[a])
        Gs[a] = boundary_geom_cache(comp_pts[a], graded)
        Na = length(comp_pts[a])
        Ra = zeros(T, Na, Na)
        kress_R!(Ra)
        Rmats[a] = Ra
    end
    if cs.symmetry === nothing
        A = Matrix{Complex{T}}(undef, N, N)
        dA = similar(A)
        ddA = similar(A)
        _composite_fredholm_full_with_derivatives!(A, dA, ddA, cs, comp_pts, Gs, Rmats, offs, kT; multithreaded)
        return A, dA, ddA
    end
    orbits = _composite_symmetry_orbits(T, cs, pts)
    m = fundamental_size(orbits)
    g2c, g2l = _composite_global_to_local(offs)
    A = Matrix{Complex{T}}(undef, m, m)
    dA = similar(A)
    ddA = similar(A)
    _composite_fredholm_reduced_with_derivatives!(A, dA, ddA, cs, comp_pts, Gs, Rmats, offs, g2c, g2l, orbits, kT; multithreaded)
    return A, dA, ddA
end

################################################################################
############################## PUBLIC API ######################################
################################################################################

# Assemble A(k), A'(k), and A''(k) for a DLP discretization using Chebyshev-interpolated H₀⁽¹⁾, H₁⁽¹⁾, J₀, and J₁. The Fredholm discretization
# and analytic wavenumber derivatives are identical to the direct EBIM path; only radial special-function evaluation is replaced by Chebyshev interpolation.
function _ebim_construct_matrices_cheb(cs::DoubleLayerPotentialSolver, pts::BoundaryPoints{T}, k, cfg::ChebyshevConfig, cache::EBIMCache; multithreaded::Bool = true) where {T<:Real}
    T === Float64 || error("Chebyshev-accelerated EBIM evaluation currently requires a Float64 kernel; received numeric type $T. Construct the ExpandedBIMSolver with use_chebyshev=false.")
    kc = ComplexF64(_bim_widen_k(T, k)); N = length(pts)
    G = cache.G; Rmat = cache.Rmat
    rmin, rmax = _cheb_geom_rminmax(G, [kc])
    plans0, plans1, plansj0, plansj1, _ = tune_cfie_cheb_plans(rmin, rmax, [kc], cfg)
    plan0 = plans0[1]; plan1 = plans1[1]; planj0 = plansj0[1]; planj1 = plansj1[1]
    cache.cheb_lookup === nothing && (cache.cheb_lookup = ChebRadialLookupCache(G, plan1, planj1; multithreaded))
    C = cache.cheb_lookup
    entry_fn = (p, R, Gc, kk, i, j) -> _dlp_kernel_entry_with_derivatives_cheb(p, R, Gc, C, kk, plan0, plan1, planj0, planj1, i, j)
    if cache.orbits === nothing
        A = Matrix{ComplexF64}(undef, N, N); dA = similar(A); ddA = similar(A)
        _ebim_fredholm_full_with_derivatives!(entry_fn, A, dA, ddA, pts, Rmat, G, kc; multithreaded)
        return A, dA, ddA
    end
    m = fundamental_size(cache.orbits)
    A = Matrix{ComplexF64}(undef, m, m); dA = similar(A); ddA = similar(A)
    _ebim_fredholm_reduced_with_derivatives!(entry_fn, A, dA, ddA, pts, Rmat, G, cache.orbits, kc; multithreaded)
    return A, dA, ddA
end

# Assemble A(k), A'(k), and A''(k) for a CFIE discretization using Chebyshev-interpolated H₀⁽¹⁾, H₁⁽¹⁾, J₀, and J₁. The Fredholm discretization
# and analytic wavenumber derivatives are identical to the direct EBIM path; only radial special-function evaluation is replaced by Chebyshev interpolation.
function _ebim_construct_matrices_cheb(cs::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, k, cfg::ChebyshevConfig, cache::EBIMCache; multithreaded::Bool = true) where {T<:Real}
    T === Float64 || error("Chebyshev-accelerated EBIM evaluation currently requires a Float64 kernel; received numeric type $T. Construct the ExpandedBIMSolver with use_chebyshev=false.")
    kc = ComplexF64(_bim_widen_k(T, k)); N = length(pts)
    G = cache.G; Rmat = cache.Rmat
    rmin, rmax = _cheb_geom_rminmax(G, [kc])
    plans0, plans1, plansj0, plansj1, _ = tune_cfie_cheb_plans(rmin, rmax, [kc], cfg)
    plan0 = plans0[1]; plan1 = plans1[1]; planj0 = plansj0[1]; planj1 = plansj1[1]
    cache.cheb_lookup === nothing && (cache.cheb_lookup = ChebRadialLookupCache(G, plan1, planj1; multithreaded))
    C = cache.cheb_lookup
    entry_fn = (p, R, Gc, kk, i, j) -> _cfie_kernel_entry_with_derivatives_cheb(p, R, Gc, C, kk, plan0, plan1, planj0, planj1, i, j)
    if cache.orbits === nothing
        A = Matrix{ComplexF64}(undef, N, N); dA = similar(A); ddA = similar(A)
        _ebim_fredholm_full_with_derivatives!(entry_fn, A, dA, ddA, pts, Rmat, G, kc; multithreaded)
        return A, dA, ddA
    end
    m = fundamental_size(cache.orbits)
    A = Matrix{ComplexF64}(undef, m, m); dA = similar(A); ddA = similar(A)
    _ebim_fredholm_reduced_with_derivatives!(entry_fn, A, dA, ddA, pts, Rmat, G, cache.orbits, kc; multithreaded)
    return A, dA, ddA
end

_ebim_construct_matrices_cheb(cs::CompositeBIMSolver, pts::BoundaryPoints, k, cfg::ChebyshevConfig, cache; multithreaded::Bool=true) = error("Chebyshev-accelerated EBIM evaluation is not yet implemented for CompositeBIMSolver kernels. Construct the ExpandedBIMSolver with use_chebyshev=false.")

# Tune the H₀⁽¹⁾, H₁⁽¹⁾, J₀, and J₁ Chebyshev approximations over the radial interval required by the boundary geometry at expansion center `k`. The
# returned configuration uses `param_strategy=:manual` so it can be reused across subsequent EBIM evaluations without repeating the tuning procedure.
function tune_ebim_cheb_config(cs::Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}, pts::BoundaryPoints, k, cfg::ChebyshevConfig)
    z = ComplexF64[ComplexF64(k)]; G = boundary_geom_cache(pts, _is_nontrivial_dlp_grading(pts))
    rmin, rmax = _cheb_geom_rminmax(G, z)
    _, _, _, _, tuned = tune_cfie_cheb_plans(rmin, rmax, z, cfg)
    return ChebyshevConfig(_bim_numeric_type(cs); n_panels_h=tuned.n_panels_h, M_h=tuned.M_h, n_panels_j=tuned.n_panels_j, M_j=tuned.M_j, tol=tuned.tol, max_iter=tuned.max_iter, sampling_points=tuned.sampling_points, grow_panels=tuned.grow_panels, grow_M=tuned.grow_M, param_strategy=:manual)
end

"""
    construct_matrices(solver::ExpandedBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config)

Assemble the Fredholm matrix and its first two wavenumber derivatives at `k`.

The returned matrices are

    A   = A(k),
    dA  = A'(k),
    ddA = A''(k),

and define the second-order local expansion used by EBIM,

    A(k + ε) = A + ε*dA + (ε²/2)*ddA + O(ε³).

If `solver.use_chebyshev` is `true`, supported kernels evaluate their radial
special functions through Chebyshev interpolation. 

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver defining the underlying BIM kernel.
* `pts::BoundaryPoints`: Discretized physical boundary used to assemble the matrices.
* `k`: Real or complex expansion wavenumber.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded matrix assembly.
* `cheb_config::ChebyshevConfig = solver.cheb_config`: Chebyshev configuration.
* `cache`: Optional precomputed boundary-geometry cache to accelerate matrix construction.

## Returns
* `A::Matrix`: Fredholm matrix `A(k)`.
* `dA::Matrix`: First wavenumber derivative `A'(k)`.
* `ddA::Matrix`: Second wavenumber derivative `A''(k)`.
"""
function construct_matrices(solver::ExpandedBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config, cache=nothing)
    solver.use_chebyshev || return _ebim_construct_matrices(solver.kernel, pts, k; multithreaded)
    cache===nothing && (cache = EBIMCache(solver.kernel, pts))
    return _ebim_construct_matrices_cheb(solver.kernel, pts, k, cheb_config, cache; multithreaded)
end

"""
    solve(solver::ExpandedBIMSolver, pts::BoundaryPoints, k, nlevels::Int; multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config)

Compute second-order EBIM eigenvalue estimates around the expansion center `k`.

The method assembles `A(k)`, `A'(k)`, and `A''(k)`, factorizes `A(k)`, and
computes the dominant eigenpairs of `A(k)⁻¹A'(k)`. These correspond to the
smallest first-order eigenvalue displacements from `k`. Each retained
eigenvalue is then corrected to second order using the associated left and
right eigenvectors.

`nlevels` specifies the minimum number of converged local eigenpairs required.
Several additional eigenpairs are requested internally to provide a small
convergence margin.

The returned `ts` contain the magnitudes of the total EBIM corrections and
therefore measure the displacement of each estimated eigenvalue from the
expansion center. They are not Fredholm residuals or eigenvalue-error
estimates.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `pts::BoundaryPoints`: Boundary discretization at the expansion center.
* `k`: Expansion wavenumber.
* `nlevels::Int`: Number of local eigenvalue estimates to compute.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded matrix assembly.
* `cheb_config::ChebyshevConfig = solver.cheb_config`: Chebyshev configuration.
* `cache`: Optional precomputed boundary-geometry cache to accelerate matrix construction.

## Returns
* `ks::Vector{Complex{T}}`: Second-order EBIM eigenvalue estimates around `k`.
* `ts::Vector{T}`: Magnitudes `|ε₁ + ε₂|` of the corresponding displacements from `k`.
"""
function solve(solver::ExpandedBIMSolver, pts::BoundaryPoints, k, nlevels::Int; multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config, cache=nothing)
    T = _bim_numeric_type(solver)
    A, dA, ddA = construct_matrices(solver, pts, k; multithreaded, cheb_config, cache)
    n = size(A, 1); nev = min(nlevels+5, n-1)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        F = lu!(A)
        Ft = adjoint(F); dAt = adjoint(dA)
        op_r = x -> F \ (dA*x)
        op_l = x -> dAt*(Ft \ x)
        μ, (VR, UL), (info_r, info_l) = KrylovKit.bieigsolve((op_r, op_l), n, nev, :LM, Complex{T}; tol=solver.tol, maxiter=solver.maxiter, krylovdim=max(solver.krylovdim, 2*nev+1))
        nconv = min(info_r.converged, info_l.converged)
        nconv >= nlevels || error("EBIM Krylov solve converged only $nconv eigenpairs; requested $nlevels")
        p = sortperm(abs.(μ[1:nconv]); rev=true); nkeep = min(nev, nconv)
        ks = Vector{Complex{T}}(undef, nkeep); ts = Vector{T}(undef, nkeep); buf = Vector{Complex{T}}(undef, n)
        @inbounds for q in 1:nkeep
            j = p[q]; λ = inv(μ[j]); v = VR[j]; u = Ft\UL[j]; ε1 = -λ
            mul!(buf, ddA, v); num = dot(u, buf)
            mul!(buf, dA, v); den = dot(u, buf)
            ε2 = abs(den)>eps(T) ? -T(0.5)*ε1^2*(num/den) : zero(ε1)
            corr = ε1+ε2; ks[q] = k+corr; ts[q] = abs(corr)
        end
    end
    return ks, ts
end

"""
    solve_wavenumber(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true, nlevels::Int=5, cheb_config::ChebyshevConfig=solver.cheb_config) where {Bi<:AbsBilliard}

Compute local second-order EBIM eigenvalue estimates around the expansion
center `k`.

The boundary discretization is generated at `k` with [`evaluate_points`](@ref),
after which the local eigenvalues are obtained from the second-order Fredholm
expansion.

`dk` is retained for interface compatibility with other accelerated BIM
solvers but does not define an EBIM search window and is not used by the local
Taylor expansion.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `billiard::Bi`: Billiard geometry.
* `k`: Expansion wavenumber.
* `dk`: Unused compatibility argument.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded matrix assembly.
* `nlevels::Int = 5`: Number of local eigenvalue estimates to compute.
* `cheb_config::ChebyshevConfig = solver.cheb_config`: Configuration for Chebyshev acceleration.

## Returns
* `ks::Vector{Complex{T}}`: Second-order EBIM eigenvalue estimates around `k`.
* `ts::Vector{T}`: Magnitudes of the corresponding EBIM displacements from `k`.
"""
function solve_wavenumber(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true, nlevels::Int=5, cheb_config::ChebyshevConfig=solver.cheb_config) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k)
    return solve(solver, pts, k, nlevels; multithreaded, cheb_config)
end

"""
    solve_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true, nlevels::Int=5) where {Bi<:AbsBilliard}

Compute local second-order EBIM eigenvalue estimates around a collection of
expansion centers.

A separate local Fredholm expansion is constructed around every entry of `k`,
and the resulting local eigenvalue estimates are concatenated into a single
spectrum. Different expansion centers may produce overlapping estimates of the
same physical eigenvalue; this function does not remove such duplicates.

`dk` is retained for interface compatibility with other accelerated BIM
solvers and is not used by the local EBIM expansion.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `billiard::Bi`: Billiard geometry.
* `k`: Collection of expansion wavenumbers.
* `dk`: Unused compatibility argument.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded matrix assembly.
* `nlevels::Int = 5`: Number of local eigenvalue estimates to compute.

## Returns
* `ks::Vector{Complex{T}}`: Concatenated second-order EBIM eigenvalue estimates.
* `ts::Vector{T}`: Magnitudes of the corresponding displacements from their expansion centers.
"""
function solve_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true, nlevels::Int=5) where {Bi<:AbsBilliard}
    T = _bim_numeric_type(solver); kv = collect(k)
    isempty(kv) && return Complex{T}[], T[]
    cheb_config = solver.cheb_config
    if solver.use_chebyshev && solver.cheb_config.param_strategy!==:manual
        imax = argmax(real.(kv))
        pts_max = evaluate_points(solver, billiard, kv[imax])
        cheb_config = tune_ebim_cheb_config(solver.kernel, pts_max, kv[imax], solver.cheb_config)
    end
    ks = Complex{T}[]; ts = T[]
    for ki in kv
        ksi, tsi = solve_wavenumber(solver, billiard, ki, dk; multithreaded, nlevels, cheb_config)
        append!(ks, ksi); append!(ts, tsi)
    end
    return ks, ts
end