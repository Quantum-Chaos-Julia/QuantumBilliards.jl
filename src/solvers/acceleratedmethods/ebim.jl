################################################################################
# EXPANDED BOUNDARY INTEGRAL METHOD (EBIM)
#
# This file implements a local second-order eigensolver for boundary-integral
# nonlinear eigenvalue problems
#
#                              A(k)v = 0,
#
# where A(k) is the Fredholm matrix produced by a DLP, CFIE, or composite BIM
# discretization. Around an expansion center k₀,
#
#   A(k₀ + ε) = A₀ + εA₁ + (ε²/2)A₂ + O(ε³),
#
# with
#
#   A₀ = A(k₀),    A₁ = A'(k₀),    A₂ = A''(k₀).
#
# The first-order displacement follows from
#
#                         A₀v = λA₁v,
#
# so that ε₁ = -λ. For the corresponding left generalized eigenvector u,
#
#                 ε₂ = -(ε₁²/2)(uᴴA₂v)/(uᴴA₁v),
#
# and
#
#                         kEBIM = k₀ + ε₁ + ε₂.
#
# MATRIX CONSTRUCTION
# -------------------
# Two matrix-construction paths are supported.
#
# Direct:
#   A(k), A'(k), and A''(k) are assembled directly from analytic derivatives
#   of the BIM kernels. This path is retained as an independent reference and
#   fallback implementation.
#
# Taylor:
#   The spectrum is tessellated by Taylor panels. For one panel centered at kc,
#
#       A(kc + δ) = Σₗ₌₀ᵖ Aₗδˡ,
#
#   the normalized coefficient matrices Aₗ=A⁽ˡ⁾(kc)/l! are assembled once
#   using the shared analytic BIM Taylor backend. All EBIM expansion centers
#   assigned to that panel obtain A, A', and A'' by simultaneous Horner
#   evaluation of the cached coefficients.
#
# Only one Taylor panel is retained at a time. Thus the Taylor storage is
# O((p+1)N²), independent of the total number of EBIM expansion centers.
################################################################################

"""
    ExpandedBIMSolver{T,K} <: AcceleratedBIMSolver

Local second-order eigensolver for boundary-integral nonlinear eigenvalue
problems.

`ExpandedBIMSolver` wraps a [`SweepBIMSolver`](@ref) and computes local
eigenvalue estimates from `A(k)`, `A'(k)`, and `A''(k)`. Matrix construction
may either use direct analytic kernel derivatives or the shared analytic
Taylor representation of the Fredholm operator.

With Taylor acceleration enabled, a polynomial panel is constructed once and
reused for all EBIM expansion centers lying within its validity radius.

## Attributes
* `kernel::K`: Wrapped [`SweepBIMSolver`](@ref) defining the Fredholm operator.
* `use_taylor::Bool`: Whether to use cached analytic Taylor panels.
* `taylor_degree::Int`: Degree of each analytic Fredholm Taylor expansion.
* `taylor_radius::T`: Maximum permitted distance from a Taylor panel center.
* `taylor_tol::T`: Relative tolerance used to validate a Taylor panel.
* `tol::T`: Convergence tolerance used by the Krylov eigensolver.
* `maxiter::Int`: Maximum number of Krylov iterations.
* `krylovdim::Int`: Minimum Krylov subspace dimension.

## API
The principal operations are [`evaluate_points`](@ref),
[`construct_matrices`](@ref), [`solve`](@ref), [`solve_wavenumber`](@ref),
[`solve_spectrum`](@ref), and [`compute_spectrum`](@ref).
"""
struct ExpandedBIMSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K
    use_taylor::Bool
    taylor_degree::Int
    taylor_radius::T
    taylor_tol::T
    tol::T
    maxiter::Int
    krylovdim::Int
end

"""
    ExpandedBIMSolver(kernel::K; use_taylor::Bool=true, taylor_degree::Int=16, taylor_radius::Real=0.5, taylor_tol::Real=1e-11, tol::Real=1e-12, maxiter::Int=5000, krylovdim::Int=40) where {K<:SweepBIMSolver}

Construct an [`ExpandedBIMSolver`](@ref).

## Arguments
* `kernel::K`: [`SweepBIMSolver`](@ref) defining the Fredholm operator.

## Keyword Arguments
* `use_taylor::Bool = true`: Enable analytic Taylor-panel acceleration.
* `taylor_degree::Int = 16`: Taylor polynomial degree.
* `taylor_radius::Real = 0.5`: Maximum permitted distance from a Taylor center.
* `taylor_tol::Real = 1e-11`: Relative Taylor validation tolerance.
* `tol::Real = 1e-12`: Krylov eigensolver tolerance.
* `maxiter::Int = 5000`: Maximum number of Krylov iterations.
* `krylovdim::Int = 40`: Minimum Krylov subspace dimension.

## Returns
* `solver::ExpandedBIMSolver`: Configured EBIM solver.
"""
function ExpandedBIMSolver(kernel::K; use_taylor::Bool = true, taylor_degree::Int = 16, taylor_radius::Real = 0.5, taylor_tol::Real = 1e-11, tol::Real = 1e-12, maxiter::Int = 5000, krylovdim::Int = 40) where {K<:SweepBIMSolver}
    T = _bim_numeric_type(kernel)
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive"))
    krylovdim >= 2 || throw(ArgumentError("krylovdim must be at least 2"))
    return ExpandedBIMSolver{T,K}(kernel, use_taylor, taylor_degree, T(taylor_radius), T(taylor_tol), T(tol), maxiter, krylovdim)
end

_bim_numeric_type(::ExpandedBIMSolver{T}) where {T} = T

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
    orbits = _fold_boundary(T, cs.billiard, N, cs.symmetry, cs.sym_characters)
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
    orbits = _fold_boundary(T, cs.billiard, N, cs.symmetry, cs.sym_characters)
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
# TAYLOR-PANEL CACHE
################################################################################

"""
    EBIMTaylorCache

Cached normalized Taylor coefficient matrices for one EBIM Taylor panel.

For a panel centered at `k0`,

    A(k0 + δ) = Σₗ₌₀ᵖ Aₗδˡ + O(δᵖ⁺¹),

the entries are stored as

    coeffs[i,j,l+1] = (Aₗ)ᵢⱼ = Aᵢⱼ⁽ˡ⁾(k0)/l!.

The first two dimensions therefore form a contiguous coefficient matrix for
each Taylor order. This layout is optimized for repeated matrix-valued Horner
evaluation during an EBIM spectrum sweep.

Only one panel is required during a panel-wise spectrum sweep.

## Attributes
* `k0::Float64`: Taylor expansion center.
* `degree::Int`: Taylor polynomial degree.
* `coeffs::Array{ComplexF64,3}`: Normalized Fredholm Taylor coefficients.
"""
struct EBIMTaylorCache
    k0::Float64
    degree::Int
    coeffs::Array{ComplexF64,3}
end

"""
    EBIMTaylorCache(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, p::Int; multithreaded::Bool=true)

Construct one cached Fredholm Taylor panel.

For a full-boundary problem, normalized Taylor coefficients are copied
directly from [`_fredholm_taylor!`](@ref). For a symmetry-reduced problem, the
raw kernel coefficients are folded over each source orbit before the Fredholm
identity is added.

The coefficient tensor is stored as

    coeffs[i,j,l+1] = Aᵢⱼ⁽ˡ⁾(k0)/l!,

so each coefficient matrix `coeffs[:,:,l+1]` occupies one contiguous block of
memory. This favors repeated matrix-valued Horner evaluation over the one-time
panel construction.

## Arguments
* `solver::SweepBIMSolver`: BIM solver defining the Fredholm operator.
* `pts::BoundaryPoints{Float64}`: Full physical-boundary discretization.
* `k0::Float64`: Taylor expansion center.
* `p::Int`: Taylor polynomial degree.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded coefficient assembly.

## Returns
* `cache::EBIMTaylorCache`: Cached Taylor panel.
"""
function EBIMTaylorCache(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, p::Int; multithreaded::Bool = true)::EBIMTaylorCache
    p >= 2 || throw(ArgumentError("EBIM Taylor degree must be at least 2"))
    cache = _taylor_cache(solver, pts)
    if solver.symmetry === nothing
        N = length(pts)
        coeffs = Array{ComplexF64,3}(undef, N, N, p + 1)
        work = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()]
        @use_threads multithreading=multithreaded for j = 1:N
            w = work[Threads.threadid()]
            @inbounds for i = 1:N
                a = _fredholm_taylor!(cache, w, k0, p, i, j)
                @simd for l = 1:p + 1
                    coeffs[i, j, l] = a[l]
                end
            end
        end
        return EBIMTaylorCache(k0, p, coeffs)
    end
    orbits = solver isa CompositeBIMSolver ? _composite_symmetry_orbits(Float64, solver, pts) : _fold_boundary(Float64, solver.billiard, length(pts.xy), solver.symmetry, solver.sym_characters)
    m = fundamental_size(orbits)
    fund = orbits.fundamental_indices
    orbit_of = orbits.orbit_of
    phase = orbits.phase
    images = [Int[] for _ = 1:m]
    @inbounds for j = eachindex(orbit_of)
        push!(images[orbit_of[j]], j)
    end
    coeffs = Array{ComplexF64,3}(undef, m, m, p + 1)
    work = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()]
    @use_threads multithreading=multithreaded for bcol = 1:m
        w = work[Threads.threadid()]
        imgs = images[bcol]
        gj = imgs[1]
        χ = phase[gj]
        @inbounds for arow = 1:m
            gi = fund[arow]
            a = _kernel_taylor!(cache, w, k0, p, gi, gj)
            @simd for l = 1:p + 1
                coeffs[arow, bcol, l] = -χ * a[l]
            end
        end
        @inbounds for ii = 2:length(imgs)
            gj = imgs[ii]
            χ = phase[gj]
            for arow = 1:m
                gi = fund[arow]
                a = _kernel_taylor!(cache, w, k0, p, gi, gj)
                @simd for l = 1:p + 1
                    coeffs[arow, bcol, l] -= χ * a[l]
                end
            end
        end
        coeffs[bcol, bcol, 1] += 1.0
    end
    return EBIMTaylorCache(k0, p, coeffs)
end

"""
    _construct_matrices!(solver::ExpandedBIMSolver, A::Matrix{ComplexF64}, dA::Matrix{ComplexF64}, ddA::Matrix{ComplexF64}, cache::EBIMTaylorCache, k::Real)

Evaluate `A(k)`, `A'(k)`, and `A''(k)` in place from a cached Taylor panel.

All three matrices are evaluated simultaneously by matrix-valued Horner
recurrence. Since each Taylor coefficient matrix occupies one contiguous block
of `cache.coeffs`, the recurrence streams linearly through memory.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `A::Matrix{ComplexF64}`: Output Fredholm matrix.
* `dA::Matrix{ComplexF64}`: Output first derivative.
* `ddA::Matrix{ComplexF64}`: Output second derivative.
* `cache::EBIMTaylorCache`: Cached Taylor panel.
* `k::Real`: Evaluation wavenumber.

## Returns
* `A::Matrix{ComplexF64}`: Fredholm matrix.
* `dA::Matrix{ComplexF64}`: First derivative.
* `ddA::Matrix{ComplexF64}`: Second derivative.
"""
function _construct_matrices!(solver::ExpandedBIMSolver, A::Matrix{ComplexF64}, dA::Matrix{ComplexF64}, ddA::Matrix{ComplexF64}, cache::EBIMTaylorCache, k::Real)
    δ = Float64(k) - cache.k0
    abs(δ) <= solver.taylor_radius || throw(ArgumentError("wavenumber $k lies outside the Taylor radius $(solver.taylor_radius) about $(cache.k0)"))
    C = cache.coeffs
    p = cache.degree
    n2 = length(A)
    off = p * n2
    @inbounds @simd for q = 1:n2
        A[q] = C[off + q]
        dA[q] = 0.0
        ddA[q] = 0.0
    end
    @inbounds for l = p:-1:1
        off = (l - 1) * n2
        @simd for q = 1:n2
            ddA[q] = ddA[q] * δ + 2 * dA[q]
            dA[q] = dA[q] * δ + A[q]
            A[q] = A[q] * δ + C[off + q]
        end
    end
    return A, dA, ddA
end

"""
    construct_matrices(solver::ExpandedBIMSolver, cache::EBIMTaylorCache, k::Real)

Evaluate `A(k)`, `A'(k)`, and `A''(k)` from a cached Taylor panel.

This allocating convenience method is intended for individual evaluations.
Spectrum sweeps reuse persistent matrix buffers through
[`_construct_matrices!`](@ref).

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `cache::EBIMTaylorCache`: Taylor panel.
* `k::Real`: Evaluation wavenumber.

## Returns
* `A::Matrix{ComplexF64}`: Fredholm matrix.
* `dA::Matrix{ComplexF64}`: First derivative.
* `ddA::Matrix{ComplexF64}`: Second derivative.
"""
function construct_matrices(solver::ExpandedBIMSolver, cache::EBIMTaylorCache, k::Real)
    N = size(cache.coeffs, 1)
    A = Matrix{ComplexF64}(undef, N, N)
    dA = similar(A)
    ddA = similar(A)
    return _construct_matrices!(solver, A, dA, ddA, cache, k)
end

"""
    construct_matrices(solver::ExpandedBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool=true)

Assemble `A(k)`, `A'(k)`, and `A''(k)` directly from analytic BIM kernel
derivatives.

This method intentionally uses the independent direct EBIM implementation.
Taylor acceleration is applied through an [`EBIMTaylorCache`](@ref), rather
than implicitly constructing a one-point Taylor panel.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k`: Expansion wavenumber.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded matrix assembly.

## Returns
* `A::Matrix`: Fredholm matrix `A(k)`.
* `dA::Matrix`: First derivative `A'(k)`.
* `ddA::Matrix`: Second derivative `A''(k)`.
"""
function construct_matrices(solver::ExpandedBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool = true)
    return _ebim_construct_matrices(solver.kernel, pts, k; multithreaded)
end

################################################################################
# LOCAL EBIM EIGENSOLVE
################################################################################

"""
    _solve(solver::ExpandedBIMSolver, A::Matrix{Complex{T}}, dA::Matrix{Complex{T}}, ddA::Matrix{Complex{T}}, k, nlevels::Int) where {T<:Real}

Solve one local second-order EBIM problem from preassembled matrices.

The dominant eigenpairs of `A⁻¹dA` provide the first-order corrections. The
corresponding left and right generalized eigenvectors are then used to form
the second-order correction from `ddA`.

`A` is destroyed by its in-place LU factorization.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `A::Matrix{Complex{T}}`: Fredholm matrix at the EBIM expansion center.
* `dA::Matrix{Complex{T}}`: First wavenumber derivative.
* `ddA::Matrix{Complex{T}}`: Second wavenumber derivative.
* `k`: EBIM expansion center.
* `nlevels::Int`: Minimum number of converged local candidates.

## Returns
* `ks::Vector{Complex{T}}`: Second-order EBIM eigenvalue estimates.
* `ts::Vector{T}`: Magnitudes of the corresponding total corrections.
"""
function _solve(solver::ExpandedBIMSolver, A::Matrix{Complex{T}}, dA::Matrix{Complex{T}}, ddA::Matrix{Complex{T}}, k, nlevels::Int) where {T<:Real}
    n::Int = size(A, 1)
    n >= 2 || return Complex{T}[], T[]
    nev::Int = min(nlevels + 5, n - 1)
    ks = Complex{T}[]
    ts = T[]
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        F = lu!(A)
        Ft = adjoint(F)
        dAt = adjoint(dA)
        op_r = x -> F \ (dA * x)
        op_l = x -> dAt * (Ft \ x)
        μ, (VR, UL), (info_r, info_l) = KrylovKit.bieigsolve((op_r, op_l), n, nev, :LM, Complex{T}; tol = solver.tol, maxiter = solver.maxiter, krylovdim = max(solver.krylovdim, 2 * nev + 1))
        nconv::Int = min(info_r.converged, info_l.converged)
        nconv >= nlevels || error("EBIM Krylov solve converged only $nconv eigenpairs; requested $nlevels")
        p = sortperm(abs.(μ[1:nconv]); rev = true)
        nkeep::Int = min(nev, nconv)
        resize!(ks, nkeep)
        resize!(ts, nkeep)
        buf = Vector{Complex{T}}(undef, n)
        @inbounds for q = 1:nkeep
            j::Int = p[q]
            λ = inv(μ[j])
            v = VR[j]
            u = Ft \ UL[j]
            ε1 = -λ
            mul!(buf, ddA, v)
            num = dot(u, buf)
            mul!(buf, dA, v)
            den = dot(u, buf)
            ε2 = abs(den) > eps(T) ? -T(0.5) * ε1^2 * (num / den) : zero(ε1)
            corr = ε1 + ε2
            ks[q] = k + corr
            ts[q] = abs(corr)
        end
    end
    return ks, ts
end

"""
    solve(solver::ExpandedBIMSolver, pts::BoundaryPoints, k, nlevels::Int; multithreaded::Bool=true)

Compute local EBIM candidates using direct matrix construction.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k`: Expansion center.
* `nlevels::Int`: Minimum number of local candidates.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded matrix construction.

## Returns
* `ks::Vector`: Local EBIM eigenvalue estimates.
* `ts::Vector`: Magnitudes of the corresponding corrections.
"""
function solve(solver::ExpandedBIMSolver, pts::BoundaryPoints, k, nlevels::Int; multithreaded::Bool = true)
    A, dA, ddA = construct_matrices(solver, pts, k; multithreaded)
    return _solve(solver, A, dA, ddA, k, nlevels)
end

"""
    solve(solver::ExpandedBIMSolver, cache::EBIMTaylorCache, k, nlevels::Int; multithreaded::Bool=true)

Compute local EBIM candidates from a cached Taylor panel.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `cache::EBIMTaylorCache`: Taylor panel.
* `k`: EBIM expansion center.
* `nlevels::Int`: Minimum number of local candidates.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded Horner evaluation. Left for compatibility with the direct construction interface.

## Returns
* `ks::Vector`: Local EBIM eigenvalue estimates.
* `ts::Vector`: Magnitudes of the corresponding corrections.
"""
function solve(solver::ExpandedBIMSolver, cache::EBIMTaylorCache, k, nlevels::Int; multithreaded::Bool = true)
    A, dA, ddA = construct_matrices(solver, cache, k)
    return _solve(solver, A, dA, ddA, k, nlevels)
end

"""
    solve_wavenumber(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true, nlevels::Int=5) where {Bi<:AbsBilliard}

Compute the EBIM eigenvalue estimate closest to `k`.

The local EBIM problem is solved around `k`, after which the candidate
minimizing `|Re(kᵢ)-k|` is returned. `dk` is used when estimating the required
number of local levels if `nlevels` is not increased explicitly.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `billiard::Bi`: Billiard geometry.
* `k`: EBIM expansion center.
* `dk`: Local spectral half-width.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded matrix construction.
* `nlevels::Int = 5`: Minimum number of local candidates.

## Returns
* `kbest`: EBIM estimate closest to `k`.
* `tbest`: Magnitude of its EBIM correction.
"""
function solve_wavenumber(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool = true, nlevels::Int = 5) where {Bi<:AbsBilliard}
    fundamental = solver.kernel.symmetry !== nothing
    nreq = max(nlevels, ceil(Int, weyl_window_count(billiard, k - dk, 2 * dk; fundamental)))
    pts = evaluate_points(solver, billiard, k)
    ks, ts = solve(solver, pts, k, nreq; multithreaded)
    isempty(ks) && return nothing, nothing
    q::Int = argmin(abs.(real.(ks) .- k))
    return ks[q], ts[q]
end

"""
    solve_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true, nlevels::Int=5) where {Bi<:AbsBilliard}

Compute all local EBIM eigenvalue estimates in `[k-dk,k+dk]`.

## Arguments
* `solver::ExpandedBIMSolver`: EBIM solver.
* `billiard::Bi`: Billiard geometry.
* `k`: EBIM expansion center.
* `dk`: Spectral half-width.

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded matrix construction.
* `nlevels::Int = 5`: Minimum number of local candidates.

## Returns
* `ks::Vector`: EBIM estimates satisfying `|Re(kᵢ)-k| <= dk`.
* `ts::Vector`: Magnitudes of their EBIM corrections.
"""
function solve_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool = true, nlevels::Int = 5) where {Bi<:AbsBilliard}
    fundamental = solver.kernel.symmetry !== nothing
    nreq = max(nlevels, ceil(Int, weyl_window_count(billiard, k - dk, 2 * dk; fundamental)))
    pts = evaluate_points(solver, billiard, k)
    ks, ts = solve(solver, pts, k, nreq; multithreaded)
    keep = findall(q -> abs(real(ks[q]) - k) <= dk, eachindex(ks))
    return ks[keep], ts[keep]
end