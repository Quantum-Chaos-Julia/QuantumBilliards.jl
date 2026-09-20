"""
    solve_wavenumber(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, k, dk; multithreaded::Bool = true)

Finds the wavenumber `k0` within `[k - dk/2, k + dk/2]` via `Optim.optimize` that minimizes the tension
computed by the sweep `solver`, together with the minimal tension `t0`.

## Arguments
* `solver::SweepBasisSolver`: The sweep solver used to solve for the tension at each wavenumber.
* `basis::AbsBasis`: The basis used to approximate the eigenstate.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `k::Real`: The center of the wavenumber search window.
* `dk::Real`: Width of the wavenumber search window, `[k - dk/2, k + dk/2]`.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `k0::Real`: The wavenumber minimizing the tension within the search window.
* `t0::Real`: The minimal tension found at `k0`.
"""
function solve_wavenumber(solver::SweepBasisSolver,basis::AbsBasis, billiard::AbsBilliard, k, dk; multithreaded=true)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    new_basis = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    function f(k)
        return solve(solver,new_basis,pts,k;multithreaded)
    end
    res =  optimize(f, k-0.5*dk, k+0.5*dk)
    k0,t0 = res.minimizer, res.minimum
    return k0, t0
end

"""
    k_sweep(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, ks; multithreaded::Bool = true) 

Computes the tension of the sweep `solver` at every wavenumber in `ks`, using a
single basis resized to the largest wavenumber in `ks`.

## Arguments
* `solver::SweepBasisSolver`: The sweep solver used to solve for the tension at each wavenumber.
* `basis::AbsBasis`: The basis used to approximate the eigenstate.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `ks::AbstractVector`: Wavenumbers at which the tension is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `res::AbstractVector`: Tensions corresponding to the wavenumbers in `ks`.
"""
function k_sweep(solver::SweepBasisSolver, basis::AbsBasis, billiard::AbsBilliard, ks; multithreaded=true)
    k = maximum(ks)
    L = CompositeCurve(get_boundary_curves(billiard)).length
    dim = max(solver.min_dim,round(Int, L*k*solver.dim_scaling_factor/(2*pi)))
    new_basis = resize_basis(basis,billiard,dim,k)
    pts = evaluate_points(solver, billiard, k)
    res = similar(ks)
    for (i,k) in enumerate(ks)
        res[i] = solve(solver,new_basis,pts,k; multithreaded)
    end
    return res
end

"""
    solve_wavenumber(solver::SweepBIMSolver, billiard::AbsBilliard, k, dk; multithreaded::Bool = true, pts::Union{Nothing,BoundaryPoints} = nothing) → (k0::Real, t0::Real)

Finds the wavenumber `k0` within `[k - dk/2, k + dk/2]` via `Optim.optimize` that minimizes the tension
computed by the boundary-integral sweep `solver`, together with the minimal tension `t0`.

## Arguments
* `solver::SweepBIMSolver`: The boundary-integral sweep solver used to solve for the tension at each wavenumber.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `k::Real`: The center of the wavenumber search window.
* `dk::Real`: Width of the wavenumber search window, `[k - dk/2, k + dk/2]`.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.
* `pts::Union{Nothing,BoundaryPoints} = nothing`: Optional precomputed boundary discretization; when `nothing`, one is generated at `k`.

## Returns
* `k0::Real`: The wavenumber minimizing the tension within the search window.
* `t0::Real`: The minimal tension found at `k0`.
"""
function solve_wavenumber(solver::SweepBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true, pts::Union{Nothing,BoundaryPoints}=nothing) where {Bi<:AbsBilliard}
    pts_ = pts===nothing ? evaluate_points(solver, billiard, k) : pts
    function f(k)
        return solve(solver, pts_, k; multithreaded)
    end
    res = optimize(f, k-0.5*dk, k+0.5*dk)
    k0, t0 = res.minimizer, res.minimum
    return k0, t0
end

"""
    k_sweep(solver::SweepBIMSolver, billiard::AbsBilliard, ks; multithreaded::Bool = true)

Computes the tension of the boundary-integral sweep `solver` at every
wavenumber in `ks`, using a single boundary discretization sized for the
largest wavenumber in `ks`.

## Arguments
* `solver::SweepBIMSolver`: The boundary-integral sweep solver used to solve for the tension at each wavenumber.
* `billiard::AbsBilliard`: The billiard whose boundary is discretized.
* `ks::AbstractVector`: Wavenumbers at which the tension is evaluated.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether the matrix construction is multithreaded.

## Returns
* `res::AbstractVector`: Tensions corresponding to the wavenumbers in `ks`.
"""
function k_sweep(solver::SweepBIMSolver, billiard::Bi, ks; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    k = maximum(ks)
    pts = evaluate_points(solver, billiard, k)
    res = similar(ks)
    for (i,k) in enumerate(ks)
        res[i] = solve(solver, pts, k; multithreaded)
    end
    return res
end

"""

    _fold_boundary(::Type{T}, xy, symmetry::AbsSymmetry, character::Tuple) where {T<:Real} → orbits::SymmetryOrbitMap{T}

Construct the symmetry-orbit map used to fold or expand a BIM boundary discretization.
The stored symmetry character is converted to the representation expected by
[`symmetry_index_orbits`](@ref). Ordinary symmetry types receive the character
entries as separate scalar arguments, while [`CompositeReflection`](@ref)
receives them as a single `Vector{Complex{T}}`.

## Arguments
* `T::Type{<:Real}`: Real scalar type used by the boundary discretization.
* `xy`: Boundary coordinates on the complete physical boundary.
* `symmetry::AbsSymmetry`: Active boundary symmetry.
* `character::Tuple`: Characters of the selected symmetry irreps.

## Returns
* `orbits::SymmetryOrbitMap{T}`: Symmetry-orbit mapping between the complete and fundamental-domain boundary discretizations.
"""
_fold_boundary(::Type{T}, xy, symmetry::AbsSymmetry, character::Tuple) where {T<:Real} = symmetry_index_orbits(T, xy, symmetry, Complex{T}.(character...))
_fold_boundary(::Type{T}, xy, symmetry::CompositeReflection, character::Tuple) where {T<:Real} = symmetry_index_orbits(T, xy, symmetry, Complex{T}[character...])

################################################################################
# BIM STATE CONSTRUCTION
#
# The native BIM layer density and the physical boundary normal derivative
# u = ∂ₙψ are distinct boundary quantities.
#
# For DLP, weighted-transpose reciprocity allows u to be recovered from the
# left singular vector of the Fredholm matrix:
#
#     A_adj = W⁻¹ Aᵀ W,     W = diag(ds),
#
#     u_raw = W⁻¹ conj(lvec).
#
# For CFIE, the left singular vector is not the physical normal derivative.
# Instead, u is reconstructed from the right singular vector μ through
#
#     u_raw = -Nμ - ik(μ + K'μ),
#
# where N is evaluated with Maue's identity and K' with the weighted transpose
# of the double-layer discretization.
#
# Consequently:
#
#     DLP:   vec = full-boundary DLP layer density
#            u   = full-boundary Rellich-normalized ∂ₙψ
#
#     CFIE:  vec = full-boundary CFIE layer density
#            u   = full-boundary Rellich-normalized ∂ₙψ
################################################################################

"""
    symmetrize_layer_density(solver::AbsBIMSolver, layer_density::AbstractVector{N}, pts::BoundaryPoints{T}, billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}

Expand a symmetry-reduced boundary density onto the complete physical boundary.

If no symmetry is active, `layer_density` must already have the same length as
`pts`. When symmetry reduction is active, a fundamental-domain density is
expanded using the [`BilliardGeometry.SymmetryOrbitMap`](@ref) defined by
`solver.symmetry` and `solver.character`. An already full-length density is
returned unchanged.

`pts` must describe the complete physical boundary. The returned density
therefore corresponds point-for-point to `pts`.

## Arguments
- `solver::AbsBIMSolver`: BIM solver defining the active symmetry and character.
- `layer_density::AbstractVector{N}`: Boundary density on either the fundamental or complete physical boundary.
- `pts::BoundaryPoints{T}`: Complete physical-boundary discretization.
- `billiard::Bi`: Associated billiard, retained for a uniform BIM state-construction API.

## Returns
- `full_density::Vector`: Boundary density on the complete physical boundary.
"""
function symmetrize_layer_density(solver::AbsBIMSolver, layer_density::AbstractVector{N}, pts::BoundaryPoints{T}, billiard::Bi) where {N<:Number,T<:Real,Bi<:AbsBilliard}
    Nfull = length(pts)
    length(layer_density) == Nfull && return layer_density
    solver.symmetry === nothing && throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected full length $Nfull because no symmetry is active"))
    orbits = _fold_boundary(T, pts.xy, solver.symmetry, solver.character)
    Nred = fundamental_size(orbits)
    length(layer_density) == Nred || throw(DimensionMismatch("Boundary data has length $(length(layer_density)); expected reduced $Nred or full $Nfull"))
    S = promote_type(N, Complex{T}); full_data = Vector{S}(undef, Nfull)
    @inbounds for q in 1:Nfull
        full_data[q] = orbits.phase[q] * layer_density[orbits.orbit_of[q]]
    end
    return full_data
end

"""
    _bim_normal_derivative(solver::DLP, pts::BoundaryPoints{T}, lvec::AbstractVector) where {T<:Real}

Recover the unnormalized physical boundary normal derivative `u = ∂ₙψ` from
the left singular vector of a DLP Fredholm matrix.

For the Nyström-discretized DLP formulation, the adjoint boundary problem is
related to the primal Fredholm matrix by the weighted transpose

    A_adj = W⁻¹ Aᵀ W,     W = diag(ds).

If `lvec` is the left singular vector associated with the smallest singular
value of `A`, the corresponding physical boundary normal derivative is

    u_raw = W⁻¹ conj(lvec).

For a symmetry-reduced solve, only the fundamental-domain entries of the
full-boundary quadrature weights are used. The returned vector therefore has
the same length as `lvec`; symmetry expansion and Rellich normalization are
performed by [`solve_state`](@ref).

This relation is specific to the DLP formulation. A CFIE left singular vector
must not in general be interpreted as the physical normal derivative.

## Arguments
- `solver::DLP`: DLP solver defining the active symmetry and character.
- `pts::BoundaryPoints{T}`: Complete physical-boundary discretization.
- `lvec::AbstractVector`: Left singular vector of the DLP Fredholm matrix.

## Returns
- `u_raw::Vector`: Unnormalized physical boundary normal derivative, reduced when symmetry is active and full-length otherwise.
"""
function _bim_normal_derivative(solver::DLP, pts::BoundaryPoints{T}, lvec::AbstractVector) where {T<:Real}
    idx = solver.symmetry === nothing ? (1:length(lvec)) : _fold_boundary(T, pts.xy, solver.symmetry, solver.character).fundamental_indices
    return conj.(lvec) ./ pts.ds[idx]
end

"""
    solve_state(solver::DLP, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool = true) where {T<:Real,Bi<:AbsBilliard}

Solve the DLP eigenproblem at `k` and recover both the native DLP layer density
and the physical boundary normal derivative `u = ∂ₙψ`.

The DLP Fredholm matrix `A(k)` is assembled once and its smallest singular
triplet is computed with a single `KrylovKit.svdsolve` call. The right singular
vector gives the native DLP layer density. The left singular vector is converted
to the physical normal derivative using [`_bim_normal_derivative`](@ref).

Both quantities are expanded onto the complete physical boundary when symmetry
reduction is active. The physical normal derivative is then normalized using
the Rellich identity. Thus `vec`, `u`, and `pts` all refer to the same complete
physical-boundary discretization.

## Arguments
- `solver::DLP`: DLP solver used to construct the Fredholm matrix.
- `pts::BoundaryPoints{T}`: Complete physical-boundary discretization.
- `k`: Eigenvalue wave number.
- `billiard::Bi`: Associated billiard.

## Keyword Arguments
- `multithreaded::Bool = true`: Use multithreaded Fredholm-matrix construction.

## Returns
- `ten::Real`: Smallest singular value of the DLP Fredholm matrix.
- `vec::Vector`: DLP layer density on the complete physical boundary.
- `u::Vector`: Rellich-normalized physical boundary normal derivative `∂ₙψ` on the complete physical boundary.
- `bnd_norm::Real`: Rellich norm of the unnormalized physical boundary derivative.
"""
function solve_state(solver::DLP, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    kT = T(k)
    A = construct_matrices(solver, pts, kT; multithreaded)
    @blas_1 vals, lvecs, rvecs, _ = KrylovKit.svdsolve(A, 1, :SR)
    ten = vals[1]
    vec = symmetrize_layer_density(solver, Vector{Complex{T}}(rvecs[1]), pts, billiard)
    u = symmetrize_layer_density(solver, _bim_normal_derivative(solver, pts, lvecs[1]), pts, billiard)
    bnd_norm = _rellich(pts, u, kT)
    u ./= sqrt(bnd_norm)
    return ten, vec, u, bnd_norm
end

"""
    _bim_dlp_cross_kernel_entry(pts::BoundaryPoints{T}, x::T, y::T, k::Union{T,Complex{T}}, j::Int) where {T<:Real} → value::Complex{T}

Evaluates one smooth double-layer Nyström entry from source node `j` on one
boundary component to a target point `(x,y)` on another component.

## Arguments
* `pts::BoundaryPoints{T}`: Source-component boundary discretization.
* `x::T`: Target x-coordinate.
* `y::T`: Target y-coordinate.
* `k::Union{T,Complex{T}}`: Helmholtz wavenumber.
* `j::Int`: Source-node index.

## Returns
* `value::Complex{T}`: Smooth cross-component double-layer entry.
"""
@inline function _bim_dlp_cross_kernel_entry(pts::BoundaryPoints{T}, x::T, y::T, k::Union{T,Complex{T}}, j::Int) where {T<:Real}
    xj, yj = pts.xy[j]; dx = x-xj; dy = y-yj; r = hypot(dx,dy)
    tx, ty = pts.tangent[j]
    return pts.ws[j]*im*k/2*(ty*dx-tx*dy)*_bim_hankelh1(1,k*r)/r
end

############### CFIE ###############

# Fourier spectral derivative with respect to the uniform 2π-periodic
# computational parameter. The Nyquist mode is zeroed for even grids.
function _cfie_derivative!(du::AbstractVector{Complex{T}}, u::AbstractVector{Complex{T}}) where {T<:Real}
    N = length(u); û = fft(u)
    @inbounds for j in 1:N
        q = j-1; q > N÷2 && (q -= N)
        iseven(N) && q == N÷2 && (q = 0)
        û[j] *= im*T(q)
    end
    du .= ifft!(û)
    return du
end

# Apply the Helmholtz single-layer operator simultaneously to three densities.
# Maue's identity requires S(∂sμ), S(nxμ) and S(nyμ), so fusing them avoids
# evaluating the same Hankel/Kress kernel three times.
function _cfie_slp3!(v1::Vector{Complex{T}}, v2::Vector{Complex{T}}, v3::Vector{Complex{T}}, comp::Vector{BoundaryPoints{T}}, q1::Vector{Complex{T}}, q2::Vector{Complex{T}}, q3::Vector{Complex{T}}, offs::Vector{Int}, Gs::Vector{BoundaryGeomCache{T}}, Rs::Vector{Matrix{T}}, k::T) where {T<:Real}
    euler_over_pi = T(Base.MathConstants.eulergamma)/T(pi)
    invtwopi = inv(2*T(pi))
    fill!(v1,zero(Complex{T})); fill!(v2,zero(Complex{T})); fill!(v3,zero(Complex{T}))
    nc = length(comp)
    @inbounds for a in 1:nc
        pa = comp[a]; offa = offs[a]; G = Gs[a]; R = Rs[a]
        for i in 1:length(pa)
            gi = offa+i-1
            s1 = zero(Complex{T}); s2 = zero(Complex{T}); s3 = zero(Complex{T})
            for j in 1:length(pa)
                gj = offa+j-1
                if i == j
                    speed = G.speed[j]
                    m1 = -invtwopi*speed
                    m2 = ((Complex{T}(0,one(T)/2)-euler_over_pi)-invtwopi*log((k^2/4)*speed^2))*speed
                    K = R[i,j]*m1+pa.ws[j]*m2
                else
                    r = G.R[i,j]; speed = G.speed[j]; h0 = _bim_hankelh1(0,k*r)
                    j0 = _bim_besselj(0,k*r,h0)
                    m1 = -invtwopi*j0*speed
                    K = R[i,j]*m1+pa.ws[j]*(Complex{T}(0,one(T)/2)*h0*speed-m1*G.logterm[i,j])
                end
                s1 += K*q1[gj]; s2 += K*q2[gj]; s3 += K*q3[gj]
            end
            xi, yi = pa.xy[i]
            for b in 1:nc
                b == a && continue
                pb = comp[b]; offb = offs[b]
                for j in 1:length(pb)
                    gj = offb+j-1
                    xj, yj = pb.xy[j]; r = hypot(xi-xj,yi-yj)
                    tx, ty = pb.tangent[j]
                    K = pb.ws[j]*Complex{T}(0,one(T)/2)*_bim_hankelh1(0,k*r)*hypot(tx,ty)
                    s1 += K*q1[gj]; s2 += K*q2[gj]; s3 += K*q3[gj]
                end
            end
            v1[gi] = s1; v2[gi] = s2; v3[gi] = s3
        end
    end
    return v1,v2,v3
end

"""
    _cfie_adjoint_dlp!(out::Vector{Complex{T}}, comp::Vector{BoundaryPoints{T}}, μ::AbstractVector{Complex{T}}, offs::Vector{Int}, Gs::Vector{BoundaryGeomCache{T}}, Rs::Vector{Matrix{T}}, k::T) where {T<:Real} → out::Vector{Complex{T}}

Applies the adjoint double-layer operator `K'` matrix-free using

    K'ₕ = W⁻¹DₕᵀW,

where `Dₕ` is exactly the Kress/Nyström double-layer discretization used by
the DLP solver and `W=diag(ds)` contains the physical boundary quadrature
weights. The transpose is not complex-conjugated.

The same-component calculation processes `(i,j)` and `(j,i)` together so
that their common Hankel and Bessel evaluations are shared. Cross-component
blocks are smooth and are processed in component pairs.

## Arguments
* `out::Vector{Complex{T}}`: Destination vector.
* `comp::Vector{BoundaryPoints{T}}`: Connected boundary components.
* `μ::AbstractVector{Complex{T}}`: Global boundary density.
* `offs::Vector{Int}`: Starting offsets of the connected components.
* `Gs::Vector{BoundaryGeomCache{T}}`: Geometry cache for each component.
* `Rs::Vector{Matrix{T}}`: Kress matrix for each component.
* `k::T`: Real Helmholtz wavenumber.

## Returns
* `out::Vector{Complex{T}}`: Discrete adjoint double-layer action `K'μ`.
"""
function _cfie_adjoint_dlp!(out::Vector{Complex{T}}, comp::Vector{BoundaryPoints{T}}, μ::AbstractVector{Complex{T}}, offs::Vector{Int}, Gs::Vector{BoundaryGeomCache{T}}, Rs::Vector{Matrix{T}}, k::T) where {T<:Real}
    fill!(out,zero(Complex{T}))
    nc = length(comp); α1 = -k/(2*T(pi)); α2 = im*k/2
    @inbounds for a in 1:nc
        p = comp[a]; G = Gs[a]; R = Rs[a]; off = offs[a]; N = length(p)
        for i in 1:N
            gi = off+i-1
            out[gi] += Complex{T}(p.ws[i]*G.kappa[i],zero(T))*p.ds[i]*μ[gi]
        end
        for j in 2:N
            gj = off+j-1
            for i in 1:j-1
                gi = off+i-1
                r = G.R[i,j]; invr = G.invR[i,j]; lt = G.logterm[i,j]
                h1 = _bim_hankelh1(1,k*r); j1 = _bim_besselj(1,k*r,h1)
                l1 = α1*G.inner[i,j]*j1*invr
                Dij = R[i,j]*l1+p.ws[j]*(α2*G.inner[i,j]*h1*invr-l1*lt)
                l1 = α1*G.inner[j,i]*j1*invr
                Dji = R[j,i]*l1+p.ws[i]*(α2*G.inner[j,i]*h1*invr-l1*lt)
                out[gj] += Dij*p.ds[i]*μ[gi]
                out[gi] += Dji*p.ds[j]*μ[gj]
            end
        end
    end
    @inbounds for b in 2:nc, a in 1:b-1
        pa = comp[a]; pb = comp[b]; offa = offs[a]; offb = offs[b]
        for i in 1:length(pa)
            gi = offa+i-1; xi, yi = pa.xy[i]; wiμi = pa.ds[i]*μ[gi]
            for j in 1:length(pb)
                gj = offb+j-1; xj, yj = pb.xy[j]; wjμj = pb.ds[j]*μ[gj]
                out[gj] += _bim_dlp_cross_kernel_entry(pb,xi,yi,k,j)*wiμi
                out[gi] += _bim_dlp_cross_kernel_entry(pa,xj,yj,k,i)*wjμj
            end
        end
    end
    @inbounds for a in 1:nc
        p = comp[a]; off = offs[a]
        @simd for j in 1:length(p)
            out[off+j-1] /= p.ds[j]
        end
    end
    return out
end

"""
    _cfie_normal_derivative(comp::Vector{BoundaryPoints{T}}, offs::Vector{Int}, μ::AbstractVector{Complex{T}}, k::T) where {T<:Real} → u::Vector{Complex{T}}

Recovers the physical boundary normal derivative `u=∂ₙψ` from a CFIE layer
density on one or more smooth, ungraded periodic boundary components.

For the convention `A(k)=I-(D(k)+ikS(k))`, the recovery formula is

    u = -Nμ - ik(μ + K'μ),

with `Nμ` evaluated through Maue's identity

    Nμ = ∂ₛS(∂ₛμ) + k² n⋅S(nμ).

All integral operators couple the complete boundary, including interactions
between different connected components and holes. Tangential Fourier
differentiation is performed separately on each component.

## Arguments
* `comp::Vector{BoundaryPoints{T}}`: Connected physical boundary components.
* `offs::Vector{Int}`: Starting offsets of the components in the global density.
* `μ::AbstractVector{Complex{T}}`: Global CFIE layer density.
* `k::T`: Real eigenwavenumber.

## Returns
* `u::Vector{Complex{T}}`: Unnormalized physical boundary normal derivative.
"""
function _cfie_normal_derivative(comp::Vector{BoundaryPoints{T}}, offs::Vector{Int}, μ::AbstractVector{Complex{T}}, k::T) where {T<:Real}
    nc = length(comp); N = length(μ)
    Gs = Vector{BoundaryGeomCache{T}}(undef,nc); Rs = Vector{Matrix{T}}(undef,nc)
    @inbounds for a in 1:nc
        Gs[a] = boundary_geom_cache(comp[a],false)
        Rs[a] = zeros(T,length(comp[a]),length(comp[a])); kress_R!(Rs[a])
    end
    dsμ = Vector{Complex{T}}(undef,N); nxμ = similar(dsμ); nyμ = similar(dsμ)
    @inbounds for a in 1:nc
        p = comp[a]; r = offs[a]:offs[a+1]-1
        _cfie_derivative!(@view(dsμ[r]),@view(μ[r]))
        for j in 1:length(p)
            g = offs[a]+j-1; tx, ty = p.tangent[j]; nx, ny = p.normal[j]
            dsμ[g] /= hypot(tx,ty)
            nxμ[g] = nx*μ[g]; nyμ[g] = ny*μ[g]
        end
    end
    Sdμ = similar(dsμ); Snxμ = similar(dsμ); Snyμ = similar(dsμ)
    _cfie_slp3!(Sdμ,Snxμ,Snyμ,comp,dsμ,nxμ,nyμ,offs,Gs,Rs,k)
    dSdμ = similar(dsμ)
    @inbounds for a in 1:nc
        p = comp[a]; r = offs[a]:offs[a+1]-1
        _cfie_derivative!(@view(dSdμ[r]),@view(Sdμ[r]))
        for j in 1:length(p)
            g = offs[a]+j-1; tx, ty = p.tangent[j]
            dSdμ[g] /= hypot(tx,ty)
        end
    end
    Kpμ = similar(dsμ)
    _cfie_adjoint_dlp!(Kpμ,comp,μ,offs,Gs,Rs,k)
    k2 = k*k
    @inbounds for a in 1:nc
        p = comp[a]; off = offs[a]
        @simd for j in 1:length(p)
            g = off+j-1; nx, ny = p.normal[j]
            Nμ = dSdμ[g]+k2*(nx*Snxμ[g]+ny*Snyμ[g])
            dSdμ[g] = -Nμ-im*k*(μ[g]+Kpμ[g])
        end
    end
    return dSdμ
end

"""
    _cfie_normal_derivative(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, μ::AbstractVector{Complex{T}}, k::T) where {T<:Real} → u::Vector{Complex{T}}
    _cfie_normal_derivative(solver::CompositeBIMSolver, pts::BoundaryPoints{T}, μ::AbstractVector{Complex{T}}, k::T) where {T<:Real} → u::Vector{Complex{T}}

Recovers the physical normal derivative from a CFIE layer density. A standard
CFIE contains one connected component; a `CompositeBIMSolver` supplies the
component partition already encoded in its flattened `BoundaryPoints`.
No modification of `CompositeBIMSolver` itself is required.

## Arguments
* `solver::CombinedFieldIntegralEquationSolver`: Single-component CFIE solver.
* `solver::CompositeBIMSolver`: Composite solver containing multiple connected boundary components.
* `pts::BoundaryPoints{T}`: Physical boundary discretization.
* `μ::AbstractVector{Complex{T}}`: CFIE layer density.
* `k::T`: Real eigenwavenumber.

## Returns
* `u::Vector{Complex{T}}`: Unnormalized physical boundary normal derivative.
"""
function _cfie_normal_derivative(::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, μ::AbstractVector{Complex{T}}, k::T) where {T<:Real}
    return _cfie_normal_derivative(BoundaryPoints{T}[pts],Int[1,length(pts)+1],μ,k)
end

function _cfie_normal_derivative(solver::CompositeBIMSolver, pts::BoundaryPoints{T}, μ::AbstractVector{Complex{T}}, k::T) where {T<:Real}
    nc = length(solver.component_solvers)
    offs = _composite_offsets(pts,nc)
    comp = [_composite_component_slice(pts,offs[a]:offs[a+1]-1,a) for a in 1:nc]
    return _cfie_normal_derivative(comp,offs,μ,k)
end

"""
    solve_state(solver::CFIE, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool = true) where {T<:Real,Bi<:AbsBilliard} → (ten::Real, vec::Vector{Complex{T}}, u::Vector{Complex{T}}, bnd_norm::T)

Solve the CFIE eigenproblem at `k` and recover both the native CFIE layer
density and the physical boundary normal derivative `u = ∂ₙψ`.

The CFIE Fredholm matrix is assembled once and its smallest right singular
vector gives the native CFIE layer density `μ`. When symmetry reduction is
active, this density is first expanded onto the complete physical boundary.

For the CFIE convention

    A(k) = I - (D(k) + ikS(k)),

the physical boundary normal derivative is recovered from the layer density as

    u_raw = -Nμ - ik(μ + K'μ),

where `N` is evaluated through Maue's identity and `K'` through the
weighted-transpose discretization of the same double-layer operator used by
the DLP solver. Unlike the DLP formulation, the CFIE left singular vector is
not used for this recovery.

Finally, `u_raw` is normalized with the Rellich identity. Thus `vec`, `u`,
and `pts` all correspond point-for-point to the complete physical boundary.

## Arguments
* `solver::CFIE`: CFIE solver used to construct the Fredholm matrix.
* `pts::BoundaryPoints{T}`: Complete physical-boundary discretization.
* `k`: Eigenvalue wavenumber.
* `billiard::Bi`: Associated billiard.

## Keyword Arguments
* `multithreaded::Bool = true`: Whether Fredholm-matrix construction is multithreaded.

## Returns
* `ten::Real`: Smallest singular value of the CFIE Fredholm matrix.
* `vec::Vector{Complex{T}}`: CFIE layer density on the complete physical boundary.
* `u::Vector{Complex{T}}`: Rellich-normalized physical boundary normal derivative `∂ₙψ`.
* `bnd_norm::T`: Rellich norm of the unnormalized physical boundary normal derivative.
"""
function solve_state(solver::CFIE, pts::BoundaryPoints{T}, k, billiard::Bi; multithreaded::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    kT = T(k)
    A = construct_matrices(solver,pts,kT; multithreaded)
    @blas_1 vals, _, rvecs, _ = KrylovKit.svdsolve(A,1,:SR)
    ten = vals[1]
    vec = symmetrize_layer_density(solver,Vector{Complex{T}}(rvecs[1]),pts,billiard)
    u = _cfie_normal_derivative(solver,pts,vec,kT)
    bnd_norm = _rellich(pts,u,kT)
    u ./= sqrt(bnd_norm)
    return ten,vec,u,bnd_norm
end