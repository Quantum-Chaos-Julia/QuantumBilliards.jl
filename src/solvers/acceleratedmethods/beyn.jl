"""
    BeynSolver{T,K} <: AcceleratedBIMSolver

`BeynSolver` is a concrete [`AcceleratedBIMSolver`](@ref) implementing Beyn's
contour-integral method for recovering every root of the nonlinear
eigenproblem `A(k)v = 0` within a wavenumber window from a single contour
solve.

## Description
`BeynSolver` wraps an inner `kernel::SweepBIMSolver` (a
[`DoubleLayerPotentialSolver`](@ref), [`CombinedFieldIntegralEquationSolver`](@ref)
or [`CompositeBIMSolver`](@ref)) supplying the Fredholm operator `A(k)`. On a
circular contour of center `k0` and radius `R`, the contour moments

    A0 = (1/2πi) ∮ A(z)⁻¹V dz,
    A1 = (1/2πi) ∮ z A(z)⁻¹V dz,

are formed against a random probing matrix `V` of rank `r` (see
[`construct_matrices`](@ref)); a rank-revealing SVD of `A0` projects the
problem onto a small dense generalized eigenproblem whose eigenpairs
approximate the roots of `A(k)v = 0` inside the contour (see [`solve`](@ref),
[`solve_vectors`](@ref)). Candidates are filtered by contour containment and
residual norm using `svd_tol`/`res_tol`.

Reference: W.-J. Beyn, "An integral method for solving nonlinear eigenvalue
problems", Linear Algebra Appl. 436 (2012).

## Attributes
* `kernel`: The wrapped [`SweepBIMSolver`](@ref) supplying `A(k)`.
* `m`: Target eigenvalue count per contour window, used for Weyl-window planning.
* `nq`: Number of contour quadrature nodes.
* `r`: Random probing rank.
* `svd_tol`: Singular-value cutoff used for rank detection in `A0`.
* `res_tol`: Residual threshold used to reject spurious roots.
* `auto_discard_spurious`: Whether candidates with residual above `res_tol` are automatically rejected.
* `use_chebyshev`: Whether Chebyshev-accelerated kernel evaluation is used.
* `cheb_config`: [`ChebyshevConfig`](@ref) bundling the Chebyshev panel/degree/auto-tuning parameters.

## API
The following functions can be evaluated for this type:
- [`evaluate_points`](@ref)
- [`construct_matrices`](@ref)
- [`solve`](@ref)
- [`solve_vectors`](@ref)
- [`solve_wavenumber`](@ref)
- [`solve_spectrum`](@ref)
- [`compute_spectrum`](@ref)

!!! note "Chebyshev acceleration"
    When `use_chebyshev=true`, `construct_matrices` tunes (or, with
    `cheb_config.param_strategy===:manual`, directly uses)
    `H₁^(1)`/`J₁` (or `H₀^(1)`/`H₁^(1)`/`J₀`/`J₁` for a
    [`CombinedFieldIntegralEquationSolver`](@ref) kernel) Chebyshev plans
    once across every contour node, then assembles all `nq` node matrices in
    a single pass reusing those plans (see `solvers/chebyshev/` in this
    package). Only [`DoubleLayerPotentialSolver`](@ref)/
    [`CombinedFieldIntegralEquationSolver`](@ref) kernels with `T===Float64`
    are currently supported; a [`CompositeBIMSolver`](@ref) kernel or a
    non-`Float64` numeric type raises an error (construct with
    `use_chebyshev=false` instead).
"""
struct BeynSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K
    m::Int
    nq::Int
    r::Int
    svd_tol::T
    res_tol::T
    auto_discard_spurious::Bool
    use_chebyshev::Bool
    cheb_config::ChebyshevConfig{T}
end

"""
    BeynSolver(kernel::K; m::Int = 10, nq::Int = 48, r::Int = 48, svd_tol::Real = 1e-12, res_tol::Real = 1e-9, auto_discard_spurious::Bool = true, use_chebyshev::Bool = true, n_panels_h::Int = 15000, M_h::Int = 5, n_panels_j::Int = 10000, M_j::Int = 5, cheb_config::Union{Nothing,ChebyshevConfig} = nothing) where {K<:SweepBIMSolver} → solver::BeynSolver

Constructs a [`BeynSolver`](@ref) wrapping the boundary-integral `kernel`.

## Arguments
* `kernel`: The [`SweepBIMSolver`](@ref) supplying the Fredholm operator `A(k)`.

## Keyword arguments
* `m::Int = 10`: Target eigenvalue count per contour window.
* `nq::Int = 48`: Number of contour quadrature nodes.
* `r::Int = 48`: Random probing rank.
* `svd_tol::Real = 1e-12`: Singular-value cutoff for rank detection.
* `res_tol::Real = 1e-9`: Residual threshold for spurious-root rejection.
* `auto_discard_spurious::Bool = true`: Whether to automatically reject high-residual candidates.
* `use_chebyshev::Bool = true`: Whether to use Chebyshev-accelerated kernel evaluation.
* `n_panels_h::Int = 15000`: Hankel-function Chebyshev panel count (ignored if `cheb_config` is given).
* `M_h::Int = 5`: Hankel-function Chebyshev polynomial degree (ignored if `cheb_config` is given).
* `n_panels_j::Int = 10000`: Bessel-J-function Chebyshev panel count (ignored if `cheb_config` is given).
* `M_j::Int = 5`: Bessel-J-function Chebyshev polynomial degree (ignored if `cheb_config` is given).
* `cheb_config::Union{Nothing,ChebyshevConfig} = nothing`: A pre-built [`ChebyshevConfig`](@ref); when `nothing`, one is constructed from `n_panels_h`/`M_h`/`n_panels_j`/`M_j` with every other `ChebyshevConfig` field left at its default.

## Returns
* `solver`: A [`BeynSolver`](@ref) instance.
"""
function BeynSolver(kernel::K; m::Int=10, nq::Int=48, r::Int=48,
                     svd_tol::Real=1e-12, res_tol::Real=1e-9,
                     auto_discard_spurious::Bool=true, use_chebyshev::Bool=true,
                     n_panels_h::Int=15000, M_h::Int=5, n_panels_j::Int=10000, M_j::Int=5,
                     cheb_config::Union{Nothing,ChebyshevConfig}=nothing) where {K<:SweepBIMSolver}
    T = _bim_numeric_type(kernel)
    cfg = cheb_config===nothing ? ChebyshevConfig(T; n_panels_h, M_h, n_panels_j, M_j) : cheb_config
    return BeynSolver{T,K}(kernel, m, nq, r, T(svd_tol), T(res_tol), auto_discard_spurious, use_chebyshev, cfg)
end

_bim_numeric_type(::BeynSolver{T}) where {T} = T

################################################################################
############################## WEYL-WINDOW HELPERS ###########################
################################################################################

"""
    weyl_window_width(billiard::Bi, k::T, m::Int; fundamental::Bool = true) where {T<:Real,Bi<:AbsBilliard} → Δk::T

Returns the wavenumber width `Δk` containing approximately `m` levels from the
leading Weyl estimate, i.e. the positive root of `A*((k+Δk)^2-k^2)/(4π) = m`
where `A` is [`fundamental_area`](@ref)`(billiard)` (or [`area`](@ref)`(billiard)`
if `fundamental = false`).
"""
@inline function weyl_window_width(billiard::Bi, k::T, m::Int; fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    A = fundamental ? fundamental_area(billiard) : area(billiard)
    return sqrt(k^2+T(4*pi*m/A))-k
end

"""
    plan_weyl_windows(billiard::Bi, k1::T, k2::T; m::Int = 10, Rmax::Real = 1.0, fundamental::Bool = true) where {T<:Real,Bi<:AbsBilliard} → iv::Vector{Tuple{T,T}}

Covers `[k1,k2]` with consecutive Weyl-balanced windows, each expected to
contain approximately `m` levels (see [`weyl_window_width`](@ref)), with every
window width capped at `2*Rmax` so that the corresponding Beyn disk has radius
`R ≤ Rmax`.
"""
function plan_weyl_windows(billiard::Bi, k1::T, k2::T; m::Int=10, Rmax::Real=1.0, fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    k2>k1 || return Tuple{T,T}[]
    m>0 || throw(ArgumentError("m must be positive; received m=$m"))
    Rmax>0 || throw(ArgumentError("Rmax must be positive; received Rmax=$Rmax"))
    iv = Tuple{T,T}[]
    maxwidth = T(2*Rmax)
    k = k1
    while k<k2
        Δk = min(weyl_window_width(billiard, k, m; fundamental=fundamental), maxwidth, k2-k)
        Δk>zero(T) || throw(ArgumentError("Weyl window width vanished at k=$k"))
        kR = k+Δk
        push!(iv, (k, kR))
        k = kR
    end
    return iv
end

"""
    beyn_disks_from_windows(iv::Vector{Tuple{T,T}}) where {T<:Real} → (k0::Vector{Complex{T}}, R::Vector{T})

Converts real windows `[kL,kR]` (as produced by [`plan_weyl_windows`](@ref)) to
circular Beyn contours with midpoint center `k0` and half-width radius `R`.
"""
function beyn_disks_from_windows(iv::Vector{Tuple{T,T}}) where {T<:Real}
    k0 = Vector{Complex{T}}(undef, length(iv))
    R = Vector{T}(undef, length(iv))
    @inbounds for (i, (kL, kR)) in pairs(iv)
        k0[i] = complex((kL+kR)/2)
        R[i] = (kR-kL)/2
    end
    return k0, R
end

################################################################################
############################# CONTOUR MOMENT BUFFERS ##########################
################################################################################

"""
    beyn_buffer_matrices(::Type{T}, N::Int, r::Int, rng::G) where {T<:Real,G} → (V, X, A0, A1)

Allocates the random probing matrix and working matrices used to form the two
Beyn contour moments

    A0 = (1/2πi) ∮ T(z)⁻¹V dz,
    A1 = (1/2πi) ∮ z T(z)⁻¹V dz.

The probing matrix `V` is kept complex even for real-valued problems because
the contour solves are generally complex.

## Returns
A tuple `(V,X,A0,A1)` where `V::Matrix{Complex{T}}` is the random `N×r`
probing matrix, `X::Matrix{Complex{T}}` is the solve workspace for
`T(z)X=V`, and `A0`/`A1::Matrix{Complex{T}}` store the two contour moments
(initialized to zero).
"""
function beyn_buffer_matrices(::Type{T}, N::Int, r::Int, rng::G) where {T<:Real,G}
    V = randn(rng, Complex{T}, N, r)
    X = similar(V)
    A0 = zeros(Complex{T}, N, r)
    A1 = zeros(Complex{T}, N, r)
    return V, X, A0, A1
end

################################################################################
###################### CHEBYSHEV-ACCELERATED MULTI-k ASSEMBLY ################
################################################################################

# Builds Tbufs[m] = A(zj[m]) for every contour node zj at once, reusing one
# set of Chebyshev Hankel/Bessel-J plans (tuned once across all nq nodes, see
# `tune_dlp_cheb_plans`/`tune_cfie_cheb_plans` in solvers/chebyshev/optimalpanelization.jl)
# and a single O(N²) pairwise-geometry pass (`_dlp_fredholm_full_multi_k_cheb!`/
# `_dlp_fredholm_reduced_multi_k_cheb!` in solvers/chebyshev/dlp.jl, and the
# CFIE analogues in cfie.jl) instead of calling the per-node value-only
# assembly `nq` times.
function _construct_matrices_multi_k_cheb(cs::DoubleLayerPotentialSolver, pts::BoundaryPoints{T}, zj::Vector{ComplexF64}, cfg::ChebyshevConfig; multithreaded::Bool=true) where {T<:Real}
    T===Float64 || error("Chebyshev-accelerated Beyn evaluation currently requires a Float64 kernel; received numeric type $T. Construct the BeynSolver with use_chebyshev=false.")
    N = length(pts)
    graded = _is_nontrivial_dlp_grading(pts)
    G = boundary_geom_cache(pts, graded)
    Rmat = zeros(T, N, N)
    kress_R!(Rmat)
    rmin, rmax = _cheb_geom_rminmax(G, zj)
    plans1, plansj1, _ = tune_dlp_cheb_plans(rmin, rmax, zj, cfg)
    if cs.symmetry===nothing
        Tbufs = [Matrix{ComplexF64}(undef, N, N) for _ in zj]
        _dlp_fredholm_full_multi_k_cheb!(Tbufs, pts, Rmat, G, zj, plans1, plansj1; multithreaded)
        return Tbufs
    else
        orbits = _fold_boundary(T, pts.xy, cs.symmetry, cs.character)
        msize = fundamental_size(orbits)
        Tbufs = [Matrix{ComplexF64}(undef, msize, msize) for _ in zj]
        _dlp_fredholm_reduced_multi_k_cheb!(Tbufs, pts, Rmat, G, orbits, zj, plans1, plansj1; multithreaded)
        return Tbufs
    end
end

function _construct_matrices_multi_k_cheb(cs::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, zj::Vector{ComplexF64}, cfg::ChebyshevConfig; multithreaded::Bool=true) where {T<:Real}
    T===Float64 || error("Chebyshev-accelerated Beyn evaluation currently requires a Float64 kernel; received numeric type $T. Construct the BeynSolver with use_chebyshev=false.")
    N = length(pts)
    graded = _is_nontrivial_dlp_grading(pts)
    G = boundary_geom_cache(pts, graded)
    Rmat = zeros(T, N, N)
    kress_R!(Rmat)
    rmin, rmax = _cheb_geom_rminmax(G, zj)
    plans0, plans1, plansj0, plansj1, _ = tune_cfie_cheb_plans(rmin, rmax, zj, cfg)
    if cs.symmetry===nothing
        Tbufs = [Matrix{ComplexF64}(undef, N, N) for _ in zj]
        _cfie_fredholm_full_multi_k_cheb!(Tbufs, pts, Rmat, G, zj, plans0, plans1, plansj0, plansj1; multithreaded)
        return Tbufs
    else
        orbits = _fold_boundary(T, pts.xy, cs.symmetry, cs.character)
        msize = fundamental_size(orbits)
        Tbufs = [Matrix{ComplexF64}(undef, msize, msize) for _ in zj]
        _cfie_fredholm_reduced_multi_k_cheb!(Tbufs, pts, Rmat, G, orbits, zj, plans0, plans1, plansj0, plansj1; multithreaded)
        return Tbufs
    end
end

_construct_matrices_multi_k_cheb(cs::CompositeBIMSolver, pts::BoundaryPoints, zj::Vector{ComplexF64}, cfg::ChebyshevConfig; multithreaded::Bool=true) = error("Chebyshev-accelerated Beyn evaluation is not yet implemented for CompositeBIMSolver kernels. Construct the BeynSolver with use_chebyshev=false.")

################################################################################
############################## CONTOUR ASSEMBLY ###############################
################################################################################

"""
    construct_matrices(solver::BeynSolver, pts::BoundaryPoints, k0, R; multithreaded::Bool = true, rng = MersenneTwister(0)) → (A0::Matrix, A1::Matrix)

Assembles the two Beyn contour moments `A0`, `A1` on a circular contour of
center `k0` and radius `R`.

## Description
For trapezoidal contour nodes `zj` and weights `wj` (`solver.nq` of them), the
method forms the wrapped kernel's Fredholm matrix `T(zj) = A(zj)` at every
node via [`construct_matrices`](@ref)`(solver.kernel, pts, zj)`, LU-factors
each `T(zj)`, and accumulates

    A0 ≈ Σⱼ wⱼ T(zⱼ)⁻¹V,
    A1 ≈ Σⱼ wⱼ zⱼ T(zⱼ)⁻¹V,

against a random probing matrix `V` of rank `solver.r` (see
[`beyn_buffer_matrices`](@ref)). If every singular value of the resulting `A0`
exceeds `solver.svd_tol` (the probing rank is saturated), the probing rank is
increased and the already-factorized `T(zⱼ)` are reused to recompute the
moments at the larger rank, repeating until the rank is resolved or the
maximum possible rank `N` is reached (an error is raised if saturation
persists at `r=N`).

The contour-node loop is left single-threaded (each node's own
`construct_matrices(solver.kernel, pts, zj)` call already parallelizes its own
`O(N²)` assembly internally via `multithreaded`, so parallelizing the node
loop as well would nest `Threads.@threads` regions); the per-node LU
factorization and triangular solves use multithreaded BLAS instead (see
[`@blas_multi_then_1`](@ref)).

## Keyword Arguments
* `multithreaded::Bool = true`: Enable multithreaded boundary-matrix construction at each contour node.
* `rng = MersenneTwister(0)`: Random-number generator used to construct the probing matrix.

## Returns
A tuple `(A0,A1)` of the two contour moments. `A0`/`A1` may have more than
`solver.r` columns if the probing rank had to be increased.
"""
function construct_matrices(solver::BeynSolver, pts::BoundaryPoints, k0, R; multithreaded::Bool=true, rng=MersenneTwister(0))
    T = _bim_numeric_type(solver)
    N = boundary_matrix_size(solver.kernel, pts)
    k0c = Complex{T}(k0)
    Rc = T(R)
    nq = solver.nq
    θ = range(zero(T), 2*T(pi); length=nq+1)[1:end-1]
    ej = cis.(θ)
    zj = k0c .+ Rc.*ej
    wj = (Rc/nq).*ej
    r = solver.r
    @debug "Beyn contour assembly started" N k0=k0c R=Rc nq r
    if solver.use_chebyshev
        Tbufs = _construct_matrices_multi_k_cheb(solver.kernel, pts, ComplexF64.(zj), solver.cheb_config; multithreaded)
    else
        Tbufs = Vector{Matrix{Complex{T}}}(undef, nq)
        @inbounds for j in 1:nq
            Tbufs[j] = construct_matrices(solver.kernel, pts, zj[j]; multithreaded)
        end
    end
    F1 = lu!(Tbufs[1]; check=false)
    Fs = Vector{typeof(F1)}(undef, nq)
    Fs[1] = F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq
        Fs[j] = lu!(Tbufs[j]; check=false)
    end
    V, X, A0, A1 = beyn_buffer_matrices(T, N, r, rng)
    xv = reshape(X, :); a0v = reshape(A0, :); a1v = reshape(A1, :)
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in eachindex(zj)
        ldiv!(X, Fs[j], V)
        BLAS.axpy!(wj[j], xv, a0v)
        BLAS.axpy!(wj[j]*zj[j], xv, a1v)
    end
    @blas_multi_then_1 MAX_BLAS_THREADS Σ = svdvals(A0)
    rk = count(>=(solver.svd_tol), Σ)
    if rk==r
        r_tmp = min(r+r, N)
        while true
            V, X, A0, A1 = beyn_buffer_matrices(T, N, r_tmp, rng)
            xv = reshape(X, :); a0v = reshape(A0, :); a1v = reshape(A1, :)
            @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in eachindex(zj)
                ldiv!(X, Fs[j], V)
                BLAS.axpy!(wj[j], xv, a0v)
                BLAS.axpy!(wj[j]*zj[j], xv, a1v)
            end
            @blas_multi_then_1 MAX_BLAS_THREADS Σ = svdvals(A0)
            rk = count(>=(solver.svd_tol), Σ)
            rk<r_tmp && break
            r_tmp==N && throw(ArgumentError("Beyn moment remains rank-saturated at the maximum probe rank N=$N"))
            r_tmp = min(r_tmp+r, N)
        end
    end
    @debug "Beyn contour assembly finished" size=size(A0) numerical_rank=rk
    return A0, A1
end

################################################################################
######################## PROJECTION, EIGENSOLVE, FILTERING ###################
################################################################################

# Shared core of `solve`/`solve_vectors`: assembles the contour moments,
# projects onto the reduced Beyn matrix `B = Uk'*A1*Wk*Σk⁻¹` (rank-revealing
# SVD of A0, cutoff `solver.svd_tol`), diagonalizes it, expands the retained
# eigenvectors back to the boundary-density basis `Φ = Uk*Y`, and filters
# candidates by contour containment (`|λ-k0|≤R`) and nonlinear residual
# (`‖A(λ)φ‖`, cutoff `solver.res_tol` when `solver.auto_discard_spurious`).
# Returns `(λ_kept::Vector{Complex{T}}, tens::Vector{T}, Φ_kept::Matrix{Complex{T}})`.
function _beyn_solve_core(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0))
    T = _bim_numeric_type(solver)
    k0c = Complex{T}(k0)
    Rc = T(dk)/2
    A0, A1 = construct_matrices(solver, pts, k0c, Rc; multithreaded, rng)
    N, rc = size(A0)
    rc==0 && return Complex{T}[], T[], Matrix{Complex{T}}(undef, N, 0)
    @blas_multi_then_1 MAX_BLAS_THREADS U, Σ, W = svd!(A0; full=false)
    rk = count(>=(solver.svd_tol), Σ)
    rk==0 && return Complex{T}[], T[], Matrix{Complex{T}}(undef, N, 0)
    Uk = @view U[:,1:rk]
    Wk = @view W[:,1:rk]
    Σk = @view Σ[1:rk]
    tmp = Matrix{Complex{T}}(undef, N, rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp, A1, Wk)
    @inbounds for j in 1:rk
        @views tmp[:,j] ./= Σk[j]
    end
    B = Matrix{Complex{T}}(undef, rk, rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(B, adjoint(Uk), tmp)
    @blas_multi_then_1 MAX_BLAS_THREADS ev = eigen!(B)
    λ = ev.values
    Φ = Uk*ev.vectors
    λ_keep = Complex{T}[]
    tens = T[]
    idx_keep = Int[]
    @inbounds for j in eachindex(λ)
        abs(λ[j]-k0c)>Rc && continue
        Aj = construct_matrices(solver.kernel, pts, λ[j]; multithreaded)
        res = norm(Aj*@view(Φ[:,j]))
        solver.auto_discard_spurious && res>=solver.res_tol && continue
        push!(λ_keep, λ[j])
        push!(tens, res)
        push!(idx_keep, j)
    end
    Φ_keep = isempty(idx_keep) ? Matrix{Complex{T}}(undef, N, 0) : Φ[:,idx_keep]
    return λ_keep, tens, Φ_keep
end

"""
    solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool = true) → (ks::Vector, ts::Vector)

Solves the Beyn contour eigenproblem centered at `k0` with radius `dk/2`,
returning every retained candidate wavenumber and its tension.

See [`_beyn_solve_core`](@ref) for the underlying contour-projection,
eigensolve and residual-filtering pipeline.

!!! note "Complex roots"
    Beyn's method finds every root of the wrapped kernel's Fredholm
    determinant inside the contour, not only physically real billiard
    eigenvalues; a genuinely complex root (with non-negligible imaginary
    part) can still pass the residual filter and is returned here as
    `real(λ)`, matching `-develop`'s own filtering scope (contour
    containment + residual only). This is only observed with contour radii
    much larger than a single Weyl-window ([`plan_weyl_windows`](@ref))
    would produce; using Weyl-balanced windows avoids it in practice.
"""
function solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    λ, ts, _ = _beyn_solve_core(solver, pts, k0, dk; multithreaded)
    return real.(λ), ts
end

"""
    solve_vectors(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool = true) → (ks::Vector, ts::Vector, X::Matrix)

Solves the Beyn contour eigenproblem centered at `k0` with radius `dk/2`,
returning every retained candidate wavenumber, its tension, and the
corresponding boundary density eigenvector (columns of `X`).

See [`_beyn_solve_core`](@ref) for the underlying contour-projection,
eigensolve and residual-filtering pipeline.
"""
function solve_vectors(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    λ, ts, Φ = _beyn_solve_core(solver, pts, k0, dk; multithreaded)
    return real.(λ), ts, Φ
end

"""
    solve_wavenumber(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool = true) where {Bi<:AbsBilliard} → (k0::Real, t0::Real)

Finds the Beyn eigenvalue candidate closest to the target wavenumber `k`
within a contour of radius `dk/2`, together with its tension.
"""
function solve_wavenumber(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k)
    ks, ts = solve(solver, pts, k, dk; multithreaded)
    isempty(ks) && error("BeynSolver found no eigenvalue candidates in the window [k-dk/2,k+dk/2]=[$(k-dk/2),$(k+dk/2)]")
    idx = findmin(abs.(ks.-k))[2]
    return ks[idx], ts[idx]
end

"""
    solve_spectrum(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool = true) where {Bi<:AbsBilliard} → (ks::Vector, ts::Vector)

Computes every Beyn eigenvalue candidate and its tension within a contour of
center `k` and radius `dk/2`.
"""
function solve_spectrum(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k)
    return solve(solver, pts, k, dk; multithreaded)
end