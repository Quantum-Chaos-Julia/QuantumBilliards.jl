################################################################################
# BEYN CONTOUR-INTEGRAL NONLINEAR EIGENSOLVER
#
# We seek nonlinear eigenpairs
#
#     A(k)u = 0,
#
# where A(k) is the boundary-integral Fredholm matrix. For a closed contour Γ
# enclosing the desired eigenvalues, Beyn's method forms the moments
#
#     A₀ = (1/2πi) ∮Γ A(z)⁻¹ V dz,
#     A₁ = (1/2πi) ∮Γ z A(z)⁻¹ V dz,
#
# with a random probing matrix V. For the circular contour
#
#     z(θ) = k₀ + R exp(iθ),
#
# the trapezoidal rule gives
#
#     A₀ ≈ Σⱼ wⱼ A(zⱼ)⁻¹V,
#     A₁ ≈ Σⱼ wⱼ zⱼ A(zⱼ)⁻¹V,
#     wⱼ = (R/nq) exp(iθⱼ).
#
# If A₀ = UΣW*, its numerical rank r determines the reduced matrix
#
#     B = Uᵣ* A₁ Wᵣ Σᵣ⁻¹.
#
# The eigenvalues of B approximate the nonlinear eigenvalues k inside Γ, while
# the corresponding layer densities are reconstructed as
#
#     X = UᵣY,
#
# where Y contains the eigenvectors of B. Candidates are validated using
#
#     ||A(k)x|| < res_tol.
#
# For closed billiards the exact spectrum is real. In a multi-window sweep we
# therefore sort projected roots globally by decreasing |Im(k)| and residual-
# check the most suspicious roots first. After `imag_k_pad` consecutive roots
# pass the residual test, the remaining roots closer to the real axis are
# accepted without constructing additional residual matrices.
#
# Weyl's leading term determines approximately m states per contour,
#
#     N(k) ≈ A k²/(4π),
#
# while `Rmax` limits the maximum contour radius.
################################################################################

"""
    BeynSolver{T,K} <: AcceleratedBIMSolver

Contour-integral nonlinear eigensolver based on Beyn's method.

The solver computes solutions of `A(k)u = 0` from contour moments on circular
contours. Matrix construction can use the Chebyshev-accelerated backend.
Multi-window spectrum calculations additionally support a global imaginary-`k`
screening strategy which residual-checks roots in decreasing `abs(imag(k))`
order and terminates after a prescribed sequence of residual-good roots.

## Fields
* `kernel::K`: Boundary-integral solver defining `A(k)`.
* `m::Int`: Target number of states per Weyl window.
* `nq::Int`: Number of contour quadrature nodes.
* `r::Int`: Initial random probing rank.
* `Rmax::T`: Maximum contour radius in spectrum sweeps.
* `svd_tol::T`: Numerical-rank threshold for the zeroth contour moment.
* `res_tol::T`: Nonlinear residual threshold as a state - keep check.
* `auto_discard_spurious::Bool`: Discard roots failing residual validation.
* `use_chebyshev::Bool`: Use Chebyshev-accelerated matrix construction.
* `cheb_config::ChebyshevConfig{T}`: Chebyshev interpolation configuration.
* `imag_k_check::Bool`: Enable global imaginary-`k` screening. In production should be true.
* `imag_k_pad::Int`: Consecutive good roots required before stopping if imaginary-`k` screening is enabled.
* `imag_k_group_size::Int`: Maximum residual-check batch size for imaginary-`k` screening. 
* `eigenvectors::Bool`: Whether `compute_spectrum` retains the layer densities returned by the Beyn solve.
"""
struct BeynSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K
    m::Int
    nq::Int
    r::Int
    Rmax::T
    svd_tol::T
    res_tol::T
    auto_discard_spurious::Bool
    use_chebyshev::Bool
    cheb_config::ChebyshevConfig{T}
    imag_k_check::Bool
    imag_k_pad::Int
    imag_k_group_size::Int
    eigenvectors::Bool
end

"""
    BeynSolver(kernel::K; m::Int=10, nq::Int=48, r::Int=48, Rmax::Real=1.0, svd_tol::Real=1e-12, res_tol::Real=1e-9, auto_discard_spurious::Bool=true, use_chebyshev::Bool=true, n_panels_h::Int=15000, M_h::Int=5, n_panels_j::Int=10000, M_j::Int=5, cheb_config::Union{Nothing,ChebyshevConfig}=nothing, imag_k_check::Bool=true, imag_k_pad::Int=20, imag_k_group_size::Int=20)

Construct a Beyn nonlinear eigensolver.

Single-contour calls through `solve` and `solve_vectors` perform exhaustive
nonlinear-residual validation. Multi-window `compute_spectrum` calculations uses
imaginary-`k` screening by default to verify the spectrum integrity.

## Arguments
* `kernel::K`: Boundary-integral solver defining the nonlinear matrix `A(k)`.

## Keyword Arguments
* `m::Int=10`: Target number of states per Weyl window.
* `nq::Int=40`: Number of contour quadrature nodes.
* `r::Int=200`: Initial random probing rank.
* `Rmax::Real=0.5`: Maximum contour radius in spectrum sweeps.
* `svd_tol::Real=1e-11`: Numerical-rank threshold for the zeroth contour moment.
* `res_tol::Real=1e-8`: Nonlinear residual threshold.
* `auto_discard_spurious::Bool=true`: Discard roots failing residual validation.
* `use_chebyshev::Bool=true`: Use Chebyshev-accelerated matrix construction.
* `n_panels_h::Int=10000`: Hankel Chebyshev initial panel count.
* `M_h::Int=5`: Hankel Chebyshev polynomial initial degree.
* `n_panels_j::Int=5000`: Bessel Chebyshev initial panel count.
* `M_j::Int=5`: Bessel Chebyshev polynomial initial degree.
* `cheb_config::Union{Nothing,ChebyshevConfig}=nothing`: Explicit Chebyshev configuration. If nothing will construct it from the provided initial panel counts and polynomial degrees.
* `imag_k_check::Bool=true`: Enable imaginary-`k` screening in `compute_spectrum`. Should always be used in production runs to verify spectrum integrity.
* `imag_k_pad::Int=20`: Consecutive residual-good roots required before stopping. Only applies when `imag_k_check` is enabled.
* `imag_k_group_size::Int=20`: Maximum residual-check batch size. Only applies when `imag_k_check` is enabled.
* `eigenvectors::Bool`: Whether `compute_spectrum` retains the layer densities returned by the Beyn solve.

## Returns
* `BeynSolver{T,K}`: Configured Beyn solver.
"""
function BeynSolver(kernel::K; m::Int=100, nq::Int=40, r::Int=200, Rmax::Real=0.5, svd_tol::Real=1e-11, res_tol::Real=1e-8, auto_discard_spurious::Bool=true, use_chebyshev::Bool=true, n_panels_h::Int=10000, M_h::Int=5, n_panels_j::Int=5000, M_j::Int=5, cheb_config::Union{Nothing,ChebyshevConfig}=nothing, imag_k_check::Bool=true, imag_k_pad::Int=20, imag_k_group_size::Int=20, eigenvectors::Bool=true) where {K<:SweepBIMSolver}
    T = _bim_numeric_type(kernel); cfg = cheb_config === nothing ? ChebyshevConfig(T; n_panels_h, M_h, n_panels_j, M_j) : cheb_config
    return BeynSolver{T,K}(kernel, m, nq, r, T(Rmax), T(svd_tol), T(res_tol), auto_discard_spurious, use_chebyshev, cfg, imag_k_check, imag_k_pad, imag_k_group_size, eigenvectors)
end

_bim_numeric_type(::BeynSolver{T}) where {T} = T

@inline function weyl_window_width(billiard::Bi, k::T, m::Int; fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    A = fundamental ? fundamental_area(billiard) : area(billiard)
    return sqrt(k^2+T(4pi*m/A))-k
end

"""
    plan_weyl_windows(billiard::Bi, k1::T, k2::T; m::Int=10, Rmax::Real=1.0, fundamental::Bool=true)

Partition `[k1,k2]` into adjacent spectral windows containing approximately
`m` states according to the leading Weyl law.

## Arguments
* `billiard::Bi`: Billiard geometry.
* `k1::T`: Lower spectral endpoint.
* `k2::T`: Upper spectral endpoint.

## Keyword Arguments
* `m::Int=10`: Approximate number of states per window.
* `Rmax::Real=1.0`: Maximum contour radius.
* `fundamental::Bool=true`: Use the fundamental-domain area.

## Returns
* `Vector{Tuple{T,T}}`: Adjacent spectral intervals `(kL,kR)`.
"""
function plan_weyl_windows(billiard::Bi, k1::T, k2::T; m::Int=10, Rmax::Real=1.0, fundamental::Bool=true) where {T<:Real,Bi<:AbsBilliard}
    k2 > k1 || return Tuple{T,T}[]; m > 0 || throw(ArgumentError("m must be positive; received m=$m")); Rmax > 0 || throw(ArgumentError("Rmax must be positive; received Rmax=$Rmax"))
    intervals = Tuple{T,T}[]; maxwidth = T(2Rmax); k = k1
    while k < k2
        dk = min(weyl_window_width(billiard, k, m; fundamental), maxwidth, k2-k); dk > zero(T) || throw(ArgumentError("Weyl window width vanished at k=$k"))
        kr = k+dk; push!(intervals, (k, kr)); k = kr
    end
    return intervals
end

function beyn_disks_from_windows(intervals::Vector{Tuple{T,T}}) where {T<:Real}
    k0 = Vector{Complex{T}}(undef, length(intervals)); R = Vector{T}(undef, length(intervals))
    @inbounds for (i, (kL, kR)) in pairs(intervals)
        k0[i] = complex((kL+kR)/2); R[i] = (kR-kL)/2
    end
    return k0, R
end

function beyn_buffer_matrices(::Type{T}, N::Int, r::Int, rng) where {T<:Real}
    V = randn(rng, Complex{T}, N, r); X = similar(V); A0 = zeros(Complex{T}, N, r); A1 = zeros(Complex{T}, N, r)
    return V, X, A0, A1
end

function _construct_matrices_multi_k_cheb(cs::DoubleLayerPotentialSolver, pts::BoundaryPoints{T}, zj::Vector{ComplexF64}, cfg::ChebyshevConfig; multithreaded::Bool=true) where {T<:Real}
    T === Float64 || error("Chebyshev-accelerated Beyn evaluation requires Float64")
    N = length(pts); graded = _is_nontrivial_dlp_grading(pts); G = boundary_geom_cache(pts, graded); Rmat = zeros(T, N, N); kress_R!(Rmat)
    rmin, rmax = _cheb_geom_rminmax(G, zj); plans1, plansj1, _ = tune_dlp_cheb_plans(rmin, rmax, zj, cfg)
    if cs.symmetry === nothing
        Tbufs = [Matrix{ComplexF64}(undef, N, N) for _ in zj]
        _dlp_fredholm_full_multi_k_cheb!(Tbufs, pts, Rmat, G, zj, plans1, plansj1; multithreaded)
    else
        orbits = _fold_boundary(T, pts.xy, cs.symmetry, cs.character); M = fundamental_size(orbits); Tbufs = [Matrix{ComplexF64}(undef, M, M) for _ in zj]
        _dlp_fredholm_reduced_multi_k_cheb!(Tbufs, pts, Rmat, G, orbits, zj, plans1, plansj1; multithreaded)
    end
    return Tbufs
end

function _construct_matrices_multi_k_cheb(cs::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, zj::Vector{ComplexF64}, cfg::ChebyshevConfig; multithreaded::Bool=true) where {T<:Real}
    T === Float64 || error("Chebyshev-accelerated Beyn evaluation requires Float64")
    N = length(pts); graded = _is_nontrivial_dlp_grading(pts); G = boundary_geom_cache(pts, graded); Rmat = zeros(T, N, N); kress_R!(Rmat)
    rmin, rmax = _cheb_geom_rminmax(G, zj); plans0, plans1, plansj0, plansj1, _ = tune_cfie_cheb_plans(rmin, rmax, zj, cfg)
    if cs.symmetry === nothing
        Tbufs = [Matrix{ComplexF64}(undef, N, N) for _ in zj]
        _cfie_fredholm_full_multi_k_cheb!(Tbufs, pts, Rmat, G, zj, plans0, plans1, plansj0, plansj1; multithreaded)
    else
        orbits = _fold_boundary(T, pts.xy, cs.symmetry, cs.character); M = fundamental_size(orbits); Tbufs = [Matrix{ComplexF64}(undef, M, M) for _ in zj]
        _cfie_fredholm_reduced_multi_k_cheb!(Tbufs, pts, Rmat, G, orbits, zj, plans0, plans1, plansj0, plansj1; multithreaded)
    end
    return Tbufs
end

_construct_matrices_multi_k_cheb(::CompositeBIMSolver, ::BoundaryPoints, ::Vector{ComplexF64}, ::ChebyshevConfig; multithreaded::Bool=true) = error("Chebyshev-accelerated Beyn is not implemented for CompositeBIMSolver")

function _accumulate_beyn_moments!(A0, A1, X, V, Fs, zj, wj)
    xv = reshape(X, :); a0v = reshape(A0, :); a1v = reshape(A1, :)
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in eachindex(zj)
        ldiv!(X, Fs[j], V); BLAS.axpy!(wj[j], xv, a0v); BLAS.axpy!(wj[j]*zj[j], xv, a1v)
    end
    return nothing
end

"""
    construct_matrices(solver::BeynSolver, pts::BoundaryPoints, k0, R; multithreaded::Bool=true, rng=MersenneTwister(0))

Construct the zeroth and first contour moments used by Beyn's nonlinear
eigensolver. Fredholm matrices are assembled and factorized once; if the
zeroth moment remains rank-saturated, the probing rank is increased while
reusing those factorizations.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Complex contour center.
* `R`: Contour radius.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `rng=MersenneTwister(0)`: Random-number generator for probing matrices.

## Returns
* `A0::Matrix{Complex{T}}`: Zeroth contour moment.
* `A1::Matrix{Complex{T}}`: First contour moment.
"""
function construct_matrices(solver::BeynSolver, pts::BoundaryPoints, k0, R; multithreaded::Bool=true, rng=MersenneTwister(0))
    T = _bim_numeric_type(solver); N = boundary_matrix_size(solver.kernel, pts); k0c = Complex{T}(k0); Rc = T(R); nq = solver.nq
    θ = range(zero(T), 2T(pi); length=nq+1)[1:end-1]; ej = cis.(θ); zj = k0c .+ Rc.*ej; wj = (Rc/nq).*ej
    if solver.use_chebyshev
        Tbufs = _construct_matrices_multi_k_cheb(solver.kernel, pts, ComplexF64.(zj), solver.cheb_config; multithreaded)
    else
        Tbufs = Vector{Matrix{Complex{T}}}(undef, nq)
        @inbounds for j in eachindex(zj)
            Tbufs[j] = construct_matrices(solver.kernel, pts, zj[j]; multithreaded)
        end
    end
    F1 = lu!(Tbufs[1]; check=false); Fs = Vector{typeof(F1)}(undef, nq); Fs[1] = F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq
        Fs[j] = lu!(Tbufs[j]; check=false)
    end
    r = min(solver.r, N)
    while true
        V, X, A0, A1 = beyn_buffer_matrices(T, N, r, rng); _accumulate_beyn_moments!(A0, A1, X, V, Fs, zj, wj)
        @blas_multi_then_1 MAX_BLAS_THREADS Σ = svdvals(A0)
        rk = count(>=(solver.svd_tol), Σ)
        rk < r && return A0, A1
        r == N && throw(ArgumentError("Beyn moment remains rank-saturated at maximum probe rank N=$N"))
        r = min(r+solver.r, N)
    end
end

"""
    _beyn_projected_solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0))

Perform the projected stage of Beyn's method without nonlinear-residual
validation. The contour has center `k0` and radius `dk/2`. The reduced Beyn
matrix is diagonalized, layer density vectors are reconstructed, and roots
outside the contour are discarded.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Contour center.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `rng=MersenneTwister(0)`: Random-number generator for probing.

## Returns
* `ks::Vector{Complex{T}}`: Complex roots inside the contour.
* `X::Matrix{Complex{T}}`: Corresponding layer density vectors as columns.
"""
function _beyn_projected_solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0))
    T = _bim_numeric_type(solver); k0c = Complex{T}(k0); Rc = T(dk)/2; 
    @blas_1 A0, A1 = construct_matrices(solver, pts, k0c, Rc; multithreaded, rng)
    N = size(A0, 1)
    @blas_multi_then_1 MAX_BLAS_THREADS U, Σ, W = svd!(A0; full=false)
    rk = count(>=(solver.svd_tol), Σ)
    rk == 0 && return Complex{T}[], Matrix{Complex{T}}(undef, N, 0)
    Uk = @view U[:, 1:rk]; Wk = @view W[:, 1:rk]; Σk = @view Σ[1:rk]; tmp = Matrix{Complex{T}}(undef, N, rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp, A1, Wk)
    @inbounds for j in 1:rk
        @views tmp[:, j] ./= Σk[j]
    end
    B = Matrix{Complex{T}}(undef, rk, rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(B, adjoint(Uk), tmp)
    @blas_multi_then_1 MAX_BLAS_THREADS ev = eigen!(B)
    ks = ev.values; X = Uk*ev.vectors; idx = findall(j -> abs(ks[j]-k0c) <= Rc, eachindex(ks))
    isempty(idx) && return Complex{T}[], Matrix{Complex{T}}(undef, N, 0)
    return ks[idx], X[:, idx]
end

# Compute the residual of a candidate root and its corresponding layer density vector.
@inline function _beyn_residual(solver::BeynSolver, pts, k, x, y; multithreaded::Bool=true)
    @blas_1 A = construct_matrices(solver.kernel, pts, k; multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(y, A, x)
    return norm(y)
end

"""
    _beyn_solve_core(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0))

Perform an exhaustive single-contour Beyn solve. Every projected candidate is
explicitly residual-checked. This is the reference path used by `solve` and
`solve_vectors`.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Contour center.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `rng=MersenneTwister(0)`: Random-number generator for probing.

## Returns
* `ks::Vector{Complex{T}}`: Accepted complex roots.
* `residuals::Vector{T}`: Corresponding nonlinear residuals.
* `X::Matrix{Complex{T}}`: Corresponding layer density vectors.
"""
function _beyn_solve_core(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0))
    T = _bim_numeric_type(solver); ks, X = _beyn_projected_solve(solver, pts, k0, dk; multithreaded, rng); N = size(X, 1)
    isempty(ks) && return Complex{T}[], T[], Matrix{Complex{T}}(undef, N, 0)
    !solver.auto_discard_spurious && return ks, fill(T(NaN), length(ks)), X
    residuals = Vector{T}(undef, length(ks)); keep = falses(length(ks)); y = Vector{Complex{T}}(undef, N)
    @inbounds for j in eachindex(ks)
        rj = _beyn_residual(solver, pts, ks[j], @view(X[:, j]), y; multithreaded); residuals[j] = rj; keep[j] = rj < solver.res_tol
    end
    idx = findall(keep)
    return ks[idx], residuals[idx], isempty(idx) ? Matrix{Complex{T}}(undef, N, 0) : X[:, idx]
end

"""
    _beyn_imag_k_check(solver::BeynSolver, ks_all, X_all, all_pts; pad::Int=solver.imag_k_pad, group_size::Int=solver.imag_k_group_size, multithreaded::Bool=true)

Apply imaginary-`k` screening to projected roots from multiple Beyn
windows.

Candidates are sorted by decreasing `abs(imag(k))`. Residual
validation therefore starts with roots furthest from the real axis. A failed
residual resets the good-root streak and is discarded when
`solver.auto_discard_spurious=true`. Once `pad` consecutive roots satisfy the
residual tolerance, all remaining roots are accepted without explicit residual
evaluation.

Candidates from the same spectral window are evaluated together using the
multi-`k` Chebyshev backend when available.

## Arguments
* `solver::BeynSolver`: Beyn solver and filtering configuration.
* `ks_all`: Complex projected roots for each window.
* `X_all`: layer density matrices corresponding to `ks_all`.
* `all_pts`: Boundary discretization associated with each window.

## Keyword Arguments
* `pad::Int=solver.imag_k_pad`: Consecutive good roots required before stopping.
* `group_size::Int=solver.imag_k_group_size`: Maximum residual-check batch size.
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
* `idx_keep::Vector{Vector{Int}}`: Retained local root indices for each window.
* `residuals::Vector{Vector{T}}`: Residuals aligned with retained roots. Roots
  accepted after early termination contain `NaN`.
"""
function _beyn_imag_k_check(solver::BeynSolver, ks_all, X_all, all_pts; pad::Int=solver.imag_k_pad, group_size::Int=solver.imag_k_group_size, multithreaded::Bool=true)
    T = _bim_numeric_type(solver); nw = length(ks_all)
    keep = [trues(length(ks_all[i])) for i in 1:nw]; residuals = [fill(T(NaN), length(ks_all[i])) for i in 1:nw]; candidates = Tuple{Int,Int,T}[]
    @inbounds for i in 1:nw
        for j in eachindex(ks_all[i])
            push!(candidates, (i, j, abs(imag(ks_all[i][j]))))
        end
    end
    isempty(candidates) && return [Int[] for _ in 1:nw], residuals
    sort!(candidates; by=c -> c[3], rev=true)
    window_js = [Int[] for _ in 1:nw]; touched = Int[]; streak = 0; pos = 1; nc = length(candidates)
    while pos <= nc
        last = min(pos+group_size-1, nc); empty!(touched)
        @inbounds for q in pos:last
            i, j, _ = candidates[q]; isempty(window_js[i]) && push!(touched, i); push!(window_js[i], j)
        end
        @inbounds for i in touched
            js = window_js[i]; pts = all_pts[i]; N = boundary_matrix_size(solver.kernel, pts); nk = length(js); y = Vector{Complex{T}}(undef, N)
            if solver.use_chebyshev && solver.kernel isa Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver}
                zg = Vector{ComplexF64}(undef, nk)
                for q in 1:nk
                    zg[q] = ComplexF64(ks_all[i][js[q]])
                end
                Abufs = _construct_matrices_multi_k_cheb(solver.kernel, pts, zg, solver.cheb_config; multithreaded)
                for q in 1:nk
                    j = js[q]
                    @blas_multi_then_1 MAX_BLAS_THREADS mul!(y, Abufs[q], @view(X_all[i][:, j]))
                    residuals[i][j] = norm(y)
                end
            else
                for j in js
                    residuals[i][j] = _beyn_residual(solver, pts, ks_all[i][j], @view(X_all[i][:, j]), y; multithreaded)
                end
            end
        end
        stop = false
        @inbounds for q in pos:last
            i, j, _ = candidates[q]; rj = residuals[i][j]
            if rj >= solver.res_tol
                solver.auto_discard_spurious && (keep[i][j] = false); streak = 0
            else
                streak += 1
                if streak >= pad
                    stop = true; break
                end
            end
        end
        @inbounds for i in touched
            empty!(window_js[i])
        end
        stop && break
        pos = last+1
    end
    idx_keep = [findall(keep[i]) for i in 1:nw]
    return idx_keep, [residuals[i][idx_keep[i]] for i in 1:nw]
end

"""
    solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)

Solve one circular Beyn contour with residual validation.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Contour center.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
* `ks::Vector{Complex{T}}`: Accepted complex eigenvalues.
* `residuals::Vector{T}`: Corresponding nonlinear residuals.
"""
function solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    ks, residuals, _ = _beyn_solve_core(solver, pts, k0, dk; multithreaded)
    return ks, residuals
end

"""
    solve_vectors(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)

Solve one circular Beyn contour with residual validation and return
the corresponding layer density vectors.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Contour center.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
* `ks::Vector{Complex{T}}`: Accepted complex eigenvalues.
* `residuals::Vector{T}`: Corresponding nonlinear residuals.
* `X::Matrix{Complex{T}}`: Corresponding layer density vectors.
"""
function solve_vectors(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    return _beyn_solve_core(solver, pts, k0, dk; multithreaded)
end

"""
    solve_wavenumber(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}

Solve for the eigenvalue closest to the given wavenumber `k` within a circular Beyn contour.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `billiard::AbsBilliard`: Billiard instance.
* `k`: Target wavenumber.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
* `k::Complex{T}`: Eigenvalue closest to the target wavenumber.
* `residual::T`: Corresponding nonlinear residual.
"""
function solve_wavenumber(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k); ks, residuals = solve(solver, pts, k, dk; multithreaded)
    isempty(ks) && error("BeynSolver found no eigenvalue candidates in window")
    idx = findmin(abs.(ks .- k))[2]
    return ks[idx], residuals[idx]
end

function solve_spectrum(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k)
    return solve(solver, pts, k, dk; multithreaded)
end