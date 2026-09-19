################################################################################
# CHEBYSHEV-CORK BOUNDARY-INTEGRAL EIGENSOLVER
#
# This file implements a compact CORK eigensolver for BIM nonlinear
# eigenvalue problems
#
#                              A(k)u = 0,
#
# where A(k) is the Fredholm matrix of a DLP, CFIE, or composite BIM
# discretization. For a requested spectral window
#
#                  Δ = dk/2,    k ∈ [k₀-Δ,k₀+Δ],
#
# `recurrences.jl` constructs a Chebyshev polynomial approximation on the
# guarded half-width Δₚ = (1 + guard)Δ,
#
#              P(t) = Σⱼ₌₀ᵖ BⱼTⱼ(t) ≈ A(k₀ + Δₚt),
#              t = (k-k₀)/Δₚ.
#
# CORK solves the polynomial eigenvalue problem P(t)u = 0 through a compact
# Chebyshev linearization. Instead of explicitly storing N-dimensional
# vectors for every linearization block, the physical blocks are represented
# in a common basis U,
#
#                              zⱼ = Ugⱼ,
#
# with cached coefficient actions
#
#                              Wⱼ = BⱼU.
#
# At the fixed inverse-iteration shift t = 0,
#
#                         P(0) = B₀-B₂+B₄-⋯,
#
# since T₂q(0) = (-1)^q and T₂q₊₁(0) = 0. One LU factorization of P(0) is
# therefore reused throughout the block-Arnoldi iteration. Eigenvalues μ of
# the projected inverse linearization are mapped back to physical
# wavenumbers by
#
#                         μ → t = 1/μ → k = k₀ + Δₚt.
#
# The compact Krylov space is grown adaptively until the requested spectrum
# is stable and the requested interval edges are represented by converged
# Ritz roots. There is no user-defined maximum Krylov dimension: the
# iteration may grow to the largest block-compatible dimension permitted by
# the physical Fredholm matrix.
#
# Main spectrum call:
#
# solve_spectrum(solver, billiard, k0, dk)      # Solve a complete spectral window
# │
# ├─ evaluate_points(solver, billiard, k0)      # Build the boundary discretization
# │    └─ evaluate_points(solver.kernel, ...)   # Delegate to the wrapped BIM solver
# │
# └─ solve(solver, pts, k0, dk)                 # Solve using existing boundary points
#      │
#      └─ _cork_solve_core(...)                 # Run the polynomial + CORK pipeline
#           │
#           ├─ build_cork_polynomial(...)       # Construct P(t) analytically
#           │    │
#           │    ├─ _taylor_cache(...)          # Precompute geometry/Kress data
#           │    │
#           │    ├─ build_B_full(...)           # Assemble Bⱼ on the full boundary
#           │    │    └─ _fredholm_taylor!      # Taylor coefficients of I-K(k)
#           │    │         └─ _kernel_taylor!   # Analytic kernel Taylor recurrence
#           │    │
#           │    └─ build_B_reduced(...)        # Assemble symmetry-reduced Bⱼ
#           │         └─ _fredholm_taylor!      # Reduced Fredholm Taylor coefficients
#           │              └─ _kernel_taylor!   # Analytic kernel Taylor recurrence
#           │
#           │       # The functions above live in `recurrences.jl` and return
#           │       # CORKPolynomial(B, k0, Δpoly, p, N).
#           │
#           ├─ validate_polynomial!(...)        # Compare P(t) with direct BIM matrices
#           │                                   # when `validate = true`
#           │
#           ├─ get_A0(B, N, p)                  # Evaluate P(0) = B₀-B₂+B₄-⋯
#           │
#           ├─ lu(P(0))                         # Factorize the fixed shift once
#           │
#           └─ adaptive_cork(...)               # Grow CORK until spectrum convergence
#                │
#                ├─ init_cork(...)              # Initialize persistent CORK state
#                │    │
#                │    ├─ initialize U           # Initial common physical basis
#                │    ├─ initialize G           # Initial compact Arnoldi block
#                │    └─ cache_block!(...)      # Cache Wⱼ = BⱼU
#                │
#                └─ for increasing m            # Extend one persistent factorization
#                     │
#                     ├─ extend!(...)            # Grow active Krylov dimension to m
#                     │    │
#                     │    ├─ compute_tail!      # Compute and retain Arnoldi residual
#                     │    │    │
#                     │    │    ├─ block_apply!(...)
#                     │    │    │    ├─ Chebyshev recurrence
#                     │    │    │    ├─ P(0)\rhs
#                     │    │    │    ├─ physical SVD
#                     │    │    │    └─ cache_block!
#                     │    │    │
#                     │    │    ├─ pack!         # Flatten compact coordinates
#                     │    │    ├─ compact_project!
#                     │    │    │                # Two-pass Arnoldi reorthogonalization
#                     │    │    ├─ compact_block_qr!
#                     │    │    │                # Normalize the residual block
#                     │    │    └─ pending tail  # Retain residual for Ritz testing
#                     │    │
#                     │    └─ promote_tail!      # Promote residual to active block
#                     │
#                     ├─ ritz_roots(...)         # Extract individual projected Ritz roots
#                     │    │
#                     │    ├─ eigen(Hm)          # Projected inverse eigenproblem
#                     │    ├─ μ → 1/μ → k        # Map Ritz values to wavenumbers
#                     │    └─ Htail*v             # Projected Arnoldi residual estimate
#                     │
#                     ├─ requested roots         # Retain roots in [k₀-Δ,k₀+Δ]
#                     ├─ physical filtering      # Apply Im(k) and residual tolerances
#                     ├─ edge check              # Check extreme requested Ritz roots
#                     └─ stability check         # Compare with previous Krylov dimension
################################################################################

# Views of the j-th Chebyshev coefficient and cached coefficient action.
@inline Bj(B::Matrix{ComplexF64}, N::Int, j::Int) = @view B[j * N + 1:(j + 1) * N,:]
@inline Wj(W::Matrix{ComplexF64}, N::Int, j::Int) = @view W[j * N + 1:(j + 1) * N,:]

"""
    evaluate_cork_polynomial(P::CORKPolynomial, t::Real) -> Matrix{ComplexF64}

Evaluate the matrix Chebyshev polynomial `P(t)=Σⱼ₌₀ᵖBⱼTⱼ(t)` using the
three-term recurrence `T₀(t)=1`, `T₁(t)=t`, and `Tⱼ₊₁(t)=2tTⱼ(t)-Tⱼ₋₁(t)`.

## Arguments
- `P::CORKPolynomial`: Chebyshev polynomial with vertically stacked coefficients `[B₀;...;Bₚ]`.
- `t::Real`: Normalized Chebyshev coordinate.

## Returns
- `Matrix{ComplexF64}`: Dense matrix `P(t)`.
"""
function evaluate_cork_polynomial(P::CORKPolynomial, t::Real)::Matrix{ComplexF64}
    τ = Float64(t); A = copy(Bj(P.B, P.N, 0))
    P.p == 0 && return A
    Tm = 1.0; Tn = τ
    axpy!(Tn, Bj(P.B, P.N, 1), A)
    @inbounds for j = 1:P.p-1
        Tp = 2τ * Tn - Tm
        axpy!(Tp, Bj(P.B, P.N, j + 1), A)
        Tm, Tn = Tn, Tp
    end
    return A
end

"""
    validate_polynomial!(solver::SweepBIMSolver, pts, P::CORKPolynomial, tol::Real; nsample::Int=5, multithreaded::Bool=true) -> Nothing

Validate the Chebyshev approximation against directly constructed BIM Fredholm matrices.
The reported relative error is `‖P(t)-A(k₀+Δt)‖/‖A(k₀+Δt)‖`. 
This is basically to check if the bounds of the interval are accurate enough.

## Arguments
- `solver::SweepBIMSolver`: BIM solver defining the direct Fredholm matrix.
- `pts`: Boundary discretization used to construct the polynomial.
- `P::CORKPolynomial`: Chebyshev polynomial to validate.
- `tol::Real`: Tolerance for the worst relative error.
- `nsample::Int`: Number of sample points in `[-1,1]`.
- `multithreaded::Bool`: Whether direct matrix construction uses threading.

## Returns
- `Nothing`: Returns `nothing` if the polynomial validation passes.
"""
function validate_polynomial!(solver::SweepBIMSolver, pts, P::CORKPolynomial, tol::Real; nsample::Int=5, multithreaded::Bool=true)::Nothing
    ts = collect(range(-1.0, 1.0, length = nsample)); worst = 0.0
    for t in ts
        k = P.k0 + P.Δ * t; Ap = evaluate_cork_polynomial(P, t)
        Ad = construct_matrices(solver, pts, k; multithreaded = multithreaded)
        err = norm(Ap - Ad) / norm(Ad); worst = max(worst, err)
    end
    worst > tol && throw(ArgumentError("Polynomial validation failed: worst relative error = $worst exceeds tolerance $tol. Try reducing the interval Rmax or increasing the polynomial degree p."))
    return nothing
end


"""
    get_A0(B::Matrix{ComplexF64}, N::Int, p::Int) -> Matrix{ComplexF64}

Evaluate the Chebyshev polynomial `P(t)=Σⱼ₌₀ᵖBⱼTⱼ(t)` at the fixed shift `t=0`. Since
`T₂q(0)=(-1)^q` and `T₂q₊₁(0)=0`, `P(0)=B₀-B₂+B₄-B₆+⋯`.

## Arguments
- `B::Matrix{ComplexF64}`: Vertically stacked coefficients `[B₀;...;Bₚ]`.
- `N::Int`: Dimension of each coefficient matrix `Bⱼ`.
- `p::Int`: Chebyshev polynomial degree.

## Returns
- `Matrix{ComplexF64}`: Dense physical shift matrix `P(0)`.
"""
function get_A0(B::Matrix{ComplexF64}, N::Int, p::Int)::Matrix{ComplexF64}
    A = zeros(ComplexF64, N, N)
    @inbounds for j = 0:2:p
        axpy!(isodd(j ÷ 2) ? -1.0 : 1.0, Bj(B, N, j), A)
    end
    return A
end

"""
    cache_block!(W::Matrix{ComplexF64}, B::Matrix{ComplexF64}, U::Matrix{ComplexF64}, N::Int, p::Int, a::Int, b::Int) -> Float64

Cache all Chebyshev coefficient actions on `U[:,a:b]` by evaluating the
independent products `Wⱼ[:,a:b]=BⱼU[:,a:b]`, `j=0,...,p`, concurrently over
the vertically stacked coefficient blocks.

## Arguments
- `W::Matrix{ComplexF64}`: Vertically stacked coefficient-action cache.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `U::Matrix{ComplexF64}`: Common physical CORK basis.
- `N::Int`: Physical Fredholm matrix dimension.
- `p::Int`: Chebyshev polynomial degree.
- `a::Int`: First physical basis column to cache.
- `b::Int`: Last physical basis column to cache.

## Returns
- `Float64`: Elapsed coefficient-caching time in seconds.
"""
@inline function cache_block!(W::Matrix{ComplexF64}, B::Matrix{ComplexF64}, U::Matrix{ComplexF64}, N::Int, p::Int, a::Int, b::Int)::Float64
    a > b && return 0.0
    X = @view U[:,a:b]; t = time_ns()
    @blas_1 Threads.@threads :static for j = 0:p
        rows = j * N + 1:(j + 1) * N
        mul!(@view(W[rows,a:b]), @view(B[rows,:]), X)
    end
    return (time_ns() - t) * 1e-9
end

"""
    compact_project!(Z::Array{ComplexF64,3}, G::Array{ComplexF64,3}, r::Int, n::Int, p::Int, H1::Matrix{ComplexF64}, H2::Matrix{ComplexF64}) -> Nothing

Orthogonalize the active compact tensor `Z[1:r,1:p,:]` against the active
compact Arnoldi basis `G[1:r,1:p,1:n]` by two-pass classical Gram-Schmidt.

The compact inner product is evaluated directly in tensor form,

    H = Σⱼ GⱼᴴZⱼ,

so inactive physical rows `r+1:rmax` are never processed and no padded
flattened representation is required. On return `H1[1:n,:]` contains the
accumulated projection coefficients from both CGS passes.

## Returns
- `Nothing`: `Z` and `H1` are modified in place.
"""
function compact_project!(Z::Array{ComplexF64,3}, G::Array{ComplexF64,3}, r::Int, n::Int, p::Int, H1::Matrix{ComplexF64}, H2::Matrix{ComplexF64}; η::Float64=inv(sqrt(2.0)))::Bool
    A = @view H1[1:n,:]; C = @view H2[1:n,:]
    fill!(A, 0); fill!(C, 0)
    b = size(Z, 3); ν0 = zeros(Float64, b); ν1 = zeros(Float64, b)
    @inbounds for c = 1:b
        s = 0.0
        for j = 1:p, i = 1:r
            s += abs2(Z[i,j,c])
        end
        ν0[c] = sqrt(s)
    end
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        for j = 1:p
            Gj = @view G[1:r,j,1:n]; Zj = @view Z[1:r,j,:]
            mul!(A, adjoint(Gj), Zj, 1 + 0im, 1 + 0im)
        end
        for j = 1:p
            Gj = @view G[1:r,j,1:n]; Zj = @view Z[1:r,j,:]
            mul!(Zj, Gj, A, -1 + 0im, 1 + 0im)
        end
    end
    @inbounds for c = 1:b
        s = 0.0
        for j = 1:p, i = 1:r
            s += abs2(Z[i,j,c])
        end
        ν1[c] = sqrt(s)
    end
    reorth = any(c -> ν1[c] < η * ν0[c], 1:b)
    if reorth
        @blas_multi_then_1 MAX_BLAS_THREADS begin
            for j = 1:p
                Gj = @view G[1:r,j,1:n]; Zj = @view Z[1:r,j,:]
                mul!(C, adjoint(Gj), Zj, 1 + 0im, 1 + 0im)
            end
            for j = 1:p
                Gj = @view G[1:r,j,1:n]; Zj = @view Z[1:r,j,:]
                mul!(Zj, Gj, C, -1 + 0im, 1 + 0im)
            end
        end
        A .+= C
    end
    return reorth
end

"""
    compact_block_qr!(Z::Matrix{ComplexF64}, b::Int) -> Tuple{Int,Matrix{ComplexF64}}

Compute the thin Householder QR factorization `Z₀ = Z*R` of a compact CORK
block. The input matrix is overwritten by the orthonormal factor `Q`.

## Arguments
- `Z::Matrix{ComplexF64}`: Compact block, overwritten by the thin orthonormal factor `Q`.
- `b::Int`: Block size.

## Returns
- `Int`: Numerical block rank.
- `Matrix{ComplexF64}`: Upper-triangular `b × b` factor `R`.
"""
function compact_block_qr!(Z::Matrix{ComplexF64}, b::Int)::Tuple{Int,Matrix{ComplexF64}}
    F = qr!(Z)
    R = Matrix(F.R)[1:b,1:b]
    Q = Matrix(F.Q[:,1:b])
    copyto!(Z, Q)
    d = abs.(diag(R)); tol = maximum(size(Z)) * eps(Float64) * maximum(d)
    return count(>(tol), d), R
end

@inline function pack!(Zf::AbstractMatrix{ComplexF64}, Z::Array{ComplexF64,3}, r::Int, p::Int, b::Int)::Nothing
    @inbounds for c = 1:b, j = 1:p, i = 1:r
        Zf[(j - 1) * r + i,c] = Z[i,j,c]
    end
    return nothing
end

@inline function unpack!(Z::Array{ComplexF64,3}, Zf::AbstractMatrix{ComplexF64}, r::Int, p::Int, b::Int)::Nothing
    @inbounds for c = 1:b, j = 1:p, i = 1:r
        Z[i,j,c] = Zf[(j - 1) * r + i,c]
    end
    return nothing
end

"""
    block_apply!(Z::Array{ComplexF64,3}, U::Matrix{ComplexF64}, W::Matrix{ComplexF64}, B::Matrix{ComplexF64}, F, Gin, N::Int, p::Int, r::Int, b::Int)

Apply one block of the inverse Chebyshev linearization at `t=0` in compact
CORK form.

For `P(t)=Σⱼ₌₀ᵖBⱼTⱼ(t)`, the recurrence `Tⱼ₊₁+Tⱼ₋₁=2tTⱼ` gives a
degree-`p` linearization. CORK represents each physical block by `zⱼ=Ugⱼ`
with `U∈ℂᴺˣʳ` and caches `Wⱼ=BⱼU`.

For input blocks `x₀,...,xₚ₋₁`, the recurrence rows at `t=0` give
`γ₀=0`, `γ₁=x₀`, `γⱼ₊₁=-γⱼ₋₁+2xⱼ`. The final row is

    Σⱼ₌₀ᵖ⁻³Bⱼzⱼ+(Bₚ₋₂-Bₚ)zₚ₋₂+Bₚ₋₁zₚ₋₁=-2tBₚzₚ₋₁.

Writing `zⱼ=Uγⱼ+Tⱼ(0)y` gives `P(0)y=rhs`, where

    rhs=-Σⱼ₌₀ᵖ⁻³Wⱼγⱼ-Wₚ₋₂γₚ₋₂+Wₚγₚ₋₂-Wₚ₋₁γₚ₋₁-2Wₚxₚ₋₁.

The solved physical block is projected twice against `U`. Its orthogonal
remainder is factorized by SVD, which detects new independent physical
directions robustly in degenerate and nearly degenerate blocks. The number of
new directions is capped by the remaining dimension of the physical space.

## Arguments
- `Z::Array{ComplexF64,3}`: Preallocated output compact tensor.
- `U::Matrix{ComplexF64}`: Common physical basis.
- `W::Matrix{ComplexF64}`: Cached vertically stacked actions `BⱼU`.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `F`: LU factorization of `P(0)`.
- `Gin`: Input compact degree coordinates.
- `N::Int`: Physical Fredholm matrix dimension.
- `p::Int`: Chebyshev degree and number of linearization blocks.
- `r::Int`: Current physical rank.
- `b::Int`: Block-Arnoldi block size.

## Returns
- `Tuple{Int,Float64,Float64,Float64,Float64}`: Updated physical rank and
  times for coefficient caching, LU solution, RHS assembly, and physical-basis
  expansion.
"""
function block_apply!(Z::Array{ComplexF64,3}, U::Matrix{ComplexF64}, W::Matrix{ComplexF64}, B::Matrix{ComplexF64}, F, Gin, N::Int, p::Int, r::Int, b::Int)
    p >= 2 || error("CORK Chebyshev action requires p >= 2")
    γ = zeros(ComplexF64, r, p, b); RHS = zeros(ComplexF64, N, b); tmp = similar(RHS); Y = similar(RHS)
    Hphys = zeros(ComplexF64, r, b); @views γ[:,2,:] .= Gin[1:r,1,:]
    for j = 1:p-2
        @views @. γ[:,j + 2,:] = -γ[:,j,:] + 2Gin[1:r,j + 1,:]
    end
    t = time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        for j = 0:p-3
            mul!(tmp, @view(Wj(W, N, j)[:,1:r]), @view(γ[:,j + 1,:])); RHS .-= tmp
        end
        mul!(tmp, @view(Wj(W, N, p - 2)[:,1:r]), @view(γ[:,p - 1,:])); RHS .-= tmp
        mul!(tmp, @view(Wj(W, N, p)[:,1:r]), @view(γ[:,p - 1,:])); RHS .+= tmp
        mul!(tmp, @view(Wj(W, N, p - 1)[:,1:r]), @view(γ[:,p,:])); RHS .-= tmp
        mul!(tmp, @view(Wj(W, N, p)[:,1:r]), @view(Gin[1:r,p,:])); @. RHS -= 2tmp
    end
    trhs = (time_ns() - t) * 1e-9
    copyto!(Y, RHS); t = time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS ldiv!(F, Y)
    tlu = (time_ns() - t) * 1e-9; t = time_ns()
    ynorm0 = norm(Y)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(Hphys, adjoint(@view(U[:,1:r])), Y)
        mul!(Y, @view(U[:,1:r]), Hphys, -1 + 0im, 1 + 0im)
    end
    H2 = zeros(ComplexF64, r, b)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(H2, adjoint(@view(U[:,1:r])), Y)
        mul!(Y, @view(U[:,1:r]), H2, -1 + 0im, 1 + 0im)
    end
    Hphys .+= H2
    ynorm = norm(Y); saturated = ynorm <= 1000eps(Float64) * max(ynorm0, 1.0)
    remaining = size(U, 2) - r
    if remaining > 0 && !saturated
        FY = nothing
        @blas_multi_then_1 MAX_BLAS_THREADS begin
            FY = svd(Y; full = false)
        end
        σ = FY.S; σ1 = isempty(σ) ? 0.0 : σ[1]
        bp = min(count(>(max(σ1 * 1e-12, 1e-14)), σ), remaining); rn = r + bp
        if bp > 0
            @views U[:,r + 1:rn] .= FY.U[:,1:bp]
            Cnew = Diagonal(σ[1:bp]) * FY.Vt[1:bp,:]
        else
            Cnew = zeros(ComplexF64, 0, b)
        end
    else
        bp = 0; rn = r; Cnew = zeros(ComplexF64, 0, b)
    end
    tphys = (time_ns() - t) * 1e-9
    tcache = bp > 0 ? cache_block!(W, B, U, N, p, r + 1, rn) : 0.0
    fill!(Z, 0); @views Z[1:r,1:p,:] .+= γ; @views Z[1:r,1,:] .+= Hphys
    bp > 0 && (@views Z[r + 1:rn,1,:] .+= Cnew)
    s = -1.0
    for j = 2:2:p-1
        @views Z[1:r,j + 1,:] .+= s .* Hphys
        bp > 0 && (@views Z[r + 1:rn,j + 1,:] .+= s .* Cnew)
        s = -s
    end
    return rn, tcache, tlu, trhs, tphys
end

mutable struct CORKState
    U::Matrix{ComplexF64}
    W::Matrix{ComplexF64}
    G::Array{ComplexF64,3}
    Hb::Matrix{ComplexF64}
    pendingG::Array{ComplexF64,3}
    Z::Array{ComplexF64,3}
    H1::Matrix{ComplexF64}
    H2::Matrix{ComplexF64}
    r::Int
    n::Int
    b::Int
    p::Int
    N::Int
    rmax::Int
    pending::Bool
    cache::Float64
    lu::Float64
    rhs::Float64
    phys::Float64
    orth::Float64
    nproject::Int
    nreorth::Int
end

"""
    init_cork(B::Matrix{ComplexF64}, p::Int, N::Int; b::Int=10, maxdim::Int=1600) -> CORKState

Initialize the common physical basis, coefficient cache, and first normalized
compact block-Arnoldi block.

## Arguments
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `p::Int`: Chebyshev polynomial degree.
- `N::Int`: Physical matrix dimension.
- `b::Int`: Block-Arnoldi block size.
- `maxdim::Int`: Maximum compact Krylov dimension.

## Returns
- `CORKState`: Initialized persistent block-CORK state.
"""
function init_cork(B::Matrix{ComplexF64}, p::Int, N::Int; b::Int = 10)::CORKState
    p <= N || error("Polynomial degree p=$p exceeds physical matrix dimension N=$N")
    mdim = (N ÷ b) * b
    mdim >= b || error("Matrix dimension N=$N is smaller than block size b=$b")
    rmax = N
    U = zeros(ComplexF64, N, rmax); W = zeros(ComplexF64, (p + 1) * N, rmax)
    G = zeros(ComplexF64, rmax, p, mdim + b); Hb = zeros(ComplexF64, mdim + b, mdim)
    rng = MersenneTwister(123); X = randn(rng, ComplexF64, N, p); FX = nothing
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        FX = qr(X)
    end
    @blas_multi_then_1 MAX_BLAS_THREADS @views U[:,1:p] .= FX.Q * Matrix{ComplexF64}(I, N, p)
    r = p
    Z0 = zeros(ComplexF64, rmax, p, b); @views randn!(rng, Z0[1:r,:,:])
    Z0f = zeros(ComplexF64, r * p, b); pack!(Z0f, Z0, r, p, b)
    bn, _ = compact_block_qr!(Z0f, b); bn == b || error("Initial block breakdown")
    fill!(Z0, 0); unpack!(Z0, Z0f, r, p, b); @views G[1:r,:,1:b] .= Z0[1:r,:,:]
    tc = cache_block!(W, B, U, N, p, 1, r)
    return CORKState(U, W, G, Hb, zeros(ComplexF64, rmax, p, b), zeros(ComplexF64, rmax, p, b), zeros(ComplexF64, mdim + b, b), zeros(ComplexF64, mdim + b, b), r, b, b, p, N, rmax, false, tc, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)
end

"""
    compute_tail!(S::CORKState, B::Matrix{ComplexF64}, F) -> Nothing

Compute the next block-Arnoldi residual without promoting it into the active
basis. The residual is retained so it can first be used for Ritz convergence
and subsequently promoted without recomputation.

## Arguments
- `S::CORKState`: Persistent compact block-CORK state.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `F`: LU factorization of `P(0)`.

## Returns
- `Nothing`: `S` is updated with a normalized pending Arnoldi tail.
"""
function compute_tail!(S::CORKState, B::Matrix{ComplexF64}, F)::Nothing
    S.pending && return nothing
    n = S.n; b = S.b; p = S.p; r = S.r; cols = n - b + 1:n; Gin = @view S.G[:,:,cols]
    fill!(S.Z, 0)
    rn, tc, tl, tr, tp = block_apply!(S.Z, S.U, S.W, B, F, Gin, S.N, p, r, b)
    S.r = rn; S.cache += tc; S.lu += tl; S.rhs += tr; S.phys += tp
    fill!(S.H1, 0); fill!(S.H2, 0); t = time_ns()
    reorth = compact_project!(S.Z, S.G, rn, n, p, S.H1, S.H2)
    S.nproject += 1; S.nreorth += reorth
    Zf = zeros(ComplexF64, rn * p, b); pack!(Zf, S.Z, rn, p, b)
    bn, Rb = compact_block_qr!(Zf, b); bn == b || error("Block breakdown at n=$n: $bn/$b")
    fill!(S.Z, 0); unpack!(S.Z, Zf, rn, p, b)
    @views S.Hb[1:n,cols] .= S.H1[1:n,:]
    @views S.Hb[n + 1:n + b,cols] .= Rb
    S.orth += (time_ns() - t) * 1e-9
    copyto!(S.pendingG, S.Z)
    S.pending = true; S.napply += 1
    return nothing
end

"""
    promote_tail!(S::CORKState) -> Nothing

Promote the normalized pending residual into the active compact Arnoldi basis.

## Arguments
- `S::CORKState`: Persistent state containing a pending residual block.

## Returns
- `Nothing`: The active dimension increases by `S.b`.
"""
function promote_tail!(S::CORKState)::Nothing
    S.pending || error("No pending tail")
    n = S.n; b = S.b; mdim = size(S.Hb, 2)
    n + b <= mdim || error("Reached full CORK Krylov dimension m=$mdim for matrix size N=$(S.N)")
    @views S.G[:,:,n + 1:n + b] .= 0
    @views S.G[1:S.r,:,n + 1:n + b] .= S.pendingG[1:S.r,:,:]
    S.n += b; S.pending = false
    return nothing
end

"""
    extend!(S::CORKState, B::Matrix{ComplexF64}, F, m::Int) -> Nothing

Extend the persistent factorization to compact Krylov dimension `m`, retaining
all existing Arnoldi vectors and leaving a fresh residual tail for Ritz tests.

## Arguments
- `S::CORKState`: Persistent CORK state.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `F`: LU factorization of `P(0)`.
- `m::Int`: Requested compact Krylov dimension.

## Returns
- `Nothing`: `S` is extended in place.
"""
function extend!(S::CORKState, B::Matrix{ComplexF64}, F, m::Int)::Nothing
    mdim = size(S.Hb, 2)
    m % S.b == 0 || error("m must be divisible by block size")
    m <= mdim || error("Requested CORK dimension m=$m exceeds available dimension $mdim")
    m >= S.n || error("Cannot shrink CORK basis")
    while S.n < m
        S.pending || compute_tail!(S, B, F)
        promote_tail!(S)
    end
    compute_tail!(S, B, F)
    return nothing
end

################################################################################
# RITZ ROOTS
################################################################################

"""
    ritz_roots(S::CORKState, k0::Float64, Δpoly::Float64, m::Int; edge_tol::Float64=1e-8, imag_search_tol::Float64=1e-4) -> Tuple{Vector{Tuple{Float64,Float64,Float64}},Matrix{ComplexF64}}

Extract individual Ritz roots and their projected right eigenvectors from the
projected inverse linearization. Every projected eigenvalue is mapped
independently by `μ→t=1/μ→k=k₀+Δpoly*t`; nearby Ritz values are never merged,
averaged, or interpreted as physical multiplicities.

For a projected right eigenpair `Hv=μv`, the block-Arnoldi residual estimate is

    ρ=‖Htail*v‖/max(1,|μ|).

The returned Ritz-vector columns remain in exactly the same order as the
returned roots.

## Arguments
- `S::CORKState`: Persistent CORK state at dimension `m`.
- `k0::Float64`: Physical expansion center.
- `Δpoly::Float64`: Guarded polynomial half-width.
- `m::Int`: Active compact Krylov dimension.
- `edge_tol::Float64`: Tolerance on the normalized guarded interval.
- `imag_search_tol::Float64`: Loose imaginary strip used during discovery.

## Returns
- `Vector{Tuple{Float64,Float64,Float64}}`: Individual Ritz roots as `(Re(k),Im(k),ρ)`.
- `Matrix{ComplexF64}`: Corresponding projected right Ritz vectors.
"""
function ritz_roots(S::CORKState, k0::Float64, Δpoly::Float64, m::Int; edge_tol::Float64=1e-8, imag_search_tol::Float64=1e-4)::Tuple{Vector{Tuple{Float64,Float64,Float64}},Matrix{ComplexF64}}
    S.n == m || error("State dimension $(S.n) != requested m=$m")
    S.pending || error("Arnoldi tail missing")
    H = Matrix(@view S.Hb[1:m,1:m]); E = nothing
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        E = eigen(H)
    end
    tail = @view S.Hb[m + 1:m + S.b,1:m]; roots = Tuple{Float64,Float64,Float64}[]; inds = Int[]
    for j = eachindex(E.values)
        μ = E.values[j]; abs(μ) > 1e-14 || continue
        t = inv(μ); k = ComplexF64(k0 + Δpoly*t)
        -1 - edge_tol <= real(t) <= 1 + edge_tol || continue
        abs(imag(k)) <= imag_search_tol || continue
        ρ = norm(tail*@view(E.vectors[:,j]))/max(1.0,abs(μ))
        push!(roots,(real(k),imag(k),ρ)); push!(inds,j)
    end
    perm = sortperm(roots,by=x -> (x[1],x[2]))
    return roots[perm],Matrix{ComplexF64}(E.vectors[:,inds[perm]])
end

"""
    reconstruct_cork_eigenvectors(S::CORKState, V::Matrix{ComplexF64}, m::Int) -> Matrix{ComplexF64}

Reconstruct Fredholm smallest singular vectors u : A(v)u(v) = 0 from projected CORK Ritz vectors.

## Arguments
- `S::CORKState`: Final persistent CORK state.
- `V::Matrix{ComplexF64}`: Projected Ritz vectors, one column per accepted root.
- `m::Int`: Final compact Krylov dimension.

## Returns
- `Matrix{ComplexF64}`: Normalized Fredholm smallest singular vectors, with column `j` corresponding to Ritz-vector column `j`.
"""
function reconstruct_cork_eigenvectors(S::CORKState, V::Matrix{ComplexF64}, m::Int)::Matrix{ComplexF64}
    size(V,1) == m || throw(DimensionMismatch("Ritz vectors have $(size(V,1)) rows but m=$m"))
    q = size(V,2); q == 0 && return zeros(ComplexF64,S.N,0)
    G0 = @view S.G[1:S.r,1,1:m]; U = @view S.U[:,1:S.r]
    C = zeros(ComplexF64,S.r,q); Ψ = zeros(ComplexF64,S.N,q)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(C,G0,V); mul!(Ψ,U,C)
    end
    @inbounds for j = 1:q
        nrm = norm(@view Ψ[:,j]); nrm > 0 || error("Zero reconstructed CORK singular vector for column $j")
        @views Ψ[:,j] ./= nrm
    end
    return Ψ
end

################################################################################
# ADAPTIVE CONVERGENCE
################################################################################

"""
    adaptive_cork(B::Matrix{ComplexF64}, F, p::Int, N::Int; k0::Float64, Δ::Float64, Δpoly::Float64, b::Int=10, mstart::Int=200, mstep::Int=100, maxdim::Int=1600, stable_checks::Int=2, imag_tol::Float64=1e-7, edge_tol::Float64=1e-8, res_tol::Float64=1e-8, stable_tol::Float64=1e-8, imag_search_tol::Float64=1e-4, eigenvectors::Bool=false, verbose::Bool=false)

Grow one persistent CORK factorization until the individually retained Ritz
spectrum in `[k₀-Δ,k₀+Δ]` is stable for `stable_checks` consecutive Krylov
dimensions and its leftmost and rightmost discovered roots satisfy the strict
physical tolerances.

Every projected Ritz eigenvalue is treated independently. Nearby Ritz values
are never merged, averaged, deduplicated, or expanded according to an inferred
multiplicity. If `eigenvectors=true`, physical Fredholm eigenvectors are
reconstructed once from the final accepted projected Ritz vectors after
convergence.

## Arguments
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `F`: LU factorization of `P(0)`.
- `p::Int`: Chebyshev polynomial degree.
- `N::Int`: Physical matrix dimension.
- `k0::Float64`: Requested interval center.
- `Δ::Float64`: Requested interval half-width.
- `Δpoly::Float64`: Guarded polynomial half-width.
- `b::Int`: Block-Arnoldi block size.
- `mstart::Int`: Initial compact Krylov dimension.
- `mstep::Int`: Krylov-dimension increment.
- `stable_checks::Int`: Required consecutive stable spectra.
- `imag_tol::Float64`: Strict physical imaginary-part tolerance.
- `edge_tol::Float64`: Normalized guarded-interval tolerance.
- `res_tol::Float64`: Ritz residual tolerance.
- `stable_tol::Float64`: Maximum accepted root drift.
- `imag_search_tol::Float64`: Loose imaginary discovery strip.
- `eigenvectors::Bool`: Whether to reconstruct final physical Fredholm eigenvectors.
- `verbose::Bool`: Whether to enable verbose CORK diagnostics.

## Returns
- `Tuple`: Final state, accepted individual spectrum, optional physical
  eigenvector matrix, all requested Ritz roots, all guarded Ritz roots,
  requested edge roots, edge acceptance flags, and final Krylov dimension.
"""
function adaptive_cork(B::Matrix{ComplexF64}, F, p::Int, N::Int; k0::Float64, Δ::Float64, Δpoly::Float64, b::Int = 10, mstart::Int = 200, mstep::Int = 100, stable_checks::Int = 2, imag_tol::Float64 = 1e-7, edge_tol::Float64 = 1e-8, res_tol::Float64 = 1e-8, stable_tol::Float64 = 1e-8, imag_search_tol::Float64 = 1e-4, eigenvectors::Bool = false, verbose::Bool = false)
    mstart % b == 0 || error("mstart must be divisible by block size")
    mstep % b == 0 || error("mstep must be divisible by block size")
    S = init_cork(B, p, N; b = b); mdim = size(S.Hb, 2)
    m = min(mstart, mdim)
    kmin = k0 - Δ; kmax = k0 + Δ; prev = Tuple{Float64,Float64,Float64}[]; nstable = 0; t0 = time_ns()
    verbose && @printf("Interval k: [%.3f, %.3f]\n", kmin, kmax)
    while true
        extend!(S, B, F, m)
        allroots, Vall = ritz_roots(S, k0, Δpoly, m; edge_tol = edge_tol, imag_search_tol = imag_search_tol)
        requested_inds = findall(x -> kmin <= x[1] <= kmax, allroots)
        requested = allroots[requested_inds]; Vrequested = @view Vall[:,requested_inds]
        phys_inds = findall(x -> abs(x[2]) <= imag_tol && x[3] <= res_tol, requested)
        phys = requested[phys_inds]; Vphys = @view Vrequested[:,phys_inds]; ks = copy(phys)
        edges = (isempty(requested) ? nothing : first(requested), isempty(requested) ? nothing : last(requested))
        edge_good = (edges[1] !== nothing && abs(edges[1][2]) <= imag_tol && edges[1][3] <= res_tol, edges[2] !== nothing && abs(edges[2][2]) <= imag_tol && edges[2][3] <= res_tol)
        edge_ok = all(edge_good)
        drift = length(prev) == length(ks) && !isempty(ks) ? maximum(abs(complex(ks[i][1], ks[i][2]) - complex(prev[i][1], prev[i][2])) for i = eachindex(ks)) : Inf
        nstable = isfinite(drift) && drift <= stable_tol ? nstable + 1 : 0
        maxρ = isempty(phys) ? Inf : maximum(x[3] for x in phys)
        verbose && @printf("m=%4d/%4d rank=%4d ritz=%4d conv=%4d states=%4d maxρ=%9.2e drift=%9.2e stable=%d/%d edges=%s applies=%4d reorth=%3d/%3d time=%7.3f\n", m, mdim, S.r, length(requested), length(phys), length(ks), maxρ, drift, nstable, stable_checks, edge_ok ? "PASS" : "FAIL", S.napply, S.nreorth, S.nproject, (time_ns() - t0) * 1e-9)
        if nstable >= stable_checks && edge_ok
            Ψ = eigenvectors ? reconstruct_cork_eigenvectors(S, Matrix{ComplexF64}(Vphys), m) : nothing
            return S, ks, Ψ, requested, allroots, edges, edge_good, m
        end
        m == mdim && error("Reached full CORK Krylov dimension m=$mdim for matrix size N=$N without spectrum stability and converged requested edge roots.")
        prev = ks; m = min(m + mstep, mdim)
    end
end

################################################################################
# SOLVER AND PUBLIC API
################################################################################

struct CORKSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K
    p::Int
    guard::T
    nlevels::Int
    Rmax::T
    b::Int
    mstart::Int
    mstep::Int
    stable_checks::Int
    imag_tol::T
    edge_tol::T
    res_tol::T
    stable_tol::T
    imag_search_tol::T
    eigenvectors::Bool
    validate::Bool
    verbose::Bool
    taylor_tol::T
end

"""
    CORKSolver(kernel::K; p::Int=20, guard::Real=0.05, nlevels::Int=200, Rmax::Real=0.9, b::Int=10, mstart::Int=600, mstep::Int=10*b, maxdim::Int=200*b, stable_checks::Int=1, imag_tol::Real=1e-8, edge_tol::Real=1e-8, res_tol::Real=1e-10, stable_tol::Real=1e-9, imag_search_tol::Real=1e-4, eigenvectors::Bool=false, validate::Bool=false, verbose::Bool=false, taylor_tol::Real=1e-10) where {K<:SweepBIMSolver} -> CORKSolver

Construct a CORK solver for a BIM Fredholm nonlinear eigenvalue problem.

For full-spectrum computations, `nlevels` is the target number of physical
levels in each requested CORK window and `Rmax` is the maximum requested
half-width `Δ`. The Chebyshev polynomial is constructed on

    Δpoly = (1 + guard)Δ,

so that the polynomial approximation and Ritz search extend beyond the
requested spectral window.

## Arguments
- `kernel::K`: BIM kernel used to construct the Fredholm operator.

## Keyword Arguments
- `p::Int=16`: Chebyshev polynomial degree.
- `guard::Real=0.05`: Relative enlargement of the requested interval used for polynomial construction and Ritz discovery.
- `nlevels::Int=200`: Target number of physical levels per full-spectrum window.
- `Rmax::Real=0.9`: Maximum requested CORK half-width `Δ`.
- `b::Int=10`: CORK block size.
- `mstart::Int=600`: Initial Krylov dimension.
- `mstep::Int=10*b`: Krylov-dimension increment between convergence checks.
- `maxdim::Int=200*b`: Maximum Krylov dimension.
- `stable_checks::Int=1`: Number of consecutive stable Ritz checks required.
- `imag_tol::Real=1e-8`: Final imaginary-part tolerance in physical `k` units.
- `edge_tol::Real=1e-8`: Normalized guarded-interval Ritz extraction tolerance.
- `res_tol::Real=1e-10`: Ritz residual tolerance.
- `stable_tol::Real=1e-9`: Spectrum-stability tolerance.
- `imag_search_tol::Real=1e-4`: Loose imaginary discovery strip used during Ritz extraction.
- `eigenvectors::Bool=false`: Whether to reconstruct physical eigenvectors. Warning: enabling will may increase memory usage and computational cost.
- `validate::Bool=false`: Whether to validate the Chebyshev approximation during individual CORK solves.
- `verbose::Bool=false`: Whether to print detailed CORK convergence diagnostics.
- `taylor_tol::Real=1e-10`: Relative tolerance used for Chebyshev-polynomial validation.

## Returns
- `CORKSolver`: Configured CORK solver.
"""
function CORKSolver(kernel::K; p::Int=16, guard::Real=0.05, nlevels::Int=200, Rmax::Real=0.8, b::Int=10, mstart::Int=30*b, mstep::Int=10*b, stable_checks::Int=1, imag_tol::Real=1e-8, edge_tol::Real=1e-8, res_tol::Real=1e-10, stable_tol::Real=1e-9, imag_search_tol::Real=1e-4, eigenvectors::Bool=false, validate::Bool=false, verbose::Bool=false, taylor_tol::Real=1e-10) where {K<:SweepBIMSolver}
    T = _bim_numeric_type(kernel)
    T === Float64 || throw(ArgumentError("CORKSolver currently requires a Float64 BIM kernel"))
    p >= 2 || throw(ArgumentError("p must be at least 2; received p=$p"))
    mstart % b == 0 || throw(ArgumentError("mstart must be divisible by b"))
    mstep % b == 0 || throw(ArgumentError("mstep must be divisible by b"))
    return CORKSolver{T,K}(kernel,p,T(guard),nlevels,T(Rmax),b,mstart,mstep,stable_checks,T(imag_tol),T(edge_tol),T(res_tol),T(stable_tol),T(imag_search_tol),eigenvectors,validate,verbose,T(taylor_tol))
end

_bim_numeric_type(::CORKSolver{T}) where {T} = T

# Delegate boundary discretization to the wrapped BIM solver.
evaluate_points(solver::CORKSolver, billiard::Bi, k) where {Bi<:AbsBilliard} = evaluate_points(solver.kernel, billiard, k)

# Compact kernel name used only by debug diagnostics.
@inline _cork_kernel_name(solver)::String = solver isa DoubleLayerPotentialSolver ? "DLP" : solver isa CombinedFieldIntegralEquationSolver ? "CFIE" : solver isa CompositeBIMSolver ? "CompositeBIM" : string(typeof(solver))

"""
    _cork_solve_core(solver::CORKSolver, pts, k0, dk; multithreaded::Bool=true, eigenvectors::Bool=false)

Execute the complete CORK pipeline on `[k₀-dk/2,k₀+dk/2]`. If
`eigenvectors=true`, reconstruct the physical Fredholm smallest singular vectors from the
final accepted projected Ritz vectors without additional Fredholm SVDs.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `pts`: Boundary discretization at the expansion center.
- `k0`: Requested interval center.
- `dk`: Full requested spectral width.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Tuple`: Polynomial, final state, accepted spectrum, optional physical
  eigenvector matrix, requested Ritz roots, all guarded Ritz roots, requested
  edge roots, edge acceptance flags, and final Krylov dimension.
"""
function _cork_solve_core(solver::CORKSolver, pts, k0, dk; multithreaded::Bool = true)
    k0f = Float64(k0); Δ = Float64(dk) / 2
    Δ > 0 || throw(ArgumentError("dk must be positive; received dk=$dk"))
    Δpoly = Δ * (1 + solver.guard)
    k0f > Δpoly || throw(ArgumentError("CORK polynomial interval reaches k=0"))
    kmin = k0f - Δ; kmax = k0f + Δ; kpmin = k0f - Δpoly; kpmax = k0f + Δpoly
    P = nothing; tbuild = 0.0
    @timeit_debug "CORK polynomial construction" begin
        P, tbuild = build_cork_polynomial(solver.kernel, pts, k0f, Δpoly, solver.p; multithreaded = multithreaded)
    end
    if solver.verbose
        @printf("CORK matrix size     = %d\nB build              = %.6f s\nB memory             = %.3f MiB\n", P.N, tbuild, Base.summarysize(P.B) / 2^20)
        solver.kernel.symmetry !== nothing && @printf("dimension reduction  = %.3fx\n", length(pts.xy) / P.N)
    end
    solver.validate && validate_polynomial!(solver.kernel, pts, P, solver.taylor_tol; multithreaded = multithreaded)
    t = time_ns(); A0 = get_A0(P.B, P.N, solver.p); ta0 = (time_ns() - t) * 1e-9
    F = nothing; t = time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        F = lu(A0)
    end
    tfact = (time_ns() - t) * 1e-9
    solver.verbose && @printf("\nP(0) assembly        = %.6f s\nLU                   = %.6f s\n\n", ta0, tfact)
    S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal = adaptive_cork(P.B, F, solver.p, P.N; k0 = k0f, Δ = Δ, Δpoly = Δpoly, b = solver.b, mstart = solver.mstart, mstep = solver.mstep, stable_checks = solver.stable_checks, imag_tol = solver.imag_tol, edge_tol = solver.edge_tol, res_tol = solver.res_tol, stable_tol = solver.stable_tol, imag_search_tol = solver.imag_search_tol, eigenvectors = solver.eigenvectors, verbose = solver.verbose)
    return P, S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal
end

"""
    solve(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)

Compute all accepted roots in `[k₀-dk/2,k₀+dk/2]`, repeating degenerate
roots according to their inferred physical multiplicity.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `pts::BoundaryPoints`: Boundary discretization at the expansion center.
- `k0`: Requested interval center.
- `dk`: Full requested spectral width.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Tuple{Vector{ComplexF64},Vector{Float64}}`: wavenumbers and their CORK residual estimates.
"""
function solve(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    P, S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal = _cork_solve_core(solver, pts, k0, dk; multithreaded = multithreaded)
    λ = ComplexF64[complex(x[1], x[2]) for x in ks]; ts = Float64[x[3] for x in ks]
    return λ, ts
end

"""
    solve_vectors(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)

Compute all accepted roots and their corresponding eigenvectors (smallest singular vectors of the Fredholm matrix) in `[k₀-dk/2,k₀+dk/2]`.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `pts::BoundaryPoints`: Boundary discretization at the expansion center.
- `k0`: Requested interval center.
- `dk`: Full requested spectral width.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Tuple{Vector{ComplexF64},Vector{Float64},Matrix{ComplexF64}}`: wavenumbers, their CORK residual estimates, and the corresponding eigenvectors.
"""
function solve_vectors(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    P, S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal = _cork_solve_core(solver, pts, k0, dk; multithreaded = multithreaded)
    λ = ComplexF64[complex(x[1], x[2]) for x in ks]; ts = Float64[x[3] for x in ks]
    return λ, ts, Ψ
end

"""
    solve_wavenumber(solver::CORKSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}

Find the accepted CORK eigenvalue nearest `k` in `[k-dk/2,k+dk/2]`.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `billiard::Bi`: Billiard geometry.
- `k`: Target and polynomial expansion center.
- `dk`: Full requested spectral width.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Tuple{ComplexF64,Float64}`: Nearest accepted wavenumber and its residual.
"""
function solve_wavenumber(solver::CORKSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k); ks, ts = solve(solver, pts, k, dk; multithreaded = multithreaded)
    isempty(ks) && error("CORKSolver found no eigenvalue candidates in [$(k - dk / 2),$(k + dk / 2)]")
    idx = findmin(abs.(ks .- k))[2]
    return ks[idx], ts[idx]
end

"""
    solve_spectrum(solver::CORKSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}

Compute the complete accepted multiplicity-expanded spectrum in
`[k-dk/2,k+dk/2]`, constructing boundary points at the expansion center `k`.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `billiard::Bi`: Billiard geometry.
- `k`: Requested interval and polynomial expansion center.
- `dk`: Full requested spectral width.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Tuple{Vector{ComplexF64},Vector{Float64}}`: Multiplicity-expanded
  wavenumbers and their CORK residual estimates.
"""
function solve_spectrum(solver::CORKSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k)
    return solve(solver, pts, k, dk; multithreaded = multithreaded)
end