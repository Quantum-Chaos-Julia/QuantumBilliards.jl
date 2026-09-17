################################################################################
# CHEBYSHEV-CORK BOUNDARY-INTEGRAL EIGENSOLVER
#
# This file implements a compact CORK eigensolver for BIM nonlinear
# eigenvalue problems
#
#                              A(k)u = 0,
#
# where A(k) is the Fredholm matrix of a DLP, CFIE, or composite BIM
# discretization. For a requested window
#
#                  Δ = dk/2,    k ∈ [k₀-Δ,k₀+Δ],
#
# `recurrences.jl` constructs a Chebyshev approximation on the guarded
# half-width Δₚ=(1+guard)Δ,
#
#              P(t)=Σⱼ₌₀ᵖ BⱼTⱼ(t) ≈ A(k₀+Δₚt),
#              t=(k-k₀)/Δₚ.
#
# CORK solves P(t)u=0 through a compact Chebyshev linearization. Instead of
# storing p independent N-dimensional linearization blocks, all physical
# blocks are represented in a common basis U,
#
#                              zⱼ = Ugⱼ,
#
# with cached coefficient actions Wⱼ=BⱼU. At the fixed shift t=0,
#
#                         P(0)=B₀-B₂+B₄-⋯,
#
# so one LU factorization of P(0) is reused throughout the block-Arnoldi
# iteration. Projected inverse eigenvalues are mapped back to wavenumbers by
#
#                         μ → t=1/μ → k=k₀+Δₚt.
#
# Nearby Ritz values are clustered, physical multiplicities are determined
# from the block-Arnoldi residual, and convergence requires both a stable
# requested spectrum and converged roots bracketing both requested edges.
#
# ==============================================================================
# API AND FUNCTION-CALL CHAIN
# ==============================================================================
#
# The public entry points are
#
#     solve(solver,pts,k0,dk)
#     solve_spectrum(solver,billiard,k0,dk)
#     solve_wavenumber(solver,billiard,k0,dk)
#
# where `dk` always denotes the FULL requested spectral width.
#
# Main spectrum call:
#
# solve_spectrum(solver,billiard,k0,dk)       # Solve a complete spectral window
# │
# ├─ evaluate_points(solver,billiard,k0)      # Build the boundary discretization
# │    └─ evaluate_points(solver.kernel,...)  # Delegate it to the wrapped BIM solver
# │
# └─ solve(solver,pts,k0,dk)                  # Solve using existing boundary points
#      │
#      └─ _cork_solve_core(...)               # Run the complete polynomial+CORK pipeline
#           │
#           ├─ build_cork_polynomial(...)     # Construct P(t) from analytic BIM recurrences
#           │    │
#           │    ├─ _taylor_cache(...)        # Precompute geometry/Kress data independent of k
#           │    │
#           │    ├─ build_B_full(...)         # Assemble Bⱼ on the full boundary
#           │    │    └─ _fredholm_taylor!    # Form Taylor coefficients of I-K(k)
#           │    │         └─ _kernel_taylor! # Evaluate analytic kernel derivatives
#           │    │
#           │    └─ build_B_reduced(...)      # Assemble symmetry-reduced Bⱼ
#           │         └─ _fredholm_taylor!    # Form reduced Fredholm Taylor coefficients
#           │              └─ _kernel_taylor! # Evaluate analytic kernel derivatives
#           │
#           │       # The functions above live in `recurrences.jl` and return
#           │       # CORKPolynomial(B,k0,Δpoly,p,N).
#           │
#           ├─ validate_polynomial(...)       # Compare P(t) with direct BIM matrices
#           │                                  # when `validate=true`
#           │
#           ├─ get_A0(B,N,p)                  # Evaluate P(0)=B₀-B₂+B₄-⋯
#           │
#           ├─ lu(P(0))                       # Factorize the fixed physical shift once
#           │
#           └─ adaptive_cork(...)             # Grow CORK until roots are stable and complete
#                │
#                ├─ init_cork(...)            # Allocate and initialize persistent CORK state
#                │    │
#                │    ├─ initialize U         # Build the initial common physical basis
#                │    ├─ initialize G         # Build the first compact Arnoldi block
#                │    └─ cache_block!(...)    # Cache Wⱼ=BⱼU for the initial basis
#                │
#                └─ for increasing m          # Extend one persistent Krylov factorization
#                     │
#                     ├─ extend!(...)          # Grow the active compact basis to dimension m
#                     │    │
#                     │    ├─ compute_tail!    # Compute and retain the next Arnoldi residual
#                     │    │    │
#                     │    │    ├─ block_apply!(...)
#                     │    │    │    │         # Apply the inverse Chebyshev linearization
#                     │    │    │    ├─ recurrence
#                     │    │    │    │         # Propagate compact Chebyshev coordinates
#                     │    │    │    ├─ P(0)\rhs
#                     │    │    │    │         # Solve the physical inverse-shift equation
#                     │    │    │    ├─ physical SVD
#                     │    │    │    │         # Detect new independent physical directions
#                     │    │    │    └─ cache_block!
#                     │    │    │              # Cache BⱼUnew for those new directions
#                     │    │    │
#                     │    │    ├─ pack!       # Flatten (physical,degree,block) coordinates
#                     │    │    ├─ compact_project_cgs2!
#                     │    │    │              # Reorthogonalize against compact Arnoldi basis
#                     │    │    ├─ compact_block_qr!
#                     │    │    │              # Normalize the residual and detect breakdown
#                     │    │    └─ pending tail
#                     │    │                   # Keep residual available for Ritz testing
#                     │    │
#                     │    └─ promote_tail!    # Turn the previous residual into an active block
#                     │
#                     ├─ ritz_clusters(...)    # Extract converged root locations/multiplicities
#                     │    │
#                     │    ├─ eigen(Hm)        # Solve the projected inverse eigenproblem
#                     │    ├─ μ→1/μ→k          # Map projected eigenvalues to wavenumbers
#                     │    ├─ _cluster_indices # Group nearby Ritz values into one root
#                     │    └─ residual SVD     # Determine converged physical multiplicity
#                     │
#                     ├─ physical_clusters(...)# Apply strict window, Im(k), and residual filters
#                     │
#                     ├─ expand_clusters(...)  # Repeat each root according to multiplicity
#                     │
#                     ├─ edge_completeness(...)# Check nearest roots on both sides of both edges
#                     │
#                     └─ spectrum_distance(...)# Measure spectral drift from previous m
#
# `adaptive_cork` stops only when:
#
#   1. the multiplicity-expanded requested spectrum is stable for
#      `stable_checks` consecutive Krylov dimensions, and
#   2. left-out, left-in, right-in, and right-out roots all satisfy the strict
#      physical tolerances.
#
# Nearest-root call:
#
# solve_wavenumber(solver,billiard,k0,dk)      # Find the root nearest k₀
# │
# ├─ evaluate_points(solver,billiard,k0)      # Build boundary points at k₀
# ├─ solve(solver,pts,k0,dk)                  # Run the same complete CORK chain
# └─ findmin(abs.(ks.-k0))                    # Select the nearest accepted root
# ==============================================================================

# Views of the j-th Chebyshev coefficient and cached coefficient action.
@inline Bj(B::Matrix{ComplexF64}, N::Int, j::Int) = @view B[j*N+1:(j+1)*N,:]
@inline Wj(W::Matrix{ComplexF64}, N::Int, j::Int) = @view W[j*N+1:(j+1)*N,:]

"""
    evaluate_cork_polynomial(P::CORKPolynomial, t::Real) -> Matrix{ComplexF64}

Evaluate the matrix Chebyshev polynomial

    P(t)=Σⱼ₌₀ᵖ BⱼTⱼ(t)

at the normalized coordinate `t` using the three-term recurrence
`T₀(t)=1`, `T₁(t)=t`, and `Tⱼ₊₁(t)=2tTⱼ(t)-Tⱼ₋₁(t)`.

## Arguments
- `P::CORKPolynomial`: Chebyshev polynomial with vertically stacked
  coefficients `[B₀;...;Bₚ]`.
- `t::Real`: Normalized Chebyshev coordinate.

## Returns
- `Matrix{ComplexF64}`: Dense matrix `P(t)`.
"""
function evaluate_cork_polynomial(P::CORKPolynomial, t::Real)::Matrix{ComplexF64}
    τ=Float64(t); A=copy(Bj(P.B,P.N,0))
    P.p==0 && return A

    Tm=1.0; Tn=τ
    axpy!(Tn,Bj(P.B,P.N,1),A)

    @inbounds for j=1:P.p-1
        Tp=2τ*Tn-Tm
        axpy!(Tp,Bj(P.B,P.N,j+1),A)
        Tm,Tn=Tn,Tp
    end
    return A
end

"""
    validate_polynomial(solver::SweepBIMSolver, pts, P::CORKPolynomial; nsample::Int=5, multithreaded::Bool=true) -> Float64

Validate the Chebyshev approximation by comparing `P(t)` with directly
constructed BIM Fredholm matrices at sample points across the polynomial
interval. The reported error is

    ‖P(t)-A(k₀+Δt)‖/‖A(k₀+Δt)‖.

For symmetry-reduced problems the direct matrix is constructed using the same
symmetry reduction as the polynomial.

## Arguments
- `solver::SweepBIMSolver`: BIM solver defining the direct Fredholm matrix.
- `pts`: Boundary discretization used to construct the polynomial.
- `P::CORKPolynomial`: Chebyshev polynomial to validate.
- `nsample::Int`: Number of sample points in `[-1,1]`.
- `multithreaded::Bool`: Whether direct matrix construction uses threading.

## Returns
- `Float64`: Maximum relative matrix error over all sample points.
"""
function validate_polynomial(solver::SweepBIMSolver, pts, P::CORKPolynomial; nsample::Int=5, multithreaded::Bool=true)::Float64
    ts=collect(range(-1.0,1.0,length=nsample)); worst=0.0
    println("\nPOLYNOMIAL VALIDATION\n","-"^104)
    for t in ts
        k=P.k0+P.Δ*t; Ap=evaluate_cork_polynomial(P,t)
        Ad=construct_matrices(solver,pts,k; multithreaded=multithreaded)
        err=norm(Ap-Ad)/norm(Ad); worst=max(worst,err)
        @printf("t=%+8.4f  k=%14.8f  relative error=%.3e\n",t,k,err)
    end
    @printf("worst relative error = %.3e\n",worst)
    return worst
end

################################################################################
# CHEBYSHEV LINEARIZATION
################################################################################

"""
    get_A0(B::Matrix{ComplexF64}, N::Int, p::Int) -> Matrix{ComplexF64}

Evaluate `P(t)=Σⱼ₌₀ᵖBⱼTⱼ(t)` at the fixed CORK shift `t=0`. Since
`T₂q(0)=(-1)^q` and `T₂q₊₁(0)=0`, `P(0)=B₀-B₂+B₄-B₆+⋯`. This physical
matrix is factorized once and reused by every inverse-linearization action.

## Arguments
- `B::Matrix{ComplexF64}`: Vertically stacked coefficients `[B₀;...;Bₚ]`.
- `N::Int`: Dimension of each coefficient matrix `Bⱼ`.
- `p::Int`: Chebyshev polynomial degree.

## Returns
- `Matrix{ComplexF64}`: Dense physical shift matrix `P(0)`.
"""
function get_A0(B::Matrix{ComplexF64}, N::Int, p::Int)::Matrix{ComplexF64}
    A = zeros(ComplexF64,N,N)
    @inbounds for j = 0:2:p
        axpy!(isodd(j÷2) ? -1.0 : 1.0,Bj(B,N,j),A)
    end
    return A
end

"""
    cache_block!(W::Matrix{ComplexF64}, B::Matrix{ComplexF64}, U::Matrix{ComplexF64}, a::Int, b::Int) -> Float64

Cache all coefficient actions on `U[:,a:b]`. Since `B=[B₀;...;Bₚ]` is
stored vertically, `W[:,a:b]=B*U[:,a:b]` simultaneously computes
`Wⱼ[:,a:b]=BⱼU[:,a:b]` for every `j=0,...,p`.

## Arguments
- `W::Matrix{ComplexF64}`: Vertically stacked coefficient-action cache.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `U::Matrix{ComplexF64}`: Common physical CORK basis.
- `a::Int`: First physical basis column to cache.
- `b::Int`: Last physical basis column to cache.

## Returns
- `Float64`: Elapsed tall-matrix multiplication time in seconds.
"""
@inline function cache_block!(W::Matrix{ComplexF64}, B::Matrix{ComplexF64}, U::Matrix{ComplexF64}, a::Int, b::Int)::Float64
    a > b && return 0.0
    t = time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(@view(W[:,a:b]),B,@view(U[:,a:b]))
    return (time_ns()-t)*1e-9
end

"""
    compact_project_cgs2!(Z::Matrix{ComplexF64}, G::Matrix{ComplexF64}, n::Int, H1::Matrix{ComplexF64}, H2::Matrix{ComplexF64}) -> Nothing

Orthogonalize `Z` against `Q=G[:,1:n]` by two-pass classical Gram-Schmidt:
`H₁=Q*Z`, `Z←Z-QH₁`, followed by `H₂=Q*Z`, `Z←Z-QH₂`. On return `H1`
contains the accumulated projection coefficients `H₁+H₂`.

## Arguments
- `Z::Matrix{ComplexF64}`: Compact block, overwritten by its orthogonal part.
- `G::Matrix{ComplexF64}`: Flattened compact Arnoldi basis.
- `n::Int`: Number of active compact Arnoldi vectors.
- `H1::Matrix{ComplexF64}`: Workspace and accumulated projection coefficients.
- `H2::Matrix{ComplexF64}`: Workspace for the second CGS pass.

## Returns
- `Nothing`: `Z` and `H1` are modified in place.
"""
function compact_project_cgs2!(Z::Matrix{ComplexF64}, G::Matrix{ComplexF64}, n::Int, H1::Matrix{ComplexF64}, H2::Matrix{ComplexF64})::Nothing
    Q = @view G[:,1:n]; A = @view H1[1:n,:]; C = @view H2[1:n,:]
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(A,adjoint(Q),Z); mul!(Z,Q,A,-1+0im,1+0im)
        mul!(C,adjoint(Q),Z); mul!(Z,Q,C,-1+0im,1+0im)
    end
    A .+= C
    return nothing
end

"""
    compact_block_qr!(Z::Matrix{ComplexF64}, b::Int) -> Tuple{Int,Matrix{ComplexF64}}

Rank-test and normalize one compact block by QR factorization. The numerical
rank is estimated from `diag(R)` with
`tol=max(10⁻¹² maxᵢ|Rᵢᵢ|,10⁻¹⁴)`. For full rank, `Z` is replaced by the
first `b` orthonormal columns of `Q`; otherwise the detected rank signals block
breakdown.

## Arguments
- `Z::Matrix{ComplexF64}`: Compact block to rank-test and normalize.
- `b::Int`: Required block-Arnoldi rank.

## Returns
- `Tuple{Int,Matrix{ComplexF64}}`: Detected rank and leading `b×b` factor `R`.
"""
function compact_block_qr!(Z::Matrix{ComplexF64}, b::Int)::Tuple{Int,Matrix{ComplexF64}}
    @blas_multi_then_1 MAX_BLAS_THREADS F = qr(Z)
    R = Matrix(F.R)[1:b,1:b]; d = abs.(diag(R))
    bn = count(>(max(maximum(d)*1e-12,1e-14)),d)
    bn == b || return bn,R
    @blas_multi_then_1 MAX_BLAS_THREADS copyto!(Z,F.Q*Matrix{ComplexF64}(I,size(Z,1),b))
    return b,R
end

"""
    pack!(Zf::Matrix{ComplexF64}, Z::Array{ComplexF64,3}, rmax::Int, p::Int, b::Int) -> Nothing

Flatten the compact tensor according to
`Zf[(j-1)rmax+i,c]=Z[i,j,c]`, giving the matrix representation used by
BLAS-based compact orthogonalization.

## Arguments
- `Zf::Matrix{ComplexF64}`: Preallocated flattened compact block.
- `Z::Array{ComplexF64,3}`: Degree-resolved compact tensor.
- `rmax::Int`: Allocated physical-rank dimension.
- `p::Int`: Number of linearization degree blocks.
- `b::Int`: Block-Arnoldi block size.

## Returns
- `Nothing`: `Zf` is filled in place.
"""
@inline function pack!(Zf::Matrix{ComplexF64}, Z::Array{ComplexF64,3}, rmax::Int, p::Int, b::Int)::Nothing
    @inbounds for c = 1:b, j = 1:p, i = 1:rmax
        Zf[(j-1)*rmax+i,c] = Z[i,j,c]
    end
    return nothing
end

"""
    unpack!(Z::Array{ComplexF64,3}, Zf::Matrix{ComplexF64}, rmax::Int, p::Int, b::Int) -> Nothing

Restore the degree-resolved tensor using
`Z[i,j,c]=Zf[(j-1)rmax+i,c]`. This is the inverse of [`pack!`](@ref).

## Arguments
- `Z::Array{ComplexF64,3}`: Preallocated degree-resolved compact tensor.
- `Zf::Matrix{ComplexF64}`: Flattened compact block.
- `rmax::Int`: Allocated physical-rank dimension.
- `p::Int`: Number of linearization degree blocks.
- `b::Int`: Block-Arnoldi block size.

## Returns
- `Nothing`: `Z` is filled in place.
"""
@inline function unpack!(Z::Array{ComplexF64,3}, Zf::Matrix{ComplexF64}, rmax::Int, p::Int, b::Int)::Nothing
    @inbounds for c = 1:b, j = 1:p, i = 1:rmax
        Z[i,j,c] = Zf[(j-1)*rmax+i,c]
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

Writing `zⱼ=Uγⱼ+Tⱼ(0)y` yields `P(0)y=rhs`, with
`P(0)=B₀-B₂+B₄-⋯` and

    rhs=-Σⱼ₌₀ᵖ⁻³Wⱼγⱼ-Wₚ₋₂γₚ₋₂+Wₚγₚ₋₂-Wₚ₋₁γₚ₋₁-2Wₚxₚ₋₁.

`F` is the precomputed LU factorization of `P(0)`. The solved physical block
is projected twice against `U`; its orthogonal remainder is factorized as
`Y=UnewΣV*`. Singular values above the physical-rank threshold define new
basis directions and `Cnew=ΣV*` their compact coordinates. The SVD is used
deliberately as a rank-revealing factorization for degenerate and nearly
degenerate physical blocks.

The output is reconstructed from `zⱼ=Uγⱼ+Tⱼ(0)Yoriginal`, using
`T₂q(0)=(-1)^q` and `T₂q₊₁(0)=0`.

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
    γ = zeros(ComplexF64,r,p,b); RHS = zeros(ComplexF64,N,b); tmp = similar(RHS); Y = similar(RHS)
    Hphys = zeros(ComplexF64,r,b); @views γ[:,2,:] .= Gin[1:r,1,:]

    for j = 1:p-2
        @views @. γ[:,j+2,:] = -γ[:,j,:]+2Gin[1:r,j+1,:]
    end

    t = time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        for j = 0:p-3
            mul!(tmp,@view(Wj(W,N,j)[:,1:r]),@view(γ[:,j+1,:])); RHS .-= tmp
        end
        mul!(tmp,@view(Wj(W,N,p-2)[:,1:r]),@view(γ[:,p-1,:])); RHS .-= tmp
        mul!(tmp,@view(Wj(W,N,p)[:,1:r]),@view(γ[:,p-1,:])); RHS .+= tmp
        mul!(tmp,@view(Wj(W,N,p-1)[:,1:r]),@view(γ[:,p,:])); RHS .-= tmp
        mul!(tmp,@view(Wj(W,N,p)[:,1:r]),@view(Gin[1:r,p,:])); @. RHS -= 2tmp
    end
    trhs = (time_ns()-t)*1e-9

    copyto!(Y,RHS); t = time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS ldiv!(F,Y)
    tlu = (time_ns()-t)*1e-9; t = time_ns()

    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(Hphys,adjoint(@view(U[:,1:r])),Y)
        mul!(Y,@view(U[:,1:r]),Hphys,-1+0im,1+0im)
    end
    H2 = zeros(ComplexF64,r,b)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(H2,adjoint(@view(U[:,1:r])),Y)
        mul!(Y,@view(U[:,1:r]),H2,-1+0im,1+0im)
    end
    Hphys .+= H2

    @blas_multi_then_1 MAX_BLAS_THREADS FY = svd(Y; full=false)
    σ = FY.S; σ1 = isempty(σ) ? 0.0 : σ[1]
    bp = count(>(max(σ1*1e-12,1e-14)),σ); rn = r+bp
    rn <= size(U,2) || error("Physical rank $rn exceeds rmax=$(size(U,2))")

    Cnew = if bp > 0
        @views U[:,r+1:rn] .= FY.U[:,1:bp]
        Diagonal(σ[1:bp])*FY.Vt[1:bp,:]
    else
        zeros(ComplexF64,0,b)
    end
    tphys = (time_ns()-t)*1e-9
    tcache = bp > 0 ? cache_block!(W,B,U,r+1,rn) : 0.0

    fill!(Z,0); @views Z[1:r,1:p,:] .+= γ; @views Z[1:r,1,:] .+= Hphys
    bp > 0 && (@views Z[r+1:rn,1,:] .+= Cnew)

    s = -1.0
    for j = 2:2:p-1
        @views Z[1:r,j+1,:] .+= s.*Hphys
        bp > 0 && (@views Z[r+1:rn,j+1,:] .+= s.*Cnew)
        s = -s
    end
    return rn,tcache,tlu,trhs,tphys
end

################################################################################
# PERSISTENT BLOCK-CORK ITERATION
################################################################################

"""
    CORKState

Persistent compact block-Arnoldi state. `U` is the common physical basis,
`W=B*U` caches all coefficient actions, `G`/`Gf` store the degree-resolved
and flattened compact Arnoldi basis, and `Hb` stores the projected inverse
linearization. The next residual block is retained in `pendingG`/`pendingGf`,
allowing adaptive Krylov growth without restarting.

## Arguments
- `U::Matrix{ComplexF64}`: Common physical basis.
- `W::Matrix{ComplexF64}`: Vertically stacked coefficient-action cache.
- `G::Array{ComplexF64,3}`: Degree-resolved compact Arnoldi basis.
- `Gf::Matrix{ComplexF64}`: Flattened compact Arnoldi basis.
- `Hb::Matrix{ComplexF64}`: Projected inverse-linearization matrix.
- `pendingG::Array{ComplexF64,3}`: Pending degree-resolved Arnoldi tail.
- `pendingGf::Matrix{ComplexF64}`: Flattened pending Arnoldi tail.
- `Z::Array{ComplexF64,3}`: Degree-resolved work tensor.
- `Zf::Matrix{ComplexF64}`: Flattened work block.
- `H1::Matrix{ComplexF64}`: First compact projection coefficients.
- `H2::Matrix{ComplexF64}`: Second compact projection coefficients.
- `r::Int`: Current physical rank.
- `n::Int`: Current compact Krylov dimension.
- `b::Int`: Block-Arnoldi block size.
- `p::Int`: Chebyshev polynomial degree.
- `N::Int`: Physical matrix dimension.
- `rmax::Int`: Allocated maximum physical rank.
- `maxdim::Int`: Maximum compact Krylov dimension.
- `pending::Bool`: Whether an unpromoted Arnoldi tail is available.
- `cache::Float64`: Accumulated coefficient-cache time.
- `lu::Float64`: Accumulated physical LU-solve time.
- `rhs::Float64`: Accumulated Chebyshev RHS time.
- `phys::Float64`: Accumulated physical-basis expansion time.
- `orth::Float64`: Accumulated compact orthogonalization time.
- `napply::Int`: Number of inverse-linearization applications.

## Returns
- `CORKState`: Persistent state used by the adaptive CORK iteration.
"""
mutable struct CORKState
    U::Matrix{ComplexF64}
    W::Matrix{ComplexF64}
    G::Array{ComplexF64,3}
    Gf::Matrix{ComplexF64}
    Hb::Matrix{ComplexF64}
    pendingG::Array{ComplexF64,3}
    pendingGf::Matrix{ComplexF64}
    Z::Array{ComplexF64,3}
    Zf::Matrix{ComplexF64}
    H1::Matrix{ComplexF64}
    H2::Matrix{ComplexF64}
    r::Int
    n::Int
    b::Int
    p::Int
    N::Int
    rmax::Int
    maxdim::Int
    pending::Bool
    cache::Float64
    lu::Float64
    rhs::Float64
    phys::Float64
    orth::Float64
    napply::Int
end

"""
    init_cork(B::Matrix{ComplexF64}, p::Int, N::Int; b::Int=10, maxdim::Int=1600, seed::Int=123) -> CORKState

Initialize the common physical basis, coefficient cache, and first normalized
compact block-Arnoldi block.

## Arguments
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `p::Int`: Chebyshev polynomial degree.
- `N::Int`: Physical matrix dimension.
- `b::Int`: Block-Arnoldi block size.
- `maxdim::Int`: Maximum compact Krylov dimension.
- `seed::Int`: Random initialization seed.

## Returns
- `CORKState`: Initialized persistent block-CORK state.
"""
function init_cork(B::Matrix{ComplexF64}, p::Int, N::Int; b::Int=10, maxdim::Int=1600, seed::Int=123)::CORKState
    maxdim % b == 0 || error("maxdim must be divisible by block size")
    rmax = min(N,p+maxdim+b+2); cd = rmax*p
    U = zeros(ComplexF64,N,rmax); W = zeros(ComplexF64,(p+1)*N,rmax)
    G = zeros(ComplexF64,rmax,p,maxdim+b); Gf = zeros(ComplexF64,cd,maxdim+b)
    Hb = zeros(ComplexF64,maxdim+b,maxdim); rng = MersenneTwister(seed)

    X = randn(rng,ComplexF64,N,p)
    @blas_multi_then_1 MAX_BLAS_THREADS FX = qr(X)
    @blas_multi_then_1 MAX_BLAS_THREADS @views U[:,1:p] .= FX.Q*Matrix{ComplexF64}(I,N,p)
    r = p

    Z0 = zeros(ComplexF64,rmax,p,b); @views randn!(rng,Z0[1:r,:,:])
    Z0f = zeros(ComplexF64,cd,b); pack!(Z0f,Z0,rmax,p,b)
    bn,_ = compact_block_qr!(Z0f,b); bn == b || error("Initial block breakdown")
    unpack!(Z0,Z0f,rmax,p,b); @views G[:,:,1:b] .= Z0; Gf[:,1:b] .= Z0f
    tc = cache_block!(W,B,U,1,r)

    return CORKState(U,W,G,Gf,Hb,zeros(ComplexF64,rmax,p,b),zeros(ComplexF64,cd,b),
        zeros(ComplexF64,rmax,p,b),zeros(ComplexF64,cd,b),zeros(ComplexF64,maxdim+b,b),
        zeros(ComplexF64,maxdim+b,b),r,b,b,p,N,rmax,maxdim,false,tc,0.0,0.0,0.0,0.0,0)
end

"""
    compute_tail!(S::CORKState, B::Matrix{ComplexF64}, F) -> Nothing

Compute the next block-Arnoldi residual without promoting it into the active
basis. The last active block is passed through the compact inverse
linearization, projected twice against the current basis, normalized by QR,
and retained in `pendingG`/`pendingGf`.

At active dimension `n`, `S.Hb[n+1:n+b,1:n]` is the residual row required for
Ritz convergence. If the adaptive solve subsequently increases `n`, the same
pending block is promoted rather than recomputed.

## Arguments
- `S::CORKState`: Persistent compact block-CORK state.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `F`: LU factorization of `P(0)`.

## Returns
- `Nothing`: `S` is updated in place with a normalized pending Arnoldi tail.
"""
function compute_tail!(S::CORKState, B::Matrix{ComplexF64}, F)::Nothing
    S.pending && return nothing
    n = S.n; b = S.b; p = S.p; r = S.r; cols = n-b+1:n; Gin = @view S.G[:,:,cols]
    fill!(S.Z,0)
    rn,tc,tl,tr,tp = block_apply!(S.Z,S.U,S.W,B,F,Gin,S.N,p,r,b)
    S.r = rn; S.cache += tc; S.lu += tl; S.rhs += tr; S.phys += tp

    pack!(S.Zf,S.Z,S.rmax,p,b); fill!(S.H1,0); fill!(S.H2,0); t = time_ns()
    compact_project_cgs2!(S.Zf,S.Gf,n,S.H1,S.H2)
    bn,Rb = compact_block_qr!(S.Zf,b); bn == b || error("Block breakdown at n=$n: $bn/$b")

    @views S.Hb[1:n,cols] .= S.H1[1:n,:]
    @views S.Hb[n+1:n+b,cols] .= Rb
    S.orth += (time_ns()-t)*1e-9

    copyto!(S.pendingGf,S.Zf); unpack!(S.pendingG,S.pendingGf,S.rmax,p,b)
    S.pending = true; S.napply += 1
    return nothing
end

"""
    promote_tail!(S::CORKState) -> Nothing

Promote the normalized pending residual into the active compact Arnoldi basis.
The block becomes columns `n+1:n+b` of `G`/`Gf`, the active dimension grows
by `b`, and the pending flag is cleared.

Separating computation from promotion lets the same block serve first as the
Ritz residual at dimension `n` and then as the next active Arnoldi block.

## Arguments
- `S::CORKState`: Persistent state containing a pending residual block.

## Returns
- `Nothing`: `S` is modified in place and its active dimension increases by
  `S.b`.
"""
function promote_tail!(S::CORKState)::Nothing
    S.pending || error("No pending tail")
    n = S.n; b = S.b; n+b <= S.maxdim || error("Exceeded maxdim=$(S.maxdim)")
    @views S.Gf[:,n+1:n+b] .= S.pendingGf
    @views S.G[:,:,n+1:n+b] .= S.pendingG
    S.n += b; S.pending = false
    return nothing
end

"""
    extend!(S::CORKState, B::Matrix{ComplexF64}, F, m::Int) -> Nothing

Extend the persistent factorization to compact Krylov dimension `m`. Existing
Arnoldi vectors are retained; only the additional blocks are generated. A
fresh pending residual tail is left available at dimension `m`.

## Arguments
- `S::CORKState`: Persistent CORK state.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `F`: LU factorization of `P(0)`.
- `m::Int`: Requested compact Krylov dimension.

## Returns
- `Nothing`: `S` is extended in place.
"""
function extend!(S::CORKState, B::Matrix{ComplexF64}, F, m::Int)::Nothing
    m % S.b == 0 || error("m must be divisible by block size")
    m <= S.maxdim || error("m > $(S.maxdim)")
    m >= S.n || error("Cannot shrink CORK basis")
    while S.n < m
        S.pending || compute_tail!(S,B,F)
        promote_tail!(S)
    end
    compute_tail!(S,B,F)
    return nothing
end

################################################################################
# RITZ ROOTS AND MULTIPLICITY
################################################################################

"""
    RitzCluster

Cluster of nearby projected Ritz values representing one physical root
location. Multiplicity is inferred from converged singular directions of the
block-Arnoldi residual.

## Arguments
- `k::ComplexF64`: Clustered physical wavenumber.
- `multiplicity::Int`: Converged physical multiplicity.
- `ritz_count::Int`: Number of projected Ritz values in the cluster.
- `residual::Float64`: Largest accepted residual singular value.
- `spread::Float64`: Maximum member distance from the cluster center.

## Returns
- `RitzCluster`: Multiplicity-aware physical Ritz cluster.
"""
struct RitzCluster
    k::ComplexF64
    multiplicity::Int
    ritz_count::Int
    residual::Float64
    spread::Float64
end

# Group nearby complex Ritz values after ordering them by real part.
function _cluster_indices(vals::Vector{ComplexF64}, tol::Float64)::Vector{Vector{Int}}
    isempty(vals) && return Vector{Vector{Int}}()
    order = sortperm(real.(vals)); groups = Vector{Vector{Int}}(); g = Int[order[1]]
    for q in order[2:end]
        kc = sum(vals[j] for j in g)/length(g)
        if abs(vals[q]-kc) <= tol
            push!(g,q)
        else
            push!(groups,g); g = Int[q]
        end
    end
    push!(groups,g)
    return groups
end

"""
    ritz_clusters(S::CORKState, k0::Float64, Δpoly::Float64, m::Int; edge_tol::Float64=1e-8, res_tol::Float64=1e-8, cluster_tol::Float64=1e-6, imag_search_tol::Float64=1e-4) -> Vector{RitzCluster}

Extract multiplicity-aware roots from the projected inverse linearization.
Projected eigenvalues are mapped by `μ→t=1/μ→k=k₀+Δpoly*t`; candidates
outside the guarded interval or loose imaginary search strip are discarded.

For each cluster, the final block-Arnoldi row gives the residual `R=Htail*Y`.
The number of residual singular values below `res_tol` determines the
converged physical multiplicity.

## Arguments
- `S::CORKState`: Persistent CORK state at dimension `m`.
- `k0::Float64`: Physical expansion center.
- `Δpoly::Float64`: Guarded polynomial half-width.
- `m::Int`: Active compact Krylov dimension.
- `edge_tol::Float64`: Tolerance on the normalized guarded interval.
- `res_tol::Float64`: Residual singular-value tolerance.
- `cluster_tol::Float64`: Absolute Ritz clustering distance.
- `imag_search_tol::Float64`: Loose imaginary strip used during discovery.

## Returns
- `Vector{RitzCluster}`: Sorted converged clusters in the guarded interval.
"""
function ritz_clusters(S::CORKState, k0::Float64, Δpoly::Float64, m::Int; edge_tol::Float64=1e-8, res_tol::Float64=1e-8, cluster_tol::Float64=1e-6, imag_search_tol::Float64=1e-4)::Vector{RitzCluster}
    S.n == m || error("State dimension $(S.n) != requested m=$m")
    S.pending || error("Arnoldi tail missing")

    H = Matrix(@view S.Hb[1:m,1:m])
    @blas_multi_then_1 MAX_BLAS_THREADS E = eigen(H)
    tail = @view S.Hb[m+1:m+S.b,1:m]
    inds = Int[]; kvals = ComplexF64[]

    for j = eachindex(E.values)
        μ = E.values[j]; abs(μ) > 1e-14 || continue
        t = inv(μ); k = ComplexF64(k0+Δpoly*t)
        -1-edge_tol <= real(t) <= 1+edge_tol || continue
        abs(imag(k)) <= imag_search_tol || continue
        push!(inds,j); push!(kvals,k)
    end
    isempty(inds) && return RitzCluster[]

    out = RitzCluster[]
    for g in _cluster_indices(kvals,cluster_tol)
        js = inds[g]; kc = sum(kvals[q] for q in g)/length(g)
        spread = maximum(abs(kvals[q]-kc) for q in g)
        @blas_multi_then_1 MAX_BLAS_THREADS Rn = copy(tail*@view(E.vectors[:,js]))
        @inbounds for q = 1:length(js)
            Rn[:,q] ./= max(1.0,abs(E.values[js[q]]))
        end
        @blas_multi_then_1 MAX_BLAS_THREADS σ = sort(svdvals(Rn))
        mult = count(<=(res_tol),σ); mult == 0 && continue
        push!(out,RitzCluster(kc,mult,length(g),σ[mult],spread))
    end

    sort!(out,by=x->real(x.k))
    return out
end

################################################################################
# PHYSICAL FILTERING AND EDGE COMPLETENESS
################################################################################

# Strict physical acceptance after the looser Ritz-discovery stage.
@inline _cluster_good(c, imag_tol::Float64, res_tol::Float64)::Bool =
    c !== nothing && abs(imag(c.k)) <= imag_tol && c.residual <= res_tol

"""
    edge_completeness(clusters::Vector{RitzCluster}, k0::Float64, Δ::Float64; imag_tol::Float64=1e-7, res_tol::Float64=1e-8)

Find the nearest roots immediately outside and inside both requested edges:
`kL,out<k₀-Δ≤kL,in` and `kR,in≤k₀+Δ<kR,out`. Completeness requires all
four clusters to satisfy the strict physical tolerances.

## Arguments
- `clusters::Vector{RitzCluster}`: Converged clusters in the guarded interval.
- `k0::Float64`: Requested interval center.
- `Δ::Float64`: Requested interval half-width.
- `imag_tol::Float64`: Strict imaginary-part tolerance.
- `res_tol::Float64`: Strict residual tolerance.

## Returns
- `Tuple`: `(complete,roots,good)`, containing the overall status, four edge
  clusters or `nothing`, and their strict acceptance flags.
"""
function edge_completeness(clusters::Vector{RitzCluster}, k0::Float64, Δ::Float64; imag_tol::Float64=1e-7, res_tol::Float64=1e-8)
    kmin = k0-Δ; kmax = k0+Δ
    inside = [c for c in clusters if kmin <= real(c.k) <= kmax]
    left = [c for c in clusters if real(c.k) < kmin]
    right = [c for c in clusters if real(c.k) > kmax]
    roots = (isempty(left) ? nothing : last(left),isempty(inside) ? nothing : first(inside),
        isempty(inside) ? nothing : last(inside),isempty(right) ? nothing : first(right))
    good = ntuple(i -> _cluster_good(roots[i],imag_tol,res_tol),4)
    return all(good),roots,good
end

# Print one edge cluster and its strict physical acceptance status.
function print_edge_root(name::String, c, good::Bool)::Nothing
    if c === nothing
        @printf("%-12s MISSING\n",name)
    else
        @printf("%-12s k=%16.10f Im=%+10.3e ρ=%10.3e mult=%2d Ritz=%2d %s\n",
            name,real(c.k),imag(c.k),c.residual,c.multiplicity,c.ritz_count,good ? "PASS" : "FAIL")
    end
    return nothing
end

# Print the four-root requested-edge completeness diagnostic.
function print_edge_check(roots, good, k0::Float64, Δ::Float64, imag_tol::Float64, res_tol::Float64)::Nothing
    names = ("left-out","left-in","right-in","right-out")
    println("\nEDGE COMPLETENESS CHECK\n","-"^104)
    for i = 1:4
        print_edge_root(names[i],roots[i],good[i])
    end
    @printf("requested interval = [%.10f, %.10f]\nimag_tol           = %.3e\nres_tol            = %.3e\n",
        k0-Δ,k0+Δ,imag_tol,res_tol)
    println("edge status        = ",all(good) ? "PASS" : "FAIL")
    return nothing
end

# Retain strictly accepted clusters inside the requested interval.
function physical_clusters(clusters::Vector{RitzCluster}, k0::Float64, Δ::Float64; imag_tol::Float64=1e-7, res_tol::Float64=1e-8)::Vector{RitzCluster}
    kmin = k0-Δ; kmax = k0+Δ
    return [c for c in clusters if kmin <= real(c.k) <= kmax && abs(imag(c.k)) <= imag_tol && c.residual <= res_tol]
end

# Expand every cluster into one tuple per physical state.
function expand_clusters(clusters::Vector{RitzCluster})::Vector{Tuple{Float64,Float64,Float64}}
    out = Tuple{Float64,Float64,Float64}[]
    for c in clusters, _ = 1:c.multiplicity
        push!(out,(real(c.k),imag(c.k),c.residual))
    end
    sort!(out,by=first)
    return out
end

# Maximum root drift between two ordered multiplicity-expanded spectra.
function spectrum_distance(a::Vector{Tuple{Float64,Float64,Float64}}, b::Vector{Tuple{Float64,Float64,Float64}})::Float64
    length(a) == length(b) || return Inf
    isempty(a) && return Inf
    return maximum(abs(complex(a[i][1],a[i][2])-complex(b[i][1],b[i][2])) for i = eachindex(a))
end

################################################################################
# ADAPTIVE CONVERGENCE
################################################################################

"""
    adaptive_cork(B::Matrix{ComplexF64}, F, p::Int, N::Int; k0::Float64, Δ::Float64, Δpoly::Float64, b::Int=10, mstart::Int=200, mstep::Int=100, maxdim::Int=1600, stable_checks::Int=2, imag_tol::Float64=1e-7, edge_tol::Float64=1e-8, res_tol::Float64=1e-8, stable_tol::Float64=1e-8, cluster_tol::Float64=1e-6, seed::Int=123, imag_search_tol::Float64=1e-4)

Grow one persistent CORK factorization until the multiplicity-expanded
requested spectrum changes by at most `stable_tol` for `stable_checks`
consecutive dimensions and converged roots bracket both requested edges.

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
- `maxdim::Int`: Maximum compact Krylov dimension.
- `stable_checks::Int`: Required consecutive stable spectra.
- `imag_tol::Float64`: Strict physical imaginary-part tolerance.
- `edge_tol::Float64`: Normalized guarded-interval tolerance.
- `res_tol::Float64`: Residual singular-value tolerance.
- `stable_tol::Float64`: Maximum accepted root drift.
- `cluster_tol::Float64`: Ritz clustering distance.
- `seed::Int`: Random initialization seed.
- `imag_search_tol::Float64`: Loose imaginary discovery strip.

## Returns
- `Tuple`: Final state, expanded spectrum, accepted clusters, all guarded
  clusters, edge roots, edge flags, and final Krylov dimension.
"""
function adaptive_cork(B::Matrix{ComplexF64}, F, p::Int, N::Int; k0::Float64, Δ::Float64, Δpoly::Float64, b::Int=10, mstart::Int=200, mstep::Int=100, maxdim::Int=1600, stable_checks::Int=2, imag_tol::Float64=1e-7, edge_tol::Float64=1e-8, res_tol::Float64=1e-8, stable_tol::Float64=1e-8, cluster_tol::Float64=1e-6, seed::Int=123, imag_search_tol::Float64=1e-4)
    mstart % b == 0 || error("mstart must be divisible by block size")
    mstep % b == 0 || error("mstep must be divisible by block size")

    S = init_cork(B,p,N; b=b,maxdim=maxdim,seed=seed)
    prev = Tuple{Float64,Float64,Float64}[]; nstable = 0; m = mstart; t0 = time_ns()
    last_all = RitzCluster[]; last_phys = RitzCluster[]
    last_roots = (nothing,nothing,nothing,nothing); last_good = (false,false,false,false)

    while m <= maxdim
        extend!(S,B,F,m)
        allclusters = ritz_clusters(S,k0,Δpoly,m; edge_tol=edge_tol,res_tol=res_tol,
            cluster_tol=cluster_tol,imag_search_tol=imag_search_tol)
        phys = physical_clusters(allclusters,k0,Δ; imag_tol=imag_tol,res_tol=res_tol)
        ks = expand_clusters(phys)
        edge_ok,edge_roots,edge_good = edge_completeness(allclusters,k0,Δ; imag_tol=imag_tol,res_tol=res_tol)

        drift = spectrum_distance(prev,ks)
        same = length(prev) == length(ks) && isfinite(drift) && drift <= stable_tol
        nstable = same ? nstable+1 : 0
        maxmult = isempty(phys) ? 0 : maximum(c.multiplicity for c in phys)
        maxρ = isempty(phys) ? Inf : maximum(c.residual for c in phys)

        @timeit_debug "CORK adaptive diagnostics" begin
            @printf("m=%4d rank=%4d loc=%4d states=%4d mult=%2d maxρ=%9.2e drift=%9.2e stable=%d/%d edges=%s applies=%4d time=%7.3f\n",
                m,S.r,length(phys),length(ks),maxmult,maxρ,drift,nstable,stable_checks,
                edge_ok ? "PASS" : "FAIL",S.napply,(time_ns()-t0)*1e-9)
        end

        last_all = allclusters; last_phys = phys; last_roots = edge_roots; last_good = edge_good
        if nstable >= stable_checks && edge_ok
            @timeit_debug "CORK adaptive diagnostics" begin
                println("Requested Ritz spectrum stabilized and both interval edges are bracketed.")
            end
            return S,ks,phys,allclusters,edge_roots,edge_good,m
        end

        prev = ks; m += mstep
    end

    @timeit_debug "CORK adaptive diagnostics" begin
        @warn "Reached maxdim without both spectrum stability and edge completeness"
    end
    return S,expand_clusters(last_phys),last_phys,last_all,last_roots,last_good,maxdim
end

struct CORKSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K
    p::Int
    guard::T
    nlevels::Int
    Rmax::T
    b::Int
    mstart::Int
    mstep::Int
    maxdim::Int
    stable_checks::Int
    imag_tol::T
    edge_tol::T
    res_tol::T
    stable_tol::T
    cluster_tol::T
    imag_search_tol::T
    seed::Int
    validate::Bool
end

"""
    CORKSolver(kernel::K; p::Int=14, guard::Real=0.15, nlevels::Int=150, Rmax::Real=0.8, b::Int=10, mstart::Int=200, mstep::Int=100, maxdim::Int=1600, stable_checks::Int=2, imag_tol::Real=1e-7, edge_tol::Real=1e-8, res_tol::Real=1e-10, stable_tol::Real=1e-9, cluster_tol::Real=1e-6, imag_search_tol::Real=1e-4, seed::Int=123, validate::Bool=true) where {K<:SweepBIMSolver} -> CORKSolver

Construct a CORK solver for a BIM Fredholm nonlinear eigenvalue problem.

For full-spectrum computations, `nlevels` is the target number of physical
levels in each requested CORK window and `Rmax` is the maximum requested
half-width `Δ`. The Chebyshev polynomial itself is constructed on the guarded
half-width Δpoly=(1+guard)Δ,
so `Rmax` limits the requested spectral interval rather than the polynomial
interval.

## Arguments
- `kernel::K`: BIM kernel used to construct the Fredholm operator.

## Keyword Arguments
- `p::Int=14`: Chebyshev polynomial degree.
- `guard::Real=0.15`: Relative enlargement of the requested interval used for polynomial construction and edge certification.
- `nlevels::Int=150`: Target number of physical levels per full-spectrum CORK window.
- `Rmax::Real=0.8`: Maximum requested CORK half-width `Δ`.
- `b::Int=10`: CORK block size.
- `mstart::Int=200`: Initial Krylov dimension.
- `mstep::Int=100`: Krylov-dimension increment.
- `maxdim::Int=1600`: Maximum Krylov dimension.
- `stable_checks::Int=2`: Number of consecutive stable Ritz checks required for convergence.
- `imag_tol::Real=1e-7`: Final imaginary-part tolerance in physical `k` units.
- `edge_tol::Real=1e-8`: Edge-completeness tolerance.
- `res_tol::Real=1e-10`: Ritz residual tolerance.
- `stable_tol::Real=1e-9`: Spectrum-stability tolerance between successive checks.
- `cluster_tol::Real=1e-6`: Ritz clustering tolerance.
- `imag_search_tol::Real=1e-4`: Loose imaginary strip used when searching for physical Ritz roots.
- `seed::Int=123`: Random seed used to initialize the block Krylov process.
- `validate::Bool=true`: Validate the Chebyshev polynomial before the CORK iteration.

## Returns
- `CORKSolver`: Configured CORK solver.
"""
function CORKSolver(kernel::K; p::Int=14, guard::Real=0.15, nlevels::Int=150, Rmax::Real=0.8, b::Int=10, mstart::Int=200, mstep::Int=100, maxdim::Int=1600, stable_checks::Int=2, imag_tol::Real=1e-7, edge_tol::Real=1e-8, res_tol::Real=1e-10, stable_tol::Real=1e-9, cluster_tol::Real=1e-6, imag_search_tol::Real=1e-4, seed::Int=123, validate::Bool=true) where {K<:SweepBIMSolver}
    T=_bim_numeric_type(kernel)
    T===Float64 || throw(ArgumentError("CORKSolver currently requires a Float64 BIM kernel"))
    p>=2 || throw(ArgumentError("p must be at least 2; received p=$p"))
    guard>=0 || throw(ArgumentError("guard must be nonnegative; received guard=$guard"))
    nlevels>0 || throw(ArgumentError("nlevels must be positive; received nlevels=$nlevels"))
    Rmax>0 || throw(ArgumentError("Rmax must be positive; received Rmax=$Rmax"))
    b>0 || throw(ArgumentError("b must be positive; received b=$b"))
    mstart>0 || throw(ArgumentError("mstart must be positive; received mstart=$mstart"))
    mstep>0 || throw(ArgumentError("mstep must be positive; received mstep=$mstep"))
    maxdim>=mstart || throw(ArgumentError("maxdim must satisfy maxdim>=mstart"))
    mstart%b==0 || throw(ArgumentError("mstart must be divisible by b"))
    mstep%b==0 || throw(ArgumentError("mstep must be divisible by b"))
    maxdim%b==0 || throw(ArgumentError("maxdim must be divisible by b"))
    stable_checks>0 || throw(ArgumentError("stable_checks must be positive"))
    return CORKSolver{T,K}(kernel,p,T(guard),nlevels,T(Rmax),b,mstart,mstep,maxdim,stable_checks,T(imag_tol),T(edge_tol),T(res_tol),T(stable_tol),T(cluster_tol),T(imag_search_tol),seed,validate)
end

_bim_numeric_type(::CORKSolver{T}) where {T} = T

# Delegate boundary discretization to the wrapped BIM solver.
evaluate_points(solver::CORKSolver, billiard::Bi, k) where {Bi<:AbsBilliard} =
    evaluate_points(solver.kernel,billiard,k)

# Compact kernel name used only by debug diagnostics.
@inline _cork_kernel_name(solver)::String =
    solver isa DoubleLayerPotentialSolver ? "DLP" :
    solver isa CombinedFieldIntegralEquationSolver ? "CFIE" :
    solver isa CompositeBIMSolver ? "CompositeBIM" : string(typeof(solver))

"""
    _cork_solve_core(solver::CORKSolver, pts, k0, dk; multithreaded::Bool=true)

Execute the complete CORK pipeline on `[k₀-dk/2,k₀+dk/2]`: construct the
analytic polynomial on `Δpoly=(1+guard)dk/2`, optionally validate it, factorize
`P(0)`, run persistent adaptive CORK, and perform the edge-completeness check.
All progress and diagnostic printing is contained in `@timeit_debug`.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `pts`: Boundary discretization at the expansion center.
- `k0`: Requested interval center.
- `dk`: Full requested spectral width.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Tuple`: Polynomial, final state, expanded spectrum, accepted clusters, all
  guarded clusters, edge roots, edge flags, and final Krylov dimension.
"""
function _cork_solve_core(solver::CORKSolver, pts, k0, dk; multithreaded::Bool=true)
    k0f=Float64(k0); Δ=Float64(dk)/2
    Δ>0 || throw(ArgumentError("dk must be positive; received dk=$dk"))
    Δpoly=Δ*(1+solver.guard)
    k0f>Δpoly || throw(ArgumentError("CORK polynomial interval reaches k=0"))
    kmin=k0f-Δ; kmax=k0f+Δ; kpmin=k0f-Δpoly; kpmax=k0f+Δpoly
    @timeit_debug "CORK setup diagnostics" begin
        println("="^112)
        println("$(_cork_kernel_name(solver.kernel)) + ANALYTIC CHEBYSHEV + MULTIPLICITY/EDGE-CERTIFIED BLOCK CORK")
        println("="^112)
        @printf("k0                   = %.8f\nrequested Δ          = %.8f\nguard                = %.3f\n",k0f,Δ,solver.guard)
        @printf("polynomial Δ         = %.8f\nrequested interval   = [%.8f, %.8f]\n",Δpoly,kmin,kmax)
        @printf("polynomial interval  = [%.8f, %.8f]\ndegree               = %d\n",kpmin,kpmax,solver.p)
        @printf("imag_tol in k units  = %.3e\nimag search strip    = %.3e\nfull boundary points = %d\n",solver.imag_tol,solver.imag_search_tol,length(pts.xy))
        println("kernel               = ",typeof(solver.kernel))
        println("symmetry             = ",solver.kernel.symmetry)
        println("character            = ",solver.kernel.character)
    end
    P=nothing; tbuild=0.0
    @timeit_debug "CORK polynomial construction" begin
        P,tbuild=build_cork_polynomial(solver.kernel,pts,k0f,Δpoly,solver.p; multithreaded=multithreaded)
    end
    @timeit_debug "CORK polynomial diagnostics" begin
        @printf("CORK matrix size     = %d\nB build              = %.6f s\nB memory             = %.3f MiB\n",P.N,tbuild,Base.summarysize(P.B)/2^20)
        solver.kernel.symmetry!==nothing && @printf("dimension reduction  = %.3fx\n",length(pts.xy)/P.N)
    end
    solver.validate && validate_polynomial(solver.kernel,pts,P; multithreaded=multithreaded)
    t=time_ns(); A0=get_A0(P.B,P.N,solver.p); ta0=(time_ns()-t)*1e-9
    F=nothing; t=time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        F=lu(A0)
    end
    tfact=(time_ns()-t)*1e-9
    @timeit_debug "CORK factorization diagnostics" begin
        @printf("\nP(0) assembly        = %.6f s\nLU                   = %.6f s\n\n",ta0,tfact)
    end
    S,ks,clusters,allclusters,edge_roots,edge_good,mfinal=adaptive_cork(P.B,F,solver.p,P.N;k0=k0f,Δ=Δ,Δpoly=Δpoly,b=solver.b,mstart=solver.mstart,mstep=solver.mstep,maxdim=solver.maxdim,stable_checks=solver.stable_checks,imag_tol=solver.imag_tol,edge_tol=solver.edge_tol,res_tol=solver.res_tol,stable_tol=solver.stable_tol,cluster_tol=solver.cluster_tol,seed=solver.seed,imag_search_tol=solver.imag_search_tol)
    @timeit_debug "CORK final diagnostics" begin
        println("\n","="^112,"\nFINAL\n","="^112)
        @printf("Krylov dimension = %d\nphysical rank     = %d\noperator applies  = %d\n",mfinal,S.r,S.napply)
        @printf("root locations    = %d\nstates w/ mult.   = %d\n",length(clusters),length(ks))
        @printf("B_j GEMM          = %.6f s\nblock LU          = %.6f s\nChebyshev RHS      = %.6f s\n",S.cache,S.lu,S.rhs)
        @printf("physical SVD      = %.6f s\ncompact CGS2/QR   = %.6f s\n",S.phys,S.orth)
        print_edge_check(edge_roots,edge_good,k0f,Δ,solver.imag_tol,solver.res_tol)
        println("\n","="^112,"\nREQUESTED INTERVAL RITZ CLUSTERS\n","="^112)
        println("       Re(k)              Im(k)       mult  Ritz-count    residual       spread")
        for c in clusters
            @printf("%18.12f  %+12.3e     %3d       %3d       %.3e     %.3e\n",real(c.k),imag(c.k),c.multiplicity,c.ritz_count,c.residual,c.spread)
        end
        println("\n","="^112,"\nGUARD ROOTS\n","="^112)
        print_edge_root("left-out",edge_roots[1],edge_good[1])
        print_edge_root("right-out",edge_roots[4],edge_good[4])
    end
    return P,S,ks,clusters,allclusters,edge_roots,edge_good,mfinal
end

"""
    solve(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)

Compute all accepted roots in `[k₀-dk/2,k₀+dk/2]`. Degenerate roots are
repeated according to the multiplicity inferred from the CORK residual.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `pts::BoundaryPoints`: Boundary discretization at the expansion center.
- `k0`: Requested interval center.
- `dk`: Full requested spectral width.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Tuple{Vector{ComplexF64},Vector{Float64}}`: Multiplicity-expanded
  wavenumbers and their CORK residual estimates.
"""
function solve(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    _,_,ks,_,_,_,_,_ = _cork_solve_core(solver,pts,k0,dk; multithreaded=multithreaded)
    λ = ComplexF64[complex(x[1],x[2]) for x in ks]; ts = Float64[x[3] for x in ks]
    return λ,ts
end

"""
    solve_wavenumber(solver::CORKSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}

Find the accepted CORK eigenvalue nearest `k` in `[k-dk/2,k+dk/2]`. Boundary
points are generated at the expansion center `k`.

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
    pts = evaluate_points(solver,billiard,k); ks,ts = solve(solver,pts,k,dk; multithreaded=multithreaded)
    isempty(ks) && error("CORKSolver found no eigenvalue candidates in [$(k-dk/2),$(k+dk/2)]")
    idx = findmin(abs.(ks.-k))[2]
    return ks[idx],ts[idx]
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
    pts = evaluate_points(solver,billiard,k)
    return solve(solver,pts,k,dk; multithreaded=multithreaded)
end