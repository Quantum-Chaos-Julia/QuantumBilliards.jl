################################################################################
# CHEBYSHEV-CORK BOUNDARY-INTEGRAL EIGENSOLVER
#
# We seek the nonlinear BIM eigenvalues
#
#                              A(k)u = 0.
#
# For the requested window
#
#                  Δ = dk/2,    k ∈ [k₀-Δ,k₀+Δ],
#
# introduce the guarded polynomial half-width and normalized coordinate
#
#                  Δₚ = (1 + guard)Δ,
#                  t = (k-k₀)/Δₚ,
#                  k = k₀ + Δₚt.
#
# `recurrences.jl` constructs
#
#                  P(t) = Σⱼ₌₀ᵖ BⱼTⱼ(t) ≈ A(k₀+Δₚt),         (0)
#
# so the BIM NEP becomes the polynomial eigenvalue problem
#
#                              P(t)u = 0.                       (1)
#
# The Chebyshev recurrence
#
#                  T₀(t) = 1,    T₁(t) = t,
#                  Tⱼ₊₁(t) + Tⱼ₋₁(t) = 2tTⱼ(t)                (2)
#
# gives a linearization by introducing
#
#                  zⱼ = Tⱼ(t)u,    j = 0,...,p-1.
#
# Hence
#
#                  z₁ = tz₀,
#                  zⱼ₊₁ + zⱼ₋₁ = 2tzⱼ,    j=1,...,p-2.       (3)
#
# The final Chebyshev term in (1) is eliminated using
#
#                  Tₚ(t)u = 2tzₚ₋₁-zₚ₋₂,
#
# which gives the final block equation
#
#       Σⱼ₌₀ᵖ⁻³ Bⱼzⱼ + (Bₚ₋₂-Bₚ)zₚ₋₂ + Bₚ₋₁zₚ₋₁
#                              = -2tBₚzₚ₋₁.                    (4)
#
# With z=[z₀;z₁;...;zₚ₋₁] ∈ ℂᵖᴺ, equations (3)-(4) define
#
#                              L₀z = tL₁z,                     (5)
#
# where L₀,L₁ ∈ ℂᵖᴺˣᵖᴺ are p×p block matrices with N×N blocks,
#
#       L₀ = [ 0   I   0   0   ⋯       0
#              I   0   I   0   ⋯       0
#              0   I   0   I   ⋱       ⋮
#              ⋮   ⋱   ⋱   ⋱   ⋱       0
#              0   ⋯   0   I   0       I
#              B₀  B₁  ⋯   Bₚ₋₃ Bₚ₋₂-Bₚ Bₚ₋₁ ],
#
#       L₁ = [ I   0    0   0   ⋯   0
#              0   2I   0   0   ⋯   0
#              0   0    2I  0   ⋱   ⋮
#              ⋮   ⋱    ⋱   ⋱   ⋱   0
#              0   ⋯    0   0   2I  0
#              0   ⋯    0   0   0  -2Bₚ ],
#
# with I ∈ ℂᴺˣᴺ. The first p-1 block rows encode (3), while the final
# block row is (4). Thus det(L₀-tL₁)=0 linearizes det P(t)=0.
#
# Shift-and-invert at t=0 gives
#
#                  M = L₀⁻¹L₁,
#                  Mz = μz,    μ = 1/t.                       (6)
#
# The pN×pN matrices L₀ and L₁ are never formed. To apply M to an input
#
#                  x = [x₀;x₁;...;xₚ₋₁],
#
# solve
#
#                              L₀z = L₁x.                      (7)
#
# The first p-1 block rows give
#
#                  z₁ = x₀,
#                  zⱼ₋₁ + zⱼ₊₁ = 2xⱼ,    j=1,...,p-2.        (8)
#
# Their general solution can be written
#
#                  zⱼ = γⱼ + Tⱼ(0)y,                          (9)
#
# with
#
#                  γ₀ = 0,
#                  γ₁ = x₀,
#                  γⱼ₊₁ = -γⱼ₋₁ + 2xⱼ.                      (10)
#
# Since
#
#                  T₂q(0) = (-1)^q,
#                  T₂q₊₁(0) = 0,
#
# substitution of (9) into the final block row of (7) gives
#
#                              P(0)y = rhs,                    (11)
#
# where
#
#                  P(0) = Σⱼ₌₀ᵖ BⱼTⱼ(0)
#                       = B₀-B₂+B₄-B₆+⋯,                     (12)
#
#                  rhs = -Σⱼ₌₀ᵖ⁻³ Bⱼγⱼ
#                        -(Bₚ₋₂-Bₚ)γₚ₋₂
#                        -Bₚ₋₁γₚ₋₁
#                        -2Bₚxₚ₋₁.                             (13)
#
# Thus one factorization
#
#                              P(0) = LU                       (14)
#
# supplies every application of M=L₀⁻¹L₁.
#
# CORK avoids storing full pN-dimensional Krylov vectors. Their N-dimensional
# physical blocks share an orthonormal basis U ∈ ℂᴺˣʳ,
#
#                              zⱼ = Ugⱼ,                       (15)
#
# while the polynomial actions are cached as
#
#                              Wⱼ = BⱼU.                       (16)
#
# Hence Bⱼzⱼ = Wⱼgⱼ. If the solution y of (11) contains a component outside
# span(U), write
#
#                  y = UUᴴy + y⊥,
#                  Uᴴy⊥ = 0,
#                  y⊥ = Unew Cnew,                            (17)
#
# and append the SVD basis Unew to U together with BⱼUnew to Wⱼ.
#
# After m compact Arnoldi vectors,
#
#                  MQₘ = QₘHₘ + Q₊Htail,                     (18)
#
# and the Ritz problem is
#
#                              Hₘv = μv.                       (19)
#
# Each Ritz value is mapped back independently by
#
#                  μ → t = 1/μ → k = k₀ + Δₚ/μ,              (20)
#
# with projected inverse-linearization residual indicator
#
#                  ρ = ‖Htail v‖ / max(1,|μ|).                (21)
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

Validate the Chebyshev approximation against directly constructed BIM
Fredholm matrices. At each normalized coordinate `t`, the reported relative
error is

    ‖P(t)-A(k₀+Δpoly*t)‖ / ‖A(k₀+Δpoly*t)‖,

where `Δpoly=P.Δ` is the guarded polynomial half-width.

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
    worst > tol && throw(ArgumentError("Polynomial validation failed: worst relative error = $worst exceeds tolerance $tol. Try reducing the polynomial half-width or increasing the polynomial degree p."))
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

Orthogonalize the active compact tensor `Z[1:r,1:p,:]` against the first `n`
compact Arnoldi vectors stored in `G` using two-pass classical Gram-Schmidt.

## Arguments
- `Z::Array{ComplexF64,3}`: Compact residual tensor, modified in place.
- `G::Array{ComplexF64,3}`: Compact Arnoldi basis.
- `r::Int`: Active physical rank.
- `n::Int`: Number of active Arnoldi vectors.
- `p::Int`: Number of compact polynomial blocks.
- `H1::Matrix{ComplexF64}`: Workspace receiving the accumulated CGS2 projection coefficients.
- `H2::Matrix{ComplexF64}`: Workspace for the second CGS projection.

## Returns
- `Nothing`: `Z` is orthogonalized in place and `H1[1:n,:]` contains the total projection coefficients.
"""
function compact_project!(Z::Array{ComplexF64,3}, G::Array{ComplexF64,3}, r::Int, n::Int, p::Int, H1::Matrix{ComplexF64}, H2::Matrix{ComplexF64})::Nothing
    A = @view H1[1:n,:]; C = @view H2[1:n,:]
    fill!(A, 0); fill!(C, 0)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        for j = 1:p
            Gj = @view G[1:r,j,1:n]; Zj = @view Z[1:r,j,:]
            mul!(A, adjoint(Gj), Zj, 1 + 0im, 1 + 0im)
        end
        for j = 1:p
            Gj = @view G[1:r,j,1:n]; Zj = @view Z[1:r,j,:]
            mul!(Zj, Gj, A, -1 + 0im, 1 + 0im)
        end
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
    return nothing
end

"""
    compact_block_qr!(Z::AbstractMatrix{ComplexF64}, b::Int) -> Tuple{Int,Matrix{ComplexF64}}

Compute a thin Householder QR factorization of a compact CORK block,

    Z₀ = QR,

where `Z₀` denotes the input matrix. The first `b` orthonormal columns of
`Q` overwrite `Z`, while the corresponding `b × b` upper-triangular factor
`R` is returned. The numerical block rank is estimated from the diagonal of `R` using the
threshold

    tol = max(size(Z)...) * eps(Float64) * maximum(abs.(diag(R))).

## Arguments
- `Z::AbstractMatrix{ComplexF64}`: Compact block with at least `b` columns and `b` rows; overwritten by the thin orthonormal factor `Q`.
- `b::Int`: Block-Arnoldi block size.

## Returns
- `Int`: Estimated numerical rank of the block.
- `Matrix{ComplexF64}`: Upper-triangular `b × b` factor `R`.
"""
function compact_block_qr!(Z::AbstractMatrix{ComplexF64}, b::Int)::Tuple{Int,Matrix{ComplexF64}}
    F = qr!(Z)
    R = Matrix(F.R)[1:b,1:b]
    Q = Matrix(F.Q[:,1:b])
    copyto!(Z,Q)
    d = abs.(diag(R)); tol = maximum(size(Z)) * eps(Float64) * maximum(d)
    return count(>(tol),d),R
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
    CORKState

 state of the compact block-CORK iteration.

The state stores the common physical basis `U`, cached coefficient actions
`Wⱼ=BⱼU`, compact Arnoldi basis `G`, projected block-Hessenberg matrix `Hb`,
the normalized pending Arnoldi tail, and all large scratch arrays required by
the inverse-linearization action and compact orthogonalization.

Only the first `r` columns of `U` and `W` and the first `r` rows of the
physical compact coordinates are active. The physical storage capacity is
`rmax=N`. The active compact Krylov dimension is `n`.

The workspaces `γ`, `RHS`, `tmp`, `Y`, `Hphys`, `Hphys2`, and `Zf` are
allocated once during initialization and reused by subsequent block
applications.

## Fields
- `U`: Common physical CORK basis.
- `W`: Vertically stacked cached actions `Wⱼ=BⱼU`.
- `G`: Compact Arnoldi basis coordinates.
- `Hb`: Projected block-Hessenberg matrix.
- `pendingG`: Normalized Arnoldi tail awaiting promotion.
- `Z`: Compact output/residual workspace.
- `H1`, `H2`: Compact CGS2 projection workspaces.
- `γ`: Chebyshev-recurrence workspace.
- `RHS`, `tmp`, `Y`: Physical `N × b` workspaces for the inverse action.
- `Hphys`, `Hphys2`: Physical-basis projection workspaces.
- `Zf`: Packed compact-block workspace used by Householder QR.
- `r`: Current physical basis rank.
- `n`: Current compact Krylov dimension.
- `b`: Block-Arnoldi block size.
- `p`: Chebyshev polynomial degree and number of compact linearization blocks.
- `N`: Physical Fredholm matrix dimension.
- `rmax`: Maximum physical rank.
- `pending`: Whether a normalized Arnoldi tail is currently available.
- `cache`, `lu`, `rhs`, `phys`, `orth`: Accumulated diagnostic timings.
- `napply`: Number of inverse-linearization block applications.
"""
mutable struct CORKState
    U::Matrix{ComplexF64}
    W::Matrix{ComplexF64}
    G::Array{ComplexF64,3}
    Hb::Matrix{ComplexF64}
    pendingG::Array{ComplexF64,3}
    Z::Array{ComplexF64,3}
    H1::Matrix{ComplexF64}
    H2::Matrix{ComplexF64}
    γ::Array{ComplexF64,3}
    RHS::Matrix{ComplexF64}
    tmp::Matrix{ComplexF64}
    Y::Matrix{ComplexF64}
    Hphys::Matrix{ComplexF64}
    Hphys2::Matrix{ComplexF64}
    Zf::Matrix{ComplexF64}
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
    napply::Int
end

"""
    block_apply!(S::CORKState, B::Matrix{ComplexF64}, F, Gin) -> Tuple{Float64,Float64,Float64,Float64}

Apply one block of the inverse Chebyshev linearization at the fixed shift
`t=0` in compact CORK form.

For

    P(t) = Σⱼ₌₀ᵖ BⱼTⱼ(t),

CORK represents every physical linearization block in the common basis `U`
as `zⱼ=Ugⱼ` and caches the coefficient actions `Wⱼ=BⱼU`.

For input compact blocks `x₀,...,xₚ₋₁`, the Chebyshev recurrence

    Tⱼ₊₁(t) + Tⱼ₋₁(t) = 2tTⱼ(t)

gives, at the inverse-iteration shift `t=0`,

    γ₀ = 0,
    γ₁ = x₀,
    γⱼ₊₁ = -γⱼ₋₁ + 2xⱼ.

The physical correction `y` is obtained from the single fixed factorization
of (check recurrences.jl)

    P(0) = B₀-B₂+B₄-⋯

by solving `P(0)y=rhs`. The solved block is projected twice against the
current physical basis `U`. Any remaining component orthogonal to `U` is
factorized by SVD, and numerically independent left singular vectors are
appended to the physical basis. The corresponding new coefficient actions
`BⱼU` are then cached.

## Arguments
- `S::CORKState`: CORK state containing the physical basis, coefficient cache, output tensor, and workspaces.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients `[B₀;...;Bₚ]`.
- `F`: LU factorization of the fixed shift matrix `P(0)`.
- `Gin`: Compact coordinates of the input Arnoldi block.

## Returns
- `Float64`: Time spent caching coefficient actions for newly added physical directions.
- `Float64`: Time spent solving with the LU factorization of `P(0)`.
- `Float64`: Time spent assembling the physical right-hand side.
- `Float64`: Time spent projecting and expanding the physical basis.
"""
function block_apply!(S::CORKState, B::Matrix{ComplexF64}, F, Gin)
    N = S.N; p = S.p; r = S.r; b = S.b
    p >= 2 || error("CORK Chebyshev action requires p >= 2")
    γ = @view S.γ[1:r,:,:]; RHS = S.RHS; tmp = S.tmp; Y = S.Y
    Hphys = @view S.Hphys[1:r,:]; H2 = @view S.Hphys2[1:r,:]
    fill!(γ,0); fill!(RHS,0); fill!(tmp,0); fill!(Y,0); fill!(Hphys,0); fill!(H2,0)
    @views γ[:,2,:] .= Gin[1:r,1,:]
    for j = 1:p-2
        @views @. γ[:,j + 2,:] = -γ[:,j,:] + 2Gin[1:r,j + 1,:]
    end
    t = time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        for j = 0:p-3
            mul!(tmp,@view(Wj(S.W,N,j)[:,1:r]),@view(γ[:,j + 1,:])); RHS .-= tmp
        end
        mul!(tmp,@view(Wj(S.W,N,p - 2)[:,1:r]),@view(γ[:,p - 1,:])); RHS .-= tmp
        mul!(tmp,@view(Wj(S.W,N,p)[:,1:r]),@view(γ[:,p - 1,:])); RHS .+= tmp
        mul!(tmp,@view(Wj(S.W,N,p - 1)[:,1:r]),@view(γ[:,p,:])); RHS .-= tmp
        mul!(tmp,@view(Wj(S.W,N,p)[:,1:r]),@view(Gin[1:r,p,:])); @. RHS -= 2tmp
    end
    trhs = (time_ns() - t) * 1e-9
    copyto!(Y,RHS); t = time_ns()
    @blas_multi_then_1 MAX_BLAS_THREADS ldiv!(F,Y)
    tlu = (time_ns() - t) * 1e-9; t = time_ns()
    ynorm0 = norm(Y)
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(Hphys,adjoint(@view(S.U[:,1:r])),Y)
        mul!(Y,@view(S.U[:,1:r]),Hphys,-1 + 0im,1 + 0im)
    end
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        mul!(H2,adjoint(@view(S.U[:,1:r])),Y)
        mul!(Y,@view(S.U[:,1:r]),H2,-1 + 0im,1 + 0im)
    end
    Hphys .+= H2
    ynorm = norm(Y); saturated = ynorm <= 1000eps(Float64) * max(ynorm0,1.0)
    remaining = S.rmax - r
    if remaining > 0 && !saturated
        FY = nothing
        @blas_multi_then_1 MAX_BLAS_THREADS begin
            FY = svd(Y; full=false)
        end
        σ = FY.S; σ1 = isempty(σ) ? 0.0 : σ[1]
        bp = min(count(>(max(σ1 * 1e-12,1e-14)),σ),remaining); rn = r + bp
        if bp > 0
            @views S.U[:,r + 1:rn] .= FY.U[:,1:bp]
            Cnew = Diagonal(σ[1:bp]) * FY.Vt[1:bp,:]
        else
            Cnew = zeros(ComplexF64,0,b)
        end
    else
        bp = 0; rn = r; Cnew = zeros(ComplexF64,0,b)
    end
    tphys = (time_ns() - t) * 1e-9
    tcache = bp > 0 ? cache_block!(S.W,B,S.U,N,p,r + 1,rn) : 0.0
    fill!(S.Z,0); @views S.Z[1:r,1:p,:] .+= γ; @views S.Z[1:r,1,:] .+= Hphys
    bp > 0 && (@views S.Z[r + 1:rn,1,:] .+= Cnew)
    s = -1.0
    for j = 2:2:p-1
        @views S.Z[1:r,j + 1,:] .+= s .* Hphys
        bp > 0 && (@views S.Z[r + 1:rn,j + 1,:] .+= s .* Cnew)
        s = -s
    end
    S.r = rn
    return tcache,tlu,trhs,tphys
end

"""
    init_cork(B::Matrix{ComplexF64}, p::Int, N::Int; b::Int=10) -> CORKState

Initialize the compact block-CORK state.

A deterministic `p`-dimensional orthonormal physical basis `U` is constructed
from a seeded random matrix, and the coefficient actions

    Wⱼ = BⱼU,    j=0,...,p,

are cached. A seeded random compact block is then generated in the initial
physical basis, normalized by Householder QR, and stored as the first `b`
compact Arnoldi vectors.

The maximum block-compatible Krylov dimension is mdim = floor(N/b)b,
and the physical rank is allowed to grow up to `N`.

## Arguments
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients `[B₀;...;Bₚ]`.
- `p::Int`: Chebyshev polynomial degree.
- `N::Int`: Physical Fredholm matrix dimension.
- `b::Int`: Block-Arnoldi block size.

## Returns
- `CORKState`: Initialized  block-CORK state with active Krylov dimension `b` and initial physical rank `p`.
"""
function init_cork(B::Matrix{ComplexF64}, p::Int, N::Int; b::Int=10)::CORKState
    p <= N || error("Polynomial degree p=$p exceeds physical matrix dimension N=$N")
    mdim = (N ÷ b) * b
    mdim >= b || error("Matrix dimension N=$N is smaller than block size b=$b")
    p * p >= b || error("Initial compact space has dimension p²=$(p * p), smaller than block size b=$b")
    rmax = N
    U = zeros(ComplexF64,N,rmax); W = zeros(ComplexF64,(p + 1) * N,rmax)
    G = zeros(ComplexF64,rmax,p,mdim + b); Hb = zeros(ComplexF64,mdim + b,mdim)
    pendingG = zeros(ComplexF64,rmax,p,b); Z = zeros(ComplexF64,rmax,p,b)
    H1 = zeros(ComplexF64,mdim + b,b); H2 = zeros(ComplexF64,mdim + b,b)
    γ = zeros(ComplexF64,rmax,p,b); RHS = zeros(ComplexF64,N,b); tmp = zeros(ComplexF64,N,b); Y = zeros(ComplexF64,N,b)
    Hphys = zeros(ComplexF64,rmax,b); Hphys2 = zeros(ComplexF64,rmax,b); Zf = zeros(ComplexF64,rmax * p,b)
    rng = MersenneTwister(123); X = randn(rng,ComplexF64,N,p); FX = nothing
    @blas_multi_then_1 MAX_BLAS_THREADS begin
        FX = qr(X)
    end
    @blas_multi_then_1 MAX_BLAS_THREADS @views U[:,1:p] .= FX.Q * Matrix{ComplexF64}(I,N,p)
    r = p
    Z0 = zeros(ComplexF64,rmax,p,b); @views randn!(rng,Z0[1:r,:,:])
    Z0f = @view Zf[1:r * p,:]; pack!(Z0f,Z0,r,p,b)
    bn, _ = compact_block_qr!(Z0f,b); bn == b || error("Initial block breakdown")
    fill!(Z0,0); unpack!(Z0,Z0f,r,p,b); @views G[1:r,:,1:b] .= Z0[1:r,:,:]
    tc = cache_block!(W,B,U,N,p,1,r)
    return CORKState(U,W,G,Hb,pendingG,Z,H1,H2,γ,RHS,tmp,Y,Hphys,Hphys2,Zf,r,b,b,p,N,rmax,false,tc,0.0,0.0,0.0,0.0,0)
end

"""
    compute_tail!(S::CORKState, B::Matrix{ComplexF64}, F) -> Nothing

Compute and normalize the next block-Arnoldi residual without promoting it to
the active compact basis.

The inverse Chebyshev linearization is applied to the current final Arnoldi
block by `block_apply!`. The resulting compact block is orthogonalized against
the active basis by two-pass classical Gram-Schmidt and normalized by
Householder QR. The projection coefficients and QR factor are written to the
corresponding block column of `S.Hb`.

The normalized residual is retained in `S.pendingG` and marked as pending.
This allows it to be used as the Arnoldi tail in Ritz residual estimates
before being promoted by `promote_tail!`, avoiding recomputation of the
inverse-linearization action.

## Arguments
- `S::CORKState`:  compact block-CORK state.
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients.
- `F`: LU factorization of `P(0)`.

## Returns
- `Nothing`: Updates `S.Hb`, `S.pendingG`, timing counters, and the pending-tail state in place.
"""
function compute_tail!(S::CORKState, B::Matrix{ComplexF64}, F)::Nothing
    S.pending && return nothing
    n = S.n; b = S.b; p = S.p; cols = n - b + 1:n; Gin = @view S.G[:,:,cols]
    tc, tl, tr, tp = block_apply!(S,B,F,Gin)
    S.cache += tc; S.lu += tl; S.rhs += tr; S.phys += tp
    rn = S.r; fill!(S.H1,0); fill!(S.H2,0); t = time_ns()
    compact_project!(S.Z,S.G,rn,n,p,S.H1,S.H2)
    Zf = @view S.Zf[1:rn * p,:]; pack!(Zf,S.Z,rn,p,b)
    bn, Rb = compact_block_qr!(Zf,b); bn == b || error("Block breakdown at n=$n: $bn/$b")
    fill!(S.Z,0); unpack!(S.Z,Zf,rn,p,b)
    @views S.Hb[1:n,cols] .= S.H1[1:n,:]
    @views S.Hb[n + 1:n + b,cols] .= Rb
    S.orth += (time_ns() - t) * 1e-9
    copyto!(S.pendingG,S.Z)
    S.pending = true; S.napply += 1
    return nothing
end

"""
    promote_tail!(S::CORKState) -> Nothing

Promote the normalized pending residual into the active compact Arnoldi basis.

## Arguments
- `S::CORKState`:  state containing a pending residual block.

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

Extend the  factorization to compact Krylov dimension `m`, retaining
all existing Arnoldi vectors and leaving a fresh residual tail for Ritz tests.

## Arguments
- `S::CORKState`:  CORK state.
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
- `S::CORKState`:  CORK state at dimension `m`.
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

Reconstruct physical Fredholm eigenvectors from projected CORK Ritz vectors.
For a projected Ritz vector `v`, the physical degree-zero component is

    u = U*G₀*v,

where `G₀` contains the degree-zero compact coordinates of the active CORK
basis. Each reconstructed vector is normalized to unit Euclidean norm.
...

## Arguments
- `S::CORKState`: Final  CORK state.
- `V::Matrix{ComplexF64}`: Projected Ritz vectors, one column per accepted root.
- `m::Int`: Final compact Krylov dimension.

## Returns
- `Matrix{ComplexF64}`: Normalized Fredholm eigenvectors, with column `j` corresponding to Ritz-vector column `j`.
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
        nrm = norm(@view Ψ[:,j]); nrm > 0 || error("Zero reconstructed CORK eigenvector for column $j")
        @views Ψ[:,j] ./= nrm
    end
    return Ψ
end

################################################################################
# ADAPTIVE CONVERGENCE
################################################################################

"""
    adaptive_cork(B::Matrix{ComplexF64}, F, p::Int, N::Int; k0::Float64, Δ::Float64, Δpoly::Float64, b::Int=10, mstart::Int=200, mstep::Int=100, stable_checks::Int=2, imag_tol::Float64=1e-7, edge_tol::Float64=1e-8, res_tol::Float64=1e-8, stable_tol::Float64=1e-8, imag_search_tol::Float64=1e-4, eigenvectors::Bool=false, verbose::Bool=false)

Grow one  block-CORK factorization until the accepted spectrum in
the requested interval `[k₀-Δ,k₀+Δ]` is stable and both requested spectral
edges are represented by converged Ritz roots.

At each Krylov dimension, projected inverse-linearization eigenvalues are
mapped by

    μ → t=1/μ → k=k₀+Δpoly*t.

Roots in the requested physical interval are accepted only when their
imaginary parts and projected CORK residual indicators satisfy `imag_tol` and
`res_tol`. The accepted spectrum must remain unchanged to within `stable_tol`
for `stable_checks` consecutive Krylov dimensions, and the leftmost and
rightmost requested Ritz roots must independently satisfy the same physical
convergence criteria. The Krylov dimension grows in increments of `mstep`
up to the largest block-compatible dimension

    mdim = floor(N/b)b.

## Arguments
- `B::Matrix{ComplexF64}`: Vertically stacked Chebyshev coefficients `[B₀;...;Bₚ]`.
- `F`: LU factorization of `P(0)`.
- `p::Int`: Chebyshev polynomial degree.
- `N::Int`: Physical Fredholm matrix dimension.
- `k0::Float64`: Center of the requested physical interval.
- `Δ::Float64`: Half-width of the requested physical interval.
- `Δpoly::Float64`: Guarded polynomial half-width.
- `b::Int`: Block-Arnoldi block size.
- `mstart::Int`: Initial compact Krylov dimension.
- `mstep::Int`: Krylov-dimension increment between convergence tests.
- `stable_checks::Int`: Number of consecutive stable spectra required.
- `imag_tol::Float64`: Final tolerance on `|Im(k)|`.
- `edge_tol::Float64`: Normalized tolerance used when extracting roots from the guarded polynomial interval.
- `res_tol::Float64`: Maximum projected CORK residual indicator for an accepted root.
- `stable_tol::Float64`: Maximum root displacement between consecutive accepted spectra.
- `imag_search_tol::Float64`: Loose imaginary strip used during Ritz discovery.
- `eigenvectors::Bool`: Whether to reconstruct physical Fredholm vectors after convergence.
- `verbose::Bool`: Whether to print convergence diagnostics.

## Returns
- `Tuple`: Final CORK state, accepted roots, optional physical vectors,
  requested Ritz roots, all guarded Ritz roots, requested edge roots, edge
  convergence flags, and final compact Krylov dimension.
"""
function adaptive_cork(B::Matrix{ComplexF64}, F, p::Int, N::Int; k0::Float64, Δ::Float64, Δpoly::Float64, b::Int = 10, mstart::Int = 200, mstep::Int = 100, stable_checks::Int = 2, imag_tol::Float64 = 1e-7, edge_tol::Float64 = 1e-8, res_tol::Float64 = 1e-8, stable_tol::Float64 = 1e-8, imag_search_tol::Float64 = 1e-4, eigenvectors::Bool = false, verbose::Bool = false)
    mstart % b == 0 || error("mstart must be divisible by block size")
    mstep % b == 0 || error("mstep must be divisible by block size")
    S = init_cork(B, p, N; b = b); mdim = size(S.Hb, 2); m = min(mstart, mdim)
    kmin = k0 - Δ; kmax = k0 + Δ; prev = Tuple{Float64,Float64,Float64}[]; have_prev = false; nstable = 0; t0 = time_ns()
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
        if have_prev && isempty(prev) && isempty(ks)
            drift = 0.0
        elseif have_prev && length(prev) == length(ks) && !isempty(ks)
            drift = maximum(abs(complex(ks[i][1], ks[i][2]) - complex(prev[i][1], prev[i][2])) for i = eachindex(ks))
        else
            drift = Inf
        end
        nstable = isfinite(drift) && drift <= stable_tol ? nstable + 1 : 0
        empty_ok = have_prev && isempty(prev) && isempty(requested) && isempty(ks)
        converged = nstable >= stable_checks && (edge_ok || empty_ok)
        maxρ = isempty(phys) ? Inf : maximum(x[3] for x in phys)
        status = empty_ok ? "EMPTY" : edge_ok ? "PASS" : "FAIL"
        verbose && @printf("m=%4d/%4d rank=%4d ritz=%4d conv=%4d states=%4d maxρ=%9.2e drift=%9.2e stable=%d/%d edges=%s applies=%4d time=%7.3f\n", m, mdim, S.r, length(requested), length(phys), length(ks), maxρ, drift, nstable, stable_checks, status, S.napply, (time_ns() - t0) * 1e-9)
        if converged
            Ψ = eigenvectors ? reconstruct_cork_eigenvectors(S, Matrix{ComplexF64}(Vphys), m) : nothing
            return S, ks, Ψ, requested, allroots, edges, edge_good, m
        end
        m == mdim && error("Reached full CORK Krylov dimension m=$mdim for matrix size N=$N without a stable requested spectrum.")
        prev = copy(ks); have_prev = true; m = min(m + mstep, mdim)
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
    CORKSolver(kernel::K; p::Int=16, guard::Real=0.05, nlevels::Int=200, Rmax::Real=0.8, b::Int=10, mstart::Int=30*b, mstep::Int=10*b, stable_checks::Int=1, imag_tol::Real=1e-8, edge_tol::Real=1e-8, res_tol::Real=1e-10, stable_tol::Real=1e-9, imag_search_tol::Real=1e-4, eigenvectors::Bool=false, validate::Bool=false, verbose::Bool=false, taylor_tol::Real=1e-10) where {K<:SweepBIMSolver}

Construct a Chebyshev-CORK eigensolver for a BIM Fredholm nonlinear
eigenvalue problem `A(k)u=0`.

For a requested physical interval with center `k₀` and half-width `Δ`, the
BIM operator is approximated by a degree-`p` Chebyshev matrix polynomial on
the guarded interval

    Δpoly = (1 + guard)Δ.

CORK applies a compact block-Arnoldi iteration to the inverse Chebyshev
linearization at `t=0`, reusing a single LU factorization of `P(0)`. The
Krylov dimension is increased adaptively until the accepted spectrum is
stable and the requested interval edges contain converged Ritz roots.

For multi-window spectrum calculations, `nlevels` controls the target number
of physical levels per Weyl window and `Rmax` limits the requested half-width
of each window.

## Arguments
- `kernel::K`: DLP, CFIE, or composite BIM solver defining the Fredholm operator.

## Keyword Arguments
- `p::Int=16`: Degree of the Chebyshev matrix polynomial.
- `guard::Real=0.05`: Relative enlargement of the requested interval used for polynomial construction and Ritz discovery.
- `nlevels::Int=200`: Target number of physical levels per spectral window.
- `Rmax::Real=0.8`: Maximum requested half-width `Δ` of a spectral window.
- `b::Int=10`: Block-Arnoldi block size.
- `mstart::Int=30*b`: Initial compact Krylov dimension.
- `mstep::Int=10*b`: Krylov-dimension increment between convergence tests.
- `stable_checks::Int=1`: Number of consecutive stable accepted spectra required.
- `imag_tol::Real=1e-8`: Final tolerance on `|Im(k)|` for accepted roots.
- `edge_tol::Real=1e-8`: Normalized tolerance used when extracting Ritz roots from the guarded polynomial interval.
- `res_tol::Real=1e-10`: Maximum projected CORK residual indicator for accepted roots.
- `stable_tol::Real=1e-9`: Maximum root displacement allowed between consecutive accepted spectra.
- `imag_search_tol::Real=1e-4`: Loose imaginary strip used during Ritz discovery.
- `eigenvectors::Bool=false`: Whether ordinary solve calls also reconstruct physical Fredholm vectors internally.
- `validate::Bool=false`: Whether to compare the Chebyshev polynomial with directly constructed BIM matrices before the CORK iteration.
- `verbose::Bool=false`: Whether to print detailed polynomial and CORK convergence diagnostics.
- `taylor_tol::Real=1e-10`: Maximum relative polynomial-validation error when `validate=true`.

## Returns
- `CORKSolver`: Configured Chebyshev-CORK solver.
"""
function CORKSolver(kernel::K; p::Int=16, guard::Real=0.05, nlevels::Int=200, Rmax::Real=0.8, b::Int=10, mstart::Int=30*b, mstep::Int=10*b, stable_checks::Int=1, imag_tol::Real=1e-8, edge_tol::Real=1e-8, res_tol::Real=1e-10, stable_tol::Real=1e-9, imag_search_tol::Real=1e-4, eigenvectors::Bool=false, validate::Bool=false, verbose::Bool=false, taylor_tol::Real=1e-10) where {K<:SweepBIMSolver} 
    T = _bim_numeric_type(kernel)
    T === Float64 || throw(ArgumentError("CORKSolver currently requires a Float64 BIM kernel"))
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
    _cork_solve_core(solver::CORKSolver, pts, k0, dk; multithreaded::Bool=true, eigenvectors::Bool=solver.eigenvectors)

Execute the complete Chebyshev-CORK pipeline on the interval `[k₀-dk/2,k₀+dk/2]`.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `pts`: Boundary discretization at the expansion center.
- `k0`: Center of the requested physical interval.
- `dk`: Full width of the requested physical interval.
- `multithreaded::Bool`: Whether polynomial assembly and direct validation matrices use Julia threading.
- `eigenvectors::Bool`: Whether to reconstruct final physical Fredholm vectors.

## Returns
- `Tuple`: Chebyshev polynomial, final CORK state, accepted roots, optional
  physical vectors, requested Ritz roots, all guarded Ritz roots, requested
  edge roots, edge convergence flags, and final compact Krylov dimension.
"""
function _cork_solve_core(solver::CORKSolver, pts, k0, dk; multithreaded::Bool=true, eigenvectors::Bool=solver.eigenvectors)
    k0f = Float64(k0); Δ = Float64(dk) / 2
    Δpoly = Δ * (1 + solver.guard)
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
    S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal = adaptive_cork(P.B, F, solver.p, P.N; k0=k0f, Δ=Δ, Δpoly=Δpoly, b=solver.b, mstart=solver.mstart, mstep=solver.mstep, stable_checks=solver.stable_checks, imag_tol=solver.imag_tol, edge_tol=solver.edge_tol, res_tol=solver.res_tol, stable_tol=solver.stable_tol, imag_search_tol=solver.imag_search_tol, eigenvectors=eigenvectors, verbose=solver.verbose)
    return P, S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal
end

"""
    solve(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)

Compute the accepted CORK roots in `[k₀-dk/2,k₀+dk/2]` using an existing
boundary discretization.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `pts::BoundaryPoints`: Boundary discretization at the expansion center.
- `k0`: Center of the requested physical interval.
- `dk`: Full width of the requested physical interval.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Vector{ComplexF64}`: Accepted physical wavenumbers.
- `Vector{Float64}`: Corresponding projected CORK residual indicators.
"""
function solve(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    P, S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal = _cork_solve_core(solver, pts, k0, dk; multithreaded = multithreaded)
    λ = ComplexF64[complex(x[1], x[2]) for x in ks]; ts = Float64[x[3] for x in ks]
    return λ, ts
end

"""
    solve_vectors(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)

Compute the accepted CORK roots and their corresponding layer densities in
`[k₀-dk/2,k₀+dk/2]`.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `pts::BoundaryPoints`: Boundary discretization at the expansion center.
- `k0`: Center of the requested physical interval.
- `dk`: Full width of the requested physical interval.
- `multithreaded::Bool=true`: Whether polynomial assembly uses Julia threads.

## Returns
- `Vector{ComplexF64}`: Accepted physical wavenumbers.
- `Vector{Float64}`: Corresponding projected CORK residual indicators.
- `Matrix{ComplexF64}`: Corresponding layer densities stored column-wise.
"""
function solve_vectors(solver::CORKSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true)
    P, S, ks, Ψ, requested, allroots, edge_roots, edge_good, mfinal = _cork_solve_core(solver, pts, k0, dk; multithreaded, eigenvectors=true)
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

Compute the complete accepted CORK spectrum in `[k-dk/2,k+dk/2]`,
constructing boundary points at the expansion center `k`.

Each accepted projected Ritz root is returned individually; nearby roots are
not merged or expanded according to an inferred multiplicity.

## Arguments
- `solver::CORKSolver`: Configured CORK eigensolver.
- `billiard::Bi`: Billiard geometry.
- `k`: Requested interval and polynomial expansion center.
- `dk`: Full requested spectral width.
- `multithreaded::Bool`: Whether polynomial assembly uses Julia threads.

## Returns
- `Tuple{Vector{ComplexF64},Vector{Float64}}`: Accepted wavenumbers and their
  CORK residual estimates.
"""
function solve_spectrum(solver::CORKSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k)
    return solve(solver, pts, k, dk; multithreaded = multithreaded)
end