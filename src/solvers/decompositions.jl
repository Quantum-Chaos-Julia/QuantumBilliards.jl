"""
    generalized_eigen(A::Symmetric, B::Symmetric; eps::Real = 1e-15) → (mu::Vector, Z::Matrix, C_scaled::Matrix)

Computes the generalized eigenvalues and eigenvectors of the system `A * x = λ * B * x`
using a truncated basis where eigenvalues of `A` smaller than `eps * max(eigenvalues(A))` 
are ignored. This optimized implementation minimizes memory allocations.

## Description
Reference: https://users.flatironinstitute.org/~ahb/thesis_html/node60.html

| Step                                      | Code Line                                                                                     | Explanation                                                                                                      |
|-------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Diagonalize A                             | `d, S = eigen(Symmetric(A))`                                                                  | Compute the eigenvalues (`d`) and eigenvectors (`S`) of `A`.                                                    |
| Truncate eigenvectors                     | `idx = d .> eps * maximum(d)`                                                                | Identify eigenvalues greater than `eps * max(eigenvalues(A))`.                                                  |
| Construct truncated eigenvector matrix C  | `C = @view S[:, idx]`                                                                         | Extract eigenvectors corresponding to retained eigenvalues.                                                     |
| Scale eigenvectors by Λ^(-1/2)            | `q = 1.0 ./ sqrt.(d[idx]); C_scaled = C .* q'`                                                | Scale selected eigenvectors by the inverse square root of the retained eigenvalues.                             |
| Form reduced matrix E                     | `mul!(tmp, B, C_scaled); mul!(E, C_scaled', tmp); E = Symmetric(E)`                            | Compute the reduced matrix `E = Λ^(-1/2) * C' * B * C * Λ^(-1/2)`.                                              |
| Solve reduced eigenproblem                | `mu, Z = eigen(Symmetric(E))`                                                                | Solve the reduced eigenproblem for eigenvalues (`mu`) and eigenvectors (`Z`).                                   |

## Arguments
* `A`: Symmetric matrix `A` in the generalized eigenproblem.
* `B`: Symmetric matrix `B` in the generalized eigenproblem.

## Keyword arguments
* `eps::Real = 1e-15`: Relative tolerance for filtering small eigenvalues of `A`.

## Returns
* `mu`: Vector of generalized eigenvalues.
* `Z`: Matrix of eigenvectors in the reduced space.
* `C_scaled`: Scaled eigenvector matrix corresponding to the truncated basis.
"""
function generalized_eigen(A,B;eps=1e-15)
    @debug "Generalized eigen decomposition started." 
    @timeit_debug "generalized_eigen" begin
        @debug "First decomposition matrix size." size=size(A)
        @timeit_debug "First decomposition" begin
            d,S=eigen(Symmetric(A))
        end
        idx=d.>eps*maximum(d)
        q=1.0./sqrt.(d[idx])
        C=@view S[:,idx]
        C_scaled=similar(C,size(C)...)
        scale_cols!(C_scaled,C,q)
        n=size(C_scaled,2)
        tmp=Matrix{eltype(B)}(undef,size(B,1),n)
        E=Matrix{eltype(B)}(undef,n,n)
        mul!(tmp,B,C_scaled)
        mul!(E,C_scaled',tmp)
        @debug "Second decomposition matrix size." size=size(E)
        @timeit_debug "Second decomposition" begin
            mu,Z=eigen!(Symmetric(E))
        end
        return mu,Z,C_scaled
    end
end

"""
    generalized_eigvals(A::Symmetric, B::Symmetric; eps::Real = 1e-15) → mu::Vector

Computes the generalized eigenvalues of the system `A * x = λ * B * x`
using a truncated basis where eigenvalues of `A` smaller than `eps * max(eigenvalues(A))` 
are ignored.

## Description
Reference: https://users.flatironinstitute.org/~ahb/thesis_html/node60.html

| Step                                      | Code Line                                                                                     | Explanation                                                                                                      |
|-------------------------------------------|-----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Diagonalize A                             | `d, S = eigen(A)`                                                                             | Compute the eigenvalues (`d`) and eigenvectors (`S`) of `A`.                                                    |
| Truncate eigenvectors                     | `idx = d .> eps * maxd`                                                                       | Identify eigenvalues greater than `eps * maxd`.                                                                 |
| Construct truncated eigenvector matrix C  | `C = @view S[:, idx]` or `C = S[:, idx]`                                                     | Extract eigenvectors corresponding to retained eigenvalues.                                                     |
| Scale eigenvectors by Lambda_r^(-1/2)     | `q = 1.0 ./ sqrt.(d[idx]); C_scaled = C .* q'`                                                | Scale selected eigenvectors by the inverse square root of the retained eigenvalues.                             |
| Form reduced matrix G'_r                  | `mul!(tmp, B, C_scaled); mul!(E, C_scaled', tmp); E = Symmetric(E)`                            | Compute the reduced matrix `G'_r = Lambda_r^(-1/2) * V_r^T * G * V_r * Lambda_r^(-1/2)`.                        |
| Solve reduced eigenproblem                | `return eigvals(E)`                                                                           | Solve the reduced eigenproblem for the eigenvalues of the system.                                               |

## Arguments
* `A`: Symmetric matrix `A` in the generalized eigenproblem.
* `B`: Symmetric matrix `B` in the generalized eigenproblem.

## Keyword arguments
* `eps::Real = 1e-15`: Relative tolerance for filtering small eigenvalues of `A`.

## Returns
* `mu`: Vector of generalized eigenvalues.
"""
function generalized_eigvals(A,B;eps=1e-15)
    @debug "Generalized eigenvals decomposition started." 
    @timeit_debug "generalized_eigenvals" begin
        @debug "First decomposition matrix size." size=size(A)
        @timeit_debug "First decomposition" begin    
            d,S=eigen(Symmetric(A))
        end
        maxd=maximum(d)
        idx=d.>eps*maxd 
        q=1.0./sqrt.(d[idx])
        C=@view S[:,idx]
        C_scaled=similar(C,size(C)...)
        scale_cols!(C_scaled,C,q)
        n=size(C_scaled,2)
        tmp=Matrix{eltype(B)}(undef,size(B,1),n)  
        E=Matrix{eltype(B)}(undef,n,n) 
        mul!(tmp,B,C_scaled)
        mul!(E,C_scaled',tmp)
        @debug "Second decomposition matrix size." size=size(E)
        @timeit_debug "Second decomposition" begin
            return eigvals!(Symmetric(E))
        end
    end  
end

"""
    generalized_eigen_all(A::AbstractMatrix, B::AbstractMatrix) → (λ::Vector{Complex{T}}, VR::Matrix{Complex{T}}, VL::Matrix{Complex{T}}) where T <: Real

Computes the generalized eigenvalues and both left and right eigenvectors of the pair of matrices `(A, B)`. There are no further restrictions on the types of matrices `(A, B)`.

## Description
```math
A * u = λ * B * u     ->     A * u = λ * dA/dk * u
```

It is important to filter the eigenvalues λ for Inf or NaN since eigen internally calls ggev3/ggev which uses QZ algorithm which gives the diagonal elements of the triangular form of A and B (they are simulatenously transformed as T_A = Q * A * Z & T_B = Q * B * Z) as vectors α (diagonals of T_A) and vectors β (diagonals of T_B). The key observation here is that matrix B in EBIM is ill-conditioned (cond(B) > 1e16) and singular (since the diagonals are all zero in using the helmholtz kernel) and therefore the QZ algorithm returns many Inf values for β. And when we construct the final eigenvalues as λ=α./β this becomes problematic, hence the need to for this check. Only when we are close to an actual eigenvalue do we have generalized eigenvalues in the dk range where the problem is constructed.

## Arguments
* `A`: Square matrix.
* `B`: Square matrix.

## Returns
* `λ`: Vector of ordered filtered eigenvalues (excluding `NaN` and `Inf` values).
* `VR`: Complex matrix where each column is a right eigenvector.
* `VL`: Complex matrix where each column is a left eigenvector.
"""
function generalized_eigen_all(A,B)
    @timeit_debug "generalized_eigen_all" begin
        @debug "Generalized eigen all decomposition started."
        @debug "Matrix size." size=size(A)
        @debug "Computing right eigenvectors."
        @timeit_debug "Right decomposition" begin 
            F=eigen(A,B)
        end
        λ=F.values
        VR=F.vectors # right eigenvectors
        @debug "Computing left eigenvectors."
        @timeit_debug "Left decomposition" begin 
            F_adj=eigen(A',B') # adjoint problem to find left eigenvectors
        end
        VL=F_adj.vectors 
        valid_indices=.!isnan.(λ).&.!isinf.(λ)  # for singular matrices give NaN λ
        λ=λ[valid_indices]
        VR=VR[:,valid_indices]
        VL=VL[:,valid_indices]
        sort_order=sortperm(abs.(λ)) 
        λ=λ[sort_order]
        VR=VR[:,sort_order]
        VL=VL[:,sort_order]
        return λ,VR,VL
    end
end

"""
    directsum(A::Matrix, B::Matrix) → M::Matrix

Constructs the direct sum of two matrices `A` and `B`. The result is a block diagonal matrix 
where `A` occupies the top-left block, `B` occupies the bottom-right block, and the off-diagonal 
blocks are filled with zeros.

```math
⎡A (m × n)  0 (m × q)⎤
⎣0 (p × n)  B (p × q)⎦
```

## Arguments
* `A`: A matrix of size `m × n`.
* `B`: A matrix of size `p × q`.

## Returns
* `M`: A block matrix of size `(m + p) × (n + q)`.
"""
directsum(A::Matrix,B::Matrix) = [A zeros(size(A,1), size(B,2)); zeros(size(B,1), size(A,2)) B]

"""
    adjust_scaling_and_samplers(solver::AbsBasisSolver, billiard::AbsBilliard) → (bs::Vector, samplers::Vector{<:AbsSampler})

Adjusts the scaling factors and samplers of the solver to match the number of fundamental 
boundary curves in the billiard (for each curve one `b` and sampler). This ensures that the solver has the appropriate number of 
scaling factors and samplers, filling in defaults where necessary.

## Arguments
* `solver`: The solver whose scaling factors and samplers need adjustment.
* `billiard`: The billiard object, which defines the fundamental boundary curves.

## Returns
* `bs`: The adjusted vector of scaling factors, with at least as many entries as fundamental boundary curves. Missing entries are filled with `minimum(solver.pts_scaling_factor)`; surplus entries are kept unchanged.
* `samplers`: The adjusted vector of samplers, with at least as many entries as fundamental boundary curves. Missing entries are filled with `solver.sampler[1]`; surplus entries are kept unchanged.
"""
function adjust_scaling_and_samplers(solver::AbsBasisSolver, billiard::AbsBilliard)
    bs = solver.pts_scaling_factor
    samplers = solver.sampler
    default = samplers[1]
    n_curves = length(get_boundary_curves(billiard))

    b_min = minimum(bs)
    while length(bs)<n_curves
        push!(bs, b_min)
    end
    
    while length(samplers)<n_curves
        push!(samplers, default)
    end
    return bs, samplers
end