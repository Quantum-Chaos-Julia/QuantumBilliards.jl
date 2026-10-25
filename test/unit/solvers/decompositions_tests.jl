# Unit tests for
#   src/solvers/decompositions.jl   (generalized_eigen, generalized_eigvals, generalized_eigen_all, directsum)
# Pencils are either diagonal (eigenvalues known in closed form) or random symmetric-definite, checked against
# LAPACK's generalized symmetric-definite solver and against residual / orthogonality identities.
#
# Convention of generalized_eigen / generalized_eigvals (A symmetric positive semidefinite, B symmetric):
# the returned mu are the eigenvalues of the pencil  B x = mu A x,  x = C_scaled * z,  so that
#   X' A X = I   and   X' B X = Diagonal(mu).
# They are therefore the reciprocals of the eigenvalues of  A x = λ B x  quoted in the docstrings.

using LinearAlgebra
using Random
using QuantumBilliards: generalized_eigen, generalized_eigvals, generalized_eigen_all, directsum

# Random symmetric positive-definite A and symmetric B (well conditioned).
function dec_random_pencil(rng, n)
    M = randn(rng, n, n)
    A = M'*M + n*I
    N = randn(rng, n, n)
    B = (N + N')/2
    return Matrix(Symmetric(A)), Matrix(Symmetric(B))
end

@testset "generalized_eigen on a diagonal pencil" begin
    a = [1.0, 2.0, 4.0]
    b = [2.0, 6.0, 1.0]
    A = Matrix(Diagonal(a))
    B = Matrix(Diagonal(b))
    mu, Z, C = generalized_eigen(A, B)

    @test mu ≈ sort(b ./ a)                                              # eigenvalues of B x = mu A x
    @test issorted(mu)
    @test size(Z) == (3, 3)
    @test size(C) == (3, 3)

    X = C*Z
    @test X'*A*X ≈ Matrix(1.0I, 3, 3) atol=1e-12                          # A-orthonormal
    @test X'*B*X ≈ Diagonal(mu) atol=1e-12
    @test B*X ≈ A*X*Diagonal(mu) atol=1e-12                               # pencil residual
    @test Z'*Z ≈ Matrix(1.0I, 3, 3) atol=1e-12                            # Z diagonalizes a symmetric matrix

    # identity pencil: every mu is one
    mu_id, _, _ = generalized_eigen(Matrix(1.0I, 4, 4), Matrix(1.0I, 4, 4))
    @test mu_id ≈ ones(4)
end

@testset "generalized_eigen on a random symmetric-definite pencil" begin
    rng = Xoshiro(20240601)
    n = 8
    A, B = dec_random_pencil(rng, n)
    mu, Z, C = generalized_eigen(A, B)
    X = C*Z

    @test mu ≈ eigvals(Symmetric(B), Symmetric(A)) rtol=1e-9              # LAPACK reference for B x = mu A x
    @test size(C) == (n, n)
    @test norm(B*X - A*X*Diagonal(mu)) <= 1e-10*norm(B)
    @test X'*A*X ≈ Matrix(1.0I, n, n) atol=1e-10
    @test X'*B*X ≈ Diagonal(mu) atol=1e-10

    # eigenvalue count and scaling of the pencil: (A, cB) has eigenvalues c*mu
    mu2, _, _ = generalized_eigen(A, 3.0*B)
    @test mu2 ≈ 3.0*mu rtol=1e-9
    # (cA, B) has eigenvalues mu/c
    mu3, _, _ = generalized_eigen(2.0*A, B)
    @test mu3 ≈ mu/2 rtol=1e-9
end

@testset "generalized_eigen truncation of small eigenvalues of A" begin
    @testset "exact null space is removed" begin
        A = Matrix(Diagonal([4.0, 1.0, 0.0]))
        B = Matrix(Diagonal([2.0, 3.0, 5.0]))
        mu, Z, C = generalized_eigen(A, B)
        @test mu ≈ [0.5, 3.0]                                            # b/a on the retained directions
        @test size(Z) == (2, 2)
        @test size(C) == (3, 2)
    end

    @testset "eps sets the relative cut-off" begin
        A = Matrix(Diagonal([1.0, 1e-8, 1e-12]))
        B = Matrix(1.0I, 3, 3)
        mu_all, _, C_all = generalized_eigen(A, B)                        # default eps = 1e-15 keeps everything
        @test size(C_all, 2) == 3
        @test mu_all ≈ [1.0, 1e8, 1e12]

        mu, _, C = generalized_eigen(A, B; eps=1e-10)                     # drops 1e-12 < 1e-10 * 1
        @test size(C, 2) == 2
        @test mu ≈ [1.0, 1e8]

        mu, _, C = generalized_eigen(A, B; eps=1e-5)                      # drops 1e-8 as well
        @test size(C, 2) == 1
        @test mu ≈ [1.0]
    end

    @testset "non-positive eigenvalues of A are discarded" begin
        A = Matrix(Diagonal([2.0, -1.0, 1.0]))
        B = Matrix(Diagonal([4.0, 4.0, 4.0]))
        mu, _, C = generalized_eigen(A, B)
        @test mu ≈ [2.0, 4.0]                                            # 4/2 and 4/1
        @test size(C) == (3, 2)
    end

    @testset "all directions removed gives empty output" begin
        mu, Z, C = generalized_eigen(zeros(3, 3), Matrix(1.0I, 3, 3))
        @test isempty(mu)
        @test size(Z) == (0, 0)
        @test size(C) == (3, 0)
    end
end

@testset "generalized_eigen element types and argument errors" begin
    A = Matrix(Diagonal([1.0, 2.0, 4.0]))
    B = Matrix(Diagonal([2.0, 2.0, 2.0]))

    @testset "Float32 input stays Float32" begin
        mu, Z, C = generalized_eigen(Float32.(A), Float32.(B))
        @test mu isa Vector{Float32}
        @test Z isa Matrix{Float32}
        @test C isa Matrix{Float32}
        @test mu ≈ Float32[0.5, 1.0, 2.0] atol=1f-5
    end

    @testset "inferred" begin
        @test @inferred(generalized_eigen(A, B)) isa Tuple{Vector{Float64},Matrix{Float64},Matrix{Float64}}
    end

    @testset "shape errors" begin
        @test_throws DimensionMismatch generalized_eigen(ones(3, 4), ones(3, 3))      # A not square
        @test_throws DimensionMismatch generalized_eigen(A, ones(4, 4))               # B size differs from A
    end
end

@testset "generalized_eigvals" begin
    @testset "diagonal pencil" begin
        a = [1.0, 2.0, 4.0]
        b = [2.0, 6.0, 1.0]
        mu = generalized_eigvals(Matrix(Diagonal(a)), Matrix(Diagonal(b)))
        @test mu ≈ sort(b ./ a)
        @test mu isa Vector{Float64}
    end

    @testset "agrees with generalized_eigen and with LAPACK on a random pencil" begin
        rng = Xoshiro(7)
        A, B = dec_random_pencil(rng, 8)
        mu = generalized_eigvals(A, B)
        @test mu ≈ generalized_eigen(A, B)[1] rtol=1e-10
        @test mu ≈ eigvals(Symmetric(B), Symmetric(A)) rtol=1e-9
    end

    @testset "truncation matches generalized_eigen" begin
        A = Matrix(Diagonal([1.0, 1e-8, 1e-12]))
        B = Matrix(1.0I, 3, 3)
        @test generalized_eigvals(A, B) ≈ [1.0, 1e8, 1e12]
        @test generalized_eigvals(A, B; eps=1e-10) ≈ [1.0, 1e8]
        @test generalized_eigvals(A, B; eps=1e-5) ≈ [1.0]
        @test generalized_eigvals(Matrix(Diagonal([4.0, 1.0, 0.0])), Matrix(Diagonal([2.0, 3.0, 5.0]))) ≈ [0.5, 3.0]
        @test isempty(generalized_eigvals(zeros(3, 3), Matrix(1.0I, 3, 3)))
    end

    @testset "Float32 input stays Float32" begin
        mu = generalized_eigvals(Float32.(Matrix(Diagonal([1.0, 2.0]))), Float32.(Matrix(Diagonal([2.0, 2.0]))))
        @test mu isa Vector{Float32}
        @test mu ≈ Float32[1.0, 2.0] atol=1f-5
    end

    @testset "shape errors" begin
        @test_throws DimensionMismatch generalized_eigvals(ones(3, 4), ones(3, 3))
        @test_throws DimensionMismatch generalized_eigvals(Matrix(1.0I, 3, 3), ones(4, 4))
    end
end

@testset "generalized_eigen_all" begin
    @testset "diagonal pencil" begin
        A = Matrix(Diagonal([3.0, 2.0, 5.0]))
        B = Matrix(Diagonal([1.0, 1.0, 2.0]))
        λ, VR, VL = generalized_eigen_all(A, B)
        @test λ ≈ [2.0, 2.5, 3.0]                                         # a/b, ordered by modulus
        @test size(VR) == (3, 3) && size(VL) == (3, 3)
        @test A*VR ≈ B*VR*Diagonal(λ) atol=1e-12
    end

    @testset "non-symmetric pencil: right and left eigenvectors, modulus ordering" begin
        A = [3.0 1.0 0.0 0.0; 0.0 -2.0 1.0 0.0; 0.0 0.0 1.0 1.0; 0.0 0.0 0.0 -5.0]   # triangular: λ = diagonal
        B = Matrix(1.0I, 4, 4)
        λ, VR, VL = generalized_eigen_all(A, B)
        @test λ ≈ [1.0, -2.0, 3.0, -5.0]
        @test issorted(abs.(λ))
        @test norm(A*VR - B*VR*Diagonal(λ)) <= 1e-12                       # right: A v = λ B v
        @test norm(VL'*A - Diagonal(conj.(λ))*VL'*B) <= 1e-12             # left:  w' A = λ w' B
    end

    @testset "complex eigenvalues of a real pencil" begin
        A = [0.0 -1.0; 1.0 0.0]                                           # rotation by π/2: λ = ±i
        λ, VR, VL = generalized_eigen_all(A, Matrix(1.0I, 2, 2))
        @test eltype(λ) <: Complex
        @test sort(imag.(λ)) ≈ [-1.0, 1.0]
        @test all(abs.(real.(λ)) .< 1e-14)
        @test A*VR ≈ VR*Diagonal(λ) atol=1e-12
    end

    @testset "infinite and undefined eigenvalues are filtered" begin
        # B singular: second eigenvalue is infinite
        λ, VR, VL = generalized_eigen_all(Matrix(Diagonal([2.0, 3.0])), Matrix(Diagonal([1.0, 0.0])))
        @test λ ≈ [2.0]
        @test size(VR) == (2, 1) && size(VL) == (2, 1)
        # A and B share a null vector: second eigenvalue is 0/0 = NaN
        λ, VR, VL = generalized_eigen_all(Matrix(Diagonal([2.0, 0.0])), Matrix(Diagonal([1.0, 0.0])))
        @test λ ≈ [2.0]
        @test size(VR) == (2, 1) && size(VL) == (2, 1)
        @test all(isfinite, λ)
    end

    @testset "shape errors" begin
        @test_throws DimensionMismatch generalized_eigen_all(ones(3, 4), ones(3, 4))
        @test_throws DimensionMismatch generalized_eigen_all(Matrix(1.0I, 3, 3), Matrix(1.0I, 4, 4))
    end
end

@testset "directsum" begin
    @testset "block structure" begin
        A = [1 2; 3 4]
        B = [5 6 7]
        M = directsum(A, B)
        @test size(M) == (3, 5)
        @test M == [1 2 0 0 0; 3 4 0 0 0; 0 0 5 6 7]
        @test M[1:2, 1:2] == A
        @test M[3:3, 3:5] == B
        @test all(iszero, M[1:2, 3:5])
        @test all(iszero, M[3:3, 1:2])
    end

    @testset "eigenvalues of a direct sum are the union" begin
        A = [2.0 1.0; 1.0 2.0]
        B = [5.0 0.0; 0.0 -1.0]
        @test sort(eigvals(directsum(A, B))) ≈ sort(vcat(eigvals(A), eigvals(B)))
    end

    @testset "empty and mixed element types" begin
        @test directsum(zeros(0, 0), [1 2]) == [1 2]
        @test directsum([1.0 2.0], [1 2; 3 4]) isa Matrix{Float64}
        @test_throws MethodError directsum(Diagonal([1.0, 2.0]), [1.0])       # defined for dense matrices only
    end
end
