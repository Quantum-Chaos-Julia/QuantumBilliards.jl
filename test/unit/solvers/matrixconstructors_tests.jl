# Unit tests for
#   src/solvers/matrixconstructors.jl   (filter_matrix!, _scale_rows!, scale_cols!, _scale_rows_sqrtw!,
#                                        _symmetrize_from_upper!, _build_Bn_inplace!)
# The in-place building blocks of the quadratic-form matrices of the decomposition method and the Vergini-Saraceno method,
# F = B' W B with W = diag(w), assembled as syrk(sqrt(nsym W) B) on the upper triangle and mirrored. Every helper is compared
# with its direct dense formula on small hand-built matrices; the basis-level wrappers basis_matrix, dk_matrix,
# gradient_matrices and basis_and_gradient_matrices are covered in test/unit/basis/basisinterface_tests.jl.
# No eigen solves, no basis evaluation. Helper definitions use the prefix `mc_`.

using LinearAlgebra: BLAS, Diagonal, issymmetric
using StaticArrays: SVector
using QuantumBilliards: filter_matrix!, _scale_rows!, scale_cols!, _scale_rows_sqrtw!, _symmetrize_from_upper!,
                        _build_Bn_inplace!

# Fixed rectangular test matrix with distinct entries and a fixed positive weight vector.
mc_matrix(m, n) = [sin(1.3i + 0.7j) + 0.1i - 0.05j for i in 1:m, j in 1:n]
mc_weights(m) = [1.0 + 0.25i for i in 1:m]

@testset "filter_matrix!" begin
    @testset "entries at or below the threshold are set to zero" begin
        M = [1.0 1e-20; -1e-20 2.0]
        out = filter_matrix!(M; ϵ=1e-10)
        @test out === M                                                   # filtered in place
        @test M == [1.0 0.0; 0.0 2.0]
    end

    @testset "the threshold is inclusive" begin
        M = [0.5 -0.5; 0.75 -0.625]
        filter_matrix!(M; ϵ=0.5)
        @test M == [0.0 0.0; 0.75 -0.625]
    end

    @testset "default threshold is machine epsilon" begin
        M = [eps(Float64) / 2, eps(Float64), 2 * eps(Float64), 1.0]
        filter_matrix!(M)
        @test M == [0.0, 0.0, 2 * eps(Float64), 1.0]
        M32 = Float32[eps(Float32), 2 * eps(Float32), 1]
        filter_matrix!(M32)
        @test M32 == Float32[0, 2 * eps(Float32), 1]
        @test eltype(M32) == Float32
    end

    @testset "complex entries are filtered by modulus" begin
        M = ComplexF64[1e-20 + 1e-20im, 1.0im, 3.0 + 0.0im]
        filter_matrix!(M; ϵ=1e-10)
        @test M == ComplexF64[0, 1.0im, 3.0]
    end

    @testset "entries above the threshold and the element type are untouched" begin
        M = mc_matrix(4, 3)
        M0 = copy(M)
        filter_matrix!(M; ϵ=0.0)                                          # only exact zeros are removed
        @test M == M0
        @test eltype(M) == Float64
    end
end

@testset "_scale_rows!" begin
    A = mc_matrix(5, 3)
    w = mc_weights(5)
    A0 = copy(A)
    out = _scale_rows!(A, w)
    @test out === A
    @test A ≈ Diagonal(w) * A0
    @test all(A[i, :] ≈ w[i] .* A0[i, :] for i in 1:5)

    @testset "unit weights leave the matrix unchanged" begin
        B = mc_matrix(4, 4)
        B0 = copy(B)
        _scale_rows!(B, ones(4))
        @test B == B0
    end

    @testset "a zero weight zeroes the row" begin
        B = mc_matrix(3, 2)
        _scale_rows!(B, [1.0, 0.0, 1.0])
        @test all(iszero, B[2, :])
        @test !iszero(B[1, 1])
    end
end

@testset "scale_cols!" begin
    X = mc_matrix(4, 3)
    s = [2.0, -1.0, 0.5]

    @testset "separate destination" begin
        Y = zeros(4, 3)
        out = scale_cols!(Y, X, s)
        @test out === Y
        @test Y ≈ X * Diagonal(s)
        @test X == mc_matrix(4, 3)                                         # source untouched
    end

    @testset "in place, destination aliases the source" begin
        Z = copy(X)
        scale_cols!(Z, Z, s)
        @test Z ≈ X * Diagonal(s)
    end

    @testset "length of the scale vector must match the column count" begin
        @test_throws AssertionError scale_cols!(zeros(4, 3), X, [1.0, 2.0])
        @test_throws AssertionError scale_cols!(zeros(4, 3), X, [1.0, 2.0, 3.0, 4.0])
    end
end

@testset "_scale_rows_sqrtw!" begin
    A0 = mc_matrix(6, 3)
    w = mc_weights(6)

    @testset "rows are multiplied by sqrt(nsym w_i)" begin
        A = copy(A0)
        out = _scale_rows_sqrtw!(A, w, 1.0)
        @test out === A
        @test A ≈ Diagonal(sqrt.(w)) * A0
        B = copy(A0)
        _scale_rows_sqrtw!(B, w, 4.0)
        @test B ≈ Diagonal(sqrt.(4.0 .* w)) * A0
        @test B ≈ 2.0 .* A                                                 # nsym enters as sqrt(nsym)
    end

    @testset "the Gram matrix of the scaled matrix is nsym A' W A" begin
        for nsym in (1.0, 2.0, 4.0)
            A = copy(A0)
            _scale_rows_sqrtw!(A, w, nsym)
            @test A' * A ≈ nsym * (A0' * Diagonal(w) * A0)
        end
    end

    @testset "integer symmetry count is promoted" begin
        A = copy(A0)
        _scale_rows_sqrtw!(A, w, 4)
        @test A ≈ Diagonal(sqrt.(4.0 .* w)) * A0
    end

    @testset "negative weights have no real square root" begin
        @test_throws DomainError _scale_rows_sqrtw!(copy(A0), [1.0, -1.0, 1.0, 1.0, 1.0, 1.0], 1.0)
    end

    @testset "zero weight removes the row" begin
        A = copy(A0)
        _scale_rows_sqrtw!(A, [1.0, 0.0, 1.0, 1.0, 1.0, 1.0], 1.0)
        @test all(iszero, A[2, :])
    end
end

@testset "_symmetrize_from_upper!" begin
    @testset "the strict lower triangle is replaced by the mirrored upper triangle" begin
        S = mc_matrix(5, 5)
        U = copy(S)
        out = _symmetrize_from_upper!(S)
        @test out === S
        @test issymmetric(S)
        @test all(S[i, j] == U[i, j] for (i, j) in Iterators.product(1:5, 1:5) if i <= j)   # upper triangle and diagonal are kept
        @test all(S[j, i] == U[i, j] for (i, j) in Iterators.product(1:5, 1:5) if i < j)
    end

    @testset "an already symmetric matrix is unchanged" begin
        A = mc_matrix(4, 4)
        S = A + A'
        S0 = copy(S)
        _symmetrize_from_upper!(S)
        @test S == S0
    end

    @testset "one-by-one matrix" begin
        S = fill(3.0, 1, 1)
        _symmetrize_from_upper!(S)
        @test S == fill(3.0, 1, 1)
    end

    @testset "strided views are accepted" begin
        P = mc_matrix(6, 6)
        V = view(P, 2:5, 2:5)
        U = copy(V)
        _symmetrize_from_upper!(V)
        @test issymmetric(V)
        @test V[1, 3] == U[1, 3] && V[3, 1] == U[1, 3]
        @test P[1, :] == mc_matrix(6, 6)[1, :]                             # outside the view nothing changes
    end

    @testset "complex matrices are mirrored without conjugation" begin
        S = ComplexF64[1 2+3im; 0 4]
        _symmetrize_from_upper!(S)
        @test S[2, 1] == 2 + 3im
    end
end

@testset "syrk assembly of F = B' W B" begin
    # The path used by construct_matrices of the decomposition and Vergini-Saraceno solvers:
    # scale the rows by sqrt(nsym w), update the upper triangle with syrk, mirror it.
    B0 = mc_matrix(8, 4)
    w = mc_weights(8)

    @testset "equals the dense weighted Gram matrix" begin
        for nsym in (1.0, 4.0)
            B = copy(B0)
            _scale_rows_sqrtw!(B, w, nsym)
            F = Matrix{Float64}(undef, 4, 4)
            BLAS.syrk!('U', 'T', 1.0, B, 0.0, F)
            _symmetrize_from_upper!(F)
            @test F ≈ nsym * (B0' * Diagonal(w) * B0)
            @test issymmetric(F)
        end
    end

    @testset "the Gram matrix is positive semi-definite for positive weights" begin
        B = copy(B0)
        _scale_rows_sqrtw!(B, w, 1.0)
        F = Matrix{Float64}(undef, 4, 4)
        BLAS.syrk!('U', 'T', 1.0, B, 0.0, F)
        _symmetrize_from_upper!(F)
        x = [1.0, -2.0, 0.5, 3.0]
        @test x' * F * x ≈ sum(w .* (B0 * x) .^ 2)
        @test x' * F * x > 0
    end

    @testset "syr2k assembly of the mixed matrix B' W dB + dB' W B" begin
        dB0 = mc_matrix(8, 4) .+ 0.3
        B = copy(B0)
        dB = copy(dB0)
        _scale_rows_sqrtw!(B, w, 1.0)
        _scale_rows_sqrtw!(dB, w, 1.0)
        Fk = Matrix{Float64}(undef, 4, 4)
        BLAS.syr2k!('U', 'T', 1.0, B, dB, 0.0, Fk)
        _symmetrize_from_upper!(Fk)
        W = Diagonal(w)
        @test Fk ≈ B0' * W * dB0 + dB0' * W * B0
        @test issymmetric(Fk)
    end
end

@testset "_build_Bn_inplace!" begin
    dX = mc_matrix(5, 3)
    dY = mc_matrix(5, 3) .+ 0.2
    normals = [SVector(cos(0.4j), sin(0.4j)) for j in 1:5]
    dX0 = copy(dX)
    dY0 = copy(dY)

    out = _build_Bn_inplace!(dX, dY, normals)
    @test out === dX                                                        # the result is written into dX
    @test dY == dY0                                                         # dY is only read
    @test dX ≈ Diagonal(first.(normals)) * dX0 + Diagonal(last.(normals)) * dY0
    @test all(dX[i, j] ≈ normals[i][1] * dX0[i, j] + normals[i][2] * dY0[i, j] for i in 1:5, j in 1:3)

    @testset "axis-aligned normals select one component" begin
        A = copy(dX0)
        _build_Bn_inplace!(A, copy(dY0), fill(SVector(1.0, 0.0), 5))
        @test A == dX0
        A = copy(dX0)
        _build_Bn_inplace!(A, copy(dY0), fill(SVector(0.0, 1.0), 5))
        @test A == dY0
        A = copy(dX0)
        _build_Bn_inplace!(A, copy(dY0), fill(SVector(-1.0, 0.0), 5))
        @test A == -dX0
    end

    @testset "the normal derivative of a linear function is its directional slope" begin
        # f(x, y) = a x + b y has gradient (a, b), so the normal derivative is n . (a, b) at every point
        a, b = 1.5, -0.5
        dXl = fill(a, 5, 1)
        dYl = fill(b, 5, 1)
        _build_Bn_inplace!(dXl, dYl, normals)
        @test vec(dXl) ≈ [a * n[1] + b * n[2] for n in normals]
    end
end
