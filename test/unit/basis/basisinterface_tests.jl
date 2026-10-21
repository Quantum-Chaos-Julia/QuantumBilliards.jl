# Unit tests for the AbsBasis interface:
#   src/abstracttypes.jl                       (AbsBasis, BasisCache)
#   src/solvers/matrixconstructors.jl          (the generic basis_matrix, dk_matrix, gradient_matrices and
#                                               basis_and_gradient_matrices wrappers, filter_matrix!)
#   src/spectra/spectralutils.jl               (uses_taylor)
#   src/basis/planewaves/realplanewaves*.jl, src/basis/fourierbessel/fourierbessel*.jl and corneradapted*.jl
#                                              (resize_basis, basis_fun, dk_fun, gradient, basis_and_gradient and basis_cache of
#                                               RealPlaneWaves, FourierBessel and CornerAdaptedFourierBessel, only as the contract
#                                               the generic wrappers and the solvers rely on)
# The wrappers read `basis.dim` and call the index-set methods of basis_fun, dk_fun, gradient and basis_and_gradient with the
# columns 1:dim, then zero the entries of magnitude below eps. Per basis type one explicit testset checks that these matrices
# equal the filtered stack of the single-index evaluators, that the threaded evaluation equals the serial one, the `hasmethod`
# contract of the AbsBasis branch (the four functions of the AbsBasis docstring plus dk_fun and basis_cache), and that the Taylor
# cache of the basis reproduces the direct matrices. A minimal toy basis shows what the generic wrappers need from a subtype.
# The values of the evaluators themselves are covered in realplanewaves_tests.jl, fourierbessel_tests.jl and
# corneradapted_tests.jl. No eigen solves, no pinned wavefunction values. Helper functions use the prefix `bi_`.

using StaticArrays: SVector
using LinearAlgebra: norm
using QuantumBilliards: BasisCache, basis_cache, uses_taylor, filter_matrix!, gradient_matrices, basis_and_gradient_matrices,
                        _weight_scaled_basis_and_dk_matrices, RealPlaneWaveTaylorCache, FourierBesselTaylorCache,
                        CornerAdaptedTaylorCache

# Fixed deterministic evaluation points, away from the symmetry axes and the origin.
bi_pts() = SVector{2,Float64}[SVector(0.3 * j - 0.7, 0.2 * j + 0.1) for j in 1:7]
const bi_PTS_T = Vector{SVector{2,Float64}}
const bi_BILLIARD_T = typeof(StadiumBilliard(0.5))

# Column-by-column stacks of the single-index evaluators, filtered like the generic wrappers.
bi_stack(f, b, k, pts) = hcat((f(b, i, k, pts) for i in 1:b.dim)...)
bi_stack_basis(b, k, pts) = filter_matrix!(bi_stack(basis_fun, b, k, pts))
bi_stack_dk(b, k, pts) = filter_matrix!(bi_stack(dk_fun, b, k, pts))
bi_stack_grad(b, k, pts) = (filter_matrix!(bi_stack((b, i, k, p) -> gradient(b, i, k, p)[1], b, k, pts)),
                            filter_matrix!(bi_stack((b, i, k, p) -> gradient(b, i, k, p)[2], b, k, pts)))
bi_stack_bg(b, k, pts) = (filter_matrix!(bi_stack((b, i, k, p) -> basis_and_gradient(b, i, k, p)[1], b, k, pts)),
                          filter_matrix!(bi_stack((b, i, k, p) -> basis_and_gradient(b, i, k, p)[2], b, k, pts)),
                          filter_matrix!(bi_stack((b, i, k, p) -> basis_and_gradient(b, i, k, p)[3], b, k, pts)))

# A minimal AbsBasis: f_i(x, y) = i k x, so ∂f_i/∂k = i x, ∇f_i = (i k, 0).
struct bi_ToyBasis <: AbsBasis
    dim::Int
end
QuantumBilliards.basis_fun(::bi_ToyBasis, indices::AbstractArray, k::Real, pts::AbstractArray; multithreaded::Bool=true) =
    [i * k * p[1] for p in pts, i in indices]
QuantumBilliards.dk_fun(::bi_ToyBasis, indices::AbstractArray, k::Real, pts::AbstractArray; multithreaded::Bool=true) =
    [i * p[1] for p in pts, i in indices]
QuantumBilliards.gradient(::bi_ToyBasis, indices::AbstractArray, k::Real, pts::AbstractArray; multithreaded::Bool=true) =
    ([i * k for p in pts, i in indices], zeros(length(pts), length(indices)))
QuantumBilliards.basis_and_gradient(b::bi_ToyBasis, indices::AbstractArray, k::Real, pts::AbstractArray; multithreaded::Bool=true) =
    (basis_fun(b, indices, k, pts), gradient(b, indices, k, pts)...)

# An AbsBasis subtype that implements nothing.
struct bi_BareBasis <: AbsBasis
    dim::Int
end

@testset "AbsBasis and BasisCache types" begin
    @test isabstracttype(AbsBasis)
    @test isabstracttype(BasisCache)
    @test !(AbsBasis <: AbsSolver)
    @test RealPlaneWaves <: AbsBasis && FourierBessel <: AbsBasis && CornerAdaptedFourierBessel <: AbsBasis
    @test all(T -> T <: BasisCache, (RealPlaneWaveTaylorCache, FourierBesselTaylorCache, CornerAdaptedTaylorCache))

    @testset "there is no generic evaluator for the abstract type" begin
        @test !hasmethod(basis_fun, Tuple{AbsBasis,Int,Float64,bi_PTS_T})
        @test !hasmethod(basis_fun, Tuple{AbsBasis,UnitRange{Int},Float64,bi_PTS_T})
        @test !hasmethod(dk_fun, Tuple{AbsBasis,Int,Float64,bi_PTS_T})
        @test !hasmethod(gradient, Tuple{AbsBasis,Int,Float64,bi_PTS_T})
        @test !hasmethod(basis_and_gradient, Tuple{AbsBasis,Int,Float64,bi_PTS_T})
        @test !hasmethod(resize_basis, Tuple{AbsBasis,bi_BILLIARD_T,Int,Float64})
    end

    @testset "the matrix wrappers are generic over AbsBasis" begin
        @test hasmethod(basis_matrix, Tuple{AbsBasis,Float64,bi_PTS_T}, (:multithreaded,))
        @test hasmethod(dk_matrix, Tuple{AbsBasis,Float64,bi_PTS_T}, (:multithreaded,))
        @test hasmethod(gradient_matrices, Tuple{AbsBasis,Float64,bi_PTS_T}, (:multithreaded,))
        @test hasmethod(basis_and_gradient_matrices, Tuple{AbsBasis,Float64,bi_PTS_T}, (:multithreaded,))
    end

    @testset "uses_taylor defaults to false for an unknown basis" begin
        @test uses_taylor(bi_ToyBasis(3)) === false
        @test uses_taylor(bi_BareBasis(3)) === false
    end
end

@testset "AbsBasis generic wrappers on a minimal basis" begin
    toy = bi_ToyBasis(3)
    pts = SVector{2,Float64}[SVector(1e-20, 0.5), SVector(0.5, 0.5), SVector(-0.25, 0.1)]
    k = 2.0

    @testset "basis_matrix evaluates the columns 1:dim and zeroes entries below eps" begin
        B = basis_matrix(toy, k, pts)
        @test size(B) == (3, 3)
        @test B[2, :] == [1.0, 2.0, 3.0] .* k * 0.5
        @test B[3, :] == [1.0, 2.0, 3.0] .* k * -0.25
        @test all(iszero, B[1, :])                                # i k 1e-20 is below eps and removed by the filter
        @test basis_fun(toy, 1:3, k, pts)[1, 1] == k * 1e-20       # the raw evaluator keeps it
    end

    @testset "dk_matrix, gradient_matrices and basis_and_gradient_matrices" begin
        @test dk_matrix(toy, k, pts) == [i * (abs(p[1]) < 1e-10 ? 0.0 : p[1]) for p in pts, i in 1:3]
        dX, dY = gradient_matrices(toy, k, pts)
        @test dX == [i * k for p in pts, i in 1:3] && all(iszero, dY)
        B, bx, by = basis_and_gradient_matrices(toy, k, pts)
        @test B == basis_matrix(toy, k, pts) && bx == dX && by == dY
    end

    @testset "multithreaded is forwarded" begin
        @test basis_matrix(toy, k, pts; multithreaded=false) == basis_matrix(toy, k, pts; multithreaded=true)
    end

    @testset "a subtype without evaluators has no fallback" begin
        bare = bi_BareBasis(2)
        @test_throws MethodError basis_matrix(bare, k, pts)
        @test_throws MethodError dk_matrix(bare, k, pts)
        @test_throws MethodError gradient_matrices(bare, k, pts)
        @test_throws MethodError basis_and_gradient_matrices(bare, k, pts)
        @test_throws MethodError resize_basis(bare, StadiumBilliard(0.5), 3, k)
    end
end

@testset "RealPlaneWaves - AbsBasis interface" begin
    k = 4.4
    pts = bi_pts()
    b = RealPlaneWaves(3)
    bs = RealPlaneWaves(4; sym_x=1)

    @testset "hasmethod contract" begin
        B = typeof(b)
        @test B <: AbsBasis && hasfield(B, :dim)
        @test hasmethod(resize_basis, Tuple{B,bi_BILLIARD_T,Int,Float64})
        for idx in (Int, UnitRange{Int})
            @test hasmethod(basis_fun, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(dk_fun, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(gradient, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(basis_and_gradient, Tuple{B,idx,Float64,bi_PTS_T})
        end
        for f in (basis_fun, dk_fun, gradient, basis_and_gradient)
            @test hasmethod(f, Tuple{B,UnitRange{Int},Float64,bi_PTS_T}, (:multithreaded,))
        end
        @test hasmethod(basis_cache, Tuple{B,Float64,bi_PTS_T})
        @test uses_taylor(b) === true
        @test resize_basis(b, StadiumBilliard(0.5), 5, k) isa RealPlaneWaves{Float64}
    end

    @testset "basis_matrix equals the stacked columns of basis_fun" begin
        @test size(basis_matrix(b, k, pts)) == (length(pts), b.dim)
        @test basis_matrix(b, k, pts) ≈ bi_stack_basis(b, k, pts) atol = 1e-14
        @test basis_matrix(bs, k, pts) ≈ bi_stack_basis(bs, k, pts) atol = 1e-14
        @test basis_matrix(b, k, pts) ≈ filter_matrix!(basis_fun(b, 1:b.dim, k, pts)) atol = 1e-14
    end

    @testset "dk, gradient and combined matrices equal the stacked single-index evaluators" begin
        @test dk_matrix(b, k, pts) ≈ bi_stack_dk(b, k, pts) atol = 1e-13
        mx, my = gradient_matrices(b, k, pts)
        sx, sy = bi_stack_grad(b, k, pts)
        @test mx ≈ sx atol = 1e-13
        @test my ≈ sy atol = 1e-13
        Bm, bx, by = basis_and_gradient_matrices(b, k, pts)
        tB, tx, ty = bi_stack_bg(b, k, pts)
        @test Bm ≈ tB atol = 1e-13
        @test bx ≈ tx atol = 1e-13
        @test by ≈ ty atol = 1e-13
        @test Bm == basis_matrix(b, k, pts) && bx == mx && by == my
        @test dk_matrix(bs, k, pts) ≈ bi_stack_dk(bs, k, pts) atol = 1e-13
    end

    @testset "threaded equals serial" begin
        @test basis_matrix(b, k, pts; multithreaded=true) == basis_matrix(b, k, pts; multithreaded=false)
        @test dk_matrix(b, k, pts; multithreaded=true) == dk_matrix(b, k, pts; multithreaded=false)
        @test gradient_matrices(b, k, pts; multithreaded=true) == gradient_matrices(b, k, pts; multithreaded=false)
        @test basis_and_gradient_matrices(b, k, pts; multithreaded=true) == basis_and_gradient_matrices(b, k, pts; multithreaded=false)
    end

    @testset "the Taylor cache reproduces the direct matrices" begin
        k0 = 4.0
        cache = basis_cache(b, k0, pts)
        @test cache isa RealPlaneWaveTaylorCache{Float64}
        @test hasmethod(_weight_scaled_basis_and_dk_matrices, Tuple{typeof(cache),Float64,Vector{Float64},Int})
        @test norm(basis_matrix(cache, k0 + 0.2) - basis_matrix(b, k0 + 0.2, pts)) < 1e-11
        @test norm(dk_matrix(cache, k0 + 0.2) - dk_matrix(b, k0 + 0.2, pts)) < 1e-11
    end
end

@testset "FourierBessel - AbsBasis interface" begin
    k = 4.4
    pts = bi_pts()
    b = FourierBessel(6; gradients=true)
    bs = FourierBessel(5; sym_x=-1, sym_y=1)

    @testset "hasmethod contract" begin
        B = typeof(b)
        @test B <: AbsBasis && hasfield(B, :dim)
        @test hasmethod(resize_basis, Tuple{B,bi_BILLIARD_T,Int,Float64})
        for idx in (Int, UnitRange{Int})
            @test hasmethod(basis_fun, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(dk_fun, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(gradient, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(basis_and_gradient, Tuple{B,idx,Float64,bi_PTS_T})
        end
        for f in (basis_fun, dk_fun, gradient, basis_and_gradient)
            @test hasmethod(f, Tuple{B,UnitRange{Int},Float64,bi_PTS_T}, (:multithreaded,))
        end
        @test hasmethod(basis_cache, Tuple{B,Float64,bi_PTS_T})
        @test uses_taylor(b) === true
        @test resize_basis(b, StadiumBilliard(0.5), 8, k) isa FourierBessel{Float64}
        @test resize_basis(b, StadiumBilliard(0.5), 8, k).dim == 8
    end

    @testset "basis_matrix equals the stacked columns of basis_fun" begin
        @test size(basis_matrix(b, k, pts)) == (length(pts), b.dim)
        @test basis_matrix(b, k, pts) ≈ bi_stack_basis(b, k, pts) atol = 1e-14
        @test basis_matrix(bs, k, pts) ≈ bi_stack_basis(bs, k, pts) atol = 1e-14
        @test basis_matrix(b, k, pts) ≈ filter_matrix!(basis_fun(b, 1:b.dim, k, pts)) atol = 1e-14
    end

    @testset "dk, gradient and combined matrices equal the stacked single-index evaluators" begin
        @test dk_matrix(b, k, pts) ≈ bi_stack_dk(b, k, pts) atol = 1e-13
        mx, my = gradient_matrices(b, k, pts)
        sx, sy = bi_stack_grad(b, k, pts)
        @test mx ≈ sx atol = 1e-13
        @test my ≈ sy atol = 1e-13
        Bm, bx, by = basis_and_gradient_matrices(b, k, pts)
        tB, tx, ty = bi_stack_bg(b, k, pts)
        @test Bm ≈ tB atol = 1e-13
        @test bx ≈ tx atol = 1e-13
        @test by ≈ ty atol = 1e-13
        @test Bm == basis_matrix(b, k, pts) && bx == mx && by == my
        @test dk_matrix(bs, k, pts) ≈ bi_stack_dk(bs, k, pts) atol = 1e-13
    end

    @testset "threaded equals serial" begin
        @test basis_matrix(b, k, pts; multithreaded=true) == basis_matrix(b, k, pts; multithreaded=false)
        @test dk_matrix(b, k, pts; multithreaded=true) == dk_matrix(b, k, pts; multithreaded=false)
        @test gradient_matrices(b, k, pts; multithreaded=true) == gradient_matrices(b, k, pts; multithreaded=false)
        @test basis_and_gradient_matrices(b, k, pts; multithreaded=true) == basis_and_gradient_matrices(b, k, pts; multithreaded=false)
    end

    @testset "the Taylor cache reproduces the direct matrices" begin
        k0 = 4.0
        cache = basis_cache(b, k0, pts)
        @test cache isa FourierBesselTaylorCache{Float64}
        @test hasmethod(_weight_scaled_basis_and_dk_matrices, Tuple{typeof(cache),Float64,Vector{Float64},Int})
        @test norm(basis_matrix(cache, k0 + 0.2) - basis_matrix(b, k0 + 0.2, pts)) < 1e-11
        @test norm(dk_matrix(cache, k0 + 0.2) - dk_matrix(b, k0 + 0.2, pts)) < 1e-11
        cx, cy = gradient_matrices(cache, b, k0 + 0.2)
        dx, dy = gradient_matrices(b, k0 + 0.2, pts)
        @test norm(cx - dx) < 1e-11 && norm(cy - dy) < 1e-11
    end
end

@testset "CornerAdaptedFourierBessel - AbsBasis interface" begin
    k = 4.4
    pts = bi_pts()
    b = CornerAdaptedFourierBessel(5, 2π / 3, SVector(0.1, 0.2), 0.3; gradients=true)

    @testset "hasmethod contract" begin
        B = typeof(b)
        @test B <: AbsBasis && hasfield(B, :dim)
        @test hasmethod(resize_basis, Tuple{B,bi_BILLIARD_T,Int,Float64})
        for idx in (Int, UnitRange{Int})
            @test hasmethod(basis_fun, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(dk_fun, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(gradient, Tuple{B,idx,Float64,bi_PTS_T})
            @test hasmethod(basis_and_gradient, Tuple{B,idx,Float64,bi_PTS_T})
        end
        for f in (basis_fun, dk_fun, gradient, basis_and_gradient)
            @test hasmethod(f, Tuple{B,UnitRange{Int},Float64,bi_PTS_T}, (:multithreaded,))
        end
        @test hasmethod(basis_cache, Tuple{B,Float64,bi_PTS_T})
        @test uses_taylor(b) === true
        @test resize_basis(b, StadiumBilliard(0.5), 8, k) isa CornerAdaptedFourierBessel{Float64}
        @test resize_basis(b, StadiumBilliard(0.5), 8, k).dim == 8
    end

    @testset "basis_matrix equals the stacked columns of basis_fun" begin
        @test size(basis_matrix(b, k, pts)) == (length(pts), b.dim)
        @test basis_matrix(b, k, pts) ≈ bi_stack_basis(b, k, pts) atol = 1e-14
        @test basis_matrix(b, k, pts) ≈ filter_matrix!(basis_fun(b, 1:b.dim, k, pts)) atol = 1e-14
    end

    @testset "dk, gradient and combined matrices equal the stacked single-index evaluators" begin
        @test dk_matrix(b, k, pts) ≈ bi_stack_dk(b, k, pts) atol = 1e-13
        mx, my = gradient_matrices(b, k, pts)
        sx, sy = bi_stack_grad(b, k, pts)
        @test mx ≈ sx atol = 1e-13
        @test my ≈ sy atol = 1e-13
        Bm, bx, by = basis_and_gradient_matrices(b, k, pts)
        tB, tx, ty = bi_stack_bg(b, k, pts)
        @test Bm ≈ tB atol = 1e-13
        @test bx ≈ tx atol = 1e-13
        @test by ≈ ty atol = 1e-13
        @test Bm == basis_matrix(b, k, pts) && bx == mx && by == my
    end

    @testset "threaded equals serial" begin
        @test basis_matrix(b, k, pts; multithreaded=true) == basis_matrix(b, k, pts; multithreaded=false)
        @test dk_matrix(b, k, pts; multithreaded=true) == dk_matrix(b, k, pts; multithreaded=false)
        @test gradient_matrices(b, k, pts; multithreaded=true) == gradient_matrices(b, k, pts; multithreaded=false)
        @test basis_and_gradient_matrices(b, k, pts; multithreaded=true) == basis_and_gradient_matrices(b, k, pts; multithreaded=false)
    end

    @testset "the Taylor cache reproduces the direct matrices" begin
        k0 = 4.0
        cache = basis_cache(b, k0, pts)
        @test cache isa CornerAdaptedTaylorCache{Float64}
        @test hasmethod(_weight_scaled_basis_and_dk_matrices, Tuple{typeof(cache),Float64,Vector{Float64},Int})
        @test norm(basis_matrix(cache, k0 + 0.2) - basis_matrix(b, k0 + 0.2, pts)) < 1e-11
        @test norm(dk_matrix(cache, k0 + 0.2) - dk_matrix(b, k0 + 0.2, pts)) < 1e-11
        cx, cy = gradient_matrices(cache, b, k0 + 0.2)
        dx, dy = gradient_matrices(b, k0 + 0.2, pts)
        @test norm(cx - dx) < 1e-11 && norm(cy - dy) < 1e-11
    end
end
