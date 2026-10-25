# Unit tests for
#   src/basis/planewaves/realplanewaves_recurrence.jl   (RealPlaneWaveTaylorCache, _real_plane_wave_taylor_coefficients!,
#                                                         basis_cache, and the cache methods of basis_fun, basis_matrix,
#                                                         dk_fun, dk_matrix and _weight_scaled_basis_and_dk_matrices)
# A basis function is f(k) = F_x(k c) F_y(k d) with c = cos(t) x, d = sin(t) y and F = cos (parity +1) or sin (parity -1).
# Its normalized Taylor coefficients a_n = f^(n)(k0) / n! are checked against the closed-form derivatives
# f' = c F_x' F_y + d F_x F_y' and f'' = c² F_x'' F_y + 2 c d F_x' F_y' + d² F_x F_y'' (with F'' = -F), and the whole
# polynomial against direct evaluation of the trigonometric basis (which lives in realplanewaves_tests.jl).
# The Horner kernels themselves are covered in solvers/taylor/recurrence_core_tests.jl.
# No eigen solves, no pinned wavefunction values. Helper functions use the prefix `rpwr_`.

using StaticArrays: SVector
using QuantumBilliards: RealPlaneWaveTaylorCache, basis_cache, BasisCache, _real_plane_wave_taylor_coefficients!,
                        _weight_scaled_basis_and_dk_matrices

# Fixed deterministic evaluation points, away from the symmetry axes.
rpwr_pts(::Type{T}=Float64) where {T} = SVector{2,T}[SVector{2,T}(0.15 * j - 0.4, 0.5 - 0.17 * j) for j in 1:6]

rpwr_F(par, a) = par == 1 ? cos(a) : sin(a)
rpwr_dF(par, a) = par == 1 ? -sin(a) : cos(a)

# All p+1 normalized Taylor coefficients of the (parx, pary) plane wave at (x, y), direction angle θ, center k0.
function rpwr_coefficients(parx, pary, x, y, θ, k0, p)
    C = zeros(p + 1, 1, 1)
    _real_plane_wave_taylor_coefficients!(C, 1, 1, x, y, cos(θ), sin(θ), parx, pary, k0, p)
    return vec(C)
end

# Plain polynomial sum Σ a_n δ^n.
rpwr_poly(a, δ) = sum(a[n+1] * δ^n for n in 0:length(a)-1)

# Closed-form f, f', f'' at k0 for the (parx, pary) plane wave.
function rpwr_closed_form(parx, pary, x, y, θ, k0)
    c, d = cos(θ) * x, sin(θ) * y
    Fx, Fy = rpwr_F(parx, k0 * c), rpwr_F(pary, k0 * d)
    dFx, dFy = rpwr_dF(parx, k0 * c), rpwr_dF(pary, k0 * d)
    f = Fx * Fy
    df = c * dFx * Fy + d * Fx * dFy
    ddf = -c^2 * f + 2c * d * dFx * dFy - d^2 * f
    return f, df, ddf
end

@testset "RealPlaneWaveTaylorCache type" begin
    @test RealPlaneWaveTaylorCache <: BasisCache
    @test fieldnames(RealPlaneWaveTaylorCache) == (:k0, :degree, :coeffs)
    cache = RealPlaneWaveTaylorCache(2.5, 3, zeros(4, 2, 5))
    @test cache isa RealPlaneWaveTaylorCache{Float64}
    @test isconcretetype(typeof(cache))
    @test (cache.k0, cache.degree, size(cache.coeffs)) == (2.5, 3, (4, 2, 5))
    @test_throws MethodError RealPlaneWaveTaylorCache(2.5, 3, zeros(Float32, 4, 2, 5))   # k0 and coefficients share one type
end

@testset "Taylor coefficients - cos-cos plane wave" begin
    parx, pary, x, y, θ, k0, p = 1, 1, 0.37, -0.22, 0.6, 3.0, 18
    a = rpwr_coefficients(parx, pary, x, y, θ, k0, p)
    f, df, ddf = rpwr_closed_form(parx, pary, x, y, θ, k0)
    @test length(a) == p + 1
    @test a[1] ≈ f atol = 1e-14
    @test a[2] ≈ df atol = 1e-14
    @test a[3] ≈ ddf / 2 atol = 1e-14
    for δ in (0.4, -0.4)
        direct = rpwr_F(parx, (k0 + δ) * cos(θ) * x) * rpwr_F(pary, (k0 + δ) * sin(θ) * y)
        @test rpwr_poly(a, δ) ≈ direct atol = 1e-12
    end
end

@testset "Taylor coefficients - sin-cos plane wave" begin
    parx, pary, x, y, θ, k0, p = -1, 1, 0.37, -0.22, 0.6, 3.0, 18
    a = rpwr_coefficients(parx, pary, x, y, θ, k0, p)
    f, df, ddf = rpwr_closed_form(parx, pary, x, y, θ, k0)
    @test a[1] ≈ f atol = 1e-14
    @test a[2] ≈ df atol = 1e-14
    @test a[3] ≈ ddf / 2 atol = 1e-14
    for δ in (0.4, -0.4)
        direct = rpwr_F(parx, (k0 + δ) * cos(θ) * x) * rpwr_F(pary, (k0 + δ) * sin(θ) * y)
        @test rpwr_poly(a, δ) ≈ direct atol = 1e-12
    end
end

@testset "Taylor coefficients - cos-sin plane wave" begin
    parx, pary, x, y, θ, k0, p = 1, -1, 0.37, -0.22, 0.6, 3.0, 18
    a = rpwr_coefficients(parx, pary, x, y, θ, k0, p)
    f, df, ddf = rpwr_closed_form(parx, pary, x, y, θ, k0)
    @test a[1] ≈ f atol = 1e-14
    @test a[2] ≈ df atol = 1e-14
    @test a[3] ≈ ddf / 2 atol = 1e-14
    for δ in (0.4, -0.4)
        direct = rpwr_F(parx, (k0 + δ) * cos(θ) * x) * rpwr_F(pary, (k0 + δ) * sin(θ) * y)
        @test rpwr_poly(a, δ) ≈ direct atol = 1e-12
    end
end

@testset "Taylor coefficients - sin-sin plane wave" begin
    parx, pary, x, y, θ, k0, p = -1, -1, 0.37, -0.22, 0.6, 3.0, 18
    a = rpwr_coefficients(parx, pary, x, y, θ, k0, p)
    f, df, ddf = rpwr_closed_form(parx, pary, x, y, θ, k0)
    @test a[1] ≈ f atol = 1e-14
    @test a[2] ≈ df atol = 1e-14
    @test a[3] ≈ ddf / 2 atol = 1e-14
    for δ in (0.4, -0.4)
        direct = rpwr_F(parx, (k0 + δ) * cos(θ) * x) * rpwr_F(pary, (k0 + δ) * sin(θ) * y)
        @test rpwr_poly(a, δ) ≈ direct atol = 1e-12
    end
end

@testset "Taylor coefficients - degree handling and structure" begin
    args = (1, -1, 0.37, -0.22, 0.6, 3.0)
    full = rpwr_coefficients(args..., 12)

    @testset "lower degrees are exact prefixes of the full expansion" begin
        @test rpwr_coefficients(args..., 0) == full[1:1]
        @test rpwr_coefficients(args..., 1) == full[1:2]
        @test rpwr_coefficients(args..., 2) == full[1:3]
        @test rpwr_coefficients(args..., 7) == full[1:8]
    end

    @testset "the coefficients decay factorially for |q| < 1" begin
        # |a_n| <= (|q_-| + |q_+|)^n / (2 n!) with |q_±| <= |x| + |y| for this point.
        q = abs(0.37) + abs(0.22)
        @test all(abs(full[n+1]) <= q^n / factorial(n) + 1e-14 for n in 0:12)
    end

    @testset "writes only the requested slot" begin
        C = fill(NaN, 5, 3, 2)
        _real_plane_wave_taylor_coefficients!(C, 2, 1, 0.1, 0.2, 0.5, 0.8, 1, 1, 3.0, 4)
        @test all(isfinite, C[:, 2, 1])
        @test all(isnan, C[:, 1, 1]) && all(isnan, C[:, 3, 1]) && all(isnan, C[:, :, 2])
    end

    @testset "at the origin only the cos-cos wave is non-constant-zero" begin
        @test rpwr_coefficients(1, 1, 0.0, 0.0, 0.6, 3.0, 6) == [1.0, 0, 0, 0, 0, 0, 0]
        @test all(iszero, rpwr_coefficients(-1, 1, 0.0, 0.0, 0.6, 3.0, 6))
        @test all(iszero, rpwr_coefficients(1, -1, 0.0, 0.0, 0.6, 3.0, 6))
        @test all(iszero, rpwr_coefficients(-1, -1, 0.0, 0.0, 0.6, 3.0, 6))
    end
end

@testset "RealPlaneWaveTaylorCache construction" begin
    k0 = 3.0
    pts = rpwr_pts()

    @testset "shape, center and degree" begin
        basis = RealPlaneWaves(2; sym_x=-1, sym_y=1, taylor_degree=10)
        cache = RealPlaneWaveTaylorCache(basis, k0, pts)
        @test cache isa RealPlaneWaveTaylorCache{Float64}
        @test cache.k0 == k0
        @test cache.degree == 10
        @test size(cache.coeffs) == (11, length(pts), basis.dim)
    end

    @testset "default degree comes from the basis" begin
        cache = RealPlaneWaveTaylorCache(RealPlaneWaves(2), k0, pts)
        @test cache.degree == 16
        @test size(cache.coeffs, 1) == 17
    end

    @testset "entries equal the single-function coefficients" begin
        basis = RealPlaneWaves(2; sym_x=-1, sym_y=1, taylor_degree=10)
        cache = RealPlaneWaveTaylorCache(basis, k0, pts)
        for (j, c) in ((1, 1), (4, 2), (6, 1))
            expected = rpwr_coefficients(basis.parity_x[c], basis.parity_y[c], pts[j][1], pts[j][2], basis.angles[c], k0, 10)
            @test cache.coeffs[:, j, c] == expected
        end
    end

    @testset "value and derivative slots match the trigonometric basis" begin
        basis = RealPlaneWaves(3)
        cache = RealPlaneWaveTaylorCache(basis, k0, pts)
        @test cache.coeffs[1, :, :] ≈ basis_fun(basis, 1:basis.dim, k0, pts) atol = 1e-14
        @test cache.coeffs[2, :, :] ≈ dk_fun(basis, 1:basis.dim, k0, pts) atol = 1e-14
    end

    @testset "basis_cache is the cache constructor" begin
        basis = RealPlaneWaves(2)
        c1 = basis_cache(basis, k0, pts)
        c2 = RealPlaneWaveTaylorCache(basis, k0, pts)
        @test c1 isa RealPlaneWaveTaylorCache{Float64}
        @test (c1.k0, c1.degree) == (c2.k0, c2.degree)
        @test c1.coeffs == c2.coeffs
        @test basis_cache(basis, k0, pts; multithreaded=false).coeffs == c2.coeffs
    end

    @testset "threaded equals serial" begin
        basis = RealPlaneWaves(5)
        threaded = RealPlaneWaveTaylorCache(basis, k0, pts; multithreaded=true)
        serial = RealPlaneWaveTaylorCache(basis, k0, pts; multithreaded=false)
        @test threaded.coeffs == serial.coeffs
    end

    @testset "empty point set" begin
        basis = RealPlaneWaves(2)
        cache = RealPlaneWaveTaylorCache(basis, k0, SVector{2,Float64}[])
        @test size(cache.coeffs) == (17, 0, basis.dim)
        @test size(basis_matrix(cache, k0)) == (0, basis.dim)
    end

    @testset "wavenumber type must match the basis type" begin
        @test_throws MethodError RealPlaneWaveTaylorCache(RealPlaneWaves(2), 3.0f0, pts)
        @test_throws MethodError RealPlaneWaveTaylorCache(RealPlaneWaves(2), 3, pts)
    end
end

@testset "Taylor cache vs direct evaluation - unrestricted basis" begin
    k0 = 3.0
    pts = rpwr_pts()
    basis = RealPlaneWaves(3)
    cache = RealPlaneWaveTaylorCache(basis, k0, pts)

    @testset "at the expansion center" begin
        @test basis_matrix(cache, k0) ≈ basis_matrix(basis, k0, pts) atol = 1e-14
        @test dk_matrix(cache, k0) ≈ dk_matrix(basis, k0, pts) atol = 1e-14
    end

    @testset "above the center" begin
        k = k0 + 0.3
        @test basis_matrix(cache, k) ≈ basis_matrix(basis, k, pts) atol = 1e-12
        @test dk_matrix(cache, k) ≈ dk_matrix(basis, k, pts) atol = 1e-12
    end

    @testset "below the center, at the reuse radius" begin
        k = k0 - basis.taylor_radius
        @test basis_matrix(cache, k) ≈ basis_matrix(basis, k, pts) atol = 1e-11
        @test dk_matrix(cache, k) ≈ dk_matrix(basis, k, pts) atol = 1e-11
    end

    @testset "single-index and index-set methods equal the matrix columns" begin
        k = k0 + 0.2
        B = basis_matrix(cache, k)
        D = dk_matrix(cache, k)
        @test basis_fun(cache, 1, k) == B[:, 1]
        @test basis_fun(cache, 8, k) == B[:, 8]
        @test dk_fun(cache, 1, k) == D[:, 1]
        @test dk_fun(cache, 8, k) == D[:, 8]
        @test basis_fun(cache, [5, 2], k) == B[:, [5, 2]]
        @test dk_fun(cache, [5, 2], k) == D[:, [5, 2]]
        @test basis_fun(cache, 1:basis.dim, k) == B
        @test dk_fun(cache, 1:basis.dim, k) == D
        @test size(basis_fun(cache, Int[], k)) == (length(pts), 0)
    end

    @testset "single-index methods agree with the trigonometric basis" begin
        k = k0 - 0.25
        @test basis_fun(cache, 6, k) ≈ basis_fun(basis, 6, k, pts) atol = 1e-12
        @test dk_fun(cache, 6, k) ≈ dk_fun(basis, 6, k, pts) atol = 1e-12
    end

    @testset "derivative of the cached basis equals the cached derivative" begin
        k, h = k0 + 0.1, 1e-6
        fd = (basis_matrix(cache, k + h) .- basis_matrix(cache, k - h)) ./ (2h)
        @test fd ≈ dk_matrix(cache, k) atol = 1e-6
    end

    @testset "origin: sine columns vanish identically, cos-cos columns are constant in k" begin
        c0 = RealPlaneWaveTaylorCache(basis, k0, [SVector(0.0, 0.0)])
        B = basis_matrix(c0, k0 + 0.2)
        D = dk_matrix(c0, k0 + 0.2)
        for i in 1:basis.dim
            is_cos_cos = basis.parity_x[i] == 1 && basis.parity_y[i] == 1
            @test B[1, i] == (is_cos_cos ? 1.0 : 0.0)
            @test D[1, i] == 0.0
        end
    end
end

@testset "Taylor cache vs direct evaluation - odd-odd sector" begin
    k0 = 3.0
    pts = rpwr_pts()
    basis = RealPlaneWaves(4; sym_x=-1, sym_y=-1)
    cache = RealPlaneWaveTaylorCache(basis, k0, pts)
    k = k0 + 0.35

    @test basis_matrix(cache, k) ≈ basis_matrix(basis, k, pts) atol = 1e-12
    @test dk_matrix(cache, k) ≈ dk_matrix(basis, k, pts) atol = 1e-12

    # The Taylor representation inherits the reflection parity of the sector: odd in x and odd in y.
    mirrored_x = RealPlaneWaveTaylorCache(basis, k0, [SVector(-p[1], p[2]) for p in pts])
    mirrored_y = RealPlaneWaveTaylorCache(basis, k0, [SVector(p[1], -p[2]) for p in pts])
    @test basis_matrix(mirrored_x, k) ≈ -basis_matrix(cache, k) atol = 1e-13
    @test basis_matrix(mirrored_y, k) ≈ -basis_matrix(cache, k) atol = 1e-13
    @test dk_matrix(mirrored_x, k) ≈ -dk_matrix(cache, k) atol = 1e-13
end

@testset "Taylor cache vs direct evaluation - half-plane sector" begin
    k0 = 3.0
    pts = rpwr_pts()
    basis = RealPlaneWaves(4; sym_x=1)
    cache = RealPlaneWaveTaylorCache(basis, k0, pts)
    k = k0 - 0.3
    @test basis.dim == 8
    @test basis_matrix(cache, k) ≈ basis_matrix(basis, k, pts) atol = 1e-12
    @test dk_matrix(cache, k) ≈ dk_matrix(basis, k, pts) atol = 1e-12
end

@testset "Taylor truncation error decreases with the degree" begin
    k0 = 3.0
    pts = rpwr_pts()
    k = k0 + 0.45
    err(deg) = begin
        basis = RealPlaneWaves(2; taylor_degree=deg)
        maximum(abs, basis_matrix(RealPlaneWaveTaylorCache(basis, k0, pts), k) .- basis_matrix(basis, k, pts))
    end
    e4, e8, e16 = err(4), err(8), err(16)
    @test e4 > 1e-8                      # a degree-4 polynomial cannot resolve a shift of 0.45 in k to machine precision
    @test e8 < e4 / 100
    @test e16 < 1e-11
    @test e16 < e8
end

@testset "_weight_scaled_basis_and_dk_matrices" begin
    k0 = 3.0
    pts = rpwr_pts()
    basis = RealPlaneWaves(3)
    cache = RealPlaneWaveTaylorCache(basis, k0, pts)
    k = k0 + 0.2
    w = [0.3, 0.1, 0.25, 0.4, 0.15, 0.2]
    B0 = basis_matrix(cache, k)
    D0 = dk_matrix(cache, k)

    @testset "rows are scaled by sqrt(nsym * w)" begin
        B, D = _weight_scaled_basis_and_dk_matrices(cache, k, w, 2)
        @test B ≈ sqrt.(2 .* w) .* B0 atol = 1e-13
        @test D ≈ sqrt.(2 .* w) .* D0 atol = 1e-13
    end

    @testset "nsym = 1 gives plain sqrt(w) scaling" begin
        B, D = _weight_scaled_basis_and_dk_matrices(cache, k, w, 1)
        @test B ≈ sqrt.(w) .* B0 atol = 1e-13
        @test D ≈ sqrt.(w) .* D0 atol = 1e-13
    end

    @testset "Gram matrix of the scaled basis is the weighted Gram matrix" begin
        B, _ = _weight_scaled_basis_and_dk_matrices(cache, k, w, 1)
        @test B' * B ≈ B0' * (w .* B0) atol = 1e-12
    end

    @testset "threaded equals serial" begin
        @test _weight_scaled_basis_and_dk_matrices(cache, k, w, 2; multithreaded=true) ==
              _weight_scaled_basis_and_dk_matrices(cache, k, w, 2; multithreaded=false)
    end

    @testset "weight length must match the cached points" begin
        @test_throws DimensionMismatch _weight_scaled_basis_and_dk_matrices(cache, k, w[1:5], 1)
        @test_throws DimensionMismatch _weight_scaled_basis_and_dk_matrices(cache, k, vcat(w, 0.1), 1)
    end
end

@testset "RealPlaneWaveTaylorCache - threaded equals serial" begin
    k0 = 3.0
    pts = rpwr_pts()
    basis = RealPlaneWaves(5)
    cache = RealPlaneWaveTaylorCache(basis, k0, pts)
    k = k0 + 0.2
    idx = 1:basis.dim

    @test basis_matrix(cache, k; multithreaded=true) == basis_matrix(cache, k; multithreaded=false)
    @test dk_matrix(cache, k; multithreaded=true) == dk_matrix(cache, k; multithreaded=false)
    @test basis_fun(cache, idx, k; multithreaded=true) == basis_fun(cache, idx, k; multithreaded=false)
    @test dk_fun(cache, idx, k; multithreaded=true) == dk_fun(cache, idx, k; multithreaded=false)
end

@testset "RealPlaneWaveTaylorCache - type stability" begin
    k0 = 3.0
    pts = rpwr_pts()
    basis = RealPlaneWaves(3)
    cache = RealPlaneWaveTaylorCache(basis, k0, pts; multithreaded=false)
    k = k0 + 0.1
    idx = 1:basis.dim
    w = fill(0.2, length(pts))

    @test @inferred(RealPlaneWaveTaylorCache(basis, k0, pts; multithreaded=false)) isa RealPlaneWaveTaylorCache{Float64}
    @test @inferred(basis_cache(basis, k0, pts; multithreaded=false)) isa RealPlaneWaveTaylorCache{Float64}
    @test @inferred(basis_fun(cache, 2, k)) isa Vector{Float64}
    @test @inferred(basis_fun(cache, idx, k)) isa Matrix{Float64}
    @test @inferred(basis_matrix(cache, k)) isa Matrix{Float64}
    @test @inferred(dk_fun(cache, 2, k)) isa Vector{Float64}
    @test @inferred(dk_fun(cache, idx, k)) isa Matrix{Float64}
    @test @inferred(dk_matrix(cache, k)) isa Matrix{Float64}
    @test @inferred(_weight_scaled_basis_and_dk_matrices(cache, k, w, 2)) isa Tuple{Matrix{Float64},Matrix{Float64}}
end

@testset "RealPlaneWaveTaylorCache - Float32" begin
    basis32 = RealPlaneWaves(3; sym_x=-1, sampler=LinearNodes(Float32), angle_arc=Float32(π), angle_shift=-Float32(π) / 2,
                             taylor_degree=10, taylor_radius=0.5f0, taylor_tol=1.0f-6)
    basis64 = RealPlaneWaves(3; sym_x=-1, taylor_degree=10)
    pts32 = rpwr_pts(Float32)
    pts64 = rpwr_pts(Float64)
    k0 = 3.0f0
    cache32 = RealPlaneWaveTaylorCache(basis32, k0, pts32)
    cache64 = RealPlaneWaveTaylorCache(basis64, 3.0, pts64)
    k = 3.25f0

    @testset "storage and evaluation stay in Float32" begin
        @test cache32 isa RealPlaneWaveTaylorCache{Float32}
        @test eltype(cache32.coeffs) == Float32
        @test size(cache32.coeffs) == (11, length(pts32), basis32.dim)
        @test @inferred(basis_matrix(cache32, k)) isa Matrix{Float32}
        @test @inferred(dk_matrix(cache32, k)) isa Matrix{Float32}
        @test @inferred(basis_fun(cache32, 1, k)) isa Vector{Float32}
        @test @inferred(dk_fun(cache32, 1, k)) isa Vector{Float32}
    end

    @testset "agrees with Float64 to single precision" begin
        @test basis_matrix(cache32, k) ≈ basis_matrix(cache64, 3.25) atol = 1e-4
        @test dk_matrix(cache32, k) ≈ dk_matrix(cache64, 3.25) atol = 1e-4
    end

    @testset "a Float64 wavenumber is rejected by a Float32 cache" begin
        @test_throws MethodError basis_matrix(cache32, 3.25)
        @test_throws MethodError basis_fun(cache32, 1, 3.25)
        @test_throws MethodError dk_fun(cache32, 1:2, 3.25)
    end
end
