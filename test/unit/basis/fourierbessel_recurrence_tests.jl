# Unit tests for
#   src/basis/fourierbessel/fourierbessel_recurrence.jl   (FourierBesselTaylorCache, basis_cache, and the cache methods of
#                                                          basis_matrix, dk_matrix, gradient_matrices,
#                                                          basis_and_gradient_matrices and
#                                                          _weight_scaled_basis_and_dk_matrices)
# The radial factor J_j(k r) of a mode is expanded about k0 with normalized coefficients a_n = r^n J_j^(n)(k0 r) / n!.
# They are checked against the closed forms a_0 = J_j(x), a_1 = r J_j'(x), a_2 = r² J_j''(x) / 2 with
# J'' = -J'/x - (1 - j²/x²) J at x = k0 r, and the whole polynomial against SpecialFunctions.besselj at shifted k. The
# angular data stored next to the coefficients are checked against cos/sin(jφ) and the global polar angle of the point
# (sinφ = (y - y0)/r, cosφ = (x - x0)/r, since φ_local + rot_angle is the global angle about the origin). The matrices built
# from the cache are compared with direct evaluation of the basis (which lives in fourierbessel_tests.jl).
# The Horner kernels themselves are covered in solvers/taylor/recurrence_core_tests.jl.
# No eigen solves, no pinned wavefunction values. Helper functions use the prefix `fbr_`.

using StaticArrays: SVector
using LinearAlgebra: norm
using SpecialFunctions: besselj
using QuantumBilliards: FourierBesselTaylorCache, basis_cache, BasisCache, _weight_scaled_basis_and_dk_matrices,
                        gradient_matrices, basis_and_gradient_matrices

# Fixed deterministic evaluation points, away from the origin and from the symmetry axes.
fbr_pts(::Type{T}=Float64) where {T} = SVector{2,T}[SVector{2,T}(0.3 * i - 0.7, 0.2 * i + 0.1) for i in 1:7]

# J_j' and J_j'' from the recurrence and the Bessel equation (independent of the package).
fbr_dJ(j, x) = (besselj(j - 1, x) - besselj(j + 1, x)) / 2
fbr_ddJ(j, x) = -fbr_dJ(j, x) / x - (1 - j^2 / x^2) * besselj(j, x)

# Plain polynomial sum Σ a_n δ^n.
fbr_poly(a, δ) = sum(a[n+1] * δ^n for n in 0:length(a)-1)

@testset "FourierBesselTaylorCache type" begin
    @test FourierBesselTaylorCache <: BasisCache
    @test fieldnames(FourierBesselTaylorCache) == (:k0, :degree, :r, :coeffs, :ang, :sinφ, :cosφ, :dang)
    cache = FourierBesselTaylorCache(2.5, 3, zeros(2), zeros(4, 2, 5), zeros(2, 5), nothing, nothing, nothing)
    @test cache isa FourierBesselTaylorCache{Float64}
    @test isconcretetype(typeof(cache))
    @test (cache.k0, cache.degree, size(cache.coeffs)) == (2.5, 3, (4, 2, 5))
    @test_throws MethodError FourierBesselTaylorCache(2.5, 3, zeros(Float32, 2), zeros(4, 2, 5), zeros(2, 5), nothing, nothing, nothing)
end

@testset "Taylor coefficients of J_j(kr) - closed forms" begin
    k0, p = 3.0, 18
    pt = SVector(0.6, 0.35)
    r = hypot(pt[1], pt[2])
    x0 = k0 * r
    basis = FourierBessel(6; taylor_degree=p)                              # orders 0, 1, 1, 2, 2, 3
    cache = FourierBesselTaylorCache(basis, k0, [pt])

    for c in 1:basis.dim
        j = basis.orders[c]
        a = cache.coeffs[:, 1, c]
        @test length(a) == p + 1
        @test a[1] ≈ besselj(j, x0) atol = 1e-14
        @test a[2] ≈ r * fbr_dJ(j, x0) atol = 1e-14
        @test a[3] ≈ r^2 * fbr_ddJ(j, x0) / 2 atol = 1e-14
        for δ in (0.5, -0.5)
            @test fbr_poly(a, δ) ≈ besselj(j, (k0 + δ) * r) atol = 1e-12
        end
    end
end

@testset "FourierBesselTaylorCache construction" begin
    k0 = 3.0
    pts = fbr_pts()

    @testset "shape, center and degree" begin
        basis = FourierBessel(5; sym_x=-1, sym_y=1, taylor_degree=10)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        @test cache isa FourierBesselTaylorCache{Float64}
        @test cache.k0 == k0
        @test cache.degree == 10
        @test length(cache.r) == length(pts)
        @test size(cache.coeffs) == (11, length(pts), basis.dim)
        @test size(cache.ang) == (length(pts), basis.dim)
    end

    @testset "default degree comes from the basis" begin
        cache = FourierBesselTaylorCache(FourierBessel(3), k0, pts)
        @test cache.degree == 16
        @test size(cache.coeffs, 1) == 17
    end

    @testset "radial and angular data match the local polar coordinates" begin
        o, θ = SVector(0.15, -0.1), 0.4
        basis = FourierBessel(6; origin=o, rot_angle=θ, taylor_degree=8)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        r = [hypot(p[1] - o[1], p[2] - o[2]) for p in pts]
        φ_global = [atan(p[2] - o[2], p[1] - o[1]) for p in pts]
        φ_local = φ_global .- θ                                            # angular coordinate in the rotated frame
        @test cache.r ≈ r atol = 1e-14
        for c in 1:basis.dim
            j = basis.orders[c]
            T = basis.trig[c] > 0 ? cos : sin
            @test cache.ang[:, c] ≈ T.(j .* φ_local) atol = 1e-13
        end
    end

    @testset "value and derivative slots match the direct basis" begin
        basis = FourierBessel(5)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        @test cache.coeffs[1, :, :] .* cache.ang ≈ basis_fun(basis, 1:basis.dim, k0, pts) atol = 1e-14
        @test cache.coeffs[2, :, :] .* cache.ang ≈ dk_fun(basis, 1:basis.dim, k0, pts) atol = 1e-14
    end

    @testset "gradient data are stored only when gradients = true" begin
        nograd = FourierBesselTaylorCache(FourierBessel(4), k0, pts)
        @test isnothing(nograd.sinφ) && isnothing(nograd.cosφ) && isnothing(nograd.dang)

        o, θ = SVector(0.15, -0.1), 0.4
        basis = FourierBessel(6; origin=o, rot_angle=θ, gradients=true)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        r = [hypot(p[1] - o[1], p[2] - o[2]) for p in pts]
        # φ_local + rot_angle is the global polar angle of the point about the origin.
        @test cache.sinφ ≈ [(p[2] - o[2]) / rr for (p, rr) in zip(pts, r)] atol = 1e-14
        @test cache.cosφ ≈ [(p[1] - o[1]) / rr for (p, rr) in zip(pts, r)] atol = 1e-14
        @test size(cache.dang) == (length(pts), basis.dim)
        φ_local = [atan(p[2] - o[2], p[1] - o[1]) - θ for p in pts]
        for c in 1:basis.dim
            j = basis.orders[c]
            ∂T = basis.trig[c] > 0 ? -j .* sin.(j .* φ_local) : j .* cos.(j .* φ_local)
            @test cache.dang[:, c] ≈ ∂T atol = 1e-13
        end
    end

    @testset "basis_cache is the cache constructor" begin
        basis = FourierBessel(4; gradients=true)
        c1 = basis_cache(basis, k0, pts)
        c2 = FourierBesselTaylorCache(basis, k0, pts)
        @test c1 isa FourierBesselTaylorCache{Float64}
        @test (c1.k0, c1.degree) == (c2.k0, c2.degree)
        @test c1.coeffs == c2.coeffs && c1.ang == c2.ang && c1.dang == c2.dang
        @test basis_cache(basis, k0, pts; multithreaded=false).coeffs == c2.coeffs
    end

    @testset "threaded equals serial" begin
        basis = FourierBessel(6; gradients=true)
        threaded = FourierBesselTaylorCache(basis, k0, pts; multithreaded=true)
        serial = FourierBesselTaylorCache(basis, k0, pts; multithreaded=false)
        @test threaded.coeffs == serial.coeffs
        @test threaded.ang == serial.ang
        @test threaded.dang == serial.dang
        @test threaded.sinφ == serial.sinφ && threaded.cosφ == serial.cosφ
    end

    @testset "empty point set" begin
        basis = FourierBessel(3)
        cache = FourierBesselTaylorCache(basis, k0, SVector{2,Float64}[])
        @test size(cache.coeffs) == (17, 0, basis.dim)
        @test size(basis_matrix(cache, k0)) == (0, basis.dim)
    end

    @testset "wavenumber type must match the basis type" begin
        @test_throws MethodError FourierBesselTaylorCache(FourierBessel(2), 3.0f0, pts)
        @test_throws MethodError FourierBesselTaylorCache(FourierBessel(2), 3, pts)
    end
end

@testset "Taylor cache vs direct evaluation - unrestricted basis" begin
    k0 = 7.0
    pts = fbr_pts()
    basis = FourierBessel(9; gradients=true, taylor_degree=16)
    cache = FourierBesselTaylorCache(basis, k0, pts)
    idx = 1:basis.dim
    tol = 1e-12

    @testset "at the expansion center" begin
        @test norm(basis_matrix(cache, k0) - basis_matrix(basis, k0, pts)) < 1e-13
        @test norm(dk_matrix(cache, k0) - dk_matrix(basis, k0, pts)) < 1e-13
    end

    @testset "above the center" begin
        k = k0 + 0.3
        @test norm(basis_matrix(cache, k) - basis_matrix(basis, k, pts)) < tol
        @test norm(dk_matrix(cache, k) - dk_matrix(basis, k, pts)) < tol
    end

    @testset "below the center, at the reuse radius" begin
        k = k0 - basis.taylor_radius
        @test norm(basis_matrix(cache, k) - basis_matrix(basis, k, pts)) < 1e-11
        @test norm(dk_matrix(cache, k) - dk_matrix(basis, k, pts)) < 1e-11
    end

    @testset "spatial gradients" begin
        k = k0 + 0.3
        dX, dY = gradient(basis, idx, k, pts)
        cx, cy = gradient_matrices(cache, basis, k)
        @test norm(cx - dX) < tol && norm(cy - dY) < tol
        cB, cdx, cdy = basis_and_gradient_matrices(cache, basis, k)
        @test norm(cB - basis_matrix(basis, k, pts)) < tol
        @test norm(cdx - dX) < tol && norm(cdy - dY) < tol
        @test cdx == cx && cdy == cy
    end

    @testset "derivative of the cached basis equals the cached derivative" begin
        k, h = k0 + 0.1, 1e-6
        fd = (basis_matrix(cache, k + h) .- basis_matrix(cache, k - h)) ./ (2h)
        @test norm(fd - dk_matrix(cache, k)) < 1e-6
    end

    @testset "cached gradient equals finite differences of the cached basis in space" begin
        k, h = k0 + 0.1, 1e-6
        cx, cy = gradient_matrices(cache, basis, k)
        shifted(dx, dy) = basis_matrix(basis, k, [SVector(p[1] + dx, p[2] + dy) for p in pts])
        @test norm(cx - (shifted(h, 0.0) - shifted(-h, 0.0)) ./ (2h)) < 1e-6
        @test norm(cy - (shifted(0.0, h) - shifted(0.0, -h)) ./ (2h)) < 1e-6
    end

    @testset "multithreaded keyword does not change the values" begin
        k = k0 + 0.2
        @test basis_matrix(cache, k; multithreaded=false) == basis_matrix(cache, k)
        @test dk_matrix(cache, k; multithreaded=false) == dk_matrix(cache, k)
        @test gradient_matrices(cache, basis, k; multithreaded=false) == gradient_matrices(cache, basis, k)
        @test basis_and_gradient_matrices(cache, basis, k; multithreaded=false) == basis_and_gradient_matrices(cache, basis, k)
    end
end

@testset "Taylor cache vs direct evaluation - symmetry sectors" begin
    k0, k = 7.0, 7.3
    pts = fbr_pts()
    tol = 1e-12

    @testset "odd-odd sector" begin
        basis = FourierBessel(6; sym_x=-1, sym_y=-1, gradients=true)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        @test norm(basis_matrix(cache, k) - basis_matrix(basis, k, pts)) < tol
        @test norm(dk_matrix(cache, k) - dk_matrix(basis, k, pts)) < tol

        # The cached basis inherits the reflection parity of the sector: odd in x and odd in y.
        mirrored_x = FourierBesselTaylorCache(basis, k0, [SVector(-p[1], p[2]) for p in pts])
        mirrored_y = FourierBesselTaylorCache(basis, k0, [SVector(p[1], -p[2]) for p in pts])
        @test basis_matrix(mirrored_x, k) ≈ -basis_matrix(cache, k) atol = 1e-12
        @test basis_matrix(mirrored_y, k) ≈ -basis_matrix(cache, k) atol = 1e-12
        @test dk_matrix(mirrored_x, k) ≈ -dk_matrix(cache, k) atol = 1e-12
    end

    @testset "even-even sector" begin
        basis = FourierBessel(6; sym_x=1, sym_y=1, gradients=true)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        @test norm(basis_matrix(cache, k) - basis_matrix(basis, k, pts)) < tol
        mirrored_x = FourierBesselTaylorCache(basis, k0, [SVector(-p[1], p[2]) for p in pts])
        @test basis_matrix(mirrored_x, k) ≈ basis_matrix(cache, k) atol = 1e-12
    end

    @testset "mixed cos/sin half-plane sectors" begin
        for (sx, sy) in ((1, nothing), (-1, nothing), (nothing, 1), (nothing, -1))
            basis = FourierBessel(8; sym_x=sx, sym_y=sy, gradients=true)
            cache = FourierBesselTaylorCache(basis, k0, pts)
            @test norm(basis_matrix(cache, k - 0.6) - basis_matrix(basis, k - 0.6, pts)) < 1e-11
            @test norm(dk_matrix(cache, k) - dk_matrix(basis, k, pts)) < tol
            cx, cy = gradient_matrices(cache, basis, k)
            dX, dY = gradient(basis, 1:8, k, pts)
            @test norm(cx - dX) < tol && norm(cy - dY) < tol
        end
    end

    @testset "shifted and rotated frame" begin
        basis = FourierBessel(8; origin=SVector(0.15, -0.1), rot_angle=0.4, gradients=true)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        @test norm(basis_matrix(cache, k) - basis_matrix(basis, k, pts)) < tol
        @test norm(dk_matrix(cache, k) - dk_matrix(basis, k, pts)) < tol
        cx, cy = gradient_matrices(cache, basis, k)
        dX, dY = gradient(basis, 1:8, k, pts)
        @test norm(cx - dX) < tol && norm(cy - dY) < tol
    end
end

@testset "Taylor cache at the origin" begin
    k0, k = 7.0, 7.3
    basis = FourierBessel(7; gradients=true)
    cache = FourierBesselTaylorCache(basis, k0, [SVector(0.0, 0.0)])

    @testset "only the j = 0 cosine is non-zero, and it is constant in k" begin
        B = basis_matrix(cache, k)
        D = dk_matrix(cache, k)
        for c in 1:basis.dim
            @test B[1, c] == (basis.orders[c] == 0 ? 1.0 : 0.0)
            @test D[1, c] == 0.0
        end
    end

    @testset "the cached spatial gradient is exactly zero at r = 0" begin
        cx, cy = gradient_matrices(cache, basis, k)
        @test all(iszero, cx) && all(iszero, cy)
        _, bx, by = basis_and_gradient_matrices(cache, basis, k)
        @test all(iszero, bx) && all(iszero, by)
    end
end

@testset "Taylor truncation error decreases with the degree" begin
    k0 = 7.0
    pts = fbr_pts()
    k = k0 + 0.3
    err(deg) = begin
        basis = FourierBessel(6; taylor_degree=deg)
        maximum(abs, basis_matrix(FourierBesselTaylorCache(basis, k0, pts), k) .- basis_matrix(basis, k, pts))
    end
    e4, e8, e16 = err(4), err(8), err(16)
    @test e4 > 1e-6                      # a degree-4 polynomial cannot resolve a shift of 0.3 in k at kr ~ 15
    @test e8 < e4 / 100
    @test e16 < 1e-11
    @test e16 < e8
end

@testset "_weight_scaled_basis_and_dk_matrices" begin
    k0 = 7.0
    pts = fbr_pts()
    basis = FourierBessel(6)
    cache = FourierBesselTaylorCache(basis, k0, pts)
    k = k0 + 0.2
    w = [0.3, 0.1, 0.25, 0.4, 0.15, 0.2, 0.35]
    B0 = basis_matrix(cache, k)
    D0 = dk_matrix(cache, k)

    @testset "rows are scaled by sqrt(nsym * w)" begin
        B, D = _weight_scaled_basis_and_dk_matrices(cache, k, w, 2)
        @test B ≈ sqrt.(2 .* w) .* B0 atol = 1e-12
        @test D ≈ sqrt.(2 .* w) .* D0 atol = 1e-12
    end

    @testset "nsym = 1 gives plain sqrt(w) scaling" begin
        B, D = _weight_scaled_basis_and_dk_matrices(cache, k, w, 1)
        @test B ≈ sqrt.(w) .* B0 atol = 1e-12
        @test D ≈ sqrt.(w) .* D0 atol = 1e-12
    end

    @testset "agrees with the direct basis" begin
        B, D = _weight_scaled_basis_and_dk_matrices(cache, k, w, 2)
        @test B ≈ sqrt.(2 .* w) .* basis_matrix(basis, k, pts) atol = 1e-11
        @test D ≈ sqrt.(2 .* w) .* dk_matrix(basis, k, pts) atol = 1e-11
    end

    @testset "Gram matrix of the scaled basis is the weighted Gram matrix" begin
        B, _ = _weight_scaled_basis_and_dk_matrices(cache, k, w, 1)
        @test B' * B ≈ B0' * (w .* B0) atol = 1e-11
    end

    @testset "threaded equals serial" begin
        @test _weight_scaled_basis_and_dk_matrices(cache, k, w, 2; multithreaded=true) ==
              _weight_scaled_basis_and_dk_matrices(cache, k, w, 2; multithreaded=false)
    end

    @testset "weight length must match the cached points" begin
        @test_throws DimensionMismatch _weight_scaled_basis_and_dk_matrices(cache, k, w[1:6], 1)
        @test_throws DimensionMismatch _weight_scaled_basis_and_dk_matrices(cache, k, vcat(w, 0.1), 1)
    end
end

@testset "Taylor cache - gradient requirements and dimension checks" begin
    k0, k = 7.0, 7.3
    pts = fbr_pts()

    @testset "caches built from a basis without gradients give no spatial gradients" begin
        basis = FourierBessel(4)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        @test_throws ArgumentError gradient_matrices(cache, basis, k)
        @test_throws ArgumentError basis_and_gradient_matrices(cache, basis, k)
        err = try gradient_matrices(cache, basis, k); nothing catch e; e end
        @test err isa ArgumentError && occursin("gradients = true", err.msg)
        # Values and k-derivatives remain available.
        @test norm(basis_matrix(cache, k) - basis_matrix(basis, k, pts)) < 1e-12
        @test norm(dk_matrix(cache, k) - dk_matrix(basis, k, pts)) < 1e-12
    end

    @testset "the basis must have the cache dimension" begin
        basis = FourierBessel(4; gradients=true)
        cache = FourierBesselTaylorCache(basis, k0, pts)
        other = FourierBessel(5; gradients=true)
        @test_throws DimensionMismatch gradient_matrices(cache, other, k)
        @test_throws DimensionMismatch basis_and_gradient_matrices(cache, other, k)
    end
end

@testset "FourierBesselTaylorCache - type stability" begin
    k0, k = 7.0, 7.3
    pts = fbr_pts()
    basis = FourierBessel(5; gradients=true)
    cache = FourierBesselTaylorCache(basis, k0, pts; multithreaded=false)
    w = fill(0.2, length(pts))

    @test @inferred(FourierBesselTaylorCache(basis, k0, pts; multithreaded=false)) isa FourierBesselTaylorCache{Float64}
    @test @inferred(basis_cache(basis, k0, pts; multithreaded=false)) isa FourierBesselTaylorCache{Float64}
    @test @inferred(basis_matrix(cache, k)) isa Matrix{Float64}
    @test @inferred(dk_matrix(cache, k)) isa Matrix{Float64}
    @test @inferred(gradient_matrices(cache, basis, k)) isa Tuple{Matrix{Float64},Matrix{Float64}}
    @test @inferred(basis_and_gradient_matrices(cache, basis, k)) isa Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    @test @inferred(_weight_scaled_basis_and_dk_matrices(cache, k, w, 2)) isa Tuple{Matrix{Float64},Matrix{Float64}}
end

@testset "FourierBesselTaylorCache - Float32" begin
    kw32 = (origin=SVector(0.0f0, 0.0f0), rot_angle=0.0f0, taylor_degree=10, taylor_radius=0.5f0, taylor_tol=1.0f-6, gradients=true)
    basis32 = FourierBessel(5; kw32...)
    basis64 = FourierBessel(5; taylor_degree=10, gradients=true)
    pts32 = fbr_pts(Float32)
    pts64 = fbr_pts(Float64)
    k0_32, k32 = 3.0f0, 3.25f0
    cache32 = FourierBesselTaylorCache(basis32, k0_32, pts32)
    cache64 = FourierBesselTaylorCache(basis64, 3.0, pts64)

    @testset "storage and evaluation stay in Float32" begin
        @test cache32 isa FourierBesselTaylorCache{Float32}
        @test eltype(cache32.coeffs) == Float32 && eltype(cache32.ang) == Float32 && eltype(cache32.dang) == Float32
        @test size(cache32.coeffs) == (11, length(pts32), basis32.dim)
        @test @inferred(basis_matrix(cache32, k32)) isa Matrix{Float32}
        @test @inferred(dk_matrix(cache32, k32)) isa Matrix{Float32}
        @test @inferred(gradient_matrices(cache32, basis32, k32)) isa Tuple{Matrix{Float32},Matrix{Float32}}
        @test @inferred(basis_and_gradient_matrices(cache32, basis32, k32)) isa Tuple{Matrix{Float32},Matrix{Float32},Matrix{Float32}}
        @test @inferred(basis_cache(basis32, k0_32, pts32)) isa FourierBesselTaylorCache{Float32}
    end

    @testset "values agree with Float64" begin
        @test basis_matrix(cache32, k32) ≈ basis_matrix(cache64, 3.25) atol = 1e-4
        @test dk_matrix(cache32, k32) ≈ dk_matrix(cache64, 3.25) atol = 1e-4
        cx32, cy32 = gradient_matrices(cache32, basis32, k32)
        cx64, cy64 = gradient_matrices(cache64, basis64, 3.25)
        @test cx32 ≈ cx64 atol = 1e-3
        @test cy32 ≈ cy64 atol = 1e-3
    end

    @testset "the weight-scaled matrices stay in Float32" begin
        w = fill(0.2f0, length(pts32))
        B, D = _weight_scaled_basis_and_dk_matrices(cache32, k32, w, 2)
        @test B isa Matrix{Float32} && D isa Matrix{Float32}
    end
end
