# Unit tests for
#   src/basis/fourierbessel/corneradapted_recurrence.jl   (CornerAdaptedTaylorCache, basis_cache, and the cache methods of
#                                                          basis_matrix, dk_matrix, gradient_matrices,
#                                                          basis_and_gradient_matrices and
#                                                          _weight_scaled_basis_and_dk_matrices)
# The radial factor J_m(k r), m = ν i with ν = π / corner_angle (fractional unless π / corner_angle is an integer), of a basis
# function is expanded about k0 with normalized coefficients a_n = r^n J_m^(n)(k0 r) / n!. They are checked against the closed
# forms a_0 = J_m(x), a_1 = r J_m'(x), a_2 = r² J_m''(x) / 2 with J'' = -J'/x - (1 - m²/x²) J at x = k0 r, and the whole
# polynomial against SpecialFunctions.besselj of fractional order at shifted k. The angular data stored next to the coefficients
# are checked against sin/cos(mφ) at the (r, φ) the test places the points at, and against the global polar angle of the point
# (sinφ = (y - y0)/r, cosφ = (x - x0)/r, since φ_local + rot_angle is the global angle about the origin). The matrices built from
# the cache are compared with direct evaluation of the basis (which lives in corneradapted_tests.jl), for wedges with integer ν
# (π/2, π/3), fractional ν (2π/3) and a re-entrant wedge (3π/2, which needs a rotation_angle_discontinuity).
# The Horner kernels themselves are covered in solvers/taylor/recurrence_core_tests.jl.
# No eigen solves, no pinned wavefunction values. Helper functions use the prefix `car_`.

using StaticArrays: SVector
using LinearAlgebra: norm
using SpecialFunctions: besselj
using QuantumBilliards: CornerAdaptedTaylorCache, basis_cache, BasisCache, _weight_scaled_basis_and_dk_matrices,
                        gradient_matrices, basis_and_gradient_matrices

const car_ANGLES = (("π/2", π / 2), ("π/3", π / 3), ("2π/3", 2π / 3), ("3π/2", 3π / 2))
const car_FRACS = (0.15, 0.4, 0.6, 0.85)
const car_RADII = (0.15, 0.4, 0.7, 1.0, 1.3)

# For corner angles above π the branch cut of atan must lie outside the wedge.
car_branch(α) = α > π ? -α / 2 : 0.0

car_make(dim, α; origin=SVector(0.0, 0.0), rot=0.0, kw...) =
    CornerAdaptedFourierBessel(dim, α, origin, rot; rotation_angle_discontinuity=car_branch(α), kw...)

# (r, φ) of the wedge sample points, radius-major, and the points themselves.
car_rows(b) = [(r, f * b.corner_angle) for r in car_RADII for f in car_FRACS]
car_wedge_pts(b) = SVector{2,Float64}[b.origin + r * SVector(cos(b.rot_angle + φ), sin(b.rot_angle + φ)) for (r, φ) in car_rows(b)]

# Reflection of the points across the bisector of the wedge.
function car_mirror(b, pts)
    u = SVector(cos(b.rot_angle + b.corner_angle / 2), sin(b.rot_angle + b.corner_angle / 2))
    return [b.origin + 2 * sum((p - b.origin) .* u) * u - (p - b.origin) for p in pts]
end

# J_m' and J_m'' from the recurrence and the Bessel equation (independent of the package).
car_dJ(m, x) = (besselj(m - 1, x) - besselj(m + 1, x)) / 2
car_ddJ(m, x) = -car_dJ(m, x) / x - (1 - m^2 / x^2) * besselj(m, x)

# Plain polynomial sum Σ a_n δ^n.
car_poly(a, δ) = sum(a[n+1] * δ^n for n in 0:length(a)-1)

@testset "CornerAdaptedTaylorCache type" begin
    @test CornerAdaptedTaylorCache <: BasisCache
    @test fieldnames(CornerAdaptedTaylorCache) == (:k0, :degree, :r, :coeffs, :sin_mφ, :sinφ, :cosφ, :cos_mφ)
    cache = CornerAdaptedTaylorCache(2.5, 3, zeros(2), zeros(4, 2, 5), zeros(2, 5), nothing, nothing, nothing)
    @test cache isa CornerAdaptedTaylorCache{Float64}
    @test isconcretetype(typeof(cache))
    @test (cache.k0, cache.degree, size(cache.coeffs)) == (2.5, 3, (4, 2, 5))
    @test_throws MethodError CornerAdaptedTaylorCache(2.5, 3, zeros(Float32, 2), zeros(4, 2, 5), zeros(2, 5), nothing, nothing, nothing)
end

@testset "Taylor coefficients of J_m(kr) - closed forms for fractional order" begin
    k0, p = 3.0, 18
    pt = SVector(0.6, 0.35)
    r = hypot(pt[1], pt[2])
    x0 = k0 * r

    for (name, α) in car_ANGLES
        @testset "corner angle $name" begin
            basis = car_make(4, α; taylor_degree=p)
            # The point is at distance r from the corner, so the radial coefficients do not depend on its direction.
            cache = CornerAdaptedTaylorCache(basis, k0, [basis.origin + pt])
            for c in 1:basis.dim
                m = (π / α) * c
                a = cache.coeffs[:, 1, c]
                @test length(a) == p + 1
                @test a[1] ≈ besselj(m, x0) atol = 1e-14
                @test a[2] ≈ r * car_dJ(m, x0) atol = 1e-14
                @test a[3] ≈ r^2 * car_ddJ(m, x0) / 2 atol = 1e-14
                for δ in (0.5, -0.5)
                    @test car_poly(a, δ) ≈ besselj(m, (k0 + δ) * r) atol = 1e-12
                end
            end
        end
    end
end

@testset "CornerAdaptedTaylorCache construction" begin
    k0 = 3.0

    @testset "shape, center and degree" begin
        basis = car_make(5, 2π / 3; taylor_degree=10)
        pts = car_wedge_pts(basis)
        cache = CornerAdaptedTaylorCache(basis, k0, pts)
        @test cache isa CornerAdaptedTaylorCache{Float64}
        @test cache.k0 == k0
        @test cache.degree == 10
        @test length(cache.r) == length(pts)
        @test size(cache.coeffs) == (11, length(pts), basis.dim)
        @test size(cache.sin_mφ) == (length(pts), basis.dim)
    end

    @testset "default degree comes from the basis" begin
        basis = car_make(3, π / 3)
        cache = CornerAdaptedTaylorCache(basis, k0, car_wedge_pts(basis))
        @test cache.degree == 16
        @test size(cache.coeffs, 1) == 17
    end

    @testset "radial and angular data match the local polar coordinates" begin
        for (name, α) in car_ANGLES
            basis = car_make(5, α; origin=SVector(0.15, -0.1), rot=0.4, taylor_degree=8)
            pts = car_wedge_pts(basis)
            cache = CornerAdaptedTaylorCache(basis, k0, pts)
            @test cache.r ≈ [r for (r, _) in car_rows(basis)] atol = 1e-14
            for c in 1:basis.dim
                m = (π / α) * c
                @test cache.sin_mφ[:, c] ≈ [sin(m * φ) for (_, φ) in car_rows(basis)] atol = 1e-13
            end
        end
    end

    @testset "value and derivative slots match the direct basis" begin
        basis = car_make(5, 2π / 3; origin=SVector(0.15, -0.1), rot=0.4)
        pts = car_wedge_pts(basis)
        cache = CornerAdaptedTaylorCache(basis, k0, pts)
        @test cache.coeffs[1, :, :] .* cache.sin_mφ ≈ basis_fun(basis, 1:basis.dim, k0, pts) atol = 1e-14
        @test cache.coeffs[2, :, :] .* cache.sin_mφ ≈ dk_fun(basis, 1:basis.dim, k0, pts) atol = 1e-14
    end

    @testset "gradient data are stored only when gradients = true" begin
        nograd = car_make(4, π / 3)
        c0 = CornerAdaptedTaylorCache(nograd, k0, car_wedge_pts(nograd))
        @test isnothing(c0.sinφ) && isnothing(c0.cosφ) && isnothing(c0.cos_mφ)

        for (name, α) in car_ANGLES
            o, θ = SVector(0.15, -0.1), 0.4
            basis = car_make(5, α; origin=o, rot=θ, gradients=true)
            pts = car_wedge_pts(basis)
            cache = CornerAdaptedTaylorCache(basis, k0, pts)
            r = [hypot(p[1] - o[1], p[2] - o[2]) for p in pts]
            # φ_local + rot_angle is the global polar angle of the point about the origin.
            @test cache.sinφ ≈ [(p[2] - o[2]) / rr for (p, rr) in zip(pts, r)] atol = 1e-14
            @test cache.cosφ ≈ [(p[1] - o[1]) / rr for (p, rr) in zip(pts, r)] atol = 1e-14
            @test size(cache.cos_mφ) == (length(pts), basis.dim)
            for c in 1:basis.dim
                m = (π / α) * c
                @test cache.cos_mφ[:, c] ≈ [cos(m * φ) for (_, φ) in car_rows(basis)] atol = 1e-13
            end
        end
    end

    @testset "basis_cache is the cache constructor" begin
        basis = car_make(4, 2π / 3; gradients=true)
        pts = car_wedge_pts(basis)
        c1 = basis_cache(basis, k0, pts)
        c2 = CornerAdaptedTaylorCache(basis, k0, pts)
        @test c1 isa CornerAdaptedTaylorCache{Float64}
        @test (c1.k0, c1.degree) == (c2.k0, c2.degree)
        @test c1.coeffs == c2.coeffs && c1.sin_mφ == c2.sin_mφ && c1.cos_mφ == c2.cos_mφ
        @test basis_cache(basis, k0, pts; multithreaded=false).coeffs == c2.coeffs
    end

    @testset "threaded equals serial" begin
        basis = car_make(6, 2π / 3; gradients=true)
        pts = car_wedge_pts(basis)
        threaded = CornerAdaptedTaylorCache(basis, k0, pts; multithreaded=true)
        serial = CornerAdaptedTaylorCache(basis, k0, pts; multithreaded=false)
        @test threaded.coeffs == serial.coeffs
        @test threaded.sin_mφ == serial.sin_mφ
        @test threaded.cos_mφ == serial.cos_mφ
        @test threaded.sinφ == serial.sinφ && threaded.cosφ == serial.cosφ
        @test threaded.r == serial.r
    end

    @testset "empty point set" begin
        basis = car_make(3, π / 3)
        cache = CornerAdaptedTaylorCache(basis, k0, SVector{2,Float64}[])
        @test size(cache.coeffs) == (17, 0, basis.dim)
        @test size(basis_matrix(cache, k0)) == (0, basis.dim)
    end

    @testset "wavenumber type must match the basis type" begin
        basis = car_make(2, π / 3)
        pts = car_wedge_pts(basis)
        @test_throws MethodError CornerAdaptedTaylorCache(basis, 3.0f0, pts)
        @test_throws MethodError CornerAdaptedTaylorCache(basis, 3, pts)
    end
end

@testset "Taylor cache vs direct evaluation" begin
    k0 = 7.0
    tol = 1e-12

    for (name, α) in car_ANGLES
        @testset "corner angle $name" begin
            basis = car_make(7, α; origin=SVector(0.15, -0.1), rot=0.4, gradients=true, taylor_degree=16)
            pts = car_wedge_pts(basis)
            cache = CornerAdaptedTaylorCache(basis, k0, pts)
            idx = 1:basis.dim

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
        end
    end

    @testset "derivative of the cached basis equals the cached derivative" begin
        basis = car_make(7, 2π / 3; gradients=true)
        cache = CornerAdaptedTaylorCache(basis, k0, car_wedge_pts(basis))
        k, h = k0 + 0.1, 1e-6
        fd = (basis_matrix(cache, k + h) .- basis_matrix(cache, k - h)) ./ (2h)
        @test norm(fd - dk_matrix(cache, k)) < 1e-6
    end

    @testset "cached gradient equals finite differences of the direct basis in space" begin
        basis = car_make(7, 2π / 3; origin=SVector(0.15, -0.1), rot=0.4, gradients=true)
        pts = car_wedge_pts(basis)
        cache = CornerAdaptedTaylorCache(basis, k0, pts)
        k, h = k0 + 0.1, 1e-6
        cx, cy = gradient_matrices(cache, basis, k)
        shifted(dx, dy) = basis_matrix(basis, k, [SVector(p[1] + dx, p[2] + dy) for p in pts])
        @test norm(cx - (shifted(h, 0.0) - shifted(-h, 0.0)) ./ (2h)) < 1e-6
        @test norm(cy - (shifted(0.0, h) - shifted(0.0, -h)) ./ (2h)) < 1e-6
    end

    @testset "multithreaded keyword does not change the values" begin
        basis = car_make(7, 2π / 3; gradients=true)
        cache = CornerAdaptedTaylorCache(basis, k0, car_wedge_pts(basis))
        k = k0 + 0.2
        @test basis_matrix(cache, k; multithreaded=false) == basis_matrix(cache, k)
        @test dk_matrix(cache, k; multithreaded=false) == dk_matrix(cache, k)
        @test gradient_matrices(cache, basis, k; multithreaded=false) == gradient_matrices(cache, basis, k)
        @test basis_and_gradient_matrices(cache, basis, k; multithreaded=false) == basis_and_gradient_matrices(cache, basis, k)
    end
end

@testset "Taylor cache - symmetry handling" begin
    k0, k = 7.0, 7.3
    refl = BilliardGeometry.AbsReflection

    @testset "attached reflection symmetries do not change the cache" begin
        plain = car_make(5, π / 3; gradients=true)
        sym = CornerAdaptedFourierBessel(5, π / 3, plain.origin, plain.rot_angle, refl[XAxisReflection()]; gradients=true)
        pts = car_wedge_pts(plain)
        c1 = CornerAdaptedTaylorCache(plain, k0, pts)
        c2 = CornerAdaptedTaylorCache(sym, k0, pts)
        @test c1.coeffs == c2.coeffs && c1.sin_mφ == c2.sin_mφ && c1.cos_mφ == c2.cos_mφ
        @test basis_matrix(c1, k) == basis_matrix(c2, k)
    end

    @testset "the cached basis has the parity (-1)^(i+1) about the bisector of the wedge" begin
        # sin(ν i (α - φ)) = (-1)^(i + 1) sin(ν i φ); the cache inherits it from the angular factor.
        for (name, α) in car_ANGLES
            basis = car_make(5, α; origin=SVector(0.2, -0.1), rot=0.5, gradients=true)
            pts = car_wedge_pts(basis)
            cache = CornerAdaptedTaylorCache(basis, k0, pts)
            mirrored = CornerAdaptedTaylorCache(basis, k0, car_mirror(basis, pts))
            B, Bm = basis_matrix(cache, k), basis_matrix(mirrored, k)
            D, Dm = dk_matrix(cache, k), dk_matrix(mirrored, k)
            for i in 1:5
                @test Bm[:, i] ≈ (-1)^(i + 1) .* B[:, i] atol = 1e-12
                @test Dm[:, i] ≈ (-1)^(i + 1) .* D[:, i] atol = 1e-12
            end
        end
    end
end

@testset "Taylor cache at the corner" begin
    k0, k = 7.0, 7.3

    for (name, α) in car_ANGLES
        @testset "corner angle $name: every cached function and derivative vanishes" begin
            basis = car_make(5, α; gradients=true)
            cache = CornerAdaptedTaylorCache(basis, k0, [basis.origin])
            @test all(iszero, basis_matrix(cache, k))
            @test all(iszero, dk_matrix(cache, k))
            cx, cy = gradient_matrices(cache, basis, k)
            @test all(iszero, cx) && all(iszero, cy)
            B, bx, by = basis_and_gradient_matrices(cache, basis, k)
            @test all(iszero, B) && all(iszero, bx) && all(iszero, by)
        end
    end

    @testset "the cached dk is finite at the corner of a re-entrant wedge where the direct dk is NaN" begin
        # See the source finding in corneradapted_tests.jl: the direct dk_fun gives 0 * Inf = NaN for ν < 1 at r = 0.
        basis = car_make(4, 3π / 2)
        cache = CornerAdaptedTaylorCache(basis, k0, [basis.origin])
        @test all(isfinite, dk_matrix(cache, k))
        @test isnan(dk_fun(basis, 1:1, k, [basis.origin])[1])
    end

    @testset "the direct matrices agree with the cache at the corner of a wedge below π" begin
        basis = car_make(5, π / 3; gradients=true)
        cache = CornerAdaptedTaylorCache(basis, k0, [basis.origin])
        @test basis_matrix(cache, k) == basis_matrix(basis, k, [basis.origin])
        @test dk_matrix(cache, k) == dk_matrix(basis, k, [basis.origin])
        @test gradient_matrices(cache, basis, k) == gradient_matrices(basis, k, [basis.origin])
    end
end

@testset "Taylor truncation error decreases with the degree" begin
    k0 = 7.0
    k = k0 + 0.3
    α = 2π / 3
    err(deg) = begin
        basis = car_make(6, α; taylor_degree=deg)
        pts = car_wedge_pts(basis)
        maximum(abs, basis_matrix(CornerAdaptedTaylorCache(basis, k0, pts), k) .- basis_matrix(basis, k, pts))
    end
    e2, e6, e16 = err(2), err(6), err(16)
    @test e2 > 1e-4                      # a degree-2 polynomial cannot resolve a shift of 0.3 in k at kr ~ 9
    @test e6 < e2 / 10
    @test e16 < 1e-11
    @test e16 < e6
end

@testset "_weight_scaled_basis_and_dk_matrices" begin
    k0 = 7.0
    basis = car_make(6, 2π / 3)
    pts = car_wedge_pts(basis)
    cache = CornerAdaptedTaylorCache(basis, k0, pts)
    k = k0 + 0.2
    w = [0.1 + 0.02 * j for j in 1:length(pts)]
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
        @test_throws DimensionMismatch _weight_scaled_basis_and_dk_matrices(cache, k, w[1:end-1], 1)
        @test_throws DimensionMismatch _weight_scaled_basis_and_dk_matrices(cache, k, vcat(w, 0.1), 1)
    end
end

@testset "Taylor cache - gradient requirements and dimension checks" begin
    k0, k = 7.0, 7.3

    @testset "caches built from a basis without gradients give no spatial gradients" begin
        basis = car_make(4, 2π / 3)
        pts = car_wedge_pts(basis)
        cache = CornerAdaptedTaylorCache(basis, k0, pts)
        @test_throws ArgumentError gradient_matrices(cache, basis, k)
        @test_throws ArgumentError basis_and_gradient_matrices(cache, basis, k)
        err = try gradient_matrices(cache, basis, k); nothing catch e; e end
        @test err isa ArgumentError && occursin("gradients = true", err.msg)
        # Values and k-derivatives remain available.
        @test norm(basis_matrix(cache, k) - basis_matrix(basis, k, pts)) < 1e-12
        @test norm(dk_matrix(cache, k) - dk_matrix(basis, k, pts)) < 1e-12
    end

    @testset "the basis must have the cache dimension" begin
        basis = car_make(4, 2π / 3; gradients=true)
        cache = CornerAdaptedTaylorCache(basis, k0, car_wedge_pts(basis))
        other = car_make(5, 2π / 3; gradients=true)
        @test_throws DimensionMismatch gradient_matrices(cache, other, k)
        @test_throws DimensionMismatch basis_and_gradient_matrices(cache, other, k)
    end
end

@testset "CornerAdaptedTaylorCache - type stability" begin
    k0, k = 7.0, 7.3
    basis = car_make(5, 2π / 3; gradients=true)
    pts = car_wedge_pts(basis)
    cache = CornerAdaptedTaylorCache(basis, k0, pts; multithreaded=false)
    w = fill(0.2, length(pts))

    @test @inferred(CornerAdaptedTaylorCache(basis, k0, pts; multithreaded=false)) isa CornerAdaptedTaylorCache{Float64}
    @test @inferred(basis_cache(basis, k0, pts; multithreaded=false)) isa CornerAdaptedTaylorCache{Float64}
    @test @inferred(basis_matrix(cache, k)) isa Matrix{Float64}
    @test @inferred(dk_matrix(cache, k)) isa Matrix{Float64}
    @test @inferred(gradient_matrices(cache, basis, k)) isa Tuple{Matrix{Float64},Matrix{Float64}}
    @test @inferred(basis_and_gradient_matrices(cache, basis, k)) isa Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    @test @inferred(_weight_scaled_basis_and_dk_matrices(cache, k, w, 2)) isa Tuple{Matrix{Float64},Matrix{Float64}}
end

@testset "CornerAdaptedTaylorCache - Float32" begin
    basis64 = car_make(5, 2π / 3; origin=SVector(0.15, -0.1), rot=0.4, taylor_degree=10, gradients=true)
    basis32 = QuantumBilliards.toFloat32(basis64)
    pts64 = car_wedge_pts(basis64)
    pts32 = [SVector{2,Float32}(p) for p in pts64]
    k0_32, k32 = 3.0f0, 3.25f0
    cache32 = CornerAdaptedTaylorCache(basis32, k0_32, pts32)
    cache64 = CornerAdaptedTaylorCache(basis64, 3.0, pts64)

    @testset "storage and evaluation stay in Float32" begin
        @test cache32 isa CornerAdaptedTaylorCache{Float32}
        @test eltype(cache32.coeffs) == Float32 && eltype(cache32.sin_mφ) == Float32 && eltype(cache32.cos_mφ) == Float32
        @test size(cache32.coeffs) == (11, length(pts32), basis32.dim)
        @test @inferred(basis_matrix(cache32, k32)) isa Matrix{Float32}
        @test @inferred(dk_matrix(cache32, k32)) isa Matrix{Float32}
        @test @inferred(gradient_matrices(cache32, basis32, k32)) isa Tuple{Matrix{Float32},Matrix{Float32}}
        @test @inferred(basis_and_gradient_matrices(cache32, basis32, k32)) isa Tuple{Matrix{Float32},Matrix{Float32},Matrix{Float32}}
        @test @inferred(basis_cache(basis32, k0_32, pts32)) isa CornerAdaptedTaylorCache{Float32}
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
