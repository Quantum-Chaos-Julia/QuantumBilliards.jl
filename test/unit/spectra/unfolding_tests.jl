# Unit tests for src/spectra/unfolding.jl
#   corner_correction, weyl_law, k_at_state, state_at_k, spectral_density, k_range_for_states
# Reference billiard: RectangleBilliard(a, b) is the FULL 2a x 2b rectangle (D2 symmetric),
#   full:        A = 4ab, L = 4(a+b), four right-angle corners  ->  N(k) = (A k^2 - L k)/(4pi) + 1/4
#   fundamental: area / 4, boundary curves reported by `get_boundary_curves`
# No eigen-solves: every expected value is analytic or a consistency identity.

using QuantumBilliards: k_at_state, corner_correction

@testset "corner_correction" begin
    # per-corner term (pi^2 - c^2)/(24 pi c)
    @test corner_correction([pi / 2, pi / 2, pi / 2, pi / 2]) ≈ 1 / 4   # rectangle: 4 * 1/16
    @test corner_correction([pi / 3, pi / 3, pi / 3]) ≈ 1 / 3           # equilateral triangle: 3 * 1/9
    @test corner_correction([pi]) ≈ 0 atol = 1e-15                      # a flat angle contributes nothing
    @test corner_correction([pi / 2]) ≈ corner_correction([pi / 2, pi / 2]) / 2
    @test corner_correction([pi / 4]) > corner_correction([pi / 2])     # sharper corners contribute more
    @test corner_correction(Float64[]) == 0
end

@testset "weyl_law" begin
    A, L = 3.0, 7.0

    @testset "closed form without corners" begin
        for k in (0.5, 4.0, 20.0)
            @test weyl_law(k, A, L) ≈ (A * k^2 - L * k) / (4pi)
        end
        @test weyl_law(0.0, A, L) == 0.0
        @test weyl_law(L / A, A, L) ≈ 0 atol = 1e-14                    # root of the leading terms
        @test @inferred(weyl_law(4.0, A, L)) isa Float64
    end

    @testset "corner correction is an additive constant" begin
        angles = [pi / 2, pi / 2, pi / 2, pi / 2]
        @test weyl_law(10.0, A, L, angles) ≈ weyl_law(10.0, A, L) + 1 / 4
        @test weyl_law(10.0, A, L, angles) - weyl_law(5.0, A, L, angles) ≈ weyl_law(10.0, A, L) - weyl_law(5.0, A, L)
        @test @inferred(weyl_law(10.0, A, L, angles)) isa Float64
    end

    @testset "vectorised over k" begin
        ks = [2.0, 5.0, 11.0]
        @test weyl_law(ks, A, L) ≈ [weyl_law(k, A, L) for k in ks]
        @test weyl_law(ks, A, L, [pi / 3, pi / 3, pi / 3]) ≈ weyl_law(ks, A, L) .+ 1 / 3
    end

    @testset "monotone beyond the minimum" begin
        ks = range(L / (2A) + 0.1, 50.0; length=40)
        @test issorted(weyl_law(collect(ks), A, L))
    end
end

@testset "k_at_state" begin
    A, L = 3.0, 7.0

    @testset "closed form without corners" begin
        for N in (1, 10, 250)
            @test k_at_state(N, A, L) ≈ (L + sqrt(L^2 + 16pi * A * N)) / (2A)
        end
        @test k_at_state(0, A, L) ≈ L / A                               # N = 0 at the nonzero root
        @test @inferred(k_at_state(10, A, L)) isa Float64
    end

    @testset "inverts weyl_law" begin
        for N in (1.0, 10.0, 250.0, 1e4)
            @test weyl_law(k_at_state(N, A, L), A, L) ≈ N
        end
        for k in (3.0, 9.0, 40.0)
            @test k_at_state(weyl_law(k, A, L), A, L) ≈ k
        end
    end

    @testset "inverts weyl_law with corner correction" begin
        angles = [pi / 2, pi / 2, pi / 2, pi / 2]
        for N in (1.0, 10.0, 250.0)
            @test weyl_law(k_at_state(N, A, L, angles), A, L, angles) ≈ N
        end
        @test k_at_state(10, A, L, angles) < k_at_state(10, A, L)       # N = weyl(k) + c: a positive c lowers the k reaching N
        @test @inferred(k_at_state(10, A, L, angles)) isa Float64
    end

    @testset "strictly increasing in state" begin
        ks = [k_at_state(N, A, L) for N in 1:50]
        @test all(diff(ks) .> 0)
    end

    @testset "billiard method matches rectangle closed form" begin
        a, b = 1.0, 0.5
        r = RectangleBilliard(a, b)
        Afull, Lfull = 4a * b, 4 * (a + b)
        N = 40
        @test k_at_state(N, r; fundamental=false) ≈ k_at_state(N, Afull, Lfull, fill(pi / 2, 4))
        @test weyl_law(k_at_state(N, r; fundamental=false), Afull, Lfull, fill(pi / 2, 4)) ≈ N
        @test k_at_state(N, r) ≈ k_at_state(N, fundamental_area(r), sum(c.length for c in get_boundary_curves(r)), corner_angles(r))
        @test k_at_state(N, r) > k_at_state(N, r; fundamental=false)    # smaller area -> larger k for the same N
    end
end

@testset "state_at_k" begin
    A, L = 3.0, 7.0
    angles = [pi / 2, pi / 2, pi / 2, pi / 2]

    @testset "area/length methods are weyl_law" begin
        @test state_at_k(12.0, A, L) == weyl_law(12.0, A, L)
        @test state_at_k(12.0, A, L, angles) == weyl_law(12.0, A, L, angles)
        @test @inferred(state_at_k(12.0, A, L)) isa Float64
    end

    @testset "round trip with k_at_state" begin
        for N in (5.0, 80.0, 1e3)
            @test state_at_k(k_at_state(N, A, L, angles), A, L, angles) ≈ N
        end
        for k in (4.0, 15.0, 60.0)
            @test k_at_state(state_at_k(k, A, L, angles), A, L, angles) ≈ k
        end
    end

    @testset "billiard method on a rectangle" begin
        a, b = 1.0, 0.5
        r = RectangleBilliard(a, b)
        Afull, Lfull = 4a * b, 4 * (a + b)
        k = 25.0
        @test state_at_k(k, r; fundamental=false) ≈ (Afull * k^2 - Lfull * k) / (4pi) + 1 / 4
        @test state_at_k(k, r) ≈ state_at_k(k, fundamental_area(r), sum(c.length for c in get_boundary_curves(r)), corner_angles(r))
        for N in (10, 100)
            @test state_at_k(k_at_state(N, r), r) ≈ N
            @test state_at_k(k_at_state(N, r; fundamental=false), r; fundamental=false) ≈ N
        end
    end
end

@testset "spectral_density" begin
    A, L = 3.0, 7.0

    @testset "closed form" begin
        for k in (4.0, 10.0, 40.0)
            @test spectral_density(k, A, L) ≈ (A * k - L / 2) / (2pi)
        end
        @test spectral_density(L / (2A), A, L) ≈ 0 atol = 1e-14
        @test @inferred(spectral_density(10.0, A, L)) isa Float64
    end

    @testset "derivative of weyl_law" begin
        h = 1e-3    # weyl_law is quadratic in k, so the central difference is exact up to rounding
        for k in (4.0, 10.0, 40.0)
            fd = (weyl_law(k + h, A, L) - weyl_law(k - h, A, L)) / (2h)
            @test spectral_density(k, A, L) ≈ fd rtol = 1e-8
        end
    end

    @testset "derivative of weyl_law with corner correction" begin
        angles = [pi / 2, pi / 2, pi / 2, pi / 2]
        h = 1e-3
        k = 12.0
        fd = (weyl_law(k + h, A, L, angles) - weyl_law(k - h, A, L, angles)) / (2h)
        @test spectral_density(k, A, L) ≈ fd rtol = 1e-8
    end

    @testset "billiard method on a rectangle" begin
        r = RectangleBilliard(1.0, 0.5)
        h = 1e-3
        k = 30.0
        for fundamental in (true, false)
            fd = (state_at_k(k + h, r; fundamental) - state_at_k(k - h, r; fundamental)) / (2h)
            @test spectral_density(k, r; fundamental) ≈ fd rtol = 1e-8
        end
        @test spectral_density(k, r; fundamental=false) ≈ (4 * 1.0 * 0.5 * k - 4 * (1.0 + 0.5) / 2) / (2pi) rtol = 1e-12
        @test spectral_density(k, r; fundamental=false) > spectral_density(k, r)
    end

    @testset "mean density integrates to the state count" begin
        r = RectangleBilliard(1.0, 0.5)
        k1, k2 = 20.0, 30.0
        n = 2000
        ks = range(k1, k2; length=n + 1)
        integral = sum((spectral_density(ks[i], r) + spectral_density(ks[i+1], r)) / 2 * step(ks) for i in 1:n)
        @test integral ≈ state_at_k(k2, r) - state_at_k(k1, r) rtol = 1e-8
    end
end

@testset "k_range_for_states" begin
    r = RectangleBilliard(1.0, 0.5)

    @testset "endpoints are k_at_state" begin
        k1, k2 = k_range_for_states(r, 10, 100)
        @test k1 == k_at_state(10, r)
        @test k2 == k_at_state(100, r)
        @test k1 < k2
        @test state_at_k(k1, r) ≈ 10
        @test state_at_k(k2, r) ≈ 100
    end

    @testset "fundamental keyword" begin
        k1, k2 = k_range_for_states(r, 10, 100; fundamental=false)
        @test (k1, k2) == (k_at_state(10, r; fundamental=false), k_at_state(100, r; fundamental=false))
        kf1, kf2 = k_range_for_states(r, 10, 100)
        @test k1 != kf1 && k2 != kf2
    end

    @testset "degenerate and reversed ranges" begin
        k1, k2 = k_range_for_states(r, 25, 25)
        @test k1 == k2
        k1, k2 = k_range_for_states(r, 100, 10)
        @test k1 > k2
    end

    @testset "argument types" begin
        @test_throws MethodError k_range_for_states(r, 1.5, 10)
        @test @inferred(k_range_for_states(r, 10, 100)) isa Tuple{Float64,Float64}
        @test k_range_for_states(r, 10, 100) isa Tuple{Float64,Float64}
    end
end
