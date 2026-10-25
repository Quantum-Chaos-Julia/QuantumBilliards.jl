# Unit tests for
#   src/solvers/taylor/recurrence_core.jl
#     _horner (vector / matrix-column / 3D-array methods)
#     _horner_with_derivative (vector / matrix-column / 3D-array methods)
#     _horner_with_12_derivatives (vector / matrix-column / 3D-array methods)
#     power_to_cheb
#     radial1_taylor!, radial0_taylor!, radial0_k_taylor!
#     TaylorWorkspace
# Horner evaluators are compared with explicit monomial sums (and their termwise derivatives). power_to_cheb is
# checked against closed-form Chebyshev expansions of x^2, x^3 and against direct evaluation of random-free
# polynomials with T_j(t) = cos(j acos t). The radial recurrences are compared with SpecialFunctions at small shifts.

using SpecialFunctions
using QuantumBilliards: _horner, _horner_with_derivative, _horner_with_12_derivatives, power_to_cheb,
                        radial1_taylor!, radial0_taylor!, radial0_k_taylor!, TaylorWorkspace

# Explicit monomial evaluation of Σ a[l+1] δ^l and its first two derivatives.
rc_poly(a, δ) = sum(a[l+1]*δ^l for l in 0:length(a)-1)
rc_dpoly(a, δ) = sum(l*a[l+1]*δ^(l-1) for l in 1:length(a)-1)
rc_ddpoly(a, δ) = sum(l*(l-1)*a[l+1]*δ^(l-2) for l in 2:length(a)-1)

# Fixed deterministic complex coefficients (no RNG) of a given length.
rc_coeffs(n) = ComplexF64[cis(0.7*l)/(l + 1) + 0.1*l*im for l in 1:n]

@testset "_horner" begin
    a = rc_coeffs(7)

    @testset "vector method matches the monomial sum" begin
        for δ in (0.0, 0.3, -0.45, 0.2 + 0.1im)
            @test _horner(a, δ) ≈ rc_poly(a, δ) rtol=1e-13
        end
        @test _horner(a, 0.0) == a[1]                         # value at the expansion point
        @test _horner([ComplexF64(2.5)], 0.7) == 2.5          # constant polynomial
        @test _horner(a, 0.3) ≈ evalpoly(0.3, a) rtol=1e-13   # Base.evalpoly agrees
        @test @inferred(_horner(a, 0.3)) isa ComplexF64
        @test @inferred(_horner(a, 0.3 + 0.2im)) isa ComplexF64
    end

    @testset "matrix-column method evaluates one column of C" begin
        C = hcat(rc_coeffs(6), 2 .* rc_coeffs(6) .+ 1, reverse(rc_coeffs(6)))
        for j in 1:3, δ in (0.25, -0.6, 0.1 - 0.3im)
            @test _horner(C, j, δ) ≈ rc_poly(C[:, j], δ) rtol=1e-13
            @test _horner(C, j, δ) ≈ _horner(C[:, j], δ) rtol=1e-14
        end
        @test @inferred(_horner(C, 2, 0.25)) isa ComplexF64
    end

    @testset "3D real-array method evaluates C[:, j, c]" begin
        for T in (Float64, Float32)
            C = T[(l + 1)/(2 + j) - 0.3*c*l for l in 0:5, j in 1:2, c in 1:3]
            for j in 1:2, c in 1:3, δ in T.((0.0, 0.2, -0.35))
                @test _horner(C, j, c, δ) ≈ rc_poly(C[:, j, c], δ) rtol=sqrt(eps(T))
            end
            @test @inferred(_horner(C, 1, 2, T(0.2))) isa T
        end
    end
end

@testset "_horner_with_derivative" begin
    a = rc_coeffs(7)

    @testset "vector method returns value and first derivative" begin
        for δ in (0.0, 0.3, -0.45, 0.2 + 0.1im)
            v, dv = _horner_with_derivative(a, δ)
            @test v ≈ rc_poly(a, δ) rtol=1e-13
            @test dv ≈ rc_dpoly(a, δ) rtol=1e-12
        end
        v0, dv0 = _horner_with_derivative(a, 0.0)
        @test v0 == a[1]
        @test dv0 == a[2]                                      # P'(0) = a₁
        @test _horner_with_derivative([ComplexF64(3.0)], 0.4) == (3.0 + 0im, 0.0 + 0im)
        @test @inferred(_horner_with_derivative(a, 0.3)) isa Tuple{ComplexF64,ComplexF64}
    end

    @testset "matrix-column method" begin
        C = hcat(rc_coeffs(6), 2 .* rc_coeffs(6) .+ 1)
        for j in 1:2, δ in (0.25, -0.6, 0.1 - 0.3im)
            v, dv = _horner_with_derivative(C, j, δ)
            @test v ≈ rc_poly(C[:, j], δ) rtol=1e-13
            @test dv ≈ rc_dpoly(C[:, j], δ) rtol=1e-12
        end
        @test @inferred(_horner_with_derivative(C, 1, 0.25)) isa Tuple{ComplexF64,ComplexF64}
    end

    @testset "3D real-array method" begin
        for T in (Float64, Float32)
            C = T[(l + 1)/(2 + j) - 0.3*c*l for l in 0:5, j in 1:2, c in 1:3]
            for j in 1:2, c in 1:3, δ in T.((0.0, 0.2, -0.35))
                v, dv = _horner_with_derivative(C, j, c, δ)
                @test v ≈ rc_poly(C[:, j, c], δ) rtol=sqrt(eps(T))
                @test dv ≈ rc_dpoly(C[:, j, c], δ) rtol=sqrt(eps(T)) atol=sqrt(eps(T))
            end
            @test @inferred(_horner_with_derivative(C, 1, 2, T(0.2))) isa Tuple{T,T}
        end
    end
end

@testset "_horner_with_12_derivatives" begin
    a = rc_coeffs(8)

    @testset "vector method returns value, first and second derivative" begin
        for δ in (0.0, 0.3, -0.45, 0.2 + 0.1im)
            v, dv, ddv = _horner_with_12_derivatives(a, δ)
            @test v ≈ rc_poly(a, δ) rtol=1e-13
            @test dv ≈ rc_dpoly(a, δ) rtol=1e-12
            @test ddv ≈ rc_ddpoly(a, δ) rtol=1e-12
        end
        _, dv0, ddv0 = _horner_with_12_derivatives(a, 0.0)
        @test dv0 == a[2]
        @test ddv0 == 2*a[3]                                   # P''(0) = 2a₂
        @test all(_horner_with_12_derivatives([ComplexF64(3.0), ComplexF64(1.0)], 0.4) .≈ (3.4 + 0im, 1.0 + 0im, 0.0 + 0im))   # 3 + δ
        @test @inferred(_horner_with_12_derivatives(a, 0.3)) isa Tuple{ComplexF64,ComplexF64,ComplexF64}
    end

    @testset "matrix-column method" begin
        C = hcat(rc_coeffs(7), 2 .* rc_coeffs(7) .+ 1)
        for j in 1:2, δ in (0.25, -0.6, 0.1 - 0.3im)
            v, dv, ddv = _horner_with_12_derivatives(C, j, δ)
            @test v ≈ rc_poly(C[:, j], δ) rtol=1e-13
            @test dv ≈ rc_dpoly(C[:, j], δ) rtol=1e-12
            @test ddv ≈ rc_ddpoly(C[:, j], δ) rtol=1e-12
        end
        @test @inferred(_horner_with_12_derivatives(C, 1, 0.25)) isa Tuple{ComplexF64,ComplexF64,ComplexF64}
    end

    @testset "3D complex-array method with real shift" begin
        C = ComplexF64[cis(0.4*l*i)/(l + j) for l in 0:6, i in 1:2, j in 1:3]
        for i in 1:2, j in 1:3, δ in (0.0, 0.2, -0.35)
            v, dv, ddv = _horner_with_12_derivatives(C, i, j, δ)
            @test v ≈ rc_poly(C[:, i, j], δ) rtol=1e-13
            @test dv ≈ rc_dpoly(C[:, i, j], δ) rtol=1e-12
            @test ddv ≈ rc_ddpoly(C[:, i, j], δ) rtol=1e-12
        end
        @test @inferred(_horner_with_12_derivatives(C, 1, 2, 0.2)) isa Tuple{ComplexF64,ComplexF64,ComplexF64}
    end
end

@testset "power_to_cheb" begin
    @testset "closed-form columns for unit half-width" begin
        C = power_to_cheb(3, 1.0)
        @test size(C) == (4, 4)
        @test C[:, 1] == [1.0, 0.0, 0.0, 0.0]                  # 1 = T₀
        @test C[:, 2] == [0.0, 1.0, 0.0, 0.0]                  # t = T₁
        @test C[:, 3] ≈ [0.5, 0.0, 0.5, 0.0]                   # t² = (T₀+T₂)/2
        @test C[:, 4] ≈ [0.0, 0.75, 0.0, 0.25]                 # t³ = (3T₁+T₃)/4
    end

    @testset "triangular, parity and leading coefficient structure" begin
        p, Δ = 8, 0.37
        C = power_to_cheb(p, Δ)
        @test size(C) == (p + 1, p + 1)
        @test all(C[j+1, l+1] == 0 for l in 0:p for j in l+1:p)                   # tˡ has degree l
        @test all(C[j+1, l+1] == 0 for l in 0:p, j in 0:p if isodd(j + l))        # parity of tˡ
        @test C[1, 1] == 1
        @test all(isapprox(C[l+1, l+1], Δ^l * 2.0^(1 - l); rtol=1e-14) for l in 1:p)   # tˡ = 2^{1-l}T_l + ...
        @test all(C[j+1, l+1] >= 0 for l in 0:p, j in 0:p)                        # powers of t have nonnegative Chebyshev coefficients
    end

    @testset "column l scales as Δˡ" begin
        p = 6
        C1 = power_to_cheb(p, 1.0)
        for Δ in (0.5, 2.0)
            CΔ = power_to_cheb(p, Δ)
            @test all(isapprox(CΔ[:, l+1], Δ^l .* C1[:, l+1]; rtol=1e-14) for l in 0:p)
        end
    end

    @testset "Chebyshev expansion reproduces the polynomial F(Δt)" begin
        p, Δ = 6, 0.37
        F = [1.0, -0.5, 0.25, 2.0, -1.5, 0.75, 0.3]
        B = power_to_cheb(p, Δ) * F
        for t in range(-1, 1; length=9)
            cheb_sum = sum(B[j+1] * cos(j * acos(t)) for j in 0:p)
            @test cheb_sum ≈ evalpoly(Δ*t, F) atol=1e-13
        end
    end

    @testset "degree zero and type stability" begin
        @test power_to_cheb(0, 0.4) == reshape([1.0], 1, 1)
        @test @inferred(power_to_cheb(5, 0.4)) isa Matrix{Float64}
    end
end

@testset "radial1_taylor!" begin
    # f(k) = k Z₁(k r); f₀ = k Z₁(kr), f₁ = k r Z₀(kr)
    k, r, p, δ = 3.0, 0.7, 16, 0.08
    f = zeros(ComplexF64, p + 1)

    @testset "J₁: Taylor series reproduces kJ₁(kr) at k+δ" begin
        radial1_taylor!(f, k, r^2, p, k*besselj(1, k*r), k*r*besselj(0, k*r))
        @test _horner(f, δ) ≈ (k + δ)*besselj(1, (k + δ)*r) rtol=1e-12
        @test _horner(f, -δ) ≈ (k - δ)*besselj(1, (k - δ)*r) rtol=1e-12
    end

    @testset "H₁⁽¹⁾: Taylor series reproduces kH₁(kr) at k+δ" begin
        h1 = besselh(1, 1, k*r); h0 = besselh(0, 1, k*r)
        radial1_taylor!(f, k, r^2, p, k*h1, k*r*h0)
        @test _horner(f, δ) ≈ (k + δ)*besselh(1, 1, (k + δ)*r) rtol=1e-12
        @test _horner(f, 0.0) == k*h1
    end

    @testset "degree zero and one only fill the supplied coefficients" begin
        g = fill(NaN + 0im, 4)
        @test radial1_taylor!(g, k, r^2, 0, 1.5, 2.5) === g
        @test g[1] == 1.5 && isnan(real(g[2]))                 # p = 0 writes only f₀
        radial1_taylor!(g, k, r^2, 1, 1.5, 2.5)
        @test g[1] == 1.5 && g[2] == 2.5 && isnan(real(g[3]))  # p = 1 writes f₀, f₁
    end

    @testset "truncation error decreases with the degree" begin
        errs = map((4, 8, 12)) do pp
            ff = zeros(ComplexF64, pp + 1)
            radial1_taylor!(ff, k, r^2, pp, k*besselj(1, k*r), k*r*besselj(0, k*r))
            abs(_horner(ff, 0.3) - (k + 0.3)*besselj(1, (k + 0.3)*r))
        end
        @test errs[1] > errs[2] > errs[3]
    end

    @testset "type stability" begin
        @test @inferred(radial1_taylor!(f, k, r^2, p, 1.0, 2.0)) isa Vector{ComplexF64}
        @test @inferred(radial1_taylor!(f, k, r^2, p, 1.0 + 0im, 2.0 + 0im)) isa Vector{ComplexF64}
    end
end

@testset "radial0_taylor!" begin
    # y(k) = Z₀(k r); y₀ = Z₀(kr), y₁ = -r Z₁(kr)
    k, r, p, δ = 3.0, 0.7, 16, 0.08
    y = zeros(ComplexF64, p + 1)

    @testset "J₀: Taylor series reproduces J₀(kr) at k±δ" begin
        radial0_taylor!(y, k, r^2, p, besselj(0, k*r), -r*besselj(1, k*r))
        @test _horner(y, δ) ≈ besselj(0, (k + δ)*r) rtol=1e-12
        @test _horner(y, -δ) ≈ besselj(0, (k - δ)*r) rtol=1e-12
    end

    @testset "H₀⁽¹⁾: Taylor series reproduces H₀(kr) at k+δ" begin
        radial0_taylor!(y, k, r^2, p, besselh(0, 1, k*r), -r*besselh(1, 1, k*r))
        @test _horner(y, δ) ≈ besselh(0, 1, (k + δ)*r) rtol=1e-12
        @test _horner(y, 0.0) == besselh(0, 1, k*r)
    end

    @testset "complex-valued shift (Beyn contour node)" begin
        radial0_taylor!(y, k, r^2, p, besselh(0, 1, k*r), -r*besselh(1, 1, k*r))
        z = δ*cis(0.9)
        @test _horner(y, z) ≈ besselh(0, 1, (k + z)*r) rtol=1e-12
    end

    @testset "degree zero and one only fill the supplied coefficients" begin
        g = fill(NaN + 0im, 4)
        @test radial0_taylor!(g, k, r^2, 0, 1.5, 2.5) === g
        @test g[1] == 1.5 && isnan(real(g[2]))
        radial0_taylor!(g, k, r^2, 1, 1.5, 2.5)
        @test g[1] == 1.5 && g[2] == 2.5 && isnan(real(g[3]))
    end

    @testset "type stability" begin
        @test @inferred(radial0_taylor!(y, k, r^2, p, 1.0, 2.0)) isa Vector{ComplexF64}
        @test @inferred(radial0_taylor!(y, k, r^2, p, 1.0 + 0im, 2.0 + 0im)) isa Vector{ComplexF64}
    end
end

@testset "radial0_k_taylor!" begin
    # f(k) = k Z₀(k r), obtained from the Taylor series of Z₀(kr): f₀ = k y₀, fₙ = k yₙ + yₙ₋₁
    k, r, p, δ = 3.0, 0.7, 16, 0.08
    f = zeros(ComplexF64, p + 1); y = zeros(ComplexF64, p + 1)

    @testset "J₀: Taylor series reproduces kJ₀(kr) at k±δ" begin
        radial0_k_taylor!(f, y, k, r^2, p, besselj(0, k*r), -r*besselj(1, k*r))
        @test _horner(f, δ) ≈ (k + δ)*besselj(0, (k + δ)*r) rtol=1e-12
        @test _horner(f, -δ) ≈ (k - δ)*besselj(0, (k - δ)*r) rtol=1e-12
    end

    @testset "H₀⁽¹⁾: Taylor series reproduces kH₀(kr) at k+δ" begin
        radial0_k_taylor!(f, y, k, r^2, p, besselh(0, 1, k*r), -r*besselh(1, 1, k*r))
        @test _horner(f, δ) ≈ (k + δ)*besselh(0, 1, (k + δ)*r) rtol=1e-12
    end

    @testset "coefficients follow fₙ = k yₙ + yₙ₋₁ and scratch holds Z₀ series" begin
        radial0_k_taylor!(f, y, k, r^2, p, besselj(0, k*r), -r*besselj(1, k*r))
        y_ref = zeros(ComplexF64, p + 1)
        radial0_taylor!(y_ref, k, r^2, p, besselj(0, k*r), -r*besselj(1, k*r))
        @test y == y_ref
        @test f[1] == k*y[1]
        @test all(f[n+1] ≈ k*y[n+1] + y[n] for n in 1:p)
    end

    @testset "returns the output vector, type stable" begin
        @test radial0_k_taylor!(f, y, k, r^2, p, 1.0, 2.0) === f
        @test @inferred(radial0_k_taylor!(f, y, k, r^2, p, 1.0, 2.0)) isa Vector{ComplexF64}
    end
end

@testset "TaylorWorkspace" begin
    for p in (0, 5)
        ws = TaylorWorkspace(p)
        @test ws isa TaylorWorkspace
        @test all(f -> getfield(ws, f) isa Vector{ComplexF64}, fieldnames(TaylorWorkspace))
        @test all(f -> length(getfield(ws, f)) == p + 1, fieldnames(TaylorWorkspace))
        @test all(f -> all(iszero, getfield(ws, f)), fieldnames(TaylorWorkspace))
    end
    @test length(fieldnames(TaylorWorkspace)) == 9
    ws = TaylorWorkspace(3)
    @test !(ws.a === ws.tmp) && !(ws.H1 === ws.J1) && !(ws.yH0 === ws.yJ0)   # buffers are independent
    @test @inferred(TaylorWorkspace(3)) isa TaylorWorkspace
end
