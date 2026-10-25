# Unit tests for
#   src/basis/fourierbessel/corneradapted.jl   (CornerAdaptedFourierBessel, Jv, Jvp (fractional orders), ca_fb, ca_fb_dk,
#                                               both constructors (without and with reflection symmetries), toFloat32,
#                                               resize_basis, basis_fun, dk_fun, gradient, basis_and_gradient)
#   src/utils/billiardutils.jl                 (adapt_basis, make_triangle_and_basis and make_veech_right_triangle_and_basis,
#                                               only to obtain a basis whose corner and edges are known)
#   src/spectra/spectralutils.jl               (uses_taylor, only as applied to CornerAdaptedFourierBessel)
#   src/solvers/matrixconstructors.jl          (the generic basis_matrix, dk_matrix, gradient_matrices and
#                                               basis_and_gradient_matrices wrappers, only as applied to this basis)
# A basis function is f_i(r, φ) = J_{ν i}(k r) sin(ν i φ) with ν = π / corner_angle in the polar frame (origin, rot_angle) whose
# angle φ is measured from the first edge of the corner and runs over (0, corner_angle) inside the wedge. The orders ν i are
# fractional unless π / corner_angle is an integer, so the evaluators are compared with SpecialFunctions.besselj of fractional
# order at (r, φ) values the test places the points at, with the Dirichlet condition on both edges of the corner, with the
# parity about the bisector of the wedge, with the Helmholtz equation (Δ + k²) f = 0, with central finite differences in x, y
# and k, and with the homogeneity identity k ∂f/∂k = (p - origin)·∇f, which ties dk_fun to gradient. The angular order,
# the branch cut handling of `rotation_angle_discontinuity` (needed for corner angles above π) and the cross-checks with the
# integer-order FourierBessel sectors are covered per corner angle. Integer-order Jvp and Float32 Jvp are covered in
# fourierbessel_tests.jl; the Taylor-cache side lives in corneradapted_recurrence_tests.jl.
# No eigen solves, no pinned wavefunction values. Helper functions use the prefix `ca_` (`ca_fb` and `ca_fb_dk` are source names).

using StaticArrays: SVector
using LinearAlgebra: norm
using SpecialFunctions: besselj
using QuantumBilliards: ca_fb, ca_fb_dk, Jv, Jvp, toFloat32, uses_taylor, filter_matrix!, gradient_matrices,
                        basis_and_gradient_matrices, adapt_basis, make_veech_right_triangle_and_basis

# Corner angles of the edge-case testsets: nu = π / angle is 2, 3, 3/2, 4, 6, π, 1 and 2/3.
const ca_ANGLES = (("π/2", π / 2), ("π/3", π / 3), ("2π/3", 2π / 3), ("π/4", π / 4), ("π/6", π / 6),
                   ("1 rad", 1.0), ("π", Float64(π)), ("3π/2", 3π / 2))

# Wedge sampling: fractions of the opening angle and radii, all strictly inside the wedge.
const ca_FRACS = (0.15, 0.4, 0.6, 0.85)
const ca_RADII = (0.15, 0.4, 0.7, 1.0, 1.3)

# For corner angles above π the branch cut of atan must lie outside the wedge; -angle/2 puts it opposite the bisector.
ca_branch(α) = α > π ? -α / 2 : 0.0

ca_make(dim, α; origin=SVector(0.0, 0.0), rot=0.0, kw...) =
    CornerAdaptedFourierBessel(dim, α, origin, rot; rotation_angle_discontinuity=ca_branch(α), kw...)

# Points of the wedge of `b`, ordered radius-major, fraction-minor.
ca_wedge_pts(b; radii=ca_RADII) = SVector{2,Float64}[b.origin + r * SVector(cos(b.rot_angle + f * b.corner_angle), sin(b.rot_angle + f * b.corner_angle))
                                                     for r in radii for f in ca_FRACS]

# Closed-form J_{ν i}(k r) sin(ν i φ) at the (r, φ) the wedge points were generated from (independent of the basis internals).
function ca_ref_matrix(b, k; radii=ca_RADII)
    ν = π / b.corner_angle
    rows = [(r, f * b.corner_angle) for r in radii for f in ca_FRACS]
    return [besselj(ν * i, k * r) * sin(ν * i * φ) for (r, φ) in rows, i in 1:b.dim]
end

# Points on the edge at angle φe (0 or corner_angle) of the wedge of `b`.
ca_edge_pts(b, φe; radii=(0.1, 0.5, 1.0, 1.7)) = SVector{2,Float64}[b.origin + r * SVector(cos(b.rot_angle + φe), sin(b.rot_angle + φe)) for r in radii]

# Central finite differences of column i in x, y and k.
ca_shift(pts, dx, dy) = [SVector(p[1] + dx, p[2] + dy) for p in pts]
ca_fd_x(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k, ca_shift(pts, h, 0.0)) .- basis_fun(basis, i, k, ca_shift(pts, -h, 0.0))) ./ (2h)
ca_fd_y(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k, ca_shift(pts, 0.0, h)) .- basis_fun(basis, i, k, ca_shift(pts, 0.0, -h))) ./ (2h)
ca_fd_k(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k + h, pts) .- basis_fun(basis, i, k - h, pts)) ./ (2h)

# Residual of the Helmholtz equation of column i at the points, via second differences with step h.
function ca_helmholtz_residual(basis, i, k, pts, h=1e-3)
    f0 = basis_fun(basis, i, k, pts)
    lap = (basis_fun(basis, i, k, ca_shift(pts, h, 0.0)) .+ basis_fun(basis, i, k, ca_shift(pts, -h, 0.0)) .+
           basis_fun(basis, i, k, ca_shift(pts, 0.0, h)) .+ basis_fun(basis, i, k, ca_shift(pts, 0.0, -h)) .- 4 .* f0) ./ h^2
    return lap .+ k^2 .* f0
end

# Field-by-field equality of two bases (symmetries are compared by type).
function ca_same_basis(a, b)
    return a.dim == b.dim && a.corner_angle == b.corner_angle && a.nu == b.nu && a.origin == b.origin &&
           a.rot_angle == b.rot_angle && a.rotation_angle_discontinuity == b.rotation_angle_discontinuity &&
           typeof.(something(a.symmetries, [])) == typeof.(something(b.symmetries, [])) &&
           (a.use_taylor, a.taylor_degree, a.taylor_radius, a.taylor_tol, a.gradients) ==
           (b.use_taylor, b.taylor_degree, b.taylor_radius, b.taylor_tol, b.gradients)
end

@testset "CornerAdaptedFourierBessel type" begin
    @test CornerAdaptedFourierBessel <: AbsBasis
    @test !isabstracttype(CornerAdaptedFourierBessel)
    @test fieldnames(CornerAdaptedFourierBessel) == (:origin, :rot_angle, :dim, :corner_angle, :nu, :symmetries,
                                                     :rotation_angle_discontinuity, :use_taylor, :taylor_degree,
                                                     :taylor_radius, :taylor_tol, :gradients)
    basis = CornerAdaptedFourierBessel(3, π / 3, SVector(0.0, 0.0), 0.0)
    @test basis isa CornerAdaptedFourierBessel{Float64}
    @test isconcretetype(typeof(basis))
end

@testset "Jv - fractional orders" begin
    for ν in (0.5, 2 / 3, 1.5, 3.0, 4.5, Float64(π)), x in (0.3, 1.0, 2.5, 7.0, 12.0)
        @test Jv(ν, x) ≈ besselj(ν, x) atol = 1e-13
    end
    @test Jv(0.5, 0.0) == 0.0 && Jv(1.5, 0.0) == 0.0
    @test Jv(0.0, 0.0) == 1.0
    @test @inferred(Jv(1.5, 2.0)) isa Float64
    @test @inferred(Jv(1.5f0, 2.0f0)) isa Float32
end

@testset "Jvp - fractional orders" begin
    xs = (0.3, 1.0, 2.5, 7.0, 12.0)
    νs = (0.5, 2 / 3, 1.5, 2.5, 3.5, Float64(π))

    @testset "equals the recurrence (J_{ν-1} - J_{ν+1}) / 2" begin
        for ν in νs, x in xs
            @test Jvp(ν, x) ≈ (besselj(ν - 1, x) - besselj(ν + 1, x)) / 2 atol = 1e-13
        end
    end

    @testset "equals the Bessel equation J_ν' = ν J_ν / x - J_{ν+1}" begin
        for ν in νs, x in xs
            @test Jvp(ν, x) ≈ ν * besselj(ν, x) / x - besselj(ν + 1, x) atol = 1e-13
        end
    end

    @testset "matches a central difference of besselj" begin
        h = 1e-6
        for ν in νs, x in xs
            @test Jvp(ν, x) ≈ (besselj(ν, x + h) - besselj(ν, x - h)) / (2h) atol = 1e-8
        end
    end

    @testset "values at the origin: J_ν'(0) = 0 for ν > 1, unbounded for ν < 1" begin
        for ν in (1.5, 2.5, 3.5, Float64(π))
            @test iszero(Jvp(ν, 0.0))
        end
        # J_ν(x) ~ x^ν has an infinite slope at x = 0 for ν < 1; the recurrence evaluates J_{ν-1}(0) = Inf.
        for ν in (0.5, 2 / 3)
            @test isinf(Jvp(ν, 0.0))
        end
    end
end

@testset "ca_fb and ca_fb_dk" begin
    k, r = 3.7, 0.8

    @testset "value J_ν(kr) sin(νφ) for fractional ν" begin
        for ν in (0.5, 2 / 3, 1.5, 3.0, 4.5), φ in (0.2, 0.6, 1.3)
            @test ca_fb(ν, k, r, φ) ≈ besselj(ν, k * r) * sin(ν * φ) atol = 1e-13
        end
    end

    @testset "the wavenumber derivative is r J_ν'(kr) sin(νφ) and matches a central difference" begin
        h = 1e-6
        for ν in (0.5, 2 / 3, 1.5, 3.0, 4.5), φ in (0.2, 0.6, 1.3)
            @test ca_fb_dk(ν, k, r, φ) ≈ r * Jvp(ν, k * r) * sin(ν * φ) atol = 1e-13
            fd = (ca_fb(ν, k + h, r, φ) - ca_fb(ν, k - h, r, φ)) / (2h)
            @test ca_fb_dk(ν, k, r, φ) ≈ fd atol = 1e-8
        end
    end

    @testset "the angular factor vanishes on the edges φ = 0 and φ = π / ν" begin
        for ν in (1.5, 3.0, 4.5), j in 1:3
            @test ca_fb(ν * j, k, r, 0.0) == 0.0
            @test abs(ca_fb(ν * j, k, r, π / ν)) < 1e-14          # the order ν j gives sin(j π) = 0 on the second edge
            @test abs(ca_fb_dk(ν * j, k, r, π / ν)) < 1e-14
        end
    end

    @testset "type stability and Float32" begin
        @test @inferred(ca_fb(1.5, k, r, 0.6)) isa Float64
        @test @inferred(ca_fb(1.5f0, 3.7f0, 0.8f0, 0.6f0)) isa Float32
        @test ca_fb(1.5f0, 3.7f0, 0.8f0, 0.6f0) ≈ Float32(ca_fb(1.5, k, r, 0.6)) atol = 1e-6
        @test ca_fb_dk(1.5f0, 3.7f0, 0.8f0, 0.6f0) ≈ ca_fb_dk(1.5, k, r, 0.6) atol = 1e-6
    end
end

@testset "CornerAdaptedFourierBessel(dim, corner_angle, origin, rot_angle)" begin
    @testset "defaults and stored fields" begin
        b = CornerAdaptedFourierBessel(4, π / 3, SVector(0.3, -0.2), 0.4)
        @test b.dim == 4
        @test b.corner_angle == π / 3 && b.origin == SVector(0.3, -0.2) && b.rot_angle == 0.4
        @test isnothing(b.symmetries)
        @test b.rotation_angle_discontinuity == 0.0
        @test b.use_taylor === true
        @test (b.taylor_degree, b.taylor_radius, b.taylor_tol, b.gradients) == (16, 0.5, 1e-11, false)
    end

    @testset "angular order ν = π / corner_angle for every corner angle" begin
        for (name, α) in ca_ANGLES
            @testset "corner angle $name" begin
                b = ca_make(3, α)
                @test b.nu ≈ π / α
                @test b.corner_angle == α
            end
        end
        @test CornerAdaptedFourierBessel(3, π / 2, SVector(0.0, 0.0), 0.0).nu ≈ 2
        @test CornerAdaptedFourierBessel(3, π / 3, SVector(0.0, 0.0), 0.0).nu ≈ 3
        @test CornerAdaptedFourierBessel(3, 2π / 3, SVector(0.0, 0.0), 0.0).nu ≈ 3 / 2
        @test CornerAdaptedFourierBessel(3, Float64(π), SVector(0.0, 0.0), 0.0).nu ≈ 1
        @test CornerAdaptedFourierBessel(3, 3π / 2, SVector(0.0, 0.0), 0.0).nu ≈ 2 / 3
    end

    @testset "keywords are stored" begin
        b = CornerAdaptedFourierBessel(3, π / 3, SVector(0.0, 0.0), 0.0; rotation_angle_discontinuity=-0.3, use_taylor=false,
                                       taylor_degree=10, taylor_radius=0.25, taylor_tol=1e-9, gradients=true)
        @test b.rotation_angle_discontinuity == -0.3 && !b.use_taylor && b.taylor_degree == 10
        @test b.taylor_radius == 0.25 && b.taylor_tol == 1e-9 && b.gradients
    end

    @testset "element type follows the corner angle, origin and rotation" begin
        @test CornerAdaptedFourierBessel(3, 1.0f0, SVector(0.0f0, 0.0f0), 0.0f0) isa CornerAdaptedFourierBessel{Float32}
        b32 = CornerAdaptedFourierBessel(3, 1.0f0, SVector(0.0f0, 0.0f0), 0.0f0)
        @test b32.origin isa SVector{2,Float32} && b32.nu isa Float32 && b32.taylor_radius isa Float32 && b32.taylor_tol isa Float32
        @test b32.nu ≈ Float32(π) atol = 1e-6
    end
end

@testset "CornerAdaptedFourierBessel(..., symmetry) - symmetry handling" begin
    refl = BilliardGeometry.AbsReflection
    o, θ = SVector(0.1, 0.2), 0.3

    @testset "the symmetries are stored as given" begin
        b = CornerAdaptedFourierBessel(4, π / 3, o, θ, refl[XAxisReflection()])
        @test b.symmetries isa Vector{BilliardGeometry.AbsReflection}
        @test typeof.(b.symmetries) == [XAxisReflection]
        both = CornerAdaptedFourierBessel(4, π / 3, o, θ, refl[YAxisReflection(), XAxisReflection()])
        @test typeof.(both.symmetries) == [YAxisReflection, XAxisReflection]
    end

    @testset "nothing is the same as the constructor without symmetry" begin
        none = CornerAdaptedFourierBessel(4, π / 3, o, θ, nothing)
        @test isnothing(none.symmetries)
        @test ca_same_basis(none, CornerAdaptedFourierBessel(4, π / 3, o, θ))
    end

    @testset "angle, frame and keywords are the same as without symmetry" begin
        b = CornerAdaptedFourierBessel(4, π / 3, o, θ, refl[XAxisReflection()]; rotation_angle_discontinuity=-0.2, use_taylor=false,
                                       taylor_degree=7, taylor_radius=0.2, taylor_tol=1e-8, gradients=true)
        @test b.nu ≈ 3 && b.origin == o && b.rot_angle == θ
        @test (b.rotation_angle_discontinuity, b.use_taylor, b.taylor_degree, b.taylor_radius, b.taylor_tol, b.gradients) ==
              (-0.2, false, 7, 0.2, 1e-8, true)
    end

    @testset "the attached symmetries do not change the basis functions" begin
        # The corner-adapted functions are fixed by the corner; the vector is bookkeeping for the solvers.
        plain = CornerAdaptedFourierBessel(4, π / 3, o, θ)
        sym = CornerAdaptedFourierBessel(4, π / 3, o, θ, refl[XAxisReflection()])
        pts = ca_wedge_pts(plain)
        @test basis_fun(sym, 1:4, 5.0, pts) == basis_fun(plain, 1:4, 5.0, pts)
        @test dk_fun(sym, 1:4, 5.0, pts) == dk_fun(plain, 1:4, 5.0, pts)
        @test gradient(sym, 1:4, 5.0, pts) == gradient(plain, 1:4, 5.0, pts)
    end

    @testset "resize_basis and toFloat32 keep the symmetries" begin
        b = CornerAdaptedFourierBessel(4, π / 3, o, θ, refl[YAxisReflection()])
        @test typeof.(resize_basis(b, StadiumBilliard(0.5), 6, 1.0).symmetries) == [YAxisReflection]
        @test typeof.(toFloat32(b).symmetries) == [YAxisReflection]
    end
end

@testset "CornerAdaptedFourierBessel - constructor argument errors" begin
    refl = BilliardGeometry.AbsReflection
    o = SVector(0.0, 0.0)

    @testset "taylor_degree must be at least 1" begin
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, π / 3, o, 0.0; taylor_degree=0)
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, π / 3, o, 0.0; taylor_degree=-3)
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, π / 3, o, 0.0, refl[XAxisReflection()]; taylor_degree=0)
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, π / 3, o, 0.0, nothing; taylor_degree=-1)
        err = try CornerAdaptedFourierBessel(4, π / 3, o, 0.0; taylor_degree=0); nothing catch e; e end
        @test err isa ArgumentError && occursin("taylor_degree", err.msg)
        @test CornerAdaptedFourierBessel(4, π / 3, o, 0.0; taylor_degree=1) isa CornerAdaptedFourierBessel
    end

    @testset "argument types" begin
        @test_throws MethodError CornerAdaptedFourierBessel(4.0, π / 3, o, 0.0)               # the dimension is an integer
        @test_throws MethodError CornerAdaptedFourierBessel(4, π / 3, SVector(0.0f0, 0.0f0), 0.0)  # one element type for angle, origin, rotation
        @test_throws MethodError CornerAdaptedFourierBessel(4, π / 3, SVector(0, 0), 0.0)
        @test_throws MethodError CornerAdaptedFourierBessel(4, π / 3, (0.0, 0.0), 0.0)         # the origin is an SVector{2}
    end

    @testset "invalid dimension, corner angle and Taylor settings are rejected" begin
        @test_throws ArgumentError CornerAdaptedFourierBessel(0, π / 3, o, 0.0)
        @test_throws ArgumentError CornerAdaptedFourierBessel(-2, π / 3, o, 0.0)
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, 0.0, o, 0.0)
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, -1.0, o, 0.0)
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, π / 3, o, 0.0; taylor_radius=-1.0)
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, π / 3, o, 0.0; taylor_tol=0.0)
        @test_throws ArgumentError CornerAdaptedFourierBessel(4, π / 3, o, 0.0; taylor_degree=0)
    end
end

@testset "toFloat32 and uses_taylor" begin
    o, θ = SVector(0.25, -0.5), 0.75

    @testset "toFloat32 converts the numeric fields and keeps the settings" begin
        b = CornerAdaptedFourierBessel(5, 2π / 3, o, θ; rotation_angle_discontinuity=-0.4, use_taylor=false, taylor_degree=9,
                                       taylor_radius=0.3, taylor_tol=1e-9, gradients=true)
        b32 = toFloat32(b)
        @test b32 isa CornerAdaptedFourierBessel{Float32}
        @test b32.origin isa SVector{2,Float32} && b32.origin == Float32.(o)
        @test b32.corner_angle == Float32(2π / 3) && b32.rot_angle == Float32(θ)
        @test b32.nu ≈ Float32(1.5) atol = 1e-6
        @test b32.rotation_angle_discontinuity == Float32(-0.4)
        @test (b32.dim, b32.use_taylor, b32.taylor_degree, b32.gradients) == (5, false, 9, true)
        @test b32.taylor_radius == Float32(0.3) && b32.taylor_tol == Float32(1e-9)
        @test isnothing(b32.symmetries)
    end

    @testset "uses_taylor follows use_taylor" begin
        @test uses_taylor(CornerAdaptedFourierBessel(3, π / 3, o, θ)) === true
        @test uses_taylor(CornerAdaptedFourierBessel(3, π / 3, o, θ; use_taylor=false)) === false
    end
end

@testset "resize_basis" begin
    billiard = StadiumBilliard(0.5)
    o, θ = SVector(0.25, -0.5), 0.75
    full = CornerAdaptedFourierBessel(4, 2π / 3, o, θ; rotation_angle_discontinuity=-0.4, use_taylor=false, taylor_degree=9,
                                      taylor_radius=0.3, taylor_tol=1e-9, gradients=true)

    @testset "same dimension is a fixed point" begin
        @test resize_basis(full, billiard, 4, 7.3) === full
    end

    @testset "growing keeps the corner, the frame and every setting" begin
        grown = resize_basis(full, billiard, 7, 7.3)
        @test grown isa CornerAdaptedFourierBessel{Float64}
        @test grown.dim == 7
        expected = CornerAdaptedFourierBessel(7, 2π / 3, o, θ; rotation_angle_discontinuity=-0.4, use_taylor=false, taylor_degree=9,
                                              taylor_radius=0.3, taylor_tol=1e-9, gradients=true)
        @test ca_same_basis(grown, expected)
        @test full.dim == 4                                                # the input basis is untouched
    end

    @testset "shrinking keeps the leading functions" begin
        small = resize_basis(full, billiard, 2, 7.3)
        pts = ca_wedge_pts(full)
        @test small.dim == 2
        @test basis_fun(small, 1:2, 4.1, pts) == basis_fun(full, 1:2, 4.1, pts)
    end

    @testset "the wavenumber is ignored" begin
        @test ca_same_basis(resize_basis(full, billiard, 6, 1.0), resize_basis(full, billiard, 6, 300.0))
    end

    @testset "return type" begin
        @test @inferred(resize_basis(full, billiard, 6, 1.0)) isa CornerAdaptedFourierBessel{Float64}
        @test resize_basis(toFloat32(full), billiard, 6, 1.0f0) isa CornerAdaptedFourierBessel{Float32}
    end
end

@testset "corner angle edge cases - wedge geometry and cross-checks" begin
    k = 7.3

    @testset "π: sin modes of the half-plane equal the sin modes of FourierBessel" begin
        # ν = 1, so f_i = J_i(kr) sin(iφ). The unrestricted FourierBessel basis has (order, trig) =
        # (0,c), (1,c), (1,s), (2,c), (2,s), ... so its columns 3 and 5 are the sin modes of order 1 and 2.
        b = ca_make(2, Float64(π))
        fb = FourierBessel(5)
        @test fb.orders[[3, 5]] == [1, 2] && fb.trig[[3, 5]] == Int8[-1, -1]
        pts = ca_wedge_pts(b)
        @test basis_fun(b, 1:2, k, pts) ≈ basis_fun(fb, [3, 5], k, pts) atol = 1e-13
        @test dk_fun(b, 1:2, k, pts) ≈ dk_fun(fb, [3, 5], k, pts) atol = 1e-13
        gx, gy = gradient(b, 1:2, k, pts)
        fx, fy = gradient(fb, [3, 5], k, pts)
        @test gx ≈ fx atol = 1e-11
        @test gy ≈ fy atol = 1e-11
    end

    @testset "π/2: even-order sin modes equal the odd-odd sector of FourierBessel" begin
        # ν = 2, so f_i = J_{2i}(kr) sin(2iφ), which are the (sym_x, sym_y) = (-1, -1) modes of order 2, 4, 6, ...
        b = ca_make(4, π / 2)
        fb = FourierBessel(4; sym_x=-1, sym_y=-1)
        @test fb.orders == [2, 4, 6, 8]
        pts = ca_wedge_pts(b)
        @test basis_fun(b, 1:4, k, pts) ≈ basis_fun(fb, 1:4, k, pts) atol = 1e-13
        @test dk_fun(b, 1:4, k, pts) ≈ dk_fun(fb, 1:4, k, pts) atol = 1e-13
        gx, gy = gradient(b, 1:4, k, pts)
        fx, fy = gradient(fb, 1:4, k, pts)
        @test gx ≈ fx atol = 1e-11
        @test gy ≈ fy atol = 1e-11
    end

    @testset "integer ν: the functions are 2π-periodic, so a branch cut through the wedge changes nothing" begin
        # For π/3 (ν = 3) the offset 3.0 puts the cut at the local angle π - 3.0 ≈ 0.14 inside the wedge (0, π/3).
        plain = ca_make(4, π / 3)
        cut = ca_make(4, π / 3; rotation_angle_discontinuity=3.0)
        pts = ca_wedge_pts(plain)
        @test basis_fun(cut, 1:4, k, pts) ≈ basis_fun(plain, 1:4, k, pts) atol = 1e-13
        @test gradient(cut, 1:4, k, pts)[1] ≈ gradient(plain, 1:4, k, pts)[1] atol = 1e-11
        # Not vacuous: for the fractional ν = 3/2 of 2π/3 the same offset does change the values.
        frac = ca_make(4, 2π / 3)
        fcut = ca_make(4, 2π / 3; rotation_angle_discontinuity=3.0)
        @test maximum(abs, basis_fun(fcut, 1:4, k, ca_wedge_pts(frac)) .- basis_fun(frac, 1:4, k, ca_wedge_pts(frac))) > 1e-3
    end

    @testset "any branch cut offset outside the wedge gives the same values" begin
        # The wedge (0, 2π/3) stays clear of the cut at the local angle π - θ0 for every θ0 in (-π, π/3].
        base = ca_make(4, 2π / 3)
        pts = ca_wedge_pts(base)
        for θ0 in (0.0, -0.3, -0.9, -π / 3 * 0.99)
            shifted = ca_make(4, 2π / 3; rotation_angle_discontinuity=θ0)
            @test basis_fun(shifted, 1:4, k, pts) ≈ basis_fun(base, 1:4, k, pts) atol = 1e-13
            @test gradient(shifted, 1:4, k, pts)[2] ≈ gradient(base, 1:4, k, pts)[2] atol = 1e-11
        end
    end
end

@testset "triangle-adapted bases (make_triangle_and_basis)" begin
    k = 7.0

    for (γ, χ) in ((π / 2, 1.0), (π / 3, 1.0), (1.1, 0.7)), edge_i in 1:3
        @testset "γ = $(round(γ, digits=3)), χ = $χ, edge $edge_i" begin
            tr, b = make_triangle_and_basis(γ, χ; edge_i)
            @test b isa CornerAdaptedFourierBessel{Float64}
            @test b.dim == 10 && isnothing(b.symmetries)
            corners = tr.fundamental_domain.corners
            others = [c for c in corners if norm(c - b.origin) > 1e-9]
            @test length(others) == 2
            # The corner angle is the angle between the two edges leaving the origin, the frame starts on one of them.
            u, v = others[1] - b.origin, others[2] - b.origin
            interior = acos(clamp(sum(u .* v) / (norm(u) * norm(v)), -1, 1))
            @test b.corner_angle ≈ interior atol = 1e-12
            @test b.nu ≈ π / interior atol = 1e-12
            dirs = [atan(w[2], w[1]) for w in (u, v)]
            @test any(d -> isapprox(mod(d - b.rot_angle + π, 2π) - π, 0.0; atol=1e-12), dirs)
            # The Dirichlet condition holds on both edges of the corner.
            for c in others, t in (0.2, 0.5, 0.9)
                @test maximum(abs, basis_fun(b, 1:b.dim, k, [b.origin + t * (c - b.origin)])) < 1e-12
            end
            # Interior points of the triangle are not on the edges: the functions are not zero there.
            centroid = sum(corners) / 3
            @test maximum(abs, basis_fun(b, 1:b.dim, k, [centroid])) > 1e-3
        end
    end

    @testset "adapt_basis returns the frame used by make_triangle_and_basis" begin
        tr, b = make_triangle_and_basis(1.1, 0.7; edge_i=2)
        angle, origin, rot_angle, symmetry = adapt_basis(tr, 4)           # make_triangle_and_basis adapts edge_i + 2
        @test angle == b.corner_angle && origin == b.origin && rot_angle == b.rot_angle
        @test isnothing(symmetry)
    end

    @testset "make_veech_right_triangle_and_basis: right-angle corner, ν = 2" begin
        for n in (4, 5, 7)
            tr, b = make_veech_right_triangle_and_basis(n)
            @test b.corner_angle ≈ π / 2 atol = 1e-12
            @test b.nu ≈ 2 atol = 1e-12
        end
    end
end

@testset "basis_fun - SpecialFunctions reference with fractional order" begin
    k = 7.3

    for (name, α) in ca_ANGLES
        @testset "corner angle $name" begin
            b = ca_make(5, α)
            pts = ca_wedge_pts(b)
            B = basis_fun(b, 1:b.dim, k, pts)
            @test size(B) == (length(pts), 5)
            @test B ≈ ca_ref_matrix(b, k) atol = 1e-13
            # A single entry, evaluated from its closed form.
            ν = π / α
            r, φ = 1.0, 0.4 * α
            p = b.origin + r * SVector(cos(φ), sin(φ))
            @test basis_fun(b, 2, k, [p])[1] ≈ besselj(2ν, k * r) * sin(2ν * φ) atol = 1e-13
        end
    end

    @testset "shifted and rotated frame" begin
        b = ca_make(5, 2π / 3; origin=SVector(0.15, -0.1), rot=0.4)
        pts = ca_wedge_pts(b)
        @test basis_fun(b, 1:5, k, pts) ≈ ca_ref_matrix(b, k) atol = 1e-13
    end

    @testset "ca_fb is the matrix entry" begin
        b = ca_make(4, 2π / 3)
        pts = ca_wedge_pts(b)
        B = basis_fun(b, 1:4, k, pts)
        p = pts[7]
        r, φ = hypot(p[1], p[2]), atan(p[2], p[1])
        @test ca_fb(b.nu * 3, k, r, φ) ≈ B[7, 3] atol = 1e-14
    end

    @testset "the corner itself: every function vanishes" begin
        for (name, α) in ca_ANGLES
            b = ca_make(4, α)
            @test all(iszero, basis_fun(b, 1:4, k, [b.origin]))
        end
    end

    @testset "k = 0 gives the zero function" begin
        b = ca_make(4, π / 3)
        @test all(iszero, basis_fun(b, 1:4, 0.0, ca_wedge_pts(b)))
    end

    @testset "|J_ν(x)| decays: bounded values" begin
        b = ca_make(6, 2π / 3)
        @test all(abs.(basis_fun(b, 1:6, 20.0, ca_wedge_pts(b))) .<= 1 + 1e-14)
    end
end

@testset "basis_fun - method consistency" begin
    k = 2.7
    b = ca_make(6, 2π / 3; origin=SVector(0.1, 0.2), rot=0.3)
    pts = ca_wedge_pts(b)
    B = basis_fun(b, 1:b.dim, k, pts)

    @testset "single-index method equals the matrix column" begin
        for i in 1:b.dim
            @test basis_fun(b, i, k, pts) ≈ B[:, i] atol = 1e-14
        end
    end

    @testset "index subsets select and reorder columns" begin
        @test basis_fun(b, [5, 2], k, pts) == B[:, [5, 2]]
        @test basis_fun(b, 3:4, k, pts) == B[:, 3:4]
        @test size(basis_fun(b, Int[], k, pts)) == (length(pts), 0)
    end

    @testset "the angular order is nu times the index, not the position" begin
        @test basis_fun(b, [4], k, pts) == B[:, 4:4]
        @test basis_fun(b, [4], k, pts) != B[:, 1:1]
    end

    @testset "empty point set" begin
        @test basis_fun(b, 1, k, SVector{2,Float64}[]) == Float64[]
        @test size(basis_fun(b, 1:b.dim, k, SVector{2,Float64}[])) == (0, b.dim)
    end

    @testset "points may be any 2-element indexable" begin
        tuples = [(p[1], p[2]) for p in pts]
        @test basis_fun(b, 1:b.dim, k, tuples) == B
    end

    @testset "multithreaded keyword does not change the values" begin
        @test basis_fun(b, 1:b.dim, k, pts; multithreaded=false) == B
        @test basis_fun(b, 1:b.dim, k, pts; multithreaded=true) == B
    end

    @testset "the wavenumber must have the basis element type" begin
        @test_throws MethodError basis_fun(b, 1, 2.7f0, pts)
        @test_throws MethodError basis_fun(b, 1:3, 2, pts)
    end
end

@testset "basis_fun - origin and rotation of the polar frame" begin
    k = 4.1
    o, θ = SVector(0.15, -0.1), 0.4
    base = ca_make(5, 2π / 3)
    pts = ca_wedge_pts(base)
    B0 = basis_fun(base, 1:5, k, pts)

    @testset "shifted origin: evaluating about o equals evaluating about 0 at pts - o" begin
        bo = ca_make(5, 2π / 3; origin=o)
        @test basis_fun(bo, 1:5, k, [p + o for p in pts]) ≈ B0 atol = 1e-13
    end

    @testset "rotated frame: the wedge and its points rotate together" begin
        s, c = sincos(θ)
        rotated = [SVector(c * p[1] - s * p[2], s * p[1] + c * p[2]) for p in pts]
        br = ca_make(5, 2π / 3; rot=θ)
        @test basis_fun(br, 1:5, k, rotated) ≈ B0 atol = 1e-13
        @test gradient(br, 1:5, k, rotated)[1] ≈ (c .* gradient(base, 1:5, k, pts)[1] .- s .* gradient(base, 1:5, k, pts)[2]) atol = 1e-11
    end

    @testset "shifted and rotated frame" begin
        s, c = sincos(θ)
        moved = [o + SVector(c * p[1] - s * p[2], s * p[1] + c * p[2]) for p in pts]
        b = ca_make(5, 2π / 3; origin=o, rot=θ)
        @test basis_fun(b, 1:5, k, moved) ≈ B0 atol = 1e-13
        @test basis_fun(b, 1:5, k, moved) ≈ ca_ref_matrix(b, k) atol = 1e-13
    end

    @testset "a rotation by 2π is the identity" begin
        @test basis_fun(ca_make(5, 2π / 3; rot=2π), 1:5, k, pts) ≈ B0 atol = 1e-12
    end
end

@testset "basis_fun - Dirichlet condition on the two corner edges" begin
    k = 7.3

    for (name, α) in ca_ANGLES
        @testset "corner angle $name" begin
            b = ca_make(5, α; origin=SVector(0.2, -0.1), rot=0.5)
            for φe in (0.0, α)
                B = basis_fun(b, 1:b.dim, k, ca_edge_pts(b, φe))
                @test maximum(abs, B) < 1e-13
                dB = dk_fun(b, 1:b.dim, k, ca_edge_pts(b, φe))
                @test maximum(abs, dB) < 1e-13
            end
            # Not vacuous: inside the wedge, close to either edge, the functions are non-zero.
            near_first = basis_fun(b, 1:b.dim, k, ca_edge_pts(b, 0.05 * α; radii=(0.5, 1.0)))
            near_second = basis_fun(b, 1:b.dim, k, ca_edge_pts(b, 0.95 * α; radii=(0.5, 1.0)))
            @test maximum(abs, near_first) > 1e-3
            @test maximum(abs, near_second) > 1e-3
        end
    end

    @testset "on the bisector φ = α/2 the angular factor is sin(i π / 2): 1, 0, -1, 0, 1" begin
        b = ca_make(5, 2π / 3)
        B = basis_fun(b, 1:5, 7.3, ca_edge_pts(b, b.corner_angle / 2; radii=(0.8,)))
        for i in 1:5
            @test B[1, i] ≈ besselj(b.nu * i, 7.3 * 0.8) * sin(i * π / 2) atol = 1e-13
        end
        @test abs(B[1, 2]) < 1e-13 && abs(B[1, 4]) < 1e-13
    end

    @testset "corner angle above π needs the branch cut outside the wedge" begin
        # 3π/2: with the default offset 0 the atan branch cut cuts through the wedge, so the second edge (angle 3π/2 maps to
        # -π/2) is not an edge of the functions; moving the cut opposite the bisector restores the Dirichlet condition.
        α = 3π / 2
        naive = CornerAdaptedFourierBessel(4, α, SVector(0.0, 0.0), 0.0)
        fixed = ca_make(4, α)
        @test fixed.rotation_angle_discontinuity == -3π / 4
        second_edge = ca_edge_pts(naive, α)
        @test maximum(abs, basis_fun(naive, 1:4, k, second_edge)) > 1e-3
        @test maximum(abs, basis_fun(fixed, 1:4, k, second_edge)) < 1e-13
        # The first edge is fine in both.
        first_edge = ca_edge_pts(naive, 0.0)
        @test maximum(abs, basis_fun(naive, 1:4, k, first_edge)) < 1e-13
        # A branch cut inside the wedge makes the function jump across it.
        @test maximum(abs, basis_fun(naive, 1:4, k, ca_wedge_pts(fixed)) .- basis_fun(fixed, 1:4, k, ca_wedge_pts(fixed))) > 1e-3
    end
end

@testset "basis_fun - parity about the bisector of the wedge" begin
    k = 5.3

    for (name, α) in ca_ANGLES
        @testset "corner angle $name" begin
            b = ca_make(5, α; origin=SVector(0.2, -0.1), rot=0.5)
            u = SVector(cos(b.rot_angle + α / 2), sin(b.rot_angle + α / 2))
            pts = ca_wedge_pts(b)
            mirrored = [b.origin + 2 * sum((p - b.origin) .* u) * u - (p - b.origin) for p in pts]
            B = basis_fun(b, 1:5, k, pts)
            Bm = basis_fun(b, 1:5, k, mirrored)
            # sin(ν i (α - φ)) = sin(i π - ν i φ) = (-1)^(i + 1) sin(ν i φ).
            for i in 1:5
                @test Bm[:, i] ≈ (-1)^(i + 1) .* B[:, i] atol = 1e-12
            end
        end
    end
end

@testset "basis_fun - Helmholtz equation" begin
    k = 3.1
    radii = (0.4, 0.7, 1.0, 1.3)

    @testset "wedges below π" begin
        for α in (π / 2, π / 3, 2π / 3)
            b = ca_make(4, α)
            pts = ca_wedge_pts(b; radii)
            for i in 1:b.dim
                @test maximum(abs, ca_helmholtz_residual(b, i, k, pts)) < 1e-4
            end
        end
    end

    @testset "re-entrant wedge, shifted and rotated frame" begin
        b = ca_make(4, 3π / 2; origin=SVector(0.15, -0.1), rot=0.4)
        pts = ca_wedge_pts(b; radii)
        for i in 1:b.dim
            @test maximum(abs, ca_helmholtz_residual(b, i, k, pts)) < 1e-4
        end
    end

    @testset "the residual check is not vacuous: a wrong wavenumber is detected" begin
        b = ca_make(3, 2π / 3)
        pts = ca_wedge_pts(b; radii)
        residual = ca_helmholtz_residual(b, 2, k, pts)
        @test maximum(abs, residual) < 1e-4
        wrong = residual .+ ((1.3 * k)^2 - k^2) .* basis_fun(b, 2, k, pts)
        @test maximum(abs, wrong) > 1e-2
    end
end

@testset "dk_fun vs finite differences" begin
    k = 7.3

    for (name, α) in ca_ANGLES
        @testset "corner angle $name, shifted and rotated frame" begin
            b = ca_make(5, α; origin=SVector(0.15, -0.1), rot=0.4)
            pts = ca_wedge_pts(b)
            dB = dk_fun(b, 1:b.dim, k, pts)
            fd = hcat((ca_fd_k(b, i, k, pts) for i in 1:b.dim)...)
            @test size(dB) == (length(pts), b.dim)
            @test norm(dB - fd) < 1e-7
        end
    end

    b = ca_make(6, 2π / 3)
    pts = ca_wedge_pts(b)

    @testset "closed form r J_ν i'(kr) sin(ν i φ)" begin
        ν = π / b.corner_angle
        rows = [(r, f * b.corner_angle) for r in ca_RADII for f in ca_FRACS]
        ref = [r * (besselj(ν * i - 1, k * r) - besselj(ν * i + 1, k * r)) / 2 * sin(ν * i * φ) for (r, φ) in rows, i in 1:b.dim]
        @test dk_fun(b, 1:b.dim, k, pts) ≈ ref atol = 1e-12
    end

    @testset "single-index and index-set methods" begin
        dB = dk_fun(b, 1:b.dim, k, pts)
        for i in 1:b.dim
            @test dk_fun(b, i, k, pts) ≈ dB[:, i] atol = 1e-14
        end
        @test dk_fun(b, [5, 2], k, pts) == dB[:, [5, 2]]
        @test dk_fun(b, 1:b.dim, k, pts; multithreaded=false) == dB
    end

    @testset "the corner: the derivative vanishes for a wedge below π" begin
        @test all(iszero, dk_fun(ca_make(4, π / 3), 1:4, k, [SVector(0.0, 0.0)]))
        @test all(iszero, dk_fun(ca_make(4, 2π / 3), 1:4, k, [SVector(0.0, 0.0)]))
    end

    @testset "the corner of a re-entrant wedge" begin
        # For corner angles above π the first order ν < 1 gives Jvp(ν, 0) = Inf; r Jvp is guarded to 0 at r == 0
        # (the limit r ∂_x J_ν(x) → 0 since J_ν ~ x^ν).
        d = dk_fun(ca_make(4, 3π / 2), 1:4, k, [SVector(0.0, 0.0)])
        @test iszero(d[1, 1])
        @test all(iszero, d[:, 2:4])
    end

    @testset "empty points" begin
        @test dk_fun(b, 1, k, SVector{2,Float64}[]) == Float64[]
        @test size(dk_fun(b, 1:3, k, SVector{2,Float64}[])) == (0, 3)
    end
end

@testset "gradient and basis_and_gradient vs finite differences" begin
    k = 7.3

    for (name, α) in ca_ANGLES
        @testset "corner angle $name, shifted and rotated frame" begin
            b = ca_make(5, α; origin=SVector(0.15, -0.1), rot=0.4)
            pts = ca_wedge_pts(b)
            dX, dY = gradient(b, 1:b.dim, k, pts)
            fdx = hcat((ca_fd_x(b, i, k, pts) for i in 1:b.dim)...)
            fdy = hcat((ca_fd_y(b, i, k, pts) for i in 1:b.dim)...)
            @test size(dX) == size(dY) == (length(pts), b.dim)
            @test norm(dX - fdx) < 1e-7
            @test norm(dY - fdy) < 1e-7
        end
    end

    b = ca_make(6, 2π / 3; origin=SVector(0.15, -0.1), rot=0.4)
    pts = ca_wedge_pts(b)

    @testset "single-index forms and basis_and_gradient" begin
        dX, dY = gradient(b, 1:b.dim, k, pts)
        gx, gy = gradient(b, 4, k, pts)
        @test gx ≈ dX[:, 4] atol = 1e-12
        @test gy ≈ dY[:, 4] atol = 1e-12
        bf, bgx, bgy = basis_and_gradient(b, 4, k, pts)
        @test bf ≈ basis_fun(b, 4, k, pts) atol = 1e-14
        @test bgx ≈ gx atol = 1e-12
        @test bgy ≈ gy atol = 1e-12
        Bv, vx, vy = basis_and_gradient(b, 1:b.dim, k, pts)
        @test Bv ≈ basis_fun(b, 1:b.dim, k, pts) atol = 1e-14
        @test vx ≈ dX atol = 1e-12
        @test vy ≈ dY atol = 1e-12
        Bs, sx, sy = basis_and_gradient(b, [5, 2], k, pts)
        @test Bs == Bv[:, [5, 2]] && sx == vx[:, [5, 2]] && sy == vy[:, [5, 2]]
    end

    @testset "the first function of the half-plane: ∇ near the corner is (0, k/2)" begin
        # J_1(kr) sin φ ≈ k y / 2 near the corner for the corner angle π with ν = 1.
        h = ca_make(2, Float64(π))
        gx, gy = gradient(h, 1, k, [SVector(1e-8, 1e-8)])
        @test gx[1] ≈ 0 atol = 1e-6
        @test gy[1] ≈ k / 2 atol = 1e-6
    end

    @testset "the corner: zero gradient for a wedge whose functions all vanish like r^m with m > 1" begin
        gx, gy = gradient(ca_make(4, π / 3), 1:4, k, [SVector(0.0, 0.0)])
        @test all(iszero, gx) && all(iszero, gy)
        bf, bx, by = basis_and_gradient(ca_make(4, π / 3), 1:4, k, [SVector(0.0, 0.0)])
        @test all(iszero, bf) && all(iszero, bx) && all(iszero, by)
    end

    @testset "empty points" begin
        gx, gy = gradient(b, 1, k, SVector{2,Float64}[])
        @test gx == Float64[] && gy == Float64[]
        @test all(size.(gradient(b, 1:3, k, SVector{2,Float64}[])) .== Ref((0, 3)))
        @test all(size.(basis_and_gradient(b, 1:3, k, SVector{2,Float64}[])) .== Ref((0, 3)))
    end
end

@testset "dk_fun and gradient - homogeneity identity k ∂f/∂k = (p - origin)·∇f" begin
    k = 5.3

    for (name, α) in (("π/2", π / 2), ("2π/3", 2π / 3), ("3π/2", 3π / 2))
        @testset "corner angle $name, shifted and rotated frame" begin
            b = ca_make(6, α; origin=SVector(0.15, -0.1), rot=0.4)
            pts = ca_wedge_pts(b)
            dB = dk_fun(b, 1:b.dim, k, pts)
            dX, dY = gradient(b, 1:b.dim, k, pts)
            relx = [p[1] - b.origin[1] for p in pts]
            rely = [p[2] - b.origin[2] for p in pts]
            @test k .* dB ≈ relx .* dX .+ rely .* dY atol = 1e-11
        end
    end
end

@testset "basis_matrix, dk_matrix and gradient matrices" begin
    k = 2.7
    b = ca_make(6, 2π / 3; origin=SVector(0.1, 0.2), rot=0.3)
    pts = ca_wedge_pts(b)

    @testset "basis_matrix equals the filtered columns of basis_fun" begin
        raw = basis_fun(b, 1:b.dim, k, pts)
        B = basis_matrix(b, k, pts)
        @test size(B) == (length(pts), b.dim)
        @test B == filter_matrix!(copy(raw))
        @test B ≈ ca_ref_matrix(b, k) atol = 1e-13
        @test basis_matrix(b, k, pts; multithreaded=false) == B
    end

    @testset "dk_matrix equals the filtered dk_fun" begin
        @test dk_matrix(b, k, pts) == filter_matrix!(dk_fun(b, 1:b.dim, k, pts))
    end

    @testset "gradient_matrices and basis_and_gradient_matrices equal the filtered evaluators" begin
        dX, dY = gradient(b, 1:b.dim, k, pts)
        mx, my = gradient_matrices(b, k, pts)
        @test mx == filter_matrix!(copy(dX)) && my == filter_matrix!(copy(dY))
        Bm, bx, by = basis_and_gradient_matrices(b, k, pts)
        @test Bm == basis_matrix(b, k, pts) && bx == mx && by == my
    end

    @testset "entries at the corner are exactly zero" begin
        @test all(iszero, basis_matrix(b, k, [b.origin]))
        @test all(iszero, dk_matrix(b, k, [b.origin]))
    end
end

@testset "CornerAdaptedFourierBessel - threaded equals serial" begin
    k = 4.4
    b = ca_make(8, 2π / 3; origin=SVector(0.1, 0.2), rot=0.3)
    pts = ca_wedge_pts(b)
    idx = 1:b.dim

    @test basis_fun(b, idx, k, pts; multithreaded=true) == basis_fun(b, idx, k, pts; multithreaded=false)
    @test dk_fun(b, idx, k, pts; multithreaded=true) == dk_fun(b, idx, k, pts; multithreaded=false)
    @test gradient(b, idx, k, pts; multithreaded=true) == gradient(b, idx, k, pts; multithreaded=false)
    @test basis_and_gradient(b, idx, k, pts; multithreaded=true) == basis_and_gradient(b, idx, k, pts; multithreaded=false)
    @test basis_matrix(b, k, pts; multithreaded=true) == basis_matrix(b, k, pts; multithreaded=false)
    @test dk_matrix(b, k, pts; multithreaded=true) == dk_matrix(b, k, pts; multithreaded=false)
    @test gradient_matrices(b, k, pts; multithreaded=true) == gradient_matrices(b, k, pts; multithreaded=false)
end

@testset "CornerAdaptedFourierBessel - type stability" begin
    b = ca_make(6, 2π / 3)
    k = 4.4
    pts = ca_wedge_pts(b)
    idx = 1:b.dim

    @test @inferred(basis_fun(b, 3, k, pts)) isa Vector{Float64}
    @test @inferred(basis_fun(b, idx, k, pts)) isa Matrix{Float64}
    @test @inferred(dk_fun(b, 3, k, pts)) isa Vector{Float64}
    @test @inferred(dk_fun(b, idx, k, pts)) isa Matrix{Float64}
    @test @inferred(gradient(b, 3, k, pts)) isa Tuple{Vector{Float64},Vector{Float64}}
    @test @inferred(gradient(b, idx, k, pts)) isa Tuple{Matrix{Float64},Matrix{Float64}}
    @test @inferred(basis_and_gradient(b, 3, k, pts)) isa Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    @test @inferred(basis_and_gradient(b, idx, k, pts)) isa Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    @test @inferred(basis_matrix(b, k, pts)) isa Matrix{Float64}
    @test @inferred(dk_matrix(b, k, pts)) isa Matrix{Float64}
    @test @inferred(CornerAdaptedFourierBessel(4, π / 3, SVector(0.0, 0.0), 0.0)) isa CornerAdaptedFourierBessel{Float64}
    @test @inferred(CornerAdaptedFourierBessel(4, π / 3, SVector(0.0, 0.0), 0.0, nothing)) isa CornerAdaptedFourierBessel{Float64}
end

@testset "CornerAdaptedFourierBessel - Float32" begin
    b64 = ca_make(5, 2π / 3; origin=SVector(0.15, -0.1), rot=0.4)
    b32 = toFloat32(b64)
    pts64 = ca_wedge_pts(b64)
    pts32 = [SVector{2,Float32}(p) for p in pts64]
    k32, k64 = 3.3f0, 3.3
    idx = 1:5

    @testset "construction keeps the element type" begin
        @test b32 isa CornerAdaptedFourierBessel{Float32}
        @test b32.nu ≈ Float32(1.5) atol = 1e-6
        @test resize_basis(b32, StadiumBilliard(0.5), 7, k32) isa CornerAdaptedFourierBessel{Float32}
    end

    @testset "evaluators stay in Float32 and agree with Float64" begin
        @test @inferred(basis_fun(b32, idx, k32, pts32)) isa Matrix{Float32}
        @test @inferred(basis_fun(b32, 2, k32, pts32)) isa Vector{Float32}
        @test @inferred(dk_fun(b32, idx, k32, pts32)) isa Matrix{Float32}
        @test @inferred(dk_fun(b32, 2, k32, pts32)) isa Vector{Float32}
        @test @inferred(gradient(b32, idx, k32, pts32)) isa Tuple{Matrix{Float32},Matrix{Float32}}
        @test @inferred(gradient(b32, 2, k32, pts32)) isa Tuple{Vector{Float32},Vector{Float32}}
        @test @inferred(basis_and_gradient(b32, idx, k32, pts32)) isa Tuple{Matrix{Float32},Matrix{Float32},Matrix{Float32}}
        @test @inferred(basis_matrix(b32, k32, pts32)) isa Matrix{Float32}
        @test @inferred(dk_matrix(b32, k32, pts32)) isa Matrix{Float32}
        @test basis_fun(b32, idx, k32, pts32) ≈ basis_fun(b64, idx, k64, pts64) atol = 2e-5
        @test dk_fun(b32, idx, k32, pts32) ≈ dk_fun(b64, idx, k64, pts64) atol = 2e-5
        dX32, dY32 = gradient(b32, idx, k32, pts32)
        dX64, dY64 = gradient(b64, idx, k64, pts64)
        @test dX32 ≈ dX64 atol = 2e-4
        @test dY32 ≈ dY64 atol = 2e-4
    end

    @testset "the Dirichlet condition holds to single precision" begin
        edge = [SVector{2,Float32}(p) for p in ca_edge_pts(b64, b64.corner_angle)]
        @test maximum(abs, basis_fun(b32, idx, k32, edge)) < 1e-4
    end
end
