using StaticArrays, LinearAlgebra, Random, SpecialFunctions
using BilliardGeometry

# Unit and integration tests for the `FourierBessel` basis
# (src/basis/fourierbessel/fourierbessel.jl, fourierbessel_recurrence.jl) and the
# 3-D `_horner` method in src/solvers/taylor/recurrence_core.jl.
# Non-exported internals are accessed as `QuantumBilliards.<name>`.

# Small fixed point cloud away from the origin (r > 0 for every point).
const _FBT_PTS = [SVector(0.3 * i - 0.7, 0.2 * i + 0.1) for i in 1:7]
const _FBT_K = 7.3

# Reference value J_j(kr) T(jφ) from SpecialFunctions.
function _fbt_reference(j, trig, k, pt)
    r = hypot(pt[1], pt[2])
    φ = atan(pt[2], pt[1])
    return besselj(j, k * r) * (trig > 0 ? cos(j * φ) : sin(j * φ))
end

@testset "FourierBessel - fb_modes sector mode lists" begin
    modes(sx, sy, dim) = collect(zip(QuantumBilliards.fb_modes(sx, sy, dim)...))
    c, s = Int8(1), Int8(-1)
    @test modes(1, 1, 4) == [(0, c), (2, c), (4, c), (6, c)]
    @test modes(1, -1, 4) == [(1, s), (3, s), (5, s), (7, s)]
    @test modes(-1, 1, 4) == [(1, c), (3, c), (5, c), (7, c)]
    @test modes(-1, -1, 4) == [(2, s), (4, s), (6, s), (8, s)]
    @test modes(1, nothing, 4) == [(0, c), (1, s), (2, c), (3, s)]
    @test modes(-1, nothing, 4) == [(1, c), (2, s), (3, c), (4, s)]
    @test modes(nothing, 1, 4) == [(0, c), (1, c), (2, c), (3, c)]
    @test modes(nothing, -1, 4) == [(1, s), (2, s), (3, s), (4, s)]
    @test modes(nothing, nothing, 5) == [(0, c), (1, c), (1, s), (2, c), (2, s)]

    orders, trig = QuantumBilliards.fb_modes(1, 1, 3)
    @test orders isa Vector{Int}
    @test trig isa Vector{Int8}
    # Prefix property: the first dim modes of a longer list are the shorter list.
    for (sx, sy) in ((1, 1), (nothing, nothing), (-1, nothing))
        o6, t6 = QuantumBilliards.fb_modes(sx, sy, 6)
        o3, t3 = QuantumBilliards.fb_modes(sx, sy, 3)
        @test o6[1:3] == o3 && t6[1:3] == t3
    end
end

@testset "FourierBessel - constructors" begin
    b = FourierBessel(5; sym_x=1, sym_y=-1)
    @test b isa FourierBessel{Float64}
    @test b.dim == 5
    @test (b.orders, b.trig) == QuantumBilliards.fb_modes(1, -1, 5)
    @test b.symmetries isa Vector
    @test map(typeof, b.symmetries) == [YAxisReflection, XYAxisReflection, XAxisReflection]
    @test b.sym_qnumbers == [1.0, -1.0, -1.0]   # [sym_x, sym_x*sym_y, sym_y]
    @test b.origin == SVector(0.0, 0.0)

    b0 = FourierBessel(4)
    @test isnothing(b0.symmetries) && isnothing(b0.sym_qnumbers)
    @test (b0.orders, b0.trig) == QuantumBilliards.fb_modes(nothing, nothing, 4)

    bx = FourierBessel(4; sym_x=-1)
    @test map(typeof, bx.symmetries) == [YAxisReflection]
    @test bx.sym_qnumbers == [-1.0]
    by = FourierBessel(4; sym_y=1)
    @test map(typeof, by.symmetries) == [XAxisReflection]
    @test by.sym_qnumbers == [1.0]

    # Raw form round-trips the stored symmetries.
    braw = FourierBessel(7, b.symmetries, b.sym_qnumbers)
    @test (braw.orders, braw.trig) == QuantumBilliards.fb_modes(1, -1, 7)
    @test braw.sym_qnumbers == b.sym_qnumbers
    @test FourierBessel(4, nothing, nothing).orders == b0.orders
    @test FourierBessel(4, bx.symmetries, bx.sym_qnumbers).orders == bx.orders
    @test FourierBessel(4, by.symmetries, by.sym_qnumbers).orders == by.orders

    # Non-default origin without symmetry is allowed and stored.
    bo = FourierBessel(3; origin=SVector(0.3, -0.2), rot_angle=0.4)
    @test bo.origin == SVector(0.3, -0.2)
    @test bo.rot_angle == 0.4

    # Keyword storage and `uses_taylor` trait.
    bk = FourierBessel(3; use_taylor=false, taylor_degree=10, taylor_radius=0.25, taylor_tol=1e-9, gradients=true)
    @test !bk.use_taylor && bk.taylor_degree == 10 && bk.taylor_radius == 0.25
    @test bk.taylor_tol == 1e-9 && bk.gradients
    @test QuantumBilliards.uses_taylor(bk) == false
    @test QuantumBilliards.uses_taylor(FourierBessel(3)) == true
    @test QuantumBilliards.uses_taylor(FourierBessel(3; use_taylor=false)) == false
end

@testset "FourierBessel - constructor argument errors" begin
    @test_throws ArgumentError FourierBessel(4; sym_x=2)
    @test_throws ArgumentError FourierBessel(4; sym_y=0)
    @test_throws ArgumentError FourierBessel(0)
    @test_throws ArgumentError FourierBessel(4; taylor_degree=0)
    @test_throws ArgumentError FourierBessel(4; taylor_radius=-1.0)
    @test_throws ArgumentError FourierBessel(4; taylor_tol=0.0)
    # Axis reflections act about the global origin.
    @test_throws ArgumentError FourierBessel(4; sym_x=1, origin=SVector(0.1, 0.0))
    @test_throws ArgumentError FourierBessel(4; sym_y=-1, rot_angle=0.3)

    # Raw form: XYAxisReflection alone (no X/Y axis reflections) is rejected.
    @test_throws ArgumentError FourierBessel(4, BilliardGeometry.AbsReflection[XYAxisReflection()], [1.0])
    # XYAxisReflection with only one axis reflection is rejected as well.
    @test_throws ArgumentError FourierBessel(4, BilliardGeometry.AbsReflection[XAxisReflection(), XYAxisReflection()], [1.0, 1.0])
    # symmetries without sym_qnumbers / mismatched lengths.
    @test_throws ArgumentError FourierBessel(4, BilliardGeometry.AbsReflection[XAxisReflection()], nothing)
    @test_throws ArgumentError FourierBessel(4, BilliardGeometry.AbsReflection[XAxisReflection()], [1.0, 1.0])
end

@testset "FourierBessel - SymmetrySector constructor" begin
    stadium = StadiumBilliard(0.5)
    for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1))
        # `YAxisReflection` is x -> -x (sym_x); `XAxisReflection` is y -> -y (sym_y).
        sector = symmetry_sector(stadium, XAxisReflection => sy, YAxisReflection => sx)
        bs = FourierBessel(6, stadium, sector)
        bk = FourierBessel(6; sym_x=sx, sym_y=sy)
        @test bs.orders == bk.orders && bs.trig == bk.trig
        @test bs.sym_qnumbers == bk.sym_qnumbers
    end
    # Reflection-only sector (one axis constrained).
    sector_y = symmetry_sector(stadium, XAxisReflection => -1)
    @test FourierBessel(6, stadium, sector_y).orders == FourierBessel(6; sym_y=-1).orders
    # Keyword arguments are forwarded.
    sector = symmetry_sector(stadium, XAxisReflection => 1, YAxisReflection => 1)
    @test !FourierBessel(4, stadium, sector; use_taylor=false).use_taylor

    # A sector built for a different billiard instance is rejected.
    other = StadiumBilliard(0.5)
    @test_throws ArgumentError FourierBessel(4, other, sector)

    # NFoldRotation sectors have no FourierBessel representation.
    c3 = C3Billiard(0.2)
    @test_throws ArgumentError FourierBessel(6, c3, symmetry_sector(c3, NFoldRotation => 0))
    @test_throws ArgumentError FourierBessel(6, c3, symmetry_sector(c3, NFoldRotation => 1))
end

@testset "FourierBessel - resize_basis" begin
    billiard = StadiumBilliard(0.5)
    b = FourierBessel(4; sym_x=1, sym_y=-1, use_taylor=false, taylor_degree=10, taylor_radius=0.25, taylor_tol=1e-9, gradients=true)
    @test resize_basis(b, billiard, 4, _FBT_K) === b

    b2 = resize_basis(b, billiard, 6, _FBT_K)
    @test b2 isa FourierBessel{Float64}
    @test b2.dim == 6
    @test (b2.orders, b2.trig) == QuantumBilliards.fb_modes(1, -1, 6)
    @test map(typeof, b2.symmetries) == map(typeof, b.symmetries)
    @test b2.sym_qnumbers == b.sym_qnumbers
    @test b2.origin == b.origin && b2.rot_angle == b.rot_angle
    @test !b2.use_taylor && b2.taylor_degree == 10 && b2.taylor_radius == 0.25
    @test b2.taylor_tol == 1e-9 && b2.gradients

    # Shrinking keeps the leading modes.
    b3 = resize_basis(b2, billiard, 3, _FBT_K)
    @test b3.orders == b2.orders[1:3] && b3.trig == b2.trig[1:3]

    # No-symmetry basis with non-trivial origin keeps its origin and rotation.
    bo = FourierBessel(3; origin=SVector(0.3, -0.2), rot_angle=0.4)
    bo2 = resize_basis(bo, billiard, 5, _FBT_K)
    @test isnothing(bo2.symmetries) && isnothing(bo2.sym_qnumbers)
    @test bo2.origin == SVector(0.3, -0.2) && bo2.rot_angle == 0.4
    @test bo2.orders == QuantumBilliards.fb_modes(nothing, nothing, 5)[1]
end

@testset "FourierBessel - values vs SpecialFunctions" begin
    b = FourierBessel(7)
    B = basis_fun(b, 1:b.dim, _FBT_K, _FBT_PTS)
    @test size(B) == (length(_FBT_PTS), b.dim)
    for c in 1:b.dim, (p, pt) in enumerate(_FBT_PTS)
        @test isapprox(B[p, c], _fbt_reference(b.orders[c], b.trig[c], _FBT_K, pt); atol=1e-12)
    end
    # fb_term agrees with the matrix entries.
    r = hypot(_FBT_PTS[2]...)
    φ = atan(_FBT_PTS[2][2], _FBT_PTS[2][1])
    @test QuantumBilliards.fb_term(b.orders[4], b.trig[4], _FBT_K, r, φ) ≈ B[2, 4]

    # Single-index and index-vector forms agree.
    for i in 1:b.dim
        @test basis_fun(b, i, _FBT_K, _FBT_PTS) ≈ B[:, i]
    end
    @test basis_fun(b, [5, 2], _FBT_K, _FBT_PTS) ≈ B[:, [5, 2]]
    @test basis_fun(b, 1:b.dim, _FBT_K, _FBT_PTS; multithreaded=false) ≈ B

    # Shifted origin: evaluating about `o` equals evaluating about 0 at pts - o.
    o = SVector(0.15, -0.1)
    bo = FourierBessel(7; origin=o)
    @test basis_fun(bo, 1:7, _FBT_K, _FBT_PTS) ≈ basis_fun(b, 1:7, _FBT_K, [p - o for p in _FBT_PTS])
end

@testset "FourierBessel - reflection parity" begin
    pts = _FBT_PTS
    pts_x = [SVector(-p[1], p[2]) for p in pts]    # x -> -x
    pts_y = [SVector(p[1], -p[2]) for p in pts]    # y -> -y
    for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, nothing), (-1, nothing), (nothing, 1), (nothing, -1))
        b = FourierBessel(10; sym_x=sx, sym_y=sy)
        B = basis_fun(b, 1:10, _FBT_K, pts)
        Bx = basis_fun(b, 1:10, _FBT_K, pts_x)
        By = basis_fun(b, 1:10, _FBT_K, pts_y)
        isnothing(sx) || @test Bx ≈ sx .* B
        isnothing(sy) || @test By ≈ sy .* B
    end
end

@testset "FourierBessel - dk_fun and gradient vs finite differences" begin
    h = 1e-6
    k = _FBT_K
    b = FourierBessel(9; gradients=true)
    idx = 1:b.dim
    pts = _FBT_PTS

    # d/dk
    dBdk = dk_fun(b, idx, k, pts)
    fd_k = (basis_fun(b, idx, k + h, pts) .- basis_fun(b, idx, k - h, pts)) ./ (2h)
    @test norm(dBdk - fd_k) < 1e-7
    @test dk_matrix(b, k, pts) ≈ dBdk
    @test dk_fun(b, 3, k, pts) ≈ dBdk[:, 3]
    # fb_term_dk agrees with the matrix entries.
    r = hypot(pts[3]...)
    φ = atan(pts[3][2], pts[3][1])
    @test QuantumBilliards.fb_term_dk(b.orders[5], b.trig[5], k, r, φ) ≈ dBdk[3, 5]

    # d/dx, d/dy
    ex, ey = SVector(h, 0.0), SVector(0.0, h)
    fd_x = (basis_fun(b, idx, k, [p + ex for p in pts]) .- basis_fun(b, idx, k, [p - ex for p in pts])) ./ (2h)
    fd_y = (basis_fun(b, idx, k, [p + ey for p in pts]) .- basis_fun(b, idx, k, [p - ey for p in pts])) ./ (2h)
    dX, dY = gradient(b, idx, k, pts)
    @test norm(dX - fd_x) < 1e-7
    @test norm(dY - fd_y) < 1e-7

    # Single-index forms and the combined basis_and_gradient.
    gx, gy = gradient(b, 4, k, pts)
    @test gx ≈ dX[:, 4] && gy ≈ dY[:, 4]
    bf, bgx, bgy = basis_and_gradient(b, 4, k, pts)
    @test bf ≈ basis_fun(b, 4, k, pts) && bgx ≈ gx && bgy ≈ gy
    Bv, vx, vy = basis_and_gradient(b, idx, k, pts)
    @test Bv ≈ basis_fun(b, idx, k, pts) && vx ≈ dX && vy ≈ dY

    # Matrix-level (no cache) forms.
    Bm, mx, my = basis_and_gradient_matrices(b, k, pts)
    @test Bm ≈ basis_fun(b, idx, k, pts) && mx ≈ dX && my ≈ dY
    gmx, gmy = QuantumBilliards.gradient_matrices(b, k, pts)
    @test gmx ≈ dX && gmy ≈ dY
end

@testset "FourierBessel - type stability" begin
    b = FourierBessel(9; gradients=true)
    k = _FBT_K
    pts = _FBT_PTS
    @inferred basis_fun(b, 3, k, pts)
    @inferred basis_fun(b, 1:9, k, pts)
    @inferred gradient(b, 3, k, pts)
    @inferred dk_fun(b, 3, k, pts)
    @test true
end

@testset "FourierBessel - Taylor cache vs direct evaluation" begin
    k0, k = 7.0, _FBT_K
    pts = _FBT_PTS
    b = FourierBessel(9; gradients=true, taylor_degree=16)
    cache = QuantumBilliards.basis_cache(b, k0, pts)
    @test cache isa QuantumBilliards.FourierBesselTaylorCache{Float64}
    @test cache.k0 == k0 && cache.degree == 16
    @test size(cache.coeffs) == (17, length(pts), b.dim)

    B = basis_fun(b, 1:b.dim, k, pts)
    dBdk = dk_fun(b, 1:b.dim, k, pts)
    dX, dY = gradient(b, 1:b.dim, k, pts)
    tol = 1e-12   # observed ~4e-15

    @test norm(basis_matrix(cache, k) - B) < tol
    @test norm(dk_matrix(cache, k) - dBdk) < tol
    @test norm(basis_matrix(cache, k; multithreaded=false) - B) < tol
    cx, cy = QuantumBilliards.gradient_matrices(cache, b, k)
    @test norm(cx - dX) < tol && norm(cy - dY) < tol
    cB, cdx, cdy = basis_and_gradient_matrices(cache, b, k)
    @test norm(cB - B) < tol && norm(cdx - dX) < tol && norm(cdy - dY) < tol

    # Evaluating at the expansion wavenumber reproduces the zeroth coefficient.
    @test norm(basis_matrix(cache, k0) - basis_fun(b, 1:b.dim, k0, pts)) < tol

    # Weight-scaled form: rows scaled by sqrt(nsym * w).
    w = collect(range(0.2, 1.0; length=length(pts)))
    nsym = 2.0
    Bw, dBw = QuantumBilliards._weight_scaled_basis_and_dk_matrices(cache, k, w, nsym)
    scale = sqrt.(nsym .* w)
    @test norm(Bw - scale .* B) < tol
    @test norm(dBw - scale .* dBdk) < tol
    @test_throws DimensionMismatch QuantumBilliards._weight_scaled_basis_and_dk_matrices(cache, k, w[1:end-1], nsym)

    # Symmetry-reduced basis (mixed cos/sin modes).
    bs = FourierBessel(8; sym_x=1, sym_y=nothing, gradients=true)
    cs = QuantumBilliards.basis_cache(bs, k0, pts)
    @test norm(basis_matrix(cs, k) - basis_fun(bs, 1:8, k, pts)) < tol

    # Caches built from a basis without `gradients` cannot give spatial gradients.
    bng = FourierBessel(4)
    cng = QuantumBilliards.basis_cache(bng, k0, pts)
    @test isnothing(cng.sinφ) && isnothing(cng.cosφ) && isnothing(cng.dang)
    @test_throws ArgumentError QuantumBilliards.gradient_matrices(cng, bng, k)
    @test_throws ArgumentError basis_and_gradient_matrices(cng, bng, k)
    @test norm(basis_matrix(cng, k) - basis_fun(bng, 1:4, k, pts)) < tol
end

@testset "_horner 3D matches manual polynomial evaluation" begin
    rng = MersenneTwister(1234)
    for T in (Float64, Float32)
        C = rand(rng, T, 6, 3, 2)
        for j in 1:3, c in 1:2, δ in (T(0), T(0.3), T(-0.45))
            manual = sum(C[l, j, c] * δ^(l - 1) for l in 1:size(C, 1))
            @test QuantumBilliards._horner(C, j, c, δ) ≈ manual rtol=sqrt(eps(T))
            # Consistent with the value part of the simultaneous value/derivative Horner.
            y, dy = QuantumBilliards._horner_with_derivative(C, j, c, δ)
            @test y ≈ QuantumBilliards._horner(C, j, c, δ) rtol=sqrt(eps(T))
            manual_d = sum((l - 1) * C[l, j, c] * δ^(l - 2) for l in 2:size(C, 1))
            @test dy ≈ manual_d rtol=sqrt(eps(T)) atol=sqrt(eps(T))
        end
    end
    # Constant polynomial.
    @test QuantumBilliards._horner(reshape([2.5], 1, 1, 1), 1, 1, 3.0) == 2.5
    # Known polynomial 1 + 2δ + 3δ²: at δ = 2 -> 17.
    @test QuantumBilliards._horner(reshape([1.0, 2.0, 3.0], 3, 1, 1), 1, 1, 2.0) == 17.0
end

# Integration: Vergini-Saraceno on the stadium. Reference spectra either come
# from the existing RealPlaneWaves testset in solvertests.jl or are values
# observed from a run of the FourierBessel implementation.
@testset "FourierBessel - Vergini Saraceno - Stadium - odd-odd sector" begin
    billiard = StadiumBilliard(0.5)
    solver = VerginiSaracenoSolver(5.0, 10.0)
    k0, dk = 110.0, 0.05
    basis = FourierBessel(12; sym_x=-1, sym_y=-1)
    ks, tens = solve_spectrum(solver, basis, billiard, k0, dk)
    # Reference: RealPlaneWaves(12, sym_x=-1, sym_y=-1), solvertests.jl (atol 1e-3).
    @test length(ks) == 2
    @test all(isapprox.(ks, [109.98398490747392, 110.04189086784442]; atol=1e-3))
    # Observed FourierBessel values.
    @test all(isapprox.(ks, [109.98380, 110.04240]; atol=1e-3))

    # Sector-form constructor gives the same spectrum.
    sector = symmetry_sector(billiard, XAxisReflection => -1, YAxisReflection => -1)
    ks_sec, _ = solve_spectrum(solver, FourierBessel(12, billiard, sector), billiard, k0, dk)
    @test all(isapprox.(ks_sec, ks; atol=1e-6))
end

@testset "FourierBessel - Vergini Saraceno - Stadium - other sectors" begin
    billiard = StadiumBilliard(0.5)
    solver = VerginiSaracenoSolver(5.0, 10.0)
    k0, dk = 110.0, 0.05
    # Observed FourierBessel values; they agree with RealPlaneWaves to ~1e-3.
    observed = Dict(
        (1, 1) => [109.9721, 110.0383],
        (1, -1) => [109.953, 109.9852, 110.0381],
        (-1, 1) => [109.9675, 110.0241, 110.0446],
    )
    for ((sx, sy), ks_ref) in observed
        ks, _ = solve_spectrum(solver, FourierBessel(12; sym_x=sx, sym_y=sy), billiard, k0, dk)
        @test length(ks) == length(ks_ref)
        @test all(isapprox.(ks, ks_ref; atol=1e-3))
    end
end

@testset "FourierBessel - compute_spectrum Taylor vs direct" begin
    billiard = StadiumBilliard(0.5)
    solver = VerginiSaracenoSolver(5.0, 10.0)
    k1, k2, dk = 105.0, 112.0, 0.5
    res_t = QuantumBilliards.compute_spectrum(solver, FourierBessel(12; sym_x=-1, sym_y=-1, use_taylor=true), billiard, k1, k2, dk; show_progress=false)
    res_d = QuantumBilliards.compute_spectrum(solver, FourierBessel(12; sym_x=-1, sym_y=-1, use_taylor=false), billiard, k1, k2, dk; show_progress=false)
    @test !isempty(res_t.k)
    @test length(res_t.k) == length(res_d.k)
    @test all(isapprox.(res_t.k, res_d.k; atol=1e-5))   # observed agreement ~1e-6
end
