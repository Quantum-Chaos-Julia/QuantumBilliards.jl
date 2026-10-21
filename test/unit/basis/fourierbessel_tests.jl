# Unit tests for
#   src/basis/fourierbessel/fourierbessel.jl   (FourierBessel, fb_modes, fb_term, fb_term_dk, _fb_Jvp,
#                                               _fb_axis_characters, the constructors (dim; sym_x, sym_y),
#                                               (dim, symmetries, sym_qnumbers) and (dim, billiard, sector),
#                                               resize_basis, basis_fun, dk_fun, gradient, basis_and_gradient)
#   src/basis/fourierbessel/corneradapted.jl   (Jv and Jvp, only as the Bessel helpers used by the basis)
#   src/solvers/matrixconstructors.jl          (the generic basis_matrix, dk_matrix, gradient_matrices and
#                                               basis_and_gradient_matrices wrappers, only as applied to FourierBessel)
# A basis function is f(r, φ) = J_j(kr) T(jφ) in the polar frame (origin, rot_angle), with T = cos for trig = +1 and
# T = sin for trig = -1. The evaluators are compared with SpecialFunctions.besselj and the trigonometric factor, with the
# reflection parities of the symmetry sectors, with the Helmholtz equation (Δ + k²) f = 0, with central finite
# differences in x, y and k, and with the homogeneity identity k ∂f/∂k = (p - origin)·∇f (f depends on k and p only
# through k (p - origin)), which ties dk_fun to gradient.
# The Taylor-cache side of the basis lives in fourierbessel_recurrence_tests.jl.
# No eigen solves, no pinned wavefunction values. Helper functions use the prefix `fb_`.

using StaticArrays: SVector
using LinearAlgebra: norm
using SpecialFunctions: besselj
using QuantumBilliards: fb_modes, fb_term, fb_term_dk, Jvp, _fb_Jvp, _fb_axis_characters, infer_quantum_numbers,
                        uses_taylor, filter_matrix!, gradient_matrices, basis_and_gradient_matrices

# Fixed deterministic evaluation points, away from the origin and from the symmetry axes.
fb_pts(::Type{T}=Float64) where {T} = SVector{2,T}[SVector{2,T}(0.3 * i - 0.7, 0.2 * i + 0.1) for i in 1:7]

# Closed-form value J_j(kr) T(jφ) in the local frame (origin, rot): the point is rotated by -rot about `origin`.
function fb_ref(j, trig, k, p, origin=SVector(0.0, 0.0), rot=0.0)
    dx, dy = p[1] - origin[1], p[2] - origin[2]
    s, c = sincos(rot)
    xl, yl = c * dx + s * dy, -s * dx + c * dy
    r, φ = hypot(xl, yl), atan(yl, xl)
    return besselj(j, k * r) * (trig > 0 ? cos(j * φ) : sin(j * φ))
end
fb_ref_matrix(basis, k, pts) = [fb_ref(basis.orders[c], basis.trig[c], k, p, basis.origin, basis.rot_angle) for p in pts, c in 1:basis.dim]

# Central finite differences of column i in x, y and k.
fb_shift(pts, dx, dy) = [SVector(p[1] + dx, p[2] + dy) for p in pts]
fb_fd_x(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k, fb_shift(pts, h, 0.0)) .- basis_fun(basis, i, k, fb_shift(pts, -h, 0.0))) ./ (2h)
fb_fd_y(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k, fb_shift(pts, 0.0, h)) .- basis_fun(basis, i, k, fb_shift(pts, 0.0, -h))) ./ (2h)
fb_fd_k(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k + h, pts) .- basis_fun(basis, i, k - h, pts)) ./ (2h)

# Whether every column of `basis` obeys f(-x, y) = sx f(x, y), f(x, -y) = sy f(x, y) (0 means "do not check") and
# f(-x, -y) = sx sy f(x, y) when both are checked.
function fb_reflection_parity(basis, k, pts, sx, sy)
    idx = 1:basis.dim
    B = basis_fun(basis, idx, k, pts)
    ok = true
    if sx != 0
        ok &= isapprox(basis_fun(basis, idx, k, [SVector(-p[1], p[2]) for p in pts]), sx .* B; atol=1e-13)
    end
    if sy != 0
        ok &= isapprox(basis_fun(basis, idx, k, [SVector(p[1], -p[2]) for p in pts]), sy .* B; atol=1e-13)
    end
    if sx != 0 && sy != 0
        ok &= isapprox(basis_fun(basis, idx, k, [SVector(-p[1], -p[2]) for p in pts]), (sx * sy) .* B; atol=1e-13)
    end
    return ok
end

# Residual of the Helmholtz equation of column i at the points, via second differences with step h.
function fb_helmholtz_residual(basis, i, k, pts, h=1e-3)
    f0 = basis_fun(basis, i, k, pts)
    lap = (basis_fun(basis, i, k, fb_shift(pts, h, 0.0)) .+ basis_fun(basis, i, k, fb_shift(pts, -h, 0.0)) .+
           basis_fun(basis, i, k, fb_shift(pts, 0.0, h)) .+ basis_fun(basis, i, k, fb_shift(pts, 0.0, -h)) .- 4 .* f0) ./ h^2
    return lap .+ k^2 .* f0
end

# Field-by-field equality of two bases (symmetries are compared by type).
function fb_same_basis(a, b)
    return a.dim == b.dim && a.orders == b.orders && a.trig == b.trig && a.sym_qnumbers == b.sym_qnumbers &&
           typeof.(something(a.symmetries, [])) == typeof.(something(b.symmetries, [])) &&
           a.origin == b.origin && a.rot_angle == b.rot_angle &&
           (a.use_taylor, a.taylor_degree, a.taylor_radius, a.taylor_tol, a.gradients) ==
           (b.use_taylor, b.taylor_degree, b.taylor_radius, b.taylor_tol, b.gradients)
end

@testset "FourierBessel type" begin
    @test FourierBessel <: AbsBasis
    @test !isabstracttype(FourierBessel)
    @test fieldnames(FourierBessel) == (:origin, :rot_angle, :dim, :symmetries, :sym_qnumbers, :orders, :trig,
                                        :use_taylor, :taylor_degree, :taylor_radius, :taylor_tol, :gradients)
    basis = FourierBessel(3)
    @test basis isa FourierBessel{Float64}
    @test isconcretetype(typeof(basis))
end

@testset "fb_modes" begin
    modes(sx, sy, dim) = collect(zip(fb_modes(sx, sy, dim)...))
    c, s = Int8(1), Int8(-1)

    @testset "sector mode lists (docstring table)" begin
        @test modes(1, 1, 4) == [(0, c), (2, c), (4, c), (6, c)]
        @test modes(1, -1, 4) == [(1, s), (3, s), (5, s), (7, s)]
        @test modes(-1, 1, 4) == [(1, c), (3, c), (5, c), (7, c)]
        @test modes(-1, -1, 4) == [(2, s), (4, s), (6, s), (8, s)]
        @test modes(1, nothing, 4) == [(0, c), (1, s), (2, c), (3, s)]
        @test modes(-1, nothing, 4) == [(1, c), (2, s), (3, c), (4, s)]
        @test modes(nothing, 1, 4) == [(0, c), (1, c), (2, c), (3, c)]
        @test modes(nothing, -1, 4) == [(1, s), (2, s), (3, s), (4, s)]
        @test modes(nothing, nothing, 5) == [(0, c), (1, c), (1, s), (2, c), (2, s)]
    end

    @testset "return types" begin
        orders, trig = fb_modes(1, 1, 3)
        @test orders isa Vector{Int}
        @test trig isa Vector{Int8}
        @test @inferred(fb_modes(1, -1, 3)) isa Tuple{Vector{Int},Vector{Int8}}
        @test @inferred(fb_modes(nothing, nothing, 3)) isa Tuple{Vector{Int},Vector{Int8}}
    end

    @testset "dim = 1 and dim = 0" begin
        @test fb_modes(nothing, nothing, 1) == ([0], Int8[1])
        @test fb_modes(1, -1, 1) == ([1], Int8[-1])
        @test fb_modes(-1, -1, 1) == ([2], Int8[-1])
        @test fb_modes(nothing, nothing, 0) == (Int[], Int8[])
    end

    @testset "the first dim modes of a longer list are the shorter list" begin
        for (sx, sy) in ((1, 1), (-1, -1), (nothing, nothing), (-1, nothing), (nothing, 1))
            o6, t6 = fb_modes(sx, sy, 6)
            o3, t3 = fb_modes(sx, sy, 3)
            @test o6[1:3] == o3 && t6[1:3] == t3
        end
    end

    @testset "modes are distinct, ordered by angular order, and have the requested reflection parities" begin
        φ0 = 0.7
        for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, nothing), (-1, nothing), (nothing, 1), (nothing, -1), (nothing, nothing))
            orders, trig = fb_modes(sx, sy, 12)
            @test allunique(zip(orders, trig))
            @test issorted(orders)
            @test all(orders .>= 0)
            @test all(j > 0 || t == 1 for (j, t) in zip(orders, trig))     # sin(0) = 0 is not a mode
            T(t, a) = t > 0 ? cos(a) : sin(a)
            for (j, t) in zip(orders, trig)
                # y -> -y is φ -> -φ, x -> -x is φ -> π - φ
                parity_y = T(t, -j * φ0) / T(t, j * φ0)
                parity_x = T(t, j * (π - φ0)) / T(t, j * φ0)
                isnothing(sy) || @test parity_y ≈ sy
                isnothing(sx) || @test parity_x ≈ sx
            end
        end
    end

    @testset "the four reflection sectors partition the unrestricted modes" begin
        sectors = [fb_modes(sx, sy, 12) for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1))]
        J = minimum(maximum(o) for (o, _) in sectors)
        union_modes = Set((j, t) for (o, tr) in sectors for (j, t) in zip(o, tr) if j <= J)
        o, tr = fb_modes(nothing, nothing, 2J + 1)
        @test union_modes == Set(zip(o, tr))
        @test length(union_modes) == 2J + 1
    end
end

@testset "FourierBessel(dim; sym_x, sym_y)" begin
    @testset "no symmetry" begin
        b = FourierBessel(4)
        @test b.dim == 4
        @test isnothing(b.symmetries) && isnothing(b.sym_qnumbers)
        @test (b.orders, b.trig) == fb_modes(nothing, nothing, 4)
        @test b.origin == SVector(0.0, 0.0) && b.rot_angle == 0.0
        @test b.use_taylor === true
        @test (b.taylor_degree, b.taylor_radius, b.taylor_tol, b.gradients) == (16, 0.5, 1e-11, false)
    end

    @testset "both reflections: symmetry order [Y, XY, X] and qnumbers [sym_x, sym_x sym_y, sym_y]" begin
        b = FourierBessel(5; sym_x=1, sym_y=-1)
        @test (b.orders, b.trig) == fb_modes(1, -1, 5)
        @test b.symmetries isa Vector{BilliardGeometry.AbsReflection}
        @test typeof.(b.symmetries) == [YAxisReflection, XYAxisReflection, XAxisReflection]
        @test b.sym_qnumbers == [1.0, -1.0, -1.0]
        @test FourierBessel(5; sym_x=-1, sym_y=-1).sym_qnumbers == [-1.0, 1.0, -1.0]
        @test FourierBessel(5; sym_x=-1, sym_y=1).sym_qnumbers == [-1.0, -1.0, 1.0]
    end

    @testset "y-axis reflection only (sym_x)" begin
        b = FourierBessel(4; sym_x=-1)
        @test typeof.(b.symmetries) == [YAxisReflection]
        @test b.sym_qnumbers == [-1.0]
        @test (b.orders, b.trig) == fb_modes(-1, nothing, 4)
    end

    @testset "x-axis reflection only (sym_y)" begin
        b = FourierBessel(4; sym_y=1)
        @test typeof.(b.symmetries) == [XAxisReflection]
        @test b.sym_qnumbers == [1.0]
        @test (b.orders, b.trig) == fb_modes(nothing, 1, 4)
    end

    @testset "non-default origin and rotation without symmetry" begin
        b = FourierBessel(3; origin=SVector(0.3, -0.2), rot_angle=0.4)
        @test b.origin == SVector(0.3, -0.2) && b.rot_angle == 0.4
        @test isnothing(b.symmetries)
        # An integer origin is promoted to the common floating-point type.
        bi = FourierBessel(3; origin=SVector(1, 2))
        @test bi isa FourierBessel{Float64}
        @test bi.origin === SVector(1.0, 2.0)
    end

    @testset "Taylor and gradient keywords are stored" begin
        b = FourierBessel(3; use_taylor=false, taylor_degree=10, taylor_radius=0.25, taylor_tol=1e-9, gradients=true)
        @test !b.use_taylor && b.taylor_degree == 10 && b.taylor_radius == 0.25
        @test b.taylor_tol == 1e-9 && b.gradients
    end

    @testset "uses_taylor follows use_taylor" begin
        @test uses_taylor(FourierBessel(3)) === true
        @test uses_taylor(FourierBessel(3; use_taylor=false)) === false
    end
end

@testset "FourierBessel - constructor argument errors" begin
    @test_throws ArgumentError FourierBessel(4; sym_x=2)
    @test_throws ArgumentError FourierBessel(4; sym_x=-3)
    @test_throws ArgumentError FourierBessel(4; sym_y=0)
    @test_throws ArgumentError FourierBessel(0)
    @test_throws ArgumentError FourierBessel(-2)
    @test_throws ArgumentError FourierBessel(4; taylor_degree=0)
    @test_throws ArgumentError FourierBessel(4; taylor_degree=-1)
    @test_throws ArgumentError FourierBessel(4; taylor_radius=-1.0)
    @test_throws ArgumentError FourierBessel(4; taylor_radius=0.0)
    @test_throws ArgumentError FourierBessel(4; taylor_tol=0.0)
    @test_throws ArgumentError FourierBessel(4; taylor_tol=-1e-9)
    @test_throws TypeError FourierBessel(4; sym_x=1.0)                    # characters are integers or nothing
    @test_throws MethodError FourierBessel(4.0)

    @testset "axis reflections act about the global origin" begin
        @test_throws ArgumentError FourierBessel(4; sym_x=1, origin=SVector(0.1, 0.0))
        @test_throws ArgumentError FourierBessel(4; sym_y=-1, origin=SVector(0.0, 0.1))
        @test_throws ArgumentError FourierBessel(4; sym_y=-1, rot_angle=0.3)
        @test_throws ArgumentError FourierBessel(4; sym_x=1, sym_y=1, rot_angle=0.3)
        # Without a symmetry the same arguments are accepted.
        @test FourierBessel(4; origin=SVector(0.1, 0.0), rot_angle=0.3) isa FourierBessel
    end

    @testset "raw form" begin
        abs_refl = BilliardGeometry.AbsReflection
        @test_throws ArgumentError FourierBessel(4, abs_refl[XYAxisReflection()], [1.0])
        @test_throws ArgumentError FourierBessel(4, abs_refl[XAxisReflection(), XYAxisReflection()], [1.0, 1.0])
        @test_throws ArgumentError FourierBessel(4, abs_refl[YAxisReflection(), XYAxisReflection()], [1.0, 1.0])
        @test_throws ArgumentError FourierBessel(4, abs_refl[XAxisReflection()], nothing)
        @test_throws ArgumentError FourierBessel(4, abs_refl[XAxisReflection()], [1.0, 1.0])
        @test_throws ArgumentError FourierBessel(4, abs_refl[DiagonalReflection()], [1.0])
        @test_throws ArgumentError FourierBessel(4, abs_refl[XAxisReflection()], [1.0]; taylor_degree=0)
        err = try FourierBessel(4, abs_refl[DiagonalReflection()], [1.0]); nothing catch e; e end
        @test err isa ArgumentError && occursin("supports only", err.msg)
    end
end

@testset "_fb_axis_characters" begin
    abs_refl = BilliardGeometry.AbsReflection
    X, Y, XY = XAxisReflection(), YAxisReflection(), XYAxisReflection()

    @testset "no symmetries" begin
        @test _fb_axis_characters(nothing, nothing) == (nothing, nothing)
        @test _fb_axis_characters(nothing, [1.0]) == (nothing, nothing)
        @test _fb_axis_characters(abs_refl[], Float64[]) == (nothing, nothing)
    end

    @testset "single reflections: Y-axis gives sym_x, X-axis gives sym_y" begin
        @test _fb_axis_characters(abs_refl[Y], [-1.0]) == (-1, nothing)
        @test _fb_axis_characters(abs_refl[X], [1.0]) == (nothing, 1)
        @test _fb_axis_characters(abs_refl[X], [-1]) == (nothing, -1)          # integer quantum numbers
        @test _fb_axis_characters(abs_refl[X], [-1.0f0]) == (nothing, -1)
    end

    @testset "both reflections, with the XY reflection in any position" begin
        @test _fb_axis_characters(abs_refl[Y, XY, X], [1.0, -1.0, -1.0]) == (1, -1)
        @test _fb_axis_characters(abs_refl[X, XY, Y], [-1.0, -1.0, 1.0]) == (1, -1)
        @test _fb_axis_characters(abs_refl[Y, X], [-1.0, 1.0]) == (-1, 1)
        @test _fb_axis_characters(abs_refl[X, Y], [-1.0, 1.0]) == (1, -1)
    end

    @testset "invalid combinations" begin
        @test_throws ArgumentError _fb_axis_characters(abs_refl[X], nothing)
        @test_throws ArgumentError _fb_axis_characters(abs_refl[X, Y], [1.0])
        @test_throws ArgumentError _fb_axis_characters(abs_refl[XY], [1.0])
        @test_throws ArgumentError _fb_axis_characters(abs_refl[X, XY], [1.0, 1.0])
        @test_throws ArgumentError _fb_axis_characters(abs_refl[Y, XY], [1.0, 1.0])
        @test_throws ArgumentError _fb_axis_characters(abs_refl[DiagonalReflection()], [1.0])
    end

    @testset "round trip with the stored symmetries of a constructed basis" begin
        for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, nothing), (-1, nothing), (nothing, 1), (nothing, -1), (nothing, nothing))
            b = FourierBessel(3; sym_x=sx, sym_y=sy)
            @test _fb_axis_characters(b.symmetries, b.sym_qnumbers) == (sx, sy)
        end
    end
end

@testset "FourierBessel(dim, symmetries, sym_qnumbers)" begin
    abs_refl = BilliardGeometry.AbsReflection

    @testset "rebuilds the basis the symmetries came from" begin
        for (sx, sy) in ((1, -1), (-1, -1), (1, nothing), (nothing, -1), (nothing, nothing))
            b = FourierBessel(4; sym_x=sx, sym_y=sy)
            @test fb_same_basis(FourierBessel(4, b.symmetries, b.sym_qnumbers), b)
        end
        b = FourierBessel(4; sym_x=1, sym_y=-1)
        @test (FourierBessel(7, b.symmetries, b.sym_qnumbers).orders, FourierBessel(7, b.symmetries, b.sym_qnumbers).trig) == fb_modes(1, -1, 7)
    end

    @testset "nothing means unrestricted" begin
        @test fb_same_basis(FourierBessel(4, nothing, nothing), FourierBessel(4))
    end

    @testset "the XY quantum number is recomputed as sym_x * sym_y" begin
        b = FourierBessel(4, abs_refl[YAxisReflection(), XYAxisReflection(), XAxisReflection()], [1.0, 1.0, -1.0])
        @test b.sym_qnumbers == [1.0, -1.0, -1.0]
    end

    @testset "keywords are forwarded" begin
        b = FourierBessel(3, abs_refl[XAxisReflection()], [1.0]; use_taylor=false, taylor_degree=6, taylor_radius=0.3, taylor_tol=1e-8, gradients=true)
        @test (b.use_taylor, b.taylor_degree, b.taylor_radius, b.taylor_tol, b.gradients) == (false, 6, 0.3, 1e-8, true)
        bo = FourierBessel(3, nothing, nothing; origin=SVector(0.2, 0.1), rot_angle=0.5)
        @test bo.origin == SVector(0.2, 0.1) && bo.rot_angle == 0.5
    end
end

@testset "FourierBessel(dim, billiard, sector)" begin
    stadium = StadiumBilliard(0.5)

    @testset "full D2 sector equals the sym_x/sym_y constructor" begin
        for (sx, sy) in ((1, 1), (1, -1), (-1, 1), (-1, -1))
            # `YAxisReflection` is x -> -x (sym_x); `XAxisReflection` is y -> -y (sym_y).
            sector = symmetry_sector(stadium, XAxisReflection => sy, YAxisReflection => sx)
            @test fb_same_basis(FourierBessel(6, stadium, sector), FourierBessel(6; sym_x=sx, sym_y=sy))
        end
    end

    @testset "single reflection sectors" begin
        @test fb_same_basis(FourierBessel(6, stadium, symmetry_sector(stadium, XAxisReflection => -1)), FourierBessel(6; sym_y=-1))
        @test fb_same_basis(FourierBessel(6, stadium, symmetry_sector(stadium, YAxisReflection => 1)), FourierBessel(6; sym_x=1))
    end

    @testset "keywords are forwarded" begin
        sector = symmetry_sector(stadium, XAxisReflection => 1, YAxisReflection => 1)
        b = FourierBessel(4, stadium, sector; use_taylor=false, taylor_degree=7, taylor_radius=0.2, taylor_tol=1e-8, gradients=true)
        @test (b.use_taylor, b.taylor_degree, b.taylor_radius, b.taylor_tol, b.gradients) == (false, 7, 0.2, 1e-8, true)
    end

    @testset "sectors FourierBessel cannot represent are rejected" begin
        c3 = C3Billiard(0.2)
        @test_throws ArgumentError FourierBessel(6, c3, symmetry_sector(c3, NFoldRotation => 0))
        @test_throws ArgumentError FourierBessel(6, c3, symmetry_sector(c3, NFoldRotation => 1))
        err = try FourierBessel(6, c3, symmetry_sector(c3, NFoldRotation => 1)); nothing catch e; e end
        @test err isa ArgumentError && occursin("XAxisReflection/YAxisReflection", err.msg)
    end

    @testset "sector of another billiard instance is rejected" begin
        sector = symmetry_sector(stadium, XAxisReflection => 1, YAxisReflection => 1)
        @test_throws ArgumentError FourierBessel(4, StadiumBilliard(0.5), sector)
    end
end

@testset "resize_basis" begin
    billiard = StadiumBilliard(0.5)

    @testset "same dimension is a fixed point" begin
        b = FourierBessel(4; sym_x=1, sym_y=-1)
        @test resize_basis(b, billiard, 4, 7.3) === b
    end

    @testset "growing keeps the symmetry sector and every setting" begin
        b = FourierBessel(4; sym_x=1, sym_y=-1, use_taylor=false, taylor_degree=10, taylor_radius=0.25, taylor_tol=1e-9, gradients=true)
        b2 = resize_basis(b, billiard, 6, 7.3)
        @test b2 isa FourierBessel{Float64}
        @test b2.dim == 6
        @test (b2.orders, b2.trig) == fb_modes(1, -1, 6)
        @test fb_same_basis(b2, FourierBessel(6; sym_x=1, sym_y=-1, use_taylor=false, taylor_degree=10, taylor_radius=0.25, taylor_tol=1e-9, gradients=true))
        @test b.dim == 4                                                   # the input basis is untouched
    end

    @testset "shrinking keeps the leading modes" begin
        b = FourierBessel(6; sym_x=1, sym_y=-1)
        b3 = resize_basis(b, billiard, 3, 7.3)
        @test b3.orders == b.orders[1:3] && b3.trig == b.trig[1:3]
    end

    @testset "origin and rotation of an unrestricted basis are kept" begin
        b = FourierBessel(3; origin=SVector(0.3, -0.2), rot_angle=0.4)
        b2 = resize_basis(b, billiard, 5, 7.3)
        @test isnothing(b2.symmetries) && isnothing(b2.sym_qnumbers)
        @test b2.origin == SVector(0.3, -0.2) && b2.rot_angle == 0.4
        @test (b2.orders, b2.trig) == fb_modes(nothing, nothing, 5)
    end

    @testset "half-plane symmetry is kept and the wavenumber is ignored" begin
        b = FourierBessel(4; sym_x=-1)
        @test fb_same_basis(resize_basis(b, billiard, 7, 1.0), resize_basis(b, billiard, 7, 300.0))
        @test fb_same_basis(resize_basis(b, billiard, 7, 1.0), FourierBessel(7; sym_x=-1))
    end

    @testset "return type" begin
        @test resize_basis(FourierBessel(3), billiard, 5, 1.0) isa FourierBessel{Float64}
        @test resize_basis(FourierBessel(3; sym_x=1), billiard, 5, 1.0) isa FourierBessel{Float64}
        # Source finding: the (dim, symmetries, sym_qnumbers; kwargs...) constructor is not inferred to a concrete type,
        # so resize_basis returns abstract `FourierBessel` to inference (RealPlaneWaves infers concretely).
        @test_broken (try; @inferred(resize_basis(FourierBessel(3), billiard, 5, 1.0)); true; catch; false; end)
    end
end

@testset "Jvp and _fb_Jvp" begin
    xs = (0.3, 1.0, 2.5, 7.0, 12.0)

    @testset "Jvp equals the recurrence (J_{ν-1} - J_{ν+1}) / 2" begin
        for ν in 0:6, x in xs
            @test Jvp(ν, x) ≈ (besselj(ν - 1, x) - besselj(ν + 1, x)) / 2 atol = 1e-13
        end
    end

    @testset "Jvp equals the Bessel equation J_ν' = ν J_ν / x - J_{ν+1}" begin
        for ν in 0:6, x in xs
            @test Jvp(ν, x) ≈ ν * besselj(ν, x) / x - besselj(ν + 1, x) atol = 1e-13
        end
    end

    @testset "J_0' = -J_1" begin
        for x in xs
            @test Jvp(0, x) ≈ -besselj(1, x) atol = 1e-13
            @test _fb_Jvp(0, x) ≈ -besselj(1, x) atol = 1e-13
        end
    end

    @testset "values at the origin: J_1'(0) = 1/2, J_ν'(0) = 0 otherwise" begin
        @test Jvp(1, 0.0) ≈ 0.5
        @test _fb_Jvp(1, 0.0) ≈ 0.5
        for ν in (0, 2, 3, 5)
            @test abs(Jvp(ν, 0.0)) < 1e-15
            @test abs(_fb_Jvp(ν, 0.0)) < 1e-15
        end
    end

    @testset "_fb_Jvp agrees with Jvp and a central difference" begin
        h = 1e-6
        for j in 0:6, x in xs
            @test _fb_Jvp(j, x) ≈ Jvp(j, x) atol = 1e-13
            @test _fb_Jvp(j, x) ≈ (besselj(j, x + h) - besselj(j, x - h)) / (2h) atol = 1e-8
        end
    end

    @testset "Float32 and type stability" begin
        @test @inferred(_fb_Jvp(2, 1.5)) isa Float64
        @test @inferred(_fb_Jvp(2, 1.5f0)) isa Float32
        @test @inferred(Jvp(2, 1.5)) isa Float64
        # Source finding: `Jvp` multiplies by the Float64 literal 0.5, so a Float32 argument returns Float64.
        @test_broken @inferred(Jvp(2, 1.5f0)) isa Float32
        @test Jvp(2, 1.5f0) ≈ Jvp(2, 1.5) atol = 1e-6
        @test _fb_Jvp(2, 1.5f0) ≈ Float32(_fb_Jvp(2, 1.5)) atol = 1e-6
    end
end

@testset "fb_term and fb_term_dk" begin
    k, r, φ = 3.7, 0.8, 0.6

    @testset "value J_j(kr) T(jφ) for both angular factors" begin
        for j in 0:5
            @test fb_term(j, 1, k, r, φ) ≈ besselj(j, k * r) * cos(j * φ) atol = 1e-13
            @test fb_term(j, -1, k, r, φ) ≈ besselj(j, k * r) * sin(j * φ) atol = 1e-13
        end
        @test fb_term(0, 1, k, r, φ) ≈ besselj(0, k * r) atol = 1e-13       # j = 0 cosine is purely radial
        @test fb_term(3, Int8(1), k, r, φ) == fb_term(3, 1, k, r, φ)         # any integer selector; > 0 means cos
        @test fb_term(3, Int8(-1), k, r, φ) == fb_term(3, -1, k, r, φ)
    end

    @testset "the wavenumber derivative is r J_j'(kr) T(jφ) and matches a central difference" begin
        h = 1e-6
        for j in 0:5, τ in (1, -1)
            @test fb_term_dk(j, τ, k, r, φ) ≈ r * Jvp(j, k * r) * (τ > 0 ? cos(j * φ) : sin(j * φ)) atol = 1e-13
            fd = (fb_term(j, τ, k + h, r, φ) - fb_term(j, τ, k - h, r, φ)) / (2h)
            @test fb_term_dk(j, τ, k, r, φ) ≈ fd atol = 1e-8
        end
    end

    @testset "at r = 0 only the j = 0 cosine is non-zero and nothing depends on k" begin
        @test fb_term(0, 1, k, 0.0, φ) == 1.0
        @test all(fb_term(j, τ, k, 0.0, φ) == 0.0 for j in 1:4, τ in (1, -1))
        @test all(fb_term_dk(j, τ, k, 0.0, φ) == 0.0 for j in 0:4, τ in (1, -1))
    end

    @testset "type stability and Float32" begin
        @test @inferred(fb_term(2, 1, k, r, φ)) isa Float64
        @test @inferred(fb_term_dk(2, -1, k, r, φ)) isa Float64
        @test @inferred(fb_term(2, Int8(1), 3.7f0, 0.8f0, 0.6f0)) isa Float32
        @test @inferred(fb_term_dk(2, Int8(-1), 3.7f0, 0.8f0, 0.6f0)) isa Float32
        @test fb_term(2, 1, 3.7f0, 0.8f0, 0.6f0) ≈ Float32(fb_term(2, 1, k, r, φ)) atol = 1e-6
    end
end

@testset "basis_fun - SpecialFunctions reference" begin
    k = 7.3
    pts = fb_pts()

    @testset "unrestricted basis: matrix of cos and sin modes" begin
        b = FourierBessel(9)
        B = basis_fun(b, 1:b.dim, k, pts)
        @test size(B) == (length(pts), 9)
        @test B ≈ fb_ref_matrix(b, k, pts) atol = 1e-13
        for c in 1:b.dim, (p, pt) in enumerate(pts)
            @test isapprox(B[p, c], fb_ref(b.orders[c], b.trig[c], k, pt); atol=1e-13)
        end
    end

    @testset "even-even sector (cos, even j)" begin
        b = FourierBessel(5; sym_x=1, sym_y=1)
        @test b.orders == [0, 2, 4, 6, 8] && all(==(1), b.trig)
        @test basis_fun(b, 1:5, k, pts) ≈ fb_ref_matrix(b, k, pts) atol = 1e-13
    end

    @testset "odd-odd sector (sin, even j >= 2)" begin
        b = FourierBessel(5; sym_x=-1, sym_y=-1)
        @test b.orders == [2, 4, 6, 8, 10] && all(==(-1), b.trig)
        @test basis_fun(b, 1:5, k, pts) ≈ fb_ref_matrix(b, k, pts) atol = 1e-13
    end

    @testset "mixed cos/sin half-plane sector" begin
        b = FourierBessel(6; sym_x=-1)
        @test basis_fun(b, 1:6, k, pts) ≈ fb_ref_matrix(b, k, pts) atol = 1e-13
    end

    @testset "the j = 0 column is radial and constant on circles about the origin" begin
        b = FourierBessel(1)
        ring = [SVector(0.9 * cos(t), 0.9 * sin(t)) for t in range(0.1, 6.0; length=7)]
        col = basis_fun(b, 1, k, ring)
        @test all(≈(besselj(0, 0.9 * k); atol=1e-13), col)
    end

    @testset "fb_term is the matrix entry" begin
        b = FourierBessel(7)
        B = basis_fun(b, 1:7, k, pts)
        p = pts[2]
        r, φ = hypot(p[1], p[2]), atan(p[2], p[1])
        @test fb_term(b.orders[4], b.trig[4], k, r, φ) ≈ B[2, 4]
    end

    @testset "Bessel decay: |J_j(kr)| bounded by one" begin
        B = basis_fun(FourierBessel(10), 1:10, 20.0, pts)
        @test all(abs.(B) .<= 1 + 1e-14)
    end

    @testset "origin: only the j = 0 cosine is non-zero" begin
        b = FourierBessel(7)
        B = basis_fun(b, 1:7, k, [SVector(0.0, 0.0)])
        for c in 1:7
            @test B[1, c] == (b.orders[c] == 0 ? 1.0 : 0.0)
        end
    end

    @testset "k = 0 gives the constant j = 0 function" begin
        b = FourierBessel(5)
        B = basis_fun(b, 1:5, 0.0, pts)
        for c in 1:5
            @test B[:, c] == (b.orders[c] == 0 ? ones(length(pts)) : zeros(length(pts)))
        end
    end
end

@testset "basis_fun - method consistency" begin
    k = 2.7
    pts = fb_pts()
    b = FourierBessel(8)
    B = basis_fun(b, 1:b.dim, k, pts)

    @testset "single-index method equals the matrix column" begin
        for i in 1:b.dim
            @test basis_fun(b, i, k, pts) == B[:, i]
        end
    end

    @testset "index subsets select and reorder columns" begin
        @test basis_fun(b, [5, 2], k, pts) == B[:, [5, 2]]
        @test basis_fun(b, 4:6, k, pts) == B[:, 4:6]
        @test size(basis_fun(b, Int[], k, pts)) == (length(pts), 0)
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
end

@testset "basis_fun - origin and rotation of the polar frame" begin
    k = 4.1
    pts = fb_pts()
    o = SVector(0.15, -0.1)
    θ = 0.4

    @testset "shifted origin: evaluating about o equals evaluating about 0 at pts - o" begin
        b = FourierBessel(7)
        bo = FourierBessel(7; origin=o)
        @test basis_fun(bo, 1:7, k, pts) ≈ basis_fun(b, 1:7, k, [p - o for p in pts]) atol = 1e-13
        @test basis_fun(bo, 1:7, k, pts) ≈ fb_ref_matrix(bo, k, pts) atol = 1e-13
    end

    @testset "rotated frame: the point is rotated by -rot_angle about the origin" begin
        b = FourierBessel(7)
        br = FourierBessel(7; rot_angle=θ)
        s, c = sincos(θ)
        local_pts = [SVector(c * p[1] + s * p[2], -s * p[1] + c * p[2]) for p in pts]
        @test basis_fun(br, 1:7, k, pts) ≈ basis_fun(b, 1:7, k, local_pts) atol = 1e-13
        @test basis_fun(br, 1:7, k, pts) ≈ fb_ref_matrix(br, k, pts) atol = 1e-13
        # A rotation by 2π is the identity.
        @test basis_fun(FourierBessel(7; rot_angle=2π), 1:7, k, pts) ≈ basis_fun(b, 1:7, k, pts) atol = 1e-12
    end

    @testset "shifted and rotated frame" begin
        b = FourierBessel(7; origin=o, rot_angle=θ)
        @test basis_fun(b, 1:7, k, pts) ≈ fb_ref_matrix(b, k, pts) atol = 1e-13
    end

    @testset "rotating the frame by a multiple of 2π/j rotates mode j onto itself" begin
        # Rotation by π/2 maps cos(jφ) to cos(jφ - jπ/2); for j = 4 this is the identity.
        b = FourierBessel(9)
        br = FourierBessel(9; rot_angle=π / 2)
        for c in findall(==(4), b.orders)
            @test basis_fun(br, c, k, pts) ≈ basis_fun(b, c, k, pts) atol = 1e-12
        end
    end
end

@testset "basis_matrix, dk_matrix and gradient matrices" begin
    k = 2.7
    pts = fb_pts()

    @testset "basis_matrix equals the filtered columns of basis_fun" begin
        b = FourierBessel(7)
        raw = basis_fun(b, 1:b.dim, k, pts)
        B = basis_matrix(b, k, pts)
        @test size(B) == (length(pts), b.dim)
        @test B == filter_matrix!(copy(raw))
        @test all(isapprox(B[:, i], basis_fun(b, i, k, pts); atol=1e-14) for i in 1:b.dim)
        @test basis_matrix(b, k, pts; multithreaded=false) == B
    end

    @testset "entries below machine epsilon are zeroed" begin
        b = FourierBessel(4; sym_x=-1, sym_y=-1)
        # sin(jφ) at a point on the symmetry axis x ≈ 0 is ~1e-16, far below the filter threshold of the row.
        pt = [SVector(1e-20, 0.3)]
        raw = basis_fun(b, 1:b.dim, k, pt)
        @test any(!iszero, raw) && all(abs.(raw) .<= eps())
        @test all(iszero, basis_matrix(b, k, pt))
        @test any(!iszero, basis_fun(b, 1:b.dim, k, [SVector(0.1, 0.3)]))
    end

    @testset "symmetry-reduced basis" begin
        b = FourierBessel(5; sym_x=-1)
        @test basis_matrix(b, k, pts) ≈ fb_ref_matrix(b, k, pts) atol = 1e-13
        @test size(basis_matrix(b, k, pts)) == (7, 5)
    end

    @testset "dk_matrix equals the filtered dk_fun" begin
        b = FourierBessel(7)
        D = dk_matrix(b, k, pts)
        @test D == filter_matrix!(dk_fun(b, 1:b.dim, k, pts))
        @test D ≈ [fb_term_dk(b.orders[c], b.trig[c], k, hypot(p[1], p[2]), atan(p[2], p[1])) for p in pts, c in 1:b.dim] atol = 1e-13
    end

    @testset "gradient_matrices and basis_and_gradient_matrices equal the filtered evaluators" begin
        b = FourierBessel(7)
        dX, dY = gradient(b, 1:b.dim, k, pts)
        mx, my = gradient_matrices(b, k, pts)
        @test mx == filter_matrix!(copy(dX)) && my == filter_matrix!(copy(dY))
        Bm, bx, by = basis_and_gradient_matrices(b, k, pts)
        @test Bm == basis_matrix(b, k, pts) && bx == mx && by == my
    end
end

@testset "basis_fun - reflection parity of symmetry sectors" begin
    k = 3.1
    pts = fb_pts()

    @testset "even-even sector (sym_x = 1, sym_y = 1)" begin
        @test fb_reflection_parity(FourierBessel(6; sym_x=1, sym_y=1), k, pts, 1, 1)
    end

    @testset "odd-even sector (sym_x = -1, sym_y = 1)" begin
        @test fb_reflection_parity(FourierBessel(6; sym_x=-1, sym_y=1), k, pts, -1, 1)
    end

    @testset "even-odd sector (sym_x = 1, sym_y = -1)" begin
        @test fb_reflection_parity(FourierBessel(6; sym_x=1, sym_y=-1), k, pts, 1, -1)
    end

    @testset "odd-odd sector (sym_x = -1, sym_y = -1)" begin
        @test fb_reflection_parity(FourierBessel(6; sym_x=-1, sym_y=-1), k, pts, -1, -1)
    end

    @testset "y-axis reflection only: x parity fixed, y parity free" begin
        even = FourierBessel(8; sym_x=1)
        odd = FourierBessel(8; sym_x=-1)
        @test fb_reflection_parity(even, k, pts, 1, 0)
        @test fb_reflection_parity(odd, k, pts, -1, 0)
        @test !fb_reflection_parity(even, k, pts, 0, 1)                    # y parity is not fixed
        # An odd-in-x function vanishes on the symmetry axis x = 0.
        @test all(<(1e-14) ∘ abs, basis_fun(odd, 1:odd.dim, k, [SVector(0.0, 0.2), SVector(0.0, -0.7)]))
    end

    @testset "x-axis reflection only: y parity fixed, x parity free" begin
        even = FourierBessel(8; sym_y=1)
        odd = FourierBessel(8; sym_y=-1)
        @test fb_reflection_parity(even, k, pts, 0, 1)
        @test fb_reflection_parity(odd, k, pts, 0, -1)
        @test !fb_reflection_parity(even, k, pts, 1, 0)
        # An odd-in-y function vanishes on the symmetry axis y = 0.
        @test all(<(1e-14) ∘ abs, basis_fun(odd, 1:odd.dim, k, [SVector(0.4, 0.0), SVector(-0.9, 0.0)]))
    end

    @testset "the parity check detects the wrong sector" begin
        @test !fb_reflection_parity(FourierBessel(6; sym_x=1, sym_y=1), k, pts, -1, 1)
        @test !fb_reflection_parity(FourierBessel(6; sym_x=-1, sym_y=-1), k, pts, 1, -1)
    end
end

@testset "basis_fun - Helmholtz equation" begin
    k = 3.1
    pts = fb_pts()

    @testset "unrestricted basis, origin at zero" begin
        b = FourierBessel(7)
        for i in 1:b.dim
            @test maximum(abs, fb_helmholtz_residual(b, i, k, pts)) < 1e-4
        end
    end

    @testset "shifted and rotated frame" begin
        b = FourierBessel(5; origin=SVector(0.15, -0.1), rot_angle=0.4)
        for i in 1:b.dim
            @test maximum(abs, fb_helmholtz_residual(b, i, k, pts)) < 1e-4
        end
    end

    @testset "the residual check is not vacuous: a wrong wavenumber is detected" begin
        b = FourierBessel(3)
        residual = fb_helmholtz_residual(b, 3, k, pts)                      # (Δ + k²) f
        @test maximum(abs, residual) < 1e-4
        # (Δ + k'²) f = residual + (k'² - k²) f is far from zero for k' = 1.3 k.
        wrong = residual .+ ((1.3 * k)^2 - k^2) .* basis_fun(b, 3, k, pts)
        @test maximum(abs, wrong) > 1e-2
    end
end

@testset "dk_fun vs finite differences" begin
    k = 7.3
    pts = fb_pts()

    @testset "all columns of an unrestricted basis" begin
        b = FourierBessel(9)
        idx = 1:b.dim
        dB = dk_fun(b, idx, k, pts)
        fd = hcat((fb_fd_k(b, i, k, pts) for i in idx)...)
        @test size(dB) == (length(pts), b.dim)
        @test norm(dB - fd) < 1e-7
    end

    @testset "single-index and index-set methods" begin
        b = FourierBessel(9)
        dB = dk_fun(b, 1:b.dim, k, pts)
        for i in 1:b.dim
            @test dk_fun(b, i, k, pts) == dB[:, i]
        end
        @test dk_fun(b, [5, 2], k, pts) == dB[:, [5, 2]]
        @test dk_fun(b, 1:b.dim, k, pts; multithreaded=false) == dB
    end

    @testset "symmetry-reduced basis, shifted origin" begin
        b = FourierBessel(6; sym_x=-1)
        @test norm(dk_fun(b, 1:6, k, pts) - hcat((fb_fd_k(b, i, k, pts) for i in 1:6)...)) < 1e-7
        bo = FourierBessel(6; origin=SVector(0.2, 0.1), rot_angle=0.3)
        @test norm(dk_fun(bo, 1:6, k, pts) - hcat((fb_fd_k(bo, i, k, pts) for i in 1:6)...)) < 1e-7
    end

    @testset "origin: derivative vanishes for every mode" begin
        b = FourierBessel(6)
        @test all(iszero, dk_fun(b, 1:6, k, [SVector(0.0, 0.0)]))
    end

    @testset "empty points" begin
        b = FourierBessel(3)
        @test dk_fun(b, 1, k, SVector{2,Float64}[]) == Float64[]
        @test size(dk_fun(b, 1:3, k, SVector{2,Float64}[])) == (0, 3)
    end
end

@testset "gradient and basis_and_gradient vs finite differences" begin
    k = 7.3
    pts = fb_pts()

    @testset "unrestricted basis" begin
        b = FourierBessel(9)
        dX, dY = gradient(b, 1:b.dim, k, pts)
        fdx = hcat((fb_fd_x(b, i, k, pts) for i in 1:b.dim)...)
        fdy = hcat((fb_fd_y(b, i, k, pts) for i in 1:b.dim)...)
        @test size(dX) == size(dY) == (length(pts), b.dim)
        @test norm(dX - fdx) < 1e-7
        @test norm(dY - fdy) < 1e-7
    end

    @testset "shifted and rotated frame" begin
        b = FourierBessel(9; origin=SVector(0.15, -0.1), rot_angle=0.4)
        dX, dY = gradient(b, 1:b.dim, k, pts)
        @test norm(dX - hcat((fb_fd_x(b, i, k, pts) for i in 1:b.dim)...)) < 1e-7
        @test norm(dY - hcat((fb_fd_y(b, i, k, pts) for i in 1:b.dim)...)) < 1e-7
    end

    @testset "symmetry-reduced bases" begin
        for (sx, sy) in ((1, 1), (-1, -1), (-1, nothing), (nothing, 1))
            b = FourierBessel(6; sym_x=sx, sym_y=sy)
            dX, dY = gradient(b, 1:6, k, pts)
            @test norm(dX - hcat((fb_fd_x(b, i, k, pts) for i in 1:6)...)) < 1e-7
            @test norm(dY - hcat((fb_fd_y(b, i, k, pts) for i in 1:6)...)) < 1e-7
        end
    end

    @testset "closed form for J_0(kr): ∇f = -k J_1(kr) (x, y) / r" begin
        b = FourierBessel(1)
        gx, gy = gradient(b, 1, k, pts)
        for (p, pt) in enumerate(pts)
            r = hypot(pt[1], pt[2])
            @test gx[p] ≈ -k * besselj(1, k * r) * pt[1] / r atol = 1e-12
            @test gy[p] ≈ -k * besselj(1, k * r) * pt[2] / r atol = 1e-12
        end
    end

    @testset "single-index forms and basis_and_gradient" begin
        b = FourierBessel(9)
        dX, dY = gradient(b, 1:9, k, pts)
        gx, gy = gradient(b, 4, k, pts)
        @test gx == dX[:, 4] && gy == dY[:, 4]
        bf, bgx, bgy = basis_and_gradient(b, 4, k, pts)
        @test bf == basis_fun(b, 4, k, pts) && bgx == gx && bgy == gy
        Bv, vx, vy = basis_and_gradient(b, 1:9, k, pts)
        @test Bv == basis_fun(b, 1:9, k, pts) && vx == dX && vy == dY
        Bs, sx, sy = basis_and_gradient(b, [5, 2], k, pts)
        @test Bs == Bv[:, [5, 2]] && sx == dX[:, [5, 2]] && sy == dY[:, [5, 2]]
    end

    @testset "origin: documented zero gradient, true limit approached nearby" begin
        b = FourierBessel(4)                                               # modes (0,cos), (1,cos), (1,sin), (2,cos)
        dX, dY = gradient(b, 1:4, k, [SVector(0.0, 0.0)])
        @test all(iszero, dX[:, [1, 3, 4]]) && all(iszero, dY)
        # Source finding: the r == 0 guard only drops the 1/r angular term, so the radial term k J_1'(0) cos(0) = k/2
        # survives for the (1, cos) mode, contradicting the docstring ("points at the origin are assigned a zero
        # gradient") and the Taylor-cache path, which returns exactly zero there.
        @test_broken iszero(dX[1, 2])
        # Near the origin J_1(kr) cos φ ≈ k x / 2, so ∇ ≈ (k/2, 0); J_1(kr) sin φ ≈ k y / 2, so ∇ ≈ (0, k/2).
        gx, gy = gradient(b, 2, k, [SVector(1e-8, 0.0)])
        @test gx[1] ≈ k / 2 atol = 1e-6
        @test abs(gy[1]) < 1e-6
        gx, gy = gradient(b, 3, k, [SVector(0.0, 1e-8)])
        @test abs(gx[1]) < 1e-6
        @test gy[1] ≈ k / 2 atol = 1e-6
    end

    @testset "empty points" begin
        b = FourierBessel(3)
        gx, gy = gradient(b, 1, k, SVector{2,Float64}[])
        @test gx == Float64[] && gy == Float64[]
        @test all(size.(gradient(b, 1:3, k, SVector{2,Float64}[])) .== Ref((0, 3)))
    end
end

@testset "dk_fun and gradient - homogeneity identity k ∂f/∂k = (p - origin)·∇f" begin
    k = 5.3
    pts = fb_pts()

    for (name, b) in (("unrestricted", FourierBessel(8)),
                      ("odd-even sector", FourierBessel(6; sym_x=-1, sym_y=1)),
                      ("shifted and rotated frame", FourierBessel(8; origin=SVector(0.15, -0.1), rot_angle=0.4)))
        @testset "$name" begin
            idx = 1:b.dim
            dB = dk_fun(b, idx, k, pts)
            dX, dY = gradient(b, idx, k, pts)
            relx = [p[1] - b.origin[1] for p in pts]
            rely = [p[2] - b.origin[2] for p in pts]
            @test k .* dB ≈ relx .* dX .+ rely .* dY atol = 1e-11
        end
    end
end

@testset "FourierBessel - threaded equals serial" begin
    k = 4.4
    pts = fb_pts()
    b = FourierBessel(9; origin=SVector(0.1, 0.2), rot_angle=0.3)
    idx = 1:b.dim

    @test basis_fun(b, idx, k, pts; multithreaded=true) == basis_fun(b, idx, k, pts; multithreaded=false)
    @test dk_fun(b, idx, k, pts; multithreaded=true) == dk_fun(b, idx, k, pts; multithreaded=false)
    @test gradient(b, idx, k, pts; multithreaded=true) == gradient(b, idx, k, pts; multithreaded=false)
    @test basis_and_gradient(b, idx, k, pts; multithreaded=true) == basis_and_gradient(b, idx, k, pts; multithreaded=false)
    @test basis_matrix(b, k, pts; multithreaded=true) == basis_matrix(b, k, pts; multithreaded=false)
    @test dk_matrix(b, k, pts; multithreaded=true) == dk_matrix(b, k, pts; multithreaded=false)
end

@testset "FourierBessel - type stability" begin
    b = FourierBessel(9)
    k = 4.4
    pts = fb_pts()
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
    @test @inferred(FourierBessel(4; sym_x=1, sym_y=-1)) isa FourierBessel{Float64}
end

@testset "FourierBessel - Float32" begin
    kw32 = (origin=SVector(0.0f0, 0.0f0), rot_angle=0.0f0, taylor_radius=0.5f0, taylor_tol=1.0f-6)
    b32 = FourierBessel(6; sym_x=-1, kw32...)
    b64 = FourierBessel(6; sym_x=-1)
    pts32 = fb_pts(Float32)
    pts64 = fb_pts(Float64)
    k32, k64 = 3.3f0, 3.3
    idx = 1:6

    @testset "construction keeps the element type" begin
        @test b32 isa FourierBessel{Float32}
        @test b32.origin isa SVector{2,Float32}
        @test b32.sym_qnumbers isa Vector{Float32} && b32.sym_qnumbers == [-1.0f0]
        @test b32.orders == b64.orders && b32.trig == b64.trig
        @test resize_basis(b32, StadiumBilliard(0.5), 8, k32) isa FourierBessel{Float32}
        @test FourierBessel(4; kw32...) isa FourierBessel{Float32}
    end

    @testset "evaluators stay in Float32 and agree with Float64" begin
        @test @inferred(basis_fun(b32, idx, k32, pts32)) isa Matrix{Float32}
        @test @inferred(basis_fun(b32, 2, k32, pts32)) isa Vector{Float32}
        @test @inferred(dk_fun(b32, idx, k32, pts32)) isa Matrix{Float32}
        @test @inferred(gradient(b32, idx, k32, pts32)) isa Tuple{Matrix{Float32},Matrix{Float32}}
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
end

@testset "infer_quantum_numbers on FourierBessel symmetries" begin
    # The helper defaults every quantum number to +1, which is the even-even FourierBessel sector.
    even = FourierBessel(4; sym_x=1, sym_y=1)
    @test infer_quantum_numbers(even.symmetries) == even.sym_qnumbers
    @test infer_quantum_numbers(even.symmetries) isa Vector{Float64}
    @test fb_same_basis(FourierBessel(4, even.symmetries, infer_quantum_numbers(even.symmetries)), even)

    # For the other sectors the stored quantum numbers differ from the +1 defaults.
    odd = FourierBessel(4; sym_x=-1, sym_y=-1)
    @test infer_quantum_numbers(odd.symmetries) != odd.sym_qnumbers
    @test infer_quantum_numbers(FourierBessel(4).symmetries) === nothing
end
