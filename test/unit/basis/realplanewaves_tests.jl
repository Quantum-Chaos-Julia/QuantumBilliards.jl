# Unit tests for
#   src/basis/planewaves/realplanewaves.jl   (RealPlaneWaves, parity_pattern, infer_quantum_numbers, the constructors
#                                             (dim; sym_x, sym_y), (dim, symmetries[, sym_qnumbers]) and
#                                             (dim, billiard, sector), resize_basis, basis_fun, gradient,
#                                             basis_and_gradient, dk_fun)
#   src/solvers/matrixconstructors.jl        (the generic basis_matrix, dk_matrix, gradient_matrices and
#                                             basis_and_gradient_matrices wrappers, only as applied to RealPlaneWaves)
# Every basis function is the separable product F_x(k cos(t) x) F_y(k sin(t) y) with F = cos for parity +1 and
# F = sin for parity -1, so the evaluators are compared with this closed form, with the reflection parities
# f(-x, y) = s_x f(x, y), f(x, -y) = s_y f(x, y), with the Helmholtz equation (Δ + k²) f = 0, and with central finite
# differences in x, y and k. The homogeneity identity k ∂f/∂k = x ∂f/∂x + y ∂f/∂y ties dk_fun to the gradient.
# The Taylor-cache side of the basis lives in realplanewaves_recurrence_tests.jl.
# No eigen solves, no pinned wavefunction values. Helper functions use the prefix `rpw_`.

using StaticArrays: SVector
using QuantumBilliards: parity_pattern, infer_quantum_numbers, filter_matrix!, gradient_matrices

# Fixed deterministic evaluation points, away from the symmetry axes.
rpw_pts(::Type{T}=Float64) where {T} = SVector{2,T}[SVector{2,T}(0.15 * j - 0.4, 0.5 - 0.17 * j) for j in 1:6]

# Closed-form value of the i-th basis function at (x, y).
rpw_trig(par, a) = par == 1 ? cos(a) : sin(a)
rpw_ref(basis, i, k, p) = rpw_trig(basis.parity_x[i], k * cos(basis.angles[i]) * p[1]) *
                          rpw_trig(basis.parity_y[i], k * sin(basis.angles[i]) * p[2])
rpw_ref_matrix(basis, k, pts) = [rpw_ref(basis, i, k, p) for p in pts, i in 1:basis.dim]

# Central finite differences of column i in x, y and k.
rpw_shift(pts, dx, dy) = [SVector(p[1] + dx, p[2] + dy) for p in pts]
rpw_fd_x(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k, rpw_shift(pts, h, 0.0)) .- basis_fun(basis, i, k, rpw_shift(pts, -h, 0.0))) ./ (2h)
rpw_fd_y(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k, rpw_shift(pts, 0.0, h)) .- basis_fun(basis, i, k, rpw_shift(pts, 0.0, -h))) ./ (2h)
rpw_fd_k(basis, i, k, pts, h=1e-6) = (basis_fun(basis, i, k + h, pts) .- basis_fun(basis, i, k - h, pts)) ./ (2h)

# Whether every column of `basis` obeys f(-x, y) = sx f(x, y) (sx = 0 means "do not check") and likewise for y.
function rpw_reflection_parity(basis, k, pts, sx, sy)
    B = basis_fun(basis, 1:basis.dim, k, pts)
    ok = true
    if sx != 0
        ok &= isapprox(basis_fun(basis, 1:basis.dim, k, [SVector(-p[1], p[2]) for p in pts]), sx .* B; atol=1e-13)
    end
    if sy != 0
        ok &= isapprox(basis_fun(basis, 1:basis.dim, k, [SVector(p[1], -p[2]) for p in pts]), sy .* B; atol=1e-13)
    end
    return ok
end

# Residual of the Helmholtz equation of column i at the points, via second differences with step h.
function rpw_helmholtz_residual(basis, i, k, pts, h=1e-3)
    f0 = basis_fun(basis, i, k, pts)
    lap = (basis_fun(basis, i, k, rpw_shift(pts, h, 0.0)) .+ basis_fun(basis, i, k, rpw_shift(pts, -h, 0.0)) .+
           basis_fun(basis, i, k, rpw_shift(pts, 0.0, h)) .+ basis_fun(basis, i, k, rpw_shift(pts, 0.0, -h)) .- 4 .* f0) ./ h^2
    return lap .+ k^2 .* f0
end

# Field-by-field equality of two bases (symmetries are compared by type).
function rpw_same_basis(a, b)
    return a.dim == b.dim && a.angles == b.angles && a.parity_x == b.parity_x && a.parity_y == b.parity_y &&
           a.sym_qnumbers == b.sym_qnumbers && typeof.(something(a.symmetries, [])) == typeof.(something(b.symmetries, [])) &&
           a.angle_arc == b.angle_arc && a.angle_shift == b.angle_shift &&
           (a.use_taylor, a.taylor_degree, a.taylor_radius, a.taylor_tol) == (b.use_taylor, b.taylor_degree, b.taylor_radius, b.taylor_tol)
end

@testset "RealPlaneWaves type" begin
    @test RealPlaneWaves <: AbsBasis
    @test !isabstracttype(RealPlaneWaves)
    @test fieldnames(RealPlaneWaves) == (:dim, :symmetries, :sym_qnumbers, :angle_arc, :angle_shift, :angles,
                                         :parity_x, :parity_y, :sampler, :use_taylor, :taylor_degree, :taylor_radius, :taylor_tol)
    basis = RealPlaneWaves(3)
    @test basis isa RealPlaneWaves{Float64,LinearNodes{Float64}}
    @test isconcretetype(typeof(basis))
end

@testset "parity_pattern" begin
    @testset "no symmetry: all four quadrants" begin
        @test parity_pattern(nothing, nothing) == ([1, 1, -1, -1], [1, -1, 1, -1])
    end

    @testset "x-axis reflection only: y parity fixed, x parity cos and sin" begin
        syms = BilliardGeometry.AbsReflection[XAxisReflection()]
        @test parity_pattern(syms, [1.0]) == ([1, -1], [1, 1])
        @test parity_pattern(syms, [-1.0]) == ([1, -1], [-1, -1])
    end

    @testset "y-axis reflection only: x parity fixed, y parity cos and sin" begin
        syms = BilliardGeometry.AbsReflection[YAxisReflection()]
        @test parity_pattern(syms, [1.0]) == ([1, 1], [1, -1])
        @test parity_pattern(syms, [-1.0]) == ([-1, -1], [1, -1])
    end

    @testset "both axis reflections: one quadrant" begin
        syms = BilliardGeometry.AbsReflection[YAxisReflection(), XYAxisReflection(), XAxisReflection()]
        @test parity_pattern(syms, [1.0, 1.0, 1.0]) == ([1], [1])
        @test parity_pattern(syms, [-1.0, -1.0, 1.0]) == ([-1], [1])     # [sym_x, sym_x*sym_y, sym_y]
        @test parity_pattern(syms, [1.0, -1.0, -1.0]) == ([1], [-1])
        @test parity_pattern(syms, [-1.0, 1.0, -1.0]) == ([-1], [-1])
        # The pattern is read off by symmetry type, not by position.
        reordered = BilliardGeometry.AbsReflection[XAxisReflection(), XYAxisReflection(), YAxisReflection()]
        @test parity_pattern(reordered, [-1.0, 1.0, 1.0]) == ([1], [-1])
    end

    @testset "unsupported reflections fall back to the unrestricted pattern" begin
        syms = BilliardGeometry.AbsReflection[DiagonalReflection()]
        @test parity_pattern(syms, [1.0]) == parity_pattern(nothing, nothing)
    end

    @testset "return types" begin
        @test @inferred(parity_pattern(nothing, nothing)) isa Tuple{Vector{Int},Vector{Int}}
        @test parity_pattern(BilliardGeometry.AbsReflection[XAxisReflection()], [1.0]) isa Tuple{Vector{Int},Vector{Int}}
    end
end

@testset "infer_quantum_numbers" begin
    @test infer_quantum_numbers(nothing) === nothing
    @test infer_quantum_numbers(BilliardGeometry.AbsReflection[]) === nothing
    @test infer_quantum_numbers(BilliardGeometry.AbsReflection[XAxisReflection()]) == [1.0]
    q = infer_quantum_numbers(BilliardGeometry.AbsReflection[YAxisReflection(), XYAxisReflection(), XAxisReflection()])
    @test q == [1.0, 1.0, 1.0]
    @test q isa Vector{Float64}
end

@testset "RealPlaneWaves(dim; sym_x, sym_y) - no symmetry" begin
    n = 5
    basis = RealPlaneWaves(n)
    @test basis.dim == 4n
    @test basis.symmetries === nothing
    @test basis.sym_qnumbers === nothing
    @test basis.angle_arc ≈ π
    @test basis.angle_shift == 0
    @test length(basis.angles) == length(basis.parity_x) == length(basis.parity_y) == 4n
    # Linear (midpoint) nodes t_i = (i - 1/2)/n; each angle is repeated once per quadrant.
    t = [(i - 0.5) / n for i in 1:n]
    @test basis.angles ≈ repeat(t .* π, inner=4)
    @test basis.parity_x == repeat([1, 1, -1, -1], n)
    @test basis.parity_y == repeat([1, -1, 1, -1], n)
    @test basis.use_taylor === true
    @test (basis.taylor_degree, basis.taylor_radius, basis.taylor_tol) == (16, 0.5, 1e-11)
end

@testset "RealPlaneWaves(dim; sym_x, sym_y) - x-axis reflection only (sym_y)" begin
    n = 4
    basis = RealPlaneWaves(n; sym_y=-1)
    @test basis.dim == 2n
    @test basis.symmetries isa Vector{BilliardGeometry.AbsReflection}
    @test typeof.(basis.symmetries) == [XAxisReflection]
    @test basis.sym_qnumbers == [-1.0]
    @test [basis.angle_arc, basis.angle_shift] ≈ [π, 0.0]
    @test basis.angles ≈ repeat([(i - 0.5) / n * π for i in 1:n], inner=2)
    @test basis.parity_x == repeat([1, -1], n)
    @test all(==(-1), basis.parity_y)
    @test RealPlaneWaves(n; sym_y=1).parity_y == fill(1, 2n)
end

@testset "RealPlaneWaves(dim; sym_x, sym_y) - y-axis reflection only (sym_x)" begin
    n = 4
    basis = RealPlaneWaves(n; sym_x=-1)
    @test basis.dim == 2n
    @test typeof.(basis.symmetries) == [YAxisReflection]
    @test basis.sym_qnumbers == [-1.0]
    @test [basis.angle_arc, basis.angle_shift] ≈ [π, -π / 2]        # right half-plane of directions
    @test basis.angles ≈ repeat([(i - 0.5) / n * π - π / 2 for i in 1:n], inner=2)
    @test all(==(-1), basis.parity_x)
    @test basis.parity_y == repeat([1, -1], n)
    @test all(a -> -π / 2 < a < π / 2, basis.angles)
    @test RealPlaneWaves(n; sym_x=1).parity_x == fill(1, 2n)
end

@testset "RealPlaneWaves(dim; sym_x, sym_y) - both reflections" begin
    n = 6
    basis = RealPlaneWaves(n; sym_x=-1, sym_y=1)
    @test basis.dim == n
    @test typeof.(basis.symmetries) == [YAxisReflection, XYAxisReflection, XAxisReflection]
    @test basis.sym_qnumbers == [-1.0, -1.0, 1.0]                    # [sym_x, sym_x*sym_y, sym_y]
    @test RealPlaneWaves(n; sym_x=-1, sym_y=-1).sym_qnumbers == [-1.0, 1.0, -1.0]
    @test [basis.angle_arc, basis.angle_shift] ≈ [π / 2, 0.0]        # first quadrant only
    @test basis.angles ≈ [(i - 0.5) / n * π / 2 for i in 1:n]
    @test all(a -> 0 < a < π / 2, basis.angles)
    @test basis.parity_x == fill(-1, n)
    @test basis.parity_y == fill(1, n)
    @test RealPlaneWaves(n; sym_x=1, sym_y=-1).parity_x == fill(1, n)
    @test RealPlaneWaves(n; sym_x=1, sym_y=-1).parity_y == fill(-1, n)
end

@testset "RealPlaneWaves - angle and sampler keywords" begin
    basis = RealPlaneWaves(4; angle_arc=1.0, angle_shift=0.25)
    @test (basis.angle_arc, basis.angle_shift) == (1.0, 0.25)
    @test basis.angles ≈ repeat([(i - 0.5) / 4 + 0.25 for i in 1:4], inner=4)

    # Gauss-Legendre nodes on [0, 1] are symmetric about 1/2, so the sampled angles are symmetric about the arc midpoint.
    gl = RealPlaneWaves(4; sampler=GaussLegendreNodes())
    @test gl.sampler isa GaussLegendreNodes
    @test gl.dim == 16
    a = unique(gl.angles)
    @test a ≈ π .- reverse(a)
    @test issorted(a) && length(a) == 4

    # Taylor parameters are stored unchanged.
    tb = RealPlaneWaves(3; use_taylor=false, taylor_degree=8, taylor_radius=0.25, taylor_tol=1e-9)
    @test tb.use_taylor === false
    @test (tb.taylor_degree, tb.taylor_radius, tb.taylor_tol) == (8, 0.25, 1e-9)
end

@testset "RealPlaneWaves(dim, symmetries[, sym_qnumbers])" begin
    @testset "explicit quantum numbers" begin
        basis = RealPlaneWaves(3, BilliardGeometry.AbsReflection[YAxisReflection()], [-1])
        @test basis.dim == 6
        @test basis.sym_qnumbers == [-1.0]
        @test all(==(-1), basis.parity_x)
        @test basis.parity_y == repeat([1, -1], 3)
        @test [basis.angle_arc, basis.angle_shift] ≈ [π, 0.0]         # no auto-adjustment in this form
    end

    @testset "nothing means unrestricted" begin
        basis = RealPlaneWaves(3, nothing, nothing)
        @test rpw_same_basis(basis, RealPlaneWaves(3))
        @test rpw_same_basis(RealPlaneWaves(3, nothing), RealPlaneWaves(3))
    end

    @testset "inferred quantum numbers are all +1" begin
        syms = BilliardGeometry.AbsReflection[YAxisReflection(), XYAxisReflection(), XAxisReflection()]
        basis = RealPlaneWaves(3, syms; angle_arc=π / 2)
        @test basis.sym_qnumbers == [1.0, 1.0, 1.0]
        @test basis.parity_x == fill(1, 3) && basis.parity_y == fill(1, 3)
        @test rpw_same_basis(basis, RealPlaneWaves(3; sym_x=1, sym_y=1))
    end

    @testset "angle and Taylor keywords are forwarded" begin
        basis = RealPlaneWaves(3, BilliardGeometry.AbsReflection[XAxisReflection()]; angle_arc=2.0, angle_shift=0.5,
                               use_taylor=false, taylor_degree=6, taylor_radius=0.3, taylor_tol=1e-8)
        @test (basis.angle_arc, basis.angle_shift) == (2.0, 0.5)
        @test (basis.use_taylor, basis.taylor_degree, basis.taylor_radius, basis.taylor_tol) == (false, 6, 0.3, 1e-8)
    end
end

@testset "RealPlaneWaves - constructor argument errors" begin
    syms = BilliardGeometry.AbsReflection[XAxisReflection()]
    @test_throws ArgumentError RealPlaneWaves(4, syms, [1.0, -1.0])        # one quantum number per symmetry
    @test_throws ArgumentError RealPlaneWaves(4, BilliardGeometry.AbsReflection[YAxisReflection(), XAxisReflection()], [1.0])
    @test_throws ArgumentError RealPlaneWaves(4; taylor_degree=0)
    @test_throws ArgumentError RealPlaneWaves(4; taylor_degree=-3)
    @test_throws ArgumentError RealPlaneWaves(4; taylor_radius=0.0)
    @test_throws ArgumentError RealPlaneWaves(4; taylor_radius=-0.5)
    @test_throws ArgumentError RealPlaneWaves(4; taylor_tol=0.0)
    @test_throws ArgumentError RealPlaneWaves(4; taylor_tol=-1e-9)
    @test_throws ArgumentError RealPlaneWaves(4, syms; taylor_degree=0)
    @test_throws ArgumentError RealPlaneWaves(4, nothing, nothing; taylor_tol=-1.0)
    @test_throws TypeError RealPlaneWaves(4; sym_x=1.0)                    # parities are integers
    @test_throws MethodError RealPlaneWaves(4.0)
    # Quantum numbers must be ±1 (or nothing).
    @test_throws ArgumentError RealPlaneWaves(4; sym_x=2)
end

@testset "RealPlaneWaves(dim, billiard, sector)" begin
    rect = RectangleBilliard(1.0, 0.6)
    n = 4

    @testset "full D2 sector equals the sym_x/sym_y constructor" begin
        # XAxisReflection constrains the y parity, YAxisReflection the x parity.
        sector = symmetry_sector(rect, XAxisReflection => -1, YAxisReflection => 1)
        basis = RealPlaneWaves(n, rect, sector)
        @test rpw_same_basis(basis, RealPlaneWaves(n; sym_x=1, sym_y=-1))
        @test basis.parity_x == fill(1, n) && basis.parity_y == fill(-1, n)
        @test basis.sym_qnumbers == [1.0, -1.0, -1.0]
    end

    @testset "single y-axis reflection sector" begin
        sector = symmetry_sector(rect, YAxisReflection => -1)
        @test rpw_same_basis(RealPlaneWaves(n, rect, sector), RealPlaneWaves(n; sym_x=-1))
    end

    @testset "single x-axis reflection sector" begin
        sector = symmetry_sector(rect, XAxisReflection => 1)
        @test rpw_same_basis(RealPlaneWaves(n, rect, sector), RealPlaneWaves(n; sym_y=1))
    end

    @testset "trivial sector is the unrestricted basis" begin
        @test rpw_same_basis(RealPlaneWaves(n, rect, symmetry_sector(rect)), RealPlaneWaves(n))
    end

    @testset "keywords are forwarded" begin
        sector = symmetry_sector(rect, XAxisReflection => 1, YAxisReflection => 1)
        basis = RealPlaneWaves(n, rect, sector; angle_arc=1.2, angle_shift=0.1, sampler=GaussLegendreNodes(),
                               use_taylor=false, taylor_degree=7, taylor_radius=0.2, taylor_tol=1e-8)
        @test (basis.angle_arc, basis.angle_shift) == (1.2, 0.1)
        @test basis.sampler isa GaussLegendreNodes
        @test (basis.use_taylor, basis.taylor_degree, basis.taylor_radius, basis.taylor_tol) == (false, 7, 0.2, 1e-8)
    end

    @testset "sectors RealPlaneWaves cannot represent are rejected" begin
        star = StarBilliard(1.0, 0.3, 3)
        @test_throws ArgumentError RealPlaneWaves(n, star, symmetry_sector(star, 1))            # rotation sector
        @test_throws ArgumentError RealPlaneWaves(n, rect, symmetry_sector(rect, XYAxisReflection => 1))
        err = try RealPlaneWaves(n, star, symmetry_sector(star, 1)); nothing catch e; e end
        @test err isa ArgumentError && occursin("XAxisReflection/YAxisReflection", err.msg)
    end

    @testset "sector of another billiard instance is rejected" begin
        sector = symmetry_sector(RectangleBilliard(1.0, 0.6), YAxisReflection => 1)
        @test_throws ArgumentError RealPlaneWaves(n, rect, sector)
    end
end

@testset "resize_basis" begin
    rect = RectangleBilliard(1.0, 0.6)

    @testset "unrestricted basis" begin
        basis = RealPlaneWaves(3; angle_arc=2.0, angle_shift=0.3, use_taylor=false, taylor_degree=9, taylor_radius=0.2, taylor_tol=1e-8)
        resized = resize_basis(basis, rect, 7, 5.0)
        @test resized isa RealPlaneWaves{Float64,LinearNodes{Float64}}
        @test resized.dim == 28
        @test rpw_same_basis(resized, RealPlaneWaves(7; angle_arc=2.0, angle_shift=0.3, use_taylor=false, taylor_degree=9, taylor_radius=0.2, taylor_tol=1e-8))
        @test resized.sampler === basis.sampler
        @test basis.dim == 12                                              # the input basis is untouched
    end

    @testset "symmetries, quantum numbers and angular range are preserved" begin
        basis = RealPlaneWaves(3; sym_x=-1, sym_y=1)
        resized = resize_basis(basis, rect, 10, 12.0)
        @test resized.dim == 10
        @test resized.symmetries == basis.symmetries
        @test resized.sym_qnumbers == basis.sym_qnumbers
        @test [resized.angle_arc, resized.angle_shift] ≈ [π / 2, 0.0]
        @test rpw_same_basis(resized, RealPlaneWaves(10; sym_x=-1, sym_y=1))

        half = RealPlaneWaves(3; sym_x=-1)
        r = resize_basis(half, rect, 5, 1.0)
        @test r.dim == 10 && r.angle_shift ≈ -π / 2
        @test rpw_same_basis(r, RealPlaneWaves(5; sym_x=-1))
    end

    @testset "same dimension is a fixed point; the wavenumber argument is ignored" begin
        basis = RealPlaneWaves(6; sym_y=-1)
        @test rpw_same_basis(resize_basis(basis, rect, 6, 3.0), basis)
        @test rpw_same_basis(resize_basis(basis, rect, 9, 3.0), resize_basis(basis, rect, 9, 300.0))
    end

    @testset "sampler is preserved" begin
        basis = RealPlaneWaves(3; sampler=GaussLegendreNodes())
        resized = resize_basis(basis, rect, 5, 1.0)
        @test resized.sampler isa GaussLegendreNodes
        @test rpw_same_basis(resized, RealPlaneWaves(5; sampler=GaussLegendreNodes()))
    end

    @testset "return type is inferred" begin
        @test @inferred(resize_basis(RealPlaneWaves(3), rect, 5, 1.0)) isa RealPlaneWaves{Float64,LinearNodes{Float64}}
    end
end

@testset "basis_fun - closed-form trig reference" begin
    k = 3.3
    pts = rpw_pts()

    @testset "cos-cos pattern (sym_x = 1, sym_y = 1)" begin
        basis = RealPlaneWaves(3; sym_x=1, sym_y=1)
        for i in 1:basis.dim
            θ = basis.angles[i]
            expected = [cos(k * cos(θ) * p[1]) * cos(k * sin(θ) * p[2]) for p in pts]
            @test basis_fun(basis, i, k, pts) ≈ expected atol = 1e-13
        end
    end

    @testset "sin-cos pattern (sym_x = -1, sym_y = 1)" begin
        basis = RealPlaneWaves(3; sym_x=-1, sym_y=1)
        for i in 1:basis.dim
            θ = basis.angles[i]
            expected = [sin(k * cos(θ) * p[1]) * cos(k * sin(θ) * p[2]) for p in pts]
            @test basis_fun(basis, i, k, pts) ≈ expected atol = 1e-13
        end
    end

    @testset "cos-sin pattern (sym_x = 1, sym_y = -1)" begin
        basis = RealPlaneWaves(3; sym_x=1, sym_y=-1)
        for i in 1:basis.dim
            θ = basis.angles[i]
            expected = [cos(k * cos(θ) * p[1]) * sin(k * sin(θ) * p[2]) for p in pts]
            @test basis_fun(basis, i, k, pts) ≈ expected atol = 1e-13
        end
    end

    @testset "sin-sin pattern (sym_x = -1, sym_y = -1)" begin
        basis = RealPlaneWaves(3; sym_x=-1, sym_y=-1)
        for i in 1:basis.dim
            θ = basis.angles[i]
            expected = [sin(k * cos(θ) * p[1]) * sin(k * sin(θ) * p[2]) for p in pts]
            @test basis_fun(basis, i, k, pts) ≈ expected atol = 1e-13
        end
    end

    @testset "unrestricted basis: matrix of all four patterns" begin
        basis = RealPlaneWaves(3)
        B = basis_fun(basis, 1:basis.dim, k, pts)
        @test size(B) == (length(pts), 12)
        @test B ≈ rpw_ref_matrix(basis, k, pts) atol = 1e-13
    end

    @testset "values are bounded by one" begin
        B = basis_fun(RealPlaneWaves(4), 1:16, 7.0, pts)
        @test all(abs.(B) .<= 1 + 1e-14)
    end

    @testset "origin: cos-cos columns are one, columns with a sine factor vanish" begin
        basis = RealPlaneWaves(3)
        B = basis_fun(basis, 1:basis.dim, k, [SVector(0.0, 0.0)])
        for i in 1:basis.dim
            @test B[1, i] == (basis.parity_x[i] == 1 && basis.parity_y[i] == 1 ? 1.0 : 0.0)
        end
    end

    @testset "k = 0 gives the constant cos-cos function" begin
        basis = RealPlaneWaves(3)
        B = basis_fun(basis, 1:basis.dim, 0.0, pts)
        @test all(B[:, i] == (basis.parity_x[i] == 1 && basis.parity_y[i] == 1 ? ones(length(pts)) : zeros(length(pts))) for i in 1:basis.dim)
    end
end

@testset "basis_fun - method consistency" begin
    k = 2.7
    pts = rpw_pts()
    basis = RealPlaneWaves(3)
    B = basis_fun(basis, 1:basis.dim, k, pts)

    @testset "single-index method equals the matrix column" begin
        @test basis_fun(basis, 1, k, pts) == B[:, 1]
        @test basis_fun(basis, 7, k, pts) == B[:, 7]
        @test basis_fun(basis, basis.dim, k, pts) == B[:, basis.dim]
    end

    @testset "index subsets select and reorder columns" begin
        @test basis_fun(basis, [5, 2], k, pts) == B[:, [5, 2]]
        @test basis_fun(basis, 4:6, k, pts) == B[:, 4:6]
        @test size(basis_fun(basis, Int[], k, pts)) == (length(pts), 0)
    end

    @testset "empty point set" begin
        @test basis_fun(basis, 1, k, SVector{2,Float64}[]) == Float64[]
        @test size(basis_fun(basis, 1:basis.dim, k, SVector{2,Float64}[])) == (0, basis.dim)
    end

    @testset "points may be any 2-element indexable" begin
        tuples = [(p[1], p[2]) for p in pts]
        @test basis_fun(basis, 1:basis.dim, k, tuples) == B
    end
end

@testset "basis_matrix" begin
    k = 2.7
    pts = rpw_pts()

    @testset "unrestricted basis equals the filtered columns of basis_fun" begin
        basis = RealPlaneWaves(3)
        raw = basis_fun(basis, 1:basis.dim, k, pts)
        B = basis_matrix(basis, k, pts)
        @test size(B) == (length(pts), basis.dim)
        @test B == filter_matrix!(copy(raw))
        @test all(isapprox(B[:, i], basis_fun(basis, i, k, pts); atol=1e-14) for i in 1:basis.dim)
    end

    @testset "entries below machine epsilon are zeroed" begin
        basis = RealPlaneWaves(3; sym_x=-1, sym_y=-1)
        # sin(k vx x) with x = 1e-20 is far below eps, so the whole row is flushed to zero.
        B = basis_matrix(basis, k, [SVector(1e-20, 0.3)])
        @test all(iszero, B)
        @test any(!iszero, basis_fun(basis, 1:basis.dim, k, [SVector(0.1, 0.3)]))
    end

    @testset "symmetry-reduced basis" begin
        basis = RealPlaneWaves(4; sym_x=-1)
        @test basis_matrix(basis, k, pts) ≈ rpw_ref_matrix(basis, k, pts) atol = 1e-13
        @test size(basis_matrix(basis, k, pts)) == (6, 8)
    end
end

@testset "basis_fun - reflection parity of symmetry sectors" begin
    k = 3.1
    pts = rpw_pts()

    @testset "even-even sector (sym_x = 1, sym_y = 1)" begin
        @test rpw_reflection_parity(RealPlaneWaves(5; sym_x=1, sym_y=1), k, pts, 1, 1)
    end

    @testset "odd-even sector (sym_x = -1, sym_y = 1)" begin
        @test rpw_reflection_parity(RealPlaneWaves(5; sym_x=-1, sym_y=1), k, pts, -1, 1)
    end

    @testset "even-odd sector (sym_x = 1, sym_y = -1)" begin
        @test rpw_reflection_parity(RealPlaneWaves(5; sym_x=1, sym_y=-1), k, pts, 1, -1)
    end

    @testset "odd-odd sector (sym_x = -1, sym_y = -1)" begin
        @test rpw_reflection_parity(RealPlaneWaves(5; sym_x=-1, sym_y=-1), k, pts, -1, -1)
    end

    @testset "y-axis reflection only: x parity fixed, y parity free" begin
        even = RealPlaneWaves(5; sym_x=1)
        odd = RealPlaneWaves(5; sym_x=-1)
        @test rpw_reflection_parity(even, k, pts, 1, 0)
        @test rpw_reflection_parity(odd, k, pts, -1, 0)
        @test !rpw_reflection_parity(even, k, pts, 0, 1)                   # y parity is not fixed
        # An odd-in-x function vanishes on the symmetry axis x = 0.
        @test all(iszero, basis_fun(odd, 1:odd.dim, k, [SVector(0.0, 0.2), SVector(0.0, -0.7)]))
    end

    @testset "x-axis reflection only: y parity fixed, x parity free" begin
        even = RealPlaneWaves(5; sym_y=1)
        odd = RealPlaneWaves(5; sym_y=-1)
        @test rpw_reflection_parity(even, k, pts, 0, 1)
        @test rpw_reflection_parity(odd, k, pts, 0, -1)
        @test !rpw_reflection_parity(even, k, pts, 1, 0)
        @test all(iszero, basis_fun(odd, 1:odd.dim, k, [SVector(0.2, 0.0), SVector(-0.7, 0.0)]))
    end

    @testset "unrestricted basis: each column has the parity its selectors promise" begin
        basis = RealPlaneWaves(4)
        B = basis_fun(basis, 1:basis.dim, k, pts)
        Bx = basis_fun(basis, 1:basis.dim, k, [SVector(-p[1], p[2]) for p in pts])
        By = basis_fun(basis, 1:basis.dim, k, [SVector(p[1], -p[2]) for p in pts])
        @test Bx ≈ B .* basis.parity_x' atol = 1e-13
        @test By ≈ B .* basis.parity_y' atol = 1e-13
    end

    @testset "sector-built basis has the sector's parity" begin
        rect = RectangleBilliard(1.0, 0.6)
        basis = RealPlaneWaves(5, rect, symmetry_sector(rect, XAxisReflection => -1, YAxisReflection => 1))
        @test rpw_reflection_parity(basis, k, pts, 1, -1)
    end
end

@testset "basis_fun - Helmholtz equation" begin
    k = 3.0
    pts = rpw_pts()
    @testset "cos-cos column" begin
        basis = RealPlaneWaves(3; sym_x=1, sym_y=1)
        @test maximum(abs, rpw_helmholtz_residual(basis, 2, k, pts)) < 1e-4
    end
    @testset "sin-sin column" begin
        basis = RealPlaneWaves(3; sym_x=-1, sym_y=-1)
        @test maximum(abs, rpw_helmholtz_residual(basis, 2, k, pts)) < 1e-4
    end
    @testset "mixed columns of the unrestricted basis" begin
        basis = RealPlaneWaves(2)
        @test all(maximum(abs, rpw_helmholtz_residual(basis, i, k, pts)) < 1e-4 for i in 1:basis.dim)
    end
    @testset "the residual is not trivially small (control: wrong wavenumber)" begin
        basis = RealPlaneWaves(3; sym_x=1, sym_y=1)
        f = basis_fun(basis, 2, k, pts)
        h = 1e-3
        wrong = (basis_fun(basis, 2, k, rpw_shift(pts, h, 0.0)) .+ basis_fun(basis, 2, k, rpw_shift(pts, -h, 0.0)) .+
                 basis_fun(basis, 2, k, rpw_shift(pts, 0.0, h)) .+ basis_fun(basis, 2, k, rpw_shift(pts, 0.0, -h)) .- 4 .* f) ./ h^2 .+ (2k)^2 .* f
        @test maximum(abs, wrong) > 1
    end
end

@testset "gradient - analytic and finite differences" begin
    k = 3.3
    pts = rpw_pts()

    @testset "analytic form for a sin-cos column" begin
        basis = RealPlaneWaves(3; sym_x=-1, sym_y=1)
        i = 2
        vx, vy = cos(basis.angles[i]), sin(basis.angles[i])
        dx, dy = gradient(basis, i, k, pts)
        @test dx ≈ [k * vx * cos(k * vx * p[1]) * cos(k * vy * p[2]) for p in pts] atol = 1e-13
        @test dy ≈ [-k * vy * sin(k * vx * p[1]) * sin(k * vy * p[2]) for p in pts] atol = 1e-13
    end

    @testset "analytic form for a cos-sin column" begin
        basis = RealPlaneWaves(3; sym_x=1, sym_y=-1)
        i = 3
        vx, vy = cos(basis.angles[i]), sin(basis.angles[i])
        dx, dy = gradient(basis, i, k, pts)
        @test dx ≈ [-k * vx * sin(k * vx * p[1]) * sin(k * vy * p[2]) for p in pts] atol = 1e-13
        @test dy ≈ [k * vy * cos(k * vx * p[1]) * cos(k * vy * p[2]) for p in pts] atol = 1e-13
    end

    @testset "central finite differences, unrestricted basis" begin
        basis = RealPlaneWaves(3)
        for i in 1:basis.dim
            dx, dy = gradient(basis, i, k, pts)
            @test dx ≈ rpw_fd_x(basis, i, k, pts) atol = 1e-6
            @test dy ≈ rpw_fd_y(basis, i, k, pts) atol = 1e-6
        end
    end

    @testset "matrix method equals the single-index method" begin
        basis = RealPlaneWaves(3)
        dBdx, dBdy = gradient(basis, 1:basis.dim, k, pts)
        @test size(dBdx) == size(dBdy) == (length(pts), basis.dim)
        for i in (1, 6, 12)
            dx, dy = gradient(basis, i, k, pts)
            @test dBdx[:, i] == dx
            @test dBdy[:, i] == dy
        end
        sx, sy = gradient(basis, [4, 2], k, pts)
        @test sx == dBdx[:, [4, 2]] && sy == dBdy[:, [4, 2]]
    end

    @testset "gradient_matrices applies the filter" begin
        basis = RealPlaneWaves(3)
        dBdx, dBdy = gradient(basis, 1:basis.dim, k, pts)
        gx, gy = gradient_matrices(basis, k, pts)
        @test gx == filter_matrix!(copy(dBdx))
        @test gy == filter_matrix!(copy(dBdy))
    end

    @testset "gradient norm bound" begin
        # With F = cos or sin, |∇f|² <= k² (vx² + vy²) = k² pointwise.
        basis = RealPlaneWaves(4)
        dBdx, dBdy = gradient(basis, 1:basis.dim, k, pts)
        @test all(dBdx .^ 2 .+ dBdy .^ 2 .<= k^2 + 1e-12)
    end
end

@testset "basis_and_gradient" begin
    k = 3.3
    pts = rpw_pts()
    basis = RealPlaneWaves(3)

    @testset "single index agrees with basis_fun and gradient" begin
        for i in (1, 5, 12)
            bf, dx, dy = basis_and_gradient(basis, i, k, pts)
            gx, gy = gradient(basis, i, k, pts)
            @test bf == basis_fun(basis, i, k, pts)
            @test dx == gx && dy == gy
        end
    end

    @testset "index set agrees with basis_fun and gradient" begin
        B, dBdx, dBdy = basis_and_gradient(basis, 1:basis.dim, k, pts)
        gx, gy = gradient(basis, 1:basis.dim, k, pts)
        @test B == basis_fun(basis, 1:basis.dim, k, pts)
        @test dBdx == gx && dBdy == gy
        B2, _, _ = basis_and_gradient(basis, [3, 1], k, pts)
        @test B2 == B[:, [3, 1]]
    end

    @testset "basis_and_gradient_matrices applies the filter" begin
        B, dBdx, dBdy = basis_and_gradient(basis, 1:basis.dim, k, pts)
        fB, fdx, fdy = basis_and_gradient_matrices(basis, k, pts)
        @test fB == filter_matrix!(copy(B)) == basis_matrix(basis, k, pts)
        @test fdx == filter_matrix!(copy(dBdx))
        @test fdy == filter_matrix!(copy(dBdy))
    end
end

@testset "dk_fun - finite differences and homogeneity" begin
    k = 3.3
    pts = rpw_pts()

    @testset "central finite differences in k, unrestricted basis" begin
        basis = RealPlaneWaves(3)
        for i in 1:basis.dim
            @test dk_fun(basis, i, k, pts) ≈ rpw_fd_k(basis, i, k, pts) atol = 1e-6
        end
    end

    @testset "analytic form for a sin-sin column" begin
        basis = RealPlaneWaves(3; sym_x=-1, sym_y=-1)
        i = 2
        vx, vy = cos(basis.angles[i]), sin(basis.angles[i])
        expected = [vx * p[1] * cos(k * vx * p[1]) * sin(k * vy * p[2]) + vy * p[2] * sin(k * vx * p[1]) * cos(k * vy * p[2]) for p in pts]
        @test dk_fun(basis, i, k, pts) ≈ expected atol = 1e-13
    end

    @testset "k ∂f/∂k = x ∂f/∂x + y ∂f/∂y" begin
        basis = RealPlaneWaves(3)
        dBdk = dk_fun(basis, 1:basis.dim, k, pts)
        dBdx, dBdy = gradient(basis, 1:basis.dim, k, pts)
        X = [p[1] for p in pts]
        Y = [p[2] for p in pts]
        @test k .* dBdk ≈ X .* dBdx .+ Y .* dBdy atol = 1e-12
    end

    @testset "matrix method equals the single-index method" begin
        basis = RealPlaneWaves(3)
        D = dk_fun(basis, 1:basis.dim, k, pts)
        @test size(D) == (length(pts), basis.dim)
        @test D[:, 1] == dk_fun(basis, 1, k, pts)
        @test D[:, 9] == dk_fun(basis, 9, k, pts)
        @test dk_fun(basis, [5, 2], k, pts) == D[:, [5, 2]]
    end

    @testset "dk_matrix applies the filter" begin
        basis = RealPlaneWaves(3)
        D = dk_fun(basis, 1:basis.dim, k, pts)
        @test dk_matrix(basis, k, pts) == filter_matrix!(copy(D))
    end

    @testset "a rescaled geometry is a rescaled wavenumber" begin
        # f(k; p) = g(k p), so ∂f/∂k (k; p) = p · g'(k p) and ∂f/∂k' (c k; p / c) = (p / c) · g'(k p): the ratio is c.
        basis = RealPlaneWaves(3)
        c = 2.0
        half = [p ./ c for p in pts]
        @test dk_fun(basis, 1:basis.dim, k, pts) ≈ c .* dk_fun(basis, 1:basis.dim, k * c, half) atol = 1e-12
    end
end

@testset "RealPlaneWaves - threaded equals serial" begin
    k = 3.3
    pts = rpw_pts()
    basis = RealPlaneWaves(5)
    idx = 1:basis.dim

    @test basis_fun(basis, idx, k, pts; multithreaded=true) == basis_fun(basis, idx, k, pts; multithreaded=false)
    @test basis_matrix(basis, k, pts; multithreaded=true) == basis_matrix(basis, k, pts; multithreaded=false)
    @test dk_fun(basis, idx, k, pts; multithreaded=true) == dk_fun(basis, idx, k, pts; multithreaded=false)
    @test dk_matrix(basis, k, pts; multithreaded=true) == dk_matrix(basis, k, pts; multithreaded=false)
    @test gradient(basis, idx, k, pts; multithreaded=true) == gradient(basis, idx, k, pts; multithreaded=false)
    @test basis_and_gradient(basis, idx, k, pts; multithreaded=true) == basis_and_gradient(basis, idx, k, pts; multithreaded=false)
    @test basis_and_gradient_matrices(basis, k, pts; multithreaded=true) == basis_and_gradient_matrices(basis, k, pts; multithreaded=false)
end

@testset "RealPlaneWaves - type stability of the hot paths" begin
    k = 3.3
    pts = rpw_pts()
    basis = RealPlaneWaves(3)
    idx = 1:basis.dim

    @test @inferred(basis_fun(basis, 2, k, pts)) isa Vector{Float64}
    @test @inferred(basis_fun(basis, idx, k, pts)) isa Matrix{Float64}
    @test @inferred(basis_fun(basis, idx, k, pts; multithreaded=false)) isa Matrix{Float64}
    @test @inferred(gradient(basis, 2, k, pts)) isa Tuple{Vector{Float64},Vector{Float64}}
    @test @inferred(gradient(basis, idx, k, pts)) isa Tuple{Matrix{Float64},Matrix{Float64}}
    @test @inferred(basis_and_gradient(basis, 2, k, pts)) isa Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    @test @inferred(basis_and_gradient(basis, idx, k, pts)) isa Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    @test @inferred(dk_fun(basis, 2, k, pts)) isa Vector{Float64}
    @test @inferred(dk_fun(basis, idx, k, pts)) isa Matrix{Float64}
    @test @inferred(basis_matrix(basis, k, pts)) isa Matrix{Float64}
    @test @inferred(dk_matrix(basis, k, pts)) isa Matrix{Float64}
    @test @inferred(gradient_matrices(basis, k, pts)) isa Tuple{Matrix{Float64},Matrix{Float64}}
    @test @inferred(basis_and_gradient_matrices(basis, k, pts)) isa Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64}}
end

@testset "RealPlaneWaves - Float32" begin
    b32 = RealPlaneWaves(3; sym_x=1, sym_y=-1, sampler=LinearNodes(Float32), angle_arc=Float32(π / 2), angle_shift=0.0f0,
                         taylor_radius=0.5f0, taylor_tol=1.0f-6)
    b64 = RealPlaneWaves(3; sym_x=1, sym_y=-1)
    pts32 = rpw_pts(Float32)
    pts64 = rpw_pts(Float64)
    k32 = 3.3f0
    idx = 1:b32.dim

    @testset "construction promotes every field to Float32" begin
        @test b32 isa RealPlaneWaves{Float32,LinearNodes{Float32}}
        @test eltype(b32.angles) == Float32
        @test b32.angle_arc isa Float32 && b32.angle_shift isa Float32
        @test b32.taylor_radius isa Float32 && b32.taylor_tol isa Float32
        @test eltype(b32.sym_qnumbers) == Float32
        @test b32.angles ≈ Float32.(b64.angles)
        @test b32.parity_x == b64.parity_x && b32.parity_y == b64.parity_y
    end

    @testset "evaluators return Float32" begin
        @test @inferred(basis_fun(b32, 1, k32, pts32)) isa Vector{Float32}
        @test @inferred(basis_fun(b32, idx, k32, pts32)) isa Matrix{Float32}
        @test @inferred(gradient(b32, idx, k32, pts32)) isa Tuple{Matrix{Float32},Matrix{Float32}}
        @test @inferred(basis_and_gradient(b32, idx, k32, pts32)) isa Tuple{Matrix{Float32},Matrix{Float32},Matrix{Float32}}
        @test @inferred(dk_fun(b32, idx, k32, pts32)) isa Matrix{Float32}
        @test @inferred(basis_matrix(b32, k32, pts32)) isa Matrix{Float32}
        @test @inferred(dk_matrix(b32, k32, pts32)) isa Matrix{Float32}
    end

    @testset "agrees with Float64 to single precision" begin
        @test basis_fun(b32, idx, k32, pts32) ≈ basis_fun(b64, idx, 3.3, pts64) atol = 1e-4
        g32x, g32y = gradient(b32, idx, k32, pts32)
        g64x, g64y = gradient(b64, idx, 3.3, pts64)
        @test g32x ≈ g64x atol = 1e-3
        @test g32y ≈ g64y atol = 1e-3
        @test dk_fun(b32, idx, k32, pts32) ≈ dk_fun(b64, idx, 3.3, pts64) atol = 1e-3
    end

    @testset "resize_basis keeps Float32" begin
        r = resize_basis(b32, RectangleBilliard(1.0, 0.6), 5, 1.0)
        @test r isa RealPlaneWaves{Float32,LinearNodes{Float32}}
        @test r.dim == 5
        @test eltype(r.angles) == Float32
    end
end
