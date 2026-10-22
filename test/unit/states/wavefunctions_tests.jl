# Unit tests for
#   src/states/wavefunctions.jl   (grid helpers pad_limits, _debroglie_grid_size, rectify_grid, boundary_limits,
#                                  _is_inside_full, _wavefunction_grid, _mask_fill, _bim_kernel;
#                                  the layer potentials phi_slp / phi_dlp / phi_cfie;
#                                  compute_psi and wavefunction for BasisEigenstate, BasisState, BIMEigenstate;
#                                  the SpectralData wrappers wavefunction(data) and wavefunction(data, i))
# regularize! (boundaryfunctions.jl) and antisym_vec (husimifunctions.jl) are not defined in this file and are covered
# with their own source files.
#
# All references are analytic; no eigen solves and no pinned values.
#
#  * Rectangle RectangleBilliard(1.0, 0.6) is the full 2a x 2b rectangle, a = 1, b = 0.6, with first-quadrant fundamental
#    domain. A single real plane wave cos/sin(kx x) cos/sin(ky y) with kx = m pi/(2a), ky = n pi/(2b) is an exact Dirichlet
#    eigenfunction (cos for odd m, n; sin for even m, n). It is built as a one-element RealPlaneWaves basis whose only
#    angle is atan(ky, kx) (angle_arc = 0, angle_shift = theta), the symmetry sector is fixed by the parities of m and n,
#    and the state has coefficient vector [1]. Reflection unfolding must reproduce the analytic mode on the full
#    rectangle, and with inside_only = false it must reproduce it on the whole grid box.
#  * Unit circle: J0(j01 r) with j01 the first zero of J0 is the lowest Dirichlet mode. It is built from a
#    FourierBessel basis (the j = 0 cosine term) with coefficient vector [1], both with and without reflection symmetries.
#  * Layer potentials on a hand-built circle discretization (48 equispaced points, outward normals, unit tangents,
#    weights 2 pi / 48). By Graf's addition theorem, for |x| = r < 1 and k = j01 (a zero of J0):
#        single layer, u = d_n psi = -k J1(k) (constant):  1/4 sum Y0(k|x-q|) u ds = -J0(k r)
#        double layer, mu = 1:                              -(i pi k / 2) H1(k) J0(k r)
#        combined field, mu = 1:                            pi k J0(k r) (H0(k) + i H1(k))
#    The discrete sums are trapezoid sums of a periodic analytic integrand; the aliasing error is of order r^48 and
#    negligible for the test radii r <= 0.55.
# BIM wavefunctions use test-local solver stubs that only provide the type the reconstruction dispatches on (DLP, CFIE or
# a generic SweepBIMSolver).
# Helper definitions use the prefix `wf_` / `WF_`.

using StaticArrays
using SpecialFunctions: besselj, hankelh1
using QuantumBilliards: pad_limits, rectify_grid, boundary_limits, _debroglie_grid_size, _is_inside_full,
                        _wavefunction_grid, _mask_fill, _bim_kernel, ϕ_slp, ϕ_dlp, ϕ_cfie, compute_psi

const WF_J01 = 2.404825557695773      # first zero of J0
const WF_J02 = 5.520078110286311      # second zero of J0
const WF_A, WF_B = 1.0, 0.6
const WF_RECT = RectangleBilliard(WF_A, WF_B)
const WF_CIRCLE = CircleBilliard(1.0)
const WF_NODES = 48

# Solver stubs: the reconstruction only dispatches on the solver type.
struct wf_SweepStub <: QuantumBilliards.SweepBIMSolver end
struct wf_DLPStub <: QuantumBilliards.DLP end
struct wf_CFIEStub <: QuantumBilliards.CFIE end
struct wf_AccStub <: QuantumBilliards.AcceleratedBIMSolver
    kernel::wf_DLPStub
end

# Unit circle discretization: points, outward normals, unit tangents, arc-length and parametrization weights.
function wf_circle_pts(N::Int = WF_NODES)
    th = range(0, 2pi, N + 1)[1:N]
    w = fill(2pi / N, N)
    return BoundaryPoints([SVector(cos(t), sin(t)) for t in th]; normal=[SVector(cos(t), sin(t)) for t in th],
                          tangent=[SVector(-sin(t), cos(t)) for t in th], ds=w, ws=w)
end
const WF_PTS = wf_circle_pts()

# State of the rectangle mode (m, n) together with its analytic function.
function wf_rect_mode(m::Int, n::Int)
    kx = m * pi / (2WF_A); ky = n * pi / (2WF_B)
    sx = isodd(m) ? 1 : -1; sy = isodd(n) ? 1 : -1
    basis = RealPlaneWaves(1; sym_x=sx, sym_y=sy, angle_arc=0.0, angle_shift=atan(ky, kx))
    fx = isodd(m) ? cos : sin; fy = isodd(n) ? cos : sin
    return BasisEigenstate(hypot(kx, ky), [1.0], 1e-12, basis), (x, y) -> fx(kx * x) * fy(ky * y)
end

wf_rect_interior(x, y) = [abs(xx) < WF_A - 1e-9 && abs(yy) < WF_B - 1e-9 for xx in x, yy in y]
wf_rect_exterior(x, y) = [abs(xx) > WF_A + 1e-9 || abs(yy) > WF_B + 1e-9 for xx in x, yy in y]
wf_analytic(f, x, y) = [f(xx, yy) for xx in x, yy in y]
wf_radius(x, y) = [hypot(xx, yy) for xx in x, yy in y]
wf_j0(x, y, k = WF_J01) = besselj.(0, k .* wf_radius(x, y))

# Unit-disc masks away from the boundary, in the first quadrant only or in the whole plane.
wf_disc_interior(x, y, r) = [hypot(xx, yy) < r for xx in x, yy in y]
wf_quadrant_interior(x, y, r) = [xx > 0 && yy > 0 && hypot(xx, yy) < r for xx in x, yy in y]
wf_disc_exterior(x, y) = [hypot(xx, yy) > 1 + 1e-9 for xx in x, yy in y]

# BIM state of the J0 mode: constant physical normal derivative u = -k J1(k) (partial state, built with the type-parameter
# constructor because the keyword constructor cannot infer K for vec === nothing, see eigenstates_tests.jl).
wf_bim_u_state(k) = BIMEigenstate{Float64,Float64}(k, 0.0, nothing, 1e-9, WF_PTS, fill(-k * besselj(1, k), WF_NODES), nothing)
wf_bim_density_state(k) = BIMEigenstate(k, ones(WF_NODES), 1e-9, WF_PTS)

@testset "pad_limits" begin
    @test pad_limits((0.0, 1.0), (-2.0, 3.0)) == ((-0.01, 1.01), (-2.01, 3.01))
    @test pad_limits((0.0, 1.0), (-2.0, 3.0); padding=0.5) == ((-0.5, 1.5), (-2.5, 3.5))
    @test pad_limits((0.0, 1.0), (-2.0, 3.0); padding=0.0) == ((0.0, 1.0), (-2.0, 3.0))
    xl, yl = pad_limits(extrema([2.0, -1.0, 4.0]), extrema([0.5, 0.25]); padding=0.1)     # extrema returns tuples
    @test xl[1] ≈ -1.1 && xl[2] ≈ 4.1
    @test yl[1] ≈ 0.15 && yl[2] ≈ 0.6
    @test @inferred(pad_limits((0.0, 1.0), (0.0, 2.0))) isa Tuple{Tuple{Float64,Float64},Tuple{Float64,Float64}}
end

@testset "_debroglie_grid_size" begin
    @test _debroglie_grid_size(2pi, 1000.0, 1.0) == 1000                        # round(k d b / 2 pi)
    @test _debroglie_grid_size(2pi, 1000.0, 2.0) == 2000
    @test _debroglie_grid_size(pi, 1000.0, 1.0) == 512                         # round(...) = 500, floored at 512
    @test _debroglie_grid_size(3.0, 4.0, 5.0) == 512                           # small systems get the minimum
    @test _debroglie_grid_size(2pi, 513.0, 1.0) == 513
    @test _debroglie_grid_size(1.0, 1.0, 1.0) isa Int
end

@testset "rectify_grid" begin
    @testset "grid containing 0 as a node: positive nodes only" begin
        @test rectify_grid(collect(range(-0.5, 1.0, 7))) ≈ [0.25, 0.5, 0.75, 1.0]
    end

    @testset "grid straddling 0 without a node at 0 is shifted so the nearest node becomes 0" begin
        grid = [-0.3, -0.1, 0.15, 0.4, 0.65]     # nearest node to zero is -0.1, nodes are 0.25 apart
        r = rectify_grid(grid)
        @test r ≈ [0.25, 0.5, 0.75]
        @test all(>(0), r)
        @test all(diff(r) .≈ 0.25)
    end

    @testset "step is preserved, first element is one step" begin
        grid = collect(range(-0.013, 1.021, 301))
        r = rectify_grid(grid)
        step = grid[2] - grid[1]
        @test r[1] ≈ step
        @test all(diff(r) .≈ step)
        @test all(>(0), r)
        @test r[end] <= grid[end] - grid[argmin(abs.(grid))] + 1e-12
    end

    @testset "grid not containing 0 is returned unchanged" begin
        pos = [0.5, 1.0, 1.5]
        neg = [-1.5, -1.0, -0.5]
        @test rectify_grid(pos) === pos
        @test rectify_grid(neg) === neg
    end

    @testset "element type is preserved" begin
        r32 = rectify_grid(Float32[-0.3, -0.1, 0.15, 0.4, 0.65])
        @test r32 isa Vector{Float32}
        @test r32 ≈ Float32[0.25, 0.5, 0.75]
        @test @inferred(rectify_grid([-1.0, 0.0, 1.0])) == [1.0]
    end
end

@testset "boundary_limits" begin
    @testset "rectangle fundamental domain" begin
        # get_boundary_curves of the fundamental domain contains only the physical walls (no symmetry walls) and the limits are
        # taken from sampled points, so a lower limit lying on a symmetry wall is only known up to one sample spacing; the
        # padding of 0.01 still covers the true extent.
        xl, yl = boundary_limits(get_boundary_curves(WF_RECT))
        @test -0.01 - 1e-10 <= xl[1] <= 0.0
        @test xl[2] ≈ WF_A + 0.01 atol = 1e-10
        @test collect(yl) ≈ [-0.01, WF_B + 0.01] atol = 1e-10
    end

    @testset "rectangle full boundary" begin
        xl, yl = boundary_limits(full_boundary(WF_RECT))
        @test collect(xl) ≈ [-WF_A - 0.01, WF_A + 0.01] atol = 1e-10
        @test collect(yl) ≈ [-WF_B - 0.01, WF_B + 0.01] atol = 1e-10
    end

    @testset "circle fundamental domain and full boundary" begin
        xl, yl = boundary_limits(get_boundary_curves(WF_CIRCLE))
        @test -0.01 - 1e-10 <= xl[1] <= 0.0          # arc samples only; see the rectangle note above
        @test xl[2] ≈ 1.01 atol = 1e-6
        @test yl[1] ≈ -0.01 atol = 1e-6
        @test yl[2] > 1.0
        @test yl[2] ≈ 1.01 atol = 1e-3
        xf, yf = boundary_limits(full_boundary(WF_CIRCLE))
        @test collect(xf) ≈ [-1.01, 1.01] atol = 1e-6
        @test collect(yf) ≈ [-1.01, 1.01] atol = 1e-6
    end

    @testset "padding keyword" begin
        xl, yl = boundary_limits(full_boundary(WF_RECT); padding=0.25)
        @test collect(xl) ≈ [-WF_A - 0.25, WF_A + 0.25] atol = 1e-10
        @test collect(yl) ≈ [-WF_B - 0.25, WF_B + 0.25] atol = 1e-10
        xz, _ = boundary_limits(full_boundary(WF_RECT); padding=0.0)
        @test collect(xz) ≈ [-WF_A, WF_A] atol = 1e-10
    end
end

@testset "_is_inside_full" begin
    @testset "rectangle: all four quadrants are inside, the fundamental domain is only the first" begin
        for pt in (SVector(0.5, 0.3), SVector(-0.5, 0.3), SVector(-0.5, -0.3), SVector(0.5, -0.3))
            @test _is_inside_full(WF_RECT, pt)
        end
        @test BilliardGeometry.is_inside(WF_RECT, SVector(0.5, 0.3))
        @test !BilliardGeometry.is_inside(WF_RECT, SVector(-0.5, -0.3))
        @test !_is_inside_full(WF_RECT, SVector(1.2, 0.0))
        @test !_is_inside_full(WF_RECT, SVector(-1.2, 0.0))
        @test !_is_inside_full(WF_RECT, SVector(0.0, 0.7))
        @test !_is_inside_full(WF_RECT, SVector(0.0, -0.7))
    end

    @testset "circle" begin
        @test _is_inside_full(WF_CIRCLE, SVector(-0.6, -0.6))
        @test _is_inside_full(WF_CIRCLE, SVector(0.6, -0.6))
        @test !_is_inside_full(WF_CIRCLE, SVector(-0.8, -0.8))
        @test !_is_inside_full(WF_CIRCLE, SVector(0.8, 0.8))
        @test _is_inside_full(WF_CIRCLE, SVector(0.0, 0.0))
    end
end

@testset "_mask_fill" begin
    @test _mask_fill(Float64, true) |> isnan
    @test _mask_fill(Float64, false) === 0.0
    @test _mask_fill(Float32, false) === 0.0f0
    @test _mask_fill(Float32, true) |> isnan
    z = _mask_fill(ComplexF64, true)
    @test isnan(real(z)) && isnan(imag(z))
    @test _mask_fill(ComplexF64, false) === 0.0 + 0.0im
end

@testset "_bim_kernel" begin
    kern = wf_DLPStub()
    @test _bim_kernel(kern) === kern
    @test _bim_kernel(wf_AccStub(kern)) === kern
end

@testset "_wavefunction_grid" begin
    b = 5.0
    @testset "fundamental domain of the rectangle" begin
        xg, yg, pts, idx, nx, ny = _wavefunction_grid(1.0, WF_RECT, b)
        @test (nx, ny) == (512, 512)
        @test length(xg) == nx && length(yg) == ny
        @test -0.01 - 1e-10 <= xg[1] <= 0.0                           # lower x limit: see boundary_limits
        @test xg[end] ≈ WF_A + 0.01
        @test yg[1] ≈ -0.01 && yg[end] ≈ WF_B + 0.01
        @test length(pts) == nx * ny
        @test pts[2] == SVector(xg[2], yg[1])                  # x varies fastest
        @test pts[nx + 1] == SVector(xg[1], yg[2])
        @test all(BilliardGeometry.is_inside(WF_RECT, pts[i]) for i in idx[1:100:end])
        dA = (xg[2] - xg[1]) * (yg[2] - yg[1])
        @test length(idx) * dA ≈ WF_A * WF_B rtol = 0.02
    end

    @testset "full domain of the rectangle" begin
        xg, yg, pts, idx, nx, ny = _wavefunction_grid(1.0, WF_RECT, b; fundamental_domain=false)
        @test xg[1] ≈ -WF_A - 0.01 && xg[end] ≈ WF_A + 0.01
        @test yg[1] ≈ -WF_B - 0.01 && yg[end] ≈ WF_B + 0.01
        @test all(abs(pts[i][1]) < WF_A && abs(pts[i][2]) < WF_B for i in idx)
        dA = (xg[2] - xg[1]) * (yg[2] - yg[1])
        @test length(idx) * dA ≈ 4 * WF_A * WF_B rtol = 0.02
    end

    @testset "inside_only = false keeps every grid point" begin
        for fd in (true, false)
            xg, yg, pts, idx, nx, ny = _wavefunction_grid(1.0, WF_RECT, b; inside_only=false, fundamental_domain=fd)
            @test idx == 1:(nx * ny)
        end
    end

    @testset "at least b points per wavelength, 512 points at minimum" begin
        k = 1000.0
        xg, yg, pts, idx, nx, ny = _wavefunction_grid(k, WF_RECT, b; inside_only=false)
        @test nx > 512
        @test xg[2] - xg[1] <= 2pi / (b * k)
        @test yg[2] - yg[1] <= 2pi / (b * k)
        @test eltype(xg) == Float64 && eltype(pts) == SVector{2,Float64}
    end

    @testset "circle: area of the masked region" begin
        xg, yg, _, idx, _, _ = _wavefunction_grid(1.0, WF_CIRCLE, b)
        @test length(idx) * (xg[2] - xg[1]) * (yg[2] - yg[1]) ≈ pi / 4 rtol = 0.02
        xg, yg, _, idx, _, _ = _wavefunction_grid(1.0, WF_CIRCLE, b; fundamental_domain=false)
        @test length(idx) * (xg[2] - xg[1]) * (yg[2] - yg[1]) ≈ pi rtol = 0.02
    end
end

@testset "layer potentials phi_slp, phi_dlp, phi_cfie on the circle" begin
    k = WF_J01
    u = fill(-k * besselj(1, k), WF_NODES)
    mu = ones(WF_NODES)
    test_points = ((0.3, 0.2), (0.0, 0.0), (0.5, 0.1), (-0.4, 0.35))

    @testset "single layer reproduces -J0(k r) from the normal derivative" begin
        for (x, y) in test_points
            @test ϕ_slp(x, y, k, WF_PTS, u, nothing, Val(:direct)) ≈ -besselj(0, k * hypot(x, y)) atol = 1e-10
        end
        @test @inferred(ϕ_slp(0.3, 0.2, k, WF_PTS, u, nothing, Val(:direct))) isa Float64
    end

    @testset "double layer, constant density" begin
        for (x, y) in test_points
            expected = -(im * pi * k / 2) * hankelh1(1, k) * besselj(0, k * hypot(x, y))
            @test ϕ_dlp(x, y, k, WF_PTS, mu, nothing, Val(:direct)) ≈ expected atol = 1e-10
        end
        @test @inferred(ϕ_dlp(0.3, 0.2, k, WF_PTS, mu, nothing, Val(:direct))) isa ComplexF64
    end

    @testset "combined field, constant density" begin
        for (x, y) in test_points
            expected = pi * k * besselj(0, k * hypot(x, y)) * (hankelh1(0, k) + im * hankelh1(1, k))
            @test ϕ_cfie(x, y, k, WF_PTS, mu, nothing, Val(:direct)) ≈ expected atol = 1e-10
        end
        @test @inferred(ϕ_cfie(0.3, 0.2, k, WF_PTS, mu, nothing, Val(:direct))) isa ComplexF64
    end

    @testset "linearity in the density" begin
        @test ϕ_slp(0.3, 0.2, k, WF_PTS, 3 .* u, nothing, Val(:direct)) ≈ 3 * ϕ_slp(0.3, 0.2, k, WF_PTS, u, nothing, Val(:direct))
        @test ϕ_dlp(0.3, 0.2, k, WF_PTS, 2 .* mu, nothing, Val(:direct)) ≈ 2 * ϕ_dlp(0.3, 0.2, k, WF_PTS, mu, nothing, Val(:direct))
        @test ϕ_cfie(0.3, 0.2, k, WF_PTS, 2 .* mu, nothing, Val(:direct)) ≈ 2 * ϕ_cfie(0.3, 0.2, k, WF_PTS, mu, nothing, Val(:direct))
        @test ϕ_slp(0.3, 0.2, k, WF_PTS, zero(u), nothing, Val(:direct)) == 0
    end

    @testset "evaluation at a boundary node skips the singular term" begin
        p = WF_PTS.xy[1]
        @test isfinite(ϕ_slp(p[1], p[2], k, WF_PTS, u, nothing, Val(:direct)))
        @test isfinite(abs(ϕ_dlp(p[1], p[2], k, WF_PTS, mu, nothing, Val(:direct))))
        @test isfinite(abs(ϕ_cfie(p[1], p[2], k, WF_PTS, mu, nothing, Val(:direct))))
    end
end

@testset "compute_psi - rectangle mode on explicit grids" begin
    st, f = wf_rect_mode(1, 1)
    xg = collect(range(-1.2, 1.2, 61)); yg = collect(range(-0.8, 0.8, 41))

    @testset "flattened layout: x varies fastest, values equal the analytic mode" begin
        psi = compute_psi(st, WF_RECT, xg, yg; inside_only=false)
        @test psi isa Vector{Float64}
        @test length(psi) == length(xg) * length(yg)
        @test reshape(psi, length(xg), length(yg)) ≈ wf_analytic(f, xg, yg) atol = 1e-12
    end

    @testset "inside_only masks the fundamental domain with zeros" begin
        psi = reshape(compute_psi(st, WF_RECT, xg, yg), length(xg), length(yg))
        inside = [xx > 1e-9 && yy > 1e-9 && xx < WF_A - 1e-9 && yy < WF_B - 1e-9 for xx in xg, yy in yg]
        outside = [xx < -1e-9 || yy < -1e-9 || xx > WF_A + 1e-9 || yy > WF_B + 1e-9 for xx in xg, yy in yg]
        @test count(inside) > 100
        @test psi[inside] ≈ wf_analytic(f, xg, yg)[inside] atol = 1e-12
        @test all(psi[outside] .== 0)
    end

    @testset "use_NaN_mask fills the excluded points with NaN" begin
        psi = reshape(compute_psi(st, WF_RECT, xg, yg; use_NaN_mask=true), length(xg), length(yg))
        outside = [xx < -1e-9 || yy < -1e-9 || xx > WF_A + 1e-9 || yy > WF_B + 1e-9 for xx in xg, yy in yg]
        @test all(isnan, psi[outside])
        @test !any(isnan, psi[.!outside])
        # no mask requested, use_NaN_mask has no effect
        psi_all = compute_psi(st, WF_RECT, xg, yg; inside_only=false, use_NaN_mask=true)
        @test !any(isnan, psi_all)
    end

    @testset "memory_limit and multithreaded do not change the result" begin
        ref = compute_psi(st, WF_RECT, xg, yg; inside_only=false)
        @test compute_psi(st, WF_RECT, xg, yg; inside_only=false, memory_limit=80.0) == ref      # chunks of 8 points
        @test compute_psi(st, WF_RECT, xg, yg; inside_only=false, multithreaded=false) == ref
    end

    @testset "coefficients are linear: vec scales the field" begin
        st2 = BasisEigenstate(st.k, [-2.5], st.ten, st.basis)
        @test compute_psi(st2, WF_RECT, xg, yg; inside_only=false) ≈ -2.5 .* compute_psi(st, WF_RECT, xg, yg; inside_only=false)
    end

    @testset "k_basis, not k, selects the wavenumber the basis is evaluated at" begin
        st_shift = BasisEigenstate(st.k + 0.3, st.k, st.vec, st.ten, st.basis)
        @test compute_psi(st_shift, WF_RECT, xg, yg; inside_only=false) ≈ compute_psi(st, WF_RECT, xg, yg; inside_only=false)
    end
end

@testset "wavefunction - rectangle even-even mode (cos x cos y)" begin
    st, f = wf_rect_mode(1, 1)

    @testset "fundamental domain: first-quadrant grid" begin
        psi, x, y = wavefunction(st, WF_RECT)
        @test psi isa Matrix{Float64}
        @test size(psi) == (length(x), length(y))
        @test all(>(0), x) && all(>(0), y)                      # reflection-reduced grids are rectified
        @test x[end] > WF_A && y[end] > WF_B
        inside = wf_rect_interior(x, y)
        @test count(inside) > 1e4
        @test psi[inside] ≈ wf_analytic(f, x, y)[inside] atol = 1e-12
        @test all(psi[wf_rect_exterior(x, y)] .== 0)
    end

    @testset "fundamental_domain = false unfolds onto the full rectangle" begin
        psi, x, y = wavefunction(st, WF_RECT; fundamental_domain=false)
        @test size(psi) == (length(x), length(y))
        @test x ≈ -reverse(x) && y ≈ -reverse(y)                # symmetric grids
        inside = wf_rect_interior(x, y)
        @test count(inside) > 4e4
        @test psi[inside] ≈ wf_analytic(f, x, y)[inside] atol = 1e-12
        @test all(psi[wf_rect_exterior(x, y)] .== 0)
        # even-even: psi(-x, y) = psi(x, y) = psi(x, -y)
        @test psi ≈ reverse(psi; dims=1) atol = 1e-12
        @test psi ≈ reverse(psi; dims=2) atol = 1e-12
    end

    @testset "inside_only = false gives the analytic mode on the whole grid box" begin
        psi_f, xf, yf = wavefunction(st, WF_RECT; inside_only=false)
        @test psi_f ≈ wf_analytic(f, xf, yf) atol = 1e-12
        psi_u, xu, yu = wavefunction(st, WF_RECT; inside_only=false, fundamental_domain=false)
        @test psi_u ≈ wf_analytic(f, xu, yu) atol = 1e-12
        @test size(psi_u) == (2 * length(xf), 2 * length(yf))
    end

    @testset "use_NaN_mask" begin
        psi, x, y = wavefunction(st, WF_RECT; use_NaN_mask=true)
        outside = wf_rect_exterior(x, y)
        @test all(isnan, psi[outside])
        @test !any(isnan, psi[wf_rect_interior(x, y)])
        psi_u, xu, yu = wavefunction(st, WF_RECT; use_NaN_mask=true, fundamental_domain=false)
        @test all(isnan, psi_u[wf_rect_exterior(xu, yu)])
        @test !any(isnan, psi_u[wf_rect_interior(xu, yu)])
    end

    @testset "memory_limit and multithreaded do not change the result" begin
        ref, _, _ = wavefunction(st, WF_RECT)
        small, _, _ = wavefunction(st, WF_RECT; memory_limit=8e4)
        serial, _, _ = wavefunction(st, WF_RECT; multithreaded=false)
        @test small == ref
        @test serial == ref
    end
end

@testset "wavefunction - rectangle odd-odd mode (sin x sin y)" begin
    st, f = wf_rect_mode(2, 2)
    @test st.basis.sym_qnumbers == [-1.0, 1.0, -1.0]

    psi, x, y = wavefunction(st, WF_RECT)
    inside = wf_rect_interior(x, y)
    @test psi[inside] ≈ wf_analytic(f, x, y)[inside] atol = 1e-12

    psi_u, xu, yu = wavefunction(st, WF_RECT; fundamental_domain=false)
    @test size(psi_u) == (length(xu), length(yu))
    inside_u = wf_rect_interior(xu, yu)
    @test psi_u[inside_u] ≈ wf_analytic(f, xu, yu)[inside_u] atol = 1e-12
    @test all(psi_u[wf_rect_exterior(xu, yu)] .== 0)
    # odd-odd: psi(-x, y) = -psi(x, y), psi(x, -y) = -psi(x, y), psi(-x, -y) = psi(x, y)
    @test psi_u ≈ -reverse(psi_u; dims=1) atol = 1e-12
    @test psi_u ≈ -reverse(psi_u; dims=2) atol = 1e-12
    @test psi_u ≈ reverse(psi_u; dims=(1, 2)) atol = 1e-12
end

@testset "wavefunction - rectangle mixed-parity mode (cos x sin y)" begin
    st, f = wf_rect_mode(1, 2)
    @test st.basis.sym_qnumbers == [1.0, -1.0, -1.0]

    psi_u, xu, yu = wavefunction(st, WF_RECT; fundamental_domain=false)
    inside = wf_rect_interior(xu, yu)
    @test psi_u[inside] ≈ wf_analytic(f, xu, yu)[inside] atol = 1e-12
    @test psi_u ≈ reverse(psi_u; dims=1) atol = 1e-12               # even in x
    @test psi_u ≈ -reverse(psi_u; dims=2) atol = 1e-12              # odd in y
end

@testset "wavefunction - circle J0 mode with a reflection-adapted FourierBessel basis" begin
    basis = FourierBessel(1; sym_x=1, sym_y=1)
    @test (basis.orders, basis.trig) == ([0], Int8[1])
    st = BasisEigenstate(WF_J01, [1.0], 1e-12, basis)

    @testset "fundamental domain" begin
        psi, x, y = wavefunction(st, WF_CIRCLE)
        @test size(psi) == (length(x), length(y))
        @test all(>(0), x) && all(>(0), y)
        inside = wf_quadrant_interior(x, y, 1 - 1e-9)
        @test count(inside) > 1e5
        @test psi[inside] ≈ wf_j0(x, y)[inside] atol = 1e-12
        @test all(psi[wf_disc_exterior(x, y)] .== 0)
    end

    @testset "fundamental_domain = false" begin
        psi, x, y = wavefunction(st, WF_CIRCLE; fundamental_domain=false)
        @test size(psi) == (length(x), length(y))
        inside = wf_disc_interior(x, y, 1 - 1e-9)
        @test count(inside) > 4e5
        @test psi[inside] ≈ wf_j0(x, y)[inside] atol = 1e-12
        @test all(psi[wf_disc_exterior(x, y)] .== 0)
        @test minimum(x) < -1 && maximum(x) > 1
    end

    @testset "inside_only = false evaluates the analytic function outside the disc as well" begin
        psi, x, y = wavefunction(st, WF_CIRCLE; inside_only=false, fundamental_domain=false)
        @test psi ≈ wf_j0(x, y) atol = 1e-12
        @test any(!iszero, psi[wf_disc_exterior(x, y)])
    end
end

@testset "wavefunction - circle J0 mode with a basis without symmetries" begin
    basis = FourierBessel(3)
    @test basis.symmetries === nothing
    ij0 = findfirst(i -> basis.orders[i] == 0 && basis.trig[i] > 0, 1:basis.dim)
    vec = zeros(basis.dim); vec[ij0] = 1.0
    st = BasisEigenstate(WF_J01, vec, 1e-12, basis)

    psi, x, y = wavefunction(st, WF_CIRCLE)
    @test size(psi) == (512, 512)                                 # grids are not rectified
    @test -0.01 - 1e-10 <= x[1] <= 0.0
    @test x[end] ≈ 1.01 atol = 1e-6
    @test y[1] ≈ -0.01 atol = 1e-6
    @test y[end] ≈ 1.01 atol = 1e-3
    inside = wf_quadrant_interior(x, y, 1 - 1e-9)
    @test psi[inside] ≈ wf_j0(x, y)[inside] atol = 1e-12
    @test all(psi[wf_disc_exterior(x, y)] .== 0)
    @test all(psi[[xx < -1e-9 || yy < -1e-9 for xx in x, yy in y]] .== 0)       # outside the quadrant

    @testset "nothing to unfold: fundamental_domain = false returns the same field" begin
        psi_u, x_u, y_u = wavefunction(st, WF_CIRCLE; fundamental_domain=false)
        @test psi_u == psi && x_u == x && y_u == y
    end
end

@testset "wavefunction - single-axis reflection basis on the Limacon" begin
    # XAxisReflection constrains the parity in y and unfolds along y; the Limacon fundamental domain is the upper half
    # (x from about -0.71 to 1.31, y > 0). wavefunction rectifies the y grid and leaves x unrectified.
    lim = LimaconBilliard(0.3)
    basis = RealPlaneWaves(3; sym_y=1)
    @test basis.symmetries[1] isa BilliardGeometry.XAxisReflection
    st = BasisEigenstate(10.0, ones(basis.dim), 0.1, basis)

    psi, x, y = wavefunction(st, lim)
    @test size(psi) == (length(x), length(y))
    @test minimum(x) < -0.5          # the grid covers the whole fundamental domain in x
    @test minimum(y) > 0             # the y grid is rectified, since y is the reflected direction

    psi_u, xu, yu = wavefunction(st, lim; fundamental_domain=false)
    @test size(psi_u) == (length(xu), length(yu))
end

@testset "wavefunction(::BasisState)" begin
    basis = FourierBessel(5)
    k = 3.0
    @test basis.orders[1] == 0 && basis.trig[1] > 0               # first function is J0(k r)

    @testset "values of single Fourier-Bessel functions on the default grid" begin
        for i in 1:3
            st = BasisState(basis, k, i)
            psi, x, y = wavefunction(st)
            @test psi isa Matrix{Float64}
            @test size(psi) == (512, 512)
            @test (x[1], x[end]) == (-2.0, 2.0) && (y[1], y[end]) == (-2.0, 2.0)
            n = basis.orders[i]
            ang = [atan(yy, xx) for xx in x, yy in y]
            expected = besselj.(n, k .* wf_radius(x, y)) .* (basis.trig[i] > 0 ? cos.(n .* ang) : sin.(n .* ang))
            @test psi ≈ expected atol = 1e-12
        end
    end

    @testset "grid limits and resolution" begin
        st = BasisState(basis, k, 1)
        psi, x, y = wavefunction(st; xlim=(-1.0, 1.0), ylim=(-3.0, 3.0), b=400.0)
        @test (x[1], x[end]) == (-1.0, 1.0) && (y[1], y[end]) == (-3.0, 3.0)
        @test length(x) == _debroglie_grid_size(k, 2.0, 400.0)
        @test length(y) == _debroglie_grid_size(k, 6.0, 400.0)
        @test length(y) > length(x) == 512
        @test size(psi) == (length(x), length(y))
        @test psi ≈ besselj.(0, k .* wf_radius(x, y)) atol = 1e-12
    end

    @testset "only the selected basis function contributes" begin
        psi1, _, _ = wavefunction(BasisState(basis, k, 1))
        psi2, _, _ = wavefunction(BasisState(basis, k, 2))
        @test psi1 != psi2
    end
end

@testset "wavefunction(::BIMEigenstate) - single-layer reconstruction of the circle J0 mode" begin
    st = wf_bim_u_state(WF_J01)
    solver = wf_SweepStub()

    @testset "fundamental domain, direct Hankel evaluation" begin
        psi, x, y = wavefunction(st, WF_CIRCLE, solver; use_chebyshev=false)
        @test psi isa Matrix{Float64}
        @test size(psi) == (length(x), length(y))
        inside = wf_quadrant_interior(x, y, 0.5)
        @test count(inside) > 4e4
        @test psi[inside] ≈ -wf_j0(x, y)[inside] atol = 1e-10         # psi = -J0(k r) for u = d_n J0
        @test all(psi[wf_disc_exterior(x, y)] .== 0)
        @test all(psi[[xx < -1e-9 || yy < -1e-9 for xx in x, yy in y]] .== 0)
    end

    @testset "fundamental_domain = false covers the whole disc" begin
        psi, x, y = wavefunction(st, WF_CIRCLE, solver; use_chebyshev=false, fundamental_domain=false)
        @test x[1] ≈ -1.01 && x[end] ≈ 1.01
        inside = wf_disc_interior(x, y, 0.5)
        @test psi[inside] ≈ -wf_j0(x, y)[inside] atol = 1e-10
        @test all(psi[wf_disc_exterior(x, y)] .== 0)
    end

    @testset "Chebyshev Hankel evaluation agrees with the analytic field" begin
        psi, x, y = wavefunction(st, WF_CIRCLE, solver; use_chebyshev=true)
        inside = wf_quadrant_interior(x, y, 0.5)
        @test psi[inside] ≈ -wf_j0(x, y)[inside] atol = 1e-6
    end

    @testset "use_NaN_mask" begin
        psi, x, y = wavefunction(st, WF_CIRCLE, solver; use_chebyshev=false, use_NaN_mask=true)
        @test all(isnan, psi[wf_disc_exterior(x, y)])
        @test !any(isnan, psi[wf_quadrant_interior(x, y, 0.5)])
    end
end

@testset "wavefunction(::BIMEigenstate) - density reconstruction selects the kernel from the solver type" begin
    st = wf_bim_density_state(WF_J01)

    @testset "DLP: psi = D mu" begin
        psi, x, y = wavefunction(st, WF_CIRCLE, wf_DLPStub(); use_chebyshev=false)
        inside = wf_quadrant_interior(x, y, 0.5)
        expected = -(im * pi * WF_J01 / 2) * hankelh1(1, WF_J01) .* wf_j0(x, y)
        @test psi isa Matrix{ComplexF64}
        @test psi[inside] ≈ expected[inside] atol = 1e-10
    end

    @testset "CFIE: psi = -(D + i k S) mu" begin
        psi, x, y = wavefunction(st, WF_CIRCLE, wf_CFIEStub(); use_chebyshev=false)
        inside = wf_quadrant_interior(x, y, 0.5)
        expected = pi * WF_J01 * (hankelh1(0, WF_J01) + im * hankelh1(1, WF_J01)) .* wf_j0(x, y)
        @test psi[inside] ≈ expected[inside] atol = 1e-10
    end

    @testset "DLP with Chebyshev Hankel evaluation" begin
        psi, x, y = wavefunction(st, WF_CIRCLE, wf_DLPStub(); use_chebyshev=true)
        inside = wf_quadrant_interior(x, y, 0.5)
        expected = -(im * pi * WF_J01 / 2) * hankelh1(1, WF_J01) .* wf_j0(x, y)
        @test psi[inside] ≈ expected[inside] atol = 1e-6
    end
end

@testset "wavefunction(states, ...) - batched BIM reconstruction" begin
    s1 = wf_bim_u_state(WF_J01); s2 = wf_bim_u_state(WF_J02)
    solver = wf_SweepStub()

    @testset "shared grid, one field per state" begin
        Psi, x, y = wavefunction([s1, s2], WF_CIRCLE, solver; use_chebyshev=false, show_progress=false)
        @test length(Psi) == 2
        @test all(size(P) == (length(x), length(y)) for P in Psi)
        inside = wf_quadrant_interior(x, y, 0.5)
        @test Psi[1][inside] ≈ -wf_j0(x, y, WF_J01)[inside] atol = 1e-10
        @test Psi[2][inside] ≈ -wf_j0(x, y, WF_J02)[inside] atol = 1e-10
    end

    @testset "invalid input" begin
        @test_throws ArgumentError wavefunction(typeof(s1)[], WF_CIRCLE, solver)
        # one state with only a density, one with only the normal derivative
        mixed = [wf_bim_density_state(WF_J01), s2]
        @test_throws ErrorException wavefunction(mixed, WF_CIRCLE, solver; show_progress=false)
    end
end

@testset "wavefunction(::SpectralData) wrappers" begin
    @testset "basis eigenstates: one grid per state" begin
        st1, f1 = wf_rect_mode(1, 1); st2, f2 = wf_rect_mode(2, 2)
        data = SpectralData(WF_RECT, nothing, [st1.k, st2.k], [1e-12, 1e-12], [false, false]; states=[st1, st2])

        Psi, xs, ys = wavefunction(data)
        @test length(Psi) == length(xs) == length(ys) == 2
        ref1 = wavefunction(st1, WF_RECT); ref2 = wavefunction(st2, WF_RECT)
        @test Psi[1] == ref1[1] && xs[1] == ref1[2] && ys[1] == ref1[3]
        @test Psi[2] == ref2[1] && xs[2] == ref2[2] && ys[2] == ref2[3]

        psi, x, y = wavefunction(data, 2)
        @test psi == ref2[1] && x == ref2[2] && y == ref2[3]
        inside = wf_rect_interior(x, y)
        @test psi[inside] ≈ wf_analytic(f2, x, y)[inside] atol = 1e-12

        @testset "keywords are forwarded" begin
            psi_u, x_u, y_u = wavefunction(data, 1; fundamental_domain=false)
            @test size(psi_u) == (length(x_u), length(y_u))
            @test length(x_u) == 2 * length(xs[1])
            Psi_u, _, _ = wavefunction(data; fundamental_domain=false)
            @test Psi_u[1] == psi_u
        end
    end

    @testset "BIM eigenstates: shared grid" begin
        s2 = wf_bim_u_state(WF_J02)
        data = SpectralData(WF_CIRCLE, wf_SweepStub(), [WF_J02], [1e-9], [false]; states=[s2])
        Psi, x, y = wavefunction(data; use_chebyshev=false, show_progress=false)
        @test length(Psi) == 1 && x isa Vector{Float64} && y isa Vector{Float64}
        inside = wf_quadrant_interior(x, y, 0.5)
        @test Psi[1][inside] ≈ -wf_j0(x, y, WF_J02)[inside] atol = 1e-10

        psi, x1, y1 = wavefunction(data, 1; use_chebyshev=false)
        @test psi == Psi[1] && x1 == x && y1 == y
    end

    @testset "data without stored states" begin
        data = SpectralData(WF_RECT, nothing, [3.0], [1e-12], [false])
        @test_throws ArgumentError wavefunction(data)
        @test_throws ArgumentError wavefunction(data, 1)
    end
end
