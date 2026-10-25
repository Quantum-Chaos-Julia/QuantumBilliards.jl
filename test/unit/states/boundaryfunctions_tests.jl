# Unit tests for
#   src/states/boundaryfunctions.jl   (regularize!, _rellich, boundary_function for basis eigenstates and BIM eigenstates,
#                                      momentum_function for a sampled function, for a non-uniform sampling and for
#                                      eigenstates, and the SpectralData wrappers)
# The boundary function is the normal derivative u(s) = d_n psi(s) on the (unfolded) boundary and the momentum function is
# its one-sided power spectral density. There is no `fundamental_domain` or `inside_only` keyword in this file; what
# corresponds to it is whether the basis carries reflection symmetries: a basis with symmetries returns the function
# unfolded onto the full boundary, a basis without symmetries returns the boundary of the fundamental domain only.
#
# No eigen solves. All references are analytic or properties:
#  * regularize!: exact neighbour averages.
#  * _rellich on hand-built boundary points. Unit circle with constant u = c: (1/2k^2) 2 pi c^2 = 1 for c = k/sqrt(pi).
#    Rectangle [0,a] x [0,b] with the normalised mode psi = 2/sqrt(ab) sin(pi x/a) sin(pi y/b), k^2 = pi^2(1/a^2 + 1/b^2):
#    the midpoint rule integrates sin^2 exactly, and (1/2k^2) oint (n.x) u^2 ds = 1 (n.x = 0 on the two walls through the
#    origin, a resp. b on the other two).
#  * boundary_function of hand-built basis states: circle J0(j01 r) (u constant), circle J1(j11 r) cos(theta) (u ~ cos(theta)),
#    rectangle plane-wave products cos/sin(kx x) cos/sin(ky y) (RealPlaneWaves with one wave), and a single-axis
#    reflection basis on the Limacon. The pipeline is verified through the identity u = grad(psi) . n evaluated at the
#    returned (unfolded) points with the returned normals, the Rellich identity  rellich(unfolded) = (number of
#    copies) * rellich(fundamental), and a continuity check of the unfolded boundary.
#  * momentum_function: u = cos(2 pi m s / L) has the one-sided density L/(4 pi) in the bin m (amplitude^2 / (2 dk)), and
#    Parseval, sum(power) * dk = mean(|u|^2). A non-uniform sampling from the smooth periodic map
#    s(t) = L (t - sin(2 pi t) / (2 pi)) gives a spectrally accurate trapezoid in t.
# BIM states are hand-built (partial states with u, built through the type-parameter constructor, see
# eigenstates_tests.jl) and the recomputation path goes through a test-local solver stub.
# Helper definitions use the prefix `bf_` / `BF_`.
#
# SOURCE FINDINGS:
#  (fixed) boundary_coords normalises the normals per element, and regularize! uses periodic neighbours and the nearest
#  finite neighbours on both sides; the corresponding assertions are plain @test below. The FourierNodes weights of
#  boundary_coords sum to the boundary length on multi-curve boundaries.

using StaticArrays
using LinearAlgebra: norm, dot
using SpecialFunctions: besselj
using QuantumBilliards: regularize!, _rellich, _basis_boundary_function_pts, gradient_matrices

const BF_J01 = 2.404825557695773       # first zero of J0
const BF_J02 = 5.520078110286311       # second zero of J0
const BF_J11 = 3.8317059702075125      # first zero of J1
const BF_A, BF_B = 1.0, 0.6
const BF_RECT = RectangleBilliard(BF_A, BF_B)
const BF_CIRCLE = CircleBilliard(1.0)

# Test-local BIM solver stub: only the extension point used when a state carries no precomputed u.
struct bf_SweepStub <: QuantumBilliards.SweepBIMSolver end
const BF_STUB_U = -3.0
const BF_STUB_NORM = 0.5
QuantumBilliards.solve_state(::bf_SweepStub, pts::BoundaryPoints, k, billiard; multithreaded::Bool=true) =
    (0.0, zeros(length(pts.xy)), fill(BF_STUB_U, length(pts.xy)), BF_STUB_NORM)

# Unit circle sampled at N equispaced angles; s = theta, ds = 2 pi / N, outward unit normals.
function bf_circle_pts(::Type{T}, N::Int) where {T<:Real}
    L = T(2pi)
    s = collect(T, 0:N-1) .* (L / N)
    xy = [SVector{2,T}(cos(t), sin(t)) for t in s]
    return BoundaryPoints(xy; normal=copy(xy), s=s, ds=fill(L / N, N))
end

# Unit circle with the smooth periodic grading s(t) = L (t - sin(2 pi t) / (2 pi)), t_j = (j - 1) / N, ds = s'(t) / N.
function bf_graded_circle_pts(N::Int)
    L = 2pi
    t = collect(0:N-1) ./ N
    s = L .* (t .- sin.(2pi .* t) ./ (2pi))
    ds = L .* (1 .- cos.(2pi .* t)) ./ N
    xy = [SVector(cos(a), sin(a)) for a in s]
    return BoundaryPoints(xy; normal=copy(xy), s=s, ds=ds)
end

# Boundary points of the rectangle [0,a] x [0,b]: N midpoints per side with outward unit normals.
function bf_rect_pts(a, b, N)
    xy = SVector{2,Float64}[]; nrm = SVector{2,Float64}[]; ds = Float64[]
    for j in 1:N
        push!(xy, SVector((j - 0.5) * a / N, 0.0)); push!(nrm, SVector(0.0, -1.0)); push!(ds, a / N)
    end
    for j in 1:N
        push!(xy, SVector(a, (j - 0.5) * b / N)); push!(nrm, SVector(1.0, 0.0)); push!(ds, b / N)
    end
    for j in 1:N
        push!(xy, SVector(a - (j - 0.5) * a / N, b)); push!(nrm, SVector(0.0, 1.0)); push!(ds, a / N)
    end
    for j in 1:N
        push!(xy, SVector(0.0, b - (j - 0.5) * b / N)); push!(nrm, SVector(-1.0, 0.0)); push!(ds, b / N)
    end
    return BoundaryPoints(xy; normal=nrm, ds=ds, s=cumsum(ds))
end

# Gradient of psi = J1(k r) cos(theta) = J1(k r) x / r (J1' = (J0 - J2) / 2).
function bf_grad_j1(x, y, k)
    r = hypot(x, y)
    j1 = besselj(1, k * r); dj1 = (besselj(0, k * r) - besselj(2, k * r)) / 2
    return SVector(k * dj1 * x^2 / r^2 + j1 * (1 / r - x^2 / r^3), k * dj1 * x * y / r^2 - j1 * x * y / r^3)
end
# Gradient of psi = J0(k r).
bf_grad_j0(x, y, k) = -k * besselj(1, k * hypot(x, y)) .* SVector(x, y) ./ hypot(x, y)

bf_j0_state() = BasisEigenstate(BF_J01, [1.0], 1e-12, FourierBessel(1; sym_x=1, sym_y=1))
bf_j1_state() = BasisEigenstate(BF_J11, [1.0], 1e-12, FourierBessel(1; sym_x=-1, sym_y=1))      # J1(k r) cos(theta)

# Rectangle mode (m, n): psi = f_x(kx x) f_y(ky y), cos for odd and sin for even indices, with kx = m pi / (2a),
# ky = n pi / (2b); built as a one-wave RealPlaneWaves state. Returns the state and the analytic gradient.
function bf_rect_mode(m::Int, n::Int)
    kx = m * pi / (2BF_A); ky = n * pi / (2BF_B)
    sx = isodd(m) ? 1 : -1; sy = isodd(n) ? 1 : -1
    basis = RealPlaneWaves(1; sym_x=sx, sym_y=sy, angle_arc=0.0, angle_shift=atan(ky, kx))
    fx = isodd(m) ? cos : sin; fy = isodd(n) ? cos : sin
    grad = (x, y) -> SVector(kx * (isodd(m) ? -sin(kx * x) : cos(kx * x)) * fy(ky * y),
                             ky * (isodd(n) ? -sin(ky * y) : cos(ky * y)) * fx(kx * x))
    return BasisEigenstate(hypot(kx, ky), [1.0], 1e-12, basis), grad
end

# Hand-built BIM state with a precomputed u (partial state, no density) on the given points.
bf_bim_state(k, u, pts) = BIMEigenstate{Float64,Float64}(k, 0.0, nothing, 1e-9, pts, u, 1.25)

# The common checks of the unfolded boundary data of a state with the analytic gradient `grad`.
function bf_check_unfolded(st, billiard, grad; ncopies, perimeter, rtol_perimeter=1e-9)
    u, pts, nrm = _basis_boundary_function_pts(st, billiard)
    @test length(u) == length(pts.xy) == length(pts.normal) == length(pts.ds) == length(pts.s)
    @test !any(isnan, u)
    @test all(isfinite, pts.s) && all(>=(-1e-12), diff(pts.s))       # non-decreasing up to rounding (zero weights at curve ends)
    @test pts.s[end] ≈ perimeter rtol = rtol_perimeter
    # u = grad(psi) . n at the unfolded points with the returned normals: positions, normals and u are consistent
    @test u ≈ [dot(grad(p[1], p[2]), n) for (p, n) in zip(pts.xy, pts.normal)] atol = 1e-10
    # the Rellich integrand is invariant under reflections, each copy contributes the same
    @test _rellich(pts, u, st.k) ≈ ncopies * nrm rtol = 1e-9
    # continuity of the unfolded boundary: consecutive nodes (and the closing pair) are close
    maxds = maximum(pts.ds)
    chords = [norm(pts.xy[mod1(i + 1, length(pts.xy))] - pts.xy[i]) for i in eachindex(pts.xy)]
    @test maximum(chords) <= 2 * maxds
    return u, pts, nrm
end

@testset "regularize!" begin
    @testset "an interior NaN is replaced by the mean of its neighbours" begin
        u = [1.0, 2.0, NaN, 6.0, 5.0]
        @test regularize!(u) === nothing
        @test u == [1.0, 2.0, 4.0, 6.0, 5.0]
    end

    @testset "a NaN in the first entry uses the second and the last entry" begin
        u = [NaN, 2.0, 3.0, 8.0]
        regularize!(u)
        @test u == [5.0, 2.0, 3.0, 8.0]
    end

    @testset "several isolated NaNs" begin
        u = [1.0, NaN, 3.0, 4.0, NaN, 10.0]
        regularize!(u)
        @test u == [1.0, 2.0, 3.0, 4.0, 7.0, 10.0]
    end

    @testset "a vector without NaN is returned unchanged" begin
        u = [1.0, -2.0, 3.0]
        u0 = copy(u)
        regularize!(u)
        @test u == u0
    end

    @testset "complex entries and element type" begin
        z = ComplexF64[1 + 1im, complex(NaN, NaN), 3 - 1im]
        regularize!(z)
        @test z == ComplexF64[1 + 1im, 2 + 0im, 3 - 1im]
        u32 = Float32[1, NaN, 3]
        regularize!(u32)
        @test u32 isa Vector{Float32} && u32 == Float32[1, 2, 3]
    end

    @testset "periodic neighbours: last entry and adjacent NaNs" begin
        # The curve is closed, so the neighbours of the last entry are u[end-1] and u[1].
        u = [1.0, 2.0, 3.0, NaN]
        regularize!(u)
        @test u[end] ≈ 2.0
        # NaNs in neighbouring entries are replaced using the nearest finite neighbours.
        v = [1.0, NaN, NaN, 4.0]
        regularize!(v)
        @test !any(isnan, v)
    end
end

@testset "_rellich" begin
    @testset "unit circle, constant normal derivative c = k / sqrt(pi) gives 1" begin
        pts = bf_circle_pts(Float64, 64)
        k = BF_J01
        @test _rellich(pts, fill(k / sqrt(pi), 64), k) ≈ 1 atol = 1e-13
    end

    @testset "quadratic homogeneity and phase invariance" begin
        pts = bf_circle_pts(Float64, 64)
        k = 3.0
        u = fill(0.7, 64)
        r = _rellich(pts, u, k)
        @test r ≈ 2pi * 0.7^2 / (2k^2)
        @test _rellich(pts, 3 .* u, k) ≈ 9r
        @test _rellich(pts, -u, k) ≈ r
        @test _rellich(pts, exp(0.9im) .* u, k) ≈ r
        @test _rellich(pts, u, 2k) ≈ r / 4
        @test _rellich(pts, zero(u), k) == 0
    end

    @testset "rectangle sin * sin mode, (1/2k^2) oint (n.x) u^2 ds = 1" begin
        a, b = BF_A, BF_B
        pts = bf_rect_pts(a, b, 8)
        k = pi * sqrt(1 / a^2 + 1 / b^2)
        c = 2 / sqrt(a * b)
        grad(p) = c .* SVector((pi / a) * cos(pi * p[1] / a) * sin(pi * p[2] / b), (pi / b) * sin(pi * p[1] / a) * cos(pi * p[2] / b))
        u = [dot(grad(p), n) for (p, n) in zip(pts.xy, pts.normal)]
        @test _rellich(pts, u, k) ≈ 1 atol = 1e-13
    end

    @testset "complex u uses |u|^2 and the result is real" begin
        pts = bf_circle_pts(Float64, 16)
        uc = fill(0.5 + 0.5im, 16)
        @test _rellich(pts, uc, 2.0) isa Float64
        @test _rellich(pts, uc, 2.0) ≈ _rellich(pts, fill(sqrt(0.5), 16), 2.0)
    end

    @testset "Float32 points give a Float32 result" begin
        pts = bf_circle_pts(Float32, 32)
        r = _rellich(pts, fill(0.5f0, 32), 2.0f0)
        @test r isa Float32
        @test r ≈ Float32(2pi * 0.25 / 8) rtol = 1e-5
    end
end

@testset "boundary_function - circle J0 mode (constant normal derivative)" begin
    st = bf_j0_state()
    u, s, nrm = boundary_function(st, BF_CIRCLE)

    @testset "return values" begin
        @test u isa Vector{Float64} && s isa Vector{Float64} && nrm isa Float64
        @test length(u) == length(s) == 4 * 513                       # four reflected copies of the quarter-arc nodes
        @test issorted(s)
        @test s[end] ≈ 2pi rtol = 1e-12                               # arc length of the full unit circle
    end

    @testset "u is constant on the boundary" begin
        @test maximum(abs, u .- u[1]) <= 1e-10 * abs(u[1])
        @test u[1] < 0                                                # d_n J0(j01 r) = -k J1(k) < 0
    end

    @testset "agrees with the unfolded boundary data and with u = grad(psi) . n" begin
        u2, pts, nrm2 = _basis_boundary_function_pts(st, BF_CIRCLE)
        @test u == u2 && s == pts.s && nrm == nrm2
        @test u ≈ [dot(bf_grad_j0(p[1], p[2], st.k), n) for (p, n) in zip(pts.xy, pts.normal)] atol = 1e-12
        @test _rellich(pts, u, st.k) ≈ 4 * nrm rtol = 1e-9            # four copies of the fundamental boundary
    end

    @testset "linearity in the coefficients, quadratic scaling of the Rellich value" begin
        st2 = BasisEigenstate(st.k, [-2.5], st.ten, st.basis)
        u2, s2, nrm2 = boundary_function(st2, BF_CIRCLE)
        @test u2 ≈ -2.5 .* u && s2 == s
        @test nrm2 ≈ 6.25 * nrm
    end

    @testset "b controls the sampling density, multithreaded does not change the values" begin
        ub, sb, _ = boundary_function(st, BF_CIRCLE; b=3000.0)
        @test length(ub) > length(u) && length(sb) == length(ub)
        @test maximum(abs, ub .- ub[1]) <= 1e-10 * abs(ub[1])
        us, ss, ns = boundary_function(st, BF_CIRCLE; multithreaded=false)
        @test us ≈ u && ss == s && ns ≈ nrm
    end

    @testset "normals of the boundary points are unit vectors" begin
        _, pts, _ = _basis_boundary_function_pts(st, BF_CIRCLE)
        @test all(n -> isapprox(norm(n), 1; atol=1e-8), pts.normal)
        @test isapprox(u[1], -st.k * besselj(1, st.k); rtol=1e-8)       # d_n J0(k r) at r = 1
        # For psi = J0(k r) / (sqrt(pi) |J1(k)|) (unit norm on the disc) the quarter-arc Rellich value is 1/4.
        c = 1 / (sqrt(pi) * abs(besselj(1, st.k)))
        _, _, nrm_unit = boundary_function(BasisEigenstate(st.k, [c], st.ten, st.basis), BF_CIRCLE)
        @test isapprox(nrm_unit, 1 / 4; rtol=1e-6)
    end
end

@testset "boundary_function - circle, fundamental domain only for a basis without symmetries" begin
    basis = FourierBessel(3)
    @test basis.symmetries === nothing
    vec = zeros(basis.dim); vec[findfirst(i -> basis.orders[i] == 0 && basis.trig[i] > 0, 1:basis.dim)] = 1.0
    st = BasisEigenstate(BF_J01, vec, 1e-12, basis)
    u, s, nrm = boundary_function(st, BF_CIRCLE)
    @test length(u) == 513                                            # nothing is unfolded
    @test s[end] ≈ pi / 2 rtol = 1e-9                                 # the quarter arc
    @test maximum(abs, u .- u[1]) <= 1e-10 * abs(u[1])
    # same fundamental-boundary values as the symmetry-adapted basis
    u_sym, _, nrm_sym = boundary_function(bf_j0_state(), BF_CIRCLE)
    @test u ≈ u_sym[1:513] && nrm ≈ nrm_sym
end

@testset "boundary_function - circle J1 cos(theta) mode unfolds with the right signs" begin
    st = bf_j1_state()
    @test st.basis.sym_qnumbers == [-1.0, -1.0, 1.0]                  # odd in x, even in y
    u, pts, _ = bf_check_unfolded(st, BF_CIRCLE, (x, y) -> bf_grad_j1(x, y, st.k); ncopies=4, perimeter=2pi)
    θ = [atan(p[2], p[1]) for p in pts.xy]
    # u is proportional to cos(theta): the ratio is constant up to the nodes where cos(theta) vanishes
    away = abs.(cos.(θ)) .> 0.1
    ratio = u[away] ./ cos.(θ[away])
    @test maximum(abs, ratio .- ratio[1]) <= 1e-8 * abs(ratio[1])
    # d_n psi = k J0(k) cos(theta) at r = 1 (J1(k) = 0)
    @test isapprox(ratio[1], st.k * besselj(0, st.k); rtol=1e-8)
end

@testset "boundary_function - rectangle modes, D2 unfolding" begin
    perimeter = 4 * (BF_A + BF_B)

    @testset "even-even mode cos x cos y" begin
        st, grad = bf_rect_mode(1, 1)
        u, pts, _ = bf_check_unfolded(st, BF_RECT, grad; ncopies=4, perimeter=perimeter, rtol_perimeter=1e-2)
        @test extrema(p[1] for p in pts.xy) == (-BF_A, BF_A) && extrema(p[2] for p in pts.xy) == (-BF_B, BF_B)
        @test all(p -> isapprox(abs(p[1]), BF_A; atol=1e-12) || isapprox(abs(p[2]), BF_B; atol=1e-12), pts.xy)    # on the walls
    end

    @testset "odd-odd mode sin x sin y" begin
        st, grad = bf_rect_mode(2, 2)
        bf_check_unfolded(st, BF_RECT, grad; ncopies=4, perimeter=perimeter, rtol_perimeter=1e-2)      # parities pY = pX = -1 enter the unfolding
    end

    @testset "mixed-parity mode cos x sin y" begin
        st, grad = bf_rect_mode(1, 2)
        bf_check_unfolded(st, BF_RECT, grad; ncopies=4, perimeter=perimeter, rtol_perimeter=1e-2)
    end

    @testset "the three-element result of boundary_function matches the internal data" begin
        st, _ = bf_rect_mode(1, 1)
        u, s, nrm = boundary_function(st, BF_RECT)
        u2, pts, nrm2 = _basis_boundary_function_pts(st, BF_RECT)
        @test u == u2 && s == pts.s && nrm == nrm2
    end

    @testset "multi-curve boundary: unit normals, quadrature weights sum to the boundary length" begin
        st, _ = bf_rect_mode(1, 1)
        _, pts, _ = _basis_boundary_function_pts(st, BF_RECT)
        @test all(n -> isapprox(norm(n), 1; atol=1e-8), pts.normal)
        @test isapprox(sum(pts.ds), 4 * (BF_A + BF_B); rtol=1e-9)
    end
end

@testset "boundary_function - single-axis reflection basis on the Limacon" begin
    # XAxisReflection only: the fundamental boundary is the upper arc and is unfolded by one reversed copy. The boundary
    # data (positions, normals, u) are checked against a direct evaluation of the basis at the unfolded points.
    lim = LimaconBilliard(0.3)
    basis = RealPlaneWaves(3; sym_y=1)
    @test length(basis.symmetries) == 1 && basis.symmetries[1] isa BilliardGeometry.XAxisReflection
    st = BasisEigenstate(10.0, ones(basis.dim), 0.1, basis)
    u, pts, nrm = _basis_boundary_function_pts(st, lim)

    @test length(pts.xy) % 2 == 0
    half = length(pts.xy) ÷ 2
    @test pts.xy[half+1:end] ≈ [SVector(p[1], -p[2]) for p in reverse(pts.xy[1:half])]       # reversed mirror image
    dX, dY = gradient_matrices(st.basis, st.k_basis, pts.xy)
    udirect = (dX * st.vec) .* getindex.(pts.normal, 1) .+ (dY * st.vec) .* getindex.(pts.normal, 2)
    @test u ≈ udirect atol = 1e-9 * maximum(abs, udirect)
    @test _rellich(pts, u, st.k) ≈ 2 * nrm rtol = 1e-9
    maxds = maximum(pts.ds)
    chords = [norm(pts.xy[mod1(i + 1, length(pts.xy))] - pts.xy[i]) for i in eachindex(pts.xy)]
    @test maximum(chords) <= 2 * maxds
    @test pts.s[end] ≈ sum(crv.length for crv in full_boundary(lim)) rtol = 1e-2       # rectangle rule on open arcs: O(spacing) error
end

@testset "boundary_function(::BIMEigenstate)" begin
    pts = bf_circle_pts(Float64, 32)
    u = collect(range(-1.0, -2.0, 32))
    st = bf_bim_state(BF_J01, u, pts)

    @testset "reads the precomputed fields; b and multithreaded are ignored" begin
        ub, sb, nb = boundary_function(st)
        @test ub === u && sb === pts.s && nb == 1.25
        @test boundary_function(st; b=100.0, multithreaded=false) == (u, pts.s, 1.25)
    end

    @testset "with a billiard: forwards to the one-argument method" begin
        @test boundary_function(st, BF_CIRCLE) == (u, pts.s, 1.25)
        @test boundary_function(st, BF_CIRCLE; b=7.0) == (u, pts.s, 1.25)
    end

    @testset "with a solver: stored u is used when present, recomputed through the kernel when missing" begin
        @test boundary_function(st, BF_CIRCLE, bf_SweepStub()) == (u, pts.s, 1.25)
        st_missing = BIMEigenstate{Float64,Float64}(BF_J01, 0.0, nothing, 1e-9, pts, nothing, nothing)
        u3, s3, n3 = boundary_function(st_missing, BF_CIRCLE, bf_SweepStub())
        @test u3 == fill(BF_STUB_U, 32) && s3 === pts.s && n3 == BF_STUB_NORM
    end

    @testset "a missing u without a solver throws instead of returning nothing" begin
        st_missing = BIMEigenstate{Float64,Float64}(BF_J01, 0.0, nothing, 1e-9, pts, nothing, nothing)
        @test_throws ArgumentError boundary_function(st_missing)
        @test_throws "no stored boundary normal derivative" boundary_function(st_missing)
        @test_throws ArgumentError boundary_function(st_missing; b=7.0, multithreaded=false)
    end

    @testset "a missing u with a billiard but no solver throws" begin
        st_missing = BIMEigenstate{Float64,Float64}(BF_J01, 0.0, nothing, 1e-9, pts, nothing, nothing)
        @test_throws ArgumentError boundary_function(st_missing, BF_CIRCLE)
        @test_throws "no stored boundary normal derivative" boundary_function(st_missing, BF_CIRCLE)
    end

    @testset "the solver form still recomputes a missing u" begin
        st_missing = BIMEigenstate{Float64,Float64}(BF_J01, 0.0, nothing, 1e-9, pts, nothing, nothing)
        u3, _, n3 = boundary_function(st_missing, BF_CIRCLE, bf_SweepStub())
        @test u3 == fill(BF_STUB_U, 32) && n3 == BF_STUB_NORM
    end
end

@testset "momentum_function(u, s) - analytic spectra" begin
    L = 7.0
    N = 64
    s = collect(1:N) .* (L / N)                                       # s[end] = L is the total arc length
    Δk = 2pi / L

    @testset "u = cos(2 pi m s / L): density amp^2 / (2 dk) = L / (4 pi) in bin m only" begin
        for m in (1, 5, 17)
            power, ks = momentum_function(cos.(2pi * m .* s ./ L), s)
            @test length(power) == length(ks) == N ÷ 2 + 1
            @test ks ≈ (0:N÷2) .* Δk
            @test power[m+1] ≈ L / (4pi) rtol = 1e-12
            rest = power[[i for i in eachindex(power) if i != m + 1]]
            @test maximum(abs, rest) <= 1e-12
        end
    end

    @testset "DC and Nyquist bins are not doubled" begin
        power, _ = momentum_function(fill(2.0, N), s)
        @test power[1] ≈ 4 / Δk && maximum(abs, power[2:end]) <= 1e-12
        power, _ = momentum_function([(-1.0)^j for j in 1:N], s)       # Nyquist: u_j = (-1)^j
        @test power[end] ≈ 1 / Δk && maximum(abs, power[1:end-1]) <= 1e-12
    end

    @testset "mixed spectrum: amplitudes and Parseval, even and odd N" begin
        for NN in (64, 65)
            ss = collect(1:NN) .* (L / NN)
            j = 1:NN
            u = 1 .+ 2 .* cos.(2pi * 3 .* j ./ NN) .+ 0.5 .* sin.(2pi * 5 .* j ./ NN)
            power, ks = momentum_function(u, ss)
            dk = ks[2] - ks[1]
            @test dk ≈ 2pi / L
            @test power[1] ≈ 1 / dk && power[4] ≈ 4 / (2dk) && power[6] ≈ 0.25 / (2dk)
            @test sum(power) * dk ≈ 1 + 2 + 0.125 rtol = 1e-12            # Parseval: mean(u^2) with orthogonal modes
            @test sum(power) * dk ≈ sum(abs2, u) / NN rtol = 1e-12
        end
        u = 1 .+ 2 .* cos.(2pi * 3 .* (1:N) ./ N) .+ 0.3 .* [(-1.0)^j for j in 1:N]
        power, ks = momentum_function(u, s)
        @test sum(power) * (ks[2] - ks[1]) ≈ 1 + 2 + 0.09 rtol = 1e-12       # Nyquist term included
    end

    @testset "invariances: cyclic shift, sign, quadratic scaling" begin
        u = 1 .+ 2 .* cos.(2pi * 3 .* (1:N) ./ N) .+ 0.5 .* sin.(2pi * 5 .* (1:N) ./ N)
        p0, _ = momentum_function(u, s)
        @test momentum_function(circshift(u, 11), s)[1] ≈ p0 atol = 1e-12
        @test momentum_function(-u, s)[1] ≈ p0
        @test momentum_function(3 .* u, s)[1] ≈ 9 .* p0
    end

    @testset "a sampled Gaussian pulse has a Gaussian spectrum" begin
        # u = exp(-(s - L/2)^2 / (2 w^2)) with w << L: the continuous transform |c_m|^2 = (w sqrt(2 pi) / L)^2 exp(-w^2 k_m^2)
        w = 0.25
        u = exp.(-(s .- L / 2) .^ 2 ./ (2w^2))
        power, ks = momentum_function(u, s)
        expected = (w * sqrt(2pi) / L)^2 .* exp.(-w^2 .* ks .^ 2) ./ Δk
        @test power[1] ≈ expected[1] rtol = 1e-8
        @test power[2:15] ≈ 2 .* expected[2:15] rtol = 1e-6
    end
end

@testset "momentum_function(u, s, ds) - non-uniform sampling" begin
    # The trapezoid sum in t is spectrally accurate while the integrand (frequency m' of the bin, plus the Bessel tail of
    # exp(-i m' sin(2 pi t))) stays below the Nyquist frequency of the t grid, i.e. for the low bins; only bins 1:41 are checked.
    L = 2pi
    N = 512
    pts = bf_graded_circle_pts(N)
    @test !all(isapprox(pts.s[i+1] - pts.s[i], pts.s[2] - pts.s[1]; rtol=1e-3) for i in 1:N-1)   # genuinely non-uniform
    s, ds = pts.s, pts.ds
    low(power, skip) = power[[i for i in 1:41 if i != skip]]

    @testset "u = cos(m s): power 1/2 in bin m, spectrally accurate on the graded grid" begin
        for m in (1, 3, 7)
            power, ks = momentum_function(cos.(m .* s), s, ds)
            @test length(power) == length(ks) == N ÷ 2 + 1
            @test ks ≈ (0:N÷2) .* (2pi / L)
            @test power[m+1] ≈ 0.5 atol = 1e-12                       # |c_m|^2 doubled: 2 (1/2)^2
            @test maximum(abs, low(power, m + 1)) <= 1e-12
        end
    end

    @testset "constant u: only the DC bin, power c^2" begin
        power, _ = momentum_function(fill(1.5, N), s, ds)
        @test power[1] ≈ 2.25 atol = 1e-12
        @test maximum(abs, low(power, 1)) <= 1e-12
    end

    @testset "complex u = exp(i m s): one-sided power 1 in bin m, no doubling" begin
        power, _ = momentum_function(exp.(3im .* s), s, ds)
        @test power[4] ≈ 1 atol = 1e-12
        @test maximum(abs, low(power, 4)) <= 1e-12
    end

    @testset "uniform real grid delegates to the FFT method" begin
        u_pts = bf_circle_pts(Float64, 64)
        u = 1 .+ cos.(2 .* u_pts.s) .+ 0.3 .* sin.(5 .* u_pts.s)
        p3, k3 = momentum_function(u, u_pts.s, u_pts.ds)
        p2, k2 = momentum_function(u, u_pts.s)
        @test p3 == p2 && k3 == k2
    end

    @testset "uniform complex grid goes through the quadrature, same answer as the non-uniform formula" begin
        u_pts = bf_circle_pts(Float64, 64)
        power, ks = momentum_function(exp.(2im .* u_pts.s), u_pts.s, u_pts.ds)
        @test power[3] ≈ 1 atol = 1e-12
        @test ks ≈ (0:32)
    end

    @testset "length mismatch" begin
        @test_throws DimensionMismatch momentum_function(ones(5), collect(1.0:6.0), ones(6))
        @test_throws DimensionMismatch momentum_function(ones(6), collect(1.0:6.0), ones(5))
    end
end

@testset "momentum_function(::BIMEigenstate)" begin
    pts = bf_graded_circle_pts(512)
    u = cos.(3 .* pts.s)
    st = bf_bim_state(BF_J01, u, pts)
    ref, ks_ref = momentum_function(u, pts.s, pts.ds)

    @testset "one-argument form uses the non-uniform quadrature" begin
        power, ks = momentum_function(st)
        @test power == ref && ks == ks_ref
        @test power[4] ≈ 0.5 atol = 1e-12
        @test momentum_function(st; b=10.0, multithreaded=false) == (ref, ks_ref)
    end

    @testset "with a billiard and with a solver" begin
        @test momentum_function(st, BF_CIRCLE) == (ref, ks_ref)
        @test momentum_function(st, BF_CIRCLE, bf_SweepStub()) == (ref, ks_ref)
    end

    @testset "missing u is recomputed through the solver stub" begin
        st_missing = BIMEigenstate{Float64,Float64}(BF_J01, 0.0, nothing, 1e-9, pts, nothing, nothing)
        power, _ = momentum_function(st_missing, BF_CIRCLE, bf_SweepStub())
        @test power[1] ≈ BF_STUB_U^2 atol = 1e-12                       # constant u: only the DC bin
        @test maximum(abs, power[2:41]) <= 1e-12
    end

    @testset "a missing u without a solver throws instead of failing downstream" begin
        st_missing = BIMEigenstate{Float64,Float64}(BF_J01, 0.0, nothing, 1e-9, pts, nothing, nothing)
        @test_throws ArgumentError momentum_function(st_missing)
        @test_throws "no stored boundary normal derivative" momentum_function(st_missing)
        @test_throws ArgumentError momentum_function(st_missing; b=10.0, multithreaded=false)
    end

    @testset "a missing u with a billiard but no solver throws" begin
        st_missing = BIMEigenstate{Float64,Float64}(BF_J01, 0.0, nothing, 1e-9, pts, nothing, nothing)
        @test_throws ArgumentError momentum_function(st_missing, BF_CIRCLE)
        @test_throws "no stored boundary normal derivative" momentum_function(st_missing, BF_CIRCLE)
    end
end

@testset "momentum_function(::BasisEigenstate)" begin
    @testset "circle J0 mode: constant u has only the DC component" begin
        st = bf_j0_state()
        u, s, _ = boundary_function(st, BF_CIRCLE)
        power, ks = momentum_function(st, BF_CIRCLE)
        @test power[1] > 0
        @test maximum(power[2:end]) <= 1e-18 * power[1]
        # Parseval holds only to O(1/N): the unfolded nodes contain a zero-weight duplicate at every quarter junction, so
        # N * ds_mean = L (N + 1) / N differs from s[end] = L by one spacing and the two normalisations in the function differ.
        @test sum(power) * (ks[2] - ks[1]) ≈ sum(abs2, u) / length(u) rtol = 1e-2
        @test momentum_function(u, s) == (power, ks)                  # it is the composition of the two functions
    end

    @testset "circle J1 cos(theta) mode: concentrated at the angular momentum 1" begin
        st = bf_j1_state()
        power, ks = momentum_function(st, BF_CIRCLE)
        @test argmax(power) == 2
        @test ks[2] ≈ 1 rtol = 1e-2                                   # k_s = m / R with L = 2 pi
        # the unfolded grid carries one zero-weight node at each quarter junction, so the sampling is uniform only
        # up to O(ds); the leakage into the other bins is far below 1 percent
        @test power[2] / sum(power) > 0.99
    end

    @testset "keywords are forwarded" begin
        st = bf_j1_state()
        p_a, _ = momentum_function(st, BF_CIRCLE; b=3000.0)
        p_b, _ = momentum_function(st, BF_CIRCLE; multithreaded=false)
        @test length(p_a) > length(momentum_function(st, BF_CIRCLE)[1])
        @test argmax(p_a) == 2
        @test p_b ≈ momentum_function(st, BF_CIRCLE)[1]
    end
end

@testset "SpectralData wrappers - boundary_function and momentum_function" begin
    @testset "basis eigenstates" begin
        s0 = bf_j0_state(); s1 = bf_j1_state()
        data = SpectralData(BF_CIRCLE, nothing, [s0.k, s1.k], [1e-12, 1e-12], [false, false]; states=[s0, s1])

        u1, sv1, n1 = boundary_function(data, 2)
        @test (u1, sv1, n1) == boundary_function(s1, BF_CIRCLE)
        all_b = boundary_function(data)
        @test length(all_b) == 2 && all_b[1] == boundary_function(s0, BF_CIRCLE) && all_b[2] == (u1, sv1, n1)
        @test boundary_function(data, 1; b=3000.0)[1] == boundary_function(s0, BF_CIRCLE; b=3000.0)[1]

        @test momentum_function(data, 2) == momentum_function(s1, BF_CIRCLE)
        all_m = momentum_function(data)
        @test length(all_m) == 2 && all_m[1] == momentum_function(s0, BF_CIRCLE) && all_m[2] == momentum_function(s1, BF_CIRCLE)
    end

    @testset "BIM eigenstates go through the solver" begin
        pts = bf_graded_circle_pts(512)
        st = bf_bim_state(BF_J01, cos.(3 .* pts.s), pts)
        data = SpectralData(BF_CIRCLE, bf_SweepStub(), [BF_J01], [1e-9], [false]; states=[st])
        @test boundary_function(data, 1) == (st.u, pts.s, 1.25)
        @test boundary_function(data) == [(st.u, pts.s, 1.25)]
        @test momentum_function(data, 1) == momentum_function(st)
        @test momentum_function(data) == [momentum_function(st)]
    end

    @testset "data without stored states" begin
        data = SpectralData(BF_CIRCLE, nothing, [3.0], [1e-12], [false])
        @test_throws ArgumentError boundary_function(data)
        @test_throws ArgumentError boundary_function(data, 1)
        @test_throws ArgumentError momentum_function(data)
        @test_throws ArgumentError momentum_function(data, 1)
    end
end
