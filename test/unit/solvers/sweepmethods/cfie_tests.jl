# Unit tests for
#   src/solvers/sweepmethods/cfie.jl   (CombinedFieldIntegralEquationSolver: evaluate_points, boundary_matrix_size,
#                                       construct_matrices, and through them _cfie_fredholm_full!, _cfie_fredholm_reduced!,
#                                       _cfie_kernel_entry)
#   src/solvers/taylor/bim_kernels_recurrence.jl   (_construct_matrices_multi_k_taylor, used only as the "Taylor matrix" the
#                                       direct CFIE matrix is compared with)
# Matrix level only: no eigen solve, no tension, no search for wavenumbers. The CFIE Fredholm matrix is A = I - (D + i k S).
# The checks are
#   * evaluate_points: the same nodes as the DLP solver, node counts, symmetry compatible counts;
#   * the unit circle against the closed form of both kernels (Kress split, explicit Kress matrix R_ij, h = 2 pi / N):
#         D_ij = R_ij L1_ij + h L2_ij,   L1 = (k r / 4 pi) J1(k r),   L2 = -(i k r / 4) H1(k r) - L1 lt_ij,   D_ii = -h / 2 pi,
#         S_ij = (R_ij M1_ij + w_j M2_ij) v_j,   M1 = -J0(k r) / 2 pi,   M2 = (i / 2) H0(k r) - M1 lt_ij,
#         S_ii = R_ii (-v_i / 2 pi) + w_i v_i ((i / 2 - gamma / pi) - log(k^2 v_i^2 / 4) / 2 pi),
#     with lt_ij = log(4 sin^2((t_i - t_j) / 2)), v the speed |gamma'(t)|, w the quadrature weights, gamma Euler's constant;
#   * the alpha combination: on any nodes, A_DLP - A_CFIE = i k S with the closed-form S above (stadium, graded rectangle,
#     complex k), also after the symmetry reduction;
#   * the direct matrix against the analytic Taylor matrix, full and symmetry reduced, smooth and graded;
#   * the symmetry reduced matrix against the orbit-summed columns of the full matrix on the same nodes (X, Y, XY, inferred
#     rectangle reflections, C3 and Star rotations), orbits from BilliardGeometry.symmetry_index_orbits;
#   * boundary_matrix_size and threaded = serial assembly.
# Helper definitions use the prefix `cf_`.

using LinearAlgebra: norm, diag
using SpecialFunctions: besselj0, besselj1, besselh
using StaticArrays: SVector
using QuantumBilliards: _construct_matrices_multi_k_taylor, _is_nontrivial_dlp_grading

# Explicit Kress matrix of the periodic kernel log(4 sin^2((t - tau) / 2)) for the N = 2n midpoint nodes.
function cf_kress_matrix(N::Int)
    n = N ÷ 2
    Rj(j) = -(2π / n) * sum(cos(m * j * π / n) / m for m in 1:n-1) - (π / n^2) * (-1)^j
    return [Rj(i - j) for i in 1:N, j in 1:N]
end

# Closed form of the Kress-split single-layer matrix S (without the factor i k) on arbitrary periodic nodes `pts`.
function cf_slp_matrix(pts, k)
    N = length(pts)
    R = cf_kress_matrix(N)
    S = zeros(ComplexF64, N, N)
    for j in 1:N, i in 1:N
        vj = norm(pts.tangent[j])
        if i == j
            m2 = ((im / 2 - MathConstants.eulergamma / π) - log(k^2 * vj^2 / 4) / (2π)) * vj
            S[i, i] = R[i, i] * (-vj / (2π)) + pts.ws[i] * m2
        else
            r = norm(pts.xy[i] - pts.xy[j])
            M1 = -besselj0(k * r) / (2π) * vj
            M2 = im / 2 * besselh(0, 1, k * r) * vj - M1 * log(4 * sin((pts.ts[i] - pts.ts[j]) / 2)^2)
            S[i, j] = R[i, j] * M1 + pts.ws[j] * M2
        end
    end
    return S
end

# Closed form of the DLP matrix A = I - D of the unit circle at the angle midpoint nodes of `pts`.
function cf_circle_dlp_matrix(pts, k)
    N = length(pts)
    h = 2π / N
    R = cf_kress_matrix(N)
    A = zeros(ComplexF64, N, N)
    for j in 1:N, i in 1:N
        if i == j
            A[i, j] = 1 + h / (2π)
        else
            r = norm(pts.xy[i] - pts.xy[j])
            L1 = k * r / (4π) * besselj1(k * r)
            L2 = -im * k * r / 4 * besselh(1, 1, k * r) - L1 * log(4 * sin((pts.ts[i] - pts.ts[j]) / 2)^2)
            A[i, j] = -(R[i, j] * L1 + h * L2)
        end
    end
    return A
end

# Orbit sum of the columns of a full matrix: E[a, b] = sum over the nodes j of orbit b of phase[j] A[fund[a], j].
function cf_orbit_sum(Afull, orbits)
    m = fundamental_size(orbits)
    fund = orbits.fundamental_indices
    E = zeros(ComplexF64, m, m)
    for j in eachindex(orbits.orbit_of), a in 1:m
        E[a, orbits.orbit_of[j]] += orbits.phase[j] * Afull[fund[a], j]
    end
    return E
end

# True if for every orbit the physical nodes of its images are exactly the images of the fundamental node under `maps`.
function cf_orbit_matches(pts, orbits, maps)
    ng = orbit_size(orbits)
    length(maps) == ng || return false
    for a in 1:fundamental_size(orbits)
        x = pts.xy[orbits.fundamental_indices[a]]
        images = [pts.xy[orbits.fund_to_full[g, a]] for g in 1:ng]
        all(f -> any(q -> norm(q - f(x)) < 1e-10, images), maps) || return false
    end
    return true
end

cf_rotate(x, φ) = SVector(cos(φ) * x[1] - sin(φ) * x[2], sin(φ) * x[1] + cos(φ) * x[2])

@testset "evaluate_points: same nodes as the DLP solver" begin
    k = 10.0

    @testset "unit circle, smooth periodic grading" begin
        circle = CircleBilliard(1.0)
        cfie = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=16)
        dlp = DoubleLayerPotentialSolver(5.0, circle; min_pts=16)
        pts = evaluate_points(cfie, circle, k)
        ref = evaluate_points(dlp, circle, k)
        @test pts isa BoundaryPoints{Float64}
        @test length(pts) == 50                                         # round(k L ppw / 2 pi) = round(10 * 2 pi * 5 / 2 pi)
        @test pts.xy == ref.xy && pts.tangent == ref.tangent && pts.tangent_2 == ref.tangent_2
        @test pts.ts == ref.ts && pts.tphys == ref.tphys && pts.ws == ref.ws && pts.ws_der == ref.ws_der
        @test pts.s == ref.s && pts.ds == ref.ds
        @test all(≈(2π / 50), pts.ws) && all(==(1.0), pts.ws_der)
        @test pts.s ≈ [(j - 1 / 2) * 2π / 50 for j in 1:50]
    end

    @testset "rectangle, corner graded" begin
        rect = RectangleBilliard(1.0, 0.6)
        cfie = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
        dlp = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
        pts = evaluate_points(cfie, rect, 3.0)
        ref = evaluate_points(dlp, rect, 3.0)
        @test length(pts) == 32
        @test _is_nontrivial_dlp_grading(pts)
        @test pts.xy == ref.xy && pts.ws_der == ref.ws_der && pts.tangent == ref.tangent
        @test sum(pts.ws) ≈ 2π
        @test norm.(pts.tangent) ./ pts.ws_der ≈ fill(6.4 / (2π), 32)   # |d gamma / d sigma| = speed(t) dt / d sigma
    end

    @testset "node count: minimum, rounding to even, growth with the scaling factor" begin
        circle = CircleBilliard(1.0)
        @test length(evaluate_points(CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=16), circle, 3.0)) == 16
        @test length(evaluate_points(CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=33), circle, 3.0)) == 34
        @test length(evaluate_points(CombinedFieldIntegralEquationSolver(10.0, circle; min_pts=16), circle, 10.0)) == 100
    end

    @testset "stadium arc length along the whole boundary" begin
        stadium = StadiumBilliard(0.3)
        L = sum(c.length for c in full_boundary(stadium))
        pts = evaluate_points(CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32), stadium, 8.0)
        @test sum(pts.ds) ≈ L
        @test issorted(pts.s; lt=<=) && 0 < first(pts.s) && last(pts.s) < L
    end

    @testset "Float32 solver gives Float32 points" begin
        circle = CircleBilliard(1.0)
        pts32 = evaluate_points(CombinedFieldIntegralEquationSolver(5f0, circle; min_pts=16), circle, 3f0)
        @test pts32 isa BoundaryPoints{Float32}
        @test length(pts32) == 16
    end
end

@testset "evaluate_points: node count with symmetry" begin
    circle = CircleBilliard(1.0)
    c3 = C3Billiard(0.02)
    rect = RectangleBilliard(1.0, 0.6)
    stadium = StadiumBilliard(0.3)

    @testset "XY reflection: multiple of four" begin
        solver = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=33, symmetry=XYAxisReflection(), sym_characters=(1, 1))
        pts = evaluate_points(solver, circle, 3.0)
        @test length(pts) == 36
        @test boundary_matrix_size(solver, pts) == 9
    end

    @testset "C3 rotation: multiple of six" begin
        solver = CombinedFieldIntegralEquationSolver(5.0, c3; min_pts=33, sym_characters=(1,))
        pts = evaluate_points(solver, c3, 3.0)
        @test length(pts) == 36
        @test boundary_matrix_size(solver, pts) == 12
    end

    @testset "inferred rectangle reflections, corner graded" begin
        solver = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=33, grading=GlobalCornerGrading(), sym_characters=(1, 1))
        pts = evaluate_points(solver, rect, 3.0)
        @test length(pts) == 36
        @test boundary_matrix_size(solver, pts) == 9
    end

    @testset "the whole boundary is discretized, not only the fundamental domain" begin
        sym = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=8, symmetry=YAxisReflection(), sym_characters=(1,))
        full = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=8)
        @test evaluate_points(sym, stadium, 8.0).xy == evaluate_points(full, stadium, 8.0).xy
    end
end

@testset "construct_matrices: unit circle against the closed form" begin
    circle = CircleBilliard(1.0)
    solver = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=24)
    k = 3.0
    pts = evaluate_points(solver, circle, k)
    N = length(pts)
    A = construct_matrices(solver, pts, k)
    S = cf_slp_matrix(pts, k)

    @testset "Kress matrix of the helper equals BilliardGeometry.kress_R!" begin
        R = zeros(N, N)
        kress_R!(R)
        @test R ≈ cf_kress_matrix(N) atol = 1e-12
    end

    @testset "all entries, real k" begin
        @test N == 24
        @test A isa Matrix{ComplexF64}
        @test size(A) == (N, N) == (boundary_matrix_size(solver, pts), boundary_matrix_size(solver, pts))
        @test isapprox(A, cf_circle_dlp_matrix(pts, k) - im * k * S; atol=1e-12, rtol=0)
    end

    @testset "diagonal" begin
        h = 2π / N
        Sii = cf_kress_matrix(N)[1, 1] * (-1 / (2π)) + h * ((im / 2 - MathConstants.eulergamma / π) - log(k^2 / 4) / (2π))
        @test all(≈(1 + h / (2π) - im * k * Sii), diag(A))              # I - D_ii - i k S_ii on the unit circle (speed 1)
    end

    @testset "circulant structure" begin
        @test A[2:end, 2:end] ≈ A[1:end-1, 1:end-1] atol = 1e-12        # kernels depend on i - j only
        @test A ≈ transpose(A) atol = 1e-12
    end

    @testset "complex wavenumber" begin
        kc = 2.5 + 0.4im
        expected = cf_circle_dlp_matrix(pts, kc) - im * kc * cf_slp_matrix(pts, kc)
        @test isapprox(construct_matrices(solver, pts, kc), expected; atol=1e-11, rtol=0)
    end

    @testset "output type follows the solver, not the wavenumber" begin
        @test eltype(construct_matrices(solver, pts, 3f0)) == ComplexF64
        solver32 = CombinedFieldIntegralEquationSolver(5f0, circle; min_pts=24)
        pts32 = evaluate_points(solver32, circle, 3f0)
        A32 = construct_matrices(solver32, pts32, 3f0)
        @test A32 isa Matrix{ComplexF32}
        @test A32 ≈ A atol = 1e-4
    end
end

@testset "construct_matrices: CFIE = DLP - i k S (alpha combination)" begin
    k = 3.0

    @testset "stadium, smooth grading" begin
        stadium = StadiumBilliard(0.3)
        cfie = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32)
        dlp = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32)
        pts = evaluate_points(cfie, stadium, k)
        Δ = construct_matrices(dlp, pts, k) - construct_matrices(cfie, pts, k)
        @test isapprox(Δ, im * k * cf_slp_matrix(pts, k); atol=1e-12, rtol=0)
        @test Δ ≈ transpose(Δ) atol = 1e-12                             # S is symmetric on a constant-speed parametrization
        @test norm(Δ) > 1e-2                                            # the single-layer part is not negligible
    end

    @testset "stadium, complex wavenumber" begin
        stadium = StadiumBilliard(0.3)
        cfie = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32)
        dlp = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32)
        pts = evaluate_points(cfie, stadium, k)
        kc = 3.0 + 0.2im
        Δ = construct_matrices(dlp, pts, kc) - construct_matrices(cfie, pts, kc)
        @test isapprox(Δ, im * kc * cf_slp_matrix(pts, kc); atol=1e-11, rtol=0)
    end

    @testset "rectangle, corner graded nodes (position dependent speed)" begin
        rect = RectangleBilliard(1.0, 0.6)
        cfie = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
        dlp = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
        pts = evaluate_points(cfie, rect, k)
        Δ = construct_matrices(dlp, pts, k) - construct_matrices(cfie, pts, k)
        @test _is_nontrivial_dlp_grading(pts)
        @test isapprox(Δ, im * k * cf_slp_matrix(pts, k); atol=1e-12, rtol=0)
        @test norm(Δ - transpose(Δ)) > 1e-3                             # the grading speed v_j makes S non-symmetric
    end

    @testset "Y reflection reduced matrices" begin
        stadium = StadiumBilliard(0.3)
        cfie = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(-1,))
        dlp = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(-1,))
        pts = evaluate_points(cfie, stadium, k)
        orbits = symmetry_index_orbits(Float64, stadium, length(pts), YAxisReflection(), -1.0 + 0im)
        Δ = construct_matrices(dlp, pts, k) - construct_matrices(cfie, pts, k)
        @test isapprox(Δ, cf_orbit_sum(im * k * cf_slp_matrix(pts, k), orbits); atol=1e-12, rtol=0)
    end
end

@testset "construct_matrices: direct matrix against the Taylor matrix" begin
    k0 = 3.0
    ks = [k0 + 0.02, k0 - 0.01, k0 + 0.01im]

    @testset "stadium, full matrix, smooth grading" begin
        stadium = StadiumBilliard(0.3)
        solver = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32)
        pts = evaluate_points(solver, stadium, k0)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        @test length(T) == 3
        @test all(M -> M isa Matrix{ComplexF64} && size(M) == (length(pts), length(pts)), T)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[2], construct_matrices(solver, pts, ks[2]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)     # complex wavenumber

        @testset "degree 0 reproduces the matrix at k0 only" begin
            T0 = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 0)
            A0 = construct_matrices(solver, pts, k0)
            @test all(M -> isapprox(M, A0; atol=1e-12, rtol=0), T0)
            @test norm(T0[1] - construct_matrices(solver, pts, ks[1])) > 1e-4       # the truncation is visible
        end

        @testset "Beyn contour variant" begin
            R, nq = 0.02, 4
            Tc = _construct_matrices_multi_k_taylor(solver, pts, k0, R, nq, 10)
            kc = [k0 + R * cis(2π * (q - 1) / nq) for q in 1:nq]
            @test length(Tc) == nq
            @test all(q -> isapprox(Tc[q], construct_matrices(solver, pts, kc[q]); atol=1e-12, rtol=0), 1:nq)
        end
    end

    @testset "rectangle, full matrix, corner graded" begin
        rect = RectangleBilliard(1.0, 0.6)
        solver = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
        pts = evaluate_points(solver, rect, k0)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        @test _is_nontrivial_dlp_grading(pts)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)
    end

    @testset "stadium, Y reflection reduced" begin
        stadium = StadiumBilliard(0.3)
        solver = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(-1,))
        pts = evaluate_points(solver, stadium, k0)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        m = boundary_matrix_size(solver, pts)
        @test all(M -> size(M) == (m, m), T)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)
    end

    @testset "C3 rotation reduced" begin
        c3 = C3Billiard(0.02)
        solver = CombinedFieldIntegralEquationSolver(5.0, c3; min_pts=36, sym_characters=(1,))
        pts = evaluate_points(solver, c3, k0)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)
    end
end

@testset "boundary_matrix_size" begin
    stadium = StadiumBilliard(0.3)
    c3 = C3Billiard(0.02)
    k = 3.0

    @testset "no symmetry: number of nodes" begin
        solver = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32)
        pts = evaluate_points(solver, stadium, k)
        @test boundary_matrix_size(solver, pts) == length(pts) == boundary_matrix_size(pts)
    end

    @testset "reflections: N / 2 and N / 4" begin
        sx = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=XAxisReflection(), sym_characters=(1,))
        pts = evaluate_points(sx, stadium, k)
        @test boundary_matrix_size(sx, pts) == length(pts) ÷ 2
        sxy = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(1, 1))
        @test boundary_matrix_size(sxy, pts) == length(pts) ÷ 4
    end

    @testset "C3 rotation: N / 3 for every sector" begin
        s0 = CombinedFieldIntegralEquationSolver(5.0, c3; min_pts=36, sym_characters=(0,))
        s2 = CombinedFieldIntegralEquationSolver(5.0, c3; min_pts=36, sym_characters=(2,))
        pts = evaluate_points(s0, c3, k)
        @test boundary_matrix_size(s0, pts) == length(pts) ÷ 3
        @test boundary_matrix_size(s2, pts) == length(pts) ÷ 3
    end

    @testset "agrees with the size of the assembled matrix" begin
        sy = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(1,))
        pts = evaluate_points(sy, stadium, k)
        m = boundary_matrix_size(sy, pts)
        @test size(construct_matrices(sy, pts, k)) == (m, m)
    end
end

@testset "reduced matrix equals orbit-summed full matrix: stadium, X reflection (even)" begin
    stadium = StadiumBilliard(0.3)
    k = 3.0
    full = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32)
    even = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=XAxisReflection(), sym_characters=(1,))
    odd = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=XAxisReflection(), sym_characters=(-1,))
    pts = evaluate_points(even, stadium, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, stadium, N, XAxisReflection(), 1.0 + 0im)
    A_even = construct_matrices(even, pts, k)

    @test size(A_even) == (N ÷ 2, N ÷ 2)
    @test cf_orbit_matches(pts, orbits, (x -> x, x -> SVector(x[1], -x[2])))
    @test isapprox(A_even, cf_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
    @test norm(A_even - construct_matrices(odd, pts, k)) > 1e-3         # the character selects the sector
end

@testset "reduced matrix equals orbit-summed full matrix: stadium, Y reflection (odd)" begin
    stadium = StadiumBilliard(0.3)
    k = 3.0
    full = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32)
    red = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(-1,))
    pts = evaluate_points(red, stadium, k)
    orbits = symmetry_index_orbits(Float64, stadium, length(pts), YAxisReflection(), -1.0 + 0im)

    @test cf_orbit_matches(pts, orbits, (x -> x, x -> SVector(-x[1], x[2])))
    @test isapprox(construct_matrices(red, pts, k), cf_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: stadium, XY reflection" begin
    stadium = StadiumBilliard(0.3)
    k = 3.0
    full = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32)
    red = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(-1, 1))
    pts = evaluate_points(red, stadium, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, stadium, N, XYAxisReflection(), -1.0 + 0im, 1.0 + 0im)
    A_red = construct_matrices(red, pts, k)

    @test size(A_red) == (N ÷ 4, N ÷ 4)
    @test cf_orbit_matches(pts, orbits, (x -> x, x -> SVector(x[1], -x[2]), x -> SVector(-x[1], x[2]), x -> -x))
    @test isapprox(A_red, cf_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: rectangle, reflections inferred from the billiard" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    full = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=32)
    red = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=32, sym_characters=(-1, 1))      # (Y, X) generators of the rectangle
    pts = evaluate_points(red, rect, k)
    orbits = symmetry_index_orbits(Float64, rect, length(pts), CompositeReflection([YAxisReflection(), XAxisReflection()]), ComplexF64[-1, 1])

    @test red.symmetry isa CompositeReflection
    @test cf_orbit_matches(pts, orbits, (x -> x, x -> SVector(-x[1], x[2]), x -> SVector(x[1], -x[2]), x -> -x))
    @test isapprox(construct_matrices(red, pts, k), cf_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: rectangle, corner graded nodes" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    full = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
    red = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading(), sym_characters=(1, -1))
    pts = evaluate_points(red, rect, k)
    orbits = symmetry_index_orbits(Float64, rect, length(pts), CompositeReflection([YAxisReflection(), XAxisReflection()]), ComplexF64[1, -1])

    @test _is_nontrivial_dlp_grading(pts)
    @test cf_orbit_matches(pts, orbits, (x -> x, x -> SVector(-x[1], x[2]), x -> SVector(x[1], -x[2]), x -> -x))
    @test isapprox(construct_matrices(red, pts, k), cf_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: C3 billiard, rotation sector 2" begin
    c3 = C3Billiard(0.02)
    k = 3.0
    full = CombinedFieldIntegralEquationSolver(5.0, c3; min_pts=36)
    red = CombinedFieldIntegralEquationSolver(5.0, c3; min_pts=36, sym_characters=(2,))
    pts = evaluate_points(red, c3, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, c3, N, get_symmetries(c3)[1], 2)
    A_red = construct_matrices(red, pts, k)

    @test red.symmetry isa NFoldRotation
    @test size(A_red) == (N ÷ 3, N ÷ 3)
    @test any(!isreal, orbits.phase)
    @test cf_orbit_matches(pts, orbits, (x -> x, x -> cf_rotate(x, 2π / 3), x -> cf_rotate(x, 4π / 3)))
    @test isapprox(A_red, cf_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: Star billiard, rotation sector 1" begin
    star = StarBilliard(1.0, 0.05, 3)
    k = 3.0
    full = CombinedFieldIntegralEquationSolver(5.0, star; min_pts=36)
    red = CombinedFieldIntegralEquationSolver(5.0, star; min_pts=36, sym_characters=(1,))
    pts = evaluate_points(red, star, k)
    orbits = symmetry_index_orbits(Float64, star, length(pts), get_symmetries(star)[1], 1)

    @test cf_orbit_matches(pts, orbits, (x -> x, x -> cf_rotate(x, 2π / 3), x -> cf_rotate(x, 4π / 3)))
    @test isapprox(construct_matrices(red, pts, k), cf_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix of the unit circle against the closed form" begin
    circle = CircleBilliard(1.0)
    k = 3.0
    red = CombinedFieldIntegralEquationSolver(5.0, circle; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(1, -1))
    pts = evaluate_points(red, circle, k)
    orbits = symmetry_index_orbits(Float64, circle, length(pts), XYAxisReflection(), 1.0 + 0im, -1.0 + 0im)
    Afull = cf_circle_dlp_matrix(pts, k) - im * k * cf_slp_matrix(pts, k)

    @test length(pts) == 32
    @test isapprox(construct_matrices(red, pts, k), cf_orbit_sum(Afull, orbits); atol=1e-12, rtol=0)
end

@testset "threaded assembly equals serial assembly" begin
    k = 3.0

    @testset "full matrix, smooth grading" begin
        stadium = StadiumBilliard(0.3)
        solver = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=64)
        pts = evaluate_points(solver, stadium, k)
        @test length(pts) >= 32                                         # large enough for the threaded loop to be used
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end

    @testset "full matrix, corner graded" begin
        rect = RectangleBilliard(1.0, 0.6)
        solver = CombinedFieldIntegralEquationSolver(5.0, rect; min_pts=64, grading=GlobalCornerGrading())
        pts = evaluate_points(solver, rect, k)
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end

    @testset "Y reflection reduced matrix" begin
        stadium = StadiumBilliard(0.3)
        solver = CombinedFieldIntegralEquationSolver(5.0, stadium; min_pts=64, symmetry=YAxisReflection(), sym_characters=(-1,))
        pts = evaluate_points(solver, stadium, k)
        @test boundary_matrix_size(solver, pts) >= 32
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end

    @testset "C3 rotation reduced matrix" begin
        c3 = C3Billiard(0.02)
        solver = CombinedFieldIntegralEquationSolver(5.0, c3; min_pts=96, sym_characters=(1,))
        pts = evaluate_points(solver, c3, k)
        @test boundary_matrix_size(solver, pts) >= 32
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end
end
