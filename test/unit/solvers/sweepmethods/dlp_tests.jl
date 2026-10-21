# Unit tests for
#   src/solvers/sweepmethods/dlp.jl   (DoubleLayerPotentialSolver: evaluate_points, boundary_matrix_size, construct_matrices,
#                                      and through them _dlp_fredholm_full!, _dlp_fredholm_reduced!, _dlp_kernel_entry)
#   src/solvers/taylor/bim_kernels_recurrence.jl   (_construct_matrices_multi_k_taylor, used only as the "Taylor matrix" the
#                                      direct DLP matrix is compared with)
# Matrix level only: no eigen solve, no tension, no search for wavenumbers. The checks are
#   * evaluate_points: node count N = max(min_pts, round(k L ppw / 2 pi)) rounded up to a symmetry compatible multiple, array
#     lengths, midpoint parameters, the uniform and the corner graded Kress parametrization, arc length, the Float32 type;
#   * construct_matrices on the unit circle against the closed form A = I - D of the Kress-split kernel
#         D_ij = R_ij L1_ij + h L2_ij,   L1 = (k r / 4 pi) J1(k r),
#         L2 = -(i k r / 4) H1(k r) - L1 log(4 sin^2((t_i - t_j) / 2)),   D_ii = -h / 2 pi,   h = 2 pi / N,
#     with the explicit Kress matrix R_ij = -(2 pi / n) sum_{m<n} cos(m (i - j) pi / n) / m - (pi / n^2) (-1)^(i-j), n = N / 2;
#   * the straight-edge identity of a rectangle: nodes on one edge do not couple, since n_j . (x_i - x_j) = 0 there;
#   * the direct matrix against the analytic Taylor matrix, full and symmetry reduced, smooth and graded;
#   * the symmetry reduced matrix against the orbit-summed columns of the full matrix on the same nodes (X, Y, XY, inferred
#     rectangle reflections, C3 and Star rotations); the orbits come from BilliardGeometry.symmetry_index_orbits and their
#     geometry is verified on the nodes;
#   * boundary_matrix_size and threaded = serial assembly.
# Helper definitions use the prefix `dlp_`.

using LinearAlgebra: norm, diag
using SpecialFunctions: besselj1, besselh
using StaticArrays: SVector
using QuantumBilliards: _construct_matrices_multi_k_taylor, _is_nontrivial_dlp_grading

# Explicit Kress matrix of the periodic kernel log(4 sin^2((t - tau) / 2)) for the N = 2n midpoint nodes.
function dlp_kress_matrix(N::Int)
    n = N ÷ 2
    Rj(j) = -(2π / n) * sum(cos(m * j * π / n) / m for m in 1:n-1) - (π / n^2) * (-1)^j
    return [Rj(i - j) for i in 1:N, j in 1:N]
end

# Closed form of A = I - D for the unit circle at the angle midpoint nodes of `pts`.
function dlp_circle_matrix(pts, k)
    N = length(pts)
    h = 2π / N
    R = dlp_kress_matrix(N)
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
function dlp_orbit_sum(Afull, orbits)
    m = fundamental_size(orbits)
    fund = orbits.fundamental_indices
    E = zeros(ComplexF64, m, m)
    for j in eachindex(orbits.orbit_of), a in 1:m
        E[a, orbits.orbit_of[j]] += orbits.phase[j] * Afull[fund[a], j]
    end
    return E
end

# True if for every orbit the physical nodes of its images are exactly the images of the fundamental node under `maps`.
function dlp_orbit_matches(pts, orbits, maps)
    ng = orbit_size(orbits)
    length(maps) == ng || return false
    for a in 1:fundamental_size(orbits)
        x = pts.xy[orbits.fundamental_indices[a]]
        images = [pts.xy[orbits.fund_to_full[g, a]] for g in 1:ng]
        all(f -> any(q -> norm(q - f(x)) < 1e-10, images), maps) || return false
    end
    return true
end

dlp_rotate(x, φ) = SVector(cos(φ) * x[1] - sin(φ) * x[2], sin(φ) * x[1] + cos(φ) * x[2])

# Edge of the rectangle RectangleBilliard(1.0, 0.6) (x = 1, x = -1, y = 0.6, y = -0.6) carrying a node.
dlp_edge(p) = abs(p[1] - 1) < 1e-9 ? 1 : abs(p[1] + 1) < 1e-9 ? 2 : abs(p[2] - 0.6) < 1e-9 ? 3 : 4

@testset "evaluate_points: unit circle, smooth periodic grading" begin
    circle = CircleBilliard(1.0)
    solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=16)
    pts = evaluate_points(solver, circle, 10.0)
    N = length(pts)
    h = 2π / N
    θ = [(j - 1 / 2) * h for j in 1:N]

    @testset "node count and array lengths" begin
        @test pts isa BoundaryPoints{Float64}
        @test N == 50                                                   # round(k L ppw / 2 pi) = round(10 * 2 pi * 5 / 2 pi)
        @test length(evaluate_points(solver, circle, 3.0)) == 16        # below min_pts the minimum is used
        @test length(evaluate_points(DoubleLayerPotentialSolver(5.0, circle; min_pts=33), circle, 3.0)) == 34   # odd counts are rounded up to even
        @test length(evaluate_points(DoubleLayerPotentialSolver(10.0, circle; min_pts=16), circle, 10.0)) == 100  # N grows with the scaling factor
        @test all(==(N), (length(pts.xy), length(pts.tangent), length(pts.tangent_2), length(pts.ts), length(pts.tphys),
                          length(pts.ws), length(pts.ws_der), length(pts.s), length(pts.ds), length(pts.normal)))
        @test boundary_matrix_size(pts) == N
    end

    @testset "midpoint parameters and weights" begin
        @test pts.ts ≈ θ
        @test pts.tphys == pts.ts                                       # no grading map: physical parameter equals sigma
        @test all(≈(h), pts.ws)
        @test sum(pts.ws) ≈ 2π
        @test all(==(1.0), pts.ws_der)                                  # dt / dsigma = 1
        @test !_is_nontrivial_dlp_grading(pts)
    end

    @testset "geometry of the nodes" begin
        @test pts.xy ≈ [SVector(cos(t), sin(t)) for t in θ]
        @test pts.tangent ≈ [SVector(-sin(t), cos(t)) for t in θ]
        @test pts.tangent_2 ≈ [SVector(-cos(t), -sin(t)) for t in θ]
        @test pts.normal ≈ pts.xy                                       # outward normal of the unit circle
        @test all(≈(h), pts.ds)
        @test sum(pts.ds) ≈ 2π
        @test pts.s ≈ θ                                                 # arc length from the start of the curve equals the angle
    end

    @testset "Float32 solver gives Float32 points" begin
        pts32 = evaluate_points(DoubleLayerPotentialSolver(5f0, circle; min_pts=16), circle, 3f0)
        @test pts32 isa BoundaryPoints{Float32}
        @test length(pts32) == 16
        @test eltype(pts32.ws) == Float32
    end
end

@testset "evaluate_points: stadium arc length" begin
    stadium = StadiumBilliard(0.3)
    L = sum(c.length for c in full_boundary(stadium))
    solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32)
    pts = evaluate_points(solver, stadium, 8.0)
    N0 = max(32, round(Int, 8.0 * L * 5.0 / (2π)))

    @test length(pts) == 2 * cld(N0, 2)
    @test N0 > 32                                                       # the k-dependent count, not min_pts, is tested
    @test sum(pts.ds) ≈ L                                               # constant speed L / 2 pi, so the midpoint rule is exact
    @test all(≈(L / length(pts)), pts.ds)
    @test issorted(pts.s; lt=<=)                                        # strictly increasing arc length along the full boundary
    @test 0 < first(pts.s) && last(pts.s) < L
    @test pts.s ≈ [L * (j - 1 / 2) / length(pts) for j in 1:length(pts)]
end

@testset "evaluate_points: node count with symmetry" begin
    circle = CircleBilliard(1.0)
    c3 = C3Billiard(0.02)
    rect = RectangleBilliard(1.0, 0.6)
    stadium = StadiumBilliard(0.3)

    @testset "XY reflection: multiple of four" begin
        solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=33, symmetry=XYAxisReflection(), sym_characters=(1, 1))
        pts = evaluate_points(solver, circle, 3.0)
        @test length(pts) == 36
        @test boundary_matrix_size(solver, pts) == 9
    end

    @testset "C3 rotation: multiple of six" begin
        solver = DoubleLayerPotentialSolver(5.0, c3; min_pts=33, sym_characters=(1,))
        pts = evaluate_points(solver, c3, 3.0)
        @test length(pts) == 36                                         # lcm(2, 3) = 6
        @test boundary_matrix_size(solver, pts) == 12
    end

    @testset "inferred rectangle reflections" begin
        solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=33, sym_characters=(1, 1))
        pts = evaluate_points(solver, rect, 3.0)
        @test length(pts) == 36
        @test boundary_matrix_size(solver, pts) == 9
    end

    @testset "single reflection: even count, half-size matrix" begin
        solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=33, symmetry=YAxisReflection(), sym_characters=(1,))
        pts = evaluate_points(solver, stadium, 3.0)
        @test length(pts) % 2 == 0 && length(pts) >= 33
        @test boundary_matrix_size(solver, pts) == length(pts) ÷ 2
    end

    @testset "the whole boundary is discretized, not only the fundamental domain" begin
        solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=8, symmetry=YAxisReflection(), sym_characters=(1,))
        full = DoubleLayerPotentialSolver(5.0, stadium; min_pts=8)
        @test evaluate_points(solver, stadium, 8.0).xy == evaluate_points(full, stadium, 8.0).xy
        @test sum(evaluate_points(solver, stadium, 8.0).ds) ≈ sum(c.length for c in full_boundary(stadium))
    end
end

@testset "evaluate_points: corner graded Kress parametrization of a rectangle" begin
    rect = RectangleBilliard(1.0, 0.6)
    perimeter = 6.4
    solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
    pts = evaluate_points(solver, rect, 3.0)
    N = length(pts)
    h = 2π / N

    @test N == 32
    @test _is_nontrivial_dlp_grading(pts)
    @test pts.ts ≈ [(j - 1 / 2) * h for j in 1:N]                       # uniform quadrature variable sigma
    @test all(≈(h), pts.ws)
    @test sum(pts.ws) ≈ 2π
    @test all(>(0), pts.ws_der)                                         # dt / dsigma > 0
    @test issorted(pts.tphys; lt=<=) && 0 < first(pts.tphys) && last(pts.tphys) < 2π
    @test maximum(abs.(pts.tphys .- pts.ts)) > 0.05                     # the physical parameter differs from sigma
    @test norm.(pts.tangent) ./ pts.ws_der ≈ fill(perimeter / (2π), N)  # chain rule: |d gamma / d sigma| = speed(t) dt / d sigma
    @test minimum(diff(pts.tphys)) < 0.2 * (2π / N)                     # nodes cluster near the corners
    @test sum(pts.ds) ≈ perimeter rtol = 1e-2

    @testset "a smooth boundary has no corners and falls back to the uniform nodes" begin
        circle = CircleBilliard(1.0)
        graded = evaluate_points(DoubleLayerPotentialSolver(5.0, circle; min_pts=32, grading=GlobalCornerGrading()), circle, 3.0)
        smooth = evaluate_points(DoubleLayerPotentialSolver(5.0, circle; min_pts=32), circle, 3.0)
        @test graded.xy == smooth.xy
        @test graded.ws_der == smooth.ws_der
        @test !_is_nontrivial_dlp_grading(graded)
    end
end

@testset "construct_matrices: unit circle against the closed form" begin
    circle = CircleBilliard(1.0)
    solver = DoubleLayerPotentialSolver(5.0, circle; min_pts=24)
    k = 3.0
    pts = evaluate_points(solver, circle, k)
    N = length(pts)
    A = construct_matrices(solver, pts, k)

    @testset "Kress matrix of the helper equals BilliardGeometry.kress_R!" begin
        R = zeros(N, N)
        kress_R!(R)
        @test R ≈ dlp_kress_matrix(N) atol = 1e-12
    end

    @testset "all entries, real k" begin
        @test N == 24
        @test A isa Matrix{ComplexF64}
        @test size(A) == (N, N) == (boundary_matrix_size(solver, pts), boundary_matrix_size(solver, pts))
        @test isapprox(A, dlp_circle_matrix(pts, k); atol=1e-12, rtol=0)
    end

    @testset "diagonal is 1 + 1 / N" begin
        @test all(≈(1 + 1 / N), diag(A))                                # curvature term w kappa = h / 2 pi
    end

    @testset "circulant structure" begin
        @test A[2:end, 2:end] ≈ A[1:end-1, 1:end-1] atol = 1e-12        # D_ij depends on i - j only
        @test A ≈ transpose(A) atol = 1e-12                             # and is even in i - j
    end

    @testset "complex wavenumber" begin
        kc = 2.5 + 0.4im
        @test isapprox(construct_matrices(solver, pts, kc), dlp_circle_matrix(pts, kc); atol=1e-11, rtol=0)
    end

    @testset "output type follows the solver, not the wavenumber" begin
        @test eltype(construct_matrices(solver, pts, 3f0)) == ComplexF64
        solver32 = DoubleLayerPotentialSolver(5f0, circle; min_pts=24)
        pts32 = evaluate_points(solver32, circle, 3f0)
        A32 = construct_matrices(solver32, pts32, 3f0)
        @test A32 isa Matrix{ComplexF32}
        @test A32 ≈ A atol = 1e-5
    end
end

@testset "construct_matrices: straight edges of a rectangle do not couple" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0

    @testset "smooth periodic grading" begin
        solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=40)
        pts = evaluate_points(solver, rect, k)
        A = construct_matrices(solver, pts, k)
        edge = dlp_edge.(pts.xy)
        same = [edge[i] == edge[j] && i != j for i in eachindex(edge), j in eachindex(edge)]
        @test maximum(abs.(A[same])) < 1e-12                            # zero curvature, zero normal projection along an edge
        @test maximum(abs.(diag(A) .- 1)) < 1e-12
        @test maximum(abs.(A[edge .!= edge'])) > 1e-3                   # different edges do couple
    end

    @testset "corner graded parametrization" begin
        solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=40, grading=GlobalCornerGrading())
        pts = evaluate_points(solver, rect, k)
        A = construct_matrices(solver, pts, k)
        edge = dlp_edge.(pts.xy)
        same = [edge[i] == edge[j] && i != j for i in eachindex(edge), j in eachindex(edge)]
        @test _is_nontrivial_dlp_grading(pts)
        @test maximum(abs.(A[same])) < 1e-12
        @test maximum(abs.(diag(A) .- 1)) < 1e-12
        @test maximum(abs.(A[edge .!= edge'])) > 1e-3
    end
end

@testset "construct_matrices: direct matrix against the Taylor matrix" begin
    k0 = 3.0
    ks = [k0 + 0.02, k0 - 0.01, k0 + 0.01im]

    @testset "stadium, full matrix, smooth grading" begin
        stadium = StadiumBilliard(0.3)
        solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32)
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
        solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
        pts = evaluate_points(solver, rect, k0)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        @test _is_nontrivial_dlp_grading(pts)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)
    end

    @testset "stadium, Y reflection reduced" begin
        stadium = StadiumBilliard(0.3)
        solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(-1,))
        pts = evaluate_points(solver, stadium, k0)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        m = boundary_matrix_size(solver, pts)
        @test all(M -> size(M) == (m, m), T)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)
    end

    @testset "C3 rotation reduced" begin
        c3 = C3Billiard(0.02)
        solver = DoubleLayerPotentialSolver(5.0, c3; min_pts=36, sym_characters=(1,))
        pts = evaluate_points(solver, c3, k0)
        T = _construct_matrices_multi_k_taylor(solver, pts, k0, ks, 10)
        @test isapprox(T[1], construct_matrices(solver, pts, ks[1]); atol=1e-12, rtol=0)
        @test isapprox(T[3], construct_matrices(solver, pts, ks[3]); atol=1e-12, rtol=0)
    end

    @testset "the full-boundary Taylor builder rejects a symmetry solver" begin
        stadium = StadiumBilliard(0.3)
        solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(1,))
        pts = evaluate_points(solver, stadium, k0)
        @test_throws ArgumentError QuantumBilliards._construct_matrices_multi_k_taylor_full(solver, pts, k0, ks, 4)
    end
end

@testset "boundary_matrix_size" begin
    stadium = StadiumBilliard(0.3)
    c3 = C3Billiard(0.02)
    k = 3.0

    @testset "no symmetry: number of nodes" begin
        solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32)
        pts = evaluate_points(solver, stadium, k)
        @test boundary_matrix_size(solver, pts) == length(pts) == boundary_matrix_size(pts)
    end

    @testset "reflections: N / 2 and N / 4" begin
        sx = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=XAxisReflection(), sym_characters=(1,))
        pts = evaluate_points(sx, stadium, k)
        @test boundary_matrix_size(sx, pts) == length(pts) ÷ 2
        sxy = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(1, 1))
        @test boundary_matrix_size(sxy, pts) == length(pts) ÷ 4
    end

    @testset "C3 rotation: N / 3 for every sector" begin
        s0 = DoubleLayerPotentialSolver(5.0, c3; min_pts=36, sym_characters=(0,))
        s1 = DoubleLayerPotentialSolver(5.0, c3; min_pts=36, sym_characters=(1,))
        pts = evaluate_points(s0, c3, k)
        @test boundary_matrix_size(s0, pts) == length(pts) ÷ 3
        @test boundary_matrix_size(s1, pts) == length(pts) ÷ 3
    end

    @testset "agrees with the size of the assembled matrix" begin
        sy = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(1,))
        pts = evaluate_points(sy, stadium, k)
        m = boundary_matrix_size(sy, pts)
        @test size(construct_matrices(sy, pts, k)) == (m, m)
    end
end

@testset "reduced matrix equals orbit-summed full matrix: stadium, X reflection (odd)" begin
    stadium = StadiumBilliard(0.3)
    k = 3.0
    full = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32)
    odd = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=XAxisReflection(), sym_characters=(-1,))
    even = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=XAxisReflection(), sym_characters=(1,))
    pts = evaluate_points(odd, stadium, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, stadium, N, XAxisReflection(), -1.0 + 0im)
    A_odd = construct_matrices(odd, pts, k)

    @test size(A_odd) == (N ÷ 2, N ÷ 2)
    @test dlp_orbit_matches(pts, orbits, (x -> x, x -> SVector(x[1], -x[2])))
    @test isapprox(A_odd, dlp_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
    @test norm(A_odd - construct_matrices(even, pts, k)) > 1e-3         # the character selects the sector
end

@testset "reduced matrix equals orbit-summed full matrix: stadium, Y reflection (even)" begin
    stadium = StadiumBilliard(0.3)
    k = 3.0
    full = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32)
    red = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=YAxisReflection(), sym_characters=(1,))
    pts = evaluate_points(red, stadium, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, stadium, N, YAxisReflection(), 1.0 + 0im)

    @test dlp_orbit_matches(pts, orbits, (x -> x, x -> SVector(-x[1], x[2])))
    @test isapprox(construct_matrices(red, pts, k), dlp_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: stadium, XY reflection" begin
    stadium = StadiumBilliard(0.3)
    k = 3.0
    full = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32)
    red = DoubleLayerPotentialSolver(5.0, stadium; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(-1, 1))
    pts = evaluate_points(red, stadium, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, stadium, N, XYAxisReflection(), -1.0 + 0im, 1.0 + 0im)
    A_red = construct_matrices(red, pts, k)

    @test size(A_red) == (N ÷ 4, N ÷ 4)
    @test dlp_orbit_matches(pts, orbits, (x -> x, x -> SVector(x[1], -x[2]), x -> SVector(-x[1], x[2]), x -> -x))
    @test isapprox(A_red, dlp_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: rectangle, reflections inferred from the billiard" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    full = DoubleLayerPotentialSolver(5.0, rect; min_pts=32)
    red = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, sym_characters=(1, -1))      # (Y, X) generators of the rectangle
    pts = evaluate_points(red, rect, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, rect, N, CompositeReflection([YAxisReflection(), XAxisReflection()]), ComplexF64[1, -1])

    @test red.symmetry isa CompositeReflection
    @test dlp_orbit_matches(pts, orbits, (x -> x, x -> SVector(-x[1], x[2]), x -> SVector(x[1], -x[2]), x -> -x))
    @test isapprox(construct_matrices(red, pts, k), dlp_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: rectangle, corner graded nodes" begin
    rect = RectangleBilliard(1.0, 0.6)
    k = 3.0
    full = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading())
    red = DoubleLayerPotentialSolver(5.0, rect; min_pts=32, grading=GlobalCornerGrading(), sym_characters=(1, -1))
    pts = evaluate_points(red, rect, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, rect, N, CompositeReflection([YAxisReflection(), XAxisReflection()]), ComplexF64[1, -1])

    @test _is_nontrivial_dlp_grading(pts)
    @test dlp_orbit_matches(pts, orbits, (x -> x, x -> SVector(-x[1], x[2]), x -> SVector(x[1], -x[2]), x -> -x))
    @test isapprox(construct_matrices(red, pts, k), dlp_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: C3 billiard, rotation sector 1" begin
    c3 = C3Billiard(0.02)
    k = 3.0
    full = DoubleLayerPotentialSolver(5.0, c3; min_pts=36)
    red = DoubleLayerPotentialSolver(5.0, c3; min_pts=36, sym_characters=(1,))
    pts = evaluate_points(red, c3, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, c3, N, get_symmetries(c3)[1], 1)
    A_red = construct_matrices(red, pts, k)

    @test red.symmetry isa NFoldRotation
    @test size(A_red) == (N ÷ 3, N ÷ 3)
    @test any(!isreal, orbits.phase)                                    # sector 1 carries the cube roots of unity
    @test dlp_orbit_matches(pts, orbits, (x -> x, x -> dlp_rotate(x, 2π / 3), x -> dlp_rotate(x, 4π / 3)))
    @test isapprox(A_red, dlp_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix equals orbit-summed full matrix: Star billiard, rotation sector 2" begin
    star = StarBilliard(1.0, 0.05, 3)
    k = 3.0
    full = DoubleLayerPotentialSolver(5.0, star; min_pts=36)
    red = DoubleLayerPotentialSolver(5.0, star; min_pts=36, sym_characters=(2,))
    pts = evaluate_points(red, star, k)
    N = length(pts)
    orbits = symmetry_index_orbits(Float64, star, N, get_symmetries(star)[1], 2)

    @test dlp_orbit_matches(pts, orbits, (x -> x, x -> dlp_rotate(x, 2π / 3), x -> dlp_rotate(x, 4π / 3)))
    @test isapprox(construct_matrices(red, pts, k), dlp_orbit_sum(construct_matrices(full, pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "reduced matrix of the unit circle against the closed form" begin
    circle = CircleBilliard(1.0)
    k = 3.0
    red = DoubleLayerPotentialSolver(5.0, circle; min_pts=32, symmetry=XYAxisReflection(), sym_characters=(1, -1))
    pts = evaluate_points(red, circle, k)
    orbits = symmetry_index_orbits(Float64, circle, length(pts), XYAxisReflection(), 1.0 + 0im, -1.0 + 0im)

    @test length(pts) == 32
    @test isapprox(construct_matrices(red, pts, k), dlp_orbit_sum(dlp_circle_matrix(pts, k), orbits); atol=1e-12, rtol=0)
end

@testset "threaded assembly equals serial assembly" begin
    k = 3.0

    @testset "full matrix, smooth grading" begin
        stadium = StadiumBilliard(0.3)
        solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=64)
        pts = evaluate_points(solver, stadium, k)
        @test length(pts) >= 32                                         # large enough for the threaded loop to be used
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end

    @testset "full matrix, corner graded" begin
        rect = RectangleBilliard(1.0, 0.6)
        solver = DoubleLayerPotentialSolver(5.0, rect; min_pts=64, grading=GlobalCornerGrading())
        pts = evaluate_points(solver, rect, k)
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end

    @testset "Y reflection reduced matrix" begin
        stadium = StadiumBilliard(0.3)
        solver = DoubleLayerPotentialSolver(5.0, stadium; min_pts=64, symmetry=YAxisReflection(), sym_characters=(-1,))
        pts = evaluate_points(solver, stadium, k)
        @test boundary_matrix_size(solver, pts) >= 32
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end

    @testset "C3 rotation reduced matrix" begin
        c3 = C3Billiard(0.02)
        solver = DoubleLayerPotentialSolver(5.0, c3; min_pts=96, sym_characters=(1,))
        pts = evaluate_points(solver, c3, k)
        @test boundary_matrix_size(solver, pts) >= 32
        @test construct_matrices(solver, pts, k; multithreaded=true) == construct_matrices(solver, pts, k; multithreaded=false)
    end
end
