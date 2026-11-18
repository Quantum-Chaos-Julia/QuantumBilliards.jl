# Integration tests for BeynSolver (src/solvers/acceleratedmethods/beyn.jl) through `compute_spectrum`.
# Parameters from the calibration log of scratchpad/test-suite-redesign-plan.md. Window edges are placed away
# from levels so that the level counts are stable. Checks per "Accelerated-solver testset shape": count equal to
# the reference, levels within SPECTRUM_ATOL_BIM, tension below TENSION_TOL_ACCEL, |Im k| below IMAG_TOL, no
# duplicates (for degenerate levels: exactly the analytic multiplicity).

# solver: Beyn contour integral
# kernel/basis: DLP, GlobalCornerGrading, ppw=5 (Taylor backend, the BeynSolver default)
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762 (tension 3e-15, ppw 6 and 10 agree; it is the only level in [5.5, 6.7]).
# functions to test: compute_spectrum
@testset "Beyn (DLP) - Full Triangle - Ground State" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    kernel = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 5.5, 6.7
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    k_ref = 6.065090986269762
    @test length(data.k) == 1
    @test isapprox(real(data.k[1]), k_ref; atol=SPECTRUM_ATOL_BIM)
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic; the window [3.0, 3.5] lies between j_{0,1}=2.4048 and j_{1,1}=3.8317 and holds no level
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Circle - empty window returns an empty spectrum" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    solver = BeynSolver(kernel; use_chebyshev=false, eigenvectors=true)
    @test isempty(circle_spectrum(1.0, 3.0, 3.5))
    data = compute_spectrum(solver, billiard, 3.0, 3.5; multithreaded=true, show_progress=false)
    @test length(data) == 0
    @test isempty(data.k) && isempty(data.ten) && isempty(data.k_im) && isempty(data.control)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, Bessel zeros j_{m,n}; the window [2, 8] holds 12 levels incl. the degenerate pairs
#   j_{1,1}=3.8317, j_{2,1}=5.1356, j_{3,1}=6.3802, j_{1,2}=7.0156, j_{4,1}=7.5883 (multiplicity 2 each)
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Circle - Spectrum incl. degenerate J1 pair" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 2.0, 8.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    ks_ref = circle_spectrum(1.0, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    # multiplicity: j_{1,1} appears exactly twice, j_{0,1} exactly once
    j11 = circle_spectrum(1.0, 3.8, 3.9)[1]
    j01 = circle_spectrum(1.0, 2.3, 2.5)[1]
    @test count(k -> abs(k - j11) < SPECTRUM_ATOL_BIM, ks) == 2
    @test count(k -> abs(k - j01) < SPECTRUM_ATOL_BIM, ks) == 1
    # no spurious clusters: every reference level is matched with its analytic multiplicity
    for kd in unique(round.(ks_ref; digits=8))
        mult = count(k -> abs(k - kd) < 1e-8, ks_ref)
        @test count(k -> abs(k - kd) < SPECTRUM_ATOL_BIM, ks) == mult
    end
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic, pi*sqrt(m^2/2^2 + n^2/1.4^2); the window [3, 10] holds 16 levels
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Rectangle - Spectrum k in [3, 10]" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 3.0, 10.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    ks_ref = rectangle_spectrum(2.0, 1.4, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    @test all(diff(ks) .> SPECTRUM_ATOL_BIM)
end

# solver: Beyn contour integral (composite, one CFIE component per boundary domain)
# kernel/basis: CFIE, SmoothPeriodicGrading, min_pts=64, ppw=10
# billiard: Annular (AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])), concentric, inner radius 0.3, outer radius 1; the nonzero center keeps the
#   unreduced construction (the origin-centered annulus is symmetric since B7); the spectrum is translation invariant
# symmetry: None
# reference: analytic roots of the annulus Bessel determinant (incl. multiplicity 2 for m > 0); window [3, 8] holds 9 levels
# functions to test: compute_spectrum
@testset "Beyn (CFIE composite) - Annular - Spectrum k in [3, 8]" begin
    billiard = AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])
    component() = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading(), min_pts=64)
    kernel = CompositeBIMSolver(component(), component())
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 3.0, 8.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    ks_ref = annulus_spectrum(0.3, 1.0, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    for kd in unique(round.(ks_ref; digits=8))
        mult = count(k -> abs(k - kd) < 1e-8, ks_ref)
        @test count(k -> abs(k - kd) < SPECTRUM_ATOL_BIM, ks) == mult
    end
end

# Left (adjoint) contour solve of BeynSolver, `solve_left_vectors` (Step 7 of scratchpad/boundary-function-spectrum-plan.md).
# One window per billiard is solved with the right moments (`solve_vectors`) and the left moments (`solve_left_vectors`) on the same
# boundary points and cached, so that both testsets (and later steps) reuse the contour solves. The left null vectors y of A(k)
# (y' A(k) = 0) give the boundary normal derivative u = y ./ ds up to a complex scale (no conjugation, as in `solve_state`), which is
# compared with the analytic u after the least-squares scaling. Windows: rectangle k0 = 6, dk = 2 (levels 5.2194, 5.4783, 6.5076,
# 6.6719, 6.9128; the nearest outside levels are 4.755 and 7.43); annulus k0 = 5, dk = 1.6 (levels 4.4124, 4.7058 x2, 5.4702 x2; the
# nearest outside levels are 3.9 and 6.49 for the pairs).
using LinearAlgebra: norm, dot, svdvals
using SpecialFunctions: besselj, bessely

const BEYN_LEFT_CACHE = Dict{Symbol,Any}()

function beyn_left_window(key::Symbol, billiard, kernel, k_pts, k0, dk)
    return get!(BEYN_LEFT_CACHE, key) do
        # res_tol = 1e-6: the residual is of the unnormalised vector and the right solve of the level 6.9128 sits at 1e-8
        solver = BeynSolver(kernel; use_chebyshev=false, res_tol=1e-6)
        pts = evaluate_points(kernel, billiard, k_pts)
        kr, resr, X = solve_vectors(solver, pts, k0, dk; multithreaded=false)
        kl, resl, Y = solve_left_vectors(solver, pts, k0, dk; multithreaded=false)
        (; pts, kr, resr, X, kl, resl, Y)
    end
end

function beyn_left_rectangle()
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    return beyn_left_window(:rectangle, billiard, kernel, 7.0, 6.0, 2.0)
end

function beyn_left_annulus()
    billiard = AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])
    component() = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading(), min_pts=64)
    return beyn_left_window(:annulus, billiard, CompositeBIMSolver(component(), component()), 5.8, 5.0, 1.6)
end

# Normal derivative of psi_{mn} of RectangleBilliard(a, b) at the boundary points.
function beyn_left_rectangle_u(pts, a, b, m, n)
    W, H = 2a, 2b
    return [begin
                x, y = pts.xy[i]; nx, ny = pts.normal[i]
                nx * (m * pi / W) * cos(m * pi * (x + a) / W) * sin(n * pi * (y + b) / H) +
                ny * (n * pi / H) * sin(m * pi * (x + a) / W) * cos(n * pi * (y + b) / H)
            end for i in eachindex(pts.xy)]
end

# Arclength-weighted relative residual of `u` after the best complex scaling onto `ref`.
function beyn_left_scaled_residual(u, ref, ds)
    w = sqrt.(ds)
    c = dot(w .* ref, w .* u) / dot(w .* ref, w .* ref)
    return norm(w .* (u .- c .* ref)) / norm(w .* u)
end

# solver: Beyn contour integral, left solve (solve_left_vectors)
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic, roots pi*sqrt(m^2/4 + n^2/1.96) in [5, 7] (5 levels), u = d_n psi_{mn}; cross-check against the right roots of the same contour
# functions to test: solve_left_vectors, _beyn_solve_core (side = :left), left residual ||A(k)' y||
@testset "Beyn (CFIE) - Rectangle - left roots, residuals and u = y ./ ds, window [5, 7]" begin
    w = beyn_left_rectangle()
    ks_ref = rectangle_spectrum(2.0, 1.4, 5.0, 7.0)
    kl, kr = sort(real.(w.kl)), sort(real.(w.kr))
    @test length(kl) == length(kr) == length(ks_ref) == 5
    @test all(isapprox.(kl, ks_ref; atol=SPECTRUM_ATOL_BIM))              # left roots against the analytic levels
    @test all(isapprox.(kl, kr; atol=SPECTRUM_ATOL_BIM))                  # left roots against the right roots
    @test all(abs.(imag.(w.kl)) .< IMAG_TOL)
    @test all(w.resl .< 1e-6)                                             # res_tol of the cached solver
    @test size(w.Y) == (length(w.pts), 5)
    # u = y ./ ds against the analytic normal derivative of the mode with the matching k
    modes = [(pi * sqrt(m^2 / 4 + n^2 / 1.96), m, n) for m in 1:12, n in 1:12 if 5.0 <= pi * sqrt(m^2 / 4 + n^2 / 1.96) <= 7.0]
    @test length(modes) == 5
    for j in eachindex(w.kl)
        _, m, n = modes[argmin([abs(k - real(w.kl[j])) for (k, _, _) in modes])]
        u = w.Y[:, j] ./ w.pts.ds
        @test beyn_left_scaled_residual(u, beyn_left_rectangle_u(w.pts, 1.0, 0.7, m, n), w.pts.ds) < BOUNDARY_U_TOL_BIM
    end
end

# solver: Beyn contour integral, left solve (solve_left_vectors)
# kernel/basis: composite CFIE (outer circle and hole circle), SmoothPeriodicGrading, min_pts=64, ppw=10
# billiard: Annular (AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])), inner radius 0.3, outer radius 1
# symmetry: None
# reference: analytic annulus levels (4.4124 simple, 4.7058 x2, 5.4702 x2 in [4.2, 5.8]); radial mode psi = J0(kr) Y0(0.3k) - Y0(kr) J0(0.3k) at
#   k = 4.41239469277677 has |u| constant on each circle with |u_out| / |u_in| = |psi'(1)| / |psi'(0.3)| = 0.536596972483469
# functions to test: solve_left_vectors, _beyn_moments_streaming (side = :left), _beyn_residual for composite kernels, u = y ./ ds
@testset "Beyn (CFIE composite) - Annular - left roots with multiplicity and radial u, window [4.2, 5.8]" begin
    w = beyn_left_annulus()
    ks_ref = annulus_spectrum(0.3, 1.0, 4.2, 5.8)
    kl, kr = sort(real.(w.kl)), sort(real.(w.kr))
    @test length(kl) == length(kr) == length(ks_ref) == 5
    @test all(isapprox.(kl, ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test all(isapprox.(kl, kr; atol=SPECTRUM_ATOL_BIM))
    @test all(abs.(imag.(w.kl)) .< IMAG_TOL_HOLES)
    @test all(w.resl .< 1e-6)
    for kd in unique(round.(ks_ref; digits=8))                            # degenerate pairs keep their multiplicity
        @test count(k -> abs(k - kd) < SPECTRUM_ATOL_BIM, kl) == count(k -> abs(k - kd) < 1e-8, ks_ref)
    end
    # the degenerate left vectors are linearly independent
    pair = findall(k -> abs(real(k) - 4.705775538709766) < SPECTRUM_ATOL_BIM, w.kl)
    Yp = w.Y[:, pair] ./ norm.(eachcol(w.Y[:, pair]))'
    @test length(pair) == 2
    @test svdvals(Yp)[2] > 1e-3
    # radial mode
    k = 4.41239469277677
    j = argmin(abs.(w.kl .- k))
    @test abs(w.kl[j] - k) < SPECTRUM_ATOL_BIM
    u = w.Y[:, j] ./ w.pts.ds
    loop = Int.(w.pts.w_dm)                                               # 1 = outer circle, 2 = hole circle
    uo, ui = abs.(u[loop .== 1]), abs.(u[loop .== 2])
    @test !isempty(uo) && !isempty(ui)
    @test (maximum(uo) - minimum(uo)) / maximum(uo) < BOUNDARY_U_TOL_BIM
    @test (maximum(ui) - minimum(ui)) / maximum(ui) < BOUNDARY_U_TOL_BIM
    dpsi(r) = k * (-besselj(1, k * r) * bessely(0, k * 0.3) + bessely(1, k * r) * besselj(0, k * 0.3))
    ratio = (sum(uo) / length(uo)) / (sum(ui) / length(ui))
    @test abs(ratio / (abs(dpsi(1.0)) / abs(dpsi(0.3))) - 1) < BOUNDARY_U_TOL_BIM
end

# Boundary functions stored in the spectrum (`states = :boundary`, Step 9 of scratchpad/boundary-function-spectrum-plan.md): compute_spectrum runs the
# left contour solve per window and stores the Rellich-normalized u = y ./ ds (no layer density). References: solve_state u at the same level on the
# stored boundary points (same left singular vector up to a complex scale; compared after the least-squares scaling, tolerance BOUNDARY_U_TOL_BIM)
# and the levels of the same solver with states = :none. Window [5, 7] of the rectangle (levels 5.2194, 5.4783, 6.5076, 6.6719, 6.9128; nearest
# outside levels 4.755 and 7.43, so the edges are far from the levels).
const BEYN_BOUNDARY_CACHE = Dict{Symbol,Any}()

function beyn_boundary_rectangle(kind::Symbol; kwargs...)
    return get!(BEYN_BOUNDARY_CACHE, Symbol(kind, kwargs)) do
        billiard = RectangleBilliard(1.0, 0.7)
        kernel = kind === :cfie ? CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading()) :
                 DoubleLayerPotentialSolver(10.0, billiard; grading=GlobalCornerGrading())
        solver = BeynSolver(kernel; use_chebyshev=false, kwargs...)
        data = compute_spectrum(solver, billiard, 5.0, 7.0; multithreaded=false, show_progress=false)
        (; billiard, kernel, data)
    end
end

# solver: Beyn contour integral, states = :boundary
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: solve_state u on the stored pts; levels of the states = :none run and the analytic levels (5 levels in [5, 7])
# functions to test: compute_spectrum (BeynSolver, states = :boundary), _boundary_states
@testset "Beyn (CFIE) - Rectangle - states = :boundary stores u equal to solve_state, window [5, 7]" begin
    r = beyn_boundary_rectangle(:cfie; states=:boundary)
    ref = beyn_boundary_rectangle(:cfie; states=:none)
    data = r.data
    ks = sort(real.(data.k))
    @test length(data) == 5
    @test all(isapprox.(ks, rectangle_spectrum(2.0, 1.4, 5.0, 7.0); atol=SPECTRUM_ATOL_BIM))
    @test ks == sort(real.(ref.data.k)) || all(isapprox.(ks, sort(real.(ref.data.k)); atol=SPECTRUM_ATOL_BIM))
    @test ref.data.states === nothing
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    @test boundary_states_wellformed(data)
    res, idx = stored_u_residuals(data, r.kernel, r.billiard)
    @test length(idx) == 5
    @test all(res .< BOUNDARY_U_TOL_BIM)
end

# solver: Beyn contour integral, states = :boundary
# kernel/basis: DLP, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: None
# reference: solve_state u on the stored pts; levels of the states = :none run
# functions to test: compute_spectrum (BeynSolver, states = :boundary), _boundary_states
@testset "Beyn (DLP) - Rectangle - states = :boundary stores u equal to solve_state, window [5, 7]" begin
    r = beyn_boundary_rectangle(:dlp; states=:boundary)
    ref = beyn_boundary_rectangle(:dlp; states=:none)
    data = r.data
    ks = sort(real.(data.k))
    @test length(data) == length(ref.data) == 5
    @test all(isapprox.(ks, sort(real.(ref.data.k)); atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    @test boundary_states_wellformed(data)
    res, idx = stored_u_residuals(data, r.kernel, r.billiard)
    @test length(idx) == 5
    @test all(res .< BOUNDARY_U_TOL_BIM)
end

# solver: Beyn contour integral, states = :boundary with imag_k_check = true (left projected solve and left imaginary-k check)
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: None
# reference: solve_state u on the stored pts; levels of the states = :none run without the check
# functions to test: compute_spectrum (imag_k_check, side = :left), _beyn_projected_solve, _beyn_imag_k_check
@testset "Beyn (CFIE) - Rectangle - states = :boundary with imag_k_check, window [5, 7]" begin
    r = beyn_boundary_rectangle(:cfie; states=:boundary, imag_k_check=true)
    ref = beyn_boundary_rectangle(:cfie; states=:none)
    data = r.data
    @test length(data) == length(ref.data) == 5
    @test all(isapprox.(sort(real.(data.k)), sort(real.(ref.data.k)); atol=SPECTRUM_ATOL_BIM))
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
    @test boundary_states_wellformed(data)
    res, idx = stored_u_residuals(data, r.kernel, r.billiard)
    @test length(idx) == 5
    @test all(res .< BOUNDARY_U_TOL_BIM)
end
