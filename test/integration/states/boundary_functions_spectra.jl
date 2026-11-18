# Integration tests for compute_boundary_functions (src/states/boundarystates.jl): the post-hoc left solve that attaches the Rellich-normalised
# boundary normal derivative u = d_n psi to every level of a BIM spectrum. Per level (or cluster of degenerate levels) A(k) is assembled on the
# state's pts (or evaluate_points(kernel, billiard, k) when data.states === nothing), A^H y = 0 is solved by adjoint inverse iteration, and u follows
# from _bim_normal_derivative with the global phase fixed so that the largest entry is real positive.
# References: u of solve_state at the same level (same left singular vector up to a complex scale), compared after the least-squares scaling,
# and analytic annulus modes. Spectra are built by hand (SpectralData with exact analytic wavenumbers), no eigen search.
# Facts used: RectangleBilliard(1.0, 0.7) is the full rectangle [-1,1] x [-0.7,0.7], k_{mn} = pi sqrt((m/2)^2 + (n/1.4)^2), so (6, 4) gives
# k = 13.0152. In the annulus (R_out = 1, R_in = 0.3, common centre (0.2, 0.1)) the modes are R_n(r) cos(n t), R_n(r) sin(n t) with
# R_n(r) = J_n(kr) Y_n(0.3k) - Y_n(kr) J_n(0.3k) and k a root of R_n(1) = 0; n = 0 at k = 4.41239469277677 has |u_out| / |u_in| = 0.536596972483469;
# for n >= 1 the cos and sin modes are degenerate (two orthogonal u). The outward normal of the domain points to the hole centre on the inner circle.
# Helper definitions use the prefix `bs_` / `BS_`.

using LinearAlgebra: norm, dot, svdvals
using SpecialFunctions: besselj, bessely
using StaticArrays: SVector
using QuantumBilliards: solve_state, _rellich, compute_boundary_functions

const BS_CENTER = (0.2, 0.1)
const BS_R_IN = 0.3

# Arclength-weighted relative residual of `u` after the best complex scaling onto `ref`.
function bs_scaled_residual(u, ref, ds)
    w = sqrt.(ds)
    c = dot(w .* ref, w .* u) / dot(w .* ref, w .* ref)
    return norm(w .* (u .- c .* ref)) / norm(w .* u)
end

# Relative residual of `u` after projection onto the span of the columns of `B` (arclength weights).
function bs_span_residual(u, B, ds)
    w = sqrt.(ds)
    Bw = w .* B
    uw = w .* u
    return norm(uw - Bw * (Bw \ uw)) / norm(uw)
end

# Spectrum with the given wavenumbers and, when `with_states`, one state per level on pts = evaluate_points(solver, billiard, k) without u.
function bs_spectrum(solver, billiard, ks; with_states=true)
    pts = [evaluate_points(solver, billiard, k) for k in ks]
    states = with_states ? [BIMEigenstate(k, nothing, 0.0, p) for (k, p) in zip(ks, pts)] : nothing
    return SpectralData(billiard, solver, collect(Float64, ks), zeros(length(ks)), fill(false, length(ks)); states), pts
end

# u and bnd_norm of solve_state at one level.
function bs_reference(solver, billiard, pts, k)
    ten, _, u, bnd_norm = solve_state(solver, pts, k, billiard; multithreaded=false)
    return u, bnd_norm
end

function bs_check_output(out, data, ks; with_states=true)
    @test out isa SpectralData
    @test length(out) == length(data) && out.k == data.k && out.ten == data.ten
    @test out.solver === data.solver && out.billiard === data.billiard
    @test out.states isa Vector{BIMEigenstate{ComplexF64,Float64}}
    for (i, st) in enumerate(out.states)
        @test st.u isa Vector{ComplexF64} && length(st.u) == length(st.pts)
        @test st.k == ks[i] && st.ten == data.ten[i]
        with_states && @test st.pts === data.states[i].pts
        @test abs(_rellich(st.pts, st.u, st.k) - 1) < RELLICH_TOL_BIM
        p = argmax(abs.(st.u))
        @test abs(imag(st.u[p])) <= 1e-12 * abs(st.u[p]) && real(st.u[p]) > 0   # global phase fixed on the largest entry (ties on the circles: tolerance)
    end
end

function bs_rectangle_check(solver, billiard; with_states=true)
    ks = [pi * sqrt(9 + (2 / 0.7)^2)]
    data, pts = bs_spectrum(solver, billiard, ks; with_states)
    out = compute_boundary_functions(data; multithreaded=false)
    bs_check_output(out, data, ks; with_states)
    st = out.states[1]
    u_ref, bnd_ref = bs_reference(solver, billiard, pts[1], ks[1])
    @test bs_scaled_residual(st.u, u_ref, st.pts.ds) < BOUNDARY_U_TOL_BIM
    @test abs(st.bnd_norm / bnd_ref - 1) < RELLICH_TOL_BIM
    return out
end

# solver: Sweep CFIE (compute_boundary_functions)
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: None
# reference: solve_state u and bnd_norm at k = pi sqrt(9 + (2/0.7)^2)
# functions to test: compute_boundary_functions (states stored), _left_null_vectors, _bim_normal_derivative, _rellich
@testset "compute_boundary_functions (CFIE) - Rectangle - u against solve_state (k = 13.0152)" begin
    b = RectangleBilliard(1.0, 0.7)
    bs_rectangle_check(CombinedFieldIntegralEquationSolver(10.0, b; grading=GlobalCornerGrading()), b)
end

# solver: Sweep DLP (compute_boundary_functions)
# kernel/basis: DLP, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: None
# reference: solve_state u and bnd_norm at k = pi sqrt(9 + (2/0.7)^2)
# functions to test: compute_boundary_functions (states stored), _left_null_vectors, _bim_normal_derivative, _rellich
@testset "compute_boundary_functions (DLP) - Rectangle - u against solve_state (k = 13.0152)" begin
    b = RectangleBilliard(1.0, 0.7)
    bs_rectangle_check(DoubleLayerPotentialSolver(10.0, b; grading=GlobalCornerGrading()), b)
end

# solver: Sweep CFIE (compute_boundary_functions)
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: None
# reference: solve_state u; the states === nothing route builds pts with evaluate_points(kernel, billiard, k) and must equal the stored-states route
# functions to test: compute_boundary_functions (data.states === nothing)
@testset "compute_boundary_functions (CFIE) - Rectangle - data.states === nothing" begin
    b = RectangleBilliard(1.0, 0.7)
    solver = CombinedFieldIntegralEquationSolver(10.0, b; grading=GlobalCornerGrading())
    out = bs_rectangle_check(solver, b; with_states=false)
    st = out.states[1]
    @test st.vec === nothing
    data, _ = bs_spectrum(solver, b, [pi * sqrt(9 + (2 / 0.7)^2)])
    st_stored = compute_boundary_functions(data; multithreaded=false).states[1]
    @test st.pts.xy == st_stored.pts.xy
    @test bs_scaled_residual(st.u, st_stored.u, st.pts.ds) < BOUNDARY_U_TOL_BIM
    @test st.bnd_norm ≈ st_stored.bnd_norm rtol = RELLICH_TOL_BIM
end

# solver: Sweep CFIE (compute_boundary_functions)
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: None
# reference: solve_state u stored in the first state (kept unchanged, converted to ComplexF64); the second state without u is computed and its layer density kept
# functions to test: compute_boundary_functions (states with u kept, vec kept)
@testset "compute_boundary_functions (CFIE) - Rectangle - stored u kept, vec kept" begin
    b = RectangleBilliard(1.0, 0.7)
    solver = CombinedFieldIntegralEquationSolver(10.0, b; grading=GlobalCornerGrading())
    k1, k2 = pi * sqrt(9 + (2 / 0.7)^2), pi * sqrt(9 / 4 + (2 / 1.4)^2)
    pts1, pts2 = evaluate_points(solver, b, k1), evaluate_points(solver, b, k2)
    ten, vec, u, bnd_norm = solve_state(solver, pts1, k1, b; multithreaded=false)
    s1 = BIMEigenstate(k1, vec, ten, pts1; u, bnd_norm)
    vec2 = ones(ComplexF64, length(pts2))
    s2 = BIMEigenstate(k2, vec2, 0.0, pts2)
    data = SpectralData(b, solver, [k1, k2], [ten, 0.0], [false, false]; states=[s1, s2])
    out = compute_boundary_functions(data; multithreaded=false)
    @test out.states[1].u == u && out.states[1].bnd_norm == bnd_norm && out.states[1].vec == vec
    @test out.states[2].vec == vec2
    @test out.states[2].u isa Vector{ComplexF64} && abs(_rellich(pts2, out.states[2].u, k2) - 1) < RELLICH_TOL_BIM
    u_ref, _ = bs_reference(solver, b, pts2, k2)
    @test bs_scaled_residual(out.states[2].u, u_ref, pts2.ds) < BOUNDARY_U_TOL_BIM
end

# Normal derivative of the mode R_n(r) cos(n t) or R_n(r) sin(n t) of the annulus at the boundary points.
function bs_annulus_mode_u(pts, k, n, trig)
    a = BS_R_IN
    dR(r) = k * ((besselj(n - 1, k * r) - besselj(n + 1, k * r)) / 2 * bessely(n, a * k) - (bessely(n - 1, k * r) - bessely(n + 1, k * r)) / 2 * besselj(n, a * k))
    return [begin
                dx, dy = pts.xy[i][1] - BS_CENTER[1], pts.xy[i][2] - BS_CENTER[2]
                r, t = hypot(dx, dy), atan(dy, dx)
                nr = (pts.normal[i][1] * dx + pts.normal[i][2] * dy) / r      # radial component of the normal (+1 outer, -1 inner); the normal of a circle is radial
                nr * dR(r) * trig(n * t)
            end for i in eachindex(pts.xy)]
end

# Smallest root of R_n(1) = 0 for k >= k_start by scanning and bisection.
function bs_annulus_root(n, k_start)
    f(k) = besselj(n, k) * bessely(n, BS_R_IN * k) - bessely(n, k) * besselj(n, BS_R_IN * k)
    k = k_start
    while f(k) * f(k + 0.01) > 0
        k += 0.01
    end
    lo, hi = k, k + 0.01
    for _ in 1:60
        mid = (lo + hi) / 2
        f(lo) * f(mid) <= 0 ? (hi = mid) : (lo = mid)
    end
    return (lo + hi) / 2
end

bs_annulus_solver(b) = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(10.0, b), CombinedFieldIntegralEquationSolver(10.0, b))

# solver: Sweep CompositeBIM (compute_boundary_functions)
# kernel/basis: composite CFIE (outer circle and hole circle), SmoothPeriodicGrading, ppw=10
# billiard: Annulus (AnnularBilliard(0.3, 0.0; center=[0.2, 0.1]), concentric, no symmetry)
# symmetry: None
# reference: analytic radial mode, |u_out| / |u_in| = 0.536596972483469 at k = 4.41239469277677; solve_state u
# functions to test: compute_boundary_functions (composite kernel, hole loop), _bim_normal_derivative, _rellich
@testset "compute_boundary_functions (CFIE composite) - Annulus - radial mode (k = 4.4124)" begin
    b = AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])
    k = 4.41239469277677
    solver = bs_annulus_solver(b)
    data, pts = bs_spectrum(solver, b, [k])
    out = compute_boundary_functions(data; multithreaded=false)
    bs_check_output(out, data, [k])
    st = out.states[1]
    loop = Int.(st.pts.w_dm)                                               # 1 = outer circle, 2 = hole circle
    uo, ui = abs.(st.u[loop .== 1]), abs.(st.u[loop .== 2])
    @test (maximum(uo) - minimum(uo)) / maximum(uo) < BOUNDARY_U_TOL_BIM
    @test (maximum(ui) - minimum(ui)) / maximum(ui) < BOUNDARY_U_TOL_BIM
    @test abs((sum(uo) / length(uo)) / (sum(ui) / length(ui)) / 0.536596972483469 - 1) < BOUNDARY_U_TOL_BIM
    u_ref, bnd_ref = bs_reference(solver, b, pts[1], k)
    @test bs_scaled_residual(st.u, u_ref, st.pts.ds) < BOUNDARY_U_TOL_BIM
    @test abs(st.bnd_norm / bnd_ref - 1) < RELLICH_TOL_BIM
end

# solver: Sweep CompositeBIM (compute_boundary_functions)
# kernel/basis: composite CFIE (outer circle and hole circle), SmoothPeriodicGrading, ppw=10
# billiard: Annulus (AnnularBilliard(0.3, 0.0; center=[0.2, 0.1]), concentric, no symmetry)
# symmetry: None
# reference: analytic cos / sin n = 1 modes at the first root of R_1(1) = 0 (Bessel zero found by bisection); solve_state u lies in their span
# functions to test: compute_boundary_functions (cluster of two degenerate levels, block inverse iteration with QR)
@testset "compute_boundary_functions (CFIE composite) - Annulus - degenerate n = 1 pair spans the eigenspace" begin
    b = AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])
    k = bs_annulus_root(1, 1.0)
    solver = bs_annulus_solver(b)
    pts1 = evaluate_points(solver, b, k)
    pts = [pts1, pts1]                                                     # levels sharing one boundary discretization form one cluster
    states = [BIMEigenstate(k, nothing, 0.0, pts1), BIMEigenstate(k + 2e-9, nothing, 0.0, pts1)]
    data = SpectralData(b, solver, [k, k + 2e-9], [0.0, 0.0], [false, false]; states)
    out = compute_boundary_functions(data; multithreaded=false)
    bs_check_output(out, data, [k, k + 2e-9])
    p = out.states[1].pts
    @test out.states[2].pts === p                                          # one cluster, one set of points
    B = hcat(bs_annulus_mode_u(p, k, 1, cos), bs_annulus_mode_u(p, k, 1, sin))
    U = hcat(out.states[1].u, out.states[2].u)
    @test bs_span_residual(out.states[1].u, B, p.ds) < BOUNDARY_U_TOL_BIM
    @test bs_span_residual(out.states[2].u, B, p.ds) < BOUNDARY_U_TOL_BIM
    # the two computed vectors span the whole eigenspace: both analytic modes are reproduced from them
    @test bs_span_residual(B[:, 1], U, p.ds) < BOUNDARY_U_TOL_BIM
    @test bs_span_residual(B[:, 2], U, p.ds) < BOUNDARY_U_TOL_BIM
    # and are linearly independent (orthogonal Y: weighted Gram matrix close to diagonal)
    w = p.ds
    G = [dot(w .* U[:, i], w .* U[:, j]) for i in 1:2, j in 1:2]
    @test abs(G[1, 2]) / sqrt(real(G[1, 1]) * real(G[2, 2])) < BOUNDARY_U_TOL_BIM
    @test svdvals(U ./ sqrt.(sum(abs2, U; dims=1)))[2] > 0.1
    # the solve_state u (one vector of the eigenspace) lies in the span of the computed pair
    u_ref, _ = bs_reference(solver, b, pts[1], k)
    @test bs_span_residual(u_ref, U, p.ds) < BOUNDARY_U_TOL_BIM
end
