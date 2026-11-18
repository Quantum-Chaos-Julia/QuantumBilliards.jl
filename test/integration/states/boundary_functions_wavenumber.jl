# Integration tests for the boundary normal derivative u = d_n psi that `solve_state` returns for the BIM solvers (src/solvers/sweepmethods/
# sweepmethods.jl: `solve_state(::Union{DLP,CFIE})`, `_bim_normal_derivative`, `_boundary_orbits`). u comes from the LEFT singular vector l of the
# Fredholm matrix at a single wavenumber, u_raw = l ./ ds without conjugation, expanded to the full boundary with the orbit phases when a
# symmetry is active, then Rellich normalised. The same code path serves DLP, CFIE and CompositeBIMSolver.
# Comparisons are made after the least-squares scaling (arbitrary complex phase and normalisation), so nothing is pinned.
# Facts used: RectangleBilliard(1.0, 0.7) is the full rectangle [-1,1] x [-0.7,0.7] (2a x 2b); psi_{mn} = sin(m pi (x+a)/2a) sin(n pi (y+b)/2b),
# k = pi sqrt((m/2a)^2 + (n/2b)^2), even in x for odd m and in y for odd n, so (m, n) = (3, 2) lies in the sector (chi_Y, chi_X) = (1, -1);
# the (6, 4) mode (k = 13.0152) is the (3, 2) mode of a 1 x 0.7 rectangle. In the annulus (R_out = 1, R_in = 0.3, common centre (0.2, 0.1))
# the radial mode psi = J0(kr) Y0(0.3k) - Y0(kr) J0(0.3k) at k = 4.41239469277677 has constant |u| on each circle; the outward normal of the
# domain points to the hole centre on the inner circle. Rellich's identity is independent of the origin for a Dirichlet eigenfunction.
# The discrete Rellich norm of the unnormalised u (unit l) has the reference value obtained from the analytic u with the same unit-l scaling.

using LinearAlgebra: norm, dot
using SpecialFunctions: besselj, bessely
using StaticArrays: SVector
using QuantumBilliards: solve_state, _rellich, _boundary_orbits

# Normal derivative of psi_{mn} of RectangleBilliard(a, b) at the boundary points.
function bf_rectangle_u(pts, a, b, m, n)
    W, H = 2a, 2b
    return [begin
                x, y = pts.xy[i]
                nx, ny = pts.normal[i]
                gx = (m * pi / W) * cos(m * pi * (x + a) / W) * sin(n * pi * (y + b) / H)
                gy = (n * pi / H) * sin(m * pi * (x + a) / W) * cos(n * pi * (y + b) / H)
                nx * gx + ny * gy
            end for i in eachindex(pts.xy)]
end

# Arclength-weighted relative residual of `u` after the best complex scaling onto `ref`.
function bf_scaled_residual(u, ref, ds)
    w = sqrt.(ds)
    c = dot(w .* ref, w .* u) / dot(w .* ref, w .* ref)
    return norm(w .* (u .- c .* ref)) / norm(w .* u)
end

# Rellich integral of |u|^2 about the origin `c`.
bf_rellich_about(pts, u, k, c) = sum((pts.normal[i][1] * (pts.xy[i][1] - c[1]) + pts.normal[i][2] * (pts.xy[i][2] - c[2])) * pts.ds[i] * abs2(u[i]) for i in eachindex(u)) / (2k^2)

# Residual of u against the analytic rectangle mode and the relative deviation of the Rellich norm of the unnormalised u from the analytic value
# (unit l means the weighted vector ds .* u_raw over the orbit representatives, or over all nodes without symmetry).
function bf_rectangle_check(solver, billiard, m, n)
    k = pi * sqrt((m / 2)^2 + (n / 1.4)^2)
    pts = evaluate_points(solver, billiard, k)
    ten, _, u, bnd_norm = solve_state(solver, pts, k, billiard; multithreaded=false)
    ref = bf_rectangle_u(pts, 1.0, 0.7, m, n)
    idx = solver.symmetry === nothing ? collect(eachindex(ref)) : _boundary_orbits(solver, pts, billiard).fundamental_indices
    c = 1 / norm(pts.ds[idx] .* ref[idx])
    rellich_ref = _rellich(pts, c .* ref, k)
    return (; ten, npts=length(pts), nu=length(u), residual=bf_scaled_residual(u, ref, pts.ds), rellich_dev=abs(bnd_norm / rellich_ref - 1),
            rellich_u=_rellich(pts, u, k))
end

function bf_check_rectangle(res)
    @test res.ten < TENSION_TOL_BIM
    @test res.nu == res.npts                                              # full-boundary u also with symmetry
    @test res.residual < BOUNDARY_U_TOL_BIM
    @test res.rellich_dev < RELLICH_TOL_BIM
    @test abs(res.rellich_u - 1) < RELLICH_TOL_BIM
end

# solver: Sweep CFIE (solve_state)
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: None
# reference: analytic, u = d_n psi_{6,4} at k = pi sqrt(9 + (2/0.7)^2)
# functions to test: solve_state, _bim_normal_derivative (left-vector u, no symmetry), _rellich
@testset "Sweep (CFIE) - Rectangle - graded u, no symmetry (k = 13.0152)" begin
    b = RectangleBilliard(1.0, 0.7)
    bf_check_rectangle(bf_rectangle_check(CombinedFieldIntegralEquationSolver(10.0, b; grading=GlobalCornerGrading()), b, 6, 4))
end

# solver: Sweep DLP (solve_state)
# kernel/basis: DLP, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: None
# reference: analytic, u = d_n psi_{6,4} at k = pi sqrt(9 + (2/0.7)^2)
# functions to test: solve_state, _bim_normal_derivative (left-vector u, no conjugation for the DLP kernel), _rellich
@testset "Sweep (DLP) - Rectangle - graded u, no symmetry (k = 13.0152)" begin
    b = RectangleBilliard(1.0, 0.7)
    bf_check_rectangle(bf_rectangle_check(DoubleLayerPotentialSolver(10.0, b; grading=GlobalCornerGrading()), b, 6, 4))
end

# solver: Sweep CFIE (solve_state)
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: D2 sector (chi_Y, chi_X) = (1, -1)
# reference: analytic, u = d_n psi_{3,2} at k = pi sqrt(9/4 + 4/1.96)
# functions to test: solve_state, _boundary_orbits, _bim_normal_derivative (reduced l expanded to the full boundary)
@testset "Sweep (CFIE) - Rectangle - graded u, sector (1, -1) (k = 6.5076)" begin
    b = RectangleBilliard(1.0, 0.7)
    bf_check_rectangle(bf_rectangle_check(CombinedFieldIntegralEquationSolver(10.0, b; grading=GlobalCornerGrading(), sym_characters=(1, -1)), b, 3, 2))
end

# solver: Sweep DLP (solve_state)
# kernel/basis: DLP, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: D2 sector (chi_Y, chi_X) = (1, -1)
# reference: analytic, u = d_n psi_{3,2} at k = pi sqrt(9/4 + 4/1.96)
# functions to test: solve_state, _boundary_orbits, _bim_normal_derivative (reduced l expanded to the full boundary)
@testset "Sweep (DLP) - Rectangle - graded u, sector (1, -1) (k = 6.5076)" begin
    b = RectangleBilliard(1.0, 0.7)
    bf_check_rectangle(bf_rectangle_check(DoubleLayerPotentialSolver(10.0, b; grading=GlobalCornerGrading(), sym_characters=(1, -1)), b, 3, 2))
end

# solver: Sweep CompositeBIM (solve_state)
# kernel/basis: composite CFIE (outer circle and hole circle), SmoothPeriodicGrading, ppw=10
# billiard: Annulus (AnnularBilliard(0.3, 0.0; center=[0.2, 0.1]), concentric, no symmetry)
# symmetry: None
# reference: analytic radial mode, |u_out| / |u_in| = |psi'(1)| / |psi'(0.3)| = 0.536596972483469 at k = 4.41239469277677; Rellich identity about two origins
# functions to test: solve_state (composite loops), _bim_normal_derivative, _rellich
@testset "Sweep (CFIE composite) - Annulus - ratio of boundary derivatives and Rellich norm (k = 4.4124)" begin
    b = AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])
    k = 4.41239469277677
    solver = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(10.0, b), CombinedFieldIntegralEquationSolver(10.0, b))
    pts = evaluate_points(solver, b, k)
    ten, _, u, _ = solve_state(solver, pts, k, b; multithreaded=false)
    @test ten < TENSION_TOL_BIM
    @test length(u) == length(pts)
    loop = Int.(pts.w_dm)                                                  # 1 = outer circle, 2 = hole circle
    uo, ui = abs.(u[loop .== 1]), abs.(u[loop .== 2])
    @test !isempty(uo) && !isempty(ui)
    @test (maximum(uo) - minimum(uo)) / maximum(uo) < BOUNDARY_U_TOL_BIM   # radial mode: constant on each circle
    @test (maximum(ui) - minimum(ui)) / maximum(ui) < BOUNDARY_U_TOL_BIM
    dpsi(r) = k * (-besselj(1, k * r) * bessely(0, k * 0.3) + bessely(1, k * r) * besselj(0, k * 0.3))
    ratio = (sum(uo) / length(uo)) / (sum(ui) / length(ui))
    @test abs(dpsi(1.0)) / abs(dpsi(0.3)) ≈ 0.536596972483469 atol = 1e-12
    @test abs(ratio / (abs(dpsi(1.0)) / abs(dpsi(0.3))) - 1) < BOUNDARY_U_TOL_BIM
    @test abs(_rellich(pts, u, k) - 1) < RELLICH_TOL_BIM                   # origin (0, 0)
    @test abs(bf_rellich_about(pts, u, k, SVector(0.2, 0.1)) - 1) < RELLICH_TOL_BIM   # origin at the common centre: the hole enters with the opposite normal
end

# Complex rotation irrep s of C3Billiard(0.2): CFIE and DLP solve_state agree on u, which is genuinely complex.
function bf_c3_check(s, k)
    b = C3Billiard(0.2)
    cfie = CombinedFieldIntegralEquationSolver(10.0, b; sym_characters=(s,))
    dlp = DoubleLayerPotentialSolver(10.0, b; sym_characters=(s,))
    pts_c = evaluate_points(cfie, b, k)
    pts_d = evaluate_points(dlp, b, k)
    ten_c, _, u_c, _ = solve_state(cfie, pts_c, k, b; multithreaded=false)
    ten_d, _, u_d, _ = solve_state(dlp, pts_d, k, b; multithreaded=false)
    @test pts_c.xy == pts_d.xy                                             # the same smooth discretization
    @test ten_c < TENSION_TOL_BIM && ten_d < TENSION_TOL_BIM
    @test length(u_c) == length(u_d) == length(pts_c)
    @test u_c isa Vector{ComplexF64}
    @test 1 - abs(dot(u_c, u_d)) / (norm(u_c) * norm(u_d)) < BOUNDARY_U_TOL_BIM
    @test bf_scaled_residual(u_c, u_d, pts_c.ds) < BOUNDARY_U_TOL_BIM
    @test abs(sum(u_c .^ 2)) / sum(abs2, u_c) < 0.1                        # no global phase makes u real: the orbit phases are cube roots of unity
    @test abs(_rellich(pts_c, u_c, k) - 1) < RELLICH_TOL_BIM
end

# solver: Sweep CFIE and DLP (solve_state)
# kernel/basis: CFIE and DLP, SmoothPeriodicGrading, ppw=10
# billiard: C3 (C3Billiard(0.2))
# symmetry: NFoldRotation(3), complex irrep s = 1
# reference: cross-kernel, CFIE and DLP u at the Beyn level k = 8.389362888535 (overlap near 1), complex phases
# functions to test: solve_state, _boundary_orbits (rotation sector), _bim_normal_derivative (complex phases)
@testset "Sweep (CFIE vs DLP) - C3 - complex irrep s = 1 (k = 8.3894)" begin
    bf_c3_check(1, 8.389362888535)
end

# solver: Sweep CFIE and DLP (solve_state)
# kernel/basis: CFIE and DLP, SmoothPeriodicGrading, ppw=10
# billiard: C3 (C3Billiard(0.2))
# symmetry: NFoldRotation(3), complex irrep s = 2
# reference: cross-kernel, CFIE and DLP u at the Beyn level k = 11.400292336799 (overlap near 1), complex phases
# functions to test: solve_state, _boundary_orbits (rotation sector), _bim_normal_derivative (complex phases)
@testset "Sweep (CFIE vs DLP) - C3 - complex irrep s = 2 (k = 11.4003)" begin
    bf_c3_check(2, 11.400292336799)
end
