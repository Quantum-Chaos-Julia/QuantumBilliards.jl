# Integration tests for the normalization contract of the stored boundary normal derivative `u` (scratchpad/compute-spectrum-refactor-plan.md,
# Step 9): every route that stores `u` returns it Rellich-normalized on its full boundary (`_rellich(pts, u, k) = 1`) with the largest-modulus
# entry real and positive. Routes: Beyn, CORK and EBIM with `states = :boundary`, Vergini-Saraceno with `states = :boundary`, and the post-hoc
# `compute_boundary_functions` on a `states = :none` Beyn spectrum.
#
# All routes solve the even-even sector of the full 2.0 x 1.4 rectangle (RectangleBilliard(1.0, 0.7), centred at the origin), whose modes are
# psi = cos(m*pi*x/2) * cos(n*pi*y/1.4) with m, n odd. Each route has its own boundary points, so the states are not compared with each other
# node by node; instead each is compared with the same analytic reference u = grad(psi) . n, normalized by `_normalize_u!`, on its own points.
# The reference u is real and takes equal values on mirror images (the normal reflects with the point), so the sign convention "largest entry
# positive" fixes it uniquely. This makes the routes agree with each other up to the discretization error of each.

using LinearAlgebra: norm, dot

const _CSU_HALF_A, _CSU_HALF_B = 1.0, 0.7

# Analytic normalized u of the even-even mode (m, n) on `pts` at wavenumber `k`.
function _csu_reference_u(pts, k, m::Int, n::Int)
    a = m * pi / (2 * _CSU_HALF_A)
    b = n * pi / (2 * _CSU_HALF_B)
    u = [-a * sin(a * p[1]) * cos(b * p[2]) * nrm[1] - b * cos(a * p[1]) * sin(b * p[2]) * nrm[2] for (p, nrm) in zip(pts.xy, pts.normal)]
    QuantumBilliards._normalize_u!(pts, u, k)
    return u
end

# Mode indices (m, n) of the even-even level nearest to `k`.
function _csu_mode(k::Real)
    modes = [(m, n) for m in 1:2:15 for n in 1:2:15]
    kmn = [pi * sqrt((m / (2 * _CSU_HALF_A))^2 + (n / (2 * _CSU_HALF_B))^2) for (m, n) in modes]
    i = argmin(abs.(kmn .- k))
    return modes[i], kmn[i]
end

# Relative arclength-weighted L2 distance of the stored `u` of every state of `data` from the analytic normalized u; also checks
# the contract (unit Rellich value, largest entry positive, level at the analytic k). Returns the worst distance.
function _csu_check(data; level_atol)
    @test length(data) > 0
    @test boundary_states_wellformed(data)
    worst = 0.0
    for st in data.states
        (m, n), kmn = _csu_mode(st.k)
        @test abs(st.k - kmn) < level_atol
        ref = _csu_reference_u(st.pts, kmn, m, n)
        w = sqrt.(st.pts.ds)
        err = norm(w .* (st.u .- ref)) / norm(w .* ref)
        worst = max(worst, err)
    end
    return worst
end

# Distance of the VS u (RealPlaneWaves(12), boundary_ppw = 5) from the analytic u; worst observed 2.0e-3 (basis truncation, not the normalization).
const _CSU_TOL_VS = 2e-2

function _csu_kernel(billiard)
    return CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading(kressq=2), sym_characters=(1, 1), min_pts=200)
end

# solver: Beyn contour integral, CORK, EBIM (states = :boundary)
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=10, min_pts=200, even-even sector (reduced N = 50)
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: even-even
# reference: analytic u = grad(psi) . n of cos(m*pi*x/2) * cos(n*pi*y/1.4), normalized with `_normalize_u!`; window [3, 10]
# functions to test: compute_spectrum (states = :boundary), _boundary_state, _normalize_u!
@testset "States u contract - Rectangle even-even - BIM routes against analytic u" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = _csu_kernel(billiard)
    ks_ref = rectangle_spectrum(2.0, 1.4, 3.0, 10.0; parity_x=1, parity_y=1)
    kw = (; multithreaded=false, show_progress=false)

    beyn = compute_spectrum(BeynSolver(kernel; states=:boundary), billiard, 3.0, 10.0; kw...)
    @test length(beyn) == length(ks_ref)
    @test all(isapprox.(sort(real.(beyn.k)), ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test _csu_check(beyn; level_atol=SPECTRUM_ATOL_BIM) < BOUNDARY_U_TOL_BIM

    cork = compute_spectrum(CORKSolver(kernel; nlevels=4, states=:boundary), billiard, 3.0, 10.0; kw...)
    @test length(cork) == length(ks_ref)
    @test all(isapprox.(sort(real.(cork.k)), ks_ref; atol=SPECTRUM_ATOL_BIM))
    @test _csu_check(cork; level_atol=SPECTRUM_ATOL_BIM) < BOUNDARY_U_TOL_BIM

    ebim = compute_spectrum(ExpandedBIMSolver(kernel; states=:boundary), billiard, 3.0, 10.0; accept_tol=nothing, kw...)
    genuine = findall(<(TENSION_TOL_EBIM), ebim.ten)
    @test length(genuine) >= length(ks_ref)
    ebim_genuine = QuantumBilliards.SpectralData(ebim.billiard, ebim.solver, ebim.k[genuine], ebim.k_im[genuine], ebim.ten[genuine], ebim.control[genuine], ebim.states[genuine])
    @test _csu_check(ebim_genuine; level_atol=SPECTRUM_ATOL_EBIM) < BOUNDARY_U_TOL_EBIM
end

# solver: Beyn contour integral (states = :none), then compute_boundary_functions
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=10, min_pts=200, even-even sector
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: even-even
# reference: analytic normalized u, as above; window [3, 10]
# functions to test: compute_boundary_functions, _posthoc_left_vectors, _boundary_state
@testset "States u contract - Rectangle even-even - compute_boundary_functions against analytic u" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = _csu_kernel(billiard)
    ks_ref = rectangle_spectrum(2.0, 1.4, 3.0, 10.0; parity_x=1, parity_y=1)
    data = compute_spectrum(BeynSolver(kernel; states=:none), billiard, 3.0, 10.0; multithreaded=false, show_progress=false)
    @test data.states === nothing
    withu = compute_boundary_functions(data; multithreaded=false, show_progress=false)
    @test length(withu) == length(ks_ref)
    @test _csu_check(withu; level_atol=SPECTRUM_ATOL_BIM) < BOUNDARY_U_TOL_BIM
end

# solver: Vergini-Saraceno (states = :boundary)
# basis: Real Plane Waves, even-even sector
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: even-even
# reference: analytic normalized u, as above; window [4, 8]
# functions to test: compute_spectrum (states = :boundary), boundary_states, _normalize_u!
@testset "States u contract - Rectangle even-even - VS basis route against analytic u" begin
    billiard = RectangleBilliard(1.0, 0.7)
    basis = RealPlaneWaves(12; sym_x=1, sym_y=1)
    ks_ref = rectangle_spectrum(2.0, 1.4, 4.0, 8.0; parity_x=1, parity_y=1)
    solver = VerginiSaracenoSolver(5.0, 10.0; states=:boundary, boundary_ppw=5.0)
    data = compute_spectrum(solver, basis, billiard, 4.0, 8.0, 0.1; multithreaded=true, show_progress=false)
    @test length(data) == length(ks_ref)
    @test _csu_check(data; level_atol=SPECTRUM_ATOL_VS) < _CSU_TOL_VS
end
