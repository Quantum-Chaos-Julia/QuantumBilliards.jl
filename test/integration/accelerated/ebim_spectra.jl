# Integration tests for ExpandedBIMSolver (src/solvers/acceleratedmethods/ebim.jl) through `compute_spectrum`.
# EBIM returns spurious candidates (tension above 2) and duplicates of genuine levels near window edges
# (finding F1 of scratchpad/test-suite-redesign-plan.md), so no count or duplicate assertions are made. Instead:
#   1. every reference level is matched by a returned level with tension < TENSION_TOL_EBIM within SPECTRUM_ATOL_EBIM;
#   2. no returned level with tension < TENSION_TOL_EBIM lies further than SPECTRUM_ATOL_EBIM from a reference level.

# solver: Expanded BIM
# kernel/basis: DLP, GlobalCornerGrading, ppw=5
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762 (it is the only level in [5.5, 6.7])
# functions to test: compute_spectrum
@testset "EBIM (DLP) - Full Triangle - Ground State" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    kernel = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
    solver = ExpandedBIMSolver(kernel)
    k1, k2 = 5.5, 6.7
    # accept_tol=nothing: the default EBIM filter (5e-4) is tighter than TENSION_TOL_EBIM and would drop genuine levels
    data = compute_spectrum(solver, billiard, k1, k2; accept_tol=nothing, multithreaded=true, show_progress=false)
    ks_ref = [6.065090986269762]
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
    @test all(i -> minimum(abs.(ks_ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
end

# solver: Expanded BIM
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, j_{0,1} = 2.4048 (window [2, 3] holds it alone; the next level 3.8317 lies outside)
# functions to test: compute_spectrum
@testset "EBIM (CFIE) - Circle - Ground State" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    solver = ExpandedBIMSolver(kernel)
    k1, k2 = 2.0, 3.0
    data = compute_spectrum(solver, billiard, k1, k2; accept_tol=nothing, multithreaded=true, show_progress=false)
    ks_ref = circle_spectrum(1.0, k1, k2)
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test length(ks_ref) == 1
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
    @test all(i -> minimum(abs.(ks_ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
end

# solver: Expanded BIM
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: None
# reference: analytic; the window [3, 10] holds 16 levels
# functions to test: compute_spectrum
@testset "EBIM (CFIE) - Rectangle - Spectrum k in [3, 10]" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    solver = ExpandedBIMSolver(kernel)
    k1, k2 = 3.0, 10.0
    data = compute_spectrum(solver, billiard, k1, k2; accept_tol=nothing, multithreaded=true, show_progress=false)
    ks_ref = rectangle_spectrum(2.0, 1.4, k1, k2)
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test length(ks_ref) == 16
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
    @test all(i -> minimum(abs.(ks_ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
end

# solver: Expanded BIM
# kernel/basis: DLP, GlobalCornerGrading, ppw=5 (a plain DLP kernel has a rank-deficient A'(k), so the generalized
#   eigenproblem (A, A') reports NaN eigenvalues; selecting the nearest candidate with a plain argmin used to lock onto one)
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: Beyn CFIE value 6.065090986269762
# functions to test: solve_wavenumber (NaN regression ported from the old chebyshevtests.jl; the old `_argmin_finite` is
#   now inlined in solve_wavenumber)
@testset "EBIM (DLP) - NaN regression - Full Triangle" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    kernel = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
    solver = ExpandedBIMSolver(kernel)
    k_ref = 6.065090986269762
    k, t = solve_wavenumber(solver, billiard, k_ref, 0.05; multithreaded=true)
    @test k !== nothing
    @test !isnan(real(k))
    @test !isnan(t)
    @test isapprox(real(k), k_ref; atol=SPECTRUM_ATOL_EBIM)
    @test t < TENSION_TOL_EBIM
end

# Boundary functions stored in the spectrum (`states = :boundary`, Step 10 of scratchpad/boundary-function-spectrum-plan.md): compute_spectrum
# takes the left generalized eigenvectors of the local EBIM problem and stores the Rellich-normalized u = y ./ ds (expanded over the symmetry
# orbits) per level. The vector is a left null vector of A(k_level) to first order in the EBIM correction, so its angle error is O(eps^2)
# (see BOUNDARY_U_TOL_EBIM). Reference: the `u` of compute_eigenstate(kernel, billiard, k) at the stored wavenumber, compared after the
# least-squares complex scaling (the stored u has its global phase fixed on the largest entry). Only the simple levels `ks_ref` are compared
# (degenerate eigenspaces have no unique u); for each of them the confident level (tension < TENSION_TOL_EBIM) nearest to it is used, which
# also covers the duplicates that EBIM leaves near window edges.
using LinearAlgebra: norm, dot

function ebs_u_residuals(data, kernel, billiard, ks_ref)
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    res = Float64[]
    for kr in ks_ref
        i = confident[argmin(abs.(ks[confident] .- kr))]
        abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM || error("no confident level near $kr")
        st = data.states[i]
        ref = compute_eigenstate(kernel, billiard, ks[i]; multithreaded=false).states[1]
        length(ref.pts) == length(st.pts) || error("stored and reference boundary points differ: $(length(st.pts)) vs $(length(ref.pts))")
        w = sqrt.(st.pts.ds)
        c = dot(w .* ref.u, w .* st.u) / dot(w .* ref.u, w .* ref.u)
        push!(res, norm(w .* (st.u .- c .* ref.u)) / norm(w .* st.u))
    end
    return res
end

# Window [3.5, 5.6] of RectangleBilliard(1.0, 0.7) (levels 3.8610, 4.7544, 5.2194, 5.4783; the neighbours outside are 2.74 and 6.1).
function ebs_rectangle_boundary(; use_taylor::Bool)
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading())
    solver = ExpandedBIMSolver(kernel; use_taylor, states=:boundary)
    data = compute_spectrum(solver, billiard, 3.5, 5.6; accept_tol=nothing, multithreaded=false, show_progress=false)
    return billiard, kernel, data
end

for use_taylor in (false, true)
    # solver: Expanded BIM, states = :boundary
    # kernel/basis: CFIE, GlobalCornerGrading, ppw=10
    # billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
    # symmetry: None
    # reference: compute_eigenstate u at the stored wavenumber; analytic levels pi*sqrt(m^2/4 + n^2/1.96)
    # functions to test: compute_spectrum (states = :boundary, left vectors of _solve), _boundary_state, overlap_and_merge_ebim!
    @testset "EBIM (CFIE, $(use_taylor ? "Taylor panels" : "direct")) - Rectangle - stored u against compute_eigenstate, window [3.5, 5.6]" begin
        billiard, kernel, data = ebs_rectangle_boundary(; use_taylor)
        @test boundary_states_wellformed(data)
        ks_ref = rectangle_spectrum(2.0, 1.4, 3.5, 5.6)
        ks = real.(data.k)
        confident = findall(<(TENSION_TOL_EBIM), data.ten)
        @test length(ks_ref) == 4
        @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
        res = ebs_u_residuals(data, kernel, billiard, ks_ref)
        @test length(res) == length(ks_ref)
        @test all(res .< BOUNDARY_U_TOL_EBIM)
    end
end

# solver: Expanded BIM, states = :boundary
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10, reflection sector sym_characters = (1, 1) (even in x and y: modes (m, n) both odd)
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: D2 sector (1, 1); the stored u covers the whole boundary (orbit expansion)
# reference: compute_eigenstate u of the same reduced kernel at the stored wavenumber; analytic levels (1,1), (3,1), (1,3) = 2.7390, 5.2194, 6.9128 of the sector in [2.5, 7]
# functions to test: compute_spectrum (states = :boundary with a symmetry-reduced kernel), _boundary_state (orbit expansion)
@testset "EBIM (CFIE, Taylor panels) - Rectangle - sector (1, 1) - stored u against compute_eigenstate, window [2.5, 7]" begin
    billiard = RectangleBilliard(1.0, 0.7)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading(), sym_characters=(1, 1))
    solver = ExpandedBIMSolver(kernel; states=:boundary)
    data = compute_spectrum(solver, billiard, 2.5, 7.0; accept_tol=nothing, multithreaded=false, show_progress=false)
    ks_ref = [pi * sqrt(m^2 / 4 + n^2 / 1.96) for (m, n) in ((1, 1), (3, 1), (1, 3))]
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test boundary_states_wellformed(data)
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
    @test all(i -> minimum(abs.(ks_ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
    res = ebs_u_residuals(data, kernel, billiard, ks_ref)
    @test length(res) == length(ks_ref)
    @test all(length(data.states[i].u) == length(data.states[i].pts) for i in confident)   # full-boundary u, not the fundamental part
    @test all(res .< BOUNDARY_U_TOL_EBIM)
end

# solver: Expanded BIM, states = :boundary
# kernel/basis: composite CFIE (outer circle and hole circle), SmoothPeriodicGrading, min_pts=64, ppw=10
# billiard: Annular (AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])), inner radius 0.3, outer radius 1
# symmetry: None
# reference: compute_eigenstate u of the composite kernel at the stored wavenumber; the radial level 4.41239469277677 is the only (and simple) level in [4.0, 4.6]
# functions to test: compute_spectrum (states = :boundary with a composite kernel with a hole), _boundary_state (composite u)
@testset "EBIM (CFIE composite) - Annular - stored u against compute_eigenstate, window [4.0, 4.6]" begin
    billiard = AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])
    component() = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading(), min_pts=64)
    kernel = CompositeBIMSolver(component(), component())
    solver = ExpandedBIMSolver(kernel; states=:boundary)
    data = compute_spectrum(solver, billiard, 4.0, 4.6; accept_tol=nothing, multithreaded=false, show_progress=false)
    ks_ref = annulus_spectrum(0.3, 1.0, 4.0, 4.6)
    ks = real.(data.k)
    confident = findall(<(TENSION_TOL_EBIM), data.ten)
    @test length(ks_ref) == 1
    @test boundary_states_wellformed(data)
    @test all(kr -> any(i -> abs(ks[i] - kr) < SPECTRUM_ATOL_EBIM, confident), ks_ref)
    @test all(i -> minimum(abs.(ks_ref .- ks[i])) < SPECTRUM_ATOL_EBIM, confident)
    res = ebs_u_residuals(data, kernel, billiard, ks_ref)
    @test all(res .< BOUNDARY_U_TOL_EBIM)
end
