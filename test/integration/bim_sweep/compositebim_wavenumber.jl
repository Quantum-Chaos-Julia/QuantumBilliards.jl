# Integration tests for CompositeBIMSolver (src/solvers/sweepmethods/compositebim.jl).
# Sweep solver: `solve_wavenumber` only. Parameters from the calibration log of
# scratchpad/test-suite-redesign-plan.md. Each window `k0 +- dk` contains exactly one level.

# Cross-method references computed inside a testset (Beyn on the same composite kernel) are cached here.
const COMPOSITE_SWEEP_REFERENCES = Dict{Symbol,Float64}()

# solver: Composite BIM, one CFIE component per boundary domain
# kernel/basis: CFIE, SmoothPeriodicGrading, min_pts=64
# billiard: Annular (AnnularBilliard(0.3, 0.0)), concentric, inner radius 0.3, outer radius 1
# symmetry: None
# reference: analytic, first root of the annulus Bessel determinant (next level 4.7058 lies outside the window)
# functions to test: solve_wavenumber
@testset "Composite CFIE - Annular - Ground State" begin
    billiard = AnnularBilliard(0.3, 0.0)
    component() = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading(), min_pts=64)
    solver = CompositeBIMSolver(component(), component())
    k_ref = annulus_spectrum(0.3, 1.0, 0.0, 6.0)[1]
    k, t = solve_wavenumber(solver, billiard, k_ref + 0.03, 0.1; multithreaded=false)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end

# solver: Composite BIM, one CFIE component per boundary domain (three: outer square and two holes)
# kernel/basis: CFIE, GlobalCornerGrading(kressq=6), min_pts=192 (discretization limited level; kressq=2 and 3 converge slowly)
# billiard: SquareWithinSquare (SquareWithinSquareBilliard(1.0))
# symmetry: None
# reference: cross-method, Beyn (Taylor backend) on the same composite kernel, window [2.5, 3.0] containing the ground state only
#   (computed in the testset, cached in COMPOSITE_SWEEP_REFERENCES). Observed ground state about 2.73107; next level 3.5559.
# functions to test: solve_wavenumber, compute_spectrum
@testset "Composite CFIE - SquareWithinSquare - Ground State" begin
    billiard = SquareWithinSquareBilliard(1.0)
    ncomponents = length(QuantumBilliards._group_boundary_by_domain_id(full_boundary(billiard)))
    solver = CompositeBIMSolver((CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading(kressq=6), min_pts=192, eps=1e-10) for _ in 1:ncomponents)...)
    k_ref = get!(COMPOSITE_SWEEP_REFERENCES, :squarewithinsquare_ground) do
        k1, k2 = 2.5, 3.0
        beyn = BeynSolver(solver; m=6, nq=24, r=12, Rmax=(k2 - k1) / 2 + 0.02, svd_tol=1e-10, res_tol=2e-5,
            taylor_degree=14, use_chebyshev=false, imag_k_check=false, eigenvectors=false)
        data = compute_spectrum(beyn, billiard, k1, k2; multithreaded=false, show_progress=false)
        @test length(data.k) == 1
        @test all(data.ten .< HIGH_K_TENSION_TOL)
        real(data.k[1])
    end
    k, t = solve_wavenumber(solver, billiard, 2.73, 0.1; multithreaded=false)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end
