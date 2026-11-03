# Integration tests for billiards with holes through CompositeBIMSolver and Beyn (Gate C of the multiply connected
# domain rework, scratchpad plan of BilliardGeometry.jl). Low k, small discretizations.

using BilliardGeometry: area, full_boundary_components
using StaticArrays: SVector

# Minimal billiard holding a MultiplyConnectedDomain with a user chosen symmetry registry.
struct QBHoleD2Billiard <: AbsBilliard
    fundamental_domain::MultiplyConnectedDomain{Float64}
    symmetries::SymmetryRegistry
end

# solver: Beyn contour integral on a Composite BIM kernel (CFIE outer component and CFIE hole component)
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10 (Taylor backend)
# billiard: AnnularBilliard(0.3, 0.0; center=[0.2, 0.1]), concentric, R_i = 0.3, R_o = 1; the nonzero center keeps the unreduced construction
#   (the origin-centered annulus is symmetric since B7); the spectrum is translation invariant
# symmetry: None
# reference: analytic, roots of J_m(k Ro) Y_m(k Ri) - Y_m(k Ro) J_m(k Ri), multiplicity 1 for m = 0 and 2 for m >= 1
# functions to test: compute_spectrum, area, full_boundary_components
@testset "Composite CFIE Beyn - Annular - Spectrum vs Bessel roots, Weyl count" begin
    billiard = AnnularBilliard(0.3, 0.0; center=[0.2, 0.1])
    ncomp = 1 + length(billiard.fundamental_domain.holes)
    @test ncomp == 2
    kernel = CompositeBIMSolver((CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading()) for _ in 1:ncomp)...)
    solver = BeynSolver(kernel; use_chebyshev=false)
    k1, k2 = 4.0, 12.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
    ks = sort(real.(data.k))
    ks_ref = annulus_spectrum(0.3, 1.0, k1, k2)

    @testset "levels and multiplicities" begin
        @test length(ks) == length(ks_ref)
        @test length(ks) == length(ks_ref) && all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
        @test all(data.ten .< TENSION_TOL_ACCEL)
        @test all(abs.(data.k_im) .< IMAG_TOL)
        for kd in unique(round.(ks_ref; digits=8))
            mult = count(k -> abs(k - kd) < 1e-8, ks_ref)
            @test count(k -> abs(k - kd) < SPECTRUM_ATOL_BIM, ks) == mult
        end
    end

    @testset "Weyl law with the hole-corrected area and the component-sum perimeter" begin
        A = area(billiard)
        P = sum(sum(c.length for c in loop) for loop in full_boundary_components(billiard).curves)
        @test A ≈ π * (1.0^2 - 0.3^2)
        @test P ≈ 2π * (1.0 + 0.3)
        weyl(k) = A * k^2 / (4π) - P * k / (4π)
        # counting function at a point halfway between levels: |N(k) - Weyl(k)| stays below one level
        for kc in (6.0, 8.5, 11.0)
            gap_ok = all(abs(kc - k) > 0.05 for k in ks_ref)
            @test gap_ok
            n_found = count(<(kc), ks)           # no level below k1 = 4 (the ground state is near 4.49)
            n_ref = count(<(kc), ks_ref)
            @test n_found == n_ref
            @test abs(n_ref - weyl(kc)) < 2.0
        end
    end
end

# solver: Composite BIM sweep (DLP outer component, CFIE hole component)
# billiard: D2 symmetry registry with an asymmetric DomainHole (RectangleBilliard(1, 1) frame, off-axis hole disc)
# reference: contract of the guards (no spectrum): unreduced evaluation works, symmetry-reduced and fundamental-domain use raises
# functions to test: evaluate_points, get_boundary_curves, area
@testset "Composite BIM - D2 billiard with asymmetric DomainHole - unreduced works, reduced raises" begin
    hole_curve = CircleSegment(0.2, 2π, 0.0, [0.3, 0.4]; domain_id=5, segment_id=1)
    hole = DomainHole(SimpleDomain{Float64}(AbsCurve[hole_curve], SVector{2,Float64}[SVector(0.5, 0.4)], 5))
    outer = RectangleBilliard(1.0, 1.0).fundamental_domain
    mcd = MultiplyConnectedDomain(outer, [hole], 1)
    billiard = QBHoleD2Billiard(mcd, D2_symmetry())
    k = 3.0

    @testset "unreduced: evaluate_points on the full boundary" begin
        # a billiard with the same hole but an empty registry is the physical (frame-coinciding) reference
        plain = QBHoleD2Billiard(MultiplyConnectedDomain(RectangleBilliard(1.0, 1.0).fundamental_domain, [hole], 1), register_symmetries())
        solver = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(5.0, plain; min_pts=64), CombinedFieldIntegralEquationSolver(5.0, plain; min_pts=24))
        pts = evaluate_points(solver, plain, k)
        @test length(pts) >= 64 + 24
        @test count(==(2.0), pts.w_dm) == 24
    end

    @testset "symmetry-reduced composite solver raises" begin
        red = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(5.0, billiard; min_pts=64, symmetry=XYAxisReflection(), sym_characters=(1, 1)),
            CombinedFieldIntegralEquationSolver(5.0, billiard; min_pts=24, symmetry=XYAxisReflection(), sym_characters=(1, 1)))
        pts = evaluate_points(red, billiard, k)            # sampling only reads the full boundary; the reduction is what is refused
        @test_throws ArgumentError QuantumBilliards._composite_symmetry_orbits(Float64, red, pts)
        @test_throws ArgumentError construct_matrices(red, pts, k)
        @test_throws ArgumentError get_boundary_curves(billiard)
    end

    @testset "fundamental-domain solvers raise" begin
        dlp = DoubleLayerPotentialSolver(5.0, billiard; min_pts=32)
        @test_throws ArgumentError evaluate_points(dlp, billiard, k)
    end
end
