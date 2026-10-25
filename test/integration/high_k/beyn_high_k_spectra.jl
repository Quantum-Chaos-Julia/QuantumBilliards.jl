# High-wavenumber integration tests: BeynSolver (Taylor backend) on a CFIE kernel, full boundary, one narrow
# window near k = 100, one explicit testset per exported concrete billiard geometry (registry of the old
# tests_all.jl). Parameters from the "Calibration log" (step 3c) of scratchpad/test-suite-redesign-plan.md:
# Beyn m=6, nq=16, r=8 (16 for the degenerate-rich circle), taylor_degree=8, Rmax = half window width + 0.01,
# res_tol=2e-5, svd_tol=1e-10, no eigenvectors, no imaginary-k check (it would force a Chebyshev config build).
# Smooth boundaries use SmoothPeriodicGrading with ppw 4 to 6, corner boundaries GlobalCornerGrading(kressq=2)
# with ppw 8. Geometries with an analytic level list (Circle, Rectangle, Triangle, Annular) assert count and
# level agreement with HIGH_K_ATOL; the other geometries run the same window at ppw and 1.5*ppw and assert equal
# count and level agreement within HIGH_K_ATOL. All testsets assert tension below HIGH_K_TENSION_TOL and levels
# inside the window. The window of the geometries without an analytic list is high_k_window(area), i.e. about
# HIGH_K_LEVELS levels on average (area of hole geometries: outer minus holes).

using StaticArrays: SVector

# Beyn spectrum of `kernel` on `[k1, k2]` with the speed settings of the high-k tier.
function high_k_beyn_spectrum(kernel, billiard, k1, k2; r=8)
    solver = BeynSolver(kernel; use_taylor=true, use_chebyshev=false, m=6, nq=16, r=r, taylor_degree=8,
        Rmax=(k2 - k1) / 2 + 0.01, res_tol=2e-5, svd_tol=1e-10, eigenvectors=false, imag_k_check=false)
    return compute_spectrum(solver, billiard, k1, k2; multithreaded=true, show_progress=false)
end

# Area of a billiard with holes: outer component minus the hole components (`area(billiard)` ignores holes).
function high_k_hole_area(billiard)
    components = QuantumBilliards._group_boundary_by_domain_id(full_boundary(billiard))
    return sum((first(c).domain_id == 1 ? 1 : -1) * area(c) for c in components)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=4, Taylor backend
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, Bessel zeros j_{m,n}; the unit disc has no level in [100, 100.19], the window (100.19, 100.25)
#   holds 10 levels (5 degenerate pairs: 100.1927, 100.2048, 100.2122, 100.2343, 100.2369)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - Circle - k near 100" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(4.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    k1, k2 = 100.19, 100.25
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2; r=16)
    ks_ref = circle_spectrum(1.0, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks_ref) == 10
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=5 and 7.5, Taylor backend
# billiard: Ellipse (EllipseBilliard(1.0, 0.8))
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(0.8*pi)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - Ellipse - k near 100" begin
    billiard = EllipseBilliard(1.0, 0.8)
    k1, k2 = high_k_window(area(billiard))
    kernel = CombinedFieldIntegralEquationSolver(5.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(7.5, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=8, Taylor backend
# billiard: Rectangle (RectangleBilliard(1.0, 0.8), the full 2.0 x 1.6 rectangle)
# symmetry: None
# reference: analytic, pi*sqrt(m^2/2^2 + n^2/1.6^2); window (100.015, 100.17) holds 4 levels
#   (100.0273, 100.0582, 100.1383, 100.1506), edges at least 0.012 from the nearest level
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, GlobalCornerGrading, Taylor) - Rectangle - k near 100" begin
    billiard = RectangleBilliard(1.0, 0.8)
    kernel = CombinedFieldIntegralEquationSolver(8.0, billiard; grading=GlobalCornerGrading(kressq=2), eps=1e-10)
    k1, k2 = 100.015, 100.17
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    ks_ref = rectangle_spectrum(2.0, 1.6, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks_ref) == 4
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=8, Taylor backend
# billiard: Triangle (TriangleBilliard(pi/2, 1.0; x0=1/3, y0=1/3)), right isosceles triangle with legs 1
# symmetry: None
# reference: analytic, pi*sqrt(m^2 + n^2), m > n >= 1; window (100.1, 100.45) holds 3 levels
#   (100.1868, 100.2360, 100.3836), edges at least 0.066 from the nearest level
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, GlobalCornerGrading, Taylor) - Triangle - k near 100" begin
    billiard = TriangleBilliard(pi / 2, 1.0; x0=1 / 3, y0=1 / 3)
    kernel = CombinedFieldIntegralEquationSolver(8.0, billiard; grading=GlobalCornerGrading(kressq=2), eps=1e-10)
    k1, k2 = 100.1, 100.45
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    ks_ref = right_triangle_spectrum(1.0, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks_ref) == 3
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=8 and 12, Taylor backend
# billiard: Polygon, regular pentagon on the unit circle
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(area)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, GlobalCornerGrading, Taylor) - Polygon - k near 100" begin
    vertices = [SVector{2,Float64}(cos(t), sin(t)) for t in range(0, 2pi; length=6)[1:5]]
    billiard = PolygonBilliard(vertices)
    k1, k2 = high_k_window(area(billiard))
    kernel = CombinedFieldIntegralEquationSolver(8.0, billiard; grading=GlobalCornerGrading(kressq=2), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(12.0, billiard; grading=GlobalCornerGrading(kressq=2), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=6 and 9, Taylor backend
# billiard: Stadium (StadiumBilliard(0.3))
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(area)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - Stadium - k near 100" begin
    billiard = StadiumBilliard(0.3)
    k1, k2 = high_k_window(area(billiard))
    kernel = CombinedFieldIntegralEquationSolver(6.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(9.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=5 and 7.5, Taylor backend
# billiard: Limacon (LimaconBilliard(0.1))
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; the explicit window (100.09, 100.12) holds the near-degenerate
#   pair at 100.1038 (high_k_window(area) = (100, 100.0995) holds no level of this nearly circular limacon)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - Limacon - k near 100" begin
    billiard = LimaconBilliard(0.1)
    k1, k2 = 100.09, 100.12
    kernel = CombinedFieldIntegralEquationSolver(5.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(7.5, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=5 and 7.5, Taylor backend
# billiard: Polar (PolarBilliard([0.0, 0.0, 0.0, 0.05]))
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(area)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - Polar - k near 100" begin
    billiard = PolarBilliard([0.0, 0.0, 0.0, 0.05])
    k1, k2 = high_k_window(area(billiard))
    kernel = CombinedFieldIntegralEquationSolver(5.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(7.5, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=5 and 7.5, Taylor backend
# billiard: Star (StarBilliard(1.0, 0.05, 3))
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(area)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - Star - k near 100" begin
    billiard = StarBilliard(1.0, 0.05, 3)
    k1, k2 = high_k_window(area(billiard))
    kernel = CombinedFieldIntegralEquationSolver(5.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(7.5, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=5 and 7.5, Taylor backend
# billiard: C3 (C3Billiard(0.02))
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(area)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - C3 - k near 100" begin
    billiard = C3Billiard(0.02)
    k1, k2 = high_k_window(area(billiard))
    kernel = CombinedFieldIntegralEquationSolver(5.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(7.5, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=5 and 7.5, Taylor backend
# billiard: Prosen (ProsenBilliard(0.02))
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(area)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - Prosen - k near 100" begin
    billiard = ProsenBilliard(0.02)
    k1, k2 = high_k_window(area(billiard))
    kernel = CombinedFieldIntegralEquationSolver(5.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(7.5, billiard; grading=SmoothPeriodicGrading(), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=8 and 12, Taylor backend
# billiard: Mushroom (MushroomBilliard(0.5, 0.7))
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(area). Levels converge only
#   algebraically at the re-entrant corners: observed ppw 8 vs 12 difference 1.8e-4, 12 vs 16 difference 9e-5, so the
#   agreement tolerance is 5e-4 (still far below the mean level spacing of about 0.03)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, GlobalCornerGrading, Taylor) - Mushroom - k near 100" begin
    billiard = MushroomBilliard(0.5, 0.7)
    k1, k2 = high_k_window(area(billiard))
    kernel = CombinedFieldIntegralEquationSolver(8.0, billiard; grading=GlobalCornerGrading(kressq=2), eps=1e-10)
    kernel_fine = CombinedFieldIntegralEquationSolver(12.0, billiard; grading=GlobalCornerGrading(kressq=2), eps=1e-10)
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=5e-4))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral (composite, one CFIE component per boundary domain)
# kernel/basis: CFIE, SmoothPeriodicGrading, min_pts=64, ppw=5, Taylor backend
# billiard: Annular (AnnularBilliard(0.3, 0.0)), concentric, inner radius 0.3, outer radius 1
# symmetry: None
# reference: analytic roots of the annulus Bessel determinant (multiplicity 2 for m > 0); window
#   high_k_window(pi*(1 - 0.3^2)) = (100, 100.1099) holds the pair at 100.0935, edges at least 0.0149 from other levels
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, SmoothPeriodicGrading, Taylor) - Annular - k near 100" begin
    billiard = AnnularBilliard(0.3, 0.0)
    component() = CombinedFieldIntegralEquationSolver(5.0, billiard; grading=SmoothPeriodicGrading(), eps=1e-10, min_pts=64)
    kernel = CompositeBIMSolver(component(), component())
    k1, k2 = high_k_window(pi * (1 - 0.3^2))
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    ks_ref = annulus_spectrum(0.3, 1.0, k1, k2)
    ks = sort(real.(data.k))
    @test length(ks_ref) == 2
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(ks, ks_ref; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral (composite, one CFIE component per boundary domain)
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), min_pts=80, ppw=8 and 12, Taylor backend
# billiard: Sinai (SinaiBilliard(1.0, 0.25)), outer square with a circular hole
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(outer minus hole area)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, GlobalCornerGrading, Taylor) - Sinai - k near 100" begin
    billiard = SinaiBilliard(1.0, 0.25)
    ncomponents = length(QuantumBilliards._group_boundary_by_domain_id(full_boundary(billiard)))
    component(ppw) = CombinedFieldIntegralEquationSolver(ppw, billiard; grading=GlobalCornerGrading(kressq=2), eps=1e-10, min_pts=80)
    kernel = CompositeBIMSolver((component(8.0) for _ in 1:ncomponents)...)
    kernel_fine = CompositeBIMSolver((component(12.0) for _ in 1:ncomponents)...)
    k1, k2 = high_k_window(high_k_hole_area(billiard))
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=HIGH_K_ATOL))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end

# solver: Beyn contour integral (composite, one CFIE component per boundary domain)
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), min_pts=64, ppw=8 and 12, Taylor backend
# billiard: SquareWithinSquare (SquareWithinSquareBilliard(1.0)), outer square with two square holes
# symmetry: None
# reference: convergence, same window at ppw and 1.5*ppw; window high_k_window(outer minus hole areas). Levels
#   converge only algebraically at the corners: observed ppw 8 vs 12 difference 1.4e-4, 12 vs 16 difference 1.3e-4,
#   so the agreement tolerance is 5e-4 (the mean level spacing is about 0.01)
# functions to test: compute_spectrum
@testset "Beyn high k (CFIE, GlobalCornerGrading, Taylor) - SquareWithinSquare - k near 100" begin
    billiard = SquareWithinSquareBilliard(1.0)
    ncomponents = length(QuantumBilliards._group_boundary_by_domain_id(full_boundary(billiard)))
    component(ppw) = CombinedFieldIntegralEquationSolver(ppw, billiard; grading=GlobalCornerGrading(kressq=2), eps=1e-10, min_pts=64)
    kernel = CompositeBIMSolver((component(8.0) for _ in 1:ncomponents)...)
    kernel_fine = CompositeBIMSolver((component(12.0) for _ in 1:ncomponents)...)
    k1, k2 = high_k_window(high_k_hole_area(billiard))
    data = high_k_beyn_spectrum(kernel, billiard, k1, k2)
    data_fine = high_k_beyn_spectrum(kernel_fine, billiard, k1, k2)
    ks = sort(real.(data.k))
    ks_fine = sort(real.(data_fine.k))
    @test !isempty(ks)
    @test length(ks) == length(ks_fine)
    @test all(isapprox.(ks, ks_fine; atol=5e-4))
    @test all(data.ten .< HIGH_K_TENSION_TOL)
    @test all(data_fine.ten .< HIGH_K_TENSION_TOL)
    @test all(k1 .<= ks .<= k2)
end
