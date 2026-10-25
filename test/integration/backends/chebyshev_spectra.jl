# Integration tests for the Chebyshev matrix backend (`use_chebyshev=true`) of BeynSolver through `compute_spectrum`.
# Chebyshev results are compared with the direct backend (`use_taylor=false, use_chebyshev=false`) or with analytic
# levels.

# Direct-backend results are reused between testsets.
const CHEB_DIRECT_CACHE = Dict{Symbol,Any}()

function cheb_veech_direct_beyn(k1, k2)
    return get!(CHEB_DIRECT_CACHE, :beyn_dlp_veech) do
        billiard = QuantumBilliards.make_veech_right_triangle(5)
        kernel = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
        solver = BeynSolver(kernel; use_taylor=false, use_chebyshev=false)
        compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    end
end

# solver: Beyn contour integral
# kernel/basis: DLP, GlobalCornerGrading, ppw=5, Chebyshev backend
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: direct backend of the same solver, and the Beyn CFIE value 6.065090986269762 (the only level in [5.5, 6.7])
# functions to test: compute_spectrum
@testset "Beyn (DLP, Chebyshev) - Full Triangle - Ground State" begin
    billiard = QuantumBilliards.make_veech_right_triangle(5)
    kernel = DoubleLayerPotentialSolver(5.0, billiard; grading=GlobalCornerGrading())
    solver = BeynSolver(kernel; use_taylor=false, use_chebyshev=true)
    k1, k2 = 5.5, 6.7
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    data_direct = cheb_veech_direct_beyn(k1, k2)
    k_ref = 6.065090986269762
    @test length(data.k) == 1
    @test length(data_direct.k) == 1
    @test isapprox(real(data.k[1]), real(data_direct.k[1]); atol=SPECTRUM_ATOL_BIM)
    @test isapprox(real(data.k[1]), k_ref; atol=SPECTRUM_ATOL_BIM)
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10, Chebyshev backend
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic, j_{0,1} = 2.4048 (window [2, 3] holds it alone)
# functions to test: compute_spectrum
@testset "Beyn (CFIE, Chebyshev) - Circle - Ground State" begin
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    solver = BeynSolver(kernel; use_taylor=false, use_chebyshev=true)
    k1, k2 = 2.0, 3.0
    data = compute_spectrum(solver, billiard, k1, k2; multithreaded=false, show_progress=false)
    ks_ref = circle_spectrum(1.0, k1, k2)
    @test length(ks_ref) == 1
    @test length(data.k) == 1
    @test isapprox(real(data.k[1]), ks_ref[1]; atol=SPECTRUM_ATOL_BIM)
    @test all(data.ten .< TENSION_TOL_ACCEL)
    @test all(abs.(data.k_im) .< IMAG_TOL)
end
