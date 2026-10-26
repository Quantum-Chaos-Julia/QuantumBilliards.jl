# Integration tests for VerginiSaracenoSolver (src/solvers/acceleratedmethods/verginisaraceno.jl).
# Accelerated basis solver: `solve_wavenumber` (nearest level) and `solve_spectrum` (all levels
# in [k0 - dk, k0 + dk]). VS levels are non-deterministic at the 1e-4 level, so levels are
# compared with SPECTRUM_ATOL_VS and tensions only thresholded (TENSION_TOL_VS); no tensions pinned.
# Low-spectrum references: Beyn CFIE spectra (Veech triangle: full boundary; Stadium: odd-odd sector)
# computed once and cached in the file-local dictionary below.

const _VS_BEYN_CACHE = Dict{Tuple,Any}()

# Full-boundary Beyn CFIE reference for the Veech right triangle: levels in [k0 - dk, k0 + dk].
# Converged in ppw (110.0893359149 at ppw = 10, 14 and 20).
function _veech_beyn_levels(k0, dk)
    return get!(_VS_BEYN_CACHE, (:veech, k0, dk)) do
        billiard = QuantumBilliards.make_veech_right_triangle(5)
        kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading(kressq=2))
        beyn = BeynSolver(kernel; m=6, nq=16, r=16, Rmax=dk + 0.05, svd_tol=1e-10, res_tol=2e-5,
            taylor_degree=6, use_chebyshev=false, imag_k_check=false, eigenvectors=false)
        data = compute_spectrum(beyn, billiard, k0 - dk, k0 + dk + 0.05; multithreaded=true, show_progress=false)
        ks = sort(data.k[(data.k .>= k0 - dk) .& (data.k .<= k0 + dk)])
        (ks, data)
    end
end

# solver: Vergini-Saraceno
# basis: Corner Adapted Fourier-Bessel
# billiard: Veech right triangle (make_veech_right_triangle_and_basis(5))
# symmetry: None
# reference: full-boundary Beyn CFIE, computed in the testset (the old fixture 110.0894534647971 was 1.2e-4 off)
# functions to test: solve_wavenumber, solve_spectrum
@testset "VS (CornerAdaptedFourierBessel) - Veech Triangle - Low Spectrum" begin
    billiard, basis = QuantumBilliards.make_veech_right_triangle_and_basis(5)
    solver = VerginiSaracenoSolver(2.0, 5.0)
    k0, dk = 110.0, 0.1
    ks_ref, ref = _veech_beyn_levels(k0, dk)
    @test all(ref.ten .< HIGH_K_TENSION_TOL)
    k, t = solve_wavenumber(solver, basis, billiard, k0, dk; multithreaded=true)
    ks, tens = solve_spectrum(solver, basis, billiard, k0, dk; multithreaded=true)
    @test isapprox(k, ks_ref[1]; atol=SPECTRUM_ATOL_VS)
    @test t < TENSION_TOL_VS
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(sort(ks), ks_ref; atol=SPECTRUM_ATOL_VS))
    @test all(tens .< TENSION_TOL_VS)
end

# solver: Vergini-Saraceno
# basis: Corner Adapted Fourier-Bessel
# billiard: Veech right triangle (make_veech_right_triangle_and_basis(5))
# symmetry: None
# reference: fixture levels 2009.976791687898, 2010.0440083055544 (old solvertests.jl, k0=2010, dk=0.05);
#   Beyn at k = 2010 needs about 15000 boundary points (dense matrices of several GB), so no Beyn reference
# functions to test: solve_wavenumber, solve_spectrum
@testset "VS (CornerAdaptedFourierBessel) - Veech Triangle - High Spectrum" begin
    billiard, basis = QuantumBilliards.make_veech_right_triangle_and_basis(5)
    solver = VerginiSaracenoSolver(2.0, 5.0)
    k0, dk = 2010.0, 0.05
    ks_ref = [2009.976791687898, 2010.0440083055544]
    k, t = solve_wavenumber(solver, basis, billiard, k0, dk; multithreaded=true)
    ks, tens = solve_spectrum(solver, basis, billiard, k0, dk; multithreaded=true)
    @test any(isapprox.(k, ks_ref; atol=SPECTRUM_ATOL_VS))
    @test t < TENSION_TOL_VS
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(sort(ks), ks_ref; atol=SPECTRUM_ATOL_VS))
    @test all(tens .< TENSION_TOL_VS)
end

# Reduced Beyn CFIE reference for the odd-odd Stadium sector: levels in [k0 - dk, k0 + dk].
function _stadium_oddodd_beyn_levels(billiard, k0, dk)
    return get!(() -> _stadium_oddodd_beyn_compute(billiard, k0, dk), _VS_BEYN_CACHE, (:stadium, k0, dk))
end

function _stadium_oddodd_beyn_compute(billiard, k0, dk)
    kernel = CombinedFieldIntegralEquationSolver(8.0, billiard; grading=GlobalCornerGrading(kressq=2), sym_characters=(-1, -1))
    beyn = BeynSolver(kernel; m=6, nq=16, r=16, Rmax=dk + 0.05, svd_tol=1e-10, res_tol=2e-5,
        taylor_degree=6, use_chebyshev=false, imag_k_check=false, eigenvectors=false)
    data = compute_spectrum(beyn, billiard, k0 - dk, k0 + dk + 0.05; multithreaded=true, show_progress=false)
    ks = sort(data.k[(data.k .>= k0 - dk) .& (data.k .<= k0 + dk)])
    return ks, data
end

# solver: Vergini-Saraceno
# basis: Real Plane Waves
# billiard: Stadium (StadiumBilliard(0.5))
# symmetry: odd-odd
# reference: reduced Beyn CFIE (sym_characters=(-1, -1)), computed in the testset
# functions to test: solve_wavenumber, solve_spectrum
@testset "VS (RealPlaneWaves) - Stadium, odd-odd sector - Low Spectrum" begin
    billiard = StadiumBilliard(0.5)
    basis = RealPlaneWaves(12; sym_x=-1, sym_y=-1)
    solver = VerginiSaracenoSolver(5.0, 10.0)
    k0, dk = 110.0, 0.05
    ks_ref, ref = _stadium_oddodd_beyn_levels(billiard, k0, dk)
    @test all(ref.ten .< HIGH_K_TENSION_TOL)
    k, t = solve_wavenumber(solver, basis, billiard, k0, dk; multithreaded=true)
    ks, tens = solve_spectrum(solver, basis, billiard, k0, dk; multithreaded=true)
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(sort(ks), ks_ref; atol=SPECTRUM_ATOL_VS))
    @test minimum(abs.(ks_ref .- k)) < SPECTRUM_ATOL_VS
    @test t < TENSION_TOL_VS
    @test all(tens .< TENSION_TOL_VS)
end

# solver: Vergini-Saraceno
# basis: Fourier-Bessel
# billiard: Stadium (StadiumBilliard(0.5))
# symmetry: odd-odd
# reference: reduced Beyn CFIE (sym_characters=(-1, -1)), computed in the testset
# functions to test: solve_wavenumber, solve_spectrum
@testset "VS (FourierBessel) - Stadium, odd-odd sector - Low Spectrum" begin
    billiard = StadiumBilliard(0.5)
    basis = FourierBessel(12; sym_x=-1, sym_y=-1)
    solver = VerginiSaracenoSolver(5.0, 10.0)
    k0, dk = 110.0, 0.05
    ks_ref, ref = _stadium_oddodd_beyn_levels(billiard, k0, dk)
    @test all(ref.ten .< HIGH_K_TENSION_TOL)
    k, t = solve_wavenumber(solver, basis, billiard, k0, dk; multithreaded=true)
    ks, tens = solve_spectrum(solver, basis, billiard, k0, dk; multithreaded=true)
    @test length(ks) == length(ks_ref)
    @test all(isapprox.(sort(ks), ks_ref; atol=SPECTRUM_ATOL_VS))
    @test minimum(abs.(ks_ref .- k)) < SPECTRUM_ATOL_VS
    @test t < TENSION_TOL_VS
    @test all(tens .< TENSION_TOL_VS)
end
