# Integration tests for reflection-symmetry sectors (src/states/symmetry/symmetrysector.jl, the `sym_characters` keyword of
# the BIM kernels in src/solvers/sweepmethods/cfie.jl) through Beyn `compute_spectrum` and `solve_wavenumber`.
#
# `sym_characters` selects the irrep per generator of `BilliardGeometry.get_symmetries(billiard)`, in that order. All D2
# billiards used here have the generators `(YAxisReflection, XAxisReflection)`, so `sym_characters = (chi_x, chi_y)` with
# chi = +1 (even) or -1 (odd) under x -> -x and y -> -y, i.e. `(parity_x, parity_y)` of `rectangle_spectrum`.
# A reduced spectrum is the set of full-boundary levels whose eigenfunctions belong to that irrep, so the four D2 sector
# spectra together must reproduce the unreduced spectrum (these billiards have no accidental degeneracies in the windows).
# The unreduced and the sector spectra are computed once and cached in the dictionary below.
#
# Level tolerances: Rectangle and Circle against analytic levels use SPECTRUM_ATOL_BIM. Reduced and unreduced discretizations
# of a graded kernel carry different quadrature errors, observed up to 1.1e-10 (Stadium), so the union check of the Stadium
# uses SPECTRUM_ATOL_BASIS.

const REFLECTION_SPECTRA_CACHE = Dict{Tuple,Any}()

# Beyn solver with the cheap settings of verginisaraceno_spectra.jl (calibrated: tensions below 1e-9 for CFIE kernels at these ppw).
_refl_beyn_light(kernel) = BeynSolver(kernel; m=6, nq=16, r=16, Rmax=0.5, svd_tol=1e-10, res_tol=2e-5,
    taylor_degree=6, use_chebyshev=false, imag_k_check=false, eigenvectors=false)

# Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle) Beyn CFIE spectrum in [3, 10], default settings of
# beyn_spectra.jl. `chars = ()` is the unreduced spectrum.
function _refl_rectangle_spectrum(chars::Tuple)
    return get!(REFLECTION_SPECTRA_CACHE, (:rectangle, chars)) do
        billiard = RectangleBilliard(1.0, 0.7)
        kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading(), sym_characters=chars)
        solver = BeynSolver(kernel; use_chebyshev=false, eigenvectors=false)
        compute_spectrum(solver, billiard, 3.0, 10.0; multithreaded=true, show_progress=false)
    end
end

# Stadium(0.5) Beyn CFIE spectrum in [3, 7] (cheap settings, ppw 8, kressq 2 as in verginisaraceno_spectra.jl).
function _refl_stadium_spectrum(chars::Tuple)
    return get!(REFLECTION_SPECTRA_CACHE, (:stadium, chars)) do
        billiard = StadiumBilliard(0.5)
        kernel = CombinedFieldIntegralEquationSolver(8.0, billiard; grading=GlobalCornerGrading(kressq=2), sym_characters=chars)
        compute_spectrum(_refl_beyn_light(kernel), billiard, 3.0, 7.0; ten_tol=nothing, multithreaded=true, show_progress=false)
    end
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: D2 reflection sectors (even-even, odd-odd, even-odd, odd-even), sym_characters = (parity_x, parity_y)
# reference: analytic, `rectangle_spectrum(...; parity_x, parity_y)` (m odd is even about the centre); window [3, 10]
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Rectangle - four reflection sectors vs analytic" begin
    k1, k2 = 3.0, 10.0
    even_even = _refl_rectangle_spectrum((1, 1))
    odd_odd = _refl_rectangle_spectrum((-1, -1))
    even_odd = _refl_rectangle_spectrum((1, -1))
    odd_even = _refl_rectangle_spectrum((-1, 1))
    ref_even_even = rectangle_spectrum(2.0, 1.4, k1, k2; parity_x=1, parity_y=1)
    ref_odd_odd = rectangle_spectrum(2.0, 1.4, k1, k2; parity_x=-1, parity_y=-1)
    ref_even_odd = rectangle_spectrum(2.0, 1.4, k1, k2; parity_x=1, parity_y=-1)
    ref_odd_even = rectangle_spectrum(2.0, 1.4, k1, k2; parity_x=-1, parity_y=1)
    @test length(ref_even_even) + length(ref_odd_odd) + length(ref_even_odd) + length(ref_odd_even) == length(rectangle_spectrum(2.0, 1.4, k1, k2))
    @test length(sort(real.(even_even.k))) == length(ref_even_even)
    @test all(isapprox.(sort(real.(even_even.k)), ref_even_even; atol=SPECTRUM_ATOL_BIM))
    @test length(odd_odd.k) == length(ref_odd_odd)
    @test all(isapprox.(sort(real.(odd_odd.k)), ref_odd_odd; atol=SPECTRUM_ATOL_BIM))
    @test length(even_odd.k) == length(ref_even_odd)
    @test all(isapprox.(sort(real.(even_odd.k)), ref_even_odd; atol=SPECTRUM_ATOL_BIM))
    @test length(odd_even.k) == length(ref_odd_even)
    @test all(isapprox.(sort(real.(odd_even.k)), ref_odd_even; atol=SPECTRUM_ATOL_BIM))
    for data in (even_even, odd_odd, even_odd, odd_even)
        @test all(data.ten .< TENSION_TOL_ACCEL)
        @test all(abs.(data.k_im) .< IMAG_TOL)
    end
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading, ppw=10
# billiard: Rectangle (RectangleBilliard(1.0, 0.7))
# symmetry: D2 sectors against None
# reference: cross-method, unreduced Beyn spectrum of the same kernel (the four sector spectra reused from the testset above)
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Rectangle - union of four sectors equals unreduced spectrum" begin
    k1, k2 = 3.0, 10.0
    full = sort(real.(_refl_rectangle_spectrum(()).k))
    union_ks = sort(vcat(real.(_refl_rectangle_spectrum((1, 1)).k), real.(_refl_rectangle_spectrum((-1, -1)).k),
        real.(_refl_rectangle_spectrum((1, -1)).k), real.(_refl_rectangle_spectrum((-1, 1)).k)))
    ks_ref = rectangle_spectrum(2.0, 1.4, k1, k2)
    @test length(full) == length(ks_ref)
    @test length(union_ks) == length(full)
    @test all(isapprox.(union_ks, full; atol=SPECTRUM_ATOL_BIM))
    @test all(diff(union_ks) .> SPECTRUM_ATOL_BIM)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=8, cheap Beyn settings
# billiard: Stadium (StadiumBilliard(0.5))
# symmetry: D2 sectors (even-even, odd-odd, even-odd, odd-even) against None
# reference: cross-method, unreduced Beyn spectrum of the same kernel; window [3, 7] holds 14 levels
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - Stadium - union of four reflection sectors equals unreduced spectrum" begin
    full_data = _refl_stadium_spectrum(())
    sectors = (_refl_stadium_spectrum((1, 1)), _refl_stadium_spectrum((-1, -1)), _refl_stadium_spectrum((1, -1)), _refl_stadium_spectrum((-1, 1)))
    full = sort(real.(full_data.k))
    union_ks = sort(vcat((real.(d.k) for d in sectors)...))
    @test length(union_ks) == length(full)
    @test all(isapprox.(union_ks, full; atol=SPECTRUM_ATOL_BASIS))
    @test all(diff(full) .> SPECTRUM_ATOL_BASIS)
    # every sector level is a level of the unreduced spectrum, and each level belongs to exactly one sector
    for d in sectors
        @test all(k -> minimum(abs.(full .- k)) < SPECTRUM_ATOL_BASIS, real.(d.k))
    end
    @test all(k -> count(d -> any(abs.(real.(d.k) .- k) .< SPECTRUM_ATOL_BASIS), sectors) == 1, full)
    @test all(full_data.ten .< TENSION_TOL_BIM)
    @test all(d -> all(d.ten .< TENSION_TOL_BIM), sectors)
end

# solver: Combined Field Integral Equation
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10
# billiard: Circle, radius 1
# symmetry: even-even sector, sym_characters = (1, 1)
# reference: analytic, j_{0,1} = 2.4048 (rotationally symmetric m = 0 mode is even under both reflections; the window
#   [2.2, 2.6] holds it alone)
# functions to test: solve_wavenumber
@testset "CFIE - Circle - even-even sector - Ground State" begin
    billiard = CircleBilliard(1.0)
    solver = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading(), sym_characters=(1, 1))
    k_ref = circle_spectrum(1.0, 2.0, 3.0)[1]
    k, t = solve_wavenumber(solver, billiard, 2.4, 0.2; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end

# solver: Combined Field Integral Equation
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10
# billiard: Circle, radius 1
# symmetry: odd-odd sector, sym_characters = (-1, -1)
# reference: analytic, j_{2,1} = 5.1356 (sin(2 theta) J_2 is the lowest mode odd under both reflections; the neighbours
#   j_{1,1} = 3.8317 and j_{0,2} = 5.5201 lie outside the window [4.93, 5.33])
# functions to test: solve_wavenumber
@testset "CFIE - Circle - odd-odd sector - First Level" begin
    billiard = CircleBilliard(1.0)
    solver = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading(), sym_characters=(-1, -1))
    k_ref = circle_spectrum(1.0, 5.0, 5.3)[1]
    k, t = solve_wavenumber(solver, billiard, 5.13, 0.2; multithreaded=true)
    @test isapprox(k, k_ref; atol=SPECTRUM_ATOL_BIM)
    @test t < TENSION_TOL_BIM
end
