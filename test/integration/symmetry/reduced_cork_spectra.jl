# Integration tests for CORKSolver on symmetry-reduced CFIE kernels (src/solvers/acceleratedmethods/cork.jl, the `sym_characters`
# keyword of src/solvers/sweepmethods/cfie.jl) through `compute_spectrum`.
#
# The four D2 sectors of RectangleBilliard(1.0, 0.7) (the full 2.0 x 1.4 rectangle) are solved with `sym_characters = (chi_x, chi_y)`
# = `(parity_x, parity_y)` of `rectangle_spectrum`, as in reflection_spectra.jl. A reduced kernel has a small matrix (N = 50 at
# `min_pts = 200`) and the Weyl count of one fundamental-domain window of `_cork_intervals` is a fraction of a level, while the
# actual number of levels per window is 0, 1 or 2; the test checks that no level is lost and that windows without levels do not abort
# the spectrum (steps 1 and 2 of scratchpad/cork-reduced-convergence-plan.md).
#
# Window: [3, 10], split by `compute_spectrum` into the windows of `_cork_intervals` (nlevels = 4). Level counts of the sectors are
# 4, 3, 4, 5 (even-even, odd-odd, even-odd, odd-even). Settings (ppw = 10, min_pts = 200, kressq = 2, nlevels = 4) were calibrated on
# all four sectors: nlevels = 1, 2, 4, 8 and min_pts = 200, 300 return the analytic counts with worst level error 7.5e-9, worst
# tension 8.5e-15 and worst |Im k| 7.2e-10; at min_pts = 100 candidates are marked unverified (|Im k| about 1e-8, level errors up to
# 3.7e-7), so min_pts = 200 is the cheapest setting with margin. The window edges of `_cork_intervals` are fixed by the Weyl layout
# and are not shifted.
# The sector spectra are computed once and cached in the dictionary below.

const REDUCED_CORK_SPECTRA_CACHE = Dict{Tuple,Any}()

# CORK CFIE spectrum in [3, 10] of the sector `chars = (chi_x, chi_y)` of RectangleBilliard(1.0, 0.7).
function _rcork_rectangle_spectrum(chars::Tuple{Int,Int})
    return get!(REDUCED_CORK_SPECTRA_CACHE, (:rectangle, chars)) do
        billiard = RectangleBilliard(1.0, 0.7)
        kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=GlobalCornerGrading(kressq=2), sym_characters=chars, min_pts=200)
        solver = CORKSolver(kernel; nlevels=4)
        compute_spectrum(solver, billiard, 3.0, 10.0; multithreaded=false, show_progress=false)
    end
end

# solver: CORK
# kernel/basis: CFIE, GlobalCornerGrading(kressq=2), ppw=10, min_pts=200 (reduced N = 50), nlevels=4
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle)
# symmetry: D2 reflection sectors (even-even, odd-odd, even-odd, odd-even), sym_characters = (parity_x, parity_y)
# reference: analytic, `rectangle_spectrum(...; parity_x, parity_y)` (m odd is even about the centre); window [3, 10]
# functions to test: compute_spectrum (CORKSolver with a reduced CFIE kernel)
@testset "CORK (CFIE) - Rectangle - four reflection sectors vs analytic" begin
    k1, k2 = 3.0, 10.0
    sectors = ((1, 1), (-1, -1), (1, -1), (-1, 1))
    nref = 0
    for (chi_x, chi_y) in sectors
        data = _rcork_rectangle_spectrum((chi_x, chi_y))
        ks_ref = rectangle_spectrum(2.0, 1.4, k1, k2; parity_x=chi_x, parity_y=chi_y)
        nref += length(ks_ref)
        ks = sort(real.(data.k))
        @test length(ks) == length(ks_ref)
        @test all(isapprox.(ks, ks_ref; atol=SPECTRUM_ATOL_BIM))
        @test all(data.ten .< TENSION_TOL_ACCEL)
        @test all(abs.(data.k_im) .< IMAG_TOL)
    end
    @test nref == length(rectangle_spectrum(2.0, 1.4, k1, k2))
    @test length(_rcork_rectangle_spectrum((1, 1)).k) == 4
    @test length(_rcork_rectangle_spectrum((-1, -1)).k) == 3
    @test length(_rcork_rectangle_spectrum((1, -1)).k) == 4
    @test length(_rcork_rectangle_spectrum((-1, 1)).k) == 5
end
