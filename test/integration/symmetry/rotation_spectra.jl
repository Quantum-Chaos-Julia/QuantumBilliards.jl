# Integration tests for rotation-symmetry sectors (src/states/symmetry/symmetrysector.jl, the `sym_characters` keyword of
# the BIM kernels in src/solvers/sweepmethods/cfie.jl) through Beyn `compute_spectrum`.
#
# For a C_N billiard `get_symmetries` returns the single generator `NFoldRotation` (m = 1), and the one entry of
# `sym_characters` is the integer sector index s in 0:N-1 (character exp(2*pi*i*s/N) of the rotation g), not the complex
# character itself. Sector s collects the levels whose eigenfunctions satisfy psi(g r) = exp(2*pi*i*s/N) psi(r).
# For C3: sector 0 is the non-degenerate (A) irrep, sectors 1 and 2 are the complex-conjugate pair (E) that time reversal
# forces to be degenerate, so each level of sectors 1 and 2 is a doublet of the unreduced spectrum, and the unreduced
# spectrum has the sector-0 levels once and the sector-1/2 levels twice (no accidental degeneracies in the window).
# Each sector spectrum is computed once and cached in the dictionary below. The window [3, 12] (six unreduced levels)
# holds levels in every sector.

const ROTATION_SPECTRA_CACHE = Dict{Tuple,Any}()

# C3Billiard(0.2) Beyn CFIE spectrum in [3, 12] (cheap settings as in reflection_spectra.jl; observed tensions below 1e-10
# at ppw 8). `chars = ()` is the unreduced spectrum.
function _rot_c3_spectrum(chars::Tuple)
    return get!(ROTATION_SPECTRA_CACHE, (:c3, chars)) do
        billiard = C3Billiard(0.2)
        kernel = CombinedFieldIntegralEquationSolver(8.0, billiard; grading=SmoothPeriodicGrading(), sym_characters=chars)
        solver = BeynSolver(kernel; m=6, nq=16, r=16, Rmax=0.5, svd_tol=1e-10, res_tol=2e-5,
            taylor_degree=6, use_chebyshev=false, imag_k_check=false, eigenvectors=false)
        compute_spectrum(solver, billiard, 3.0, 12.0; ten_tol=nothing, multithreaded=true, show_progress=false)
    end
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=8, cheap Beyn settings
# billiard: C3 (C3Billiard(0.2))
# symmetry: C3 rotation sectors s = 0, 1, 2 (sym_characters = (s,)) against None
# reference: cross-method, unreduced Beyn spectrum of the same kernel; window [3, 12] holds 6 levels
#   (one singlet, two doublets, one more singlet: 2 levels in sector 0, 2 in each of sectors 1 and 2)
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - C3 - union of rotation sectors 0, 1, 2 equals unreduced spectrum" begin
    full_data = _rot_c3_spectrum(())
    sectors = (_rot_c3_spectrum((0,)), _rot_c3_spectrum((1,)), _rot_c3_spectrum((2,)))
    full = sort(real.(full_data.k))
    union_ks = sort(vcat((real.(d.k) for d in sectors)...))
    @test length(full) == 6
    @test length(union_ks) == length(full)
    @test all(isapprox.(union_ks, full; atol=SPECTRUM_ATOL_BIM))
    # every sector level is a level of the unreduced spectrum
    for d in sectors
        @test all(k -> minimum(abs.(full .- k)) < SPECTRUM_ATOL_BIM, real.(d.k))
    end
    # sector 0 levels occur once in the unreduced spectrum, sector 1 and 2 levels twice
    @test all(k -> count(abs.(full .- k) .< SPECTRUM_ATOL_BIM) == 1, real.(sectors[1].k))
    @test all(k -> count(abs.(full .- k) .< SPECTRUM_ATOL_BIM) == 2, real.(sectors[2].k))
    @test all(k -> count(abs.(full .- k) .< SPECTRUM_ATOL_BIM) == 2, real.(sectors[3].k))
    @test all(full_data.ten .< TENSION_TOL_BIM)
    @test all(d -> all(d.ten .< TENSION_TOL_BIM), sectors)
    @test all(d -> all(abs.(d.k_im) .< IMAG_TOL), sectors)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=8, cheap Beyn settings
# billiard: C3 (C3Billiard(0.2))
# symmetry: C3 rotation sectors 1 and 2 (complex-conjugate irreps)
# reference: time-reversal degeneracy, sectors 1 and 2 have the same levels; sector 0 shares no level with them
# functions to test: compute_spectrum
@testset "Beyn (CFIE) - C3 - rotation sectors 1 and 2 degenerate, sector 0 disjoint" begin
    s0 = sort(real.(_rot_c3_spectrum((0,)).k))
    s1 = sort(real.(_rot_c3_spectrum((1,)).k))
    s2 = sort(real.(_rot_c3_spectrum((2,)).k))
    @test length(s1) == length(s2)
    @test all(isapprox.(s1, s2; atol=SPECTRUM_ATOL_BIM))
    @test length(s1) > 0
    @test all(k -> minimum(abs.(s1 .- k)) > 0.1, s0)
    @test all(diff(s1) .> SPECTRUM_ATOL_BIM)
    @test all(diff(s0) .> SPECTRUM_ATOL_BIM)
end
