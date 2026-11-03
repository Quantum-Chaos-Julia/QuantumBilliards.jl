# Integration test for symmetry-reduced Beyn spectra of SquareWithinSquareBilliard with one concentric square obstacle (symmetric holes,
# Phase 2): positions = [0.0] gives the D4 registry, the outer fundamental triangle and a SymmetricHole (triangle piece), G' = D4.
# Reference: cross-method, the unreduced Beyn spectrum of the same kernel (same node counts); the four D4 1D irreps
# (chi_Y, chi_Diagonal) contain every non-degenerate level and the unreduced levels left over are doubly degenerate (E).
# Caveat on convergence: the hole has four 3pi/2 corners on the diagonals. Levels with eigenfunctions that do not vanish on the
# diagonals are only converged to about 1e-3 at ppw 10 (observed spread between kressq 2, 3, 4 and ppw 10, 15), and carry |Im k| up to
# 5e-3. The sector test is therefore a consistency test on identical discretizations (reduced operator = projection of the unreduced one),
# not an accuracy test, and |Im k| is not asserted. Levels with chi_Diagonal = -1 (zero on the diagonal) are converged to 1e-9.
# Window [4, 9] (edges away from levels; nearest observed 4.14 and 8.92).

# solver: Beyn contour integral
# kernel/basis: composite CFIE (GlobalCornerGrading(kressq=2) on both loops), ppw=10, min_pts=96
# billiard: SquareWithinSquare (SquareWithinSquareBilliard(1.0; positions=[0.0], sizes=[0.5])), concentric square obstacle
# symmetry: D4 1D irreps (chi_Y, chi_Diagonal) = (+-1, +-1) against None
# reference: cross-method, unreduced Beyn spectrum of the same kernel; leftover levels are E doublets
# functions to test: compute_spectrum, symmetry_sector (reduced composite solver with a polygonal SymmetricHole)
@testset "Beyn (CFIE composite) - SquareWithinSquare centered - D4 irreps and E doublets vs unreduced" begin
    b = sh_billiard(:sws_centered) do
        SquareWithinSquareBilliard(1.0; positions=[0.0], sizes=[0.5])
    end
    @test length(get_symmetry_group(b)) == 7                 # D4 without the identity
    k1, k2 = 4.0, 9.0
    kw = (; hole_grading=GlobalCornerGrading(kressq=2))
    full_data = sh_spectrum(:sws_centered, b, k1, k2; kw...)
    full = sh_levels(full_data)
    @test length(full) > 10
    @test all(full_data.ten .< TENSION_TOL_ACCEL)

    irreps = [sh_spectrum(:sws_centered, b, k1, k2, DiagonalReflection => cd, YAxisReflection => cy; kw...) for cd in (1, -1), cy in (1, -1)]
    @test all(d -> all(d.ten .< TENSION_TOL_ACCEL), irreps)
    @test all(d -> !isempty(d.k), irreps)
    all_irrep = sort(reduce(vcat, [sh_levels(d) for d in irreps]))
    worst, left = sh_match_levels(all_irrep, full)
    @test worst < SPECTRUM_ATOL_SECTOR
    @test !isempty(left) && sh_are_doublets(left, SPECTRUM_ATOL_SECTOR)
end
