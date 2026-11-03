# Integration test for the Gordon-Webb-Wolpert isospectral polygon pair (two non-congruent, non-convex polygons built
# from 7 unit right isosceles triangles/squares, with identical Dirichlet spectra). The Beyn solver with the DLP kernel
# computes the spectra of both polygons in a common wavenumber window and the two level lists are compared; the
# isospectrality is the reference, so no level values are pinned.
# The polygons have reflex corners (3*pi/2) so accuracy is limited by corner resolution at ppw = 16 with global corner
# grading; tolerances SPECTRUM_ATOL_ISOSPECTRAL and IMAG_TOL_POLYGON (common.jl) are set from the observed level
# differences (worst 5.3e-4) and imaginary parts (worst 4.1e-4), see the constants there.
# The window [7.4, 9.6] is a common k-range for both polygons (their Weyl windows coincide, since areas and perimeters
# are equal). It was chosen with edges away from levels: levels observed near the edges are 7.226 and 7.567 on the lower
# side (edge 0.17 and 0.17 away) and 9.430 and 9.739 on the upper side (edge 0.17 and 0.14 away), so the level count is
# stable under the observed corner-resolution error.

using StaticArrays: SVector

_iso_tri(a, b, c) = [SVector(a...), SVector(b...), SVector(c...)]
_iso_square(x, y) = [SVector(x, y), SVector(x + 1.0, y), SVector(x + 1.0, y + 1.0), SVector(x, y + 1.0)]

# solver: Beyn contour integral
# kernel/basis: DLP, GlobalCornerGrading(kressq = 4, min_t_spacing = 1e-14), ppw = 16
# billiard: Gordon-Webb-Wolpert isospectral pair A and B (PolygonBilliard)
# symmetry: None
# reference: cross-validation, the two polygons are isospectral (A against B); common window [7.4, 9.6]
# functions to test: compute_spectrum (k-range), PolygonBilliard, DoubleLayerPotentialSolver, BeynSolver
@testset "Beyn (DLP) - Gordon-Webb-Wolpert isospectral polygons - Low Spectrum" begin
    A = PolygonBilliard([_iso_square(0.0, 0.0), _iso_square(0.0, 1.0),
        _iso_tri((0.0, 0.0), (-1.0, 0.0), (0.0, -1.0)), _iso_tri((0.0, 0.0), (1.0, 0.0), (0.0, -1.0)),
        _iso_tri((1.0, 1.0), (1.0, 2.0), (2.0, 1.0))])
    B = PolygonBilliard([_iso_square(0.0, -1.0), _iso_square(1.0, -2.0),
        _iso_tri((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), _iso_tri((0.0, 0.0), (-1.0, 0.0), (0.0, -1.0)),
        _iso_tri((1.0, -1.0), (1.0, -2.0), (0.0, -1.0))])
    k1, k2 = 7.4, 9.6
    spectrum(b) = compute_spectrum(
        BeynSolver(DoubleLayerPotentialSolver(16.0, b; grading=GlobalCornerGrading(kressq=4, min_t_spacing=1e-14))),
        b, k1, k2; multithreaded=true, show_progress=false)
    data_A = spectrum(A)
    data_B = spectrum(B)
    ks_A = sort(real.(data_A.k))
    ks_B = sort(real.(data_B.k))
    @test !isempty(ks_A)
    @test length(ks_A) == length(ks_B)
    @test all(isapprox.(ks_A, ks_B; atol=SPECTRUM_ATOL_ISOSPECTRAL))
    for data in (data_A, data_B)
        @test all(data.ten .< TENSION_TOL_ACCEL)
        @test all(abs.(data.k_im) .< IMAG_TOL_POLYGON)
    end
end
