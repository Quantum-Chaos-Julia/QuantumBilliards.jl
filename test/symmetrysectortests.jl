using StaticArrays
using BilliardGeometry

# Cross-package smoke test: a `SymmetrySector` for a real D2 billiard,
# resolved via `_resolve_bim_symmetry` and fed into
# `BilliardGeometry.symmetry_index_orbits`, must reduce the boundary to the
# same group order/fundamental-domain fraction as the same billiard's
# `RealPlaneWaves(dim, billiard, sector)` basis's effective symmetry-reduced
# quadrant count -- the one place the geometric-orbit-folding (Layer 1) and
# basis/BIM representation (Layer 2) layers should agree with each other end
# to end. Both quantities are computed directly from the actual code (no
# fabricated reference values).
@testset "SymmetrySector - BilliardGeometry orbit-folding agreement" begin
    billiard = RectangleBilliard(1.0, 0.6)
    sector = symmetry_sector(billiard, XAxisReflection=>-1, YAxisReflection=>1)

    generator, characters = QuantumBilliards._resolve_bim_symmetry(billiard, sector)
    @test generator isa CompositeReflection

    N = 40
    orbits = symmetry_index_orbits(Float64, billiard, N, generator, ComplexF64.(collect(characters)))
    # Full D2 reduction: the boundary folds into 4-element orbits (order of
    # the resolved reflection group), i.e. a quarter of the full boundary.
    @test orbit_size(orbits) == 4
    @test fundamental_size(orbits) == N ÷ 4

    basis = RealPlaneWaves(8, billiard, sector)
    patterns = unique(collect(zip(basis.parity_x, basis.parity_y)))
    # RealPlaneWaves collapses the same full D2 sector to a single quadrant
    # (one cos/sin pattern), matching the geometric orbit's group order 4.
    @test length(patterns) == 1
    @test orbit_size(orbits) == 4 ÷ length(patterns)
    @test basis.dim == 8 * length(patterns)

    # Resolving a sector against a different billiard instance (even a
    # geometrically identical one) must be rejected: sym_id assignment is
    # per-billiard registration order.
    other_billiard = RectangleBilliard(1.0, 0.6)
    @test_throws ArgumentError QuantumBilliards._resolve_bim_symmetry(other_billiard, sector)
end
