# Tutorial
This section will guide you through getting started with QuantumBilliards.jl.

## Installation
TODO: `Pkg.add`/`Pkg.dev` instructions.

## Core components
TODO: brief overview of the main types/functions a new user should know about,
linking to them with `[`Symbol`](@ref)`.

## Billiards with holes and symmetry
A billiard with holes has the symmetry group `G'` of the elements of its
registered group `G` that map every hole onto itself
(`BilliardGeometry.get_symmetry_group`). Holes that carry their own symmetry are
built from one fundamental piece with `SymmetricHole`; copies of a hole placed
by the symmetry of the billiard are built with `SymmetrizedHoles`. The origin-centered
`SinaiBilliard` (`D₄`), `AnnularBilliard` (mirror `y → -y`) and a single
concentric `SquareWithinSquareBilliard` (`D₄`) are built this way; a nonzero
`center` gives the unreduced billiard with an empty registry.

A composite boundary-integral solver acts on the reduced billiard through the
irreducible representations of `G'`. The one-dimensional irreps are selected
with `symmetry_sector`; `symmetry = nothing` solves the whole billiard.

```julia
b = SinaiBilliard(1.0, 0.3)
sector = symmetry_sector(b, DiagonalReflection => 1, YAxisReflection => -1)
```

A symmetry that a hole breaks is rejected with an error that names `G'` and the
first offending hole; use a subgroup of `G'` or `symmetry = nothing`.
