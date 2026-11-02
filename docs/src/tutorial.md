# Tutorial
This section will guide you through getting started with QuantumBilliards.jl.

## Installation
```julia
using Pkg
Pkg.add(url="https://github.com/Quantum-Chaos-Julia/QuantumBilliards.jl")
```

## Core components
A computation combines a billiard from `BilliardGeometry.jl`, a solver and,
for basis solvers, a basis. Boundary-integral solvers such as
[`BeynSolver`](@ref) and [`CORKSolver`](@ref) wrap a kernel
([`DoubleLayerPotentialSolver`](@ref), [`CombinedFieldIntegralEquationSolver`](@ref)
or [`CompositeBIMSolver`](@ref)); basis solvers such as
[`VerginiSaracenoSolver`](@ref) use a basis such as [`FourierBessel`](@ref),
[`CornerAdaptedFourierBessel`](@ref) or [`RealPlaneWaves`](@ref).
[`compute_spectrum`](@ref) returns a [`SpectralData`](@ref) from which
[`wavefunction`](@ref), [`boundary_function`](@ref) and
[`husimi_function`](@ref) reconstruct the states.

```julia
using QuantumBilliards, BilliardGeometry
billiard = C3Billiard(0.2)
kernel = DoubleLayerPotentialSolver(8.0, billiard)
solver = BeynSolver(kernel; nlevels = 10)
k1, k2 = k_range_for_states(billiard, 250, 257)
spectrum = compute_spectrum(solver, billiard, k1, k2)
```

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

For a dihedral group `D_d` the two-dimensional irreps `E_k` (`1 ≤ k < d/2`) are
selected with the `irrep2d` keyword. `E_k` restricts to the cyclic sectors `s = k`
and `s = d-k`, so the solve runs in the cyclic sector `s = k`, which contains every
`E_k` level once; each level is twofold degenerate in the full problem
(`symmetry_degeneracy(sector) == 2`). `symmetry_irreps(b).irreps_2d` lists the valid
`k`, and `symmetry_partner(f, b)` gives the degenerate partner `f∘σ` of a state.

```julia
b = SinaiBilliard(1.0, 0.3)               # D₄: E₁ is the only 2D irrep
sector = symmetry_sector(b; irrep2d = 1)
```
