
"""
CoordinateSystem

`CoordinateSystem` is the abstract supertype for local coordinate frames used
to evaluate basis functions and their gradients.

## Description
A `CoordinateSystem` bundles an origin, a rotation angle and the corresponding
affine maps (and their inverses) needed to transform points between the global
Cartesian frame and a local frame in which basis functions are naturally
expressed, e.g. a Cartesian frame aligned with a symmetry axis, or a polar
frame centered at a corner. Concrete subtypes are [`CartesianCS`](@ref) and
[`PolarCS`](@ref).
"""
abstract type CoordinateSystem end

"""
AbsBasis

`AbsBasis` is the abstract supertype for all basis representations used to
approximate eigenstates and boundary solutions of a quantum billiard.

## Description
A concrete subtype of `AbsBasis` stores the parameters defining a family of
basis functions (e.g. plane waves or corner-adapted Fourier-Bessel functions)
at a fixed dimension `dim` and wavenumber. Solvers combine an `AbsBasis` with
a boundary discretization to construct the matrices used to determine
eigenvalues and eigenvectors. Concrete subtypes include
[`RealPlaneWaves`](@ref) and [`CornerAdaptedFourierBessel`](@ref).

## API
Every concrete subtype of `AbsBasis` is expected to implement:
- `resize_basis`
- `basis_fun`
- `gradient`
- `basis_and_gradient`
"""
abstract type AbsBasis end

"""
AbsSolver

`AbsSolver` is the abstract supertype for all algorithms that determine
quantum billiard eigenvalues (wavenumbers) and eigenvectors from a boundary
discretization.

## Description
`AbsSolver` is the top-level abstraction shared by every solver algorithm in
the package, regardless of the representation used to determine the spectrum.
Its concrete-algorithm branch is [`AbsBasisSolver`](@ref), the supertype of
all solvers that determine the spectrum by expanding the solution in a basis.

## API
The following functions can be evaluated for any `AbsSolver`:
- `evaluate_points`
- [`adjust_scaling_and_samplers`](@ref)
- `compute_spectrum`
"""
abstract type AbsSolver end

"""
AbsBIMSolver <: AbsSolver

`AbsBIMSolver` is the abstract supertype for all algorithms that determine
quantum billiard eigenvalues (wavenumbers) from a Nyström/boundary-integral
discretization of a Fredholm operator acting directly on unknown boundary
densities.

## Description
Unlike [`AbsBasisSolver`](@ref), no [`AbsBasis`](@ref) expansion is involved:
the unknowns are the boundary density values themselves, sampled at the
boundary discretization points. `AbsBIMSolver` mirrors the
sweep/accelerated split of `AbsBasisSolver` through its two direct
concrete-algorithm branches, [`SweepBIMSolver`](@ref) and
[`AcceleratedBIMSolver`](@ref).

## API
The following functions can be evaluated for any `AbsBIMSolver`:
- `evaluate_points`
- `construct_matrices`
- `solve`
"""
abstract type AbsBIMSolver <: AbsSolver end

"""
SweepBIMSolver <: AbsBIMSolver

`SweepBIMSolver` is the abstract supertype for boundary-integral solvers that
locate quantum billiard eigenvalues by sweeping over a range of individual
wavenumbers and minimizing a tension function at each one.

## Description
At each wavenumber `k`, a `SweepBIMSolver` assembles the Fredholm matrix
`A(k)` (see `construct_matrices`) from a boundary discretization and defines
the tension as a function of the smallest singular value / nullspace residual
of `A(k)` (see `solve`); scanning this tension over a range of wavenumbers
locates the billiard's eigenvalues, exactly as [`SweepBasisSolver`](@ref) does
for basis-expansion solvers. Concrete implementations are
[`DoubleLayerPotentialSolver`](@ref), [`CombinedFieldIntegralEquationSolver`](@ref)
and [`CompositeBIMSolver`](@ref). `solve_state` is generic across the whole
branch: it assembles `A(k)` once and reuses the *same* Krylov singular-value
solve `solve_vect` performs to also recover the physical boundary normal
derivative `∂ₙψ`, via `_bim_normal_derivative`'s Helmholtz-kernel-reciprocity
relation between `A(k)`'s left singular vector and its weighted-transpose
adjoint's right singular vector — no concrete solver needs to implement this
itself unless its kernel does not follow that reciprocity convention.

## API
The following functions can be evaluated for any `SweepBIMSolver`:
- `evaluate_points`
- `construct_matrices`
- `solve`
- `solve_vect`
- [`solve_wavenumber`](@ref)
- [`k_sweep`](@ref)
- `solve_state`
"""
abstract type SweepBIMSolver <: AbsBIMSolver end

abstract type DLP <: SweepBIMSolver end
abstract type CFIE <: SweepBIMSolver end

"""
AcceleratedBIMSolver <: AbsBIMSolver

`AcceleratedBIMSolver` is the abstract supertype for boundary-integral solvers
that recover eigenvalues near a target wavenumber `k` from the nonlinear
eigenproblem `A(k)v = 0` without a wavenumber-by-wavenumber sweep.

## Description
Every concrete `AcceleratedBIMSolver` wraps an inner `kernel::SweepBIMSolver`
(a [`DoubleLayerPotentialSolver`](@ref), [`CombinedFieldIntegralEquationSolver`](@ref)
or [`CompositeBIMSolver`](@ref)) supplying the Fredholm operator, and adds its
own root-finding strategy on top of it: [`BeynSolver`](@ref) recovers every
root within a contour via Beyn's contour-integral method, while
[`ExpandedBIMSolver`](@ref) recovers a single locally-corrected root via a
second-order local Taylor expansion of `A(k)`. Every concrete
`AcceleratedBIMSolver` also stores every one of its own numerical tuning
parameters (Beyn's `m`/`nq`/`r`/`svd_tol`/`res_tol`, EBIM's Chebyshev
configuration) as fields, so its widest-scope entry point,
[`compute_spectrum`](@ref), takes only the wavenumber range `[k1,k2]` (plus a
small number of merge-strategy keyword arguments) — unlike `-develop`'s
`solve_spectrum_beyn`/`solve_spectrum_ebim`, which scatter these same knobs
across dozens of call-site keyword arguments because `-develop`'s
`BeynSolver`/`EBIMSolver` are mere `Union` traits, not parameter-holding
structs. `BeynSolver`'s [`compute_spectrum`](@ref) covers `[k1,k2]` with
consecutive Weyl-balanced contour windows and concatenates each window's
already-filtered result (no fuzzy overlap merge needed, windows are
disjoint by construction); `ExpandedBIMSolver`'s corrects a dense adaptive
grid of trial wavenumbers and merges the densely-overlapping results with
[`overlap_and_merge_ebim!`](@ref) (a spacing-adaptive clustering merge,
unlike the window-boundary-based [`overlap_and_merge!`](@ref) used
elsewhere).

## API
The following functions can be evaluated for any `AcceleratedBIMSolver`:
- `evaluate_points`
- `construct_matrices`
- `solve`
- [`solve_wavenumber`](@ref)
- `solve_spectrum`
- [`compute_spectrum`](@ref)
"""
abstract type AcceleratedBIMSolver <: AbsBIMSolver end

abstract type AbsState end

"""
AbsBasisSolver <: AbsSolver

`AbsBasisSolver` is the abstract supertype for all algorithms that determine
quantum billiard eigenvalues (wavenumbers) and eigenvectors from a boundary
discretization by expanding the solution in a basis.

## Description
Concrete solvers hold the parameters controlling boundary sampling and basis
dimension scaling (e.g. `dim_scaling_factor`, `pts_scaling_factor`, `sampler`,
`min_dim`, `min_pts`) as well as the numerical tolerance `eps` used when
filtering the generalized eigenvalue problem. `AbsBasisSolver` has two direct
concrete-algorithm branches, [`SweepBasisSolver`](@ref) and
[`AcceleratedBasisSolver`](@ref), distinguished by whether the spectrum is
scanned one wavenumber at a time or obtained in windows via a single
diagonalization.

## API
The following functions can be evaluated for any `AbsBasisSolver`:
- `evaluate_points`
- [`adjust_scaling_and_samplers`](@ref)
- `compute_spectrum`
"""
abstract type AbsBasisSolver <: AbsSolver end

"""
SweepBasisSolver <: AbsBasisSolver

`SweepBasisSolver` is the abstract supertype for solvers that locate quantum
billiard eigenvalues by sweeping over a range of individual wavenumbers and
minimizing a tension function at each one.

## Description
At each wavenumber `k`, a `SweepBasisSolver` constructs matrices from a
boundary quadrature and solves a generalized eigenvalue problem whose
smallest eigenvalue defines a tension quantifying how well the boundary
condition is satisfied; scanning this tension over a range of wavenumbers
locates the billiard's eigenvalues. The concrete implementation is
[`DecompositionMethodSolver`](@ref).

## API
The following functions can be evaluated for any `SweepBasisSolver`:
- `construct_matrices`
- `solve`
- `solve_vect`
- [`solve_wavenumber`](@ref)
- [`k_sweep`](@ref)
- [`compute_eigenstate`](@ref)
"""
abstract type SweepBasisSolver <: AbsBasisSolver end

"""
AcceleratedBasisSolver <: AbsBasisSolver

`AcceleratedBasisSolver` is the abstract supertype for solvers that recover
every eigenvalue within a wavenumber window `dk` of a target wavenumber `k`
from a single diagonalization.

## Description
An `AcceleratedBasisSolver` constructs a generalized eigenvalue problem whose
spectrum, restricted to the window around `k`, approximates the tensions of
all billiard eigenstates in that window, avoiding the need to scan
wavenumber-by-wavenumber as [`SweepBasisSolver`](@ref) does. The concrete
implementation is [`VerginiSaracenoSolver`](@ref).

## API
The following functions can be evaluated for any `AcceleratedBasisSolver`:
- `construct_matrices`
- `solve`
- `solve_vectors`
- [`solve_wavenumber`](@ref)
- `solve_spectrum`
- [`compute_eigenstate`](@ref)
"""
abstract type AcceleratedBasisSolver <: AbsBasisSolver end

"""
AbsState

`AbsState` is the abstract supertype for all representations of a quantum
billiard state expressed through a coefficient vector `vec` at a wavenumber
`k`.

## Description
Every concrete subtype of `AbsState` stores at least a wavenumber `k`, the
wavenumber `k_basis` at which the associated basis coefficients `vec` were
evaluated, the dimension `dim` of `vec`, and a numerical precision threshold
`eps` (see `set_precision`) below which coefficients are treated as zero.
[`AbsState`](@ref) is the branch of `AbsState` for states expressed in
a genuine [`AbsBasis`](@ref); [`GaussianRandomState`](@ref) is a direct
`AbsState` subtype not tied to any specific basis, used as a random-wave
reference ensemble.

## API
The following functions can be evaluated for any `AbsState`:
- `boundary_function`
- `momentum_function`
- `wavefunction`
- `husimi_function`
"""

