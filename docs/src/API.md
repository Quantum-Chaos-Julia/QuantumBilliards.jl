# API
The following pages document and explain the functionality of all exported types
and functions in the library.

```@meta
CurrentModule = QuantumBilliards
```

## Index
```@index
```

## Abstract types
```@docs
AbsBasis
AbsSolver
AbsBasisSolver
SweepBasisSolver
AcceleratedBasisSolver
```

## Utilities
```@docs
make_triangle_and_basis
adapt_basis
```

## Basis
```@docs
RealPlaneWaves
CornerAdaptedFourierBessel
FourierBessel
fb_modes
resize_basis
basis_fun
dk_fun
gradient
basis_and_gradient
ca_fb
ca_fb_dk
fb_term
fb_term_dk
```

## Solvers
```@docs
BoundaryPoints
basis_matrix
basis_and_gradient_matrices
dk_matrix
VerginiSaracenoSolver
DecompositionMethodSolver
evaluate_points
construct_matrices
solve
solve_vect
solve_wavenumber
solve_spectrum
k_sweep
boundary_coords
adjust_scaling_and_samplers
generalized_eigen
generalized_eigvals
sm_results
solve_vectors
```

## Spectra
```@docs
```
Missing documentation: `SpectralData`, `compute_spectrum`, `merge_spectra`,
`overlap_and_merge!`, `weyl_law` — defined in `src/spectra/spectralutils.jl`
and `src/spectra/unfolding.jl`, none currently have docstrings.

## States
```@docs
BasisEigenstate
BasisState
GaussianRandomState
compute_eigenstate
wavefunction
compute_psi
boundary_limits
get_boundary_curves_with_ignored
boundary_function
momentum_function
husimi_function
pad_limits
rectify_grid
apply_symmetries_to_wavefunction
apply_symmetries_to_boundary_function
apply_symmetries_to_boundary_points
regularize!
_rellich
antisym_vec
```

