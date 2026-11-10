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
AbsBIMSolver
SweepBasisSolver
AcceleratedBasisSolver
SweepBIMSolver
AcceleratedBIMSolver
AbsState
```

## Utilities
```@docs
make_triangle_and_basis
adapt_basis
@use_threads
@blas_multi
@blas_1
@blas_multi_then_1
try_MKL!
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
boundary_s
component_offsets
points_in_billiard
BoundaryPanelArrays
BoundaryGeomCache
boundary_geom_cache
component_normals
flatten_boundary_components
flatten_boundary_ds
boundary_matrix_size
basis_matrix
basis_and_gradient_matrices
dk_matrix
VerginiSaracenoSolver
DecompositionMethodSolver
ParticularSolutionsMethodSolver
BoundaryGrading
SmoothPeriodicGrading
CornerGrading
GlobalCornerGrading
DoubleLayerPotentialSolver
CombinedFieldIntegralEquationSolver
CompositeBIMSolver
ExpandedBIMSolver
BeynSolver
CORKSolver
ChebyshevConfig
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
solve_state
weyl_window_width
plan_weyl_windows
beyn_disks_from_windows
beyn_buffer_matrices
log_solver
```

## Computation planning
```@docs
plan_computation
ComputationPlan
WindowPlan
expected_states
with_params
estimate_memory
MemoryEstimate
default_memory_limit
```

## Spectra
```@docs
SpectralData
compute_spectrum
overlap_and_merge!
overlap_and_merge_ebim!
k_range
sweep_windows
weyl_window_count
weyl_law
state_at_k
k_at_state
spectral_density
k_range_for_states
```

## Symmetry
```@docs
SymmetrySector
symmetry_sector
symmetry_irreps
log_symmetries
symmetry_degeneracy
symmetry_partner
```

## States
```@docs
BasisEigenstate
BasisState
BIMEigenstate
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

