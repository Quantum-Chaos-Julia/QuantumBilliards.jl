# Step 1 — Quick wins: `DecompositionMethodSolver.rellich_origin` + `ParticularSolutionsMethod`

## Goal

Land the two `SweepBasisSolver` items that need **zero** new
`BilliardGeometry.jl` infrastructure and reuse 100% of the existing
basis-solver machinery, before touching anything boundary-integral. Confirms
the existing Triangle/Stadium fixtures and generic sweep infrastructure are
solid before Step 2 starts extending `BoundaryPoints`.

## Preconditions

None — this is the first step.

## Part A — verify `rellich_origin` (should already be done)

[solvers/sweepmethods/decompositionmethod.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/decompositionmethod.jl)
already has the `rellich_origin::SVector{2,T}` field, both constructors
default it to the origin, and `evaluate_points` computes
`rn = dot.(xy .- Ref(solver.rellich_origin), normal)`. There is nothing left
to port. Confirm only:

1. `get_errors` on the file and on `QuantumBilliards.jl` — must be clean.
2. Run the existing `test/solvertests.jl` "Decomposition Method"/"Vergini
   Saraceno" testsets unaffected (the default `rellich_origin` must reproduce
   the exact numbers already recorded there, since it defaults to the
   origin — i.e. behaviorally a no-op change).
3. Manually construct `DecompositionMethodSolver(2.0, 5.0; rellich_origin =
   SVector(0.1, 0.0))` against `StadiumBilliard(0.5)` and confirm
   `evaluate_points` runs and returns finite `w_dm` values (no numerical
   regression test needed yet for a non-default origin — that's the Julia
   Test Writer's job at the end of this step).

If any of the above fails, fix `decompositionmethod.jl` directly before
moving on — do not proceed to Part B with a broken baseline.

## Part B — `ParticularSolutionsMethod`

### Source index

* `QuantumBilliards-develop/src/solvers/sweepmethods/basis_sweep/particular_solutions_method.jl`
  — full solver: struct, both constructors, `evaluate_points`,
  `construct_matrices`(+`_benchmark`), `solve_full`, `solve_with_rank_reduction`,
  `solve`, `solve_vect`.
* `QuantumBilliards-develop/src/solvers/sweepmethods/basis_sweep/basis_sweep_methods.jl`
  — check for any `SweepSolver`-wide generic PSM relies on beyond
  `solve_wavenumber`/`k_sweep` (these two are already implemented generically
  for every `SweepBasisSolver` in main's `sweepmethods.jl` — do not re-port
  them).
* `random_interior_points(billiard, M_int)` — used by `evaluate_points`; search
  `QuantumBilliards-develop` and `BilliardGeometry-develop` for its
  definition (likely a billiard-geometry utility, e.g. rejection-sampling
  interior points via `is_inside`). If it lives in `BilliardGeometry-develop`,
  port it into `BilliardGeometry.jl/src/geometry/utils.jl` (or the file
  containing `is_inside`) alongside the other geometry utilities, following
  the `julia-refactor-wire-feature` skill. If it lives in
  `QuantumBilliards-develop` itself (solver-adjacent utility), port it into
  `QuantumBilliards.jl/src/solvers/boundarypoints.jl` next to
  `points_in_billiard`.

### Destination

[solvers/sweepmethods/particularsolutions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/particularsolutions.jl)
already has the struct, both constructors, and every method signature
stubbed with `error(_PSM_NOT_IMPLEMENTED)`. This step only replaces the
method **bodies** — do not change the struct fields or constructor
signatures, they're already the committed API (Step 1 plan §4.1).

### Implementation steps

1. Port `random_interior_points` first (see above) so `evaluate_points` has
   something to call.
2. Replace the `evaluate_points` stub body with the ported logic: sample each
   fundamental boundary curve exactly like
   `DecompositionMethodSolver.evaluate_points` does (reuse
   `adjust_scaling_and_samplers`, `_determine_bp_sizes`, `curve`,
   `domain_gradient_vector`/normal normalization — copy the same idiom already
   in `decompositionmethod.jl`, don't reinvent it), then compute
   `M_int = max(solver.min_int_pts, round(Int, k*L0*solver.int_pts_scaling_factor/(2π)))`
   and `xy_int = random_interior_points(billiard, M_int)`. Populate
   `BoundaryPoints(...; xy_int = xy_int)`.
3. Replace `construct_matrices`: `B = basis_matrix(basis, k, pts.xy;
   multithreaded)`, `B_int = basis_matrix(basis, k, pts.xy_int;
   multithreaded)`, wrapped in `@blas_1` per the `-develop` version (confirm
   `@blas_1` exists in main's `utils/macros.jl` — it does).
4. Replace `solve_full`: `svdvals(B, B_int)` (generalized SVD), tension =
   `minimum(...)`. Reuses `LinearAlgebra.svdvals` directly — no new solver
   utility needed. Wrap in `@blas_multi_then_1 MAX_BLAS_THREADS` exactly like
   `-develop`; if `MAX_BLAS_THREADS` isn't already a constant in main, check
   `verginisaraceno.jl`/`acceleratedmethods.jl` first (it's a existing
   thread-budget constant used elsewhere) rather than inventing a new one.
5. Replace `solve_with_rank_reduction`: port the rank-revealing-QR + BLAS
   `syrk!`/`_symmetrize_from_upper!` + `eigmin` body verbatim (this is exactly
   the kind of low-level numeric routine that must be copied as-is per the
   fidelity rule in [00-index.md](00-index.md)) — confirm
   `_symmetrize_from_upper!` already exists in main (search
   `matrixconstructors.jl`/`decompositions.jl`; `VerginiSaracenoSolver` likely
   already uses it) and reuse it rather than re-defining.
6. Replace `solve`/`solve_vect`: `solve` dispatches to
   `solve_with_rank_reduction` by default (confirm the `-develop` default via
   reading the rest of `particular_solutions_method.jl` past what was
   previewed, specifically the `solve`/`solve_vect` bodies) . `solve_vect`
   additionally needs to recover the eigenvector `x` in the *original* basis
   coefficients (undo the QR pivoting/triangular-solve change of basis from
   `solve_with_rank_reduction` if rank reduction was used) — read the
   `-develop` `solve_vect` body carefully for this back-substitution step
   before porting; it's easy to introduce a subtle bug returning a vector in
   the wrong (reduced) basis.
7. Do **not** implement `compute_eigenstate` for PSM here — it is already free
   via the shared `SweepBasisSolver` generic in `states/eigenstates.jl` as
   long as `solve_vect` returns `(t, x)` with `x` in the original basis
   coefficient ordering that `BasisEigenstate` expects.

### Performance & fidelity notes

* `construct_matrices`/`solve_full`/`solve_with_rank_reduction` are exactly
  the kind of hot path this ecosystem cares about — no `push!`, no
  re-evaluating the basis twice, reuse `basis_matrix` (never hand-roll basis
  evaluation).
* Keep `@blas_1`/`@blas_multi_then_1` wrapping exactly where `-develop` has
  it — these bound BLAS threading around the small dense
  QR/SVD/`eigmin` calls to avoid oversubscription with the outer
  `multithreaded` basis-evaluation loop, per the mode's `@blas_multi`
  guidance.

### Tests & user verification

1. `get_errors` on `particularsolutions.jl` and `QuantumBilliards.jl`.
2. Manual smoke test: `ParticularSolutionsMethod(2.0, 5.0, 5.0)` +
   `CornerAdaptedFourierBessel`/Veech triangle fixture (same
   `make_veech_right_triangle_and_basis` helper `test/solvertests.jl` already
   uses) — call `solve_wavenumber(solver, basis, billiard, k0, dk)` around a
   `k0` where `VerginiSaracenoSolver`/`DecompositionMethodSolver` already find
   an eigenvalue in `test/solvertests.jl`, and confirm PSM's `k0` agrees to a
   few `atol=1e-2` (PSM is a different method, exact tension values will
   differ, but the located eigenvalue must match the two solvers already
   tested there).
3. Confirm `compute_eigenstate(solver, basis, billiard, k)` and
   `compute_psi(state, x_grid, y_grid)` run without error and produce a
   wavefunction that visually matches (or numerically correlates with) the
   corresponding `VerginiSaracenoSolver` eigenstate at the same `k` — this
   validates `solve_vect`'s basis-ordering fidelity end-to-end.
4. Ask the user to run `QBPlotting.plot_wavefunction!`/`plot_sweep!` on the
   PSM solver against Stadium/Triangle for a final visual sanity check — no
   `QBPlotting.jl` code changes are needed for this step since PSM is a plain
   `SweepBasisSolver` and every plotting function already dispatches on
   `QuantumBilliards.AbsBasisSolver`/`AbsState`, not on the concrete solver
   type.
5. Tell the user to invoke the **Julia Test Writer** subagent to add
   `ParticularSolutionsMethod` regression testsets to
   `test/solvertests.jl`, mirroring the existing Veech-triangle/Stadium
   testset style (ground state + low + high spectrum, recorded
   `k`/tension/`Psi` reference values from a verified run).

## Definition of done

* `rellich_origin` verified behaviorally unchanged at its default.
* `ParticularSolutionsMethod` fully implemented (no `error(...)` bodies left),
  loads cleanly, and its located eigenvalues agree with
  `VerginiSaracenoSolver`/`DecompositionMethodSolver` on the shared Veech
  triangle/Stadium fixtures.
* User has visually/numerically verified at least one PSM eigenstate.
* Julia Test Writer invoked (by the user) for regression coverage.
