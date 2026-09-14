# Step 7 — `CompositeBIMSolver`

## Goal

Implement multiply-connected-geometry support in
[solvers/sweepmethods/compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl),
generalizing `-develop`'s CFIE-only `CFIE_kress_composite_solver` to wrap
*either* `DoubleLayerPotentialSolver` or `CombinedFieldIntegralEquationSolver`
components (or a mix), per the Step 1 plan §2.6/§4.5.

## Preconditions

Steps 3, 4 and 6 complete and verified (Step 4's symmetry/full-boundary fixes
apply to `CompositeBIMSolver` exactly as they do to its DLP/CFIE components).

## Source index

`QuantumBilliards-develop/src/solvers/sweepmethods/cfie/cfie_kress.jl` —
`CFIE_kress_composite_solver` struct + methods (search near the end of the
file, after the single-curve/global-corner variants). This is the *only*
`-develop` composite implementation; there is no DLP-composite equivalent to
compare against, so the DLP-component code path in this step is new (adapted
from the CFIE-composite pattern, not copied from an existing DLP-composite
source).

## Files touched in main

Only [solvers/sweepmethods/compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl).
The struct and `Vararg` constructor (validating shared `symmetry` across
components) are already scaffolded from Step 1 — only the method bodies need
filling in.

## Implementation steps

1. **`evaluate_points(solver::CompositeBIMSolver, billiard, k)`**: sample each
   connected boundary component with its own `component_solvers[i]`'s
   `evaluate_points` (reusing Steps 3 and 6's already-working per-component
   sampling unchanged), producing a `Vector{BoundaryPoints{T}}` — component 1
   is the outer boundary, components 2:end are holes with **reversed
   orientation** (outward normal of a hole points into the solid, i.e. away
   from the domain interior — confirm the sign convention against
   `-develop`'s composite solver rather than assuming, since getting the
   normal orientation backwards silently flips the sign of every off-diagonal
   block). Validate
   `length(component_solvers) == number of connected boundary components of
   billiard` here (lazily, since `billiard` isn't known at construction time
   per the Step 1 plan §2.1/§4.5) — raise `ArgumentError` on mismatch exactly
   as already documented on the struct.
2. **`construct_matrices(solver, pts::Vector{BoundaryPoints{T}}, k;
   multithreaded)`**: assemble one block per pair of components `(a, b)` —
   diagonal blocks `(a, a)` are each component's own single-solver kernel
   (`construct_matrices(component_solvers[a], pts[a], k)`, unchanged from
   Steps 3 and 6), off-diagonal blocks `(a, b)` are the *same* kernel type's
   free-space Green's-function coupling between component `a`'s and `b`'s
   nodes (no `kress_R!` singular splitting needed off-diagonal, since sources
   and observation points never coincide across different components — reuse
   `boundary_geom_cache`-style pairwise distances but computed between two
   different `BoundaryPoints` sets, not the self-pairwise case; this may need
   a small new two-argument overload of the pairwise-distance computation
   used inside `boundary_geom_cache`, kept private to this file, or exposed
   from Step 2's `boundarygeomcache.jl` if genuinely reusable — prefer
   reusing/extending Step 2's cache machinery over writing a third
   independent implementation of pairwise Hankel-kernel evaluation). Use
   `component_offsets(pts)` (Step 2) to place each block into the full dense
   matrix.
3. **`solve`/`solve_vect`**: identical pattern to Steps 3 and 6 — smallest
   singular value / nullspace residual of the assembled block matrix.
4. `_bim_numeric_type(::CompositeBIMSolver{T}) = T`.

## Performance & fidelity notes

* The off-diagonal block assembly is new code (no direct singular-splitting
  needed), but is still a hot path for multi-component domains — apply the
  same preallocate/`@inbounds`/single-outer-loop-threading discipline as
  Steps 3 and 6, not a relaxed standard just because it's "new" rather than
  ported.
* Reuse the diagonal-block kernels from Steps 3 and 6 unchanged (call through,
  don't duplicate their bodies).

## Tests & user verification

Per [00-index.md](00-index.md)'s flagged limitation: **no multiply-connected
billiard exists yet** (Triangle/Stadium are both simply connected). Do the
best available verification now, and schedule a follow-up after Step 11:

1. `get_errors` on `compositebim.jl` and `QuantumBilliards.jl`.
2. Structural smoke test: construct a `CompositeBIMSolver` from two
   `DoubleLayerPotentialSolver`s wrapping two *disjoint* copies of the
   Triangle/Stadium fixture placed far apart (e.g. one billiard translated
   far from the other, simulated by manually offsetting the sampled `xy`
   points post-`evaluate_points`, or by constructing two independent
   single-component solves and manually verifying the assembled composite
   matrix is block-diagonal to numerical precision at large separation, since
   the off-diagonal coupling should vanish as separation → ∞). Confirm the
   composite solver's found eigenvalues at large separation match the union
   of the two independent single-component solves' eigenvalues — this is a
   legitimate correctness check on the off-diagonal-block coupling and
   symmetry-validation logic without needing a real hole.
3. Confirm `evaluate_points` produces the right total point count
   (`boundary_matrix_size(pts) == sum of each component's own count`) and
   correct per-component orientation sign (spot-check one hole-oriented
   normal by hand against the corresponding outer-boundary-oriented normal at
   a symmetric point, if the synthetic two-copy test above is set up with a
   deliberately reversed second copy).
4. `QBPlotting.jl`: `plot_boundary!`
   ([wavefunctionplotting.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/wavefunctionplotting.jl))
   currently plots a single `AbsBilliard`'s boundary — confirm it already
   handles multiply-connected billiards' boundary curve lists (check
   `get_all_curves`/`get_boundary_curves` behavior for composite domains in
   `BilliardGeometry.jl`); if not, this is tracked for Step 11 (once a real
   hole-billiard exists) rather than fixed blindly here.
5. Tell the user to invoke the **Julia Test Writer** subagent now for the
   two-disjoint-copies structural regression test, and add a reminder
   (tracked in Step 11's task list) to invoke it again for full
   multiply-connected numerical regression once a hole-billiard lands.

## Step 5.5 follow-up (required)

`_bim_grid_scale(solver::AbsBIMSolver) = solver.pts_scaling_factor[1]`
(`sweepmethods.jl`, Step 5.5) is used by `wavefunction(state::BIMEigenstate;
b=:auto)`'s default grid density, but `CompositeBIMSolver` has no
`pts_scaling_factor` field of its own. Add
`_bim_grid_scale(solver::CompositeBIMSolver) =
solver.component_solvers[1].pts_scaling_factor[1]` (or another principled
choice across component solvers) in this file as part of this step, otherwise
`wavefunction(state; b=:auto)` errors for a `CompositeBIMSolver`-derived
`BIMEigenstate`. Everything else — `solve_state`/`_bim_normal_derivative`/
`symmetrize_layer_density`/`boundary_function`/`momentum_function`/
`husimi_function` — is already generic and needs no new code here, provided
`CompositeBIMSolver`'s assembled matrix still follows the plain
`W = diag(ds)` Nyström reciprocity (true for both its `DoubleLayerPotentialSolver`
and `CombinedFieldIntegralEquationSolver` component kernels).


## Definition of done

* `CompositeBIMSolver` fully implemented for both DLP and CFIE component
  kernels.
* Two-disjoint-copies structural verification passes (composite eigenvalues
  match the union of independent single-component solves).
* Follow-up task recorded for full validation after Step 11.
* Julia Test Writer invoked for the structural regression test.
