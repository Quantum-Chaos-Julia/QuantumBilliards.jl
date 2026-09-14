# Step 6 — `CombinedFieldIntegralEquationSolver`

## Goal

Implement `A(k) = I - (D(k) + ikS(k))` in
[solvers/sweepmethods/cfie.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/cfie.jl),
reusing everything Step 3 already built (`BoundaryGeomCache`, `kress_R!`,
`symmetry_index_orbits`, the `BoundaryGrading` dispatch pattern, `BIMEigenstate`/
`compute_eigenstate(::SweepBIMSolver, ...)`). This step is intentionally the
"copy Step 3's shape, change the kernel" step — do not redesign anything.

## Preconditions

Step 3 complete and verified (`DoubleLayerPotentialSolver` working for both
gradings, `BIMEigenstate`/`compute_eigenstate` wired).

## Source index

* `QuantumBilliards-develop/src/solvers/sweepmethods/cfie/cfie_kress.jl` — full
  file: `CFIE_kress`/`CFIE_kress_corners`/`CFIE_kress_global_corners`/
  `CFIE_kress_composite_solver` structs, constructors, `construct_matrices`
  (the `D(k) + ikS(k)` assembly — reuses the same `BoundaryGeomCache` pairwise
  data as DLP plus an additional single-layer Hankel-kernel term), `solve`/
  `solve_vect`. Read this file in full; it wasn't read during planning beyond
  the reference-list header.

## Files touched in main

Only [solvers/sweepmethods/cfie.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/cfie.jl).
`CFIE_kress_composite_solver` is **not** ported here — it becomes Step 7's
`CompositeBIMSolver`, generalized to wrap DLP too (per Step 1 plan §2.6/§4.5).

## Implementation steps

1. **`evaluate_points`**: identical structure to Step 3's, but must also
   support the single-curve `CornerGrading` case (`CFIE_kress_corners`
   equivalent) in addition to `SmoothPeriodicGrading`/`GlobalCornerGrading`.
   `CornerGrading` grades a single curve's own parametric corner (e.g. a
   polygon edge with a corner baked into its own `[0,1)` parametrization at
   `t=0`) rather than detecting corners between composite segments — read
   `-develop`'s `CFIE_kress_corners` `evaluate_points` (search
   `cfie_kress.jl`, likely nearby the `CFIE_kress` struct or in a section
   dedicated to single-curve grading) to get the exact grading-map difference
   right versus `GlobalCornerGrading`'s composite-junction detection from
   Step 2/3.
2. **`construct_matrices`**: build `D(k)` exactly as Step 3's DLP does (same
   `BoundaryGeomCache`, same Hankel kernel, same `kress_R!` split), then add
   the single-layer term `S(k)` (Hankel-function kernel without the
   tangential/normal derivative factor DLP's kernel has) scaled by the fixed
   coupling `ik` (confirmed in the Step 1 plan §4.4 as *not* a tunable field —
   do not add a coupling-constant field to the struct). Assemble
   `A = I - (D(k) + im*k*S(k))`. Reuse a shared internal helper for the
   `D(k)` piece if practical (e.g. factor DLP's kernel assembly into a
   private function `_dlp_kernel!(out, G, k, Rmat)` in `dlp.jl` and call it
   from both `DoubleLayerPotentialSolver` and
   `CombinedFieldIntegralEquationSolver`'s `construct_matrices`) — this is
   exactly the kind of cross-solver duplication the mode instructions ask to
   streamline via a shared function, as long as the shared function is itself
   allocation-free/in-place and doesn't sacrifice the single hot loop's
   performance (pass preallocated buffers in, don't allocate inside the
   shared helper).
3. **`solve`/`solve_vect`**: identical pattern to Step 3 (smallest singular
   value / nullspace residual of `A(k)`, `use_krylov` following whatever
   Step 3 landed on).
4. `_bim_numeric_type(::CombinedFieldIntegralEquationSolver{T}) = T`.
5. No new `compute_eigenstate`/`BIMEigenstate` work needed — already generic
   over `SweepBIMSolver` from Step 3, extended by Step 5.5 to also populate
   `state.pts`/`state.u`/`state.bnd_norm` (and therefore
   `boundary_function`/`momentum_function`/`wavefunction`/`husimi_function`)
   automatically via the generic `solve_state`. **One check required**:
   `_bim_normal_derivative`'s default assumes the assembled `A(k) =
   I-(D(k)+ikS(k))` is still a plain Nyström discretization with diagonal
   quadrature weight `W = diag(ds)` (true for both the `D(k)` and `S(k)`
   terms individually, same reciprocity argument as `DoubleLayerPotentialSolver`)
   — confirm this once `construct_matrices` is implemented; if it doesn't
   hold, add an override
   `_bim_normal_derivative(solver::CombinedFieldIntegralEquationSolver, pts, lvec)`
   in this file instead of re-introducing a separate adjoint-matrix assembly.

## Performance & fidelity notes

Same as Step 3 — `construct_matrices` is hot, port kernel algebra verbatim.
The only new numerical piece versus Step 3 is the single-layer `S(k)` term and
the `ik` scaling; get the sign/factor convention exactly right by comparing
against `-develop`'s literal `A(k)=I-(D(k)+ikS(k))` formula already
documented in `cfie.jl`'s docstring.

## Tests & user verification

1. `get_errors` on `cfie.jl` and `QuantumBilliards.jl`.
2. Same cross-validation as Step 3: `k_sweep`/`solve_wavenumber` on
   Stadium/Triangle, located `k0` compared against
   `VerginiSaracenoSolver`/`DecompositionMethodSolver`/`DoubleLayerPotentialSolver`
   (all four should now agree on the same physical eigenvalues from four
   independent formulations — a strong end-to-end correctness signal).
3. Confirm CFIE with `use_krylov`/dense path both converge to consistent
   `k0`/tension at a representative wavenumber (regression against DLP's
   already-verified numbers, not a new analytic reference).
4. `QBPlotting.jl`: no additional change needed beyond Step 3's
   `plot_sweep!(::SweepBIMSolver, ...)` overload — it already dispatches on
   the abstract branch, so it works for CFIE for free. Ask the user to
   visually confirm a CFIE tension-vs-`k` plot on Stadium.
5. Tell the user to invoke the **Julia Test Writer** subagent for
   `CombinedFieldIntegralEquationSolver` regression tests across all three
   gradings.

## Definition of done

* `CombinedFieldIntegralEquationSolver` fully implemented for
  `SmoothPeriodicGrading`, `CornerGrading`, `GlobalCornerGrading`.
* Located eigenvalues agree with `DoubleLayerPotentialSolver`/basis solvers on
  shared fixtures.
* Julia Test Writer invoked.
