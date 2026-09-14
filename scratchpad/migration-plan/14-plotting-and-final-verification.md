# Step 14 — Plotting & final integration pass

## Goal

Final cross-cutting sweep once Steps 1–13 are individually complete: dedicated
`QBPlotting.jl` support for `BIMEigenstate` (wavefunction/boundary-function
plots via the layer-potential representation formula), and a whole-ecosystem
smoke test across every package touched by this migration. Each earlier step
already lands its own narrow plotting update (`plot_sweep!` overloads for
`SweepBIMSolver`/`AcceleratedBIMSolver` in Steps 3 and 8, plus the
`BIMEigenstate` wavefunction/boundary-function/Husimi plots already wired for
`DoubleLayerPotentialSolver` in Step 5) — this step is only what's left over
once every solver exists.

## Preconditions

Steps 1–13 complete.

## Files touched

* [QBPlotting.jl/src/wavefunctionplotting.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/wavefunctionplotting.jl)
* [QBPlotting.jl/src/boundaryfunctionsplotting.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/boundaryfunctionsplotting.jl)
* [QBPlotting.jl/src/husimiplotting.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/husimiplotting.jl)
* [QBPlotting.jl/src/testplots.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/testplots.jl)

## Implementation steps

1. **`wavefunction`/`compute_psi` for `BIMEigenstate`**: implement the
   Helmholtz representation formula evaluating the wavefunction at an
   interior point `z` from the boundary density `state.vec` stored on
   `BIMEigenstate` (double-layer or combined-field representation formula,
   matching whichever kernel `state.solver` used — dispatch on the kernel
   type stored in the state, mirroring how `-develop` recovers a wavefunction
   from a converged Nyström density, if `-develop` has this at all; search
   `QuantumBilliards-develop/src/states/`-equivalent location, or
   `solvers/`-adjacent, for any such function before assuming it needs to be
   written from scratch — the representation formula itself is standard
   boundary-integral-equation theory (single/double-layer potential
   evaluated off the boundary) even if `-develop` never implemented a
   convenience wrapper for it). This unblocks `plot_wavefunction!`/
   `plot_probability!` for BIM states, currently generic over `AbsState` but
   silently broken for `BIMEigenstate` since `wavefunction` has no method for
   it yet.
2. **`boundary_function`/`momentum_function` for `BIMEigenstate`**: for a BIM
   state the boundary function *is* the stored density (up to a
   normalization convention) rather than something derived from a basis
   expansion — implement the thin adapter converting `state.vec`/`state.pts`
   into the same `(s, u)`-style return shape
   `plot_boundary_function!`/`plot_momentum_function!` already expect from
   `AbsState`.
3. Add `plot_wavefunction!`/`plot_probability!`/`plot_boundary_function!`/
   `plot_momentum_function!`/`plot_husimi_function!` methods if any of them
   need a `BIMEigenstate`-specific dispatch beyond what the generic
   `AbsState` methods already provide (check each file — most already
   dispatch on `state::QuantumBilliards.AbsState`, so once `wavefunction`/
   `boundary_function`/`husimi_function` have `BIMEigenstate` methods per
   steps 1–2 above, these plotting functions likely need **no** changes at
   all; only add a bespoke method if a `BIMEigenstate`-specific rendering
   detail is genuinely needed, e.g. overlaying the boundary density directly
   alongside the derived wavefunction).
4. Update [testplots.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/testplots.jl)'s
   `plot_state_tests!` to also accept `SweepBIMSolver`/`AcceleratedBIMSolver`
   solvers (currently only exercised against basis solvers) — a thin
   dispatch addition, not a rewrite, following the same
   solver/basis/billiard argument convention already there (BIM solvers omit
   the `basis` argument, matching every other BIM-specific signature already
   established in this migration).
5. **Whole-ecosystem smoke test**: with every step done, run a short script
   exercising, in one session: every solver type from Steps 1–7 (basis and
   BIM, sweep and accelerated) on Triangle, Stadium, and at least one Step 11
   billiard (`circle.jl` plus `AnnularBilliard`, per Steps 12–13's
   `MultiplyConnectedDomain` refactor and `CompositeBIMSolver` verification),
   `compute_eigenstate`, `compute_psi`, and every `QBPlotting.jl` plotting
   function, confirming nothing across the five packages
   (`QuantumBilliards.jl`, `BilliardGeometry.jl`, `QCPlotting.jl`,
   `QBPlotting.jl`, `SpectralStatistics.jl` if spectral-statistics functions
   are exercised on the newly available spectra) throws or silently produces
   `NaN`/`Inf`.
6. Run `get_errors` across the whole workspace (all five package folders), not
   just the files touched in this step.

## Performance & fidelity notes

Wavefunction evaluation from a boundary density (implementation step 1) is
itself potentially a hot path if plotted on a fine grid — apply the same
preallocate/`@use_threads`-over-grid-points/`@inbounds` discipline the
existing basis-solver `wavefunction`/`compute_psi` implementation already
uses (read `states/wavefunctions.jl`'s existing `compute_psi` for the pattern
to mirror, since this is new code, not a `-develop` port, for the BIM case).

## Tests & user verification

1. `get_errors` across the whole workspace.
2. Visual verification (user-facing, ask the user to look at the plots): a
   `DoubleLayerPotentialSolver` eigenstate's wavefunction plot should visually
   resemble the corresponding `VerginiSaracenoSolver` eigenstate at the same
   `k` (same nodal pattern) on Stadium/Triangle, and — once Step 11's
   `circle.jl` lands — match the known analytic Bessel-mode pattern on the
   circle.
3. Confirm `plot_husimi_function!`/`plot_boundary_function!`/
   `plot_momentum_function!` all run without error for a `BIMEigenstate`.
4. Whole-ecosystem smoke script (implementation step 5) runs clean end to end.
5. Tell the user to invoke the **Julia Test Writer** subagent one final time
   for `BIMEigenstate` wavefunction/boundary-function regression tests
   (comparing against the matching basis-solver eigenstate's wavefunction at
   the same `k`, and against the analytic circle spectrum from Step 11), and
   to review `QBPlotting.jl/test/`/`QCPlotting.jl/test/` coverage for the new
   plotting methods.

## Definition of done

* `BIMEigenstate` fully supports `wavefunction`/`compute_psi`/
  `boundary_function`/`momentum_function`/`husimi_function`.
* Every `QBPlotting.jl` plotting function works for both basis and BIM
  states/solvers.
* Whole-ecosystem smoke test passes with no errors across all five packages.
* Julia Test Writer invoked for final regression coverage.
* This migration plan's every step (1–13) is marked done and cross-validated.
