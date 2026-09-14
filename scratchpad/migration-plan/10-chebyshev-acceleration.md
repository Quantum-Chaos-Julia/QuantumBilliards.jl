# Step 10 — Chebyshev-accelerated Hankel/Bessel evaluation

## Goal

Implement the opt-in performance backend behind the already-scaffolded
`use_chebyshev::Bool` fields (and each solver's `cheb_config::ChebyshevConfig`,
see [09.5-compute-spectrum.md](09.5-compute-spectrum.md)) on
`BeynSolver`/`ExpandedBIMSolver`, replacing the direct
`error("... not yet implemented ...")` guards added in Steps 8–9 with real
Chebyshev-interpolated Hankel/Bessel-J evaluation. Purely additive — every
solver from Steps 3, 6–9 and 9.5 must already be fully correct with
`use_chebyshev=false` before this step starts.

## Preconditions

Steps 3, 6–9 and 9.5 complete and verified. In particular, Step 9.5 already
gives every accelerated solver a `cheb_config::ChebyshevConfig` field (in
place of the four separate `n_panels_h`/`M_h`/`n_panels_j`/`M_j` fields
`-develop` never consolidated) and a wide-range `compute_spectrum` driver
(`compute_spectrum(::BeynSolver,...)`/`compute_spectrum(::ExpandedBIMSolver,...)`)
that this step must make actually respect `use_chebyshev=true` end to end,
not just at the single-window `solve`/`construct_matrices` level.

## Source index

`QuantumBilliards-develop/src/chebyshev/` (all six files — read each in full,
they were only listed, not read, during planning):

* `chebyshev_core.jl` — the generic Chebyshev-interpolation machinery
  (panelization, coefficient fitting, evaluation) not specific to any one
  special function; this is the piece every other file in this folder builds
  on.
* `chebyshev_bessels.jl` — Chebyshev interpolants for the raw Hankel/Bessel-J
  special functions themselves (and presumably their derivatives, needed by
  Step 9's EBIM path).
* `chebyshev_optimal_panelization.jl` — logic for choosing panel
  boundaries/counts (`n_panels_h`/`M_h`/`n_panels_j`/`M_j`, now the
  `n_panels_h`/`M_h`/`n_panels_j`/`M_j` fields of a `ChebyshevConfig`, see
  Step 9.5) to hit a target accuracy; also owns the `cheb_tol`/`max_iter`/
  `sampling_points`/`grow_panels`/`grow_M` auto-tuning knobs, which are
  likewise already `ChebyshevConfig` fields (`tol`/`max_iter`/
  `sampling_points`/`grow_panels`/`grow_M`) waiting for this step's
  `chebyshev_params`-equivalent auto-tuner to read/update them.
* `chebyshev_dlp.jl` — DLP-specific Chebyshev workspace (the direct,
  non-Kress kernel — since the plain `BoundaryIntegralMethod` was dropped,
  confirm whether this file's logic is still needed standalone or only via
  `chebyshev_dlp_kress.jl`'s Kress-aware version; if the latter fully
  supersedes it for every case main actually ports, this file may not need a
  separate port — decide based on what `build_derivative_chebyshev_workspace`
  actually dispatches to for `DoubleLayerPotentialSolver`).
* `chebyshev_dlp_kress.jl` — Kress-aware DLP Chebyshev workspace
  (`build_derivative_chebyshev_workspace`/`construct_matrices_chebyshev_with_derivatives!`
  equivalents for `DLP_kress`/`DLP_kress_global_corners`).
* `chebyshev_cfie_kress.jl` — same for the CFIE-Kress family.

## Files touched in main

New directory `QuantumBilliards.jl/src/solvers/chebyshev/` mirroring the
`-develop` file layout (`core.jl`, `bessels.jl`, `optimalpanelization.jl`,
`dlp.jl`, `cfie.jl` — drop the redundant `chebyshev_` prefix since the
directory name already scopes it, per main's existing naming convention of
not repeating the folder name in the filename). Included from
`QuantumBilliards.jl`'s module file right after
`solvers/acceleratedmethods/chebyshevconfig.jl` (see
[09.5-compute-spectrum.md](09.5-compute-spectrum.md); do **not** reintroduce
separate `n_panels_h`/`M_h`/`n_panels_j`/`M_j` fields anywhere in this new
directory's structs — thread `cfg::ChebyshevConfig` through instead). Then
edit
[solvers/acceleratedmethods/beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl),
[solvers/acceleratedmethods/ebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl)
and
[spectra/spectralutils.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/spectralutils.jl)
to remove the `use_chebyshev`-guard `error(...)` and dispatch to the new
Chebyshev path when `solver.use_chebyshev == true`, at both the
single-window (`solve`/`construct_matrices`) and wide-range
(`compute_spectrum`) levels.

## Implementation steps

1. Port `chebyshev_core.jl` verbatim first — every other file depends on it.
2. Port `chebyshev_bessels.jl` and `chebyshev_optimal_panelization.jl`,
   verbatim, in either order (independent of each other, both depend on core).
   `chebyshev_optimal_panelization.jl`'s auto-tuning entry point
   (`-develop`'s `chebyshev_params`) should be adapted to take and return a
   [`ChebyshevConfig`](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/chebyshevconfig.jl)
   (reading its `tol`/`max_iter`/`sampling_points`/`grow_panels`/`grow_M`
   fields as the tuning inputs and its `n_panels_h`/`M_h`/`n_panels_j`/`M_j`
   as the starting point, returning an updated `ChebyshevConfig` with the
   tuned panel/degree fields) instead of `-develop`'s bare
   `(n_panels_h,M_h,n_panels_j,M_j)` tuple return — this is the one place
   `ChebyshevConfig` actually needs to be *produced*, not just carried
   around.
3. Port `chebyshev_dlp_kress.jl`/`chebyshev_cfie_kress.jl` (and `chebyshev_dlp.jl`
   if step-8.source-index's open question above resolves to "yes, still
   needed"), adapting only the kernel/struct names to
   `DoubleLayerPotentialSolver`/`CombinedFieldIntegralEquationSolver` (the
   Step 1 plan's unified two-family types replacing `-develop`'s many
   `DLP_kress*`/`CFIE_kress*` variants) — the workspace-construction logic
   itself (panel layout, coefficient caching, batched evaluation across many
   `k`) is the performance-critical numeric core and must be copied verbatim
   per the fidelity rule.
4. Wire `BeynSolver`/`ExpandedBIMSolver`'s `construct_matrices` to branch on
   `solver.use_chebyshev`: `true` calls into the new Chebyshev workspace path
   (build/reuse a cache across the many `k` values a contour/derivative
   evaluation needs, reading panel/degree counts from `solver.cheb_config`),
   `false` keeps calling the direct `Bessels.hankelh1`/`besselj` evaluation
   from Steps 3, 6 and 9 unchanged.
5. Because a contour solve (`BeynSolver`) or a derivative evaluation
   (`ExpandedBIMSolver`) evaluates the *same* kernel at many `k` values, the
   Chebyshev cache should be built once per `solve`/`solve_wavenumber` call
   (not once per individual `k` inside the loop) — confirm the cache-build
   call site matches `-develop`'s `build_ebim_cheb_cache`/analogous
   Beyn-side cache builder placement (outside the per-node loop, not inside
   it) to actually realize the intended speedup.
6. Wire the two `compute_spectrum` methods from
   [09.5-compute-spectrum.md](09.5-compute-spectrum.md) to actually use
   Chebyshev acceleration end to end, not just at the single-window level:
   * `compute_spectrum(solver::BeynSolver, ...)` currently builds a fresh
     boundary discretization per Weyl window and calls `solve` once per
     window; with `solver.use_chebyshev=true`, auto-tune (or reuse a
     manually-supplied) `ChebyshevConfig` **once per window** (matching
     `-develop`'s `solve_spectrum_beyn`'s `do_INFO_init`/representative-disk
     tuning, simplified since main has no `do_INFO_init`/`Progress`
     machinery to preserve) rather than once per contour node, mirroring
     implementation step 5 above at the window-sweep level.
   * `compute_spectrum(solver::ExpandedBIMSolver, ...)` should honor
     `solver.cheb_config.param_strategy`: `:global` tunes one
     `ChebyshevConfig` from the largest `k` in `[k1,k2]` and reuses it for
     every segment (matching `-develop`'s default); `:segment` re-tunes once
     per `seg_reuse_frac`-defined segment (already iterated over in the
     Step 9.5 loop — add the re-tune call at the same `seg_last==seg_first ||
     ...` branch that already rebuilds `pts`); `:manual` never re-tunes,
     using `solver.cheb_config` exactly as constructed. Reject any other
     `param_strategy` value the same way `ChebyshevConfig`'s constructor
     already validates it.
7. Investigate the pre-existing `ExpandedBIMSolver` + plain
   `DoubleLayerPotentialSolver` (no `ikS(k)` term) `NaN` result flagged by
   Step 9.5's "Performance & fidelity notes" (`solve(solver::ExpandedBIMSolver,
   pts, k)` returns `NaN` for every trial `k` tested on the Veech triangle
   with `GlobalCornerGrading`, including `k` essentially equal to a
   Beyn-verified true eigenvalue) — this step touches exactly the same
   kernel-derivative code (`_dlp_kernel_entry_with_derivatives`,
   `_ebim_lin1_deriv`) the Chebyshev path must also interpolate, so it is a
   natural place to also determine whether the direct-evaluation path's
   `eigen(A,dA)`/`eigen(A',dA')` primal/adjoint eigenvector pairing (or
   `A'(k)`'s conditioning for a pure-DLP kernel) is the root cause, before
   assuming the Chebyshev-accelerated derivative kernels will behave any
   better.

## Performance & fidelity notes

This entire step is performance infrastructure — every file here is exactly
the kind of "low-level implementation the algorithm requires" the mode
instructions call out as off-limits for simplification. Port verbatim,
including panel/coefficient buffer reuse patterns; do not consolidate
DLP/CFIE Chebyshev workspace buffer types unless `-develop` already shares
them (check before merging — if `-develop` keeps them separate because their
underlying kernels have genuinely different derivative structure, keep them
separate in main too).

## Tests & user verification

1. `get_errors` on every new file plus `beyn.jl`/`ebim.jl`/`spectralutils.jl`/`QuantumBilliards.jl`.
2. Accuracy check: for a fixed `BeynSolver`/`ExpandedBIMSolver` call, compare
   `use_chebyshev=true` vs. `use_chebyshev=false` results (recovered
   eigenvalues/root corrections) — they must agree to within the Chebyshev
   interpolation's target accuracy (a few `atol` orders tighter than
   `res_tol`/`svd_tol`, not just "roughly the same").
3. Performance check: time a representative `BeynSolver` contour solve with
   `use_chebyshev=true` vs. `false` on a moderately large boundary
   discretization (e.g. `k≈500`+ on Stadium) and confirm a genuine speedup —
   if there's no measurable speedup, something in the caching/batching wiring
   from implementation step 5 above is likely wrong (cache rebuilt too often).
4. Repeat checks 2–3 at the wide-range `compute_spectrum` level (implementation
   step 6): a `compute_spectrum(::BeynSolver, billiard, k1, k2)` sweep with
   `use_chebyshev=true` should recover the same `SpectralData.k`/`.ten` as
   `use_chebyshev=false` (see [09.5-compute-spectrum.md](09.5-compute-spectrum.md)'s
   Veech-triangle/symmetric-circle fixtures) with a genuine wall-clock
   speedup; for `ExpandedBIMSolver`, check all three `cheb_param_strategy`
   values (`:global`/`:segment`/`:manual`) produce equivalent merged spectra.
5. Resolve or explicitly document the finding of implementation step 7 (the
   pure-DLP `ExpandedBIMSolver` `NaN` issue) — either fixed alongside the
   derivative-kernel work this step already requires, or, if out of scope,
   recorded with a clear reproduction case for a dedicated follow-up.
6. `QBPlotting.jl`: no changes needed — this step is invisible to plotting,
   only affects internal performance of already-plotted solvers.
7. Tell the user to invoke the **Julia Test Writer** subagent for Chebyshev
   accuracy-vs-direct-evaluation regression tests (both single-window and
   wide-range `compute_spectrum`) once the above checks pass.

## Definition of done

* Chebyshev Hankel/Bessel-J evaluation pathway fully ported and wired into
  `BeynSolver`/`ExpandedBIMSolver`, reading panel/degree/auto-tuning
  parameters exclusively from `solver.cheb_config::ChebyshevConfig`.
* `use_chebyshev=true` results agree with `use_chebyshev=false` to target
  accuracy, with a measured speedup, at both the single-window and
  `compute_spectrum` wide-range levels.
* The pure-DLP `ExpandedBIMSolver` `NaN` issue flagged by Step 9.5 is
  resolved or explicitly documented for follow-up.
* Julia Test Writer invoked.
