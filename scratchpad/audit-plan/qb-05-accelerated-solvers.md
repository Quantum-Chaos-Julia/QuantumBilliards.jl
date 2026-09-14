# Step qb-05: Accelerated solvers (Vergini–Saraceno, Beyn, Expanded BIM)

## Goal
Audit the accelerated-method solvers and their shared wrapper
infrastructure.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/acceleratedmethods.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/verginisaraceno.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/chebyshevconfig.jl`

## Required background reading
- **Load the `julia-add-solver` skill first.**
- The **`qb-01`, `qb-03`, and `qb-04` findings files**.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/08-beyn-solver.md`,
  `09-expanded-bim-solver.md`, `09.5-compute-spectrum.md` — original design
  plans; `09.5` in particular introduced `ChebyshevConfig` and four
  `compute_spectrum` methods split across
  `AcceleratedBasisSolver`/`AcceleratedBIMSolver` branches — confirm all
  four still exist and dispatch correctly.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/12.5-spectral-api-consolidation.md`
  — rescoped `compute_spectrum` to `AcceleratedBasisSolver` (fixed a
  "false dispatch bug — no `SweepBasisSolver` `solve_spectrum` exists"),
  deleted a dead struct-field-access method, and added `_finalize_spectrum`
  as a shared tail helper. Confirm `_finalize_spectrum` is actually used by
  all applicable methods here (not reimplemented ad hoc in `beyn.jl`/
  `ebim.jl`).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/10.5-chebyshev-gap-analysis.md`
  — flags `BeynSolver`'s multi-`k` assembly doing `nq` separate `O(N²)`
  passes instead of one combined pass using already-existing-but-unused
  combinator functions in `bessels.jl` as "the largest performance
  regression", still unimplemented as of that plan. Confirm current status
  (fixed, or still open — if still open this is a real "missing
  implementation"/performance finding, not just historical).

## Audit checklist
1. **Dead code** — any helper in these five files with no call site.
2. **Missing wiring** — do `BeynSolver`/`ExpandedBIMSolver` both get
   `evaluate_points` "for free" via `acceleratedmethods.jl`'s delegation
   to `solver.kernel` as the migration plan claims? Is anything here
   exported that should be, or vice versa?
3. **Missing implementations** — per `julia-add-solver`'s checklist for
   `AbsSolver`/accelerated branches; the Step 10.5 Beyn multi-pass
   performance gap above; any `ChebyshevConfig` field read by these files
   but never actually set by a constructor (or vice versa).
4. **Unstable APIs** — compare `BeynSolver`/`ExpandedBIMSolver` constructor
   signatures (contour/window parameters, `cheb_config` keyword) for
   consistency; confirm `compute_spectrum`'s four dispatch variants share a
   consistent return type (`SpectralData` per the 12.5 plan) with no
   leftover variant returning a bare tuple/array.
5. **Vulnerabilities** — contour-integral moment assembly in `beyn.jl`:
   unchecked assumption about contour placement relative to true
   eigenvalues (silently missing eigenvalues near the contour boundary
   instead of warning); `overlap_and_merge_ebim!`'s clustering merge —
   any silent drop of a real eigenvalue pair that's actually
   far enough apart to be distinct.
6. **Consolidation opportunities** — shared contour/window-planning logic
   between `beyn.jl` and `ebim.jl` that could move into
   `acceleratedmethods.jl`.
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`.
8. **Threading/BLAS** — per the mode's conventions, especially around the
   per-window/per-quadrature-node parallel moment-matrix assembly in
   `beyn.jl` (a classic place to accidentally nest threading if the
   inner kernel evaluation is itself parallelized).

## Out of scope for this step
- Non-accelerated sweep solvers → `qb-04`.
- Chebyshev evaluation backend internals → `qb-06`.
- `spectra/spectralutils.jl`'s `SpectralData`/`overlap_and_merge!` shared
  definitions (only check that these solvers *call* them correctly) →
  `qb-07`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Tell it to
load the `julia-add-solver` skill first, then paste this file's
Scope/Required background/Checklist/Out-of-scope sections into the prompt.
State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-05-accelerated-solvers-findings.md`.
