# Step qb-09: Wavefunctions, boundary functions & Husimi functions

## Goal
Audit the physical-observable reconstruction layer: wavefunction
evaluation, boundary (normal-derivative) functions, and Husimi
(phase-space) functions, for both basis-solver and BIM-solver eigenstates.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/boundaryfunctions.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/husimifunctions.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/gradients.jl` (if
  present — confirm during the audit whether this file still exists /
  is still `include`d; the module-file listing in `qb-01`'s findings is
  authoritative)
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/randomstates.jl` (same caveat)

## Required background reading
- The **`qb-08` findings file** (the state fields this layer reads).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/05-bim-wavefunction-husimi-green.md`
  — ports the single-layer-potential (`ϕ_slp`) Green's-function
  reconstruction; wires `wavefunction`/`boundary_function`/
  `husimi_function` for `BIMEigenstate`, initially scoped to
  `DoubleLayerPotentialSolver` only; also switches `BasisEigenstate`'s
  `wavefunction` to use the same Green's-function integral by default.
  Confirm scope has since widened correctly to other `SweepBIMSolver`s
  per step 5.5 (see `qb-08`), not left DLP-only by accident.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/05.6-general-nonuniform-husimi.md`
  — `husimi_function`'s fast sliding-window stencil is only valid for
  uniformly-spaced arclength grids; a general windowed
  (`searchsortedfirst`/`searchsortedlast`, physical-`ds`-weighted)
  quadrature core was ported as an automatic fallback for non-uniform
  grids (e.g. `GlobalCornerGrading`). **Confirm the fallback is actually
  triggered automatically** (not requiring a manual flag) whenever the
  grid is non-uniform, and that the previous `@warn`-with-no-fallback
  behavior is fully gone.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/10.5-chebyshev-gap-analysis.md`
  gap (4): the still-unported SLP/CFIE wavefunction Chebyshev
  reconstruction — cross-reference with `qb-06`'s findings; confirm
  whether `wavefunction.jl` here has any Chebyshev-aware code path at all
  yet.

## Audit checklist
1. **Dead code** — any evaluation branch (e.g. an old uniform-grid-only
   Husimi path) left in place after the general fallback was added, if it's
   now provably unreachable.
2. **Missing wiring** — are `wavefunction`/`compute_psi`/`boundary_limits`/
   `get_boundary_curves_with_ignored`/`boundary_function`/
   `momentum_function`/`husimi_function` all exported per `qb-01`'s
   findings?
3. **Missing implementations** — the automatic non-uniform-grid Husimi
   fallback (verify it's not a manual opt-in still); the DLP-only-vs-general
   `BIMEigenstate` wavefunction scope question above.
4. **Unstable APIs** — do `wavefunction`/`boundary_function`/
   `husimi_function` share one consistent dispatch signature style across
   `BasisEigenstate` vs. `BIMEigenstate` inputs, or does the caller need to
   know which state type it has to call the right variant?
5. **Vulnerabilities** — Green's-function reconstruction near-singular
   kernel evaluation close to the boundary (self-intersection distance
   handling); Husimi windowed quadrature — off-by-one in
   `searchsortedfirst`/`searchsortedlast` window bounds at the array edges
   (a classic place for a silent wraparound or truncated window bug).
6. **Consolidation opportunities** — shared quadrature/grid-spacing logic
   between `boundaryfunctions.jl` and `husimifunctions.jl` that could be
   one helper instead of two near-duplicates.
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`.
8. **Threading** — confirm any per-point wavefunction/Husimi grid
   evaluation loop follows "inner single-threaded, outer parallel" with
   preallocated output arrays (no `push!` inside the loop).

## Out of scope for this step
- `states/symmetry/*` (reflection-based boundary curve reconstruction for
  symmetric billiards, as opposed to raw evaluation) → `qb-10`.
- State field definitions themselves → `qb-08`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-09-wavefunctions-boundary-husimi-findings.md`.
