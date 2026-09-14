# Step qb-07: Spectra (spectral utilities & unfolding)

## Goal
Audit the spectral-statistics/state-counting utilities: `SpectralData`
construction/merging and Weyl-law unfolding.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/spectralutils.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/unfolding.jl`

## Required background reading
- The **`qb-05` findings file** (its `compute_spectrum`/`_finalize_spectrum`
  findings are directly relevant here since those live in
  `spectralutils.jl`).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/12.5-spectral-api-consolidation.md`
  — the authoritative design record for this exact file pair: moved
  `area`/`fundamental_area`/`symmetry_reduction_factor`/`corner_angles` OUT
  to `BilliardGeometry.jl/src/geometry/area.jl` (confirm they are indeed
  gone from here, not duplicated); added `state_at_k`/`spectral_density`/
  `k_range_for_states`; consolidated the sort/empty-check/construct
  `SpectralData` tail into `_finalize_spectrum`, documented as safe under a
  "workers write disjoint slots, one thread finalizes" pattern.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/09.5-compute-spectrum.md`
  — `overlap_and_merge!` (window-boundary-based) vs.
  `overlap_and_merge_ebim!` (spacing-adaptive clustering) — two genuinely
  different merge strategies; confirm both still exist distinctly and
  aren't accidentally unified into one wrong-for-one-case function.

## Audit checklist
1. **Dead code** — confirm `area`/`fundamental_area`/
   `symmetry_reduction_factor`/`corner_angles` are fully gone from this
   package (not a half-migration leaving a duplicate or a thin
   re-exporting shim nobody documented).
2. **Missing wiring** — are `state_at_k`/`spectral_density`/
   `k_range_for_states`/`weyl_law` all exported from the module file per
   `qb-01`'s findings?
3. **Missing implementations** — any `compute_spectrum` overload the
   `qb-05` findings flagged as missing/mis-dispatched, now viewed from the
   definition side (`spectralutils.jl`) rather than the call side.
4. **Unstable APIs** — `SpectralData`'s field types (concrete numeric
   vectors, not `Any`); `_finalize_spectrum`'s return type consistency
   across every caller.
5. **Vulnerabilities** — `_finalize_spectrum`'s documented "workers write
   disjoint slots, one thread finalizes" thread-safety pattern: verify
   every actual caller (cross-reference `qb-05`) truly writes disjoint
   slots and there's no read-before-write race; `overlap_and_merge!`/
   `overlap_and_merge_ebim!` — any silent eigenvalue drop at a merge
   boundary (an off-by-epsilon tolerance bug is easy to introduce here).
6. **Consolidation opportunities** — do `overlap_and_merge!` and
   `overlap_and_merge_ebim!` share enough structure that a common helper
   should be factored out (only if genuinely safe — they're documented as
   "genuinely different merge strategies", so don't force unification if
   it isn't warranted).
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`.

## Out of scope for this step
- `BilliardGeometry.jl`'s `area.jl` (the destination of the 12.5 move) →
  covered in `bg-01`.
- The accelerated solvers that call into `compute_spectrum` → `qb-05`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-07-spectra-findings.md`.
