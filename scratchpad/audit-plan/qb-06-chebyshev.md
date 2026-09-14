# Step qb-06: Chebyshev-accelerated Hankel/Bessel evaluation backend

## Goal
Audit the Chebyshev interpolation performance backend behind the
`use_chebyshev`/`cheb_config` fields audited in `qb-05`, and cross-check it
against the already-completed (read-only) gap analysis from the migration
plan.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/chebyshev/core.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/chebyshev/bessels.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/chebyshev/optimalpanelization.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/chebyshev/dlp.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/chebyshev/cfie.jl`

## Required background reading
- The **`qb-04` and `qb-05` findings files**.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/10-chebyshev-acceleration.md`
  — original port plan for all 5 files here.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/10.5-chebyshev-gap-analysis.md`
  — **the single most important background document for this step**: a
  prior read-only audit that already found and prioritized 5 specific gaps:
  (1) missing ±5% radial-interval padding margin before building Chebyshev
  plans (accuracy risk), (2) Beyn's multi-`k` assembly doing `nq` separate
  `O(N²)` passes instead of one combined pass using existing-but-unused
  combinator functions in `bessels.jl` (also flagged in `qb-05`), (3) a
  missing k-independent panel-index cache across a sweep, (4) the
  still-unported SLP/CFIE wavefunction Chebyshev reconstruction, (5) a
  negligible redundant-computation style issue in the derivative kernel
  entries. **This step's primary job is to determine, for each of these 5
  items, whether it is still open, partially fixed, or fully fixed** —
  don't independently re-derive them from scratch; verify against the
  current file contents.
- Also re-confirms it found **no regression** for a previously-suspected
  missing `eval_h`/`eval_j` accuracy guard (traced the actual hot-loop call
  path) — confirm that finding still holds too.

## Audit checklist
1. **Dead code** — any combinator function in `bessels.jl` that the 10.5
   analysis flagged as "existing unused" — is it still unused, or now
   wired in per item (2) above?
2. **Missing wiring** — is `optimalpanelization.jl`'s panel-index cache
   (item 3) present? Is it actually consumed by `beyn.jl`/`ebim.jl` (check
   with a light read of those files, deep audit already done in `qb-05`)?
3. **Missing implementations** — items (1) and (4) above: padding margin,
   SLP/CFIE wavefunction Chebyshev reconstruction.
4. **Unstable APIs** — `core.jl`'s Chebyshev plan struct(s): concrete field
   types, no `Any`; `dlp.jl`/`cfie.jl` kernel-evaluation function signature
   consistency with their non-Chebyshev counterparts in `qb-04`/`qb-05`
   (should be drop-in alternatives, not diverging APIs).
5. **Vulnerabilities** — interpolation accuracy: any place where an
   out-of-range query point (outside the padded radial interval) is
   silently extrapolated instead of erroring or re-planning; item (1)'s
   missing padding margin is itself an accuracy vulnerability if still open
   — flag with severity accordingly.
6. **Consolidation opportunities** — item (5), the redundant-computation
   style issue in derivative kernel entries — still applicable?
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`.
8. **Performance note (not a fix)**: since this whole subsystem exists
   purely for performance, explicitly re-time or reason about whether item
   (2) (Beyn's `O(N²)` multi-pass) is still the dominant cost if unfixed —
   useful for prioritizing the later remediation pass.

## Out of scope for this step
- Non-Chebyshev kernel evaluation (the `use_chebyshev=false` path) →
  `qb-04`/`qb-05`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt, emphasizing that `10.5-chebyshev-gap-analysis.md` is required
reading before forming any conclusion. State this is a whole-package audit
step (package-wide branch of `julia-refactor-scoped`), read-only: report
only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-06-chebyshev-findings.md`,
with an explicit "10.5 gap status" table (gap # → still open / partially
fixed / fixed) as the first section.
