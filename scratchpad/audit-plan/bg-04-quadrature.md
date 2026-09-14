# Step bg-04: Quadrature (sampling & Kress grading)

## Goal
Audit the numerical-quadrature layer: node samplers and Kress corner/periodic
grading used by boundary discretization.

## Scope — files to read in full
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/quadrature/samplers.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/quadrature/kressgrading.jl`

## Required background reading
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/02-shared-bim-infrastructure.md`
  — original port plan for `kress_R!`/`kress_R_even!`/`kress_R_odd!`/
  `s_mid`/`kress_graded_nodes_data`/`multi_kress_graded_nodes_data`; confirm
  every function that plan called for actually exists and matches the
  documented signature (this is shared infrastructure every BIM solver in
  `qb-04` depends on).
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/known-bugs.md`
  and `geometry-audit-10.9.md` — check for any quadrature-adjacent notes
  (e.g. non-uniform-grid Husimi quadrature, step 5.6 in the migration plan,
  is a `QuantumBilliards.jl`-side consumer of node spacing from this file —
  don't re-audit that consumer here, just confirm this file exposes what it
  needs).

## Audit checklist
1. **Dead code** — any sampler/grading function with no call site (check
   both `BilliardGeometry.jl` and `QuantumBilliards.jl`, since
   `boundarypoints.jl`/`boundarygeomcache.jl` there are the main consumers).
2. **Missing wiring** — is every public-looking sampler
   (`LinearNodes`/`GaussLegendreNodes`/`FourierNodes`/`sample_points`) and
   grading function actually exported from `BilliardGeometry.jl`'s module
   file?
3. **Missing implementations** — any grading/sampling case
   (single-corner vs. multi-corner, even vs. odd Kress power) that's
   documented/expected but not implemented, or implemented only for one
   numeric type.
4. **Unstable APIs** — do `kress_R!`/`kress_R_even!`/`kress_R_odd!` mutate
   in place consistently (as the `!` suffix implies) without also
   allocating a fresh array internally? Confirm return-type consistency
   between `kress_graded_nodes_data` and `multi_kress_graded_nodes_data`.
5. **Vulnerabilities** — off-by-one or wraparound bugs in periodic node
   indexing (`mod1`-style arithmetic is easy to get subtly wrong at
   segment boundaries); unchecked degenerate corner angle (angle ≈ 0 or 2π)
   causing division blow-up in the grading formula.
6. **Consolidation opportunities** — do `kress_graded_nodes_data` and
   `multi_kress_graded_nodes_data` share enough logic that one should call
   the other instead of duplicating the grading-formula body?
7. **Export audit** — per `julia-refactor-scoped`.
8. **Performance note (not a fix)**: since this is a hot path for every BIM
   solve, flag (don't fix) any obviously avoidable per-call allocation in
   the node-generation functions for the later performance-focused
   remediation pass.

## Out of scope for this step
- Everything upstream of quadrature (curves/domains/symmetry) → steps
  `bg-01`–`bg-03`.
- `QuantumBilliards.jl`'s consumers of this module
  (`boundarypoints.jl`/`boundarygeomcache.jl`) → step `qb-03`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/bg-04-quadrature-findings.md`.
