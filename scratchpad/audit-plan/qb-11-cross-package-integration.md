# Step qb-11: Cross-package integration (final pass)

## Goal
A final cross-cutting pass, run only after all other 15 steps have their
findings files written, since it's meant to cross-reference them rather
than re-read every source file from scratch. Covers: whole-module
export/dependency sanity for both `BilliardGeometry.jl` and
`QuantumBilliards.jl` together, and a read-only check for whether the
sibling plotting/statistics packages call anything that no longer exists.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/QuantumBilliards.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/Project.toml`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/BilliardGeometry.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/Project.toml`
- Read-only, grep-level pass (not full audits) over:
  `/home/clozej/.julia/dev/QBPlotting.jl/src/`,
  `/home/clozej/.julia/dev/BilliardGeometryPlotting.jl/src/`,
  `/home/clozej/.julia/dev/QCPlotting.jl/src/`,
  `/home/clozej/.julia/dev/SpectralStatistics.jl/src/` — only to find
  call sites referencing `QuantumBilliards`/`BilliardGeometry` symbols, not
  to audit those packages' own internals (out of scope — the user asked
  for an audit of `QuantumBilliards`/`BilliardGeometry` only).

## Required background reading
- **All 15 prior findings files** in
  `QuantumBilliardsTests/scratchpad/audit-plan/findings/` — read every one
  before starting. This step's entire value is in cross-referencing them.

## Audit checklist
1. **Dead code** — cross-package: any exported symbol from either package
   with zero usages anywhere in the whole 8-folder workspace (not just its
   own package) — the strongest possible "safe to remove" signal.
2. **Missing wiring** — reconcile every "missing export" flagged across the
   15 prior findings files into one consolidated list per package; check
   `Project.toml` `[deps]` for a package that's `using`-imported in source
   but not declared, or declared but never `using`-imported anywhere (an
   unused dependency).
3. **Missing implementations** — consolidate every "still open" gap from
   steps `bg-01`–`qb-10` into one prioritized list (this is the step's main
   deliverable value: a single view across all 15 passes).
4. **Unstable APIs** — any cross-package API mismatch, e.g.
   `QuantumBilliards.jl` calling a `BilliardGeometry.jl` function with an
   argument type that only works for some concrete `AbsCurve`/`AbsDomain`
   subtype, silently failing to dispatch for others found in the billiards
   catalogue (`bg-05`).
5. **Vulnerabilities** — consolidate any cross-file silent-fallback pattern
   noticed repeatedly across steps (e.g. multiple steps independently
   flagging "resolves to trivial representation instead of erroring" is a
   stronger signal than one isolated report).
6. **Consolidation opportunities** — any utility duplicated across
   `BilliardGeometry.jl`'s `utils.jl`/`geometryutils.jl` and
   `QuantumBilliards.jl`'s `utils/geometryutils.jl` (flagged narrowly in
   `qb-01`) — resolve with a concrete recommendation now that both sides'
   findings are available.
7. **Export audit** — the definitive merged list, superseding the
   per-step lists in `qb-01`'s and each `bg-*`/`qb-*` file's own findings.
8. **Broken external call sites**: does any grep hit in
   `QBPlotting.jl`/`BilliardGeometryPlotting.jl`/`QCPlotting.jl`/
   `SpectralStatistics.jl` reference a symbol that a prior finding flagged
   as renamed/removed/never-existed (e.g. the `PolarSegment` rename from
   `bg-03`, or the `symmetry_irrep_character` removal from `bg-02`)? Report
   these as high-severity even though fixing the plotting packages
   themselves is out of scope — the user should know if a rename already
   broke a downstream package.

## Out of scope for this step
- Deep review of `QBPlotting.jl`/`BilliardGeometryPlotting.jl`/
  `QCPlotting.jl`/`SpectralStatistics.jl` internals (only grep-level call-site
  checks against the two audited packages, per the user's explicit
  two-package scope — see the open question about this in chat).

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. This
invocation's prompt should be the largest of all 16 steps: paste this file's
Scope/Required background/Checklist sections **and** a concatenation (or
tool-read) of all 15 prior findings files, since the subagent is stateless
and this step's whole point is synthesis across them. State this is the
final step of a whole-package audit (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-11-cross-package-integration-findings.md`.
This should also serve as the **executive summary** of the whole 16-step
audit — after this file is written, do the final user triage pass described
in `00-index.md` before any remediation work begins.
