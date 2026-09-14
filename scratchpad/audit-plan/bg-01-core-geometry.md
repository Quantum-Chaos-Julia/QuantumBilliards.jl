# Step bg-01: Core geometry primitives & abstract types

## Goal
Audit the shared geometric building blocks that every curve, domain and
billiard in the package is built from: curve arclength/derivative machinery,
generic geometric utilities, boundary-condition/type scaffolding, and area
computation.

## Scope — files to read in full
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/BilliardGeometry.jl` (module file — abstract-type block + `include`/`export` list)
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/geometry.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/arclength.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/curvederivatives.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/utils.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/inversions.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/poincarebirkhoff.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarycomponents.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/area.jl`

## Required background reading
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/geometry-audit-10.9.md`
  — records that `geometry.jl`/`arclength.jl`/`utils.jl`/`inversions.jl`/
  `poincarebirkhoff.jl`/`boundarytypes.jl` were already confirmed
  functionally identical to `-develop` (only a docstring pass flagged as
  missing) and that `tangent_vec`/`normal_vec`/`curvature` were added to
  `curvederivatives.jl` (Gap 4, fixed). Don't re-report the same
  already-fixed gaps; do check whether the promised docstring pass ever
  happened.
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/known-bugs.md`
  — the `connect_curves` multi-component fix (Step 11) lives in `utils.jl`;
  confirm the fix is still present and no regression was reintroduced.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/12.5-spectral-api-consolidation.md`
  — moved `area`/`fundamental_area`/`symmetry_reduction_factor`/
  `corner_angles` into this package's new `area.jl`; verify that move is
  complete and nothing was left half-migrated in `QuantumBilliards.jl`.

## Audit checklist
1. **Dead code** — any function in these files with zero call sites
   anywhere in the workspace (search across all 8 workspace packages, not
   just this one).
2. **Missing wiring** — every `struct`/`function` defined here that looks
   public (documented, used elsewhere) but is missing from
   `BilliardGeometry.jl`'s top-level `export` list; every abstract type
   declared in the module file whose documented API (if any) isn't fully
   implemented by these files' concrete pieces.
3. **Missing implementations** — leftover `error(...)`/`TODO`/`@warn
   "not implemented"` stubs; the promised-but-maybe-never-done docstring
   pass flagged in `geometry-audit-10.9.md`.
4. **Unstable APIs** — `curvature`/`tangent_vec`/`normal_vec` generic
   dispatch over `<:AbsCurve`: confirm no method silently falls back to
   `Any`/abstract container types; check `poincarebirkhoff.jl` and
   `inversions.jl` for any non-concrete field or return type.
5. **Vulnerabilities** — unchecked bounds/index arithmetic in
   `boundarycomponents.jl`/`utils.jl` (`connect_curves` and friends) given
   they process caller-supplied curve lists; silent empty-result fallbacks
   that should instead error.
6. **Consolidation opportunities** — duplicated arclength/derivative
   formulas between `arclength.jl` and `curvederivatives.jl`; anything in
   `utils.jl` that duplicates a `LinearAlgebra`/`StaticArrays` builtin.
7. **Export audit** — as defined in the `julia-refactor-scoped` skill,
   applied to this file set specifically.
8. **Area-specific**: confirm `area.jl`'s `corner_angles` is actually
   consumed by every billiard/domain that has corners (cross-check briefly
   against `abstracttypes`/domain code, but do not deep-dive billiards here
   — that's step `bg-05`).

## Out of scope for this step
- `geometry/symmetry.jl`, `symmetryregistry.jl`, `symmetryorbits.jl`,
  `fullboundary.jl` → step `bg-02`.
- `geometry/domains/*`, `geometry/segments/*` → step `bg-03`.
- `quadrature/*` → step `bg-04`.
- `geometry/billiards/*` → step `bg-05`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections directly
into the prompt (the subagent is stateless and cannot re-read this plan on
its own initiative unless told to). State explicitly: "This is a
user-requested whole-package audit step for `BilliardGeometry.jl`, not a
single-file review — use the package-wide-audit branch of
`julia-refactor-scoped`. Read-only pass: report findings only, do not edit
any source file."

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/bg-01-core-geometry-findings.md`.
