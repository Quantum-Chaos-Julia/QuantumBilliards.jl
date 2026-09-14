# Step bg-03: Concrete domains & curve segments

## Goal
Audit the concrete `AbsDomain`/`AbsCurve` implementations that billiards are
assembled from: circular/composite/multiply-connected domains, and
line/circle/composite/polar curve segments.

## Scope — files to read in full
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/circular.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/compositedomains.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/multiplyconnecteddomains.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/polygons.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/segments/circlesegment.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/segments/compositecurves.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/segments/linesegment.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/segments/polarcurves.jl`

## Required background reading
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/geometry-audit-10.9.md`
  — item C: `PolarSegment`→`FourierCoeffPolarSegment` rename plus a new
  function-based `PolarSegment{T,BC,F}` (arbitrary `r_func(φ)`,
  `ForwardDiff`-based derivatives) both live in `segments/polarcurves.jl`.
  Confirm both types are complete (constructor `promote_type` fix included)
  and that `curvederivatives.jl` (step `bg-01`) has real (not stub)
  `tangent`/`tangent_2` methods for the function-based variant.
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/known-bugs.md`
  — `connect_curves` fix (affects how these domain types compose curves
  into full boundaries) and the Step 12 `AbsMultiplyConnectedDomain`
  formalization.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/12-multiply-connected-domains.md`
  — design record for `multiplyconnecteddomains.jl`'s `genus`
  attribute and the `AnnularBilliard` refactor; confirm the implementation
  matches this plan (no half-done rename artifacts).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/11-billiards-and-domains.md`
  — original porting plan for this whole file set; cross-check for anything
  the plan called for that never landed.

## Audit checklist
1. **Dead code** — any domain/segment constructor or helper with no call
   site in any billiard (step `bg-05`) or test.
2. **Missing wiring** — do `PolarSegment` (function-based) and
   `FourierCoeffPolarSegment` both implement every method `AbsCurve`
   requires (check against `abstracttypes`/`geometry.jl`'s documented
   curve API from step `bg-01`)? Any domain type missing a required
   `AbsDomain`/`AbsPolarDomain` method?
3. **Missing implementations** — any segment/domain constructor that
   silently no-ops or partially builds a curve for an edge-case input
   (zero-length segment, degenerate polygon, single-component "multiply
   connected" domain).
4. **Unstable APIs** — constructor type-parameter propagation: do
   `LineSegment`/`CircleSegment`/`PolarSegment`/`FourierCoeffPolarSegment`
   all `promote_type` their numeric fields consistently? Compare signatures
   across sibling segment types for gratuitous inconsistency (keyword vs.
   positional order, `center::SVector{2,T}` vs. separate `x,y`).
5. **Vulnerabilities** — unchecked negative radius/degenerate-geometry
   inputs in `circular.jl`/`polygons.jl` constructors; `polarcurves.jl`'s
   `ForwardDiff`-based derivative path for arbitrary user `r_func` (does a
   non-differentiable or non-finite user function propagate a NaN silently
   instead of erroring near it)?
6. **Consolidation opportunities** — duplicated point-in-domain or
   curve-length logic between `circular.jl` and `polygons.jl`;
   `compositedomains.jl` vs. `multiplyconnecteddomains.jl` overlap now that
   Step 12 formalized the latter — is `compositedomains.jl` still doing
   anything `multiplyconnecteddomains.jl` doesn't already cover, or is part
   of it now redundant?
7. **Export audit** — per `julia-refactor-scoped`, including both new
   `PolarSegment`/`FourierCoeffPolarSegment` names post-rename (make sure no
   stale `export PolarSegment` refers to the wrong type, and both are
   actually exported).

## Out of scope for this step
- Symmetry framework → step `bg-02` (covered separately; only check that
  domain/segment types correctly *implement* the `_apply_symmetry_to_curve`
  contract if visible here, don't re-audit the framework itself).
- Quadrature/sampling → step `bg-04`.
- Concrete billiards using these domains/segments → step `bg-05`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/bg-03-domains-segments-findings.md`.
