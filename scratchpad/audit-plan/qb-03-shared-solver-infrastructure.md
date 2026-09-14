# Step qb-03: Shared solver infrastructure

## Goal
Audit the generic infrastructure every concrete solver (steps `qb-04`,
`qb-05`) builds on: boundary point/geometry caching, matrix construction,
generalized-eigenvalue decompositions, and the generic sweep-method/grading
dispatch layer.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/boundarypoints.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/boundarygeomcache.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/decompositions.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/matrixconstructors.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/boundarygrading.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl`

## Required background reading
- The **`qb-01` findings file**.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/00-index.md`
  "Shared infrastructure already in place" section — lists exactly what
  this file set is supposed to already provide (`solve_wavenumber`/
  `k_sweep` generics, `evaluate_points` delegation, `BoundaryGrading`
  traits, `generalized_eigen`/`generalized_eigvals`/etc.) — confirm nothing
  regressed since steps 1–16 landed.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/05.5-bim-state-normal-derivative.md`
  — `solve_state`/`_bim_normal_derivative` generic added to
  `sweepmethods.jl`; confirm it's still generic over every `SweepBIMSolver`
  (not accidentally specialized back to one solver) and that no redundant
  second matrix assembly/Krylov solve was reintroduced.
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/geometry-audit-10.9.md`
  — Gap 4 note: `boundarygeomcache.jl`'s `kappa` field is a *differently
  normalized* curvature quantity than `BilliardGeometry.jl`'s generic
  `curvature()` (speed² vs. speed³ denominator) and the two must **not** be
  merged. Confirm this distinction is still respected (no accidental
  "consolidation" that breaks the DLP/CFIE diagonal kernel limit).

## Audit checklist
1. **Dead code** — any struct field or function in this set with no call
   site across the whole solver tree.
2. **Missing wiring** — is every public-looking type/function here
   exported? Does every `AbsBIMSolver`/`AbsBasisSolver` concrete type
   (steps `qb-04`/`qb-05`) actually get `solve_wavenumber`/`k_sweep`/
   `solve_state` "for free" as the migration plan claims, or does any
   solver silently fail to dispatch into this generic path?
3. **Missing implementations** — any `BoundaryGrading` subtype
   (`SmoothPeriodicGrading`/`CornerGrading`/`GlobalCornerGrading`) missing
   its `_bim_numeric_type` hook or another required grading-trait method.
4. **Unstable APIs** — `BoundaryPanelArrays`/`BoundaryGeomCache` field
   concreteness; `generalized_eigen`/`generalized_eigvals`/
   `generalized_eigen_all` return-type consistency (do all three return the
   same eigenvector/eigenvalue element type family for the same input
   type?).
5. **Vulnerabilities** — `matrixconstructors.jl`: any `@inbounds` block
   whose bound is guaranteed only under an assumption not checked at the
   call site; silent NaN/Inf propagation in matrix filling instead of an
   early error; `decompositions.jl`'s generalized eigenproblem solvers —
   any silent fallback when the matrix pencil is ill-conditioned instead of
   surfacing that to the caller.
6. **Consolidation opportunities** — per `matrixconstructors.jl`'s existing
   "INTERNAL FUNCTIONS" comment-block convention (referenced by the
   `julia-refactor-scoped` skill) — are there other files in this set that
   should adopt the same internal/public grouping convention instead of
   ad hoc underscore-prefixing?
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`.
8. **Threading/BLAS** — confirm every parallel loop here follows
   "inner single-threaded, outer parallel", and every BLAS-calling
   threaded region is wrapped with `@blas_multi`/equivalent to avoid
   oversubscription, per the mode's conventions.

## Out of scope for this step
- Concrete solver bodies (`dlp.jl`, `cfie.jl`, etc.) → steps `qb-04`/`qb-05`.
- Chebyshev backend → `qb-06`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-03-shared-solver-infrastructure-findings.md`.
