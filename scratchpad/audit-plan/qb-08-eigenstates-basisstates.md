# Step qb-08: Eigenstates & basis states

## Goal
Audit the state representation layer that solvers produce and later
wavefunction/boundary-function/Husimi code (step `qb-09`) consumes.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/eigenstates.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/basisstates.jl`

## Required background reading
- The **`qb-04` findings file** (BIM solvers populate `BIMEigenstate`).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/05.5-bim-state-normal-derivative.md`
  — `BIMEigenstate{K,T,S,Bi}` now stores `pts`/`u`/`bnd_norm` (in addition
  to `vec`/`ten`) populated by the generic `solve_state`, making
  `boundary_function`/`momentum_function`/`husimi_function`/`wavefunction`
  plain field reads for **any** `SweepBIMSolver`. Confirm the struct fields
  and their types still match this description exactly (a common drift
  point: someone widens a field to `Union{Nothing,...}` "just in case" and
  breaks the "plain field read" guarantee).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/01-quick-wins-decomposition-psm.md`
  — `BasisState`/`BasisEigenstate` should already be stable from before the
  BIM work started; confirm no accidental field/API drift from later steps
  touching shared code.

## Audit checklist
1. **Dead code** — any field on `BasisEigenstate`/`BasisState`/
   `BIMEigenstate` never read anywhere; any constructor overload with no
   call site.
2. **Missing wiring** — are `BasisEigenstate`/`BasisState`/`BIMEigenstate`/
   `compute_eigenstate` all exported per `qb-01`'s findings?
3. **Missing implementations** — any `AbsState`-documented method (check
   `abstracttypes.jl` from `qb-01`) not implemented for one of these
   concrete state types.
4. **Unstable APIs** — `BIMEigenstate{K,T,S,Bi}`'s four type parameters:
   confirm all four are always concretely inferred at construction (no
   parameter defaults to `Any` under some code path); compare
   `BasisEigenstate`'s type-parameterization style for consistency.
5. **Vulnerabilities** — any place a state's `vec`/`ten`/`bnd_norm` field
   could be read before being populated (partially-constructed state
   escaping to caller code); unchecked assumption that `pts`/`u` arrays
   are non-empty in downstream consumers.
6. **Consolidation opportunities** — shared normalization/evaluation logic
   between `BasisEigenstate` and `BIMEigenstate` that's duplicated instead
   of being one generic `AbsState` method.
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`.

## Out of scope for this step
- Actual wavefunction/boundary-function/Husimi evaluation logic that reads
  these states → `qb-09`.
- Solver-side population of these states → `qb-04`/`qb-05`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-08-eigenstates-basisstates-findings.md`.
