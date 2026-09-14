# Step qb-01: Abstract types, module wiring & utilities

## Goal
Audit the package's abstract-type contracts, the top-level module file's
`include`/`export` completeness, and the low-level shared utilities every
other layer depends on.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/abstracttypes.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/QuantumBilliards.jl` (module file)
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/coordinatesystems.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/geometryutils.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/typeutils.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/macros.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/billiardutils.jl`

## Required background reading
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/00-index.md`
  section "Shared infrastructure already in place" — lists `@use_threads`/
  `@blas_multi`/`@blas_1`/`@blas_multi_then_1` in `macros.jl` as done;
  confirm signatures still match what every solver step (`qb-04`–`qb-06`)
  actually calls.
- The mode's own operating instructions (already in context) define the
  required conventions for type stability, threading (`@use_threads`,
  `@blas_multi`) and macro use — treat these as the correctness bar for
  `utils/*.jl` specifically, since that's where these conventions are
  implemented, not just followed.

## Audit checklist
1. **Dead code** — any abstract type in `abstracttypes.jl` with zero
   concrete subtypes anywhere in the workspace; any utility function in
   `utils/*.jl` with no call site.
2. **Missing wiring** — cross-check `QuantumBilliards.jl`'s `include(...)`
   list against every `.jl` file that actually exists under `src/` (find
   any file never `include`d — true dead weight); cross-check its `export`
   list against every `struct`/`function` defined across the whole `src/`
   tree that looks public (documented, used across files) but isn't
   exported.
3. **Missing implementations** — for each abstract type
   (`AbsBasis`/`AbsSolver`/`AbsBasisSolver`/`AbsBIMSolver`/`AbsState`/etc.,
   whatever is declared here), does its docstring's documented API list
   match what concrete subtypes across the package actually implement? Flag
   any documented-but-never-implemented method, and any concrete subtype
   missing a documented method (cross-reference lightly with steps
   `qb-02`/`qb-04`/`qb-05`/`qb-08`, but the authoritative per-solver check
   happens there — this step only flags contract-vs-declaration mismatches
   visible from `abstracttypes.jl` itself).
4. **Unstable APIs** — any abstract type field (there shouldn't be any —
   abstract types here should be pure interface markers); any utility
   function in `utils/*.jl` with an `Any`-typed argument/return or an
   abstractly-typed local variable in a loop.
5. **Vulnerabilities** — `macros.jl`: does `@use_threads`/`@blas_multi`
   handle a `multithreading=false` path correctly (no silent thread-count
   mismatch with BLAS)? Any macro that `eval`s or splices user-supplied
   expressions in a way that could break hygiene.
6. **Consolidation opportunities** — overlap between `geometryutils.jl` and
   `BilliardGeometry.jl`'s own geometry utilities (should some of this live
   in the geometry package instead, or vice versa — flag only, this is a
   cross-package design question, not a one-file fix); overlap between
   `typeutils.jl` and Julia/StaticArrays builtins.
7. **Export audit** — per `julia-refactor-scoped`, applied package-wide to
   the module file specifically (this is the one step that should produce
   the definitive "missing exports" list other steps can reference instead
   of re-deriving it).

## Out of scope for this step
- Basis/solver/state/spectra concrete implementations — steps `qb-02`
  through `qb-10` each audit their own `export` completeness against the
  list this step produces, but this step itself should not deep-read those
  files beyond what's needed to confirm an `include`/`export` line exists.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-01-abstracttypes-utils-findings.md`.
Since this step's export/include audit is meant to be authoritative for
later steps, make sure the saved report includes the full list of any
files-not-included and symbols-not-exported it finds, not just a summary.
