# Step qb-02: Basis functions (plane waves & corner-adapted Fourier–Bessel)

## Goal
Audit the two concrete `AbsBasis` implementations against the abstract
contract from `qb-01`, and against each other for API consistency.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/fourierbessel/corneradapted.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/abstracttypes.jl` (the
  `AbsBasis` docstring: `resize_basis`/`basis_fun`/`gradient`/
  `basis_and_gradient` are the documented required API)

## Required background reading
- The **`qb-01` findings file** (read first).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/15-symmetry-representation-refactor.md`
  — added `RealPlaneWaves(dim, billiard, sector)` as a new constructor
  overload implementing the `SymmetrySector` representation choice
  (verified to reproduce the old `sym_x`/`sym_y` keyword constructor
  exactly). Confirm both constructor forms still coexist correctly and
  agree.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/16-bim-symmetrysector-migration.md`
  — explicitly flags (as an out-of-scope non-goal for that plan, but a real
  gap): `RealPlaneWaves`/`CornerAdaptedFourierBessel` cannot represent an
  `NFoldRotation` sector at all. Confirm this is still true and not
  silently wrong (e.g. does it error clearly, or silently build a basis
  for the wrong sector?).

## Audit checklist
1. **Dead code** — unused fields/helper functions in either basis file.
2. **Missing wiring** — do both types implement all four `AbsBasis`
   API methods (`resize_basis`, `basis_fun`, `gradient`,
   `basis_and_gradient`)? Are both exported?
3. **Missing implementations** — the `NFoldRotation` sector gap above:
   confirm current behavior (error message quality, or silent wrong
   result) and whether any partial/stub attempt exists.
4. **Unstable APIs** — compare `RealPlaneWaves`/`CornerAdaptedFourierBessel`
   constructor signatures for gratuitous inconsistency (keyword names,
   `dim` positional-vs-keyword, type-parameter propagation from
   `billiard`/`sector` inputs); confirm `basis_fun`/`gradient` hot paths
   don't allocate per-call in a way inconsistent with the mode's
   memory-efficiency conventions (preallocated output, `@view`).
5. **Vulnerabilities** — any unchecked assumption that a caller-supplied
   `dim` is positive/finite; corner-adapted basis's handling of a
   zero/negative corner angle input.
6. **Consolidation opportunities** — any logic duplicated between the two
   basis files that could be a shared helper in `utils/`.
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`'s authoritative list.
8. **Threading** — if either basis's matrix-filling path is
   multithreaded, confirm it follows the mode's "inner single-threaded,
   outer parallel" rule (one `@use_threads` over independent columns, no
   nested threading, disjoint per-thread writes).

## Out of scope for this step
- How these bases are consumed by solvers (`matrixconstructors.jl`) →
  step `qb-03`.
- `SymmetrySector`/`symmetry_sector` itself (the representation-choice
  struct, as opposed to how basis constructors consume it) → step `qb-10`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-02-basis-findings.md`.
