# Step qb-10: Symmetry states (`SymmetrySector` representation & reflections)

## Goal
Audit the `QuantumBilliards.jl`-side half of the symmetry story: how a
geometric symmetry group registered on a `BilliardGeometry.jl` billiard
(audited in `bg-02`/`bg-05`) becomes a chosen wavefunction/BIM
*representation*, and how symmetric boundary curves are reflected back
into a full-domain plot/evaluation.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/reflections.jl`

## Required background reading
- The **`bg-02` and `bg-05` findings files** (this step is the direct
  continuation of those on the `QuantumBilliards.jl` side — read them
  first, don't re-derive the geometric-symmetry-group analysis here).
- The **`qb-02` findings file** (`RealPlaneWaves(dim, billiard, sector)`
  constructor).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/15-symmetry-representation-refactor.md`
  — `SymmetrySector`/`symmetry_sector(billiard, GenType=>value...)` is the
  authoritative design for this exact file; confirm the implementation
  matches (resolves against a specific billiard's registered `sym_id`s,
  no leftover positional-vector-order dependency the refactor was meant to
  eliminate).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/16-bim-symmetrysector-migration.md`
  — "plan only, not started (v2)": adds an additive `character::Tuple`
  field, a `_fold_boundary` dispatch adapter, and a fully general
  `_resolve_bim_symmetry(billiard, sector::SymmetrySector)` helper
  (including full `Z2×Z2` folding via a freshly-built `CompositeReflection`
  and correct multi-`sym_id` `Zn` sector recovery), explicitly flagging one
  irreducible validation error (a lone `XYAxisReflection` character alone
  cannot determine a unique `Z2×Z2` irrep) and one non-goal (no dihedral/2-D
  irrep support). **Since this plan is marked "not started", confirm
  `symmetrysector.jl` does NOT yet contain `_resolve_bim_symmetry`/
  `_fold_boundary`** — if it does, the migration-plan status note is stale
  and should be flagged; if it doesn't, confirm this is a real, currently
  open "missing implementation" (not a dead plan) since `qb-04`'s findings
  about DLP/CFIE/CompositeBIM's bare `symmetry` field depend on it.

## Audit checklist
1. **Dead code** — any symmetry-sector helper with no call site.
2. **Missing wiring** — is `SymmetrySector`/`symmetry_sector` exported per
   `qb-01`'s findings?
3. **Missing implementations** — Step 16's `_resolve_bim_symmetry`/
   `character` field status (see above) — this is the step's headline
   finding if still missing, since it directly explains the `qb-04`
   DLP/CFIE/CompositeBIM limitation.
4. **Unstable APIs** — `symmetry_sector(billiard, GenType=>value...)`'s
   variadic-pair argument style: confirm it validates unknown/duplicate
   generator types with a clear error rather than silently ignoring extras.
5. **Vulnerabilities** — silent resolution to the trivial representation
   when a caller passes a `sym_id` not registered on the given `billiard`
   (cross-reference `bg-02`'s Gap 6 note about no
   `SymmetryWall`/`billiard.symmetries`/solver-sector consistency check —
   this is the layer where such a check would actually need to live).
6. **Consolidation opportunities** — overlap between `reflections.jl`'s
   boundary-curve reflection logic and `BilliardGeometry.jl`'s
   `_apply_symmetry_to_curve`/`fullboundary.jl` machinery (audited in
   `bg-02`) — is there duplicated reflection-formula code across the
   package boundary that should call the geometry package's version
   instead?
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`.

## Out of scope for this step
- Geometric symmetry-group registration itself → `bg-02`/`bg-05`.
- Basis-constructor consumption of a `SymmetrySector` → `qb-02`.
- BIM solver `symmetry` field mechanics beyond confirming Step 16's status
  → detailed solver-body audit already done in `qb-04`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-10-symmetry-states-findings.md`.
