# Step bg-02: Symmetry framework (`SymmetryRegistry`/`SymmetryWall`/`SymmetryOrbitMap`/`full_boundary`)

## Goal
Audit the structural machinery that makes billiard symmetries usable: the
geometric symmetry-group registry, the boundary-condition marker for
symmetry walls, the fundamental-to-full-boundary orbit folding, and the
full-boundary reconstruction from a fundamental domain. This is the part of
the package the user specifically flagged ("find any missing implementations
for symmetries on billiards that don't work") — this step covers the
*generic machinery*; step `bg-05` covers *per-billiard* consumption of it.

## Scope — files to read in full
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryregistry.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/fullboundary.jl`

## Required background reading
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/geometry-audit-10.9.md`
  — full history of Gaps 1–6 for this exact file set. **Gap 1** (antisymmetric
  `symmetry_index_orbits` never wiring in `symmetry_irrep_character`) and
  **Gap 3** (`full_boundary` missing polar-curve symmetry methods) are marked
  fixed — verify they are *still* fixed after the Step 15/16 symmetry
  refactor rewrote this exact machinery (the note predates that refactor).
  **Gap 2** (no multicomponent/multi-ring `symmetry_index_orbits` overload,
  blocks symmetric multiply-connected billiards) and **Gap 6** (no
  cross-check between `SymmetryWall` markers, `billiard.symmetries`, and a
  solver's chosen sector) were explicitly *deferred* pending a symmetric
  multiply-connected fixture — check whether `AnnularBilliard` (Step 12) or
  any other billiard now provides that fixture, and if so, whether Gap 2/6
  are still open or have been silently fixed/silently still broken.
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/known-bugs.md`
  — "Symmetry framework refactor (Step 15)" section: `symmetry_irrep_character`
  was *removed* and replaced by `sym_id`-only geometric symmetries plus
  `QuantumBilliards.jl`'s `SymmetrySector`. Confirm `symmetry.jl`/
  `symmetryorbits.jl` match this description exactly (no leftover
  `irrep_character`-based code path).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/15-symmetry-representation-refactor.md`
  and `16-bim-symmetrysector-migration.md` — the authoritative design record
  for the current `sym_id`/`SymmetryRegistry`/`SymmetryWall` API. Step 16 is
  "plan only, not started" — check whether any of its `BilliardGeometry.jl`-side
  prerequisites (if any) were already implemented anyway.

## Audit checklist
1. **Dead code** — any symmetry-type dispatch method with zero
   call sites (e.g. leftover from before the Step 15 rewrite); any field on
   `AbsReflection`/`NFoldRotation` subtypes no longer read anywhere.
2. **Missing wiring** — every concrete symmetry type
   (`XAxisReflection`/`YAxisReflection`/`XYAxisReflection`/
   `DiagonalReflection`/`AntiDiagonalReflection`/`CompositeReflection`/
   `NFoldRotation`) must implement the same dispatch trio documented in
   `geometry-audit-10.9.md`'s "Symmetry-framework extensibility verdict"
   (`apply_symmetry`, `symmetry_node_multiple`+`symmetry_index_orbits`,
   `_apply_symmetry_to_curve`/`_orientation_reversing` in
   `fullboundary.jl`) — find any type missing one of these.
3. **Missing implementations** — is Gap 2 (multicomponent orbit folding)
   genuinely still missing given `AnnularBilliard` now exists? If so, is it
   silently wrong (returns something) or does it error loudly? Also check
   whether `CompositeBIMSolver` (see step `qb-04`) actually calls the
   single-ring overload on multi-ring points today, per the Gap 2 note.
4. **Unstable APIs** — `SymmetryRegistry`/`SymmetryWall` field types;
   confirm `sym_id`/`sector_id` are concrete `Int` everywhere, not
   `Union{Nothing,Int}` used inconsistently.
5. **Vulnerabilities** — silent fallback to the trivial/fully-symmetric
   representation when a caller-supplied `character`/`sector` doesn't match
   any registered `sym_id` (should this error instead of silently
   mis-folding?); unchecked assumption that `billiard.symmetries` and a
   solver's chosen sector reference the *same* billiard (Gap 6).
6. **Consolidation opportunities** — repeated per-symmetry-type boilerplate
   in `symmetry.jl` that could be one generic method plus a small per-type
   trait table.
7. **Export audit** — per `julia-refactor-scoped`.
8. **Symmetry-specific**: for each concrete symmetry type, is there at
   least one billiard in the catalogue (step `bg-05`) that actually
   registers and exercises it? Flag any symmetry type that exists in code
   but is provably unused by every billiard — a strong "dead/never-tested"
   signal even if not technically dead code.

## Out of scope for this step
- Core geometry (`geometry.jl`, `arclength.jl`, etc.) → step `bg-01`.
- Concrete domains/segments → step `bg-03`.
- Per-billiard symmetry registration and runtime behavior → step `bg-05`.
- `QuantumBilliards.jl`'s `SymmetrySector`/`symmetry_sector` (the
  representation-choice layer built on top of this) → step `qb-10`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt. State explicitly this is a whole-package audit step (package-wide
branch of `julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/bg-02-symmetry-framework-findings.md`.
