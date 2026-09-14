# Step bg-05: Billiards catalogue (audited last — per-billiard symmetry correctness)

## Goal
This is the step that directly answers the user's question: **which
billiards' symmetries don't actually work?** Audit every concrete billiard
constructor for correct registration and consumption of the symmetry
machinery audited in step `bg-02`, using the geometry/domain/quadrature
building blocks from steps `bg-01`/`bg-03`/`bg-04`. Audited last, as
requested, since it's the integration point for everything above.

## Scope — files to read in full
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/annular.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/c3.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/circle.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/ellipse.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/limacon.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/mushroom.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/polar.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/polygon.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/prosen.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/rectangle.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/stadium.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/star.jl`
- `/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/triangle.jl`

## Required background reading
- The **findings files from `bg-01` through `bg-04`** (read all four before
  starting this step — this step is explicitly meant to cross-reference
  them, not repeat their analysis).
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/geometry-audit-10.9.md`
  — Gap 2 (multicomponent symmetry-orbit folding, needs a symmetric
  multiply-connected fixture) and Gap 6 (no `SymmetryWall`/
  `billiard.symmetries`/solver-sector consistency check) were deferred
  pending exactly the fixtures this step's file list now provides
  (`annular.jl` for multiply-connected, `c3.jl`/`star.jl` for rotation
  groups, `stadium.jl`/`rectangle.jl`/`ellipse.jl` for `Z2×Z2` reflection
  groups). Determine whether these gaps are now triggerable/reproducible.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/15-symmetry-representation-refactor.md`
  and `16-bim-symmetrysector-migration.md` — the `register_symmetries`/
  `SymmetryRegistry`/`SymmetryWall(sym_id,sector_id)` API every billiard here
  must use consistently; Step 16 ("plan only, not started") flags that
  `RealPlaneWaves`/`CornerAdaptedFourierBessel` still cannot represent an
  `NFoldRotation` sector — confirm whether `c3.jl`/`star.jl` (both rotation
  billiards) still only work with `SweepBIMSolver`s and not basis solvers, or
  whether that gap has since been closed.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/13-compositebim-annulus-verification.md`
  — the only existing *numerical* verification of a multiply-connected
  billiard (`annular.jl` against `CompositeBIMSolver`); note what was and
  wasn't verified.

## Audit checklist
1. **Dead code** — any billiard file with unused helper functions,
   commented-out alternate constructors, or fields never read after
   construction.
2. **Missing wiring** — for each billiard: does it call
   `register_symmetries` with the full geometric symmetry group the shape
   actually has (e.g. `RectangleBilliard`/`StadiumBilliard` should register
   the full `Z2×Z2`; `C3Billiard` the full `Z3`; `StarBilliard`/whatever
   `N`-fold shape uses `Zn`)? Flag any billiard whose registered group is a
   strict subgroup of its true geometric symmetry (a "silently
   under-symmetric" bug, not a crash).
3. **Missing implementations** — any billiard whose `symmetries` field is
   an empty/placeholder registry despite the shape having obvious symmetry
   (e.g. `ellipse.jl`, `mushroom.jl` — does it have a genuine mirror
   symmetry left unregistered)? Any billiard where `full_boundary` cannot
   be called at all due to a missing curve-symmetry method from step
   `bg-02`?
4. **Unstable APIs** — constructor signature consistency across all 13
   billiards (keyword names/defaults for size, angle, `x0`/`y0` shift
   conventions); do all constructors return the same concrete `AbsBilliard`
   pattern (fields promoted to a common `T`)?
5. **Vulnerabilities** — unchecked degenerate parameter ranges (radius ≤ 0,
   angle outside `(0, 2π)`, `c3`/`star`'s wedge angle not evenly dividing a
   full rotation) that would silently build a geometrically wrong billiard
   instead of erroring.
6. **Consolidation opportunities** — per the migration plan's "billiard
   constructor streamlining" non-gap note (10.9 said no shared helper was
   warranted at the time) — re-evaluate now that 13 billiards exist: is
   there now real, provable duplication (e.g. every polygon-based billiard
   repeating the same corner-angle + symmetry-registration boilerplate)
   worth factoring out?
7. **Export audit** — per `julia-refactor-scoped`: every billiard
   constructor exported from the module file.
8. **Runtime symmetry smoke-check (see note below)**: for each billiard
   that registers at least one non-trivial symmetry, verify — by actually
   constructing the billiard and calling `full_boundary(billiard)` — that
   the reconstructed full boundary is a closed, non-self-overlapping curve
   matching the shape's known full extent (not just the fundamental
   domain). This requires running Julia code, not just reading it; use
   available terminal/tool access to do this for every symmetric billiard
   in the list, not just a sample.

## Note on step scope: static review vs. runtime verification
Unlike steps `bg-01`–`bg-04`, "does this billiard's symmetry actually work"
is inherently a runtime question, not something resolvable by code reading
alone. The Julia Refactoring Agent subagent should do the static parts of
the checklist (items 1–7) via code review as usual, **and also** run item 8
by actually executing Julia code against each billiard. Per the confirmed
smoke-check convention (see `00-index.md`), every such check must be saved
as a standalone, re-runnable script at
`QuantumBilliardsTests/scratchpad/audit-plan/smoke-checks/bg-05-<billiard-name>.jl`
(one file per billiard, or one file per symmetry-group family if that reads
better — subagent's judgement, but every script must be self-contained:
`using BilliardGeometry`, construct the billiard, call `full_boundary`,
assert/print closure + non-self-overlap + expected arc length/angle count)
so it can later be handed to the Julia Test Writer agent to become a
permanent `@testset` — do not just paste a REPL transcript into the findings
report instead of saving the script. If the subagent's environment cannot
execute Julia at all, it must say so explicitly in its report rather than
silently skipping item 8, so a follow-up runtime-only pass can be scheduled.

## Out of scope for this step
- The generic symmetry machinery itself (already audited in `bg-02`) —
  reference its findings, don't re-derive them.
- `QuantumBilliards.jl`-side consumption of these billiards (basis
  construction, solver sector selection) → `qb-04`/`qb-10`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Paste this
file's Scope/Required background/Checklist/Out-of-scope sections into the
prompt, plus a summary of the `bg-01`–`bg-04` findings files. State this is
a whole-package audit step (package-wide branch of `julia-refactor-scoped`).
For the static-review checklist items (1–7): read-only, report only, no
edits. Item 8 (runtime smoke-check) explicitly requires executing code —
tell the subagent it must save every such check as a standalone `.jl`
script under `QuantumBilliardsTests/scratchpad/audit-plan/smoke-checks/`
(not just run it and discard), per the confirmed convention in
`00-index.md`. It must not modify any package source file.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/bg-05-billiards-catalogue-findings.md`,
including the per-billiard runtime smoke-check results as an explicit table
(billiard name → symmetry group registered → `full_boundary` check
pass/fail/not-applicable → relative path to its saved smoke-check script in
`smoke-checks/`). Every script referenced in the table must actually exist
on disk under `QuantumBilliardsTests/scratchpad/audit-plan/smoke-checks/`
before this step is considered complete.
