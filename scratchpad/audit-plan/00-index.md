# Full ecosystem code-quality audit — execution plan

Status: **Planning documents only. No audit steps have been executed yet.**
Execution model confirmed with the user: strictly sequential, one subagent
at a time, read-only findings first, smoke-checks saved as reusable
scripts, memory promotion after the full pass. See "Open questions" below
for the full confirmation record.

## Purpose and relationship to the existing migration plan

`../migration-plan/` (steps 1–16, `00-index.md`) is the **feature-completeness**
plan: it tracked porting solvers/geometry/symmetry features from the
`-develop` reference repos into `BilliardGeometry.jl`/`QuantumBilliards.jl`
until they were numerically correct and wired in. That plan is functionally
done (steps 1–15 done, step 16 planned-only).

This directory is a **different, orthogonal** exercise, explicitly requested
by the user: a whole-package **code-quality audit** of the two now-mature
packages, looking for:

1. **Dead code** — unused functions/structs/branches, unreachable dispatch
   arms, leftover stub bodies from the migration (`error("not implemented")`)
   that were superseded but never deleted.
2. **Missing wiring** — files/functions that exist but aren't `include`d or
   `export`ed, abstract-type API methods (see each package's
   `abstracttypes.jl` / `BilliardGeometry.jl` top-level `abstract type`
   block) that some concrete subtype never implements.
3. **Missing implementations** — TODOs, partially-ported features, silent
   fallbacks to a wrong/default behavior instead of a real implementation.
4. **Unstable APIs** — type instability (`Any`, abstract struct fields,
   non-concrete containers), inconsistent constructor signatures across
   sibling concrete types of the same abstract type, surprising keyword
   defaults.
5. **Vulnerabilities** — unsafe/unchecked assumptions on inputs that cross a
   boundary from user-facing code (constructors, `solve`, plotting call
   sites), `@inbounds` used without an actually-guaranteed bound, silent
   catch-and-ignore error handling that could hide real numerical failures.
6. **Consolidation opportunities** — duplicated logic across sibling files
   that should share one helper, boilerplate that mirrors an existing macro
   (`@use_threads`, `@blas_multi`) but was hand-rolled instead.
7. **Missing or redundant exports** — the same export audit the
   `julia-refactor-scoped` skill already defines, applied file-by-file.

This plan additionally singles out, per the user's explicit request,
**whether every billiard's registered symmetry group actually works** —
i.e. whether `full_boundary`, `SymmetryWall` sector tagging, and the
solver-side `SymmetrySector`/`symmetry_index_orbits` folding are mutually
consistent for each concrete billiard in the catalogue, not just
type-correct in isolation.

## How this plan is meant to be used

Each numbered step file below (`bg-XX-*.md` for `BilliardGeometry.jl`,
`qb-XX-*.md` for `QuantumBilliards.jl`) is a **self-contained work order**
for one audit pass over one mathematically-coherent area, sized to match
that package's folder structure.

**Execution model (confirmed):** steps are run **strictly one at a time, in
order, with only one subagent invocation active at any moment** — never in
parallel. Each step's own "Required background reading" section already
names which prior findings files it depends on; do not start step N+1 until
step N's findings file has been written and (briefly) reviewed, since later
steps are written to consume earlier findings rather than re-derive them.
Each step is its own invocation of the **Julia Refactoring Agent** subagent
(`runSubagent` with `agentName: "Julia Refactoring Agent"`).

**Read-only vs. fixing (confirmed):** every step below is a **read-only**
audit pass — findings only, no source edits. Implementing any fix is a
separate, later, explicitly scoped `julia-refactor-scoped` session per
finding, never folded into the audit step itself.

Every step file has the same sections:
- **Goal** — what this pass is trying to learn.
- **Scope — files to read in full** — the exact file list (absolute paths).
  Read every one of them, plus the abstract-type contract(s) they implement.
- **Required background reading** — pointers into `../migration-plan/` and
  `memories/repo/` that already recorded relevant design decisions or known
  issues for this area, so the audit doesn't re-litigate settled questions
  or re-report an already-tracked deferral as new.
- **Audit checklist** — the 7 categories above, plus area-specific checks.
- **Out of scope for this step** — neighboring files owned by a different
  step, to keep each pass focused.
- **How to invoke** — reminder that the subagent is stateless: paste this
  step file's own Scope/Background/Checklist/Out-of-scope content directly
  into the `runSubagent` prompt (don't just hand it this plan's file path),
  and explicitly tell it this is a user-requested whole-package audit (so it
  uses `julia-refactor-scoped`'s package-wide-audit branch, not the
  single-file default), and that **this is a read-only pass — report
  findings only, do not edit any source file.**
- **Deliverable** — save the subagent's final categorized report verbatim
  to `findings/<step-id>-findings.md` (create `findings/` on first use)
  before starting the next step. This is what makes 16 stateless subagent
  calls compose into one coherent audit: each finding file is short,
  greppable, and can be triaged with the user afterward without re-running
  the pass that produced it.

Fixing anything found is explicitly a **separate, later pass**: run the
whole plan first, collect all findings, then triage with the user (some
"weaknesses" may be intentional design decisions already recorded in
`memories/repo/`) before any code changes are made.

## Step order and rationale

### `BilliardGeometry.jl` (audited first, per user request)

1. [bg-01-core-geometry.md](bg-01-core-geometry.md) — shared geometry
   primitives and abstract types (curves, arclength, curvature, domains'
   common utilities). Everything else in the package builds on this.
2. [bg-02-symmetry-framework.md](bg-02-symmetry-framework.md) — the
   `SymmetryRegistry`/`SymmetryWall`/`SymmetryOrbitMap`/`full_boundary`
   machinery. This is where "does symmetry actually work" lives structurally
   (per-billiard behavioral verification is deferred to step 5).
3. [bg-03-domains-segments.md](bg-03-domains-segments.md) — concrete
   `AbsDomain`/`AbsCurve` implementations (polar/composite/multiply-connected
   domains, line/circle/polar/composite curve segments).
4. [bg-04-quadrature.md](bg-04-quadrature.md) — sampling and Kress-grading
   nodes, the numerical-quadrature layer everything above feeds into.
5. [bg-05-billiards-catalogue.md](bg-05-billiards-catalogue.md) — **audited
   last, as requested**: every concrete billiard constructor, cross-checked
   against steps 1–4's findings. This is the step that answers "which
   billiards' symmetries don't actually work" — see the step file's note on
   why this needs a runtime smoke-check in addition to static review.

### `QuantumBilliards.jl` (audited second, ordered by folder/math structure)

6. [qb-01-abstracttypes-utils.md](qb-01-abstracttypes-utils.md) —
   `abstracttypes.jl`, the top-level module file's `include`/`export` list,
   and `utils/`. The contract every later step's "missing wiring" checks are
   measured against.
7. [qb-02-basis.md](qb-02-basis.md) — `basis/planewaves`,
   `basis/fourierbessel` (the `AbsBasis` concrete types).
8. [qb-03-shared-solver-infrastructure.md](qb-03-shared-solver-infrastructure.md)
   — `solvers/boundarypoints.jl`, `boundarygeomcache.jl`,
   `decompositions.jl`, `matrixconstructors.jl`,
   `sweepmethods/boundarygrading.jl`, `sweepmethods/sweepmethods.jl`: the
   generic infrastructure every concrete solver in steps 9–10 reuses.
9. [qb-04-sweep-bim-solvers.md](qb-04-sweep-bim-solvers.md) — concrete
   `SweepBasisSolver`/`SweepBIMSolver`s: decomposition method, particular
   solutions, DLP, CFIE, composite BIM.
10. [qb-05-accelerated-solvers.md](qb-05-accelerated-solvers.md) — concrete
    `AcceleratedBasisSolver`/`AcceleratedBIMSolver`s: Vergini–Saraceno, Beyn,
    expanded BIM.
11. [qb-06-chebyshev.md](qb-06-chebyshev.md) — the Chebyshev-accelerated
    Hankel/Bessel evaluation backend behind steps 9–10's `use_chebyshev`
    fields (cross-check against the already-completed
    `../migration-plan/10.5-chebyshev-gap-analysis.md` read-only audit —
    confirm which of its 5 flagged gaps are still open).
12. [qb-07-spectra.md](qb-07-spectra.md) — `spectra/spectralutils.jl`,
    `spectra/unfolding.jl`.
13. [qb-08-eigenstates-basisstates.md](qb-08-eigenstates-basisstates.md) —
    `states/eigenstates.jl`, `states/basisstates.jl`.
14. [qb-09-wavefunctions-boundary-husimi.md](qb-09-wavefunctions-boundary-husimi.md)
    — `states/wavefunctions.jl`, `states/boundaryfunctions.jl`,
    `states/husimifunctions.jl`.
15. [qb-10-symmetry-states.md](qb-10-symmetry-states.md) —
    `states/symmetry/symmetrysector.jl`, `states/symmetry/reflections.jl`:
    the `QuantumBilliards.jl`-side half of the symmetry story from step 2,
    closing the loop between geometric symmetry groups and chosen
    representations.
16. [qb-11-cross-package-integration.md](qb-11-cross-package-integration.md)
    — whole-module export/dependency sanity for both packages together, plus
    a read-only check for broken call sites in the sibling plotting packages.
    Deliberately last: only meaningful once steps 1–15 have already surfaced
    the individual gaps it needs to cross-reference.

## Findings storage convention

After each step's subagent run, save its final report to:

```
QuantumBilliardsTests/scratchpad/audit-plan/findings/<step-id>-findings.md
```

e.g. `findings/bg-01-core-geometry-findings.md`. Prefix each file with the
date it was produced. Once all 16 findings files exist, do a final triage
pass with the user before opening any fix-it work (which should itself be
split back into scoped `julia-refactor-scoped` sessions per finding, not one
giant patch).

## Runtime smoke-check scripts (confirmed convention)

Step `bg-05` (and any other step whose checklist requires actually running
Julia code rather than static review, e.g. a per-billiard `full_boundary`
check) must save every smoke-check it runs as a standalone, re-runnable
`.jl` script — never as a throwaway REPL snippet only pasted into the
findings report. Location:

```
QuantumBilliardsTests/scratchpad/audit-plan/smoke-checks/<step-id>-<short-name>.jl
```

e.g. `smoke-checks/bg-05-annular-full-boundary.jl`. Each script should be
self-contained (activate/`using` the right packages, construct the
billiard(s) under test, assert/print the property being checked) so it can
be handed to the Julia Test Writer agent later to fold into a permanent
`@testset` in the relevant package's `test/` suite, rather than being
re-derived from scratch. The findings report should still summarize the
result inline (pass/fail table) and link to the corresponding script file(s)
by relative path.

## Post-audit memory promotion (confirmed)

Once all 16 findings files exist and the final `qb-11` synthesis pass is
done, promote the highest-priority, still-open findings into
`memories/repo/` (one file per package, following the existing
`geometry-audit-10.9.md`/`known-bugs.md` convention already used in
`BilliardGeometry.jl/memories/repo/`) so future sessions don't need to
re-read all 16 findings files to recall the audit's conclusions. This is a
separate, explicit step done after the full triage pass — not automatic
after each individual step.

## Open questions / proposals for the user

Resolved for this run:
1. Read-only audit, fixes separated — **confirmed**.
2. Plotting-package scope in `qb-11` stays grep-level only, no expansion —
   **confirmed**.
3. Execution is strictly sequential, one subagent at a time — **confirmed**.
4. Findings get promoted into `memories/repo/` after the full audit —
   **confirmed**.
5. Runtime smoke-checks are saved as standalone `.jl` scripts in
   `smoke-checks/` for later integration into the real test suites, not
   left as ephemeral snippets — **confirmed**, see convention above.
