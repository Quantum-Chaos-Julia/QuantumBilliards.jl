# Step qb-04: Sweep/BIM & basis solvers (concrete `SweepBasisSolver`/`SweepBIMSolver`s)

## Goal
Audit every concrete sweep-method solver — the two basis-solver-family
methods and the three boundary-integral-method solvers — against the
`julia-add-solver` skill's required-method contract and each other's API
conventions.

## Scope — files to read in full
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/decompositionmethod.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/particularsolutions.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/cfie.jl`
- `/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl`

## Required background reading
- **Load the `julia-add-solver` skill first**
  (`/home/clozej/Programs/QuantumBilliardsTests/.github/skills/julia-add-solver/SKILL.md`)
  — it documents exactly which methods each `AbsSolver` branch gets for
  free via dispatch vs. which four a concrete solver must implement; use it
  as the literal checklist for "missing implementation" per solver here,
  not an ad hoc guess.
- The **`qb-01` and `qb-03` findings files**.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/01-quick-wins-decomposition-psm.md`,
  `03-double-layer-potential-solver.md`, `06-cfie-solver.md`,
  `07-composite-bim-solver.md` — original per-solver design/port plans;
  check each solver still matches its own plan (no drift since).
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/13-compositebim-annulus-verification.md`
  — the `KrylovKit.svdsolve` `ArgumentError` bug fix for
  `CompositeBIMSolver.solve`/`solve_vect`; confirm the fix is intact.
- `/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/geometry-audit-10.9.md`
  Gap 2 — `CompositeBIMSolver` calling the single-ring
  `symmetry_index_orbits` overload on flattened multi-ring points is
  flagged as wrong for a *symmetric* multiply-connected billiard; check
  `compositebim.jl` for this exact call pattern and report current status.
- `/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/16-bim-symmetrysector-migration.md`
  — "plan only, not started": DLP/CFIE/CompositeBIM currently store a bare
  `symmetry::Union{Nothing,AbsSymmetry}` field with no character/
  representation choice (every BIM fold today is silently restricted to
  the trivial, fully-symmetric representation). Confirm this is still
  accurate — this is a **missing implementation** finding, not merely a
  historical note, if still true.

## Audit checklist
1. **Dead code** — unused helper functions/branches in any of the five
   solver files.
2. **Missing wiring** — does each concrete solver actually get
   `solve_wavenumber`/`k_sweep`/`solve_state` for free per `qb-03`'s
   findings, or does any of them override something it shouldn't need to?
3. **Missing implementations** — per `julia-add-solver`'s checklist, does
   every solver implement its required four methods? Specifically confirm
   `CompositeBIMSolver`'s Gap 2 status and the Step 16 `symmetry` field
   limitation above.
4. **Unstable APIs** — compare constructor signatures across all five
   solvers (are `k1,k2,dk`/`N1,N2`-style sweep-range arguments consistent?
   Is the `symmetry`/grading keyword named and defaulted consistently?);
   any field typed `Union{Nothing,AbsSymmetry}` or similar that forces
   runtime branching in a hot path.
5. **Vulnerabilities** — Krylov nullspace/SVD solve paths
   (`smallest_nullvec_krylov!` and friends): any silent non-convergence
   swallowed instead of surfaced; unchecked assumption about matrix
   conditioning for near-degenerate `k`.
6. **Consolidation opportunities** — how much of `dlp.jl`/`cfie.jl`'s
   `construct_matrices`/kernel-assembly structure is duplicated instead of
   shared (the migration plan explicitly designed CFIE as "DLP + one extra
   term" — confirm that's reflected in actual code sharing, not copy-paste)?
7. **Export audit** — per `julia-refactor-scoped`, cross-checked against
   `qb-01`.
8. **Threading/BLAS** — confirm matrix-assembly loops in each solver follow
   the mode's threading conventions (outer parallel over boundary points/
   panels, `@blas_multi` around any nested BLAS call).

## Out of scope for this step
- Accelerated wrappers (`BeynSolver`/`ExpandedBIMSolver`/Vergini–Saraceno)
  → step `qb-05`.
- Chebyshev-accelerated kernel evaluation → `qb-06`.
- `BIMEigenstate`/`solve_state` output consumption
  (`wavefunction`/`boundary_function`/`husimi_function`) → `qb-09`.

## How to invoke
Use `runSubagent` with `agentName: "Julia Refactoring Agent"`. Tell it to
load the `julia-add-solver` skill first, then paste this file's
Scope/Required background/Checklist/Out-of-scope sections into the prompt.
State this is a whole-package audit step (package-wide branch of
`julia-refactor-scoped`), read-only: report only, no edits.

## Deliverable
Save the subagent's categorized report verbatim to
`QuantumBilliardsTests/scratchpad/audit-plan/findings/qb-04-sweep-bim-solvers-findings.md`.
