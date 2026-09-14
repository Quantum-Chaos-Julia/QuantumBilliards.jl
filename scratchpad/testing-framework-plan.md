# Comprehensive solver/billiard testing framework — plan

Status: **planning + templates**. This document is the design deliverable;
`reference/generate_reference_spectra.jl` and `reference/reference_spectra.jl`
implement a *working, generic* harness and a small real (not fabricated)
subset of reference values validated by actually running the solvers.
[spectrumtests_template.jl](../spectrumtests_template.jl) and
[plottingtests_comprehensive.jl](../plottingtests_comprehensive.jl) are
runnable templates that consume the reference file and the same case tables.
Filling in the remaining cells of the matrix below (mostly the expensive
`k≈1000` accelerated-solver cases) is left as follow-up execution work, not
done in this pass — see §7.

---

## 1. Goal

Build test coverage that, across the whole solver/billiard matrix:

* Exercises **every solver family and variant** (sweep vs. accelerated,
  basis-expansion vs. boundary-integral, every grading/kernel/symmetry
  option) at least once.
* Exercises **every billiard symmetry class** already wired into
  `BilliardGeometry.jl`/`QuantumBilliards.jl` fixtures, reusing the existing
  `StadiumBilliard`/veech `TriangleBilliard`/`PolarBilliard`(circle) fixtures
  rather than inventing new ones.
* Deliberately **overlaps**: each billiard is solved by every solver that can
  legally act on it, so a regression in one solver is caught by every
  billiard fixture it shares with other solvers, and a regression in one
  billiard's geometry/quadrature is caught by every solver family that uses
  it.
* Respects **solver applicability limits** (basis-expansion solvers need a
  convex/star-shaped domain for the plane-wave/Fourier-Bessel expansion to
  converge; BIM solvers need a well-defined boundary parametrization but no
  convexity) and **known open bugs** (flagged, not silently worked around).

Three deliverables, mirroring the user's request://
1. A **reference-value generator script** that runs the real solvers and
   writes named `const` wavenumber vectors to a plain Julia file (no
   hand-typed/guessed numbers — regenerate by re-running the script).
2. A **spectrum-test template** (`Test.jl`, `@testset` per
   solver/billiard/variant) that reads those constants and checks
   `solve_wavenumber`/`solve_spectrum`/`compute_spectrum`-family results
   against them.
3. A **plotting-test template**, generalizing the existing
   [plottingtests.jl](../plottingtests.jl) one-off blocks into a loop over
   the same case tables, keeping the exact figure signage
   (`save_figure_basis`/`save_figure_bim`, title with `k`, tension `t`,
   solver name, `b`/`d` scaling factors) already established there.

---

## 2. Solver inventory (from `abstracttypes.jl` + `src/solvers/`)

```
AbsSolver
├── AbsBasisSolver            (needs basis::AbsBasis + billiard)
│   ├── SweepBasisSolver         — tension(k) scan, no solve_spectrum
│   │     • DecompositionMethodSolver          (solvers/sweepmethods/decompositionmethod.jl)
│   │     • ParticularSolutionsMethod  (PSM)    (solvers/sweepmethods/particularsolutions.jl)
│   └── AcceleratedBasisSolver    — one diagonalization → many roots in a window
│         • VerginiSaracenoSolver (VS)          (solvers/acceleratedmethods/verginisaraceno.jl)
└── AbsBIMSolver               (no basis; unknowns are boundary densities)
    ├── SweepBIMSolver            — tension(k) scan via smallest singular value
    │     • DoubleLayerPotentialSolver (DLP)     (solvers/sweepmethods/dlp.jl)
    │     • CombinedFieldIntegralEquationSolver (CFIE) (solvers/sweepmethods/cfie.jl)
    │     • CompositeBIMSolver (wraps a Tuple of the above, per-component)  (solvers/sweepmethods/compositebim.jl)
    └── AcceleratedBIMSolver      — nonlinear eigenproblem A(k)v=0 near target k
          • BeynSolver (wraps kernel::SweepBIMSolver)   (solvers/acceleratedmethods/beyn.jl)
          • ExpandedBIMSolver (wraps kernel::SweepBIMSolver) (solvers/acceleratedmethods/ebim.jl)
```

Key API asymmetry that drives the test harness design (verified by grep,
not assumed):

* `solve_spectrum` (many roots near one `k0` from a single call) is **only**
  defined for `AcceleratedBasisSolver`/`BeynSolver`/`ExpandedBIMSolver`. It
  does **not** exist for `SweepBasisSolver`/`SweepBIMSolver`.
* `compute_spectrum(solver::AbsBasisSolver, basis, billiard, k1, k2, dk)`
  (spectra/spectralutils.jl) internally calls `solve_spectrum`, so it is only
  actually usable for `AcceleratedBasisSolver` today, despite its type
  annotation reading `AbsBasisSolver` — **do not use it for
  `DecompositionMethodSolver`/`ParticularSolutionsMethod`**, it will
  `MethodError`.
* Every `SweepBasisSolver`/`SweepBIMSolver` **does** expose `k_sweep`
  (tension across an explicit `ks` grid) and `solve_wavenumber` (single
  local refinement around one `k0`). The harness gets "N states near k0" for
  sweep solvers by scanning `k_sweep` for tension-curve local minima and
  refining each with `solve_wavenumber` — see §5.

## 3. Basis families

* `RealPlaneWaves(dim; sym_x, sym_y, ...)` — plane waves, any symmetry
  sector via `sym_x,sym_y ∈ {nothing,±1}` (4 combinations → `D2` sectors).
  Requires a domain the plane-wave ansatz converges on (convex/star-shaped
  about the origin).
* `CornerAdaptedFourierBessel(dim, corner_angle, cs, symmetry)` — built by
  `QuantumBilliards.make_triangle_and_basis`/`make_veech_right_triangle_and_basis`
  for `TriangleBilliard`; no symmetry reduction used for the veech triangle
  fixture (`symmetries` field is empty on `TriangleBilliard`).

## 4. Billiard fixtures (reused, not new)

| Billiard | Constructor | Symmetry classes available | Corners? | Grading for BIM |
|---|---|---|---|---|
| Circle | `BilliardGeometry.PolarBilliard([0,0])` (BIM tests) / `CircleBilliard()` (basis tests, D2 quadrant fundamental domain) | full `D2` (`XAxisReflection`,`YAxisReflection`,`XYAxisReflection`) via `RealPlaneWaves(sym_x,sym_y)` | none (smooth) | `SmoothPeriodicGrading()` |
| Stadium | `StadiumBilliard(half_width)` | `D2`, same 4 `RealPlaneWaves(sym_x,sym_y)` sectors as circle (existing tests already cover `(-1,-1)`; this plan adds the other 3) | none (boundary is `C¹`, straight/arc junctions are tangent-continuous) | `GlobalCornerGrading()` (auto-falls back to smooth if no true corners detected — safe default for any composite boundary) |
| Veech right triangle | `QuantumBilliards.make_veech_right_triangle(n)` / `..._and_basis(n)`, `n=5` used throughout (matches existing `solvertests.jl`) | none (`TriangleBilliard.symmetries` is empty) | 3 true corners | `GlobalCornerGrading()` (required — has genuine corners) |

All three are **convex** (a capsule/stadium shape is convex despite chaotic
dynamics — the defocusing mechanism comes from the circular arcs, not from
non-convexity), so all three are valid for basis-expansion solvers too. This
is exactly why the user's prompt calls out Stadium/Triangle as "good" basis
fixtures. Circle is additionally the only fixture with a fully smooth
(corner-free) boundary, so it is the natural fixture for the
`SmoothPeriodicGrading` BIM path, while Triangle is the natural fixture for
`GlobalCornerGrading`. Stadium sits in between (no true corners, but a
composite/multi-segment boundary) and is included in the BIM matrix mainly
for the symmetry-reduced case — which is currently blocked, see §6.

No non-convex or multiply-connected fixture is added to the main coverage
matrix — `AnnularBilliard` exists but `CompositeBIMSolver.solve_vect` is
known-broken on it (and even on simply-connected `CircleBilliard`), see §6.

**Newly-discovered limitation (found while building
`plottingtests_comprehensive.jl`, 2026-09-10)**: plotting (not solving) a
basis-solver ground state on `CircleBilliard` — `plot_state_tests!` →
`plot_boundary_function!` → `boundary_function` →
`apply_symmetries_to_boundary_points` — raises a `MethodError` constructing
`BoundaryPoints` (argument-count mismatch against the current
`BoundaryPoints{T}` constructor in
`QuantumBilliards.jl/src/solvers/boundarypoints.jl`). `StadiumBilliard`
(also `D2`-symmetric) does **not** hit this, so it is specific to
`CircleBilliard`'s fixture, not to basis-solver+symmetry plotting in
general. Solving (not plotting) ground states on `CircleBilliard` with basis
solvers works fine (verified in `spectrumtests_template.jl`). This is a real
package bug, out of scope to fix here — flagged for the "Julia Refactoring
Agent"/package maintainers; `plottingtests_comprehensive.jl` accordingly
excludes Circle from its basis-solver plotting cases (BIM-solver plotting on
Circle is unaffected and included).

## 5. Wavenumber targets and how each is obtained

| Regime | Applies to | Target | Mechanism |
|---|---|---|---|
| Ground state | every applicable solver × {Circle, Veech Triangle} | 1 state, `k0` seeded at the known first zero (`≈2.405` circle / `≈6.065` triangle, matching existing `solvertests.jl` seeds) | `solve_wavenumber(solver,[basis,]billiard,k0,dk)` directly |
| Low sweep spectrum | `SweepBasisSolver`, `SweepBIMSolver` (Decomposition, PSM, DLP, CFIE) | ~5 states near `k≈20` | `sweep_states_near` helper (§5.1): `k_sweep` over a grid around `k0`, local-minima detection, `solve_wavenumber` refinement of each candidate, adaptive window widening until ≥5 found |
| Mid accelerated spectrum | `AcceleratedBasisSolver` (VS), `AcceleratedBIMSolver` (Beyn, EBIM) | ~10 states near `k≈100` | `accelerated_states_near` helper (§5.2): `compute_spectrum(solver,...,k1,k2,dk)` (walks the search range in small, correctness-sized windows and merges overlaps — a single `solve_spectrum(solver,...,k0,dk)` diagonalization is *not* trusted to resolve every state across a widened window) with adaptive search-range widening until ≥10 found, keep the 10 closest to `k0` |
| High accelerated spectrum | same accelerated solvers | ~20 states near `k≈1000` | same helper, larger starting `half_width0` (accelerated methods are the whole point of testing at this regime — sweep solvers are not required to reach `k≈1000`, consistent with the user's split) |

Both helpers are **adaptive**, not pre-computed from Weyl's law: rather than
guessing a window width from `N(k) ≈ Area·k²/4π − Perimeter·k/4π` (no
`area`/`perimeter` helper currently exists in `BilliardGeometry.jl` to plug
into that formula anyway), the harness starts from a modest window and
geometrically grows it until the target count is reached, then trims to
exactly the target count closest to `k0`. This is robust to the fact that
mean level spacing differs enormously between billiards/regimes and avoids
hand-tuning a window per case.

### 5.1 `sweep_states_near` (sweep solvers)

```julia
function sweep_states_near(k_sweep_fn, solve_wavenumber_fn; k0, n_target,
                            half_width0=1.0, npts_per_unit=200, growth=1.6,
                            refine_dk=0.02, max_iter=12)
    # k_sweep_fn(ks::Vector) -> tens::Vector
    # solve_wavenumber_fn(k0,dk) -> (k,t1)
    ...
end
```

### 5.2 `accelerated_states_near` (accelerated solvers)

```julia
function accelerated_states_near(compute_spectrum_fn; k0, n_target,
                                  half_width0=0.5, growth=1.7, max_iter=14)
    # compute_spectrum_fn(k1,k2) -> (ks::Vector, tens::Vector), a closure over
    # compute_spectrum(solver,...,k1,k2,dk) with dk fixed at a
    # correctness-sized window (NOT a single solve_spectrum(solver,...,k0,dk)
    # diagonalization, which only reliably resolves states near the center
    # of its own window and cannot be trusted across a widened one)
    ...
end
```

Both are implemented, generically, in
[reference/generate_reference_spectra.jl](../reference/generate_reference_spectra.jl)
and reused unchanged by the spectrum-test template (import the file or copy
the two functions — see that file's header comment).

## 6. Applicability matrix / known limitations

Legend: ✅ covered by templates & (partially) validated reference values,
🧩 planned but not yet executed (expensive — left for a follow-up run),
🚫 blocked by a known bug (do not silently skip — flagged with a
`@test_broken`/comment pointing at the bug, so a fix shows up as an
unexpected pass).

| Solver | Circle (ground) | Circle (sweep k≈20 / accel k≈100,1000) | Triangle (ground) | Triangle (sweep/accel) | Stadium (any sector) |
|---|---|---|---|---|---|
| DecompositionMethodSolver (+ RealPlaneWaves / CAFB) | ✅ | 🧩 sweep k≈20 | ✅ (existing test) | 🧩 sweep k≈20 | 🧩 4 `D2` sectors, 1 already covered by nothing (new) |
| ParticularSolutionsMethod | ✅ | 🧩 | ✅ (existing test) | 🧩 | 🧩 |
| VerginiSaracenoSolver | ✅ | 🧩 accel k≈100/1000 | ✅ (existing test) | ✅ (existing k≈110/2010 tests double as the k≈100/1000 regime) | 🧩 (existing tests cover only `(-1,-1)`; add `(1,1),(1,-1),(-1,1)`) |
| DoubleLayerPotentialSolver, `SmoothPeriodicGrading` | ✅ (existing test) | 🧩 sweep k≈20 | n/a (has corners, wrong grading) | n/a | n/a (no true corners, but composite boundary — use `GlobalCornerGrading` instead) |
| DoubleLayerPotentialSolver, `GlobalCornerGrading` | n/a (no corners) | n/a | ✅ (existing test) | 🧩 sweep k≈20 | 🚫 symmetry-reduced case: `construct_matrices` fails its own algebraic-consistency test (`solvertests.jl` "Double Layer Potential - Stadium (YAxisReflection)"). Unreduced (no `symmetry=`) full-boundary case is 🧩. |
| CombinedFieldIntegralEquationSolver, `SmoothPeriodicGrading` | ✅ (existing test) | 🧩 | n/a | n/a | n/a |
| CombinedFieldIntegralEquationSolver, `GlobalCornerGrading` | n/a | n/a | ✅ (existing test) | 🧩 | 🚫 same symmetry-reduction bug class as DLP (not yet independently re-verified for CFIE — treat as 🚫 until checked) |
| CompositeBIMSolver | ✅ single-component DLP kernel on `PolarBilliard` (single closed curve) works and matches plain `DoubleLayerPotentialSolver` exactly — `bim-solver-notes.md`'s reported `KrylovKit.svdsolve ArgumentError` was **not reproduced** in this configuration. 🚫 still applies to `CircleBilliard` (D2 quadrant fundamental domain) and `AnnularBilliard` (multiply-connected) per that note — not independently re-verified here, do not assume fixed for those | 🚫 (multi-component/`AnnularBilliard` case unverified) | n/a (has corners; would need `GlobalCornerGrading`, untested) | 🚫 | 🚫 |
| BeynSolver(kernel=DLP) | ✅ (existing test) | 🧩 accel k≈100/1000 | ✅ (existing test) | 🧩 accel k≈100/1000 | 🚫 (inherits DLP symmetry-reduction bug if a symmetry kernel is used; unreduced full-boundary case is 🧩) |
| BeynSolver(kernel=CFIE) | ✅ (existing test) | 🧩 | ✅ (existing test) | 🧩 | 🧩 (unreduced only) |
| ExpandedBIMSolver(kernel=DLP) | 🧩 (ground state not yet in `solvertests.jl`; NaN-selection bug already fixed per `bim-solver-notes.md`) | 🚫 wide-window tuning is known messy/poorly-clustered with default keywords (`bim-solver-notes.md`, Step 11 note) — use `compute_spectrum` narrow-window mode for k≈100, treat the k≈1000 case as 🧩/best-effort only | 🧩 | same 🚫 caveat | 🧩 |
| ExpandedBIMSolver(kernel=CFIE) | 🧩 | same 🚫 caveat | 🧩 | same 🚫 caveat | 🧩 |

Practical read of this table: the matrix is intentionally **not** fully
executed in this pass (many 🧩 cells, especially every `k≈1000`
accelerated-solver case, are individually expensive — minutes each). What
*is* delivered and validated: the harness functions (§5.1/§5.2) actually run
against the real solvers (see `reference/reference_spectra.jl`'s populated
constants and the "validated" note in its header), the full case tables are
written out (with an `enabled=false`/commented flag for the 🧩 cells) so
running the generator script end-to-end fills in the rest, and every 🚫 cell
is a deliberate, documented skip (a `@test_skip`/comment in the spectrum
template referencing this file), not a silent gap.

## 7. File layout

```
QuantumBilliardsTests/
├── scratchpad/testing-framework-plan.md      (this file)
├── reference/
│   ├── spectrum_harness.jl                   (shared adaptive-search helpers; included by both files below)
│   ├── generate_reference_spectra.jl         (case tables; run to (re)populate the file below)
│   └── reference_spectra.jl                  (generated: named `const` Float64/Vector{Float64} per case)
├── spectrumtests_template.jl                 (Test.jl testset template, one @testset per matrix cell, reads reference_spectra.jl; 43/43 pass for the populated cells)
└── plottingtests_comprehensive.jl            (loops case tables through save_figure_basis/save_figure_bim; 11/11 cases ran clean)
```

Naming convention used in `reference_spectra.jl` (mirrors `solvertests.jl`'s
existing `k_test`/`ks_test`/`tens_test` local-variable convention, but as
file-level constants so they're shared across the spectrum-test template and
future ad-hoc scripts):

* Ground state: `K_GROUND_<BILLIARD>_<SOLVER>[_<VARIANT>]`,
  `T_GROUND_<BILLIARD>_<SOLVER>[_<VARIANT>]`.
* Windowed spectrum: `KS_<BILLIARD>_<SOLVER>[_<VARIANT>]_K<TARGET>`,
  `TENS_<BILLIARD>_<SOLVER>[_<VARIANT>]_K<TARGET>` (`<TARGET>` ∈
  `20`,`100`,`1000`).

`<BILLIARD>` ∈ `CIRCLE`,`TRIANGLE`,`STADIUM_PP`,`STADIUM_PM`,`STADIUM_MP`,
`STADIUM_MM` (`P`/`M` = `sym_x`/`sym_y` sign, matching `plottingtests.jl`'s
`sym_string`). `<VARIANT>` encodes grading/kernel where relevant, e.g.
`_SMOOTH`, `_GLOBALCORNER`, `_DLPKERNEL`, `_CFIEKERNEL`.

## 8. Execution checklist (follow-up, not all done here)

1. Run `reference/generate_reference_spectra.jl` case-by-case (it prints
   progress per case) to fill in the remaining 🧩 cells; commit the
   regenerated `reference_spectra.jl`.
2. Flesh out `spectrumtests_template.jl` with one `@testset` per newly
   populated constant (copy the ground-state testsets already present —
   they're fully wired against the constants generated in this pass).
3. Flesh out `plottingtests_comprehensive.jl`'s case tables to match.
4. Re-check the two 🚫 rows marked "not yet independently re-verified" (CFIE
   symmetry reduction) against `bim-solver-notes.md`/`known-bugs.md` before
   assuming they're still broken — bugs may have been fixed since.
5. Hand off to the "Julia Test Writer" subagent to turn
   `spectrumtests_template.jl` into a real, CI-integrated `Test.jl` suite
   once the reference file is fully populated (per this workspace's testing
   policy: solver/billiard-level regression tests are written by that
   subagent, not inline).
