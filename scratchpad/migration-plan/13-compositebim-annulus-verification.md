# Step 13 — `CompositeBIMSolver` full numerical verification (deferred from Steps 7/11)

## Status update (2026-09-13) — root cause reconciled, plan refined

This step's original bug hypothesis (a `KrylovKit.svdsolve`/`@blas_1` call-site
defect local to `compositebim.jl`) **does not hold up** under fresh
reproduction and is superseded by the findings below. Two memory sources
disagreed on how broadly the bug reproduces:

* `QuantumBilliards.jl/memories/repo/bim-solver-notes.md` (Step 11, original
  report): claimed the `ArgumentError` reproduces even for a **single**
  component wrapping a simply-connected billiard (`CircleBilliard`).
* `QuantumBilliardsTests/scratchpad/testing-framework-plan.md` §6 /
  `reference/spectrum_cases.jl` (later, independent re-verification): found
  single-component `CompositeBIMSolver` works fine and matches plain
  `DoubleLayerPotentialSolver` exactly on several full-boundary, symmetry-free
  `PolarBilliard`-style fixtures (circle/ellipse/limacon/C3/star — all now
  `enabled=true` cases), and explicitly could **not** reproduce the bug in
  that configuration. Only `CircleBilliard`/`AnnularBilliard` were flagged as
  "still unverified, not fixed".

Direct REPL reproduction (this session) resolves the disagreement in favor of
the second source, and goes further to pin down the exact defect:

1. `CompositeBIMSolver(DoubleLayerPotentialSolver(...))` (single component,
   no hole) on `BilliardGeometry.CircleBilliard(1.0)` (the D2 quadrant
   fundamental-domain fixture) — **no error**. `evaluate_points` returns 200
   points (from the open quarter-arc, artificially closed — a pre-existing,
   unrelated fixture-choice issue per `bim-solver-notes.md`'s own "Important
   gotcha" section), `construct_matrices`/`solve_vect` run fine
   (non-convergent tension ≈0.95, meaningless because the quarter-arc is not
   a real closed boundary, but **not a crash**).
2. `CompositeBIMSolver(DoubleLayerPotentialSolver(...))` on
   `BilliardGeometry.PolarBilliard([0,0])` (full, un-reduced circle, no
   symmetry) — **no error**, tension ≈0.01 at `k=2.4`, consistent with
   `spectrum_cases.jl`'s `CIRCLE_COMPOSITE_DLP` case.
3. `CompositeBIMSolver(DoubleLayerPotentialSolver(...), DoubleLayerPotentialSolver(...))`
   (two components) on `AnnularBilliard(2.0, 1.0)` — `construct_matrices`
   returns a `400×400` matrix containing **`NaN` in ~99.5% of the 200×200
   diagonal block belonging to the inner (hole) component**; `svdvals(A)`
   (not just `KrylovKit.svdsolve`) then raises `ArgumentError: invalid
   argument #4 to LAPACK call` on the NaN-poisoned matrix. `KrylovKit`'s
   "operator and its adjoint are not compatible" error (the original report)
   is the same NaN-poisoning manifesting through a different code path in
   `KrylovKit`'s Lanczos iteration — **a downstream symptom, not the root
   defect.** The `solve`/`solve_vect`/`@blas_1`/`KrylovKit.svdsolve` call
   sites in `compositebim.jl` are confirmed byte-for-byte structurally
   identical to `dlp.jl`'s/`cfie.jl`'s working calls (diffed directly) — **no
   fix is needed there.**

### Root cause, fully isolated and confirmed

`CompositeBIMSolver.evaluate_points` (compositebim.jl) reverses each hole's
curves via `[BilliardGeometry._reverse_curve(c) for c in reverse(groups[a])]`
for every component `a > 1`. For `AnnularBilliard`, the inner boundary is a
single `CircleSegment(radius=1.0, arc_angle=2π, shift_angle=0, ...)`.

`BilliardGeometry._reverse_curve(c::CircleSegment)` computes
`new_arc = -c.arc_angle`. This is correct and safe at its **only other**
call site — `fullboundary.jl`'s `full_boundary(billiard)`, which always pairs
it with a preceding `_apply_symmetry_to_curve(sym, c)` for an
orientation-reversing reflection (which itself already negates `arc_angle`),
so the *net* arc angle after `reverse∘reflect` ends up positive again
(confirmed: `full_boundary(StadiumBilliard(0.5))`'s 4 reflection-image
`CircleSegment`s all have strictly positive `.length`, e.g. `π/2`, not
negative — this is a different, already-tracked, still-open bug, see below).

But `CompositeBIMSolver` calls `_reverse_curve` **standalone**, directly on a
curve with a genuinely *positive* original `arc_angle` (no prior reflection).
This flips `arc_angle` to **negative** (`-2π` for the annulus's inner
circle). `CircleSegment`'s convenience constructors
(`BilliardGeometry.jl/src/geometry/segments/circlesegment.jl`, both the
`(R,arc_angle,shift_angle,center)` and `(R,arc_angle,shift_angle,x0,y0)`
kwargs forms) compute the `.length` field as the raw signed product
`L = R*arc_angle`, **without `abs`** — so the reversed hole curve ends up
with `.length == -2π` (confirmed directly in the REPL).

`BilliardGeometry.component_lengths`/`_global_t_to_segment_u`
(`geometry/boundarycomponents.jl`), used by `_dlp_evaluate_points`/
`_cfie_evaluate_points` (via `_composite_component_points`) to map a global
periodic parameter `t∈[0,2π)` to a local curve parameter, contain the guard
`s >= Ltot && return 1, zero(T)`. With `Ltot = -2π` (negative), `s =
(t/2π)*Ltot` is negative for every `t∈(0,2π)`, so `s >= Ltot` is true for
**almost every** `t` — the entire 200-node discretization of the reversed
hole collapses to the single degenerate point `u=0` (confirmed: all 200
inner-boundary points sample to the identical `xy=[1.0, ≈0]` regardless of
their spread-out `ts`/`tphys` values). This degenerate point cloud produces
near-zero pairwise distances `r` in the same-component double-layer kernel
entries (`invr=inv(r)`, `hankelh1(1,k*r)`), poisoning that block with `NaN`,
which then poisons the SVD/Krylov solve downstream.

**Confirmed fix (simulated in the REPL, not yet applied):** manually
rebuilding the reversed `CircleSegment` with `length = abs(radius*arc_angle)`
(instead of the signed `radius*arc_angle`) makes `_global_t_to_segment_u`
correctly walk `u` across the full `[0,1]` range as `t` sweeps `[0,2π)`,
producing a properly-spread, non-degenerate discretization of the reversed
circle (verified: 5 sample points at `t=0,1,3,5,6` now land at 5 genuinely
different `xy` locations on the unit circle, not one repeated point).
`FourierCoeffPolarSegment`/`PolarSegment`/`LimaconSegment` are **not**
affected — their `.length` is computed via `arc_length(crv, one(T))`, a
`quadgk` integral of `norm(ForwardDiff.derivative(...))`, which is inherently
non-negative regardless of `arc_angle`'s sign. `LineSegment`'s `.length` is a
Euclidean distance, also sign-safe. `grep`-confirmed `R*arc_angle` (the
buggy shortcut) appears in exactly the two `CircleSegment` constructors and
nowhere else in `BilliardGeometry.jl`; `.length` itself is only ever consumed
as a magnitude (perimeter/arc-length accumulation in `area.jl`,
`compositecurves.jl`, `component_lengths`) — no call site relies on a signed
value, so taking `abs` is a safe, non-breaking correction.

### Reconciling the two contradictory memory notes

`bim-solver-notes.md`'s original claim ("reproduces even for a single
component on `CircleBilliard`") does not reproduce under fresh testing — the
single-component path never calls `_reverse_curve` at all (only
`groups[2:end]` are reversed), so it cannot hit this bug. Treat that specific
claim in `bim-solver-notes.md` as superseded by this session's findings; the
`testing-framework-plan.md`/`spectrum_cases.jl` characterization ("bug
specific to `AnnularBilliard`/multiply-connected geometries, not a generic
composite/single-component defect") is the one confirmed correct.

### Not the same bug as the pre-existing Stadium symmetry failure

The already-documented, separately-tracked "Double Layer Potential - Stadium
(YAxisReflection) - symmetry-reduced matrix consistency" failure
(`solvertests.jl`, noted as pre-existing in Steps 11/12/12.5) is **not**
caused by this defect — `full_boundary(StadiumBilliard(0.5))`'s
reflection-image `CircleSegment`s all have correctly positive lengths (the
paired reflect+reverse composition already cancels the sign issue, as
designed). That failure has a different, still-unidentified cause and
remains explicitly out of scope for this step.

### Design refinement adopted (2026-09-13): hole detection by `orientation`, not by component position

Independent of the length bug, `CompositeBIMSolver.evaluate_points`'s
original rule for deciding which components to reverse was purely
positional (`a == 1 ? outer : hole`). This is fragile/incorrect in general:
two genuinely disjoint, independent outer boundaries (e.g. Step 7's
two-copies-of-Triangle/Stadium structural smoke test) would have their
second component wrongly reversed even though neither is a hole. The fix
adopted instead: use each curve's own `orientation` field (already the
exact marker `BilliardGeometry.MultiplyConnectedDomain`/`AnnularBilliard`
use for hole semantics in `is_inside`: `orientation=1` outer,
`orientation=-1` hole) as the sole signal for whether a connected component
is reversed, regardless of its position in `groups`. A new private helper,
`_group_is_hole(group)`, reads `group[1].orientation`, asserts every curve in
the component shares it (mixed-orientation components are rejected), and
returns `true`/`false` for `orientation ∈ {-1, 1}` (erroring on any other
value). `evaluate_points` now reverses component `a` iff
`_group_is_hole(groups[a])`, not iff `a > 1`. This is a strictly more
correct generalization — verified no behavior change for `AnnularBilliard`
(inner circle still has `orientation=-1`, still reversed) and for every
existing single-component case (`orientation=1`, never reversed), while now
also correctly leaving a genuine second *outer* boundary (e.g. two disjoint
circles, both `orientation=1`) unreversed — confirmed via a new ad hoc
`TwoDiskBilliard` fixture in this session's REPL verification (see below).

## Goal

Fix the reproducible boundary-degeneracy bug (a `CircleSegment.length`
sign defect in `BilliardGeometry.jl`, not a `QuantumBilliards.jl`
solver/Krylov bug), and replace `CompositeBIMSolver`'s positional
outer/hole assumption with an explicit `orientation`-based convention, then
validate `CompositeBIMSolver`'s actual eigenvalues against a real
multiply-connected billiard with a known analytic spectrum (the annulus).

## Preconditions

Step 12 complete (`AnnularBilliard` now built on `MultiplyConnectedDomain`,
structurally re-verified to be unchanged from Step 11's working geometry).

## Files touched (corrected from the original plan)

* **`BilliardGeometry.jl/src/geometry/segments/circlesegment.jl`** (NEW —
  the original plan did not anticipate touching `BilliardGeometry.jl` at
  all): fix both convenience constructors'
  `L = R*arc_angle` → `L = R*abs(arc_angle)` (or equivalently
  `L = abs(R*arc_angle)`; `R` is always positive so either form is
  equivalent, but prefer whichever reads more clearly next to the existing
  code).
* **`QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl`**: `solve`/
  `solve_vect`'s `KrylovKit.svdsolve` calls needed **no change** (confirmed
  identical in structure to `dlp.jl`/`cfie.jl`). `evaluate_points` DID need a
  change, but not a bug fix in the strict sense — a new private helper
  `_group_is_hole(group)` replaces the positional `a == 1` outer/hole
  assumption with a check of each component's curves' own `orientation`
  field, per the design refinement above. The docstring was updated to
  match.

## Source index

* `QuantumBilliards.jl/memories/repo/bim-solver-notes.md` — original bug
  report (Step 11 finding, now partially superseded — see reconciliation
  above).
* `QuantumBilliardsTests/scratchpad/testing-framework-plan.md` §4/§6 and
  `QuantumBilliardsTests/reference/spectrum_cases.jl` — the later,
  independent re-verification that narrowed the bug to the multiply-connected
  case and supplied several already-passing single-component
  `CompositeBIMSolver` regression fixtures (`CIRCLE_COMPOSITE_DLP`,
  `ELLIPSE_COMPOSITE_DLP`, `LIMACON_COMPOSITE_DLP`, `C3_COMPOSITE_DLP`,
  `STAR_COMPOSITE_DLP`) — reuse these as regression fixtures confirming the
  fix introduces no regression in the already-working single-component path.
* [BilliardGeometry.jl/src/geometry/segments/circlesegment.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/segments/circlesegment.jl) —
  the actual defect (`L = R*arc_angle` in both convenience constructors,
  lines ~21-23 and ~27-29).
* [BilliardGeometry.jl/src/geometry/fullboundary.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/fullboundary.jl) —
  `_reverse_curve(c::CircleSegment)` (the function whose standalone use
  exposes the defect) and its only other, correctly-working call site
  (`full_boundary`'s reflect+reverse composition) — read this to confirm the
  fix doesn't change that path's behavior (it doesn't: `abs` is a no-op
  wherever the net arc angle is already positive).
* [BilliardGeometry.jl/src/geometry/boundarycomponents.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarycomponents.jl) —
  `component_lengths`/`_global_t_to_segment_u`/`_eval_composite_geom_global_t`
  (the consumers whose `s >= Ltot` guard breaks on negative `Ltot`) — no
  change needed here; confirm after the constructor fix that a positive
  `Ltot` restores correct behavior (already verified via direct REPL
  simulation, see above).
* [solvers/sweepmethods/compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl) —
  `evaluate_points`'s hole-reversal call site and `solve`/`solve_vect`
  (confirmed not the bug; keep as read-only reference during verification).

## Implementation steps — ALL DONE (2026-09-13)

1. **Applied the length fix.** In `circlesegment.jl`, changed
   `L = R*arc_angle` to `L = R*abs(arc_angle)` in both
   `CircleSegment(R, arc_angle, shift_angle, center; ...)` and
   `CircleSegment(R, arc_angle, shift_angle, x0, y0; ...)`.
2. **Applied the orientation-based hole convention.** In
   `compositebim.jl`, added `_group_is_hole(group)` and changed
   `evaluate_points` to reverse component `a` iff
   `_group_is_hole(groups[a])` instead of iff `a > 1`; updated the
   docstring to match.
3. **Re-ran the reproduction cases** — single component on
   `PolarBilliard` circle (tension ≈0.0100 at k=2.4, unchanged), two
   components on `AnnularBilliard(2.0,1.0)` (`any(isnan,A)==false`,
   `solve_vect` returns a finite tension ≈0.00604 at k=10), and a new
   ad hoc two-disjoint-circles `TwoDiskBilliard` fixture (both
   `orientation=1`, confirmed neither gets reversed, no NaN, finite
   tension) — all pass.
4. **No regression** confirmed: `get_errors` clean on both changed files;
   the single-component cases are numerically unchanged (same tension
   values as before the fix, since `abs`/`_group_is_hole` are no-ops for
   `orientation=1`, non-reversed components).
5. **Numerical verification against the annulus's known spectrum —
   PASSED.** Annulus ($R_1=1$, $R_2=2$) analytic root equation
   $$J_m(kR_1)\,Y_m(kR_2) - J_m(kR_2)\,Y_m(kR_1) = 0$$
   bisected for $m=0$: $k_0 = 3.1230309195852897$.
   `CompositeBIMSolver(DoubleLayerPotentialSolver(8.0),
   DoubleLayerPotentialSolver(8.0))` on `AnnularBilliard(2.0,1.0)` swept
   over $k\in[k_0-0.3,k_0+0.3]$ shows a clean tension minimum exactly at
   $k=3.1230309195852897$ with tension $\approx 9.6\times10^{-12}$ — an
   essentially exact match (BIM discretization/floating-point level
   agreement, not just "close").
6. Step 7's/Step 11's earlier structural checks are unaffected (same
   `evaluate_points`/`construct_matrices` dimensions/`domain_id` grouping;
   only the reversal *decision* changed, not the grouping algorithm).
7. Step 10.9's "Gap 2" (multicomponent symmetric `symmetry_index_orbits`)
   remains explicitly out of scope — not touched, `AnnularBilliard` still
   has no symmetry.

## Performance & fidelity notes

Both changes are small, non-hot-loop corrections: a sign-only struct-field
computation in `BilliardGeometry.jl`'s `CircleSegment` constructors, and a
O(component-size) orientation-consistency check (`_group_is_hole`) run once
per component in `evaluate_points` (not inside any per-point/O(N²) hot
loop). No performance-sensitive code changed.

## Tests & user verification

1. `get_errors` on both `BilliardGeometry.jl` and `QuantumBilliards.jl` —
   clean, no errors.
2. The reproduction cases (single component on `PolarBilliard` circle, two
   components on `AnnularBilliard`, two disjoint independent circles) all
   succeed (no `NaN`, no `ArgumentError`, finite tension).
3. `CompositeBIMSolver` wrapping `AnnularBilliard`'s two components
   reproduces the annulus's analytic $m=0$ root to within
   $\approx 10^{-11}$ tension at the exact analytic $k$ — see step 5 above.
4. No regression in Step 7's Triangle/Stadium two-disjoint-copies smoke
   test, Step 11's `AnnularBilliard` dimension/`domain_id` checks, or the
   `spectrum_cases.jl` single-component `CompositeBIMSolver` cases (all
   numerically unchanged).
5. **Next action for the user:** invoke the **Julia Test Writer** subagent
   for: (a) a `BilliardGeometry.jl` unit test on `CircleSegment`/
   `_reverse_curve`/`_group_is_hole`-style orientation checks confirming
   `.length` stays non-negative after a standalone reversal, and that
   mixed-orientation components are rejected; (b) a
   `CompositeBIMSolver`-on-`AnnularBilliard` regression test comparing
   against the analytic annulus spectrum computed in implementation step 5;
   (c) a regression test for the new two-independent-outer-boundaries case
   (orientation-based convention correctly not reversing a non-hole second
   component).

## Definition of done — ALL SATISFIED

* `CircleSegment`'s `.length` field is non-negative for every valid
  `arc_angle` (positive or negative), fixed in `BilliardGeometry.jl`.
* `CompositeBIMSolver.evaluate_points` decides which components are holes
  from each component's own curve `orientation` field (`-1` ⇒ hole, `1` ⇒
  outer/independent boundary), not from component position — implemented
  via `_group_is_hole`.
* `CompositeBIMSolver`'s `AnnularBilliard` case no longer produces `NaN` in
  its assembled matrix and no longer raises the `KrylovKit`/LAPACK
  `ArgumentError`, confirmed root-caused (not merely worked around) to the
  `CircleSegment.length` sign defect combined with the positional hole
  assumption, not a `QuantumBilliards.jl` solver/Krylov bug.
* `CompositeBIMSolver` validated against the annulus's analytic spectrum
  (tension ≈9.6e-12 at the exact analytic root `k=3.1230309195852897`), not
  just structurally.
* No regression in any earlier step's `CompositeBIMSolver` checks or in the
  already-passing single-component `spectrum_cases.jl` cases; the new
  orientation-based convention additionally fixes a latent incorrectness
  for genuinely disjoint independent (non-hole) multi-component boundaries.
* Julia Test Writer still to be invoked (by the user) for the geometry-level
  regression test, the analytic-spectrum regression test, and the
  independent-outer-boundaries regression test.
* The migration plan's last remaining "known limitation" (flagged since
  Step 7) is closed.

