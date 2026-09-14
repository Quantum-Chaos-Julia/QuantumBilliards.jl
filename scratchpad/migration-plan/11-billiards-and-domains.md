# Step 11 — Billiards, domains and curve segments from `BilliardGeometry-develop`

## Goal

Port the missing billiard/domain/curve-segment catalogue from
`BilliardGeometry-develop` into `BilliardGeometry.jl`, last, as specified by
the user. Triangle and Stadium (already in main) are the only fixtures used
by Steps 1–8; this step both fills out the geometry catalogue and back-fills
the numerical validation Steps 5–7 could only partially do without a
multiply-connected or analytically-known-spectrum billiard.

## Preconditions

Steps 1–8 complete. This step has no numerical dependency on Steps 1–8 (it's
pure geometry), but is ordered last per the user's explicit instruction, and
because its main practical value (real validation fixtures) only pays off
once every solver already exists to validate against.

## Source index — full inventory of what's missing

| Category | In `BilliardGeometry-develop` but not in main | Notes |
|---|---|---|
| Billiards | `c3.jl`, `circle.jl`, `circle_with_hole.jl`, `ellipse.jl`, `polygon.jl`, `prosen.jl`, `rectangle.jl`, `rectangle_in_rectangle.jl`, `sinai.jl`, `star.jl` | Main already has `limacon.jl`, `mushroom.jl`, `polar.jl`, `stadium.jl`, `triangle.jl` — do not re-port those, diff first to check for any bugfix-only drift before assuming they're identical. |
| Domains | none missing — `circular.jl`, `compositedomains.jl`, `polygons.jl` present in both, confirmed identical file lists. | Diff contents anyway; `-develop` domains may have picked up fixes while adding the new billiards above. |
| Segments | `curve_derivatives.jl` (develop only), `polarsegment.jl` (develop) vs. main's `polarcurves.jl` (naming drift — confirm these are the same concept under a different name, not two different things) | Main has `circlesegment.jl`, `compositecurves.jl`, `linesegment.jl`, `polarcurves.jl`; develop has `circlesegment.jl`, `compositecurves.jl`, `curve_derivatives.jl`, `linesegment.jl`, `polarsegment.jl`. Diff `polarcurves.jl` vs `polarsegment.jl` directly before assuming either is redundant. |

## Files touched in main

`BilliardGeometry.jl/src/geometry/billiards/*.jl` (new files for each missing
billiard), plus whatever `segments/curve_derivatives.jl` turns out to add
(new file if genuinely new functionality, e.g. curve-derivative utilities
needed by the new billiards' curvature/tangent computations).

## Implementation steps

1. Diff every *already-present* file (`limacon.jl`, `mushroom.jl`, `polar.jl`,
   `stadium.jl`, `triangle.jl` in both `billiards/` folders; every `domains/`
   file; `circlesegment.jl`/`compositecurves.jl`/`linesegment.jl` in
   `segments/`) before porting anything new — if `-develop` picked up bug
   fixes to shared infrastructure while these billiards were being added,
   those fixes need to land in main too, following the same
   `julia-refactor-port-develop` "re-derive against main's current API, don't
   copy verbatim" discipline as every other step.
2. Resolve the `polarcurves.jl`/`polarsegment.jl` naming question (see source
   index table) before porting `circle.jl`/`ellipse.jl`/`prosen.jl`/`star.jl`,
   since polar-parametrized billiards likely depend on whichever of these is
   the actual polar-curve segment type.
3. Port each missing billiard one at a time (not all at once), in an order
   that unblocks the earlier steps' deferred validation first:
   1. `circle.jl` — analytically known Bessel-zero spectrum; the single most
      valuable validation fixture for every solver in Steps 3 and 6–9 (DLP/CFIE/
      Beyn/EBIM can all be checked against exact `J_m(kR)=0`/`J_m'(kR)=0`
      eigenvalues, not just cross-solver agreement).
   2. `sinai.jl` or `circle_with_hole.jl` — a genuinely multiply-connected
      billiard, needed to close out Step 7's deferred `CompositeBIMSolver`
      full numerical verification. `rectangle_in_rectangle.jl` is the third
      option if the other two prove harder to port cleanly — pick whichever
      is the simplest multiply-connected geometry to get right first.
   3. `ellipse.jl`, `rectangle.jl`, `star.jl`, `c3.jl`, `prosen.jl`,
      `polygon.jl` — remaining billiards, any order, each independent.
   Each billiard follows the existing `AbsBilliard` construction pattern
   already used by `StadiumBilliard`/`TriangleBilliard` (read those two main
   files first as the structural template per the `julia-add-solver`-style
   "closest existing sibling" guidance, even though this is geometry not a
   solver) — do not invent a new billiard-construction convention.
4. For each newly ported billiard, immediately re-run the deferred
   validation from earlier steps where applicable:
   * `circle.jl`: re-verify `DoubleLayerPotentialSolver`/
     `CombinedFieldIntegralEquationSolver`/`BeynSolver`/`ExpandedBIMSolver`
     against the exact Bessel-zero spectrum (a strictly stronger check than
     the cross-solver-agreement checks done in Steps 3 and 6–9, since it validates
     absolute correctness, not just mutual consistency of independently
     buggy implementations).
   * `sinai.jl`/`circle_with_hole.jl`/`rectangle_in_rectangle.jl`: re-run
     Step 7's `CompositeBIMSolver` full numerical verification (real
     eigenvalues on a real multiply-connected domain, not the synthetic
     two-disjoint-copies smoke test).
   * Any billiard with `D2`/`NFoldRotation` symmetry (e.g. `star.jl`,
     `c3.jl`): re-verify `symmetry_index_orbits`/`symmetry_node_multiple`
     (Step 2) end-to-end against a real symmetric billiard rather than the
     synthetic circle-point-set test done there.

## Performance & fidelity notes

Geometry code (curve parametrizations, `tangent`/`tangent_2`/`arc_length`) is
typically evaluated many times inside solver hot loops — preserve
`-develop`'s closed-form derivative expressions verbatim rather than
re-deriving them, and keep `SVector{2,T}`/concrete `T<:Real` typing throughout
exactly as the existing `stadium.jl`/`triangle.jl` already do.

## Tests & user verification

1. `get_errors` after each individual billiard port (not batched) — confirm
   `BilliardGeometry.jl` still loads after every single addition.
2. Boundary sanity check per billiard: `plot_boundary!` (`QBPlotting.jl`,
   already generic over any `AbsBilliard`) renders a closed, correctly
   oriented curve; `BilliardGeometry.is_inside` correctly classifies a known
   interior and exterior point.
3. Run the re-verification checklist in implementation step 4 above for
   `circle.jl` and the chosen multiply-connected billiard as soon as they
   land — these are high-value correctness checks for the *whole* migration,
   not just this step.
4. `QBPlotting.jl`: confirm `plot_boundary!`/`plot_wavefunction!`/
   `plot_probability!` all already work unmodified for every new billiard
   (they're generic over `AbsBilliard`/`AbsState` already) — if any billiard
   type needs a bespoke plotting tweak (e.g. `circle_with_hole`'s inner
   boundary needing a distinct line style), that's a small addition to
   [wavefunctionplotting.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/wavefunctionplotting.jl)'s
   `plot_boundary!`, not a new function.
5. Tell the user to invoke the **Julia Test Writer** subagent for each new
   billiard's geometry unit tests (boundary closure, area, `is_inside`) and,
   for `circle.jl` specifically, a dedicated analytic-spectrum regression
   test across every solver family from Steps 1–7.

## Definition of done

* Every billiard/domain/segment file diffed and ported (or explicitly
  confirmed identical/not needed).
* `circle.jl` validates every solver family against an exact analytic
  spectrum.
* A multiply-connected billiard validates `CompositeBIMSolver` fully,
  closing out Step 7's deferred item.
* A symmetric billiard validates `symmetry_index_orbits` end-to-end, closing
  out Step 2's synthetic-only verification.
* Julia Test Writer invoked for geometry + analytic-spectrum regression
  coverage.
