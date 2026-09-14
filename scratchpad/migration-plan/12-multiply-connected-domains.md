# Step 12 — `AbsMultiplyConnectedDomain` / `MultiplyConnectedDomain`

## Goal

Give multiply connected planar domains (regions with one or more holes) a
dedicated, explicitly-named geometry abstraction in `BilliardGeometry.jl`
instead of the implicit `SimpleDomain`-with-opposite-curve-orientations hack
`AnnularBilliard` currently relies on, and use it to harden — not redesign —
`CompositeBIMSolver`'s existing (already-working at the sampling/assembly
level) wiring in `QuantumBilliards.jl`. Rename
`billiards/circle_with_hole.jl` → `billiards/annular.jl` to match the struct
name it already defines (`AnnularBilliard`) and refactor it onto the new
type.


## Preconditions

Step 11 complete (`AnnularBilliard` already exists and is verified to produce
a correct, fully closed two-ring boundary — see
`BilliardGeometry.jl/memories/repo/known-bugs.md`'s `connect_curves` fix and
`QuantumBilliards.jl/memories/repo/bim-solver-notes.md`'s Step 11 findings).
This step is pure refactoring/hardening of already-working geometry, not new
numerical functionality.

## Current state (read this before changing anything)

* [BilliardGeometry.jl/src/geometry/billiards/circle_with_hole.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/circle_with_hole.jl):
  `AnnularBilliard{T} <: AbsBilliard` holds one `SimpleDomain{T}` whose
  `.boundary` is `AbsCurve[outer, inner]` — an outer full `CircleSegment`
  (`orientation=1`, `domain_id=1`) and an inner full `CircleSegment`
  (`orientation=-1`, `domain_id=2`). The "hole" behavior (point must be
  inside the outer circle **and** outside the inner one) falls entirely out
  of `is_inside(domain::D, pt) where D<:AbsDomain` ([geometry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/geometry.jl))
  already being an `all(...)`-over-`domain.boundary`-curves check combined
  with the inner curve's flipped `orientation`. There is no dedicated type,
  and nothing marks this domain as topologically different from a plain
  simply connected `SimpleDomain` — a reader (or a future billiard with two
  holes) has to already know the orientation trick to reproduce it correctly.
* `CompositeBIMSolver` ([solvers/sweepmethods/compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl))
  already groups the flat curve list returned by `get_boundary_curves`/
  `full_boundary` by each curve's `domain_id` (`_group_boundary_by_domain_id`),
  assigns `component_solvers[1]` to the lowest `domain_id` group (interpreted
  as the outer boundary) and `component_solvers[2:end]` to the rest
  (interpreted as holes, reversed). This already works structurally for
  `AnnularBilliard` today (per `bim-solver-notes.md`: "`AnnularBilliard(2.0,1.0)`
  gives 400 points, correct 400×400 matrix, correct `domain_id` grouping") —
  only `solve_vect`'s `KrylovKit.svdsolve` call itself fails
  (`ArgumentError("operator and its adjoint are not compatible")`), which is
  **out of scope for this step** and deferred to Step 13.
* Two call sites dispatch on domain *type* (not just `AbsBilliard`) and would
  silently `MethodError` if `AnnularBilliard.fundamental_domain` stopped being
  an `AbsSimpleDomain` without a matching new method:
  1. [BilliardGeometry.jl/src/geometry/boundarytypes.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl):
     `get_boundary_curves(domain::D) where D<:AbsSimpleDomain`.
  2. [QuantumBilliards.jl/src/solvers/boundarypoints.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/boundarypoints.jl):
     `get_boundary_curves_with_ignored(domain::D) where D<:AbsSimpleDomain`
     (used by `boundary_coords`, and directly by `QBPlotting.jl`'s
     `plot_boundary!`/`husimiplotting.jl`).
  Everything else that branches on domain type
  (`get_all_domains`/`get_domain`/`update_boundary_condition` in
  `boundarytypes.jl`) does so with `typeof(domain) <: AbsCompositeDomain`
  (an `if/else`, not multiple dispatch) and already treats any non-composite
  domain — `SimpleDomain` today, `MultiplyConnectedDomain` after this step —
  identically as "the billiard's one fundamental domain object", so these
  need **no changes**. `is_inside(domain::D, pt) where D<:AbsDomain` is
  already dispatched on the *top-level* `AbsDomain` abstract type, not
  `AbsSimpleDomain`, so it also needs **no changes** — confirm this with
  `mcp_pylance`-style/`grep_search` re-verification immediately before
  editing, not just trusted from this plan, in case new call sites were added
  since this audit.

## Design decisions

1. **Where in the type hierarchy**: `AbsMultiplyConnectedDomain` is a new
   sibling of `AbsSimpleDomain`/`AbsCompositeDomain` directly under
   `AbsDomain` (declared in [BilliardGeometry.jl/src/BilliardGeometry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/BilliardGeometry.jl)),
   **not** a subtype of `AbsCompositeDomain`. `AbsCompositeDomain`'s
   `get_boundary_curves`/is-inside semantics are a *union* over subdomains
   (correct for `StadiumBilliard`'s overlapping rectangle+circle-cap shape);
   a domain with a hole needs the opposite — an *intersection* ("inside the
   outer curve **and** outside every hole") — so inheriting from
   `AbsCompositeDomain` would be semantically wrong even if it happened to
   compile. This mirrors the reasoning already recorded in
   `bim-solver-notes.md`'s "`AnnularBilliard` design note".
2. **Field shape**: keep `MultiplyConnectedDomain{T}` structurally close to
   `SimpleDomain{T}` (flat `boundary::Vector{AbsCurve}` + `corners` + `id`)
   rather than nesting separate `outer`/`holes` sub-domain objects, so every
   existing curve-level convention (per-curve `domain_id`/`orientation`,
   `all(...)`-over-`.boundary` `is_inside`) keeps working via the already
   `AbsDomain`-generic methods with zero behavioral change — only the type
   name and the explicit `genus` field are new. Add a `genus::Int64` field
   (number of holes = number of connected physical boundary components − 1;
   `1` for `AnnularBilliard`) as the user requested, even though it is
   technically derivable from the curves' distinct `domain_id`s — storing it
   explicitly makes it a cheap, explicit invariant `CompositeBIMSolver` (and
   future multi-hole billiards) can validate against without re-deriving it
   from curve bookkeeping every time.
3. **`full_boundary`/`get_boundary_curves` "outermost first, more curves via
   dispatch"**: `get_boundary_curves`/`full_boundary` already return curves
   grouped into contiguous per-component runs, in **first-listed-first**
   order, once `connect_curves` peels one connected chain at a time (already
   fixed for Step 11 — see `known-bugs.md`). "Outermost first" is therefore
   already guaranteed **by construction**, not by any curve-list
   post-processing: `MultiplyConnectedDomain.boundary` must list the outer
   ring's curves before any hole's curves (mirroring
   `AnnularBilliard`'s current `AbsCurve[outer, inner]` order). "Multiply
   dispatched to return more curves" is satisfied by adding the new
   `get_boundary_curves(domain::D) where D<:AbsMultiplyConnectedDomain`
   method itself (item 1 of the current-state call-site list above): for an
   `AbsSimpleDomain` this dispatch returns one ring's curves, for an
   `AbsMultiplyConnectedDomain` the *same*-looking call returns every ring's
   curves concatenated (already true today only because `AnnularBilliard`
   happens to reuse the `AbsSimpleDomain` method — this step makes that
   correct-by-construction on its own dedicated type instead of by
   incidental inheritance). `full_boundary(billiard)` itself
   ([fullboundary.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/fullboundary.jl))
   needs **no changes**: it already just calls `get_boundary_curves(billiard)`
   and appends symmetry images of *every* curve in that flat list, which
   remains correct once `get_boundary_curves` is dispatched correctly.
4. **New `boundary_components` accessor (recommended, not strictly required)**:
   add `boundary_components(domain) → Vector{Vector{AbsCurve}}`, multiply
   dispatched: `boundary_components(::AbsSimpleDomain)`/
   `boundary_components(::AbsCompositeDomain)` return a single-element vector
   (`[get_boundary_curves(domain)]`), `boundary_components(::AbsMultiplyConnectedDomain)`
   returns one vector per connected ring (outer first, then each hole, using
   the domain's per-curve `domain_id` to split — same grouping logic
   `CompositeBIMSolver._group_boundary_by_domain_id` already implements
   privately, but exposed once from `BilliardGeometry.jl` instead of
   duplicated/reinvented per BIM solver). `boundary_components(billiard) =
   boundary_components(billiard.fundamental_domain)`. This lets
   `CompositeBIMSolver.evaluate_points` call one public, tested
   `BilliardGeometry` function instead of re-deriving component grouping from
   raw `domain_id` scanning — see "Wiring into `QuantumBilliards.jl`" below.
   Add a matching `genus(billiard) = genus(billiard.fundamental_domain)`
   accessor (`genus(::AbsSimpleDomain) = 0`, `genus(::AbsCompositeDomain) = 0`,
   `genus(d::AbsMultiplyConnectedDomain) = d.genus`) so any solver can
   sanity-check `length(component_solvers) == 1 + genus(billiard)` up front.

## Files touched

* [BilliardGeometry.jl/src/BilliardGeometry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/BilliardGeometry.jl) —
  new `abstract type AbsMultiplyConnectedDomain <: AbsDomain end` + export.
* New `BilliardGeometry.jl/src/geometry/domains/multiplyconnecteddomains.jl` —
  `MultiplyConnectedDomain{T}` struct + constructor(s).
* [BilliardGeometry.jl/src/geometry/boundarytypes.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl) —
  new `get_boundary_curves(::AbsMultiplyConnectedDomain)` method (factor the
  shared "filter physical curves + `connect_curves`" body into a small
  private helper reused by the existing `AbsSimpleDomain` method, rather than
  copy-pasting the two-line body); new `boundary_components`/`genus`
  functions + exports (design decision 4).
* [BilliardGeometry.jl/src/geometry/geometry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/geometry.jl) —
  `include`/`export` wiring for the new domain file; rename the
  `circle_with_hole.jl` include line to `annular.jl`.
* Rename `BilliardGeometry.jl/src/geometry/billiards/circle_with_hole.jl` →
  `BilliardGeometry.jl/src/geometry/billiards/annular.jl`; refactor
  `AnnularBilliard`'s constructor to build a `MultiplyConnectedDomain`
  instead of a bare `SimpleDomain`.
* [QuantumBilliards.jl/src/solvers/boundarypoints.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/boundarypoints.jl) —
  new `get_boundary_curves_with_ignored(::AbsMultiplyConnectedDomain)` method
  (same factor-a-shared-helper treatment as above), so `boundary_coords`/
  `QBPlotting.jl`'s `plot_boundary!` keep working for `AnnularBilliard`.
* [QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl) —
  additive hardening only (see "Wiring" section below); no change to the
  actual kernel-assembly numerics (that machinery already works — Step 13
  fixes the one remaining `solve_vect` bug).

## Implementation steps

1. Declare `abstract type AbsMultiplyConnectedDomain <: AbsDomain end` next
   to `AbsSimpleDomain`/`AbsCompositeDomain` in `BilliardGeometry.jl`'s module
   header; add it to the module's `export` line.
2. Create `domains/multiplyconnecteddomains.jl`:
   ```julia
   struct MultiplyConnectedDomain{T} <: AbsMultiplyConnectedDomain where T<:Real
       boundary::Vector{AbsCurve}
       corners::Vector{SVector{2,T}}
       id::Int64
       genus::Int64
   end
   ```
   Add a convenience constructor taking the outer curve(s), a
   `Vector{<:Vector{<:AbsCurve}}` of hole curve groups, and `corners`,
   computing `genus = length(hole_groups)` and concatenating
   `boundary = AbsCurve[outer_curves...; reduce(vcat, hole_groups)...]` —
   this is the constructor `AnnularBilliard` (and any future two-or-more-hole
   billiard) calls; do not hand-roll per-billiard `genus`/`boundary`
   concatenation at each call site.
3. In `geometry.jl`, add
   `include("domains/multiplyconnecteddomains.jl")`/
   `export MultiplyConnectedDomain` next to the existing
   `include("domains/compositedomains.jl")` line; rename the
   `circle_with_hole.jl` include to `annular.jl`.
4. In `boundarytypes.jl`: factor the existing
   `get_boundary_curves(domain::D) where D<:AbsSimpleDomain` body into a
   private `_connected_physical_curves(boundary::Vector{AbsCurve})` helper
   (the `is_outer`-filter + `connect_curves` two-liner, unchanged logic), then
   define both
   `get_boundary_curves(domain::D) where D<:AbsSimpleDomain = _connected_physical_curves(domain.boundary)`
   and the new
   `get_boundary_curves(domain::D) where D<:AbsMultiplyConnectedDomain = _connected_physical_curves(domain.boundary)`
   off the same helper. Add `genus(domain::AbsDomain) = 0`,
   `genus(domain::AbsMultiplyConnectedDomain) = domain.genus`,
   `genus(billiard::AbsBilliard) = genus(billiard.fundamental_domain)`, and
   `boundary_components` per design decision 4 (implement the
   `AbsMultiplyConnectedDomain` case by grouping `domain.boundary` by each
   curve's `.domain_id`, preserving first-seen order — this is the same
   grouping algorithm as `CompositeBIMSolver._group_boundary_by_domain_id`;
   port it here once and have `compositebim.jl` call the new public function
   in step 6 instead of keeping its own private copy). Export
   `genus`/`boundary_components`.
5. Rename `billiards/circle_with_hole.jl` → `billiards/annular.jl` (`git mv`,
   not a manual copy+delete, to preserve history) and refactor
   `AnnularBilliard`'s constructor to call the new
   `MultiplyConnectedDomain(...)` convenience constructor from step 2 instead
   of building `SimpleDomain{T}(AbsCurve[outer,inner], vertices, 1)` by hand —
   `outer` passed as a one-element outer-curve vector, `[[inner]]` as the
   single hole-group, `genus` comes out as `1` automatically. Keep every
   numeric argument/keyword (`R_outer`, `R_inner`, `center`) and validation
   check byte-for-byte identical; this is a type-machinery refactor, not a
   geometry change — `AnnularBilliard(R_outer, R_inner)` must produce
   bit-identical curves to before.
6. In `compositebim.jl`: replace the private
   `_group_boundary_by_domain_id`/its call site in `evaluate_points` with a
   call to the new public `BilliardGeometry.boundary_components(billiard)`
   (falling back to grouping by curve `domain_id` only if the billiard's
   domain isn't `AbsMultiplyConnectedDomain` — i.e. keep supporting the
   existing "any billiard with `domain_id`-tagged curves" contract
   `CompositeBIMSolver`'s docstring already describes, not narrowing it to
   only `AbsMultiplyConnectedDomain` billiards). Add one new upfront
   validation line to `evaluate_points`:
   `billiard.fundamental_domain isa AbsMultiplyConnectedDomain && length(solver.component_solvers) != 1 + BilliardGeometry.genus(billiard) && throw(ArgumentError(...))`
   — a clearer, earlier error message than the existing post-grouping
   `length(groups) == nc` check (keep that check too, as a defense-in-depth
   fallback for non-`AbsMultiplyConnectedDomain` billiards).
7. Rename-safety pass: `grep_search` the whole workspace for
   `circle_with_hole` (docs, plan files, diagrams) and update any remaining
   references to `annular.jl`/`AnnularBilliard` — the master index table row
   for Step 11 in `00-index.md` and `10.9-geometry-audit.md`'s prose mentions
   are historical narrative and do not need editing, but double-check no
   *code* (e.g. `docs/make.jl`, `Manifest.toml`-adjacent doc source) still
   references the old filename.

## Wiring into `QuantumBilliards.jl` (what actually changes vs. what already works)

* `evaluate_points(solver::CompositeBIMSolver, billiard, k)` already correctly
  splits `AnnularBilliard`'s boundary into outer/hole component groups today
  (verified in Step 11 — see current-state section) purely from curve
  `domain_id`s; this step's `compositebim.jl` change (implementation step 6)
  is a robustness/API-cleanliness improvement (one shared, tested,
  `BilliardGeometry`-exported grouping function instead of a private
  per-solver copy; an earlier/clearer `genus`-based error message) — **not**
  a bug fix, because there is no grouping bug to fix. Do not conflate this
  with Step 13's actual `solve_vect` numerical bug.
* No changes needed to `construct_matrices`/`solve`/`solve_vect`, the
  same-component or cross-component kernel-entry helpers, or
  `_merge_composite_points`/`_composite_offsets` — none of that logic
  inspects domain *type*, only the already-correct flat `BoundaryPoints` +
  `domain_id` bookkeeping `evaluate_points` produces, which is unchanged by
  this step's refactor.
* `BIMEigenstate`'s `wavefunction`/`boundary_function`/`husimi_function`
  (Step 5.5/5.6) are also unaffected — they consume `state.pts`, not the
  billiard's domain object.

## Performance & fidelity notes

This is a type/dispatch-layer refactor with no new numerics: every curve
constructed for `AnnularBilliard` must remain bit-identical to today's output
(`CircleSegment(R_outer,...)`/`CircleSegment(R_inner,...)` with the same
`orientation`/`domain_id` tags). Verify this concretely (implementation
step 5's own instruction) rather than assuming a "structural-only" refactor
is automatically safe.

## Tests & user verification

1. `get_errors` on `BilliardGeometry.jl` and `QuantumBilliards.jl` after each
   file change (not batched at the end).
2. `AnnularBilliard(2.0, 1.0)` still produces `boundary_matrix_size(pts) ==
   400` for the same discretization settings Step 11 used, and
   `BilliardGeometry.genus(billiard) == 1`.
3. Byte-for-byte curve check: compare `get_boundary_curves(AnnularBilliard(2.0,1.0))`
   before/after this step (e.g. compare `.center`/`.radius`/`.arc_angle`/
   `.orientation`/`.domain_id` field-by-field for both curves) to confirm the
   refactor changed no geometry.
4. `is_inside(AnnularBilliard(2.0,1.0), pt)` still correctly classifies a
   point inside the annulus (`true`), inside the hole (`false`), and outside
   the outer circle (`false`).
5. `QBPlotting.jl`'s `plot_boundary!(ax, AnnularBilliard(2.0,1.0))` still
   renders both rings (confirms the new
   `get_boundary_curves_with_ignored(::AbsMultiplyConnectedDomain)` method is
   wired correctly, not silently falling through to a `MethodError` that
   would previously not have existed).
6. `CompositeBIMSolver`'s existing structural checks from Step 7/11
   (`evaluate_points`/`construct_matrices` dimensions, correct component
   grouping) re-run against the refactored `AnnularBilliard` and produce
   identical results to Step 11's findings — this step must not regress
   anything Step 11 already validated. Full numerical `solve`/`solve_vect`
   verification remains deferred to Step 13.
7. Tell the user to invoke the **Julia Test Writer** subagent for
   `MultiplyConnectedDomain`/`AnnularBilliard` geometry unit tests (`genus`,
   `boundary_components`, `is_inside`, boundary closure/orientation) once the
   above manual checks pass.

## Definition of done

* `AbsMultiplyConnectedDomain`/`MultiplyConnectedDomain` exist, exported,
  documented, and are the type used by `AnnularBilliard`.
* `billiards/annular.jl` replaces `billiards/circle_with_hole.jl` with
  identical produced geometry.
* Both previously `AbsSimpleDomain`-only dispatch sites
  (`get_boundary_curves`, `get_boundary_curves_with_ignored`) have matching
  `AbsMultiplyConnectedDomain` methods; `genus`/`boundary_components` exist
  and are exported from `BilliardGeometry.jl`.
* `CompositeBIMSolver.evaluate_points` uses the new shared
  `boundary_components`/`genus` instead of a private domain_id scan, with no
  behavioral change to already-working structural checks.
* Julia Test Writer invoked for the new geometry unit tests.
* The known `CompositeBIMSolver.solve_vect` `KrylovKit` `ArgumentError` is
  explicitly **not** addressed here — confirmed still reproducible and
  handed off to Step 13.
