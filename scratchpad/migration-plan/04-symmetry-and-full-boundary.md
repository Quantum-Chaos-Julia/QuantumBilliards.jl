# Step 4 — Symmetry & full-boundary infrastructure (`BilliardGeometry.jl`)

## Goal

Close every gap between main's current, minimal `AbsReflection`/`NFoldRotation`/
`SymmetryOrbitMap` machinery and what `-develop` actually relies on, and
replace `-develop`'s per-billiard hand-maintained `full_boundary` field with a
single reusable **function** that reconstructs the complete physical boundary
from `fundamental_domain` + `symmetries`. This step produces no new solver
behavior by itself, but every symmetry-restricted BIM solve from Step 3
onward is only as correct as this infrastructure — and Step 3 explicitly left
the symmetric-boundary-sampling question unresolved (see "Known gap found"
below). Steps 5 (wavefunction/Husimi), 6 (CFIE), 7 (Composite), 8 (Beyn), 9
(EBIM) all depend on this step being correct, not just loadable.

## Preconditions

Steps 1–3 complete (in particular, `DoubleLayerPotentialSolver` and
`SymmetryOrbitMap` load and the no-symmetry case is verified — the
symmetry-enabled case is exactly what this step fixes).

## Known gap found while auditing Steps 2–3 (read this first)

`DoubleLayerPotentialSolver.evaluate_points` (Step 3,
[solvers/sweepmethods/dlp.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl))
calls `comp = get_boundary_curves(billiard)` regardless of whether
`solver.symmetry === nothing`. But
[`get_boundary_curves(billiard::AbsBilliard) = get_boundary_curves(billiard.fundamental_domain)`](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl#L33)
**only ever returns the fundamental domain's own `SpecularReflection` curves**
— for a symmetric billiard (e.g. `StadiumBilliard`, whose `fundamental_domain`
is the quarter-stadium `D₂` sector) this is a small fraction of the true
physical boundary, not the complete closed curve `symmetry_index_orbits`
implicitly assumes when it folds `pts.xy` by exact index permutation. Right
now this "works" only in the degenerate sense that `_dlp_evaluate_points`
discretizes *whatever curve list it is given* as if it were a full periodic
loop; it has never been exercised against a real symmetric billiard end to
end (Step 3's own test-verification section 5 only says "if the fixture used
has that symmetry", and was not confirmed). This step must:

1. Confirm this diagnosis by constructing a small script:
   `DoubleLayerPotentialSolver(...; symmetry=BilliardGeometry.YAxisReflection())`
   on `StadiumBilliard(0.5)`, print `length(get_boundary_curves(billiard))`
   and compare against the full stadium's physical boundary — expect a clear
   mismatch (fundamental domain's curve set is much shorter/smaller).
2. Fix `evaluate_points` in `dlp.jl` (and, going forward, any other
   `SweepBIMSolver`) to discretize the *true full physical boundary* — via
   the new `full_boundary`-reconstruction function this step adds (see §4.1
   below) — whenever `solver.symmetry !== nothing`, not the fundamental
   domain's own curve list.

## Source index

* `BilliardGeometry-develop/src/geometry/billiards/*.jl` (all 11 files) —
  read each billiard's constructor(s) to see exactly which curves go into the
  hand-written `full_boundary::Vector{AbsCurve}` field vs. `fundamental_domain`
  (e.g. `triangle.jl`'s `IsoscelesTriangleBilliard`: `fundamental_domain` has
  a `ReflectionSymmetry(YAxisReflection(),2)`-tagged "symmetry wall" edge that
  is *excluded* from `full_boundary`, which instead has the two physical
  edges the wall would reflect into). This is the exact reconstruction rule
  Step 4.1 below must reproduce generically: take every `SpecularReflection`
  (physical) curve from `fundamental_domain`, and for every curve whose `bc`
  is `ReflectionSymmetry(sym, N_sectors)`, replace it by nothing (it's a
  virtual wall, not a physical curve) while using `sym`/`N_sectors` to know
  how the physical curves repeat.
* `BilliardGeometry-develop/src/geometry/symmetry.jl` — full file (already
  read in full during planning, reproduced in this plan's research). Contains
  the richer `AbsReflection` definitions (`parity_x`/`parity_y` fields on
  `XAxisReflection`/`YAxisReflection`/`XYAxisReflection`), `DiagonalReflection`,
  `AntiDiagonalReflection`, `CompositeReflection`, `NFoldRotation` with a
  `sector` field, and `symmetry_irrep_character` — **none of which exist yet**
  in main's [BilliardGeometry.jl/src/geometry/symmetry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl)
  (main's reflections currently have **no fields at all**, hard-coding the
  trivial/even representation with no way to express an odd sector).
* `QuantumBilliards-develop/src/states/symmetry/symmetry_orbits.jl` — full
  file (already read in full). This is the *actual*, currently-used
  reference implementation of exact-index-permutation symmetry-orbit folding
  (`SymmetryOrbitMap`, `_build_periodic_symmetry_orbit_map`,
  `_rotation_orbits`, `_reflection_orbits`, `_composite_reflection_orbits`),
  richer than what Step 2 scaffolded into
  [BilliardGeometry.jl/src/geometry/symmetryorbits.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl)
  (main's `SymmetryOrbitMap` only has `fundamental_indices`/`orbit_of`/`phase`
  — the forward `fundamental → all orbit images` direction, i.e. `-develop`'s
  `fund_to_full`/`fund_to_scale` matrices, does not exist yet, and only
  `XAxisReflection`/`YAxisReflection`/`XYAxisReflection`/`NFoldRotation` have
  orbit-building methods, not `DiagonalReflection`/`AntiDiagonalReflection`/
  `CompositeReflection`).
* `QuantumBilliards-develop/src/states/symmetry/reflections.jl` — full file
  (already read in full). Contains `estimate_rmin_rmax(pts, symmetry)`
  (needed later by Step 5/10's Chebyshev radial-interval tuning) and
  `apply_symmetries_to_boundary_points`/`apply_symmetries_to_boundary_function`/
  `apply_symmetries_to_wavefunction` (the *basis*-solver symmetry-expansion
  helpers — already effectively covered in main by
  [states/symmetry/reflections.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/reflections.jl),
  check what's there first before re-porting anything from this file).
* `QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl` (already fully
  implemented, Step 3) — read `_dlp_fredholm_reduced!`,
  `boundary_matrix_size`, and `evaluate_points` again with this step's fix in
  mind; the `images = [Int[] for _ in 1:m]` array built fresh on every
  `construct_matrices` call is exactly the kind of thing the new
  `fund_to_full` reverse-orbit matrix (built once) should replace, but do
  **not** change `dlp.jl`'s numerical algebra in this step beyond the minimal
  `evaluate_points`/full-boundary fix in the "Known gap" section above —
  the `fund_to_full`-based cleanup of `_dlp_fredholm_reduced!` is optional
  polish, not required for this step's Definition of Done (flag it as a
  follow-up if not done here, since Step 3's `_dlp_fredholm_reduced!` is
  already numerically correct, just not maximally efficient).

## Files touched in main

### 4.1 — `full_boundary(billiard)` reconstruction function (new)

New file `BilliardGeometry.jl/src/geometry/fullboundary.jl` (included from
`geometry/geometry.jl` after `symmetryorbits.jl`), exporting a single new
generic function:

```julia
full_boundary(billiard::Bi) where {Bi<:AbsBilliard} -> curves::Vector{AbsCurve}
```

Implementation: start from `get_all_curves(billiard)` (every curve in every
subdomain of `fundamental_domain`, physical *and* symmetry-wall), keep only
curves whose `bc isa SpecularReflection` (the physical ones — Transparent/
PeriodicX-tagged curves belong to internal subdomain seams, not the outer
boundary, and are already excluded the same way `get_boundary_curves` excludes
them), then for every `sym in billiard.symmetries`, in composition order,
apply `apply_symmetry(sym, ·)` to every already-collected physical curve to
produce its reflected/rotated image curve(s), appending them in the
orientation/ordering convention `-develop`'s hand-written `full_boundary`
fields use (canonical start at `+x`, CCW traversal — reproduce this exactly by
comparing against 2–3 `-develop` billiards' hand-written `full_boundary`
field ordering, e.g. `IsoscelesTriangleBilliard`'s `right,left,base` order,
`StadiumBilliard`'s circle-then-rectangle-quadrant order). Each concrete
curve type (`LineSegment`, `CircleSegment`, arbitrary `AbsCurve`) needs its
own reflected-curve constructor (reflecting endpoints/control data through
`apply_symmetry`, not just sampled points) — check what generic
curve-transformation helpers `BilliardGeometry.jl` already has (search
`inversions.jl`/`utils.jl` for an existing "transform a curve by a
`LinearMap`" helper) before writing new per-curve-type code; reuse a generic
`AbsCurve` transformation if one already exists, otherwise add the minimal
one needed (`LineSegment`/`CircleSegment` are the only two curve types Triangle/
Stadium need for now — full generality across every `BilliardGeometry-develop`
curve/segment type is explicitly deferred to Step 11).

`full_boundary` must produce curves in the same closed, periodic, canonically
oriented ordering `symmetry_index_orbits`' exact-index-permutation model
requires (`_idx_reflect_x`/`_idx_reflect_y`/`_idx_rotate` in
`symmetryorbits.jl` assume a specific midpoint-node canonical ordering) —
verify this by discretizing `full_boundary(billiard)` at `N` points, applying
`apply_symmetry` to a sampled point, and confirming it lands within tolerance
on the index `_idx_reflect_x(q,N)`/etc. predicts, for both `TriangleBilliard`
(no symmetry — trivial `full_boundary(billiard) == get_boundary_curves(billiard)`)
and `IsoscelesTriangleBilliard`/`StadiumBilliard` (`YAxisReflection`/`D₂`).

No billiard struct gains a `full_boundary` field — this is the explicit
replacement for that `-develop` convention per the user's request. Update
`evaluate_points` in `dlp.jl` (per the "Known gap" fix above) to call
`full_boundary(billiard)` instead of `get_boundary_curves(billiard)` whenever
`solver.symmetry !== nothing`, keeping `get_boundary_curves(billiard)` (i.e.
`solver.symmetry === nothing` ⇒ discretize only the fundamental domain, which
*is* the full boundary in that case) for the non-symmetric path unchanged.

### 4.2 — Extend `AbsReflection`/`NFoldRotation` with irrep data (`BilliardGeometry.jl/src/geometry/symmetry.jl`)

Port verbatim from `BilliardGeometry-develop/src/geometry/symmetry.jl`:

* Add `parity_y::Int` to `XAxisReflection`, `parity_x::Int` to
  `YAxisReflection`, `parity_x::Int,parity_y::Int` to `XYAxisReflection`,
  each with a default constructor matching `-develop`
  (`XAxisReflection()=XAxisReflection(-1)`, etc.) so every existing call site
  (`XAxisReflection()`, used by `D2_symmetry`, `StadiumBilliard`, DLP tests)
  keeps working unchanged — **this is an additive, non-breaking field
  addition**, not a rename; existing positional-argument-free construction
  still works via the new default outer constructor.
* Add `DiagonalReflection(parity::Int=-1)`, `AntiDiagonalReflection(parity::Int=-1)`
  (new types, both `<: AbsReflection`), with `apply_symmetry` methods using
  new `reflect_diag`/`reflect_antidiag` `LinearMap` constants (port verbatim).
* Add `CompositeReflection(reflections::Vector{AbsReflection})` (+ splat
  constructor) — new type combining several reflection generators (needed by
  Step 11's more complex `BilliardGeometry-develop` billiards, e.g. squares
  with `D₄` symmetry combining axis + diagonal reflections; not exercised by
  Triangle/Stadium but cheap to port now alongside its sibling types since
  it's defined in the same source file).
* Add `sector::Int` and an `angle::Float64`-consistent constructor to
  `NFoldRotation` (currently missing the irrep sector — add as
  `NFoldRotation(N,m,sector::Int=0)`, keeping the existing 2-argument call
  sites working via the new default).
* Add `symmetry_irrep_character(::Type{T}, sym) where {T<:Real} → Complex{T}`
  — one method per concrete symmetry type, verbatim port of `-develop`'s
  parity/character formulas (`Complex{T}(sym.parity_y)` for `XAxisReflection`,
  etc., `cis(2π*sector*m/N)` for `NFoldRotation`).
* Export every new type/field/function from `BilliardGeometry.jl`.

This is additive and must not break any existing basis-solver symmetry usage
(`CornerAdaptedFourierBessel`, `RealPlaneWaves`) — check both compile and
their existing tests still pass after this change (they construct
`XAxisReflection()`/etc. with no arguments, which keeps working).

### 4.3 — Extend `SymmetryOrbitMap` with reverse orbit-expansion data (`BilliardGeometry.jl/src/geometry/symmetryorbits.jl`)

Add the two matrices `-develop`'s richer representation has and main's
currently lacks:

```julia
struct SymmetryOrbitMap{T<:Real}
    fundamental_indices::Vector{Int}
    orbit_of::Vector{Int}
    phase::Vector{Complex{T}}
    full_size::Int
    fundamental_size::Int
    fund_to_full::Matrix{Int}        # NEW: fund_to_full[g,b] = full-boundary index of group image g of fundamental node b
    fund_to_scale::Matrix{Complex{T}} # NEW: fund_to_scale[g,b] = irrep factor for that image
end
```

This is a breaking field addition to an internal (not yet used outside
`dlp.jl`/`BilliardGeometry.jl`) struct — since `SymmetryOrbitMap` is
constructed exclusively inside `_build_symmetry_orbit_map`/
`symmetry_index_orbits`, update that single builder to also fill
`fund_to_full`/`fund_to_scale` (the same information the builder already
computes transiently as `perms`, just also retained in fundamental-column
order) rather than adding a new parallel type. Add a
`symmetry_orbit(orbits, b) -> (@view(fund_to_full[:,b]), @view(fund_to_scale[:,b]))`
accessor (verbatim port of `-develop`'s `symmetry_orbit`, zero-allocation
view). This lets `_dlp_fredholm_reduced!`'s `images = [Int[] for _ in 1:m]`
workaround be replaced by direct `symmetry_orbit(orbits, b)` calls if the
implementing agent chooses to also do that optional `dlp.jl` cleanup (not
required, see Source-index note above).

Add exact-permutation orbit-building methods for the four new/extended
symmetry types added in §4.2, generalizing the existing
`_idx_reflect_x`/`_idx_reflect_y`/`_idx_rotate` helpers:

* `symmetry_node_multiple(::DiagonalReflection) = 8`,
  `symmetry_node_multiple(::AntiDiagonalReflection) = 8`,
  `symmetry_node_multiple(sym::CompositeReflection) = foldl(lcm, symmetry_node_multiple.(sym.reflections); init=1)`
  (verbatim port of `-develop`'s values — confirm by re-deriving from
  `_composite_reflection_orbits`, don't just copy the number blindly).
* `symmetry_index_orbits(::Type{T}, xy, symmetry::DiagonalReflection)`/
  `AntiDiagonalReflection` — new exact index maps for the diagonal/
  anti-diagonal periodic reflections (`-develop`'s `_idx_reflect_diag`-style
  helpers — locate the exact function names by reading
  `QuantumBilliards-develop/src/states/symmetry/symmetry_orbits.jl` past the
  point already read in this plan's research, since `_composite_reflection_orbits`
  references `_idx_reflect_diag`/`_idx_reflect_antidiag`-style generators that
  weren't shown in the excerpt captured so far).
* `symmetry_index_orbits(::Type{T}, xy, symmetry::CompositeReflection)` — port
  `_composite_reflection_orbits`'s group-closure-by-composition algorithm
  verbatim (generate the finite group from the constituent generators,
  building up `perms`/`scales` by composing permutations/multiplying irrep
  factors, matching-and-erroring on inconsistent parities exactly as
  `-develop` does).
* Update `_build_symmetry_orbit_map`(now filling `fund_to_full`/`fund_to_scale`
  too, see above) to keep working unchanged for the existing four types.

Only the **trivial representation** needs to keep working end-to-end for BIM
folding after this step (matches Step 2's already-documented scope limit) —
the new `phase`/`fund_to_scale`/`symmetry_irrep_character` plumbing is added
so that a *future* odd/anti-symmetric BIM sector is representable without
another breaking struct change, not because this step wires an odd BIM sector
through to `dlp.jl`'s matrix assembly (`dlp.jl` itself is not touched beyond
the `evaluate_points` fix in §4.1).

### 4.4 — `estimate_rmin_rmax` (`QuantumBilliards.jl/src/solvers/boundarypoints.jl` or a same-folder new file)

Port `estimate_rmin_rmax(pts::BoundaryPoints{T}, symmetry)` verbatim from
`QuantumBilliards-develop/src/states/symmetry/reflections.jl` (both the
`::Nothing` no-symmetry pairwise-distance version and the symmetry-orbit
version using `symmetry_index_orbits`/`fundamental_size`/`orbit_size`/
`fund_to_full`). This is not used by anything in this step, but is a direct,
small, self-contained prerequisite Step 5/10 need for Chebyshev-interpolated
Green's-function radial-interval tuning — port it now while the surrounding
symmetry API is fresh, rather than rediscovering it during Step 5/10.

## Performance & fidelity notes

* `full_boundary(billiard)` is called once per `evaluate_points` call (i.e.
  once per `k` in a sweep unless cached) — it's geometry construction, not a
  numerical hot loop; correctness (exact reproducibility of `-develop`'s
  hand-written curve orderings) matters far more here than micro-optimizing
  allocations, but still avoid `push!`-growing the curve vector inside a
  per-point loop (build it curve-by-curve, curves are few).
* `_build_symmetry_orbit_map`/exact-index-permutation orbit builders remain
  `@inbounds`, allocation-conscious per the existing Step 2 conventions —
  extending them with two new output matrices doesn't change this.
* Do not regress `dlp.jl`'s existing non-symmetric numerical path — only the
  `solver.symmetry !== nothing` branch of `evaluate_points` changes.

## Tests & user verification

1. `get_errors` on every new/edited file plus `BilliardGeometry.jl`/
   `QuantumBilliards.jl` top-level module files.
2. `full_boundary(TriangleBilliard(p1,p2,p3))` (no symmetry) must exactly
   equal `get_boundary_curves(billiard)` (same curves, same order) — this is
   the trivial-symmetry-group sanity check.
3. Construct `IsoscelesTriangleBilliard`-equivalent 
   (you are allowed to port over the implementation for this billiard) and/or
   `StadiumBilliard(0.5)` fixtures already in main; confirm
   `full_boundary(billiard)`'s total arc length equals the known full
   physical perimeter (e.g. full stadium perimeter `2*half_width + 2*π`
   for `StadiumBilliard(half_width)` with unit radius), not the fundamental
   domain's (quarter) perimeter.
4. Discretize `full_boundary(billiard)` at `N` points (`N` a multiple of
   `symmetry_node_multiple(symmetry)`), apply `apply_symmetry(symmetry, xy[i])`
   for a handful of `i`, and confirm the nearest full-boundary node matches
   `_idx_reflect_x(i,N)`/`_idx_reflect_y(i,N)`/etc. within tolerance — this is
   the exact-permutation-consistency check the whole BIM symmetry-folding
   model depends on.
5. Re-run `DoubleLayerPotentialSolver(...; symmetry=BilliardGeometry.YAxisReflection())`
   on the Stadium/Triangle fixture from Step 3's test plan, now sampling
   `full_boundary(billiard)` instead of `get_boundary_curves(billiard)`, and
   confirm the located `k0` now matches the non-symmetric
   `DoubleLayerPotentialSolver` solve restricted to the same parity subspace
   (cross-check against `RealPlaneWaves(...; sym_y=-1)`-restricted
   `VerginiSaracenoSolver`, as Step 3 already intended but could not fully
   verify due to this gap).
6. Confirm `symmetry_irrep_character`/new reflection parity fields don't
   change any existing basis-solver (`CornerAdaptedFourierBessel`/
   `RealPlaneWaves`) behavior — re-run (or ask the user to re-run)
   `test/solvertests.jl` and confirm all existing testsets still pass
   unchanged.
7. Add the unit tests that you already performed to the testsuit. You are explicitly allowed to do this in this case.

8. Tell the user to invoke the **Julia Test Writer** subagent for unit tests
   covering `full_boundary` (trivial + `D₂`/reflection cases), the new
   `DiagonalReflection`/`AntiDiagonalReflection`/`CompositeReflection`
   orbit-building methods, and `estimate_rmin_rmax`.

## Definition of done

* `full_boundary(billiard)` implemented and verified against at least one
  trivial-symmetry and one reflection-symmetric fixture; no billiard struct
  gained a stored `full_boundary` field.
* `DoubleLayerPotentialSolver`'s symmetric-case boundary sampling fixed to use
  `full_boundary(billiard)`, and a symmetric DLP solve now verified end to end
  against a known-parity `VerginiSaracenoSolver` result (closing the gap Step
  3 could not confirm).
* `AbsReflection`/`NFoldRotation` extended with irrep parity/sector data and
  `symmetry_irrep_character`, additively (no existing call sites broken).
* `SymmetryOrbitMap` extended with `fund_to_full`/`fund_to_scale` +
  `symmetry_orbit` accessor; orbit-building methods added for
  `DiagonalReflection`/`AntiDiagonalReflection`/`CompositeReflection`.
* `estimate_rmin_rmax` ported and available for Step 5/10.
* Existing basis-solver tests (`test/solvertests.jl`) confirmed unaffected.
* Julia Test Writer invoked for this step's new code.
