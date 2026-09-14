# Step 16 — Unified `SymmetrySector` representation API for basis and BIM solvers

Status: **Plan only, not started (v2, comprehensive rework).** Deferred out
of Step 15
([15-symmetry-representation-refactor.md](15-symmetry-representation-refactor.md),
"Implementation notes" / "One deviation from the plan as written"), then
reworked end-to-end after a design critique of v1 surfaced that the
originally proposed `_resolve_bim_symmetry` silently mishandled the two most
common cases in the current billiard catalogue (D2 reflection pairs and
multi-generator rotation groups). This version replaces v1 entirely rather
than patching it further. Not a `-develop` port — this closes a gap
introduced by Step 15's own Layer 1/Layer 2 split, and independently closes
[10.9-geometry-audit.md](10.9-geometry-audit.md)'s Gap F (nothing
cross-checks a BIM solver's `symmetry` field against the billiard it is
actually used with).

## Goal

One representation-selection object, `SymmetrySector` (already introduced by
Step 15), usable identically by **every** solver family:

```julia
sector = symmetry_sector(billiard, XAxisReflection=>-1, YAxisReflection=>-1)
basis  = RealPlaneWaves(dim, billiard, sector)
solver = DoubleLayerPotentialSolver(pts_scaling_factor, billiard, sector)
```

with **every** representation that is *group-theoretically well-posed* for
the billiard's actual registered symmetry group resolved correctly and
automatically — no `ArgumentError` for a request a user would reasonably
expect to work. The only requests this plan still rejects are ones that are
**inherently ambiguous or self-contradictory as one-dimensional
representations**, not ones that merely weren't implemented; each such case
is derived from the actual group theory of the (always abelian: trivial,
`Z2`, `Z2×Z2`, or `Zn`) symmetry groups present in the current billiard
catalogue, not from an arbitrary scope cut. This distinction — *irreducible
mathematical restriction* vs. *implementation gap* — is the organizing
principle of this rework and is called out explicitly at every point it
applies.

## Key insight that v1 missed: what `billiard.symmetries` actually stores

`SymmetryRegistry`'s own docstring
([symmetryregistry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryregistry.jl))
states it plainly: it stores **every non-identity element of the billiard's
symmetry group**, not a minimal generating set. Concretely, across the whole
catalogue
([BilliardGeometry.jl/src/geometry/billiards/](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards)):

* `D2_symmetry()` (`RectangleBilliard`, `StadiumBilliard`, `CircleBilliard`,
  `EllipseBilliard`, `ProsenBilliard`) = `register_symmetries(YAxisReflection(),
  XYAxisReflection(), XAxisReflection())` — the **three** non-identity
  elements of the four-element group `D2 ≅ Z2×Z2 = {id, reflect_x,
  reflect_y, rotate_π}` (`XYAxisReflection` is the `rotate_π` element,
  `= reflect_x ∘ reflect_y`).
* `Cn_symmetry(n)` (`C3Billiard`, `StarBilliard`) = `register_symmetries((NFoldRotation(n,i)
  for i in 1:(n-1))...)` — all `n-1` non-identity elements of the cyclic
  group `Zn`.
* `LimaconBilliard` = `register_symmetries(XAxisReflection())` — the one
  non-identity element of `Z2`.
* Everything else (`PolygonBilliard`, `TriangleBilliard`, `AnnularBilliard`,
  `PolarBilliard`) registers no symmetry at all.

So **"choosing a representation" is never about picking a character per
registered `sym_id` independently** — it is about picking the natural
minimal set of quantum numbers for the *group's actual shape*, which then
determines every registered element's character via the group's (always
abelian) multiplication table:

| Group shape | Billiards | Minimal quantum numbers | Derived characters |
|---|---|---|---|
| Trivial | Polygon, Triangle, Annular, Polar | none | n/a |
| `Z2` | Limaçon | one sign `χ` | the one registered element gets `χ` |
| `Z2×Z2` (`D2`) | Rectangle, Stadium, Circle, Ellipse, Prosen | two independent signs `χx,χy` | `reflect_x↦χx`, `reflect_y↦χy`, `rotate_π↦χx·χy` |
| `Zn` (cyclic) | C3 (`n=3`), Star (`n=n`) | one integer sector `s∈0:n-1` | the element registered as `m`'s image gets `cis(2π·s·m/n)` |

`SymmetrySector.characters::Dict{Int,ComplexF64}`
([symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl))
already stores **one character per `sym_id`**, which is exactly enough
information to represent all four rows above once every registered `sym_id`
that participates gets its correctly-derived character — this already
happens correctly today for the `Zn` row (`_sector_character(sym::NFoldRotation,
sector) = cis(2π·sector·sym.m/sym.order)`, evaluated per matched instance
using *that instance's own* `m`) and for the `Z2`/trivial rows. **The one row
`symmetry_sector`'s current filter-by-type construction does not, by itself,
make foolproof is `Z2×Z2`**: nothing stops a caller from supplying only one
of the two axes, and nothing stops a caller from supplying `XYAxisReflection=>val`
alone — both are genuinely under-determined (two different `(χx,χy)` pairs
give the same `χ_xy = χx·χy`) — **this is not an implementation gap, it is a
mathematical fact about the group**, so it is the one case in this whole
design that legitimately keeps a validation error (see "Irreducible
restriction" below), not something to route around.

## What already fully works today (do not touch)

* `SymmetrySector`/`symmetry_sector(billiard, GenType=>value...)`
  ([symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl))
  — **no changes needed**. Its `Dict{Int,ComplexF64}` shape and its
  `_sector_character` validation (`±1` for reflections, `0:order-1` for
  rotations, each stored per matched `sym_id` using that specific instance's
  own data) are already exactly right for every row of the table above,
  *including* `Zn`'s multi-`sym_id` case — the only place v1 went wrong was
  in a **BIM-side reconstruction helper** trying to invert the Dict back
  into a single integer sector while wrongly assuming every matched entry
  had `m=1` (fixed below, see "Recovering the group's natural quantum
  numbers").
* `RealPlaneWaves(dim, billiard, sector)`
  ([realplanewaves.jl:231-241](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl#L231))
  — **no changes needed**. It already looks for `XAxisReflection`/
  `YAxisReflection` matches independently and derives `sym_x`/`sym_y`
  accordingly, which already correctly covers the entire `Z2×Z2` row (both
  axes) and the `Z2`/trivial rows for any billiard whose lone reflection is
  axis-aligned. This is already the "give both independent quantum numbers,
  not the combined element" convention this rework generalizes to BIM.

## The one irreducible restriction (not an implementation gap)

A `SymmetrySector` built from `XYAxisReflection=>val` **alone** (with no
accompanying `XAxisReflection`/`YAxisReflection` entry) cannot be resolved
into a unique 1-D representation: `χ_xy = χx·χy` is two-to-one — `(χx,χy) =
(+1,-1)` and `(-1,+1)` both give `χ_xy=-1`, and nothing in a single complex
number can distinguish them. This holds regardless of how much engineering
effort is spent on it: it is a statement about `Z2×Z2`'s character group,
not a gap in this codebase. The fix, and the only thing a user ever needs to
know, is: **always specify the individual axis reflections
(`XAxisReflection`, `YAxisReflection`, and — see below — `DiagonalReflection`/
`AntiDiagonalReflection` if a billiard ever registers those) — never the
combined element alone.** Specifying both axes is *strictly more general*
than specifying the combined element could ever be (it reaches all four
irreps of `Z2×Z2`, not just the two reachable via a single `χ_xy`), so this
restriction costs a user nothing they could otherwise have expressed.
`_resolve_bim_symmetry` (below) detects exactly this one case and raises a
precise, actionable error pointing at the fix; every other combination —
one axis alone, both axes together, both axes together with the redundant
combined element also supplied for cross-checking, a rotation sector via any
number of the billiard's registered rotation images, or no symmetry at all —
resolves automatically with no error.

## Design: a fully general BIM-side resolver reusing existing, already-validated code

The key realization that makes full generality achievable without new
`BilliardGeometry.jl` group-theory code: **`CompositeReflection`
([symmetry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl),
folding implemented in
[symmetryorbits.jl:257-317](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl#L257-L317))
is already a fully general, already-correct, already-validated finite-group
closure algorithm for *any* combination of `XAxisReflection`/
`YAxisReflection`/`XYAxisReflection`/`DiagonalReflection`/
`AntiDiagonalReflection` generators — it builds the group by BFS closure
under composition, assigns each generated element's character as the product
of the generators' characters along the path used to reach it, and **already
raises `ArgumentError("Composite reflection parities are inconsistent")`**
if two different composition paths to the same element disagree. Routing
*every* reflection-family `SymmetrySector` (one axis or many, single billiard
generator or several) through a freshly-built `CompositeReflection` therefore
gives, for free:

* A lone reflection (`LimaconBilliard`, or a partial fold of a `D2` billiard
  using only one of its two axes): `CompositeReflection([gen])` produces the
  identical two-element orbit map the dedicated single-type method would
  (verified by inspection of the loop body — a single `XAxisReflection`
  entry pushes exactly the same `_idx_reflect_x` permutation/character pair
  the dedicated method uses).
* Two independent axes (`D2` billiards' `XAxisReflection`+`YAxisReflection`,
  the case v1 rejected): `CompositeReflection([X,Y])` closes to the full
  four-element `D2` group with independent `(χx,χy)`, `χ_xy` derived
  automatically as their product — matching (and generalizing) the
  dedicated `XYAxisReflection` method exactly.
* Any future billiard registering `DiagonalReflection`/`AntiDiagonalReflection`
  (none currently do; confirmed by grep) needs **zero** new code — the same
  `CompositeReflection` path already handles them.
* A redundant/over-specified request (all three of `X`,`Y`,`XY` supplied at
  once) is automatically **validated for consistency** by the existing BFS
  loop, not silently accepted or silently ignored.

This makes `_resolve_bim_symmetry`'s reflection branch a single, uniform
code path with no per-count special-casing — the "one generator vs. several
generators" distinction that drove v1's errors disappears entirely.

### `_fold_boundary`: bridging one real API inconsistency

`symmetry_index_orbits`'s trailing-argument shape is **not** uniform across
generator types: reflections/`NFoldRotation` take `N` separate scalar
arguments (splat-compatible with a stored `Tuple`), but
`CompositeReflection` takes **one** `Vector{Complex{T}}` argument — splatting
a `Tuple` of characters into it would call the wrong method shape. Rather
than branch on this at every one of the ~10 call sites, add one small
adapter (multiple dispatch, not a conditional) alongside the other shared
BIM infrastructure in
[sweepmethods.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl):

```julia
# Bridges a solver's stored `(generator, character::Tuple)` pair to
# symmetry_index_orbits' per-generator-type trailing-argument shape: every
# type except CompositeReflection takes N separate scalar arguments
# (splat-compatible); CompositeReflection takes one Vector argument.
_fold_boundary(::Type{T}, xy, symmetry::AbsSymmetry, character::Tuple) where {T<:Real} =
    BilliardGeometry.symmetry_index_orbits(T, xy, symmetry, character...)
_fold_boundary(::Type{T}, xy, symmetry::BilliardGeometry.CompositeReflection, character::Tuple) where {T<:Real} =
    BilliardGeometry.symmetry_index_orbits(T, xy, symmetry, Complex{T}[character...])
```

Every existing `symmetry_index_orbits(T, pts.xy, solver.symmetry, solver.character...)`
call site becomes `_fold_boundary(T, pts.xy, solver.symmetry, solver.character)`.
This is the only new abstraction this plan introduces, and it exists solely
because the underlying function's own dispatch table is genuinely
non-uniform — not a speculative convenience layer.

## Recovering the group's natural quantum numbers from a `SymmetrySector`

```julia
function _resolve_bim_symmetry(billiard::AbsBilliard, sector::SymmetrySector)
    isempty(sector.characters) && return nothing, ()   # trivial/no symmetry requested
    gens = [BilliardGeometry.symmetry_of(billiard.symmetries, id) => char for (id, char) in sector.characters]
    return _resolve_bim_symmetry(gens)
end

# Reflection family: always fold via the generated group's full closure,
# reusing BilliardGeometry.jl's existing CompositeReflection algorithm
# (handles 1 generator, 2 independent generators i.e. a full D2 fold, or any
# larger combination, uniformly and with automatic consistency validation).
function _resolve_bim_symmetry(gens::Vector{<:Pair{<:BilliardGeometry.AbsReflection}})
    if length(gens) == 1 && gens[1].first isa BilliardGeometry.XYAxisReflection
        throw(ArgumentError(
            "A lone XYAxisReflection character does not determine a unique " *
            "1D representation of the registered D2 symmetry group (both " *
            "(χx,χy)=(+1,-1) and (-1,+1) give the same combined character). " *
            "Specify the individual axis reflections instead, e.g. " *
            "symmetry_sector(billiard, XAxisReflection=>χx, YAxisReflection=>χy)."))
    end
    generators = BilliardGeometry.AbsReflection[g for (g,_) in gens]
    characters = ComplexF64[c for (_,c) in gens]
    return BilliardGeometry.CompositeReflection(generators), Tuple(characters)
end

# NFoldRotation family: Cn_symmetry(n) registers n-1 non-identity elements of
# the *same* cyclic group (m=1:n-1), so a "pick sector s" choice legitimately
# produces several sym_ids at once. Recover s from each matched entry using
# *that entry's own* m (character = cis(2π·s·m/order)) and cross-check all
# matched entries agree, instead of assuming m=1 for whichever entry happens
# to be inspected first.
function _resolve_bim_symmetry(gens::Vector{<:Pair{BilliardGeometry.NFoldRotation}})
    order = first(gens).first.order
    all(g -> g.first.order == order, gens) || throw(ArgumentError("Mismatched NFoldRotation orders within one SymmetrySector"))
    sectors = [mod(round(Int, angle(char)*order/(2π*gen.m)), order) for (gen,char) in gens]
    all(==(sectors[1]), sectors) || throw(ArgumentError("Inconsistent rotation sector recovered across sym_ids: $sectors"))
    return first(gens).first, (sectors[1],)
end
```

No dihedral/mixed reflection+rotation method is provided — see "Explicit,
deliberate non-goal" below for why that is not a gap in the same sense as
the one just closed.

## Sub-steps

### 16.1 — Empirical audit (no code changes)

Confirm, with a short scratch script against `StadiumBilliard`, that
`fundamental_size(orbits)`/`orbit_size(orbits)` are identical for the
trivial character vs. an explicit non-trivial character for (a) a lone
`YAxisReflection`, (b) `CompositeReflection([XAxisReflection(),
YAxisReflection()])`, and (c) an `NFoldRotation` sector on `C3Billiard` — in
every case only the resulting matrix *values* (`fund_to_scale`) should
differ, never `fund_to_full`'s orbit *membership*. This generalizes v1's
audit to also cover the `CompositeReflection` path (its BFS closure
enumerates the same permutation structure regardless of character values,
so trivial characters never trigger the "parities are inconsistent" check).
If confirmed, `boundary_matrix_size`/`estimate_rmin_rmax`'s symmetric branch
do **not** need `character`/`_fold_boundary` threaded through — only
`construct_matrices`'s reduced branch and `_bim_normal_derivative` do.

### 16.2 — Add the `character` field

`DoubleLayerPotentialSolver`, `CombinedFieldIntegralEquationSolver`,
`CompositeBIMSolver`: add `character::Ch` (new type parameter, `Ch<:Tuple`,
default `()`) alongside the existing `symmetry` field, in every struct and
constructor (keyword `character::Tuple=()`). `()` ⇒ trivial representation,
byte-for-byte unchanged behavior for every existing call site (only real one:
[solvertests.jl:325](/home/clozej/.julia/dev/QuantumBilliards.jl/test/solvertests.jl#L325)).
Update the `@debug` logs
([dlp.jl:305](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl#L305),
[cfie.jl:373](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/cfie.jl#L373),
[compositebim.jl:439](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl#L439))
to also print `character=solver.character`. `CompositeBIMSolver`'s
constructor validates/propagates `character` the same way it already does
`symmetry` (shared across all component solvers; equality check extended to
compare both fields — see the note on this check's pre-existing limits under
"Tests & user verification").

### 16.3 — Add `_fold_boundary` and thread it through every call site

Add the two-method `_fold_boundary` adapter (above) to
[sweepmethods.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl).
Per 16.1's findings: at minimum, replace
`symmetry_index_orbits(T, pts.xy, solver.symmetry)` with `_fold_boundary(T,
pts.xy, solver.symmetry, solver.character)` in `construct_matrices`'s
reduced branch in dlp.jl, cfie.jl, compositebim.jl, and in
`_bim_normal_derivative`
([sweepmethods.jl:240](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl#L240)).
Leave `boundary_matrix_size`/`estimate_rmin_rmax` on the plain 3-argument
call if 16.1 confirms character-independence (comment inline why, so a
future reader doesn't "fix" an apparent inconsistency). `BeynSolver`/
`ExpandedBIMSolver` need **no** file changes (they read `cs.symmetry`/
`cs.character` off their wrapped kernel automatically) — only re-run their
existing tests.

### 16.4 — `_resolve_bim_symmetry` (both methods, above)

Add to
[QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl).
No changes to `SymmetrySector`/`symmetry_sector` themselves are needed (see
"What already fully works today").

### 16.5 — New outer constructors

```julia
function DoubleLayerPotentialSolver(pts_scaling_factor, billiard::AbsBilliard, sector::SymmetrySector; kwargs...)
    generator, character = _resolve_bim_symmetry(billiard, sector)
    return DoubleLayerPotentialSolver(pts_scaling_factor; symmetry=generator, character=character, kwargs...)
end
```

Same overload for `CombinedFieldIntegralEquationSolver`. No new
`CompositeBIMSolver`-level constructor is needed — see the note on its
equality check under "Tests & user verification" for the one thing to
verify when composing sector-built component solvers.

### 16.6 — Close 10.9 Gap F, document the one restriction

Docstring update on both new constructor overloads: constructing via
`(billiard, sector)` is now the recommended, validated path (`symmetry_of`
throws immediately if `sector`'s `sym_id` doesn't belong to
`billiard.symmetries`), and it supports every representation
`RealPlaneWaves(dim, billiard, sector)` supports for reflection symmetries,
plus rotation sectors, which `RealPlaneWaves` does not (see "Explicit,
deliberate non-goal"). Document the one restriction (lone `XYAxisReflection`)
directly in the docstring with the fix (name both axes), not just in this
plan file. The old bare-`AbsSymmetry`/`character=` keyword constructor is
kept, documented as unvalidated/expert-use (matches `RealPlaneWaves`'s
existing precedent from Step 15).

## Files touched

| Sub-step | File |
|---|---|
| 16.2 | [QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl) |
| 16.2 | [QuantumBilliards.jl/src/solvers/sweepmethods/cfie.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/cfie.jl) |
| 16.2 | [QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl) |
| 16.3 | [QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl) |
| 16.3 (verification only) | [QuantumBilliards.jl/src/solvers/boundarypoints.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/boundarypoints.jl), [QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl), [QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl) |
| 16.4, 16.5, 16.6 | [QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl) |

Not touched: `BilliardGeometry.jl` (this rework's central finding —
`CompositeReflection`'s existing closure algorithm already provides full
generality with zero new geometry-package code; no change to
`symmetry_index_orbits`, `SymmetryRegistry`, `symmetry_of`, `SymmetryWall`);
`RealPlaneWaves`/`basis/planewaves/realplanewaves.jl` (already fully general
for the reflection family, see "What already fully works today");
`QBPlotting.jl` (plotting only ever consumes already-folded `(pts,u)`/
wavefunction grids, never a solver's `symmetry`/`character` fields).

## Explicit, deliberate non-goal: rotation representations in `RealPlaneWaves`

For true parity, a `SymmetrySector` built with an `NFoldRotation` sector
should also be usable by a basis solver on `C3Billiard`/`StarBilliard`. It
currently cannot be: `RealPlaneWaves`
([realplanewaves.jl:231-241](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl#L231))
only ever inspects `XAxisReflection`/`YAxisReflection` matches, and
`CornerAdaptedFourierBessel`
([corneradapted.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/fourierbessel/corneradapted.jl))
has its own, separate, untyped `symmetry::Union{Vector{Any},Nothing}` field
entirely disconnected from `SymmetrySector`. Representing a genuine
`Zn`-rotation sector in a plane-wave basis requires restricting the angular
Fourier content of the basis modulo `n` — a real basis-construction feature,
not a solver-migration wiring task, and one this plan has not verified the
feasibility or exact numerical form of. **This is explicitly out of scope
for Step 16** (which is a BIM-solver migration) and is flagged here, plainly,
as the one remaining basis/BIM asymmetry after this step — recommended as a
follow-up (`RealPlaneWaves` or `CornerAdaptedFourierBessel` rotation-sector
support) rather than invented speculatively here. It does not block Step 16:
`DoubleLayerPotentialSolver(pts, billiard, symmetry_sector(billiard,
NFoldRotation=>s))` is fully supported by this plan on its own.

Similarly out of scope, but for a different (non-abelian-group) reason: no
billiard in the current catalogue registers both a reflection and a rotation
generator together (a dihedral group), and no part of this codebase —
`RealPlaneWaves`, `CornerAdaptedFourierBessel`, or `symmetry_index_orbits` —
has any 2-dimensional-irrep machinery, which non-abelian dihedral groups
require for `n>2`. This is not a gap introduced or left open by this step;
it is a class of symmetry no billiard currently needs and no part of the
existing framework (1-D characters only, everywhere) is built to express.
Flagged explicitly rather than silently ignored, with no action item unless
a concrete billiard/test case requires it.

## Tests & user verification

Before calling any sub-step done, hand-verify (or short scratch script):

* 16.1's audit result (character-independence of orbit sizing), including
  the `CompositeReflection` case — record the finding inline regardless of
  outcome.
* Reconstruct the existing
  [solvertests.jl:325](/home/clozej/.julia/dev/QuantumBilliards.jl/test/solvertests.jl#L325)
  fixture after 16.2/16.3 and confirm byte-for-byte unchanged output
  (`character=()` default) — the core regression check.
* **`Z2×Z2` full generality (the case v1 could not express at all).** On
  `StadiumBilliard`, construct
  `DoubleLayerPotentialSolver(10.0, billiard, symmetry_sector(billiard,
  XAxisReflection=>-1, YAxisReflection=>-1))` and confirm `solver.symmetry
  isa CompositeReflection` with two reflection generators. Cross-check its
  assembled matrix against the *existing* dedicated-type path
  `DoubleLayerPotentialSolver(10.0; symmetry=BilliardGeometry.XYAxisReflection(),
  character=(ComplexF64(-1), ComplexF64(-1)))` — both must produce an
  identical Fredholm matrix (same group, same characters, two different code
  paths to reach it). Repeat for the three other sign combinations
  `(+,+),(+,-),(-,+)`, and for a partial fold using only one axis (must
  match the pre-existing single-axis behavior unchanged).
* **`Zn` multi-`sym_id` resolution.** On `C3Billiard`, confirm
  `symmetry_sector(billiard, NFoldRotation=>2)` produces a two-entry
  `SymmetrySector` and `_resolve_bim_symmetry` recovers `sector=2`
  consistently regardless of `Dict` iteration order (not `sector=1`, the
  wrong answer a naive "assume `m=1`" inversion would give). Confirm the
  resulting solver's spectrum matches a manually-constructed
  `DoubleLayerPotentialSolver(10.0; symmetry=BilliardGeometry.NFoldRotation(3,1),
  character=(2,))` exactly. Repeat for `StarBilliard`.
* **New numerical capability check.** For at least one non-trivial
  representation from each row of the group-shape table (reflection,
  `Z2×Z2`, `Zn`), confirm the symmetry-reduced spectrum is a strict subset
  of the unreduced (`symmetry=nothing`) full-boundary spectrum on the same
  billiard/k-range.
* **The one remaining, intentional error case.** Confirm
  `symmetry_sector(rectangle_billiard, XYAxisReflection=>-1)` fed into
  `DoubleLayerPotentialSolver(10.0, billiard, sector)` raises the precise
  `ArgumentError` from `_resolve_bim_symmetry`'s reflection branch, with a
  message that names the fix (specify both axes) — not a generic
  `MethodError` or a silently wrong matrix.
* **`CompositeBIMSolver` equality-check caveat.** The extended `cs.symmetry
  == symmetry && cs.character == character` check compares `CompositeReflection`
  objects (and, inside them, the wrapped generator instances) by structural
  equality, which includes each generator's `sym_id`. Confirm this still
  passes for two component solvers built from `(billiard, sector)` where
  both components share the *same* billiard's registry (the existing/only
  tested `CompositeBIMSolver` usage pattern); if a future multiply-connected
  billiard ever gives its components independently-numbered registries for
  the same logical symmetry, this equality check would need revisiting —
  note this as a known limitation of the pre-existing check, not a new one
  introduced here.
* Repeat the trivial-character regression check for
  `CombinedFieldIntegralEquationSolver` and re-run `BeynSolver`/
  `ExpandedBIMSolver` on the existing trivial-character fixture (no code
  change expected there) to confirm no regression.

As with every other step in this plan, do not write `Test.jl` files
directly — once the implementation and the manual checks above look right,
ask the user to invoke the **Julia Test Writer** subagent.

## Plotting updates

None expected, by the same reasoning as Step 15 (`QBPlotting.jl` never reads
`solver.symmetry`/`solver.character` directly).
