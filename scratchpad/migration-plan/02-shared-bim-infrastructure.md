# Step 2 — Shared boundary-integral infrastructure (`BoundaryPoints` extension, Kress matrices, geometry caches, symmetry orbits)

## Goal

Land every piece of shared, solver-independent infrastructure that
`DoubleLayerPotentialSolver`/`CombinedFieldIntegralEquationSolver`/
`CompositeBIMSolver`/`BeynSolver`/`ExpandedBIMSolver` (Steps 3 and 6–9) all depend
on. This step produces **no working solver** by itself — it only makes the
Step 3 `DoubleLayerPotentialSolver` body implementable without also having to
invent boundary-parametrization plumbing from scratch. Get this step reviewed
and loading cleanly before starting Step 3.

## Preconditions

Step 1 complete (not a hard dependency, just do it in order).

## Source index

All four pieces below live in one file in `-develop`:
`QuantumBilliards-develop/src/solvers/boundary_points.jl`. Read it in full
(it was already read in full during planning — ~900 lines) before starting.
It contains, in order: the extended `BoundaryPoints` struct + constructors,
`boundary_coords` (full-boundary sampling, **already ported to main
unchanged** — do not re-port, main's `boundary_coords`/
`get_boundary_curves_with_ignored`/`_determine_bp_sizes` are already real,
working code), `points_in_billiard`, `kress_R_even!`/`kress_R_odd!`/`kress_R!`,
`BoundaryPanelArrays`/`_boundary_panel_arrays_cache`, `_boundary_components`,
`component_lengths`, corner-junction detection
(`_unit_tangent_at_start`/`_end`, `_junction_angle`, `_is_true_corner`,
`_component_corner_locations`, `print_component_junctions`),
`component_normals`, `flatten_boundary_components`/`flatten_boundary_ds`,
`_global_t_to_segment_u`/`_eval_composite_geom_global_t`, and
`BoundaryGeomCache`/`boundary_geom_cache`.

Symmetry-orbit folding usage sites (to understand the contract
`symmetry_index_orbits` must satisfy) are inside
`QuantumBilliards-develop/src/solvers/sweepmethods/dlp/dlp_kress.jl` and
`cfie/cfie_kress.jl` (search for `SymmetryOrbitMap`/`orbits`/`symmetry_index_orbits`
in both — this step only needs to understand the *call contract*, Steps 3 and 6
implement the actual matrix folding). Symmetry types themselves
(`XAxisReflection`, `YAxisReflection`, `XYAxisReflection`, `NFoldRotation`) are
defined identically in `BilliardGeometry.jl/src/geometry/symmetry.jl` already
(confirmed identical to `-develop`) — no port needed for the types, only new
methods dispatching on them.

## Files touched in main

### 2.1 — `BoundaryPoints` extension (`QuantumBilliards.jl/src/solvers/boundarypoints.jl`)

Main's current struct only has `xy, normal, kappa, s, ds, rdotn, w_vs, w_dm,
xy_int` (used by `VerginiSaracenoSolver`/`DecompositionMethodSolver`/PSM).
`-develop`'s struct has `xy, normal, s, ds, w, w_n, curvature, xy_int,
shift_x, shift_y, tangent, tangent_2, ts, tphys, ws, ws_der, compid,
is_periodic, xL, xR, tL, tR`.

**Do not rename or remove any existing field** (`kappa`, `rdotn`, `w_vs`,
`w_dm` stay exactly as-is — every basis solver from Step 1 depends on them
verbatim). **Add** the fields that exist in `-develop` but not in main:
`w`, `w_n`, `curvature`, `shift_x`, `shift_y`, `tangent`, `tangent_2`, `ts`,
`tphys`, `ws`, `ws_der`, `compid`, `is_periodic`, `xL`, `xR`, `tL`, `tR`. Keep
the exact field names from `-develop` (not renamed) since the ported
`dlp_kress.jl`/`cfie_kress.jl` bodies in Steps 3 and 6 reference `pts.tangent`,
`pts.ts`, etc. directly — renaming here just to "clean up" would force
re-deriving every downstream reference and risks transcription bugs in
numerically sensitive code (explicitly against the fidelity rule in
[00-index.md](00-index.md)).

1. Update the inner constructor's length-validation loop to cover every new
   vector-valued field (mirror `-develop`'s validation list exactly:
   `normal, s, ds, w, w_n, curvature, tangent, tangent_2, ts, tphys, ws,
   ws_der` all either empty or `length == length(xy)`; existing fields
   `kappa, rdotn, w_vs, w_dm` keep their existing validation entries too).
2. Update the keyword convenience constructor
   `BoundaryPoints(xy; normal=..., kappa=..., ...)` to accept the new fields
   as keywords with the same defaults `-develop` uses (`shift_x=zero(T)`,
   `shift_y=zero(T)`, `compid=1`, `is_periodic=true`,
   `xL=xR=tL=tR=SVector{2,T}(zero(T),zero(T))`, everything else empty
   vectors).
3. Port the parametrized constructor overload verbatim:
   `BoundaryPoints(xy, tangent, tangent_2, ts, tphys, ws, ws_der, s, ds,
   compid, is_periodic, xL, xR, tL, tR)` — computes `normal` from `tangent` via
   `n = (t_y, -t_x)/|t|` and calls through to the keyword constructor. This is
   the constructor Steps 3 and 6's Kress-graded `evaluate_points` will call.
4. Port the multi-component free functions that operate on
   `Vector{BoundaryPoints{T}}` (needed by `CompositeBIMSolver` in Step 7 and by
   `GlobalCornerGrading` in Steps 3 and 6 for composite boundaries): overload
   `boundary_matrix_size(pts::Vector{BoundaryPoints{T}})`,
   `boundary_s(pts::Vector{BoundaryPoints{T}})`,
   `component_offsets(pts::Vector{BoundaryPoints{T}})` (and the existing
   single-component `boundary_matrix_size(pts::BoundaryPoints)` — check main
   doesn't already have a conflicting single-argument version before adding;
   if `boundary_matrix_size(pts::BoundaryPoints) = length(pts.xy)` isn't
   defined yet, add it too, matching the placeholder solver files' existing
   calls to it).
5. Port `points_in_billiard(pts, billiard) = BilliardGeometry.is_inside(billiard, pts)`
   if not already present (check first — main may already have an equivalent
   used elsewhere, e.g. by `random_interior_points` from Step 1).

### 2.2 — Kress correction matrices → `BilliardGeometry.jl/src/quadrature/kressgrading.jl` (new file)

Port `kress_R_even!`, `kress_R_odd!`, `kress_R!` verbatim (FFT-based circulant
construction — pure numeric kernel, copy the arithmetic exactly per the
fidelity rule). This needs `FFTW` — confirm `BilliardGeometry.jl`'s `Project.toml`
either already depends on it or add the dependency (check first; if absent,
flag to the user before adding a new dependency per the mode's constraint on
new external dependencies — `FFTW` is already a `QuantumBilliards.jl`
dependency, so this is "moving" a dependency's usage site into
`BilliardGeometry.jl`, not introducing a new one to the ecosystem, but still
confirm before editing `Project.toml`).

Also port the corner/grading-map helpers that build on top of `kress_R!` for
the graded (non-uniform) case: search
`QuantumBilliards-develop/src/solvers/sweepmethods/dlp/dlp_kress.jl` and
`cfie/cfie_kress.jl` for the grading-map construction functions referenced but
not shown in `boundary_points.jl` (e.g. a Kress `w(σ)` grading transform
builder, likely named something like `kress_grading_map`/`build_grading` —
locate the exact name by reading past the `DLP_kress_global_corners` struct
definition in `dlp_kress.jl`, continuing where the read in Step 3's source
index leaves off) and port those alongside `kress_R!` in this same file,
since they're boundary-parametrization utilities, not DLP/CFIE-specific.

Export `kress_R!`, `kress_R_even!`, `kress_R_odd!` and the grading-map
builder(s) from `BilliardGeometry.jl`.

### 2.3 — Boundary geometry caches → `BilliardGeometry.jl/src/quadrature/boundarygeomcache.jl` (new file)

Port verbatim: `BoundaryPanelArrays`/`_boundary_panel_arrays_cache`,
`_boundary_components`, `component_lengths`, `_unit_tangent_at_start`/`_end`,
`_junction_angle`, `_is_true_corner`, `_component_corner_locations`,
`print_component_junctions`, `component_normals`,
`flatten_boundary_components`/`flatten_boundary_ds`,
`_global_t_to_segment_u`/`_eval_composite_geom_global_t`,
`BoundaryGeomCache`/`boundary_geom_cache`. These operate on
`QuantumBilliards.BoundaryPoints`/`BilliardGeometry.AbsCurve`, so this file
needs `BoundaryPoints` visible — since `BoundaryPoints` lives in
`QuantumBilliards.jl` (not `BilliardGeometry.jl`), and `BilliardGeometry.jl`
is a *dependency* of `QuantumBilliards.jl` (not the reverse), **this file
cannot live in `BilliardGeometry.jl` as originally sketched in
[../BIM-solver-migration-plan.md](../BIM-solver-migration-plan.md) §6.3**.
Resolve this dependency-direction conflict before porting:

* Move `BoundaryGeomCache`/`BoundaryPanelArrays`/`boundary_geom_cache`/
  `flatten_boundary_components`/`flatten_boundary_ds` into
  `QuantumBilliards.jl/src/solvers/boundarygeomcache.jl` instead (new file,
  included from `QuantumBilliards.jl` right after `boundarypoints.jl`) since
  they take/return `BoundaryPoints`.
  * Only the pieces that operate purely on `AbsCurve`/geometry data with **no**
  `BoundaryPoints` argument (`component_lengths`, the tangent/junction/corner
  helpers, `print_component_junctions`, `_boundary_components`) belong in
  `BilliardGeometry.jl` (new file
  `BilliardGeometry.jl/src/geometry/boundarycomponents.jl`, included from
  `geometry/geometry.jl`), since those only need curve/domain types.
* Update this step's own guidance (and flag it back to the user) that this is
  a deviation from the original placement sketch in the Step 1 plan, made
  necessary by the package dependency direction
  (`QuantumBilliards.jl` depends on `BilliardGeometry.jl`, not vice versa).

### 2.4 — `symmetry_index_orbits` concrete methods (`BilliardGeometry.jl/src/geometry/symmetryorbits.jl`)

`SymmetryOrbitMap` and the two methodless generic functions already exist in
this file (Step 1 scaffold). Add concrete methods:

```julia
symmetry_node_multiple(::XAxisReflection) = 2
symmetry_node_multiple(::YAxisReflection) = 2
symmetry_node_multiple(::XYAxisReflection) = 4
symmetry_node_multiple(sym::NFoldRotation) = sym.order
```

(confirm these multiplicities by re-deriving from how `-develop` actually uses
symmetry to fold a periodic boundary — search
`QuantumBilliards-develop/src/solvers/sweepmethods/{dlp,cfie}/*.jl` for where
node counts are divided/checked against symmetry before assuming the above
values are exactly right; `XYAxisReflection` combines two reflections so
should fold the boundary into a quarter, i.e. multiple `4`, but verify against
actual `-develop` logic rather than by inspection alone).

For `symmetry_index_orbits(::Type{T}, xy, symmetry)`: this must, for a fully
discretized periodic boundary `xy` known to be invariant under `symmetry`,
identify which nodes map onto which under `apply_symmetry(symmetry, ·)` (already
defined in `BilliardGeometry.jl/src/geometry/symmetry.jl`), pick one
representative per orbit (`fundamental_indices`), record `orbit_of[i]` for
every full-boundary index, and the irrep `phase` factor (trivial
representation ⇒ `phase .= one(Complex{T})`, since Step 1's docstring only
requires the trivial-representation case be supported first — check whether
`-develop` ever uses a non-trivial phase for BIM folding, e.g. an odd
irrep of a reflection changing the density's sign, before hardcoding
`phase = 1`; if `-develop`'s DLP/CFIE Kress code only ever folds boundary
*points* without an odd/even irrep split for the unknown density itself, note
that explicitly and keep `phase` trivial for now, deferring odd/even BIM
symmetry sectors to a future step).

Implementation approach: for each full-boundary node `i` not yet assigned to
an orbit, apply the symmetry to `xy[i]`, find the nearest full-boundary node
to the transformed point (within a small tolerance — this is a discretized
periodic sample, not an exact map, so use nearest-neighbor matching, not exact
equality), and union them into the same orbit; the first node encountered per
orbit becomes `fundamental_indices`. This is a modest new implementation
(`-develop` doesn't do this exact operation as a separate reusable function,
since it works directly with `billiard`-level fundamental-domain restriction
rather than post-hoc orbit folding of a full discretization — read
`-develop`'s DLP/CFIE Kress symmetry handling carefully to confirm this before
writing new code, and prefer adapting whatever equivalent logic already
exists there over inventing an unrelated algorithm).

## Performance & fidelity notes

* `kress_R!`/`boundary_geom_cache` are exactly the low-level numeric kernels
  the fidelity rule protects — port arithmetic verbatim, only adjust field
  access to match the extended `BoundaryPoints`.
* `symmetry_index_orbits` is new code (no direct `-develop` equivalent
  function to copy), so ordinary type-stability/allocation conventions from
  the mode instructions apply in full (preallocate `orbit_of`/`phase`,
  `@inbounds` the nearest-neighbor loop, no `push!` inside it — use a
  preallocated `Vector{Int}(undef, n)` for `fundamental_indices` construction
  via a `sizehint!`ed temporary or a two-pass count-then-fill instead).

## Tests & user verification

1. `get_errors` on every new/edited file plus `BilliardGeometry.jl` and
   `QuantumBilliards.jl` top-level module files — must load cleanly.
2. Manually construct a `BoundaryPoints` via the new parametrized constructor
   with a small synthetic `tangent`/`ts` set and confirm the computed `normal`
   matches a hand-computed value.
3. Manually call `boundary_geom_cache` on a `BoundaryPoints` sampled from the
   existing `StadiumBilliard`/Triangle fixtures (via the ported parametrized
   constructor, feeding in `tangent`/`tangent_2` computed from
   `BilliardGeometry.tangent`/`tangent_2` on the billiard's curves) and sanity
   check `R`/`invR`/`kappa` are finite, `R`'s diagonal is `1` (not `0`, per
   the `-develop` convention of setting the diagonal to `one(T)` before
   inverting), and `kappa` roughly matches the known curvature of a circular
   arc segment of the Stadium.
4. Manually verify `symmetry_index_orbits` on a small synthetic boundary
   point set invariant under `YAxisReflection`/`XAxisReflection` (e.g. sample
   points on a circle) — confirm `fundamental_size` is `full_size ÷
   symmetry_node_multiple(symmetry)` and every `orbit_of` entry points to a
   fundamental index whose symmetry image lands within tolerance of the
   original point.
5. This step has no independent user-facing plotting/eigenvalue behavior to
   verify yet (no working solver uses it), so there's no `QBPlotting.jl`
   change here.
6. Tell the user to invoke the **Julia Test Writer** subagent to add unit
   tests (not full solver regression tests yet) for `BoundaryPoints`'
   extended constructor, `boundary_geom_cache`, `kress_R!`'s known circulant
   structure (e.g. checking symmetry `R[i,j] == R[j,i]`-style structural
   invariants), and `symmetry_index_orbits`' orbit/phase consistency on a
   synthetic symmetric point set — these are the right level of granularity
   before any BIM solver exists to test end-to-end.

## Definition of done

* Extended `BoundaryPoints` loads cleanly, existing basis-solver fields/tests
  from Step 1 unaffected.
* `kress_R!`/`boundary_geom_cache`/`BoundaryPanelArrays` ported and
  independently verified against hand computations.
* `symmetry_index_orbits`/`symmetry_node_multiple` implemented for all four
  symmetry types and independently verified on a synthetic symmetric point
  set.
* User informed of the `BoundaryGeomCache` placement deviation (moved to
  `QuantumBilliards.jl` instead of `BilliardGeometry.jl`) and the reasoning.
* Julia Test Writer invoked for unit-level coverage of this step's new code.
