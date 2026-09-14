Date produced: 2026-09-14

# Audit Report — Step bg-01: Core Geometry Primitives & Abstract Types (BilliardGeometry.jl)

Read-only pass. No files were edited.

## 1. Dead code (zero call sites anywhere in the 8-package workspace)

| Function | File | Status |
|---|---|---|
| `construct_arc_length_interpolation` | [arclength.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/arclength.jl#L21) | **New finding.** Exported, but no call site anywhere (main or `-develop`, any package). |
| `point_curve_parameter` | [utils.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/utils.jl#L83) | **New finding.** Exported, zero call sites (also unused in `-develop`). |
| `invert_curve_nelder_mead` | [inversions.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/inversions.jl#L126) | **New finding.** Not exported, zero call sites, not even reachable via `invert_curve`'s `:method` dispatch (only `:auto/:roots/:optim/:grid_optim` are wired). |
| `pb_coords`, `fundamental_pb_coords`, `get_pb_curve` | [poincarebirkhoff.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/poincarebirkhoff.jl#L6-L36) | **New finding.** All exported (`pb_coords`) or internal helpers, but zero external call sites. Only `pb_sectors` (same file) is actually consumed, by [QBPlotting.jl/husimiplotting.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/husimiplotting.jl#L9) and [boundaryfunctionsplotting.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/boundaryfunctionsplotting.jl#L14). |
| `_boundary_components` | [boundarycomponents.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarycomponents.jl#L9) | **New finding.** Only referenced from its own `test/runtests.jl` testset; no `src/` consumer in either package (contrast with `-develop`'s `QuantumBilliards-develop`, where an analogous helper is used heavily in `cfie_kress.jl`/`dlp_kress.jl`/`wavefunctions.jl` — that usage was apparently never ported over to main's `dlp.jl`/`cfie.jl`, which instead call the public `boundary_components(billiard)` directly). |
| `get_curve`, `get_domain`, `update_boundary_condition` | [boundarytypes.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl#L74-L136) | Zero call sites in either main or `-develop` — pre-existing, not a regression. Plausibly intentional public introspection API rather than true dead code; flagging per the checklist, not recommending removal. |

Not previously tracked in either memory file — all of the above are new observations.

## 2. Missing wiring

None found. Every struct/function defined in these files is exported via [BilliardGeometry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/BilliardGeometry.jl) → [geometry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/geometry.jl)'s per-include export lines (cross-checked `PoincareBirkhoff`, `SymmetryWall`/`SpecularReflection`/`Transparent`/`PeriodicX`/`QuantumSolverIgnore`, `tangent_vec`/`normal_vec`/`curvature`, `area`/`fundamental_area`/`corner_angles`, etc. — all present). No abstract type in the module file (`AbsCurve`, `AbsBilliard`, `AbsDomain`, etc.) has an unimplemented documented API within this file set.

## 3. Missing implementations

- No leftover `error(...)`/TODO/"not implemented" stubs found (only legitimate `error`/`@warn` in [inversions.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/inversions.jl#L25-L35), both intentional method-dispatch guards).
- **Docstring backfill (item G, tracked as "still open/low-priority" in [geometry-audit-10.9.md](/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/geometry-audit-10.9.md)): confirmed still open, partially done.** [arclength.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/arclength.jl) and [poincarebirkhoff.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/poincarebirkhoff.jl) have **zero** docstrings; [utils.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/utils.jl) has zero (aside from the `_boundary_components`-style ones added later in `boundarycomponents.jl`); [boundarytypes.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl) is partial (`SymmetryWall`, `genus`, `boundary_components` documented; `SpecularReflection`/`Transparent`/`PeriodicX`/`QuantumSolverIgnore`/`get_boundary_curves`/`get_all_domains`/`get_domain`/`get_all_curves`/`get_curve`/`update_boundary_condition` are not). By contrast, `inversions.jl`, `curvederivatives.jl` (Gap 4 addition), `boundarycomponents.jl`, and `area.jl` (12.5 addition) already have good docstring coverage. Also missing: docstrings on the module-file abstract types themselves (`AbsCurve`, `AbsBilliard`, etc. have no doc blocks at all) and on `geometry.jl`'s `is_inside`/`domain_fun`/`domain_gradient_vector`.

## 4. Unstable APIs

- **`get_pb_curve`** ([poincarebirkhoff.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/poincarebirkhoff.jl#L15-L21)): falls off the end of the loop with an implicit `nothing` return if no matching `(domain_id, segment_id)` is found, vs. a `Tuple` on success — genuine return-type instability. Low practical severity since `get_pb_curve`/`pb_coords` are dead code (§1), but worth flagging as it's a footgun if ever revived.
- **`area(curves::AbstractVector{<:AbsCurve})`** ([area.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/area.jl#L52)): `isempty(curves) && return 0.0` returns a hardcoded `Float64` regardless of the curves' element type `T` (e.g. would silently widen a `Float32`-typed empty-curve computation to `Float64`). **New finding**, not in memory files. Fix: `return zero(T)`, computed the same way the non-empty branch derives `T`.
- **`curvature`/`tangent`/`tangent_2` generic array methods** ([curvederivatives.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/curvederivatives.jl#L1-L21,#L136-L154)): dispatch is over `crv::AbsCurve` (abstract) but Julia specializes per concrete `typeof(crv)` at each call site, so these do **not** actually fall back to `Any` in practice — verified no instability here despite the abstract annotation (false lead, ruled out).
- No non-concrete struct fields found in `poincarebirkhoff.jl`/`inversions.jl` (`PoincareBirkhoff{T}` is properly parametric).

## 5. Vulnerabilities

- **`point_curve_parameter`** ([utils.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/utils.jl#L83-L98)): if `roots_y` is empty, `return roots_y[1]` throws an unguarded `BoundsError` instead of a clear error; if `length(roots_y)>1` and no `(t_x,t_y)` pair matches, falls through returning `nothing` silently. **New finding** — low severity given it's dead code (§1), but should either be fixed or removed rather than left as a latent footgun.
- **`get_curve`/`get_domain`** ([boundarytypes.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl#L74-L121)): same silent-`nothing`-on-no-match pattern as `get_pb_curve`. Pre-existing in `-develop` too (not a regression), but still worth a `throw(ArgumentError(...))` instead of silent `nothing` given they process caller-supplied `domain_id`/`segment_id`.
- `connect_curves`/`is_closed`/`is_connected` ([utils.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/utils.jl#L8-L35)): confirmed the Step 11 multi-component fix from [known-bugs.md](/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/known-bugs.md) is **still present, no regression** — the greedy "peel off one chain at a time" loop is unchanged. No new bounds issues found in this function or `_global_t_to_segment_u`/`_eval_composite_geom_global_t` (both properly clamp indices).

## 6. Consolidation opportunities

- **New finding**: `arclength.jl`'s `_arc_length_integrand` ([arclength.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/arclength.jl#L1-L4)) re-differentiates `curve(crv,t)` via `ForwardDiff.derivative` on every `quadgk` node, instead of reusing the already-analytic `tangent(crv,t)` from `curvederivatives.jl` (which is closed-form for `LineSegment`/`CircleSegment`/`FourierCoeffPolarSegment`, and only falls back to `ForwardDiff` for the generic `PolarSegment`). This duplicates the derivative logic and pays a redundant nested-AD cost inside every `arc_length`/`area`-driving quadrature for the three analytic curve types. Suggest routing `_arc_length_integrand` through `tangent(crv,t)`.
- **New finding**: `_polar_radius_derivative`/`_polar_radius_derivative_2` ([curvederivatives.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/curvederivatives.jl#L52-L72)) allocate two new slice arrays (`polar.coef[1:2:end]`, `polar.coef[2:2:end]`) on *every* call, and these are called from `tangent`/`tangent_2`, which in turn feed `arc_length`/`area` quadrature and per-point boundary evaluation in solvers — a real allocation-in-hot-path violation of the project's memory-efficiency convention. Suggest `@view` instead of slicing, or precomputing the split coefficients once at `FourierCoeffPolarSegment` construction time.
- No duplication found between `utils.jl` and `LinearAlgebra`/`StaticArrays` builtins (`angle(a,b)` legitimately extends `Base.angle` via the module's implicit `using Base`, not a namespace collision).

## 7. Export audit

No missing exports found; no exported-but-clearly-premature symbols found beyond the dead-code items already listed in §1 (which remain exported despite being unused — a soft argument for either wiring them up or removing the export, not a hard bug).

## 8. Area-specific check

**Confirmed complete migration**, matching [12.5-spectral-api-consolidation.md](/home/clozej/Programs/QuantumBilliardsTests/scratchpad/migration-plan/12.5-spectral-api-consolidation.md):
- `area`, `fundamental_area`, `symmetry_reduction_factor`, `maximal_symmetry`, `corner_angles` all live in [area.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/area.jl), exported (`area`/`fundamental_area`/`corner_angles`) or intentionally unexported (`symmetry_reduction_factor`/`maximal_symmetry`) exactly as planned.
- [QuantumBilliards.jl/src/QuantumBilliards.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/QuantumBilliards.jl) has **no** re-definition or re-export of `area`/`fundamental_area`/`corner_angles` — nothing left half-migrated.
- `corner_angles(billiard)` is actively consumed by `k_at_state`/`state_at_k`/`spectral_density`/`k_range_for_states` in [QuantumBilliards.jl/src/spectra/unfolding.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/unfolding.jl#L44-L88), confirming it's load-bearing for every corner-having billiard's Weyl-law/`compute_spectrum` path, not an orphaned accessor.
- `maximal_symmetry`'s use of `billiard.symmetries` (now a `SymmetryRegistry`, per the Step 15 refactor in [known-bugs.md](/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/known-bugs.md)) still works correctly — `SymmetryRegistry <: AbstractVector{AbsSymmetry}` supports `isempty`/`findmax`/indexing transparently, so `area.jl`'s dependency on the old `Vector{AbsSymmetry}`-style API was **not broken** by that later refactor. Confirmed no regression, even though this cross-checks an out-of-scope file (`symmetryregistry.jl`).

## Not re-reported (already tracked, confirmed fixed/no regression)
- Gap 1–5 (symmetryorbits/fullboundary/`tangent_vec`+`normal_vec`+`curvature`/`PolarSegment` rename) — all confirmed still fixed, no regressions, per direct inspection of `curvederivatives.jl` and `boundarytypes.jl`.
- `connect_curves` multi-component fix (Step 11) — confirmed still present, no regression.
