Date produced: 2026-09-14

# Consolidated Findings — BilliardGeometry.jl Audit (Steps bg-01 through bg-05)

This document consolidates and triages the five per-step findings files
produced by the `BilliardGeometry.jl` portion of the whole-ecosystem
code-quality audit (`00-index.md`). It does **not** replace those files —
each retains full detail, file/line references, and per-checklist-category
breakdowns. This document re-organizes everything by **severity/actionability**
across all five steps, dedupes cross-references between steps, and gives a
single before/after status table for every previously-tracked gap
(`geometry-audit-10.9.md`, `known-bugs.md`, migration-plan steps 11/12/15/16).

Source files consolidated:
- [bg-01-core-geometry-findings.md](bg-01-core-geometry-findings.md)
- [bg-02-symmetry-framework-findings.md](bg-02-symmetry-framework-findings.md)
- [bg-03-domains-segments-findings.md](bg-03-domains-segments-findings.md)
- [bg-04-quadrature-findings.md](bg-04-quadrature-findings.md)
- [bg-05-billiards-catalogue-findings.md](bg-05-billiards-catalogue-findings.md)

This is a **read-only synthesis** — no source files were edited to produce
it. Per the plan's confirmed convention, promotion into
`/home/clozej/.julia/dev/BilliardGeometry.jl/memories/repo/` is a separate,
later step, done once the full 16-step ecosystem audit (including the
`QuantumBilliards.jl`/`qb-*` steps) is complete — not done here.

---

## 1. Critical: broken symmetry / crashes / silently wrong results

These directly answer the user's original question ("find any missing
implementations for symmetries on billiards that don't work") and are the
highest-priority items in the whole audit.

| # | Finding | Location | Root cause | Step |
|---|---|---|---|---|
| C1 | `LimaconBilliard.full_boundary` **crashes** | [fullboundary.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/fullboundary.jl) | No `_apply_symmetry_to_curve` method exists for `LimaconSegment` — only `LineSegment`/`CircleSegment`/`FourierCoeffPolarSegment`/`PolarSegment` have one. | bg-05 |
| C2 | `MushroomBilliard.full_boundary` **silently returns a non-closed curve** (2.0-unit gap), even at default settings | [fullboundary.jl:169-178](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/fullboundary.jl#L169-L178) | `full_boundary`'s generic algorithm assumes the fundamental domain's physical curves form one connected chain. Mushroom's `triangle_dom` contributes zero physical curves, so its fundamental boundary is genuinely two disjoint chains — the generic "append all originals then all images" logic doesn't interleave them correctly. This is an **algorithmic gap in `full_boundary` itself**, not just a Mushroom-specific bug — any future billiard with a disjoint-chain fundamental domain will hit the same issue. | bg-05 |
| C3 | `apply_symmetry_pb` has **no method for `NFoldRotation`/`AbsRotation`** → guaranteed `MethodError` for `C3Billiard`/`StarBilliard` | [symmetry.jl:137](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl#L137) | Only an `AbsReflection` method exists. `pb_sectors`/`pb_coords` call it for every registered generator. Runtime-confirmed to crash both billiards. This breaks `QBPlotting.jl`'s `plot_husimi_function!` for any rotation-symmetric billiard (Husimi/Poincaré-section plotting path), while `full_boundary` (the BIM-solver path) works fine for the same billiards — **two different consumers of the same symmetry registry, only one is broken**. | bg-02, confirmed live by bg-05 |
| C4 | `apply_symmetry_pb`'s body never reads its own `sym` argument | [symmetry.jl:137](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl#L137) | Dispatches only on `sym_sector ∈ {1,2,3,4}` with hardcoded reflection formulas; only correct by coincidence of `D2_symmetry()`'s fixed registration order `[Y,XY,X]`. Not generalizable to `Cn_symmetry(n)` for any `n` — even fixing C3 wouldn't make this correct in general. | bg-02 |
| C5 | `StarBilliard(center≠(0,0))` **silently produces broken geometry** | [star.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/star.jl) | No guard, unlike Circle/Ellipse/Rectangle/Prosen. `NFoldRotation` always rotates about the true coordinate origin, not the billiard's `center`. Runtime-confirmed: off-center construction produces a 0.42-unit `full_boundary` gap. | bg-05 |
| C6 | `MushroomBilliard(origin≠[0,0])` **silently produces worse-broken geometry** (winding≈0, non-simple) | [mushroom.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/mushroom.jl) | No guard; internal coordinates mix `cx` into x-coords inconsistently with `cy`. Compounds C2. | bg-05 |
| C7 | `AnnularBilliard` under-registers its symmetry (trivial group registered despite genuine D2 geometry) | [annular.jl:27](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/annular.jl#L27) | Confirms Gap 2 (multi-ring `symmetry_index_orbits` folding) is still open — `AnnularBilliard` doesn't yet provide the fixture needed to close it. Not a crash today only because no folding is attempted. | bg-02, bg-05 |
| C8 | `CompositeBIMSolver` would **silently misfold** a multi-ring billiard's boundary points if one were ever combined with a registered symmetry | [compositebim.jl:449](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl#L449) (QuantumBilliards.jl, cross-referenced) | `symmetry_index_orbits` treats `pts.xy` as one contiguous ring regardless of component boundaries; no topology guard exists. Currently unexercised (no billiard triggers this combination) but a real latent correctness bug once C7 is fixed. | bg-02 |

**Recommended fix order**: C3/C4 (symmetry_pb rotation support) and C1 (Limaçon
`_apply_symmetry_to_curve`) are the most surgical, highest-value fixes —
each is a missing-method gap with a clear signature to add. C2 (Mushroom)
requires a genuine algorithm change to `full_boundary`'s chain-interleaving
logic and should be scoped as its own `julia-refactor-scoped` session. C5/C6
just need constructor guards (`ArgumentError` on `center ≠ (0,0)`), mirroring
the existing D2 billiards' pattern. C7/C8 are correctly deferred pending a
real symmetric multiply-connected fixture, per the original plan.

---

## 2. Type instability & precision correctness (not crashes, but violate the project's type-stability convention)

| Finding | Location | Step |
|---|---|---|
| `area(::AbstractVector{<:AbsCurve})` hardcodes `Float64` on the empty-input fallback instead of `zero(T)` | [area.jl#L52](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/area.jl#L52) | bg-01 |
| All reflection transforms (`reflect_x`/`reflect_y`/etc.) and `NFoldRotation` are hardcoded `LinearMap{SMatrix{2,2,Float64,4}}` — silently narrows/widens precision for `BigFloat`/`Float32` billiards | [symmetry.jl:3-7,179-186](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl#L3-L7) | bg-02 |
| `LineSegment`/`CircleSegment` constructors don't `promote_type` their numeric args (unlike the already-fixed `FourierCoeffPolarSegment`/`PolarSegment`) — confirmed via direct `MethodError` reproduction | [linesegment.jl#L19-L25](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/segments/linesegment.jl#L19-L25), [circlesegment.jl#L19-L28](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/segments/circlesegment.jl#L19-L28) | bg-03 |
| `Polygon`'s constructor uses only the first corner's `eltype`, not a promoted common type | [polygons.jl#L11](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/polygons.jl#L11) | bg-03 |
| `LinearNodes`/`GaussLegendreNodes` hardcoded to `Float64` (via `range(0,1.0,...)` and `gausslegendre(N)`), unlike the properly parametric `FourierNodes` | [samplers.jl#L6-L19](/home/clozej/.julia/dev/BilliardGeometry.jl/src/quadrature/samplers.jl#L6-L19) | bg-04 |
| `MushroomBilliard` hardcodes `SimpleDomain{Float64}(...)` regardless of the actual promoted type of its inputs | [mushroom.jl#L12](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/mushroom.jl#L12) | bg-05 |
| Several billiard constructors (`StadiumBilliard`, `MushroomBilliard`, `LimaconBilliard`, `TriangleBilliard`) have no `<:Real` constraint, so e.g. `StadiumBilliard(1)` (an `Int`) silently builds an `Int`-typed billiard | billiards/*.jl | bg-05 |
| `get_pb_curve` has inconsistent return type (implicit `nothing` vs. `Tuple`) | [poincarebirkhoff.jl#L15-L21](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/poincarebirkhoff.jl#L15-L21) | bg-01 (low severity — dead code) |

---

## 3. Vulnerabilities (unchecked/unvalidated inputs)

| Finding | Location | Step |
|---|---|---|
| `point_curve_parameter` throws unguarded `BoundsError` on empty result, silent `nothing` on ambiguous match | [utils.jl#L83-L98](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/utils.jl#L83-L98) | bg-01 (dead code, low severity) |
| `get_curve`/`get_domain` silently return `nothing` on no match instead of erroring | [boundarytypes.jl#L74-L121](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl#L74-L121) | bg-01 |
| No cross-check between `SymmetryWall`, `billiard.symmetries`, and a solver's chosen sector (Gap 6) | — | bg-02, confirmed still open, concrete new evidence in bg-05 (`SymmetryWall.sector_id` never read anywhere; inconsistent tagging across billiards is invisible) |
| No radius/degenerate-geometry validation anywhere in `circular.jl`/`circlesegment.jl`/`polygons.jl` — negative/zero `R` accepted silently; 1–2 corner `Polygon` builds degenerate zero-length edges | domains/segments files | bg-03 |
| `PolarSegment`'s `ForwardDiff`-based derivative path silently propagates NaN/Inf for a non-differentiable/discontinuous user `r_func` | [curvederivatives.jl#L86-L96](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/curvederivatives.jl#L86-L96) | bg-03 |
| `LimaconBilliard(a)`, `C3Billiard(a)` — no validation that the curve stays simple/non-self-intersecting (unlike `StarBilliard`'s explicit `R>abs(a)` check) | limacon.jl, c3.jl | bg-05 |
| `TriangleBilliard(gamma,chi;...)` — no validation that derived angles stay positive/valid | triangle.jl | bg-05 |

---

## 4. Dead code

| Item | Location | Status |
|---|---|---|
| `construct_arc_length_interpolation` | [arclength.jl#L21](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/arclength.jl#L21) | Exported, zero call sites |
| `point_curve_parameter` | [utils.jl#L83](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/utils.jl#L83) | Exported, zero call sites |
| `invert_curve_nelder_mead` | [inversions.jl#L126](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/inversions.jl#L126) | Unreachable via `invert_curve`'s dispatch |
| `pb_coords`, `fundamental_pb_coords`, `get_pb_curve` | [poincarebirkhoff.jl#L6-L36](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/poincarebirkhoff.jl#L6-L36) | Zero external call sites (only `pb_sectors` is used, by QBPlotting.jl) |
| `_boundary_components` | [boundarycomponents.jl#L9](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarycomponents.jl#L9) | Only used in its own test |
| `get_curve`, `get_domain`, `update_boundary_condition` | [boundarytypes.jl#L74-L136](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl#L74-L136) | Zero call sites; plausibly intentional public introspection API |
| `ident = IdentityTransformation()` | [symmetry.jl#L2](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl#L2) | Never referenced |
| `CircleCap{T}` | [circular.jl#L2-L6](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/circular.jl#L2-L6) | No constructor, not exported, zero usages — **confirmed still dead in bg-05** |
| `CircleWedge` | [circular.jl#L9-L23](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/circular.jl#L9-L23) | **UPDATED STATUS: no longer dead** — `StadiumBilliard` now constructs it (see §5, the `bc=bc=bcs[3]` bug is now live code) |
| `reset_ids!` | [compositedomains.jl#L13-L17](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/compositedomains.jl#L13-L17) | Dead **and non-functional** (see §5) |
| `random_interior_points` | [samplers.jl#L77](/home/clozej/.julia/dev/BilliardGeometry.jl/src/quadrature/samplers.jl#L77) | Fully shadowed by a better-typed `QuantumBilliards.jl` reimplementation |
| Two commented-out blocks (`chebyshev_nodes`, old `fourier_nodes`) | [samplers.jl#L21-L28,96-120](/home/clozej/.julia/dev/BilliardGeometry.jl/src/quadrature/samplers.jl#L21-L28) | Pure clutter |
| Commented-out `arc_length` override | [limacon.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/billiards/limacon.jl) (~L55-63) | Superseded by generic `arc_length` |
| `DiagonalReflection`/`AntiDiagonalReflection` | symmetry.jl | Exist, exported, but confirmed unused by **any** real billiard (test-only) — a "never truly exercised" signal, not technically dead code |

---

## 5. Active bugs found during dead-code investigation (non-crashing but confirmed wrong)

- **`reset_ids!` is a complete no-op**, confirmed by direct execution (see attached terminal reproduction). `@set sd.id = i` (Accessors.jl) returns a new immutable value that's discarded every loop iteration — ids are never actually changed despite the `!` naming convention implying mutation. [compositedomains.jl#L13-L17](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/compositedomains.jl#L13-L17)
- **`bc = bc = bcs[3]` copy-paste bug**, now confirmed **live** (not just theoretical) since `StadiumBilliard` constructs a `CircleWedge` on every call. [circular.jl#L19](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/domains/circular.jl#L19) — numerically harmless (Julia's chained-assignment semantics happen to still set `bc` correctly) but fragile/confusing and now on a real, exercised code path.

---

## 6. Performance (hot-path, flag-only)

| Finding | Location | Impact |
|---|---|---|
| `circshift(R0[:,j-1],1)` allocates a fresh length-`N` array on every one of `N-1` loop iterations inside `kress_R_even!`/`kress_R_odd!` | [kressgrading.jl#L27-L29,52-54](/home/clozej/.julia/dev/BilliardGeometry.jl/src/quadrature/kressgrading.jl#L27-L29) | **O(N²) avoidable memory traffic** on the hottest path in the package — called by every BIM solver's `construct_matrices` (`dlp.jl`, `cfie.jl`, `ebim.jl`, `beyn.jl`, `compositebim.jl`). Highest-priority performance finding in the whole audit. Fix: `circshift!` with a reused scratch buffer. |
| `_polar_radius_derivative`/`_polar_radius_derivative_2` allocate two slice arrays on every call | [curvederivatives.jl#L52-L72](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/curvederivatives.jl#L52-L72) | Called from `tangent`/`tangent_2`, feeding `arc_length`/`area` quadrature and per-point boundary evaluation. |
| `_arc_length_integrand` re-differentiates via `ForwardDiff` instead of reusing the already-analytic `tangent(crv,t)` | [arclength.jl#L1-L4](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/arclength.jl#L1-L4) | Redundant nested-AD cost inside every `arc_length`/`area` quadrature for analytic curve types. |

---

## 7. Consolidation opportunities

- `apply_symmetry` for 5 reflection types (10 near-identical methods) could become one generic method using the existing `_sym_matrix` trait table from `fullboundary.jl` — would also fix the Float64-hardcoding in §2. ([symmetry.jl#L108-L131](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl#L108-L131))
- `symmetry_index_orbits` for 4 reflection types (index-permutation duplication) — same trait-table pattern applicable. ([symmetryorbits.jl#L181-L247](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl#L181-L247))
- The q-retry-with-warning control-flow block is duplicated near-verbatim between `kress_graded_nodes_data` and `multi_kress_graded_nodes_data` — extract a shared helper (the grading math itself should stay separate, it's genuinely different). ([kressgrading.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/quadrature/kressgrading.jl))
- 4 "quadrant, D2-symmetric" billiards (Circle/Ellipse/Prosen/Rectangle) repeat near-identical `center==(0,0)` guard + `D2_symmetry()` + wall-construction boilerplate — a shared `_quadrant_symmetry_walls(...)` helper is now justified with 4 real instances.
- `C3Billiard`/`StarBilliard` are near-identical modulo coefficient vector and `n` — worth consolidating into one parametric constructor.

---

## 8. Export audit summary

No missing exports found across any of the 5 steps. Minor notes:
- `CircleCap` is the sole non-exported symbol among the domains/segments files — consistent with being unfinished (no constructor).
- Several dead functions remain exported despite zero call sites (`construct_arc_length_interpolation`, `reset_ids!`, `CircleWedge`, etc.) — a soft "premature/misleading export" pattern, not a hard bug.
- `triangle_corners`, `limacon_eq`, `limacon_arc_length` are non-exported file-local helpers not yet underscore-prefixed, inconsistent with the package's `_foo` convention elsewhere.

---

## 9. Docstrings & test-suite staleness (flagged, not fixed — out of scope for source-editing agents)

- Docstring backfill (tracked in `geometry-audit-10.9.md` as low-priority/open) confirmed **still incomplete**: `arclength.jl`, `poincarebirkhoff.jl`, `utils.jl` have zero docstrings; `boundarytypes.jl` is partial; abstract types in the module file are entirely undocumented.
- `test/runtests.jl`'s `symmetryorbits.jl` testset still calls `symmetry_irrep_character(Float64, sym)` — a function **confirmed removed** by the Step 15 refactor. The test suite currently cannot pass as-is. Tracked in `known-bugs.md` as "needs updating" — confirmed still open/unaddressed. This should go to the Julia Test Writer agent, not a source-editing session.

---

## 10. Status of every previously-tracked gap (before/after this audit)

| Tracked item | Source | Status after bg-01..bg-05 |
|---|---|---|
| Gap 1 (antisymmetric `symmetry_index_orbits` never wiring) | geometry-audit-10.9.md | **Confirmed still fixed**, mechanism changed (struct field → trailing arg) but intact |
| Gap 2 (no multicomponent/multi-ring `symmetry_index_orbits` overload) | geometry-audit-10.9.md | **Confirmed still open, still deferred** — `AnnularBilliard` doesn't yet provide the fixture (registers trivial symmetry, C7 above) |
| Gap 3 (`full_boundary` missing polar-curve symmetry methods) | geometry-audit-10.9.md | **Confirmed still fixed** — but a **new, distinct** gap found for `LimaconSegment` (C1) and a **new** algorithmic gap for disjoint-chain fundamental domains (C2) |
| Gap 4 (`tangent_vec`/`normal_vec`/`curvature` added) | geometry-audit-10.9.md | **Confirmed still fixed**, no regression |
| Gap 5 (`FourierCoeffPolarSegment`/`PolarSegment` `promote_type` fix) | geometry-audit-10.9.md | **Confirmed still fixed** for those two types, but **never generalized** to `LineSegment`/`CircleSegment`/`Polygon` (§2) |
| Gap 6 (no `SymmetryWall`/`billiard.symmetries`/solver-sector consistency check) | geometry-audit-10.9.md | **Confirmed still open, still deferred** — concrete new evidence found (`SymmetryWall.sector_id` never read anywhere; inconsistent tagging across billiards) |
| `connect_curves` multi-component fix (Step 11) | known-bugs.md | **Confirmed still present, no regression** |
| Symmetry framework refactor (Step 15: `symmetry_irrep_character` removed, `sym_id`-only + `SymmetrySector`) | known-bugs.md | **Confirmed matches description exactly**, no leftover `irrep_character` code path in source (only in the stale test file, §9) |
| Step 12 `AbsMultiplyConnectedDomain`/`genus`/`AnnularBilliard` | migration-plan/12 | **Confirmed complete, no regression**; `compositedomains.jl` vs. `multiplyconnecteddomains.jl` confirmed genuinely non-redundant |
| Step 12.5 `area`/`fundamental_area`/`corner_angles` migration | migration-plan/12.5 | **Confirmed complete**, nothing left half-migrated in `QuantumBilliards.jl`; `corner_angles` confirmed load-bearing for Weyl-law/unfolding |
| Step 16 BIM/SymmetrySector migration ("plan only, not started") | migration-plan/16 | **Status label is stale** — `QuantumBilliards.jl`'s `symmetrysector.jl` already implements the plan's design almost verbatim; plan itself says no `BilliardGeometry.jl`-side prerequisites exist, so nothing was missed here (a `QuantumBilliards.jl`-side finding for a future `qb-*` step) |
| 11-billiards-and-domains.md inventory | migration-plan/11 | **Confirmed still accurate**, nothing called for was missed |

---

## Suggested next actions (not performed — triage/planning only)

1. **Highest priority fix session**: C1 (Limaçon `_apply_symmetry_to_curve`) + C3/C4 (`apply_symmetry_pb` rotation support) — both are missing-method gaps with clear signatures, good candidates for a single `julia-refactor-scoped` or `julia-add-solver`-adjacent fix session.
2. **Second priority**: C2 (Mushroom/`full_boundary` disjoint-chain algorithm) — needs careful design work, its own scoped session.
3. **Quick wins**: C5/C6 constructor guards (mirror existing D2 billiards' `center==(0,0)` check pattern); the `circshift!` performance fix (§6); the `reset_ids!` no-op fix or removal; the `bc=bc=bcs[3]` cleanup now that it's live.
4. **Defer**: Gap 2/Gap 6/C7/C8 — correctly deferred pending a real symmetric multiply-connected billiard fixture; no action needed until one is added.
5. **Hand off to Julia Test Writer**: stale `symmetry_irrep_character` test call (§9); potentially new regression tests for C1–C6 using the smoke-check scripts already saved under [smoke-checks/](/home/clozej/Programs/QuantumBilliardsTests/scratchpad/audit-plan/smoke-checks/) as a starting point.
6. Continue the plan with the `qb-*` steps (`QuantumBilliards.jl`) per `00-index.md`, then do the full cross-package `qb-11` synthesis and memory promotion at the very end.
