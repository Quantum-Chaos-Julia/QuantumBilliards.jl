# Step 17: Symmetry Framework Consolidation — Report & Refactoring Plan

Date produced: 2026-09-27.
**Status: fresh analysis, produced by directly reading current source and
running the test suite — not a re-issue of the bg-02/qb-10 audit findings
below, which are dated 2026-09-14 and are partially stale (the git history
shows three intervening commits — "Symmetry rework", "Rewrote symmetry
orbits", "Refactored symmetries" — that already fixed several of their
headline findings; this report says explicitly, per item, which old findings
are now fixed and which are still open.**

Builds on, and supersedes where noted:
- [bg-02-symmetry-framework-findings.md](/home/clozej/.julia/dev/QuantumBilliards.jl/scratchpad/audit-plan/findings/bg-02-symmetry-framework-findings.md)
- [qb-10-symmetry-states-findings.md](/home/clozej/.julia/dev/QuantumBilliards.jl/scratchpad/audit-plan/findings/qb-10-symmetry-states-findings.md)
- [15-symmetry-representation-refactor.md](15-symmetry-representation-refactor.md)
- [16-bim-symmetrysector-migration.md](16-bim-symmetrysector-migration.md)

Companion document (BilliardGeometry.jl-scoped, requested separately):
[symmetry-test-coverage-report.md](/home/clozej/.julia/dev/BilliardGeometry.jl/scratchpad/symmetry-test-coverage-report.md).

---

## 1. Architecture map (current, verified state)

**Layer 1 — geometric symmetry group (`BilliardGeometry.jl`)**

| File | Responsibility |
|---|---|
| [symmetry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl) | Concrete generator types (`XAxisReflection`, `YAxisReflection`, `XYAxisReflection`, `DiagonalReflection`, `AntiDiagonalReflection`, `CompositeReflection`, `NFoldRotation`), `apply_symmetry`, `apply_symmetry_pb`, `D2_symmetry`/`Cn_symmetry` convenience registrars. |
| [symmetryregistry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryregistry.jl) | `SymmetryRegistry` (stable `sym_id`-tagged `AbstractVector` wrapper), `register_symmetries`, `symmetry_of`. |
| [symmetryorbits.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl) | Exact integer-algebra permutation/orbit machinery: `symmetry_index_orbits`, `SymmetryOrbitMap`, `get_symmetries` (minimal generator reduction), `symmetry_node_multiple`. |
| [fullboundary.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/fullboundary.jl) | `full_boundary`, `_apply_symmetry_to_curve`, `_orientation_reversing`, `_sym_matrix` (private per-type trait table). |
| [poincarebirkhoff.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/poincarebirkhoff.jl) | `pb_coords`/`pb_sectors`, consuming `apply_symmetry_pb`. |
| [boundarytypes.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/boundarytypes.jl) | `SymmetryWall` boundary-condition marker (`sym_id`+`sector_id`). |

**Layer 2 — representation/solve-time choice (`QuantumBilliards.jl`)**

| File | Responsibility |
|---|---|
| [states/symmetry/symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl) | `SymmetrySector`, `symmetry_sector`, `_resolve_bim_symmetry` (→ `(AbsSymmetry, character-tuple)` for BIM solvers). |
| [states/symmetry/reflections.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/reflections.jl) | `apply_symmetries_to_wavefunction`/`_boundary_function`/`_boundary_points` — unfold a fundamental-domain solution back onto the full billiard. |
| [basis/planewaves/realplanewaves.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl) | Own `symmetries::Vector{AbsReflection}`/`sym_qnumbers` fields + `sym_x`/`sym_y` quantum-number convention; a `RealPlaneWaves(dim, billiard, sector::SymmetrySector)` adapter constructor. |
| [basis/fourierbessel/corneradapted.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/fourierbessel/corneradapted.jl) | `symmetries::Union{Vector{Any},Nothing}` field — see §3.7. |
| [solvers/sweepmethods/{dlp,cfie,compositebim,sweepmethods}.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/) | `symmetry::Union{Nothing,AbsSymmetry}` + `sym_characters::Tuple` fields; `_fold_boundary`/`_infer_bim_symmetry`/`_composite_symmetry_orbits`. |
| [solvers/taylor/{recurrences,beyn_recurrence}.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/taylor/) | Reuse the same `solver.symmetry`/`solver.sym_characters` fields via `_fold_boundary`. |

## 2. What has already improved since the last audit (2026-09-14 → now)

Verified directly against current source, not assumed from the old findings:

- **Fixed**: `apply_symmetry_pb` is now `apply_symmetry_pb(sym::AbsSymmetry, sym_sector, s, p, L)` and dispatches purely on `_orientation_reversing(sym)` (`(k+1)*L-s,-p` vs. `k*L+s,p`) — genuinely generic over any registered generator, including `NFoldRotation`. The bg-02 finding ("`apply_symmetry_pb` has no `AbsRotation` method, guaranteed `MethodError` for `C3Billiard`/`StarBilliard`") is **resolved**.
- **Fixed**: `SymmetrySector`'s cross-billiard identity check now exists and is wired in. `_check_sector_billiard(billiard, sector)` ([symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl)) is called from both `_resolve_bim_symmetry(billiard, sector)` and `RealPlaneWaves(dim, billiard, sector)`. The qb-10 headline vulnerability ("a `SymmetrySector` built against one billiard can be silently consumed against a different billiard with coincidentally-matching `sym_id`s") is **resolved** — `sector.billiard` is no longer a dead field.
- **Improved**: `CompositeBIMSolver`'s `_composite_symmetry_orbits` now folds **each connected boundary component independently** using that component's own sub-billiard (`solver.component_solvers[a].billiard`), rather than treating the whole concatenated `pts.xy` as one ring. This is materially better than the bg-02 finding described (a single-ring fold over the flattened array) — the "silently wrong for multi-ring symmetric billiards" risk from Gap 2 is architecturally addressed. **However** (see §4, item L) there is still no billiard in the catalogue that registers a nontrivial symmetry on `AnnularBilliard`/any genus>0 billiard, so this path remains completely unexercised — a test-fixture gap, not an implementation gap.
- **Confirmed still open** (re-verified independently, not just trusted): the duplicate `Pair` key silent-overwrite in `symmetry_sector(billiard, choices::Pair...)`; the `apply_symmetries_to_*` export/documentation inconsistency; the `_orientation_reversing` boolean-literal duplication in `reflections.jl`; `DiagonalReflection`/`AntiDiagonalReflection` still unused by any real billiard; the dead `ident = IdentityTransformation()` binding.

## 3. Consolidation opportunities (ranked by impact)

### 3.1 Unify the four parallel "generator ↔ character" representations (highest impact)
Today the same fact — "which registered generator(s) get which irrep
character" — is expressed in **four independent shapes** across the two
packages:

1. `SymmetrySector.characters::Dict{Int,ComplexF64}` (billiard-level, canonical, validated against `sym_id`).
2. `sym_characters::Tuple` on every BIM solver, paired **positionally** with `BilliardGeometry.get_symmetries(billiard)`'s generator order (`dlp.jl`/`cfie.jl`/`sweepmethods.jl`).
3. `symmetries::Vector{AbsReflection}` + parallel `sym_qnumbers::Vector{T}` arrays, independently on `RealPlaneWaves` *and* on every `apply_symmetries_to_*` function in `reflections.jl`.
4. `sym_x::Union{Int,Nothing}`, `sym_y::Union{Int,Nothing}` — `RealPlaneWaves`'s own native quantum-number convention.

The clearest concrete symptom: [`_infer_bim_symmetry`](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl#L174) (representation 2) and [`symmetry_sector(billiard, characters::Integer...)`](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl#L138) (representation 1) implement **near-identical logic** — both pair a flat tuple/vararg of characters positionally with `get_symmetries(billiard)`, both special-case a lone `NFoldRotation` vs. wrapping reflections in `CompositeReflection` — as two independent, hand-written code paths that must be kept in sync by hand.

**Recommendation:**
- Make `SymmetrySector` (representation 1) the single canonical in-package
  representation. Rewrite `_infer_bim_symmetry(billiard, sym_characters::Tuple)`
  to call `symmetry_sector(billiard, sym_characters...)` then
  `_resolve_bim_symmetry(billiard, sector)`, deleting its hand-rolled
  duplicate of the generator-pairing/`CompositeReflection`-wrapping logic.
  This is a same-file, low-risk change (`sweepmethods.jl` already imports
  `symmetrysector.jl`'s exported API).
- Give `reflections.jl`'s three `apply_symmetries_to_*` functions an
  additional `SymmetrySector`-based overload (thin adapter: resolve
  `symmetries`/`sym_qnumbers` parallel arrays from `sector.characters` +
  `billiard.symmetries`, then delegate to the existing implementation) so
  callers holding a `SymmetrySector` don't have to manually rebuild parallel
  arrays. Keep the low-level parallel-array methods too (`RealPlaneWaves`
  genuinely needs its own `sym_x`/`sym_y` convention internally for basis
  construction — not worth forcing through `SymmetrySector` there).
- Do **not** attempt to collapse representation 4 (`sym_x`/`sym_y`) into
  `SymmetrySector` at the `RealPlaneWaves` struct-field level — the adapter
  constructor `RealPlaneWaves(dim, billiard, sector)` already does this
  translation at construction time, which is the right boundary; forcing the
  struct itself to store a `SymmetrySector` would leak a `BilliardGeometry`
  billiard-specific type into a basis object that today has no billiard
  dependency at all (a real regression in decoupling).

### 3.2 Collapse `apply_symmetry`'s 10 near-identical reflection methods
[symmetry.jl:108-131](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl) has one point-method and one
vector-method per reflection type (10 total), differing only in which
module-level `LinearMap` constant (`reflect_x`/`reflect_y`/`reflect_xy`/
`reflect_diag`/`reflect_antidiag`) is applied. `fullboundary.jl` already has
exactly the right per-type trait table for this (`_sym_matrix(sym)`,
[fullboundary.jl:36-41](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/fullboundary.jl#L36-L41)), just defined in the wrong file to be reused
from `symmetry.jl` (`fullboundary.jl` is `include`d after `symmetry.jl`).

**Recommendation:** move the `_sym_matrix` trait table (and the module-level
`reflect_*` constants) into `symmetry.jl` itself, ahead of `apply_symmetry`,
and replace all 10 methods with:
```julia
apply_symmetry(sym::AbsReflection, pt::SVector{2,T}) where T<:Real = SVector{2,T}(_sym_matrix(sym) * pt)
apply_symmetry(sym::AbsReflection, pts) = [apply_symmetry(sym, pt) for pt in pts]
```
This simultaneously fixes the `Float64`-hardcoding precision issue below
(§3.3) if `_sym_matrix` is made to return a `T`-typed matrix rather than a
`Float64`-typed `LinearMap`'s `.linear` field.

### 3.3 Make reflection/rotation transforms generic over `T`, not hardcoded `Float64`
`reflect_x`/`reflect_y`/`reflect_diag`/`reflect_antidiag`/`reflect_xy`
([symmetry.jl:3-7](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl#L3-L7)) are module-level
`LinearMap{SMatrix{2,2,Float64,4}}` constants, and `NFoldRotation`
([symmetry.jl:179-186](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl)) hardcodes
`angle::Float64`/`sym_map::LinearMap{SMatrix{2,2,Float64,4}}` even though the
struct is nominally parametric (`NFoldRotation{T<:Real}`, with `T` correctly
threaded through its *own* constructor) — wait, `NFoldRotation` **is** already
parametric (confirmed from the current `symmetry.jl` read directly), so this
half of the old bg-02 finding is **already fixed**. What remains hardcoded to
`Float64` is only the plain reflections' `reflect_*` constants (`XAxisReflection`
etc. have no type parameter at all — they are singleton-like `sym_id`-only
structs, by design, since a mirror-line reflection matrix has no continuous
parameter to store). `_apply_symmetry_to_curve` for `LineSegment`
([fullboundary.jl:52-56](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/fullboundary.jl#L52-L56)) re-wraps the
`Float64`-computed product back into `SVector{2,T}(...)`, silently narrowing
precision for a `BigFloat`-typed billiard.

**Recommendation:** this is genuinely low priority in practice (no billiard
in the catalogue is instantiated at `T != Float64` today per `bg-05`'s
catalogue audit), but since §3.2's refactor touches this exact code path
anyway, make `_sym_matrix(sym)` return `SMatrix{2,2,T}` for the caller's `T`
(a `±1`/`0` integer-literal matrix promotes losslessly to any `T<:Real`) —
free correctness improvement bundled into an already-planned change, not a
standalone project.

### 3.4 Collapse `symmetry_index_orbits`'s 4 near-identical two-element-orbit methods
`XAxisReflection`/`YAxisReflection`/`DiagonalReflection`/`AntiDiagonalReflection`'s
`symmetry_index_orbits` methods ([symmetryorbits.jl:460-487](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl))
are structurally identical calls into `_boundary_symmetry_permutation`/
`_build_symmetry_orbit_map`, differing only in the concrete reflection type
threaded through (which already fully determines behavior via
`_reflection_sector_permutation`/`_orientation_reversing`/`_symmetry_key`).

**Recommendation:** replace the 4 near-duplicate methods with one
`symmetry_index_orbits(::Type{T}, billiard, N, symmetry::AbsReflection, character::Complex{T}=one(Complex{T})) where T` method (excluding `XYAxisReflection`/`CompositeReflection`, which genuinely need a different, multi-character signature and should keep their own methods). This is a pure deduplication with no behavior change — the per-type dispatch was never doing type-specific work beyond routing.

### 3.5 Share `_orientation_reversing` across the package boundary instead of re-deriving it
`reflections.jl`'s `apply_symmetries_to_boundary_points`
([reflections.jl:224-226](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/reflections.jl)) hardcodes the
same orientation-reversal fact `BilliardGeometry._orientation_reversing`
already encodes, as literal `true`/`false` arguments at each call site
(`push_reflection!(get_sym(YAxisReflection), true)`, `..XYAxisReflection.., false)`,
`..XAxisReflection.., true)`), and `apply_symmetries_to_wavefunction`/
`_boundary_function` re-encode the identical fact a third and fourth time via
which branches call `reverse(...)`. **Confirmed still present** (re-read the
current file directly).

**Recommendation:** export `_orientation_reversing` from `BilliardGeometry.jl`
(rename to a public `orientation_reversing(sym)` — dropping the leading
underscore is the only breaking part of this change, and it is already used
outside `fullboundary.jl` in spirit) and call it from `reflections.jl` instead
of the 3 independent hardcoded encodings. Low risk, removes a real
correctness-drift hazard (a future `DiagonalReflection`-based billiard would
need this fact updated in 4 places today, 1 of them in a different package).

### 3.6 Deduplicate the has_x/has_y/get_qnum/get_sym boilerplate inside `reflections.jl`
All three `apply_symmetries_to_*` functions independently reimplement
`has_x = any(s -> s isa XAxisReflection, symmetries)` / `has_y = any(...)` /
the same 4-way branch skeleton (`has_x&&has_y` / `has_y` / `has_x` / else).
**Recommendation:** factor a small private helper
`_reflection_case(symmetries) → (has_x, has_y)` plus a shared
`get_qnum(::Type{S})`/`get_sym(::Type{S})` closure-builder, reused by all
three. Diminishing-returns beyond that point since the three functions
operate on different data shapes (2D grid, 1D boundary vector, structured
`BoundaryPoints`) — do not force a single generic driver.

### 3.7 `CornerAdaptedFourierBessel`'s `symmetries::Union{Vector{Any},Nothing}` field is untyped and appears unused
The struct's docstring says `symmetries: Optional vector of symmetries
applied to the basis, or nothing`, but the field is typed
`Union{Vector{Any},Nothing}` (an `Any`-vector — a direct violation of this
project's own type-stability convention) rather than
`Union{Vector{<:AbsSymmetry},Nothing}`. **Action for a follow-up step (not
this report):** grep confirms no constructor in `corneradapted.jl` actually
populates this field with anything but `nothing`, and no consumer reads it
for anything beyond existence. Recommend either (a) tightening the type to
`Union{Vector{AbsSymmetry},Nothing}` if the field has a real future use, or
(b) removing it entirely if it is vestigial — needs a short, targeted read of
`corneradapted.jl` in full (out of scope for this pass, which is
symmetry-focused, not basis-focused) before deciding; flagged here since it
was surfaced while tracing `AbsSymmetry` consumers.

### 3.8 Minor cleanups (batch together, low risk, low effort)
- Remove the dead `ident = IdentityTransformation()` binding in [symmetry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl) (confirmed zero references anywhere in the workspace).
- Remove the stray `println(sym_sector)` debug leftover in [poincarebirkhoff.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/poincarebirkhoff.jl)'s `pb_sectors` — pollutes stdout on every call, clearly a forgotten debug statement rather than intentional logging (should be `@debug` at most, per this package's own conventions).
- Add a duplicate-`GenType` guard to `symmetry_sector(billiard, choices::Pair...)` ([symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl)): currently `symmetry_sector(b, XAxisReflection=>1, XAxisReflection=>-1)` silently keeps only the last pair. A one-line `allunique(first.(choices)) || throw(ArgumentError(...))` closes this.
- Export `apply_symmetries_to_boundary_function`/`apply_symmetries_to_boundary_points` (currently neither exported nor in `docs/src/API.md`, despite equal documentation standard to `apply_symmetries_to_wavefunction`, which is documented but *also* not exported) — resolve by exporting all three and adding the missing two `@docs` entries, for consistency (they are genuinely public-facing per their use from `wavefunction`/`boundary_function`).

## 4. Refactoring plan (phased, actionable, ordered by risk/dependency)

| Phase | Item(s) | Files touched | Risk | Depends on |
|---|---|---|---|---|
| 1 | §3.8 minor cleanups (dead code, debug print, export/doc fixes, duplicate-pair guard) | `symmetry.jl`, `poincarebirkhoff.jl`, `symmetrysector.jl`, `QuantumBilliards.jl` exports, `docs/src/API.md` | Very low — no behavior change for any passing call | none |
| 2 | §3.1 unify `_infer_bim_symmetry` onto `symmetry_sector`/`_resolve_bim_symmetry` | `sweepmethods.jl` | Low — same external `(symmetry, sym_characters)` contract, internal implementation swap only | none (independent of Phase 3) |
| 3 | §3.2 + §3.3 collapse `apply_symmetry`, hoist `_sym_matrix`, make it `T`-generic | `symmetry.jl`, `fullboundary.jl` (move trait table up) | Medium — touches every reflection call site indirectly via dispatch; must re-verify `fullboundary.jl`'s own use of `_sym_matrix` still works after the move | none |
| 4 | §3.4 collapse `symmetry_index_orbits`'s 4 two-element-orbit methods | `symmetryorbits.jl` | Medium — this is the most heavily exact-integer-algebra code in the framework; **must** be done only after the test-coverage gaps in the companion report are closed (see below), since there is currently no passing regression test that would catch a mistake here | companion test report's Phase 1 |
| 5 | §3.5 export `orientation_reversing`, wire `reflections.jl` to it | `fullboundary.jl`/module exports, `reflections.jl` | Low-medium — small breaking rename (`_orientation_reversing`→public name) is internal-only (no external caller depends on the underscored name, confirmed by grep) | none |
| 6 | §3.6 dedupe `reflections.jl` has_x/has_y boilerplate | `reflections.jl` | Low | ideally after Phase 5 (touches the same file) |
| 7 | §3.7 `CornerAdaptedFourierBessel.symmetries` field decision | `corneradapted.jl` | Needs its own short investigation first (not this report's scope) | none |

**Explicit recommendation on sequencing:** do **not** start Phase 4 (the
exact-permutation-algebra consolidation, the single highest-value-but-highest-
blast-radius item) until the [companion test coverage report](/home/clozej/.julia/dev/BilliardGeometry.jl/scratchpad/symmetry-test-coverage-report.md)'s Phase 1 tests exist and pass — right now, as documented there, the
existing `symmetryorbits.jl` test file in `BilliardGeometry.jl` is
**completely non-functional against the current API** (fails on the very
first orbit-construction call), so there is zero regression coverage today
for exactly the code Phase 4 would rewrite.

## 5. Explicit non-goals

- No dihedral (2-D irrep, mixed reflection+rotation) representation support
  — confirmed intentional in `_resolve_bim_symmetry`'s own docstring, not a
  gap to close in this pass.
- No attempt to give `AnnularBilliard`/multiply-connected billiards a real
  symmetric fixture as part of this consolidation — that is new-feature work
  (a new billiard constructor), not a refactor, and belongs in its own step.
- No attempt to unify `RealPlaneWaves`'s `sym_x`/`sym_y` struct fields with
  `SymmetrySector` at the storage level (see §3.1's explicit reasoning for
  keeping the adapter boundary where it is).
