# QB-11: Final Cross-Cutting Synthesis — Whole-Ecosystem Code-Quality Audit (Steps bg-01 → qb-10)

**Use case: package-wide audit branch of `julia-refactor-scoped`.** Read-only. All 15 prior findings files were read in full; this step adds direct verification of module wiring (`QuantumBilliards.jl`, `BilliardGeometry.jl` module files + `Project.toml`s) and grep-level call-site checks against `QBPlotting.jl`, `BilliardGeometryPlotting.jl`, `QCPlotting.jl`, `SpectralStatistics.jl`. No source files were edited.

---

## TOP-LEVEL PRIORITY LIST (most severe/highest-value across all 16 steps)

1. **`merge_spectra` is an exported, public function that unconditionally throws `UndefVarError`** — calls three undefined names (`interval`, `intersect_interval`, `in_interval`); `IntervalArithmetic` is declared in [Project.toml](/home/clozej/.julia/dev/QuantumBilliards.jl/Project.toml) but never `using`'d. (qb-07, confirmed here as also a Missing-Wiring/unused-dependency issue.)
2. **`LimaconBilliard.full_boundary` and `MushroomBilliard.full_boundary` are both broken** (crash / silently non-closed boundary) — the two clearest "symmetry doesn't work" findings from the original audit motivation. (bg-05)
3. **`apply_symmetry_pb` has no method for `NFoldRotation`**, so `pb_sectors`/`pb_coords` crash for `C3Billiard`/`StarBilliard` — **confirmed in this step to already be a live, broken downstream call site**: `QBPlotting.jl`'s `plot_husimi_function!` (both methods) and both boundary-function-plotting functions call `pb_sectors(billiard)` unconditionally. Any user plotting a Husimi function or boundary function for a rotation-symmetric billiard hits this today. (bg-02/bg-05, sharpened here)
4. **`SymmetrySector` carries no billiard-identity check** — a sector built for one D2 billiard silently resolves against a different D2 billiard with coincidentally-matching `sym_id`s, producing a wrong-but-non-trivial (not erroring, not obviously trivial) representation. The fix is to start consuming the already-stored (currently dead) `sector.billiard` field. (qb-10)
5. **`CornerAdaptedFourierBessel`'s type parameter is hardcoded `{Float64,Nothing}`** regardless of input types — breaks its own `toFloat32` conversion function outright, and silently drops all symmetry information. (qb-02)
6. **`RealPlaneWaves` silently ignores an `NFoldRotation` `SymmetrySector`** — returns the full unrestricted basis with no error/warning when a rotation sector is requested. (qb-02)
7. **4 confirmed unused dependencies in `QuantumBilliards.jl`** (`ForwardDiff`, `FastGaussQuadrature`, `StatsBase`, `IntervalArithmetic`) and **1 in `BilliardGeometry.jl`** (`StatsBase`) — verified in this step by direct grep (zero references anywhere in either package's `src/`, beyond the `using`/`[deps]` declarations). This is new, first-confirmed-here evidence for the Missing-Wiring checklist.
8. **`StarBilliard`/`MushroomBilliard` off-center construction silently produces broken geometry** — no guard, unlike every D2 billiard. (bg-05)
9. **`circshift` allocation in `kress_R_even!`/`kress_R_odd!`** — O(N²) avoidable memory traffic on the hottest path in the whole geometry package, hit by every BIM solver. (bg-04)
10. **Non-exception-safe, racy global `BLAS.set_num_threads` macros** (`@blas_1`/`@blas_multi_then_1`/`@blas_multi`) — confirmed hit by every sweep/accelerated BIM and basis solver's `solve`/`construct_matrices`, and confirmed to run under `BeynSolver`'s threaded per-window loop, i.e. a real (not hypothetical) data race. (qb-01, reconfirmed qb-03/qb-04/qb-05)

---

## 1. Dead code — cross-package (strongest "safe to remove" signal)

Symbols confirmed to have **zero usages anywhere in the whole 8-folder workspace** (not just their own package):

| Symbol | Package | Notes |
|---|---|---|
| `construct_arc_length_interpolation` | BilliardGeometry.jl | Exported, zero call sites (bg-01) |
| `point_curve_parameter` | BilliardGeometry.jl | Exported, zero call sites, also buggy (bg-01) |
| `invert_curve_nelder_mead` | BilliardGeometry.jl | Unreachable via `invert_curve` dispatch (bg-01) |
| `pb_coords`, `fundamental_pb_coords`, `get_pb_curve` | BilliardGeometry.jl | Only `pb_sectors` (same file) has any consumer (bg-01) |
| `_boundary_components` | BilliardGeometry.jl | Only used in its own test (bg-01) |
| `CircleCap` | BilliardGeometry.jl | No constructor, not exported, unconstructible as shipped (bg-03/bg-05) |
| `reset_ids!` | BilliardGeometry.jl | Exported, zero call sites, **also non-functional** (§5 below) (bg-03) |
| `random_interior_points` (in `samplers.jl`) | BilliardGeometry.jl | Fully shadowed by a better `QuantumBilliards.jl` reimplementation (bg-04) |
| `ident = IdentityTransformation()` | BilliardGeometry.jl | Never referenced (bg-02) |
| `_dlp_fredholm_full_cheb!`, `_dlp_fredholm_reduced_cheb!`, `_cfie_fredholm_full_cheb!`, `_cfie_fredholm_reduced_cheb!` | QuantumBilliards.jl | Orphaned by the multi-k batch rewrite (qb-06) |
| `_cheb_clenshaw_col` + 10 matrix-arg `_cheb_clenshaw_*` overloads | QuantumBilliards.jl | Never match main's per-panel-vector data layout (qb-06) |
| `panel_and_geom` | QuantumBilliards.jl | The exact primitive the still-open Gap 3 panel-index cache needs (qb-06) |
| `directsum` | QuantumBilliards.jl | Documented, zero call sites (qb-03) |
| `_scale_rows!` | QuantumBilliards.jl | Superseded by `_scale_rows_sqrtw!` (qb-03) |
| `polar_to_cartesian`, `cartesian_to_polar` | QuantumBilliards.jl | Exported in `-develop`, dropped here, unused (qb-01) |
| `merge_spectra` | QuantumBilliards.jl | Dead **and** broken — see Priority #1 (qb-07) |
| `dim`, `k_basis`, `eps` fields on `BasisState`/`BIMEigenstate` | QuantumBilliards.jl | Write-only struct fields, never read (qb-08) |

**Exported-but-never-defined** (confirmed by qb-01, re-verified against [QuantumBilliards.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/QuantumBilliards.jl) directly in this step — lines 60–62): `print_benchmark_info`, `BoundaryPointsSM`, `BoundaryPointsDM`, `construct_matrices_benchmark`. `QuantumBilliards.print_benchmark_info` etc. throw `UndefVarError` if ever accessed.

## 2. Missing wiring (consolidated across all 15 steps)

**Missing exports — merged, deduplicated master list:**

*QuantumBilliards.jl:*
- `AbsState` (qb-01, re-affirmed qb-08) — most significant single gap; every other abstract-type branch is exported.
- `k_at_state` (qb-07) — direct inverse of exported `state_at_k`, `@ref`-cross-referenced by 3 other exported docstrings.
- `adjust_scaling_and_samplers` (qb-01, qb-03) — documented `AbsSolver`/`AbsBasisSolver` API member, never exported.
- `CoordinateSystem`, `CartesianCS`, `PolarCS` (qb-01).
- `make_veech_right_triangle`, `make_veech_right_triangle_and_basis` (qb-01) — accessed only via fully-qualified calls in sibling test scripts, a strong signal of intended-public status.
- `generalized_eigen`, `generalized_eigvals`, `generalized_eigen_all` (qb-03) — actively used by every basis solver, referenced in the pre-built docs HTML.
- `random_interior_points`, `solve_full`, `solve_with_rank_reduction`, `symmetrize_layer_density` (qb-04).
- `apply_symmetries_to_wavefunction` (doc'd but not exported), `apply_symmetries_to_boundary_function`, `apply_symmetries_to_boundary_points` (qb-10) — inconsistent doc/export tiers among 3 functions of identical standing.
- `sm_results` (qb-05) — better as internal-only (underscore-prefix), given zero cross-file usage; flagged here for completeness of the master list.

*BilliardGeometry.jl:* No missing exports found across bg-01–bg-05 (only soft "premature export" issues on dead code, §1).

**Reconciliation note:** none of these missing-export findings conflict across steps; they compose cleanly into the two lists above.

**Unused dependencies (Project.toml `[deps]` declared but never `using`'d/referenced), confirmed by direct grep in this step:**
- **QuantumBilliards.jl**: `ForwardDiff`, `FastGaussQuadrature`, `StatsBase`, `IntervalArithmetic` — none appear in the module's `using` list ([QuantumBilliards.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/QuantumBilliards.jl#L2-L12)) and zero calls to any of their functions exist anywhere in `src/`. (`IntervalArithmetic`'s absence is the direct root cause of Priority #1 — `merge_spectra` looks like an abandoned attempt to use it.)
- **BilliardGeometry.jl**: `StatsBase` — `using StatsBase` is present ([BilliardGeometry.jl#L9](/home/clozej/.julia/dev/BilliardGeometry.jl/src/BilliardGeometry.jl#L9)), but no `mean`/`std`/`var`/`median`/`Weights`/etc. call exists anywhere in `src/`. (`Accessors`, `FastGaussQuadrature`, `Elliptic`, `DataInterpolations`, `Roots`, `Optim` were all cross-checked and confirmed genuinely used — not flagged.)

## 3. Missing implementations — prioritized master list (main deliverable value)

**Genuinely still open (not fixed by any later step):**
1. `LimaconSegment` has no `_apply_symmetry_to_curve` method → `LimaconBilliard.full_boundary` crashes (bg-05).
2. `full_boundary`'s disjoint-chain algorithm gap → `MushroomBilliard.full_boundary` silently non-closed (bg-05) — a generic algorithmic limitation, not billiard-specific.
3. `apply_symmetry_pb` has no `AbsRotation`/`NFoldRotation` method, and never reads its own `sym` argument (only works by coincidence of `D2_symmetry()`'s fixed registration order) → crashes `C3Billiard`/`StarBilliard`'s Husimi/PB-coordinate path, **now confirmed to break `QBPlotting.jl`'s `plot_husimi_function!`/boundary-function plotting today** (bg-02, bg-05, this step).
4. `symmetry_index_orbits` has no multicomponent/multi-ring overload (Gap 2) → `AnnularBilliard` under-registers its symmetry, and `CompositeBIMSolver` would silently misfold a symmetric multiply-connected billiard's boundary the first time one is constructed (bg-02, bg-05, qb-04 — three independent confirmations of the same latent gap).
5. `SymmetrySector` has no billiard-identity check despite storing `sector.billiard` — silently resolves against the wrong same-shape-family billiard (qb-10) — this is the concrete, previously-abstract "Gap 6" (no `SymmetryWall`/`billiard.symmetries`/solver-sector consistency check) finally pinned to a specific missing `===` check.
6. `CornerAdaptedFourierBessel` has zero `(dim,billiard,sector::SymmetrySector)` constructor — total absence of symmetry-sector support (worse than `RealPlaneWaves`'s partial reflection-only support) (qb-02).
7. `RealPlaneWaves` silently ignores rotation sectors (qb-02).
8. `wavefunction(state::BasisEigenstate)`'s planned `method=:green` keyword (migration-plan 05.6 "definition of done") was never implemented — `compute_psi`'s dense basis path is still the only option (qb-09).
9. `compute_psi` has no `BIMEigenstate` method despite being documented as universal over `S<:AbsState` — throws a raw field-access error if ever called directly on a BIM state (qb-08).
10. `StarBilliard`/`MushroomBilliard`/`LimaconBilliard`/`C3Billiard`/`TriangleBilliard` lack degenerate/off-center-parameter guards present on every other billiard (bg-05).
11. Beyn/EBIM's `use_chebyshev=true` default has no fail-fast `T===Float64` check at construction — only errors on first solve (qb-05).
12. Chebyshev panel-index cache (Gap 3, still open) and combined H0/H1 scalar evaluator (Gap 5, still open) — both explicitly deferred/profiling-gated by the 10.5 plan itself, not regressions (qb-06).

**Confirmed CLOSED (contrary to stale status labels — important for not re-litigating):**
- Step 16 (BIM/`SymmetrySector` migration) — fully implemented in `symmetrysector.jl`/`dlp.jl`/`cfie.jl`/`compositebim.jl`, despite the migration-plan doc still saying "Plan only, not started" (bg-02, qb-04, qb-10 — three independent confirmations).
- `CompositeBIMSolver`'s docstring claiming "API scaffold only, every method raises an error" is stale/false — fully implemented (qb-04).
- 10.5's Beyn multi-pass gap and SLP/CFIE Chebyshev wavefunction reconstruction gap — both implemented and fully wired (qb-06).
- `area`/`fundamental_area`/`corner_angles` 12.5 migration — clean, complete, no half-migrated remnants (bg-01, qb-07 — both sides confirmed).

## 4. Unstable APIs — cross-package mismatches

- **`CornerAdaptedFourierBessel{Float64,Nothing}` hardcoding** breaks its own `toFloat32` function — the most severe unstable-API finding in the whole audit (qb-02).
- **Reflection transforms (`reflect_x`/etc.) and `NFoldRotation` hardcoded to `Float64`** in `BilliardGeometry.jl`'s `symmetry.jl` — silently narrows/widens precision for `BigFloat`/`Float32` billiards; **directly consumed by `QuantumBilliards.jl`'s solver layer**, making this a genuine cross-package precision leak, not just an internal one (bg-02).
- **`LineSegment`/`CircleSegment`/`Polygon` constructors don't `promote_type`**, unlike the already-fixed `PolarSegment`/`FourierCoeffPolarSegment` — confirmed via direct `MethodError` reproduction (bg-03).
- **`LinearNodes`/`GaussLegendreNodes` hardcoded to `Float64`**, unlike `FourierNodes` — these are the exact sampler types `QuantumBilliards.jl`'s `DecompositionMethodSolver`/`ParticularSolutionsMethod`/`VerginiSaracenoSolver` construct by default (`sampler = [GaussLegendreNodes()]`, confirmed by grep in this step) — so any attempt to run those solvers in non-`Float64` precision would silently fall back to `Float64` sampling regardless of the solver's own `T` (bg-04, sharpened here with the confirmed cross-package call sites).
- **`generalized_eigen`/`generalized_eigvals` hardcode `1.0` instead of `one(eltype(d))`** — silently promotes a `Float32` solve to `Float64` internally (qb-03).
- **`dim`'s meaning differs between basis types** (`RealPlaneWaves`: sampled-angle count; `CornerAdaptedFourierBessel`: actual basis size) with no doc cross-reference on the shared `AbsBasis` contract (qb-02).
- **`AbsState`'s documented API oversells `BasisState`'s actual contract** — `boundary_function`/`momentum_function`/`husimi_function` would all throw a field-access error on a `BasisState`, since it lacks `billiard`/`solver`/`ten` fields (qb-08).

## 5. Vulnerabilities — consolidated silent-fallback patterns (repeated-finding signal is stronger)

The **"silent wrong result instead of a loud error"** pattern was independently flagged by essentially every step in this audit — this repetition across 8+ unrelated files/subsystems is itself the strongest single piece of evidence that the ecosystem has a systemic gap in input validation at trust boundaries, not just isolated bugs:

- `SymmetrySector` used against the wrong billiard (qb-10) — resolves to a wrong, non-trivial representation.
- `RealPlaneWaves` ignoring a rotation sector (qb-02) — resolves to the full unrestricted basis.
- `CompositeBIMSolver`'s latent multi-ring symmetry misfold (bg-02/qb-04) — would silently fold across ring boundaries.
- `StarBilliard`/`MushroomBilliard` off-center construction (bg-05) — silently broken, non-closed/non-simple geometry.
- `generalized_eigen*`'s empty-`mu`/`Z` return on full rank-deficiency (qb-03) — silently returns nothing instead of erroring.
- KrylovKit `svdsolve`'s discarded `info.converged` across every BIM solver (qb-04) — non-convergence silently accepted as correct.
- `overlap_and_merge_ebim!`'s adaptive clustering merging genuinely distinct close levels (qb-05, qb-07) — silently discards one of two real, physically distinct roots with no way to inspect what was dropped.
- Beyn contour-boundary double-counting (qb-05) — a root at a shared window edge is silently counted **twice**.
- `momentum_function`'s uniform-grid detector checking only `s`, not `ds` (unlike `husimi_function`'s already-fixed dual check) (qb-09) — could silently produce a wrong momentum spectrum for a non-uniform-weight discretization.
- Chebyshev panel clamping silently extrapolating out-of-range `r` instead of erroring, now pinpointed to a reachable near-boundary wavefunction-reconstruction code path (qb-06, qb-09).
- `Polygon`/`MultiplyConnectedDomain`/`CircleSegment` degenerate-input silent acceptance (negative radius, 1-2 corners, empty hole groups) (bg-03).

**Concurrency/exception-safety (cross-cutting, hits nearly every solver):** the non-restoring, non-`try/finally` `@blas_1`/`@blas_multi_then_1`/`@blas_multi` macros (qb-01) are called from `solve`/`solve_vect`/`solve_state` in **every** sweep and accelerated BIM/basis solver (qb-03, qb-04, qb-05 each independently confirm new call sites), and are confirmed to run under `BeynSolver`'s genuinely multithreaded per-window loop — this is a real, not hypothetical, global-state data race.

## 6. Consolidation opportunities

- **`angle(a,b)` duplicated verbatim** between `BilliardGeometry.jl/utils.jl` and `QuantumBilliards.jl/utils/geometryutils.jl` (qb-01, cross-referencing bg-01) — `BilliardGeometry.jl` doesn't export `angle` (it legitimately extends `Base.angle`), which is presumably why `QuantumBilliards.jl` re-implements it. **Recommendation now that both sides are visible**: export `angle` from `BilliardGeometry.jl` and have `QuantumBilliards.jl` reuse it via `using BilliardGeometry`, rather than maintaining two copies of the same one-line formula. Low risk, `Base.angle`-extension semantics are unaffected either way.
- **`_orientation_reversing` reflection-orientation logic re-derived by hand in `QuantumBilliards.jl`'s `reflections.jl`** instead of reused from `BilliardGeometry.jl`'s own (private) `_orientation_reversing` in `fullboundary.jl` (qb-10) — a genuine duplicated-fact-across-the-package-boundary risk that would silently drift if either side's reflection-type roster ever changes (e.g. a new billiard finally using `DiagonalReflection`). Recommend exporting `_orientation_reversing` (or an equivalent) from `BilliardGeometry.jl`.
- **`apply_symmetry`'s 10 near-identical reflection methods** and **`symmetry_index_orbits`'s 4 near-identical index-permutation methods** in `BilliardGeometry.jl` (bg-02) — both could collapse to one generic method via the existing `_sym_matrix`-style trait table already present in `fullboundary.jl`; doing so would also fix the `Float64`-hardcoding in §4 "for free."
- **The `circshift`-in-loop allocation** in `kress_R_even!`/`kress_R_odd!` (bg-04) remains the single highest-value performance fix in the whole audit, given it sits under every BIM solver's `construct_matrices`.
- **`_dlp_evaluate_points`/`_cfie_evaluate_points` byte-for-byte duplication**, and the **`D(k)` kernel algebra duplicated rather than shared** between `dlp.jl`/`cfie.jl` (contrary to the migration plan's explicit intent) (qb-04) — both real, in-scope consolidation targets.
- **The negligible-coefficient filtering block duplicated 3x** across `BasisEigenstate`'s two constructors and `BIMEigenstate`'s constructor (qb-08) — a single private helper would also resolve the `set_precision(vec[1])` vs. `set_precision(real(vec[1]))` inconsistency in one place.
- **The de Broglie grid-sizing formula copy-pasted 3x** in `wavefunctions.jl` (qb-09).
- **Two independently-written "is this arc-length grid uniform" checks** (`momentum_function`'s vs. `_bim_boundary_uniformly_spaced`) with different criteria (qb-09) — unifying them would simultaneously fix the §5 correctness gap.
- 4 "quadrant D2-symmetric" billiard constructors' boilerplate, and `C3Billiard`/`StarBilliard`'s near-identical shape (bg-05) — both good candidates for a shared helper/parametric constructor now that 4-5 real instances exist.

## 7. Export audit — definitive merged list

Superseding all per-step lists; see §2 above for the full missing-exports tables (QuantumBilliards.jl: 9 symbols; BilliardGeometry.jl: none) and the dead-export list (`print_benchmark_info`, `BoundaryPointsSM`, `BoundaryPointsDM`, `construct_matrices_benchmark`). No new export gaps were found in this step's own file scope ([QuantumBilliards.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/QuantumBilliards.jl), [BilliardGeometry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/BilliardGeometry.jl)) beyond what's already listed.

## 8. Broken external call sites (downstream impact of renames/removals)

Grep-level pass over `QBPlotting.jl`, `BilliardGeometryPlotting.jl`, `QCPlotting.jl`, `SpectralStatistics.jl`:

- **`QCPlotting.jl` and `SpectralStatistics.jl` have zero references to `QuantumBilliards`/`BilliardGeometry`** — they don't depend on either package. No exposure, nothing to check further for these two.
- **`BilliardGeometryPlotting.jl`** references only `BilliardGeometry.AbsCompositeDomain`, `AbsSimpleDomain`, `Transparent`, `SymmetryWall`, `AbsCompositeCurve` — all confirmed stable/unchanged across all 5 bg-* steps. **No broken call sites.**
- **`QBPlotting.jl` — CONFIRMED HIGH-SEVERITY BREAKAGE**: `husimiplotting.jl` (both `plot_husimi_function!` methods) and `boundaryfunctionsplotting.jl` (both functions) call `pb_sectors(billiard)` unconditionally, with **no guard for rotation-symmetric billiards**. Since `pb_sectors` → `apply_symmetry_pb` has no `NFoldRotation` method (bg-02/bg-05's tracked gap), **any user calling `plot_husimi_function!` or the boundary-function plotting entry points on a `C3Billiard` or `StarBilliard` hits a `MethodError` today** — this is not a latent risk, it's a live, reachable break in a sibling package caused by a gap in the audited packages. No other renamed/removed symbol from the 15 prior findings (`PolarSegment`, `symmetry_irrep_character`, `CircleWedge`/`CircleCap`, `reset_ids!`, `k_at_state`, `merge_spectra`, `AbsState`) is referenced by `QBPlotting.jl` — everything else it uses (`QuantumBilliards.AbsState`, `.BasisState`, `.BIMEigenstate`, `.AbsBasis`, `.AbsBilliard`, `.SweepBasisSolver`, `.SweepBIMSolver`, `.AcceleratedBasisSolver`) is accessed via fully-qualified names, which resolve correctly regardless of export status (Julia's `export` only affects `using`-brought-into-scope names, not qualified access) — so the `AbsState`-not-exported gap does **not** break `QBPlotting.jl` in practice, unlike the `pb_sectors` gap.

---

## Summary

The audit's original motivating question — "which billiard symmetries don't actually work" — has three concrete, confirmed answers spanning both packages and reaching all the way into a sibling plotting package: **`LimaconBilliard`/`MushroomBilliard`'s `full_boundary`** (BIM-solver path) and **`C3Billiard`/`StarBilliard`'s `pb_sectors`** (Husimi/PB-plotting path, now confirmed broken in `QBPlotting.jl` itself). Beyond that headline finding, the strongest cross-cutting signal from consolidating all 15 steps is a systemic pattern of **silent wrong-result fallbacks at trust boundaries** (symmetry-sector/billiard mismatches, ignored rotation sectors, discarded convergence info, adaptive-clustering data loss) rather than isolated one-off bugs — this pattern, plus the newly-confirmed 5 unused Project.toml dependencies and the exported-but-undefined `merge_spectra`, are the highest-value items for a follow-up fix session before any new features are added.
