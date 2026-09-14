Date produced: 2026-09-14

# Consolidated Findings & Triage — QuantumBilliards.jl Audit (Steps qb-01 through qb-11)

This document consolidates and **triages** the eleven per-step findings files
produced by the `QuantumBilliards.jl` portion of the whole-ecosystem
code-quality audit (`00-index.md`), in the same style as
[00-consolidated-bg-findings.md](00-consolidated-bg-findings.md) already did
for `BilliardGeometry.jl`. It does **not** replace those files — each
retains full detail, file/line references, and per-checklist-category
breakdowns. This document re-organizes everything by
**severity/actionability**, dedupes cross-references between steps, and adds
an explicit **triage decision** (Fix now / Fix soon / Defer / Accept-as-is)
for every item so a follow-up remediation pass can start directly from this
file without re-reading all eleven.

Source files consolidated:
- [qb-01-abstracttypes-utils-findings.md](qb-01-abstracttypes-utils-findings.md)
- [qb-02-basis-findings.md](qb-02-basis-findings.md)
- [qb-03-shared-solver-infrastructure-findings.md](qb-03-shared-solver-infrastructure-findings.md)
- [qb-04-sweep-bim-solvers-findings.md](qb-04-sweep-bim-solvers-findings.md)
- [qb-05-accelerated-solvers-findings.md](qb-05-accelerated-solvers-findings.md)
- [qb-06-chebyshev-findings.md](qb-06-chebyshev-findings.md)
- [qb-07-spectra-findings.md](qb-07-spectra-findings.md)
- [qb-08-eigenstates-basisstates-findings.md](qb-08-eigenstates-basisstates-findings.md)
- [qb-09-wavefunctions-boundary-husimi-findings.md](qb-09-wavefunctions-boundary-husimi-findings.md)
- [qb-10-symmetry-states-findings.md](qb-10-symmetry-states-findings.md)
- [qb-11-cross-package-integration-findings.md](qb-11-cross-package-integration-findings.md) — the whole-ecosystem executive summary; cross-package/dependency items below are sourced from it.

This is a **read-only synthesis** — no source files were edited to produce
it. Per the plan's confirmed convention, promotion into
`/home/clozej/.julia/dev/QuantumBilliards.jl/memories/repo/` (and a
combined-ecosystem summary) is the separate, later step that follows this
triage, not part of it.

**Triage legend:**
- **P0 (fix now)** — broken/exported-but-non-functional public API, or a confirmed silent-wrong-result bug reachable from realistic usage.
- **P1 (fix soon)** — real correctness/robustness gap, not yet reachable by any shipped billiard/workflow but will break the first time one is added, or a significant type-stability/precision violation.
- **P2 (worth doing, low urgency)** — consolidation, missing export, dead code, docstring/status cleanup, minor performance.
- **Defer** — correctly deferred already per an existing plan (e.g. no fixture exists yet to need it).
- **Accept-as-is** — investigated and confirmed to be an intentional/harmless design choice, not a defect.

---

## 1. Critical: broken/exported-but-non-functional APIs & silent-wrong-result bugs

| # | Finding | Location | Root cause | Step | Triage |
|---|---|---|---|---|---|
| C1 | `merge_spectra` is **exported** and **unconditionally throws `UndefVarError`** | [spectralutils.jl#L193-L213](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/spectralutils.jl#L193-L213) | Calls `interval`/`intersect_interval`/`in_interval` — none defined anywhere; `IntervalArithmetic` is declared in `Project.toml` but never `using`'d | qb-07, qb-11 | **P0** — either implement with plain scalar interval logic (no new dependency needed) or remove from the export list entirely; zero call sites anywhere, so removal has no blast radius |
| C2 | `SymmetrySector` has **no billiard-identity check** despite storing `sector.billiard` | [symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl#L114-L117) | `_resolve_bim_symmetry`/`RealPlaneWaves(dim,billiard,sector)` only check "does *this* billiard register a symmetry with this `sym_id`", not "is this the *same* billiard the sector was built for" — every `sym_id` is assigned by registration order, so any two D2-symmetric billiards collide | qb-10 | **P0** — silently produces a wrong-but-non-trivial representation for the single most common billiard family (D2); fix is one `===`/equality check using the already-stored field |
| C3 | `RealPlaneWaves` **silently ignores** an `NFoldRotation` `SymmetrySector` | [realplanewaves.jl#L225-L237](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl#L225-L237) | Only looks for `XAxisReflection`/`YAxisReflection` matches; a rotation-only sector matches nothing and falls through to "no symmetry" | qb-02 | **P0** — confirmed live via direct execution; returns the full unrestricted basis with no error/warning when the caller explicitly asked for a `Z_n` sector |
| C4 | `CornerAdaptedFourierBessel`'s type parameter is **hardcoded `{Float64,Nothing}`** regardless of input types | [corneradapted.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/fourierbessel/corneradapted.jl) (all 4 constructors) | Never derives `T` from `corner_angle`/`origin`/`rot_angle`'s actual eltype | qb-02 | **P0** — breaks the package's own `toFloat32` conversion function outright (reproduced live: `MethodError` converting `PolarCS{Float32}` to `PolarCS{Float64}`); also silently discards all symmetry info via the dead `Sy` param |
| C5 | `CornerAdaptedFourierBessel` has **zero** `(dim,billiard,sector::SymmetrySector)` constructor | [corneradapted.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/fourierbessel/corneradapted.jl) | No overload exists at all; legacy `symmetries` field is untyped, unconnected to `SymmetrySector`, and never read by evaluation | qb-02 | **P1** — total absence, not partial like C3; needed before corner-adapted bases can be used with any symmetry sector at all |
| C6 | 4 confirmed **unused Project.toml dependencies** in `QuantumBilliards.jl` | `Project.toml` | `ForwardDiff`, `FastGaussQuadrature`, `StatsBase`, `IntervalArithmetic` — zero references anywhere in `src/` | qb-11 (new, confirmed by direct grep) | **P1** — `IntervalArithmetic`'s absence is the direct root cause of C1; the other 3 are pure removal candidates (verify no docs/test-only usage before dropping) |

**Recommended fix order**: C1 and C3 are the fastest, highest-value fixes
(C1 = delete or trivially reimplement; C3 = add an explicit
`error("rotation sectors not yet supported by RealPlaneWaves")` at minimum,
ideally implement rotation-sector support). C2 is the most *consequential*
silent-correctness bug in the whole `QuantumBilliards.jl` half of the audit
and should be scoped as its own `julia-refactor-scoped` session alongside
its `BilliardGeometry.jl`-side twin (bg's C3/C4, `apply_symmetry_pb`). C4/C5
should be scoped together as one "corner-adapted basis type/symmetry
parity" session. C6 is a 15-minute `Project.toml` cleanup, but do it
*after* C1 is resolved (in case the fix path chosen for C1 is "actually
implement using `IntervalArithmetic`").

---

## 2. Type instability & precision correctness

| Finding | Location | Step | Triage |
|---|---|---|---|
| `generalized_eigen`/`generalized_eigvals` hardcode `1.0` instead of `one(eltype(d))` | [decompositions.jl#L32-L57](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/decompositions.jl#L32-L57) | qb-03 | **P1** — silently promotes a `Float32` solve to `Float64` internally; one-line fix |
| `ParticularSolutionsMethod.sampler::Vector` is untyped (`Vector`, not `Vector{<:AbsSampler}`) | [particularsolutions.jl#L44](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/particularsolutions.jl#L44) | qb-04 | **P2** — faithfully replicates a pre-existing `DecompositionMethodSolver` pattern (out of qb-04's scope); fix both together if ever revisited |
| `wavefunction`'s keyword surface diverges between `BasisEigenstate`/`BIMEigenstate` (Chebyshev toggles vs. `fundamental_domain`/`memory_limit`), unlike the other 3 state-evaluation functions which keep one shared keyword set | [wavefunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl) | qb-09 | **P2** — API polish; would compound if/when `method=:green` (see §3) is added generically |
| `boundary_limits` uses `Vector{Any}()` for `x_bnd`/`y_bnd` | [wavefunctions.jl#L69-L70](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl#L69-L70) | qb-09 | **P2** — easy fix (`Vector{Float64}()` or infer `T`), called once per reconstruction, not hot-path |
| `set_precision(vec[1])` (real-or-complex-unsafe) vs. `set_precision(real(vec[1]))` inconsistency between `BasisEigenstate`/`BIMEigenstate` constructors | [eigenstates.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/eigenstates.jl) | qb-01 (flagged), qb-08 (confirmed reachable failure mode) | **P1** — currently harmless only because every basis produces real eigenvectors; would throw `MethodError: no method matching isless(::ComplexF64,::Float64)` the moment a complex-valued basis exists. Fix alongside the §6 consolidation (one shared filtering helper) |
| `momentum_function`'s uniform-grid check tests only `s`, not `ds` (unlike `husimi_function`'s already-fixed dual check) | [boundaryfunctions.jl#L282-L288](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/boundaryfunctions.jl#L282-L288) | qb-09 | **P1** — could silently produce a wrong momentum spectrum for a non-uniform-weight discretization; same class of bug already fixed once elsewhere in this file set |
| `utils/typeutils.jl`'s `set_precision` gives `Complex{Float32}`/`BigFloat` the same `1e-16` threshold as `Float64`, not scaled to their actual precision | [typeutils.jl#L2-L5](/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/typeutils.jl#L2-L5) | qb-01 | **P2** — inconsistent precision floor, not currently triggering a known bug |
| `utils/billiardutils.jl` builds `Vector{AbsBoundaryCondition}(undef,3)` — abstractly-typed container | [billiardutils.jl#L34](/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/billiardutils.jl#L34) | qb-01 | **P2** — 3-element vector, built once per triangle construction, not a hot loop |

---

## 3. Missing implementations (beyond §1's critical items)

| Finding | Location | Step | Triage |
|---|---|---|---|
| `wavefunction(state::BasisEigenstate)`'s planned `method::Symbol=:green` keyword (migration-plan 05.6 "definition of done") was never implemented | [wavefunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl) | qb-09 | **P1** — real, unflagged-anywhere-as-intentional gap against the plan's own stated done-criteria; `compute_psi`'s dense path is still the only option |
| `compute_psi` has no `BIMEigenstate` method despite `AbsState`'s docstring implying universality | [eigenstates.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/eigenstates.jl) / consumer in [wavefunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl) | qb-08 | **P1** — currently latent (nothing calls it directly on a BIM state yet); throws an unclear field-access error if ever invoked |
| `AbsState`'s documented API oversells `BasisState`'s actual contract (`boundary_function`/`momentum_function`/`husimi_function` would all field-error) | [abstracttypes.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/abstracttypes.jl) | qb-01, qb-08 | **P2** — docstring-only fix: narrow the wording rather than implement the full contract for `BasisState` |
| `CompositeBIMSolver`'s docstring claims "API scaffold only, every method raises an error" | [compositebim.jl#L28-L34](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl#L28-L34) | qb-04 | **P2** — stale/false; solver is fully implemented. Delete the note |
| Migration-plan 16 (BIM `SymmetrySector`) status header says "plan only, not started" | `migration-plan/16-bim-symmetrysector-migration.md` | qb-04, qb-10, qb-11 (three independent confirmations) | **P2** — doc-only fix: update the status header, code is fine |
| `CompositeBIMSolver`'s Gap 2 (multi-ring `symmetry_index_orbits`) — confirmed still open | [compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl), depends on `BilliardGeometry.jl`'s `symmetryorbits.jl` | qb-04 (QuantumBilliards.jl side), bg-02/bg-05 (BilliardGeometry.jl side) | **Defer** — correctly deferred; no symmetric multiply-connected billiard exists yet to exercise this. Do not fix speculatively; fix together with bg's Gap 2 once a real fixture exists |
| Beyn/EBIM's `use_chebyshev=true` default has no fail-fast `T===Float64` check at construction | [beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl), [ebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl) | qb-05 | **P2** — construction-time footgun, not a numerical-correctness bug; easy guard to add |
| Chebyshev panel-index cache (Gap 3) and combined H0/H1 scalar evaluator (Gap 5) | [chebyshev/*.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/chebyshev/) | qb-06 | **Defer** — explicitly deferred/profiling-gated by the 10.5 plan itself, not a regression |
| `BeynSolver.construct_matrices`/`ExpandedBIMSolver.construct_matrices*` lack `@timeit_debug` instrumentation present in sibling `VerginiSaracenoSolver` | [beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl), [ebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl) | qb-05 | **P2** — good candidate for the `julia-solver-debug-timing` skill in a dedicated pass |

---

## 4. Vulnerabilities (systemic pattern: silent-wrong-result at trust boundaries)

The single strongest cross-cutting signal from this audit: **the same
"silently produce a wrong-but-plausible result instead of erroring" failure
mode recurs independently across 8+ unrelated subsystems.** This is a
systemic gap in input validation at trust boundaries, not a collection of
unrelated bugs — worth treating as one theme for remediation planning
(e.g. a shared checklist/convention for new solver/basis code: "does this
silently substitute a default when the input doesn't match what's
expected?").

| Finding | Location | Step | Triage |
|---|---|---|---|
| `SymmetrySector` used against the wrong billiard (§1 C2) | symmetrysector.jl | qb-10 | **P0** (see §1) |
| `RealPlaneWaves` ignoring a rotation sector (§1 C3) | realplanewaves.jl | qb-02 | **P0** (see §1) |
| KrylovKit `svdsolve`'s `info.converged` discarded in every BIM solver's `solve`/`solve_vect`/`solve_state` | dlp.jl, cfie.jl, compositebim.jl, sweepmethods.jl | qb-04 | **P1** — uniform across 3+ solvers; add a convergence check + warning at the one shared `solve_state` call site if possible |
| `generalized_eigen`/`generalized_eigvals` silently return empty `mu`/`Z` on full rank-deficiency instead of erroring | [decompositions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/decompositions.jl) | qb-03 | **P2** — genuinely degenerate input is rare; still worth an explicit check + clear error |
| `overlap_and_merge_ebim!`'s adaptive clustering can merge genuinely distinct, closely-spaced levels with no way to inspect what was discarded | [spectralutils.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/spectralutils.jl) | qb-05, qb-07 (independently confirmed) | **P2** — inherent to the clustering design, not a coding bug; consider exposing the discarded candidate via `control`/a debug log rather than changing the tolerance |
| Beyn contour-boundary double-counting (root at shared window edge counted twice) | [beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl) (`_beyn_solve_core`) | qb-05 | **P2** — floating-point-exact edge case in practice; fix via `>=`/`<` boundary convention or a dedup pass if revisited |
| Chebyshev panel clamping silently extrapolates out-of-range `r` instead of erroring — pinpointed to a reachable near-boundary wavefunction-reconstruction path | [bessels.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/chebyshev/bessels.jl), consumed by [wavefunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl) | qb-06, qb-09 | **P1** — the two steps together make this a *reachable*, not just theoretical, accuracy risk for grid points near the boundary |
| `ϕ_slp`'s near-singular kernel only guarded for **exact** coincidence (`r2==0`), not near-boundary grid points | [wavefunctions.jl#L131-L146](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl#L131-L146) | qb-09 | **P1** — real accuracy risk for `inside_only=true` grid points close to (not on) the boundary |
| `@blas_1`/`@blas_multi_then_1`/`@blas_multi` are non-exception-safe (no `try/finally`) and race on process-global `BLAS.set_num_threads` under `BeynSolver`'s threaded per-window loop; also ignore the caller's `multithreaded=false` flag | [macros.jl#L48-L73](/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/macros.jl#L48-L73) | qb-01, reconfirmed as new call sites by qb-03/qb-04/qb-05 | **P1** — a real, not hypothetical, data race (confirmed threaded call path); also silently ignores an explicit user request to disable threading. Fix once, centrally, in `macros.jl` — every solver inherits the fix for free |
| `adjust_scaling_and_samplers` mutates the solver's own stored fields via aliased `push!` despite presenting as a pure query | [decompositions.jl#L195-L209](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/decompositions.jl#L195-L209) | qb-03 | **P2** — not currently a race (no basis-solver windowed-threading path exists yet), but a latent footgun; add a docstring note or switch to `copy` |
| `_scale_rows!`/`_scale_rows_sqrtw!`/`_build_Bn_inplace!` use `@inbounds` on a second array indexed by the first array's row index, with no length check in the function body itself (only upheld by the caller's `BoundaryPoints` inner constructor) | [matrixconstructors.jl#L123-L169](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/matrixconstructors.jl#L123-L169) | qb-03 | **P2** — low severity today; add a one-line precondition comment |
| `compute_eigenstate(::AcceleratedBasisSolver,...)`'s `findmin` throws an unclear `ArgumentError` on an empty candidate array instead of a clear "no state found in this window" message | [eigenstates.jl#L194](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/eigenstates.jl#L194) | qb-08 | **P2** — error-message quality issue, not silent-wrong-result |
| `_finalize_spectrum`'s docstring claims a "disjoint-slots, multi-thread-safe" pattern that doesn't match any of its 4 actual (all-effectively-single-threaded) call sites | [spectralutils.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/spectralutils.jl) | qb-07 | **P2** — no live race (good), but the docstring's stated rationale is inaccurate; correct the docstring |
| `overlap_and_merge!`'s tail-append logic relies on an unenforced "`k_right` is sorted / overlap window is a contiguous prefix" invariant | [spectralutils.jl#L52-L94](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/spectralutils.jl#L52-L94) | qb-07 | **P2** — currently upheld structurally by callers; add an `@assert issorted(...)` guard if ever revisited |

---

## 5. Dead code

| Symbol | Location | Note | Step | Triage |
|---|---|---|---|---|
| `merge_spectra` | spectralutils.jl | Dead **and** broken — see §1 C1 | qb-07 | **P0** (see §1) |
| `dim`, `k_basis`, `eps` fields on `BasisState`/`BIMEigenstate` | eigenstates.jl / basisstates.jl | Write-only struct fields, never read | qb-01 (flagged), qb-08 (confirmed) | **P2** — safe to remove, but a breaking struct-field change; batch with other struct cleanups |
| `polar_to_cartesian`, `cartesian_to_polar` | coordinatesystems.jl | Exported in `-develop`, dropped here, unused | qb-01 | **P2** — either export (parity with `-develop`) or delete |
| `directsum` | decompositions.jl | Documented, zero call sites | qb-03 | **P2** — export if intended as public utility, else delete |
| `_scale_rows!` | matrixconstructors.jl | Superseded by `_scale_rows_sqrtw!` | qb-03 | **P2** — delete |
| `_dlp_fredholm_full_cheb!`, `_dlp_fredholm_reduced_cheb!`, `_cfie_fredholm_full_cheb!`, `_cfie_fredholm_reduced_cheb!` | chebyshev/dlp.jl, chebyshev/cfie.jl | Orphaned by the multi-k batch rewrite; multi-k versions are a strict superset | qb-06 | **P2** — delete (not "consolidate", per qb-06's own framing) |
| `_cheb_clenshaw_col` + 10 matrix-arg `_cheb_clenshaw_*` overloads | chebyshev/core.jl | Never match main's per-panel-vector data layout | qb-06 | **P2** — delete |
| `panel_and_geom` | chebyshev/bessels.jl | The exact primitive Gap 3's (deferred) panel-index cache would consume | qb-06 | **Defer** — keep until Gap 3 is implemented, then either use it or delete |
| `sm_results` | verginisaraceno.jl | Fully documented, called only within its own file | qb-05 | **P2** — underscore-prefix (`_sm_results`) or export; recommend the former |
| Exported-but-never-defined: `print_benchmark_info`, `BoundaryPointsSM`, `BoundaryPointsDM`, `construct_matrices_benchmark` | QuantumBilliards.jl export list | `UndefVarError` if ever accessed | qb-01 | **P1** — remove from export list; these are actively misleading about the package's public surface |
| `CornerAdaptedFourierBessel.symmetries` field / `Sy` type parameter | corneradapted.jl | Stored, threaded through `resize_basis`, never read by any evaluation method | qb-02 | **Superseded by C5** — resolve as part of the corner-adapted symmetry-support fix, not standalone |

---

## 6. Consolidation opportunities

| Finding | Location | Step | Triage |
|---|---|---|---|
| `angle(a,b)` duplicated verbatim vs. `BilliardGeometry.jl`'s own `angle` | [geometryutils.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/utils/geometryutils.jl) vs. `BilliardGeometry.jl/src/geometry/utils.jl` | qb-01, qb-11 | **P2** — export `angle` from `BilliardGeometry.jl` and reuse; low risk |
| `_orientation_reversing` reflection-orientation logic re-derived by hand in `reflections.jl` instead of reused from `BilliardGeometry.jl`'s own (private) version | [reflections.jl#L224-L226](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/reflections.jl#L224-L226) | qb-10, qb-11 | **P1** — genuine duplicated-fact-across-package-boundary risk; would silently drift if either side's reflection roster changes. Export `_orientation_reversing` (or equivalent) from `BilliardGeometry.jl` |
| 3 near-identical `has_x`/`has_y`/4-way-branch skeletons across `apply_symmetries_to_wavefunction`/`_to_boundary_function`/`_to_boundary_points` | [reflections.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/reflections.jl) | qb-10 | **P2** — partial shared helper for `has_x`/`has_y`/lookup would help; full unification has diminishing returns (different data shapes) |
| `_dlp_evaluate_points`/`_cfie_evaluate_points` byte-for-byte duplication | dlp.jl, cfie.jl | qb-04 | **P1** — genuine copy-paste, not "DLP + extra term"; lift into one shared helper |
| `D(k)` kernel algebra duplicated (not shared, contrary to migration plan 06's explicit intent) between DLP's and CFIE's dense "full" assembly functions | dlp.jl, cfie.jl | qb-04 | **P1** — extract a shared `@inline` scalar helper for `(l1,l2)` from `(αL1,αL2,inn,j1,h1,invr,lt)`; zero performance cost |
| `boundary_matrix_size(solver,pts)` defined identically 3 times (DLP/CFIE/Composite) | dlp.jl, cfie.jl, compositebim.jl | qb-04 | **P2** — hoist to one generic method on `AbsBIMSolver` |
| Negligible-coefficient filtering block duplicated 3x across `BasisEigenstate`'s two constructors and `BIMEigenstate`'s constructor | [eigenstates.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/eigenstates.jl) | qb-08 | **P1** — single helper also fixes the §2 `set_precision` inconsistency in one move |
| De Broglie grid-sizing formula copy-pasted 3x | [wavefunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl) | qb-09 | **P2** — trivial helper extraction |
| Two independently-written "is this arc-length grid uniform" checks with different criteria (`momentum_function`'s vs. `_bim_boundary_uniformly_spaced`) | boundaryfunctions.jl, husimifunctions.jl | qb-09 | **P1** — unifying simultaneously fixes the §2/§4 correctness gap; do this and the type-instability fix together |
| `matrixconstructors.jl`'s "# INTERNAL FUNCTIONS" banner convention not adopted by sibling solver files | solvers/*.jl | qb-03 | **P2** — cosmetic, no behavior change |
| `_bim_numeric_type` defined identically for `BeynSolver`/`ExpandedBIMSolver` | beyn.jl, ebim.jl | qb-05 | **Accept-as-is** — structurally forced duplication (no type parameter on the abstract type); not worth a larger design change |
| Item 5 (qb-06): redundant `z`/cutoff recomputation in `_dlp_kernel_entry_with_derivatives_cheb`/CFIE analogue | chebyshev/dlp.jl, chebyshev/cfie.jl | qb-06 | **Defer** — explicitly profiling-gated by the 10.5 plan |

---

## 7. Export audit — consolidated

**Missing exports (public-quality, documented, and/or `@ref`-cross-referenced, but not exported):**

- `AbsState` — the most significant single gap; every sibling abstract branch is exported. (qb-01, qb-08)
- `k_at_state` — direct inverse of exported `state_at_k`, `@ref`-cross-referenced by 3 other exported docstrings. (qb-07)
- `adjust_scaling_and_samplers` — documented `AbsSolver`/`AbsBasisSolver` API member. (qb-01, qb-03)
- `CoordinateSystem`, `CartesianCS`, `PolarCS` (qb-01)
- `make_veech_right_triangle`, `make_veech_right_triangle_and_basis` — accessed only via fully-qualified calls in sibling test scripts. (qb-01)
- `generalized_eigen`, `generalized_eigvals`, `generalized_eigen_all` — actively used by every basis solver, referenced in the pre-built docs HTML. (qb-03)
- `random_interior_points`, `solve_full`, `solve_with_rank_reduction`, `symmetrize_layer_density` (qb-04)
- `apply_symmetries_to_wavefunction` (doc'd but not exported), `apply_symmetries_to_boundary_function`, `apply_symmetries_to_boundary_points` — inconsistent doc/export tiers among 3 functions of identical standing. (qb-10)
- `sm_results` — recommend underscore-prefixing instead (see §5). (qb-05)

**Exported but never defined (remove from export list):** `print_benchmark_info`, `BoundaryPointsSM`, `BoundaryPointsDM`, `construct_matrices_benchmark`. (qb-01)

**Triage**: batch all of the above into a single, mechanical "export list cleanup" PR (add the 14 missing exports, remove the 4 dead ones) — **P1** for the dead-export removal (actively misleading), **P2** for the missing-export additions (no functional impact until someone tries to use them without `QuantumBilliards.` qualification).

---

## 8. Docstring & migration-plan status staleness (flag only, no source-editing needed)

- `SweepBasisSolver`'s docstring says "the concrete implementation is `DecompositionMethodSolver`" (singular) — should say "implementations are" and list `ParticularSolutionsMethod` too. (qb-01) — **P2**
- `AbsState`'s docstring has no `abstract type AbsState end` statement following it — the orphaned docstring attaches to nothing; the real (undocumented) declaration is earlier in the file. (qb-01) — **P2**
- `docs/src/API.md`/`docs/build/API.html` are stale — predate the BIM-solver branch and `AbsState` entirely, and the generated build references retired names (`AcceleratedSolver`, `SweepSolver`, `Eigenstate`, `ca_fb`, `ca_fb_dk`, `boundary_coords`). (qb-01) — **P2**, batch with a docs-regeneration pass
- `CompositeBIMSolver`'s "API scaffold only" docstring note — stale/false. (qb-04) — **P2**, delete the note
- Migration-plan 16's "plan only, not started" status header — stale; code is fully implemented. (qb-04, qb-10, qb-11) — **P2**, update the header
- `merge_spectra` has no docstring at all, unlike every other exported function in the file — consistent with being unfinished. (qb-07) — resolved automatically once §1 C1 is fixed or removed

---

## 9. Cross-cutting / cross-package items (sourced from qb-11's synthesis)

These involve both `QuantumBilliards.jl` and `BilliardGeometry.jl` and/or a
sibling plotting package; full detail lives in
[qb-11-cross-package-integration-findings.md](qb-11-cross-package-integration-findings.md).

| Finding | Triage |
|---|---|
| `pb_sectors`/`apply_symmetry_pb` has no `NFoldRotation` method → **confirmed live breakage** in `QBPlotting.jl`'s `plot_husimi_function!` and boundary-function plotting for `C3Billiard`/`StarBilliard` | **P0** — this is a `BilliardGeometry.jl`-side fix (tracked as bg's C3/C4 in [00-consolidated-bg-findings.md](00-consolidated-bg-findings.md)) but is flagged here again because it's the most severe *reachable* downstream breakage found in the whole audit |
| `LimaconBilliard`/`MushroomBilliard.full_boundary` broken | **P0** — `BilliardGeometry.jl`-side (bg's C1/C2); blocks any BIM solve on these two billiards |
| Reflection transforms/`NFoldRotation` hardcoded to `Float64` in `BilliardGeometry.jl`'s `symmetry.jl`, directly consumed by `QuantumBilliards.jl`'s solver layer | **P1** — cross-package precision leak; `BilliardGeometry.jl`-side fix, `QuantumBilliards.jl` is just a downstream consumer |
| `LinearNodes`/`GaussLegendreNodes` hardcoded to `Float64`, used as the default sampler by `DecompositionMethodSolver`/`ParticularSolutionsMethod`/`VerginiSaracenoSolver` | **P1** — cross-package; any attempt to run these solvers in non-`Float64` precision silently falls back to `Float64` sampling |
| 5 confirmed unused `Project.toml` dependencies total (4 in `QuantumBilliards.jl` — see §1 C6 — 1 `StatsBase` in `BilliardGeometry.jl`) | **P1**/**P2** split — see §1 |
| `QCPlotting.jl`/`SpectralStatistics.jl` have zero references to either audited package | **Accept-as-is** — no exposure, nothing to check |
| `BilliardGeometryPlotting.jl` references only stable, unchanged symbols | **Accept-as-is** — no broken call sites |

---

## 10. Status of every previously-tracked gap (before/after this audit)

| Tracked item | Source | Status after qb-01..qb-11 |
|---|---|---|
| Step 05.6 general non-uniform Husimi fallback | migration-plan/05.6 | **Confirmed fully implemented**, fully automatic, no manual opt-in, old `@warn`-only path fully gone |
| Step 05 BIM wavefunction/Husimi/Green scope (DLP-only vs. general `SweepBIMSolver`) | migration-plan/05 | **Confirmed widened correctly** to all `SweepBIMSolver`s, not left DLP-only |
| Step 05.6 `method=:green` keyword for `BasisEigenstate.wavefunction` | migration-plan/05 §5.6 | **Confirmed NOT implemented** — genuine, unflagged gap (§3) |
| 10.5 Chebyshev gap analysis, all 5 gaps | migration-plan/10.5 | Gaps 1, 2, 4 **confirmed fixed and wired**; Gaps 3, 5 **confirmed still open, correctly deferred** per the plan's own status |
| Step 16 BIM/`SymmetrySector` migration ("plan only, not started") | migration-plan/16 | **Status label confirmed stale** — fully implemented, matches design almost verbatim (3 independent confirmations: qb-04, qb-10, qb-11) |
| `CompositeBIMSolver`'s Gap 2 (multi-ring symmetry folding) | geometry-audit-10.9.md / bg-02 | **Confirmed still open, correctly deferred** — no fixture exists yet |
| 12.5 `area`/`fundamental_area`/`corner_angles` migration | migration-plan/12.5 | **Confirmed complete** — clean migration to `BilliardGeometry.jl`, nothing half-migrated (qb-07, cross-confirmed with bg-01) |
| 09.5 `compute_spectrum` design (4→6 dispatch variants) | migration-plan/09.5 | **Confirmed complete and correctly dispatching**, all end in `_finalize_spectrum` |

---

## Suggested next actions (triage complete — this is the actionable output)

**Tier 1 — fix immediately (P0), highest value / lowest risk:**
1. C1 `merge_spectra` — remove from exports or fix (trivial logic, no dependency needed).
2. C3 `RealPlaneWaves` silently ignoring rotation sectors — at minimum, raise an explicit error.
3. C2 `SymmetrySector` cross-billiard identity check — one-line guard using the already-stored field.
4. C4 `CornerAdaptedFourierBessel` type-parameter hardcoding — derive `T`/`Sy` from inputs like `RealPlaneWaves` does.

**Tier 2 — scope as dedicated sessions soon (P1):**
5. `@blas_1`/`@blas_multi_then_1`/`@blas_multi` exception-safety + `multithreaded=false` respect — fix once in `macros.jl`, benefits every solver.
6. `ϕ_slp`/Chebyshev near-boundary singular-kernel handling (qb-06+qb-09 combined finding).
7. `_orientation_reversing` cross-package duplication + the `LinearNodes`/`GaussLegendreNodes`/reflection-transform `Float64` hardcoding (coordinate with a `BilliardGeometry.jl` fix session).
8. DLP/CFIE `_dlp_evaluate_points`/`D(k)` kernel duplication — extract shared helpers.
9. `eigenstates.jl`'s negligible-coefficient-filtering triplication + `set_precision` real/complex inconsistency.
10. `momentum_function`/`husimi_function` uniform-grid-check unification.
11. Export list cleanup (add 14 missing, remove 4 dead) — mechanical, batch as one PR.
12. `compute_psi(::BIMEigenstate)` + `wavefunction(...; method=:green)` — scope together as one "wavefunction reconstruction API completeness" session.

**Tier 3 — batch as low-urgency cleanup (P2):** all remaining dead-code removals, docstring/status corrections, consolidation opportunities, and instrumentation gaps listed in §2/§3/§5/§6/§8 above.

**Defer (no action until a fixture/need exists):** `CompositeBIMSolver` multi-ring symmetry folding (needs a symmetric multiply-connected billiard first — coordinate with `BilliardGeometry.jl`'s own deferred Gap 2/7/8), Chebyshev panel-index cache and combined H0/H1 evaluator (explicitly profiling-gated).

**Hand off to Julia Test Writer** (once fixes land): regression tests for C1-C5 above; a test exercising `SymmetrySector` reused across two same-shape-family billiards (C2) to lock in the fix; a test for `RealPlaneWaves`+rotation-sector behavior (C3) once implemented or explicitly erroring.

**Cross-reference**: Tier 1 item 3 (C2) and the `BilliardGeometry.jl`-side C3/C4 (`apply_symmetry_pb`) in [00-consolidated-bg-findings.md](00-consolidated-bg-findings.md) are two halves of the same "does symmetry actually work end-to-end" theme — consider scoping them as a joint remediation effort rather than two fully separate sessions, since both trace back to the same `sym_id`-registration-order design.
