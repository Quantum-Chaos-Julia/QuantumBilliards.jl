# BIM/PSM solver migration — execution plan (Step 2+)

Status: **Execution plan.** [`../BIM-solver-migration-plan.md`](../BIM-solver-migration-plan.md)
("Step 1") already did the inventory, architecture decisions and API scaffolding:
every struct, constructor and method *signature* for every solver in the table
below already exists in `QuantumBilliards.jl`/`BilliardGeometry.jl`, with method
bodies that `error(...)`. This directory is **Step 2**: a fully detailed,
file-by-file plan for replacing those stub bodies with real, numerically
faithful implementations ported from `QuantumBilliards-develop` /
`BilliardGeometry-develop`, extending `BilliardGeometry.jl` where needed, and
updating `QBPlotting.jl` and the test suites as each solver lands.

Read [`../BIM-solver-migration-plan.md`](../BIM-solver-migration-plan.md) §2
(architecture decisions) and §4 (struct/API design) before starting **any**
step below — every step must produce code that matches that already-committed
API, not the raw `-develop` API.

## How to use this plan

Each numbered file in this directory is a **self-contained work order** for one
feature, meant to be handed to a coding agent (or done by hand) one at a time,
in order. A step is not started until the previous ones are `Definition of
done`. Every step file has the same sections: Goal, Preconditions, Source
index (files to read in `-develop`), Files touched in main, Implementation
steps, Performance/fidelity notes, Tests & user verification, Plotting
updates. Do not skip the "Tests & user verification" section of a step to jump
ahead — later steps assume earlier solvers are verified working.

Two skills apply throughout: `julia-refactor-port-develop` (how to port
without copy-pasting non-conforming API/conventions) and `julia-add-solver`
(which methods are free via dispatch on the abstract branch vs. required per
concrete type). Load them at the start of any step that adds/changes a solver.
Test files are never written by the implementing agent directly — every step's
final task is to ask the user to invoke the **Julia Test Writer** subagent.

## Step order and rationale

1. **Quick wins** — finish `DecompositionMethodSolver`/`rellich_origin`
   verification and implement `ParticularSolutionsMethod`. Both are
   `SweepBasisSolver`s that reuse 100% of existing basis-solver infrastructure
   (`matrixconstructors.jl`, `decompositions.jl`, `sweepmethods.jl` generics,
   `compute_eigenstate`). Zero new `BilliardGeometry.jl` work, zero new
   abstractions. Establishes momentum and validates the existing
   Triangle/Stadium test fixtures still pass before touching anything BIM.
2. **Shared BIM infrastructure** — the one unavoidable prerequisite before
   *any* boundary-integral solver body can be written: extend `BoundaryPoints`
   with the parametric/Kress fields, and port the Kress logarithmic-correction
   matrices, symmetry-orbit folding and boundary-geometry caches into
   `BilliardGeometry.jl`. Every BIM solver step below (3, 6, 7, 8 and 9)
   depends on this one. Doing it once, up front, avoids each solver step
   re-deriving its own half-version of this infrastructure.
3. **`DoubleLayerPotentialSolver`** — the simplest Fredholm kernel
   (`A(k) = I - D(k)`, one operator, no combined-field coupling term). Ships
   with `SmoothPeriodicGrading` first (no corner handling at all), then adds
   `GlobalCornerGrading`. This is the reference implementation every other BIM
   solver step's `construct_matrices`/`solve`/`solve_vect` triad follows.
4. **Symmetry & full-boundary infrastructure** — a prerequisite discovered
   while auditing steps 2–3: main's `AbsReflection`/`NFoldRotation` carry no
   irrep parity data, `SymmetryOrbitMap` is missing the reverse
   fundamental-to-full orbit-expansion arrays `-develop` relies on, and
   `DoubleLayerPotentialSolver`'s symmetric-case boundary sampling was left
   unresolved by step 3 (it currently discretizes only the fundamental
   domain, not the true full physical boundary). Also replaces every
   `-develop` billiard's hand-maintained `full_boundary` field with a single
   `full_boundary(billiard)` function reconstructing the complete boundary
   from `fundamental_domain` + `symmetries`. Every later BIM solver
   (steps 6–9) and the wavefunction step below depend on this being correct,
   not just loadable.
5. **BIM wavefunction/boundary-function/Husimi reconstruction (Green's
   function)** — ports the single-layer-potential (`ϕ_slp`) Green's-function
   reconstruction and wires `wavefunction`/`boundary_function`/
   `husimi_function` for `BIMEigenstate`, scoped to `DoubleLayerPotentialSolver`
   only (its density is directly `∂ₙψ`); other BIM solvers get this once they
   exist. Also switches `BasisEigenstate`'s `wavefunction` to use the same
   Green's-function integral by default (cheaper than dense basis-matrix grid
   evaluation), since it already produces the same `(u,pts)` boundary data via
   the existing `boundary_function`. Depends on step 4's `full_boundary`/
   `SymmetryOrbitMap` fixes to correctly expand a symmetry-reduced density.
5.5. **`BIMEigenstate` stores `∂ₙψ` for free; generic `solve_state`** — a
   post-step-5 optimization/generalization
   ([05.5-bim-state-normal-derivative.md](05.5-bim-state-normal-derivative.md)):
   `∂ₙψ`'s adjoint-nullspace computation turns out to be recoverable directly
   from the *same* `KrylovKit.svdsolve` call `solve_vect` already performs
   (primal/adjoint problems are transpose-related), so `BIMEigenstate` now
   stores `pts`/`u`/`bnd_norm` up front via a new generic `solve_state`,
   making `boundary_function`/`momentum_function`/`husimi_function`/
   `wavefunction` plain field reads for **any** `SweepBIMSolver` (not just
   `DoubleLayerPotentialSolver`) — removing the redundant second matrix
   assembly and second Krylov solve step 5 introduced, and generalizing its
   scope ahead of steps 6/7 needing to do so themselves.
5.6. **General (non-uniform-arclength) Husimi quadrature** — a correctness
   fix discovered while auditing step 5.5:
   ([05.6-general-nonuniform-husimi.md](05.6-general-nonuniform-husimi.md))
   `husimi_function`'s only kernel was a fast sliding-window stencil valid
   only for uniformly-spaced arc-length grids; a `GlobalCornerGrading` BIM
   solver's non-uniform corner-clustered grid previously only triggered a
   `@warn` with no correct fallback. Ports `-develop`'s general windowed
   (`searchsortedfirst`/`searchsortedlast`, physical-`ds`-weighted)
   quadrature core, adjusted to match main's existing prefactor/sign
   conventions (not a verbatim port — see the step file for why), and wires
   it in as an automatic fallback for both `BIMEigenstate` and generic
   `AbsState` callers. Confirms `BIMEigenstate` needed no new fields:
   `state.pts` (added in step 5.5) already carries both `s` and `ds`.
6. **`CombinedFieldIntegralEquationSolver`** — same shape as step 3, one extra
   operator term (`S(k)`, single-layer) and the extra single-curve
   `CornerGrading` case DLP never needed. Reuses nearly everything from step 3
   (workspace types, Kress matrix, geometry cache) — only the kernel-assembly
   body differs.
7. **`CompositeBIMSolver`** — generalizes steps 3 and 6 to multiply-connected
   geometries. Pure composition/dispatch on top of already-working component
   solvers; no new numerical kernel. Flags that full numerical verification is
   deferred until step 11 lands a billiard with a hole.
8. **`BeynSolver`** (accelerated) — first accelerated BIM solver; wraps *any*
   already-working `SweepBIMSolver` kernel from steps 3, 6 and 7, so it must
   come after them. Contour-integral moment assembly reuses the same
   `construct_matrices`-style kernel evaluation as steps 3, 6 and 7, just at
   complex `k`.
9. **`ExpandedBIMSolver`** (accelerated) — second accelerated wrapper, needs
   the same kernel plus its first/second `k`-derivatives; smaller and more
   local than Beyn, done last among the two per the user's
   "accelerated methods can come later" ordering and because its Taylor
   expansion reuses Beyn's derivative-evaluation groundwork conceptually.
9.5. **`compute_spectrum` for accelerated BIM solvers**
   ([09.5-compute-spectrum.md](09.5-compute-spectrum.md)) — a post-step-9 gap
   discovered while auditing steps 8/9: `BeynSolver`/`ExpandedBIMSolver` only
   had a *single-window* `solve_spectrum` (one contour, or one batch of
   trial roots supplied by the caller); nothing yet covered a *whole*
   wavenumber range the way the generic `compute_spectrum(::AbsBasisSolver,
   ...)` already does for basis solvers. Ports `-develop`'s two genuinely
   different per-method merge strategies as new `compute_spectrum` methods
   dispatched on the concrete accelerated-solver type: `BeynSolver` covers
   `[k1,k2]` with disjoint Weyl-balanced contour windows and simply
   concatenates each window's already-filtered result (no fuzzy merge
   needed), while `ExpandedBIMSolver` corrects a dense adaptive grid of
   trial wavenumbers and merges the densely-overlapping results with a new
   spacing-adaptive clustering merge, `overlap_and_merge_ebim!` (ported into
   `spectra/spectralutils.jl` alongside the existing window-boundary-based
   `overlap_and_merge!`). Also introduces `ChebyshevConfig`, a single struct
   bundling every Chebyshev panel/degree/auto-tuning parameter, replacing the
   scattered `n_panels_h`/`M_h`/`n_panels_j`/`M_j` fields on `BeynSolver`/
   `ExpandedBIMSolver` with one `cheb_config::ChebyshevConfig` field each —
   purely a configuration-object placeholder until step 10 ports the actual
   Chebyshev interpolation machinery that will read it.
10. **Chebyshev-accelerated Hankel/Bessel evaluation** — pure performance
    backend behind the already-scaffolded `use_chebyshev::Bool` fields (and
    each solver's `cheb_config::ChebyshevConfig`, see step 9.5) on
    `BeynSolver`/`ExpandedBIMSolver` (and the SLP wavefunction reconstruction
    from step 5). Strictly opt-in and additive; done last among the numerical
    work because steps 3, 6–9 and 9.5 are fully correct and testable with
    `use_chebyshev=false` (direct `Bessels.jl` evaluation) first.
10.5. **Chebyshev acceleration gap analysis & porting plan**
    ([10.5-chebyshev-gap-analysis.md](10.5-chebyshev-gap-analysis.md)) — a
    post-step-10 audit (read-only, no code changes) comparing every
    Chebyshev call site in `-develop` against main's Step 10 port. Corrects a
    prior repo-memory claim about a missing `eval_h`/`eval_j` accuracy guard
    (traced the actual hot-loop call path and found no regression there),
    and instead identifies and prioritizes five real gaps: a missing ±5%
    radial-interval padding margin before building Chebyshev plans (accuracy
    risk), `BeynSolver`'s multi-k assembly doing `nq` separate O(N²) passes
    instead of one combined pass using combinator functions that already
    exist unused in `bessels.jl` (the largest performance regression), a
    missing k-independent panel-index cache across a sweep (minor), the
    still-unported SLP/CFIE wavefunction Chebyshev reconstruction (the gap
    the user explicitly flagged), and a negligible redundant-computation
    style issue in the derivative kernel entries. 
    Plan only — implementation
    deferred to a future session.    
10.9. **Geometry/symmetry audit** ([10.9-geometry-audit.md](10.9-geometry-audit.md))
    — a read-only structural audit of `BilliardGeometry.jl` vs
    `BilliardGeometry-develop`'s geometry/symmetry files (excluding the
    billiard catalogue itself), done just before Step 11 per the user's
    request so the new billiards are written against a final, not
    soon-to-change, API. Confirms most shared geometry files
    (`geometry.jl`/`arclength.jl`/`utils.jl`/`inversions.jl`/
    `poincarebirkhoff.jl`/`boundarytypes.jl`/`domains/*`/`limacon.jl`/
    `symmetry.jl`) are already functionally identical to `-develop` (only a
    docstring pass is missing). Finds and schedules four real fixes required
    before Step 11: (A) `symmetry_index_orbits` for `XAxisReflection`/
    `YAxisReflection`/`XYAxisReflection`/`NFoldRotation` never wires in the
    already-existing `symmetry_irrep_character`, silently limiting every BIM
    solver to the fully-symmetric sector for these types; (B) `full_boundary`
    has no `_apply_symmetry_to_curve` method for polar curves, blocking any
    symmetric polar billiard; (C) executes the user-requested rename
    `PolarSegment`→`FourierCoeffPolarSegment` (fixing a missing
    `promote_type` call in the process) and adds a new, more general
    function-based `PolarSegment{T,BC,F}` from `-develop`; (D) ports
    `-develop`'s missing generic `tangent_vec`/`normal_vec`/`curvature`
    utilities. Explicitly defers (with recorded trigger conditions) a
    multicomponent symmetry-orbit-folding gap affecting symmetric
    multiply-connected billiards, and a symmetry/boundary-condition
    consistency-check helper, until Step 11 produces concrete fixtures to
    validate them against — and confirms the existing symmetry framework's
    extensibility (add a new symmetry type via three dispatch methods, no
    registry/macro needed) requires no redesign.
11. **Billiards & domains from `BilliardGeometry-develop`** — new billiard/
    domain/curve-segment types, ported last as the user specified. Triangle and
    Stadium (already in main) remain the only fixtures used to verify steps
    1–10; step 11 both ports the missing geometry catalogue and back-fills
    verification for anything steps 3 and 6–9 could only smoke-test (multiply
    connected domains for step 7, symmetry-orbit folding for symmetric
    billiards, circle for analytically-known BIM eigenvalues).
12. **`AbsMultiplyConnectedDomain`/`MultiplyConnectedDomain`** — formalizes
    the domain-with-a-hole shape Step 11's `AnnularBilliard` already uses (an
    implicit `SimpleDomain`-with-opposite-curve-orientations hack) into a
    dedicated, explicitly-named `BilliardGeometry.jl` abstraction with its
    own `genus` attribute, renaming `billiards/circle_with_hole.jl` to
    `billiards/annular.jl` in the process. Pure type/dispatch-layer
    refactoring — no new numerics, no behavior change to `AnnularBilliard`'s
    produced geometry or to `CompositeBIMSolver`'s already-working
    `evaluate_points` component-grouping — but it is the natural, explicit
    hook multiply-connected billiards (and `CompositeBIMSolver`, which exists
    exactly for this case) should be built against going forward, instead of
    every future domain-with-holes re-deriving the orientation trick by hand.
12.5. **`compute_spectrum` API consolidation & unfolding/geometry split**
    ([12.5-spectral-api-consolidation.md](12.5-spectral-api-consolidation.md))
    — **Done.** A structural cleanup found while auditing
    `spectra/spectralutils.jl`/`spectra/unfolding.jl` for API consistency.
    Rescopes the generic `compute_spectrum(::AbsBasisSolver,...)` methods to
    `AcceleratedBasisSolver` (fixing a live false-dispatch bug — no
    `SweepBasisSolver` `solve_spectrum` exists), deletes a dead
    struct-field-access `N1,N2,dN` method, and adds four consistent
    `compute_spectrum` methods across the `AcceleratedBasisSolver`/
    `AcceleratedBIMSolver` branches (`k1,k2,dk`/`N1,N2` for basis solvers;
    `k1,k2`/`N1,N2` for BIM solvers, dispatching to the already-distinct
    Step 9.5 `BeynSolver`/`ExpandedBIMSolver` bodies). Moves the purely
    geometric `area`/`fundamental_area`/`symmetry_reduction_factor` out of
    `unfolding.jl` into a new `BilliardGeometry.jl/src/geometry/area.jl`
    (no new dependency — that package already depends on `QuadGK` and owns
    `AbsCurve`/`CompositeCurve`/the symmetry types involved), leaving
    `unfolding.jl` as pure Weyl-law state-counting utilities and adding new
    exported `state_at_k`/`spectral_density`/`k_range_for_states`/
    `corner_angles` accessors. Also consolidates the repeated
    sort/empty-check/construct `SpectralData` tail into one
    `_finalize_spectrum` helper, documented as safe under the existing
    "workers write disjoint slots, one thread finalizes" pattern already
    used by `BeynSolver`/`ExpandedBIMSolver`. See the step file's
    "Implementation notes" section for the one confirmed deviation (the
    `k1,k2,dk` basis-solver method now always returns `SpectralData`,
    requiring two `QuantumBilliardsTests` call-site fixes) and verification
    results.
13. **`CompositeBIMSolver` full numerical verification against the annulus**
    — closes the "known limitation" flagged since Step 7: fixes the
    reproducible `KrylovKit.svdsolve` `ArgumentError` found while testing
    Step 11's `AnnularBilliard` (isolated to `solve`/`solve_vect`, not matrix
    assembly, and reproducible even for a single-component wrap of an
    existing simply-connected billiard), then validates `CompositeBIMSolver`
    against the annulus's analytic Bessel-based spectrum — the first *real*
    (not structural-only) numerical check this solver has ever received.
14. **Plotting & final integration pass** — `QBPlotting.jl` updates for BIM
   eigenstates (`BIMEigenstate` wavefunction/boundary-function plotting,
   already wired for `DoubleLayerPotentialSolver` in step 5) and the new
   billiards, plus a whole-ecosystem smoke test across every package.
   Individual steps 1–13 each already include their own narrow plotting/test
    update; this step is the final cross-cutting sweep once everything above
    is merged.
15. **Symmetry framework refactor: stable ids, geometric group vs.
    representation** ([15-symmetry-representation-refactor.md](15-symmetry-representation-refactor.md))
    — **Done (Option A), 2026-09-13.** A structural weakness found by direct
    user audit, not a `-develop` port. `billiard.symmetries` used to be a
    positionally ordered `Vector{AbsSymmetry}` (sector-to-quadrant mapping
    depended on vector order, e.g. `poincarebirkhoff.jl`'s
    `symmetries[sym_sector-1]`), the per-curve `ReflectionSymmetry`/
    `N_sectors` boundary-condition marker was inert (never read back, no
    rotation-wall equivalent), and the same `AbsReflection`/`AbsRotation`
    structs were used both as pure geometric group generators
    (`billiard.symmetries`, `full_boundary`) and as the chosen wavefunction
    representation label (`symmetry_irrep_character`, `RealPlaneWaves`'s
    `sym_x`/`sym_y`), with no link enforcing the two stayed consistent. Now
    every symmetry struct carries only a stable `sym_id::Int` assigned by a
    new `register_symmetries`/`SymmetryRegistry` (`billiard.symmetries` is a
    `SymmetryRegistry`); `SymmetryWall(sym_id, sector_id)` replaces
    `ReflectionSymmetry`/`N_sectors` and also now tags rotation wedge cuts
    (`C3Billiard`/`StarBilliard`, previously untagged); `symmetry_irrep_character`
    was removed and `symmetry_index_orbits` gained explicit trailing
    `character`/`sector` arguments (trivial-representation default, purely
    additive so `DoubleLayerPotentialSolver`/`CombinedFieldIntegralEquationSolver`
    needed no changes); a new `QuantumBilliards.jl` `SymmetrySector`/
    `symmetry_sector(billiard, GenType=>value...)` plus a new
    `RealPlaneWaves(dim, billiard, sector)` constructor overload implement
    the Layer-2 representation choice (verified to reproduce
    `RealPlaneWaves(dim; sym_x, sym_y)` exactly). All 13 billiards ported;
    full `QuantumBilliards.jl` test suite passes unchanged. One deviation:
    BIM solvers' `symmetry::Union{Nothing,AbsSymmetry}` field was
    deliberately *not* migrated to a `SymmetrySector`-based constructor
    overload (deferred, see the step file's "Implementation notes"). The
    `BilliardGeometry.jl` `symmetryorbits.jl` test referencing the removed
    `symmetry_irrep_character` still needs a Julia Test Writer pass.
16. **Unified `SymmetrySector` representation API for basis and BIM solvers**
    ([16-bim-symmetrysector-migration.md](16-bim-symmetrysector-migration.md))
    — **Plan only, not started (v2, comprehensive rework).** Closes the
    deferral flagged at the end of Step 15: `DoubleLayerPotentialSolver`/
    `CombinedFieldIntegralEquationSolver`/`CompositeBIMSolver` still store a
    bare `symmetry::Union{Nothing,AbsSymmetry}` field with no character/
    representation choice (every BIM fold today is silently restricted to
    the trivial, fully-symmetric representation) and no link back to a
    `billiard` object (10.9's Gap F). A first draft of this step scoped
    `SymmetrySector` support to a single active `sym_id`, but a design
    critique found this rejected the two most common cases in the current
    billiard catalogue — D2 reflection pairs (`RectangleBilliard`,
    `StadiumBilliard`, etc., which register 3 sym_ids for one `Z2×Z2`
    group) and multi-generator rotation groups (`C3Billiard`/`StarBilliard`,
    which register `n-1` sym_ids for one `Zn` group) — so the plan was
    reworked end-to-end rather than patched. The reworked plan adds an
    additive `character::Tuple` field (default `()`, byte-for-byte backward
    compatible), a `_fold_boundary` dispatch adapter bridging
    `symmetry_index_orbits`'s non-uniform trailing-argument shapes, and a
    fully general `_resolve_bim_symmetry(billiard, sector::SymmetrySector)`
    helper that resolves *any* group-theoretically well-posed representation
    — including full `Z2×Z2` folding via a freshly-built `CompositeReflection`
    (reusing `BilliardGeometry.jl`'s existing, already-validated BFS
    group-closure algorithm with zero new geometry-package code) and correct
    multi-`sym_id` `Zn` sector recovery — leaving exactly one irreducible,
    mathematically-necessary validation error (a lone `XYAxisReflection`
    character alone cannot determine a unique `Z2×Z2` irrep) rather than the
    broad "not yet supported" rejections v1 had. New `(pts_scaling_factor,
    billiard, sector)` constructor overloads for DLP/CFIE mirror Step 15's
    `RealPlaneWaves(dim, billiard, sector)` pattern exactly. `BeynSolver`/
    `ExpandedBIMSolver` need no file changes, and no `BilliardGeometry.jl` or
    `QBPlotting.jl` changes are needed. Explicitly flags one honest,
    out-of-scope basis-side asymmetry left after this step (`RealPlaneWaves`/
    `CornerAdaptedFourierBessel` still cannot represent an `NFoldRotation`
    sector at all, a basis-construction feature gap predating and
    independent of this BIM-solver migration) and one non-goal (no billiard
    combines reflections and rotations into a non-abelian dihedral group, and
    no part of this codebase has 2-D-irrep machinery, so this is not
    attempted).

## Master index — where the code lives (source → destination)

| # | Solver / feature | `-develop` source (read-only reference) | Main destination |
|---|---|---|---|
| 1 | `DecompositionMethodSolver.rellich_origin` | already ported (verify only) | [solvers/sweepmethods/decompositionmethod.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/decompositionmethod.jl) |
| 1 | `ParticularSolutionsMethod` | `QuantumBilliards-develop/src/solvers/sweepmethods/basis_sweep/particular_solutions_method.jl` | [solvers/sweepmethods/particularsolutions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/particularsolutions.jl) |
| 2 | `BoundaryPoints` extension | `QuantumBilliards-develop/src/solvers/boundary_points.jl` | [solvers/boundarypoints.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/boundarypoints.jl) |
| 2 | Kress correction matrices (`kress_R!`, `kress_R_even!`, `kress_R_odd!`, corner grading maps) | same file, tail section | new `BilliardGeometry.jl/src/quadrature/kressgrading.jl` |
| 2 | `BoundaryGeomCache`, `BoundaryPanelArrays`, `component_lengths`, junction/corner detection | same file, tail section | new `BilliardGeometry.jl/src/quadrature/boundarygeomcache.jl` |
| 2 | `SymmetryOrbitMap` concrete methods (`symmetry_node_multiple`, `symmetry_index_orbits`) | `QuantumBilliards-develop/src/solvers/sweepmethods/{dlp,cfie}/*_kress.jl` (usage sites) + `BilliardGeometry-develop/src/geometry/symmetry.jl` (symmetry types) | [BilliardGeometry.jl/src/geometry/symmetryorbits.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl) (struct + stubs already scaffolded — add methods) |
| 3 | `DoubleLayerPotentialSolver` | `QuantumBilliards-develop/src/solvers/sweepmethods/dlp/dlp.jl`, `dlp_kress.jl` | [solvers/sweepmethods/dlp.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl) |
| 4 | `full_boundary(billiard)` reconstruction | `BilliardGeometry-develop/src/geometry/billiards/*.jl` (hand-written `full_boundary` fields, used as the reconstruction reference) | new `BilliardGeometry.jl/src/geometry/fullboundary.jl` |
| 4 | Extended `AbsReflection`/`NFoldRotation` irrep data (`DiagonalReflection`, `AntiDiagonalReflection`, `CompositeReflection`, `symmetry_irrep_character`) | `BilliardGeometry-develop/src/geometry/symmetry.jl` | [BilliardGeometry.jl/src/geometry/symmetry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetry.jl) |
| 4 | Extended `SymmetryOrbitMap` (`fund_to_full`/`fund_to_scale`) + orbit methods for the new reflection types | `QuantumBilliards-develop/src/states/symmetry/symmetry_orbits.jl` | [BilliardGeometry.jl/src/geometry/symmetryorbits.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl) |
| 4 | `estimate_rmin_rmax` | `QuantumBilliards-develop/src/states/symmetry/reflections.jl` | [solvers/boundarypoints.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/boundarypoints.jl) |
| 5 | `symmetrize_layer_density`, `adjoint_fredholm_matrix!`, `smallest_nullvec_krylov!` | `QuantumBilliards-develop/src/states/boundary_and_layer_density_functions.jl`, `solvers/sweepmethods/dlp/dlp.jl` | [solvers/sweepmethods/dlp.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl) |
| 5 | `ϕ_slp` Green's-function kernel + `BIMEigenstate`/`BasisEigenstate` `wavefunction`/`boundary_function` wiring | `QuantumBilliards-develop/src/states/wavefunctions.jl` | [states/wavefunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl), [states/boundaryfunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/boundaryfunctions.jl) |
| 5.5 | Generic `solve_state`/`_bim_normal_derivative` (free `∂ₙψ` from the primal Krylov solve, generalized to every `SweepBIMSolver`) | n/a (new optimization, not ported from `-develop`) | [solvers/sweepmethods/sweepmethods.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl), [states/eigenstates.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/eigenstates.jl) |
| 5.6 | General non-uniform-arclength Husimi quadrature (`_husimi_uniform_arclength_grid`-equivalent dispatch, windowed quadrature core) | `QuantumBilliards-develop/src/states/husimifunctions.jl` | [states/husimifunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/husimifunctions.jl) |
| 6 | `CombinedFieldIntegralEquationSolver` | `QuantumBilliards-develop/src/solvers/sweepmethods/cfie/cfie_kress.jl` | [solvers/sweepmethods/cfie.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/cfie.jl) |
| 7 | `CompositeBIMSolver` | `QuantumBilliards-develop` `CFIE_kress_composite_solver` (inside `cfie_kress.jl`) | [solvers/sweepmethods/compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl) |
| 8 | `BeynSolver` | `QuantumBilliards-develop/src/solvers/acceleratedmethods/beyn.jl` | [solvers/acceleratedmethods/beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl) |
| 9 | `ExpandedBIMSolver` | `QuantumBilliards-develop/src/solvers/acceleratedmethods/ebim.jl` | [solvers/acceleratedmethods/ebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl) |
| 9.5 | `compute_spectrum(::BeynSolver,...)`, `compute_spectrum(::ExpandedBIMSolver,...)`, `overlap_and_merge_ebim!`, `ChebyshevConfig` | `QuantumBilliards-develop/src/solvers/acceleratedmethods/accelerated_methods.jl` (`solve_spectrum_beyn`/`solve_spectrum_ebim` dispatch), `beyn.jl` (`solve_spectrum_beyn`, `plan_weyl_windows`), `ebim.jl` (`solve_spectrum_ebim`, `overlap_and_merge_ebim!`) | [spectra/spectralutils.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/spectralutils.jl), [solvers/acceleratedmethods/chebyshevconfig.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/chebyshevconfig.jl) |
| 10 | Chebyshev Hankel/Bessel acceleration | `QuantumBilliards-develop/src/chebyshev/*.jl` (all 6 files) | new `QuantumBilliards.jl/src/solvers/chebyshev/*.jl` |
| 10.5 | Chebyshev gap analysis & porting plan (padding margin, Beyn single-pass multi-k assembly, panel-index cache, SLP/CFIE wavefunction Chebyshev, minor kernel-entry redundancy) | `QuantumBilliards-develop/src/chebyshev/chebyshev_dlp.jl`, `chebyshev_dlp_kress.jl`, `chebyshev_cfie_kress.jl`, `chebyshev_bessels.jl`, `states/wavefunctions.jl` | [solvers/chebyshev/{optimalpanelization,dlp,cfie,bessels}.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/chebyshev), [solvers/acceleratedmethods/beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl), [states/wavefunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/wavefunctions.jl) — plan only, see [10.5-chebyshev-gap-analysis.md](10.5-chebyshev-gap-analysis.md) |
| 10.9 | Geometry/symmetry audit: antisymmetric `symmetry_index_orbits` fix, polar `full_boundary`, `PolarSegment`→`FourierCoeffPolarSegment` rename + new function-based `PolarSegment`, `tangent_vec`/`normal_vec`/`curvature` | `BilliardGeometry-develop/src/geometry/symmetry.jl`, `.../segments/polarsegment.jl`, `.../segments/curve_derivatives.jl`; `QuantumBilliards-develop/src/states/symmetry/symmetry_orbits.jl` | [BilliardGeometry.jl/src/geometry/{symmetryorbits.jl,fullboundary.jl,curvederivatives.jl,segments/polarcurves.jl,billiards/polar.jl}](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry) — plan only, see [10.9-geometry-audit.md](10.9-geometry-audit.md) |
| 11 | Billiards/domains/segments | `BilliardGeometry-develop/src/geometry/{billiards,domains,segments}/*.jl` | [BilliardGeometry.jl/src/geometry/{billiards,domains,segments}/](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry) |
| 12 | `AbsMultiplyConnectedDomain`/`MultiplyConnectedDomain`, `genus`, `boundary_components`, `AnnularBilliard` refactor | n/a (new abstraction formalizing Step 11's `AnnularBilliard`, not a `-develop` port) | [BilliardGeometry.jl/src/BilliardGeometry.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/BilliardGeometry.jl), [BilliardGeometry.jl/src/geometry/{boundarytypes.jl,domains/multiplyconnecteddomains.jl,billiards/annular.jl}](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry), [QuantumBilliards.jl/src/solvers/{boundarypoints.jl,sweepmethods/compositebim.jl}](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers) — see [12-multiply-connected-domains.md](12-multiply-connected-domains.md) |
| 12.5 | `compute_spectrum` API consolidation, `area`/`fundamental_area`/`symmetry_reduction_factor`/`corner_angles` moved to `BilliardGeometry.jl`, new `state_at_k`/`spectral_density`/`k_range_for_states`, `_finalize_spectrum` | n/a (main-only API consolidation, not a `-develop` port) | [QuantumBilliards.jl/src/spectra/{spectralutils.jl,unfolding.jl}](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra), new [BilliardGeometry.jl/src/geometry/area.jl](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry) — **done**, see [12.5-spectral-api-consolidation.md](12.5-spectral-api-consolidation.md) |
| 13 | `CompositeBIMSolver` `solve`/`solve_vect` bug fix + annulus analytic-spectrum verification | n/a (bug fix + new verification, not a `-develop` port) | [solvers/sweepmethods/compositebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl) — see [13-compositebim-annulus-verification.md](13-compositebim-annulus-verification.md) |
| 14 | Plotting | n/a (new code, follows existing `QBPlotting.jl` conventions) | [QBPlotting.jl/src/](/home/clozej/.julia/dev/QBPlotting.jl/src) |
| 15 | `SymmetryRegistry`/`sym_id`, `SymmetryWall(sym_id,sector_id)` boundary condition, `SymmetrySector` representation struct | n/a (structural refactor from user audit, not a `-develop` port) | [BilliardGeometry.jl/src/geometry/{symmetryregistry.jl,symmetry.jl,boundarytypes.jl,billiards/*.jl,fullboundary.jl,poincarebirkhoff.jl,symmetryorbits.jl}](/home/clozej/.julia/dev/BilliardGeometry.jl/src/geometry), [QuantumBilliards.jl/src/states/symmetry/{symmetrysector.jl,reflections.jl}](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry), [QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/basis/planewaves/realplanewaves.jl) — plan only, see [15-symmetry-representation-refactor.md](15-symmetry-representation-refactor.md) |
| 16 | Unified `SymmetrySector` API: DLP/CFIE/CompositeBIM `character` field, `_fold_boundary` adapter, fully general `_resolve_bim_symmetry` (CompositeReflection-based D2 folding, multi-sym_id Zn sector recovery), `SymmetrySector`-based BIM constructor overloads | n/a (closes a Step-15 deferral, not a `-develop` port) | [QuantumBilliards.jl/src/solvers/sweepmethods/{dlp.jl,cfie.jl,compositebim.jl,sweepmethods.jl}](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods), [QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/symmetry/symmetrysector.jl) — plan only, see [16-bim-symmetrysector-migration.md](16-bim-symmetrysector-migration.md) |

## Shared infrastructure already in place (do not re-implement)

Confirmed present and working in main as of this plan — every step below must
reuse these, never duplicate them:

* `abstracttypes.jl`: full docstrings for `AbsBIMSolver`/`SweepBIMSolver`/
  `AcceleratedBIMSolver` (Step 1, done).
* [solvers/sweepmethods/sweepmethods.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/sweepmethods.jl):
  `solve_wavenumber`/`k_sweep` generics for `SweepBIMSolver` — already fully
  implemented (not stubs). Every concrete `SweepBIMSolver` gets these for free
  once `evaluate_points`/`solve` work.
* [solvers/acceleratedmethods/acceleratedmethods.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/acceleratedmethods.jl):
  `evaluate_points(::AcceleratedBIMSolver, ...)` already delegates to
  `solver.kernel` — done, do not re-implement per accelerated solver.
* [solvers/sweepmethods/boundarygrading.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/boundarygrading.jl):
  `BoundaryGrading`/`SmoothPeriodicGrading`/`CornerGrading`/
  `GlobalCornerGrading` traits + `_bim_numeric_type` hook — done.
* `BilliardGeometry.jl/src/geometry/symmetryorbits.jl`: `SymmetryOrbitMap`
  struct + `fundamental_size`/`Base.length` — done; only the two generic
  function *bodies* (`symmetry_node_multiple`, `symmetry_index_orbits`) are
  missing (step 2).
* `states/eigenstates.jl`: `BIMEigenstate{K,T,S,Bi}` now also carries
  `pts`/`u`/`bnd_norm` (Step 5.5) alongside `vec`/`ten`, populated by the
  generic `solve_state(solver::SweepBIMSolver, pts, k, billiard; ...)`
  (`sweepmethods.jl`) — steps 6/7 get this, and full
  `boundary_function`/`wavefunction`/`husimi_function` support, for free the
  moment their `construct_matrices`/`solve`/`solve_vect` land; no per-solver
  state-wiring work item remains for them (see Step 5.5's "Adjustments to
  later steps" for the one small `_bim_grid_scale` follow-up Step 7 still
  needs, and the `_bim_normal_derivative` override Step 6 should check for).
* `utils/macros.jl`: `@use_threads`, `@blas_multi`, `@blas_1`,
  `@blas_multi_then_1` — identical names/signatures to what `-develop` code
  already calls; port bodies as-is, do not rewrite the macro calls.
* `solvers/decompositions.jl`: `generalized_eigen`/`generalized_eigvals`/
  `generalized_eigen_all`/`adjust_scaling_and_samplers` — reused unchanged by
  step 1; BIM steps use plain `svdvals`/Krylov nullspace routines instead (new
  in step 3, kept in `solvers/decompositions.jl` alongside the existing ones,
  not a new file).
* `solvers/matrixconstructors.jl`: `basis_matrix`/`gradient_matrices`/
  `basis_and_gradient_matrices`/`dk_matrix` — reused unchanged by step 1 only
  (BIM solvers have no `AbsBasis`, they build the Fredholm matrix directly
  from `BoundaryGeomCache`).
* `BilliardGeometry.jl/src/quadrature/samplers.jl`: identical between main and
  `-develop` already — no port needed.
* [spectra/spectralutils.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/spectra/spectralutils.jl)
  (Step 9.5, done): `compute_spectrum(::BeynSolver,...)`,
  `compute_spectrum(::ExpandedBIMSolver,...)`, `overlap_and_merge_ebim!` and
  the extended `SpectralData` (optional `ten2` field) — step 10 must call
  into these unchanged at the merge-strategy level, only adding a Chebyshev
  branch inside the per-window/per-segment `solve`/`construct_matrices` calls
  they already make.
* [solvers/acceleratedmethods/chebyshevconfig.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/chebyshevconfig.jl)
  (Step 9.5, done): `ChebyshevConfig{T}` struct, already the `cheb_config`
  field of both `BeynSolver` and `ExpandedBIMSolver` — step 10 populates the
  actual Chebyshev machinery behind it, it does not need to design a new
  configuration object.

## Non-negotiable numerical-fidelity rule (applies to every step)

Per the user's explicit direction: the `-develop` kernel-assembly bodies
(pairwise geometry caches, Kress correction matrices, Chebyshev evaluation,
contour quadrature) are performance-critical and already optimized. For these
low-level numeric kernels, **port the arithmetic verbatim** (same loop
structure, same in-place buffers, same BLAS calls) — only rename
symbols/fields to match main's already-committed API (§2 of the Step 1 plan)
and re-home shared pieces into `BilliardGeometry.jl` per the placement
decisions above. Do not "clean up" a hot loop's algebra while porting it.
Simplification/deduplication is welcome at the *API/dispatch* layer (fewer
struct variants via the `BoundaryGrading` trait, shared workspace types across
DLP/CFIE) but never inside a single-threaded numeric inner loop.

## Testing & verification policy (applies to every step)

Every step file ends with a `Tests & user verification` section specifying
concrete billiards/wavenumbers/expected tensions to check by hand (or with a
short scratch script) before calling the step done. No step's implementing
agent writes `Test.jl` test files itself — the step's final instruction is
always to tell the user to invoke the **Julia Test Writer** subagent once the
implementation and manual verification look right. `test/solvertests.jl`
already has the pattern to follow (see e.g. the Veech-triangle/Vergini-Saraceno
testsets) for the numerical-regression style expected of the eventual tests.

## Known limitation flagged up front (Step 7) — now closed by Steps 12–13

Because billiards were ported last (step 11) and Triangle/Stadium are both
simply connected, `CompositeBIMSolver` (step 7) could not be fully
numerically validated against a known multiply-connected spectrum until step
11 landed `AnnularBilliard`. Step 7 only did the best available structural
verification (dimensions, symmetry, `evaluate_points` producing the right
number of components) using two disjoint copies of Triangle/Stadium. Step 11
did land `AnnularBilliard` and confirmed `evaluate_points`/`construct_matrices`
work correctly for it, but also found a real, reproducible bug
(`KrylovKit.svdsolve` `ArgumentError` in `solve`/`solve_vect`, not specific to
multiply-connected billiards) that blocked full verification. Step 12 first
formalizes `AnnularBilliard`'s domain shape into a dedicated
`AbsMultiplyConnectedDomain`/`MultiplyConnectedDomain` abstraction (pure
refactor, no numerics change); Step 13 then fixes that bug and performs the
actual deferred numerical verification against the annulus's analytic
spectrum — closing this limitation out.
