# Step 3 — `DoubleLayerPotentialSolver`

## Goal

Replace every stub body in
[solvers/sweepmethods/dlp.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl)
with a real implementation, starting with `SmoothPeriodicGrading` (single
smooth closed curve or smooth composite, no corners) and then adding
`GlobalCornerGrading`. This is the reference implementation for every later
`SweepBIMSolver` — Step 6 (`CombinedFieldIntegralEquationSolver`) copies this
step's structure almost exactly, only swapping the kernel term.

## Preconditions

Step 2 complete: extended `BoundaryPoints`, `kress_R!`, `boundary_geom_cache`,
`symmetry_index_orbits` all load and are independently verified.

## Source index

* `QuantumBilliards-develop/src/solvers/sweepmethods/dlp/dlp_kress.jl` — full
  file, already partially read (struct definitions, constructors,
  `build_Rmat_dlp_kress`, `_is_dlp_kress_graded`/`_dlp_kress_use_reduced`
  dispatch helpers). Continue reading past where the earlier read stopped
  (`_composite_arclength` and onward) for: the actual double-layer kernel
  evaluation, `DLPKressWorkspace`/`DLPKressReducedWorkspace` construction
  (`build_dlp_kress_workspace` or similarly named), `construct_matrices`(the
  real Nyström assembly combining `boundary_geom_cache` pairwise data with
  Hankel-function evaluation and the `kress_R!`/`Rmat` split), and
  `solve`/`solve_vect` (nullspace/smallest-singular-value extraction, with and
  without symmetry reduction).
* `QuantumBilliards-develop/src/solvers/sweepmethods/dlp/dlp.jl` — the plain
  uncorrected `BoundaryIntegralMethod`. **Do not port this file's solver at
  all** — already confirmed dropped in
  [../BIM-solver-migration-plan.md](../BIM-solver-migration-plan.md) §4.3/§8.2,
  its behavior is dominated by `SmoothPeriodicGrading`. It's still worth a
  quick read for the *unweighted* Nyström double-layer kernel formula itself
  (Helmholtz fundamental solution derivative), since `DLP_kress`'s smooth part
  uses the same kernel, just splits off the log-singular part via `kress_R!`.

## Files touched in main

Only [solvers/sweepmethods/dlp.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl)
for the solver itself. If a new internal workspace type is needed
(`DLPKressWorkspace`/`DLPKressReducedWorkspace` equivalents), add it to this
same file (or a same-folder `dlpworkspace.jl` if it grows large) — keep it
private (not exported), it's an internal `construct_matrices` implementation
detail exactly like `-develop`'s.

## Implementation steps

1. **`evaluate_points(solver::DoubleLayerPotentialSolver, billiard, k)`**:
   dispatch on `solver.grading`.
   * `SmoothPeriodicGrading`: sample the boundary with uniform periodic
     `LinearNodes`/trapezoid-style nodes (matching `-develop`'s
     `DLP_kress`, which always uses `BilliardGeometry.LinearNodes()` — reuse
     that sampler type, don't introduce a new one) at
     `N = max(solver.min_pts, round(Int, k*L*solver.pts_scaling_factor[i]/2π))`
     points per fundamental curve, compute `xy`, `tangent`, `tangent_2`
     (`BilliardGeometry.tangent`/`tangent_2` on each curve, already used
     elsewhere), `ts`/`tphys`/`ws`/`ws_der` per the periodic parametrization
     convention `-develop` uses, and build the result via the Step 2
     parametrized `BoundaryPoints` constructor. If `solver.symmetry !==
     nothing`, only sample the fundamental-domain arc and then use
     `symmetry_index_orbits` to fold it — **re-derive** this against how
     `-develop`'s `_dlp_kress_use_reduced`/`DLPKressReducedWorkspace` actually
     handles it (it may sample the *full* periodic boundary first and use
     `SymmetryOrbitMap` only during matrix assembly, not during point
     sampling — confirm exactly which before implementing, since getting this
     backwards silently produces a wrong-sized matrix later).
   * `GlobalCornerGrading`: additionally detect true corners via
     `_component_corner_locations`/`_is_true_corner` (Step 2), build the
     Kress grading map `t = w(σ)` at order `solver.grading.kressq`, and use
     `_eval_composite_geom_global_t` to evaluate the composite boundary at the
     graded nodes. Falls back to the `SmoothPeriodicGrading` path automatically
     when no corners are detected (mirrors `-develop`'s documented fallback
     behavior in the `GlobalCornerGrading` docstring already in
     `boundarygrading.jl`).
2. **`boundary_matrix_size(solver, pts)`**: `symmetry === nothing ?
   length(pts) : fundamental_size(orbits)` — needs the `SymmetryOrbitMap` to
   be retrievable from `pts` or recomputed; decide (and document in the
   struct/method docstring) whether the orbit map is recomputed on every call
   or cached — prefer recomputing from `pts.xy`/`pts.normal` inside
   `construct_matrices` and passing to `solve`/`solve_vect` only if truly
   needed twice, to avoid a hidden mutable cache field on an otherwise
   immutable-feeling call chain; if `-develop`'s `DLPKressReducedWorkspace`
   caches this because it's expensive, mirror that by having
   `construct_matrices` return (or internally memoize via a closure/workspace
   object) the orbit map once per `pts`, not recomputed per wavenumber `k`
   inside a `k_sweep`.
3. **`construct_matrices(solver, pts, k; multithreaded)`**: build
   `G = boundary_geom_cache(pts, is_globalcornergrading)` (Step 2), evaluate
   the Helmholtz double-layer kernel `D(k)` from `G.R`, `G.inner`, `G.kappa`,
   Hankel-function values at `k*G.R` (`Bessels.hankelh1`, already a
   `QuantumBilliards.jl` dependency), split the diagonal/near-diagonal
   log-singularity via `kress_R!`'s `Rmat` (or the graded corner map's `Rmat`
   equivalent for `GlobalCornerGrading`), and assemble `A = I - D(k)` as a
   dense `Matrix{Complex{T}}`. Reuse the exact same weighting/splitting
   algebra `-develop`'s `construct_matrices` uses — this is the single most
   performance/numerically-sensitive piece in this whole migration, port
   verbatim per the fidelity rule, only adapting field names
   (`pts.tangent`/`pts.ts`/`G.R`/etc.) to the Step 2 API.
   * `multithreaded::Bool`: the outer loop over matrix columns (or rows) is
     the correct place for `@use_threads`, exactly as the mode instructions
     specify — confirm `-develop`'s assembly loop structure supports this
     (single outer loop, inner loop allocation-free) before parallelizing; if
     `-develop`'s version is single-threaded because it relies on vectorized
     `Bessels`/broadcasting instead of explicit loops, keep it vectorized and
     only add `@use_threads` if there's a genuine per-column independent loop
     to parallelize (don't force a loop structure that doesn't exist just to
     add threading).
   * If `solver.symmetry !== nothing`: fold the assembled full matrix onto the
     fundamental domain using the `SymmetryOrbitMap`'s `orbit_of`/`phase`
     (sum contributions from every full-boundary image of each fundamental
     column, weighted by `phase`), matching `-develop`'s
     `DLPKressReducedWorkspace` logic.
4. **`solve(solver, pts, k; multithreaded, use_krylov=true)`**: tension from
   the smallest singular value of `A(k)` — port `-develop`'s exact choice of
   method (dense `svdvals` for small `N`, a Krylov nullspace-residual iterative
   method, e.g. via `KrylovKit.jl`/`ArnoldiMethod.jl`, for large `N` when
   `use_krylov=true`). **Check whether `-develop` already depends on a Krylov
   package** (search its `Project.toml`) before adding one to
   `QuantumBilliards.jl` — if so it's a straightforward matching dependency
   add, not a new unjustified one; if `-develop` only ever uses dense
   `svdvals`, keep `use_krylov` as a scaffolded-but-currently-`false`-only
   option and flag to the user that Krylov support needs its own follow-up
   once a concrete algorithm is identified, rather than guessing a Krylov
   recipe that was never in `-develop`.
5. **`solve_vect(solver, pts, k; multithreaded)`**: same as `solve` but also
   return the corresponding right/left singular vector as the boundary
   density `x::Vector{Complex{T}}` — if `symmetry !== nothing`, `x` is the
   fundamental-domain density; document that a caller wanting the density at
   every full-boundary point must expand it back via `orbits.orbit_of`/
   `phase` (needed later for wavefunction evaluation from the boundary
   density, out of scope for this step).
6. Add the `_bim_numeric_type(::DoubleLayerPotentialSolver{T}) = T` method (a
   one-liner, mirrors the pattern already used for `_bim_numeric_type` in
   `boundarygrading.jl` — check it's not already present from the Step 1
   scaffold before re-adding).
7. **`BIMEigenstate` wiring**: add a constructor for
   `BIMEigenstate{K,T,S,Bi}` (struct already declared in
   `states/eigenstates.jl`, unused so far) and a
   `compute_eigenstate(solver::SweepBIMSolver, billiard, k; multithreaded)`
   generic (placed in `states/eigenstates.jl` alongside the existing
   `SweepBasisSolver`/`AcceleratedBasisSolver` overloads, dispatching on the
   `SweepBIMSolver` abstract branch so it's shared with Step 6/7 for free) —
   calls `evaluate_points`+`solve_vect` and stores the boundary density `vec`
   plus the `pts`/`billiard` needed later for wavefunction evaluation via the
   layer-potential representation formula. Full `wavefunction`/`husimi_function`
   support for `BIMEigenstate` is **out of scope** for this step (flagged
   already in the Step 1 plan §3 as a later phase) — only the state
   *representation* needs to exist so `solve_wavenumber`'s return value can be
   turned into something inspectable; document this limitation directly in
   `BIMEigenstate`'s docstring (`wavefunction`/`husimi_function` not yet
   implemented for this state type).

## Performance & fidelity notes

* `construct_matrices` is the hottest path in this whole migration segment —
  copy `-develop`'s pairwise-kernel algebra exactly (no rewriting the Hankel
  evaluation, no changing `BoundaryGeomCache`'s precomputed arrays).
* Preallocate the dense `Complex{T}` matrix once per call; do not build it via
  concatenation/`hcat`.
* If `-develop` uses `@blas_multi`-style BLAS-thread bounding around any
  dense linear algebra inside `solve`, port that wrapping unchanged.

## Tests & user verification

1. `get_errors` on `dlp.jl` and `QuantumBilliards.jl`.
2. Analytic sanity check without a new billiard: a `DoubleLayerPotentialSolver`
   on the existing `StadiumBilliard(0.5)` (a single smooth closed curve made
   of two straight segments and two half-circles — mostly smooth, has two
   true corners where straight meets curved... actually the Stadium's
   boundary is `C^1` there, confirm via `_is_true_corner`/`_junction_angle`
   whether the Stadium billiard registers any true corner at all before
   picking `SmoothPeriodicGrading` vs `GlobalCornerGrading` for this test —
   if the junction is smooth, `SmoothPeriodicGrading` is the correct choice
   and is exactly what validates the "simplest case first" ordering).
3. Run `k_sweep`/`solve_wavenumber` with `DoubleLayerPotentialSolver` over a
   `k`-window already known from `test/solvertests.jl`'s
   `VerginiSaracenoSolver`-on-Stadium testsets, and confirm the located `k0`
   matches (within a reasonable `atol`, since these are two independent
   discretizations of the same physical eigenvalue problem — DLP tension
   scale is unrelated to VS tension scale, only the located `k0` is
   comparable).
4. If the Stadium does register a true corner, additionally test
   `GlobalCornerGrading` and confirm it also finds the same `k0`, at fewer or
   comparable boundary points than an ungraded periodic discretization would
   need for the same accuracy (this is the entire point of Kress grading —
   verify it isn't a no-op).
5. Confirm `symmetry = BilliardGeometry.YAxisReflection()`-restricted solves
   (if the Stadium/Triangle fixture used has that symmetry) find the subset of
   eigenvalues with matching parity, consistent with what
   `RealPlaneWaves(...; sym_y=-1)`-restricted `VerginiSaracenoSolver` finds.
6. `QBPlotting.jl` update: `plot_sweep!` already dispatches on
   `QuantumBilliards.SweepBasisSolver`/`AcceleratedBasisSolver`, **not**
   `SweepBIMSolver` — add a `plot_sweep!(ax, k_min, k_max, dk,
   solver::QuantumBilliards.SweepBIMSolver, billiard; ...)` method to
   [benchmarkplots.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/benchmarkplots.jl)
   mirroring the existing `SweepBasisSolver` method but without a `basis`
   argument (calls `k_sweep(solver, billiard, ks)` directly). Ask the user to
   visually confirm the tension-vs-`k` plot for `DoubleLayerPotentialSolver`
   on Stadium shows dips at the same `k` values as the existing
   `VerginiSaracenoSolver` plot.
7. Tell the user to invoke the **Julia Test Writer** subagent for
   `DoubleLayerPotentialSolver` regression tests (both gradings, with and
   without symmetry), following the `test/solvertests.jl` style, once the
   manual checks above pass.

## Definition of done

* `DoubleLayerPotentialSolver` fully implemented for `SmoothPeriodicGrading`
  and `GlobalCornerGrading`, with and without `symmetry`.
* Located eigenvalues cross-validated against existing
  `VerginiSaracenoSolver`/`DecompositionMethodSolver` results on
  Stadium/Triangle.
* `BIMEigenstate` constructed and `compute_eigenstate` wired generically for
  `SweepBIMSolver` (shared by Steps 6–7 for free). Full `wavefunction`/
  `boundary_function`/`husimi_function` support for `BIMEigenstate` lands in
  Step 5 (after the Step 4 symmetry/full-boundary prerequisites), not here.
* `QBPlotting.jl` has a working `plot_sweep!` overload for `SweepBIMSolver`,
  visually verified by the user.
* Julia Test Writer invoked for regression coverage.
