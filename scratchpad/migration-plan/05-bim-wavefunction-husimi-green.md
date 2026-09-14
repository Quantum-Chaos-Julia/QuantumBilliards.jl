# Step 5 — BIM wavefunction, boundary-function & Husimi reconstruction via the Green's-function integral (`DoubleLayerPotentialSolver` first)

## Goal

Replace `BIMEigenstate`'s "not yet implemented" `wavefunction`/
`boundary_function`/`husimi_function` (flagged as out of scope in Step 3) with
a real implementation, **scoped to `DoubleLayerPotentialSolver` only** — CFIE/
Composite/accelerated solvers get their own wavefunction wiring once they
exist (Steps 6–9), following the same pattern this step establishes. Also
make the single-layer-potential (SLP) Green's-function reconstruction the
**shared, default** wavefunction backend for `BasisEigenstate` too, since a
basis-expanded eigenstate can already produce the same `(u, pts)` boundary
data via the existing `boundary_function(state::BasisEigenstate)` — this
collapses two previously-separate wavefunction code paths (`compute_psi`'s
direct `basis_matrix` evaluation vs. a hypothetical BIM-only Green's-function
path) into one, per the "add a method instead of a special case" dispatch
guidance, and is more efficient than dense basis-matrix evaluation on a fine
Cartesian grid for large basis dimensions.

## Preconditions

Steps 3 and 4 complete and verified (`DoubleLayerPotentialSolver` working
end-to-end, including the symmetric case fixed in Step 4; `full_boundary`,
extended `SymmetryOrbitMap`, and `estimate_rmin_rmax` all available).

## Source index

* `QuantumBilliards-develop/src/states/wavefunctions.jl` — full file (already
  read in full during planning). Relevant sections, in order: `inside_mask`
  (fundamental-domain vs. full-physical-boundary interior test),
  `chebyshev_params_slp`/`_slp_chebyshev_plans` (Chebyshev `H₀⁽¹⁾`/`Y₀` tuning
  — **defer to Step 10**, this step ships `use_chebyshev=false` only, see
  below), `ϕ_slp` (the actual SLP Green's-function kernel — **the core
  formula this step ports**), `wavefunctions(solver::Union{BoundaryIntegralMethod,DLP_kress,DLP_kress_global_corners}, ...)`
  (batch grid reconstruction driver for DLP — port the structure, not the
  `BoundaryIntegralMethod`/`DLP_kress_global_corners` dispatch since only
  `DoubleLayerPotentialSolver` exists in main), `compute_psi`/`wavefunction`
  for basis states (existing generic grid-construction logic — reuse the
  grid-sizing/masking parts, replace only the per-point evaluation kernel).
* `QuantumBilliards-develop/src/states/boundary_and_layer_density_functions.jl`
  — full file (already read in full). Relevant sections: `regularize!`
  (already ported in main, confirm), `_rellich` (**already ported** in main's
  [states/boundaryfunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/boundaryfunctions.jl)
  — do not re-port, just reuse), `symmetrize_layer_density` (expand a
  symmetry-reduced boundary density back onto the full physical boundary
  using `orbits.full_to_scale`/`full_to_fund` — **new for BIM, needed here**),
  `boundary_function(solver::DLP, pts, billiard, k)` (constructs `u=∂ₙψ` from
  the **adjoint/weighted-transpose** Fredholm nullspace, Rellich-normalizes
  it, and symmetry-expands it — **the key function this step ports**),
  `adjoint_fredholm_matrix!`/`smallest_nullvec_krylov!` (`dlp/dlp.jl`, search
  there — these are *not* the same as `construct_matrices`'s primal `A(k)`;
  read carefully to understand exactly how the adjoint/weighted-transpose
  matrix differs from `A(k)` before assuming `solve_vect`'s eigenvector can be
  reused directly).
* `QuantumBilliards-develop/src/solvers/sweepmethods/dlp/dlp.jl` — read the
  `adjoint_fredholm_matrix!`/`smallest_nullvec_krylov!` definitions (both
  referenced above but defined in this file, not `boundary_and_layer_density_functions.jl`).
* `QuantumBilliards.jl/src/states/husimifunctions.jl` (main, already fully
  implemented) — `_husimi_uniform_arclength`/`husimi_function(k,u,s,L;...)`
  operate generically on `(k, s, ds, u, L)`; **no porting needed**, this step
  only needs to wire `husimi_function(state::BIMEigenstate)` to call the
  already-existing generic function with the `(u, pts)` this step's
  `boundary_function(state::BIMEigenstate)` produces. Same for
  `momentum_function(u, s, ds)` in
  [states/boundaryfunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/boundaryfunctions.jl).

## Files touched in main

### 5.1 — `symmetrize_layer_density` (new, `QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl` or a same-folder `dlpdensity.jl`)

Port `symmetrize_layer_density(solver::DoubleLayerPotentialSolver, layer_density, pts, billiard)`
verbatim (both the single-state and batched/vectorized overloads) using the
Step 4 `SymmetryOrbitMap.full_to_fund`/`full_to_scale` fields: if
`length(layer_density) == length(pts)` return unchanged (already full); else
expand via `full_data[q] = orbits.full_to_scale[q] * layer_density[orbits.full_to_fund[q]]`.
Requires `pts` to already be the discretization of `full_boundary(billiard)`
(Step 4's fix) — document this precondition directly in the docstring.

### 5.2 — Adjoint Fredholm nullspace (new, same file)

Port `adjoint_fredholm_matrix!` (both the `::Nothing` and `::SymmetryOrbitMap`
dispatches — re-derive the exact weighted-transpose relationship to
`construct_matrices`'s primal `A(k)` from `-develop`'s definition rather than
guessing; it is **not** simply `A(k)'`, since a naive transpose does not
account for the physical quadrature-weight asymmetry of the Nyström
discretization — read `-develop`'s comments/derivation around this function
carefully) and `smallest_nullvec_krylov!` (thin `KrylovKit`-based nullspace
extraction — `KrylovKit` is already a `QuantumBilliards.jl` dependency via
`solve`/`solve_vect`'s `KrylovKit.svdsolve` calls in `dlp.jl`, so this is not
a new dependency). Keep both private (not exported) — internal
`boundary_function` implementation detail, exactly as in `-develop`.

### 5.3 — `boundary_function(state::BIMEigenstate; b=5.0)` (`QuantumBilliards.jl/src/states/boundaryfunctions.jl`)

Add the `BIMEigenstate` overload alongside the existing
`boundary_function(state::S; b, multithreaded) where {S<:AbsState}` generic
(check whether that generic already special-cases on `state.solver`/`state.basis`
internally, or whether a dedicated method dispatching on `state::BIMEigenstate`
is cleaner — prefer a dedicated method per the "add a method, not a branch"
guidance, since `BIMEigenstate` has no `.basis` field the existing generic's
body likely assumes). Implementation: re-discretize the boundary at `state.k`
via `evaluate_points(state.solver, state.billiard, state.k)` (same `pts` the
solve used, or freshly recomputed — confirm which `-develop` does; likely
freshly recomputed since `BIMEigenstate` doesn't store `pts` currently, check
its field list in `states/eigenstates.jl` before assuming), build the adjoint
matrix via §5.2, extract the nullspace `u`, expand to the full boundary via
§5.1, Rellich-normalize via the **already-ported** `_rellich` (do not
re-implement), and return `(u, pts)` in the same `(u::Vector, pts::BoundaryPoints)`
order `boundary_function(state::BasisEigenstate)` already returns, so both
state types are interchangeable to every downstream consumer
(`momentum_function`, `husimi_function`, the new §5.4 wavefunction integral).
Note for the caller: for `DoubleLayerPotentialSolver`, `state.vec` (from
`solve_vect`, the **primal** nullspace) is *not* the same vector as this `u`
— document this distinction prominently in the docstring, since it is the
single easiest mistake to make when reusing `dlp.jl`'s existing
`solve_vect` output for wavefunction reconstruction instead of recomputing
via the adjoint operator.

### 5.4 — SLP Green's-function kernel `ϕ_slp` (new, `QuantumBilliards.jl/src/states/wavefunctions.jl`)

Port `ϕ_slp(x,y,k,pts,u; float32_bessel, use_chebyshev=false, cheb=nothing)`
verbatim — the single-layer potential representation
`ψ(x,y) = (1/4)Σⱼ Y₀(k|x-q_j|)u_j ds_j` (`Bessels.bessely0`, already a
dependency). **Ship `use_chebyshev=false` only in this step** (always
evaluate `Y₀` directly, either `Float32` or full precision per
`float32_bessel`) — the Chebyshev-interpolated `H₀⁽¹⁾`/`Y₀` tuning
(`chebyshev_params_slp`) is explicitly deferred to Step 10, matching that
step's existing scope and the `BeynSolver`/`ExpandedBIMSolver` precedent of
shipping a working `use_chebyshev=false` path first. If
`use_chebyshev=true` is requested before Step 10 lands, `error("Chebyshev
SLP wavefunction reconstruction not yet implemented; see migration plan step
10")` rather than silently ignoring the flag.

This same kernel is the one both `BIMEigenstate` (§5.5) and, by design,
`BasisEigenstate` (§5.6) evaluate against — it depends only on `(k, pts, u)`,
never on `state.solver`/`state.basis`, which is exactly what makes it
shareable.

### 5.5 — `wavefunction(state::BIMEigenstate; ...)` (`QuantumBilliards.jl/src/states/wavefunctions.jl`)

Add a `wavefunction(state::BIMEigenstate; b=:auto, inside_only=true, use_float32_bessel=true, MIN_CHUNK=4096)`
method: get `(u, pts) = boundary_function(state)` (§5.3), build the Cartesian
grid from `full_boundary(state.billiard)` (Step 4 — **not**
`get_boundary_curves`, since that would only cover the fundamental domain)
via `boundary_limits`, exactly mirroring `-develop`'s grid-sizing logic in its
`wavefunctions(solver::Union{...}, ...)` batch driver but for a single state;
mask interior points via `inside_mask`-equivalent logic (port `inside_mask`
if main doesn't already have an equivalent `is_inside`-based full-vs-fundamental
mask — check `BilliardGeometry.is_inside` first); evaluate `ϕ_slp` at every
masked grid point, threaded over the flattened masked-point index (mirror
`-develop`'s `Threads.@threads :static` chunking pattern, or use
`@use_threads` if the loop shape fits that macro's convention — confirm
which before choosing, since `-develop`'s manual chunking exists specifically
to keep `MIN_CHUNK` points per thread and avoid oversubscription for a
possibly-small masked-point count). Return `(Psi2d, x_grid, y_grid)`, matching
the existing `wavefunction(state::BasisEigenstate)` return signature exactly.

### 5.6 — Make Green's-function reconstruction the default for `BasisEigenstate` too

`boundary_function(state::BasisEigenstate)` already exists and returns
`(u, pts)` (Rellich-normalized `∂ₙψ` on the full physical boundary, already
symmetry-expanded). Add a new keyword to the existing
`wavefunction(state::BasisEigenstate; ...)` — e.g. `method::Symbol=:green` —
defaulting to the new SLP-integral reconstruction (§5.4's `ϕ_slp`, called
with this state's own `(u,pts)`) instead of the current `compute_psi`/
`basis_matrix`-based direct evaluation, keeping the old path available as
`method=:basis` for comparison/regression purposes (do not delete
`compute_psi`, existing tests/plots may depend on it and it remains the
right choice for very coarse grids or basis-debugging). Document in the
docstring that `:green` is the new default because it is asymptotically
cheaper than a dense `basis.dim × n_grid_points` matrix construction for
typical basis dimensions, matching the user's explicit direction that the
Green's-function path "should be used by default."  Confirm this default
switch does not change any existing plotted/tested wavefunction beyond
floating-point-level reconstruction differences (both paths reconstruct the
*same* physical eigenfunction; ask the user to visually compare a
`VerginiSaracenoSolver`-on-Stadium wavefunction plot under both `method`
values before calling this done).

### 5.7 — `husimi_function`/`momentum_function` for `BIMEigenstate`

Both already dispatch on `state::S where {S<:AbsState}` and internally call
`boundary_function(state; b)` first (confirm by reading
[states/husimifunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/husimifunctions.jl)/
[states/boundaryfunctions.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/states/boundaryfunctions.jl)'s
existing generic bodies) — since §5.3 now makes `boundary_function` work for
`BIMEigenstate`, these likely need **no new code**, only verification that
the existing generic `S<:AbsState` methods don't have a hidden
`BasisEigenstate`-only assumption (e.g. accessing `state.basis` directly
instead of going through `boundary_function`). If such an assumption exists,
add the minimal `BIMEigenstate`-specific override rather than restructuring
the generic method.

## Performance & fidelity notes

* `ϕ_slp`'s per-point kernel is the hot loop of this step — port verbatim
  (including the `r2==0` exact-coincidence guard and the `Float32` fast path),
  `@inbounds @fastmath`, no allocations inside the loop.
* The outer loop over masked grid points is the correct (and only) place to
  parallelize (`multithreaded`/thread-chunking as in `-develop`); the inner
  `ϕ_slp` call is itself allocation-free and single-threaded — do not nest
  threading inside it.
* `adjoint_fredholm_matrix!`/`smallest_nullvec_krylov!` build one dense
  `N×N` (or fundamental-size, if symmetric) complex matrix per state, same
  cost class as `construct_matrices` — reuse `@blas_1`/`@blas_multi_then_1`
  wrapping exactly as `-develop` does around the dense linear algebra.

## Tests & user verification

1. `get_errors` on every new/edited file.
2. Construct a `DoubleLayerPotentialSolver`, locate a `k0` on
   `StadiumBilliard(0.5)` (reuse Step 3/4's fixture), build
   `state = compute_eigenstate(solver, billiard, k0)`, call
   `boundary_function(state)` and confirm `u` is finite, real (DLP densities
   are real for a Dirichlet eigenfunction — confirm this assumption against
   `-develop`), and Rellich-normalized (`∫(x·n)|u|²ds/(2k²) ≈ 1`, check
   directly).
3. Call `wavefunction(state)` and visually confirm (ask the user to plot,
   e.g. via `QBPlotting.jl`'s existing wavefunction heatmap helper) a
   recognizable nodal pattern with the right number of nodal lines for `k0`'s
   position in the known Stadium spectrum (cross-check against the existing
   `VerginiSaracenoSolver` wavefunction plot at the same/closest `k`).
4. Repeat 2–3 with `symmetry=BilliardGeometry.YAxisReflection()` on the
   symmetric fixture from Step 4, confirming the reconstructed wavefunction
   still looks correct on the **full** billiard (not just the fundamental
   domain half) and has the expected parity (mirror symmetric/antisymmetric
   about the reflection axis).
5. For `BasisEigenstate`: compute a `VerginiSaracenoSolver` eigenstate, call
   `wavefunction(state; method=:green)` and `wavefunction(state; method=:basis)`,
   and confirm the two matrices agree to within a small tolerance on the
   shared masked-interior grid (they reconstruct the same physical function
   via two different numerical routes).
6. `QBPlotting.jl` update: confirm (or add, if missing) a wavefunction/
   boundary-function plotting helper accepts a `BIMEigenstate` exactly as it
   already accepts a `BasisEigenstate` (both now share the same `(u,pts)`/
   `(Psi2d,x_grid,y_grid)` return shapes) — this should require **no new
   plotting code**, only confirming existing helpers don't hard-code
   `state::BasisEigenstate` in their type signature; relax to `state::AbsState`
   if they do.
7. Tell the user to invoke the **Julia Test Writer** subagent for
   `boundary_function`/`wavefunction`/`husimi_function` regression coverage
   on `BIMEigenstate` (both symmetric and non-symmetric cases), and for the
   `method=:green` vs `method=:basis` cross-check on `BasisEigenstate`, once
   the manual checks above pass.

## Definition of done

* `boundary_function`/`wavefunction`/`husimi_function`/`momentum_function` all
  work for `BIMEigenstate` constructed from a `DoubleLayerPotentialSolver`,
  with and without symmetry, cross-validated against existing basis-solver
  results at the same wavenumber.
* `wavefunction(state::BasisEigenstate)` defaults to the Green's-function
  (`method=:green`) reconstruction, with the previous direct basis-matrix
  path preserved as an explicit `method=:basis` opt-out.
* `BIMEigenstate`'s docstring no longer says wavefunction/Husimi support is
  unimplemented; the distinction between `solve_vect`'s primal density and
  `boundary_function`'s adjoint-derived `u` is documented.
* `QBPlotting.jl` plotting helpers confirmed (or widened) to accept
  `BIMEigenstate` alongside `BasisEigenstate`.
* Julia Test Writer invoked for regression coverage.
