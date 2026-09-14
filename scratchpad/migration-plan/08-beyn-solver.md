# Step 8 — `BeynSolver`

## Goal

Implement Beyn's contour-integral nonlinear-eigenvalue extraction in
[solvers/acceleratedmethods/beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl),
wrapping any already-working `SweepBIMSolver` kernel from Steps 3, 6 and 7.
First accelerated solver in this migration — do not start before Steps 3, 6
and 7 are verified, since `BeynSolver` calls straight into `construct_matrices`
of whatever `kernel` it wraps, just at complex `k`.

## Preconditions

Steps 3, 6 and 7 complete and verified (at least `DoubleLayerPotentialSolver`
working end-to-end is required; CFIE/Composite kernels should also work
before considering this step fully generic, but initial development/testing
of `BeynSolver` itself can proceed against `DoubleLayerPotentialSolver` alone).

## Source index

`QuantumBilliards-develop/src/solvers/acceleratedmethods/beyn.jl` — already
read in full during planning. Key pieces to port:

* `weyl_window_width`, `plan_weyl_windows`, `beyn_disks_from_windows` — free
  functions operating on `billiard`+`m`, not solver state (per the Step 1 plan
  §4.6, kept as free functions, not struct methods).
* `beyn_buffer_matrices(::Type{T}, N, r, rng)` — preallocates the random
  probing matrix `V` and workspaces `X`, `A0`, `A1`.
* `construct_B_matrix(solver, pts, N, k0, R; kwargs...)` — the actual contour
  quadrature: trapezoidal nodes `zj`/weights `wj` around the circle
  `(k0, R)`, accumulating `A0 += wj*T(zj)⁻¹V`, `A1 += wj*zj*T(zj)⁻¹V` via
  repeated `construct_matrices(kernel, pts, zj)` + linear solve, then the
  rank-revealing SVD of `A0`, retained-rank projection, and the reduced
  dense matrix `B = Uk'*A1*Wk*Σk⁻¹`. Read past where the earlier preview
  stopped (the docstring only, not the body) for the actual solve loop,
  probing-dimension auto-increase logic, and the eigenvalue/residual
  filtering step (`|k−k0|≤R` containment + residual tolerance).
* `#TODO` comments at the top of the file (Backer's real-Green's-function
  idea, NLFEAST) are **not** part of this migration — informational only,
  do not implement speculative future algorithms flagged as `#TODO` in the
  reference.

Note `-develop`'s `BeynSolver` is a `Union` type alias over every kernel
struct — per the Step 1 plan §2.3/§4.6, main's `BeynSolver{T,K}` is a genuine
wrapper struct (already scaffolded); every function below takes
`solver::BeynSolver` and reaches the kernel via `solver.kernel`, not via
`Union` dispatch.

## Files touched in main

[solvers/acceleratedmethods/beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl)
only. The struct/constructor are already committed (Step 1 scaffold); only
method bodies change. `weyl_window_width`/`plan_weyl_windows`/
`beyn_disks_from_windows` are new free functions to add to this same file
(exported, since they're documented in the Step 1 plan §4.6 as staying free
functions for a future multi-window `solve_spectrum` driver).

## Special function notes
Both Bessels.jl and SpecialFunctions.jl contain implementations of Hankel nad Bessel functions. 
Bessels.jl are generally faster but may not contain functions for complex arguments so fall back on special functions if it is not available.

## Implementation steps

1. Port `weyl_window_width`/`plan_weyl_windows`/`beyn_disks_from_windows`
   verbatim — pure geometry/Weyl-law arithmetic, needs `fundamental_area`/
   `area` on `AbsBilliard` (confirm these already exist in
   `BilliardGeometry.jl`/`QuantumBilliards.jl`'s `spectra/unfolding.jl`
   `weyl_law`-adjacent utilities before assuming; if missing, port them from
   `BilliardGeometry-develop` first as a small prerequisite, not silently
   invented).
2. Port `beyn_buffer_matrices` verbatim (trivial preallocation).
3. **`construct_matrices(solver::BeynSolver, pts, k0, R; rng=MersenneTwister(0),
   multithreaded)`**: this is `construct_B_matrix` from `-develop`, renamed to
   match main's already-committed signature (Step 1 plan §4.6 — returns
   `(A0, A1)`, not the further-reduced `B`; confirm whether `-develop`'s
   `construct_B_matrix` returns the raw `(A0,A1)` moments or the
   already-reduced `B` and adjust: if `-develop` conflates moment-forming and
   projection into one function, split it at the API boundary already fixed
   in main — `construct_matrices` returns the two moments only, and `solve`
   does the SVD/projection/eigenproblem step, matching the established
   `construct_matrices`-then-`solve` split every other solver in this
   ecosystem follows).
   * The contour quadrature loop (one linear solve per node `zj`) is the
     natural outer-parallel loop: each node's `T(zj)⁻¹V` solve is independent
     — wrap it with `@use_threads multithreading=multithreaded for ... end`
     per the mode's single-parallel-level convention, with each thread
     writing to its own preallocated slice/accumulator before a final
     reduction into `A0`/`A1` (never accumulate into the same `A0`/`A1`
     matrix from multiple threads without a private-then-reduce pattern).
     Each node's own `construct_matrices(kernel, pts, zj)` +
     linear solve is itself potentially BLAS-heavy — wrap with `@blas_multi`
     around the per-node dense solve to avoid oversubscribing BLAS threads
     under the outer `@use_threads`, matching `-develop`'s presumed threading
     intent (confirm `-develop` doesn't already parallelize this loop some
     other way before choosing the wrapping).
4. **`solve(solver::BeynSolver, pts, k0, dk; multithreaded)`**: call
   `construct_matrices` for `R = dk/2`, do the rank-revealing SVD of `A0`
   (`svd_tol` cutoff), form the reduced generalized/plain eigenproblem
   `B = Uk'*A1*Wk*Σk⁻¹`, `eigen(B)`, filter candidates by contour containment
   (`|k−k0|≤R`) and residual (`res_tol`, `auto_discard_spurious`), return
   `(ks, ts)` — port `-develop`'s exact filtering logic, this is the part most
   prone to a silent correctness bug if re-derived instead of copied.
5. **`solve_vectors(solver, pts, k0, dk; multithreaded)`**: same as `solve`
   plus the eigenvectors, expanded back from the reduced `B`-eigenbasis to the
   full boundary-density basis via `Uk`/`Wk` (`x = Wk*eigvec` or similar —
   confirm exact expansion formula from `-develop` rather than guessing).
6. **`solve_wavenumber`/`solve_spectrum`**: thin wrappers picking the single
   best candidate near `k0`/returning every candidate in the window, calling
   `evaluate_points(solver, billiard, k0)` (already free via
   `acceleratedmethods.jl`'s existing `AcceleratedBIMSolver` generic
   delegating to `solver.kernel`) then `solve`/`solve_vectors`.
7. `use_chebyshev` stays `false`-only for this step (Step 10 fills it in) —
   confirm `construct_matrices` never branches into a Chebyshev path yet;
   if `solver.use_chebyshev == true` is passed before Step 10 lands, raise a
   clear `error("Chebyshev-accelerated Beyn evaluation not yet implemented,
   see migration plan step 10")` rather than silently falling back, so a user
   can't be misled into thinking it's already accelerated.

## Performance & fidelity notes

* The contour-quadrature loop is the single most expensive part of this
  solver (one dense linear solve per node) — this is exactly the
  "outer-parallel, inner-single-threaded, `@blas_multi`-bounded" pattern the
  mode instructions describe; do not nest `@use_threads` inside the
  per-node solve.
* Port the SVD-rank-detection/probing-dimension-auto-increase logic verbatim
  — it's a numerically delicate piece of `-develop`'s implementation.

## Tests & user verification

1. `get_errors` on `beyn.jl` and `QuantumBilliards.jl`.
2. Wrap a verified `DoubleLayerPotentialSolver` (Step 3) in `BeynSolver` and
   confirm it recovers the *same* eigenvalues, in the same window, that
   `k_sweep`/`solve_wavenumber` on the bare `DoubleLayerPotentialSolver`
   already found on Stadium/Triangle — this cross-validates both the contour
   assembly and the sweep solver from Step 3 against each other.
3. Test a window `dk` wide enough to contain 2–3 known eigenvalues (from the
   existing `test/solvertests.jl` "Low Spectrum"/"High Spectrum" testsets) and
   confirm `solve`/`solve_spectrum` recovers all of them in one contour solve,
   with residuals below `res_tol`.
4. Confirm spurious-root rejection: deliberately shrink `r`/`nq` and check
   that either fewer valid roots are returned or `auto_discard_spurious`
   correctly filters bad candidates rather than silently returning garbage.
5. `QBPlotting.jl`: `plot_sweep!` doesn't apply to `AcceleratedBIMSolver`
   (matches how `plot_sweep!(::AcceleratedBasisSolver, ...)` already differs
   in style from the `SweepBasisSolver` overload in
   [benchmarkplots.jl](/home/clozej/.julia/dev/QBPlotting.jl/src/benchmarkplots.jl) —
   read that existing `AcceleratedBasisSolver` method first) — add an
   analogous `plot_sweep!(ax, k_min, k_max, dk,
   solver::QuantumBilliards.AcceleratedBIMSolver, billiard; ...)` overload
   that windows through `[k_min,k_max]` calling `solve_spectrum` per window
   and scatter-plots the recovered `(k, t)` pairs, mirroring the
   `AcceleratedBasisSolver` version's plotting style. Ask the user to
   visually confirm the recovered points line up with the DLP tension-dip
   plot from Step 3.
6. Tell the user to invoke the **Julia Test Writer** subagent for
   `BeynSolver` regression tests once the above checks pass.

## Definition of done

* `BeynSolver` fully implemented (contour assembly, SVD projection,
  eigenpair recovery, filtering).
* Recovered eigenvalues cross-validated against the wrapped kernel's own
  `k_sweep` results on Stadium/Triangle.
* `QBPlotting.jl` has a working `AcceleratedBIMSolver` overload of
  `plot_sweep!`.
* Julia Test Writer invoked.
