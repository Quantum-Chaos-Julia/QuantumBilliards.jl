# Step 9 — `ExpandedBIMSolver`

## Goal

Implement the second-order local Taylor-expansion root correction in
[solvers/acceleratedmethods/ebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl),
wrapping the same kernel types as `BeynSolver`. Done after Beyn per the user's
"accelerated methods can come later" ordering, and because it's the smaller,
more local of the two accelerated methods.

## Preconditions

Step 8 complete (not a hard numerical dependency — `ExpandedBIMSolver` doesn't
call into `BeynSolver` — but keeping the accelerated methods in this order
matches the plan and lets both share any `construct_matrices`-with-derivatives
helper conventions established while doing Beyn).

## Source index

`QuantumBilliards-develop/src/solvers/acceleratedmethods/ebim.jl` — already
read in full during planning through the Chebyshev-cache section. Key
mathematical content (from the file's own header comment): for the Fredholm
matrix `A(k)`,

```
A(k+ε) = A(k) + εA'(k) + ½ε²A''(k) + O(ε³)
A(k)v = λA'(k)v        (generalized eigenproblem → ε₁ = -λ)
ε₂ = -½ε₁² [u†A''(k)v] / [u†A'(k)v]     (u = left eigenvector)
k_corr = k + ε₁ + ε₂
```

Port: `allocate_ebim_matrices(solver, pts)` (preallocates `A`, `dA`, `ddA` as
`N×N` `ComplexF64` — check whether this should be genuinely generic in `T`
rather than hardcoded `ComplexF64`, per the mode's "propagate concrete type
parameters" rule; `-develop` hardcodes `ComplexF64` here, likely because the
whole EBIM pathway was only ever exercised in double precision — flag this to
the user rather than silently either keeping the `Float64`-only shortcut or
unilaterally genericizing it, since genericizing touches the Chebyshev cache
types too (out of scope until Step 10)). The actual `construct_matrices`
producing `(A, dA, ddA)` (the direct, non-Chebyshev path — search past the
Chebyshev-cache section already read for a plain/direct derivative-matrix
assembly function, likely named something like
`construct_matrices_with_derivatives!`), and the `solve` root-correction body
implementing the formula above via `generalized_eigen`-style tooling (check
whether `-develop` reuses `QuantumBilliards`'s existing
`generalized_eigen`/`generalized_eigvals` from `solvers/decompositions.jl` for
the `A(k)v=λA'(k)v` step, or has its own — prefer reusing the existing
`decompositions.jl` utilities if the shapes line up, per the "reuse existing
utilities" mode guidance, but don't force a mismatched API onto them if
`-develop`'s problem is genuinely non-Hermitian/needs a different eigensolver
path than the real generalized-eigenvalue problems `decompositions.jl`
currently handles).

## Files touched in main

[solvers/acceleratedmethods/ebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl)
only. Struct/constructor already committed from Step 1 (this is the file
where the Step 1 pass also fixed the stale `EBIMSolver`→`ExpandedBIMSolver`
export bug — confirmed already correct, no further renaming needed).

## Implementation steps

1. Port `allocate_ebim_matrices` (adapted signature: main's
   `boundary_matrix_size(solver.kernel, pts)` instead of `-develop`'s
   `boundary_matrix_size(solver, pts)`, since main's `boundary_matrix_size`
   dispatches on the concrete `SweepBIMSolver` kernel, not on the wrapping
   `ExpandedBIMSolver` itself — confirm this dispatch works once Step 3's
   `boundary_matrix_size(::DoubleLayerPotentialSolver, ...)` exists).
2. **`construct_matrices(solver::ExpandedBIMSolver, pts, k; multithreaded)`**:
   port the direct (non-Chebyshev) `(A, dA, ddA)` assembly. This likely means
   evaluating the kernel's Hankel-function terms *and* their first/second
   `k`-derivatives at every pairwise node — reuse Step 3/4's
   `BoundaryGeomCache` for the shared pairwise geometry (`R`, `inner`,
   `kappa`, `speed`) and only add the derivative-of-Hankel-function pieces
   here (`Bessels.hankelh1` has known recurrence relations for its
   `k`-derivative in terms of `hankelh1`/`hankelh0`-type identities — port
   `-develop`'s exact derivative formulas rather than re-deriving them from
   scratch, to avoid introducing a sign/recurrence error).
3. **`solve(solver::ExpandedBIMSolver, pts, k; multithreaded)`**: assemble
   `(A, dA, ddA)` via `construct_matrices`, solve `A v = λ dA v` for the
   dominant/smallest-|λ| generalized eigenpair (and its left eigenvector
   `u`), compute `ε₁ = -λ`, `ε₂` per the formula above, return
   `(k + ε₁ + ε₂, t0)` where `t0` is a residual-based tension consistent with
   how `-develop` reports it (confirm the exact tension definition
   `-develop` returns alongside `k_corr` rather than inventing one).
4. **`solve_wavenumber`/`solve_spectrum`**: thin wrappers over `solve`,
   consistent with the already-committed signatures in the Step 1 plan §4.7.
5. `use_chebyshev` stays `false`-only until Step 10, same guard pattern as
   `BeynSolver` (raise a clear "not yet implemented" error rather than
   silently ignoring the flag).

## Performance & fidelity notes

* The derivative-kernel evaluation is the new numerically sensitive piece
  here (not present in Steps 3, 6 and 7) — port `-develop`'s exact derivative
  formulas, do not re-derive Hankel-function derivatives from first
  principles independently, to avoid a subtle sign error that would silently
  produce a wrong root correction.
* Reuse `BoundaryGeomCache` from Step 2/3 rather than recomputing pairwise
  geometry a third time.

## Tests & user verification

1. `get_errors` on `ebim.jl` and `QuantumBilliards.jl`.
2. Wrap a verified `DoubleLayerPotentialSolver` (Step 3) in `ExpandedBIMSolver`
   and confirm, starting from a `k` already known (from `k_sweep`) to be close
   to a true eigenvalue, that `solve` returns a corrected `k_corr` at least as
   close to the reference eigenvalue as the starting `k`, and ideally closer
   (second-order local correction should meaningfully sharpen a good initial
   guess).
3. Cross-check `ExpandedBIMSolver`'s corrected `k_corr` against `BeynSolver`'s
   (Step 8) recovered eigenvalue in the same window — both accelerated
   methods should agree with each other and with the Step 3 sweep result.
4. `QBPlotting.jl`: no new plotting overload strictly required beyond what
   Step 8 already added for `AcceleratedBIMSolver` (shared abstract branch) —
   confirm the Step 8 `plot_sweep!` overload also works correctly for
   `ExpandedBIMSolver` instances (it dispatches on the abstract
   `AcceleratedBIMSolver` type, so it should, but verify `solve_spectrum`'s
   return shape matches what that plotting method expects).
5. Tell the user to invoke the **Julia Test Writer** subagent for
   `ExpandedBIMSolver` regression tests once the above checks pass.

## Definition of done

* `ExpandedBIMSolver` fully implemented (direct derivative-matrix assembly,
  second-order root correction).
* Corrected wavenumbers cross-validated against `BeynSolver` and the Step 3
  sweep solver on shared fixtures.
* Julia Test Writer invoked.
