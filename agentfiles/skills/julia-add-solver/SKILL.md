---
name: julia-add-solver
description: "Use when implementing a new solver (AbsBasisSolver, AbsBIMSolver, or any other AbsSolver branch) in QuantumBilliards.jl, or porting a solver from a -develop reference repo, so its API matches sibling solvers in the same class and every required multiple-dispatch method exists. Trigger phrases: add a new solver, implement a solver, new eigenvalue solver, port a solver, add a sweep/accelerated/BIM solver method, replicate solver structure."
---

# Adding/Porting a Solver in QuantumBilliards.jl

Implement a new concrete solver type so it slots into the existing
`AbsSolver` hierarchy purely via multiple dispatch. This skill assumes the
invoking agent's domain context and numerical/threading conventions (type
stability, memory efficiency, `@use_threads`/`@blas_multi`, single-parallel-
level threading) already apply.

## First principle: the struct shape is not universal — it's determined by the class of methods

Different classes of solvers need genuinely different numerical parameters,
and this skill must not force every new solver into one fixed template.
For example, a basis-expansion solver (`AbsBasisSolver` branch — spectral
methods that scale a basis dimension and a boundary point count) needs
entirely different attributes than a boundary-integral-method solver
(`AbsBIMSolver` branch — panel/Chebyshev discretization, Krylov iteration
tolerances). Forcing BIM parameters into the `AbsBasisSolver` six-field shape
(or vice versa) would be wrong.

The actual rule to follow:
1. **The root abstract type for the class of methods is the guideline for
   which attributes are needed**, not a fixed field list. Read that abstract
   type's docstring/API contract in `abstracttypes.jl` to see what the class
   as a whole is expected to support (e.g. `AbsBasisSolver`'s API implies it
   needs whatever fields let it scale a basis dimension and sample boundary
   points at a given wavenumber; `AbsBIMSolver`'s API — once documented for
   a concrete implementation — implies whatever fields let it build/solve a
   boundary integral equation).
2. **The struct must contain the numerical parameters the concrete algorithm
   actually needs** to run (tolerances, discretization/resolution controls,
   iterative-solver controls) — don't add fields the algorithm doesn't use
   just to match another class's shape, and don't omit a field the
   algorithm genuinely requires just to look uniform.
3. **Within one class (i.e. the same immediate abstract branch), sibling
   concrete solvers should share a similar structure** — same general shape,
   naming style, and section order — because that's what lets the class's
   shared/generic methods (see below) dispatch and operate on any member
   uniformly. "Similar" means consistent conventions (e.g. every field that
   is a tolerance ends in a recognizable name, every field driving
   resolution is named consistently), not identical field lists copy-pasted
   from an unrelated class.
4. When multiple sibling concrete types already exist in the target
   abstract branch, use the **closest existing sibling in that same branch**
   as the structural template (field naming/order/style), not a solver from
   a different branch.
5. When no sibling exists yet in the main repo for this class (e.g. the
   `AbsBIMSolver` branch currently only has stub definitions in
   `QuantumBilliards.jl` — see below), look at the closest analogous
   concrete implementation in the `-develop` reference repo for structural
   inspiration, but re-derive the fields against the main repo's own
   `abstracttypes.jl` API contract rather than copying verbatim (that
   `-develop` version may carry outdated fields/API).

## The key insight: within a class, most of the API is free via dispatch on the abstract type

For the `AbsBasisSolver` branch specifically, `abstracttypes.jl` documents
the full API (`evaluate_points`, `construct_matrices`, `solve`, `solve_vect`/
`solve_vectors`, `solve_wavenumber`, `k_sweep`/`solve_spectrum`,
`compute_eigenstate`) and **most of these are already implemented generically**
on the abstract branch, not per concrete type:

| Function | Where it's generic | A new concrete solver must still... |
|---|---|---|
| `adjust_scaling_and_samplers` | generic on `AbsBasisSolver` (`solvers/decompositions.jl`) | do nothing — works via field names alone |
| `solve_wavenumber` | generic on `SweepBasisSolver`/`AcceleratedBasisSolver` (`sweepmethods.jl`/`acceleratedmethods.jl`) | do nothing — calls `evaluate_points`/`solve` |
| `k_sweep` (sweep) / `solve_spectrum` (accelerated) | generic on the same abstract branch | do nothing |
| `compute_eigenstate` | generic on `SweepBasisSolver`/`AcceleratedBasisSolver` (`states/eigenstates.jl`) | do nothing |
| `evaluate_points` | **not generic** | implement for the concrete type |
| `construct_matrices` | **not generic** | implement for the concrete type |
| `solve` | **not generic** | implement for the concrete type |
| `solve_vect` (sweep) / `solve_vectors` (accelerated) | **not generic** | implement for the concrete type |

This "struct fields → free generic methods" relationship is specific to how
`AbsBasisSolver`'s generic methods are written (they read
`solver.dim_scaling_factor`/`solver.pts_scaling_factor`/`solver.sampler`/
`solver.min_dim`/`solver.min_pts` directly). **A different class of solvers
will have a different, or no, set of free generic methods** — check the
target abstract branch's own generic methods (search for `function
<name>(solver::<AbstractBranch>` across `solvers/`) before assuming anything
is free. If the class provides no generic methods yet (e.g. `AbsBIMSolver`
currently has none implemented in the main repo), every required function in
that abstract type's documented API must be implemented for the new concrete
type.

So for an `AbsBasisSolver`-branch solver specifically, the actual amount of
new code needed is small: a struct, two constructors, and four methods.
Everything else is inherited for free **provided** the struct uses the exact
field names below. For other classes, re-derive which methods are free vs.
required from that class's own abstract-type API and generic methods.

## Struct shape for an `AbsBasisSolver`-branch solver (worked example)

The pattern below is the established convention for this specific class —
use it as-is when the new solver is genuinely a basis-expansion method
(subtypes `SweepBasisSolver` or `AcceleratedBasisSolver`). Do **not** reuse
this exact field set for a different class (e.g. `AbsBIMSolver`) — see
"Adapting the struct to a different class of methods" below for that case.

Subtype `SweepBasisSolver` (wavenumber-by-wavenumber tension scan) or
`AcceleratedBasisSolver` (single-diagonalization window around `k`) —
decide which based on the algorithm's actual behavior, not convenience.

```julia
struct NewMethodSolver{T} <: SweepBasisSolver where {T<:Real}     # or AcceleratedBasisSolver
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
end
```

Within the `AbsBasisSolver` class specifically, these six field names are
load-bearing: `adjust_scaling_and_samplers`, `solve_wavenumber`,
`k_sweep`/`solve_spectrum` all read `solver.dim_scaling_factor`,
`solver.pts_scaling_factor`, `solver.sampler`, `solver.min_dim`/
`solver.min_pts` directly. Do not rename or add an indirection layer for a
basis-expansion solver. If the algorithm genuinely needs extra parameters
(e.g. a window half-width default, a Krylov tolerance), add them as **extra
trailing fields** — never remove or rename the six above. This load-bearing
requirement is specific to this class; it does not apply to other classes
(see below).

## Adapting the struct to a different class of methods

When the new solver belongs to a different abstract branch — e.g.
`AbsBIMSolver`/`SweepBIMSolver`/`AcceleratedBIMSolver` for boundary integral
methods, or any future class — do not reuse the `AbsBasisSolver` six-field
shape. Instead:

1. Read the target abstract type's docstring in `abstracttypes.jl` for its
   documented API and description — this tells you what the struct's fields
   need to *support* (e.g. building/solving a boundary integral equation),
   not what to literally name them.
2. Enumerate the actual numerical parameters the concrete algorithm needs
   (e.g. for a Chebyshev-panel boundary integral method: panel-count/growth
   controls, Chebyshev interpolation tolerance, Krylov solver tolerance and
   subspace dimension, batch size for multi-wavenumber solves — see the
   `-develop` repo's `DLP`/`CFIE`-family structs for a concrete sense of
   what this class of parameter set looks like, purely as inspiration, not
   a template to copy field-for-field).
3. If one or more concrete solvers already exist under the same immediate
   abstract branch in the main repo, mirror their field naming/order/style
   exactly — this is what "solvers in the same class share a similar
   structure" means in practice, and it's what lets any shared/generic
   methods for that branch (if present) dispatch uniformly.
4. If none exist yet in the main repo for this class, it's acceptable (and
   expected) that this new solver establishes the first concrete shape for
   that branch — pick clear, mathematically meaningful field names
   consistent with the rest of the package's naming conventions, favor
   concrete parametric types (`T<:Real`, `Int`), and document the shape via
   the struct's docstring `## Attributes` section so future sibling solvers
   in the same class have something to mirror.
5. Do not assume the four "required methods" list below (`evaluate_points`/
   `construct_matrices`/`solve`/`solve_vect(ors)`) applies verbatim to a
   different class — re-derive the required method set from that class's
   own abstract-type API list; a BIM solver might need a matrix assembly +
   Krylov solve pair instead of a generalized-eigenvalue pair, for example.

## Required constructors (two, matching every existing solver)

```julia
function NewMethodSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}; min_dim = 100, min_pts = 500) where T<:Real
    d = dim_scaling_factor
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    sampler = [GaussLegendreNodes()]
    return NewMethodSolver(d, bs, sampler, eps(T), min_dim, min_pts)
end

function NewMethodSolver(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}}, samplers::Vector{AbsSampler}; min_dim = 100, min_pts = 500) where {T<:Real}
    d = dim_scaling_factor
    bs = typeof(pts_scaling_factor) == T ? [pts_scaling_factor] : pts_scaling_factor
    return NewMethodSolver(d, bs, samplers, eps(T), min_dim, min_pts)
end
```

## Required method: `evaluate_points` (`AbsBasisSolver` worked example)

The following applies to the `AbsBasisSolver` class specifically. A
different class's boundary-sampling/setup step may have a different name,
signature, or not exist at all (e.g. a BIM solver's setup might instead
build panel/Chebyshev discretization data) — check that class's abstract
type API before assuming `evaluate_points` is the right function to
implement.

```julia
function evaluate_points(solver::NewMethodSolver, billiard::Bi, k) where {Bi<:AbsBilliard}
    bs, samplers = adjust_scaling_and_samplers(solver, billiard)   # free, generic
    curves = get_boundary_curves(billiard)
    # sample each curve with its sampler, compute algorithm-specific weights
    # (e.g. w_vs for Vergini-Saraceno, w_dm for decomposition method)
    return BoundaryPoints(vcat(xy_all...); <algorithm_specific_kwargs...>)
end
```
Populate only the `BoundaryPoints` (`QuantumBilliards.jl/src/solvers/boundarypoints.jl`)
fields the algorithm actually needs (`w_vs`, `w_dm`, `normal`, `ds`, etc.) —
unused fields default to empty vectors, don't fabricate values for them.

## Required method: `construct_matrices` (`AbsBasisSolver` worked example)

```julia
function construct_matrices(solver::NewMethodSolver, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool = true) where {Ba<:AbsBasis}
    @timeit_debug "construct_matrices" begin
        # build the algorithm's matrices from basis_matrix/gradient_matrices/dk_matrix
        # (solvers/matrixconstructors.jl), scaled by pts weights and nsym via
        # _scale_rows_sqrtw! + BLAS.syrk!/syr2k! + _symmetrize_from_upper!
        # (see the julia-solver-debug-timing skill for the debug/timing pattern)
    end
end
```
Reuse `basis_matrix`/`gradient_matrices`/`basis_and_gradient_matrices`/
`dk_matrix` from `matrixconstructors.jl` — never re-evaluate the basis by
hand. Reuse the BLAS rank-k-update + symmetrize pattern (`_scale_rows_sqrtw!`,
`BLAS.syrk!`/`syr2k!`, `_symmetrize_from_upper!`, `_build_Bn_inplace!`) from
the same file instead of writing new matrix-multiply boilerplate. Load the
`julia-solver-debug-timing` skill to add the `@timeit_debug`/`@debug`
instrumentation expected by every existing `construct_matrices`.

## Required methods: `solve` and `solve_vect`/`solve_vectors` (`AbsBasisSolver` worked example)

- `SweepBasisSolver` branch: implement `solve(solver, basis, pts, k; multithreaded)` returning a scalar tension (used by `k_sweep`/`solve_wavenumber`'s `Optim.optimize`), and `solve_vect(solver, basis, pts, k; multithreaded)` returning the eigenvector plus tension for `compute_eigenstate` (`QuantumBilliards.jl/src/states/eigenstates.jl`).
- `AcceleratedBasisSolver` branch: implement `solve(solver, basis, pts, k, dk; multithreaded)` returning `(ks, ts)` candidate wavenumbers/tensions within the window, and `solve_vectors(solver, basis, pts, k, dk; multithreaded)` returning the corresponding eigenvectors too.
- Build the generalized eigenproblem with `generalized_eigen`/`generalized_eigvals` (`QuantumBilliards.jl/src/solvers/decompositions.jl`), truncated via `solver.eps` — don't call `eigen`/`eigen!` directly, these wrappers already have the correct truncation, symmetrization and debug/timing instrumentation.
- For a different class (e.g. a BIM solver solving a boundary integral equation via Krylov iteration instead of a generalized eigenvalue problem), the equivalent "solve" step should still follow that class's own abstract-type API and reuse whatever shared linear-algebra utilities already exist for it (don't invent a new eigen/Krylov wrapper if one already exists elsewhere in `solvers/`).

## Wiring checklist (apply after the above compiles)

1. Place the new file in the subfolder matching the algorithm's class (e.g. `solvers/sweepmethods/` or `solvers/acceleratedmethods/` for `AbsBasisSolver`-branch solvers; create/use the analogous subfolder for other classes if one doesn't already exist).
2. Add `include("solvers/.../newmethod.jl")` to the matching grouping file for that class (e.g. `sweepmethods.jl`/`acceleratedmethods.jl`), not directly in `QuantumBilliards.jl`.
3. Export the new struct's name (and any new public helper, e.g. a `_results` converter) from `QuantumBilliards.jl` alongside the other solver exports for that class.
4. Do not re-export a function that's already exported once and shared across a class (e.g. `evaluate_points`, `construct_matrices`, `solve`) — check `QuantumBilliards.jl`'s existing export list before adding a duplicate.
5. If genuinely stuck on wiring/export placement, delegate to the `julia-refactor-wire-feature` skill rather than guessing.

## Approach
1. Read `abstracttypes.jl` (`QuantumBilliards.jl/src/abstracttypes.jl`) fully to determine the correct class/branch for the new solver (e.g. `AbsBasisSolver`'s `SweepBasisSolver`/`AcceleratedBasisSolver` for basis-expansion methods, `AbsBIMSolver`'s `SweepBIMSolver`/`AcceleratedBIMSolver` for boundary-integral methods, or another existing/new branch) based on the algorithm's actual mathematical behavior, not convenience.
2. Read that abstract type's docstring (API list, description, attributes hints) to determine what the struct's fields need to support and which functions must exist for the class.
3. Search for any concrete sibling(s) already implemented under that same immediate branch in the main repo (e.g. `VerginiSaracenoSolver`/`DecompositionMethodSolver` for `AbsBasisSolver`) and read the closest one in full as the structural template — match its section order (struct → constructors → setup/point-evaluation step → matrix/system construction → results conversion → solve step(s)). If no sibling exists yet for this class, look at the `-develop` repo's closest analogous implementation purely for inspiration into what parameters the algorithm needs, then re-derive the shape against the main repo's `abstracttypes.jl`.
4. Search for any generic methods already implemented directly on the target abstract branch (e.g. `solve_wavenumber`, `k_sweep`/`solve_spectrum`, `compute_eigenstate`, `adjust_scaling_and_samplers` for `AbsBasisSolver`) to determine which functions are inherited for free versus which must be implemented for the new concrete type.
5. If porting from a `-develop` repo, read the source there first, then re-derive the struct/constructor/method shapes rather than copying — the `-develop` version's field names/API may be stale.
6. Implement the concrete type's struct, constructors, and required methods, reusing existing shared utilities for that class (e.g. `matrixconstructors.jl`/`decompositions.jl` for `AbsBasisSolver`-branch solvers) and the numeric/threading conventions (preallocate, `@view`, single-level `@use_threads`, BLAS rank-k updates where applicable).
7. Wire the file in per the checklist above.
8. Run `get_errors` on the new file and `QuantumBilliards.jl` to confirm the package still loads/compiles.
9. Report which class/branch was chosen and why, the struct's field set and how it maps to that class's numerical needs, which methods were implemented vs. inherited for free, and confirm any free/generic functions for that class now work for the new solver without modification.
10. Remind the user that debug/timing instrumentation can be added or reviewed via the `julia-solver-debug-timing` skill, and tests via the Julia Test Writer subagent (never invoke it yourself).

## Output Format
Directly edit/create the source file(s). Then give a brief summary (not a new
markdown file): which abstract branch was chosen and why, which methods were
implemented vs. inherited for free, and any wiring changes made to the
grouping file / `QuantumBilliards.jl` exports.
