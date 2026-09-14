# Migration plan: boundary-integral / Kress solvers, `QuantumBilliards-develop` → `QuantumBilliards.jl`

Status: **API scaffolding implemented, algorithm bodies pending.** This
document was the Step-1 deliverable (inventory of the prototype, gap analysis
against the main package's established solver conventions, architecture
decisions, and a concrete struct/API design for every solver family). All
open questions in §8 have since been resolved with the user, and the
resulting structs, constructors and method *signatures* have been scaffolded
directly into `QuantumBilliards.jl`/`BilliardGeometry.jl` (every method body
currently raises a descriptive `error` — see §9). Porting the actual
matrix-assembly/numerical bodies into those stubs is Step 2 and is *not* done
here.

---

## 1. Scope / inventory of `QuantumBilliards-develop`

`AbsSolver` in develop branches into `SweepSolver` / `AcceleratedSolver`. Two
sub-families exist beneath each, mixed together under those two abstract types:

**Basis-expansion solvers** (already fully mirrored in main as
`SweepBasisSolver`/`AcceleratedBasisSolver`):

| develop type | status in main |
|---|---|
| `DecompositionMethodSolver` | ported ([decompositionmethod.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/decompositionmethod.jl)), missing only the `rellich_origin` field (develop added a Rellich-weight origin `c₀`) |
| `VerginiSaracenoSolver` | ported ([verginisaraceno.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/verginisaraceno.jl)) |
| `ParticularSolutionsMethod` (PSM) | **not ported at all** — no placeholder exists in main yet |

**Boundary-integral (Nyström/Kress) solvers** (main has only empty placeholders,
under the newer `AbsBIMSolver`/`SweepBIMSolver`/`AcceleratedBIMSolver` branch
that develop doesn't have):

| develop type(s) | purpose | main placeholder |
|---|---|---|
| `BoundaryIntegralMethod` | direct (uncorrected) Nyström DLP, `A(k)=I-K(k)` | — (no equivalent placeholder) |
| `DLP_kress` | Kress log-corrected DLP, single smooth closed curve | [dlp.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/dlp.jl) → `DoubleLayerPotentialSolver{T}` (empty) |
| `DLP_kress_global_corners` | Kress-graded DLP, composite/cornered boundary | same placeholder |
| `CFIE_kress` | Kress log-corrected CFIE, `A(k)=I-(D(k)+ikS(k))`, single smooth closed curve | [cfie.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/cfie.jl) → `CombinedFieldIntegralEquationSolver{T}` (empty) |
| `CFIE_kress_corners` | Kress-graded CFIE, single curve, parametric corner clustering | same placeholder |
| `CFIE_kress_global_corners` | Kress-graded CFIE, composite/multi-segment boundary, corners auto-detected | same placeholder |
| `CFIE_kress_composite_solver` | multiply-connected geometry, one CFIE-Kress solver per connected component | same placeholder |
| `BeynSolver` (a `Union` trait, not a struct!) | Beyn contour-integral nonlinear-eigenvalue extraction, wraps *any* of the above Fredholm kernels | [beyn.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/beyn.jl) → `BeynSolver{T}` (empty struct) |
| `EBIMSolver` (also a `Union` trait) | local Taylor-expansion (2nd order) root correction on the same Fredholm kernels | [ebim.jl](/home/clozej/.julia/dev/QuantumBilliards.jl/src/solvers/acceleratedmethods/ebim.jl) → `ExpandedBIMSolver{T}` (empty struct) |

Supporting infrastructure in develop that the BIM solvers depend on, and that
does **not** exist yet in main (`QuantumBilliards.jl` or `BilliardGeometry.jl`):
Kress logarithmic-correction matrices (`kress_R!`, `kress_R_even!`,
`kress_R_odd!`, single- and multi-corner grading maps), `SymmetryOrbitMap` /
`symmetry_index_orbits` (fold a full periodic boundary onto a symmetry
fundamental domain), `BoundaryGeomCache`/`BoundaryPanelArrays` (reusable
pairwise-geometry workspaces), and the whole Chebyshev-interpolated
Hankel/Bessel evaluation pathway (`chebyshev_*.jl`, used to accelerate repeated
kernel evaluation across many wavenumbers, e.g. inside a Beyn contour or an
EBIM/dense sweep). These are **prerequisites**, not part of the solver API
itself — flagged in §6, ported separately (likely into `BilliardGeometry.jl`
for the geometry-only pieces).

---

## 2. Architecture decisions (deviations from the develop prototype)

The develop code is an experimental prototype; several of its choices don't
match the conventions already established by `DecompositionMethodSolver` /
`VerginiSaracenoSolver`. These are fixed **before** designing the structs below.

1. **No `billiard` field inside the solver struct.** Every `*_kress*` struct in
   develop stores `billiard::Bi` directly, which is redundant given that
   `evaluate_points(solver, billiard, k)` already receives the billiard at
   call time (same signature used by every existing main solver), and it makes
   the solver object geometry-bound (can't be reused across billiards, can't
   be part of a generic sweep helper). Main solvers keep `billiard` as a
   call-time argument only. Geometry-dependent precomputation (corner
   detection, Kress grading maps, symmetry-orbit maps, panel/geometry caches)
   is built by `evaluate_points`/`construct_matrices` and returned as part of
   `BoundaryPoints`/a solver-owned workspace, exactly like develop's
   `DLPKressWorkspace` already does internally.

2. **One struct per solver family, not one struct per grading strategy.**
   Develop has 3 separate `DLP_kress*` types and 4 separate `CFIE_kress*`
   types differing only in *how the boundary is graded/split into
   components*. Splitting behavior into a small `BoundaryGrading` trait
   (dispatched on, like `AbsBasis`/`AbsSymmetry` already are) collapses this
   into exactly the two family structs the main placeholders already commit to
   (`DoubleLayerPotentialSolver`, `CombinedFieldIntegralEquationSolver`),
   matching "add a method instead of a struct" from the multiple-dispatch
   guidance, and is far more user-friendly (one type to import per family, one
   keyword to change behavior).

3. **`Beyn`/`EBIM` become genuine parameter-holding wrapper solvers, not
   `Union` traits.** In develop, `BeynSolver`/`EBIMSolver` are just type
   aliases naming "the solvers this algorithm accepts"; all of Beyn's numeric
   knobs (`nq`, `r`, `svd_tol`, `res_tol`, Chebyshev panel counts, …) are
   scattered across keyword arguments at every call site. Every existing main
   `AcceleratedBasisSolver`/`SweepBasisSolver` instead stores *all* its
   numerical parameters as fields with a keyword constructor
   (`VerginiSaracenoSolver(dim_scaling_factor, pts_scaling_factor; min_dim, min_pts)`).
   For consistency, `BeynSolver`/`ExpandedBIMSolver` become wrapper structs that
   hold `kernel::K` (any `SweepBIMSolver` instance, i.e. a
   `DoubleLayerPotentialSolver` or `CombinedFieldIntegralEquationSolver`) plus
   their own tunable fields, mirroring how `AcceleratedBasisSolver` wraps a
   basis-independent numerical recipe.

4. **BIM `solve`/`construct_matrices` drop the dummy basis argument.** Develop
   passes a placeholder `AbstractHankelBasis()` through `solve`/`solve_vect`
   purely so it can reuse the *basis-family* `solve_wavenumber`/`k_sweep`
   generic code. Main already has a distinct `AbsBIMSolver` branch sitting
   next to `AbsBasisSolver`, so this is unnecessary: `SweepBIMSolver`/
   `AcceleratedBIMSolver` get their **own** `solve_wavenumber`/`k_sweep`/
   `solve_spectrum` generics (in `solvers/sweepmethods/sweepmethods.jl` /
   `solvers/acceleratedmethods/acceleratedmethods.jl`, alongside the existing
   basis-solver ones) with signatures that never mention `basis`. Drop
   `AbstractHankelBasis` entirely.

5. **Symmetry stored the same way `CornerAdaptedFourierBessel` already does
   it.** `CornerAdaptedFourierBessel{T,Sy} where Sy<:Union{AbsSymmetry,Nothing}`
   is the existing convention for "an optional symmetry, baked into a type
   parameter". BIM solvers reuse exactly that pattern (`symmetry::Sy` field,
   `Sy<:Union{AbsSymmetry,Nothing}`), since for BIM there is no basis to carry
   the symmetry — the unknowns are boundary densities, so the boundary
   discretization/quadrature itself must be symmetry-aware.

6. **Composite (mixed per-component) solving generalizes to both families.**
   Develop only gives `CFIE_kress_composite_solver` (DLP has no composite
   variant). Main instead gets one small generic wrapper,
   `CompositeBIMSolver{T,CS<:Tuple}`, usable for either family (a tuple of
   component solvers, each itself a `DoubleLayerPotentialSolver` or
   `CombinedFieldIntegralEquationSolver`), instead of a bespoke CFIE-only type.

---

## 3. Proposed abstract-type documentation additions

`abstracttypes.jl` in main already declares `AbsBIMSolver`/`SweepBIMSolver`/
`AcceleratedBIMSolver` but without the docstrings that every other abstract
solver type has. Add docstrings analogous to the existing
`AbsBasisSolver`/`SweepBasisSolver`/`AcceleratedBasisSolver` ones, stating:

- `AbsBIMSolver <: AbsSolver`: solvers that determine the spectrum from a
  Nyström/boundary-integral discretization of a Fredholm operator `A(k)` acting
  directly on boundary densities (no `AbsBasis` expansion involved).
- `SweepBIMSolver <: AbsBIMSolver`: tension at fixed `k` is (a function of) the
  smallest singular value / nullspace residual of `A(k)`; concrete types
  `DoubleLayerPotentialSolver`, `CombinedFieldIntegralEquationSolver`,
  `CompositeBIMSolver`.
- `AcceleratedBIMSolver <: AbsBIMSolver`: recovers several roots of the
  nonlinear eigenproblem `A(k)v=0` near a target `k` from one contour solve
  (`BeynSolver`) or one local Taylor expansion (`ExpandedBIMSolver`), each
  wrapping an inner `SweepBIMSolver` kernel.

State representation note (flag for a later phase, not solved here): a BIM
eigenstate is a boundary *density* vector plus the `BoundaryPoints` it was
solved on, evaluated pointwise via the layer-potential representation formula
— it is **not** expressed in an `AbsBasis`, so it cannot reuse `StationaryState`/
`Eigenstate` as-is. A new `AbsState` branch (e.g. `BIMEigenstate <: AbsState`)
will be needed in `states/eigenstates.jl` when wavefunction evaluation is
migrated; out of scope for the solver-struct pass.

---

## 4. Struct / API design per solver family

Only field lists and function **signatures** are given (no bodies) — this is
the API contract Step 2 implements against.

### 4.1 `ParticularSolutionsMethod` (new — currently unported, basis family)

```julia
struct ParticularSolutionsMethod{T} <: SweepBasisSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    int_pts_scaling_factor::T
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
    min_int_pts::Int64
end

ParticularSolutionsMethod(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}},
                          int_pts_scaling_factor::T;
                          min_dim::Int=100, min_pts::Int=500, min_int_pts::Int=500) where {T<:Real}
ParticularSolutionsMethod(dim_scaling_factor::T, pts_scaling_factor::Union{T,Vector{T}},
                          int_pts_scaling_factor::T, samplers::Vector{<:AbsSampler};
                          min_dim::Int=100, min_pts::Int=500, min_int_pts::Int=500) where {T<:Real}

evaluate_points(solver::ParticularSolutionsMethod, billiard::Bi, k) where {Bi<:AbsBilliard} -> pts::BoundaryPoints  # xy, normal, s, ds, xy_int populated
construct_matrices(solver::ParticularSolutionsMethod, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true) where {Ba<:AbsBasis} -> (B::Matrix, B_int::Matrix)
solve_full(solver::ParticularSolutionsMethod, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true) where {Ba<:AbsBasis} -> t::Real
solve_with_rank_reduction(solver::ParticularSolutionsMethod, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true) where {Ba<:AbsBasis} -> t::Real
solve(solver::ParticularSolutionsMethod, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true) where {Ba<:AbsBasis} -> t::Real
solve_vect(solver::ParticularSolutionsMethod, basis::Ba, pts::BoundaryPoints, k; multithreaded::Bool=true) where {Ba<:AbsBasis} -> (t::Real, x::Vector)
```

`solve_wavenumber`/`k_sweep` are inherited for free from the shared
`SweepBasisSolver` generics — no new methods needed (this is exactly the
`DecompositionMethodSolver` pattern).

*Note:* develop's PSM identifies the tension as the smallest singular value of
`B` restricted to the interior-normalized `B_int` (a genuine least-squares
"particular solution" method, unlike the generalized-eigenvalue tension of
`DecompositionMethodSolver`), so `min_int_pts`/`int_pts_scaling_factor` are
kept as genuinely new fields (interior sampling is unique to PSM).

### 4.2 `DecompositionMethodSolver` (ported — add missing field only)

Add the one field develop introduced beyond what's already in main:

```julia
struct DecompositionMethodSolver{T} <: SweepBasisSolver where {T<:Real}
    dim_scaling_factor::T
    pts_scaling_factor::Vector{T}
    sampler::Vector
    eps::T
    min_dim::Int64
    min_pts::Int64
    rellich_origin::SVector{2,T}   # NEW: origin c₀ for the Rellich boundary weight w_dm = ds*((x-c₀)⋅n)/(2k²)
end
```

Both existing constructors gain a `rellich_origin::SVector{2,T} = SVector{2,T}(zero(T), zero(T))`
keyword default so this is a non-breaking, additive change; `evaluate_points`
changes its `w_dm` formula to use `x - solver.rellich_origin` instead of `x`.

### 4.3 `DoubleLayerPotentialSolver` (fills existing placeholder)

```julia
abstract type BoundaryGrading end
struct SmoothPeriodicGrading <: BoundaryGrading end                       # DLP_kress equivalent
struct GlobalCornerGrading{T<:Real} <: BoundaryGrading                    # DLP_kress_global_corners equivalent
    kressq::Int
    min_t_spacing::T
end
GlobalCornerGrading(; kressq::Int=2, min_t_spacing::T=1e-12) where {T<:Real}

struct DoubleLayerPotentialSolver{T<:Real,G<:BoundaryGrading,Sy<:Union{AbsSymmetry,Nothing}} <: SweepBIMSolver
    pts_scaling_factor::Vector{T}
    min_pts::Int64
    grading::G
    symmetry::Sy
    eps::T                      # numerical tolerance for nullspace/tension extraction
end

DoubleLayerPotentialSolver(pts_scaling_factor::Union{T,Vector{T}};
                           min_pts::Int=200, grading::BoundaryGrading=SmoothPeriodicGrading(),
                           symmetry::Union{Nothing,AbsSymmetry}=nothing, eps::T=T(1e-15)) where {T<:Real}

evaluate_points(solver::DoubleLayerPotentialSolver, billiard::Bi, k) where {Bi<:AbsBilliard} -> pts::BoundaryPoints  # dispatches internally on solver.grading
boundary_matrix_size(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints) -> N::Int   # accounts for symmetry-orbit folding
construct_matrices(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> A::Matrix{Complex{T}}
solve(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true, use_krylov::Bool=true) -> t::Real          # smallest singular value / Krylov nullspace residual of A(k)
solve_vect(solver::DoubleLayerPotentialSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> (t::Real, x::Vector{Complex{T}})   # boundary density (not basis coefficients)
```

`min_dim`/`dim_scaling_factor` from develop's compatibility fields are dropped:
they existed purely so BIM structs satisfied the *basis-solver* generic
`solve_wavenumber`, which (per §2.4) BIM solvers no longer share.

`solve_wavenumber(solver::SweepBIMSolver, billiard, k, dk; multithreaded=true)`
and `k_sweep(solver::SweepBIMSolver, billiard, ks; multithreaded=true)` become
new shared generics in `sweepmethods.jl` (BIM branch) — same optimize-the-tension
/ scan-the-tension logic as the basis-solver versions, just without a `basis`
argument and without the `resize_basis` step.

*Confirmed dropped:* the plain uncorrected `BoundaryIntegralMethod` (no Kress
log-splitting) is **not** migrated, in any form. It is strictly dominated in
accuracy by `SmoothPeriodicGrading` at negligible extra cost, and its behavior
is already fully contained in `DoubleLayerPotentialSolver` with the default
`grading = SmoothPeriodicGrading()`. No `UncorrectedGrading` variant will be
added.

### 4.4 `CombinedFieldIntegralEquationSolver` (fills existing placeholder)

Same `BoundaryGrading` trait as DLP, plus the single-curve corner case DLP
never had:

```julia
struct CornerGrading{T<:Real} <: BoundaryGrading                          # CFIE_kress_corners equivalent (one curve, parametric corner clustering)
    kressq::Int
    min_t_spacing::T
end
CornerGrading(; kressq::Int=2, min_t_spacing::T=1e-12) where {T<:Real}
# GlobalCornerGrading (defined in §4.3) covers CFIE_kress_global_corners too

struct CombinedFieldIntegralEquationSolver{T<:Real,G<:BoundaryGrading,Sy<:Union{AbsSymmetry,Nothing}} <: SweepBIMSolver
    pts_scaling_factor::Vector{T}
    min_pts::Int64
    grading::G
    symmetry::Sy
    eps::T
end

CombinedFieldIntegralEquationSolver(pts_scaling_factor::Union{T,Vector{T}};
                                     min_pts::Int=200, grading::BoundaryGrading=SmoothPeriodicGrading(),
                                     symmetry::Union{Nothing,AbsSymmetry}=nothing, eps::T=T(1e-15)) where {T<:Real}

evaluate_points(solver::CombinedFieldIntegralEquationSolver, billiard::Bi, k) where {Bi<:AbsBilliard} -> pts::BoundaryPoints
boundary_matrix_size(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints) -> N::Int
construct_matrices(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> A::Matrix{Complex{T}}   # A(k) = I - (D(k) + i k S(k))
solve(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true, use_krylov::Bool=true) -> t::Real
solve_vect(solver::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> (t::Real, x::Vector{Complex{T}})
```

The `1/2πi` coupling coefficient in `A(k)=I-(D(k)+ikS(k))` is fixed at `ik`, not
a tunable field (matches develop; no evidence in the prototype of a
user-adjustable coupling constant).

### 4.5 `CompositeBIMSolver` (new — generalizes `CFIE_kress_composite_solver`)

```julia
struct CompositeBIMSolver{T<:Real,CS<:Tuple,Sy<:Union{AbsSymmetry,Nothing}} <: SweepBIMSolver
    component_solvers::CS     # one DoubleLayerPotentialSolver or CombinedFieldIntegralEquationSolver per connected boundary component
    symmetry::Sy              # must match every component solver's symmetry
end

CompositeBIMSolver(component_solvers::Vararg{SweepBIMSolver})

evaluate_points(solver::CompositeBIMSolver, billiard::Bi, k) where {Bi<:AbsBilliard} -> pts::BoundaryPoints  # component 1 = outer boundary; components 2:end = holes, orientation-reversed
construct_matrices(solver::CompositeBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> A::Matrix{Complex{T}}
solve(solver::CompositeBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> t::Real
solve_vect(solver::CompositeBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> (t::Real, x::Vector{Complex{T}})
```

Validated at construction: `length(component_solvers) == number of connected
components of `billiard.full_boundary``  is checked lazily inside
`evaluate_points` (billiard isn't known at construction time, per decision §2.1),
raising an `ArgumentError` on mismatch.

### 4.6 `BeynSolver` (fills existing placeholder, accelerated family)

```julia
struct BeynSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K              # DoubleLayerPotentialSolver / CombinedFieldIntegralEquationSolver / CompositeBIMSolver
    m::Int                 # target eigenvalue count per contour window (Weyl-window planning)
    nq::Int                 # contour quadrature node count
    r::Int                  # random probing rank
    svd_tol::T               # singular-value cutoff for rank detection
    res_tol::T               # residual threshold for spurious-root rejection
    auto_discard_spurious::Bool
    use_chebyshev::Bool      # opt in to Chebyshev-accelerated kernel evaluation
    n_panels_h::Int
    M_h::Int
    n_panels_j::Int
    M_j::Int
end

BeynSolver(kernel::K; m::Int=10, nq::Int=48, r::Int=48, svd_tol::T=T(1e-12), res_tol::T=T(1e-9),
           auto_discard_spurious::Bool=true, use_chebyshev::Bool=true,
           n_panels_h::Int=15000, M_h::Int=5, n_panels_j::Int=10000, M_j::Int=5) where {T<:Real,K<:SweepBIMSolver}

construct_matrices(solver::BeynSolver, pts::BoundaryPoints, k0::Complex, R::Real; rng=MersenneTwister(0), multithreaded::Bool=true) -> (A0::Matrix, A1::Matrix)
solve(solver::BeynSolver, pts::BoundaryPoints, k0::Complex, dk::Real; multithreaded::Bool=true) -> (ks::Vector, ts::Vector)
solve_vectors(solver::BeynSolver, pts::BoundaryPoints, k0::Complex, dk::Real; multithreaded::Bool=true) -> (ks::Vector, ts::Vector, X::Matrix)
solve_wavenumber(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard} -> (k0::Real, t0::Real)
solve_spectrum(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard} -> (ks::Vector, ts::Vector)
```

`weyl_window_width`/`plan_weyl_windows`/`beyn_disks_from_windows` remain
free functions (they operate on a billiard + `m`, not on solver state) used
internally by a higher-level `k_sweep`-style multi-window driver, added once
`solve_spectrum` needs to cover a wide range rather than one window.

### 4.7 `ExpandedBIMSolver` (fills existing placeholder, accelerated family)

```julia
struct ExpandedBIMSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K
    use_chebyshev::Bool
    n_panels_h::Int
    M_h::Int
    n_panels_j::Int
    M_j::Int
end

ExpandedBIMSolver(kernel::K; use_chebyshev::Bool=true,
                   n_panels_h::Int=15000, M_h::Int=5, n_panels_j::Int=10000, M_j::Int=5) where {K<:SweepBIMSolver}

construct_matrices(solver::ExpandedBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> (A::Matrix, dA::Matrix, ddA::Matrix)   # A, A', A'' at k
solve(solver::ExpandedBIMSolver, pts::BoundaryPoints, k; multithreaded::Bool=true) -> (k_corr::Real, t0::Real)     # 2nd-order local root correction
solve_wavenumber(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard} -> (k0::Real, t0::Real)
solve_spectrum(solver::ExpandedBIMSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard} -> (ks::Vector, ts::Vector)
```

---

## 5. Summary table of every solver to migrate

| Family | Concrete type(s) planned in main | Abstract branch |
|---|---|---|
| Basis, sweep | `DecompositionMethodSolver` (+`rellich_origin`) | `SweepBasisSolver` |
| Basis, sweep | `ParticularSolutionsMethod` (new) | `SweepBasisSolver` |
| Basis, accelerated | `VerginiSaracenoSolver` (already done) | `AcceleratedBasisSolver` |
| BIM, sweep | `DoubleLayerPotentialSolver{T,G,Sy}` (`G ∈ {SmoothPeriodicGrading, GlobalCornerGrading}`) | `SweepBIMSolver` |
| BIM, sweep | `CombinedFieldIntegralEquationSolver{T,G,Sy}` (`G ∈ {SmoothPeriodicGrading, CornerGrading, GlobalCornerGrading}`) | `SweepBIMSolver` |
| BIM, sweep | `CompositeBIMSolver{T,CS,Sy}` | `SweepBIMSolver` |
| BIM, accelerated | `BeynSolver{T,K}` | `AcceleratedBIMSolver` |
| BIM, accelerated | `ExpandedBIMSolver{T,K}` | `AcceleratedBIMSolver` |

---

## 6. Prerequisite infrastructure (must land before/with Step 2 implementation)

Not solver-API design, but blocking dependencies for the bodies of the
functions signed off in §4 — flagged now so Step 2 isn't blindsided:

1. **Kress grading utilities** (`kress_R!`, `kress_R_even!`, `kress_R_odd!`,
   single-/multi-corner grading maps) — port into `BilliardGeometry.jl`
   (quadrature-adjacent) or `QuantumBilliards.jl/utils/` (needs a decision —
   these are pure boundary-quadrature helpers, arguably belong in
   `BilliardGeometry.jl` next to `samplers.jl`).
2. **Symmetry-orbit folding** (`SymmetryOrbitMap`, `symmetry_index_orbits`,
   `fundamental_size`) — needed by every `symmetry !== nothing` BIM solver;
   `BilliardGeometry.jl` already has `AbsSymmetry`/reflections but not orbit
   maps.
3. **Boundary geometry caches** (`BoundaryGeomCache`, `BoundaryPanelArrays`,
   `component_lengths`, `component_normals`, `component_offsets`) — reusable
   pairwise-geometry workspace, internal to `construct_matrices`, not part of
   the public struct API in §4.
4. **Chebyshev-accelerated Hankel/Bessel evaluation** — purely a performance
   backend selected via the `use_chebyshev::Bool` fields above; safe to stub
   out (`use_chebyshev=false` path only) in an initial implementation pass and
   add later without touching the struct API.
5. **`AbsState` branch for BIM eigenstates** — needed once
   `compute_eigenstate`/`wavefunction` are migrated for BIM solvers (see §3
   note); tracked here, designed separately.

---

## 7. Suggested migration order (Step 2+, not part of this plan)

1. `abstracttypes.jl` docstrings for `AbsBIMSolver`/`SweepBIMSolver`/`AcceleratedBIMSolver` (§3).
2. `DecompositionMethodSolver` + `rellich_origin` (small, additive, no new abstractions).
3. `ParticularSolutionsMethod` (reuses 100% of existing `SweepBasisSolver` infra).
4. Kress/symmetry-orbit/geometry-cache prerequisites (§6.1–§6.3).
5. `DoubleLayerPotentialSolver` with `SmoothPeriodicGrading` only (simplest Kress case), then `GlobalCornerGrading`.
6. `CombinedFieldIntegralEquationSolver` (`SmoothPeriodicGrading` → `CornerGrading` → `GlobalCornerGrading`).
7. `CompositeBIMSolver`.
8. `BeynSolver`, then `ExpandedBIMSolver` (both depend on 5–7 being solvable already).
9. Chebyshev acceleration backend (§6.4), opt-in, last.

---

## 8. Decisions confirmed with the user

1. **Confirmed.** The `BoundaryGrading` trait design (§2.2/§4.3) is used as
   specified, collapsing develop's one-struct-per-grading naming into the two
   family structs `DoubleLayerPotentialSolver`/`CombinedFieldIntegralEquationSolver`
   parameterized by `grading`.
2. **Confirmed.** The plain uncorrected `BoundaryIntegralMethod` is dropped
   entirely; its behavior is already covered by `DoubleLayerPotentialSolver`
   with `grading = SmoothPeriodicGrading()` (see §4.3 note).
3. **Confirmed.** `BeynSolver`/`ExpandedBIMSolver` wrap a `kernel::SweepBIMSolver`
   field (§2.3) rather than staying a `Union`-based trait applied to standalone
   keyword arguments.
4. **Confirmed.** Symmetry-orbit folding lives in `BilliardGeometry.jl` as a
   genuine symmetry *representation* type, [`SymmetryOrbitMap`](@ref) (see
   §9.2), alongside the existing `AbsSymmetry`/reflection types. Kress-grading
   quadrature utilities (`kress_R!` and friends) follow the same placement
   recommendation (`BilliardGeometry.jl`, quadrature-adjacent) but have not yet
   been scaffolded — tracked as a remaining prerequisite in §6.1.

---

## 9. Scaffolding implemented (this pass)

The following are now real, loadable Julia code (structs, constructors and
fully-typed method signatures); every method body raises
`error("... is not yet implemented (API scaffold only, ...)")` until Step 2
fills it in. Both `QuantumBilliards.jl` and `BilliardGeometry.jl` were
verified to precompile and load successfully, and the new types were smoke
tested (construction, dispatch, and stub-error behavior) end-to-end against a
real `StadiumBilliard`.

1. **`abstracttypes.jl`**: docstrings added for `AbsBIMSolver`, `SweepBIMSolver`,
   `AcceleratedBIMSolver` (§3), matching the style of `AbsBasisSolver`/
   `SweepBasisSolver`/`AcceleratedBasisSolver`.
2. **[`solvers/sweepmethods/boundarygrading.jl`](../../.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/boundarygrading.jl)**
   (new file): `BoundaryGrading` abstract type, `SmoothPeriodicGrading`,
   `CornerGrading{T}`, `GlobalCornerGrading{T}`, plus the internal
   `_bim_numeric_type` dispatch hook used by `BeynSolver`/`ExpandedBIMSolver`
   to infer their numeric type from a wrapped kernel.
3. **`solvers/sweepmethods/dlp.jl`**: `DoubleLayerPotentialSolver{T,G,Sy}`
   struct + keyword constructor + stub `evaluate_points`,
   `boundary_matrix_size`, `construct_matrices`, `solve`, `solve_vect`.
4. **`solvers/sweepmethods/cfie.jl`**: `CombinedFieldIntegralEquationSolver{T,G,Sy}`,
   same method set as DLP.
5. **[`solvers/sweepmethods/compositebim.jl`](../../.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl)**
   (new file): `CompositeBIMSolver{T,CS,Sy}` + `Vararg` constructor (validates
   shared `symmetry` across components) + stub `evaluate_points`,
   `construct_matrices`, `solve`, `solve_vect`.
6. **[`solvers/sweepmethods/particularsolutions.jl`](../../.julia/dev/QuantumBilliards.jl/src/solvers/sweepmethods/particularsolutions.jl)**
   (new file): `ParticularSolutionsMethod{T} <: SweepBasisSolver` + both
   constructors + stub `evaluate_points`, `construct_matrices`, `solve_full`,
   `solve_with_rank_reduction`, `solve`, `solve_vect`. `solve_wavenumber`/
   `k_sweep` are already fully functional (inherited for free from the shared
   `SweepBasisSolver` generics) once the stubs above are implemented.
7. **`solvers/sweepmethods/decompositionmethod.jl`**: added the `rellich_origin::SVector{2,T}`
   field (non-breaking — both constructors default it to the origin), and
   `evaluate_points` now computes `rn` relative to `solver.rellich_origin`.
8. **`solvers/sweepmethods/sweepmethods.jl`**: added the new file includes,
   plus fully-implemented (non-stub) `solve_wavenumber(::SweepBIMSolver, ...)`
   and `k_sweep(::SweepBIMSolver, ...)` generics, mirroring the basis-solver
   versions but without a `basis`/`resize_basis` step.
9. **`solvers/acceleratedmethods/beyn.jl`**: `BeynSolver{T,K}` struct wrapping
   `kernel::K<:SweepBIMSolver` + keyword constructor (infers `T` via
   `_bim_numeric_type`) + stub `construct_matrices`, `solve`, `solve_vectors`,
   `solve_wavenumber`, `solve_spectrum`.
10. **`solvers/acceleratedmethods/ebim.jl`**: renamed the placeholder struct to
    `ExpandedBIMSolver{T,K}` (fixing a pre-existing bug where the module file
    exported a non-existent `EBIMSolver` name) wrapping `kernel::K<:SweepBIMSolver`
    + keyword constructor + stub `construct_matrices`, `solve`,
    `solve_wavenumber`, `solve_spectrum`.
11. **`solvers/acceleratedmethods/acceleratedmethods.jl`**: added a genuinely
    implemented (non-stub) `evaluate_points(::AcceleratedBIMSolver, ...)` that
    delegates to `solver.kernel`, since the boundary discretization is always
    fully determined by the wrapped kernel regardless of the accelerated
    root-finding strategy on top of it.
12. **`QuantumBilliards.jl`** (module file): added includes/exports for every
    new name above, and fixed the export list (`ExpandedBIMSolver` instead of
    the stale `EBIMSolver`; added `AbsBIMSolver`, `SweepBIMSolver`,
    `AcceleratedBIMSolver`, `ParticularSolutionsMethod`, `BoundaryGrading` and
    its three concrete subtypes, `CompositeBIMSolver`, `boundary_matrix_size`,
    `solve_vectors`).
13. **[`BilliardGeometry.jl/src/geometry/symmetryorbits.jl`](../../.julia/dev/BilliardGeometry.jl/src/geometry/symmetryorbits.jl)**
    (new file): `SymmetryOrbitMap{T}` — the symmetry *representation* type
    requested in decision §8.4, storing `fundamental_indices`, `orbit_of`,
    `phase`, `full_size`, `fundamental_size` — plus `fundamental_size(...)`,
    `Base.length(...)`, and the currently-methodless generic function stubs
    `symmetry_node_multiple`/`symmetry_index_orbits` (declared via
    `function f end` so every concrete `AbsSymmetry` can add a method in Step 2).
    Included from `geometry/geometry.jl` right after `symmetry.jl`, and
    exported from `BilliardGeometry.jl`.

### Remaining before Step 2 can start filling in bodies

* Concrete `symmetry_node_multiple`/`symmetry_index_orbits` methods for
  `XAxisReflection`/`YAxisReflection`/`XYAxisReflection`/`NFoldRotation` (§6.2).
* Kress grading utilities (`kress_R!`, `kress_R_even!`, `kress_R_odd!`,
  single-/multi-corner grading maps) — still need a `BilliardGeometry.jl`
  home and a scaffold pass of their own (§6.1, not yet scaffolded).
* Boundary geometry caches (`BoundaryGeomCache`, `BoundaryPanelArrays`, …) (§6.3).
* Chebyshev-accelerated Hankel/Bessel evaluation backend (§6.4) — the
  `use_chebyshev::Bool` fields are already scaffolded on `BeynSolver`/
  `ExpandedBIMSolver`, so this can land later without touching the struct API.
* `AbsState` branch for BIM eigenstates (§6.5, §3 note) — needed once
  `compute_eigenstate`/`wavefunction` are migrated for BIM solvers.
