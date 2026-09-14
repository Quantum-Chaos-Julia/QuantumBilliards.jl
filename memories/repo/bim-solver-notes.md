
# BIM solver (DLP/CFIE/CompositeBIM) migration notes

## Step 13 (CompositeBIMSolver annulus verification) — DONE 2026-09-13

Fixed the root cause (see correction note below) and validated
`CompositeBIMSolver` against the annulus's analytic spectrum. Two changes:

1. **`BilliardGeometry.jl/src/geometry/segments/circlesegment.jl`**: both
   `CircleSegment` convenience constructors computed `.length = R*arc_angle`
   (signed) instead of `R*abs(arc_angle)`. A curve with negative `arc_angle`
   (produced by `_reverse_curve`) got a negative `.length`, which corrupted
   `BilliardGeometry.component_lengths`/`_global_t_to_segment_u`'s
   `s >= Ltot` guard and collapsed the entire reversed curve's
   discretization to one degenerate point → NaN in the assembled Fredholm
   matrix → `KrylovKit`/LAPACK `ArgumentError` downstream. Fixed to
   `R*abs(arc_angle)`.
2. **`QuantumBilliards.jl/src/solvers/sweepmethods/compositebim.jl`**:
   replaced `CompositeBIMSolver.evaluate_points`'s positional hole
   assumption (`a == 1 ? outer : hole`, i.e. reverse every component after
   the first) with a new `_group_is_hole(group)` helper that reads each
   component's curves' own `orientation` field (`-1` ⇒ hole, `1` ⇒ outer),
   the same marker `MultiplyConnectedDomain`/`AnnularBilliard` already use
   for `is_inside` semantics. This is a deliberate design convention (user
   request): "reversed orientation means the curve is treated as a hole",
   made explicit and enforced (mixed-orientation components error) rather
   than inferred from component ordering. Verified this also fixes a latent
   incorrectness for genuinely disjoint independent (non-hole) boundaries —
   a new `TwoDiskBilliard` two-independent-circles fixture confirmed neither
   component gets reversed now (both `orientation=1`), whereas the old code
   would have wrongly reversed the second one.

**Verification**: `CompositeBIMSolver(DoubleLayerPotentialSolver(8.0),
DoubleLayerPotentialSolver(8.0))` on `AnnularBilliard(2.0,1.0)` now produces
a finite (`NaN`-free) `400×400` matrix, and its tension sweep near the
annulus's analytic `m=0` root (bisected: `k=3.1230309195852897` from
`J₀(k·1)Y₀(k·2) - J₀(k·2)Y₀(k·1) = 0`) shows a clean minimum of
`≈9.6e-12` exactly at that `k` — essentially exact agreement. No regression
in single-component cases (`CIRCLE_COMPOSITE_DLP` etc. from
`spectrum_cases.jl`, all `orientation=1`, tensions numerically unchanged).

## Correction (2026-09-13): CompositeBIMSolver "single-component" bug claim was wrong

The Step 11 note below claims the `KrylovKit.svdsolve` `ArgumentError`
"reproduced even with a single already-working `DoubleLayerPotentialSolver`
component ... on `CircleBilliard`". Fresh REPL reproduction (see
`QuantumBilliardsTests/scratchpad/migration-plan/13-compositebim-annulus-verification.md`)
found NO error for that exact case (`CompositeBIMSolver(DoubleLayerPotentialSolver(...))`
on `BilliardGeometry.CircleBilliard(1.0)`, single component). The bug is
real but far narrower than originally reported: it only triggers when
`CompositeBIMSolver.evaluate_points` reverses a **hole** curve
(`groups[2:end]`, i.e. genuinely multiply-connected billiards like
`AnnularBilliard`) — `BilliardGeometry._reverse_curve(c::CircleSegment)`
applied standalone to a positive-`arc_angle` curve yields a NEGATIVE
`.length` field (the constructor computes `L = R*arc_angle` without `abs`),
which corrupts `component_lengths`/`_global_t_to_segment_u`'s `s >= Ltot`
guard and collapses the entire reversed hole's discretization to one
degenerate point → NaN in the assembled matrix → the `KrylovKit`/LAPACK
`ArgumentError` is a downstream symptom of that NaN, not a Krylov call-site
bug. `solve`/`solve_vect` in `compositebim.jl` need no code change; the fix
belongs in `BilliardGeometry.jl/src/geometry/segments/circlesegment.jl`
(`L = R*abs(arc_angle)`). See Step 13's plan file for the full isolation.
`testing-framework-plan.md`/`spectrum_cases.jl`'s later, independent
re-verification (single-component composite works fine on several
symmetry-free `PolarBilliard` fixtures) is the correct characterization —
treat the "single-component" claim below as superseded.

## Step 11 (billiards/domains from BilliardGeometry-develop) findings (2026-09-10)

Implemented `CircleBilliard`, `EllipseBilliard`, `RectangleBilliard`,
`PolygonBilliard`, `StarBilliard`, `C3Billiard`, `ProsenBilliard`,
`AnnularBilliard` in `BilliardGeometry.jl/src/geometry/billiards/`. All use
main's existing 2-field `AbsBilliard` pattern (`fundamental_domain`,
`symmetries`, no stored `full_boundary` field — `BilliardGeometry.jl`'s
generic `full_boundary(billiard)` function already reconstructs it), NOT
`BilliardGeometry-develop`'s newer 3-field `full_boundary`-as-stored-field
rewrite of these same billiards (that rewrite is architecturally incompatible
with main's Step-4 `full_boundary(billiard)`-as-function decision; do not
copy it verbatim in future porting work).

**Real bug found and fixed**: `full_boundary(billiard)` for `Star`/`C3`/
rotational-sector billiards must use ONLY `Cn_symmetry(n)` (n-1 rotations) to
tile the `2π/n` fundamental sector back to `2π` — `-develop`'s own
`StarBilliard`/`CircleStarBilliard` additionally append `YAxisReflection()`/
`XYAxisReflection()` when `n` is even, and `-develop`'s `ProsenBilliard`
appends `Cn_symmetry(4)` on top of `D2_symmetry`. Since the fundamental
sector's rotations alone already tile the complete `2π` boundary with zero
gap, any additional symmetry appended on top necessarily maps the sector onto
an already-covered region, causing `full_boundary` to emit duplicate/
overlapping curves. Fixed by using ONLY `Cn_symmetry(n)` for rotational-cut
billiards (`Star`/`C3`, walls = `QuantumSolverIgnore()`) and ONLY
`D2_symmetry` for reflection-walled quadrant billiards (`Prosen`, walls =
`ReflectionSymmetry(...,4)`) — never both together. Verified: `full_boundary`
curve count/total length for Star/C3/Prosen all check out (e.g. Prosen: 4
curves, total length ≈ 6.528).

**`connect_curves` multiply-connected bug**: see
`BilliardGeometry.jl/memories/repo/known-bugs.md` — fixed as a prerequisite
for `AnnularBilliard` (`get_boundary_curves`/`full_boundary` were silently
dropping every closed component after the first for any billiard with a
disjoint hole).

**`AnnularBilliard` design note**: uses a single `SimpleDomain` (NOT a
`CompositeDomain`) holding both the outer circle (`orientation=1`) and inner
circle (`orientation=-1`) as its `.boundary`. `is_inside` for a
`CompositeDomain` is a union (`any(...)` over subdomains) — correct for
Stadium's endcap-union shape, but WRONG for a domain-with-a-hole (needs the
intersection "inside outer AND outside inner", which is exactly what a
single `SimpleDomain`'s `all(...)`-over-curves check gives once the inner
curve's `orientation=-1`). No symmetry reduction is applied (kept simple);
outer/inner curves are tagged `domain_id=1`/`domain_id=2` respectively so
`CompositeBIMSolver`'s `_group_boundary_by_domain_id` splits them correctly.

**Validated**: `CircleBilliard` ground state matches the exact first zero of
`J₀` (`k=2.4048255919...` vs. analytic `2.4048255577...`) via
`VerginiSaracenoSolver`+`RealPlaneWaves(sym_x=1,sym_y=1)` — chose a basis
solver over `DoubleLayerPotentialSolver` for this check because of the two
pre-existing (NOT introduced by Step 11) solver bugs below, both reproduced
identically with pre-existing billiards (`StadiumBilliard`/pre-Step-11 code),
confirming they are out of scope for Step 11:
- `DoubleLayerPotentialSolver`'s symmetry-reduced `construct_matrices` fails
  its own internal algebraic-consistency test
  (`test/solvertests.jl:322`, "Double Layer Potential - Stadium
  (YAxisReflection) - symmetry-reduced matrix consistency") — pre-existing,
  reproduces identically with `git stash` of all Step 11 changes.
- `CompositeBIMSolver.solve_vect` raises `ArgumentError("operator and its
  adjoint are not compatible")` from `KrylovKit.svdsolve`, even wrapping a
  single already-working `DoubleLayerPotentialSolver` component on an
  existing simply-connected billiard (`CircleBilliard`, not just the new
  `AnnularBilliard`) — `evaluate_points`/`construct_matrices` work correctly
  (verified: `AnnularBilliard(2.0,1.0)` gives 400 points, correct 400×400
  matrix, correct `domain_id` grouping), only `solve_vect`'s Krylov call
  fails. This matches the migration plan's own acknowledged "Known
  limitation": `CompositeBIMSolver`'s full numerical `solve_vect` path was
  never actually exercised end-to-end before Step 11 (Step 7 only did
  structural checks). Flagged as a follow-up, not fixed here (BIM-solver
  internals, out of scope for a geometry-porting step).

**Deferred (not implemented this step, all consistent with the plan's own
scoping)**: `SinaiBilliard`, `RectangleWithinRectangleBilliard`/
`DiagonalRectangleWithinRectangleBilliard` (CAFB-specific fields, unrelated
feature), `CircleStarBilliard`, `PentagonBilliard` — `AnnularBilliard` alone
already satisfies the "one multiply-connected fixture" goal; the others are
redundant for that purpose. Gap 2 from the Step 10.9 audit (multicomponent
`symmetry_index_orbits` for a *symmetric* multiply-connected billiard) is
still open — `AnnularBilliard` deliberately uses no symmetry reduction to
side-step it; a future symmetric multiply-connected fixture would still need
Gap 2 fixed first.



Chebyshev mid-range cutoff gap — solvers/chebyshev/bessels.jl: scalar eval_h/eval_h_multi_ks! were missing the |k·r| < 0.2 direct-evaluation fallback that -develop applies. This mattered concretely for Beyn's multi-k contour batches, where rmin is floored using the largest |k| across nodes — smaller-|k| nodes could then evaluate the Chebyshev panel fit inside the numerically sensitive near-origin band. Fixed by adding the same cutoff branch; verified the fix recovers SpecialFunctions.besselh to ~1.6e-16 in a scenario that previously would have used the raw (less accurate) panel fit.

ExpandedBIMSolver NaN bug (substep 7, root cause) — not an eigenvector-pairing issue as hypothesized, but: a plain DoubleLayerPotentialSolver kernel's A'(k) is genuinely rank-deficient (~half its 200 singular values below 1e-6·max, cond ~5e17, vs. CFIE's well-conditioned cond ~400 — CFIE's derivative has an extra full-rank i·S(k) term DLP lacks). This makes LinearAlgebra.eigen(A,dA) legitimately return a couple of NaN eigenvalues. The real bug: argmin(abs.(λ)) in ebim.jl's solve is NaN-poisoned (Julia's min-based reduction can lock onto a NaN regardless of position), even though a valid tiny true root was always present elsewhere in the array. Fixed with a new _argmin_finite helper used for both the primal and adjoint eigenvalue selection. Verified: ExpandedBIMSolver(DoubleLayerPotentialSolver(...)) now correctly returns k=6.065090994323157 (matches Beyn to ~5e-9) instead of NaN.

Docstring inconsistency — states/wavefunctions.jl's ϕ_slp note now correctly states the Chebyshev SLP-wavefunction path was deliberately scoped out of Step 10 (confirmed via -develop that SLPWavefunctionChebPlan is a real, not aspirational, capability).

Tests added: test/chebyshevtests.jl (6 testsets, 24 assertions, wired into runtests.jl) covering the cutoff-fix unit test, the NaN regression, and Beyn/EBIM DLP+CFIE Chebyshev-vs-direct cross-checks. Full suite (30 testsets) passes.

Verified but not code-changed: Beyn DLP direct-vs-Chebyshev speedup on a k=60 triangle (N=2336) is ~3.1x with results agreeing to 3.5e-16. Also found that compute_spectrum(::ExpandedBIMSolver,...) gives messy/poorly-clustered results on wide windows with default keywords — confirmed this reproduces identically with use_chebyshev=false, so it's a pre-existing Step 9.5 tuning issue, unrelated to Step 10; flagged for a separate look if needed.

## Step 10.5 (Chebyshev gap analysis) — ground-truth correction + implementation (2026-09-10)

**Correction to the "Chebyshev mid-range cutoff gap" note above**: re-audited
in Step 10.5 and found the claim as stated is wrong — main's scalar `eval_h`
(bessels.jl) already HAS the `pidx==0 || abs(z)<hankel_z_chebyshev_cutoff`
guard (confirmed by reading the current source, not by re-deriving). Neither
repo's DLP/CFIE derivative-kernel hot loop was ever missing the guard either.
No action item follows from the original wording; superseded by the real
findings below.

Step 10.5 implemented, in priority order:
* **Finding 1 (padding)**: `_cheb_geom_rminmax` (optimalpanelization.jl) now
  takes `pad::Tuple{Float64,Float64}=(0.95,1.05)` and widens the raw observed
  pairwise-distance extrema by ±5% before the near-zero floor, matching
  `-develop`'s `build_dlp_kress_block_cache` default — guards against
  silent out-of-range extrapolation (`panel_t` clamps with no error).
* **Finding 2 (Beyn single-pass multi-k assembly)**: added
  `_dlp_fredholm_full_multi_k_cheb!`/`_dlp_fredholm_reduced_multi_k_cheb!`
  (chebyshev/dlp.jl) and the CFIE analogues (chebyshev/cfie.jl), each doing
  ONE O(N²) pairwise-geometry pass for ALL contour nodes at once (using the
  already-present-but-previously-unused `h1_j1_multi_ks_at_r!`/
  `h0_h1_j0_j1_multi_ks_at_r!` combinators), instead of `beyn.jl`'s old
  `_construct_matrices_multi_k_cheb` calling the per-node value-only function
  `nq` times. Verified numerically equivalent to the old per-node path (and
  to `use_chebyshev=false`) to ~1e-15 for both DLP/CFIE, full and
  symmetry-reduced (Stadium+YAxisReflection), via manual sanity checks; full
  existing test suite (`test/runtests.jl`) still passes.
  * **Real bug found while wiring this up**: `eval_j_multi_ks!`
    (bessels.jl) — dead code since Step 9.5/10, first exercised by this
    change — was missing the `pidx==0` near-zero fallback that the scalar
    `eval_j` already has (no `r` parameter at all in the old signature).
    This caused an out-of-bounds `panels[0]` access under `@inbounds`
    (silent heap corruption → segfault on the next read), triggered
    whenever the near-zero floor in `_cheb_geom_rminmax` exceeds the padded
    observed `rmin` (common with Step A's padding pulling `rmin` down).
    Fixed by adding an `r::Float64` parameter and the same
    `SpecialFunctions.besselj` fallback branch as the scalar version;
    threaded the extra argument through its two callers
    (`h1_j1_multi_ks_at_r!`, `h0_h1_j0_j1_multi_ks_at_r!`).
  * **Thread-buffer-sizing gotcha (new, worth remembering)**: sizing
    per-thread scratch buffers with `Threads.nthreads()` is UNSAFE in this
    Julia version/config — `Threads.threadid()` can return values up to
    `Threads.nthreads(:default)+Threads.nthreads(:interactive)` (the master
    thread defaults to the `:interactive` pool since Julia 1.9), which
    exceeded `Threads.nthreads()` in this dev environment (`nthreads()==1`
    but observed `threadid()==2` inside `Threads.@threads`). Added
    `_cheb_nthreads_buf()` (chebyshev/core.jl) computing the safe upper
    bound; use it (not bare `Threads.nthreads()`) for any new
    `[... for _ in 1:N]` per-thread buffer array in this codebase.
* **Finding 4 (SLP wavefunction Chebyshev)**: added
  `SLPWavefunctionChebPlan`/`plan_slp_wavefunction`/`_eval_y0_slp_cheb`
  (chebyshev/bessels.jl) — a thin wrapper around the existing
  `ChebHankelPlanH(ν=0,κ=1)` machinery (`Y₀(kr)=Im(H₀^{(1)}(kr))`); no
  separate CFIE wavefunction plan was needed since main's `ϕ_slp`/
  `wavefunction(::BIMEigenstate)` reconstruction is already unified across
  every `SweepBIMSolver` (unlike `-develop`, which had a separate DLP vs.
  CFIE wavefunction-reconstruction kernel). `ϕ_slp(...; use_chebyshev=true,
  cheb=<plan>)` now works instead of erroring; `wavefunction(state::
  BIMEigenstate)` gained `use_chebyshev`/`cheb_npanels`/`cheb_M` keywords and
  builds one plan per call (not per point). Verified: relative error
  ~6.9e-8 vs. `use_chebyshev=false` on the Veech-triangle DLP ground state
  with default `cheb_npanels=4000,M=6` (no auto-tuner was ported for this
  path — deliberately out of scope, keep it simple; increase
  `cheb_npanels`/`cheb_M` manually if tighter accuracy is needed).
* **Findings 3 (k-independent panel cache) and 5 (combined H0/H1 scalar
  evaluator)** were explicitly skipped per the user's "do not waste too much
  time on profiling" direction (both were profiling-gated in the plan).

Full `test/runtests.jl` (all pre-existing testsets, including
`chebyshevtests.jl`) passes after these changes. No new test file was
written by the implementing agent — see the plan's testing policy; invoke
the Julia Test Writer subagent for regression coverage of the new
single-pass/reduced Chebyshev paths and the SLP wavefunction Chebyshev path.



- Step 3 (`DoubleLayerPotentialSolver`, solvers/sweepmethods/dlp.jl) is
  fully implemented and numerically verified (not a stub) as of 2026-09-07.
  Do not re-implement; if bodies look like stubs again, something regressed.
- `_global_t_to_segment_u`/`_eval_composite_geom_global_t` live in
  BilliardGeometry.jl/src/geometry/boundarycomponents.jl (ported, unexported).
- `BIMEigenstate{K,T,S,Bi}` struct has `k::K`/`vec::Vector{K}` sharing the
  SAME type param K. Since BIM boundary densities are complex, `k` must be
  promoted to `Complex{T}` in the `BIMEigenstate(k,vec,ten,solver,billiard)`
  constructor (`kK = eltype(vec)(k)`), and `eps = set_precision(real(vec[1]))`
  (NOT `set_precision(vec[1])`) so `eps::T` matches `ten::T` (real). Otherwise
  you get a MethodError (eps ends up Complex, ten stays real -> no match).
- `KrylovKit` is a genuine dependency of both QuantumBilliards.jl and
  QuantumBilliards-develop (confirmed in both Project.toml `[deps]`), and
  -develop's own `DLP_kress` `solve`/`solve_vect` do use Krylov nullspace
  methods (`smallest_nullvec_krylov!`, `@svd_or_det_solve` dispatching on
  `use_krylov`) — so `dlp.jl`'s `KrylovKit.svdsolve(A,1,:SR)` in `solve`/
  `solve_vect` is a justified match, not a new/guessed dependency.

## Important gotcha: `get_boundary_curves` is NOT the full physical boundary
  for every billiard fixture — it filters to only `SpecularReflection`-typed
  curves in the `fundamental_domain`, dropping `Transparent`/
  `ReflectionSymmetry`/`QuantumSolverIgnore` curves. This means:
  - `StadiumBilliard(half_width)` (`BilliardGeometry.jl/src/geometry/billiards/stadium.jl`)
    is built with a D2-symmetry-reduced quarter fundamental domain (one
    `CircleSegment` + one `LineSegment`, total perimeter ≈ quarter of the
    real stadium). `get_boundary_curves(StadiumBilliard(0.5))` returns just
    that quarter, artificially closed — closing that loop creates a FAKE
    90°-angle "corner" at the seam (confirmed via `_component_corner_locations`
    returning `[0.0]` with junction angle ≈ π/2). This is fine for
    `RealPlaneWaves`-based basis solvers (plane waves already satisfy the
    reflection BCs across the omitted symmetry lines) but is **wrong** for
    any BIM solver (DLP/CFIE) — do not use `StadiumBilliard` as a BIM test
    fixture without first building/using its true full un-reduced boundary.
  - `make_triangle_and_basis(gamma, chi; edge_i)` (`utils/billiardutils.jl`)
    sets 2 of 3 triangle edges to `QuantumSolverIgnore()` (only `edge_i` is
    real) — it's built for the Veech corner-adapted basis method, and its
    `get_boundary_curves` returns only ONE edge. Also not usable directly
    for BIM.
  - For a genuine full closed billiard boundary (BIM fixture), use
    `BilliardGeometry.TriangleBilliard(gamma, chi)` **without** the `bcs`
    keyword (default is all 3 edges `SpecularReflection()`, no symmetry) —
    same physical shape/spectrum as `make_triangle_and_basis(gamma,chi)`'s
    triangle (translation doesn't change the spectrum), so it cross-validates
    against `test/solvertests.jl`'s Veech-triangle reference k0's. Confirmed:
    low-k ground state k0≈6.06509 matches VS reference 6.065082959967892 to
    ~8e-6 with `DoubleLayerPotentialSolver(5.0; grading=GlobalCornerGrading())`.
  - `PolarBilliard(coef)` (`r(φ)=1+Σaₙcos+Σbₙsin`, `coef=[b1,a1,b2,a2,...]`)
    with `coef=[0.0,0.0]` gives a genuine full-boundary unit circle (no
    symmetry, no corners) — great `SmoothPeriodicGrading` analytic-eigenvalue
    fixture (Bessel zeros): DLP matched J0/J1 first zeros to ~1e-8/~1e-9.
    `coef=[0,0,0,0.3]` (i.e. `r=1+0.3cos(2φ)`) gives a genuinely D2-symmetric
    FULL boundary (no pre-reduction) — good fixture for verifying
    `solver.symmetry=BilliardGeometry.XAxisReflection()` folding: confirmed
    the symmetric-sector sweep reproduces symmetric minima exactly and
    correctly OMITS antisymmetric ones present in the unrestricted sweep.

## `solve_wavenumber`/`Optim.optimize` gotcha at high k
  At high k (e.g. k~110 on a triangle of modest area), Weyl-law eigenvalue
  spacing can be smaller than a "reasonable-looking" search window `dk=0.1-0.2`.
  `solve_wavenumber`'s bounded `Optim.optimize` can converge to the WRONG
  nearby local tension minimum if two eigenvalues both fall inside the
  window (confirmed: window [109.99,110.19] contains minima at both k≈110.09
  AND k≈110.14, and Brent's method picked the k≈110.14 one first). Not a bug
  in the solver — narrow the window (`dk` small enough to isolate a single
  local minimum, verified via a `k_sweep` scan first) when cross-validating
  a specific reference eigenvalue at high k.

## QBPlotting.jl environment
  `QBPlotting.jl` has no `Manifest.toml` in this workspace (never
  instantiated) — `using QBPlotting` fails on missing `StaticArrays` etc.
  This is pre-existing/unrelated to any BIM work; `get_errors` (static) is
  the only available signal for QBPlotting.jl changes unless someone runs
  `Pkg.instantiate()` there (not done automatically — avoid unless asked,
  it's a broad environment change).

## Step 12 (AbsMultiplyConnectedDomain/MultiplyConnectedDomain) — DONE 2026-09-11
  Implemented per QuantumBilliardsTests/scratchpad/migration-plan/12-multiply-connected-domains.md.
  - BilliardGeometry.jl: new `abstract type AbsMultiplyConnectedDomain <: AbsDomain end`
    (sibling of AbsSimpleDomain/AbsCompositeDomain, exported); new
    `MultiplyConnectedDomain{T}` (geometry/domains/multiplyconnecteddomains.jl,
    boundary/corners/id/genus fields + a convenience constructor taking
    outer curves + Vector{<:Vector{<:AbsCurve}} hole groups, computing
    genus=length(hole groups)); `billiards/circle_with_hole.jl` renamed
    (`git mv`) to `billiards/annular.jl`, `AnnularBilliard`'s
    `fundamental_domain` is now `MultiplyConnectedDomain{T}` instead of bare
    `SimpleDomain{T}` (curves themselves byte-identical - same
    CircleSegment calls/orientation/domain_id). boundarytypes.jl: factored
    the AbsSimpleDomain get_boundary_curves body into a private
    `_connected_physical_curves` helper reused by a new
    `get_boundary_curves(::AbsMultiplyConnectedDomain)` method; added
    `genus(domain)`/`genus(billiard)` (0 for Simple/Composite, `.genus`
    field for MultiplyConnected) and `boundary_components(domain)`/
    `boundary_components(billiard)` (one curve group per connected
    component, outer first; groups AbsMultiplyConnectedDomain's boundary by
    curve domain_id via new private `_group_curves_by_domain_id`), both
    exported.
  - QuantumBilliards.jl: boundarypoints.jl got the matching
    `get_boundary_curves_with_ignored(::AbsMultiplyConnectedDomain)` method
    off a factored `_connected_physical_or_ignored_curves` helper (same
    pattern). compositebim.jl: `_group_boundary_by_domain_id` now just
    delegates to `BilliardGeometry._group_curves_by_domain_id` (no
    duplicate algorithm); `evaluate_points` gained an early
    `genus(domain)+1 == length(component_solvers)` check when
    `billiard.fundamental_domain isa AbsMultiplyConnectedDomain`, and uses
    `boundary_components(billiard)` directly (equivalent to the old
    domain_id-grouping in this case) ONLY when `solver.symmetry === nothing`
    - when symmetry is set, `comp = full_boundary(billiard)` includes
      symmetry images that `boundary_components` (which reads
      `domain.boundary` directly, not `full_boundary`) does NOT know about,
      so the domain_id-grouping fallback on `comp` is still required there
      (and for any non-AbsMultiplyConnectedDomain billiard, e.g. Stadium's
      CompositeDomain, preserving the pre-Step-12 general contract).
  - Verified in REPL (QuantumBilliards.jl project): `AnnularBilliard(2.0,1.0)`
    -> `MultiplyConnectedDomain{Float64}`, `genus==1`,
    `get_boundary_curves` returns 2 curves, `is_inside` correctly classifies
    inside-annulus/inside-hole/outside-outer points, `boundary_components`
    returns `[[outer(domain_id=1)], [inner(domain_id=2)]]`.
    `CompositeBIMSolver(DoubleLayerPotentialSolver(2.0;grading=GlobalCornerGrading()) x2)`
    against `AnnularBilliard(2.0,1.0)` at k=10: `evaluate_points` gives 400
    pts, `boundary_matrix_size==400` (matches Step 11's number exactly, no
    regression). Genus-mismatch check confirmed:
    `CompositeBIMSolver(outer_solver_only)` against the same billiard raises
    a clear `ArgumentError` before ever calling `get_boundary_curves`.
  - Both `BilliardGeometry.jl` and `QuantumBilliards.jl` full test suites
    re-run: BilliardGeometry.jl all green. QuantumBilliards.jl has ONE
    pre-existing failing test ("Double Layer Potential - Stadium
    (YAxisReflection) - symmetry-reduced matrix consistency",
    solvertests.jl:353, off by ~0.05 not 1e-10) — confirmed via `git stash`
    that this fails identically with Step 12's changes fully reverted, so
    it predates this step and is NOT a regression; out of scope here.
  - `circle_with_hole` string still appears in migration-plan `*.md` files
    (historical narrative, per the Step 12 plan explicitly not requiring
    edits there) and in the untouched `BilliardGeometry-develop` reference
    repo - both expected/out of scope.
  - Full numerical `solve`/`solve_vect` verification (the known
    `KrylovKit.svdsolve` `ArgumentError`) remains deferred to Step 13, as
    planned - not touched by this step.
  - Julia Test Writer NOT yet invoked for
    MultiplyConnectedDomain/AnnularBilliard geometry unit tests - still
    pending, per the plan's step 7 (tests & user verification).

## Step 12.5 (`compute_spectrum` API consolidation & unfolding/geometry split) — DONE 2026-09-12

  Implemented per
  QuantumBilliardsTests/scratchpad/migration-plan/12.5-spectral-api-consolidation.md.
  - Moved `area`/`fundamental_area`/`symmetry_reduction_factor`/
    `maximal_symmetry` from `QuantumBilliards.jl/src/spectra/unfolding.jl`
    into new `BilliardGeometry.jl/src/geometry/area.jl` (included after
    `geometry/geometry.jl`, exported: `area, fundamental_area,
    corner_angles`), with no behavior change (verified `area`/
    `fundamental_area` values identical before/after on a Veech-triangle
    fixture). `QuantumBilliards.jl` no longer exports `area`/
    `fundamental_area` itself — they arrive transitively via `using
    BilliardGeometry` (every consumer script already does both `using`s).
  - Added `_component_corner_angles(T, comp; angle_tol)` to
    `BilliardGeometry.jl/src/geometry/boundarycomponents.jl` (interior
    angle `γ = π - τ` at each true corner, `τ` from the existing
    `_junction_angle`) and a new `corner_angles(billiard;
    fundamental=true, angle_tol=1e-8)` accessor in `area.jl` (built from
    `boundary_components(billiard)` when `fundamental=true`, or
    `full_boundary(billiard)` treated as one component when `false`).
    Verified `corner_angles(TriangleBilliard(pi/2,1.5))` reproduces
    `Polygon.angles` exactly — confirms the `π - τ` convention.
  - `QuantumBilliards.jl/src/spectra/unfolding.jl` now contains only
    Weyl-law state-counting functions: kept `corner_correction`/`weyl_law`/
    `k_at_state` (2-method A,L form), added billiard-level `k_at_state`,
    `state_at_k` (2 plain-arg methods + billiard-level), `spectral_density`
    (2 methods), `k_range_for_states` — all newly exported except the
    plain `k_at_state`/`state_at_k`(A,L,...) forms.
  - `QuantumBilliards.jl/src/spectra/spectralutils.jl`: added
    `_finalize_spectrum` (sort + empty-check + `SpectralData` construct,
    right after the `SpectralData` constructor); rescoped
    `compute_spectrum(::AbsBasisSolver, basis, billiard, k1,k2,dk;...)` to
    `AcceleratedBasisSolver` (was dispatching on the whole `AbsBasisSolver`
    hierarchy even though `SweepBasisSolver` has no `solve_spectrum` — live
    bug, now fixed), `dk` is now `Union{Real,Function}` (adaptive step via
    `spectral_density`), and it now **returns `SpectralData`** via
    `_finalize_spectrum` instead of the old bare `(ks,tens,control)` tuple.
    Added its `N1,N2` counterpart (`N_expect` keyword, `dk = k ->
    N_expect/spectral_density(k,billiard)`). Deleted the dead
    `compute_spectrum(::AbsBasisSolver,...,N1,N2,dN)` method (accessed
    nonexistent `billiard.area`/`billiard.length`/`billiard.angles` struct
    fields — would `MethodError` on any real billiard). Added
    `compute_spectrum(solver::BeynSolver/ExpandedBIMSolver, billiard,
    N1::Int, N2::Int; kwargs...)` (each: `k_range_for_states` then the
    existing `k1,k2` method; `fundamental = solver.kernel.symmetry!==
    nothing`, matching the existing convention inside those `k1,k2`
    bodies). Updated the `BeynSolver`/`ExpandedBIMSolver` `k1,k2` methods'
    tails to use `_finalize_spectrum` too (same sort/empty-check/construct
    logic, now shared).
  - **Confirmed deviation from the written plan**: the plan's own "Ends
    with `_finalize_spectrum`" instruction for the basis-solver `k1,k2,dk`
    method means it now returns `SpectralData`, not a tuple — this broke
    two `QuantumBilliardsTests` call sites that destructured the old
    3-tuple (`ks, tens, _ = compute_spectrum(solver, basis, billiard,
    k1,k2,0.1)`):
    `QuantumBilliardsTests/reference/generate_reference_spectra.jl`'s
    `accel_basis_case` and `QuantumBilliardsTests/spectrumtests_template.jl`'s
    "Vergini Saraceno - Circle - k≈100" testset. Both fixed to unwrap
    `data.k`/`data.ten`, matching the pre-existing `accel_bim_case`
    pattern. If you see `ks, tens, _ = compute_spectrum(...)` anywhere
    else in `QuantumBilliardsTests`, it needs the same fix.
  - Verified end-to-end: `compute_spectrum(::VerginiSaracenoSolver, basis,
    CircleBilliard, N1, N2)`, `compute_spectrum(::BeynSolver,
    PolarBilliard, N1, N2)`, `compute_spectrum(::ExpandedBIMSolver,
    PolarBilliard, N1, N2)` all run and return `SpectralData` with
    plausible state counts.
  - Full test suites re-run: `BilliardGeometry.jl` all green.
    `QuantumBilliards.jl` has the same pre-existing "Double Layer Potential
    - Stadium (YAxisReflection)" failure noted under Step 12 above, plus an
    **intermittent, unrelated flake** in "Particular Solutions Method -
    Veech Triangle - Ground State" (`Psi` comparison at `atol=1e-3`,
    max-abs-diff observed ranging ~2e-4 to ~5e-4 across repeated runs of
    the *identical* unmodified code — confirmed via `git stash` + 3
    standalone reruns that it is nondeterministic, likely multithreaded
    reduction-order sensitivity in `compute_psi`, and predates/is
    unrelated to this step). Every other testset (Vergini-Saraceno low/high
    spectrum, DLP/CFIE/Beyn ground states, `boundarypointstests.jl`)
    passes cleanly.
  - Julia Test Writer NOT yet invoked for the four consolidated
    `compute_spectrum` methods — still pending, per the plan's own testing
    policy (left to the user).

