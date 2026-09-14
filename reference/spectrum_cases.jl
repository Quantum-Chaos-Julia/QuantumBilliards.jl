# Shared case table: fixtures + solver builders for every
# billiard/basis/solver combination exercised by BOTH
# reference/generate_reference_spectra.jl (produces reference constants) and
# plottingtests_comprehensive.jl (produces state-test figures). Do not fork
# this list -- both scripts `include` this same file so a case's fixture/
# solver is guaranteed identical whichever consumer runs it.
#
# See scratchpad/testing-framework-plan.md for the full solver/billiard
# matrix and which cells are populated here vs. left as follow-up work.
#
# Case-set framework (see reference/case_builder.jl, reference/solver_naming.jl):
# * Each billiard gets one `BilliardCaseSet` (its fixtures + default k0/dk0),
#   populated with `add_basis_solvers!`/`add_bim_solvers!`/
#   `add_accelerated_solvers!`/`add_composite_bim_solvers!`/`add_custom!`
#   instead of a flat list of hand-written `Case(...)` literals -- every
#   case name is derived deterministically from the solver actually
#   constructed (see `case_name`/`describe`), so it can never drift out of
#   sync, and adding a solver/kernel/grading combination to a billiard is a
#   one-line bulk-adder call/argument instead of N new `Case(...)` literals.
# * Every `Case` carries one ground-state target `k0` (solved with
#   `solve_wavenumber`) plus zero or more extra multi-state spectrum windows
#   in `states` (each a `(suffix, k0, n_target)` tuple), solved with
#   `accelerated_states_near` -- which for a *sweep* solver first upgrades
#   it to its accelerated sibling via `accel_solver_for` (`VerginiSaracenoSolver`
#   for a `SweepBasisSolver`, `BeynSolver` for a `SweepBIMSolver`) and for an
#   already-accelerated solver just uses it directly.
# * `states` defaults (via `default_states`) to a single `("K20", 20.0, 5)`
#   window for sweep solvers or `("K100", 100.0, 10)` window for already
#   -accelerated solvers.
# * `plot::Bool` (default `false` on every `add_*!` adder) marks the subset
#   of cases verified safe to run through `plot_state_tests!` (see
#   plottingtests_comprehensive.jl); most cases are generation-only until
#   smoke-tested for plotting too (see that file's own notes on the
#   `BoundaryPoints`/D2-symmetry plotting bug).

using QuantumBilliards
using BilliardGeometry
using StaticArrays

include(joinpath(@__DIR__, "spectrum_harness.jl"))
include(joinpath(@__DIR__, "case_builder.jl"))

# ---------------------------------------------------------------------------
# Shared billiard/basis fixtures
# ---------------------------------------------------------------------------

const TRIANGLE_N = 5

make_veech_right_triangle_and_basis(n; edge_i=1) = QuantumBilliards.make_veech_right_triangle_and_basis(n; edge_i)
make_veech_right_triangle(n) = QuantumBilliards.make_veech_right_triangle(n)

# --- Circle fixtures ---------------------------------------------------

function circle_basis_fixture()
    billiard = BilliardGeometry.CircleBilliard(1.0)
    sector = symmetry_sector(billiard, BilliardGeometry.XAxisReflection=>1, BilliardGeometry.YAxisReflection=>1)
    basis = RealPlaneWaves(12, billiard, sector)
    return billiard, basis
end
circle_bim_billiard() = BilliardGeometry.PolarBilliard([0.0, 0.0])

# --- Triangle fixtures ---------------------------------------------------

triangle_basis_fixture() = make_veech_right_triangle_and_basis(TRIANGLE_N)
triangle_bim_billiard() = make_veech_right_triangle(TRIANGLE_N)

# --- Ellipse fixtures ----------------------------------------------------
# EllipseBilliard(1.0,0.8) itself is a D2 quadrant fundamental domain (like
# CircleBilliard) -- fine for basis solvers directly (basis built via the
# Step 15/16 `symmetry_sector`/`RealPlaneWaves(dim, billiard, sector)` API),
# but NOT usable as-is for BIM solvers: `get_boundary_curves` only returns
# the quadrant arc, and passing `symmetry=` to trigger `full_boundary` hits
# the same known symmetry-reduced `construct_matrices` bug as
# Stadium+YAxisReflection (verified empirically -- never converges; this is
# unrelated to the Step 16 SymmetrySector/character resolution, which is not
# exercised here). Instead build the full, un-reduced ellipse boundary
# directly as its own single-curve PolarBilliard (solver.symmetry=nothing
# path, `BilliardGeometry.register_symmetries()` for its own empty
# registry), verified to converge to ~1e-15 for every BIM solver.
function ellipse_basis_fixture()
    billiard = BilliardGeometry.EllipseBilliard(1.0, 0.8)
    sector = symmetry_sector(billiard, BilliardGeometry.XAxisReflection=>1, BilliardGeometry.YAxisReflection=>1)
    return billiard, RealPlaneWaves(12, billiard, sector)
end
function ellipse_bim_billiard(a=1.0, b=0.8)
    r(phi) = a*b/sqrt((b*cos(phi))^2 + (a*sin(phi))^2)
    segment = BilliardGeometry.PolarSegment(r; R=max(a,b), center=[0.0, 0.0])
    r0 = BilliardGeometry.curve(segment, 0.0)
    dom = BilliardGeometry.PolarDomain{Float64}([segment], [r0], 1)
    return BilliardGeometry.PolarBilliard{Float64}(dom, BilliardGeometry.register_symmetries())
end

# --- Limacon fixtures ------------------------------------------------------
# LimaconBilliard(0.5) itself IS usable directly for basis solvers (single
# XAxisReflection-symmetric half-domain): built via
# `symmetry_sector(billiard, XAxisReflection=>1)` +
# `RealPlaneWaves(dim, billiard, sector)` -- RealPlaneWaves' own sym_x/sym_y
# swap convention (sym_x<->YAxisReflection/sym_y<->XAxisReflection) is
# resolved internally by that constructor, so callers only ever name the
# axis reflection type. Same BIM limitation as Ellipse -- build the full
# un-reduced closed limacon curve directly (empty
# `BilliardGeometry.register_symmetries()`).
const LIMACON_A = 0.5
function limacon_basis_fixture()
    billiard = BilliardGeometry.LimaconBilliard(LIMACON_A)
    sector = symmetry_sector(billiard, BilliardGeometry.XAxisReflection=>1)
    return billiard, RealPlaneWaves(20, billiard, sector)
end
function limacon_bim_billiard(a=LIMACON_A; R=1.0)
    r(phi) = R*(1+a*cos(phi))
    segment = BilliardGeometry.PolarSegment(r; R=R*(1+abs(a)), center=[0.0, 0.0])
    r0 = BilliardGeometry.curve(segment, 0.0)
    dom = BilliardGeometry.PolarDomain{Float64}([segment], [r0], 1)
    return BilliardGeometry.PolarBilliard{Float64}(dom, BilliardGeometry.register_symmetries())
end

# --- C3/Star fixtures --------------------------------------------------
# C3Billiard/StarBilliard only carry Cn (rotation-only) symmetry, which
# RealPlaneWaves cannot represent at all (its symmetry constructor is typed
# `{BG<:AbsReflection}` -- rotations are not accepted), and empirically no
# basis-solver cases are included for C3/Star -- see
# scratchpad/testing-framework-plan.md. BIM solvers work perfectly once
# given the full (not wedge-restricted) closed Fourier-coefficient boundary
# curve directly.
function c3_bim_billiard(a=0.2; scale=1.0)
    R = scale/2
    amp = a*scale/2
    coef = zeros(12)
    coef[6] = amp
    coef[11] = -amp
    segment = BilliardGeometry.FourierCoeffPolarSegment(coef; R=R)
    r0 = BilliardGeometry.curve(segment, 0.0)
    dom = BilliardGeometry.PolarDomain{Float64}([segment], [r0], 1)
    return BilliardGeometry.PolarBilliard{Float64}(dom, BilliardGeometry.register_symmetries())
end
function star_bim_billiard(R=1.0, a=0.2, n=5)
    coef = zeros(2n)
    coef[2n] = a
    segment = BilliardGeometry.FourierCoeffPolarSegment(coef; R=R)
    r0 = BilliardGeometry.curve(segment, 0.0)
    dom = BilliardGeometry.PolarDomain{Float64}([segment], [r0], 1)
    return BilliardGeometry.PolarBilliard{Float64}(dom, BilliardGeometry.register_symmetries())
end

# --- Mushroom fixtures -------------------------------------------------
# MushroomBilliard has a CompositeDomain (3 glued sub-domains: circle cap,
# triangular cap-fill, rectangular stem) with genuine geometric corners.
# Same BIM limitation pattern as Ellipse/Limacon (symmetry-reduced
# construct_matrices never converges) PLUS its own physical curves keep
# per-sub-domain `domain_id`s (1/2/3), so CompositeBIMSolver mis-detects 3
# components -- deliberately excluded (see plan doc). Basis solvers were
# tried and don't converge either (genuine sharp corners need a corner
# -adapted basis, out of scope) -- also excluded. Only DLP/CFIE/Beyn
# /ExpandedBIM work, on the fully-reconstructed (un-reduced) closed boundary
# via a minimal local `<:AbsBilliard` wrapper around `full_boundary(mush)`.
struct FullBoundaryBilliard{T} <: BilliardGeometry.AbsBilliard where T<:Real
    fundamental_domain::BilliardGeometry.SimpleDomain{T}
    symmetries::BilliardGeometry.SymmetryRegistry
end
function mushroom_bim_billiard(half_width=0.5, stem_height=1.0; R=1.0)
    mush = BilliardGeometry.MushroomBilliard(half_width, stem_height; R=R)
    fb = BilliardGeometry.full_boundary(mush)
    corners = SVector{2,Float64}[BilliardGeometry.curve(c, 0.0) for c in fb]
    dom = BilliardGeometry.SimpleDomain{Float64}(fb, corners, 1)
    return FullBoundaryBilliard{Float64}(dom, BilliardGeometry.register_symmetries())
end

# ---------------------------------------------------------------------------
# CASES: built up per-billiard below via `BilliardCaseSet` + `add_*!`. Kept
# as a single flat `Vector{Case}` at the end since generate_reference_spectra.jl
# and plottingtests_comprehensive.jl only ever iterate over it -- they don't
# need to know it was assembled from several `BilliardCaseSet`s.
#
# `plot=true` marks the subset already smoke-tested through
# `plot_state_tests!` (see plottingtests_comprehensive.jl's own notes on
# which combinations hit the known D2-symmetry `BoundaryPoints` plotting bug
# and are therefore left at plot=false even though they generate reference
# constants fine).
#
# Known-broken cell, deliberately NOT included: `CompositeBIMSolver` on
# `CircleBilliard`/`AnnularBilliard` -- even a single-component, simply
# connected kernel raises a KrylovKit.svdsolve ArgumentError in `solve_vect`
# (see memories/repo/bim-solver-notes.md, Step 11). The `PolarBilliard`-based
# composite cases below (built on the full, un-reduced closed curve, not the
# D2 quadrant-fundamental-domain billiard) were empirically verified to work
# and are included as real, enabled cases.
# ---------------------------------------------------------------------------

CASES = Case[]

# =============================================================================
# Circle (R=1.0) -- D2 symmetric. Basis fixture uses CircleBilliard directly
# (RealPlaneWaves quadrant fundamental domain); BIM fixture uses
# circle_bim_billiard() (full, un-reduced closed curve).
#
# NOTE: basis-solver plotting is deliberately left at plot=false -- it hits
# the known D2-symmetry `BoundaryPoints` bug documented in
# plottingtests_comprehensive.jl; only Circle *BIM* cases are plotted.
# =============================================================================

circle = BilliardCaseSet("CIRCLE"; basis_fixture=circle_basis_fixture, bim_fixture=circle_bim_billiard, k0=2.4)
add_basis_solvers!(circle; dim=5.0, pts=10.0, int_pts=2.0)
add_bim_solvers!(circle; gradings=(SmoothPeriodicGrading(),), plot=true)
let kernels = [DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()),
               CombinedFieldIntegralEquationSolver(5.0; grading=SmoothPeriodicGrading())]
    add_accelerated_solvers!(circle; kernels, accelerators=(:beyn,), use_chebyshev=(false,), plot=true)
    add_accelerated_solvers!(circle; kernels, accelerators=(:beyn,), use_chebyshev=(true,), plot=true)
    add_accelerated_solvers!(circle; kernels, accelerators=(:ebim,), use_chebyshev=(false, true), plot=true)
end
add_composite_bim_solvers!(circle, DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()))
append!(CASES, circle.cases)

# =============================================================================
# Veech right triangle (n=5) -- no symmetry, genuine corners. Basis fixture
# uses the corner-adapted Fourier-Bessel basis; BIM fixture uses the full,
# un-reduced closed boundary.
# =============================================================================

triangle = BilliardCaseSet("TRIANGLE"; basis_fixture=triangle_basis_fixture, bim_fixture=triangle_bim_billiard, k0=6.1)
add_basis_solvers!(triangle; dim=2.0, pts=5.0, int_pts=2.0, plot=true)
add_bim_solvers!(triangle; gradings=(GlobalCornerGrading(),), plot=true)
let kernels = [DoubleLayerPotentialSolver(5.0; grading=GlobalCornerGrading()),
               CombinedFieldIntegralEquationSolver(5.0; grading=GlobalCornerGrading())]
    add_accelerated_solvers!(triangle; kernels, accelerators=(:beyn,), use_chebyshev=(false,), plot=true)
    add_accelerated_solvers!(triangle; kernels, accelerators=(:beyn,), use_chebyshev=(true,), plot=true)
    add_accelerated_solvers!(triangle; kernels, accelerators=(:ebim,), use_chebyshev=(false, true), plot=true)
end
append!(CASES, triangle.cases)

# =============================================================================
# Ellipse (a=1.0,b=0.8) -- D2 symmetric, star-shaped/convex. Basis fixture
# uses EllipseBilliard directly (quadrant fundamental domain, like Circle);
# BIM fixture uses ellipse_bim_billiard() (full, un-reduced closed curve --
# see that function's docstring/comment for why).
# =============================================================================

ellipse = BilliardCaseSet("ELLIPSE"; basis_fixture=ellipse_basis_fixture, bim_fixture=ellipse_bim_billiard, k0=2.72)
add_basis_solvers!(ellipse; dim=8.0, pts=15.0, int_pts=2.0)
add_bim_solvers!(ellipse; gradings=(SmoothPeriodicGrading(),))
let kernels = [DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()),
               CombinedFieldIntegralEquationSolver(5.0; grading=SmoothPeriodicGrading())]
    add_accelerated_solvers!(ellipse; kernels, plot=true)
end
add_composite_bim_solvers!(ellipse, DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()))
append!(CASES, ellipse.cases)

# =============================================================================
# Limacon (a=0.5) -- single XAxisReflection symmetry. Basis fixture uses
# LimaconBilliard directly (RealPlaneWaves(sym_y=1), NOT sym_x -- see
# limacon_basis_fixture's comment); BIM fixture uses limacon_bim_billiard()
# (full, un-reduced closed curve).
# =============================================================================

limacon = BilliardCaseSet("LIMACON"; basis_fixture=limacon_basis_fixture, bim_fixture=limacon_bim_billiard, k0=3.7)
add_basis_solvers!(limacon; dim=10.0, pts=20.0, int_pts=2.0)
add_bim_solvers!(limacon; gradings=(SmoothPeriodicGrading(),))
let kernels = [DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()),
               CombinedFieldIntegralEquationSolver(5.0; grading=SmoothPeriodicGrading())]
    add_accelerated_solvers!(limacon; kernels, plot=true)
end
add_composite_bim_solvers!(limacon, DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()))
append!(CASES, limacon.cases)

# =============================================================================
# C3Billiard (a=0.2, Cn=3 rotational symmetry only). BIM-only -- see
# c3_bim_billiard's comment above for why basis solvers are excluded (no
# basis_fixture given).
# =============================================================================

c3 = BilliardCaseSet("C3"; bim_fixture=c3_bim_billiard, k0=5.65)
add_bim_solvers!(c3; gradings=(SmoothPeriodicGrading(),))
let kernels = [DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()),
               CombinedFieldIntegralEquationSolver(5.0; grading=SmoothPeriodicGrading())]
    add_accelerated_solvers!(c3; kernels, plot=true)
end
add_composite_bim_solvers!(c3, DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()))
append!(CASES, c3.cases)

# =============================================================================
# StarBilliard (R=1.0, a=0.2, n=5, Cn=5 rotational symmetry only). BIM-only,
# same reasoning as C3 above.
# =============================================================================

star = BilliardCaseSet("STAR"; bim_fixture=star_bim_billiard, k0=5.3)
add_bim_solvers!(star; gradings=(SmoothPeriodicGrading(),))
let kernels = [DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()),
               CombinedFieldIntegralEquationSolver(5.0; grading=SmoothPeriodicGrading())]
    add_accelerated_solvers!(star; kernels, plot=true)
end
add_composite_bim_solvers!(star, DoubleLayerPotentialSolver(5.0; grading=SmoothPeriodicGrading()))
append!(CASES, star.cases)

# =============================================================================
# MushroomBilliard(0.5, 1.0) -- genuine corners, no CompositeBIM (3
# domain_ids mismatch a 1-component solver; see mushroom_bim_billiard's
# comment above). Basis solvers excluded (don't converge, see comment; no
# basis_fixture given).
# =============================================================================

mushroom = BilliardCaseSet("MUSHROOM"; bim_fixture=mushroom_bim_billiard, k0=4.9)
add_bim_solvers!(mushroom; gradings=(GlobalCornerGrading(),))
let kernels = [DoubleLayerPotentialSolver(5.0; grading=GlobalCornerGrading()),
               CombinedFieldIntegralEquationSolver(5.0; grading=GlobalCornerGrading())]
    add_accelerated_solvers!(mushroom; kernels, plot=true)
end
append!(CASES, mushroom.cases)
