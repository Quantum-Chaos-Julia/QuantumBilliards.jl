# Comprehensive plotting-test template.
#
# Consumes the SAME `Case` table as reference/generate_reference_spectra.jl
# (both `include` reference/spectrum_cases.jl) so a case's fixture/solver is
# guaranteed identical whichever script runs it. For every case with
# `plot=true`:
#   * the ground state at `case.k0` is always plotted;
#   * for each extra multi-state window in `case.states` (e.g. "K20"/"K100"),
#     the LOWEST wavenumber of that window is additionally plotted -- i.e.
#     every multi-state spectrum window also contributes one extra entry to
#     the plotting list, not just the original ground-state k0.
#
# Unlike the previous version of this script, plotting no longer re-solves
# any wavenumber: reference/generate_reference_spectra.jl already solved
# every enabled case exactly once and recorded the result as
# `K_GROUND_<NAME>`/`T_GROUND_<NAME>`/`KS_<NAME>_<SUFFIX>`/
# `TENS_<NAME>_<SUFFIX>` constants in reference/reference_spectra.jl
# (included below). Plotting looks those up directly and hands them to
# `plot_state_tests!` via `QBPlotting.ReferenceWavenumber(k, t)` -- a plain
# dispatch choice (see QBPlotting.jl/src/testplots.jl) that replaces the old
# `compute_wavenumber::Bool`/`t1` keyword pair entirely: passing a
# `ReferenceWavenumber` *is* "compute_wavenumber=false", there is no boolean
# to default. A case missing its reference constant (e.g. one of the
# `[error]` cells in reference_spectra.jl) falls back to solving live, with a
# `@warn`, so a stale/partial reference file never silently plots garbage.
#
# `save_figure_basis`/`save_figure_bim` group output as
# `figures/<case.billiard_label>/<family_of(solver)>/<case.name>_k0_<k>_<suffix>.<filetype>`
# -- folders are grouped by solver FAMILY (`family_of`, e.g. every Beyn
# kernel/grading/Chebyshev variant lands in one "BEYN" folder) rather than
# `nameof(typeof(billiard))`/`nameof(typeof(solver))`, which can collide
# across fixtures sharing an underlying geometry/solver type (e.g.
# `PolarBilliard` for Circle/Ellipse/Limacon/C3/Star). The full deterministic
# case name is embedded in the filename itself, so same-family variants
# never overwrite each other. Title annotations (k, tension, solver family,
# numeric tuning params) come from `plot_state_tests!`/`format_params` (see
# reference/testplots.jl, reference/solver_naming.jl).
#
# `plot_ground`/`plot_states` dispatch on `Val(case.kind)` (`:basis`/`:bim`)
# instead of an `if case.kind === :basis ... else ... end` conditional --
# idiomatic multiple dispatch, per this ecosystem's conventions, in place of
# a runtime branch on a type-erased `Symbol` field.
#
# See scratchpad/testing-framework-plan.md for the full solver/billiard
# matrix. `plot=true` in reference/spectrum_cases.jl marks the subset already
# smoke-tested through `plot_state_tests!`; every `add_*!` bulk adder in
# reference/case_builder.jl defaults to `plot=false`, so most cases are
# generation-only until explicitly verified here -- see the note below on
# the known D2-symmetry `BoundaryPoints` plotting bug that excludes e.g.
# Circle basis-solver cases.
#
# NOTE: Circle basis-solver plotting cases are deliberately `plot=false` in
# reference/spectrum_cases.jl. `save_figure_basis` on `CircleBilliard`
# (RealPlaneWaves, D2 symmetry) hits `plot_boundary_function!` ->
# `boundary_function` -> `apply_symmetries_to_boundary_points`, which raises
# a `MethodError` constructing `BoundaryPoints` (arg-count mismatch against
# the current `BoundaryPoints{T}` constructor in
# QuantumBilliards.jl/src/solvers/boundarypoints.jl) -- a genuine,
# newly-discovered bug (reproduced 2026-09-10), NOT a limitation of this
# test framework. `StadiumBilliard` (also D2 symmetry) does NOT hit this.

using Revise
using CairoMakie

using QuantumBilliards
using BilliardGeometry
using QCPlotting
using QBPlotting

set_theme!(qc_theme)

# ---------------------------------------------------------------------------
# Shared case table (fixtures + solvers + BilliardCaseSet/Case + naming, see
# reference/case_builder.jl and reference/solver_naming.jl), `testplots.jl`
# (`ReferenceWavenumber`/`plot_state_tests!` -- moved here from
# QBPlotting.jl/src/testplots.jl), and the reference constants generated
# once by generate_reference_spectra.jl. Order matters: `ReferenceWavenumber`
# must exist before it's used as a type annotation below.
# ---------------------------------------------------------------------------

include(joinpath(@__DIR__, "reference", "spectrum_cases.jl"))
include(joinpath(@__DIR__, "reference", "testplots.jl"))
include(joinpath(@__DIR__, "reference", "reference_spectra.jl"))

# ---------------------------------------------------------------------------
# Figure-saving helpers. Folders are grouped by billiard label + solver
# family (`family_of`, e.g. "BEYN" groups every Beyn kernel/grading/Chebyshev
# variant together -- see reference/solver_naming.jl); the full deterministic
# case name is embedded in the filename itself so same-family variants never
# collide/overwrite each other.
# ---------------------------------------------------------------------------

function save_figure_basis(case::Case, solver, basis, billiard, source::ReferenceWavenumber; size=(400, 300), filetype="pdf", suffix="", log=false, inside_only=true)
    f = Figure(size=size)
    plot_state_tests!(f, solver, basis, billiard, source; b=10.0, log=log, inside_only=inside_only)

    outdir = joinpath(pwd(), "figures", case.billiard_label, family_of(solver))
    mkpath(outdir)
    filename = "$(case.name)_k0_$(source.k)_$(suffix).$(filetype)"

    save(joinpath(outdir, filename), f)
    return f
end

function save_figure_bim(case::Case, solver, billiard, source::ReferenceWavenumber; size=(400, 300), filetype="pdf", suffix="", log=false, inside_only=true)
    f = Figure(size=size)
    plot_state_tests!(f, solver, billiard, source; b=10.0, log=log, inside_only=inside_only)

    outdir = joinpath(pwd(), "figures", case.billiard_label, family_of(solver))
    mkpath(outdir)
    filename = "$(case.name)_k0_$(source.k)_$(suffix).$(filetype)"

    save(joinpath(outdir, filename), f)
    return f
end

# ---------------------------------------------------------------------------
# Reference-constant lookup
# ---------------------------------------------------------------------------

_maybe_const(name::Symbol) = isdefined(@__MODULE__, name) ? getfield(@__MODULE__, name) : nothing

"""
    ground_reference(case::Case) -> (k, t)

Looks up `case`'s precomputed `K_GROUND_<NAME>`/`T_GROUND_<NAME>` constants
(generated by `reference/generate_reference_spectra.jl`), returning
`(nothing, nothing)` if `case.name` has no recorded reference (e.g. one of
the `[error]` cells in reference_spectra.jl).
"""
ground_reference(case::Case) = _maybe_const(Symbol("K_GROUND_$(case.name)")), _maybe_const(Symbol("T_GROUND_$(case.name)"))

"""
    states_reference(case::Case, suffix::String) -> (ks, tens)

Looks up `case`'s precomputed `KS_<NAME>_<SUFFIX>`/`TENS_<NAME>_<SUFFIX>`
multi-state spectrum-window constants, or `(nothing, nothing)` if missing.
"""
states_reference(case::Case, suffix::String) = _maybe_const(Symbol("KS_$(case.name)_$(suffix)")), _maybe_const(Symbol("TENS_$(case.name)_$(suffix)"))

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

# Resolves the ground-state ReferenceWavenumber for `case`, falling back to a
# live `solve_wavenumber` (with a `@warn`) if no reference constant exists.
function resolve_ground(case::Case, solver, basis, billiard)
    k, t = ground_reference(case)
    if k === nothing
        @warn "no reference ground state, solving live" case.name
        k, t = solve_wavenumber(solver, basis, billiard, case.k0, case.dk0)
    end
    return ReferenceWavenumber(k, t)
end
function resolve_ground(case::Case, solver, billiard)
    k, t = ground_reference(case)
    if k === nothing
        @warn "no reference ground state, solving live" case.name
        k, t = solve_wavenumber(solver, billiard, case.k0, case.dk0)
    end
    return ReferenceWavenumber(k, t)
end

# Resolves the ReferenceWavenumber for the lowest state in a case's
# `(suffix, k0_target, n_target)` spectrum window, falling back to a live
# `accelerated_states_near` (with a `@warn`) if no reference constant exists.
function resolve_states(case::Case, suffix::String, solver, basis, billiard; k0_target::Real, n_target::Int)
    ks, tens = states_reference(case, suffix)
    if ks === nothing
        @warn "no reference states, solving live" case.name suffix
        ks, tens = accelerated_states_near(solver, basis, billiard; k0=k0_target, n_target=n_target)
    end
    idx = argmin(ks)
    return ReferenceWavenumber(ks[idx], tens[idx])
end
function resolve_states(case::Case, suffix::String, solver, billiard; k0_target::Real, n_target::Int)
    ks, tens = states_reference(case, suffix)
    if ks === nothing
        @warn "no reference states, solving live" case.name suffix
        ks, tens = accelerated_states_near(solver, billiard; k0=k0_target, n_target=n_target)
    end
    idx = argmin(ks)
    return ReferenceWavenumber(ks[idx], tens[idx])
end

# `plot_ground`/`plot_states` dispatch on `Val(case.kind)` instead of an
# `if case.kind === :basis ... else ... end` conditional.
plot_ground(case::Case, solver) = plot_ground(case, solver, Val(case.kind))

function plot_ground(case::Case, solver, ::Val{:basis})
    try
        billiard, basis = case.fixture()
        source = resolve_ground(case, solver, basis, billiard)
        @info "plotting (ground)" case.name source.k
        save_figure_basis(case, solver, basis, billiard, source; suffix="")
    catch e
        @error "plotting (ground) failed" case.name case.k0 e
    end
end

function plot_ground(case::Case, solver, ::Val{:bim})
    try
        billiard = case.fixture()
        source = resolve_ground(case, solver, billiard)
        @info "plotting (ground)" case.name source.k
        save_figure_bim(case, solver, billiard, source; suffix="")
    catch e
        @error "plotting (ground) failed" case.name case.k0 e
    end
end

plot_states(case::Case, solver, suffix::String, k0_target::Real, n_target::Int) =
    plot_states(case, solver, suffix, k0_target, n_target, Val(case.kind))

function plot_states(case::Case, solver, suffix::String, k0_target::Real, n_target::Int, ::Val{:basis})
    try
        billiard, basis = case.fixture()
        source = resolve_states(case, suffix, solver, basis, billiard; k0_target, n_target)
        @info "plotting (states)" case.name suffix source.k
        save_figure_basis(case, solver, basis, billiard, source; suffix)
    catch e
        @error "plotting (states) failed" case.name suffix k0_target n_target e
    end
end

function plot_states(case::Case, solver, suffix::String, k0_target::Real, n_target::Int, ::Val{:bim})
    try
        billiard = case.fixture()
        source = resolve_states(case, suffix, solver, billiard; k0_target, n_target)
        @info "plotting (states)" case.name suffix source.k
        save_figure_bim(case, solver, billiard, source; suffix)
    catch e
        @error "plotting (states) failed" case.name suffix k0_target n_target e
    end
end

function run_case(case::Case)
    case.plot || return nothing
    solver = case.solver()
    plot_ground(case, solver)
    for (suffix, k0s, n) in case.states
        plot_states(case, solver, suffix, k0s, n)
    end
    return nothing
end

function main(; cases=CASES)
    for case in cases
        run_case(case)
    end
    return nothing
end

main()
