# Case-set builder: reduces the billiard/solver test matrix (previously a
# flat list of ~90 hand-written `Case(...)` literals in spectrum_cases.jl,
# one per billiard/basis/solver/kernel/grading/Chebyshev combination) to a
# small number of declarative `add_*!` calls per billiard.
#
# `BilliardCaseSet` bundles a billiard's fixtures (`basis_fixture`/
# `bim_fixture`) and default `k0`/`dk0`; the `add_basis_solvers!`/
# `add_bim_solvers!`/`add_accelerated_solvers!` bulk adders build the cross
# product of solver families x kernels x gradings x Chebyshev variants
# requested, skipping/warning on combinations known to be invalid (e.g.
# `CornerGrading` with a `DoubleLayerPotentialSolver` kernel, or Chebyshev
# acceleration wrapping a `CompositeBIMSolver` kernel) instead of
# constructing a solver that would only fail later. `add_composite_bim_solvers!`
# is kept separate (not part of the default/bulk matrix) since
# `CompositeBIMSolver` is only meaningful for multiply-connected geometries.
# `add_custom!` is the escape hatch for anything the bulk matrix can't
# express (a bespoke solver config, a non-default fixture/k0).
#
# Every case name is derived deterministically from `describe`/`case_name`
# (see solver_naming.jl) -- never hand-typed -- so it can never drift out of
# sync with the solver it actually names.

include(joinpath(@__DIR__, "solver_naming.jl"))

# ---------------------------------------------------------------------------
# Case / default_states
# ---------------------------------------------------------------------------

"""
    default_states(solver) -> Vector{Tuple{String,Float64,Int}}

The default extra multi-state spectrum window(s) attached to a `Case` for
`solver`, unless `states` is given explicitly to an `add_*!` call.
Already-accelerated solvers get a single 10-state window near `k0=100`;
sweep solvers (upgraded via `accel_solver_for` when actually run, see
spectrum_harness.jl) get a single 5-state window near `k0=20`.
"""
default_states(solver::Union{AcceleratedBasisSolver,AcceleratedBIMSolver}) = [("K100", 100.0, 10)]
default_states(solver::Union{SweepBasisSolver,SweepBIMSolver}) = [("K20", 20.0, 5)]

struct Case
    name::String            # deterministic, from `case_name` -- never hand-typed
    billiard_label::String  # e.g. "CIRCLE" -- given explicitly at `BilliardCaseSet` creation,
                             # used for figure folder grouping (see plottingtests_comprehensive.jl)
    kind::Symbol             # :basis | :bim
    enabled::Bool            # include in reference-constant generation
    plot::Bool               # include in plottingtests_comprehensive.jl
    fixture::Function        # :basis -> () -> (billiard,basis); :bim -> () -> billiard
    solver::Function         # () -> native solver (sweep or already accelerated)
    k0::Float64
    dk0::Float64
    states::Vector{Tuple{String,Float64,Int}}  # extra (suffix, k0, n_target) windows
end

# ---------------------------------------------------------------------------
# BilliardCaseSet
# ---------------------------------------------------------------------------

"""
    BilliardCaseSet(label; basis_fixture=nothing, bim_fixture=nothing, k0, dk0=0.01)

Builder collecting every `Case` for one billiard. `label` is given
explicitly (not derived from `nameof(typeof(billiard))`, which can collide
across fixtures sharing an underlying geometry type, e.g. `PolarBilliard`
for Circle/Ellipse/Limacon/C3/Star) and is used both as the case-name prefix
and the top-level figure folder for this billiard.

`k0`/`dk0` are the default ground-state target/tolerance shared by every
`add_*!` call on this set (overridable per call) -- `dk0` in particular is
the single knob most likely to need per-billiard/per-solver tuning, since it
can dominate convergence, hence being easy to override is a first-class
concern of every adder below.

At least one of `basis_fixture`/`bim_fixture` must be given. Populate a set
with `add_basis_solvers!`/`add_bim_solvers!`/`add_accelerated_solvers!`/
`add_composite_bim_solvers!`/`add_custom!`, then `append!(CASES, set.cases)`.
"""
mutable struct BilliardCaseSet
    label::String
    basis_fixture::Union{Function,Nothing}
    bim_fixture::Union{Function,Nothing}
    k0::Float64
    dk0::Float64
    cases::Vector{Case}
end

function BilliardCaseSet(label::AbstractString; basis_fixture::Union{Function,Nothing}=nothing,
                          bim_fixture::Union{Function,Nothing}=nothing, k0::Real, dk0::Real=0.01)
    isempty(label) && throw(ArgumentError("BilliardCaseSet label must not be empty"))
    basis_fixture === nothing && bim_fixture === nothing &&
        throw(ArgumentError("BilliardCaseSet \"$(label)\" needs at least one of basis_fixture/bim_fixture"))
    k0 > 0 || throw(ArgumentError("k0 must be positive, got $(k0)"))
    dk0 > 0 || throw(ArgumentError("dk0 must be positive, got $(dk0)"))
    return BilliardCaseSet(String(label), basis_fixture, bim_fixture, Float64(k0), Float64(dk0), Case[])
end

_require_basis_fixture(set::BilliardCaseSet) = set.basis_fixture === nothing &&
    throw(ArgumentError("BilliardCaseSet \"$(set.label)\" has no basis_fixture"))
_require_bim_fixture(set::BilliardCaseSet) = set.bim_fixture === nothing &&
    throw(ArgumentError("BilliardCaseSet \"$(set.label)\" has no bim_fixture"))

# Builds and appends one `Case` to `set`, deriving its name from `case_name`.
function _add_case!(set::BilliardCaseSet, kind::Symbol, solver_factory::Function;
                     dk0::Real, plot::Bool, states, enabled::Bool=true, k0::Real=set.k0,
                     fixture::Union{Function,Nothing}=nothing)
    used_fixture = fixture === nothing ? (kind === :basis ? set.basis_fixture : set.bim_fixture) : fixture
    used_fixture === nothing && throw(ArgumentError("BilliardCaseSet \"$(set.label)\" has no $(kind) fixture"))

    probe = solver_factory()
    basis_probe = kind === :basis ? used_fixture()[2] : nothing
    name = case_name(set.label, probe; basis=basis_probe)
    st = states === nothing ? default_states(probe) : collect(states)

    case = Case(name, set.label, kind, enabled, plot, used_fixture, solver_factory, Float64(k0), Float64(dk0), st)
    push!(set.cases, case)
    return case
end

# ---------------------------------------------------------------------------
# add_basis_solvers!
# ---------------------------------------------------------------------------

const BASIS_SOLVER_FACTORY = Dict(
    :decomp => (d, b, bint) -> DecompositionMethodSolver(d, b),
    :psm    => (d, b, bint) -> ParticularSolutionsMethod(d, b, bint),
    :vs     => (d, b, bint) -> VerginiSaracenoSolver(d, b),
)

"""
    add_basis_solvers!(set; dim, pts, int_pts=2.0, families=(:decomp,:psm,:vs),
                        dk0=set.dk0, plot=false, states=nothing, enabled=true) -> set

Adds one `Case` per requested basis-solver `family` (`:decomp` ->
[`DecompositionMethodSolver`](@ref), `:psm` -> [`ParticularSolutionsMethod`](@ref),
`:vs` -> [`VerginiSaracenoSolver`](@ref)), all built from the shared
`dim`/`pts`/`int_pts` scaling factors (`int_pts` is only used by `:psm`).
Requires `set.basis_fixture`.
"""
function add_basis_solvers!(set::BilliardCaseSet; dim::Real, pts::Real, int_pts::Real=2.0,
                             families=(:decomp, :psm, :vs), dk0::Real=set.dk0, plot::Bool=false,
                             states=nothing, enabled::Bool=true)
    _require_basis_fixture(set)
    for fname in families
        factory = get(BASIS_SOLVER_FACTORY, fname) do
            throw(ArgumentError("unknown basis solver family :$(fname); expected :decomp, :psm or :vs"))
        end
        solver_factory = () -> factory(dim, pts, int_pts)
        _add_case!(set, :basis, solver_factory; dk0, plot, states, enabled)
    end
    return set
end

# ---------------------------------------------------------------------------
# add_bim_solvers!
# ---------------------------------------------------------------------------

const BIM_KERNEL_FACTORY = Dict(
    :dlp  => (b; kwargs...) -> DoubleLayerPotentialSolver(b; kwargs...),
    :cfie => (b; kwargs...) -> CombinedFieldIntegralEquationSolver(b; kwargs...),
)

# Validity of a (kernel, grading) pairing -- see BoundaryGrading/CornerGrading
# docstrings: CornerGrading is a CFIE-only single-curve parametric-corner
# variant that DoubleLayerPotentialSolver does not support.
_valid_grading(::Val, ::BoundaryGrading) = true
_valid_grading(::Val{:dlp}, ::CornerGrading) = false

"""
    add_bim_solvers!(set; kernels=(:dlp,:cfie), b=5.0, min_pts=200,
                      gradings=(SmoothPeriodicGrading(),), dk0=set.dk0,
                      plot=false, states=nothing, enabled=true) -> set

Adds one `Case` per valid `(kernel, grading)` pairing in the cross product of
`kernels` (`:dlp` -> [`DoubleLayerPotentialSolver`](@ref), `:cfie` ->
[`CombinedFieldIntegralEquationSolver`](@ref)) and `gradings`. Invalid
pairings (e.g. `CornerGrading` with `:dlp`) are skipped with an `@info`
instead of constructing a solver that would only fail later. Requires
`set.bim_fixture`.
"""
function add_bim_solvers!(set::BilliardCaseSet; kernels=(:dlp, :cfie), b::Real=5.0, min_pts::Int=200,
                           gradings=(SmoothPeriodicGrading(),), dk0::Real=set.dk0, plot::Bool=false,
                           states=nothing, enabled::Bool=true)
    _require_bim_fixture(set)
    for kname in kernels
        factory = get(BIM_KERNEL_FACTORY, kname) do
            throw(ArgumentError("unknown BIM kernel :$(kname); expected :dlp or :cfie"))
        end
        for grading in gradings
            if !_valid_grading(Val(kname), grading)
                @info "add_bim_solvers!: skipping invalid kernel/grading combination" billiard=set.label kernel=kname grading=typeof(grading)
                continue
            end
            solver_factory = () -> factory(b; min_pts, grading)
            _add_case!(set, :bim, solver_factory; dk0, plot, states, enabled)
        end
    end
    return set
end

# ---------------------------------------------------------------------------
# add_accelerated_solvers!
# ---------------------------------------------------------------------------

# Chebyshev-accelerated kernel evaluation is unsupported for a
# CompositeBIMSolver kernel (see BeynSolver/ExpandedBIMSolver docstrings).
_supports_chebyshev(::SweepBIMSolver) = true
_supports_chebyshev(::CompositeBIMSolver) = false

"""
    add_accelerated_solvers!(set; kernels::Vector{<:SweepBIMSolver},
                              accelerators=(:beyn,:ebim), use_chebyshev=(false,true),
                              dk0=set.dk0, plot=false, states=nothing, enabled=true,
                              beyn_kwargs=NamedTuple(), ebim_kwargs=NamedTuple()) -> set

Adds one `Case` per combination in the cross product of `kernels` x
`accelerators` (`:beyn` -> [`BeynSolver`](@ref), `:ebim` ->
[`ExpandedBIMSolver`](@ref)) x `use_chebyshev`. Combinations requesting
Chebyshev acceleration with a kernel that doesn't support it (a
[`CompositeBIMSolver`](@ref) kernel) are skipped with an `@info`. Requires
`set.bim_fixture`.
"""
function add_accelerated_solvers!(set::BilliardCaseSet; kernels::Vector{<:SweepBIMSolver},
                                   accelerators=(:beyn, :ebim), use_chebyshev=(false, true),
                                   dk0::Real=set.dk0, plot::Bool=false, states=nothing, enabled::Bool=true,
                                   beyn_kwargs::NamedTuple=NamedTuple(), ebim_kwargs::NamedTuple=NamedTuple())
    _require_bim_fixture(set)
    for kernel in kernels, aname in accelerators, cheb in use_chebyshev
        if cheb && !_supports_chebyshev(kernel)
            @info "add_accelerated_solvers!: skipping unsupported Chebyshev acceleration" billiard=set.label kernel=typeof(kernel) accelerator=aname
            continue
        end
        solver_factory = if aname === :beyn
            () -> BeynSolver(kernel; use_chebyshev=cheb, beyn_kwargs...)
        elseif aname === :ebim
            () -> ExpandedBIMSolver(kernel; use_chebyshev=cheb, ebim_kwargs...)
        else
            throw(ArgumentError("unknown accelerator :$(aname); expected :beyn or :ebim"))
        end
        _add_case!(set, :bim, solver_factory; dk0, plot, states, enabled)
    end
    return set
end

# ---------------------------------------------------------------------------
# add_composite_bim_solvers! -- deliberately separate from the default/bulk
# matrix: CompositeBIMSolver is only meaningful for multiply-connected
# geometries, so it must be opted into per-billiard, not cross-producted.
# ---------------------------------------------------------------------------

"""
    add_composite_bim_solvers!(set, component_solvers::SweepBIMSolver...;
                                dk0=set.dk0, plot=false, states=nothing, enabled=true) -> case::Case

Adds a single [`CompositeBIMSolver`](@ref) `Case` from `component_solvers`
(outer boundary first, holes `2:end` -- see `CompositeBIMSolver`'s own
docstring). Kept separate from `add_bim_solvers!`/`add_accelerated_solvers!`
since a composite solver is only useful for multiply-connected billiards,
not something every billiard should get by default. Requires
`set.bim_fixture`.
"""
function add_composite_bim_solvers!(set::BilliardCaseSet, component_solvers::SweepBIMSolver...;
                                     dk0::Real=set.dk0, plot::Bool=false, states=nothing, enabled::Bool=true)
    _require_bim_fixture(set)
    isempty(component_solvers) && throw(ArgumentError("add_composite_bim_solvers! requires at least one component solver"))
    solver_factory = () -> CompositeBIMSolver(component_solvers...)
    return _add_case!(set, :bim, solver_factory; dk0, plot, states, enabled)
end

# ---------------------------------------------------------------------------
# add_custom! -- escape hatch for anything the bulk matrix above can't
# express (a bespoke solver config, a non-default fixture/k0).
# ---------------------------------------------------------------------------

"""
    add_custom!(set, kind::Symbol, solver_factory::Function; suffix,
                fixture=nothing, k0=set.k0, dk0=set.dk0, states=nothing,
                plot=false, enabled=true) -> case::Case

Adds a single case built from an arbitrary `solver_factory` (a `() ->
solver` closure), for configurations the bulk `add_*!` adders above can't
express. The case name is always `"<SET.LABEL>_<SUFFIX>"` (`suffix`
uppercased) -- the billiard-label prefix is never skippable, so folder
grouping (by `set.label`) and the case name can never disagree. `fixture`
defaults to `set.basis_fixture`/`set.bim_fixture` depending on `kind`, or can
be overridden explicitly (e.g. a one-off, non-default billiard fixture).
"""
function add_custom!(set::BilliardCaseSet, kind::Symbol, solver_factory::Function; suffix::AbstractString,
                      fixture::Union{Function,Nothing}=nothing, k0::Real=set.k0, dk0::Real=set.dk0,
                      states=nothing, plot::Bool=false, enabled::Bool=true)
    kind in (:basis, :bim) || throw(ArgumentError("kind must be :basis or :bim, got :$(kind)"))
    name = "$(uppercase(set.label))_$(uppercase(suffix))"
    probe = solver_factory()
    used_fixture = fixture === nothing ? (kind === :basis ? set.basis_fixture : set.bim_fixture) : fixture
    used_fixture === nothing && throw(ArgumentError("BilliardCaseSet \"$(set.label)\" has no $(kind) fixture; pass `fixture=` explicitly"))
    st = states === nothing ? default_states(probe) : collect(states)
    case = Case(name, set.label, kind, enabled, plot, used_fixture, solver_factory, Float64(k0), Float64(dk0), st)
    push!(set.cases, case)
    return case
end
