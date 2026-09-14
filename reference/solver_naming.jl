# Deterministic solver/basis descriptors shared by the case-generation
# (reference/case_builder.jl, reference/spectrum_cases.jl) and plotting
# (plottingtests_comprehensive.jl, reference/testplots.jl) frameworks.
#
# `describe(x)` is the single source of truth for turning a solver/kernel/
# grading/basis/sampler into:
#   * a `name` fragment -- purely CATEGORICAL info (solver family, kernel,
#     grading variant, Chebyshev on/off, basis type, sampler) used to build
#     deterministic, collision-free case names (see `case_name`) and
#     family-grouped plot folders (see `family_of`);
#   * a `params` fragment -- purely NUMERICAL tuning knobs (dim/pts scaling
#     factors, Beyn's m/nq/r, ...). These are assumed already tuned to a
#     good value for producing a reference spectrum/wavefunction and are
#     therefore only ever *displayed* (plot titles, via `format_params`),
#     never encoded in a name/folder -- so retuning one of these knobs never
#     renames/relocates a case.
#
# Adding a new solver/basis/grading/sampler type to the ecosystem only ever
# requires ONE new `describe` method here -- everything else (naming,
# folder grouping, title annotation) is derived automatically.

# ---------------------------------------------------------------------------
# Leaf categorical descriptors: gradings and samplers are plain Strings.
# ---------------------------------------------------------------------------

describe(::SmoothPeriodicGrading) = "SMOOTH"
describe(::CornerGrading) = "CORNER"
describe(::GlobalCornerGrading) = "GLOBALCORNER"

describe(::GaussLegendreNodes) = "GL"
describe(::LinearNodes) = "LIN"
describe(::FourierNodes) = "FOURIER"

# ---------------------------------------------------------------------------
# Basis descriptors.
# ---------------------------------------------------------------------------

describe(b::RealPlaneWaves) = (; name=(; basis="RPW", sampler=describe(b.sampler)), params=(; dim=b.dim))
describe(b::CornerAdaptedFourierBessel) = (; name=(; basis="FB"), params=(; dim=b.dim, nu=b.nu))

# ---------------------------------------------------------------------------
# Basis-family solver descriptors (sweep + accelerated).
# ---------------------------------------------------------------------------

describe(s::DecompositionMethodSolver) =
    (; name=(; family="DECOMP"), params=(; d=s.dim_scaling_factor, b=s.pts_scaling_factor))
describe(s::ParticularSolutionsMethod) =
    (; name=(; family="PSM"), params=(; d=s.dim_scaling_factor, b=s.pts_scaling_factor, bint=s.int_pts_scaling_factor))
describe(s::VerginiSaracenoSolver) =
    (; name=(; family="VS"), params=(; d=s.dim_scaling_factor, b=s.pts_scaling_factor))

# ---------------------------------------------------------------------------
# BIM sweep-solver descriptors.
# ---------------------------------------------------------------------------

describe(s::DoubleLayerPotentialSolver) =
    (; name=(; family="DLP", grading=describe(s.grading)), params=(; b=s.pts_scaling_factor))
describe(s::CombinedFieldIntegralEquationSolver) =
    (; name=(; family="CFIE", grading=describe(s.grading)), params=(; b=s.pts_scaling_factor))
describe(s::CompositeBIMSolver) =
    (; name=(; family="COMPOSITE", components=Tuple(describe(cs).name for cs in s.component_solvers)),
       params=(; components=Tuple(describe(cs).params for cs in s.component_solvers)))

# ---------------------------------------------------------------------------
# Accelerated BIM-solver descriptors (wrap an inner SweepBIMSolver kernel).
# ---------------------------------------------------------------------------

describe(s::BeynSolver) =
    (; name=(; family="BEYN", kernel=describe(s.kernel).name, cheb=(s.use_chebyshev ? "CHEB" : nothing)),
       params=(; kernel=describe(s.kernel).params, m=s.m, nq=s.nq, r=s.r))
describe(s::ExpandedBIMSolver) =
    (; name=(; family="EBIM", kernel=describe(s.kernel).name, cheb=(s.use_chebyshev ? "CHEB" : nothing)),
       params=(; kernel=describe(s.kernel).params))

# ---------------------------------------------------------------------------
# Flattening helpers: walk a (possibly nested) NamedTuple/Tuple descriptor
# into, respectively, an ordered list of name fragments or key=>value pairs.
# ---------------------------------------------------------------------------

flatten_name(::Nothing) = String[]
flatten_name(x::AbstractString) = [uppercase(x)]
flatten_name(x::NamedTuple) = reduce(vcat, (flatten_name(v) for v in values(x)); init=String[])
flatten_name(x::Tuple) = reduce(vcat, (flatten_name(v) for v in x); init=String[])

flatten_params(::Nothing) = Pair{String,Any}[]
function flatten_params(nt::NamedTuple)
    out = Pair{String,Any}[]
    for (k, v) in pairs(nt)
        if v isa NamedTuple || v isa Tuple
            append!(out, flatten_params(v))
        else
            push!(out, string(k) => v)
        end
    end
    return out
end
flatten_params(t::Tuple) = reduce(vcat, (flatten_params(v) for v in t); init=Pair{String,Any}[])

# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------

"""
    case_name(label::AbstractString, solver; basis=nothing) -> String

Deterministic, collision-free case name built from `label` (a billiard name
given explicitly by the caller -- see `BilliardCaseSet` -- since
`nameof(typeof(billiard))` can collide/be inconsistent across fixtures) plus
the categorical `name` descriptor of `solver` (see `describe`), and of
`basis` when given (basis-family solvers).
"""
function case_name(label::AbstractString, solver; basis=nothing)
    parts = String[uppercase(label)]
    append!(parts, flatten_name(describe(solver).name))
    basis === nothing || append!(parts, flatten_name(describe(basis).name))
    return join(parts, "_")
end

"""
    family_of(solver) -> String

Top-level solver family (`"DECOMP"`, `"PSM"`, `"VS"`, `"DLP"`, `"CFIE"`,
`"COMPOSITE"`, `"BEYN"`, `"EBIM"`), used to group plot output folders by
solver family regardless of kernel/grading/Chebyshev variant.
"""
family_of(solver) = describe(solver).name.family

"""
    display_name(solver) -> String

Human-readable, space-joined rendering of `describe(solver).name`'s
categorical fragments (e.g. `"BEYN DLP SMOOTH"`), for use in plot titles --
see `case_name` for the underscore-joined, label-prefixed variant used for
case names/filenames/folders.
"""
display_name(solver) = join(flatten_name(describe(solver).name), " ")

"""
    latex_display_name(solver) -> String

Same fragments as [`display_name`](@ref), joined with an explicit LaTeX
inter-word space (`"\\\\ "`) instead of a plain space -- math mode (as used
inside a `\\mathrm{...}` plot title, see reference/testplots.jl) otherwise
collapses ordinary whitespace and renders the fragments run together.
"""
latex_display_name(solver) = join(flatten_name(describe(solver).name), "\\ ")

_fmt_param(v::AbstractFloat) = string(round(v; sigdigits=6))
_fmt_param(v::AbstractVector) = "[" * join((_fmt_param(x) for x in v), ",") * "]"
_fmt_param(v) = string(v)

"""
    format_params(solver; basis=nothing, latex=false) -> String

Formats the numerical tuning `params` descriptor of `solver` (and `basis`,
when given) as a compact `"key=value, key=value, ..."` string for display in
plot titles/annotations -- see `describe`. With `latex=true`, fragments are
joined with an explicit LaTeX inter-word space (`",\\\\ "`) instead of a
plain `", "`, since math mode (as used inside a `\\mathrm{...}`/`L"..."`
plot title, see reference/testplots.jl) otherwise collapses ordinary
whitespace.
"""
function format_params(solver; basis=nothing, latex::Bool=false)
    pairs_ = flatten_params(describe(solver).params)
    basis === nothing || append!(pairs_, flatten_params(describe(basis).params))
    sep = latex ? ",\\ " : ", "
    return join(("$(k)=$(_fmt_param(v))" for (k, v) in pairs_), sep)
end
