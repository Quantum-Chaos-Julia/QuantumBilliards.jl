# Shared test harness: definitions only. Included by runtests.jl before the tier files.
#
# Selection (all optional, combined with AND):
#   QB_TEST_GROUP   unit | integration | all (default)  - tier
#   QB_TEST_FILTER  comma-separated substrings of @group names
#   QB_TEST_FILE    comma-separated substrings of test file paths relative to test/ (e.g. "unit/solvers/dlp,beyn")
# Tier files include test files only through `run_file("<path relative to test/>")`, which is what QB_TEST_FILE filters and
# which runs each file in its own module.

using Test
using Logging
using QuantumBilliards, BilliardGeometry

global_logger(ConsoleLogger(stderr, Logging.Error))

const QB_TEST_ROOT = @__DIR__
const QB_TEST_GROUPS = ("unit", "integration", "all")
const QB_TEST_GROUP = get(ENV, "QB_TEST_GROUP", "all")
QB_TEST_GROUP in QB_TEST_GROUPS ||
    error("Unknown QB_TEST_GROUP=\"$QB_TEST_GROUP\"; expected one of $(join(QB_TEST_GROUPS, ", "))")

_env_patterns(name) = filter(!isempty, strip.(lowercase.(split(get(ENV, name, ""), ','))))
const QB_TEST_FILTER = _env_patterns("QB_TEST_FILTER")
const QB_TEST_FILE = _env_patterns("QB_TEST_FILE")

file_enabled(rel::AbstractString) =
    isempty(QB_TEST_FILE) || any(f -> occursin(f, lowercase(rel)), QB_TEST_FILE)
group_name_enabled(name::AbstractString) =
    isempty(QB_TEST_FILTER) || any(f -> occursin(f, lowercase(name)), QB_TEST_FILTER)

# A group runs when its name passes QB_TEST_FILTER and at least one of its run_file paths passes QB_TEST_FILE.
group_enabled(name::AbstractString, files::AbstractVector{<:AbstractString}=String[]) =
    group_name_enabled(name) && (isempty(files) || any(file_enabled, files))

const RAN_FILES = String[]

# Each test file runs in its own anonymous module, so it sees only the package, the shared fixtures and the names it defines
# itself. Shared definition-only files (fixtures) are loaded once into their own module, whose names are exported and
# brought into a test module with `using`.
const FIXTURE_MODULES = Dict{String,Module}()
const ALWAYS_FIXTURE = "holes_fixtures.jl"
const INTEGRATION_FIXTURE = "integration/common.jl"

function _prepare_module!(m::Module, rel::AbstractString)
    Core.eval(m, :(using Test, Logging, QuantumBilliards, BilliardGeometry))
    Core.eval(m, Meta.parse("import Main: @requires"))
    rel == ALWAYS_FIXTURE || use_fixture!(m, ALWAYS_FIXTURE)
    startswith(rel, "integration/") && rel != INTEGRATION_FIXTURE && use_fixture!(m, INTEGRATION_FIXTURE)
    return m
end

function _load_fixture(rel::AbstractString)
    modname = Symbol("QBFixture_", replace(rel, r"\W" => "_"))
    m = Module(modname)
    _prepare_module!(m, rel)
    Base.include(m, joinpath(QB_TEST_ROOT, rel))
    skip = (:eval, :include, modname)
    defined = Base.invokelatest(names, m; all=true, imported=false)
    exported = [n for n in defined if n ∉ skip && !startswith(string(n), "#")]
    Core.eval(m, Expr(:export, exported...))
    Core.eval(Main, :(const $modname = $m))
    return m
end

"""
    use_fixture!(target, rel)

Load the definitions-only file `rel` (path relative to `test/`) once per run and bring its names into `target` with `using`.
"""
function use_fixture!(target::Module, rel::AbstractString)
    m = get!(() -> _load_fixture(rel), FIXTURE_MODULES, rel)
    Core.eval(target, Expr(:using, Expr(:., :Main, nameof(m))))
    return nothing
end

"""
    @requires "path"

First line of a test file that needs a shared definitions-only file `path` (relative to `test/`).
"""
macro requires(rel)
    return :(use_fixture!(@__MODULE__, $(esc(rel))))
end

"""
    run_file(rel)

Run the test file `rel` (path relative to `test/`) in a fresh module if it passes `QB_TEST_FILE`.
"""
function run_file(rel::AbstractString)
    file_enabled(rel) || return nothing
    push!(RAN_FILES, rel)
    m = Module(:QBTestFile)
    _prepare_module!(m, rel)
    return Base.include(m, joinpath(QB_TEST_ROOT, rel))
end

# Paths of the `run_file("...")` calls inside a @group body, used to skip groups with no selected file.
function _group_files(ex)
    files = String[]
    walk(x) = nothing
    function walk(x::Expr)
        if x.head === :call && length(x.args) == 2 && x.args[1] === :run_file && x.args[2] isa AbstractString
            push!(files, x.args[2])
        else
            foreach(walk, x.args)
        end
        return nothing
    end
    walk(ex)
    return files
end

# Each @group is a top-level verbose testset, so its summary prints as soon as it finishes.
# A failing group is recorded and the next group still runs; the run errors at the end.
const FAILED_GROUPS = String[]
macro group(name, body)
    files = _group_files(body)
    return esc(quote
        if group_enabled($name, $files)
            try
                @testset verbose=true $name $body
            catch e
                e isa Test.TestSetException || rethrow()
                push!(FAILED_GROUPS, $name)
            end
        end
    end)
end

function finish_tests()
    isempty(QB_TEST_FILE) || !isempty(RAN_FILES) ||
        error("QB_TEST_FILE=\"", join(QB_TEST_FILE, ","), "\" matched no test file in tier \"", QB_TEST_GROUP, "\"")
    isempty(FAILED_GROUPS) || error("Failed test groups: ", join(FAILED_GROUPS, ", "))
    return nothing
end
