# Run selected test files: julia --project=test -t 4 test/run.jl <path-or-substring>...
# Arguments are test file paths (absolute, or relative to the working directory or to test/) or substrings of
# paths relative to test/ (e.g. `boundarygrading`, `unit/solvers`). They become QB_TEST_FILE. Combine with
# QB_TEST_GROUP=unit|integration to restrict the tier.

isempty(ARGS) && error("usage: julia --project=test test/run.jl <path-or-substring>...")

const _test_dir = @__DIR__
_pattern(arg) = isfile(arg) ? relpath(abspath(arg), _test_dir) :
                isfile(joinpath(_test_dir, arg)) ? arg : arg

ENV["QB_TEST_FILE"] = join(_pattern.(ARGS), ",")
include(joinpath(_test_dir, "runtests.jl"))
