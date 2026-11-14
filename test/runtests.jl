# Entry point. Shared setup, selection variables (QB_TEST_GROUP, QB_TEST_FILTER, QB_TEST_FILE) and the @group macro
# live in harness.jl. For a single file or a subset use `julia --project=test test/run.jl <path-or-substring>...`.

include("harness.jl")

if QB_TEST_GROUP in ("unit", "all")
    include("unit/unittests.jl")
end

if QB_TEST_GROUP in ("integration", "all")
    include("integration/common.jl")
    include("integration/integrationtests.jl")
end

finish_tests()
