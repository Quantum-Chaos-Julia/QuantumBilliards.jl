using Test
using QuantumBilliards, BilliardGeometry

const QB_TEST_GROUPS = ("unit", "integration", "all")
const QB_TEST_GROUP = get(ENV, "QB_TEST_GROUP", "all")
QB_TEST_GROUP in QB_TEST_GROUPS ||
    error("Unknown QB_TEST_GROUP=\"$QB_TEST_GROUP\"; expected one of $(join(QB_TEST_GROUPS, ", "))")

# Outer testset: a failing inner testset no longer aborts the remaining ones.
@testset "QuantumBilliards" begin
    if QB_TEST_GROUP in ("unit", "all")
        include("unit/unittests.jl")
    end

    if QB_TEST_GROUP in ("integration", "all")
        include("integration/common.jl")
        include("integration/integrationtests.jl")
    end
end
