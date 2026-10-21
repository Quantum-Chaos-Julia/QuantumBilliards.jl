using Test
using QuantumBilliards, BilliardGeometry

const QB_TEST_GROUPS = ("unit", "integration", "high_k", "legacy", "all")
const QB_TEST_GROUP = get(ENV, "QB_TEST_GROUP", "all")
QB_TEST_GROUP in QB_TEST_GROUPS ||
    error("Unknown QB_TEST_GROUP=\"$QB_TEST_GROUP\"; expected one of $(join(QB_TEST_GROUPS, ", "))")

if QB_TEST_GROUP in ("unit", "all")
    include("unit/unittests.jl")
end

if QB_TEST_GROUP in ("integration", "high_k", "all")
    include("integration/common.jl")
end

if QB_TEST_GROUP in ("integration", "all")
    include("integration/integrationtests.jl")
end

if QB_TEST_GROUP == "high_k"
    include("integration/high_k/beyn_high_k_spectra.jl")
end

if QB_TEST_GROUP in ("legacy", "all")
    include("tests_all.jl")
    include("solvertests.jl")
    include("boundarypointstests.jl")
    include("chebyshevtests.jl")
    include("symmetrysectortests.jl")
    include("fourierbesseltests.jl")
end
