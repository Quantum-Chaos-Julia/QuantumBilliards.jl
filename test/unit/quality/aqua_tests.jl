# Package quality checks with Aqua.jl, one testset per check.

using Aqua

@testset "Aqua - method ambiguities" begin
    Aqua.test_ambiguities(QuantumBilliards; recursive=false)
end

@testset "Aqua - unbound type parameters" begin
    Aqua.test_unbound_args(QuantumBilliards)
end

@testset "Aqua - undefined exports" begin
    Aqua.test_undefined_exports(QuantumBilliards)
end

@testset "Aqua - project extras" begin
    Aqua.test_project_extras(QuantumBilliards)
end

@testset "Aqua - stale dependencies" begin
    Aqua.test_stale_deps(QuantumBilliards)
end

@testset "Aqua - dependency compat bounds" begin
    Aqua.test_deps_compat(QuantumBilliards)
end
