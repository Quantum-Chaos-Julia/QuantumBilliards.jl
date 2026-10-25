# Package quality checks with Aqua.jl, one testset per check.

using Aqua

@testset "Aqua - method ambiguities" begin
    Aqua.test_ambiguities(QuantumBilliards; recursive=false)
end

# Source findings: `BIMEigenstate(k, vec::Union{Nothing,Vector{K}}, ...)` (states/eigenstates.jl) and
# `RealPlaneWaves(dim, symmetries::Union{Nothing,Vector{BG}}, ...)` (basis/planewaves/realplanewaves.jl) leave
# `K` / `BG` unbound when the argument is `nothing`.
@testset "Aqua - unbound type parameters" begin
    Aqua.test_unbound_args(QuantumBilliards; broken=true)
end

# Source findings: exported but undefined are BoundaryPointsDM, BoundaryPointsSM, blas_1, blas_multi,
# blas_multi_then_1, construct_matrices_benchmark, print_benchmark_info and use_threads.
@testset "Aqua - undefined exports" begin
    Aqua.test_undefined_exports(QuantumBilliards; broken=true)
end

@testset "Aqua - project extras" begin
    Aqua.test_project_extras(QuantumBilliards)
end

# Source finding: IntervalArithmetic is listed in [deps] but never loaded.
@testset "Aqua - stale dependencies" begin
    @test_broken isempty(Aqua.find_stale_deps(QuantumBilliards))
end

# Source findings: [compat] lacks entries for BilliardGeometry, FastGaussQuadrature, LinearAlgebra, Printf,
# ProgressMeter, Random, StatsBase (see the `deps` check) and for the `Test` extra.
@testset "Aqua - dependency compat bounds" begin
    Aqua.test_deps_compat(QuantumBilliards; broken=true, check_extras=(; broken=true))
end
