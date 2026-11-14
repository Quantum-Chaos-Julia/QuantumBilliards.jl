# Unit tier: mirrors src/. Each subgroup is a nested @testset so the summary groups results.

@group "spectra" begin
    run_file("unit/spectra/spectralutils_tests.jl")
    run_file("unit/spectra/unfolding_tests.jl")
end

@group "solvers: infrastructure" begin
    run_file("unit/solvers/boundarypoints_tests.jl")
    run_file("unit/solvers/boundarygeomcache_tests.jl")
    run_file("unit/solvers/boundarygrading_tests.jl")
    run_file("unit/solvers/decompositions_tests.jl")
    run_file("unit/solvers/taylor/recurrence_core_tests.jl")
    run_file("unit/solvers/chebyshev/bessels_tests.jl")
    run_file("unit/solvers/chebyshev/optimalpanelization_tests.jl")
    run_file("unit/solvers/acceleratedmethods/chebyshevconfig_tests.jl")
end

@group "states: symmetry" begin
    run_file("unit/states/symmetry/symmetrysector_tests.jl")
    run_file("unit/states/symmetry/reflections_tests.jl")
end

@group "basis" begin
    run_file("unit/basis/realplanewaves_tests.jl")
    run_file("unit/basis/realplanewaves_recurrence_tests.jl")
    run_file("unit/basis/fourierbessel_tests.jl")
    run_file("unit/basis/fourierbessel_recurrence_tests.jl")
    run_file("unit/basis/corneradapted_tests.jl")
    run_file("unit/basis/corneradapted_recurrence_tests.jl")
    run_file("unit/basis/basisinterface_tests.jl")
end

@group "solvers: basis methods" begin
    run_file("unit/solvers/solverconstructors_tests.jl")
    run_file("unit/solvers/computationplan_tests.jl")
    run_file("unit/solvers/matrixconstructors_tests.jl")
    run_file("unit/solvers/sweepmethods/sweepmethods_tests.jl")
    run_file("unit/solvers/sweepmethods/decompositionmethod_tests.jl")
    run_file("unit/solvers/sweepmethods/particularsolutions_tests.jl")
    run_file("unit/solvers/acceleratedmethods/verginisaraceno_tests.jl")
end

@group "solvers: BIM" begin
    run_file("unit/solvers/sweepmethods/dlp_tests.jl")
    run_file("unit/solvers/sweepmethods/cfie_tests.jl")
    run_file("unit/solvers/sweepmethods/compositebim_tests.jl")
    run_file("unit/solvers/chebyshev/dlp_cheb_tests.jl")
    run_file("unit/solvers/chebyshev/cfie_cheb_tests.jl")
    run_file("unit/solvers/taylor/bim_kernels_recurrence_tests.jl")
end

@group "solvers: accelerated" begin
    run_file("unit/solvers/acceleratedmethods/beyn_tests.jl")
    run_file("unit/solvers/acceleratedmethods/cork_tests.jl")
    run_file("unit/solvers/acceleratedmethods/ebim_tests.jl")
end

@group "states" begin
    run_file("unit/states/eigenstates_tests.jl")
    run_file("unit/states/wavefunctions_tests.jl")
    run_file("unit/states/boundaryfunctions_tests.jl")
    run_file("unit/states/husimifunctions_tests.jl")
end

@group "quality" begin
    run_file("unit/quality/aqua_tests.jl")
end
