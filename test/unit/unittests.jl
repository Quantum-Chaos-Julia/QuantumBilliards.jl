# Unit tier: mirrors src/. Each subgroup is a nested @testset so the summary groups results.

@group "spectra" begin
    include("spectra/spectralutils_tests.jl")
    include("spectra/unfolding_tests.jl")
end

@group "solvers: infrastructure" begin
    include("solvers/boundarypoints_tests.jl")
    include("solvers/boundarygeomcache_tests.jl")
    include("solvers/boundarygrading_tests.jl")
    include("solvers/decompositions_tests.jl")
    include("solvers/taylor/recurrence_core_tests.jl")
    include("solvers/chebyshev/bessels_tests.jl")
    include("solvers/chebyshev/optimalpanelization_tests.jl")
    include("solvers/acceleratedmethods/chebyshevconfig_tests.jl")
end

@group "states: symmetry" begin
    include("states/symmetry/symmetrysector_tests.jl")
    include("states/symmetry/reflections_tests.jl")
end

@group "basis" begin
    include("basis/realplanewaves_tests.jl")
    include("basis/realplanewaves_recurrence_tests.jl")
    include("basis/fourierbessel_tests.jl")
    include("basis/fourierbessel_recurrence_tests.jl")
    include("basis/corneradapted_tests.jl")
    include("basis/corneradapted_recurrence_tests.jl")
    include("basis/basisinterface_tests.jl")
end

@group "solvers: basis methods" begin
    include("solvers/solverconstructors_tests.jl")
    include("solvers/matrixconstructors_tests.jl")
    include("solvers/sweepmethods/sweepmethods_tests.jl")
    include("solvers/sweepmethods/decompositionmethod_tests.jl")
    include("solvers/sweepmethods/particularsolutions_tests.jl")
    include("solvers/acceleratedmethods/verginisaraceno_tests.jl")
end

@group "solvers: BIM" begin
    include("solvers/sweepmethods/dlp_tests.jl")
    include("solvers/sweepmethods/cfie_tests.jl")
    include("solvers/sweepmethods/compositebim_tests.jl")
    include("solvers/chebyshev/dlp_cheb_tests.jl")
    include("solvers/chebyshev/cfie_cheb_tests.jl")
    include("solvers/taylor/bim_kernels_recurrence_tests.jl")
end

@group "solvers: accelerated" begin
    include("solvers/acceleratedmethods/beyn_tests.jl")
    include("solvers/acceleratedmethods/cork_tests.jl")
    include("solvers/acceleratedmethods/ebim_tests.jl")
end

@group "states" begin
    include("states/eigenstates_tests.jl")
    include("states/wavefunctions_tests.jl")
    include("states/boundaryfunctions_tests.jl")
    include("states/husimifunctions_tests.jl")
end

@group "quality" begin
    include("quality/aqua_tests.jl")
end
