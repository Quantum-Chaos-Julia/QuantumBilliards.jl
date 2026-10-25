# Unit tests for solver constructors and constructor-level helpers in src/solvers/ and src/abstracttypes.jl:
#   src/solvers/decompositions.jl                  (adjust_scaling_and_samplers)
#   src/solvers/sweepmethods/decompositionmethod.jl, particularsolutions.jl,
#   src/solvers/acceleratedmethods/verginisaraceno.jl   (DecompositionMethodSolver, ParticularSolutionsMethodSolver,
#                                                        VerginiSaracenoSolver: field defaults, typed keywords)
#   src/solvers/sweepmethods/dlp.jl, cfie.jl, compositebim.jl   (DoubleLayerPotentialSolver,
#                                                        CombinedFieldIntegralEquationSolver, CompositeBIMSolver:
#                                                        defaults, symmetry inference, validation)
#   src/solvers/acceleratedmethods/beyn.jl, ebim.jl, cork.jl     (BeynSolver, ExpandedBIMSolver, CORKSolver)
# Every solver type gets one explicit testset: the type hierarchy of src/abstracttypes.jl, field defaults, type parameters,
# argument validation (@test_throws, only for checks the constructors really perform) and the `hasmethod` contract of the
# abstract branch it belongs to (SweepBasisSolver, AcceleratedBasisSolver, SweepBIMSolver, AcceleratedBIMSolver).
# The contract lists below are read off the docstrings of the abstract types and checked against the method tables.
# No eigen solves and no matrix assembly. Helper definitions use the prefix `sc_`.

using StaticArrays: SVector
using QuantumBilliards: adjust_scaling_and_samplers, DLP, CFIE, _bim_numeric_type

# `hasmethod` with a plain argument-type list.
sc_has(f, types...) = hasmethod(f, Tuple{types...})

# Whether the method of `f` selected by the argument-type list accepts keyword `kw`. Only the keyword's presence is
# checked (its default value is not observable without running the method).
sc_has_kw(f, kw::Symbol, types...) = kw in Base.kwarg_decl(which(f, Tuple{types...}))

# Concrete argument types used in the `hasmethod` contracts.
const sc_BASIS_T = typeof(RealPlaneWaves(4))
const sc_PTS_T = BoundaryPoints{Float64}
const sc_RECT_T = typeof(RectangleBilliard(1.0, 0.7))

@testset "adjust_scaling_and_samplers" begin
    circle = CircleBilliard(1.0)                      # 1 fundamental boundary curve
    stadium = StadiumBilliard(0.5)                    # 2 fundamental boundary curves
    triangle = TriangleBilliard(π/2, 1.0)             # 3 fundamental boundary curves

    @testset "ParticularSolutionsMethodSolver: missing entries are filled" begin
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 4.0)       # one scaling factor, one sampler
        bs, samplers = adjust_scaling_and_samplers(solver, stadium)
        @test length(get_boundary_curves(stadium)) == 2
        @test bs == [3.0, 3.0]
        @test length(samplers) == 2
        @test all(s -> s isa GaussLegendreNodes{Float64}, samplers)
        @test eltype(bs) == Float64
    end

    @testset "DecompositionMethodSolver: user supplied entries are kept, defaults are filled" begin
        user_samplers = AbsSampler[LinearNodes(), GaussLegendreNodes()]
        solver = DecompositionMethodSolver(2.0, [3.0, 2.0], user_samplers)
        n = length(get_boundary_curves(triangle))
        @test n == 3
        bs, samplers = adjust_scaling_and_samplers(solver, triangle)
        @test bs == [3.0, 2.0, 2.0]                                     # padded with minimum(bs)
        @test length(samplers) == n
        @test samplers[1] isa LinearNodes
        @test samplers[2] isa GaussLegendreNodes
        @test samplers[3] isa LinearNodes                               # padded with the first sampler
    end

    @testset "VerginiSaracenoSolver: scalar input is expanded per curve" begin
        solver = VerginiSaracenoSolver(2.0, 5.0)
        bs, samplers = adjust_scaling_and_samplers(solver, triangle)
        @test bs == [5.0, 5.0, 5.0]
        @test length(samplers) == 3
        @test all(s -> s isa GaussLegendreNodes, samplers)
    end

    @testset "padding uses the smallest scaling factor, not the last one" begin
        solver = DecompositionMethodSolver(2.0, [7.0, 3.0], AbsSampler[GaussLegendreNodes(), LinearNodes()])
        bs, samplers = adjust_scaling_and_samplers(solver, triangle)
        @test bs[3] == 3.0
        @test samplers[3] isa GaussLegendreNodes                        # first sampler, not the last (LinearNodes)
    end

    @testset "already complete vectors are unchanged" begin
        solver = DecompositionMethodSolver(2.0, [4.0, 6.0], AbsSampler[LinearNodes(), GaussLegendreNodes()])
        bs, samplers = adjust_scaling_and_samplers(solver, stadium)
        @test bs == [4.0, 6.0]
        @test length(samplers) == 2
        @test samplers[1] isa LinearNodes && samplers[2] isa GaussLegendreNodes
    end

    @testset "single-curve billiard" begin
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 4.0)
        bs, samplers = adjust_scaling_and_samplers(solver, circle)
        @test length(get_boundary_curves(circle)) == 1
        @test bs == [3.0]
        @test length(samplers) == 1
    end

    @testset "surplus entries" begin
        solver = ParticularSolutionsMethodSolver(2.0, [3.0, 1.0, 5.0, 7.0], 4.0)
        bs, samplers = adjust_scaling_and_samplers(solver, circle)
        @test bs[1:4] == [3.0, 1.0, 5.0, 7.0]                           # never reordered or reduced in value
        @test length(bs) >= length(get_boundary_curves(circle))         # surplus entries are kept
    end

    @testset "Float32 solver" begin
        solver = VerginiSaracenoSolver(2f0, 5f0)
        bs, samplers = adjust_scaling_and_samplers(solver, stadium)
        @test bs isa Vector{Float32}
        @test bs == Float32[5, 5]
        @test length(samplers) == 2
    end

    @testset "argument errors" begin
        @test_throws MethodError adjust_scaling_and_samplers(1.0, stadium)               # not an AbsBasisSolver
        @test_throws MethodError adjust_scaling_and_samplers(VerginiSaracenoSolver(2.0, 5.0), 1.0)   # not a billiard
        @test_throws ArgumentError adjust_scaling_and_samplers(
            ParticularSolutionsMethodSolver(2.0, Float64[], 4.0), stadium)               # no scaling factor to take the minimum of
        @test_throws BoundsError adjust_scaling_and_samplers(
            ParticularSolutionsMethodSolver(2.0, [3.0], 4.0, AbsSampler[]), stadium)     # no default sampler
    end
end

@testset "solver type hierarchy" begin
    @test isabstracttype(AbsSolver)
    @test AbsBasisSolver <: AbsSolver && AbsBIMSolver <: AbsSolver
    @test !(AbsBIMSolver <: AbsBasisSolver) && !(AbsBasisSolver <: AbsBIMSolver)
    @test SweepBasisSolver <: AbsBasisSolver && AcceleratedBasisSolver <: AbsBasisSolver
    @test SweepBIMSolver <: AbsBIMSolver && AcceleratedBIMSolver <: AbsBIMSolver
    @test DLP <: SweepBIMSolver && CFIE <: SweepBIMSolver
    @test !(DLP <: CFIE) && !(CFIE <: DLP)
    @test all(isabstracttype, (AbsSolver, AbsBasisSolver, AbsBIMSolver, SweepBasisSolver, AcceleratedBasisSolver,
                               SweepBIMSolver, AcceleratedBIMSolver, DLP, CFIE))
    @test DecompositionMethodSolver <: SweepBasisSolver
    @test ParticularSolutionsMethodSolver <: SweepBasisSolver
    @test VerginiSaracenoSolver <: AcceleratedBasisSolver
    @test DoubleLayerPotentialSolver <: DLP
    @test CombinedFieldIntegralEquationSolver <: CFIE
    @test CompositeBIMSolver <: CFIE                                     # composite solvers reuse the CFIE state recovery
    @test BeynSolver <: AcceleratedBIMSolver
    @test ExpandedBIMSolver <: AcceleratedBIMSolver
    @test CORKSolver <: AcceleratedBIMSolver
    @test !(DoubleLayerPotentialSolver <: AbsBasisSolver)
    @test !(DecompositionMethodSolver <: AbsBIMSolver)
end

@testset "DecompositionMethodSolver" begin
    @testset "defaults of the scalar constructor" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        @test solver isa DecompositionMethodSolver{Float64}
        @test solver.dim_scaling_factor == 2.0
        @test solver.pts_scaling_factor == [3.0]
        @test solver.pts_scaling_factor isa Vector{Float64}
        @test length(solver.sampler) == 1
        @test solver.sampler[1] isa GaussLegendreNodes{Float64}
        @test solver.eps == eps(Float64)
        @test solver.min_dim == 100
        @test solver.min_pts == 500
        @test solver.rellich_origin === SVector(0.0, 0.0)
        @test fieldnames(DecompositionMethodSolver) ==
              (:dim_scaling_factor, :pts_scaling_factor, :sampler, :eps, :min_dim, :min_pts, :rellich_origin)
        @test !ismutabletype(DecompositionMethodSolver)
    end

    @testset "vector scaling factors, samplers and keywords" begin
        solver = DecompositionMethodSolver(2.0, [3.0, 4.0]; min_dim=12, min_pts=64, rellich_origin=SVector(0.5, -0.5))
        @test solver.pts_scaling_factor == [3.0, 4.0]
        @test solver.min_dim == 12 && solver.min_pts == 64
        @test solver.rellich_origin === SVector(0.5, -0.5)

        samplers = AbsSampler[LinearNodes(), GaussLegendreNodes()]
        solver = DecompositionMethodSolver(2.0, 3.0, samplers; min_dim=7)
        @test solver.pts_scaling_factor == [3.0]                       # scalar is wrapped in a vector
        @test solver.sampler === samplers                              # the sampler vector is stored, not copied
        @test solver.min_dim == 7 && solver.min_pts == 500
    end

    @testset "Float32 parameters" begin
        solver = DecompositionMethodSolver(2f0, 3f0; rellich_origin=SVector(1f0, 2f0))
        @test solver isa DecompositionMethodSolver{Float32}
        @test solver.eps == eps(Float32)
        @test solver.pts_scaling_factor isa Vector{Float32}
        @test solver.rellich_origin isa SVector{2,Float32}
        @test DecompositionMethodSolver(2f0, 3f0).rellich_origin === SVector(0f0, 0f0)
    end

    @testset "argument validation" begin
        # Element types must agree. The error is an UndefVarError from the untyped `where T` rather than a MethodError
        # (source finding), so only the rejection itself is asserted.
        @test_throws Exception DecompositionMethodSolver(2.0, 3)
        @test_throws Exception DecompositionMethodSolver(2.0, [3, 4])
        @test_throws Exception DecompositionMethodSolver(2, 3)                           # no eps for an integer type
        @test_throws MethodError DecompositionMethodSolver(2.0)                          # pts_scaling_factor is required
        @test_throws TypeError DecompositionMethodSolver(2.0, 3.0; rellich_origin=[0.5, 0.5])
        @test_throws TypeError DecompositionMethodSolver(2.0, 3.0; rellich_origin=SVector(0.5f0, 0.5f0))
        @test_throws MethodError DecompositionMethodSolver(2.0, 3.0; min_dimension=5)     # unknown keyword
    end

    @testset "SweepBasisSolver contract" begin
        solver = DecompositionMethodSolver(2.0, 3.0)
        S = typeof(solver)
        @test sc_has(evaluate_points, S, sc_RECT_T, Float64)
        @test sc_has(adjust_scaling_and_samplers, S, sc_RECT_T)
        @test sc_has(construct_matrices, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test sc_has(solve, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test sc_has(solve, S, Matrix{Float64}, Matrix{Float64})        # from precomputed F and G
        @test sc_has(solve_vect, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test sc_has(solve_wavenumber, S, sc_BASIS_T, sc_RECT_T, Float64, Float64)
        @test sc_has(k_sweep, S, sc_BASIS_T, sc_RECT_T, Vector{Float64})
        @test sc_has(compute_eigenstate, S, sc_BASIS_T, sc_RECT_T, Float64)
        @test !sc_has(construct_matrices, S, sc_PTS_T, Float64)         # basis solvers need a basis
    end
end

@testset "ParticularSolutionsMethodSolver" begin
    @testset "defaults of the scalar constructor" begin
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 4.0)
        @test solver isa ParticularSolutionsMethodSolver{Float64}
        @test solver.dim_scaling_factor == 2.0
        @test solver.pts_scaling_factor == [3.0]
        @test solver.int_pts_scaling_factor == 4.0
        @test length(solver.sampler) == 1 && solver.sampler[1] isa GaussLegendreNodes{Float64}
        @test solver.eps == eps(Float64)
        @test solver.min_dim == 100
        @test solver.min_pts == 500
        @test solver.min_int_pts == 500
        @test fieldnames(ParticularSolutionsMethodSolver) ==
              (:dim_scaling_factor, :pts_scaling_factor, :int_pts_scaling_factor, :sampler, :eps, :min_dim, :min_pts, :min_int_pts)
        @test !ismutabletype(ParticularSolutionsMethodSolver)
    end

    @testset "vector scaling factors, samplers and keywords" begin
        solver = ParticularSolutionsMethodSolver(2.0, [3.0, 4.0], 5.0; min_dim=12, min_pts=64, min_int_pts=32)
        @test solver.pts_scaling_factor == [3.0, 4.0]
        @test solver.int_pts_scaling_factor == 5.0
        @test (solver.min_dim, solver.min_pts, solver.min_int_pts) == (12, 64, 32)

        samplers = [LinearNodes(), LinearNodes()]                        # a concretely typed sampler vector is accepted here
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 4.0, samplers)
        @test solver.sampler === samplers
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 4.0, AbsSampler[LinearNodes()]; min_int_pts=9)
        @test solver.min_int_pts == 9 && solver.min_dim == 100
    end

    @testset "Float32 parameters" begin
        solver = ParticularSolutionsMethodSolver(2f0, 3f0, 4f0)
        @test solver isa ParticularSolutionsMethodSolver{Float32}
        @test solver.eps == eps(Float32)
        @test solver.int_pts_scaling_factor isa Float32
    end

    @testset "argument validation" begin
        @test_throws MethodError ParticularSolutionsMethodSolver(2.0, 3.0)               # int_pts_scaling_factor is required
        @test_throws Exception ParticularSolutionsMethodSolver(2.0, 3.0, 4)              # element types must agree (UndefVarError, see DM)
        @test_throws Exception ParticularSolutionsMethodSolver(2.0, [3, 4], 4.0)
        @test_throws TypeError ParticularSolutionsMethodSolver(2.0, 3.0, 4.0; min_dim=5.5)   # minimum sizes are typed Int
        @test_throws TypeError ParticularSolutionsMethodSolver(2.0, 3.0, 4.0; min_pts=5.5)
        @test_throws TypeError ParticularSolutionsMethodSolver(2.0, 3.0, 4.0; min_int_pts=5.5)
        @test_throws MethodError ParticularSolutionsMethodSolver(2.0, 3.0, 4.0, 1.0)     # fourth argument must be samplers
    end

    @testset "SweepBasisSolver contract" begin
        solver = ParticularSolutionsMethodSolver(2.0, 3.0, 4.0)
        S = typeof(solver)
        @test sc_has(evaluate_points, S, sc_RECT_T, Float64)
        @test sc_has(adjust_scaling_and_samplers, S, sc_RECT_T)
        @test sc_has(construct_matrices, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test sc_has(solve, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test sc_has(solve_vect, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test sc_has(solve_wavenumber, S, sc_BASIS_T, sc_RECT_T, Float64, Float64)
        @test sc_has(k_sweep, S, sc_BASIS_T, sc_RECT_T, Vector{Float64})
        @test sc_has(compute_eigenstate, S, sc_BASIS_T, sc_RECT_T, Float64)
        @test sc_has(QuantumBilliards.solve_full, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test sc_has(QuantumBilliards.solve_with_rank_reduction, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test !sc_has(solve, S, Matrix{Float64}, Matrix{Float64})        # unlike DM, there is no precomputed-matrix method
    end
end

@testset "VerginiSaracenoSolver" begin
    @testset "defaults of the scalar constructor" begin
        solver = VerginiSaracenoSolver(2.0, 5.0)
        @test solver isa VerginiSaracenoSolver{Float64}
        @test solver.dim_scaling_factor == 2.0
        @test solver.pts_scaling_factor == [5.0]
        @test length(solver.sampler) == 1 && solver.sampler[1] isa GaussLegendreNodes{Float64}
        @test solver.eps == eps(Float64)
        @test solver.min_dim == 100
        @test solver.min_pts == 500
        @test solver.eigenvectors === true
        @test fieldnames(VerginiSaracenoSolver) ==
              (:dim_scaling_factor, :pts_scaling_factor, :sampler, :eps, :min_dim, :min_pts, :eigenvectors)
    end

    @testset "vector scaling factors, samplers and keywords" begin
        solver = VerginiSaracenoSolver(2.0, [5.0, 6.0]; min_dim=10, min_pts=40, eigenvectors=false)
        @test solver.pts_scaling_factor == [5.0, 6.0]
        @test solver.min_dim == 10 && solver.min_pts == 40
        @test solver.eigenvectors === false

        samplers = AbsSampler[LinearNodes(), GaussLegendreNodes()]
        solver = VerginiSaracenoSolver(2.0, 5.0, samplers; eigenvectors=false)
        @test solver.sampler === samplers
        @test solver.eigenvectors === false
    end

    @testset "solver is mutable" begin
        solver = VerginiSaracenoSolver(2.0, 5.0)
        @test ismutabletype(VerginiSaracenoSolver)
        solver.eigenvectors = false
        @test solver.eigenvectors === false
        solver.min_dim = 20
        @test solver.min_dim == 20
    end

    @testset "Float32 parameters" begin
        solver = VerginiSaracenoSolver(2f0, 5f0)
        @test solver isa VerginiSaracenoSolver{Float32}
        @test solver.eps == eps(Float32)
        @test solver.pts_scaling_factor isa Vector{Float32}
    end

    @testset "argument validation" begin
        @test_throws MethodError VerginiSaracenoSolver(2.0)
        @test_throws Exception VerginiSaracenoSolver(2.0, 5)                             # element types must agree (UndefVarError, see DM)
        @test_throws Exception VerginiSaracenoSolver(2, 5)
        @test_throws TypeError VerginiSaracenoSolver(2.0, 5.0; min_dim=5.5)              # minimum sizes are typed Int
        @test_throws TypeError VerginiSaracenoSolver(2.0, 5.0; min_pts=5.5)
        @test_throws TypeError VerginiSaracenoSolver(2.0, 5.0; eigenvectors=1)           # flag is typed Bool
        @test_throws MethodError VerginiSaracenoSolver(2.0, 5.0; rellich_origin=SVector(0.0, 0.0))   # DM-only keyword
    end

    @testset "AcceleratedBasisSolver contract" begin
        solver = VerginiSaracenoSolver(2.0, 5.0)
        S = typeof(solver)
        @test sc_has(evaluate_points, S, sc_RECT_T, Float64)
        @test sc_has(adjust_scaling_and_samplers, S, sc_RECT_T)
        @test sc_has(construct_matrices, S, sc_BASIS_T, sc_PTS_T, Float64)
        @test sc_has(solve, S, sc_BASIS_T, sc_PTS_T, Float64, Float64)
        @test sc_has(solve, S, Matrix{Float64}, Matrix{Float64}, Float64, Float64)       # from precomputed F and Fk
        @test sc_has(solve_vectors, S, sc_BASIS_T, sc_PTS_T, Float64, Float64)
        @test sc_has(solve_wavenumber, S, sc_BASIS_T, sc_RECT_T, Float64, Float64)
        @test sc_has(solve_spectrum, S, sc_BASIS_T, sc_RECT_T, Float64, Float64)
        @test sc_has(compute_eigenstate, S, sc_BASIS_T, sc_RECT_T, Float64)
        @test sc_has(compute_spectrum, S, sc_BASIS_T, sc_RECT_T, Float64, Float64, Float64)
        @test sc_has_kw(compute_spectrum, :ten_tol, S, sc_BASIS_T, sc_RECT_T, Float64, Float64, Float64)   # k1, k2, dk form
        @test sc_has_kw(compute_spectrum, :ten_tol, S, sc_BASIS_T, sc_RECT_T, Int, Int)                    # N1, N2 form
        @test !sc_has(k_sweep, S, sc_BASIS_T, sc_RECT_T, Vector{Float64})              # no sweep for accelerated solvers
        @test !sc_has(solve, S, sc_BASIS_T, sc_PTS_T, Float64)                          # needs the window half-width dk
    end
end

@testset "DoubleLayerPotentialSolver" begin
    circle = CircleBilliard(1.0)
    rect = RectangleBilliard(1.0, 0.7)

    @testset "defaults and type parameters" begin
        solver = DoubleLayerPotentialSolver(10.0, circle)
        @test solver isa DoubleLayerPotentialSolver{Float64,SmoothPeriodicGrading,Nothing,typeof(circle),Tuple{}}
        @test solver isa DLP
        @test solver.pts_scaling_factor == [10.0]
        @test solver.billiard === circle
        @test solver.min_pts == 200
        @test solver.grading isa SmoothPeriodicGrading
        @test solver.symmetry === nothing
        @test solver.sym_characters == ()
        @test solver.eps == 1e-15
        @test fieldnames(DoubleLayerPotentialSolver) ==
              (:pts_scaling_factor, :billiard, :min_pts, :grading, :symmetry, :sym_characters, :eps)
        @test _bim_numeric_type(solver) === Float64
        @test !ismutabletype(DoubleLayerPotentialSolver)
    end

    @testset "vector scaling factors, keywords and grading" begin
        solver = DoubleLayerPotentialSolver([10.0, 12.0], rect; min_pts=64, eps=1e-12, grading=GlobalCornerGrading())
        @test solver.pts_scaling_factor == [10.0, 12.0]
        @test solver.min_pts == 64
        @test solver.eps == 1e-12
        @test solver.grading isa GlobalCornerGrading{Float64}
        @test solver isa DoubleLayerPotentialSolver{Float64,GlobalCornerGrading{Float64},Nothing,typeof(rect),Tuple{}}
    end

    @testset "Float32 scaling factor" begin
        solver = DoubleLayerPotentialSolver(10f0, circle)
        @test solver isa DoubleLayerPotentialSolver{Float32}
        @test solver.eps === 1f-15
        @test _bim_numeric_type(solver) === Float32
        @test solver.billiard === circle                                   # the billiard keeps its own element type
    end

    @testset "symmetry is inferred from sym_characters" begin
        solver = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, -1))
        @test solver.symmetry isa CompositeReflection
        @test length(solver.symmetry.reflections) == 2
        @test solver.sym_characters isa Tuple{ComplexF64,ComplexF64}
        @test sort(real.(collect(solver.sym_characters))) == [-1.0, 1.0]
        @test solver isa DoubleLayerPotentialSolver{Float64,SmoothPeriodicGrading,CompositeReflection,typeof(rect),Tuple{ComplexF64,ComplexF64}}

        sector = symmetry_sector(rect, 1, -1)
        from_sector = DoubleLayerPotentialSolver(10.0, sector; min_pts=64)
        @test from_sector.billiard === rect
        @test from_sector.symmetry isa CompositeReflection
        @test from_sector.sym_characters == solver.sym_characters
        @test from_sector.min_pts == 64                                    # keywords are forwarded through the sector method

        explicit = DoubleLayerPotentialSolver(10.0, rect; symmetry=XAxisReflection(), sym_characters=(1,))
        @test explicit.symmetry isa XAxisReflection                        # an explicit symmetry is kept as given
        @test explicit.sym_characters == (1,)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1,))        # one character per generator
        @test_throws ArgumentError DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, 1, 1))
        @test_throws ArgumentError DoubleLayerPotentialSolver(10.0, rect; sym_characters=(2, 1))      # reflection characters are +-1
        @test_throws TypeError DoubleLayerPotentialSolver(10.0, rect; grading=3)
        @test_throws TypeError DoubleLayerPotentialSolver(10f0, circle; eps=1e-15)                    # eps must have the solver type
        @test_throws TypeError DoubleLayerPotentialSolver(10.0, circle; min_pts=2.5)
        @test_throws InexactError DoubleLayerPotentialSolver(10, circle)                               # eps default 1e-15 is not an Int
        @test_throws MethodError DoubleLayerPotentialSolver(10.0)                                      # the billiard is required
        @test_throws MethodError DoubleLayerPotentialSolver(circle)
        @test_throws MethodError DoubleLayerPotentialSolver(10.0, 1.0)
    end

    @testset "SweepBIMSolver contract" begin
        solver = DoubleLayerPotentialSolver(10.0, circle)
        S = typeof(solver)
        @test sc_has(evaluate_points, S, typeof(circle), Float64)
        @test sc_has(boundary_matrix_size, S, sc_PTS_T)
        @test sc_has(construct_matrices, S, sc_PTS_T, Float64)
        @test sc_has(solve, S, sc_PTS_T, Float64)
        @test sc_has(solve_vect, S, sc_PTS_T, Float64)
        @test sc_has(solve_wavenumber, S, typeof(circle), Float64, Float64)
        @test sc_has(k_sweep, S, typeof(circle), Vector{Float64})
        @test sc_has(solve_state, S, sc_PTS_T, Float64, typeof(circle))
        @test sc_has(compute_eigenstate, S, typeof(circle), Float64)
        @test !sc_has(construct_matrices, S, sc_BASIS_T, sc_PTS_T, Float64)    # no basis is involved
    end
end

@testset "CombinedFieldIntegralEquationSolver" begin
    circle = CircleBilliard(1.0)
    rect = RectangleBilliard(1.0, 0.7)

    @testset "defaults and type parameters" begin
        solver = CombinedFieldIntegralEquationSolver(10.0, circle)
        @test solver isa CombinedFieldIntegralEquationSolver{Float64,SmoothPeriodicGrading,Nothing,typeof(circle),Tuple{}}
        @test solver isa CFIE
        @test !(solver isa DLP)
        @test solver.pts_scaling_factor == [10.0]
        @test solver.billiard === circle
        @test solver.min_pts == 200
        @test solver.grading isa SmoothPeriodicGrading
        @test solver.symmetry === nothing
        @test solver.sym_characters == ()
        @test solver.eps == 1e-15
        @test fieldnames(CombinedFieldIntegralEquationSolver) ==
              (:pts_scaling_factor, :billiard, :min_pts, :grading, :symmetry, :sym_characters, :eps)
        @test _bim_numeric_type(solver) === Float64
    end

    @testset "keywords, Float32 and symmetry inference" begin
        solver = CombinedFieldIntegralEquationSolver([8.0], rect; min_pts=32, grading=GlobalCornerGrading(), sym_characters=(-1, 1))
        @test solver.min_pts == 32
        @test solver.grading isa GlobalCornerGrading{Float64}
        @test solver.symmetry isa CompositeReflection
        @test sort(real.(collect(solver.sym_characters))) == [-1.0, 1.0]

        @test CombinedFieldIntegralEquationSolver(10f0, circle) isa CombinedFieldIntegralEquationSolver{Float32}

        from_sector = CombinedFieldIntegralEquationSolver(10.0, symmetry_sector(rect, 1, 1))
        @test from_sector.billiard === rect
        @test from_sector.symmetry isa CompositeReflection
    end

    @testset "argument validation" begin
        @test_throws ArgumentError CombinedFieldIntegralEquationSolver(10.0, rect; sym_characters=(1,))
        @test_throws ArgumentError CombinedFieldIntegralEquationSolver(10.0, rect; sym_characters=(0, 1))
        @test_throws TypeError CombinedFieldIntegralEquationSolver(10.0, rect; grading="corner")
        @test_throws TypeError CombinedFieldIntegralEquationSolver(10f0, circle; eps=1e-15)
        @test_throws MethodError CombinedFieldIntegralEquationSolver(10.0)
    end

    @testset "SweepBIMSolver contract" begin
        solver = CombinedFieldIntegralEquationSolver(10.0, circle)
        S = typeof(solver)
        @test sc_has(evaluate_points, S, typeof(circle), Float64)
        @test sc_has(boundary_matrix_size, S, sc_PTS_T)
        @test sc_has(construct_matrices, S, sc_PTS_T, Float64)
        @test sc_has(solve, S, sc_PTS_T, Float64)
        @test sc_has(solve_vect, S, sc_PTS_T, Float64)
        @test sc_has(solve_wavenumber, S, typeof(circle), Float64, Float64)
        @test sc_has(k_sweep, S, typeof(circle), Vector{Float64})
        @test sc_has(solve_state, S, sc_PTS_T, Float64, typeof(circle))
        @test sc_has(compute_eigenstate, S, typeof(circle), Float64)
    end
end

@testset "CompositeBIMSolver" begin
    annulus = AnnularBilliard(0.3, 0.0)
    rect = RectangleBilliard(1.0, 0.7)

    @testset "fields and type parameters" begin
        outer = DoubleLayerPotentialSolver(10.0, annulus)
        inner = CombinedFieldIntegralEquationSolver(8.0, annulus)
        solver = CompositeBIMSolver(outer, inner)
        @test solver isa CompositeBIMSolver{Float64,typeof((outer, inner)),Nothing,Tuple{}}
        @test solver isa CFIE
        @test solver.component_solvers === (outer, inner)                   # stored in the given order, DLP and CFIE may be mixed
        @test solver.symmetry === nothing
        @test solver.sym_characters == ()
        @test fieldnames(CompositeBIMSolver) == (:component_solvers, :symmetry, :sym_characters)
        @test _bim_numeric_type(solver) === Float64
    end

    @testset "single component and common symmetry" begin
        one_component = CompositeBIMSolver(DoubleLayerPotentialSolver(10.0, annulus))
        @test length(one_component.component_solvers) == 1

        first_solver = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, 1))
        second_solver = DoubleLayerPotentialSolver(10.0, rect; symmetry=first_solver.symmetry, sym_characters=first_solver.sym_characters)
        solver = CompositeBIMSolver(first_solver, second_solver)
        @test solver.symmetry === first_solver.symmetry                     # the symmetry of the first component is shared
        @test solver.sym_characters == first_solver.sym_characters
    end

    @testset "numeric type follows the first component" begin
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(10f0, annulus))
        @test _bim_numeric_type(solver) === Float32
        @test solver isa CompositeBIMSolver{Float32}
    end

    @testset "argument validation" begin
        @test_throws ArgumentError CompositeBIMSolver()                                       # at least one component
        @test_throws MethodError CompositeBIMSolver(1.0)                                      # components must be SweepBIMSolvers
        @test_throws MethodError CompositeBIMSolver(DecompositionMethodSolver(2.0, 3.0))
        @test_throws MethodError CompositeBIMSolver(BeynSolver(DoubleLayerPotentialSolver(10.0, annulus)))
        symmetric = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, 1))
        plain = DoubleLayerPotentialSolver(10.0, rect)
        @test_throws ArgumentError CompositeBIMSolver(symmetric, plain)                        # symmetry and characters must agree
        @test_throws ArgumentError CompositeBIMSolver(plain, symmetric)
        other_character = DoubleLayerPotentialSolver(10.0, rect; symmetry=symmetric.symmetry, sym_characters=(1.0 + 0im, -1.0 + 0im))
        @test_throws ArgumentError CompositeBIMSolver(symmetric, other_character)
    end

    @testset "independently built components with the same sym_characters" begin
        a = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, 1))
        b = DoubleLayerPotentialSolver(10.0, rect; sym_characters=(1, 1))
        @test a.sym_characters == b.sym_characters
        @test a.symmetry == b.symmetry
        @test (CompositeBIMSolver(a, b); true)
    end

    @testset "SweepBIMSolver contract" begin
        solver = CompositeBIMSolver(DoubleLayerPotentialSolver(10.0, annulus))
        S = typeof(solver)
        @test sc_has(evaluate_points, S, typeof(annulus), Float64)
        @test sc_has(boundary_matrix_size, S, sc_PTS_T)
        @test sc_has(construct_matrices, S, sc_PTS_T, Float64)
        @test sc_has(solve, S, sc_PTS_T, Float64)
        @test sc_has(solve_vect, S, sc_PTS_T, Float64)
        @test sc_has(solve_wavenumber, S, typeof(annulus), Float64, Float64)
        @test sc_has(k_sweep, S, typeof(annulus), Vector{Float64})
        @test sc_has(solve_state, S, sc_PTS_T, Float64, typeof(annulus))
        @test sc_has(compute_eigenstate, S, typeof(annulus), Float64)
    end
end

@testset "BeynSolver" begin
    circle = CircleBilliard(1.0)
    kernel = DoubleLayerPotentialSolver(10.0, circle)

    @testset "defaults and type parameters" begin
        solver = BeynSolver(kernel)
        @test solver isa BeynSolver{Float64,typeof(kernel)}
        @test solver isa AcceleratedBIMSolver
        @test solver.kernel === kernel
        @test solver.m == 100
        @test solver.nq == 40
        @test solver.r == 200
        @test solver.Rmax == 0.5
        @test solver.svd_tol == 1e-11
        @test solver.res_tol == 1e-8
        @test solver.auto_discard_spurious === true
        @test solver.use_taylor === true
        @test solver.taylor_degree == 14
        @test solver.taylor_tol == 1e-12
        @test solver.use_chebyshev === false
        @test solver.imag_k_check === true
        @test solver.imag_k_pad == 20
        @test solver.imag_k_group_size == 20
        @test solver.eigenvectors === true
        @test _bim_numeric_type(solver) === Float64
        @test ismutabletype(BeynSolver)
    end

    @testset "Chebyshev configuration" begin
        solver = BeynSolver(kernel)
        @test solver.cheb_config isa ChebyshevConfig{Float64}
        @test (solver.cheb_config.n_panels_h, solver.cheb_config.M_h) == (10000, 5)
        @test (solver.cheb_config.n_panels_j, solver.cheb_config.M_j) == (5000, 5)

        solver = BeynSolver(kernel; n_panels_h=100, M_h=4, n_panels_j=50, M_j=3)
        @test (solver.cheb_config.n_panels_h, solver.cheb_config.M_h, solver.cheb_config.n_panels_j, solver.cheb_config.M_j) ==
              (100, 4, 50, 3)

        cfg = ChebyshevConfig(Float64; n_panels_h=7)
        solver = BeynSolver(kernel; cheb_config=cfg, n_panels_h=100)         # an explicit configuration wins over the panel keywords
        @test solver.cheb_config === cfg
    end

    @testset "keywords are converted to the kernel type" begin
        solver = BeynSolver(kernel; m=10, nq=16, r=8, Rmax=1, svd_tol=1e-10, res_tol=1e-6, use_taylor=false, use_chebyshev=true,
                            taylor_degree=8, imag_k_check=false, eigenvectors=false)
        @test (solver.m, solver.nq, solver.r) == (10, 16, 8)
        @test solver.Rmax === 1.0
        @test solver.svd_tol === 1e-10 && solver.res_tol === 1e-6
        @test solver.use_taylor === false && solver.use_chebyshev === true
        @test solver.taylor_degree == 8
        @test solver.imag_k_check === false && solver.eigenvectors === false

        solver32 = BeynSolver(DoubleLayerPotentialSolver(10f0, circle))
        @test solver32 isa BeynSolver{Float32}
        @test solver32.Rmax === 0.5f0
        @test solver32.cheb_config isa ChebyshevConfig{Float32}
    end

    @testset "every sweep kernel is accepted" begin
        annulus = AnnularBilliard(0.3, 0.0)
        cfie = CombinedFieldIntegralEquationSolver(10.0, circle)
        composite = CompositeBIMSolver(DoubleLayerPotentialSolver(10.0, annulus))
        @test BeynSolver(cfie) isa BeynSolver{Float64,typeof(cfie)}
        @test BeynSolver(composite) isa BeynSolver{Float64,typeof(composite)}
    end

    @testset "argument validation" begin
        @test_throws MethodError BeynSolver(1.0)                                              # the kernel must be a SweepBIMSolver
        @test_throws MethodError BeynSolver(DecompositionMethodSolver(2.0, 3.0))
        @test_throws MethodError BeynSolver(BeynSolver(kernel))                               # no nesting of accelerated solvers
        @test_throws MethodError BeynSolver()
        @test_throws TypeError BeynSolver(kernel; m=2.5)
        @test_throws TypeError BeynSolver(kernel; nq=2.5)
        @test_throws TypeError BeynSolver(kernel; use_taylor=1)
        @test_throws MethodError BeynSolver(kernel; cheb_config=ChebyshevConfig(Float32))   # wrong element type of the configuration
        @test_throws MethodError BeynSolver(kernel; Rmin=0.1)
    end

    @testset "AcceleratedBIMSolver contract" begin
        solver = BeynSolver(kernel)
        S = typeof(solver)
        @test sc_has(evaluate_points, S, typeof(circle), Float64)
        @test sc_has(construct_matrices, S, sc_PTS_T, Float64, Float64)
        @test sc_has(solve, S, sc_PTS_T, Float64, Float64)
        @test sc_has(solve_vectors, S, sc_PTS_T, Float64, Float64)
        @test sc_has(solve_wavenumber, S, typeof(circle), Float64, Float64)
        @test sc_has(solve_spectrum, S, typeof(circle), Float64, Float64)
        @test sc_has(compute_spectrum, S, typeof(circle), Float64, Float64)
        @test sc_has(compute_spectrum, S, typeof(circle), Int, Int)
        @test sc_has_kw(compute_spectrum, :ten_tol, S, typeof(circle), Float64, Float64)
        @test sc_has_kw(compute_spectrum, :ten_tol, S, typeof(circle), Int, Int)
        @test sc_has(compute_eigenstate, S, typeof(circle), Float64)
        @test !sc_has(k_sweep, S, typeof(circle), Vector{Float64})
    end
end

@testset "ExpandedBIMSolver" begin
    circle = CircleBilliard(1.0)
    kernel = DoubleLayerPotentialSolver(10.0, circle)

    @testset "defaults and type parameters" begin
        solver = ExpandedBIMSolver(kernel)
        @test solver isa ExpandedBIMSolver{Float64,typeof(kernel)}
        @test solver isa AcceleratedBIMSolver
        @test solver.kernel === kernel
        @test solver.use_taylor === true
        @test solver.taylor_degree == 16
        @test solver.taylor_radius == 0.5
        @test solver.taylor_tol == 1e-11
        @test solver.tol == 1e-12
        @test solver.maxiter == 5000
        @test solver.krylovdim == 40
        @test fieldnames(ExpandedBIMSolver) ==
              (:kernel, :use_taylor, :taylor_degree, :taylor_radius, :taylor_tol, :tol, :maxiter, :krylovdim)
        @test _bim_numeric_type(solver) === Float64
        @test !ismutabletype(ExpandedBIMSolver)
    end

    @testset "keywords are converted to the kernel type" begin
        solver = ExpandedBIMSolver(kernel; use_taylor=false, taylor_degree=8, taylor_radius=1, taylor_tol=1e-9, tol=1e-8, maxiter=10, krylovdim=2)
        @test solver.use_taylor === false
        @test solver.taylor_degree == 8
        @test solver.taylor_radius === 1.0
        @test solver.taylor_tol === 1e-9 && solver.tol === 1e-8
        @test (solver.maxiter, solver.krylovdim) == (10, 2)

        solver32 = ExpandedBIMSolver(DoubleLayerPotentialSolver(10f0, circle))
        @test solver32 isa ExpandedBIMSolver{Float32}
        @test solver32.taylor_radius === 0.5f0
        @test solver32.tol === 1f-12
    end

    @testset "argument validation" begin
        @test_throws ArgumentError ExpandedBIMSolver(kernel; maxiter=0)
        @test_throws ArgumentError ExpandedBIMSolver(kernel; maxiter=-3)
        @test_throws ArgumentError ExpandedBIMSolver(kernel; krylovdim=1)
        @test_throws ArgumentError ExpandedBIMSolver(kernel; krylovdim=0)
        @test ExpandedBIMSolver(kernel; krylovdim=2).krylovdim == 2                         # smallest accepted value
        @test ExpandedBIMSolver(kernel; maxiter=1).maxiter == 1
        @test_throws MethodError ExpandedBIMSolver(1.0)
        @test_throws MethodError ExpandedBIMSolver(BeynSolver(kernel))
        @test_throws TypeError ExpandedBIMSolver(kernel; taylor_degree=2.5)
        @test_throws MethodError ExpandedBIMSolver(kernel; nlevels=5)                       # nlevels belongs to solve, not to the solver
    end

    @testset "AcceleratedBIMSolver contract" begin
        solver = ExpandedBIMSolver(kernel)
        S = typeof(solver)
        @test sc_has(evaluate_points, S, typeof(circle), Float64)
        @test sc_has(construct_matrices, S, sc_PTS_T, Float64)
        @test sc_has(solve, S, sc_PTS_T, Float64, Int)                                      # (pts, k, nlevels)
        @test sc_has(solve_wavenumber, S, typeof(circle), Float64, Float64)
        @test sc_has(solve_spectrum, S, typeof(circle), Float64, Float64)
        @test sc_has(compute_spectrum, S, typeof(circle), Float64, Float64)
        @test sc_has(compute_spectrum, S, typeof(circle), Int, Int)
        @test sc_has_kw(compute_spectrum, :ten_tol, S, typeof(circle), Float64, Float64)
        @test sc_has_kw(compute_spectrum, :ten_tol, S, typeof(circle), Int, Int)
        @test sc_has(compute_eigenstate, S, typeof(circle), Float64)
        @test !sc_has(solve_vectors, S, sc_PTS_T, Float64, Float64)                         # EBIM returns no eigenvectors
    end
end

@testset "CORKSolver" begin
    circle = CircleBilliard(1.0)
    kernel = DoubleLayerPotentialSolver(10.0, circle)

    @testset "defaults and type parameters" begin
        solver = CORKSolver(kernel)
        @test solver isa CORKSolver{Float64,typeof(kernel)}
        @test solver isa AcceleratedBIMSolver
        @test solver.kernel === kernel
        @test solver.p == 12
        @test solver.guard == 0.05
        @test solver.nlevels == 100
        @test solver.Rmax == 0.6
        @test solver.b == 10
        @test solver.mstart == 500                                          # b*ceil(5 nlevels/b)
        @test solver.mstep == 50                                            # b*ceil(50/b)
        @test solver.stable_checks == 1
        @test solver.imag_tol == 1e-4
        @test solver.edge_tol == 1e-8
        @test solver.res_tol == 1e-10
        @test solver.stable_tol == 1e-9
        @test solver.imag_search_tol == 1e-4
        @test solver.eigenvectors === false
        @test solver.validate === false
        @test solver.verbose === false
        @test solver.taylor_tol == 3e-10
        @test _bim_numeric_type(solver) === Float64
        @test ismutabletype(CORKSolver)
    end

    @testset "Krylov dimensions follow nlevels and b" begin
        solver = CORKSolver(kernel; nlevels=20)
        @test solver.mstart == 100
        solver = CORKSolver(kernel; nlevels=21)
        @test solver.mstart == 110                                          # rounded up to a multiple of b
        solver = CORKSolver(kernel; b=5, nlevels=10)
        @test (solver.b, solver.mstart, solver.mstep) == (5, 50, 50)
        solver = CORKSolver(kernel; mstart=30, mstep=20)
        @test (solver.mstart, solver.mstep) == (30, 20)
    end

    @testset "argument validation" begin
        @test_throws ArgumentError CORKSolver(kernel; mstart=15)                             # not divisible by b = 10
        @test_throws ArgumentError CORKSolver(kernel; mstep=15)
        @test_throws ArgumentError CORKSolver(kernel; b=4, mstart=30)
        @test_throws ArgumentError CORKSolver(DoubleLayerPotentialSolver(10f0, circle))      # Float64 kernels only
        @test_throws MethodError CORKSolver(1.0)
        @test_throws MethodError CORKSolver(BeynSolver(kernel))
        @test_throws TypeError CORKSolver(kernel; p=2.5)
        @test_throws TypeError CORKSolver(kernel; validate=1)
        @test_throws MethodError CORKSolver(kernel; krylov=3)
    end

    @testset "AcceleratedBIMSolver contract" begin
        solver = CORKSolver(kernel)
        S = typeof(solver)
        @test sc_has(evaluate_points, S, typeof(circle), Float64)
        @test sc_has(solve, S, sc_PTS_T, Float64, Float64)
        @test sc_has(solve_vectors, S, sc_PTS_T, Float64, Float64)
        @test sc_has(solve_wavenumber, S, typeof(circle), Float64, Float64)
        @test sc_has(solve_spectrum, S, typeof(circle), Float64, Float64)
        @test sc_has(compute_spectrum, S, typeof(circle), Float64, Float64)
        @test sc_has(compute_spectrum, S, typeof(circle), Int, Int)
        @test sc_has_kw(compute_spectrum, :ten_tol, S, typeof(circle), Float64, Float64)
        @test sc_has_kw(compute_spectrum, :ten_tol, S, typeof(circle), Int, Int)
        @test sc_has(compute_eigenstate, S, typeof(circle), Float64)
        # CORK assembles its Chebyshev polynomial internally and does not define `construct_matrices`.
        @test !sc_has(construct_matrices, S, sc_PTS_T, Float64)
    end
end
