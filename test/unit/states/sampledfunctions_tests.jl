# Unit tests for
#   src/states/sampledfunctions.jl   (the sampled function objects Wavefunction, BoundaryFunction and HusimiFunction:
#                                     constructor validation, field contents, concrete type parameters, iteration as the
#                                     legacy tuples, length and show)
#
# The objects are plain containers, so the references are structural identities and properties:
#  * the inner constructors keep the given arrays (no copies) and fix the type parameters from the arguments,
#  * inconsistent sizes raise DimensionMismatch, a non-finite or non-positive wavenumber raises ArgumentError,
#  * a single object iterates as the tuples returned before the objects existed, (psi, x, y), (u, s, norm) and (H, qs, ps),
#    so destructuring without k keeps working, and has length 3.
# The objects are hand-built from small arrays; no eigen solves and no values pinned beyond the stored inputs.
# Helper definitions use the prefix `sf_`.

const SF_PSI = [1.0 2.0 3.0; 4.0 5.0 6.0]       # 2 x 3, psi[ix, iy]
const SF_X = [0.0, 1.0]
const SF_Y = [0.0, 0.5, 1.0]
const SF_U = [1.0, -2.0, 3.0, -4.0]
const SF_S = [0.0, 1.0, 2.0, 3.0]
const SF_DS = fill(1.0, 4)
const SF_H = [0.1 0.2 0.3; 0.1 0.1 0.2]         # 2 x 3, H[iq, ip]
const SF_QS = [0.0, 1.0]
const SF_PS = [-1.0, 0.0, 1.0]

sf_wavefunction(k = 1.5) = Wavefunction(SF_PSI, SF_X, SF_Y, k)
sf_boundary_function(k = 1.5) = BoundaryFunction(SF_U, SF_S, SF_DS, k, 0.9, 4.0, 1:4)
sf_husimi_function(k = 1.5) = HusimiFunction(SF_H, SF_QS, SF_PS, k)

@testset "Wavefunction" begin
    @testset "fields and type parameters" begin
        wf = sf_wavefunction()
        @test wf isa Wavefunction{Float64,Float64}
        @test wf.psi === SF_PSI && wf.x === SF_X && wf.y === SF_Y
        @test wf.k === 1.5
        @test fieldnames(Wavefunction) == (:psi, :x, :y, :k)
        wc = Wavefunction(complex.(SF_PSI, 1.0), SF_X, SF_Y, 2.0)
        @test wc isa Wavefunction{Float64,ComplexF64}
        @test wc.psi isa Matrix{ComplexF64} && wc.x isa Vector{Float64}
    end

    @testset "concrete type is inferred" begin
        @test @inferred(Wavefunction(SF_PSI, SF_X, SF_Y, 1.5)) isa Wavefunction{Float64,Float64}
        @test isconcretetype(typeof(sf_wavefunction()))
    end

    @testset "Float32 coordinates and wavenumber" begin
        psi32 = Float32.(SF_PSI)
        wf = @inferred Wavefunction(psi32, Float32.(SF_X), Float32.(SF_Y), 1.5f0)
        @test wf isa Wavefunction{Float32,Float32}
        @test wf.k isa Float32 && wf.x isa Vector{Float32} && wf.y isa Vector{Float32}
        wm = Wavefunction(ComplexF64.(SF_PSI), Float32.(SF_X), Float32.(SF_Y), 1.5f0)      # value type independent of T
        @test wm isa Wavefunction{Float32,ComplexF64}
    end

    @testset "strict constructor: one real type for the coordinates and k, concrete array types" begin
        @test_throws MethodError Wavefunction(SF_PSI, SF_X, SF_Y, 1)                       # k of another type than x, y
        @test_throws MethodError Wavefunction(SF_PSI, Float32.(SF_X), SF_Y, 1.5)           # x and y of different types
        @test_throws MethodError Wavefunction(view(SF_PSI, :, :), SF_X, SF_Y, 1.5)         # not a Matrix
        @test_throws MethodError Wavefunction(SF_PSI, view(SF_X, :), SF_Y, 1.5)            # not a Vector
    end

    @testset "DimensionMismatch for an inconsistent psi" begin
        @test_throws DimensionMismatch Wavefunction(SF_PSI, [0.0, 1.0, 2.0], SF_Y, 1.5)    # x too long
        @test_throws DimensionMismatch Wavefunction(SF_PSI, SF_X, [0.0, 1.0], 1.5)         # y too short
        @test_throws DimensionMismatch Wavefunction(SF_PSI, SF_Y, SF_X, 1.5)               # x and y swapped
        @test_throws DimensionMismatch Wavefunction(copy(SF_PSI'), SF_X, SF_Y, 1.5)        # psi transposed
        @test_throws DimensionMismatch Wavefunction(zeros(2, 3), Float64[], SF_Y, 1.5)
    end

    @testset "ArgumentError for an invalid wavenumber" begin
        for k in (0.0, -1.0, NaN, Inf, -Inf)
            @test_throws ArgumentError Wavefunction(SF_PSI, SF_X, SF_Y, k)
        end
        @test_throws ArgumentError Wavefunction(Float32.(SF_PSI), Float32.(SF_X), Float32.(SF_Y), 0.0f0)
        @test_throws ArgumentError Wavefunction(Float32.(SF_PSI), Float32.(SF_X), Float32.(SF_Y), Inf32)
        @test Wavefunction(SF_PSI, SF_X, SF_Y, 1e-12).k == 1e-12                           # any positive finite k is valid
    end

    @testset "iteration yields (psi, x, y); length is 3" begin
        wf = sf_wavefunction()
        @test length(wf) == 3
        parts = collect(wf)
        @test length(parts) == 3
        @test parts[1] === wf.psi && parts[2] === wf.x && parts[3] === wf.y
        psi, x, y = wf
        @test psi === wf.psi && x === wf.x && y === wf.y
        @test Base.iterate(wf, 4) === nothing
        @test first(wf) === wf.psi
        psi2, x2 = wf                                                                      # a prefix can be destructured
        @test psi2 === wf.psi && x2 === wf.x
    end

    @testset "show" begin
        wf = sf_wavefunction()
        s = sprint(show, wf)
        @test occursin("Wavefunction", s) && occursin("1.5", s)
        @test !occursin('\n', s)
        @test sprint(show, MIME("text/plain"), wf) isa String
        @test sprint(show, [wf, wf]) isa String                                            # inside a container
    end
end

@testset "BoundaryFunction" begin
    @testset "fields and type parameters" begin
        bf = sf_boundary_function()
        @test bf isa BoundaryFunction{Float64,Float64}
        @test bf.u === SF_U && bf.s === SF_S && bf.ds === SF_DS
        @test bf.k === 1.5 && bf.norm === 0.9 && bf.L === 4.0
        @test bf.outer === 1:4
        @test fieldnames(BoundaryFunction) == (:u, :s, :ds, :k, :norm, :L, :outer)
        bc = BoundaryFunction(complex.(SF_U, 1.0), SF_S, SF_DS, 1.5, 0.9, 4.0, 1:4)
        @test bc isa BoundaryFunction{Float64,ComplexF64}
        @test bc.u isa Vector{ComplexF64} && bc.s isa Vector{Float64}
    end

    @testset "concrete type is inferred" begin
        @test @inferred(BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9, 4.0, 1:4)) isa BoundaryFunction{Float64,Float64}
        @test isconcretetype(typeof(sf_boundary_function()))
        @test isconcretetype(typeof(BoundaryFunction(complex.(SF_U), SF_S, SF_DS, 1.5, 0.9, 4.0, 1:4)))
    end

    @testset "Float32 coordinates and wavenumber" begin
        args32 = (Float32.(SF_U), Float32.(SF_S), Float32.(SF_DS), 1.5f0, 0.9f0, 4.0f0, 1:4)
        bf = @inferred BoundaryFunction(args32...)
        @test bf isa BoundaryFunction{Float32,Float32}
        @test bf.k isa Float32 && bf.norm isa Float32 && bf.L isa Float32
        @test bf.s isa Vector{Float32} && bf.ds isa Vector{Float32}
        bm = BoundaryFunction(ComplexF32.(SF_U), Float32.(SF_S), Float32.(SF_DS), 1.5f0, 0.9f0, 4.0f0, 1:4)
        @test bm isa BoundaryFunction{Float32,ComplexF32}
    end

    @testset "strict constructor: one real type for s, ds, k, norm, L; outer is a UnitRange{Int}" begin
        @test_throws MethodError BoundaryFunction(SF_U, SF_S, SF_DS, 1, 0.9, 4.0, 1:4)               # k of another type
        @test_throws MethodError BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9f0, 4.0, 1:4)          # norm of another type
        @test_throws MethodError BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9, 4.0f0, 1:4)          # L of another type
        @test_throws MethodError BoundaryFunction(SF_U, SF_S, Float32.(SF_DS), 1.5, 0.9, 4.0, 1:4)  # ds of another type
        @test_throws MethodError BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9, 4.0, 1:2:4)         # not a UnitRange
        @test_throws MethodError BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9, 4.0, [1, 2, 3, 4])
        @test_throws MethodError BoundaryFunction(view(SF_U, :), SF_S, SF_DS, 1.5, 0.9, 4.0, 1:4)   # not a Vector
    end

    @testset "DimensionMismatch for inconsistent lengths" begin
        @test_throws DimensionMismatch BoundaryFunction(SF_U, SF_S[1:3], SF_DS, 1.5, 0.9, 4.0, 1:4)     # s short
        @test_throws DimensionMismatch BoundaryFunction(SF_U, SF_S, SF_DS[1:3], 1.5, 0.9, 4.0, 1:4)     # ds short
        @test_throws DimensionMismatch BoundaryFunction(SF_U[1:3], SF_S, SF_DS, 1.5, 0.9, 4.0, 1:3)     # u short
        @test_throws DimensionMismatch BoundaryFunction(SF_U, [SF_S; 4.0], [SF_DS; 1.0], 1.5, 0.9, 4.0, 1:4)   # u shorter than s, ds
    end

    @testset "DimensionMismatch for outer outside the node indices" begin
        @test_throws DimensionMismatch BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9, 4.0, 1:5)          # past the end
        @test_throws DimensionMismatch BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9, 4.0, 0:3)          # before the start
        @test_throws DimensionMismatch BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9, 4.0, 4:6)
        for outer in (1:4, 1:2, 3:4, 2:3)                                                               # sub-ranges are valid
            @test BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, 0.9, 4.0, outer).outer === outer
        end
    end

    @testset "ArgumentError for an invalid wavenumber" begin
        for k in (0.0, -1.0, NaN, Inf, -Inf)
            @test_throws ArgumentError BoundaryFunction(SF_U, SF_S, SF_DS, k, 0.9, 4.0, 1:4)
        end
        @test_throws ArgumentError BoundaryFunction(Float32.(SF_U), Float32.(SF_S), Float32.(SF_DS), 0.0f0, 0.9f0, 4.0f0, 1:4)
        @test_throws ArgumentError BoundaryFunction(Float32.(SF_U), Float32.(SF_S), Float32.(SF_DS), NaN32, 0.9f0, 4.0f0, 1:4)
    end

    @testset "a NaN norm means no stored Rellich value and is accepted" begin
        bf = BoundaryFunction(SF_U, SF_S, SF_DS, 1.5, NaN, 4.0, 1:4)
        @test isnan(bf.norm)
        _, _, nrm = bf
        @test isnan(nrm)
    end

    @testset "iteration yields (u, s, norm) without k; length is 3" begin
        bf = sf_boundary_function()
        @test length(bf) == 3
        parts = collect(bf)
        @test length(parts) == 3
        @test parts[1] === bf.u && parts[2] === bf.s && parts[3] === bf.norm
        u, s, nrm = bf
        @test u === bf.u && s === bf.s && nrm === bf.norm
        @test Base.iterate(bf, 4) === nothing
        @test first(bf) === bf.u
    end

    @testset "show" begin
        bf = sf_boundary_function()
        s = sprint(show, bf)
        @test occursin("BoundaryFunction", s) && occursin("1.5", s) && occursin("4", s)
        @test !occursin('\n', s)
        @test sprint(show, MIME("text/plain"), bf) isa String
        @test sprint(show, [bf, bf]) isa String
    end
end

@testset "HusimiFunction" begin
    @testset "fields and type parameters" begin
        hf = sf_husimi_function()
        @test hf isa HusimiFunction{Float64}
        @test hf.H === SF_H && hf.qs === SF_QS && hf.ps === SF_PS
        @test hf.k === 1.5
        @test fieldnames(HusimiFunction) == (:H, :qs, :ps, :k)
    end

    @testset "concrete type is inferred" begin
        @test @inferred(HusimiFunction(SF_H, SF_QS, SF_PS, 1.5)) isa HusimiFunction{Float64}
        @test isconcretetype(typeof(sf_husimi_function()))
    end

    @testset "Float32 grids and wavenumber" begin
        hf = @inferred HusimiFunction(Float32.(SF_H), Float32.(SF_QS), Float32.(SF_PS), 1.5f0)
        @test hf isa HusimiFunction{Float32}
        @test hf.H isa Matrix{Float32} && hf.qs isa Vector{Float32} && hf.ps isa Vector{Float32} && hf.k isa Float32
    end

    @testset "strict constructor: one real type for H, qs, ps and k" begin
        @test_throws MethodError HusimiFunction(SF_H, SF_QS, SF_PS, 1)
        @test_throws MethodError HusimiFunction(SF_H, Float32.(SF_QS), SF_PS, 1.5)
        @test_throws MethodError HusimiFunction(Float32.(SF_H), SF_QS, SF_PS, 1.5)
        @test_throws MethodError HusimiFunction(complex.(SF_H), SF_QS, SF_PS, 1.5)               # H is real
        @test_throws MethodError HusimiFunction(view(SF_H, :, :), SF_QS, SF_PS, 1.5)
    end

    @testset "DimensionMismatch for an inconsistent H" begin
        @test_throws DimensionMismatch HusimiFunction(SF_H, [0.0, 1.0, 2.0], SF_PS, 1.5)        # qs too long
        @test_throws DimensionMismatch HusimiFunction(SF_H, SF_QS, [0.0, 1.0], 1.5)             # ps too short
        @test_throws DimensionMismatch HusimiFunction(SF_H, SF_PS, SF_QS, 1.5)                  # qs and ps swapped
        @test_throws DimensionMismatch HusimiFunction(copy(SF_H'), SF_QS, SF_PS, 1.5)           # H transposed
    end

    @testset "ArgumentError for an invalid wavenumber" begin
        for k in (0.0, -1.0, NaN, Inf, -Inf)
            @test_throws ArgumentError HusimiFunction(SF_H, SF_QS, SF_PS, k)
        end
        @test_throws ArgumentError HusimiFunction(Float32.(SF_H), Float32.(SF_QS), Float32.(SF_PS), -2.0f0)
        @test_throws ArgumentError HusimiFunction(Float32.(SF_H), Float32.(SF_QS), Float32.(SF_PS), Inf32)
    end

    @testset "iteration yields (H, qs, ps) without k; length is 3" begin
        hf = sf_husimi_function()
        @test length(hf) == 3
        parts = collect(hf)
        @test length(parts) == 3
        @test parts[1] === hf.H && parts[2] === hf.qs && parts[3] === hf.ps
        H, qs, ps = hf
        @test H === hf.H && qs === hf.qs && ps === hf.ps
        @test Base.iterate(hf, 4) === nothing
        @test first(hf) === hf.H
    end

    @testset "show" begin
        hf = sf_husimi_function()
        s = sprint(show, hf)
        @test occursin("HusimiFunction", s) && occursin("1.5", s)
        @test !occursin('\n', s)
        @test sprint(show, MIME("text/plain"), hf) isa String
        @test sprint(show, [hf, hf]) isa String
    end
end
