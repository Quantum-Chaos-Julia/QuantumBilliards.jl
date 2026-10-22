# Unit tests for
#   src/states/eigenstates.jl   (BasisEigenstate and its two constructors, BIMEigenstate and its constructor,
#                                the four compute_eigenstate methods: SweepBasisSolver, AcceleratedBasisSolver,
#                                SweepBIMSolver, AcceleratedBIMSolver)
#   src/states/basisstates.jl   (BasisState and its constructor)
# The enclosing SpectralData (its indexing, filtering, merging) is covered in spectra/spectralutils_tests.jl; here it
# is only checked that compute_eigenstate wraps exactly one state in a length-1 SpectralData with the right fields.
#
# No eigen solves. compute_eigenstate is exercised through test-local solver stubs that implement only the extension
# points the method calls (evaluate_points, solve_vect / solve_vectors / solve_state / solve_wavenumber) and return
# hand-chosen numbers, so every assertion is about the wiring of compute_eigenstate:
#   * the basis dimension rule  dim = max(min_dim, round(Int, L k s / 2 pi))  with L the length of the physical boundary
#     of the fundamental domain of the unit circle, L = pi/2 (the quarter arc; the two radii are symmetry walls and are
#     not part of get_boundary_curves),
#   * the nearest-candidate rule of the accelerated basis method (candidates are k + dk * (-0.5, 0.2, 0.9), the middle
#     one is nearest and its tension / vector column must be selected, k_basis stays the requested k),
#   * the accelerated BIM method delegating to its kernel at the refined wavenumber.
# Boundary data is a hand-built BoundaryPoints (8 points on the unit circle); it is never used numerically.
# Helper definitions use the prefix `es_` / `ES_`.

using StaticArrays
using QuantumBilliards: BasisState

const ES_CIRCLE = CircleBilliard(1.0)
const ES_PTS = BoundaryPoints([SVector(cos(t), sin(t)) for t in range(0, 2pi, 9)[1:8]])
const ES_BASIS = FourierBessel(2; sym_x=1, sym_y=1)
const ES_L = pi / 2         # physical boundary length of the circle fundamental domain (quarter arc)

# Stub solvers: only the extension points called by compute_eigenstate are implemented.
struct es_SweepBasis <: QuantumBilliards.SweepBasisSolver
    min_dim::Int
    dim_scaling_factor::Float64
    ten::Float64
end
QuantumBilliards.evaluate_points(::es_SweepBasis, ::AbsBilliard, k) = ES_PTS
QuantumBilliards.solve_vect(s::es_SweepBasis, basis::AbsBasis, pts, k; multithreaded::Bool=true) =
    (s.ten, collect(1.0:basis.dim))

struct es_AccBasis <: QuantumBilliards.AcceleratedBasisSolver
    min_dim::Int
    dim_scaling_factor::Float64
end
QuantumBilliards.evaluate_points(::es_AccBasis, ::AbsBilliard, k) = ES_PTS
function QuantumBilliards.solve_vectors(::es_AccBasis, basis::AbsBasis, pts, k, dk; multithreaded::Bool=true)
    ks = k .+ dk .* [-0.5, 0.2, 0.9]
    tens = [0.3, 0.1, 0.2]
    X = hcat((Float64(i) .* ones(basis.dim) for i in 1:3)...)
    return ks, tens, X
end

struct es_SweepBIM <: QuantumBilliards.SweepBIMSolver end
QuantumBilliards.evaluate_points(::es_SweepBIM, ::AbsBilliard, k) = ES_PTS
QuantumBilliards.solve_state(::es_SweepBIM, pts::BoundaryPoints, k, billiard; multithreaded::Bool=true) =
    (0.25, collect(1.0:length(pts)), fill(-1.0, length(pts)), 2.5)

struct es_AccBIM <: QuantumBilliards.AcceleratedBIMSolver
    kernel::es_SweepBIM
end
QuantumBilliards.solve_wavenumber(::es_AccBIM, ::AbsBilliard, k, dk; multithreaded::Bool=true) = (k + 0.5 * dk, 1e-9)

@testset "BasisEigenstate struct" begin
    st = BasisEigenstate(2.5, [1.0, 2.0], 0.1, ES_BASIS)
    @test st isa BasisEigenstate{Float64,Float64,typeof(ES_BASIS)}
    @test st isa QuantumBilliards.AbsState
    @test fieldnames(BasisEigenstate) == (:k, :k_basis, :vec, :ten, :basis)
    @test isconcretetype(typeof(st))
    @test !isabstracttype(BasisEigenstate)
    @test st.basis === ES_BASIS
end

@testset "BasisEigenstate constructor - k_basis defaults to k" begin
    vec = [1.0, -2.0, 0.5]
    st = BasisEigenstate(7.25, vec, 3e-4, ES_BASIS)
    @test st.k == 7.25
    @test st.k_basis == 7.25
    @test st.vec == vec
    @test st.ten == 3e-4
    @test st.basis === ES_BASIS
    @test st.vec isa Vector{Float64}
    @test st.ten isa Float64

    @testset "Float32 wavenumber and tension" begin
        st32 = BasisEigenstate(1.5f0, Float32[1, 2], 0.25f0, ES_BASIS)
        @test st32 isa BasisEigenstate{Float32,Float32,typeof(ES_BASIS)}
        @test st32.k === 1.5f0
        @test st32.k_basis === 1.5f0
        @test st32.ten === 0.25f0
    end
end

@testset "BasisEigenstate constructor - explicit k_basis" begin
    st = BasisEigenstate(7.25, 7.0, [1.0, 2.0], 0.1, ES_BASIS)
    @test st.k == 7.25
    @test st.k_basis == 7.0
    @test st.vec == [1.0, 2.0]
    @test st.ten == 0.1
    @test st.basis === ES_BASIS

    # k_basis equal to k reproduces the four-argument constructor field by field
    a = BasisEigenstate(3.0, [1.0, 2.0], 0.1, ES_BASIS)
    b = BasisEigenstate(3.0, 3.0, [1.0, 2.0], 0.1, ES_BASIS)
    @test (a.k, a.k_basis, a.vec, a.ten) == (b.k, b.k_basis, b.vec, b.ten)
end

@testset "BasisEigenstate coefficient filtering" begin
    @testset "real coefficients: |v| <= precision is zeroed" begin
        # precision for Float64 is 1e-16 (set_precision); the comparison is strict, so |v| == 1e-16 is zeroed
        vec = [1.0, 1e-17, -1e-17, 1e-16, 2e-16, -3e-16, 0.5]
        for st in (BasisEigenstate(2.0, vec, 0.1, ES_BASIS), BasisEigenstate(2.0, 2.0, vec, 0.1, ES_BASIS))
            @test st.vec == [1.0, 0.0, 0.0, 0.0, 2e-16, -3e-16, 0.5]
            @test st.vec isa Vector{Float64}
        end
    end

    @testset "input vector is not modified" begin
        vec = [1.0, 1e-17, 2.0]
        copy_before = copy(vec)
        st = BasisEigenstate(2.0, vec, 0.1, ES_BASIS)
        @test vec == copy_before
        @test st.vec !== vec
    end

    @testset "Float32 precision is 1e-8" begin
        st = BasisEigenstate(2.0f0, Float32[1, 5f-9, 2f-8, -5f-9], 0.1f0, ES_BASIS)
        @test st.vec == Float32[1, 0, 2f-8, 0]
        @test eltype(st.vec) == Float32
    end

    @testset "complex coefficients are left unfiltered" begin
        vec = ComplexF64[1 + 1im, 1e-20, 0, 2im]
        st = BasisEigenstate(2.0 + 0im, vec, 0.1, ES_BASIS)
        @test st isa BasisEigenstate{ComplexF64,Float64,typeof(ES_BASIS)}
        @test st.vec == vec
        @test st.k == 2.0 + 0im
        @test st.k_basis == 2.0 + 0im
    end
end

@testset "BasisState" begin
    basis = FourierBessel(5)
    @testset "struct" begin
        st = BasisState(basis, 3.0, 2)
        @test st isa BasisState{Float64,Float64,typeof(basis)}
        @test st isa QuantumBilliards.AbsState
        @test fieldnames(BasisState) == (:k, :k_basis, :vec, :idx, :dim, :eps, :basis)
        @test isconcretetype(typeof(st))
    end

    @testset "unit coefficient vector for every index" begin
        for i in 1:basis.dim
            st = BasisState(basis, 3.0, i)
            @test st.vec == [j == i ? 1.0 : 0.0 for j in 1:basis.dim]
            @test st.idx == i
            @test count(!iszero, st.vec) == 1
        end
    end

    @testset "scalar fields" begin
        st = BasisState(basis, 3.0, 4)
        @test st.k == 3.0
        @test st.k_basis == st.k
        @test st.dim == basis.dim == length(st.vec)
        @test st.eps == 1e-16
        @test st.basis === basis
    end

    @testset "Float32 wavenumber" begin
        st = BasisState(basis, 3.0f0, 1)
        @test st isa BasisState{Float32,Float32,typeof(basis)}
        @test st.vec isa Vector{Float32}
        @test st.eps === 1f-8
        @test st.k === 3.0f0
    end

    @testset "symmetry-adapted basis" begin
        sbasis = FourierBessel(3; sym_x=-1, sym_y=1)
        st = BasisState(sbasis, 2.0, 3)
        @test st.dim == 3
        @test st.vec == [0.0, 0.0, 1.0]
        @test st.basis.symmetries === sbasis.symmetries
    end

    @testset "invalid index" begin
        @test_throws BoundsError BasisState(basis, 3.0, 0)
        @test_throws BoundsError BasisState(basis, 3.0, basis.dim + 1)
    end
end

@testset "BIMEigenstate struct" begin
    st = BIMEigenstate(2.5, [1.0, 2.0], 0.1, ES_PTS)
    @test st isa BIMEigenstate{Float64,Float64}
    @test st isa QuantumBilliards.AbsState
    @test fieldnames(BIMEigenstate) == (:k, :k_im, :vec, :ten, :pts, :u, :bnd_norm)
    @test isconcretetype(typeof(st))
end

@testset "BIMEigenstate constructor" begin
    @testset "real wavenumber, density only" begin
        vec = [1.0, 2.0, 3.0]
        st = BIMEigenstate(2.5, vec, 0.1, ES_PTS)
        @test st.k == 2.5
        @test st.k_im == 0.0
        @test st.vec === vec
        @test st.ten == 0.1
        @test st.pts === ES_PTS
        @test st.u === nothing
        @test st.bnd_norm === nothing
    end

    @testset "complex wavenumber is split into k and k_im" begin
        st = BIMEigenstate(2.5 + 0.01im, [1.0], 0.1, ES_PTS)
        @test st.k == 2.5
        @test st.k_im == 0.01
        @test st.k isa Float64 && st.k_im isa Float64
    end

    @testset "integer wavenumber is converted to the real scalar type" begin
        st = BIMEigenstate(3, [1.0], 0.1, ES_PTS)
        @test st.k === 3.0
        @test st.k_im === 0.0
    end

    @testset "boundary normal derivative and Rellich norm" begin
        vec = [1.0, 2.0]; u = [-1.0, -2.0]
        st = BIMEigenstate(2.5, vec, 0.1, ES_PTS; u, bnd_norm=4.0)
        @test st.vec === vec
        @test st.u === u
        @test st.bnd_norm === 4.0
    end

    @testset "complex coefficients" begin
        vec = ComplexF64[1 + 1im, 2]
        st = BIMEigenstate(2.5, vec, 0.1, ES_PTS; u=copy(vec))
        @test st isa BIMEigenstate{ComplexF64,Float64}
        @test st.vec === vec
        @test st.k isa Float64
    end

    @testset "partial state: u only, through the type-parameter constructor" begin
        u = [-1.0, -2.0]
        st = BIMEigenstate{Float64,Float64}(2.5, 0.0, nothing, 0.1, ES_PTS, u, nothing)
        @test st.vec === nothing
        @test st.u === u
    end

    @testset "partial state with vec = nothing through the keyword constructor" begin
        # K is inferred from `u`, else Complex{T}.
        u = [-1.0, -2.0]
        st = BIMEigenstate(2.5, nothing, 0.1, ES_PTS; u)
        @test st.u === u
        @test st.vec === nothing
        @test st isa BIMEigenstate{Float64,Float64}
        st2 = BIMEigenstate(2.5, nothing, 0.1, ES_PTS)
        @test st2.vec === nothing
        @test st2.u === nothing
        @test st2 isa BIMEigenstate{ComplexF64,Float64}
    end
end

@testset "compute_eigenstate - SweepBasisSolver" begin
    k = 10.0
    solver = es_SweepBasis(4, 2.0, 0.25)
    dim = max(solver.min_dim, round(Int, ES_L * k * solver.dim_scaling_factor / (2pi)))
    @test dim > solver.min_dim                    # the scaling rule, not the floor, determines the dimension here

    data = compute_eigenstate(solver, ES_BASIS, ES_CIRCLE, k)
    @test data isa SpectralData
    @test length(data) == 1
    @test data.billiard === ES_CIRCLE
    @test data.solver === solver
    @test data.k == [k]
    @test data.k_im == [0.0]
    @test data.ten == [solver.ten]
    @test data.control == [false]
    st = only(data.states)
    @test st isa BasisEigenstate
    @test st.k == k
    @test st.k_basis == k
    @test st.ten == solver.ten
    @test st.basis.dim == dim
    @test st.vec == collect(1.0:dim)
    @test st.basis.symmetries == ES_BASIS.symmetries
    @test st.basis.sym_qnumbers == ES_BASIS.sym_qnumbers

    @testset "minimum dimension floor" begin
        data_low = compute_eigenstate(solver, ES_BASIS, ES_CIRCLE, 1.0)
        @test round(Int, ES_L * 1.0 * solver.dim_scaling_factor / (2pi)) < solver.min_dim
        @test only(data_low.states).basis.dim == solver.min_dim
        @test length(only(data_low.states).vec) == solver.min_dim
    end

    @testset "dimension grows with the scaling factor" begin
        data_dense = compute_eigenstate(es_SweepBasis(4, 4.0, 0.25), ES_BASIS, ES_CIRCLE, k)
        @test only(data_dense.states).basis.dim == round(Int, ES_L * k * 4.0 / (2pi))
    end

    @testset "multithreaded keyword is accepted" begin
        data_st = compute_eigenstate(solver, ES_BASIS, ES_CIRCLE, k; multithreaded=false)
        @test only(data_st.states).basis.dim == dim
    end
end

@testset "compute_eigenstate - AcceleratedBasisSolver" begin
    k = 10.0
    solver = es_AccBasis(4, 2.0)
    dim = max(solver.min_dim, round(Int, ES_L * k * solver.dim_scaling_factor / (2pi)))

    @testset "default window dk = 0.1: nearest candidate is selected" begin
        data = compute_eigenstate(solver, ES_BASIS, ES_CIRCLE, k)
        @test data isa SpectralData
        @test length(data) == 1
        @test data.billiard === ES_CIRCLE
        @test data.solver === solver
        @test data.k ≈ [k + 0.2 * 0.1]
        @test data.ten == [0.1]
        @test data.control == [false]
        st = only(data.states)
        @test st isa BasisEigenstate
        @test st.k ≈ k + 0.2 * 0.1
        @test st.k_basis == k                     # basis was evaluated at the requested wavenumber
        @test st.ten == 0.1
        @test st.basis.dim == dim
        @test st.vec == fill(2.0, dim)            # column 2 of the candidate matrix
    end

    @testset "dk is forwarded to the solver" begin
        data = compute_eigenstate(solver, ES_BASIS, ES_CIRCLE, k; dk=0.4)
        @test data.k ≈ [k + 0.2 * 0.4]
        @test only(data.states).k_basis == k
    end

    @testset "multithreaded keyword is accepted" begin
        data = compute_eigenstate(solver, ES_BASIS, ES_CIRCLE, k; multithreaded=false)
        @test length(data) == 1
    end
end

@testset "compute_eigenstate - SweepBIMSolver" begin
    k = 4.0
    solver = es_SweepBIM()
    data = compute_eigenstate(solver, ES_CIRCLE, k)
    @test data isa SpectralData
    @test length(data) == 1
    @test data.billiard === ES_CIRCLE
    @test data.solver === solver
    @test data.k == [k]
    @test data.k_im == [0.0]
    @test data.ten == [0.25]
    @test data.control == [false]
    st = only(data.states)
    @test st isa BIMEigenstate{Float64,Float64}
    @test st.k == k
    @test st.k_im == 0.0
    @test st.ten == 0.25
    @test st.pts === ES_PTS
    @test st.vec == collect(1.0:length(ES_PTS))
    @test st.u == fill(-1.0, length(ES_PTS))
    @test st.bnd_norm == 2.5
    @test compute_eigenstate(solver, ES_CIRCLE, k; multithreaded=false) isa SpectralData
end

@testset "compute_eigenstate - AcceleratedBIMSolver" begin
    kernel = es_SweepBIM()
    solver = es_AccBIM(kernel)
    k = 4.0

    @testset "state is computed by the kernel at the refined wavenumber" begin
        data = compute_eigenstate(solver, ES_CIRCLE, k)
        @test data isa SpectralData
        @test length(data) == 1
        @test data.solver === kernel              # the wrapped sweep kernel, not the accelerated wrapper
        @test data.billiard === ES_CIRCLE
        @test data.k ≈ [k + 0.5 * 0.1]
        st = only(data.states)
        @test st isa BIMEigenstate
        @test st.k ≈ k + 0.5 * 0.1
        @test st.vec !== nothing && st.u !== nothing && st.bnd_norm !== nothing   # complete state
    end

    @testset "dk is forwarded to solve_wavenumber" begin
        data = compute_eigenstate(solver, ES_CIRCLE, k; dk=0.3)
        @test data.k ≈ [k + 0.5 * 0.3]
    end
end
