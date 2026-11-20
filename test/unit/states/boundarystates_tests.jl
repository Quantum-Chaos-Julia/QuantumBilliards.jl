# Unit tests for
#   src/states/boundarystates.jl   (boundary_states: basis eigenstates -> BIMEigenstate u-states)
# boundary_states turns coefficient states into boundary states that store the Rellich-normalized normal derivative
# u = d_n psi on the (unfolded) boundary and no layer density. States are grouped by (objectid(basis), k_basis); a group
# shares one boundary grid (from the largest k of the group), one gradient-matrix evaluation and one matrix product U = G X.
#
# No eigen solves. States are hand-built with prescribed coefficients. References are
#  * the per-state boundary_function(state, billiard; ppw).u (raw, sign arbitrary), which uses the same grid when the group has one k (or one state),
#    the boundary state u is parallel to it: u = sign * u_raw / sqrt(bnd_norm) with the sign making the largest entry positive,
#  * the analytic gradient of rectangle plane-wave modes, u = grad(psi) . n at the returned points, up to the positive normalization,
#  * the Rellich identity: u is normalized after unfolding, rellich(unfolded u) = 1 and bnd_norm = (number of copies) * the fundamental raw value,
#  * a test-local stub basis whose gradient matrices contain a NaN, for the regularisation.
# Helper definitions use the prefix `bs_` / `BS_`.

using StaticArrays
using LinearAlgebra: dot, norm, qr, Diagonal, I
using Random: MersenneTwister
using QuantumBilliards: _level_clusters, _pts_groups, _complex_state, _adjoint_inverse_iteration, _left_residuals, _left_null_vectors
using QuantumBilliards: _boundary_state, _boundary_states, _normalize_u!, _group_indices
using QuantumBilliards: boundary_states, _rellich, get_boundary_curves_with_ignored, FourierNodes, boundary_coords,
                        _debroglie_grid_size, gradient_matrices

const BS_J01 = 2.404825557695773
const BS_J11 = 3.8317059702075125
const BS_A, BS_B = 1.0, 0.6
const BS_RECT = RectangleBilliard(BS_A, BS_B)
const BS_CIRCLE = CircleBilliard(1.0)

bs_j0_state(basis=FourierBessel(1; sym_x=1, sym_y=1); c=1.0) = BasisEigenstate(BS_J01, [c], 1e-12, basis)

# Boundary-state u expected from the raw reference `ref = boundary_function(state, billiard)`: normalized after unfolding on
# `copies` images (raw unfolded Rellich value = copies * ref.norm), largest-modulus entry positive.
function bs_expected_u(ref; copies=4)
    p = argmax(abs.(ref.u))
    return sign(ref.u[p]) .* ref.u ./ sqrt(copies * ref.norm)
end

# Rectangle mode (m, n) as a one-wave RealPlaneWaves state, with the analytic gradient.
function bs_rect_mode(m::Int, n::Int)
    kx = m * pi / (2BS_A); ky = n * pi / (2BS_B)
    sx = isodd(m) ? 1 : -1; sy = isodd(n) ? 1 : -1
    basis = RealPlaneWaves(1; sym_x=sx, sym_y=sy, angle_arc=0.0, angle_shift=atan(ky, kx))
    fx = isodd(m) ? cos : sin; fy = isodd(n) ? cos : sin
    grad = (x, y) -> SVector(kx * (isodd(m) ? -sin(kx * x) : cos(kx * x)) * fy(ky * y),
                             ky * (isodd(n) ? -sin(ky * y) : cos(ky * y)) * fx(kx * x))
    return BasisEigenstate(hypot(kx, ky), [1.0], 1e-12, basis), grad
end

# Boundary points of a state in the way boundary_states samples them for a group with largest wavenumber k.
function bs_grid(billiard, k; ppw=5.0)
    boundary = get_boundary_curves_with_ignored(billiard)
    N = _debroglie_grid_size(k, CompositeCurve(boundary).length, ppw)
    return boundary_coords(billiard, FourierNodes([2, 3, 5], [crv.length for crv in boundary]), N)
end

# Stub basis: gradient matrices dX = 2 x, dY = y for one column, with a NaN in row `bs_NAN_ROW` of dX.
# u = n . (2x, y) = 1 + cos^2(theta) on the unit circle.
struct bs_StubBasis <: QuantumBilliards.AbsBasis
    dim::Int
    symmetries::Nothing
    sym_characters::Nothing
end
bs_StubBasis() = bs_StubBasis(1, nothing, nothing)
const BS_NAN_ROW = 5
function QuantumBilliards.gradient_matrices(::bs_StubBasis, k, pts::Vector{SVector{2,T}}; multithreaded::Bool=true) where {T<:Real}
    dX = reshape(T[2 * p[1] for p in pts], :, 1)
    dY = reshape(T[p[2] for p in pts], :, 1)
    dX[BS_NAN_ROW, 1] = T(NaN)
    return dX, dY
end

@testset "boundary_states - single states against boundary_function" begin
    @testset "circle J0 (reflection symmetric basis): u equals boundary_function(state).u" begin
        st = bs_j0_state()
        out = boundary_states([st], BS_CIRCLE)
        ref = boundary_function(st, BS_CIRCLE)
        @test out isa Vector{BIMEigenstate{Float64,Float64}} && length(out) == 1
        @test out[1].vec === nothing
        @test out[1].u isa Vector{Float64}
        @test out[1].u ≈ bs_expected_u(ref) rtol = 1e-12
        @test out[1].bnd_norm ≈ 4 * ref.norm rtol = 1e-12                # Rellich value of the raw unfolded u
        @test _rellich(out[1].pts, out[1].u, st.k) ≈ 1 rtol = 1e-12
        @test out[1].u[argmax(abs.(out[1].u))] > 0
        @test out[1].k === st.k && out[1].ten == st.ten
        @test length(out[1].pts.xy) == length(ref.u) == length(out[1].u)
        @test out[1].pts.s == ref.s && out[1].pts.ds == ref.ds
    end

    @testset "basis without symmetries: fundamental boundary only" begin
        basis = FourierBessel(3)
        vec = zeros(basis.dim); vec[findfirst(i -> basis.orders[i] == 0 && basis.trig[i] > 0, 1:basis.dim)] = 1.0
        st = BasisEigenstate(BS_J01, vec, 1e-12, basis)
        out = boundary_states([st], BS_CIRCLE)[1]
        ref = boundary_function(st, BS_CIRCLE)
        @test out.u ≈ bs_expected_u(ref; copies=1) rtol = 1e-12
        @test out.bnd_norm ≈ ref.norm rtol = 1e-12
        @test _rellich(out.pts, out.u, st.k) ≈ 1 rtol = 1e-12
        @test length(out.u) == 513
    end

    @testset "ppw is forwarded and multithreaded does not change the values" begin
        st = bs_j0_state()
        coarse = boundary_states([st], BS_CIRCLE)[1]
        fine = boundary_states([st], BS_CIRCLE; ppw=3000.0)[1]
        @test length(fine.u) > length(coarse.u)
        @test fine.u ≈ bs_expected_u(boundary_function(st, BS_CIRCLE; ppw=3000.0)) rtol = 1e-12
        @test boundary_states([st], BS_CIRCLE; multithreaded=false)[1].u ≈ coarse.u rtol = 1e-12
    end
end

@testset "boundary_states - real reflection symmetry (rectangle modes)" begin
    for (m, n) in ((1, 1), (2, 1), (1, 2), (2, 2))
        st, grad = bs_rect_mode(m, n)
        out = boundary_states([st], BS_RECT)[1]
        pts = out.pts
        @testset "mode ($m, $n)" begin
            @test out.u isa Vector{Float64}                                  # real irrep: real u
            @test eltype(out.u) === Float64 && !any(isnan, out.u)
            @test length(out.u) == length(pts.xy) == length(pts.normal) == length(pts.ds)
            # the unfolded grid is longer than the fundamental one
            @test length(pts.xy) > length(bs_grid(BS_RECT, st.k).xy)
            # u is parallel to grad(psi) . n at the unfolded points with the unfolded normals; normalized to unit Rellich value
            g = [dot(grad(p[1], p[2]), nrm) for (p, nrm) in zip(pts.xy, pts.normal)]
            @test _rellich(pts, out.u, st.k) ≈ 1 rtol = 1e-9
            @test out.bnd_norm ≈ _rellich(pts, g, st.k) rtol = 1e-9          # Rellich value of the unnormalized unfolded u
            @test out.u ≈ sign(g[argmax(abs.(g))]) .* g ./ sqrt(out.bnd_norm) atol = 1e-10
            @test out.u[argmax(abs.(out.u))] > 0
            # the Rellich integrand is reflection invariant: bnd_norm is 4 copies of the fundamental raw value
            ref = boundary_function(st, BS_RECT)
            @test out.bnd_norm ≈ 4 * ref.norm rtol = 1e-9
            @test out.u ≈ bs_expected_u(ref) rtol = 1e-12
        end
    end
end

@testset "boundary_states - grouping by basis and k_basis" begin
    st_a = bs_j0_state()                                   # basis A, k_basis = J01
    basis_a = st_a.basis
    st_a2 = bs_j0_state(basis_a; c=2.0)                    # same basis object, same k_basis: same group as st_a
    st_b = BasisEigenstate(BS_J11, [1.0], 1e-12, FourierBessel(1; sym_x=-1, sym_y=1))   # other basis
    st_a3 = BasisEigenstate(BS_J01, BS_J01 + 0.1, [1.0], 1e-12, basis_a)                # same basis, other k_basis
    states = [st_a, st_b, st_a2, st_a3]
    out = boundary_states(states, BS_CIRCLE)

    @testset "output order and per-state data" begin
        @test length(out) == 4
        @test [o.k for o in out] == [s.k for s in states]
        @test all(o -> o.vec === nothing && o.u isa Vector{Float64} && isfinite(o.bnd_norm), out)
    end

    @testset "states of one group share pts, other groups do not" begin
        @test out[1].pts === out[3].pts
        @test out[1].pts !== out[2].pts
        @test out[1].pts !== out[4].pts
        @test out[2].pts !== out[4].pts
    end

    @testset "batch u equals the single-state result for every group" begin
        for (i, s) in enumerate(states)
            @test out[i].u ≈ bs_expected_u(boundary_function(s, BS_CIRCLE)) rtol = 1e-10
            @test _rellich(out[i].pts, out[i].u, s.k) ≈ 1 rtol = 1e-10
        end
        @test out[3].u ≈ out[1].u rtol = 1e-12                      # normalization removes the coefficient scale
        @test out[3].bnd_norm ≈ 4 * out[1].bnd_norm rtol = 1e-12    # Rellich value of the raw u is quadratic
    end

    @testset "a group uses the grid of its largest wavenumber" begin
        basis = FourierBessel(3)
        vec1 = zeros(basis.dim); vec1[1] = 1.0
        vec2 = zeros(basis.dim); vec2[end] = 1.0
        kb = 5.0
        lo = BasisEigenstate(2.0, kb, vec1, 1e-12, basis)
        hi = BasisEigenstate(10.0, kb, vec2, 1e-12, basis)
        res = boundary_states([lo, hi], BS_CIRCLE)
        @test res[1].pts === res[2].pts
        @test length(res[1].u) == length(bs_grid(BS_CIRCLE, 10.0).xy)
        # the raw u depends only on the grid; the Rellich value scales as 1/k^2, so normalizing at k = 2 instead of 10 scales u by 2/10
        lo_hi = BasisEigenstate(10.0, kb, vec1, 1e-12, basis)
        @test res[1].u ≈ (lo.k / hi.k) .* bs_expected_u(boundary_function(lo_hi, BS_CIRCLE); copies=1) rtol = 1e-10     # basis without symmetries: one copy
        @test res[2].u ≈ bs_expected_u(boundary_function(hi, BS_CIRCLE); copies=1) rtol = 1e-10
    end
end

@testset "boundary_states - NaN entries of the normal derivative are regularised" begin
    basis = bs_StubBasis()
    st = BasisEigenstate(3.0, [1.0], 1e-12, basis)
    out = boundary_states([st], BS_CIRCLE)[1]
    pts = bs_grid(BS_CIRCLE, st.k)
    ref = [p[2]^2 + 2 * p[1]^2 for p in pts.xy]                                    # n . (2x, y) on the unit circle
    @test !any(isnan, out.u) && isfinite(out.bnd_norm)
    @test length(out.u) == length(ref)
    ref[BS_NAN_ROW] = (ref[BS_NAN_ROW-1] + ref[BS_NAN_ROW+1]) / 2                  # regularised entry: mean of the neighbours
    @test out.bnd_norm ≈ _rellich(out.pts, ref, st.k) rtol = 1e-9
    @test out.u ≈ ref ./ sqrt(out.bnd_norm) atol = 1e-12                           # all-positive u: sign +
    @test _rellich(out.pts, out.u, st.k) ≈ 1 rtol = 1e-9
end

@testset "_normalize_u!" begin
    pts = bs_grid(BS_CIRCLE, 5.0)
    k = 5.0

    @testset "real u: unit Rellich value, largest entry positive, returns the unscaled Rellich value" begin
        u = -2.0 .* (1 .+ 0.3 .* cos.(3 .* pts.s))                              # largest modulus entry is negative
        u0 = copy(u)
        r0 = _rellich(pts, u0, k)
        v = _normalize_u!(pts, u, k)
        @test v isa Float64
        @test v ≈ r0 rtol = 1e-12
        @test _rellich(pts, u, k) ≈ 1 rtol = 1e-12
        @test u[argmax(abs.(u))] > 0
        @test u ≈ -u0 ./ sqrt(r0) rtol = 1e-12                                  # rotation by the sign -1, then scaling
    end

    @testset "complex u: phase removed at the largest entry" begin
        u = ComplexF64.(1 .+ 0.3 .* cos.(2 .* pts.s)) .* cis(0.7)
        u0 = copy(u)
        r0 = _rellich(pts, u0, k)
        v = _normalize_u!(pts, u, k)
        p = argmax(abs.(u0))
        @test v ≈ r0 rtol = 1e-12
        @test _rellich(pts, u, k) ≈ 1 rtol = 1e-12
        @test imag(u[p]) ≈ 0 atol = 1e-12
        @test real(u[p]) > 0
        @test u ≈ u0 .* (conj(u0[p]) / abs(u0[p])) ./ sqrt(r0) rtol = 1e-12
    end

    @testset "already normalized u is a fixed point" begin
        u = 1 .+ 0.3 .* cos.(3 .* pts.s)
        _normalize_u!(pts, u, k)
        u1 = copy(u)
        @test _normalize_u!(pts, u, k) ≈ 1 rtol = 1e-12
        @test u ≈ u1 rtol = 1e-12
    end
end

@testset "_group_indices" begin
    @test _group_indices(identity, [3, 1, 3, 2, 1, 3]) == [[1, 3, 6], [2, 5], [4]]   # first-occurrence order, increasing indices
    @test _group_indices(isodd, [2, 4, 1, 6, 3]) == [[1, 2, 4], [3, 5]]
    @test _group_indices(identity, [5, 5, 5]) == [[1, 2, 3]]
    @test _group_indices(identity, [1, 2, 3]) == [[1], [2], [3]]
    @test isempty(_group_indices(identity, Int[]))
    @test _group_indices(x -> x[1], [(:a, 1), (:b, 2), (:a, 3)]) == [[1, 3], [2]]
    @test _group_indices(identity, view([4, 7, 4, 7], 1:3)) == [[1, 3], [2]]         # works on views
end

@testset "boundary_states - edge cases and type stability" begin
    st = bs_j0_state()
    @testset "empty vector" begin
        out = boundary_states(typeof(st)[], BS_CIRCLE)
        @test out isa Vector{BIMEigenstate{Float64,Float64}}
        @test isempty(out)
    end

    @testset "return type is inferred" begin
        @test @inferred(boundary_states([st], BS_CIRCLE)) isa Vector{BIMEigenstate{Float64,Float64}}
    end

    @testset "boundary_function(vector, billiard) agrees with the per-state objects" begin
        st1, _ = bs_rect_mode(1, 1)
        bfs = boundary_function([st1], BS_RECT)
        ref = boundary_function(st1, BS_RECT)
        @test bfs isa Vector{<:BoundaryFunction} && length(bfs) == 1
        @test bfs[1].u ≈ bs_expected_u(ref) rtol = 1e-12
        @test bfs[1].k === ref.k
    end
end

# compute_boundary_functions helpers (no eigen solves, synthetic matrices and hand-built states).
#  * _level_clusters: clusters of sorted-neighbour gaps <= tol, indices into ks,
#  * _pts_groups: levels grouped by identity of the stored boundary points,
#  * _complex_state: conversion of a BIMEigenstate to Complex{T} storage,
#  * _adjoint_inverse_iteration, _left_residuals, _left_null_vectors on A = U diag(s) V' with known left null vectors (the last columns of U).

@testset "_level_clusters - clusters of sorted neighbours" begin
    @test _level_clusters([1.0, 3.0, 1 + 1e-7, 2.0, 1 + 3e-7], 1e-6) == [[1, 3, 5], [4], [2]]
    @test _level_clusters([1.0, 2.0, 3.0], 1e-6) == [[1], [2], [3]]
    @test _level_clusters([1.0, 1.0, 1.0], 0.0) == [[1, 2, 3]]                  # gap == tol joins the cluster
    @test _level_clusters(Float64[], 1e-6) == Vector{Vector{Int}}()
    @test _level_clusters([5.0], 1e-6) == [[1]]
    @test _level_clusters([1.0, 1.0 + 3e-6, 1.0 + 6e-6], 4e-6) == [[1, 2, 3]]   # chaining through neighbours, not distance to the first member
    @test _level_clusters([1.0, 1.0 + 3e-6, 1.0 + 9e-6], 4e-6) == [[1, 2], [3]]
end

@testset "_pts_groups - grouping by boundary points identity" begin
    @test _pts_groups(nothing, 4) == [[1, 2, 3, 4]]
    pts1 = bs_grid(BS_RECT, 3.0)
    pts2 = bs_grid(BS_RECT, 3.0)                                                 # equal content, different object
    mk(pts) = BIMEigenstate{Float64,Float64}(3.0, 0.0, nothing, 0.0, pts, nothing, nothing)
    states = [mk(pts1), mk(pts2), mk(pts1)]
    @test _pts_groups(states, 3) == [[1, 3], [2]]
    @test _pts_groups([mk(pts1), mk(pts1)], 2) == [[1, 2]]
end

@testset "_complex_state - conversion to Complex storage" begin
    pts = bs_grid(BS_RECT, 3.0)
    n = length(pts.xy)
    u = collect(range(0.5, 1.5; length=n))
    vec = collect(range(-1.0, 1.0; length=n))
    st = BIMEigenstate{Float64,Float64}(3.2, 1e-9, vec, 1e-8, pts, u, 0.7)
    cs = _complex_state(st)
    @test cs isa BIMEigenstate{ComplexF64,Float64}
    @test cs.u == complex.(u) && cs.vec == complex.(vec)
    @test cs.pts === pts
    @test (cs.k, cs.k_im, cs.ten, cs.bnd_norm) == (3.2, 1e-9, 1e-8, 0.7)
    bare = _complex_state(BIMEigenstate{Float64,Float64}(3.2, 0.0, nothing, 0.0, pts, nothing, nothing))
    @test bare isa BIMEigenstate{ComplexF64,Float64}
    @test bare.vec === nothing && bare.u === nothing && bare.bnd_norm === nothing
    already = BIMEigenstate{ComplexF64,Float64}(3.2, 0.0, nothing, 0.0, pts, complex.(u), 0.7)
    @test _complex_state(already).u == already.u
    @test @inferred(_complex_state(st)) isa BIMEigenstate{ComplexF64,Float64}
end

# A = U diag(s) V' (complex), with the last m singular values `tiny`; returns A and the unitary U.
function bs_synthetic_matrix(n, m; tiny=fill(1e-12, m), seed=1)
    rng = MersenneTwister(seed)
    U = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
    V = Matrix(qr(randn(rng, ComplexF64, n, n)).Q)
    s = vcat(collect(range(1.0, 2.0; length=n - m)), tiny)
    return U * Diagonal(s) * V', U
end

@testset "_adjoint_inverse_iteration - left null vectors of a synthetic matrix" begin
    n = 8
    @testset "one left null vector" begin
        A, U = bs_synthetic_matrix(n, 1)
        Y = _adjoint_inverse_iteration(A, 1, MersenneTwister(3))
        @test size(Y) == (n, 1)
        @test norm(Y' * Y - I(1)) < 1e-12
        @test abs(dot(U[:, end], Y[:, 1])) > 1 - 1e-10                           # equal up to a global phase
        @test only(_left_residuals(A, Y)) < 1e-10
    end
    @testset "degenerate pair spans the left null space" begin
        A, U = bs_synthetic_matrix(n, 2; seed=2)
        Y = _adjoint_inverse_iteration(A, 2, MersenneTwister(4))
        @test norm(Y' * Y - I(2)) < 1e-12
        N = U[:, end-1:end]
        @test norm(Y - N * (N' * Y)) < 1e-9                                       # columns inside span of the two null vectors
        @test all(<(1e-9), _left_residuals(A, Y))
    end
    @testset "singular factorization returns nothing" begin
        A = ComplexF64[2 0 0; 0 3 0; 0 0 0]
        @test _adjoint_inverse_iteration(A, 1, MersenneTwister(0)) === nothing
    end
    @testset "deterministic for a fixed rng" begin
        A, _ = bs_synthetic_matrix(n, 1)
        @test _adjoint_inverse_iteration(A, 1, MersenneTwister(9)) == _adjoint_inverse_iteration(A, 1, MersenneTwister(9))
    end
end

@testset "_left_residuals - norm of A' y" begin
    A = ComplexF64[1 2; 3 4]
    Y = ComplexF64[1 0; 0 1]
    @test _left_residuals(A, Y) ≈ [norm(A'[:, 1]), norm(A'[:, 2])]
    @test _left_residuals(A, ComplexF64[1; 0;;]) ≈ [norm(A'[:, 1])]
end

@testset "_left_null_vectors - inverse iteration and singular-vector fallback" begin
    n = 8
    A, U = bs_synthetic_matrix(n, 1)
    Y = _left_null_vectors(A, 1, 5.0, 1e-6, MersenneTwister(5))
    @test abs(dot(U[:, end], Y[:, 1])) > 1 - 1e-10
    @testset "fallback: residual above res_tol (svdsolve left vector)" begin
        Yf = _left_null_vectors(A, 1, 5.0, 1e-20, MersenneTwister(5))
        @test size(Yf) == (n, 1)
        @test abs(dot(U[:, end], Yf[:, 1])) > 1 - 1e-8
    end
    @testset "fallback: singular LU" begin
        As = ComplexF64[2 0 0 0; 0 3 0 0; 0 0 4 0; 0 0 0 0]
        Ys = _left_null_vectors(As, 1, 5.0, 1e-6, MersenneTwister(5))
        @test abs(Ys[4, 1]) > 1 - 1e-10
    end
    @testset "pair with resolved tiny singular values, fallback keeps the orthonormal span" begin
        # svdsolve starts from one vector: pairs closer than about 1e-12 are not resolved (see report), so the gap here is 1e-9.
        A2, U2 = bs_synthetic_matrix(n, 2; tiny=[1e-12, 1e-9], seed=2)
        Y2 = _left_null_vectors(A2, 2, 5.0, 1e-20, MersenneTwister(5))
        N = U2[:, end-1:end]
        @test size(Y2) == (n, 2)
        @test norm(Y2 - N * (N' * Y2)) < 1e-7
        @test norm(Y2' * Y2 - I(2)) < 1e-8
    end
end

# _boundary_state / _boundary_states: u-state from a hand-built left vector y (no solve). Kernel: DLP on the unit circle without symmetry,
# so u_raw = y ./ ds. Expected: u = e^{i phi} y ./ ds / sqrt(bnd_norm) with phi making the largest entry real positive, bnd_norm the Rellich
# value of the unrotated y ./ ds (the Rellich integrand is phase invariant), and the Rellich norm of the stored u equal to 1.
bs_left_vector(n, seed) = randn(MersenneTwister(seed), ComplexF64, n)

@testset "_boundary_state - u-state from a hand-built left vector" begin
    kernel = DoubleLayerPotentialSolver(10.0, BS_CIRCLE)
    k = 3.0
    pts = evaluate_points(kernel, BS_CIRCLE, k)
    n = length(pts.xy)
    y = bs_left_vector(n, 1)
    u_raw = y ./ pts.ds
    st = _boundary_state(kernel, BS_CIRCLE, pts, k, 1e-12, y)

    @testset "type and fields" begin
        @test st isa BIMEigenstate{ComplexF64,Float64}
        @test st.vec === nothing
        @test st.u isa Vector{ComplexF64} && length(st.u) == n
        @test st.pts === pts
        @test st.k == k && st.k_im == 0.0 && st.ten == 1e-12
        @test st.bnd_norm isa Float64
    end

    @testset "largest entry real and positive" begin
        p = argmax(abs.(st.u))
        @test p == argmax(abs.(u_raw))
        @test imag(st.u[p]) == 0 || abs(imag(st.u[p])) <= 1e-14 * abs(st.u[p])
        @test real(st.u[p]) > 0
    end

    @testset "global phase and normalization" begin
        @test st.bnd_norm ≈ _rellich(pts, u_raw, k) rtol = 1e-12
        @test _rellich(pts, st.u, k) ≈ 1 rtol = 1e-12
        c = dot(u_raw, st.u) / dot(u_raw, u_raw)                         # st.u is a scalar multiple of u_raw ...
        @test st.u ≈ c .* u_raw rtol = 1e-12
        @test abs(c) ≈ 1 / sqrt(st.bnd_norm) rtol = 1e-12               # ... of modulus 1 / sqrt(bnd_norm)
    end

    @testset "global phase of y does not change the state" begin
        st_rot = _boundary_state(kernel, BS_CIRCLE, pts, k, 1e-12, cis(0.7) .* y)
        @test st_rot.u ≈ st.u rtol = 1e-12
        @test st_rot.bnd_norm ≈ st.bnd_norm rtol = 1e-12
    end

    @testset "complex k is split into k and k_im" begin
        sc = _boundary_state(kernel, BS_CIRCLE, pts, 3.0 + 0.02im, 1e-12, y)
        @test sc.k == 3.0 && sc.k_im == 0.02
        @test sc.u ≈ st.u rtol = 1e-12                                   # Rellich uses the real part of k
    end

    @testset "a view of a column is accepted" begin
        Y = hcat(y, bs_left_vector(n, 2))
        @test _boundary_state(kernel, BS_CIRCLE, pts, k, 1e-12, view(Y, :, 1)).u ≈ st.u rtol = 1e-12
    end

    @testset "type inferred" begin
        @test @inferred(_boundary_state(kernel, BS_CIRCLE, pts, k, 1e-12, y)) isa BIMEigenstate{ComplexF64,Float64}
    end
end

@testset "_boundary_states - one state per column of the left vectors" begin
    kernel = DoubleLayerPotentialSolver(10.0, BS_CIRCLE)
    pts = evaluate_points(kernel, BS_CIRCLE, 3.0)
    n = length(pts.xy)
    Y = hcat(bs_left_vector(n, 3), bs_left_vector(n, 4), bs_left_vector(n, 5))
    ks = [2.9, 3.0 + 0.01im, 3.1]
    ts = [1e-10, 2e-10, 3e-10]
    states = _boundary_states(kernel, BS_CIRCLE, ks, ts, Y, pts)
    @test states isa Vector{BIMEigenstate{ComplexF64,Float64}}
    @test length(states) == 3
    @test all(s -> s.pts === pts && s.vec === nothing, states)
    @test [s.k for s in states] == real.(ks) && [s.k_im for s in states] == imag.(ks) && [s.ten for s in states] == ts
    for j in 1:3
        ref = _boundary_state(kernel, BS_CIRCLE, pts, ks[j], ts[j], Y[:, j])
        @test states[j].u == ref.u && states[j].bnd_norm == ref.bnd_norm      # column j gives level j
        @test _rellich(pts, states[j].u, real(ks[j])) ≈ 1 rtol = 1e-12
    end
    @test !(states[1].u ≈ states[2].u)
    @test isempty(_boundary_states(kernel, BS_CIRCLE, Float64[], Float64[], zeros(ComplexF64, n, 0), pts))
end
