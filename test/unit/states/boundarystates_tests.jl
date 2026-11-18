# Unit tests for
#   src/states/boundarystates.jl   (boundary_states: basis eigenstates -> BIMEigenstate u-states)
# boundary_states turns coefficient states into boundary states that store the Rellich-normalized normal derivative
# u = d_n psi on the (unfolded) boundary and no layer density. States are grouped by (objectid(basis), k_basis); a group
# shares one boundary grid (from the largest k of the group), one gradient-matrix evaluation and one matrix product U = G X.
#
# No eigen solves. States are hand-built with prescribed coefficients. References are
#  * the per-state boundary_function(state, billiard; ppw).u, which uses the same grid when the group has one k (or one state),
#  * the analytic gradient of rectangle plane-wave modes, u = grad(psi) . n at the returned points,
#  * the Rellich identity rellich(unfolded) = (number of copies) * bnd_norm,
#  * a test-local stub basis whose gradient matrices contain a NaN, for the regularisation.
# Helper definitions use the prefix `bs_` / `BS_`.

using StaticArrays
using LinearAlgebra: dot
using QuantumBilliards: boundary_states, _rellich, get_boundary_curves_with_ignored, FourierNodes, boundary_coords,
                        _debroglie_grid_size, gradient_matrices

const BS_J01 = 2.404825557695773
const BS_J11 = 3.8317059702075125
const BS_A, BS_B = 1.0, 0.6
const BS_RECT = RectangleBilliard(BS_A, BS_B)
const BS_CIRCLE = CircleBilliard(1.0)

bs_j0_state(basis=FourierBessel(1; sym_x=1, sym_y=1); c=1.0) = BasisEigenstate(BS_J01, [c], 1e-12, basis)

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
        @test out[1].u ≈ ref.u rtol = 1e-12
        @test out[1].bnd_norm ≈ ref.norm rtol = 1e-12
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
        @test out.u ≈ ref.u rtol = 1e-12
        @test length(out.u) == 513
    end

    @testset "ppw is forwarded and multithreaded does not change the values" begin
        st = bs_j0_state()
        coarse = boundary_states([st], BS_CIRCLE)[1]
        fine = boundary_states([st], BS_CIRCLE; ppw=3000.0)[1]
        @test length(fine.u) > length(coarse.u)
        @test fine.u ≈ boundary_function(st, BS_CIRCLE; ppw=3000.0).u rtol = 1e-12
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
            # u = grad(psi) . n at the unfolded points with the unfolded normals
            @test out.u ≈ [dot(grad(p[1], p[2]), nrm) for (p, nrm) in zip(pts.xy, pts.normal)] atol = 1e-10
            # the Rellich integrand is reflection invariant: unfolded value is 4 copies of the stored fundamental one
            @test _rellich(pts, out.u, st.k) ≈ 4 * out.bnd_norm rtol = 1e-9
            @test out.u ≈ boundary_function(st, BS_RECT).u rtol = 1e-12
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
            @test out[i].u ≈ boundary_function(s, BS_CIRCLE).u rtol = 1e-10
        end
        @test out[3].u ≈ 2 .* out[1].u rtol = 1e-12                 # linearity in the coefficients
        @test out[3].bnd_norm ≈ 4 * out[1].bnd_norm rtol = 1e-12    # Rellich norm is quadratic
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
        # u does not depend on the wavenumber label, only on the grid: the single high-k state gives the same grid
        lo_hi = BasisEigenstate(10.0, kb, vec1, 1e-12, basis)
        @test res[1].u ≈ boundary_function(lo_hi, BS_CIRCLE).u rtol = 1e-10
        @test res[2].u ≈ boundary_function(hi, BS_CIRCLE).u rtol = 1e-10
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
    @test out.u[BS_NAN_ROW] ≈ (ref[BS_NAN_ROW-1] + ref[BS_NAN_ROW+1]) / 2 atol = 1e-12   # mean of the neighbours
    mask = trues(length(ref)); mask[BS_NAN_ROW] = false
    @test out.u[mask] ≈ ref[mask] atol = 1e-12
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
        @test bfs[1].u ≈ ref.u rtol = 1e-12
        @test bfs[1].k === ref.k
    end
end
