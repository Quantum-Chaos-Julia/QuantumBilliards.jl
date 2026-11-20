# Unit tests for
#   src/states/husimifunctions.jl   (antisym_vec, _normalize_husimi!, _is_global_phase_real, the phase-factored explicit-grid
#                                    husimi_function(k, s, ds, u, L, qs, ps), the automatic-grid form
#                                    husimi_function(k, s, ds, u, L), the BoundaryPoints form, the eigenstate forms
#                                    for basis states and BIM states, the batched forms, the SpectralData wrappers, the
#                                    outer-loop selection of husimi_function(boundary_function(...)) through bf.outer /
#                                    _outer_range, and _collect_functions)
# The state forms are husimi_function(boundary_function(state, ...); c, w, full_p). _foreign_kwargs and _apply_state (also
# defined in this source file) are tested in wavefunctions_tests.jl next to _drop_kwargs.
# The Poincare-Husimi amplitude is h(q, p) = sum_j u_j exp(-k (s_j - q)^2 / 2) exp(-i k p (s_j - q)) ds_j and the returned
# density is H = |h|^2 / sum |h|^2. This file has no `fundamental_only` or `inside_only` keyword to test: the
# state-based forms always use the boundary returned by boundary_function (unfolded when the basis has reflection symmetries).
#
# All references are analytic Gaussian integrals (sigma = 1/sqrt(k), truncation w = 7 gives a relative error of e^{-24}):
#  * u = const:                  h = sqrt(2 pi / k) exp(-k p^2 / 2)           H(q, p) ~ exp(-k p^2), independent of q,
#                                                                              <p^2> = 1 / (2 k)
#  * u = cos(k p0 s):            H(q, p) ~ g(p - p0) + g(p + p0) + 2 exp(-k (p^2 + p0^2)) cos(2 k p0 q), g(x) = exp(-k x^2)
#  * u = exp(i k p0 s):          H(q, p) ~ exp(-k (p - p0)^2)   (momentum p0 only; needs full_p = true, which is selected
#                                                                 automatically for a genuinely complex u)
# on the unit circle (L = 2 pi) where cos(m s) is periodic for integer m = k p0 (k = 20, p0 = 1/2, m = 10). The marginal
# over q of the density of the state J1(j11 r) cos(theta) is the sum of two Gaussians at p = +-1/k, because the
# interference term averages out over q.
# Sampling grids are hand-built (equispaced, or the smooth periodic grading s(t) = L (t - sin(2 pi t)/(2 pi))), basis
# states are the hand-built circle modes (no eigen solves), BIM states are partial states with u set through the
# type-parameter constructor, and the recomputation path uses a test-local solver stub.
# The normal scaling problem of boundary_function(::BasisEigenstate) (see boundaryfunctions_tests.jl, finding 1) does
# not affect the Husimi density of a single-curve boundary, since H is normalised to sum 1.
# Helper definitions use the prefix `hu_` / `HU_`.
#
# FINDINGS (no source change): the batched forms drop states that throw, silently (a @debug message only). This is current
# documented/observed
# behaviour and is pinned below. Single-state forms return a HusimiFunction (iterating as (H, qs, ps)), batched and
# SpectralData forms return a Vector{HusimiFunction}; every object carries the k of its state (or the numeric k argument).
# The BoundaryFunction forms husimi_function(bf[, qs, ps]) are checked against the numeric forms on the stored arrays and
# against the state forms (husimi_function(boundary_function(state, billiard)).H == husimi_function(state, billiard).H).
# The single-axis reflection path was checked through boundary_function in boundaryfunctions_tests.jl (Limacon) and is
# consistent; the single-axis bug of apply_symmetries_to_wavefunction does not affect the boundary-function path.

using StaticArrays
using LinearAlgebra: norm, dot
using SpecialFunctions: besselj
using QuantumBilliards: antisym_vec, _normalize_husimi!, _is_global_phase_real, _outer_range, _collect_functions

const HU_J01 = 2.404825557695773       # first zero of J0
const HU_J02 = 5.520078110286311       # second zero of J0
const HU_J11 = 3.8317059702075125      # first zero of J1
const HU_K = 20.0                      # sigma = 1/sqrt(k) = 0.22, window w sigma = 1.56 < L/2: one periodic image suffices
const HU_P0 = 0.5                      # k * p0 = 10, an integer number of periods on the unit circle
const HU_CIRCLE = CircleBilliard(1.0)

struct hu_SweepStub <: QuantumBilliards.SweepBIMSolver end
const HU_STUB_U = -3.0
QuantumBilliards.solve_state(::hu_SweepStub, pts::BoundaryPoints, k, billiard; multithreaded::Bool=true) =
    (0.0, zeros(length(pts.xy)), fill(HU_STUB_U, length(pts.xy)), 0.5)

# Equispaced periodic grid on the unit circle: s_j = (j-1) L / N, ds = L / N.
function hu_uniform_grid(::Type{T}, N::Int) where {T<:Real}
    L = T(2pi)
    return collect(T, 0:N-1) .* (L / N), fill(L / N, N), L
end

# Smooth periodic grading: s(t) = L (t - sin(2 pi t) / (2 pi)), ds = s'(t) / N, t_j = (j-1)/N; non-uniform in s and ds.
function hu_graded_grid(N::Int)
    L = 2pi
    t = collect(0:N-1) ./ N
    return L .* (t .- sin.(2pi .* t) ./ (2pi)), L .* (1 .- cos.(2pi .* t)) ./ N, L
end

# BoundaryPoints of the unit circle on a given s grid (angle = s).
hu_pts(s, ds) = BoundaryPoints([SVector(cos(a), sin(a)) for a in s]; normal=[SVector(cos(a), sin(a)) for a in s], s=collect(s), ds=collect(ds))

# Unnormalised analytic densities (the common factor 2 pi / k and the 1/4 are dropped, all densities are normalised).
hu_const_density(k, p) = exp(-k * p^2)
hu_cos_density(k, p0, q, p) = exp(-k * (p - p0)^2) + exp(-k * (p + p0)^2) + 2exp(-k * (p^2 + p0^2)) * cos(2k * p0 * q)
hu_plane_density(k, p0, p) = exp(-k * (p - p0)^2)

# Analytic density on the returned grids, normalised to sum 1.
hu_expected(f, qs, ps) = (E = [f(q, p) for q in qs, p in ps]; E ./ sum(E))

# Largest deviation relative to the largest entry of the expected density.
hu_err(H, E) = maximum(abs, H .- E) / maximum(E)

# Common properties of a Poincare-Husimi density.
function hu_check_density(H, qs, ps)
    @test size(H) == (length(qs), length(ps))
    @test all(isfinite, H) && all(>=(0), H)
    @test sum(H) ≈ 1 rtol = 1e-10
end

# Checks for a constant boundary function: H(q, p) / H(q, 0) = exp(-k p^2) for every q, and <p^2> = 1/(2k).
function hu_check_constant(H, qs, ps, k; var=true)
    mid = argmin(abs.(ps))
    @test abs(ps[mid]) < 1e-12
    ratio = H ./ H[:, mid:mid]
    @test maximum(abs, ratio .- reshape(hu_const_density.(k, ps), 1, :)) < 1e-6
    @test maximum(abs, H .- H[1:1, :]) <= 1e-8 * maximum(H)                  # independent of q
    var && @test sum(H .* reshape(ps .^ 2, 1, :)) ≈ 1 / (2k) rtol = 1e-5
end

hu_bim_state(k, u, pts) = BIMEigenstate{Float64,Float64}(k, 0.0, nothing, 1e-9, pts, u, nothing)

# Two HusimiFunction objects with equal type, density, grids and wavenumber (the objects define no ==).
hu_same(a::HusimiFunction, b::HusimiFunction) = typeof(a) == typeof(b) && a.H == b.H && a.qs == b.qs && a.ps == b.ps && a.k == b.k
hu_j0_state() = BasisEigenstate(HU_J01, [1.0], 1e-12, FourierBessel(1; sym_x=1, sym_y=1))
hu_j1_state() = BasisEigenstate(HU_J11, [1.0], 1e-12, FourierBessel(1; sym_x=-1, sym_y=1))      # J1(k r) cos(theta)

@testset "antisym_vec" begin
    @test antisym_vec([0.0, 1.0, 2.0, 3.0]) == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    @test antisym_vec([0, 1, 2]) == [-2, -1, 0, 1, 2]
    @test antisym_vec([5.0]) == [5.0]
    @test antisym_vec([1.0, 2.0]) == [-2.0, 1.0, 2.0]                 # the first entry is not duplicated, nor negated
    v = collect(0.0:0.25:1.0)
    w = antisym_vec(v)
    @test length(w) == 2 * length(v) - 1
    @test w == -reverse(w)                                            # odd about the centre when v[1] = 0
    @test w[length(v):end] == v
    @test eltype(antisym_vec(Float32[0, 1, 2])) == Float32
end

@testset "_normalize_husimi!" begin
    H = [1.0 2.0; 3.0 4.0]
    @test _normalize_husimi!(H) === H
    @test sum(H) ≈ 1
    @test H ≈ [1.0 2.0; 3.0 4.0] ./ 10
end

@testset "_is_global_phase_real" begin
    @test _is_global_phase_real([1.0, -2.0, 0.5])
    @test _is_global_phase_real(exp(0.7im) .* [1.0, -2.0, 0.5])
    @test _is_global_phase_real(ComplexF64[1, 2, 3])
    @test !_is_global_phase_real(ComplexF64[1, 1im, 2])
    @test !_is_global_phase_real(exp.(1im .* (0:7)))
    @test _is_global_phase_real(zeros(ComplexF64, 4))
    @test _is_global_phase_real([1.0, 2.0, 3.0] .+ 1e-9im)            # within the default rtol = 1e-6
    @test !_is_global_phase_real([1.0, 2.0, 3.0] .+ 1e-3im)
    @test _is_global_phase_real([1.0, 2.0, 3.0] .+ 1e-3im; rtol=1e-2)
end

@testset "husimi_function(k, s, ds, u, L) - constant boundary function, uniform nodes" begin
    k = HU_K
    s, ds, L = hu_uniform_grid(Float64, 256)
    u = ones(256)

    @testset "analytic Gaussian in p, independent of q" begin
        H, qs, ps = husimi_function(k, s, ds, u, L)
        hu_check_density(H, qs, ps)
        hu_check_constant(H, qs, ps, k)
        @test hu_err(H, hu_expected((q, p) -> hu_const_density(k, p), qs, ps)) < 1e-8
    end

    @testset "grids: signed momentum grid in [-1, 1] with p = 0 at the centre, evenly spaced boundary positions" begin
        H, qs, ps = husimi_function(k, s, ds, u, L)
        np = ceil(Int, 10 * sqrt(k))                                  # c = 10 samples per sigma in p
        @test length(ps) == 2np + 1 && ps[1] ≈ -1 && ps[end] ≈ 1
        @test ps ≈ -reverse(ps) atol = 1e-14
        @test issorted(qs) && qs[1] == 0 && qs[end] < L
        σ = 1 / sqrt(k)
        @test all(diff(qs) .≈ qs[2] - qs[1]) && qs[2] - qs[1] <= σ / 10
        @test eltype(H) == Float64 && eltype(qs) == Float64 && eltype(ps) == Float64
    end

    @testset "c sets the phase-space sampling density" begin
        _, qs1, ps1 = husimi_function(k, s, ds, u, L; c=5.0)
        _, qs2, ps2 = husimi_function(k, s, ds, u, L; c=10.0)
        @test length(ps1) == 2 * ceil(Int, 5 * sqrt(k)) + 1 < length(ps2)
        @test length(qs1) <= length(qs2)
    end

    @testset "w sets the Gaussian truncation: a narrow window changes the density" begin
        H, qs, ps = husimi_function(k, s, ds, u, L; w=0.5)
        E = hu_expected((q, p) -> hu_const_density(k, p), qs, ps)
        @test hu_err(H, E) > 1e-3
        @test sum(H) ≈ 1
    end

    @testset "full_p = true gives the same density for a real u" begin
        Hf, qf, pf = husimi_function(k, s, ds, u, L; full_p=true)
        Hh, qh, ph = husimi_function(k, s, ds, u, L; full_p=false)
        @test pf ≈ ph && qf == qh
        @test Hf ≈ Hh atol = 1e-12 * maximum(Hf)
    end
end

@testset "husimi_function - real plane wave cos(k p0 s) with interference term" begin
    k = HU_K; p0 = HU_P0
    s, ds, L = hu_uniform_grid(Float64, 256)
    u = cos.(k * p0 .* s)
    f = (q, p) -> hu_cos_density(k, p0, q, p)

    @testset "analytic density, default full_p = false (reflection p -> -p)" begin
        H, qs, ps = husimi_function(k, s, ds, u, L)
        hu_check_density(H, qs, ps)
        @test hu_err(H, hu_expected(f, qs, ps)) < 1e-8
        @test H ≈ reverse(H; dims=2) atol = 1e-12 * maximum(H)      # H(q, -p) = H(q, p) for a real u
    end

    @testset "full_p = true is evaluated explicitly and agrees" begin
        H, qs, ps = husimi_function(k, s, ds, u, L; full_p=true)
        @test hu_err(H, hu_expected(f, qs, ps)) < 1e-8
        Hr, _, _ = husimi_function(k, s, ds, u, L; full_p=false)
        @test H ≈ Hr atol = 1e-12 * maximum(H)
    end

    @testset "marginal over q: two Gaussians at p = +-p0, interference averages out" begin
        H, qs, ps = husimi_function(k, s, ds, u, L)
        M = vec(sum(H; dims=1))
        E = hu_const_density.(k, ps .- p0) .+ hu_const_density.(k, ps .+ p0)
        @test maximum(abs, M .- E ./ sum(E)) / maximum(M) < 1e-8
        @test abs(ps[argmax(M)]) ≈ p0 atol = 1 / 45
        i0 = argmin(abs.(ps .- p0))
        @test M[i0] > 10 * M[argmin(abs.(ps))]                        # concentrated at |p| = p0, not at p = 0
    end
end

@testset "husimi_function - complex plane wave exp(i k p0 s): momentum p0 only, full_p selected automatically" begin
    k = HU_K; p0 = HU_P0
    s, ds, L = hu_uniform_grid(Float64, 256)
    u = exp.(1im * k * p0 .* s)

    @testset "automatic escalation to full_p, peak at p = +p0" begin
        H, qs, ps = husimi_function(k, s, ds, u, L)                   # full_p = false requested by default
        @test length(ps) == 2 * ceil(Int, 10 * sqrt(k)) + 1 && ps[1] ≈ -1
        hu_check_density(H, qs, ps)
        @test hu_err(H, hu_expected((q, p) -> hu_plane_density(k, p0, p), qs, ps)) < 1e-8
        @test maximum(abs, H .- H[1:1, :]) <= 1e-8 * maximum(H)      # independent of q (a pure phase)
        @test ps[argmax(H[1, :])] ≈ p0 atol = 1 / 45
    end

    @testset "complex conjugation reverses the momentum: H[conj(u)](q, p) = H[u](q, -p)" begin
        H, _, _ = husimi_function(k, s, ds, u, L)
        Hc, _, _ = husimi_function(k, s, ds, conj.(u), L)
        @test Hc ≈ reverse(H; dims=2) atol = 1e-12 * maximum(H)
    end

    @testset "explicit full_p = true gives the same density" begin
        H, _, _ = husimi_function(k, s, ds, u, L)
        Hf, _, _ = husimi_function(k, s, ds, u, L; full_p=true)
        @test Hf ≈ H atol = 1e-12 * maximum(H)
    end
end

@testset "husimi_function - invariances of the normalised density" begin
    k = HU_K; p0 = HU_P0
    s, ds, L = hu_uniform_grid(Float64, 256)
    u = cos.(k * p0 .* s) .+ 0.4 .* cos.(2k * p0 .* s .+ 0.3)
    H, qs, _ = husimi_function(k, s, ds, u, L)
    @test length(qs) == ceil(Int, L * 10 * sqrt(k))                   # evenly spaced grid, independent of the node positions

    @testset "real scale factors and a global complex phase do not change the density" begin
        @test husimi_function(k, s, ds, -3.7 .* u, L).H ≈ H atol = 1e-12 * maximum(H)
        Hp, _, ps = husimi_function(k, s, ds, exp(0.8im) .* u, L)        # real up to a global phase: full_p = false is accepted
        @test Hp ≈ H atol = 1e-12 * maximum(H)
    end

    @testset "a cyclic shift of u by n nodes shifts the density by n node spacings in q" begin
        # u'_j = u_{j-n}  =>  h'(q) = h(q - n ds)
        n = 17
        ps = collect(range(0, 1; length=21))
        Hs, _, _ = husimi_function(k, s, ds, circshift(u, n), L, qs, ps)
        Hr, _, _ = husimi_function(k, s, ds, u, L, qs .- n * ds[1], ps)
        @test hu_err(Hs, Hr) < 1e-10
    end
end

@testset "husimi_function(k, s, ds, u, L, qs, ps) - prescribed grids (analytic densities)" begin
    k = HU_K; p0 = HU_P0
    s, ds, L = hu_uniform_grid(Float64, 256)
    u = cos.(k * p0 .* s)
    f = (q, p) -> hu_cos_density(k, p0, q, p)
    qs = [0.3, 1.1, 2.9, 4.0, 4.6, 5.7]                              # includes q within w / sqrt(k) of L
    ps = collect(range(-1, 1; length=41))

    @testset "analytic density on arbitrary q and p, grids returned as given" begin
        H, qo, po = husimi_function(k, s, ds, u, L, qs, ps; full_p=true)
        @test qo == qs && po == ps
        hu_check_density(H, qo, po)
        @test hu_err(H, hu_expected(f, qs, ps)) < 1e-8
    end

    @testset "full_p = false evaluates p >= 0 and reflects" begin
        psh = collect(range(0, 1; length=21))
        H, qo, po = husimi_function(k, s, ds, u, L, qs, psh)
        @test po ≈ antisym_vec(psh) && length(po) == 41
        @test H ≈ reverse(H; dims=2) atol = 1e-12 * maximum(H)
        @test hu_err(H, hu_expected(f, qo, po)) < 1e-8
    end

    @testset "q is periodic with period L" begin
        H, _, _ = husimi_function(k, s, ds, u, L, [0.5, 0.5 + L, 0.5 - L], ps; full_p=true)
        @test H[2, :] ≈ H[1, :] atol = 1e-12 * maximum(H)
        @test H[3, :] ≈ H[1, :] atol = 1e-12 * maximum(H)
    end

    @testset "the Gaussian window is not truncated near q = L" begin
        H, qo, po = husimi_function(k, s, ds, u, L, [5.7], ps; full_p=true)
        @test hu_err(H, hu_expected(f, qo, po)) < 1e-8
    end

    @testset "full_p = false argument checks" begin
        @test_throws ArgumentError husimi_function(k, s, ds, u, L, qs, ps)                       # negative momenta
        @test_throws ArgumentError husimi_function(k, s, ds, u, L, qs, collect(range(0.1, 1; length=5)))   # not starting at 0
        @test_throws ArgumentError husimi_function(k, s, ds, exp.(1im .* k .* p0 .* s), L, qs, collect(range(0, 1; length=5)))   # genuinely complex u
        H, _, _ = husimi_function(k, s, ds, exp.(1im .* k .* p0 .* s), L, qs, ps; full_p=true)   # allowed with full_p = true
        @test sum(H) ≈ 1
        @test hu_err(H, hu_expected((q, p) -> hu_plane_density(k, p0, p), qs, ps)) < 1e-8
    end
end

# Naive reference of the explicit-grid evaluator: the direct window sum
#   h(q, p) = sum_j u_j ds_j exp(-k d_j^2 / 2) exp(-i k p d_j),  d_j = s_j + m L - q,  q = mod(q, L),
# over all periodic images m and all nodes with |d_j| <= w / sqrt(k), H = |h|^2 normalised to sum 1. Evaluated in Float64.
# The momentum grid is used as given (for full_p = false the caller passes the reflected grid antisym_vec(ps)).
function hu_naive(k, s, ds, u, L, qs, ps; w=7.0)
    k, L = Float64(k), Float64(L)
    s, ds, qs, ps = Float64.(s), Float64.(ds), Float64.(qs), Float64.(ps)
    u = Complex{Float64}.(u)
    width = w / sqrt(k)
    nc = ceil(Int, width / L) + 2                                     # more images than needed, the window cut decides
    H = zeros(length(qs), length(ps))
    for (iq, q0) in enumerate(qs), (ip, p) in enumerate(ps)
        q = mod(q0, L)
        h = 0.0im
        for m in -nc:nc, j in eachindex(s)
            d = s[j] + m * L - q
            abs(d) <= width || continue
            h += u[j] * ds[j] * exp(-k * d^2 / 2) * cis(-k * p * d)
        end
        H[iq, ip] = abs2(h)
    end
    return H ./ sum(H)
end

# Graded periodic nodes on [0, L): s(t) = L (t - sin(2 pi t) / (2 pi)), ds = s'(t) / N.
function hu_graded_nodes(L, N)
    t = collect(0:N-1) ./ N
    return L .* (t .- sin.(2pi .* t) ./ (2pi)), L .* (1 .- cos.(2pi .* t)) ./ N
end

@testset "husimi_function(k, s, ds, u, L, qs, ps) - against the direct window sum" begin
    k = 20.0

    @testset "uniform nodes, real u, full_p = false" begin
        s, ds, L = hu_uniform_grid(Float64, 64)
        u = cos.(10 .* s) .+ 0.3 .* sin.(3 .* s)
        qs = [0.0, 0.37, 1.9, 3.14, 5.0, 6.0]
        ps = collect(range(0, 1; length=9))
        H, qo, po = husimi_function(k, s, ds, u, L, qs, ps)
        @test qo == qs && po ≈ antisym_vec(ps)
        @test size(H) == (6, 17)
        @test sum(H) ≈ 1 rtol = 1e-10
        E = hu_naive(k, s, ds, u, L, qs, po)
        @test hu_err(H, E) < 1e-10
    end

    @testset "graded nodes, complex u, full_p = true" begin
        s, ds = hu_graded_nodes(2pi, 80)
        L = 2pi
        u = exp.(1im .* (7 .* s .+ 0.4 .* sin.(s))) .* (1 .+ 0.2 .* cos.(s))
        qs = [0.2, 1.0, 2.7, 4.4, 6.1]
        ps = collect(range(-1, 1; length=15))
        H, qo, po = husimi_function(k, s, ds, u, L, qs, ps; full_p=true)
        @test qo == qs && po == ps
        @test sum(H) ≈ 1 rtol = 1e-10
        @test hu_err(H, hu_naive(k, s, ds, u, L, qs, ps)) < 1e-10
        @test maximum(abs, H .- reverse(H; dims=2)) > 1e-3 * maximum(H)     # genuinely asymmetric in p: the test is not trivial
    end

    @testset "full_p = false and full_p = true agree for a real u" begin
        s, ds = hu_graded_nodes(2pi, 80)
        L = 2pi
        u = cos.(5 .* s) .+ 0.5 .* cos.(2 .* s .+ 0.3)
        qs = [0.1, 1.3, 2.2, 5.9]
        ps_half = collect(range(0, 1; length=11))
        Hh, _, ph = husimi_function(k, s, ds, u, L, qs, ps_half; full_p=false)
        Hf, _, pf = husimi_function(k, s, ds, u, L, qs, antisym_vec(ps_half); full_p=true)
        @test ph ≈ pf
        @test hu_err(Hh, Hf) < 1e-10
        @test Hh ≈ reverse(Hh; dims=2) atol = 1e-12 * maximum(Hh)
        @test hu_err(Hh, hu_naive(k, s, ds, u, L, qs, ph)) < 1e-10
        @test sum(Hh) ≈ 1 rtol = 1e-10
        @test sum(Hf) ≈ 1 rtol = 1e-10
    end

    @testset "zero-weight nodes (seam-like) contribute nothing" begin
        s, ds, L = hu_uniform_grid(Float64, 64)
        ds[[1, 2, 32, 64]] .= 0.0
        u = cos.(10 .* s) .+ 0.2
        qs = [0.0, 0.2, 3.1, 6.2]
        ps = collect(range(0, 1; length=9))
        H, _, po = husimi_function(k, s, ds, u, L, qs, ps)
        @test sum(H) ≈ 1 rtol = 1e-10
        @test hu_err(H, hu_naive(k, s, ds, u, L, qs, po)) < 1e-10
        # a zero-weight node is the same as a removed one, whatever its u is
        keep = ds .> 0
        u2 = copy(u); u2[.!keep] .= 1e6
        H2, _, _ = husimi_function(k, s, ds, u2, L, qs, ps)
        @test H2 ≈ H atol = 1e-10 * maximum(H)
        H3, _, _ = husimi_function(k, s[keep], ds[keep], u[keep], L, qs, ps)
        @test H3 ≈ H atol = 1e-10 * maximum(H)
    end

    @testset "seam: q near 0 and near L, and q outside [0, L)" begin
        s, ds = hu_graded_nodes(2pi, 80)
        L = 2pi
        u = cos.(10 .* s) .+ 0.3 .* cos.(3 .* s)
        qs = [0.0, 0.05, L - 0.05, L + 0.1, -0.1, L, 2L + 0.3]            # window w / sqrt(k) = 1.56 crosses the seam
        ps = collect(range(-1, 1; length=11))
        H, _, _ = husimi_function(k, s, ds, u, L, qs, ps; full_p=true)
        @test sum(H) ≈ 1 rtol = 1e-10
        @test hu_err(H, hu_naive(k, s, ds, u, L, qs, ps)) < 1e-10
        # q is periodic: q + L, q - L and q give the same row up to the common normalisation
        Hp, _, _ = husimi_function(k, s, ds, u, L, [0.1, 0.1 + L, 0.1 - L], ps; full_p=true)
        @test Hp[2, :] ≈ Hp[1, :] atol = 1e-10 * maximum(Hp)
        @test Hp[3, :] ≈ Hp[1, :] atol = 1e-10 * maximum(Hp)
    end

    @testset "small k: several periodic copies (ncopies > 1)" begin
        k = 0.5; L = 4.0                                              # w / sqrt(k) = 9.9 > 2 L: ncopies = 3
        s, ds = hu_graded_nodes(L, 40)
        u = cos.(2pi / L .* 2 .* s) .+ 0.4
        qs = [0.0, 0.7, 2.0, 3.9, 4.3]
        ps = collect(range(0, 1; length=9))
        H, _, po = husimi_function(k, s, ds, u, L, qs, ps)
        @test sum(H) ≈ 1 rtol = 1e-10
        @test hu_err(H, hu_naive(k, s, ds, u, L, qs, po)) < 1e-10
        Hc, _, pc = husimi_function(k, s, ds, exp.(1im .* 1.3 .* s), L, qs, collect(range(-1, 1; length=9)); full_p=true)
        @test hu_err(Hc, hu_naive(k, s, ds, exp.(1im .* 1.3 .* s), L, qs, pc)) < 1e-10
        Hw, _, pw = husimi_function(k, s, ds, u, L, qs, ps; w=3.0)          # w / sqrt(k) = 4.2: ncopies = 2
        @test hu_err(Hw, hu_naive(k, s, ds, u, L, qs, pw; w=3.0)) < 1e-10
    end

    @testset "w selects the truncation window" begin
        s, ds = hu_graded_nodes(2pi, 80)
        L = 2pi
        u = cos.(10 .* s)
        qs = [0.5, 2.5]; ps = collect(range(0, 1; length=7))
        H, _, po = husimi_function(k, s, ds, u, L, qs, ps; w=2.0)
        @test hu_err(H, hu_naive(k, s, ds, u, L, qs, po; w=2.0)) < 1e-10
        Hd, _, _ = husimi_function(k, s, ds, u, L, qs, ps)
        @test hu_err(H, Hd) > 1e-6                                    # the window matters
    end

    @testset "type stability and Float32" begin
        s, ds, L = hu_uniform_grid(Float64, 64)
        u = cos.(10 .* s)
        qs = [0.3, 1.1, 5.9]; ps = collect(range(0, 1; length=5))
        @inferred husimi_function(k, s, ds, u, L, qs, ps)
        @inferred husimi_function(k, s, ds, ComplexF64.(u), L, qs, ps; full_p=true)
        hf = husimi_function(k, s, ds, u, L, qs, ps)
        @test hf isa HusimiFunction{Float64}

        s32, ds32, L32 = hu_uniform_grid(Float32, 64)
        u32 = cos.(10f0 .* s32)
        qs32 = Float32[0.3, 1.1, 5.9]; ps32 = collect(range(0f0, 1f0; length=5))
        k32 = 20.0f0
        @inferred husimi_function(k32, s32, ds32, u32, L32, qs32, ps32)
        h32 = husimi_function(k32, s32, ds32, u32, L32, qs32, ps32)
        @test h32 isa HusimiFunction{Float32}
        @test h32.H isa Matrix{Float32} && h32.qs isa Vector{Float32} && h32.ps isa Vector{Float32}
        @test sum(h32.H) ≈ 1 rtol = 1e-4
        @test hu_err(Float64.(h32.H), hu_naive(k32, s32, ds32, u32, L32, qs32, h32.ps)) < 1e-4
    end

    @testset "argument checks of full_p = false" begin
        s, ds, L = hu_uniform_grid(Float64, 64)
        u = cos.(10 .* s)
        qs = [0.3, 1.1]
        @test_throws ArgumentError husimi_function(k, s, ds, u, L, qs, collect(range(-1, 1; length=5)))          # negative momenta
        @test_throws ArgumentError husimi_function(k, s, ds, u, L, qs, collect(range(0.1, 1; length=5)))        # not starting at 0
        @test_throws ArgumentError husimi_function(k, s, ds, exp.(1im .* 10 .* s), L, qs, collect(range(0, 1; length=5)))   # complex u
        @test sum(husimi_function(k, s, ds, exp.(1im .* 10 .* s), L, qs, collect(range(-1, 1; length=5)); full_p=true).H) ≈ 1
    end
end

@testset "husimi_function(k, s, ds, u, L) - automatic evenly spaced grid" begin
    @testset "uniform nodes, real u: grid shapes, full_p = false" begin
        k = 20.0; c = 10.0
        s, ds, L = hu_uniform_grid(Float64, 64)                       # nodes are not the q grid
        u = cos.(10 .* s)
        H, qs, ps = husimi_function(k, s, ds, u, L)
        nq = max(2, ceil(Int, L * c * sqrt(k))); np = max(1, ceil(Int, c * sqrt(k)))
        @test qs ≈ collect(range(0, L; length=nq + 1))[1:end-1]
        @test length(qs) == nq && qs[1] == 0 && qs[end] < L
        @test ps ≈ antisym_vec(collect(range(0, 1; length=np + 1)))
        @test size(H) == (nq, 2np + 1)
        @test sum(H) ≈ 1 rtol = 1e-10
        @test length(qs) != length(s)
        @test hu_err(H, hu_naive(k, s, ds, u, L, qs, ps)) < 1e-10
    end

    @testset "graded nodes, complex u: full_p escalates, signed grid in [-1, 1]" begin
        k = 9.0; c = 4.0
        s, ds = hu_graded_nodes(2pi, 50)
        L = 2pi
        u = exp.(1im .* 3 .* s)
        H, qs, ps = husimi_function(k, s, ds, u, L; c=c)
        nq = ceil(Int, L * c * sqrt(k)); np = ceil(Int, c * sqrt(k))
        @test length(qs) == nq && all(diff(qs) .≈ L / nq)
        @test ps ≈ collect(range(-1, 1; length=2np + 1))
        @test size(H) == (nq, 2np + 1)
        @test hu_err(H, hu_naive(k, s, ds, u, L, qs, ps)) < 1e-10
        Hf, qf, pf = husimi_function(k, s, ds, real.(u), L; c=c, full_p=true)
        @test pf ≈ ps && qf == qs
        @test hu_err(Hf, hu_naive(k, s, ds, real.(u), L, qf, pf)) < 1e-10
    end

    @testset "minimum grid sizes: nq >= 2, np >= 1" begin
        s, ds, L = hu_uniform_grid(Float64, 16)
        L = 0.1; s = s ./ (2pi) .* L; ds = ds ./ (2pi) .* L
        H, qs, ps = husimi_function(1.0, s, ds, ones(16), L; c=0.5)       # ceil(0.1 * 0.5) = 1 -> nq = 2, np = ceil(0.5) = 1
        @test length(qs) == 2 && qs ≈ [0.0, 0.05]
        @test ps ≈ [-1.0, 0.0, 1.0] && size(H) == (2, 3)
        @test sum(H) ≈ 1 rtol = 1e-10
    end

    @testset "equals the explicit-grid method on the same grid" begin
        k = 20.0
        s, ds = hu_graded_nodes(2pi, 60)
        L = 2pi
        u = cos.(10 .* s)
        hf = husimi_function(k, s, ds, u, L; c=5.0, w=6.0)
        ref = husimi_function(k, s, ds, u, L, hf.qs, hf.ps[length(hf.ps)÷2+1:end]; w=6.0)
        @test hf.H == ref.H && hf.qs == ref.qs && hf.ps == ref.ps && hf.k === k
    end

    @testset "Float32 grids" begin
        s, ds, L = hu_uniform_grid(Float32, 64)
        hf = husimi_function(20.0f0, s, ds, cos.(10f0 .* s), L)
        @test hf isa HusimiFunction{Float32}
        @test hf.qs isa Vector{Float32} && hf.ps isa Vector{Float32}
        nq = ceil(Int, L * 10 * sqrt(20.0f0))
        @test length(hf.qs) == nq && hf.qs[1] == 0f0 && hf.qs[end] < L
        @test sum(hf.H) ≈ 1 rtol = 1e-4
    end
end

@testset "husimi_function - non-uniform arclength grid" begin
    k = HU_K; p0 = HU_P0
    s, ds, L = hu_graded_grid(512)

    @testset "constant u: analytic Gaussian; grids are equispaced in q with spacing <= sigma / c" begin
        H, qs, ps = husimi_function(k, s, ds, ones(512), L)
        hu_check_density(H, qs, ps)
        σ = 1 / sqrt(k)
        @test qs[1] == 0 && qs[end] < L
        @test all(diff(qs) .≈ qs[2] - qs[1]) && qs[2] - qs[1] <= σ / 10
        @test length(ps) == 2 * ceil(Int, 10 * sqrt(k)) + 1
        Hi, qi = H, qs
        hu_check_constant(Hi, qi, ps, k)
        @test maximum(abs, H .- H[1:1, :]) <= 1e-8 * maximum(H)
    end

    @testset "cos(k p0 s): analytic density with interference" begin
        H, qs, ps = husimi_function(k, s, ds, cos.(k * p0 .* s), L)
        @test H ≈ reverse(H; dims=2) atol = 1e-12 * maximum(H)
        @test hu_err(H, hu_expected((q, p) -> hu_cos_density(k, p0, q, p), qs, ps)) < 1e-6
    end
end

@testset "husimi_function - window wider than half the period" begin
    k = 4.0; p0 = HU_P0                                               # sigma = 0.5, w sigma = 3.5 > L / 2, k p0 = 2
    s, ds, L = hu_uniform_grid(Float64, 256)
    @test 7 / sqrt(k) >= L / 2
    H, qs, ps = husimi_function(k, s, ds, cos.(k * p0 .* s), L)
    @test qs[1] == 0 && all(diff(qs) .≈ qs[2] - qs[1])                # equispaced boundary positions
    hu_check_density(H, qs, ps)
    @test hu_err(H, hu_expected((q, p) -> hu_cos_density(k, p0, q, p), qs, ps)) < 1e-6
    Hc, qc, pc = husimi_function(k, s, ds, ones(256), L)
    hu_check_constant(Hc, qc, pc, k; var=false)
end

@testset "husimi_function - Float32" begin
    s, ds, L = hu_uniform_grid(Float32, 256)
    k = 20.0f0
    H, qs, ps = husimi_function(k, s, ds, ones(Float32, 256), L)
    @test H isa Matrix{Float32} && qs isa Vector{Float32} && ps isa Vector{Float32}
    @test sum(H) ≈ 1 rtol = 1e-4
    mid = argmin(abs.(ps))
    @test maximum(abs, (H ./ H[:, mid:mid])[1, :] .- hu_const_density.(k, ps)) < 1e-3
end

@testset "husimi_function(k, pts::BoundaryPoints, u)" begin
    k = HU_K
    s, ds, L = hu_uniform_grid(Float64, 256)
    pts = hu_pts(s, ds)
    u = cos.(k * HU_P0 .* s)
    @test husimi_function(k, pts, u) isa HusimiFunction{Float64}
    @test hu_same(husimi_function(k, pts, u), husimi_function(k, s, ds, u, sum(ds)))
    @test hu_same(husimi_function(k, pts, u; c=5.0, w=6.0), husimi_function(k, s, ds, u, sum(ds); c=5.0, w=6.0))
    qs = [0.2, 1.0, 3.0]; ps = collect(range(0, 1; length=11))
    @test hu_same(husimi_function(k, pts, u, qs, ps), husimi_function(k, s, ds, u, sum(ds), qs, ps))
    @test husimi_function(k, pts, exp.(1im .* k .* HU_P0 .* s), qs, collect(range(-1, 1; length=11)); full_p=true).H ≈
          husimi_function(k, s, ds, exp.(1im .* k .* HU_P0 .* s), sum(ds), qs, collect(range(-1, 1; length=11)); full_p=true).H
    @test husimi_function(k, pts, u).k === k && husimi_function(k, pts, u, qs, ps).k === k      # the numeric k is stored
end

@testset "husimi_function numeric forms return a HusimiFunction carrying the numeric k" begin
    k = HU_K
    s, ds, L = hu_uniform_grid(Float64, 256)
    u = cos.(k * HU_P0 .* s)
    qs = [0.3, 1.1, 2.9]; ps = collect(range(-1, 1; length=21))

    @testset "automatic grid, uniform nodes" begin
        hf = husimi_function(k, s, ds, u, L)
        @test hf isa HusimiFunction{Float64} && hf.k === k
        @test hf.H isa Matrix{Float64} && hf.qs isa Vector{Float64} && hf.ps isa Vector{Float64}
        @test size(hf.H) == (length(hf.qs), length(hf.ps))
        H, qs0, ps0 = hf
        @test H === hf.H && qs0 === hf.qs && ps0 === hf.ps
    end

    @testset "automatic grid, graded nodes" begin
        sg, dsg, Lg = hu_graded_grid(512)
        hf = husimi_function(k, sg, dsg, cos.(k * HU_P0 .* sg), Lg)
        @test hf isa HusimiFunction{Float64} && hf.k === k
        @test size(hf.H) == (length(hf.qs), length(hf.ps))
    end

    @testset "prescribed grids are stored as given" begin
        hf = husimi_function(k, s, ds, u, L, qs, ps; full_p=true)
        @test hf isa HusimiFunction{Float64} && hf.k === k
        @test hf.qs == qs && hf.ps == ps
        @test size(hf.H) == (3, 21)
    end

    @testset "Float32 data give a Float32 object" begin
        s32, ds32, L32 = hu_uniform_grid(Float32, 256)
        hf = husimi_function(20.0f0, s32, ds32, ones(Float32, 256), L32)
        @test hf isa HusimiFunction{Float32} && hf.k isa Float32 && hf.k === 20.0f0
        @test hf.H isa Matrix{Float32} && hf.qs isa Vector{Float32} && hf.ps isa Vector{Float32}
    end
end

@testset "husimi_function(::BoundaryFunction)" begin
    k = HU_K
    s, ds, L = hu_uniform_grid(Float64, 256)
    u = cos.(k * HU_P0 .* s)
    bf = BoundaryFunction(u, s, ds, k, 0.9, L, 1:256)

    @testset "equals the numeric form on the outer loop with k = bf.k and L = bf.L" begin
        hf = husimi_function(bf)
        @test hf isa HusimiFunction{Float64} && hf.k === bf.k
        @test hu_same(hf, husimi_function(k, s, ds, u, L))
        @test hu_same(husimi_function(bf; c=5.0, w=6.0), husimi_function(k, s, ds, u, L; c=5.0, w=6.0))
        @test hu_err(hf.H, hu_expected((q, p) -> hu_cos_density(k, HU_P0, q, p), hf.qs, hf.ps)) < 1e-8
    end

    @testset "prescribed grids" begin
        qs = [0.3, 1.1, 2.9, 4.0]
        psfull = collect(range(-1, 1; length=21)); pshalf = collect(range(0, 1; length=11))
        hf = husimi_function(bf, qs, psfull; full_p=true)
        @test hu_same(hf, husimi_function(k, s, ds, u, L, qs, psfull; full_p=true))
        @test hf.qs == qs && hf.ps == psfull && hf.k === k
        hh = husimi_function(bf, qs, pshalf)                               # full_p = false: p >= 0 evaluated and reflected
        @test hu_same(hh, husimi_function(k, s, ds, u, L, qs, pshalf))
        @test hh.ps ≈ antisym_vec(pshalf)
        @test hh.H ≈ reverse(hh.H; dims=2) atol = 1e-12 * maximum(hh.H)
        @test_throws ArgumentError husimi_function(bf, qs, psfull)         # negative momenta need full_p = true
    end

    @testset "complex u switches to the full momentum grid" begin
        bfc = BoundaryFunction(exp.(1im * k * HU_P0 .* s), s, ds, k, 0.9, L, 1:256)
        hf = husimi_function(bfc)
        @test hf.ps[1] ≈ -1
        @test hu_err(hf.H, hu_expected((q, p) -> hu_plane_density(k, HU_P0, p), hf.qs, hf.ps)) < 1e-8
    end

    @testset "only the nodes of the outer loop are used" begin
        # 256 outer nodes followed by 32 junk nodes of a hole loop: the Husimi function ignores them
        sj = [s; L .+ (1:32) .* 0.1]; dsj = [ds; fill(0.1, 32)]; uj = [u; fill(1e3, 32)]
        bfj = BoundaryFunction(uj, sj, dsj, k, 0.9, L, 1:256)
        @test hu_same(husimi_function(bfj), husimi_function(bf))
        @test hu_same(husimi_function(bfj, [0.3, 1.1], collect(range(-1, 1; length=11)); full_p=true),
                      husimi_function(bf, [0.3, 1.1], collect(range(-1, 1; length=11)); full_p=true))
    end
end

@testset "husimi_function(::BasisEigenstate) - circle J0 mode" begin
    st = hu_j0_state()
    k = st.k
    H, qs, ps = husimi_function(st, HU_CIRCLE)

    @testset "constant normal derivative gives the analytic Gaussian in p, independent of q" begin
        hu_check_density(H, qs, ps)
        hu_check_constant(H, qs, ps, k; var=false)
        @test ps[argmax(H[1, :])] ≈ 0 atol = 1e-12                    # concentrated at normal incidence p = 0
    end

    @testset "grids: c samples per sigma in q and p" begin
        σ = 1 / sqrt(k)
        @test qs[1] == 0 && qs[end] < 2pi && all(diff(qs) .≈ qs[2] - qs[1])
        @test σ / 20 < qs[2] - qs[1] <= σ / 10
        @test length(ps) == 2 * ceil(Int, 10 * sqrt(k)) + 1
        Hc, qc, pc = husimi_function(st, HU_CIRCLE; c=5.0)
        @test length(pc) == 2 * ceil(Int, 5 * sqrt(k)) + 1 && length(qc) < length(qs)
    end

    @testset "equals the Husimi function of its own boundary data" begin
        bf = boundary_function(st, HU_CIRCLE)
        H2, q2, p2 = husimi_function(k, bf.s, bf.ds, bf.u, sum(bf.ds))
        @test H ≈ H2 rtol = 1e-8
        @test q2 ≈ qs && p2 ≈ ps
    end

    @testset "full_p, ppw, w, multithreaded" begin
        @test husimi_function(st, HU_CIRCLE; full_p=true).H ≈ H atol = 1e-12 * maximum(H)
        @test husimi_function(st, HU_CIRCLE; multithreaded=false).H ≈ H atol = 1e-12 * maximum(H)
        Hb, qb, _ = husimi_function(st, HU_CIRCLE; ppw=3000.0)
        hu_check_constant(Hb, qb, ps, k; var=false)
        Hw, _, pw = husimi_function(st, HU_CIRCLE; w=1.0)
        @test hu_err(Hw, H) > 1e-3                                    # a narrow window truncates the Gaussian
    end
end

@testset "husimi_function(::BasisEigenstate) - circle J1 cos(theta) mode" begin
    st = hu_j1_state()
    k = st.k
    H, qs, ps = husimi_function(st, HU_CIRCLE)
    hu_check_density(H, qs, ps)
    @test H ≈ reverse(H; dims=2) atol = 1e-12 * maximum(H)            # real state
    # u = A cos(theta) = A cos(k p1 s) with p1 = 1/k, theta = s on the unit circle: H(q, p) is the cos plane-wave density.
    # The q-marginal (two Gaussians) cannot be used, the interference term only averages out over a full period.
    p1 = 1 / k
    # the boundary nodes enter with their cumulative weights (one node spacing offset, 0.003 rad), which perturbs the
    # density at the 1e-3 level
    @test hu_err(H, hu_expected((q, p) -> hu_cos_density(k, p1, q, p), qs, ps)) < 5e-3
    @test husimi_function(st, HU_CIRCLE; full_p=true).H ≈ H atol = 1e-12 * maximum(H)
end

@testset "husimi_function(::BasisEigenstate) - single-axis reflection basis on the Limacon" begin
    lim = LimaconBilliard(0.3)
    basis = RealPlaneWaves(3; sym_y=1)
    st = BasisEigenstate(10.0, ones(basis.dim), 0.1, basis)
    H, qs, ps = husimi_function(st, lim)
    hu_check_density(H, qs, ps)
    @test H ≈ reverse(H; dims=2) atol = 1e-10 * maximum(H)            # real u: p -> -p symmetry
    bf = boundary_function(st, lim)
    H2, _, _ = husimi_function(st.k, bf.s, bf.ds, bf.u, sum(bf.ds); c=10.0)
    @test size(H2) == size(H)
    @test qs[end] < sum(crv.length for crv in full_boundary(lim))     # grid covers one period of the full boundary
end

@testset "husimi_function(::BIMEigenstate)" begin
    @testset "uniform grid, J02 mode: constant u" begin
        s, ds, L = hu_uniform_grid(Float64, 256)
        st = hu_bim_state(HU_J02, fill(-HU_J02 * besselj(1, HU_J02), 256), hu_pts(s, ds))
        H, qs, ps = husimi_function(st, HU_CIRCLE, hu_SweepStub())
        hu_check_density(H, qs, ps)
        hu_check_constant(H, qs, ps, st.k; var=false)                 # exp(-k) at |p| = 1 is not negligible for k = 5.5: <p^2> is truncated
        @test husimi_function(st, HU_CIRCLE, hu_SweepStub(); full_p=true).H ≈ H atol = 1e-12 * maximum(H)
    end

    @testset "graded grid, J01 mode: constant u" begin
        s, ds, _ = hu_graded_grid(512)
        st = hu_bim_state(HU_J01, fill(-HU_J01 * besselj(1, HU_J01), 512), hu_pts(s, ds))
        H, qs, ps = husimi_function(st, HU_CIRCLE, hu_SweepStub())
        hu_check_density(H, qs, ps)
        hu_check_constant(H, qs, ps, st.k; var=false)
    end

    @testset "missing u is recomputed through the kernel of the solver" begin
        s, ds, _ = hu_graded_grid(512)
        st = BIMEigenstate{Float64,Float64}(HU_J01, 0.0, nothing, 1e-9, hu_pts(s, ds), nothing, nothing)
        H, qs, ps = husimi_function(st, HU_CIRCLE, hu_SweepStub())    # the stub returns a constant u
        hu_check_constant(H, qs, ps, st.k; var=false)
    end

    @testset "several states: one HusimiFunction each, equal to the single-state result" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        pts = hu_pts(s, ds)
        s1 = hu_bim_state(HU_J02, fill(1.0, 256), pts)
        s2 = hu_bim_state(HU_J02 + 0.5, cos.(10 .* s), pts)
        hs = husimi_function([s1, s2], HU_CIRCLE, hu_SweepStub())
        @test hs isa Vector{HusimiFunction{Float64}}
        @test length(hs) == 2
        for (i, st) in enumerate((s1, s2))
            @test hu_same(hs[i], husimi_function(st, HU_CIRCLE, hu_SweepStub()))
            @test hs[i].k === st.k                                    # k of the own state of every object
        end
        @test hs[1].ps[1] ≈ -1 && hs[1].qs[1] >= 0
    end

    @testset "single state: HusimiFunction with k = state.k" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        st = hu_bim_state(HU_J02, fill(1.0, 256), hu_pts(s, ds))
        hf = husimi_function(st, HU_CIRCLE, hu_SweepStub())
        @test hf isa HusimiFunction{Float64} && hf.k === st.k
        @test hf.H isa Matrix{Float64} && hf.qs isa Vector{Float64} && hf.ps isa Vector{Float64}
        @test size(hf.H) == (length(hf.qs), length(hf.ps))
    end
end

# Solver-free form husimi_function(state, billiard): composes the solver-free boundary_function(state, billiard).
@testset "husimi_function(::BIMEigenstate, billiard) - solver-free" begin
    @testset "uniform grid, J02 mode: analytic constant density" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        st = hu_bim_state(HU_J02, fill(-HU_J02 * besselj(1, HU_J02), 256), hu_pts(s, ds))
        hf = husimi_function(st, HU_CIRCLE)
        @test hf isa HusimiFunction{Float64} && hf.k === st.k
        H, qs, ps = hf
        hu_check_density(H, qs, ps)
        hu_check_constant(H, qs, ps, st.k; var=false)
    end

    @testset "graded grid, cos(10 s): analytic two-Gaussian density" begin
        s, ds, _ = hu_graded_grid(512)
        k = HU_K
        st = hu_bim_state(k, cos.(k * HU_P0 .* s), hu_pts(s, ds))
        H, qs, ps = husimi_function(st, HU_CIRCLE)
        hu_check_density(H, qs, ps)
        E = hu_expected((q, p) -> hu_cos_density(k, HU_P0, q, p), qs, ps)
        @test hu_err(H, E) < 1e-5
    end

    @testset "equals the solver-taking result for a u-only state" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        st = hu_bim_state(HU_J02, cos.(10 .* s), hu_pts(s, ds))
        @test hu_same(husimi_function(st, HU_CIRCLE), husimi_function(st, HU_CIRCLE, hu_SweepStub()))
        @test hu_same(husimi_function(st, HU_CIRCLE; c=5.0, w=5.0, full_p=true),
                      husimi_function(st, HU_CIRCLE, hu_SweepStub(); c=5.0, w=5.0, full_p=true))
        @test hu_same(husimi_function(st, HU_CIRCLE), husimi_function(boundary_function(st, HU_CIRCLE)))
    end

    @testset "missing u throws ArgumentError" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        st = BIMEigenstate{Float64,Float64}(HU_J01, 0.0, nothing, 1e-9, hu_pts(s, ds), nothing, nothing)
        @test_throws ArgumentError husimi_function(st, HU_CIRCLE)
    end
end

@testset "husimi_function(states, billiard) - several basis eigenstates" begin
    s0 = hu_j0_state(); s1 = hu_j1_state()

    @testset "one HusimiFunction each, equal to the single-state results" begin
        hs = husimi_function([s0, s1], HU_CIRCLE)
        @test hs isa Vector{HusimiFunction{Float64}}
        @test length(hs) == 2
        for (i, st) in enumerate((s0, s1))
            ref = husimi_function(st, HU_CIRCLE)                      # batch u is Rellich-normalized: H agrees up to rounding
            @test hs[i].H ≈ ref.H rtol = 1e-10
            @test hs[i].qs == ref.qs && hs[i].ps == ref.ps && hs[i].k == ref.k
            @test hs[i].k === st.k                                    # k of the own state of every object
        end
        hc = husimi_function([s0, s1], HU_CIRCLE; c=5.0)
        @test length(hc[1].ps) == 2 * ceil(Int, 5 * sqrt(s0.k)) + 1
    end

    @testset "a state with a coefficient vector of the wrong length throws (batch path, no per-state dropping)" begin
        # The vector method builds the boundary functions in one batch (boundary_states), so a malformed state
        # makes the whole batch throw instead of being dropped as before.
        bad = BasisEigenstate(s0.k, [1.0, 2.0], 1e-12, s0.basis)       # coefficient vector of the wrong length
        @test_throws DimensionMismatch husimi_function([s0, bad, s1], HU_CIRCLE)
        @test_throws DimensionMismatch husimi_function([bad, bad], HU_CIRCLE)
    end
end

@testset "SpectralData wrappers - husimi_function" begin
    @testset "basis eigenstates" begin
        s0 = hu_j0_state(); s1 = hu_j1_state()
        data = SpectralData(HU_CIRCLE, nothing, [s0.k, s1.k], [1e-12, 1e-12], [false, false]; states=[s0, s1])
        hf1 = husimi_function(data, 2)
        @test hf1 isa HusimiFunction{Float64} && hf1.k === s1.k
        @test hu_same(hf1, husimi_function(s1, HU_CIRCLE))
        @test hu_same(husimi_function(data, 1; c=5.0), husimi_function(s0, HU_CIRCLE; c=5.0))
        hs = husimi_function(data)
        hr = husimi_function([s0, s1], HU_CIRCLE)
        @test hs isa Vector{HusimiFunction{Float64}} && length(hs) == 2
        @test all(hu_same(a, b) for (a, b) in zip(hs, hr))
        @test hs[1].k === s0.k && hs[2].k === s1.k
    end

    @testset "BIM eigenstates go through the solver" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        st = hu_bim_state(HU_J02, ones(256), hu_pts(s, ds))
        data = SpectralData(HU_CIRCLE, hu_SweepStub(), [HU_J02], [1e-9], [false]; states=[st])
        hf = husimi_function(data, 1)
        @test hf isa HusimiFunction{Float64} && hf.k === st.k
        @test hu_same(hf, husimi_function(st, HU_CIRCLE, hu_SweepStub()))
        hs = husimi_function(data)
        @test hs isa Vector{HusimiFunction{Float64}} && length(hs) == 1
        @test hu_same(hs[1], husimi_function(st, HU_CIRCLE, hu_SweepStub())) && hs[1].k === st.k
    end

    @testset "data without stored states" begin
        data = SpectralData(HU_CIRCLE, nothing, [3.0], [1e-12], [false])
        @test_throws ArgumentError husimi_function(data)
        @test_throws ArgumentError husimi_function(data, 1)
    end
end

# Billiards with holes: the boundary data come from a CompositeBIMSolver discretization of the annulus (unit circle with an
# eccentric hole of radius 0.3, 32 outer and 24 hole nodes). No eigen solve: u is hand-built.
function hu_annulus_composite_pts(k)
    annulus = AnnularBilliard(0.3, 0.1)
    solver = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=32), CombinedFieldIntegralEquationSolver(5.0, annulus; min_pts=24))
    return annulus, solver, evaluate_points(solver, annulus, k)
end

# Outer-loop selection of husimi_function(boundary_function(state, billiard, ...)): the BoundaryFunction carries the outer
# nodes (bf.outer, from _outer_range of the points) and the outer perimeter bf.L, and the Husimi function uses only those.
@testset "husimi_function(boundary_function(...)) outer-loop selection" begin
    @testset "_outer_range: every node without composite bookkeeping, the outer block with it" begin
        s, ds, _ = hu_uniform_grid(Float64, 64)
        @test _outer_range(hu_pts(s, ds)) == 1:64
        _, _, pts = hu_annulus_composite_pts(3.0)
        @test _outer_range(pts) == 1:32                                   # the 32 outer nodes, 24 hole nodes follow
    end

    @testset "billiard without holes: the full boundary, perimeter of the full billiard" begin
        s, ds, _ = hu_uniform_grid(Float64, 64)
        pts = hu_pts(s, ds)
        u = collect(1.0:64.0)
        bf = boundary_function(hu_bim_state(HU_J02, u, pts), HU_CIRCLE)
        @test bf.outer == 1:64 && bf.u === u && bf.s === pts.s && bf.ds === pts.ds
        @test bf.L ≈ 2π
        # the perimeter is that of the full boundary, not of the fundamental domain
        rect = RectangleBilliard(1.0, 0.6)
        prect = hu_pts(collect(0.0:3.0), fill(1.0, 4))
        @test boundary_function(hu_bim_state(HU_J02, ones(4), prect), rect).L ≈ 2 * (2 * 1.0 + 2 * 0.6)
    end

    @testset "billiard with a hole: the outer block only" begin
        annulus, solver, pts = hu_annulus_composite_pts(3.0)
        u = collect(1.0:length(pts))
        st = hu_bim_state(3.0, u, pts)
        bf = boundary_function(st, annulus, solver)
        @test length(bf.u) == length(pts)                                 # u and s cover every loop
        @test bf.outer == 1:32
        @test bf.L ≈ 2π                                                   # outer perimeter, not 2 pi (1 + 0.3)
        @test sum(bf.ds[bf.outer]) ≈ bf.L
        @test all(<(bf.L), bf.s[bf.outer]) && issorted(bf.s[bf.outer])
        @test all(x -> hypot(x[1], x[2]) ≈ 1.0, pts.xy[bf.outer])         # no hole node in the outer block
        # the Husimi function is that of the outer block alone
        hf = husimi_function(bf)
        ref = husimi_function(bf.k, bf.s[1:32], bf.ds[1:32], u[1:32], bf.L)
        @test hf.H == ref.H && hf.qs == ref.qs && hf.ps == ref.ps
    end

    @testset "billiard with a hole, points without composite bookkeeping: no error, every node is the outer loop" begin
        # behaviour change of the streamlined state API: this configuration used to raise an ArgumentError naming
        # CompositeBIMSolver; the BoundaryFunction now treats all nodes as the outer loop
        s, ds, _ = hu_uniform_grid(Float64, 64)
        annulus = AnnularBilliard(0.3, 0.1)
        st = hu_bim_state(3.0, ones(64), hu_pts(s, ds))
        bf = boundary_function(st, annulus)
        @test bf.outer == 1:64 && bf.L ≈ 2π
        hf = @test_nowarn husimi_function(bf)
        @test hf isa HusimiFunction{Float64}
        @test sum(hf.H) ≈ 1 rtol = 1e-10
    end
end

@testset "husimi_function(::BIMEigenstate) on a billiard with a hole" begin
    annulus, solver, pts = hu_annulus_composite_pts(3.0)
    st = hu_bim_state(3.0, fill(1.0, length(pts)), pts)
    H, qs, ps = husimi_function(st, annulus, solver; c=5.0)
    @test size(H) == (length(qs), length(ps))
    @test sum(H) ≈ 1 rtol = 1e-10
    @test all(isfinite, H) && all(>=(0), H)
    @test 0 <= minimum(qs) && maximum(qs) < 2π + 1e-12                   # the outer perimeter only, no hole arclength
end

@testset "husimi_function(boundary_function(state, billiard)) equals husimi_function(state, billiard)" begin
    @testset "basis state" begin
        for st in (hu_j0_state(), hu_j1_state())
            bf = boundary_function(st, HU_CIRCLE)
            hf = husimi_function(bf)
            ref = husimi_function(st, HU_CIRCLE)
            @test hf isa HusimiFunction{Float64} && hf.k === st.k
            @test hf.H == ref.H && hf.qs == ref.qs && hf.ps == ref.ps
            # the same density on a prescribed grid, evaluated from the stored arrays of the boundary function
            qs = ref.qs[1:7:end]; ps = collect(range(-1.0, 1.0; length=15))
            @test husimi_function(bf, qs, ps; full_p=true).H ≈ husimi_function(bf.k, bf.s, bf.ds, bf.u, bf.L, qs, ps; full_p=true).H
        end
    end

    @testset "BIM state with stored u (uniform grid) and with the solver form" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        pts = hu_pts(s, ds)
        st = hu_bim_state(HU_J02, cos.(10 .* s), pts)
        ref = husimi_function(st, HU_CIRCLE, hu_SweepStub())
        for bf in (boundary_function(st, HU_CIRCLE), boundary_function(st, HU_CIRCLE, hu_SweepStub()))
            hf = husimi_function(bf)
            @test hf.H == ref.H && hf.qs == ref.qs && hf.ps == ref.ps && hf.k === st.k
        end
        # without a billiard L = sum(ds) = 2 pi of the circle, so the solver-free object gives the same density
        @test husimi_function(boundary_function(st)).H == ref.H
        # prescribed grid: consistent with the numeric form on the stored arrays
        bf = boundary_function(st, HU_CIRCLE)
        qs = [0.3, 1.1, 2.9]; ps = collect(range(-1.0, 1.0; length=15))
        @test husimi_function(bf, qs, ps; full_p=true).H ≈ husimi_function(st.k, pts, st.u, qs, ps; full_p=true).H
    end

    @testset "BIM state with a graded grid (general path) and a recomputed u" begin
        s, ds, _ = hu_graded_grid(512)
        pts = hu_pts(s, ds)
        st = hu_bim_state(HU_J01, fill(-HU_J01 * besselj(1, HU_J01), 512), pts)
        ref = husimi_function(st, HU_CIRCLE, hu_SweepStub())
        hf = husimi_function(boundary_function(st, HU_CIRCLE))
        @test hf.H == ref.H && hf.qs == ref.qs && hf.ps == ref.ps
        st_missing = BIMEigenstate{Float64,Float64}(HU_J01, 0.0, nothing, 1e-9, pts, nothing, nothing)
        refm = husimi_function(st_missing, HU_CIRCLE, hu_SweepStub())
        hfm = husimi_function(boundary_function(st_missing, HU_CIRCLE, hu_SweepStub()))
        @test hfm.H == refm.H && hfm.k === st_missing.k
    end

    @testset "billiard with a hole: only the outer loop enters" begin
        annulus, solver, pts = hu_annulus_composite_pts(3.0)
        st = hu_bim_state(3.0, collect(range(1.0, 2.0, length(pts))), pts)
        bf = boundary_function(st, annulus, solver)
        @test bf.outer == 1:32 && bf.L ≈ 2π
        hf = husimi_function(bf; c=5.0)
        ref = husimi_function(st, annulus, solver; c=5.0)
        @test hf.H == ref.H && hf.qs == ref.qs && hf.ps == ref.ps && hf.k === st.k
        @test maximum(hf.qs) < bf.L                                           # the outer perimeter only
        # the solver-free object has the same outer-loop L
        @test husimi_function(boundary_function(st); c=5.0).H ≈ ref.H
    end
end

@testset "HusimiFunction.k equals state.k in the single, vector and SpectralData forms" begin
    s0 = hu_j0_state(); s1 = hu_j1_state()
    @test husimi_function(s0, HU_CIRCLE).k === s0.k
    @test [h.k for h in husimi_function([s0, s1], HU_CIRCLE)] == [s0.k, s1.k]
    data = SpectralData(HU_CIRCLE, nothing, [s0.k, s1.k], [1e-12, 1e-12], [false, false]; states=[s0, s1])
    @test husimi_function(data, 1).k === s0.k && husimi_function(data, 2).k === s1.k
    @test [h.k for h in husimi_function(data)] == [s0.k, s1.k]

    s, ds, _ = hu_uniform_grid(Float64, 256)
    pts = hu_pts(s, ds)
    b1 = hu_bim_state(HU_J02, ones(256), pts); b2 = hu_bim_state(HU_J02 + 0.7, ones(256), pts)
    @test husimi_function(b2, HU_CIRCLE, hu_SweepStub()).k === b2.k
    @test [h.k for h in husimi_function([b1, b2], HU_CIRCLE, hu_SweepStub())] == [b1.k, b2.k]
    bdata = SpectralData(HU_CIRCLE, hu_SweepStub(), [b1.k, b2.k], [1e-9, 1e-9], [false, false]; states=[b1, b2])
    @test husimi_function(bdata, 2).k === b2.k
    @test [h.k for h in husimi_function(bdata)] == [b1.k, b2.k]
end

struct hu_KState
    k::Float64
end

@testset "_collect_functions" begin
    f(st) = st.k > 0 ? 2 * st.k : throw(DomainError(st.k, "non-positive k"))
    states = [hu_KState(1.0), hu_KState(-2.0), hu_KState(3.0), hu_KState(-4.0), hu_KState(5.0)]

    @testset "all states succeed: Vector{F} in the order of the states" begin
        ks = [hu_KState(float(i)) for i in 1:20]
        out = _collect_functions(f, Float64, ks)
        @test out isa Vector{Float64}
        @test out == [2.0 * i for i in 1:20]
    end

    @testset "a throwing state is dropped, the order of the others is kept" begin
        @test _collect_functions(f, Float64, states) == [2.0, 6.0, 10.0]
        @test _collect_functions(f, Float64, states) isa Vector{Float64}
    end

    @testset "the failure is logged with @debug at the state's k, nothing at the default level" begin
        @test_logs (:debug, "Float64 fail at k=-2.0") (:debug, "Float64 fail at k=-4.0") min_level=Logging.Debug _collect_functions(f, Float64, states)
        @test_logs _collect_functions(f, Float64, states)
    end

    @testset "multithreaded = true and false give the same result" begin
        ks = [hu_KState(isodd(i) ? float(i) : -float(i)) for i in 1:40]
        serial = _collect_functions(f, Float64, ks; multithreaded=false)
        threaded = _collect_functions(f, Float64, ks; multithreaded=true)
        @test serial == threaded == [2.0 * i for i in 1:2:39]
    end

    @testset "every state fails: empty vector of the requested element type" begin
        out = _collect_functions(f, Float64, [hu_KState(-1.0), hu_KState(-2.0)])
        @test isempty(out) && out isa Vector{Float64}
    end

    @testset "the element type F is the container type" begin
        @test _collect_functions(st -> Float32(st.k), Float32, [hu_KState(1.0)]) isa Vector{Float32}
    end
end
