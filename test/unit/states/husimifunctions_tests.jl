# Unit tests for
#   src/states/husimifunctions.jl   (antisym_vec, _normalize_husimi!, _is_global_phase_real, _husimi_uniform_arclength_grid,
#                                    _husimi_symmetric_window, the two quadratures _husimi_uniform_arclength and the general
#                                    husimi_function(k, s, ds, u, L[, qs, ps]), the BoundaryPoints form, the eigenstate forms
#                                    for basis states and BIM states, the batched forms and the SpectralData wrappers)
# The Poincare-Husimi amplitude is h(q, p) = sum_j u_j exp(-k (s_j - q)^2 / 2) exp(-i k p (s_j - q)) ds_j and the returned
# density is H = |h|^2 / sum |h|^2. This file has no `fundamental_domain` or `inside_only` keyword to test: the
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
# FINDINGS (no source change): the general quadrature husimi_function(k, s, ds, u, L, qs, ps) maps q into [L, 2L) but the
# periodic extension s_ext ends at 2L, so for q > L - w / sqrt(k) the upper part of the Gaussian window is cut off (rows near
# q = L are wrong; the uniform fast path is unaffected). Checks on the general path use the interior rows (hu_interior) and
# the full-range checks are @test_broken.
# FINDINGS (no source change): the batched forms return (Hs, ps, qs) while the single-state forms return (H, qs, ps);
# the batched basis form drops states that throw, silently. Both are current documented/observed behaviour and are pinned below.
# The single-axis reflection path was checked through boundary_function in boundaryfunctions_tests.jl (Limacon) and is
# consistent; the single-axis bug of apply_symmetries_to_wavefunction does not affect the boundary-function path.

using StaticArrays
using LinearAlgebra: norm, dot
using SpecialFunctions: besselj
using QuantumBilliards: antisym_vec, _normalize_husimi!, _is_global_phase_real, _husimi_uniform_arclength_grid,
                        _husimi_symmetric_window, _husimi_uniform_arclength, _basis_boundary_function_pts

const HU_J01 = 2.404825557695773       # first zero of J0
const HU_J02 = 5.520078110286311       # second zero of J0
const HU_J11 = 3.8317059702075125      # first zero of J1
const HU_K = 20.0                      # sigma = 1/sqrt(k) = 0.22, window w sigma = 1.56 < L/2: fast uniform path
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

# Rows of a density with q <= L - w / sqrt(k), renormalised to sum 1. The general quadrature truncates the window for the
# remaining rows (source finding, see the broken tests); the interior rows are exact.
function hu_interior(H, qs, k, L; w=7.0)
    m = qs .<= L - w / sqrt(k)
    Hm = H[m, :]
    return Hm ./ sum(Hm), qs[m]
end

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

@testset "_husimi_uniform_arclength_grid" begin
    s = (0:9) .* 0.1; ds = fill(0.1, 10)
    @test _husimi_uniform_arclength_grid(s, ds)
    s2 = collect(s); s2[5] += 1e-3
    @test !_husimi_uniform_arclength_grid(s2, ds)
    ds2 = copy(ds); ds2[3] = 0.05
    @test !_husimi_uniform_arclength_grid(s, ds2)
    ds3 = copy(ds); ds3[1] = 0.0                                      # a zero weight (as at the curve ends of boundary nodes)
    @test !_husimi_uniform_arclength_grid(s, ds3)
    @test _husimi_uniform_arclength_grid(s2, ds; rtol=1e-2)
    @test _husimi_uniform_arclength_grid(Float32.(s), fill(0.1f0, 10))
    sg, dsg, _ = hu_graded_grid(64)
    @test !_husimi_uniform_arclength_grid(sg, dsg)
end

@testset "_husimi_symmetric_window" begin
    s = (0:9) .* 0.1; ds = fill(0.1, 10)
    x, dx, idx = _husimi_symmetric_window(s, ds, 0.35)                # nside = 1 + floor(0.35 / 0.1) = 4 nodes on each side
    @test idx == 4
    @test x ≈ [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    @test dx == fill(0.1, 7)
    @test x == -reverse(x)
    x, dx, idx = _husimi_symmetric_window(s, ds, 100.0)               # a wide window is capped at the grid size
    @test idx == 10 && length(x) == length(dx) == 19
    x, _, idx = _husimi_symmetric_window(s, ds, 0.0)                  # zero width: only the centre node
    @test idx == 1 && x == [0.0]
    dsv = collect(1.0:10.0)
    _, dx, _ = _husimi_symmetric_window(s, dsv, 0.25)                 # weights are mirrored around the centre node
    @test dx == [3.0, 2.0, 1.0, 2.0, 3.0]
end

@testset "husimi_function(k, s, ds, u, L) - constant boundary function (fast uniform path)" begin
    k = HU_K
    s, ds, L = hu_uniform_grid(Float64, 256)
    @test _husimi_uniform_arclength_grid(s, ds) && 7 / sqrt(k) < L / 2        # the fast path is selected
    u = ones(256)

    @testset "analytic Gaussian in p, independent of q" begin
        H, qs, ps = husimi_function(k, s, ds, u, L)
        hu_check_density(H, qs, ps)
        hu_check_constant(H, qs, ps, k)
        @test hu_err(H, hu_expected((q, p) -> hu_const_density(k, p), qs, ps)) < 1e-8
    end

    @testset "grids: signed momentum grid in [-1, 1] with p = 0 at the centre, boundary positions on the boundary" begin
        H, qs, ps = husimi_function(k, s, ds, u, L)
        np = ceil(Int, 10 * sqrt(k))                                  # c = 10 samples per sigma in p
        @test length(ps) == 2np + 1 && ps[1] ≈ -1 && ps[end] ≈ 1
        @test ps ≈ -reverse(ps) atol = 1e-14
        @test issorted(qs) && qs[1] >= 0 && qs[end] < L
        σ = 1 / sqrt(k)
        @test all(diff(qs) .>= ds[1] - 1e-12) && maximum(diff(qs)) <= σ / 10 + ds[1]
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
    @test length(qs) == 256                                           # one boundary position per node at this k

    @testset "real scale factors and a global complex phase do not change the density" begin
        @test husimi_function(k, s, ds, -3.7 .* u, L)[1] ≈ H atol = 1e-12 * maximum(H)
        Hp, _, ps = husimi_function(k, s, ds, exp(0.8im) .* u, L)        # real up to a global phase: full_p = false is accepted
        @test Hp ≈ H atol = 1e-12 * maximum(H)
    end

    @testset "a cyclic shift of u by n nodes shifts the density by n boundary positions" begin
        Hs, _, _ = husimi_function(k, s, ds, circshift(u, 17), L)
        @test Hs ≈ circshift(H, (17, 0)) atol = 1e-12 * maximum(H)
    end
end

@testset "husimi_function(k, s, ds, u, L, qs, ps) - prescribed grids (general quadrature)" begin
    k = HU_K; p0 = HU_P0
    s, ds, L = hu_uniform_grid(Float64, 256)
    u = cos.(k * p0 .* s)
    f = (q, p) -> hu_cos_density(k, p0, q, p)
    qs = [0.3, 1.1, 2.9, 4.0, 4.6]                                   # all below L - w / sqrt(k) = 4.72, see the broken test below
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

    @testset "agrees with the fast uniform evaluator on the same grid" begin
        Hf, qf, pf = _husimi_uniform_arclength(k, s, ds, u, L; full_p=true)
        Hg, _, _ = husimi_function(k, s, ds, u, L, qf, pf; full_p=true)
        keep = qf .<= L - 7 / sqrt(k)
        Gk, Fk = Hg[keep, :], Hf[keep, :]
        @test Gk ./ sum(Gk) ≈ Fk ./ sum(Fk) atol = 1e-10 * maximum(Fk ./ sum(Fk))     # each density is normalised over its own rows
    end

    @testset "the Gaussian window is not truncated near q = L" begin
        # The general quadrature shifts q into [L, 2L) while the periodic extension of s ends at 2L, so for q > L - w / sqrt(k)
        # the upper part of the window is cut off and the density deviates from the analytic one (source finding).
        H, qo, po = husimi_function(k, s, ds, u, L, [5.7], ps; full_p=true)
        @test_broken hu_err(H, hu_expected(f, qo, po)) < 1e-8
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

@testset "husimi_function - non-uniform arclength grid (general path)" begin
    k = HU_K; p0 = HU_P0
    s, ds, L = hu_graded_grid(512)
    @test !_husimi_uniform_arclength_grid(s, ds)

    @testset "constant u: analytic Gaussian; grids are equispaced in q with spacing <= sigma / c" begin
        H, qs, ps = husimi_function(k, s, ds, ones(512), L)
        hu_check_density(H, qs, ps)
        σ = 1 / sqrt(k)
        @test qs[1] == 0 && qs[end] < L
        @test all(diff(qs) .≈ qs[2] - qs[1]) && qs[2] - qs[1] <= σ / 10
        @test length(ps) == 2 * ceil(Int, 10 * sqrt(k)) + 1
        Hi, qi = hu_interior(H, qs, k, L)
        hu_check_constant(Hi, qi, ps, k)
        @test_broken maximum(abs, H .- H[1:1, :]) <= 1e-8 * maximum(H)           # rows with q > L - w sigma lose part of the window
    end

    @testset "cos(k p0 s): analytic density with interference" begin
        H, qs, ps = husimi_function(k, s, ds, cos.(k * p0 .* s), L)
        @test H ≈ reverse(H; dims=2) atol = 1e-12 * maximum(H)
        Hi, qi = hu_interior(H, qs, k, L)
        @test hu_err(Hi, hu_expected((q, p) -> hu_cos_density(k, p0, q, p), qi, ps)) < 1e-6
        @test_broken hu_err(H, hu_expected((q, p) -> hu_cos_density(k, p0, q, p), qs, ps)) < 1e-6
    end
end

@testset "husimi_function - window wider than half the period uses the general path" begin
    k = 4.0; p0 = HU_P0                                               # sigma = 0.5, w sigma = 3.5 > L / 2, k p0 = 2
    s, ds, L = hu_uniform_grid(Float64, 256)
    @test _husimi_uniform_arclength_grid(s, ds) && 7 / sqrt(k) >= L / 2
    H, qs, ps = husimi_function(k, s, ds, cos.(k * p0 .* s), L)
    @test qs[1] == 0 && all(diff(qs) .≈ qs[2] - qs[1])                # the general path builds equispaced boundary positions
    hu_check_density(H, qs, ps)
    Hi, qi = hu_interior(H, qs, k, L)
    @test hu_err(Hi, hu_expected((q, p) -> hu_cos_density(k, p0, q, p), qi, ps)) < 1e-6
    Hc, qc, pc = husimi_function(k, s, ds, ones(256), L)
    Hci, qci = hu_interior(Hc, qc, k, L)
    hu_check_constant(Hci, qci, pc, k; var=false)
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
    @test husimi_function(k, pts, u) == husimi_function(k, s, ds, u, sum(ds))
    @test husimi_function(k, pts, u; c=5.0, w=6.0) == husimi_function(k, s, ds, u, sum(ds); c=5.0, w=6.0)
    qs = [0.2, 1.0, 3.0]; ps = collect(range(0, 1; length=11))
    @test husimi_function(k, pts, u, qs, ps) == husimi_function(k, s, ds, u, sum(ds), qs, ps)
    @test husimi_function(k, pts, exp.(1im .* k .* HU_P0 .* s), qs, collect(range(-1, 1; length=11)); full_p=true)[1] ≈
          husimi_function(k, s, ds, exp.(1im .* k .* HU_P0 .* s), sum(ds), qs, collect(range(-1, 1; length=11)); full_p=true)[1]
end

@testset "husimi_function(::BasisEigenstate) - circle J0 mode" begin
    st = hu_j0_state()
    k = st.k
    H, qs, ps = husimi_function(st, HU_CIRCLE)

    @testset "constant normal derivative gives the analytic Gaussian in p, independent of q" begin
        hu_check_density(H, qs, ps)
        Hi, qi = hu_interior(H, qs, k, 2pi)
        hu_check_constant(Hi, qi, ps, k; var=false)
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
        u, pts, _ = _basis_boundary_function_pts(st, HU_CIRCLE)
        H2, q2, p2 = husimi_function(k, pts, u)
        @test H ≈ H2 rtol = 1e-8
        @test q2 ≈ qs && p2 ≈ ps
    end

    @testset "full_p, b, w, multithreaded" begin
        @test husimi_function(st, HU_CIRCLE; full_p=true)[1] ≈ H atol = 1e-12 * maximum(H)
        @test husimi_function(st, HU_CIRCLE; multithreaded=false)[1] ≈ H atol = 1e-12 * maximum(H)
        Hb, qb, _ = husimi_function(st, HU_CIRCLE; b=3000.0)
        Hbi, qbi = hu_interior(Hb, qb, k, 2pi)
        hu_check_constant(Hbi, qbi, ps, k; var=false)
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
    # The q-marginal (two Gaussians) cannot be used on the interior rows, the interference term only averages out over a full period.
    p1 = 1 / k
    Hi, qi = hu_interior(H, qs, k, 2pi)
    # the boundary nodes enter with their cumulative weights (one node spacing offset, 0.003 rad), which perturbs the
    # density at the 1e-3 level
    @test hu_err(Hi, hu_expected((q, p) -> hu_cos_density(k, p1, q, p), qi, ps)) < 5e-3
    @test husimi_function(st, HU_CIRCLE; full_p=true)[1] ≈ H atol = 1e-12 * maximum(H)
end

@testset "husimi_function(::BasisEigenstate) - single-axis reflection basis on the Limacon" begin
    lim = LimaconBilliard(0.3)
    basis = RealPlaneWaves(3; sym_y=1)
    st = BasisEigenstate(10.0, ones(basis.dim), 0.1, basis)
    H, qs, ps = husimi_function(st, lim)
    hu_check_density(H, qs, ps)
    @test H ≈ reverse(H; dims=2) atol = 1e-10 * maximum(H)            # real u: p -> -p symmetry
    u, pts, _ = _basis_boundary_function_pts(st, lim)
    H2, _, _ = husimi_function(st.k, pts, u; c=10.0)
    @test size(H2) == size(H)
    @test qs[end] < sum(crv.length for crv in full_boundary(lim))     # grid covers one period of the full boundary
end

@testset "husimi_function(::BIMEigenstate)" begin
    @testset "uniform grid, J02 mode: constant u, fast path" begin
        s, ds, L = hu_uniform_grid(Float64, 256)
        st = hu_bim_state(HU_J02, fill(-HU_J02 * besselj(1, HU_J02), 256), hu_pts(s, ds))
        @test 7 / sqrt(st.k) < L / 2 && _husimi_uniform_arclength_grid(s, ds)
        H, qs, ps = husimi_function(st, HU_CIRCLE, hu_SweepStub())
        hu_check_density(H, qs, ps)
        hu_check_constant(H, qs, ps, st.k; var=false)                 # exp(-k) at |p| = 1 is not negligible for k = 5.5: <p^2> is truncated
        @test husimi_function(st, HU_CIRCLE, hu_SweepStub(); full_p=true)[1] ≈ H atol = 1e-12 * maximum(H)
    end

    @testset "graded grid, J01 mode: constant u, general path" begin
        s, ds, _ = hu_graded_grid(512)
        st = hu_bim_state(HU_J01, fill(-HU_J01 * besselj(1, HU_J01), 512), hu_pts(s, ds))
        H, qs, ps = husimi_function(st, HU_CIRCLE, hu_SweepStub())
        hu_check_density(H, qs, ps)
        Hi, qi = hu_interior(H, qs, st.k, 2pi)
        hu_check_constant(Hi, qi, ps, st.k; var=false)
    end

    @testset "missing u is recomputed through the kernel of the solver" begin
        s, ds, _ = hu_graded_grid(512)
        st = BIMEigenstate{Float64,Float64}(HU_J01, 0.0, nothing, 1e-9, hu_pts(s, ds), nothing, nothing)
        H, qs, ps = husimi_function(st, HU_CIRCLE, hu_SweepStub())    # the stub returns a constant u
        Hi, qi = hu_interior(H, qs, st.k, 2pi)
        hu_check_constant(Hi, qi, ps, st.k; var=false)
    end

    @testset "several states: Hs, ps, qs, each equal to the single-state result" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        pts = hu_pts(s, ds)
        s1 = hu_bim_state(HU_J02, fill(1.0, 256), pts)
        s2 = hu_bim_state(HU_J02, cos.(10 .* s), pts)                 # note: tests the (Hs, ps, qs) order of the batch form
        Hs, pss, qss = husimi_function([s1, s2], HU_CIRCLE, hu_SweepStub())
        @test length(Hs) == length(pss) == length(qss) == 2
        for (i, st) in enumerate((s1, s2))
            H, qs, ps = husimi_function(st, HU_CIRCLE, hu_SweepStub())
            @test Hs[i] == H && qss[i] == qs && pss[i] == ps
        end
        @test pss[1][1] ≈ -1 && qss[1][1] >= 0                         # second element holds the momentum grids
    end
end

@testset "husimi_function(states, billiard) - several basis eigenstates" begin
    s0 = hu_j0_state(); s1 = hu_j1_state()

    @testset "Hs, ps, qs equal the single-state results" begin
        Hs, pss, qss = husimi_function([s0, s1], HU_CIRCLE)
        @test Hs isa Vector{Matrix{Float64}} && pss isa Vector{Vector{Float64}} && qss isa Vector{Vector{Float64}}
        @test length(Hs) == 2
        for (i, st) in enumerate((s0, s1))
            H, qs, ps = husimi_function(st, HU_CIRCLE)
            @test Hs[i] == H && qss[i] == qs && pss[i] == ps
        end
        Hc, pc, qc = husimi_function([s0, s1], HU_CIRCLE; c=5.0)
        @test length(pc[1]) == 2 * ceil(Int, 5 * sqrt(s0.k)) + 1
    end

    @testset "a state that throws is dropped from the result (current behaviour)" begin
        bad = BasisEigenstate(s0.k, [1.0, 2.0], 1e-12, s0.basis)       # coefficient vector of the wrong length
        Hs, pss, qss = @test_logs husimi_function([s0, bad, s1], HU_CIRCLE)
        @test length(Hs) == length(pss) == length(qss) == 2
        @test Hs[1] == husimi_function(s0, HU_CIRCLE)[1] && Hs[2] == husimi_function(s1, HU_CIRCLE)[1]
    end
end

@testset "SpectralData wrappers - husimi_function" begin
    @testset "basis eigenstates" begin
        s0 = hu_j0_state(); s1 = hu_j1_state()
        data = SpectralData(HU_CIRCLE, nothing, [s0.k, s1.k], [1e-12, 1e-12], [false, false]; states=[s0, s1])
        @test husimi_function(data, 2) == husimi_function(s1, HU_CIRCLE)
        @test husimi_function(data, 1; c=5.0) == husimi_function(s0, HU_CIRCLE; c=5.0)
        Hs, pss, qss = husimi_function(data)
        Hr, pr, qr = husimi_function([s0, s1], HU_CIRCLE)
        @test Hs == Hr && pss == pr && qss == qr
    end

    @testset "BIM eigenstates go through the solver" begin
        s, ds, _ = hu_uniform_grid(Float64, 256)
        st = hu_bim_state(HU_J02, ones(256), hu_pts(s, ds))
        data = SpectralData(HU_CIRCLE, hu_SweepStub(), [HU_J02], [1e-9], [false]; states=[st])
        @test husimi_function(data, 1) == husimi_function(st, HU_CIRCLE, hu_SweepStub())
        Hs, pss, qss = husimi_function(data)
        @test length(Hs) == 1 && Hs[1] == husimi_function(st, HU_CIRCLE, hu_SweepStub())[1]
    end

    @testset "data without stored states" begin
        data = SpectralData(HU_CIRCLE, nothing, [3.0], [1e-12], [false])
        @test_throws ArgumentError husimi_function(data)
        @test_throws ArgumentError husimi_function(data, 1)
    end
end
