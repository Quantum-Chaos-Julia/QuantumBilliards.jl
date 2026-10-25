# Integration tests for the SpectralData state API (src/states/*.jl, src/spectra/spectralutils.jl): eigenstates stored by
# `compute_spectrum` (BeynSolver with eigenvectors=true, VerginiSaracenoSolver with eigenvectors=true) are reconstructed with
# `boundary_function`, `husimi_function` and `wavefunction` and compared with analytic mode shapes. Every comparison is made
# after normalisation (least-squares scale or division by the mean), so no scale, phase or tension is pinned.
# Conventions of the current API: `husimi_function(data, i)` returns (H, qs, ps) with H of size (length(qs), length(ps)),
# `husimi_function(data)` returns (Hs, ps, qs); `wavefunction(data, i)` returns (Psi, x, y) with Psi[ix, iy];
# `boundary_function(state, billiard, solver)` is needed for Beyn BIM states, because `state.u` is not stored (findings F8/F9 of
# the plan: the Husimi general-path edge rows are wrong, so Husimi checks use interior rows only).

using LinearAlgebra: norm, dot
using Statistics: mean

const _SS_SPECTRA = Dict{Symbol,Any}()

# Circle (radius 1), CFIE Beyn, window [2.2, 4.0]: j_{0,1} = 2.4048 (once) and the degenerate pair j_{1,1} = 3.8317.
function _ss_circle_spectrum()
    get!(_SS_SPECTRA, :circle) do
        billiard = CircleBilliard(1.0)
        kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
        solver = BeynSolver(kernel; use_chebyshev=false, eigenvectors=true)
        compute_spectrum(solver, billiard, 2.2, 4.0; multithreaded=false, show_progress=false)
    end
end

# Rectangle (RectangleBilliard(1.0, 0.7), full 2.0 x 1.4 rectangle centred at the origin), VS, even-even sector, k in [20, 23].
function _ss_rectangle_spectrum()
    get!(_SS_SPECTRA, :rectangle) do
        billiard = RectangleBilliard(1.0, 0.7)
        basis = RealPlaneWaves(20; sym_x=1, sym_y=1)
        solver = VerginiSaracenoSolver(5.0, 10.0; eigenvectors=true)
        compute_spectrum(solver, basis, billiard, 20.0, 23.0, 0.2; ten_tol=nothing, multithreaded=false, show_progress=false)
    end
end

# Relative residual of `psi` after the best least-squares (complex) scaling onto `ref`.
function _ss_scaled_residual(psi, ref)
    c = dot(ref, psi) / dot(ref, ref)
    return norm(psi - c * ref) / norm(psi)
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10, eigenvectors=true
# billiard: Circle, radius 1
# symmetry: None
# reference: analytic; the J0 mode psi = J_0(k r) has constant normal derivative on the boundary, a Husimi function that is
#   concentrated at p = 0 (normal incidence) and is independent of q, and a wavefunction proportional to besselj(0, k r)
# functions to test: boundary_function, husimi_function, wavefunction (SpectralData methods / BIMEigenstate)
@testset "States in SpectralData - Circle CFIE - J0 mode" begin
    data = _ss_circle_spectrum()
    billiard = data.billiard
    solver = data.solver
    j01 = circle_spectrum(1.0, 2.3, 2.5)[1]
    i0 = argmin(abs.(data.k .- j01))
    @test abs(data.k[i0] - j01) < SPECTRUM_ATOL_BIM

    # boundary function: constant after normalisation
    u, s, _ = boundary_function(data.states[i0], billiard, solver.kernel)
    @test length(u) == length(s)
    um = mean(u)
    @test abs(um) > 0
    @test maximum(abs.(u ./ um .- 1)) < 1e-8

    # Husimi function: normalised, concentrated at p = 0, independent of q on interior rows
    H, qs, ps = husimi_function(data, i0)
    @test size(H) == (length(qs), length(ps))
    @test sum(H) ≈ 1 atol = 1e-8
    @test all(H .>= -1e-12)
    L = 2pi
    rows = findall(q -> 0.15 * L <= q <= 0.85 * L, qs)
    @test !isempty(rows)
    Hp = vec(sum(H[rows, :]; dims=1))
    # the coherent-state window is broad at k = 2.4 (width ~ 1/sqrt(k)), so concentration means: single maximum at p = 0,
    # p -> -p symmetric, monotone decay with |p|, and the edge value well below the peak
    ip0 = argmax(Hp)
    @test abs(ps[ip0]) < 1e-12
    @test Hp ≈ reverse(Hp) rtol = 1e-8
    @test issorted(Hp[1:ip0]) && issorted(Hp[ip0:end]; rev=true)
    @test Hp[1] < 0.2 * Hp[ip0] && Hp[end] < 0.2 * Hp[ip0]
    @test all(abs.(H[rows, :] .- mean(H[rows, :]; dims=1)) .<= 1e-6 * maximum(H))
    # batch form returns (Hs, ps, qs)
    Hs, pss, qss = husimi_function(data)
    @test length(Hs) == length(data)
    @test Hs[i0] ≈ H
    @test pss[i0] ≈ ps && qss[i0] ≈ qs

    # wavefunction proportional to besselj(0, k r) on interior points
    Psi, xg, yg = wavefunction(data, i0; fundamental_domain=false, use_chebyshev=false)
    @test size(Psi) == (length(xg), length(yg))
    k = data.k[i0]
    idx = [(ix, iy) for ix in 1:length(xg), iy in 1:length(yg) if xg[ix]^2 + yg[iy]^2 < 0.8^2]
    psi = [Psi[ix, iy] for (ix, iy) in idx]
    ref = [besselj(0, k * hypot(xg[ix], yg[iy])) for (ix, iy) in idx]
    @test length(idx) > 1000
    @test _ss_scaled_residual(psi, ref) < 1e-6
end

# solver: Beyn contour integral
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=10, eigenvectors=true (cached circle spectrum, window [2.2, 4.0], first level j_{0,1})
# billiard: Circle, radius 1
# symmetry: None
# reference: structural; Beyn BIM states store the layer density only, so `state.u` is `nothing`. The solver-free
#   `boundary_function(state)` must refuse with an ArgumentError, while the SpectralData method recomputes `u` through the solver
# functions to test: boundary_function (BIMEigenstate without stored u, SpectralData method)
@testset "States in SpectralData - Circle CFIE Beyn - missing u throws without a solver" begin
    data = _ss_circle_spectrum()
    @test data.states[1].u === nothing
    @test_throws ArgumentError boundary_function(data.states[1])
    @test_throws "no stored boundary normal derivative" boundary_function(data.states[1], data.billiard)
    u, s, _ = boundary_function(data, 1)
    @test u !== nothing
    @test length(u) == length(s)
end

# solver: Vergini-Saraceno
# kernel/basis: Real Plane Waves, even-even sector (sym_x = 1, sym_y = 1), scaling factors (5.0, 10.0), eigenvectors=true
# billiard: Rectangle (RectangleBilliard(1.0, 0.7), the full 2.0 x 1.4 rectangle centred at the origin)
# symmetry: even-even
# reference: analytic; psi = cos(m*pi*x/2) * cos(n*pi*y/1.4) for odd m, n (the even-even sin*sin modes of the box [-1,1]x[-0.7,0.7]),
#   with (m, n) identified from the computed k; window [20, 23] holds 8 levels
# functions to test: wavefunction (SpectralData methods / BasisEigenstate, fundamental_domain = false)
@testset "States in SpectralData - Rectangle VS - wavefunction proportional to sin*sin" begin
    data = _ss_rectangle_spectrum()
    @test length(data) == 8
    for i in 1:length(data)
        k = data.k[i]
        modes = [(m, n) for m in 1:2:15, n in 1:2:15 if abs(pi * sqrt((m / 2)^2 + (n / 1.4)^2) - k) < SPECTRUM_ATOL_VS]
        @test length(modes) == 1
        m, n = only(modes)
        Psi, xg, yg = wavefunction(data, i; fundamental_domain=false, multithreaded=false)
        @test size(Psi) == (length(xg), length(yg))
        idx = [(ix, iy) for ix in 1:length(xg), iy in 1:length(yg) if abs(xg[ix]) < 0.95 && abs(yg[iy]) < 0.65]
        psi = [Psi[ix, iy] for (ix, iy) in idx]
        ref = [cos(m * pi * xg[ix] / 2) * cos(n * pi * yg[iy] / 1.4) for (ix, iy) in idx]
        @test length(idx) > 1000
        @test _ss_scaled_residual(psi, ref) < 0.05
    end
end

# solver: Beyn contour integral (circle) and Vergini-Saraceno (rectangle)
# kernel/basis: CFIE (circle), Real Plane Waves (rectangle), eigenvectors=true
# billiard: Circle radius 1 (window [2.2, 4.0], 3 levels incl. the degenerate J1 pair), Rectangle (window [20, 23], 8 levels)
# symmetry: None (circle), even-even (rectangle)
# reference: structural invariants of SpectralData: equal field lengths, index i selects entry i of every field, iteration
#   visits every entry, vector indexing, k_range and filter keep fields and states aligned
# functions to test: SpectralData accessors (length, getindex, iterate, k_range, filter)
@testset "SpectralData accessors - index consistency" begin
    for data in (_ss_circle_spectrum(), _ss_rectangle_spectrum())
        n = length(data)
        @test n > 1
        @test length(data.k) == length(data.k_im) == length(data.ten) == length(data.control) == length(data.states) == n
        @test issorted(data.k)
        @test all(i -> isapprox(real(data.states[i].k), data.k[i]; atol=SPECTRUM_ATOL_VS), 1:n)
        for i in 1:n
            di = data[i]
            @test length(di) == 1
            @test di.k == [data.k[i]] && di.k_im == [data.k_im[i]] && di.ten == [data.ten[i]] && di.control == [data.control[i]]
            @test di.states[1] === data.states[i]
            @test di.billiard === data.billiard && di.solver === data.solver
        end
        @test collect(k for d in data for k in d.k) == data.k
        sub = data[[1, n]]
        @test sub.k == data.k[[1, n]] && sub.ten == data.ten[[1, n]] && sub.states[2] === data.states[n]
        kmid = (data.k[1] + data.k[2]) / 2
        part = QuantumBilliards.k_range(data, kmid, data.k[end] + 1)
        @test part.k == data.k[2:end] && length(part.states) == n - 1
        kept = filter(d -> d.k[1] > kmid, data)
        @test kept.k == data.k[2:end] && kept.states[1] === data.states[2]
    end
    # compute_spectrum without eigenvectors stores no states, and the state wrappers refuse
    billiard = CircleBilliard(1.0)
    kernel = CombinedFieldIntegralEquationSolver(10.0, billiard; grading=SmoothPeriodicGrading())
    nostates = compute_spectrum(BeynSolver(kernel; use_chebyshev=false, eigenvectors=false), billiard, 2.2, 2.6; multithreaded=false, show_progress=false)
    @test nostates.states === nothing
    @test_throws ArgumentError wavefunction(nostates, 1)
end
