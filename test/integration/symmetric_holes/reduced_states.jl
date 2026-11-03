# Integration tests for the state constructors after a symmetry-reduced Beyn solve on a billiard with a symmetric hole (Phase 2):
# `boundary_function`, `husimi_function`, `wavefunction` and `compute_eigenstate` for the sector (chi_X, chi_Y) = (+1, +1) of the
# Sinai billiard (D4 registry, SymmetricHole), see sinai_sector_spectra.jl for the setup. A smoke test: only sizes, finiteness and
# normalisation-independent properties are asserted; no tension, wavefunction or boundary value is pinned.
# Facts used: the reduced solver stores the layer density on the whole boundary (outer square then hole circle, one arclength
# parametrisation of total length 8 + 2 pi 0.25); the Husimi function lives on the outer component only (L = 8) and is normalised
# to sum 1; the unfolded wavefunction of the sector (+1, +1) is even under x -> -x and y -> -y and vanishes inside the hole and
# outside the square (inside_only mask).

using LinearAlgebra: norm, dot

const REDUCED_STATES_CACHE = Dict{Symbol,Any}()

# Reduced Beyn spectrum of the sector (+1, +1) with eigenvectors, and the states it holds.
function _rs_sinai_data()
    return get!(REDUCED_STATES_CACHE, :data) do
        sh_spectrum(:sinai, sh_sinai(), SINAI_WINDOW..., XAxisReflection => 1, YAxisReflection => 1; eigenvectors=true)
    end
end

# solver: Beyn contour integral
# kernel/basis: composite CFIE (outer square graded, disc hole), ppw=10, min_pts=96, eigenvectors=true
# billiard: Sinai (SinaiBilliard(1.0, 0.25))
# symmetry: D2 sector (chi_X, chi_Y) = (+1, +1) of D4
# reference: structural (sizes, finiteness), the arclength of the loops (analytic), Husimi normalisation
# functions to test: boundary_function, husimi_function
@testset "States after reduced Beyn - Sinai sector (+1, +1) - boundary function and Husimi function" begin
    b = sh_sinai()
    data = _rs_sinai_data()
    @test length(data) > 0
    @test length(data.states) == length(data.k)
    @test all(data.ten .< TENSION_TOL_ACCEL)
    state = data.states[1]
    solver = data.solver

    u, s, nrm = boundary_function(state, b, solver)
    @test length(u) == length(s) == length(state.pts)
    @test all(isfinite, u) && isfinite(nrm)
    @test issorted(s) && s[1] > 0
    @test s[end] < 8.0 + 2pi * 0.25          # cumulative arclength over the outer square and the hole circle
    @test norm(u) > 0

    H, qs, ps = husimi_function(state, b, solver)
    @test size(H) == (length(qs), length(ps))
    @test all(isfinite, H) && all(H .>= -1e-12)
    @test sum(H) ≈ 1 atol = 1e-8
    @test minimum(qs) >= 0 && maximum(qs) <= 8.0     # outer component only
    @test minimum(ps) >= -1 && maximum(ps) <= 1

    # the SpectralData methods give the same objects (u only up to the arbitrary complex phase of the eigenvector)
    u2, s2, nrm2 = boundary_function(data, 1)
    @test s2 ≈ s && nrm2 ≈ nrm
    @test abs(dot(u, u2)) / (norm(u) * norm(u2)) ≈ 1 atol = 1e-10
    H2, _, _ = husimi_function(data, 1)
    @test H2 ≈ H
end

# solver: Beyn contour integral
# kernel/basis: composite CFIE (outer square graded, disc hole), ppw=10, min_pts=96, eigenvectors=true
# billiard: Sinai (SinaiBilliard(1.0, 0.25))
# symmetry: D2 sector (chi_X, chi_Y) = (+1, +1) of D4
# reference: symmetry of the unfolded wavefunction (even in x and in y), mask of the geometry (zero inside the hole and outside the square)
# functions to test: wavefunction (fundamental_only = false and true)
@testset "States after reduced Beyn - Sinai sector (+1, +1) - wavefunction" begin
    b = sh_sinai()
    data = _rs_sinai_data()
    state = data.states[1]
    solver = data.solver

    @testset "unfolded (fundamental_only = false)" begin
        Psi, xg, yg = wavefunction(state, b, solver; ppw=3.0, use_chebyshev=false, fundamental_only=false)
        @test size(Psi) == (length(xg), length(yg))
        @test all(isfinite, Psi)
        scale = maximum(abs, Psi)
        @test scale > 0
        @test xg ≈ -reverse(xg) && yg ≈ -reverse(yg)
        @test maximum(abs.(Psi .- reverse(Psi; dims=1))) < 1e-8 * scale      # even in x
        @test maximum(abs.(Psi .- reverse(Psi; dims=2))) < 1e-8 * scale      # even in y
        inside_hole = [hypot(x, y) < 0.25 - 0.02 for x in xg, y in yg]
        outside_square = [max(abs(x), abs(y)) > 1.0 + 1e-9 for x in xg, y in yg]
        @test count(inside_hole) > 100
        @test all(iszero, Psi[inside_hole]) && all(iszero, Psi[outside_square])
        interior = [0.3 < hypot(x, y) && max(abs(x), abs(y)) < 0.97 for x in xg, y in yg]
        @test count(!iszero, Psi[interior]) > 0.99 * count(interior)         # the mask keeps the interior
    end

    @testset "fundamental domain (fundamental_only = true)" begin
        Psi, xg, yg = wavefunction(state, b, solver; ppw=3.0, use_chebyshev=false, fundamental_only=true)
        @test size(Psi) == (length(xg), length(yg))
        @test all(isfinite, Psi)
        @test xg[1] <= 0.05 && xg[end] >= 0.95       # grid covers the fundamental triangle, not just its physical edge x = 1
    end
end

# solver: Sweep CompositeBIM (reduced kernel of the Beyn solver above)
# kernel/basis: composite CFIE (outer square graded, disc hole), ppw=10, min_pts=96
# billiard: Sinai (SinaiBilliard(1.0, 0.25))
# symmetry: D2 sector (chi_X, chi_Y) = (+1, +1) of D4
# reference: cross-method, a level of the reduced Beyn spectrum (the sweep tension at a root is a small threshold, not a pinned value)
# functions to test: compute_eigenstate (SweepBIMSolver, reduced composite kernel)
@testset "States after reduced Beyn - Sinai sector (+1, +1) - compute_eigenstate" begin
    b = sh_sinai()
    data = _rs_sinai_data()
    k0 = real(data.k[argmin(real.(data.k))])
    single = compute_eigenstate(data.solver.kernel, b, k0)       # SpectralData with one state
    @test length(single) == 1
    @test single.k[1] ≈ k0
    @test single.ten[1] < TENSION_TOL_BIM
    @test length(single.states[1].vec) == length(single.states[1].pts)
end
