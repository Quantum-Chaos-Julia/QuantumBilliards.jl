# Integration tests for symmetric holes (Phase 2 of the symmetric-holes plan, scratchpad of BilliardGeometry.jl): a reduced
# CompositeBIMSolver on a billiard whose holes are invariant under the group G' of the billiard, compared with the unreduced
# solver on the same nodes. The Fredholm matrix of the reduced solver for the irrep chi acts on the orbit representatives of the
# unreduced matrix, so its eigenvalues are a subset of the unreduced eigenvalues, and the union over all irreps of G' is the
# whole unreduced spectrum. The comparison is made on the eigenvalues of the matrices at a fixed k (no root search), with forced
# small node counts (pts_scaling_factor 0.01, min_pts = node count) and low k = 6.3. The tolerance SECTOR_UNION_ATOL of common.jl is
# calibrated on the observed agreement (1.1e-14). Hole fixtures come from test/holes_fixtures.jl (quarter-square billiard with
# D2_symmetry(); the fixture helpers are copies of the BilliardGeometry ones).

using LinearAlgebra: eigvals
using StaticArrays: SVector
using QuantumBilliards: _composite_symmetry_orbits

# Distance of `λ` to the nearest element of `pool`.
_hs_dist(λ, pool) = minimum(abs.(pool .- λ))

# Largest distance between the eigenvalues in `reduced` and `full`, matched one to one (each element of `full` is used once).
# Returns the worst distance and the number of elements of `full` left unmatched.
function _hs_match(reduced::Vector{ComplexF64}, full::Vector{ComplexF64})
    pool = copy(full)
    worst = 0.0
    for λ in reduced
        j = argmin(abs.(pool .- λ))
        worst = max(worst, abs(pool[j] - λ))
        deleteat!(pool, j)
    end
    return worst, length(pool)
end

# Composite DLP solver (outer solver and one hole solver) with node counts outer_N / hole_N and the given reduction, its nodes and its matrix.
function _hs_dlp_matrix(b, outer_N, hole_N, k; kwargs...)
    solver = CompositeBIMSolver(DoubleLayerPotentialSolver(0.01, b; min_pts=outer_N, kwargs...),
                                DoubleLayerPotentialSolver(0.01, b; min_pts=hole_N, kwargs...))
    pts = evaluate_points(solver, b, k)
    return solver, pts, construct_matrices(solver, pts, k; multithreaded=false)
end

# solver: Composite BIM sweep (DLP outer and hole components), matrix level
# kernel/basis: DLP, SmoothPeriodicGrading, forced node counts 32 (outer) and 12 (per hole loop), k = 6.3
# billiard: quarter square [0,1]^2 of RectangleBilliard(1, 1) with D2_symmetry() and SymmetrizedHoles(DomainHole(disc of radius 0.1 about (0.4, 0.2)), D2_symmetry())
# symmetry: G' = D2, the four irreps (chi_Y, chi_X) of CompositeReflection(Y, X)
# reference: cross-method, the unreduced composite matrix (80 nodes) on the same nodes
# functions to test: evaluate_points, boundary_matrix_size, construct_matrices (reduced composite with hole copies)
@testset "Composite DLP - D2 quarter square, SymmetrizedHoles(disc, D2) - four sectors vs unreduced matrix" begin
    b = hf_symmetrized_disc_billiard()
    k = 6.3
    _, pts_full, A_full = _hs_dlp_matrix(b, 32, 12, k)
    @test size(A_full) == (80, 80)
    λ_full = ComplexF64.(eigvals(A_full))
    sym = CompositeReflection(YAxisReflection(), XAxisReflection())

    sectors = Vector{ComplexF64}[]
    for χ_Y in (1, -1), χ_X in (1, -1)
        solver, pts, A = _hs_dlp_matrix(b, 32, 12, k; symmetry=sym, sym_characters=(χ_Y, χ_X))
        @test pts.xy == pts_full.xy                                      # the reduced solver samples the whole boundary
        @test boundary_matrix_size(solver, pts) == 20
        @test size(A) == (20, 20)
        λ = ComplexF64.(eigvals(A))
        @test length(λ) == 20
        @test maximum(_hs_dist(l, λ_full) for l in λ) < SECTOR_UNION_ATOL      # every sector level is an unreduced level
        push!(sectors, λ)
    end

    union_λ = reduce(vcat, sectors)
    @test length(union_λ) == 80
    worst, left = _hs_match(union_λ, λ_full)
    @test worst < SECTOR_UNION_ATOL                                      # the four sectors together are the unreduced spectrum
    @test left == 0
end

# solver: Composite BIM sweep (DLP outer and hole components), matrix level
# kernel/basis: DLP, SmoothPeriodicGrading, forced node counts 32 (outer) and 12 (hole), k = 6.3
# billiard: same quarter square with SymmetricHole(quarter disc of radius 0.1 about (0.5, 0), D2_symmetry(), center (0.5, 0)), G' = {X}
# symmetry: XAxisReflection with chi_X = +1 and -1 (Y is not in G')
# reference: cross-method, the unreduced composite matrix (44 nodes) on the same nodes; the G' error for Y
# functions to test: evaluate_points, boundary_matrix_size, construct_matrices
@testset "Composite DLP - D2 quarter square, SymmetricHole (G' = {X}) - two sectors vs unreduced matrix" begin
    b = hf_quarter_disc_billiard()
    k = 6.3
    _, pts_full, A_full = _hs_dlp_matrix(b, 32, 12, k)
    @test size(A_full) == (44, 44)
    λ_full = ComplexF64.(eigvals(A_full))

    sectors = Vector{ComplexF64}[]
    for χ_X in (1, -1)
        solver, pts, A = _hs_dlp_matrix(b, 32, 12, k; symmetry=XAxisReflection(), sym_characters=(χ_X,))
        @test pts.xy == pts_full.xy
        @test boundary_matrix_size(solver, pts) == 22
        @test size(A) == (22, 22)
        λ = ComplexF64.(eigvals(A))
        @test maximum(_hs_dist(l, λ_full) for l in λ) < SECTOR_UNION_ATOL
        push!(sectors, λ)
    end
    worst, left = _hs_match(reduce(vcat, sectors), λ_full)
    @test worst < SECTOR_UNION_ATOL
    @test left == 0

    @testset "Y is not in G'" begin
        bad = CompositeBIMSolver(DoubleLayerPotentialSolver(0.01, b; min_pts=32, symmetry=YAxisReflection(), sym_characters=(1,)),
                                 DoubleLayerPotentialSolver(0.01, b; min_pts=12, symmetry=YAxisReflection(), sym_characters=(1,)))
        pts = evaluate_points(bad, b, k)
        @test_throws ArgumentError boundary_matrix_size(bad, pts)
        @test_throws "G'" boundary_matrix_size(bad, pts)
    end
end

# solver: Composite BIM sweep (CFIE outer and hole components), solve_state
# kernel/basis: CFIE, SmoothPeriodicGrading, forced node counts 32 (outer) and 12 (per hole loop), k = 6.3
# billiard: quarter square with SymmetrizedHoles(disc, D2), G' = D2
# symmetry: CompositeReflection(Y, X), (chi_Y, chi_X) = (1, -1)
# reference: the defining orbit identity vec[g(q)] = chi_g vec[q] of the expansion to the full boundary (BilliardGeometry orbit map)
# functions to test: solve_state (symmetrize_layer_density, _cfie_normal_derivative for a reduced composite solver), _composite_symmetry_orbits
@testset "Composite CFIE - D2 quarter square, SymmetrizedHoles(disc, D2) - solve_state expands the reduced density" begin
    b = hf_symmetrized_disc_billiard()
    k = 6.3
    sym = CompositeReflection(YAxisReflection(), XAxisReflection())
    solver = CompositeBIMSolver(CombinedFieldIntegralEquationSolver(0.01, b; min_pts=32, symmetry=sym, sym_characters=(1, -1)),
                                CombinedFieldIntegralEquationSolver(0.01, b; min_pts=12, symmetry=sym, sym_characters=(1, -1)))
    pts = evaluate_points(solver, b, k)
    ten, vec, u, bnd_norm = QuantumBilliards.solve_state(solver, pts, k, b; multithreaded=false)
    @test length(vec) == 80
    @test length(u) == 80
    @test isfinite(ten) && isfinite(bnd_norm)

    orbits = _composite_symmetry_orbits(Float64, solver, pts)
    @test orbit_size(orbits) == 4 && fundamental_size(orbits) == 20
    # every orbit image carries its character times the value at the representative
    ok = all(1:fundamental_size(orbits)) do a
        rep = vec[orbits.fundamental_indices[a]]
        all(g -> isapprox(vec[orbits.fund_to_full[g, a]], orbits.fund_to_scale[g, a] * rep; atol=1e-12, rtol=1e-10), 1:orbit_size(orbits))
    end
    @test ok
end
