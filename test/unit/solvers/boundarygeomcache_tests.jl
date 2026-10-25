# Unit tests for
#   src/solvers/boundarygeomcache.jl     (BoundaryPanelArrays, _boundary_panel_arrays_cache, component_normals,
#                                         BoundaryGeomCache, boundary_geom_cache)
#   src/solvers/chebyshev/optimalpanelization.jl   (_cheb_geom_rminmax: radial interval taken from the geom cache)
# No eigen solves: the cache is checked against closed forms for a circle parametrized by its angle and against
# an independent broadcast evaluation on an ellipse.

using LinearAlgebra: norm
using StaticArrays: SVector
using QuantumBilliards: _boundary_panel_arrays_cache, _cheb_geom_rminmax, hankel_z_chebyshev_cutoff

# Circle of radius R, parametrized by the angle at midpoint nodes (all derivatives analytic).
function bgc_circle_points(::Type{T}, N::Int, R) where {T<:Real}
    h = T(2π)/N
    θ = T[(j - T(1)/2)*h for j in 1:N]
    Rt = T(R)
    xy = [SVector{2,T}(Rt*cos(t), Rt*sin(t)) for t in θ]
    tangent = [SVector{2,T}(-Rt*sin(t), Rt*cos(t)) for t in θ]
    tangent_2 = [SVector{2,T}(-Rt*cos(t), -Rt*sin(t)) for t in θ]
    z = SVector{2,T}(zero(T), zero(T))
    bp = BoundaryPoints(xy, tangent, tangent_2, θ, copy(θ), fill(h, N), ones(T, N),
                        Rt .* θ, fill(Rt*h, N), 1, true, z, z, z, z)
    return bp, θ
end

# Ellipse (a cos t, b sin t) at non-uniform parameter values, to test against broadcast formulas.
function bgc_ellipse_points(N::Int, a, b)
    θ = [2π*(j - 0.5)/N + 0.1*sin(2π*(j - 0.5)/N) for j in 1:N]
    xy = [SVector(a*cos(t), b*sin(t)) for t in θ]
    tangent = [SVector(-a*sin(t), b*cos(t)) for t in θ]
    tangent_2 = [SVector(-a*cos(t), -b*sin(t)) for t in θ]
    return BoundaryPoints(xy; tangent=tangent, tangent_2=tangent_2, ts=θ), θ
end

@testset "_boundary_panel_arrays_cache" begin
    R = 1.5
    bp, _ = bgc_circle_points(Float64, 24, R)
    panels = _boundary_panel_arrays_cache(bp)

    @test panels isa BoundaryPanelArrays{Float64}
    @test panels.X == getindex.(bp.xy, 1)
    @test panels.Y == getindex.(bp.xy, 2)
    @test panels.dX == getindex.(bp.tangent, 1)
    @test panels.dY == getindex.(bp.tangent, 2)
    @test panels.speed ≈ hypot.(panels.dX, panels.dY)
    @test all(isapprox.(panels.speed, R; atol=1e-12))        # |γ'(t)| = R for the angle parametrization
    @test @inferred(_boundary_panel_arrays_cache(bp)) isa BoundaryPanelArrays{Float64}

    bp32, _ = bgc_circle_points(Float32, 8, 1)
    @test _boundary_panel_arrays_cache(bp32) isa BoundaryPanelArrays{Float32}
    @test eltype(_boundary_panel_arrays_cache(bp32).speed) == Float32
end

@testset "component_normals" begin
    @testset "normals and speed of a parametrized circle" begin
        R = 2.0
        bp, _ = bgc_circle_points(Float64, 20, R)
        nx, ny, speed = component_normals(bp)
        @test nx ≈ getindex.(bp.xy, 1) ./ R
        @test ny ≈ getindex.(bp.xy, 2) ./ R
        @test all(isapprox.(hypot.(nx, ny), 1.0; atol=1e-12))
        @test all(isapprox.(speed, R; atol=1e-12))
    end

    @testset "speed defaults to one without tangent data" begin
        xy = [SVector(0.0, 0.0), SVector(1.0, 0.0)]
        bp = BoundaryPoints(xy; normal=[SVector(0.0, -1.0), SVector(0.0, -1.0)])
        nx, ny, speed = component_normals(bp)
        @test nx == [0.0, 0.0]
        @test ny == [-1.0, -1.0]
        @test speed == [1.0, 1.0]
        @test eltype(speed) == Float64
        @test @inferred(component_normals(bp)) isa Tuple{Vector{Float64},Vector{Float64},Vector{Float64}}
    end

    @testset "missing normals" begin
        bp = BoundaryPoints([SVector(0.0, 0.0), SVector(1.0, 0.0)])
        @test_throws ArgumentError component_normals(bp)
    end
end

@testset "boundary_geom_cache on a circle" begin
    R = 1.5
    N = 32
    bp, θ = bgc_circle_points(Float64, N, R)
    G = boundary_geom_cache(bp)

    @test G isa BoundaryGeomCache{Float64}
    @test size(G.R) == size(G.invR) == size(G.inner) == size(G.logterm) == (N, N)
    @test length(G.speed) == length(G.kappa) == N

    chord = [2R*abs(sin((θ[i] - θ[j])/2)) for i in 1:N, j in 1:N]
    offdiag = [i != j for i in 1:N, j in 1:N]

    @testset "pairwise distances" begin
        @test all(G.R[i, i] == 1.0 for i in 1:N)              # diagonal convention: one
        @test G.R[offdiag] ≈ chord[offdiag]                    # chord length 2R|sin(Δθ/2)|
        @test G.R == G.R'
    end

    @testset "inverse distances" begin
        @test all(G.invR[i, i] == 0.0 for i in 1:N)            # diagonal convention: zero
        @test G.invR[offdiag] ≈ 1 ./ chord[offdiag]
        @test G.invR == G.invR'
    end

    @testset "tangential interaction term" begin
        # for γ = R(cos t, sin t): γ'(t_j) × (γ(t_i) - γ(t_j)) = R^2 (cos(t_i - t_j) - 1), symmetric in (i, j)
        expected = [R^2*(cos(θ[i] - θ[j]) - 1) for i in 1:N, j in 1:N]
        @test all(G.inner[i, i] == 0.0 for i in 1:N)
        @test G.inner[offdiag] ≈ expected[offdiag]
    end

    @testset "periodic logarithmic kernel" begin
        expected = [log(4*sin((θ[i] - θ[j])/2)^2) for i in 1:N, j in 1:N]
        @test all(G.logterm[i, i] == 0.0 for i in 1:N)
        @test G.logterm[offdiag] ≈ expected[offdiag]
        @test G.logterm == G.logterm'
    end

    @testset "speed and scaled curvature" begin
        @test all(isapprox.(G.speed, R; atol=1e-12))
        # kappa = -(γ' × γ'')/(2π |γ'|^2) = -1/(2π): the 1/R curvature is cancelled by the speed R
        @test all(isapprox.(G.kappa, -1/(2π); atol=1e-12))
    end

    @testset "original_ts only on request" begin
        @test isempty(G.original_ts)
        @test isempty(boundary_geom_cache(bp, false).original_ts)
        Gk = boundary_geom_cache(bp, true)
        @test Gk.original_ts == bp.ts
        @test Gk.original_ts !== bp.ts                          # a copy, not an alias
        @test Gk.R == G.R && Gk.logterm == G.logterm
    end

    @test @inferred(boundary_geom_cache(bp)) isa BoundaryGeomCache{Float64}
end

@testset "boundary_geom_cache on an ellipse (independent broadcast formulas)" begin
    N = 20
    bp, θ = bgc_ellipse_points(N, 2.0, 1.0)
    G = boundary_geom_cache(bp)

    X = getindex.(bp.xy, 1); Y = getindex.(bp.xy, 2)
    dX = getindex.(bp.tangent, 1); dY = getindex.(bp.tangent, 2)
    ddX = getindex.(bp.tangent_2, 1); ddY = getindex.(bp.tangent_2, 2)
    ΔX = X .- X'
    ΔY = Y .- Y'
    R_ref = hypot.(ΔX, ΔY)
    inner_ref = permutedims(dY) .* ΔX .- permutedims(dX) .* ΔY   # (dY_j ΔX_ij - dX_j ΔY_ij)
    logterm_ref = log.(4 .* sin.((θ .- θ') ./ 2) .^ 2)
    kappa_ref = -(dX .* ddY .- dY .* ddX) ./ (2π .* (dX .^ 2 .+ dY .^ 2))
    mask = [i != j for i in 1:N, j in 1:N]

    @test G.R[mask] ≈ R_ref[mask]
    @test G.invR[mask] ≈ (1 ./ R_ref)[mask]
    @test G.inner[mask] ≈ inner_ref[mask]
    @test G.logterm[mask] ≈ logterm_ref[mask]
    @test G.speed ≈ hypot.(dX, dY)
    @test G.kappa ≈ kappa_ref
    # an ellipse is a linear image of the circle, so the cross product scales by det = a*b:
    # inner_ij = a*b*(cos(t_i - t_j) - 1), which is symmetric
    @test G.inner[mask] ≈ [2.0*1.0*(cos(θ[i] - θ[j]) - 1) for i in 1:N, j in 1:N][mask]
    @test G.inner ≈ G.inner'
end

@testset "boundary_geom_cache inner term is not symmetric for a non-affine curve" begin
    # inner_ij = γ'(t_j) × (γ(t_i) - γ(t_j)); for a non-affine curve the two orderings differ.
    N = 16
    θ = [2π*(j - 0.5)/N for j in 1:N]
    xy = [SVector((1 + 0.3*cos(3t))*cos(t), (1 + 0.3*cos(3t))*sin(t)) for t in θ]
    dr(t) = -0.9*sin(3t)
    tangent = [SVector(dr(t)*cos(t) - (1 + 0.3*cos(3t))*sin(t), dr(t)*sin(t) + (1 + 0.3*cos(3t))*cos(t)) for t in θ]
    d2r(t) = -2.7*cos(3t)
    r(t) = 1 + 0.3*cos(3t)
    tangent_2 = [SVector(d2r(t)*cos(t) - 2*dr(t)*sin(t) - r(t)*cos(t),
                         d2r(t)*sin(t) + 2*dr(t)*cos(t) - r(t)*sin(t)) for t in θ]
    G = boundary_geom_cache(BoundaryPoints(xy; tangent=tangent, tangent_2=tangent_2, ts=θ))
    @test !(G.inner ≈ G.inner')
end

@testset "boundary_geom_cache Float32" begin
    bp, _ = bgc_circle_points(Float32, 16, 1)
    G = boundary_geom_cache(bp, true)
    @test G isa BoundaryGeomCache{Float32}
    @test eltype(G.R) == Float32 && eltype(G.invR) == Float32
    @test eltype(G.inner) == Float32 && eltype(G.logterm) == Float32
    @test eltype(G.speed) == Float32 && eltype(G.kappa) == Float32 && eltype(G.original_ts) == Float32
    @test all(isapprox.(G.kappa, -1f0/(2f0*Float32(π)); atol=1f-5))
end

@testset "boundary_geom_cache argument errors" begin
    xy = [SVector(1.0, 0.0), SVector(0.0, 1.0), SVector(-1.0, 0.0)]
    t1 = fill(SVector(0.0, 1.0), 3)
    t2 = fill(SVector(-1.0, 0.0), 3)
    ts = [0.0, 1.0, 2.0]

    @test_throws ArgumentError boundary_geom_cache(BoundaryPoints(xy; tangent_2=t2, ts=ts))              # no tangent
    @test_throws ArgumentError boundary_geom_cache(BoundaryPoints(xy; tangent=t1, ts=ts))                # no tangent_2
    @test_throws ArgumentError boundary_geom_cache(BoundaryPoints(xy; tangent=t1, tangent_2=t2))         # no ts
    @test boundary_geom_cache(BoundaryPoints(xy; tangent=t1, tangent_2=t2, ts=ts)) isa BoundaryGeomCache{Float64}
end

@testset "_cheb_geom_rminmax (radial interval from the geometry cache)" begin
    # unit square corners: nearest pair distance 1, farthest pair distance sqrt(2)
    xy = SVector{2,Float64}[(0, 0), (1, 0), (1, 1), (0, 1)]
    pts = BoundaryPoints(xy; tangent=fill(SVector(1.0, 0.0), 4), tangent_2=fill(SVector(0.0, 0.0), 4),
                         ts=collect(0:3) .* (π/2))
    G = boundary_geom_cache(pts)

    @testset "padding of the extreme distances" begin
        rmin, rmax = _cheb_geom_rminmax(G, ComplexF64[2]; pad=(1.0, 1.0))
        @test rmin ≈ 1.0
        @test rmax ≈ sqrt(2)
        rmin, rmax = _cheb_geom_rminmax(G, ComplexF64[2])           # default pad (0.95, 1.05)
        @test rmin ≈ 0.95
        @test rmax ≈ 1.05*sqrt(2)
        rmin, rmax = _cheb_geom_rminmax(G, ComplexF64[2]; pad=(0.5, 2.0))
        @test rmin ≈ 0.5
        @test rmax ≈ 2*sqrt(2)
    end

    @testset "lower bound from the Hankel cutoff" begin
        c = hankel_z_chebyshev_cutoff
        # cutoff radius c/|k| below the padded minimum distance (1): no effect
        rmin, _ = _cheb_geom_rminmax(G, ComplexF64[10c]; pad=(1.0, 1.0))
        @test rmin ≈ 1.0
        # cutoff radius 1.2 lies between rmin = 1 and rmax = sqrt(2): it replaces the padded minimum
        rmin, rmax = _cheb_geom_rminmax(G, ComplexF64[c/1.2]; pad=(1.0, 1.0))
        @test rmin ≈ 1.2
        @test rmax ≈ sqrt(2)
        # cutoff radius 2 exceeds rmax: empty interval
        @test_throws ArgumentError _cheb_geom_rminmax(G, ComplexF64[c/2]; pad=(1.0, 1.0))
    end

    @testset "largest |k| in the window sets the bound" begin
        c = hankel_z_chebyshev_cutoff
        rmin, _ = _cheb_geom_rminmax(G, ComplexF64[c/1.3, c/1.2]; pad=(1.0, 1.0))
        @test rmin ≈ 1.2                                   # c / max|k|
        rmin_swapped, _ = _cheb_geom_rminmax(G, ComplexF64[c/1.2, c/1.3]; pad=(1.0, 1.0))
        @test rmin_swapped ≈ rmin
    end

    @testset "complex wavenumbers use the modulus" begin
        c = hankel_z_chebyshev_cutoff
        k = (c/1.2) * (0.6 + 0.8im)                         # |k| = c/1.2
        rmin, rmax = _cheb_geom_rminmax(G, ComplexF64[k]; pad=(1.0, 1.0))
        @test rmin ≈ 1.2 && rmax ≈ sqrt(2)
    end

    @testset "errors" begin
        @test_throws ArgumentError _cheb_geom_rminmax(G, ComplexF64[2]; pad=(2.0, 1.0))   # empty interval
        single = boundary_geom_cache(BoundaryPoints([SVector(0.0, 0.0)]; tangent=[SVector(1.0, 0.0)],
                                                    tangent_2=[SVector(0.0, 0.0)], ts=[0.0]))
        @test_throws ArgumentError _cheb_geom_rminmax(single, ComplexF64[2])             # no off-diagonal distance
    end
end
