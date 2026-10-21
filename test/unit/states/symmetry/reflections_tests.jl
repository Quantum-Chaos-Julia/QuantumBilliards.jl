# Unit tests for
#   src/states/symmetry/reflections.jl   (_reflection_case, _reflection_index, apply_symmetries_to_wavefunction,
#                                         apply_symmetries_to_boundary_function, apply_symmetries_to_boundary_points)
# All expected values are analytic. The "known parity function" is f(x, y) = x^i y^j, whose reflections are exact:
#   f(-x, y) = (-1)^i f(x, y)   (YAxisReflection parity pY = (-1)^i),
#   f(x, -y) = (-1)^j f(x, y)   (XAxisReflection parity pX = (-1)^j).
# For the boundary functions the fundamental boundary is a quarter circle sampled at midpoint angles, whose unfolding
# must reproduce the full circle in counter-clockwise order, with u = cos(m*theta) or sin(m*theta) as parity functions.
# One testset ties the three unfolding functions together on the boundary of a RectangleBilliard (D2).
# No eigen solves. Helper functions use the prefix `rf_`.

using LinearAlgebra: norm, dot
using StaticArrays: SVector
using QuantumBilliards: _reflection_case, _reflection_index, boundary_coords

const rf_Y = YAxisReflection(1)
const rf_XY = XYAxisReflection(2)
const rf_X = XAxisReflection(3)

# Midpoint angles of a quarter circle (N points) and of the full circle (4N points) with the same spacing.
rf_quarter_angles(::Type{T}, N) where {T<:Real} = T[(T(pi) / 2) * (j - T(1) / 2) / N for j in 1:N]

# Quarter-circle BoundaryPoints with outward normals, constant ds and cumulative s.
function rf_quarter_circle_points(::Type{T}, N) where {T<:Real}
    θ = rf_quarter_angles(T, N)
    xy = [SVector{2,T}(cos(t), sin(t)) for t in θ]
    ds = fill(T(pi) / 2 / N, N)
    return BoundaryPoints(xy; normal=copy(xy), s=cumsum(ds), ds=ds)
end

# Cyclic consecutive distances of a closed polygon.
rf_chords(xy) = [norm(xy[mod1(i + 1, length(xy))] - xy[i]) for i in eachindex(xy)]

@testset "_reflection_case" begin
    @test _reflection_case([rf_Y, rf_X]) == (true, true)
    @test _reflection_case([rf_X, rf_Y]) == (true, true)
    @test _reflection_case([rf_Y, rf_XY, rf_X]) == (true, true)
    @test _reflection_case([rf_X]) == (true, false)
    @test _reflection_case([rf_Y]) == (false, true)
    @test _reflection_case([rf_XY]) == (false, false)                          # XY alone is neither an X nor a Y generator
    @test _reflection_case([DiagonalReflection(1)]) == (false, false)
    @test _reflection_case(BilliardGeometry.AbsReflection[]) == (false, false)
    @test _reflection_case(RectangleBilliard(1.0, 0.6).symmetries) == (true, true)   # SymmetryRegistry is an AbstractVector
    @test _reflection_case(TriangleBilliard(pi / 2, pi / 3).symmetries) == (false, false)
end

@testset "_reflection_index" begin
    syms = [rf_Y, rf_XY, rf_X]
    @test _reflection_index(YAxisReflection, syms) == 1
    @test _reflection_index(XYAxisReflection, syms) == 2
    @test _reflection_index(XAxisReflection, syms) == 3
    @test _reflection_index(XAxisReflection, [rf_X, rf_Y]) == 1
    @test _reflection_index(XAxisReflection, [rf_Y]) === nothing
    @test _reflection_index(DiagonalReflection, syms) === nothing
    @test _reflection_index(BilliardGeometry.AbsReflection, syms) == 1         # abstract type matches the first element
    @test _reflection_index(XAxisReflection, RectangleBilliard(1.0, 0.6).symmetries) == 3
end

@testset "apply_symmetries_to_wavefunction - both axes, known parity function" begin
    xg = [0.1, 0.25, 0.4]
    yg = [0.2, 0.5]
    full_x = vcat(-reverse(xg), xg)
    full_y = vcat(-reverse(yg), yg)

    @testset "f = x^i y^j reproduces on the full grid for every parity pair" begin
        for i in 1:2, j in 1:2
            f(x, y) = x^i * y^j
            Psi = [f(x, y) for x in xg, y in yg]
            pY = (-1.0)^i
            pX = (-1.0)^j
            full_Psi, px, py = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_Y, rf_X], [pY, pX])
            @test px == full_x
            @test py == full_y
            @test size(full_Psi) == (2 * length(xg), 2 * length(yg))
            @test full_Psi == [f(x, y) for x in full_x, y in full_y]
        end
    end

    @testset "quadrant layout: Q1 = base, rows are x, columns are y" begin
        f(x, y) = x^3 * y^2                                                    # pY = -1, pX = +1
        Psi = [f(x, y) for x in xg, y in yg]
        full_Psi, _, _ = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_Y, rf_X], [-1.0, 1.0])
        nx, ny = length(xg), length(yg)
        @test full_Psi[nx+1:end, ny+1:end] == Psi                              # x > 0, y > 0
        @test full_Psi[1:nx, ny+1:end] == -reverse(Psi, dims=1)                # x < 0, y > 0 (Y-reflected, parity -1)
        @test full_Psi[nx+1:end, 1:ny] == reverse(Psi, dims=2)                 # x > 0, y < 0 (X-reflected, parity +1)
        @test full_Psi[1:nx, 1:ny] == -reverse(Psi, dims=(1, 2))               # x < 0, y < 0 (XY-reflected, parity pX*pY = -1)
    end

    @testset "parities are looked up by generator type, not by position" begin
        f(x, y) = x^1 * y^2                                                    # pY = -1, pX = +1
        Psi = [f(x, y) for x in xg, y in yg]
        a, _, _ = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_Y, rf_X], [-1.0, 1.0])
        b, _, _ = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_X, rf_Y], [1.0, -1.0])
        @test a == b
        @test a == [f(x, y) for x in full_x, y in full_y]
    end

    @testset "an XYAxisReflection entry in the registry does not change the result" begin
        f(x, y) = x^2 * y^1                                                    # pY = +1, pX = -1
        Psi = [f(x, y) for x in xg, y in yg]
        a, _, _ = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_Y, rf_X], [1.0, -1.0])
        b, _, _ = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_Y, rf_XY, rf_X], [1.0, 1.0, -1.0])
        @test a == b
    end

    @testset "the input is not modified" begin
        Psi = [x^3 * y^2 for x in xg, y in yg]
        Psi0 = copy(Psi)
        apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_Y, rf_X], [-1.0, 1.0])
        @test Psi == Psi0
    end

    @testset "element and grid types are preserved (parities of the same precision)" begin
        # Float64 parities would promote a Float32 Psi to Float64 (p .* Psi); the basis stores sym_qnumbers as Float64.
        xg32 = Float32[0.1, 0.25, 0.4]
        yg32 = Float32[0.2, 0.5]
        Psi32 = Float32[x^3 * y^2 for x in xg32, y in yg32]
        P, px, py = apply_symmetries_to_wavefunction(Psi32, xg32, yg32, [rf_Y, rf_X], Float32[-1, 1])
        @test eltype(P) == Float32
        @test eltype(px) == Float32 && eltype(py) == Float32

        PsiC = complex.(Psi32, 2 .* Psi32)
        P, _, _ = apply_symmetries_to_wavefunction(PsiC, xg32, yg32, [rf_Y, rf_X], Float32[-1, 1])
        @test eltype(P) == ComplexF32
        @test P[end, end] == PsiC[end, end]
    end

    @testset "non-uniform grid and a single column" begin
        xg2 = [0.05, 0.3, 0.31, 0.9]
        yg2 = [0.4]
        f(x, y) = x^2 * y^3                                                    # pY = +1, pX = -1
        Psi = [f(x, y) for x in xg2, y in yg2]
        P, px, py = apply_symmetries_to_wavefunction(Psi, xg2, yg2, [rf_Y, rf_X], [1.0, -1.0])
        @test px == vcat(-reverse(xg2), xg2)
        @test py == [-0.4, 0.4]
        @test P == [f(x, y) for x in px, y in py]
    end
end

@testset "apply_symmetries_to_wavefunction - no supported reflection" begin
    xg = [0.1, 0.25, 0.4]
    yg = [0.2, 0.5]
    Psi = [x * y for x in xg, y in yg]
    P, px, py = apply_symmetries_to_wavefunction(Psi, xg, yg, BilliardGeometry.AbsReflection[], Float64[])
    @test P === Psi && px === xg && py === yg
    P, px, py = apply_symmetries_to_wavefunction(Psi, xg, yg, [DiagonalReflection(1)], [1.0])
    @test P === Psi && px === xg && py === yg
    P, px, py = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_XY], [-1.0])  # XY alone is not an axis generator
    @test P === Psi && px === xg && py === yg
end

# Single-axis unfolding. Documented behaviour: "the grid is unfolded across that axis". The returned grids are
# correct (YAxisReflection doubles x, XAxisReflection doubles y), but the returned Psi is joined along the wrong
# dimension: Y-only gives size (nx, 2ny) and X-only gives (2nx, ny), so size(full_Psi) != (length(full_x), length(full_y)).
# SOURCE FINDING in apply_symmetries_to_wavefunction: the `has_y` branch uses `hcat` (joins dim 2) after reversing
# dims=1, and the `has_x` branch uses `vcat` (joins dim 1) after reversing dims=2; the join dims must be swapped,
# i.e. Y-only: vcat(reverse(p .* Psi, dims=1), Psi); X-only: hcat(reverse(p .* Psi, dims=2), Psi).
@testset "apply_symmetries_to_wavefunction - single axis" begin
    xg = [0.1, 0.25, 0.4]
    yg = [0.2, 0.5]

    @testset "YAxisReflection: grids" begin
        Psi = [x^3 * y^2 for x in xg, y in yg]
        _, px, py = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_Y], [-1.0])
        @test px == vcat(-reverse(xg), xg)
        @test py == yg
    end

    @testset "XAxisReflection: grids" begin
        Psi = [x^2 * y^3 for x in xg, y in yg]
        _, px, py = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_X], [-1.0])
        @test px == xg
        @test py == vcat(-reverse(yg), yg)
    end

    @testset "YAxisReflection: values on the unfolded grid (known failure)" begin
        f(x, y) = x^3 * y^2                                                    # pY = -1
        Psi = [f(x, y) for x in xg, y in yg]
        P, px, py = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_Y], [-1.0])
        @test_broken size(P) == (length(px), length(py))
        @test_broken P == [f(x, y) for x in px, y in py]
    end

    @testset "XAxisReflection: values on the unfolded grid (known failure)" begin
        f(x, y) = x^2 * y^3                                                    # pX = -1
        Psi = [f(x, y) for x in xg, y in yg]
        P, px, py = apply_symmetries_to_wavefunction(Psi, xg, yg, [rf_X], [-1.0])
        @test_broken size(P) == (length(px), length(py))
        @test_broken P == [f(x, y) for x in px, y in py]
    end
end

@testset "apply_symmetries_to_boundary_function - quarter circle, known parity functions" begin
    N = 7
    θ = rf_quarter_angles(Float64, N)
    θfull = [(pi / 2) * (j - 0.5) / N for j in 1:4N]                           # midpoint angles of the full circle

    @testset "u = cos(m*theta) and sin(m*theta), both axes" begin
        # Y: theta -> pi - theta, X: theta -> -theta.
        #   cos(theta):  pY = -1, pX = +1      sin(theta):  pY = +1, pX = -1
        #   cos(2theta): pY = +1, pX = +1      sin(2theta): pY = -1, pX = -1
        cases = (
            (cos, 1, -1.0, 1.0), (sin, 1, 1.0, -1.0),
            (cos, 2, 1.0, 1.0), (sin, 2, -1.0, -1.0),
        )
        for (fun, m, pY, pX) in cases
            u = fun.(m .* θ)
            full = apply_symmetries_to_boundary_function(u, [rf_Y, rf_XY, rf_X], [pY, 1.0, pX])
            @test length(full) == 4N
            @test full ≈ fun.(m .* θfull) atol = 1e-13
        end
    end

    @testset "block structure: base, Y, XY, X" begin
        u = cos.(θ)                                                            # pY = -1, pX = +1
        full = apply_symmetries_to_boundary_function(u, [rf_Y, rf_XY, rf_X], [-1.0, 1.0, 1.0])
        @test full[1:N] == u
        @test full[N+1:2N] == -reverse(u)                                      # Y-reflected, reversed
        @test full[2N+1:3N] == -u                                              # XY-reflected, orientation preserving, parity pX*pY
        @test full[3N+1:4N] == reverse(u)                                      # X-reflected, reversed
    end

    @testset "the XYAxisReflection entry is not needed" begin
        # `symmetries` of a basis may list only the generators (Y, X); the XY copy is derived from pX*pY.
        u = sin.(2 .* θ)
        a = apply_symmetries_to_boundary_function(u, [rf_Y, rf_X], [-1.0, -1.0])
        b = apply_symmetries_to_boundary_function(u, [rf_Y, rf_XY, rf_X], [-1.0, 1.0, -1.0])
        @test a == b
        @test a ≈ sin.(2 .* θfull) atol = 1e-13
    end

    @testset "parities are looked up by generator type" begin
        u = cos.(θ)
        a = apply_symmetries_to_boundary_function(u, [rf_Y, rf_X], [-1.0, 1.0])
        b = apply_symmetries_to_boundary_function(u, [rf_X, rf_Y], [1.0, -1.0])
        @test a == b
    end

    @testset "single axis" begin
        u = cos.(θ)
        full = apply_symmetries_to_boundary_function(u, [rf_Y], [-1.0])
        @test length(full) == 2N
        @test full[1:N] == u
        @test full[N+1:end] == -reverse(u)                                     # Y-reflection reverses orientation
        full = apply_symmetries_to_boundary_function(u, [rf_X], [1.0])
        @test length(full) == 2N
        @test full[1:N] == u
        @test full[N+1:end] == reverse(u)
    end

    @testset "no reflections: input returned unchanged" begin
        u = cos.(θ)
        @test apply_symmetries_to_boundary_function(u, BilliardGeometry.AbsReflection[], Float64[]) === u
    end

    @testset "the input is not modified" begin
        u = cos.(θ)
        u0 = copy(u)
        apply_symmetries_to_boundary_function(u, [rf_Y, rf_XY, rf_X], [-1.0, 1.0, 1.0])
        @test u == u0
    end

    @testset "element types" begin
        u32 = Float32.(cos.(θ))
        full32 = apply_symmetries_to_boundary_function(u32, [rf_Y, rf_X], [-1.0, 1.0])
        @test eltype(full32) == Float32
        uc = ComplexF64.(cos.(θ)) .* im
        fullc = apply_symmetries_to_boundary_function(uc, [rf_Y, rf_X], [-1, 1])   # integer parities
        @test eltype(fullc) == ComplexF64
        @test fullc[N+1:2N] == -reverse(uc)
    end
end

@testset "apply_symmetries_to_boundary_points - quarter circle" begin
    N = 5
    pts = rf_quarter_circle_points(Float64, N)
    billiard = RectangleBilliard(1.0, 0.6)                                     # only forwarded, not used by the function
    θfull = [(pi / 2) * (j - 0.5) / N for j in 1:4N]

    @testset "full D2 unfolding gives the counter-clockwise full circle" begin
        full = apply_symmetries_to_boundary_points(pts, BilliardGeometry.AbsReflection[rf_Y, rf_XY, rf_X], billiard)
        @test full isa BoundaryPoints{Float64}
        @test length(full.xy) == 4N
        @test maximum(norm.(full.xy .- [SVector(cos(t), sin(t)) for t in θfull])) < 1e-13
        @test maximum(norm.(full.normal .- full.xy)) < 1e-13                   # outward unit normals of the unit circle
        @test full.ds == fill(pi / 2 / N, 4N)
        @test full.s ≈ cumsum(full.ds)
        @test full.s[end] ≈ 2pi
        # Continuity: all consecutive chords (including the wrap-around) are equal for equispaced midpoint angles.
        c = rf_chords(full.xy)
        @test maximum(c) - minimum(c) < 1e-13
        @test all(c .≈ 2 * sin(pi / (4N)))
    end

    @testset "block structure: base, Y (reversed), XY, X (reversed)" begin
        full = apply_symmetries_to_boundary_points(pts, BilliardGeometry.AbsReflection[rf_Y, rf_XY, rf_X], billiard)
        @test full.xy[1:N] == pts.xy
        @test full.xy[N+1:2N] ≈ [SVector(-p[1], p[2]) for p in reverse(pts.xy)]
        @test full.xy[2N+1:3N] ≈ [SVector(-p[1], -p[2]) for p in pts.xy]
        @test full.xy[3N+1:4N] ≈ [SVector(p[1], -p[2]) for p in reverse(pts.xy)]
        @test full.normal[N+1:2N] ≈ [SVector(-p[1], p[2]) for p in reverse(pts.normal)]
    end

    @testset "single axis" begin
        full = apply_symmetries_to_boundary_points(pts, BilliardGeometry.AbsReflection[rf_Y], billiard)
        @test length(full.xy) == 2N && length(full.normal) == 2N && length(full.ds) == 2N && length(full.s) == 2N
        @test full.xy[1:N] == pts.xy
        @test full.xy[N+1:end] ≈ [SVector(-p[1], p[2]) for p in reverse(pts.xy)]
        @test full.s ≈ cumsum(full.ds)

        full = apply_symmetries_to_boundary_points(pts, BilliardGeometry.AbsReflection[rf_X], billiard)
        @test length(full.xy) == 2N
        @test full.xy[N+1:end] ≈ [SVector(p[1], -p[2]) for p in reverse(pts.xy)]
    end

    @testset "no reflections: input returned unchanged" begin
        @test apply_symmetries_to_boundary_points(pts, BilliardGeometry.AbsReflection[], billiard) === pts
    end

    @testset "input is not modified and optional fields are not propagated" begin
        pts2 = rf_quarter_circle_points(Float64, N)
        xy0, ds0, normal0 = copy(pts2.xy), copy(pts2.ds), copy(pts2.normal)
        full = apply_symmetries_to_boundary_points(pts2, BilliardGeometry.AbsReflection[rf_Y, rf_XY, rf_X], billiard)
        @test pts2.xy == xy0 && pts2.ds == ds0 && pts2.normal == normal0
        @test length(pts2.xy) == N
        @test isempty(full.kappa) && isempty(full.w_vs) && isempty(full.tangent)
    end

    @testset "Float32 points stay Float32" begin
        pts32 = rf_quarter_circle_points(Float32, N)
        full32 = apply_symmetries_to_boundary_points(pts32, BilliardGeometry.AbsReflection[rf_Y, rf_XY, rf_X], billiard)
        @test full32 isa BoundaryPoints{Float32}
        @test eltype(full32.ds) == Float32
        @test length(full32.xy) == 4N
    end

    # SOURCE FINDING in apply_symmetries_to_boundary_points: the XY copy is fetched from `symmetries` by type
    # (`get_sym(XYAxisReflection)`), so a symmetry list holding only the generators (Y, X) -- the form accepted by
    # apply_symmetries_to_boundary_function and the wavefunction unfolding -- fails with the opaque
    # "ArgumentError: invalid index: nothing". The XY element is implied by X and Y and should be derived.
    @testset "generator-only list (Y, X) without an XYAxisReflection entry (known failure)" begin
        ok = try
            full = apply_symmetries_to_boundary_points(pts, BilliardGeometry.AbsReflection[rf_Y, rf_X], billiard)
            length(full.xy) == 4N
        catch err
            err isa ArgumentError || rethrow()
            false
        end
        @test_broken ok
    end
end

# The three unfolding functions must agree with one another on a real billiard boundary: unfolding the points of the
# fundamental boundary of a RectangleBilliard (D2) gives the full rectangle boundary in order, and unfolding the
# values of a parity function f = x^i y^j sampled on the fundamental boundary gives exactly f on the unfolded points.
@testset "apply_symmetries_to_* - orbit folding on the Rectangle D2 boundary" begin
    a, b = 1.0, 0.6                                                            # full rectangle is [-a, a] x [-b, b]
    rect = RectangleBilliard(a, b)
    curves = BilliardGeometry.get_boundary_curves(rect)
    pts = boundary_coords(rect, AbsSampler[LinearNodes() for _ in curves], fill(8, length(curves)))
    n = length(pts.xy)
    syms = BilliardGeometry.AbsReflection[rf_Y, rf_XY, rf_X]
    full = apply_symmetries_to_boundary_points(pts, syms, rect)

    @testset "unfolded points lie on the full rectangle boundary" begin
        @test length(full.xy) == 4n
        @test all(p -> max(abs(p[1]) / a, abs(p[2]) / b) ≈ 1, full.xy)
        @test full.xy[1:n] == pts.xy
    end

    @testset "traversal is continuous and counter-clockwise" begin
        c = rf_chords(full.xy)
        @test maximum(c) < 0.2                                                 # no jumps: nodes are at most 1/6 apart
        # signed area of the polygon through the points is positive for counter-clockwise order
        A = sum(full.xy[i][1] * full.xy[mod1(i + 1, 4n)][2] - full.xy[mod1(i + 1, 4n)][1] * full.xy[i][2] for i in 1:4n) / 2
        @test A > 0
        @test A < 4 * a * b                                                    # inscribed polygon of the rectangle
    end

    @testset "boundary length is four times the fundamental one" begin
        @test sum(full.ds) ≈ 4 * sum(pts.ds)
        @test full.s[end] ≈ sum(full.ds)
    end

    @testset "parity functions unfold consistently with the points" begin
        for i in 1:2, j in 1:2
            f(p) = p[1]^i * p[2]^j
            pY = (-1.0)^i
            pX = (-1.0)^j
            u_full = apply_symmetries_to_boundary_function(f.(pts.xy), syms, [pY, 1.0, pX])
            @test u_full ≈ f.(full.xy) atol = 1e-13
        end
    end

    @testset "normals of the unfolded boundary point outward" begin
        # Only the direction is checked: the normal length of boundary_coords is not the subject of this file.
        @test length(full.normal) == 4n
        @test all(i -> dot(full.normal[i], full.xy[i]) > 0, eachindex(full.xy))
    end
end
