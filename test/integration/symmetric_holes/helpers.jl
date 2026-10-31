# Shared helpers for the symmetric-holes integration group (Phase 2 of the symmetric-holes plan, scratchpad of BilliardGeometry.jl).
# Not a test file: definitions only, no @testset. Included first by the "symmetric holes" group of integrationtests.jl, so that
# the files of the group (and a QB_TEST_FILTER run of that group alone) are self-contained. All names carry the prefix `sh_`.
#
# The common setup is a composite CFIE kernel (one component per boundary group: the outer boundary, then one per hole object)
# inside a BeynSolver with the Taylor backend off (use_chebyshev = false). Spectra are cached in SH_CACHE, so the unreduced
# spectrum of a billiard is computed once and shared by every testset that needs it.

using StaticArrays: SVector

const SH_CACHE = Dict{Tuple,Any}()

# Billiard instance cached under `key`. A SymmetrySector is bound to the billiard instance it was built from, so every spectrum of one
# billiard (unreduced and per sector) must use the same object.
sh_billiard(make, key::Symbol) = get!(make, SH_CACHE, (:billiard, key))

# Composite CFIE kernel for `arg` (a billiard or a SymmetrySector): outer component graded with `grading` (corners of the outer
# loop), `nholes` hole components, each `hole_grading` (default grading of the kernel when `nothing`).
function sh_composite_kernel(arg; nholes::Int=1, grading=GlobalCornerGrading(kressq=2), hole_grading=nothing, ppw::Real=10.0, min_pts::Int=96)
    hole() = hole_grading === nothing ? CombinedFieldIntegralEquationSolver(ppw, arg; min_pts=min_pts) :
             CombinedFieldIntegralEquationSolver(ppw, arg; min_pts=min_pts, grading=hole_grading)
    return CompositeBIMSolver(CombinedFieldIntegralEquationSolver(ppw, arg; min_pts=min_pts, grading=grading), (hole() for _ in 1:nholes)...)
end

# Cached Beyn spectrum in [k1, k2] of `b` on the sector `symmetry_sector(b, choices...)` (the whole unreduced spectrum for no
# choices). `name` identifies the billiard and the kernel settings (`kwargs`, see sh_composite_kernel); `eigenvectors = true`
# stores the layer densities for the state constructors.
function sh_spectrum(name::Symbol, b, k1::Real, k2::Real, choices::Pair...; eigenvectors::Bool=false, kwargs...)
    return get!(SH_CACHE, (:spectrum, name, Float64(k1), Float64(k2), choices, eigenvectors)) do
        arg = isempty(choices) ? b : symmetry_sector(b, choices...)
        solver = BeynSolver(sh_composite_kernel(arg; kwargs...); use_chebyshev=false, eigenvectors=eigenvectors)
        compute_spectrum(solver, b, k1, k2; multithreaded=true, show_progress=false)
    end
end

# Sorted real parts of the levels of a SpectralData.
sh_levels(data) = sort(real.(data.k))

# Matches every level of `reduced` to the nearest not yet used level of `full`. Returns the worst distance of a match and the
# sorted levels of `full` that stay unmatched.
function sh_match_levels(reduced::AbstractVector{<:Real}, full::AbstractVector{<:Real})
    pool = sort(collect(Float64, full))
    worst = 0.0
    for k in reduced
        isempty(pool) && return Inf, pool
        j = argmin(abs.(pool .- k))
        worst = max(worst, abs(pool[j] - k))
        deleteat!(pool, j)
    end
    return worst, pool
end

# True when the sorted levels come in equal pairs (doubly degenerate levels) within `atol`.
sh_are_doublets(levels::AbstractVector{<:Real}, atol::Real) =
    iseven(length(levels)) && all(abs.(levels[1:2:end] .- levels[2:2:end]) .< atol)

# Cross-method comparison of a union of sector spectra with an unreduced spectrum: same count, every level matched.
function sh_union_matches(sectors, full; atol)
    union_ks = sort(reduce(vcat, sectors; init=Float64[]))
    length(union_ks) == length(full) || return false
    worst, left = sh_match_levels(union_ks, full)
    return worst < atol && isempty(left)
end

# Checks common to every Beyn spectrum of a testset: tension below TENSION_TOL_ACCEL and |Im k| below `imag_tol`.
sh_quality_ok(data; imag_tol=IMAG_TOL_HOLES) = all(data.ten .< TENSION_TOL_ACCEL) && all(abs.(data.k_im) .< imag_tol)

# Counterclockwise square of half side `h` about `c` as a SimpleDomain with the given domain id (segment ids 1..4).
function sh_square_domain(c, h, id)
    p = (SVector(c[1] - h, c[2] - h), SVector(c[1] + h, c[2] - h), SVector(c[1] + h, c[2] + h), SVector(c[1] - h, c[2] + h))
    curves = AbsCurve[LineSegment(p[k], p[mod1(k + 1, 4)]; domain_id=id, segment_id=k) for k in 1:4]
    return SimpleDomain{Float64}(curves, collect(p), id)
end

# Billiard in the full frame [-1,1]^2 (empty registry) with the given hole objects: the Phase 1 reference for the symmetric-hole
# billiards of the quarter-square fixtures of holes_fixtures.jl.
sh_full_square_billiard(holes) = hf_HoleBilliard{Float64}(MultiplyConnectedDomain(sh_square_domain((0.0, 0.0), 1.0, 1), holes, 1), register_symmetries())

# C3Billiard(0.2) (outer sector with SymmetryWall radii and registry Cn_symmetry(3)) with a disc hole of radius `R` about the origin
# built as SymmetricHole from a 2pi/n wedge and Cn_symmetry(n): G' = C3 for n = 6 (C3 is a subgroup of C6), G' = {e} for n = 5.
# The geometry does not depend on n (a disc), only the group the hole declares does.
function sh_c3_disc_billiard(n::Int, R::Real)
    c3 = C3Billiard(0.2)
    wedge = CircleWedge(R, 2pi / n, 0.0, 0.0, 0.0, 2; bcs=AbsBoundaryCondition[SpecularReflection(), SymmetryWall(1, 2), SymmetryWall(n - 1, 2)])
    hole = SymmetricHole(wedge, Cn_symmetry(n), SVector(0.0, 0.0))
    return hf_HoleBilliard{Float64}(MultiplyConnectedDomain(c3.fundamental_domain, [hole], 1), c3.symmetries)
end

# Sinai billiard SinaiBilliard(1.0, 0.25) (D4 registry, SymmetricHole) and its Beyn spectra in the window [4, 9] (see sinai_sector_spectra.jl).
const SINAI_WINDOW = (4.0, 9.0)
sh_sinai() = sh_billiard(:sinai) do
    SinaiBilliard(1.0, 0.25)
end
sh_sinai_spectrum(choices::Pair...; kwargs...) = sh_spectrum(:sinai, sh_sinai(), SINAI_WINDOW..., choices...; kwargs...)

# C3 billiard with the disc hole declared as C6 (G' = C3) and Beyn spectra in the window [8, 16] (see c3_rotation_hole_spectra.jl);
# `name` is the cache name of the billiard (:c3_c6_disc or :c3_c5_disc), the kernel uses SmoothPeriodicGrading.
const C3_HOLE_WINDOW = (8.0, 16.0)
sh_c3_c6_billiard() = sh_billiard(:c3_c6_disc) do
    sh_c3_disc_billiard(6, 0.15)
end
sh_c3_spectrum(name::Symbol, b, choices::Pair...) = sh_spectrum(name, b, C3_HOLE_WINDOW..., choices...; grading=SmoothPeriodicGrading())
