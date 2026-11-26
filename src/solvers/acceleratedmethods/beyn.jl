################################################################################
# BEYN CONTOUR-INTEGRAL NONLINEAR EIGENSOLVER
#
# We seek nonlinear eigenpairs
#
#     A(k)u = 0,
#
# where A(k) is the boundary-integral Fredholm matrix. For a closed contour Γ
# enclosing the desired eigenvalues, Beyn's method forms the moments
#
#     A₀ = (1/2πi) ∮Γ A(z)⁻¹ V dz,
#     A₁ = (1/2πi) ∮Γ z A(z)⁻¹ V dz,
#
# with a random probing matrix V. For the circular contour
#
#     z(θ) = k₀ + R exp(iθ),
#
# the trapezoidal rule gives
#
#     A₀ ≈ Σⱼ wⱼ A(zⱼ)⁻¹V,
#     A₁ ≈ Σⱼ wⱼ zⱼ A(zⱼ)⁻¹V,
#     wⱼ = (R/nq) exp(iθⱼ).
#
# If A₀ = UΣW*, its numerical rank r determines the reduced matrix
#
#     B = Uᵣ* A₁ Wᵣ Σᵣ⁻¹.
#
# The eigenvalues of B approximate the nonlinear eigenvalues k inside Γ, while
# the corresponding layer densities are reconstructed as
#
#     X = UᵣY,
#
# where Y contains the eigenvectors of B. Candidates are validated using
#
#     ||A(k)x|| < res_tol.
#
# For closed billiards the exact spectrum is real. In a multi-window sweep we
# therefore sort projected roots globally by decreasing |Im(k)| and residual-
# check the most suspicious roots first. After `imag_k_pad` consecutive roots
# pass the residual test, the remaining roots closer to the real axis are
# accepted without constructing additional residual matrices.
#
# Weyl's leading term determines approximately m states per contour,
#
#     N(k) ≈ A k²/(4π),
#
# while `Rmax` limits the maximum contour radius.
################################################################################

"""
    BeynSolver{T,K} <: AcceleratedBIMSolver

Contour-integral nonlinear eigensolver based on Beyn's method.

The solver computes solutions of `A(k)u = 0` from contour moments on circular
contours. Matrix construction can use the Chebyshev-accelerated backend.
Multi-window spectrum calculations additionally support a global imaginary-`k`
screening strategy which residual-checks roots in decreasing `abs(imag(k))`
order and terminates after a prescribed sequence of residual-good roots.

## Fields
* `kernel::K`: Boundary-integral solver defining `A(k)`.
* `nlevels::Int`: Target number of states per Weyl window.
* `nq::Int`: Number of contour quadrature nodes.
* `r::Int`: Initial random probing rank.
* `Rmax::T`: Maximum contour radius in spectrum sweeps.
* `svd_tol::T`: Numerical-rank threshold for the zeroth contour moment.
* `res_tol::T`: Nonlinear residual threshold as a state - keep check.
* `auto_discard_spurious::Bool`: Discard roots failing residual validation.
* `use_taylor::Bool`: Use Fredholm matrix via Taylor expansion.
* `taylor_degree::Int`: Taylor degree used by the analytic Taylor-Horner backend.
* `taylor_tol::T`: Tolerance for the Taylor expansion.
* `use_chebyshev::Bool`: Use Chebyshev-accelerated matrix construction.
* `cheb_config::ChebyshevConfig{T}`: Chebyshev interpolation configuration.
* `imag_k_check::Bool`: Validate candidates jointly across sweep windows; full mode uses imaginary-`k` early stopping.
* `imag_k_pad::Int`: Consecutive good roots required before full-contour screening stops; unused in half mode.
* `imag_k_group_size::Int`: Maximum root-processing batch size; half-contour Taylor products use all roots in a window.
* `eigenvectors::Bool`: Retain layer densities in full mode or physical boundary functions in half mode.
* `use_half_contour::Bool`: Use the real-beta half contour; `nq` counts the full paired rule.
* `beta::T`: Real Green-function coupling, default 1.
"""
mutable struct BeynSolver{T<:Real,K<:SweepBIMSolver} <: AcceleratedBIMSolver
    kernel::K
    nlevels::Int
    nq::Int
    r::Int
    Rmax::T
    svd_tol::T
    res_tol::T
    auto_discard_spurious::Bool
    use_taylor::Bool
    taylor_degree::Int
    taylor_tol::T
    use_chebyshev::Bool
    cheb_config::ChebyshevConfig{T}
    imag_k_check::Bool
    imag_k_pad::Int
    imag_k_group_size::Int
    eigenvectors::Bool
    use_half_contour::Bool
    beta::T
end

"""
    BeynSolver(kernel::K; nlevels::Int=100, nq::Int=40, r::Int=200, Rmax::Real=0.5, svd_tol::Real=1e-11, res_tol::Real=1e-8, auto_discard_spurious::Bool=true, use_taylor::Bool=true, taylor_degree::Int=14, taylor_tol::Real=1e-12, use_chebyshev::Bool=false, cheb_config::Union{Nothing,ChebyshevConfig}=nothing, imag_k_check::Bool=true, imag_k_pad::Int=20, imag_k_group_size::Int=20, eigenvectors::Bool=true, use_half_contour::Bool=false, beta::Real=1.0)

Construct a Beyn nonlinear eigensolver.

With `auto_discard_spurious=true`, single-contour `solve` and `solve_vectors`
calls exhaustively validate nonlinear residuals. By default, multi-window
`compute_spectrum` uses imaginary-`k` screening in full mode and exhaustive
full-Hankel residual checks in half mode.

## Arguments
* `kernel::K`: Boundary-integral solver defining the nonlinear matrix `A(k)`.

## Keyword Arguments
* `nlevels::Int=100`: Target number of states per Weyl window.
* `nq::Int=40`: Number of contour quadrature nodes.
* `r::Int=200`: Initial random probing rank.
* `Rmax::Real=0.5`: Maximum contour radius in spectrum sweeps.
* `svd_tol::Real=1e-11`: Numerical-rank threshold for the zeroth contour moment.
* `res_tol::Real=1e-8`: Nonlinear residual threshold.
* `auto_discard_spurious::Bool=true`: Discard roots failing residual validation.
* `use_taylor::Bool=true`: Use Fredholm matrix via Taylor expansion.
* `taylor_degree::Int=14`: Taylor degree used by the analytic Taylor-Horner backend.
* `taylor_tol::Real=1e-12`: Tolerance for the Taylor expansion.
* `use_chebyshev::Bool=false`: Use Chebyshev-accelerated matrix construction.
* `cheb_config::Union{Nothing,ChebyshevConfig}=nothing`: Chebyshev configuration. If `nothing`, `ChebyshevConfig(T)` defaults are used.
* `imag_k_check::Bool=true`: Enable imaginary-`k` screening in `compute_spectrum`. Should always be used in production runs to verify spectrum integrity.
* `imag_k_pad::Int=20`: Consecutive residual-good roots required before stopping. Only applies when `imag_k_check` is enabled.
* `imag_k_group_size::Int=20`: Maximum residual-check batch size. Only applies when `imag_k_check` is enabled.
* `eigenvectors::Bool=true`: Whether `compute_spectrum` retains the layer densities returned by the Beyn solve.
* `use_half_contour::Bool=false`: Use the real-beta half contour; `nq` counts the full paired rule.
* `beta::Real=1.0`: Real Green-function coupling.

## Returns
* `BeynSolver{T,K}`: Configured Beyn solver.
"""
function BeynSolver(kernel::K; nlevels::Int=100, nq::Int=40, r::Int=200, Rmax::Real=0.5, svd_tol::Real=1e-11, res_tol::Real=1e-8, auto_discard_spurious::Bool=true, use_taylor::Bool=true, taylor_degree::Int=14, taylor_tol::Real=1e-12, use_chebyshev::Bool=false, cheb_config::Union{Nothing,ChebyshevConfig}=nothing, imag_k_check::Bool=true, imag_k_pad::Int=20, imag_k_group_size::Int=20, eigenvectors::Bool=true, use_half_contour::Bool=false, beta::Real=1.0) where {K<:SweepBIMSolver}
    T = _bim_numeric_type(kernel); cfg = cheb_config === nothing ? ChebyshevConfig(T) : cheb_config
    coupling = T(beta)
    if use_half_contour
        # TODO: corner-graded half contours.
        T===Float64 && kernel isa DoubleLayerPotentialSolver && kernel.grading isa SmoothPeriodicGrading && use_taylor && !use_chebyshev && iseven(nq) ||
            throw(ArgumentError("Half contours require Float64 Taylor DLP, smooth periodic grading and even nq"))
    end
    return BeynSolver{T,K}(kernel, nlevels, nq, r, T(Rmax), T(svd_tol), T(res_tol), auto_discard_spurious, use_taylor, taylor_degree, T(taylor_tol), use_chebyshev, cfg, imag_k_check, imag_k_pad, imag_k_group_size, eigenvectors, use_half_contour, coupling)
end


_bim_numeric_type(::BeynSolver{T}) where {T} = T

"""
    weyl_window_width(billiard, k, nlevels; fundamental_only=false)

Width `dk` of the window starting at `k` that contains approximately `nlevels` levels
according to the leading Weyl law, i.e. `sqrt(k^2 + 4π*nlevels/A) - k`.

## Arguments
* `billiard::Bi`: Billiard geometry.
* `k::T`: Lower wavenumber.
* `nlevels::Int`: Target number of levels in the window.

## Keyword Arguments
* `fundamental_only::Bool=false`: Use the fundamental-domain area instead of the full area.

## Returns
* `T`: Window width.
"""
@inline function weyl_window_width(billiard::Bi, k::T, nlevels::Int; fundamental_only::Bool=false) where {T<:Real,Bi<:AbsBilliard}
    A = fundamental_only ? fundamental_area(billiard) : area(billiard)
    return sqrt(k^2+T(4pi*nlevels/A))-k
end

"""
    plan_weyl_windows(billiard::Bi, k1::T, k2::T; nlevels::Int=10, Rmax::Real=1.0, fundamental_only::Bool=false)

Partition `[k1,k2]` into adjacent spectral windows containing approximately
`nlevels` states according to the leading Weyl law.

## Arguments
* `billiard::Bi`: Billiard geometry.
* `k1::T`: Lower spectral endpoint.
* `k2::T`: Upper spectral endpoint.

## Keyword Arguments
* `nlevels::Int=10`: Approximate number of states per window.
* `Rmax::Real=1.0`: Maximum contour radius.
* `fundamental_only::Bool=false`: Use the desymmetrized fundamental-domain area (Dirichlet walls).

## Returns
* `Vector{Tuple{T,T}}`: Adjacent spectral intervals `(kL,kR)`.
"""
function plan_weyl_windows(billiard::Bi, k1::T, k2::T; nlevels::Int=10, Rmax::Real=1.0, fundamental_only::Bool=false) where {T<:Real,Bi<:AbsBilliard}
    k2 > k1 || return Tuple{T,T}[]; nlevels > 0 || throw(ArgumentError("nlevels must be positive; received nlevels=$nlevels")); Rmax > 0 || throw(ArgumentError("Rmax must be positive; received Rmax=$Rmax"))
    intervals = Tuple{T,T}[]; maxwidth = T(2Rmax); k = k1
    while k < k2
        dk = min(weyl_window_width(billiard, k, nlevels; fundamental_only), maxwidth, k2-k); dk > zero(T) || throw(ArgumentError("Weyl window width vanished at k=$k"))
        kr = k+dk; push!(intervals, (k, kr)); k = kr
    end
    return intervals
end

"""
    beyn_disks_from_windows(intervals)

Convert real spectral windows into the contour disks used by Beyn's method: the
center is the window midpoint and the radius is half the window width.

## Arguments
* `intervals::Vector{Tuple{T,T}}`: Windows `(kL, kR)`.

## Returns
* `Tuple{Vector{Complex{T}},Vector{T}}`: Disk centers and radii.
"""
function beyn_disks_from_windows(intervals::Vector{Tuple{T,T}}) where {T<:Real}
    k0 = Vector{Complex{T}}(undef, length(intervals)); R = Vector{T}(undef, length(intervals))
    @inbounds for (i, (kL, kR)) in pairs(intervals)
        k0[i] = complex((kL+kR)/2); R[i] = (kR-kL)/2
    end
    return k0, R
end

"""
    beyn_buffer_matrices(::Type{T}, N, r, rng)

Allocate the work matrices of Beyn's method: a random probing matrix `V` of size
`N`×`r` drawn from `rng`, and the buffers `X`, `A0`, `A1` of the same size.

## Arguments
* `T`: Real element type.
* `N::Int`: Matrix dimension of the Fredholm operator.
* `r::Int`: Number of probing columns.
* `rng`: Random number generator.

## Returns
* `NTuple{4,Matrix{Complex{T}}}`: `(V, X, A0, A1)`.
"""
function beyn_buffer_matrices(::Type{T},::Val{:random}, N::Int, r::Int, rng) where {T<:Real}
    V = randn(rng, Complex{T}, N, r); X = similar(V); A0 = zeros(Complex{T}, N, r); A1 = zeros(Complex{T}, N, r)
    return V, X, A0, A1
end

beyn_buffer_matrices(::Type{T}, N::Int, r::Int, rng) where {T<:Real} = beyn_buffer_matrices(T, Val(:random), N, r, rng)

# Complex rotational half contours await an exact reflection map from geometry.
function _beyn_half_symmetry_rotation_fix(kernel::DoubleLayerPotentialSolver, pts=nothing)
    symmetry = kernel.symmetry
    symmetry isa NFoldRotation || return kernel, nothing
    sector = isempty(kernel.sym_characters) ? 0 : only(kernel.sym_characters)
    # TODO: complex rotational sectors need an exact reflection map from geometry, no API yet.
    mod(2*sector, symmetry.order)==0 || throw(ArgumentError("Half contours do not yet support complex rotational symmetry sectors; use use_half_contour=false"))
    return kernel, nothing
end

"""
    _beyn_interior_sources(kernel::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, k0::Real, r::Int, rng::AbstractRNG; multithreaded::Bool=true)

Choose `r` distinct interior points `s_j`, fixed throughout one contour.

## Arguments
* `kernel::DoubleLayerPotentialSolver`: Float64 Euclidean DLP solver defining
  the billiard and optional symmetry sector.
* `pts::BoundaryPoints{Float64}`: Full physical-boundary nodes and arclength
  weights used to estimate the mean spacing `h̄`.
* `k0::Real`: Positive real contour center; `2π/k0` sets the nominal grid spacing.
* `r::Int`: Positive number of distinct source points to select.
* `rng::AbstractRNG`: Generator used to shuffle candidate points without replacement.

## Keyword Arguments
* `multithreaded::Bool=true`: Parallelize the interior-membership mask.

## Returns
* `full::DoubleLayerPotentialSolver{Float64}`: Unreduced, smooth-periodic DLP
  kernel used to construct the full-boundary probe for reuse.
* `sources::Vector{SVector{2,Float64}}`: The `r` selected interior points.

Returns `(full, sources)`. Throws `ArgumentError` if the clearance band does
not contain enough candidate points.
"""
@experimental function _beyn_interior_sources(kernel::DoubleLayerPotentialSolver, pts::BoundaryPoints{Float64}, k0, r::Int, rng; multithreaded::Bool=true)
    #TODO Hot garbage, but it works. Needs an actual rewrite with some thought behind it but the geometry API is still under development...
    n = length(pts)
    # Fourfold boundary sampling refinining the boundary choice for easier interior source points placement 
    # ppw = 1.0 does nothing due to high min_pts, very hacky
    full = DoubleLayerPotentialSolver(1.0, kernel.billiard; min_pts=4n, grading=SmoothPeriodicGrading(), symmetry=nothing, sym_characters=())
    # Distance checks need only the same fine-boundary positions, not its
    # tangents, arclength coordinates or quadrature fields.
    _require_no_holes(full, full.billiard)
    comp = full_boundary(full.billiard)
    isempty(comp) && error("Boundary cannot be empty.")
    #TODO: Filthy AI hack that I needed to put comments for myself just to understand what tf is going on, just really needed it to work. 
    # Črt maybe you can rewrite this properly with the updated geometry API
    lens, cum, Ltot = component_lengths(comp)
    fine_count = max(full.min_pts, round(Int, Float64(k0)*Ltot*full.pts_scaling_factor[1]/(2*pi)))
    fine_count = _bim_round_nodes(fine_count, lcm(2, _bim_solver_node_multiple(full)), nothing)
    fine_xy = Vector{SVector{2,Float64}}(undef, fine_count)
    @inbounds for i in eachindex(fine_xy)
        t = Float64(s_mid(i, fine_count))
        s = (t/Float64(2*pi))*Ltot
        if s >= Ltot
            segment = 1; u = 0.0
        else
            segment = clamp(searchsortedlast(cum, s), 1, length(lens))
            while segment < length(lens) && s >= cum[segment+1]
                segment += 1
            end
            local_s = s-cum[segment]
            u = lens[segment]==0.0 ? 0.0 : local_s/lens[segment]
            u = clamp(u, 0.0, 1.0)
        end
        fine_xy[i] = curve(comp[segment], u)
    end
    step = sum(pts.ds)/n # easy estimate of mean spacing h̄
    # With h=step and 20h cells, all nodes within 10h lie in the 3x3 neighborhood.
    cellsize = 20*step; cells = Dict{Tuple{Int,Int},Vector{Int}}()
    # Conservative bounds protect distance arithmetic, integer indices and bin precision.
    coordinate_limit = sqrt(floatmax(Float64))/4; quotient_limit = min(2.0^45, Float64(typemax(Int))/2)
    safe_cells = isfinite(cellsize) && sqrt(floatmin(Float64))<cellsize<coordinate_limit
    if safe_cells
        for j in eachindex(fine_xy)
            point = fine_xy[j]; qx = point[1]/cellsize; qy = point[2]/cellsize
            # Unsafe coordinates disable binning; retain the original full scan.
            if !(isfinite(qx) && isfinite(qy) && max(abs(point[1]),abs(point[2]))<coordinate_limit && max(abs(qx),abs(qy))<quotient_limit)
                safe_cells = false; break
            end
            # Store fine-boundary indices by cell.
            push!(get!(() -> Int[], cells, (floor(Int,qx),floor(Int,qy))), j)
        end
    end
    # Select sources in the fundamental domain when a sector is active.
    billiard = kernel.billiard; fundamental = kernel.symmetry!==nothing
    curves = fundamental ? get_boundary_curves(billiard) : full_boundary(billiard)
    xb = Float64[]; yb = Float64[]
    # Bounding-box heuristics, in principle arbitrary #TODO Maybe make this more adaptive to the geometry
    # Omit t=1 to avoid repeating the endpoint, leaving one fewer sample.
    # This is extremely hacky but provides a dense set of candidate points for interior source placement.
    for crv in curves
        t = range(0.0, 1.0; length=max(512, round(Int, 1000/crv.length)))[1:end-1]
        boundary = curve(crv, t)
        append!(xb, getindex.(boundary, 1)); append!(yb, getindex.(boundary, 2))
    end
    xb[end] = xb[1]; yb[end] = yb[1]
    xlim = extrema(xb); ylim = extrema(yb); dx = 2*pi/Float64(real(k0))
    # Candidate spacing is at most 2π/k_0 (wavelength scale). The 512-node floor per axis supplies a dense pool at low k
    nx = max(512, ceil(Int, (xlim[2]-xlim[1])/dx)+1); ny = max(512, ceil(Int, (ylim[2]-ylim[1])/dx)+1)
    xgrid = collect(range(xlim[1], xlim[2]; length=nx)); ygrid = collect(range(ylim[1], ylim[2]; length=ny))
    grid = vec([SVector{2,Float64}(x,y) for x in xgrid, y in ygrid]); mask = Vector{Bool}(undef, length(grid))
    # Cache domain lists and test curves lazily; retain the geometry predicate for holes.
    domains = billiard.fundamental_domain isa BilliardGeometry.AbsMultiplyConnectedDomain ? nothing : BilliardGeometry.get_all_domains(billiard)
    inside_domain(point) = domains===nothing ? BilliardGeometry.is_inside(billiard, point) : any(domain -> all(crv -> BilliardGeometry.is_inside(crv, point), domain.boundary), domains)
    @use_threads multithreading=multithreaded for i in eachindex(grid) # do domain membership check for each candidate point #TODO Make it more efficient
        point = grid[i]; inside = inside_domain(point)
        if !fundamental && !inside
            inside = any(sym -> inside_domain(BilliardGeometry.apply_symmetry(sym, point)), billiard.symmetries)
        end
        mask[i] = inside
    end
    indices = findall(mask)
    # Shuffle without replacement, ensuring there is no pattern to sources to prevent nodal line evals #TODO Pathological cases check coupling to eigenspaces
    shuffle!(rng, indices)
    sources = eltype(pts.xy)[] # preallocate the array for selected sources
    for i in indices
        source = grid[i]
        # Exclude the candidate-grid perimeter.
        first(xgrid)<source[1]<last(xgrid) && first(ygrid)<source[2]<last(ygrid) || continue
        qx = source[1]/cellsize; qy = source[2]/cellsize
        if safe_cells && isfinite(qx) && isfinite(qy) && max(abs(source[1]),abs(source[2]))<coordinate_limit && max(abs(qx),abs(qy))<quotient_limit
            cx = floor(Int,qx); cy = floor(Int,qy); d = Inf
            # Same nearest distance when d<=10h; otherwise the source is rejected anyway.
            for iy in cy-1:cy+1, ix in cx-1:cx+1
                neighbors = get(cells, (ix,iy), nothing)
                neighbors===nothing && continue
                for j in neighbors
                    d = min(d, norm(source-fine_xy[j]))
                end
            end
        else
            d = minimum(norm(source-x) for x in fine_xy)
        end
        # step=mean(ds); several boundary spacings away (5h away as emphasized by Barnett: https://arxiv.org/abs/1310.5352 (Remark 6): Evaluation of layer potentials close to the boundary for Laplace and Helmholtz problems on analytic planar domains), while capping their depth (10 steps from the boundary provided good results - this is in preparation for magnetic edge states where not a lot of mass is in the deep interior)
        5*step<=d<=10*step || continue # tipically enough
        push!(sources, source); length(sources)==r && break # we want only r sources
    end
    length(sources)==r || throw(ArgumentError("Could not populate the sources grid, maybe check boundary resolution or step size"))
    return full, sources
end

"""
    beyn_buffer_matrices(T, Val(:interior), N, r, rng; kernel, pts, k0, p, sources=nothing, beta=1.0, Rmax=0.5, multithreaded=true, conjugation_map=nothing)

Build a fixed interior-source Taylor probe and allocate the Beyn buffers:

    Gβ = -(Y₀+beta*J₀)/4,  q = 2∂ₙGβ,
    b_l = E†W ∂ₖ^lq(k₀)/l!,  # Taylor coefficients of the weighted forcing term
    c_j² = Σ_l Rmax^(2l) ‖b_l[:,j]‖₂², # rescaling of each column while preserving holomorphy
    C[l+1,(j-1)N+i] = b_l[i,j]/c_j. # normalized Taylor coefficient for column j

`W=diag(pts.ds)` and `E` extends sector values (`E=I` without symmetry).
Set `M=E†WE`; `u_j` is the reduced physical boundary normal derivative.
Automatic setup later gives the weighted response `X[:,j] ≈ M u_j/c_j`.
Complex rotational sectors are currently unsupported. The reserved path
`b_l ← (b_l+H*conj(b_l))/2` reflects weighted rows and averages a source pair, pending a future geometry API.
Fixed sources and scales preserve holomorphy.

## Arguments
* `T::Type{<:Real}`: Buffer element type (`Float64` for half contours).
* `N::Int`: Reduced boundary dimension.
* `r::Int`: Probe column count.
* `rng::AbstractRNG`: Automatic source-placement generator.

## Keyword Arguments
* `kernel::DoubleLayerPotentialSolver`: Smooth-periodic Euclidean DLP kernel.
* `pts::BoundaryPoints{Float64}`: Full boundary nodes and arclength weights.
* `k0::Real`, `p::Int`: Positive real center and Taylor degree.
* `sources::Union{Nothing,AbstractVector}=nothing`: Optional `r` fixed interior points.
* `beta::Real=1.0`, `Rmax::Real=0.5`: Real coupling and column-scaling radius.
* `multithreaded::Bool=true`: Parallelize source and coefficient construction.
* `conjugation_map::Union{Nothing,NamedTuple}=nothing`: Optional `(rows, factors)`
  map for lower-contour reconstruction; reserved for a future symmetry API.

## Returns
* `V::Matrix{Complex{T}}`: Uninitialized `(N,r)` forcing buffer, later filled as
  `V_{ij}(k)=Σ_{l=0}^p C[l+1,(j-1)N+i](k-k₀)^l`.
* `X::Matrix{Complex{T}}`: Uninitialized `(N,r)` response buffer, later solving
  `A_{s,χ}(k)ᵀ X(k)=V(k)` with the source-normal matrix in a supported real sector.
* `A0, A1::Matrix{Complex{T}}`: Zeroed `(N,r)` accumulators for the contour moments
  `A0 ≈ (1/2πi)∮ X(k)dk` and `A1 ≈ (1/2πi)∮ kX(k)dk`.
* `C::Matrix{ComplexF64}`: `(p+1,N*r)` Taylor coefficients of `V(k)`, including
  boundary weights and column normalization in automatic setup. Explicit sources
  return raw helper coefficients. Evaluate with `_horner(C, (j-1)*N+i, k-k0)`.

Returns `(V, X, A0, A1, C)`.
"""
function beyn_buffer_matrices(::Type{T}, ::Val{:interior}, N::Int, r::Int, rng; kernel, pts, sources=nothing, k0, p::Int, beta::Real=1.0, Rmax::Real=0.5, multithreaded::Bool=true, conjugation_map=nothing) where {T<:Real}
    if sources===nothing
        # Start with n full boundary nodes. Symmetry groups them into m orbits, one reduced boundary value represents all nodes in each orbit (as everywhere else the symmetry is applied).
        n = length(pts); m = boundary_matrix_size(kernel, pts)
        orbits = kernel.symmetry===nothing ? nothing : _fold_boundary(Float64, kernel.billiard, n, kernel.symmetry, kernel.sym_characters)
        # Conjugation swaps complex rotation sectors; TODO reflection map awaits the geometry API.
        conjugation_map===nothing && ((_, conjugation_map) = _beyn_half_symmetry_rotation_fix(kernel, pts)) #TODO Not yet implemented for complex irreps
        # The rotation gate above proves real characters from the integer sector;
        # cis(pi) may still leave imaginary roundoff in their stored phases.
        orbits===nothing || conjugation_map!==nothing || kernel.symmetry isa NFoldRotation || all(z -> isreal(z) && abs(real(z))==1, orbits.phase) || throw(ArgumentError("Complex half contours require a reflection-compatible rotational sector")) #TODO Not yet implemented for complex irreps
        # Keep source positions fixed throughout this contour.
        full, sources = _beyn_interior_sources(kernel, pts, k0, r, rng; multithreaded)
        # For source j, q_{ij}(k)=2∂_{n_i}G_β(x_i,s_j;k).
        # Generate each Taylor coefficient directly into its weighted orbit sum.
        # Retain the raw helper's division by ds before applying the row weight.
        coupling = _real_beta_coupling(beta)
        source_factors = ComplexF64[conj(one(ComplexF64))/(1*pts.ds[i]) for i in 1:n]
        source_workspaces = [TaylorWorkspace(p) for _ in 1:Threads.maxthreadid()]
        C = zeros(ComplexF64, p+1, m*r); scales = zeros(r)
        # Why multiply by ds? The library uses in general a source-normal matrix A_s, #TODO Potentially in the future migrate to the target-normal convention always, as both PH and Wavef can be immediately constructed.
        # Freeze captured bindings so the threaded loop keeps concrete types.
        let C = C, conjugation_map = conjugation_map, full = full, sources = sources
            @use_threads multithreading=multithreaded for j in 1:r
                source_workspace = source_workspaces[Threads.threadid()]
                sx, sy = Float64(sources[j][1]), Float64(sources[j][2])
                @inbounds for i in 1:n
                    source_coefficients = _kernel_taylor!(full, source_workspace, pts, sx, sy, Float64(real(k0)), p, i)
                    row = orbits===nothing ? i : orbits.orbit_of[i]
                    # Apply the same row sum to every Taylor degree l.
                    weight = pts.ds[i]*(orbits===nothing ? one(T) : conjugation_map===nothing ? real(orbits.phase[i]) : conj(orbits.phase[i]))
                    # The probe is real at real k_0; symmetry phases can make the sum complex.
                    @simd for ell in 1:p+1
                        probe = source_factors[i]*(real(source_coefficients[ell])-coupling*imag(source_coefficients[ell]))
                        C[ell, (j-1)*m+row] += weight*real(probe)
                    end
                end
                # All m rows for source j occupy this block of C's columns.
                # The contour evaluation uses C[:,(j-1)*m+a] to fill V[a,j].
                columns = (j-1)*m+1:j*m
                if conjugation_map!==nothing #TODO Not yet implemented for complex irreps
                    # Conjugation changes sector χ to χ^{bar}; reflection reverses the
                    # rotation and brings it back to χ, allowing lower-node "free" construction.
                    rows = conjugation_map.rows; factors = conjugation_map.factors; offset = (j-1)*m
                    @inbounds for a in 1:m
                        b = rows[a]; a<=b || continue
                        @simd for ell in 1:p+1
                            u = C[ell,offset+a]; v = C[ell,offset+b]
                            C[ell,offset+a] = (u+factors[a]*conj(v))/2
                            C[ell,offset+b] = (v+factors[b]*conj(u))/2
                        end
                    end
                end
                #TODO Thanks ChatGPT for this suggestion, was quite annoying sometimes for columns with very different magnitudes
                # Measure the typical size of each source's probe column, then scale
                # the columns to balance their influence on the contour moments.
                # For source j, let b_{l,j}=b_l[:,j] and
                # P_j(δ)=Σ_{l=0}^p b_{l,j} δ^{l}.
                # On the reference circle δ=Rmax*exp(iθ), this becomes
                # P_j=Σ_{l=0}^{p} b_{l,j} R_{max}^{l} exp(ilθ).
                # Squaring its norm produces both individual and cross terms:
                # ‖P_j‖_2^{2}=Σ_{l,h=0}^{p} R_{max}^{l+h}(b_{h,j}† b_{l,j}) exp(i(l-h)θ).
                # Averaging this squared norm over one full circle has average zero for l≠h and one for l=h:
                # (1/2π)∫_0^{2π} exp(i(l-h)θ)dθ = 0 (l≠h), 1 (l=h).
                # All cross terms disappear, leaving
                # c_j^{2}=(1/2π)∫_0^{2π} ‖P_j(R_{max} exp(iθ))‖_2^{2} dθ =Σ_{l=0}^{p} R_{max}^{2l} ‖b_{l,j}‖_2^{2}.
                # Thus c_j is the root-mean-square size of this probe column.
                # Dividing every coefficient in column j by the same c_j makes
                # its average squared norm on the reference circle equal to one.
                # Compute c_j once from the coefficients, without contour evaluations.
                # Keeping it independent of k preserves the probe's analytic dependence on k.
                scale2 = 0.0; power = 1.0
                for ell in 1:p+1
                    scale2 += power^2*sum(abs2, @view C[ell,columns])
                    power *= Rmax
                end
                scales[j] = sqrt(scale2)
                # A zero column cannot be normalized.
                scales[j]>0 || throw(ArgumentError("An interior source gives a zero column"))
                # P_j(k-k_0) approximates rhs_j(k). Divide its coefficients by c_j:
                # V_j(k)=P_j(k-k_0)/c_j, so we have for the following j-column eq:
                # A_{s,χ}^{bar}(k)ᵀ X_j(k)=V_j(k),
                # so X_j≈M u_{red,j}/c_j. solve_vectors divides each row by its total_weight.
                # Fixed nonzero c_j preserves its poles and residue ranks.
                @views C[:, columns] ./= scales[j]
            end
        end
    else
        # Provided sources use the existing raw helper API. The caller must handle
        # the weights, matrix orientation and any column normalization.
        C = beyn_interior_probe_taylor(kernel, pts, sources, Float64(real(k0)), p; beta, multithreaded)
    end
    V = Matrix{Complex{T}}(undef, N, r); X = similar(V); A0 = zeros(Complex{T}, N, r); A1 = zeros(Complex{T}, N, r)
    return V, X, A0, A1, C
end

function tune_cheb_config(cs::DoubleLayerPotentialSolver, pts::BoundaryPoints, k, cfg::ChebyshevConfig)
    z = ComplexF64[ComplexF64(k)]; G = boundary_geom_cache(pts, _is_nontrivial_dlp_grading(pts))
    rmin, rmax = _cheb_geom_rminmax(G, z)
    _, _, tuned = tune_dlp_cheb_plans(rmin, rmax, z, cfg)
    return ChebyshevConfig(_bim_numeric_type(cs); n_panels_h=tuned.n_panels_h, M_h=tuned.M_h, n_panels_j=tuned.n_panels_j, M_j=tuned.M_j, tol=tuned.tol, max_iter=tuned.max_iter, sampling_points=tuned.sampling_points, grow_panels=tuned.grow_panels, grow_M=tuned.grow_M, param_strategy=:manual)
end

function tune_cheb_config(cs::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints, k, cfg::ChebyshevConfig)
    z = ComplexF64[ComplexF64(k)]; G = boundary_geom_cache(pts, _is_nontrivial_dlp_grading(pts))
    rmin, rmax = _cheb_geom_rminmax(G, z)
    _, _, _, _, tuned = tune_cfie_cheb_plans(rmin, rmax, z, cfg)
    return ChebyshevConfig(_bim_numeric_type(cs); n_panels_h=tuned.n_panels_h, M_h=tuned.M_h, n_panels_j=tuned.n_panels_j, M_j=tuned.M_j, tol=tuned.tol, max_iter=tuned.max_iter, sampling_points=tuned.sampling_points, grow_panels=tuned.grow_panels, grow_M=tuned.grow_M, param_strategy=:manual)
end

# Chebyshev-accelerated matrix construction is not implemented for
# CompositeBIMSolver (see `_construct_matrices_multi_k_cheb` below), so there
# is no Chebyshev plan to tune. `compute_spectrum` still calls
# `tune_cheb_config` unconditionally whenever `imag_k_check` is enabled
# (independent of `use_chebyshev`), so this falls back to returning `cfg`
# unchanged rather than erroring.
tune_cheb_config(::CompositeBIMSolver, ::BoundaryPoints, k, cfg::ChebyshevConfig) = cfg

function _construct_matrices_multi_k_cheb(cs::DoubleLayerPotentialSolver, pts::BoundaryPoints{T}, zj::Vector{ComplexF64}, cfg::ChebyshevConfig; multithreaded::Bool = true) where {T<:Real}
    T === Float64 || error("Chebyshev-accelerated Beyn evaluation requires Float64")
    N = length(pts); graded = _is_nontrivial_dlp_grading(pts); G = boundary_geom_cache(pts, graded); Rmat = zeros(T, N, N); kress_R!(Rmat)
    rmin, rmax = _cheb_geom_rminmax(G, zj); plans1, plansj1, _ = tune_dlp_cheb_plans(rmin, rmax, zj, cfg)
    C = ChebRadialLookupCache(G, plans1[1], plansj1[1]; multithreaded)
    if cs.symmetry === nothing
        Tbufs = [Matrix{ComplexF64}(undef, N, N) for _ in zj]
        _dlp_fredholm_full_multi_k_cheb!(Tbufs, pts, Rmat, G, C, zj, plans1, plansj1; multithreaded)
    else
        orbits = _fold_boundary(T, cs.billiard, N, cs.symmetry, cs.sym_characters); M = fundamental_size(orbits)
        Tbufs = [Matrix{ComplexF64}(undef, M, M) for _ in zj]
        _dlp_fredholm_reduced_multi_k_cheb!(Tbufs, pts, Rmat, G, C, orbits, zj, plans1, plansj1; multithreaded)
    end
    return Tbufs
end

function _construct_matrices_multi_k_cheb(cs::CombinedFieldIntegralEquationSolver, pts::BoundaryPoints{T}, zj::Vector{ComplexF64}, cfg::ChebyshevConfig; multithreaded::Bool = true) where {T<:Real}
    T === Float64 || error("Chebyshev-accelerated Beyn evaluation requires Float64")
    N = length(pts); graded = _is_nontrivial_dlp_grading(pts); G = boundary_geom_cache(pts, graded); Rmat = zeros(T, N, N); kress_R!(Rmat)
    rmin, rmax = _cheb_geom_rminmax(G, zj); plans0, plans1, plansj0, plansj1, _ = tune_cfie_cheb_plans(rmin, rmax, zj, cfg)
    C = ChebRadialLookupCache(G, plans1[1], plansj1[1]; multithreaded)
    if cs.symmetry === nothing
        Tbufs = [Matrix{ComplexF64}(undef, N, N) for _ in zj]
        _cfie_fredholm_full_multi_k_cheb!(Tbufs, pts, Rmat, G, C, zj, plans0, plans1, plansj0, plansj1; multithreaded)
    else
        orbits = _fold_boundary(T, cs.billiard, N, cs.symmetry, cs.sym_characters); M = fundamental_size(orbits)
        Tbufs = [Matrix{ComplexF64}(undef, M, M) for _ in zj]
        _cfie_fredholm_reduced_multi_k_cheb!(Tbufs, pts, Rmat, G, C, orbits, zj, plans0, plans1, plansj0, plansj1; multithreaded)
    end
    return Tbufs
end

_construct_matrices_multi_k_cheb(::CompositeBIMSolver, ::BoundaryPoints, ::Vector{ComplexF64}, ::ChebyshevConfig; multithreaded::Bool=true) = error("Chebyshev-accelerated Beyn is not implemented for CompositeBIMSolver")

"""
    validate_beyn_taylor(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, p::Int; nsample::Int=5, multithreaded::Bool=true) -> Float64

Validate the Taylor approximation against directly constructed BIM Fredholm
matrices on the circular Beyn contour. At each contour point `z`, the reported
relative error is

    ‖Aₚ(z)-A(z)‖ / ‖A(z)‖,

where `Aₚ` is the degree-`p` Taylor approximation about `k0`.

## Arguments
- `solver::SweepBIMSolver`: BIM solver defining the Fredholm matrix.
- `pts::BoundaryPoints{Float64}`: Boundary discretization.
- `k0::Float64`: Taylor expansion center.
- `R::Float64`: Beyn contour radius.
- `p::Int`: Taylor expansion degree.

## Keyword Arguments
- `nsample::Int=5`: Number of validation points on the contour.
- `multithreaded::Bool=true`: Whether matrix construction uses threading.
- `reference=nothing`: Directly constructed matrices at the `nsample` contour points (see `_beyn_taylor_reference`), reused to avoid rebuilding them for every degree.

## Returns
- `Float64`: Worst relative error over the sampled contour points.
"""
function validate_beyn_taylor(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, p::Int; nsample::Int=5, multithreaded::Bool=true, reference=nothing)::Float64
    Ap = _construct_matrices_multi_k_taylor(solver, pts, k0, R, nsample, p; multithreaded)
    ref = reference === nothing ? _beyn_taylor_reference(solver, pts, k0, R, nsample; multithreaded) : reference
    worst = 0.0
    for q in 1:nsample
        Ad = ref[q]
        worst = max(worst, norm(Ap[q]-Ad)/norm(Ad))
    end
    return worst
end

function _beyn_taylor_reference(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, nsample::Int; multithreaded::Bool=true)
    θ = 2π .* (0:nsample-1) ./ nsample
    return [construct_matrices(solver, pts, k0+R*cis(θ[q]); multithreaded) for q in 1:nsample]
end

"""
    _tune_beyn_taylor_degree!(solver::BeynSolver, pts, k0, R; multithreaded::Bool=true) -> Int

Determine the Taylor degree used for Beyn contour-matrix construction.

The degree is validated on the first contour having the largest radius and is
increased or decreased from `solver.taylor_degree` until the smallest degree
satisfying `solver.taylor_tol` is found. The resulting degree is stored in
`solver.taylor_degree`.

## Arguments
- `solver::BeynSolver`: Beyn solver and Taylor configuration.
- `pts`: Boundary discretizations for the spectral windows.
- `k0`: Beyn contour centers.
- `R`: Beyn contour radii.

## Keyword Arguments
- `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
- `Int`: Validated Taylor degree.
"""
function _tune_beyn_taylor_degree!(solver::BeynSolver, pts, k0, R; multithreaded::Bool=true)::Int
    Rcal, ical = findmax(R); kcal = Float64(real(k0[ical])); p0 = solver.taylor_degree; p = p0
    reference = _beyn_taylor_reference(solver.kernel, pts[ical], kcal, Float64(Rcal), 5; multithreaded)
    err = validate_beyn_taylor(solver.kernel, pts[ical], kcal, Float64(Rcal), p; multithreaded, reference)
    if err>solver.taylor_tol
        while err>solver.taylor_tol
            p += 1
            err = validate_beyn_taylor(solver.kernel, pts[ical], kcal, Float64(Rcal), p; multithreaded, reference)
        end
    else
        while p>2
            err1 = validate_beyn_taylor(solver.kernel, pts[ical], kcal, Float64(Rcal), p-1; multithreaded, reference)
            err1>solver.taylor_tol && break
            p -= 1; err = err1
        end
    end
    solver.taylor_degree = p
    return p
end

function _accumulate_beyn_moments!(A0, A1, X, V, Fs, zj, wj)
    xv = reshape(X, :); a0v = reshape(A0, :); a1v = reshape(A1, :)
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in eachindex(zj)
        ldiv!(X, Fs[j], V); BLAS.axpy!(wj[j], xv, a0v); BLAS.axpy!(wj[j]*zj[j], xv, a1v)
    end
    return nothing
end

"""
    construct_matrices(solver::BeynSolver, pts::BoundaryPoints, k0, R; multithreaded::Bool=true, rng=MersenneTwister(0), cheb_config::ChebyshevConfig=solver.cheb_config, taylor_degree::Int=solver.taylor_degree)

Construct the zeroth and first contour moments used by Beyn's nonlinear
eigensolver. Fredholm matrices are assembled and factorized once; if the
zeroth moment remains rank-saturated, the probing rank is increased while
reusing those factorizations.

Matrix construction uses the analytic Taylor-Horner backend when
`solver.use_taylor=true`, otherwise the radial-Chebyshev backend when
`solver.use_chebyshev=true`, and direct construction otherwise.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Complex contour center.
* `R`: Contour radius.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `rng=MersenneTwister(0)`: Random-number generator for probing matrices.
* `cheb_config::ChebyshevConfig=solver.cheb_config`: Chebyshev configuration used by the radial-Chebyshev backend.
* `taylor_degree::Int=solver.taylor_degree`: Taylor degree used by the analytic Taylor-Horner backend.

## Returns
* `A0::Matrix{Complex{T}}`: Zeroth contour moment.
* `A1::Matrix{Complex{T}}`: First contour moment.
"""
function construct_matrices(solver::BeynSolver, pts::BoundaryPoints, k0, R; multithreaded::Bool=true, rng=MersenneTwister(0), cheb_config::ChebyshevConfig=solver.cheb_config, taylor_degree::Int=solver.taylor_degree)
    return _construct_matrices(solver, Val(solver.use_half_contour), pts, k0, R; multithreaded, rng, cheb_config, taylor_degree)
end

function _construct_matrices(solver::BeynSolver, ::Val{true}, pts::BoundaryPoints, k0, R; multithreaded::Bool=true, rng=MersenneTwister(0), cheb_config::ChebyshevConfig=solver.cheb_config, taylor_degree::Int=solver.taylor_degree, coefficients=nothing)
    kernel = solver.kernel; nq = solver.nq; N = boundary_matrix_size(kernel, pts); r = min(solver.r, N); p = taylor_degree
    isreal(k0) && real(k0)>R || throw(ArgumentError("Half contours require a real center and a disk in Re(k)>0"))
    center = Float64(real(k0)); radius = Float64(R)
    source, conjugation_map = _beyn_half_symmetry_rotation_fix(kernel, pts)
    V, X, A0, A1, C = beyn_buffer_matrices(Float64, Val(:interior), N, r, rng; kernel, pts, k0=center, p, beta=solver.beta, Rmax=solver.Rmax, multithreaded, conjugation_map)
    t = ComplexF64[cis((2*j-1)*pi/nq) for j in 1:nq÷2]; ks = center.+radius.*t
    # Evaluate all probes together: vec(V_j)=Cᵀ[1,δ_j,...,δ_j^p], δ_j=ks[j]-center.
    powers = Matrix{ComplexF64}(undef, p+1, length(ks))
    @inbounds for j in eachindex(ks)
        powers[1,j] = 1.0
        for l in 1:p
            powers[l+1,j] = powers[l,j]*(ks[j]-center)
        end
    end
    probes = Array{ComplexF64}(undef, N, r, length(ks))
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(reshape(probes, N*r, length(ks)), transpose(C), powers)
    matrices = _construct_matrices_multi_k_taylor_real_beta(source, pts, center, ks, p; beta=solver.beta, multithreaded, coefficients)
    @blas_multi_then_1 MAX_BLAS_THREADS Fs = [lu!(A; check=false) for A in matrices]
    @blas_multi_then_1 MAX_BLAS_THREADS for j in eachindex(ks)
        ldiv!(X, transpose(Fs[j]), @view(probes[:,:,j]))
        weight = (radius/nq)*t[j]
        if conjugation_map===nothing
            @. A0 += 2*real(weight*X)
            @. A1 += 2*real(weight*ks[j]*X)
        else
            # The lower node is conj(ks[j]), but #TODO not yet implemented for complex irreps
            rows = conjugation_map.rows; factors = conjugation_map.factors; weight1 = weight*ks[j]
            @inbounds for b in 1:r
                @simd for a in 1:N
                    lower = factors[a]*conj(X[rows[a],b])
                    A0[a,b] += weight*X[a,b]+conj(weight)*lower
                    A1[a,b] += weight1*X[a,b]+conj(weight1)*lower
                end
            end
        end
    end
    return A0, A1 # x=M*u
end

function _construct_matrices(solver::BeynSolver, ::Val{false}, pts::BoundaryPoints, k0, R; multithreaded::Bool=true, rng=MersenneTwister(0), cheb_config::ChebyshevConfig=solver.cheb_config, taylor_degree::Int=solver.taylor_degree)
    T = _bim_numeric_type(solver); N = boundary_matrix_size(solver.kernel, pts); k0c = Complex{T}(k0); Rc = T(R); nq = solver.nq
    θ = range(zero(T), 2T(pi); length=nq+1)[1:end-1]; ej = cis.(θ); zj = k0c .+ Rc.*ej; wj = (Rc/nq).*ej
    if solver.use_taylor
        Tbufs = _construct_matrices_multi_k_taylor(solver.kernel, pts, Float64(real(k0c)), Float64(Rc), nq, taylor_degree; multithreaded)
    elseif solver.use_chebyshev
        Tbufs = _construct_matrices_multi_k_cheb(solver.kernel, pts, ComplexF64.(zj), cheb_config; multithreaded)
    else
        Tbufs = Vector{Matrix{Complex{T}}}(undef, nq)
        @inbounds for j in eachindex(zj)
            Tbufs[j] = construct_matrices(solver.kernel, pts, zj[j]; multithreaded)
        end
    end
    F1 = lu!(Tbufs[1]; check=false); Fs = Vector{typeof(F1)}(undef, nq); Fs[1] = F1
    @blas_multi_then_1 MAX_BLAS_THREADS @inbounds for j in 2:nq
        Fs[j] = lu!(Tbufs[j]; check=false)
    end
    r = min(solver.r, N)
    while true
        V, X, A0, A1 = beyn_buffer_matrices(T, N, r, rng); _accumulate_beyn_moments!(A0, A1, X, V, Fs, zj, wj)
        @blas_multi_then_1 MAX_BLAS_THREADS Σ = svdvals(A0)
        rk = count(>=(solver.svd_tol), Σ)
        rk < r && return A0, A1
        r == N && throw(ArgumentError("Beyn moment remains rank-saturated at maximum probe rank N=$N"))
        r = min(r+solver.r, N)
    end
end

"""
    _beyn_projected_solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0), cheb_config::ChebyshevConfig=solver.cheb_config)

Perform the projected stage of Beyn's method without nonlinear-residual
validation. The contour has center `k0` and radius `dk/2`. The reduced Beyn
matrix is diagonalized, layer density vectors are reconstructed, and roots
outside the contour are discarded.
In half mode, `coefficients` optionally retains the full Taylor expansion
for subsequent residual checks in the same window.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Contour center.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `rng=MersenneTwister(0)`: Random-number generator for probing.
* `cheb_config::ChebyshevConfig=solver.cheb_config`: Chebyshev configuration.

## Returns
* `ks::Vector{Complex{T}}`: Complex roots inside the contour.
* `X::Matrix{Complex{T}}`: Layer density columns in full mode; weighted boundary functions `M*u` in half mode.
"""
function _beyn_projected_solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0), cheb_config::ChebyshevConfig=solver.cheb_config, coefficients=nothing)
    T = _bim_numeric_type(solver); k0c = Complex{T}(k0); Rc = T(dk)/2
    if coefficients===nothing
        @blas_1 A0, A1 = construct_matrices(solver, pts, k0c, Rc; multithreaded, rng, cheb_config)
    else
        @blas_1 A0, A1 = _construct_matrices(solver, Val(true), pts, k0c, Rc; multithreaded, rng, cheb_config, coefficients)
    end
    N = size(A0, 1)
    @blas_multi_then_1 MAX_BLAS_THREADS U, Σ, W = svd!(A0; full=false)
    rk = count(>=(solver.svd_tol), Σ)
    solver.use_half_contour && rk==size(A0,2) && throw(ArgumentError("Half-contour probe is saturated; increase r or reduce the window"))
    rk == 0 && return Complex{T}[], Matrix{Complex{T}}(undef, N, 0)
    Uk = @view U[:, 1:rk]; Wk = copy(@view W[:, 1:rk]); Σk = @view Σ[1:rk]; tmp = Matrix{Complex{T}}(undef, N, rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(tmp, A1, Wk)
    @inbounds for j in 1:rk
        @views tmp[:, j] ./= Σk[j]
    end
    B = Matrix{Complex{T}}(undef, rk, rk)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(B, adjoint(Uk), tmp)
    @blas_multi_then_1 MAX_BLAS_THREADS ev = eigen!(B)
    ks = ev.values; idx = findall(j -> abs(ks[j]-k0c)<=Rc, eachindex(ks))
    isempty(idx) && return Complex{T}[], Matrix{Complex{T}}(undef, N, 0)
    X = Matrix{Complex{T}}(undef, N, length(idx))
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(X, Uk, ev.vectors[:,idx])
    return ks[idx], X
end

# Compute a candidate residual: A*x for full-contour densities, Aᵀ*x for half-contour x=M*u.
@inline function _beyn_residual(solver::BeynSolver, pts, k, x, y; multithreaded::Bool=true)
    kernel = solver.use_half_contour ? first(_beyn_half_symmetry_rotation_fix(solver.kernel)) : solver.kernel
    @blas_1 A = construct_matrices(kernel, pts, k; multithreaded)
    @blas_multi_then_1 MAX_BLAS_THREADS mul!(y, solver.use_half_contour ? transpose(A) : A, x)
    return norm(y)
end

"""
    _beyn_solve_core(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0), cheb_config::ChebyshevConfig=solver.cheb_config)

Perform a single-contour Beyn solve. With `auto_discard_spurious=true`, every
projected candidate is residual-checked; otherwise validation is skipped and
residuals are `NaN`. This is the path used by `solve` and `solve_vectors`.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Contour center.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `rng=MersenneTwister(0)`: Random-number generator for probing.
* `cheb_config::ChebyshevConfig=solver.cheb_config`: Chebyshev configuration.

## Returns
* `ks::Vector{Complex{T}}`: Accepted complex roots.
* `residuals::Vector{T}`: Corresponding nonlinear residuals.
* `X::Matrix{Complex{T}}`: Corresponding layer density vectors.
"""
function _beyn_solve_core(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, rng=MersenneTwister(0), cheb_config::ChebyshevConfig=solver.cheb_config)
    T = _bim_numeric_type(solver); N = boundary_matrix_size(solver.kernel, pts)
    # Retain one window's full A_l while constructing its real-beta matrices.
    # Validate with those same coefficients, then release them with this window.
    coefficients = solver.use_half_contour && solver.auto_discard_spurious ? Array{ComplexF64}(undef, N, N, solver.taylor_degree+1) : nothing
    ks, X = _beyn_projected_solve(solver, pts, k0, dk; multithreaded, rng, cheb_config, coefficients)
    isempty(ks) && return Complex{T}[], T[], Matrix{Complex{T}}(undef, N, 0)
    !solver.auto_discard_spurious && return ks, fill(T(NaN), length(ks)), X
    if solver.use_half_contour
        values = [ks]; vectors = [X]
        keep, residuals = _beyn_imag_k_check(solver, values, vectors, [pts]; multithreaded, cheb_config, windows=([k0], [T(dk)/2]), coefficients)
        idx = keep[1]
        return values[1][idx], residuals[1], isempty(idx) ? Matrix{Complex{T}}(undef, N, 0) : vectors[1][:,idx]
    end
    residuals = Vector{T}(undef, length(ks)); keep = falses(length(ks)); y = Vector{Complex{T}}(undef, N)
    @inbounds for j in eachindex(ks)
        rj = _beyn_residual(solver, pts, ks[j], @view(X[:,j]), y; multithreaded); residuals[j] = rj; keep[j] = rj<solver.res_tol
    end
    idx = findall(keep)
    return ks[idx], residuals[idx], isempty(idx) ? Matrix{Complex{T}}(undef, N, 0) : X[:,idx]
end

#TODO Imag k check does not work for half contour, since it seems that it worked with full Green func kernel
# due to the im*J part of the Hankels (anyway now with cheap residual evals not that important)
# Still a cool accuracy proxy diagnostic for full contour.
# one can check for stadium geometry that at tipical ppw ≈ 10 we dont get anywhere close to machine ϵ since
# not C^2 (curvature jumps at the circle-line joins)

"""
    _beyn_imag_k_check(solver::BeynSolver, ks_all, X_all, all_pts;
        pad::Int=solver.imag_k_pad, group_size::Int=solver.imag_k_group_size,
        multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config,
        windows=nothing, block_size::Int=512, block_bytes::Int=512*1024^2)

Process projected roots in batches of at most `group_size`.
Full contours process decreasing `abs(imag(k))` and accept the remaining roots
once `pad` consecutive checks pass. Half contours check every root in window
order. With `windows`, full Hankel Taylor coefficients are shared across all
candidate vectors in each window using BLAS-3 products.

With `auto_discard_spurious=true`, discard failing checked candidates.
Half contours check every candidate and also reject nonfinite residuals.

## Arguments
* `solver::BeynSolver`: Beyn solver and filtering configuration.
* `ks_all::AbstractVector{<:AbstractVector{Complex{T}}}`: Projected roots per window.
* `X_all::AbstractVector{<:AbstractMatrix{Complex{T}}}`: Projected vector columns,
  with column j associated with `ks_all[i][j]` in window i.
* `all_pts::AbstractVector{<:BoundaryPoints}`: Boundary discretization per window.

## Keyword Arguments
* `pad::Int=solver.imag_k_pad`: Consecutive passing checks before full-contour
  early termination; unused for half contours.
* `group_size::Int=solver.imag_k_group_size`: Maximum root-processing batch size;
  half-contour Taylor products use all candidate vectors in a window.
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `cheb_config::ChebyshevConfig=solver.cheb_config`: Chebyshev residual configuration.
* `windows::Union{Nothing,Tuple}=nothing`: Original `(centers, radii)` per window;
  supplied centers enable full Hankel Taylor checks for half contours.
* `block_size::Int=512`: Maximum coefficient-block width for half-contour Taylor checks.
* `block_bytes::Int=512*1024^2`: Coefficient-buffer budget in bytes for those checks;
  forwarded to `_beyn_taylor_residuals`, separately from other workspaces.
* `coefficients::Union{Nothing,Array{ComplexF64,3}}=nothing`: Full Fredholm
  Taylor coefficients retained for a single window; skips their reassembly.

## Returns
* `idx_keep::Vector{Vector{Int}}`: Retained local root indices per window.
* `residuals::Vector{Vector{T}}`: Nonlinear residual norms aligned with retained
  roots; unevaluated entries remain `NaN`. A batch may evaluate beyond the
  full-contour stopping point.

Returns `(idx_keep, residuals)`, with `T=_bim_numeric_type(solver)`.
"""
function _beyn_imag_k_check(solver::BeynSolver, ks_all, X_all, all_pts; pad::Int=solver.imag_k_pad, group_size::Int=solver.imag_k_group_size, multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config, windows=nothing, block_size::Int=512, block_bytes::Int=512*1024^2, coefficients=nothing)
    T = _bim_numeric_type(solver); nw = length(ks_all); half = solver.use_half_contour
    kernel = half ? first(_beyn_half_symmetry_rotation_fix(solver.kernel)) : solver.kernel
    taylor = half && windows!==nothing; batched = taylor || (solver.use_chebyshev && solver.kernel isa Union{DoubleLayerPotentialSolver,CombinedFieldIntegralEquationSolver})
    computed = taylor ? falses(nw) : Bool[]
    keep = [trues(length(ks_all[i])) for i in 1:nw]; residuals = [fill(T(NaN), length(ks_all[i])) for i in 1:nw]; candidates = Tuple{Int,Int,T}[]
    @inbounds for i in 1:nw
        for j in eachindex(ks_all[i])
            push!(candidates, (i, j, abs(imag(ks_all[i][j]))))
        end
    end
    isempty(candidates) && return [Int[] for _ in 1:nw], residuals
    half || sort!(candidates; by=c -> c[3], rev=true)
    window_js = [Int[] for _ in 1:nw]; touched = Int[]; y = Complex{T}[]
    streak = 0; pos = 1; nc = length(candidates)
    while pos <= nc
        last = min(pos+group_size-1, nc); empty!(touched)
        @inbounds for q in pos:last
            i, j, _ = candidates[q]; isempty(window_js[i]) && push!(touched, i); push!(window_js[i], j)
        end
        @inbounds for i in touched
            js = window_js[i]; pts = all_pts[i]; resize!(y, size(X_all[i], 1))
            if taylor
                if !computed[i]
                    residuals[i] = _beyn_taylor_residuals(kernel, pts, Float64(real(windows[1][i])), ks_all[i], X_all[i], solver.taylor_degree; block_size, block_bytes, multithreaded, coefficients)
                    computed[i] = true
                end
            elseif batched
                zg = ComplexF64[ks_all[i][j] for j in js]
                Abufs = _construct_matrices_multi_k_cheb(kernel, pts, zg, cheb_config; multithreaded)
                @blas_multi_then_1 MAX_BLAS_THREADS for q in eachindex(js)
                    j = js[q]; mul!(y, Abufs[q], @view(X_all[i][:,j]))
                    residuals[i][j] = norm(y)
                end
            else
                for j in js
                    residuals[i][j] = _beyn_residual(solver, pts, ks_all[i][j], @view(X_all[i][:, j]), y; multithreaded)
                end
            end
        end
        stop = false
        @inbounds for q in pos:last
            i, j, _ = candidates[q]; rj = residuals[i][j]
            if rj >= solver.res_tol || (half && !isfinite(rj))
                solver.auto_discard_spurious && (keep[i][j] = false); streak = 0
            else
                streak += 1
                if !half && streak >= pad
                    stop = true; break
                end
            end
        end
        @inbounds for i in touched
            empty!(window_js[i])
        end
        stop && break
        pos = last+1
    end
    idx_keep = [findall(keep[i]) for i in 1:nw]
    return idx_keep, [residuals[i][idx_keep[i]] for i in 1:nw]
end

"""
    solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config)

Solve one circular Beyn contour with residual validation.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Contour center.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `cheb_config::ChebyshevConfig=solver.cheb_config`: Chebyshev configuration.

## Returns
* `ks::Vector{Complex{T}}`: Accepted complex eigenvalues.
* `residuals::Vector{T}`: Corresponding nonlinear residuals.
"""
function solve(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config)
    ks, residuals, _ = _beyn_solve_core(solver, pts, k0, dk; multithreaded, cheb_config)
    return ks, residuals
end

"""
    solve_vectors(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config)

Solve one circular Beyn contour with residual validation and return
source-layer densities in full mode or physical boundary functions in half mode.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `pts::BoundaryPoints`: Boundary discretization.
* `k0`: Contour center.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.
* `cheb_config::ChebyshevConfig=solver.cheb_config`: Chebyshev configuration.

## Returns
* `ks::Vector{Complex{T}}`: Accepted complex eigenvalues.
* `residuals::Vector{T}`: Corresponding nonlinear residuals.
* `X::Matrix{Complex{T}}`: Source-layer densities in full mode; physical boundary functions in half mode.
"""
function solve_vectors(solver::BeynSolver, pts::BoundaryPoints, k0, dk; multithreaded::Bool=true, cheb_config::ChebyshevConfig=solver.cheb_config)
    ks, residuals, X = _beyn_solve_core(solver, pts, k0, dk; multithreaded, cheb_config)
    if solver.use_half_contour
        weights = pts.ds
        if solver.kernel.symmetry!==nothing
            orbits = _fold_boundary(Float64, solver.kernel.billiard, length(pts), solver.kernel.symmetry, solver.kernel.sym_characters)
            weights = zeros(Float64, length(orbits.fundamental_indices))
            @inbounds for i in eachindex(pts.ds)
                weights[orbits.orbit_of[i]] += pts.ds[i]*abs2(orbits.phase[i])
            end
        end
        X ./= weights
    end
    return ks, residuals, X
end

"""
    solve_wavenumber(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}

Solve for the eigenvalue closest to the given wavenumber `k` within a circular Beyn contour.

## Arguments
* `solver::BeynSolver`: Beyn solver.
* `billiard::AbsBilliard`: Billiard instance.
* `k`: Target wavenumber.
* `dk`: Full contour diameter.

## Keyword Arguments
* `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
* `k::Complex{T}`: Eigenvalue closest to the target wavenumber, or `nothing` if the contour contains none.
* `residual::T`: Corresponding nonlinear residual, or `nothing` if the contour contains none.
"""
function solve_wavenumber(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k); ks, residuals = solve(solver, pts, k, dk; multithreaded)
    isempty(ks) && return nothing, nothing
    idx = findmin(abs.(ks .- k))[2]
    return ks[idx], residuals[idx]
end

function solve_spectrum(solver::BeynSolver, billiard::Bi, k, dk; multithreaded::Bool=true) where {Bi<:AbsBilliard}
    pts = evaluate_points(solver, billiard, k)
    return solve(solver, pts, k, dk; multithreaded)
end
