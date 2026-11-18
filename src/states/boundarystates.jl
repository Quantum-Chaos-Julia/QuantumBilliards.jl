"""
    boundary_states(states::AbstractVector{<:BasisEigenstate{K}}, billiard::AbsBilliard; ppw::Real=5.0, multithreaded::Bool=true) → Vector{BIMEigenstate{K,K}}

Convert basis eigenstates to boundary states that store the Rellich-normalized
normal derivative `u = ∂ₙψ` on the complete physical boundary and no layer
density.

## Description
States are grouped by `(objectid(basis), k_basis)`, so that states sharing a
basis evaluation share one boundary grid and one gradient-matrix evaluation.
For each group the grid is the `FourierNodes` boundary grid of
[`boundary_function`](@ref) with `max(round(Int, k*L*ppw/(2π)), 512)` points,
where `k` is the largest wavenumber of the group. The normal-derivative matrix
`G = nₓ ∂ₓB + n_y ∂_yB` is formed once and applied to all coefficient vectors
of the group with one matrix product, `U = G X`. Each column is regularized
with [`regularize!`](@ref), its Rellich norm is evaluated with
[`_rellich`](@ref) on the fundamental boundary, and `u` and the grid are
unfolded onto the full boundary with `apply_symmetries_to_boundary_function`
and `apply_symmetries_to_boundary_points`. Basis coefficients are real, so
`u` is real.

## Arguments
* `states`: Basis eigenstates, possibly from different bases or basis wavenumbers.
* `billiard`: Billiard the states were computed on.

## Keyword Arguments
* `ppw::Real=5.0`: Boundary points per wavelength of the boundary grid.
* `multithreaded::Bool=true`: Whether the gradient matrices are built multithreaded and the per-state post-processing runs threaded.

## Returns
* `Vector{BIMEigenstate{K,K}}`: One state per input state, in input order, with `vec = nothing`, `u` and `bnd_norm` the Rellich value of the fundamental-boundary `u`. States of one group share `pts`.
"""
function boundary_states(states::AbstractVector{BasisEigenstate{K,Tt,Ba}}, billiard::AbsBilliard; ppw::Real=5.0, multithreaded::Bool=true) where {K<:Real,Tt<:Real,Ba<:AbsBasis}
    out = Vector{BIMEigenstate{K,K}}(undef, length(states))
    groups = Dict{Tuple{UInt,K},Vector{Int}}()
    order = Tuple{UInt,K}[]
    for i in eachindex(states)
        key = (objectid(states[i].basis), states[i].k_basis)
        if haskey(groups, key)
            push!(groups[key], i)
        else
            groups[key] = [i]
            push!(order, key)
        end
    end
    boundary = get_boundary_curves_with_ignored(billiard)
    sampler = FourierNodes([2, 3, 5], [crv.length for crv in boundary])
    L = CompositeCurve(boundary).length
    for key in order
        _boundary_states_group!(out, states, groups[key], billiard, sampler, L, ppw, multithreaded)
    end
    return out
end

function _boundary_states_group!(out, states::AbstractVector{BasisEigenstate{K,Tt,Ba}}, idx::Vector{Int}, billiard::AbsBilliard, sampler, L::Real, ppw::Real, multithreaded::Bool) where {K<:Real,Tt<:Real,Ba<:AbsBasis}
    basis = states[first(idx)].basis
    k_basis = states[first(idx)].k_basis
    kmax = maximum(states[i].k for i in idx)
    N = _debroglie_grid_size(kmax, L, ppw)
    pts = boundary_coords(billiard, sampler, N)
    G, dY = gradient_matrices(basis, k_basis, pts.xy; multithreaded)
    nx = getindex.(pts.normal, 1)
    ny = getindex.(pts.normal, 2)
    @. G = nx * G + ny * dY
    X = Matrix{K}(undef, size(G, 2), length(idx))
    for (j, i) in enumerate(idx)
        X[:, j] = states[i].vec
    end
    U = G * X
    symmetries = basis.symmetries
    full_pts = isnothing(symmetries) ? pts : apply_symmetries_to_boundary_points(pts, symmetries, billiard)
    @use_threads multithreading=multithreaded for j in eachindex(idx)
        state = states[idx[j]]
        u = U[:, j]
        regularize!(u)
        bnd_norm = K(_rellich(pts, u, state.k))
        isnothing(symmetries) || (u = apply_symmetries_to_boundary_function(u, symmetries, basis.sym_characters))
        out[idx[j]] = BIMEigenstate(state.k, nothing, K(state.ten), full_pts; u, bnd_norm)
    end
    return nothing
end

"""
    _level_clusters(ks::AbstractVector{<:Real}, tol::Real) → Vector{Vector{Int}}

Group levels whose sorted neighbours differ by at most `tol`. Each cluster lists
indices into `ks` in increasing order of `ks`; clusters are ordered by wavenumber.
"""
function _level_clusters(ks::AbstractVector{<:Real}, tol::Real)
    clusters = Vector{Vector{Int}}()
    previous = 0
    for i in sortperm(ks)
        if previous != 0 && ks[i] - ks[previous] <= tol
            push!(clusters[end], i)
        else
            push!(clusters, [i])
        end
        previous = i
    end
    return clusters
end

# Indices of the levels grouped by identity of their boundary points; one group holding every level when no states are stored.
function _pts_groups(states::Nothing, n::Int)
    return [collect(1:n)]
end

function _pts_groups(states::AbstractVector{<:BIMEigenstate}, n::Int)
    groups = Dict{UInt,Vector{Int}}()
    order = UInt[]
    for i in 1:n
        key = objectid(states[i].pts)
        if haskey(groups, key)
            push!(groups[key], i)
        else
            groups[key] = [i]
            push!(order, key)
        end
    end
    return [groups[key] for key in order]
end

function _complex_state(state::BIMEigenstate{K,T}) where {K<:Number,T<:Real}
    vec = state.vec === nothing ? nothing : Vector{Complex{T}}(state.vec)
    u = state.u === nothing ? nothing : Vector{Complex{T}}(state.u)
    return BIMEigenstate{Complex{T},T}(state.k, state.k_im, vec, state.ten, state.pts, u, state.bnd_norm)
end

# Orthonormal `m` left near-null vectors of `A` from two steps of adjoint inverse iteration `Y ← A⁻ᴴY` with QR re-orthonormalisation,
# or `nothing` when the factorization is singular or the iterates are not finite.
function _adjoint_inverse_iteration(A::AbstractMatrix{C}, m::Int, rng::AbstractRNG; steps::Int=2) where {C<:Number}
    F = lu!(copy(A); check=false)
    F.info == 0 || return nothing
    Y = Matrix(qr!(randn(rng, C, size(A, 1), m)).Q)
    for _ in 1:steps
        ldiv!(adjoint(F), Y)
        all(isfinite, Y) || return nothing
        Y = Matrix(qr!(Y).Q)
    end
    return Y
end

function _left_residuals(A::AbstractMatrix{C}, Y::AbstractMatrix{C}) where {C<:Number}
    R = adjoint(A) * Y
    return [norm(view(R, :, j)) for j in axes(R, 2)]
end

# Left near-null vectors of `A(k)` as columns: adjoint inverse iteration, with the `KrylovKit.svdsolve` left singular vectors as fallback when the residual `‖Aᴴy‖` exceeds `res_tol`.
function _left_null_vectors(A::AbstractMatrix{C}, m::Int, k::Real, res_tol::Real, rng::AbstractRNG) where {C<:Number}
    Y = _adjoint_inverse_iteration(A, m, rng)
    Y !== nothing && all(<=(res_tol), _left_residuals(A, Y)) && return Y
    vals, lvecs, _, _ = KrylovKit.svdsolve(A, m, :SR)
    if length(lvecs) >= m
        return reduce(hcat, lvecs[1:m])
    end
    Y === nothing && error("left solve failed at k = $k: $(length(lvecs)) of $m singular vectors converged")
    @warn "left solve at k = $k: singular-vector fallback converged $(length(lvecs)) of $m vectors; keeping inverse-iteration vectors with residual above res_tol = $res_tol"
    return Y
end

# Left null vectors (columns, reduced size of the kernel matrix) of the levels `ks` on the boundary points `pts`, by adjoint inverse
# iteration on `A(k)`; levels closer than `cluster_tol` share one matrix and one block iteration.
function _posthoc_left_vectors(kernel::AbsBIMSolver, pts::BoundaryPoints{T}, ks::AbstractVector{<:Real}; multithreaded::Bool=true, cluster_tol::Real=1e-5, res_tol::Real=1e-6, rng::AbstractRNG=MersenneTwister(0)) where {T<:Real}
    geom = _bim_geometry(kernel, pts)
    Y = nothing
    for cluster in _level_clusters(ks, cluster_tol)
        kc = sum(ks[cluster]) / length(cluster)
        A = construct_matrices(kernel, pts, T(kc), geom; multithreaded)
        Yc = _left_null_vectors(A, length(cluster), kc, res_tol, rng)
        Y === nothing && (Y = zeros(ComplexF64, size(Yc, 1), length(ks)))
        Y[:, cluster] .= Yc
    end
    return Y === nothing ? zeros(ComplexF64, 0, 0) : Y
end

# `u`-state of one level from a left null vector `y` of the (possibly symmetry-reduced) kernel matrix: `u = y ./ ds` expanded with the orbit phases, rotated so that its largest entry is real and positive, and Rellich-normalized.
function _boundary_state(kernel::AbsBIMSolver, billiard::AbsBilliard, pts::BoundaryPoints{T}, k::Number, ten::Real, y::AbstractVector) where {T<:Real}
    u = Vector{Complex{T}}(_bim_normal_derivative(kernel, pts, y, billiard))
    p = findmax(abs, u)[2]
    u .*= conj(u[p]) / abs(u[p])
    bnd_norm = T(_rellich(pts, u, T(real(k))))
    u ./= sqrt(bnd_norm)
    return BIMEigenstate{Complex{T},T}(T(real(k)), T(imag(k)), nothing, T(ten), pts, u, bnd_norm)
end

# `u`-states of the levels `ks` of one window from the columns of the left vectors `Y`; the vectors are not retained.
function _boundary_states(kernel::AbsBIMSolver, billiard::AbsBilliard, ks::AbstractVector{<:Number}, ts::AbstractVector{<:Real}, Y::AbstractMatrix, pts::BoundaryPoints{T}) where {T<:Real}
    states = Vector{BIMEigenstate{Complex{T},T}}(undef, length(ks))
    for j in eachindex(ks)
        states[j] = _boundary_state(kernel, billiard, pts, ks[j], ts[j], view(Y, :, j))
    end
    return states
end

"""
    compute_boundary_functions(data::SpectralData; multithreaded::Bool=true, show_progress::Bool=false, cluster_tol::Real=1e-5, res_tol::Real=1e-6, rng::AbstractRNG=MersenneTwister(0)) → SpectralData

Compute the Rellich-normalized boundary normal derivative `u = ∂ₙψ` of every
level of a boundary-integral spectrum and return a copy of `data` whose states
store it.

## Description
This is the post-hoc route for spectra computed with a `states` mode other than
`:boundary`, valid for every boundary-integral kernel (DLP, CFIE, composites
with holes), grading and symmetry sector. For each level `A(kⱼ)` is assembled on
the boundary points of the level (the stored `pts` of its state, or
`evaluate_points(kernel, billiard, k)` when `data.states === nothing`) and
factorized once. Two steps of adjoint inverse iteration `y ← A⁻ᴴy` from a random
start give the left null vector. The residual `‖A(kⱼ)ᴴy‖` is checked against
`res_tol`; on failure the left singular vector of `KrylovKit.svdsolve` is used
instead. Then `u = y ./ ds` is expanded with the orbit phases of the symmetry
sector ([`_bim_normal_derivative`](@ref)), rotated by a global phase so that its
largest entry is real and positive, and normalized with the Rellich identity.

Levels whose wavenumbers differ by at most `cluster_tol` share one matrix, one
LU factorization and one block iteration with QR re-orthonormalisation, so
degenerate levels receive orthonormal vectors spanning the left eigenspace; the
matrix of a cluster is assembled at its mean wavenumber. States that already
store `u` are kept unchanged. Levels are processed sequentially; matrix
assembly and BLAS use the available threads.

## Arguments
* `data::SpectralData`: Spectrum of a boundary-integral solver (sweep kernel, Beyn, CORK or ExpandedBIM).

## Keyword Arguments
* `multithreaded::Bool=true`: Multithreaded matrix assembly and BLAS.
* `show_progress::Bool=false`: Display a progress bar over the clusters.
* `cluster_tol::Real=1e-5`: Maximum wavenumber gap inside a cluster of degenerate levels.
* `res_tol::Real=1e-6`: Largest accepted residual `‖A(k)ᴴy‖` of inverse iteration.
* `rng::AbstractRNG=MersenneTwister(0)`: Source of the random start vectors.

## Returns
* `SpectralData`: Same levels and solver as `data`, with `BIMEigenstate{Complex{T},T}` states holding `u`, `bnd_norm` and the original layer density when present.
"""
function compute_boundary_functions(data::SpectralData{T,Bi,So}; multithreaded::Bool=true, show_progress::Bool=false, cluster_tol::Real=1e-5, res_tol::Real=1e-6, rng::AbstractRNG=MersenneTwister(0)) where {T<:Real,Bi<:AbsBilliard,So<:AbsBIMSolver}
    kernel = _bim_kernel(data.solver)
    billiard = data.billiard
    states = data.states
    n = length(data)
    out = Vector{BIMEigenstate{Complex{T},T}}(undef, n)
    jobs = Vector{Vector{Int}}()
    for idx in _pts_groups(states, n)
        pending = Int[]
        for i in idx
            if states !== nothing && states[i].u !== nothing
                out[i] = _complex_state(states[i])
            else
                push!(pending, i)
            end
        end
        for cluster in _level_clusters(data.k[pending], cluster_tol)
            push!(jobs, pending[cluster])
        end
    end
    nblas = multithreaded ? MAX_BLAS_THREADS : 1
    geom_key = UInt(0)
    geom = nothing
    @maybe_showprogress show_progress for members in jobs
        kc = sum(data.k[members]) / length(members)
        pts = states === nothing ? evaluate_points(kernel, billiard, kc) : states[first(members)].pts
        key = objectid(pts)
        if states === nothing || key != geom_key
            geom = _bim_geometry(kernel, pts)
            geom_key = key
        end
        old_blas = BLAS.get_num_threads()
        BLAS.set_num_threads(nblas)
        local Y
        try
            A = construct_matrices(kernel, pts, T(kc), geom; multithreaded)
            Y = _left_null_vectors(A, length(members), kc, res_tol, rng)
        finally
            BLAS.set_num_threads(old_blas)
        end
        for (j, i) in enumerate(members)
            s = _boundary_state(kernel, billiard, pts, complex(data.k[i], data.k_im[i]), data.ten[i], view(Y, :, j))
            vec = states === nothing || states[i].vec === nothing ? nothing : Vector{Complex{T}}(states[i].vec)
            out[i] = BIMEigenstate{Complex{T},T}(s.k, s.k_im, vec, s.ten, pts, s.u, s.bnd_norm)
        end
    end
    return SpectralData{T,Bi,So,typeof(out)}(billiard, data.solver, data.k, data.k_im, data.ten, data.control, out)
end
