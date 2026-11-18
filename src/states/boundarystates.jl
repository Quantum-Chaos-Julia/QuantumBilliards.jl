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
