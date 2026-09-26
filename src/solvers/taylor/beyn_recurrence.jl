# Full-boundary Taylor construction.
# For every boundary pair (i,j), _fredholm_taylor! generates the normalized
# coefficients a_l = A_ij^(l)(k0) / l!,    l = 0,...,p.
# The coefficients are consumed immediately and evaluated at all Beyn contour
# offsets δ_q = R exp(2πim(q-1)/nq) by Horner's rule. 
function _construct_matrices_multi_k_taylor_full(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, nq::Int, p::Int; multithreaded::Bool=true)::Vector{Matrix{ComplexF64}}
    solver.symmetry === nothing || throw(ArgumentError("full Taylor construction requires solver.symmetry === nothing"))
    cache = _taylor_cache(solver, pts)
    N::Int = cache.N
    δ::Vector{ComplexF64} = ComplexF64.(R .* cis.(2π .* (0:nq-1) ./ nq))
    Tbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, N, N) for _ = 1:nq]
    work::Vector{TaylorWorkspace} = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()]
    @use_threads multithreading=multithreaded for j = 1:N
        w::TaylorWorkspace = work[Threads.threadid()]
        @inbounds for i = 1:N
            a::Vector{ComplexF64} = _fredholm_taylor!(cache, w, k0, p, i, j)
            for q = 1:nq
                v::ComplexF64 = a[p+1]
                for l = p:-1:1
                    v = v * δ[q] + a[l]
                end
                Tbufs[q][i,j] = v
            end
        end
    end
    return Tbufs
end

# Symmetry-reduced Taylor construction.
# For reduced indices a,b, fold the Taylor coefficients over the source orbit,
# Ã_ab,l = δ_l0 δ_ab - Σ_{j∈O_b} χ_j K_iₐj,l.
# Source images are processed before target rows to preserve cache locality.
# The folded coefficients are accumulated in a thread-local (p+1)×m buffer
# and evaluated at all contour nodes by Horner's rule.
function _construct_matrices_multi_k_taylor_reduced(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, nq::Int, p::Int; multithreaded::Bool=true)::Vector{Matrix{ComplexF64}}
    solver.symmetry === nothing && throw(ArgumentError("reduced Taylor construction requires an active symmetry"))
    cache = _taylor_cache(solver, pts)
    orbits = solver isa CompositeBIMSolver ? _composite_symmetry_orbits(Float64, solver, pts) : _fold_boundary(Float64, solver.billiard, length(pts.xy), solver.symmetry, solver.character)
    m::Int = fundamental_size(orbits)
    fund = orbits.fundamental_indices
    orbit_of = orbits.orbit_of
    phase = orbits.phase
    images::Vector{Vector{Int}} = [Int[] for _ = 1:m]
    @inbounds for j = eachindex(orbit_of)
        push!(images[orbit_of[j]], j)
    end
    δ::Vector{ComplexF64} = ComplexF64.(R .* cis.(2π .* (0:nq-1) ./ nq))
    Tbufs::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, m, m) for _ = 1:nq]
    work::Vector{TaylorWorkspace} = [TaylorWorkspace(p) for _ = 1:Threads.maxthreadid()]
    coeff::Vector{Matrix{ComplexF64}} = [Matrix{ComplexF64}(undef, p+1, m) for _ = 1:Threads.maxthreadid()]
    @use_threads multithreading=multithreaded for bcol = 1:m
        tid::Int = Threads.threadid()
        w::TaylorWorkspace = work[tid]
        C::Matrix{ComplexF64} = coeff[tid]
        imgs::Vector{Int} = images[bcol]
        gj::Int = imgs[1]
        χ = phase[gj]
        @inbounds for arow = 1:m
            gi::Int = fund[arow]
            a::Vector{ComplexF64} = _kernel_taylor!(cache, w, k0, p, gi, gj)
            @simd for l = 1:p+1
                C[l,arow] = -χ * a[l]
            end
        end
        @inbounds for ii = 2:length(imgs)
            gj = imgs[ii]
            χ = phase[gj]
            for arow = 1:m
                gi = fund[arow]
                a = _kernel_taylor!(cache, w, k0, p, gi, gj)
                @simd for l = 1:p+1
                    C[l,arow] -= χ * a[l]
                end
            end
        end
        @inbounds for arow = 1:m
            arow == bcol && (C[1,arow] += 1.0)
            for q = 1:nq
                v::ComplexF64 = C[p+1,arow]
                for l = p:-1:1
                    v = v * δ[q] + C[l,arow]
                end
                Tbufs[q][arow,bcol] = v
            end
        end
    end
    return Tbufs
end

"""
    _construct_matrices_multi_k_taylor(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, nq::Int, p::Int; multithreaded::Bool=true)::Vector{Matrix{ComplexF64}}

Construct the Fredholm matrices on a circular Beyn contour using an analytic
Taylor expansion about `k0`,

    A(k0 + δ) ≈ Σₗ₌₀ᵖ Aₗ δˡ,    Aₗ = A⁽ˡ⁾(k0)/l!.

The coefficients are evaluated at the `nq` contour points
`δ_q = R exp(2πim(q-1)/nq)` by Horner's rule. The optimized full-boundary or
symmetry-reduced construction is selected automatically.

## Arguments
- `solver::SweepBIMSolver`: BIM solver defining the Fredholm operator.
- `pts::BoundaryPoints{Float64}`: Full-boundary discretization.
- `k0::Float64`: Center of the circular contour.
- `R::Float64`: Contour radius.
- `nq::Int`: Number of contour quadrature points.
- `p::Int`: Taylor expansion degree.
- `multithreaded::Bool=true`: Enable multithreaded matrix construction.

## Returns
- `::Vector{Matrix{ComplexF64}}`: Fredholm matrices evaluated at the `nq` contour points.
"""
function _construct_matrices_multi_k_taylor(solver::SweepBIMSolver, pts::BoundaryPoints{Float64}, k0::Float64, R::Float64, nq::Int, p::Int; multithreaded::Bool=true)::Vector{Matrix{ComplexF64}}
    if solver.symmetry === nothing
        return _construct_matrices_multi_k_taylor_full(solver, pts, k0, R, nq, p; multithreaded)
    end
    return _construct_matrices_multi_k_taylor_reduced(solver, pts, k0, R, nq, p; multithreaded)
end