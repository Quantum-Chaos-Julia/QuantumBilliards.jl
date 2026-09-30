
"""
    regularize!(u::AbstractVector) → Nothing

Removes `NaN` values from a boundary function `u` in place, mutating `u`
directly.

## Description
The boundary normal derivative computed by [`boundary_function`](@ref) can be
`NaN` at points where the boundary normal is not smooth (e.g. at corners of
the domain). Each such entry is replaced by the average of its two nearest
neighbours along the arc-length parametrization of the boundary. Since the
boundary is a closed curve, the first sample's neighbours are its next entry
and the last entry of `u`.

## Arguments
* `u`: The boundary function values to regularize, modified in place.

## Returns
*  `nothing` : `u` is modified in place; no meaningful value is returned.
"""
function regularize!(u)
    idx = findall(isnan, u)
    for i in idx
        if i != 1
            u[i] = (u[i+1] + u[i-1])/2.0
        else
            u[i] = (u[i+1] + u[end])/2.0
        end
    end
end

"""
    _rellich(pts::BoundaryPoints{T}, u::AbstractVector{N}, k::T) where {N<:Number,T<:Real} → norm::T

Evaluates the Rellich-identity normalization of the boundary function `u`
(the normal derivative of an eigenstate) at wavenumber `k`.

## Description
For a Dirichlet eigenstate normalized to unit interior norm, Rellich's
identity relates that normalization to a boundary integral of the normal
derivative:

```math
\\frac{1}{2k^2}\\int_{\\partial\\Omega} (\\mathbf{n}\\cdot\\mathbf{x})\\, |\\partial_n\\psi|^2\\, ds = 1.
```

This function evaluates the discretized left-hand side, providing a cheap
normalization consistency check that is computed alongside
[`boundary_function`](@ref).

## Arguments
* `pts`: The boundary points at which `u` was evaluated, providing the
  positions `xy`, outward unit normals `normal` and arc-length weights `ds`.
* `u`: The boundary function (normal derivative) values at `pts`.
* `k`: The wavenumber of the eigenstate.

## Returns
*  `norm` : The Rellich-normalized boundary integral, expected to be close to
   `1` for a correctly normalized eigenstate.
"""
function _rellich(pts::BoundaryPoints{T},u::AbstractVector{N},k::T) where {N<:Number,T<:Real}
    acc=zero(T)
    @inbounds @simd for i in eachindex(u)
        n=pts.normal[i]
        xy=pts.xy[i]
        w=(n[1]*xy[1]+n[2]*xy[2])*pts.ds[i]
        acc+=w*abs2(u[i])
    end
    return acc/(2*k^2)
end

"""
    boundary_function(state::S; b::Real = 5.0, multithreaded::Bool = true) where {S<:AbsState} → (u::Vector, s::Vector, norm::Real)

Computes the normal-derivative boundary function \$u(s) = \\partial_n\\psi(s)\$ of
an eigenstate along its boundary, together with the arc-length coordinate `s`
and a Rellich-identity normalization check `norm`. The default `b = 5.0`
gives roughly `5` boundary sample points per de Broglie wavelength.

## Description
The boundary is discretized with `FourierNodes` quadrature on
`N = max(round(Int, k*L*b/(2\\pi)), 512)` points, where `L` is the total
boundary length. The gradient matrices of the basis are evaluated at these
points and contracted with the outward unit normal to give

```math
\\partial_n\\psi(s) = n_x(s)\\,\\partial_x\\psi(s) + n_y(s)\\,\\partial_y\\psi(s).
```

[`regularize!`](@ref) then removes any `NaN` artifacts arising at non-smooth
points of the boundary, and [`_rellich`](@ref) evaluates the Rellich-identity
normalization of the resulting function. If the basis carries symmetries,
the boundary points and function are subsequently unfolded onto the full
boundary via `apply_symmetries_to_boundary_points` and
`apply_symmetries_to_boundary_function`.

## Arguments
* `state`: The eigenstate for which the boundary function is computed.

## Keyword arguments
*  `b::Real = 5.0` : Oversampling factor controlling the boundary point
   density; the boundary is sampled at `max(round(Int, k*L*b/(2\\pi)), 512)`
   points.
*  `multithreaded::Bool = true` : Whether the gradient matrix construction is
   multithreaded.

## Returns
*  `u` : The normal-derivative boundary function values (symmetry-unfolded if
   applicable).
*  `s` : The arc-length coordinates corresponding to `u`.
*  `norm` : The Rellich-identity normalization of the boundary function
   computed by [`_rellich`](@ref), before any symmetry unfolding; should be
   close to `1` for a correctly normalized eigenstate.
"""
function _basis_boundary_function_pts(state::S, billiard::AbsBilliard; b=5.0, multithreaded = true) where {S<:AbsState}
    let vec = state.vec, k = state.k, k_basis = state.k_basis, new_basis = state.basis
        type = eltype(vec)
        boundary = get_boundary_curves_with_ignored(billiard)
        crv_lengths = [crv.length for crv in boundary]
        sampler = FourierNodes([2,3,5],crv_lengths) 
        L = CompositeCurve(boundary).length
        N = _debroglie_grid_size(k, L, b)
        pts = boundary_coords(billiard, sampler, N)
        dX, dY = gradient_matrices(new_basis, k_basis, pts.xy; multithreaded)
        nx = getindex.(pts.normal,1)
        ny = getindex.(pts.normal,2)
        dX = nx .* dX 
        dY = ny .* dY
        U::Array{type,2} = dX .+ dY
        u::Vector{type} = U * vec
        regularize!(u)
        #compute the boundary norm
        norm = _rellich(pts, u, k)
        if isnothing(new_basis.symmetries) == false
            pts = apply_symmetries_to_boundary_points(pts, new_basis.symmetries, billiard)
            u = apply_symmetries_to_boundary_function(u, new_basis.symmetries, new_basis.sym_qnumbers)
        end
        return u, pts, norm
    end
end

function boundary_function(state::S, billiard::AbsBilliard; b=5.0, multithreaded = true) where {S<:AbsState}
    type = eltype(state.vec)
    u, pts, norm = _basis_boundary_function_pts(state, billiard; b, multithreaded)
    #println(norm)
    return u, pts.s::Vector{type}, norm
end

"""
    boundary_function(state::BIMEigenstate; b::Real = 5.0, multithreaded::Bool = true) → (u::Vector, s::Vector, norm::Real)

Returns the Rellich-normalized boundary normal derivative \$u(s) = \\partial_n\\psi(s)\$
of a [`BIMEigenstate`](@ref), computed with any [`SweepBIMSolver`](@ref).

## Description
Unlike [`boundary_function(state::S) where {S<:AbsState}`](@ref), `state.vec`
(the *primal* density from [`solve_vect`](@ref)) is not directly usable as
`∂ₙψ`. Since [`compute_eigenstate`](@ref) already computes the physical
boundary normal derivative via [`solve_state`](@ref)/[`_bim_normal_derivative`](@ref)
(from the *same* Krylov solve as `vec`/`ten`, at no extra cost), this is now a
plain read of the precomputed `state.u`/`state.pts`/`state.bnd_norm` fields —
no matrix is assembled and no solve happens here.

The `b`/`multithreaded` keywords are accepted only for interface
compatibility with the generic `S<:AbsState` methods
([`momentum_function`](@ref), [`husimi_function`](@ref)); both are unused
because a BIM boundary discretization/density is fixed at `compute_eigenstate`
time, not resampled or re-solved per call.

## Returns
*  `u` : The Rellich-normalized physical boundary normal derivative `∂ₙψ`, on the complete physical boundary.
*  `s` : The arc-length coordinates corresponding to `u`. Not, in general, uniformly spaced (a `GlobalCornerGrading` solver clusters nodes near corners) — see [`husimi_function(state::BIMEigenstate)`](@ref).
*  `norm` : The Rellich-identity value of the raw (pre-rescaling) density used to normalize `u` (`state.bnd_norm`); kept for interface parity with the basis-solver method, not itself expected to be close to `1`.
"""
function boundary_function(state::BIMEigenstate; b=5.0, multithreaded=true)
    return state.u, state.pts.s, state.bnd_norm
end

# Disambiguates `boundary_function(bim_state, billiard)` from the generic
# `S<:AbsState` method above, which would otherwise silently match and fail
# on fields (`k_basis`, `basis`) BIMEigenstate doesn't have. No `solver` is
# available here to recompute a missing `state.u`, so this simply forwards
# to the 1-arg read of the precomputed fields.
boundary_function(state::BIMEigenstate, billiard::AbsBilliard; b=5.0, multithreaded=true) = boundary_function(state; b, multithreaded)

"""
    boundary_function(state::BIMEigenstate, billiard::AbsBilliard, solver::AbsBIMSolver; b::Real = 5.0, multithreaded::Bool = true) → (u::Vector, s::Vector, norm::Real)

Returns the Rellich-normalized boundary normal derivative of a
[`BIMEigenstate`](@ref), recomputing it via [`solve_state`](@ref) when the
state's precomputed `u` is unavailable (e.g. states produced by an
accelerated BIM solver that only retained a layer density, see
[`compute_eigenstate`](@ref)). When `state.u !== nothing`, this is identical
to [`boundary_function(state::BIMEigenstate)`](@ref).

## Keyword arguments
*  `b::Real = 5.0` : Unused when `state.u !== nothing`; accepted for interface compatibility.
*  `multithreaded::Bool = true` : Unused when `state.u !== nothing`; accepted for interface compatibility.
"""
function boundary_function(state::BIMEigenstate, billiard::AbsBilliard, solver::AbsBIMSolver; b=5.0, multithreaded=true)
    u = state.u; norm = state.bnd_norm
    if u === nothing
        _, _, u, norm = solve_state(_bim_kernel(solver), state.pts, state.k, billiard)
    end
    return u, state.pts.s, norm
end

"""
    momentum_function(u::AbstractVector, s::AbstractVector) → (power::Vector, ks::Vector)

Computes the discrete momentum-space representation of a boundary function
`u(s)` sampled at equally spaced arc-length points `s`, returning the
one-sided power spectral density and the corresponding angular wavenumbers.

## Description
The real FFT of `u` is taken with `rfft`, and the associated frequencies from
`rfftfreq` are rescaled to angular wavenumbers via \$k_s = 2\\pi f\$. Each
coefficient is normalized by \$N^2\$ (with `N = length(u)`) and further
divided by the wavenumber spacing \$\\Delta k = 2\\pi/L\$ (with
`L = s[end]` the total arc length), converting the discrete Fourier
coefficients into a power *spectral density* in `k`. All bins except the DC
term (and the Nyquist term, when `N` is even) are doubled to account for the
folded negative-frequency content, giving a one-sided power spectral density
whose integral over `ks` recovers the mean-square value of `u`:

```math
\\int \\mathrm{power}(k)\\, dk \\;\\approx\\; \\sum_j \\mathrm{power}_j\\, \\Delta k \\;=\\; \\frac{1}{L}\\int_0^L |u(s)|^2\\, ds \\;\\approx\\; \\frac{1}{N}\\sum_i |u_i|^2.
```

## Arguments
* `u`: The boundary function values, typically obtained from
  [`boundary_function`](@ref), sampled at equally spaced arc-length points.
* `s`: The arc-length coordinates of `u`; only the sample spacing
  `diff(s)[1]` (for the sampling rate) and `s[end]` (for the total arc
  length `L`) are used.

## Returns
*  `power` : The one-sided power spectral density of `u` as a function of
   `ks`, normalized so that `sum(power) * (ks[2] - ks[1]) ≈ mean(abs2, u)`.
*  `ks` : The angular wavenumbers corresponding to `power`.
"""
function momentum_function(u,s)
    N = length(u)
    fu = rfft(u)
    power = abs2.(fu) ./ N^2 ./ (2*pi/s[end]) 
    power[2:(iseven(N) ? end-1 : end)] .*= 2
    sr = 1.0/diff(s)[1]
    ks = rfftfreq(N,sr).*(2*pi)
    return power, ks
end

"""
    momentum_function(state::S; b::Real = 5.0, multithreaded::Bool = true) where {S<:AbsState} → (power::Vector, ks::Vector)

Computes the momentum-space representation of an eigenstate's boundary
function directly from `state`, by combining [`boundary_function`](@ref) and
[`momentum_function(u, s)`](@ref momentum_function).

## Arguments
* `state`: The eigenstate for which the boundary momentum function is
  computed.

## Keyword arguments
*  `b::Real = 5.0` : Oversampling factor passed to [`boundary_function`](@ref)
   controlling the boundary point density.
*  `multithreaded::Bool = true` : Whether the underlying gradient matrix
   construction is multithreaded.

## Returns
*  `power` : The normalized power spectrum of the boundary function.
*  `ks` : The angular wavenumbers corresponding to `power`.
"""
function momentum_function(state::S, billiard::AbsBilliard; b=5.0, multithreaded = true) where {S<:AbsState}
    u, s, norm = boundary_function(state, billiard; b, multithreaded)
    return momentum_function(u,s)
end

"""
    momentum_function(u::AbstractVector, s::AbstractVector, ds::AbstractVector; rtol::Real = 100*eps(...)) → (power::Vector, ks::Vector)

Computes the one-sided boundary momentum distribution `|c_m|²` of a boundary
function `u(s)` from its physical arc-length coordinates `s` and quadrature
weights `ds`, without assuming `s` is uniformly spaced.

## Description
If `s` is (to within `rtol`) uniformly spaced, the fast one-sided real FFT
path [`momentum_function(u, s)`](@ref momentum_function) is used (normalized
identically). Otherwise, each coefficient is evaluated directly by physical
quadrature,

```math
c_m = \\frac{1}{L}\\int_0^L u(s)\\, e^{-ik_m s}\\, ds \\approx \\frac{1}{L}\\sum_j u_j\\, e^{-ik_m s_j}\\, ds_j,
```

with `L = sum(ds)` and `k_m = 2\\pi m/L`, `m = 0,\\dots,\\lfloor N/2\\rfloor`.
This is needed for boundary-integral-method (BIM) boundary functions, whose
Kress-graded discretization is not, in general, uniformly spaced in arc
length (see [`GlobalCornerGrading`](@ref)).

## Arguments
* `u`: The boundary function values.
* `s`: The physical arc-length coordinates of `u`.
* `ds`: The physical quadrature weights corresponding to `u`.

## Keyword arguments
*  `rtol::Real = 100*eps(T)` : Relative tolerance used to detect uniform arc-length spacing.

## Returns
*  `power` : The one-sided boundary momentum weight `|c_m|²`.
*  `ks` : The angular wavenumbers corresponding to `power`.
"""
function momentum_function(u::AbstractVector{U}, s::AbstractVector{T}, ds::AbstractVector{T}; rtol::Real=100*eps(T)) where {U<:Number,T<:Real}
    N = length(u)
    N == length(s) == length(ds) || throw(DimensionMismatch("u, s and ds must have equal length"))
    Δs = s[2]-s[1]
    uniform = all(isapprox(s[i+1]-s[i], Δs; rtol=rtol, atol=eps(T)*max(one(T),abs(Δs))) for i in 2:N-1)
    if uniform && U<:Real
        return momentum_function(u, s)
    end
    L = sum(ds)
    M = div(N,2)
    ks = T.(2*pi/L.*(0:M))
    power = Vector{T}(undef, M+1)
    @fastmath @inbounds for m in 0:M
        km = ks[m+1]
        acc = zero(Complex{T})
        @simd for j in eachindex(u)
            acc += u[j]*cis(-km*s[j])*ds[j]
        end
        power[m+1] = abs2(acc/L)
    end
    if U<:Real
        lastdouble = iseven(N) ? M : M+1
        lastdouble >= 2 && (power[2:lastdouble] .*= 2)
    end
    return power, ks
end

"""
    momentum_function(state::BIMEigenstate{K,T,S,Bi}; b::Real = 5.0, multithreaded::Bool = true) where {K,T,S<:SweepBIMSolver,Bi} → (power::Vector, ks::Vector)

Computes the momentum-space representation of a [`BIMEigenstate`](@ref)
computed with any [`SweepBIMSolver`](@ref), via the non-uniform-arc-length-
aware [`momentum_function(u, s, ds)`](@ref momentum_function) applied
directly to the precomputed `state.u`/`state.pts`, since the BIM boundary
discretization is not, in general, uniformly spaced in arc length.

## Keyword arguments
*  `b::Real = 5.0` : Unused, accepted for interface compatibility (see [`boundary_function(state::BIMEigenstate)`](@ref)).
*  `multithreaded::Bool = true` : Unused, accepted for interface compatibility; `state.u` is already computed by [`compute_eigenstate`](@ref).
"""
function momentum_function(state::BIMEigenstate; b=5.0, multithreaded=true)
    return momentum_function(state.u, state.pts.s, state.pts.ds)
end

# Disambiguates `momentum_function(bim_state, billiard)` from the generic
# `S<:AbsState` method below (which would call the uniform-arclength-only
# `momentum_function(u,s)` FFT path, incorrect for BIM's generally
# non-uniform arc-length grid). No `solver` is available here to recompute a
# missing `state.u`, so this forwards to the 1-arg non-uniform-arclength-aware form.
momentum_function(state::BIMEigenstate, billiard::AbsBilliard; b=5.0, multithreaded=true) = momentum_function(state; b, multithreaded)

"""
    momentum_function(state::BIMEigenstate, billiard::AbsBilliard, solver::AbsBIMSolver; b::Real = 5.0, multithreaded::Bool = true) → (power::Vector, ks::Vector)

Computes the momentum-space representation of a [`BIMEigenstate`](@ref) via
[`boundary_function(state, billiard, solver)`](@ref), recomputing `u` when
unavailable on `state` (see that method), then applying the non-uniform-
arc-length-aware [`momentum_function(u, s, ds)`](@ref momentum_function).
"""
function momentum_function(state::BIMEigenstate, billiard::AbsBilliard, solver::AbsBIMSolver; b=5.0, multithreaded=true)
    u, s, _ = boundary_function(state, billiard, solver; b, multithreaded)
    return momentum_function(u, s, state.pts.ds)
end

################################################################################
# SpectralData API
################################################################################

_boundary_function_state(state::BIMEigenstate, billiard::AbsBilliard, solver; kwargs...) = boundary_function(state, billiard, solver; kwargs...)
_boundary_function_state(state::BasisEigenstate, billiard::AbsBilliard, solver; kwargs...) = boundary_function(state, billiard; kwargs...)
_momentum_function_state(state::BIMEigenstate, billiard::AbsBilliard, solver; kwargs...) = momentum_function(state, billiard, solver; kwargs...)
_momentum_function_state(state::BasisEigenstate, billiard::AbsBilliard, solver; kwargs...) = momentum_function(state, billiard; kwargs...)

"""
    boundary_function(data::SpectralData, i::Integer; kwargs...) → (u, s, norm)

Compute the boundary normal derivative of the `i`-th eigenstate stored in `data`.
"""
function boundary_function(data::SpectralData, i::Integer; kwargs...)
    states = _require_states(data)
    return _boundary_function_state(states[i], data.billiard, data.solver; kwargs...)
end

"""
    boundary_function(data::SpectralData; kwargs...) → (us, ss, norms)

Compute the boundary normal derivative of every eigenstate stored in `data`,
returning a vector of `(u, s, norm)` results.
"""
function boundary_function(data::SpectralData; kwargs...)
    states = _require_states(data)
    return [boundary_function(data, i; kwargs...) for i in eachindex(states)]
end

"""
    momentum_function(data::SpectralData, i::Integer; kwargs...) → (power, ks)

Compute the momentum-space representation of the `i`-th eigenstate stored in `data`.
"""
function momentum_function(data::SpectralData, i::Integer; kwargs...)
    states = _require_states(data)
    return _momentum_function_state(states[i], data.billiard, data.solver; kwargs...)
end

"""
    momentum_function(data::SpectralData; kwargs...) → Vector{Tuple}

Compute the momentum-space representation of every eigenstate stored in `data`.
"""
function momentum_function(data::SpectralData; kwargs...)
    states = _require_states(data)
    return [momentum_function(data, i; kwargs...) for i in eachindex(states)]
end
