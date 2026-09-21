"""
    ChebyshevConfig{T<:Real}

Configuration for Chebyshev interpolation of the radial special functions used
by accelerated boundary-integral solvers.

`ChebyshevConfig` controls the piecewise-Chebyshev approximations of the Hankel
and Bessel-J functions appearing in BIM kernel evaluations. Hankel and Bessel-J
functions use separate panelizations because their approximation requirements
can differ substantially over the radial interval encountered by a spectral
sweep. Bessel-J are far easier to handle accurately with fewer panels and lower polynomial degrees.

The configuration is stored by [`AcceleratedBIMSolver`](@ref) implementations
such as [`BeynSolver`](@ref) and [`ExpandedBIMSolver`](@ref), allowing the same
interpolation parameters to be reused across repeated matrix evaluations.

The initial approximation uses `n_panels_h` and `n_panels_j` radial panels and
polynomial degrees `M_h` and `M_j`. When automatic tuning is enabled, the
approximation is validated at `sampling_points` radial locations against the
direct special-function evaluation. If the error exceeds `tol`, the panel
counts are multiplied by `grow_panels` and the polynomial degrees are increased
by `grow_M`, up to `max_iter` tuning iterations.

## Attributes
* `n_panels_h::Int`: Initial number of radial Chebyshev panels used for Hankel functions.
* `M_h::Int`: Chebyshev polynomial degree used on each Hankel panel.
* `n_panels_j::Int`: Initial number of radial Chebyshev panels used for Bessel-J functions.
* `M_j::Int`: Chebyshev polynomial degree used on each Bessel-J panel.
* `tol::T`: Target absolute interpolation error used by automatic parameter tuning.
* `max_iter::Int`: Maximum number of automatic tuning iterations.
* `sampling_points::Int`: Number of radial points used to validate the interpolation error during tuning.
* `grow_panels::T`: Multiplicative factor used to increase the number of panels when additional resolution is required.
* `grow_M::Int`: Additive increase in polynomial degree during each tuning step.
* `param_strategy::Symbol`: Parameter-reuse strategy. `:global` determines one configuration for the complete spectral sweep, `:segment` permits retuning between sweep segments, and `:manual` uses the supplied interpolation parameters without automatic tuning.

!!! note "Supported kernels and numeric types"
    Chebyshev-accelerated matrix construction is available for
    [`DoubleLayerPotentialSolver`](@ref) and
    [`CombinedFieldIntegralEquationSolver`](@ref) kernels with `Float64`
    arithmetic. [`CompositeBIMSolver`](@ref) kernels and other numeric types
    are not supported by this acceleration path. Use `use_chebyshev=false`
    for those cases.
"""
struct ChebyshevConfig{T<:Real}
    n_panels_h::Int
    M_h::Int
    n_panels_j::Int
    M_j::Int
    tol::T
    max_iter::Int
    sampling_points::Int
    grow_panels::T
    grow_M::Int
    param_strategy::Symbol
end

"""
    ChebyshevConfig(::Type{T}=Float64; n_panels_h::Int=10000, M_h::Int=5, n_panels_j::Int=5000, M_j::Int=5, tol::Real=1e-12, max_iter::Int=20, sampling_points::Int=10_000, grow_panels::Real=1.5, grow_M::Int=2, param_strategy::Symbol=:global) where {T<:Real}

Construct a [`ChebyshevConfig`](@ref) controlling the piecewise-Chebyshev
approximation used by accelerated BIM kernel evaluations.

The panel counts and polynomial degrees specify the initial Hankel and Bessel-J
approximations. With automatic tuning, these parameters are increased until the
validation error satisfies `tol` or `max_iter` iterations have been reached.
For most calculations the defaults are intended to provide a high-accuracy
starting point.

## Arguments
* `::Type{T} = Float64`: Real numeric type used to store tolerances and growth parameters.

## Keyword Arguments
* `n_panels_h::Int = 10000`: Initial number of radial Chebyshev panels for Hankel functions.
* `M_h::Int = 5`: Chebyshev polynomial degree on each Hankel panel.
* `n_panels_j::Int = 5000`: Initial number of radial Chebyshev panels for Bessel-J functions.
* `M_j::Int = 5`: Chebyshev polynomial degree on each Bessel-J panel.
* `tol::Real = 1e-12`: Target absolute interpolation error for automatic tuning.
* `max_iter::Int = 20`: Maximum number of automatic tuning iterations.
* `sampling_points::Int = 10_000`: Number of radial validation points used to estimate the interpolation error.
* `grow_panels::Real = 1.5`: Multiplicative panel-count growth factor applied during tuning.
* `grow_M::Int = 2`: Additive polynomial-degree increase applied during tuning.
* `param_strategy::Symbol = :global`: Parameter-reuse strategy; one of `:global`, `:segment`, or `:manual`.

## Returns
* `cfg::ChebyshevConfig{T}`: Chebyshev interpolation configuration.
"""
function ChebyshevConfig(::Type{T}=Float64; n_panels_h::Int=10000, M_h::Int=5, n_panels_j::Int=5000, M_j::Int=5, tol::Real=1e-12, max_iter::Int=20, sampling_points::Int=10_000, grow_panels::Real=1.5, grow_M::Int=2, param_strategy::Symbol=:global) where {T<:Real}
    param_strategy in (:global, :segment, :manual) || throw(ArgumentError("param_strategy must be :global, :segment or :manual; received $param_strategy"))
    return ChebyshevConfig{T}(n_panels_h, M_h, n_panels_j, M_j, T(tol), max_iter, sampling_points, T(grow_panels), grow_M, param_strategy)
end