"""
    ChebyshevConfig{T<:Real}

Bundles every tunable parameter controlling Chebyshev-interpolated
Hankel/Bessel-J kernel evaluation into a single configuration object, instead
of scattering `n_panels_h`/`M_h`/`n_panels_j`/`M_j`/`cheb_tol`/... across many
keyword arguments the way `QuantumBilliards-develop` does. A `ChebyshevConfig`
is stored as the `cheb_config` field of every [`AcceleratedBIMSolver`](@ref)
(`BeynSolver`, `ExpandedBIMSolver`), so it can be constructed once and reused
across an entire [`compute_spectrum`](@ref) sweep.

## Attributes
* `n_panels_h::Int`: Hankel-function Chebyshev panel count.
* `M_h::Int`: Hankel-function Chebyshev polynomial degree per panel.
* `n_panels_j::Int`: Bessel-J-function Chebyshev panel count.
* `M_j::Int`: Bessel-J-function Chebyshev polynomial degree per panel.
* `tol::T`: Target absolute error used by the auto-tuning routine when growing panel counts/degrees.
* `max_iter::Int`: Maximum number of auto-tuning iterations.
* `sampling_points::Int`: Number of radial validation points used during auto-tuning.
* `grow_panels::T`: Multiplicative panel-count growth factor during auto-tuning.
* `grow_M::Int`: Additive polynomial-degree growth during auto-tuning.
* `param_strategy::Symbol`: One of `:global`, `:segment` or `:manual`, controlling how often the panelization is retuned across a wide sweep.

!!! note "Chebyshev acceleration"
    The Chebyshev interpolation machinery (panel construction, coefficient
    fitting, auto-tuning, Chebyshev-accelerated `construct_matrices`) lives in
    `solvers/chebyshev/` and is used by [`BeynSolver`](@ref)/
    [`ExpandedBIMSolver`](@ref) whenever `use_chebyshev=true`, for
    [`DoubleLayerPotentialSolver`](@ref)/[`CombinedFieldIntegralEquationSolver`](@ref)
    kernels with `T===Float64` (a [`CompositeBIMSolver`](@ref) kernel or a
    non-`Float64` numeric type raises an error; construct with
    `use_chebyshev=false` instead).
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
    ChebyshevConfig(::Type{T} = Float64; n_panels_h::Int = 15000, M_h::Int = 5, n_panels_j::Int = 10000, M_j::Int = 5, tol::Real = 1e-13, max_iter::Int = 20, sampling_points::Int = 50_000, grow_panels::Real = 1.5, grow_M::Int = 2, param_strategy::Symbol = :global) where {T<:Real} → cfg::ChebyshevConfig

Constructs a [`ChebyshevConfig`](@ref) with `-develop`-matching defaults.

## Keyword Arguments
* `n_panels_h::Int = 15000`: Hankel-function Chebyshev panel count.
* `M_h::Int = 5`: Hankel-function Chebyshev polynomial degree.
* `n_panels_j::Int = 10000`: Bessel-J-function Chebyshev panel count.
* `M_j::Int = 5`: Bessel-J-function Chebyshev polynomial degree.
* `tol::Real = 1e-13`: Target absolute error for auto-tuning.
* `max_iter::Int = 20`: Maximum auto-tuning iterations.
* `sampling_points::Int = 50_000`: Radial validation points used during auto-tuning.
* `grow_panels::Real = 1.5`: Panel-count growth factor during auto-tuning.
* `grow_M::Int = 2`: Polynomial-degree growth during auto-tuning.
* `param_strategy::Symbol = :global`: One of `:global`, `:segment` or `:manual`.

## Returns
* `cfg`: A [`ChebyshevConfig`](@ref) instance.
"""
function ChebyshevConfig(::Type{T}=Float64; n_panels_h::Int=15000, M_h::Int=5, n_panels_j::Int=10000, M_j::Int=5, tol::Real=1e-13, max_iter::Int=20, sampling_points::Int=50_000, grow_panels::Real=1.5, grow_M::Int=2, param_strategy::Symbol=:global) where {T<:Real}
    param_strategy in (:global, :segment, :manual) || throw(ArgumentError("param_strategy must be :global, :segment or :manual; received $param_strategy"))
    return ChebyshevConfig{T}(n_panels_h, M_h, n_panels_j, M_j, T(tol), max_iter, sampling_points, T(grow_panels), grow_M, param_strategy)
end
