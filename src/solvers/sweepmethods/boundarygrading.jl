"""
BoundaryGrading

`BoundaryGrading` is the abstract supertype for the boundary discretization
strategies used by [`SweepBIMSolver`](@ref) concrete types
([`DoubleLayerPotentialSolver`](@ref), [`CombinedFieldIntegralEquationSolver`](@ref)).

## Description
A `BoundaryGrading` selects how `evaluate_points` samples, and optionally
reparametrizes, each boundary curve before the Nyström/Kress-corrected
Fredholm matrix is assembled. Concrete subtypes are
[`SmoothPeriodicGrading`](@ref) (a single smooth closed curve, or a smooth
composite boundary with no true corners, using ungraded periodic
quadrature), [`CornerGrading`](@ref) (a single curve parametrization with
known parametric corner singularities, Kress-graded), and
[`GlobalCornerGrading`](@ref) (a composite/piecewise-smooth boundary,
globally Kress-graded relative to all detected corners).

## API
Concrete subtypes of `BoundaryGrading` are dispatched on by `evaluate_points`
for the [`SweepBIMSolver`](@ref) they are attached to.
"""
abstract type BoundaryGrading end

"""
    SmoothPeriodicGrading <: BoundaryGrading

Ungraded periodic discretization for a single smooth closed boundary curve, or
a smooth composite boundary with no true corners. The periodic Kress
logarithmic singularity correction is applied without any parameter
reclustering. This is the default, and lowest-overhead, grading strategy.
"""
struct SmoothPeriodicGrading <: BoundaryGrading end

"""
    CornerGrading{T<:Real} <: BoundaryGrading

Kress-graded discretization for a single closed curve parametrized on
`[0, 1)` whose corner(s) are known parametric singularities (e.g. a polygon
edge parametrization with a corner at `t = 0`).

## Attributes
* `kressq::Int`: Order of the Kress grading transformation `t = w(σ)`.
* `min_t_spacing::T`: Minimum permitted physical-parameter spacing after grading, guarding against machine-precision node collisions for large `kressq`.
"""
struct CornerGrading{T<:Real} <: BoundaryGrading
    kressq::Int
    min_t_spacing::T
end

"""
    CornerGrading(; kressq::Int = 2, min_t_spacing::Real = 1e-12) → grading::CornerGrading

Constructs a [`CornerGrading`](@ref) with default Kress grading order `2`.

## Keyword arguments
* `kressq::Int = 2`: Order of the Kress grading transformation.
* `min_t_spacing::Real = 1e-12`: Minimum permitted physical-parameter spacing after grading.

## Returns
* `grading`: A [`CornerGrading`](@ref) instance.
"""
CornerGrading(; kressq::Int=2, min_t_spacing::Real=1e-12) = CornerGrading(kressq, min_t_spacing)

"""
    GlobalCornerGrading{T<:Real} <: BoundaryGrading

Kress-graded discretization for a composite/piecewise-smooth closed boundary
built from several joined curve segments. True corners are detected from
tangent discontinuities between adjacent segments and a single global grading
map is built relative to all of them simultaneously; if no corners are
detected, the boundary is discretized as an ungraded uniform periodic mesh
(equivalent to [`SmoothPeriodicGrading`](@ref)).

## Attributes
* `kressq::Int`: Order of the global Kress grading transformation.
* `min_t_spacing::T`: Minimum permitted physical-parameter spacing after grading.
"""
struct GlobalCornerGrading{T<:Real} <: BoundaryGrading
    kressq::Int
    min_t_spacing::T
end

"""
    GlobalCornerGrading(; kressq::Int = 2, min_t_spacing::Real = 1e-12) → grading::GlobalCornerGrading

Constructs a [`GlobalCornerGrading`](@ref) with default Kress grading order `2`.

## Keyword arguments
* `kressq::Int = 2`: Order of the global Kress grading transformation.
* `min_t_spacing::Real = 1e-12`: Minimum permitted physical-parameter spacing after grading.

## Returns
* `grading`: A [`GlobalCornerGrading`](@ref) instance.
"""
GlobalCornerGrading(; kressq::Int=2, min_t_spacing::Real=1e-12) = GlobalCornerGrading(kressq, min_t_spacing)

"""
    _bim_numeric_type(solver::SweepBIMSolver) → Type{<:Real}

Internal helper returning the real scalar type `T` used by a
[`SweepBIMSolver`](@ref) kernel, used by [`BeynSolver`](@ref) and
[`ExpandedBIMSolver`](@ref) to infer their own numeric type from the wrapped
kernel. Every concrete `SweepBIMSolver` is expected to add a method.
"""
function _bim_numeric_type end
