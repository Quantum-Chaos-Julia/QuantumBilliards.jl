"""
    with_params(solver; kwargs...) -> typeof(solver)

Return a shallow copy of the immutable `solver` with the named fields replaced
by the given values. Unspecified fields, including the kernel and any basis,
are shared with the original, which is never modified.

## Arguments
* `solver`: Solver value whose type has a positional field constructor.

## Keyword Arguments
* `kwargs...`: Field names and replacement values.

## Returns
* A new solver of the same family with the replaced fields.
"""
function with_params(solver::S; kwargs...) where {S}
    names = fieldnames(S)
    for k in keys(kwargs)
        k in names || throw(ArgumentError("$(nameof(S)) has no field `$k`"))
    end
    vals = ntuple(i -> get(kwargs, names[i], getfield(solver, names[i])), length(names))
    return S.name.wrapper(vals...)
end

"""
    expected_states(billiard::AbsBilliard, k1, k2; fundamental_only::Bool=false) -> Float64

Weyl-law estimate of the number of levels with `k1 ≤ k ≤ k2`, clamped at zero.

## Arguments
* `billiard::AbsBilliard`: Billiard geometry.
* `k1`, `k2`: Interval bounds.

## Keyword Arguments
* `fundamental_only::Bool=false`: Count only the fundamental domain (symmetry-reduced).

## Returns
* `Float64`: Expected number of states in the interval.
"""
expected_states(billiard::AbsBilliard, k1, k2; fundamental_only::Bool=false) = max(0.0, Float64(state_at_k(k2, billiard; fundamental_only)) - Float64(state_at_k(k1, billiard; fundamental_only)))

"""
    WindowPlan{T<:Real}

Resolved description of one spectral window or panel.

## Fields
* `kcenter::T`: Window center.
* `halfwidth::T`: Window half-width.
* `expected_states::Float64`: Weyl estimate of the levels in the window.
* `dim::Int`: Basis size, or number of full boundary points for BIM solvers.
* `matrix_dim::Int`: Dimension of the matrix handled in the window.
"""
struct WindowPlan{T<:Real}
    kcenter::T
    halfwidth::T
    expected_states::Float64
    dim::Int
    matrix_dim::Int
end

"""
    ComputationPlan{T<:Real,S}

Side-effect-free description of a `compute_spectrum` run.

## Fields
* `solver::S`: Solver with resolved parameters.
* `k1::T`, `k2::T`: Requested interval.
* `expected_total_states::Float64`: Weyl estimate for the whole interval.
* `density_k1::Float64`, `density_k2::Float64`: Mean density of states at the interval ends.
* `windows::Vector{WindowPlan{T}}`: Window layout.
* `memory::MemoryEstimate`: Memory estimate.
* `nthreads::Int`: Julia threads available.
* `notes::Vector{String}`: Parameter adjustments and warnings.
"""
struct ComputationPlan{T<:Real,S}
    solver::S
    k1::T
    k2::T
    expected_total_states::Float64
    density_k1::Float64
    density_k2::Float64
    windows::Vector{WindowPlan{T}}
    memory::MemoryEstimate
    nthreads::Int
    notes::Vector{String}
end

function Base.show(io::IO, ::MIME"text/plain", plan::ComputationPlan)
    w = plan.windows; n = length(w)
    println(io, "ComputationPlan for ", nameof(typeof(plan.solver)))
    println(io, "  interval        : [", plan.k1, ", ", plan.k2, "]")
    println(io, "  expected states : ", round(plan.expected_total_states; digits=1), "  (density ", round(plan.density_k1; digits=2), " to ", round(plan.density_k2; digits=2), ")")
    println(io, "  windows         : ", n)
    if n > 0
        s = sort!([x.expected_states for x in w]); d = [x.dim for x in w]; m = [x.matrix_dim for x in w]
        smed = isodd(n) ? s[(n+1)÷2] : (s[n÷2] + s[n÷2+1]) / 2
        println(io, "  states/window   : min ", round(first(s); digits=1), ", median ", round(smed; digits=1), ", max ", round(last(s); digits=1))
        println(io, "  dim             : ", minimum(d), " to ", maximum(d))
        println(io, "  matrix dim      : ", minimum(m), " to ", maximum(m))
    end
    println(io, "  memory          : ", plan.memory)
    print(io, "  threads         : ", plan.nthreads)
    for note in plan.notes
        print(io, "\n  note            : ", note)
    end
end

Base.show(io::IO, plan::ComputationPlan) = print(io, "ComputationPlan(", nameof(typeof(plan.solver)), ", ", length(plan.windows), " windows, ", round(plan.expected_total_states; digits=1), " states)")
