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
