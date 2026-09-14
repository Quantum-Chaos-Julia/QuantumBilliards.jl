# Shared harness for building the "accelerated sibling" of any solver and
# for locating N wavenumber states near a target k0 via Weyl's law.
#
# Included (not just conceptually mirrored) by BOTH reference/spectrum_cases.jl
# (shared by reference/generate_reference_spectra.jl and
# plottingtests_comprehensive.jl) and spectrumtests_template.jl, so every
# caller re-derives states with the *exact* same algorithm/parameters. Do not
# fork/reimplement these functions elsewhere -- the whole point of sharing
# this file is that every call site is bit-for-bit reproducible given the
# same solver/billiard/k0/n_target.
#
# See scratchpad/testing-framework-plan.md §5 for the rationale (why sweep
# solvers used to need a k_sweep + local-minima search instead of
# `compute_spectrum` -- they now go through it too, via `accel_solver_for`).

"""
    accel_solver_for(solver) -> solver′

Returns the solver used to compute a *multi-state* spectrum near a target
wavenumber, i.e. one with a `compute_spectrum(solver′, [basis,] billiard,
N1, N2)` method.

## Description
Already-accelerated solvers ([`VerginiSaracenoSolver`](@ref),
[`BeynSolver`](@ref), [`ExpandedBIMSolver`](@ref)) are returned unchanged --
`compute_spectrum` already exists for them. A [`SweepBasisSolver`](@ref)
([`DecompositionMethodSolver`](@ref), [`ParticularSolutionsMethod`](@ref)) is
converted to a [`VerginiSaracenoSolver`](@ref) reusing its
`dim_scaling_factor`/`pts_scaling_factor` (the two fields both solver
families share). A [`SweepBIMSolver`](@ref)
([`DoubleLayerPotentialSolver`](@ref),
[`CombinedFieldIntegralEquationSolver`](@ref), [`CompositeBIMSolver`](@ref))
is wrapped as `BeynSolver(solver)`, using it directly as the kernel supplying
the Fredholm operator.
"""
accel_solver_for(solver::AcceleratedBasisSolver) = solver
accel_solver_for(solver::AcceleratedBIMSolver) = solver
accel_solver_for(solver::SweepBasisSolver) = VerginiSaracenoSolver(solver.dim_scaling_factor, solver.pts_scaling_factor)
accel_solver_for(solver::SweepBIMSolver) = BeynSolver(solver; use_chebyshev=false)

"""
    accelerated_states_near(solver::AbsBasisSolver, basis::AbsBasis, billiard::AbsBilliard; k0, n_target, fundamental=true) -> (ks, tens)
    accelerated_states_near(solver::AbsBIMSolver, billiard::AbsBilliard; k0, n_target, fundamental=true) -> (ks, tens)

Locates `n_target` states starting at the Weyl-law state index nearest `k0`.

## Description
`N1 = round(Int, state_at_k(k0, billiard; fundamental))` (clamped to at
least `1`) and `N2 = N1 + n_target`, then delegates to
`compute_spectrum(accel_solver_for(solver), [basis,] billiard, N1, N2)` --
see [`accel_solver_for`](@ref) for how any `solver` (sweep or already
accelerated, basis or BIM) is upgraded to one whose `compute_spectrum(⋅,⋅,
N1,N2)` method actually exists. This replaces the old adaptive
half-width-search/tension-minima approach: since `N1`/`N2` are derived
directly from Weyl's law, no widening loop is needed.

## Returns
* `(ks, tens)`: Wavenumbers and tensions from [`SpectralData`](@ref), sorted by `k`.
"""
function accelerated_states_near(solver::AbsBasisSolver, basis::AbsBasis, billiard::AbsBilliard;
                                  k0::Real, n_target::Int, fundamental::Bool=true)
    accel = accel_solver_for(solver)
    N1 = max(1, round(Int, state_at_k(k0, billiard; fundamental)))
    N2 = N1 + n_target
    data = compute_spectrum(accel, basis, billiard, N1, N2)
    return data.k, data.ten
end

function accelerated_states_near(solver::AbsBIMSolver, billiard::AbsBilliard;
                                  k0::Real, n_target::Int, fundamental::Bool=true)
    accel = accel_solver_for(solver)
    N1 = max(1, round(Int, state_at_k(k0, billiard; fundamental)))
    N2 = N1 + n_target
    data = compute_spectrum(accel, billiard, N1, N2)
    return data.k, data.ten
end
