"""
    ReferenceWavenumber(k, t=zero(k))

Dispatch marker for [`plot_state_tests!`](@ref) requesting an already-known
`(k, t)` pair (e.g. pulled from precomputed reference data) be used directly,
instead of re-solving via `solve_wavenumber`.

## Description
Replaces the `compute_wavenumber::Bool`/`t1` keyword pair with a plain
multiple-dispatch choice: call `plot_state_tests!(f, solver, [basis,]
billiard, k0, dk; ...)` (two `Real`s) to solve for the wavenumber as before,
or `plot_state_tests!(f, solver, [basis,] billiard,
ReferenceWavenumber(k, t); ...)` to skip solving entirely and plot the given
state directly.

## Attributes
* `k::T`: Wavenumber to plot at.
* `t::T`: Tension/residual value shown in the figure title (purely
  cosmetic — not used to (re)solve anything).
"""
struct ReferenceWavenumber{T<:Real}
    k::T
    t::T
end
ReferenceWavenumber(k::Real, t::Real=zero(k)) = ReferenceWavenumber(promote(k, t)...)

function plot_state_tests!(f, solver, basis, billiard, k0::Real, dk::Real; b = 10.0, log=false, inside_only=true, fundamental_domain=false)
    k, t1 = solve_wavenumber(solver, basis, billiard, k0, dk)
    return plot_state_tests!(f, solver, basis, billiard, ReferenceWavenumber(k, t1); b, log, inside_only, fundamental_domain)
end

function plot_state_tests!(f, solver, basis, billiard, source::ReferenceWavenumber; b = 10.0, log=false, inside_only=true, fundamental_domain=false)
    k, t1 = source.k, source.t
    state = compute_eigenstate(solver, basis, billiard, k)
    params_str = format_params(solver; basis=basis, latex=true)
    ax1, hmap1 = plot_probability!(f[1:3,1:5],state; b=b,log=log, inside_only=inside_only, fundamental_domain=fundamental_domain, cbar=false)
    ax1.title = L"k = %$(round(k,digits=8)),\ t = %$(round(t1,digits=8)),\quad \mathrm{%$(latex_display_name(solver))}:\ %$(params_str)"
    hidedecorations!(ax1)  
    hidespines!(ax1)
    
    ax2 = plot_boundary_function!(f[4:5,1:2],state; b=2*b)
    ax2.title = L"\mathrm{Boundary\ function}"
    ax2.xlabel = L"s"
    ax2.ylabel = L"u(s)"    
    
    ax3, hmap3 = plot_husimi_function!(f[4:5,3:5],state; b=2*b, cbar=false)
    ax3.title = L"\mathrm{Husimi\ function}"
    ax3.xlabel = L"s"
    ax3.ylabel = L"p"
end

function plot_state_tests!(f, solver::QuantumBilliards.AbsBIMSolver, billiard::QuantumBilliards.AbsBilliard, k0::Real, dk::Real; b = :auto, log=false, inside_only=true)
    k, t1 = solve_wavenumber(solver, billiard, k0, dk)
    return plot_state_tests!(f, solver, billiard, ReferenceWavenumber(k, t1); b, log, inside_only)
end

function plot_state_tests!(f, solver::QuantumBilliards.AbsBIMSolver, billiard::QuantumBilliards.AbsBilliard, source::ReferenceWavenumber; b = :auto, log=false, inside_only=true)
    k, t1 = source.k, source.t
    state = compute_eigenstate(solver, billiard, k)
    params_str = format_params(solver; latex=true)
    ax1, hmap1 = plot_probability!(f[1:3,1:5],state; b=b, log=log, inside_only=inside_only, cbar=false)
    ax1.title = L"k = %$(round(k,digits=8)),\ t = %$(round(t1,digits=8)),\quad \mathrm{%$(latex_display_name(solver))}:\ %$(params_str)"
    hidedecorations!(ax1)
    hidespines!(ax1)

    ax2 = plot_boundary_function!(f[4:5,1:2],state)
    ax2.title = L"\mathrm{Boundary\ function}"
    ax2.xlabel = L"s"
    ax2.ylabel = L"u(s)"

    ax3, hmap3 = plot_husimi_function!(f[4:5,3:5],state; cbar=false)
    ax3.title = L"\mathrm{Husimi\ function}"
    ax3.xlabel = L"s"
    ax3.ylabel = L"p"
end
