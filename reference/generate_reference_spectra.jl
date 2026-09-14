# Generic harness for producing REAL (not guessed) reference wavenumbers for
# the QuantumBilliardsTests comprehensive test matrix.
#
# Run this file (`julia --project=@quantumchaos reference/generate_reference_spectra.jl`,
# or `include` it and call `main()`) to (re)generate `reference_spectra.jl`.
# Every constant written there comes from actually calling
# `solve_wavenumber`/`accelerated_states_near` on the real solvers -- never
# hand-edit `reference_spectra.jl` with guessed numbers.
#
# The billiard/basis/solver fixtures and the shared `Case` table live in
# reference/spectrum_cases.jl (also `include`d by plottingtests_comprehensive.jl,
# so both scripts run the exact same fixtures/solvers). See
# scratchpad/testing-framework-plan.md for the full solver/billiard matrix.
#
# `main(; recompute_all=true)`: pass `recompute_all=false` to reuse any
# existing entry already present in `reference_spectra.jl` (matched by
# case name -- see `parse_existing_consts`/`reused_case_lines`) instead of
# re-solving it, and only actually run the cases that don't have a
# (complete) entry yet. Under `recompute_all=false`, a case that previously
# errored (a `# [error] <name>: ...` line, no `const` entries) is retried by
# default (`recompute_errors=true`); pass `recompute_errors=false` to instead
# leave previously-errored cases untouched (see `parse_existing_errors`).

using Revise
using QuantumBilliards
using BilliardGeometry

include(joinpath(@__DIR__, "spectrum_cases.jl"))

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

function run_case(case::Case)
    
    lines = String[]
    solver = case.solver()
    try
        if case.kind === :basis
            billiard, basis = case.fixture()
            k, t1 = solve_wavenumber(solver, basis, billiard, case.k0, case.dk0)
            println("[ground] $(case.name): k=$k t1=$t1")
            push!(lines, "const K_GROUND_$(case.name) = $(repr(k))")
            push!(lines, "const T_GROUND_$(case.name) = $(repr(t1))")
            for (suffix, k0s, n) in case.states
                ks, tens = accelerated_states_near(solver, basis, billiard; k0=k0s, n_target=n)
                println("[states] $(case.name)_$(suffix): found $(length(ks)) states: $ks")
                push!(lines, "const KS_$(case.name)_$(suffix) = $(repr(collect(ks)))")
                push!(lines, "const TENS_$(case.name)_$(suffix) = $(repr(collect(tens)))")
            end
        else
            billiard = case.fixture()
            k, t1 = solve_wavenumber(solver, billiard, case.k0, case.dk0)
            println("[ground] $(case.name): k=$k t1=$t1")
            push!(lines, "const K_GROUND_$(case.name) = $(repr(k))")
            push!(lines, "const T_GROUND_$(case.name) = $(repr(t1))")
            for (suffix, k0s, n) in case.states
                ks, tens = accelerated_states_near(solver, billiard; k0=k0s, n_target=n)
                println("[states] $(case.name)_$(suffix): found $(length(ks)) states: $ks")
                push!(lines, "const KS_$(case.name)_$(suffix) = $(repr(collect(ks)))")
                push!(lines, "const TENS_$(case.name)_$(suffix) = $(repr(collect(tens)))")
            end
        end
    catch e
        println("[error] $(case.name): $e")
        return ["# [error] $(case.name): $e"]
    end
    return lines
end

# Parse a previously-generated reference_spectra.jl into a Dict mapping each
# top-level const name (e.g. "K_GROUND_CIRCLE_DECOMP_RPW_LIN") to its full
# source line, so `main(recompute_all=false)` can identify which cases
# already have an entry and reuse their lines verbatim instead of re-running
# the (potentially expensive) solve.
function parse_existing_consts(path)
    consts = Dict{String,String}()
    isfile(path) || return consts
    for line in eachline(path)
        m = match(r"^const (\w+) = ", line)
        m === nothing && continue
        consts[m.captures[1]] = line
    end
    return consts
end

# Parse a previously-generated reference_spectra.jl into a Dict mapping each
# errored case's name (e.g. "CIRCLE_DECOMP_RPW_LIN") to its
# `# [error] <name>: ...` line, so `main(recompute_errors=false)` can leave
# previously-errored cases untouched instead of retrying them.
function parse_existing_errors(path)
    errors = Dict{String,String}()
    isfile(path) || return errors
    for line in eachline(path)
        m = match(r"^# \[error\] (\S+):", line)
        m === nothing && continue
        errors[m.captures[1]] = line
    end
    return errors
end

# Reuse the previously-generated lines for `case` (ground state, plus every
# states window in `case.states`) if and only if all of them are already
# present in `existing` -- otherwise return `nothing` so the caller falls
# back to actually re-solving the case.
function reused_case_lines(case::Case, existing::Dict{String,String})
    keys_needed = ["K_GROUND_$(case.name)", "T_GROUND_$(case.name)"]
    for (suffix, _, _) in case.states
        push!(keys_needed, "KS_$(case.name)_$(suffix)")
        push!(keys_needed, "TENS_$(case.name)_$(suffix)")
    end
    all(k -> haskey(existing, k), keys_needed) || return nothing
    return [existing[k] for k in keys_needed]
end

function main(; write_output::Bool=true, cases=CASES, recompute_all::Bool=true, recompute_errors::Bool=true)
    lines = String[
        "# Auto-generated by reference/generate_reference_spectra.jl -- DO NOT hand-edit.",
        "# Regenerate by running that script (`julia --project=@quantumchaos reference/generate_reference_spectra.jl`).",
        "# See scratchpad/testing-framework-plan.md for the full solver/billiard matrix",
        "# and which cells are populated here vs. left as follow-up work.",
        "#",
        "# NOTE on T_GROUND_*_EBIM_* (ExpandedBIMSolver): unlike every other T_GROUND_*",
        "# constant (a genuine tension/residual), ExpandedBIMSolver's solve_wavenumber",
        "# returns the magnitude of its second-order local Taylor correction |corr|",
        "# (see solvers/acceleratedmethods/ebim.jl docstring) -- it is NOT comparable",
        "# to other solvers' tensions and should only be checked for consistency",
        "# against itself on re-runs, not compared numerically across solvers.",
        "#",
        "# NOTE on KS_*/TENS_* constants: these come from accelerated_states_near,",
        "# which locates n_target states starting at the Weyl-law state index",
        "# nearest each case's own K20/K100 target via compute_spectrum(N1,N2) --",
        "# for a sweep solver this is computed by its accelerated sibling",
        "# (VerginiSaracenoSolver for a basis solver, BeynSolver for a BIM solver,",
        "# see spectrum_harness.jl's accel_solver_for), not the sweep solver itself.",
        "",
    ]
    existing = recompute_all ? Dict{String,String}() : parse_existing_consts(joinpath(@__DIR__, "reference_spectra.jl"))
    existing_errors = recompute_all ? Dict{String,String}() : parse_existing_errors(joinpath(@__DIR__, "reference_spectra.jl"))
    for case in cases
        case.enabled || continue
        reused = recompute_all ? nothing : reused_case_lines(case, existing)
        if reused !== nothing
            println("[skip] $(case.name): reusing existing entry")
            append!(lines, reused)
        elseif !recompute_all && !recompute_errors && haskey(existing_errors, case.name)
            println("[skip] $(case.name): reusing existing error entry")
            push!(lines, existing_errors[case.name])
        else
            append!(lines, run_case(case))
        end
    end
    if write_output
        outpath = joinpath(@__DIR__, "reference_spectra.jl")
        write(outpath, join(lines, "\n") * "\n")
        println("wrote $(outpath)")
    end
    return nothing
end

main(recompute_all = false)  # pass `recompute_all=true` to re-solve every case instead of reusing existing entries 
