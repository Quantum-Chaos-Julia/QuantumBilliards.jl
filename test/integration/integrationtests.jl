# Integration tier: spectrum-focused pipeline tests, grouped in nested @testsets.
# `common.jl` (reference generators and tolerances) is shared with every integration file through the harness.

@group "high_k" begin
    run_file("integration/high_k/beyn_high_k_spectra.jl")
end

@group "basis methods" begin
    run_file("integration/basis_methods/decompositionmethod_wavenumber.jl")
    run_file("integration/basis_methods/particularsolutions_wavenumber.jl")
    run_file("integration/basis_methods/verginisaraceno_spectra.jl")
end

@group "BIM sweep" begin
    run_file("integration/bim_sweep/dlp_wavenumber.jl")
    run_file("integration/bim_sweep/cfie_wavenumber.jl")
    run_file("integration/bim_sweep/compositebim_wavenumber.jl")
    run_file("integration/bim_sweep/compositebim_holes.jl")
end

@group "accelerated" begin
    run_file("integration/accelerated/beyn_spectra.jl")
    run_file("integration/accelerated/beyn_isospectral_spectra.jl")
    run_file("integration/accelerated/ebim_spectra.jl")
    run_file("integration/accelerated/cork_spectra.jl")
    run_file("integration/accelerated/profiling_timers.jl")
end

@group "backends" begin
    run_file("integration/backends/taylor_spectra.jl")
    run_file("integration/backends/chebyshev_spectra.jl")
end

@group "symmetry" begin
    run_file("integration/symmetry/reflection_spectra.jl")
    run_file("integration/symmetry/rotation_spectra.jl")
    run_file("integration/symmetry/hole_sector_matrices.jl")
end

# Symmetric holes: reduced composite solvers on billiards with SymmetricHole / SymmetrizedHoles. helpers.jl holds definitions
# only (cached Beyn spectra, kernel builder) and is loaded by the `@requires` line of each file of the group.
@group "symmetric holes" begin
    run_file("integration/symmetric_holes/sinai_sector_spectra.jl")
    run_file("integration/symmetric_holes/reduced_ebim_spectra.jl")
    run_file("integration/symmetric_holes/reduced_states.jl")
    run_file("integration/symmetric_holes/annular_sector_spectra.jl")
    run_file("integration/symmetric_holes/squarewithinsquare_sector_spectra.jl")
    run_file("integration/symmetric_holes/symmetrized_holes_spectra.jl")
    run_file("integration/symmetric_holes/c3_rotation_hole_spectra.jl")
    run_file("integration/symmetric_holes/trivial_group_fallback.jl")
end

@group "drivers and states" begin
    run_file("integration/drivers/compute_spectrum_drivers.jl")
    run_file("integration/states/spectrum_states.jl")
end

@group "docs" begin
    run_file("integration/docs/readme_examples.jl")
end
