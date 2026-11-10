# Integration tier: spectrum-focused pipeline tests, grouped in nested @testsets.
# `common.jl` (reference generators and tolerances) is included by runtests.jl before this file.

@group "high_k" begin
    include("high_k/beyn_high_k_spectra.jl")
end

@group "basis methods" begin
    include("basis_methods/decompositionmethod_wavenumber.jl")
    include("basis_methods/particularsolutions_wavenumber.jl")
    include("basis_methods/verginisaraceno_spectra.jl")
end

@group "BIM sweep" begin
    include("bim_sweep/dlp_wavenumber.jl")
    include("bim_sweep/cfie_wavenumber.jl")
    include("bim_sweep/compositebim_wavenumber.jl")
    include("bim_sweep/compositebim_holes.jl")
end

@group "accelerated" begin
    include("accelerated/beyn_spectra.jl")
    include("accelerated/beyn_isospectral_spectra.jl")
    include("accelerated/ebim_spectra.jl")
    include("accelerated/cork_spectra.jl")
    include("accelerated/profiling_timers.jl")
end

@group "backends" begin
    include("backends/taylor_spectra.jl")
    include("backends/chebyshev_spectra.jl")
end

@group "symmetry" begin
    include("symmetry/reflection_spectra.jl")
    include("symmetry/rotation_spectra.jl")
    include("symmetry/hole_sector_matrices.jl")
end

# Symmetric holes (Phase 2): reduced composite solvers on billiards with SymmetricHole / SymmetrizedHoles. helpers.jl holds definitions
# only (cached Beyn spectra, kernel builder); it comes first so that the group is self-contained under QB_TEST_FILTER.
@group "symmetric holes" begin
    include("symmetric_holes/helpers.jl")
    include("symmetric_holes/sinai_sector_spectra.jl")
    include("symmetric_holes/reduced_ebim_spectra.jl")
    include("symmetric_holes/reduced_states.jl")
    include("symmetric_holes/annular_sector_spectra.jl")
    include("symmetric_holes/squarewithinsquare_sector_spectra.jl")
    include("symmetric_holes/symmetrized_holes_spectra.jl")
    include("symmetric_holes/c3_rotation_hole_spectra.jl")
    include("symmetric_holes/trivial_group_fallback.jl")
end

@group "drivers and states" begin
    include("drivers/compute_spectrum_drivers.jl")
    include("states/spectrum_states.jl")
end
