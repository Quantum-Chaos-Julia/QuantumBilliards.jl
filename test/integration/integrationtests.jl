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
end

@group "accelerated" begin
    include("accelerated/beyn_spectra.jl")
    include("accelerated/ebim_spectra.jl")
    include("accelerated/cork_spectra.jl")
end

@group "backends" begin
    include("backends/taylor_spectra.jl")
    include("backends/chebyshev_spectra.jl")
end

@group "symmetry" begin
    include("symmetry/reflection_spectra.jl")
    include("symmetry/rotation_spectra.jl")
end

@group "drivers and states" begin
    include("drivers/compute_spectrum_drivers.jl")
    include("states/spectrum_states.jl")
end
