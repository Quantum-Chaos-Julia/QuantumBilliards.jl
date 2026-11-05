# Integration tests for the @timeit_debug / @debug instrumentation of the accelerated solvers (compute_spectrum for
# VerginiSaracenoSolver, BeynSolver, ExpandedBIMSolver and CORKSolver in src/spectra/spectralutils.jl and
# src/solvers/acceleratedmethods/). Each solver runs once on a small case taken from its calibrated *_spectra.jl file with
# the QuantumBilliards debug timings enabled; the expected labels are looked up in TimerOutputs.DEFAULT_TIMER. Result
# invariance is checked by comparing the levels with those from a run with the timings disabled. No timings, tensions or
# wavefunction values are asserted. Requires TimerOutputs in test/Project.toml.

using TimerOutputs
using Logging

# Timer node at the given label path, or nothing when any label is missing.
function prof_node(to::TimerOutput, path::AbstractString...)
    node = to
    for label in path
        haskey(node.inner_timers, label) || return nothing
        node = node.inner_timers[label]
    end
    return node
end

prof_has(path::AbstractString...) = prof_node(TimerOutputs.DEFAULT_TIMER, path...) !== nothing

# Toggling the debug timings redefines a method of QuantumBilliards, so the timings are switched only twice per file:
# every run with timings disabled (`prof_off`) is done first, then the timings are enabled once and every instrumented run
# (`prof_on`) is done, the timings are disabled again at the end of the file. `invokelatest` is needed because of the
# world-age change. The enabled run leaves its timer in TimerOutputs.DEFAULT_TIMER.
prof_off(f) = (TimerOutputs.reset_timer!(); data = Base.invokelatest(f); (data, isempty(TimerOutputs.DEFAULT_TIMER.inner_timers)))
function prof_on(f)
    TimerOutputs.reset_timer!()
    return Base.invokelatest(f)
end

# Solver set-ups shared by the disabled and the enabled run.
const PROF_VS_BILLIARD, PROF_VS_BASIS = QuantumBilliards.make_veech_right_triangle_and_basis(5)
const PROF_VS_SOLVER = VerginiSaracenoSolver(2.0, 5.0)
const PROF_TRI = QuantumBilliards.make_veech_right_triangle(5)
const PROF_TRI_KERNEL = DoubleLayerPotentialSolver(5.0, PROF_TRI; grading=GlobalCornerGrading())
const PROF_BEYN_SOLVER = BeynSolver(PROF_TRI_KERNEL; use_chebyshev=false)
const PROF_EBIM_SOLVER = ExpandedBIMSolver(PROF_TRI_KERNEL)
const PROF_CIRCLE = CircleBilliard(1.0)
const PROF_CORK_SOLVER = CORKSolver(CombinedFieldIntegralEquationSolver(20.0, PROF_CIRCLE; grading=SmoothPeriodicGrading(), min_pts=800); nlevels=40)

run_vs() = compute_spectrum(PROF_VS_SOLVER, PROF_VS_BASIS, PROF_VS_BILLIARD, 110.0, 110.1, 0.1; multithreaded=false, show_progress=false)
run_beyn() = compute_spectrum(PROF_BEYN_SOLVER, PROF_TRI, 5.5, 6.7; multithreaded=false, show_progress=false)
run_ebim() = compute_spectrum(PROF_EBIM_SOLVER, PROF_TRI, 5.5, 6.7; accept_tol=nothing, multithreaded=false, show_progress=false)
run_cork() = compute_spectrum(PROF_CORK_SOLVER, PROF_CIRCLE, 3.0, 5.0; multithreaded=false, show_progress=false)

TimerOutputs.disable_debug_timings(QuantumBilliards)
const PROF_OFF = Dict(:vs => prof_off(run_vs), :beyn => prof_off(run_beyn), :ebim => prof_off(run_ebim), :cork => prof_off(run_cork))
TimerOutputs.enable_debug_timings(QuantumBilliards)

# solver: Vergini-Saraceno
# basis: Corner Adapted Fourier-Bessel
# billiard: Veech right triangle (make_veech_right_triangle_and_basis(5))
# symmetry: None
# reference: timer labels of the compute_spectrum sweep; levels with timings on equal levels with timings off
# functions to test: compute_spectrum (VS), _vs_solve_window, _finalize_spectrum
@testset "VS timers - Veech Triangle - labels, invariance and window debug log" begin
    data_off, off_recorded_nothing = PROF_OFF[:vs]
    data_on = prof_on(run_vs)
    @test prof_has("compute_spectrum_vs")
    @test off_recorded_nothing
    @test prof_has("compute_spectrum_vs", "preparation")
    @test prof_has("compute_spectrum_vs", "solution")
    @test prof_has("compute_spectrum_vs", "postprocessing")
    # window solves sit below the solution stage; locate the label anywhere under compute_spectrum_vs
    function find_label(node::TimerOutput, label)
        haskey(node.inner_timers, label) && return true
        return any(c -> find_label(c, label), values(node.inner_timers))
    end
    @test find_label(prof_node(TimerOutputs.DEFAULT_TIMER, "compute_spectrum_vs"), "vs_window_solve")
    @test length(data_on.k) > 0
    @test isequal(data_on.k, data_off.k)
    # debug message emitted once per solved window
    @test_logs (:debug, r"VS window solved") match_mode=:any min_level=Logging.Debug run_vs()
end

# solver: Beyn contour integral
# kernel/basis: DLP, GlobalCornerGrading, ppw=5 (Taylor backend, the BeynSolver default)
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: timer labels of the compute_spectrum sweep; levels with timings on equal levels with timings off
# functions to test: compute_spectrum (Beyn)
@testset "Beyn timers - Full Triangle - labels and invariance" begin
    data_off, off_recorded_nothing = PROF_OFF[:beyn]
    data_on = prof_on(run_beyn)
    @test prof_has("compute_spectrum_beyn")
    @test off_recorded_nothing
    for label in ("preparation", "solution")
        @test prof_has("compute_spectrum_beyn", label)
    end
    @test prof_has("compute_spectrum_beyn", "solution", "beyn_projected_solve")
    @test prof_has("compute_spectrum_beyn", "solution", "beyn_projected_solve", "beyn_construct_matrices")
    @test prof_has("compute_spectrum_beyn", "solution", "beyn_projected_solve", "svd")
    @test prof_has("compute_spectrum_beyn", "solution", "beyn_projected_solve", "reduced_eigen")
    @test length(data_on.k) > 0
    @test isequal(data_on.k, data_off.k)
end

# solver: Expanded BIM
# kernel/basis: DLP, GlobalCornerGrading, ppw=5
# billiard: Veech right triangle, full boundary (make_veech_right_triangle(5))
# symmetry: None
# reference: timer labels of the compute_spectrum sweep; levels with timings on equal levels with timings off
# functions to test: compute_spectrum (EBIM)
@testset "EBIM timers - Full Triangle - labels and invariance" begin
    data_off, off_recorded_nothing = PROF_OFF[:ebim]
    data_on = prof_on(run_ebim)
    @test prof_has("compute_spectrum_ebim")
    @test off_recorded_nothing
    for label in ("preparation", "solution", "postprocessing")
        @test prof_has("compute_spectrum_ebim", label)
    end
    @test length(data_on.k) > 0
    @test isequal(data_on.k, data_off.k)
end

# solver: CORK
# kernel/basis: CFIE, SmoothPeriodicGrading, ppw=20, min_pts=800, nlevels=40
# billiard: Circle, radius 1
# symmetry: None
# reference: timer labels of the compute_spectrum sweep; levels with timings on equal levels with timings off
# functions to test: compute_spectrum (CORK)
@testset "CORK timers - Circle - labels and invariance" begin
    data_off, off_recorded_nothing = PROF_OFF[:cork]
    data_on = prof_on(run_cork)
    @test prof_has("compute_spectrum_cork")
    @test off_recorded_nothing
    for label in ("preparation", "solution")
        @test prof_has("compute_spectrum_cork", label)
    end
    @test length(data_on.k) > 0
    @test isequal(data_on.k, data_off.k)
end

TimerOutputs.disable_debug_timings(QuantumBilliards)
