---
name: julia-solver-debug-timing
description: "Use to add or review @debug logging and @timeit_debug timing instrumentation in construct_matrices (or similar matrix-construction/decomposition) methods in QuantumBilliards.jl, following the existing TimerOutputs/Logging pattern. Trigger phrases: add debug info, add timing, instrument construct_matrices, add @timeit_debug, add benchmarking, profile matrix construction."
---

# Debug & Timing Instrumentation for Matrix Construction

Add the existing `@debug`/`@timeit_debug` instrumentation pattern to a
`construct_matrices` method (or a related decomposition/eigenproblem
function) — this is **not** a new macro to invent, it's the established
convention already used in
`verginisaraceno.jl` (`QuantumBilliards.jl/src/solvers/acceleratedmethods/verginisaraceno.jl`),
`decompositionmethod.jl` (`QuantumBilliards.jl/src/solvers/sweepmethods/decompositionmethod.jl`),
and `decompositions.jl` (`QuantumBilliards.jl/src/solvers/decompositions.jl`).
Both macros come from packages already in `Project.toml` and already
`using`-imported at the top of `QuantumBilliards.jl`:

```julia
using Logging, TimerOutputs
```

- `@debug "message" var1 var2` — stdlib `Logging.@debug`; a no-op unless the
  user's logger level is set to show debug messages, so it has ~zero runtime
  cost in normal use. Never use `@info`/`println` for this — `@debug` is the
  established level for internal solver diagnostics.
- `@timeit_debug "label" begin ... end` — from `TimerOutputs.jl`; wraps a
  block in a named timer section that is compiled out entirely unless
  timing is enabled, so it also has ~zero cost by default.

## Placement conventions (copy exactly, do not invent a new structure)

1. Wrap the **entire function body** in one outer `@timeit_debug
   "<function_name>" begin ... end` (e.g. `@timeit_debug "construct_matrices"
   begin ... end`, `@timeit_debug "generalized_eigen" begin ... end`).
2. Wrap each logically distinct numerical step inside in its own nested
   `@timeit_debug "<step_name>" begin ... end`, using short snake_case labels
   matching what the step computes (`"basis_matrix"`, `"dk_matrix"`,
   `"compute_F"`, `"compute_Fk"`, `"First decomposition"`, `"Second
   decomposition"`) — mirror the granularity already used in the reference
   files above, one timer per BLAS call / basis evaluation / eigen call, not
   per individual line.
3. Immediately before each major step and immediately after it, add one
   `@debug` call reporting the input sizes/parameters and the output shape:
   ```julia
   @debug "Matrix construction started" N M k nsym
   @timeit_debug "basis_matrix" begin
       G = basis_matrix(basis, k, xy; multithreaded)
   end
   @debug "Basis matrix computed" size=size(G)
   ```
   Use bare variable names as key=value shorthand (`N M k nsym` expands to
   `N=N M=M k=k nsym=nsym`) for cheap scalars, and explicit `key=expr` (e.g.
   `size=size(G)`) when the message needs a derived value rather than the
   variable itself.
4. Do not add `@debug`/`@timeit_debug` around trivial one-line
   allocations/assignments (e.g. `N = basis.dim`) — only around the
   numerically significant steps (basis/gradient evaluation, BLAS rank-k
   updates, eigen decompositions, symmetrization).
5. Never change the function's return value or control flow to accommodate
   instrumentation — `@timeit_debug ... begin ... end` transparently returns
   the block's value, so a `return` inside it still works exactly as before.

## Constraints
- DO NOT introduce a new timing/logging macro, a `TimerOutput()` instance
  stored on the solver struct, or a custom `@mytimer` wrapper — the package
  already uses TimerOutputs' global default timer implicitly via
  `@timeit_debug`; reuse it.
- DO NOT wrap multithreaded inner loops individually with `@timeit_debug`
  per-iteration — only wrap the whole loop/call as one step, since the
  timer itself is not meant to run inside a hot per-element loop.
- DO NOT add instrumentation to functions that aren't matrix-construction or
  decomposition/eigenproblem code (e.g. don't instrument basis evaluation
  internals or simple utility functions) unless explicitly asked.
- DO NOT change existing `@debug`/`@timeit_debug` labels in code that
  already has them correctly — only add instrumentation where it's missing.

## Approach
1. Read the target `construct_matrices` (or decomposition) function in full,
   plus one already-instrumented reference file from the list above, to
   confirm the exact nesting/labeling style to replicate.
2. Identify the function's logically distinct numerical steps (basis/
   gradient evaluation, weight scaling, BLAS rank-k updates,
   symmetrization, eigendecomposition) — these become the nested
   `@timeit_debug` sections.
3. Wrap the whole function body in one outer `@timeit_debug
   "<function_name>"` section.
4. Add a nested `@timeit_debug "<step>"` section per step identified in (2).
5. Add a `@debug` call before the first step (reporting overall
   sizes/parameters) and after each step (reporting the step's output
   shape), matching the reference files' message style.
6. Run `get_errors` on the changed file, and if convenient, run the function
   once in the REPL with debug logging enabled
   (`ENV["JULIA_DEBUG"] = "QuantumBilliards"` or
   `with_logger(ConsoleLogger(stderr, Logging.Debug))`) to confirm the debug
   messages actually print and the function's return value is unchanged.

## Output Format
Directly edit the target file(s) to add the instrumentation. Then give a
brief summary (not a new markdown file) of which steps got a
`@timeit_debug` section and which got `@debug` calls, confirming the
function's return value/control flow is unchanged.
