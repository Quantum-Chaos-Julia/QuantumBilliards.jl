---
name: julia-refactor-wire-feature
description: "Use for wiring a newly added Julia feature file into the QuantumBilliards.jl ecosystem (module includes, exports, abstract-type conformance, placement conventions). Trigger phrases: wire in this new feature, integrate this new file, hook up new basis/solver/curve, missing exports for new type."
---

# Julia Refactoring — Use Case B: Wiring a New Feature File Into the Library

Use this when a new source file already exists (containing a new basis,
solver, curve/domain, statistic, model, etc.) and needs to be correctly
connected to the rest of the package rather than reviewed for quality.

This skill assumes the shared domain context and numerical/threading
conventions from the invoking agent's body already apply — it only adds the
use-case-specific procedure below.

## What "wired in correctly" means here
1. **Module include/export**: the new file has an `include("path/to/file.jl")`
   in the correct place in the package's top-level module file (grouped with
   its mathematical concept, matching the ordering/grouping already used
   there), and every genuinely public symbol it defines is `export`-ed
   alongside that include, matching the existing style (e.g.
   `include("basis/planewaves/newbasis.jl"); export NewBasis`).
2. **Abstract-type conformance**: if the new type is meant to implement an
   existing interface (`AbsBasis`, `AbsSolver`, `AbsState`, `AbsCurve`,
   `AbsDomain`, `AbsBilliard`, etc.), verify it actually subtypes it and
   implements every function listed in that abstract type's documented API
   (read the abstract type's docstring in `abstracttypes.jl` to get the
   exact list) — report any missing method as a concrete gap, not a style
   note.
3. **Placement convention**: verify the file lives in the folder matching
   its mathematical concept (basis/solvers/spectra/states/geometry/
   quadrature/utils, per the existing folder layout) — flag if misplaced.
4. **Local convention compliance**: verify the new file itself follows the
   type-stability/memory/threading conventions above (it's new code, so
   these should be correct from the start, not just flagged).
5. **Downstream reuse check**: verify the new feature reuses existing
   shared utilities it should depend on (e.g. `basis_matrix`,
   `filter_matrix!`, `@use_threads`, existing quadrature/sampler utilities)
   rather than duplicating them.

## Approach
1. Read the new file in full.
2. Identify the closest existing abstract type/interface it should
   implement (or confirm it's a standalone addition needing no interface).
3. Read that abstract type's docstring/API list and the package's top-level
   module file's `include`/`export` block for the matching concept area.
4. Check each of the 5 points above concretely (search for the `include`,
   check `export`, check method implementations, check file location, check
   conventions, search for reusable utilities it should call instead).
5. Report every gap found (missing include, missing export, missing method,
   wrong folder, unused existing utility, convention violation in the new
   file).
6. If asked to fix the wiring, apply only the wiring changes (add the
   `include`/`export` lines, add missing methods/fields, move the file if
   genuinely misplaced) — do not redesign the feature's internals beyond
   what's needed to satisfy the interface, and confirm with the user before
   moving files or changing signatures.
7. Run `get_errors` on the changed files (and the top-level module file) to
   confirm the package still loads/compiles cleanly.
8. If the newly wired feature is a simple, self-contained function you
   already verified by running it, you may add a small `@testset` for it
   yourself (never fabricate the expected value). For anything more
   involved (solver/basis/billiard-level behavior), remind the user that
   test coverage should go through the Julia Test Writer agent.

## Output Format
Present findings as a concise categorized report (not a new markdown file):
- **Module wiring** (include/export status)
- **Abstract-type conformance** (missing methods, if any)
- **Placement** (correct folder or not)
- **Convention compliance** (in the new file)
- **Reuse opportunities** (existing utilities it should call instead)

Each item: file/location reference, one-line issue, one-line suggestion.
If edits were applied, summarize exactly what changed after the report.
