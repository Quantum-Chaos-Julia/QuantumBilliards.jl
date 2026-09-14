---
name: julia-refactor-scoped
description: "Use for Julia code review/refactoring scoped to a specific file or feature (not a global audit unless explicitly requested) in the QuantumBilliards.jl ecosystem. Trigger phrases: refactor this file/function, review this code, find weaknesses, code smell, missing exports, should this be internal, unnecessary abstraction, simplify dispatch, macro opportunity."
---

# Julia Refactoring — Use Case A: Scoped Analysis & Refactoring

Reviewing/improving code that already exists, tightly scoped to a specific
function, file, or feature. A workspace/package-wide audit is a different,
much larger operation and must NOT be assumed — only do it if the user
explicitly asks for a global/whole-package review.

This skill assumes the shared domain context and numerical/threading
conventions from the invoking agent's body already apply — it only adds the
use-case-specific procedure below.

## Scope discipline (read first)
- Default scope is exactly what the user pointed at: one function, one file,
  or one clearly named feature (e.g. "the `RealPlaneWaves` basis", "the
  decomposition solver"). Analyze and report on that scope only.
- Do NOT expand to sibling files, the whole module, or the whole package
  "while you're at it" — even if you notice something suspicious elsewhere,
  only note it as a one-line "outside current scope" pointer in the report,
  don't analyze it in depth.
- Only perform a workspace-/package-wide audit (scanning all of `src/` for
  exports, internal candidates, abstractions, macros across the entire
  package or across multiple packages) when the user explicitly asks for a
  global/whole-package/whole-repo review. If the request is ambiguous about
  scope, ask before broadening beyond a single file/feature.

## Specialized Analysis Capabilities

### 1. Export audit
Within the current scope, compare `export` statements in the package's
top-level module file against the `function`/`struct`/`abstract type`
definitions found in the file(s) under review. Report:
- Public-looking symbols (used across files, documented, part of an abstract
  type's API list) that are **not exported** — candidates to add.
- Exported symbols that are unused elsewhere in the package and have no
  docstring — verify intent before flagging as possibly premature exports.

### 2. Internal-only candidates
For the file(s)/function(s) under review, use search/usage tools to
determine, per non-exported function, whether it is referenced only within
its own file (candidate to mark clearly as internal, e.g. via a leading
underscore following existing local conventions like `_rpw_fun`, or grouped
under an "INTERNAL FUNCTIONS" comment block as already done in
`solvers/matrixconstructors.jl`) versus used across multiple files (should
likely stay a regular internal-but-cross-file helper, not underscore-prefixed).

### 3. Abstraction analysis
- **Unnecessary abstraction**: abstract types with only one concrete subtype
  and no evidence of a second implementation planned, or wrapper
  structs/indirection that add no dispatch value — recommend flattening.
- **Missing abstraction**: near-duplicate logic repeated across the
  reviewed scope with no shared abstract type or common method — recommend
  introducing a shared abstract supertype + method(s), following the
  existing `AbsBasis`/`AbsSolver`/`AbsState`-style pattern (fields +
  documented API list in the abstract type's docstring).
- Prefer the smallest abstraction that enables correct dispatch; do not
  propose speculative generality for hypothetical future cases.

### 4. Macro opportunities
Identify repeated boilerplate within scope (e.g. manual threading wrap,
repeated bounds/validation checks, repeated matrix-filtering patterns) that
mirrors what existing macros already solve (`@use_threads`, `@blas_multi`,
`filter_matrix!`) — recommend reusing those first. Only propose a *new*
macro when the same non-trivial boilerplate is duplicated 3+ times within
scope and a plain function/method cannot express it (e.g. it needs to
operate on the call-site expression itself, like the threading wrap does);
otherwise prefer a regular function.

## Approach
1. Confirm scope: exactly one function, file, or named feature, unless the
   user explicitly requested a global/whole-package audit. If unclear, ask.
2. For a whole-package audit only (explicitly requested), use the todo list
   to track each area (exports, internal-candidates, abstractions, macros,
   type-stability/threading violations) as separate items.
3. Read the target code fully, plus its abstract-type contract and, at most,
   the 1-2 most directly related files, before judging anything.
4. Run concrete searches: `export` statements vs. definitions, symbol usages
   across the workspace, repeated code patterns — don't rely on guesswork.
5. Produce a structured findings report (see Output Format) categorizing
   each issue by type and severity, with a one-line rationale and concrete
   suggestion per finding.
6. If the user asks you to apply fixes, implement only the confirmed ones,
   within the same scope, following the exact conventions above, then run
   `get_errors` on changed files to confirm no diagnostics were introduced.
7. If a simple, self-contained function was changed and you already
   verified its real output by running it, you may add a small `@testset`
   for it yourself (never fabricate the expected value). For anything more
   involved, remind the user that test coverage for the refactored code
   should go through the Julia Test Writer agent (do not invoke it
   automatically).

## Output Format
Present findings as a concise categorized report (not a new markdown file):
- **Weaknesses** (type-stability/memory/threading convention violations)
- **Missing exports**
- **Internal-only candidates**
- **Abstraction opportunities** (unnecessary / missing)
- **Macro opportunities**

Each item: file/location reference, one-line issue, one-line suggestion.
If edits were applied, summarize exactly what changed after the report.
