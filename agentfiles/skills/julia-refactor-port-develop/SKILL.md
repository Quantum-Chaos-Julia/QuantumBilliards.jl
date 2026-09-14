---
name: julia-refactor-port-develop
description: "Use for porting an experimental feature from a package's -develop reference repo (e.g. QuantumBilliards-develop, BilliardGeometry-develop) into the corresponding main repo. Trigger phrases: port this from develop, bring this over from the develop repo, integrate feature from -develop."
---

# Julia Refactoring — Use Case C: Porting a Feature from a `-develop` Reference Repo

Use this when the user wants an experimental feature that already exists in a
`-develop` sibling repo (e.g. a new basis, solver, curve, billiard, or
statistic present in `QuantumBilliards-develop` or `BilliardGeometry-develop`)
ported into the corresponding main repo (`QuantumBilliards.jl`,
`BilliardGeometry.jl`).

This skill assumes the shared domain context and numerical/threading
conventions from the invoking agent's body already apply — it only adds the
use-case-specific procedure below.

### `-develop` reference repos
Some packages have a sibling workspace folder with the same name suffixed
`-develop` (currently `QuantumBilliards-develop` alongside `QuantumBilliards.jl`,
and `BilliardGeometry-develop` alongside `BilliardGeometry.jl`). These
`-develop` repos contain experimental but functionally working code for many
features not yet present in the main repo. They exist purely as a **reference
codebase** to port features from — never treat a `-develop` repo as the
target to edit, and never assume its internal conventions (includes, exports,
file layout, helper names) are already correct for the main repo; the main
repo's own `abstracttypes.jl`, top-level module file, and existing
conventions are always the source of truth for how ported code must look.

This is **not** a copy-paste operation and **not** a review of the develop
repo's own quality — the develop repo is read-only reference material. The
job is to reproduce the feature's behavior/algorithm inside the main repo
while making it fully comply with the main repo's current wiring and
conventions, exactly as if it were newly written for Use Case B (new feature
wiring).

## Approach
1. Identify the source file(s) in the `-develop` repo implementing the
   requested feature (search by name/concept if the user didn't give an exact
   path) and read them in full.
2. Identify the equivalent location/module structure in the main repo (same
   mathematical concept folder: basis/solvers/spectra/states/geometry/
   quadrature/utils) — the main repo's folder layout is authoritative, even if
   the `-develop` repo organizes things differently.
3. Read the main repo's `abstracttypes.jl`-style interfaces relevant to the
   feature (e.g. `AbsBasis`, `AbsSolver`, `AbsCurve`, `AbsDomain`,
   `AbsBilliard`) to determine the exact API the ported type/function must
   satisfy in the main repo — this may differ from what the `-develop` version
   implements if the main repo's interfaces have since evolved.
4. Port the logic into a new file (or addition to an existing file) in the
   main repo, rewriting as needed to:
   - satisfy the main repo's current abstract-type API exactly (not the
     `-develop` repo's possibly-outdated one),
   - follow the type-stability/memory/threading conventions above,
   - reuse the main repo's existing shared utilities (`@use_threads`,
     `filter_matrix!`, quadrature/sampler utilities, etc.) instead of any
     duplicate/older helpers the `-develop` version relied on,
   - match main-repo naming and file-organization conventions.
5. Wire the ported file in following the exact same steps as Use Case B
   (module include placed with its concept group, exports, abstract-type
   conformance, placement, downstream reuse).
6. Run `get_errors` on the new/changed files and the top-level module file to
   confirm the main repo still loads/compiles cleanly.
7. Report any behavioral or API differences between the `-develop` version
   and the ported version explicitly (e.g. "renamed field X to match current
   `AbsBasis` API", "dropped helper Y, replaced with existing `filter_matrix!`")
   so the user can verify the port preserves intended behavior.
8. If the ported feature is a simple, self-contained function you already
   verified by running it, you may add a small `@testset` for it yourself
   (never fabricate the expected value). For anything more involved
   (solver/basis/billiard-level behavior, regression tests needing
   hardcoded reference spectra), remind the user that test coverage should
   go through the Julia Test Writer agent.

## Constraints specific to this use case
- DO NOT edit any `-develop` repo as part of a port — it is a read-only
  reference; all edits land in the main repo only.
- DO NOT copy `-develop` code verbatim if it violates the main repo's
  conventions (type stability, threading, macros, file placement) — port the
  algorithm/logic, not the exact source text, and rewrite it to comply.
- Keep edits scoped to the ported file(s) plus their wiring points in the
  main repo — don't touch unrelated files.

## Output Format
Present findings as a concise categorized report (not a new markdown file):
- **Source reviewed** (`-develop` file(s) the port is based on)
- **API/behavioral differences** (vs. the `-develop` version, and why)
- **Module wiring** (include/export status in the main repo)
- **Abstract-type conformance** (missing methods, if any)
- **Placement** (correct folder or not)
- **Convention compliance** (in the ported file)
- **Reuse opportunities** (existing main-repo utilities used instead of `-develop` duplicates)

Each item: file/location reference, one-line issue, one-line suggestion.
If edits were applied, summarize exactly what changed after the report.
