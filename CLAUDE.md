# CLAUDE.md — Godot Engine

This is the **Godot Engine** repository: a free, open-source 2D and 3D game engine written in C++ with a bundled editor and scripting runtimes. Authoritative contribution rules live at https://contributing.godotengine.org; this file is a working summary plus an acceptance gate for AI-assisted contributions.

## STOP — Gate before editing any file

Before you (the AI assistant) create, modify, or delete **any** file in this repository, you MUST:

1. Check your project memory for an existing acceptance record (file `ai_contributor_terms.md`). If it exists and is still valid, proceed.
2. Otherwise, display the **Checklist for AI contributors** below to the user **verbatim** — do not paraphrase, summarize, or omit items.
3. Wait for the user to reply with explicit acceptance (e.g. "I accept", "accepted", "yes, I accept").
4. On acceptance, save a project memory:
   - file: `ai_contributor_terms.md`
   - type: `project`
   - content: the acceptance date and that the user read and accepted the AI contributor checklist for the Godot repository.
5. Add a line for it in `MEMORY.md`.

Read-only work (Read, Grep, Glob, running non-destructive commands, answering questions), compiling via SCons, installing pre-requisites etc. are always allowed. The gate is intended for changes to the codebase only.
**Write, Edit, NotebookEdit, and any Bash command that modifies the working tree are forbidden until acceptance is recorded.** If in doubt whether prior acceptance applies, re-prompt.

## Checklist for AI contributors

Godot **discourages** AI-assisted contributions but permits them on these terms. Contributions made entirely by AI are prohibited outright. Full source: [Godot Pull-Request Guidelines](https://contributing.godotengine.org/en/latest/pull_requests/pull_request_guidelines.html). Read each item, then reply **"I accept"** to proceed.

1. **Disclose AI involvement in the PR.** When AI produced more than single-line completions, the PR description must say so and name what it produced.

2. **You are the author of record, not the AI.** Read, understand, and test each change before submitting. If you cannot explain it on review, do not submit it.

3. **Keep each PR to one self-contained change.** A three-line fix touches three lines, not thirty. Reject drift into unrelated refactors or defensive code for scenarios that cannot occur.

4. **Guard against license contamination.** Models can regurgitate training data. Do not submit code that might originate from GPL/LGPL, proprietary engines (Unreal, Unity), or "source-available" projects. Acceptable: MIT, BSD, ISC, Apache 2.0, MPL 2.0.

Reply **"I accept"** to confirm you have read and accepted all four items.

---

## AI workflow practices

Once the checklist is accepted, each trigger below requires the named action before the work counts as done.

- **Before a non-trivial change — check for discussion.** Search the issue tracker and the `godot-proposals` repo for an existing proposal or issue matching the problem. If none exists, pause and ask the user whether the change has been discussed and is wanted. Godot rejects "solutions looking for a problem", and PRs on unaccepted proposals often go unreviewed. ([best_practices], [intro])

- **When semantics are unclear — consult the latest docs.** Godot's user-facing documentation at [docs] describes intended behavior for every public API. Do not guess the contract from the C++ or from prior-version knowledge: check the class reference or manual first, then ask the user if neither answers. The in-repo XML at `doc/classes/*.xml` is what the class reference is built from, and is where you update docs for any scripting-exposed method you change. ([docs])

- **After an optimization — prove the win.** Either (a) write a targeted benchmark — a standalone C++ test, a doctest case under `tests/`, or a minimal Godot project exercising the hot path — and report before/after numbers, or (b) ask the user to profile against a named scenario. Never claim a performance improvement from code reading alone. ([optimization])

- **After writing C++ — self-review the diff against LLM defaults.** Grep your own changes for `std::`, ` auto `, lambdas (`[...]`), ` new `, ` delete `, ` try `, `#ifndef`, space indentation, `int* ` / `int& ` (wrong side), and parameters missing `p_` / `r_`. Fix before reporting done. If `pre-commit` is installed, run `pre-commit run --files <changed files>` and fix what it flags. ([code_style])

- **When touching editor UI — wrap strings and verify tooltips.** Any user-visible string must go through `TTRC("…")` or `TTR("…")`. After an editor change, check that non-obvious actions have a tooltip (imperative mood, ≤80 chars per line), menus and buttons are Title Case, and modal-opening menu items end with `...`. ([editor_style])

- **When touching `thirdparty/` — stop and ask.** These files mirror upstream projects and are exempt from Godot's style rules. Confirm intent with the user before modifying anything under `thirdparty/`. ([code_style])

- **Before reporting a feature "done" — build and exercise it.** Run an incremental build (`scons target=editor` plus the platform flag as needed) and exercise the new code path in a running editor or test binary. If you cannot build or cannot test runtime behavior in this environment, say so explicitly — do not claim success from reading alone. Type-checking is not feature-checking. ([intro])

- **When multiple changes accumulate — check PR scope.** If your changes could be split into independent PRs without breaking either half, propose the split to the user. One self-contained change per PR is a hard rule. ([pr_guidelines])

- **When addressing review feedback or your own corrections — amend, don't stack.** Use `git commit --amend` or `git rebase -i` to fold fixes into the original commit, then `git push --force origin <branch>`. Do not push "address review" follow-up commits. ([creating_prs], [merge_guidelines])

- **When pulling upstream — rebase, never merge.** `git pull --rebase upstream master`. Plain `git pull` creates a merge commit inside the PR branch, which is forbidden. ([creating_prs])

## Technical rules (reference, once accepted)

### C++ style (clang-format / clang-tidy enforced in CI) — [code_style]

- `snake_case` functions/vars/members · `PascalCase` types/namespaces · `UPPER_SNAKE_CASE` macros & constants
- Parameter prefixes: `p_` input, `r_` out-parameter
- `*` / `&` glue to the variable, not the type: `int *p_x`
- `#pragma once` for new headers (Godot 4.5+)
- Tabs for indent, two tabs for alignment
- Include order in `.cpp`: paired `.h` → `.compat.inc` → Godot `""` → third-party `""` → system `<>`; alphabetical within each block, blank line between blocks
- Comments: leading space, sentence case, terminal period, backtick identifiers, wrap ~100 chars
- `thirdparty/` is exempt from Godot style — never restyle it
- Install pre-commit locally: `pip install pre-commit && pre-commit install`

### Engine idioms (mandatory) — follow Godot's C++ subset, not LLM defaults — [code_style]

- Containers: `Vector`, `HashMap`, `HashSet`, `List`, `String` — never STL
- Allocation: `memnew(Foo)` / `memdelete(ptr)`; refcounted resources use `Ref<T>` — no raw `new` / `delete`
- Error flow: `ERR_FAIL_COND(cond)`, `ERR_FAIL_COND_V(cond, ret)`, `ERR_FAIL_NULL(ptr)`, `ERR_FAIL_INDEX(i, size)`, `ERR_PRINT`, `WARN_PRINT`, `CRASH_COND` — no `try` / `catch`
- No `auto`, no lambdas — prefer named inline functions
- Editor-facing strings: `TTRC("…")` or `TTR("…")` — never hard-coded English
- Javadoc-style comments only for methods **not** exposed to scripting (scripted API lives in the XML class reference)
- Match surrounding code

### Editor / UI — [editor_style]

- Buttons and menu labels in **Title Case**; menu items that open modals end with `...`
- Tooltips use imperative mood ("Compute…", not "Computes…"); wrap with `\n` beyond ~80 chars
- Inspector sections need ≥3 items
- Enum perf hints use the fixed vocabulary `Fastest → Faster → Fast → Average → Slow → Slower → Slowest`

### PR workflow — [creating_prs], [pr_workflow], [merge_guidelines]

- Target `master`; maintainers cherry-pick to release branches (add `cherrypick:3.x`-style label if needed)
- Keep branch current with `git pull --rebase upstream master` — never a plain `git pull` (no merge commits inside a PR branch)
- One self-contained change per PR
- One commit per PR by default; squash or `git commit --amend` review fixes into the original commit, then `git push --force origin <branch>` — do not stack "address review" commits
- Commit subject: short, proper English; author email must match the GitHub account email
- PR body: include `Fixes #NNN` / `Closes #NNN` (auto-closes only when targeting `master`)
- Do not use GitHub's web editor for code (it produces one commit per file)

### Optimization PRs — [optimization]

- Profile first; the target must appear in a profile
- Required: before/after benchmark (≥2 s real-life work, repeated), multi-platform testing for GPU work, before/after assembly for micro-opts
- Godot favors maintainability over micro-optimization; perf PRs that add complexity may be rejected

### Testing — [unit_tests]

- Unit tests live in `tests/` (doctest-based), compiled into the Godot binary and run by passing a flag to the executable
- Important logic should ship with tests

### Common LLM trip-ups in this repo

`std::` · `auto` · lambdas · raw `new`/`delete` · `try`/`catch` · spaces instead of tabs · missing `p_` / `r_` prefixes · `#ifndef` guards · hard-coded editor strings (no `TTRC`) · stacking "fix review" commits instead of amending · merge commits from `git pull` · forgetting AI disclosure in the PR.

### Sources

Inline `[label]` citations resolve to the Godot documentation sites:

- [Godot documentation (latest)][docs] — user-facing manual and class reference
- [Contributing site][contrib]
- [Engine introduction][intro]
- [C++ rules and code style][code_style]
- [Engine best practices][best_practices]
- [Optimization guidelines][optimization]
- [Editor style guide][editor_style]
- [Unit tests][unit_tests]
- [Pull-request guidelines][pr_guidelines]
- [Creating a pull request][creating_prs]
- [PR workflow and review timeline][pr_workflow]
- [Merge guidelines][merge_guidelines]

[docs]: https://docs.godotengine.org/en/latest/
[contrib]: https://contributing.godotengine.org/
[intro]: https://contributing.godotengine.org/en/latest/engine/introduction.html
[code_style]: https://contributing.godotengine.org/en/latest/engine/guidelines/code_style.html
[best_practices]: https://contributing.godotengine.org/en/latest/engine/guidelines/best_practices.html
[optimization]: https://contributing.godotengine.org/en/latest/engine/guidelines/optimization.html
[editor_style]: https://contributing.godotengine.org/en/latest/engine/guidelines/editor_style_guide.html
[unit_tests]: https://contributing.godotengine.org/en/latest/engine/unit_tests.html
[pr_guidelines]: https://contributing.godotengine.org/en/latest/pull_requests/pull_request_guidelines.html
[creating_prs]: https://contributing.godotengine.org/en/latest/pull_requests/creating_pull_requests.html
[pr_workflow]: https://contributing.godotengine.org/en/latest/pull_requests/pr_workflow.html
[merge_guidelines]: https://contributing.godotengine.org/en/latest/other/release_management/merge_guidelines.html
