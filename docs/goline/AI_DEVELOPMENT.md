# Goline — AI Development Rules

Strict rules for AI coding agents (such as OpenCode) contributing to Goline.
These complement `GOLINE.md` and apply to every change to this repository.

---

## 1. Inspect before modifying

- Fully understand a component, its callers, and its tests before changing it.
- Prefer targeted searches and reads over guesses.

## 2. Make small changes

- Keep each change small and logically separated.
- Small changes are easier to review, test, and roll back.

## 3. Never make unrelated changes

- Do not edit files or systems outside the scope of the current task.
- Keep the diff focused; unrelated edits risk regressions and obscure review.

## 4. Preserve existing Godot behavior

- Regressions are bugs. Assume upstream behavior is correct until proven
  otherwise.
- Prefer additive, backward-compatible changes.

## 5. Explain architectural changes

- If a change affects architecture or structure, explain *why* it is
  necessary.
- Do not modify the renderer, core, editor, scene system, or build system
  before a roadmap stage authorizes it.

## 6. Run appropriate tests after modifications

- Run the relevant test and build checks after making changes.
- Report what you ran and the result.

## 7. Never delete functionality just to make something work

- Do not remove or gut working features to satisfy a build or test.
- Fix the *cause* of a failure, not the symptom.

## 8. Never silently change project configuration

- Do not change build or project configuration without explaining it.
- Configuration changes must be explicit and reviewable.

## 9. Never execute destructive commands without explicit approval

- Do not run destructive commands (reset, revert, clean, force operations,
  etc.) without explicit user approval.

## 10. Stop and ask when requirements are ambiguous

- Do not guess. If a requirement or expected behavior is ambiguous, stop and
  ask the user rather than proceeding on an assumption.

---

These rules are mandatory for all AI agents working in this repository.
