# Contribution Standards

These standards are mandatory for all merged changes.

## 1) Source of Truth Rules

- Use one canonical doc per topic.
- Do not commit internal/personal AI artifacts.
- Do not commit historical investigation dumps in docs.
- Keep IDE-local settings untracked (`.vscode/`, `.idea/`).

## 2) Code and Docs Quality

- Keep changes scoped and reviewable.
- Update user-facing docs when behavior changes.
- Keep commands runnable and platform-accurate.
- Run link checks for doc changes:
  - `python3 scripts/docs/check_links.py docs README.md BUILDING.md CONTRIBUTING.md`

## 3) Testing Expectations

- At minimum, run guard checks for code changes:
  - `python3 tests/ci/run_module_tests.py --guard-only`
- Run category-specific validation for affected subsystems.
- Include executed commands and outcomes in PR notes.

## 4) PR Checklist

- [ ] Scope is focused and justified.
- [ ] Canonical docs updated (if needed).
- [ ] Internal/personal artifacts are not included.
- [ ] Validation commands and outputs are included.
- [ ] CI-relevant workflows are not left in drift.
