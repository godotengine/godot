# Local CI Helpers

This directory contains optional local scripts for running repeatable validation.

## Platform Scope

The local pipeline scripts in this directory are currently Windows-focused:

- `ci/local_pipeline.py`
- `ci/simple_pipeline.py`
- `ci/watch.py`

They invoke Windows build targets/binaries (for example `platform=windows` and `godot.windows.editor.x86_64.exe`).

## Common Commands (Windows)

```bash
python ci/local_pipeline.py
python ci/local_pipeline.py --quick
python ci/watch.py
```

```bat
ci\scripts\run_ci.bat
ci\scripts\quick_test.bat
ci\scripts\watch_files.bat
```

## Linux/macOS Note

Use the canonical Python test runners under `tests/ci` and `tests/runtime` from the repository root for local validation instead of the Windows-local pipeline wrappers in `ci/`.

## Dependencies

```bash
pip install -r ci/requirements.txt
```

## Canonical CI Source of Truth

Use `.github/workflows/` for authoritative CI gates and required checks.
