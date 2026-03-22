# Gaussian Splatting Build and Test Guide

This module supports the same desktop targets declared in `modules/gaussian_splatting/config.py`:

- `windows`
- `linuxbsd`
- `macos`

## Build (SCons)

Run from repository root (`/mnt/c/projects/godotgs` in this workspace):

### Windows

```powershell
scons platform=windows target=editor dev_build=yes modules/gaussian_splatting
```

### Linux

```bash
scons platform=linuxbsd target=editor dev_build=yes modules/gaussian_splatting
```

### macOS

```bash
scons platform=macos target=editor dev_build=yes modules/gaussian_splatting
```

## Validation Guards (Fast, Cross-Platform)

These checks do not require a full editor build and should be run before opening a PR:

```bash
python3 modules/gaussian_splatting/tests/check_build_metadata_consistency.py
python3 tests/ci/run_module_tests.py --guard-only
```

The metadata guard enforces:

- SCsub source directories stay aligned with CMake IDE metadata.
- CMake does not reference missing source directories.
- `config.py` doc contracts map to real XML files in `doc_classes/`.

## Module Test Runner

`tests/ci/run_module_tests.py` is the canonical module test/guard entry point used by CI.

### Auto-detect `godot` in PATH

```bash
python3 tests/ci/run_module_tests.py
```

### Explicit binary path

```bash
python3 tests/ci/run_module_tests.py --godot-binary /path/to/godot-editor
```

On Windows, the explicit path is typically `bin\\godot.windows.editor.dev.x86_64.exe` (the `.dev` segment is added by `dev_build=yes`).

## Convenience Wrappers

- `run_tests.bat` (repository root) remains a Windows convenience wrapper.
- `tests/ci/run_module_tests.py` remains the cross-platform canonical command.
