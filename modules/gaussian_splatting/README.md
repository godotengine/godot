# Gaussian Splatting Module

This directory contains the Godot engine module implementation.

## Core Entry Points

- Registration and module lifecycle: `register_types.cpp`
- Scene node APIs: `nodes/`
- Renderer pipeline: `renderer/`
- Data and manager systems: `core/`
- Import/load/save paths: `io/`
- GLSL shader sources: `shaders/`
- Runtime and integration tests: `tests/`

## Build Integration

Build through the bundled Godot source tree:

```bash
scons -C ../.. platform=<windows|linuxbsd|macos> target=editor dev_build=yes modules/gaussian_splatting
```

Use repository-level build docs for full platform details:

- `../../BUILDING.md`
- `../../docs/BUILDING.md`
- `docs/BUILD_AND_TEST.md`

## Validation

Run baseline and module guards from repository root:

```bash
python3 modules/gaussian_splatting/tests/check_build_metadata_consistency.py
python3 tests/ci/run_baseline_qa.py
python3 tests/ci/run_module_tests.py --guard-only
```
