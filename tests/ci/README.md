# Baseline QA Automation

This folder contains maintained CI runners for Gaussian Splatting.

## Primary Entrypoints

- `tests/ci/run_baseline_qa.py`: orchestrates baseline lanes
- `tests/ci/run_module_tests.py`: static/render guards and module doctests

## Quick Start

```bash
python3 tests/ci/run_baseline_qa.py
python3 tests/ci/run_module_tests.py --guard-only
```

## Runtime Gate

```bash
python3 tests/runtime/run_runtime_validation.py --godot-binary "$GODOT_BINARY" --gd-mode headless --profile release-ci
```

## Production Workflow

Canonical GitHub Actions gate:

- `.github/workflows/gaussian_production_gates.yml`
