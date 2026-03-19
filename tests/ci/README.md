# Baseline QA Automation

This folder contains maintained CI runners for Gaussian Splatting.

## Primary Entrypoints

- `tests/ci/run_baseline_qa.py`: orchestrates baseline lanes
- `tests/ci/run_module_tests.py`: static/render guards and module doctests

## Quick Start

```bash
python3 tests/ci/run_baseline_qa.py
python3 tests/ci/run_module_tests.py --guard-only
python3 tests/ci/validate_automation.py --contracts-only
```

## Static Contracts (`ISSUE-041`..`ISSUE-045`)

`tests/ci/validate_automation.py --contracts-only` enforces marker contracts for:

- painterly shader toggle evidence routing (`ISSUE-041`)
- unified sorter policy/traits hooks (`ISSUE-042`)
- theoretical complexity/scaling documentation anchors (`ISSUE-043`)
- clip blend self/cycle guard + depth cap anchors (`ISSUE-044`)
- incremental saver layout-version read/write contract anchors (`ISSUE-045`)

## Runtime Gate

```bash
python3 tests/runtime/run_runtime_validation.py --godot-binary "$GODOT_BINARY" --gd-mode headless --profile release-ci
```

## Production Workflow

Canonical GitHub Actions gate:

- `.github/workflows/gaussian_production_gates.yml`
