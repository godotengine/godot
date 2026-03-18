# Testing Setup Guide

## Purpose

Run Gaussian Splatting test lanes through maintained runners in `tests/ci/` and `tests/runtime/`.

## Usage

| Task | Command | Reference |
| --- | --- | --- |
| Build a test-enabled Godot binary | `scons platform=<platform> target=editor dev_build=yes tests=yes -j<jobs>` | [../../README.md](../../README.md), [../../BUILDING.md](../../BUILDING.md), `modules/gaussian_splatting/SCsub:69` |
| Set default Godot binary for Python runners | `export GODOT_BINARY=/path/to/godot` | `tests/ci/run_baseline_qa.py:763`, `tests/ci/run_module_tests.py:350`, `tests/runtime/run_runtime_validation.py:98` |
| Run quick baseline subset | `python3 tests/ci/run_baseline_qa.py --quick` | `tests/ci/run_baseline_qa.py:725`, `tests/ci/run_baseline_qa.py:284` |
| Run one baseline category | `python3 tests/ci/run_baseline_qa.py --category <ply|pipeline|sorting|runtime|module|qa>` | `tests/ci/run_baseline_qa.py:730`, `tests/ci/run_baseline_qa.py:281` |
| Run unified benchmark scene (report-only) | `godot --path tests/examples/godot/test_project --scene res://scenes/benchmark_unified.tscn --benchmark-headless-summary` | `tests/examples/godot/test_project/scenes/benchmark_unified.gd` |
| Run small baseline benchmark scene (high-FPS reference) | `godot --path tests/examples/godot/test_project --scene res://scenes/benchmark_small_baseline.tscn --benchmark-headless-summary` | `tests/examples/godot/test_project/scenes/benchmark_small_baseline.gd` |
| Run benchmark lane suite (profile runner) | `python3 tests/runtime/run_benchmark_suite.py --profile quick --generate-dummy-assets` | `tests/runtime/run_benchmark_suite.py`, `tests/examples/godot/test_project/scenes/benchmark_suite/` |
| Compare QA output against stored baseline (strict) | `python3 tests/ci/run_baseline_qa.py --category qa --qa-baseline tests/ci/baselines/qa_results.json --require-qa-baseline` | `tests/ci/run_baseline_qa.py:730`, `tests/ci/run_baseline_qa.py:735`, `tests/ci/run_baseline_qa.py:745`, `tests/ci/run_baseline_qa.py:796` |
| Refresh QA baseline snapshot from current run | `python3 tests/ci/run_baseline_qa.py --category qa --update-qa-baseline` | `tests/ci/run_baseline_qa.py:740`, `tests/ci/run_baseline_qa.py:456` |
| Run renderer/static guards without Godot tests | `python3 tests/ci/run_module_tests.py --guard-only` | `tests/ci/run_module_tests.py:354`, `tests/ci/run_module_tests.py:439` |
| Run module doctests after guards | `python3 tests/ci/run_module_tests.py --godot-binary "$GODOT_BINARY"` | `tests/ci/run_module_tests.py:350`, `tests/ci/run_module_tests.py:454` |
| Run runtime validation harnesses | `python3 tests/runtime/run_runtime_validation.py --godot-binary "$GODOT_BINARY" --gd-mode headless` | `tests/runtime/run_runtime_validation.py:98`, `tests/runtime/run_runtime_validation.py:103`, `tests/runtime/run_runtime_validation.py:957` |

| Baseline category | Executed target | Reference |
| --- | --- | --- |
| `ply` | `tests/ci/test_ply_loader_ci.gd` | `tests/ci/run_baseline_qa.py:233` |
| `pipeline` | `tests/ci/test_ply_pipeline_ci.gd` | `tests/ci/run_baseline_qa.py:239` |
| `sorting` | `tests/ci/test_gpu_sorting_ci.gd` | `tests/ci/run_baseline_qa.py:245` |
| `runtime` | `python3 tests/runtime/run_runtime_validation.py` | `tests/ci/run_baseline_qa.py:251` |
| `module` | `python3 tests/ci/run_module_tests.py` | `tests/ci/run_baseline_qa.py:258` |
| `qa` | `godot --headless --path tests/examples/godot/test_project --script res://scripts/qa_test_runner.gd --qa-output tests/ci/qa_results.json` | `tests/ci/run_baseline_qa.py:269`, `tests/ci/run_baseline_qa.py:271`, `tests/ci/run_baseline_qa.py:273` |

## API

| Entry point | Key options | Behavior | Artifacts | Reference |
| --- | --- | --- | --- | --- |
| `tests/ci/run_baseline_qa.py` | `--godot`, `--quick`, `--category`, `--qa-baseline`, `--update-qa-baseline`, `--require-qa-baseline`, `--baseline-report`, `--baseline-summary` | Orchestrates baseline categories and QA baseline compare/update flow. | `baseline_qa_results.json`, `tests/ci/qa_results.json`, `baseline_qa_regression_report.json`, `baseline_qa_regression_summary.md` | `tests/ci/run_baseline_qa.py:723`, `tests/ci/run_baseline_qa.py:735`, `tests/ci/run_baseline_qa.py:750` |
| `tests/ci/run_module_tests.py` | `--godot-binary`, `--base-ref`, `--guard-only`, `--skip-render-guards`, `--skip-static-guards`, `--tests-unavailable-mode`, `--allow-tests-unavailable` | Runs renderer guards and module doctests filtered by `*GaussianSplatting*`. | Console output | `tests/ci/run_module_tests.py:350`, `tests/ci/run_module_tests.py:354`, `tests/ci/run_module_tests.py:453` |
| `tests/runtime/run_runtime_validation.py` | `--godot-binary`, `--gd-mode`, `--skip-cpp`, `--skip-gd`, `--fail-on-skip`, `--allow-skips` | Runs C++ runtime harnesses and GDScript runtime harnesses with skip/fail policy. | `tests/runtime/runtime_validation_report.json` | `tests/runtime/run_runtime_validation.py:95`, `tests/runtime/run_runtime_validation.py:98`, `tests/runtime/run_runtime_validation.py:970` |
| `tests/run_gaussian_pipeline_tests.py` | `--godot`, `--category` (`ply|pipeline|sorting|runtime`) | Legacy wrapper that forwards to `run_baseline_qa.py`. | Baseline runner artifacts | `tests/run_gaussian_pipeline_tests.py:27`, `tests/run_gaussian_pipeline_tests.py:34` |
| `test_data/ci_test_runner.py` | `--godot`, `--docker`, `--quick`, `--category` (`ply|pipeline|sorting|runtime|module|qa`) | Legacy compatibility wrapper for baseline runner. | Baseline runner artifacts | `test_data/ci_test_runner.py:34`, `test_data/ci_test_runner.py:40` |
| `test_data/run_all_tests.py` | Pass-through args | Legacy pass-through wrapper for baseline runner. | Baseline runner artifacts | `test_data/run_all_tests.py:21` |

| Scripted entry point | Command surface | Reference |
| --- | --- | --- |
| `Makefile` | `make test` | `Makefile:29` |
| `package.json` | `npm run test` | `package.json:6` |
| `run_tests.bat` | Windows batch full module run | `run_tests.bat:24`, `run_tests.bat:31` |
| `ci/scripts/run_module_tests.bat` | Windows quick guard + module run | `ci/scripts/run_module_tests.bat:23`, `ci/scripts/run_module_tests.bat:47` |

## Examples

```bash
python3 tests/ci/run_baseline_qa.py --help
python3 tests/ci/run_module_tests.py --help
python3 tests/runtime/run_runtime_validation.py --help
python3 tests/run_gaussian_pipeline_tests.py --help
python3 test_data/ci_test_runner.py --help
python3 test_data/run_all_tests.py --help
python3 tests/ci/run_module_tests.py --guard-only
```

## Troubleshooting

| Symptom | Cause | Action | Reference |
| --- | --- | --- | --- |
| `Could not find Godot binary` | `GODOT_BINARY` is unset and no `godot` binary is on `PATH`. | Set `GODOT_BINARY` or pass `--godot`/`--godot-binary`. | `tests/ci/run_baseline_qa.py:776` |
| `--quick` is ignored | `--category` was also passed. | Run only one selector mode at a time. | `tests/ci/run_baseline_qa.py:781` |
| Module tests report disabled test runner | Binary was not built with `tests=yes`. | Rebuild the editor from repository root with `tests=yes` so the output in `bin/` includes the in-tree module and test runner. | `tests/ci/run_module_tests.py:459`, `modules/gaussian_splatting/SCsub:69` |
| Pipeline wrapper rejects `module` or `qa` category | `tests/run_gaussian_pipeline_tests.py` only exposes `ply|pipeline|sorting|runtime`. | Use `tests/ci/run_baseline_qa.py` (or `test_data/ci_test_runner.py`) for `module` and `qa`. | `tests/run_gaussian_pipeline_tests.py:28`, `tests/ci/run_baseline_qa.py:731`, `test_data/ci_test_runner.py:40` |
| Runtime run fails on skips in non-headless mode | Default skip policy is strict outside headless mode. | Add `--allow-skips` or switch to `--gd-mode headless`. | `tests/runtime/run_runtime_validation.py:916`, `tests/runtime/run_runtime_validation.py:920` |
