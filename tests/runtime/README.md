# Runtime Validation Harness

`tests/runtime/run_runtime_validation.py` runs runtime harnesses (C++ and GDScript) and writes:

- `tests/runtime/runtime_validation_report.json`

## Scenario Profiles

Runtime scenarios are defined declaratively in:

- `tests/runtime/runtime_scenarios.json`

The canonical release profile is `release-ci` (default).  
List profiles:

```bash
python3 tests/runtime/run_runtime_validation.py --list-profiles
```

Run a specific profile:

```bash
python3 tests/runtime/run_runtime_validation.py --profile stress-only
```

Override profile selection with explicit tests:

```bash
python3 tests/runtime/run_runtime_validation.py \
  --profile release-ci \
  --gd-test "GPU Streaming Stress" \
  --cpp-test "Runtime Modifications"
```

Use explicit script paths instead of named GDS tests (mutually exclusive with `--gd-test`):

```bash
python3 tests/runtime/run_runtime_validation.py \
  --gd-script tests/runtime/test_gpu_streaming_stress.gd
```

## C++ Harness Link Modes

By default, C++ runtime harnesses compile in `standalone` mode (mock-only).

To exercise module-linked harness builds, pass `--cpp-link-mode module-linked`
and a JSON manifest via `--cpp-build-manifest`:

```bash
python3 tests/runtime/run_runtime_validation.py \
  --cpp-link-mode module-linked \
  --cpp-build-manifest tests/runtime/module_link_manifest.json \
  --skip-gd
```

Manifest schema:

```json
{
  "include_dirs": [".", "modules/gaussian_splatting"],
  "compile_flags": ["-DDEBUG_ENABLED"],
  "link_flags": ["-L/path/to/libs", "-lgaussian_splatting_runtime"]
}
```

## Schema Validation

The runner validates the generated summary payload structure before exit.

- `schema_valid=true` and `schema_errors=[]` are required for a successful run.
- Schema validation failure returns a non-zero exit code.

This makes report-shape regressions fail in CI instead of silently passing.

## CI Integration

`tests/ci/run_baseline_qa.py` executes the runtime gate with:

- `--profile release-ci`

That keeps CI pinned to the canonical, declarative release-ready scenario set.

## Synthetic Asset Prep

Runtime and benchmark scenes depend on deterministic synthetic fixtures.

Generate/update them:

```bash
python3 tests/runtime/prepare_synthetic_assets.py --quiet
```

Validate canonical fixture policy:

```bash
python3 tests/runtime/prepare_synthetic_assets.py --check
```

Canonical generated PLY paths:

- `tests/fixtures/test_splats.ply`
- `tests/examples/godot/test_project/tests/fixtures/test_splats.ply`
- `templates/gaussian_splat_template/assets/template_splats.ply`
