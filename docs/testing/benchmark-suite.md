# Benchmark Suite Runner

## Out-of-the-box Start (single path)

Use exactly this command for a deterministic, repository-only benchmark start:

```bash
python3 tests/runtime/run_benchmark_suite.py \
  --godot-binary ./bin/godot.linuxbsd.editor.dev.x86_64 \
  --project-path ./tests/examples/godot/test_project \
  --profile quick
```

Why this is the default path:

- benchmark and QA scenes now point to the versioned core fixture set under `res://tests/fixtures/`
- no external assets are required for the default quick run
- the runner performs a preflight dependency validation for `res://` references before launching lanes and fails early with a clear list when dependencies are missing

## Core Fixture Set

Fixture location: `tests/examples/godot/test_project/tests/fixtures/`

- Manifest: `core_fixture_set.json`
- Version: `1.0.0`
- Default asset: `test_splats.ply`
- Default world: `test_splats.gsplatworld`

## Optional Asset Provisioning Modes

Use these only when you intentionally benchmark with alternative datasets.

Manifest injection (lane-specific assets):

```bash
python3 tests/runtime/run_benchmark_suite.py \
  --profile quick \
  --asset-manifest tests/runtime/benchmark_assets_manifest.example.json
```

Synthetic asset generation:

```bash
python3 tests/runtime/run_benchmark_suite.py \
  --profile quick \
  --generate-dummy-assets
```

## Purpose

Run a profile-based set of benchmark lanes with one command, gather per-lane JSON reports, and produce an aggregate suite report.

## Profiles

- `quick`: short smoke profile
- `performance`: longer engineering profile
- `showcase`: longer visual/demo profile
- `parity`: fidelity-locked profile (requires GPU timestamp timing)

## Lane Set

Scene lanes (under `res://scenes/benchmark_suite/`):

- `lane_static_baseline.tscn`
- `lane_streaming_corridor.tscn`
- `lane_city_flyover.tscn`
- `lane_instance_storm.tscn`
- `lane_lighting_stress.tscn`
- `lane_animation_arena.tscn`
- `lane_lod_torture.tscn`
- `lane_integrity_sentinel.tscn`
- `lane_parity_fidelity.tscn`
- `lane_long_soak.tscn`

Composite lane:

- `res://scenes/benchmark_unified.tscn` (included by the Python runner as `unified_composite`)

## Common Usage

Run a subset of lanes:

```bash
python3 tests/runtime/run_benchmark_suite.py \
  --profile quick \
  --lane static_baseline \
  --lane lod_torture
```

Use custom duration scaling:

```bash
python3 tests/runtime/run_benchmark_suite.py --profile performance --duration-scale 0.5
```

Force lane instancing mode:

```bash
python3 tests/runtime/run_benchmark_suite.py \
  --profile performance \
  --lane instance_storm \
  --benchmark-instancing-mode serial
```

Capture benchmark screenshots and generate the HTML/SVG dashboard:

```bash
python3 tests/runtime/run_benchmark_suite.py \
  --profile quick \
  --capture-lane integrity_sentinel \
  --capture-lane parity_fidelity
```

Compare captured frames against golden references with SSIM/PSNR:

```bash
python3 tests/runtime/run_benchmark_suite.py \
  --profile parity \
  --reference-dir <reference-dir>
```

`<reference-dir>` must contain `<lane_id>/<lane_id>_capture*.png` matches for every selected capture lane.

Require GPU timestamps:

```bash
python3 tests/runtime/run_benchmark_suite.py \
  --profile performance \
  --require-gpu-timestamps
```

`parity` profile and `parity_fidelity` lane require GPU timestamps automatically.

## Outputs

Default output directory:

`tests/output/benchmark_suite/<timestamp>/`

Generated files:

- `<lane_id>.json`: per-lane benchmark report
- `<lane_id>.log`: per-lane stdout/stderr + executed command
- `benchmark_suite_report.json`: suite aggregate + lane list
- `benchmark_suite_summary.md`: human-readable summary table
- `captures/*.png`: captured benchmark frames for selected lanes
- `benchmark_suite_dashboard.html`: HTML dashboard with chart panels and capture gallery
- `benchmark_suite_*.svg`: SVG charts for score, FPS, GPU ms, SSIM, and PSNR

Notes:

- deterministic screenshot capture defaults to `integrity_sentinel` and `parity_fidelity`
- pass `--no-captures` to disable screenshot capture entirely
- when `--reference-dir` is provided, captured frames are compared against matching reference PNGs and the suite reports `SSIM`/`PSNR` minima per lane
