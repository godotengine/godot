# Unified Benchmark Scene

## Purpose

Provide a single, deterministic, one-button benchmark run that demonstrates and stresses the major Gaussian Splatting systems in one pass:

- instance pipeline
- streaming
- lighting and post-processing/color grading
- LOD transitions
- wind/effector animation

The benchmark is report-only and ends with tuning suggestions based on measured behavior.

## Usage

### Editor / one-button

`tests/examples/godot/test_project/project.godot` now uses:

`run/main_scene="res://scenes/benchmark_unified.tscn"`

So pressing Play in the sample project launches the unified benchmark directly.

### CLI

```bash
./bin/godot.linuxbsd.editor.dev.x86_64 \
  --path tests/examples/godot/test_project \
  --scene res://scenes/benchmark_unified.tscn
```

Optional arguments:

- `--benchmark-duration=<seconds>`: override default duration (`180`).
- `--benchmark-output=<path>`: output JSON path (default `user://benchmark_unified_results.json`).
- `--benchmark-headless-summary`: print condensed summary and exit (useful for automation).

Example fast smoke run:

```bash
./bin/godot.linuxbsd.editor.dev.x86_64 \
  --path tests/examples/godot/test_project \
  --scene res://scenes/benchmark_unified.tscn \
  --benchmark-duration=30 \
  --benchmark-headless-summary
```

## Runtime Flow

The run is deterministic and camera-driven (no user input required):

1. Warmup
2. Instance
3. Streaming
4. Lighting
5. Effects
6. LOD
7. Finalize

At completion:

- an in-scene results panel shows score, per-phase metrics, and per-feature tuning suggestions.
- a JSON report is written to `--benchmark-output` (or default path).

## Output and Suggestions

The report includes:

- overall FPS/frame-time stability metrics (avg, P1, P99, max)
- per-phase summaries
- monitor maxima for key streaming/LOD/raster stats
- per-feature recommendation entries (`setting`, `current`, `suggested`, `reason`, `tradeoff`)

The benchmark does not hard-fail on low performance; it always reports and suggests.

## Small Baseline Comparison

For apples-to-apples high-FPS comparison against the heavy unified scene, run:

```bash
./bin/godot.linuxbsd.editor.dev.x86_64 \
  --path tests/examples/godot/test_project \
  --scene res://scenes/benchmark_small_baseline.tscn \
  --benchmark-duration=20 \
  --benchmark-headless-summary
```

Default report output is `user://benchmark_small_baseline_results.json`.

## Suite Runner

For multi-lane benchmark runs (scene set + aggregate report), use:

```bash
python3 tests/runtime/run_benchmark_suite.py --profile quick --generate-dummy-assets
```

See [docs/testing/benchmark-suite.md](benchmark-suite.md) for profiles, lane list, and asset override workflow.

## Controls

- `Esc`: quit
- `R`: rerun benchmark scene
