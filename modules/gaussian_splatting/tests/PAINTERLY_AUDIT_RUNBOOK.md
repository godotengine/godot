# Painterly Audit and Demo-Readiness Runbook

Use this runbook when validating issue `#815` scope (stability/perf audit + demo readiness) without adding new painterly features.

## 1) Prerequisites

- Godot binary in `PATH` (or pass full path in commands below).
- Vulkan/RenderingDevice available for shader permutation coverage.
- Run from repository root.

## 2) Baseline CI guard checks

```bash
python3 tests/ci/run_module_tests.py --guard-only
```

Expected result:
- Static/render guards pass.
- Command exits `0`.

## 3) Run painterly regression audit

Recommended reproducible invocation:

```bash
godot --headless --script scripts/tools/run_painterly_regression.gd -- \
  --image-size=128x128 \
  --artifact-dir=user://painterly_audit/regression \
  --summary-path=user://painterly_audit/regression/summary.json \
  --require-rendering-device \
  --perf-budget-ms=160
```

Outputs:
- Console marker `PAINTERLY_TEST_PASSED` on success, `PAINTERLY_TEST_FAILED` on failure.
- Per-scene PNG artifacts in `user://painterly_audit/regression/<scene>/`.
- JSON summary at `user://painterly_audit/regression/summary.json` with:
  - per-scene coverage/luminance/delta/stability metrics
  - shader permutation compile status
  - per-scene timings (`compile`, `render_camera0`, comparison render, total)
  - warnings/failures list

Notes:
- Performance budgets are warning-only unless `--enforce-performance-budget` is provided.
- Thresholds are defined per scene in `modules/gaussian_splatting/tests/painterly_scenes/*.json` under `audit`.

## 4) Refresh painterly reference captures

```bash
godot --headless --script scripts/tools/capture_painterly_references.gd -- \
  --image-size=256x256 \
  --output-dir=res://modules/gaussian_splatting/tests/painterly_scenes/references \
  --manifest-path=res://modules/gaussian_splatting/tests/painterly_scenes/references/manifest.json \
  --require-rendering-device
```

Outputs:
- Reference PNGs per configured camera index for each scene.
- Manifest `manifest.json` with captured files, compile results, and per-camera image stats.
- Console marker `PAINTERLY_REFERENCE_CAPTURE_COMPLETE` on success.

## 5) Demo-readiness checklist

1. Run steps 2-4 with no failures.
2. Confirm all scene definitions in `modules/gaussian_splatting/tests/painterly_scenes/*.json` report non-empty artifacts and no shader compile failures.
3. Spot-check demo scenes in editor:
   - `modules/gaussian_splatting/tests/painterly_scenes/dense_painterly_demo.tscn`
   - `modules/gaussian_splatting/tests/painterly_scenes/sparse_painterly_demo.tscn`
   - `modules/gaussian_splatting/tests/painterly_scenes/animated_painterly_demo.tscn`
4. Run smoke toggle script in test project and require `PAINTERLY_TEST_PASSED` output:
   - `godot --headless --path tests/examples/godot/test_project --script res://scenes/painterly_test.gd`
5. Archive the regression summary JSON + reference manifest with the demo handoff notes.

## 6) Known limitations

- Headless metrics are synthetic and deterministic; they are a stability gate, not a replacement for final visual art review.
- Timing budgets are machine-dependent and should be treated as regression signals, not absolute performance guarantees.
