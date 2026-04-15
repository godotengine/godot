# Timestamp Monitor Investigation — `gpu_time_raster_ms` reads 0

**Status**: open, partial fix applied (commit removing one of the per-frame syncronous readbacks).
**Date**: 2026-04-15
**Branch context**: investigated on `test/tier2-pr232-plus-235` (≈ master + tier-2 + #232).

## Symptom

`Performance.get_custom_monitor("gaussian_splatting/gpu_time_raster_ms")` reads `0.000` on direct-node scenes (GaussianSplatNode3D, no streaming world). `gpu_time_binning_ms`, `gpu_time_prefix_ms`, `gpu_time_resolve_ms` likely also zero (only raster was probed).

`gpu_time_sort_ms` and `gpu_time_cull_ms` work — those use a different timing source (sort pipeline / culler), not the tile-renderer timestamp pool.

## What is confirmed

Verified via temporary `print_line()` diagnostics (since reverted):

| Check | Result |
|---|---|
| `TileRenderer::resolve_gpu_timestamps_async()` called every frame | yes |
| `gpu_timestamp_capture_enabled` | true (default) |
| `_get_submission_device()` returns a valid device | yes |
| `device->get_captured_timestamps_count()` at resolve time | **0 every frame** |

Function exits at the early-return at `tile_renderer.cpp:2384` (`if (available == 0) return;`) before ever entering `_compute_stage_durations()`.

## What is mechanically true in the code

- Raster stage emits `capture_timestamp("TileRaster_<serial>_Begin/_End")` at `tile_render_rasterizer_stage.cpp:86/114` (compute path) and `:149/227` (graphics path).
- The capture device is `submission_device` = `_acquire_submission_device()` = `_get_submission_device()` — same device the resolve queries. Not a device-mismatch bug.
- `_reset_timestamp_tracking()` at `tile_renderer.cpp:2175-2193` keeps previous values when `gpu_timestamp_capture_enabled=true`, so stale values would persist if any capture had succeeded — they haven't.
- The code itself flags the cause at `tile_renderer.cpp:495-497`:
  > "GaussianSplat_Begin/End timestamps are NOT captured because buffer_get_data() in the prefix sum forces a flush that resets the timestamp buffer. Only TileOverlapCount markers (pre-flush) can be read."

## What was tried and ruled out

- **`active_renderer` registration**: `GaussianSplattingPerformanceMonitors::register_renderer(TileRenderer*)` is never called in production code (only in tests at `tests/test_diagnostics.h`). So the monitor's "direct" path (`active_renderer->get_last_gpu_raster_time_ms()`) is always 0 and the monitor depends entirely on the snapshot fallback. **Not the root cause** — fallback chain is wired correctly.
- **Calling `resolve_gpu_timestamps_async()` from the monitor getter**: tried and reverted. Caused per-poll GPU sync, halving FPS. Do not repeat.
- **Removing per-frame `sorted_indices_preview` readback** (`render_diagnostics_orchestrator.cpp:1002`): committed as a separate perf win. Did **not** restore raster_ms — there's at least one more in-frame flush wiping the timestamp buffer.
- **Prefix-scan policy-gated readbacks** (`tile_render_prefix_scan.cpp:230/418/65/558/579/603`): all default-off and not overridden in the GH project. Not the source.

## Hypotheses still open

1. Some other in-frame `buffer_get_data()` or device-flushing operation between raster `capture_timestamp()` and end-of-frame `resolve_gpu_timestamps_async()` query. Need to enumerate every device-touching call on the per-frame path post-raster.
2. Multi-context boundary: the local compute device may not actually share the timestamp buffer with whatever device returns from `_get_submission_device()` at resolve time, even though the pointers compare equal in our diagnostic. Worth checking whether device pointer identity → timestamp pool identity in Godot's RD layer.
3. Timestamp buffer is reset by Godot at frame boundaries before our resolve runs — possible if `update_gpu_pass_metrics_from_tile_renderer()` runs in a phase where the previous frame's captures have already been wiped.

## Why this matters

Without `raster_ms` (and the other tile-renderer monitors), all subsequent GPU-side optimization work is blind. The dream scene runs at ~60 FPS with 227k visible splats, and we need to know whether the bulk of the 16ms is in raster, in CPU setup, or in something else entirely. Codex audits identified candidate wins (per-pixel SSBO chain, 40KB shared-memory occupancy throttle, cross-device sort sync, resident contract republish), but we can't quantify their impact without the monitor.

## Next investigation steps

- **Enumerate all per-frame device-touching calls** on the tile_renderer path after raster capture. Targets: `buffer_get_data`, `safe_sync`, `safe_submit_and_sync`, `_flush_pending_submission(true)`, `compute_list_add_barrier` of full-barrier type, `submit`, `sync`. Categorize: which fire by default, which only on debug/sync paths.
- **Cross-check device identity** between capture site and resolve site: log `submission_device` pointer at both call sites; if they're different objects despite both coming from `_get_submission_device()`, that's the bug.
- **Try a synthetic capture-then-immediately-query** to isolate whether timestamps survive AT ALL on this path, independent of the raster pipeline. If a degenerate `capture("test_Begin"); capture("test_End"); get_count()` returns 0, the timestamp pool is being wiped by something further upstream than we think.

## Related codex audit findings (not the monitor itself, but in the same area)

- `gpu_sorting_pipeline.cpp:2604-2609` — `safe_submit_and_sync(compute_rd)` before `sort_async()` on cross-device sort. Per-frame full sync. Likely candidate for "what wipes the timestamp buffer" if cross-device path is active.
- `tile_render_resolve.cpp:1243-1249` — `safe_sync()` after resolve dispatch on non-main RD. Another flush.
- `tile_render_prefix_scan.cpp:417/499/603` — sync readbacks on policy-gated paths.

## Workaround until fixed

Use frame-time variance (`Engine.get_frames_per_second()` over a sample window) and CPU-side timing markers (e.g. `Time.get_ticks_usec()` around code regions) to estimate where time goes. Imprecise, but the only signal we have on this path right now.
