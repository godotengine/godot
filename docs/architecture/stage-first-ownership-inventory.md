# Stage-First Ownership Inventory

W0.1 artifact for the stage-first decomposition plan.

This document records the current ownership boundary, single-writer rules, and hidden mutable state for the four highest-value decomposition targets:

- `GaussianStreamingSystem`
- `TileRenderer`
- `GPUSortingPipeline`
- `GaussianSplatRenderer`

The goal is to make the current implementation readable as an ownership map before code is moved. This is not the target end-state; it is the contract the refactor has to preserve while W1 and W2 land.

## Ownership Rules

- One subsystem owns each mutable state bucket.
- Worker threads may produce payloads, but they do not mutate renderer-owned state directly.
- Render-thread dispatch, async readback, and resource tracking must have a single authoritative owner.
- Hidden process-global caches are temporary debt unless they are explicitly assigned to an owner.

## Current Ownership Matrix

| System | Current owner | Mutable state buckets | Single-writer rule | Extraction risk |
| --- | --- | --- | --- | --- |
| `GaussianStreamingSystem` | `core/gaussian_streaming.*` | `VisibilityState`, `EvictionState`, `SchedulerState`, `DiagnosticsState`, `GlobalAtlasState`, `ConfigOverrides`, `PackTelemetry`, `frame_data`, `uploads`, `budget`, `chunks`, `atlas_allocator`, quantization buffers, `memory_stream_proxy`, `last_upload_device` | Main streaming update path owns frame ordering, visibility, eviction, upload scheduling, and atlas sync. Pack workers only build payloads for queued jobs and do not own streaming state. | High. Configuration overrides, frame ordering, and atlas sync are all coupled in `_run_streaming_frame_pipeline()` and `_apply_config_overrides()`. |
| `TileRenderer` | `renderer/tile_renderer.*` plus its stage helpers | `render_settings`, `shader_resources`, `diagnostics`, `async_readback`, `tracked_device_manager`, tracked output RIDs, `adaptive_overlap_budget` runtime state, subgroup-support cache | The renderer owns tile frame execution. Readback callbacks may update the readback state machine, but they must not take ownership of resource lifetime or output tracking. | High. There are still process-global mutable caches keyed by renderer/device identity. |
| `GPUSortingPipeline` | `interfaces/gpu_sorting_pipeline.*` | Sort buffer RIDs and capacities, depth compute RIDs, CPU byte buffers, `SortReadbackState`, `InstanceCountReadbackState`, `InstancePipelineInputs`, cached camera transforms, `last_compute_error` | The pipeline owns its sort buffers, depth resources, and readback generations. Renderer state must be reached through an explicit seam, not a raw pointer handoff. | High. The current implementation still mutates renderer-owned state directly in `_apply_sorted_results()`. |
| `GaussianSplatRenderer` | `renderer/gaussian_splat_renderer.*` | `DeviceState`, `ResourceState`, `PerformanceState`, `PipelineState`, `FrameState`, `ViewState`, `SortingState`, `StreamingState`, `SceneState`, `SubsystemState`, `RenderFrameContextManager`, render-thread dispatch primitives, orchestrator pointers | The renderer owns frame entry, orchestration, and Godot-facing API. Concrete subsystem mutation should happen through orchestrators or service objects, not through direct cross-subsystem writes. | Medium. The facade is already partly decomposed, but it still owns render-thread dispatch and several mutable caches. |

## Subsystem Details

### GaussianStreamingSystem

Current state is already split into named buckets in `core/gaussian_streaming.h`:

- `VisibilityState` at `gaussian_streaming.h:215`
- `EvictionState` at `gaussian_streaming.h:264`
- `PackTelemetry` at `gaussian_streaming.h:294`
- `SchedulerState` at `gaussian_streaming.h:584`
- `DiagnosticsState` at `gaussian_streaming.h:642`
- atlas and quantization state at `gaussian_streaming.h:706` through `gaussian_streaming.h:739`

Current write ownership:

- `VisibilityState` is updated from the main streaming frame pipeline.
- `EvictionState` is updated by eviction helpers only.
- `SchedulerState` is updated by the per-frame orchestration path.
- `GlobalAtlasState` and atlas RIDs are updated only at the end of the streaming frame pipeline.
- Pack workers only create or transform `PackJob` and `PendingChunkUpload` payloads, then hand results back to the system.

Implementation anchors:

- Frame ordering is explicit in `_run_streaming_frame_pipeline()` and ends with `_sync_global_atlas_state()`.
- Runtime config overrides are applied in `_apply_config_overrides()`, where they touch `visibility`, `uploads`, and `budget` together.

Refactor note:

- W1 and W2 should not split `VisibilityState`, `EvictionState`, or `SchedulerState` until the ownership matrix is stable and the frame-order tests are pinned.

### TileRenderer

Current state is a mix of member-owned state and process-global caches:

- Adaptive overlap runtime state is currently kept in a static pointer-keyed map in `tile_renderer.cpp:139` through `tile_renderer.cpp:157`
- Subgroup support is cached in a static device-id keyed cache in `tile_renderer.cpp:1927` through `tile_renderer.cpp:1947`
- Resource ownership and tracking are handled by `_resolve_texture_owner()`, `track_output_resources()`, and `clear_output_resource_tracking()`

Current write ownership:

- The tile renderer owns the frame execution path and the tile stage helpers.
- `adaptive_overlap_budget_runtime_states` is a temporary process-global cache keyed by `TileRenderer *`.
- `subgroup_support_cache` is a temporary process-global cache keyed by rendering device ID.
- Output RID tracking belongs to the renderer instance and the `RenderDeviceManager`, not to the caller.

Implementation anchors:

- `track_output_resources()` and `clear_output_resource_tracking()` own the current output-RID lifecycle.
- `_resolve_texture_owner()` is the contract gate for cross-device resource ownership.

Refactor note:

- W1 should extract the tile resource controller before W2 expands the execution-path split.
- The current static caches are temporary debt and should not survive the W1/W2 migration.

### GPUSortingPipeline

Current state in `interfaces/gpu_sorting_pipeline.h` is concentrated but still too coupled:

- Sort buffers and sort capacity live at `gpu_sorting_pipeline.h:132` through `gpu_sorting_pipeline.h:145`
- Depth compute resources live at `gpu_sorting_pipeline.h:150` through `gpu_sorting_pipeline.h:176`
- CPU-side sort and depth buffers live at `gpu_sorting_pipeline.h:178` through `gpu_sorting_pipeline.h:183`
- `SortReadbackState` and `InstanceCountReadbackState` live at `gpu_sorting_pipeline.h:197` through `gpu_sorting_pipeline.h:212`
- `pending_renderer` lives at `gpu_sorting_pipeline.h:213`

Current write ownership:

- The pipeline owns its RIDs, capacities, and readback generation counters.
- `sort_readback_state` controls publication of sorted results.
- `instance_count_readback_state` controls publication of instance count readbacks.
- Renderer-owned state is currently mutated inside `_apply_sorted_results()` and reached via `pending_renderer`.

Implementation anchors:

- `_apply_sorted_results()` writes through `GaussianSplatRenderer` state accessors and updates culler, sorting, frame, and performance state in one step.
- `ensure_sort_buffers()` still takes a renderer pointer and reaches into renderer state to validate/size resources.
- `clear_instance_pipeline_inputs()` and the readback reset path are the current state reset boundary for instance-pipeline work.

Refactor note:

- W1 should replace the renderer pointer seam with `ISortResultSink` plus a minimal host context.
- W2 should extract the sort buffer, depth compute, execution, and validation stages only after the W1 seam exists.

### GaussianSplatRenderer

Current state is spread across explicit sub-buckets and orchestrators:

- Frame and view state are grouped in `RenderFrameContextManager`
- `DeviceState` is the owner of the current render-device and scene-render handles
- `ResourceState` owns buffer lifecycle and per-frame GPU resource state
- `SubsystemState` owns orchestrator and interface handles
- `render_thread_dispatch_mutex`, `render_thread_dispatch_semaphore`, and related dispatch counters own the render-thread synchronization path

Implementation anchors:

- `gaussian_splat_renderer.h:531` through `gaussian_splat_renderer.h:538` hold the render-thread dispatch state.
- `gaussian_splat_renderer.h:518` through `gaussian_splat_renderer.h:529` hold the orchestrator graph.
- `_dispatch_call_on_render_thread_blocking()` and `_safe_submit_sync()` are the current synchronization chokepoints.
- `gaussian_splat_renderer.cpp:156` through `gaussian_splat_renderer.cpp:163` hold the frame-log settings cache.

Current write ownership:

- The renderer owns Godot-facing API entry points, frame setup, orchestration, and cross-subsystem lifecycle.
- Orchestrators own the finer-grained subsystem work that has already been peeled out.
- Render-thread dispatch remains renderer-owned for now, but it should move behind a dedicated seam in W1.

Refactor note:

- W1 should remove synchronization from the renderer facade.
- W2 should continue slimming the facade, but only after W1 has made ownership explicit.

## Hidden Mutable State Inventory

| Location | Symbol / state | Why it matters | Current owner status |
| --- | --- | --- | --- |
| `renderer/gaussian_splat_renderer.cpp:156` | `g_frame_log_settings` | Process-global cache for frame logging/debug behavior. It is updated from project settings and shared by all renderer instances. | Temporary global cache, should become explicit renderer or process service state. |
| `renderer/tile_renderer.cpp:139` | `adaptive_overlap_budget_runtime_states` | Process-global cache keyed by `TileRenderer *`. This is hidden instance state stored outside the instance. | Temporary debt, should move into `TileRenderer` ownership or a dedicated controller. |
| `renderer/tile_renderer.cpp:1927` | `subgroup_support_cache` | Process-global cache keyed by device ID. It affects shader path selection and must not be reinterpreted as per-instance state. | Temporary cache, should be explicitly owned or documented as device capability cache. |
| `renderer/gaussian_splat_renderer.h:531` through `:538` | render-thread dispatch mutex, semaphore, request/completion counters, timeout | Synchronization state is embedded in the renderer facade. | Renderer-owned for now, but the W1 dispatcher seam should take it over. |
| `interfaces/gpu_sorting_pipeline.h:213` | `pending_renderer` | Cross-object mutable handoff state. It makes the pipeline depend on renderer lifetime and shape. | Temporary coupling, should be removed by W1. |
| `core/gaussian_streaming.cpp:2400` | `debug_frame` static counter | Telemetry pacing is process-global, not instance-owned. | Benign but still hidden mutable state; keep the scope explicit. |

## What This Means For Decomposition

- `W0` must stay stable: the ownership matrix and the characterization tests are the hard gate.
- `W1` should cut concrete seams first: sort result sink, render-thread dispatcher, tile resource controller, and hidden-static cleanup.
- `W2` should extract execution stages only after ownership is explicit.
- `W3` should be cleanup only: remove the shims, slim the facades, and enforce the dependency rules.

## Related Docs

- Architecture overview: [overview.md](overview.md)
- Render pipeline details: [render-pipeline.md](render-pipeline.md)
- Module architecture map: [../../modules/gaussian_splatting/ARCHITECTURE.md](../../modules/gaussian_splatting/ARCHITECTURE.md)
- Memory and residency invariants: [../../modules/gaussian_splatting/MEMORY_SUBSYSTEM.md](../../modules/gaussian_splatting/MEMORY_SUBSYSTEM.md)
