# Render Pipeline Architecture

This document describes the runtime render pipeline in detail: frame entry, route selection, stage execution, and fallback behavior.

## Canonical Pipeline Entry Points

- Frame entry: [../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp) (`render_scene_instance`)
- Streaming path orchestration: [../../modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp)
- Instancing execution mode decisions: [../../modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp)
- Stage runner interface: [../../modules/gaussian_splatting/renderer/render_pipeline_stages.h](../../modules/gaussian_splatting/renderer/render_pipeline_stages.h)
- Stage implementations: [../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp](../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp)
- Tile raster/resolve backend: [../../modules/gaussian_splatting/renderer/tile_render_stages.cpp](../../modules/gaussian_splatting/renderer/tile_render_stages.cpp)

## Frame Execution Flow

1. `GaussianSplatRenderer::render_scene_instance` initializes per-frame state and camera/view context.
2. Renderer decides route: streaming route via `RenderStreamingOrchestrator` when streaming buffers/readiness are valid, otherwise resident fallback route.
3. `RenderPipelineStages` runs stage sequence: cull (`execute_cull_stage`), sort (`execute_sort_stage`), then raster/composite (`render_sorted_splats_with_context`).
4. Output and diagnostics are finalized.

## Execution Modes (Single-Pass vs Serial)

Instancing mode policy is controlled by project settings:

- `rendering/gaussian_splatting/instance_pipeline/true_single_pass_enabled`
- `rendering/gaussian_splatting/instance_pipeline/benchmark_allow_serial_multi_asset`

Mode behavior in orchestrators:

- `single_pass`: production path (one cull/sort/raster chain for the frame)
- `serial`: benchmark/diagnostic replay path when explicitly allowed
- `single_pass_forced`: serial was requested but policy forced single-pass

Related sources:

- [../../modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp)
- [../../modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp)

## Stage Contracts

### Cull Stage

- Inputs: view transform/projection + viewport + frame/provider context
- Output: visible count and visible-domain information
- Contract owner: `RenderPipelineStages::CullStage`

### Sort Stage

- Inputs: world-to-camera transform + cull outputs
- Output: sorted indices and output-domain metadata
- Contract owner: `RenderPipelineStages::SortStage`

### Raster and Composite

- `RenderPipelineStages::render_sorted_splats_with_context` prepares raster/composite inputs
- `TileRenderer` performs tile pipeline and resolve passes
- Output compositor handles final target/viewport handoff

## Fallback and Failure Semantics

When prerequisites are missing (device, buffers, or readiness invariants), the renderer records route/fallback information and avoids publishing invalid stage results.

Relevant code:

- [../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp)
- [../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp](../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp)
- [../../modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp)

## Related Docs

- High-level architecture: [overview.md](overview.md)
- Lighting details: [lighting-system.md](lighting-system.md)
- Memory/residency design: [../../modules/gaussian_splatting/MEMORY_SUBSYSTEM.md](../../modules/gaussian_splatting/MEMORY_SUBSYSTEM.md)
