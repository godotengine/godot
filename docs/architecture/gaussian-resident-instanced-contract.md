# Resident-Instanced Renderer Contract

Status: Stage 2A design checkpoint  
Scope: resident-instanced backend contract for the shared instance renderer  
Audience: rendering and pipeline engineers

## Purpose

This document defines the resident-instanced GPU contract that Stage 2B must implement before any backend-policy split lands in code.

It answers five questions explicitly:

1. What is the resident-instanced GPU contract?
2. Does it reuse atlas-shaped inputs or define alternative stage inputs?
3. Which current stage assumptions block resident instancing today?
4. What exact code areas will Stage 2B need to modify together?
5. What diagnostics and route labels will prove the new path is active?

The answer is: Stage 2B should keep the current atlas-shaped instance-stage contract and make resident data satisfy it without a `GaussianStreamingSystem`.

## Decision Summary

- Chosen contract: resident data emulates the existing atlas-shaped instance pipeline inputs.
- Chosen reason: it preserves the Stage 1 north star of one shared submission contract and keeps cull, sort, raster, and tile/binning logic stable.
- Minimal acceptable Stage 2B result: a resident scene can publish a valid `InstancePipelineBuffers` contract, pass readiness, and render through the shared instance path with no `current_streaming_system`.
- Explicit non-goal for Stage 2B: do not redesign cull/sort/raster around a second resident-only input model.

## Current Contract, As Implemented

The current shared renderer path is already stage-driven, but the instance contract is atlas-shaped and still produced by the streaming backend:

- Contract definition: [instance_pipeline_contract.h](../../modules/gaussian_splatting/renderer/instance_pipeline_contract.h)
- Shared buffer struct: [render_pipeline_io_types.h](../../modules/gaussian_splatting/renderer/render_types/render_pipeline_io_types.h)
- Current producer: [render_streaming_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp)
- Current readiness gate: [render_instancing_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp)

### Stage Inputs Required Today

| Stage | Required contract inputs | Current consumer |
| --- | --- | --- |
| Cull | `instance_buffer`, `asset_meta_buffer`, `asset_chunk_index_buffer`, `chunk_meta_buffer`, `visible_chunk_buffer`, `counter_buffer`, `instance_count`, `dispatch_chunk_count`, `max_visible_chunks` | [gpu_culler.h](../../modules/gaussian_splatting/interfaces/gpu_culler.h), [render_pipeline_stages.cpp](../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp) |
| Sort | `atlas_gaussian_buffer`, `instance_buffer`, `chunk_meta_buffer`, `visible_chunk_buffer`, `splat_ref_buffer`, `sort_key_buffer`, `sort_value_buffer`, `counter_buffer`, `chunk_dispatch_buffer`, `indirect_count_buffer`, `instance_count_buffer`, `max_visible_splats`, `max_chunk_splats`, optional `quantization_buffer` | [gpu_sorting_pipeline.h](../../modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.h), [render_pipeline_stages.cpp](../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp) |
| Raster | `atlas_gaussian_buffer`, sorted index buffer from the sorting pipeline, `instance_buffer`, `splat_ref_buffer`, `instance_count_buffer`, `atlas_gaussian_count`, optional `quantization_buffer` | [render_pipeline_stages.cpp](../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp) |
| Tile / binning | `atlas_gaussian_buffer`, sorted index buffer, `instance_buffer`, `splat_ref_buffer`, indirect count/dispatch buffers, optional `quantization_buffer`, optional `chunk_meta_buffer` when quantized | [tile_render_binning.cpp](../../modules/gaussian_splatting/renderer/tile_render_binning.cpp), [tile_render_rasterizer_stage.cpp](../../modules/gaussian_splatting/renderer/tile_render_rasterizer_stage.cpp) |

### What “Atlas-Shaped” Means In Practice

The code-level contract is not just “a gaussian buffer plus instances.” It assumes a dense, multi-asset atlas layout:

- `atlas_gaussian_buffer` is the stage-visible packed gaussian store.
- `asset_meta_buffer` maps dense asset IDs to atlas ranges and chunk metadata.
- `asset_chunk_index_buffer` maps each asset to its chunk table.
- `chunk_meta_buffer` carries per-chunk bounds and addressing.
- `instance_buffer` references assets by dense asset ID, not by original asset identity.

Those buffers are currently sourced from `GaussianStreamingSystem::get_global_atlas_state()` in [gaussian_streaming.h](../../modules/gaussian_splatting/core/gaussian_streaming.h) and [streaming_global_atlas_registry.h](../../modules/gaussian_splatting/core/streaming_global_atlas_registry.h).

## Chosen Stage 2B Contract

Stage 2B should keep the atlas-shaped contract and make resident data satisfy it directly.

### Resident-Instanced GPU Contract

The resident backend must publish the same stage-visible contract currently expressed by `InstancePipelineBuffers`, split into three groups:

1. Dataset contract
   - `atlas_gaussian_buffer`
   - `atlas_gaussian_count`
   - `asset_meta_buffer`
   - `asset_chunk_index_buffer`
   - `chunk_meta_buffer`
   - optional `quantization_buffer`
   - stable dense asset-ID mapping for `InstanceDataGPU`

2. Per-frame submission contract
   - `instance_buffer`
   - `instance_count`

3. Runtime scratch contract
   - `visible_chunk_buffer`
   - `counter_buffer`
   - `chunk_dispatch_buffer`
   - `indirect_count_buffer`
   - `instance_count_buffer`
   - `splat_ref_buffer`
   - `sort_key_buffer`
   - `sort_value_buffer`
   - `dispatch_chunk_count`
   - `max_visible_chunks`
   - `max_visible_splats`
   - `max_chunk_splats`

### Backend-Policy Selection Model

Stage 2B must not invent a second policy resolver in renderer code.

The current canonical policy-resolution site is [gs_project_settings.h](../../modules/gaussian_splatting/core/gs_project_settings.h), specifically `gs::settings::get_streaming_route_policy(ProjectSettings *)`.

Stage 2B should keep that as the only settings-boundary policy accessor and make the renderer/backend consume only its resolved enum output.

Required Stage 2B rule:

- no renderer, orchestrator, or backend component may read compatibility booleans directly
- `gs::settings::get_streaming_route_policy()` is the sole accessor for the canonical enum route policy
- backend selection then consumes:
  - resolved route policy from `get_streaming_route_policy()`
  - submission residency hint when present
  - runtime/device feasibility
  - budget feasibility

That gives Stage 2B an explicit collapse model for the overlap called out in the refactor plan:

- `route_policy` is the canonical policy seen by renderer code
- after Bucket B, the legacy toggle is gone and `get_streaming_route_policy()` is a thin enum accessor rather than a compatibility translator

For diagnostics, Stage 2B should expose the selection inputs on the existing stats surface:

- `requested_route_policy`
- `requested_route_policy_source`
- `instance_backend_policy`
- `backend_selection_reason`

`requested_route_policy_source` should distinguish at least:

- `route_policy`
- `default_fallback`

### Resident Semantics

For the resident backend, those atlas-shaped inputs have different provenance but identical stage meaning:

- all referenced chunks are already fully resident
- no streaming visibility, prefetch, upload queue, or eviction is needed to make them usable
- `asset_meta_buffer`, `asset_chunk_index_buffer`, and `chunk_meta_buffer` still exist because cull/sort/tile already consume them
- dense asset-ID remap still exists because instances reference dense asset IDs, not arbitrary submission IDs

In other words, the resident backend should behave like a fully materialized atlas, not like a second stage contract.

### Dense Asset-ID Remap Ownership

Stage 2B must make the dense asset-ID remap a renderer-owned published sidecar to the instance contract, not a hidden method on a backend object.

The contract boundary is:

- stage-visible contract: `InstancePipelineBuffers`
- non-stage sidecar contract: published dense asset-ID remap table

That sidecar should be published and cleared atomically with `InstancePipelineBuffers`.

Required ownership rule:

- the remap table lives in renderer-owned state, adjacent to `instance_pipeline_buffers`
- it is published by whichever backend currently owns atlas publication
  - streaming publisher derives it from `GaussianStreamingSystem`
  - resident publisher derives it from the resident atlas builder
- it is queried only through renderer-owned accessors
- `update_instance_buffer()` must consume the published renderer-side remap, not call a backend object directly

The minimum useful shape is a mapping from submission asset identity to dense atlas asset ID plus a generation value. Exact type names may vary, but Stage 2B should treat it as one published object, for example:

- `PublishedInstanceAssetRemap`
  - `asset_to_dense_id`
  - `generation`
  - readiness/validity flag

The key design point is publication, not the container choice. `InstancePipelineBuffers` does not need a RID handle for the remap, but the renderer must have a first-class published remap surface alongside it.

## Rejected Alternative

### Alternative Resident Stage Inputs

Rejected for Stage 2B: define a second resident-only stage input model such as:

- global-gaussian cull without asset/chunk indirection
- resident-only sort inputs that bypass `asset_meta_buffer` and `asset_chunk_index_buffer`
- resident-only raster/tile bindings that read a different descriptor layout

### Why It Was Rejected

- It creates a second renderer-facing contract instead of a backend policy split behind one contract.
- It forces simultaneous branching in cull, sort, raster, tile, validation, and diagnostics.
- It makes Stage 2B larger than necessary and increases the chance of redesigning Stage 2 mid-flight.
- It would turn backend selection into stage-contract selection, which is explicitly not the Stage 1 to Stage 2 architecture direction.

This alternative can be reconsidered only if atlas emulation proves impossible on measured technical grounds. That is not the case from the current code.

## Minimum Acceptable Contract For Stage 2B

Stage 2B is acceptable if all of the following are true together:

- resident upload produces a valid dataset contract without creating `current_streaming_system`
- `update_instance_buffer()` can remap dense asset IDs without depending on `GaussianStreamingSystem`
- readiness succeeds when resident atlas-shaped inputs are valid
- `InstancePipelineContract::has_atlas_buffers()`, `has_cull_buffers()`, `has_sort_buffers()`, and `has_raster_buffers()` all pass for the resident backend
- tile runtime bindings still validate without a resident-only branch
- the renderer can render a resident scene through the shared instance path and report that route explicitly

## Current Assumptions Blocking Resident Instancing

These are the concrete blockers today.

### 1. Upload path is hardwired to streaming-system creation

[render_data_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_data_orchestrator.cpp) always creates `memory_stream` and `GaussianStreamingSystem` in `update_gpu_buffers_with_real_data()`, even when the policy should be resident.

### 2. Instance upload depends on streaming for asset-ID remap

[gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp) in `update_instance_buffer()` calls `current_streaming_system->remap_instance_asset_ids(...)` and refuses to upload instances when no streaming system exists.

### 3. Instanced readiness explicitly requires a streaming system

[render_instancing_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp) treats `STREAMING_SYSTEM_UNAVAILABLE` as an instanced-readiness failure even if all instance buffers are otherwise valid.

### 4. Resident fast path still gates on streaming-system validity

[gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp) in `_render_resident_frame()` only enters the instanced fast path when `current_streaming_system.is_valid() && has_instance_pipeline_buffers()`.

### 5. Active data-source planning is streaming-owned

[gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp) in `get_active_data_source()` only accepts atlas data from `current_streaming_system->get_global_atlas_state()`.

### 6. Atlas-shaped buffer publication lives inside the streaming orchestrator

[render_streaming_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp) is the only place that currently:

- publishes atlas metadata into `InstancePipelineBuffers`
- allocates resident/runtime scratch buffers used by cull, sort, and tile
- computes `dispatch_chunk_count`, `max_visible_chunks`, and `max_visible_splats`

## What Can Stay Unchanged In Stage 2B

If Stage 2B keeps atlas-shaped emulation, these pieces should stay functionally unchanged:

- cull-stage GPU contract in [gpu_culler.h](../../modules/gaussian_splatting/interfaces/gpu_culler.h)
- sort-stage GPU contract in [gpu_sorting_pipeline.h](../../modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.h)
- stage runner logic in [render_pipeline_stages.cpp](../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp), except for backend-neutral readiness/data-source plumbing
- tile/binning descriptor layout in [tile_render_binning.cpp](../../modules/gaussian_splatting/renderer/tile_render_binning.cpp) and [tile_render_rasterizer_stage.cpp](../../modules/gaussian_splatting/renderer/tile_render_rasterizer_stage.cpp)
- Stage 1 submission model in [gaussian_splat_scene_director.h](../../modules/gaussian_splatting/core/gaussian_splat_scene_director.h)

## What Must Change Together In Stage 2B

These implementation areas are coupled and must change as one slice.

### 1. Upload and backend policy selection

- [render_data_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_data_orchestrator.cpp)
- [gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp)
- likely one new resident-atlas builder or backend-neutral atlas publisher extracted from current streaming code
- [gs_project_settings.h](../../modules/gaussian_splatting/core/gs_project_settings.h)

Required change:

- keep `gs::settings::get_streaming_route_policy()` as the sole settings-boundary accessor for backend choice
- choose resident vs streaming backend from the resolved enum plus submission hint and runtime feasibility
- stop unconditional streaming-system creation for resident data
- publish atlas-shaped resident dataset buffers and dense asset-ID mapping

### 2. Instance remap and readiness

- [gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp)
- [render_instancing_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_instancing_orchestrator.cpp)

Required change:

- move instance asset-ID remap behind a renderer-owned published contract sidecar
- change readiness from “streaming system exists” to “backend contract is valid”

### 3. Shared instance-contract publication

- [render_streaming_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp)
- [render_pipeline_io_types.h](../../modules/gaussian_splatting/renderer/render_types/render_pipeline_io_types.h)
- [instance_pipeline_contract.h](../../modules/gaussian_splatting/renderer/instance_pipeline_contract.h)

Required change:

- keep one `InstancePipelineBuffers` contract
- allow resident and streaming producers to publish it, together with the renderer-owned remap sidecar
- keep validation stage-oriented and backend-neutral

### 4. Routing and data-source reporting

- [gaussian_splat_renderer.cpp](../../modules/gaussian_splatting/renderer/gaussian_splat_renderer.cpp)
- [render_pipeline_stages.cpp](../../modules/gaussian_splatting/renderer/render_pipeline_stages.cpp)
- [render_diagnostics_orchestrator.cpp](../../modules/gaussian_splatting/renderer/render_diagnostics_orchestrator.cpp)
- [render_route_labels.cpp](../../modules/gaussian_splatting/renderer/render_route_labels.cpp)
- [render_debug_state_orchestrator.h](../../modules/gaussian_splatting/renderer/render_debug_state_orchestrator.h)

Required change:

- add an explicit resident-instanced route
- report backend policy and contract readiness through existing diagnostics surfaces
- make `get_active_data_source()` work for resident atlas publication too

### 5. Validation and tests

- [instance_pipeline_contract.h](../../modules/gaussian_splatting/renderer/instance_pipeline_contract.h)
- module tests under [../../modules/gaussian_splatting/tests](../../modules/gaussian_splatting/tests)

Required change:

- validate resident atlas publication through existing invariant classes
- add resident-specific readiness and route assertions without forking the invariant model

## Route And Diagnostic Proof For Stage 2B

Stage 2B should prove the resident-instanced path through existing route and stats surfaces, not through a second debug system.

### Required route evidence

- add a new top-level `route_uid` for the resident-instanced path
  - recommended: `INSTANCE.RESIDENT`
- keep stage route IDs unchanged when the shared stages are reused
  - `cull_route_uid = INSTANCE.CULL.GPU`
  - `sort_route_uid = INSTANCE.SORT.GPU`
  - raster route remains one of the existing instance raster UIDs

### Required stats and HUD evidence

Expose these through the existing diagnostics and HUD surfaces:

- `route_uid = INSTANCE.RESIDENT`
- `route_label = Resident instanced path`
- `instance_backend_policy = resident`
- `instance_contract_shape = atlas_emulation`
- `instance_contract_ready = true|false`
- `requested_route_policy`
- `requested_route_policy_source`
- `backend_selection_reason`

Resident success should therefore read as:

- route: resident instanced path
- cull: GPU chunk culling
- sort: GPU instance sort
- raster: existing instance raster label

### Required failure evidence

Resident failures should use the same validation surface, but with resident-relevant reasons:

- missing atlas contract buffers
- missing resident instance remap table
- missing resident scratch buffers
- resident contract published, but not ready for cull/sort/raster

## Accepted Stage 2B Behavior And Current Limits

- The resident atlas publisher intentionally rejects per-chunk quantization. That rejection is surfaced as `resident_quantization_unsupported` and is treated as a backend-selection input, not as a trigger to invent a second resident-only stage contract.
- When resident publication is rejected and the renderer then succeeds on streaming, `backend_selection_reason` intentionally preserves both parts of the story with ` -> ` chaining, for example `submission_hint_resident:world_submission_not_feasible:resident_quantization_unsupported -> streaming_contract_published`.
- Submission-hint collapse is conservative in the accepted implementation. Conflicting instance-submission hints on one shared renderer collapse to no effective hint (`mixed_instance_submissions`), while preview submissions and active world submissions still take precedence over instance hints. Cross-source mixed-hint normalization is intentionally deferred because it would change backend-policy semantics.
- Explicit resident route requests may still render through the legacy resident path when resident atlas publication is rejected. That fallback is an accepted Stage 2B compatibility behavior, not an accidental bypass of the shared instance contract.

## Likely Stage 2B Tests

Add or update focused tests for:

- resident route renders through the instance/shared renderer with no `current_streaming_system`
- resident readiness passes when atlas-shaped resident buffers are published
- resident readiness fails with clear invariant reasons when atlas/cull/sort/raster inputs are incomplete
- `update_instance_buffer()` remaps asset IDs through the published renderer-side remap
- diagnostics report `INSTANCE.RESIDENT` and the expected human-readable label
- route policy `resident` does not create a streaming system for resident-fit scenes
- Bucket B removes the legacy `streaming/enabled` translation. Route-policy diagnostics now distinguish only explicit `route_policy` and default fallback.

Runtime/integration coverage should include:

- small resident-fit scene
- multi-instance scene on one shared renderer
- Stage 1B world-backed submission feeding a resident-fit scene without a streaming backend

## Exact Answers To The Stage 2A Questions

### 1. What is the resident-instanced GPU contract?

It is the existing atlas-shaped `InstancePipelineBuffers` contract, satisfied by a resident backend instead of a `GaussianStreamingSystem`.

### 2. Does it reuse atlas-shaped inputs or define alternative stage inputs?

It reuses atlas-shaped inputs.

### 3. Which current stage assumptions block resident instancing today?

The blockers are upload-time streaming-system creation, instance asset-ID remap through `GaussianStreamingSystem`, instanced-readiness gating on streaming-system presence, resident fast-path gating on streaming-system presence, and streaming-owned data-source planning.

### 4. What exact code areas will Stage 2B need to modify together?

- upload/backend selection in `render_data_orchestrator.cpp` and the canonical settings resolver in `gs_project_settings.h`
- instance remap publication and resident fast-path gating in `gaussian_splat_renderer.cpp`
- readiness in `render_instancing_orchestrator.cpp`
- shared contract publication in `render_streaming_orchestrator.cpp` plus a resident counterpart
- route, diagnostics, and validation surfaces in `render_pipeline_stages.cpp`, `render_diagnostics_orchestrator.cpp`, `render_route_labels.cpp`, `render_debug_state_orchestrator.h`, and `instance_pipeline_contract.h`

### 5. What diagnostics and route labels will prove the new path is active?

`route_uid = INSTANCE.RESIDENT` plus the existing `cull_route_uid`, `sort_route_uid`, and raster route labels on the existing HUD/stats surfaces.

## Exit Check

- The resident-instanced contract is written down clearly enough to code Stage 2B without redesign: Yes.
- The document explicitly states whether Stage 2B will emulate atlas inputs or add alternative stage inputs: Yes, atlas-shaped emulation.
- The document names the coupled implementation areas that must change together: Yes.
- The document is consistent with the accepted Stage 1 architecture: Yes.
