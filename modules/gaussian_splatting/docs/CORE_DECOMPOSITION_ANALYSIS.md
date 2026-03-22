# Core Module Decomposition Analysis

## Executive Summary
The core folder concentrates multiple responsibilities into a handful of very large files. The largest file, `gaussian_streaming.cpp` (7,648 lines, ~130 function definitions), acts as a multi-system orchestrator: chunk lifecycle, visibility & LOD policy, VRAM budgeting, upload scheduling, quantization, and global atlas metadata all live in one translation unit. This makes the system powerful but hard to reason about, and the nested state machines (visibility, eviction, upload queue, budget regulator) are already class-like seams that could be extracted without changing behavior.

`gaussian_data.cpp` (2,104 lines, ~44 function definitions) is a classic "data + everything else" class. It owns storage but also performs file I/O, spatial acceleration, runtime edit overlays, animation integration, GPU buffer packing, and color-grading baking. This broad surface increases coupling to renderer, persistence, animation, and editor-facing logic.

Size snapshot: 2026-03-02 (`wc -l`), point-in-time only.
Refresh command:
`wc -l modules/gaussian_splatting/core/gaussian_streaming.cpp modules/gaussian_splatting/core/gaussian_data.cpp`

Top 3 recommendations:
1. Split the streaming pipeline into distinct subsystems (visibility/LOD, upload queue, global atlas) with a thin `GaussianStreamingSystem` facade.
2. Move `GaussianData` file I/O + octree + runtime overlay/brush + animation into dedicated helpers, leaving `GaussianData` as a pure data container + minimal GPU upload entry points.
3. Isolate quantization + VRAM budget regulation into standalone modules to reduce churn in streaming logic and enable focused testing.

## File Analysis

### gaussian_streaming.cpp (7,648 lines)

#### Responsibilities
1. **Chunk lifecycle & residency**: chunk creation, load/unload, LRU eviction, residency requests, atlas slot allocation.
2. **Visibility + frustum culling**: camera tracking, frustum testing, visible list building.
3. **LOD policy**: blend factors, hysteresis, Octree-GS distance reduction, SH band level scaling.
4. **VRAM budget regulation**: config, device memory queries, adaptive chunk limits, thrash detection.
5. **Upload pipeline**: async pack threads, upload budgeting, GPU submission, per-frame throttling.
6. **Quantization**: per-chunk quantization bounds, GPU quantization buffer, stats.
7. **Global atlas metadata**: asset/chunk metadata tables, dirty tracking, residency sync.
8. **Telemetry & debug**: per-frame analytics snapshot, logging, debug stats.

#### Extraction Candidates
(Line estimates are based on current line ranges in `gaussian_streaming.cpp`.)

| Candidate | Lines (approx) | Description | Effort |
|-----------|----------------|-------------|--------|
| `StreamingVisibilityController` | ~700 | Frustum culling, visibility lists, predictive prefetch, LOD blend calc (`VisibilityState`, `_update_chunk_visibility`, prefetch helpers) | Medium |
| `StreamingUploadPipeline` | ~450 | Async pack threads, upload budgeting, queue processing (`StreamingUploadPipeline`, `_pack_chunk_data`, `_process_upload_queue`) — **extracted** to `core/streaming_upload_pipeline.h/.cpp` | Done |
| `GlobalAtlasRegistry` | ~550 | Asset/chunk metadata, atlas buffers, residency sync (`_build_global_atlas_cpu_state`, `_sync_global_atlas_state`, request/evict/load helpers) | Medium-High |
| `ChunkQuantization` | ~280 | `ChunkQuantizationInfo` + quantization config, buffer build/upload (`_compute_chunk_quantization`, `_upload_quantization_buffer`) | Medium |
| `VRAMBudgetRegulator` | ~260 | Move to its own file with config + stats + update logic | Low |
| `LODPolicyOctreeGS` | ~200 | LOD param updates, debug stats collection, blend utilities | Low-Medium |
| `AssetRegistry/DenseIdMap` | ~250 | Asset registration, dense ID mapping, asset state accessors | Low-Medium |
| `StreamingTelemetry` | ~100 | `_log_streaming_telemetry`, analytics snapshot generation | Low |

#### Recommendations
- **Short term**: Extract `VRAMBudgetRegulator` + `ChunkQuantization` into separate files (lowest coupling, well-bounded).
- **Medium term**: Extract `VisibilityState` into a companion class with clear inputs/outputs to simplify `update_streaming`. (`UploadQueueState` already extracted as `StreamingUploadPipeline`.)
- **Long term**: Move global atlas + residency into a dedicated registry module. This is the largest seam and will reduce complexity in streaming's main update loop.

#### Notes / Potential dead or incomplete areas
- `ConfigOverrides.override_io_source` exists in the header and is set from `gaussian_splat_world_3d.cpp`, but there is no usage in `gaussian_streaming.cpp`. This looks like an unimplemented override path worth either wiring or removing in a future cleanup.

---

### gaussian_data.cpp (2,104 lines)

#### Responsibilities
1. **Core storage**: manages `LocalVector<Gaussian>` and SH coefficient arrays.
2. **File I/O**: PLY load/save via `PLYLoader` and manual binary writer.
3. **Spatial acceleration**: octree build/query + frustum gathering.
4. **Runtime edit overlays**: per-splat overrides, commit/revert, brush strokes + serialization.
5. **GPU buffer management**: pack + upload `PackedGaussian` buffers.
6. **Animation integration**: state machine hooks, caches, sampling, incremental saver wiring.
7. **Color grading baking**: CPU color grading and bake/restore lifecycle.

#### Extraction Candidates

| Candidate | Lines (approx) | Description | Effort |
|-----------|----------------|-------------|--------|
| `GaussianDataIO` | ~160 | `load_from_file` / `save_to_file` and format-specific helpers | Low |
| `GaussianOctreeIndex` | ~300 | `build_octree`, `_subdivide_octree_node`, `query_octree`, frustum gathering | Medium |
| `GaussianRuntimeEdits` | ~270 | runtime overlays, brush strokes, commit/revert logic | Medium |
| `GaussianAnimationAdapter` | ~210 | animation state machine integration + caches | Medium |
| `GaussianColorGradingBaker` | ~80 | bake/restore + CPU grading | Low |
| `GaussianGpuUploader` | ~70 | buffer creation/update + packing | Low |

#### Recommendations
- **Short term**: Extract file I/O and color-grading bake (small, contained).
- **Medium term**: Move octree + runtime edits into separate helpers to reduce coupling and reduce the size of `GaussianData`'s public API.
- **Long term**: Isolate animation integration into a dedicated adapter so animation and data storage can evolve independently.

---

## Dependency Graph
(High-level summary derived from include analysis of `core/*.cpp`.)

- **Common dependencies across core**: `core/math/math_funcs.h`, `core/config/project_settings.h`, `../logger/gs_logger.h`, `../renderer/gaussian_gpu_layout.h`, `servers/rendering/rendering_device.h`.
- **`gaussian_streaming.cpp`** depends on:
  - core data (`gaussian_data.h`), LOD policy (`../lod/lod_config.h`), renderer GPU layout/quantization, streaming manager (`gaussian_splat_manager.h`), rendering device + server.
  - also pulls in memory streaming + debug/telemetry paths.
- **`gaussian_data.cpp`** depends on:
  - I/O (`../io/ply_loader.h`), persistence (`../persistence/incremental_saver.h`), animation (`../animation/animation_state_machine.h`), renderer GPU layout, color grading resource.

This shows **strong coupling between core/data and renderer/IO/animation**, which amplifies change impact and complicates testing. Extracting the IO + animation + GPU upload layers would reduce cross-module churn.

## Recommendations

### High Priority
1. **Streaming pipeline split**: carve out `Visibility + LOD`, `UploadQueue`, and `GlobalAtlasRegistry` into dedicated classes with narrow interfaces (est. 2-3 refactor passes).
2. **GaussianData decomposition**: isolate file I/O and octree logic so `GaussianData` remains a storage + access class (est. 1-2 passes).
3. **Quantization module**: move `ChunkQuantizationInfo` and related GPU buffer creation to its own file (low risk, minimal API changes).

### Medium Priority
1. **VRAM budget regulator isolation**: extract to `vram_budget_regulator.*` to reduce streaming complexity.
2. **Runtime edit overlays**: move brush + overlay logic into a `GaussianRuntimeEdits` helper to clarify editor-only responsibilities.
3. **Animation adapter**: separate animation caching and sampling from data storage to reduce incidental coupling.

### Low Priority / Future
1. **Config override cleanup**: either implement `override_io_source` in streaming or remove unused override fields.
2. **Telemetry consolidation**: move analytics snapshot assembly into a small utility class.

## Risk Assessment
- **Risks**: high coupling to renderer and threading (upload queue), potential performance regressions if boundaries introduce extra copies or synchronization, subtle changes in streaming timing.
- **Benefits**: improved testability, clearer ownership, easier incremental optimization, safer parallel work by multiple contributors.
- **Recommended approach**: incremental extraction with a stable `GaussianStreamingSystem` / `GaussianData` facade, one subsystem per PR, add perf-regression checks and targeted unit tests for extracted utilities.
