# Streaming

## Purpose

Load and evict Gaussian splat chunks on demand so that datasets larger than available VRAM can be rendered without running out of memory. `GaussianStreamingSystem` manages a ring buffer of GPU chunks, performing frustum culling, predictive prefetch, and distance-based LOD to keep the visible portion of a scene resident.

## When to use streaming

| Scenario | Recommendation |
| --- | --- |
| Dataset fits entirely in VRAM (under ~256 MB) | Streaming adds overhead; load the dataset directly through `GaussianSplatNode3D`. |
| Dataset exceeds available VRAM or contains more than 500K splats | Enable streaming to stay within GPU memory limits. |
| Camera moves continuously through a large environment | Enable predictive prefetch to preload chunks along the camera path. |
| Multiple Gaussian assets share a single world | Use multi-asset registration so the atlas shares VRAM across assets. |

## Enabling streaming

Streaming is enabled globally through a project setting and is active by default.

| Step | Action | Implementation reference |
| --- | --- | --- |
| 1 | Confirm `rendering/gaussian_splatting/streaming/enabled` is `true` in Project Settings. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:933` |
| 2 | Create a `GaussianStreamingSystem` instance (the node creates one automatically when streaming is enabled). | `modules/gaussian_splatting/core/gaussian_streaming.h:31` |
| 3 | Call `initialize(data)` with a loaded `GaussianData` resource to partition splats into chunks. | `modules/gaussian_splatting/core/gaussian_streaming.cpp:927` |
| 4 | Each frame, call `update_streaming(camera_transform, projection)` to drive chunk loading, culling, and eviction. | `modules/gaussian_splatting/core/gaussian_streaming.h:770` |

## VRAM budget configuration

The streaming system allocates a fixed number of chunk slots in GPU memory. Each slot holds `CHUNK_SIZE` (65 536) splats. Budget limits can be configured globally or per quality tier.

| Project setting | Default | Description | Implementation reference |
| --- | --- | --- | --- |
| `rendering/gaussian_splatting/streaming/max_upload_mb_per_frame` | `128` | Maximum MB uploaded to VRAM in a single frame. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1054` |
| `rendering/gaussian_splatting/streaming/max_upload_mb_per_slice` | `16` | Maximum MB per upload slice within a frame. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1056` |
| `rendering/gaussian_splatting/streaming/max_chunk_loads_per_frame` | `16` | Cap on chunk load operations per frame. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1041` |
| `rendering/gaussian_splatting/streaming/eviction_hysteresis_frames` | `5` | Frames a chunk must be unused before eviction is considered. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1060` |
| `rendering/gaussian_splatting/streaming/max_evictions_per_frame` | `4` | Maximum chunk evictions per frame. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1062` |

Quality tier presets (`rendering/gaussian_splatting/quality/tier_preset`) override streaming budgets when `tier_apply_streaming_budgets` is enabled. See [quality_tier_config.h](../../modules/gaussian_splatting/core/quality_tier_config.h) for tier-specific values.

## Chunk LOD behavior

Chunks transition between LOD levels based on distance from the camera. The system uses an Octree-GS-inspired approach that reduces SH band level, increases splat skip factor, and modulates opacity as distance grows.

| Per-chunk LOD parameter | Effect | Implementation reference |
| --- | --- | --- |
| `current_lod_level` / `target_lod_level` | Drive smooth LOD transitions. | `modules/gaussian_splatting/core/gaussian_streaming.h:106` |
| `sh_band_level` (0 -- 3) | 0 = DC only, 3 = full spherical harmonics. | `modules/gaussian_splatting/core/gaussian_streaming.h:110` |
| `splat_skip_factor` | 1 = render all, higher values skip splats. | `modules/gaussian_splatting/core/gaussian_streaming.h:111` |
| `opacity_multiplier` | Distance-based opacity fade. | `modules/gaussian_splatting/core/gaussian_streaming.h:112` |
| `lod_blend_factor` | Smooth cross-fade between LOD levels. | `modules/gaussian_splatting/core/gaussian_streaming.h:104` |

LOD blend behavior is controlled through the `LODBlendConfig` on the visibility state. Toggle blending with `set_lod_blend_enabled(true)` and adjust the blend distance with `set_lod_blend_distance(value)`.

## Predictive prefetch

When the camera is in motion, the system predicts where the camera will be and begins loading chunks ahead of time.

| Project setting | Default | Description | Implementation reference |
| --- | --- | --- | --- |
| `rendering/gaussian_splatting/streaming/predictive_prefetch_enabled` | `true` | Enables velocity-based prefetch. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1031` |
| `rendering/gaussian_splatting/streaming/prefetch_lookahead_distance` | `10.0` | Distance (in world units) to look ahead for prefetch. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1033` |
| `rendering/gaussian_splatting/streaming/max_prefetch_loads_per_frame` | `6` | Prefetch load cap per frame. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1043` |

## Monitoring streaming performance

### GDScript queries

```gdscript
var streaming: GaussianStreamingSystem = splat_node.get_streaming_system()
print("VRAM usage bytes: ", streaming.get_vram_usage())
print("Loaded chunks: ", streaming.get_loaded_chunks())
print("Visible count: ", streaming.get_visible_count())
print("Effective splat count after LOD: ", streaming.get_effective_splat_count())

var analytics: Dictionary = streaming.get_streaming_analytics()
print("Analytics: ", analytics)

var culling: Dictionary = streaming.get_chunk_culling_stats()
print("Culling stats: ", culling)

var vram: Dictionary = streaming.get_vram_debug_stats()
print("VRAM debug: ", vram)

var lod: Dictionary = streaming.get_lod_debug_stats()
print("LOD debug: ", lod)
```

### Key methods

| Method | Returns | Description | Implementation reference |
| --- | --- | --- | --- |
| `get_vram_usage()` | `int` | Total VRAM bytes consumed by loaded chunks. | `modules/gaussian_splatting/core/gaussian_streaming.h:789` |
| `get_loaded_chunks()` | `int` | Number of chunks currently resident in VRAM. | `modules/gaussian_splatting/core/gaussian_streaming.h:790` |
| `get_visible_count()` | `int` | Splat count visible this frame after culling. | `modules/gaussian_splatting/core/gaussian_streaming.h:780` |
| `get_effective_splat_count()` | `int` | Splat count after LOD reduction is applied. | `modules/gaussian_splatting/core/gaussian_streaming.h:873` |
| `get_streaming_analytics()` | `Dictionary` | Detailed analytics including pack/upload timing. | `modules/gaussian_splatting/core/gaussian_streaming.h:795` |
| `get_chunk_culling_stats()` | `Dictionary` | Total, visible, culled, and loaded chunk counts. | `modules/gaussian_splatting/core/gaussian_streaming.h:810` |
| `get_vram_debug_stats()` | `Dictionary` | VRAM regulator state and budget utilization. | `modules/gaussian_splatting/core/gaussian_streaming.h:814` |
| `get_lod_debug_stats()` | `Dictionary` | LOD level distribution and transition counts. | `modules/gaussian_splatting/core/gaussian_streaming.h:872` |
| `is_vram_budget_warning_active()` | `bool` | True when VRAM usage approaches the configured budget. | `modules/gaussian_splatting/core/gaussian_streaming.h:815` |

## Troubleshooting

| Symptom | Cause | Fix | Implementation reference |
| --- | --- | --- | --- |
| No splats visible after loading a large dataset | Chunks have not finished uploading; initial load is in progress. | Wait several frames and check `get_loaded_chunks()` is increasing. Increase `max_chunk_loads_per_frame` for faster initial population. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1041` |
| VRAM usage exceeds expectations | Quality tier override is raising the VRAM budget ceiling. | Check `rendering/gaussian_splatting/quality/tier_preset` and `tier_apply_streaming_budgets`. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1023` |
| Chunks pop in and out frequently | Eviction hysteresis is too low or VRAM budget is too tight. | Increase `eviction_hysteresis_frames` and raise `max_upload_mb_per_frame` so the system can keep more chunks loaded. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1060` |
| Zero visible splats persist for many frames | Camera is outside the bounding boxes of all chunks, or recovery is not triggering. | Verify camera position is within the dataset bounds. Check `rendering/gaussian_splatting/streaming/zero_visible_recovery_mode`. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:940` |
| Prefetch does not load chunks ahead of camera | Prefetch is disabled or lookahead distance is too small. | Set `predictive_prefetch_enabled` to `true` and increase `prefetch_lookahead_distance`. | `modules/gaussian_splatting/core/gaussian_splat_manager.cpp:1031` |
| LOD transitions cause visible flickering | LOD blend is disabled or blend distance is too short. | Call `set_lod_blend_enabled(true)` and increase the blend distance with `set_lod_blend_distance()`. | `modules/gaussian_splatting/core/gaussian_streaming.h:821` |
