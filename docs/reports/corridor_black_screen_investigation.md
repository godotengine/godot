# Corridor Black Screen Investigation

**Date:** 2026-04-11
**Scene:** `lane_open_world_corridor_proof.tscn` (20M splat `.gsplatworld`)
**Symptom:** Grey background, zero visible gaussian splats despite 104K sorted splats reaching the tile rasterizer.

## Problem Summary

The open-world corridor proof scene loads a 2.9GB `.gsplatworld` file containing 20M gaussians. The streaming system successfully loads 16 atlas chunks (~104K splats), the cull/sort/raster/composite pipeline executes without errors, and the compositor copies the raster output to the viewport framebuffer — yet the viewport shows only the Godot grey background with no visible splats.

**Single-cabin scene (`single_cabin.tscn`) renders correctly** using the identical code path with 410 splats. Both scenes use the streaming backend, instance pipeline, tile rasterizer (compute), and `copy_to_fb_rect` compositor.

## What Was Ruled Out

| Stage | Status | Evidence |
|-------|--------|----------|
| Scene loading | OK | `.gsplatworld` loads, world set on `GaussianSplatWorld3D` |
| Streaming init | OK | 16 atlas chunks published, `atlas_pub_chunks=16` |
| Route policy | OK | `streaming_req=YES streaming_ready=YES` |
| Cull stage | OK | `vis_splat_count=104609` (chunk-level cull passes 16 chunks) |
| Sort stage | OK | `sorted_count=104609 domain=SPLAT_REF` |
| Raster dispatch | OK | All buffers valid: `gauss_buf=Y sort_idx=Y splat_ref=Y chunk_meta=Y inst_buf=Y` |
| Raster output | OK | `output=valid depth=valid has_depth=yes` |
| Compositor | OK | `render_buffers_rd=valid final_output=valid rt_fb=valid` |
| Device match | OK | `main_rd == godot_rd (same=YES)`, all textures valid on both |
| File vs memory source | Same | Swapping `payload_source` → `data` preference did NOT fix it |
| Project settings | OK | `enable_pipeline_trace=true` causes init hang (separate issue), removed |

## Key Diagnostic Data

```
[DIAG-DISPATCH] splat_count=104609 total_gauss=1048576 gauss_buf=Y sort_idx=Y splat_ref=Y chunk_meta=Y inst_buf=Y max_vis=5000000
[DIAG-RRESULT] splats=104609 overlaps=0 tiles=0 empty=0 max_per_tile=0 avg=0.0 compute=yes
[DIAG-COMPOSITOR] render_buffers_rd=valid final_output=valid viewport=1920x1080 defer=yes
[DIAG-COPY-DEVICE] same=YES src_valid_mgr=yes src_valid_godot=yes fb_valid_godot=yes
```

The `overlaps=0` in `DIAG-RRESULT` is NOT diagnostic — these stats are async readback and lag behind the actual GPU execution. The same `overlaps=0` appears for single_cabin which renders correctly.

## Streaming System Warnings

```
WARNING: [Streaming] Skipping Morton sort for 20000000 splats (threshold=2000000); using contiguous chunk layout.
WARNING: [Streaming] Static chunk layout is non-contiguous; falling back to contiguous runtime chunk partitioning
  (usage=io reason=hint_non_contiguous_coverage detail_a=15333 detail_b=4194)
WARNING: create_gpu_buffer: 20000000 gaussians would require 2.7 GB; exceeds 2 GB staging limit
WARNING: [Streaming][Diag:queue_pressure] qsrc=cap qwhy=vram_chunk_cap
```

The static chunks from the `.gsplatworld` are rejected (non-contiguous coverage), and contiguous runtime chunks are used instead. The VRAM chunk cap limits loaded chunks to 16 (`max_chunks_in_vram=16` in project.godot).

## Remaining Hypotheses

### 1. GPU-side data content issue (MOST LIKELY)
The atlas buffer, sorted indices, or splat_ref buffer contains correct RIDs and sizes, but the actual GPU memory content may be wrong. The tile rasterizer dispatches compute work for 104K splats but the output texture is fully transparent. This could mean:
- Splat positions in the atlas project outside the viewport (all off-screen)
- Splat opacity values are zero in the packed data
- The splat_ref → atlas index mapping is wrong, causing the shader to read uninitialized atlas slots
- The contiguous chunk fallback creates chunk boundaries that don't match the atlas slot assignments

### 2. Instance transform issue
The corridor world uses a single instance at identity transform. If the world's gaussian positions are in LOCAL space (per-source-instance, not baked to world space), all 20M splats would cluster near origin instead of along the corridor layout. The camera at (0, 11, 34) might not see them.

### 3. Chunk AABB → visibility mismatch
With contiguous chunk fallback, chunk AABBs are computed from sequential gaussian index ranges. If the gaussians in the `.gsplatworld` are stored in an order that doesn't correspond to spatial locality, chunk AABBs would be huge/overlapping and all chunks would pass visibility — but the actual loaded subset might cover a region the camera can't see.

## Suggested Next Steps

1. **GPU debug capture** (RenderDoc/Nsight): Capture a frame, inspect the tile rasterizer's input buffers to verify gaussian positions and splat_ref mappings are valid.
2. **CPU-side data audit**: Log the first few gaussian positions from packed chunk data at upload time to verify positions are in the expected world-space range (x: 0-60, z: 0-1000 for the corridor).
3. **Minimal world test**: Create a small `.gsplatworld` with ~1000 splats at known positions and test if the world→streaming→raster→compositor pipeline produces visible output.
4. **Camera override test**: Move the camera to origin (0,0,0) and check if splats appear — this would confirm if the issue is splat positions vs camera location.

## Files Changed (Diagnostics)

- `gaussian_splat_renderer.cpp`: Added `DIAG-RSI` (render_scene_instance entry), `DIAG-ROUTE` (backend routing), `DIAG-RESIDENT` (resident path check)
- `render_pipeline_stages.cpp`: Added `DIAG-RASTER` (sorted splat count at raster entry), `DIAG-DISPATCH` (tile rasterizer dispatch params), `DIAG-RRESULT` (raster stats after render)
- `output_compositor.cpp`: Existing `DIAG-COMPOSITOR`, `DIAG-COMPOSITOR2`, `DIAG-COPY-DEVICE` diagnostics
- `gaussian_streaming.cpp`: Guard fixes for `payload_source` validity, `DIAG-SYNC-LOAD` diagnostics
- `project.godot`: Removed `enable_pipeline_trace` (causes init hang — separate bug)
