# Tier-2 Cluster Culling Spec

Status: implementation spec  
Scope: tier-2-native coarse culling for the instance/chunk pipeline  
Audience: renderer, streaming, and GPU-pipeline engineers

## 1. Summary

Cluster culling adds a coarse rejection stage between tier-2 chunk culling and per-splat depth/projection work: each visible chunk is subdivided into chunk-local clusters, each cluster has a conservative bound plus a splat range, and invisible clusters are rejected before `depth_compute.glsl` emits `splat_ref_buffer` entries. The goal is to reduce per-splat projection, sort-key generation, and downstream tile pressure for large chunks. Non-goals: replacing chunk culling, changing tile binning heuristics from PR #237, redesigning the tier-2 instance contract, or introducing an alternative resident-only render path.

## 2. Current Architecture Relevant Files

- `modules/gaussian_splatting/interfaces/gpu_culler.h/.cpp`: active cull entry point; tier-2 path is `GPUCuller::cull_for_view()` -> `_gpu_frustum_cull_instance()`, which fills `visible_chunk_buffer`.
- `modules/gaussian_splatting/compute/frustum_cull.glsl`: Stage A instance/chunk frustum pass; writes `VisibleChunkRefGPU` and chunk counters.
- `modules/gaussian_splatting/interfaces/gpu_sorting_pipeline.cpp`: active Stage B sort/depth path; runs `instance_chunk_dispatch.glsl`, `depth_compute.glsl`, then `instance_count_clamp.glsl`.
- `modules/gaussian_splatting/compute/depth_compute.glsl`: current per-splat expansion pass; reads `visible_chunk_buffer`, emits `splat_ref_buffer`, sort keys, and visible splat count.
- `modules/gaussian_splatting/shaders/tile_binning.glsl`: current projection/binning stage; consumes sorted indices plus `splat_ref_buffer`. Cluster cull must reduce its inputs, not replace it.
- `modules/gaussian_splatting/shaders/includes/gs_culling_utils.glsl`: shared distance/hotspot pruning helpers used after sort/projection.
- `modules/gaussian_splatting/renderer/render_pipeline_stages.cpp`: wires instance cull and sort stages together and selects the chunk-domain path.
- `modules/gaussian_splatting/renderer/render_streaming_orchestrator.cpp`: streaming publisher of atlas, `visible_chunk_buffer`, `splat_ref_buffer`, counters, and sort buffers.
- `modules/gaussian_splatting/renderer/resident_instance_contract_publisher.cpp`: resident publisher of the same tier-2 contract.
- `modules/gaussian_splatting/interfaces/cluster_culler.h/.cpp`: existing registered stub; it compiles and allocates buffers, but is not invoked by `GPUCuller::cull_for_view()` or `GPUSortingPipeline`.
- `modules/gaussian_splatting/lod/cluster_builder.h/.cpp`: existing CPU Morton clustering helper; today it operates on whole `GaussianData` and packs legacy AABB data, not tier-2 chunk-local instance metadata.
- `modules/gaussian_splatting/compute/cluster_cull.glsl`: legacy standalone coarse-pass shader; not wired into the tier-2 pipeline.

## 3. Design

Clusters are chunk-local and are generated at runtime when chunk payloads are already being read on CPU for atlas publication. Do not extend the active renderer around legacy `GaussianData` global clustering. Instead, publish cluster metadata as part of the tier-2 instance contract so the active flow becomes:

`frustum_cull.glsl -> visible_chunk_buffer -> cluster cull -> cluster-range expansion -> splat_ref_buffer -> sort/tile`

Use a default target of `512` splats per cluster, with project-setting clamp `256..1024`. Tier-2 chunks are up to `65536` splats, so `512` yields about `128` clusters/chunk: coarse enough to amortize dispatch, fine enough to reject occluded or off-frustum subregions inside a chunk.

Cluster record layout should be `32` bytes, `std430` friendly:

- `vec4 center_radius` where `xyz` is chunk-local center and `w` is conservative radius.
- `uvec4 range` where `x = splat_offset_in_chunk`, `y = splat_count`, `z/w = reserved`.

This is sphere-based, not AABB-based, because tier-2 already transforms chunk-local bounds through instance transforms in `frustum_cull.glsl`; cluster cull should reuse that model. Existing `ClusterCuller`/`ClusterBuilder` names and settings may be reused, but their implementation should be rewritten around instance/chunk inputs.

## 4. Data Flow

1. Streaming/resident publication allocates and fills:
   - `chunk_cluster_range_buffer`: per chunk `{ cluster_base, cluster_count }`
   - `cluster_meta_buffer`: global cluster records
   - `cluster_visible_buffer`: compacted visible-cluster refs `{ visible_chunk_index, cluster_index }`
   - `cluster_splat_ranges_buffer`: compacted fine-pass input `{ visible_chunk_index, splat_offset, splat_count, pad }`
   - `cluster_dispatch_buffer`: indirect dispatch for the fine cluster-expansion pass
2. `frustum_cull.glsl` remains unchanged and produces `visible_chunk_buffer`.
3. New `instance_cluster_dispatch.glsl` reads the visible-chunk counter and writes indirect dispatch for the cluster pass: `x = ceil(max_clusters_per_chunk / 64)`, `y = min(visible_chunk_count, max_visible_chunks)`, `z = 1`. It also clears the shared counter buffer for cluster counting. **The clamp to `max_visible_chunks` is required**: `frustum_cull.glsl:136-140` increments `visible_chunk_count` via `atomicAdd` *before* checking the cap and only writes `visible_chunk_buffer` entries for in-range indices; overflowed chunks are counted but not stored. Using the raw counter as `y` would let the cluster pass read past initialized entries. This is the same clamp pattern as `instance_chunk_dispatch.glsl:26-27` (`min(raw_count, max_visible_chunks)`).
4. New cluster-cull pass reads `visible_chunk_buffer`, `chunk_meta_buffer`, `chunk_cluster_range_buffer`, `cluster_meta_buffer`, and `instance_buffer`; it frustum-tests each cluster, compacts visible clusters into `cluster_visible_buffer`, writes `cluster_splat_ranges_buffer`, and increments `visible_cluster_count`. Apply the same overflow discipline as `frustum_cull.glsl`: count under cap, drop above cap, expose an overflow counter.
5. New `cluster_range_dispatch.glsl` converts `visible_cluster_count` into `cluster_dispatch_buffer`: `x = ceil(max_cluster_splats / 256)`, `y = min(visible_cluster_count, max_visible_clusters)`, `z = 1`; then it clears the shared counter buffer for visible splat counting. Same clamp rationale as step 3.
6. New cluster-aware depth pass expands only visible cluster ranges into `splat_ref_buffer`, `sort_key_buffer`, and `sort_value_buffer`, then `instance_count_clamp.glsl` continues unchanged and publishes `visible_splat_count`.

## 5. Shader / Dispatch-Level Changes

- `modules/gaussian_splatting/compute/instance_cluster_dispatch.glsl`: visible-chunk-count -> indirect cluster-pass dispatch.
- `modules/gaussian_splatting/compute/cluster_cull_instance.glsl`: tier-2 cluster frustum pass; compact visible clusters and splat ranges.
- `modules/gaussian_splatting/compute/cluster_range_dispatch.glsl`: visible-cluster-count -> indirect fine-pass dispatch.
- `modules/gaussian_splatting/compute/cluster_depth_compute.glsl`: cluster-range expansion to `splat_ref_buffer` and sort buffers.
- `modules/gaussian_splatting/shaders/tile_binning.glsl`: no functional change required; it should keep consuming `splat_ref_buffer`.

Workgroup sizes:

- Chunk/cluster cull: `local_size_x = 64`
- Fine expansion: `local_size_x = 256`
- Dispatch/clamp helpers: `1x1x1`

## 6. Import / Cluster Generation

Initial implementation should generate clusters at runtime, not by changing the asset file format. The code already reads chunk-local gaussian snapshots during streaming upload (`gaussian_streaming.cpp`) and resident atlas publish (`resident_instance_contract_publisher.cpp`); cluster metadata should be derived there and cached with the published instance contract generation.

Algorithm:

1. For each chunk snapshot, sort chunk-local splats by Morton code using chunk-local bounds.
2. Partition sorted splats into contiguous groups of `target_cluster_size`, respecting `min/max` cluster size.
3. Compute conservative center/radius from the reordered chunk-local splats.
4. Emit cluster-local splat offsets/counts relative to the chunk, not the atlas.

Storage for the first implementation is in-memory/GPU only. Extending `.gsplatworld` version `1` is deferred; current chunk records are `56` bytes and have no cluster fields.

## 7. Integration With Existing Systems

- PR #237 hotspot pruning remains complementary. Cluster cull removes whole groups before sort/projection; hotspot pruning still operates inside `tile_binning.glsl` on the surviving splats.
- Streaming must load/unload cluster metadata with the same parent chunks that own `chunk_meta_buffer` entries. Cluster buffers are atlas-contract resources, not a separate residency system.
- PR #236 resident contract republish skip should include cluster buffers in the same contract/upload fingerprints so unchanged cluster metadata is not republished.
- Keep fallback simple: if cluster buffers are missing or the feature is disabled, use the current `visible_chunk_buffer -> depth_compute.glsl` path unchanged.

## 8. Tests

- Add `modules/gaussian_splatting/tests/test_cluster_culling.h` and include it from `modules/gaussian_splatting/tests/test_gaussian_splatting.h`.
- Add a GPU integration case beside `test_gpu_streaming.cpp` coverage that publishes tier-2 buffers, runs cluster off/on, and compares the resulting visible splat set with hotspot pruning disabled.
- Invariants:
  - visible splat set must match the non-cluster path for the same frame inputs
  - per-chunk cluster ranges must cover every splat exactly once
  - `visible_cluster_count <= total_cluster_count`
  - cluster-cull GPU time should be lower than the per-splat work it removes on clustered scenes; use reporting, not a hard CI threshold, initially
- Extend `PerformanceBenchmark::benchmark_culling_efficiency()` with a clustered camera sweep and report cluster-off vs cluster-on timings.

## 9. Metrics

Register monitors and snapshot fields for:

- `cluster_total`
- `cluster_visible`
- `cluster_cull_ms`
- `cluster_cull_ratio_pct`

Publish them through `PerformanceMetrics.streaming_state` and `GaussianSplattingPerformanceMonitors`. Add pipeline trace events for:

- contract publication with cluster buffer validity/capacity
- cluster dispatch counts
- visible cluster count and visible splat count after cluster expansion

## 10. Risks And Open Questions

- Small scenes or scenes with few visible chunks may lose performance because the extra dispatches outweigh saved per-splat work.
- Very large clusters may not reject enough work; very small clusters may increase dispatch and buffer pressure.
- Runtime generation avoids format churn but shifts work to upload time.
- Human decision needed before implementation: keep runtime-only generation for v1, or pay the `.gsplatworld` format/version cost to persist cluster metadata offline.

## 11. Implementation Checklist

1. Add an architecture-neutral `ClusterMetaGPU`/`ChunkClusterRangeGPU` layout and optional cluster fields to `InstancePipelineBuffers`.
2. Add contract validation helpers for cluster buffers without breaking the current non-cluster path.
3. Refactor `lod/cluster_builder.*` to build chunk-local sphere clusters and ranges from a chunk snapshot.
4. Generate cluster metadata in the streaming upload path and publish GPU buffers/capacities from `render_streaming_orchestrator.cpp`.
5. Generate the same metadata in `resident_instance_contract_publisher.cpp`.
6. Add `instance_cluster_dispatch.glsl`, `cluster_cull_instance.glsl`, and `cluster_range_dispatch.glsl`.
7. Add `cluster_depth_compute.glsl` and route `GPUSortingPipeline` through it when cluster culling is enabled and buffers are valid.
8. Record cluster metrics in `PerformanceMetrics`, `streaming_state`, performance monitors, and debug trace.
9. Add doctest coverage for cluster build invariants and GPU equivalence against the non-cluster path.
10. Add benchmark reporting and only then decide whether runtime generation is sufficient or file-format persistence is required.
