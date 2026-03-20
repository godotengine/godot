#[compute]

#version 450

#include "includes/platform_compat.glsl"

#ifndef TILE_SIZE
#define TILE_SIZE GS_TILE_SIZE
#endif
#ifndef SPLATS_PER_TILE
#define SPLATS_PER_TILE GS_TILE_SPLAT_CAPACITY
#endif
#include "includes/tile_projection_common.glsl"

#include "includes/gs_instance_layout.glsl"

// ============================================================================
// Shared memory requirement
// ============================================================================
// This shader allocates shared memory for tile-local splat caching:
//   gs_shared_sorted_values:         SPLATS_PER_TILE * 4 bytes  (uint array)
//   gs_shared_projected_gaussians:   SPLATS_PER_TILE * sizeof(ProjectedGaussian) bytes
//     ProjectedGaussian = 9 uints = 36 bytes (or 8 uints = 32 bytes with GS_PACKED_STAGE_DATA)
//   + 4 scalar shared uints = 16 bytes
//
// Total shared memory (default, unpacked): SPLATS_PER_TILE * 40 + 16 bytes
//   e.g. 1024 * 40 + 16 = 40,976 bytes
// Vulkan minimum guaranteed: 16,384 bytes (most desktop GPUs: 32,768-65,536).
// The C++ code that compiles this shader should query maxComputeSharedMemorySize
// at compile time and warn if the requirement exceeds the device limit.
// ============================================================================

#define GS_TILE_RASTER_USE_SHARED 1
#define GS_TILE_RASTER_SHARED_PAYLOAD 1

layout(local_size_x = GS_TILE_LOCAL_SIZE_X, local_size_y = GS_TILE_LOCAL_SIZE_Y, local_size_z = 1) in;

struct Gaussian {
    vec3 position;
    float opacity;

    vec3 scale;
    float area;

    vec4 rotation;

    vec4 sh_dc;
    float sh_encoded[12];

    vec3 normal;
    float stroke_age;

    vec2 brush_axes;
    uint painterly_meta;
    uint sh_metadata;
};

layout(set = 0, binding = 0, std430) readonly buffer GaussianBuffer {
    Gaussian gaussians[];
} gaussian_buffer;

layout(set = 0, binding = 1, std430) readonly buffer SortedIndices {
    uint indices[];
} sorted_indices;

layout(set = 0, binding = 12, std430) readonly buffer SplatRefBuffer {
    SplatRefGPU splat_refs[];
} splat_ref_buffer;

layout(set = 0, binding = 2, std430) readonly buffer TileRanges {
    uvec2 ranges[];
} tile_ranges;

layout(set = 0, binding = 3, std430) buffer OverflowStatisticsBuffer {
    uint overflow_tile_count;
    uint overflow_splats_clamped;
    uint overflow_splats_aggregated;
    uint raster_sample_count;
    uint raster_splats_iterated;
    uint raster_splats_contributed;
    uint raster_reject_sorted_idx_oob;
    uint raster_reject_gaussian_idx_oob;
    uint raster_reject_base_opacity;
    uint raster_reject_nan_inf;
    uint raster_reject_weight;
    uint raster_reject_alpha;
    uint raster_break_remaining_alpha;
    uint raster_break_final_alpha;
    uint raster_has_depth;
    uint raster_alpha_sum_q10;
    uint raster_reject_index_mismatch;
    uint raster_break_subgroup_early_exit;  // Tiles where all pixels were alpha-saturated
} overflow_stats;

layout(set = 0, binding = 4, std430) readonly buffer ProjectionBuffer {
    ProjectedGaussian projected_gaussians[];
} projection_buffer;

layout(set = 0, binding = 5, std430) readonly buffer SortedValues {
    uint values[];
} sorted_values;

// GPU indirect dispatch buffer - provides current-frame element_count
// This avoids using the stale CPU uniform params.overlap_record_count
layout(set = 0, binding = 6, std430) readonly buffer IndirectDispatch {
    uint dispatch_x;
    uint dispatch_y;
    uint dispatch_z;
    uint element_count;  // Current-frame overlap record count from GPU
    uint overflow_flag;
    uint unclamped_total;
} indirect_dispatch;

#include "includes/gs_render_params.glsl"

layout(set = 1, binding = 1, std140) uniform InteractiveState {
    vec4 state_params;
    vec4 highlight_color;
    vec4 outline_color;
} interactive_state;

layout(set = 2, binding = 0, rgba8) writeonly uniform image2D out_color_image;
layout(set = 2, binding = 1, r32f) writeonly uniform image2D out_depth_image;
layout(set = 2, binding = 2, rgba16f) writeonly uniform image2D out_normal_image;

shared uint gs_shared_sorted_values[SPLATS_PER_TILE];
shared uint gs_shared_range_start;
shared uint gs_shared_splat_count;
shared uint gs_shared_total_splat_count;
shared uint gs_shared_original_splat_count;
shared uint gs_shared_all_saturated;
#ifdef GS_TILE_RASTER_SHARED_PAYLOAD
shared ProjectedGaussian gs_shared_projected_gaussians[SPLATS_PER_TILE];
#endif

#include "includes/tile_raster_common.glsl"

// Compute entry point for tile-local batched rasterization.
void main() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    // ALL threads must participate in barrier() and cooperative shared memory loads.
    // Early return before barrier is undefined behavior in GLSL compute shaders.
    bool in_viewport = pixel.x < int(params.viewport_size.x) && pixel.y < int(params.viewport_size.y);

    if (gl_LocalInvocationIndex == 0) {
        ivec2 tile_coord = ivec2(gl_WorkGroupID.xy);
        ivec2 tile_count = ivec2(max(params.tile_count, vec2(1.0)));
        tile_coord = clamp(tile_coord, ivec2(0), tile_count - ivec2(1));

        uint tiles_x = uint(tile_count.x);
        uint tile_idx = uint(tile_coord.y) * tiles_x + uint(tile_coord.x);

        uvec2 range = tile_ranges.ranges[tile_idx];
        uint range_start = range.x;
        uint total_splat_count = range.y;
        uint original_splat_count = total_splat_count;
        // Use GPU's current-frame element_count instead of stale CPU uniform
        uint record_count = indirect_dispatch.element_count;
        if (range_start >= record_count) {
            if (total_splat_count > 0u) {
                atomicAdd(overflow_stats.overflow_tile_count, 1u);
            }
            total_splat_count = 0u;
        } else {
            uint available_records = record_count - range_start;
            if (total_splat_count > available_records) {
                atomicAdd(overflow_stats.overflow_splats_clamped, total_splat_count - available_records);
                total_splat_count = available_records;
            }
        }
        gs_shared_range_start = range_start;
        gs_shared_total_splat_count = total_splat_count;
        gs_shared_original_splat_count = original_splat_count;
    }
    barrier();

    uint range_start = gs_shared_range_start;
    uint splat_count = gs_shared_total_splat_count;
    uint original_splat_count = gs_shared_original_splat_count;
#ifdef GS_MAX_RASTER_SPLATS_PER_TILE
    if (gl_LocalInvocationIndex == 0 && splat_count > uint(GS_MAX_RASTER_SPLATS_PER_TILE)) {
        atomicAdd(overflow_stats.overflow_tile_count, 1u);
        atomicAdd(overflow_stats.overflow_splats_clamped, splat_count - uint(GS_MAX_RASTER_SPLATS_PER_TILE));
    }
    splat_count = min(splat_count, uint(GS_MAX_RASTER_SPLATS_PER_TILE));
#endif
    uint local_invocations = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    uint local_index = gl_LocalInvocationIndex;
    vec2 frag_coord = vec2(pixel) + vec2(0.5);

    // ========================================================================
    // Multi-pass batched rasterization
    // ========================================================================
    // Process splats in batches of SPLATS_PER_TILE.  Each batch is cooperatively
    // loaded into shared memory so that ALL reads hit the fast shared path.
    // Per-pixel blending state is carried across batches.  The workgroup exits
    // early once all pixels in the tile are alpha-saturated.

    // Per-pixel state carried across batches
    vec4 final_color = vec4(0.0);
    float final_depth = 1.0;
    vec3 final_normal = vec3(0.0);
    bool has_depth = false;
    bool pixel_saturated = false;

    // Pre-compute per-pixel invariants
    vec2 pixel_center = frag_coord - vec2(0.5) + vec2(0.5); // == frag_coord
    float lod_blend = gs_get_lod_blend_factor();
    vec3 pixel_dither = gs_dither_noise_rgb(pixel_center) * GS_DITHER_AMPLITUDE;
    float highlight_strength = interactive_state.state_params.x;
    float outline_width = interactive_state.state_params.y;
    uint render_state = uint(interactive_state.state_params.z + 0.5);

    for (uint batch_start = 0u; batch_start < splat_count; batch_start += uint(SPLATS_PER_TILE)) {
        uint batch_end = min(batch_start + uint(SPLATS_PER_TILE), splat_count);
        uint batch_size = batch_end - batch_start;

        // Cooperative shared memory load for this batch
        if (gl_LocalInvocationIndex == 0) {
            gs_shared_splat_count = batch_size;
            gs_shared_all_saturated = 1u;
        }
        barrier();

        for (uint i = local_index; i < batch_size; i += local_invocations) {
            uint sorted_idx = sorted_values.values[range_start + batch_start + i];
            gs_shared_sorted_values[i] = sorted_idx;
#ifdef GS_TILE_RASTER_SHARED_PAYLOAD
            if (sorted_idx < gs_get_visible_gaussian_count()) {
                gs_shared_projected_gaussians[i] = projection_buffer.projected_gaussians[sorted_idx];
            }
#endif
        }
        barrier();

        // Per-pixel: rasterize this batch from shared memory
        if (in_viewport && !pixel_saturated) {
            pixel_saturated = gs_rasterize_splat_batch(
                    pixel_center, batch_size, lod_blend, pixel_dither,
                    highlight_strength, outline_width, render_state,
                    final_color, final_depth, final_normal, has_depth);
            if (!pixel_saturated) {
                atomicAnd(gs_shared_all_saturated, 0u);
            }
        }
        barrier();

        // Workgroup-wide early exit: all pixels are alpha-saturated
        if (gs_shared_all_saturated != 0u) {
            break;
        }
    }

    // Out-of-viewport threads participated in barriers above; now exit.
    if (!in_viewport) {
        return;
    }

    // ========================================================================
    // Output
    // ========================================================================
    float depth_out = has_depth ? final_depth : 1.0;

    // Debug overlay paths (same as gs_rasterize_pixel finalization)
    bool debug_tile_grid = params.debug_flags.x > 0.5;
    bool debug_tiles = params.debug_flags.z > 0.5;
    bool debug_density_heatmap = params.debug_overlay_flags.x > 0.5;
    bool debug_depth_visualization = params.debug_overlay_flags.y > 0.5;

    if (debug_tiles) {
        vec3 debug_color = vec3(0.0, 0.0, 0.3);
        float avg_records = max(1.0, float(indirect_dispatch.element_count) / max(1.0, params.tile_count.x * params.tile_count.y));
        float occupancy = clamp(float(splat_count) / avg_records, 0.0, 4.0) * 0.25;
        bool tile_clamped = original_splat_count > splat_count;
        if (original_splat_count == 0u) {
            debug_color = vec3(0.0, 0.0, 0.3);
        } else if (tile_clamped) {
            debug_color = vec3(0.85, 0.1, 0.1);
        } else if (!has_depth) {
            debug_color = vec3(0.7, 0.0, 0.7);
        } else {
            debug_color = vec3(0.05, 0.2 + 0.8 * occupancy, 0.05);
        }
        vec2 tile_uv = fract(frag_coord / float(TILE_SIZE));
        float edge = 1.0 / float(TILE_SIZE);
        bool on_edge = (tile_uv.x < edge) || (tile_uv.x > (1.0 - edge)) || (tile_uv.y < edge) || (tile_uv.y > (1.0 - edge));
        if (on_edge) { debug_color *= 0.35; }
        if (debug_tile_grid) { debug_color = gs_apply_tile_grid(frag_coord, debug_color, params.debug_overlay_opacity); }
        imageStore(out_color_image, pixel, vec4(debug_color, 1.0));
        imageStore(out_depth_image, pixel, vec4(1.0, 0.0, 0.0, 0.0));
        imageStore(out_normal_image, pixel, vec4(0.0));
        return;
    }

    if (debug_density_heatmap) {
        float avg_records = max(1.0, float(indirect_dispatch.element_count) / max(1.0, params.tile_count.x * params.tile_count.y));
        float density = clamp(float(splat_count) / avg_records, 0.0, 4.0) * 0.25;
        vec3 heat_color = gs_spectral_heatmap(density);
        if (debug_tile_grid) { heat_color = gs_apply_tile_grid(frag_coord, heat_color, params.debug_overlay_opacity); }
        imageStore(out_color_image, pixel, vec4(heat_color, 1.0));
        imageStore(out_depth_image, pixel, vec4(depth_out, 0.0, 0.0, 0.0));
        imageStore(out_normal_image, pixel, vec4(0.0));
        return;
    }

    if (debug_depth_visualization) {
        float depth_range = max(params.far_plane - params.near_plane, 0.001);
        float depth_norm = clamp((depth_out - params.near_plane) / depth_range, 0.0, 1.0);
        vec3 depth_color = gs_spectral_heatmap(1.0 - depth_norm);
        if (debug_tile_grid) { depth_color = gs_apply_tile_grid(frag_coord, depth_color, params.debug_overlay_opacity); }
        imageStore(out_color_image, pixel, vec4(depth_color, 1.0));
        imageStore(out_depth_image, pixel, vec4(depth_out, 0.0, 0.0, 0.0));
        imageStore(out_normal_image, pixel, vec4(0.0));
        return;
    }

    // Production path
    if (splat_count == 0u || !has_depth) {
        imageStore(out_color_image, pixel, vec4(0.0));
        imageStore(out_depth_image, pixel, vec4(1.0, 0.0, 0.0, 0.0));
        imageStore(out_normal_image, pixel, vec4(0.0));
        return;
    }

    final_color.a = clamp(final_color.a + pixel_dither.r, 0.0, 1.0);
    if (params.force_solid_coverage != 0u) {
        final_color.a = max(final_color.a, params.alpha_floor);
    }
    if (debug_tile_grid) {
        final_color.rgb = gs_apply_tile_grid(frag_coord, final_color.rgb, params.debug_overlay_opacity);
    }
    imageStore(out_color_image, pixel, clamp(final_color, vec4(0.0), vec4(1.0)));
    imageStore(out_depth_image, pixel, vec4(depth_out, 0.0, 0.0, 0.0));
    imageStore(out_normal_image, pixel, vec4(final_normal, final_color.a));
}
