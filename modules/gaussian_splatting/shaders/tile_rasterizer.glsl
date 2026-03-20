#[vertex]

#version 450

layout(location = 0) out vec2 uv;

// Vertex entry point for the fullscreen raster quad.
void main() {
    vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    uv = pos;
    pos = pos * 2.0 - 1.0;
    gl_Position = vec4(pos, 0.0, 1.0);
}

#[fragment]

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

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_depth;
layout(location = 2) out vec4 out_normal;

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

#include "includes/tile_raster_common.glsl"

// Fragment entry point for tile rasterization and compositing.
void main() {
    // Compute tile info first for diagnostics
    vec2 frag_coord = gl_FragCoord.xy;
    vec2 viewport_size = max(params.viewport_size, vec2(1.0));

    ivec2 tile_coord = ivec2((frag_coord - vec2(0.5)) / float(TILE_SIZE));
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

    // Clamp to available records first.
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

    uint splat_count = total_splat_count;
#ifdef GS_MAX_RASTER_SPLATS_PER_TILE
    splat_count = min(splat_count, uint(GS_MAX_RASTER_SPLATS_PER_TILE));
#endif

    vec4 raster_color;
    float raster_depth;
    vec4 raster_normal;
    gs_rasterize_pixel(frag_coord, range_start, splat_count, original_splat_count, raster_color, raster_depth, raster_normal);

    out_color = raster_color;
    out_depth = vec4(raster_depth, 0.0, 0.0, 0.0);
    out_normal = raster_normal;
}
