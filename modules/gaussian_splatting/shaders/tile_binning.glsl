// tile_binning.glsl — Tile-based Gaussian binning compute shader.
//
// Core projection, tile overlap counting (COUNT pass), and sort-key emission
// (EMIT pass) for the tile-based Gaussian Splatting renderer.
//
// Extracted helper includes (ISSUE-030):
//   includes/gs_sh_binning.glsl       — SH metadata, basis, evaluation
//   includes/gs_eigen_binning.glsl    — EigenInfo, compute_eigen, opacity-aware sigma
//   includes/gs_quat_utils.glsl       — quaternion_to_matrix (no-normalize), rotate, mul
//   includes/gs_culling_utils.glsl    — distance culling, hashing, overlap filter

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
#include "includes/quantization_dequant.glsl"

#include "includes/gs_render_params.glsl"
#include "includes/gs_deformation.glsl"

// Define GS_COMPUTE_SHADER to enable compute shader compatibility shim
// for gl_FragCoord (used in Godot's scene_forward_lights_inc.glsl)
#define GS_COMPUTE_SHADER
#include "includes/gs_lighting_bridge.glsl"
#include "includes/gs_directional_shadow.glsl"
#include "includes/gs_lighting_common.glsl"

#ifndef GS_SORT_KEY_BITS
#define GS_SORT_KEY_BITS 64
#define GS_SORT_TILE_BITS 32
#define GS_SORT_DEPTH_BITS 32
#endif

layout(local_size_x = GS_DISPATCH_LOCAL_SIZE_X, local_size_y = 1, local_size_z = 1) in;

struct Gaussian {
    vec3 position;
    float opacity;

    vec3 scale;
    float area;

    vec4 rotation; // Quaternion

    vec4 sh_dc;
    float sh_encoded[12];

    vec3 normal;
    float stroke_age;

    vec2 brush_axes;
    uint painterly_meta; // lower 16 bits: palette id
    uint sh_metadata;     // packed SH layout metadata
};


const float GAUSSIAN_EPSILON = 1e-6;
// MIN_VARIANCE must ensure: sqrt(MIN_VARIANCE) * MAX_SIGMA >= MIN_SPLAT_RADIUS
// With MIN_SPLAT_RADIUS=0.1 and MAX_SIGMA=3.0: need sqrt(MIN_VARIANCE) >= 0.0333
// Using 0.002 gives sqrt(0.002)=0.0447, yielding min_radius = 0.134 pixels
const float MIN_VARIANCE = 0.002;
const float MAX_SIGMA = 3.0;

#include "includes/gs_sh_binning.glsl"
#include "includes/gs_eigen_binning.glsl"

#if defined(USE_QUANTIZED_GAUSSIANS)
layout(set = 0, binding = 0, std430) readonly buffer AtlasGaussianBuffer {
    GaussianQuantized gaussians[];
} atlas_gaussian_buffer;
#else
layout(set = 0, binding = 0, std430) readonly buffer AtlasGaussianBuffer {
    Gaussian gaussians[];
} atlas_gaussian_buffer;
#endif

layout(set = 0, binding = 1, std430) readonly buffer SortedIndices {
    uint indices[];
} sorted_indices;

layout(set = 0, binding = 12, std430) readonly buffer SplatRefBuffer {
    SplatRefGPU splat_refs[];
} splat_ref_buffer;

layout(set = 0, binding = 13, std430) readonly buffer InstanceBuffer {
    InstanceDataGPU instances[];
} instance_buffer;

#if defined(USE_QUANTIZED_GAUSSIANS)
layout(set = 0, binding = 14, std430) readonly buffer QuantizationChunkBuffer {
    ChunkQuantization chunks[];
} quantization_buffer;
#endif

layout(set = 0, binding = 15, std430) readonly buffer InstanceIndirectDispatch {
    uint dispatch_xyz[3];
    uint element_count;
    uint overflow_flag;
    uint unclamped_total;
} instance_indirect;

// Subpixel hysteresis history (packed key + visible flag).
layout(set = 0, binding = 16, std430) buffer SubpixelHistory {
    uint entries[];
} subpixel_history;

// Per-frame subpixel visibility (COUNT computes, EMIT consumes).
layout(set = 0, binding = 17, std430) buffer SubpixelVisibility {
    uint visible[];
} subpixel_visibility;

// Return the number of visible Gaussians scheduled for this dispatch.
uint gs_get_visible_gaussian_count() {
    return instance_indirect.element_count;
}

// Overflow statistics tracking
layout(set = 0, binding = 3, std430) buffer OverflowStats {
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

layout(set = 0, binding = 5, std430) buffer TileCounts {
    uint counts[];
} tile_counts;

#ifdef GS_TILE_GLOBAL_SORT_EMIT_PASS
layout(set = 0, binding = 2, std430) buffer GlobalSortKeys {
#if GS_SORT_KEY_BITS == 32
    uint keys[];
#else
    uvec2 keys[];
#endif
} global_sort_keys;

layout(set = 0, binding = 4, std430) buffer GlobalSortValues {
    uint values[];
} global_sort_values;

layout(set = 0, binding = 7, std430) buffer ProjectionBuffer {
    ProjectedGaussian projected_gaussians[];
} projection_buffer;

#endif

#ifdef GS_TILE_GLOBAL_SORT_EMIT_PASS
layout(set = 0, binding = 8, std430) readonly buffer TileRanges {
    uvec2 ranges[];
} tile_ranges;

layout(set = 0, binding = 9, std430) readonly buffer IndirectDispatch {
    uint dispatch_xyz[3];
    uint element_count;
    uint overflow_flag;
    uint unclamped_total;
} indirect_dispatch;
#endif

// Debug counters for tracking rejection reasons
// Note: Binding 6 chosen to avoid conflict with tile_rasterizer which uses bindings 3-5
layout(set = 0, binding = 6, std430) buffer DebugCounters {
    uint total_processed;
    uint near_far_reject;
    uint view_distance_reject;
    uint quaternion_reject;
    uint scale_reject;
    uint clip_w_reject;
    uint clip_bounds_reject;
    uint screen_nan_reject;
    uint focal_length_reject;
    uint z_inverse_reject;
    uint covariance_nan_reject;
    uint determinant_reject;
    uint radius_reject;
    uint distance_cull_reject;
    uint viewport_bounds_reject;
    uint bbox_integrity_reject;
    uint tile_extent_reject;
    uint success_count;
    uint extreme_conic_count;
    uint index_mismatch_count; // reserved for raster index validation (shared layout)
    // Diagnostic counters for radial stretching investigation
    uint depth_discrepancy_count;      // splats where safe_depth > positive_depth
    uint depth_discrepancy_sum_q8;     // sum of (safe_depth - positive_depth) in Q8.8 fixed-point
    uint high_aspect_ratio_count;      // splats with aspect > 5 before clamping
    uint max_aspect_q8;                // max aspect ratio seen (Q8.8 fixed-point, post-clamp)
    uint max_aspect_preclamp_q8;       // max aspect ratio BEFORE any clamping (Q8.8)
    uint j_col2_clamp_count;           // splats where J_col2 hit the ±1e4 clamp
    uint sh_cache_hits;                // SH amortization cache hits
    uint sh_cache_updates;             // SH amortization cache updates
    uint sh_cache_forced_updates;      // SH amortization forced refreshes
    // Subpixel culling diagnostics (Q8.8 fixed-point, 256 = 1.0 px)
    uint tiny_splat_param_q8;          // params.tiny_splat_screen_radius (max across threads)
    uint min_allowed_radius_q8;        // max(MIN_SPLAT_RADIUS, tiny_splat) (max across threads)
    uint min_radius_min_q8_inv;        // inverted min(min_radius) across threads: 0xFFFFFFFF - q8
} debug_counters;

const uint GS_DEBUG_SPLAT_AUDIT_MAX_SAMPLES = 64u;
const uint GS_AUDIT_FLAG_PROJECTED = 1u << 0u;
const uint GS_AUDIT_FLAG_IN_VIEWPORT = 1u << 1u;
const uint GS_AUDIT_FLAG_ITERATED = 1u << 2u;
const uint GS_AUDIT_FLAG_CONTRIBUTED = 1u << 3u;
const uint GS_AUDIT_FLAG_ALPHA_SKIPPED = 1u << 4u;

struct DebugSplatAuditEntry {
    uint global_idx;
    uint expected_x;
    uint expected_y;
    uint flags;
};

layout(set = 0, binding = 10, std430) buffer DebugSplatAudit {
    uint enabled;
    uint sample_count;
    uint frame_id;
    uint reserved;
    DebugSplatAuditEntry entries[];
} debug_audit;

#if defined(GS_SH_AMORTIZATION) && defined(GS_TILE_GLOBAL_SORT_EMIT_PASS)
layout(set = 0, binding = 11, std430) buffer SHColorCache {
    uint colors[];
} sh_color_cache;
#endif

#ifdef GS_TILE_GLOBAL_SORT_EMIT_PASS
#if GS_SORT_KEY_BITS == 32
// Pack tile index and depth into a global sort key.
uint gs_pack_sort_key(uint tile_idx, float linear_depth) {
    float clamped_depth = clamp(linear_depth, 0.0, 0.999999);
#if GS_SORT_DEPTH_BITS >= 32
    uint depth_key = floatBitsToUint(clamped_depth);
    return depth_key;
#else
    uint depth_mask = (1u << GS_SORT_DEPTH_BITS) - 1u;
    uint depth_key = uint(clamped_depth * float(depth_mask));
#if GS_SORT_TILE_BITS >= 32
    uint tile_mask = 0xFFFFFFFFu;
#else
    uint tile_mask = (1u << GS_SORT_TILE_BITS) - 1u;
#endif
    uint tile_key = tile_idx & tile_mask;
    return (tile_key << GS_SORT_DEPTH_BITS) | depth_key;
#endif
}
#else
// Pack tile index and depth into a global sort key.
uvec2 gs_pack_sort_key(uint tile_idx, float linear_depth) {
    return uvec2(floatBitsToUint(linear_depth), tile_idx);
}
#endif
#endif

// ============================================================================
// Debug counter macros - can be disabled for production builds
// ============================================================================
// In production, define GS_DEBUG_COUNTERS_DISABLED to eliminate atomic storms.
// Each debug atomicAdd costs ~10-20 cycles and causes L2 cache thrashing.
// With 84K splats, this can waste millions of GPU cycles per frame.
//
// NOTE: We use simple per-thread atomics even when subgroups are available
// because many counters are in early-return paths where subgroup ballot
// operations would be undefined behavior (not all threads participate).
#ifdef GS_DEBUG_COUNTERS_DISABLED
    #define GS_DEBUG_INCREMENT(counter)
    #define GS_DEBUG_INCREMENT_BY(counter, value)
    #define GS_DEBUG_MAX(counter, value)
#else
    #define GS_DEBUG_INCREMENT(counter) atomicAdd(debug_counters.counter, 1u)
    #define GS_DEBUG_INCREMENT_BY(counter, value) atomicAdd(debug_counters.counter, (value))
    #define GS_DEBUG_MAX(counter, value) atomicMax(debug_counters.counter, (value))
#endif

#include "includes/color_grading_binning.glsl"
#include "includes/gs_quat_utils.glsl"
#include "includes/gs_culling_utils.glsl"

// Pack quantized spherical-harmonic metadata for the renderer.
uint gs_build_quantized_sh_metadata(uint encoded_total) {
    uint first_count = min(encoded_total, 3u);
    uint high_count = encoded_total > first_count ? (encoded_total - first_count) : 0u;
    return first_count | (high_count << 8u) | (encoded_total << 16u) | (SH_ENCODING_RGB9E5 << 24u);
}

// Project a Gaussian into screen space and derive its 2D covariance.
vec3 project_gaussian_2d(Gaussian g, out vec2 screen_pos, out mat2 cov2d, out float linear_depth, out float raw_min_radius) {
    raw_min_radius = 0.0;  // Default: will be set after valid cov2d computation
    linear_depth = 1.0;
    // WARNING: Do NOT negate view_pos.x here - it causes "splats rotate with camera" bug.
    // The X-axis offset issue must be fixed elsewhere (likely in PLY loader or data transform).
    vec4 view_pos = params.view_matrix * vec4(g.position, 1.0);

    float positive_depth = -view_pos.z;
    // Apply a tolerance band on near/far to reduce popping at clip planes.
    float clip_tolerance = max(params.cull_far_tolerance, 0.0);
    float near_margin = params.near_plane * (1.0 - clip_tolerance);
    float far_margin = params.far_plane * (1.0 + clip_tolerance);

    if (positive_depth > far_margin) {
        GS_DEBUG_INCREMENT(near_far_reject);
        return vec3(0.0);
    }
    if (positive_depth < near_margin) {
        // Reject near-plane/behind-camera splats.
        // Clamping these to near caused large translucent overlays at close range.
        GS_DEBUG_INCREMENT(near_far_reject);
        return vec3(0.0);
    }

    // Validate quaternion normalization (should be close to unit length)
    // Optimization: Use squared length to avoid sqrt() - check if |q|² is in [0.81, 1.21]
    // Original: abs(length(q) - 1.0) > 0.1, i.e., length in [0.9, 1.1]
    // Squared: length² in [0.81, 1.21]
    float quat_length_sq = dot(g.rotation, g.rotation);
    if (quat_length_sq < 0.81 || quat_length_sq > 1.21) {
        GS_DEBUG_INCREMENT(quaternion_reject);
        return vec3(0.0); // Invalid quaternion
    }

    // Validate scale values are reasonable
    if (any(lessThan(g.scale, vec3(1e-6))) || any(greaterThan(g.scale, vec3(100.0)))) {
        GS_DEBUG_INCREMENT(scale_reject);
        return vec3(0.0); // Invalid scale
    }

    // PERF-5 (#676): Fast quaternion correction instead of full normalize()
    // Since we validated quat_length_sq is in [0.81, 1.21], we can use a single Newton-Raphson
    // iteration of inverse sqrt for fast correction: inv_sqrt(x) ≈ 0.5 * (3 - x)
    // For x in [0.81, 1.21], this gives error < 1%, acceptable for rotation matrices
    float inv_len = 0.5 * (3.0 - quat_length_sq);  // Fast approx of 1/sqrt(quat_length_sq)
    vec4 q_corrected = g.rotation * inv_len;
    mat3 rotation = quaternion_to_matrix(q_corrected);

    // PERF-4 (#675): Optimized 3D covariance computation
    // Instead of building scale2 matrix and doing R * S² * R^T (27 mul + 18 add),
    // we compute only the 6 unique elements of the symmetric cov3d directly.
    // For R = [r0, r1, r2] (columns) and s² = scale², the covariance is:
    // cov3d[i][j] = sum_k(R[i][k] * s²[k] * R[j][k])
    // This reduces to ~30 ops (vs ~45 ops for mat3 multiply)
    vec3 s2 = g.scale * g.scale;

    // Scale each column of rotation matrix by corresponding squared scale
    vec3 rs0 = rotation[0] * s2.x;  // First column scaled
    vec3 rs1 = rotation[1] * s2.y;  // Second column scaled
    vec3 rs2 = rotation[2] * s2.z;  // Third column scaled

    // Compute symmetric 3x3 covariance: cov3d = R * diag(s²) * R^T
    // Only compute 6 unique elements (upper triangle)
    float c00 = rs0.x * rotation[0].x + rs1.x * rotation[1].x + rs2.x * rotation[2].x;
    float c01 = rs0.x * rotation[0].y + rs1.x * rotation[1].y + rs2.x * rotation[2].y;
    float c02 = rs0.x * rotation[0].z + rs1.x * rotation[1].z + rs2.x * rotation[2].z;
    float c11 = rs0.y * rotation[0].y + rs1.y * rotation[1].y + rs2.y * rotation[2].y;
    float c12 = rs0.y * rotation[0].z + rs1.y * rotation[1].z + rs2.y * rotation[2].z;
    float c22 = rs0.z * rotation[0].z + rs1.z * rotation[1].z + rs2.z * rotation[2].z;

    mat3 cov3d = mat3(
        c00, c01, c02,
        c01, c11, c12,
        c02, c12, c22
    );

    // Bring covariance into view space before projection (W term from 3DGS Eq. 5)
    mat3 view_rot = mat3(params.view_matrix);
    cov3d = view_rot * cov3d * transpose(view_rot);

    vec4 clip_pos = params.projection_matrix * view_pos;

    // Validate clip space coordinates before perspective divide
    if (abs(clip_pos.w) < 1e-6) {
        GS_DEBUG_INCREMENT(clip_w_reject);
        return vec3(0.0); // Avoid division by near-zero
    }

    // Check for extreme clip coordinates that could cause precision issues
    // Use very relaxed threshold (1000x) - viewport clamping handles off-screen cases safely
    // Overly aggressive checks here cause view-dependent holes
    if (any(greaterThan(abs(clip_pos.xyz), vec3(abs(clip_pos.w) * 1000.0)))) {
        GS_DEBUG_INCREMENT(clip_bounds_reject);
        return vec3(0.0); // Outside reasonable clip bounds
    }

    screen_pos = (clip_pos.xy / clip_pos.w) * 0.5 + 0.5;
    screen_pos *= params.viewport_size;

    // Validate final screen coordinates
    if (any(isnan(screen_pos)) || any(isinf(screen_pos))) {
        GS_DEBUG_INCREMENT(screen_nan_reject);
        return vec3(0.0); // Invalid screen position
    }

    // Signed focal lengths - preserve flip_y sign for correct Jacobian orientation
    float focal_x = params.projection_matrix[0][0] * params.viewport_size.x * 0.5;
    float focal_y = params.projection_matrix[1][1] * params.viewport_size.y * 0.5;
    // Absolute values for validation only
    float focal_x_abs = abs(focal_x);
    float focal_y_abs = abs(focal_y);

    // Validate focal lengths are reasonable (use absolute values)
    if (focal_x_abs < 1.0 || focal_y_abs < 1.0 || focal_x_abs > params.viewport_size.x * 2.0 || focal_y_abs > params.viewport_size.y * 2.0) {
        GS_DEBUG_INCREMENT(focal_length_reject);
        return vec3(0.0); // Invalid focal length
    }

    float near_far_range = max(params.far_plane - params.near_plane, 1e-4);
    linear_depth = clamp((positive_depth - params.near_plane) / near_far_range, 0.0, 1.0);

    float depth = positive_depth;
    float clamped_depth = clamp(depth, params.near_plane, params.far_plane);

    // Radius-aware minimum depth: when camera is inside/near a large splat, use the splat's
    // radius to set a floor on the Jacobian depth. This prevents covariance explosion that
    // causes blocky artifacts instead of proper ellipses.
    float splat_radius = max(g.scale.x, max(g.scale.y, g.scale.z));

    // Diagnostic toggle: bypass_radius_depth_floor (jacobian_diag_flags.x > 0.5)
    float min_safe_depth;
    if (params.jacobian_diag_flags.x > 0.5) {
        // Bypass: use only near_plane, ignore radius-based floor
        min_safe_depth = params.near_plane;
    } else {
        // Normal: radius-aware depth floor
        min_safe_depth = max(params.near_plane, max(0.02, splat_radius * 0.5));
    }

    if (clamped_depth < 1e-6) {
        GS_DEBUG_INCREMENT(z_inverse_reject);
        return vec3(0.0);
    }

    float safe_depth = max(clamped_depth, min_safe_depth);

    // Diagnostic: track when safe_depth differs significantly from actual depth
    float depth_diff = safe_depth - positive_depth;
    if (depth_diff > 0.01) {  // More than 1cm discrepancy
        GS_DEBUG_INCREMENT(depth_discrepancy_count);
        // Store sum as Q8.8 fixed-point (max ~16 million before overflow)
        uint diff_q8 = uint(clamp(depth_diff * 256.0, 0.0, 65535.0));
        GS_DEBUG_INCREMENT_BY(depth_discrepancy_sum_q8, diff_q8);
    }

    // Use NEGATIVE z_inv to match the view-space depth convention (negative for visible)
    // Legacy: z_inv = 1.0 / view_pos.z where view_pos.z < 0
    // Tile: safe_depth = -view_pos.z > 0, so we negate to get same z_inv sign
    float z_inv = -1.0 / safe_depth;
    float z_inv_sq = z_inv * z_inv;

    // Columns of the Jacobian (column-major layout)
    // With negative z_inv, J[0][0] = focal_x * z_inv (negative if focal_x > 0)
    vec3 J_col0 = vec3(focal_x * z_inv, 0.0, 0.0);
    vec3 J_col1 = vec3(0.0, focal_y * z_inv, 0.0);
    // J_col2: Legacy uses -focal_x * view_pos.x * z_inv_sq
    // With z_inv_sq always positive, we need explicit negative sign
    //
    // Diagnostic toggle: invert_j_col2_sign (jacobian_diag_flags.z > 0.5) flips sign
    float j2_sign = (params.jacobian_diag_flags.z > 0.5) ? 1.0 : -1.0;
    float j2_x_raw = j2_sign * focal_x * view_pos.x * z_inv_sq;
    float j2_y_raw = j2_sign * focal_y * view_pos.y * z_inv_sq;

    // Diagnostic toggle: bypass_j_col2_clamp (jacobian_diag_flags.y > 0.5)
    float j2_x, j2_y;
    if (params.jacobian_diag_flags.y > 0.5) {
        // Bypass: no clamping
        j2_x = j2_x_raw;
        j2_y = j2_y_raw;
    } else {
        // Normal: clamp to ±1e4
        j2_x = clamp(j2_x_raw, -1e4, 1e4);
        j2_y = clamp(j2_y_raw, -1e4, 1e4);
        // Track when clamp is hit
        if (abs(j2_x_raw) > 1e4 || abs(j2_y_raw) > 1e4) {
            GS_DEBUG_INCREMENT(j_col2_clamp_count);
        }
    }
    vec3 J_col2 = vec3(j2_x, j2_y, 0.0);
    mat3 J = mat3(J_col0, J_col1, J_col2);

    mat3 cov_proj = J * cov3d * transpose(J);
    cov2d = mat2(cov_proj[0][0], cov_proj[0][1], cov_proj[1][0], cov_proj[1][1]);

    // Validate projected covariance for numerical stability
    if (any(isnan(cov2d[0])) || any(isnan(cov2d[1])) ||
        any(isinf(cov2d[0])) || any(isinf(cov2d[1]))) {
        GS_DEBUG_INCREMENT(covariance_nan_reject);
        return vec3(0.0); // Invalid covariance projection
    }

    // SUBPIXEL CULLING (#797): Compute raw minor eigenvalue BEFORE low-pass filter.
    // The low-pass filter adds a minimum variance floor for stability, which can
    // otherwise inflate tiny splats and defeat subpixel culling.
    // Capture true minor eigenvalue here for distance-based culling.
    {
        float trace_raw = cov2d[0][0] + cov2d[1][1];
        float det_raw = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
        det_raw = max(det_raw, 1e-8);  // Prevent negative sqrt
        float disc_raw = max(trace_raw * trace_raw * 0.25 - det_raw, 0.0);
        float root_raw = sqrt(disc_raw);
        float lambda_min_raw = max(trace_raw * 0.5 - root_raw, 1e-8);
        raw_min_radius = sqrt(lambda_min_raw);  // Minor eigenvalue (true screen-space radius)
    }

    // Low-pass filter to prevent degenerate ellipses and view-dependent holes.
    // Standard EWA/mip-splatting technique: add minimum variance to the diagonal.
    // The value is runtime configurable to balance stability (higher) vs sharpness (lower).
    float low_pass_filter = clamp(params._pad_before_jacobian, 0.05, 2.0);
    cov2d[0][0] += low_pass_filter;
    cov2d[1][1] += low_pass_filter;

    float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
    det = max(det, low_pass_filter * low_pass_filter);

    float inv_det = 1.0 / det;
    return vec3(cov2d[1][1] * inv_det, -cov2d[0][1] * inv_det, cov2d[0][0] * inv_det);
}

// Compute entry point for the active tile binning pass.
void main() {
    uint global_idx = gl_GlobalInvocationID.x;

    if (global_idx >= gs_get_visible_gaussian_count()) {
        return;
    }

    GS_DEBUG_INCREMENT(total_processed);

    uint splat_ref_idx = sorted_indices.indices[global_idx];
    if (splat_ref_idx >= gs_get_visible_gaussian_count()) {
        GS_DEBUG_INCREMENT(index_mismatch_count);
        return;
    }
    SplatRefGPU splat_ref = splat_ref_buffer.splat_refs[splat_ref_idx];
    uint gaussian_idx = splat_ref.atlas_index;
    if (gaussian_idx >= params.total_gaussians) {
        GS_DEBUG_INCREMENT(index_mismatch_count);
        return;
    }
    InstanceDataGPU instance = instance_buffer.instances[splat_ref.instance_id];

#if defined(USE_QUANTIZED_GAUSSIANS)
    GaussianQuantized src = atlas_gaussian_buffer.gaussians[gaussian_idx];
    uint quant_id = extract_chunk_id(src.position_chunk);
    ChunkQuantization quant = quantization_buffer.chunks[quant_id];
    vec3 local_position = dequantize_position(extract_quantized_position(src.position_chunk), quant);
    vec3 local_scale = LOAD_SCALE_QUANTIZED(src, quant);
    vec4 local_rotation = extract_rotation(src.rotation_lo, src.rotation_hi);

    Gaussian g;
    g.opacity = src.opacity;
    g.sh_dc = src.sh_dc;
    g.sh_metadata = gs_build_quantized_sh_metadata(6u);
    for (int i = 0; i < 12; ++i) {
        g.sh_encoded[i] = 0.0;
    }
    g.sh_encoded[0] = uintBitsToFloat(src.sh_encoded_01.x);
    g.sh_encoded[1] = uintBitsToFloat(src.sh_encoded_01.y);
    g.sh_encoded[2] = uintBitsToFloat(src.sh_encoded_23.x);
    g.sh_encoded[3] = uintBitsToFloat(src.sh_encoded_23.y);
    g.sh_encoded[4] = uintBitsToFloat(src.sh_encoded_45.x);
    g.sh_encoded[5] = uintBitsToFloat(src.sh_encoded_45.y);
    g.normal = vec3(0.0);
    g.stroke_age = 0.0;
    g.brush_axes = vec2(0.0);
    g.painterly_meta = 0u;
    g.area = 0.0;
#else
    Gaussian g = atlas_gaussian_buffer.gaussians[gaussian_idx];
    vec3 local_position = g.position;
    vec3 local_scale = g.scale;
    vec4 local_rotation = g.rotation;
#endif

    uint instance_flags = instance.ids.y;
    float uniform_scale = abs(instance.translation_scale.w);
    vec3 scaled_position = (instance_flags & GS_INSTANCE_FLAG_SCALE_IDENTITY) != 0u
            ? local_position
            : local_position * uniform_scale;
    vec3 world_position = (instance_flags & GS_INSTANCE_FLAG_ROTATION_IDENTITY) != 0u
            ? scaled_position
            : gs_quat_rotate(instance.rotation, scaled_position);
    if ((instance_flags & GS_INSTANCE_FLAG_TRANSLATION_ZERO) == 0u) {
        world_position += instance.translation_scale.xyz;
    }
    float instance_intensity = max(instance.params.z, 0.0);
    float instance_wind_mode = instance.params.w;
    uint stable_seed = gaussian_idx ^ (splat_ref.instance_id * 0x9e3779b9u);
    world_position = gs_apply_wind_deformation(world_position,
            stable_seed,
            g.opacity,
            instance_intensity,
            instance_wind_mode,
            instance.wind_params,
            params.wind_dir_strength,
            params.wind_time_config,
            params.effector_sphere,
            params.effector_config);
    vec3 world_scale = (instance_flags & GS_INSTANCE_FLAG_SCALE_IDENTITY) != 0u
            ? local_scale
            : local_scale * uniform_scale;
    vec4 world_rotation = (instance_flags & GS_INSTANCE_FLAG_ROTATION_IDENTITY) != 0u
            ? local_rotation
            : gs_quat_mul(instance.rotation, local_rotation);

    g.position = world_position;
    g.scale = world_scale;
    g.rotation = world_rotation;

    vec2 screen_pos;
    mat2 cov2d;
    float linear_depth;
    float raw_min_radius_proj;  // Pre-low-pass-filter minor radius for subpixel culling (#797)
    vec3 conic = project_gaussian_2d(g, screen_pos, cov2d, linear_depth, raw_min_radius_proj);
    if (conic == vec3(0.0)) {
        // Counter already incremented in project_gaussian_2d
        return;
    }

    float world_dist = length(params.camera_position.xyz - g.position);
    // Use a stable per-splat key for distance culling; global_idx is rank-based and
    // changes with camera motion as sorted order changes, which causes temporal flicker.
    uint distance_cull_key = gaussian_idx;
    if (splat_ref.instance_id != 0u) {
        distance_cull_key = gs_hash_u32(distance_cull_key ^ (splat_ref.instance_id * 0x9e3779b9u));
    }
    if (gs_should_distance_cull(distance_cull_key, world_dist)) {
        GS_DEBUG_INCREMENT(distance_cull_reject);
        return;
    }

    // Align screen_pos with packed precision to keep binning consistent with rasterization.
    vec2 screen_pos_full = screen_pos;
    vec2 screen_pos_packed = gs_unpack_screen_xy(gs_pack_screen_xy(screen_pos));
    vec2 screen_pos_error = abs(screen_pos_packed - screen_pos_full);
    screen_pos = screen_pos_packed;

    float max_sigma = MAX_SIGMA;
#ifdef GS_TIGHTER_BOUNDS
    // Tighten bounds by reducing the max sigma used for binning.
    max_sigma = MAX_SIGMA * 0.85;
#endif
    EigenInfo eigen = compute_eigen(cov2d);

    // Check for skip signal from compute_eigen (astronomically large covariance)
    if (eigen.radius0 <= 0.0 || eigen.radius1 <= 0.0) {
        GS_DEBUG_INCREMENT(radius_reject);
        return;
    }
    bool rebuild_conic = eigen.clamped;

    // compute_eigen already caps eigenvalues. ALWAYS rebuild conic from the capped values
    // to ensure the rasterizer gets proper gaussian falloff for large splats.
    float r0 = eigen.radius0;
    float r1 = eigen.radius1;
    float big = max(r0, r1);
    float small = max(min(r0, r1), 1e-6);
    float aspect = big / small;

    // Track PRE-CLAMP max aspect ratio (before any clamping) - this is the raw distribution
    uint aspect_preclamp_q8 = uint(clamp(aspect * 256.0, 0.0, 65535.0));
    GS_DEBUG_MAX(max_aspect_preclamp_q8, aspect_preclamp_q8);

    // Diagnostic: track high aspect ratios before clamping
    if (aspect > 5.0) {
        GS_DEBUG_INCREMENT(high_aspect_ratio_count);
    }

    // Use configurable max aspect ratio from RenderParams.
    float max_aspect = max(params.max_conic_aspect, 1.0);  // Ensure at least 1:1
    float aspect_fade = 1.0;
    if (aspect > max_aspect) {
        // Quadratic falloff for more aggressive fade on extreme aspect ratios
        float ratio = max_aspect / aspect;
        aspect_fade = ratio * ratio;
        GS_DEBUG_INCREMENT(extreme_conic_count);
        float clamped_small = big / max_aspect;
        if (r0 < r1) {
            r0 = clamped_small;
        } else {
            r1 = clamped_small;
        }
        eigen.radius0 = r0;
        eigen.radius1 = r1;
        rebuild_conic = true;
    }

    // Track POST-CLAMP max aspect ratio (after max_aspect clamp)
    float aspect_postclamp = max(r0, r1) / max(min(r0, r1), 1e-6);
    uint aspect_postclamp_q8 = uint(clamp(aspect_postclamp * 256.0, 0.0, 65535.0));
    GS_DEBUG_MAX(max_aspect_q8, aspect_postclamp_q8);

    if (rebuild_conic) {
        // Rebuild cov2d and conic when eigenvalues are clamped to keep rasterizer inputs consistent.
        mat2 cov2d_rebuilt = mat2(
                r0 * r0 * eigen.axis0.x * eigen.axis0.x + r1 * r1 * eigen.axis1.x * eigen.axis1.x,
                r0 * r0 * eigen.axis0.x * eigen.axis0.y + r1 * r1 * eigen.axis1.x * eigen.axis1.y,
                r0 * r0 * eigen.axis0.y * eigen.axis0.x + r1 * r1 * eigen.axis1.y * eigen.axis1.x,
                r0 * r0 * eigen.axis0.y * eigen.axis0.y + r1 * r1 * eigen.axis1.y * eigen.axis1.y);
        cov2d = cov2d_rebuilt;
        float det_rebuilt = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
        det_rebuilt = max(det_rebuilt, 1e-6);
        float inv_det_rebuilt = 1.0 / det_rebuilt;
        conic = vec3(cov2d[1][1] * inv_det_rebuilt, -cov2d[0][1] * inv_det_rebuilt, cov2d[0][0] * inv_det_rebuilt);
    }

    // ========================================================================
    // Opacity-Aware Bounding (FlashGS Optimization)
    // ========================================================================
    // When enabled, we compute an effective sigma based on the opacity and
    // visibility threshold. Low-opacity splats get smaller bounds, reducing
    // the number of tile-Gaussian pairs by up to 94%.
    //
    // The effective sigma is: sqrt(2 * ln(alpha/tau))
    // where alpha is the splat opacity and tau is the visibility threshold.
    // ========================================================================
    float effective_sigma = max_sigma; // Default to conservative bounds

    if (gs_is_opacity_aware_culling_enabled()) {
        float visibility_threshold = gs_get_visibility_threshold();
        // Use the base opacity (before any fades) for radius calculation
        // This ensures consistent culling regardless of size/aspect fades
        float base_opacity_for_culling = clamp(g.opacity * params.opacity_multiplier, 0.0, 1.0);

        // Compute opacity-aware sigma (number of standard deviations to extend)
        effective_sigma = compute_opacity_aware_sigma(base_opacity_for_culling, visibility_threshold, max_sigma);

        // If the splat is effectively invisible (opacity <= threshold), skip it entirely
        if (effective_sigma <= 0.0) {
            GS_DEBUG_INCREMENT(radius_reject);
            return;
        }
    }

    vec2 axis_extent0 = eigen.axis0 * eigen.radius0 * effective_sigma;
    vec2 axis_extent1 = eigen.axis1 * eigen.radius1 * effective_sigma;

    float max_radius = max(eigen.radius0, eigen.radius1) * effective_sigma;
    float min_radius = min(eigen.radius0, eigen.radius1) * effective_sigma;

    // Use RAW min radius (pre-aspect-clamp) for subpixel culling (#797)
    // The aspect-clamped min_radius is artificially inflated (e.g., 1.8px minimum),
    // preventing effective culling of truly subpixel splats at distance.
    // Use RAW minor radius from projection (computed BEFORE low-pass filter) for subpixel culling (#797)
    // The low-pass floor inflates projected radii for anti-aliasing, so raw radius is the stable cull signal.
    // raw_min_radius_proj is the true screen-space minor radius before any anti-aliasing.
    float raw_min_radius_px = raw_min_radius_proj * effective_sigma;

    const float MIN_SPLAT_RADIUS = 0.1;
    float min_allowed_radius = max(MIN_SPLAT_RADIUS, params.tiny_splat_screen_radius);

    // Debug: capture subpixel culling inputs (Q8.8 fixed-point)
    uint tiny_param_q8 = uint(clamp(params.tiny_splat_screen_radius * 256.0, 0.0, 4294967295.0));
    uint min_allowed_q8 = uint(clamp(min_allowed_radius * 256.0, 0.0, 4294967295.0));
    uint raw_min_q8 = uint(clamp(raw_min_radius_px * 256.0, 0.0, 4294967295.0));
    GS_DEBUG_MAX(tiny_splat_param_q8, tiny_param_q8);
    GS_DEBUG_MAX(min_allowed_radius_q8, min_allowed_q8);
    GS_DEBUG_MAX(min_radius_min_q8_inv, 0xFFFFFFFFu - raw_min_q8);

    // Hysteresis for subpixel culling to prevent flicker at the boundary.
    // Visible splats stay visible until they drop below a lower threshold.
    // Wide gap (1/3 ratio) handles view-dependent radius oscillation during camera rotation.
    float enter_threshold = min_allowed_radius;
    float exit_threshold = max(MIN_SPLAT_RADIUS, min_allowed_radius * (1.0 / 3.0));
    uint history_size = max(params.total_gaussians, 1u);
    uint history_idx = gaussian_idx;
    uint history_key = gs_hash_u32(gaussian_idx) & 0x7FFFFFFFu;
    if (splat_ref.instance_id != 0u) {
        history_idx = gs_hash_u32(gaussian_idx ^ (splat_ref.instance_id * 0x9e3779b9u)) % history_size;
        history_key = gs_hash_u32(gaussian_idx ^ (splat_ref.instance_id * 0x85ebca6bu)) & 0x7FFFFFFFu;
    }
    uint history_packed = subpixel_history.entries[history_idx];
    bool was_visible = ((history_packed & 1u) != 0u) && ((history_packed >> 1u) == history_key);
    float threshold = was_visible ? exit_threshold : enter_threshold;

    bool is_visible;
#ifdef GS_TILE_GLOBAL_SORT_COUNT_PASS
    // COUNT computes visibility once and writes it for EMIT to consume.
    is_visible = raw_min_radius_px >= threshold;
    subpixel_visibility.visible[global_idx] = is_visible ? 1u : 0u;
#elif defined(GS_TILE_GLOBAL_SORT_EMIT_PASS)
    // EMIT consumes COUNT's decision to avoid COUNT/EMIT divergence.
    is_visible = subpixel_visibility.visible[global_idx] != 0u;
#else
    // Non-global sort path: compute visibility directly.
    is_visible = raw_min_radius_px >= threshold;
#endif
    // Only EMIT pass writes to history buffer to prevent COUNT/EMIT visibility mismatch.
    // If both passes write, EMIT sees COUNT's writes and may make different decisions.
#ifdef GS_TILE_GLOBAL_SORT_EMIT_PASS
    subpixel_history.entries[history_idx] = (history_key << 1u) | (is_visible ? 1u : 0u);
#endif

    // Reject truly subpixel splats using pre-low-pass-filter minor radius (#797).
    if (!is_visible) {
        GS_DEBUG_INCREMENT(radius_reject);
        return;
    }

    // Soft fade for very large splats: reduces blocky close-up artifacts without dropping tiles entirely.
    // Use gentle fade (to 70%) starting at much larger radii to avoid making objects see-through.
    const float FADE_START_RADIUS = float(TILE_SIZE) * 16.0;  // 128px - start fade for very large splats
    const float FADE_END_RADIUS = float(TILE_SIZE) * 32.0;    // 256px - fully faded
    float fade_t = smoothstep(FADE_START_RADIUS, FADE_END_RADIUS, max_radius);
    float size_fade = mix(1.0, 0.7, fade_t);  // Gentle fade to 70% opacity, not 15%

    // Clamp extreme radii, but never below what's needed to cover the viewport.
    // This avoids screen-edge cutoffs when the camera is inside/near large splats.
    vec2 viewport_size = max(params.viewport_size, vec2(1.0));
    vec2 far_edge = max(screen_pos, viewport_size - screen_pos);
    float viewport_cover_radius = length(far_edge);
    float max_allowed_radius = max(float(TILE_SIZE) * 20.0, viewport_cover_radius);
    if (max_radius > max_allowed_radius && max_radius > 0.0) {
        float clamp_scale = max_allowed_radius / max_radius;
        // Scale radii to keep conic consistent with the clamped bbox.
        eigen.radius0 *= clamp_scale;
        eigen.radius1 *= clamp_scale;
        r0 = eigen.radius0;
        r1 = eigen.radius1;
        max_radius = max_allowed_radius;
        min_radius *= clamp_scale;
        axis_extent0 *= clamp_scale;
        axis_extent1 *= clamp_scale;

        // Rebuild cov2d and conic after clamping radii.
        cov2d = mat2(
                r0 * r0 * eigen.axis0.x * eigen.axis0.x + r1 * r1 * eigen.axis1.x * eigen.axis1.x,
                r0 * r0 * eigen.axis0.x * eigen.axis0.y + r1 * r1 * eigen.axis1.x * eigen.axis1.y,
                r0 * r0 * eigen.axis0.y * eigen.axis0.x + r1 * r1 * eigen.axis1.y * eigen.axis1.x,
                r0 * r0 * eigen.axis0.y * eigen.axis0.y + r1 * r1 * eigen.axis1.y * eigen.axis1.y);
        float det_rebuilt2 = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1];
        det_rebuilt2 = max(det_rebuilt2, 1e-6);
        float inv_det_rebuilt2 = 1.0 / det_rebuilt2;
        conic = vec3(cov2d[1][1] * inv_det_rebuilt2, -cov2d[0][1] * inv_det_rebuilt2, cov2d[0][0] * inv_det_rebuilt2);
    }

    // Suppress near-camera giant splats that create "lens dirt/fireflies" overlays.
    // Keep this soft (opacity fade) so close-up transitions remain stable.
    float viewport_diag = length(viewport_size);
    float close_radius_start = max(float(TILE_SIZE) * 8.0, viewport_diag * 0.22);
    float close_radius_end = max(close_radius_start + float(TILE_SIZE) * 4.0, viewport_diag * 0.55);
    float radius_pressure = smoothstep(close_radius_start, close_radius_end, max_radius);
    float near_pressure = 1.0 - smoothstep(0.006, 0.03, linear_depth);
    float lens_fade = 1.0 - (radius_pressure * near_pressure);
    if (lens_fade <= 0.02) {
        GS_DEBUG_INCREMENT(radius_reject);
        return;
    }

    vec2 ellipse_half_extent = vec2(abs(axis_extent0.x) + abs(axis_extent1.x),
            abs(axis_extent0.y) + abs(axis_extent1.y));


    vec2 bbox_min = screen_pos - ellipse_half_extent;
    vec2 bbox_max = screen_pos + ellipse_half_extent;

    // Enhanced viewport bounds checking with margin for precision
    // Skip hard reject to avoid false positives that create holes; clamp instead.

    // Validate bounding box integrity
    if (bbox_min.x >= bbox_max.x || bbox_min.y >= bbox_max.y ||
        any(isnan(bbox_min)) || any(isnan(bbox_max)) ||
        any(isinf(bbox_min)) || any(isinf(bbox_max))) {
        GS_DEBUG_INCREMENT(bbox_integrity_reject);
        return; // Invalid bounding box
    }

    float inv_tile = 1.0 / float(TILE_SIZE);
    // Pad bounds before tile quantization to avoid edge cutoffs.
    // For close-up splats with large footprints, we need more padding because:
    // 1. Half-float quantization errors grow with coordinate magnitude
    // 2. Large splats have more numerical error in their bbox computation
    // 3. Edge tiles are sensitive to sub-pixel rounding
    // Base padding of half a tile, plus a fraction of the splat's footprint.
    float base_pad = float(TILE_SIZE) * 0.5;
    float footprint_pad = max(ellipse_half_extent.x, ellipse_half_extent.y) * 0.02;
    float quant_pad = max(screen_pos_error.x, screen_pos_error.y);
    float tile_pad = max(base_pad, max(footprint_pad, quant_pad));
    vec2 padded_min = bbox_min - vec2(tile_pad);
    vec2 padded_max = bbox_max + vec2(tile_pad);
    int min_tile_x = int(floor(padded_min.x * inv_tile));
    int max_tile_x = int(ceil(padded_max.x * inv_tile));
    int min_tile_y = int(floor(padded_min.y * inv_tile));
    int max_tile_y = int(ceil(padded_max.y * inv_tile));

    ivec2 tile_extent = max(ivec2(params.tile_count) - ivec2(1), ivec2(0));
    min_tile_x = clamp(min_tile_x, 0, tile_extent.x);
    max_tile_x = clamp(max_tile_x, 0, tile_extent.x);
    min_tile_y = clamp(min_tile_y, 0, tile_extent.y);
    max_tile_y = clamp(max_tile_y, 0, tile_extent.y);

    if (min_tile_x > max_tile_x || min_tile_y > max_tile_y) {
        GS_DEBUG_INCREMENT(tile_extent_reject);
        return;
    }

    if (debug_audit.enabled != 0u && debug_audit.sample_count > 0u) {
        uint audit_count = min(debug_audit.sample_count, GS_DEBUG_SPLAT_AUDIT_MAX_SAMPLES);
        for (uint a = 0u; a < audit_count; ++a) {
            if (debug_audit.entries[a].global_idx == global_idx) {
                uint flags = GS_AUDIT_FLAG_PROJECTED;
                vec2 audit_viewport = max(params.viewport_size, vec2(1.0));
                if (screen_pos.x >= 0.0 && screen_pos.y >= 0.0 &&
                        screen_pos.x < audit_viewport.x && screen_pos.y < audit_viewport.y) {
                    flags |= GS_AUDIT_FLAG_IN_VIEWPORT;
                    vec2 clamped_px = clamp(floor(screen_pos), vec2(0.0), audit_viewport - vec2(1.0));
                    debug_audit.entries[a].expected_x = uint(clamped_px.x);
                    debug_audit.entries[a].expected_y = uint(clamped_px.y);
                }
                atomicOr(debug_audit.entries[a].flags, flags);
                break;
            }
        }
    }

    // Validate conic BEFORE the COUNT/EMIT split to prevent divergence.
    // Previously this check only existed in the EMIT section (after COUNT returned),
    // causing COUNT to record tile overlaps for NaN-conic gaussians that EMIT would
    // skip.  The mismatch left stale records in the overlap buffer that the radix
    // sort redistributed to random tiles, producing visible tile corruption.
    if (any(isnan(conic)) || any(isinf(conic))) {
        GS_DEBUG_INCREMENT(covariance_nan_reject);
        return;
    }

    uint tiles_covered_x = uint(max_tile_x - min_tile_x + 1);
    uint tiles_covered_y = uint(max_tile_y - min_tile_y + 1);

#ifdef GS_TILE_GLOBAL_SORT_COUNT_PASS
    // Pass 1: count overlaps per tile (no keys/values emitted).
    for (int ty = min_tile_y; ty <= max_tile_y; ++ty) {
        for (int tx = min_tile_x; tx <= max_tile_x; ++tx) {
            uint tile_idx = uint(ty) * uint(params.tile_count.x) + uint(tx);
            if (!gs_keep_overlap_record(gaussian_idx, splat_ref.instance_id, tile_idx)) {
                continue;
            }
            atomicAdd(tile_counts.counts[tile_idx], 1u);
        }
    }
    return;
#endif

    // Use the fade computed earlier for opacity; we already skipped if nearly invisible.
    float base_opacity = clamp(g.opacity * params.opacity_multiplier * size_fade * aspect_fade * lens_fade, 0.0, 0.99);

    // Evaluate SH for view-dependent color using configurable band level
    vec3 view_dir = normalize(params.camera_position.xyz - g.position);
    vec3 view_dir_local = (instance.ids.y & GS_INSTANCE_FLAG_ROTATION_IDENTITY) != 0u
            ? view_dir
            : gs_quat_rotate(instance.inv_rotation, view_dir);
    uint sh_band_level = gs_get_sh_band_level();
#if defined(GS_SH_AMORTIZATION) && defined(GS_TILE_GLOBAL_SORT_EMIT_PASS)
    uint sh_divisor = gs_get_sh_amortization_divisor();
    uint sh_phase = gs_get_sh_amortization_phase();
    bool sh_force_update = gs_get_sh_amortization_force_update();
    vec3 sh_color = vec3(0.0);
    bool use_cache = sh_divisor > 1u;
    bool update_sh = !use_cache || sh_force_update;
    uint cached_color = 0u;
    if (use_cache && !sh_force_update) {
        cached_color = sh_color_cache.colors[gaussian_idx];
        bool phase_match = ((gaussian_idx + sh_phase) % sh_divisor) == 0u;
        update_sh = phase_match || cached_color == 0u;
        if (!update_sh) {
            sh_color = gs_unpack_color_r11g11b10(cached_color);
            GS_DEBUG_INCREMENT(sh_cache_hits);
        }
    }
    if (update_sh) {
        sh_color = evaluate_sh_with_bands(g, view_dir_local, sh_band_level);
        sh_color = max(sh_color, vec3(0.0));
        // Cache ungraded SH color so color grading changes immediately affect all splats
        if (use_cache) {
            sh_color_cache.colors[gaussian_idx] = gs_pack_color_r11g11b10(sh_color);
            GS_DEBUG_INCREMENT(sh_cache_updates);
            if (sh_force_update) {
                GS_DEBUG_INCREMENT(sh_cache_forced_updates);
            }
        }
    }
    // Apply color grading after cache logic so it affects both cached and fresh SH
    sh_color = apply_color_grading_binning(sh_color);
#else
    vec3 sh_color = evaluate_sh_with_bands(g, view_dir_local, sh_band_level);
    // Clamp SH color to non-negative after evaluation
    // SH basis functions can produce negative contributions but final color should not be negative
    sh_color = max(sh_color, vec3(0.0));
    sh_color = apply_color_grading_binning(sh_color);
#endif

    // Debug: force white albedo to isolate lighting contribution (debug_overlay_flags.w).
    if (params.debug_overlay_flags.w > 0.5) {
        sh_color = vec3(0.7);
    }
    vec3 sh_albedo = clamp(sh_color, vec3(0.0), vec3(1.0));

    vec3 view_pos_scene = (scene_data_block.data.view_matrix * vec4(g.position, 1.0)).xyz;
    vec3 view_dir_scene = normalize(-view_pos_scene);
    uint normal_mode = uint(params.lighting_config.w + 0.5);
    vec3 normal_view = view_dir_scene;
    if (normal_mode != 1u) {
        if (dot(g.normal, g.normal) > 1e-6) {
            vec3 normal_world = (instance_flags & GS_INSTANCE_FLAG_ROTATION_IDENTITY) != 0u
                    ? g.normal
                    : gs_quat_rotate(world_rotation, g.normal);
            normal_view = normalize(mat3(scene_data_block.data.view_matrix) * normal_world);
        } else {
            // Fallback: derive a stable normal from the thinnest Gaussian axis
            // to avoid view-dependent lighting when normals are missing.
            vec3 scale_abs = abs(g.scale);
            vec3 axis = vec3(0.0, 0.0, 1.0);
            if (scale_abs.x <= scale_abs.y && scale_abs.x <= scale_abs.z) {
                axis = vec3(1.0, 0.0, 0.0);
            } else if (scale_abs.y <= scale_abs.z) {
                axis = vec3(0.0, 1.0, 0.0);
            }
            vec3 normal_world = gs_quat_rotate(world_rotation, axis);
            normal_view = normalize(mat3(scene_data_block.data.view_matrix) * normal_world);
        }
    }

    // Per-splat receiver bias to reduce self-shadowing (scaled by splat radius).
    float splat_radius = max(g.scale.x, max(g.scale.y, g.scale.z));
    float receiver_bias = max(params.shadow_bias_config.y, params.shadow_bias_config.x * splat_radius);
    if (params.shadow_bias_config.z > 0.0) {
        receiver_bias = min(receiver_bias, params.shadow_bias_config.z);
    }

    // Per-splat direct lighting (Option A): add real-time lights before rasterization.
    vec3 final_color = sh_color * params.lighting_config.y;
    float shadow_strength = clamp(params.shadow_strength.x, 0.0, 1.0);
    bool shadow_sampling_enabled = shadow_strength > 0.0;
    float sh_occlusion = 0.0;
    if (params.lighting_config.z > 0.5) {
        uint lighting_mode = params.lighting_mode.x;
        if (lighting_mode == 1u || lighting_mode == 2u) {
            vec3 view_pos = view_pos_scene;
            vec3 view_dir = view_dir_scene;

            vec2 uv = screen_pos / max(params.viewport_size, vec2(1.0));

            hvec3 h_normal_base = hvec3(normal_view);
            hvec3 h_view = hvec3(view_dir);
            // Use neutral grey albedo for lighting by default. If indirect SH is disabled,
            // fall back to SH color as albedo so splat colors still show up.
            hvec3 h_albedo = hvec3(0.5);
            if (params.lighting_config.y <= 0.001) {
                h_albedo = hvec3(sh_albedo);
            }
            half roughness = half(1.0);
            half metallic = half(0.0);
            half specular = half(0.5);
            hvec3 f0 = F0(metallic, specular, h_albedo);
            hvec3 diffuse_light = hvec3(0.0);
            hvec3 specular_light = hvec3(0.0);
            half alpha = half(base_opacity);
            hvec3 energy_compensation = hvec3(1.0);

            uint light_mask = params.light_counts.w;
            gs_accumulate_directional_lights(view_pos_scene, normal_view, receiver_bias, shadow_sampling_enabled,
                    light_mask, h_normal_base, h_view, h_albedo, roughness, metallic, f0, alpha, uv,
                    energy_compensation, diffuse_light, specular_light, sh_occlusion);

            bool use_clustered = gs_use_clustered_lights();
            bool shadow_modulate_sh = shadow_strength > 0.0;
            if (use_clustered) {
                uint cluster_offset;
                uint cluster_z;
                uint cluster_type_size;
                uint max_cluster_element_count_div_32;
                gs_get_cluster_params(screen_pos, view_pos, cluster_offset, cluster_z, cluster_type_size,
                        max_cluster_element_count_div_32);
                if (shadow_modulate_sh) {
                    gs_accumulate_clustered_omni_spot_sh_occlusion(cluster_offset, cluster_z, cluster_type_size,
                            max_cluster_element_count_div_32, view_pos, normal_view, shadow_sampling_enabled, light_mask,
                            sh_occlusion);
                }
                gs_accumulate_clustered_omni_spot_direct(cluster_offset, cluster_z, cluster_type_size,
                        max_cluster_element_count_div_32, view_pos, h_normal_base, h_view, h_albedo,
                        roughness, metallic, f0, alpha, uv, energy_compensation, light_mask,
                        diffuse_light, specular_light);
            } else {
                if (shadow_modulate_sh) {
                    gs_accumulate_unclustered_omni_spot_sh_occlusion(view_pos, normal_view, shadow_sampling_enabled,
                            light_mask, sh_occlusion);
                }
                gs_accumulate_unclustered_omni_spot_direct(view_pos, h_normal_base, h_view, h_albedo,
                        roughness, metallic, f0, alpha, uv, energy_compensation, light_mask,
                        diffuse_light, specular_light);
            }

            if (shadow_strength > 0.0 && sh_occlusion > 0.0) {
                float sh_factor = 1.0 - shadow_strength * clamp(sh_occlusion, 0.0, 1.0);
                final_color *= sh_factor;
            }

            // Match Godot's forward path: diffuse light is multiplied by albedo at the end.
            diffuse_light *= h_albedo;
            diffuse_light *= (half(1.0) - metallic);
            vec3 direct = vec3(diffuse_light + specular_light) * params.lighting_config.x;
            final_color += direct;
        }
    }

    sh_color = max(final_color, vec3(0.0));

    // PERF-8 (#679): Redundant safety net — the primary NaN/Inf conic check now lives
    // before the COUNT/EMIT split to prevent divergence.  Kept as defence-in-depth.
    if (any(isnan(conic)) || any(isinf(conic))) {
        GS_DEBUG_INCREMENT(covariance_nan_reject);
        return;
    }

    // Pack ProjectedGaussian payload using the active layout (packed or full).
    ProjectedGaussian payload;
    gs_pack_projected_gaussian(payload, screen_pos, linear_depth, base_opacity, sh_color, normal_view, conic, global_idx);

#ifdef GS_TILE_GLOBAL_SORT_EMIT_PASS
    projection_buffer.projected_gaussians[global_idx] = payload;

    // Pass 2: emit one overlap record per covered tile into the global key/value arrays.
    for (int ty = min_tile_y; ty <= max_tile_y; ++ty) {
        for (int tx = min_tile_x; tx <= max_tile_x; ++tx) {
            uint tile_idx = uint(ty) * uint(params.tile_count.x) + uint(tx);
            if (!gs_keep_overlap_record(gaussian_idx, splat_ref.instance_id, tile_idx)) {
                continue;
            }
            uvec2 range = tile_ranges.ranges[tile_idx];
            uint local_offset = atomicAdd(tile_counts.counts[tile_idx], 1u);
            if (local_offset >= range.y) {
                // Undo the cursor increment so tile_counts reflects emitted (clamped) records.
                atomicAdd(tile_counts.counts[tile_idx], 0xFFFFFFFFu); // -1
                atomicAdd(overflow_stats.overflow_splats_clamped, 1u);
                continue;
            }
            uint record_idx = range.x + local_offset;
            uint overlap_limit = indirect_dispatch.element_count;
            if (record_idx >= overlap_limit) {
                // Global overlap budget exhausted. Do not write past the allocated key/value range.
                atomicAdd(overflow_stats.overflow_splats_clamped, 1u);
                continue;
            }
            global_sort_keys.keys[record_idx] = gs_pack_sort_key(tile_idx, linear_depth);
            global_sort_values.values[record_idx] = global_idx;
        }
    }

    atomicAdd(overflow_stats.overflow_splats_aggregated, tiles_covered_x * tiles_covered_y);
    GS_DEBUG_INCREMENT(success_count);
    return;
#endif
}
