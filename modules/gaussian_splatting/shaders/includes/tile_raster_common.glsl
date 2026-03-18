#ifndef GS_TILE_RASTER_COMMON_GLSL
#define GS_TILE_RASTER_COMMON_GLSL

#include "gs_instance_layout.glsl"

const uint GS_DEBUG_SPLAT_AUDIT_MAX_SAMPLES = 64u;
const uint GS_DEBUG_SPLAT_AUDIT_MATCHES_MAX = 4u;
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

layout(set = 0, binding = 15, std430) readonly buffer InstanceIndirectDispatch {
    uint dispatch_xyz[3];
    uint element_count;
    uint overflow_flag;
    uint unclamped_total;
} instance_indirect;

// Dithering constants for 8-bit output quantization artifact mitigation
// For R8G8B8A8_UNORM (8-bit per channel), 1 LSB = 1/255
// Use slightly more than 1 LSB for effective banding reduction
const float GS_DITHER_AMPLITUDE = 1.0 / 255.0;

// Simple hash-based noise for dithering (spatially varying)
// Uses fragment position to generate pseudo-random value in [-0.5, 0.5]
float gs_dither_noise(vec2 frag_coord) {
    // Simple but effective hash function for dithering
    vec2 p = frag_coord * 0.06711056 + 0.00583715;
    return fract(52.9829189 * fract(dot(p, vec2(12.9898, 78.233)))) - 0.5;
}

// Generate RGB dither noise using different offsets for each channel
// This breaks up color banding from quantization
vec3 gs_dither_noise_rgb(vec2 frag_coord) {
    return vec3(
        gs_dither_noise(frag_coord),
        gs_dither_noise(frag_coord + vec2(17.0, 23.0)),
        gs_dither_noise(frag_coord + vec2(37.0, 41.0))
    );
}

// Apply dithering to a color to reduce quantization banding
// Uses flat dithering (not scaled by color) for consistent banding reduction across all tones
vec3 gs_apply_color_dither(vec3 color, vec2 frag_coord) {
    vec3 dither = gs_dither_noise_rgb(frag_coord) * GS_DITHER_AMPLITUDE;
    // Direct addition - dither noise is in [-0.5, 0.5] * amplitude range
    return color + dither;
}

vec3 gs_spectral_heatmap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.0, 0.0, 0.5);
    vec3 c1 = vec3(0.0, 0.5, 1.0);
    vec3 c2 = vec3(0.0, 1.0, 0.0);
    vec3 c3 = vec3(1.0, 1.0, 0.0);
    vec3 c4 = vec3(1.0, 0.0, 0.0);
    float band = t * 4.0;
    if (band < 1.0) {
        return mix(c0, c1, band);
    }
    if (band < 2.0) {
        return mix(c1, c2, band - 1.0);
    }
    if (band < 3.0) {
        return mix(c2, c3, band - 2.0);
    }
    return mix(c3, c4, band - 3.0);
}

vec3 gs_apply_tile_grid(vec2 frag_coord, vec3 color, float opacity) {
    vec2 tile_uv = fract(frag_coord / float(TILE_SIZE));
    float edge = 1.0 / float(TILE_SIZE);
    bool on_edge = (tile_uv.x < edge) || (tile_uv.x > (1.0 - edge)) ||
            (tile_uv.y < edge) || (tile_uv.y > (1.0 - edge));
    if (!on_edge) {
        return color;
    }
    float mix_factor = clamp(opacity, 0.0, 1.0);
    return mix(color, color * 0.5, mix_factor);
}

uint gs_read_sorted_value(uint local_index, uint range_start) {
#ifdef GS_TILE_RASTER_USE_SHARED
    if (local_index < gs_shared_splat_count) {
        return gs_shared_sorted_values[local_index];
    }
#endif
    return sorted_values.values[range_start + local_index];
}

ProjectedGaussian gs_read_projected_gaussian(uint local_index, uint sorted_idx) {
#ifdef GS_TILE_RASTER_SHARED_PAYLOAD
    if (local_index < gs_shared_splat_count) {
        return gs_shared_projected_gaussians[local_index];
    }
#endif
    return projection_buffer.projected_gaussians[sorted_idx];
}

uint gs_get_visible_gaussian_count() {
    return instance_indirect.element_count;
}

// ============================================================================
// Multi-pass batch rasterization (compute shader only)
// ============================================================================
// Only compiled for compute shaders that define GS_TILE_RASTER_USE_SHARED,
// since this function reads from shared memory arrays (gs_shared_sorted_values,
// gs_shared_projected_gaussians) which are not available in fragment shaders.
#if GS_TILE_RASTER_USE_SHARED

// Processes a batch of splats that are ALL in shared memory (indices 0..batch_size-1).
// Per-pixel state (final_color, final_depth, final_normal, has_depth) is carried
// across batches by the caller.  Returns true if the pixel is alpha-saturated.
//
// This function contains the core blending loop extracted from gs_rasterize_pixel.
// It omits debug audit tracking and raster stats for simplicity; those are only
// available in the single-pass path.
bool gs_rasterize_splat_batch(
        vec2 pixel_center,
        uint batch_size,
        float lod_blend,
        vec3 pixel_dither,
        float highlight_strength,
        float outline_width,
        uint interactive_render_state,
        inout vec4 final_color,
        inout float final_depth,
        inout vec3 final_normal,
        inout bool has_depth) {
    for (uint i = 0u; i < batch_size; ++i) {
        uint sorted_idx = gs_shared_sorted_values[i];
        if (sorted_idx >= gs_get_visible_gaussian_count()) {
            continue;
        }

        uint splat_ref_idx = sorted_indices.indices[sorted_idx];
        if (splat_ref_idx >= gs_get_visible_gaussian_count()) {
            continue;
        }
        uint gaussian_idx = splat_ref_buffer.splat_refs[splat_ref_idx].atlas_index;
        if (gaussian_idx >= params.total_gaussians) {
            continue;
        }

#ifdef GS_TILE_RASTER_SHARED_PAYLOAD
        ProjectedGaussian payload = gs_shared_projected_gaussians[i];
#else
        ProjectedGaussian payload = projection_buffer.projected_gaussians[sorted_idx];
#endif

        vec2 screen_px;
        float linear_depth;
        float base_opacity;
        vec3 unpacked_color;
        vec3 unpacked_normal;
        vec3 conic;
        uint stored_global_idx;
        gs_unpack_projected_gaussian(payload, screen_px, linear_depth, base_opacity, unpacked_color, unpacked_normal, conic, stored_global_idx);

        if (stored_global_idx != sorted_idx) {
            continue;
        }
        if (base_opacity <= 0.0) {
            continue;
        }

        vec2 diff = pixel_center - screen_px;
        float quadratic = conic.x * diff.x * diff.x + 2.0 * conic.y * diff.x * diff.y + conic.z * diff.y * diff.y;
        quadratic = max(quadratic, 0.0);
        float weight = gs_exp_fast(-0.5 * quadratic);
        if (weight <= 0.0) {
            continue;
        }

        float alpha = clamp(base_opacity * weight, 0.0, 0.99);
        alpha *= lod_blend;
        if (alpha <= 1e-4) {
            continue;
        }

        vec3 base_color = max(unpacked_color + pixel_dither, vec3(0.0));
        if (interactive_render_state == 3u) {
            alpha *= 0.5;
            base_color *= 0.4;
        }
        if (highlight_strength > 0.0) {
            base_color = mix(base_color, interactive_state.highlight_color.rgb,
                    clamp(highlight_strength, 0.0, 1.0));
        }
        if (outline_width > 0.0) {
            float outline_factor = clamp((1.0 - weight) * outline_width, 0.0, 1.0);
            base_color = mix(base_color, interactive_state.outline_color.rgb, outline_factor);
        }

        float remaining_alpha = 1.0 - final_color.a;
        if (remaining_alpha <= 1e-3) {
            return true;
        }

        float blend_alpha = alpha * remaining_alpha;
        if (blend_alpha <= 0.0) {
            continue;
        }

        final_color.rgb += base_color * blend_alpha;
        final_color.a = clamp(final_color.a + blend_alpha, 0.0, 1.0);
        final_normal += unpacked_normal * blend_alpha;
        final_depth = has_depth ? min(final_depth, linear_depth) : linear_depth;
        has_depth = true;

        if (final_color.a >= 0.995) {
            return true;
        }

#if GS_SUBGROUP_VOTE_AVAILABLE
        bool pixel_near_saturated = (final_color.a >= 0.99);
        bool all_saturated;
        GS_SUBGROUP_EARLY_EXIT_IF_ALL_SATURATED(pixel_near_saturated, all_saturated);
        if (all_saturated) {
            return true;
        }
#endif
    }
    return false;
}

#endif // GS_TILE_RASTER_USE_SHARED

void gs_rasterize_pixel(vec2 frag_coord, uint range_start, uint splat_count, uint original_splat_count,
        out vec4 out_color, out float out_depth, out vec4 out_normal) {
    vec2 pixel_pos = frag_coord - vec2(0.5);
    vec2 viewport_size = max(params.viewport_size, vec2(1.0));

    bool debug_tile_grid = params.debug_flags.x > 0.5;
    bool debug_tiles = params.debug_flags.z > 0.5;
    bool debug_projection = params.debug_flags.w > 0.5;
    bool debug_density_heatmap = params.debug_overlay_flags.x > 0.5;
    bool debug_depth_visualization = params.debug_overlay_flags.y > 0.5;

    // Keep the tile debug overlay usable even if projection debug is also enabled.
    if (!debug_tiles && debug_projection) {
        vec2 normalized = frag_coord / viewport_size;
        vec3 diag_color = vec3(normalized, 0.25 + 0.5 * normalized.y);
        vec2 tile_uv = fract(frag_coord / float(TILE_SIZE));
        bool on_edge = (tile_uv.x < 0.02) || (tile_uv.x > 0.98) || (tile_uv.y < 0.02) || (tile_uv.y > 0.98);
        float edge = on_edge ? 1.0 : 0.0;
        diag_color = mix(diag_color, vec3(1.0, 0.35, 0.15), edge);
        out_color = vec4(diag_color, 1.0);
        out_depth = 0.0;
        out_normal = vec4(0.0);
        return;
    }

#ifdef GS_COLLECT_RASTER_STATS
    bool sample_raster_stats = false;
    {
        bool collect_raster_stats = params.debug_flags.y > 0.5;
        if (collect_raster_stats) {
            ivec2 pixel_i = ivec2(frag_coord);
            int local_x = pixel_i.x % int(TILE_SIZE);
            int local_y = pixel_i.y % int(TILE_SIZE);
            sample_raster_stats = (local_x == (int(TILE_SIZE) >> 1)) && (local_y == (int(TILE_SIZE) >> 1));
        }
    }

    uint local_iterated = 0u;
    uint local_contributed = 0u;
    uint local_reject_sorted = 0u;
    uint local_reject_gaussian = 0u;
    uint local_reject_base_opacity = 0u;
    uint local_reject_nan_inf = 0u;
    uint local_reject_weight = 0u;
    uint local_reject_alpha = 0u;
    bool local_break_remaining = false;
    bool local_break_final_alpha = false;
    bool local_break_subgroup_early_exit = false;
#endif

    vec4 final_color = vec4(0.0);
    vec3 final_normal = vec3(0.0);
    float final_depth = 1.0;
    bool has_depth = false;

    float highlight_strength = interactive_state.state_params.x;
    float outline_width = interactive_state.state_params.y;
    uint state = uint(interactive_state.state_params.z + 0.5);

    vec2 pixel_center = pixel_pos + vec2(0.5);
    uint audit_match_count = 0u;
    uint audit_match_indices[GS_DEBUG_SPLAT_AUDIT_MATCHES_MAX];
    uint audit_match_global_indices[GS_DEBUG_SPLAT_AUDIT_MATCHES_MAX];
    if (debug_audit.enabled != 0u && debug_audit.sample_count > 0u) {
        ivec2 audit_pixel = ivec2(floor(pixel_center));
        uint audit_count = min(debug_audit.sample_count, GS_DEBUG_SPLAT_AUDIT_MAX_SAMPLES);
        for (uint a = 0u; a < audit_count && audit_match_count < GS_DEBUG_SPLAT_AUDIT_MATCHES_MAX; ++a) {
            DebugSplatAuditEntry entry = debug_audit.entries[a];
            if ((entry.flags & GS_AUDIT_FLAG_IN_VIEWPORT) == 0u) {
                continue;
            }
            if (entry.expected_x == uint(audit_pixel.x) && entry.expected_y == uint(audit_pixel.y)) {
                audit_match_indices[audit_match_count] = a;
                audit_match_global_indices[audit_match_count] = entry.global_idx;
                audit_match_count++;
            }
        }
    }

    // Optimization: Hoist loop-invariant computations outside splat loop
    // LOD blend factor depends only on uniforms, not per-splat data
    float lod_blend = gs_get_lod_blend_factor();
    // Dither noise depends only on pixel position, compute once per pixel
    vec3 pixel_dither = gs_dither_noise_rgb(pixel_center) * GS_DITHER_AMPLITUDE;

    for (uint i = 0u; i < splat_count; ++i) {
#ifdef GS_COLLECT_RASTER_STATS
        if (sample_raster_stats) {
            local_iterated++;
        }
#endif

        uint sorted_idx = gs_read_sorted_value(i, range_start);
        if (sorted_idx >= gs_get_visible_gaussian_count()) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_reject_sorted++;
            }
#endif
            continue;
        }

        uint splat_ref_idx = sorted_indices.indices[sorted_idx];
        if (splat_ref_idx >= gs_get_visible_gaussian_count()) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_reject_gaussian++;
            }
#endif
            continue;
        }
        uint gaussian_idx = splat_ref_buffer.splat_refs[splat_ref_idx].atlas_index;
        if (gaussian_idx >= params.total_gaussians) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_reject_gaussian++;
            }
#endif
            continue;
        }

        ProjectedGaussian payload = gs_read_projected_gaussian(i, sorted_idx);

        // Unpack the 24-byte packed format
        vec2 screen_px;
        float linear_depth;
        float base_opacity;
        vec3 unpacked_color;
        vec3 unpacked_normal;
        vec3 conic;
        uint stored_global_idx;
        gs_unpack_projected_gaussian(payload, screen_px, linear_depth, base_opacity, unpacked_color, unpacked_normal, conic, stored_global_idx);

        if (stored_global_idx != sorted_idx) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                atomicAdd(overflow_stats.raster_reject_index_mismatch, 1u);
            }
#endif
            continue;
        }

        uint audit_entry_index = 0xFFFFFFFFu;
        if (audit_match_count > 0u) {
            for (uint m = 0u; m < audit_match_count; ++m) {
                if (sorted_idx == audit_match_global_indices[m]) {
                    audit_entry_index = audit_match_indices[m];
                    break;
                }
            }
        }
        bool audit_active = (audit_entry_index != 0xFFFFFFFFu);
        if (audit_active) {
            atomicOr(debug_audit.entries[audit_entry_index].flags, GS_AUDIT_FLAG_ITERATED);
        }

        if (base_opacity <= 0.0) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_reject_base_opacity++;
            }
#endif
            if (audit_active) {
                atomicOr(debug_audit.entries[audit_entry_index].flags, GS_AUDIT_FLAG_ALPHA_SKIPPED);
            }
            continue;
        }

        // PERF-8 (#679): NaN/Inf validation moved to tile_binning.glsl projection stage
        // This eliminates per-pixel * per-splat validation overhead.
        // The projection shader now validates conic before packing, so we only need
        // a lightweight check for debugging builds.
#ifndef GS_DEBUG_COUNTERS_DISABLED
        if (any(isnan(screen_px)) || any(isinf(screen_px)) || any(isnan(conic)) || any(isinf(conic))) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_reject_nan_inf++;
            }
#endif
            continue;
        }
#endif

        vec2 diff = pixel_center - screen_px;
        float dx = diff.x;
        float dy = diff.y;
        float quadratic = conic.x * dx * dx + 2.0 * conic.y * dx * dy + conic.z * dy * dy;
        quadratic = max(quadratic, 0.0);

        float weight = gs_exp_fast(-0.5 * quadratic);
        // PERF-8 (#679): Simplified weight check - with conic validated upstream and
        // quadratic clamped to >= 0, gs_exp_fast cannot produce NaN/Inf.
        // exp(-x) for x >= 0 gives (0, 1], so we only check weight > 0.
        // Note: gs_exp_fast clamps input to [-16, 0] so minimum weight is ~1e-7
        if (weight <= 0.0) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_reject_weight++;
            }
#endif
            continue;
        }

        float alpha = clamp(base_opacity * weight, 0.0, 0.99);

        // Apply LOD blend factor (LODGE technique) - smooth LOD transitions
        // Note: lod_blend is hoisted outside loop since it only depends on uniforms
        alpha *= lod_blend;

        if (alpha <= 1e-4) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_reject_alpha++;
            }
#endif
            if (audit_active) {
                atomicOr(debug_audit.entries[audit_entry_index].flags, GS_AUDIT_FLAG_ALPHA_SKIPPED);
            }
            continue;
        }

        // Apply dithering to break up quantization banding from R11G11B10 color compression
        // Note: pixel_dither is hoisted outside loop since it only depends on pixel position
        vec3 base_color = unpacked_color + pixel_dither;

        // Clamp color to non-negative after dithering (safety check)
        base_color = max(base_color, vec3(0.0));

        if (state == 3u) {
            alpha *= 0.5;
            base_color *= 0.4;
        }

        if (highlight_strength > 0.0) {
            base_color = mix(base_color, interactive_state.highlight_color.rgb,
                    clamp(highlight_strength, 0.0, 1.0));
        }

        if (outline_width > 0.0) {
            float outline_factor = clamp((1.0 - weight) * outline_width, 0.0, 1.0);
            base_color = mix(base_color, interactive_state.outline_color.rgb, outline_factor);
        }

        float remaining_alpha = 1.0 - final_color.a;
        if (remaining_alpha <= 1e-3) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_break_remaining = true;
            }
#endif
            break;
        }

        float blend_alpha = alpha * remaining_alpha;
        if (blend_alpha <= 0.0) {
            continue;
        }

        if (audit_active) {
            atomicOr(debug_audit.entries[audit_entry_index].flags, GS_AUDIT_FLAG_CONTRIBUTED);
        }

#ifdef GS_COLLECT_RASTER_STATS
        if (sample_raster_stats) {
            local_contributed++;
        }
#endif

        final_color.rgb += base_color * blend_alpha;
        final_color.a = clamp(final_color.a + blend_alpha, 0.0, 1.0);
        final_normal += unpacked_normal * blend_alpha;

        final_depth = has_depth ? min(final_depth, linear_depth) : linear_depth;
        has_depth = true;

        if (final_color.a >= 0.995) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_break_final_alpha = true;
            }
#endif
            break;
        }

        // Subgroup early-exit optimization: if ALL pixels in the subgroup are
        // near-saturated (alpha >= 0.99), skip remaining splats for the entire tile.
        // This reduces iterations in dense scenes where tiles become opaque quickly.
        // The threshold (0.99) is slightly lower than the per-pixel break (0.995) to
        // allow the optimization to trigger a bit earlier for collective benefit.
#if GS_SUBGROUP_VOTE_AVAILABLE
        bool pixel_near_saturated = (final_color.a >= 0.99);
        bool all_saturated;
        GS_SUBGROUP_EARLY_EXIT_IF_ALL_SATURATED(pixel_near_saturated, all_saturated);
        if (all_saturated) {
#ifdef GS_COLLECT_RASTER_STATS
            if (sample_raster_stats) {
                local_break_subgroup_early_exit = true;
            }
#endif
            break;
        }
#endif
    }

    float depth_out = has_depth ? final_depth : 1.0;

#ifdef GS_COLLECT_RASTER_STATS
    if (sample_raster_stats) {
        atomicAdd(overflow_stats.raster_sample_count, 1u);
        atomicAdd(overflow_stats.raster_splats_iterated, local_iterated);
        atomicAdd(overflow_stats.raster_splats_contributed, local_contributed);
        atomicAdd(overflow_stats.raster_reject_sorted_idx_oob, local_reject_sorted);
        atomicAdd(overflow_stats.raster_reject_gaussian_idx_oob, local_reject_gaussian);
        atomicAdd(overflow_stats.raster_reject_base_opacity, local_reject_base_opacity);
        atomicAdd(overflow_stats.raster_reject_nan_inf, local_reject_nan_inf);
        atomicAdd(overflow_stats.raster_reject_weight, local_reject_weight);
        atomicAdd(overflow_stats.raster_reject_alpha, local_reject_alpha);
        if (local_break_remaining) {
            atomicAdd(overflow_stats.raster_break_remaining_alpha, 1u);
        }
        if (local_break_final_alpha) {
            atomicAdd(overflow_stats.raster_break_final_alpha, 1u);
        }
        if (local_break_subgroup_early_exit) {
            atomicAdd(overflow_stats.raster_break_subgroup_early_exit, 1u);
        }
        if (has_depth) {
            atomicAdd(overflow_stats.raster_has_depth, 1u);
        }
        uint alpha_q10 = uint(clamp(final_color.a, 0.0, 1.0) * 1024.0 + 0.5);
        atomicAdd(overflow_stats.raster_alpha_sum_q10, alpha_q10);
    }
#endif

    // DEBUG: Tile classification overlay (controlled by debug_flags.z / debug_show_overflow_tiles)
    // Colors:
    // - Dark blue: empty tile
    // - Red: tile overflowed during binning (capacity clamped)
    // - Purple: splats present but none contributed to any pixel in this tile
    // - Green (intensity): occupancy ratio for tiles that rendered something
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
        if (on_edge) {
            debug_color *= 0.35;
        }

        if (debug_tile_grid) {
            debug_color = gs_apply_tile_grid(frag_coord, debug_color, params.debug_overlay_opacity);
        }
        out_color = vec4(debug_color, 1.0);
        out_depth = 1.0;
        out_normal = vec4(0.0);
        return;
    }

    if (debug_density_heatmap) {
        float avg_records = max(1.0, float(indirect_dispatch.element_count) / max(1.0, params.tile_count.x * params.tile_count.y));
        float density = clamp(float(splat_count) / avg_records, 0.0, 4.0) * 0.25;
        vec3 heat_color = gs_spectral_heatmap(density);
        if (debug_tile_grid) {
            heat_color = gs_apply_tile_grid(frag_coord, heat_color, params.debug_overlay_opacity);
        }
        out_color = vec4(heat_color, 1.0);
        out_depth = depth_out;
        out_normal = vec4(0.0);
        return;
    }

    if (debug_depth_visualization) {
        float depth_range = max(params.far_plane - params.near_plane, 0.001);
        float depth_norm = clamp((depth_out - params.near_plane) / depth_range, 0.0, 1.0);
        vec3 depth_color = gs_spectral_heatmap(1.0 - depth_norm);
        if (debug_tile_grid) {
            depth_color = gs_apply_tile_grid(frag_coord, depth_color, params.debug_overlay_opacity);
        }
        out_color = vec4(depth_color, 1.0);
        out_depth = depth_out;
        out_normal = vec4(0.0);
        return;
    }

    // Production path: For empty tiles or tiles where no splats rendered,
    // output transparent so the background shows through
    if (splat_count == 0u || !has_depth) {
        out_color = vec4(0.0, 0.0, 0.0, 0.0);
        out_depth = 1.0;
        out_normal = vec4(0.0);
        return;
    }

    // Dither alpha to soften 8-bit quantization on silhouettes.
    final_color.a = clamp(final_color.a + pixel_dither.r, 0.0, 1.0);

    // Optional solid-coverage mode: enforce a minimum alpha wherever splats contributed.
    if (params.force_solid_coverage != 0u) {
        final_color.a = max(final_color.a, params.alpha_floor);
    }

    // Output the correctly composited color directly
    // The accumulation loop already computed the final color using the "over" operator
    // No alpha conversion/division is needed - final_color.rgb is already correct
    if (debug_tile_grid) {
        final_color.rgb = gs_apply_tile_grid(frag_coord, final_color.rgb, params.debug_overlay_opacity);
    }
    out_color = clamp(final_color, vec4(0.0), vec4(1.0));
    out_depth = depth_out;
    out_normal = vec4(final_normal, final_color.a);
}

#endif
