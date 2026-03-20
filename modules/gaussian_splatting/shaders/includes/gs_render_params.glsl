#ifndef GS_RENDER_PARAMS_GLSL
#define GS_RENDER_PARAMS_GLSL
#define GS_RENDER_PARAMS_LAYOUT_VERSION 16 // Keep in sync with gaussian_gpu_layout.h

// Instance pipeline only: uses SplatRef indirection and asset-local quantization.

layout(set = 1, binding = 0, std140) uniform RenderParams {
    mat4 view_matrix; // World-to-view matrix (may include instance transform)
    mat4 inv_view_matrix; // Inverse of view_matrix
    mat4 projection_matrix; // View-to-clip projection matrix
    mat4 inv_projection_matrix; // Clip-to-view inverse projection matrix (GS render projection)
    vec2 viewport_size; // Viewport size in pixels
    vec2 tile_count; // Tile grid dimensions (x,y)
    uint total_gaussians; // Total splats in the source buffer
    uint visible_gaussians; // Visible splats after culling
    float near_plane; // Camera near plane distance
    float far_plane; // Camera far plane distance
    // Debug overlay controls
    vec4 debug_flags; // x: tile_grid_or_bounds, y: collect_raster_stats, z: overflow_tiles, w: projection_issues
    float debug_overlay_opacity; // Blend strength for debug overlays
    float opacity_multiplier; // Global opacity scaling factor
    float _pad0; // Explicit padding to align camera_position to 16 bytes (std140)
    float _pad1; // std140 padding
    vec4 camera_position; // xyz used, w reserved
    float alpha_floor; // Minimum alpha when force_solid_coverage is enabled
    uint force_solid_coverage; // Nonzero enforces alpha_floor for any covered pixel
    uint overlap_record_count; // Total overlap records in the sorted buffer
    uint _pad_overlap; // std140 padding
    float cull_far_tolerance; // Extra tolerance past the far plane for culling
    float tiny_splat_screen_radius; // Drop splats smaller than this pixel radius
    float max_conic_aspect;  // Max aspect ratio for conic clamping (lower = less stretching)
    float _pad_before_jacobian; // std140: align jacobian_diag_flags (vec4) to 16 bytes
    // Jacobian diagnostic toggles: x=bypass_radius_depth_floor, y=bypass_j_col2_clamp, z=invert_j_col2_sign, w=reserved
    vec4 jacobian_diag_flags;
    // Debug overlay flags:
    // x=overlay_mode (0=off, 1=density_heatmap, 2=shadow_opacity),
    // y=depth_visualization, z=projection_z_mismatch, w=white_albedo_lighting_isolation
    vec4 debug_overlay_flags;
    // Spherical Harmonics configuration:
    // x=sh_bands (0-3), y=amortization_divisor, z=amortization_phase, w=force_full_update
    // sh_bands: 0=DC only, 1=1st order, 2=2nd order, 3=3rd order (full)
    vec4 sh_config;
    // SH decode configuration:
    // x=dc_logit (1.0=decode DC with sigmoid), yzw=reserved
    vec4 sh_decode_config;
    // Opacity-aware culling (FlashGS optimization):
    // x=enabled (1.0=true), y=visibility_threshold (tau), z=reserved, w=reserved
    // When enabled, splat radii are calculated as: r = sqrt(2 * ln(alpha/tau) * lambda_max)
    // This reduces tile-Gaussian pairs by ~94% for low-opacity splats
    vec4 opacity_culling_config;
    // LOD blending configuration (LODGE technique):
    // x=lod_blend_factor (0-1, global blend factor for the frame)
    // y=lod_blend_enabled (0 or 1)
    // z=lod_blend_distance (world units)
    // w=reserved
    vec4 lod_blend_config;
    // Distance-based culling configuration:
    // x=start_distance, y=max_cull_rate, z=enabled, w=overlap_keep_ratio
    // overlap_keep_ratio is used by global-sort binning to thin overlap records
    // deterministically when the overlap budget is saturated.
    vec4 distance_cull_config;
    // Color grading parameters (aligned to vec4):
    // x=enabled (0/1), y=exposure, z=contrast, w=saturation
    vec4 color_grading_primary;
    // x=temperature, y=tint, z=hue_shift (degrees), w=reserved
    vec4 color_grading_secondary;
    // Lighting configuration:
    // x=direct_light_scale, y=indirect_sh_scale, z=enable_direct_lighting,
    // w=normal_mode (0=depth gradients, 1=view dir, 2=depth gradients with fallback)
    vec4 lighting_config;
    // Shadow configuration:
    // x=shadow_strength (0..1), yzw=reserved
    vec4 shadow_strength;
    // Shadow receiver bias configuration:
    // x=receiver_bias_scale (per-splat radius multiplier),
    // y=receiver_bias_min, z=receiver_bias_max (0=disabled), w=reserved
    vec4 shadow_bias_config;
    // Lighting mode:
    // x=direct_lighting_mode (0=resolve, 1=per-splat, 2=both), yzw=reserved
    uvec4 lighting_mode;
    // Light counts:
    // x=omni_light_count, y=spot_light_count, z=cluster_enabled, w=light_mask
    uvec4 light_counts;
    // Cluster configuration:
    // x=cluster_shift, y=cluster_width, z=cluster_type_size, w=max_cluster_element_count_div_32
    uvec4 cluster_config;
    // Instance rotation inverse for SH view direction correction (mat3 stored as 3 vec4s for std140).
    vec4 instance_rotation_inv_col0;
    vec4 instance_rotation_inv_col1;
    vec4 instance_rotation_inv_col2;
    // Global procedural wind configuration:
    // wind_dir_strength: xyz = direction (world), w = displacement strength (meters)
    vec4 wind_dir_strength;
    // wind_time_config: x = time_seconds, y = temporal_frequency, z = spatial_frequency, w = enabled (0/1)
    vec4 wind_time_config;
    // Single global sphere effector (foundation for capped multi-effector support):
    // effector_sphere: xyz = center (world), w = radius
    // effector_config: x = enabled (0/1), y = displacement strength (meters), z = falloff exponent, w = reserved
    vec4 effector_sphere;
    vec4 effector_config;
} params;

// Helper to get current SH band level from params
uint gs_get_sh_band_level() {
    return uint(clamp(params.sh_config.x, 0.0, 3.0));
}

// Return the SH amortization divisor from render params.
uint gs_get_sh_amortization_divisor() {
    return max(1u, uint(params.sh_config.y + 0.5));
}

// Return the SH amortization phase from render params.
uint gs_get_sh_amortization_phase() {
    return uint(params.sh_config.z + 0.5);
}

// Return whether SH amortization is enabled.
bool gs_is_sh_amortization_enabled() {
    return gs_get_sh_amortization_divisor() > 1u;
}

// Return whether SH amortization should force an update this frame.
bool gs_get_sh_amortization_force_update() {
    return params.sh_config.w > 0.5;
}

// Return whether DC logit decoding is enabled.
bool gs_is_dc_logit_enabled() {
    return params.sh_decode_config.x > 0.5;
}

// Helper to check if opacity-aware culling is enabled
bool gs_is_opacity_aware_culling_enabled() {
    return params.opacity_culling_config.x > 0.5;
}

// Helper to get visibility threshold (tau)
float gs_get_visibility_threshold() {
    return params.opacity_culling_config.y;
}

// Helper to get LOD blend factor from params (LODGE technique)
float gs_get_lod_blend_factor() {
    if (params.lod_blend_config.y < 0.5) {
        return 1.0;  // LOD blending disabled
    }
    return clamp(params.lod_blend_config.x, 0.0, 1.0);
}

// Helper to check if LOD blending is enabled
bool gs_is_lod_blend_enabled() {
    return params.lod_blend_config.y > 0.5;
}

// Return the overlap keep ratio used by culling and diagnostics.
float gs_get_overlap_keep_ratio() {
    return clamp(params.distance_cull_config.w, 0.0, 1.0);
}

// Return whether wind deformation is enabled.
bool gs_is_wind_enabled() {
    return params.wind_time_config.w > 0.5;
}

// Return whether the sphere effector is enabled.
bool gs_is_sphere_effector_enabled() {
    return params.effector_config.x > 0.5 && params.effector_sphere.w > 0.0;
}

#endif // GS_RENDER_PARAMS_GLSL
