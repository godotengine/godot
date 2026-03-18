#[compute]

#version 450

#include "includes/gs_render_params.glsl"

// Define GS_COMPUTE_SHADER to enable compute shader compatibility shim
// for gl_FragCoord (used in Godot's scene_forward_lights_inc.glsl)
#define GS_COMPUTE_SHADER
#include "includes/gs_lighting_bridge.glsl"
#include "includes/gs_directional_shadow.glsl"
#include "includes/gs_lighting_common.glsl"
#undef projection_matrix
#undef inv_projection_matrix

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D input_color;
layout(set = 0, binding = 1) uniform sampler2D input_depth;
layout(set = 0, binding = 4) uniform sampler2D input_normal;

#ifndef TILE_RESOLVE_FORMAT
#define TILE_RESOLVE_FORMAT 0
#endif

#if TILE_RESOLVE_FORMAT == 0
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D output_color;
#elif TILE_RESOLVE_FORMAT == 1
layout(set = 0, binding = 2, rgba16f) uniform writeonly image2D output_color;
#elif TILE_RESOLVE_FORMAT == 2
layout(set = 0, binding = 2, rgba32f) uniform writeonly image2D output_color;
#else
#error "Unsupported TILE_RESOLVE_FORMAT value"
#endif

layout(set = 0, binding = 3, r32f) uniform writeonly image2D output_depth;

layout(push_constant, std430) uniform ResolveParams {
    int viewport_width;
    int viewport_height;
    int tile_size_pixels;
    float feather_pixels;
    int tiles_x;
    int tiles_y;
    int last_tile_width;
    int last_tile_height;
    int debug_visualize_tiles;
    int use_texel_fetch_sampling;
    int output_is_premultiplied;
    int padding1;
} resolve_params;

vec4 sample_input_color(ivec2 coord, vec2 uv) {
    if (resolve_params.use_texel_fetch_sampling != 0) {
        return texelFetch(input_color, coord, 0);
    }
    return texture(input_color, uv);
}

float sample_input_depth(ivec2 coord, vec2 uv) {
    if (resolve_params.use_texel_fetch_sampling != 0) {
        return texelFetch(input_depth, coord, 0).r;
    }
    return texture(input_depth, uv).r;
}

vec4 sample_input_normal(ivec2 coord, vec2 uv) {
    if (resolve_params.use_texel_fetch_sampling != 0) {
        return texelFetch(input_normal, coord, 0);
    }
    return texture(input_normal, uv);
}

float sanitize_linear_depth(float depth_value) {
    if (!(depth_value >= 0.0 && depth_value <= 1.0)) {
        return 1.0;
    }
    return depth_value;
}

float compute_feather_weight(ivec2 coord) {
    float feather = max(resolve_params.feather_pixels, 0.0);
    if (feather <= 0.0 || resolve_params.tile_size_pixels <= 0) {
        return 1.0;
    }

    ivec2 tile_coord = coord / max(resolve_params.tile_size_pixels, 1);
    int last_tile_x = max(resolve_params.tiles_x - 1, 0);
    int last_tile_y = max(resolve_params.tiles_y - 1, 0);
    tile_coord = clamp(tile_coord, ivec2(0), ivec2(last_tile_x, last_tile_y));

    int tile_width = resolve_params.tile_size_pixels;
    int tile_height = resolve_params.tile_size_pixels;
    bool is_leftmost = tile_coord.x == 0;
    bool is_rightmost = tile_coord.x == last_tile_x;
    bool is_topmost = tile_coord.y == 0;
    bool is_bottommost = tile_coord.y == last_tile_y;
    if (is_rightmost) {
        tile_width = max(resolve_params.last_tile_width, 1);
    }
    if (is_bottommost) {
        tile_height = max(resolve_params.last_tile_height, 1);
    }

    ivec2 tile_origin = tile_coord * resolve_params.tile_size_pixels;
    ivec2 local = coord - tile_origin;

    float dist_left = float(local.x);
    float dist_right = float(tile_width - 1 - local.x);
    float dist_top = float(local.y);
    float dist_bottom = float(tile_height - 1 - local.y);
    const float kNoFeather = 1000000.0;
    if (is_leftmost) dist_left = kNoFeather;
    if (is_rightmost) dist_right = kNoFeather;
    if (is_topmost) dist_top = kNoFeather;
    if (is_bottommost) dist_bottom = kNoFeather;

    float edge_dist = min(min(dist_left, dist_right), min(dist_top, dist_bottom));
    return clamp(edge_dist / feather, 0.0, 1.0);
}

vec3 reconstruct_view_pos(mat4 inv_proj, vec2 screen_uv, float linear_depth, float z_near, float z_far, bool is_ortho) {
    float depth = mix(z_near, z_far, clamp(linear_depth, 0.0, 1.0));
    vec2 ndc = screen_uv * 2.0 - 1.0;

    vec4 view_far = inv_proj * vec4(ndc, 1.0, 1.0);
    vec3 view_far_pos = view_far.xyz / max(view_far.w, 1e-6);

    if (is_ortho) {
        vec4 view_near = inv_proj * vec4(ndc, -1.0, 1.0);
        vec3 view_near_pos = view_near.xyz / max(view_near.w, 1e-6);
        vec3 ray_dir = normalize(view_far_pos - view_near_pos);
        return view_near_pos + ray_dir * (depth - z_near);
    }

    float scale = depth / max(-view_far_pos.z, 1e-6);
    return view_far_pos * scale;
}

vec4 apply_tile_debug_overlay(vec4 color, ivec2 coord) {
    if (resolve_params.debug_visualize_tiles == 0) {
        return color;
    }
    int tile_size = max(resolve_params.tile_size_pixels, 1);
    bool on_vertical_edge = (coord.x % tile_size) == 0;
    bool on_horizontal_edge = (coord.y % tile_size) == 0;
    if (on_vertical_edge || on_horizontal_edge) {
        vec3 overlay = vec3(0.1, 0.9, 0.7);
        color.rgb = mix(color.rgb, overlay, 0.65);
        color.a = max(color.a, 1.0);
    }
    return color;
}

void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    if (coord.x >= resolve_params.viewport_width || coord.y >= resolve_params.viewport_height) {
        return;
    }

    vec2 viewport_size = vec2(max(resolve_params.viewport_width, 1), max(resolve_params.viewport_height, 1));
    vec2 uv = (vec2(coord) + vec2(0.5)) / viewport_size;

    vec4 color = sample_input_color(coord, uv);
    float depth = sanitize_linear_depth(sample_input_depth(coord, uv));

    if (color.a > 0.0) {
        color.rgb /= color.a;
    }

    // Shadow debug overlay (debug_overlay_flags.x == 2).
    // Shows shadow factor per light type: R=dir shadow, G=omni shadow, B=spot shadow
    // Bright = lit (shadow=1), Dark = shadowed (shadow=0)
    if (params.debug_overlay_flags.x > 1.5 && depth < 1.0 && color.a > 0.001) {
        bool gs_is_ortho = abs(params.projection_matrix[2][3]) < 0.5;
        vec3 view_pos_gs = reconstruct_view_pos(params.inv_projection_matrix, uv, depth, params.near_plane, params.far_plane, gs_is_ortho);
        vec3 view_pos = (scene_data_block.data.view_matrix * params.inv_view_matrix * vec4(view_pos_gs, 1.0)).xyz;

        vec3 normal = sample_input_normal(coord, uv).xyz;
        if (length(normal) < 0.001) {
            normal = vec3(0.0, 0.0, 1.0);
        }
        normal = normalize(normal);

        uint dir_count = min(scene_data_block.data.directional_light_count, uint(MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS));
        uint omni_count = params.light_counts.x;
        uint spot_count = params.light_counts.y;

        float dir_shadow = 1.0;   // 1 = lit
        float omni_shadow = 1.0;
        float spot_shadow = 1.0;
        bool has_dir = false, has_omni = false, has_spot = false;

        // Sample directional shadows
        for (uint i = 0u; i < dir_count; ++i) {
            if (directional_lights.data[i].shadow_opacity > 0.001) {
                has_dir = true;
                float receiver_bias = params.shadow_bias_config.x;
                float s = gs_directional_shadow(i, view_pos, normal, scene_data_block.data.taa_frame_count, receiver_bias);
                dir_shadow = min(dir_shadow, s);
            }
        }

        // Sample omni shadows
        for (uint i = 0u; i < omni_count; ++i) {
            if (omni_lights.data[i].shadow_opacity > 0.001) {
                has_omni = true;
                float atten;
                float s = gs_omni_shadow_factor(i, view_pos, normal, scene_data_block.data.taa_frame_count, atten);
                omni_shadow = min(omni_shadow, s);
            }
        }

        // Sample spot shadows
        for (uint i = 0u; i < spot_count; ++i) {
            if (spot_lights.data[i].shadow_opacity > 0.001) {
                has_spot = true;
                float atten;
                float s = gs_spot_shadow_factor(i, view_pos, normal, scene_data_block.data.taa_frame_count, atten);
                spot_shadow = min(spot_shadow, s);
            }
        }

        // RGB = shadow factor per type (bright = lit, dark = shadowed)
        // If no light of that type, show 0.2 (dim) instead of 0 or 1
        vec3 debug_color = vec3(
            has_dir ? dir_shadow : 0.2,
            has_omni ? omni_shadow : 0.2,
            has_spot ? spot_shadow : 0.2
        );

        imageStore(output_color, coord, vec4(debug_color, 1.0));
        imageStore(output_depth, coord, vec4(depth, 0.0, 0.0, 0.0));
        return;
    } else if (params.debug_overlay_flags.x > 1.5) {
        // No splat at this pixel - show dark blue
        imageStore(output_color, coord, vec4(0.0, 0.0, 0.2, 1.0));
        imageStore(output_depth, coord, vec4(1.0, 0.0, 0.0, 0.0));
        return;
    }

    vec3 base_color = color.rgb;
    // Debug: force white albedo to isolate lighting contribution (debug_overlay_flags.w).
    // Only apply in resolve-direct mode; per-splat mode already bakes lighting into the color.
    uint lighting_mode = params.lighting_mode.x;
    bool resolve_direct = (lighting_mode == 0u) || (lighting_mode == 2u);
    if (params.debug_overlay_flags.w > 0.5 && resolve_direct) {
        base_color = vec3(0.7);  // Neutral grey to see lighting clearly
    }
    float base_scale = (lighting_mode == 0u) ? params.lighting_config.y : 1.0;
    vec3 final_rgb = base_color * base_scale;
    float shadow_strength = clamp(params.shadow_strength.x, 0.0, 1.0);
    bool shadow_sampling_enabled = shadow_strength > 0.0;
    float sh_occlusion = 0.0;

    // Apply lighting to pixels with valid depth and non-zero alpha
    if (params.lighting_config.z > 0.5 && resolve_direct && depth < 1.0 && color.a > 0.001) {
        bool gs_is_ortho = abs(params.projection_matrix[2][3]) < 0.5;
        bool scene_is_ortho = abs(scene_data_block.data.projection_matrix[2][3]) < 0.5;
        vec3 view_pos_gs = reconstruct_view_pos(params.inv_projection_matrix, uv, depth, params.near_plane, params.far_plane, gs_is_ortho);
        // Convert GS view-space position to scene view-space so light positions match.
        vec4 world_pos = params.inv_view_matrix * vec4(view_pos_gs, 1.0);
        vec3 view_pos = (scene_data_block.data.view_matrix * world_pos).xyz;

        // DEBUG: Shadow opacity visualization (F7 + F9 together)
        // SOLID MAGENTA = debug triggered, then encode shadow_opacity
        if (params.debug_overlay_flags.z > 0.5 && params.debug_overlay_flags.w > 0.5) {
            float shadow_opacity_val = 0.0;
            uint omni_count = params.light_counts.x;
            if (omni_count > 0u) {
                shadow_opacity_val = omni_lights.data[0].shadow_opacity;
            }
            // Yellow = shadow enabled, Cyan = shadow disabled, Magenta = no lights
            vec3 debug_color;
            if (omni_count == 0u) {
                debug_color = vec3(1.0, 0.0, 1.0); // Magenta = no lights
            } else if (shadow_opacity_val > 0.001) {
                debug_color = vec3(1.0, 1.0, 0.0); // Yellow = shadow_opacity > 0
            } else {
                debug_color = vec3(0.0, 1.0, 1.0); // Cyan = shadow_opacity == 0
            }
            imageStore(output_color, coord, vec4(debug_color, 1.0));
            imageStore(output_depth, coord, vec4(depth, 0.0, 0.0, 0.0));
            return;
        }

        // Debug: visualize view-space Z mismatch between GS projection and scene_data projection (F7 only).
        if (params.debug_overlay_flags.z > 0.5) {
            vec3 view_pos_scene = reconstruct_view_pos(scene_data_block.data.inv_projection_matrix, uv, depth,
                    params.near_plane, params.far_plane, scene_is_ortho);
            float diff = abs(view_pos_gs.z - view_pos_scene.z);
            float range = max(params.far_plane - params.near_plane, 0.01);
            float t = clamp(diff / range, 0.0, 1.0);
            vec3 heat = mix(vec3(0.0, 0.1, 0.4), vec3(1.0, 0.2, 0.0), t);
            imageStore(output_color, coord, vec4(heat, 1.0));
            imageStore(output_depth, coord, vec4(depth, 0.0, 0.0, 0.0));
            return;
        }
        vec3 view_dir = normalize(-view_pos);
        vec4 normal_sample = sample_input_normal(coord, uv);
        vec3 normal = view_dir;
        if (normal_sample.a > 1e-4) {
            vec3 normal_candidate = normal_sample.rgb / max(normal_sample.a, 1e-6);
            if (dot(normal_candidate, normal_candidate) > 1e-6) {
                normal = normalize(normal_candidate);
            }
        }

        hvec3 h_normal_base = hvec3(normal);
        hvec3 h_view = hvec3(view_dir);
        // Use neutral grey albedo for lighting by default. If indirect SH is disabled,
        // fall back to the base color so splat colors still show up in direct lighting.
        hvec3 h_albedo = hvec3(0.5);
        if (base_scale <= 0.001) {
            h_albedo = hvec3(clamp(base_color, vec3(0.0), vec3(1.0)));
        }
        half roughness = half(1.0);
        half metallic = half(0.0);
        half specular = half(0.5);
        hvec3 f0 = F0(metallic, specular, h_albedo);
        hvec3 diffuse_light = hvec3(0.0);
        hvec3 specular_light = hvec3(0.0);
        half alpha = half(color.a);
        hvec3 energy_compensation = hvec3(1.0);
        float receiver_bias = params.shadow_bias_config.y;
        if (params.shadow_bias_config.z > 0.0) {
            receiver_bias = min(receiver_bias, params.shadow_bias_config.z);
        }

        uint light_mask = params.light_counts.w;
        gs_accumulate_directional_lights(view_pos, normal, receiver_bias, shadow_sampling_enabled,
                light_mask, h_normal_base, h_view, h_albedo, roughness, metallic, f0, alpha, uv,
                energy_compensation, diffuse_light, specular_light, sh_occlusion);

        bool use_clustered = gs_use_clustered_lights();
        bool shadow_modulate_sh = shadow_strength > 0.0;
        uint cluster_offset = 0u;
        uint cluster_z = 0u;
        uint cluster_type_size = 0u;
        uint max_cluster_element_count_div_32 = 0u;
        if (use_clustered) {
            gs_get_cluster_params(vec2(coord), view_pos, cluster_offset, cluster_z, cluster_type_size,
                    max_cluster_element_count_div_32);

            if (shadow_modulate_sh) {
                gs_accumulate_clustered_omni_spot_sh_occlusion(cluster_offset, cluster_z, cluster_type_size,
                        max_cluster_element_count_div_32, view_pos, normal, shadow_sampling_enabled, light_mask,
                        sh_occlusion);
            }
        } else {
            if (shadow_modulate_sh) {
                gs_accumulate_unclustered_omni_spot_sh_occlusion(view_pos, normal, shadow_sampling_enabled,
                        light_mask, sh_occlusion);
            }
        }

        if (shadow_strength > 0.0 && sh_occlusion > 0.0) {
            float sh_factor = 1.0 - shadow_strength * clamp(sh_occlusion, 0.0, 1.0);
            final_rgb *= sh_factor;
        }
        if (use_clustered) {
            gs_accumulate_clustered_omni_spot_direct(cluster_offset, cluster_z, cluster_type_size,
                    max_cluster_element_count_div_32, view_pos, h_normal_base, h_view, h_albedo, roughness,
                    metallic, f0, alpha, uv, energy_compensation, light_mask, diffuse_light, specular_light);
        } else {
            gs_accumulate_unclustered_omni_spot_direct(view_pos, h_normal_base, h_view, h_albedo, roughness,
                    metallic, f0, alpha, uv, energy_compensation, light_mask, diffuse_light, specular_light);
        }

        // Match Godot's forward path: diffuse light is multiplied by albedo at the end.
        diffuse_light *= h_albedo;
        diffuse_light *= (half(1.0) - metallic);
        vec3 direct = vec3(diffuse_light + specular_light) * params.lighting_config.x;
        // Scale lighting by alpha^2 for smooth edge falloff
        float lighting_blend = color.a * color.a;
        final_rgb += direct * lighting_blend;

    }

    color.rgb = final_rgb;

    float feather_weight = compute_feather_weight(coord);
    if (feather_weight < 1.0) {
        color.rgb *= feather_weight;
        color.a *= feather_weight;
    }

    color = apply_tile_debug_overlay(color, coord);

    if (resolve_params.output_is_premultiplied != 0) {
        color.rgb *= color.a;
    }

    imageStore(output_color, coord, color);
    imageStore(output_depth, coord, vec4(depth, 0.0, 0.0, 0.0));
}
