#ifndef GS_LIGHTING_COMMON_GLSL
#define GS_LIGHTING_COMMON_GLSL

// Read the packed item range for one clustered-light record.
void cluster_get_item_range(uint offset, out uint item_min, out uint item_max, out uint item_from, out uint item_to) {
    uint item_min_max = cluster_buffer.data[offset];
    item_min = item_min_max & 0xFFFFu;
    item_max = item_min_max >> 16;
    item_from = item_min >> 5;
    item_to = (item_max == 0u) ? 0u : ((item_max - 1u) >> 5) + 1u;
}

// Compute the clip mask used to reject clustered-light slices outside range.
uint cluster_get_range_clip_mask(uint i, uint z_min, uint z_max) {
    int local_min = clamp(int(z_min) - int(i) * 32, 0, 31);
    int mask_width = min(int(z_max) - int(z_min), 32 - local_min);
    return bitfieldInsert(uint(0), uint(0xFFFFFFFFu), local_min, mask_width);
}

// Return whether clustered lights are active for this frame.
bool gs_use_clustered_lights() {
    return (params.light_counts.z != 0u) && (params.cluster_config.z != 0u);
}

void gs_get_cluster_params(vec2 pixel_pos, vec3 view_pos, out uint cluster_offset, out uint cluster_z,
        out uint cluster_type_size, out uint max_cluster_element_count_div_32) {
    uint cluster_shift = params.cluster_config.x;
    uint cluster_width = params.cluster_config.y;
    cluster_type_size = params.cluster_config.z;
    max_cluster_element_count_div_32 = params.cluster_config.w;

    vec2 viewport_size = max(params.viewport_size, vec2(1.0));
    vec2 clamped_pos = clamp(pixel_pos, vec2(0.0), viewport_size - vec2(1.0));
    uvec2 cluster_pos = uvec2(clamped_pos) >> cluster_shift;
    cluster_offset = (cluster_width * cluster_pos.y + cluster_pos.x) * (max_cluster_element_count_div_32 + 32u);
    cluster_z = uint(clamp((-view_pos.z / scene_data_block.data.z_far) * 32.0, 0.0, 31.0));
}

void gs_accumulate_directional_lights(vec3 view_pos, vec3 normal, float receiver_bias, bool shadow_sampling_enabled,
        uint light_mask, hvec3 h_normal_base, hvec3 h_view, hvec3 h_albedo, half roughness, half metallic, hvec3 f0,
        half alpha, vec2 uv, hvec3 energy_compensation, inout hvec3 diffuse_light, inout hvec3 specular_light,
        inout float sh_occlusion) {
    uint dir_count = min(scene_data_block.data.directional_light_count, uint(MAX_DIRECTIONAL_LIGHT_DATA_STRUCTS));
    for (uint i = 0u; i < dir_count; ++i) {
        DirectionalLightData light = directional_lights.data[i];
        if ((light.mask & light_mask) == 0u) {
            continue;
        }
        vec3 light_dir_f = normalize(light.direction);
        vec3 shadow_normal = normal;
        if (dot(shadow_normal, light_dir_f) < 0.0) {
            shadow_normal = -shadow_normal;
        }
        float shadow = shadow_sampling_enabled
                ? gs_directional_shadow(i, view_pos, shadow_normal, scene_data_block.data.taa_frame_count, receiver_bias)
                : 1.0;
        sh_occlusion = max(sh_occlusion, 1.0 - shadow);
        hvec3 light_dir = normalize(hvec3(light_dir_f));
        hvec3 light_color = hvec3(light.color) * half(light.energy);
        hvec3 h_normal = h_normal_base;
        if (dot(h_normal, light_dir) < half(0.0)) {
            h_normal = -h_normal;
        }
        light_compute(h_normal, light_dir, h_view, half(0.0), light_color, true, half(shadow), f0, roughness,
                metallic, half(light.specular), h_albedo, alpha, uv, energy_compensation,
                diffuse_light, specular_light);
    }
}

void gs_accumulate_clustered_omni_spot_sh_occlusion(uint cluster_offset, uint cluster_z, uint cluster_type_size,
        uint max_cluster_element_count_div_32, vec3 view_pos, vec3 normal, bool shadow_sampling_enabled,
        uint light_mask, inout float sh_occlusion) {
    if (sh_occlusion >= 1.0) {
        return;
    }
    uint item_min, item_max, item_from, item_to;
    cluster_get_item_range(cluster_offset + max_cluster_element_count_div_32 + cluster_z,
            item_min, item_max, item_from, item_to);
    for (uint i = item_from; i < item_to; ++i) {
        if (sh_occlusion >= 1.0) {
            break;
        }
        uint mask = cluster_buffer.data[cluster_offset + i];
        mask &= cluster_get_range_clip_mask(i, item_min, item_max);
        while (mask != 0u) {
            uint bit = uint(findMSB(mask));
            mask &= ~(1u << bit);
            uint omni_index = 32u * i + bit;
            if (omni_index >= params.light_counts.x) continue;
            if ((omni_lights.data[omni_index].mask & light_mask) == 0u) continue;
            float omni_atten = 0.0;
            vec3 light_rel_vec = omni_lights.data[omni_index].position - view_pos;
            vec3 light_rel_vec_norm = light_rel_vec / max(length(light_rel_vec), 1e-6);
            vec3 shadow_normal = normal;
            if (dot(shadow_normal, light_rel_vec_norm) < 0.0) {
                shadow_normal = -shadow_normal;
            }
            float omni_shadow = shadow_sampling_enabled
                    ? gs_omni_shadow_factor(omni_index, view_pos, shadow_normal, scene_data_block.data.taa_frame_count, omni_atten)
                    : 1.0;
            sh_occlusion = max(sh_occlusion, (1.0 - omni_shadow) * clamp(omni_atten, 0.0, 1.0));
            if (sh_occlusion >= 1.0) {
                break;
            }
        }
    }

    uint cluster_spot_offset = cluster_offset + cluster_type_size;
    cluster_get_item_range(cluster_spot_offset + max_cluster_element_count_div_32 + cluster_z,
            item_min, item_max, item_from, item_to);
    for (uint i = item_from; i < item_to; ++i) {
        if (sh_occlusion >= 1.0) {
            break;
        }
        uint mask = cluster_buffer.data[cluster_spot_offset + i];
        mask &= cluster_get_range_clip_mask(i, item_min, item_max);
        while (mask != 0u) {
            uint bit = uint(findMSB(mask));
            mask &= ~(1u << bit);
            uint spot_index = 32u * i + bit;
            if (spot_index >= params.light_counts.y) continue;
            if ((spot_lights.data[spot_index].mask & light_mask) == 0u) continue;
            float spot_atten = 0.0;
            vec3 light_rel_vec = spot_lights.data[spot_index].position - view_pos;
            vec3 light_rel_vec_norm = light_rel_vec / max(length(light_rel_vec), 1e-6);
            vec3 shadow_normal = normal;
            if (dot(shadow_normal, light_rel_vec_norm) < 0.0) {
                shadow_normal = -shadow_normal;
            }
            float spot_shadow = shadow_sampling_enabled
                    ? gs_spot_shadow_factor(spot_index, view_pos, shadow_normal, scene_data_block.data.taa_frame_count, spot_atten)
                    : 1.0;
            sh_occlusion = max(sh_occlusion, (1.0 - spot_shadow) * clamp(spot_atten, 0.0, 1.0));
            if (sh_occlusion >= 1.0) {
                break;
            }
        }
    }
}

void gs_accumulate_unclustered_omni_spot_sh_occlusion(vec3 view_pos, vec3 normal, bool shadow_sampling_enabled,
        uint light_mask, inout float sh_occlusion) {
    if (sh_occlusion >= 1.0) {
        return;
    }
    uint omni_count = min(params.light_counts.x, uint(GS_MAX_OMNI_LIGHTS));
    for (uint i = 0u; i < omni_count; ++i) {
        if (sh_occlusion >= 1.0) {
            break;
        }
        if ((omni_lights.data[i].mask & light_mask) == 0u) continue;
        float omni_atten = 0.0;
        vec3 light_rel_vec = omni_lights.data[i].position - view_pos;
        vec3 light_rel_vec_norm = light_rel_vec / max(length(light_rel_vec), 1e-6);
        vec3 shadow_normal = normal;
        if (dot(shadow_normal, light_rel_vec_norm) < 0.0) {
            shadow_normal = -shadow_normal;
        }
        float omni_shadow = shadow_sampling_enabled
                ? gs_omni_shadow_factor(i, view_pos, shadow_normal, scene_data_block.data.taa_frame_count, omni_atten)
                : 1.0;
        sh_occlusion = max(sh_occlusion, (1.0 - omni_shadow) * clamp(omni_atten, 0.0, 1.0));
    }

    uint spot_count = min(params.light_counts.y, uint(GS_MAX_SPOT_LIGHTS));
    for (uint i = 0u; i < spot_count; ++i) {
        if (sh_occlusion >= 1.0) {
            break;
        }
        if ((spot_lights.data[i].mask & light_mask) == 0u) continue;
        float spot_atten = 0.0;
        vec3 light_rel_vec = spot_lights.data[i].position - view_pos;
        vec3 light_rel_vec_norm = light_rel_vec / max(length(light_rel_vec), 1e-6);
        vec3 shadow_normal = normal;
        if (dot(shadow_normal, light_rel_vec_norm) < 0.0) {
            shadow_normal = -shadow_normal;
        }
        float spot_shadow = shadow_sampling_enabled
                ? gs_spot_shadow_factor(i, view_pos, shadow_normal, scene_data_block.data.taa_frame_count, spot_atten)
                : 1.0;
        sh_occlusion = max(sh_occlusion, (1.0 - spot_shadow) * clamp(spot_atten, 0.0, 1.0));
    }
}

void gs_accumulate_clustered_omni_spot_direct(uint cluster_offset, uint cluster_z, uint cluster_type_size,
        uint max_cluster_element_count_div_32, vec3 view_pos, hvec3 h_normal_base, hvec3 h_view,
        hvec3 h_albedo, half roughness, half metallic, hvec3 f0, half alpha, vec2 uv, hvec3 energy_compensation,
        uint light_mask, inout hvec3 diffuse_light, inout hvec3 specular_light) {
    uint item_min, item_max, item_from, item_to;
    cluster_get_item_range(cluster_offset + max_cluster_element_count_div_32 + cluster_z,
            item_min, item_max, item_from, item_to);
    for (uint i = item_from; i < item_to; ++i) {
        uint mask = cluster_buffer.data[cluster_offset + i];
        mask &= cluster_get_range_clip_mask(i, item_min, item_max);
        while (mask != 0u) {
            uint bit = uint(findMSB(mask));
            mask &= ~(1u << bit);
            uint omni_index = 32u * i + bit;
            if (omni_index >= params.light_counts.x) continue;
            if ((omni_lights.data[omni_index].mask & light_mask) == 0u) continue;
            vec3 light_rel_vec = omni_lights.data[omni_index].position - view_pos;
            vec3 light_rel_vec_norm = light_rel_vec / max(length(light_rel_vec), 1e-6);
            hvec3 h_normal = h_normal_base;
            if (dot(h_normal, hvec3(light_rel_vec_norm)) < half(0.0)) {
                h_normal = -h_normal;
            }
            light_process_omni(omni_index, view_pos, h_view, h_normal, vec3(0.0), vec3(0.0), f0, roughness, metallic,
                    scene_data_block.data.taa_frame_count, h_albedo, alpha, uv, energy_compensation,
                    diffuse_light, specular_light);
        }
    }

    uint cluster_spot_offset = cluster_offset + cluster_type_size;
    cluster_get_item_range(cluster_spot_offset + max_cluster_element_count_div_32 + cluster_z,
            item_min, item_max, item_from, item_to);
    for (uint i = item_from; i < item_to; ++i) {
        uint mask = cluster_buffer.data[cluster_spot_offset + i];
        mask &= cluster_get_range_clip_mask(i, item_min, item_max);
        while (mask != 0u) {
            uint bit = uint(findMSB(mask));
            mask &= ~(1u << bit);
            uint spot_index = 32u * i + bit;
            if (spot_index >= params.light_counts.y) continue;
            if ((spot_lights.data[spot_index].mask & light_mask) == 0u) continue;
            vec3 light_rel_vec = spot_lights.data[spot_index].position - view_pos;
            vec3 light_rel_vec_norm = light_rel_vec / max(length(light_rel_vec), 1e-6);
            hvec3 h_normal = h_normal_base;
            if (dot(h_normal, hvec3(light_rel_vec_norm)) < half(0.0)) {
                h_normal = -h_normal;
            }
            light_process_spot(spot_index, view_pos, h_view, h_normal, vec3(0.0), vec3(0.0), f0, roughness, metallic,
                    scene_data_block.data.taa_frame_count, h_albedo, alpha, uv, energy_compensation,
                    diffuse_light, specular_light);
        }
    }
}

void gs_accumulate_unclustered_omni_spot_direct(vec3 view_pos, hvec3 h_normal_base, hvec3 h_view,
        hvec3 h_albedo, half roughness, half metallic, hvec3 f0, half alpha, vec2 uv, hvec3 energy_compensation,
        uint light_mask, inout hvec3 diffuse_light, inout hvec3 specular_light) {
    uint omni_count = min(params.light_counts.x, uint(GS_MAX_OMNI_LIGHTS));
    for (uint i = 0u; i < omni_count; ++i) {
        if ((omni_lights.data[i].mask & light_mask) == 0u) continue;
        vec3 light_rel_vec = omni_lights.data[i].position - view_pos;
        vec3 light_rel_vec_norm = light_rel_vec / max(length(light_rel_vec), 1e-6);
        hvec3 h_normal = h_normal_base;
        if (dot(h_normal, hvec3(light_rel_vec_norm)) < half(0.0)) {
            h_normal = -h_normal;
        }
        light_process_omni(i, view_pos, h_view, h_normal, vec3(0.0), vec3(0.0), f0, roughness, metallic,
                scene_data_block.data.taa_frame_count, h_albedo, alpha, uv, energy_compensation,
                diffuse_light, specular_light);
    }

    uint spot_count = min(params.light_counts.y, uint(GS_MAX_SPOT_LIGHTS));
    for (uint i = 0u; i < spot_count; ++i) {
        if ((spot_lights.data[i].mask & light_mask) == 0u) continue;
        vec3 light_rel_vec = spot_lights.data[i].position - view_pos;
        vec3 light_rel_vec_norm = light_rel_vec / max(length(light_rel_vec), 1e-6);
        hvec3 h_normal = h_normal_base;
        if (dot(h_normal, hvec3(light_rel_vec_norm)) < half(0.0)) {
            h_normal = -h_normal;
        }
        light_process_spot(i, view_pos, h_view, h_normal, vec3(0.0), vec3(0.0), f0, roughness, metallic,
                scene_data_block.data.taa_frame_count, h_albedo, alpha, uv, energy_compensation,
                diffuse_light, specular_light);
    }
}

#endif // GS_LIGHTING_COMMON_GLSL
