#ifndef GS_DIRECTIONAL_SHADOW_GLSL
#define GS_DIRECTIONAL_SHADOW_GLSL

// Directional shadow sampling for Gaussian splats.
// Adapted from Godot's forward clustered directional shadow path (no soft shadows).
float gs_directional_shadow(uint idx, vec3 vertex, vec3 geo_normal, float taa_frame_count, float receiver_bias) {
    if (directional_lights.data[idx].shadow_opacity <= 0.001) {
        return 1.0;
    }

    float depth_z = -vertex.z;
    vec3 light_dir = directional_lights.data[idx].direction;
    vec3 base_normal_bias = geo_normal * (1.0 - max(0.0, dot(light_dir, -geo_normal)));

    vec4 pssm_coord;
    float blur_factor;

    if (depth_z < directional_lights.data[idx].shadow_split_offsets.x) {
        vec4 v = vec4(vertex, 1.0);
        v.xyz += light_dir * (directional_lights.data[idx].shadow_bias.x + receiver_bias);
        vec3 normal_bias = base_normal_bias * directional_lights.data[idx].shadow_normal_bias.x;
        normal_bias -= light_dir * dot(light_dir, normal_bias);
        v.xyz += normal_bias;
        pssm_coord = directional_lights.data[idx].shadow_matrix1 * v;
        blur_factor = 1.0;
    } else if (depth_z < directional_lights.data[idx].shadow_split_offsets.y) {
        vec4 v = vec4(vertex, 1.0);
        v.xyz += light_dir * (directional_lights.data[idx].shadow_bias.y + receiver_bias);
        vec3 normal_bias = base_normal_bias * directional_lights.data[idx].shadow_normal_bias.y;
        normal_bias -= light_dir * dot(light_dir, normal_bias);
        v.xyz += normal_bias;
        pssm_coord = directional_lights.data[idx].shadow_matrix2 * v;
        blur_factor = directional_lights.data[idx].shadow_split_offsets.x / directional_lights.data[idx].shadow_split_offsets.y;
    } else if (depth_z < directional_lights.data[idx].shadow_split_offsets.z) {
        vec4 v = vec4(vertex, 1.0);
        v.xyz += light_dir * (directional_lights.data[idx].shadow_bias.z + receiver_bias);
        vec3 normal_bias = base_normal_bias * directional_lights.data[idx].shadow_normal_bias.z;
        normal_bias -= light_dir * dot(light_dir, normal_bias);
        v.xyz += normal_bias;
        pssm_coord = directional_lights.data[idx].shadow_matrix3 * v;
        blur_factor = directional_lights.data[idx].shadow_split_offsets.x / directional_lights.data[idx].shadow_split_offsets.z;
    } else {
        vec4 v = vec4(vertex, 1.0);
        v.xyz += light_dir * (directional_lights.data[idx].shadow_bias.w + receiver_bias);
        vec3 normal_bias = base_normal_bias * directional_lights.data[idx].shadow_normal_bias.w;
        normal_bias -= light_dir * dot(light_dir, normal_bias);
        v.xyz += normal_bias;
        pssm_coord = directional_lights.data[idx].shadow_matrix4 * v;
        blur_factor = directional_lights.data[idx].shadow_split_offsets.x / directional_lights.data[idx].shadow_split_offsets.w;
    }

    pssm_coord /= pssm_coord.w;

    float shadow = sample_directional_pcf_shadow(
            directional_shadow_atlas,
            scene_data_block.data.directional_shadow_pixel_size * directional_lights.data[idx].soft_shadow_scale *
                    (blur_factor + (1.0 - blur_factor) * float(directional_lights.data[idx].blend_splits)),
            pssm_coord,
            taa_frame_count);

    if (directional_lights.data[idx].blend_splits) {
        float pssm_blend;
        float blur_factor2;
        if (depth_z < directional_lights.data[idx].shadow_split_offsets.x) {
            vec4 v = vec4(vertex, 1.0);
            v.xyz += light_dir * (directional_lights.data[idx].shadow_bias.y + receiver_bias);
            vec3 normal_bias = base_normal_bias * directional_lights.data[idx].shadow_normal_bias.y;
            normal_bias -= light_dir * dot(light_dir, normal_bias);
            v.xyz += normal_bias;
            pssm_coord = directional_lights.data[idx].shadow_matrix2 * v;
            pssm_blend = smoothstep(directional_lights.data[idx].shadow_split_offsets.x - directional_lights.data[idx].shadow_split_offsets.x * 0.1,
                    directional_lights.data[idx].shadow_split_offsets.x, depth_z);
            blur_factor2 = directional_lights.data[idx].shadow_split_offsets.x / directional_lights.data[idx].shadow_split_offsets.y;
        } else if (depth_z < directional_lights.data[idx].shadow_split_offsets.y) {
            vec4 v = vec4(vertex, 1.0);
            v.xyz += light_dir * (directional_lights.data[idx].shadow_bias.z + receiver_bias);
            vec3 normal_bias = base_normal_bias * directional_lights.data[idx].shadow_normal_bias.z;
            normal_bias -= light_dir * dot(light_dir, normal_bias);
            v.xyz += normal_bias;
            pssm_coord = directional_lights.data[idx].shadow_matrix3 * v;
            pssm_blend = smoothstep(directional_lights.data[idx].shadow_split_offsets.y - directional_lights.data[idx].shadow_split_offsets.y * 0.1,
                    directional_lights.data[idx].shadow_split_offsets.y, depth_z);
            blur_factor2 = directional_lights.data[idx].shadow_split_offsets.x / directional_lights.data[idx].shadow_split_offsets.z;
        } else if (depth_z < directional_lights.data[idx].shadow_split_offsets.z) {
            vec4 v = vec4(vertex, 1.0);
            v.xyz += light_dir * (directional_lights.data[idx].shadow_bias.w + receiver_bias);
            vec3 normal_bias = base_normal_bias * directional_lights.data[idx].shadow_normal_bias.w;
            normal_bias -= light_dir * dot(light_dir, normal_bias);
            v.xyz += normal_bias;
            pssm_coord = directional_lights.data[idx].shadow_matrix4 * v;
            pssm_blend = smoothstep(directional_lights.data[idx].shadow_split_offsets.z - directional_lights.data[idx].shadow_split_offsets.z * 0.1,
                    directional_lights.data[idx].shadow_split_offsets.z, depth_z);
            blur_factor2 = directional_lights.data[idx].shadow_split_offsets.x / directional_lights.data[idx].shadow_split_offsets.w;
        } else {
            pssm_blend = 0.0;
            blur_factor2 = 1.0;
        }

        pssm_coord /= pssm_coord.w;

        float shadow2 = sample_directional_pcf_shadow(
                directional_shadow_atlas,
                scene_data_block.data.directional_shadow_pixel_size * directional_lights.data[idx].soft_shadow_scale *
                        (blur_factor2 + (1.0 - blur_factor2) * float(directional_lights.data[idx].blend_splits)),
                pssm_coord,
                taa_frame_count);

        shadow = mix(shadow, shadow2, pssm_blend);
    }

    shadow = mix(shadow, 1.0, smoothstep(directional_lights.data[idx].fade_from, directional_lights.data[idx].fade_to, vertex.z));
    shadow = mix(1.0, shadow, directional_lights.data[idx].shadow_opacity);
    return shadow;
}

// Omni shadow factor + attenuation (hard shadow path, soft shadows disabled).
float gs_omni_shadow_factor(uint idx, vec3 vertex, vec3 geo_normal, float taa_frame_count, out float attenuation) {
    vec3 light_rel_vec = omni_lights.data[idx].position - vertex;
    float light_length = length(light_rel_vec);
    attenuation = float(get_omni_attenuation(light_length, omni_lights.data[idx].inv_radius, omni_lights.data[idx].attenuation));

    if (attenuation <= float(HALF_FLT_MIN) || omni_lights.data[idx].shadow_opacity <= 0.001) {
        return 1.0;
    }

    vec2 texel_size = scene_data_block.data.shadow_atlas_pixel_size;
    vec4 base_uv_rect = omni_lights.data[idx].atlas_rect;
    base_uv_rect.xy += texel_size;
    base_uv_rect.zw -= texel_size * 2.0;

    // Omni lights use direction.xy to store the offset between the two paraboloid regions.
    vec2 flip_offset = omni_lights.data[idx].direction.xy;

    vec3 local_vert = (omni_lights.data[idx].shadow_matrix * vec4(vertex, 1.0)).xyz;
    float shadow_len = length(local_vert);
    vec3 shadow_dir = normalize(local_vert);

    vec3 local_normal = normalize(mat3(omni_lights.data[idx].shadow_matrix) * vec3(geo_normal));
    vec3 normal_bias = local_normal * omni_lights.data[idx].shadow_normal_bias * (1.0 - abs(dot(local_normal, shadow_dir)));

    vec4 uv_rect = base_uv_rect;
    vec3 shadow_sample = normalize(shadow_dir + normal_bias);
    if (shadow_sample.z >= 0.0) {
        uv_rect.xy += flip_offset;
        flip_offset *= -1.0;
    }

    shadow_sample.z = 1.0 + abs(shadow_sample.z);
    vec2 pos = shadow_sample.xy / shadow_sample.z;
    float depth = shadow_len - omni_lights.data[idx].shadow_bias;
    depth *= omni_lights.data[idx].inv_radius;
    depth = 1.0 - depth;

    float shadow = mix(1.0, sample_omni_pcf_shadow(shadow_atlas,
                    omni_lights.data[idx].soft_shadow_scale / shadow_sample.z,
                    pos, uv_rect, flip_offset, depth, taa_frame_count),
            omni_lights.data[idx].shadow_opacity);
    return shadow;
}

// Spot shadow factor + attenuation (hard shadow path, soft shadows disabled).
float gs_spot_shadow_factor(uint idx, vec3 vertex, vec3 geo_normal, float taa_frame_count, out float attenuation) {
    vec3 light_rel_vec = spot_lights.data[idx].position - vertex;
    float light_length = length(light_rel_vec);
    vec3 light_rel_vec_norm = light_rel_vec / max(light_length, 1e-6);

    float spot_attenuation = float(get_omni_attenuation(light_length, spot_lights.data[idx].inv_radius, spot_lights.data[idx].attenuation));
    vec3 spot_dir = spot_lights.data[idx].direction;
    float cone_angle = spot_lights.data[idx].cone_angle;
    float scos = max(dot(-light_rel_vec_norm, spot_dir), cone_angle);
    float spot_rim = max(1e-4, (1.0 - scos) / max(1e-6, (1.0 - cone_angle)));
    spot_attenuation *= (1.0 - pow(spot_rim, spot_lights.data[idx].cone_attenuation));

    attenuation = spot_attenuation;
    if (spot_attenuation <= float(HALF_FLT_MIN) || spot_lights.data[idx].shadow_opacity <= 0.001) {
        return 1.0;
    }

    vec3 normal_bias = vec3(geo_normal) * light_length * spot_lights.data[idx].shadow_normal_bias *
            (1.0 - abs(dot(geo_normal, light_rel_vec_norm)));

    vec4 v = vec4(vertex + normal_bias, 1.0);
    vec4 splane = (spot_lights.data[idx].shadow_matrix * v);
    splane.z += spot_lights.data[idx].shadow_bias;
    splane /= splane.w;

    vec3 shadow_uv = vec3(splane.xy * spot_lights.data[idx].atlas_rect.zw + spot_lights.data[idx].atlas_rect.xy, splane.z);
    float shadow = mix(1.0, sample_pcf_shadow(shadow_atlas,
                    spot_lights.data[idx].soft_shadow_scale * scene_data_block.data.shadow_atlas_pixel_size,
                    shadow_uv, taa_frame_count),
            spot_lights.data[idx].shadow_opacity);
    return shadow;
}

#endif // GS_DIRECTIONAL_SHADOW_GLSL
