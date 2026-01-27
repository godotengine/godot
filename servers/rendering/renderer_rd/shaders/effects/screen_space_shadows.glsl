#[compute]
#version 450

#VERSION_DEFINES

#define BRDF_NDOTL_BIAS 0.1
#define MAX_SAMPLE_COUNT 128
#define PI 3.14159265359

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba8, set = 0, binding = 0) uniform image2D color_image;
layout(set = 1, binding = 0) uniform sampler2D depth_image;
layout(set = 1, binding = 1) uniform sampler2D normal_image;
layout(set = 1, binding = 2) uniform sampler2D history_image;
layout(set = 1, binding = 3) uniform sampler2D motion_image;
layout(set = 1, binding = 4, r8) uniform writeonly image2D history_write_image;

layout(set = 2, binding = 0, std140) uniform Params {
    mat4 proj;
    mat4 proj_inv;
    mat4 view;
    mat4 view_inv;
} params;

layout(push_constant, std430) uniform PushConstant {
    vec2 screen_size_rcp;
    ivec2 screen_size;
    vec3 light_dir;
    float thickness;
    float max_dist;
    float intensity;
    uint sample_count;
    uint use_normals;
    
    vec3 camera_pos;
    uint frame_count;
    float light_radius;
    float thickness_falloff;
    float contact_shadow_distance;
    float shadow_fade_range;
    float history_blend;
    uint use_history;
    vec2 history_pad;
} push_constant;

vec3 hash33(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.xxy + p.yxx) * p.zyx);
}

vec3 hemisphere_point_cos(float u, float v, vec3 normal) {
    float phi = 2.0 * PI * u;
    float cos_theta = sqrt(v);
    float sin_theta = sqrt(1.0 - v);
    
    vec3 tangent = normalize(cross(normal, vec3(0.0, 1.0, 0.0)));
    if (length(tangent) < 0.001) {
        tangent = normalize(cross(normal, vec3(1.0, 0.0, 0.0)));
    }
    vec3 bitangent = cross(normal, tangent);
    
    return tangent * (cos(phi) * sin_theta) + 
           bitangent * (sin(phi) * sin_theta) + 
           normal * cos_theta;
}

vec3 get_world_position(vec2 uv, float depth) {
    vec4 clip_pos = vec4(uv * 2.0 - 1.0, depth, 1.0);
    vec4 view_pos = params.proj_inv * clip_pos;
    view_pos.xyz /= view_pos.w;
    return (params.view_inv * vec4(view_pos.xyz, 1.0)).xyz;
}

float get_linear_depth(vec3 world_pos) {
    return length(world_pos - push_constant.camera_pos);
}

bool is_valid_uv(vec2 uv) {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

float trace_shadow_ray(vec3 origin_ws, vec3 direction_ws, vec2 origin_uv, float linear_depth, out float fade) {
    
    if (linear_depth <= 0.0) {
        fade = 0.0;
        return 0.0;
    }

    vec3 noise_seed = vec3(origin_uv * vec2(push_constant.screen_size), push_constant.frame_count) * 0.001 * linear_depth;
    vec3 blue_noise = hash33(noise_seed);
    
    float base_ray_length = push_constant.max_dist > 0.0 ? push_constant.max_dist * 2.0 : 2000.0;
    float ray_length = base_ray_length * clamp(linear_depth * 0.05, 0.8, 4.0);
    
    float light_radius = push_constant.light_radius;
    vec3 jittered_light_dir = direction_ws;
    
    if (light_radius > 0.0) {
        vec3 hemisphere_sample = hemisphere_point_cos(blue_noise.x, blue_noise.y, direction_ws);
        jittered_light_dir = normalize(direction_ws + hemisphere_sample * light_radius * 0.1);
    }

    float ray_bias = push_constant.contact_shadow_distance > 0.0 ? push_constant.contact_shadow_distance : 0.02;
    vec3 ray_start = origin_ws + jittered_light_dir * ray_bias;
    vec3 ray_end = origin_ws + jittered_light_dir * ray_length;
    
    vec4 clip_start = params.proj * params.view * vec4(ray_start, 1.0);
    vec4 clip_end = params.proj * params.view * vec4(ray_end, 1.0);
    
    clip_start.xyz /= clip_start.w;
    clip_end.xyz /= clip_end.w;
    
    if (clip_start.z <= 0.0 || clip_end.z <= 0.0) {
        fade = 0.0;
        return 0.0;
    }
    
    vec2 uv_start = clip_start.xy * 0.5 + 0.5;
    vec2 uv_end = clip_end.xy * 0.5 + 0.5;
    vec2 uv_delta = uv_end - uv_start;
    
    int samples = int(min(push_constant.sample_count, uint(MAX_SAMPLE_COUNT)));
    float step_size = 1.0 / float(samples);
    
    float dither = blue_noise.z;
    float t = step_size * dither;
    
    float adaptive_thickness = push_constant.thickness * (1.0 / (1.0 + linear_depth * push_constant.thickness_falloff));
    float depth_threshold = adaptive_thickness;
    
    float occlusion = 0.0;
    float hit_distance = 1.0;
    bool found_occlusion = false;
    
    for (int i = 0; i < samples; i++) {
        vec2 sample_uv = uv_start + uv_delta * t;
        
        if (!is_valid_uv(sample_uv)) {
            break;
        }
        
        float sample_depth = textureLod(depth_image, sample_uv, 0.0).r;
        vec3 ray_current = mix(clip_start.xyz, clip_end.xyz, t);
        float ray_depth = ray_current.z;
        
        float depth_diff = sample_depth - ray_depth;
        
        if (depth_diff > 0.0 && depth_diff < depth_threshold && ray_depth > 0.0) {
            float progress = t;
            occlusion = 1.0 - smoothstep(0.0, 1.0, progress);
            hit_distance = t;
            found_occlusion = true;
            break;
        }
        
        t += step_size;
        if (t > 1.0) break;
    }
    
    vec2 hit_uv = mix(uv_start, uv_end, hit_distance);
    vec2 centered_uv = abs(hit_uv * 2.0 - 1.0);
    float edge_fade = 1.0 - smoothstep(0.8, 1.0, max(centered_uv.x, centered_uv.y));
    
    float distance_fade = 1.0;
    if (push_constant.shadow_fade_range > 0.0) {
        distance_fade = 1.0 - smoothstep(push_constant.shadow_fade_range * 0.5, push_constant.shadow_fade_range * 2.0, linear_depth);
    }
    
    fade = occlusion * edge_fade * distance_fade;
    return found_occlusion ? 1.0 : 0.0;
}

void main() {
    ivec2 iuv = ivec2(gl_GlobalInvocationID.xy);

    if (iuv.x >= push_constant.screen_size.x || iuv.y >= push_constant.screen_size.y) {
        return;
    }

    vec2 uv = (vec2(iuv) + 0.5) * push_constant.screen_size_rcp;
    vec4 orig_color = imageLoad(color_image, iuv);

    float depth = texelFetch(depth_image, iuv, 0).r;
    if (depth >= 0.999) {
        if (push_constant.use_history != 0u) {
            imageStore(history_write_image, iuv, vec4(0.0));
        }
        return;
    }

    vec3 world_pos = get_world_position(uv, depth);
    float linear_depth = get_linear_depth(world_pos);
    
    vec3 light_dir = normalize(push_constant.light_dir);
    
    vec3 normal_ws = vec3(0.0, 0.0, 1.0);
    if (push_constant.use_normals != 0u) {
        vec4 encoded_normal = texelFetch(normal_image, iuv, 0);
        if (length(encoded_normal.xyz) > 0.1) {
            vec3 normal_vs = normalize(encoded_normal.xyz * 2.0 - 1.0);
            normal_ws = normalize(mat3(params.view_inv) * normal_vs);
            float ndotl = dot(normal_ws, -light_dir);
            
            if (ndotl <= BRDF_NDOTL_BIAS) {
                if (push_constant.use_history != 0u) {
                    imageStore(history_write_image, iuv, vec4(0.0));
                }
                imageStore(color_image, iuv, orig_color);
                return;
            }
            
            float normal_offset = push_constant.contact_shadow_distance > 0.0 ? 
                                push_constant.contact_shadow_distance : 0.015;
            world_pos += normal_ws * normal_offset;
        }
    }

    float fade = 0.0;
    float shadow = trace_shadow_ray(world_pos, -light_dir, uv, linear_depth, fade);
    
    float current_shadow_strength = shadow * fade * push_constant.intensity;
    float final_shadow_strength = current_shadow_strength;

    // ACUMULARE TEMPORALĂ OPTIMIZATĂ FINALĂ
    if (push_constant.use_history != 0u) {
        vec2 velocity = textureLod(motion_image, uv, 0.0).xy;
        vec2 prev_uv = uv - velocity;
        bool valid_prev = is_valid_uv(prev_uv);

        float history_value = valid_prev ? textureLod(history_image, prev_uv, 0.0).r : 0.0;
        
        // STRATEGIE AVANSATĂ DE ACUMULARE
        float adaptive_blend = push_constant.history_blend;
        
        // Factor de încredere în istorie bazat pe stabilitate
        float history_confidence = 1.0;
        
        // Verifică variația locală a istoriei pentru a detecta discontinuități
        float neighborhood_min = history_value;
        float neighborhood_max = history_value;
        const ivec2 offsets[4] = ivec2[4](ivec2(1,0), ivec2(-1,0), ivec2(0,1), ivec2(0,-1));
        
        for (int i = 0; i < 4; i++) {
            ivec2 sample_pos = iuv + offsets[i];
            if (all(greaterThanEqual(sample_pos, ivec2(0))) && all(lessThan(sample_pos, push_constant.screen_size))) {
                float neighbor_history = texelFetch(history_image, sample_pos, 0).r;
                neighborhood_min = min(neighborhood_min, neighbor_history);
                neighborhood_max = max(neighborhood_max, neighbor_history);
            }
        }
        
        // Dacă istoria este inconsistentă în vecinătate, reducem încrederea
        float neighborhood_range = neighborhood_max - neighborhood_min;
        if (neighborhood_range > 0.3) {
            history_confidence = 0.3;
        } else if (neighborhood_range > 0.15) {
            history_confidence = 0.7;
        }
        if (!valid_prev) {
            history_confidence = 0.0;
        }
        
        // Ajustare blend bazată pe multiple criterii
        if (current_shadow_strength > 0.15) {
            // Umbră puternică curentă - păstrăm mai mult istorie dacă e consistentă
            adaptive_blend = push_constant.history_blend * (0.3 + 0.7 * history_confidence);
        } else if (history_value > 0.25 && current_shadow_strength < 0.08) {
            // Istorie puternică, curent slab - păstrăm istoria
            adaptive_blend = 0.1 * history_confidence;
        } else if (current_shadow_strength < 0.02 && history_value < 0.05) {
            // Ambele slabe - resetare rapidă
            adaptive_blend = 0.8;
        }
        
        // Clamping inteligent al istoriei
        float clamped_current = clamp(current_shadow_strength, 
                                    neighborhood_min - 0.1, 
                                    neighborhood_max + 0.1);
        
        // Acumulare finală
        final_shadow_strength = mix(history_value, clamped_current, adaptive_blend);
        
        // Asigură-te că tranzițiile bruște sunt păstrate
        if (abs(current_shadow_strength - history_value) > 0.4) {
            final_shadow_strength = mix(history_value, current_shadow_strength, 0.7);
        }
        
        final_shadow_strength = clamp(final_shadow_strength, 0.0, 1.0);
        imageStore(history_write_image, iuv, vec4(final_shadow_strength, 0.0, 0.0, 0.0));
    }
    
    // APLICARE UMBRĂ FINALĂ
    if (final_shadow_strength > 0.008) {
        // Efect de umbră non-linear pentru rezultate mai naturale
        float shadow_intensity = 1.0 - pow(1.0 - final_shadow_strength, 1.5);
        vec3 shadowed_color = orig_color.rgb * (1.0 - shadow_intensity * 0.92);
        imageStore(color_image, iuv, vec4(shadowed_color, orig_color.a));
    } else {
        imageStore(color_image, iuv, orig_color);
    }
}
