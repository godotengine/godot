#ifndef GS_DEFORMATION_GLSL
#define GS_DEFORMATION_GLSL

uint gs_wind_hash_u32(uint v) {
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    v ^= v >> 16u;
    return v;
}

uint gs_decode_instance_wind_mode(float encoded_mode) {
    return uint(clamp(floor(encoded_mode + 0.5), 0.0, float(GS_INSTANCE_WIND_MODE_FORCE_ENABLED)));
}

bool gs_is_wind_enabled_for_instance(float encoded_mode, vec4 wind_time_config) {
    bool wind_enabled = wind_time_config.w > 0.5;
    uint mode = gs_decode_instance_wind_mode(encoded_mode);
    if (mode == GS_INSTANCE_WIND_MODE_FORCE_ENABLED) {
        return true;
    }
    if (mode == GS_INSTANCE_WIND_MODE_FORCE_DISABLED) {
        return false;
    }
    return wind_enabled;
}

// Sphere effector: animates splats within a spherical region
// effector_sphere: xyz = center, w = radius
// effector_config: x = enabled, y = strength, z = falloff, w = animation frequency (Hz)
vec3 gs_apply_sphere_effector(vec3 world_position, vec4 effector_sphere, vec4 effector_config, float time_seconds, uint stable_seed) {
    if (effector_config.x <= 0.5) {
        return world_position;
    }

    float radius = effector_sphere.w;
    if (radius <= 1e-6) {
        return world_position;
    }

    vec3 delta = world_position - effector_sphere.xyz;
    float distance_to_center = length(delta);
    if (distance_to_center >= radius) {
        return world_position;
    }

    float normalized = clamp(1.0 - (distance_to_center / radius), 0.0, 1.0);
    float falloff = max(effector_config.z, 0.001);
    float influence = pow(normalized, falloff);
    float strength = effector_config.y;
    if (abs(strength) <= 1e-8) {
        return world_position;
    }

    // Animation: pulse the displacement using time
    float anim_freq = effector_config.w > 0.0 ? effector_config.w : 2.0; // Default 2 Hz
    // Add per-splat phase jitter for organic look
    uint hashed = gs_wind_hash_u32(stable_seed);
    float phase_jitter = (float(hashed & 0xFFFFu) / 65535.0) * 6.28318530718;
    float anim_phase = time_seconds * anim_freq * 6.28318530718 + phase_jitter;
    float anim_factor = 0.5 + 0.5 * sin(anim_phase); // Oscillates 0..1

    vec3 direction = distance_to_center > 1e-6 ? (delta / distance_to_center) : vec3(0.0, 1.0, 0.0);
    return world_position + direction * (strength * influence * anim_factor);
}

vec3 gs_apply_wind_deformation(vec3 world_position,
        uint stable_seed,
        float opacity,
        float instance_intensity,
        float instance_wind_mode,
        vec4 instance_wind_config,
        vec4 wind_dir_strength,
        vec4 wind_time_config,
        vec4 effector_sphere,
        vec4 effector_config) {
    float time_seconds = wind_time_config.x;
    bool wind_enabled = gs_is_wind_enabled_for_instance(instance_wind_mode, wind_time_config);
    if (!wind_enabled) {
        return gs_apply_sphere_effector(world_position, effector_sphere, effector_config, time_seconds, stable_seed);
    }

    vec3 direction = wind_dir_strength.xyz;
    if (dot(instance_wind_config.xyz, instance_wind_config.xyz) > 1e-8) {
        direction = instance_wind_config.xyz;
    }
    float direction_len = length(direction);
    float strength = max(wind_dir_strength.w, 0.0);
    if (direction_len <= 1e-6 || strength <= 0.0) {
        return gs_apply_sphere_effector(world_position, effector_sphere, effector_config, time_seconds, stable_seed);
    }

    float instance_frequency_scale = instance_wind_config.w > 0.0 ? instance_wind_config.w : 1.0;
    float temporal_frequency = max(wind_time_config.y, 0.0) * instance_frequency_scale;
    float spatial_frequency = wind_time_config.z;
    float clamped_intensity = max(instance_intensity, 0.0);
    if (clamped_intensity <= 0.0) {
        return gs_apply_sphere_effector(world_position, effector_sphere, effector_config, time_seconds, stable_seed);
    }

    // Reuse opacity as a rough "stiffness" signal until a dedicated attribute exists.
    float opacity_resistance = clamp(1.0 - opacity, 0.0, 1.0);
    float resistance = mix(0.2, 1.0, opacity_resistance);

    uint hashed = gs_wind_hash_u32(stable_seed);
    float phase_jitter = (float(hashed & 0xFFFFu) / 65535.0) * 6.28318530718;
    float phase = dot(world_position.xz, vec2(spatial_frequency)) + time_seconds * temporal_frequency + phase_jitter;
    float displacement = sin(phase) * strength * resistance * clamped_intensity;

    vec3 deformed = world_position + (direction / direction_len) * displacement;
    return gs_apply_sphere_effector(deformed, effector_sphere, effector_config, time_seconds, stable_seed);
}

#endif // GS_DEFORMATION_GLSL
