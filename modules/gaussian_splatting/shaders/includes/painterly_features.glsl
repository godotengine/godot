#ifndef PAINTERLY_FEATURES_GLSL
#define PAINTERLY_FEATURES_GLSL

#define PAINTERLY_MAX_PALETTE_COLORS 8
#define PAINTERLY_MAX_LIGHTS 4

#define PAINTERLY_STYLE_REALISTIC 0
#define PAINTERLY_STYLE_CEL 1
#define PAINTERLY_STYLE_PAINTERLY 2
#define PAINTERLY_STYLE_GOOCH 3

#ifdef PAINTERLY_ENABLE_PALETTE
layout(set = 1, binding = 0, std140) uniform PainterlyPalette {
    vec4 colors[PAINTERLY_MAX_PALETTE_COLORS];
    vec4 params; // x: count, y: blend strength, z: noise strength, w: preserve luminance (>0.5)
} painterly_palette;

vec3 painterly_apply_palette_quantization(vec3 color, vec2 seeds) {
    int count = int(round(painterly_palette.params.x));
    count = clamp(count, 0, PAINTERLY_MAX_PALETTE_COLORS);

    if (count == 0) {
        return color;
    }

    // Optimization: Use squared distance to avoid sqrt() in hot loop
    // Since we only compare distances, squared comparison gives same result
    float best_distance_sq = 1e20;
    vec3 closest_color = color;
    for (int i = 0; i < count; i++) {
        vec3 candidate = painterly_palette.colors[i].rgb;
        vec3 diff = color - candidate;
        float distance_sq = dot(diff, diff);  // squared distance, avoids sqrt
        if (distance_sq < best_distance_sq) {
            best_distance_sq = distance_sq;
            closest_color = candidate;
        }
    }

    float blend_strength = clamp(painterly_palette.params.y, 0.0, 1.0);
    float noise_strength = painterly_palette.params.z;
    float preserve_luminance = painterly_palette.params.w;

    float jitter = (seeds.x * 2.0 - 1.0) * noise_strength;
    float blend_amount = clamp(blend_strength + jitter, 0.0, 1.0);

    if (preserve_luminance > 0.5) {
        float src_luma = dot(color, vec3(0.299, 0.587, 0.114));
        float dst_luma = dot(closest_color, vec3(0.299, 0.587, 0.114));
        if (dst_luma > 1e-4) {
            closest_color *= src_luma / dst_luma;
        } else {
            closest_color *= src_luma;
        }
    }

    vec3 blended = mix(color, closest_color, blend_amount);
    return clamp(blended, 0.0, 1.0);
}
#endif // PAINTERLY_ENABLE_PALETTE

#ifdef PAINTERLY_ENABLE_BRUSH
layout(set = 1, binding = 1, std140) uniform PainterlyBrush {
    vec4 params0; // x: scale, y: softness, z: anisotropy, w: rotation jitter
    vec4 params1; // x: texture noise strength, yzw: reserved
} painterly_brush;

float painterly_apply_brush_modulation(vec2 uv, vec2 seeds) {
    vec2 centered = uv;

    float jitter_angle = painterly_brush.params0.w * (seeds.y * 2.0 - 1.0) * 3.14159265;
    float c = cos(jitter_angle);
    float s = sin(jitter_angle);
    mat2 rotation = mat2(c, -s, s, c);
    vec2 rotated = rotation * centered;

    float anisotropy = max(painterly_brush.params0.z, 0.1);
    rotated.x *= anisotropy;

    float scale = max(painterly_brush.params0.x, 0.0001);
    float softness = max(painterly_brush.params0.y, 0.0001);
    float radial = dot(rotated, rotated) * scale;
    float brush_shape = exp(-radial);

    float mask = pow(clamp(brush_shape, 0.0, 1.0), softness);

    float texture_noise = painterly_brush.params1.x;
    if (texture_noise > 0.0) {
        float swirl = sin((rotated.x + rotated.y + seeds.x) * 12.9898);
        float noise = 0.5 + 0.5 * swirl;
        mask *= mix(1.0, noise, clamp(texture_noise, 0.0, 1.0));
    }

    return clamp(mask, 0.0, 1.0);
}
#endif // PAINTERLY_ENABLE_BRUSH

#ifdef PAINTERLY_ENABLE_LIGHTING
struct PainterlyLight {
    vec4 color_intensity;    // rgb: color, a: intensity multiplier
    vec4 direction_falloff;  // xyz: direction (view space), w: reserved falloff
    vec4 response;           // x: diffuse, y: specular, z: rim strength, w: specular power
};

layout(set = 1, binding = 2, std140) uniform PainterlyLighting {
    vec4 ambient_color;      // rgb: ambient tint, a: ambient intensity
    vec4 shadow_color;       // rgb: artistic shadow tint, a: shadow strength
    vec4 highlight_color;    // rgb: highlight tint, a: highlight strength
    vec4 rim_color_power;    // rgb: rim tint, a: rim power
    vec4 style_params0;      // x: shading style, y: cel band count, z: painterly mix strength, w: brush influence
    vec4 style_params1;      // x: rim blend, y: color temperature (K), z: temperature strength, w: temporal stability
    vec4 style_params2;      // x: gooch cool mix, y: gooch warm mix, z: cel softness, w: unused
    vec4 light_control;      // x: light count, y: global intensity, z/w: reserved
    PainterlyLight lights[PAINTERLY_MAX_LIGHTS];
} painterly_lighting;

vec3 painterly_color_temperature(float kelvin) {
    float k = clamp(kelvin, 1000.0, 40000.0) * 0.01;
    float r;
    float g;
    float b;

    if (k <= 66.0) {
        r = 1.0;
        g = clamp(0.39008157876901960784 * log(max(k, 1.0)) - 0.63184144378862745098, 0.0, 1.0);
        if (k <= 19.0) {
            b = 0.0;
        } else {
            b = clamp(0.54320678911019607843 * log(max(k - 10.0, 1.0)) - 1.19625408914, 0.0, 1.0);
        }
    } else {
        float t = max(k - 60.0, 1.0);
        r = clamp(1.2929361860627451 * pow(t, -0.1332047592), 0.0, 1.0);
        g = clamp(1.1298908608952941 * pow(t, -0.0755148492), 0.0, 1.0);
        b = 1.0;
    }

    return vec3(r, g, b);
}

vec3 painterly_apply_temperature(vec3 color, float kelvin, float strength) {
    float tint_strength = clamp(strength, 0.0, 1.0);
    if (tint_strength <= 0.0) {
        return color;
    }
    vec3 tint = painterly_color_temperature(kelvin);
    return mix(color, color * tint, tint_strength);
}

vec3 cel_shade(vec3 color, vec3 normal, vec3 light_dir, int bands) {
    float ndotl = max(dot(normal, painterly_safe_normalize(light_dir, vec3(0.0, 0.0, -1.0))), 0.0);
    if (bands <= 1) {
        return color * ndotl;
    }

    float band_count = float(max(bands, 1));
    float denom = max(band_count - 1.0, 1.0);
    float quantized = floor(ndotl * band_count) / denom;
    float softness = clamp(painterly_lighting.style_params2.z, 0.0, 1.0);
    float stability = clamp(painterly_lighting.style_params1.w, 0.0, 1.0);
    float smoothed = mix(quantized, ndotl, softness);
    float stabilized = mix(quantized, smoothed, stability);
    return color * clamp(stabilized, 0.0, 1.0);
}

vec3 rim_light(vec3 color, vec3 normal, vec3 view_dir, float power) {
    float rim = pow(clamp(1.0 - max(dot(normal, painterly_safe_normalize(view_dir, vec3(0.0, 0.0, 1.0))), 0.0), 0.0, 1.0), max(power, 0.001));
    return color * rim;
}

vec3 gooch_shade_with_dir(vec3 cool_color, vec3 warm_color, vec3 normal, vec3 light_dir) {
    float ndotl = clamp(dot(normal, painterly_safe_normalize(light_dir, vec3(0.0, 0.0, -1.0))), -1.0, 1.0);
    float blend = ndotl * 0.5 + 0.5;
    return mix(cool_color, warm_color, blend);
}

vec3 gooch_shade(vec3 cool_color, vec3 warm_color, vec3 normal) {
    vec3 light_dir = painterly_lighting.lights[0].direction_falloff.xyz;
    return gooch_shade_with_dir(cool_color, warm_color, normal, light_dir);
}

vec3 painterly_mix(vec3 base, vec3 light, float brush_texture) {
    float influence = clamp(painterly_lighting.style_params0.w, 0.0, 1.0);
    float factor = clamp(brush_texture * influence, 0.0, 1.0);
    return mix(base, light, factor);
}

vec3 painterly_apply_stylized_lighting(vec3 albedo, vec3 normal_vs, vec3 view_dir_vs) {
    vec3 normal = painterly_safe_normalize(normal_vs, vec3(0.0, 0.0, 1.0));
    vec3 view_dir = painterly_safe_normalize(view_dir_vs, vec3(0.0, 0.0, 1.0));

    int style = int(clamp(round(painterly_lighting.style_params0.x), 0.0, 3.0));
    int band_count = int(clamp(round(painterly_lighting.style_params0.y), 1.0, 16.0));
    float painterly_strength = clamp(painterly_lighting.style_params0.z, 0.0, 1.0);
    float rim_mix = clamp(painterly_lighting.style_params1.x, 0.0, 1.0);
    float temporal_stability = clamp(painterly_lighting.style_params1.w, 0.0, 1.0);

    vec3 ambient = albedo * painterly_lighting.ambient_color.rgb * painterly_lighting.ambient_color.a;
    vec3 shadow_tint = painterly_lighting.shadow_color.rgb;
    float shadow_strength = clamp(painterly_lighting.shadow_color.a, 0.0, 1.0);
    vec3 highlight_tint = painterly_lighting.highlight_color.rgb;
    float highlight_strength = clamp(painterly_lighting.highlight_color.a, 0.0, 4.0);
    vec3 rim_color = painterly_lighting.rim_color_power.rgb;
    float rim_power = max(painterly_lighting.rim_color_power.a, 0.1);

    float color_temperature = painterly_lighting.style_params1.y;
    float color_temperature_strength = clamp(painterly_lighting.style_params1.z, 0.0, 1.0);
    float gooch_cool_mix = clamp(painterly_lighting.style_params2.x, 0.0, 1.0);
    float gooch_warm_mix = clamp(painterly_lighting.style_params2.y, 0.0, 1.0);

    int light_count = int(clamp(round(painterly_lighting.light_control.x), 0.0, float(PAINTERLY_MAX_LIGHTS)));
    float global_intensity = painterly_lighting.light_control.y;

    vec3 lighting = ambient;
    float total_diffuse = 0.0;
    float total_specular = 0.0;
    float total_rim = 0.0;

    float brush_seed = 0.5 + 0.5 * dot(normal, vec3(0.57735, 0.57735, 0.57735));
    brush_seed = clamp(brush_seed, 0.0, 1.0);
    brush_seed = mix(brush_seed, 0.5, temporal_stability);

    for (int i = 0; i < PAINTERLY_MAX_LIGHTS; i++) {
        if (i >= light_count) {
            break;
        }

        PainterlyLight light = painterly_lighting.lights[i];
        vec3 light_dir = painterly_safe_normalize(light.direction_falloff.xyz, vec3(0.0, 0.0, -1.0));
        float intensity = light.color_intensity.a * global_intensity;
        if (intensity <= 0.0) {
            continue;
        }

        vec3 light_color = light.color_intensity.rgb * intensity;
        float ndotl = max(dot(normal, light_dir), 0.0);
        vec3 half_vector = painterly_safe_normalize(light_dir + view_dir, light_dir);
        float specular_power = max(light.response.w, 1.0);
        float specular_term = pow(max(dot(normal, half_vector), 0.0), specular_power);
        float rim_term = pow(clamp(1.0 - max(dot(normal, view_dir), 0.0), 0.0, 1.0), rim_power);

        vec3 diffuse_contrib = vec3(0.0);

        if (style == PAINTERLY_STYLE_CEL) {
            diffuse_contrib = cel_shade(albedo * light_color * light.response.x, normal, light_dir, band_count);
        } else if (style == PAINTERLY_STYLE_PAINTERLY) {
            vec3 lambert = albedo * light_color * ndotl * light.response.x;
            float brush_signal = clamp(brush_seed * (0.5 + 0.5 * ndotl), 0.0, 1.0);
            vec3 mixed = painterly_mix(albedo, lambert, brush_signal);
            diffuse_contrib = mix(lambert, mixed, painterly_strength);
        } else if (style == PAINTERLY_STYLE_GOOCH) {
            vec3 cool = mix(albedo, shadow_tint, gooch_cool_mix) * light_color;
            vec3 warm = mix(albedo, highlight_tint, gooch_warm_mix) * light_color;
            diffuse_contrib = gooch_shade_with_dir(cool, warm, normal, light_dir) * light.response.x;
        } else {
            diffuse_contrib = albedo * light_color * ndotl * light.response.x;
        }

        vec3 specular_color = mix(light_color, highlight_tint, clamp(highlight_strength, 0.0, 1.0));
        vec3 specular_contrib = specular_color * specular_term * light.response.y;

        lighting += diffuse_contrib + specular_contrib;
        total_diffuse += ndotl * light.response.x * intensity;
        total_specular += specular_term * light.response.y * intensity;
        total_rim += rim_term * light.response.z * rim_mix * intensity;
    }

    if (total_rim > 0.0) {
        lighting += rim_light(rim_color, normal, view_dir, rim_power) * total_rim;
    }

    if (shadow_strength > 0.0) {
        float shadow_factor = clamp(1.0 - total_diffuse, 0.0, 1.0);
        lighting = mix(lighting, lighting * shadow_tint, shadow_strength * shadow_factor);
    }

    if (highlight_strength > 0.0) {
        lighting += highlight_tint * highlight_strength * total_specular;
    }

    lighting = painterly_apply_temperature(lighting, color_temperature, color_temperature_strength);
    return clamp(lighting, 0.0, 1.0);
}
#endif // PAINTERLY_ENABLE_LIGHTING

#endif // PAINTERLY_FEATURES_GLSL
