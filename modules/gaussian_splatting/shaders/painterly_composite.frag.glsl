#[fragment]

#version 450

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler2D painterly_color;
layout(set = 0, binding = 1) uniform sampler2D painterly_depth;
layout(set = 0, binding = 2) uniform sampler2D scene_depth;

layout(push_constant, std430) uniform CompositePush {
    vec2 inv_viewport_size;
    float depth_bias;
    float blend_strength;
    float near_plane;
    float far_plane;
    float proj_22;
    float proj_32;
    float proj_23;
} params;

// Convert normalized scene depth to comparable view-space depth.
float linearize_scene_depth(float raw_depth) {
    if (abs(params.proj_23) < 0.5) {
        float ndc = raw_depth * 2.0 - 1.0;
        return abs(-(ndc * (params.far_plane - params.near_plane) - (params.far_plane + params.near_plane)) / 2.0);
    }
    return abs(params.proj_32 / (params.proj_22 + raw_depth));
}

// Clamp invalid depth values to a sentinel for comparisons.
float sanitize_view_depth(float depth_value) {
    if (isnan(depth_value) || isinf(depth_value)) {
        return -1.0;
    }
    return abs(depth_value);
}

// Detect whether the sampled scene depth corresponds to the background clear value.
bool is_scene_background_depth(float raw_scene_depth, float scene_view_depth) {
    float scene_depth_from_zero = sanitize_view_depth(linearize_scene_depth(0.0));
    float scene_depth_from_one = sanitize_view_depth(linearize_scene_depth(1.0));
    if (scene_depth_from_zero < 0.0 || scene_depth_from_one < 0.0) {
        return false;
    }

    float clear_raw_depth = scene_depth_from_zero >= scene_depth_from_one ? 0.0 : 1.0;
    float far_view_depth = max(scene_depth_from_zero, scene_depth_from_one);
    float far_tolerance = max(params.depth_bias * 2.0, 1e-3);

    bool raw_matches_clear = abs(raw_scene_depth - clear_raw_depth) <= 1e-6;
    bool view_matches_far = abs(scene_view_depth - far_view_depth) <= far_tolerance;
    return raw_matches_clear || view_matches_far;
}

// Fragment entry point for the painterly composite pass.
void main() {
    vec2 sample_uv = clamp(gl_FragCoord.xy * params.inv_viewport_size, vec2(0.0), vec2(1.0));

    vec4 painterly_sample = texture(painterly_color, sample_uv);
    if (painterly_sample.a <= 0.0001) {
        discard;
    }

    float painterly_depth_value = texture(painterly_depth, sample_uv).r;
    if (painterly_depth_value <= 0.0) {
        discard;
    }
    float splat_linear_depth = sanitize_view_depth(mix(params.near_plane, params.far_plane, clamp(painterly_depth_value, 0.0, 1.0)));

    float scene_depth_value = texture(scene_depth, sample_uv).r;
    if (scene_depth_value >= 0.0 && scene_depth_value <= 1.0 && painterly_depth_value < 0.999999) {
        float scene_linear_depth = sanitize_view_depth(linearize_scene_depth(scene_depth_value));
        bool scene_is_background = is_scene_background_depth(scene_depth_value, scene_linear_depth);
        if (!scene_is_background && scene_linear_depth >= 0.0 && splat_linear_depth >= 0.0 &&
                scene_linear_depth <= splat_linear_depth - params.depth_bias) {
            discard;
        }
    }

    float alpha = clamp(painterly_sample.a * params.blend_strength, 0.0, 1.0);
    if (alpha <= 0.0) {
        discard;
    }

    float alpha_scale = painterly_sample.a > 0.0001 ? alpha / painterly_sample.a : 0.0;
    out_color = vec4(painterly_sample.rgb * alpha_scale, alpha);
}
