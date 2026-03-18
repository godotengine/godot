#[compute]

#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_color_sampler;
layout(set = 0, binding = 1) uniform sampler2D u_edge_sampler;
layout(set = 0, binding = 2, rgba8) uniform writeonly image2D u_stylized_target;

layout(push_constant, std430) uniform Push {
    float stroke_opacity;
    float edge_strength;
    float stroke_length;
    float gamma;
} params;

vec3 fetch_color(vec2 uv) {
    return texture(u_color_sampler, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
}

void main() {
    ivec2 target_size = imageSize(u_stylized_target);
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (pixel.x >= target_size.x || pixel.y >= target_size.y) {
        return;
    }

    vec2 uv = (vec2(pixel) + vec2(0.5)) / vec2(target_size);

    vec4 color_sample = texture(u_color_sampler, uv);
    vec4 edge_sample = texture(u_edge_sampler, uv);

    float edge_mask = edge_sample.x;
    float gradient = edge_sample.y;
    vec2 direction_sample = edge_sample.zw * 2.0 - 1.0;
    vec2 edge_direction = length(direction_sample) > 1e-4 ? normalize(direction_sample) : vec2(0.0, 1.0);

    float opacity = clamp(params.stroke_opacity, 0.0, 1.0);
    float strength = max(params.edge_strength, 0.0);
    float length_scale = max(params.stroke_length, 0.001);

    vec3 accum = color_sample.rgb;
    float weight = 1.0;

    if (opacity > 0.001) {
        vec2 tangent = vec2(-edge_direction.y, edge_direction.x);
        float pixel_scale = length_scale / float(max(target_size.x, target_size.y));

        for (int i = 1; i <= 3; i++) {
            float f = float(i);
            float falloff = exp(-f * 0.75) * opacity;

            vec2 offset = tangent * pixel_scale * f;
            vec3 sample_forward = fetch_color(uv + offset);
            vec3 sample_backward = fetch_color(uv - offset);

            accum += (sample_forward + sample_backward) * falloff;
            weight += 2.0 * falloff;
        }

        accum /= max(weight, 1e-4);
    }

    float edge_darkening = clamp(pow(gradient, strength), 0.0, 1.0) * edge_mask;
    vec3 ink = mix(accum, accum * (1.0 - edge_darkening), edge_mask);

    vec3 final_color = pow(clamp(ink, 0.0, 1.0), vec3(1.0 / max(params.gamma, 0.001)));

    imageStore(u_stylized_target, pixel, vec4(final_color * color_sample.a, color_sample.a));
}
