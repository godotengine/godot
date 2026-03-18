#[compute]

#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_color_sampler;
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D u_edge_target;

layout(push_constant, std430) uniform Push {
    vec2 texel_size;
    float edge_intensity;
    float edge_threshold;
} params;

const vec3 LUMA = vec3(0.299, 0.587, 0.114);

vec3 sample_color(vec2 uv, vec2 offset) {
    vec2 sample_uv = clamp(uv + offset * params.texel_size, vec2(0.0), vec2(1.0));
    return texture(u_color_sampler, sample_uv).rgb;
}

void main() {
    ivec2 target_size = imageSize(u_edge_target);
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);

    if (pixel.x >= target_size.x || pixel.y >= target_size.y) {
        return;
    }

    vec2 uv = (vec2(pixel) + vec2(0.5)) * params.texel_size;

    vec3 c00 = sample_color(uv, vec2(-1.0, -1.0));
    vec3 c10 = sample_color(uv, vec2(0.0, -1.0));
    vec3 c20 = sample_color(uv, vec2(1.0, -1.0));
    vec3 c01 = sample_color(uv, vec2(-1.0, 0.0));
    vec3 c11 = sample_color(uv, vec2(0.0, 0.0));
    vec3 c21 = sample_color(uv, vec2(1.0, 0.0));
    vec3 c02 = sample_color(uv, vec2(-1.0, 1.0));
    vec3 c12 = sample_color(uv, vec2(0.0, 1.0));
    vec3 c22 = sample_color(uv, vec2(1.0, 1.0));

    float tl = dot(c00, LUMA);
    float tc = dot(c10, LUMA);
    float tr = dot(c20, LUMA);
    float ml = dot(c01, LUMA);
    float mc = dot(c11, LUMA);
    float mr = dot(c21, LUMA);
    float bl = dot(c02, LUMA);
    float bc = dot(c12, LUMA);
    float br = dot(c22, LUMA);

    float gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    float gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

    vec2 gradient = vec2(gx, gy);
    float magnitude = length(gradient);
    float scaled = clamp(magnitude * params.edge_intensity, 0.0, 4.0);
    float edge_strength = clamp(scaled, 0.0, 1.0);
    float edge_mask = smoothstep(params.edge_threshold, 1.0, edge_strength);

    vec2 direction = magnitude > 1e-5 ? normalize(gradient) : vec2(0.0);
    vec2 encoded_direction = direction * 0.5 + 0.5;

    imageStore(u_edge_target, pixel, vec4(edge_mask, edge_strength, encoded_direction));
}
