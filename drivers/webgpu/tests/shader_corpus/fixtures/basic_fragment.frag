#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_world_pos;

layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view_projection;
    vec4 camera_pos;
} camera;

layout(set = 1, binding = 0) uniform sampler2D albedo_tex;
layout(set = 1, binding = 1) uniform sampler2D normal_tex;

layout(push_constant) uniform PushConstants {
    mat4 model_matrix;
    uint instance_index;
    float time;
} pc;

void main() {
    vec4 albedo = texture(albedo_tex, v_uv);
    vec3 N = normalize(v_normal);
    vec3 L = normalize(vec3(1.0, 1.0, 0.5));
    float NdotL = max(dot(N, L), 0.0);
    frag_color = vec4(albedo.rgb * (0.2 + 0.8 * NdotL), albedo.a);
}
