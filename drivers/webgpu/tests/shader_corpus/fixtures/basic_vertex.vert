#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_normal;
layout(location = 2) out vec3 v_world_pos;

layout(push_constant) uniform PushConstants {
    mat4 model_matrix;
    uint instance_index;
    float time;
} pc;

layout(set = 0, binding = 0) uniform CameraData {
    mat4 view_projection;
    vec4 camera_pos;
} camera;

void main() {
    vec4 world = pc.model_matrix * vec4(position, 1.0);
    v_world_pos = world.xyz;
    v_uv = uv;
    v_normal = mat3(pc.model_matrix) * normal;
    gl_Position = camera.view_projection * world;
}
