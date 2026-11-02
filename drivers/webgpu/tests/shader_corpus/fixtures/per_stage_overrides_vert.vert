#version 450

// Vertex stage: only defines spec constant @id(0).
// Fragment stage defines @id(1) and @id(2) but NOT @id(0).
// This tests that per-stage override filtering passes the right
// constants to the right stages — WebGPU rejects unknown IDs.

layout(constant_id = 0) const uint VERTEX_FLAGS = 0u;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out float v_scale;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} pc;

void main() {
    v_uv = uv;
    // Use the spec constant: bit 0 controls scale
    v_scale = ((VERTEX_FLAGS & 1u) != 0u) ? 2.0 : 1.0;
    gl_Position = pc.mvp * vec4(position * v_scale, 1.0);
}
