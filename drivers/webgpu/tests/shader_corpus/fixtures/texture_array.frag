#version 450

// Texture array indexed by a constant and by a push constant.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform texture2D layers[4];
layout(set = 0, binding = 1) uniform sampler layer_sampler;

layout(push_constant) uniform PushConstants {
    uint layer;
} pc;

void main() {
    frag_color = texture(sampler2D(layers[0], layer_sampler), v_uv) + texture(sampler2D(layers[pc.layer], layer_sampler), v_uv);
}
