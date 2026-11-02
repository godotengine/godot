#version 450

// Array of combined samplers with a following binding that must not collide.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D detail[2];
layout(set = 0, binding = 1) uniform sampler2D albedo;

void main() {
    frag_color = texture(detail[0], v_uv) * texture(albedo, v_uv);
}
