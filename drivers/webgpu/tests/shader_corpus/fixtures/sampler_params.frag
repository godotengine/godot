#version 450

// One helper samples two different combined samplers; each call must keep its own texture.

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D first_tex;
layout(set = 0, binding = 1) uniform sampler2D second_tex;

vec4 sample_twice(sampler2D tex, vec2 uv) {
    return texture(tex, uv) + texture(tex, uv * 2.0);
}

void main() {
    frag_color = sample_twice(first_tex, v_uv) - sample_twice(second_tex, v_uv);
}
