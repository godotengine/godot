#version 450

// Fragment stage: defines @id(1) and @id(2) but NOT @id(0).
// Vertex stage defines @id(0). Per-stage filtering must route
// each constant only to the stage that declares it.

layout(constant_id = 1) const uint FRAG_MODE = 0u;
layout(constant_id = 2) const float BRIGHTNESS = 1.0;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in float v_scale;

layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
} pc;

void main() {
    vec4 base = texture(tex, v_uv);

    if (FRAG_MODE == 0u) {
        // Normal mode
        frag_color = vec4(base.rgb * BRIGHTNESS, base.a);
    } else if (FRAG_MODE == 1u) {
        // Grayscale mode
        float gray = dot(base.rgb, vec3(0.299, 0.587, 0.114));
        frag_color = vec4(vec3(gray * BRIGHTNESS), base.a);
    } else {
        // Debug: show UV
        frag_color = vec4(v_uv, v_scale * 0.5, 1.0);
    }
}
