#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(constant_id = 0) const bool USE_TEXTURE = true;
layout(constant_id = 1) const int NUM_LIGHTS = 4;
layout(constant_id = 2) const float AMBIENT_STRENGTH = 0.1;

layout(set = 0, binding = 0) uniform sampler2D albedo_tex;

layout(set = 1, binding = 0) uniform LightArray {
    vec4 positions[16];
    vec4 colors[16];
} lights;

layout(push_constant) uniform PushConstants {
    uint base_index;
    float time;
} pc;

void main() {
    vec4 base_color;
    if (USE_TEXTURE) {
        base_color = texture(albedo_tex, v_uv);
    } else {
        base_color = vec4(1.0, 1.0, 1.0, 1.0);
    }

    vec3 lighting = vec3(AMBIENT_STRENGTH);
    for (int i = 0; i < NUM_LIGHTS; i++) {
        vec3 light_dir = normalize(lights.positions[i].xyz);
        lighting += lights.colors[i].rgb * max(dot(light_dir, vec3(0.0, 1.0, 0.0)), 0.0);
    }

    frag_color = vec4(base_color.rgb * lighting, base_color.a);
}
