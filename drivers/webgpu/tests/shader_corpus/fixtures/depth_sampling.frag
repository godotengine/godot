#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2DShadow shadow_map;
layout(set = 0, binding = 1) uniform sampler2D depth_texture;

layout(set = 1, binding = 0) uniform LightData {
    mat4 light_vp;
    vec4 light_dir;
    float bias;
    float pad0;
    float pad1;
    float pad2;
} light;

void main() {
    vec4 light_space = light.light_vp * vec4(v_uv, 0.0, 1.0);
    vec3 proj = light_space.xyz / light_space.w;
    proj.xy = proj.xy * 0.5 + 0.5;

    float shadow = texture(shadow_map, vec3(proj.xy, proj.z - light.bias));
    float depth = texture(depth_texture, v_uv).r;

    frag_color = vec4(vec3(shadow * depth), 1.0);
}
