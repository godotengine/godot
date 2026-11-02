#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_normal;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D tex_albedo;
layout(set = 0, binding = 1) uniform sampler2D tex_roughness;
layout(set = 0, binding = 2) uniform sampler2D tex_metallic;
layout(set = 0, binding = 3) uniform sampler2D tex_normal;
layout(set = 0, binding = 4) uniform sampler2D tex_emission;
layout(set = 0, binding = 5) uniform samplerCube env_map;

layout(set = 1, binding = 0) uniform MaterialParams {
    vec4 albedo_color;
    float roughness_factor;
    float metallic_factor;
    float emission_strength;
    float pad;
} material;

layout(push_constant) uniform PushConstants {
    uint flags;
    float time;
} pc;

void main() {
    vec4 albedo = texture(tex_albedo, v_uv) * material.albedo_color;
    float roughness = texture(tex_roughness, v_uv).r * material.roughness_factor;
    float metallic = texture(tex_metallic, v_uv).r * material.metallic_factor;
    vec3 emission = texture(tex_emission, v_uv).rgb * material.emission_strength;

    vec3 N = normalize(v_normal);
    vec3 R = reflect(-N, N);
    vec3 env = texture(env_map, R).rgb;

    vec3 result = mix(albedo.rgb, env, metallic) + emission;
    frag_color = vec4(result, albedo.a);
}
