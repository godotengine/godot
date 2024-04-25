#include "lib/mx_microfacet_sheen.glsl"

void mx_sheen_bsdf_reflection(vec3 L, vec3 V, vec3 P, float occlusion, float weight, vec3 color, float roughness, vec3 N, inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    N = mx_forward_facing_normal(N, V);

    vec3 H = normalize(L + V);

    float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);
    float NdotH = clamp(dot(N, H), M_FLOAT_EPS, 1.0);

    vec3 fr = color * mx_imageworks_sheen_brdf(NdotL, NdotV, NdotH, roughness);
    float dirAlbedo = mx_imageworks_sheen_dir_albedo(NdotV, roughness);
    bsdf.throughput = vec3(1.0 - dirAlbedo * weight);

    // We need to include NdotL from the light integral here
    // as in this case it's not cancelled out by the BRDF denominator.
    bsdf.response = fr * NdotL * occlusion * weight;
}

void mx_sheen_bsdf_indirect(vec3 V, float weight, vec3 color, float roughness, vec3 N, inout BSDF bsdf)
{
    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    N = mx_forward_facing_normal(N, V);

    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    float dirAlbedo = mx_imageworks_sheen_dir_albedo(NdotV, roughness);
    bsdf.throughput = vec3(1.0 - dirAlbedo * weight);

    vec3 Li = mx_environment_irradiance(N);
    bsdf.response = Li * color * dirAlbedo * weight;
}
