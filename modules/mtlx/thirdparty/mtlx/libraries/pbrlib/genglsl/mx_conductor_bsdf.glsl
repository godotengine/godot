#include "lib/mx_microfacet_specular.glsl"

void mx_conductor_bsdf_reflection(vec3 L, vec3 V, vec3 P, float occlusion, float weight, vec3 ior_n, vec3 ior_k, vec2 roughness, vec3 N, vec3 X, int distribution, inout BSDF bsdf)
{
    bsdf.throughput = vec3(0.0);

    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    N = mx_forward_facing_normal(N, V);

    X = normalize(X - dot(X, N) * N);
    vec3 Y = cross(N, X);
    vec3 H = normalize(L + V);

    float NdotL = clamp(dot(N, L), M_FLOAT_EPS, 1.0);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);
    float VdotH = clamp(dot(V, H), M_FLOAT_EPS, 1.0);

    vec2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);
    vec3 Ht = vec3(dot(H, X), dot(H, Y), dot(H, N));

    FresnelData fd;
    if (bsdf.thickness > 0.0)
        fd = mx_init_fresnel_conductor_airy(ior_n, ior_k, bsdf.thickness, bsdf.ior);
    else
        fd = mx_init_fresnel_conductor(ior_n, ior_k);

    vec3 F = mx_compute_fresnel(VdotH, fd);
    float D = mx_ggx_NDF(Ht, safeAlpha);
    float G = mx_ggx_smith_G2(NdotL, NdotV, avgAlpha);

    vec3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);

    // Note: NdotL is cancelled out
    bsdf.response = D * F * G * comp * occlusion * weight / (4.0 * NdotV);
}

void mx_conductor_bsdf_indirect(vec3 V, float weight, vec3 ior_n, vec3 ior_k, vec2 roughness, vec3 N, vec3 X, int distribution, inout BSDF bsdf)
{
    bsdf.throughput = vec3(0.0);

    if (weight < M_FLOAT_EPS)
    {
        return;
    }

    N = mx_forward_facing_normal(N, V);

    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    FresnelData fd;
    if (bsdf.thickness > 0.0)
        fd = mx_init_fresnel_conductor_airy(ior_n, ior_k, bsdf.thickness, bsdf.ior);
    else
        fd = mx_init_fresnel_conductor(ior_n, ior_k);

    vec3 F = mx_compute_fresnel(NdotV, fd);

    vec2 safeAlpha = clamp(roughness, M_FLOAT_EPS, 1.0);
    float avgAlpha = mx_average_alpha(safeAlpha);
    vec3 comp = mx_ggx_energy_compensation(NdotV, avgAlpha, F);

    vec3 Li = mx_environment_radiance(N, V, X, safeAlpha, distribution, fd);

    bsdf.response = Li * comp * weight;
}
