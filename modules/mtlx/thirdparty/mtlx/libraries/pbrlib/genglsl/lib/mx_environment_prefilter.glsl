#include "mx_microfacet_specular.glsl"

vec3 mx_environment_radiance(vec3 N, vec3 V, vec3 X, vec2 alpha, int distribution, FresnelData fd)
{
    N = mx_forward_facing_normal(N, V);
    vec3 L = fd.refraction ? mx_refraction_solid_sphere(-V, N, fd.ior.x) : -reflect(V, N);

    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);

    float avgAlpha = mx_average_alpha(alpha);
    vec3 F = mx_compute_fresnel(NdotV, fd);
    float G = mx_ggx_smith_G2(NdotV, NdotV, avgAlpha);
    vec3 FG = fd.refraction ? vec3(1.0) - (F * G) : F * G;

    vec3 Li = mx_latlong_map_lookup(L, $envMatrix, mx_latlong_alpha_to_lod(avgAlpha), $envRadiance);
    return Li * FG;
}

vec3 mx_environment_irradiance(vec3 N)
{
    return mx_latlong_map_lookup(N, $envMatrix, 0.0, $envIrradiance);
}
