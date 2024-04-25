#include "mx_microfacet_specular.glsl"

vec3 mx_environment_radiance(vec3 N, vec3 V, vec3 X, vec2 roughness, int distribution, FresnelData fd)
{
    return vec3(0.0);
}

vec3 mx_environment_irradiance(vec3 N)
{
    return vec3(0.0);
}
