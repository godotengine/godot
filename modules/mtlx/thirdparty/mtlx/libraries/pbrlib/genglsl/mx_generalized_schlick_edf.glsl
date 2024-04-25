#include "lib/mx_microfacet.glsl"

void mx_generalized_schlick_edf(vec3 N, vec3 V, vec3 color0, vec3 color90, float exponent, EDF base, out EDF result)
{
    N = mx_forward_facing_normal(N, V);
    float NdotV = clamp(dot(N, V), M_FLOAT_EPS, 1.0);
    vec3 f = mx_fresnel_schlick(NdotV, color0, color90, exponent);
    result = base * f;
}
