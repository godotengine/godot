#include "lib/mx_noise.glsl"

void mx_noise3d_vector4(vec4 amplitude, float pivot, vec3 position, out vec4 result)
{
    vec3 xyz = mx_perlin_noise_vec3(position);
    float w = mx_perlin_noise_float(position + vec3(19, 73, 29));
    result = vec4(xyz, w) * amplitude + pivot;
}
