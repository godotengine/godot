#include "lib/mx_noise.glsl"

void mx_noise3d_vector3(vec3 amplitude, float pivot, vec3 position, out vec3 result)
{
    vec3 value = mx_perlin_noise_vec3(position);
    result = value * amplitude + pivot;
}
