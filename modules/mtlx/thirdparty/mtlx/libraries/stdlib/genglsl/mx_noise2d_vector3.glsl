#include "lib/mx_noise.glsl"

void mx_noise2d_vector3(vec3 amplitude, float pivot, vec2 texcoord, out vec3 result)
{
    vec3 value = mx_perlin_noise_vec3(texcoord);
    result = value * amplitude + pivot;
}
