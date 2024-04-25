#include "lib/mx_noise.glsl"

void mx_noise2d_vector2(vec2 amplitude, float pivot, vec2 texcoord, out vec2 result)
{
    vec3 value = mx_perlin_noise_vec3(texcoord);
    result = value.xy * amplitude + pivot;
}
