#include "lib/mx_noise.glsl"

void mx_fractal3d_vector3(vec3 amplitude, int octaves, float lacunarity, float diminish, vec3 position, out vec3 result)
{
    vec3 value = mx_fractal_noise_vec3(position, octaves, lacunarity, diminish);
    result = value * amplitude;
}
