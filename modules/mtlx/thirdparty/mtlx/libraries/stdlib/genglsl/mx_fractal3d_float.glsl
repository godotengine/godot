#include "lib/mx_noise.glsl"

void mx_fractal3d_float(float amplitude, int octaves, float lacunarity, float diminish, vec3 position, out float result)
{
    float value = mx_fractal_noise_float(position, octaves, lacunarity, diminish);
    result = value * amplitude;
}
