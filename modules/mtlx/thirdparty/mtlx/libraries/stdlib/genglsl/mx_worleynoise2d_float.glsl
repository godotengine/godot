#include "lib/mx_noise.glsl"

void mx_worleynoise2d_float(vec2 texcoord, float jitter, out float result)
{
    result = mx_worley_noise_float(texcoord, jitter, 0);
}
