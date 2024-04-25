#include "lib/mx_noise.glsl"

void mx_worleynoise3d_vector2(vec3 position, float jitter, out vec2 result)
{
    result = mx_worley_noise_vec2(position, jitter, 0);
}
