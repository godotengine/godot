#include "lib/mx_noise.glsl"

void mx_worleynoise3d_vector3(vec3 position, float jitter, out vec3 result)
{
    result = mx_worley_noise_vec3(position, jitter, 0);
}
