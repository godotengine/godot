#include "mx_smoothstep_float.glsl"

void mx_smoothstep_vector3(vec3 val, vec3 low, vec3 high, out vec3 result)
{
    mx_smoothstep_float(val.x, low.x, high.x, result.x);
    mx_smoothstep_float(val.y, low.y, high.y, result.y);
    mx_smoothstep_float(val.z, low.z, high.z, result.z);
}
