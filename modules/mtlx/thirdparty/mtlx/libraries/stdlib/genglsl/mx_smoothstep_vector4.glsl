#include "mx_smoothstep_float.glsl"

void mx_smoothstep_vector4(vec4 val, vec4 low, vec4 high, out vec4 result)
{
    mx_smoothstep_float(val.x, low.x, high.x, result.x);
    mx_smoothstep_float(val.y, low.y, high.y, result.y);
    mx_smoothstep_float(val.z, low.z, high.z, result.z);
    mx_smoothstep_float(val.w, low.w, high.w, result.w);
}
