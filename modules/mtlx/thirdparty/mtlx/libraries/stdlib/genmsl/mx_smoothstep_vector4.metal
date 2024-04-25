#include "mx_smoothstep_float.metal"

void mx_smoothstep_vector4(vec4 val, vec4 low, vec4 high, out vec4 result)
{
    float f;
    mx_smoothstep_float(val.x, low.x, high.x, f); result.x = f;
    mx_smoothstep_float(val.y, low.y, high.y, f); result.y = f;
    mx_smoothstep_float(val.z, low.z, high.z, f); result.z = f;
    mx_smoothstep_float(val.w, low.w, high.w, f); result.w = f;
}
