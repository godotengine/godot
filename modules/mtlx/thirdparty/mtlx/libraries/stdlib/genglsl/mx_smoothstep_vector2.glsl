#include "mx_smoothstep_float.glsl"

void mx_smoothstep_vector2(vec2 val, vec2 low, vec2 high, out vec2 result)
{
    mx_smoothstep_float(val.x, low.x, high.x, result.x);
    mx_smoothstep_float(val.y, low.y, high.y, result.y);
}
