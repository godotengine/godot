#include "mx_burn_float.glsl"

void mx_burn_color4(vec4 fg, vec4 bg, float mixval, out vec4 result)
{
    mx_burn_float(fg.x, bg.x, mixval, result.x);
    mx_burn_float(fg.y, bg.y, mixval, result.y);
    mx_burn_float(fg.z, bg.z, mixval, result.z);
    mx_burn_float(fg.w, bg.w, mixval, result.w);
}
