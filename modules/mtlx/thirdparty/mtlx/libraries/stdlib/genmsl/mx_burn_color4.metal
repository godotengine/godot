#include "mx_burn_float.metal"

void mx_burn_color4(vec4 fg, vec4 bg, float mixval, out vec4 result)
{
    float f;
    mx_burn_float(fg.x, bg.x, mixval, f); result.x = f;
    mx_burn_float(fg.y, bg.y, mixval, f); result.y = f;
    mx_burn_float(fg.z, bg.z, mixval, f); result.z = f;
    mx_burn_float(fg.w, bg.w, mixval, f); result.w = f;
}
