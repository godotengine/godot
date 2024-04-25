#include "mx_burn_float.metal"

void mx_burn_color3(vec3 fg, vec3 bg, float mixval, out vec3 result)
{
    float f;
    mx_burn_float(fg.x, bg.x, mixval, f); result.x = f;
    mx_burn_float(fg.y, bg.y, mixval, f); result.y = f;
    mx_burn_float(fg.z, bg.z, mixval, f); result.z = f;
}
