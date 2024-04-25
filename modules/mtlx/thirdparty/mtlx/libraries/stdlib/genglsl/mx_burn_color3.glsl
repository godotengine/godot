#include "mx_burn_float.glsl"

void mx_burn_color3(vec3 fg, vec3 bg, float mixval, out vec3 result)
{
    mx_burn_float(fg.x, bg.x, mixval, result.x);
    mx_burn_float(fg.y, bg.y, mixval, result.y);
    mx_burn_float(fg.z, bg.z, mixval, result.z);
}
