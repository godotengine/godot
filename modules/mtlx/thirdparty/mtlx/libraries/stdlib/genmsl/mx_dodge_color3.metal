#include "mx_dodge_float.metal"

void mx_dodge_color3(vec3 fg, vec3 bg, float mixval, out vec3 result)
{
    float f;
    mx_dodge_float(fg.x, bg.x, mixval, f); result.x = f;
    mx_dodge_float(fg.y, bg.y, mixval, f); result.y = f;
    mx_dodge_float(fg.z, bg.z, mixval, f); result.z = f;
}
