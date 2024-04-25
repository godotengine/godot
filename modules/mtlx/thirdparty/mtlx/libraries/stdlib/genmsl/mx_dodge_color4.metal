#include "mx_dodge_float.metal"

void mx_dodge_color4(vec4 fg , vec4 bg , float mixval, out vec4 result)
{
    float f;
    mx_dodge_float(fg.x, bg.x, mixval, f); result.x = f;
    mx_dodge_float(fg.y, bg.y, mixval, f); result.y = f;
    mx_dodge_float(fg.z, bg.z, mixval, f); result.z = f;
    mx_dodge_float(fg.w, bg.w, mixval, f); result.w = f;
}
