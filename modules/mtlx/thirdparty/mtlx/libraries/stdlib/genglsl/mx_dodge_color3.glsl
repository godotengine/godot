#include "mx_dodge_float.glsl"

void mx_dodge_color3(vec3 fg, vec3 bg, float mixval, out vec3 result)
{
    mx_dodge_float(fg.x, bg.x, mixval, result.x);
    mx_dodge_float(fg.y, bg.y, mixval, result.y);
    mx_dodge_float(fg.z, bg.z, mixval, result.z);
}
