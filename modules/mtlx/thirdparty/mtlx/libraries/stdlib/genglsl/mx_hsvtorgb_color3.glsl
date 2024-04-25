#include "lib/mx_hsv.glsl"

void mx_hsvtorgb_color3(vec3 _in, out vec3 result)
{
    result = mx_hsvtorgb(_in);
}
