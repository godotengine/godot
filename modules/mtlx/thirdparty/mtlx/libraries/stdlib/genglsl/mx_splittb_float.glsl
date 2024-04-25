#include "mx_aastep.glsl"

void mx_splittb_float(float valuet, float valueb, float center, vec2 texcoord, out float result)
{
    result = mix(valuet, valueb, mx_aastep(center, texcoord.y));
}
