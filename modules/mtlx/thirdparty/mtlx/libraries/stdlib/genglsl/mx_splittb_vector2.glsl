#include "mx_aastep.glsl"

void mx_splittb_vector2(vec2 valuet, vec2 valueb, float center, vec2 texcoord, out vec2 result)
{
    result = mix(valuet, valueb, mx_aastep(center, texcoord.y));
}
