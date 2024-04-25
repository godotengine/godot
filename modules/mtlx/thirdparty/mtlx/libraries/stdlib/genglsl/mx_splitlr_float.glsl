#include "mx_aastep.glsl"

void mx_splitlr_float(float valuel, float valuer, float center, vec2 texcoord, out float result)
{
    result = mix(valuel, valuer, mx_aastep(center, texcoord.x));
}
