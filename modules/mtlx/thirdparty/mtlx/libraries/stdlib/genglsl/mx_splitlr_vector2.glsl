#include "mx_aastep.glsl"

void mx_splitlr_vector2(vec2 valuel, vec2 valuer, float center, vec2 texcoord, out vec2 result)
{
    result = mix(valuel, valuer, mx_aastep(center, texcoord.x));
}
