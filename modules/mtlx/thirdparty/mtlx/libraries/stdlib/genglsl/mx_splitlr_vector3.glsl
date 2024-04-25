#include "mx_aastep.glsl"

void mx_splitlr_vector3(vec3 valuel, vec3 valuer, float center, vec2 texcoord, out vec3 result)
{
    result = mix(valuel, valuer, mx_aastep(center, texcoord.x));
}
