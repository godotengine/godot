#version 300 es

int imax, imin;
uint umax, umin;

vec3 x, y;    // ERROR, needs default precision
bvec3 bv;

uint uy;
uvec2 uv2c;
uvec2 uv2y;
uvec2 uv2x;
uvec4 uv4y;

ivec3 iv3a;
ivec3 iv3b;

ivec4 iv4a;
ivec4 iv4b;

float f;

vec2 v2a, v2b;
vec4 v4;

void main()
{
    // 1.3 int
    vec3 v = mix(x, y, bv);
    ivec4 iv10 = abs(iv4a);
    ivec4 iv11 = sign(iv4a);
    ivec4 iv12 = min(iv4a, iv4b);
    ivec4 iv13 = min(iv4a, imin);
    uvec2 u = min(uv2x, uv2y);
    uvec4 uv = min(uv4y, uy);
    ivec3 iv14 = max(iv3a, iv3b);
    ivec4 iv15 = max(iv4a, imax);
    uvec2 u10 = max(uv2x, uv2y);
    uvec2 u11 = max(uv2x, uy);
    ivec4 iv16 = clamp(iv4a, iv4a, iv4b);
    ivec4 iv17 = clamp(iv4a, imin, imax);
    uvec2 u12 = clamp(uv2x, uv2y, uv2c);
    uvec4 uv10 = clamp(uv4y, umin, umax);

    // 1.3 float
    vec3 modfOut;
    vec3 v11 = modf(x, modfOut);

    float t = trunc(f);
    vec2 v12 = round(v2a);
    vec2 v13 = roundEven(v2a);
    bvec2 b10 = isnan(v2a);
    bvec4 b11 = isinf(v4);

    // 3.3 float
    int i = floatBitsToInt(f);
    uvec4 uv11 = floatBitsToUint(v4);
    vec4 v14 = intBitsToFloat(iv4a);
    vec2 v15 = uintBitsToFloat(uv2c);

    // 4.0  pack
    uint u19 = packSnorm2x16(v2a);
    vec2 v20 = unpackSnorm2x16(uy);
    uint u15 = packUnorm2x16(v2a);
    vec2 v16 = unpackUnorm2x16(uy);
    uint u17 = packHalf2x16(v2b);
    vec2 v18 = unpackHalf2x16(uy);

    // not present
    noise2(v18);  // ERROR, not present

    float t__;  // ERROR, no __ until revision 310

      // ERROR, no __ until revision 310
    #define __D
}
