#version 450 core

#extension GL_ARB_gpu_shader_int64: enable
#extension GL_AMD_gpu_shader_half_float: enable
#extension GL_AMD_gpu_shader_int16: enable

layout(binding = 0) uniform Uniforms
{
    uint i;
};

// int16/uint16 in block
layout(std140, binding = 1) uniform Block
{
    i16vec3  i16v;
    uint16_t u16;
} block;

// int16/uint16 for input
layout(location = 0) in flat u16vec3 iu16v;
layout(location = 1) in flat int16_t ii16;

void literal()
{
    const int16_t i16c[3] =
    {
        0x111S,         // Hex
        -2s,            // Dec
        0400s,          // Oct
    };

    const uint16_t u16c[] =
    {
        0xFFFFus,       // Hex
        65535US,        // Dec
        0177777us,      // Oct
    };

    uint16_t u16 = i16c[i] + u16c[i];
}

void operators()
{
    u16vec3  u16v;
    int16_t  i16;
    uint16_t u16;
    int      i;
    uint     u;
    bool     b;

    // Unary
    u16v++;
    i16--;
    ++i16;
    --u16v;

    u16v = ~u16v;

    i16 = +i16;
    u16v = -u16v;

    // Arithmetic
    u16  += i16;
    u16v -= u16v;
    i16  *= i16;
    u16v /= u16v;
    u16v %= i16;

    u16v = u16v + u16v;
    u16  = i16 - u16;
    u16v = u16v * i16;
    i16  = i16 * i16;
    i16  = i16 % i16;

    // Shift
    u16v <<= i16;
    i16  >>= u16v.y;

    i16  = i16 << u16v.z;
    u16v = u16v << i16;

    // Relational
    b = (u16v.x != i16);
    b = (i16 == u16v.x);
    b = (u16v.x > u16v.y);
    b = (i16 < u);
    b = (u16v.y >= u16v.x);
    b = (i16 <= i);

    // Bitwise
    u16v |= i16;
    u16  = i16 | u16;
    i16  &= i16;
    u16v = u16v & u16v;
    u16v ^= i16;
    u16v = u16v ^ i16;
}

void typeCast()
{
    bvec2 bv;
    ivec2 iv;
    uvec2 uv;
    vec2  fv;
    dvec2 dv;

    f16vec2 f16v;
    i64vec2 i64v;
    u64vec2 u64v;
    i16vec2 i16v;
    u16vec2 u16v;

    i16v = i16vec2(bv);   // bool -> int16
    u16v = u16vec2(bv);   // bool -> uint16
    bv   = bvec2(i16v);   // int16  -> bool
    bv   = bvec2(u16v);   // uint16 -> bool

    i16v = i16vec2(iv);   // int -> int16
    u16v = u16vec2(iv);   // int -> uint16
    iv   = i16v;          // int16  -> int
    iv   = ivec2(u16v);   // uint16 -> int

    i16v = i16vec2(uv);   // uint -> int16
    u16v = u16vec2(uv);   // uint -> uint16
    uv   = i16v;          // int16  -> uint
    uv   = u16v;          // uint16 -> uint

    i16v = i16vec2(fv);   // float -> int16
    u16v = u16vec2(fv);   // float -> uint16
    fv   = i16v;          // int16  -> float
    fv   = u16v;          // uint16 -> float

    i16v = i16vec2(dv);   // double -> int16
    u16v = u16vec2(dv);   // double -> uint16
    dv   = i16v;          // int16  -> double
    dv   = u16v;          // uint16 -> double

    i16v = i16vec2(f16v); // float16 -> int16
    u16v = u16vec2(f16v); // float16 -> uint16
    f16v = i16v;          // int16  -> float16
    f16v = u16v;          // uint16 -> float16

    i16v = i16vec2(i64v); // int64 -> int16
    u16v = u16vec2(i64v); // int64 -> uint16
    i64v = i16v;          // int16  -> int64
    i64v = i64vec2(u16v); // uint16 -> int64

    i16v = i16vec2(u64v); // uint64 -> int16
    u16v = u16vec2(u64v); // uint64 -> uint16
    u64v = i16v;          // int16  -> uint64
    u64v = u16v;          // uint16 -> uint64

    i16v = i16vec2(u16v); // uint16 -> int16
    u16v = i16v;          // int16 -> uint16
}

void builtinFuncs()
{
    i16vec2  i16v;
    u16vec3  u16v;
    f16vec3  f16v;
    bvec3    bv;

    int16_t  i16;
    uint16_t u16;

    // abs()
    i16v = abs(i16v);

    // sign()
    i16v  = sign(i16v);

    // min()
    i16v = min(i16v, i16);
    i16v = min(i16v, i16vec2(-1s));
    u16v = min(u16v, u16);
    u16v = min(u16v, u16vec3(0us));

    // max()
    i16v = max(i16v, i16);
    i16v = max(i16v, i16vec2(-1s));
    u16v = max(u16v, u16);
    u16v = max(u16v, u16vec3(0us));

    // clamp()
    i16v = clamp(i16v, -i16, i16);
    i16v = clamp(i16v, -i16v, i16v);
    u16v = clamp(u16v, -u16, u16);
    u16v = clamp(u16v, -u16v, u16v);

    // mix()
    i16  = mix(i16v.x, i16v.y, true);
    i16v = mix(i16vec2(i16), i16vec2(-i16), bvec2(false));
    u16  = mix(u16v.x, u16v.y, true);
    u16v = mix(u16vec3(u16), u16vec3(-u16), bvec3(false));

    // frexp()
    i16vec3 exp;
    f16v = frexp(f16v, exp);

    // ldexp()
    f16v = ldexp(f16v, exp);

    // float16BitsToInt16()
    i16v = float16BitsToInt16(f16v.xy);

    // float16BitsToUint16()
    u16v.x = float16BitsToUint16(f16v.z);

    // int16BitsToFloat16()
    f16v.xy = int16BitsToFloat16(i16v);

    // uint16BitsToFloat16()
    f16v = uint16BitsToFloat16(u16v);

    // packInt2x16()
    int packi = packInt2x16(i16v);

    // unpackInt2x16()
    i16v = unpackInt2x16(packi);

    // packUint2x16()
    uint packu = packUint2x16(u16v.xy);

    // unpackUint2x16()
    u16v.xy = unpackUint2x16(packu);

    // packInt4x16()
    int64_t packi64 = packInt4x16(i16vec4(i16));

    // unpackInt4x16()
    i16v = unpackInt4x16(packi64).xy;

    // packUint4x16()
    uint64_t packu64 = packUint4x16(u16vec4(u16));

    // unpackUint4x16()
    u16v = unpackUint4x16(packu64).xyz;

    // lessThan()
    bv    = lessThan(u16v, u16vec3(u16));
    bv.xy = lessThan(i16v, i16vec2(i16));

    // lessThanEqual()
    bv    = lessThanEqual(u16v, u16vec3(u16));
    bv.xy = lessThanEqual(i16v, i16vec2(i16));

    // greaterThan()
    bv    = greaterThan(u16v, u16vec3(u16));
    bv.xy = greaterThan(i16v, i16vec2(i16));

    // greaterThanEqual()
    bv    = greaterThanEqual(u16v, u16vec3(u16));
    bv.xy = greaterThanEqual(i16v, i16vec2(i16));

    // equal()
    bv    = equal(u16v, u16vec3(u16));
    bv.xy = equal(i16v, i16vec2(i16));

    // notEqual()
    bv    = notEqual(u16v, u16vec3(u16));
    bv.xy = notEqual(i16v, i16vec2(i16));
}

// Type conversion for specialization constant
layout(constant_id = 100) const int64_t  si64 = -10L;
layout(constant_id = 101) const uint64_t su64 = 20UL;
layout(constant_id = 102) const int  si = -5;
layout(constant_id = 103) const uint su = 4;
layout(constant_id = 104) const bool sb = true;
layout(constant_id = 105) const int16_t si16 = -5S;
layout(constant_id = 106) const uint16_t su16 = 4US;

// bool <-> int16/uint16
const bool i16_to_b = bool(si16);
const bool u16_to_b = bool(su16);
const int16_t  b_to_i16 = int16_t(sb);
const uint16_t b_to_u16 = uint16_t(sb);

// int <-> int16/uint16
const int i16_to_i = int(si16);
const int u16_to_i = int(su16);
const int16_t  i_to_i16 = int16_t(si);
const uint16_t i_to_u16 = uint16_t(si);

// uint <-> int16/uint16
const uint i16_to_u = uint(si16);
const uint u16_to_u = uint(su16);
const int16_t  u_to_i16 = int16_t(su);
const uint16_t u_to_u16 = uint16_t(su);

// int64 <-> int16/uint16
const int64_t i16_to_i64 = int64_t(si16);
const int64_t u16_to_i64 = int64_t(su16);
const int16_t  i64_to_i16 = int16_t(si64);
const uint16_t i64_to_u16 = uint16_t(si64);

// uint64 <-> int16/uint16
const uint64_t i16_to_u64 = uint64_t(si16);
const uint64_t u16_to_u64 = uint64_t(su16);
const int16_t  u64_to_i16 = int16_t(su64);
const uint16_t u64_to_u16 = uint16_t(su64);

// int16 <-> uint16
const uint16_t i16_to_u16 = uint16_t(si16);
const int16_t  u16_to_i16 = int16_t(su16);

void main()
{
    literal();
    operators();
    typeCast();
    builtinFuncs();
}
