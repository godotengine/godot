#version 450

#extension GL_ARB_gpu_shader_int64: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require

layout(binding = 0) uniform Uniforms
{
    uint index;
};

layout(std140, binding = 1) uniform Block
{
    i64vec3  i64v;
    uint64_t u64;
} block;

void main()
{
}

void literal()
{
    const int64_t i64Const[3] =
    {
        -0x1111111111111111l,   // Hex
        -1l,                    // Dec
        040000000000l,          // Oct
    };

    int64_t i64 = i64Const[index];

    const uint64_t u64Const[] =
    {
        0xFFFFFFFFFFFFFFFFul,   // Hex
        4294967296UL,           // Dec
        077777777777ul,         // Oct
    };

    uint64_t u64 = u64Const[index];
}

void typeCast()
{
    bvec2 bv;
    ivec2 iv;
    uvec2 uv;
    vec2  fv;
    dvec2 dv;

    i64vec2 i64v;
    u64vec2 u64v;

    i64v = i64vec2(bv);   // bool -> int64
    u64v = u64vec2(bv);   // bool -> uint64

    i64v = iv;            // int   -> int64
    iv = ivec2(i64v);     // int64 -> int

    u64v = uv;            // uint   -> uint64
    uv = uvec2(u64v);     // uint64 -> uint

    fv = vec2(i64v);      // int64 -> float
    dv = i64v;            // int64 -> double

    fv = vec2(u64v);      // uint64 -> float
    dv = u64v;            // uint64 -> double

    i64v = i64vec2(fv);   // float  -> int64
    i64v = i64vec2(dv);   // double -> int64

    u64v = u64vec2(fv);   // float  -> uint64
    u64v = u64vec2(dv);   // double -> uint64

    bv = bvec2(i64v);     // int64  -> bool
    bv = bvec2(u64v);     // uint64 -> bool

    u64v = i64v;          // int64  -> uint64
    i64v = i64vec2(u64v); // uint64 -> int64

    uv = uvec2(i64v);     // int64 -> uint
    i64v = i64vec2(uv);   // uint -> int64
    iv = ivec2(u64v);     // uint64 -> int
    u64v = iv;            // int -> uint64
}

void operators()
{
    u64vec3 u64v;
    int64_t i64;
    uvec3   uv;
    int     i;
    bool    b;

    // Unary
    u64v++;
    i64--;
    ++i64;
    --u64v;

    u64v = ~u64v;

    i64 = +i64;
    u64v = -u64v;

    // Arithmetic
    i64  += i64;
    u64v -= u64v;
    i64  *= i;
    u64v /= uv;
    u64v %= i;

    u64v = u64v + uv;
    i64  = i64 - i;
    u64v = u64v * uv;
    i64  = i64 * i;
    i64  = i64 % i;

    // Shift
    u64v = u64v << i;
    i64 = i64 >> uv.y;
    u64v <<= i;
    i64  >>= uv.y;

    i64  = i64 << u64v.z;
    u64v = u64v << i64;

    // Relational
    b = (u64v.x != i64);
    b = (i64 == u64v.x);
    b = (u64v.x > uv.y);
    b = (i64 < i);
    b = (u64v.y >= uv.x);
    b = (i64 <= i);

    // Bitwise
    u64v |= i;
    i64  = i64 | i;
    i64  &= i;
    u64v = u64v & uv;
    u64v ^= i64;
    u64v = u64v ^ i64;
}

void builtinFuncs()
{
    i64vec2  i64v;
    u64vec3  u64v;
    dvec3    dv;
    bvec3    bv;

    int64_t  i64;
    uint64_t u64;

    // abs()
    i64v = abs(i64v);

    // sign()
    i64  = sign(i64);

    // min()
    i64v = min(i64v, i64);
    i64v = min(i64v, i64vec2(-1));
    u64v = min(u64v, u64);
    u64v = min(u64v, u64vec3(0));

    // max()
    i64v = max(i64v, i64);
    i64v = max(i64v, i64vec2(-1));
    u64v = max(u64v, u64);
    u64v = max(u64v, u64vec3(0));

    // clamp()
    i64v = clamp(i64v, -i64, i64);
    i64v = clamp(i64v, -i64v, i64v);
    u64v = clamp(u64v, -u64, u64);
    u64v = clamp(u64v, -u64v, u64v);

    // mix()
    i64  = mix(i64v.x, i64v.y, true);
    i64v = mix(i64vec2(i64), i64vec2(-i64), bvec2(false));
    u64  = mix(u64v.x, u64v.y, true);
    u64v = mix(u64vec3(u64), u64vec3(-u64), bvec3(false));

    // doubleBitsToInt64()
    i64v = doubleBitsToInt64(dv.xy);

    // doubleBitsToUint64()
    u64v.x = doubleBitsToUint64(dv.z);

    // int64BitsToDouble()
    dv.xy = int64BitsToDouble(i64v);

    // uint64BitsToDouble()
    dv = uint64BitsToDouble(u64v);

    // packInt2x32()
    i64 = packInt2x32(ivec2(1, 2));

    // unpackInt2x32()
    ivec2 iv = unpackInt2x32(i64);

    // packUint2x32()
    u64 = packUint2x32(uvec2(2, 3));

    // unpackUint2x32()
    uvec2 uv = unpackUint2x32(u64);

    // lessThan()
    bv    = lessThan(u64v, u64vec3(u64));
    bv.xy = lessThan(i64v, i64vec2(i64));

    // lessThanEqual()
    bv    = lessThanEqual(u64v, u64vec3(u64));
    bv.xy = lessThanEqual(i64v, i64vec2(i64));

    // greaterThan()
    bv    = greaterThan(u64v, u64vec3(u64));
    bv.xy = greaterThan(i64v, i64vec2(i64));

    // greaterThanEqual()
    bv    = greaterThanEqual(u64v, u64vec3(u64));
    bv.xy = greaterThanEqual(i64v, i64vec2(i64));

    // equal()
    bv    = equal(u64v, u64vec3(u64));
    bv.xy = equal(i64v, i64vec2(i64));

    // notEqual()
    bv    = notEqual(u64v, u64vec3(u64));
    bv.xy = notEqual(i64v, i64vec2(i64));
}

// Type conversion for specialization constant
layout(constant_id = 100) const int64_t  si64 = -10L;
layout(constant_id = 101) const uint64_t su64 = 20UL;
layout(constant_id = 102) const int  si = -5;
layout(constant_id = 103) const uint su = 4;
layout(constant_id = 104) const bool sb = true;
layout(constant_id = 105) const uint64_t su64inc = su64 + 1UL;

// bool <-> int64/uint64
const bool i64_to_b = bool(si64);
const bool u64_to_b = bool(su64);
const int64_t  b_to_i64 = int64_t(sb);
const uint64_t b_to_u64 = uint64_t(sb);

// int <-> int64
const int     i64_to_i = int(si64);
const int64_t i_to_i64 = int64_t(si);

// uint <-> uint64
const uint     u64_to_u = uint(su64);
const uint64_t u_to_u64 = uint64_t(su);

// int64 <-> uint64
const int64_t  u64_to_i64 = int64_t(su64);
const uint64_t i64_to_u64 = uint64_t(si64);

// int <-> uint64
const int      u64_to_i = int(su64);
const uint64_t i_to_u64 = uint64_t(si);

// uint <-> int64
const uint    i64_to_u = uint(si64);
const int64_t u_to_i64 = int64_t(su);

#define UINT64_MAX  18446744073709551615ul
uint64_t u64Max = UINT64_MAX;
