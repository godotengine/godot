#version 450

#extension GL_EXT_shader_explicit_arithmetic_types: enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_int32: require
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float32: require
#extension GL_EXT_shader_explicit_arithmetic_types_float64: require

layout(binding = 0) uniform Uniforms
{
    uint index;
};

layout(std140, binding = 1) uniform Block
{
    int32_t   i32;
    i32vec2   i32v2;
    i32vec3   i32v3;
    i32vec4   i32v4;
    uint32_t  u32;
    u32vec2   u32v2;
    u32vec3   u32v3;
    u32vec4   u32v4;
} block;

void main()
{
}

void literal()
{

    const int32_t i32Const[3] =
    {
        -0x11111111,           // Hex
        -1,                    // Dec
        04000000000,           // Oct
    };

    int32_t i32 = i32Const[index];

    const uint32_t u32Const[] =
    {
        0xFFFFFFFF,             // Hex
        4294967295,             // Dec
        017777777777,           // Oct
    };

    uint32_t u32 = u32Const[index];
}

void typeCast32()
{
    i8vec2 i8v;
    u8vec2 u8v;
    i16vec2 i16v;
    u16vec2 u16v;
    i32vec2 i32v;
    u32vec2 u32v;
    i64vec2 i64v;
    u64vec2 u64v;
    f16vec2 f16v;
    f32vec2 f32v;
    f64vec2 f64v;
    bvec2   bv;

    u32v = i32v;     // int32_t  ->  uint32_t
    i64v = i32v;     // int32_t  ->   int64_t
    u64v = i32v;     // int32_t  ->  uint64_t
    i64v = u32v;     // uint32_t ->   int64_t
    u64v = u32v;     // uint32_t ->  uint64_t
    f32v = i32v;     // int32_t  ->  float32_t
    f64v = i32v;     // int32_t  ->  float64_t
    f32v = u32v;     // uint32_t ->  float32_t
    f64v = u32v;     // uint32_t ->  float64_t

    i8v =  i8vec2(i32v);       // int32_t   ->   int8_t
    i8v =  i8vec2(u32v);       // uint32_t  ->   int8_t
    i16v = i16vec2(i32v);      // int32_t   ->   int16_t
    i16v = i16vec2(u32v);      // uint32_t  ->   int16_t
    i32v = i32vec2(i32v);      // int32_t   ->   int32_t
    i32v = i32vec2(u32v);      // uint32_t  ->   int32_t
    i64v = i64vec2(i32v);      // int32_t   ->   int64_t
	i64v = i64vec2(u32v);      // uint32_t  ->   int64_t
	u8v =  u8vec2(i32v);       // int32_t   ->   uint8_t
    u8v =  u8vec2(u32v);       // uint32_t  ->   uint8_t
    u16v = u16vec2(i32v);      // int32_t   ->   uint16_t
    u16v = u16vec2(u32v);      // uint32_t  ->   uint16_t
    u32v = u32vec2(i32v);      // int32_t   ->   uint32_t
    u32v = u32vec2(u32v);      // uint32_t  ->   uint32_t
    u64v = u64vec2(i32v);      // int32_t   ->   uint64_t
    u64v = u64vec2(u32v);      // uint32_t  ->   uint64_t

    f16v = f16vec2(i32v);      // int32_t   ->  float16_t
    f32v = f32vec2(i32v);      // int32_t   ->  float32_t
    f64v = f64vec2(i32v);      // int32_t   ->  float64_t
    f16v = f16vec2(u32v);      // uint32_t  ->  float16_t
    f32v = f32vec2(u32v);      // uint32_t  ->  float32_t
    f64v = f64vec2(u32v);      // uint32_t  ->  float64_t

    i32v = i32vec2(bv);       // bool     ->   int32
    u32v = u32vec2(bv);       // bool     ->   uint32
    bv   = bvec2(i32v);       // int32    ->   bool
    bv   = bvec2(u32v);       // uint32   ->   bool
}

void operators()
{
    u32vec3 u32v;
    int32_t i32;
    uvec3   uv;
    int32_t i;
    int64_t i64;
    bool    b;

    // Unary
    u32v++;
    i32--;
    ++i32;
    --u32v;

    u32v = ~u32v;

    i32 = +i32;
    u32v = -u32v;

    // Arithmetic
    i32  += i32;
    u32v -= u32v;
    i  *= i32;
    uv /= u32v;
    uv %= i32;

    uv = u32v + uv;
    i64  = i32 - i64;
    uv = u32v * uv;
    i64  = i32 * i64;
    i  = i32 % i;

    // Shift
    u32v <<= i32;
    i32  >>= u32v.y;

    i64  = i64 << u32v.z;
    uv = u32v << i;

    // Relational
    b = (u32v.x != i32);
    b = (i32 == u32v.x);
    b = (u32v.x > uv.y);
    b = (i32 < i);
    b = (u32v.y >= uv.x);
    b = (i32 <= i);

    // Bitwise
    uv |= i32;
    i  = i32 | i;
    i64  &= i32;
    uv = u32v & uv;
    uv ^= i32;
    u32v = u32v ^ i32;
}

void builtinFuncs()
{
    i32vec2  i32v;
    i32vec4  i32v4;
    u32vec3  u32v;
    u32vec2  u32v2;
    u32vec4  u32v4;
    bvec3   bv;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;
    i8vec4  i8v4;
    u8vec4  u8v4;
    i16vec2  i16v2;
    u16vec2  u16v2;

    // abs()
    i32v = abs(i32v);

    // sign()
    i32  = sign(i32);

    // min()
    i32v = min(i32v, i32);
    i32v = min(i32v, i32vec2(-1));
    u32v = min(u32v, u32);
    u32v = min(u32v, u32vec3(0));

    // max()
    i32v = max(i32v, i32);
    i32v = max(i32v, i32vec2(-1));
    u32v = max(u32v, u32);
    u32v = max(u32v, u32vec3(0));

    // clamp()
    i32v = clamp(i32v, -i32, i32);
    i32v = clamp(i32v, -i32v, i32v);
    u32v = clamp(u32v, -u32, u32);
    u32v = clamp(u32v, -u32v, u32v);

    // mix()
    i32  = mix(i32v.x, i32v.y, true);
    i32v = mix(i32vec2(i32), i32vec2(-i32), bvec2(false));
    u32  = mix(u32v.x, u32v.y, true);
    u32v = mix(u32vec3(u32), u32vec3(-u32), bvec3(false));

    //pack
    i32 = pack32(i8v4);
    i32 = pack32(i16v2);
    u32 = pack32(u8v4);
    u32 = pack32(u16v2);

    i32v  = unpack32(i64);
    u32v2  = unpack32(u64);

    // lessThan()
    bv    = lessThan(u32v, u32vec3(u32));
    bv.xy = lessThan(i32v, i32vec2(i32));

    // lessThanEqual()
    bv    = lessThanEqual(u32v, u32vec3(u32));
    bv.xy = lessThanEqual(i32v, i32vec2(i32));

    // greaterThan()
    bv    = greaterThan(u32v, u32vec3(u32));
    bv.xy = greaterThan(i32v, i32vec2(i32));

    // greaterThanEqual()
    bv    = greaterThanEqual(u32v, u32vec3(u32));
    bv.xy = greaterThanEqual(i32v, i32vec2(i32));

    // equal()
    bv    = equal(u32v, u32vec3(u32));
    bv.xy = equal(i32v, i32vec2(i32));

    // notEqual()
    bv    = notEqual(u32v, u32vec3(u32));
    bv.xy = notEqual(i32v, i32vec2(i32));
}

// Type conversion for specialization constant
layout(constant_id = 100) const int32_t  si32 = -10;
layout(constant_id = 101) const uint32_t su32 = 20U;
layout(constant_id = 102) const int  si = -5;
layout(constant_id = 103) const uint su = 4;
layout(constant_id = 104) const bool sb = true;

#define UINT32_MAX  4294967295u
uint32_t u32Max = UINT32_MAX;
