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
    int16_t   i16;
    i16vec2   i16v2;
    i16vec3   i16v3;
    i16vec4   i16v4;
    uint16_t  u16;
    u16vec2   u16v2;
    u16vec3   u16v3;
    u16vec4   u16v4;
} block;

void main()
{
}

void literal()
{
    const int16_t i16Const[3] =
    {
        int16_t(-0x1111),           // Hex
        int16_t(-1),                // Dec
        int16_t(040000),            // Oct
    };

    int16_t i16 = i16Const[index];

    const uint16_t u16Const[] =
    {
        uint16_t(0xFFFF),             // Hex
        uint16_t(65535),              // Dec
        uint16_t(077777),             // Oct
    };

    uint16_t u16 = u16Const[index];
}

void typeCast16()
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

    i32v = i16v;     // int16_t  ->   int32_t
    i32v = u16v;     // uint16_t ->   int32_t
    u16v = i16v;     // int16_t  ->  uint16_t
    u32v = i16v;     // int16_t  ->  uint32_t
    i64v = i16v;     // int16_t  ->   int64_t
    u64v = i16v;     // int16_t  ->  uint64_t
    u32v = u16v;     // uint16_t ->  uint32_t
    i64v = u16v;     // uint16_t ->   int64_t
    u64v = u16v;     // uint16_t ->  uint64_t
    f16v = i16v;     // int16_t  ->  float16_t
    f32v = i16v;     // int16_t  ->  float32_t
    f64v = i16v;     // int16_t  ->  float64_t
    f16v = u16v;     // uint16_t ->  float16_t
    f32v = u16v;     // uint16_t ->  float32_t
    f64v = u16v;     // uint16_t ->  float64_t

    i32v = i32vec2(i16v);     // int16_t  ->   int32_t
    i32v = i32vec2(u16v);     // uint16_t ->   int32_t
    u16v = u16vec2(i16v);     // int16_t  ->  uint16_t
    u32v = u32vec2(i16v);     // int16_t  ->  uint32_t
    i64v = i64vec2(i16v);     // int16_t  ->   int64_t
    u64v = i64vec2(i16v);     // int16_t  ->  uint64_t
    u32v = u32vec2(u16v);     // uint16_t ->  uint32_t
    i64v = i64vec2(u16v);     // uint16_t ->   int64_t
    u64v = i64vec2(u16v);     // uint16_t ->  uint64_t
    f16v = f16vec2(i16v);     // int16_t  ->  float16_t
    f32v = f32vec2(i16v);     // int16_t  ->  float32_t
    f64v = f64vec2(i16v);     // int16_t  ->  float64_t
    f16v = f16vec2(u16v);     // uint16_t ->  float16_t
    f32v = f32vec2(u16v);     // uint16_t ->  float32_t
    f64v = f64vec2(u16v);     // uint16_t ->  float64_t

    i8v  = i8vec2(i16v);      // int16_t  ->   int8_t
    i8v  = i8vec2(u16v);      // uint16_t ->   int8_t
    u8v  = u8vec2(i16v);      // int16_t  ->  uint8_t
    u8v  = u8vec2(u16v);      // uint16_t ->  uint8_t
    i16v = u8vec2(u16v);      // uint16_t ->   int16_t
    i16v = i16vec2(bv);       // bool     ->   int16
    u16v = u16vec2(bv);       // bool     ->   uint16
    bv   = bvec2(i16v);       // int16    ->   bool
    bv   = bvec2(u16v);       // uint16   ->   bool
}
void operators()
{
    u16vec3 u16v;
    int16_t i16;
    uvec3   uv;
    int32_t i;
    int64_t i64;
    bool    b;

    // Unary
    u16v++;
    i16--;
    ++i16;
    --u16v;

    u16v = ~u16v;

    i16 = +i16;
    u16v = -u16v;

    // Arithmetic
    i16  += i16;
    u16v -= u16v;
    i  *= i16;
    uv /= u16v;
    uv %= i16;

    uv = u16v + uv;
    i64  = i16 - i64;
    uv = u16v * uv;
    i64  = i16 * i64;
    i  = i16 % i;

    // Shift
    u16v <<= i16;
    i16  >>= u16v.y;

    i16  = i16 << u16v.z;
    uv = u16v << i;

    // Relational
    b = (u16v.x != i16);
    b = (i16 == u16v.x);
    b = (u16v.x > uv.y);
    b = (i16 < i);
    b = (u16v.y >= uv.x);
    b = (i16 <= i);

    // Bitwise
    uv |= i16;
    i  = i16 | i;
    i64  &= i16;
    uv = u16v & uv;
    uv ^= i16;
    u16v = u16v ^ i16;
}

void builtinFuncs()
{
    i16vec2  i16v;
    i16vec4  i16v4;
    u16vec3  u16v;
    u16vec2  u16v2;
    u16vec4  u16v4;
    bvec3   bv;
    int16_t i16;
    uint16_t u16;
    int32_t i32;
    uint32_t u32;
    int64_t i64;
    uint64_t u64;

    // abs()
    i16v = abs(i16v);

    // sign()
    i16  = sign(i16);

    // min()
    i16v = min(i16v, i16);
    i16v = min(i16v, i16vec2(-1));
    u16v = min(u16v, u16);
    u16v = min(u16v, u16vec3(0));

    // max()
    i16v = max(i16v, i16);
    i16v = max(i16v, i16vec2(-1));
    u16v = max(u16v, u16);
    u16v = max(u16v, u16vec3(0));

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

    //pack
    i32 = pack32(i16v);
    i64 = pack64(i16v4);
    u32 = pack32(u16v2);
    u64 = pack64(u16v4);

    i16v  = unpack16(i32);
    i16v4 = unpack16(i64);
    u16v2 = unpack16(u32);
    u16v4 = unpack16(u64);

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
layout(constant_id = 100) const int16_t  si16 = int16_t(-10);
layout(constant_id = 101) const uint16_t su16 = uint16_t(20);
