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
    int8_t   i8;
    i8vec2   i8v2;
    i8vec3   i8v3;
    i8vec4   i8v4;
    uint8_t  u8;
    u8vec2   u8v2;
    u8vec3   u8v3;
    u8vec4   u8v4;
} block;

void main()
{
}

void literal()
{
    const int8_t i8Const[3] =
    {
        int8_t(-0x11),           // Hex
        int8_t(-1),              // Dec
        int8_t(0400),            // Oct
    };

    int8_t i8 = i8Const[index];

    const uint8_t u8Const[] =
    {
        uint8_t(0xFF),             // Hex
        uint8_t(255),              // Dec
        uint8_t(0177),             // Oct
    };

    uint8_t u8 = u8Const[index];
}

void typeCast8()
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

    u8v = i8v;      // int8_t  ->  uint8_t
    i16v = i8v;     // int8_t  ->   int16_t
    i16v = u8v;     // uint8_t ->   int16_t
    i32v = i8v;     // int8_t  ->   int32_t
    i32v = u8v;     // uint8_t ->   int32_t
    u32v = i8v;     // int8_t  ->  uint32_t
    i64v = i8v;     // int8_t  ->   int64_t
    u64v = i8v;     // int8_t  ->  uint64_t
    u32v = u8v;     // uint8_t ->  uint32_t
    i64v = u8v;     // uint8_t ->   int64_t
    u64v = u8v;     // uint8_t ->  uint64_t
    f16v = i8v;     // int8_t  ->  float16_t
    f32v = i8v;     // int8_t  ->  float32_t
    f64v = i8v;     // int8_t  ->  float64_t
    f16v = u8v;     // uint8_t ->  float16_t
    f32v = u8v;     // uint8_t ->  float32_t
    f64v = u8v;     // uint8_t ->  float64_t

    i8v =  i8vec2(u8v);       // uint8_t  ->   int8_t
    i16v = i16vec2(i8v);      // int8_t   ->   int16_t
    i16v = i16vec2(u8v);      // uint8_t  ->   int16_t
    i32v = i32vec2(i8v);      // int8_t   ->   int32_t
    i32v = i32vec2(u8v);      // uint8_t  ->   int32_t
    i64v = i64vec2(i8v);      // int8_t   ->   int64_t
    u64v = i64vec2(i8v);      // int8_t   ->  uint64_t
    u16v = u16vec2(i8v);      // int8_t   ->  uint16_t
    u16v = u16vec2(u8v);      // uint8_t  ->  uint16_t
    u32v = u32vec2(u8v);      // uint8_t  ->  uint32_t
    i64v = i64vec2(u8v);      // uint8_t  ->   int64_t
    u64v = i64vec2(u8v);      // uint8_t  ->  uint64_t
    f16v = f16vec2(i8v);      // int8_t   ->  float16_t
    f32v = f32vec2(i8v);      // int8_t   ->  float32_t
    f64v = f64vec2(i8v);      // int8_t   ->  float64_t
    f16v = f16vec2(u8v);      // uint8_t  ->  float16_t
    f32v = f32vec2(u8v);      // uint8_t  ->  float32_t
    f64v = f64vec2(u8v);      // uint8_t  ->  float64_t

    i8v = i8vec2(bv);       // bool     ->   int8
    u8v = u8vec2(bv);       // bool     ->   uint8
    bv  = bvec2(i8v);       // int8    ->   bool
    bv  = bvec2(u8v);       // uint8   ->   bool
}

void operators()
{
    u8vec3 u8v;
    int8_t i8;
    uvec3   uv;
    int32_t i;
    int16_t i16;
    bool    b;

    // Unary
    u8v++;
    i8--;
    ++i8;
    --u8v;

    u8v = ~u8v;

    i8 = +i8;
    u8v = -u8v;

    // Arithmetic
    i8  += i8;
    u8v -= u8v;
    i  *= i8;
    uv /= u8v;
    uv %= i8;

    uv = u8v + uv;
    i16  = i8 - i16;
    uv = u8v * uv;
    i16  = i8 * i16;
    i  = i8 % i;

    // Shift
    u8v <<= i8;
    i8  >>= u8v.y;

    i8  = i8 << u8v.z;
    u8v = u8v << i8;

    // Relational
    b = (u8v.x != i8);
    b = (i8 == u8v.x);
    b = (u8v.x > uv.y);
    b = (i8 < i);
    b = (u8v.y >= uv.x);
    b = (i8 <= i);

    // Bitwise
    uv |= i8;
    i  = i8 | i;
    i16  &= i8;
    uv = u8v & uv;
    uv ^= i8;
    u8v = u8v ^ i8;
}

void builtinFuncs()
{
    i8vec2  i8v;
    i8vec4  i8v4;
    u8vec3  u8v;
    u8vec2  u8v2;
    u8vec4  u8v4;
    bvec3   bv;
    int16_t i16;
    int32_t i32;
    uint16_t u16;
    uint32_t u32;

    int8_t  i8;
    uint8_t u8;

    // abs()
    i8v = abs(i8v);

    // sign()
    i8  = sign(i8);

    // min()
    i8v = min(i8v, i8);
    i8v = min(i8v, i8vec2(-1));
    u8v = min(u8v, u8);
    u8v = min(u8v, u8vec3(0));

    // max()
    i8v = max(i8v, i8);
    i8v = max(i8v, i8vec2(-1));
    u8v = max(u8v, u8);
    u8v = max(u8v, u8vec3(0));

    // clamp()
    i8v = clamp(i8v, -i8, i8);
    i8v = clamp(i8v, -i8v, i8v);
    u8v = clamp(u8v, -u8, u8);
    u8v = clamp(u8v, -u8v, u8v);

    // mix()
    i8  = mix(i8v.x, i8v.y, true);
    i8v = mix(i8vec2(i8), i8vec2(-i8), bvec2(false));
    u8  = mix(u8v.x, u8v.y, true);
    u8v = mix(u8vec3(u8), u8vec3(-u8), bvec3(false));

    //pack
    i16 = pack16(i8v);
    i32 = pack32(i8v4);
    u16 = pack16(u8v2);
    u32 = pack32(u8v4);

    i8v  = unpack8(i16);
    i8v4 = unpack8(i32);
    u8v2 = unpack8(u16);
    u8v4 = unpack8(u32);

    // lessThan()
    bv    = lessThan(u8v, u8vec3(u8));
    bv.xy = lessThan(i8v, i8vec2(i8));

    // lessThanEqual()
    bv    = lessThanEqual(u8v, u8vec3(u8));
    bv.xy = lessThanEqual(i8v, i8vec2(i8));

    // greaterThan()
    bv    = greaterThan(u8v, u8vec3(u8));
    bv.xy = greaterThan(i8v, i8vec2(i8));

    // greaterThanEqual()
    bv    = greaterThanEqual(u8v, u8vec3(u8));
    bv.xy = greaterThanEqual(i8v, i8vec2(i8));

    // equal()
    bv    = equal(u8v, u8vec3(u8));
    bv.xy = equal(i8v, i8vec2(i8));

    // notEqual()
    bv    = notEqual(u8v, u8vec3(u8));
    bv.xy = notEqual(i8v, i8vec2(i8));
}

// Type conversion for specialization constant
layout(constant_id = 100) const int8_t  si8 = int8_t(-10);
layout(constant_id = 101) const uint8_t su8 = uint8_t(20);
