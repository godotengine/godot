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

void typeCast64()
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

    u64v = i64v;     // int64_t  ->  uint64_t
    f64v = i64v;     // int64_t  ->  float64_t
    f64v = u64v;     // uint64_t ->  float64_t

    i8v =  i8vec2(i64v);       // int64_t   ->   int8_t
    i8v =  i8vec2(u64v);       // uint64_t  ->   int8_t
    i16v = i16vec2(i64v);      // int64_t   ->   int16_t
    i16v = i16vec2(u64v);      // uint64_t  ->   int16_t
    i32v = i32vec2(i64v);      // int64_t   ->   int32_t
    i32v = i32vec2(u64v);      // uint64_t  ->   int32_t
	i64v = i64vec2(u64v);      // uint64_t  ->   int64_t
	u8v =  u8vec2(i64v);       // int64_t   ->   uint8_t
    u8v =  u8vec2(u64v);       // uint64_t  ->   uint8_t
    u16v = u16vec2(i64v);      // int64_t   ->   uint16_t
    u16v = u16vec2(u64v);      // uint64_t  ->   uint16_t
    u32v = u32vec2(i64v);      // int64_t   ->   uint32_t
    u32v = u32vec2(u64v);      // uint64_t  ->   uint32_t
    u64v = u64vec2(i64v);      // int64_t   ->   uint64_t
    u64v = u64vec2(u64v);      // uint64_t  ->   uint64_t

    f16v = f16vec2(i64v);      // int64_t   ->  float16_t
    f32v = f32vec2(i64v);      // int64_t   ->  float32_t
    f64v = f64vec2(i64v);      // int64_t   ->  float64_t
    f16v = f16vec2(u64v);      // uint64_t  ->  float16_t
    f32v = f32vec2(u64v);      // uint64_t  ->  float32_t
    f64v = f64vec2(u64v);      // uint64_t  ->  float64_t

    i64v = i64vec2(bv);       // bool     ->   int64
    u64v = u64vec2(bv);       // bool     ->   uint64
    bv   = bvec2(i64v);       // int64    ->   bool
    bv   = bvec2(u64v);       // uint64   ->   bool
}
