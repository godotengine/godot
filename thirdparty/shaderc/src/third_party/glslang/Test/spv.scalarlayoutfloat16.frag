#version 450 core

#extension GL_EXT_shader_16bit_storage: enable
#extension GL_EXT_scalar_block_layout : enable

// Block memory layout
struct S
{
    float16_t      a;   // offset 0
    f16vec2        b;   // offset 2
    double         c;   // offset 8
    float16_t      d;   // offset 16
    f16vec3        e;   // offset 18
    float16_t      f;   // offset 24
    // size = 26, align = 8
};

layout(column_major, scalar) uniform B1
{
    float16_t      a;     // offset = 0
    f16vec2        b;     // offset = 2
    f16vec3        c;     // offset = 6
    float16_t      d[2];  // offset = 12 stride = 2
    float16_t      g;     // offset = 16
    S              h;     // offset = 24 (aligned to multiple of 8)
    S              i[2];  // offset = 56 (aligned to multiple of 8) stride = 32
};

void main()
{
}
