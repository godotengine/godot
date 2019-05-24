#version 450 core

#extension GL_EXT_scalar_block_layout : enable

// Block memory layout
struct S
{
    float      a;   // offset 0
    vec2       b;   // offset 4
    double     c;   // offset 16
    float      d;   // offset 24
    vec3       e;   // offset 28
    float      f;   // offset 40
    // size = 44, align = 8
};

layout(column_major, scalar) uniform B1
{
    float      a;     // offset = 0
    vec2       b;     // offset = 4
    vec3       c;     // offset = 12
    float      d[2];  // offset = 24
    mat2x3     e;     // offset = 32, takes 24 bytes, matrixstride = 12
    mat2x3     f[2];  // offset = 56, takes 48 bytes, matrixstride = 12, arraystride = 24
    float      g;     // offset = 104
    S          h;     // offset = 112 (aligned to multiple of 8)
    S          i[2];  // offset = 160 (aligned to multiple of 8) stride = 48
};

void main()
{
}
