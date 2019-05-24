#version 450 core

#extension GL_EXT_shader_16bit_storage : enable

struct S
{
    uint16_t  x;
    u16vec2    y;
    u16vec3    z;
};

layout(column_major, std140) uniform B1
{
    uint16_t  a;
    u16vec2    b;
    u16vec3    c;
    uint16_t  d[2];
    S          g;
    S          h[2];
    uint        j;
} b1;

layout(row_major, std430) buffer B2
{
    uint16_t  o;
    u16vec2    p;
    u16vec3    q;
    uint16_t  r[2];
    S          u;
    S          v[2];
    u16vec2    x[100];
    uint16_t  w[];
} b2;

layout(row_major, std140) uniform B5
{
    uint16_t  o;
    u16vec2    p;
    u16vec3    q;
    uint16_t  r[2];
    S          u;
    S          v[2];
    u16vec2    x[100];
    uint16_t  w[100];
} b5;

struct S2 {
    mat4x4 x;
    uint16_t y;
    uint z;
};

struct S3 {
    S2 x;
};

layout(row_major, std430) buffer B3
{
    S2 x;
} b3;

layout(column_major, std430) buffer B4
{
    S2 x;
    S3 y;
} b4;

void main()
{
    b2.o = b1.a;
    b2.p = u16vec2(uvec3(b2.q).xy);
    b2.p = u16vec2(uvec3(b5.q).xy);
    b2.r[0] = b2.r[0];
    b2.r[1] = b5.r[1];
    b2.p = b2.p;
    uint x0 = uint(b1.a);
    uvec4 x1 = uvec4(b1.a, b2.p, 1);
    b4.x.x = b3.x.x;
    b2.o = uint16_t(uvec2(b2.p).x);
    b2.p = b2.v[1].y;
    uvec3 v3 = uvec3(b2.w[b1.j], b2.w[b1.j+1], b2.w[b1.j+2]);
    uvec3 u3 = uvec3(b5.w[b1.j], b5.w[b1.j+1], b5.w[b1.j+2]);
    b2.x[0] = b2.x[0];
    b2.x[1] = b5.x[1];
    b2.p.x = b1.a;
    b2.o = b2.p.x;
    b2.p = u16vec2(uvec2(1, 2));
    b2.o = uint16_t(3u);
}

