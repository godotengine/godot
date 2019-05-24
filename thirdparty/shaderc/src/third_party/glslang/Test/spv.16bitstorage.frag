#version 450 core

#extension GL_EXT_shader_16bit_storage : enable

struct S
{
    float16_t  x;
    f16vec2    y;
    f16vec3    z;
};

layout(column_major, std140) uniform B1
{
    float16_t  a;
    f16vec2    b;
    f16vec3    c;
    float16_t  d[2];
    S          g;
    S          h[2];
    int        j;
} b1;

layout(row_major, std430) buffer B2
{
    float16_t  o;
    f16vec2    p;
    f16vec3    q;
    float16_t  r[2];
    S          u;
    S          v[2];
    f16vec2    x[100];
    float16_t  w[];
} b2;

layout(row_major, std140) uniform B5
{
    float16_t  o;
    f16vec2    p;
    f16vec3    q;
    float16_t  r[2];
    S          u;
    S          v[2];
    f16vec2    x[100];
    float16_t  w[100];
} b5;

struct S2 {
    mat4x4 x;
    float16_t y;
    float z;
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
    b2.p = f16vec2(vec3(b2.q).xy);
    b2.p = f16vec2(vec3(b5.q).xy);
    b2.r[0] = b2.r[0];
    b2.r[1] = b5.r[1];
    b2.p = b2.p;
    float x0 = float(b1.a);
    vec4 x1 = vec4(b1.a, b2.p, 1.0);
    b4.x.x = b3.x.x;
    b2.o = float16_t(vec2(b2.p).x);
    b2.p = b2.v[1].y;
    vec3 v3 = vec3(b2.w[b1.j], b2.w[b1.j+1], b2.w[b1.j+2]);
    vec3 u3 = vec3(b5.w[b1.j], b5.w[b1.j+1], b5.w[b1.j+2]);
    b2.x[0] = b2.x[0];
    b2.x[1] = b5.x[1];
    b2.p.x = b1.a;
    b2.o = b2.p.x;
    b2.p = f16vec2(vec2(1.0, 2.0));
    b2.o = float16_t(3.0);
}

