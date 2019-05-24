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
    float16_t  w[];
} b2;

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
} b4;

void func3(S2 x) {
}

S2 func4() {
    return b4.x;
}

float func(float16_t a) {
    return 0.0;
}

struct S4 {
    float x;
    float16_t y;
};

float func2(float a) { return 0.0; }

void main()
{
    b2.o = b2.q[1];
    b2.p = b2.q.xy;
    b2.o = max(b1.a, b1.a);
    bvec2 bv = lessThan(b2.p, b2.p);
    b2.o = b1.a + b1.a;
    b2.o = -b1.a;
    b2.o = b1.a + 1.0;
    b2.p = b2.p.yx;
    b4.x = b3.x;
    float16_t f0;
    S2 f1;
    S3 f2;
    if (b1.a == b1.a) {}
    b2.r = b2.r;
    b2.o = 1.0HF;
    b2.p = f16vec2(3.0, 4.0);
    f16vec2[2](f16vec2(vec2(1.0,2.0)), f16vec2(vec2(3.0,4.0)));
    // NOT ERRORING YET
    b3.x;
    S4(0.0, float16_t(0.0));
    func2(b1.a);
}


layout(column_major, std140) uniform B6
{
    f16mat2x3  e;
} b6;

