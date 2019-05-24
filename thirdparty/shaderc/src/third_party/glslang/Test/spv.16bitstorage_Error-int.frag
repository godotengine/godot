#version 450 core

#extension GL_EXT_shader_16bit_storage : enable

struct S
{
    int16_t  x;
    i16vec2    y;
    i16vec3    z;
};

layout(column_major, std140) uniform B1
{
    int16_t  a;
    i16vec2    b;
    i16vec3    c;
    int16_t  d[2];
    S          g;
    S          h[2];
    int        j;
} b1;

layout(row_major, std430) buffer B2
{
    int16_t  o;
    i16vec2    p;
    i16vec3    q;
    int16_t  r[2];
    S          u;
    S          v[2];
    int16_t  w[];
} b2;

struct S2 {
    mat4x4 x;
    int16_t y;
    int z;
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

int func(int16_t a) {
    return 0;
}

struct S4 {
    int x;
    int16_t y;
};

int func2(int a) { return 0; }

void main()
{
    b2.o = b2.q[1];
    b2.p = b2.q.xy;
    b2.o = max(b1.a, b1.a);
    bvec2 bv = lessThan(b2.p, b2.p);
    b2.o = b1.a + b1.a;
    b2.o = -b1.a;
    b2.o = b1.a + 1;
    b2.p = b2.p.yx;
    b4.x = b3.x;
    int16_t f0;
    S2 f1;
    S3 f2;
    if (b1.a == b1.a) {}
    b2.r = b2.r;
    b2.p = i16vec2(3, 4);
    i16vec2[2](i16vec2(ivec2(1,2)), i16vec2(ivec2(3,4)));
    // NOT ERRORING YET
    b3.x;
    S4(0, int16_t(0));
    func2(b1.a);
}


layout(column_major, std140) uniform B6
{
    i16mat2x3  e;
} b6;

