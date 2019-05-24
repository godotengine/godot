#version 450 core

#extension GL_EXT_shader_8bit_storage : enable

struct S
{
    int8_t  x;
    i8vec2    y;
    i8vec3    z;
};

layout(column_major, std140) uniform B1
{
    int8_t  a;
    i8vec2    b;
    i8vec3    c;
    int8_t  d[2];
    S          g;
    S          h[2];
    int        j;
} b1;

layout(row_major, std430) buffer B2
{
    int8_t  o;
    i8vec2    p;
    i8vec3    q;
    int8_t  r[2];
    S          u;
    S          v[2];
    int8_t  w[];
} b2;

struct S2 {
    mat4x4 x;
    int8_t y;
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

int func(int8_t a) {
    return 0;
}

struct S4 {
    int x;
    int8_t y;
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
    int8_t f0;
    S2 f1;
    S3 f2;
    if (b1.a == b1.a) {}
    b2.r = b2.r;
    b2.p = i8vec2(3, 4);
    i8vec2[2](i8vec2(ivec2(1,2)), i8vec2(ivec2(3,4)));
    // NOT ERRORING YET
    b3.x;
    S4(0, int8_t(0));
    func2(b1.a);
}


layout(column_major, std140) uniform B6
{
    i8mat2x3  e;
} b6;

