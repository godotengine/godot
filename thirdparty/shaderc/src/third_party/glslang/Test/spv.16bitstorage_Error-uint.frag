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
    uint16_t  w[];
} b2;

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
} b4;

void func3(S2 x) {
}

S2 func4() {
    return b4.x;
}

uint func(uint16_t a) {
    return 0;
}

struct S4 {
    uint x;
    uint16_t y;
};

uint func2(uint a) { return 0; }

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
    uint16_t f0;
    S2 f1;
    S3 f2;
    if (b1.a == b1.a) {}
    b2.r = b2.r;
    b2.p = u16vec2(3, 4);
    u16vec2[2](u16vec2(uvec2(1,2)), u16vec2(uvec2(3,4)));
    // NOT ERRORING YET
    b3.x;
    S4(0u, uint16_t(0u));
    func2(b1.a);
}


layout(column_major, std140) uniform B6
{
    u16mat2x3  e;
} b6;

