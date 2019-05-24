#version 450 core

#extension GL_EXT_shader_8bit_storage : enable

struct S
{
    uint8_t  x;
    u8vec2    y;
    u8vec3    z;
};

layout(column_major, std140) uniform B1
{
    uint8_t  a;
    u8vec2    b;
    u8vec3    c;
    uint8_t  d[2];
    S          g;
    S          h[2];
    uint        j;
} b1;

layout(row_major, std430) buffer B2
{
    uint8_t  o;
    u8vec2    p;
    u8vec3    q;
    uint8_t  r[2];
    S          u;
    S          v[2];
    uint8_t  w[];
} b2;

struct S2 {
    mat4x4 x;
    uint8_t y;
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

uint func(uint8_t a) {
    return 0;
}

struct S4 {
    uint x;
    uint8_t y;
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
    uint8_t f0;
    S2 f1;
    S3 f2;
    if (b1.a == b1.a) {}
    b2.r = b2.r;
    b2.p = u8vec2(3, 4);
    u8vec2[2](u8vec2(uvec2(1,2)), u8vec2(uvec2(3,4)));
    // NOT ERRORING YET
    b3.x;
    S4(0u, uint8_t(0u));
    func2(b1.a);
}


layout(column_major, std140) uniform B6
{
    u8mat2x3  e;
} b6;

