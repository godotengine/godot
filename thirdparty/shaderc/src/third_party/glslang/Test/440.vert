#version 440

// Note 'location' tests for enhanced layouts are in 330.frag

layout(location = 2, component = 2) in vec2 a; 
layout(location = 2, component = 1) in float b; 

layout(location = 3, component = 2) in vec3 c;      // ERROR: c overflows components 2 and 3

layout(location = 0, component = 3) in float d[4]; 

layout(location = 4, component = 0) in vec3 e[5];
layout(location = 4, component = 3) in float f[5];

layout(location = 9, component = 4) in float g[6];   // ERROR, component too big

layout(location = 4, component = 2) in vec2 h;       // component overlap okay for vertex in

layout(location = 3, component = 2) out vec2 i;
layout(location = 3, component = 0) out vec2 j;

layout(location = 4, component = 2) out vec2 k;
layout(location = 4, component = 2) out vec2 m;      // ERROR, component overlap

layout(location = 2, component = 2) out vec2 n;
layout(location = 2, component = 0) out vec3 p;      // ERROR, component overlap

layout(location = 10, component = 3) out float q[6]; 
layout(location = 10, component = 0) out vec3 r[6];

layout(location = 15, component = 3) out float s;    // ERROR, overlap
layout(location = 10, component = 1) out float t;    // ERROR, overlap

layout(location = 20, component = 2) out float u;
layout(location = 20, component = 0) out float v;
layout(location = 20, component = 3) out float w;
layout(location = 20, component = 1) out vec2 x;     // ERROR, overlap

layout(location = 30, component = 3) out vec2 y;     // ERROR, goes to component 4
layout(location = 31, component = 1) out vec4 z;     // ERROR, goes to component 4

layout(location = 32, component = 1) out mat4 ba;               // ERROR
layout(location = 33, component = 1) out struct S {int a;} Ss;  // ERROR
layout(location = 34, component = 1) out bn { int a;} bb;       // ERROR

layout(component = 1) out float bc;    // ERROR, no location

out blockname {
    layout(location = 40, component = 2) out float u;
    layout(location = 40, component = 0) out float v;
    layout(location = 40, component = 3) out float w;
    layout(location = 40, component = 1) out vec2 x;     // ERROR, overlap

    layout(location = 41, component = 3) out vec2 y;     // ERROR, goes to component 4
    layout(location = 42, component = 1) out vec4 z;     // ERROR, goes to component 4

    layout(location = 42, component = 1) out mat4 ba;    // ERROR
    layout(location = 43, component = 1) out S Ss;       // ERROR
} bd;

layout(location = 1, component = 1) out;                 // ERROR, no global setting

layout(location = 50, component = 3) out int be;
layout(location = 50, component = 0) out vec3 bf;

layout(location = 51, component = 1) out double dfo;     // ERROR, odd component
layout(location = 52, component = 2) out dvec2 dvo;      // ERROR, overflow
layout(location = 53) out double dfo2;
layout(location = 53, component = 2) out vec2 ffv2;      // okay, fits
layout(location = 54) out dvec4 dvec4out;                // uses up location 55 too
layout(location = 55) out float overf;                   // ERROR, collides with previous dvec4
layout(location = 56, component = 1) out vec2 df2o;
layout(location = 56, component = 3) out float sf2o;
layout(location = 57, component = 2) out vec2 dv3o;
layout(location = 57, component = 3) out float sf4o;     // ERROR, overlapping component
layout(location=58) out flat dvec3 dv3o2;                // uses part of location 59
layout(location=59, component=2) out flat double dfo3;   // okay, fits
layout(location=59, component=0) out flat double dfo4;   // ERROR, overlaps the dvec3 in starting in 58

out bblck1 {
    vec4 bbv;
} bbinst1;

out bblck2 {
    layout(xfb_offset=64) vec4 bbv;
} bbinst2;

layout(xfb_buffer = 3, xfb_stride = 64) out;  // default buffer is 3

out bblck3 {
    layout(xfb_offset=16) vec4 bbv;  // in xfb_buffer 3
} bbinst3;

uniform ubblck3 {
    layout(xfb_offset=16) vec4 bbv;  // ERROR, not in a uniform
} ubbinst3;

layout(xfb_buffer=2, xfb_offset=48, xfb_stride=80) out vec4 bg;
layout(              xfb_offset=32, xfb_stride=64) out vec4 bh;

layout(xfb_offset=48) out; // ERROR

layout(xfb_stride=80, xfb_buffer=2, xfb_offset=16) out bblck4 {
    vec4 bbv1;
    vec4 bbv2;
} bbinst4;

out bblck5 {
    layout(xfb_offset=0) vec4 bbv1;
    layout(xfb_stride=64, xfb_buffer=3, xfb_offset=48) vec4 bbv2;
    layout(xfb_buffer=2) vec4 bbv3;                               // ERROR, wrong buffer
} bbinst5;

out layout(xfb_buffer=2) bblck6 {
    layout(xfb_offset=0) vec4 bbv1;
    layout(xfb_stride=64, xfb_buffer=3, xfb_offset=32) vec4 bbv2; // ERROR, overlap 32 from bh, and buffer contradiction
    layout(xfb_buffer=2, xfb_offset=0) vec4 bbv3;                 // ERROR, overlap 0 from bbinst5
    layout(xfb_buffer=2) vec4 bbv5;
    layout(xfb_offset=24) float bbf6;                             // ERROR, overlap 24 from bbv1 in bbinst4
} bbinst6;

layout(xfb_stride=48) out;                   // ERROR, stride of buffer 3

layout(xfb_buffer=1) out;  // default buffer is 1
layout(xfb_offset=4) out float bj;
layout(xfb_offset=0) out ivec2 bk;           // ERROR, overlap 4

layout(xfb_buffer=3, xfb_stride=48) out;     // ERROR, stride of buffer 3 (default is now 3)
layout(xfb_stride=48) out float bl;          // ERROR, stride of buffer 3

layout(xfb_stride=48) out bblck7 {           // ERROR, stride of buffer 3
    layout(xfb_stride=64) vec4 bbv1;
    layout(xfb_stride=32) vec4 bbv2;         // ERROR, stride of buffer 3
} bbinst7;

struct S5 {
    int i;    // 4 bytes plus 4 byte hole
    double d; // 8 bytes
    float f;  // 4 bytes
};  // total size = 20

struct T {
    bool b;   // 4 plus 4 byte hole
    S5 s;     // 20 
    vec2 v2;  // 8
};  // total size = 36

out layout(xfb_buffer=0, xfb_offset=0, xfb_stride=92) bblck8 {  // ERROR, stride not multiple of 8
    bool b;    // offset 0
    T t;       // offset 8, size 40
    int i;     // offset 40 + 4 = 48
    mat3x3 m3; // offset 52
    float f;   // offset 52 + 9*4 = 88
    float g;   // ERROR, overflow stride
} bbinst8;

out layout(xfb_buffer=4) bblck9 {
    layout(xfb_offset=1) bool b;     // ERROR
    layout(xfb_offset=12) T t;       // ERROR
    layout(xfb_offset=52) mat3x3 m3; // non-multiple of 8 okay
    layout(xfb_offset=90) int i;     // ERROR
    layout(xfb_offset=98) double d;  // ERROR
    layout(xfb_offset=108) S s;      // non-multiple of 8 okay
} bbinst9;

layout(xfb_buffer=5, xfb_stride=6) out;     // link ERROR, stride not multiple of 4
layout(xfb_offset=0) out float bm;

layout(xfb_buffer=6, xfb_stride=2000) out;  // ERROR, stride too big

out layout(xfb_buffer=7, xfb_offset=0) bblck10 {  // link ERROR, implicit stride too big
    dmat4x4 m1;
    dmat4x4 m2;
    float f;
} bbinst10;

layout(xfb_buffer = 3) out;
layout(xfb_offset = 32) out gl_PerVertex {
    layout(xfb_buffer = 2) float gl_PointSize; // ERROR, change in xfb_buffer
    vec4 gl_Position;
};

int drawParamsBad()
{
    return gl_BaseVertexARB + gl_BaseInstanceARB + gl_DrawIDARB; // ERROR, extension not requested
}

#extension GL_ARB_shader_draw_parameters: enable

int drawParams()
{
    return gl_BaseVertexARB + gl_BaseInstanceARB + gl_DrawIDARB;
    gl_BaseVertexARB = 3;       // ERROR, can't write to shader 'in'
    gl_BaseInstanceARB = 3;     // ERROR, can't write to shader 'in'
    gl_DrawIDARB = 3;           // ERROR, can't write to shader 'in'
    glBaseInstanceARB;          // ERROR, not defined
}
