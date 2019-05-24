#version 430 core

layout(location = 3) vec4 v4;  // ERROR

layout(location = 4) uniform vec4 uv4;

layout(location = 2) in   inb1 { vec4 v; } b1;  // ERROR
layout(location = 2) out outb1 { vec4 v; } b2;  // ERROR

out gl_PerVertex {
    float gl_ClipDistance[];
};

void foo()
{
    gl_ClipDistance[2] = 3.7;
}

struct sp {
    highp float f;
    in float g;             // ERROR
    uniform float h;        // ERROR
    invariant float i;      // ERROR
    volatile float j;       // ERROR
    layout(row_major) mat3 m3; // ERROR
};

void foo3(invariant vec4 v4,                 // ERROR
          volatile vec3 v3,
          layout(location = 3) vec2 v2,      // ERROR
          centroid vec3 cv3)                 // ERROR
{
}

struct S {
    mat3x2 m[7];  // needs 7*3 locations
    float f;      // needs 1 location
};                // needs 22 locations

layout(location = 10) out S cs[2];     // 10 through 10 + 2 * 22 - 1 = 53
layout(location = 54) out float cf;
layout(location = 53) out float cg; // ERROR, collision at 31

layout(location = 10) in vec4 alias1;
layout(location = 10) in vec4 alias2;  // okay for vertex input on desktop

out float gl_ClipDistance[17];  // ERROR, size too big

// enhanced_layouts (most tests are in 440.*)

layout(location = start*start - 2 - 4) in vec4 v6e;    // ERROR

layout(location = 28) in inblock2e {
    layout(location = 25) float f2;                     // ERROR
} ininst2e;

in ublock4e {
    layout(location = 50) float f1;                      // ERROR
    layout(location = 51) float f2;                      // ERROR
} in4e;

layout(align=16, std140) uniform  ubl4e { int a; } inst4e;// ERROR

layout(align=32) uniform ubl9e {                          // ERROR
    layout(offset=12, align=4) float f;                   // ERROR
    layout(offset=20) float g;                            // ERROR
} inst9e;

layout(std140) uniform blocke {
                        vec4   a;
    layout(offset = 32) vec3   b;                          // ERROR
} spinste;

int aconste[gl_MaxTransformFeedbackBuffers];               // ERROR
int bconste[gl_MaxTransformFeedbackInterleavedComponents]; // ERROR

out bblck2 {
    layout(xfb_offset=64) vec4 bbv;                              // ERROR
} bbinst2;

layout(xfb_buffer = 3, xfb_stride = 64) out;                     // ERROR

layout(xfb_buffer=2, xfb_offset=48, xfb_stride=80) out vec4 bge; // ERROR
layout(              xfb_offset=32, xfb_stride=64) out vec4 bhe; // ERROR

layout(xfb_stride=80, xfb_buffer=2, xfb_offset=16) out bblck4e { // ERROR
    vec4 bbv1;
    vec4 bbv2;
} bbinst4e;

out bblck5e {
    layout(xfb_offset=0) vec4 bbv1;                               // ERROR
    layout(xfb_stride=64, xfb_buffer=3, xfb_offset=48) vec4 bbv2; // ERROR
} bbinst5e;

#extension GL_ARB_enhanced_layouts : enable

layout(align=16, std140) uniform  ubl4 { int a; } inst4;
layout(std430) uniform;

layout(align=32) uniform ubl9 {
    layout(offset=12, align=4) float f;
    layout(offset=20) float g;
} inst9;

layout(std140) uniform block {
                        vec4   a;     // a takes offsets 0-15
    layout(offset = 32) vec3   b;     // b takes offsets 32-43
} spinst;

int aconst[gl_MaxTransformFeedbackBuffers];
int bconst[gl_MaxTransformFeedbackInterleavedComponents];

const int start2 = 5;
layout(location = start2 * start2 - 2 - 4) in vec4 v6;

layout(location = 28) in inblock2 {  // ERROR, input block in vertex shader, other errors are valid checks still...
    bool b1;
    float f1;
    layout(location = 25) float f2;
} ininst2;

in ublock4 {                         // ERROR, input block in vertex shader, other errors are valid checks still...
    layout(location = 50) float f1;
    layout(location = 51) float f2;
} in4;

out bblck2g {
    layout(xfb_offset=64) vec4 bbv;
} bbinst2g;

layout(xfb_buffer = 1, xfb_stride = 80) out;  // default buffer is 3

layout(xfb_buffer=1, xfb_offset=48, xfb_stride=80) out vec4 bg;
layout(              xfb_offset=32, xfb_stride=80) out vec4 bh;

layout(xfb_stride=80, xfb_buffer=1, xfb_offset=16) out bblck4 {
    vec4 bbv1;
} bbinst4;

out bblck5 {
    layout(xfb_offset=0) vec4 bbv1;
    layout(xfb_stride=80, xfb_buffer=1, xfb_offset=64) vec4 bbv2;
} bbinst5;

shared vec4 sharedv;                // ERROR

void fooBarrier()
{
    barrier();                       // ERROR
    memoryBarrier();
    memoryBarrierAtomicCounter();
    memoryBarrierBuffer();
    memoryBarrierShared();           // ERROR
    memoryBarrierImage();
    groupMemoryBarrier();            // ERROR
}

buffer vec4 v;  // ERROR

uniform sampler2DMS s2dms;
uniform usampler2DMSArray us2dmsa;
layout(rgba32i) uniform iimage2DMS ii2dms;
layout(rgba32f) uniform image2DMSArray i2dmsa;

void fooq()
{
    int s = textureSamples(s2dms); // ERROR
    s += textureSamples(us2dmsa);  // ERROR
    s += imageSamples(ii2dms);     // ERROR
    s += imageSamples(i2dmsa);     // ERROR
}

#extension GL_ARB_shader_texture_image_samples : enable

void fooq2()
{
    int s = textureSamples(s2dms);
    s += textureSamples(us2dmsa); 
    s += imageSamples(ii2dms);    
    s += imageSamples(i2dmsa);    
}

uniform sampler1D samp1D;
uniform usampler2D usamp2D;
uniform isampler3D isamp3D;
uniform isamplerCube isampCube; 
uniform isampler1DArray isamp1DA;
uniform sampler2DArray samp2DA;
uniform usamplerCubeArray usampCubeA;

uniform sampler1DShadow samp1Ds;
uniform sampler2DShadow samp2Ds;
uniform samplerCubeShadow sampCubes;
uniform sampler1DArrayShadow samp1DAs;
uniform sampler2DArrayShadow samp2DAs;
uniform samplerCubeArrayShadow sampCubeAs;

uniform samplerBuffer sampBuf;
uniform sampler2DRect sampRect;

void qlod()
{
    int levels;

    levels = textureQueryLevels(samp1D);
    levels = textureQueryLevels(usamp2D);
    levels = textureQueryLevels(isamp3D);
    levels = textureQueryLevels(isampCube);
    levels = textureQueryLevels(isamp1DA);
    levels = textureQueryLevels(samp2DA);
    levels = textureQueryLevels(usampCubeA);

    levels = textureQueryLevels(samp1Ds);
    levels = textureQueryLevels(samp2Ds);
    levels = textureQueryLevels(sampCubes);
    levels = textureQueryLevels(samp1DAs);
    levels = textureQueryLevels(samp2DAs);
    levels = textureQueryLevels(sampCubeAs);

    levels = textureQueryLevels(sampBuf);    // ERROR
    levels = textureQueryLevels(sampRect);   // ERROR
}
