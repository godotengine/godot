#version 330 compatibility

in vec4 inVar;
layout(location=0, index=0) out vec4 outVar;

varying vec4 varyingVar;

void main()
{
    gl_FragColor = varyingVar;  // link ERROR: user output was used
    gl_FragData[1] = inVar;     // link ERROR: user output was used
    int buffer = 4;
}

#extension GL_ARB_separate_shader_objects : enable

in gl_PerFragment {
    vec4 gl_Color;
};

void foo()
{
    vec4 c = gl_Color;
    outVar = inVar;
}

in gl_block { // ERROR
    int gl_i;
} gl_name;

in myBlock {
    int gl_i;  // ERROR
} gl_name;     // ERROR

in gl_PerVertex {  // ERROR
    vec4 gl_FragCoord;
} gl_in[];

in gl_PerVertex {  // ERROR
    vec4 gl_FragCoord;
};  // ERROR

const int start = 6;
layout(location = -2) in vec4 v1;         // ERROR
layout(location = start + 2) in vec4 v2;  // ERROR
layout(location = 4.7e10) in vec4 v20;    // ERROR
layout(location = +60) in float v21;      // ERROR
layout(location = (2)) in float v22;      // ERROR

struct S {
    float f1;
    layout(location = 3) float f2;        // ERROR
};

layout(location = 1) in inblock {         // ERROR
    float f1;
    layout(location = 3) float f2;        // ERROR
};

layout(location = 1) uniform ublock {     // ERROR
    float f1;
    layout(location = 3) float f2;        // ERROR
} uinst;

#extension GL_ARB_enhanced_layouts : enable

layout(location = start) in vec4 v3;
layout(location = -2) in vec4 v4;         // ERROR
layout(location = -start) in vec4 v5;     // ERROR
layout(location = start*start - 2 - 4) in vec4 v6;
layout(location = +61) in float v23;
layout(location = (62)) in float v24;

struct S2 {
    float f1;
    layout(location = 3) float f2;        // ERROR
};

layout(location = 28) in inblock2 {
    bool b1;
    float f1;
    layout(location = 25) float f2;
    vec4 f3;
    layout(location = 21) S2 s2;
    vec4 f4;
    vec4 f5;
} ininst2;

layout(location = 13) uniform ublock2 {   // ERROR
    float f1;
    layout(location = 3) float f2;        // ERROR
} uinst2;

in inblock3 {                             // ERROR, mix of location internal with no location external
    float f1;
    layout(location = 40) float f2;
} in3;

in ublock4 {
    layout(location = 50) float f1;
    layout(location = 51) float f2;
} in4;

layout(location = 33) in struct SS {
    vec3 a;    // gets location 33
    mat2 b;    // gets locations 34 and 35
    vec4 c[2]; // gets locations 36 and 37
    layout (location = 38) vec2 A; // ERROR, can't use on struct member
} s;

layout(location = 44) in block {
    vec4 d; // gets location 44
    vec4 e; // gets location 45
    layout(location = 47) vec4 f; // gets location 47
    vec4 g; // gets location 48
    layout (location = 41) vec4 h; // gets location 41
    vec4 i; // gets location 42
    vec4 j; // gets location 43
    vec4 k; // ERROR, location 44 already used
};

layout(index=0) out vec4 outVar2; // ERROR: missing explicit location
layout(location=0, index=1) out vec4 outVar3; // no error even though location is overlapping
layout(location=0, index=1) out vec4 outVar4; // ERROR overlapping
layout(location=27, index=0) in vec4 indexIn; // ERROR, not on in
layout(location=0, index=0) in; // ERROR, not just on in
layout(location=0, index=0) out; // ERROR, need a variable
layout(location=26, index=0) out indexBlock { int a; } indexBlockI; // ERROR, not on a block

uniform sampler1D samp1D;
uniform sampler2DShadow samp2Ds;

void qlod()
{
    vec2 lod;
    float pf;
    vec2 pf2;
    vec3 pf3;

    lod = textureQueryLod(samp1D, pf);      // ERROR, not until 400
    lod = textureQueryLod(samp2Ds, pf2);    // ERROR, not until 400
}

int precise;                // okay, not a keyword yet
struct SKeyMem { int precise; } KeyMem; // okay, not a keyword yet

void fooKeyMem()
{
    KeyMem.precise;
}

layout(location=28, index=2) out vec4 outIndex2; // ERROR index out of range