#version 400 core

layout(vertices = 4) out; // ERROR
layout(quads, cw) in;
layout(triangles) in;     // ERROR
layout(isolines) in;      // ERROR

layout(ccw) in;           // ERROR
layout(cw) in;

layout(fractional_odd_spacing) in;    
layout(equal_spacing) in;              // ERROR
layout(fractional_even_spacing) in;    // ERROR

layout(point_mode) in;

patch in vec4 patchIn;
patch out vec4 patchOut;  // ERROR

void main()
{
    barrier(); // ERROR

    int a = gl_MaxTessEvaluationInputComponents +
            gl_MaxTessEvaluationOutputComponents +
            gl_MaxTessEvaluationTextureImageUnits +
            gl_MaxTessEvaluationUniformComponents +
            gl_MaxTessPatchComponents +
            gl_MaxPatchVertices +
            gl_MaxTessGenLevel;

    vec4 p = gl_in[1].gl_Position;
    float ps = gl_in[1].gl_PointSize;
    float cd = gl_in[1].gl_ClipDistance[2];

    int pvi = gl_PatchVerticesIn;
    int pid = gl_PrimitiveID;
    vec3 tc = gl_TessCoord;
    float tlo = gl_TessLevelOuter[3];
    float tli = gl_TessLevelInner[1];

    gl_Position = p;
    gl_PointSize = ps;
    gl_ClipDistance[2] = cd;
}

smooth patch in vec4 badp1;         // ERROR
flat patch in vec4 badp2;           // ERROR
noperspective patch in vec4 badp3;  // ERROR
patch sample in vec3 badp4;         // ERROR

#extension GL_ARB_separate_shader_objects : enable

in gl_PerVertex
{
    float gl_ClipDistance[1];
} gl_in[];

in gl_PerVertex            // ERROR, second redeclaration of gl_in
{
    float gl_ClipDistance[1];
} gl_in[];

layout(quads, cw) out;     // ERROR
layout(triangles) out;     // ERROR
layout(isolines) out;      // ERROR
layout(cw) out;            // ERROR
layout(fractional_odd_spacing) out;    // ERROR
layout(equal_spacing) out;             // ERROR
layout(fractional_even_spacing) out;   // ERROR
layout(point_mode) out;                // ERROR

in vec2 ina;   // ERROR, not array
in vec2 inb[];
in vec2 inc[18];  // ERROR, wrong size
in vec2 ind[gl_MaxPatchVertices];

in testbla {
    int f;
} bla;        // ERROR, not array

in testblb {
    int f;
} blb[];

in testblc {
    int f;
} blc[18]; // ERROR wrong size

in testbld {
    int f;
} bld[gl_MaxPatchVertices];

layout(location = 23) in vec4 ivla[];
layout(location = 24) in vec4 ivlb[];
layout(location = 24) in vec4 ivlc[];  // ERROR

layout(location = 23) out vec4 ovla[2];
layout(location = 24) out vec4 ovlb[2];  // ERROR

in float gl_TessLevelOuter[4];           // ERROR, can't redeclare

patch in pinbn {
    int a;
} pinbi;

void devi()
{
    gl_DeviceIndex; // ERROR, no extension
    gl_ViewIndex;   // ERROR, no extension
}

#ifdef GL_EXT_device_group
#extension GL_EXT_device_group : enable
#endif

#ifdef GL_EXT_device_group
#extension GL_EXT_multiview : enable
#endif

void devie()
{
    gl_DeviceIndex;
    gl_ViewIndex;
}
