#version 320 es

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
    float ps = gl_in[1].gl_PointSize;        // ERROR, need point_size extension
    float cd = gl_in[1].gl_ClipDistance[2];  // ERROR, not in ES

    int pvi = gl_PatchVerticesIn;
    int pid = gl_PrimitiveID;
    vec3 tc = gl_TessCoord;
    float tlo = gl_TessLevelOuter[3];
    float tli = gl_TessLevelInner[1];

    gl_Position = p;
    gl_PointSize = ps;             // ERROR, need point_size extension
    gl_ClipDistance[2] = cd;       // ERROR, not in ES
}

smooth patch in vec4 badp1;         // ERROR
flat patch in vec4 badp2;           // ERROR
noperspective patch in vec4 badp3;  // ERROR
patch sample in vec3 badp4;         // ERROR

#extension GL_ARB_separate_shader_objects : enable

in gl_PerVertex
{
    vec4 gl_Position;
} gl_in[];

in gl_PerVertex           // ERROR, second redeclaration of gl_in
{
    vec4 gl_Position;
} gl_in[];

layout(quads, cw) out;     // ERROR
layout(triangles) out;     // ERROR
layout(isolines) out;      // ERROR
layout(cw) out;            // ERROR
layout(fractional_odd_spacing) out;    // ERROR
layout(equal_spacing) out;             // ERROR
layout(fractional_even_spacing) out;   // ERROR
layout(point_mode) out;                // ERROR

in vec2 ina;      // ERROR, not array
in vec2 inb[];
in vec2 inc[18];  // ERROR, wrong size
in vec2 ind[gl_MaxPatchVertices];

in testbla {      // ERROR, not array
    int f;
} bla;

in testblb {
    int f;
} blb[];

in testblc { // ERROR wrong size
    int f;
} blc[18];

in testbld {
    int f;
} bld[gl_MaxPatchVertices];

layout(location = 23) in vec4 ivla[];
layout(location = 24) in vec4 ivlb[];
layout(location = 24) in vec4 ivlc[];  // ERROR, overlap

layout(location = 23) out vec4 ovla[2];
layout(location = 24) out vec4 ovlb[2];  // ERROR, overlap

in float gl_TessLevelOuter[4];           // ERROR, can't redeclare

patch in pinbn {
    int a;
} pinbi;

centroid out vec3 myColor2;
centroid in vec3 centr[];
sample out vec4 perSampleColor;

void bbbad()
{
    gl_BoundingBoxOES; // ERROR, wrong stage
}
