#version 320 es

precision mediump float;

in fromVertex {
    in vec3 color;
} fromV[];

in vec4 nonBlockUnsized[];

out toFragment {
    out vec3 color;
} toF;

out fromVertex {  // okay to reuse a block name for another block name
    vec3 color;
};

out fooB {        // ERROR, cannot reuse block name as block instance
    vec2 color;
} fromVertex;

int fromVertex;   // ERROR, cannot reuse a block name for something else

out fooC {        // ERROR, cannot have same name for block and instance name
    vec2 color;
} fooC;

void main()
{
    EmitVertex();
    EndPrimitive();
    EmitStreamVertex(1);    // ERROR
    EndStreamPrimitive(0);  // ERROR

    color = fromV[0].color;
    gl_ClipDistance[3] =              // ERROR, no ClipDistance
        gl_in[1].gl_ClipDistance[2];  // ERROR, no ClipDistance
    gl_Position = gl_in[0].gl_Position;

    gl_PrimitiveID = gl_PrimitiveIDIn;
    gl_Layer = 2;
}

layout(stream = 4) out vec4 ov4; // ERROR, no streams

layout(line_strip, points, triangle_strip, points, triangle_strip) out;  // just means triangle_strip"

out ooutb { vec4 a; } ouuaa6;

layout(max_vertices = 200) out;
layout(max_vertices = 300) out;   // ERROR, too big
void foo(layout(max_vertices = 4) int a)  // ERROR
{
    ouuaa6.a = vec4(1.0);
}

layout(line_strip, points, triangle_strip, points) out;  // ERROR, changing output primitive
layout(line_strip, points) out; // ERROR, changing output primitive
layout(triangle_strip) in; // ERROR, not an input primitive
layout(triangle_strip) uniform; // ERROR
layout(triangle_strip) out vec4 badv4;  // ERROR, not on a variable
layout(triangle_strip) in vec4 bad2v4[];  // ERROR, not on a variable or input
layout(invocations = 3) out outbn { int a; }; // 2 ERROR, not on a block, not until 4.0
out outbn2 {
    layout(invocations = 3)  int a; // 2 ERRORs, not on a block member, not until 4.0
    layout(max_vertices = 3) int b; // ERROR, not on a block member
    layout(triangle_strip)   int c; // ERROR, not on a block member
} outbi;

layout(lines) out;  // ERROR, not on output
layout(lines_adjacency) in;
layout(triangles) in;             // ERROR, can't change it
layout(triangles_adjacency) in;   // ERROR, can't change it
layout(invocations = 4) in;

in sameName {
    int a15;
} insn[];

out sameName {
    float f15;
};

uniform sameName {
    bool b15;
};

const int summ = gl_MaxVertexAttribs +
             gl_MaxGeometryInputComponents +
             gl_MaxGeometryOutputComponents +
             gl_MaxGeometryImageUniforms +
             gl_MaxGeometryTextureImageUnits +
             gl_MaxGeometryOutputVertices +
             gl_MaxGeometryTotalOutputComponents +
             gl_MaxGeometryUniformComponents +
             gl_MaxGeometryAtomicCounters +
             gl_MaxGeometryAtomicCounterBuffers +
             gl_MaxVertexTextureImageUnits +
             gl_MaxCombinedTextureImageUnits +
             gl_MaxTextureImageUnits +
             gl_MaxDrawBuffers;

void fooe1()
{
    gl_ViewportIndex;  // ERROR, not in ES
    gl_MaxViewports;   // ERROR, not in ES
    insn.length();     // 4: lines_adjacency
    int inv = gl_InvocationID;
}

in vec4 explArray[4];
in vec4 explArrayBad[5];  // ERROR, wrong size
in vec4 nonArrayed;       // ERROR, not an array
flat out vec3 myColor1;
centroid out vec3 myColor2;
centroid in vec3 centr[];
sample out vec4 perSampleColor;  // ERROR without sample extensions

layout(max_vertices = 200) out;  // matching redecl

layout(location = 7, component = 2) in float comp[];  // ERROR, es has no component

void notHere()
{
    gl_MaxGeometryVaryingComponents;  // ERROR, not in ES
    gl_VerticesIn;                    // ERROR, not in ES
}

void pointSize2()
{
    highp float ps = gl_in[3].gl_PointSize;  // ERROR, need extension
    gl_PointSize = ps;                       // ERROR, need extension
}
