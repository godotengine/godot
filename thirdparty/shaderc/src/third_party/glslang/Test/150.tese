#version 150

#extension GL_ARB_tessellation_shader : enable

layout(quads, cw) in;
layout(fractional_odd_spacing) in;    
layout(point_mode) in;
patch in vec4 patchIn;

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
