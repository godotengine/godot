#version 450 core

in gl_PerVertex {
    float gl_CullDistance[3];
} gl_in[gl_MaxPatchVertices];

out gl_PerVertex {
    float gl_CullDistance[3];
} gl_out[4];

void main()
{
    gl_out[gl_InvocationID].gl_CullDistance[2] = gl_in[1].gl_CullDistance[2];
}

layout(location = 4) out bName1 {
    float f;
    layout(location = 5) float g;
} bInst1[2];
layout(location = 6) out bName2 {
    float f;
    layout(location = 7) float g;  // ERROR, location on array
} bInst2[2][3];
