#version 450
#extension GL_NV_viewport_array2 :require

layout(vertices = 4) out;

out gl_PerVertex {
    int gl_ViewportMask[2];
} gl_out[4];

layout (viewport_relative) out highp int gl_Layer;

void main()
{
    gl_out[gl_InvocationID].gl_ViewportMask[0] = 1;
    gl_ViewportIndex = 2;
}
