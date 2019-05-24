#version 450

#extension GL_NV_viewport_array2 :require
#extension GL_NV_stereo_view_rendering : require

layout(vertices = 4) out;

out gl_PerVertex {
    int gl_SecondaryViewportMaskNV[2];
    vec4 gl_SecondaryPositionNV;
} gl_out[4];

layout (viewport_relative, secondary_view_offset = 1) out highp int gl_Layer;

void main()
{
    gl_out[gl_InvocationID].gl_SecondaryViewportMaskNV[0]            = 1;
    gl_out[gl_InvocationID].gl_SecondaryPositionNV                   = gl_in[1].gl_Position;
}
