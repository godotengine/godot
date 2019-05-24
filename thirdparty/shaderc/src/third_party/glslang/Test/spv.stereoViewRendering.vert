#version 450

#extension GL_NV_viewport_array2 :require
#extension GL_NV_stereo_view_rendering : require

layout (viewport_relative, secondary_view_offset = 2) out highp int gl_Layer;
void main()
{
    gl_SecondaryViewportMaskNV[0] = 1;
    gl_SecondaryPositionNV        = gl_Position;
}

