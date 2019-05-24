#version 450
#extension GL_ARB_shader_viewport_layer_array : require
#extension GL_NV_viewport_array2 : require

layout (viewport_relative) out highp int gl_Layer;
void main()
{
    gl_ViewportMask[0]            = 1;
    gl_ViewportIndex              = 2;
}