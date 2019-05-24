#version 450

#extension GL_NVX_multiview_per_view_attributes :require

void main()
{
    gl_ViewportMaskPerViewNV[0]    = 1;
    gl_PositionPerViewNV[0]        = gl_Position;
}

