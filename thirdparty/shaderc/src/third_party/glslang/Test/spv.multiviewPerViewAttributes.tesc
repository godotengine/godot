#version 450

#extension GL_NVX_multiview_per_view_attributes :require

layout(vertices = 4) out;
out gl_PerVertex {
    int gl_ViewportMaskPerViewNV[];
    vec4 gl_PositionPerViewNV[];
 } gl_out[];
void main()
{
    gl_out[gl_InvocationID].gl_ViewportMaskPerViewNV[0]    = 1;
    gl_out[gl_InvocationID].gl_PositionPerViewNV[0]        =  gl_in[1].gl_Position;
}
