#version 450 core

layout(triangles) in;

layout(line_strip) out;
layout(max_vertices = 127) out;
layout(invocations = 4) in;

void main()
{
    gl_PointSize = gl_in[1].gl_PointSize;
    gl_Layer = 2;
    gl_ViewportIndex = 3;
}
