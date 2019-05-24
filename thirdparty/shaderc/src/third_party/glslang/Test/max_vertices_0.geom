#version 330

layout(points) in;
layout(triangle_strip, max_vertices = 0) out;
in highp vec4 v_geom_FragColor[];
out highp vec4 v_frag_FragColor;

void main (void)
{
    EndPrimitive();
    EndPrimitive();
}
