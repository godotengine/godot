#version 330 core

precision highp float;

layout(triangles) in;
layout(triangle_strip, max_vertices = 4) out;

in block
{
    vec2 Color;
    vec2 Texcoord;
    flat ivec3 in_a;
} In[];

out block
{
    vec4 Color;
    vec4 a;
    vec2 b[3];
} Out;

void main()
{
    for(int i = 0; i < gl_in.length(); ++i)
    {
        gl_Position = gl_in[i].gl_Position;
        Out.Color = vec4(In[i].Color, 0, 1);
        EmitVertex();
    }
    EndPrimitive();
}
