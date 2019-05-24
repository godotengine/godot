#version 330 core
#extension GL_ARB_separate_shader_objects : enable

in gl_PerVertex
{
    float gl_ClipDistance[1];
    vec4 gl_Position;
} gl_in[];

out gl_PerVertex
{
    vec4 gl_Position;
    float gl_ClipDistance[1];
};

layout( triangles ) in;
layout( triangle_strip, max_vertices = 3 ) out;

void main()
{
    vec4 v;
    gl_Position = gl_in[1].gl_Position;
    gl_ClipDistance[0] = gl_in[1].gl_ClipDistance[0];
    EmitVertex();
    EndPrimitive();
}
