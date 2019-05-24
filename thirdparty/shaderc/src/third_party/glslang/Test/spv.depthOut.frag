#version 450

in vec4 Color;
in float Depth;

layout(depth_greater) out float gl_FragDepth;

void main()
{
    gl_FragDepth = Depth;
}
