#version 450

layout(location = 0) in sample vec4 samp;
layout(location = 0) out vec4 color;

void main()
{
    color = samp;
}