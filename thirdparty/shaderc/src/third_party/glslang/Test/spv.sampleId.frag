#version 450

layout(location = 0) in vec4 samp;
layout(location = 0) out vec4 color;

void main()
{
    if (gl_SampleID < 3)
        color = samp;
    else
        color = 2 * samp;
}