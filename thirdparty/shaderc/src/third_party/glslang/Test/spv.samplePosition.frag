#version 450

layout(location = 0) in vec4 samp;
layout(location = 0) out vec4 color;

void main()
{
    if (gl_SamplePosition.y < 0.5)
        color = samp;
    else
        color = 2 * samp;
}