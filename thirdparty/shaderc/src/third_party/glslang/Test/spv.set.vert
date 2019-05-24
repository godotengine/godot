#version 450

layout(set = 4, binding = 7) uniform sampler2D samp2D;

layout(set = 0, binding = 8) buffer setBuf {
    vec4 color;
} setBufInst;

out vec4 color;

void main()
{
    color = setBufInst.color;
}
