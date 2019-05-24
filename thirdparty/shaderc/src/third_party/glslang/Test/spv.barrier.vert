#version 450

layout(location=0) out vec4 c0;
layout(location=1) out vec4 c1;

void main()
{
    c0 = vec4(1.0);
    memoryBarrier();
    c1 = vec4(1.0);
    memoryBarrierBuffer();
    ++c0;
    memoryBarrierImage();
    ++c0;
}