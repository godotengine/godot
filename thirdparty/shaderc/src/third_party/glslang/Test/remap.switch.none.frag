#version 450

precision highp float;

layout(location = 0) out mediump vec4 FragColor;
layout(location = 0) in vec4 in0;

void main()
{
    switch(int(in0.w)) {
    case 0: FragColor = vec4(in0.x + 0); break;
    case 1: FragColor = vec4(in0.y + 1); break;
    case 2: FragColor = vec4(in0.z + 2); break;
    default: FragColor = vec4(-1);
    }
}
