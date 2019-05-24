#version 450

#extension GL_NV_shading_rate_image : require

layout (location = 0) out vec2 FragmentSize;
layout (location = 2) out int InvocationsPerPixel;

void main () {
    FragmentSize = gl_FragmentSizeNV;
    InvocationsPerPixel = gl_InvocationsPerPixelNV;
}