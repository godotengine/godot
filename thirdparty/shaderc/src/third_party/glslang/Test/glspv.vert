#version 450

layout(push_constant) uniform Material { int a; } mat;            // ERROR, can't use push_constant

layout(set = 0, binding = 0, std140) uniform Bt1 { int a; } bt1;
layout(set = 1, binding = 0, std140) uniform Bt2 { int a; } bt2;  // ERROR, set has to be 0

layout(shared) uniform Bt3 { int a; } bt3;                        // ERROR, no shared, no binding
layout(packed) uniform Bt4 { int a; } bt4;                        // ERROR, no shared, no binding

void main()
{
    gl_VertexIndex;   // ERROR, not preset
    gl_InstanceIndex; // ERROR, not present
    gl_VertexID;
    gl_InstanceID;
    gl_DepthRangeParameters; // ERROR, not present
}

uniform sampler s; // ERROR, no sampler
