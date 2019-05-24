#version 420 core

layout(depth_any) out float gl_FragDepth;
layout(depth_greater) out float gl_FragDepth; // ERROR: redeclaration with different qualifier

void main()
{
    gl_FragDepth = 0.3;
}

layout(depth_less) in float depth; // ERROR: depth_less only applies to gl_FragDepth
layout(depth_any) out float gl_FragDepth;  // ERROR, done after use

layout(binding=0) uniform atomic_uint a[];
