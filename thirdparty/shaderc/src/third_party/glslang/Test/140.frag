#version 140

varying vec4 v;

in vec4 i;
out vec4 o;

in float gl_ClipDistance[5];

void main()
{
    float clip = gl_ClipDistance[2];
}
#ifdef GL_ES
#error GL_ES is set
#else
#error GL_ES is not set
#endif

in struct S { float f; } s; // ERROR

float patch = 3.1;

layout(location=3) in vec4 vl;  // ERROR

layout(location = 3) out vec4 factorBad;  // ERROR

#extension GL_ARB_explicit_attrib_location : enable

layout(location = 5) out vec4 factor;

#extension GL_ARB_separate_shader_objects : enable

layout(location=4) in vec4 vl2;

float fooi();

void foo()
{
    vec2 r1 = modf(v.xy, v.zw);  // ERROR, v.zw not l-value
    vec2 r2 = modf(o.xy, o.zw);
    o.z = fooi();
}

// Test extra-function initializers

float i1 = gl_FrontFacing ? -2.0 : 2.0;
float i2 = 102;

float fooi()
{
    return i1 + i2;
}
