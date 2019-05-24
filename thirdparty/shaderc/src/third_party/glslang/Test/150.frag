#version 150 core

in vec4 gl_FragCoord;
layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;  // ERROR
layout(pixel_center_integer) in vec4 gl_FragCoord;  // ERROR
layout(origin_upper_left) in vec4 foo;  // ERROR
layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;

void main()
{
    vec4 c = gl_FragCoord;
}

layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;  // ERROR, declared after use

in struct S { float f; } s;

float patch = 3.1;

uniform sampler2DMS sms;
uniform isampler2DMS isms;
uniform usampler2DMS usms;
uniform sampler2DMSArray smsa;
uniform isampler2DMSArray ismsa;
uniform usampler2DMSArray usmsa;

flat in ivec2 p2;
flat in ivec3 p3;
flat in int samp;

void barWxyz()
{
    ivec2 t11 = textureSize( sms);
    ivec2 t12 = textureSize(isms);
    ivec2 t13 = textureSize(usms);
    ivec3 t21 = textureSize( smsa);
    ivec3 t22 = textureSize(ismsa);
    ivec3 t23 = textureSize(usmsa);
     vec4 t31 = texelFetch( sms, p2, samp);
    ivec4 t32 = texelFetch(isms, p2, samp);
    uvec4 t33 = texelFetch(usms, p2, 3);
     vec4 t41 = texelFetch( smsa, p3, samp);
    ivec4 t42 = texelFetch(ismsa, ivec3(2), samp);
    uvec4 t43 = texelFetch(usmsa, p3, samp);
}

int primitiveID()
{
   return gl_PrimitiveID;
}
