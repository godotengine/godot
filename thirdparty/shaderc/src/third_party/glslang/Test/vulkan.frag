#version 450

uniform sampler s;         // ERROR, no binding
uniform sampler sA[4];     // ERROR, no binding
uniform texture2D t2d;     // ERROR, no binding
uniform texture3D t3d[4];  // ERROR, no binding
int i;
uniform samplerShadow sShadow;
uniform texture3D t3d5[5];
writeonly uniform image2D i2d;

void badConst()
{
    sampler2D(t2d);       // ERROR, need 2 args
    sampler2D(s, s);      // ERROR, wrong type
    sampler2D(i, i);      // ERROR, wrong type
    sampler2D(t2d, i);    // ERROR, wrong type
    sampler2D(t2d, t2d);  // ERROR, wrong type
    sampler2D(t2d, sA);   // ERROR, wrong type

    sampler3D[4](t3d5, sA[2]);    // ERROR, can't make array
    sampler2D(i2d, s);            // ERROR, image instead of texture
    sampler2D(t3d[1], s);         // ERROR, 3D not 2D
    sampler2D(t2d, sShadow);
    sampler2DShadow(t2d, s);
}

sampler2D s2D = sampler2D(t2d, s);            // ERROR, no sampler constructor
sampler3D s3d[4] = sampler3D[4](t3d, sA[2]);  // ERROR, no sampler constructor

out vec4 color; // ERROR, no location

void main()
{
    color = texture(s2D, vec2(0.5));
    color += texture(s3d[i], vec3(0.5));
}

layout(push_constant) buffer pcb {            // ERROR, not on a buffer
    int a;
} pcbInst;

layout(push_constant) uniform float pcfloat;  // ERROR 2X: not on a non-block, and non-opaque outside block

layout(push_constant) uniform;                // ERROR, needs an object
layout(std430, push_constant) uniform pcb1 { int a; } pcb1inst;
layout(push_constant) uniform pcb2 {
    int a;
};                                            // Okay now to have no instance name

layout(input_attachment_index = 2) uniform subpassInput subD;
layout(input_attachment_index = 3) uniform texture2D subDbad1;          // ERROR, not a texture
layout(input_attachment_index = 4) writeonly uniform image2D subDbad2;  // ERROR, not an image
uniform subpassInput subDbad3;                                          // ERROR, need attachment number
layout(input_attachment_index = 2) uniform subpassInputMS subDMS;

void foo()
{
    vec4 v = subpassLoad(subD);
    v += subpassLoadMS(subD);      // ERROR, no such function
    v += subpassLoad(subD, 2);     // ERROR, no such sig.
    v += subpassLoad(subDMS, 2);
    v += subpassLoadMS(subDMS, 2); // ERROR, no such function
}

subroutine int fooS;                              // ERROR, not in SPV
subroutine int fooSub();                          // ERROR, not in SPV

uniform vec4 dv4;                                 // ERROR, no default uniforms

void fooTex()
{
    texture(t2d, vec2(1.0));                 // ERROR, need a sampler, not a pure texture
    imageStore(t2d, ivec2(4, 5), vec4(1.2)); // ERROR, need an image, not a pure texture
}

precision highp float;

layout(location=0) in vec2 vTexCoord;
layout(location=0) out vec4 FragColor;

vec4 userTexture(mediump sampler2D samp, vec2 coord)
{
    return texture(samp, coord);
}

bool cond;

void callUserTexture()
{
    userTexture(sampler2D(t2d,s), vTexCoord);                            // ERROR, not point of use
    userTexture((sampler2D(t2d,s)), vTexCoord);                          // ERROR, not point of use
    userTexture((sampler2D(t2d,s), sampler2D(t2d,s)), vTexCoord);        // ERROR, not point of use
    userTexture(cond ? sampler2D(t2d,s) : sampler2D(t2d,s), vTexCoord);  // ERROR, no ?:, not point of use

    gl_NumSamples;   // ERROR, not for Vulkan
}

void noise()
{
    noise1(dv4);
    noise2(4.0);
    noise3(vec2(3));
    noise4(dv4);
}
