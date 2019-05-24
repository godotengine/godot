#version 310 es
highp float nodef3(float); // ERROR, no default precision
precision mediump float;
precision highp usampler2D;
precision highp sampler2D;
precision highp isampler2DArray;

layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;  // ERROR, not supported

layout(location = 2) in vec3 v3;
layout(location = 2) in mat4 yi;  // ERROR, locations conflict with xi

uniform sampler2D arrayedSampler[5];
uniform usampler2D usamp2d;
uniform usampler2DRect samp2dr;      // ERROR, reserved
uniform isampler2DArray isamp2DA;

in vec2 c2D;
uniform int i;

void main()
{
    vec4 v = texture(arrayedSampler[i], c2D);  // ERROR

    ivec2 offsets[4];
    const ivec2 constOffsets[4] = ivec2[4](ivec2(1,2), ivec2(3,4), ivec2(15,16), ivec2(-2,0));
    uvec4 uv4 = textureGatherOffsets(samp2dr, c2D, offsets, 2);  // ERROR, not supported
    vec4 v4 = textureGather(arrayedSampler[0], c2D);
    ivec4 iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), 3);
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), i);  // ERROR, last argument not const
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), 4);  // ERROR, last argument out of range
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), 1+2);
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(0.5));
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(i));     // ERROR, offset not constant
}

out vec4 outp;
void foo23()
{
    const ivec2[3] offsets = ivec2[3](ivec2(1,2), ivec2(3,4), ivec2(15,16));

    textureProjGradOffset(usamp2d, outp, vec2(0.0), vec2(0.0), ivec2(c2D));     // ERROR, offset not constant
    textureProjGradOffset(usamp2d, outp, vec2(0.0), vec2(0.0), offsets[1]);
    textureProjGradOffset(usamp2d, outp, vec2(0.0), vec2(0.0), offsets[2]);     // ERROR, offset out of range
    textureProjGradOffset(usamp2d, outp, vec2(0.0), vec2(0.0), ivec2(-10, 20)); // ERROR, offset out of range

    if (gl_HelperInvocation)
        ++outp;

    int sum = gl_MaxVertexImageUniforms +
              gl_MaxFragmentImageUniforms +
              gl_MaxComputeImageUniforms +
              gl_MaxCombinedImageUniforms +
              gl_MaxCombinedShaderOutputResources;

    bool b1, b2, b3, b;

    b1 = mix(b2, b3, b);
    uvec3 um3 = mix(uvec3(i), uvec3(i), bvec3(b));
    ivec4 im4 = mix(ivec4(i), ivec4(i), bvec4(b));
}

layout(binding=3) uniform sampler2D s1;
layout(binding=3) uniform sampler2D s2; // ERROR: overlapping bindings?  Don't see that in the 310 spec.
highp layout(binding=2) uniform writeonly image2D      i2D;
      layout(binding=4) uniform readonly  image3D      i3D;    // ERROR, no default precision
      layout(binding=5) uniform           imageCube    iCube;  // ERROR, no default precision
      layout(binding=6) uniform           image2DArray i2DA;   // ERROR, no default precision
      layout(binding=6) uniform coherent volatile restrict image2D i2Dqualified;    // ERROR, no default precision

layout(binding = 1) uniform bb {
    int foo;
    layout(binding = 2) float f;     // ERROR
} bbi;

in centroid vec4 centroidIn;
layout(location = 200000) uniform vec4 bigl;  // ERROR, location too big

layout(early_fragment_tests) in;

layout(location = 40) out vec4 bigout1;  // ERROR, too big
layout(location = 40) out vec4 bigout2;  // ERROR, overlap
layout(location = -2) out vec4 neg;      // ERROR, negative

layout(std430) buffer b430 {
    int i;
} b430i;

layout(shared) uniform bshar {
    int i;
} bshari;

in smooth vec4 smoothIn;
in flat int flatIn;

uniform sampler2DMS s2dms;  // ERROR, no default precision qualifier

void foots()
{
    highp ivec2 v2 = textureSize(s1, 2);
    highp ivec3 v3 = textureSize(isamp2DA, 3);
    v2 = textureSize(s2dms);
    v2 = imageSize(i2D);
    v3 = imageSize(i3D);
    v2 = imageSize(iCube);
    v3 = imageSize(i2DA);
    v2 = imageSize(i2Dqualified);
}

out bool bout;          // ERROR
highp out image2D imageOut;   // ERROR
out mat2x3 mout;        // ERROR

in bool inb;         // ERROR
in sampler2D ino;    // ERROR
in float ina[4];
in float inaa[4][2]; // ERROR
struct S { float f; };
in S ins;
in S[4] inasa;       // ERROR
in S insa[4];        // ERROR
struct SA { float f[4]; };
in SA inSA;          // ERROR
struct SS { float f; S s; };
in SS inSS;          // ERROR

#ifndef GL_EXT_shader_io_blocks
#error GL_EXT_shader_io_blocks not defined
#endif

#extension GL_EXT_shader_io_blocks : enable

out outbname { int a; } outbinst;   // ERROR, not out block in fragment shader

in inbname {
    int a;
    vec4 v;
    struct { int b; } s;     // ERROR, nested struct definition
} inbinst;

in inbname2 {
    layout(location = 12) int aAnon;
    layout(location = 13) centroid in vec4 vAnon;
};

in layout(location = 13) vec4 aliased; // ERROR, aliased

in inbname2 {                // ERROR, reuse of block name
    int aAnon;
    centroid in vec4 vAnon;
};

in badmember {               // ERROR, aAnon already in global scope
    int aAnon;
};

int inbname;                 // ERROR, redefinition of block name

vec4 vAnon;                  // ERROR, anon in global scope; redefinition

in arrayed {
    float f;
} arrayedInst[4];

void fooIO()
{
    vec4 v = inbinst.v + vAnon;
    v *= arrayedInst[2].f;
    v *= arrayedInst[i].f;
}

in vec4 gl_FragCoord;
layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;  // ERROR, non-ES

layout(early_fragment_tests) in;
out float gl_FragDepth;
layout(depth_any) out float gl_FragDepth;  // ERROR, non-ES

void foo_IO()
{
    gl_FragDepth = 0.2;  // ERROR, early_fragment_tests declared
    gl_Layer;            // ERROR, not present
    gl_PrimitiveID;      // ERROR, not present
    bool f = gl_FrontFacing;
}

out float gl_FragDepth;

#extension GL_OES_geometry_shader : enable

void foo_GS()
{
    highp int l = gl_Layer;
    highp int p = gl_PrimitiveID;
}

in vec2 inf, ing;
uniform ivec2 offsets[4];
uniform sampler2D sArray[4];
uniform int sIndex;
layout(binding = 0) uniform atomic_uint auArray[2];
uniform ubName { int i; } ubInst[4];
buffer bbName { int i; } bbInst[4];
highp uniform writeonly image2D iArray[5];
const ivec2 constOffsets[4] = ivec2[4](ivec2(0.1), ivec2(0.2), ivec2(0.3), ivec2(0.4));

void pfooBad()
{
    precise vec2 h;                                            // ERROR reserved
    h = fma(inf, ing, h);                                      // ERROR, not available
    textureGatherOffset(sArray[0], vec2(0.1), ivec2(inf));     // ERROR, offset not constant
    textureGatherOffsets(sArray[0], vec2(0.1), constOffsets);  // ERROR, not available
}

#extension GL_OES_gpu_shader5 : enable

void pfoo()
{
    precise vec2 h;
    h = fma(inf, ing, h);
    textureGatherOffset(sArray[0], vec2(0.1), ivec2(inf));
    textureGatherOffsets(sArray[0], vec2(0.1), constOffsets);
    textureGatherOffsets(sArray[0], vec2(0.1), offsets);       // ERROR, offset not constant
}

#extension GL_EXT_texture_cube_map_array : enable

precision highp imageCubeArray        ;
precision highp iimageCubeArray       ;
precision highp uimageCubeArray       ;

precision highp samplerCubeArray      ;
precision highp samplerCubeArrayShadow;
precision highp isamplerCubeArray     ;
precision highp usamplerCubeArray     ;

uniform writeonly imageCubeArray  CA1;
uniform writeonly iimageCubeArray CA2;
uniform writeonly uimageCubeArray CA3;

#ifdef GL_EXT_texture_cube_map_array
uniform samplerCubeArray          CA4;
uniform samplerCubeArrayShadow    CA5;
uniform isamplerCubeArray         CA6;
uniform usamplerCubeArray         CA7;
#endif

void CAT()
{
    highp vec4 b4 = texture(CA4, vec4(0.5), 0.24);
    highp ivec4 b6 = texture(CA6, vec4(0.5), 0.26);
    highp uvec4 b7 = texture(CA7, vec4(0.5), 0.27);
}

void badSample()
{
    lowp     int  a1 = gl_SampleID;         // ERROR, need extension
    mediump  vec2 a2 = gl_SamplePosition;   // ERROR, need extension
    highp    int  a3 = gl_SampleMaskIn[0];  // ERROR, need extension
    gl_SampleMask[0] = a3;                  // ERROR, need extension
    mediump int n = gl_NumSamples;          // ERROR, need extension
}

#ifdef GL_OES_sample_variables
#extension GL_OES_sample_variables : enable
#endif

void goodSample()
{
    lowp     int  a1 = gl_SampleID;       
    mediump  vec2 a2 = gl_SamplePosition; 
    highp    int  a3 = gl_SampleMaskIn[0];
    gl_SampleMask[0] = a3;
    mediump int n1 = gl_MaxSamples;
    mediump int n2 = gl_NumSamples;
}

uniform layout(r32f)  highp  image2D im2Df;
uniform layout(r32ui) highp uimage2D im2Du;
uniform layout(r32i)  highp iimage2D im2Di;
uniform ivec2 P;

void badImageAtom()
{
    float datf;
    int dati;
    uint datu;

    imageAtomicAdd(     im2Di, P, dati);        // ERROR, need extension
    imageAtomicAdd(     im2Du, P, datu);        // ERROR, need extension
    imageAtomicMin(     im2Di, P, dati);        // ERROR, need extension
    imageAtomicMin(     im2Du, P, datu);        // ERROR, need extension
    imageAtomicMax(     im2Di, P, dati);        // ERROR, need extension
    imageAtomicMax(     im2Du, P, datu);        // ERROR, need extension
    imageAtomicAnd(     im2Di, P, dati);        // ERROR, need extension
    imageAtomicAnd(     im2Du, P, datu);        // ERROR, need extension
    imageAtomicOr(      im2Di, P, dati);        // ERROR, need extension
    imageAtomicOr(      im2Du, P, datu);        // ERROR, need extension
    imageAtomicXor(     im2Di, P, dati);        // ERROR, need extension
    imageAtomicXor(     im2Du, P, datu);        // ERROR, need extension
    imageAtomicExchange(im2Di, P, dati);        // ERROR, need extension
    imageAtomicExchange(im2Du, P, datu);        // ERROR, need extension
    imageAtomicExchange(im2Df, P, datf);        // ERROR, need extension
    imageAtomicCompSwap(im2Di, P,  3, dati);    // ERROR, need extension
    imageAtomicCompSwap(im2Du, P, 5u, datu);    // ERROR, need extension
}

#ifdef GL_OES_shader_image_atomic 
#extension GL_OES_shader_image_atomic : enable
#endif

uniform layout(rgba32f)  highp  image2D badIm2Df;  // ERROR, needs readonly or writeonly
uniform layout(rgba8ui) highp uimage2D badIm2Du;   // ERROR, needs readonly or writeonly
uniform layout(rgba16i)  highp iimage2D badIm2Di;  // ERROR, needs readonly or writeonly

void goodImageAtom()
{
    float datf;
    int dati;
    uint datu;

    imageAtomicAdd(     im2Di, P, dati);
    imageAtomicAdd(     im2Du, P, datu);
    imageAtomicMin(     im2Di, P, dati);
    imageAtomicMin(     im2Du, P, datu);
    imageAtomicMax(     im2Di, P, dati);
    imageAtomicMax(     im2Du, P, datu);
    imageAtomicAnd(     im2Di, P, dati);
    imageAtomicAnd(     im2Du, P, datu);
    imageAtomicOr(      im2Di, P, dati);
    imageAtomicOr(      im2Du, P, datu);
    imageAtomicXor(     im2Di, P, dati);
    imageAtomicXor(     im2Du, P, datu);
    imageAtomicExchange(im2Di, P, dati);
    imageAtomicExchange(im2Du, P, datu);
    imageAtomicExchange(im2Df, P, datf);
    imageAtomicCompSwap(im2Di, P,  3, dati);
    imageAtomicCompSwap(im2Du, P, 5u, datu);

    imageAtomicMax(badIm2Di, P, dati);      // ERROR, not an allowed layout() on the image
    imageAtomicMax(badIm2Du, P, datu);      // ERROR, not an allowed layout() on the image
    imageAtomicExchange(badIm2Df, P, datf); // ERROR, not an allowed layout() on the image
}

sample in vec4 colorSampInBad;       // ERROR, reserved
centroid out vec4 colorCentroidBad;  // ERROR
flat out vec4 colorBadFlat;          // ERROR
smooth out vec4 colorBadSmooth;      // ERROR
noperspective out vec4 colorBadNo;   // ERROR
flat centroid in vec2 colorfc;
in float scalarIn;

void badInterp()
{
    interpolateAtCentroid(colorfc);             // ERROR, need extension
    interpolateAtSample(colorfc, 1);            // ERROR, need extension
    interpolateAtOffset(colorfc, vec2(0.2));    // ERROR, need extension
}

#if defined GL_OES_shader_multisample_interpolation
#extension GL_OES_shader_multisample_interpolation : enable
#endif

sample in vec4 colorSampIn;
sample out vec4 colorSampleBad;     // ERROR
flat sample in vec4 colorfsi;
sample in vec3 sampInArray[4];

void interp()
{
    float res;
    vec2 res2;
    vec3 res3;
    vec4 res4;

    res2 = interpolateAtCentroid(colorfc);
    res4 = interpolateAtCentroid(colorSampIn);
    res4 = interpolateAtCentroid(colorfsi);
    res  = interpolateAtCentroid(scalarIn);
    res3 = interpolateAtCentroid(sampInArray);         // ERROR
    res3 = interpolateAtCentroid(sampInArray[2]);
    res2 = interpolateAtCentroid(sampInArray[2].xy);   // ERROR

    res3 = interpolateAtSample(sampInArray, 1);        // ERROR
    res3 = interpolateAtSample(sampInArray[i], 0);
    res2 = interpolateAtSample(sampInArray[2].xy, 2);  // ERROR
    res  = interpolateAtSample(scalarIn, 1);

    res3 = interpolateAtOffset(sampInArray, vec2(0.2));         // ERROR
    res3 = interpolateAtOffset(sampInArray[2], vec2(0.2));
    res2 = interpolateAtOffset(sampInArray[2].xy, vec2(0.2));   // ERROR, no swizzle
    res  = interpolateAtOffset(scalarIn + scalarIn, vec2(0.2)); // ERROR, no binary ops other than dereference
    res  = interpolateAtOffset(scalarIn, vec2(0.2));

    float f;
    res  = interpolateAtCentroid(f);           // ERROR, not interpolant
    res4 = interpolateAtSample(outp, 0);       // ERROR, not interpolant
}

layout(blend_support_softlight) out;           // ERROR, need extension

#ifdef GL_KHR_blend_equation_advanced
#extension GL_KHR_blend_equation_advanced : enable
#endif

layout(blend_support_multiply) out;
layout(blend_support_screen) out;
layout(blend_support_overlay) out;
layout(blend_support_darken, blend_support_lighten) out;
layout(blend_support_colordodge) layout(blend_support_colorburn) out;
layout(blend_support_hardlight) out;
layout(blend_support_softlight) out;
layout(blend_support_difference) out;
layout(blend_support_exclusion) out;
layout(blend_support_hsl_hue) out;
layout(blend_support_hsl_saturation) out;
layout(blend_support_hsl_color) out;
layout(blend_support_hsl_luminosity) out;
layout(blend_support_all_equations) out;

layout(blend_support_hsl_luminosity) out;              // okay to repeat

layout(blend_support_hsl_luminosity) in;                       // ERROR, only on "out"
layout(blend_support_hsl_luminosity) out vec4;                 // ERROR, only on standalone
layout(blend_support_hsl_luminosity) out vec4 badout;          // ERROR, only on standalone
layout(blend_support_hsl_luminosity) struct badS {int i;};     // ERROR, only on standalone
layout(blend_support_hsl_luminosity) void blendFoo() { }       // ERROR, only on standalone
void blendFoo(layout(blend_support_hsl_luminosity) vec3 v) { } // ERROR, only on standalone
layout(blend_support_flizbit) out;                             // ERROR, no flizbit

out vec4 outAA[2][2];  // ERROR

void devi()
{
    gl_DeviceIndex; // ERROR, no extension
    gl_ViewIndex;   // ERROR, no extension
}

#ifdef GL_EXT_device_group
#extension GL_EXT_device_group : enable
#endif

#ifdef GL_EXT_device_group
#extension GL_EXT_multiview : enable
#endif

void devie()
{
    gl_DeviceIndex;
    gl_ViewIndex;
}
