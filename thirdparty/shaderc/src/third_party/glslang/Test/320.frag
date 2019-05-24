#version 320 es

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
uniform int i;
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

void main()
{
    gl_FragDepth = 0.2;  // ERROR, early_fragment_tests declared
    bool f = gl_FrontFacing;
}

out float gl_FragDepth;

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

void pfoo()
{
    precise vec2 h;
    h = fma(inf, ing, h);
    textureGatherOffset(sArray[0], vec2(0.1), ivec2(inf));
    textureGatherOffsets(sArray[0], vec2(0.1), constOffsets);
    textureGatherOffsets(sArray[0], vec2(0.1), offsets);       // ERROR, offset not constant
}

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

centroid out vec4 colorCentroidBad;  // ERROR
flat out vec4 colorBadFlat;          // ERROR
smooth out vec4 colorBadSmooth;      // ERROR
noperspective out vec4 colorBadNo;   // ERROR
flat centroid in vec2 colorfc;
in float scalarIn;

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
