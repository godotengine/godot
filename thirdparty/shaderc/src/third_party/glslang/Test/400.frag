#version 400 core

in vec2 c2D;
flat in int i;
out vec4 outp;
uniform sampler2D arrayedSampler[5];
uniform usampler2DRect samp2dr;
uniform isampler2DArray isamp2DA;

void main()
{
    vec4 v;
    v = texture(arrayedSampler[i], c2D);
    outp.x = gl_ClipDistance[1];

    ivec2 offsets[4];
    const ivec2 constOffsets[4] = ivec2[4](ivec2(1,2), ivec2(3,4), ivec2(15,16), ivec2(-2,0));
    uvec4 uv4 = textureGatherOffsets(samp2dr, c2D, offsets, 2);  // ERROR, offsets not constant
    uv4 = textureGatherOffsets(samp2dr, c2D, constOffsets, 2);
    vec4 v4 = textureGather(arrayedSampler[0], c2D);
    ivec4 iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), 3);
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), i);  // ERROR, last argument not const
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), 4);  // ERROR, last argument out of range
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(1), 1+2);
    iv4 = textureGatherOffset(isamp2DA, vec3(0.1), ivec2(i));

    vec4 c = gl_FragCoord;
}

layout(location = 4) in vec4 vl; // ERROR, not supported

#ifdef GL_ARB_separate_shader_objects
#extension GL_ARB_separate_shader_objects : enable
#endif

layout(location = 6) in vec4 vl2;

layout(location = 3) uniform vec3 uv3;

layout(location = 5) in vec4 gl_Color;      // ERROR, layout
noperspective in float gl_ClipDistance[4];  // ERROR, can't change qualifier

layout(origin_upper_left, pixel_center_integer) in vec4 gl_FragCoord;  // ERROR, declared after use

uniform sampler2DRectShadow u2drs;

void foo23()
{
    const ivec2[3] offsets = ivec2[3](ivec2(1,2), ivec2(3,4), ivec2(15,16));

    textureProjGradOffset(u2drs, outp, vec2(0.0), vec2(0.0), ivec2(c2D));     // ERROR, offset not constant
    textureProjGradOffset(u2drs, outp, vec2(0.0), vec2(0.0), offsets[1]);
    textureProjGradOffset(u2drs, outp, vec2(0.0), vec2(0.0), offsets[2]);     // ERROR, offset out of range
    textureProjGradOffset(u2drs, outp, vec2(0.0), vec2(0.0), ivec2(-10, 20)); // ERROR, offset out of range
}

patch in vec4 patchIn;              // ERROR
patch out vec4 patchOut;            // ERROR

void foo24()
{
    dvec3 df, di;
    df = modf(dvec3(outp.xyz), di);
}

in float in1;
in vec2 in2;
in vec3 in3;
in vec4 in4;

void foodc1()
{
    vec2 v2 = dFdxFine(in2);           // ERROR
    vec3 v3 = dFdyCoarse(in3);         // ERROR
    vec4 v4 = fwidthCoarse(in4) + fwidthFine(in4);   // ERROR
}

#extension GL_ARB_derivative_control : enable

void foodc2()
{
    vec2 v2 = dFdxFine(in2);
    vec3 v3 = dFdyCoarse(in3);
    vec4 v4 = fwidthCoarse(in4) + fwidthFine(in4);

    uint u1;
    ivec3 i3;
    ivec2 i2;
    v2 = frexp(v2, i2);
    v3 = ldexp(v3, i3);

    u1 = packUnorm4x8(v4);
    u1 = packSnorm4x8(v4);
    v4 = unpackUnorm4x8(u1);
    v4 = unpackSnorm4x8(u1);

    double d;
    uvec2 u2;
    d = packDouble2x32(u2);
    u2 = unpackDouble2x32(d);
}

sample in vec4 colorSampIn;
sample out vec4 colorSampleBad;     // ERROR
noperspective in vec4 colorfsi;
sample in vec3 sampInArray[4];
smooth in float scalarIn;
flat centroid in vec2 colorfc;

struct S {
    float x;
};

in S s1;
sample S s2;

void interp()
{
    interpolateAtCentroid(colorfc);
    interpolateAtCentroid(colorSampIn);
    interpolateAtCentroid(colorfsi);
    interpolateAtCentroid(scalarIn);
    interpolateAtCentroid(sampInArray);         // ERROR
    interpolateAtCentroid(sampInArray[2]);
    interpolateAtCentroid(sampInArray[2].xy);   // ERROR

    interpolateAtSample(sampInArray, 1);        // ERROR
    interpolateAtSample(sampInArray[i], 0);
    interpolateAtSample(s1.x, 2);
    interpolateAtSample(scalarIn, 1);

    interpolateAtOffset(sampInArray, vec2(0.2));         // ERROR
    interpolateAtOffset(sampInArray[2], vec2(0.2));
    interpolateAtOffset(sampInArray[2].xy, vec2(0.2));   // ERROR, no swizzle
    interpolateAtOffset(scalarIn + scalarIn, vec2(0.2)); // ERROR, no binary ops other than dereference
    interpolateAtOffset(s2.x, vec2(0.2));      // ERROR

    float f;
    interpolateAtCentroid(f);           // ERROR, not interpolant
    interpolateAtSample(outp, 0);       // ERROR, not interpolant
}

uniform sampler1D samp1D;
uniform isampler2D isamp2D;
uniform usampler3D usamp3D;
uniform samplerCube sampCube; 
uniform isampler1DArray isamp1DA;
uniform usampler2DArray usamp2DA;
uniform isamplerCubeArray isampCubeA;

uniform sampler1DShadow samp1Ds;
uniform sampler2DShadow samp2Ds;
uniform samplerCubeShadow sampCubes;
uniform sampler1DArrayShadow samp1DAs;
uniform sampler2DArrayShadow samp2DAs;
uniform samplerCubeArrayShadow sampCubeAs;

uniform samplerBuffer sampBuf;
uniform sampler2DRect sampRect;

void qlod()
{
    vec2 lod;
    float pf;
    vec2 pf2;
    vec3 pf3;

    lod = textureQueryLod(samp1D, pf);
    lod = textureQueryLod(isamp2D, pf2);
    lod = textureQueryLod(usamp3D, pf3);
    lod = textureQueryLod(sampCube, pf3);
    lod = textureQueryLod(isamp1DA, pf);
    lod = textureQueryLod(usamp2DA, pf2);
    lod = textureQueryLod(isampCubeA, pf3);

    lod = textureQueryLod(samp1Ds, pf);
    lod = textureQueryLod(samp2Ds, pf2);
    lod = textureQueryLod(sampCubes, pf3);
    lod = textureQueryLod(samp1DAs, pf);
    lod = textureQueryLod(samp2DAs, pf2);
    lod = textureQueryLod(sampCubeAs, pf3);

    lod = textureQueryLod(sampBuf, pf);     // ERROR
    lod = textureQueryLod(sampRect, pf2);   // ERROR
}

uniform uint uu;
out uint iout;

void bitwiseConv()
{
    iout = uu & i;
    iout += uu ^ i;
    iout += i | uu;
}

subroutine(subT1, subT2);
subroutine float subT1() { return 1.0; }
subroutine float subT2() { return 1.0; }

struct SKeyMem { int precise; } KeyMem;     // ERROR, keyword can't be a member
