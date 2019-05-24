#version 430 core

uniform sampler1D samp1D;
uniform isampler2D isamp2D;
uniform usampler2D usamp2D;
uniform isampler3D isamp3D;
uniform usampler3D usamp3D;
uniform samplerCube sampCube; 
uniform isamplerCube isampCube; 
uniform isampler1DArray isamp1DA;
uniform sampler2DArray samp2DA;
uniform usampler2DArray usamp2DA;
uniform isamplerCubeArray isampCubeA;
uniform usamplerCubeArray usampCubeA;

uniform sampler1DShadow samp1Ds;
uniform sampler2DShadow samp2Ds;
uniform samplerCubeShadow sampCubes;
uniform sampler1DArrayShadow samp1DAs;
uniform sampler2DArrayShadow samp2DAs;
uniform samplerCubeArrayShadow sampCubeAs;

uniform samplerBuffer sampBuf;
uniform sampler2DRect sampRect;

void main()
{
    vec2 lod;
    float pf;
    vec2 pf2;
    vec3 pf3;

    lod = textureQueryLod(samp1D, pf);
    lod += textureQueryLod(isamp2D, pf2);
    lod += textureQueryLod(usamp3D, pf3);
    lod += textureQueryLod(sampCube, pf3);
    lod += textureQueryLod(isamp1DA, pf);
    lod += textureQueryLod(usamp2DA, pf2);
    lod += textureQueryLod(isampCubeA, pf3);

    lod += textureQueryLod(samp1Ds, pf);
    lod += textureQueryLod(samp2Ds, pf2);
    lod += textureQueryLod(sampCubes, pf3);
    lod += textureQueryLod(samp1DAs, pf);
    lod += textureQueryLod(samp2DAs, pf2);
    lod += textureQueryLod(sampCubeAs, pf3);

    int levels;

    levels = textureQueryLevels(samp1D);
    levels += textureQueryLevels(usamp2D);
    levels += textureQueryLevels(isamp3D);
    levels += textureQueryLevels(isampCube);
    levels += textureQueryLevels(isamp1DA);
    levels += textureQueryLevels(samp2DA);
    levels += textureQueryLevels(usampCubeA);

    levels = textureQueryLevels(samp1Ds);
    levels += textureQueryLevels(samp2Ds);
    levels += textureQueryLevels(sampCubes);
    levels += textureQueryLevels(samp1DAs);
    levels += textureQueryLevels(samp2DAs);
    levels += textureQueryLevels(sampCubeAs);
}
