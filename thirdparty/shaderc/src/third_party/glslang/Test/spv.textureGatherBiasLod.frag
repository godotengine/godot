#version 450 core

#extension GL_ARB_sparse_texture2: enable
#extension GL_AMD_texture_gather_bias_lod: enable

uniform sampler2D           s2D;
uniform sampler2DArray      s2DArray;
uniform samplerCube         sCube;
uniform samplerCubeArray    sCubeArray;

in vec2 c2;
in vec3 c3;
in vec4 c4;

in float lod;
in float bias;

out vec4 fragColor;

void main()
{
    vec4 texel  = vec4(0.0);
    vec4 result = vec4(0.0);

    const ivec2 offsets[4] = { ivec2(0, 0), ivec2(0, 1), ivec2(1, 0), ivec2(1, 1) };

    texel += textureGather(s2D,        c2, 0, bias);
    texel += textureGather(s2DArray,   c3, 1, bias);
    texel += textureGather(sCube,      c3, 2, bias);
    texel += textureGather(sCubeArray, c4, 3, bias);

    texel += textureGatherOffset(s2D,        c2, offsets[0], 0, bias);
    texel += textureGatherOffset(s2DArray,   c3, offsets[1], 1, bias);

    texel += textureGatherOffsets(s2D,        c2, offsets, 0, bias);
    texel += textureGatherOffsets(s2DArray,   c3, offsets, 1, bias);

    sparseTextureGatherARB(s2D,        c2, result, 0, bias);
    texel += result;
    sparseTextureGatherARB(s2DArray,   c3, result, 1, bias);
    texel += result;
    sparseTextureGatherARB(sCube,      c3, result, 2, bias);
    texel += result;
    sparseTextureGatherARB(sCubeArray, c4, result, 2, bias);
    texel += result;

    sparseTextureGatherOffsetARB(s2D,      c2, offsets[0], result, 0, bias);
    texel += result;
    sparseTextureGatherOffsetARB(s2DArray, c3, offsets[1], result, 1, bias);
    texel += result;

    sparseTextureGatherOffsetsARB(s2D,      c2, offsets, result, 0, bias);
    texel += result;
    sparseTextureGatherOffsetsARB(s2DArray, c3, offsets, result, 1, bias);
    texel += result;

    texel += textureGatherLodAMD(s2D,        c2, lod);
    texel += textureGatherLodAMD(s2DArray,   c3, lod, 1);
    texel += textureGatherLodAMD(sCube,      c3, lod, 2);
    texel += textureGatherLodAMD(sCubeArray, c4, lod, 3);

    texel += textureGatherLodOffsetAMD(s2D,        c2, lod, offsets[0]);
    texel += textureGatherLodOffsetAMD(s2DArray,   c3, lod, offsets[1], 1);

    texel += textureGatherLodOffsetsAMD(s2D,       c2, lod, offsets);
    texel += textureGatherLodOffsetsAMD(s2DArray,  c3, lod, offsets, 1);

    sparseTextureGatherLodAMD(s2D,        c2, lod, result);
    texel += result;
    sparseTextureGatherLodAMD(s2DArray,   c3, lod, result, 1);
    texel += result;
    sparseTextureGatherLodAMD(sCube,      c3, lod, result, 2);
    texel += result;
    sparseTextureGatherLodAMD(sCubeArray, c4, lod, result, 2);
    texel += result;

    sparseTextureGatherLodOffsetAMD(s2D,      c2, lod, offsets[0], result);
    texel += result;
    sparseTextureGatherLodOffsetAMD(s2DArray, c3, lod, offsets[1], result, 1);
    texel += result;

    sparseTextureGatherLodOffsetsAMD(s2D,      c2, lod, offsets, result);
    texel += result;
    sparseTextureGatherLodOffsetsAMD(s2DArray, c3, lod, offsets, result, 1);
    texel += result;

    fragColor = texel;
}
