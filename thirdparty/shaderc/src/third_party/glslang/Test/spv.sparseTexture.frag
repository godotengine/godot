#version 450
#extension GL_ARB_sparse_texture2: enable

uniform sampler2D               s2D;
uniform sampler3D               s3D;
uniform sampler2DShadow         s2DShadow;
uniform samplerCubeShadow       sCubeShadow;
uniform sampler2DArrayShadow    s2DArrayShadow;
uniform sampler2DRectShadow     s2DRectShadow;
uniform samplerCubeArrayShadow  sCubeArrayShadow;
uniform sampler2DMS             s2DMS;

uniform isamplerCube            isCube;
uniform isampler2DArray         is2DArray;

uniform usamplerCubeArray       usCubeArray;
uniform usampler2DRect          us2DRect;

layout(rgba32f) uniform image2D i2D;
layout(rgba32i) uniform iimage3D ii3D;
layout(rgba32f) uniform image2DMS i2DMS;

in vec2 c2;
in vec3 c3;
in vec4 c4;

in flat ivec2 ic2;
in flat ivec3 ic3;

in flat ivec2 offsets[4];

out vec4 outColor;

void main()
{
    int   resident = 0;
    vec4  texel  = vec4(0.0);
    ivec4 itexel = ivec4(0);
    uvec4 utexel = uvec4(0);

    resident |= sparseTextureARB(s2D, c2, texel);
    resident |= sparseTextureARB(s3D, c3, texel, 2.0);
    resident |= sparseTextureARB(isCube, c3, itexel);
    resident |= sparseTextureARB(s2DShadow, c3, texel.x);
    resident |= sparseTextureARB(sCubeArrayShadow, c4, 1.0, texel.x);

    resident |= sparseTextureLodARB(s2D, c2, 2.0, texel);
    resident |= sparseTextureLodARB(usCubeArray, c4, 1.0, utexel);
    resident |= sparseTextureLodARB(s2DShadow, c3, 2.0, texel.y);

    resident |= sparseTextureOffsetARB(s3D, c3, ivec3(2), texel, 2.0);
    resident |= sparseTextureOffsetARB(us2DRect, c2, ivec2(3), utexel);
    resident |= sparseTextureOffsetARB(s2DArrayShadow, c4, ivec2(5), texel.z);

    resident |= sparseTexelFetchARB(s2D, ivec2(c2), 2, texel);
    resident |= sparseTexelFetchARB(us2DRect, ivec2(c2), utexel);
    resident |= sparseTexelFetchARB(s2DMS, ivec2(c2), 4, texel);

    resident |= sparseTexelFetchOffsetARB(s3D, ivec3(c3), 2, ivec3(4), texel);
    resident |= sparseTexelFetchOffsetARB(us2DRect, ivec2(c2), ivec2(3), utexel);

    resident |= sparseTextureLodOffsetARB(s2D, c2, 2.0, ivec2(5), texel);
    resident |= sparseTextureLodOffsetARB(is2DArray, c3, 2.0, ivec2(6), itexel);
    resident |= sparseTextureLodOffsetARB(s2DShadow, c3, 2.0, ivec2(7), texel.z);

    resident |= sparseTextureGradARB(s3D, c3, c3, c3, texel);
    resident |= sparseTextureGradARB(sCubeShadow, c4, c3, c3, texel.y);
    resident |= sparseTextureGradARB(usCubeArray, c4, c3, c3, utexel);

    resident |= sparseTextureGradOffsetARB(s2D, c2, c2, c2, ivec2(5), texel);
    resident |= sparseTextureGradOffsetARB(s2DRectShadow, c3, c2, c2, ivec2(6), texel.w);
    resident |= sparseTextureGradOffsetARB(is2DArray, c3, c2, c2, ivec2(2), itexel);

    resident |= sparseTextureGatherARB(s2D, c2, texel);
    resident |= sparseTextureGatherARB(is2DArray, c3, itexel, 2);
    resident |= sparseTextureGatherARB(s2DArrayShadow, c3, 2.0, texel);

    resident |= sparseTextureGatherOffsetARB(s2D, c2, ivec2(4), texel);
    resident |= sparseTextureGatherOffsetARB(is2DArray, c3, ivec2(5), itexel, 2);
    resident |= sparseTextureGatherOffsetARB(s2DRectShadow, c2, 2.0, ivec2(7), texel);

    resident |= sparseTextureGatherOffsetsARB(s2D, c2, offsets, texel);
    resident |= sparseTextureGatherOffsetsARB(is2DArray, c3, offsets, itexel, 2);
    resident |= sparseTextureGatherOffsetsARB(s2DRectShadow, c2, 2.0, offsets, texel);

    resident |= sparseImageLoadARB(i2D, ic2, texel);
    resident |= sparseImageLoadARB(ii3D, ic3, itexel);
    resident |= sparseImageLoadARB(i2DMS, ic2, 3, texel);

    outColor = sparseTexelsResidentARB(resident) ? texel : vec4(itexel) + vec4(utexel);
}