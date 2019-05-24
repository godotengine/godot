#version 450 core

#extension GL_ARB_sparse_texture2: enable
#extension GL_ARB_sparse_texture_clamp: enable
#extension GL_AMD_gpu_shader_half_float: enable
#extension GL_AMD_gpu_shader_half_float_fetch: enable
#extension GL_AMD_texture_gather_bias_lod: enable

layout(set = 0, binding =  0) uniform f16sampler1D            s1D;
layout(set = 0, binding =  1) uniform f16sampler2D            s2D;
layout(set = 0, binding =  2) uniform f16sampler3D            s3D;
layout(set = 0, binding =  3) uniform f16sampler2DRect        s2DRect;
layout(set = 0, binding =  4) uniform f16samplerCube          sCube;
layout(set = 0, binding =  5) uniform f16samplerBuffer        sBuffer;
layout(set = 0, binding =  6) uniform f16sampler2DMS          s2DMS;
layout(set = 0, binding =  7) uniform f16sampler1DArray       s1DArray;
layout(set = 0, binding =  8) uniform f16sampler2DArray       s2DArray;
layout(set = 0, binding =  9) uniform f16samplerCubeArray     sCubeArray;
layout(set = 0, binding = 10) uniform f16sampler2DMSArray     s2DMSArray;

layout(set = 0, binding = 11) uniform f16sampler1DShadow          s1DShadow;
layout(set = 0, binding = 12) uniform f16sampler2DShadow          s2DShadow;
layout(set = 0, binding = 13) uniform f16sampler2DRectShadow      s2DRectShadow;
layout(set = 0, binding = 14) uniform f16samplerCubeShadow        sCubeShadow;
layout(set = 0, binding = 15) uniform f16sampler1DArrayShadow     s1DArrayShadow;
layout(set = 0, binding = 16) uniform f16sampler2DArrayShadow     s2DArrayShadow;
layout(set = 0, binding = 17) uniform f16samplerCubeArrayShadow   sCubeArrayShadow;

layout(set = 1, binding =  0) layout(rgba16f) uniform f16image1D          i1D;
layout(set = 1, binding =  1) layout(rgba16f) uniform f16image2D          i2D;
layout(set = 1, binding =  2) layout(rgba16f) uniform f16image3D          i3D;
layout(set = 1, binding =  3) layout(rgba16f) uniform f16image2DRect      i2DRect;
layout(set = 1, binding =  4) layout(rgba16f) uniform f16imageCube        iCube;
layout(set = 1, binding =  5) layout(rgba16f) uniform f16image1DArray     i1DArray;
layout(set = 1, binding =  6) layout(rgba16f) uniform f16image2DArray     i2DArray;
layout(set = 1, binding =  7) layout(rgba16f) uniform f16imageCubeArray   iCubeArray;
layout(set = 1, binding =  8) layout(rgba16f) uniform f16imageBuffer      iBuffer;
layout(set = 1, binding =  9) layout(rgba16f) uniform f16image2DMS        i2DMS;
layout(set = 1, binding = 10) layout(rgba16f) uniform f16image2DMSArray   i2DMSArray;

layout(set = 2, binding =  0) uniform f16texture1D           t1D;
layout(set = 2, binding =  1) uniform f16texture2D           t2D;
layout(set = 2, binding =  2) uniform f16texture3D           t3D;
layout(set = 2, binding =  3) uniform f16texture2DRect       t2DRect;
layout(set = 2, binding =  4) uniform f16textureCube         tCube;
layout(set = 2, binding =  5) uniform f16texture1DArray      t1DArray;
layout(set = 2, binding =  6) uniform f16texture2DArray      t2DArray;
layout(set = 2, binding =  7) uniform f16textureCubeArray    tCubeArray;
layout(set = 2, binding =  8) uniform f16textureBuffer       tBuffer;
layout(set = 2, binding =  9) uniform f16texture2DMS         t2DMS;
layout(set = 2, binding = 10) uniform f16texture2DMSArray    t2DMSArray;

layout(set = 2, binding = 11) uniform sampler s;
layout(set = 2, binding = 12) uniform samplerShadow sShadow;

layout(set = 3, binding = 0, input_attachment_index = 0) uniform f16subpassInput   subpass;
layout(set = 3, binding = 1, input_attachment_index = 0) uniform f16subpassInputMS subpassMS;

layout(location =  0) in float c1;
layout(location =  1) in vec2  c2;
layout(location =  2) in vec3  c3;
layout(location =  3) in vec4  c4;

layout(location =  4) in float compare;
layout(location =  5) in float lod;
layout(location =  6) in float bias;
layout(location =  7) in float lodClamp;

layout(location =  8) in float dPdxy1;
layout(location =  9) in vec2  dPdxy2;
layout(location = 10) in vec3  dPdxy3;

layout(location = 11) in float16_t f16c1;
layout(location = 12) in f16vec2   f16c2;
layout(location = 13) in f16vec3   f16c3;
layout(location = 14) in f16vec4   f16c4;

layout(location = 15) in float16_t f16lod;
layout(location = 16) in float16_t f16bias;
layout(location = 17) in float16_t f16lodClamp;

layout(location = 18) in float16_t f16dPdxy1;
layout(location = 19) in f16vec2   f16dPdxy2;
layout(location = 20) in f16vec3   f16dPdxy3;

const int   offset1 = 1;
const ivec2 offset2 = ivec2(1);
const ivec3 offset3 = ivec3(1);
const ivec2 offsets[4] = { offset2, offset2, offset2, offset2 };

layout(location = 0) out vec4 fragColor;

f16vec4 testTexture()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += texture(s1D, c1);
    texel   += texture(s1D, f16c1, f16bias);
    texel   += texture(s2D, c2);
    texel   += texture(s2D, f16c2, f16bias);
    texel   += texture(s3D, c3);
    texel   += texture(s3D, f16c3, f16bias);
    texel   += texture(sCube, c3);
    texel   += texture(sCube, f16c3, f16bias);
    texel.x += texture(s1DShadow, c3);
    texel.x += texture(s1DShadow, f16c2, compare, f16bias);
    texel.x += texture(s2DShadow, c3);
    texel.x += texture(s2DShadow, f16c2, compare, f16bias);
    texel.x += texture(sCubeShadow, c4);
    texel.x += texture(sCubeShadow, f16c3, compare, f16bias);
    texel   += texture(s1DArray, c2);
    texel   += texture(s1DArray, f16c2, f16bias);
    texel   += texture(s2DArray, c3);
    texel   += texture(s2DArray, f16c3, f16bias);
    texel   += texture(sCubeArray, c4);
    texel   += texture(sCubeArray, f16c4, f16bias);
    texel.x += texture(s1DArrayShadow, c3);
    texel.x += texture(s1DArrayShadow, f16c2, compare, f16bias);
    texel.x += texture(s2DArrayShadow, c4);
    texel.x += texture(s2DArrayShadow, f16c3, compare);
    texel   += texture(s2DRect, c2);
    texel   += texture(s2DRect, f16c2);
    texel.x += texture(s2DRectShadow, c3);
    texel.x += texture(s2DRectShadow, f16c2, compare);
    texel.x += texture(sCubeArrayShadow, c4, compare);
    texel.x += texture(sCubeArrayShadow, f16c4, compare);

    return texel;
}

f16vec4 testTextureProj()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureProj(s1D, c2);
    texel   += textureProj(s1D, f16c2, f16bias);
    texel   += textureProj(s1D, c4);
    texel   += textureProj(s1D, f16c4, f16bias);
    texel   += textureProj(s2D, c3);
    texel   += textureProj(s2D, f16c3, f16bias);
    texel   += textureProj(s2D, c4);
    texel   += textureProj(s2D, f16c4, f16bias);
    texel   += textureProj(s3D, c4);
    texel   += textureProj(s3D, f16c4, f16bias);
    texel.x += textureProj(s1DShadow, c4);
    texel.x += textureProj(s1DShadow, f16c3, compare, f16bias);
    texel.x += textureProj(s2DShadow, c4);
    texel.x += textureProj(s2DShadow, f16c3, compare, f16bias);
    texel   += textureProj(s2DRect, c3);
    texel   += textureProj(s2DRect, f16c3);
    texel   += textureProj(s2DRect, c4);
    texel   += textureProj(s2DRect, f16c4);
    texel.x += textureProj(s2DRectShadow, c4);
    texel.x += textureProj(s2DRectShadow, f16c3, compare);

    return texel;
}

f16vec4 testTextureLod()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureLod(s1D, c1, lod);
    texel   += textureLod(s1D, f16c1, f16lod);
    texel   += textureLod(s2D, c2, lod);
    texel   += textureLod(s2D, f16c2, f16lod);
    texel   += textureLod(s3D, c3, lod);
    texel   += textureLod(s3D, f16c3, f16lod);
    texel   += textureLod(sCube, c3, lod);
    texel   += textureLod(sCube, f16c3, f16lod);
    texel.x += textureLod(s1DShadow, c3, lod);
    texel.x += textureLod(s1DShadow, f16c2, compare, f16lod);
    texel.x += textureLod(s2DShadow, c3, lod);
    texel.x += textureLod(s2DShadow, f16c2, compare, f16lod);
    texel   += textureLod(s1DArray, c2, lod);
    texel   += textureLod(s1DArray, f16c2, f16lod);
    texel   += textureLod(s2DArray, c3, lod);
    texel   += textureLod(s2DArray, f16c3, f16lod);
    texel.x += textureLod(s1DArrayShadow, c3, lod);
    texel.x += textureLod(s1DArrayShadow, f16c2, compare, f16lod);
    texel   += textureLod(sCubeArray, c4, lod);
    texel   += textureLod(sCubeArray, f16c4, f16lod);

    return texel;
}

f16vec4 testTextureOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureOffset(s1D, c1, offset1);
    texel   += textureOffset(s1D, f16c1, offset1, f16bias);
    texel   += textureOffset(s2D, c2, offset2);
    texel   += textureOffset(s2D, f16c2, offset2, f16bias);
    texel   += textureOffset(s3D, c3, offset3);
    texel   += textureOffset(s3D, f16c3, offset3, f16bias);
    texel   += textureOffset(s2DRect, c2, offset2);
    texel   += textureOffset(s2DRect, f16c2, offset2);
    texel.x += textureOffset(s2DRectShadow, c3, offset2);
    texel.x += textureOffset(s2DRectShadow, f16c2, compare, offset2);
    texel.x += textureOffset(s1DShadow, c3, offset1);
    texel.x += textureOffset(s1DShadow, f16c2, compare, offset1, f16bias);
    texel.x += textureOffset(s2DShadow, c3, offset2);
    texel.x += textureOffset(s2DShadow, f16c2, compare, offset2, f16bias);
    texel   += textureOffset(s1DArray, c2, offset1);
    texel   += textureOffset(s1DArray, f16c2, offset1, f16bias);
    texel   += textureOffset(s2DArray, c3, offset2);
    texel   += textureOffset(s2DArray, f16c3, offset2, f16bias);
    texel.x += textureOffset(s1DArrayShadow, c3, offset1);
    texel.x += textureOffset(s1DArrayShadow, f16c2, compare, offset1, f16bias);
    texel.x += textureOffset(s2DArrayShadow, c4, offset2);
    texel.x += textureOffset(s2DArrayShadow, f16c3, compare, offset2);

    return texel;
}

f16vec4 testTextureProjOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureProjOffset(s1D, c2, offset1);
    texel   += textureProjOffset(s1D, f16c2, offset1, f16bias);
    texel   += textureProjOffset(s1D, c4, offset1);
    texel   += textureProjOffset(s1D, f16c4, offset1, f16bias);
    texel   += textureProjOffset(s2D, c3, offset2);
    texel   += textureProjOffset(s2D, f16c3, offset2, f16bias);
    texel   += textureProjOffset(s2D, c4, offset2);
    texel   += textureProjOffset(s2D, f16c4, offset2, f16bias);
    texel   += textureProjOffset(s3D, c4, offset3);
    texel   += textureProjOffset(s3D, f16c4, offset3, f16bias);
    texel   += textureProjOffset(s2DRect, c3, offset2);
    texel   += textureProjOffset(s2DRect, f16c3, offset2);
    texel   += textureProjOffset(s2DRect, c4, offset2);
    texel   += textureProjOffset(s2DRect, f16c4, offset2);
    texel.x += textureProjOffset(s2DRectShadow, c4, offset2);
    texel.x += textureProjOffset(s2DRectShadow, f16c3, compare, offset2);
    texel.x += textureProjOffset(s1DShadow, c4, offset1);
    texel.x += textureProjOffset(s1DShadow, f16c3, compare, offset1, f16bias);
    texel.x += textureProjOffset(s2DShadow, c4, offset2);
    texel.x += textureProjOffset(s2DShadow, f16c3, compare, offset2, f16bias); 

    return texel;
}

f16vec4 testTextureLodOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureLodOffset(s1D, c1, lod, offset1);
    texel   += textureLodOffset(s1D, f16c1, f16lod, offset1);
    texel   += textureLodOffset(s2D, c2, lod, offset2);
    texel   += textureLodOffset(s2D, f16c2, f16lod, offset2);
    texel   += textureLodOffset(s3D, c3, lod, offset3);
    texel   += textureLodOffset(s3D, f16c3, f16lod, offset3);
    texel.x += textureLodOffset(s1DShadow, c3, lod, offset1);
    texel.x += textureLodOffset(s1DShadow, f16c2, compare, f16lod, offset1);
    texel.x += textureLodOffset(s2DShadow, c3, lod, offset2);
    texel.x += textureLodOffset(s2DShadow, f16c2, compare, f16lod, offset2);
    texel   += textureLodOffset(s1DArray, c2, lod, offset1);
    texel   += textureLodOffset(s1DArray, f16c2, f16lod, offset1);
    texel   += textureLodOffset(s2DArray, c3, lod, offset2);
    texel   += textureLodOffset(s2DArray, f16c3, f16lod, offset2);
    texel.x += textureLodOffset(s1DArrayShadow, c3, lod, offset1);
    texel.x += textureLodOffset(s1DArrayShadow, f16c2, compare, f16lod, offset1);

    return texel;
}

f16vec4 testTextureProjLodOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureProjLodOffset(s1D, c2, lod, offset1);
    texel   += textureProjLodOffset(s1D, f16c2, f16lod, offset1);
    texel   += textureProjLodOffset(s1D, c4, lod, offset1);
    texel   += textureProjLodOffset(s1D, f16c4, f16lod, offset1);
    texel   += textureProjLodOffset(s2D, c3, lod, offset2);
    texel   += textureProjLodOffset(s2D, f16c3, f16lod, offset2);
    texel   += textureProjLodOffset(s2D, c4, lod, offset2);
    texel   += textureProjLodOffset(s2D, f16c4, f16lod, offset2);
    texel   += textureProjLodOffset(s3D, c4, lod, offset3);
    texel   += textureProjLodOffset(s3D, f16c4, f16lod, offset3);
    texel.x += textureProjLodOffset(s1DShadow, c4, lod, offset1);
    texel.x += textureProjLodOffset(s1DShadow, f16c3, compare, f16lod, offset1);
    texel.x += textureProjLodOffset(s2DShadow, c4, lod, offset2);
    texel.x += textureProjLodOffset(s2DShadow, f16c3, compare, f16lod, offset2);

    return texel;
}

f16vec4 testTexelFetch()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += texelFetch(s1D, int(c1), int(lod));
    texel   += texelFetch(s2D, ivec2(c2), int(lod));
    texel   += texelFetch(s3D, ivec3(c3), int(lod));
    texel   += texelFetch(s2DRect, ivec2(c2));
    texel   += texelFetch(s1DArray, ivec2(c2), int(lod));
    texel   += texelFetch(s2DArray, ivec3(c3), int(lod));
    texel   += texelFetch(sBuffer, int(c1));
    texel   += texelFetch(s2DMS, ivec2(c2), 1);
    texel   += texelFetch(s2DMSArray, ivec3(c3), 2);

    return texel;
}

f16vec4 testTexelFetchOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += texelFetchOffset(s1D, int(c1), int(lod), offset1);
    texel   += texelFetchOffset(s2D, ivec2(c2), int(lod), offset2);
    texel   += texelFetchOffset(s3D, ivec3(c3), int(lod), offset3);
    texel   += texelFetchOffset(s2DRect, ivec2(c2), offset2);
    texel   += texelFetchOffset(s1DArray, ivec2(c2), int(lod), offset1);
    texel   += texelFetchOffset(s2DArray, ivec3(c3), int(lod), offset2);

    return texel;
}

f16vec4 testTextureGrad()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGrad(s1D, c1, dPdxy1, dPdxy1);
    texel   += textureGrad(s1D, f16c1, f16dPdxy1, f16dPdxy1);
    texel   += textureGrad(s2D, c2, dPdxy2, dPdxy2);
    texel   += textureGrad(s2D, f16c2, f16dPdxy2, f16dPdxy2);
    texel   += textureGrad(s3D, c3, dPdxy3, dPdxy3);
    texel   += textureGrad(s3D, f16c3, f16dPdxy3, f16dPdxy3);
    texel   += textureGrad(sCube, c3, dPdxy3, dPdxy3);
    texel   += textureGrad(sCube, f16c3, f16dPdxy3, f16dPdxy3);
    texel   += textureGrad(s2DRect, c2, dPdxy2, dPdxy2);
    texel   += textureGrad(s2DRect, f16c2, f16dPdxy2, f16dPdxy2);
    texel.x += textureGrad(s2DRectShadow, c3, dPdxy2, dPdxy2);
    texel.x += textureGrad(s2DRectShadow, f16c2, compare, f16dPdxy2, f16dPdxy2);
    texel.x += textureGrad(s1DShadow, c3, dPdxy1, dPdxy1);
    texel.x += textureGrad(s1DShadow, f16c2, compare, f16dPdxy1, f16dPdxy1);
    texel.x += textureGrad(s2DShadow, c3, dPdxy2, dPdxy2);
    texel.x += textureGrad(s2DShadow, f16c2, compare, f16dPdxy2, f16dPdxy2);
    texel.x += textureGrad(sCubeShadow, c4, dPdxy3, dPdxy3);
    texel.x += textureGrad(sCubeShadow, f16c3, compare, f16dPdxy3, f16dPdxy3);
    texel   += textureGrad(s1DArray, c2, dPdxy1, dPdxy1);
    texel   += textureGrad(s1DArray, f16c2, f16dPdxy1, f16dPdxy1);
    texel   += textureGrad(s2DArray, c3, dPdxy2, dPdxy2);
    texel   += textureGrad(s2DArray, f16c3, f16dPdxy2, f16dPdxy2);
    texel.x += textureGrad(s1DArrayShadow, c3, dPdxy1, dPdxy1);
    texel.x += textureGrad(s1DArrayShadow, f16c2, compare, f16dPdxy1, f16dPdxy1);
    texel.x += textureGrad(s2DArrayShadow, c4, dPdxy2, dPdxy2);
    texel.x += textureGrad(s2DArrayShadow, f16c3, compare, f16dPdxy2, f16dPdxy2);
    texel   += textureGrad(sCubeArray, c4, dPdxy3, dPdxy3);
    texel   += textureGrad(sCubeArray, f16c4, f16dPdxy3, f16dPdxy3);

    return texel;
}

f16vec4 testTextureGradOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGradOffset(s1D, c1, dPdxy1, dPdxy1, offset1);
    texel   += textureGradOffset(s1D, f16c1, f16dPdxy1, f16dPdxy1, offset1);
    texel   += textureGradOffset(s2D, c2, dPdxy2, dPdxy2, offset2);
    texel   += textureGradOffset(s2D, f16c2, f16dPdxy2, f16dPdxy2, offset2);
    texel   += textureGradOffset(s3D, c3, dPdxy3, dPdxy3, offset3);
    texel   += textureGradOffset(s3D, f16c3, f16dPdxy3, f16dPdxy3, offset3);
    texel   += textureGradOffset(s2DRect, c2, dPdxy2, dPdxy2, offset2);
    texel   += textureGradOffset(s2DRect, f16c2, f16dPdxy2, f16dPdxy2, offset2);
    texel.x += textureGradOffset(s2DRectShadow, c3, dPdxy2, dPdxy2, offset2);
    texel.x += textureGradOffset(s2DRectShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, offset2);
    texel.x += textureGradOffset(s1DShadow, c3, dPdxy1, dPdxy1, offset1);
    texel.x += textureGradOffset(s1DShadow, f16c2, compare, f16dPdxy1, f16dPdxy1, offset1);
    texel.x += textureGradOffset(s2DShadow, c3, dPdxy2, dPdxy2, offset2);
    texel.x += textureGradOffset(s2DShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, offset2);
    texel   += textureGradOffset(s1DArray, c2, dPdxy1, dPdxy1, offset1);
    texel   += textureGradOffset(s1DArray, f16c2, f16dPdxy1, f16dPdxy1, offset1);
    texel   += textureGradOffset(s2DArray, c3, dPdxy2, dPdxy2, offset2);
    texel   += textureGradOffset(s2DArray, f16c3, f16dPdxy2, f16dPdxy2, offset2);
    texel.x += textureGradOffset(s1DArrayShadow, c3, dPdxy1, dPdxy1, offset1);
    texel.x += textureGradOffset(s1DArrayShadow, f16c2, compare, f16dPdxy1, f16dPdxy1, offset1);
    texel.x += textureGradOffset(s2DArrayShadow, c4, dPdxy2, dPdxy2, offset2);
    texel.x += textureGradOffset(s2DArrayShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, offset2);

    return texel;
}

f16vec4 testTextureProjGrad()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureProjGrad(s1D, c2, dPdxy1, dPdxy1);
    texel   += textureProjGrad(s1D, f16c2, f16dPdxy1, f16dPdxy1);
    texel   += textureProjGrad(s1D, c4, dPdxy1, dPdxy1);
    texel   += textureProjGrad(s1D, f16c4, f16dPdxy1, f16dPdxy1);
    texel   += textureProjGrad(s2D, c3, dPdxy2, dPdxy2);
    texel   += textureProjGrad(s2D, f16c3, f16dPdxy2, f16dPdxy2);
    texel   += textureProjGrad(s2D, c4, dPdxy2, dPdxy2);
    texel   += textureProjGrad(s2D, f16c4, f16dPdxy2, f16dPdxy2);
    texel   += textureProjGrad(s3D, c4, dPdxy3, dPdxy3);
    texel   += textureProjGrad(s3D, f16c4, f16dPdxy3, f16dPdxy3);
    texel   += textureProjGrad(s2DRect, c3, dPdxy2, dPdxy2);
    texel   += textureProjGrad(s2DRect, f16c3, f16dPdxy2, f16dPdxy2);
    texel   += textureProjGrad(s2DRect, c4, dPdxy2, dPdxy2);
    texel   += textureProjGrad(s2DRect, f16c4, f16dPdxy2, f16dPdxy2);
    texel.x += textureProjGrad(s2DRectShadow, c4, dPdxy2, dPdxy2);
    texel.x += textureProjGrad(s2DRectShadow, f16c3, compare, f16dPdxy2, f16dPdxy2);
    texel.x += textureProjGrad(s1DShadow, c4, dPdxy1, dPdxy1);
    texel.x += textureProjGrad(s1DShadow, f16c3, compare, f16dPdxy1, f16dPdxy1);
    texel.x += textureProjGrad(s2DShadow, c4, dPdxy2, dPdxy2);
    texel.x += textureProjGrad(s2DShadow, f16c3, compare, f16dPdxy2, f16dPdxy2);

    return texel;
}

f16vec4 testTextureProjGradoffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureProjGradOffset(s1D, c2, dPdxy1, dPdxy1, offset1);
    texel   += textureProjGradOffset(s1D, f16c2, f16dPdxy1, f16dPdxy1, offset1);
    texel   += textureProjGradOffset(s1D, c4, dPdxy1, dPdxy1, offset1);
    texel   += textureProjGradOffset(s1D, f16c4, f16dPdxy1, f16dPdxy1, offset1);
    texel   += textureProjGradOffset(s2D, c3, dPdxy2, dPdxy2, offset2);
    texel   += textureProjGradOffset(s2D, f16c3, f16dPdxy2, f16dPdxy2, offset2);
    texel   += textureProjGradOffset(s2D, c4, dPdxy2, dPdxy2, offset2);
    texel   += textureProjGradOffset(s2D, f16c4, f16dPdxy2, f16dPdxy2, offset2);
    texel   += textureProjGradOffset(s2DRect, c3, dPdxy2, dPdxy2, offset2);
    texel   += textureProjGradOffset(s2DRect, f16c3, f16dPdxy2, f16dPdxy2, offset2);
    texel   += textureProjGradOffset(s2DRect, c4, dPdxy2, dPdxy2, offset2);
    texel   += textureProjGradOffset(s2DRect, f16c4, f16dPdxy2, f16dPdxy2, offset2);
    texel.x += textureProjGradOffset(s2DRectShadow, c4, dPdxy2, dPdxy2, offset2);
    texel.x += textureProjGradOffset(s2DRectShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, offset2);
    texel   += textureProjGradOffset(s3D, c4, dPdxy3, dPdxy3, offset3);
    texel   += textureProjGradOffset(s3D, f16c4, f16dPdxy3, f16dPdxy3, offset3);
    texel.x += textureProjGradOffset(s1DShadow, c4, dPdxy1, dPdxy1, offset1);
    texel.x += textureProjGradOffset(s1DShadow, f16c3, compare, f16dPdxy1, f16dPdxy1, offset1);
    texel.x += textureProjGradOffset(s2DShadow, c4, dPdxy2, dPdxy2, offset2);
    texel.x += textureProjGradOffset(s2DShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, offset2);

    return texel;
}

f16vec4 testTextureGather()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGather(s2D, c2, 0);
    texel   += textureGather(s2D, f16c2, 0, f16bias);
    texel   += textureGather(s2DArray, c3, 0);
    texel   += textureGather(s2DArray, f16c3, 0, f16bias);
    texel   += textureGather(sCube, c3, 0);
    texel   += textureGather(sCube, f16c3, 0, f16bias);
    texel   += textureGather(sCubeArray, c4, 0);
    texel   += textureGather(sCubeArray, f16c4, 0, f16bias);
    texel   += textureGather(s2DRect, c2, 0);
    texel   += textureGather(s2DRect, f16c2, 0);
    texel   += textureGather(s2DShadow, c2, compare);
    texel   += textureGather(s2DShadow, f16c2, compare);
    texel   += textureGather(s2DArrayShadow, c3, compare);
    texel   += textureGather(s2DArrayShadow, f16c3, compare);
    texel   += textureGather(sCubeShadow, c3, compare);
    texel   += textureGather(sCubeShadow, f16c3, compare);
    texel   += textureGather(sCubeArrayShadow, c4, compare);
    texel   += textureGather(sCubeArrayShadow, f16c4, compare);
    texel   += textureGather(s2DRectShadow, c2, compare);
    texel   += textureGather(s2DRectShadow, f16c2, compare);

    return texel;
}

f16vec4 testTextureGatherOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGatherOffset(s2D, c2, offset2, 0);
    texel   += textureGatherOffset(s2D, f16c2, offset2, 0, f16bias);
    texel   += textureGatherOffset(s2DArray, c3, offset2, 0);
    texel   += textureGatherOffset(s2DArray, f16c3, offset2, 0, f16bias);
    texel   += textureGatherOffset(s2DRect, c2, offset2, 0);
    texel   += textureGatherOffset(s2DRect, f16c2, offset2, 0);
    texel   += textureGatherOffset(s2DShadow, c2, compare, offset2);
    texel   += textureGatherOffset(s2DShadow, f16c2, compare, offset2);
    texel   += textureGatherOffset(s2DArrayShadow, c3, compare, offset2);
    texel   += textureGatherOffset(s2DArrayShadow, f16c3, compare, offset2);
    texel   += textureGatherOffset(s2DRectShadow, c2, compare, offset2);
    texel   += textureGatherOffset(s2DRectShadow, f16c2, compare, offset2);

    return texel;
}

f16vec4 testTextureGatherOffsets()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGatherOffsets(s2D, c2, offsets, 0);
    texel   += textureGatherOffsets(s2D, f16c2, offsets, 0, f16bias);
    texel   += textureGatherOffsets(s2DArray, c3, offsets, 0);
    texel   += textureGatherOffsets(s2DArray, f16c3, offsets, 0, f16bias);
    texel   += textureGatherOffsets(s2DRect, c2, offsets, 0);
    texel   += textureGatherOffsets(s2DRect, f16c2, offsets, 0);
    texel   += textureGatherOffsets(s2DShadow, c2, compare, offsets);
    texel   += textureGatherOffsets(s2DShadow, f16c2, compare, offsets);
    texel   += textureGatherOffsets(s2DArrayShadow, c3, compare, offsets);
    texel   += textureGatherOffsets(s2DArrayShadow, f16c3, compare, offsets);
    texel   += textureGatherOffsets(s2DRectShadow, c2, compare, offsets);
    texel   += textureGatherOffsets(s2DRectShadow, f16c2, compare, offsets);

    return texel;
}

f16vec4 testTextureGatherLod()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGatherLodAMD(s2D, c2, lod, 0);
    texel   += textureGatherLodAMD(s2D, f16c2, f16lod, 0);
    texel   += textureGatherLodAMD(s2DArray, c3, lod, 0);
    texel   += textureGatherLodAMD(s2DArray, f16c3, f16lod, 0);
    texel   += textureGatherLodAMD(sCube, c3, lod, 0);
    texel   += textureGatherLodAMD(sCube, f16c3, f16lod, 0);
    texel   += textureGatherLodAMD(sCubeArray, c4, lod, 0);
    texel   += textureGatherLodAMD(sCubeArray, f16c4, f16lod, 0);

    return texel;
}

f16vec4 testTextureGatherLodOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGatherLodOffsetAMD(s2D, c2, lod, offset2, 0);
    texel   += textureGatherLodOffsetAMD(s2D, f16c2, f16lod, offset2, 0);
    texel   += textureGatherLodOffsetAMD(s2DArray, c3, lod, offset2, 0);
    texel   += textureGatherLodOffsetAMD(s2DArray, f16c3, f16lod, offset2, 0);

    return texel;
}

f16vec4 testTextureGatherLodOffsets()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGatherLodOffsetsAMD(s2D, c2, lod, offsets, 0);
    texel   += textureGatherLodOffsetsAMD(s2D, f16c2, f16lod, offsets, 0);
    texel   += textureGatherLodOffsetsAMD(s2DArray, c3, lod, offsets, 0);
    texel   += textureGatherLodOffsetsAMD(s2DArray, f16c3, f16lod, offsets, 0);

    return texel;
}

ivec4 testTextureSize()
{
    ivec4 size = ivec4(0);

    size.x      += textureSize(s1D, int(lod));
    size.xy     += textureSize(s2D, int(lod));
    size.xyz    += textureSize(s3D, int(lod));
    size.xy     += textureSize(sCube, int(lod));
    size.x      += textureSize(s1DShadow, int(lod));
    size.xy     += textureSize(s2DShadow, int(lod));
    size.xy     += textureSize(sCubeShadow, int(lod));
    size.xyz    += textureSize(sCubeArray, int(lod));
    size.xyz    += textureSize(sCubeArrayShadow, int(lod));
    size.xy     += textureSize(s2DRect);
    size.xy     += textureSize(s2DRectShadow);
    size.xy     += textureSize(s1DArray, int(lod));
    size.xyz    += textureSize(s2DArray, int(lod));
    size.xy     += textureSize(s1DArrayShadow, int(lod));
    size.xyz    += textureSize(s2DArrayShadow, int(lod));
    size.x      += textureSize(sBuffer);
    size.xy     += textureSize(s2DMS);
    size.xyz    += textureSize(s2DMSArray);

    return size;
}

vec2 testTextureQueryLod()
{
    vec2 lod = vec2(0.0);

    lod  += textureQueryLod(s1D, c1);
    lod  += textureQueryLod(s1D, f16c1);
    lod  += textureQueryLod(s2D, c2);
    lod  += textureQueryLod(s2D, f16c2);
    lod  += textureQueryLod(s3D, c3);
    lod  += textureQueryLod(s3D, f16c3);
    lod  += textureQueryLod(sCube, c3);
    lod  += textureQueryLod(sCube, f16c3);
    lod  += textureQueryLod(s1DArray, c1);
    lod  += textureQueryLod(s1DArray, f16c1);
    lod  += textureQueryLod(s2DArray, c2);
    lod  += textureQueryLod(s2DArray, f16c2);
    lod  += textureQueryLod(sCubeArray, c3);
    lod  += textureQueryLod(sCubeArray, f16c3);
    lod  += textureQueryLod(s1DShadow, c1);
    lod  += textureQueryLod(s1DShadow, f16c1);
    lod  += textureQueryLod(s2DShadow, c2);
    lod  += textureQueryLod(s2DShadow, f16c2);
    lod  += textureQueryLod(sCubeArrayShadow, c3);
    lod  += textureQueryLod(sCubeArrayShadow, f16c3);
    lod  += textureQueryLod(s1DArrayShadow, c1);
    lod  += textureQueryLod(s1DArrayShadow, f16c1);
    lod  += textureQueryLod(s2DArrayShadow, c2);
    lod  += textureQueryLod(s2DArrayShadow, f16c2);
    lod  += textureQueryLod(sCubeArrayShadow, c3);
    lod  += textureQueryLod(sCubeArrayShadow, f16c3);

    return lod;
}

int testTextureQueryLevels()
{
    int levels = 0;

    levels  += textureQueryLevels(s1D);
    levels  += textureQueryLevels(s2D);
    levels  += textureQueryLevels(s3D);
    levels  += textureQueryLevels(sCube);
    levels  += textureQueryLevels(s1DShadow);
    levels  += textureQueryLevels(s2DShadow);
    levels  += textureQueryLevels(sCubeShadow);
    levels  += textureQueryLevels(sCubeArray);
    levels  += textureQueryLevels(sCubeArrayShadow);
    levels  += textureQueryLevels(s1DArray);
    levels  += textureQueryLevels(s2DArray);
    levels  += textureQueryLevels(s1DArrayShadow);
    levels  += textureQueryLevels(s2DArrayShadow);

    return levels;
}

int testTextureSamples()
{
    int samples = 0;

    samples += textureSamples(s2DMS);
    samples += textureSamples(s2DMSArray);

    return samples;
}

f16vec4 testImageLoad()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel += imageLoad(i1D, int(c1));
    texel += imageLoad(i2D, ivec2(c2));
    texel += imageLoad(i3D, ivec3(c3));
    texel += imageLoad(i2DRect, ivec2(c2));
    texel += imageLoad(iCube, ivec3(c3));
    texel += imageLoad(iBuffer, int(c1));
    texel += imageLoad(i1DArray, ivec2(c2));
    texel += imageLoad(i2DArray, ivec3(c3));
    texel += imageLoad(iCubeArray, ivec3(c3));
    texel += imageLoad(i2DMS, ivec2(c2), 1);
    texel += imageLoad(i2DMSArray, ivec3(c3), 1);

    return texel;
}

void testImageStore(f16vec4 data)
{
    imageStore(i1D, int(c1), data);
    imageStore(i2D, ivec2(c2), data);
    imageStore(i3D, ivec3(c3), data);
    imageStore(i2DRect, ivec2(c2), data);
    imageStore(iCube, ivec3(c3), data);
    imageStore(iBuffer, int(c1), data);
    imageStore(i1DArray, ivec2(c2), data);
    imageStore(i2DArray, ivec3(c3), data);
    imageStore(iCubeArray, ivec3(c3), data);
    imageStore(i2DMS, ivec2(c2), 1, data);
    imageStore(i2DMSArray, ivec3(c3), 1, data);
}

f16vec4 testSparseTexture()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureARB(s2D, c2, texel);
    sparseTextureARB(s2D, f16c2, texel, f16bias);
    sparseTextureARB(s3D, c3, texel);
    sparseTextureARB(s3D, f16c3, texel, f16bias);
    sparseTextureARB(sCube, c3, texel);
    sparseTextureARB(sCube, f16c3, texel, f16bias);
    sparseTextureARB(s2DShadow, c3, texel.x);
    sparseTextureARB(s2DShadow, f16c2, compare, texel.x, f16bias);
    sparseTextureARB(sCubeShadow, c4, texel.x);
    sparseTextureARB(sCubeShadow, f16c3, compare, texel.x, f16bias);
    sparseTextureARB(s2DArray, c3, texel);
    sparseTextureARB(s2DArray, f16c3, texel, f16bias);
    sparseTextureARB(sCubeArray, c4, texel);
    sparseTextureARB(sCubeArray, f16c4, texel, f16bias);
    sparseTextureARB(s2DArrayShadow, c4, texel.x);
    sparseTextureARB(s2DArrayShadow, f16c3, compare, texel.x);
    sparseTextureARB(s2DRect, c2, texel);
    sparseTextureARB(s2DRect, f16c2, texel);
    sparseTextureARB(s2DRectShadow, c3, texel.x);
    sparseTextureARB(s2DRectShadow, f16c2, compare, texel.x);
    sparseTextureARB(sCubeArrayShadow, c4, compare, texel.x);
    sparseTextureARB(sCubeArrayShadow, f16c4, compare, texel.x);

    return texel;
}

f16vec4 testSparseTextureLod()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureLodARB(s2D, c2, lod, texel);
    sparseTextureLodARB(s2D, f16c2, f16lod, texel);
    sparseTextureLodARB(s3D, c3, lod, texel);
    sparseTextureLodARB(s3D, f16c3, f16lod, texel);
    sparseTextureLodARB(sCube, c3, lod, texel);
    sparseTextureLodARB(sCube, f16c3, f16lod, texel);
    sparseTextureLodARB(s2DShadow, c3, lod, texel.x);
    sparseTextureLodARB(s2DShadow, f16c2, compare, f16lod, texel.x);
    sparseTextureLodARB(s2DArray, c3, lod, texel);
    sparseTextureLodARB(s2DArray, f16c3, f16lod, texel);
    sparseTextureLodARB(sCubeArray, c4, lod, texel);
    sparseTextureLodARB(sCubeArray, f16c4, f16lod, texel);

    return texel;
}

f16vec4 testSparseTextureOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureOffsetARB(s2D, c2, offset2, texel);
    sparseTextureOffsetARB(s2D, f16c2, offset2, texel, f16bias);
    sparseTextureOffsetARB(s3D, c3, offset3, texel);
    sparseTextureOffsetARB(s3D, f16c3, offset3, texel, f16bias);
    sparseTextureOffsetARB(s2DRect, c2, offset2, texel);
    sparseTextureOffsetARB(s2DRect, f16c2, offset2, texel);
    sparseTextureOffsetARB(s2DRectShadow, c3, offset2, texel.x);
    sparseTextureOffsetARB(s2DRectShadow, f16c2, compare, offset2, texel.x);
    sparseTextureOffsetARB(s2DShadow, c3, offset2, texel.x);
    sparseTextureOffsetARB(s2DShadow, f16c2, compare, offset2, texel.x, f16bias);
    sparseTextureOffsetARB(s2DArray, c3, offset2, texel);
    sparseTextureOffsetARB(s2DArray, f16c3, offset2, texel, f16bias);
    sparseTextureOffsetARB(s2DArrayShadow, c4, offset2, texel.x);
    sparseTextureOffsetARB(s2DArrayShadow, f16c3, compare, offset2, texel.x);

    return texel;
}

f16vec4 testSparseTextureLodOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureLodOffsetARB(s2D, c2, lod, offset2, texel);
    sparseTextureLodOffsetARB(s2D, f16c2, f16lod, offset2, texel);
    sparseTextureLodOffsetARB(s3D, c3, lod, offset3, texel);
    sparseTextureLodOffsetARB(s3D, f16c3, f16lod, offset3, texel);
    sparseTextureLodOffsetARB(s2DShadow, c3, lod, offset2, texel.x);
    sparseTextureLodOffsetARB(s2DShadow, f16c2, compare, f16lod, offset2, texel.x);
    sparseTextureLodOffsetARB(s2DArray, c3, lod, offset2, texel);
    sparseTextureLodOffsetARB(s2DArray, f16c3, f16lod, offset2, texel);

    return texel;
}

f16vec4 testSparseTextureGrad()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGradARB(s2D, c2, dPdxy2, dPdxy2, texel);
    sparseTextureGradARB(s2D, f16c2, f16dPdxy2, f16dPdxy2, texel);
    sparseTextureGradARB(s3D, c3, dPdxy3, dPdxy3, texel);
    sparseTextureGradARB(s3D, f16c3, f16dPdxy3, f16dPdxy3, texel);
    sparseTextureGradARB(sCube, c3, dPdxy3, dPdxy3, texel);
    sparseTextureGradARB(sCube, f16c3, f16dPdxy3, f16dPdxy3, texel);
    sparseTextureGradARB(s2DRect, c2, dPdxy2, dPdxy2, texel);
    sparseTextureGradARB(s2DRect, f16c2, f16dPdxy2, f16dPdxy2, texel);
    sparseTextureGradARB(s2DRectShadow, c3, dPdxy2, dPdxy2, texel.x);
    sparseTextureGradARB(s2DRectShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, texel.x);
    sparseTextureGradARB(s2DShadow, c3, dPdxy2, dPdxy2, texel.x);
    sparseTextureGradARB(s2DShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, texel.x);
    sparseTextureGradARB(sCubeShadow, c4, dPdxy3, dPdxy3, texel.x);
    sparseTextureGradARB(sCubeShadow, f16c3, compare, f16dPdxy3, f16dPdxy3, texel.x);
    sparseTextureGradARB(s2DArray, c3, dPdxy2, dPdxy2, texel);
    sparseTextureGradARB(s2DArray, f16c3, f16dPdxy2, f16dPdxy2, texel);
    sparseTextureGradARB(s2DArrayShadow, c4, dPdxy2, dPdxy2, texel.x);
    sparseTextureGradARB(s2DArrayShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, texel.x);
    sparseTextureGradARB(sCubeArray, c4, dPdxy3, dPdxy3, texel);
    sparseTextureGradARB(sCubeArray, f16c4, f16dPdxy3, f16dPdxy3, texel);

    return texel;
}

f16vec4 testSparseTextureGradOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGradOffsetARB(s2D, c2, dPdxy2, dPdxy2, offset2, texel);
    sparseTextureGradOffsetARB(s2D, f16c2, f16dPdxy2, f16dPdxy2, offset2, texel);
    sparseTextureGradOffsetARB(s3D, c3, dPdxy3, dPdxy3, offset3, texel);
    sparseTextureGradOffsetARB(s3D, f16c3, f16dPdxy3, f16dPdxy3, offset3, texel);
    sparseTextureGradOffsetARB(s2DRect, c2, dPdxy2, dPdxy2, offset2, texel);
    sparseTextureGradOffsetARB(s2DRect, f16c2, f16dPdxy2, f16dPdxy2, offset2, texel);
    sparseTextureGradOffsetARB(s2DRectShadow, c3, dPdxy2, dPdxy2, offset2, texel.x);
    sparseTextureGradOffsetARB(s2DRectShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, offset2, texel.x);
    sparseTextureGradOffsetARB(s2DShadow, c3, dPdxy2, dPdxy2, offset2, texel.x);
    sparseTextureGradOffsetARB(s2DShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, offset2, texel.x);
    sparseTextureGradOffsetARB(s2DArray, c3, dPdxy2, dPdxy2, offset2, texel);
    sparseTextureGradOffsetARB(s2DArray, f16c3, f16dPdxy2, f16dPdxy2, offset2, texel);
    sparseTextureGradOffsetARB(s2DArrayShadow, c4, dPdxy2, dPdxy2, offset2, texel.x);
    sparseTextureGradOffsetARB(s2DArrayShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, offset2, texel.x);

    return texel;
}

f16vec4 testSparseTexelFetch()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTexelFetchARB(s2D, ivec2(c2), int(lod), texel);
    sparseTexelFetchARB(s3D, ivec3(c3), int(lod), texel);
    sparseTexelFetchARB(s2DRect, ivec2(c2), texel);
    sparseTexelFetchARB(s2DArray, ivec3(c3), int(lod), texel);
    sparseTexelFetchARB(s2DMS, ivec2(c2), 1, texel);
    sparseTexelFetchARB(s2DMSArray, ivec3(c3), 2, texel);

    return texel;
}

f16vec4 testSparseTexelFetchOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTexelFetchOffsetARB(s2D, ivec2(c2), int(lod), offset2, texel);
    sparseTexelFetchOffsetARB(s3D, ivec3(c3), int(lod), offset3, texel);
    sparseTexelFetchOffsetARB(s2DRect, ivec2(c2), offset2, texel);
    sparseTexelFetchOffsetARB(s2DArray, ivec3(c3), int(lod), offset2, texel);

    return texel;
}

f16vec4 testSparseTextureGather()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGatherARB(s2D, c2, texel, 0);
    sparseTextureGatherARB(s2D, f16c2, texel, 0, f16bias);
    sparseTextureGatherARB(s2DArray, c3, texel, 0);
    sparseTextureGatherARB(s2DArray, f16c3, texel, 0, f16bias);
    sparseTextureGatherARB(sCube, c3, texel, 0);
    sparseTextureGatherARB(sCube, f16c3, texel, 0, f16bias);
    sparseTextureGatherARB(sCubeArray, c4, texel, 0);
    sparseTextureGatherARB(sCubeArray, f16c4, texel, 0, f16bias);
    sparseTextureGatherARB(s2DRect, c2, texel, 0);
    sparseTextureGatherARB(s2DRect, f16c2, texel, 0);
    sparseTextureGatherARB(s2DShadow, c2, compare, texel);
    sparseTextureGatherARB(s2DShadow, f16c2, compare, texel);
    sparseTextureGatherARB(s2DArrayShadow, c3, compare, texel);
    sparseTextureGatherARB(s2DArrayShadow, f16c3, compare, texel);
    sparseTextureGatherARB(sCubeShadow, c3, compare, texel);
    sparseTextureGatherARB(sCubeShadow, f16c3, compare, texel);
    sparseTextureGatherARB(sCubeArrayShadow, c4, compare, texel);
    sparseTextureGatherARB(sCubeArrayShadow, f16c4, compare, texel);
    sparseTextureGatherARB(s2DRectShadow, c2, compare, texel);
    sparseTextureGatherARB(s2DRectShadow, f16c2, compare, texel);

    return texel;
}

f16vec4 testSparseTextureGatherOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGatherOffsetARB(s2D, c2, offset2, texel, 0);
    sparseTextureGatherOffsetARB(s2D, f16c2, offset2, texel, 0, f16bias);
    sparseTextureGatherOffsetARB(s2DArray, c3, offset2, texel, 0);
    sparseTextureGatherOffsetARB(s2DArray, f16c3, offset2, texel, 0, f16bias);
    sparseTextureGatherOffsetARB(s2DRect, c2, offset2, texel, 0);
    sparseTextureGatherOffsetARB(s2DRect, f16c2, offset2, texel, 0);
    sparseTextureGatherOffsetARB(s2DShadow, c2, compare, offset2, texel);
    sparseTextureGatherOffsetARB(s2DShadow, f16c2, compare, offset2, texel);
    sparseTextureGatherOffsetARB(s2DArrayShadow, c3, compare, offset2, texel);
    sparseTextureGatherOffsetARB(s2DArrayShadow, f16c3, compare, offset2, texel);
    sparseTextureGatherOffsetARB(s2DRectShadow, c2, compare, offset2, texel);
    sparseTextureGatherOffsetARB(s2DRectShadow, f16c2, compare, offset2, texel);

    return texel;
}

f16vec4 testSparseTextureGatherOffsets()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGatherOffsetsARB(s2D, c2, offsets, texel, 0);
    sparseTextureGatherOffsetsARB(s2D, f16c2, offsets, texel, 0, f16bias);
    sparseTextureGatherOffsetsARB(s2DArray, c3, offsets, texel, 0);
    sparseTextureGatherOffsetsARB(s2DArray, f16c3, offsets, texel, 0, f16bias);
    sparseTextureGatherOffsetsARB(s2DRect, c2, offsets, texel, 0);
    sparseTextureGatherOffsetsARB(s2DRect, f16c2, offsets, texel, 0);
    sparseTextureGatherOffsetsARB(s2DShadow, c2, compare, offsets, texel);
    sparseTextureGatherOffsetsARB(s2DShadow, f16c2, compare, offsets, texel);
    sparseTextureGatherOffsetsARB(s2DArrayShadow, c3, compare, offsets, texel);
    sparseTextureGatherOffsetsARB(s2DArrayShadow, f16c3, compare, offsets, texel);
    sparseTextureGatherOffsetsARB(s2DRectShadow, c2, compare, offsets, texel);
    sparseTextureGatherOffsetsARB(s2DRectShadow, f16c2, compare, offsets, texel);

    return texel;
}

f16vec4 testSparseTextureGatherLod()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGatherLodAMD(s2D, c2, lod, texel, 0);
    sparseTextureGatherLodAMD(s2D, f16c2, f16lod, texel, 0);
    sparseTextureGatherLodAMD(s2DArray, c3, lod, texel, 0);
    sparseTextureGatherLodAMD(s2DArray, f16c3, f16lod, texel, 0);
    sparseTextureGatherLodAMD(sCube, c3, lod, texel, 0);
    sparseTextureGatherLodAMD(sCube, f16c3, f16lod, texel, 0);
    sparseTextureGatherLodAMD(sCubeArray, c4, lod, texel, 0);
    sparseTextureGatherLodAMD(sCubeArray, f16c4, f16lod, texel, 0);

    return texel;
}

f16vec4 testSparseTextureGatherLodOffset()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGatherLodOffsetAMD(s2D, c2, lod, offset2, texel, 0);
    sparseTextureGatherLodOffsetAMD(s2D, f16c2, f16lod, offset2, texel, 0);
    sparseTextureGatherLodOffsetAMD(s2DArray, c3, lod, offset2, texel, 0);
    sparseTextureGatherLodOffsetAMD(s2DArray, f16c3, f16lod, offset2, texel, 0);

    return texel;
}

f16vec4 testSparseTextureGatherLodOffsets()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGatherLodOffsetsAMD(s2D, c2, lod, offsets, texel, 0);
    sparseTextureGatherLodOffsetsAMD(s2D, f16c2, f16lod, offsets, texel, 0);
    sparseTextureGatherLodOffsetsAMD(s2DArray, c3, lod, offsets, texel, 0);
    sparseTextureGatherLodOffsetsAMD(s2DArray, f16c3, f16lod, offsets, texel, 0);

    return texel;
}

f16vec4 testSparseImageLoad()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseImageLoadARB(i2D, ivec2(c2), texel);
    sparseImageLoadARB(i3D, ivec3(c3), texel);
    sparseImageLoadARB(i2DRect, ivec2(c2), texel);
    sparseImageLoadARB(iCube, ivec3(c3), texel);
    sparseImageLoadARB(i2DArray, ivec3(c3), texel);
    sparseImageLoadARB(iCubeArray, ivec3(c3), texel);
    sparseImageLoadARB(i2DMS, ivec2(c2), 1, texel);
    sparseImageLoadARB(i2DMSArray, ivec3(c3), 2, texel);

    return texel;
}

f16vec4 testSparseTextureClamp()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureClampARB(s2D, c2, lodClamp, texel);
    sparseTextureClampARB(s2D, f16c2, f16lodClamp, texel, f16bias);
    sparseTextureClampARB(s3D, c3, lodClamp, texel);
    sparseTextureClampARB(s3D, f16c3, f16lodClamp, texel, f16bias);
    sparseTextureClampARB(sCube, c3, lodClamp, texel);
    sparseTextureClampARB(sCube, f16c3, f16lodClamp, texel, f16bias);
    sparseTextureClampARB(s2DShadow, c3, lodClamp, texel.x);
    sparseTextureClampARB(s2DShadow, f16c2, compare, f16lodClamp, texel.x, f16bias);
    sparseTextureClampARB(sCubeShadow, c4, lodClamp, texel.x);
    sparseTextureClampARB(sCubeShadow, f16c3, compare, f16lodClamp, texel.x, f16bias);
    sparseTextureClampARB(s2DArray, c3, lodClamp, texel);
    sparseTextureClampARB(s2DArray, f16c3, f16lodClamp, texel, f16bias);
    sparseTextureClampARB(sCubeArray, c4, lodClamp, texel);
    sparseTextureClampARB(sCubeArray, f16c4, f16lodClamp, texel, f16bias);
    sparseTextureClampARB(s2DArrayShadow, c4, lodClamp, texel.x);
    sparseTextureClampARB(s2DArrayShadow, f16c3, compare, f16lodClamp, texel.x);
    sparseTextureClampARB(sCubeArrayShadow, c4, compare, lodClamp, texel.x);
    sparseTextureClampARB(sCubeArrayShadow, f16c4, compare, f16lodClamp, texel.x);

    return texel;
}

f16vec4 testTextureClamp()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureClampARB(s1D, c1, lodClamp);
    texel   += textureClampARB(s1D, f16c1, f16lodClamp, f16bias);
    texel   += textureClampARB(s2D, c2, lodClamp);
    texel   += textureClampARB(s2D, f16c2, f16lodClamp, f16bias);
    texel   += textureClampARB(s3D, c3, lodClamp);
    texel   += textureClampARB(s3D, f16c3, f16lodClamp, f16bias);
    texel   += textureClampARB(sCube, c3, lodClamp);
    texel   += textureClampARB(sCube, f16c3, f16lodClamp, f16bias);
    texel.x += textureClampARB(s1DShadow, c3, lodClamp);
    texel.x += textureClampARB(s1DShadow, f16c2, compare, f16lodClamp, f16bias);
    texel.x += textureClampARB(s2DShadow, c3, lodClamp);
    texel.x += textureClampARB(s2DShadow, f16c2, compare, f16lodClamp, f16bias);
    texel.x += textureClampARB(sCubeShadow, c4, lodClamp);
    texel.x += textureClampARB(sCubeShadow, f16c3, compare, f16lodClamp, f16bias);
    texel   += textureClampARB(s1DArray, c2, lodClamp);
    texel   += textureClampARB(s1DArray, f16c2, f16lodClamp, f16bias);
    texel   += textureClampARB(s2DArray, c3, lodClamp);
    texel   += textureClampARB(s2DArray, f16c3, f16lodClamp, f16bias);
    texel   += textureClampARB(sCubeArray, c4, lodClamp);
    texel   += textureClampARB(sCubeArray, f16c4, f16lodClamp, f16bias);
    texel.x += textureClampARB(s1DArrayShadow, c3, lodClamp);
    texel.x += textureClampARB(s1DArrayShadow, f16c2, compare, f16lodClamp, f16bias);
    texel.x += textureClampARB(s2DArrayShadow, c4, lodClamp);
    texel.x += textureClampARB(s2DArrayShadow, f16c3, compare, f16lodClamp);
    texel.x += textureClampARB(sCubeArrayShadow, c4, compare, lodClamp);
    texel.x += textureClampARB(sCubeArrayShadow, f16c4, compare, f16lodClamp);

    return texel;
}

f16vec4 testSparseTextureOffsetClamp()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureOffsetClampARB(s2D, c2, offset2, lodClamp, texel);
    sparseTextureOffsetClampARB(s2D, f16c2, offset2, f16lodClamp, texel, f16bias);
    sparseTextureOffsetClampARB(s3D, c3, offset3, lodClamp, texel);
    sparseTextureOffsetClampARB(s3D, f16c3, offset3, f16lodClamp, texel, f16bias);
    sparseTextureOffsetClampARB(s2DShadow, c3, offset2, lodClamp, texel.x);
    sparseTextureOffsetClampARB(s2DShadow, f16c2, compare, offset2, f16lodClamp, texel.x, f16bias);
    sparseTextureOffsetClampARB(s2DArray, c3, offset2, lodClamp, texel);
    sparseTextureOffsetClampARB(s2DArray, f16c3, offset2, f16lodClamp, texel, f16bias);
    sparseTextureOffsetClampARB(s2DArrayShadow, c4, offset2, lodClamp, texel.x);
    sparseTextureOffsetClampARB(s2DArrayShadow, f16c3, compare, offset2, f16lodClamp, texel.x);

    return texel;
}

f16vec4 testTextureOffsetClamp()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureOffsetClampARB(s1D, c1, offset1, lodClamp);
    texel   += textureOffsetClampARB(s1D, f16c1, offset1, f16lodClamp, f16bias);
    texel   += textureOffsetClampARB(s2D, c2, offset2, lodClamp);
    texel   += textureOffsetClampARB(s2D, f16c2, offset2, f16lodClamp, f16bias);
    texel   += textureOffsetClampARB(s3D, c3, offset3, lodClamp);
    texel   += textureOffsetClampARB(s3D, f16c3, offset3, f16lodClamp, f16bias);
    texel.x += textureOffsetClampARB(s1DShadow, c3, offset1, lodClamp);
    texel.x += textureOffsetClampARB(s1DShadow, f16c2, compare, offset1, f16lodClamp, f16bias);
    texel.x += textureOffsetClampARB(s2DShadow, c3, offset2, lodClamp);
    texel.x += textureOffsetClampARB(s2DShadow, f16c2, compare, offset2, f16lodClamp, f16bias);
    texel   += textureOffsetClampARB(s1DArray, c2, offset1, lodClamp);
    texel   += textureOffsetClampARB(s1DArray, f16c2, offset1, f16lodClamp, f16bias);
    texel   += textureOffsetClampARB(s2DArray, c3, offset2, lodClamp);
    texel   += textureOffsetClampARB(s2DArray, f16c3, offset2, f16lodClamp, f16bias);
    texel.x += textureOffsetClampARB(s1DArrayShadow, c3, offset1, lodClamp);
    texel.x += textureOffsetClampARB(s1DArrayShadow, f16c2, compare, offset1, f16lodClamp, f16bias);
    texel.x += textureOffsetClampARB(s2DArrayShadow, c4, offset2, lodClamp);
    texel.x += textureOffsetClampARB(s2DArrayShadow, f16c3, compare, offset2, f16lodClamp);
    
    return texel;
}

f16vec4 testSparseTextureGradClamp()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGradClampARB(s2D, c2, dPdxy2, dPdxy2, lodClamp, texel);
    sparseTextureGradClampARB(s2D, f16c2, f16dPdxy2, f16dPdxy2, f16lodClamp, texel);
    sparseTextureGradClampARB(s3D, c3, dPdxy3, dPdxy3, lodClamp, texel);
    sparseTextureGradClampARB(s3D, f16c3, f16dPdxy3, f16dPdxy3, f16lodClamp, texel);
    sparseTextureGradClampARB(sCube, c3, dPdxy3, dPdxy3, lodClamp, texel);
    sparseTextureGradClampARB(sCube, f16c3, f16dPdxy3, f16dPdxy3, f16lodClamp, texel);
    sparseTextureGradClampARB(s2DShadow, c3, dPdxy2, dPdxy2, lodClamp, texel.x);
    sparseTextureGradClampARB(s2DShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, f16lodClamp, texel.x);
    sparseTextureGradClampARB(sCubeShadow, c4, dPdxy3, dPdxy3, lodClamp, texel.x);
    sparseTextureGradClampARB(sCubeShadow, f16c3, compare, f16dPdxy3, f16dPdxy3, f16lodClamp, texel.x);
    sparseTextureGradClampARB(s2DArray, c3, dPdxy2, dPdxy2, lodClamp, texel);
    sparseTextureGradClampARB(s2DArray, f16c3, f16dPdxy2, f16dPdxy2, f16lodClamp, texel);
    sparseTextureGradClampARB(s2DArrayShadow, c4, dPdxy2, dPdxy2, lodClamp, texel.x);
    sparseTextureGradClampARB(s2DArrayShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, f16lodClamp, texel.x);
    sparseTextureGradClampARB(sCubeArray, c4, dPdxy3, dPdxy3, lodClamp, texel);
    sparseTextureGradClampARB(sCubeArray, f16c4, f16dPdxy3, f16dPdxy3, f16lodClamp, texel);

    return texel;
}

f16vec4 testTextureGradClamp()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGradClampARB(s1D, c1, dPdxy1, dPdxy1, lodClamp);
    texel   += textureGradClampARB(s1D, f16c1, f16dPdxy1, f16dPdxy1, f16lodClamp);
    texel   += textureGradClampARB(s2D, c2, dPdxy2, dPdxy2, lodClamp);
    texel   += textureGradClampARB(s2D, f16c2, f16dPdxy2, f16dPdxy2, f16lodClamp);
    texel   += textureGradClampARB(s3D, c3, dPdxy3, dPdxy3, lodClamp);
    texel   += textureGradClampARB(s3D, f16c3, f16dPdxy3, f16dPdxy3, f16lodClamp);
    texel   += textureGradClampARB(sCube, c3, dPdxy3, dPdxy3, lodClamp);
    texel   += textureGradClampARB(sCube, f16c3, f16dPdxy3, f16dPdxy3, f16lodClamp);
    texel.x += textureGradClampARB(s1DShadow, c3, dPdxy1, dPdxy1, lodClamp);
    texel.x += textureGradClampARB(s1DShadow, f16c2, compare, f16dPdxy1, f16dPdxy1, f16lodClamp);
    texel.x += textureGradClampARB(s2DShadow, c3, dPdxy2, dPdxy2, lodClamp);
    texel.x += textureGradClampARB(s2DShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, f16lodClamp);
    texel.x += textureGradClampARB(sCubeShadow, c4, dPdxy3, dPdxy3, lodClamp);
    texel.x += textureGradClampARB(sCubeShadow, f16c3, compare, f16dPdxy3, f16dPdxy3, f16lodClamp);
    texel   += textureGradClampARB(s1DArray, c2, dPdxy1, dPdxy1, lodClamp);
    texel   += textureGradClampARB(s1DArray, f16c2, f16dPdxy1, f16dPdxy1, f16lodClamp);
    texel   += textureGradClampARB(s2DArray, c3, dPdxy2, dPdxy2, lodClamp);
    texel   += textureGradClampARB(s2DArray, f16c3, f16dPdxy2, f16dPdxy2, f16lodClamp);
    texel.x += textureGradClampARB(s1DArrayShadow, c3, dPdxy1, dPdxy1, lodClamp);
    texel.x += textureGradClampARB(s1DArrayShadow, f16c2, compare, f16dPdxy1, f16dPdxy1, f16lodClamp);
    texel.x += textureGradClampARB(s2DArrayShadow, c4, dPdxy2, dPdxy2, lodClamp);
    texel.x += textureGradClampARB(s2DArrayShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, f16lodClamp);
    texel   += textureGradClampARB(sCubeArray, c4, dPdxy3, dPdxy3, lodClamp);
    texel   += textureGradClampARB(sCubeArray, f16c4, f16dPdxy3, f16dPdxy3, f16lodClamp);

    return texel;
}

f16vec4 testSparseTextureGradOffsetClamp()
{
    f16vec4 texel = f16vec4(0.0hf);

    sparseTextureGradOffsetClampARB(s2D, c2, dPdxy2, dPdxy2, offset2, lodClamp, texel);
    sparseTextureGradOffsetClampARB(s2D, f16c2, f16dPdxy2, f16dPdxy2, offset2, f16lodClamp, texel);
    sparseTextureGradOffsetClampARB(s3D, c3, dPdxy3, dPdxy3, offset3, lodClamp, texel);
    sparseTextureGradOffsetClampARB(s3D, f16c3, f16dPdxy3, f16dPdxy3, offset3, f16lodClamp, texel);
    sparseTextureGradOffsetClampARB(s2DShadow, c3, dPdxy2, dPdxy2, offset2, lodClamp, texel.x);
    sparseTextureGradOffsetClampARB(s2DShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, offset2, f16lodClamp, texel.x);
    sparseTextureGradOffsetClampARB(s2DArray, c3, dPdxy2, dPdxy2, offset2, lodClamp, texel);
    sparseTextureGradOffsetClampARB(s2DArray, f16c3, f16dPdxy2, f16dPdxy2, offset2, f16lodClamp, texel);
    sparseTextureGradOffsetClampARB(s2DArrayShadow, c4, dPdxy2, dPdxy2, offset2, lodClamp, texel.x);
    sparseTextureGradOffsetClampARB(s2DArrayShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, offset2, f16lodClamp, texel.x);

    return texel;
}

f16vec4 testTextureGradOffsetClamp()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += textureGradOffsetClampARB(s1D, c1, dPdxy1, dPdxy1, offset1, lodClamp);
    texel   += textureGradOffsetClampARB(s1D, f16c1, f16dPdxy1, f16dPdxy1, offset1, f16lodClamp);
    texel   += textureGradOffsetClampARB(s2D, c2, dPdxy2, dPdxy2, offset2, lodClamp);
    texel   += textureGradOffsetClampARB(s2D, f16c2, f16dPdxy2, f16dPdxy2, offset2, f16lodClamp);
    texel   += textureGradOffsetClampARB(s3D, c3, dPdxy3, dPdxy3, offset3, lodClamp);
    texel   += textureGradOffsetClampARB(s3D, f16c3, f16dPdxy3, f16dPdxy3, offset3, f16lodClamp);
    texel.x += textureGradOffsetClampARB(s1DShadow, c3, dPdxy1, dPdxy1, offset1, lodClamp);
    texel.x += textureGradOffsetClampARB(s1DShadow, f16c2, compare, f16dPdxy1, f16dPdxy1, offset1, f16lodClamp);
    texel.x += textureGradOffsetClampARB(s2DShadow, c3, dPdxy2, dPdxy2, offset2, lodClamp);
    texel.x += textureGradOffsetClampARB(s2DShadow, f16c2, compare, f16dPdxy2, f16dPdxy2, offset2, f16lodClamp);
    texel   += textureGradOffsetClampARB(s1DArray, c2, dPdxy1, dPdxy1, offset1, lodClamp);
    texel   += textureGradOffsetClampARB(s1DArray, f16c2, f16dPdxy1, f16dPdxy1, offset1, f16lodClamp);
    texel   += textureGradOffsetClampARB(s2DArray, c3, dPdxy2, dPdxy2, offset2, lodClamp);
    texel   += textureGradOffsetClampARB(s2DArray, f16c3, f16dPdxy2, f16dPdxy2, offset2, f16lodClamp);
    texel.x += textureGradOffsetClampARB(s1DArrayShadow, c3, dPdxy1, dPdxy1, offset1, lodClamp);
    texel.x += textureGradOffsetClampARB(s1DArrayShadow, f16c2, compare, f16dPdxy1, f16dPdxy1, offset1, f16lodClamp);
    texel.x += textureGradOffsetClampARB(s2DArrayShadow, c4, dPdxy2, dPdxy2, offset2, lodClamp);
    texel.x += textureGradOffsetClampARB(s2DArrayShadow, f16c3, compare, f16dPdxy2, f16dPdxy2, offset2, f16lodClamp);

    return texel;
}

f16vec4 testCombinedTextureSampler()
{
    f16vec4 texel = f16vec4(0.0hf);

    texel   += texture(f16sampler1D(t1D, s), c1);
    texel   += texture(f16sampler1D(t1D, s), f16c1, f16bias);
    texel   += texture(f16sampler2D(t2D, s), c2);
    texel   += texture(f16sampler2D(t2D, s), f16c2, f16bias);
    texel   += texture(f16sampler3D(t3D, s), c3);
    texel   += texture(f16sampler3D(t3D, s), f16c3, f16bias);
    texel   += texture(f16samplerCube(tCube, s), c3);
    texel   += texture(f16samplerCube(tCube, s), f16c3, f16bias);
    texel.x += texture(f16sampler1DShadow(t1D, sShadow), c3);
    texel.x += texture(f16sampler1DShadow(t1D, sShadow), f16c2, compare, f16bias);
    texel.x += texture(f16sampler2DShadow(t2D, sShadow), c3);
    texel.x += texture(f16sampler2DShadow(t2D, sShadow), f16c2, compare, f16bias);
    texel.x += texture(f16samplerCubeShadow(tCube, sShadow), c4);
    texel.x += texture(f16samplerCubeShadow(tCube, sShadow), f16c3, compare, f16bias);
    texel   += texture(f16sampler1DArray(t1DArray, s), c2);
    texel   += texture(f16sampler1DArray(t1DArray, s), f16c2, f16bias);
    texel   += texture(f16sampler2DArray(t2DArray, s), c3);
    texel   += texture(f16sampler2DArray(t2DArray, s), f16c3, f16bias);
    texel   += texture(f16samplerCubeArray(tCubeArray, s), c4);
    texel   += texture(f16samplerCubeArray(tCubeArray, s), f16c4, f16bias);
    texel.x += texture(f16sampler1DArrayShadow(t1DArray, sShadow), c3);
    texel.x += texture(f16sampler1DArrayShadow(t1DArray, sShadow), f16c2, compare, f16bias);
    texel.x += texture(f16sampler2DArrayShadow(t2DArray, sShadow), c4);
    texel.x += texture(f16sampler2DArrayShadow(t2DArray, sShadow), f16c3, compare);
    texel   += texture(f16sampler2DRect(t2DRect, s), c2);
    texel   += texture(f16sampler2DRect(t2DRect, s), f16c2);
    texel.x += texture(f16sampler2DRectShadow(t2DRect, sShadow), c3);
    texel.x += texture(f16sampler2DRectShadow(t2DRect, sShadow), f16c2, compare);
    texel.x += texture(f16samplerCubeArrayShadow(tCubeArray, sShadow), c4, compare);
    texel.x += texture(f16samplerCubeArrayShadow(tCubeArray, sShadow), f16c4, compare);

    return texel;
}

f16vec4 testSubpassLoad()
{
    return subpassLoad(subpass) + subpassLoad(subpassMS, 2);
}

void main()
{
    f16vec4 result = f16vec4(0.0hf);

    result  += testTexture();
    result  += testTextureProj();
    result  += testTextureLod();
    result  += testTextureOffset();
    result  += testTextureLodOffset();
    result  += testTextureProjLodOffset();
    result  += testTexelFetch();
    result  += testTexelFetchOffset();
    result  += testTextureGrad();
    result  += testTextureGradOffset();
    result  += testTextureProjGrad();
    result  += testTextureProjGradoffset();
    result  += testTextureGather();
    result  += testTextureGatherOffset();
    result  += testTextureGatherOffsets();
    result  += testTextureGatherLod();
    result  += testTextureGatherLodOffset();
    result  += testTextureGatherLodOffsets();

    result    += f16vec4(testTextureSize());
    result.xy += f16vec2(testTextureQueryLod());
    result.x  += float16_t(testTextureQueryLevels());
    result.x  += float16_t(testTextureSamples());

    result  += testImageLoad();
    testImageStore(result);

    result += testSparseTexture();
    result += testSparseTextureLod();
    result += testSparseTextureOffset();
    result += testSparseTextureLodOffset();
    result += testSparseTextureGrad();
    result += testSparseTextureGradOffset();
    result += testSparseTexelFetch();
    result += testSparseTexelFetchOffset();
    result += testSparseTextureGather();
    result += testSparseTextureGatherOffset();
    result += testSparseTextureGatherOffsets();
    result += testSparseTextureGatherLod();
    result += testSparseTextureGatherLodOffset();
    result += testSparseTextureGatherLodOffsets();

    result += testSparseImageLoad();

    result += testSparseTextureClamp();
    result += testTextureClamp();
    result += testSparseTextureOffsetClamp();
    result += testTextureOffsetClamp();
    result += testSparseTextureGrad();
    result += testTextureGrad();
    result += testSparseTextureGradOffsetClamp();
    result += testTextureGradOffsetClamp();

    result += testCombinedTextureSampler();
    result += testSubpassLoad();

    fragColor = result;
}
