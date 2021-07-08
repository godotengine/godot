//
// Copyright 2019 The ANGLE Project. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// blit.metal: Implements blitting texture content to current frame buffer.

#include "common.h"

using namespace rx::mtl_shader;

// function_constant(0) is already used by common.h
constant bool kPremultiplyAlpha [[function_constant(1)]];
constant bool kUnmultiplyAlpha [[function_constant(2)]];
constant int kSourceTextureType [[function_constant(3)]];   // Source color/depth texture type.
constant int kSourceTexture2Type [[function_constant(4)]];  // Source stencil texture type.
constant bool kSourceTextureType2D      = kSourceTextureType == kTextureType2D;
constant bool kSourceTextureType2DArray = kSourceTextureType == kTextureType2DArray;
constant bool kSourceTextureType2DMS    = kSourceTextureType == kTextureType2DMultisample;
constant bool kSourceTextureTypeCube    = kSourceTextureType == kTextureTypeCube;
constant bool kSourceTextureType3D      = kSourceTextureType == kTextureType3D;

constant bool kSourceTexture2Type2D      = kSourceTexture2Type == kTextureType2D;
constant bool kSourceTexture2Type2DArray = kSourceTexture2Type == kTextureType2DArray;
constant bool kSourceTexture2Type2DMS    = kSourceTexture2Type == kTextureType2DMultisample;
constant bool kSourceTexture2TypeCube    = kSourceTexture2Type == kTextureTypeCube;

struct BlitParams
{
    // 0: lower left, 1: lower right, 2: upper left
    float2 srcTexCoords[3];
    int srcLevel;   // Source texture level. Used for color & depth blitting.
    int srcLayer;   // Source texture layer. Used for color & depth blitting.
    int srcLevel2;  // Second source level. Used for stencil blitting.
    int srcLayer2;  // Second source layer. Used for stencil blitting.

    bool dstFlipViewportX;
    bool dstFlipViewportY;
    bool dstLuminance;  // destination texture is luminance. Unused by depth & stencil blitting.
};

struct BlitVSOut
{
    float4 position [[position]];
    float2 texCoords [[user(locn1)]];
};

vertex BlitVSOut blitVS(unsigned int vid [[vertex_id]], constant BlitParams &options [[buffer(0)]])
{
    BlitVSOut output;
    output.position  = float4(gCorners[vid], 0.0, 1.0);
    output.texCoords = options.srcTexCoords[vid];

    if (options.dstFlipViewportX)
    {
        output.position.x = -output.position.x;
    }
    if (!options.dstFlipViewportY)
    {
        // If viewport is not flipped, we have to flip Y in normalized device coordinates.
        // Since NDC has Y is opposite direction of viewport coodrinates.
        output.position.y = -output.position.y;
    }

    return output;
}

template <typename SrcTexture2d>
static uint2 getImageCoords(SrcTexture2d srcTexture, float2 texCoords)
{
    uint2 dimens(srcTexture.get_width(), srcTexture.get_height());
    uint2 coords = uint2(texCoords * float2(dimens));

    return coords;
}

template <typename T>
static inline vec<T, 4> blitSampleTextureMS(texture2d_ms<T> srcTexture, float2 texCoords)
{
    uint2 coords = getImageCoords(srcTexture, texCoords);
    return resolveTextureMS(srcTexture, coords);
}

template <typename T>
static inline vec<T, 4> blitSampleTexture3D(texture3d<T> srcTexture,
                                            sampler textureSampler,
                                            float2 texCoords,
                                            constant BlitParams &options)
{
    uint depth   = srcTexture.get_depth(options.srcLevel);
    float zCoord = (float(options.srcLayer) + 0.5) / float(depth);

    return srcTexture.sample(textureSampler, float3(texCoords, zCoord), level(options.srcLevel));
}

// clang-format off
#define BLIT_COLOR_FS_PARAMS(TYPE)                                                               \
    BlitVSOut input [[stage_in]],                                                                \
    texture2d<TYPE> srcTexture2d [[texture(0), function_constant(kSourceTextureType2D)]],        \
    texture2d_array<TYPE> srcTexture2dArray                                                      \
    [[texture(0), function_constant(kSourceTextureType2DArray)]],                                \
    texture2d_ms<TYPE> srcTexture2dMS [[texture(0), function_constant(kSourceTextureType2DMS)]], \
    texturecube<TYPE> srcTextureCube [[texture(0), function_constant(kSourceTextureTypeCube)]],  \
    texture3d<TYPE> srcTexture3d [[texture(0), function_constant(kSourceTextureType3D)]],        \
    sampler textureSampler [[sampler(0)]],                                                       \
    constant BlitParams &options [[buffer(0)]]
// clang-format on

#define FORWARD_BLIT_COLOR_FS_PARAMS                                                      \
    input, srcTexture2d, srcTexture2dArray, srcTexture2dMS, srcTextureCube, srcTexture3d, \
        textureSampler, options

template <typename T>
static inline MultipleColorOutputs<T> blitFS(BLIT_COLOR_FS_PARAMS(T))
{
    vec<T, 4> output;

    switch (kSourceTextureType)
    {
        case kTextureType2D:
            output = srcTexture2d.sample(textureSampler, input.texCoords, level(options.srcLevel));
            break;
        case kTextureType2DArray:
            output = srcTexture2dArray.sample(textureSampler, input.texCoords, options.srcLayer,
                                              level(options.srcLevel));
            break;
        case kTextureType2DMultisample:
            output = blitSampleTextureMS(srcTexture2dMS, input.texCoords);
            break;
        case kTextureTypeCube:
            output = srcTextureCube.sample(textureSampler,
                                           cubeTexcoords(input.texCoords, options.srcLayer),
                                           level(options.srcLevel));
            break;
        case kTextureType3D:
            output = blitSampleTexture3D(srcTexture3d, textureSampler, input.texCoords, options);
            break;
    }

    if (kPremultiplyAlpha)
    {
        output.xyz *= output.a;
    }
    else if (kUnmultiplyAlpha)
    {
        if (output.a != 0.0)
        {
            output.xyz /= output.a;
        }
    }

    if (options.dstLuminance)
    {
        output.g = output.b = output.r;
    }

    return toMultipleColorOutputs(output);
}

fragment MultipleColorOutputs<float> blitFloatFS(BLIT_COLOR_FS_PARAMS(float))
{
    return blitFS(FORWARD_BLIT_COLOR_FS_PARAMS);
}
fragment MultipleColorOutputs<int> blitIntFS(BLIT_COLOR_FS_PARAMS(int))
{
    return blitFS(FORWARD_BLIT_COLOR_FS_PARAMS);
}
fragment MultipleColorOutputs<uint> blitUIntFS(BLIT_COLOR_FS_PARAMS(uint))
{
    return blitFS(FORWARD_BLIT_COLOR_FS_PARAMS);
}

// Depth & stencil blitting.
struct FragmentDepthOut
{
    float depth [[depth(any)]];
};

static inline float sampleDepth(
    texture2d<float> srcTexture2d [[function_constant(kSourceTextureType2D)]],
    texture2d_array<float> srcTexture2dArray [[function_constant(kSourceTextureType2DArray)]],
    texture2d_ms<float> srcTexture2dMS [[function_constant(kSourceTextureType2DMS)]],
    texturecube<float> srcTextureCube [[function_constant(kSourceTextureTypeCube)]],
    float2 texCoords,
    constant BlitParams &options)
{
    float4 output;

    constexpr sampler textureSampler(mag_filter::nearest, min_filter::nearest);

    switch (kSourceTextureType)
    {
        case kTextureType2D:
            output = srcTexture2d.sample(textureSampler, texCoords, level(options.srcLevel));
            break;
        case kTextureType2DArray:
            output = srcTexture2dArray.sample(textureSampler, texCoords, options.srcLayer,
                                              level(options.srcLevel));
            break;
        case kTextureType2DMultisample:
            // Always use sample 0 for depth resolve:
            output = srcTexture2dMS.read(getImageCoords(srcTexture2dMS, texCoords), 0);
            break;
        case kTextureTypeCube:
            output =
                srcTextureCube.sample(textureSampler, cubeTexcoords(texCoords, options.srcLayer),
                                      level(options.srcLevel));
            break;
    }

    return output.r;
}

fragment FragmentDepthOut blitDepthFS(BlitVSOut input [[stage_in]],
                                      texture2d<float> srcTexture2d
                                      [[texture(0), function_constant(kSourceTextureType2D)]],
                                      texture2d_array<float> srcTexture2dArray
                                      [[texture(0), function_constant(kSourceTextureType2DArray)]],
                                      texture2d_ms<float> srcTexture2dMS
                                      [[texture(0), function_constant(kSourceTextureType2DMS)]],
                                      texturecube<float> srcTextureCube
                                      [[texture(0), function_constant(kSourceTextureTypeCube)]],
                                      constant BlitParams &options [[buffer(0)]])
{
    FragmentDepthOut re;

    re.depth = sampleDepth(srcTexture2d, srcTexture2dArray, srcTexture2dMS, srcTextureCube,
                           input.texCoords, options);

    return re;
}

static inline uint32_t sampleStencil(
    texture2d<uint32_t> srcTexture2d [[function_constant(kSourceTexture2Type2D)]],
    texture2d_array<uint32_t> srcTexture2dArray [[function_constant(kSourceTexture2Type2DArray)]],
    texture2d_ms<uint32_t> srcTexture2dMS [[function_constant(kSourceTexture2Type2DMS)]],
    texturecube<uint32_t> srcTextureCube [[function_constant(kSourceTexture2TypeCube)]],
    float2 texCoords,
    int srcLevel,
    int srcLayer)
{
    uint4 output;
    constexpr sampler textureSampler(mag_filter::nearest, min_filter::nearest);

    switch (kSourceTexture2Type)
    {
        case kTextureType2D:
            output = srcTexture2d.sample(textureSampler, texCoords, level(srcLevel));
            break;
        case kTextureType2DArray:
            output = srcTexture2dArray.sample(textureSampler, texCoords, srcLayer, level(srcLevel));
            break;
        case kTextureType2DMultisample:
            // Always use sample 0 for stencil resolve:
            output = srcTexture2dMS.read(getImageCoords(srcTexture2dMS, texCoords), 0);
            break;
        case kTextureTypeCube:
            output = srcTextureCube.sample(textureSampler, cubeTexcoords(texCoords, srcLayer),
                                           level(srcLevel));
            break;
    }

    return output.r;
}

// Write stencil to a buffer
struct BlitStencilToBufferParams
{
    float2 srcStartTexCoords;
    float2 srcTexCoordSteps;
    int srcLevel;
    int srcLayer;

    uint2 dstSize;
    uint dstBufferRowPitch;
    bool resolveMS;
};

kernel void blitStencilToBufferCS(ushort2 gIndices [[thread_position_in_grid]],
                                  texture2d<uint32_t> srcTexture2d
                                  [[texture(1), function_constant(kSourceTexture2Type2D)]],
                                  texture2d_array<uint32_t> srcTexture2dArray
                                  [[texture(1), function_constant(kSourceTexture2Type2DArray)]],
                                  texture2d_ms<uint32_t> srcTexture2dMS
                                  [[texture(1), function_constant(kSourceTexture2Type2DMS)]],
                                  texturecube<uint32_t> srcTextureCube
                                  [[texture(1), function_constant(kSourceTexture2TypeCube)]],
                                  constant BlitStencilToBufferParams &options [[buffer(0)]],
                                  device uchar *buffer [[buffer(1)]])
{
    if (gIndices.x >= options.dstSize.x || gIndices.y >= options.dstSize.y)
    {
        return;
    }

    float2 srcTexCoords = options.srcStartTexCoords + float2(gIndices) * options.srcTexCoordSteps;
    
    if (kSourceTexture2Type == kTextureType2DMultisample && !options.resolveMS)
    {
        uint samples      = srcTexture2dMS.get_num_samples();
        uint2 imageCoords = getImageCoords(srcTexture2dMS, srcTexCoords);
        uint bufferOffset = options.dstBufferRowPitch * gIndices.y + samples * gIndices.x;

        for (uint sample = 0; sample < samples; ++sample)
        {
            uint stencilPerSample         = srcTexture2dMS.read(imageCoords, sample).r;
            buffer[bufferOffset + sample] = static_cast<uchar>(stencilPerSample);
        }
    }
    else
    {
        uint32_t stencil =
            sampleStencil(srcTexture2d, srcTexture2dArray, srcTexture2dMS, srcTextureCube,
                          srcTexCoords, options.srcLevel, options.srcLayer);

        buffer[options.dstBufferRowPitch * gIndices.y + gIndices.x] = static_cast<uchar>(stencil);
    }
}

#if __METAL_VERSION__ >= 210

struct FragmentStencilOut
{
    uint32_t stencil [[stencil]];
};

struct FragmentDepthStencilOut
{
    float depth [[depth(any)]];
    uint32_t stencil [[stencil]];
};

fragment FragmentStencilOut blitStencilFS(
    BlitVSOut input [[stage_in]],
    texture2d<uint32_t> srcTexture2d [[texture(1), function_constant(kSourceTexture2Type2D)]],
    texture2d_array<uint32_t> srcTexture2dArray
    [[texture(1), function_constant(kSourceTexture2Type2DArray)]],
    texture2d_ms<uint32_t> srcTexture2dMS
    [[texture(1), function_constant(kSourceTexture2Type2DMS)]],
    texturecube<uint32_t> srcTextureCube [[texture(1), function_constant(kSourceTexture2TypeCube)]],
    constant BlitParams &options [[buffer(0)]])
{
    FragmentStencilOut re;

    re.stencil = sampleStencil(srcTexture2d, srcTexture2dArray, srcTexture2dMS, srcTextureCube,
                               input.texCoords, options.srcLevel2, options.srcLayer2);

    return re;
}

fragment FragmentDepthStencilOut blitDepthStencilFS(
    BlitVSOut input [[stage_in]],
    // Source depth texture
    texture2d<float> srcDepthTexture2d [[texture(0), function_constant(kSourceTextureType2D)]],
    texture2d_array<float> srcDepthTexture2dArray
    [[texture(0), function_constant(kSourceTextureType2DArray)]],
    texture2d_ms<float> srcDepthTexture2dMS
    [[texture(0), function_constant(kSourceTextureType2DMS)]],
    texturecube<float> srcDepthTextureCube
    [[texture(0), function_constant(kSourceTextureTypeCube)]],

    // Source stencil texture
    texture2d<uint32_t> srcStencilTexture2d
    [[texture(1), function_constant(kSourceTexture2Type2D)]],
    texture2d_array<uint32_t> srcStencilTexture2dArray
    [[texture(1), function_constant(kSourceTexture2Type2DArray)]],
    texture2d_ms<uint32_t> srcStencilTexture2dMS
    [[texture(1), function_constant(kSourceTexture2Type2DMS)]],
    texturecube<uint32_t> srcStencilTextureCube
    [[texture(1), function_constant(kSourceTexture2TypeCube)]],

    constant BlitParams &options [[buffer(0)]])
{
    FragmentDepthStencilOut re;

    re.depth = sampleDepth(srcDepthTexture2d, srcDepthTexture2dArray, srcDepthTexture2dMS,
                           srcDepthTextureCube, input.texCoords, options);
    re.stencil =
        sampleStencil(srcStencilTexture2d, srcStencilTexture2dArray, srcStencilTexture2dMS,
                      srcStencilTextureCube, input.texCoords, options.srcLevel2, options.srcLayer2);
    return re;
}
#endif
