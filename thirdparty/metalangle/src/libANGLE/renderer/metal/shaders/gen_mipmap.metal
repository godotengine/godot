//
// Copyright 2020 The ANGLE Project. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "common.h"

using namespace rx::mtl_shader;

#define kThreadGroupXYZ                                                      \
    (kGenerateMipThreadGroupSizePerDim * kGenerateMipThreadGroupSizePerDim * \
     kGenerateMipThreadGroupSizePerDim)

#define kThreadGroupXY (kGenerateMipThreadGroupSizePerDim * kGenerateMipThreadGroupSizePerDim)
#define kThreadGroupX kGenerateMipThreadGroupSizePerDim

#define TEXEL_STORE(index, texel) \
    sR[index] = texel.r;          \
    sG[index] = texel.g;          \
    sB[index] = texel.b;          \
    sA[index] = texel.a;

#define TEXEL_LOAD(index) float4(sR[index], sG[index], sB[index], sA[index])

#define OUT_OF_BOUND_CHECK(edgeValue, targetValue, condition) \
    (condition) ? (edgeValue) : (targetValue)

struct GenMipParams
{
    uint srcLevel;
    uint numMipLevelsToGen;
};

// NOTE(hqle): For numMipLevelsToGen > 1, this function assumes the texture is power of two. If it
// is not, quality will not be good.
kernel void generate3DMipmaps(uint lIndex [[thread_index_in_threadgroup]],
                              ushort3 gIndices [[thread_position_in_grid]],
                              texture3d<float> srcTexture [[texture(0)]],
                              texture3d<float, access::write> dstMip1 [[texture(1)]],
                              texture3d<float, access::write> dstMip2 [[texture(2)]],
                              texture3d<float, access::write> dstMip3 [[texture(3)]],
                              texture3d<float, access::write> dstMip4 [[texture(4)]],
                              constant GenMipParams &options [[buffer(0)]])
{
    ushort3 mipSize    = ushort3(dstMip1.get_width(), dstMip1.get_height(), dstMip1.get_depth());
    bool validThread   = gIndices.x < mipSize.x && gIndices.y < mipSize.y && gIndices.z < mipSize.z;

    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear, mip_filter::linear);

    // NOTE(hqle): Use simd_group function whenever available. That could avoid barrier use.

    // Use struct of array style to avoid bank conflict.
    threadgroup float sR[kThreadGroupXYZ];
    threadgroup float sG[kThreadGroupXYZ];
    threadgroup float sB[kThreadGroupXYZ];
    threadgroup float sA[kThreadGroupXYZ];

    // ----- First mip level -------
    float4 texel1;
    if (validThread)
    {
        float3 texCoords = (float3(gIndices) + float3(0.5, 0.5, 0.5)) / float3(mipSize);
        texel1           = srcTexture.sample(textureSampler, texCoords, level(options.srcLevel));

        // Write to texture
        dstMip1.write(texel1, gIndices);
    }
    else
    {
        // This will invalidate all subsequent checks
        lIndex = 0xffffffff;
    }

    if (options.numMipLevelsToGen == 1)
    {
        return;
    }

    // ---- Second mip level --------

    // Write to shared memory
    TEXEL_STORE(lIndex, texel1);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be even
    if ((lIndex & 0x49) == 0)  // (lIndex & b1001001) == 0
    {
        bool3 atEdge = gIndices == (mipSize - ushort3(1));

        // (x+1, y, z)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 1), atEdge.x);
        // (x, y+1, z)
        float4 texel3 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + kThreadGroupX), atEdge.y);
        // (x, y, z+1)
        float4 texel4 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + kThreadGroupXY), atEdge.z);
        // (x+1, y+1, z)
        float4 texel5 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (kThreadGroupX + 1)),
                                           atEdge.x | atEdge.y);
        // (x+1, y, z+1)
        float4 texel6 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (kThreadGroupXY + 1)),
                                           atEdge.x | atEdge.z);
        // (x, y+1, z+1)
        float4 texel7 = OUT_OF_BOUND_CHECK(
            texel3, TEXEL_LOAD(lIndex + (kThreadGroupXY + kThreadGroupX)), atEdge.y | atEdge.z);
        // (x+1, y+1, z+1)
        float4 texel8 =
            OUT_OF_BOUND_CHECK(texel5, TEXEL_LOAD(lIndex + (kThreadGroupXY + kThreadGroupX + 1)),
                               atEdge.x | atEdge.y | atEdge.z);

        texel1 = (texel1 + texel2 + texel3 + texel4 + texel5 + texel6 + texel7 + texel8) / 8.0;

        dstMip2.write(texel1, gIndices >> 1);

        // Write to shared memory
        TEXEL_STORE(lIndex, texel1);
    }

    if (options.numMipLevelsToGen == 2)
    {
        return;
    }

    // ---- 3rd mip level --------
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be multiple of 4
    if ((lIndex & 0xdb) == 0)  // (lIndex & b11011011) == 0
    {
        mipSize      = max(mipSize >> 1, ushort3(1));
        bool3 atEdge = (gIndices >> 1) == (mipSize - ushort3(1));

        // (x+1, y, z)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 2), atEdge.x);
        // (x, y+1, z)
        float4 texel3 =
            OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + (2 * kThreadGroupX)), atEdge.y);
        // (x, y, z+1)
        float4 texel4 =
            OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + (2 * kThreadGroupXY)), atEdge.z);
        // (x+1, y+1, z)
        float4 texel5 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (2 * kThreadGroupX + 2)),
                                           atEdge.x | atEdge.y);
        // (x+1, y, z+1)
        float4 texel6 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (2 * kThreadGroupXY + 2)),
                                           atEdge.x | atEdge.z);
        // (x, y+1, z+1)
        float4 texel7 = OUT_OF_BOUND_CHECK(
            texel3, TEXEL_LOAD(lIndex + (2 * kThreadGroupXY + 2 * kThreadGroupX)),
            atEdge.y | atEdge.z);
        // (x+1, y+1, z+1)
        float4 texel8 = OUT_OF_BOUND_CHECK(
            texel5, TEXEL_LOAD(lIndex + (2 * kThreadGroupXY + 2 * kThreadGroupX + 2)),
            atEdge.x | atEdge.y | atEdge.z);

        texel1 = (texel1 + texel2 + texel3 + texel4 + texel5 + texel6 + texel7 + texel8) / 8.0;

        dstMip3.write(texel1, gIndices >> 2);

        // Write to shared memory
        TEXEL_STORE(lIndex, texel1);
    }

    if (options.numMipLevelsToGen == 3)
    {
        return;
    }

    // ---- 4th mip level --------
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be multiple of 8
    if ((lIndex & 0x1ff) == 0)  // (lIndex & b111111111) == 0
    {
        mipSize      = max(mipSize >> 1, ushort3(1));
        bool3 atEdge = (gIndices >> 2) == (mipSize - ushort3(1));

        // (x+1, y, z)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 4), atEdge.x);
        // (x, y+1, z)
        float4 texel3 =
            OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + (4 * kThreadGroupX)), atEdge.y);
        // (x, y, z+1)
        float4 texel4 =
            OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + (4 * kThreadGroupXY)), atEdge.z);
        // (x+1, y+1, z)
        float4 texel5 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (4 * kThreadGroupX + 4)),
                                           atEdge.x | atEdge.y);
        // (x+1, y, z+1)
        float4 texel6 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (4 * kThreadGroupXY + 4)),
                                           atEdge.x | atEdge.z);
        // (x, y+1, z+1)
        float4 texel7 = OUT_OF_BOUND_CHECK(
            texel3, TEXEL_LOAD(lIndex + (4 * kThreadGroupXY + 4 * kThreadGroupX)),
            atEdge.y | atEdge.z);
        // (x+1, y+1, z+1)
        float4 texel8 = OUT_OF_BOUND_CHECK(
            texel5, TEXEL_LOAD(lIndex + (4 * kThreadGroupXY + 4 * kThreadGroupX + 4)),
            atEdge.x | atEdge.y | atEdge.z);

        texel1 = (texel1 + texel2 + texel3 + texel4 + texel5 + texel6 + texel7 + texel8) / 8.0;

        dstMip4.write(texel1, gIndices >> 3);
    }
}

kernel void generate2DMipmaps(uint lIndex [[thread_index_in_threadgroup]],
                              ushort2 gIndices [[thread_position_in_grid]],
                              texture2d<float> srcTexture [[texture(0)]],
                              texture2d<float, access::write> dstMip1 [[texture(1)]],
                              texture2d<float, access::write> dstMip2 [[texture(2)]],
                              texture2d<float, access::write> dstMip3 [[texture(3)]],
                              texture2d<float, access::write> dstMip4 [[texture(4)]],
                              constant GenMipParams &options [[buffer(0)]])
{
    uint firstMipLevel = options.srcLevel + 1;
    ushort2 mipSize =
        ushort2(srcTexture.get_width(firstMipLevel), srcTexture.get_height(firstMipLevel));
    bool validThread = gIndices.x < mipSize.x && gIndices.y < mipSize.y;

    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear, mip_filter::linear);

    // NOTE(hqle): Use simd_group function whenever available. That could avoid barrier use.

    // Use struct of array style to avoid bank conflict.
    threadgroup float sR[kThreadGroupXY];
    threadgroup float sG[kThreadGroupXY];
    threadgroup float sB[kThreadGroupXY];
    threadgroup float sA[kThreadGroupXY];

    // ----- First mip level -------
    float4 texel1;
    if (validThread)
    {
        float2 texCoords = (float2(gIndices) + float2(0.5, 0.5)) / float2(mipSize);
        texel1           = srcTexture.sample(textureSampler, texCoords, level(options.srcLevel));

        // Write to texture
        dstMip1.write(texel1, gIndices);
    }
    else
    {
        // This will invalidate all subsequent checks
        lIndex = 0xffffffff;
    }

    if (options.numMipLevelsToGen == 1)
    {
        return;
    }

    // ---- Second mip level --------

    // Write to shared memory
    TEXEL_STORE(lIndex, texel1);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be even
    if ((lIndex & 0x09) == 0)  // (lIndex & b001001) == 0
    {
        bool2 atEdge = gIndices == (mipSize - ushort2(1));

        // (x+1, y)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 1), atEdge.x);
        // (x, y+1)
        float4 texel3 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + kThreadGroupX), atEdge.y);
        // (x+1, y+1)
        float4 texel4 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (kThreadGroupX + 1)),
                                           atEdge.x | atEdge.y);

        texel1 = (texel1 + texel2 + texel3 + texel4) / 4.0;

        dstMip2.write(texel1, gIndices >> 1);

        // Write to shared memory
        TEXEL_STORE(lIndex, texel1);
    }

    if (options.numMipLevelsToGen == 2)
    {
        return;
    }

    // ---- 3rd mip level --------
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be multiple of 4
    if ((lIndex & 0x1b) == 0)  // (lIndex & b011011) == 0
    {
        mipSize      = max(mipSize >> 1, ushort2(1));
        bool2 atEdge = (gIndices >> 1) == (mipSize - ushort2(1));

        // (x+1, y)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 2), atEdge.x);
        // (x, y+1)
        float4 texel3 =
            OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 2 * kThreadGroupX), atEdge.y);
        // (x+1, y+1)
        float4 texel4 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (2 * kThreadGroupX + 2)),
                                           atEdge.x | atEdge.y);

        texel1 = (texel1 + texel2 + texel3 + texel4) / 4.0;

        dstMip3.write(texel1, gIndices >> 2);

        // Write to shared memory
        TEXEL_STORE(lIndex, texel1);
    }

    if (options.numMipLevelsToGen == 3)
    {
        return;
    }

    // ---- 4th mip level --------
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be multiple of 8
    if ((lIndex & 0x3f) == 0)  // (lIndex & b111111) == 0
    {
        mipSize      = max(mipSize >> 1, ushort2(1));
        bool2 atEdge = (gIndices >> 2) == (mipSize - ushort2(1));

        // (x+1, y)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 4), atEdge.x);
        // (x, y+1)
        float4 texel3 =
            OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 4 * kThreadGroupX), atEdge.y);
        // (x+1, y+1)
        float4 texel4 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (4 * kThreadGroupX + 4)),
                                           atEdge.x | atEdge.y);

        texel1 = (texel1 + texel2 + texel3 + texel4) / 4.0;

        dstMip4.write(texel1, gIndices >> 3);
    }
}

template <typename TextureTypeR, typename TextureTypeW>
static __attribute__((always_inline)) void generateCubeOr2DArray2ndAndMoreMipmaps(
    uint lIndex,
    ushort3 gIndices,
    TextureTypeR srcTexture,
    TextureTypeW dstMip2,
    TextureTypeW dstMip3,
    TextureTypeW dstMip4,
    ushort2 mip1Size,
    float4 mip1Texel,
    threadgroup float *sR,
    threadgroup float *sG,
    threadgroup float *sB,
    threadgroup float *sA,
    constant GenMipParams &options)
{
    ushort2 mipSize = mip1Size;
    float4 texel1   = mip1Texel;

    // ---- Second mip level --------

    // Write to shared memory
    TEXEL_STORE(lIndex, texel1);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be even
    if ((lIndex & 0x09) == 0)  // (lIndex & b001001) == 0
    {
        bool2 atEdge = gIndices.xy == (mipSize - ushort2(1));

        // (x+1, y)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 1), atEdge.x);
        // (x, y+1)
        float4 texel3 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + kThreadGroupX), atEdge.y);
        // (x+1, y+1)
        float4 texel4 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (kThreadGroupX + 1)),
                                           atEdge.x | atEdge.y);

        texel1 = (texel1 + texel2 + texel3 + texel4) / 4.0;

        dstMip2.write(texel1, gIndices.xy >> 1, gIndices.z);

        // Write to shared memory
        TEXEL_STORE(lIndex, texel1);
    }

    if (options.numMipLevelsToGen == 2)
    {
        return;
    }

    // ---- 3rd mip level --------
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be multiple of 4
    if ((lIndex & 0x1b) == 0)  // (lIndex & b011011) == 0
    {
        mipSize      = max(mipSize >> 1, ushort2(1));
        bool2 atEdge = (gIndices.xy >> 1) == (mipSize - ushort2(1));

        // (x+1, y)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 2), atEdge.x);
        // (x, y+1)
        float4 texel3 =
            OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 2 * kThreadGroupX), atEdge.y);
        // (x+1, y+1)
        float4 texel4 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (2 * kThreadGroupX + 2)),
                                           atEdge.x | atEdge.y);

        texel1 = (texel1 + texel2 + texel3 + texel4) / 4.0;

        dstMip3.write(texel1, gIndices.xy >> 2, gIndices.z);

        // Write to shared memory
        TEXEL_STORE(lIndex, texel1);
    }

    if (options.numMipLevelsToGen == 3)
    {
        return;
    }

    // ---- 4th mip level --------
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Index must be multiple of 8
    if ((lIndex & 0x3f) == 0)  // (lIndex & b111111) == 0
    {
        mipSize      = max(mipSize >> 1, ushort2(1));
        bool2 atEdge = (gIndices.xy >> 2) == (mipSize - ushort2(1));

        // (x+1, y)
        // If the width of mip is 1, texel2 will equal to texel1:
        float4 texel2 = OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 4), atEdge.x);
        // (x, y+1)
        float4 texel3 =
            OUT_OF_BOUND_CHECK(texel1, TEXEL_LOAD(lIndex + 4 * kThreadGroupX), atEdge.y);
        // (x+1, y+1)
        float4 texel4 = OUT_OF_BOUND_CHECK(texel2, TEXEL_LOAD(lIndex + (4 * kThreadGroupX + 4)),
                                           atEdge.x | atEdge.y);

        texel1 = (texel1 + texel2 + texel3 + texel4) / 4.0;

        dstMip4.write(texel1, gIndices.xy >> 3, gIndices.z);
    }
}

kernel void generateCubeMipmaps(uint lIndex [[thread_index_in_threadgroup]],
                                ushort3 gIndices [[thread_position_in_grid]],
                                texturecube<float> srcTexture [[texture(0)]],
                                texturecube<float, access::write> dstMip1 [[texture(1)]],
                                texturecube<float, access::write> dstMip2 [[texture(2)]],
                                texturecube<float, access::write> dstMip3 [[texture(3)]],
                                texturecube<float, access::write> dstMip4 [[texture(4)]],
                                constant GenMipParams &options [[buffer(0)]])
{
    uint firstMipLevel = options.srcLevel + 1;
    ushort2 mip1Size =
        ushort2(srcTexture.get_width(firstMipLevel), srcTexture.get_height(firstMipLevel));
    bool validThread = gIndices.x < mip1Size.x && gIndices.y < mip1Size.y;

    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear, mip_filter::linear);

    // ----- First mip level -------
    float4 mip1Texel;
    if (validThread)
    {
        float2 texCoords = (float2(gIndices.xy) + float2(0.5, 0.5)) / float2(mip1Size);
        mip1Texel = srcTexture.sample(textureSampler, cubeTexcoords(texCoords, int(gIndices.z)),
                                      level(options.srcLevel));

        // Write to texture
        dstMip1.write(mip1Texel, gIndices.xy, gIndices.z);
    }
    else
    {
        // This will invalidate all subsequent checks
        lIndex = 0xffffffff;
    }

    if (options.numMipLevelsToGen == 1)
    {
        return;
    }

    // Use struct of array style to avoid bank conflict.
    threadgroup float sR[kThreadGroupXY];
    threadgroup float sG[kThreadGroupXY];
    threadgroup float sB[kThreadGroupXY];
    threadgroup float sA[kThreadGroupXY];

    generateCubeOr2DArray2ndAndMoreMipmaps(lIndex, gIndices, srcTexture, dstMip2, dstMip3, dstMip4,
                                           mip1Size, mip1Texel, sR, sG, sB, sA, options);
}

kernel void generate2DArrayMipmaps(uint lIndex [[thread_index_in_threadgroup]],
                                   ushort3 gIndices [[thread_position_in_grid]],
                                   texture2d_array<float> srcTexture [[texture(0)]],
                                   texture2d_array<float, access::write> dstMip1 [[texture(1)]],
                                   texture2d_array<float, access::write> dstMip2 [[texture(2)]],
                                   texture2d_array<float, access::write> dstMip3 [[texture(3)]],
                                   texture2d_array<float, access::write> dstMip4 [[texture(4)]],
                                   constant GenMipParams &options [[buffer(0)]])
{
    uint firstMipLevel = options.srcLevel + 1;
    ushort2 mip1Size =
        ushort2(srcTexture.get_width(firstMipLevel), srcTexture.get_height(firstMipLevel));
    bool validThread = gIndices.x < mip1Size.x && gIndices.y < mip1Size.y;

    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear, mip_filter::linear);

    // ----- First mip level -------
    float4 mip1Texel;
    if (validThread)
    {
        float2 texCoords = (float2(gIndices.xy) + float2(0.5, 0.5)) / float2(mip1Size);
        mip1Texel =
            srcTexture.sample(textureSampler, texCoords, gIndices.z, level(options.srcLevel));

        // Write to texture
        dstMip1.write(mip1Texel, gIndices.xy, gIndices.z);
    }
    else
    {
        // This will invalidate all subsequent checks
        lIndex = 0xffffffff;
    }

    if (options.numMipLevelsToGen == 1)
    {
        return;
    }

    // Use struct of array style to avoid bank conflict.
    threadgroup float sR[kThreadGroupXY];
    threadgroup float sG[kThreadGroupXY];
    threadgroup float sB[kThreadGroupXY];
    threadgroup float sA[kThreadGroupXY];

    generateCubeOr2DArray2ndAndMoreMipmaps(lIndex, gIndices, srcTexture, dstMip2, dstMip3, dstMip4,
                                           mip1Size, mip1Texel, sR, sG, sB, sA, options);
}
