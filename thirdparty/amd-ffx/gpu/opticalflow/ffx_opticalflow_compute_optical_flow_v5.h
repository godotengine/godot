// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef FFX_OPTICALFLOW_COMPUTE_OPTICAL_FLOW_V5_H
#define FFX_OPTICALFLOW_COMPUTE_OPTICAL_FLOW_V5_H

#define CompareSize (4 * 2)
#define BlockSizeY 8
#define BlockSizeX 8
#define ThreadCount (4 * 16)
#define SearchRadiusX (8)
#define SearchRadiusY (8)
#define BlockCount 2

#define SearchBufferSizeX ((CompareSize + SearchRadiusX*2)/4)
#define SearchBufferSizeY  (CompareSize + SearchRadiusY*2)

FFX_GROUPSHARED FfxUInt32 pixels[CompareSize][CompareSize / 4];
FFX_GROUPSHARED FfxUInt32 searchBuffer[1][SearchBufferSizeY * SearchBufferSizeX];
#define bankBreaker 1
FFX_GROUPSHARED FfxUInt32 sadMapBuffer[4][SearchRadiusY * 2][(SearchRadiusX * 2) / 4 + bankBreaker];

#define MaxWaves 2
FFX_GROUPSHARED FfxUInt32 sWaveSad[MaxWaves];
FFX_GROUPSHARED FfxUInt32 sWaveMin[MaxWaves];

FfxUInt32 BlockSad64(FfxUInt32 blockSadSum, FfxInt32 iLocalIndex, FfxInt32 iLaneToBlockId, FfxInt32 block)
{
    if (iLaneToBlockId != block)
    {
        blockSadSum = 0u;
    }
    blockSadSum = ffxWaveSum(blockSadSum);

    if (ffxWaveLaneCount() == 32)
    {
        FfxInt32 waveId = iLocalIndex >> 5u;
        if (ffxWaveIsFirstLane())
        {
            sWaveSad[waveId] = blockSadSum;
        }
        FFX_GROUP_MEMORY_BARRIER;
        blockSadSum += sWaveSad[waveId ^ 1];
    }

    return blockSadSum;
}

FfxUInt32 SadMapMinReduction256(FfxInt32x2 iSearchId, FfxInt32 iLocalIndex)
{
    FfxUInt32 min01 = ffxMin(sadMapBuffer[0][iSearchId.y][iSearchId.x], sadMapBuffer[1][iSearchId.y][iSearchId.x]);
    FfxUInt32 min23 = ffxMin(sadMapBuffer[2][iSearchId.y][iSearchId.x], sadMapBuffer[3][iSearchId.y][iSearchId.x]);
    FfxUInt32 min0123 = ffxMin(min01, min23);
    min0123 = ffxWaveMin(min0123);

    if (ffxWaveLaneCount() == 32)
    {
        FfxInt32 waveId = iLocalIndex >> 5u;

        if (ffxWaveIsFirstLane())
        {
            sWaveMin[waveId] = min0123;
        }
        FFX_GROUP_MEMORY_BARRIER;
        min0123 = ffxMin(min0123, sWaveMin[waveId ^ 1]);
    }

    return min0123;
}

void LoadSearchBuffer(FfxInt32 iLocalIndex, FfxInt32x2 iPxPosShifted)
{
    FfxInt32 baseX = (iPxPosShifted.x - SearchRadiusX);
    FfxInt32 baseY = (iPxPosShifted.y - SearchRadiusY);

    for (FfxInt32 id = iLocalIndex; id < SearchBufferSizeX * SearchBufferSizeY; id += ThreadCount)
    {
        FfxInt32 idx = id % SearchBufferSizeX;
        FfxInt32 idy = id / SearchBufferSizeX;
        FfxInt32 x = baseX + idx * 4;
        FfxInt32 y = baseY + idy;
        searchBuffer[0][id] = LoadSecondImagePackedLuma(FfxInt32x2(x, y));
    }
    FFX_GROUP_MEMORY_BARRIER;
}

FfxUInt32x4 CalculateQSads2(FfxInt32x2 iSearchId)
{
    FfxUInt32x4 sad = ffxBroadcast4(0u);

#if FFX_OPTICALFLOW_USE_MSAD4_INSTRUCTION == 1

    FfxInt32 idx = iSearchId.y * 6 + iSearchId.x;

    sad = msad4(pixels[0][0], FfxUInt32x2(searchBuffer[0][idx],     searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[0][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[1][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[1][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[2][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[2][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[3][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[3][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[4][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[4][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[5][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[5][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[6][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[6][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);
    idx += 6;
    sad = msad4(pixels[7][0], FfxUInt32x2(searchBuffer[0][idx], searchBuffer[0][idx + 1]), sad);
    sad = msad4(pixels[7][1], FfxUInt32x2(searchBuffer[0][idx + 1], searchBuffer[0][idx + 2]), sad);

#else
    for (FfxInt32 dy = 0; dy < CompareSize; dy++)
    {
        FfxInt32 rowOffset = (iSearchId.y + dy) * SearchBufferSizeX;
        FfxUInt32 a0 = searchBuffer[0][rowOffset + iSearchId.x];
        FfxUInt32 a1 = searchBuffer[0][rowOffset + iSearchId.x + 1];
        FfxUInt32 a2 = searchBuffer[0][rowOffset + iSearchId.x + 2];
        sad += QSad(a0, a1, pixels[dy][0]);
        sad += QSad(a1, a2, pixels[dy][1]);
    }
#endif

    return sad;
}

FfxUInt32x2 abs_2(FfxInt32x2 val)
{
    FfxInt32x2 tmp = val;
    FfxInt32x2 mask = tmp >> 31;
    FfxUInt32x2 res = (tmp + mask) ^ mask;
    return res;
}

FfxUInt32 EncodeSearchCoord(FfxInt32x2 coord)
{
#if FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS == 1
    FfxUInt32x2 absCoord = FfxUInt32x2(abs_2(coord - 8));
    return FfxUInt32(absCoord.y << 12) | FfxUInt32(absCoord.x << 8) | FfxUInt32(coord.y << 4) | FfxUInt32(coord.x);
#else //FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS == 1
    return FfxUInt32(coord.y << 8) | FfxUInt32(coord.x);
#endif //FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS == 1
}

FfxInt32x2 DecodeSearchCoord(FfxUInt32 bits)
{
#if FFX_OPTICALFLOW_FIX_TOP_LEFT_BIAS == 1
    FfxInt32 dx = FfxInt32(bits & 0xfu) - SearchRadiusX;
    FfxInt32 dy = FfxInt32((bits >> 4) & 0xfu) - SearchRadiusY;

    return FfxInt32x2(dx, dy);
#else
    FfxInt32 dx = FfxInt32(bits & 0xffu) - SearchRadiusX;
    FfxInt32 dy = FfxInt32((bits >> 8) & 0xffu) - SearchRadiusY;

    return FfxInt32x2(dx, dy);
#endif
}

void PrepareSadMap(FfxInt32x2 iSearchId, FfxUInt32x4 qsad)
{
    sadMapBuffer[0][iSearchId.y][iSearchId.x] = (qsad.x << 16) | EncodeSearchCoord(FfxInt32x2(iSearchId.x * 4 + 0, iSearchId.y));
    sadMapBuffer[1][iSearchId.y][iSearchId.x] = (qsad.y << 16) | EncodeSearchCoord(FfxInt32x2(iSearchId.x * 4 + 1, iSearchId.y));
    sadMapBuffer[2][iSearchId.y][iSearchId.x] = (qsad.z << 16) | EncodeSearchCoord(FfxInt32x2(iSearchId.x * 4 + 2, iSearchId.y));
    sadMapBuffer[3][iSearchId.y][iSearchId.x] = (qsad.w << 16) | EncodeSearchCoord(FfxInt32x2(iSearchId.x * 4 + 3, iSearchId.y));
    FFX_GROUP_MEMORY_BARRIER;
}


uint ABfe(uint src, uint off, uint bits) { uint mask = (1u << bits) - 1u; return (src >> off) & mask; }
uint ABfi(uint src, uint ins, uint mask) { return (ins & mask) | (src & (~mask)); }
uint ABfiM(uint src, uint ins, uint bits) { uint mask = (1u << bits) - 1u; return (ins & mask) | (src & (~mask)); }
void MapThreads(in FfxInt32x2 iGroupId, in FfxInt32 iLocalIndex,
                out FfxInt32x2 iSearchId, out FfxInt32x2 iPxPos, out FfxInt32 iLaneToBlockId)
{
    iSearchId = FfxInt32x2(ABfe(iLocalIndex, 0u, 2u), ABfe(iLocalIndex, 2u, 4u));
    iLaneToBlockId = FfxInt32(ABfe(iLocalIndex, 1u, 1u) | (ABfe(iLocalIndex, 5u, 1u) << 1u));
    iPxPos = (iGroupId << 4u) + iSearchId * FfxInt32x2(4, 1);
}

void ComputeOpticalFlowAdvanced(FfxInt32x2 iGlobalId, FfxInt32x2 iLocalId, FfxInt32x2 iGroupId, FfxInt32 iLocalIndex)
{
    FfxInt32x2 iSearchId;
    FfxInt32x2 iPxPos;
    FfxInt32 iLaneToBlockId;
    MapThreads(iGroupId, iLocalIndex, iSearchId, iPxPos, iLaneToBlockId);

    FfxInt32x2 currentOFPos = iPxPos >> 3u;

    if (IsSceneChanged())
    {
        if ((iSearchId.y & 0x7) == 0 && (iSearchId.x & 0x1) == 0)
        {
            StoreOpticalFlow(currentOFPos, FfxInt32x2(0, 0));
        }

        return;
    }

    const FfxBoolean bUsePredictionFromPreviousLevel = (OpticalFlowPyramidLevel() != OpticalFlowPyramidLevelCount() - 1);

    FfxUInt32 packedLuma_4blocks = LoadFirstImagePackedLuma(iPxPos);

#if FFX_LOCAL_SEARCH_FALLBACK == 1
    FfxUInt32 prevPackedLuma_4blocks = LoadSecondImagePackedLuma(iPxPos);
    FfxUInt32 sad_4blocks = Sad(packedLuma_4blocks, prevPackedLuma_4blocks);
#endif //FFX_LOCAL_SEARCH_FALLBACK

    FfxInt32x2 ofGroupOffset = iGroupId << 1u;
    FfxInt32x2 pixelGroupOffset = iGroupId << 4u;

    FfxInt32x2 blockId;
    for (blockId.y = 0; blockId.y < BlockCount; blockId.y++)
    {
        for (blockId.x = 0; blockId.x < BlockCount; blockId.x++)
        {
            FfxInt32x2 currentVector = LoadRwOpticalFlow(ofGroupOffset + blockId);
            if (!bUsePredictionFromPreviousLevel)
            {
                currentVector = FfxInt32x2(0, 0);
            }

            if (iLaneToBlockId == blockId.y * 2 + blockId.x)
            {
                pixels[iSearchId.y & 0x7][iSearchId.x & 0x1] = packedLuma_4blocks;
            }

            LoadSearchBuffer(iLocalIndex, pixelGroupOffset + blockId * 8 + currentVector);

            FfxUInt32x4 qsad = CalculateQSads2(iSearchId);

            PrepareSadMap(iSearchId, qsad);
            FfxUInt32 minSad = SadMapMinReduction256(iSearchId, iLocalIndex);

            FfxInt32x2 minSadCoord = DecodeSearchCoord(minSad);
            FfxInt32x2 newVector = currentVector + minSadCoord;

#if FFX_LOCAL_SEARCH_FALLBACK == 1
            FfxUInt32 blockSadSum = BlockSad64(sad_4blocks, iLocalIndex, iLaneToBlockId, blockId.x + blockId.y * 2);
            if (OpticalFlowPyramidLevel() == 0 && blockSadSum <= (minSad >> 16u))
            {
                newVector = FfxInt32x2(0, 0);
            }
#endif //FFX_LOCAL_SEARCH_FALLBACK

            {
                StoreOpticalFlow(ofGroupOffset + blockId, newVector);
            }
        }
    }
}

#endif // FFX_OPTICALFLOW_COMPUTE_OPTICAL_FLOW_V5_H
