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

#ifndef FFX_OPTICALFLOW_GENERATE_SCD_HISTOGRAM_H
#define FFX_OPTICALFLOW_GENERATE_SCD_HISTOGRAM_H

#define LBASE 10

FFX_GROUPSHARED FfxUInt32 scdBuffer[256 * LBASE];

void GenerateSceneChangeDetectionHistogram(FfxInt32x3 iGlobalId, FfxInt32x2 iLocalId, FfxInt32 iLocalIndex, FfxInt32x2 iGroupId, FfxInt32x2 iGroupSize)
{
    FFX_STATIC const FfxUInt32 HistogramsPerDim = 3;

    FfxUInt32 divX = DisplaySize().x / HistogramsPerDim;
    FfxUInt32 divY = DisplaySize().y / HistogramsPerDim;

    FfxUInt32 nx = iGlobalId.z % HistogramsPerDim;
    FfxUInt32 ny = iGlobalId.z / HistogramsPerDim;
    FfxUInt32 startX = divX * nx;
    FfxUInt32 startY = divY * ny;
    FfxUInt32 stopX  = startX + divX;
    FfxUInt32 stopY  = startY + divY;

    const FfxUInt32 bufferOffset = iLocalIndex * LBASE;

    for (FfxInt32 i = 0; i < LBASE; i++)
    {
        scdBuffer[bufferOffset + i] = 0;
    }
    FFX_GROUP_MEMORY_BARRIER;

    FfxInt32x2 coord = FfxInt32x2(startX + (4 * iGlobalId.x), startY + iGlobalId.y);
    if (coord.x < stopX)
    {
        for (; coord.y < stopY; coord.y += 128)
        {
            FfxUInt32x4 color = FfxUInt32x4(
                LoadOpticalFlowInput(coord + FfxInt32x2(0, 0)),
                LoadOpticalFlowInput(coord + FfxInt32x2(1, 0)),
                LoadOpticalFlowInput(coord + FfxInt32x2(2, 0)),
                LoadOpticalFlowInput(coord + FfxInt32x2(3, 0))
            );
            color *= LBASE;

            FfxUInt32 scramblingOffset = iLocalIndex % LBASE;

#if defined(FFX_HLSL)
            InterlockedAdd(scdBuffer[color.x + scramblingOffset], 1);
            InterlockedAdd(scdBuffer[color.y + scramblingOffset], 1);
            InterlockedAdd(scdBuffer[color.z + scramblingOffset], 1);
            InterlockedAdd(scdBuffer[color.w + scramblingOffset], 1);
#elif defined(FFX_GLSL)
            atomicAdd(scdBuffer[color.x + scramblingOffset], 1);
            atomicAdd(scdBuffer[color.y + scramblingOffset], 1);
            atomicAdd(scdBuffer[color.z + scramblingOffset], 1);
            atomicAdd(scdBuffer[color.w + scramblingOffset], 1);
#endif
        }
    }
    FFX_GROUP_MEMORY_BARRIER;

    FfxUInt32 value = 0;
    for (FfxInt32 i = 0; i < LBASE; i++)
    {
        value += scdBuffer[bufferOffset + i];
    }

    FfxUInt32 histogramStart = (iGroupSize.x * iGroupSize.y) * iGlobalId.z;
    AtomicIncrementSCDHistogram(FfxInt32(histogramStart + iLocalIndex), value);
}

#endif // FFX_OPTICALFLOW_GENERATE_SCD_HISTOGRAM_H
