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

#ifndef FFX_OPTICALFLOW_COMPUTE_SCD_DIVERGENCE_H
#define FFX_OPTICALFLOW_COMPUTE_SCD_DIVERGENCE_H

FFX_GROUPSHARED FfxFloat32   sourceHistogram[256];
FFX_GROUPSHARED FfxFloat32   filteredHistogram[256];
FFX_GROUPSHARED FfxFloat32   tempBuffer[256];
FFX_GROUPSHARED FfxFloat32x2 tempBuffer2[256];

void ComputeSCDHistogramsDivergence(FfxInt32x3 iGlobalId, FfxInt32x2 iLocalId, FfxInt32 iLocalIndex, FfxInt32x2 iGroupId, FfxInt32x2 iGroupSize)
{
    FFX_STATIC const FfxFloat32 Factor = 1000000.0;
    FFX_STATIC const FfxInt32   WhereToStop = 3*9 - 1;
    FFX_STATIC const FfxInt32   HistogramCount = 3 * 3;

    FFX_STATIC const FfxFloat32 Kernel[] = {
        0.0088122291, 0.027143577, 0.065114059, 0.12164907, 0.17699835, 0.20056541
    };

    sourceHistogram[iLocalIndex] = FfxFloat32(LoadRwSCDHistogram(iGlobalId.x));
    FFX_GROUP_MEMORY_BARRIER;

    const FfxInt32 kernelShift = -5;
    const FfxInt32 indexToRead = iLocalIndex + kernelShift;

    FfxFloat32 val = 0.0;
    val += Kernel[0] * sourceHistogram[ffxClamp(indexToRead + 0, 0, 255)];
    val += Kernel[1] * sourceHistogram[ffxClamp(indexToRead + 1, 0, 255)];
    val += Kernel[2] * sourceHistogram[ffxClamp(indexToRead + 2, 0, 255)];
    val += Kernel[3] * sourceHistogram[ffxClamp(indexToRead + 3, 0, 255)];
    val += Kernel[4] * sourceHistogram[ffxClamp(indexToRead + 4, 0, 255)];
    val += Kernel[5] * sourceHistogram[ffxClamp(indexToRead + 5, 0, 255)];
    val += Kernel[4] * sourceHistogram[ffxClamp(indexToRead + 6, 0, 255)];
    val += Kernel[3] * sourceHistogram[ffxClamp(indexToRead + 7, 0, 255)];
    val += Kernel[2] * sourceHistogram[ffxClamp(indexToRead + 8, 0, 255)];
    val += Kernel[1] * sourceHistogram[ffxClamp(indexToRead + 9, 0, 255)];
    val += Kernel[0] * sourceHistogram[ffxClamp(indexToRead + 10, 0, 255)];

    val += 1.0;

    if (iGlobalId.y == 0)
    {
        if (iLocalIndex == 0)
            filteredHistogram[255] = 1.0;
        else
            filteredHistogram[iLocalIndex - 1] = val;
    }
    else if (iGlobalId.y == 1)
    {
        filteredHistogram[iLocalIndex] = val;
    }
    else if (iGlobalId.y == 2)
    {
        if (iLocalIndex == 255)
            filteredHistogram[0] = 1.0;
        else
            filteredHistogram[iLocalIndex + 1] = val;
    }
    FFX_GROUP_MEMORY_BARRIER;

    tempBuffer[iLocalIndex] = filteredHistogram[iLocalIndex];
    FFX_GROUP_MEMORY_BARRIER;

    if (iLocalIndex < 128) tempBuffer[iLocalIndex] += tempBuffer[iLocalIndex + 128];
    FFX_GROUP_MEMORY_BARRIER;

    if (iLocalIndex < 64) tempBuffer[iLocalIndex] += tempBuffer[iLocalIndex + 64];
    FFX_GROUP_MEMORY_BARRIER;

    if (iLocalIndex < 32) tempBuffer[iLocalIndex] += tempBuffer[iLocalIndex + 32];
    if (iLocalIndex < 16) tempBuffer[iLocalIndex] += tempBuffer[iLocalIndex + 16];
    if (iLocalIndex < 8 ) tempBuffer[iLocalIndex] += tempBuffer[iLocalIndex + 8];
    if (iLocalIndex < 4 ) tempBuffer[iLocalIndex] += tempBuffer[iLocalIndex + 4];
    if (iLocalIndex < 2 ) tempBuffer[iLocalIndex] += tempBuffer[iLocalIndex + 2];
    if (iLocalIndex < 1 ) tempBuffer[iLocalIndex] += tempBuffer[iLocalIndex + 1];
    FFX_GROUP_MEMORY_BARRIER;

    filteredHistogram[iLocalIndex] /= tempBuffer[0];

    FfxFloat32 currentFilteredHistogramsValue = filteredHistogram[iLocalIndex];
    FfxFloat32 previousHistogramsValue = LoadRwSCDPreviousHistogram(iGlobalId.x);

    tempBuffer2[iLocalIndex] = FfxFloat32x2(
        currentFilteredHistogramsValue * log(currentFilteredHistogramsValue / previousHistogramsValue),
        previousHistogramsValue * log(previousHistogramsValue / currentFilteredHistogramsValue)
    );
    FFX_GROUP_MEMORY_BARRIER;

    if (iLocalIndex < 128) tempBuffer2[iLocalIndex] += tempBuffer2[iLocalIndex + 128];
    FFX_GROUP_MEMORY_BARRIER;

    if (iLocalIndex < 64) tempBuffer2[iLocalIndex] += tempBuffer2[iLocalIndex + 64];
    FFX_GROUP_MEMORY_BARRIER;

    if (iLocalIndex < 32) tempBuffer2[iLocalIndex] += tempBuffer2[iLocalIndex + 32];
    if (iLocalIndex < 16) tempBuffer2[iLocalIndex] += tempBuffer2[iLocalIndex + 16];
    if (iLocalIndex < 8 ) tempBuffer2[iLocalIndex] += tempBuffer2[iLocalIndex + 8];
    if (iLocalIndex < 4 ) tempBuffer2[iLocalIndex] += tempBuffer2[iLocalIndex + 4];
    if (iLocalIndex < 2 ) tempBuffer2[iLocalIndex] += tempBuffer2[iLocalIndex + 2];

    if (iLocalIndex == 0)
    {
        FfxFloat32x2 sum = tempBuffer2[0] + tempBuffer2[1];

        FfxFloat32 resFloat = 1 - exp(-(abs(sum.x) + abs(sum.y)));
        FfxUInt32 resUInt = FfxUInt32((resFloat / FfxFloat32(HistogramCount)) * Factor);
        AtomicIncrementSCDTemp(iGlobalId.y, resUInt);

        FfxUInt32 oldFinishedGroupCount = AtomicIncrementSCDOutput(SCD_OUTPUT_COMPLETED_WORKGROUPS_SLOT, 1);
        if (oldFinishedGroupCount == WhereToStop)
        {
            FfxUInt32 res0 = LoadRwSCDTemp(0);
            FfxUInt32 res1 = LoadRwSCDTemp(1);
            FfxUInt32 res2 = LoadRwSCDTemp(2);
            FfxFloat32 sceneChangeValue = ffxMin(res0, ffxMin(res1, res2)) / Factor;

            FfxUInt32 history = LoadRwSCDOutput(SCD_OUTPUT_HISTORY_BITS_SLOT) << 1;
            if (CrossedSceneChangeThreshold(sceneChangeValue))
            {
                history |= 1;
            }
            StoreSCDOutput(SCD_OUTPUT_SCENE_CHANGE_SLOT, ffxAsUInt32(sceneChangeValue));
            StoreSCDOutput(SCD_OUTPUT_HISTORY_BITS_SLOT, history);
            StoreSCDOutput(SCD_OUTPUT_COMPLETED_WORKGROUPS_SLOT, 0);

            ResetSCDTemp();
        }
    }

    if (iGlobalId.y == 1)
    {
        StoreSCDPreviousHistogram(iGlobalId.x, currentFilteredHistogramsValue);

        StoreSCDHistogram(iGlobalId.x, 0);
    }
}

#endif // FFX_OPTICALFLOW_COMPUTE_SCD_DIVERGENCE_H
