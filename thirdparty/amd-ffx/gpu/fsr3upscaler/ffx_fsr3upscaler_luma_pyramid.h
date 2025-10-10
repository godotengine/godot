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

FFX_GROUPSHARED FfxUInt32 spdCounter;

void SpdIncreaseAtomicCounter(FfxUInt32 slice)
{
    SPD_IncreaseAtomicCounter(spdCounter);
}

FfxUInt32 SpdGetAtomicCounter()
{
    return spdCounter;
}

void SpdResetAtomicCounter(FfxUInt32 slice)
{
    SPD_ResetAtomicCounter();
}

#ifndef SPD_PACKED_ONLY
FFX_GROUPSHARED FfxFloat32 spdIntermediateR[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateG[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateB[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateA[16][16];

FFX_STATIC const FfxInt32 LOG_LUMA        = 0;
FFX_STATIC const FfxInt32 LUMA            = 1;
FFX_STATIC const FfxInt32 DEPTH_IN_METERS = 2;

FfxFloat32x4 SpdLoadSourceImage(FfxFloat32x2 iPxPos, FfxUInt32 slice)
{
    //We assume linear data. if non-linear input (sRGB, ...),
    //then we should convert to linear first and back to sRGB on output.
    const FfxInt32x2 iPxSamplePos = ClampLoad(FfxInt32x2(iPxPos), FfxInt32x2(0, 0), FfxInt32x2(RenderSize()));

    const FfxFloat32 fLuma                  = LoadCurrentLuma(iPxSamplePos);
    const FfxFloat32 fLogLuma               = ffxMax(FSR3UPSCALER_EPSILON, log(fLuma));
    const FfxFloat32 fFarthestDepthInMeters = LoadFarthestDepth(iPxSamplePos);

    FfxFloat32x4 fOutput        = FfxFloat32x4(0.0f, 0.0f, 0.0f, 0.0f);
    fOutput[LOG_LUMA]           = fLogLuma;
    fOutput[LUMA]               = fLuma;
    fOutput[DEPTH_IN_METERS]    = fFarthestDepthInMeters;

    return fOutput;
}

FfxFloat32x4 SpdLoad(FfxInt32x2 tex, FfxUInt32 slice)
{
    return FfxFloat32x4(RWLoadPyramid(tex, 5), 0, 0);
}

FfxFloat32x4 SpdReduce4(FfxFloat32x4 v0, FfxFloat32x4 v1, FfxFloat32x4 v2, FfxFloat32x4 v3)
{
    return (v0 + v1 + v2 + v3) * 0.25f;
}

void SpdStore(FfxInt32x2 pix, FfxFloat32x4 outValue, FfxUInt32 index, FfxUInt32 slice)
{
    if (index == 5)
    {
        StorePyramid(pix, outValue.xy, index);
    }
    else if (index == 0) {
        StoreFarthestDepthMip1(pix, outValue[DEPTH_IN_METERS]);
    }

    if (index == MipCount() - 1) { //accumulate on 1x1 level

        if (all(FFX_EQUAL(pix, FfxInt32x2(0, 0))))
        {
            FfxFloat32x4 frameInfo          = LoadFrameInfo();
            const FfxFloat32 fSceneAvgLuma  = outValue[LUMA];
            const FfxFloat32 fPrevLogLuma   = frameInfo[FRAME_INFO_LOG_LUMA];
            FfxFloat32 fLogLuma             = outValue[LOG_LUMA];

            if (fPrevLogLuma < resetAutoExposureAverageSmoothing) // Compare Lavg, so small or negative values
            {
                fLogLuma = fPrevLogLuma + (fLogLuma - fPrevLogLuma) * (1.0f - exp(-DeltaTime()));
                fLogLuma = ffxMax(0.0f, fLogLuma);
            }

            frameInfo[FRAME_INFO_EXPOSURE]             = ComputeAutoExposureFromLavg(fLogLuma);
            frameInfo[FRAME_INFO_LOG_LUMA]             = fLogLuma;
            frameInfo[FRAME_INFO_SCENE_AVERAGE_LUMA]   = fSceneAvgLuma;

            StoreFrameInfo(frameInfo);
        }
    }
}

FfxFloat32x4 SpdLoadIntermediate(FfxUInt32 x, FfxUInt32 y)
{
    return FfxFloat32x4(
        spdIntermediateR[x][y],
        spdIntermediateG[x][y],
        spdIntermediateB[x][y],
        spdIntermediateA[x][y]);
}
void SpdStoreIntermediate(FfxUInt32 x, FfxUInt32 y, FfxFloat32x4 value)
{
    spdIntermediateR[x][y] = value.x;
    spdIntermediateG[x][y] = value.y;
    spdIntermediateB[x][y] = value.z;
    spdIntermediateA[x][y] = value.w;
}

#endif

// define fetch and store functions Packed
#if FFX_HALF

FFX_GROUPSHARED FfxFloat16x2 spdIntermediateRG[16][16];
FFX_GROUPSHARED FfxFloat16x2 spdIntermediateBA[16][16];

FfxFloat16x4 SpdLoadSourceImageH(FfxFloat32x2 tex, FfxUInt32 slice)
{
    return FfxFloat16x4(0, 0, 0, 0);
}

FfxFloat16x4 SpdLoadH(FfxInt32x2 p, FfxUInt32 slice)
{
    return FfxFloat16x4(0, 0, 0, 0);
}

void SpdStoreH(FfxInt32x2 p, FfxFloat16x4 value, FfxUInt32 mip, FfxUInt32 slice)
{
}

FfxFloat16x4 SpdLoadIntermediateH(FfxUInt32 x, FfxUInt32 y)
{
    return FfxFloat16x4(
        spdIntermediateRG[x][y].x,
        spdIntermediateRG[x][y].y,
        spdIntermediateBA[x][y].x,
        spdIntermediateBA[x][y].y);
}

void SpdStoreIntermediateH(FfxUInt32 x, FfxUInt32 y, FfxFloat16x4 value)
{
    spdIntermediateRG[x][y] = value.xy;
    spdIntermediateBA[x][y] = value.zw;
}

FfxFloat16x4 SpdReduce4H(FfxFloat16x4 v0, FfxFloat16x4 v1, FfxFloat16x4 v2, FfxFloat16x4 v3)
{
    return (v0 + v1 + v2 + v3) * FfxFloat16(0.25);
}
#endif

#include "../spd/ffx_spd.h"

void ComputeAutoExposure(FfxUInt32x3 WorkGroupId, FfxUInt32 LocalThreadIndex)
{
#if FFX_HALF
    SpdDownsampleH(
        FfxUInt32x2(WorkGroupId.xy),
        FfxUInt32(LocalThreadIndex),
        FfxUInt32(MipCount()),
        FfxUInt32(NumWorkGroups()),
        FfxUInt32(WorkGroupId.z),
        FfxUInt32x2(WorkGroupOffset()));
#else
    SpdDownsample(
        FfxUInt32x2(WorkGroupId.xy),
        FfxUInt32(LocalThreadIndex),
        FfxUInt32(MipCount()),
        FfxUInt32(NumWorkGroups()),
        FfxUInt32(WorkGroupId.z),
        FfxUInt32x2(WorkGroupOffset()));
#endif
}
