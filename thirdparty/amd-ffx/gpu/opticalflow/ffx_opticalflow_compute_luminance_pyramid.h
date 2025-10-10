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

#ifndef FFX_OPTICALFLOW_COMPUTE_LUMINANCE_PYRAMID_H
#define FFX_OPTICALFLOW_COMPUTE_LUMINANCE_PYRAMID_H

FFX_GROUPSHARED FfxFloat32 spdIntermediateR[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateG[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateB[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateA[16][16];

void SPD_IncreaseAtomicCounter(inout FfxUInt32 spdCounter)
{
}

void SPD_ResetAtomicCounter()
{
}

FfxFloat32x4 SpdLoadSourceImage(FfxFloat32x2 tex, FfxUInt32 slice)
{
    FfxFloat32 luma = LoadRwOpticalFlowInput(FfxInt32x2(tex));
    return FfxFloat32x4(luma, 0, 0, 0);
}

FfxFloat32x4 SpdLoad(FfxInt32x2 tex, FfxUInt32 slice)
{
    return FfxFloat32x4(0, 0, 0, 0);
}

void SpdStore(FfxInt32x2 pix, FfxFloat32x4 outValue, FfxUInt32 index, FfxUInt32 slice)
{
    SPD_SetMipmap(pix, index, outValue.r);
}

void SpdIncreaseAtomicCounter(FfxUInt32 slice)
{
}

FfxUInt32 SpdGetAtomicCounter()
{
    return 0;
}

void SpdResetAtomicCounter(FfxUInt32 slice)
{
    SPD_ResetAtomicCounter();
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
FfxFloat32x4 SpdReduce4(FfxFloat32x4 v0, FfxFloat32x4 v1, FfxFloat32x4 v2, FfxFloat32x4 v3)
{
    return (v0 + v1 + v2 + v3) * 0.25;
}

#ifdef FFX_HALF
#undef FFX_HALF
#endif

// https://github.com/GPUOpen-Effects/FidelityFX-SPD/blob/master/docs/FidelityFX_SPD.pdf
#include "../spd/ffx_spd.h"

void ComputeOpticalFlowInputPyramid(FfxInt32x2 iGroupId, FfxInt32 iLocalIndex)
{
    SpdDownsample(
        FfxUInt32x2(iGroupId.xy),
        FfxUInt32(iLocalIndex),
        6, // mip levels to generate
        FfxUInt32(NumWorkGroups()),
        1 // single slice
    );
}

#endif // FFX_OPTICALFLOW_COMPUTE_LUMINANCE_PYRAMID_H
