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

#ifndef FFX_FRAMEINTERPOLATION_COMPUTE_INPAINTING_PYRAMID_H
#define FFX_FRAMEINTERPOLATION_COMPUTE_INPAINTING_PYRAMID_H

//--------------------------------------------------------------------------------------
// Buffer definitions - global atomic counter
//--------------------------------------------------------------------------------------

FFX_GROUPSHARED FfxUInt32 spdCounter;
FFX_GROUPSHARED FfxFloat32 spdIntermediateR[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateG[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateB[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateA[16][16];

FfxFloat32x4 SpdLoadSourceImage(FfxInt32x2 tex, FfxUInt32 slice)
{
    FfxFloat32x4 fColor = LoadFrameInterpolationOutput(tex) * FfxFloat32(DisplaySize().x > 0);

    // reverse sample weights
    fColor.w = ffxSaturate(1.0f - fColor.w);


    if (tex.x < InterpolationRectBase().x || tex.x >= (InterpolationRectSize().x + InterpolationRectBase().x) || tex.y < InterpolationRectBase().y ||
        tex.y >= (InterpolationRectSize().y + InterpolationRectBase().y))
    {
        fColor.w = 0.0f; // don't take contributions from outside of the interpolation rect
    }

    return fColor;
}

FfxFloat32x4 SpdLoad(FfxInt32x2 tex, FfxUInt32 slice)
{
    return RWLoadInpaintingPyramid(tex, 5);
}

void SpdStore(FfxInt32x2 pix, FfxFloat32x4 outValue, FfxUInt32 index, FfxUInt32 slice)
{
    StoreInpaintingPyramid(pix, outValue, index);
}

void SpdIncreaseAtomicCounter(FfxUInt32 slice)
{
    AtomicIncreaseCounter(COUNTER_SPD, spdCounter);
}

FfxUInt32 SpdGetAtomicCounter()
{
    return spdCounter;
}
void SpdResetAtomicCounter(FfxUInt32 slice)
{
    StoreCounter(COUNTER_SPD, 0);
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
    FfxFloat32x4 w = FfxFloat32x4(v0.w, v1.w, v2.w, v3.w);

    FfxFloat32 sum = (w[0] + w[1] + w[2] + w[3]);

    if (sum == 0.0f) {
        return FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
    }

    return (v0 * w[0] + v1 * w[1] + v2 * w[2] + v3 * w[3]) / sum;
}

#include "../spd/ffx_spd.h"

void computeFrameinterpolationInpaintingPyramid(FfxInt32x3 iGroupId, FfxInt32 iLocalIndex)
{
    SpdDownsample(
        FfxUInt32x2(iGroupId.xy),
        FfxUInt32(iLocalIndex),
        FfxUInt32(NumMips()),
        FfxUInt32(NumWorkGroups()),
        FfxUInt32(iGroupId.z),
        FfxUInt32x2(WorkGroupOffset()));
}

#endif // FFX_FRAMEINTERPOLATION_COMPUTE_INPAINTING_PYRAMID_H
