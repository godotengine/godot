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

#include "../ffx_core.h"

#if FFX_HALF
    #define FFX_SPD_PACKED_ONLY 1
#endif // FFX_HALF

#if FFX_SPD_OPTION_LINEAR_SAMPLE
    #define SPD_LINEAR_SAMPLER 1
#endif // FFX_SPD_OPTION_LINEAR_SAMPLE

#if FFX_SPD_OPTION_WAVE_INTEROP_LDS
    #define FFX_SPD_NO_WAVE_OPERATIONS 1
#endif // FFX_SPD_OPTION_WAVE_INTEROP_LDS

FFX_GROUPSHARED FfxUInt32 spdCounter;

void SpdIncreaseAtomicCounter(FfxUInt32 slice)
{
    IncreaseAtomicCounter(slice, spdCounter);
}

FfxUInt32 SpdGetAtomicCounter()
{
    return spdCounter;
}

void SpdResetAtomicCounter(FfxUInt32 slice)
{
    ResetAtomicCounter(slice);
}

#if FFX_HALF

FFX_GROUPSHARED FfxFloat16x2 spdIntermediateRG[16][16];
FFX_GROUPSHARED FfxFloat16x2 spdIntermediateBA[16][16];

FfxFloat16x4 SpdLoadSourceImageH(FfxInt32x2 tex, FfxUInt32 slice)
{
#if defined SPD_LINEAR_SAMPLER
    return SampleSrcImageH(tex, slice);
#else
    return LoadSrcImageH(tex, slice);
#endif // SPD_LINEAR_SAMPLER
}

FfxFloat16x4 SpdLoadH(FfxInt32x2 p, FfxUInt32 slice)
{
    return LoadMidMipH(p, slice);
}

void SpdStoreH(FfxInt32x2 pix, FfxFloat16x4 value, FfxUInt32 mip, FfxUInt32 slice)
{
    if (mip == 5)
        StoreMidMipH(value, pix, slice);
    else
        StoreSrcMipH(value, pix, slice, mip + 1);
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
#if FFX_SPD_OPTION_DOWNSAMPLE_FILTER == 1
    return min(min(v0, v1), min(v2, v3));
#elif FFX_SPD_OPTION_DOWNSAMPLE_FILTER == 2
    return max(max(v0, v1), max(v2, v3));
#else
    return (v0 + v1 + v2 + v3) * FfxFloat16(0.25);
#endif
}

#else

FFX_GROUPSHARED FfxFloat32 spdIntermediateR[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateG[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateB[16][16];
FFX_GROUPSHARED FfxFloat32 spdIntermediateA[16][16];

FfxFloat32x4 SpdLoadSourceImage(FfxInt32x2 tex, FfxUInt32 slice)
{
#if defined SPD_LINEAR_SAMPLER
    return SampleSrcImage(tex, slice);
#else
    return LoadSrcImage(tex, slice);
#endif // SPD_LINEAR_SAMPLER
}

FfxFloat32x4 SpdLoad(FfxInt32x2 tex, FfxUInt32 slice)
{
    return LoadMidMip(tex, slice);
}

void SpdStore(FfxInt32x2 pix, FfxFloat32x4 outValue, FfxUInt32 mip, FfxUInt32 slice)
{
    if (mip == 5)
        StoreMidMip(outValue, pix, slice);
    else
        StoreSrcMip(outValue, pix, slice, mip + 1);
}

FfxFloat32x4 SpdLoadIntermediate(FfxUInt32 x, FfxUInt32 y)
{
    return FfxFloat32x4(spdIntermediateR[x][y], spdIntermediateG[x][y], spdIntermediateB[x][y], spdIntermediateA[x][y]);
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
#if FFX_SPD_OPTION_DOWNSAMPLE_FILTER == 1
    return ffxMin(ffxMin(v0, v1), ffxMin(v2, v3));
#elif FFX_SPD_OPTION_DOWNSAMPLE_FILTER == 2
    return ffxMax(ffxMax(v0, v1), ffxMax(v2, v3));
#else
    return (v0 + v1 + v2 + v3) * 0.25;
#endif
}

#endif // FFX_HALF

#include "ffx_spd.h"

void DOWNSAMPLE(FfxUInt32 LocalThreadId, FfxUInt32x3 WorkGroupId)
{
#if FFX_HALF
    SpdDownsampleH(WorkGroupId.xy, LocalThreadId, Mips(), NumWorkGroups(), WorkGroupId.z, WorkGroupOffset());
#else
    SpdDownsample(WorkGroupId.xy, LocalThreadId, Mips(), NumWorkGroups(), WorkGroupId.z, WorkGroupOffset());
#endif // FFX_HALF
}
