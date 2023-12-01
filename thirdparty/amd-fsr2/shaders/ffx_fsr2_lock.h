// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
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

#ifndef FFX_FSR2_LOCK_H
#define FFX_FSR2_LOCK_H

void ClearResourcesForNextFrame(in FfxInt32x2 iPxHrPos)
{
    if (all(FFX_LESS_THAN(iPxHrPos, FfxInt32x2(RenderSize()))))
    {
#if FFX_FSR2_OPTION_INVERTED_DEPTH
        const FfxUInt32 farZ = 0x0;
#else
        const FfxUInt32 farZ = 0x3f800000;
#endif
        SetReconstructedDepth(iPxHrPos, farZ);
    }
}

FfxBoolean ComputeThinFeatureConfidence(FfxInt32x2 pos)
{
    const FfxInt32 RADIUS = 1;

    FfxFloat32 fNucleus = LoadLockInputLuma(pos);

    FfxFloat32 similar_threshold = 1.05f;
    FfxFloat32 dissimilarLumaMin = FSR2_FLT_MAX;
    FfxFloat32 dissimilarLumaMax = 0;

    /*
     0 1 2
     3 4 5
     6 7 8
    */

    #define SETBIT(x) (1U << x)

    FfxUInt32 mask = SETBIT(4); //flag fNucleus as similar

    const FfxUInt32 uNumRejectionMasks = 4;
    const FfxUInt32 uRejectionMasks[uNumRejectionMasks] = {
        SETBIT(0) | SETBIT(1) | SETBIT(3) | SETBIT(4), //Upper left
        SETBIT(1) | SETBIT(2) | SETBIT(4) | SETBIT(5), //Upper right
        SETBIT(3) | SETBIT(4) | SETBIT(6) | SETBIT(7), //Lower left
        SETBIT(4) | SETBIT(5) | SETBIT(7) | SETBIT(8), //Lower right
    };

    FfxInt32 idx = 0;
    FFX_UNROLL
    for (FfxInt32 y = -RADIUS; y <= RADIUS; y++) {
        FFX_UNROLL
        for (FfxInt32 x = -RADIUS; x <= RADIUS; x++, idx++) {
            if (x == 0 && y == 0) continue;

            FfxInt32x2 samplePos = ClampLoad(pos, FfxInt32x2(x, y), FfxInt32x2(RenderSize()));

            FfxFloat32 sampleLuma = LoadLockInputLuma(samplePos);
            FfxFloat32 difference = ffxMax(sampleLuma, fNucleus) / ffxMin(sampleLuma, fNucleus);

            if (difference > 0 && (difference < similar_threshold)) {
                mask |= SETBIT(idx);
            } else {
                dissimilarLumaMin = ffxMin(dissimilarLumaMin, sampleLuma);
                dissimilarLumaMax = ffxMax(dissimilarLumaMax, sampleLuma);
            }
        }
    }

    FfxBoolean isRidge = fNucleus > dissimilarLumaMax || fNucleus < dissimilarLumaMin;

    if (FFX_FALSE == isRidge) {

        return false;
    }

    FFX_UNROLL
    for (FfxInt32 i = 0; i < 4; i++) {

        if ((mask & uRejectionMasks[i]) == uRejectionMasks[i]) {
            return false;
        }
    }
    
    return true;
}

void ComputeLock(FfxInt32x2 iPxLrPos)
{
    if (ComputeThinFeatureConfidence(iPxLrPos))
    {
        StoreNewLocks(ComputeHrPosFromLrPos(iPxLrPos), 1.f);
    }

    ClearResourcesForNextFrame(iPxLrPos);
}

#endif // FFX_FSR2_LOCK_H
