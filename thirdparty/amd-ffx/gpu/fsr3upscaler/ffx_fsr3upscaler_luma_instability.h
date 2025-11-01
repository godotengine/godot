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

struct LumaInstabilityFactorData
{
    FfxFloat32x4 fLumaHistory;
    FfxFloat32 fLumaInstabilityFactor;
};

LumaInstabilityFactorData ComputeLumaInstabilityFactor(LumaInstabilityFactorData data, FfxFloat32 fCurrentFrameLuma, FfxFloat32 fFarthestDepthInMeters)
{
    const FfxInt32 N_MINUS_1 = 0;
    const FfxInt32 N_MINUS_2 = 1;
    const FfxInt32 N_MINUS_3 = 2;
    const FfxInt32 N_MINUS_4 = 3;

    FfxFloat32 fLumaInstability     = 0.0f;
    const FfxFloat32 fDiffs0        = (fCurrentFrameLuma - data.fLumaHistory[N_MINUS_1]);
    const FfxFloat32 fSimilarity0   = MinDividedByMax(fCurrentFrameLuma, data.fLumaHistory[N_MINUS_1], 1.0f);

    FfxFloat32 fMaxSimilarity = fSimilarity0;

    if (fSimilarity0 < 1.0f) {
        for (int i = N_MINUS_2; i <= N_MINUS_4; i++) {
            const FfxFloat32 fDiffs1 = (fCurrentFrameLuma - data.fLumaHistory[i]);
            const FfxFloat32 fSimilarity1 = MinDividedByMax(fCurrentFrameLuma, data.fLumaHistory[i]);

            if (sign(fDiffs0) == sign(fDiffs1)) {

                fMaxSimilarity = ffxMax(fMaxSimilarity, fSimilarity1);
            }
        }

        fLumaInstability = FfxFloat32(fMaxSimilarity > fSimilarity0);
    }

    // Shift history
    data.fLumaHistory[N_MINUS_4] = data.fLumaHistory[N_MINUS_3];
    data.fLumaHistory[N_MINUS_3] = data.fLumaHistory[N_MINUS_2];
    data.fLumaHistory[N_MINUS_2] = data.fLumaHistory[N_MINUS_1];
    data.fLumaHistory[N_MINUS_1] = fCurrentFrameLuma;

    data.fLumaHistory /= Exposure();

    data.fLumaInstabilityFactor = fLumaInstability * FfxFloat32(data.fLumaHistory[N_MINUS_4] != 0);

    return data;
}

void LumaInstability(FfxInt32x2 iPxPos)
{
    LumaInstabilityFactorData data;
    data.fLumaInstabilityFactor = 0.0f;
    data.fLumaHistory = FfxFloat32x4(0.0f, 0.0f, 0.0f, 0.0f);

    const FfxFloat32x2 fDilatedMotionVector = LoadDilatedMotionVector(iPxPos);
    const FfxFloat32x2 fUv = (iPxPos + 0.5f) / RenderSize();
    const FfxFloat32x2 fUvCurrFrameJittered = fUv + Jitter() / RenderSize();
    const FfxFloat32x2 fUvPrevFrameJittered = fUv + PreviousFrameJitter() / PreviousFrameRenderSize();
    const FfxFloat32x2 fReprojectedUv = fUvPrevFrameJittered + fDilatedMotionVector;

    if (IsUvInside(fReprojectedUv))
    {
        const FfxFloat32x2 fUvReactive_HW = ClampUv(fUvCurrFrameJittered, RenderSize(), MaxRenderSize());

        const FfxFloat32x4 fDilatedReactiveMasks = SampleDilatedReactiveMasks(fUvReactive_HW);
        const FfxFloat32 fReactiveMask = ffxSaturate(fDilatedReactiveMasks[REACTIVE]);
        const FfxFloat32 fDisocclusion = ffxSaturate(fDilatedReactiveMasks[DISOCCLUSION]);
        const FfxFloat32 fShadingChange = ffxSaturate(fDilatedReactiveMasks[SHADING_CHANGE]);
        const FfxFloat32 fAccumulation = ffxSaturate(fDilatedReactiveMasks[ACCUMULAION]);

        const FfxBoolean bAccumulationFactor = fAccumulation > 0.9f;

        const FfxBoolean bComputeInstability = bAccumulationFactor;

        if (bComputeInstability) {

            const FfxFloat32x2 fUv_HW = ClampUv(fUvCurrFrameJittered, RenderSize(), MaxRenderSize());
            const FfxFloat32 fCurrentFrameLuma = SampleCurrentLuma(fUv_HW) * Exposure();

            const FfxFloat32x2 fReprojectedUv_HW = ClampUv(fReprojectedUv, PreviousFrameRenderSize(), MaxRenderSize());
            data.fLumaHistory                    = SampleLumaHistory(fReprojectedUv_HW) * DeltaPreExposure() * Exposure();

            const FfxFloat32x2 fFarthestDepthUv_HW = ClampUv(fUvCurrFrameJittered, RenderSize() / 2, GetFarthestDepthMip1ResourceDimensions());
            const FfxFloat32 fFarthestDepthInMeters = SampleFarthestDepthMip1(fFarthestDepthUv_HW);

            data = ComputeLumaInstabilityFactor(data, fCurrentFrameLuma, fFarthestDepthInMeters);

            const FfxFloat32 fVelocityWeight = 1.0f - ffxSaturate(Get4KVelocity(fDilatedMotionVector) / 20.0f);
            data.fLumaInstabilityFactor *= fVelocityWeight * (1.0f - fDisocclusion) * (1.0f - fReactiveMask) * (1.0f - fShadingChange);
        }
    }

    StoreLumaHistory(iPxPos, data.fLumaHistory);
    StoreLumaInstability(iPxPos, data.fLumaInstabilityFactor);
}
