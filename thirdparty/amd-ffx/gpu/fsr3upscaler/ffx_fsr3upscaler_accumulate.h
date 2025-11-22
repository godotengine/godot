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

void Accumulate(const AccumulationPassCommonParams params, FFX_PARAMETER_INOUT AccumulationPassData data)
{
    // Avoid invalid values when accumulation and upsampled weight is 0
    data.fHistoryWeight *= FfxFloat32(data.fHistoryWeight > FSR3UPSCALER_FP16_MIN);
    data.fHistoryWeight = ffxMax(FSR3UPSCALER_EPSILON, data.fHistoryWeight + data.fUpsampledWeight);

#if FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT
    //YCoCg -> RGB -> Tonemap -> YCoCg (Use RGB tonemapper to avoid color desaturation)
    data.fUpsampledColor    = RGBToYCoCg(Tonemap(YCoCgToRGB(data.fUpsampledColor)));
    data.fHistoryColor      = RGBToYCoCg(Tonemap(YCoCgToRGB(data.fHistoryColor)));
#endif

    const FfxFloat32 fAlpha = ffxSaturate(data.fUpsampledWeight / data.fHistoryWeight);
    data.fHistoryColor      = ffxLerp(data.fHistoryColor, data.fUpsampledColor, fAlpha);
    data.fHistoryColor      = YCoCgToRGB(data.fHistoryColor);

#if FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT
    data.fHistoryColor      = InverseTonemap(data.fHistoryColor);
#endif
}

void RectifyHistory(
    const AccumulationPassCommonParams params,
    FFX_PARAMETER_INOUT AccumulationPassData data
)
{
    const FfxFloat32 f4kVelocityFactor = ffxSaturate(params.f4KVelocity / 20.0f);
    const FfxFloat32 fDistanceFactor        = ffxSaturate(0.75f - params.fFarthestDepthInMeters / 20.0f);
    const FfxFloat32 fAccumulationFactor    = 1.0f - params.fAccumulation;
    const FfxFloat32 fReactiveFactor        = ffxPow(params.fReactiveMask, 1.0f / 2.0f);
    const FfxFloat32 fShadingChangeFactor   = params.fShadingChange;
    const FfxFloat32 fBoxScaleT             = ffxMax(f4kVelocityFactor, ffxMax(fDistanceFactor, ffxMax(fAccumulationFactor, ffxMax(fReactiveFactor, fShadingChangeFactor))));
    
    const FfxFloat32   fBoxScale     = ffxLerp(3.0f, 1.0f, fBoxScaleT);
    const FfxFloat32x3 fScaledBoxVec = data.clippingBox.boxVec * FfxFloat32x3(1.7f, 1.0f, 1.0f) * fBoxScale;

    const FfxFloat32x3 fClampedScaledBoxVec     = ffxMax(fScaledBoxVec, FfxFloat32x3(1.193e-7f, 1.193e-7f, 1.193e-7f));
    const FfxFloat32x3 fTransformedHistoryColor = (data.fHistoryColor - data.clippingBox.boxCenter) / fClampedScaledBoxVec;

    if (length(fTransformedHistoryColor)>1.f) {
        const FfxFloat32x3 fClampedHistoryColor = normalize(fTransformedHistoryColor);
        const FfxFloat32x3 fFinalClampedHistoryColor = (fClampedHistoryColor * fScaledBoxVec) + data.clippingBox.boxCenter;

        // Scale history color using rectification info, also using accumulation mask to avoid potential invalid color protection
        const FfxFloat32 fHistoryContribution = ffxMax(params.fLumaInstabilityFactor, data.fLockContributionThisFrame) * params.fAccumulation * (1 - params.fDisocclusion);
        data.fHistoryColor = ffxLerp(fFinalClampedHistoryColor, data.fHistoryColor, ffxSaturate(fHistoryContribution));
    }
}

void UpdateLockStatus(AccumulationPassCommonParams params, FFX_PARAMETER_INOUT AccumulationPassData data)
{
    data.fLock *= FfxFloat32(params.bIsNewSample == false);

    const FfxFloat32 fLifetimeDecreaseFactor = ffxMax(ffxSaturate(params.fShadingChange), ffxMax(params.fReactiveMask, params.fDisocclusion));
    data.fLock = ffxMax(0.0f, data.fLock - fLifetimeDecreaseFactor * fLockMax);

    // Compute this frame lock contribution
    data.fLockContributionThisFrame = ffxSaturate(ffxSaturate(data.fLock - fLockThreshold) * (fLockMax - fLockThreshold));

    const FfxFloat32 fNewLockIntensity = LoadRwNewLocks(params.iPxHrPos) * (1.0f - ffxMax(params.fShadingChange * 0, params.fReactiveMask));
    data.fLock = ffxMax(0.0f, ffxMin(data.fLock + fNewLockIntensity, fLockMax));

    // Preparing for next frame
    const FfxFloat32 fLifetimeDecrease = (0.1f / JitterSequenceLength()) * (1.0f - fLifetimeDecreaseFactor);
    data.fLock = ffxMax(0.0f, data.fLock - fLifetimeDecrease);

    // we expect similar motion for next frame
    // kill lock if that location is outside screen, avoid locks to be clamped to screen borders
    const FfxFloat32x2 fEstimatedUvNextFrame = params.fHrUv - params.fMotionVector;
    data.fLock *= FfxFloat32(IsUvInside(fEstimatedUvNextFrame) == true);
}

void ComputeBaseAccumulationWeight(const AccumulationPassCommonParams params, FFX_PARAMETER_INOUT AccumulationPassData data)
{
    FfxFloat32 fBaseAccumulation = params.fAccumulation;

    fBaseAccumulation = ffxMin(fBaseAccumulation, ffxLerp(fBaseAccumulation, 0.15f, ffxSaturate(ffxMax(0.0f, (params.f4KVelocity * VelocityFactor()) / 0.5f))));

    data.fHistoryWeight = fBaseAccumulation;
}

void InitPassData(FfxInt32x2 iPxHrPos, FFX_PARAMETER_INOUT AccumulationPassCommonParams params, FFX_PARAMETER_INOUT AccumulationPassData data)
{
    // Init constant params
    params.iPxHrPos                     = iPxHrPos;
    const FfxFloat32x2 fHrUv            = (iPxHrPos + 0.5f) / UpscaleSize();
    params.fHrUv                        = fHrUv;
    params.fLrUvJittered                = fHrUv + Jitter() / RenderSize();
    params.fLrUv_HwSampler              = ClampUv(params.fLrUvJittered, RenderSize(), MaxRenderSize());

    params.fMotionVector                = GetMotionVector(iPxHrPos, fHrUv);
    params.f4KVelocity                  = Get4KVelocity(params.fMotionVector);

    ComputeReprojectedUVs(params);

    const FfxFloat32x2 fLumaInstabilityUv_HW  = ClampUv(fHrUv, RenderSize(), MaxRenderSize());
    params.fLumaInstabilityFactor       = SampleLumaInstability(fLumaInstabilityUv_HW);

    const FfxFloat32x2 fFarthestDepthUv = ClampUv(params.fLrUvJittered, RenderSize() / 2, GetFarthestDepthMip1ResourceDimensions());
    params.fFarthestDepthInMeters       = SampleFarthestDepthMip1(fFarthestDepthUv);
    params.bIsNewSample                 = (params.bIsExistingSample == false || 0 == FrameIndex());

    const FfxFloat32x4 fDilatedReactiveMasks = SampleDilatedReactiveMasks(params.fLrUv_HwSampler);
    params.fReactiveMask                = ffxSaturate(fDilatedReactiveMasks[REACTIVE]);
    params.fDisocclusion                = ffxSaturate(fDilatedReactiveMasks[DISOCCLUSION]);
    params.fShadingChange               = ffxSaturate(fDilatedReactiveMasks[SHADING_CHANGE]);
    params.fAccumulation                = ffxSaturate(fDilatedReactiveMasks[ACCUMULAION]);
    params.fAccumulation *= FfxFloat32(round(params.fAccumulation * 100.0f) > 1.0f);

    // Init variable data
    data.fUpsampledColor                = FfxFloat32x3(0.0f, 0.0f, 0.0f);
    data.fHistoryColor                  = FfxFloat32x3(0.0f, 0.0f, 0.0f);
    data.fHistoryWeight                 = 1.0f;
    data.fUpsampledWeight               = 0.0f;
    data.fLock                          = 0.0f;
    data.fLockContributionThisFrame     = 0.0f;
}

void Accumulate(FfxInt32x2 iPxHrPos)
{
    AccumulationPassCommonParams params;
    AccumulationPassData data;
    InitPassData(iPxHrPos, params, data);

    if (params.bIsExistingSample && !params.bIsNewSample) {
        ReprojectHistoryColor(params, data);
    }
    
    UpdateLockStatus(params, data);

    ComputeBaseAccumulationWeight(params, data);

    ComputeUpsampledColorAndWeight(params, data);

    RectifyHistory(params, data);

    Accumulate(params, data);

    data.fHistoryColor /= Exposure();

    data.fHistoryColor = ffxMax(data.fHistoryColor, FfxFloat32x3(0.0f, 0.0f, 0.0f));

    StoreInternalColorAndWeight(iPxHrPos, FfxFloat32x4(data.fHistoryColor, data.fLock));

    // Output final color when RCAS is disabled
#if FFX_FSR3UPSCALER_OPTION_APPLY_SHARPENING == 0
    StoreUpscaledOutput(iPxHrPos, data.fHistoryColor);
#endif
    StoreNewLocks(iPxHrPos, 0);
}
