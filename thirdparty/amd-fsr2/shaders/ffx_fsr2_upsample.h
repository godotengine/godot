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

#ifndef FFX_FSR2_UPSAMPLE_H
#define FFX_FSR2_UPSAMPLE_H

FFX_STATIC const FfxUInt32 iLanczos2SampleCount = 16;

void Deringing(RectificationBox clippingBox, FFX_PARAMETER_INOUT FfxFloat32x3 fColor)
{
    fColor = clamp(fColor, clippingBox.aabbMin, clippingBox.aabbMax);
}
#if FFX_HALF
void Deringing(RectificationBoxMin16 clippingBox, FFX_PARAMETER_INOUT FFX_MIN16_F3 fColor)
{
    fColor = clamp(fColor, clippingBox.aabbMin, clippingBox.aabbMax);
}
#endif

#ifndef FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE
#define FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE 2 // Approximate
#endif

FfxFloat32 GetUpsampleLanczosWeight(FfxFloat32x2 fSrcSampleOffset, FfxFloat32 fKernelWeight)
{
    FfxFloat32x2 fSrcSampleOffsetBiased = fSrcSampleOffset * fKernelWeight.xx;
#if FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 0 // LANCZOS_TYPE_REFERENCE
    FfxFloat32 fSampleWeight = Lanczos2(length(fSrcSampleOffsetBiased));
#elif FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 1 // LANCZOS_TYPE_LUT
    FfxFloat32 fSampleWeight = Lanczos2_UseLUT(length(fSrcSampleOffsetBiased));
#elif FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2 // LANCZOS_TYPE_APPROXIMATE
    FfxFloat32 fSampleWeight = Lanczos2ApproxSq(dot(fSrcSampleOffsetBiased, fSrcSampleOffsetBiased));
#else
#error "Invalid Lanczos type"
#endif
    return fSampleWeight;
}

#if FFX_HALF
FFX_MIN16_F GetUpsampleLanczosWeight(FFX_MIN16_F2 fSrcSampleOffset, FFX_MIN16_F fKernelWeight)
{
    FFX_MIN16_F2 fSrcSampleOffsetBiased = fSrcSampleOffset * fKernelWeight.xx;
#if FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 0 // LANCZOS_TYPE_REFERENCE
    FFX_MIN16_F fSampleWeight = Lanczos2(length(fSrcSampleOffsetBiased));
#elif FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 1 // LANCZOS_TYPE_LUT
    FFX_MIN16_F fSampleWeight = Lanczos2_UseLUT(length(fSrcSampleOffsetBiased));
#elif FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2 // LANCZOS_TYPE_APPROXIMATE
    FFX_MIN16_F fSampleWeight = Lanczos2ApproxSq(dot(fSrcSampleOffsetBiased, fSrcSampleOffsetBiased));

    // To Test: Save reciproqual sqrt compute
    // FfxFloat32 fSampleWeight = Lanczos2Sq_UseLUT(dot(fSrcSampleOffsetBiased, fSrcSampleOffsetBiased));
#else
#error "Invalid Lanczos type"
#endif
    return fSampleWeight;
}
#endif

FfxFloat32 ComputeMaxKernelWeight() {
    const FfxFloat32 fKernelSizeBias = 1.0f;

    FfxFloat32 fKernelWeight = FfxFloat32(1) + (FfxFloat32(1.0f) / FfxFloat32x2(DownscaleFactor()) - FfxFloat32(1)).x * FfxFloat32(fKernelSizeBias);

    return ffxMin(FfxFloat32(1.99f), fKernelWeight);
}

FfxFloat32x4 ComputeUpsampledColorAndWeight(const AccumulationPassCommonParams params,
    FFX_PARAMETER_INOUT RectificationBox clippingBox, FfxFloat32 fReactiveFactor)
{
    #if FFX_FSR2_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF && FFX_HALF
    #include "ffx_fsr2_force16_begin.h"
    #endif
    // We compute a sliced lanczos filter with 2 lobes (other slices are accumulated temporaly)
    FfxFloat32x2 fDstOutputPos = FfxFloat32x2(params.iPxHrPos) + FFX_BROADCAST_FLOAT32X2(0.5f);      // Destination resolution output pixel center position
    FfxFloat32x2 fSrcOutputPos = fDstOutputPos * DownscaleFactor();                   // Source resolution output pixel center position
    FfxInt32x2 iSrcInputPos = FfxInt32x2(floor(fSrcOutputPos));                     // TODO: what about weird upscale factors...

    #if FFX_FSR2_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF && FFX_HALF
    #include "ffx_fsr2_force16_end.h"
    #endif

    FfxFloat32x3 fSamples[iLanczos2SampleCount];

    FfxFloat32x2 fSrcUnjitteredPos = (FfxFloat32x2(iSrcInputPos) + FfxFloat32x2(0.5f, 0.5f)) - Jitter(); // This is the un-jittered position of the sample at offset 0,0

    FfxInt32x2 offsetTL;
    offsetTL.x = (fSrcUnjitteredPos.x > fSrcOutputPos.x) ? FfxInt32(-2) : FfxInt32(-1);
    offsetTL.y = (fSrcUnjitteredPos.y > fSrcOutputPos.y) ? FfxInt32(-2) : FfxInt32(-1);

    //Load samples
    // If fSrcUnjitteredPos.y > fSrcOutputPos.y, indicates offsetTL.y = -2, sample offset Y will be [-2, 1], clipbox will be rows [1, 3].
    // Flip row# for sampling offset in this case, so first 0~2 rows in the sampled array can always be used for computing the clipbox.
    // This reduces branch or cmove on sampled colors, but moving this overhead to sample position / weight calculation time which apply to less values.
    const FfxBoolean bFlipRow = fSrcUnjitteredPos.y > fSrcOutputPos.y;
    const FfxBoolean bFlipCol = fSrcUnjitteredPos.x > fSrcOutputPos.x;

    FfxFloat32x2 fOffsetTL = FfxFloat32x2(offsetTL);

    FFX_UNROLL
    for (FfxInt32 row = 0; row < 3; row++) {

        FFX_UNROLL
            for (FfxInt32 col = 0; col < 3; col++) {
                FfxInt32 iSampleIndex = col + (row << 2);

                FfxInt32x2 sampleColRow = FfxInt32x2(bFlipCol ? (3 - col) : col, bFlipRow ? (3 - row) : row);
                FfxInt32x2 iSrcSamplePos = FfxInt32x2(iSrcInputPos) + offsetTL + sampleColRow;

                const FfxInt32x2 sampleCoord = ClampLoad(iSrcSamplePos, FfxInt32x2(0, 0), FfxInt32x2(RenderSize()));

                fSamples[iSampleIndex] = LoadPreparedInputColor(FfxInt32x2(sampleCoord));
            }
    }

    FfxFloat32x4 fColorAndWeight = FfxFloat32x4(0.0f, 0.0f, 0.0f, 0.0f);

    FfxFloat32x2 fBaseSampleOffset = FfxFloat32x2(fSrcUnjitteredPos - fSrcOutputPos);

    // Identify how much of each upsampled color to be used for this frame
    const FfxFloat32 fKernelReactiveFactor = ffxMax(fReactiveFactor, FfxFloat32(params.bIsNewSample));
    const FfxFloat32 fKernelBiasMax = ComputeMaxKernelWeight() * (1.0f - fKernelReactiveFactor);

    const FfxFloat32 fKernelBiasMin = ffxMax(1.0f, ((1.0f + fKernelBiasMax) * 0.3f));
    const FfxFloat32 fKernelBiasFactor = ffxMax(0.0f, ffxMax(0.25f * params.fDepthClipFactor, fKernelReactiveFactor));
    const FfxFloat32 fKernelBias = ffxLerp(fKernelBiasMax, fKernelBiasMin, fKernelBiasFactor);

    const FfxFloat32 fRectificationCurveBias = ffxLerp(-2.0f, -3.0f, ffxSaturate(params.fHrVelocity / 50.0f));

    FFX_UNROLL
    for (FfxInt32 row = 0; row < 3; row++) {
        FFX_UNROLL
        for (FfxInt32 col = 0; col < 3; col++) {
            FfxInt32 iSampleIndex = col + (row << 2);

            const FfxInt32x2 sampleColRow = FfxInt32x2(bFlipCol ? (3 - col) : col, bFlipRow ? (3 - row) : row);
            const FfxFloat32x2 fOffset = fOffsetTL + FfxFloat32x2(sampleColRow);
            FfxFloat32x2 fSrcSampleOffset = fBaseSampleOffset + fOffset;

            FfxInt32x2 iSrcSamplePos = FfxInt32x2(iSrcInputPos) + FfxInt32x2(offsetTL) + sampleColRow;

            const FfxFloat32 fOnScreenFactor = FfxFloat32(IsOnScreen(FfxInt32x2(iSrcSamplePos), FfxInt32x2(RenderSize())));
            FfxFloat32 fSampleWeight = fOnScreenFactor * FfxFloat32(GetUpsampleLanczosWeight(fSrcSampleOffset, fKernelBias));

            fColorAndWeight += FfxFloat32x4(fSamples[iSampleIndex] * fSampleWeight, fSampleWeight);

            // Update rectification box
            {
                const FfxFloat32 fSrcSampleOffsetSq = dot(fSrcSampleOffset, fSrcSampleOffset);
                const FfxFloat32 fBoxSampleWeight = exp(fRectificationCurveBias * fSrcSampleOffsetSq);

                const FfxBoolean bInitialSample = (row == 0) && (col == 0);
                RectificationBoxAddSample(bInitialSample, clippingBox, fSamples[iSampleIndex], fBoxSampleWeight);
            }
        }
    }

    RectificationBoxComputeVarianceBoxData(clippingBox);

    fColorAndWeight.w *= FfxFloat32(fColorAndWeight.w > FSR2_EPSILON);

    if (fColorAndWeight.w > FSR2_EPSILON) {
        // Normalize for deringing (we need to compare colors)
        fColorAndWeight.xyz = fColorAndWeight.xyz / fColorAndWeight.w;
        fColorAndWeight.w *= fUpsampleLanczosWeightScale;

        Deringing(clippingBox, fColorAndWeight.xyz);
    }

    #if FFX_FSR2_OPTION_UPSAMPLE_SAMPLERS_USE_DATA_HALF && FFX_HALF
    #include "ffx_fsr2_force16_end.h"
    #endif

    return fColorAndWeight;
}

#endif //!defined( FFX_FSR2_UPSAMPLE_H )
