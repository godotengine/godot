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

#ifndef FFX_FRAMEINTERPOLATION_DISOCCLUSION_MASK_H
#define FFX_FRAMEINTERPOLATION_DISOCCLUSION_MASK_H

FFX_STATIC const FfxFloat32 DepthClipBaseScale = 1.0f;

FfxFloat32 ComputeSampleDepthClip(FfxInt32x2 iPxSamplePos, FfxFloat32 fPreviousDepth, FfxFloat32 fPreviousDepthBilinearWeight, FfxFloat32 fCurrentDepthViewSpace)
{
    FfxFloat32 fPrevNearestDepthViewSpace = ConvertFromDeviceDepthToViewSpace(fPreviousDepth);

    // Depth separation logic ref: See "Minimum Triangle Separation for Correct Z-Buffer Occlusion"
    // Intention: worst case of formula in Figure4 combined with Ksep factor in Section 4
    const FfxFloat32 fHalfViewportWidth = RenderSize().x * 0.5f;
    FfxFloat32       fDepthThreshold    = ffxMax(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);

    // WARNING: Ksep only works with reversed-z with infinite projection.
    const FfxFloat32 Ksep                     = 1.37e-05f;
    FfxFloat32       fRequiredDepthSeparation = Ksep * fDepthThreshold * TanHalfFoV() * fHalfViewportWidth;
    FfxFloat32       fDepthDiff               = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;

    FfxFloat32 fDepthClipFactor = (fDepthDiff > 0) ? ffxSaturate(fRequiredDepthSeparation / fDepthDiff) : 1.0f;

    return fPreviousDepthBilinearWeight * fDepthClipFactor * ffxLerp(1.0f, DepthClipBaseScale, ffxSaturate(fDepthDiff * fDepthDiff));
}

FfxFloat32 LoadEstimatedDepth(FfxUInt32 estimatedIndex, FfxInt32x2 iSamplePos)
{
    const FfxFloat32x2 fUv = FfxFloat32x2(iSamplePos + 0.5f) / RenderSize();
    const FfxFloat32x2 fDistortionFieldUv = SampleDistortionField(fUv);
    FfxInt32x2 iDistortionPixelOffset = FfxInt32x2(fDistortionFieldUv.xy * RenderSize());

    if (estimatedIndex == 0)
    {
        return LoadReconstructedDepthPreviousFrame(iSamplePos + iDistortionPixelOffset);
    }
    else if (estimatedIndex == 1)
    {
        return LoadDilatedDepth(iSamplePos + iDistortionPixelOffset);
    }

    return 0;
}

FfxFloat32 ComputeDepthClip(FfxUInt32 estimatedIndex, FfxFloat32x2 fUvSample, FfxFloat32 fCurrentDepthSample)
{
    FfxFloat32 fCurrentDepthViewSpace = ConvertFromDeviceDepthToViewSpace(fCurrentDepthSample);
    BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fUvSample, RenderSize());

    FfxFloat32 fDilatedSum = 0.0f;
    FfxFloat32 fDepth = 0.0f;
    FfxFloat32 fWeightSum = 0.0f;
    for (FfxInt32 iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++) {

        const FfxInt32x2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
        const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;

        if (IsOnScreen(iSamplePos, RenderSize())) {
            const FfxFloat32 fWeight = bilinearInfo.fWeights[iSampleIndex];
            if (fWeight > fReconstructedDepthBilinearWeightThreshold) {

                const FfxFloat32 fPrevDepthSample = LoadEstimatedDepth(estimatedIndex, iSamplePos);
                const FfxFloat32 fPrevNearestDepthViewSpace = ConvertFromDeviceDepthToViewSpace(fPrevDepthSample);

                const FfxFloat32 fDepthDiff = fCurrentDepthViewSpace - fPrevNearestDepthViewSpace;

                if (fDepthDiff > 0.0f) {

#if FFX_FRAMEINTERPOLATION_OPTION_INVERTED_DEPTH
                    const FfxFloat32 fPlaneDepth = ffxMin(fPrevDepthSample, fCurrentDepthSample);
#else
                    const FfxFloat32 fPlaneDepth = ffxMax(fPrevDepthSample, fCurrentDepthSample);
#endif

                    const FfxFloat32x3 fCenter = GetViewSpacePosition(FfxInt32x2(RenderSize() * 0.5f), RenderSize(), fPlaneDepth);
                    const FfxFloat32x3 fCorner = GetViewSpacePosition(FfxInt32x2(0, 0), RenderSize(), fPlaneDepth);

                    const FfxFloat32 fHalfViewportWidth = length(FfxFloat32x2(RenderSize()));
                    const FfxFloat32 fDepthThreshold = ffxMin(fCurrentDepthViewSpace, fPrevNearestDepthViewSpace);

                    const FfxFloat32 Ksep = 1.37e-05f;
                    const FfxFloat32 Kfov = length(fCorner) / length(fCenter);
                    const FfxFloat32 fRequiredDepthSeparation = Ksep * Kfov * fHalfViewportWidth * fDepthThreshold;

                    const FfxFloat32 fResolutionFactor = ffxSaturate(length(FfxFloat32x2(RenderSize())) / length(FfxFloat32x2(1920.0f, 1080.0f)));
                    const FfxFloat32 fPower = ffxLerp(1.0f, 3.0f, fResolutionFactor);

                    fDepth += FfxFloat32((fRequiredDepthSeparation / fDepthDiff) >= 1.0f) * fWeight;
                    fWeightSum += fWeight;
                }
            }
        }
    }

    return (fWeightSum > 0.0f) ? ffxSaturate(1.0f - fDepth / fWeightSum) : 0.0f;
}

void computeDisocclusionMask(FfxInt32x2 iPxPos)
{
    FfxFloat32 fDilatedDepth = LoadEstimatedInterpolationFrameDepth(iPxPos);

    FfxFloat32x2 fDepthUv = (iPxPos + 0.5f) / RenderSize();
    FfxFloat32 fCurrentDepthViewSpace = ConvertFromDeviceDepthToViewSpace(fDilatedDepth);

    VectorFieldEntry gameMv;
    LoadInpaintedGameFieldMv(fDepthUv, gameMv);

    const FfxFloat32 fDepthClipInterpolatedToPrevious   = 1.0f - ComputeDepthClip(0, fDepthUv + gameMv.fMotionVector, fDilatedDepth);
    const FfxFloat32 fDepthClipInterpolatedToCurrent    = 1.0f - ComputeDepthClip(1, fDepthUv - gameMv.fMotionVector, fDilatedDepth);
    FfxFloat32x2 fDisocclusionMask = FfxFloat32x2(fDepthClipInterpolatedToPrevious, fDepthClipInterpolatedToCurrent);

    fDisocclusionMask = FfxFloat32x2(FFX_GREATER_THAN_EQUAL(fDisocclusionMask, ffxBroadcast2(FFX_FRAMEINTERPOLATION_EPSILON)));

    // Avoid false disocclusion if primary game vector pointer outside screen area
    const FfxFloat32x2 fSrcMotionVector   = gameMv.fMotionVector * 2.0f;
    const FfxInt32x2   iSamplePosPrevious = FfxInt32x2((fDepthUv + fSrcMotionVector) * RenderSize());
    fDisocclusionMask.x = ffxSaturate(fDisocclusionMask.x + FfxFloat32(!IsOnScreen(iSamplePosPrevious, RenderSize())));

    const FfxInt32x2 iSamplePosCurrent = FfxInt32x2((fDepthUv - fSrcMotionVector) * RenderSize());
    fDisocclusionMask.y = ffxSaturate(fDisocclusionMask.y + FfxFloat32(!IsOnScreen(iSamplePosCurrent, RenderSize())));

    StoreDisocclusionMask(iPxPos, fDisocclusionMask);

}

#endif // FFX_FRAMEINTERPOLATION_DISOCCLUSION_MASK_H
