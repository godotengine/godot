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

#ifndef FFX_FRAMEINTERPOLATION_INPAINTING_H
#define FFX_FRAMEINTERPOLATION_INPAINTING_H

FfxFloat32x4 ComputeInpaintingLevel(FfxFloat32x2 fUv, const FfxInt32 iMipLevel, const FfxInt32x2 iTexSize)
{
    BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fUv, iTexSize);

    FfxFloat32x4 fColor = FfxFloat32x4(0.0, 0.0, 0.0, 0.0);

    for (FfxInt32 iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++) {

        const FfxInt32x2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
        const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;

        if (IsOnScreen(iSamplePos, iTexSize)) {

            FfxFloat32x4 fSample = LoadInpaintingPyramid(iMipLevel, iSamplePos);

            const FfxFloat32 fWeight = bilinearInfo.fWeights[iSampleIndex] * FfxFloat32(fSample.w > 0.0f);

            fColor += FfxFloat32x4(fSample.rgb * fWeight, fWeight);
        }
    }

    return fColor;
}

FfxFloat32x3 ComputeInpainting(FfxInt32x2 iPxPos)
{
    FfxFloat32x2 fUv = (iPxPos + 0.5f) / (DisplaySize());

    FfxFloat32x4 fColor = FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
    FfxFloat32 fWeightSum = 0.0f;
    FfxInt32x2 iTexSize = DisplaySize();

    for (FfxInt32 iMipLevel = 0; iMipLevel < 10; iMipLevel++) {

        iTexSize /= 2;

        FfxFloat32x4 fMipColor = ComputeInpaintingLevel(fUv, iMipLevel, iTexSize);

        if (fMipColor.w > 0.0f) {
            const FfxFloat32x3 fNormalizedMipColor = fMipColor.rgb / fMipColor.w;
            const FfxFloat32 fMipWeight = ffxPow(1.0f - iMipLevel / 10.0f, 3.0f) * fMipColor.w;

            fColor += FfxFloat32x4(fNormalizedMipColor, 1.0f) * fMipWeight;
        }
    }

    return fColor.rgb / fColor.w;
}

void drawDebugTearLines(FfxInt32x2 iPxPos, inout FfxFloat32x3 fColor, inout FfxBoolean bWriteColor)
{
    if (iPxPos.x < 16)
    {
        fColor.g = 1.f;
        bWriteColor = true;
    }
    else if (iPxPos.x > DisplaySize().x - 16)
    {
        fColor += GetDebugBarColor();
        bWriteColor = true;
    }

}

void drawDebugResetIndicators(FfxInt32x2 iPxPos, inout FfxFloat32x3 fColor, inout FfxBoolean bWriteColor)
{
    if (iPxPos.y < 32 && Reset())
    {
        fColor.r    = 1.f;
        bWriteColor = true;
    }
    else if (iPxPos.y > 32 && iPxPos.y < 64 && HasSceneChanged())
    {
        fColor.b    = 1.f;
        bWriteColor = true;
    }
}

void computeInpainting(FfxInt32x2 iPxPos)
{
    FfxBoolean bWriteColor = false;
    FfxFloat32x4 fInterpolatedColor    = RWLoadFrameinterpolationOutput(iPxPos);

    const FfxFloat32 fInPaintingWeight = fInterpolatedColor.w;
    if (fInPaintingWeight > FFX_FRAMEINTERPOLATION_EPSILON)
    {
        fInterpolatedColor.rgb = ffxLerp(fInterpolatedColor.rgb, ComputeInpainting(iPxPos) * FfxFloat32(DisplaySize().x > 0), fInPaintingWeight);
        bWriteColor = true;
    }

    if (GetHUDLessAttachedFactor() == 1)
    {
        const FfxFloat32x3 fCurrentInterpolationSource = LoadCurrentBackbuffer(iPxPos).rgb;
        const FfxFloat32x3 fPresentColor               = LoadPresentBackbuffer(iPxPos).rgb;

        if (any(FFX_GREATER_THAN(abs(fCurrentInterpolationSource - fPresentColor), FfxFloat32x3(0.0, 0.0, 0.0))))
        {
            const FfxFloat32 fStaticFactor = CalculateStaticContentFactor(RawRGBToLinear(fCurrentInterpolationSource), RawRGBToLinear(fPresentColor));

            if (fStaticFactor > FFX_FRAMEINTERPOLATION_EPSILON)
            {
                fInterpolatedColor.rgb = ffxLerp(fInterpolatedColor.rgb, fPresentColor, fStaticFactor);
                bWriteColor = true;
            }
        }
    }

    if ((GetDispatchFlags() & FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_TEAR_LINES) != 0)
    {
        drawDebugTearLines(iPxPos, fInterpolatedColor.rgb, bWriteColor);
    }
    
    if ((GetDispatchFlags() & FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_RESET_INDICATORS) != 0)
    {
        drawDebugResetIndicators(iPxPos, fInterpolatedColor.rgb, bWriteColor);
    }

    if (bWriteColor)
    {
        StoreFrameinterpolationOutput(iPxPos, FfxFloat32x4(fInterpolatedColor.rgb, 1.0f));
    }
    
}

#endif  // FFX_FRAMEINTERPOLATION_INPAINTING_H
