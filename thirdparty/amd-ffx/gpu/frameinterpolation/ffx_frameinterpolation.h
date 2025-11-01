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

#ifndef FFX_FRAMEINTERPOLATION_H
#define FFX_FRAMEINTERPOLATION_H

struct InterpolationSourceColor
{
    FfxFloat32x3 fRaw;
    FfxFloat32x3 fLinear;
    FfxFloat32   fBilinearWeightSum;
};

InterpolationSourceColor NewInterpolationSourceColor()
{
    InterpolationSourceColor c;
    c.fRaw = FfxFloat32x3(0.0, 0.0, 0.0);
    c.fLinear = FfxFloat32x3(0.0, 0.0, 0.0);
    c.fBilinearWeightSum = 0.0;
    return c;
}

InterpolationSourceColor SampleTextureBilinear(FfxBoolean isCurrent, FfxFloat32x2 fUv, FfxFloat32x2 fMotionVector, FfxInt32x2 texSize)
{
    InterpolationSourceColor result = NewInterpolationSourceColor();

    FfxFloat32x2 fReprojectedUv = fUv + fMotionVector;
    BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fReprojectedUv, texSize);

    FfxFloat32x3 fColor = FfxFloat32x3(0.0, 0.0, 0.0);
    FfxFloat32 fWeightSum = 0.0f;
    for (FfxInt32 iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++) {

        const FfxInt32x2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
        const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;

        if (IsInRect(iSamplePos, InterpolationRectBase(), InterpolationRectSize()))
        {
            FfxFloat32 fWeight = bilinearInfo.fWeights[iSampleIndex];

            if (isCurrent)
                fColor += LoadCurrentBackbuffer(iSamplePos).rgb * fWeight;
            else
                fColor += LoadPreviousBackbuffer(iSamplePos).rgb * fWeight;
            fWeightSum += fWeight;
        }
    }

    //normalize colors
    fColor = (fWeightSum != 0.0f) ? fColor / fWeightSum : FfxFloat32x3(0.0f, 0.0f, 0.0f);

    result.fRaw               = fColor;
    result.fLinear            = RawRGBToLinear(fColor);
    result.fBilinearWeightSum = fWeightSum;

    return result;
}

void updateInPaintingWeight(inout FfxFloat32 fInPaintingWeight, FfxFloat32 fFactor)
{
    fInPaintingWeight = ffxSaturate(ffxMax(fInPaintingWeight, fFactor));
}

void computeInterpolatedColor(FfxUInt32x2 iPxPos, out FfxFloat32x3 fInterpolatedColor, inout FfxFloat32 fInPaintingWeight)
{
    const FfxFloat32x2 fUvInInterpolationRect = (FfxFloat32x2(iPxPos - InterpolationRectBase()) + 0.5f) / InterpolationRectSize();
    const FfxFloat32x2 fUvInScreenSpace       = (FfxFloat32x2(iPxPos) + 0.5f) / DisplaySize();
    const FfxFloat32x2 fLrUvInInterpolationRect = fUvInInterpolationRect * (FfxFloat32x2(RenderSize()) / GetMaxRenderSize());

    const FfxFloat32x2 fUvLetterBoxScale = FfxFloat32x2(InterpolationRectSize()) / DisplaySize();

    // game MV are top left aligned, the function scales them to render res UV
    VectorFieldEntry gameMv;
    LoadInpaintedGameFieldMv(fUvInInterpolationRect, gameMv);

    // OF is done on the back buffers which already have black bars
    VectorFieldEntry ofMv;
    SampleOpticalFlowMotionVectorField(fUvInScreenSpace, ofMv);

    // Binarize disucclusion factor
    FfxFloat32x2 fDisocclusionFactor = FfxFloat32x2(FFX_EQUAL(ffxSaturate(SampleDisocclusionMask(fLrUvInInterpolationRect).xy), FfxFloat32x2(1.0, 1.0)));

    InterpolationSourceColor fPrevColorGame = SampleTextureBilinear(false, fUvInScreenSpace, +gameMv.fMotionVector * fUvLetterBoxScale, DisplaySize()); // Get in previous frame buffer, the color of interpolated pixel
    InterpolationSourceColor fCurrColorGame = SampleTextureBilinear(true, fUvInScreenSpace, -gameMv.fMotionVector * fUvLetterBoxScale, DisplaySize()); // Get color in current framebuffer, of color of interpolated pixel

    InterpolationSourceColor fPrevColorOF = SampleTextureBilinear(false, fUvInScreenSpace, +ofMv.fMotionVector * fUvLetterBoxScale, DisplaySize());
    InterpolationSourceColor fCurrColorOF = SampleTextureBilinear(true, fUvInScreenSpace, -ofMv.fMotionVector * fUvLetterBoxScale, DisplaySize());

    FfxFloat32 fDisoccludedFactor = 0.0f;

    // Disocclusion logic
    {
        fDisocclusionFactor.x *= FfxFloat32(!gameMv.bPosOutside); // fDisocclusionFactor.x of 1 means the pos of interpolated pixel is within bounds of previous frame.
        fDisocclusionFactor.y *= FfxFloat32(!gameMv.bNegOutside); // fDisocclusionFactor.y of 1 means the pos of interpolated pixel is within bounds of current frame

        // Inpaint in bi-directional disocclusion areas
        updateInPaintingWeight(fInPaintingWeight, FfxFloat32(length(fDisocclusionFactor) <= FFX_FRAMEINTERPOLATION_EPSILON));

        FfxFloat32 t = 0.5f;
        t += 0.5f * (1 - (fDisocclusionFactor.x));
        t -= 0.5f * (1 - (fDisocclusionFactor.y));
        // Say if fDisocclusionFactor.x is 1 and fDisocclusionFactor.y = 0, then t will be 0. fInterpolatedColor will be entirely from fPrevColorGame 
        fInterpolatedColor = ffxLerp(fPrevColorGame.fRaw, fCurrColorGame.fRaw, ffxSaturate(t));
        fDisoccludedFactor = ffxSaturate(1 - ffxMin(fDisocclusionFactor.x, fDisocclusionFactor.y));

        if (fPrevColorGame.fBilinearWeightSum == 0.0f)
        {
            fInterpolatedColor = fCurrColorGame.fRaw;
        }
        else if (fCurrColorGame.fBilinearWeightSum == 0.0f)
        {
            fInterpolatedColor = fPrevColorGame.fRaw;
        }
        if (fPrevColorGame.fBilinearWeightSum == 0 && fCurrColorGame.fBilinearWeightSum == 0)
        {
            fInPaintingWeight = 1.0f;
        }
    }

    {

        FfxFloat32 ofT = 0.5f;

        if (fPrevColorOF.fBilinearWeightSum > 0 && fCurrColorOF.fBilinearWeightSum > 0)
        {
            ofT = 0.5f;
        }
        else if (fPrevColorOF.fBilinearWeightSum > 0)
        {
            ofT = 0;
        } else {
            ofT = 1;
        }

        const FfxFloat32x3 ofColor = ffxLerp(fPrevColorOF.fRaw, fCurrColorOF.fRaw, ofT);

        FfxFloat32 fOF_Sim = NormalizedDot3(fPrevColorOF.fRaw, fCurrColorOF.fRaw);
        FfxFloat32 fGame_Sim = NormalizedDot3(fPrevColorGame.fRaw, fCurrColorGame.fRaw);

        fGame_Sim = ffxLerp(ffxMax(FFX_FRAMEINTERPOLATION_EPSILON, fGame_Sim), 1.0f, ffxSaturate(fDisoccludedFactor));
        FfxFloat32 fGameMvBias = ffxPow(ffxSaturate(fGame_Sim / ffxMax(FFX_FRAMEINTERPOLATION_EPSILON, fOF_Sim)), 1.0f);

        const FfxFloat32 fFrameIndexFactor = FfxFloat32(FrameIndexSinceLastReset() < 10);
        fGameMvBias = ffxLerp(fGameMvBias, 1.0f, fFrameIndexFactor);

        fInterpolatedColor = ffxLerp(ofColor, fInterpolatedColor, ffxSaturate(fGameMvBias));
    }
}

void computeFrameinterpolation(FfxInt32x2 iPxPos)
{
    FfxFloat32x3 fColor            = FfxFloat32x3(0, 0, 0);
    FfxFloat32   fInPaintingWeight = 0.0f;

    if (IsInRect(iPxPos, InterpolationRectBase(), InterpolationRectSize()) == false || FrameIndexSinceLastReset() == 0)
    {
        // if we just reset or we are out of the interpolation rect, copy the current back buffer and don't interpolate
        fColor = LoadCurrentBackbuffer(iPxPos);
    }
    else
    {
        computeInterpolatedColor(iPxPos, fColor, fInPaintingWeight);
    }

    StoreFrameinterpolationOutput(FfxInt32x2(iPxPos), FfxFloat32x4(fColor, fInPaintingWeight));
}

#endif  // FFX_FRAMEINTERPOLATION_H
