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


#if FFX_HALF && (FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2) && defined(__XBOX_SCARLETT) && defined(__XBATG_EXTRA_16_BIT_OPTIMISATION) && (__XBATG_EXTRA_16_BIT_OPTIMISATION == 1)
#define FFX_FSR2_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS 1
#else
#define FFX_FSR2_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS 0
#endif

#if FFX_FSR2_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS

FFX_MIN16_F2 Bool2ToFloat16x2(bool x, bool y)
{
    uint lo = x ? 0x00003c00 : 0x00000000;
    uint hi = y ? 0x3c000000 : 0x00000000;

    return FFX_MIN16_F2(__XB_AsHalf(lo).x, __XB_AsHalf(hi).y);
}

struct PairedRectificationBoxAndAccumulatedColorAndWeight
{
    FFX_MIN16_F2 aabbMinRG;
    FFX_MIN16_F2 aabbMinB;

    FFX_MIN16_F2 aabbMaxRG;
    FFX_MIN16_F2 aabbMaxB;

    FFX_MIN16_F2 boxCenterRG;
    FFX_MIN16_F2 boxCenterB;

    FFX_MIN16_F2 boxVecRG;
    FFX_MIN16_F2 boxVecB;

    FFX_MIN16_F2 fBoxCenterWeight;

    FFX_MIN16_F2 fColorRG;
    FFX_MIN16_F2 fColorB;
    FFX_MIN16_F2 fWeight;

    FFX_MIN16_F fKernelBiasSq;
    FfxFloat32 fRectificationCurveBias;

    void setKernelBiasAndRectificationCurveBias(FfxFloat32 kernelBias, FfxFloat32 rectificationCurveBias)
    {
        fKernelBiasSq = FFX_MIN16_F(kernelBias * kernelBias);
        fRectificationCurveBias = rectificationCurveBias;
    }

    void init(FFX_MIN16_F fSrcSampleOffsetSq, bool sampleOnScreenX, bool sampleOnScreenY, FFX_MIN16_F3 colorSample)
    {
        // NOTE: make sure exp has 32-bit precision
        const FFX_MIN16_F fBoxSampleWeight = FFX_MIN16_F(
            exp(fRectificationCurveBias * FfxFloat32(fSrcSampleOffsetSq))
        );

#if FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2 // LANCZOS_TYPE_APPROXIMATE
        const FFX_MIN16_F2 LanczosUpsampleWeight = PairedLanczos2ApproxSq(fSrcSampleOffsetSq * fKernelBiasSq);
#else
#error "Only LANCZOS_TYPE_APPROXIMATE is supported in paired version so far"
#endif
        const FFX_MIN16_F2 fSampleWeight = FFX_MIN16_F2((sampleOnScreenX && sampleOnScreenY ? 1.0 : 0.0), 0.0) * LanczosUpsampleWeight;

        aabbMinRG = colorSample.rg;
        aabbMinB = colorSample.bb;

        aabbMaxRG = colorSample.rg;
        aabbMaxB = colorSample.bb;

        boxCenterRG = colorSample.rg * fBoxSampleWeight.x;
        boxCenterB = colorSample.bb * fBoxSampleWeight;

        boxVecRG = colorSample.rg * boxCenterRG;
        boxVecB = colorSample.bb * boxCenterB;

        fBoxCenterWeight = fBoxSampleWeight;

        fColorRG = colorSample.rg * fSampleWeight.x;
        fColorB = colorSample.bb * fSampleWeight;
        fWeight = fSampleWeight;
    }

    void addSample(FFX_MIN16_F2 fSrcSampleOffsetSq, bool sample0OnScreen, bool sample1OnScreen, bool sample01OnScreen, FFX_MIN16_F3 ColorSample0, FFX_MIN16_F3 ColorSample1)
    {
        // NOTE: make sure exp has 32-bit precision
        const FFX_MIN16_F2 fBoxSampleWeight = FFX_MIN16_F2(
            exp(fRectificationCurveBias * FfxFloat32(fSrcSampleOffsetSq.x)),
            exp(fRectificationCurveBias * FfxFloat32(fSrcSampleOffsetSq.y))
        );

#if FFX_FSR2_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2 // LANCZOS_TYPE_APPROXIMATE
        const FFX_MIN16_F2 LanczosUpsampleWeight = PairedLanczos2ApproxSq(fSrcSampleOffsetSq * fKernelBiasSq);
#else
#error "Only LANCZOS_TYPE_APPROXIMATE is supported in paired version so far"
#endif
        const FFX_MIN16_F2 fSampleWeight = Bool2ToFloat16x2(sample0OnScreen && sample01OnScreen, sample1OnScreen && sample01OnScreen) * LanczosUpsampleWeight;

        FFX_MIN16_F2 colorSampleB = FFX_MIN16_F2(ColorSample0.b, ColorSample1.b);

        aabbMinRG = ffxMin(aabbMinRG, ColorSample0.rg);
        aabbMinRG = ffxMin(aabbMinRG, ColorSample1.rg);
        aabbMinB = ffxMin(aabbMinB, colorSampleB);

        aabbMaxRG = ffxMax(aabbMaxRG, ColorSample0.rg);
        aabbMaxRG = ffxMax(aabbMaxRG, ColorSample1.rg);
        aabbMaxB = ffxMax(aabbMaxB, colorSampleB);

        FFX_MIN16_F2 weightedColorSampleRG0 = ColorSample0.rg * fBoxSampleWeight.x;
        FFX_MIN16_F2 weightedColorSampleRG1 = ColorSample1.rg * fBoxSampleWeight.y;
        FFX_MIN16_F2 weightedColorSampleB = colorSampleB * fBoxSampleWeight;

        boxCenterRG += weightedColorSampleRG0;
        boxCenterRG += weightedColorSampleRG1;
        boxCenterB += weightedColorSampleB;

        boxVecRG += ColorSample0.rg * weightedColorSampleRG0;
        boxVecRG += ColorSample1.rg * weightedColorSampleRG1;
        boxVecB += colorSampleB * weightedColorSampleB;

        fBoxCenterWeight += fBoxSampleWeight;

        fWeight += fSampleWeight;
        fColorRG += (ColorSample0.rg * fSampleWeight.x) + (ColorSample1.rg * fSampleWeight.y);
        fColorB += colorSampleB * fSampleWeight;
    }

    void finalize(FFX_PARAMETER_INOUT RectificationBox rectificationBox, FFX_PARAMETER_INOUT FfxFloat32x4 outColorAndWeight)
    {
        rectificationBox.aabbMin.r = FfxFloat32(aabbMinRG.x);
        rectificationBox.aabbMin.g = FfxFloat32(aabbMinRG.y);
        rectificationBox.aabbMin.b = FfxFloat32(ffxMin(aabbMinB.x, aabbMinB.y));

        rectificationBox.aabbMax.r = FfxFloat32(aabbMaxRG.x);
        rectificationBox.aabbMax.g = FfxFloat32(aabbMaxRG.y);
        rectificationBox.aabbMax.b = FfxFloat32(ffxMax(aabbMaxB.x, aabbMaxB.y));

        rectificationBox.boxCenter.r = FfxFloat32(boxCenterRG.x);
        rectificationBox.boxCenter.g = FfxFloat32(boxCenterRG.y);
        rectificationBox.boxCenter.b = FfxFloat32(boxCenterB.x + boxCenterB.y);

        rectificationBox.boxVec.r = FfxFloat32(boxVecRG.x);
        rectificationBox.boxVec.g = FfxFloat32(boxVecRG.y);
        rectificationBox.boxVec.b = FfxFloat32(boxVecB.x + boxVecB.y);

        rectificationBox.fBoxCenterWeight = FfxFloat32(fBoxCenterWeight.x + fBoxCenterWeight.y);

        outColorAndWeight = FfxFloat32x4(fColorRG, fColorB.x + fColorB.y, fWeight.x + fWeight.y);
    }
};
#endif

FfxFloat32x4 ComputeUpsampledColorAndWeight(const AccumulationPassCommonParams params,
    FFX_PARAMETER_INOUT RectificationBox clippingBox, FfxFloat32 fReactiveFactor)
{
    // We compute a sliced lanczos filter with 2 lobes (other slices are accumulated temporaly)
    FfxFloat32x2 fDstOutputPos = FfxFloat32x2(params.iPxHrPos) + FFX_BROADCAST_FLOAT32X2(0.5f);      // Destination resolution output pixel center position
    FfxFloat32x2 fSrcOutputPos = fDstOutputPos * DownscaleFactor();                   // Source resolution output pixel center position
    FfxInt32x2 iSrcInputPos = FfxInt32x2(floor(fSrcOutputPos));                     // TODO: what about weird upscale factors...

#if FFX_FSR2_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS
    FFX_MIN16_F3 fSamples[iLanczos2SampleCount];
#else
    FfxFloat32x3 fSamples[iLanczos2SampleCount];
#endif

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

#if FFX_FSR2_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS
    // Unroll the loop to load samples on Scarlett to help the shader compiler
    const FFX_MIN16_F2 fSampleOffsetX02 = __XB_AsHalf(bFlipCol ? __XB_AsUInt(FFX_MIN16_F2( 1, -1)) : __XB_AsUInt(FFX_MIN16_F2(-1, 1)));
    const FFX_MIN16_F2 fSampleOffsetY02 = __XB_AsHalf(bFlipRow ? __XB_AsUInt(FFX_MIN16_F2( 1, -1)) : __XB_AsUInt(FFX_MIN16_F2(-1, 1)));

    typedef FfxInt32 FfxTexCoordI;
    typedef FfxInt32x2 FfxTexCoordI2;

    const FfxTexCoordI2 iSrcSamplePosX01 = FfxTexCoordI2(iSrcInputPos.xx) + (bFlipCol ? FfxTexCoordI2( 1,  0) : FfxTexCoordI2(-1, 0));
    const FfxTexCoordI2 iSrcSamplePosX23 = FfxTexCoordI2(iSrcInputPos.xx) + (bFlipCol ? FfxTexCoordI2(-1, -2) : FfxTexCoordI2( 1, 2));

    const FfxTexCoordI2 iSrcSamplePosY01 = FfxTexCoordI2(iSrcInputPos.yy) + (bFlipRow ? FfxTexCoordI2( 1,  0) : FfxTexCoordI2(-1, 0));
    const FfxTexCoordI2 iSrcSamplePosY23 = FfxTexCoordI2(iSrcInputPos.yy) + (bFlipRow ? FfxTexCoordI2(-1, -2) : FfxTexCoordI2( 1, 2));

    const FfxTexCoordI2 renderSizeLastTexelCoord = FfxTexCoordI2(RenderSize()) - FfxTexCoordI2(1, 1);

    const FfxTexCoordI2 iSrcSamplePosX01Clamped = FfxTexCoordI2(
        __XB_Med3_I32(iSrcSamplePosX01.x, 0, renderSizeLastTexelCoord.x),
        __XB_Med3_I32(iSrcSamplePosX01.y, 0, renderSizeLastTexelCoord.x)
    );

    const FfxTexCoordI2 iSrcSamplePosX23Clamped = FfxTexCoordI2(
        __XB_Med3_I32(iSrcSamplePosX23.x, 0, renderSizeLastTexelCoord.x),
        __XB_Med3_I32(iSrcSamplePosX23.y, 0, renderSizeLastTexelCoord.x)
    );

    const FfxTexCoordI2 iSrcSamplePosY01Clamped = FfxTexCoordI2(
        __XB_Med3_I32(iSrcSamplePosY01.x, 0, renderSizeLastTexelCoord.y),
        __XB_Med3_I32(iSrcSamplePosY01.y, 0, renderSizeLastTexelCoord.y)
    );

    const FfxTexCoordI2 iSrcSamplePosY23Clamped = FfxTexCoordI2(
        __XB_Med3_I32(iSrcSamplePosY23.x, 0, renderSizeLastTexelCoord.y),
        __XB_Med3_I32(iSrcSamplePosY23.y, 0, renderSizeLastTexelCoord.y)
    );

    fSamples[ 0] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX01Clamped.x, iSrcSamplePosY01Clamped.x));
    fSamples[ 1] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX01Clamped.y, iSrcSamplePosY01Clamped.x));
    fSamples[ 2] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX23Clamped.x, iSrcSamplePosY01Clamped.x));

    fSamples[4 + 0] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX01Clamped.x, iSrcSamplePosY01Clamped.y));
    fSamples[4 + 1] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX01Clamped.y, iSrcSamplePosY01Clamped.y));
    fSamples[4 + 2] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX23Clamped.x, iSrcSamplePosY01Clamped.y));

    fSamples[8 + 0] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX01Clamped.x, iSrcSamplePosY23Clamped.x));
    fSamples[8 + 1] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX01Clamped.y, iSrcSamplePosY23Clamped.x));
    fSamples[8 + 2] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX23Clamped.x, iSrcSamplePosY23Clamped.x));

    fSamples[12 + 0] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX01Clamped.x, iSrcSamplePosY23Clamped.y));
    fSamples[12 + 1] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX01Clamped.y, iSrcSamplePosY23Clamped.y));
    fSamples[12 + 2] = LoadPreparedInputColorHalf(FfxTexCoordI2(iSrcSamplePosX23Clamped.x, iSrcSamplePosY23Clamped.y));

#else
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
#endif

    FfxFloat32x4 fColorAndWeight = FfxFloat32x4(0.0f, 0.0f, 0.0f, 0.0f);

    FfxFloat32x2 fBaseSampleOffset = FfxFloat32x2(fSrcUnjitteredPos - fSrcOutputPos);

    // Identify how much of each upsampled color to be used for this frame
    const FfxFloat32 fKernelReactiveFactor = ffxMax(fReactiveFactor, FfxFloat32(params.bIsNewSample));
    const FfxFloat32 fKernelBiasMax = ComputeMaxKernelWeight() * (1.0f - fKernelReactiveFactor);

    const FfxFloat32 fKernelBiasMin = ffxMax(1.0f, ((1.0f + fKernelBiasMax) * 0.3f));
    const FfxFloat32 fKernelBiasFactor = ffxMax(0.0f, ffxMax(0.25f * params.fDepthClipFactor, fKernelReactiveFactor));
    const FfxFloat32 fKernelBias = ffxLerp(fKernelBiasMax, fKernelBiasMin, fKernelBiasFactor);

    const FfxFloat32 fRectificationCurveBias = ffxLerp(-2.0f, -3.0f, ffxSaturate(params.fHrVelocity / 50.0f));

#if FFX_FSR2_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS
    // Unroll the loop to load samples on Scarlett to help the shader compiler
    const bool coordX0OnScreen = iSrcSamplePosX01.x == iSrcSamplePosX01Clamped.x;
    const bool coordX1OnScreen = iSrcSamplePosX01.y == iSrcSamplePosX01Clamped.y;
    const bool coordX2OnScreen = iSrcSamplePosX23.x == iSrcSamplePosX23Clamped.x;

    const bool coordY0OnScreen = iSrcSamplePosY01.x == iSrcSamplePosY01Clamped.x;
    const bool coordY1OnScreen = iSrcSamplePosY01.y == iSrcSamplePosY01Clamped.y;
    const bool coordY2OnScreen = iSrcSamplePosY23.x == iSrcSamplePosY23Clamped.x;

    const FFX_MIN16_F2 fBaseSampleOffsetHalf = FFX_MIN16_F2(fBaseSampleOffset);

    const FFX_MIN16_F2 fSrcSampleOffsetX_02 = fBaseSampleOffsetHalf.xx + fSampleOffsetX02;
    const FFX_MIN16_F2 fSrcSampleOffsetY_02 = fBaseSampleOffsetHalf.yy + fSampleOffsetY02;

    const FFX_MIN16_F2 fSrcSampleOffsetXSq_02 = fSrcSampleOffsetX_02 * fSrcSampleOffsetX_02;
    const FFX_MIN16_F2 fSrcSampleOffsetYSq_02 = fSrcSampleOffsetY_02 * fSrcSampleOffsetY_02;
    const FFX_MIN16_F2 fSrcSampleOffsetXYSq_11 = fBaseSampleOffsetHalf * fBaseSampleOffsetHalf;

    PairedRectificationBoxAndAccumulatedColorAndWeight pairedBox;
    pairedBox.setKernelBiasAndRectificationCurveBias(fKernelBias, fRectificationCurveBias);

    // init by o o o
    //         o x o
    //         o o o
    pairedBox.init(
        fSrcSampleOffsetXYSq_11.x + fSrcSampleOffsetXYSq_11.y,
        coordX1OnScreen, coordY1OnScreen,
        fSamples[5]
    );

    // add remaining two samples from 1st row x o x
    //                                        o * o
    //                                        o o o
    pairedBox.addSample(
        fSrcSampleOffsetXSq_02 + fSrcSampleOffsetYSq_02.xx,
        coordX0OnScreen, coordX2OnScreen, coordY0OnScreen,
        fSamples[0 + 0], fSamples[0 + 2]
    );

    // add two samples from 2nd row * o *
    //                              o * o
    //                              x o x
    pairedBox.addSample(
        fSrcSampleOffsetXSq_02 + fSrcSampleOffsetYSq_02.yy,
        coordX0OnScreen, coordX2OnScreen, coordY2OnScreen,
        fSamples[8 + 0], fSamples[8 + 2]
    );

    // add two samples from 3rd row * o *
    //                              x * x
    //                              * o *
    pairedBox.addSample(
        fSrcSampleOffsetXSq_02 + fSrcSampleOffsetXYSq_11.yy,
        coordX0OnScreen, coordX2OnScreen, coordY1OnScreen,
        fSamples[4 + 0], fSamples[4 + 2]
    );

    // add remaining samples * x *
    //                       * * *
    //                       * x *
    pairedBox.addSample(
        fSrcSampleOffsetXYSq_11.xx + fSrcSampleOffsetYSq_02,
        coordY0OnScreen, coordY2OnScreen, coordX1OnScreen,
        fSamples[0 + 1], fSamples[8 + 1]
    );

    pairedBox.finalize(clippingBox, fColorAndWeight);
#else
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
#endif

    RectificationBoxComputeVarianceBoxData(clippingBox);

    fColorAndWeight.w *= FfxFloat32(fColorAndWeight.w > FSR2_EPSILON);

    if (fColorAndWeight.w > FSR2_EPSILON) {
        // Normalize for deringing (we need to compare colors)
        fColorAndWeight.xyz = fColorAndWeight.xyz / fColorAndWeight.w;
        fColorAndWeight.w *= fUpsampleLanczosWeightScale;

        Deringing(clippingBox, fColorAndWeight.xyz);
    }

    return fColorAndWeight;
}

#endif //!defined( FFX_FSR2_UPSAMPLE_H )
