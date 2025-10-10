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

void Deringing(RectificationBox clippingBox, FFX_PARAMETER_INOUT FfxFloat32x3 fColor)
{
    fColor = clamp(fColor, clippingBox.aabbMin, clippingBox.aabbMax);
}

#ifndef FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE
#define FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE 2 // Approximate
#endif

FfxFloat32 GetUpsampleLanczosWeight(FfxFloat32x2 fSrcSampleOffset, FfxFloat32 fKernelWeight)
{
    FfxFloat32x2 fSrcSampleOffsetBiased = fSrcSampleOffset * fKernelWeight.xx;
#if FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 0 // LANCZOS_TYPE_REFERENCE
    FfxFloat32 fSampleWeight = Lanczos2(length(fSrcSampleOffsetBiased));
#elif FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 1 // LANCZOS_TYPE_LUT
    FfxFloat32 fSampleWeight = Lanczos2_UseLUT(length(fSrcSampleOffsetBiased));
#elif FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2 // LANCZOS_TYPE_APPROXIMATE
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
#if FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 0 // LANCZOS_TYPE_REFERENCE
    FFX_MIN16_F fSampleWeight = Lanczos2(length(fSrcSampleOffsetBiased));
#elif FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 1 // LANCZOS_TYPE_LUT
    FFX_MIN16_F fSampleWeight = Lanczos2_UseLUT(length(fSrcSampleOffsetBiased));
#elif FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2 // LANCZOS_TYPE_APPROXIMATE
    FFX_MIN16_F fSampleWeight = Lanczos2ApproxSq(dot(fSrcSampleOffsetBiased, fSrcSampleOffsetBiased));

    // To Test: Save reciproqual sqrt compute
    // FfxFloat32 fSampleWeight = Lanczos2Sq_UseLUT(dot(fSrcSampleOffsetBiased, fSrcSampleOffsetBiased));
#else
#error "Invalid Lanczos type"
#endif
    return fSampleWeight;
}
#endif

FfxFloat32 ComputeMaxKernelWeight(const AccumulationPassCommonParams params, FFX_PARAMETER_INOUT AccumulationPassData data) {

    const FfxFloat32 fKernelSizeBias = 1.0f + (1.0f / FfxFloat32x2(DownscaleFactor()) - 1.0f).x;

    return ffxMin(FfxFloat32(1.99f), fKernelSizeBias);
}

FfxFloat32x3 LoadPreparedColor(FfxInt32x2 iSamplePos)
{
    const FfxFloat32x3 fRgb             = ffxMax(FfxFloat32x3(0, 0, 0), LoadInputColor(iSamplePos)) * Exposure();
    const FfxFloat32x3 fPreparedYCoCg   = RGBToYCoCg(fRgb);

    return fPreparedYCoCg;
}

#if FFX_HALF && (FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2) && defined(__XBOX_SCARLETT) && defined(__XBATG_EXTRA_16_BIT_OPTIMISATION) && (__XBATG_EXTRA_16_BIT_OPTIMISATION == 1)
#define FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS 1
#else
#define FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS 0
#endif

#if FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS

void LoadPreparedColorPairedRgb(FFX_PARAMETER_OUT FFX_MIN16_F2 r,
                                FFX_PARAMETER_OUT FFX_MIN16_F2 g,
                                FFX_PARAMETER_OUT FFX_MIN16_F2 b,
                                FfxInt32x2 iSamplePos0,
                                FfxInt32x2 iSamplePos1)
{
    const FFX_MIN16_F3 sample0 = FFX_MIN16_F3(LoadInputColor(iSamplePos0));
    const FFX_MIN16_F3 sample1 = FFX_MIN16_F3(LoadInputColor(iSamplePos1));

    r = ffxMax(FFX_MIN16_F2(0, 0), FFX_MIN16_F2(sample0.r, sample1.r));
    g = ffxMax(FFX_MIN16_F2(0, 0), FFX_MIN16_F2(sample0.g, sample1.g));
    b = ffxMax(FFX_MIN16_F2(0, 0), FFX_MIN16_F2(sample0.b, sample1.b));

    r = FFX_MIN16_F2(r * Exposure());
    g = FFX_MIN16_F2(g * Exposure());
    b = FFX_MIN16_F2(b * Exposure());
}

void TonemapPaired(FFX_PARAMETER_INOUT FFX_MIN16_F2 r, FFX_PARAMETER_INOUT FFX_MIN16_F2 g, FFX_PARAMETER_INOUT FFX_MIN16_F2 b)
{
    FFX_MIN16_F2 denomF16 = ffxMax(ffxMax(ffxMax(0.0, r), g), b) + FFX_MIN16_F2(1.0, 1.0);

    // NOTE: expect 2 x v_cvt_f32_f16
    FfxFloat32x2 denomF32 = FfxFloat32x2(denomF16);
    // NOTE: expect 2 x v_rcp_f32
    FfxFloat32x2 normF32 = FfxFloat32x2(1.0, 1.0) / denomF32;
    // NOTE: expect 2 x v_cvt_f16_f32
    FFX_MIN16_F2 normF16 = FFX_MIN16_F2(normF32);

    r *= normF16;
    g *= normF16;
    b *= normF16;
}

void RGBToYCoCgPaired(FFX_PARAMETER_INOUT FFX_MIN16_F2 r, FFX_PARAMETER_INOUT FFX_MIN16_F2 g, FFX_PARAMETER_INOUT FFX_MIN16_F2 b)
{
    /**
     *  NOTE: given the following conversion
     *
     *      fYCoCg = FfxFloat32x3(
     *          0.25f * fRgb.r + 0.5f * fRgb.g + 0.25f * fRgb.b,
     *           0.5f * fRgb.r - 0.5f * fRgb.b,
     *         -0.25f * fRgb.r + 0.5f * fRgb.g - 0.25f * fRgb.b);
     *
     *  it's possible to notice that we can compute:
     *      RplusBdiv4 = 0.25 * (R + B)
     *
     *  so everything else is computed in 3 instructions
     *      Y  = G * 0.5 + RplusBdiv4
     *      Co = 2 * RplusBdiv4 - G
     *      Cg = G * 0.5 - RplusBdiv4
     */

    // NOTE: expect v_pk_add_f32 + v_pk_mul_f32
    FFX_MIN16_F2 RplusBdiv4 = (r + b) * 0.25;
    FFX_MIN16_F2 G = g;
    FFX_MIN16_F2 B = b;

    // NOTE: expect 3x v_pk_fma_f32
    r = G * 0.5 + RplusBdiv4;
    g = RplusBdiv4 * 2.0 - B;
    b = G * 0.5 - RplusBdiv4;
}

FFX_MIN16_F2 Compute3x3SamplesMinMaxPaired(FFX_PARAMETER_IN FFX_MIN16_F2 sampleCenter,
                                           FFX_PARAMETER_IN FFX_MIN16_F2 sample0,
                                           FFX_PARAMETER_IN FFX_MIN16_F2 sample1,
                                           FFX_PARAMETER_IN FFX_MIN16_F2 sample2,
                                           FFX_PARAMETER_IN FFX_MIN16_F2 sample3)
{
    FFX_MIN16_F2 twoMinValues = ffxMin(ffxMin(sample0, sample1), ffxMin(sample2, sample3));
    FFX_MIN16_F2 twoMaxValues = ffxMax(ffxMax(sample0, sample1), ffxMax(sample2, sample3));

    return FFX_MIN16_F2(
        ffxMin3Half(twoMinValues.x, twoMinValues.y, sampleCenter.x),
        ffxMax3Half(twoMaxValues.x, twoMaxValues.y, sampleCenter.x)
    );
}


FFX_MIN16_F2 Bool2ToFloat16x2(bool x, bool y)
{
    uint lo = x ? 0x00003c00 : 0x00000000;
    uint hi = y ? 0x3c000000 : 0x00000000;
    return FFX_MIN16_F2(__XB_AsHalf(lo).x, __XB_AsHalf(hi).y);
}

struct PairedRectificationBoxAndAccumulatedColorAndWeight
{
    FFX_MIN16_F2 boxCenterR;
    FFX_MIN16_F2 boxCenterG;
    FFX_MIN16_F2 boxCenterB;

    FFX_MIN16_F2 boxVecR;
    FFX_MIN16_F2 boxVecG;
    FFX_MIN16_F2 boxVecB;

    FFX_MIN16_F2 fBoxCenterWeight;

    FFX_MIN16_F2 fColorR;
    FFX_MIN16_F2 fColorG;
    FFX_MIN16_F2 fColorB;
    FFX_MIN16_F2 fWeight;

    FFX_MIN16_F fKernelBiasSq;
    FFX_MIN16_F fRectificationCurveBias;

    void setKernelBiasAndRectificationCurveBias(FfxFloat32 kernelBias, FfxFloat32 rectificationCurveBias)
    {
        fKernelBiasSq = FFX_MIN16_F(kernelBias * kernelBias);
        fRectificationCurveBias = FFX_MIN16_F(rectificationCurveBias);
    }

    void initUpscaledColor(FFX_MIN16_F fSrcSampleOffsetSq, FFX_MIN16_F fOnScreenWeight, FFX_MIN16_F2 sampleR, FFX_MIN16_F2 sampleG, FFX_MIN16_F2 sampleB)
    {
        #if FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2 // LANCZOS_TYPE_APPROXIMATE
            const FFX_MIN16_F2 LanczosUpsampleWeight = FFX_MIN16_F2(
                PairedLanczos2ApproxSq(fSrcSampleOffsetSq * fKernelBiasSq).x,
                0.0
            );
        #else
            #error "Only LANCZOS_TYPE_APPROXIMATE is supported in paired version so far"
        #endif
        const FFX_MIN16_F2 fSampleWeight = fOnScreenWeight * LanczosUpsampleWeight;

        fColorR = sampleR * fSampleWeight;
        fColorG = sampleG * fSampleWeight;
        fColorB = sampleB * fSampleWeight;
        fWeight = fSampleWeight;
    }

    void initBox(FFX_MIN16_F fSrcSampleOffsetSq, FFX_MIN16_F fOnScreenWeight, FFX_MIN16_F2 sampleR, FFX_MIN16_F2 sampleG, FFX_MIN16_F2 sampleB)
    {
        const FFX_MIN16_F2 fBoxSampleWeight = FFX_MIN16_F2(
            exp(fRectificationCurveBias * fSrcSampleOffsetSq) * fOnScreenWeight,
            0.0
        );

        FFX_MIN16_F2 weightedSampleR = sampleR * fBoxSampleWeight;
        FFX_MIN16_F2 weightedSampleG = sampleG * fBoxSampleWeight;
        FFX_MIN16_F2 weightedSampleB = sampleB * fBoxSampleWeight;

        boxCenterR = weightedSampleR;
        boxCenterG = weightedSampleG;
        boxCenterB = weightedSampleB;

        boxVecR = sampleR * weightedSampleR;
        boxVecG = sampleG * weightedSampleG;
        boxVecB = sampleB * weightedSampleB;

        fBoxCenterWeight = fBoxSampleWeight;
    }

    void addUpscaledColorSample(FFX_MIN16_F2 fSrcSampleOffsetSq, FFX_MIN16_F2 fOnScreenWeight, FFX_MIN16_F2 sampleR, FFX_MIN16_F2 sampleG, FFX_MIN16_F2 sampleB)
    {
        #if FFX_FSR3UPSCALER_OPTION_UPSAMPLE_USE_LANCZOS_TYPE == 2 // LANCZOS_TYPE_APPROXIMATE
            const FFX_MIN16_F2 LanczosUpsampleWeight = PairedLanczos2ApproxSq(fSrcSampleOffsetSq * fKernelBiasSq);
        #else
            #error "Only LANCZOS_TYPE_APPROXIMATE is supported in paired version so far"
        #endif
        const FFX_MIN16_F2 fSampleWeight = fOnScreenWeight * LanczosUpsampleWeight;

        fColorR += sampleR * fSampleWeight;
        fColorG += sampleG * fSampleWeight;
        fColorB += sampleB * fSampleWeight;
        fWeight += fSampleWeight;
    }

    void addBoxSample(FFX_MIN16_F2 fSrcSampleOffsetSq, FFX_MIN16_F2 fOnScreenWeight, FFX_MIN16_F2 sampleR, FFX_MIN16_F2 sampleG, FFX_MIN16_F2 sampleB)
    {
        // NOTE: ideally expect here 2x v_fma_mix + 2x v_exp_f32 + 2x v_fma_mix
        const FFX_MIN16_F2 fBoxSampleWeight = exp(fRectificationCurveBias * fSrcSampleOffsetSq) * fOnScreenWeight;

        FFX_MIN16_F2 weightedSampleR = sampleR * fBoxSampleWeight;
        FFX_MIN16_F2 weightedSampleG = sampleG * fBoxSampleWeight;
        FFX_MIN16_F2 weightedSampleB = sampleB * fBoxSampleWeight;

        boxCenterR += weightedSampleR;
        boxCenterG += weightedSampleG;
        boxCenterB += weightedSampleB;

        boxVecR += sampleR * weightedSampleR;
        boxVecG += sampleG * weightedSampleG;
        boxVecB += sampleB * weightedSampleB;

        fBoxCenterWeight += fBoxSampleWeight;
    }

    void finalizeUpscaledColor(FFX_PARAMETER_OUT FfxFloat32x4 upscaledColorAndWeight)
    {
        upscaledColorAndWeight.r = fColorR.x + fColorR.y;
        upscaledColorAndWeight.g = fColorG.x + fColorG.y;
        upscaledColorAndWeight.b = fColorB.x + fColorB.y;

        upscaledColorAndWeight.a = fWeight.x + fWeight.y;
    }

    void finalizeBox(FFX_PARAMETER_OUT FfxFloat32x2 boxCenterAndVecR,
                     FFX_PARAMETER_OUT FfxFloat32x2 boxCenterAndVecG,
                     FFX_PARAMETER_OUT FfxFloat32x2 boxCenterAndVecB,
                     FFX_PARAMETER_OUT FfxFloat32   boxCenterWeight)
    {
        boxCenterAndVecR = FfxFloat32x2(boxCenterR.x + boxCenterR.y, boxVecR.x + boxVecR.y);
        boxCenterAndVecG = FfxFloat32x2(boxCenterG.x + boxCenterG.y, boxVecG.x + boxVecG.y);
        boxCenterAndVecB = FfxFloat32x2(boxCenterB.x + boxCenterB.y, boxVecB.x + boxVecB.y);

        boxCenterWeight = fBoxCenterWeight.x + fBoxCenterWeight.y;
    }
};
#endif // #if FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS

void ComputeUpsampledColorAndWeight(const AccumulationPassCommonParams params, FFX_PARAMETER_INOUT AccumulationPassData data)
{
    // We compute a sliced lanczos filter with 2 lobes (other slices are accumulated temporaly)
    const FfxFloat32x2 fDstOutputPos        = FfxFloat32x2(params.iPxHrPos) + FFX_BROADCAST_FLOAT32X2(0.5f);
    const FfxFloat32x2 fSrcOutputPos        = fDstOutputPos * DownscaleFactor();
    const FfxInt32x2   iSrcInputPos         = FfxInt32x2(floor(fSrcOutputPos));
    const FfxFloat32x2 fSrcUnjitteredPos    = (FfxFloat32x2(iSrcInputPos) + FfxFloat32x2(0.5f, 0.5f)) - Jitter(); // This is the un-jittered position of the sample at offset 0,0
    const FfxFloat32x2 fBaseSampleOffset    = FfxFloat32x2(fSrcUnjitteredPos - fSrcOutputPos);

    FfxInt32x2 offsetTL;
    offsetTL.x = (fSrcUnjitteredPos.x > fSrcOutputPos.x) ? FfxInt32(-2) : FfxInt32(-1);
    offsetTL.y = (fSrcUnjitteredPos.y > fSrcOutputPos.y) ? FfxInt32(-2) : FfxInt32(-1);

    //Load samples
    // If fSrcUnjitteredPos.y > fSrcOutputPos.y, indicates offsetTL.y = -2, sample offset Y will be [-2, 1], clipbox will be rows [1, 3].
    // Flip row# for sampling offset in this case, so first 0~2 rows in the sampled array can always be used for computing the clipbox.
    // This reduces branch or cmove on sampled colors, but moving this overhead to sample position / weight calculation time which apply to less values.
    const FfxBoolean bFlipRow = fSrcUnjitteredPos.y > fSrcOutputPos.y;
    const FfxBoolean bFlipCol = fSrcUnjitteredPos.x > fSrcOutputPos.x;
    const FfxFloat32x2 fOffsetTL = FfxFloat32x2(offsetTL);

    const FfxBoolean bIsInitialSample = (params.fAccumulation == 0.0f);

#if FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS
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

    FFX_MIN16_F2 TopCornerR, BotCornerR, HorzR, VertR, CenterR;
    FFX_MIN16_F2 TopCornerG, BotCornerG, HorzG, VertG, CenterG;
    FFX_MIN16_F2 TopCornerB, BotCornerB, HorzB, VertB, CenterB;

    LoadPreparedColorPairedRgb(TopCornerR, TopCornerG, TopCornerB,
        FfxTexCoordI2(iSrcSamplePosX01Clamped.x, iSrcSamplePosY01Clamped.x),
        FfxTexCoordI2(iSrcSamplePosX23Clamped.x, iSrcSamplePosY01Clamped.x)
    );

    LoadPreparedColorPairedRgb(BotCornerR, BotCornerG, BotCornerB,
        FfxTexCoordI2(iSrcSamplePosX01Clamped.x, iSrcSamplePosY23Clamped.x),
        FfxTexCoordI2(iSrcSamplePosX23Clamped.x, iSrcSamplePosY23Clamped.x)
    );

    LoadPreparedColorPairedRgb(HorzR, HorzG, HorzB,
        FfxTexCoordI2(iSrcSamplePosX01Clamped.x, iSrcSamplePosY01Clamped.y),
        FfxTexCoordI2(iSrcSamplePosX23Clamped.x, iSrcSamplePosY01Clamped.y)
    );

    LoadPreparedColorPairedRgb(VertR, VertG, VertB,
        FfxTexCoordI2(iSrcSamplePosX01Clamped.y, iSrcSamplePosY01Clamped.x),
        FfxTexCoordI2(iSrcSamplePosX01Clamped.y, iSrcSamplePosY23Clamped.x)
    );

    // NOTE: duplicated data
    LoadPreparedColorPairedRgb(CenterR, CenterG, CenterB,
        FfxTexCoordI2(iSrcSamplePosX01Clamped.y, iSrcSamplePosY01Clamped.y),
        FfxTexCoordI2(iSrcSamplePosX01Clamped.y, iSrcSamplePosY01Clamped.y)
    );

    #if FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT
    if (bIsInitialSample)
    {
        TonemapPaired(TopCornerR, TopCornerG, TopCornerB);
        TonemapPaired(BotCornerR, BotCornerG, BotCornerB);
        TonemapPaired(HorzR, HorzG, HorzB);
        TonemapPaired(VertR, VertG, VertB);
        TonemapPaired(CenterR, CenterG, CenterB);
    }
    #endif

    RGBToYCoCgPaired(TopCornerR, TopCornerG, TopCornerB);
    RGBToYCoCgPaired(BotCornerR, BotCornerG, BotCornerB);
    RGBToYCoCgPaired(HorzR, HorzG, HorzB);
    RGBToYCoCgPaired(VertR, VertG, VertB);
    RGBToYCoCgPaired(CenterR, CenterG, CenterB);

#else
    FfxFloat32x3 fSamples[9];
    FfxInt32 iSampleIndex = 0;

    FFX_UNROLL
    for (FfxInt32 row = 0; row < 3; row++) {
        FFX_UNROLL
        for (FfxInt32 col = 0; col < 3; col++) {
            const FfxInt32x2 iSampleColRow = FfxInt32x2(bFlipCol ? (3 - col) : col, bFlipRow ? (3 - row) : row);
            const FfxInt32x2 iSrcSamplePos = FfxInt32x2(iSrcInputPos) + offsetTL + iSampleColRow;
            const FfxInt32x2 iSampleCoord = ClampLoad(iSrcSamplePos, FfxInt32x2(0, 0), FfxInt32x2(RenderSize()));

            fSamples[iSampleIndex] = LoadPreparedColor(iSampleCoord);

            ++iSampleIndex;
        }
    }

#if FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT
    if (bIsInitialSample)
    {
        for (iSampleIndex = 0; iSampleIndex < 9; ++iSampleIndex)
        {
            //YCoCg -> RGB -> Tonemap -> YCoCg (Use RGB tonemapper to avoid color desaturation)
            fSamples[iSampleIndex] = RGBToYCoCg(Tonemap(YCoCgToRGB(fSamples[iSampleIndex])));
        }
    }
#endif

#endif // #if FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS

    // Identify how much of each upsampled color to be used for this frame
    const FfxFloat32 fKernelBiasMax          = ComputeMaxKernelWeight(params, data);
    const FfxFloat32 fKernelBiasMin          = ffxMax(1.0f, ((1.0f + fKernelBiasMax) * 0.3f));

    const FfxFloat32 fKernelBiasWeight =
        ffxMin(1.0f - params.fDisocclusion * 0.5f,
        ffxMin(1.0f - params.fShadingChange,
        ffxSaturate(data.fHistoryWeight * 5.0f)
        ));

    const FfxFloat32 fKernelBias             = ffxLerp(fKernelBiasMin, fKernelBiasMax, fKernelBiasWeight);
    
#if FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS
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

    const FfxFloat32 fRectificationCurveBias = -2.3f;
    PairedRectificationBoxAndAccumulatedColorAndWeight pairedBox;
    pairedBox.setKernelBiasAndRectificationCurveBias(fKernelBias, fRectificationCurveBias);

    // init by o o o
    //         o x o
    //         o o o
    pairedBox.initBox(
        fSrcSampleOffsetXYSq_11.x + fSrcSampleOffsetXYSq_11.y,
        Bool2ToFloat16x2(coordX1OnScreen && coordY1OnScreen, false).x,
        CenterR, CenterG, CenterB
    );

    // add remaining two samples from 1st row x o x
    //                                        o * o
    //                                        o o o
    pairedBox.addBoxSample(
        fSrcSampleOffsetXSq_02 + fSrcSampleOffsetYSq_02.xx,
        Bool2ToFloat16x2(coordX0OnScreen && coordY0OnScreen, coordX2OnScreen && coordY0OnScreen),
        TopCornerR, TopCornerG, TopCornerB
    );

    // add two samples from 2nd row * o *
    //                              o * o
    //                              x o x
    pairedBox.addBoxSample(
        fSrcSampleOffsetXSq_02 + fSrcSampleOffsetYSq_02.yy,
        Bool2ToFloat16x2(coordX0OnScreen && coordY2OnScreen, coordX2OnScreen && coordY2OnScreen),
        BotCornerR, BotCornerG, BotCornerB
    );

    // add two samples from 3rd row * o *
    //                              x * x
    //                              * o *
    pairedBox.addBoxSample(
        fSrcSampleOffsetXSq_02 + fSrcSampleOffsetXYSq_11.yy,
        Bool2ToFloat16x2(coordX0OnScreen && coordY1OnScreen, coordX2OnScreen && coordY1OnScreen),
        HorzR, HorzG, HorzB
    );

    // add remaining samples * x *
    //                       * * *
    //                       * x *
    pairedBox.addBoxSample(
        fSrcSampleOffsetXYSq_11.xx + fSrcSampleOffsetYSq_02,
        Bool2ToFloat16x2(coordX1OnScreen && coordY0OnScreen, coordX1OnScreen && coordY2OnScreen),
        VertR, VertG, VertB
    );

    FfxFloat32x2 boxCenterAndVecR, boxCenterAndVecG, boxCenterAndVecB;
    FfxFloat32 boxCenterWeight;
    pairedBox.finalizeBox(boxCenterAndVecR, boxCenterAndVecG, boxCenterAndVecB, boxCenterWeight);

    if (!bIsInitialSample)
    {
        pairedBox.initUpscaledColor(
            fSrcSampleOffsetXYSq_11.x + fSrcSampleOffsetXYSq_11.y,
            Bool2ToFloat16x2(coordX1OnScreen && coordY1OnScreen, false).x,
            CenterR, CenterG, CenterB
        );

        // add remaining two samples from 1st row x o x
        //                                        o * o
        //                                        o o o
        pairedBox.addUpscaledColorSample(
            fSrcSampleOffsetXSq_02 + fSrcSampleOffsetYSq_02.xx,
            Bool2ToFloat16x2(coordX0OnScreen && coordY0OnScreen, coordX2OnScreen && coordY0OnScreen),
            TopCornerR, TopCornerG, TopCornerB
        );

        // add two samples from 2nd row * o *
        //                              o * o
        //                              x o x
        pairedBox.addUpscaledColorSample(
            fSrcSampleOffsetXSq_02 + fSrcSampleOffsetYSq_02.yy,
            Bool2ToFloat16x2(coordX0OnScreen && coordY2OnScreen, coordX2OnScreen && coordY2OnScreen),
            BotCornerR, BotCornerG, BotCornerB
        );

        // add two samples from 3rd row * o *
        //                              x * x
        //                              * o *
        pairedBox.addUpscaledColorSample(
            fSrcSampleOffsetXSq_02 + fSrcSampleOffsetXYSq_11.yy,
            Bool2ToFloat16x2(coordX0OnScreen && coordY1OnScreen, coordX2OnScreen && coordY1OnScreen),
            HorzR, HorzG, HorzB
        );

        // add remaining samples * x *
        //                       * * *
        //                       * x *
        pairedBox.addUpscaledColorSample(
            fSrcSampleOffsetXYSq_11.xx + fSrcSampleOffsetYSq_02,
            Bool2ToFloat16x2(coordX1OnScreen && coordY0OnScreen, coordX1OnScreen && coordY2OnScreen),
            VertR, VertG, VertB
        );

        FfxFloat32x4 upscaledColorAndWeight = 0.0;
        pairedBox.finalizeUpscaledColor(upscaledColorAndWeight);

        data.fUpsampledColor    = FfxFloat32x3(upscaledColorAndWeight.rgb);
        data.fUpsampledWeight   = FfxFloat32(upscaledColorAndWeight.w);
    }

    FFX_MIN16_F2 aabbMinMaxR = Compute3x3SamplesMinMaxPaired(CenterR, TopCornerR, BotCornerR, HorzR, VertR);
    FFX_MIN16_F2 aabbMinMaxG = Compute3x3SamplesMinMaxPaired(CenterG, TopCornerG, BotCornerG, HorzG, VertG);
    FFX_MIN16_F2 aabbMinMaxB = Compute3x3SamplesMinMaxPaired(CenterB, TopCornerB, BotCornerB, HorzB, VertB);

    data.clippingBox.boxCenter          = FfxFloat32x3(boxCenterAndVecR.x, boxCenterAndVecG.x, boxCenterAndVecB.x);
    data.clippingBox.boxVec             = FfxFloat32x3(boxCenterAndVecR.y, boxCenterAndVecG.y, boxCenterAndVecB.y);
    data.clippingBox.aabbMin            = FfxFloat32x3(aabbMinMaxR.x, aabbMinMaxG.x, aabbMinMaxB.x);
    data.clippingBox.aabbMax            = FfxFloat32x3(aabbMinMaxR.y, aabbMinMaxG.y, aabbMinMaxB.y);
    data.clippingBox.fBoxCenterWeight   = FfxFloat32(boxCenterWeight);
#else

    iSampleIndex = 0;

    FFX_UNROLL
    for (FfxInt32 row = 0; row < 3; row++)
    {
        FFX_UNROLL
        for (FfxInt32 col = 0; col < 3; col++)
        {
            const FfxInt32x2   sampleColRow     = FfxInt32x2(bFlipCol ? (3 - col) : col, bFlipRow ? (3 - row) : row);
            const FfxFloat32x2 fOffset          = fOffsetTL + FfxFloat32x2(sampleColRow);
            const FfxFloat32x2 fSrcSampleOffset = fBaseSampleOffset + fOffset;

            const FfxInt32x2 iSrcSamplePos   = FfxInt32x2(iSrcInputPos) + FfxInt32x2(offsetTL) + sampleColRow;
            const FfxFloat32 fOnScreenFactor = FfxFloat32(IsOnScreen(FfxInt32x2(iSrcSamplePos), FfxInt32x2(RenderSize())));

            if (!bIsInitialSample)
            {
                const FfxFloat32 fSampleWeight = fOnScreenFactor * FfxFloat32(GetUpsampleLanczosWeight(fSrcSampleOffset, fKernelBias));

                data.fUpsampledColor += fSamples[iSampleIndex] * fSampleWeight;
                data.fUpsampledWeight += fSampleWeight;
            }

            // Update rectification box
            {
                const FfxFloat32 fRectificationCurveBias = -2.3f;
                const FfxFloat32 fSrcSampleOffsetSq = dot(fSrcSampleOffset, fSrcSampleOffset);
                const FfxFloat32 fBoxSampleWeight   = exp(fRectificationCurveBias * fSrcSampleOffsetSq) * fOnScreenFactor;

                const FfxBoolean bInitialSample = (row == 0) && (col == 0);
                RectificationBoxAddSample(bInitialSample, data.clippingBox, fSamples[iSampleIndex], fBoxSampleWeight);
            }
            ++iSampleIndex;
        }
    }
	
#endif // #if FFX_FSR3UPSCALER_USE_XBOX_PAIRED_16BIT_MATH_OPTIMIZATIONS

    RectificationBoxComputeVarianceBoxData(data.clippingBox);

    data.fUpsampledWeight *= FfxFloat32(data.fUpsampledWeight > FSR3UPSCALER_EPSILON);

    if (data.fUpsampledWeight > FSR3UPSCALER_EPSILON) {
        // Normalize for deringing (we need to compare colors)
        data.fUpsampledColor = data.fUpsampledColor / data.fUpsampledWeight;
        data.fUpsampledWeight *= fAverageLanczosWeightPerFrame;

        Deringing(data.clippingBox, data.fUpsampledColor);
    }

    // Initial samples using tonemapped upsampling
    if (bIsInitialSample) {
#if FFX_FSR3UPSCALER_OPTION_HDR_COLOR_INPUT
        data.fUpsampledColor  = RGBToYCoCg(InverseTonemap(YCoCgToRGB(data.clippingBox.boxCenter)));
#else
        data.fUpsampledColor  = data.clippingBox.boxCenter;
#endif
        data.fUpsampledWeight = 1.0f;
        data.fHistoryWeight   = 0.0f;
    }
}
