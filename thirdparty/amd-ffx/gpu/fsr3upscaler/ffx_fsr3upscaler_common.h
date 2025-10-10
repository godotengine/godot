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

#if !defined(FFX_FSR3UPSCALER_COMMON_H)
#define FFX_FSR3UPSCALER_COMMON_H

#if defined(FFX_GPU)
#pragma warning(error   : 3206)  // treat vector truncation warnings as errors
#pragma warning(disable : 3205)  // conversion from larger type to smaller
#pragma warning(disable : 3571)  // in ffxPow(f, e), f could be negative

FFX_STATIC const FfxFloat32 FSR3UPSCALER_FP16_MIN           = 6.10e-05f;
FFX_STATIC const FfxFloat32 FSR3UPSCALER_FP16_MAX           = 65504.0f;
FFX_STATIC const FfxFloat32 FSR3UPSCALER_EPSILON            = FSR3UPSCALER_FP16_MIN;
FFX_STATIC const FfxFloat32 FSR3UPSCALER_TONEMAP_EPSILON    = FSR3UPSCALER_FP16_MIN;
FFX_STATIC const FfxFloat32 FSR3UPSCALER_FP32_MAX           = 3.402823466e+38f;
FFX_STATIC const FfxFloat32 FSR3UPSCALER_FP32_MIN           = 1.175494351e-38f;

// Reconstructed depth usage
FFX_STATIC const FfxFloat32 fReconstructedDepthBilinearWeightThreshold  = FSR3UPSCALER_EPSILON * 10;

FfxFloat32 ReconstructedDepthMvPxThreshold(FfxFloat32 fNearestDepthInMeters)
{
    return ffxLerp(0.25f, 0.75f, ffxSaturate(fNearestDepthInMeters / 100.0f));
}

// Accumulation
FFX_STATIC const FfxFloat32 fUpsampleLanczosWeightScale     = 1.0f / 16.0f;
FFX_STATIC const FfxFloat32 fAverageLanczosWeightPerFrame   = 0.74f * fUpsampleLanczosWeightScale; // Average lanczos weight for jitter accumulated samples
FFX_STATIC const FfxFloat32 fAccumulationMaxOnMotion        = 3.0f * fUpsampleLanczosWeightScale;

#define SHADING_CHANGE_SET_SIZE 5
FFX_STATIC const FfxInt32 iShadingChangeMipStart = 0;
FFX_STATIC const FfxFloat32 fShadingChangeSamplePow = 1.0f / 1.0f;


FFX_STATIC const FfxFloat32 fLockThreshold  = 1.0f;
FFX_STATIC const FfxFloat32 fLockMax        = 2.0f;

FFX_STATIC const FfxInt32 REACTIVE          = 0;
FFX_STATIC const FfxInt32 DISOCCLUSION      = 1;
FFX_STATIC const FfxInt32 SHADING_CHANGE    = 2;
FFX_STATIC const FfxInt32 ACCUMULAION       = 3;

FFX_STATIC const FfxInt32 FRAME_INFO_EXPOSURE           = 0;
FFX_STATIC const FfxInt32 FRAME_INFO_LOG_LUMA           = 1;
FFX_STATIC const FfxInt32 FRAME_INFO_SCENE_AVERAGE_LUMA = 2;

FfxBoolean TonemapFirstFrame()
{
    const FfxBoolean bEnabled = true;
    return FrameIndex() == 0 && bEnabled;
}

FfxFloat32 AverageLanczosWeightPerFrame()
{
    return 0.74f;
}

FfxInt32x2 ShadingChangeRenderSize()
{
    return FfxInt32x2(RenderSize() * 0.5f);
}

FfxInt32x2 ShadingChangeMaxRenderSize()
{
    return FfxInt32x2(MaxRenderSize() * 0.5f);
}

FfxInt32x2 PreviousFrameShadingChangeRenderSize()
{
    return FfxInt32x2(PreviousFrameRenderSize() * 0.5f);
}

#if defined(FSR3UPSCALER_BIND_SRV_FRAME_INFO)
FfxFloat32 SceneAverageLuma()
{
    return FrameInfo()[FRAME_INFO_SCENE_AVERAGE_LUMA];
}
#endif

// Auto exposure
FFX_STATIC const FfxFloat32 resetAutoExposureAverageSmoothing = 1e4f;

struct AccumulationPassCommonParams
{
    FfxInt32x2      iPxHrPos;
    FfxFloat32x2    fHrUv;
    FfxFloat32x2    fLrUvJittered;
    FfxFloat32x2    fLrUv_HwSampler;
    FfxFloat32x2    fMotionVector;
    FfxFloat32x2    fReprojectedHrUv;
    FfxFloat32      f4KVelocity;
    FfxFloat32      fDisocclusion;
    FfxFloat32      fReactiveMask;
    FfxFloat32      fShadingChange;
    FfxFloat32      fAccumulation;
    FfxFloat32      fLumaInstabilityFactor;
    FfxFloat32      fFarthestDepthInMeters;

    FfxBoolean      bIsExistingSample;
    FfxBoolean      bIsNewSample;
};

FfxFloat32 Get4KVelocity(FfxFloat32x2 fMotionVector)
{
    return length(fMotionVector * FfxFloat32x2(3840.0f, 2160.0f));
}

struct RectificationBox
{
    FfxFloat32x3        boxCenter;
    FfxFloat32x3        boxVec;
    FfxFloat32x3        aabbMin;
    FfxFloat32x3        aabbMax;
    FfxFloat32          fBoxCenterWeight;
};

struct AccumulationPassData
{
    RectificationBox    clippingBox;
    FfxFloat32x3        fUpsampledColor;
    FfxFloat32          fUpsampledWeight;
    FfxFloat32x3        fHistoryColor;
    FfxFloat32          fHistoryWeight;
    FfxFloat32          fLock;
    FfxFloat32          fLockContributionThisFrame;
};

void RectificationBoxAddInitialSample(FFX_PARAMETER_INOUT RectificationBox rectificationBox, const FfxFloat32x3 colorSample, const FfxFloat32 fSampleWeight)
{
    rectificationBox.aabbMin = colorSample;
    rectificationBox.aabbMax = colorSample;

    FfxFloat32x3 weightedSample = colorSample * fSampleWeight;
    rectificationBox.boxCenter = weightedSample;
    rectificationBox.boxVec = colorSample * weightedSample;
    rectificationBox.fBoxCenterWeight = fSampleWeight;
}

void RectificationBoxAddSample(FfxBoolean bInitialSample, FFX_PARAMETER_INOUT RectificationBox rectificationBox, const FfxFloat32x3 colorSample, const FfxFloat32 fSampleWeight)
{
    if (bInitialSample) {
        RectificationBoxAddInitialSample(rectificationBox, colorSample, fSampleWeight);
    } else {
        rectificationBox.aabbMin = ffxMin(rectificationBox.aabbMin, colorSample);
        rectificationBox.aabbMax = ffxMax(rectificationBox.aabbMax, colorSample);

        FfxFloat32x3 weightedSample = colorSample * fSampleWeight;
        rectificationBox.boxCenter += weightedSample;
        rectificationBox.boxVec += colorSample * weightedSample;
        rectificationBox.fBoxCenterWeight += fSampleWeight;
    }
}

void RectificationBoxComputeVarianceBoxData(FFX_PARAMETER_INOUT RectificationBox rectificationBox)
{
    rectificationBox.fBoxCenterWeight = (abs(rectificationBox.fBoxCenterWeight) > FfxFloat32(FSR3UPSCALER_FP32_MIN) ? rectificationBox.fBoxCenterWeight : FfxFloat32(1.f));
    rectificationBox.boxCenter /= rectificationBox.fBoxCenterWeight;
    rectificationBox.boxVec /= rectificationBox.fBoxCenterWeight;
    FfxFloat32x3 stdDev = sqrt(abs(rectificationBox.boxVec - rectificationBox.boxCenter * rectificationBox.boxCenter));
    rectificationBox.boxVec = stdDev;
}

FfxFloat32x3 SafeRcp3(FfxFloat32x3 v)
{
    return (all(FFX_NOT_EQUAL(v, FfxFloat32x3(0, 0, 0)))) ? (FfxFloat32x3(1, 1, 1) / v) : FfxFloat32x3(0, 0, 0);
}

FfxFloat32 MinDividedByMax(const FfxFloat32 v0, const FfxFloat32 v1, const FfxFloat32 fOnZeroReturnValue)
{
    const FfxFloat32 m = ffxMax(v0, v1);
    return m != 0 ? ffxMin(v0, v1) / m : fOnZeroReturnValue;
}

FfxFloat32 MinDividedByMax(const FfxFloat32 v0, const FfxFloat32 v1)
{
    const FfxFloat32 m = ffxMax(v0, v1);
    return m != 0 ? ffxMin(v0, v1) / m : 0;
}

FfxFloat32x3 YCoCgToRGB(FfxFloat32x3 fYCoCg)
{
    FfxFloat32x3 fRgb;

    fRgb = FfxFloat32x3(
        fYCoCg.x + fYCoCg.y - fYCoCg.z,
        fYCoCg.x + fYCoCg.z,
        fYCoCg.x - fYCoCg.y - fYCoCg.z);

    return fRgb;
}

FfxFloat32x3 RGBToYCoCg(FfxFloat32x3 fRgb)
{
    FfxFloat32x3 fYCoCg;

    fYCoCg = FfxFloat32x3(
        0.25f * fRgb.r + 0.5f * fRgb.g + 0.25f * fRgb.b,
        0.5f * fRgb.r - 0.5f * fRgb.b,
        -0.25f * fRgb.r + 0.5f * fRgb.g - 0.25f * fRgb.b);

    return fYCoCg;
}

FfxFloat32 RGBToLuma(FfxFloat32x3 fLinearRgb)
{
    return dot(fLinearRgb, FfxFloat32x3(0.2126f, 0.7152f, 0.0722f));
}

FfxFloat32 RGBToPerceivedLuma(FfxFloat32x3 fLinearRgb)
{
    FfxFloat32 fLuminance = RGBToLuma(fLinearRgb);

    FfxFloat32 fPercievedLuminance = 0;
    if (fLuminance <= 216.0f / 24389.0f) {
        fPercievedLuminance = fLuminance * (24389.0f / 27.0f);
    }
    else {
        fPercievedLuminance = ffxPow(fLuminance, 1.0f / 3.0f) * 116.0f - 16.0f;
    }

    return fPercievedLuminance * 0.01f;
}

FfxFloat32x3 Tonemap(FfxFloat32x3 fRgb)
{
    return fRgb / (ffxMax(ffxMax(0.f, fRgb.r), ffxMax(fRgb.g, fRgb.b)) + 1.f).xxx;
}

FfxFloat32x3 InverseTonemap(FfxFloat32x3 fRgb)
{
    return fRgb / ffxMax(FSR3UPSCALER_TONEMAP_EPSILON, 1.f - ffxMax(fRgb.r, ffxMax(fRgb.g, fRgb.b))).xxx;
}

FfxBoolean IsUvInside(FfxFloat32x2 fUv)
{
    return (fUv.x >= 0.0f && fUv.x <= 1.0f) && (fUv.y >= 0.0f && fUv.y <= 1.0f);
}

FfxInt32x2 ClampLoad(FfxInt32x2 iPxSample, FfxInt32x2 iPxOffset, FfxInt32x2 iTextureSize)
{
    FfxInt32x2 result = iPxSample + iPxOffset;
    result.x = ffxMax(0, ffxMin(result.x, iTextureSize.x - 1));
    result.y = ffxMax(0, ffxMin(result.y, iTextureSize.y - 1));
    return result;
}

FfxFloat32x2 ClampUv(FfxFloat32x2 fUv, FfxInt32x2 iTextureSize, FfxInt32x2 iResourceSize)
{
    const FfxFloat32x2 fSampleLocation = fUv * iTextureSize;
    const FfxFloat32x2 fClampedLocation = ffxMax(FfxFloat32x2(0.5f, 0.5f), ffxMin(fSampleLocation, FfxFloat32x2(iTextureSize) - FfxFloat32x2(0.5f, 0.5f)));
    const FfxFloat32x2 fClampedUv = fClampedLocation / FfxFloat32x2(iResourceSize);

    return fClampedUv;
}

FfxBoolean IsOnScreen(FfxInt32x2 pos, FfxInt32x2 size)
{
    return all(FFX_LESS_THAN(FfxUInt32x2(pos), FfxUInt32x2(size)));
}

FfxFloat32 ComputeAutoExposureFromLavg(FfxFloat32 Lavg)
{
    Lavg = exp(Lavg);

    const FfxFloat32 S = 100.0f; //ISO arithmetic speed
    const FfxFloat32 K = 12.5f;
    FfxFloat32 ExposureISO100 = log2((Lavg * S) / K);

    const FfxFloat32 q = 0.65f;
    FfxFloat32 Lmax = (78.0f / (q * S)) * ffxPow(2.0f, ExposureISO100);

    return 1.0f / Lmax;
}

FfxInt32x2 ComputeHrPosFromLrPos(FfxInt32x2 iPxLrPos)
{
    FfxFloat32x2 fSrcJitteredPos = FfxFloat32x2(iPxLrPos) + 0.5f - Jitter();
    FfxFloat32x2 fLrPosInHr = (fSrcJitteredPos / RenderSize()) * UpscaleSize();
    FfxInt32x2 iPxHrPos = FfxInt32x2(floor(fLrPosInHr));
    return iPxHrPos;
}

FfxFloat32x2 ComputeNdc(FfxFloat32x2 fPxPos, FfxInt32x2 iSize)
{
    return fPxPos / FfxFloat32x2(iSize) * FfxFloat32x2(2.0f, -2.0f) + FfxFloat32x2(-1.0f, 1.0f);
}

FfxFloat32 GetViewSpaceDepth(FfxFloat32 fDeviceDepth)
{
    const FfxFloat32x4 fDeviceToViewDepth = DeviceToViewSpaceTransformFactors();

    // fDeviceToViewDepth details found in ffx_fsr3upscaler.cpp
    return (fDeviceToViewDepth[1] / (fDeviceDepth - fDeviceToViewDepth[0]));
}

FfxFloat32 GetViewSpaceDepthInMeters(FfxFloat32 fDeviceDepth)
{
    return GetViewSpaceDepth(fDeviceDepth) * ViewSpaceToMetersFactor();
}

FfxFloat32x3 GetViewSpacePosition(FfxInt32x2 iViewportPos, FfxInt32x2 iViewportSize, FfxFloat32 fDeviceDepth)
{
    const FfxFloat32x4 fDeviceToViewDepth = DeviceToViewSpaceTransformFactors();

    const FfxFloat32 Z = GetViewSpaceDepth(fDeviceDepth);

    const FfxFloat32x2 fNdcPos = ComputeNdc(iViewportPos, iViewportSize);
    const FfxFloat32 X = fDeviceToViewDepth[2] * fNdcPos.x * Z;
    const FfxFloat32 Y = fDeviceToViewDepth[3] * fNdcPos.y * Z;

    return FfxFloat32x3(X, Y, Z);
}

FfxFloat32x3 GetViewSpacePositionInMeters(FfxInt32x2 iViewportPos, FfxInt32x2 iViewportSize, FfxFloat32 fDeviceDepth)
{
    return GetViewSpacePosition(iViewportPos, iViewportSize, fDeviceDepth) * ViewSpaceToMetersFactor();
}

FfxFloat32 GetMaxDistanceInMeters()
{
#if FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH
    return GetViewSpaceDepth(0.0f) * ViewSpaceToMetersFactor();
#else
    return GetViewSpaceDepth(1.0f) * ViewSpaceToMetersFactor();
#endif
}


struct BilinearSamplingData
{
    FfxInt32x2 iOffsets[4];
    FfxFloat32 fWeights[4];
    FfxInt32x2 iBasePos;
};

BilinearSamplingData GetBilinearSamplingData(FfxFloat32x2 fUv, FfxInt32x2 iSize)
{
    BilinearSamplingData data;

    FfxFloat32x2 fPxSample = (fUv * iSize) - FfxFloat32x2(0.5f, 0.5f);
    data.iBasePos = FfxInt32x2(floor(fPxSample));
    FfxFloat32x2 fPxFrac = ffxFract(fPxSample);

    data.iOffsets[0] = FfxInt32x2(0, 0);
    data.iOffsets[1] = FfxInt32x2(1, 0);
    data.iOffsets[2] = FfxInt32x2(0, 1);
    data.iOffsets[3] = FfxInt32x2(1, 1);

    data.fWeights[0] = (1 - fPxFrac.x) * (1 - fPxFrac.y);
    data.fWeights[1] = (fPxFrac.x) * (1 - fPxFrac.y);
    data.fWeights[2] = (1 - fPxFrac.x) * (fPxFrac.y);
    data.fWeights[3] = (fPxFrac.x) * (fPxFrac.y);

    return data;
}

struct PlaneData
{
    FfxFloat32x3 fNormal;
    FfxFloat32 fDistanceFromOrigin;
};

PlaneData GetPlaneFromPoints(FfxFloat32x3 fP0, FfxFloat32x3 fP1, FfxFloat32x3 fP2)
{
    PlaneData plane;

    FfxFloat32x3 v0 = fP0 - fP1;
    FfxFloat32x3 v1 = fP0 - fP2;
    plane.fNormal = normalize(cross(v0, v1));
    plane.fDistanceFromOrigin = -dot(fP0, plane.fNormal);

    return plane;
}

FfxFloat32 PointToPlaneDistance(PlaneData plane, FfxFloat32x3 fPoint)
{
    return abs(dot(plane.fNormal, fPoint) + plane.fDistanceFromOrigin);
}

#endif // #if defined(FFX_GPU)

#endif //!defined(FFX_FSR3UPSCALER_COMMON_H)
