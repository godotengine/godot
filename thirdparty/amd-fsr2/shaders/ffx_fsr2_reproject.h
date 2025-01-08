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

#ifndef FFX_FSR2_REPROJECT_H
#define FFX_FSR2_REPROJECT_H

#ifndef FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE
#define FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE 0 // Reference
#endif

FfxFloat32x4 WrapHistory(FfxInt32x2 iPxSample)
{
    return LoadHistory(iPxSample);
}

#if FFX_HALF
FFX_MIN16_F4 WrapHistory(FFX_MIN16_I2 iPxSample)
{
    return FFX_MIN16_F4(LoadHistory(iPxSample));
}
#endif


#if FFX_FSR2_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF && FFX_HALF
DeclareCustomFetchBicubicSamplesMin16(FetchHistorySamples, WrapHistory)
DeclareCustomTextureSampleMin16(HistorySample, FFX_FSR2_GET_LANCZOS_SAMPLER1D(FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE), FetchHistorySamples)
#else
DeclareCustomFetchBicubicSamples(FetchHistorySamples, WrapHistory)
DeclareCustomTextureSample(HistorySample, FFX_FSR2_GET_LANCZOS_SAMPLER1D(FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE), FetchHistorySamples)
#endif

FfxFloat32x4 WrapLockStatus(FfxInt32x2 iPxSample)
{
    FfxFloat32x4 fSample = FfxFloat32x4(LoadLockStatus(iPxSample), 0.0f, 0.0f);
    return fSample;
}

#if FFX_HALF
FFX_MIN16_F4 WrapLockStatus(FFX_MIN16_I2 iPxSample)
{
    FFX_MIN16_F4 fSample = FFX_MIN16_F4(LoadLockStatus(iPxSample), 0.0, 0.0);

    return fSample;
}
#endif

#if 1
#if FFX_FSR2_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF && FFX_HALF
DeclareCustomFetchBilinearSamplesMin16(FetchLockStatusSamples, WrapLockStatus)
DeclareCustomTextureSampleMin16(LockStatusSample, Bilinear, FetchLockStatusSamples)
#else
DeclareCustomFetchBilinearSamples(FetchLockStatusSamples, WrapLockStatus)
DeclareCustomTextureSample(LockStatusSample, Bilinear, FetchLockStatusSamples)
#endif
#else
#if FFX_FSR2_OPTION_REPROJECT_SAMPLERS_USE_DATA_HALF && FFX_HALF
DeclareCustomFetchBicubicSamplesMin16(FetchLockStatusSamples, WrapLockStatus)
DeclareCustomTextureSampleMin16(LockStatusSample, FFX_FSR2_GET_LANCZOS_SAMPLER1D(FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE), FetchLockStatusSamples)
#else
DeclareCustomFetchBicubicSamples(FetchLockStatusSamples, WrapLockStatus)
DeclareCustomTextureSample(LockStatusSample, FFX_FSR2_GET_LANCZOS_SAMPLER1D(FFX_FSR2_OPTION_REPROJECT_USE_LANCZOS_TYPE), FetchLockStatusSamples)
#endif
#endif

FfxFloat32x2 GetMotionVector(FfxInt32x2 iPxHrPos, FfxFloat32x2 fHrUv)
{
#if FFX_FSR2_OPTION_LOW_RESOLUTION_MOTION_VECTORS
    FfxFloat32x2 fDilatedMotionVector = LoadDilatedMotionVector(FFX_MIN16_I2(fHrUv * RenderSize()));
#else
    FfxFloat32x2 fDilatedMotionVector = LoadInputMotionVector(iPxHrPos);
#endif

    return fDilatedMotionVector;
}

FfxBoolean IsUvInside(FfxFloat32x2 fUv)
{
    return (fUv.x >= 0.0f && fUv.x <= 1.0f) && (fUv.y >= 0.0f && fUv.y <= 1.0f);
}

void ComputeReprojectedUVs(const AccumulationPassCommonParams params, FFX_PARAMETER_OUT FfxFloat32x2 fReprojectedHrUv, FFX_PARAMETER_OUT FfxBoolean bIsExistingSample)
{
    fReprojectedHrUv = params.fHrUv + params.fMotionVector;

    bIsExistingSample = IsUvInside(fReprojectedHrUv);
}

void ReprojectHistoryColor(const AccumulationPassCommonParams params, FFX_PARAMETER_OUT FfxFloat32x3 fHistoryColor, FFX_PARAMETER_OUT FfxFloat32 fTemporalReactiveFactor, FFX_PARAMETER_OUT FfxBoolean bInMotionLastFrame)
{
    FfxFloat32x4 fHistory = HistorySample(params.fReprojectedHrUv, DisplaySize());

    fHistoryColor = PrepareRgb(fHistory.rgb, Exposure(), PreviousFramePreExposure());

    fHistoryColor = RGBToYCoCg(fHistoryColor);

    //Compute temporal reactivity info
    fTemporalReactiveFactor = ffxSaturate(abs(fHistory.w));
    bInMotionLastFrame = (fHistory.w < 0.0f);
}

LockState ReprojectHistoryLockStatus(const AccumulationPassCommonParams params, FFX_PARAMETER_OUT FfxFloat32x2 fReprojectedLockStatus)
{
    LockState state = { FFX_FALSE, FFX_FALSE };
    const FfxFloat32 fNewLockIntensity = LoadRwNewLocks(params.iPxHrPos);
    state.NewLock = fNewLockIntensity > (127.0f / 255.0f);

    FfxFloat32 fInPlaceLockLifetime = state.NewLock ? fNewLockIntensity : 0;

    fReprojectedLockStatus = SampleLockStatus(params.fReprojectedHrUv);

    if (fReprojectedLockStatus[LOCK_LIFETIME_REMAINING] != FfxFloat32(0.0f)) {
        state.WasLockedPrevFrame = true;
    }

    return state;
}

#endif //!defined( FFX_FSR2_REPROJECT_H )
