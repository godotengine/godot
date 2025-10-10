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

#if !defined(FFX_FRAMEINTERPOLATION_COMMON_H)
#define FFX_FRAMEINTERPOLATION_COMMON_H

#define FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_TEAR_LINES       (1 << 0)
#define FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_RESET_INDICATORS (1 << 1)
#define FFX_FRAMEINTERPOLATION_DISPATCH_DRAW_DEBUG_VIEW             (1 << 2)

FFX_STATIC const FfxFloat32 FFX_FRAMEINTERPOLATION_EPSILON = 1e-03f;
FFX_STATIC const FfxFloat32 FFX_FRAMEINTERPOLATION_FLT_MAX = 3.402823466e+38f;
FFX_STATIC const FfxFloat32 FFX_FRAMEINTERPOLATION_FLT_MIN = 1.175494351e-38f;

FFX_STATIC const FfxFloat32 fReconstructedDepthBilinearWeightThreshold = FFX_FRAMEINTERPOLATION_EPSILON;

FfxFloat32 RGBToLuma(FfxFloat32x3 fLinearRgb)
{
    return dot(fLinearRgb, FfxFloat32x3(0.2126f, 0.7152f, 0.0722f));
}

FfxFloat32 LinearRec2020ToLuminance(FfxFloat32x3 linearRec2020RGB)
{
    FfxFloat32 fY = 0.2627 * linearRec2020RGB.x + 0.678 * linearRec2020RGB.y + 0.0593 * linearRec2020RGB.z;
    return fY;
}

FfxFloat32x3 ffxscRGBToLinear(FfxFloat32x3 value, FfxFloat32 minLuminance, FfxFloat32 maxLuminance)
{
    FfxFloat32x3 p = value - ffxBroadcast3(minLuminance / 80.0f);
    return p / ffxBroadcast3((maxLuminance - minLuminance) / 80.0f);
}

FfxFloat32x3 RawRGBToLinear(FfxFloat32x3 fRawRgb)
{
    FfxFloat32x3 fLinearRgb;

    switch (BackBufferTransferFunction())
    {
    case 0:
        fLinearRgb = ffxLinearFromSrgb(fRawRgb);
        break;
    case 1:
        fLinearRgb = ffxLinearFromPQ(fRawRgb) * (10000.0f / MaxLuminance());
        break;
    case 2:
        fLinearRgb = ffxscRGBToLinear(fRawRgb, MinLuminance(), MaxLuminance());
        break;
    }

    return fLinearRgb;
}

FfxFloat32 RawRGBToLuminance(FfxFloat32x3 fRawRgb)
{
    FfxFloat32 fLuminance = 0.0f;

    switch (BackBufferTransferFunction())
    {
    case 0:
        fLuminance = RGBToLuma(RawRGBToLinear(fRawRgb));
        break;
    case 1:
        fLuminance = LinearRec2020ToLuminance(RawRGBToLinear(fRawRgb));
        break;
    case 2:
        fLuminance = RGBToLuma(RawRGBToLinear(fRawRgb));
        break;
    }

    return fLuminance;
}

FfxFloat32 RawRGBToPerceivedLuma(FfxFloat32x3 fRawRgb)
{
    FfxFloat32 fLuminance = RawRGBToLuminance(fRawRgb);

    FfxFloat32 fPercievedLuminance = 0;
    if (fLuminance <= 216.0f / 24389.0f)
    {
        fPercievedLuminance = fLuminance * (24389.0f / 27.0f);
    }
    else
    {
        fPercievedLuminance = ffxPow(fLuminance, 1.0f / 3.0f) * 116.0f - 16.0f;
    }

    return fPercievedLuminance * 0.01f;
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
    data.iBasePos          = FfxInt32x2(floor(fPxSample));
    FfxFloat32x2 fPxFrac   = ffxFract(fPxSample);

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

#if defined(FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)
FfxFloat32 ConvertFromDeviceDepthToViewSpace(FfxFloat32 fDeviceDepth)
{
    const FfxFloat32x4 deviceToViewDepth = DeviceToViewSpaceTransformFactors();
    return deviceToViewDepth[1] / (fDeviceDepth - deviceToViewDepth[0]);
}

FfxFloat32x2 ComputeNdc(FfxFloat32x2 fPxPos, FfxInt32x2 iSize)
{
    return fPxPos / FfxFloat32x2(iSize) * FfxFloat32x2(2.0f, -2.0f) + FfxFloat32x2(-1.0f, 1.0f);
}

FfxFloat32x3 GetViewSpacePosition(FfxInt32x2 iViewportPos, FfxInt32x2 iViewportSize, FfxFloat32 fDeviceDepth)
{
    const FfxFloat32x4 fDeviceToViewDepth = DeviceToViewSpaceTransformFactors();

    const FfxFloat32 Z = ConvertFromDeviceDepthToViewSpace(fDeviceDepth);

    const FfxFloat32x2 fNdcPos = ComputeNdc(iViewportPos, iViewportSize);
    const FfxFloat32   X       = fDeviceToViewDepth[2] * fNdcPos.x * Z;
    const FfxFloat32   Y       = fDeviceToViewDepth[3] * fNdcPos.y * Z;

    return FfxFloat32x3(X, Y, Z);
}
#endif

FfxBoolean IsOnScreen(FfxInt32x2 pos, FfxInt32x2 size)
{
    return all(FFX_LESS_THAN(FfxUInt32x2(pos), FfxUInt32x2(size)));
}

FfxBoolean IsUvInside(FfxFloat32x2 fUv)
{
    return (fUv.x > 0.0f && fUv.x < 1.0f) && (fUv.y > 0.0f && fUv.y < 1.0f);
}

FfxBoolean IsInRect(FfxInt32x2 pos, FfxInt32x2 iRectCorner, FfxInt32x2 iRectSize)
{
    return (pos.x >= iRectCorner.x && pos.x < (iRectSize.x + iRectCorner.x) && pos.y >= iRectCorner.y && pos.y < (iRectSize.y + iRectCorner.y));
}

FfxFloat32 MinDividedByMax(const FfxFloat32 v0, const FfxFloat32 v1)
{
    const FfxFloat32 m = ffxMax(v0, v1);
    return m != 0 ? ffxMin(v0, v1) / m : 0;
}

FfxFloat32 NormalizedDot3(const FfxFloat32x3 v0, const FfxFloat32x3 v1)
{
    FfxFloat32 fMaxLength = ffxMax(length(v0), length(v1));

    return fMaxLength > 0.0f ? dot(v0 / fMaxLength, v1 / fMaxLength) : 1.0f;
}

FfxFloat32 NormalizedDot2(const FfxFloat32x2 v0, const FfxFloat32x2 v1)
{
    FfxFloat32 fMaxLength = ffxMax(length(v0), length(v1));

    return fMaxLength > 0.0f ? dot(v0 / fMaxLength, v1 / fMaxLength) : 1.0f;
}

FfxFloat32 CalculateStaticContentFactor(FfxFloat32x3 fCurrentInterpolationSource, FfxFloat32x3 fPresentColor)
{
    const FfxFloat32x3 fFactor = ffxSaturate(FfxFloat32x3(
        ffxSaturate((1.0f - MinDividedByMax(fCurrentInterpolationSource.r, fPresentColor.r)) / 0.1f),
        ffxSaturate((1.0f - MinDividedByMax(fCurrentInterpolationSource.g, fPresentColor.g)) / 0.1f),
        ffxSaturate((1.0f - MinDividedByMax(fCurrentInterpolationSource.b, fPresentColor.b)) / 0.1f)
    ));

    return max(fFactor.x, max(fFactor.y, fFactor.z));
}

// 
// MOTION VECTOR FIELD
// 

FFX_STATIC const FfxUInt32 MOTION_VECTOR_FIELD_ENTRY_BIT_COUNT = 32;

// Make sure all bit counts add up to MOTION_VECTOR_FIELD_ENTRY_BIT_COUNT
FFX_STATIC const FfxUInt32 MOTION_VECTOR_FIELD_VECTOR_COEFFICIENT_BIT_COUNT = 16;
FFX_STATIC const FfxUInt32 MOTION_VECTOR_FIELD_PRIORITY_LOW_BIT_COUNT = 5;
FFX_STATIC const FfxUInt32 MOTION_VECTOR_FIELD_PRIORITY_HIGH_BIT_COUNT = 10;
FFX_STATIC const FfxUInt32 MOTION_VECTOR_PRIMARY_VECTOR_INDICATION_BIT_COUNT = 1;

FFX_STATIC const FfxUInt32 MOTION_VECTOR_FIELD_PRIMARY_VECTOR_INDICATION_BIT = (1U << (MOTION_VECTOR_FIELD_ENTRY_BIT_COUNT - 1));

FFX_STATIC const FfxUInt32 PRIORITY_LOW_MAX = (1U << MOTION_VECTOR_FIELD_PRIORITY_LOW_BIT_COUNT) - 1;
FFX_STATIC const FfxUInt32 PRIORITY_HIGH_MAX = (1U << MOTION_VECTOR_FIELD_PRIORITY_HIGH_BIT_COUNT) - 1;

FFX_STATIC const FfxUInt32 PRIORITY_LOW_OFFSET = MOTION_VECTOR_FIELD_VECTOR_COEFFICIENT_BIT_COUNT;
FFX_STATIC const FfxUInt32 PRIORITY_HIGH_OFFSET = PRIORITY_LOW_OFFSET + MOTION_VECTOR_FIELD_PRIORITY_LOW_BIT_COUNT;
FFX_STATIC const FfxUInt32 PRIMARY_VECTOR_INDICATION_OFFSET = PRIORITY_HIGH_OFFSET + MOTION_VECTOR_FIELD_PRIORITY_HIGH_BIT_COUNT;

struct VectorFieldEntry
{
    FfxFloat32x2 fMotionVector;
    FfxFloat32   uHighPriorityFactor;
    FfxFloat32   uLowPriorityFactor;
    FfxBoolean   bValid;
    FfxBoolean   bPrimary;
    FfxBoolean   bSecondary;
    FfxBoolean   bInPainted;
    FfxFloat32   fVelocity;
    FfxBoolean   bNegOutside;
    FfxBoolean   bPosOutside;
};

VectorFieldEntry NewVectorFieldEntry()
{
    VectorFieldEntry vfe;
    vfe.fMotionVector = FfxFloat32x2(0.0, 0.0);
    vfe.uHighPriorityFactor = 0.0;
    vfe.uLowPriorityFactor = 0.0;
    vfe.bValid = false;
    vfe.bPrimary = false;
    vfe.bSecondary = false;
    vfe.bInPainted = false;
    vfe.fVelocity = 0.0;
    vfe.bNegOutside = false;
    vfe.bPosOutside = false;
    return vfe;
}

FfxBoolean PackedVectorFieldEntryIsPrimary(FfxUInt32 packedEntry)
{
    return ((packedEntry & MOTION_VECTOR_FIELD_PRIMARY_VECTOR_INDICATION_BIT) != 0);
}

FfxUInt32x2 PackVectorFieldEntries(FfxBoolean bIsPrimary, FfxUInt32 uHighPriorityFactor, FfxUInt32 uLowPriorityFactor, FfxFloat32x2 fMotionVector)
{
    const FfxUInt32 uPriority =
        (FfxUInt32(bIsPrimary) * MOTION_VECTOR_FIELD_PRIMARY_VECTOR_INDICATION_BIT)
        | ((uHighPriorityFactor & PRIORITY_HIGH_MAX) << PRIORITY_HIGH_OFFSET)
        | ((uLowPriorityFactor & PRIORITY_LOW_MAX) << PRIORITY_LOW_OFFSET);

    FfxUInt32 packedX = uPriority | ffxF32ToF16(fMotionVector.x);
    FfxUInt32 packedY = uPriority | ffxF32ToF16(fMotionVector.y);

    return FfxUInt32x2(packedX, packedY);
}

void UnpackVectorFieldEntries(FfxUInt32x2 packed, out VectorFieldEntry vfElement)
{
    vfElement.uHighPriorityFactor = FfxFloat32((packed.x >> PRIORITY_HIGH_OFFSET) & PRIORITY_HIGH_MAX) / PRIORITY_HIGH_MAX;
    vfElement.uLowPriorityFactor = FfxFloat32((packed.x >> PRIORITY_LOW_OFFSET) & PRIORITY_LOW_MAX) / PRIORITY_LOW_MAX;

    vfElement.bPrimary = PackedVectorFieldEntryIsPrimary(packed.x);
    vfElement.bValid = (vfElement.uHighPriorityFactor > 0.0f);
    vfElement.bSecondary = vfElement.bValid && !vfElement.bPrimary;

    // Reverse priority factor for secondary vectors
    if (vfElement.bSecondary)
    {
        vfElement.uHighPriorityFactor = 1.0f - vfElement.uHighPriorityFactor;
    }

    vfElement.fMotionVector.x = ffxUnpackF32(packed.x).x;
    vfElement.fMotionVector.y = ffxUnpackF32(packed.y).x;
    vfElement.bInPainted      = false;
}

// 
// MOTION VECTOR FIELD
// 

#if defined(FFX_FRAMEINTERPOLATION_BIND_SRV_INPAINTING_PYRAMID)
FfxFloat32x4 ComputeMvInpaintingLevel(FfxFloat32x2 fUv, const FfxInt32 iMipLevel, const FfxInt32x2 iTexSize)
{
    BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fUv, iTexSize);

    FfxFloat32   fSum   = 0.0f;
    FfxFloat32x4 fColor = FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
    fColor.z            = 0;

    const FfxFloat32 fMaxPriorityFactor = 1.0f;

    for (FfxInt32 iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++)
    {
        const FfxInt32x2 iOffset    = bilinearInfo.iOffsets[iSampleIndex];
        const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;

        if (IsOnScreen(iSamplePos, iTexSize))
        {
            FfxFloat32x4 fSample = LoadInpaintingPyramid(iMipLevel, iSamplePos);

            const FfxFloat32 fPriorityFactor = fSample.z;
            const FfxFloat32 fValidMvFactor  = FfxFloat32(fSample.z > 0);
            const FfxFloat32 fSampleWeight   = bilinearInfo.fWeights[iSampleIndex] * fValidMvFactor * fPriorityFactor;

            fSum += fSampleWeight;
            fColor += fSample * fSampleWeight;
        }
    }

    fColor /= (fSum > 0.0f) ? fSum : 1.0f;

    return fColor;
}
#if defined(FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION) && \
    defined(FFX_FRAMEINTERPOLATION_BIND_SRV_GAME_MOTION_VECTOR_FIELD_X) && \
    defined(FFX_FRAMEINTERPOLATION_BIND_SRV_GAME_MOTION_VECTOR_FIELD_Y)

void LoadInpaintedGameFieldMv(FfxFloat32x2 fUv, out VectorFieldEntry vfElement)
{    
    FfxInt32x2 iPxSample = FfxInt32x2(fUv * RenderSize());
    FfxUInt32x2 packedGameFieldMv = LoadGameFieldMv(iPxSample);
    UnpackVectorFieldEntries(packedGameFieldMv, vfElement);

    if (!vfElement.bValid)
    {
        //FfxFloat32x2 fUv = (FfxFloat32x2(iPxSample) + 0.5f) / RenderSize();
        FfxInt32x2 iTexSize = RenderSize();

        FfxFloat32x4 fInPaintedVector = FfxFloat32x4(0.0, 0.0, 0.0, 0.0);
        for (FfxInt32 iMipLevel = 0; iMipLevel < 11 && (fInPaintedVector.w == 0.0f); iMipLevel++)
        {
            iTexSize /= 2;

            fInPaintedVector = ComputeMvInpaintingLevel(fUv, iMipLevel, iTexSize);
        }

        vfElement.fMotionVector         = fInPaintedVector.xy;
        vfElement.uHighPriorityFactor   = fInPaintedVector.z;
        vfElement.uLowPriorityFactor    = fInPaintedVector.w;
        vfElement.bInPainted            = true;
    }

    vfElement.bNegOutside = !IsUvInside(fUv - vfElement.fMotionVector);
    vfElement.bPosOutside = !IsUvInside(fUv + vfElement.fMotionVector);
    vfElement.fVelocity   = length(vfElement.fMotionVector);
}
#endif
#endif

#if defined(FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_X) && \
    defined(FFX_FRAMEINTERPOLATION_BIND_SRV_OPTICAL_FLOW_MOTION_VECTOR_FIELD_Y) && \
    defined(FFX_FRAMEINTERPOLATION_BIND_CB_FRAMEINTERPOLATION)
void SampleOpticalFlowMotionVectorField(FfxFloat32x2 fUv, out VectorFieldEntry vfElement)
{
    const FfxFloat32 scaleFactor = 1.0f;

    BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fUv, FfxInt32x2(GetOpticalFlowSize2() * scaleFactor));

    vfElement = NewVectorFieldEntry();

    FfxFloat32 fWeightSum = 0.0f;
    for (FfxInt32 iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++)
    {
        const FfxInt32x2 iOffset    = bilinearInfo.iOffsets[iSampleIndex];
        const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;

        if (IsOnScreen(iSamplePos, FfxInt32x2(GetOpticalFlowSize2() * scaleFactor)))
        {
            const FfxFloat32 fWeight = bilinearInfo.fWeights[iSampleIndex];

            VectorFieldEntry fOfVectorSample = NewVectorFieldEntry();
            FfxInt32x2 packedOpticalFlowMv = FfxInt32x2(LoadOpticalFlowFieldMv(iSamplePos));
            UnpackVectorFieldEntries(packedOpticalFlowMv, fOfVectorSample);

            vfElement.fMotionVector += fOfVectorSample.fMotionVector * fWeight;
            vfElement.uHighPriorityFactor += fOfVectorSample.uHighPriorityFactor * fWeight;
            vfElement.uLowPriorityFactor += fOfVectorSample.uLowPriorityFactor * fWeight;

            fWeightSum += fWeight;
        }
    }

    if (fWeightSum > 0.0f)
    {
        vfElement.fMotionVector /= fWeightSum;
        vfElement.uHighPriorityFactor /= fWeightSum;
        vfElement.uLowPriorityFactor /= fWeightSum;
    }

    vfElement.bNegOutside = !IsUvInside(fUv - vfElement.fMotionVector);
    vfElement.bPosOutside = !IsUvInside(fUv + vfElement.fMotionVector);
    vfElement.fVelocity   = length(vfElement.fMotionVector);
}
#endif

FfxFloat32x3 Tonemap(FfxFloat32x3 fRgb)
{
    return fRgb / (ffxMax(ffxMax(0.f, fRgb.r), ffxMax(fRgb.g, fRgb.b)) + 1.f).xxx;
}

FfxFloat32x3 InverseTonemap(FfxFloat32x3 fRgb)
{
    return fRgb / ffxMax(FFX_TONEMAP_EPSILON, 1.f - ffxMax(fRgb.r, ffxMax(fRgb.g, fRgb.b))).xxx;
}

FfxInt32x2 ComputeHrPosFromLrPos(FfxInt32x2 iPxLrPos)
{
    FfxFloat32x2 fSrcJitteredPos = FfxFloat32x2(iPxLrPos) + 0.5f - Jitter();
    FfxFloat32x2 fLrPosInHr      = (fSrcJitteredPos / RenderSize()) * DisplaySize();
    FfxInt32x2   iPxHrPos        = FfxInt32x2(floor(fLrPosInHr));
    return iPxHrPos;
}
#if FFX_HALF
FFX_MIN16_I2 ComputeHrPosFromLrPos(FFX_MIN16_I2 iPxLrPos)
{
    FFX_MIN16_F2 fSrcJitteredPos = FFX_MIN16_F2(iPxLrPos) + FFX_MIN16_F(0.5f) - FFX_MIN16_F2(Jitter());
    FFX_MIN16_F2 fLrPosInHr      = (fSrcJitteredPos / FFX_MIN16_F2(RenderSize())) * FFX_MIN16_F2(DisplaySize());
    FFX_MIN16_I2 iPxHrPos        = FFX_MIN16_I2(floor(fLrPosInHr));
    return iPxHrPos;
}
#endif

#endif //!defined(FFX_FRAMEINTERPOLATION_COMMON_H)
