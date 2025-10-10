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

#ifndef FFX_FRAMEINTERPOLATION_OPTICAL_FLOW_VECTOR_FIELD_H
#define FFX_FRAMEINTERPOLATION_OPTICAL_FLOW_VECTOR_FIELD_H

void computeOpticalFlowFieldMvs(FfxUInt32x2 dtID, FfxFloat32x2 fOpticalFlowVector)
{
    FfxFloat32x2 fUv = FfxFloat32x2(FfxFloat32x2(dtID)+0.5f) / GetOpticalFlowSize2();

    const FfxFloat32 scaleFactor = 1.0f;
    FfxFloat32x2 fMotionVectorHalf = fOpticalFlowVector * 0.5f;

    // pixel position in current frame + fOpticalFlowVector-> pixel position in previous frame
    FfxFloat32x3 prevBackbufferCol = SamplePreviousBackbuffer(fUv + fOpticalFlowVector).xyz; // returns previous backbuffer color of current frame pixel position in previous frame
    FfxFloat32x3 curBackbufferCol  = SampleCurrentBackbuffer(fUv).xyz; // returns current backbuffer color at current frame pixel position

    FfxFloat32 prevLuma = 0.001f + RawRGBToLuminance(prevBackbufferCol);
    FfxFloat32 currLuma = 0.001f + RawRGBToLuminance(curBackbufferCol);

    FfxFloat32 fVelocity = length(fOpticalFlowVector * InterpolationRectSize());
    FfxUInt32  uHighPriorityFactor = FfxUInt32(fVelocity > 1.0f) * FfxUInt32(ffxSaturate(fVelocity / length(InterpolationRectSize() * 0.05f)) * PRIORITY_HIGH_MAX);

    if(uHighPriorityFactor > 0) {
        FfxUInt32 uLowPriorityFactor = FfxUInt32(ffxRound(ffxPow(MinDividedByMax(prevLuma, currLuma), 1.0f / 1.0f) * PRIORITY_LOW_MAX))
            * FfxUInt32(IsUvInside(fUv + fOpticalFlowVector));

        // Project current depth into previous frame locations.
        // Push to all pixels having some contribution if reprojection is using bilinear logic.

        const FfxUInt32x2 packedVectorPrimary = PackVectorFieldEntries(true, uHighPriorityFactor, uLowPriorityFactor, fMotionVectorHalf);

        BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fUv + fMotionVectorHalf, GetOpticalFlowSize2());
        for (FfxInt32 iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++)
        {
            const FfxInt32x2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
            const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;

            if (IsOnScreen(iSamplePos, GetOpticalFlowSize2()))
            {
                UpdateOpticalflowMotionVectorField(iSamplePos, packedVectorPrimary);
            }
        }
    }
}

void computeOpticalFlowVectorField(FfxInt32x2 iPxPos)
{
    FfxFloat32x2 fOpticalFlowVector = FfxFloat32x2(0.0, 0.0);
    FfxFloat32x2 fOpticalFlowVector3x3Avg = FfxFloat32x2(0.0, 0.0);
    FfxInt32 size = 1;
    FfxFloat32 sw = 0.0f;

    for(FfxInt32 y = -size; y <= size; y++) {
        for(FfxInt32 x = -size; x <= size; x++) {

            FfxInt32x2 samplePos = iPxPos + FfxInt32x2(x, y);

            FfxFloat32x2 vs = LoadOpticalFlow(samplePos);
            FfxFloat32   fConfidenceFactor = ffxMax(FFX_FRAMEINTERPOLATION_EPSILON, LoadOpticalFlowConfidence(samplePos));


            FfxFloat32 len        = length(vs * InterpolationRectSize());
            FfxFloat32 len_factor = ffxMax(0.0f, 512.0f - len) * FfxFloat32(len > 1.0f);
            FfxFloat32 w = len_factor;

            fOpticalFlowVector3x3Avg += vs * w;

            sw += w;
        }
    }

    fOpticalFlowVector3x3Avg /= sw;


    sw = 0.0f;
    for(FfxInt32 y = -size; y <= size; y++) {
        for(FfxInt32 x = -size; x <= size; x++) {

            FfxInt32x2 samplePos = iPxPos + FfxInt32x2(x, y);

            FfxFloat32x2 vs = LoadOpticalFlow(samplePos);

            FfxFloat32 fConfidenceFactor = ffxMax(FFX_FRAMEINTERPOLATION_EPSILON, LoadOpticalFlowConfidence(samplePos));
            FfxFloat32 len               = length(vs * InterpolationRectSize());
            FfxFloat32 len_factor        = ffxMax(0.0f, 512.0f - len) * FfxFloat32(len > 1.0f);


            FfxFloat32 w = ffxMax(0.0f, ffxPow(dot(fOpticalFlowVector3x3Avg, vs), 1.25f)) * len_factor;

            fOpticalFlowVector += vs * w;
            sw += w;
        }
    }

    if (sw > FFX_FRAMEINTERPOLATION_EPSILON)
    {
        fOpticalFlowVector /= sw;
    }

    computeOpticalFlowFieldMvs(iPxPos, fOpticalFlowVector);
}

#endif // FFX_FRAMEINTERPOLATION_OPTICAL_FLOW_VECTOR_FIELD_H
