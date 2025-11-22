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

#ifndef FFX_FRAMEINTERPOLATION_GAME_MOTION_VECTOR_FIELD_H
#define FFX_FRAMEINTERPOLATION_GAME_MOTION_VECTOR_FIELD_H

FfxUInt32 getPriorityFactorFromViewSpaceDepth(FfxFloat32 fViewSpaceDepthInMeters)
{
    fViewSpaceDepthInMeters = ffxPow(fViewSpaceDepthInMeters, 0.33f);

    FfxUInt32 uPriorityFactor = FfxUInt32(FfxFloat32(1 - (fViewSpaceDepthInMeters * (1.0f / (1.0f + fViewSpaceDepthInMeters)))) * PRIORITY_HIGH_MAX);

    return ffxMax(1, uPriorityFactor);
}

void computeGameFieldMvs(FfxInt32x2 iPxPos)
{
    const FfxFloat32x2 fUvInScreenSpace            = (FfxFloat32x2(iPxPos) + 0.5f) / RenderSize();

    const FfxFloat32x2 fDistortionFieldUv          = SampleDistortionField(fUvInScreenSpace);
    FfxInt32x2 iDistortionPixelOffset              = FfxInt32x2(fDistortionFieldUv.xy * RenderSize());

    const FfxFloat32x2 fUvInInterpolationRectStart = FfxFloat32x2(InterpolationRectBase()) / DisplaySize();
    const FfxFloat32x2 fUvLetterBoxScale           = FfxFloat32x2(InterpolationRectSize()) / DisplaySize();
    const FfxFloat32x2 fUvInInterpolationRect      = fUvInInterpolationRectStart + fUvInScreenSpace * fUvLetterBoxScale;

    const FfxFloat32 fDepthSample = LoadDilatedDepth(iPxPos + iDistortionPixelOffset);
    const FfxFloat32x2 fGameMotionVector = LoadDilatedMotionVector(iPxPos + iDistortionPixelOffset);
    const FfxFloat32x2 fMotionVectorHalf = fGameMotionVector * 0.5f;
    const FfxFloat32x2 fInterpolatedLocationUv = fUvInScreenSpace + fMotionVectorHalf;

    const FfxFloat32 fViewSpaceDepth = ConvertFromDeviceDepthToViewSpace(fDepthSample);
    const FfxUInt32 uHighPriorityFactorPrimary = getPriorityFactorFromViewSpaceDepth(fViewSpaceDepth);

    // pixel position in current frame + Game Motion Vector -> pixel position in previous frame
    FfxFloat32x3 prevBackbufferCol = SamplePreviousBackbuffer(fUvInInterpolationRect+ fGameMotionVector * fUvLetterBoxScale).xyz; //returns color of current frame's pixel in previous frame buffer
    FfxFloat32x3 curBackbufferCol  = SampleCurrentBackbuffer(fUvInInterpolationRect).xyz; // returns color of current frame's pixel in current frame buffer
    FfxFloat32   prevLuma          = 0.001f + RawRGBToLuminance(prevBackbufferCol);
    FfxFloat32   currLuma          = 0.001f + RawRGBToLuminance(curBackbufferCol);

    FfxUInt32 uLowPriorityFactor = FfxUInt32(ffxRound(ffxPow(MinDividedByMax(prevLuma, currLuma), 1.0f / 1.0f) * PRIORITY_LOW_MAX))
    * FfxUInt32(IsUvInside(fUvInInterpolationRect + fGameMotionVector * fUvLetterBoxScale));

    // Update primary motion vectors
    {
        const FfxUInt32x2 packedVectorPrimary = PackVectorFieldEntries(true, uHighPriorityFactorPrimary, uLowPriorityFactor, fMotionVectorHalf);

        BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fInterpolatedLocationUv, RenderSize());
        for (FfxInt32 iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++)
        {
            const FfxInt32x2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
            const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;

            if (IsOnScreen(iSamplePos, RenderSize()))
            {
                UpdateGameMotionVectorField(iSamplePos, packedVectorPrimary);
            }
        }
    }

    // Update secondary vectors
    // Main purpose of secondary vectors is to improve quality of inpainted vectors
    const FfxBoolean bWriteSecondaryVectors = length(fMotionVectorHalf * RenderSize()) > FFX_FRAMEINTERPOLATION_EPSILON;
    if (bWriteSecondaryVectors)
    {
        FfxBoolean bWriteSecondary = true;
        FfxUInt32 uNumPrimaryHits = 0;
        const FfxFloat32 fSecondaryStepScale = length(1.0f / RenderSize());
        const FfxFloat32x2 fStepMv = normalize(fGameMotionVector);
        const FfxFloat32 fBreakDist = ffxMin(length(fMotionVectorHalf), length(FfxFloat32x2(0.5f, 0.5f)));

        for (FfxFloat32 fMvScale = fSecondaryStepScale; fMvScale <= fBreakDist && bWriteSecondary; fMvScale += fSecondaryStepScale)
        {
            const FfxFloat32x2 fSecondaryLocationUv = fInterpolatedLocationUv - fStepMv * fMvScale;
            BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fSecondaryLocationUv, RenderSize());

            // Reverse depth prio for secondary vectors
            FfxUInt32 uHighPriorityFactorSecondary = ffxMax(1, PRIORITY_HIGH_MAX - uHighPriorityFactorPrimary);

            const FfxFloat32x2 fToCenter = normalize(FfxFloat32x2(0.5f, 0.5f) - fSecondaryLocationUv);
            uLowPriorityFactor = FfxUInt32(ffxMax(0.0f, dot(fToCenter, fStepMv)) * PRIORITY_LOW_MAX);
            const FfxUInt32x2 packedVectorSecondary = PackVectorFieldEntries(false, uHighPriorityFactorSecondary, uLowPriorityFactor, fMotionVectorHalf);

            // Only write secondary mvs to single bilinear location
            for (FfxInt32 iSampleIndex = 0; iSampleIndex < 1; iSampleIndex++)
            {
                const FfxInt32x2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
                const FfxInt32x2 iSamplePos = bilinearInfo.iBasePos + iOffset;

                bWriteSecondary = bWriteSecondary && IsOnScreen(iSamplePos, RenderSize());

                if (bWriteSecondary)
                {
                    const FfxUInt32 uExistingVectorFieldEntry = UpdateGameMotionVectorFieldEx(iSamplePos, packedVectorSecondary);

                    uNumPrimaryHits += FfxUInt32(PackedVectorFieldEntryIsPrimary(uExistingVectorFieldEntry));
                    bWriteSecondary = bWriteSecondary && (uNumPrimaryHits <= 3);
                }
            }
        }
    }
}

#endif // FFX_FRAMEINTERPOLATION_GAME_MOTION_VECTOR_FIELD_H
