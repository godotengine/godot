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

void ReconstructPrevDepth(FfxInt32x2 iPxPos, FfxFloat32 fDepth, FfxFloat32x2 fMotionVector)
{
    const FfxFloat32 fNearestDepthInMeters = ffxMin(GetViewSpaceDepthInMeters(fDepth), FSR3UPSCALER_FP16_MAX);
    const FfxFloat32 fReconstructedDeptMvThreshold = ReconstructedDepthMvPxThreshold(fNearestDepthInMeters);

    // Discard small mvs
    fMotionVector *= FfxFloat32(Get4KVelocity(fMotionVector) > fReconstructedDeptMvThreshold);

    const FfxFloat32x2 fUv = (iPxPos + FfxFloat32(0.5)) / RenderSize();
    const FfxFloat32x2 fReprojectedUv = fUv + fMotionVector;
    const BilinearSamplingData bilinearInfo = GetBilinearSamplingData(fReprojectedUv, RenderSize());

    // Project current depth into previous frame locations.
    // Push to all pixels having some contribution if reprojection is using bilinear logic.
    for (FfxInt32 iSampleIndex = 0; iSampleIndex < 4; iSampleIndex++) {
        
        const FfxInt32x2 iOffset = bilinearInfo.iOffsets[iSampleIndex];
        const FfxFloat32 fWeight = bilinearInfo.fWeights[iSampleIndex];

        if (fWeight > fReconstructedDepthBilinearWeightThreshold) {

            const FfxInt32x2 iStorePos = bilinearInfo.iBasePos + iOffset;
            if (IsOnScreen(iStorePos, RenderSize())) {
                StoreReconstructedDepth(iStorePos, fDepth);
            }
        }
    }
}

struct DepthExtents
{
    FfxFloat32 fNearest;
    FfxInt32x2 fNearestCoord;
    FfxFloat32 fFarthest;
};

DepthExtents FindDepthExtents(FFX_PARAMETER_IN FfxInt32x2 iPxPos)
{
    DepthExtents extents;
    const FfxInt32 iSampleCount = 9;
    const FfxInt32x2 iSampleOffsets[iSampleCount] = {
        FfxInt32x2(+0, +0),
        FfxInt32x2(+1, +0),
        FfxInt32x2(+0, +1),
        FfxInt32x2(+0, -1),
        FfxInt32x2(-1, +0),
        FfxInt32x2(-1, +1),
        FfxInt32x2(+1, +1),
        FfxInt32x2(-1, -1),
        FfxInt32x2(+1, -1),
    };

    // pull out the depth loads to allow SC to batch them
    FfxFloat32 depth[9];
    FfxInt32 iSampleIndex = 0;
    FFX_UNROLL
    for (iSampleIndex = 0; iSampleIndex < iSampleCount; ++iSampleIndex) {

        FfxInt32x2 iPos     = iPxPos + iSampleOffsets[iSampleIndex];
        depth[iSampleIndex] = LoadInputDepth(iPos);
    }

    // find closest depth
    extents.fNearestCoord   = iPxPos;
    extents.fNearest        = depth[0];
    extents.fFarthest       = depth[0];
    FFX_UNROLL
    for (iSampleIndex = 1; iSampleIndex < iSampleCount; ++iSampleIndex) {

        const FfxInt32x2 iPos = iPxPos + iSampleOffsets[iSampleIndex];
        if (IsOnScreen(iPos, RenderSize())) {

            FfxFloat32 fNdDepth = depth[iSampleIndex];
#if FFX_FSR3UPSCALER_OPTION_INVERTED_DEPTH
            if (fNdDepth > extents.fNearest) {
                extents.fFarthest       = ffxMin(extents.fFarthest, fNdDepth);
#else
            if (fNdDepth < extents.fNearest) {
                extents.fFarthest       = ffxMax(extents.fFarthest, fNdDepth);
#endif
                extents.fNearestCoord   = iPos;
                extents.fNearest        = fNdDepth;
            }
        }
    }

    return extents;
}

FfxFloat32x2 DilateMotionVector(FfxInt32x2 iPxPos, const DepthExtents depthExtents)
{
#if FFX_FSR3UPSCALER_OPTION_LOW_RESOLUTION_MOTION_VECTORS
    const FfxInt32x2 iSamplePos       = iPxPos;
    const FfxInt32x2 iMotionVectorPos = depthExtents.fNearestCoord;
#else
    const FfxInt32x2 iSamplePos       = ComputeHrPosFromLrPos(iPxPos);
    const FfxInt32x2 iMotionVectorPos = ComputeHrPosFromLrPos(depthExtents.fNearestCoord);
#endif

    const FfxFloat32x2 fDilatedMotionVector = LoadInputMotionVector(iMotionVectorPos);

    return fDilatedMotionVector;
}

FfxFloat32 GetCurrentFrameLuma(FfxInt32x2 iPxPos)
{
    //We assume linear data. if non-linear input (sRGB, ...),
    //then we should convert to linear first and back to sRGB on output.
    const FfxFloat32x3 fRgb = ffxMax(FfxFloat32x3(0, 0, 0), LoadInputColor(iPxPos));
    const FfxFloat32 fLuma  = RGBToLuma(fRgb);

    return fLuma;
}

void PrepareInputs(FfxInt32x2 iPxPos)
{
    const DepthExtents depthExtents = FindDepthExtents(iPxPos);
    const FfxFloat32x2 fDilatedMotionVector = DilateMotionVector(iPxPos, depthExtents);

    ReconstructPrevDepth(iPxPos, depthExtents.fNearest, fDilatedMotionVector);

    StoreDilatedMotionVector(iPxPos, fDilatedMotionVector);
    StoreDilatedDepth(iPxPos, depthExtents.fNearest);

    const FfxFloat32 fFarthestDepthInMeters = ffxMin(GetViewSpaceDepthInMeters(depthExtents.fFarthest), FSR3UPSCALER_FP16_MAX);
    StoreFarthestDepth(iPxPos, fFarthestDepthInMeters);

    const FfxFloat32 fLuma = GetCurrentFrameLuma(iPxPos);
    StoreCurrentLuma(iPxPos, fLuma);
}
