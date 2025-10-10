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

#ifndef FFX_FRAMEINTERPOLATION_DEBUG_VIEW_H
#define FFX_FRAMEINTERPOLATION_DEBUG_VIEW_H

struct FfxFrameInterpolationDebugViewport
{
    FfxInt32x2 offset;
    FfxInt32x2 size;
};

// Macro to cull and draw debug viewport
#define DRAW_VIEWPORT(function, pos, vp)    \
    {                                       \
        if (pointIsInsideViewport(pos, vp)) \
        {                                   \
            function(pos, vp);              \
        }                                   \
    }

FfxFloat32x2 getTransformedUv(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxFloat32x2 fUv = (FfxFloat32x2(iPxPos - vp.offset) + 0.5f) / vp.size;

    return fUv;
}

FfxFloat32x4 getMotionVectorColor(FfxFloat32x2 fMotionVector)
{
    return FfxFloat32x4(0.5f + fMotionVector * DisplaySize() * 0.1f, 0.5f, 1.0f);
}

FfxFloat32x4 getUnusedIndicationColor(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxInt32x2 basePos = iPxPos - vp.offset;

    FfxFloat32 ar = FfxFloat32(vp.size.x) / FfxFloat32(vp.size.y);

    return FfxFloat32x4(basePos.x == FfxInt32(basePos.y * ar), 0, 0, 1);
}

void drawGameMotionVectorFieldVectors(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    VectorFieldEntry gameMv;
    LoadInpaintedGameFieldMv(fUv, gameMv);

    StoreFrameinterpolationOutput(iPxPos, getMotionVectorColor(gameMv.fMotionVector));
}

void drawGameMotionVectorFieldDepthPriority(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    VectorFieldEntry gameMv;
    LoadInpaintedGameFieldMv(fUv, gameMv);

    StoreFrameinterpolationOutput(iPxPos, FfxFloat32x4(0, gameMv.uHighPriorityFactor, 0, 1));
}

void drawOpticalFlowMotionVectorField(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    VectorFieldEntry ofMv;
    SampleOpticalFlowMotionVectorField(fUv, ofMv);

    StoreFrameinterpolationOutput(iPxPos, getMotionVectorColor(ofMv.fMotionVector));
}

void drawDisocclusionMask(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x2 fLrUv = fUv * (FfxFloat32x2(RenderSize()) / GetMaxRenderSize());

    FfxFloat32x2 fDisocclusionFactor = ffxSaturate(SampleDisocclusionMask(fLrUv).xy);

    StoreFrameinterpolationOutput(iPxPos, FfxFloat32x4(fDisocclusionFactor, 0, 1));
}

FfxFloat32x4 getDistortionField(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x2 fDistortionFieldUv = abs(SampleDistortionField(fUv).xy);

    return FfxFloat32x4(fDistortionFieldUv * 10.0f, 0.0f, 1.0f);
}

void drawPresentBackbuffer(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x4 fPresentColor = getDistortionField(iPxPos, vp);

    if (GetHUDLessAttachedFactor() == 1)
    {
        fPresentColor = SamplePresentBackbuffer(fUv);
    }

    StoreFrameinterpolationOutput(iPxPos, fPresentColor);
}

void drawCurrentInterpolationSource(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x4 fCurrentBackBuffer = FfxFloat32x4(SampleCurrentBackbuffer(fUv), 1.0f);

    StoreFrameinterpolationOutput(iPxPos, fCurrentBackBuffer);
}

FfxBoolean pointIsInsideViewport(FfxInt32x2 iPxPos, FfxFrameInterpolationDebugViewport vp)
{
    FfxInt32x2 extent = vp.offset + vp.size;

    return (iPxPos.x >= vp.offset.x && iPxPos.x < extent.x) && (iPxPos.y >= vp.offset.y && iPxPos.y < extent.y);
}

void computeDebugView(FfxInt32x2 iPxPos)
{
#define VIEWPORT_GRID_SIZE_X 3
#define VIEWPORT_GRID_SIZE_Y 3

    FfxFloat32x2 fViewportScale = FfxFloat32x2(1.0f / VIEWPORT_GRID_SIZE_X, 1.0f / VIEWPORT_GRID_SIZE_Y);
    FfxInt32x2   iViewportSize  = FfxInt32x2(DisplaySize() * fViewportScale);

    // compute grid [y][x] for easier placement of viewports
    FfxFrameInterpolationDebugViewport vp[VIEWPORT_GRID_SIZE_Y][VIEWPORT_GRID_SIZE_X];
    for (FfxInt32 y = 0; y < VIEWPORT_GRID_SIZE_Y; y++)
    {
        for (FfxInt32 x = 0; x < VIEWPORT_GRID_SIZE_X; x++)
        {
            vp[y][x].offset = iViewportSize * FfxInt32x2(x, y);
            vp[y][x].size   = iViewportSize;
        }
    }

    // top row
    DRAW_VIEWPORT(drawGameMotionVectorFieldVectors,         iPxPos, vp[0][0]);
    DRAW_VIEWPORT(drawGameMotionVectorFieldDepthPriority,   iPxPos, vp[0][1]);
    DRAW_VIEWPORT(drawOpticalFlowMotionVectorField,         iPxPos, vp[0][2]);

    // bottom row
    DRAW_VIEWPORT(drawDisocclusionMask,                     iPxPos, vp[2][0]);
    DRAW_VIEWPORT(drawCurrentInterpolationSource,           iPxPos, vp[2][1]);
    DRAW_VIEWPORT(drawPresentBackbuffer,                    iPxPos, vp[2][2]);
}

#endif  // FFX_FRAMEINTERPOLATION_DEBUG_VIEW_H
