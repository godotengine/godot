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

struct FfxDebugViewport
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

FfxFloat32x2 getTransformedUv(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxFloat32x2 fUv = (FfxFloat32x2(iPxPos - vp.offset) + 0.5f) / vp.size;

    return fUv;
}

FfxFloat32x3 getMotionVectorColor(FfxFloat32x2 fMotionVector)
{
    return FfxFloat32x3(0.5f + fMotionVector * RenderSize() * 0.5f, 0.5f);
}

FfxFloat32x4 getUnusedIndicationColor(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxInt32x2 basePos = iPxPos - vp.offset;

    FfxFloat32 ar = FfxFloat32(vp.size.x) / FfxFloat32(vp.size.y);

    return FfxFloat32x4(basePos.x == FfxInt32(basePos.y * ar), 0, 0, 1);
}

void drawDilatedMotionVectors(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x2 fUv_HW = ClampUv(fUv, RenderSize(), MaxRenderSize());

    FfxFloat32x2 fMotionVector = SampleDilatedMotionVector(fUv_HW);

    StoreUpscaledOutput(iPxPos, getMotionVectorColor(fMotionVector));
}

void drawDisocclusionMask(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x2 fUv_HW = ClampUv(fUv, RenderSize(), MaxRenderSize());

    FfxFloat32 fDisocclusionFactor = ffxSaturate(SampleDilatedReactiveMasks(fUv_HW)[DISOCCLUSION]);

    StoreUpscaledOutput(iPxPos, FfxFloat32x3(0, fDisocclusionFactor, 0));
}

void drawDetailProtectionTakedown(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x2 fUv_HW = ClampUv(fUv, RenderSize(), MaxRenderSize());

    FfxFloat32 fProtectionTakedown = ffxSaturate(SampleDilatedReactiveMasks(fUv_HW)[REACTIVE]);

    StoreUpscaledOutput(iPxPos, FfxFloat32x3(0, fProtectionTakedown, 0));
}

void drawReactiveness(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x2 fUv_HW = ClampUv(fUv, RenderSize(), MaxRenderSize());

    FfxFloat32 fShadingChange = ffxSaturate(SampleDilatedReactiveMasks(fUv_HW)[SHADING_CHANGE]);

    StoreUpscaledOutput(iPxPos, FfxFloat32x3(0, fShadingChange, 0));
}

void drawProtectedAreas(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32 fProtection = ffxSaturate(SampleHistory(fUv).w - fLockThreshold);

    StoreUpscaledOutput(iPxPos, FfxFloat32x3(fProtection, 0, 0));
}

void drawDilatedDepthInMeters(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxFloat32x2 fUv = getTransformedUv(iPxPos, vp);

    FfxFloat32x2 fUv_HW = ClampUv(fUv, RenderSize(), MaxRenderSize());

    const FfxFloat32 fDilatedDepth = SampleDilatedDepth(fUv_HW);
    const FfxFloat32 fDepthInMeters = GetViewSpaceDepthInMeters(fDilatedDepth);

    StoreUpscaledOutput(iPxPos, FfxFloat32x3(ffxSaturate(fDepthInMeters / 25.0f), 0, 0));
}

FfxBoolean pointIsInsideViewport(FfxInt32x2 iPxPos, FfxDebugViewport vp)
{
    FfxInt32x2 extent = vp.offset + vp.size;

    return (iPxPos.x >= vp.offset.x && iPxPos.x < extent.x) && (iPxPos.y >= vp.offset.y && iPxPos.y < extent.y);
}

void DebugView(FfxInt32x2 iPxPos)
{
#define VIEWPORT_GRID_SIZE_X 3
#define VIEWPORT_GRID_SIZE_Y 3

    FfxFloat32x2 fViewportScale = FfxFloat32x2(1.0f / VIEWPORT_GRID_SIZE_X, 1.0f / VIEWPORT_GRID_SIZE_Y);
    FfxInt32x2   iViewportSize  = FfxInt32x2(UpscaleSize() * fViewportScale);

    // compute grid [y][x] for easier placement of viewports
    FfxDebugViewport vp[VIEWPORT_GRID_SIZE_Y][VIEWPORT_GRID_SIZE_X];
    for (FfxInt32 y = 0; y < VIEWPORT_GRID_SIZE_Y; y++)
    {
        for (FfxInt32 x = 0; x < VIEWPORT_GRID_SIZE_X; x++)
        {
            vp[y][x].offset = iViewportSize * FfxInt32x2(x, y);
            vp[y][x].size   = iViewportSize;
        }
    }

    // top row
    DRAW_VIEWPORT(drawDilatedMotionVectors,                 iPxPos, vp[0][0]);
    DRAW_VIEWPORT(drawProtectedAreas,                       iPxPos, vp[0][1]);
    DRAW_VIEWPORT(drawDilatedDepthInMeters,                 iPxPos, vp[0][2]);

    // bottom row
    DRAW_VIEWPORT(drawDisocclusionMask,                     iPxPos, vp[2][0]);
    DRAW_VIEWPORT(drawReactiveness,                         iPxPos, vp[2][1]);
    DRAW_VIEWPORT(drawDetailProtectionTakedown,             iPxPos, vp[2][2]);
}
