/*
 * Copyright (c) 2026 ThorVG project. All rights reserved.

 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:

 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.

 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _TVG_GPU_COMMON_H_
#define _TVG_GPU_COMMON_H_

#include "tvgMath.h"
#include "tvgRender.h"

namespace tvg
{

void gpuOptimize(const RenderPath& in, RenderPath& out, const Matrix& matrix, bool& thin, bool& skipFill);
bool gpuEdgesCross(const Point& p0, const Point& p1, const Point& p2, const Point& p3);
bool gpuStrokeDash(const RenderShape& rs, RenderPath& out, const Matrix* transform);

// Conservative triangle-fan safety check
// The fill is emitted as (v0,v1,v2), (v0,v2,v3), (v0,v3,v4), ...
// Usage:
// 1. Setup: default-construct `ConvexProbe probe` for one tessellation pass.
// 2. Feed one contour at a time: call `nextContour()` on MoveTo, then stream
//    each contour edge with `addEdge(curr - prev)`, and finish with
//    `addContourClose(first - prev)` on Close. For cubic segments, callers may
//    pre-reject looping control polygons with `gpuEdgesCross(start, ctrl1, ctrl2, end)`.
// 3. Read `probe.convex` as the result. `true` means the contour stayed within
//    the cheap triangle-fan fast-path assumptions; `false` means fall back.
// Cost: O(1) work per edge and O(1) memory.
struct GpuConvexProbe
{
    bool convex = true;
    int8_t winding = 0;
    Point firstEdge = {};
    Point prevEdge = {};
    int8_t prevXDir = 0;
    int8_t prevYDir = 0;
    uint8_t xDirChanges = 0;
    uint8_t yDirChanges = 0;
    uint8_t reversals = 0;
    bool contourHasEdges = false;

    void nextContour()
    {
        if (contourHasEdges) convex = false;
        resetContour();
    }

    void addContourClose(const Point& edge)
    {
        addEdge(edge);
        if (convex && !zero(firstEdge)) addEdge(firstEdge);
    }

    void addEdge(const Point& edge);

private:
    enum : uint8_t
    {
        MaxAxisDirChanges = 3,
        MaxCollinearReversals = 2
    };

    void resetContour()
    {
        firstEdge = {};
        prevEdge = {};
        prevXDir = prevYDir = 0;
        xDirChanges = yDirChanges = 0;
        reversals = 0;
        contourHasEdges = false;
    }

    void updateDir(float value, int8_t& prevDir, uint8_t& changes);
};

}  // namespace tvg

#endif  //_TVG_GPU_COMMON_H_
