/*
 * Copyright (c) 2020 - 2026 ThorVG project. All rights reserved.

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

#include "tvgMath.h"
#include "tvgSwCommon.h"

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

void utilExport(SwOutline* outline, const Matrix& transform, BBox& bbox)
{
    outline->out.reserve(outline->in.count);

    bbox = {{FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX}};

    ARRAY_FOREACH(pt, outline->in) {
        auto t = *pt * transform;
        if (bbox.min.x > t.x) bbox.min.x = t.x;
        if (bbox.max.x < t.x) bbox.max.x = t.x;
        if (bbox.min.y > t.y) bbox.min.y = t.y;
        if (bbox.max.y < t.y) bbox.max.y = t.y;
        outline->out.push({int32_t(t.x * 64.0f), int32_t(t.y * 64.0f)});
    }
}

bool utilBBox(const BBox& bbox, const RenderRegion& clipBox, RenderRegion& renderBox, bool fastTrack)
{
    if (fastTrack) {
        renderBox.min = {(int32_t)round(bbox.min.x), (int32_t)round(bbox.min.y)};
        renderBox.max = {(int32_t)round(bbox.max.x), (int32_t)round(bbox.max.y)};
    } else {
        renderBox.min = {(int32_t)floorf(bbox.min.x), (int32_t)floorf(bbox.min.y)};
        renderBox.max = {(int32_t)ceilf(bbox.max.x), (int32_t)ceilf(bbox.max.y)};
    }
    renderBox.intersect(clipBox);
    return renderBox.valid();
}
