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
/* Internal Class Implementation                                        */
/************************************************************************/

static SwOutline* _genOutline(SwImage& image, SwMpool* mpool, unsigned tid)
{
    auto outline = mpool->outline(tid);
    outline->in.reserve(5);
    outline->types.reserve(5);
    outline->cntrs.reserve(1);
    outline->closed.reserve(1);

    auto w = static_cast<float>(image.w);
    auto h = static_cast<float>(image.h);

    outline->in.push({0.0f, 0.0f});
    outline->in.push({w, 0.0f});
    outline->in.push({w, h});
    outline->in.push({0.0f, h});
    outline->in.push({0.0f, 0.0f});

    outline->types.push(SW_CURVE_TYPE_POINT);
    outline->types.push(SW_CURVE_TYPE_POINT);
    outline->types.push(SW_CURVE_TYPE_POINT);
    outline->types.push(SW_CURVE_TYPE_POINT);
    outline->types.push(SW_CURVE_TYPE_POINT);
    outline->cntrs.push(outline->in.count - 1);
    outline->closed.push(true);

    outline->fillRule = FillRule::NonZero;

    return outline;
}

/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

bool imagePrepare(SwImage& image, const Matrix& transform, const RenderRegion& clipBox, RenderRegion& renderBox, SwMpool* mpool, unsigned tid)
{
    // only shifted
    image.direct = tvg::equal(transform.e11, 1.0f) && tvg::equal(transform.e22, 1.0f) && tvg::zero(transform.e12) && tvg::zero(transform.e21);

    // fast track: Non-transformed image but just shifted.
    if (image.direct) {
        image.ox = -static_cast<int32_t>(nearbyint(transform.e13));
        image.oy = -static_cast<int32_t>(nearbyint(transform.e23));
    // figure out the scale factor by transform matrix
    } else {
        image.scale = scaling(transform);
        image.scaled = (tvg::zero(transform.e12) && tvg::zero(transform.e21)) ? true : false;
    }
    image.outline = _genOutline(image, mpool, tid);
    if (!image.outline) return false;

    BBox bbox;
    utilExport(image.outline, transform, bbox);
    return utilBBox(bbox, clipBox, renderBox, image.direct);
}

bool imageGenRle(SwImage& image, const RenderRegion& renderBox, SwMpool* mpool, unsigned tid, bool antiAlias)
{
    image.rle = rleRender(image.rle, image.outline, renderBox, mpool, tid, antiAlias);
    return image.rle ? true : false;
}

void imageReset(SwImage& image)
{
    rleReset(image.rle);
}

void imageFree(SwImage& image)
{
    rleFree(image.rle);
    image.rle = nullptr;
}
