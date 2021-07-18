/*
 * Copyright (c) 2020-2021 Samsung Electronics Co., Ltd. All rights reserved.

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
#include <algorithm>
#include "tvgSwCommon.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static bool _genOutline(SwImage* image, const Picture* pdata, const Matrix* transform, SwMpool* mpool,  unsigned tid)
{
    float w, h;
    pdata->viewbox(nullptr, nullptr, &w, &h);
    if (w == 0 || h == 0) return false;

    image->outline = mpoolReqOutline(mpool, tid);
    auto outline = image->outline;

    outline->reservedPtsCnt = 5;
    outline->pts = static_cast<SwPoint*>(realloc(outline->pts, outline->reservedPtsCnt * sizeof(SwPoint)));
    outline->types = static_cast<uint8_t*>(realloc(outline->types, outline->reservedPtsCnt * sizeof(uint8_t)));

    outline->reservedCntrsCnt = 1;
    outline->cntrs = static_cast<uint32_t*>(realloc(outline->cntrs, outline->reservedCntrsCnt * sizeof(uint32_t)));

    Point to[4] = {{0 ,0}, {w, 0}, {w, h}, {0, h}};
    for (int i = 0; i < 4; i++) {
        outline->pts[outline->ptsCnt] = mathTransform(&to[i], transform);
        outline->types[outline->ptsCnt] = SW_CURVE_TYPE_POINT;
        ++outline->ptsCnt;
    }

    outline->pts[outline->ptsCnt] = outline->pts[0];
    outline->types[outline->ptsCnt] = SW_CURVE_TYPE_POINT;
    ++outline->ptsCnt;

    outline->cntrs[outline->cntrsCnt] = outline->ptsCnt - 1;
    ++outline->cntrsCnt;

    outline->opened = false;

    image->outline = outline;
    image->w = w;
    image->h = h;

    return true;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/


bool imagePrepare(SwImage* image, const Picture* pdata, const Matrix* transform, const SwBBox& clipRegion, SwBBox& renderRegion, SwMpool* mpool, unsigned tid)
{
    if (!_genOutline(image, pdata, transform, mpool, tid)) return false;
    return mathUpdateOutlineBBox(image->outline, clipRegion, renderRegion);
}


bool imagePrepared(const SwImage* image)
{
    return image->rle ? true : false;
}


bool imageGenRle(SwImage* image, TVG_UNUSED const Picture* pdata, const SwBBox& renderRegion, bool antiAlias)
{
    if ((image->rle = rleRender(image->rle, image->outline, renderRegion, antiAlias))) return true;

    return false;
}


void imageDelOutline(SwImage* image, SwMpool* mpool, uint32_t tid)
{
    mpoolRetOutline(mpool, tid);
    image->outline = nullptr;
}


void imageReset(SwImage* image)
{
    rleReset(image->rle);
    image->rle = nullptr;
}


void imageFree(SwImage* image)
{
    rleFree(image->rle);
}
