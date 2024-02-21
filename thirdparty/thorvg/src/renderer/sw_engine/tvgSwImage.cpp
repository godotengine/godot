/*
 * Copyright (c) 2020 - 2024 the ThorVG project. All rights reserved.

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

static inline bool _onlyShifted(const Matrix* m)
{
    if (mathEqual(m->e11, 1.0f) && mathEqual(m->e22, 1.0f) && mathZero(m->e12) && mathZero(m->e21)) return true;
    return false;
}


static bool _genOutline(SwImage* image, const RenderMesh* mesh, const Matrix* transform, SwMpool* mpool, unsigned tid)
{
    image->outline = mpoolReqOutline(mpool, tid);
    auto outline = image->outline;

    outline->pts.reserve(5);
    outline->types.reserve(5);
    outline->cntrs.reserve(1);
    outline->closed.reserve(1);

    Point to[4];
    if (mesh->triangleCnt > 0) {
        // TODO: Optimise me. We appear to calculate this exact min/max bounding area in multiple
        // places. We should be able to re-use one we have already done? Also see:
        //   tvgPicture.h --> bounds
        //   tvgSwRasterTexmap.h --> _rasterTexmapPolygonMesh
        //
        // TODO: Should we calculate the exact path(s) of the triangle mesh instead?
        // i.e. copy tvgSwShape.capp -> _genOutline?
        //
        // TODO: Cntrs?
        auto triangles = mesh->triangles;
        auto min = triangles[0].vertex[0].pt;
        auto max = triangles[0].vertex[0].pt;

        for (uint32_t i = 0; i < mesh->triangleCnt; ++i) {
            if (triangles[i].vertex[0].pt.x < min.x) min.x = triangles[i].vertex[0].pt.x;
            else if (triangles[i].vertex[0].pt.x > max.x) max.x = triangles[i].vertex[0].pt.x;
            if (triangles[i].vertex[0].pt.y < min.y) min.y = triangles[i].vertex[0].pt.y;
            else if (triangles[i].vertex[0].pt.y > max.y) max.y = triangles[i].vertex[0].pt.y;

            if (triangles[i].vertex[1].pt.x < min.x) min.x = triangles[i].vertex[1].pt.x;
            else if (triangles[i].vertex[1].pt.x > max.x) max.x = triangles[i].vertex[1].pt.x;
            if (triangles[i].vertex[1].pt.y < min.y) min.y = triangles[i].vertex[1].pt.y;
            else if (triangles[i].vertex[1].pt.y > max.y) max.y = triangles[i].vertex[1].pt.y;

            if (triangles[i].vertex[2].pt.x < min.x) min.x = triangles[i].vertex[2].pt.x;
            else if (triangles[i].vertex[2].pt.x > max.x) max.x = triangles[i].vertex[2].pt.x;
            if (triangles[i].vertex[2].pt.y < min.y) min.y = triangles[i].vertex[2].pt.y;
            else if (triangles[i].vertex[2].pt.y > max.y) max.y = triangles[i].vertex[2].pt.y;
        }
        to[0] = {min.x, min.y};
        to[1] = {max.x, min.y};
        to[2] = {max.x, max.y};
        to[3] = {min.x, max.y};
    } else {
        auto w = static_cast<float>(image->w);
        auto h = static_cast<float>(image->h);
        to[0] = {0, 0};
        to[1] = {w, 0};
        to[2] = {w, h};
        to[3] = {0, h};
    }

    for (int i = 0; i < 4; i++) {
        outline->pts.push(mathTransform(&to[i], transform));
        outline->types.push(SW_CURVE_TYPE_POINT);
    }

    outline->pts.push(outline->pts[0]);
    outline->types.push(SW_CURVE_TYPE_POINT);
    outline->cntrs.push(outline->pts.count - 1);
    outline->closed.push(true);

    image->outline = outline;

    return true;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

bool imagePrepare(SwImage* image, const RenderMesh* mesh, const Matrix* transform, const SwBBox& clipRegion, SwBBox& renderRegion, SwMpool* mpool, unsigned tid)
{
    image->direct = _onlyShifted(transform);

    //Fast track: Non-transformed image but just shifted.
    if (image->direct) {
        image->ox = -static_cast<int32_t>(round(transform->e13));
        image->oy = -static_cast<int32_t>(round(transform->e23));
    //Figure out the scale factor by transform matrix
    } else {
        auto scaleX = sqrtf((transform->e11 * transform->e11) + (transform->e21 * transform->e21));
        auto scaleY = sqrtf((transform->e22 * transform->e22) + (transform->e12 * transform->e12));
        image->scale = (fabsf(scaleX - scaleY) > 0.01f) ? 1.0f : scaleX;

        if (mathZero(transform->e12) && mathZero(transform->e21)) image->scaled = true;
        else image->scaled = false;
    }

    if (!_genOutline(image, mesh, transform, mpool, tid)) return false;
    return mathUpdateOutlineBBox(image->outline, clipRegion, renderRegion, image->direct);
}


bool imageGenRle(SwImage* image, const SwBBox& renderRegion, bool antiAlias)
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
}


void imageFree(SwImage* image)
{
    rleFree(image->rle);
}
