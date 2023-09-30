/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

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

#ifndef _TVG_PICTURE_IMPL_H_
#define _TVG_PICTURE_IMPL_H_

#include <string>
#include "tvgPaint.h"
#include "tvgLoader.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

struct PictureIterator : Iterator
{
    Paint* paint = nullptr;
    Paint* ptr = nullptr;

    PictureIterator(Paint* p) : paint(p) {}

    const Paint* next() override
    {
        if (!ptr) ptr = paint;
        else ptr = nullptr;
        return ptr;
    }

    uint32_t count() override
    {
        if (paint) return 1;
        else return 0;
    }

    void begin() override
    {
        ptr = nullptr;
    }
};


struct Picture::Impl
{
    shared_ptr<LoadModule> loader = nullptr;

    Paint* paint = nullptr;           //vector picture uses
    Surface* surface = nullptr;       //bitmap picture uses
    RenderData rd = nullptr;          //engine data
    float w = 0, h = 0;
    RenderMesh rm;                    //mesh data
    Picture* picture = nullptr;
    bool resizing = false;
    bool needComp = false;            //need composition

    Impl(Picture* p) : picture(p)
    {
    }

    ~Impl()
    {
        delete(paint);
        delete(surface);
    }

    bool dispose(RenderMethod& renderer)
    {
        if (paint) paint->pImpl->dispose(renderer);
        else if (surface) renderer.dispose(rd);
        rd = nullptr;
        return true;
    }

    RenderTransform resizeTransform(const RenderTransform* pTransform)
    {
        //Overriding Transformation by the desired image size
        auto sx = w / loader->w;
        auto sy = h / loader->h;
        auto scale = sx < sy ? sx : sy;

        RenderTransform tmp;
        tmp.m = {scale, 0, 0, 0, scale, 0, 0, 0, 1};

        if (!pTransform) return tmp;
        else return RenderTransform(pTransform, &tmp);
    }

    bool needComposition(uint8_t opacity)
    {
        //In this case, paint(scene) would try composition itself.
        if (opacity < 255) return false;

        //Composition test
        const Paint* target;
        auto method = picture->composite(&target);
        if (!target || method == tvg::CompositeMethod::ClipPath) return false;
        if (target->pImpl->opacity == 255 || target->pImpl->opacity == 0) return false;

        return true;
    }

    RenderData update(RenderMethod &renderer, const RenderTransform* pTransform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag pFlag, bool clipper)
    {
        auto flag = load();

        if (surface) {
            auto transform = resizeTransform(pTransform);
            rd = renderer.prepare(surface, &rm, rd, &transform, clips, opacity, static_cast<RenderUpdateFlag>(pFlag | flag));
        } else if (paint) {
            if (resizing) {
                loader->resize(paint, w, h);
                resizing = false;
            }
            needComp = needComposition(opacity) ? true : false;
            rd = paint->pImpl->update(renderer, pTransform, clips, opacity, static_cast<RenderUpdateFlag>(pFlag | flag), clipper);
        }
        return rd;
    }

    bool render(RenderMethod &renderer)
    {
        bool ret = false;
        if (surface) return renderer.renderImage(rd);
        else if (paint) {
            Compositor* cmp = nullptr;
            if (needComp) {
                cmp = renderer.target(bounds(renderer), renderer.colorSpace());
                renderer.beginComposite(cmp, CompositeMethod::None, 255);
            }
            ret = paint->pImpl->render(renderer);
            if (cmp) renderer.endComposite(cmp);
        }
        return ret;
    }

    bool size(float w, float h)
    {
        this->w = w;
        this->h = h;
        resizing = true;
        return true;
    }

    bool bounds(float* x, float* y, float* w, float* h, bool stroking)
    {
        if (rm.triangleCnt > 0) {
            auto triangles = rm.triangles;
            auto min = triangles[0].vertex[0].pt;
            auto max = triangles[0].vertex[0].pt;

            for (uint32_t i = 0; i < rm.triangleCnt; ++i) {
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
            if (x) *x = min.x;
            if (y) *y = min.y;
            if (w) *w = max.x - min.x;
            if (h) *h = max.y - min.y;
        } else {
            if (x) *x = 0;
            if (y) *y = 0;
            if (w) *w = this->w;
            if (h) *h = this->h;
        }
        return true;
    }

    RenderRegion bounds(RenderMethod& renderer)
    {
        if (rd) return renderer.region(rd);
        if (paint) return paint->pImpl->bounds(renderer);
        return {0, 0, 0, 0};
    }

    Result load(const string& path)
    {
        if (paint || surface) return Result::InsufficientCondition;
        if (loader) loader->close();
        bool invalid;  //Invalid Path
        loader = LoaderMgr::loader(path, &invalid);
        if (!loader) {
            if (invalid) return Result::InvalidArguments;
            return Result::NonSupport;
        }
        if (!loader->read()) return Result::Unknown;
        w = loader->w;
        h = loader->h;
        return Result::Success;
    }

    Result load(const char* data, uint32_t size, const string& mimeType, bool copy)
    {
        if (paint || surface) return Result::InsufficientCondition;
        if (loader) loader->close();
        loader = LoaderMgr::loader(data, size, mimeType, copy);
        if (!loader) return Result::NonSupport;
        if (!loader->read()) return Result::Unknown;
        w = loader->w;
        h = loader->h;
        return Result::Success;
    }

    Result load(uint32_t* data, uint32_t w, uint32_t h, bool copy)
    {
        if (paint || surface) return Result::InsufficientCondition;
        if (loader) loader->close();
        loader = LoaderMgr::loader(data, w, h, copy);
        if (!loader) return Result::FailedAllocation;
        this->w = loader->w;
        this->h = loader->h;
        return Result::Success;
    }

    void mesh(const Polygon* triangles, const uint32_t triangleCnt)
    {
        if (triangles && triangleCnt > 0) {
            this->rm.triangleCnt = triangleCnt;
            this->rm.triangles = (Polygon*)malloc(sizeof(Polygon) * triangleCnt);
            memcpy(this->rm.triangles, triangles, sizeof(Polygon) * triangleCnt);
        } else {
            free(this->rm.triangles);
            this->rm.triangles = nullptr;
            this->rm.triangleCnt = 0;
        }
    }

    Paint* duplicate()
    {
        load();

        auto ret = Picture::gen();

        auto dup = ret.get()->pImpl;
        if (paint) dup->paint = paint->duplicate();

        dup->loader = loader;
        if (surface) {
            dup->surface = new Surface;
            *dup->surface = *surface;
            //TODO: A dupilcation is not a proxy... it needs copy of the pixel data?
            dup->surface->owner = false;
        }
        dup->w = w;
        dup->h = h;
        dup->resizing = resizing;

        if (rm.triangleCnt > 0) {
            dup->rm.triangleCnt = rm.triangleCnt;
            dup->rm.triangles = (Polygon*)malloc(sizeof(Polygon) * rm.triangleCnt);
            memcpy(dup->rm.triangles, rm.triangles, sizeof(Polygon) * rm.triangleCnt);
        }

        return ret.release();
    }

    Iterator* iterator()
    {
        load();
        return new PictureIterator(paint);
    }

    uint32_t* data(uint32_t* w, uint32_t* h)
    {
        //Try it, If not loaded yet.
        load();

        if (loader) {
            if (w) *w = static_cast<uint32_t>(loader->w);
            if (h) *h = static_cast<uint32_t>(loader->h);
        } else {
            if (w) *w = 0;
            if (h) *h = 0;
        }
        if (surface) return surface->buf32;
        else return nullptr;
    }

    RenderUpdateFlag load();
};

#endif //_TVG_PICTURE_IMPL_H_
