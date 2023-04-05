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
    Polygon* triangles = nullptr;     //mesh data
    uint32_t triangleCnt = 0;       //mesh triangle count
    void* rdata = nullptr;            //engine data
    float w = 0, h = 0;
    bool resizing = false;
    uint32_t rendererColorSpace = 0;

    ~Impl()
    {
        if (paint) delete(paint);
        free(triangles);
        free(surface);
    }

    bool dispose(RenderMethod& renderer)
    {
        bool ret = true;
        if (paint) {
            ret = paint->pImpl->dispose(renderer);
        } else if (surface) {
            ret =  renderer.dispose(rdata);
            rdata = nullptr;
        }
        return ret;
    }

    uint32_t reload()
    {
        if (loader) {
            if (!paint) {
                if (auto p = loader->paint()) {
                    paint = p.release();
                    loader->close();
                    if (w != loader->w || h != loader->h) {
                        if (!resizing) {
                            w = loader->w;
                            h = loader->h;
                        }
                        loader->resize(paint, w, h);
                        resizing = false;
                    }
                    if (paint) return RenderUpdateFlag::None;
                }
            }
            free(surface);
            if ((surface = loader->bitmap(rendererColorSpace).release())) {
                loader->close();
                return RenderUpdateFlag::Image;
            }
        }
        return RenderUpdateFlag::None;
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

    void* update(RenderMethod &renderer, const RenderTransform* pTransform, uint32_t opacity, Array<RenderData>& clips, RenderUpdateFlag pFlag, bool clipper)
    {
        rendererColorSpace = renderer.colorSpace();
        auto flag = reload();

        if (surface) {
            auto transform = resizeTransform(pTransform);
            rdata = renderer.prepare(surface, triangles, triangleCnt, rdata, &transform, opacity, clips, static_cast<RenderUpdateFlag>(pFlag | flag));
        } else if (paint) {
            if (resizing) {
                loader->resize(paint, w, h);
                resizing = false;
            }
            rdata = paint->pImpl->update(renderer, pTransform, opacity, clips, static_cast<RenderUpdateFlag>(pFlag | flag), clipper);
        }
        return rdata;
    }

    bool render(RenderMethod &renderer)
    {
        if (surface) {
            if (triangles) return renderer.renderImageMesh(rdata);
            else return renderer.renderImage(rdata);
        }
        else if (paint) return paint->pImpl->render(renderer);
        return false;
    }

    bool viewbox(float* x, float* y, float* w, float* h)
    {
        if (!loader) return false;
        if (x) *x = loader->vx;
        if (y) *y = loader->vy;
        if (w) *w = loader->vw;
        if (h) *h = loader->vh;
        return true;
    }

    bool size(float w, float h)
    {
        this->w = w;
        this->h = h;
        resizing = true;
        return true;
    }

    bool bounds(float* x, float* y, float* w, float* h)
    {
        if (triangleCnt > 0) {
            Point min = { triangles[0].vertex[0].pt.x, triangles[0].vertex[0].pt.y };
            Point max = { triangles[0].vertex[0].pt.x, triangles[0].vertex[0].pt.y };

            for (uint32_t i = 0; i < triangleCnt; ++i) {
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
        if (rdata) return renderer.region(rdata);
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
        if (!loader) return Result::NonSupport;
        this->w = loader->w;
        this->h = loader->h;
        return Result::Success;
    }

    void mesh(const Polygon* triangles, const uint32_t triangleCnt)
    {
        if (triangles && triangleCnt > 0) {
            this->triangleCnt = triangleCnt;
            this->triangles = (Polygon*)malloc(sizeof(Polygon) * triangleCnt);
            memcpy(this->triangles, triangles, sizeof(Polygon) * triangleCnt);
        } else {
            free(this->triangles);
            this->triangles = nullptr;
            this->triangleCnt = 0;
        }
    }

    Paint* duplicate()
    {
        reload();

        auto ret = Picture::gen();

        auto dup = ret.get()->pImpl;
        if (paint) dup->paint = paint->duplicate();

        dup->loader = loader;
        if (surface) {
            dup->surface = static_cast<Surface*>(malloc(sizeof(Surface)));
            *dup->surface = *surface;
        }
        dup->w = w;
        dup->h = h;
        dup->resizing = resizing;

        if (triangleCnt > 0) {
            dup->triangleCnt = triangleCnt;
            dup->triangles = (Polygon*)malloc(sizeof(Polygon) * triangleCnt);
            memcpy(dup->triangles, triangles, sizeof(Polygon) * triangleCnt);
        }

        return ret.release();
    }

    Iterator* iterator()
    {
        reload();
        return new PictureIterator(paint);
    }
};

#endif //_TVG_PICTURE_IMPL_H_
