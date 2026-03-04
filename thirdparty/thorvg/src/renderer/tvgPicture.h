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

#ifndef _TVG_PICTURE_H_
#define _TVG_PICTURE_H_

#include "tvgPaint.h"
#include "tvgScene.h"
#include "tvgLoader.h"

namespace tvg
{

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


struct PictureImpl : Picture
{
    Paint::Impl impl;
    ImageLoader* loader = nullptr;
    Paint* vector = nullptr;          //vector picture uses
    RenderSurface* bitmap = nullptr;  //bitmap picture uses
    AssetResolver* resolver = nullptr;
    Point origin = {};
    float w = 0, h = 0;
    bool resizing = false;

    PictureImpl() : impl(Paint::Impl(this))
    {
    }

    ~PictureImpl()
    {
        LoaderMgr::retrieve(loader);
        tvg::free(resolver);
        if (vector) vector->unref();
    }

    bool skip(RenderUpdateFlag flag)
    {
        if (flag == RenderUpdateFlag::None) return true;
        return false;
    }

    bool update(RenderMethod* renderer, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flag, TVG_UNUSED bool clipper)
    {
        load();

        auto pivot = Point{-origin.x * float(w), -origin.y * float(h)};

        if (bitmap) {
            //Overriding Transformation by the desired image size
            auto sx = w / loader->w;
            auto sy = h / loader->h;
            auto scale = sx < sy ? sx : sy;
            auto m = transform * Matrix{scale, 0, pivot.x, 0, scale, pivot.y, 0, 0, 1};
            impl.rd = renderer->prepare(bitmap, impl.rd, m, clips, opacity, flag);
        } else if (vector) {
            if (resizing) {
                loader->resize(vector, w, h);
                resizing = false;
            }
            needComposition(opacity);
            vector->blend(pImpl->blendMethod); //propagate blend method to nested vector scene
            translateR(const_cast<Matrix*>(&transform), pivot);
            return vector->pImpl->update(renderer, transform, clips, opacity, flag, false);
        }
        return true;
    }

    void size(float w, float h)
    {
        this->w = w;
        this->h = h;
        resizing = true;
    }

    Result size(float* w, float* h) const
    {
        if (!loader) return Result::InsufficientCondition;
        if (w) *w = this->w;
        if (h) *h = this->h;
        return Result::Success;
    }

    bool intersects(const RenderRegion& region)
    {
        if (!impl.renderer) return false;
        load();
        if (impl.rd) return impl.renderer->intersectsImage(impl.rd, region);
        else if (vector) return to<SceneImpl>(vector)->intersects(region);
        return false;
    }

    bool bounds(Point* pt4, const Matrix& m, TVG_UNUSED bool obb)
    {
        pt4[0] = Point{0.0f, 0.0f} * m;
        pt4[1] = Point{w, 0.0f} * m;
        pt4[2] = Point{w, h} * m;
        pt4[3] = Point{0.0f, h} * m;
        return true;
    }

    Result load(const char* filename)
    {
        if (vector || bitmap) return Result::InsufficientCondition;

        bool invalid;  //Invalid Path
        auto loader = static_cast<ImageLoader*>(LoaderMgr::loader(filename, &invalid));
        if (!loader) {
            if (invalid) return Result::InvalidArguments;
            return Result::NonSupport;
        }
        return load(loader);
    }

    Result load(const char* data, uint32_t size, const char* mimeType, const char* rpath, bool copy)
    {
        if (!data || size <= 0) return Result::InvalidArguments;
        if (vector || bitmap) return Result::InsufficientCondition;
        auto loader = static_cast<ImageLoader*>(LoaderMgr::loader(data, size, mimeType, rpath, copy));
        if (!loader) return Result::NonSupport;
        return load(loader);
    }

    Result load(const uint32_t* data, uint32_t w, uint32_t h, ColorSpace cs, bool copy)
    {
        if (!data || w <= 0 || h <= 0 || cs == ColorSpace::Unknown)  return Result::InvalidArguments;
        if (vector || bitmap) return Result::InsufficientCondition;

        auto loader = static_cast<ImageLoader*>(LoaderMgr::loader(data, w, h, cs, copy));
        if (!loader) return Result::FailedAllocation;

        return load(loader);
    }

    Result set(std::function<bool(Paint* paint, const char* src, void* data)> resolver, void* data)
    {
        if (loader) return Result::InsufficientCondition;

        if (!resolver) {
            tvg::free(this->resolver);
            this->resolver = nullptr;
            return Result::Success;
        }

        if (!this->resolver) this->resolver = tvg::calloc<AssetResolver>(1, sizeof(AssetResolver));
        *(this->resolver) = {resolver, data};
        return Result::Success;
    }

    Paint* duplicate(Paint* ret)
    {
        if (ret) TVGERR("RENDERER", "TODO: duplicate()");

        load();

        auto picture = Picture::gen();
        auto dup = to<PictureImpl>(picture);

        if (vector) {
            dup->vector = vector->duplicate();
            PAINT(dup->vector)->parent = picture;
        }

        if (loader) {
            dup->loader = loader;
            ++dup->loader->sharing;
            PAINT(picture)->mark(RenderUpdateFlag::Image);
        }

        dup->bitmap = bitmap;
        dup->origin = origin;
        dup->w = w;
        dup->h = h;
        dup->resizing = resizing;

        return picture;
    }

    Iterator* iterator()
    {
        load();
        return new PictureIterator(vector);
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
        if (bitmap) return bitmap->buf32;
        else return nullptr;
    }

    void load()
    {
        if (loader) {
            if (vector) {
                loader->sync();
            } else if ((vector = loader->paint())) {
                vector->ref();
                PAINT(vector)->parent = this;
                if (w != loader->w || h != loader->h) {
                    if (!resizing) {
                        w = loader->w;
                        h = loader->h;
                    }
                    loader->resize(vector, w, h);
                    resizing = false;
                }
            } else if (!bitmap) {
                bitmap = loader->bitmap();
            }
        }
    }

    void needComposition(uint8_t opacity)
    {
        impl.cmpFlag = CompositionFlag::Invalid;  //must clear after the rendering

        //In this case, paint(scene) would try composition itself.
        if (opacity < 255) return;

        //Composition test
        const Paint* target;
        PAINT(this)->mask(&target);
        if (!target || target->pImpl->opacity == 255 || target->pImpl->opacity == 0) return;
        impl.mark(CompositionFlag::Opacity);
    }

    bool render(RenderMethod* renderer)
    {
        auto ret = true;

        if (bitmap) {
            renderer->blend(impl.blendMethod);
            return renderer->renderImage(impl.rd);
        } else if (vector) {
            RenderCompositor* cmp = nullptr;
            if (impl.cmpFlag) {
                cmp = renderer->target(bounds(), renderer->colorSpace(), impl.cmpFlag);
                renderer->beginComposite(cmp, MaskMethod::None, 255);
            }
            ret = vector->pImpl->render(renderer);
            if (cmp) renderer->endComposite(cmp);
        }
        return ret;
    }

    RenderRegion bounds()
    {
        if (vector) return vector->pImpl->bounds();
        return impl.renderer->region(impl.rd);
    }

    Result load(ImageLoader* loader)
    {
        //Same resource has been loaded.
        if (this->loader == loader) {
            this->loader->sharing--;  //make it sure the reference counting.
            return Result::Success;
        } else if (this->loader) {
            LoaderMgr::retrieve(this->loader);
        }

        this->loader = loader;
        loader->set(resolver);
        if (!loader->read()) return Result::Unknown;

        this->w = loader->w;
        this->h = loader->h;

        impl.mark(RenderUpdateFlag::All);

        return Result::Success;
    }
};

}

#endif //_TVG_PICTURE_H_
