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
#include "tvgAccessor.h"
#include "tvgLoaderMgr.h"

namespace tvg
{

struct PictureImpl : Picture
{
    Paint::Impl impl;
    ImageLoader* loader = nullptr;
    Paint* vector = nullptr;          //vector picture uses
    RenderSurface* bitmap = nullptr;  //bitmap picture uses
    AssetResolver* resolver = nullptr;
    Point origin = {};
    float w = 0, h = 0;
    FilterMethod filter = FilterMethod::Bilinear;
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
        // The media have its own playback update
        return !loader || (flag == RenderUpdateFlag::None && loader->type != FileType::Media);
    }

    bool update(RenderMethod* renderer, const Matrix& transform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flag, TVG_UNUSED bool clipper)
    {
        flag |= load();
        if (flag == RenderUpdateFlag::None) return true;

        auto pivot = Point{-origin.x * float(w), -origin.y * float(h)};

        if (bitmap) {
            if (bitmap->cs == ColorSpace::Unknown) {
                TVGERR("RENDERER", "Unknown colorspace picture data");
                return false;
            }
            //Overriding Transformation by the desired image size
            auto sx = w / loader->w;
            auto sy = h / loader->h;
            auto scale = sx < sy ? sx : sy;
            auto m = transform * Matrix{scale, 0, pivot.x, 0, scale, pivot.y, 0, 0, 1};
            impl.rd = renderer->prepare(bitmap, impl.rd, m, clips, opacity, filter, flag);
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

    Result filterMethod(FilterMethod method)
    {
        if (method != filter) {
            impl.mark(RenderUpdateFlag::Image);
            filter = method;
        }
        return Result::Success;
    }

    Result size(float* w, float* h) const
    {
        if (!loader) return Result::InsufficientCondition;
        if (w) *w = this->w;
        if (h) *h = this->h;
        return Result::Success;
    }

    bool intersects(const RenderRegion& region, bool visibleOnly)
    {
        if (!impl.renderer) return false;
        impl.mark(load());
        if (impl.rd) return impl.renderer->intersectsImage(impl.rd, region);
        else if (vector) return PAINT(vector)->intersects(region, visibleOnly);
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

        PictureOps ops = {Ownership::Transfer, resolver, nullptr, accessible};
        auto ret = Result::InvalidArguments;
        auto loader = LoaderMgr::loader(filename, ops, ret);
        return loader ? load(loader) : ret;
    }

    Result load(const char* data, uint32_t size, const char* mimeType, const char* rpath, const Ownership owner)
    {
        if (!data || size <= 0) return Result::InvalidArguments;
        if (vector || bitmap) return Result::InsufficientCondition;

        PictureOps ops = {owner, resolver, rpath, accessible};
        auto ret = Result::InvalidArguments;
        auto loader = LoaderMgr::loader(data, size, mimeType, ops, ret);
        return loader ? load(loader) : ret;
    }

    Result load(const uint32_t* data, uint32_t w, uint32_t h, ColorSpace cs, Ownership owner)
    {
        if (!data || w <= 0 || h <= 0 || cs == ColorSpace::Unknown)  return Result::InvalidArguments;
        if (vector) return Result::InsufficientCondition;
        auto loader = LoaderMgr::loader(data, w, h, cs, owner);
        return loader ? load(loader) : Result::NonSupport;
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

        impl.mark(load());

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
        dup->filter = filter;
        dup->resizing = resizing;

        return picture;
    }

    AccessorIterator* iterator()
    {
        impl.mark(load());

        struct PictureIterator : AccessorIterator
        {
            Paint* ptr = nullptr;

            PictureIterator(Paint* p) : ptr(p) {}

            const Paint* next() override
            {
                auto ret = ptr;
                ptr = nullptr;
                return ret;
            }
        };

        return new PictureIterator(vector);
    }

    RenderUpdateFlag load()
    {
        if (!loader) return RenderUpdateFlag::None;

        // reload the next frame if any
        if (vector || bitmap) {
            // sync call must be guaranteed.
            if (loader->sync() && bitmap) return RenderUpdateFlag::Image;
        // load the first frame
        } else {
            if ((vector = loader->paint())) {
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
            } else {
                bitmap = loader->bitmap();
                if (bitmap) return RenderUpdateFlag::Image;
            }
        }
        // animations updates the properties essentially. here update is not necessary.
        return RenderUpdateFlag::None;
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

    bool render(RenderMethod* renderer, TVG_UNUSED CompositionFlag flag)
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
        else if (impl.renderer) return impl.renderer->region(impl.rd);
        return {};
    }

    Result load(Loader* loader)
    {
        //Same resource has been loaded.
        if (this->loader == loader) {
            this->loader->sharing--;  //make it sure the reference counting.
            if (bitmap) impl.mark(RenderUpdateFlag::Image);  // force the bitmap updated
            return Result::Success;
        } else if (this->loader) {
            LoaderMgr::retrieve(this->loader);
        }

        this->loader = static_cast<ImageLoader*>(loader);
        if (!loader->read()) return Result::Unknown;

        this->w = this->loader->w;
        this->h = this->loader->h;

        impl.mark(RenderUpdateFlag::All);

        return Result::Success;
    }

    const AccessorEntity* access(uint32_t id)
    {
        if (loader) return loader->access(id);
        return nullptr;
    }

    void access(AccessorCallback& cb)
    {
        if (loader) loader->access(cb);
    }

    template<class T>
    T* fetch(FileType expect)
    {
        if (loader) {
            if (loader->type == expect) return static_cast<T*>(loader);
            TVGERR("RENDERER", "Invalid loaded data type (expected: %d, got: %d)", (int)expect, (int)loader->type);
        }
        return nullptr;
    }
};

}

#endif //_TVG_PICTURE_H_
