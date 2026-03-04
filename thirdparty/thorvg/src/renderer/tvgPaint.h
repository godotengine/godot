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

#ifndef _TVG_PAINT_H_
#define _TVG_PAINT_H_

#include "tvgCommon.h"
#include "tvgRender.h"
#include "tvgMath.h"


#define PAINT(A) ((Paint::Impl*)A->pImpl)

namespace tvg
{
enum ContextFlag : uint8_t {Default = 0, FastTrack = 1};

struct Iterator
{
    virtual ~Iterator() {}
    virtual const Paint* next() = 0;
    virtual uint32_t count() = 0;
    virtual void begin() = 0;
};

struct Mask
{
    Paint* target;
    Paint* source;
    MaskMethod method;
};

struct Paint::Impl
{
    Paint* paint = nullptr;
    Paint* parent = nullptr;
    Mask* maskData = nullptr;
    Shape* clipper = nullptr;
    RenderMethod* renderer = nullptr;
    RenderData rd = nullptr;

    struct {
        Matrix m;                 //input matrix
        float degree;             //rotation degree
        float scale;              //scale factor
        bool overriding;          //user transform?

        void update()
        {
            if (overriding) return;
            m.e11 = 1.0f;
            m.e12 = 0.0f;
            m.e21 = 0.0f;
            m.e22 = 1.0f;
            m.e31 = 0.0f;
            m.e32 = 0.0f;
            m.e33 = 1.0f;
            tvg::scale(&m, {scale, scale});
            tvg::rotate(&m, degree);
        }
    } tr;
    RenderUpdateFlag renderFlag = RenderUpdateFlag::None;
    CompositionFlag cmpFlag = CompositionFlag::Invalid;
    BlendMethod blendMethod;
    uint16_t refCnt = 0;       //reference count
    uint8_t ctxFlag;           //See enum ContextFlag
    uint8_t opacity;
    bool hidden : 1;

    Impl(Paint* pnt) : paint(pnt)
    {
        pnt->pImpl = this;
        hidden = false;
        reset();
    }

    virtual ~Impl()
    {
        if (maskData) {
            PAINT(maskData->target)->unref();
            tvg::free(maskData);
        }

        if (clipper) PAINT(clipper)->unref();

        if (renderer) {
            if (rd) renderer->dispose(rd);
            if (renderer->unref() == 0) delete(renderer);
        }
    }

    uint16_t ref()
    {
        return ++refCnt;
    }

    uint16_t unref(bool free = true)
    {
        parent = nullptr;
        return unrefx(free);
    }

    uint16_t unrefx(bool free)
    {
        if (refCnt > 0) --refCnt;

        if (free && refCnt == 0) {
            delete(paint);
            return 0;
        }

        return refCnt;
    }

    void damage(const RenderRegion& vport)
    {
        if (renderer) renderer->damage(rd, vport);
    }

    void damage()
    {
        if (renderer) renderer->damage(rd, bounds());
    }

    void mark(CompositionFlag flag)
    {
        cmpFlag = CompositionFlag(uint8_t(cmpFlag) | uint8_t(flag));
    }

    bool marked(CompositionFlag flag)
    {
        return (uint8_t(cmpFlag) & uint8_t(flag)) ? true : false;
    }

    bool marked(RenderUpdateFlag flag)
    {
        return (renderFlag & flag) ? true : false;
    }

    void mark(RenderUpdateFlag flag)
    {
        renderFlag |= flag;
    }

    bool transform(const Matrix& m)
    {
        if (&tr.m != &m) tr.m = m;
        tr.overriding = true;
        mark(RenderUpdateFlag::Transform);

        return true;
    }

    Matrix& transform()
    {
        //update transform
        if (renderFlag & RenderUpdateFlag::Transform) tr.update();
        return tr.m;
    }

    Matrix ptransform()
    {
        auto p = this;
        auto tm = tvg::identity();
        while (p->parent) {
            p = PAINT(p->parent);
            tm = p->transform() * tm;
        }
        return tm;
    }

    Result clip(Shape* clp)
    {
        if (clp && PAINT(clp)->parent) return Result::InsufficientCondition;
        if (clipper) PAINT(clipper)->unref(clipper != clp);
        clipper = clp;
        if (clp) {
            clp->ref();
            PAINT(clp)->parent = parent;
        }
        return Result::Success;
    }

    Result mask(Paint* target, MaskMethod method)
    {
        if (target && PAINT(target)->parent) return Result::InsufficientCondition;

        if (maskData) {
            PAINT(maskData->target)->unref(maskData->target != target);
            tvg::free(maskData);
            maskData = nullptr;
        }

        if (method == MaskMethod::None) return (target ? Result::InvalidArguments : Result::Success);

        maskData = tvg::malloc<Mask>(sizeof(Mask));
        target->ref();
        maskData->target = target;
        PAINT(target)->parent = parent;
        maskData->source = paint;
        maskData->method = method;
        return Result::Success;
    }

    MaskMethod mask(const Paint** target) const
    {
        if (maskData) {
            if (target) *target = maskData->target;
            return maskData->method;
        } else {
            if (target) *target = nullptr;
            return MaskMethod::None;
        }
    }

    void reset()
    {
        if (clipper) {
            PAINT(clipper)->unref();
            clipper = nullptr;
        }

        if (maskData) {
            PAINT(maskData->target)->unref();
            tvg::free(maskData);
            maskData = nullptr;
        }

        tvg::identity(&tr.m);
        tr.degree = 0.0f;
        tr.scale = 1.0f;
        tr.overriding = false;

        parent = nullptr;
        blendMethod = BlendMethod::Normal;
        renderFlag = RenderUpdateFlag::None;
        ctxFlag = ContextFlag::Default;
        opacity = 255;
        paint->id = 0;
    }

    bool rotate(float degree)
    {
        if (tr.overriding) return false;
        if (tvg::equal(degree, tr.degree)) return true;
        tr.degree = degree;
        mark(RenderUpdateFlag::Transform);

        return true;
    }

    bool scale(float factor)
    {
        if (tr.overriding) return false;
        if (tvg::equal(factor, tr.scale)) return true;
        tr.scale = factor;
        mark(RenderUpdateFlag::Transform);

        return true;
    }

    bool translate(float x, float y)
    {
        if (tr.overriding) return false;
        if (tvg::equal(x, tr.m.e13) && tvg::equal(y, tr.m.e23)) return true;
        tr.m.e13 = x;
        tr.m.e23 = y;
        mark(RenderUpdateFlag::Transform);

        return true;
    }

    void blend(BlendMethod method)
    {
        if (blendMethod != method) {
            blendMethod = method;
            mark(RenderUpdateFlag::Blend);
        }
    }

    Result visible(bool hidden)
    {
        if (this->hidden != hidden) {
            this->hidden = hidden;
            damage();
        }
        return Result::Success;
    }

    bool intersects(const RenderRegion& region);
    RenderRegion bounds();
    bool bounds(Point* pt4, const Matrix* pm, bool obb);
    Iterator* iterator();
    RenderData update(RenderMethod* renderer, const Matrix& pm, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag pFlag, bool clipper = false);
    bool render(RenderMethod* renderer);
    Paint* duplicate(Paint* ret = nullptr);
};

}

#endif //_TVG_PAINT_H_
