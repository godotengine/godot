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
#include <float.h>
#include <math.h>
#include "tvgPaint.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

static inline bool FLT_SAME(float a, float b)
{
    return (fabsf(a - b) < FLT_EPSILON);
}

static bool _clipPathFastTrack(Paint* cmpTarget, const RenderTransform* pTransform, RenderTransform* rTransform, RenderRegion& viewport)
{
    /* Access Shape class by Paint is bad... but it's ok still it's an internal usage. */
    auto shape = static_cast<Shape*>(cmpTarget);

    //Rectangle Candidates?
    const Point* pts;
     if (shape->pathCoords(&pts) != 4) return false;

    if (rTransform) rTransform->update();

    //No Rotation?
    if (pTransform && (pTransform->m.e12 != 0 || pTransform->m.e21 != 0 || pTransform->m.e11 != pTransform->m.e22)) return false;
    if (rTransform && (rTransform->m.e12 != 0 || rTransform->m.e21 != 0 || rTransform->m.e11 != rTransform->m.e22)) return false;

    //Othogonal Rectangle?
    auto pt1 = pts + 0;
    auto pt2 = pts + 1;
    auto pt3 = pts + 2;
    auto pt4 = pts + 3;

    if ((FLT_SAME(pt1->x, pt2->x) && FLT_SAME(pt2->y, pt3->y) && FLT_SAME(pt3->x, pt4->x) && FLT_SAME(pt1->y, pt4->y)) ||
        (FLT_SAME(pt2->x, pt3->x) && FLT_SAME(pt1->y, pt2->y) && FLT_SAME(pt1->x, pt4->x) && FLT_SAME(pt3->y, pt4->y))) {

        auto x1 = pt1->x;
        auto y1 = pt1->y;
        auto x2 = pt3->x;
        auto y2 = pt3->y;

        if (rTransform) {
            x1 = x1 * rTransform->m.e11 + rTransform->m.e13;
            y1 = y1 * rTransform->m.e22 + rTransform->m.e23;
            x2 = x2 * rTransform->m.e11 + rTransform->m.e13;
            y2 = y2 * rTransform->m.e22 + rTransform->m.e23;
        }

        if (pTransform) {
            x1 = x1 * pTransform->m.e11 + pTransform->m.e13;
            y1 = y1 * pTransform->m.e22 + pTransform->m.e23;
            x2 = x2 * pTransform->m.e11 + pTransform->m.e13;
            y2 = y2 * pTransform->m.e22 + pTransform->m.e23;
        }

        viewport.x = static_cast<uint32_t>(x1);
        viewport.y = static_cast<uint32_t>(y1);
        viewport.w = static_cast<uint32_t>(roundf(x2 - x1 + 0.5f));
        viewport.h = static_cast<uint32_t>(roundf(y2 - y1 + 0.5f));

        return true;
    }

    return false;
}


Paint* Paint::Impl::duplicate()
{
    auto ret = smethod->duplicate();
    if (!ret) return nullptr;

    //duplicate Transform
    if (rTransform) {
        ret->pImpl->rTransform = new RenderTransform();
        if (ret->pImpl->rTransform) {
            *ret->pImpl->rTransform = *rTransform;
            ret->pImpl->flag |= RenderUpdateFlag::Transform;
        }
    }

    ret->pImpl->opacity = opacity;

    if (cmpTarget) ret->pImpl->cmpTarget = cmpTarget->duplicate();

    ret->pImpl->cmpMethod = cmpMethod;

    return ret;
}


bool Paint::Impl::rotate(float degree)
{
    if (rTransform) {
        if (fabsf(degree - rTransform->degree) <= FLT_EPSILON) return true;
    } else {
        if (fabsf(degree) <= FLT_EPSILON) return true;
        rTransform = new RenderTransform();
        if (!rTransform) return false;
    }
    rTransform->degree = degree;
    if (!rTransform->overriding) flag |= RenderUpdateFlag::Transform;

    return true;
}


bool Paint::Impl::scale(float factor)
{
    if (rTransform) {
        if (fabsf(factor - rTransform->scale) <= FLT_EPSILON) return true;
    } else {
        if (fabsf(factor) <= FLT_EPSILON) return true;
        rTransform = new RenderTransform();
        if (!rTransform) return false;
    }
    rTransform->scale = factor;
    if (!rTransform->overriding) flag |= RenderUpdateFlag::Transform;

    return true;
}


bool Paint::Impl::translate(float x, float y)
{
    if (rTransform) {
        if (fabsf(x - rTransform->x) <= FLT_EPSILON && fabsf(y - rTransform->y) <= FLT_EPSILON) return true;
    } else {
        if (fabsf(x) <= FLT_EPSILON && fabsf(y) <= FLT_EPSILON) return true;
        rTransform = new RenderTransform();
        if (!rTransform) return false;
    }
    rTransform->x = x;
    rTransform->y = y;
    if (!rTransform->overriding) flag |= RenderUpdateFlag::Transform;

    return true;
}


bool Paint::Impl::render(RenderMethod& renderer)
{
    Compositor* cmp = nullptr;

    /* Note: only ClipPath is processed in update() step.
        Create a composition image. */
    if (cmpTarget && cmpMethod != CompositeMethod::ClipPath) {
        auto region = smethod->bounds(renderer);
        if (region.w == 0 || region.h == 0) return false;
        cmp = renderer.target(region);
        renderer.beginComposite(cmp, CompositeMethod::None, 255);
        cmpTarget->pImpl->render(renderer);
    }

    if (cmp) renderer.beginComposite(cmp, cmpMethod, cmpTarget->pImpl->opacity);

    auto ret = smethod->render(renderer);

    if (cmp) renderer.endComposite(cmp);

    return ret;
}


void* Paint::Impl::update(RenderMethod& renderer, const RenderTransform* pTransform, uint32_t opacity, Array<RenderData>& clips, uint32_t pFlag)
{
    if (flag & RenderUpdateFlag::Transform) {
        if (!rTransform) return nullptr;
        if (!rTransform->update()) {
            delete(rTransform);
            rTransform = nullptr;
        }
    }

    /* 1. Composition Pre Processing */
    void *cmpData = nullptr;
    RenderRegion viewport;
    bool cmpFastTrack = false;

    if (cmpTarget) {
        /* If transform has no rotation factors && ClipPath is a simple rectangle,
           we can avoid regular ClipPath sequence but use viewport for performance */
        if (cmpMethod == CompositeMethod::ClipPath) {
            RenderRegion viewport2;
            if ((cmpFastTrack = _clipPathFastTrack(cmpTarget, pTransform, cmpTarget->pImpl->rTransform, viewport2))) {
                viewport = renderer.viewport();
                viewport2.intersect(viewport);
                renderer.viewport(viewport2);
            }
        }

        if (!cmpFastTrack) {
            cmpData = cmpTarget->pImpl->update(renderer, pTransform, 255, clips, pFlag);
            if (cmpMethod == CompositeMethod::ClipPath) clips.push(cmpData);
        }
    }

    /* 2. Main Update */
    void *edata = nullptr;
    auto newFlag = static_cast<RenderUpdateFlag>(pFlag | flag);
    flag = RenderUpdateFlag::None;
    opacity = (opacity * this->opacity) / 255;

    if (rTransform && pTransform) {
        RenderTransform outTransform(pTransform, rTransform);
        edata = smethod->update(renderer, &outTransform, opacity, clips, newFlag);
    } else {
        auto outTransform = pTransform ? pTransform : rTransform;
        edata = smethod->update(renderer, outTransform, opacity, clips, newFlag);
    }

    /* 3. Composition Post Processing */
    if (cmpFastTrack) renderer.viewport(viewport);
    else if (cmpData && cmpMethod == CompositeMethod::ClipPath) clips.pop();

    return edata;
}


Paint :: Paint() : pImpl(new Impl())
{
}


Paint :: ~Paint()
{
    delete(pImpl);
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Result Paint::rotate(float degree) noexcept
{
    if (pImpl->rotate(degree)) return Result::Success;
    return Result::FailedAllocation;
}


Result Paint::scale(float factor) noexcept
{
    if (pImpl->scale(factor)) return Result::Success;
    return Result::FailedAllocation;
}


Result Paint::translate(float x, float y) noexcept
{
    if (pImpl->translate(x, y)) return Result::Success;
    return Result::FailedAllocation;
}


Result Paint::transform(const Matrix& m) noexcept
{
    if (pImpl->transform(m)) return Result::Success;
    return Result::FailedAllocation;
}

Matrix Paint::transform() noexcept
{
    auto pTransform = pImpl->transform();
    if (pTransform) return *pTransform;
    return {1, 0, 0, 0, 1, 0, 0, 0, 1};
}

Result Paint::bounds(float* x, float* y, float* w, float* h) const noexcept
{
    if (pImpl->bounds(x, y, w, h)) return Result::Success;
    return Result::InsufficientCondition;
}


Paint* Paint::duplicate() const noexcept
{
    return pImpl->duplicate();
}


Result Paint::composite(std::unique_ptr<Paint> target, CompositeMethod method) noexcept
{
    if (pImpl->composite(target.release(), method)) return Result::Success;
    return Result::InvalidArguments;
}


CompositeMethod Paint::composite(const Paint** target) const noexcept
{
    if (target) *target = pImpl->cmpTarget;

    return pImpl->cmpMethod;
}


Result Paint::opacity(uint8_t o) noexcept
{
    if (pImpl->opacity == o) return Result::Success;

    pImpl->opacity = o;
    pImpl->flag |= RenderUpdateFlag::Color;

    return Result::Success;
}


uint8_t Paint::opacity() const noexcept
{
    return pImpl->opacity;
}
