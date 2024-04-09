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
#include "tvgPaint.h"
#include "tvgShape.h"
#include "tvgPicture.h"
#include "tvgScene.h"
#include "tvgText.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

#define PAINT_METHOD(ret, METHOD) \
    switch (id) { \
        case TVG_CLASS_ID_SHAPE: ret = P((Shape*)paint)->METHOD; break; \
        case TVG_CLASS_ID_SCENE: ret = P((Scene*)paint)->METHOD; break; \
        case TVG_CLASS_ID_PICTURE: ret = P((Picture*)paint)->METHOD; break; \
        case TVG_CLASS_ID_TEXT: ret = P((Text*)paint)->METHOD; break; \
        default: ret = {}; \
    }



static Result _compFastTrack(Paint* cmpTarget, const RenderTransform* pTransform, RenderTransform* rTransform, RenderRegion& viewport)
{
    /* Access Shape class by Paint is bad... but it's ok still it's an internal usage. */
    auto shape = static_cast<Shape*>(cmpTarget);

    //Rectangle Candidates?
    const Point* pts;
    auto ptsCnt = shape->pathCoords(&pts);

    //nothing to clip
    if (ptsCnt == 0) return Result::InvalidArguments;

    if (ptsCnt != 4) return Result::InsufficientCondition;

    if (rTransform) rTransform->update();

    //No rotation and no skewing
    if (pTransform && (!mathRightAngle(&pTransform->m) || mathSkewed(&pTransform->m))) return Result::InsufficientCondition;
    if (rTransform && (!mathRightAngle(&rTransform->m) || mathSkewed(&rTransform->m))) return Result::InsufficientCondition;

    //Perpendicular Rectangle?
    auto pt1 = pts + 0;
    auto pt2 = pts + 1;
    auto pt3 = pts + 2;
    auto pt4 = pts + 3;

    if ((mathEqual(pt1->x, pt2->x) && mathEqual(pt2->y, pt3->y) && mathEqual(pt3->x, pt4->x) && mathEqual(pt1->y, pt4->y)) ||
        (mathEqual(pt2->x, pt3->x) && mathEqual(pt1->y, pt2->y) && mathEqual(pt1->x, pt4->x) && mathEqual(pt3->y, pt4->y))) {

        auto v1 = *pt1;
        auto v2 = *pt3;

        if (rTransform) {
            mathMultiply(&v1, &rTransform->m);
            mathMultiply(&v2, &rTransform->m);
        }

        if (pTransform) {
            mathMultiply(&v1, &pTransform->m);
            mathMultiply(&v2, &pTransform->m);
        }

        //sorting
        if (v1.x > v2.x) {
            auto tmp = v2.x;
            v2.x = v1.x;
            v1.x = tmp;
        }

        if (v1.y > v2.y) {
            auto tmp = v2.y;
            v2.y = v1.y;
            v1.y = tmp;
        }

        viewport.x = static_cast<int32_t>(v1.x);
        viewport.y = static_cast<int32_t>(v1.y);
        viewport.w = static_cast<int32_t>(ceil(v2.x - viewport.x));
        viewport.h = static_cast<int32_t>(ceil(v2.y - viewport.y));

        if (viewport.w < 0) viewport.w = 0;
        if (viewport.h < 0) viewport.h = 0;

        return Result::Success;
    }
    return Result::InsufficientCondition;
}


RenderRegion Paint::Impl::bounds(RenderMethod* renderer) const
{
    RenderRegion ret;
    PAINT_METHOD(ret, bounds(renderer));
    return ret;
}


Iterator* Paint::Impl::iterator()
{
    Iterator* ret;
    PAINT_METHOD(ret, iterator());
    return ret;
}


Paint* Paint::Impl::duplicate()
{
    Paint* ret;
    PAINT_METHOD(ret, duplicate());

    //duplicate Transform
    if (rTransform) {
        ret->pImpl->rTransform = new RenderTransform();
        *ret->pImpl->rTransform = *rTransform;
        ret->pImpl->renderFlag |= RenderUpdateFlag::Transform;
    }

    ret->pImpl->opacity = opacity;

    if (compData) ret->pImpl->composite(ret, compData->target->duplicate(), compData->method);

    return ret;
}


bool Paint::Impl::rotate(float degree)
{
    if (rTransform) {
        if (mathEqual(degree, rTransform->degree)) return true;
    } else {
        if (mathZero(degree)) return true;
        rTransform = new RenderTransform();
    }
    rTransform->degree = degree;
    if (!rTransform->overriding) renderFlag |= RenderUpdateFlag::Transform;

    return true;
}


bool Paint::Impl::scale(float factor)
{
    if (rTransform) {
        if (mathEqual(factor, rTransform->scale)) return true;
    } else {
        if (mathEqual(factor, 1.0f)) return true;
        rTransform = new RenderTransform();
    }
    rTransform->scale = factor;
    if (!rTransform->overriding) renderFlag |= RenderUpdateFlag::Transform;

    return true;
}


bool Paint::Impl::translate(float x, float y)
{
    if (rTransform) {
        if (mathEqual(x, rTransform->x) && mathEqual(y, rTransform->y)) return true;
    } else {
        if (mathZero(x) && mathZero(y)) return true;
        rTransform = new RenderTransform();
    }
    rTransform->x = x;
    rTransform->y = y;
    if (!rTransform->overriding) renderFlag |= RenderUpdateFlag::Transform;

    return true;
}


bool Paint::Impl::render(RenderMethod* renderer)
{
    Compositor* cmp = nullptr;

    /* Note: only ClipPath is processed in update() step.
        Create a composition image. */
    if (compData && compData->method != CompositeMethod::ClipPath && !(compData->target->pImpl->ctxFlag & ContextFlag::FastTrack)) {
        RenderRegion region;
        PAINT_METHOD(region, bounds(renderer));

        if (MASK_REGION_MERGING(compData->method)) region.add(P(compData->target)->bounds(renderer));
        if (region.w == 0 || region.h == 0) return true;
        cmp = renderer->target(region, COMPOSITE_TO_COLORSPACE(renderer, compData->method));
        if (renderer->beginComposite(cmp, CompositeMethod::None, 255)) {
            compData->target->pImpl->render(renderer);
        }
    }

    if (cmp) renderer->beginComposite(cmp, compData->method, compData->target->pImpl->opacity);

    renderer->blend(blendMethod);

    bool ret;
    PAINT_METHOD(ret, render(renderer));

    if (cmp) renderer->endComposite(cmp);

    return ret;
}


RenderData Paint::Impl::update(RenderMethod* renderer, const RenderTransform* pTransform, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag pFlag, bool clipper)
{
    if (this->renderer != renderer) {
        if (this->renderer) TVGERR("RENDERER", "paint's renderer has been changed!");
        renderer->ref();
        this->renderer = renderer;
    }

    if (renderFlag & RenderUpdateFlag::Transform) {
        if (!rTransform) return nullptr;
        rTransform->update();
    }

    /* 1. Composition Pre Processing */
    RenderData trd = nullptr;                 //composite target render data
    RenderRegion viewport;
    Result compFastTrack = Result::InsufficientCondition;
    bool childClipper = false;

    if (compData) {
        auto target = compData->target;
        auto method = compData->method;
        target->pImpl->ctxFlag &= ~ContextFlag::FastTrack;   //reset

        /* If the transformation has no rotational factors and the ClipPath/Alpha(InvAlpha)Masking involves a simple rectangle,
           we can optimize by using the viewport instead of the regular ClipPath/AlphaMasking sequence for improved performance. */
        auto tryFastTrack = false;
        if (target->identifier() == TVG_CLASS_ID_SHAPE) {
            if (method == CompositeMethod::ClipPath) tryFastTrack = true;
            else {
                auto shape = static_cast<Shape*>(target);
                uint8_t a;
                shape->fillColor(nullptr, nullptr, nullptr, &a);
                //no gradient fill & no compositions of the composition target.
                if (!shape->fill() && !(PP(shape)->compData)) {
                    if (method == CompositeMethod::AlphaMask && a == 255 && PP(shape)->opacity == 255) tryFastTrack = true;
                    else if (method == CompositeMethod::InvAlphaMask && (a == 0 || PP(shape)->opacity == 0)) tryFastTrack = true;
                }
            }
            if (tryFastTrack) {
                RenderRegion viewport2;
                if ((compFastTrack = _compFastTrack(target, pTransform, target->pImpl->rTransform, viewport2)) == Result::Success) {
                    viewport = renderer->viewport();
                    viewport2.intersect(viewport);
                    renderer->viewport(viewport2);
                    target->pImpl->ctxFlag |= ContextFlag::FastTrack;
                }
            }
        }
        if (compFastTrack == Result::InsufficientCondition) {
            childClipper = compData->method == CompositeMethod::ClipPath ? true : false;
            trd = target->pImpl->update(renderer, pTransform, clips, 255, pFlag, childClipper);
            if (childClipper) clips.push(trd);
        }
    }

    /* 2. Main Update */
    auto newFlag = static_cast<RenderUpdateFlag>(pFlag | renderFlag);
    renderFlag = RenderUpdateFlag::None;
    opacity = MULTIPLY(opacity, this->opacity);

    RenderData rd = nullptr;
    RenderTransform outTransform(pTransform, rTransform);
    PAINT_METHOD(rd, update(renderer, &outTransform, clips, opacity, newFlag, clipper));

    /* 3. Composition Post Processing */
    if (compFastTrack == Result::Success) renderer->viewport(viewport);
    else if (childClipper) clips.pop();

    return rd;
}


bool Paint::Impl::bounds(float* x, float* y, float* w, float* h, bool transformed, bool stroking)
{
    Matrix* m = nullptr;
    bool ret;

    //Case: No transformed, quick return!
    if (!transformed || !(m = this->transform())) {
        PAINT_METHOD(ret, bounds(x, y, w, h, stroking));
        return ret;
    }

    //Case: Transformed
    auto tx = 0.0f;
    auto ty = 0.0f;
    auto tw = 0.0f;
    auto th = 0.0f;

    PAINT_METHOD(ret, bounds(&tx, &ty, &tw, &th, stroking));

    //Get vertices
    Point pt[4] = {{tx, ty}, {tx + tw, ty}, {tx + tw, ty + th}, {tx, ty + th}};

    //New bounding box
    auto x1 = FLT_MAX;
    auto y1 = FLT_MAX;
    auto x2 = -FLT_MAX;
    auto y2 = -FLT_MAX;

    //Compute the AABB after transformation
    for (int i = 0; i < 4; i++) {
        mathMultiply(&pt[i], m);

        if (pt[i].x < x1) x1 = pt[i].x;
        if (pt[i].x > x2) x2 = pt[i].x;
        if (pt[i].y < y1) y1 = pt[i].y;
        if (pt[i].y > y2) y2 = pt[i].y;
    }

    if (x) *x = x1;
    if (y) *y = y1;
    if (w) *w = x2 - x1;
    if (h) *h = y2 - y1;

    return ret;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Paint :: Paint() : pImpl(new Impl(this))
{
}


Paint :: ~Paint()
{
    delete(pImpl);
}


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


TVG_DEPRECATED Result Paint::bounds(float* x, float* y, float* w, float* h) const noexcept
{
    return this->bounds(x, y, w, h, false);
}


Result Paint::bounds(float* x, float* y, float* w, float* h, bool transform) const noexcept
{
    if (pImpl->bounds(x, y, w, h, transform, true)) return Result::Success;
    return Result::InsufficientCondition;
}


Paint* Paint::duplicate() const noexcept
{
    return pImpl->duplicate();
}


Result Paint::composite(std::unique_ptr<Paint> target, CompositeMethod method) noexcept
{
    auto p = target.release();
    if (pImpl->composite(this, p, method)) return Result::Success;
    delete(p);
    return Result::InvalidArguments;
}


CompositeMethod Paint::composite(const Paint** target) const noexcept
{
    if (pImpl->compData) {
        if (target) *target = pImpl->compData->target;
        return pImpl->compData->method;
    } else {
        if (target) *target = nullptr;
        return CompositeMethod::None;
    }
}


Result Paint::opacity(uint8_t o) noexcept
{
    if (pImpl->opacity == o) return Result::Success;

    pImpl->opacity = o;
    pImpl->renderFlag |= RenderUpdateFlag::Color;

    return Result::Success;
}


uint8_t Paint::opacity() const noexcept
{
    return pImpl->opacity;
}


uint32_t Paint::identifier() const noexcept
{
    return pImpl->id;
}


Result Paint::blend(BlendMethod method) const noexcept
{
    if (pImpl->blendMethod != method) {
        pImpl->blendMethod = method;
        pImpl->renderFlag |= RenderUpdateFlag::Blend;
    }

    return Result::Success;
}


BlendMethod Paint::blend() const noexcept
{
    return pImpl->blendMethod;
}
