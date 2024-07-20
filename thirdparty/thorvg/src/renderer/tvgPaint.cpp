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


static Result _clipRect(RenderMethod* renderer, const Point* pts, const RenderTransform* pTransform, RenderTransform* rTransform, RenderRegion& before)
{
    //sorting
    Point tmp[4];
    Point min = {FLT_MAX, FLT_MAX};
    Point max = {0.0f, 0.0f};

    for (int i = 0; i < 4; ++i) {
        tmp[i] = pts[i];
        if (rTransform) tmp[i] *= rTransform->m;
        if (pTransform) tmp[i] *= pTransform->m;
        if (tmp[i].x < min.x) min.x = tmp[i].x;
        if (tmp[i].x > max.x) max.x = tmp[i].x;
        if (tmp[i].y < min.y) min.y = tmp[i].y;
        if (tmp[i].y > max.y) max.y = tmp[i].y;
    }

    float region[4] = {float(before.x), float(before.x + before.w), float(before.y), float(before.y + before.h)};

    //figure out if the clipper is a superset of the current viewport(before) region
    if (min.x <= region[0] && max.x >= region[1] && min.y <= region[2] && max.y >= region[3]) {
        //viewport region is same, nothing to do.
        return Result::Success;
    //figure out if the clipper is totally outside of the viewport
    } else if (max.x <= region[0] || min.x >= region[1] || max.y <= region[2] || min.y >= region[3]) {
        renderer->viewport({0, 0, 0, 0});
        return Result::Success;
    }
    return Result::InsufficientCondition;
}


static Result _compFastTrack(RenderMethod* renderer, Paint* cmpTarget, const RenderTransform* pTransform, RenderTransform* rTransform, RenderRegion& before)
{
    /* Access Shape class by Paint is bad... but it's ok still it's an internal usage. */
    auto shape = static_cast<Shape*>(cmpTarget);

    //Rectangle Candidates?
    const Point* pts;
    auto ptsCnt = shape->pathCoords(&pts);

    //nothing to clip
    if (ptsCnt == 0) return Result::InvalidArguments;

    if (ptsCnt != 4) return Result::InsufficientCondition;

    if (rTransform && (cmpTarget->pImpl->renderFlag & RenderUpdateFlag::Transform)) rTransform->update();

    //No rotation and no skewing, still can try out clipping the rect region.
    auto tryClip = false;

    if (pTransform && (!mathRightAngle(&pTransform->m) || mathSkewed(&pTransform->m))) tryClip = true;
    if (rTransform && (!mathRightAngle(&rTransform->m) || mathSkewed(&rTransform->m))) tryClip = true;

    if (tryClip) return _clipRect(renderer, pts, pTransform, rTransform, before);

    //Perpendicular Rectangle?
    auto pt1 = pts + 0;
    auto pt2 = pts + 1;
    auto pt3 = pts + 2;
    auto pt4 = pts + 3;

    if ((mathEqual(pt1->x, pt2->x) && mathEqual(pt2->y, pt3->y) && mathEqual(pt3->x, pt4->x) && mathEqual(pt1->y, pt4->y)) ||
        (mathEqual(pt2->x, pt3->x) && mathEqual(pt1->y, pt2->y) && mathEqual(pt1->x, pt4->x) && mathEqual(pt3->y, pt4->y))) {

        RenderRegion after;

        auto v1 = *pt1;
        auto v2 = *pt3;

        if (rTransform) {
            v1 *= rTransform->m;
            v2 *= rTransform->m;
        }

        if (pTransform) {
            v1 *= pTransform->m;
            v2 *= pTransform->m;
        }

        //sorting
        if (v1.x > v2.x) std::swap(v1.x, v2.x);
        if (v1.y > v2.y) std::swap(v1.y, v2.y);

        after.x = static_cast<int32_t>(v1.x);
        after.y = static_cast<int32_t>(v1.y);
        after.w = static_cast<int32_t>(ceil(v2.x - after.x));
        after.h = static_cast<int32_t>(ceil(v2.y - after.y));

        if (after.w < 0) after.w = 0;
        if (after.h < 0) after.h = 0;

        after.intersect(before);
        renderer->viewport(after);

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
        if (rTransform->overriding) return false;
        if (mathEqual(degree, rTransform->degree)) return true;
    } else {
        if (mathZero(degree)) return true;
        rTransform = new RenderTransform();
    }
    rTransform->degree = degree;
    renderFlag |= RenderUpdateFlag::Transform;

    return true;
}


bool Paint::Impl::scale(float factor)
{
    if (rTransform) {
        if (rTransform->overriding) return false;
        if (mathEqual(factor, rTransform->scale)) return true;
    } else {
        if (mathEqual(factor, 1.0f)) return true;
        rTransform = new RenderTransform();
    }
    rTransform->scale = factor;
    renderFlag |= RenderUpdateFlag::Transform;

    return true;
}


bool Paint::Impl::translate(float x, float y)
{
    if (rTransform) {
        if (rTransform->overriding) return false;
        if (mathEqual(x, rTransform->m.e13) && mathEqual(y, rTransform->m.e23)) return true;
    } else {
        if (mathZero(x) && mathZero(y)) return true;
        rTransform = new RenderTransform();
    }
    rTransform->m.e13 = x;
    rTransform->m.e23 = y;
    renderFlag |= RenderUpdateFlag::Transform;

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

    if (renderFlag & RenderUpdateFlag::Transform) rTransform->update();

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
                viewport = renderer->viewport();
                if ((compFastTrack = _compFastTrack(renderer, target, pTransform, target->pImpl->rTransform, viewport)) == Result::Success) {
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
        pt[i] *= *m;

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
    return Result::InsufficientCondition;
}


Result Paint::scale(float factor) noexcept
{
    if (pImpl->scale(factor)) return Result::Success;
    return Result::InsufficientCondition;
}


Result Paint::translate(float x, float y) noexcept
{
    if (pImpl->translate(x, y)) return Result::Success;
    return Result::InsufficientCondition;
}


Result Paint::transform(const Matrix& m) noexcept
{
    if (pImpl->transform(m)) return Result::Success;
    return Result::InsufficientCondition;
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
