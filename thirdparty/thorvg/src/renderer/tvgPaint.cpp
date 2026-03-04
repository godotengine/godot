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
    switch (paint->type()) { \
        case Type::Shape: ret = to<ShapeImpl>(paint)->METHOD; break; \
        case Type::Scene: ret = to<SceneImpl>(paint)->METHOD; break; \
        case Type::Picture: ret = to<PictureImpl>(paint)->METHOD; break; \
        case Type::Text: ret = to<TextImpl>(paint)->METHOD; break; \
        default: ret = {}; \
    }


static bool _clipRect(RenderMethod* renderer, const Point* pts, const Matrix& m, RenderRegion& before)
{
    Point c[4];  //corners
    for (int i = 0; i < 4; ++i) {
        c[i] = pts[i] * m;
    }

    //figure out if the clipper is a superset of the current viewport(before) region
    auto pointInConvexQuad = [](const Point& p, const Point* quad) {
        auto sign = [](const Point& p1, const Point& p2, const Point& p3) {
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
        };
        auto b1 = sign(p, quad[0], quad[1]) < 0.0f;
        auto b2 = sign(p, quad[1], quad[2]) < 0.0f;
        auto b3 = sign(p, quad[2], quad[3]) < 0.0f;
        auto b4 = sign(p, quad[3], quad[0]) < 0.0f;
        return ((b1 == b2) && (b2 == b3) && (b3 == b4));
    };

    if (!pointInConvexQuad({float(before.min.x), float(before.min.y)}, c)) return false;
    if (!pointInConvexQuad({float(before.max.x), float(before.min.y)}, c)) return false;
    if (!pointInConvexQuad({float(before.max.x), float(before.max.y)}, c)) return false;
    if (!pointInConvexQuad({float(before.min.x), float(before.max.y)}, c)) return false;

    //same viewport
    return true;
}


static bool _compFastTrack(RenderMethod* renderer, Paint* cmpTarget, const Matrix& pm, RenderRegion& before)
{
    /* Access Shape class by Paint is bad... but it's ok still it's an internal usage. */
    auto shape = static_cast<Shape*>(cmpTarget);

    //Trimming likely makes the shape non-rectangular
    if (to<ShapeImpl>(shape)->rs.trimpath()) return false;

    //Rectangle Candidates?
    const Point* pts;
    uint32_t ptsCnt;
    shape->path(nullptr, nullptr, &pts, &ptsCnt);

    //No rectangle format
    if (ptsCnt != 4) return false;

    //No rotation and no skewing, still can try out clipping the rect region.
    auto tm = pm * cmpTarget->transform();

    //Perpendicular Rectangle?
    if (rightAngle(tm) && !skewed(tm)) {
        auto pt1 = pts + 0;
        auto pt2 = pts + 1;
        auto pt3 = pts + 2;
        auto pt4 = pts + 3;

        if ((tvg::equal(pt1->x, pt2->x) && tvg::equal(pt2->y, pt3->y) && tvg::equal(pt3->x, pt4->x) && tvg::equal(pt1->y, pt4->y)) ||
            (tvg::equal(pt2->x, pt3->x) && tvg::equal(pt1->y, pt2->y) && tvg::equal(pt1->x, pt4->x) && tvg::equal(pt3->y, pt4->y))) {

            RenderRegion after;

            auto v1 = *pt1;
            auto v2 = *pt3;
            v1 *= tm;
            v2 *= tm;

            //sorting
            if (v1.x > v2.x) std::swap(v1.x, v2.x);
            if (v1.y > v2.y) std::swap(v1.y, v2.y);

            after.min.x = static_cast<int32_t>(nearbyint(v1.x));
            after.min.y = static_cast<int32_t>(nearbyint(v1.y));
            after.max.x = static_cast<int32_t>(nearbyint(v2.x));
            after.max.y = static_cast<int32_t>(nearbyint(v2.y));

            if (after.max.x < after.min.x) after.max.x = after.min.x;
            if (after.max.y < after.min.y) after.max.y = after.min.y;

            after.intersect(before);
            renderer->viewport(after);
            return true;
        }
    }
    return _clipRect(renderer, pts, tm, before);
}


RenderRegion Paint::Impl::bounds()
{
    RenderRegion ret;
    PAINT_METHOD(ret, bounds());
    return ret;
}


Iterator* Paint::Impl::iterator()
{
    Iterator* ret;
    PAINT_METHOD(ret, iterator());
    return ret;
}


Paint* Paint::Impl::duplicate(Paint* ret)
{
    if (ret) ret->mask(nullptr, MaskMethod::None);

    PAINT_METHOD(ret, duplicate(ret));

    if (maskData) ret->mask(maskData->target->duplicate(), maskData->method);
    if (clipper) ret->clip(static_cast<Shape*>(clipper->duplicate()));

    ret->pImpl->tr = tr;
    ret->pImpl->blendMethod = blendMethod;
    ret->pImpl->opacity = opacity;
    ret->pImpl->hidden = hidden;
    ret->pImpl->mark(RenderUpdateFlag::All);

    return ret;
}


bool Paint::Impl::render(RenderMethod* renderer)
{
    if (hidden || opacity == 0) return true;

    RenderCompositor* cmp = nullptr;

    //OPTIMIZE: bounds(renderer) calls could dismiss the parallelization
    if (maskData && !(maskData->target->pImpl->ctxFlag & ContextFlag::FastTrack)) {
        RenderRegion region;
        PAINT_METHOD(region, bounds());

        auto mData = maskData;
        while (mData) {
            if (MASK_REGION_MERGING(mData->method)) region.add(PAINT(mData->target)->bounds());
            if (region.invalid()) return true;
            mData = PAINT(mData->target)->maskData;
        }
        cmp = renderer->target(region, MASK_TO_COLORSPACE(renderer, maskData->method), CompositionFlag::Masking);
        if (renderer->beginComposite(cmp, MaskMethod::None, 255)) {
            maskData->target->pImpl->render(renderer);
        }
    }

    if (cmp) renderer->beginComposite(cmp, maskData->method, maskData->target->pImpl->opacity);

    bool ret;
    PAINT_METHOD(ret, render(renderer));

    if (cmp) renderer->endComposite(cmp);

    return ret;
}


RenderData Paint::Impl::update(RenderMethod* renderer, const Matrix& pm, Array<RenderData>& clips, uint8_t opacity, RenderUpdateFlag flag, bool clipper)
{
    bool ret;
    PAINT_METHOD(ret, skip((flag | renderFlag)));

    if (ret) return rd;

    cmpFlag = CompositionFlag::Invalid;  //must clear after the rendering

    if (this->renderer != renderer) {
        if (this->renderer) TVGERR("RENDERER", "paint's renderer has been changed!");
        renderer->ref();
        this->renderer = renderer;
    }

    if (renderFlag & RenderUpdateFlag::Transform) tr.update();

    /* 1. Composition Pre Processing */
    RenderData trd = nullptr;                 //composite target render data
    RenderRegion viewport;
    auto compFastTrack = false;

    if (maskData) {
        auto target = maskData->target;
        auto method = maskData->method;
        PAINT(target)->ctxFlag &= ~ContextFlag::FastTrack;   //reset

        /* If the transformation has no rotational factors and the Alpha(InvAlpha) Masking involves a simple rectangle,
           we can optimize by using the viewport instead of the regular Alphaing sequence for improved performance. */
        if (target->type() == Type::Shape) {
            auto shape = static_cast<Shape*>(target);
            uint8_t a;
            shape->fill(nullptr, nullptr, nullptr, &a);
            //no gradient fill & no maskings of the masking target.
            if (!shape->fill() && !(PAINT(shape)->maskData)) {
                if ((method == MaskMethod::Alpha && a == 255 && PAINT(shape)->opacity == 255) || (method == MaskMethod::InvAlpha && (a == 0 || PAINT(shape)->opacity == 0))) {
                    viewport = renderer->viewport();
                    if ((compFastTrack = _compFastTrack(renderer, target, pm, viewport))) {
                        PAINT(target)->ctxFlag |= ContextFlag::FastTrack;
                    }
                }
            }
        }
        if (!compFastTrack) {
            trd = PAINT(target)->update(renderer, pm, clips, 255, flag, false);
        }
    }

    /* 2. Clipping */
    if (this->clipper) {
        auto pclip = PAINT(this->clipper);
        pclip->ctxFlag &= ~ContextFlag::FastTrack;   //reset
        viewport = renderer->viewport();
        if (!pclip->clipper && to<ShapeImpl>(this->clipper)->rs.strokeWidth() == 0.0f && _compFastTrack(renderer, this->clipper, pm, viewport)) {
            pclip->ctxFlag |= ContextFlag::FastTrack;
            compFastTrack = true;
        } else {
            mark(RenderUpdateFlag::Clip);
            trd = pclip->update(renderer, pm, clips, 255, flag, true);
            clips.push(trd);
        }
    }

    /* 3. Main Update */
    opacity = MULTIPLY(opacity, this->opacity);
    PAINT_METHOD(ret, update(renderer, pm * tr.m, clips, opacity, (flag | renderFlag), clipper));

    /* 4. Composition Post Processing */
    if (compFastTrack) renderer->viewport(viewport);
    else if (this->clipper) clips.pop();

    renderFlag = RenderUpdateFlag::None;

    return rd;
}


bool Paint::Impl::bounds(Point* pt4, const Matrix* pm, bool obb)
{
    bool ret;
    PAINT_METHOD(ret, bounds(pt4, pm * transform(), obb));
    return ret;
}


bool Paint::Impl::intersects(const RenderRegion& region)
{
    if (renderer) {
        bool ret;
        PAINT_METHOD(ret, intersects(region));
        return ret;
    }
    return false;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

Paint :: Paint() = default;
Paint :: ~Paint() = default;


void Paint::rel(Paint* paint) noexcept
{
    if (paint && paint->refCnt() <= 0) delete(paint);
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


Matrix& Paint::transform() noexcept
{
    return pImpl->transform();
}


Result Paint::bounds(float* x, float* y, float* w, float* h) noexcept
{
    Point pt4[4] = {};
    const auto pm = pImpl->ptransform();
    if (pImpl->bounds(pt4, &pm, false)) {
        BBox box = {{FLT_MAX, FLT_MAX}, {-FLT_MAX, -FLT_MAX}};
        for (int i = 0; i < 4; ++i) {
            if (pt4[i].x < box.min.x) box.min.x = pt4[i].x;
            if (pt4[i].x > box.max.x) box.max.x = pt4[i].x;
            if (pt4[i].y < box.min.y) box.min.y = pt4[i].y;
            if (pt4[i].y > box.max.y) box.max.y = pt4[i].y;
        }
        if (x) *x = box.min.x;
        if (y) *y = box.min.y;
        if (w) *w = box.max.x - box.min.x;
        if (h) *h = box.max.y - box.min.y;
        return Result::Success;
    }
    return Result::InsufficientCondition;
}


Result Paint::bounds(Point* pt4) noexcept
{
    if (!pt4) return Result::InvalidArguments;
    auto pm = pImpl->ptransform();
    if (pImpl->bounds(pt4, &pm, true)) return Result::Success;
    return Result::InsufficientCondition;
}


bool Paint::intersects(int32_t x, int32_t y, int32_t w, int32_t h) noexcept
{
    if (w <= 0 || h <= 0) return false;
    return pImpl->intersects({{x, y}, {x + w, y + h}});
}


Paint* Paint::duplicate() const noexcept
{
    return pImpl->duplicate();
}


Result Paint::clip(Shape* clipper) noexcept
{
    return pImpl->clip(clipper);
}


Shape* Paint::clip() const noexcept
{
    return pImpl->clipper;
}


Result Paint::mask(Paint* target, MaskMethod method) noexcept
{
    if (method > MaskMethod::Darken) return Result::InvalidArguments;
    return pImpl->mask(target, method);
}


MaskMethod Paint::mask(const Paint** target) const noexcept
{
    return pImpl->mask(target);
}


Result Paint::opacity(uint8_t o) noexcept
{
    if (pImpl->opacity != o) {
        pImpl->opacity = o;
        pImpl->mark(RenderUpdateFlag::Color);
    }
    return Result::Success;
}


uint8_t Paint::opacity() const noexcept
{
    return pImpl->opacity;
}


Result Paint::blend(BlendMethod method) noexcept
{
    if (method <= BlendMethod::Add || method == BlendMethod::Composition) {
        pImpl->blend(method);
        return Result::Success;
    }
    return Result::InvalidArguments;
}


uint16_t Paint::ref() noexcept
{
    return pImpl->ref();
}


uint16_t Paint::unref(bool free) noexcept
{
    return pImpl->unrefx(free);
}


uint16_t Paint::refCnt() const noexcept
{
    return pImpl->refCnt;
}


const Paint* Paint::parent() const noexcept
{
    return pImpl->parent;
}


Result Paint::visible(bool on) noexcept
{
    return pImpl->visible(!on);
}


bool Paint::visible() const noexcept
{
    return !pImpl->hidden;
}
