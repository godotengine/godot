/*
 * Copyright (c) 2023 - 2024 the ThorVG project. All rights reserved.

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

#include "tvgPaint.h"
#include "tvgFill.h"
#include "tvgLottieModel.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/



/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

LottieImage::~LottieImage()
{
    free(b64Data);
    free(mimeType);

    if (picture && PP(picture)->unref() == 0) {
        delete(picture);
    }
}


void LottieTrimpath::segment(float frameNo, float& start, float& end)
{
    auto s = this->start(frameNo) * 0.01f;
    auto e = this->end(frameNo) * 0.01f;
    auto o = fmodf(this->offset(frameNo), 360.0f) / 360.0f;  //0 ~ 1

    auto diff = fabs(s - e);
    if (mathZero(diff)) {
        start = 0.0f;
        end = 0.0f;
        return;
    }
    if (mathEqual(diff, 1.0f) || mathEqual(diff, 2.0f)) {
        start = 0.0f;
        end = 1.0f;
        return;
    }

    s += o;
    e += o;

    auto loop = true;

    //no loop
    if (s > 1.0f && e > 1.0f) loop = false;
    if (s < 0.0f && e < 0.0f) loop = false;
    if (s >= 0.0f && s <= 1.0f && e >= 0.0f  && e <= 1.0f) loop = false;

    if (s > 1.0f) s -= 1.0f;
    if (s < 0.0f) s += 1.0f;
    if (e > 1.0f) e -= 1.0f;
    if (e < 0.0f) e += 1.0f;

    if (loop) {
        start = s > e ? s : e;
        end = s < e ? s : e;
    } else {
        start = s < e ? s : e;
        end = s > e ? s : e;
    }
}


Fill* LottieGradient::fill(float frameNo)
{
    Fill* fill = nullptr;

    //Linear Graident
    if (id == 1) {
        fill = LinearGradient::gen().release();
        static_cast<LinearGradient*>(fill)->linear(start(frameNo).x, start(frameNo).y, end(frameNo).x, end(frameNo).y);
    }
    //Radial Gradient
    if (id == 2) {
        fill = RadialGradient::gen().release();

        auto sx = start(frameNo).x;
        auto sy = start(frameNo).y;
        auto ex = end(frameNo).x;
        auto ey = end(frameNo).y;
        auto w = fabsf(ex - sx);
        auto h = fabsf(ey - sy);
        auto r = (w > h) ? (w + 0.375f * h) : (h + 0.375f * w);
        auto progress = this->height(frameNo) * 0.01f;

        if (mathZero(progress)) {
            P(static_cast<RadialGradient*>(fill))->radial(sx, sy, r, sx, sy, 0.0f);
        } else {
            if (mathEqual(progress, 1.0f)) progress = 0.99f;
            auto startAngle = mathRad2Deg(atan2(ey - sy, ex - sx));
            auto angle = mathDeg2Rad((startAngle + this->angle(frameNo)));
            auto fx = sx + cos(angle) * progress * r;
            auto fy = sy + sin(angle) * progress * r;
            // Lottie dosen't have any focal radius concept
            P(static_cast<RadialGradient*>(fill))->radial(sx, sy, r, fx, fy, 0.0f);
        }
    }

    if (!fill) return nullptr;

    colorStops(frameNo, fill);

    return fill;
}


void LottieGroup::prepare(LottieObject::Type type)
{
    LottieObject::type = type;

    if (children.count == 0) return;

    size_t strokeCnt = 0;
    size_t fillCnt = 0;

    for (auto c = children.end() - 1; c >= children.begin(); --c) {
        auto child = static_cast<LottieObject*>(*c);

        if (child->type == LottieObject::Type::Trimpath) trimpath = true;

        /* Figure out if this group is a simple path drawing.
           In that case, the rendering context can be sharable with the parent's. */
        if (allowMerge && (child->type == LottieObject::Group || !child->mergeable())) allowMerge = false;

        if (reqFragment) continue;

        /* Figure out if the rendering context should be fragmented.
           Multiple stroking or grouping with a stroking would occur this.
           This fragment resolves the overlapped stroke outlines. */
        if (child->type == LottieObject::Group && !child->mergeable()) {
            if (strokeCnt > 0 || fillCnt > 0) reqFragment = true;
        } else if (child->type == LottieObject::SolidStroke || child->type == LottieObject::GradientStroke) {
            if (strokeCnt > 0) reqFragment = true;
            else ++strokeCnt;
        } else if (child->type == LottieObject::SolidFill || child->type == LottieObject::GradientFill) {
            if (fillCnt > 0) reqFragment = true;
            else ++fillCnt;
        }
    }

    //Reverse the drawing order if this group has a trimpath.
    if (!trimpath) return;

    for (uint32_t i = 0; i < children.count - 1; ) {
        auto child2 = children[i + 1];
        if (!child2->mergeable() || child2->type == LottieObject::Transform) {
            i += 2;
            continue;
        }
        auto child = children[i];
        if (!child->mergeable() || child->type == LottieObject::Transform) {
            i++;
            continue;
        }
        children[i] = child2;
        children[i + 1] = child;
        i++;
    }
}


LottieLayer::~LottieLayer()
{
    if (refId) {
        //No need to free assets children because the Composition owns them.
        children.clear();
        free(refId);
    }

    for (auto m = masks.begin(); m < masks.end(); ++m) {
        delete(*m);
    }

    delete(matte.target);
    delete(transform);
}

void LottieLayer::prepare()
{
    /* if layer is hidden, only useful data is its transform matrix.
       so force it to be a Null Layer and release all resource. */
    if (hidden) {
        type = LottieLayer::Null;
        children.reset();
        return;
    }
    LottieGroup::prepare(LottieObject::Layer);
}


float LottieLayer::remap(float frameNo)
{
    if (timeRemap.frames || timeRemap.value) {
        frameNo = comp->frameAtTime(timeRemap(frameNo));
    } else {
        frameNo -= startFrame;
    }
    return (frameNo / timeStretch);
}


LottieComposition::~LottieComposition()
{
    if (!initiated && root) delete(root->scene);

    delete(root);
    free(version);
    free(name);

    //delete interpolators
    for (auto i = interpolators.begin(); i < interpolators.end(); ++i) {
        free((*i)->key);
        free(*i);
    }

    //delete assets
    for (auto a = assets.begin(); a < assets.end(); ++a) {
        delete(*a);
    }

    //delete fonts
    for (auto f = fonts.begin(); f < fonts.end(); ++f) {
        delete(*f);
    }

    //delete slots
    for (auto s = slots.begin(); s < slots.end(); ++s) {
        delete(*s);
    }
    
    for (auto m = markers.begin(); m < markers.end(); ++m) {
        delete(*m);
    }
}
