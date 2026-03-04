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

#ifndef _TVG_FILL_H_
#define _TVG_FILL_H_

#include "tvgCommon.h"
#include "tvgMath.h"

#define LINEAR(A) static_cast<LinearGradientImpl*>(A)
#define CONST_LINEAR(A) static_cast<const LinearGradientImpl*>(A)

#define RADIAL(A) static_cast<RadialGradientImpl*>(A)
#define CONST_RADIAL(A) static_cast<const RadialGradientImpl*>(A)

struct Fill::Impl
{
    ColorStop* colorStops = nullptr;
    Matrix transform = tvg::identity();
    uint16_t cnt = 0;
    FillSpread spread = FillSpread::Pad;

    virtual ~Impl()
    {
        tvg::free(colorStops);
    }

    void copy(const Fill::Impl& dup)
    {
        cnt = dup.cnt;
        spread = dup.spread;
        colorStops = tvg::malloc<ColorStop>(sizeof(ColorStop) * dup.cnt);
        if (dup.cnt > 0) memcpy(colorStops, dup.colorStops, sizeof(ColorStop) * dup.cnt);
        transform = dup.transform;
    }

    Result update(const ColorStop* colorStops, uint32_t cnt)
    {
        if ((!colorStops && cnt > 0) || (colorStops && cnt == 0)) return Result::InvalidArguments;

        if (cnt == 0) {
            if (this->colorStops) {
                tvg::free(this->colorStops);
                this->colorStops = nullptr;
                this->cnt = 0;
            }
            return Result::Success;
        }

        if (cnt != this->cnt) {
            this->colorStops = tvg::realloc<ColorStop>(this->colorStops, cnt * sizeof(ColorStop));
        }

        this->cnt = cnt;
        memcpy(this->colorStops, colorStops, cnt * sizeof(ColorStop));

        return Result::Success;
    }
};


struct RadialGradientImpl : RadialGradient
{
    Fill::Impl impl;
    Point center{}, focal{};
    float r = 0.0f, fr = 0.0f;

    RadialGradientImpl()
    {
        Fill::pImpl = &impl;
    }

    Fill* duplicate() const
    {
        auto ret = RadialGradient::gen();
        RADIAL(ret)->impl.copy(this->impl);
        RADIAL(ret)->center = center;
        RADIAL(ret)->r = r;
        RADIAL(ret)->focal = focal;
        RADIAL(ret)->fr = fr;

        return ret;
    }

    Result radial(float cx, float cy, float r, float fx, float fy, float fr)
    {
        if (r < 0 || fr < 0) return Result::InvalidArguments;

        this->center = {cx, cy};
        this->r = r;
        this->focal = {fx, fy};
        this->fr = fr;

        return Result::Success;
    }

    Result radial(float* cx, float* cy, float* r, float* fx, float* fy, float* fr) const
    {
        if (cx) *cx = center.x;
        if (cy) *cy = center.y;
        if (r) *r = this->r;
        if (fx) *fx = focal.x;
        if (fy) *fy = focal.y;
        if (fr) *fr = this->fr;

        return Result::Success;
    }

    //TODO: remove this logic once SVG 2.0 is adopted by sw and wg engines (gl already supports it); lottie-specific handling will then be delegated entirely to the loader
    //clamp focal point and shrink start circle if needed to avoid invalid gradient setup
    bool correct(float& fx, float& fy, float& fr) const
    {
        constexpr float PRECISION = 0.01f;
        if (r < PRECISION) return false;  // too small, treated as solid fill

        auto dist = tvg::length(center, focal);

        // clamp focal point to inside end circle if outside
        if (this->r - dist <  PRECISION) {
            auto diff = center - focal;
            if (dist < PRECISION) dist = diff.x = PRECISION;
            auto scale = this->r * (1.0f - PRECISION) / dist;
            diff *= scale;
            dist *= scale;  // update effective dist after scaling
            fx = center.x - diff.x;
            fy = center.y - diff.y;
        } else {
            fx = focal.x;
            fy = focal.y;
        }
        // ensure start circle radius fr doesn't exceed the difference
        auto maxFr = (r - dist) * (1.0f - PRECISION);
        fr = (this->fr > maxFr) ? std::max(0.0f, maxFr) : this->fr;
        return true;
    }
};


struct LinearGradientImpl :  LinearGradient
{
    Fill::Impl impl;
    Point p1{}, p2{};

    LinearGradientImpl()
    {
        Fill::pImpl = &impl;
    }

    Fill* duplicate() const
    {
        auto ret = LinearGradient::gen();
        LINEAR(ret)->impl.copy(this->impl);
        LINEAR(ret)->p1 = p1;
        LINEAR(ret)->p2 = p2;

        return ret;
    }

    Result linear(float x1, float y1, float x2, float y2) noexcept
    {
        p1 = {x1, y1};
        p2 = {x2, y2};

        return Result::Success;
    }

    Result linear(float* x1, float* y1, float* x2, float* y2) const noexcept
    {
        if (x1) *x1 = p1.x;
        if (x2) *x2 = p2.x;
        if (y1) *y1 = p1.y;
        if (y2) *y2 = p2.y;

        return Result::Success;
    }
};


#endif  //_TVG_FILL_H_
