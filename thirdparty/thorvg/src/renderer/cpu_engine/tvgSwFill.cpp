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

#include "tvgSwCommon.h"
#include "tvgFill.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

#define FIXPT_BITS 8
#define FIXPT_SIZE (1<<FIXPT_BITS)
#define MARGIN_MAX 40 // The maximum number of color table entries for the anti-aliasing (AA) margin at both ends.

static void _adjustAAMargin(uint32_t& iMargin, uint32_t count)
{
    constexpr float threshold = 0.1f;

    auto iThreshold = static_cast<uint32_t>(count * threshold);
    if (iMargin > iThreshold) iMargin = iThreshold;
    if (iMargin > MARGIN_MAX) iMargin = MARGIN_MAX;
}

static void _applyRepeatAA(SwFill* fill, const Fill::ColorStop* colors, uint32_t cnt)
{
    if (cnt == 1) return;

    // Calculate count that satisfies (i + 0.5) / SW_COLOR_TABLE < offset
    auto countBegin = static_cast<uint32_t>(max(0.0f, floor((colors + 1)->offset * SW_COLOR_TABLE - 0.5f) + 1.0f));
    auto countEnd = static_cast<uint32_t>(max(0.0f, floor((1.0f - (colors + cnt - 2)->offset) * SW_COLOR_TABLE - 0.5f) + 1.0f));

    // estimate AA margine
    constexpr float marginScaleFactor = 800.0f;
    auto begin = tvg::zero(fill->extent) ? 0 : static_cast<uint32_t>(marginScaleFactor / fill->extent);
    _adjustAAMargin(begin, countBegin);
    auto end = begin;
    _adjustAAMargin(end, countEnd);

    // apply AA
    if (begin == 0 || end == 0) return;

    auto unblend = [](uint32_t c) {
        auto a = (c >> 24);
        if (a == 255 || a == 0) return c;
        auto invA = 255.0f / static_cast<float>(a);
        auto c0 = static_cast<uint8_t>(static_cast<float>((c >> 16) & 0xFF) * invA);
        auto c1 = static_cast<uint8_t>(static_cast<float>((c >> 8) & 0xFF) * invA);
        auto c2 = static_cast<uint8_t>(static_cast<float>(c & 0xFF) * invA);

        return (a << 24) | (c0 << 16) | (c1 << 8) | c2;
    };

    auto i = SW_COLOR_TABLE - end;
    auto rgbaEnd = unblend(fill->ctable[i]);
    auto rgbaBegin = unblend(fill->ctable[begin]);

    auto dt = 1.0f / (begin + end + 1.0f);
    float t = dt;
    while (i != begin) {
        auto dist = 255 - static_cast<int32_t>(255 * t);
        auto color = INTERPOLATE(rgbaEnd, rgbaBegin, dist);
        fill->ctable[i++] = ALPHA_BLEND((color | 0xff000000), (color >> 24));

        if (i == SW_COLOR_TABLE) i = 0;
        t += dt;
    }
}

// Generate the ctable entries over the [begin, end) range
static void _genColorTable(SwFill* fill, const Fill::ColorStop* colors, uint32_t cnt, const SwSurface* surface, uint8_t opacity, uint32_t begin, uint32_t end)
{
    auto pColors = colors;

    auto a = MULTIPLY(pColors->a, opacity);
    if (a < 255) fill->translucent = true;

    auto r = pColors->r;
    auto g = pColors->g;
    auto b = pColors->b;
    auto rgba = surface->join(r, g, b, a);
    auto inc = 1.0f / static_cast<float>(SW_COLOR_TABLE);
    auto pos = (begin + 0.5f) * inc;
    auto i = begin;

    auto padColor = ALPHA_BLEND(rgba | 0xff000000, a);

    while (i < end && (i == 0 || pos <= pColors->offset)) {
        fill->ctable[i++] = padColor;
        pos += inc;
    }

    for (uint32_t j = 0; j < cnt - 1; ++j) {
        auto curr = colors + j;
        auto next = curr + 1;
        auto div = next->offset - curr->offset;
        auto delta = div != 0.0f ? (1.0f / div) : 0.0f;
        auto a2 = MULTIPLY(next->a, opacity);

        if (!fill->translucent && a2 < 255) fill->translucent = true;

        auto rgba2 = surface->join(next->r, next->g, next->b, a2);

        while (i < end && pos < next->offset) {
            auto t = (pos - curr->offset) * delta;
            auto dist = static_cast<int32_t>(255 * t);
            auto dist2 = 255 - dist;
            auto color = INTERPOLATE(rgba, rgba2, dist2);
            fill->ctable[i] = ALPHA_BLEND((color | 0xff000000), (color >> 24));
            ++i;
            pos += inc;
        }
        rgba = rgba2;
        a = a2;
    }
    rgba = ALPHA_BLEND((rgba | 0xff000000), a);

    for (; i < end; ++i) {
        fill->ctable[i] = rgba;
    }

    if (end == SW_COLOR_TABLE) fill->ctable[SW_COLOR_TABLE - 1] = rgba;
}

static bool _updateColorTable(SwFill* fill, const Fill* fdata, const SwSurface* surface, uint8_t opacity)
{
    if (fill->solid) return true;

    const Fill::ColorStop* colors;
    auto cnt = fdata->colorStops(&colors);
    if (cnt == 0 || !colors) return false;

    _genColorTable(fill, colors, cnt, surface, opacity, 0, SW_COLOR_TABLE);
    if (fill->spread == FillSpread::Repeat && fdata->type() != Type::ConicGradient) _applyRepeatAA(fill, colors, cnt);

    return true;
}

// Update only the ctable corresponding to the margin based on the updated extent
static bool _updateAAMargin(SwFill* fill, const Fill* fdata, const SwSurface* surface, uint8_t opacity)
{
    const Fill::ColorStop* colors;
    auto cnt = fdata->colorStops(&colors);
    if (cnt == 0) return false;

    _genColorTable(fill, colors, cnt, surface, opacity, 0, MARGIN_MAX);
    _genColorTable(fill, colors, cnt, surface, opacity, SW_COLOR_TABLE - MARGIN_MAX, SW_COLOR_TABLE);
    _applyRepeatAA(fill, colors, cnt);

    return true;
}

static inline uint32_t _clamp(const SwFill* fill, int32_t pos)
{
    switch (fill->spread) {
        case FillSpread::Pad: {
            if (pos >= SW_COLOR_TABLE) pos = SW_COLOR_TABLE - 1;
            else if (pos < 0) pos = 0;
            break;
        }
        case FillSpread::Repeat: {
            pos = pos % SW_COLOR_TABLE;
            if (pos < 0) pos = SW_COLOR_TABLE + pos;
            break;
        }
        case FillSpread::Reflect: {
            auto limit = SW_COLOR_TABLE * 2;
            pos = pos % limit;
            if (pos < 0) pos = limit + pos;
            if (pos >= SW_COLOR_TABLE) pos = (limit - pos - 1);
            break;
        }
    }
    return pos;
}

static inline uint32_t _pixel(const SwFill* fill, float pos)
{
    auto i = static_cast<int32_t>(pos * (SW_COLOR_TABLE - 1) + 0.5f);
    return fill->ctable[_clamp(fill, i)];
}

/************************************************************************/
/* Radial Implementation                                                */
/************************************************************************/

#define RADIAL_A_THRESHOLD 0.0005f

/*
 * quadratic equation with the following coefficients (rx and ry defined in the _calculateCoefficients()):
 * A = a  // fill->radial.a
 * B = 2 * (dr * fr + rx * dx + ry * dy)
 * C = fr^2 - rx^2 - ry^2
 * Derivatives are computed with respect to dx.
 * This procedure aims to optimize and eliminate the need to calculate all values from the beginning
 * for consecutive x values with a constant y. The Taylor series expansions are computed as long as
 * its terms are non-zero.
 */
static void _calculateCoefficients(const SwFill* fill, uint32_t x, uint32_t y, float& b, float& deltaB, float& det, float& deltaDet, float& deltaDeltaDet)
{
    auto radial = &fill->radial;

    auto rx = (x + 0.5f) * radial->a11 + (y + 0.5f) * radial->a12 + radial->a13 - radial->fx;
    auto ry = (x + 0.5f) * radial->a21 + (y + 0.5f) * radial->a22 + radial->a23 - radial->fy;

    b = (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy) * radial->invA;
    deltaB = (radial->a11 * radial->dx + radial->a21 * radial->dy) * radial->invA;

    auto rr = rx * rx + ry * ry;
    auto deltaRr = 2.0f * (rx * radial->a11 + ry * radial->a21) * radial->invA;
    auto deltaDeltaRr = 2.0f * (radial->a11 * radial->a11 + radial->a21 * radial->a21) * radial->invA;

    det = b * b + (rr - radial->fr * radial->fr) * radial->invA;
    deltaDet = 2.0f * b * deltaB + deltaB * deltaB + deltaRr + deltaDeltaRr * 0.5f;
    deltaDeltaDet = 2.0f * deltaB * deltaB + deltaDeltaRr;
}

bool _prepareRadial(SwFill* fill, const RadialGradient* radial, const Matrix& pTransform, bool& extentChanged)
{
    float cx, cy, r, fx, fy, fr;
    radial->radial(&cx, &cy, &r, &fx, &fy, &fr);
    if ((fill->solid = !CONST_RADIAL(radial)->correct(fx, fy, fr))) return true;

    fill->radial.dr = r - fr;
    fill->radial.dx = cx - fx;
    fill->radial.dy = cy - fy;
    fill->radial.fr = fr;
    fill->radial.fx = fx;
    fill->radial.fy = fy;
    fill->radial.a = fill->radial.dr * fill->radial.dr - fill->radial.dx * fill->radial.dx - fill->radial.dy * fill->radial.dy;

    // relative epsilon
    auto precision = std::max(1e-5f, std::max(fill->radial.dr * fill->radial.dr, 1.0f) * 1e-4f);

    if (fill->radial.a < precision) fill->radial.a = precision;
    fill->radial.invA = 1.0f / fill->radial.a;

    const auto& transform = pTransform * radial->transform();

    Matrix itransform;
    if (!inverse(&transform, &itransform)) return false;

    fill->radial.a11 = itransform.e11;
    fill->radial.a12 = itransform.e12;
    fill->radial.a13 = itransform.e13;
    fill->radial.a21 = itransform.e21;
    fill->radial.a22 = itransform.e22;
    fill->radial.a23 = itransform.e23;

    // The transform maps the gradient circle to an ellipse. Take its semi-minor axis
    // (the smaller singular value), where the table packs tightest and needs most margin.
    auto sx = transform.e11 * transform.e11 + transform.e21 * transform.e21;
    auto sy = transform.e12 * transform.e12 + transform.e22 * transform.e22;
    auto sxy = transform.e11 * transform.e12 + transform.e21 * transform.e22;
    auto d = sqrtf((sx - sy) * (sx - sy) + 4.0f * sxy * sxy);
    auto ext = r * sqrtf(0.5f * std::max(0.0f, sx + sy - d));
    if (!tvg::equal(ext, fill->extent)) {
        extentChanged = true;
        fill->extent = ext;
    }

    return true;
}

static void _fillRadial(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwAlpha alpha, uint8_t csize, uint8_t opacity)
{
    //edge case
    if (fill->radial.a < RADIAL_A_THRESHOLD) {
        auto radial = &fill->radial;
        auto rx = (x + 0.5f) * radial->a11 + (y + 0.5f) * radial->a12 + radial->a13 - radial->fx;
        auto ry = (x + 0.5f) * radial->a21 + (y + 0.5f) * radial->a22 + radial->a23 - radial->fy;

        if (opacity == 255) {
            for (uint32_t i = 0 ; i < len ; ++i, ++dst, cmp += csize) {
                auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
                *dst = opBlendNormal(_pixel(fill, x0), *dst, alpha(cmp));
                rx += radial->a11;
                ry += radial->a21;
            }
        } else {
            for (uint32_t i = 0 ; i < len ; ++i, ++dst, cmp += csize) {
                auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
                *dst = opBlendNormal(_pixel(fill, x0), *dst, MULTIPLY(opacity, alpha(cmp)));
                rx += radial->a11;
                ry += radial->a21;
            }
        }
    } else {
        float b, deltaB, det, deltaDet, deltaDeltaDet;
        _calculateCoefficients(fill, x, y, b, deltaB, det, deltaDet, deltaDeltaDet);

        if (opacity == 255) {
            for (uint32_t i = 0 ; i < len ; ++i, ++dst, cmp += csize) {
                *dst = opBlendNormal(_pixel(fill, sqrtf(det) - b), *dst, alpha(cmp));
                det += deltaDet;
                deltaDet += deltaDeltaDet;
                b += deltaB;
            }
        } else {
            for (uint32_t i = 0 ; i < len ; ++i, ++dst, cmp += csize) {
                *dst = opBlendNormal(_pixel(fill, sqrtf(det) - b), *dst, MULTIPLY(opacity, alpha(cmp)));
                det += deltaDet;
                deltaDet += deltaDeltaDet;
                b += deltaB;
            }
        }
    }
}

static void _fillRadial(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, uint8_t a)
{
    if (fill->radial.a < RADIAL_A_THRESHOLD) {
        auto radial = &fill->radial;
        auto rx = (x + 0.5f) * radial->a11 + (y + 0.5f) * radial->a12 + radial->a13 - radial->fx;
        auto ry = (x + 0.5f) * radial->a21 + (y + 0.5f) * radial->a22 + radial->a23 - radial->fy;
        for (uint32_t i = 0; i < len; ++i, ++dst) {
            auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
            *dst = op(_pixel(fill, x0), *dst, a);
            rx += radial->a11;
            ry += radial->a21;
        }
    } else {
        float b, deltaB, det, deltaDet, deltaDeltaDet;
        _calculateCoefficients(fill, x, y, b, deltaB, det, deltaDet, deltaDeltaDet);

        for (uint32_t i = 0; i < len; ++i, ++dst) {
            *dst = op(_pixel(fill, sqrtf(det) - b), *dst, a);
            det += deltaDet;
            deltaDet += deltaDeltaDet;
            b += deltaB;
        }
    }
}

static void _fillRadial(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, SwMask maskOp, uint8_t a)
{
    if (fill->radial.a < RADIAL_A_THRESHOLD) {
        auto radial = &fill->radial;
        auto rx = (x + 0.5f) * radial->a11 + (y + 0.5f) * radial->a12 + radial->a13 - radial->fx;
        auto ry = (x + 0.5f) * radial->a21 + (y + 0.5f) * radial->a22 + radial->a23 - radial->fy;
        for (uint32_t i = 0 ; i < len ; ++i, ++dst) {
            auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
            auto src = MULTIPLY(a, A(_pixel(fill, x0)));
            *dst = maskOp(src, *dst, ~src);
            rx += radial->a11;
            ry += radial->a21;
        }
    } else {
        float b, deltaB, det, deltaDet, deltaDeltaDet;
        _calculateCoefficients(fill, x, y, b, deltaB, det, deltaDet, deltaDeltaDet);

        for (uint32_t i = 0 ; i < len ; ++i, ++dst) {
            auto src = MULTIPLY(a, A(_pixel(fill, sqrtf(det) - b)));
            *dst = maskOp(src, *dst, ~src);
            det += deltaDet;
            deltaDet += deltaDeltaDet;
            b += deltaB;
        }
    }
}

static void _fillRadial(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwMask maskOp, uint8_t a)
{
    if (fill->radial.a < RADIAL_A_THRESHOLD) {
        auto radial = &fill->radial;
        auto rx = (x + 0.5f) * radial->a11 + (y + 0.5f) * radial->a12 + radial->a13 - radial->fx;
        auto ry = (x + 0.5f) * radial->a21 + (y + 0.5f) * radial->a22 + radial->a23 - radial->fy;
        for (uint32_t i = 0 ; i < len ; ++i, ++dst, ++cmp) {
            auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
            auto src = MULTIPLY(A(A(_pixel(fill, x0))), a);
            auto tmp = maskOp(src, *cmp, 0);
            *dst = tmp + MULTIPLY(*dst, ~tmp);
            rx += radial->a11;
            ry += radial->a21;
        }
    } else {
        float b, deltaB, det, deltaDet, deltaDeltaDet;
        _calculateCoefficients(fill, x, y, b, deltaB, det, deltaDet, deltaDeltaDet);

        for (uint32_t i = 0 ; i < len ; ++i, ++dst, ++cmp) {
            auto src = MULTIPLY(A(_pixel(fill, sqrtf(det))), a);
            auto tmp = maskOp(src, *cmp, 0);
            *dst = tmp + MULTIPLY(*dst, ~tmp);
            det += deltaDet;
            deltaDet += deltaDeltaDet;
            b += deltaB;
        }
    }
}

static void _fillRadial(const SwSurface* surface, const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, SwBlender op2, uint8_t a)
{
    if (fill->radial.a < RADIAL_A_THRESHOLD) {
        auto radial = &fill->radial;
        auto rx = (x + 0.5f) * radial->a11 + (y + 0.5f) * radial->a12 + radial->a13 - radial->fx;
        auto ry = (x + 0.5f) * radial->a21 + (y + 0.5f) * radial->a22 + radial->a23 - radial->fy;

        if (a == 255) {
            for (uint32_t i = 0; i < len; ++i, ++dst) {
                auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
                auto tmp = op(_pixel(fill, x0), *dst, 255);
                *dst = op2(surface, tmp, *dst);
                rx += radial->a11;
                ry += radial->a21;
            }
        } else {
            for (uint32_t i = 0; i < len; ++i, ++dst) {
                auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
                auto tmp = op(_pixel(fill, x0), *dst, 255);
                auto tmp2 = op2(surface, tmp, *dst);
                *dst = INTERPOLATE(tmp2, *dst, a);
                rx += radial->a11;
                ry += radial->a21;
            }
        }
    } else {
        float b, deltaB, det, deltaDet, deltaDeltaDet;
        _calculateCoefficients(fill, x, y, b, deltaB, det, deltaDet, deltaDeltaDet);
        if (a == 255) {
            for (uint32_t i = 0 ; i < len ; ++i, ++dst) {
                auto tmp = op(_pixel(fill, sqrtf(det) - b), *dst, 255);
                *dst = op2(surface, tmp, *dst);
                det += deltaDet;
                deltaDet += deltaDeltaDet;
                b += deltaB;
            }
        } else {
            for (uint32_t i = 0 ; i < len ; ++i, ++dst) {
                auto tmp = op(_pixel(fill, sqrtf(det) - b), *dst, 255);
                auto tmp2 = op2(surface, tmp, *dst);
                *dst = INTERPOLATE(tmp2, *dst, a);
                det += deltaDet;
                deltaDet += deltaDeltaDet;
                b += deltaB;
            }
        }
    }
}

/************************************************************************/
/* Conic Implementation                                                 */
/************************************************************************/

struct ConicAARange
{
    float begin = 0.0f;
    float end = 0.0f;
    float distance = 0.0f;
};

static ConicAARange _conicAARange(const SwFill* fill, float rx, float ry, uint32_t len)
{
    auto conic = &fill->conic;
    if (conic->invFwidth <= 0.0f || len == 0) return {};

    auto offset = Point{rx, ry};

    // normal distance limits AA to the band around the infinite boundary line
    auto distance = dot(conic->normal, offset) * conic->invFwidth;
    auto distanceDx = conic->distanceDx;
    auto range = ConicAARange{0.0f, static_cast<float>(len)};
    if (tvg::zero(distanceDx)) {
        if (distance <= -0.5f || distance >= 0.5f) return {};
    } else {
        auto first = (-0.5f - distance) / distanceDx;  // distance + first * distanceDx = -0.5
        auto last = (0.5f - distance) / distanceDx;    // distance + last * distanceDx = 0.5
        if (first > last) std::swap(first, last);
        range.begin = std::max(range.begin, floorf(first) + 1.0f);
        range.end = std::min(range.end, ceilf(last));
    }

    // exclude the half of the AA band opposite the conic seam
    auto seam = Point{conic->normal.y, -conic->normal.x};
    auto seamProjection = dot(offset, seam);
    auto seamProjectionDx = conic->seamProjectionDx;
    if (tvg::zero(seamProjectionDx)) {
        if (seamProjection < 0.0f) return {};
    } else if (seamProjectionDx > 0.0f) {
        range.begin = std::max(range.begin, ceilf(-seamProjection / seamProjectionDx));
    } else {
        range.end = std::min(range.end, floorf(-seamProjection / seamProjectionDx) + 1.0f);
    }

    if (range.begin >= range.end) return {};

    range.distance = distance + range.begin * distanceDx;
    return range;
}

static uint32_t _conicPixel(const SwFill* fill, float rx, float ry, uint32_t i, const ConicAARange& range)
{
    auto t = [](const SwFill* fill, float rx, float ry) {
        // atan2 starts at 3 o'clock; offset rotates to the configured start angle.
        auto t = atan2f(ry, rx) * (0.5f / MATH_PI) + fill->conic.offset;
        return t - floorf(t);  // wrap to [0..1)
    };

    if (i < range.begin || i >= range.end) return _pixel(fill, t(fill, rx, ry));
    auto distance = range.distance + (i - range.begin) * fill->conic.distanceDx;
    return INTERPOLATE(fill->ctable[0], fill->ctable[SW_COLOR_TABLE - 1], static_cast<uint8_t>(255.0f * (distance + 0.5f)));  // distance is kept within (-0.5f, 0.5f) by _conicAARange()
}

bool _prepareConic(SwFill* fill, const ConicGradient* conic, const Matrix& pTransform)
{
    float cx, cy, angle;
    conic->conic(&cx, &cy, &angle);

    fill->conic.cx = cx;
    fill->conic.cy = cy;
    fill->conic.offset = -angle / 360.0f;

    const auto& transform = pTransform * conic->transform();

    Matrix itransform;
    if (!inverse(&transform, &itransform)) return false;

    fill->conic.a11 = itransform.e11;
    fill->conic.a12 = itransform.e12;
    fill->conic.a13 = itransform.e13;
    fill->conic.a21 = itransform.e21;
    fill->conic.a22 = itransform.e22;
    fill->conic.a23 = itransform.e23;

    // prepare conic anti-aliasing variables
    auto radian = angle * (MATH_PI / 180.0f);
    auto seam = Point{cosf(radian), sinf(radian)};
    fill->conic.normal = {-seam.y, seam.x};
    auto xStep = Point{fill->conic.a11, fill->conic.a21};
    auto yStep = Point{fill->conic.a12, fill->conic.a22};
    auto dFdx = dot(fill->conic.normal, xStep);
    auto dFdy = dot(fill->conic.normal, yStep);
    auto fwidth = fabsf(dFdx) + fabsf(dFdy);

    fill->conic.invFwidth = (tvg::zero(dFdx) || tvg::zero(dFdy)) ? 0.0f : 1.0f / fwidth;
    fill->conic.distanceDx = dFdx * fill->conic.invFwidth;
    fill->conic.seamProjectionDx = dot(seam, xStep);

    return true;
}

static void _fillConic(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwAlpha alpha, uint8_t csize, uint8_t opacity)
{
    auto conic = &fill->conic;
    auto rx = (x + 0.5f) * conic->a11 + (y + 0.5f) * conic->a12 + conic->a13 - conic->cx;
    auto ry = (x + 0.5f) * conic->a21 + (y + 0.5f) * conic->a22 + conic->a23 - conic->cy;
    auto range = _conicAARange(fill, rx, ry, len);

    if (opacity == 255) {
        for (uint32_t i = 0; i < len; ++i, ++dst, cmp += csize) {
            *dst = opBlendNormal(_conicPixel(fill, rx, ry, i, range), *dst, alpha(cmp));
            rx += conic->a11;
            ry += conic->a21;
        }
    } else {
        for (uint32_t i = 0; i < len; ++i, ++dst, cmp += csize) {
            *dst = opBlendNormal(_conicPixel(fill, rx, ry, i, range), *dst, MULTIPLY(opacity, alpha(cmp)));
            rx += conic->a11;
            ry += conic->a21;
        }
    }
}

static void _fillConic(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, uint8_t a)
{
    auto conic = &fill->conic;
    auto rx = (x + 0.5f) * conic->a11 + (y + 0.5f) * conic->a12 + conic->a13 - conic->cx;
    auto ry = (x + 0.5f) * conic->a21 + (y + 0.5f) * conic->a22 + conic->a23 - conic->cy;
    auto range = _conicAARange(fill, rx, ry, len);

    for (uint32_t i = 0; i < len; ++i, ++dst) {
        *dst = op(_conicPixel(fill, rx, ry, i, range), *dst, a);
        rx += conic->a11;
        ry += conic->a21;
    }
}

static void _fillConic(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, SwMask maskOp, uint8_t a)
{
    auto conic = &fill->conic;
    auto rx = (x + 0.5f) * conic->a11 + (y + 0.5f) * conic->a12 + conic->a13 - conic->cx;
    auto ry = (x + 0.5f) * conic->a21 + (y + 0.5f) * conic->a22 + conic->a23 - conic->cy;
    auto range = _conicAARange(fill, rx, ry, len);

    for (uint32_t i = 0; i < len; ++i, ++dst) {
        auto src = MULTIPLY(a, A(_conicPixel(fill, rx, ry, i, range)));
        *dst = maskOp(src, *dst, ~src);
        rx += conic->a11;
        ry += conic->a21;
    }
}

static void _fillConic(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwMask maskOp, uint8_t a)
{
    auto conic = &fill->conic;
    auto rx = (x + 0.5f) * conic->a11 + (y + 0.5f) * conic->a12 + conic->a13 - conic->cx;
    auto ry = (x + 0.5f) * conic->a21 + (y + 0.5f) * conic->a22 + conic->a23 - conic->cy;
    auto range = _conicAARange(fill, rx, ry, len);

    for (uint32_t i = 0; i < len; ++i, ++dst, ++cmp) {
        auto src = MULTIPLY(A(_conicPixel(fill, rx, ry, i, range)), a);
        auto tmp = maskOp(src, *cmp, 0);
        *dst = tmp + MULTIPLY(*dst, ~tmp);
        rx += conic->a11;
        ry += conic->a21;
    }
}

static void _fillConic(const SwSurface* surface, const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, SwBlender op2, uint8_t a)
{
    auto conic = &fill->conic;
    auto rx = (x + 0.5f) * conic->a11 + (y + 0.5f) * conic->a12 + conic->a13 - conic->cx;
    auto ry = (x + 0.5f) * conic->a21 + (y + 0.5f) * conic->a22 + conic->a23 - conic->cy;
    auto range = _conicAARange(fill, rx, ry, len);

    if (a == 255) {
        for (uint32_t i = 0; i < len; ++i, ++dst) {
            auto tmp = op(_conicPixel(fill, rx, ry, i, range), *dst, 255);
            *dst = op2(surface, tmp, *dst);
            rx += conic->a11;
            ry += conic->a21;
        }
    } else {
        for (uint32_t i = 0; i < len; ++i, ++dst) {
            auto tmp = op(_conicPixel(fill, rx, ry, i, range), *dst, 255);
            auto tmp2 = op2(surface, tmp, *dst);
            *dst = INTERPOLATE(tmp2, *dst, a);
            rx += conic->a11;
            ry += conic->a21;
        }
    }
}

/************************************************************************/
/* Linear Implementation                                                */
/************************************************************************/

static inline uint32_t _fixedPixel(const SwFill* fill, int32_t pos)
{
    int32_t i = (pos + (FIXPT_SIZE / 2)) >> FIXPT_BITS;
    return fill->ctable[_clamp(fill, i)];
}

bool _prepareLinear(SwFill* fill, const LinearGradient* linear, const Matrix& pTransform, bool& extentChanged)
{
    float x1, x2, y1, y2;
    linear->linear(&x1, &y1, &x2, &y2);

    fill->linear.dx = x2 - x1;
    fill->linear.dy = y2 - y1;
    auto len = fill->linear.dx * fill->linear.dx + fill->linear.dy * fill->linear.dy;

    if (len < FLOAT_EPSILON) {
        if (tvg::zero(fill->linear.dx) && tvg::zero(fill->linear.dy)) {
            fill->solid = true;
        }
        return true;
    }

    fill->linear.dx /= len;
    fill->linear.dy /= len;
    fill->linear.offset = -fill->linear.dx * x1 - fill->linear.dy * y1;

    const auto& transform = pTransform * linear->transform();

    Matrix itransform;
    if (!inverse(&transform, &itransform)) return false;

    fill->linear.offset += fill->linear.dx * itransform.e13 + fill->linear.dy * itransform.e23;

    auto dx = fill->linear.dx;
    fill->linear.dx = dx * itransform.e11 + fill->linear.dy * itransform.e21;
    fill->linear.dy = dx * itransform.e12 + fill->linear.dy * itransform.e22;

    // dx/dy is amount of change in screen space. The reciprocal of its magnitude is
    // the pixel span of one full color table cycle.
    auto ext = 1.0f / sqrtf(fill->linear.dx * fill->linear.dx + fill->linear.dy * fill->linear.dy);
    if (!tvg::equal(ext, fill->extent)) {
        extentChanged = true;
        fill->extent = ext;
    }

    return true;
}

static void _fillLinear(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwAlpha alpha, uint8_t csize, uint8_t opacity)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (SW_COLOR_TABLE - 1);
    float inc = (fill->linear.dx) * (SW_COLOR_TABLE - 1);

    if (opacity == 255) {
        if (tvg::zero(inc)) {
            auto color = _fixedPixel(fill, static_cast<int32_t>(t * FIXPT_SIZE));
            for (uint32_t i = 0; i < len; ++i, ++dst, cmp += csize) {
                *dst = opBlendNormal(color, *dst, alpha(cmp));
            }
            return;
        }

        auto vMax = static_cast<float>(INT32_MAX >> (FIXPT_BITS + 1));
        auto vMin = -vMax;
        auto v = t + (inc * len);

        //we can use fixed point math
        if (v < vMax && v > vMin) {
            auto t2 = static_cast<int32_t>(t * FIXPT_SIZE);
            auto inc2 = static_cast<int32_t>(inc * FIXPT_SIZE);
            for (uint32_t j = 0; j < len; ++j, ++dst, cmp += csize) {
                *dst = opBlendNormal(_fixedPixel(fill, t2), *dst, alpha(cmp));
                t2 += inc2;
            }
        //we have to fallback to float math
        } else {
            uint32_t counter = 0;
            while (counter++ < len) {
                *dst = opBlendNormal(_pixel(fill, t / SW_COLOR_TABLE), *dst, alpha(cmp));
                ++dst;
                t += inc;
                cmp += csize;
            }
        }
    } else {
        if (tvg::zero(inc)) {
            auto color = _fixedPixel(fill, static_cast<int32_t>(t * FIXPT_SIZE));
            for (uint32_t i = 0; i < len; ++i, ++dst, cmp += csize) {
                *dst = opBlendNormal(color, *dst, MULTIPLY(alpha(cmp), opacity));
            }
            return;
        }

        auto vMax = static_cast<float>(INT32_MAX >> (FIXPT_BITS + 1));
        auto vMin = -vMax;
        auto v = t + (inc * len);

        //we can use fixed point math
        if (v < vMax && v > vMin) {
            auto t2 = static_cast<int32_t>(t * FIXPT_SIZE);
            auto inc2 = static_cast<int32_t>(inc * FIXPT_SIZE);
            for (uint32_t j = 0; j < len; ++j, ++dst, cmp += csize) {
                *dst = opBlendNormal(_fixedPixel(fill, t2), *dst, MULTIPLY(alpha(cmp), opacity));
                t2 += inc2;
            }
        //we have to fallback to float math
        } else {
            uint32_t counter = 0;
            while (counter++ < len) {
                *dst = opBlendNormal(_pixel(fill, t / SW_COLOR_TABLE), *dst, MULTIPLY(opacity, alpha(cmp)));
                ++dst;
                t += inc;
                cmp += csize;
            }
        }
    }
}

static void _fillLinear(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, SwMask maskOp, uint8_t a)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (SW_COLOR_TABLE - 1);
    float inc = (fill->linear.dx) * (SW_COLOR_TABLE - 1);

    if (tvg::zero(inc)) {
        auto src = MULTIPLY(a, A(_fixedPixel(fill, static_cast<int32_t>(t * FIXPT_SIZE))));
        for (uint32_t i = 0; i < len; ++i, ++dst) {
            *dst = maskOp(src, *dst, ~src);
        }
        return;
    }

    auto vMax = static_cast<float>(INT32_MAX >> (FIXPT_BITS + 1));
    auto vMin = -vMax;
    auto v = t + (inc * len);

    //we can use fixed point math
    if (v < vMax && v > vMin) {
        auto t2 = static_cast<int32_t>(t * FIXPT_SIZE);
        auto inc2 = static_cast<int32_t>(inc * FIXPT_SIZE);
        for (uint32_t j = 0; j < len; ++j, ++dst) {
            auto src = MULTIPLY(A(_fixedPixel(fill, t2)), a);
            *dst = maskOp(src, *dst, ~src);
            t2 += inc2;
        }
    //we have to fallback to float math
    } else {
        uint32_t counter = 0;
        while (counter++ < len) {
            auto src = MULTIPLY(A(_pixel(fill, t / SW_COLOR_TABLE)), a);
            *dst = maskOp(src, *dst, ~src);
            ++dst;
            t += inc;
        }
    }
}

static void _fillLinear(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwMask maskOp, uint8_t a)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (SW_COLOR_TABLE - 1);
    float inc = (fill->linear.dx) * (SW_COLOR_TABLE - 1);

    if (tvg::zero(inc)) {
        auto src = A(_fixedPixel(fill, static_cast<int32_t>(t * FIXPT_SIZE)));
        src = MULTIPLY(src, a);
        for (uint32_t i = 0; i < len; ++i, ++dst, ++cmp) {
            auto tmp = maskOp(src, *cmp, 0);
            *dst = tmp + MULTIPLY(*dst, ~tmp);
        }
        return;
    }

    auto vMax = static_cast<float>(INT32_MAX >> (FIXPT_BITS + 1));
    auto vMin = -vMax;
    auto v = t + (inc * len);

    //we can use fixed point math
    if (v < vMax && v > vMin) {
        auto t2 = static_cast<int32_t>(t * FIXPT_SIZE);
        auto inc2 = static_cast<int32_t>(inc * FIXPT_SIZE);
        for (uint32_t j = 0; j < len; ++j, ++dst, ++cmp) {
            auto src = MULTIPLY(a, A(_fixedPixel(fill, t2)));
            auto tmp = maskOp(src, *cmp, 0);
            *dst = tmp + MULTIPLY(*dst, ~tmp);
            t2 += inc2;
        }
    //we have to fallback to float math
    } else {
        uint32_t counter = 0;
        while (counter++ < len) {
            auto src = MULTIPLY(A(_pixel(fill, t / SW_COLOR_TABLE)), a);
            auto tmp = maskOp(src, *cmp, 0);
            *dst = tmp + MULTIPLY(*dst, ~tmp);
            ++dst;
            ++cmp;
            t += inc;
        }
    }
}

static void _fillLinear(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, uint8_t a)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (SW_COLOR_TABLE - 1);
    float inc = (fill->linear.dx) * (SW_COLOR_TABLE - 1);

    if (tvg::zero(inc)) {
        auto color = _fixedPixel(fill, static_cast<int32_t>(t * FIXPT_SIZE));
        for (uint32_t i = 0; i < len; ++i, ++dst) {
            *dst = op(color, *dst, a);
        }
        return;
    }

    auto vMax = static_cast<float>(INT32_MAX >> (FIXPT_BITS + 1));
    auto vMin = -vMax;
    auto v = t + (inc * len);

    //we can use fixed point math
    if (v < vMax && v > vMin) {
        auto t2 = static_cast<int32_t>(t * FIXPT_SIZE);
        auto inc2 = static_cast<int32_t>(inc * FIXPT_SIZE);
        for (uint32_t j = 0; j < len; ++j, ++dst) {
            *dst = op(_fixedPixel(fill, t2), *dst, a);
            t2 += inc2;
        }
    //we have to fallback to float math
    } else {
        uint32_t counter = 0;
        while (counter++ < len) {
            *dst = op(_pixel(fill, t / SW_COLOR_TABLE), *dst, a);
            ++dst;
            t += inc;
        }
    }
}

static void _fillLinear(const SwSurface* surface, const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, SwBlender op2, uint8_t a)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (SW_COLOR_TABLE - 1);
    float inc = (fill->linear.dx) * (SW_COLOR_TABLE - 1);

    if (tvg::zero(inc)) {
        auto color = _fixedPixel(fill, static_cast<int32_t>(t * FIXPT_SIZE));
        if (a == 255) {
            for (uint32_t i = 0; i < len; ++i, ++dst) {
                auto tmp = op(color, *dst, a);
                *dst = op2(surface, tmp, *dst);
            }
        } else {
            for (uint32_t i = 0; i < len; ++i, ++dst) {
                auto tmp = op(color, *dst, a);
                auto tmp2 = op2(surface, tmp, *dst);
                *dst = INTERPOLATE(tmp2, *dst, a);
            }
        }
        return;
    }

    auto vMax = static_cast<float>(INT32_MAX >> (FIXPT_BITS + 1));
    auto vMin = -vMax;
    auto v = t + (inc * len);

    if (a == 255) {
        //we can use fixed point math
        if (v < vMax && v > vMin) {
            auto t2 = static_cast<int32_t>(t * FIXPT_SIZE);
            auto inc2 = static_cast<int32_t>(inc * FIXPT_SIZE);
            for (uint32_t j = 0; j < len; ++j, ++dst) {
                auto tmp = op(_fixedPixel(fill, t2), *dst, 255);
                *dst = op2(surface, tmp, *dst);
                t2 += inc2;
            }
        //we have to fallback to float math
        } else {
            uint32_t counter = 0;
            while (counter++ < len) {
                auto tmp = op(_pixel(fill, t / SW_COLOR_TABLE), *dst, 255);
                *dst = op2(surface, tmp, *dst);
                ++dst;
                t += inc;
            }
        }
    } else {
        //we can use fixed point math
        if (v < vMax && v > vMin) {
            auto t2 = static_cast<int32_t>(t * FIXPT_SIZE);
            auto inc2 = static_cast<int32_t>(inc * FIXPT_SIZE);
            for (uint32_t j = 0; j < len; ++j, ++dst) {
                auto tmp = op(_fixedPixel(fill, t2), *dst, 255);
                auto tmp2 = op2(surface, tmp, *dst);
                *dst = INTERPOLATE(tmp2, *dst, a);
                t2 += inc2;
            }
        //we have to fallback to float math
        } else {
            uint32_t counter = 0;
            while (counter++ < len) {
                auto tmp = op(_pixel(fill, t / SW_COLOR_TABLE), *dst, 255);
                auto tmp2 = op2(surface, tmp, *dst);
                *dst = INTERPOLATE(tmp2, *dst, a);
                ++dst;
                t += inc;
            }
        }
    }
}

/************************************************************************/
/* External Interace Implementation                                     */
/************************************************************************/

bool fillPrepare(SwFill*& fill, const Fill* fdata, const Matrix& transform, SwSurface* surface, uint8_t opacity, bool ctable)
{
    if (!fdata) return true;  // a normal case

    if (!fill) {
        fill = tvg::calloc<SwFill>(1, sizeof(SwFill));
        ctable = true;
    } else if (ctable) {
        fillReset(fill);
    }

    // single color stop is treated as solid
    if (fdata->colorStops(nullptr) == 1) {
        fill->solid = true;
        return true;
    }

    auto extentChanged = false;

    fill->spread = fdata->spread();
    fill->type = fdata->type();

    if (fill->type == Type::LinearGradient) {
        if (!_prepareLinear(fill, static_cast<const LinearGradient*>(fdata), transform, extentChanged)) return false;
    } else if (fill->type == Type::RadialGradient) {
        if (!_prepareRadial(fill, static_cast<const RadialGradient*>(fdata), transform, extentChanged)) return false;
    } else if (fill->type == Type::ConicGradient) {
        if (!_prepareConic(fill, static_cast<const ConicGradient*>(fdata), transform)) return false;
    }

    if (ctable) return _updateColorTable(fill, fdata, surface, opacity);
    else if (extentChanged && fill->spread == FillSpread::Repeat) return _updateAAMargin(fill, fdata, surface, opacity);
    return true;
}

const Fill::ColorStop* fillFetchSolid(const SwFill* fill, const Fill* fdata)
{
    if (!fill->solid) return nullptr;

    const Fill::ColorStop* colors;
    auto cnt = fdata->colorStops(&colors);
    if (cnt == 0 || !colors) return nullptr;

    return colors + cnt - 1;
}

void fillReset(SwFill* fill)
{
    fill->translucent = false;
    fill->solid = false;
}

// composite masking ver.
void fillRaster(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, SwMask op, uint8_t a)
{
    if (fill->type == Type::LinearGradient) _fillLinear(fill, dst, y, x, len, op, a);
    if (fill->type == Type::RadialGradient) _fillRadial(fill, dst, y, x, len, op, a);
    if (fill->type == Type::ConicGradient) _fillConic(fill, dst, y, x, len, op, a);
}

// direct masking ver.
void fillRaster(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwMask op, uint8_t a)
{
    if (fill->type == Type::LinearGradient) _fillLinear(fill, dst, y, x, len, cmp, op, a);
    if (fill->type == Type::RadialGradient) _fillRadial(fill, dst, y, x, len, cmp, op, a);
    if (fill->type == Type::ConicGradient) _fillConic(fill, dst, y, x, len, cmp, op, a);
}

// blending ver.
void fillRaster(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, uint8_t a)
{
    if (fill->type == Type::LinearGradient) _fillLinear(fill, dst, y, x, len, op, a);
    if (fill->type == Type::RadialGradient) _fillRadial(fill, dst, y, x, len, op, a);
    if (fill->type == Type::ConicGradient) _fillConic(fill, dst, y, x, len, op, a);
}

// matting ver.
void fillRaster(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwAlpha alpha, uint8_t csize, uint8_t opacity)
{
    if (fill->type == Type::LinearGradient) _fillLinear(fill, dst, y, x, len, cmp, alpha, csize, opacity);
    if (fill->type == Type::RadialGradient) _fillRadial(fill, dst, y, x, len, cmp, alpha, csize, opacity);
    if (fill->type == Type::ConicGradient) _fillConic(fill, dst, y, x, len, cmp, alpha, csize, opacity);
}

// blending + BlendingMethod(op2) ver.
void fillRaster(const SwSurface* surface, const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, SwBlender op2, uint8_t a)
{
    if (fill->type == Type::LinearGradient) _fillLinear(surface, fill, dst, y, x, len, op, op2, a);
    if (fill->type == Type::RadialGradient) _fillRadial(surface, fill, dst, y, x, len, op, op2, a);
    if (fill->type == Type::ConicGradient) _fillConic(surface, fill, dst, y, x, len, op, op2, a);
}
