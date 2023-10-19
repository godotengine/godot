/*
 * Copyright (c) 2020 - 2023 the ThorVG project. All rights reserved.

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
#include "tvgSwCommon.h"
#include "tvgFill.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

#define RADIAL_A_THRESHOLD 0.0005f
#define GRADIENT_STOP_SIZE 1024
#define FIXPT_BITS 8
#define FIXPT_SIZE (1<<FIXPT_BITS)

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
    deltaDet = 2.0f * b * deltaB + deltaB * deltaB + deltaRr + deltaDeltaRr;
    deltaDeltaDet = 2.0f * deltaB * deltaB + deltaDeltaRr;
}


static bool _updateColorTable(SwFill* fill, const Fill* fdata, const SwSurface* surface, uint8_t opacity)
{
    if (!fill->ctable) {
        fill->ctable = static_cast<uint32_t*>(malloc(GRADIENT_STOP_SIZE * sizeof(uint32_t)));
        if (!fill->ctable) return false;
    }

    const Fill::ColorStop* colors;
    auto cnt = fdata->colorStops(&colors);
    if (cnt == 0 || !colors) return false;

    auto pColors = colors;

    auto a = MULTIPLY(pColors->a, opacity);
    if (a < 255) fill->translucent = true;

    auto r = pColors->r;
    auto g = pColors->g;
    auto b = pColors->b;
    auto rgba = surface->join(r, g, b, a);

    auto inc = 1.0f / static_cast<float>(GRADIENT_STOP_SIZE);
    auto pos = 1.5f * inc;
    uint32_t i = 0;

    fill->ctable[i++] = ALPHA_BLEND(rgba | 0xff000000, a);

    while (pos <= pColors->offset) {
        fill->ctable[i] = fill->ctable[i - 1];
        ++i;
        pos += inc;
    }

    for (uint32_t j = 0; j < cnt - 1; ++j) {
        auto curr = colors + j;
        auto next = curr + 1;
        auto delta = 1.0f / (next->offset - curr->offset);
        auto a2 = MULTIPLY(next->a, opacity);
        if (!fill->translucent && a2 < 255) fill->translucent = true;

        auto rgba2 = surface->join(next->r, next->g, next->b, a2);

        while (pos < next->offset && i < GRADIENT_STOP_SIZE) {
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

    for (; i < GRADIENT_STOP_SIZE; ++i)
        fill->ctable[i] = rgba;

    //Make sure the last color stop is represented at the end of the table
    fill->ctable[GRADIENT_STOP_SIZE - 1] = rgba;

    return true;
}


bool _prepareLinear(SwFill* fill, const LinearGradient* linear, const Matrix* transform)
{
    float x1, x2, y1, y2;
    if (linear->linear(&x1, &y1, &x2, &y2) != Result::Success) return false;

    fill->linear.dx = x2 - x1;
    fill->linear.dy = y2 - y1;
    fill->linear.len = fill->linear.dx * fill->linear.dx + fill->linear.dy * fill->linear.dy;

    if (fill->linear.len < FLT_EPSILON) return true;

    fill->linear.dx /= fill->linear.len;
    fill->linear.dy /= fill->linear.len;
    fill->linear.offset = -fill->linear.dx * x1 - fill->linear.dy * y1;

    auto gradTransform = linear->transform();
    bool isTransformation = !mathIdentity((const Matrix*)(&gradTransform));

    if (isTransformation) {
        if (transform) gradTransform = mathMultiply(transform, &gradTransform);
    } else if (transform) {
        gradTransform = *transform;
        isTransformation = true;
    }

    if (isTransformation) {
        Matrix invTransform;
        if (!mathInverse(&gradTransform, &invTransform)) return false;

        fill->linear.offset += fill->linear.dx * invTransform.e13 + fill->linear.dy * invTransform.e23;

        auto dx = fill->linear.dx;
        fill->linear.dx = dx * invTransform.e11 + fill->linear.dy * invTransform.e21;
        fill->linear.dy = dx * invTransform.e12 + fill->linear.dy * invTransform.e22;

        fill->linear.len = fill->linear.dx * fill->linear.dx + fill->linear.dy * fill->linear.dy;
        if (fill->linear.len < FLT_EPSILON) return true;
    }

    return true;
}


bool _prepareRadial(SwFill* fill, const RadialGradient* radial, const Matrix* transform)
{
    auto cx = P(radial)->cx;
    auto cy = P(radial)->cy;
    auto r = P(radial)->r;
    auto fx = P(radial)->fx;
    auto fy = P(radial)->fy;
    auto fr = P(radial)->fr;

    if (r < FLT_EPSILON) return true;

    fill->radial.dr = r - fr;
    fill->radial.dx = cx - fx;
    fill->radial.dy = cy - fy;
    fill->radial.fr = fr;
    fill->radial.fx = fx;
    fill->radial.fy = fy;
    fill->radial.a = fill->radial.dr * fill->radial.dr - fill->radial.dx * fill->radial.dx - fill->radial.dy * fill->radial.dy;

    //This condition fulfills the SVG 1.1 std:
    //the focal point, if outside the end circle, is moved to be on the end circle
    //See: the SVG 2 std requirements: https://www.w3.org/TR/SVG2/pservers.html#RadialGradientNotes
    if (fill->radial.a < 0) {
        auto dist = sqrtf(fill->radial.dx * fill->radial.dx + fill->radial.dy * fill->radial.dy);
        fill->radial.fx = cx + r * (fx - cx) / dist;
        fill->radial.fy = cy + r * (fy - cy) / dist;
        fill->radial.dx = cx - fill->radial.fx;
        fill->radial.dy = cy - fill->radial.fy;
        fill->radial.a = fill->radial.dr * fill->radial.dr - fill->radial.dx * fill->radial.dx - fill->radial.dy * fill->radial.dy;
    }

    if (fill->radial.a > 0) fill->radial.invA = 1.0f / fill->radial.a;

    auto gradTransform = radial->transform();
    bool isTransformation = !mathIdentity((const Matrix*)(&gradTransform));

    if (transform) {
        if (isTransformation) gradTransform = mathMultiply(transform, &gradTransform);
        else {
            gradTransform = *transform;
            isTransformation = true;
        }
    }

    if (isTransformation) {
        Matrix invTransform;
        if (!mathInverse(&gradTransform, &invTransform)) return false;
        fill->radial.a11 = invTransform.e11;
        fill->radial.a12 = invTransform.e12;
        fill->radial.a13 = invTransform.e13;
        fill->radial.a21 = invTransform.e21;
        fill->radial.a22 = invTransform.e22;
        fill->radial.a23 = invTransform.e23;
    } else {
        fill->radial.a11 = fill->radial.a22 = 1.0f;
        fill->radial.a12 = fill->radial.a13 = 0.0f;
        fill->radial.a21 = fill->radial.a23 = 0.0f;
    }
    return true;
}


static inline uint32_t _clamp(const SwFill* fill, int32_t pos)
{
    switch (fill->spread) {
        case FillSpread::Pad: {
            if (pos >= GRADIENT_STOP_SIZE) pos = GRADIENT_STOP_SIZE - 1;
            else if (pos < 0) pos = 0;
            break;
        }
        case FillSpread::Repeat: {
            pos = pos % GRADIENT_STOP_SIZE;
            if (pos < 0) pos = GRADIENT_STOP_SIZE + pos;
            break;
        }
        case FillSpread::Reflect: {
            auto limit = GRADIENT_STOP_SIZE * 2;
            pos = pos % limit;
            if (pos < 0) pos = limit + pos;
            if (pos >= GRADIENT_STOP_SIZE) pos = (limit - pos - 1);
            break;
        }
    }
    return pos;
}


static inline uint32_t _fixedPixel(const SwFill* fill, int32_t pos)
{
    int32_t i = (pos + (FIXPT_SIZE / 2)) >> FIXPT_BITS;
    return fill->ctable[_clamp(fill, i)];
}


static inline uint32_t _pixel(const SwFill* fill, float pos)
{
    auto i = static_cast<int32_t>(pos * (GRADIENT_STOP_SIZE - 1) + 0.5f);
    return fill->ctable[_clamp(fill, i)];
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/


void fillRadial(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwAlpha alpha, uint8_t csize, uint8_t opacity)
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


void fillRadial(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlender op, uint8_t a)
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


void fillRadial(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, SwMask maskOp, uint8_t a)
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


void fillRadial(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwMask maskOp, uint8_t a)
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
            deltaDet += deltaDeltaDet;
            b += deltaB;
        }
    }
}


void fillRadial(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlender op, SwBlender op2, uint8_t a)
{
    if (fill->radial.a < RADIAL_A_THRESHOLD) {
        auto radial = &fill->radial;
        auto rx = (x + 0.5f) * radial->a11 + (y + 0.5f) * radial->a12 + radial->a13 - radial->fx;
        auto ry = (x + 0.5f) * radial->a21 + (y + 0.5f) * radial->a22 + radial->a23 - radial->fy;

        if (a == 255) {
            for (uint32_t i = 0; i < len; ++i, ++dst) {
                auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
                auto tmp = op(_pixel(fill, x0), *dst, 255);
                *dst = op2(tmp, *dst, 255);
                rx += radial->a11;
                ry += radial->a21;
            }
        } else {
            for (uint32_t i = 0; i < len; ++i, ++dst) {
                auto x0 = 0.5f * (rx * rx + ry * ry - radial->fr * radial->fr) / (radial->dr * radial->fr + rx * radial->dx + ry * radial->dy);
                auto tmp = op(_pixel(fill, x0), *dst, 255);
                auto tmp2 = op2(tmp, *dst, 255);
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
                *dst = op2(tmp, *dst, 255);
                det += deltaDet;
                deltaDet += deltaDeltaDet;
                b += deltaB;
            }
        } else {
            for (uint32_t i = 0 ; i < len ; ++i, ++dst) {
                auto tmp = op(_pixel(fill, sqrtf(det) - b), *dst, 255);
                auto tmp2 = op2(tmp, *dst, 255);
                *dst = INTERPOLATE(tmp2, *dst, a);
                det += deltaDet;
                deltaDet += deltaDeltaDet;
                b += deltaB;
            }
        }
    }
}


void fillLinear(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwAlpha alpha, uint8_t csize, uint8_t opacity)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (GRADIENT_STOP_SIZE - 1);
    float inc = (fill->linear.dx) * (GRADIENT_STOP_SIZE - 1);

    if (opacity == 255) {
        if (mathZero(inc)) {
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
                *dst = opBlendNormal(_pixel(fill, t / GRADIENT_STOP_SIZE), *dst, alpha(cmp));
                ++dst;
                t += inc;
                cmp += csize;
            }
        }
    } else {
        if (mathZero(inc)) {
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
                *dst = opBlendNormal(_pixel(fill, t / GRADIENT_STOP_SIZE), *dst, MULTIPLY(opacity, alpha(cmp)));
                ++dst;
                t += inc;
                cmp += csize;
            }
        }
    }
}


void fillLinear(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, SwMask maskOp, uint8_t a)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (GRADIENT_STOP_SIZE - 1);
    float inc = (fill->linear.dx) * (GRADIENT_STOP_SIZE - 1);

    if (mathZero(inc)) {
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
            auto src = MULTIPLY(_fixedPixel(fill, t2), a);
            *dst = maskOp(src, *dst, ~src);
            t2 += inc2;
        }
    //we have to fallback to float math
    } else {
        uint32_t counter = 0;
        while (counter++ < len) {
            auto src = MULTIPLY(_pixel(fill, t / GRADIENT_STOP_SIZE), a);
            *dst = maskOp(src, *dst, ~src);
            ++dst;
            t += inc;
        }
    }
}


void fillLinear(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwMask maskOp, uint8_t a)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (GRADIENT_STOP_SIZE - 1);
    float inc = (fill->linear.dx) * (GRADIENT_STOP_SIZE - 1);

    if (mathZero(inc)) {
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
            auto src = MULTIPLY(A(_pixel(fill, t / GRADIENT_STOP_SIZE)), a);
            auto tmp = maskOp(src, *cmp, 0);
            *dst = tmp + MULTIPLY(*dst, ~tmp);
            ++dst;
            ++cmp;
            t += inc;
        }
    }
}


void fillLinear(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlender op, uint8_t a)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (GRADIENT_STOP_SIZE - 1);
    float inc = (fill->linear.dx) * (GRADIENT_STOP_SIZE - 1);

    if (mathZero(inc)) {
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
            *dst = op(_pixel(fill, t / GRADIENT_STOP_SIZE), *dst, a);
            ++dst;
            t += inc;
        }
    }
}


void fillLinear(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlender op, SwBlender op2, uint8_t a)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (GRADIENT_STOP_SIZE - 1);
    float inc = (fill->linear.dx) * (GRADIENT_STOP_SIZE - 1);

    if (mathZero(inc)) {
        auto color = _fixedPixel(fill, static_cast<int32_t>(t * FIXPT_SIZE));
        if (a == 255) {
            for (uint32_t i = 0; i < len; ++i, ++dst) {
                auto tmp = op(color, *dst, a);
                *dst = op2(tmp, *dst, 255);
            }
        } else {
            for (uint32_t i = 0; i < len; ++i, ++dst) {
                auto tmp = op(color, *dst, a);
                auto tmp2 = op2(tmp, *dst, 255);
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
                *dst = op2(tmp, *dst, 255);
                t2 += inc2;
            }
        //we have to fallback to float math
        } else {
            uint32_t counter = 0;
            while (counter++ < len) {
                auto tmp = op(_pixel(fill, t / GRADIENT_STOP_SIZE), *dst, 255);
                *dst = op2(tmp, *dst, 255);
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
                auto tmp2 = op2(tmp, *dst, 255);
                *dst = INTERPOLATE(tmp2, *dst, a);
                t2 += inc2;
            }
        //we have to fallback to float math
        } else {
            uint32_t counter = 0;
            while (counter++ < len) {
                auto tmp = op(_pixel(fill, t / GRADIENT_STOP_SIZE), *dst, 255);
                auto tmp2 = op2(tmp, *dst, 255);
                *dst = INTERPOLATE(tmp2, *dst, a);
                ++dst;
                t += inc;
            }
        }
    }
}


bool fillGenColorTable(SwFill* fill, const Fill* fdata, const Matrix* transform, SwSurface* surface, uint8_t opacity, bool ctable)
{
    if (!fill) return false;

    fill->spread = fdata->spread();

    if (ctable) {
        if (!_updateColorTable(fill, fdata, surface, opacity)) return false;
    }

    if (fdata->identifier() == TVG_CLASS_ID_LINEAR) {
        return _prepareLinear(fill, static_cast<const LinearGradient*>(fdata), transform);
    } else if (fdata->identifier() == TVG_CLASS_ID_RADIAL) {
        return _prepareRadial(fill, static_cast<const RadialGradient*>(fdata), transform);
    }

    //LOG: What type of gradient?!

    return false;
}


void fillReset(SwFill* fill)
{
    if (fill->ctable) {
        free(fill->ctable);
        fill->ctable = nullptr;
    }
    fill->translucent = false;
}


void fillFree(SwFill* fill)
{
    if (!fill) return;

    if (fill->ctable) free(fill->ctable);

    free(fill);
}
