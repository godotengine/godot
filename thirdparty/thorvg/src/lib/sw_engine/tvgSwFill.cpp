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
#include "tvgMath.h"
#include "tvgSwCommon.h"


/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

#define GRADIENT_STOP_SIZE 1024
#define FIXPT_BITS 8
#define FIXPT_SIZE (1<<FIXPT_BITS)


static bool _updateColorTable(SwFill* fill, const Fill* fdata, const SwSurface* surface, uint32_t opacity)
{
    if (!fill->ctable) {
        fill->ctable = static_cast<uint32_t*>(malloc(GRADIENT_STOP_SIZE * sizeof(uint32_t)));
        if (!fill->ctable) return false;
    }

    const Fill::ColorStop* colors;
    auto cnt = fdata->colorStops(&colors);
    if (cnt == 0 || !colors) return false;

    auto pColors = colors;

    auto a = (pColors->a * opacity) / 255;
    if (a < 255) fill->translucent = true;

    auto r = pColors->r;
    auto g = pColors->g;
    auto b = pColors->b;
    auto rgba = surface->blender.join(r, g, b, a);

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
        auto a2 = (next->a * opacity) / 255;
        if (!fill->translucent && a2 < 255) fill->translucent = true;

        auto rgba2 = surface->blender.join(next->r, next->g, next->b, a2);

        while (pos < next->offset && i < GRADIENT_STOP_SIZE) {
            auto t = (pos - curr->offset) * delta;
            auto dist = static_cast<int32_t>(255 * t);
            auto dist2 = 255 - dist;

            auto color = INTERPOLATE(dist2, rgba, rgba2);
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
    float radius, cx, cy;
    if (radial->radial(&cx, &cy, &radius) != Result::Success) return false;
    if (radius < FLT_EPSILON) return true;

    float invR = 1.0f / radius;
    fill->radial.shiftX = -cx;
    fill->radial.shiftY = -cy;
    fill->radial.a = radius;

    auto gradTransform = radial->transform();
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

        fill->radial.a11 = invTransform.e11 * invR;
        fill->radial.a12 = invTransform.e12 * invR;
        fill->radial.shiftX += invTransform.e13;
        fill->radial.a21 = invTransform.e21 * invR;
        fill->radial.a22 = invTransform.e22 * invR;
        fill->radial.shiftY += invTransform.e23;
        fill->radial.detSecDeriv = 2.0f * fill->radial.a11 * fill->radial.a11 + 2 * fill->radial.a21 * fill->radial.a21;

        fill->radial.a *= sqrt(pow(invTransform.e11, 2) + pow(invTransform.e21, 2));
    } else {
        fill->radial.a11 = fill->radial.a22 = invR;
        fill->radial.a12 = fill->radial.a21 = 0.0f;
        fill->radial.detSecDeriv = 2.0f * invR * invR;
    }
    fill->radial.shiftX *= invR;
    fill->radial.shiftY *= invR;

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

void fillFetchRadial(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len)
{
    auto rx = (x + 0.5f) * fill->radial.a11 + (y + 0.5f) * fill->radial.a12 + fill->radial.shiftX;
    auto ry = (x + 0.5f) * fill->radial.a21 + (y + 0.5f) * fill->radial.a22 + fill->radial.shiftY;

    // detSecondDerivative = d(detFirstDerivative)/dx = d( d(det)/dx )/dx
    auto detSecondDerivative = fill->radial.detSecDeriv;
    // detFirstDerivative = d(det)/dx
    auto detFirstDerivative = 2.0f * (fill->radial.a11 * rx + fill->radial.a21 * ry) + 0.5f * detSecondDerivative;
    auto det = rx * rx + ry * ry;

    for (uint32_t i = 0 ; i < len ; ++i) {
        *dst = _pixel(fill, sqrtf(det));
        ++dst;
        det += detFirstDerivative;
        detFirstDerivative += detSecondDerivative;
    }
}


void fillFetchLinear(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len)
{
    //Rotation
    float rx = x + 0.5f;
    float ry = y + 0.5f;
    float t = (fill->linear.dx * rx + fill->linear.dy * ry + fill->linear.offset) * (GRADIENT_STOP_SIZE - 1);
    float inc = (fill->linear.dx) * (GRADIENT_STOP_SIZE - 1);

    if (mathZero(inc)) {
        auto color = _fixedPixel(fill, static_cast<int32_t>(t * FIXPT_SIZE));
        rasterRGBA32(dst, color, 0, len);
        return;
    }

    auto vMax = static_cast<float>(INT32_MAX >> (FIXPT_BITS + 1));
    auto vMin = -vMax;
    auto v = t + (inc * len);

    //we can use fixed point math
    if (v < vMax && v > vMin) {
        auto t2 = static_cast<int32_t>(t * FIXPT_SIZE);
        auto inc2 = static_cast<int32_t>(inc * FIXPT_SIZE);
        for (uint32_t j = 0; j < len; ++j) {
            *dst = _fixedPixel(fill, t2);
            ++dst;
            t2 += inc2;
        }
    //we have to fallback to float math
    } else {
        uint32_t counter = 0;
        while (counter++ < len) {
            *dst = _pixel(fill, t / GRADIENT_STOP_SIZE);
            ++dst;
            t += inc;
        }
    }
}


bool fillGenColorTable(SwFill* fill, const Fill* fdata, const Matrix* transform, SwSurface* surface, uint32_t opacity, bool ctable)
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
