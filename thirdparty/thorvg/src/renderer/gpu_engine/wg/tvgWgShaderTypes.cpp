/*
 * Copyright (c) 2023 - 2026 ThorVG project. All rights reserved.

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

#include "tvgWgShaderTypes.h"
#include <cassert>
#include "tvgMath.h"
#include "tvgFill.h"

//************************************************************************
// WgShaderTypeMat4x4f
//************************************************************************

WgShaderTypeMat4x4f::WgShaderTypeMat4x4f()
{
    identity();
}


WgShaderTypeMat4x4f::WgShaderTypeMat4x4f(const Matrix& transform)
{
    update(transform);
}


void WgShaderTypeMat4x4f::identity()
{
    mat[0]  = 1.0f; mat[1]  = 0.0f; mat[2]  = 0.0f; mat[3]  = 0.0f;
    mat[4]  = 0.0f; mat[5]  = 1.0f; mat[6]  = 0.0f; mat[7]  = 0.0f;
    mat[8]  = 0.0f; mat[9]  = 0.0f; mat[10] = 1.0f; mat[11] = 0.0f;
    mat[12] = 0.0f; mat[13] = 0.0f; mat[14] = 0.0f; mat[15] = 1.0f;
}


WgShaderTypeMat4x4f::WgShaderTypeMat4x4f(size_t w, size_t h)
{
    update(w, h);
}


void WgShaderTypeMat4x4f::update(const Matrix& transform)
{

    mat[0]  = transform.e11;
    mat[1]  = transform.e21;
    mat[2]  = 0.0f;
    mat[3]  = transform.e31;
    mat[4]  = transform.e12;
    mat[5]  = transform.e22;
    mat[6]  = 0.0f;
    mat[7]  = transform.e32;
    mat[8]  = 0.0f;
    mat[9]  = 0.0f;
    mat[10] = 1.0f;
    mat[11] = 0.0f;
    mat[12] = transform.e13;
    mat[13] = transform.e23;
    mat[14] = 0.0f;
    mat[15] = transform.e33;
}


void WgShaderTypeMat4x4f::update(size_t w, size_t h)
{
    mat[0]  = +2.0f / w; mat[1]  = +0.0f;     mat[2]  = +0.0f; mat[3]  = +0.0f;
    mat[4]  = +0.0f;     mat[5]  = -2.0f / h; mat[6]  = +0.0f; mat[7]  = +0.0f;
    mat[8]  = +0.0f;     mat[9]  = +0.0f;     mat[10] = -1.0f; mat[11] = +0.0f;
    mat[12] = -1.0f;     mat[13] = +1.0f;     mat[14] = +0.0f; mat[15] = +1.0f;
}

//************************************************************************
// WgShaderTypeVec4f
//************************************************************************

WgShaderTypeVec4f::WgShaderTypeVec4f(const ColorSpace colorSpace, uint8_t o)
{
    update(colorSpace, o);
}


WgShaderTypeVec4f::WgShaderTypeVec4f(const RenderColor& c)
{
    update(c);
}


void WgShaderTypeVec4f::update(const ColorSpace colorSpace, uint8_t o)
{
    vec[0] = (uint32_t)colorSpace;
    vec[3] = o / 255.0f;
}


void WgShaderTypeVec4f::update(const RenderColor& c)
{
    vec[0] = c.r / 255.0f; // red
    vec[1] = c.g / 255.0f; // green
    vec[2] = c.b / 255.0f; // blue
    vec[3] = c.a / 255.0f; // alpha
}

void WgShaderTypeVec4f::update(const RenderRegion& r)
{
    vec[0] = r.min.x;
    vec[1] = r.min.y;
    vec[2] = r.max.x - 1;
    vec[3] = r.max.y - 1;
}

//************************************************************************
// WgShaderTypeGradSettings
//************************************************************************

void WgShaderTypeGradSettings::update(const Fill* fill, const Matrix* modelTransform)
{
    assert(fill);
    // update transform matrix
    Matrix invTransform;
    if (inverse(&fill->transform(), &invTransform)) {
        Matrix invModel;
        if (modelTransform && inverse(modelTransform, &invModel)) invTransform = invTransform * invModel;
        transform.update(invTransform);
    } else transform.identity();
    // update gradient base points
    if (fill->type() == Type::LinearGradient)
        ((LinearGradient*)fill)->linear(&coords.vec[0], &coords.vec[1], &coords.vec[2], &coords.vec[3]);
    else if (fill->type() == Type::RadialGradient) {
        ((RadialGradient*)fill)->radial(&coords.vec[0], &coords.vec[1], &coords.vec[2], &focal.vec[0], &focal.vec[1], &focal.vec[2]);
        CONST_RADIAL(fill)->correct(focal.vec[0], focal.vec[1], focal.vec[2]);
    }
}

//************************************************************************
// WgShaderTypeGradientData
//************************************************************************

void WgShaderTypeGradientData::update(const Fill* fill)
{
    if (!fill) return;
    const Fill::ColorStop* stops = nullptr;
    auto stopCnt = fill->colorStops(&stops);
    if (stopCnt == 0) return;
    static Array<Fill::ColorStop> sstops(stopCnt);
    sstops.clear();
    sstops.push(stops[0]);
    // filter by increasing offset
    for (uint32_t i = 1; i < stopCnt; i++)
        if (sstops.last().offset < stops[i].offset)
            sstops.push(stops[i]);
        else if (sstops.last().offset == stops[i].offset)
            sstops.last() = stops[i];
    // head
    uint32_t range_s = 0;
    uint32_t range_e = uint32_t(sstops[0].offset * (WG_TEXTURE_GRADIENT_SIZE-1));
    for (uint32_t ti = range_s; (ti < range_e) && (ti < WG_TEXTURE_GRADIENT_SIZE); ti++) {
        data[ti * 4 + 0] = sstops[0].r;
        data[ti * 4 + 1] = sstops[0].g;
        data[ti * 4 + 2] = sstops[0].b;
        data[ti * 4 + 3] = sstops[0].a;
    }
    // body
    for (uint32_t di = 1; di < sstops.count; di++) {
        range_s = uint32_t(sstops[di-1].offset * (WG_TEXTURE_GRADIENT_SIZE-1));
        range_e = uint32_t(sstops[di-0].offset * (WG_TEXTURE_GRADIENT_SIZE-1));
        float delta = 1.0f/(range_e - range_s);
        for (uint32_t ti = range_s; (ti < range_e) && (ti < WG_TEXTURE_GRADIENT_SIZE); ti++) {
            float t = (ti - range_s) * delta;
            data[ti * 4 + 0] = tvg::lerp(sstops[di-1].r, sstops[di].r, t);
            data[ti * 4 + 1] = tvg::lerp(sstops[di-1].g, sstops[di].g, t);
            data[ti * 4 + 2] = tvg::lerp(sstops[di-1].b, sstops[di].b, t);
            data[ti * 4 + 3] = tvg::lerp(sstops[di-1].a, sstops[di].a, t);
        }
    }
    // tail
    const tvg::Fill::ColorStop& colorStopLast = sstops.last();
    range_s = uint32_t(colorStopLast.offset * (WG_TEXTURE_GRADIENT_SIZE-1));
    range_e = WG_TEXTURE_GRADIENT_SIZE;
    for (uint32_t ti = range_s; ti < range_e; ti++) {
        data[ti * 4 + 0] = colorStopLast.r;
        data[ti * 4 + 1] = colorStopLast.g;
        data[ti * 4 + 2] = colorStopLast.b;
        data[ti * 4 + 3] = colorStopLast.a;
    }
}

//************************************************************************
// WgShaderTypeEffectParams
//************************************************************************

bool WgShaderTypeEffectParams::update(RenderEffectGaussianBlur* gaussian, const Matrix& transform)
{
    assert(gaussian);
    params[0] = gaussian->sigma;
    params[1] = std::sqrt(transform.e11 * transform.e11 + transform.e12 * transform.e12);
    params[2] = 2 * gaussian->sigma * params[1];
    extend = params[2] * 2; // kernel
    gaussian->valid = (extend > 0);
    return gaussian->valid;
}


bool WgShaderTypeEffectParams::update(RenderEffectDropShadow* dropShadow, const Matrix& transform)
{
    assert(dropShadow);
    const auto scale = std::sqrt(transform.e11 * transform.e11 + transform.e12 * transform.e12);
    const auto kernel = 2 * dropShadow->sigma * scale;
    const auto radian = tvg::deg2rad(90.0f - dropShadow->angle) - tvg::radian(transform);
    const Point offset = {dropShadow->distance * cosf(radian) * scale, -dropShadow->distance * sinf(radian) * scale};
    params[0] = dropShadow->sigma;
    params[1] = scale;
    params[2] = kernel;
    params[3] = 0.0f;
    params[7] = dropShadow->color[3] / 255.0f; // alpha
    //Color is premultiplied to avoid multiplication in the fragment shader:
    params[4] = dropShadow->color[0] / 255.0f * params[7]; // red
    params[5] = dropShadow->color[1] / 255.0f * params[7]; // green
    params[6] = dropShadow->color[2] / 255.0f * params[7]; // blue
    params[8] = offset.x;
    params[9] = offset.y;
    extend = 2 * std::max(dropShadow->sigma * scale + std::abs(offset.x), dropShadow->sigma * scale + std::abs(offset.y));

    dropShadow->valid = (extend >= 0);
    return dropShadow->valid;
}


bool WgShaderTypeEffectParams::update(RenderEffectFill* fill)
{
    params[0] = fill->color[0] / 255.0f;
    params[1] = fill->color[1] / 255.0f;
    params[2] = fill->color[2] / 255.0f;
    params[3] = fill->color[3] / 255.0f;

    fill->valid = true;
    return true;
}


bool WgShaderTypeEffectParams::update(RenderEffectTint* tint)
{
    params[0] = tint->black[0] / 255.0f;
    params[1] = tint->black[1] / 255.0f;
    params[2] = tint->black[2] / 255.0f;
    params[3] = 0.0f;
    params[4] = tint->white[0] / 255.0f;
    params[5] = tint->white[1] / 255.0f;
    params[6] = tint->white[2] / 255.0f;
    params[7] = 0.0f;
    params[8] = tint->intensity / 255.0f;

    tint->valid = (tint->intensity > 0);
    return tint->valid;
}


bool WgShaderTypeEffectParams::update(RenderEffectTritone* tritone)
{
    params[0] = tritone->shadow[0] / 255.0f;
    params[1] = tritone->shadow[1] / 255.0f;
    params[2] = tritone->shadow[2] / 255.0f;
    params[3] = 0.0f;
    params[4] = tritone->midtone[0] / 255.0f;
    params[5] = tritone->midtone[1] / 255.0f;
    params[6] = tritone->midtone[2] / 255.0f;
    params[7] = 0.0f;
    params[8] = tritone->highlight[0] / 255.0f;
    params[9] = tritone->highlight[1] / 255.0f;
    params[10] = tritone->highlight[2] / 255.0f;
    params[11] = tritone->blender / 255.0f;

    tritone->valid = tritone->blender < 255;
    return true;
}
