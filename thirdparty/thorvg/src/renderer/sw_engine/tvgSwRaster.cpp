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
#include "tvgRender.h"
#include "tvgSwCommon.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/

constexpr auto DOWN_SCALE_TOLERANCE = 0.5f;

struct FillLinear
{
    void operator()(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, SwMask op, uint8_t a)
    {
        fillLinear(fill, dst, y, x, len, op, a);
    }

    void operator()(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwMask op, uint8_t a)
    {
        fillLinear(fill, dst, y, x, len, cmp, op, a);
    }

    void operator()(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, uint8_t a)
    {
        fillLinear(fill, dst, y, x, len, op, a);
    }

    void operator()(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwAlpha alpha, uint8_t csize, uint8_t opacity)
    {
        fillLinear(fill, dst, y, x, len, cmp, alpha, csize, opacity);
    }

    void operator()(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, SwBlender op2, uint8_t a)
    {
        fillLinear(fill, dst, y, x, len, op, op2, a);
    }

};

struct FillRadial
{
    void operator()(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, SwMask op, uint8_t a)
    {
        fillRadial(fill, dst, y, x, len, op, a);
    }

    void operator()(const SwFill* fill, uint8_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwMask op, uint8_t a)
    {
        fillRadial(fill, dst, y, x, len, cmp, op, a);
    }

    void operator()(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, uint8_t a)
    {
        fillRadial(fill, dst, y, x, len, op, a);
    }

    void operator()(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, uint8_t* cmp, SwAlpha alpha, uint8_t csize, uint8_t opacity)
    {
        fillRadial(fill, dst, y, x, len, cmp, alpha, csize, opacity);
    }

    void operator()(const SwFill* fill, uint32_t* dst, uint32_t y, uint32_t x, uint32_t len, SwBlenderA op, SwBlender op2, uint8_t a)
    {
        fillRadial(fill, dst, y, x, len, op, op2, a);
    }
};


static inline uint8_t _alpha(uint8_t* a)
{
    return *a;
}


static inline uint8_t _ialpha(uint8_t* a)
{
    return ~(*a);
}


static inline uint8_t _abgrLuma(uint8_t* c)
{
    auto v = *(uint32_t*)c;
    return ((((v&0xff)*54) + (((v>>8)&0xff)*182) + (((v>>16)&0xff)*19))) >> 8; //0.2126*R + 0.7152*G + 0.0722*B
}


static inline uint8_t _argbLuma(uint8_t* c)
{
    auto v = *(uint32_t*)c;
    return ((((v&0xff)*19) + (((v>>8)&0xff)*182) + (((v>>16)&0xff)*54))) >> 8; //0.0722*B + 0.7152*G + 0.2126*R
}


static inline uint8_t _abgrInvLuma(uint8_t* c)
{
    return ~_abgrLuma(c);
}


static inline uint8_t _argbInvLuma(uint8_t* c)
{
    return ~_argbLuma(c);
}


static inline uint32_t _abgrJoin(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    return (a << 24 | b << 16 | g << 8 | r);
}


static inline uint32_t _argbJoin(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    return (a << 24 | r << 16 | g << 8 | b);
}

static inline bool _blending(const SwSurface* surface)
{
    return (surface->blender) ? true : false;
}


/* OPTIMIZE_ME: Probably, we can separate masking(8bits) / composition(32bits)
   This would help to enhance the performance by avoiding the unnecessary matting from the composition */
static inline bool _compositing(const SwSurface* surface)
{
    if (!surface->compositor || surface->compositor->method == MaskMethod::None) return false;
    return true;
}


static inline bool _matting(const SwSurface* surface)
{
    if ((int)surface->compositor->method < (int)MaskMethod::Add) return true;
    else return false;
}

static inline uint8_t _opMaskNone(uint8_t s, TVG_UNUSED uint8_t d, TVG_UNUSED uint8_t a)
{
    return s;
}

static inline uint8_t _opMaskAdd(uint8_t s, uint8_t d, uint8_t a)
{
    return s + MULTIPLY(d, a);
}


static inline uint8_t _opMaskSubtract(uint8_t s, uint8_t d, TVG_UNUSED uint8_t a)
{
   return MULTIPLY(s, 255 - d);
}


static inline uint8_t _opMaskIntersect(uint8_t s, uint8_t d, TVG_UNUSED uint8_t a)
{
   return MULTIPLY(s, d);
}


static inline uint8_t _opMaskDifference(uint8_t s, uint8_t d, uint8_t a)
{
    return MULTIPLY(s, 255 - d) + MULTIPLY(d, a);
}


static inline uint8_t _opMaskLighten(uint8_t s, uint8_t d, uint8_t a)
{
    return (s > d) ? s : d;
}


static inline uint8_t _opMaskDarken(uint8_t s, uint8_t d, uint8_t a)
{
    return (s < d) ? s : d;
}


static inline bool _direct(MaskMethod method)
{
    if (method == MaskMethod::Subtract || method == MaskMethod::Intersect || method == MaskMethod::Darken) return true;
    return false;
}


static inline SwMask _getMaskOp(MaskMethod method)
{
    switch (method) {
        case MaskMethod::Add: return _opMaskAdd;
        case MaskMethod::Subtract: return _opMaskSubtract;
        case MaskMethod::Difference: return _opMaskDifference;
        case MaskMethod::Intersect: return _opMaskIntersect;
        case MaskMethod::Lighten: return _opMaskLighten;
        case MaskMethod::Darken: return _opMaskDarken;
        default: return nullptr;
    }
}


static bool _compositeMaskImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox)
{
    auto dbuffer = &surface->buf8[bbox.min.y * surface->stride + bbox.min.x];
    auto sbuffer = image.buf8 + (bbox.min.y + image.oy) * image.stride + (bbox.min.x + image.ox);

    for (auto y = bbox.min.y; y < bbox.max.y; ++y) {
        auto dst = dbuffer;
        auto src = sbuffer;
        for (auto x = bbox.min.x; x < bbox.max.x; x++, dst++, src++) {
            *dst = *src + MULTIPLY(*dst, ~*src);
        }
        dbuffer += surface->stride;
        sbuffer += image.stride;
    }
    return true;
}


#include "tvgSwRasterTexmap.h"
#include "tvgSwRasterC.h"
#include "tvgSwRasterAvx.h"
#include "tvgSwRasterNeon.h"


static inline uint32_t _sampleSize(float scale)
{
    auto sampleSize = static_cast<uint32_t>(0.5f / scale);
    if (sampleSize == 0) sampleSize = 1;
    return sampleSize;
}


//Bilinear Interpolation
//OPTIMIZE_ME: Skip the function pointer access
static uint32_t _interpUpScaler(const uint32_t *img, TVG_UNUSED uint32_t stride, uint32_t w, uint32_t h, float sx, float sy, TVG_UNUSED int32_t miny, TVG_UNUSED int32_t maxy, TVG_UNUSED int32_t n)
{
    auto rx = (size_t)(sx);
    auto ry = (size_t)(sy);
    auto rx2 = rx + 1;
    if (rx2 >= w) rx2 = w - 1;
    auto ry2 = ry + 1;
    if (ry2 >= h) ry2 = h - 1;

    auto dx = (sx > 0.0f) ? static_cast<uint8_t>((sx - rx) * 255.0f) : 0;
    auto dy = (sy > 0.0f) ? static_cast<uint8_t>((sy - ry) * 255.0f) : 0;

    auto c1 = img[rx + ry * w];
    auto c2 = img[rx2 + ry * w];
    auto c3 = img[rx + ry2 * w];
    auto c4 = img[rx2 + ry2 * w];

    return INTERPOLATE(INTERPOLATE(c4, c3, dx), INTERPOLATE(c2, c1, dx), dy);
}


//2n x 2n Mean Kernel
//OPTIMIZE_ME: Skip the function pointer access
static uint32_t _interpDownScaler(const uint32_t *img, uint32_t stride, uint32_t w, uint32_t h, float sx, TVG_UNUSED float sy, int32_t miny, int32_t maxy, int32_t n)
{
    size_t c[4] = {0, 0, 0, 0};

    int32_t minx = (int32_t)sx - n;
    if (minx < 0) minx = 0;

    int32_t maxx = (int32_t)sx + n;
    if (maxx >= (int32_t)w) maxx = w;

    int32_t inc = (n / 2) + 1;
    n = 0;

    auto src = img + minx + miny * stride;

    for (auto y = miny; y < maxy; y += inc) {
        auto p = src;
        for (auto x = minx; x < maxx; x += inc, p += inc) {
            c[0] += A(*p);
            c[1] += C1(*p);
            c[2] += C2(*p);
            c[3] += C3(*p);
            ++n;
        }
        src += (stride * inc);
    }

    c[0] /= n;
    c[1] /= n;
    c[2] /= n;
    c[3] /= n;

    return (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3];
}


/************************************************************************/
/* Rect                                                                 */
/************************************************************************/

static bool _rasterCompositeMaskedRect(SwSurface* surface, const RenderRegion& bbox, SwMask maskOp, uint8_t a)
{
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.min.y * cstride + bbox.min.x);   //compositor buffer
    auto ialpha = 255 - a;

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        auto cmp = cbuffer;
        for (uint32_t x = 0; x < bbox.w(); ++x, ++cmp) {
            *cmp = maskOp(a, *cmp, ialpha);
        }
        cbuffer += cstride;
    }
    return _compositeMaskImage(surface, surface->compositor->image, surface->compositor->bbox);
}


static bool _rasterDirectMaskedRect(SwSurface* surface, const RenderRegion& bbox, SwMask maskOp, uint8_t a)
{
    auto cbuffer = surface->compositor->image.buf8 + (bbox.min.y * surface->compositor->image.stride + bbox.min.x);   //compositor buffer
    auto dbuffer = surface->buf8 + (bbox.min.y * surface->stride + bbox.min.x);   //destination buffer

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        auto cmp = cbuffer;
        auto dst = dbuffer;
        for (uint32_t x = 0; x < bbox.w(); ++x, ++cmp, ++dst) {
            auto tmp = maskOp(a, *cmp, 0);   //not use alpha.
            *dst = tmp + MULTIPLY(*dst, ~tmp);
        }
        cbuffer += surface->compositor->image.stride;
        dbuffer += surface->stride;
    }
    return true;
}


static bool _rasterMaskedRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    //8bit masking channels composition
    if (surface->channelSize != sizeof(uint8_t)) return false;

    TVGLOG("SW_ENGINE", "Masked(%d) Rect [Region: %d %d %d %d]", (int)surface->compositor->method, bbox.min.x, bbox.min.y, bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y);

    auto maskOp = _getMaskOp(surface->compositor->method);
    if (_direct(surface->compositor->method)) return _rasterDirectMaskedRect(surface, bbox, maskOp, c.a);
    else return _rasterCompositeMaskedRect(surface, bbox, maskOp, c.a);
    return false;
}


static bool _rasterMattedRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    auto csize = surface->compositor->image.channelSize;
    auto cbuffer = surface->compositor->image.buf8 + ((bbox.min.y * surface->compositor->image.stride + bbox.min.x) * csize);   //compositor buffer
    auto alpha = surface->alpha(surface->compositor->method);

    TVGLOG("SW_ENGINE", "Matted(%d) Rect [Region: %u %u %u %u]", (int)surface->compositor->method, bbox.x(), bbox.y(), bbox.w(), bbox.h());

    //32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, c.a);
        auto buffer = surface->buf32 + (bbox.min.y * surface->stride) + bbox.min.x;
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            auto dst = &buffer[y * surface->stride];
            auto cmp = &cbuffer[y * surface->compositor->image.stride * csize];
            for (uint32_t x = 0; x < bbox.w(); ++x, ++dst, cmp += csize) {
                auto tmp = ALPHA_BLEND(color, alpha(cmp));
                *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
            }
        }
    //8bits grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto buffer = surface->buf8 + (bbox.min.y * surface->stride) + bbox.min.x;
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            auto dst = &buffer[y * surface->stride];
            auto cmp = &cbuffer[y * surface->compositor->image.stride * csize];
            for (uint32_t x = 0; x < bbox.w(); ++x, ++dst, cmp += csize) {
                *dst = INTERPOLATE8(c.a, *dst, alpha(cmp));
            }
        }
    }
    return true;
}


static bool _rasterBlendingRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    if (surface->channelSize != sizeof(uint32_t)) return false;

    auto color = surface->join(c.r, c.g, c.b, c.a);
    auto buffer = surface->buf32 + (bbox.min.y * surface->stride) + bbox.min.x;

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        auto dst = &buffer[y * surface->stride];
        for (uint32_t x = 0; x < bbox.w(); ++x, ++dst) {
            *dst = surface->blender(color, *dst);
        }
    }
    return true;
}


static bool _rasterTranslucentRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
#if defined(THORVG_AVX_VECTOR_SUPPORT)
    return avxRasterTranslucentRect(surface, bbox, c);
#elif defined(THORVG_NEON_VECTOR_SUPPORT)
    return neonRasterTranslucentRect(surface, bbox, c);
#else
    return cRasterTranslucentRect(surface, bbox, c);
#endif
}


static bool _rasterSolidRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    //32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, 255);
        auto buffer = surface->buf32 + (bbox.min.y * surface->stride);
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            rasterPixel32(buffer + y * surface->stride, color, bbox.min.x, bbox.w());
        }
        return true;
    }
    //8bits grayscale
    if (surface->channelSize == sizeof(uint8_t)) {
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            rasterGrayscale8(surface->buf8, 255, (y + bbox.min.y) * surface->stride + bbox.min.x, bbox.w());
        }
        return true;
    }
    return false;
}


static bool _rasterRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterMattedRect(surface, bbox, c);
        else return _rasterMaskedRect(surface, bbox, c);
    } else if (_blending(surface)) {
        return _rasterBlendingRect(surface, bbox, c);
    } else {
        if (c.a == 255) return _rasterSolidRect(surface, bbox, c);
        else return _rasterTranslucentRect(surface, bbox, c);
    }
    return false;
}


/************************************************************************/
/* Rle                                                                  */
/************************************************************************/

static bool _rasterCompositeMaskedRle(SwSurface* surface, SwRle* rle, const RenderRegion& bbox, SwMask maskOp, uint8_t a)
{
    auto cbuffer = surface->compositor->image.buf8;
    auto cstride = surface->compositor->image.stride;
    const SwSpan* end;
    int32_t x, len;
    uint8_t src;

    for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
        if (!span->fetch(bbox, x, len)) continue;
        auto cmp = &cbuffer[span->y * cstride + x];
        if (span->coverage == 255) src = a;
        else src = MULTIPLY(a, span->coverage);
        auto ialpha = 255 - src;
        for (auto x = 0; x < len; ++x, ++cmp) {
            *cmp = maskOp(src, *cmp, ialpha);
        }
    }
    return _compositeMaskImage(surface, surface->compositor->image, surface->compositor->bbox);
}


static bool _rasterDirectMaskedRle(SwSurface* surface, SwRle* rle, const RenderRegion& bbox, SwMask maskOp, uint8_t a)
{
    auto cbuffer = surface->compositor->image.buf8;
    auto cstride = surface->compositor->image.stride;
    const SwSpan* end;
    int32_t x, len;
    uint8_t src;

    for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
        if (!span->fetch(bbox, x, len)) continue;
        auto cmp = &cbuffer[span->y * cstride + x];
        auto dst = &surface->buf8[span->y * surface->stride + x];
        if (span->coverage == 255) src = a;
        else src = MULTIPLY(a, span->coverage);
        for (auto x = 0; x < len; ++x, ++cmp, ++dst) {
            auto tmp = maskOp(src, *cmp, 0);     //not use alpha
            *dst = tmp + MULTIPLY(*dst, ~tmp);
        }
    }
    return true;
}


static bool _rasterMaskedRle(SwSurface* surface, SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
    TVGLOG("SW_ENGINE", "Masked(%d) Rle", (int)surface->compositor->method);

    //8bit masking channels composition
    if (surface->channelSize != sizeof(uint8_t)) return false;

    auto maskOp = _getMaskOp(surface->compositor->method);
    if (_direct(surface->compositor->method)) return _rasterDirectMaskedRle(surface, rle, bbox, maskOp, c.a);
    else return _rasterCompositeMaskedRle(surface, rle, bbox, maskOp, c.a);
    return false;
}


static bool _rasterMattedRle(SwSurface* surface, SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
    TVGLOG("SW_ENGINE", "Matted(%d) Rle", (int)surface->compositor->method);

    auto cbuffer = surface->compositor->image.buf8;
    auto csize = surface->compositor->image.channelSize;
    auto alpha = surface->alpha(surface->compositor->method);
    const SwSpan* end;
    int32_t x, len;

    //32bit channels
    if (surface->channelSize == sizeof(uint32_t)) {
        uint32_t src;
        auto color = surface->join(c.r, c.g, c.b, c.a);
        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            auto dst = &surface->buf32[span->y * surface->stride + x];
            auto cmp = &cbuffer[(span->y * surface->compositor->image.stride + x) * csize];
            if (span->coverage == 255) src = color;
            else src = ALPHA_BLEND(color, span->coverage);
            for (auto x = 0; x < len; ++x, ++dst, cmp += csize) {
                auto tmp = ALPHA_BLEND(src, alpha(cmp));
                *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
            }
        }
    //8bit grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        uint8_t src;
        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            auto dst = &surface->buf8[span->y * surface->stride + x];
            auto cmp = &cbuffer[(span->y * surface->compositor->image.stride + x) * csize];
            if (span->coverage == 255) src = c.a;
            else src = MULTIPLY(c.a, span->coverage);
            for (auto x = 0; x < len; ++x, ++dst, cmp += csize) {
                *dst = INTERPOLATE8(src, *dst, alpha(cmp));
            }
        }
    }
    return true;
}


static bool _rasterBlendingRle(SwSurface* surface, const SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
    if (surface->channelSize != sizeof(uint32_t)) return false;

    auto color = surface->join(c.r, c.g, c.b, c.a);
    const SwSpan* end;
    int32_t x, len;

    for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
        if (!span->fetch(bbox, x, len)) continue;
        auto dst = &surface->buf32[span->y * surface->stride + x];
        if (span->coverage == 255) {
            for (auto x = 0; x < len; ++x, ++dst) {
                *dst = surface->blender(color, *dst);
            }
        } else {
            for (auto x = 0; x < len; ++x, ++dst) {
                *dst = INTERPOLATE(surface->blender(color, *dst), *dst, span->coverage);
            }
        }
    }
    return true;
}


static bool _rasterTranslucentRle(SwSurface* surface, const SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
#if defined(THORVG_AVX_VECTOR_SUPPORT)
    return avxRasterTranslucentRle(surface, rle, bbox, c);
#elif defined(THORVG_NEON_VECTOR_SUPPORT)
    return neonRasterTranslucentRle(surface, rle, bbox, c);
#else
    return cRasterTranslucentRle(surface, rle, bbox, c);
#endif
}


static bool _rasterSolidRle(SwSurface* surface, const SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
    const SwSpan* end;
    int32_t x, len;

    //32bit channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, 255);
        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            if (span->coverage == 255) rasterPixel32(surface->buf32 + span->y * surface->stride, color, x, len);
            else {
                auto dst = &surface->buf32[span->y * surface->stride + x];
                auto src = ALPHA_BLEND(color, span->coverage);
                auto ialpha = 255 - span->coverage;
                for (auto x = 0; x < len; ++x, ++dst) {
                    *dst = src + ALPHA_BLEND(*dst, ialpha);
                }
            }
        }
    //8bit grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            if (span->coverage == 255) rasterGrayscale8(surface->buf8, span->coverage, span->y * surface->stride + x, len);
            else {
                auto dst = &surface->buf8[span->y * surface->stride + x];
                auto ialpha = 255 - span->coverage;
                for (auto x = 0; x < len; ++x, ++dst) {
                    *dst = span->coverage + MULTIPLY(*dst, ialpha);
                }
            }
        }
    }
    return true;
}


static bool _rasterRle(SwSurface* surface, SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
    if (!rle || rle->invalid()) return false;

    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterMattedRle(surface, rle, bbox, c);
        else return _rasterMaskedRle(surface, rle, bbox, c);
    } else if (_blending(surface)) {
        return _rasterBlendingRle(surface, rle, bbox, c);
    } else {
        if (c.a == 255) return _rasterSolidRle(surface, rle, bbox, c);
        else return _rasterTranslucentRle(surface, rle, bbox, c);
    }
    return false;
}


/************************************************************************/
/* RLE Scaled Image                                                     */
/************************************************************************/

#define SCALED_IMAGE_RANGE_Y(y) \
    auto sy = (y) * itransform->e22 + itransform->e23 - 0.49f; \
    if (sy <= -0.5f || (uint32_t)(sy + 0.5f) >= image.h) continue; \
    if (scaleMethod == _interpDownScaler) { \
        auto my = (int32_t)nearbyint(sy); \
        miny = my - (int32_t)sampleSize; \
        if (miny < 0) miny = 0; \
        maxy = my + (int32_t)sampleSize; \
        if (maxy >= (int32_t)image.h) maxy = (int32_t)image.h; \
    }

#define SCALED_IMAGE_RANGE_X \
    auto sx = (x) * itransform->e11 + itransform->e13 - 0.49f; \
    if (sx <= -0.5f || (uint32_t)(sx + 0.5f) >= image.w) continue; \

static bool _rasterScaledMaskedRleImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    TVGERR("SW_ENGINE", "Not Supported Scaled Masked(%d) Rle Image", (int)surface->compositor->method);
    return false;
}


static bool _rasterScaledMattedRleImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    TVGLOG("SW_ENGINE", "Scaled Matted(%d) Rle Image", (int)surface->compositor->method);

    auto csize = surface->compositor->image.channelSize;
    auto alpha = surface->alpha(surface->compositor->method);
    auto scaleMethod = image.scale < DOWN_SCALE_TOLERANCE ? _interpDownScaler : _interpUpScaler;
    auto sampleSize = _sampleSize(image.scale);
    int32_t miny = 0, maxy = 0;

    ARRAY_FOREACH(span, image.rle->spans) {
        SCALED_IMAGE_RANGE_Y(span->y)
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        auto cmp = &surface->compositor->image.buf8[(span->y * surface->compositor->image.stride + span->x) * csize];
        auto a = MULTIPLY(span->coverage, opacity);
        for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst, cmp += csize) {
            SCALED_IMAGE_RANGE_X
            auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
            src = ALPHA_BLEND(src, (a == 255) ? alpha(cmp) : MULTIPLY(alpha(cmp), a));
            *dst = src + ALPHA_BLEND(*dst, IA(src));
        }
    }
    return true;
}


static bool _rasterScaledBlendingRleImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    auto scaleMethod = image.scale < DOWN_SCALE_TOLERANCE ? _interpDownScaler : _interpUpScaler;
    auto sampleSize = _sampleSize(image.scale);
    int32_t miny = 0, maxy = 0;

    ARRAY_FOREACH(span, image.rle->spans) {
        SCALED_IMAGE_RANGE_Y(span->y)
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        auto alpha = MULTIPLY(span->coverage, opacity);
        if (alpha == 255) {
            for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                SCALED_IMAGE_RANGE_X
                auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                *dst = INTERPOLATE(surface->blender(rasterUnpremultiply(src), *dst), *dst, A(src));
            }
        } else {
            for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                SCALED_IMAGE_RANGE_X
                auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                *dst = INTERPOLATE(surface->blender(rasterUnpremultiply(src), *dst), *dst, MULTIPLY(alpha, A(src)));
            }
        }
    }
    return true;
}


static bool _rasterScaledRleImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    auto scaleMethod = image.scale < DOWN_SCALE_TOLERANCE ? _interpDownScaler : _interpUpScaler;
    auto sampleSize = _sampleSize(image.scale);
    int32_t miny = 0, maxy = 0;

    ARRAY_FOREACH(span, image.rle->spans) {
        SCALED_IMAGE_RANGE_Y(span->y)
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        auto alpha = MULTIPLY(span->coverage, opacity);
        for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
            SCALED_IMAGE_RANGE_X
            auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
            if (alpha < 255) src = ALPHA_BLEND(src, alpha);
            *dst = src + ALPHA_BLEND(*dst, IA(src));
        }
    }
    return true;
}


/************************************************************************/
/* RLE Direct Image                                                     */
/************************************************************************/

static bool _rasterDirectMattedRleImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    TVGLOG("SW_ENGINE", "Direct Matted(%d) Rle Image", (int)surface->compositor->method);

    auto csize = surface->compositor->image.channelSize;
    auto cbuffer = surface->compositor->image.buf8;
    auto alpha = surface->alpha(surface->compositor->method);
    const SwSpan* end;
    int32_t x, len;

    for (auto span = image.rle->fetch(bbox, &end); span < end; ++span) {
        if (!span->fetch(bbox, x, len)) continue;
        auto dst = &surface->buf32[span->y * surface->stride + x];
        auto cmp = &cbuffer[(span->y * surface->compositor->image.stride + x) * csize];
        auto img = image.buf32 + (span->y + image.oy) * image.stride + (x + image.ox);
        auto a = MULTIPLY(span->coverage, opacity);
        if (a == 255) {
            for (auto x = 0; x < len; ++x, ++dst, ++img, cmp += csize) {
                auto tmp = ALPHA_BLEND(*img, alpha(cmp));
                *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
            }
        } else {
            for (auto x = 0; x < len; ++x, ++dst, ++img, cmp += csize) {
                auto tmp = ALPHA_BLEND(*img, MULTIPLY(a, alpha(cmp)));
                *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
            }
        }
    }
    return true;
}


static bool _rasterDirectBlendingRleImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    const SwSpan* end;
    int32_t x, len;

    for (auto span = image.rle->fetch(bbox, &end); span < end; ++span) {
        if (!span->fetch(bbox, x, len)) continue;
        auto dst = &surface->buf32[span->y * surface->stride + x];
        auto src = image.buf32 + (span->y + image.oy) * image.stride + (x + image.ox);
        auto alpha = MULTIPLY(span->coverage, opacity);
        if (alpha == 255) {
            for (auto x = 0; x < len; ++x, ++dst, ++src) {
                *dst = surface->blender(rasterUnpremultiply(*src), *dst);
            }
        } else {
            for (auto x = 0; x < len; ++x, ++dst, ++src) {
                *dst = INTERPOLATE(surface->blender(rasterUnpremultiply(*src), *dst), *dst, MULTIPLY(alpha, A(*src)));
            }
        }
    }
    return true;
}


static bool _rasterDirectRleImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    const SwSpan* end;
    int32_t x, len;

    for (auto span = image.rle->fetch(bbox, &end); span < end; ++span) {
        if (!span->fetch(bbox, x, len)) continue;
        auto dst = &surface->buf32[span->y * surface->stride + x];
        auto img = image.buf32 + (span->y + image.oy) * image.stride + (x + image.ox);
        auto alpha = MULTIPLY(span->coverage, opacity);
        rasterTranslucentPixel32(dst, img, len, alpha);
    }
    return true;
}


static bool _rasterDirectMaskedRleImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    TVGERR("SW_ENGINE", "Not Supported Direct Masked(%d) Rle Image", (int)surface->compositor->method);
    return false;
}


/************************************************************************/
/*Scaled Image                                                          */
/************************************************************************/

static bool _rasterScaledMaskedImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    TVGERR("SW_ENGINE", "Not Supported Scaled Masked Image!");
    return false;
}


static bool _rasterScaledMattedImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale scaled matted image!");
        return false;
    }

    auto dbuffer = surface->buf32 + (bbox.min.y * surface->stride + bbox.min.x);
    auto csize = surface->compositor->image.channelSize;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.min.y * surface->compositor->image.stride + bbox.min.x) * csize;
    auto alpha = surface->alpha(surface->compositor->method);

    TVGLOG("SW_ENGINE", "Scaled Matted(%d) Image [Region: %d %d %d %d]", (int)surface->compositor->method, bbox.min.x, bbox.min.y, bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y);

    auto scaleMethod = image.scale < DOWN_SCALE_TOLERANCE ? _interpDownScaler : _interpUpScaler;
    auto sampleSize = _sampleSize(image.scale);
    int32_t miny = 0, maxy = 0;

    for (auto y = bbox.min.y; y < bbox.max.y; ++y) {
        SCALED_IMAGE_RANGE_Y(y)
        auto dst = dbuffer;
        auto cmp = cbuffer;
        for (auto x = bbox.min.x; x < bbox.max.x; ++x, ++dst, cmp += csize) {
            SCALED_IMAGE_RANGE_X
            auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
            auto tmp = ALPHA_BLEND(src, opacity == 255 ? alpha(cmp) : MULTIPLY(opacity, alpha(cmp)));
            *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
        }
        dbuffer += surface->stride;
        cbuffer += surface->compositor->image.stride * csize;
    }
    return true;
}


static bool _rasterScaledBlendingImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale scaled blending image!");
        return false;
    }

    auto dbuffer = surface->buf32 + (bbox.min.y * surface->stride + bbox.min.x);
    auto scaleMethod = image.scale < DOWN_SCALE_TOLERANCE ? _interpDownScaler : _interpUpScaler;
    auto sampleSize = _sampleSize(image.scale);
    int32_t miny = 0, maxy = 0;

    for (auto y = bbox.min.y; y < bbox.max.y; ++y, dbuffer += surface->stride) {
        SCALED_IMAGE_RANGE_Y(y)
        auto dst = dbuffer;
        for (auto x = bbox.min.x; x < bbox.max.x; ++x, ++dst) {
            SCALED_IMAGE_RANGE_X
            auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
            *dst = INTERPOLATE(surface->blender(rasterUnpremultiply(src), *dst), *dst, MULTIPLY(opacity, A(src)));
        }
    }
    return true;
}


static bool _rasterScaledImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    auto scaleMethod = image.scale < DOWN_SCALE_TOLERANCE ? _interpDownScaler : _interpUpScaler;
    auto sampleSize = _sampleSize(image.scale);
    int32_t miny = 0, maxy = 0;

    //32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto buffer = surface->buf32 + (bbox.min.y * surface->stride + bbox.min.x);
        for (auto y = bbox.min.y; y < bbox.max.y; ++y, buffer += surface->stride) {
            SCALED_IMAGE_RANGE_Y(y)
            auto dst = buffer;
            for (auto x = bbox.min.x; x < bbox.max.x; ++x, ++dst) {
                SCALED_IMAGE_RANGE_X
                auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                if (opacity < 255) src = ALPHA_BLEND(src, opacity);
                *dst = src + ALPHA_BLEND(*dst, IA(src));
            }
        }
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto buffer = surface->buf8 + (bbox.min.y * surface->stride + bbox.min.x);
        for (auto y = bbox.min.y; y < bbox.max.y; ++y, buffer += surface->stride) {
            SCALED_IMAGE_RANGE_Y(y)
            auto dst = buffer;
            for (auto x = bbox.min.x; x < bbox.max.x; ++x, ++dst) {
                SCALED_IMAGE_RANGE_X
                auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                *dst = MULTIPLY(A(src), opacity);
            }
        }
    }
    return true;
}


/************************************************************************/
/* Direct Image                                                         */
/************************************************************************/

static bool _rasterDirectMaskedImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    TVGERR("SW_ENGINE", "Not Supported: Direct Masked Image");
    return false;
}


static bool _rasterDirectMattedImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    auto csize = surface->compositor->image.channelSize;
    auto alpha = surface->alpha(surface->compositor->method);
    auto sbuffer = image.buf32 + (bbox.min.y + image.oy) * image.stride + (bbox.min.x + image.ox);
    auto cbuffer = surface->compositor->image.buf8 + (bbox.min.y * surface->compositor->image.stride + bbox.min.x) * csize; //compositor buffer

    TVGLOG("SW_ENGINE", "Direct Matted(%d) Image  [Region: %u %u %u %u]", (int)surface->compositor->method, bbox.x(), bbox.y(), bbox.w(), bbox.h());

    //32 bits
    if (surface->channelSize == sizeof(uint32_t)) {
        auto dbuffer = surface->buf32 + (bbox.min.y * surface->stride) + bbox.min.x;
        for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
            auto cmp = cbuffer;
            auto src = sbuffer;
            if (opacity == 255) {
                for (auto dst = dbuffer; dst < dbuffer + w; ++dst, ++src, cmp += csize) {
                    auto tmp = ALPHA_BLEND(*src, alpha(cmp));
                    *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
                }
            } else {
                for (auto dst = dbuffer; dst < dbuffer + w; ++dst, ++src, cmp += csize) {
                    auto tmp = ALPHA_BLEND(*src, MULTIPLY(opacity, alpha(cmp)));
                    *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
                }
            }
            cbuffer += surface->compositor->image.stride * csize;
        }
    //8 bits
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto dbuffer = surface->buf8 + (bbox.min.y * surface->stride) + bbox.min.x;
        for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
            auto cmp = cbuffer;
            auto src = sbuffer;
            if (opacity == 255) {
                for (auto dst = dbuffer; dst < dbuffer + w; ++dst, ++src, cmp += csize) {
                    auto tmp = MULTIPLY(A(*src), alpha(cmp));
                    *dst = tmp + MULTIPLY(*dst, 255 - tmp);
                }
            } else {
                for (auto dst = dbuffer; dst < dbuffer + w; ++dst, ++src, cmp += csize) {
                    auto tmp = MULTIPLY(A(*src), MULTIPLY(opacity, alpha(cmp)));
                    *dst = tmp + MULTIPLY(*dst, 255 - tmp);
                }
            }
            cbuffer += surface->compositor->image.stride * csize;
        }
    }
    return true;
}


static bool _rasterDirectBlendingImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale image!");
        return false;
    }

    auto dbuffer = &surface->buf32[bbox.min.y * surface->stride + bbox.min.x];
    auto sbuffer = image.buf32 + (bbox.min.y + image.oy) * image.stride + (bbox.min.x + image.ox);

    for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
        auto src = sbuffer;
        if (opacity == 255) {
            for (auto dst = dbuffer; dst < dbuffer + w; dst++, src++) {
                *dst = INTERPOLATE(surface->blender(rasterUnpremultiply(*src), *dst), *dst, A(*src));
            }
        } else {
            for (auto dst = dbuffer; dst < dbuffer + w; dst++, src++) {
                *dst = INTERPOLATE(surface->blender(rasterUnpremultiply(*src), *dst), *dst, MULTIPLY(opacity, A(*src)));
            }
        }
    }
    return true;
}


static bool _rasterDirectImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    auto sbuffer = image.buf32 + (bbox.min.y + image.oy) * image.stride + (bbox.min.x + image.ox);

    //32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto dbuffer = &surface->buf32[bbox.min.y * surface->stride + bbox.min.x];
        for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
            rasterTranslucentPixel32(dbuffer, sbuffer, w, opacity);
        }
    //8bits grayscale
    //32 -> 8 direct converting seems an avoidable stage. maybe draw to a masking image after an intermediate scene. Can get rid of this?
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto dbuffer = &surface->buf8[bbox.min.y * surface->stride + bbox.min.x];
        for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
            auto src = sbuffer;
            if (opacity == 255) {
                for (auto dst = dbuffer; dst < dbuffer + w; dst++, src++) {
                    *dst = A(*src) + MULTIPLY(*dst, IA(*src));
                }
            } else {
                for (auto dst = dbuffer; dst < dbuffer + w; dst++, src++) {
                    *dst = INTERPOLATE8(A(*src), *dst, opacity);
                }
            }
        }
    }
    return true;
}


static bool _rasterDirectMattedBlendingImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale image!");
        return false;
    }

    auto csize = surface->compositor->image.channelSize;
    auto alpha = surface->alpha(surface->compositor->method);
    auto sbuffer = image.buf32 + (bbox.min.y + image.oy) * image.stride + (bbox.min.x + image.ox);
    auto cbuffer = surface->compositor->image.buf8 + (bbox.min.y * surface->compositor->image.stride + bbox.min.x) * csize; //compositor buffer
    auto dbuffer = surface->buf32 + (bbox.min.y * surface->stride) + bbox.min.x;

    for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
        auto cmp = cbuffer;
        auto src = sbuffer;
        if (opacity == 255) {
            for (auto dst = dbuffer; dst < dbuffer + w; ++dst, ++src, cmp += csize) {
                *dst = INTERPOLATE(surface->blender(*src, *dst), *dst, MULTIPLY(A(*src), alpha(cmp)));
            }
        } else {
            for (auto dst = dbuffer; dst < dbuffer + w; ++dst, ++src, cmp += csize) {
                *dst = INTERPOLATE(surface->blender(*src, *dst), *dst, MULTIPLY(MULTIPLY(A(*src), alpha(cmp)), opacity));
            }
        }
        cbuffer += surface->compositor->image.stride * csize;
    }
    return true;
}


/************************************************************************/
/* Rect Gradient                                                        */
/************************************************************************/

template<typename fillMethod>
static bool _rasterCompositeGradientMaskedRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill, SwMask maskOp)
{
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.min.y * cstride + bbox.min.x);

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillMethod()(fill, cbuffer, bbox.min.y + y, bbox.min.x, bbox.w(), maskOp, 255);
        cbuffer += surface->stride;
    }
    return _compositeMaskImage(surface, surface->compositor->image, surface->compositor->bbox);
}


template<typename fillMethod>
static bool _rasterDirectGradientMaskedRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill, SwMask maskOp)
{
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.min.y * cstride + bbox.min.x);
    auto dbuffer = surface->buf8 + (bbox.min.y * surface->stride + bbox.min.x);

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillMethod()(fill, dbuffer, bbox.min.y + y, bbox.min.x, bbox.w(), cbuffer, maskOp, 255);
        cbuffer += cstride;
        dbuffer += surface->stride;
    }
    return true;
}


template<typename fillMethod>
static bool _rasterGradientMaskedRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    auto method = surface->compositor->method;

    TVGLOG("SW_ENGINE", "Masked(%d) Gradient [Region: %d %d %d %d]", (int)method, bbox.min.x, bbox.min.y, bbox.max.x - bbox.min.x, bbox.max.y - bbox.min.y);

    auto maskOp = _getMaskOp(method);

    if (_direct(method)) return _rasterDirectGradientMaskedRect<fillMethod>(surface, bbox, fill, maskOp);
    else return _rasterCompositeGradientMaskedRect<fillMethod>(surface, bbox, fill, maskOp);

    return false;
}


template<typename fillMethod>
static bool _rasterGradientMattedRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    auto buffer = surface->buf32 + (bbox.min.y * surface->stride) + bbox.min.x;
    auto csize = surface->compositor->image.channelSize;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.min.y * surface->compositor->image.stride + bbox.min.x) * csize;
    auto alpha = surface->alpha(surface->compositor->method);

    TVGLOG("SW_ENGINE", "Matted(%d) Gradient [Region: %u %u %u %u]", (int)surface->compositor->method, bbox.x(), bbox.y(), bbox.w(), bbox.h());

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillMethod()(fill, buffer, bbox.min.y + y, bbox.min.x, bbox.w(), cbuffer, alpha, csize, 255);
        buffer += surface->stride;
        cbuffer += surface->stride * csize;
    }
    return true;
}


template<typename fillMethod>
static bool _rasterBlendingGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    auto buffer = surface->buf32 + (bbox.min.y * surface->stride) + bbox.min.x;

    if (fill->translucent) {
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            fillMethod()(fill, buffer + y * surface->stride, bbox.min.y + y, bbox.min.x, bbox.w(), opBlendPreNormal, surface->blender, 255);
        }
    } else {
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            fillMethod()(fill, buffer + y * surface->stride, bbox.min.y + y, bbox.min.x, bbox.w(), opBlendSrcOver, surface->blender, 255);
        }
    }
    return true;
}

template<typename fillMethod>
static bool _rasterTranslucentGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    //32 bits
    if (surface->channelSize == sizeof(uint32_t)) {
        auto buffer = surface->buf32 + (bbox.min.y * surface->stride) + bbox.min.x;
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            fillMethod()(fill, buffer, bbox.min.y + y, bbox.min.x, bbox.w(), opBlendPreNormal, 255);
            buffer += surface->stride;
        }
    //8 bits
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto buffer = surface->buf8 + (bbox.min.y * surface->stride) + bbox.min.x;
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            fillMethod()(fill, buffer, bbox.min.y + y, bbox.min.x, bbox.w(), _opMaskAdd, 255);
            buffer += surface->stride;
        }
    }
    return true;
}


template<typename fillMethod>
static bool _rasterSolidGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    //32 bits
    if (surface->channelSize == sizeof(uint32_t)) {
        auto buffer = surface->buf32 + (bbox.min.y * surface->stride) + bbox.min.x;
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            fillMethod()(fill, buffer, bbox.min.y + y, bbox.min.x, bbox.w(), opBlendSrcOver, 255);
            buffer += surface->stride;
        }
    //8 bits
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto buffer = surface->buf8 + (bbox.min.y * surface->stride) + bbox.min.x;
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            fillMethod()(fill, buffer, bbox.min.y + y, bbox.min.x, bbox.w(), _opMaskNone, 255);
            buffer += surface->stride;
        }
    }
    return true;
}


static bool _rasterLinearGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterGradientMattedRect<FillLinear>(surface, bbox, fill);
        else return _rasterGradientMaskedRect<FillLinear>(surface, bbox, fill);
    } else if (_blending(surface)) {
        return _rasterBlendingGradientRect<FillLinear>(surface, bbox, fill);
    } else {
        if (fill->translucent) return _rasterTranslucentGradientRect<FillLinear>(surface, bbox, fill);
        else _rasterSolidGradientRect<FillLinear>(surface, bbox, fill);
    }
    return false;
}


static bool _rasterRadialGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterGradientMattedRect<FillRadial>(surface, bbox, fill);
        else return _rasterGradientMaskedRect<FillRadial>(surface, bbox, fill);
    } else if (_blending(surface)) {
        return _rasterBlendingGradientRect<FillRadial>(surface, bbox, fill);
    } else {
        if (fill->translucent) return _rasterTranslucentGradientRect<FillRadial>(surface, bbox, fill);
        else _rasterSolidGradientRect<FillRadial>(surface, bbox, fill);
    }
    return false;
}


/************************************************************************/
/* Rle Gradient                                                         */
/************************************************************************/

template<typename fillMethod>
static bool _rasterCompositeGradientMaskedRle(SwSurface* surface, const SwRle* rle, const SwFill* fill, SwMask maskOp)
{
    auto span = rle->data();
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8;

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto cmp = &cbuffer[span->y * cstride + span->x];
        fillMethod()(fill, cmp, span->y, span->x, span->len, maskOp, span->coverage);
    }
    return _compositeMaskImage(surface, surface->compositor->image, surface->compositor->bbox);
}


template<typename fillMethod>
static bool _rasterDirectGradientMaskedRle(SwSurface* surface, const SwRle* rle, const SwFill* fill, SwMask maskOp)
{
    auto span = rle->data();
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8;
    auto dbuffer = surface->buf8;

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto cmp = &cbuffer[span->y * cstride + span->x];
        auto dst = &dbuffer[span->y * surface->stride + span->x];
        fillMethod()(fill, dst, span->y, span->x, span->len, cmp, maskOp, span->coverage);
    }
    return true;
}


template<typename fillMethod>
static bool _rasterGradientMaskedRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto method = surface->compositor->method;

    TVGLOG("SW_ENGINE", "Masked(%d) Rle Linear Gradient", (int)method);

    auto maskOp = _getMaskOp(method);

    if (_direct(method)) return _rasterDirectGradientMaskedRle<fillMethod>(surface, rle, fill, maskOp);
    else return _rasterCompositeGradientMaskedRle<fillMethod>(surface, rle, fill, maskOp);
    return false;
}


template<typename fillMethod>
static bool _rasterGradientMattedRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    TVGLOG("SW_ENGINE", "Matted(%d) Rle Linear Gradient", (int)surface->compositor->method);

    auto span = rle->data();
    auto csize = surface->compositor->image.channelSize;
    auto cbuffer = surface->compositor->image.buf8;
    auto alpha = surface->alpha(surface->compositor->method);

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        auto cmp = &cbuffer[(span->y * surface->compositor->image.stride + span->x) * csize];
        fillMethod()(fill, dst, span->y, span->x, span->len, cmp, alpha, csize, span->coverage);
    }
    return true;
}


template<typename fillMethod>
static bool _rasterBlendingGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto span = rle->data();

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        fillMethod()(fill, dst, span->y, span->x, span->len, opBlendPreNormal, surface->blender, span->coverage);
    }
    return true;
}


template<typename fillMethod>
static bool _rasterTranslucentGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto span = rle->data();

    //32 bits
    if (surface->channelSize == sizeof(uint32_t)) {
        for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
            auto dst = &surface->buf32[span->y * surface->stride + span->x];
            if (span->coverage == 255) fillMethod()(fill, dst, span->y, span->x, span->len, opBlendPreNormal, 255);
            else fillMethod()(fill, dst, span->y, span->x, span->len, opBlendNormal, span->coverage);
        }
    //8 bits
    } else if (surface->channelSize == sizeof(uint8_t)) {
        for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
            auto dst = &surface->buf8[span->y * surface->stride + span->x];
            fillMethod()(fill, dst, span->y, span->x, span->len, _opMaskAdd, span->coverage);
        }
    }
    return true;
}


template<typename fillMethod>
static bool _rasterSolidGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto span = rle->data();

    //32 bits
    if (surface->channelSize == sizeof(uint32_t)) {
        for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
            auto dst = &surface->buf32[span->y * surface->stride + span->x];
            if (span->coverage == 255) fillMethod()(fill, dst, span->y, span->x, span->len, opBlendSrcOver, 255);
            else fillMethod()(fill, dst, span->y, span->x, span->len, opBlendInterp, span->coverage);
        }
    //8 bits
    } else if (surface->channelSize == sizeof(uint8_t)) {
        for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
            auto dst = &surface->buf8[span->y * surface->stride + span->x];
            if (span->coverage == 255) fillMethod()(fill, dst, span->y, span->x, span->len, _opMaskNone, 255);
            else fillMethod()(fill, dst, span->y, span->x, span->len, _opMaskAdd, span->coverage);
        }
    }

    return true;
}


static bool _rasterLinearGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterGradientMattedRle<FillLinear>(surface, rle, fill);
        else return _rasterGradientMaskedRle<FillLinear>(surface, rle, fill);
    } else if (_blending(surface)) {
        return _rasterBlendingGradientRle<FillLinear>(surface, rle, fill);
    } else {
        if (fill->translucent) return _rasterTranslucentGradientRle<FillLinear>(surface, rle, fill);
        else return _rasterSolidGradientRle<FillLinear>(surface, rle, fill);
    }
    return false;
}


static bool _rasterRadialGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterGradientMattedRle<FillRadial>(surface, rle, fill);
        else return _rasterGradientMaskedRle<FillRadial>(surface, rle, fill);
    } else if (_blending(surface)) {
        return _rasterBlendingGradientRle<FillRadial>(surface, rle, fill);
    } else {
        if (fill->translucent) return _rasterTranslucentGradientRle<FillRadial>(surface, rle, fill);
        else return _rasterSolidGradientRle<FillRadial>(surface, rle, fill);
    }
    return false;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

void rasterTranslucentPixel32(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity)
{
    //TODO: Support SIMD accelerations
    cRasterTranslucentPixels(dst, src, len, opacity);
}


void rasterPixel32(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity)
{
    //TODO: Support SIMD accelerations
    cRasterPixels(dst, src, len, opacity);
}


void rasterGrayscale8(uint8_t *dst, uint8_t val, uint32_t offset, int32_t len)
{
#if defined(THORVG_AVX_VECTOR_SUPPORT)
    avxRasterGrayscale8(dst, val, offset, len);
#elif defined(THORVG_NEON_VECTOR_SUPPORT)
    neonRasterGrayscale8(dst, val, offset, len);
#else
    cRasterPixels(dst, val, offset, len);
#endif
}


void rasterPixel32(uint32_t *dst, uint32_t val, uint32_t offset, int32_t len)
{
#if defined(THORVG_AVX_VECTOR_SUPPORT)
    avxRasterPixel32(dst, val, offset, len);
#elif defined(THORVG_NEON_VECTOR_SUPPORT)
    neonRasterPixel32(dst, val, offset, len);
#else
    cRasterPixels(dst, val, offset, len);
#endif
}


bool rasterCompositor(SwSurface* surface)
{
    //See MaskMethod, Alpha:1, InvAlpha:2, Luma:3, InvLuma:4
    surface->alphas[0] = _alpha;
    surface->alphas[1] = _ialpha;

    if (surface->cs == ColorSpace::ABGR8888 || surface->cs == ColorSpace::ABGR8888S) {
        surface->join = _abgrJoin;
        surface->alphas[2] = _abgrLuma;
        surface->alphas[3] = _abgrInvLuma;
    } else if (surface->cs == ColorSpace::ARGB8888 || surface->cs == ColorSpace::ARGB8888S) {
        surface->join = _argbJoin;
        surface->alphas[2] = _argbLuma;
        surface->alphas[3] = _argbInvLuma;
    } else {
        TVGERR("SW_ENGINE", "Unsupported Colorspace(%d) is expected!", (int)surface->cs);
        return false;
    }
    return true;
}


bool rasterClear(SwSurface* surface, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    if (!surface || !surface->buf32 || surface->stride == 0 || surface->w == 0 || surface->h == 0) return false;

    //32 bits
    if (surface->channelSize == sizeof(uint32_t)) {
        uint32_t val = 0;
        //full clear
        if (w == surface->stride) {
            rasterPixel32(surface->buf32, val, surface->stride * y, w * h);
        //partial clear
        } else {
            for (uint32_t i = 0; i < h; i++) {
                rasterPixel32(surface->buf32, val, (surface->stride * y + x) + (surface->stride * i), w);
            }
        }
    //8 bits
    } else if (surface->channelSize == sizeof(uint8_t)) {
        //full clear
        if (w == surface->stride) {
            rasterGrayscale8(surface->buf8, 0x00, surface->stride * y, w * h);
        //partial clear
        } else {
            for (uint32_t i = 0; i < h; i++) {
                rasterGrayscale8(surface->buf8, 0x00, (surface->stride * y + x) + (surface->stride * i), w);
            }
        }
    }
    return true;
}


uint32_t rasterUnpremultiply(uint32_t data)
{
    auto a = A(data);
    if (a == 255 || a == 0) return data;

    uint8_t r = std::min(C1(data) * 255u / a, 255u);
    uint8_t g = std::min(C2(data) * 255u / a, 255u);
    uint8_t b = std::min(C3(data) * 255u / a, 255u);

    return JOIN(a, r, g, b);
}


void rasterUnpremultiply(RenderSurface* surface)
{
    if (surface->channelSize != sizeof(uint32_t)) return;

    TVGLOG("SW_ENGINE", "Unpremultiply [Size: %d x %d]", surface->w, surface->h);

    //OPTIMIZE_ME: +SIMD
    for (uint32_t y = 0; y < surface->h; y++) {
        auto buffer = surface->buf32 + surface->stride * y;
        for (uint32_t x = 0; x < surface->w; ++x) {
            buffer[x] = rasterUnpremultiply(buffer[x]);
        }
    }
    surface->premultiplied = false;
}


void rasterPremultiply(RenderSurface* surface)
{
    ScopedLock lock(surface->key);
    if (surface->premultiplied || (surface->channelSize != sizeof(uint32_t))) return;
    surface->premultiplied = true;

    TVGLOG("SW_ENGINE", "Premultiply [Size: %d x %d]", surface->w, surface->h);

    //OPTIMIZE_ME: +SIMD
    auto buffer = surface->buf32;
    for (uint32_t y = 0; y < surface->h; ++y, buffer += surface->stride) {
        auto dst = buffer;
        for (uint32_t x = 0; x < surface->w; ++x, ++dst) {
            auto c = *dst;
            if (A(c) == 255) continue;
            *dst = PREMULTIPLY(c, A(c));
        }
    }
}


bool rasterScaledImage(SwSurface* surface, const SwImage& image, const Matrix& transform, const RenderRegion& bbox, uint8_t opacity)
{
    Matrix itransform;

    if (!inverse(&transform, &itransform)) return true;

    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterScaledMattedImage(surface, image, &itransform, bbox, opacity);
        else return _rasterScaledMaskedImage(surface, image, &itransform, bbox, opacity);
    } else if (_blending(surface)) {
        return _rasterScaledBlendingImage(surface, image, &itransform, bbox, opacity);
    } else {
        return _rasterScaledImage(surface, image, &itransform, bbox, opacity);
    }
    return false;
}


bool rasterDirectImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    //calculate an actual drawing image size
    auto w = std::min(bbox.max.x - bbox.min.x, int32_t(image.w) - (bbox.min.x + image.ox));
    auto h = std::min(bbox.max.y - bbox.min.y, int32_t(image.h) - (bbox.min.y + image.oy));

    if (_compositing(surface)) {
        if (_matting(surface)) {
            if (_blending(surface)) return _rasterDirectMattedBlendingImage(surface, image, bbox, w, h, opacity);
            else return _rasterDirectMattedImage(surface, image, bbox, w, h, opacity);
        } else return _rasterDirectMaskedImage(surface, image, bbox, w, h, opacity);
    } else if (_blending(surface)) {
        return _rasterDirectBlendingImage(surface, image, bbox, w, h, opacity);
    } else {
        return _rasterDirectImage(surface, image, bbox, w, h, opacity);
    }
    return false;
}


bool rasterScaledRleImage(SwSurface* surface, const SwImage& image, const Matrix& transform, const RenderRegion& bbox, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported scaled rle image!");
        return false;
    }

    Matrix itransform;

    if (!inverse(&transform, &itransform)) return true;

    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterScaledMattedRleImage(surface, image, &itransform, bbox, opacity);
        else return _rasterScaledMaskedRleImage(surface, image, &itransform, bbox, opacity);
    } else if (_blending(surface)) {
        return _rasterScaledBlendingRleImage(surface, image, &itransform, bbox, opacity);
    } else {
        return _rasterScaledRleImage(surface, image, &itransform, bbox, opacity);
    }
    return false;
}


bool rasterDirectRleImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale rle image!");
        return false;
    }

    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterDirectMattedRleImage(surface, image, bbox, opacity);
        else return _rasterDirectMaskedRleImage(surface, image, bbox, opacity);
    } else if (_blending(surface)) {
        return _rasterDirectBlendingRleImage(surface, image, bbox, opacity);
    } else {
        return _rasterDirectRleImage(surface, image, bbox, opacity);
    }
    return false;
}


bool rasterGradientShape(SwSurface* surface, SwShape* shape, const RenderRegion& bbox, const Fill* fdata, uint8_t opacity)
{
    if (!shape->fill) return false;

    if (auto color = fillFetchSolid(shape->fill, fdata)) {
        auto a = MULTIPLY(color->a, opacity);
        RenderColor c = {color->r, color->g, color->b, a};
        return a > 0 ? rasterShape(surface, shape, bbox, c) : true;
    }

    auto type = fdata->type();
    if (shape->fastTrack) {
        if (type == Type::LinearGradient) return _rasterLinearGradientRect(surface, bbox, shape->fill);
        else if (type == Type::RadialGradient)return _rasterRadialGradientRect(surface, bbox, shape->fill);
    } else if (shape->rle && shape->rle->valid()) {
        if (type == Type::LinearGradient) return _rasterLinearGradientRle(surface, shape->rle, shape->fill);
        else if (type == Type::RadialGradient) return _rasterRadialGradientRle(surface, shape->rle, shape->fill);
    } return false;
}


bool rasterGradientStroke(SwSurface* surface, SwShape* shape, const RenderRegion& bbox, const Fill* fdata, uint8_t opacity)
{
    if (!shape->stroke || !shape->stroke->fill || !shape->strokeRle || shape->strokeRle->invalid()) return false;

    if (auto color = fillFetchSolid(shape->stroke->fill, fdata)) {
        RenderColor c = {color->r, color->g, color->b, color->a};
        c.a = MULTIPLY(c.a, opacity);
        return c.a > 0 ? rasterStroke(surface, shape, bbox, c) : true;
    }

    auto type = fdata->type();
    if (type == Type::LinearGradient) return _rasterLinearGradientRle(surface, shape->strokeRle, shape->stroke->fill);
    else if (type == Type::RadialGradient) return _rasterRadialGradientRle(surface, shape->strokeRle, shape->stroke->fill);
    return false;
}


bool rasterShape(SwSurface* surface, SwShape* shape, const RenderRegion& bbox, RenderColor& c)
{
    if (c.a < 255) {
        c.r = MULTIPLY(c.r, c.a);
        c.g = MULTIPLY(c.g, c.a);
        c.b = MULTIPLY(c.b, c.a);
    }
    if (shape->fastTrack) return _rasterRect(surface, bbox, c);
    else return _rasterRle(surface, shape->rle, bbox, c);
}


bool rasterStroke(SwSurface* surface, SwShape* shape, const RenderRegion& bbox, RenderColor& c)
{
    if (c.a < 255) {
        c.r = MULTIPLY(c.r, c.a);
        c.g = MULTIPLY(c.g, c.a);
        c.b = MULTIPLY(c.b, c.a);
    }

    return _rasterRle(surface, shape->strokeRle, bbox, c);
}


bool rasterConvertCS(RenderSurface* surface, ColorSpace to)
{
    ScopedLock lock(surface->key);
    if (surface->cs == to) return true;

    //TODO: Support SIMD accelerations
    auto from = surface->cs;

    if (((from == ColorSpace::ABGR8888) || (from == ColorSpace::ABGR8888S)) && ((to == ColorSpace::ARGB8888) || (to == ColorSpace::ARGB8888S))) {
        surface->cs = to;
        return cRasterABGRtoARGB(surface);
    }
    if (((from == ColorSpace::ARGB8888) || (from == ColorSpace::ARGB8888S)) && ((to == ColorSpace::ABGR8888) || (to == ColorSpace::ABGR8888S))) {
        surface->cs = to;
        return cRasterARGBtoABGR(surface);
    }
    return false;
}


//TODO: SIMD OPTIMIZATION?
void rasterXYFlip(uint32_t* src, uint32_t* dst, int32_t stride, int32_t w, int32_t h, const RenderRegion& bbox, bool flipped)
{
    constexpr int32_t BLOCK = 8;  //experimental decision

    if (flipped) {
        src += ((bbox.min.x * stride) + bbox.min.y);
        dst += ((bbox.min.y * stride) + bbox.min.x);
    } else {
        src += ((bbox.min.y * stride) + bbox.min.x);
        dst += ((bbox.min.x * stride) + bbox.min.y);
    }

    #pragma omp parallel for
    for (int32_t x = 0; x < w; x += BLOCK) {
        auto bx = std::min(w, x + BLOCK) - x;
        auto in = &src[x];
        auto out = &dst[x * stride];
        for (int32_t y = 0; y < h; y += BLOCK) {
            auto p = &in[y * stride];
            auto q = &out[y];
            auto by = std::min(h, y + BLOCK) - y;
            for (int32_t xx = 0; xx < bx; ++xx) {
                for (int32_t yy = 0; yy < by; ++yy) {
                    *q = *p;
                    p += stride;
                    ++q;
                }
                p += 1 - by * stride;
                q += stride - by;
            }
        }
    }
}


//TODO: can be moved in tvgColor
void rasterRGB2HSL(uint8_t r, uint8_t g, uint8_t b, float* h, float* s, float* l)
{
    auto rf = r / 255.0f;
    auto gf = g / 255.0f;
    auto bf = b / 255.0f;
    auto maxVal = std::max(std::max(rf, gf), bf);
    auto minVal = std::min(std::min(rf, gf), bf);
    auto delta = maxVal - minVal;

    //lightness
    float t = 0.0f;
    if (l || s) {
        t = (maxVal + minVal) * 0.5f;
        if (l) *l = t;
    }

    if (tvg::zero(delta)) {
        if (h) *h = 0.0f;
        if (s) *s = 0.0f;
    } else {
        //saturation
        if (s) {
            *s = (t < 0.5f) ? (delta / (maxVal + minVal)) : (delta / (2.0f - maxVal - minVal));
        }
        //hue
        if (h) {
            if (maxVal == rf) *h = (gf - bf) / delta + (gf < bf ? 6.0f : 0.0f);
            else if (maxVal == gf) *h = (bf - rf) / delta + 2.0f;
            else *h = (rf - gf) / delta + 4.0f;
            *h *= 60.0f; //directly convert to degrees
        }
    }
}
