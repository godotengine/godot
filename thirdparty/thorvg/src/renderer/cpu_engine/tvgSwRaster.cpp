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

#include "tvgSwRasterCore.h"

/************************************************************************/
/* Helpers                                                              */
/************************************************************************/

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
    return surface->compositor && surface->compositor->method != MaskMethod::None;
}

static inline bool _matting(const SwSurface* surface)
{
    return ((int)surface->compositor->method < (int)MaskMethod::Add);
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
    return (method == MaskMethod::Subtract || method == MaskMethod::Intersect || method == MaskMethod::Darken);
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
    auto dbuffer = &surface->buf8[bbox.yMin * surface->stride + bbox.xMin];
    auto sbuffer = image.buf8 + (bbox.yMin + image.oy) * image.stride + (bbox.xMin + image.ox);

    for (auto y = bbox.yMin; y < bbox.yMax; ++y) {
        auto dst = dbuffer;
        auto src = sbuffer;
        for (auto x = bbox.xMin; x < bbox.xMax; x++, dst++, src++) {
            *dst = *src + MULTIPLY(*dst, ~*src);
        }
        dbuffer += surface->stride;
        sbuffer += image.stride;
    }
    return true;
}

static inline void _rasterPixel8(uint8_t *dst, uint8_t val, uint32_t offset, int32_t len)
{
    memset(dst + offset, val, len);
}

#include "tvgSwRasterTexmap.h"

/************************************************************************/
/* Image Sampling                                                       */
/************************************************************************/

static inline uint32_t _sampleSize(float scale)
{
    auto sampleSize = static_cast<uint32_t>(0.5f / scale);
    if (sampleSize == 0) sampleSize = 1;
    return sampleSize;
}

//Nearest Interpolation
static uint32_t _interpNoScaler(const uint32_t *img, uint32_t stride, TVG_UNUSED uint32_t w, TVG_UNUSED uint32_t h, float sx, float sy, TVG_UNUSED int32_t miny, TVG_UNUSED int32_t maxy, TVG_UNUSED int32_t n)
{
    return img[uint32_t(sx) + uint32_t(sy) * stride];
}

//Bilinear Interpolation
//OPTIMIZE_ME: Skip the function pointer access
static uint32_t _interpUpScaler(const uint32_t *img, uint32_t stride, uint32_t w, uint32_t h, float sx, float sy, TVG_UNUSED int32_t miny, TVG_UNUSED int32_t maxy, TVG_UNUSED int32_t n)
{
    auto rx = (size_t)(sx);
    auto ry = (size_t)(sy);
    auto rx2 = rx + 1;
    if (rx2 >= w) rx2 = w - 1;
    auto ry2 = ry + 1;
    if (ry2 >= h) ry2 = h - 1;

    auto dx = (sx > 0.0f) ? static_cast<uint8_t>((sx - rx) * 255.0f) : 0;
    auto dy = (sy > 0.0f) ? static_cast<uint8_t>((sy - ry) * 255.0f) : 0;

    auto c1 = img[rx + ry * stride];
    auto c2 = img[rx2 + ry * stride];
    auto c3 = img[rx + ry2 * stride];
    auto c4 = img[rx2 + ry2 * stride];

    return INTERPOLATE(INTERPOLATE(c4, c3, dx), INTERPOLATE(c2, c1, dx), dy);
}

using ImageScaleFilter = uint32_t (*)(const uint32_t* img, uint32_t stride, uint32_t w, uint32_t h, float sx, float sy, int32_t miny, int32_t maxy, int32_t n);

static ImageScaleFilter _scaleMethod(const SwImage& image)
{
    if (image.filter == FilterMethod::Bilinear) return image.scale < 0.5f ? rasterDownScaler : _interpUpScaler;
    return _interpNoScaler;
}

/************************************************************************/
/* Rect                                                                 */
/************************************************************************/

static bool _rasterCompositeMaskedRect(SwSurface* surface, const RenderRegion& bbox, SwMask maskOp, uint8_t a)
{
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.yMin * cstride + bbox.xMin);  // compositor buffer
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
    auto cbuffer = surface->compositor->image.buf8 + (bbox.yMin * surface->compositor->image.stride + bbox.xMin);  // compositor buffer
    auto dbuffer = surface->buf8 + (bbox.yMin * surface->stride + bbox.xMin);                                      // destination buffer

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
    if (surface->channelSize != sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "channel size is not 8bits?");
        return false;
    }

    auto maskOp = _getMaskOp(surface->compositor->method);
    if (_direct(surface->compositor->method)) return _rasterDirectMaskedRect(surface, bbox, maskOp, c.a);
    else return _rasterCompositeMaskedRect(surface, bbox, maskOp, c.a);
    return false;
}

static bool _rasterMattedRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    auto csize = surface->compositor->image.channelSize;
    auto cbuffer = surface->compositor->image.buf8 + ((bbox.yMin * surface->compositor->image.stride + bbox.xMin) * csize);  // compositor buffer
    auto alpha = surface->alpha(surface->compositor->method);

    //32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, c.a);
        auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;
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
        auto buffer = surface->buf8 + (bbox.yMin * surface->stride) + bbox.xMin;
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
    if (surface->channelSize != sizeof(uint32_t)) {
        TVGERR("SW_ENGINE", "channel size is not 32bits?");
        return false;
    }

    auto color = surface->join(c.r, c.g, c.b, c.a);
    auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        auto dst = &buffer[y * surface->stride];
        for (uint32_t x = 0; x < bbox.w(); ++x, ++dst) {
            *dst = surface->blender(surface, color, *dst);
        }
    }
    return true;
}

static bool _rasterSolidRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    //32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, 255);
        auto buffer = surface->buf32 + (bbox.yMin * surface->stride);
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            rasterSolidPixel32(buffer + y * surface->stride, color, bbox.xMin, bbox.w());
        }
        return true;
    }
    //8bits grayscale
    if (surface->channelSize == sizeof(uint8_t)) {
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            _rasterPixel8(surface->buf8, 255, (y + bbox.yMin) * surface->stride + bbox.xMin, bbox.w());
        }
        return true;
    }
    return false;
}

static bool _rasterRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterMattedRect(surface, bbox, c);
        return _rasterMaskedRect(surface, bbox, c);
    }
    if (_blending(surface)) return _rasterBlendingRect(surface, bbox, c);
    if (c.a == 255) return _rasterSolidRect(surface, bbox, c);
    return rasterTranslucentRect(surface, bbox, c);
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
    //8bit masking channels composition
    if (surface->channelSize != sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "channel size is not 8bits?");
        return false;
    }

    auto maskOp = _getMaskOp(surface->compositor->method);
    if (_direct(surface->compositor->method)) return _rasterDirectMaskedRle(surface, rle, bbox, maskOp, c.a);
    else return _rasterCompositeMaskedRle(surface, rle, bbox, maskOp, c.a);
    return false;
}

static bool _rasterMattedRle(SwSurface* surface, SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
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
    if (surface->channelSize != sizeof(uint32_t)) {
        TVGERR("SW_ENGINE", "channel size is not 32bits?");
        return false;
    }

    auto color = surface->join(c.r, c.g, c.b, c.a);
    const SwSpan* end;
    int32_t x, len;

    for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
        if (!span->fetch(bbox, x, len)) continue;
        auto dst = &surface->buf32[span->y * surface->stride + x];
        if (span->coverage == 255) {
            for (auto x = 0; x < len; ++x, ++dst) {
                *dst = surface->blender(surface, color, *dst);
            }
        } else {
            for (auto x = 0; x < len; ++x, ++dst) {
                *dst = INTERPOLATE(surface->blender(surface, color, *dst), *dst, span->coverage);
            }
        }
    }
    return true;
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
            if (span->coverage == 255) rasterSolidPixel32(surface->buf32 + span->y * surface->stride, color, x, len);
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
            if (span->coverage == 255) _rasterPixel8(surface->buf8, span->coverage, span->y * surface->stride + x, len);
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
        return _rasterMaskedRle(surface, rle, bbox, c);
    }
    if (_blending(surface)) return _rasterBlendingRle(surface, rle, bbox, c);
    if (c.a == 255) return _rasterSolidRle(surface, rle, bbox, c);
    return rasterTranslucentRle(surface, rle, bbox, c);
}

/************************************************************************/
/* Scaled Image                                                         */
/************************************************************************/

#define SCALED_IMAGE_RANGE_Y(y) \
    auto sy = (y) * itransform->e22 + itransform->e23 - 0.49f; \
    if (sy <= -0.5f || (uint32_t)(sy + 0.5f) >= image.h) continue; \
    if (scaleMethod == rasterDownScaler) { \
        auto my = (int32_t)nearbyint(sy); \
        miny = my - (int32_t)sampleSize; \
        if (miny < 0) miny = 0; \
        maxy = my + (int32_t)sampleSize; \
        if (maxy >= (int32_t)image.h) maxy = (int32_t)image.h; \
    }

#define SCALED_IMAGE_RANGE_X \
    auto sx = (x) * itransform->e11 + itransform->e13 - 0.49f; \
    if (sx <= -0.5f || (uint32_t)(sx + 0.5f) >= image.w) continue; \

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

    auto csize = surface->compositor->image.channelSize;
    auto alpha = surface->alpha(surface->compositor->method);
    auto scaleMethod = _scaleMethod(image);
    auto sampleSize = _sampleSize(image.scale);

    #pragma omp parallel for
    for (auto y = bbox.yMin; y < bbox.yMax; ++y) {
        int32_t miny = 0, maxy = 0;
        SCALED_IMAGE_RANGE_Y(y)
        auto dst = surface->buf32 + y * surface->stride + bbox.xMin;
        auto cmp = surface->compositor->image.buf8 + (y * surface->compositor->image.stride + bbox.xMin) * csize;
        for (auto x = bbox.xMin; x < bbox.xMax; ++x, ++dst, cmp += csize) {
            SCALED_IMAGE_RANGE_X
            auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
            auto tmp = ALPHA_BLEND(src, opacity == 255 ? alpha(cmp) : MULTIPLY(opacity, alpha(cmp)));
            *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
        }
    }
    return true;
}

static bool _rasterScaledBlendingImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale scaled blending image!");
        return false;
    }

    auto scaleMethod = _scaleMethod(image);
    auto sampleSize = _sampleSize(image.scale);

    #pragma omp parallel for
    for (auto y = bbox.yMin; y < bbox.yMax; ++y) {
        int32_t miny = 0, maxy = 0;
        SCALED_IMAGE_RANGE_Y(y)
        auto dst = surface->buf32 + y * surface->stride + bbox.xMin;
        for (auto x = bbox.xMin; x < bbox.xMax; ++x, ++dst) {
            SCALED_IMAGE_RANGE_X
            auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
            *dst = INTERPOLATE(surface->blender(surface, UNPREMULTIPLY(src), *dst), *dst, MULTIPLY(opacity, A(src)));
        }
    }
    return true;
}

static bool _rasterScaledImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    auto scaleMethod = _scaleMethod(image);
    auto sampleSize = _sampleSize(image.scale);

    // 32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        #pragma omp parallel for
        for (auto y = bbox.yMin; y < bbox.yMax; ++y) {
            int32_t miny = 0, maxy = 0;
            SCALED_IMAGE_RANGE_Y(y)
            auto dst = surface->buf32 + y * surface->stride + bbox.xMin;
            if (image.alphaIgnored) {
                if (opacity == 255) {
                    for (auto x = bbox.xMin; x < bbox.xMax; ++x, ++dst) {
                        SCALED_IMAGE_RANGE_X
                        *dst = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                    }
                } else {
                    for (auto x = bbox.xMin; x < bbox.xMax; ++x, ++dst) {
                        SCALED_IMAGE_RANGE_X
                        auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                        *dst = INTERPOLATE(src, *dst, opacity);
                    }
                }
            } else {
                if (opacity == 255) {
                    for (auto x = bbox.xMin; x < bbox.xMax; ++x, ++dst) {
                        SCALED_IMAGE_RANGE_X
                        auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                        *dst = src + ALPHA_BLEND(*dst, IA(src));
                    }
                } else {
                    for (auto x = bbox.xMin; x < bbox.xMax; ++x, ++dst) {
                        SCALED_IMAGE_RANGE_X
                        auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                        src = ALPHA_BLEND(src, opacity);
                        *dst = src + ALPHA_BLEND(*dst, IA(src));
                    }
                }
            }
        }
    } else if (surface->channelSize == sizeof(uint8_t)) {
        #pragma omp parallel for
        for (auto y = bbox.yMin; y < bbox.yMax; ++y) {
            int32_t miny = 0, maxy = 0;
            SCALED_IMAGE_RANGE_Y(y)
            auto dst = surface->buf8 + y * surface->stride + bbox.xMin;
            for (auto x = bbox.xMin; x < bbox.xMax; ++x, ++dst) {
                SCALED_IMAGE_RANGE_X
                auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                *dst = MULTIPLY(A(src), opacity);
            }
        }
    }
    return true;
}

static bool _rasterScaledMaskedRleImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    TVGERR("SW_ENGINE", "Not Supported Scaled Masked(%d) Rle Image", (int)surface->compositor->method);
    return false;
}

static bool _rasterScaledMattedRleImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale scaled matted rle image!");
        return false;
    }

    auto csize = surface->compositor->image.channelSize;
    auto alpha = surface->alpha(surface->compositor->method);
    auto scaleMethod = _scaleMethod(image);
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
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale scaled blending rle image!");
        return false;
    }

    auto scaleMethod = _scaleMethod(image);
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
                *dst = INTERPOLATE(surface->blender(surface, UNPREMULTIPLY(src), *dst), *dst, A(src));
            }
        } else {
            for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                SCALED_IMAGE_RANGE_X
                auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                *dst = INTERPOLATE(surface->blender(surface, UNPREMULTIPLY(src), *dst), *dst, MULTIPLY(alpha, A(src)));
            }
        }
    }
    return true;
}

static bool _rasterScaledRleImage(SwSurface* surface, const SwImage& image, const Matrix* itransform, const RenderRegion& bbox, uint8_t opacity)
{
    auto scaleMethod = _scaleMethod(image);
    auto sampleSize = _sampleSize(image.scale);
    int32_t miny = 0, maxy = 0;

    if (surface->channelSize == sizeof(uint32_t)) {
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
    } else if (surface->channelSize == sizeof(uint8_t)) {
        ARRAY_FOREACH(span, image.rle->spans) {
            SCALED_IMAGE_RANGE_Y(span->y)
            auto dst = &surface->buf8[span->y * surface->stride + span->x];
            auto alpha = MULTIPLY(span->coverage, opacity);
            for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                SCALED_IMAGE_RANGE_X
                auto src = scaleMethod(image.buf32, image.stride, image.w, image.h, sx, sy, miny, maxy, sampleSize);
                *dst = MULTIPLY(A(src), alpha);
            }
        }
    }
    return true;
}

template<bool Rle>
static bool _rasterScaledImageDispatch(SwSurface* surface, const SwImage& image, const Matrix& transform, const RenderRegion& bbox, uint8_t opacity)
{
    Matrix itransform;
    if (!inverse(&transform, &itransform)) return true;

    if (_compositing(surface)) {
        if (_matting(surface)) return (Rle ? _rasterScaledMattedRleImage : _rasterScaledMattedImage)(surface, image, &itransform, bbox, opacity);
        return (Rle ? _rasterScaledMaskedRleImage : _rasterScaledMaskedImage)(surface, image, &itransform, bbox, opacity);
    }
    if (_blending(surface)) return (Rle ? _rasterScaledBlendingRleImage : _rasterScaledBlendingImage)(surface, image, &itransform, bbox, opacity);
    return (Rle ? _rasterScaledRleImage : _rasterScaledImage)(surface, image, &itransform, bbox, opacity);
}

/************************************************************************/
/* RLE Direct Image                                                     */
/************************************************************************/

static bool _rasterDirectMattedRleImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
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
                *dst = surface->blender(surface, UNPREMULTIPLY(*src), *dst);
            }
        } else {
            for (auto x = 0; x < len; ++x, ++dst, ++src) {
                *dst = INTERPOLATE(surface->blender(surface, UNPREMULTIPLY(*src), *dst), *dst, MULTIPLY(alpha, A(*src)));
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
        rasterTranslucentPixels(dst, img, len, alpha);
    }
    return true;
}


static bool _rasterDirectMaskedRleImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    TVGERR("SW_ENGINE", "Not Supported Direct Masked(%d) Rle Image", (int)surface->compositor->method);
    return false;
}

/************************************************************************/
/* Direct Image                                                         */
/************************************************************************/

static bool _rasterDirectMaskedImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    TVGERR("SW_ENGINE", "Not Supported: Direct Masked Image");
    return false;
}

static bool _rasterDirectMattedGrayscale(SwSurface* surface, const SwImage& image, SwCompositor* compositor, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    auto csize = compositor->image.channelSize;
    auto alpha = surface->alpha(compositor->method);
    auto sbuffer = image.buf32 + (bbox.yMin + image.oy) * image.stride + (bbox.xMin + image.ox);
    auto cbuffer = compositor->image.buf8 + (bbox.yMin * compositor->image.stride + bbox.xMin) * csize;  // compositor buffer
    auto dbuffer = surface->buf8 + (bbox.yMin * surface->stride) + bbox.xMin;

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
        cbuffer += compositor->image.stride * csize;
    }
    return true;
}

static bool _rasterDirectMattedImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) return _rasterDirectMattedGrayscale(surface, image, surface->compositor, bbox, w, h, opacity);

    auto csize = surface->compositor->image.channelSize;
    auto alpha = surface->alpha(surface->compositor->method);
    auto sbuffer = image.buf32 + (bbox.yMin + image.oy) * image.stride + (bbox.xMin + image.ox);
    auto cbuffer = surface->compositor->image.buf8 + (bbox.yMin * surface->compositor->image.stride + bbox.xMin) * csize;  // compositor buffer
    auto dbuffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;

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
    return true;
}

static bool _rasterDirectImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    auto sbuffer = image.buf32 + (bbox.yMin + image.oy) * image.stride + (bbox.xMin + image.ox);

    //32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto dbuffer = &surface->buf32[bbox.yMin * surface->stride + bbox.xMin];
        for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
            if (image.alphaIgnored) rasterSolidPixels(dbuffer, sbuffer, w, opacity);
            else rasterTranslucentPixels(dbuffer, sbuffer, w, opacity);
        }
    //8bits grayscale
    //32 -> 8 direct converting seems an avoidable stage. maybe draw to a masking image after an intermediate scene. Can get rid of this?
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto dbuffer = &surface->buf8[bbox.yMin * surface->stride + bbox.xMin];
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

static bool _rasterDirectMattedBlendingImage(SwSurface* surface, const SwImage& image, SwCompositor* compositor, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    // blending doesn't work for grayscale
    if (surface->channelSize == sizeof(uint8_t)) return _rasterDirectMattedGrayscale(surface, image, compositor, bbox, w, h, opacity);

    auto csize = compositor->image.channelSize;
    auto alpha = surface->alpha(compositor->method);
    auto sbuffer = image.buf32 + (bbox.yMin + image.oy) * image.stride + (bbox.xMin + image.ox);
    auto cbuffer = compositor->image.buf8 + (bbox.yMin * compositor->image.stride + bbox.xMin) * csize;  // compositor buffer
    auto dbuffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;

    for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
        auto cmp = cbuffer;
        auto src = sbuffer;
        if (opacity == 255) {
            for (auto dst = dbuffer; dst < dbuffer + w; ++dst, ++src, cmp += csize) {
                *dst = INTERPOLATE(surface->blender(surface, *src, *dst), *dst, MULTIPLY(A(*src), alpha(cmp)));
            }
        } else {
            for (auto dst = dbuffer; dst < dbuffer + w; ++dst, ++src, cmp += csize) {
                *dst = INTERPOLATE(surface->blender(surface, *src, *dst), *dst, MULTIPLY(MULTIPLY(A(*src), alpha(cmp)), opacity));
            }
        }
        cbuffer += compositor->image.stride * csize;
    }
    return true;
}

static bool _rasterDirectBlendingImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, int32_t w, int32_t h, uint8_t opacity)
{
    // blending doesn't work for grayscale
    if (surface->channelSize == sizeof(uint8_t)) return _rasterDirectImage(surface, image, bbox, w, h, opacity);

    // fast-track: mix the blending & masking composition
    auto injecting = [](const SwCompositor* cmp) {
        return cmp && cmp->recoverCmp && (cmp->recoverCmp->method != MaskMethod::None) && _matting(cmp->recoverSfc) && !_blending(cmp->recoverSfc);
    };

    if (injecting(surface->compositor)) return _rasterDirectMattedBlendingImage(surface, image, surface->compositor->recoverCmp, bbox, w, h, opacity);

    auto dbuffer = &surface->buf32[bbox.yMin * surface->stride + bbox.xMin];
    auto sbuffer = image.buf32 + (bbox.yMin + image.oy) * image.stride + (bbox.xMin + image.ox);

    for (auto y = 0; y < h; ++y, dbuffer += surface->stride, sbuffer += image.stride) {
        auto src = sbuffer;
        if (opacity == 255) {
            for (auto dst = dbuffer; dst < dbuffer + w; dst++, src++) {
                *dst = INTERPOLATE(surface->blender(surface, UNPREMULTIPLY(*src), *dst), *dst, A(*src));
            }
        } else {
            for (auto dst = dbuffer; dst < dbuffer + w; dst++, src++) {
                *dst = INTERPOLATE(surface->blender(surface, UNPREMULTIPLY(*src), *dst), *dst, MULTIPLY(opacity, A(*src)));
            }
        }
    }

    return true;
}

/************************************************************************/
/* Rect Gradient                                                        */
/************************************************************************/

static bool _rasterCompositeGradientMaskedRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill, SwMask maskOp)
{
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.yMin * cstride + bbox.xMin);

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillRaster(fill, cbuffer, bbox.yMin + y, bbox.xMin, bbox.w(), maskOp, 255);
        cbuffer += cstride;
    }
    return _compositeMaskImage(surface, surface->compositor->image, surface->compositor->bbox);
}

static bool _rasterDirectGradientMaskedRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill, SwMask maskOp)
{
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.yMin * cstride + bbox.xMin);
    auto dbuffer = surface->buf8 + (bbox.yMin * surface->stride + bbox.xMin);

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillRaster(fill, dbuffer, bbox.yMin + y, bbox.xMin, bbox.w(), cbuffer, maskOp, 255);
        cbuffer += cstride;
        dbuffer += surface->stride;
    }
    return true;
}

static bool _rasterGradientMaskedRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    auto method = surface->compositor->method;
    auto maskOp = _getMaskOp(method);

    if (_direct(method)) return _rasterDirectGradientMaskedRect(surface, bbox, fill, maskOp);
    else return _rasterCompositeGradientMaskedRect(surface, bbox, fill, maskOp);

    return false;
}

static bool _rasterGradientMattedRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale gradient!");
        return false;
    }

    auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;
    auto csize = surface->compositor->image.channelSize;
    auto cbuffer = surface->compositor->image.buf8 + (bbox.yMin * surface->compositor->image.stride + bbox.xMin) * csize;
    auto cstride = surface->compositor->image.stride * csize;
    auto alpha = surface->alpha(surface->compositor->method);

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillRaster(fill, buffer, bbox.yMin + y, bbox.xMin, bbox.w(), cbuffer, alpha, csize, 255);
        buffer += surface->stride;
        cbuffer += cstride;
    }
    return true;
}

static bool _rasterGradientRect8(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    auto buffer = surface->buf8 + (bbox.yMin * surface->stride) + bbox.xMin;

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillRaster(fill, buffer, bbox.yMin + y, bbox.xMin, bbox.w(), _opMaskAdd, 255);
        buffer += surface->stride;
    }
    return true;
}

static bool _rasterBlendingGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;

    if (fill->translucent) {
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            fillRaster(surface, fill, buffer + y * surface->stride, bbox.yMin + y, bbox.xMin, bbox.w(), opBlendPreNormal, surface->blender, 255);
        }
    } else {
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            fillRaster(surface, fill, buffer + y * surface->stride, bbox.yMin + y, bbox.xMin, bbox.w(), opBlendSrcOver, surface->blender, 255);
        }
    }
    return true;
}

static bool _rasterTranslucentGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillRaster(fill, buffer, bbox.yMin + y, bbox.xMin, bbox.w(), opBlendPreNormal, 255);
        buffer += surface->stride;
    }
    return true;
}

static bool _rasterSolidGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;

    for (uint32_t y = 0; y < bbox.h(); ++y) {
        fillRaster(fill, buffer, bbox.yMin + y, bbox.xMin, bbox.w(), opBlendSrcOver, 255);
        buffer += surface->stride;
    }
    return true;
}

static bool _rasterGradientRect(SwSurface* surface, const RenderRegion& bbox, const SwFill* fill)
{
    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterGradientMattedRect(surface, bbox, fill);
        return _rasterGradientMaskedRect(surface, bbox, fill);
    }
    //Rasterize 8-bit masks before dispatching to the 32-bit color paths.
    if (surface->channelSize == sizeof(uint8_t)) return _rasterGradientRect8(surface, bbox, fill);
    if (_blending(surface)) return _rasterBlendingGradientRect(surface, bbox, fill);
    if (fill->translucent) return _rasterTranslucentGradientRect(surface, bbox, fill);
    return _rasterSolidGradientRect(surface, bbox, fill);
}

/************************************************************************/
/* Rle Gradient                                                         */
/************************************************************************/

static bool _rasterCompositeGradientMaskedRle(SwSurface* surface, const SwRle* rle, const SwFill* fill, SwMask maskOp)
{
    auto span = rle->data();
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8;

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto cmp = &cbuffer[span->y * cstride + span->x];
        fillRaster(fill, cmp, span->y, span->x, span->len, maskOp, span->coverage);
    }
    return _compositeMaskImage(surface, surface->compositor->image, surface->compositor->bbox);
}

static bool _rasterDirectGradientMaskedRle(SwSurface* surface, const SwRle* rle, const SwFill* fill, SwMask maskOp)
{
    auto span = rle->data();
    auto cstride = surface->compositor->image.stride;
    auto cbuffer = surface->compositor->image.buf8;
    auto dbuffer = surface->buf8;

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto cmp = &cbuffer[span->y * cstride + span->x];
        auto dst = &dbuffer[span->y * surface->stride + span->x];
        fillRaster(fill, dst, span->y, span->x, span->len, cmp, maskOp, span->coverage);
    }
    return true;
}

static bool _rasterGradientMaskedRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto method = surface->compositor->method;
    auto maskOp = _getMaskOp(method);

    if (_direct(method)) return _rasterDirectGradientMaskedRle(surface, rle, fill, maskOp);
    else return _rasterCompositeGradientMaskedRle(surface, rle, fill, maskOp);
    return false;
}

static bool _rasterGradientMattedRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale gradient!");
        return false;
    }

    auto span = rle->data();
    auto csize = surface->compositor->image.channelSize;
    auto cbuffer = surface->compositor->image.buf8;
    auto alpha = surface->alpha(surface->compositor->method);

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        auto cmp = &cbuffer[(span->y * surface->compositor->image.stride + span->x) * csize];
        fillRaster(fill, dst, span->y, span->x, span->len, cmp, alpha, csize, span->coverage);
    }
    return true;
}

static bool _rasterGradientRle8(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto span = rle->data();

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto dst = &surface->buf8[span->y * surface->stride + span->x];
        fillRaster(fill, dst, span->y, span->x, span->len, _opMaskAdd, span->coverage);
    }
    return true;
}

static bool _rasterBlendingGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto span = rle->data();

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        fillRaster(surface, fill, dst, span->y, span->x, span->len, opBlendPreNormal, surface->blender, span->coverage);
    }
    return true;
}

static bool _rasterTranslucentGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto span = rle->data();

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        if (span->coverage == 255) fillRaster(fill, dst, span->y, span->x, span->len, opBlendPreNormal, 255);
        else fillRaster(fill, dst, span->y, span->x, span->len, opBlendNormal, span->coverage);
    }
    return true;
}

static bool _rasterSolidGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    auto span = rle->data();

    for (uint32_t i = 0; i < rle->size(); ++i, ++span) {
        auto dst = &surface->buf32[span->y * surface->stride + span->x];
        if (span->coverage == 255) fillRaster(fill, dst, span->y, span->x, span->len, opBlendSrcOver, 255);
        else fillRaster(fill, dst, span->y, span->x, span->len, opBlendInterp, span->coverage);
    }

    return true;
}

static bool _rasterGradientRle(SwSurface* surface, const SwRle* rle, const SwFill* fill)
{
    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterGradientMattedRle(surface, rle, fill);
        return _rasterGradientMaskedRle(surface, rle, fill);
    }
    //Rasterize 8-bit masks before dispatching to the 32-bit color paths.
    if (surface->channelSize == sizeof(uint8_t)) return _rasterGradientRle8(surface, rle, fill);
    if (_blending(surface)) return _rasterBlendingGradientRle(surface, rle, fill);
    if (fill->translucent) return _rasterTranslucentGradientRle(surface, rle, fill);
    return _rasterSolidGradientRle(surface, rle, fill);
}

/************************************************************************/
/* External Interfaces                                                  */
/************************************************************************/

Result rasterCompositor(SwSurface* surface)
{
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
        return Result::NonSupport;
    }
    return Result::Success;
}

bool rasterClear(SwSurface* surface, uint32_t x, uint32_t y, uint32_t w, uint32_t h)
{
    if (!surface || !surface->buf32 || surface->stride == 0 || surface->w == 0 || surface->h == 0) return false;

    //32 bits
    if (surface->channelSize == sizeof(uint32_t)) {
        uint32_t val = 0;
        //full clear
        if (w == surface->stride) {
            rasterSolidPixel32(surface->buf32, val, surface->stride * y, w * h);
        //partial clear
        } else {
            for (uint32_t i = 0; i < h; i++) {
                rasterSolidPixel32(surface->buf32, val, (surface->stride * y + x) + (surface->stride * i), w);
            }
        }
    //8 bits
    } else if (surface->channelSize == sizeof(uint8_t)) {
        //full clear
        if (w == surface->stride) {
            _rasterPixel8(surface->buf8, 0x00, surface->stride * y, w * h);
        //partial clear
        } else {
            for (uint32_t i = 0; i < h; i++) {
                _rasterPixel8(surface->buf8, 0x00, (surface->stride * y + x) + (surface->stride * i), w);
            }
        }
    }
    return true;
}

void rasterUnpremultiplySurface(RenderSurface* surface)
{
    if (surface->channelSize != sizeof(uint32_t)) return;

    TVGLOG("SW_ENGINE", "Unpremultiply [Size: %d x %d]", surface->w, surface->h);

    #pragma omp parallel for
    for (int32_t y = 0; y < (int32_t)surface->h; y++) {
        rasterUnpremultiply(surface->buf32 + surface->stride * uint32_t(y), surface->w);
    }
    surface->premultiplied = false;
}

void rasterPremultiplySurface(RenderSurface* surface)
{
    ScopedLock lock(surface->key);
    if (surface->premultiplied || (surface->channelSize != sizeof(uint32_t))) return;
    surface->premultiplied = true;

    TVGLOG("SW_ENGINE", "Premultiply [Size: %d x %d]", surface->w, surface->h);

    #pragma omp parallel for
    for (int32_t y = 0; y < (int32_t)surface->h; ++y) {
        rasterPremultiply(surface->buf32 + surface->stride * uint32_t(y), surface->w);
    }
}

bool rasterScaledImage(SwSurface* surface, const SwImage& image, const Matrix& transform, const RenderRegion& bbox, uint8_t opacity)
{
    return _rasterScaledImageDispatch<false>(surface, image, transform, bbox, opacity);
}

bool rasterScaledRleImage(SwSurface* surface, const SwImage& image, const Matrix& transform, const RenderRegion& bbox, uint8_t opacity)
{
    return _rasterScaledImageDispatch<true>(surface, image, transform, bbox, opacity);
}

bool rasterDirectImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    //calculate an actual drawing image size
    auto w = std::min(bbox.xMax - bbox.xMin, int32_t(image.w) - (bbox.xMin + image.ox));
    auto h = std::min(bbox.yMax - bbox.yMin, int32_t(image.h) - (bbox.yMin + image.oy));

    if (_compositing(surface)) {
        if (_matting(surface)) {
            if (_blending(surface)) return _rasterDirectMattedBlendingImage(surface, image, surface->compositor, bbox, w, h, opacity);
            return _rasterDirectMattedImage(surface, image, bbox, w, h, opacity);
        }
        return _rasterDirectMaskedImage(surface, image, bbox, w, h, opacity);
    }
    if (_blending(surface)) return _rasterDirectBlendingImage(surface, image, bbox, w, h, opacity);
    return _rasterDirectImage(surface, image, bbox, w, h, opacity);
}

bool rasterDirectRleImage(SwSurface* surface, const SwImage& image, const RenderRegion& bbox, uint8_t opacity)
{
    if (surface->channelSize == sizeof(uint8_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale rle image!");
        return false;
    }

    if (_compositing(surface)) {
        if (_matting(surface)) return _rasterDirectMattedRleImage(surface, image, bbox, opacity);
        return _rasterDirectMaskedRleImage(surface, image, bbox, opacity);
    }
    if (_blending(surface)) return _rasterDirectBlendingRleImage(surface, image, bbox, opacity);
    return _rasterDirectRleImage(surface, image, bbox, opacity);
}

bool rasterGradientShape(SwSurface* surface, SwShape* shape, const RenderRegion& bbox, const Fill* fdata, uint8_t opacity)
{
    if (!shape->fill) return false;

    if (auto color = fillFetchSolid(shape->fill, fdata)) {
        auto a = MULTIPLY(color->a, opacity);
        RenderColor c = {color->r, color->g, color->b, a};
        return a > 0 ? rasterShape(surface, shape, bbox, c) : true;
    }

    if (shape->fastTrack) return _rasterGradientRect(surface, bbox, shape->fill);
    else if (shape->rle && shape->rle->valid()) return _rasterGradientRle(surface, shape->rle, shape->fill);
    return false;
}

bool rasterGradientStroke(SwSurface* surface, SwShape* shape, const RenderRegion& bbox, const Fill* fdata, uint8_t opacity)
{
    if (!shape->stroke || !shape->stroke->fill || !shape->strokeRle || shape->strokeRle->invalid()) return false;

    if (auto color = fillFetchSolid(shape->stroke->fill, fdata)) {
        RenderColor c = {color->r, color->g, color->b, color->a};
        c.a = MULTIPLY(c.a, opacity);
        return c.a > 0 ? rasterStroke(surface, shape, bbox, c) : true;
    }
    return _rasterGradientRle(surface, shape->strokeRle, shape->stroke->fill);
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

    auto from = surface->cs;

    if (((from == ColorSpace::ABGR8888) || (from == ColorSpace::ABGR8888S)) && ((to == ColorSpace::ARGB8888) || (to == ColorSpace::ARGB8888S))) {
        surface->cs = to;
        return rasterABGRtoARGB(surface);
    }
    if (((from == ColorSpace::ARGB8888) || (from == ColorSpace::ARGB8888S)) && ((to == ColorSpace::ABGR8888) || (to == ColorSpace::ABGR8888S))) {
        surface->cs = to;
        return rasterABGRtoARGB(surface);
    }
    return false;
}
