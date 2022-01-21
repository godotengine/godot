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
#include "tvgRender.h"
#include "tvgSwCommon.h"

/************************************************************************/
/* Internal Class Implementation                                        */
/************************************************************************/
constexpr auto DOWN_SCALE_TOLERANCE = 0.5f;


static inline uint32_t _multiplyAlpha(uint32_t c, uint32_t a)
{
    return ((c * a + 0xff) >> 8);
}


static inline uint32_t _alpha(uint32_t c)
{
    return (c >> 24);
}


static inline uint32_t _ialpha(uint32_t c)
{
    return (~c >> 24);
}


static inline uint32_t _abgrJoin(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    return (a << 24 | b << 16 | g << 8 | r);
}


static inline uint32_t _argbJoin(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    return (a << 24 | r << 16 | g << 8 | b);
}


#include "tvgSwRasterTexmap.h"
#include "tvgSwRasterC.h"
#include "tvgSwRasterAvx.h"
#include "tvgSwRasterNeon.h"


static inline bool _compositing(const SwSurface* surface)
{
    if (!surface->compositor || surface->compositor->method == CompositeMethod::None) return false;
    return true;
}


static inline uint32_t _halfScale(float scale)
{
    auto halfScale = static_cast<uint32_t>(0.5f / scale);
    if (halfScale == 0) halfScale = 1;
    return halfScale;
}

//Bilinear Interpolation
static uint32_t _interpUpScaler(const uint32_t *img, uint32_t w, uint32_t h, float sx, float sy)
{
    auto rx = (uint32_t)(sx);
    auto ry = (uint32_t)(sy);
    auto rx2 = rx + 1;
    if (rx2 >= w) rx2 = w - 1;
    auto ry2 = ry + 1;
    if (ry2 >= h) ry2 = h - 1;

    auto dx = static_cast<uint32_t>((sx - rx) * 255.0f);
    auto dy = static_cast<uint32_t>((sy - ry) * 255.0f);

    auto c1 = img[rx + ry * w];
    auto c2 = img[rx2 + ry * w];
    auto c3 = img[rx2 + ry2 * w];
    auto c4 = img[rx + ry2 * w];

    return INTERPOLATE(dy, INTERPOLATE(dx, c3, c4), INTERPOLATE(dx, c2, c1));
}


//2n x 2n Mean Kernel
static uint32_t _interpDownScaler(const uint32_t *img, uint32_t stride, uint32_t w, uint32_t h, uint32_t rx, uint32_t ry, uint32_t n)
{
    uint32_t c[4] = {0, 0, 0, 0};
    auto n2 = n * n;
    auto src = img + rx - n + (ry - n) * stride;

    for (auto y = ry - n; y < ry + n; ++y) {
        if (y >= h) continue;
        auto p = src;
        for (auto x = rx - n; x < rx + n; ++x, ++p) {
            if (x >= w) continue;
            c[0] += *p >> 24;
            c[1] += (*p >> 16) & 0xff;
            c[2] += (*p >> 8) & 0xff;
            c[3] += *p & 0xff;
        }
        src += stride;
    }
    for (auto i = 0; i < 4; ++i) {
        c[i] = (c[i] >> 2) / n2;
    }
    return (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3];
}


/************************************************************************/
/* Rect                                                                 */
/************************************************************************/

static bool _rasterMaskedRect(SwSurface* surface, const SwBBox& region, uint32_t color, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Masked Rect");

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto w = static_cast<uint32_t>(region.max.x - region.min.x);
    auto h = static_cast<uint32_t>(region.max.y - region.min.y);

    auto cbuffer = surface->compositor->image.data + (region.min.y * surface->compositor->image.stride) + region.min.x;   //compositor buffer

    for (uint32_t y = 0; y < h; ++y) {
        auto dst = &buffer[y * surface->stride];
        auto cmp = &cbuffer[y * surface->stride];
        for (uint32_t x = 0; x < w; ++x, ++dst, ++cmp) {
            auto tmp = ALPHA_BLEND(color, blendMethod(*cmp));
            *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
        }
    }
    return true;
}


static bool _rasterSolidRect(SwSurface* surface, const SwBBox& region, uint32_t color)
{
    auto buffer = surface->buffer + (region.min.y * surface->stride);
    auto w = static_cast<uint32_t>(region.max.x - region.min.x);
    auto h = static_cast<uint32_t>(region.max.y - region.min.y);

    for (uint32_t y = 0; y < h; ++y) {
        rasterRGBA32(buffer + y * surface->stride, color, region.min.x, w);
    }
    return true;
}


static bool _rasterRect(SwSurface* surface, const SwBBox& region, uint32_t color, uint8_t opacity)
{
    if (_compositing(surface)) {
        if (surface->compositor->method == CompositeMethod::AlphaMask) {
            return _rasterMaskedRect(surface, region, color, _alpha);
        } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
            return _rasterMaskedRect(surface, region, color, _ialpha);
        }
    } else {
        if (opacity == 255) {
            return _rasterSolidRect(surface, region, color);
        } else {
#if defined(THORVG_AVX_VECTOR_SUPPORT)
            return avxRasterTranslucentRect(surface, region, color);
#elif defined(THORVG_NEON_VECTOR_SUPPORT)
            return neonRasterTranslucentRect(surface, region, color);
#else
            return cRasterTranslucentRect(surface, region, color);
#endif
        }
    }
    return false;
}


/************************************************************************/
/* Rle                                                                  */
/************************************************************************/

static bool _rasterMaskedRle(SwSurface* surface, SwRleData* rle, uint32_t color, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Masked Rle");

    auto span = rle->spans;
    uint32_t src;
    auto cbuffer = surface->compositor->image.data;

    for (uint32_t i = 0; i < rle->size; ++i, ++span) {
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        auto cmp = &cbuffer[span->y * surface->compositor->image.stride + span->x];
        if (span->coverage == 255) src = color;
        else src = ALPHA_BLEND(color, span->coverage);
        for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp) {
            auto tmp = ALPHA_BLEND(src, blendMethod(*cmp));
            *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
        }
    }
    return true;
}


static bool _rasterSolidRle(SwSurface* surface, const SwRleData* rle, uint32_t color)
{
    auto span = rle->spans;

    for (uint32_t i = 0; i < rle->size; ++i, ++span) {
        if (span->coverage == 255) {
            rasterRGBA32(surface->buffer + span->y * surface->stride, color, span->x, span->len);
        } else {
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            auto src = ALPHA_BLEND(color, span->coverage);
            auto ialpha = 255 - span->coverage;
            for (uint32_t x = 0; x < span->len; ++x, ++dst) {
                *dst = src + ALPHA_BLEND(*dst, ialpha);
            }
        }
    }
    return true;
}


static bool _rasterRle(SwSurface* surface, SwRleData* rle, uint32_t color, uint8_t opacity)
{
    if (!rle) return false;

    if (_compositing(surface)) {
        if (surface->compositor->method == CompositeMethod::AlphaMask) {
            return _rasterMaskedRle(surface, rle, color, _alpha);
        } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
            return _rasterMaskedRle(surface, rle, color, _ialpha);
        }
    } else {
        if (opacity == 255) {
            return _rasterSolidRle(surface, rle, color);
        } else {
#if defined(THORVG_AVX_VECTOR_SUPPORT)
            return avxRasterTranslucentRle(surface, rle, color);
#elif defined(THORVG_NEON_VECTOR_SUPPORT)
            return neonRasterTranslucentRle(surface, rle, color);
#else
            return cRasterTranslucentRle(surface, rle, color);
#endif
        }
    }
    return false;
}


/************************************************************************/
/* RLE Transformed RGBA Image                                           */
/************************************************************************/

static bool _transformedRleRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* transform, uint32_t opacity)
{
    if (_compositing(surface)) {
        if (surface->compositor->method == CompositeMethod::AlphaMask) {
            return _rasterTexmapPolygon(surface, image, transform, nullptr, opacity, _alpha);
        } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
            return _rasterTexmapPolygon(surface, image, transform, nullptr, opacity, _ialpha);
        }
    } else {
        return _rasterTexmapPolygon(surface, image, transform, nullptr, opacity, nullptr);
    }
    return false;
}

/************************************************************************/
/* RLE Scaled RGBA Image                                                */
/************************************************************************/

static bool _rasterScaledMaskedTranslucentRleRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* itransform, const SwBBox& region, uint32_t opacity, uint32_t halfScale, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Scaled Masked Translucent Rle Image");

    auto span = image->rle->spans;

    //Center (Down-Scaled)
    if (image->scale < DOWN_SCALE_TOLERANCE) {
        for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
            auto sy = (uint32_t)(span->y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            auto cmp = &surface->compositor->image.data[span->y * surface->compositor->image.stride + span->x];
            auto alpha = _multiplyAlpha(span->coverage, opacity);
            for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst, ++cmp) {
                auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                if (sx >= image->w) continue;
                auto src = ALPHA_BLEND(_interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale), alpha);
                auto tmp = ALPHA_BLEND(src, blendMethod(*cmp));
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        }
    //Center (Up-Scaled)
    } else {
        for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
            auto sy = span->y * itransform->e22 + itransform->e23;
            if ((uint32_t)sy >= image->h) continue;
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            auto cmp = &surface->compositor->image.data[span->y * surface->compositor->image.stride + span->x];
            auto alpha = _multiplyAlpha(span->coverage, opacity);
            for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst, ++cmp) {
                auto sx = x * itransform->e11 + itransform->e13;
                if ((uint32_t)sx >= image->w) continue;
                auto src = ALPHA_BLEND(_interpUpScaler(image->data, image->w, image->h, sx, sy), alpha);
                auto tmp = ALPHA_BLEND(src, blendMethod(*cmp));
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        }
    }
    return true;
}


static bool _rasterScaledMaskedRleRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* itransform, const SwBBox& region, uint32_t halfScale, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Scaled Masked Rle Image");

    auto span = image->rle->spans;

    //Center (Down-Scaled)
    if (image->scale < DOWN_SCALE_TOLERANCE) {
        for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
            auto sy = (uint32_t)(span->y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            auto cmp = &surface->compositor->image.data[span->y * surface->compositor->image.stride + span->x];
            if (span->coverage == 255) {
                for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst, ++cmp) {
                    auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                    if (sx >= image->w) continue;
                    auto tmp = ALPHA_BLEND(_interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale), blendMethod(*cmp));
                    *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
                }
            } else {
                for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst, ++cmp) {
                    auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                    if (sx >= image->w) continue;
                    auto src = ALPHA_BLEND(_interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale), span->coverage);
                    auto tmp = ALPHA_BLEND(src, blendMethod(*cmp));
                    *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
                }
            }
        }
    //Center (Up-Scaled)
    } else {
        for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
            auto sy = span->y * itransform->e22 + itransform->e23;
            if ((uint32_t)sy >= image->h) continue;
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            auto cmp = &surface->compositor->image.data[span->y * surface->compositor->image.stride + span->x];
            if (span->coverage == 255) {
                for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst, ++cmp) {
                    auto sx = x * itransform->e11 + itransform->e13;
                    if ((uint32_t)sx >= image->w) continue;
                    auto tmp = ALPHA_BLEND(_interpUpScaler(image->data, image->w, image->h, sx, sy), blendMethod(*cmp));
                    *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
                }
            } else {
                for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst, ++cmp) {
                    auto sx = x * itransform->e11 + itransform->e13;
                    if ((uint32_t)sx >= image->w) continue;
                    auto src = ALPHA_BLEND(_interpUpScaler(image->data, image->w, image->h, sx, sy), span->coverage);
                    auto tmp = ALPHA_BLEND(src, blendMethod(*cmp));
                    *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
                }
            }
        }
    }
    return true;
}


static bool _rasterScaledTranslucentRleRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* itransform, const SwBBox& region, uint32_t opacity, uint32_t halfScale)
{
    auto span = image->rle->spans;

    //Center (Down-Scaled)
    if (image->scale < DOWN_SCALE_TOLERANCE) {
        for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
            auto sy = (uint32_t)(span->y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            auto alpha = _multiplyAlpha(span->coverage, opacity);
            for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                if (sx >= image->w) continue;
                auto src = ALPHA_BLEND(_interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale), alpha);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
        }
    //Center (Up-Scaled)
    } else {
        for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
            auto sy = span->y * itransform->e22 + itransform->e23;
            if ((uint32_t)sy >= image->h) continue;
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            auto alpha = _multiplyAlpha(span->coverage, opacity);
            for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                auto sx = x * itransform->e11 + itransform->e13;
                if ((uint32_t)sx >= image->w) continue;
                auto src = ALPHA_BLEND(_interpUpScaler(image->data, image->w, image->h, sx, sy), alpha);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
        }
    }
    return true;
}


static bool _rasterScaledRleRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* itransform, const SwBBox& region, uint32_t opacity, uint32_t halfScale)
{
    auto span = image->rle->spans;

    //Center (Down-Scaled)
    if (image->scale < DOWN_SCALE_TOLERANCE) {
        for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
            auto sy = (uint32_t)(span->y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            if (span->coverage == 255) {
                for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                    auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                    if (sx >= image->w) continue;
                    auto src = _interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale);
                    *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
                }
            } else {
                for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                    auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                    if (sx >= image->w) continue;
                    auto src = ALPHA_BLEND(_interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale), span->coverage);
                    *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
                }
            }
        }
    //Center (Up-Scaled)
    } else {
        for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
            auto sy = span->y * itransform->e22 + itransform->e23;
            if ((uint32_t)sy >= image->h) continue;
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            if (span->coverage == 255) {
                for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                    auto sx = x * itransform->e11 + itransform->e13;
                    if ((uint32_t)sx >= image->w) continue;
                    auto src = _interpUpScaler(image->data, image->w, image->h, sx, sy);
                    *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
                }
            } else {
                for (uint32_t x = static_cast<uint32_t>(span->x); x < static_cast<uint32_t>(span->x) + span->len; ++x, ++dst) {
                    auto sx = x * itransform->e11 + itransform->e13;
                    if ((uint32_t)sx >= image->w) continue;
                    auto src = ALPHA_BLEND(_interpUpScaler(image->data, image->w, image->h, sx, sy), span->coverage);
                    *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
                }
            }
        }
    }
    return true;
}


static bool _scaledRleRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* transform, const SwBBox& region, uint32_t opacity)
{
    Matrix itransform;
    if (transform && !mathInverse(transform, &itransform)) return false;

    auto halfScale = _halfScale(image->scale);

    if (_compositing(surface)) {
        if (opacity == 255) {
            if (surface->compositor->method == CompositeMethod::AlphaMask) {
                return _rasterScaledMaskedRleRGBAImage(surface, image, &itransform, region, halfScale, _alpha);
            } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
                return _rasterScaledMaskedRleRGBAImage(surface, image, &itransform, region, halfScale, _ialpha);
            }
        } else {
            if (surface->compositor->method == CompositeMethod::AlphaMask) {
                return _rasterScaledMaskedTranslucentRleRGBAImage(surface, image, &itransform, region, opacity, halfScale, _alpha);
            } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
                return _rasterScaledMaskedTranslucentRleRGBAImage(surface, image, &itransform, region, opacity, halfScale, _ialpha);
            }
        }
    } else {
        if (opacity == 255) return _rasterScaledRleRGBAImage(surface, image, &itransform, region, opacity, halfScale);
        else return _rasterScaledTranslucentRleRGBAImage(surface, image, &itransform, region, opacity, halfScale);
    }
    return false;
}


/************************************************************************/
/* RLE Direct RGBA Image                                                */
/************************************************************************/

static bool _rasterDirectMaskedTranslucentRleRGBAImage(SwSurface* surface, const SwImage* image, uint32_t opacity, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Direct Masked Rle Image");

    auto span = image->rle->spans;
    auto cbuffer = surface->compositor->image.data;

    for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        auto cmp = &cbuffer[span->y * surface->compositor->image.stride + span->x];
        auto img = image->data + (span->y + image->oy) * image->stride + (span->x + image->ox);
        auto alpha = _multiplyAlpha(span->coverage, opacity);
        if (alpha == 255) {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp, ++img) {
                auto tmp = ALPHA_BLEND(*img, blendMethod(*cmp));
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        } else {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp, ++img) {
                auto tmp = ALPHA_BLEND(*img, _multiplyAlpha(alpha, blendMethod(*cmp)));
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        }
    }
    return true;
}


static bool _rasterDirectMaskedRleRGBAImage(SwSurface* surface, const SwImage* image, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Direct Masked Rle Image");

    auto span = image->rle->spans;
    auto cbuffer = surface->compositor->image.data;

    for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        auto cmp = &cbuffer[span->y * surface->compositor->image.stride + span->x];
        auto img = image->data + (span->y + image->oy) * image->stride + (span->x + image->ox);
        if (span->coverage == 255) {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp, ++img) {
                auto tmp = ALPHA_BLEND(*img, blendMethod(*cmp));
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        } else {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp, ++img) {
                auto tmp = ALPHA_BLEND(*img, _multiplyAlpha(span->coverage, blendMethod(*cmp)));
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        }
    }
    return true;
}


static bool _rasterDirectTranslucentRleRGBAImage(SwSurface* surface, const SwImage* image, uint32_t opacity)
{
    auto span = image->rle->spans;

    for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        auto img = image->data + (span->y + image->oy) * image->stride + (span->x + image->ox);
        auto alpha = _multiplyAlpha(span->coverage, opacity);
        for (uint32_t x = 0; x < span->len; ++x, ++dst, ++img) {
            auto src = ALPHA_BLEND(*img, alpha);
            *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
        }
    }
    return true;
}


static bool _rasterDirectRleRGBAImage(SwSurface* surface, const SwImage* image)
{
    auto span = image->rle->spans;

    for (uint32_t i = 0; i < image->rle->size; ++i, ++span) {
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        auto img = image->data + (span->y + image->oy) * image->stride + (span->x + image->ox);
        if (span->coverage == 255) {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++img) {
                *dst = *img + ALPHA_BLEND(*dst, _ialpha(*img));
            }
        } else {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++img) {
                auto src = ALPHA_BLEND(*img, span->coverage);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
        }
    }
    return true;
}


static bool _directRleRGBAImage(SwSurface* surface, const SwImage* image, uint32_t opacity)
{
    if (_compositing(surface)) {
        if (opacity == 255) {
            if (surface->compositor->method == CompositeMethod::AlphaMask) {
                return _rasterDirectMaskedRleRGBAImage(surface, image, _alpha);
            } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
                return _rasterDirectMaskedRleRGBAImage(surface, image, _ialpha);
            }
        } else {
            if (surface->compositor->method == CompositeMethod::AlphaMask) {
                return _rasterDirectMaskedTranslucentRleRGBAImage(surface, image, opacity, _alpha);
            } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
                return _rasterDirectMaskedTranslucentRleRGBAImage(surface, image, opacity, _ialpha);
            }
        }
    } else {
        if (opacity == 255) return _rasterDirectRleRGBAImage(surface, image);
        else return _rasterDirectTranslucentRleRGBAImage(surface, image, opacity);
    }
    return false;
}


/************************************************************************/
/* Transformed RGBA Image                                               */
/************************************************************************/

static bool _transformedRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* transform, const SwBBox& region, uint32_t opacity)
{
    if (_compositing(surface)) {
        if (surface->compositor->method == CompositeMethod::AlphaMask) {
            return _rasterTexmapPolygon(surface, image, transform, &region, opacity, _alpha);
        } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
            return _rasterTexmapPolygon(surface, image, transform, &region, opacity, _ialpha);
        }
    } else {
        return _rasterTexmapPolygon(surface, image, transform, &region, opacity, nullptr);
    }
    return false;
}


/************************************************************************/
/*Scaled RGBA Image                                                     */
/************************************************************************/


static bool _rasterScaledMaskedTranslucentRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* itransform, const SwBBox& region, uint32_t opacity, uint32_t halfScale, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Scaled Masked Image");

    auto dbuffer = surface->buffer + (region.min.y * surface->stride + region.min.x);
    auto cbuffer = surface->compositor->image.data + (region.min.y * surface->compositor->image.stride + region.min.x);

    // Down-Scaled
    if (image->scale < DOWN_SCALE_TOLERANCE) {
        for (auto y = region.min.y; y < region.max.y; ++y) {
            auto sy = (uint32_t)(y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = dbuffer;
            auto cmp = cbuffer;
            for (auto x = region.min.x; x < region.max.x; ++x, ++dst, ++cmp) {
                auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                if (sx >= image->w) continue;
                auto alpha = _multiplyAlpha(opacity, blendMethod(*cmp));
                auto src = ALPHA_BLEND(_interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale), alpha);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
            dbuffer += surface->stride;
            cbuffer += surface->compositor->image.stride;
        }
    // Up-Scaled
    } else {
        for (auto y = region.min.y; y < region.max.y; ++y) {
            auto sy = y * itransform->e22 + itransform->e23;
            if ((uint32_t)sy >= image->h) continue;
            auto dst = dbuffer;
            auto cmp = cbuffer;
            for (auto x = region.min.x; x < region.max.x; ++x, ++dst, ++cmp) {
                auto sx = x * itransform->e11 + itransform->e13;
                if ((uint32_t)sx >= image->w) continue;
                auto alpha = _multiplyAlpha(opacity, blendMethod(*cmp));
                auto src = ALPHA_BLEND(_interpUpScaler(image->data, image->w, image->h, sx, sy), alpha);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
            dbuffer += surface->stride;
            cbuffer += surface->compositor->image.stride;
        }
    }
    return true;
}


static bool _rasterScaledMaskedRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* itransform, const SwBBox& region, uint32_t halfScale, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Scaled Masked Image");

    auto dbuffer = surface->buffer + (region.min.y * surface->stride + region.min.x);
    auto cbuffer = surface->compositor->image.data + (region.min.y * surface->compositor->image.stride + region.min.x);

    // Down-Scaled
    if (image->scale < DOWN_SCALE_TOLERANCE) {
        for (auto y = region.min.y; y < region.max.y; ++y) {
            auto sy = (uint32_t)(y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = dbuffer;
            auto cmp = cbuffer;
            for (auto x = region.min.x; x < region.max.x; ++x, ++dst, ++cmp) {
                auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                if (sx >= image->w) continue;
                auto src = ALPHA_BLEND(_interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale), blendMethod(*cmp));
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
            dbuffer += surface->stride;
            cbuffer += surface->compositor->image.stride;
        }
    // Up-Scaled
    } else {
        for (auto y = region.min.y; y < region.max.y; ++y) {
            auto sy = y * itransform->e22 + itransform->e23;
            if ((uint32_t)sy >= image->h) continue;
            auto dst = dbuffer;
            auto cmp = cbuffer;
            for (auto x = region.min.x; x < region.max.x; ++x, ++dst, ++cmp) {
                auto sx = x * itransform->e11 + itransform->e13;
                if ((uint32_t)sx >= image->w) continue;
                auto src = ALPHA_BLEND(_interpUpScaler(image->data, image->w, image->h, sx, sy), blendMethod(*cmp));
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
            dbuffer += surface->stride;
            cbuffer += surface->compositor->image.stride;
        }
    }
    return true;
}


static bool _rasterScaledTranslucentRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* itransform, const SwBBox& region, uint32_t opacity, uint32_t halfScale)
{
    auto dbuffer = surface->buffer + (region.min.y * surface->stride + region.min.x);

    // Down-Scaled
    if (image->scale < DOWN_SCALE_TOLERANCE) {
        for (auto y = region.min.y; y < region.max.y; ++y, dbuffer += surface->stride) {
            auto sy = (uint32_t)(y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = dbuffer;
            for (auto x = region.min.x; x < region.max.x; ++x, ++dst) {
                auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                if (sx >= image->w) continue;
                auto src = ALPHA_BLEND(_interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale), opacity);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
        }
    // Up-Scaled
    } else {
        for (auto y = region.min.y; y < region.max.y; ++y, dbuffer += surface->stride) {
            auto sy = fabsf(y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = dbuffer;
            for (auto x = region.min.x; x < region.max.x; ++x, ++dst) {
                auto sx = x * itransform->e11 + itransform->e13;
                if ((uint32_t)sx >= image->w) continue;
                auto src = ALPHA_BLEND(_interpUpScaler(image->data, image->w, image->h, sx, sy), opacity);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
        }
    }
    return true;
}


static bool _rasterScaledRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* itransform, const SwBBox& region, uint32_t halfScale)
{
    auto dbuffer = surface->buffer + (region.min.y * surface->stride + region.min.x);

    // Down-Scaled
    if (image->scale < DOWN_SCALE_TOLERANCE) {
        for (auto y = region.min.y; y < region.max.y; ++y, dbuffer += surface->stride) {
            auto sy = (uint32_t)(y * itransform->e22 + itransform->e23);
            if (sy >= image->h) continue;
            auto dst = dbuffer;
            for (auto x = region.min.x; x < region.max.x; ++x, ++dst) {
                auto sx = (uint32_t)(x * itransform->e11 + itransform->e13);
                if (sx >= image->w) continue;
                auto src = _interpDownScaler(image->data, image->stride, image->w, image->h, sx, sy, halfScale);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
        }
    // Up-Scaled
    } else {
        for (auto y = region.min.y; y < region.max.y; ++y, dbuffer += surface->stride) {
            auto sy = y * itransform->e22 + itransform->e23;
            if ((uint32_t)sy >= image->h) continue;
            auto dst = dbuffer;
            for (auto x = region.min.x; x < region.max.x; ++x, ++dst) {
                auto sx = x * itransform->e11 + itransform->e13;
                if ((uint32_t)sx >= image->w) continue;
                auto src = _interpUpScaler(image->data, image->w, image->h, sx, sy);
                *dst = src + ALPHA_BLEND(*dst, _ialpha(src));
            }
        }
    }
    return true;
}


static bool _scaledRGBAImage(SwSurface* surface, const SwImage* image, const Matrix* transform, const SwBBox& region, uint32_t opacity)
{
    Matrix itransform;
    if (transform && !mathInverse(transform, &itransform)) return false;

    auto halfScale = _halfScale(image->scale);

    if (_compositing(surface)) {
        if (opacity == 255) {
            if (surface->compositor->method == CompositeMethod::AlphaMask) {
                return _rasterScaledMaskedRGBAImage(surface, image, &itransform, region, halfScale, _alpha);
            } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
                return _rasterScaledMaskedRGBAImage(surface, image, &itransform, region, halfScale, _ialpha);
            }
        } else {
            if (surface->compositor->method == CompositeMethod::AlphaMask) {
                return _rasterScaledMaskedTranslucentRGBAImage(surface, image, &itransform, region, opacity, halfScale, _alpha);
            } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
                return _rasterScaledMaskedTranslucentRGBAImage(surface, image, &itransform, region, opacity, halfScale, _ialpha);
            }
        }
    } else {
        if (opacity == 255) return _rasterScaledRGBAImage(surface, image, &itransform, region, halfScale);
        else return _rasterScaledTranslucentRGBAImage(surface, image, &itransform, region, opacity, halfScale);
    }
    return false;
}


/************************************************************************/
/* Direct RGBA Image                                                    */
/************************************************************************/

static bool _rasterDirectMaskedRGBAImage(SwSurface* surface, const SwImage* image, const SwBBox& region, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Direct Masked Image");

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto h2 = static_cast<uint32_t>(region.max.y - region.min.y);
    auto w2 = static_cast<uint32_t>(region.max.x - region.min.x);

    auto sbuffer = image->data + (region.min.y + image->oy) * image->stride + (region.min.x + image->ox);
    auto cbuffer = surface->compositor->image.data + (region.min.y * surface->compositor->image.stride) + region.min.x;   //compositor buffer

    for (uint32_t y = 0; y < h2; ++y) {
        auto dst = buffer;
        auto cmp = cbuffer;
        auto src = sbuffer;
        for (uint32_t x = 0; x < w2; ++x, ++dst, ++src, ++cmp) {
            auto tmp = ALPHA_BLEND(*src, blendMethod(*cmp));
            *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
        }
        buffer += surface->stride;
        cbuffer += surface->compositor->image.stride;
        sbuffer += image->stride;
    }
    return true;
}


static bool _rasterDirectMaskedTranslucentRGBAImage(SwSurface* surface, const SwImage* image, const SwBBox& region, uint32_t opacity, uint32_t (*blendMethod)(uint32_t))
{
    TVGLOG("SW_ENGINE", "Direct Masked Translucent Image");

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto h2 = static_cast<uint32_t>(region.max.y - region.min.y);
    auto w2 = static_cast<uint32_t>(region.max.x - region.min.x);

    auto sbuffer = image->data + (region.min.y + image->oy) * image->stride + (region.min.x + image->ox);
    auto cbuffer = surface->compositor->image.data + (region.min.y * surface->compositor->image.stride) + region.min.x;   //compositor buffer

    for (uint32_t y = 0; y < h2; ++y) {
        auto dst = buffer;
        auto cmp = cbuffer;
        auto src = sbuffer;
        for (uint32_t x = 0; x < w2; ++x, ++dst, ++src, ++cmp) {
            auto tmp = ALPHA_BLEND(*src, _multiplyAlpha(opacity, blendMethod(*cmp)));
            *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
        }
        buffer += surface->stride;
        cbuffer += surface->compositor->image.stride;
        sbuffer += image->stride;
    }
    return true;
}


static bool _rasterDirectTranslucentRGBAImage(SwSurface* surface, const SwImage* image, const SwBBox& region, uint32_t opacity)
{
    auto dbuffer = &surface->buffer[region.min.y * surface->stride + region.min.x];
    auto sbuffer = image->data + (region.min.y + image->oy) * image->stride + (region.min.x + image->ox);

    for (auto y = region.min.y; y < region.max.y; ++y) {
        auto dst = dbuffer;
        auto src = sbuffer;
        for (auto x = region.min.x; x < region.max.x; ++x, ++dst, ++src) {
            auto tmp = ALPHA_BLEND(*src, opacity);
            *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
        }
        dbuffer += surface->stride;
        sbuffer += image->stride;
    }
    return true;
}


static bool _rasterDirectRGBAImage(SwSurface* surface, const SwImage* image, const SwBBox& region)
{
    auto dbuffer = &surface->buffer[region.min.y * surface->stride + region.min.x];
    auto sbuffer = image->data + (region.min.y + image->oy) * image->stride + (region.min.x + image->ox);

    for (auto y = region.min.y; y < region.max.y; ++y) {
        auto dst = dbuffer;
        auto src = sbuffer;
        for (auto x = region.min.x; x < region.max.x; x++, dst++, src++) {
            *dst = *src + ALPHA_BLEND(*dst, _ialpha(*src));
        }
        dbuffer += surface->stride;
        sbuffer += image->stride;
    }
    return true;
}


//Blenders for the following scenarios: [Composition / Non-Composition] * [Opaque / Translucent]
static bool _directRGBAImage(SwSurface* surface, const SwImage* image, const SwBBox& region, uint32_t opacity)
{
    if (_compositing(surface)) {
        if (opacity == 255) {
            if (surface->compositor->method == CompositeMethod::AlphaMask) {
                return _rasterDirectMaskedRGBAImage(surface, image, region, _alpha);
            } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
                return _rasterDirectMaskedRGBAImage(surface, image, region, _ialpha);
            }
        } else {
            if (surface->compositor->method == CompositeMethod::AlphaMask) {
                return _rasterDirectMaskedTranslucentRGBAImage(surface, image, region, opacity, _alpha);
            } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
                return _rasterDirectMaskedTranslucentRGBAImage(surface, image, region, opacity, _ialpha);
            }
        }
    } else {
        if (opacity == 255) return _rasterDirectRGBAImage(surface, image, region);
        else return _rasterDirectTranslucentRGBAImage(surface, image, region, opacity);
    }
    return false;
}


//Blenders for the following scenarios: [RLE / Whole] * [Direct / Scaled / Transformed]
static bool _rasterRGBAImage(SwSurface* surface, SwImage* image, const Matrix* transform, const SwBBox& region, uint32_t opacity)
{
    //RLE Image
    if (image->rle) {
        if (image->direct) return _directRleRGBAImage(surface, image, opacity);
        else if (image->scaled) return _scaledRleRGBAImage(surface, image, transform, region, opacity);
        else return _transformedRleRGBAImage(surface, image, transform, opacity);
    //Whole Image
    } else {
        if (image->direct) return _directRGBAImage(surface, image, region, opacity);
        else if (image->scaled) return _scaledRGBAImage(surface, image, transform, region, opacity);
        else return _transformedRGBAImage(surface, image, transform, region, opacity);
    }
}


/************************************************************************/
/* Rect Linear Gradient                                                 */
/************************************************************************/

static bool _rasterLinearGradientMaskedRect(SwSurface* surface, const SwBBox& region, const SwFill* fill, uint32_t (*blendMethod)(uint32_t))
{
    if (fill->linear.len < FLT_EPSILON) return false;

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto h = static_cast<uint32_t>(region.max.y - region.min.y);
    auto w = static_cast<uint32_t>(region.max.x - region.min.x);
    auto cbuffer = surface->compositor->image.data + (region.min.y * surface->compositor->image.stride) + region.min.x;

    auto sbuffer = static_cast<uint32_t*>(alloca(w * sizeof(uint32_t)));
    if (!sbuffer) return false;

    for (uint32_t y = 0; y < h; ++y) {
        fillFetchLinear(fill, sbuffer, region.min.y + y, region.min.x, w);
        auto dst = buffer;
        auto cmp = cbuffer;
        auto src = sbuffer;
        for (uint32_t x = 0; x < w; ++x, ++dst, ++cmp, ++src) {
            auto tmp = ALPHA_BLEND(*src, blendMethod(*cmp));
            *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
        }
        buffer += surface->stride;
        cbuffer += surface->stride;
    }
    return true;
}


static bool _rasterTranslucentLinearGradientRect(SwSurface* surface, const SwBBox& region, const SwFill* fill)
{
    if (fill->linear.len < FLT_EPSILON) return false;

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto h = static_cast<uint32_t>(region.max.y - region.min.y);
    auto w = static_cast<uint32_t>(region.max.x - region.min.x);

    auto sbuffer = static_cast<uint32_t*>(alloca(w * sizeof(uint32_t)));
    if (!sbuffer) return false;

    for (uint32_t y = 0; y < h; ++y) {
        auto dst = buffer;
        fillFetchLinear(fill, sbuffer, region.min.y + y, region.min.x, w);
        for (uint32_t x = 0; x < w; ++x, ++dst) {
            *dst = sbuffer[x] + ALPHA_BLEND(*dst, _ialpha(sbuffer[x]));
        }
        buffer += surface->stride;
    }
    return true;
}


static bool _rasterSolidLinearGradientRect(SwSurface* surface, const SwBBox& region, const SwFill* fill)
{
    if (fill->linear.len < FLT_EPSILON) return false;

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto w = static_cast<uint32_t>(region.max.x - region.min.x);
    auto h = static_cast<uint32_t>(region.max.y - region.min.y);

    for (uint32_t y = 0; y < h; ++y) {
        fillFetchLinear(fill, buffer + y * surface->stride, region.min.y + y, region.min.x, w);
    }
    return true;
}


static bool _rasterLinearGradientRect(SwSurface* surface, const SwBBox& region, const SwFill* fill)
{
    if (_compositing(surface)) {
        if (surface->compositor->method == CompositeMethod::AlphaMask) {
            return _rasterLinearGradientMaskedRect(surface, region, fill, _alpha);
        } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
            return _rasterLinearGradientMaskedRect(surface, region, fill, _ialpha);
        }
    } else {
        if (fill->translucent) return _rasterTranslucentLinearGradientRect(surface, region, fill);
        else _rasterSolidLinearGradientRect(surface, region, fill);
    }
    return false;
}


/************************************************************************/
/* Rle Linear Gradient                                                  */
/************************************************************************/

static bool _rasterLinearGradientMaskedRle(SwSurface* surface, const SwRleData* rle, const SwFill* fill, uint32_t (*blendMethod)(uint32_t))
{
    if (fill->linear.len < FLT_EPSILON) return false;

    auto span = rle->spans;
    auto cbuffer = surface->compositor->image.data;
    auto buffer = static_cast<uint32_t*>(alloca(surface->w * sizeof(uint32_t)));
    if (!buffer) return false;

    for (uint32_t i = 0; i < rle->size; ++i, ++span) {
        fillFetchLinear(fill, buffer, span->y, span->x, span->len);
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        auto cmp = &cbuffer[span->y * surface->compositor->image.stride + span->x];
        auto src = buffer;
        if (span->coverage == 255) {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp, ++src) {
                auto tmp = ALPHA_BLEND(*src, blendMethod(*cmp));
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        } else {
            auto ialpha = 255 - span->coverage;
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp, ++src) {
                auto tmp = ALPHA_BLEND(*src, blendMethod(*cmp));
                tmp = ALPHA_BLEND(tmp, span->coverage) + ALPHA_BLEND(*dst, ialpha);
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        }
    }
    return true;
}


static bool _rasterTranslucentLinearGradientRle(SwSurface* surface, const SwRleData* rle, const SwFill* fill)
{
    if (fill->linear.len < FLT_EPSILON) return false;

    auto span = rle->spans;
    auto buffer = static_cast<uint32_t*>(alloca(surface->w * sizeof(uint32_t)));
    if (!buffer) return false;

    for (uint32_t i = 0; i < rle->size; ++i, ++span) {
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        fillFetchLinear(fill, buffer, span->y, span->x, span->len);
        if (span->coverage == 255) {
            for (uint32_t i = 0; i < span->len; ++i, ++dst) {
                *dst = buffer[i] + ALPHA_BLEND(*dst, _ialpha(buffer[i]));
            }
        } else {
            for (uint32_t i = 0; i < span->len; ++i, ++dst) {
                auto tmp = ALPHA_BLEND(buffer[i], span->coverage);
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        }
    }
    return true;
}


static bool _rasterSolidLinearGradientRle(SwSurface* surface, const SwRleData* rle, const SwFill* fill)
{
    if (fill->linear.len < FLT_EPSILON) return false;

    auto buf = static_cast<uint32_t*>(alloca(surface->w * sizeof(uint32_t)));
    if (!buf) return false;

    auto span = rle->spans;

    for (uint32_t i = 0; i < rle->size; ++i, ++span) {
        if (span->coverage == 255) {
            fillFetchLinear(fill, surface->buffer + span->y * surface->stride + span->x, span->y, span->x, span->len);
        } else {
            fillFetchLinear(fill, buf, span->y, span->x, span->len);
            auto dst = &surface->buffer[span->y * surface->stride + span->x];
            for (uint32_t i = 0; i < span->len; ++i) {
                dst[i] = INTERPOLATE(span->coverage, buf[i], dst[i]);
            }
        }
    }
    return true;
}


static bool _rasterLinearGradientRle(SwSurface* surface, const SwRleData* rle, const SwFill* fill)
{
    if (!rle) return false;

    if (_compositing(surface)) {
        if (surface->compositor->method == CompositeMethod::AlphaMask) {
            return _rasterLinearGradientMaskedRle(surface, rle, fill, _alpha);
        } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
            return _rasterLinearGradientMaskedRle(surface, rle, fill, _ialpha);
        }
    } else {
        if (fill->translucent) return _rasterTranslucentLinearGradientRle(surface, rle, fill);
        else return _rasterSolidLinearGradientRle(surface, rle, fill);
    }
    return false;
}


/************************************************************************/
/* Rect Radial Gradient                                                 */
/************************************************************************/

static bool _rasterRadialGradientMaskedRect(SwSurface* surface, const SwBBox& region, const SwFill* fill, uint32_t (*blendMethod)(uint32_t))
{
    if (fill->radial.a < FLT_EPSILON) return false;

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto h = static_cast<uint32_t>(region.max.y - region.min.y);
    auto w = static_cast<uint32_t>(region.max.x - region.min.x);
    auto cbuffer = surface->compositor->image.data + (region.min.y * surface->compositor->image.stride) + region.min.x;

    auto sbuffer = static_cast<uint32_t*>(alloca(w * sizeof(uint32_t)));
    if (!sbuffer) return false;

    for (uint32_t y = 0; y < h; ++y) {
        fillFetchRadial(fill, sbuffer, region.min.y + y, region.min.x, w);
        auto dst = buffer;
        auto cmp = cbuffer;
        auto src = sbuffer;
        for (uint32_t x = 0; x < w; ++x, ++dst, ++cmp, ++src) {
             auto tmp = ALPHA_BLEND(*src, blendMethod(*cmp));
             *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
        }
        buffer += surface->stride;
        cbuffer += surface->stride;
    }
    return true;
}


static bool _rasterTranslucentRadialGradientRect(SwSurface* surface, const SwBBox& region, const SwFill* fill)
{
    if (fill->radial.a < FLT_EPSILON) return false;

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto h = static_cast<uint32_t>(region.max.y - region.min.y);
    auto w = static_cast<uint32_t>(region.max.x - region.min.x);

    auto sbuffer = static_cast<uint32_t*>(alloca(w * sizeof(uint32_t)));
    if (!sbuffer) return false;

    for (uint32_t y = 0; y < h; ++y) {
        auto dst = buffer;
        fillFetchRadial(fill, sbuffer, region.min.y + y, region.min.x, w);
        for (uint32_t x = 0; x < w; ++x, ++dst) {
            *dst = sbuffer[x] + ALPHA_BLEND(*dst, _ialpha(sbuffer[x]));
        }
        buffer += surface->stride;
    }
    return true;
}


static bool _rasterSolidRadialGradientRect(SwSurface* surface, const SwBBox& region, const SwFill* fill)
{
    if (fill->radial.a < FLT_EPSILON) return false;

    auto buffer = surface->buffer + (region.min.y * surface->stride) + region.min.x;
    auto h = static_cast<uint32_t>(region.max.y - region.min.y);
    auto w = static_cast<uint32_t>(region.max.x - region.min.x);

    for (uint32_t y = 0; y < h; ++y) {
        auto dst = &buffer[y * surface->stride];
        fillFetchRadial(fill, dst, region.min.y + y, region.min.x, w);
    }
    return true;
}


static bool _rasterRadialGradientRect(SwSurface* surface, const SwBBox& region, const SwFill* fill)
{
    if (_compositing(surface)) {
        if (surface->compositor->method == CompositeMethod::AlphaMask) {
            return _rasterRadialGradientMaskedRect(surface, region, fill, _alpha);
        } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
            return _rasterRadialGradientMaskedRect(surface, region, fill, _ialpha);
        }
    } else {
        if (fill->translucent) return _rasterTranslucentRadialGradientRect(surface, region, fill);
        else return _rasterSolidRadialGradientRect(surface, region, fill);
    }
    return false;
}


/************************************************************************/
/* RLE Radial Gradient                                                  */
/************************************************************************/

static bool _rasterRadialGradientMaskedRle(SwSurface* surface, const SwRleData* rle, const SwFill* fill, uint32_t (*blendMethod)(uint32_t))
{
    if (fill->radial.a < FLT_EPSILON) return false;

    auto span = rle->spans;
    auto cbuffer = surface->compositor->image.data;
    auto buffer = static_cast<uint32_t*>(alloca(surface->w * sizeof(uint32_t)));
    if (!buffer) return false;

    for (uint32_t i = 0; i < rle->size; ++i, ++span) {
        fillFetchRadial(fill, buffer, span->y, span->x, span->len);
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        auto cmp = &cbuffer[span->y * surface->compositor->image.stride + span->x];
        auto src = buffer;
        if (span->coverage == 255) {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp, ++src) {
                auto tmp = ALPHA_BLEND(*src, blendMethod(*cmp));
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        } else {
            for (uint32_t x = 0; x < span->len; ++x, ++dst, ++cmp, ++src) {
                auto tmp = INTERPOLATE(span->coverage, ALPHA_BLEND(*src, blendMethod(*cmp)), *dst);
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        }
    }
    return true;
}


static bool _rasterTranslucentRadialGradientRle(SwSurface* surface, const SwRleData* rle, const SwFill* fill)
{
    if (fill->radial.a < FLT_EPSILON) return false;

    auto span = rle->spans;
    auto buffer = static_cast<uint32_t*>(alloca(surface->w * sizeof(uint32_t)));
    if (!buffer) return false;

    for (uint32_t i = 0; i < rle->size; ++i, ++span) {
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        fillFetchRadial(fill, buffer, span->y, span->x, span->len);
        if (span->coverage == 255) {
            for (uint32_t i = 0; i < span->len; ++i, ++dst) {
                *dst = buffer[i] + ALPHA_BLEND(*dst, _ialpha(buffer[i]));
            }
        } else {
           for (uint32_t i = 0; i < span->len; ++i, ++dst) {
                auto tmp = ALPHA_BLEND(buffer[i], span->coverage);
                *dst = tmp + ALPHA_BLEND(*dst, _ialpha(tmp));
            }
        }
    }
    return true;
}


static bool _rasterSolidRadialGradientRle(SwSurface* surface, const SwRleData* rle, const SwFill* fill)
{
    if (fill->radial.a < FLT_EPSILON) return false;

    auto buf = static_cast<uint32_t*>(alloca(surface->w * sizeof(uint32_t)));
    if (!buf) return false;

    auto span = rle->spans;

    for (uint32_t i = 0; i < rle->size; ++i, ++span) {
        auto dst = &surface->buffer[span->y * surface->stride + span->x];
        if (span->coverage == 255) {
            fillFetchRadial(fill, dst, span->y, span->x, span->len);
        } else {
            fillFetchRadial(fill, buf, span->y, span->x, span->len);
            auto ialpha = 255 - span->coverage;
            for (uint32_t i = 0; i < span->len; ++i, ++dst) {
                *dst = ALPHA_BLEND(buf[i], span->coverage) + ALPHA_BLEND(*dst, ialpha);
            }
        }
    }
    return true;
}


static bool _rasterRadialGradientRle(SwSurface* surface, const SwRleData* rle, const SwFill* fill)
{
    if (!rle) return false;

    if (_compositing(surface)) {
        if (surface->compositor->method == CompositeMethod::AlphaMask) {
            return _rasterRadialGradientMaskedRle(surface, rle, fill, _alpha);
        } else if (surface->compositor->method == CompositeMethod::InvAlphaMask) {
            return _rasterRadialGradientMaskedRle(surface, rle, fill, _ialpha);
        }
    } else {
        if (fill->translucent) _rasterTranslucentRadialGradientRle(surface, rle, fill);
        else return _rasterSolidRadialGradientRle(surface, rle, fill);
    }
    return false;
}


/************************************************************************/
/* External Class Implementation                                        */
/************************************************************************/

void rasterRGBA32(uint32_t *dst, uint32_t val, uint32_t offset, int32_t len)
{
#if defined(THORVG_AVX_VECTOR_SUPPORT)
    avxRasterRGBA32(dst, val, offset, len);
#elif defined(THORVG_NEON_VECTOR_SUPPORT)
    neonRasterRGBA32(dst, val, offset, len);
#else
    cRasterRGBA32(dst, val, offset, len);
#endif
}


bool rasterCompositor(SwSurface* surface)
{
    if (surface->cs == SwCanvas::ABGR8888 || surface->cs == SwCanvas::ABGR8888_STRAIGHT) {
        surface->blender.join = _abgrJoin;
    } else if (surface->cs == SwCanvas::ARGB8888 || surface->cs == SwCanvas::ARGB8888_STRAIGHT) {
        surface->blender.join = _argbJoin;
    } else {
        //What Color Space ???
        return false;
    }
    return true;
}


bool rasterClear(SwSurface* surface)
{
    if (!surface || !surface->buffer || surface->stride <= 0 || surface->w <= 0 || surface->h <= 0) return false;

    if (surface->w == surface->stride) {
        rasterRGBA32(surface->buffer, 0x00000000, 0, surface->w * surface->h);
    } else {
        for (uint32_t i = 0; i < surface->h; i++) {
            rasterRGBA32(surface->buffer + surface->stride * i, 0x00000000, 0, surface->w);
        }
    }
    return true;
}


void rasterUnpremultiply(SwSurface* surface)
{
    //OPTIMIZE_ME: +SIMD
    for (uint32_t y = 0; y < surface->h; y++) {
        auto buffer = surface->buffer + surface->stride * y;
        for (uint32_t x = 0; x < surface->w; ++x) {
            uint8_t a = buffer[x] >> 24;
            if (a == 255) {
                continue;
            } else if (a == 0) {
                buffer[x] = 0x00ffffff;
            } else {
                uint16_t r = ((buffer[x] >> 8) & 0xff00) / a;
                uint16_t g = ((buffer[x]) & 0xff00) / a;
                uint16_t b = ((buffer[x] << 8) & 0xff00) / a;
                if (r > 0xff) r = 0xff;
                if (g > 0xff) g = 0xff;
                if (b > 0xff) b = 0xff;
                buffer[x] = (a << 24) | (r << 16) | (g << 8) | (b);
            }
        }
    }
}


bool rasterGradientShape(SwSurface* surface, SwShape* shape, unsigned id)
{
    if (!shape->fill) return false;

    if (shape->fastTrack) {
        if (id == TVG_CLASS_ID_LINEAR) return _rasterLinearGradientRect(surface, shape->bbox, shape->fill);
        else if (id == TVG_CLASS_ID_RADIAL)return _rasterRadialGradientRect(surface, shape->bbox, shape->fill);
    } else {
        if (id == TVG_CLASS_ID_LINEAR) return _rasterLinearGradientRle(surface, shape->rle, shape->fill);
        else if (id == TVG_CLASS_ID_RADIAL) return _rasterRadialGradientRle(surface, shape->rle, shape->fill);
    }
    return false;
}


bool rasterGradientStroke(SwSurface* surface, SwShape* shape, unsigned id)
{
    if (!shape->stroke || !shape->stroke->fill || !shape->strokeRle) return false;

    if (id == TVG_CLASS_ID_LINEAR) return _rasterLinearGradientRle(surface, shape->strokeRle, shape->stroke->fill);
    else if (id == TVG_CLASS_ID_RADIAL) return _rasterRadialGradientRle(surface, shape->strokeRle, shape->stroke->fill);

    return false;
}


bool rasterShape(SwSurface* surface, SwShape* shape, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (a < 255) {
        r = _multiplyAlpha(r, a);
        g = _multiplyAlpha(g, a);
        b = _multiplyAlpha(b, a);
    }

    auto color = surface->blender.join(r, g, b, a);

    if (shape->fastTrack) return _rasterRect(surface, shape->bbox, color, a);
    else return _rasterRle(surface, shape->rle, color, a);
}


bool rasterStroke(SwSurface* surface, SwShape* shape, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    if (a < 255) {
        r = _multiplyAlpha(r, a);
        g = _multiplyAlpha(g, a);
        b = _multiplyAlpha(b, a);
    }

    auto color = surface->blender.join(r, g, b, a);

    return _rasterRle(surface, shape->strokeRle, color, a);
}


bool rasterImage(SwSurface* surface, SwImage* image, const Matrix* transform, const SwBBox& bbox, uint32_t opacity)
{
    //Verify Boundary
    if (bbox.max.x < 0 || bbox.max.y < 0 || bbox.min.x >= surface->w || bbox.min.y >= surface->h) return false;

    //TOOD: switch (image->format)
    //TODO: case: _rasterRGBImage()
    //TODO: case: _rasterGrayscaleImage()
    //TODO: case: _rasterAlphaImage()
    return _rasterRGBAImage(surface, image, transform, bbox, opacity);
}