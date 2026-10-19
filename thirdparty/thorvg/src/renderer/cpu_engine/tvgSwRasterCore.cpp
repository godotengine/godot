/*
 * Copyright (c) 2021 - 2026 ThorVG project. All rights reserved.

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
/* See tvgSwRasterSimd.cpp                                              */
/************************************************************************/

#ifdef THORVG_AVX_SUPPORT
    #if defined(_MSC_VER)
        #include <intrin.h>
    #endif
    uint32_t avxRasterDownScaler(const uint32_t* img, uint32_t stride, uint32_t w, TVG_UNUSED uint32_t h, float sx, TVG_UNUSED float sy, int32_t miny, int32_t maxy, int32_t n);
    void avxRasterTranslucentPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity);
    void avxRasterSolidPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity);
    void avxRasterSolidPixel32(uint32_t* dst, uint32_t val, uint32_t offset, int32_t len);
    bool avxRasterTranslucentRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c);
    bool avxRasterTranslucentRle(SwSurface* surface, const SwRle* rle, const RenderRegion& bbox, const RenderColor& c);
    void avxRasterUnpremultiply(uint32_t* buffer, uint32_t width);
    void avxRasterPremultiply(uint32_t* buffer, uint32_t width);
    bool avxRasterABGRtoARGB(RenderSurface* surface);
    void avxRasterGaussianFilterRow(uint8_t* dst, const uint8_t* src, int32_t w, int32_t dimension, float iarr);
    void avxRasterGaussianXYFlipBlock(const uint32_t* src, uint32_t* dst, int32_t stride);
#elif defined(THORVG_NEON_SUPPORT)
    #if defined(__linux__) && defined(__arm__)
        #include <sys/auxv.h>
        #include <asm/hwcap.h>
    #endif
    void neonRasterUnpremultiply(uint32_t* buffer, uint32_t width);
    void neonRasterPremultiply(uint32_t* buffer, uint32_t width);
    uint32_t neonRasterDownScaler(const uint32_t* img, uint32_t stride, uint32_t w, TVG_UNUSED uint32_t h, float sx, TVG_UNUSED float sy, int32_t miny, int32_t maxy, int32_t n);
    void neonRasterTranslucentPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity);
    void neonRasterSolidPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity);
    void neonRasterSolidPixel32(uint32_t* dst, uint32_t val, uint32_t offset, int32_t len);
    bool neonRasterTranslucentRle(SwSurface* surface, const SwRle* rle, const RenderRegion& bbox, const RenderColor& c);
    bool neonRasterTranslucentRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c);
    bool neonRasterABGRtoARGB(RenderSurface* surface);
    void neonRasterGaussianFilterRow(uint8_t* dst, const uint8_t* src, int32_t w, int32_t dimension, float iarr);
    void neonRasterGaussianXYFlipBlock(const uint32_t* src, uint32_t* dst, int32_t stride);
#endif

/************************************************************************/
/* Scalar                                                               */
/************************************************************************/

void cRasterGaussianFilterRow(uint8_t* dst, const uint8_t* src, int32_t w, int32_t dimension, float iarr)
{
    auto end = w - 1;
    auto i = 0;                 // current index
    auto l = -(dimension + 1);  // left index
    auto r = dimension;         // right index
    int acc[4] = {0, 0, 0, 0};  // sliding accumulator

    // initial accumulation
    for (int x = l; x < r; ++x) {
        auto id = tvg::clamp(x, 0, end) * 4;
        acc[0] += src[id++];
        acc[1] += src[id++];
        acc[2] += src[id++];
        acc[3] += src[id];
    }
    // perform filtering
    for (int x = 0; x < w; ++x, ++r, ++l) {
        auto rid = tvg::clamp(r, 0, end) * 4;
        auto lid = tvg::clamp(l, 0, end) * 4;
        acc[0] += src[rid++] - src[lid++];
        acc[1] += src[rid++] - src[lid++];
        acc[2] += src[rid++] - src[lid++];
        acc[3] += src[rid] - src[lid];
        // ignored rounding for the performance. It should be originally: acc[idx] * iarr + 0.5f
        dst[i++] = static_cast<uint8_t>(acc[0] * iarr);
        dst[i++] = static_cast<uint8_t>(acc[1] * iarr);
        dst[i++] = static_cast<uint8_t>(acc[2] * iarr);
        dst[i++] = static_cast<uint8_t>(acc[3] * iarr);
    }
}

void cRasterUnpremultiply(uint32_t* buffer, uint32_t width)
{
    for (uint32_t x = 0; x < width; ++x) {
        buffer[x] = UNPREMULTIPLY(buffer[x]);
    }
}

void cRasterPremultiply(uint32_t* buffer, uint32_t width)
{
    for (uint32_t x = 0; x < width; ++x) {
        auto c = buffer[x];
        if (A(c) != 255) buffer[x] = PREMULTIPLY(c, A(c));
    }
}

uint32_t cRasterDownScaler(const uint32_t* img, uint32_t stride, uint32_t w, TVG_UNUSED uint32_t h, float sx, TVG_UNUSED float sy, int32_t miny, int32_t maxy, int32_t n)
{
    size_t c[4] = {0, 0, 0, 0};

    auto minx = static_cast<int32_t>(sx) - n;
    if (minx < 0) minx = 0;

    auto maxx = static_cast<int32_t>(sx) + n;
    if (maxx >= static_cast<int32_t>(w)) maxx = w;

    auto inc = (n / 2) + 1;
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

void cRasterTranslucentPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity)
{
    if (opacity == 255) {
        for (uint32_t x = 0; x < len; ++x, ++dst, ++src) {
            *dst = *src + ALPHA_BLEND(*dst, IA(*src));
        }
    } else {
        for (uint32_t x = 0; x < len; ++x, ++dst, ++src) {
            auto tmp = ALPHA_BLEND(*src, opacity);
            *dst = tmp + ALPHA_BLEND(*dst, IA(tmp));
        }
    }
}

void cRasterSolidPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity)
{
    if (opacity == 255) memcpy(dst, src, size_t(len) * sizeof(uint32_t));
    else cRasterTranslucentPixels(dst, src, len, opacity);
}

void cRasterSolidPixel32(uint32_t* dst, uint32_t val, uint32_t offset, int32_t len)
{
    dst += offset;
    for (int32_t x = 0; x < len; ++x) *dst++ = val;
}

bool cRasterTranslucentRle(SwSurface* surface, const SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
    const SwSpan* end;
    int32_t x, len;

    //32bit channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, c.a);
        uint32_t src;
        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            auto dst = &surface->buf32[span->y * surface->stride + x];
            if (span->coverage < 255) src = ALPHA_BLEND(color, span->coverage);
            else src = color;
            auto ialpha = IA(src);
            for (auto x = 0; x < len; ++x, ++dst) {
                *dst = src + ALPHA_BLEND(*dst, ialpha);
            }
        }
    //8bit grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        uint8_t src;
        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            auto dst = &surface->buf8[span->y * surface->stride + x];
            if (span->coverage < 255) src = MULTIPLY(span->coverage, c.a);
            else src = c.a;
            auto ialpha = ~c.a;
            for (auto x = 0; x < len; ++x, ++dst) {
                *dst = src + MULTIPLY(*dst, ialpha);
            }
        }
    }
    return true;
}

bool cRasterTranslucentRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    //32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, c.a);
        auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;
        auto ialpha = 255 - c.a;
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            auto dst = &buffer[y * surface->stride];
            for (uint32_t x = 0; x < bbox.w(); ++x, ++dst) {
                *dst = color + ALPHA_BLEND(*dst, ialpha);
            }
        }
    //8bit grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto buffer = surface->buf8 + (bbox.yMin * surface->stride) + bbox.xMin;
        auto ialpha = ~c.a;
        for (uint32_t y = 0; y < bbox.h(); ++y) {
            auto dst = &buffer[y * surface->stride];
            for (uint32_t x = 0; x < bbox.w(); ++x, ++dst) {
                *dst = c.a + MULTIPLY(*dst, ialpha);
            }
        }
    }
    return true;
}

bool cRasterABGRtoARGB(RenderSurface* surface)
{
    TVGLOG("SW_ENGINE", "Convert ColorSpace ABGR - ARGB [Size: %d x %d]", surface->w, surface->h);

    auto buffer = surface->buf32;
    #pragma omp parallel for
    for (int32_t y = 0; y < (int32_t)surface->h; ++y) {
        auto dst = buffer + uint32_t(y) * surface->stride;
        for (uint32_t x = 0; x < surface->w; ++x, ++dst) {
            auto c = *dst;
            //flip Blue, Red channels
            *dst = (c & 0xff000000) + ((c & 0x00ff0000) >> 16) + (c & 0x0000ff00) + ((c & 0x000000ff) << 16);
        }
    }

    return true;
}

/************************************************************************/
/* Runtime Dispatch                                                     */
/************************************************************************/

#if defined(THORVG_AVX_SUPPORT) || defined(THORVG_NEON_SUPPORT)
    uint32_t (*rasterDownScaler)(const uint32_t*, uint32_t, uint32_t, uint32_t, float, float, int32_t, int32_t, int32_t) = cRasterDownScaler;
    void (*rasterUnpremultiply)(uint32_t*, uint32_t) = cRasterUnpremultiply;
    void (*rasterPremultiply)(uint32_t*, uint32_t) = cRasterPremultiply;
    void (*rasterTranslucentPixels)(uint32_t*, uint32_t*, uint32_t, uint8_t) = cRasterTranslucentPixels;
    void (*rasterSolidPixels)(uint32_t*, uint32_t*, uint32_t, uint8_t) = cRasterSolidPixels;
    void (*rasterSolidPixel32)(uint32_t*, uint32_t, uint32_t, int32_t) = cRasterSolidPixel32;
    bool (*rasterTranslucentRle)(SwSurface*, const SwRle*, const RenderRegion&, const RenderColor&) = cRasterTranslucentRle;
    bool (*rasterTranslucentRect)(SwSurface*, const RenderRegion&, const RenderColor&) = cRasterTranslucentRect;
    bool (*rasterABGRtoARGB)(RenderSurface*) = cRasterABGRtoARGB;
    void (*rasterGaussianFilterRow)(uint8_t*, const uint8_t*, int32_t, int32_t, float) = cRasterGaussianFilterRow;
    void (*rasterGaussianXYFlipBlock)(const uint32_t*, uint32_t*, int32_t) = nullptr;  // A null keeps the scalar path in the caller.
#endif

void rasterInit()
{
#if defined(THORVG_AVX_SUPPORT)
    // AVX2 also requires OS support for saving XMM/YMM state.
    #if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7) return;
        __cpuidex(info, 1, 0);
        if ((info[2] & 0x18000000) != 0x18000000) return;
        if ((_xgetbv(0) & 6) != 6) return;
        __cpuidex(info, 7, 0);
        if (!(info[1] & (1 << 5))) return;
    #elif defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();
        if (!__builtin_cpu_supports("avx2")) return;
    #else
        return;
    #endif
    rasterDownScaler = avxRasterDownScaler;
    rasterUnpremultiply = avxRasterUnpremultiply;
    rasterPremultiply = avxRasterPremultiply;
    rasterTranslucentPixels = avxRasterTranslucentPixels;
    rasterSolidPixels = avxRasterSolidPixels;
    rasterSolidPixel32 = avxRasterSolidPixel32;
    rasterTranslucentRle = avxRasterTranslucentRle;
    rasterTranslucentRect = avxRasterTranslucentRect;
    rasterABGRtoARGB = avxRasterABGRtoARGB;
    rasterGaussianFilterRow = avxRasterGaussianFilterRow;
    rasterGaussianXYFlipBlock = avxRasterGaussianXYFlipBlock;
    TVGLOG("SW_ENGINE", "Run AVX Vectorization.");
#elif defined(THORVG_NEON_SUPPORT)
    #if defined(__linux__) && defined(__arm__)
        if (!(getauxval(AT_HWCAP) & HWCAP_NEON)) return;
    #endif
    rasterDownScaler = neonRasterDownScaler;
    rasterUnpremultiply = neonRasterUnpremultiply;
    rasterPremultiply = neonRasterPremultiply;
    rasterTranslucentPixels = neonRasterTranslucentPixels;
    rasterSolidPixels = neonRasterSolidPixels;
    rasterSolidPixel32 = neonRasterSolidPixel32;
    rasterTranslucentRle = neonRasterTranslucentRle;
    rasterTranslucentRect = neonRasterTranslucentRect;
    rasterABGRtoARGB = neonRasterABGRtoARGB;
    rasterGaussianFilterRow = neonRasterGaussianFilterRow;
    rasterGaussianXYFlipBlock = neonRasterGaussianXYFlipBlock;
    TVGLOG("SW_ENGINE", "Run NEON Vectorization.");
#endif
}
