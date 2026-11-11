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
/* AVX2                                                                 */
/************************************************************************/

#ifdef THORVG_AVX_SUPPORT

#include <immintrin.h>

#define N_32BITS_IN_256REG 8

static __m256i avxScalePixels(__m256i pixels, __m256i factors)
{
    // Each 32-bit lane contains one pixel and a factor in [0, 256].
    // Scale alternating channels in 16-bit lanes without unpacking the pixels.
    const auto mask = _mm256_set1_epi32(0x00ff00ff);
    factors = _mm256_or_si256(factors, _mm256_slli_epi32(factors, 16));
    auto even = _mm256_mullo_epi16(_mm256_and_si256(pixels, mask), factors);
    auto odd = _mm256_mullo_epi16(_mm256_srli_epi16(pixels, 8), factors);
    return _mm256_or_si256(_mm256_srli_epi16(even, 8), _mm256_andnot_si256(mask, odd));
}

uint32_t avxRasterDownScaler(const uint32_t* img, uint32_t stride, uint32_t w, TVG_UNUSED uint32_t h, float sx, TVG_UNUSED float sy, int32_t miny, int32_t maxy, int32_t n)
{
    // At most 4*4 pixels contribute, so each channel sum fits in uint16_t.
    auto sum = _mm_setzero_si128();

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
            sum = _mm_add_epi16(sum, _mm_cvtepu8_epi16(_mm_cvtsi32_si128(*p)));
            ++n;
        }
        src += (stride * inc);
    }

    auto c0 = _mm_extract_epi16(sum, 0) / n;
    auto c1 = _mm_extract_epi16(sum, 1) / n;
    auto c2 = _mm_extract_epi16(sum, 2) / n;
    auto c3 = static_cast<uint32_t>(_mm_extract_epi16(sum, 3)) / n;

    return (c3 << 24) | (c2 << 16) | (c1 << 8) | c0;
}

void avxRasterTranslucentPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity)
{
    const auto opacityVec = _mm256_set1_epi32(opacity + 1);
    const auto full = _mm256_set1_epi32(256);
    uint32_t i = 0;
    for (; len - i >= 8; i += 8) {
        auto source = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i));
        if (opacity != 255) source = avxScalePixels(source, opacityVec);
        auto inverse = _mm256_sub_epi32(full, _mm256_srli_epi32(source, 24));
        auto target = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), _mm256_add_epi32(source, avxScalePixels(target, inverse)));
    }
    cRasterTranslucentPixels(dst + i, src + i, len - i, opacity);
}

void avxRasterSolidPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity)
{
    if (opacity == 255) memcpy(dst, src, size_t(len) * sizeof(uint32_t));
    else avxRasterTranslucentPixels(dst, src, len, opacity);
}

void avxRasterSolidPixel32(uint32_t* dst, uint32_t val, uint32_t offset, int32_t len)
{
    if (len <= 0) return;
    // 1. calculate how many iterations we need to cover the length
    uint32_t iterations = len / N_32BITS_IN_256REG;
    uint32_t avxFilled = iterations * N_32BITS_IN_256REG;

    // 2. set the beginning of the array
    dst += offset;

    // 3. fill the octets
    for (uint32_t i = 0; i < iterations; ++i, dst += N_32BITS_IN_256REG) {
        _mm256_storeu_si256((__m256i*)dst, _mm256_set1_epi32(val));
    }

    // 4. fill leftovers (in the first step we have to set the pointer to the place where the avx job is done)
    int32_t leftovers = len - avxFilled;
    while (leftovers--) *dst++ = val;
}

bool avxRasterTranslucentRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    auto h = bbox.h();
    auto w = bbox.w();

    // 32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, c.a);
        auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;
        uint32_t ialpha = 255 - c.a;
        auto avxColor = _mm256_set1_epi32(color);
        auto avxIalpha = _mm256_set1_epi32(ialpha + 1);
        for (uint32_t y = 0; y < h; ++y) {
            auto dst = &buffer[y * surface->stride];
            uint32_t x = 0;
            for (; w - x >= 8; x += 8) {
                auto pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + x));
                auto result = _mm256_add_epi32(avxColor, avxScalePixels(pixels, avxIalpha));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), result);
            }
            for (; x < w; ++x) {
                dst[x] = color + ALPHA_BLEND(dst[x], ialpha);
            }
        }
    // 8bit grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto buffer = surface->buf8 + (bbox.yMin * surface->stride) + bbox.xMin;
        const auto ialpha = static_cast<uint8_t>(~c.a);
        const auto mask = _mm256_set1_epi16(0x00ff);
        const auto avxIalpha = _mm256_set1_epi16(ialpha);
        const auto avxAlpha = _mm256_set1_epi8(c.a);
        for (uint32_t y = 0; y < h; ++y) {
            auto dst = &buffer[y * surface->stride];
            uint32_t x = 0;
            for (; w - x >= 32; x += 32) {
                auto pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + x));
                auto even = _mm256_mullo_epi16(_mm256_and_si256(pixels, mask), avxIalpha);
                auto odd = _mm256_mullo_epi16(_mm256_srli_epi16(pixels, 8), avxIalpha);
                even = _mm256_srli_epi16(_mm256_add_epi16(even, mask), 8);
                odd = _mm256_andnot_si256(mask, _mm256_add_epi16(odd, mask));
                auto result = _mm256_add_epi8(avxAlpha, _mm256_or_si256(even, odd));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), result);
            }
            for (; x < w; ++x) {
                dst[x] = c.a + MULTIPLY(dst[x], ialpha);
            }
        }
    }
    return true;
}

bool avxRasterTranslucentRle(SwSurface* surface, const SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
    const SwSpan* end;
    int32_t x, len;

    // 32bit channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, c.a);
        uint32_t src;

        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            if (span->coverage < 255) src = ALPHA_BLEND(color, span->coverage);
            else src = color;
            auto dst = &surface->buf32[span->y * surface->stride + x];
            auto ialpha = IA(src);
            auto avxSrc = _mm256_set1_epi32(src);
            auto avxIalpha = _mm256_set1_epi32(ialpha + 1);
            int32_t i = 0;
            for (; i <= len - 8; i += 8) {
                auto pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
                auto result = _mm256_add_epi32(avxSrc, avxScalePixels(pixels, avxIalpha));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), result);
            }
            for (; i < len; ++i) {
                dst[i] = src + ALPHA_BLEND(dst[i], ialpha);
            }
        }
    // 8bit grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        const auto mask = _mm256_set1_epi16(0x00ff);
        const auto ialpha = static_cast<uint8_t>(~c.a);
        const auto avxIalpha = _mm256_set1_epi16(ialpha);
        uint8_t src;
        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            auto dst = &surface->buf8[span->y * surface->stride + x];
            if (span->coverage < 255) src = MULTIPLY(span->coverage, c.a);
            else src = c.a;
            const auto avxSrc = _mm256_set1_epi8(src);
            int32_t i = 0;
            for (; i <= len - 32; i += 32) {
                auto pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
                auto even = _mm256_mullo_epi16(_mm256_and_si256(pixels, mask), avxIalpha);
                auto odd = _mm256_mullo_epi16(_mm256_srli_epi16(pixels, 8), avxIalpha);
                even = _mm256_srli_epi16(_mm256_add_epi16(even, mask), 8);
                odd = _mm256_andnot_si256(mask, _mm256_add_epi16(odd, mask));
                auto result = _mm256_add_epi8(avxSrc, _mm256_or_si256(even, odd));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), result);
            }
            for (; i < len; ++i) {
                dst[i] = src + MULTIPLY(dst[i], ialpha);
            }
        }
    }
    return true;
}

void avxRasterUnpremultiply(uint32_t* buffer, uint32_t width)
{
    const auto mask = _mm256_set1_epi32(0xff);
    const auto full = _mm256_set1_epi32(255);
    const auto one = _mm256_set1_epi32(1);
    uint32_t x = 0;
    for (; width - x >= 8; x += 8) {
        auto pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer + x));
        auto alpha = _mm256_srli_epi32(pixels, 24);
        auto unchanged = _mm256_or_si256(_mm256_cmpeq_epi32(alpha, full), _mm256_cmpeq_epi32(alpha, _mm256_setzero_si256()));
        if (_mm256_movemask_epi8(unchanged) == -1) continue;
        auto divisor = _mm256_cvtepi32_ps(_mm256_max_epi32(alpha, one));
        auto result = _mm256_slli_epi32(alpha, 24);
        for (int shift = 0; shift < 24; shift += 8) {
            auto bits = _mm_cvtsi32_si128(shift);
            auto channel = _mm256_and_si256(_mm256_srl_epi32(pixels, bits), mask);
            auto scaled = _mm256_mullo_epi32(channel, full);
            auto quotient = _mm256_cvttps_epi32(_mm256_div_ps(_mm256_cvtepi32_ps(scaled), divisor));
            result = _mm256_or_si256(result, _mm256_sll_epi32(_mm256_min_epi32(quotient, full), bits));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer + x), _mm256_blendv_epi8(result, pixels, unchanged));
    }
    cRasterUnpremultiply(buffer + x, width - x);
}

void avxRasterPremultiply(uint32_t* buffer, uint32_t width)
{
    const auto full = _mm256_set1_epi32(255);
    const auto alphaMask = _mm256_set1_epi32(0xff000000);
    uint32_t x = 0;
    for (; width - x >= 8; x += 8) {
        auto pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buffer + x));
        auto alpha = _mm256_srli_epi32(pixels, 24);
        // A factor of 256 preserves opaque pixels; other pixels use alpha / 256.
        auto factors = _mm256_sub_epi32(alpha, _mm256_cmpeq_epi32(alpha, full));
        auto result = avxScalePixels(pixels, factors);
        result = _mm256_blendv_epi8(result, pixels, alphaMask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(buffer + x), result);
    }
    cRasterPremultiply(buffer + x, width - x);
}

bool avxRasterABGRtoARGB(RenderSurface* surface)
{
    const auto shuffle = _mm256_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
                                          2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
    const auto end = surface->w & ~7u;
    #pragma omp parallel for
    for (int32_t y = 0; y < (int32_t)surface->h; ++y) {
        auto dst = surface->buf32 + uint32_t(y) * surface->stride;
        uint32_t x = 0;
        for (; x < end; x += 8) {
            auto pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + x));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + x), _mm256_shuffle_epi8(pixels, shuffle));
        }
        for (; x < surface->w; ++x) {
            auto c = dst[x];
            dst[x] = (c & 0xff00ff00) | ((c & 0x00ff0000) >> 16) | ((c & 0x000000ff) << 16);
        }
    }
    return true;
}

void avxRasterGaussianFilterRow(uint8_t* dst, const uint8_t* src, int32_t w, int32_t dimension, float iarr)
{
    auto end = w - 1;
    auto i = 0;
    auto l = -(dimension + 1);
    auto r = dimension;
    auto acc = _mm_setzero_si128();
    const auto zero = _mm_setzero_si128();
    const auto scale = _mm_set1_ps(iarr);

    for (int x = l; x < r; ++x) {
        auto id = tvg::clamp(x, 0, end) * 4;
        uint32_t pixel;
        memcpy(&pixel, src + id, sizeof(pixel));
        acc = _mm_add_epi32(acc, _mm_cvtepu8_epi32(_mm_cvtsi32_si128(pixel)));
    }

    for (int x = 0; x < w; ++x, ++r, ++l) {
        auto rid = tvg::clamp(r, 0, end) * 4;
        auto lid = tvg::clamp(l, 0, end) * 4;
        uint32_t right, left;
        memcpy(&right, src + rid, sizeof(right));
        memcpy(&left, src + lid, sizeof(left));
        auto added = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(right));
        auto removed = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(left));
        acc = _mm_add_epi32(acc, _mm_sub_epi32(added, removed));
        auto values = _mm_cvttps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(acc), scale));
        auto packed = _mm_packus_epi16(_mm_packs_epi32(values, zero), zero);
        auto pixel = static_cast<uint32_t>(_mm_cvtsi128_si32(packed));
        memcpy(dst + i, &pixel, sizeof(pixel));
        i += 4;
    }
}

void avxRasterGaussianXYFlipBlock(const uint32_t* src, uint32_t* dst, int32_t stride)
{
    // Transpose one complete 8x8 block using four 4x4 tiles.
    for (int32_t x = 0; x < 8; x += 4) {
        for (int32_t y = 0; y < 8; y += 4) {
            auto s = src + x + y * stride;
            auto d = dst + x * stride + y;
            auto r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
            auto r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + stride));
            auto r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 2 * stride));
            auto r3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 3 * stride));
            auto t0 = _mm_unpacklo_epi32(r0, r1);
            auto t1 = _mm_unpackhi_epi32(r0, r1);
            auto t2 = _mm_unpacklo_epi32(r2, r3);
            auto t3 = _mm_unpackhi_epi32(r2, r3);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d), _mm_unpacklo_epi64(t0, t2));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d + stride), _mm_unpackhi_epi64(t0, t2));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 2 * stride), _mm_unpacklo_epi64(t1, t3));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(d + 3 * stride), _mm_unpackhi_epi64(t1, t3));
        }
    }
}

#endif

/************************************************************************/
/* Neon                                                                 */
/************************************************************************/

#ifdef THORVG_NEON_SUPPORT

#include <arm_neon.h>

// TODO: Verify MSVC ARM64 NEON header and intrinsic compatibility, including the Meson probe.
#if defined(__ARM_64BIT_STATE) || defined(_M_ARM64)
    #define TVG_AARCH64 1
#else
    #define TVG_AARCH64 0
#endif

static uint8x8_t ALPHA_BLEND(uint8x8_t c, uint8x8_t a)
{
    return vshrn_n_u16(vmull_u8(c, a), 8);
}

static uint8x8_t neonScalePixels(uint8x8_t pixels, uint32x2_t factors)
{
    auto expanded = vcombine_u16(vdup_n_u16(static_cast<uint16_t>(vget_lane_u32(factors, 0))),
                                 vdup_n_u16(static_cast<uint16_t>(vget_lane_u32(factors, 1))));
    return vmovn_u16(vshrq_n_u16(vmulq_u16(vmovl_u8(pixels), expanded), 8));
}

void neonRasterUnpremultiply(uint32_t* buffer, uint32_t width)
{
    const auto mask = vdupq_n_u32(255);
    const auto one = vdupq_n_u32(1);
    uint32_t x = 0;
    for (; x + 4 <= width; x += 4) {
        auto pixels = vld1q_u32(buffer + x);
        auto alpha = vshrq_n_u32(pixels, 24);
        auto unchanged = vorrq_u32(vceqq_u32(alpha, mask), vceqq_u32(alpha, vdupq_n_u32(0)));
        if (vget_lane_u64(vreinterpret_u64_u32(vand_u32(vget_low_u32(unchanged), vget_high_u32(unchanged))), 0) == UINT64_MAX) continue;
        auto divisor = vmaxq_u32(alpha, one);
        auto divisorFloat = vcvtq_f32_u32(divisor);
        auto reciprocal = vrecpeq_f32(divisorFloat);
        reciprocal = vmulq_f32(reciprocal, vrecpsq_f32(divisorFloat, reciprocal));
        reciprocal = vmulq_f32(reciprocal, vrecpsq_f32(divisorFloat, reciprocal));
        auto result = vshlq_n_u32(alpha, 24);
        for (int shift = 0; shift < 24; shift += 8) {
            auto bits = vdupq_n_s32(shift);
            auto channel = vandq_u32(vshlq_u32(pixels, vnegq_s32(bits)), mask);
            auto numerator = vmulq_u32(channel, mask);
            auto quotient = vcvtq_u32_f32(vmulq_f32(vcvtq_f32_u32(numerator), reciprocal));
            auto product = vmulq_u32(quotient, divisor);
            quotient = vsubq_u32(quotient, vandq_u32(vcgtq_u32(product, numerator), one));
            product = vmulq_u32(quotient, divisor);
            quotient = vaddq_u32(quotient, vandq_u32(vcgeq_u32(numerator, vaddq_u32(product, divisor)), one));
            result = vorrq_u32(result, vshlq_u32(vminq_u32(quotient, mask), bits));
        }
        vst1q_u32(buffer + x, vbslq_u32(unchanged, pixels, result));
    }
    cRasterUnpremultiply(buffer + x, width - x);
}

void neonRasterPremultiply(uint32_t* buffer, uint32_t width)
{
    const auto mask = vdupq_n_u32(255);
    uint32_t x = 0;
    for (; x + 4 <= width; x += 4) {
        auto pixels = vld1q_u32(buffer + x);
        auto alpha = vshrq_n_u32(pixels, 24);
        auto result = vshlq_n_u32(alpha, 24);
        for (int shift = 0; shift < 24; shift += 8) {
            auto bits = vdupq_n_s32(shift);
            auto channel = vandq_u32(vshlq_u32(pixels, vnegq_s32(bits)), mask);
            auto scaled = vshrq_n_u32(vmulq_u32(channel, alpha), 8);
            result = vorrq_u32(result, vshlq_u32(scaled, bits));
        }
        auto opaque = vceqq_u32(alpha, mask);
        vst1q_u32(buffer + x, vbslq_u32(opaque, pixels, result));
    }
    cRasterPremultiply(buffer + x, width - x);
}

uint32_t neonRasterDownScaler(const uint32_t* img, uint32_t stride, uint32_t w, TVG_UNUSED uint32_t h, float sx, TVG_UNUSED float sy, int32_t miny, int32_t maxy, int32_t n)
{
    // Each clipped span is at most 2*n; stepping by floor(n/2)+1 takes at most 4 samples per axis.
    // At most 4*4 pixels contribute, so each channel sum is <= 16*255=4080 and fits in uint16_t.
    // vaddw_u8 widens and accumulates 8-bit channels into uint16x8_t in one instruction.
    uint16x8_t sum = vdupq_n_u16(0);

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
            sum = vaddw_u8(sum, vreinterpret_u8_u32(vdup_n_u32(*p)));
            ++n;
        }
        src += (stride * inc);
    }

    uint32_t c[4];
    vst1q_u32(c, vmovl_u16(vrev64_u16(vget_low_u16(sum))));
    c[0] /= n;
    c[1] /= n;
    c[2] /= n;
    c[3] /= n;

    return (c[0] << 24) | (c[1] << 16) | (c[2] << 8) | c[3];
}

void neonRasterTranslucentPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity)
{
    const auto opacityVec = vdup_n_u32(opacity + 1);
    uint32_t i = 0;
    for (; len - i >= 2; i += 2) {
        auto source = vld1_u8(reinterpret_cast<const uint8_t*>(src + i));
        if (opacity != 255) source = neonScalePixels(source, opacityVec);
        auto inverse = vsub_u32(vdup_n_u32(256), vshr_n_u32(vreinterpret_u32_u8(source), 24));
        auto target = vld1_u8(reinterpret_cast<const uint8_t*>(dst + i));
        vst1_u8(reinterpret_cast<uint8_t*>(dst + i), vadd_u8(source, neonScalePixels(target, inverse)));
    }
    cRasterTranslucentPixels(dst + i, src + i, len - i, opacity);
}

void neonRasterSolidPixels(uint32_t* dst, uint32_t* src, uint32_t len, uint8_t opacity)
{
    if (opacity == 255) memcpy(dst, src, size_t(len) * sizeof(uint32_t));
    else neonRasterTranslucentPixels(dst, src, len, opacity);
}

void neonRasterSolidPixel32(uint32_t* dst, uint32_t val, uint32_t offset, int32_t len)
{
    dst += offset;

    uint32x4_t vecVal = vdupq_n_u32(val);

#if TVG_AARCH64
    uint32_t iterations = len / 16;
    uint32_t neonFilled = iterations * 16;
    uint32x4x4_t valQuad = {vecVal, vecVal, vecVal, vecVal};
    for (uint32_t i = 0; i < iterations; ++i, dst += 16) {
        vst4q_u32(dst, valQuad);
    }
#else
    uint32_t iterations = len / 4;
    uint32_t neonFilled = iterations * 4;
    for (uint32_t i = 0; i < iterations; ++i, dst += 4) {
        vst1q_u32(dst, vecVal);
    }
#endif
    auto leftovers = len - neonFilled;
    while (leftovers--) *dst++ = val;
}

bool neonRasterTranslucentRle(SwSurface* surface, const SwRle* rle, const RenderRegion& bbox, const RenderColor& c)
{
    const SwSpan* end;
    int32_t x, len;

    // 32bit channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, c.a);
        uint32_t src;
        uint8x8_t* vDst = nullptr;
        int32_t align;

        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            if (span->coverage < 255) src = ALPHA_BLEND(color, span->coverage);
            else src = color;

            auto dst = &surface->buf32[span->y * surface->stride + x];
            auto ialpha = IA(src);

            if ((((uintptr_t)dst) & 0x7) != 0) {
                // fill not aligned byte
                *dst = src + ALPHA_BLEND(*dst, ialpha);
                vDst = (uint8x8_t*)(dst + 1);
                align = 1;
            } else {
                vDst = (uint8x8_t*)dst;
                align = 0;
            }

            uint8x8_t vSrc = (uint8x8_t)vdup_n_u32(src);
            uint8x8_t vIalpha = vdup_n_u8((uint8_t)ialpha);

            for (int32_t x = 0; x < (len - align) / 2; ++x) {
                vDst[x] = vadd_u8(vSrc, ALPHA_BLEND(vDst[x], vIalpha));
            }

            auto leftovers = (len - align) % 2;
            if (leftovers > 0) dst[len - 1] = src + ALPHA_BLEND(dst[len - 1], ialpha);
        }
    // 8bit grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        const auto ialpha = static_cast<uint8_t>(~c.a);
        const auto vIalpha = vdup_n_u8(ialpha);
        const auto bias = vdupq_n_u16(255);
        uint8_t src;
        for (auto span = rle->fetch(bbox, &end); span < end; ++span) {
            if (!span->fetch(bbox, x, len)) continue;
            auto dst = &surface->buf8[span->y * surface->stride + x];
            if (span->coverage < 255) src = MULTIPLY(span->coverage, c.a);
            else src = c.a;
            const auto vSrc = vdupq_n_u8(src);
            int32_t i = 0;
            for (; len - i >= 16; i += 16) {
                auto pixels = vld1q_u8(dst + i);
                auto low = vmlal_u8(bias, vget_low_u8(pixels), vIalpha);
                auto high = vmlal_u8(bias, vget_high_u8(pixels), vIalpha);
                auto scaled = vcombine_u8(vshrn_n_u16(low, 8), vshrn_n_u16(high, 8));
                vst1q_u8(dst + i, vaddq_u8(vSrc, scaled));
            }
            for (; i < len; ++i) {
                dst[i] = src + MULTIPLY(dst[i], ialpha);
            }
        }
    }
    return true;
}

bool neonRasterTranslucentRect(SwSurface* surface, const RenderRegion& bbox, const RenderColor& c)
{
    auto h = bbox.h();
    auto w = bbox.w();

    // 32bits channels
    if (surface->channelSize == sizeof(uint32_t)) {
        auto color = surface->join(c.r, c.g, c.b, c.a);
        auto buffer = surface->buf32 + (bbox.yMin * surface->stride) + bbox.xMin;
        auto ialpha = 255 - c.a;
        auto vColor = vdup_n_u32(color);
        auto vIalpha = vdup_n_u8((uint8_t)ialpha);
        uint8x8_t* vDst = nullptr;
        uint32_t align;

        for (uint32_t y = 0; y < h; ++y) {
            auto dst = &buffer[y * surface->stride];

            if ((((uintptr_t)dst) & 0x7) != 0) {
                // fill not aligned byte
                *dst = color + ALPHA_BLEND(*dst, ialpha);
                vDst = (uint8x8_t*)(dst + 1);
                align = 1;
            } else {
                vDst = (uint8x8_t*)dst;
                align = 0;
            }

            for (uint32_t x = 0; x <  (w - align) / 2; ++x) {
                vDst[x] = vadd_u8((uint8x8_t)vColor, ALPHA_BLEND(vDst[x], vIalpha));
            }

            auto leftovers = (w - align) % 2;
            if (leftovers > 0) dst[w - 1] = color + ALPHA_BLEND(dst[w - 1], ialpha);
        }
    // 8bit grayscale
    } else if (surface->channelSize == sizeof(uint8_t)) {
        auto buffer = surface->buf8 + (bbox.yMin * surface->stride) + bbox.xMin;
        const auto ialpha = static_cast<uint8_t>(~c.a);
        const auto vIalpha = vdup_n_u8(ialpha);
        const auto vAlpha = vdupq_n_u8(c.a);
        const auto bias = vdupq_n_u16(255);
        for (uint32_t y = 0; y < h; ++y) {
            auto dst = &buffer[y * surface->stride];
            uint32_t x = 0;
            for (; w - x >= 16; x += 16) {
                auto pixels = vld1q_u8(dst + x);
                auto low = vmlal_u8(bias, vget_low_u8(pixels), vIalpha);
                auto high = vmlal_u8(bias, vget_high_u8(pixels), vIalpha);
                auto scaled = vcombine_u8(vshrn_n_u16(low, 8), vshrn_n_u16(high, 8));
                vst1q_u8(dst + x, vaddq_u8(vAlpha, scaled));
            }
            for (; x < w; ++x) {
                dst[x] = c.a + MULTIPLY(dst[x], ialpha);
            }
        }
    }
    return true;
}

bool neonRasterABGRtoARGB(RenderSurface* surface)
{
    const auto mask = vdupq_n_u32(0xff00ff00);
    const auto end = surface->w & ~3u;
    #pragma omp parallel for
    for (int32_t y = 0; y < (int32_t)surface->h; ++y) {
        auto dst = surface->buf32 + uint32_t(y) * surface->stride;
        uint32_t x = 0;
        for (; x < end; x += 4) {
            auto pixels = vld1q_u32(dst + x);
            auto swapped = vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(pixels)));
            vst1q_u32(dst + x, vbslq_u32(mask, pixels, swapped));
        }
        for (; x < surface->w; ++x) {
            auto c = dst[x];
            dst[x] = (c & 0xff00ff00) | ((c & 0x00ff0000) >> 16) | ((c & 0x000000ff) << 16);
        }
    }
    return true;
}

void neonRasterGaussianFilterRow(uint8_t* dst, const uint8_t* src, int32_t w, int32_t dimension, float iarr)
{
    auto end = w - 1;
    auto i = 0;
    auto l = -(dimension + 1);
    auto r = dimension;
    auto acc = vdupq_n_s32(0);
    const auto scale = vdupq_n_f32(iarr);

    for (int x = l; x < r; ++x) {
        auto id = tvg::clamp(x, 0, end) * 4;
        uint32_t pixel;
        memcpy(&pixel, src + id, sizeof(pixel));
        auto channels = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(pixel))));
        acc = vaddq_s32(acc, vreinterpretq_s32_u32(channels));
    }

    for (int x = 0; x < w; ++x, ++r, ++l) {
        auto rid = tvg::clamp(r, 0, end) * 4;
        auto lid = tvg::clamp(l, 0, end) * 4;
        uint32_t right, left;
        memcpy(&right, src + rid, sizeof(right));
        memcpy(&left, src + lid, sizeof(left));
        auto added = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(right))));
        auto removed = vmovl_u16(vget_low_u16(vmovl_u8(vcreate_u8(left))));
        acc = vaddq_s32(acc, vsubq_s32(vreinterpretq_s32_u32(added), vreinterpretq_s32_u32(removed)));
        auto values = vcvtq_s32_f32(vmulq_f32(vcvtq_f32_s32(acc), scale));
        auto packed = vmovn_u16(vcombine_u16(vqmovun_s32(values), vdup_n_u16(0)));
        auto pixel = vget_lane_u32(vreinterpret_u32_u8(packed), 0);
        memcpy(dst + i, &pixel, sizeof(pixel));
        i += 4;
    }
}

void neonRasterGaussianXYFlipBlock(const uint32_t* src, uint32_t* dst, int32_t stride)
{
    // Transpose one complete 8x8 block using four 4x4 tiles.
    for (int32_t x = 0; x < 8; x += 4) {
        for (int32_t y = 0; y < 8; y += 4) {
            auto s = src + x + y * stride;
            auto d = dst + x * stride + y;
            auto r0 = vld1q_u32(s);
            auto r1 = vld1q_u32(s + stride);
            auto r2 = vld1q_u32(s + 2 * stride);
            auto r3 = vld1q_u32(s + 3 * stride);
            auto t0 = vtrnq_u32(r0, r1);
            auto t1 = vtrnq_u32(r2, r3);
            vst1q_u32(d, vcombine_u32(vget_low_u32(t0.val[0]), vget_low_u32(t1.val[0])));
            vst1q_u32(d + stride, vcombine_u32(vget_low_u32(t0.val[1]), vget_low_u32(t1.val[1])));
            vst1q_u32(d + 2 * stride, vcombine_u32(vget_high_u32(t0.val[0]), vget_high_u32(t1.val[0])));
            vst1q_u32(d + 3 * stride, vcombine_u32(vget_high_u32(t0.val[1]), vget_high_u32(t1.val[1])));
        }
    }
}

#endif
