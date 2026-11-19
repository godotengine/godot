/*
 * Copyright (c) 2024 - 2026 ThorVG project. All rights reserved.

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
/* Gaussian Blur Implementation                                         */
/************************************************************************/

struct SwGaussianBlur
{
    static constexpr int MAX_LEVEL = 3;
    int level;
    int kernel[MAX_LEVEL];
    int extends;
};

static inline int _gaussianEdgeExtend(int end, int idx)
{
    if (idx < 0) return 0;
    else if (idx > end) return end;
    return idx;
}

static void _gaussianFilter(uint8_t* dst, uint8_t* src, int32_t stride, int32_t w, int32_t h, const RenderRegion& bbox, int32_t dimension, bool flipped)
{
    if (flipped) {
        src += (bbox.xMin * stride + bbox.yMin) << 2;
        dst += (bbox.xMin * stride + bbox.yMin) << 2;
    } else {
        src += (bbox.yMin * stride + bbox.xMin) << 2;
        dst += (bbox.yMin * stride + bbox.xMin) << 2;
    }

    auto iarr = 1.0f / (dimension + dimension + 1);

    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        rasterGaussianFilterRow(dst + y * stride * 4, src + y * stride * 4, w, dimension, iarr);
    }
}

static void _gaussianXYFlip(uint32_t* src, uint32_t* dst, int32_t stride, int32_t w, int32_t h, const RenderRegion& bbox, bool flipped)
{
    constexpr int32_t BLOCK = 8;  // experimental decision

    if (flipped) {
        src += ((bbox.xMin * stride) + bbox.yMin);
        dst += ((bbox.yMin * stride) + bbox.xMin);
    } else {
        src += ((bbox.yMin * stride) + bbox.xMin);
        dst += ((bbox.xMin * stride) + bbox.yMin);
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
#if defined(THORVG_AVX_SUPPORT) || defined(THORVG_NEON_SUPPORT)
            if (bx == BLOCK && by == BLOCK && rasterGaussianXYFlipBlock) {
                rasterGaussianXYFlipBlock(p, q, stride);
                continue;
            }
#endif
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

//Fast Almost-Gaussian Filtering Method by Peter Kovesi
static int _gaussianInit(SwGaussianBlur* data, float sigma, int quality)
{
    const auto MAX_LEVEL = SwGaussianBlur::MAX_LEVEL;

    if (tvg::zero(sigma)) return 0;

    data->level = int(SwGaussianBlur::MAX_LEVEL * ((quality - 1) * 0.01f)) + 1;

    //compute box kernel sizes
    auto wl = (int) sqrt((12 * sigma / MAX_LEVEL) + 1);
    if (wl % 2 == 0) --wl;
    auto wu = wl + 2;
    auto mi = (12 * sigma - MAX_LEVEL * wl * wl - 4 * MAX_LEVEL * wl - 3 * MAX_LEVEL) / (-4 * wl - 4);
    auto m = int(mi + 0.5f);
    auto extends = 0;

    for (int i = 0; i < data->level; i++) {
        data->kernel[i] = ((i < m ? wl : wu) - 1) / 2;
        extends += data->kernel[i];
    }

    return extends;
}

bool effectGaussianBlurRegion(RenderEffectGaussianBlur* params)
{
    //region expansion for feathering
    auto& bbox = params->extend;
    auto extra = static_cast<SwGaussianBlur*>(params->rd)->extends;

    if (params->direction != 2) {
        bbox.xMin = -extra;
        bbox.xMax = extra;
    }
    if (params->direction != 1) {
        bbox.yMin = -extra;
        bbox.yMax = extra;
    }

    return true;
}

void effectGaussianBlurUpdate(RenderEffectGaussianBlur* params, const Matrix& transform)
{
    if (!params->rd) params->rd = tvg::malloc<SwGaussianBlur>(sizeof(SwGaussianBlur));
    auto rd = static_cast<SwGaussianBlur*>(params->rd);

    //compute box kernel sizes
    auto scaleSquared = transform.e11 * transform.e11 + transform.e12 * transform.e12;
    rd->extends = _gaussianInit(rd, params->sigma * params->sigma * scaleSquared, params->quality);

    params->valid = (rd->extends > 0);
}

bool effectGaussianBlur(SwCompositor* cmp, SwSurface* surface, const RenderEffectGaussianBlur* params)
{
    auto& buffer = surface->compositor->image;
    auto data = static_cast<SwGaussianBlur*>(params->rd);
    auto& bbox = cmp->bbox;
    auto w = (bbox.xMax - bbox.xMin);
    auto h = (bbox.yMax - bbox.yMin);
    auto stride = cmp->image.stride;
    auto front = cmp->image.buf32;
    auto back = buffer.buf32;
    auto swapped = false;

    TVGLOG("SW_ENGINE", "GaussianFilter region(%d, %d, %d, %d) params(%f %d), level(%d)", bbox.xMin, bbox.yMin, bbox.xMax, bbox.yMax, params->sigma, params->direction, data->level);

    /* It is best to take advantage of the Gaussian blur’s separable property
       by dividing the process into two passes. horizontal and vertical.
       We can expect fewer calculations. */

    //horizontal
    if (params->direction != 2) {
        for (int i = 0; i < data->level; ++i) {
            _gaussianFilter(reinterpret_cast<uint8_t*>(back), reinterpret_cast<uint8_t*>(front), stride, w, h, bbox, data->kernel[i], false);
            std::swap(front, back);
            swapped = !swapped;
        }
    }

    //vertical. x/y flipping and horionztal access is pretty compatible with the memory architecture.
    if (params->direction != 1) {
        _gaussianXYFlip(front, back, stride, w, h, bbox, false);
        std::swap(front, back);

        for (int i = 0; i < data->level; ++i) {
            _gaussianFilter(reinterpret_cast<uint8_t*>(back), reinterpret_cast<uint8_t*>(front), stride, h, w, bbox, data->kernel[i], true);
            std::swap(front, back);
            swapped = !swapped;
        }

        _gaussianXYFlip(front, back, stride, h, w, bbox, true);
        std::swap(front, back);
    }

    if (swapped) std::swap(cmp->image.buf8, buffer.buf8);

    return true;
}

/************************************************************************/
/* Motion Blur Implementation                                           */
/************************************************************************/

struct SwMotionBlur
{
    static constexpr int MAX_TAPS = 4 * 257;

    // integer pixel offsets with accumulated bilinear weights scaled by 65536.
    struct Tap
    {
        int32_t x, y;
        uint32_t weight;
    } taps[MAX_TAPS];

    int count;  // sample count, before merging bilinear taps
    int tapCount;
    SwPoint extent;
};

void effectMotionBlurUpdate(RenderEffectMotionBlur* params, const Matrix& transform)
{
    Point offset;
    int samples;
    if (!params->update(transform, offset, samples)) return;

    if (!params->rd) params->rd = tvg::malloc<SwMotionBlur>(sizeof(SwMotionBlur));
    auto data = static_cast<SwMotionBlur*>(params->rd);
    data->extent = {int32_t(ceilf(fabsf(offset.x) * 0.5f)), int32_t(ceilf(fabsf(offset.y) * 0.5f))};
    data->count = samples;

    // flatten bilinear samples into nonzero integer taps.
    data->tapCount = 0;
    for (int i = 0; i < data->count; ++i) {
        auto pt = offset * ((i + 0.5f) / data->count - 0.5f);
        auto x = int32_t(floorf(pt.x));
        auto y = int32_t(floorf(pt.y));
        auto fx = uint32_t((pt.x - x) * 256.0f + 0.5f);
        auto fy = uint32_t((pt.y - y) * 256.0f + 0.5f);
        uint32_t weights[] = {(256 - fx) * (256 - fy), fx * (256 - fy), (256 - fx) * fy, fx * fy};
        for (int k = 0; k < 4; ++k) {
            if (weights[k]) data->taps[data->tapCount++] = {x + (k & 1), y + (k >> 1), weights[k]};
        }
    }

    // merge repeated coordinates without changing the total integer weight.
    std::qsort(data->taps, data->tapCount, sizeof(SwMotionBlur::Tap), [](const void* lhs, const void* rhs) {
        auto a = static_cast<const SwMotionBlur::Tap*>(lhs);
        auto b = static_cast<const SwMotionBlur::Tap*>(rhs);
        if (a->y != b->y) return (a->y > b->y) ? 1 : -1;
        return (a->x > b->x) - (a->x < b->x);
    });
    int merged = 0;
    for (int i = 0; i < data->tapCount; ++i) {
        auto& tap = data->taps[i];
        if (merged > 0 && data->taps[merged - 1].x == tap.x && data->taps[merged - 1].y == tap.y) {
            data->taps[merged - 1].weight += tap.weight;
        } else {
            data->taps[merged++] = tap;
        }
    }
    data->tapCount = merged;
    params->valid = true;
}

bool effectMotionBlurRegion(RenderEffectMotionBlur* params)
{
    auto data = static_cast<SwMotionBlur*>(params->rd);
    params->extend = {-data->extent.x, -data->extent.y, data->extent.x, data->extent.y};
    return true;
}

bool effectMotionBlur(SwCompositor* cmp, SwSurface* surface, const RenderEffectMotionBlur* params)
{
    auto& buffer = surface->compositor->image;
    auto data = static_cast<SwMotionBlur*>(params->rd);
    auto& bbox = cmp->bbox;
    auto stride = cmp->image.stride;
    auto w = bbox.xMax - bbox.xMin;
    auto h = bbox.yMax - bbox.yMin;
    auto divisor = uint64_t(data->count) * 65536;
    auto src = cmp->image.buf32 + bbox.yMin * stride + bbox.xMin;
    auto dst = buffer.buf32 + bbox.yMin * stride + bbox.xMin;

    #pragma omp parallel for
    for (int32_t y = 0; y < h; ++y) {
        // Y clamping and row addressing are shared by all pixels in this row.
        const uint32_t* tapRows[SwMotionBlur::MAX_TAPS];
        for (int i = 0; i < data->tapCount; ++i) {
            auto sy = _gaussianEdgeExtend(h - 1, y + data->taps[i].y);
            tapRows[i] = src + sy * stride;
        }
        // Accumulate weighted taps with X coordinates clamped to the row edges.
        // Each sample contributes 65536 total weight; normalize with rounding.
        auto row = (uint8_t*)(dst + y * stride);
        for (int32_t x = 0; x < w; ++x) {
            uint64_t acc[4] = {};
            for (int i = 0; i < data->tapCount; ++i) {
                auto& tap = data->taps[i];
                auto sx = _gaussianEdgeExtend(w - 1, x + tap.x);
                auto pixel = (uint8_t*)(tapRows[i] + sx);
                // Filter premultiplied channels together to preserve alpha.
                for (int c = 0; c < 4; ++c) {
                    acc[c] += uint64_t(pixel[c]) * tap.weight;
                }
            }
            for (int c = 0; c < 4; ++c) {
                row[x * 4 + c] = uint8_t((acc[c] + divisor / 2) / divisor);
            }
        }
    }

    std::swap(cmp->image.buf8, buffer.buf8);
    return true;
}

/************************************************************************/
/* Drop Shadow Implementation                                           */
/************************************************************************/

struct SwDropShadow : SwGaussianBlur
{
    SwPoint offset;
};

//TODO: SIMD OPTIMIZATION?
static void _dropShadowFilter(uint32_t* dst, uint32_t* src, int stride, int w, int h, const RenderRegion& bbox, int32_t dimension, uint32_t color, bool flipped)
{
    if (flipped) {
        src += (bbox.xMin * stride + bbox.yMin);
        dst += (bbox.xMin * stride + bbox.yMin);
    } else {
        src += (bbox.yMin * stride + bbox.xMin);
        dst += (bbox.yMin * stride + bbox.xMin);
    }
    auto iarr = 1.0f / (dimension + dimension + 1);
    auto end = w - 1;

    #pragma omp parallel for
    for (int y = 0; y < h; ++y) {
        auto p = y * stride;
        auto i = p;                     //current index
        auto l = -(dimension + 1);      //left index
        auto r = dimension;             //right index
        int acc = 0;                    //sliding accumulator

        //initial accumulation
        for (int x = l; x < r; ++x) {
            auto id = _gaussianEdgeExtend(end, x) + p;
            acc += A(src[id]);
        }
        //perform filtering
        for (int x = 0; x < w; ++x, ++r, ++l) {
            auto rid = _gaussianEdgeExtend(end, r) + p;
            auto lid = _gaussianEdgeExtend(end, l) + p;
            acc += A(src[rid]) - A(src[lid]);
            //ignored rounding for the performance. It should be originally: acc * iarr
            dst[i++] = ALPHA_BLEND(color, static_cast<uint8_t>(acc * iarr));
        }
    }
}

static void _shift(uint32_t** dst, uint32_t** src, int dstride, int sstride, int wmax, int hmax, const RenderRegion& bbox, SwPoint offset, SwSize& size)
{
    size.w = bbox.xMax - bbox.xMin;
    size.h = bbox.yMax - bbox.yMin;

    //shift
    if (offset.x < 0) {
        *src -= offset.x;
        size.w += offset.x;
    } else {
        *dst += offset.x;
        size.w -= offset.x;
    }

    if (offset.y < 0) {
        *src -= (offset.y * sstride);
        size.h += offset.y;
    } else {
        *dst += (offset.y * dstride);
        size.h -= offset.y;
    }
}

// TODO: simd & openmp optimization?
static void _dropShadowNoFilter(uint32_t* dst, uint32_t* src, int dstride, int sstride, int dw, int dh, const RenderRegion& bbox, const SwPoint& offset, uint32_t color, uint8_t opacity, bool direct)
{
    src += (bbox.yMin * sstride + bbox.xMin);
    dst += (bbox.yMin * dstride + bbox.xMin);

    SwSize size;
    _shift(&dst, &src, dstride, sstride, dw, dh, bbox, offset, size);

    for (auto y = 0; y < size.h; ++y) {
        auto s2 = src;
        auto d2 = dst;
        for (int x = 0; x < size.w; ++x, ++d2, ++s2) {
            auto a = MULTIPLY(opacity, A(*s2));
            if (!direct || a == 255) *d2 = ALPHA_BLEND(color, a);
            else *d2 = INTERPOLATE(color, *d2, a);
        }
        src += sstride;
        dst += dstride;
    }
}

static void _dropShadowNoFilter(SwImage* dimg, SwImage* simg, const RenderRegion& bbox, const SwPoint& offset, uint32_t color)
{
    int dstride = dimg->stride;
    int sstride = simg->stride;

    //shadow image
    _dropShadowNoFilter(dimg->buf32, simg->buf32, dstride, sstride, dimg->w, dimg->h, bbox, offset, color, 255, false);

    //original image
    auto src = simg->buf32 + (bbox.yMin * sstride + bbox.xMin);
    auto dst = dimg->buf32 + (bbox.yMin * dstride + bbox.xMin);

    // TODO: openmp optimization?
    for (auto y = 0; y < (bbox.yMax - bbox.yMin); ++y) {
        rasterTranslucentPixels(dst, src, bbox.xMax - bbox.xMin, 255);
        src += sstride;
        dst += dstride;
    }
}

static void _dropShadowShift(uint32_t* dst, uint32_t* src, int dstride, int sstride, int dw, int dh, const RenderRegion& bbox, const SwPoint& offset, uint8_t opacity, bool direct)
{
    src += (bbox.yMin * sstride + bbox.xMin);
    dst += (bbox.yMin * dstride + bbox.xMin);

    SwSize size;
    _shift(&dst, &src, dstride, sstride, dw, dh, bbox, offset, size);

    // TODO: openmp optimization?
    for (auto y = 0; y < size.h; ++y) {
        if (direct) rasterTranslucentPixels(dst, src, size.w, opacity);
        else rasterSolidPixels(dst, src, size.w, opacity);
        src += sstride;
        dst += dstride;
    }
}

bool effectDropShadowRegion(RenderEffectDropShadow* params)
{
    //region expansion for feathering
    auto& bbox = params->extend;
    auto& offset = static_cast<SwDropShadow*>(params->rd)->offset;
    auto extra = static_cast<SwDropShadow*>(params->rd)->extends;

    bbox.xMin = -extra;
    bbox.yMin = -extra;
    bbox.xMax = extra;
    bbox.yMax = extra;

    if (offset.x < 0) bbox.xMin += (int32_t)offset.x;
    else bbox.xMax += offset.x;

    if (offset.y < 0) bbox.yMin += (int32_t)offset.y;
    else bbox.yMax += offset.y;

    return true;
}

void effectDropShadowUpdate(RenderEffectDropShadow* params, const Matrix& transform)
{
    Point offset;
    if (!params->update(transform, offset)) return;

    if (!params->rd) params->rd = tvg::malloc<SwDropShadow>(sizeof(SwDropShadow));
    auto rd = static_cast<SwDropShadow*>(params->rd);

    // The sum of the box radii covers the full blur support.
    auto scaleSquared = transform.e11 * transform.e11 + transform.e12 * transform.e12;
    rd->extends = _gaussianInit(rd, params->sigma * params->sigma * scaleSquared, params->quality);

    rd->offset = {(int32_t)offset.x, (int32_t)offset.y};
}

//A quite same integration with effectGaussianBlur(). See it for detailed comments.
//surface[0]: the original image, to overlay it into the filtered image.
//surface[1]: temporary buffer for generating the filtered image.
bool effectDropShadow(SwCompositor* cmp, SwSurface* surface[2], const RenderEffectDropShadow* params, bool direct)
{
    //FIXME: if the body is partially visible due to clipping, the shadow also becomes partially visible.

    auto data = static_cast<SwDropShadow*>(params->rd);
    auto& bbox = cmp->bbox;
    auto w = (bbox.xMax - bbox.xMin);
    auto h = (bbox.yMax - bbox.yMin);

    //outside the screen
    if (abs(data->offset.x) >= w || abs(data->offset.y) >= h) return true;

    SwImage* buffer[] = {&surface[0]->compositor->image, &surface[1]->compositor->image};
    auto color = cmp->recoverSfc->join(params->color[0], params->color[1], params->color[2], 255);
    auto stride = cmp->image.stride;
    auto front = cmp->image.buf32;
    auto back = buffer[1]->buf32;

    auto opacity = direct ? MULTIPLY(params->color[3], cmp->opacity) : params->color[3];

    TVGLOG("SW_ENGINE", "DropShadow region(%d, %d, %d, %d) params(%f %f %f), level(%d)", bbox.xMin, bbox.yMin, bbox.xMax, bbox.yMax, params->angle, params->distance, params->sigma, data->level);

    //no filter required
    if (data->extends == 0)  {
        if (direct) {
            _dropShadowNoFilter(cmp->recoverSfc->buf32, cmp->image.buf32, cmp->recoverSfc->stride, cmp->image.stride, cmp->recoverSfc->w, cmp->recoverSfc->h, bbox, data->offset, color, opacity, direct);
        } else {
            _dropShadowNoFilter(buffer[1], &cmp->image, bbox, data->offset, color);
            std::swap(cmp->image.buf32, buffer[1]->buf32);
        }
        return true;
    }

    //saving the original image in order to overlay it into the filtered image.
    _dropShadowFilter(back, front, stride, w, h, bbox, data->kernel[0], color, false);
    std::swap(front, buffer[0]->buf32);
    std::swap(front, back);

    //horizontal
    for (int i = 1; i < data->level; ++i) {
        _dropShadowFilter(back, front, stride, w, h, bbox, data->kernel[i], color, false);
        std::swap(front, back);
    }

    //vertical
    _gaussianXYFlip(front, back, stride, w, h, bbox, false);
    std::swap(front, back);

    for (int i = 0; i < data->level; ++i) {
        _dropShadowFilter(back, front, stride, h, w, bbox, data->kernel[i], color, true);
        std::swap(front, back);
    }

    _gaussianXYFlip(front, back, stride, h, w, bbox, true);
    std::swap(cmp->image.buf32, back);

    //draw to the main surface directly
    if (direct) {
        _dropShadowShift(cmp->recoverSfc->buf32, cmp->image.buf32, cmp->recoverSfc->stride, cmp->image.stride, cmp->recoverSfc->w, cmp->recoverSfc->h, bbox, data->offset, opacity, direct);
        std::swap(cmp->image.buf32, buffer[0]->buf32);
        return true;
    }

    //draw to the intermediate surface
    rasterClear(surface[1], bbox.xMin, bbox.yMin, w, h);
    _dropShadowShift(buffer[1]->buf32, cmp->image.buf32, buffer[1]->stride, cmp->image.stride, buffer[1]->w, buffer[1]->h, bbox, data->offset, opacity, direct);
    std::swap(cmp->image.buf32, buffer[1]->buf32);

    //compositing shadow and body
    auto s = buffer[0]->buf32 + (bbox.yMin * buffer[0]->stride + bbox.xMin);
    auto d = cmp->image.buf32 + (bbox.yMin * cmp->image.stride + bbox.xMin);

    // TODO: openmp optimization?
    for (auto y = 0; y < h; ++y) {
        rasterTranslucentPixels(d, s, w, 255);
        s += buffer[0]->stride;
        d += cmp->image.stride;
    }

    return true;
}

/************************************************************************/
/* Fill Implementation                                                  */
/************************************************************************/

void effectFillUpdate(RenderEffectFill* params)
{
    params->valid = true;
}

// TODO: simd & openmp optimization?
bool effectFill(SwCompositor* cmp, const RenderEffectFill* params, bool direct)
{
    auto opacity = direct ? MULTIPLY(params->color[3], cmp->opacity) : params->color[3];

    auto& bbox = cmp->bbox;
    auto w = size_t(bbox.xMax - bbox.xMin);
    auto h = size_t(bbox.yMax - bbox.yMin);
    auto color = cmp->recoverSfc->join(params->color[0], params->color[1], params->color[2], 255);

    TVGLOG("SW_ENGINE", "Fill region(%d, %d, %d, %d), param(%d %d %d %d)", bbox.xMin, bbox.yMin, bbox.xMax, bbox.yMax, params->color[0], params->color[1], params->color[2], params->color[3]);

    if (direct) {
        auto dbuffer = cmp->recoverSfc->buf32 + (bbox.yMin * cmp->recoverSfc->stride + bbox.xMin);
        auto sbuffer = cmp->image.buf32 + (bbox.yMin * cmp->image.stride + bbox.xMin);
        for (size_t y = 0; y < h; ++y) {
            auto dst = dbuffer;
            auto src = sbuffer;
            for (size_t x = 0; x < w; ++x, ++dst, ++src) {
                auto a = MULTIPLY(opacity, A(*src));
                auto tmp = ALPHA_BLEND(color, a);
                *dst = tmp + ALPHA_BLEND(*dst, 255 - a);
            }
            dbuffer += cmp->recoverSfc->stride;
            sbuffer += cmp->image.stride;
        }
        cmp->valid = true;  //no need the subsequent composition
    } else {
        auto dbuffer = cmp->image.buf32 + (bbox.yMin * cmp->image.stride + bbox.xMin);
        for (size_t y = 0; y < h; ++y) {
            auto dst = dbuffer;
            for (size_t x = 0; x < w; ++x, ++dst) {
                *dst = ALPHA_BLEND(color, MULTIPLY(opacity, A(*dst)));
            }
            dbuffer += cmp->image.stride;
        }
    }
    return true;
}

/************************************************************************/
/* Tint Implementation                                                  */
/************************************************************************/

void effectTintUpdate(RenderEffectTint* params)
{
    params->valid = (params->intensity > 0);
}

// TODO: simd & openmp optimization?
bool effectTint(SwCompositor* cmp, const RenderEffectTint* params, bool direct)
{
    auto& bbox = cmp->bbox;
    auto w = size_t(bbox.xMax - bbox.xMin);
    auto h = size_t(bbox.yMax - bbox.yMin);
    auto black = cmp->recoverSfc->join(params->black[0], params->black[1], params->black[2], 255);
    auto white = cmp->recoverSfc->join(params->white[0], params->white[1], params->white[2], 255);
    auto opacity = cmp->opacity;
    auto luma = cmp->recoverSfc->alphas[2];  //luma function

    TVGLOG("SW_ENGINE", "Tint region(%d, %d, %d, %d), param(%d %d %d, %d %d %d, %d)", bbox.xMin, bbox.yMin, bbox.xMax, bbox.yMax, params->black[0], params->black[1], params->black[2], params->white[0], params->white[1], params->white[2], params->intensity);

    if (direct) {
        auto dbuffer = cmp->recoverSfc->buf32 + (bbox.yMin * cmp->recoverSfc->stride + bbox.xMin);
        auto sbuffer = cmp->image.buf32 + (bbox.yMin * cmp->image.stride + bbox.xMin);
        for (size_t y = 0; y < h; ++y) {
            auto dst = dbuffer;
            auto src = sbuffer;
            for (size_t x = 0; x < w; ++x, ++dst, ++src) {
                auto val = INTERPOLATE(white, black, luma((uint8_t*)src));
                if (params->intensity < 255) val = INTERPOLATE(val, *src, params->intensity);
                *dst = INTERPOLATE(val, *dst, MULTIPLY(opacity, A(*src)));
            }
            dbuffer += cmp->recoverSfc->stride;
            sbuffer += cmp->image.stride;
        }
        cmp->valid = true;  //no need the subsequent composition
    } else {
        auto dbuffer = cmp->image.buf32 + (bbox.yMin * cmp->image.stride + bbox.xMin);
        for (size_t y = 0; y < h; ++y) {
            auto dst = dbuffer;
            for (size_t x = 0; x < w; ++x, ++dst) {
                auto val = INTERPOLATE(white, black, luma((uint8_t*)dst));
                if (params->intensity < 255) val = INTERPOLATE(val, *dst, params->intensity);
                *dst = ALPHA_BLEND(val, MULTIPLY(opacity, A(*dst)));
            }
            dbuffer += cmp->image.stride;
        }
    }

    return true;
}

/************************************************************************/
/* Tritone Implementation                                              */
/************************************************************************/

static uint32_t _trintone(uint32_t s, uint32_t m, uint32_t h, int l)
{
    if (l < 128) {
        auto a = std::min(l * 2, 255);
        return ALPHA_BLEND(s, 255 - a) + ALPHA_BLEND(m, a);
    } else {
        auto a = 2 * std::max(0, l - 128);
        return ALPHA_BLEND(m, 255 - a) + ALPHA_BLEND(h, a);
    }
}

void effectTritoneUpdate(RenderEffectTritone* params)
{
    params->valid = (params->blender < 255);
}

// TODO: simd & openmp optimization?
bool effectTritone(SwCompositor* cmp, const RenderEffectTritone* params, bool direct)
{
    auto& bbox = cmp->bbox;
    auto w = size_t(bbox.xMax - bbox.xMin);
    auto h = size_t(bbox.yMax - bbox.yMin);
    auto shadow = cmp->recoverSfc->join(params->shadow[0], params->shadow[1], params->shadow[2], 255);
    auto midtone = cmp->recoverSfc->join(params->midtone[0], params->midtone[1], params->midtone[2], 255);
    auto highlight = cmp->recoverSfc->join(params->highlight[0], params->highlight[1], params->highlight[2], 255);
    auto opacity = cmp->opacity;
    auto luma = cmp->recoverSfc->alphas[2];  //luma function

    TVGLOG("SW_ENGINE", "Tritone region(%d, %d, %d, %d), param(%d %d %d, %d %d %d, %d %d %d, %d)", bbox.xMin, bbox.yMin, bbox.xMax, bbox.yMax, params->shadow[0], params->shadow[1], params->shadow[2], params->midtone[0], params->midtone[1], params->midtone[2], params->highlight[0], params->highlight[1], params->highlight[2], params->blender);

    if (direct) {
        auto dbuffer = cmp->recoverSfc->buf32 + (bbox.yMin * cmp->recoverSfc->stride + bbox.xMin);
        auto sbuffer = cmp->image.buf32 + (bbox.yMin * cmp->image.stride + bbox.xMin);
        for (size_t y = 0; y < h; ++y) {
            auto dst = dbuffer;
            auto src = sbuffer;
            if (params->blender == 0) {
                for (size_t x = 0; x < w; ++x, ++dst, ++src) {
                    *dst = INTERPOLATE(_trintone(shadow, midtone, highlight, luma((uint8_t*)src)), *dst, MULTIPLY(opacity, A(*src)));
                }
            } else {
                for (size_t x = 0; x < w; ++x, ++dst, ++src) {
                    *dst = INTERPOLATE(INTERPOLATE(*src, _trintone(shadow, midtone, highlight, luma((uint8_t*)src)), params->blender), *dst, MULTIPLY(opacity, A(*src)));
                }
            }
            dbuffer += cmp->recoverSfc->stride;
            sbuffer += cmp->image.stride;
        }
        cmp->valid = true;  //no need the subsequent composition
    } else {
        auto dbuffer = cmp->image.buf32 + (bbox.yMin * cmp->image.stride + bbox.xMin);
        for (size_t y = 0; y < h; ++y) {
            auto dst = dbuffer;
            if (params->blender == 0) {
                for (size_t x = 0; x < w; ++x, ++dst) {
                    *dst = ALPHA_BLEND(_trintone(shadow, midtone, highlight, luma((uint8_t*)dst)), MULTIPLY(A(*dst), opacity));
                }
            } else {
                for (size_t x = 0; x < w; ++x, ++dst) {
                    *dst = ALPHA_BLEND(INTERPOLATE(*dst, _trintone(shadow, midtone, highlight, luma((uint8_t*)dst)), params->blender), MULTIPLY(A(*dst), opacity));
                }
            }
            dbuffer += cmp->image.stride;
        }
    }

    return true;
}
