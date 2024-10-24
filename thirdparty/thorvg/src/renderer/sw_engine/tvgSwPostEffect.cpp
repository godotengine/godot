/*
 * Copyright (c) 2024 the ThorVG project. All rights reserved.

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

/************************************************************************/
/* Gaussian Filter Implementation                                       */
/************************************************************************/

struct SwGaussianBlur
{
    static constexpr int MAX_LEVEL = 3;
    int level;
    int kernel[MAX_LEVEL];
};


static void _gaussianExtendRegion(RenderRegion& region, int extra, int8_t direction)
{
    //bbox region expansion for feathering
    if (direction == 0 || direction == 1) {
        region.x = -extra;
        region.y = -extra;
    }

    if (direction == 0 || direction == 2) {
        region.w = extra * 2;
        region.h = extra * 2;
    }
}


static int _gaussianRemap(int end, int idx, int border)
{
    //wrap
    if (border == 1) return idx % end;

    //duplicate
    if (idx < 0) return 0;
    else if (idx >= end) return end - 1;
    return idx;
}


//TODO: SIMD OPTIMIZATION?
static void _gaussianBlur(uint8_t* src, uint8_t* dst, int32_t stride, int32_t w, int32_t h, const SwBBox& bbox, int32_t dimension, int border, bool flipped)
{
    if (flipped) {
        src += ((bbox.min.x * stride) + bbox.min.y) << 2;
        dst += ((bbox.min.x * stride) + bbox.min.y) << 2;
    } else {
        src += ((bbox.min.y * stride) + bbox.min.x) << 2;
        dst += ((bbox.min.y * stride) + bbox.min.x) << 2;
    }

    auto iarr = 1.0f / (dimension + dimension + 1);

    #pragma omp parallel for
    for (int x = 0; x < h; x++) {
        auto p = x * stride;
        auto i = p * 4;                 //current index
        auto l = -(dimension + 1);      //left index
        auto r = dimension;             //right index
        int acc[4] = {0, 0, 0, 0};      //sliding accumulator

        //initial acucmulation
        for (int x2 = l; x2 < r; ++x2) {
            auto id = (_gaussianRemap(w, x2, border) + p) * 4;
            acc[0] += src[id++];
            acc[1] += src[id++];
            acc[2] += src[id++];
            acc[3] += src[id];
        }
        //perform filtering
        for (int x2 = 0; x2 < w; ++x2, ++r, ++l) {
            auto rid = (_gaussianRemap(w, r, border) + p) * 4;
            auto lid = (_gaussianRemap(w, l, border) + p) * 4;
            acc[0] += src[rid++] - src[lid++];
            acc[1] += src[rid++] - src[lid++];
            acc[2] += src[rid++] - src[lid++];
            acc[3] += src[rid] - src[lid];
            dst[i++] = static_cast<uint8_t>(acc[0] * iarr + 0.5f);
            dst[i++] = static_cast<uint8_t>(acc[1] * iarr + 0.5f);
            dst[i++] = static_cast<uint8_t>(acc[2] * iarr + 0.5f);
            dst[i++] = static_cast<uint8_t>(acc[3] * iarr + 0.5f);
        }
    }
}


static int _gaussianInit(int* kernel, float sigma, int level)
{
    const auto MAX_LEVEL = SwGaussianBlur::MAX_LEVEL;

    //compute the kernel
    auto wl = (int) sqrt((12 * sigma / MAX_LEVEL) + 1);
    if (wl % 2 == 0) --wl;
    auto wu = wl + 2;
    auto mi = (12 * sigma - MAX_LEVEL * wl * wl - 4 * MAX_LEVEL * wl - 3 * MAX_LEVEL) / (-4 * wl - 4);
    auto m = int(mi + 0.5f);
    auto extends = 0;

    for (int i = 0; i < level; i++) {
        kernel[i] = ((i < m ? wl : wu) - 1) / 2;
        extends += kernel[i];
    }

    return extends;
}


bool effectGaussianPrepare(RenderEffectGaussian* params)
{
    auto data = (SwGaussianBlur*)malloc(sizeof(SwGaussianBlur));

    //compute box kernel sizes
    data->level = int(SwGaussianBlur::MAX_LEVEL * ((params->quality - 1) * 0.01f)) + 1;
    auto extends = _gaussianInit(data->kernel, params->sigma * params->sigma, data->level);

    //skip, if the parameters are invalid.
    if (extends == 0) {
        params->invalid = true;
        free(data);
        return false;
    }

    _gaussianExtendRegion(params->extend, extends, params->direction);

    params->rd = data;

    return true;
}


/* It is best to take advantage of the Gaussian blur’s separable property
   by dividing the process into two passes. horizontal and vertical.
   We can expect fewer calculations. */
bool effectGaussianBlur(SwImage& image, SwImage& buffer, const SwBBox& bbox, const RenderEffectGaussian* params)
{
    if (params->invalid) return false;

    if (image.channelSize != sizeof(uint32_t)) {
        TVGERR("SW_ENGINE", "Not supported grayscale Gaussian Blur!");
        return false;
    }

    auto data = static_cast<SwGaussianBlur*>(params->rd);
    auto w = (bbox.max.x - bbox.min.x);
    auto h = (bbox.max.y - bbox.min.y);
    auto stride = image.stride;
    auto front = image.buf8;
    auto back = buffer.buf8;
    auto swapped = false;

    //fine-tuning for low-quality (experimental)
    auto threshold = (std::min(w, h) < 300) ? 2 : 1;

    TVGLOG("SW_ENGINE", "GaussianFilter region(%ld, %ld, %ld, %ld) params(%f %d %d), level(%d)", bbox.min.x, bbox.min.y, bbox.max.x, bbox.max.y, params->sigma, params->direction, params->border, data->level);

    //horizontal
    if (params->direction == 0 || params->direction == 1) {
        for (int i = 0; i < data->level; ++i) {
            auto k = data->kernel[i] / threshold;
            if (k == 0) continue;
            _gaussianBlur(front, back, stride, w, h, bbox, k, params->border, false);
            std::swap(front, back);
            swapped = !swapped;
        }
    }

    //vertical. x/y flipping and horionztal access is pretty compatible with the memory architecture.
    if (params->direction == 0 || params->direction == 2) {
        rasterXYFlip(reinterpret_cast<uint32_t*>(front), reinterpret_cast<uint32_t*>(back), stride, w, h, bbox, false);
        std::swap(front, back);

        for (int i = 0; i < data->level; ++i) {
            auto k = data->kernel[i] / threshold;
            if (k == 0) continue;
            _gaussianBlur(front, back, stride, h, w, bbox, k, params->border, true);
            std::swap(front, back);
            swapped = !swapped;
        }

        rasterXYFlip(reinterpret_cast<uint32_t*>(front), reinterpret_cast<uint32_t*>(back), stride, h, w, bbox, true);
        std::swap(front, back);
    }

    if (swapped) std::swap(image.buf8, buffer.buf8);

    return true;
}
