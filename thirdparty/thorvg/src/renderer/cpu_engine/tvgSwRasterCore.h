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

#ifndef _TVG_SW_RASTER_CORE_H_
#define _TVG_SW_RASTER_CORE_H_

#include "tvgSwCommon.h"

uint32_t cRasterDownScaler(const uint32_t*, uint32_t, uint32_t, uint32_t, float, float, int32_t, int32_t, int32_t);
void cRasterUnpremultiply(uint32_t*, uint32_t);
void cRasterPremultiply(uint32_t*, uint32_t);
void cRasterTranslucentPixels(uint32_t*, uint32_t*, uint32_t, uint8_t);
void cRasterSolidPixels(uint32_t*, uint32_t*, uint32_t, uint8_t);
void cRasterSolidPixel32(uint32_t*, uint32_t, uint32_t, int32_t);
bool cRasterTranslucentRle(SwSurface*, const SwRle*, const RenderRegion&, const RenderColor&);
bool cRasterTranslucentRect(SwSurface*, const RenderRegion&, const RenderColor&);
bool cRasterABGRtoARGB(RenderSurface*);
void cRasterGaussianFilterRow(uint8_t*, const uint8_t*, int32_t, int32_t, float);

#if defined(THORVG_AVX_SUPPORT) || defined(THORVG_NEON_SUPPORT)
    extern uint32_t (*rasterDownScaler)(const uint32_t*, uint32_t, uint32_t, uint32_t, float, float, int32_t, int32_t, int32_t);
    extern void (*rasterUnpremultiply)(uint32_t*, uint32_t);
    extern void (*rasterPremultiply)(uint32_t*, uint32_t);
    extern void (*rasterTranslucentPixels)(uint32_t*, uint32_t*, uint32_t, uint8_t);
    extern void (*rasterSolidPixels)(uint32_t*, uint32_t*, uint32_t, uint8_t);
    extern void (*rasterSolidPixel32)(uint32_t*, uint32_t, uint32_t, int32_t);
    extern bool (*rasterTranslucentRle)(SwSurface*, const SwRle*, const RenderRegion&, const RenderColor&);
    extern bool (*rasterTranslucentRect)(SwSurface*, const RenderRegion&, const RenderColor&);
    extern bool (*rasterABGRtoARGB)(RenderSurface*);
    extern void (*rasterGaussianFilterRow)(uint8_t*, const uint8_t*, int32_t, int32_t, float);
    extern void (*rasterGaussianXYFlipBlock)(const uint32_t*, uint32_t*, int32_t);
#else
    // Resolve scalar calls directly when SIMD dispatch is not built.
    #define rasterDownScaler cRasterDownScaler
    #define rasterUnpremultiply cRasterUnpremultiply
    #define rasterPremultiply cRasterPremultiply
    #define rasterTranslucentPixels cRasterTranslucentPixels
    #define rasterSolidPixels cRasterSolidPixels
    #define rasterSolidPixel32 cRasterSolidPixel32
    #define rasterTranslucentRle cRasterTranslucentRle
    #define rasterTranslucentRect cRasterTranslucentRect
    #define rasterABGRtoARGB cRasterABGRtoARGB
    #define rasterGaussianFilterRow cRasterGaussianFilterRow
#endif

#endif // _TVG_SW_RASTER_CORE_H_
