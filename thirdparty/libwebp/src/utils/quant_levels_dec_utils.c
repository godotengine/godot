// Copyright 2013 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Implement gradient smoothing: we replace a current alpha value by its
// surrounding average if it's close enough (that is: the change will be less
// than the minimum distance between two quantized level).
// We use sliding window for computing the 2d moving average.
//
// Author: Skal (pascal.massimino@gmail.com)

#include "src/utils/quant_levels_dec_utils.h"

#include <string.h>   // for memset

#include "src/utils/utils.h"
#include "src/webp/types.h"

// #define USE_DITHERING   // uncomment to enable ordered dithering (not vital)

#define FIX 16     // fix-point precision for averaging
#define LFIX 2     // extra precision for look-up table
#define LUT_SIZE ((1 << (8 + LFIX)) - 1)  // look-up table size

#if defined(USE_DITHERING)

#define DFIX 4           // extra precision for ordered dithering
#define DSIZE 4          // dithering size (must be a power of two)
// cf. https://en.wikipedia.org/wiki/Ordered_dithering
static const uint8_t kOrderedDither[DSIZE][DSIZE] = {
  {  0,  8,  2, 10 },     // coefficients are in DFIX fixed-point precision
  { 12,  4, 14,  6 },
  {  3, 11,  1,  9 },
  { 15,  7, 13,  5 }
};

#else
#define DFIX 0
#endif

typedef struct {
  int width, height;   // dimension
  int stride;          // stride in bytes
  int row;             // current input row being processed
  uint8_t* src;        // input pointer
  uint8_t* dst;        // output pointer

  int radius;          // filter radius (=delay)
  int scale;           // normalization factor, in FIX bits precision

  void* mem;           // all memory

  // various scratch buffers
  uint16_t* start;
  uint16_t* cur;
  uint16_t* end;
  uint16_t* top;
  uint16_t* average;

  // input levels distribution
  int num_levels;       // number of quantized levels
  int min, max;         // min and max level values
  int min_level_dist;   // smallest distance between two consecutive levels

  int16_t* correction;  // size = 1 + 2*LUT_SIZE  -> ~4k memory
} SmoothParams;

//------------------------------------------------------------------------------

#define CLIP_8b_MASK (int)(~0U << (8 + DFIX))
static WEBP_INLINE uint8_t clip_8b(int v) {
  return (!(v & CLIP_8b_MASK)) ? (uint8_t)(v >> DFIX) : (v < 0) ? 0u : 255u;
}
#undef CLIP_8b_MASK

// vertical accumulation
static void VFilter(SmoothParams* const p) {
  const uint8_t* src = p->src;
  const int w = p->width;
  uint16_t* const cur = p->cur;
  const uint16_t* const top = p->top;
  uint16_t* const out = p->end;
  uint16_t sum = 0;               // all arithmetic is modulo 16bit
  int x;

  for (x = 0; x < w; ++x) {
    uint16_t new_value;
    sum += src[x];
    new_value = top[x] + sum;
    out[x] = new_value - cur[x];  // vertical sum of 'r' pixels.
    cur[x] = new_value;
  }
  // move input pointers one row down
  p->top = p->cur;
  p->cur += w;
  if (p->cur == p->end) p->cur = p->start;  // roll-over
  // We replicate edges, as it's somewhat easier as a boundary condition.
  // That's why we don't update the 'src' pointer on top/bottom area:
  if (p->row >= 0 && p->row < p->height - 1) {
    p->src += p->stride;
  }
}

// horizontal accumulation. We use mirror replication of missing pixels, as it's
// a little easier to implement (surprisingly).
static void HFilter(SmoothParams* const p) {
  const uint16_t* const in = p->end;
  uint16_t* const out = p->average;
  const uint32_t scale = p->scale;
  const int w = p->width;
  const int r = p->radius;

  int x;
  for (x = 0; x <= r; ++x) {   // left mirroring
    const uint16_t delta = in[x + r - 1] + in[r - x];
    out[x] = (delta * scale) >> FIX;
  }
  for (; x < w - r; ++x) {     // bulk middle run
    const uint16_t delta = in[x + r] - in[x - r - 1];
    out[x] = (delta * scale) >> FIX;
  }
  for (; x < w; ++x) {         // right mirroring
    const uint16_t delta =
        2 * in[w - 1] - in[2 * w - 2 - r - x] - in[x - r - 1];
    out[x] = (delta * scale) >> FIX;
  }
}

// emit one filtered output row
static void ApplyFilter(SmoothParams* const p) {
  const uint16_t* const average = p->average;
  const int w = p->width;
  const int16_t* const correction = p->correction;
#if defined(USE_DITHERING)
  const uint8_t* const dither = kOrderedDither[p->row % DSIZE];
#endif
  uint8_t* const dst = p->dst;
  int x;
  for (x = 0; x < w; ++x) {
    const int v = dst[x];
    if (v < p->max && v > p->min) {
      const int c = (v << DFIX) + correction[average[x] - (v << LFIX)];
#if defined(USE_DITHERING)
      dst[x] = clip_8b(c + dither[x % DSIZE]);
#else
      dst[x] = clip_8b(c);
#endif
    }
  }
  p->dst += p->stride;  // advance output pointer
}

//------------------------------------------------------------------------------
// Initialize correction table

static void InitCorrectionLUT(int16_t* const lut, int min_dist) {
  // The correction curve is:
  //   f(x) = x for x <= threshold2
  //   f(x) = 0 for x >= threshold1
  // and a linear interpolation for range x=[threshold2, threshold1]
  // (along with f(-x) = -f(x) symmetry).
  // Note that: threshold2 = 3/4 * threshold1
  const int threshold1 = min_dist << LFIX;
  const int threshold2 = (3 * threshold1) >> 2;
  const int max_threshold = threshold2 << DFIX;
  const int delta = threshold1 - threshold2;
  int i;
  for (i = 1; i <= LUT_SIZE; ++i) {
    int c = (i <= threshold2) ? (i << DFIX)
          : (i < threshold1) ? max_threshold * (threshold1 - i) / delta
          : 0;
    c >>= LFIX;
    lut[+i] = +c;
    lut[-i] = -c;
  }
  lut[0] = 0;
}

static void CountLevels(SmoothParams* const p) {
  int i, j, last_level;
  uint8_t used_levels[256] = { 0 };
  const uint8_t* data = p->src;
  p->min = 255;
  p->max = 0;
  for (j = 0; j < p->height; ++j) {
    for (i = 0; i < p->width; ++i) {
      const int v = data[i];
      if (v < p->min) p->min = v;
      if (v > p->max) p->max = v;
      used_levels[v] = 1;
    }
    data += p->stride;
  }
  // Compute the mininum distance between two non-zero levels.
  p->min_level_dist = p->max - p->min;
  last_level = -1;
  for (i = 0; i < 256; ++i) {
    if (used_levels[i]) {
      ++p->num_levels;
      if (last_level >= 0) {
        const int level_dist = i - last_level;
        if (level_dist < p->min_level_dist) {
          p->min_level_dist = level_dist;
        }
      }
      last_level = i;
    }
  }
}

// Initialize all params.
static int InitParams(uint8_t* const data, int width, int height, int stride,
                      int radius, SmoothParams* const p) {
  const int R = 2 * radius + 1;  // total size of the kernel

  const size_t size_scratch_m = (R + 1) * width * sizeof(*p->start);
  const size_t size_m =  width * sizeof(*p->average);
  const size_t size_lut = (1 + 2 * LUT_SIZE) * sizeof(*p->correction);
  const size_t total_size = size_scratch_m + size_m + size_lut;
  uint8_t* mem = (uint8_t*)WebPSafeMalloc(1U, total_size);

  if (mem == NULL) return 0;
  p->mem = (void*)mem;

  p->start = (uint16_t*)mem;
  p->cur = p->start;
  p->end = p->start + R * width;
  p->top = p->end - width;
  memset(p->top, 0, width * sizeof(*p->top));
  mem += size_scratch_m;

  p->average = (uint16_t*)mem;
  mem += size_m;

  p->width = width;
  p->height = height;
  p->stride = stride;
  p->src = data;
  p->dst = data;
  p->radius = radius;
  p->scale = (1 << (FIX + LFIX)) / (R * R);  // normalization constant
  p->row = -radius;

  // analyze the input distribution so we can best-fit the threshold
  CountLevels(p);

  // correction table
  p->correction = ((int16_t*)mem) + LUT_SIZE;
  InitCorrectionLUT(p->correction, p->min_level_dist);

  return 1;
}

static void CleanupParams(SmoothParams* const p) {
  WebPSafeFree(p->mem);
}

int WebPDequantizeLevels(uint8_t* const data, int width, int height, int stride,
                         int strength) {
  int radius = 4 * strength / 100;

  if (strength < 0 || strength > 100) return 0;
  if (data == NULL || width <= 0 || height <= 0) return 0;  // bad params

  // limit the filter size to not exceed the image dimensions
  if (2 * radius + 1 > width) radius = (width - 1) >> 1;
  if (2 * radius + 1 > height) radius = (height - 1) >> 1;

  if (radius > 0) {
    SmoothParams p;
    memset(&p, 0, sizeof(p));
    if (!InitParams(data, width, height, stride, radius, &p)) return 0;
    if (p.num_levels > 2) {
      for (; p.row < p.height; ++p.row) {
        VFilter(&p);  // accumulate average of input
        // Need to wait few rows in order to prime the filter,
        // before emitting some output.
        if (p.row >= p.radius) {
          HFilter(&p);
          ApplyFilter(&p);
        }
      }
    }
    CleanupParams(&p);
  }
  return 1;
}
