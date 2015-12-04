// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Spatial prediction using various filters
//
// Author: Urvang (urvang@google.com)

#include "./dsp.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// Helpful macro.

# define SANITY_CHECK(in, out)                                                 \
  assert(in != NULL);                                                          \
  assert(out != NULL);                                                         \
  assert(width > 0);                                                           \
  assert(height > 0);                                                          \
  assert(stride >= width);                                                     \
  assert(row >= 0 && num_rows > 0 && row + num_rows <= height);                \
  (void)height;  // Silence unused warning.

static WEBP_INLINE void PredictLine(const uint8_t* src, const uint8_t* pred,
                                    uint8_t* dst, int length, int inverse) {
  int i;
  if (inverse) {
    for (i = 0; i < length; ++i) dst[i] = src[i] + pred[i];
  } else {
    for (i = 0; i < length; ++i) dst[i] = src[i] - pred[i];
  }
}

//------------------------------------------------------------------------------
// Horizontal filter.

static WEBP_INLINE void DoHorizontalFilter(const uint8_t* in,
                                           int width, int height, int stride,
                                           int row, int num_rows,
                                           int inverse, uint8_t* out) {
  const uint8_t* preds;
  const size_t start_offset = row * stride;
  const int last_row = row + num_rows;
  SANITY_CHECK(in, out);
  in += start_offset;
  out += start_offset;
  preds = inverse ? out : in;

  if (row == 0) {
    // Leftmost pixel is the same as input for topmost scanline.
    out[0] = in[0];
    PredictLine(in + 1, preds, out + 1, width - 1, inverse);
    row = 1;
    preds += stride;
    in += stride;
    out += stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    // Leftmost pixel is predicted from above.
    PredictLine(in, preds - stride, out, 1, inverse);
    PredictLine(in + 1, preds, out + 1, width - 1, inverse);
    ++row;
    preds += stride;
    in += stride;
    out += stride;
  }
}

//------------------------------------------------------------------------------
// Vertical filter.

static WEBP_INLINE void DoVerticalFilter(const uint8_t* in,
                                         int width, int height, int stride,
                                         int row, int num_rows,
                                         int inverse, uint8_t* out) {
  const uint8_t* preds;
  const size_t start_offset = row * stride;
  const int last_row = row + num_rows;
  SANITY_CHECK(in, out);
  in += start_offset;
  out += start_offset;
  preds = inverse ? out : in;

  if (row == 0) {
    // Very first top-left pixel is copied.
    out[0] = in[0];
    // Rest of top scan-line is left-predicted.
    PredictLine(in + 1, preds, out + 1, width - 1, inverse);
    row = 1;
    in += stride;
    out += stride;
  } else {
    // We are starting from in-between. Make sure 'preds' points to prev row.
    preds -= stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    PredictLine(in, preds, out, width, inverse);
    ++row;
    preds += stride;
    in += stride;
    out += stride;
  }
}

//------------------------------------------------------------------------------
// Gradient filter.

static WEBP_INLINE int GradientPredictor(uint8_t a, uint8_t b, uint8_t c) {
  const int g = a + b - c;
  return ((g & ~0xff) == 0) ? g : (g < 0) ? 0 : 255;  // clip to 8bit
}

static WEBP_INLINE void DoGradientFilter(const uint8_t* in,
                                         int width, int height, int stride,
                                         int row, int num_rows,
                                         int inverse, uint8_t* out) {
  const uint8_t* preds;
  const size_t start_offset = row * stride;
  const int last_row = row + num_rows;
  SANITY_CHECK(in, out);
  in += start_offset;
  out += start_offset;
  preds = inverse ? out : in;

  // left prediction for top scan-line
  if (row == 0) {
    out[0] = in[0];
    PredictLine(in + 1, preds, out + 1, width - 1, inverse);
    row = 1;
    preds += stride;
    in += stride;
    out += stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    int w;
    // leftmost pixel: predict from above.
    PredictLine(in, preds - stride, out, 1, inverse);
    for (w = 1; w < width; ++w) {
      const int pred = GradientPredictor(preds[w - 1],
                                         preds[w - stride],
                                         preds[w - stride - 1]);
      out[w] = in[w] + (inverse ? pred : -pred);
    }
    ++row;
    preds += stride;
    in += stride;
    out += stride;
  }
}

#undef SANITY_CHECK

//------------------------------------------------------------------------------

static void HorizontalFilter(const uint8_t* data, int width, int height,
                             int stride, uint8_t* filtered_data) {
  DoHorizontalFilter(data, width, height, stride, 0, height, 0, filtered_data);
}

static void VerticalFilter(const uint8_t* data, int width, int height,
                           int stride, uint8_t* filtered_data) {
  DoVerticalFilter(data, width, height, stride, 0, height, 0, filtered_data);
}


static void GradientFilter(const uint8_t* data, int width, int height,
                           int stride, uint8_t* filtered_data) {
  DoGradientFilter(data, width, height, stride, 0, height, 0, filtered_data);
}


//------------------------------------------------------------------------------

static void VerticalUnfilter(int width, int height, int stride, int row,
                             int num_rows, uint8_t* data) {
  DoVerticalFilter(data, width, height, stride, row, num_rows, 1, data);
}

static void HorizontalUnfilter(int width, int height, int stride, int row,
                               int num_rows, uint8_t* data) {
  DoHorizontalFilter(data, width, height, stride, row, num_rows, 1, data);
}

static void GradientUnfilter(int width, int height, int stride, int row,
                             int num_rows, uint8_t* data) {
  DoGradientFilter(data, width, height, stride, row, num_rows, 1, data);
}

//------------------------------------------------------------------------------
// Init function

WebPFilterFunc WebPFilters[WEBP_FILTER_LAST];
WebPUnfilterFunc WebPUnfilters[WEBP_FILTER_LAST];

extern void VP8FiltersInitMIPSdspR2(void);
extern void VP8FiltersInitSSE2(void);

static volatile VP8CPUInfo filters_last_cpuinfo_used =
    (VP8CPUInfo)&filters_last_cpuinfo_used;

WEBP_TSAN_IGNORE_FUNCTION void VP8FiltersInit(void) {
  if (filters_last_cpuinfo_used == VP8GetCPUInfo) return;

  WebPUnfilters[WEBP_FILTER_NONE] = NULL;
  WebPUnfilters[WEBP_FILTER_HORIZONTAL] = HorizontalUnfilter;
  WebPUnfilters[WEBP_FILTER_VERTICAL] = VerticalUnfilter;
  WebPUnfilters[WEBP_FILTER_GRADIENT] = GradientUnfilter;

  WebPFilters[WEBP_FILTER_NONE] = NULL;
  WebPFilters[WEBP_FILTER_HORIZONTAL] = HorizontalFilter;
  WebPFilters[WEBP_FILTER_VERTICAL] = VerticalFilter;
  WebPFilters[WEBP_FILTER_GRADIENT] = GradientFilter;

  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_USE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8FiltersInitSSE2();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      VP8FiltersInitMIPSdspR2();
    }
#endif
  }
  filters_last_cpuinfo_used = VP8GetCPUInfo;
}
