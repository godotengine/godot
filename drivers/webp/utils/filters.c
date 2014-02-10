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

#include "./filters.h"
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

static void HorizontalFilter(const uint8_t* data, int width, int height,
                             int stride, uint8_t* filtered_data) {
  DoHorizontalFilter(data, width, height, stride, 0, height, 0, filtered_data);
}

static void HorizontalUnfilter(int width, int height, int stride, int row,
                               int num_rows, uint8_t* data) {
  DoHorizontalFilter(data, width, height, stride, row, num_rows, 1, data);
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

static void VerticalFilter(const uint8_t* data, int width, int height,
                           int stride, uint8_t* filtered_data) {
  DoVerticalFilter(data, width, height, stride, 0, height, 0, filtered_data);
}

static void VerticalUnfilter(int width, int height, int stride, int row,
                             int num_rows, uint8_t* data) {
  DoVerticalFilter(data, width, height, stride, row, num_rows, 1, data);
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

static void GradientFilter(const uint8_t* data, int width, int height,
                           int stride, uint8_t* filtered_data) {
  DoGradientFilter(data, width, height, stride, 0, height, 0, filtered_data);
}

static void GradientUnfilter(int width, int height, int stride, int row,
                             int num_rows, uint8_t* data) {
  DoGradientFilter(data, width, height, stride, row, num_rows, 1, data);
}

#undef SANITY_CHECK

// -----------------------------------------------------------------------------
// Quick estimate of a potentially interesting filter mode to try.

#define SMAX 16
#define SDIFF(a, b) (abs((a) - (b)) >> 4)   // Scoring diff, in [0..SMAX)

WEBP_FILTER_TYPE EstimateBestFilter(const uint8_t* data,
                                    int width, int height, int stride) {
  int i, j;
  int bins[WEBP_FILTER_LAST][SMAX];
  memset(bins, 0, sizeof(bins));

  // We only sample every other pixels. That's enough.
  for (j = 2; j < height - 1; j += 2) {
    const uint8_t* const p = data + j * stride;
    int mean = p[0];
    for (i = 2; i < width - 1; i += 2) {
      const int diff0 = SDIFF(p[i], mean);
      const int diff1 = SDIFF(p[i], p[i - 1]);
      const int diff2 = SDIFF(p[i], p[i - width]);
      const int grad_pred =
          GradientPredictor(p[i - 1], p[i - width], p[i - width - 1]);
      const int diff3 = SDIFF(p[i], grad_pred);
      bins[WEBP_FILTER_NONE][diff0] = 1;
      bins[WEBP_FILTER_HORIZONTAL][diff1] = 1;
      bins[WEBP_FILTER_VERTICAL][diff2] = 1;
      bins[WEBP_FILTER_GRADIENT][diff3] = 1;
      mean = (3 * mean + p[i] + 2) >> 2;
    }
  }
  {
    int filter;
    WEBP_FILTER_TYPE best_filter = WEBP_FILTER_NONE;
    int best_score = 0x7fffffff;
    for (filter = WEBP_FILTER_NONE; filter < WEBP_FILTER_LAST; ++filter) {
      int score = 0;
      for (i = 0; i < SMAX; ++i) {
        if (bins[filter][i] > 0) {
          score += i;
        }
      }
      if (score < best_score) {
        best_score = score;
        best_filter = (WEBP_FILTER_TYPE)filter;
      }
    }
    return best_filter;
  }
}

#undef SMAX
#undef SDIFF

//------------------------------------------------------------------------------

const WebPFilterFunc WebPFilters[WEBP_FILTER_LAST] = {
  NULL,              // WEBP_FILTER_NONE
  HorizontalFilter,  // WEBP_FILTER_HORIZONTAL
  VerticalFilter,    // WEBP_FILTER_VERTICAL
  GradientFilter     // WEBP_FILTER_GRADIENT
};

const WebPUnfilterFunc WebPUnfilters[WEBP_FILTER_LAST] = {
  NULL,                // WEBP_FILTER_NONE
  HorizontalUnfilter,  // WEBP_FILTER_HORIZONTAL
  VerticalUnfilter,    // WEBP_FILTER_VERTICAL
  GradientUnfilter     // WEBP_FILTER_GRADIENT
};

//------------------------------------------------------------------------------

