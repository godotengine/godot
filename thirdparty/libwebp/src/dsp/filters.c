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

#include "src/dsp/dsp.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

//------------------------------------------------------------------------------
// Helpful macro.

# define SANITY_CHECK(in, out)                                                 \
  assert((in) != NULL);                                                        \
  assert((out) != NULL);                                                       \
  assert(width > 0);                                                           \
  assert(height > 0);                                                          \
  assert(stride >= width);                                                     \
  assert(row >= 0 && num_rows > 0 && row + num_rows <= height);                \
  (void)height;  // Silence unused warning.

#if !WEBP_NEON_OMIT_C_CODE
static WEBP_INLINE void PredictLine_C(const uint8_t* src, const uint8_t* pred,
                                      uint8_t* dst, int length, int inverse) {
  int i;
  if (inverse) {
    for (i = 0; i < length; ++i) dst[i] = (uint8_t)(src[i] + pred[i]);
  } else {
    for (i = 0; i < length; ++i) dst[i] = (uint8_t)(src[i] - pred[i]);
  }
}

//------------------------------------------------------------------------------
// Horizontal filter.

static WEBP_INLINE void DoHorizontalFilter_C(const uint8_t* in,
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
    PredictLine_C(in + 1, preds, out + 1, width - 1, inverse);
    row = 1;
    preds += stride;
    in += stride;
    out += stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    // Leftmost pixel is predicted from above.
    PredictLine_C(in, preds - stride, out, 1, inverse);
    PredictLine_C(in + 1, preds, out + 1, width - 1, inverse);
    ++row;
    preds += stride;
    in += stride;
    out += stride;
  }
}

//------------------------------------------------------------------------------
// Vertical filter.

static WEBP_INLINE void DoVerticalFilter_C(const uint8_t* in,
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
    PredictLine_C(in + 1, preds, out + 1, width - 1, inverse);
    row = 1;
    in += stride;
    out += stride;
  } else {
    // We are starting from in-between. Make sure 'preds' points to prev row.
    preds -= stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    PredictLine_C(in, preds, out, width, inverse);
    ++row;
    preds += stride;
    in += stride;
    out += stride;
  }
}
#endif  // !WEBP_NEON_OMIT_C_CODE

//------------------------------------------------------------------------------
// Gradient filter.

static WEBP_INLINE int GradientPredictor_C(uint8_t a, uint8_t b, uint8_t c) {
  const int g = a + b - c;
  return ((g & ~0xff) == 0) ? g : (g < 0) ? 0 : 255;  // clip to 8bit
}

#if !WEBP_NEON_OMIT_C_CODE
static WEBP_INLINE void DoGradientFilter_C(const uint8_t* in,
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
    PredictLine_C(in + 1, preds, out + 1, width - 1, inverse);
    row = 1;
    preds += stride;
    in += stride;
    out += stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    int w;
    // leftmost pixel: predict from above.
    PredictLine_C(in, preds - stride, out, 1, inverse);
    for (w = 1; w < width; ++w) {
      const int pred = GradientPredictor_C(preds[w - 1],
                                           preds[w - stride],
                                           preds[w - stride - 1]);
      out[w] = (uint8_t)(in[w] + (inverse ? pred : -pred));
    }
    ++row;
    preds += stride;
    in += stride;
    out += stride;
  }
}
#endif  // !WEBP_NEON_OMIT_C_CODE

#undef SANITY_CHECK

//------------------------------------------------------------------------------

#if !WEBP_NEON_OMIT_C_CODE
static void HorizontalFilter_C(const uint8_t* data, int width, int height,
                               int stride, uint8_t* filtered_data) {
  DoHorizontalFilter_C(data, width, height, stride, 0, height, 0,
                       filtered_data);
}

static void VerticalFilter_C(const uint8_t* data, int width, int height,
                             int stride, uint8_t* filtered_data) {
  DoVerticalFilter_C(data, width, height, stride, 0, height, 0, filtered_data);
}

static void GradientFilter_C(const uint8_t* data, int width, int height,
                             int stride, uint8_t* filtered_data) {
  DoGradientFilter_C(data, width, height, stride, 0, height, 0, filtered_data);
}
#endif  // !WEBP_NEON_OMIT_C_CODE

//------------------------------------------------------------------------------

static void HorizontalUnfilter_C(const uint8_t* prev, const uint8_t* in,
                                 uint8_t* out, int width) {
  uint8_t pred = (prev == NULL) ? 0 : prev[0];
  int i;
  for (i = 0; i < width; ++i) {
    out[i] = (uint8_t)(pred + in[i]);
    pred = out[i];
  }
}

#if !WEBP_NEON_OMIT_C_CODE
static void VerticalUnfilter_C(const uint8_t* prev, const uint8_t* in,
                               uint8_t* out, int width) {
  if (prev == NULL) {
    HorizontalUnfilter_C(NULL, in, out, width);
  } else {
    int i;
    for (i = 0; i < width; ++i) out[i] = (uint8_t)(prev[i] + in[i]);
  }
}
#endif  // !WEBP_NEON_OMIT_C_CODE

static void GradientUnfilter_C(const uint8_t* prev, const uint8_t* in,
                               uint8_t* out, int width) {
  if (prev == NULL) {
    HorizontalUnfilter_C(NULL, in, out, width);
  } else {
    uint8_t top = prev[0], top_left = top, left = top;
    int i;
    for (i = 0; i < width; ++i) {
      top = prev[i];  // need to read this first, in case prev==out
      left = (uint8_t)(in[i] + GradientPredictor_C(left, top, top_left));
      top_left = top;
      out[i] = left;
    }
  }
}

//------------------------------------------------------------------------------
// Init function

WebPFilterFunc WebPFilters[WEBP_FILTER_LAST];
WebPUnfilterFunc WebPUnfilters[WEBP_FILTER_LAST];

extern VP8CPUInfo VP8GetCPUInfo;
extern void VP8FiltersInitMIPSdspR2(void);
extern void VP8FiltersInitMSA(void);
extern void VP8FiltersInitNEON(void);
extern void VP8FiltersInitSSE2(void);

WEBP_DSP_INIT_FUNC(VP8FiltersInit) {
  WebPUnfilters[WEBP_FILTER_NONE] = NULL;
#if !WEBP_NEON_OMIT_C_CODE
  WebPUnfilters[WEBP_FILTER_HORIZONTAL] = HorizontalUnfilter_C;
  WebPUnfilters[WEBP_FILTER_VERTICAL] = VerticalUnfilter_C;
#endif
  WebPUnfilters[WEBP_FILTER_GRADIENT] = GradientUnfilter_C;

  WebPFilters[WEBP_FILTER_NONE] = NULL;
#if !WEBP_NEON_OMIT_C_CODE
  WebPFilters[WEBP_FILTER_HORIZONTAL] = HorizontalFilter_C;
  WebPFilters[WEBP_FILTER_VERTICAL] = VerticalFilter_C;
  WebPFilters[WEBP_FILTER_GRADIENT] = GradientFilter_C;
#endif

  if (VP8GetCPUInfo != NULL) {
#if defined(WEBP_HAVE_SSE2)
    if (VP8GetCPUInfo(kSSE2)) {
      VP8FiltersInitSSE2();
    }
#endif
#if defined(WEBP_USE_MIPS_DSP_R2)
    if (VP8GetCPUInfo(kMIPSdspR2)) {
      VP8FiltersInitMIPSdspR2();
    }
#endif
#if defined(WEBP_USE_MSA)
    if (VP8GetCPUInfo(kMSA)) {
      VP8FiltersInitMSA();
    }
#endif
  }

#if defined(WEBP_HAVE_NEON)
  if (WEBP_NEON_OMIT_C_CODE ||
      (VP8GetCPUInfo != NULL && VP8GetCPUInfo(kNEON))) {
    VP8FiltersInitNEON();
  }
#endif

  assert(WebPUnfilters[WEBP_FILTER_HORIZONTAL] != NULL);
  assert(WebPUnfilters[WEBP_FILTER_VERTICAL] != NULL);
  assert(WebPUnfilters[WEBP_FILTER_GRADIENT] != NULL);
  assert(WebPFilters[WEBP_FILTER_HORIZONTAL] != NULL);
  assert(WebPFilters[WEBP_FILTER_VERTICAL] != NULL);
  assert(WebPFilters[WEBP_FILTER_GRADIENT] != NULL);
}
