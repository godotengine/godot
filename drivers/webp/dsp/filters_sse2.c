// Copyright 2015 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// SSE2 variant of alpha filters
//
// Author: Skal (pascal.massimino@gmail.com)

#include "./dsp.h"

#if defined(WEBP_USE_SSE2)

#include <assert.h>
#include <emmintrin.h>
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

static void PredictLineTop(const uint8_t* src, const uint8_t* pred,
                           uint8_t* dst, int length, int inverse) {
  int i;
  const int max_pos = length & ~31;
  assert(length >= 0);
  if (inverse) {
    for (i = 0; i < max_pos; i += 32) {
      const __m128i A0 = _mm_loadu_si128((const __m128i*)&src[i +  0]);
      const __m128i A1 = _mm_loadu_si128((const __m128i*)&src[i + 16]);
      const __m128i B0 = _mm_loadu_si128((const __m128i*)&pred[i +  0]);
      const __m128i B1 = _mm_loadu_si128((const __m128i*)&pred[i + 16]);
      const __m128i C0 = _mm_add_epi8(A0, B0);
      const __m128i C1 = _mm_add_epi8(A1, B1);
      _mm_storeu_si128((__m128i*)&dst[i +  0], C0);
      _mm_storeu_si128((__m128i*)&dst[i + 16], C1);
    }
    for (; i < length; ++i) dst[i] = src[i] + pred[i];
  } else {
    for (i = 0; i < max_pos; i += 32) {
      const __m128i A0 = _mm_loadu_si128((const __m128i*)&src[i +  0]);
      const __m128i A1 = _mm_loadu_si128((const __m128i*)&src[i + 16]);
      const __m128i B0 = _mm_loadu_si128((const __m128i*)&pred[i +  0]);
      const __m128i B1 = _mm_loadu_si128((const __m128i*)&pred[i + 16]);
      const __m128i C0 = _mm_sub_epi8(A0, B0);
      const __m128i C1 = _mm_sub_epi8(A1, B1);
      _mm_storeu_si128((__m128i*)&dst[i +  0], C0);
      _mm_storeu_si128((__m128i*)&dst[i + 16], C1);
    }
    for (; i < length; ++i) dst[i] = src[i] - pred[i];
  }
}

// Special case for left-based prediction (when preds==dst-1 or preds==src-1).
static void PredictLineLeft(const uint8_t* src, uint8_t* dst, int length,
                            int inverse) {
  int i;
  if (length <= 0) return;
  if (inverse) {
    const int max_pos = length & ~7;
    __m128i last = _mm_set_epi32(0, 0, 0, dst[-1]);
    for (i = 0; i < max_pos; i += 8) {
      const __m128i A0 = _mm_loadl_epi64((const __m128i*)(src + i));
      const __m128i A1 = _mm_add_epi8(A0, last);
      const __m128i A2 = _mm_slli_si128(A1, 1);
      const __m128i A3 = _mm_add_epi8(A1, A2);
      const __m128i A4 = _mm_slli_si128(A3, 2);
      const __m128i A5 = _mm_add_epi8(A3, A4);
      const __m128i A6 = _mm_slli_si128(A5, 4);
      const __m128i A7 = _mm_add_epi8(A5, A6);
      _mm_storel_epi64((__m128i*)(dst + i), A7);
      last = _mm_srli_epi64(A7, 56);
    }
    for (; i < length; ++i) dst[i] = src[i] + dst[i - 1];
  } else {
    const int max_pos = length & ~31;
    for (i = 0; i < max_pos; i += 32) {
      const __m128i A0 = _mm_loadu_si128((const __m128i*)(src + i +  0    ));
      const __m128i B0 = _mm_loadu_si128((const __m128i*)(src + i +  0 - 1));
      const __m128i A1 = _mm_loadu_si128((const __m128i*)(src + i + 16    ));
      const __m128i B1 = _mm_loadu_si128((const __m128i*)(src + i + 16 - 1));
      const __m128i C0 = _mm_sub_epi8(A0, B0);
      const __m128i C1 = _mm_sub_epi8(A1, B1);
      _mm_storeu_si128((__m128i*)(dst + i +  0), C0);
      _mm_storeu_si128((__m128i*)(dst + i + 16), C1);
    }
    for (; i < length; ++i) dst[i] = src[i] - src[i - 1];
  }
}

static void PredictLineC(const uint8_t* src, const uint8_t* pred,
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
    PredictLineLeft(in + 1, out + 1, width - 1, inverse);
    row = 1;
    preds += stride;
    in += stride;
    out += stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    // Leftmost pixel is predicted from above.
    PredictLineC(in, preds - stride, out, 1, inverse);
    PredictLineLeft(in + 1, out + 1, width - 1, inverse);
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
    PredictLineLeft(in + 1, out + 1, width - 1, inverse);
    row = 1;
    in += stride;
    out += stride;
  } else {
    // We are starting from in-between. Make sure 'preds' points to prev row.
    preds -= stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    PredictLineTop(in, preds, out, width, inverse);
    ++row;
    preds += stride;
    in += stride;
    out += stride;
  }
}

//------------------------------------------------------------------------------
// Gradient filter.

static WEBP_INLINE int GradientPredictorC(uint8_t a, uint8_t b, uint8_t c) {
  const int g = a + b - c;
  return ((g & ~0xff) == 0) ? g : (g < 0) ? 0 : 255;  // clip to 8bit
}

static void GradientPredictDirect(const uint8_t* const row,
                                  const uint8_t* const top,
                                  uint8_t* const out, int length) {
  const int max_pos = length & ~7;
  int i;
  const __m128i zero = _mm_setzero_si128();
  for (i = 0; i < max_pos; i += 8) {
    const __m128i A0 = _mm_loadl_epi64((const __m128i*)&row[i - 1]);
    const __m128i B0 = _mm_loadl_epi64((const __m128i*)&top[i]);
    const __m128i C0 = _mm_loadl_epi64((const __m128i*)&top[i - 1]);
    const __m128i D = _mm_loadl_epi64((const __m128i*)&row[i]);
    const __m128i A1 = _mm_unpacklo_epi8(A0, zero);
    const __m128i B1 = _mm_unpacklo_epi8(B0, zero);
    const __m128i C1 = _mm_unpacklo_epi8(C0, zero);
    const __m128i E = _mm_add_epi16(A1, B1);
    const __m128i F = _mm_sub_epi16(E, C1);
    const __m128i G = _mm_packus_epi16(F, zero);
    const __m128i H = _mm_sub_epi8(D, G);
    _mm_storel_epi64((__m128i*)(out + i), H);
  }
  for (; i < length; ++i) {
    out[i] = row[i] - GradientPredictorC(row[i - 1], top[i], top[i - 1]);
  }
}

static void GradientPredictInverse(const uint8_t* const in,
                                   const uint8_t* const top,
                                   uint8_t* const row, int length) {
  if (length > 0) {
    int i;
    const int max_pos = length & ~7;
    const __m128i zero = _mm_setzero_si128();
    __m128i A = _mm_set_epi32(0, 0, 0, row[-1]);   // left sample
    for (i = 0; i < max_pos; i += 8) {
      const __m128i tmp0 = _mm_loadl_epi64((const __m128i*)&top[i]);
      const __m128i tmp1 = _mm_loadl_epi64((const __m128i*)&top[i - 1]);
      const __m128i B = _mm_unpacklo_epi8(tmp0, zero);
      const __m128i C = _mm_unpacklo_epi8(tmp1, zero);
      const __m128i tmp2 = _mm_loadl_epi64((const __m128i*)&in[i]);
      const __m128i D = _mm_unpacklo_epi8(tmp2, zero);   // base input
      const __m128i E = _mm_sub_epi16(B, C);  // unclipped gradient basis B - C
      __m128i out = zero;                     // accumulator for output
      __m128i mask_hi = _mm_set_epi32(0, 0, 0, 0xff);
      int k = 8;
      while (1) {
        const __m128i tmp3 = _mm_add_epi16(A, E);        // delta = A + B - C
        const __m128i tmp4 = _mm_min_epi16(tmp3, mask_hi);
        const __m128i tmp5 = _mm_max_epi16(tmp4, zero);  // clipped delta
        const __m128i tmp6 = _mm_add_epi16(tmp5, D);     // add to in[] values
        A = _mm_and_si128(tmp6, mask_hi);                // 1-complement clip
        out = _mm_or_si128(out, A);                      // accumulate output
        if (--k == 0) break;
        A = _mm_slli_si128(A, 2);                        // rotate left sample
        mask_hi = _mm_slli_si128(mask_hi, 2);            // rotate mask
      }
      A = _mm_srli_si128(A, 14);       // prepare left sample for next iteration
      _mm_storel_epi64((__m128i*)&row[i], _mm_packus_epi16(out, zero));
    }
    for (; i < length; ++i) {
      row[i] = in[i] + GradientPredictorC(row[i - 1], top[i], top[i - 1]);
    }
  }
}

static WEBP_INLINE void DoGradientFilter(const uint8_t* in,
                                         int width, int height, int stride,
                                         int row, int num_rows,
                                         int inverse, uint8_t* out) {
  const size_t start_offset = row * stride;
  const int last_row = row + num_rows;
  SANITY_CHECK(in, out);
  in += start_offset;
  out += start_offset;

  // left prediction for top scan-line
  if (row == 0) {
    out[0] = in[0];
    PredictLineLeft(in + 1, out + 1, width - 1, inverse);
    row = 1;
    in += stride;
    out += stride;
  }

  // Filter line-by-line.
  while (row < last_row) {
    if (inverse) {
      PredictLineC(in, out - stride, out, 1, inverse);  // predict from above
      GradientPredictInverse(in + 1, out + 1 - stride, out + 1, width - 1);
    } else {
      PredictLineC(in, in - stride, out, 1, inverse);
      GradientPredictDirect(in + 1, in + 1 - stride, out + 1, width - 1);
    }
    ++row;
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
// Entry point

extern void VP8FiltersInitSSE2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8FiltersInitSSE2(void) {
  WebPUnfilters[WEBP_FILTER_HORIZONTAL] = HorizontalUnfilter;
  WebPUnfilters[WEBP_FILTER_VERTICAL] = VerticalUnfilter;
  WebPUnfilters[WEBP_FILTER_GRADIENT] = GradientUnfilter;

  WebPFilters[WEBP_FILTER_HORIZONTAL] = HorizontalFilter;
  WebPFilters[WEBP_FILTER_VERTICAL] = VerticalFilter;
  WebPFilters[WEBP_FILTER_GRADIENT] = GradientFilter;
}

#else  // !WEBP_USE_SSE2

WEBP_DSP_INIT_STUB(VP8FiltersInitSSE2)

#endif  // WEBP_USE_SSE2
