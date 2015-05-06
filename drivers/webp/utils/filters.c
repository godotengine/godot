// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Spatial prediction using various filters
//
// Author: Urvang (urvang@google.com)

#include "./filters.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Helpful macro.

# define SANITY_CHECK(in, out)                              \
  assert(in != NULL);                                       \
  assert(out != NULL);                                      \
  assert(width > 0);                                        \
  assert(height > 0);                                       \
  assert(bpp > 0);                                          \
  assert(stride >= width * bpp);

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
    int width, int height, int bpp, int stride, int inverse, uint8_t* out) {
  int h;
  const uint8_t* preds = (inverse ? out : in);
  SANITY_CHECK(in, out);

  // Filter line-by-line.
  for (h = 0; h < height; ++h) {
    // Leftmost pixel is predicted from above (except for topmost scanline).
    if (h == 0) {
      memcpy((void*)out, (const void*)in, bpp);
    } else {
      PredictLine(in, preds - stride, out, bpp, inverse);
    }
    PredictLine(in + bpp, preds, out + bpp, bpp * (width - 1), inverse);
    preds += stride;
    in += stride;
    out += stride;
  }
}

static void HorizontalFilter(const uint8_t* data, int width, int height,
                             int bpp, int stride, uint8_t* filtered_data) {
  DoHorizontalFilter(data, width, height, bpp, stride, 0, filtered_data);
}

static void HorizontalUnfilter(const uint8_t* data, int width, int height,
                               int bpp, int stride, uint8_t* recon_data) {
  DoHorizontalFilter(data, width, height, bpp, stride, 1, recon_data);
}

//------------------------------------------------------------------------------
// Vertical filter.

static WEBP_INLINE void DoVerticalFilter(const uint8_t* in,
    int width, int height, int bpp, int stride, int inverse, uint8_t* out) {
  int h;
  const uint8_t* preds = (inverse ? out : in);
  SANITY_CHECK(in, out);

  // Very first top-left pixel is copied.
  memcpy((void*)out, (const void*)in, bpp);
  // Rest of top scan-line is left-predicted.
  PredictLine(in + bpp, preds, out + bpp, bpp * (width - 1), inverse);

  // Filter line-by-line.
  for (h = 1; h < height; ++h) {
    in += stride;
    out += stride;
    PredictLine(in, preds, out, bpp * width, inverse);
    preds += stride;
  }
}

static void VerticalFilter(const uint8_t* data, int width, int height,
                           int bpp, int stride, uint8_t* filtered_data) {
  DoVerticalFilter(data, width, height, bpp, stride, 0, filtered_data);
}

static void VerticalUnfilter(const uint8_t* data, int width, int height,
                             int bpp, int stride, uint8_t* recon_data) {
  DoVerticalFilter(data, width, height, bpp, stride, 1, recon_data);
}

//------------------------------------------------------------------------------
// Gradient filter.

static WEBP_INLINE int GradientPredictor(uint8_t a, uint8_t b, uint8_t c) {
  const int g = a + b - c;
  return (g < 0) ? 0 : (g > 255) ? 255 : g;
}

static WEBP_INLINE
void DoGradientFilter(const uint8_t* in, int width, int height,
                      int bpp, int stride, int inverse, uint8_t* out) {
  const uint8_t* preds = (inverse ? out : in);
  int h;
  SANITY_CHECK(in, out);

  // left prediction for top scan-line
  memcpy((void*)out, (const void*)in, bpp);
  PredictLine(in + bpp, preds, out + bpp, bpp * (width - 1), inverse);

  // Filter line-by-line.
  for (h = 1; h < height; ++h) {
    int w;
    preds += stride;
    in += stride;
    out += stride;
    // leftmost pixel: predict from above.
    PredictLine(in, preds - stride, out, bpp, inverse);
    for (w = bpp; w < width * bpp; ++w) {
      const int pred = GradientPredictor(preds[w - bpp],
                                         preds[w - stride],
                                         preds[w - stride - bpp]);
      out[w] = in[w] + (inverse ? pred : -pred);
    }
  }
}

static void GradientFilter(const uint8_t* data, int width, int height,
                           int bpp, int stride, uint8_t* filtered_data) {
  DoGradientFilter(data, width, height, bpp, stride, 0, filtered_data);
}

static void GradientUnfilter(const uint8_t* data, int width, int height,
                             int bpp, int stride, uint8_t* recon_data) {
  DoGradientFilter(data, width, height, bpp, stride, 1, recon_data);
}

#undef SANITY_CHECK

// -----------------------------------------------------------------------------
// Quick estimate of a potentially interesting filter mode to try, in addition
// to the default NONE.

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
    WEBP_FILTER_TYPE filter, best_filter = WEBP_FILTER_NONE;
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
        best_filter = filter;
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

const WebPFilterFunc WebPUnfilters[WEBP_FILTER_LAST] = {
  NULL,                // WEBP_FILTER_NONE
  HorizontalUnfilter,  // WEBP_FILTER_HORIZONTAL
  VerticalUnfilter,    // WEBP_FILTER_VERTICAL
  GradientUnfilter     // WEBP_FILTER_GRADIENT
};

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
