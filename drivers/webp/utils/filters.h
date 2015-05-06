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

#ifndef WEBP_UTILS_FILTERS_H_
#define WEBP_UTILS_FILTERS_H_

#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// Filters.
typedef enum {
  WEBP_FILTER_NONE = 0,
  WEBP_FILTER_HORIZONTAL,
  WEBP_FILTER_VERTICAL,
  WEBP_FILTER_GRADIENT,
  WEBP_FILTER_LAST = WEBP_FILTER_GRADIENT + 1,  // end marker
  WEBP_FILTER_BEST,
  WEBP_FILTER_FAST
} WEBP_FILTER_TYPE;

typedef void (*WebPFilterFunc)(const uint8_t* in, int width, int height,
                               int bpp, int stride, uint8_t* out);

// Filter the given data using the given predictor.
// 'in' corresponds to a 2-dimensional pixel array of size (stride * height)
// in raster order.
// 'bpp' is number of bytes per pixel, and
// 'stride' is number of bytes per scan line (with possible padding).
// 'out' should be pre-allocated.
extern const WebPFilterFunc WebPFilters[WEBP_FILTER_LAST];

// Reconstruct the original data from the given filtered data.
extern const WebPFilterFunc WebPUnfilters[WEBP_FILTER_LAST];

// Fast estimate of a potentially good filter.
extern WEBP_FILTER_TYPE EstimateBestFilter(const uint8_t* data,
                                           int width, int height, int stride);

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_FILTERS_H_ */
