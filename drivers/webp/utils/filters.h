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

#ifndef WEBP_UTILS_FILTERS_H_
#define WEBP_UTILS_FILTERS_H_

#include "../webp/types.h"

#ifdef __cplusplus
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
                               int stride, uint8_t* out);
typedef void (*WebPUnfilterFunc)(int width, int height, int stride,
                                 int row, int num_rows, uint8_t* data);

// Filter the given data using the given predictor.
// 'in' corresponds to a 2-dimensional pixel array of size (stride * height)
// in raster order.
// 'stride' is number of bytes per scan line (with possible padding).
// 'out' should be pre-allocated.
extern const WebPFilterFunc WebPFilters[WEBP_FILTER_LAST];

// In-place reconstruct the original data from the given filtered data.
// The reconstruction will be done for 'num_rows' rows starting from 'row'
// (assuming rows upto 'row - 1' are already reconstructed).
extern const WebPUnfilterFunc WebPUnfilters[WEBP_FILTER_LAST];

// Fast estimate of a potentially good filter.
WEBP_FILTER_TYPE EstimateBestFilter(const uint8_t* data,
                                    int width, int height, int stride);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_FILTERS_H_ */
