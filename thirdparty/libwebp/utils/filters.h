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
#include "../dsp/dsp.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fast estimate of a potentially good filter.
WEBP_FILTER_TYPE WebPEstimateBestFilter(const uint8_t* data,
                                        int width, int height, int stride);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_FILTERS_H_ */
