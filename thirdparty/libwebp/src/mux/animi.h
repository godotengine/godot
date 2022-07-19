// Copyright 2016 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Internal header for animation related functions.
//
// Author: Hui Su (huisu@google.com)

#ifndef WEBP_MUX_ANIMI_H_
#define WEBP_MUX_ANIMI_H_

#include "src/webp/mux.h"

#ifdef __cplusplus
extern "C" {
#endif

// Picks the optimal rectangle between two pictures, starting with initial
// values of offsets and dimensions that are passed in. The initial
// values will be clipped, if necessary, to make sure the rectangle is
// within the canvas. "use_argb" must be true for both pictures.
// Parameters:
//   prev_canvas, curr_canvas - (in) two input pictures to compare.
//   is_lossless, quality - (in) encoding settings.
//   x_offset, y_offset, width, height - (in/out) rectangle between the two
//                                                input pictures.
// Returns true on success.
int WebPAnimEncoderRefineRect(
    const struct WebPPicture* const prev_canvas,
    const struct WebPPicture* const curr_canvas,
    int is_lossless, float quality, int* const x_offset, int* const y_offset,
    int* const width, int* const height);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_MUX_ANIMI_H_
