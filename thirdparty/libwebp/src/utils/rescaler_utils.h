// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Rescaling functions
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_UTILS_RESCALER_UTILS_H_
#define WEBP_UTILS_RESCALER_UTILS_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "src/webp/types.h"

#define WEBP_RESCALER_RFIX 32   // fixed-point precision for multiplies
#define WEBP_RESCALER_ONE (1ull << WEBP_RESCALER_RFIX)
#define WEBP_RESCALER_FRAC(x, y) \
    ((uint32_t)(((uint64_t)(x) << WEBP_RESCALER_RFIX) / (y)))

// Structure used for on-the-fly rescaling
typedef uint32_t rescaler_t;   // type for side-buffer
typedef struct WebPRescaler WebPRescaler;
struct WebPRescaler {
  int x_expand;               // true if we're expanding in the x direction
  int y_expand;               // true if we're expanding in the y direction
  int num_channels;           // bytes to jump between pixels
  uint32_t fx_scale;          // fixed-point scaling factors
  uint32_t fy_scale;          // ''
  uint32_t fxy_scale;         // ''
  int y_accum;                // vertical accumulator
  int y_add, y_sub;           // vertical increments
  int x_add, x_sub;           // horizontal increments
  int src_width, src_height;  // source dimensions
  int dst_width, dst_height;  // destination dimensions
  int src_y, dst_y;           // row counters for input and output
  uint8_t* dst;
  int dst_stride;
  rescaler_t* irow, *frow;    // work buffer
};

// Initialize a rescaler given scratch area 'work' and dimensions of src & dst.
void WebPRescalerInit(WebPRescaler* const rescaler,
                      int src_width, int src_height,
                      uint8_t* const dst,
                      int dst_width, int dst_height, int dst_stride,
                      int num_channels,
                      rescaler_t* const work);

// If either 'scaled_width' or 'scaled_height' (but not both) is 0 the value
// will be calculated preserving the aspect ratio, otherwise the values are
// left unmodified. Returns true on success, false if either value is 0 after
// performing the scaling calculation.
int WebPRescalerGetScaledDimensions(int src_width, int src_height,
                                    int* const scaled_width,
                                    int* const scaled_height);

// Returns the number of input lines needed next to produce one output line,
// considering that the maximum available input lines are 'max_num_lines'.
int WebPRescaleNeededLines(const WebPRescaler* const rescaler,
                           int max_num_lines);

// Import multiple rows over all channels, until at least one row is ready to
// be exported. Returns the actual number of lines that were imported.
int WebPRescalerImport(WebPRescaler* const rescaler, int num_rows,
                       const uint8_t* src, int src_stride);

// Export as many rows as possible. Return the numbers of rows written.
int WebPRescalerExport(WebPRescaler* const rescaler);

// Return true if input is finished
static WEBP_INLINE
int WebPRescalerInputDone(const WebPRescaler* const rescaler) {
  return (rescaler->src_y >= rescaler->src_height);
}
// Return true if output is finished
static WEBP_INLINE
int WebPRescalerOutputDone(const WebPRescaler* const rescaler) {
  return (rescaler->dst_y >= rescaler->dst_height);
}

// Return true if there are pending output rows ready.
static WEBP_INLINE
int WebPRescalerHasPendingOutput(const WebPRescaler* const rescaler) {
  return !WebPRescalerOutputDone(rescaler) && (rescaler->y_accum <= 0);
}

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_UTILS_RESCALER_UTILS_H_
