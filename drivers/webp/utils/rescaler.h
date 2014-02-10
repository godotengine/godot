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

#ifndef WEBP_UTILS_RESCALER_H_
#define WEBP_UTILS_RESCALER_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "../webp/types.h"

// Structure used for on-the-fly rescaling
typedef struct {
  int x_expand;               // true if we're expanding in the x direction
  int num_channels;           // bytes to jump between pixels
  int fy_scale, fx_scale;     // fixed-point scaling factor
  int64_t fxy_scale;          // ''
  // we need hpel-precise add/sub increments, for the downsampled U/V planes.
  int y_accum;                // vertical accumulator
  int y_add, y_sub;           // vertical increments (add ~= src, sub ~= dst)
  int x_add, x_sub;           // horizontal increments (add ~= src, sub ~= dst)
  int src_width, src_height;  // source dimensions
  int dst_width, dst_height;  // destination dimensions
  uint8_t* dst;
  int dst_stride;
  int32_t* irow, *frow;       // work buffer
} WebPRescaler;

// Initialize a rescaler given scratch area 'work' and dimensions of src & dst.
void WebPRescalerInit(WebPRescaler* const rescaler,
                      int src_width, int src_height,
                      uint8_t* const dst,
                      int dst_width, int dst_height, int dst_stride,
                      int num_channels,
                      int x_add, int x_sub,
                      int y_add, int y_sub,
                      int32_t* const work);

// Returns the number of input lines needed next to produce one output line,
// considering that the maximum available input lines are 'max_num_lines'.
int WebPRescaleNeededLines(const WebPRescaler* const rescaler,
                           int max_num_lines);

// Import a row of data and save its contribution in the rescaler.
// 'channel' denotes the channel number to be imported.
void WebPRescalerImportRow(WebPRescaler* const rescaler,
                           const uint8_t* const src, int channel);

// Import multiple rows over all channels, until at least one row is ready to
// be exported. Returns the actual number of lines that were imported.
int WebPRescalerImport(WebPRescaler* const rescaler, int num_rows,
                       const uint8_t* src, int src_stride);

// Return true if there is pending output rows ready.
static WEBP_INLINE
int WebPRescalerHasPendingOutput(const WebPRescaler* const rescaler) {
  return (rescaler->y_accum <= 0);
}

// Export one row from rescaler. Returns the pointer where output was written,
// or NULL if no row was pending.
uint8_t* WebPRescalerExportRow(WebPRescaler* const rescaler);

// Export as many rows as possible. Return the numbers of rows written.
int WebPRescalerExport(WebPRescaler* const rescaler);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_RESCALER_H_ */
