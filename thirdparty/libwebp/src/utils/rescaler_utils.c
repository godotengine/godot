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

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "src/dsp/dsp.h"
#include "src/utils/rescaler_utils.h"

//------------------------------------------------------------------------------

void WebPRescalerInit(WebPRescaler* const wrk, int src_width, int src_height,
                      uint8_t* const dst,
                      int dst_width, int dst_height, int dst_stride,
                      int num_channels, rescaler_t* const work) {
  const int x_add = src_width, x_sub = dst_width;
  const int y_add = src_height, y_sub = dst_height;
  wrk->x_expand = (src_width < dst_width);
  wrk->y_expand = (src_height < dst_height);
  wrk->src_width = src_width;
  wrk->src_height = src_height;
  wrk->dst_width = dst_width;
  wrk->dst_height = dst_height;
  wrk->src_y = 0;
  wrk->dst_y = 0;
  wrk->dst = dst;
  wrk->dst_stride = dst_stride;
  wrk->num_channels = num_channels;

  // for 'x_expand', we use bilinear interpolation
  wrk->x_add = wrk->x_expand ? (x_sub - 1) : x_add;
  wrk->x_sub = wrk->x_expand ? (x_add - 1) : x_sub;
  if (!wrk->x_expand) {  // fx_scale is not used otherwise
    wrk->fx_scale = WEBP_RESCALER_FRAC(1, wrk->x_sub);
  }
  // vertical scaling parameters
  wrk->y_add = wrk->y_expand ? y_add - 1 : y_add;
  wrk->y_sub = wrk->y_expand ? y_sub - 1 : y_sub;
  wrk->y_accum = wrk->y_expand ? wrk->y_sub : wrk->y_add;
  if (!wrk->y_expand) {
    // This is WEBP_RESCALER_FRAC(dst_height, x_add * y_add) without the cast.
    // Its value is <= WEBP_RESCALER_ONE, because dst_height <= wrk->y_add, and
    // wrk->x_add >= 1;
    const uint64_t ratio =
        (uint64_t)dst_height * WEBP_RESCALER_ONE / (wrk->x_add * wrk->y_add);
    if (ratio != (uint32_t)ratio) {
      // When ratio == WEBP_RESCALER_ONE, we can't represent the ratio with the
      // current fixed-point precision. This happens when src_height ==
      // wrk->y_add (which == src_height), and wrk->x_add == 1.
      // => We special-case fxy_scale = 0, in WebPRescalerExportRow().
      wrk->fxy_scale = 0;
    } else {
      wrk->fxy_scale = (uint32_t)ratio;
    }
    wrk->fy_scale = WEBP_RESCALER_FRAC(1, wrk->y_sub);
  } else {
    wrk->fy_scale = WEBP_RESCALER_FRAC(1, wrk->x_add);
    // wrk->fxy_scale is unused here.
  }
  wrk->irow = work;
  wrk->frow = work + num_channels * dst_width;
  memset(work, 0, 2 * dst_width * num_channels * sizeof(*work));

  WebPRescalerDspInit();
}

int WebPRescalerGetScaledDimensions(int src_width, int src_height,
                                    int* const scaled_width,
                                    int* const scaled_height) {
  assert(scaled_width != NULL);
  assert(scaled_height != NULL);
  {
    int width = *scaled_width;
    int height = *scaled_height;

    // if width is unspecified, scale original proportionally to height ratio.
    if (width == 0 && src_height > 0) {
      width =
          (int)(((uint64_t)src_width * height + src_height - 1) / src_height);
    }
    // if height is unspecified, scale original proportionally to width ratio.
    if (height == 0 && src_width > 0) {
      height =
          (int)(((uint64_t)src_height * width + src_width - 1) / src_width);
    }
    // Check if the overall dimensions still make sense.
    if (width <= 0 || height <= 0) {
      return 0;
    }

    *scaled_width = width;
    *scaled_height = height;
    return 1;
  }
}

//------------------------------------------------------------------------------
// all-in-one calls

int WebPRescaleNeededLines(const WebPRescaler* const wrk, int max_num_lines) {
  const int num_lines = (wrk->y_accum + wrk->y_sub - 1) / wrk->y_sub;
  return (num_lines > max_num_lines) ? max_num_lines : num_lines;
}

int WebPRescalerImport(WebPRescaler* const wrk, int num_lines,
                       const uint8_t* src, int src_stride) {
  int total_imported = 0;
  while (total_imported < num_lines && !WebPRescalerHasPendingOutput(wrk)) {
    if (wrk->y_expand) {
      rescaler_t* const tmp = wrk->irow;
      wrk->irow = wrk->frow;
      wrk->frow = tmp;
    }
    WebPRescalerImportRow(wrk, src);
    if (!wrk->y_expand) {     // Accumulate the contribution of the new row.
      int x;
      for (x = 0; x < wrk->num_channels * wrk->dst_width; ++x) {
        wrk->irow[x] += wrk->frow[x];
      }
    }
    ++wrk->src_y;
    src += src_stride;
    ++total_imported;
    wrk->y_accum -= wrk->y_sub;
  }
  return total_imported;
}

int WebPRescalerExport(WebPRescaler* const rescaler) {
  int total_exported = 0;
  while (WebPRescalerHasPendingOutput(rescaler)) {
    WebPRescalerExportRow(rescaler);
    ++total_exported;
  }
  return total_exported;
}

//------------------------------------------------------------------------------
