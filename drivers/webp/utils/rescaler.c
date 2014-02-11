// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Rescaling functions
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>
#include "./rescaler.h"

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define RFIX 30
#define MULT_FIX(x,y) (((int64_t)(x) * (y) + (1 << (RFIX - 1))) >> RFIX)

void WebPRescalerInit(WebPRescaler* const wrk, int src_width, int src_height,
                      uint8_t* const dst, int dst_width, int dst_height,
                      int dst_stride, int num_channels, int x_add, int x_sub,
                      int y_add, int y_sub, int32_t* const work) {
  wrk->x_expand = (src_width < dst_width);
  wrk->src_width = src_width;
  wrk->src_height = src_height;
  wrk->dst_width = dst_width;
  wrk->dst_height = dst_height;
  wrk->dst = dst;
  wrk->dst_stride = dst_stride;
  wrk->num_channels = num_channels;
  // for 'x_expand', we use bilinear interpolation
  wrk->x_add = wrk->x_expand ? (x_sub - 1) : x_add - x_sub;
  wrk->x_sub = wrk->x_expand ? (x_add - 1) : x_sub;
  wrk->y_accum = y_add;
  wrk->y_add = y_add;
  wrk->y_sub = y_sub;
  wrk->fx_scale = (1 << RFIX) / x_sub;
  wrk->fy_scale = (1 << RFIX) / y_sub;
  wrk->fxy_scale = wrk->x_expand ?
      ((int64_t)dst_height << RFIX) / (x_sub * src_height) :
      ((int64_t)dst_height << RFIX) / (x_add * src_height);
  wrk->irow = work;
  wrk->frow = work + num_channels * dst_width;
}

void WebPRescalerImportRow(WebPRescaler* const wrk,
                           const uint8_t* const src, int channel) {
  const int x_stride = wrk->num_channels;
  const int x_out_max = wrk->dst_width * wrk->num_channels;
  int x_in = channel;
  int x_out;
  int accum = 0;
  if (!wrk->x_expand) {
    int sum = 0;
    for (x_out = channel; x_out < x_out_max; x_out += x_stride) {
      accum += wrk->x_add;
      for (; accum > 0; accum -= wrk->x_sub) {
        sum += src[x_in];
        x_in += x_stride;
      }
      {        // Emit next horizontal pixel.
        const int32_t base = src[x_in];
        const int32_t frac = base * (-accum);
        x_in += x_stride;
        wrk->frow[x_out] = (sum + base) * wrk->x_sub - frac;
        // fresh fractional start for next pixel
        sum = (int)MULT_FIX(frac, wrk->fx_scale);
      }
    }
  } else {        // simple bilinear interpolation
    int left = src[channel], right = src[channel];
    for (x_out = channel; x_out < x_out_max; x_out += x_stride) {
      if (accum < 0) {
        left = right;
        x_in += x_stride;
        right = src[x_in];
        accum += wrk->x_add;
      }
      wrk->frow[x_out] = right * wrk->x_add + (left - right) * accum;
      accum -= wrk->x_sub;
    }
  }
  // Accumulate the new row's contribution
  for (x_out = channel; x_out < x_out_max; x_out += x_stride) {
    wrk->irow[x_out] += wrk->frow[x_out];
  }
}

uint8_t* WebPRescalerExportRow(WebPRescaler* const wrk) {
  if (wrk->y_accum <= 0) {
    int x_out;
    uint8_t* const dst = wrk->dst;
    int32_t* const irow = wrk->irow;
    const int32_t* const frow = wrk->frow;
    const int yscale = wrk->fy_scale * (-wrk->y_accum);
    const int x_out_max = wrk->dst_width * wrk->num_channels;

    for (x_out = 0; x_out < x_out_max; ++x_out) {
      const int frac = (int)MULT_FIX(frow[x_out], yscale);
      const int v = (int)MULT_FIX(irow[x_out] - frac, wrk->fxy_scale);
      dst[x_out] = (!(v & ~0xff)) ? v : (v < 0) ? 0 : 255;
      irow[x_out] = frac;   // new fractional start
    }
    wrk->y_accum += wrk->y_add;
    wrk->dst += wrk->dst_stride;
    return dst;
  } else {
    return NULL;
  }
}

#undef MULT_FIX
#undef RFIX

//------------------------------------------------------------------------------
// all-in-one calls

int WebPRescalerImport(WebPRescaler* const wrk, int num_lines,
                       const uint8_t* src, int src_stride) {
  int total_imported = 0;
  while (total_imported < num_lines && wrk->y_accum > 0) {
    int channel;
    for (channel = 0; channel < wrk->num_channels; ++channel) {
      WebPRescalerImportRow(wrk, src, channel);
    }
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

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
