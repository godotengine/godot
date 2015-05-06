// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Alpha-plane decompression.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include "./vp8i.h"
#include "./vp8li.h"
#include "../utils/filters.h"
#include "../utils/quant_levels.h"
#include "../webp/format_constants.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// TODO(skal): move to dsp/ ?
static void CopyPlane(const uint8_t* src, int src_stride,
                      uint8_t* dst, int dst_stride, int width, int height) {
  while (height-- > 0) {
    memcpy(dst, src, width);
    src += src_stride;
    dst += dst_stride;
  }
}

//------------------------------------------------------------------------------
// Decodes the compressed data 'data' of size 'data_size' into the 'output'.
// The 'output' buffer should be pre-allocated and must be of the same
// dimension 'height'x'stride', as that of the image.
//
// Returns 1 on successfully decoding the compressed alpha and
//         0 if either:
//           error in bit-stream header (invalid compression mode or filter), or
//           error returned by appropriate compression method.

static int DecodeAlpha(const uint8_t* data, size_t data_size,
                       int width, int height, int stride, uint8_t* output) {
  uint8_t* decoded_data = NULL;
  const size_t decoded_size = height * width;
  uint8_t* unfiltered_data = NULL;
  WEBP_FILTER_TYPE filter;
  int pre_processing;
  int rsrv;
  int ok = 0;
  int method;

  assert(width > 0 && height > 0 && stride >= width);
  assert(data != NULL && output != NULL);

  if (data_size <= ALPHA_HEADER_LEN) {
    return 0;
  }

  method = (data[0] >> 0) & 0x03;
  filter = (data[0] >> 2) & 0x03;
  pre_processing = (data[0] >> 4) & 0x03;
  rsrv = (data[0] >> 6) & 0x03;
  if (method < ALPHA_NO_COMPRESSION ||
      method > ALPHA_LOSSLESS_COMPRESSION ||
      filter >= WEBP_FILTER_LAST ||
      pre_processing > ALPHA_PREPROCESSED_LEVELS ||
      rsrv != 0) {
    return 0;
  }

  if (method == ALPHA_NO_COMPRESSION) {
    ok = (data_size >= decoded_size);
    decoded_data = (uint8_t*)data + ALPHA_HEADER_LEN;
  } else {
    decoded_data = (uint8_t*)malloc(decoded_size);
    if (decoded_data == NULL) return 0;
    ok = VP8LDecodeAlphaImageStream(width, height,
                                    data + ALPHA_HEADER_LEN,
                                    data_size - ALPHA_HEADER_LEN,
                                    decoded_data);
  }

  if (ok) {
    WebPFilterFunc unfilter_func = WebPUnfilters[filter];
    if (unfilter_func != NULL) {
      unfiltered_data = (uint8_t*)malloc(decoded_size);
      if (unfiltered_data == NULL) {
        ok = 0;
        goto Error;
      }
      // TODO(vikas): Implement on-the-fly decoding & filter mechanism to decode
      // and apply filter per image-row.
      unfilter_func(decoded_data, width, height, 1, width, unfiltered_data);
      // Construct raw_data (height x stride) from alpha data (height x width).
      CopyPlane(unfiltered_data, width, output, stride, width, height);
      free(unfiltered_data);
    } else {
      // Construct raw_data (height x stride) from alpha data (height x width).
      CopyPlane(decoded_data, width, output, stride, width, height);
    }
    if (pre_processing == ALPHA_PREPROCESSED_LEVELS) {
      ok = DequantizeLevels(decoded_data, width, height);
    }
  }

 Error:
  if (method != ALPHA_NO_COMPRESSION) {
    free(decoded_data);
  }
  return ok;
}

//------------------------------------------------------------------------------

const uint8_t* VP8DecompressAlphaRows(VP8Decoder* const dec,
                                      int row, int num_rows) {
  const int stride = dec->pic_hdr_.width_;

  if (row < 0 || num_rows < 0 || row + num_rows > dec->pic_hdr_.height_) {
    return NULL;    // sanity check.
  }

  if (row == 0) {
    // Decode everything during the first call.
    if (!DecodeAlpha(dec->alpha_data_, (size_t)dec->alpha_data_size_,
                     dec->pic_hdr_.width_, dec->pic_hdr_.height_, stride,
                     dec->alpha_plane_)) {
      return NULL;  // Error.
    }
  }

  // Return a pointer to the current decoded row.
  return dec->alpha_plane_ + row * stride;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
