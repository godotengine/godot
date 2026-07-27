// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Alpha-plane decompression.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>

#include "src/dec/alphai_dec.h"
#include "src/dec/vp8_dec.h"
#include "src/dec/vp8i_dec.h"
#include "src/dec/vp8li_dec.h"
#include "src/dec/webpi_dec.h"
#include "src/dsp/dsp.h"
#include "src/utils/quant_levels_dec_utils.h"
#include "src/utils/utils.h"
#include "src/webp/decode.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// ALPHDecoder object.

// Allocates a new alpha decoder instance.
WEBP_NODISCARD static ALPHDecoder* ALPHNew(void) {
  ALPHDecoder* const dec = (ALPHDecoder*)WebPSafeCalloc(1ULL, sizeof(*dec));
  return dec;
}

// Clears and deallocates an alpha decoder instance.
static void ALPHDelete(ALPHDecoder* const dec) {
  if (dec != NULL) {
    VP8LDelete(dec->vp8l_dec);
    dec->vp8l_dec = NULL;
    WebPSafeFree(dec);
  }
}

//------------------------------------------------------------------------------
// Decoding.

// Initialize alpha decoding by parsing the alpha header and decoding the image
// header for alpha data stored using lossless compression.
// Returns false in case of error in alpha header (data too short, invalid
// compression method or filter, error in lossless header data etc).
WEBP_NODISCARD static int ALPHInit(ALPHDecoder* const dec, const uint8_t* data,
                                   size_t data_size, const VP8Io* const src_io,
                                   uint8_t* output) {
  int ok = 0;
  const uint8_t* const alpha_data = data + ALPHA_HEADER_LEN;
  const size_t alpha_data_size = data_size - ALPHA_HEADER_LEN;
  int rsrv;
  VP8Io* const io = &dec->io;

  assert(data != NULL && output != NULL && src_io != NULL);

  VP8FiltersInit();
  dec->output = output;
  dec->width = src_io->width;
  dec->height = src_io->height;
  assert(dec->width > 0 && dec->height > 0);

  if (data_size <= ALPHA_HEADER_LEN) {
    return 0;
  }

  dec->method = (data[0] >> 0) & 0x03;
  dec->filter = (WEBP_FILTER_TYPE)((data[0] >> 2) & 0x03);
  dec->pre_processing = (data[0] >> 4) & 0x03;
  rsrv = (data[0] >> 6) & 0x03;
  if (dec->method < ALPHA_NO_COMPRESSION ||
      dec->method > ALPHA_LOSSLESS_COMPRESSION ||
      dec->filter >= WEBP_FILTER_LAST ||
      dec->pre_processing > ALPHA_PREPROCESSED_LEVELS ||
      rsrv != 0) {
    return 0;
  }

  // Copy the necessary parameters from src_io to io
  if (!VP8InitIo(io)) {
    return 0;
  }
  WebPInitCustomIo(NULL, io);
  io->opaque = dec;
  io->width = src_io->width;
  io->height = src_io->height;

  io->use_cropping = src_io->use_cropping;
  io->crop_left = src_io->crop_left;
  io->crop_right = src_io->crop_right;
  io->crop_top = src_io->crop_top;
  io->crop_bottom = src_io->crop_bottom;
  // No need to copy the scaling parameters.

  if (dec->method == ALPHA_NO_COMPRESSION) {
    const size_t alpha_decoded_size = dec->width * dec->height;
    ok = (alpha_data_size >= alpha_decoded_size);
  } else {
    assert(dec->method == ALPHA_LOSSLESS_COMPRESSION);
    ok = VP8LDecodeAlphaHeader(dec, alpha_data, alpha_data_size);
  }

  return ok;
}

// Decodes, unfilters and dequantizes *at least* 'num_rows' rows of alpha
// starting from row number 'row'. It assumes that rows up to (row - 1) have
// already been decoded.
// Returns false in case of bitstream error.
WEBP_NODISCARD static int ALPHDecode(VP8Decoder* const dec, int row,
                                     int num_rows) {
  ALPHDecoder* const alph_dec = dec->alph_dec;
  const int width = alph_dec->width;
  const int height = alph_dec->io.crop_bottom;
  if (alph_dec->method == ALPHA_NO_COMPRESSION) {
    int y;
    const uint8_t* prev_line = dec->alpha_prev_line;
    const uint8_t* deltas = dec->alpha_data + ALPHA_HEADER_LEN + row * width;
    uint8_t* dst = dec->alpha_plane + row * width;
    assert(deltas <= &dec->alpha_data[dec->alpha_data_size]);
    assert(WebPUnfilters[alph_dec->filter] != NULL);
    for (y = 0; y < num_rows; ++y) {
      WebPUnfilters[alph_dec->filter](prev_line, deltas, dst, width);
      prev_line = dst;
      dst += width;
      deltas += width;
    }
    dec->alpha_prev_line = prev_line;
  } else {  // alph_dec->method == ALPHA_LOSSLESS_COMPRESSION
    assert(alph_dec->vp8l_dec != NULL);
    if (!VP8LDecodeAlphaImageStream(alph_dec, row + num_rows)) {
      return 0;
    }
  }

  if (row + num_rows >= height) {
    dec->is_alpha_decoded = 1;
  }
  return 1;
}

WEBP_NODISCARD static int AllocateAlphaPlane(VP8Decoder* const dec,
                                             const VP8Io* const io) {
  const int stride = io->width;
  const int height = io->crop_bottom;
  const uint64_t alpha_size = (uint64_t)stride * height;
  assert(dec->alpha_plane_mem == NULL);
  dec->alpha_plane_mem =
      (uint8_t*)WebPSafeMalloc(alpha_size, sizeof(*dec->alpha_plane));
  if (dec->alpha_plane_mem == NULL) {
    return VP8SetError(dec, VP8_STATUS_OUT_OF_MEMORY,
                       "Alpha decoder initialization failed.");
  }
  dec->alpha_plane = dec->alpha_plane_mem;
  dec->alpha_prev_line = NULL;
  return 1;
}

void WebPDeallocateAlphaMemory(VP8Decoder* const dec) {
  assert(dec != NULL);
  WebPSafeFree(dec->alpha_plane_mem);
  dec->alpha_plane_mem = NULL;
  dec->alpha_plane = NULL;
  ALPHDelete(dec->alph_dec);
  dec->alph_dec = NULL;
}

//------------------------------------------------------------------------------
// Main entry point.

WEBP_NODISCARD const uint8_t* VP8DecompressAlphaRows(VP8Decoder* const dec,
                                                     const VP8Io* const io,
                                                     int row, int num_rows) {
  const int width = io->width;
  const int height = io->crop_bottom;

  assert(dec != NULL && io != NULL);

  if (row < 0 || num_rows <= 0 || row + num_rows > height) {
    return NULL;
  }

  if (!dec->is_alpha_decoded) {
    if (dec->alph_dec == NULL) {    // Initialize decoder.
      dec->alph_dec = ALPHNew();
      if (dec->alph_dec == NULL) {
        VP8SetError(dec, VP8_STATUS_OUT_OF_MEMORY,
                    "Alpha decoder initialization failed.");
        return NULL;
      }
      if (!AllocateAlphaPlane(dec, io)) goto Error;
      if (!ALPHInit(dec->alph_dec, dec->alpha_data, dec->alpha_data_size,
                    io, dec->alpha_plane)) {
        VP8LDecoder* const vp8l_dec = dec->alph_dec->vp8l_dec;
        VP8SetError(dec,
                    (vp8l_dec == NULL) ? VP8_STATUS_OUT_OF_MEMORY
                                       : vp8l_dec->status,
                    "Alpha decoder initialization failed.");
        goto Error;
      }
      // if we allowed use of alpha dithering, check whether it's needed at all
      if (dec->alph_dec->pre_processing != ALPHA_PREPROCESSED_LEVELS) {
        dec->alpha_dithering = 0;    // disable dithering
      } else {
        num_rows = height - row;     // decode everything in one pass
      }
    }

    assert(dec->alph_dec != NULL);
    assert(row + num_rows <= height);
    if (!ALPHDecode(dec, row, num_rows)) goto Error;

    if (dec->is_alpha_decoded) {   // finished?
      ALPHDelete(dec->alph_dec);
      dec->alph_dec = NULL;
      if (dec->alpha_dithering > 0) {
        uint8_t* const alpha = dec->alpha_plane + io->crop_top * width
                             + io->crop_left;
        if (!WebPDequantizeLevels(alpha,
                                  io->crop_right - io->crop_left,
                                  io->crop_bottom - io->crop_top,
                                  width, dec->alpha_dithering)) {
          goto Error;
        }
      }
    }
  }

  // Return a pointer to the current decoded row.
  return dec->alpha_plane + row * width;

 Error:
  WebPDeallocateAlphaMemory(dec);
  return NULL;
}
