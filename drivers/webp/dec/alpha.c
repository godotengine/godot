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

#include <stdlib.h>
#include "./alphai.h"
#include "./vp8i.h"
#include "./vp8li.h"
#include "../utils/quant_levels_dec.h"
#include "../webp/format_constants.h"

//------------------------------------------------------------------------------
// ALPHDecoder object.

ALPHDecoder* ALPHNew(void) {
  ALPHDecoder* const dec = (ALPHDecoder*)calloc(1, sizeof(*dec));
  return dec;
}

void ALPHDelete(ALPHDecoder* const dec) {
  if (dec != NULL) {
    VP8LDelete(dec->vp8l_dec_);
    dec->vp8l_dec_ = NULL;
    free(dec);
  }
}

//------------------------------------------------------------------------------
// Decoding.

// Initialize alpha decoding by parsing the alpha header and decoding the image
// header for alpha data stored using lossless compression.
// Returns false in case of error in alpha header (data too short, invalid
// compression method or filter, error in lossless header data etc).
static int ALPHInit(ALPHDecoder* const dec, const uint8_t* data,
                    size_t data_size, int width, int height, uint8_t* output) {
  int ok = 0;
  const uint8_t* const alpha_data = data + ALPHA_HEADER_LEN;
  const size_t alpha_data_size = data_size - ALPHA_HEADER_LEN;
  int rsrv;

  assert(width > 0 && height > 0);
  assert(data != NULL && output != NULL);

  dec->width_ = width;
  dec->height_ = height;

  if (data_size <= ALPHA_HEADER_LEN) {
    return 0;
  }

  dec->method_ = (data[0] >> 0) & 0x03;
  dec->filter_ = (data[0] >> 2) & 0x03;
  dec->pre_processing_ = (data[0] >> 4) & 0x03;
  rsrv = (data[0] >> 6) & 0x03;
  if (dec->method_ < ALPHA_NO_COMPRESSION ||
      dec->method_ > ALPHA_LOSSLESS_COMPRESSION ||
      dec->filter_ >= WEBP_FILTER_LAST ||
      dec->pre_processing_ > ALPHA_PREPROCESSED_LEVELS ||
      rsrv != 0) {
    return 0;
  }

  if (dec->method_ == ALPHA_NO_COMPRESSION) {
    const size_t alpha_decoded_size = dec->width_ * dec->height_;
    ok = (alpha_data_size >= alpha_decoded_size);
  } else {
    assert(dec->method_ == ALPHA_LOSSLESS_COMPRESSION);
    ok = VP8LDecodeAlphaHeader(dec, alpha_data, alpha_data_size, output);
  }
  return ok;
}

// Decodes, unfilters and dequantizes *at least* 'num_rows' rows of alpha
// starting from row number 'row'. It assumes that rows up to (row - 1) have
// already been decoded.
// Returns false in case of bitstream error.
static int ALPHDecode(VP8Decoder* const dec, int row, int num_rows) {
  ALPHDecoder* const alph_dec = dec->alph_dec_;
  const int width = alph_dec->width_;
  const int height = alph_dec->height_;
  WebPUnfilterFunc unfilter_func = WebPUnfilters[alph_dec->filter_];
  uint8_t* const output = dec->alpha_plane_;
  if (alph_dec->method_ == ALPHA_NO_COMPRESSION) {
    const size_t offset = row * width;
    const size_t num_pixels = num_rows * width;
    assert(dec->alpha_data_size_ >= ALPHA_HEADER_LEN + offset + num_pixels);
    memcpy(dec->alpha_plane_ + offset,
           dec->alpha_data_ + ALPHA_HEADER_LEN + offset, num_pixels);
  } else {  // alph_dec->method_ == ALPHA_LOSSLESS_COMPRESSION
    assert(alph_dec->vp8l_dec_ != NULL);
    if (!VP8LDecodeAlphaImageStream(alph_dec, row + num_rows)) {
      return 0;
    }
  }

  if (unfilter_func != NULL) {
    unfilter_func(width, height, width, row, num_rows, output);
  }

  if (alph_dec->pre_processing_ == ALPHA_PREPROCESSED_LEVELS) {
    if (!DequantizeLevels(output, width, height, row, num_rows)) {
      return 0;
    }
  }

  if (row + num_rows == dec->pic_hdr_.height_) {
    dec->is_alpha_decoded_ = 1;
  }
  return 1;
}

//------------------------------------------------------------------------------
// Main entry point.

const uint8_t* VP8DecompressAlphaRows(VP8Decoder* const dec,
                                      int row, int num_rows) {
  const int width = dec->pic_hdr_.width_;
  const int height = dec->pic_hdr_.height_;

  if (row < 0 || num_rows <= 0 || row + num_rows > height) {
    return NULL;    // sanity check.
  }

  if (row == 0) {
    // Initialize decoding.
    assert(dec->alpha_plane_ != NULL);
    dec->alph_dec_ = ALPHNew();
    if (dec->alph_dec_ == NULL) return NULL;
    if (!ALPHInit(dec->alph_dec_, dec->alpha_data_, dec->alpha_data_size_,
                  width, height, dec->alpha_plane_)) {
      ALPHDelete(dec->alph_dec_);
      dec->alph_dec_ = NULL;
      return NULL;
    }
  }

  if (!dec->is_alpha_decoded_) {
    int ok = 0;
    assert(dec->alph_dec_ != NULL);
    ok = ALPHDecode(dec, row, num_rows);
    if (!ok || dec->is_alpha_decoded_) {
      ALPHDelete(dec->alph_dec_);
      dec->alph_dec_ = NULL;
    }
    if (!ok) return NULL;  // Error.
  }

  // Return a pointer to the current decoded row.
  return dec->alpha_plane_ + row * width;
}

