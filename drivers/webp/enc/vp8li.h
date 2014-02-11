// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Lossless encoder: internal header.
//
// Author: Vikas Arora (vikaas.arora@gmail.com)

#ifndef WEBP_ENC_VP8LI_H_
#define WEBP_ENC_VP8LI_H_

#include "./histogram.h"
#include "../utils/bit_writer.h"
#include "../webp/encode.h"
#include "../webp/format_constants.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

typedef struct {
  const WebPConfig* config_;    // user configuration and parameters
  const WebPPicture* pic_;      // input picture.

  uint32_t* argb_;              // Transformed argb image data.
  uint32_t* argb_scratch_;      // Scratch memory for argb rows
                                // (used for prediction).
  uint32_t* transform_data_;    // Scratch memory for transform data.
  int       current_width_;     // Corresponds to packed image width.

  // Encoding parameters derived from quality parameter.
  int histo_bits_;
  int transform_bits_;
  int cache_bits_;        // If equal to 0, don't use color cache.

  // Encoding parameters derived from image characteristics.
  int use_cross_color_;
  int use_subtract_green_;
  int use_predict_;
  int use_palette_;
  int palette_size_;
  uint32_t palette_[MAX_PALETTE_SIZE];
} VP8LEncoder;

//------------------------------------------------------------------------------
// internal functions. Not public.

// Encodes the picture.
// Returns 0 if config or picture is NULL or picture doesn't have valid argb
// input.
int VP8LEncodeImage(const WebPConfig* const config,
                    const WebPPicture* const picture);

// Encodes the main image stream using the supplied bit writer.
WebPEncodingError VP8LEncodeStream(const WebPConfig* const config,
                                   const WebPPicture* const picture,
                                   VP8LBitWriter* const bw);

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_ENC_VP8LI_H_ */
