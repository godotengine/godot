// Copyright 2013 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Alpha plane de-quantization utility
//
// Author:  Vikas Arora (vikasa@google.com)

#ifndef WEBP_UTILS_QUANT_LEVELS_DEC_H_
#define WEBP_UTILS_QUANT_LEVELS_DEC_H_

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Apply post-processing to input 'data' of size 'width'x'height' assuming that
// the source was quantized to a reduced number of levels.
// Strength is in [0..100] and controls the amount of dithering applied.
// Returns false in case of error (data is NULL, invalid parameters,
// malloc failure, ...).
int WebPDequantizeLevels(uint8_t* const data, int width, int height,
                         int strength);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_QUANT_LEVELS_DEC_H_ */
