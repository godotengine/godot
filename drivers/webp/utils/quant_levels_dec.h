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
// the source was quantized to a reduced number of levels. The post-processing
// will be applied to 'num_rows' rows of 'data' starting from 'row'.
// Returns false in case of error (data is NULL, invalid parameters, ...).
int DequantizeLevels(uint8_t* const data, int width, int height,
                     int row, int num_rows);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_QUANT_LEVELS_DEC_H_ */
