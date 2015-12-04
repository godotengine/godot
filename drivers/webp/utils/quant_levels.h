// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Alpha plane quantization utility
//
// Author:  Vikas Arora (vikasa@google.com)

#ifndef WEBP_UTILS_QUANT_LEVELS_H_
#define WEBP_UTILS_QUANT_LEVELS_H_

#include <stdlib.h>

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Replace the input 'data' of size 'width'x'height' with 'num-levels'
// quantized values. If not NULL, 'sse' will contain the sum of squared error.
// Valid range for 'num_levels' is [2, 256].
// Returns false in case of error (data is NULL, or parameters are invalid).
int QuantizeLevels(uint8_t* const data, int width, int height, int num_levels,
                   uint64_t* const sse);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_QUANT_LEVELS_H_ */
