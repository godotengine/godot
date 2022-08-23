// Copyright 2022 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Gamma correction utilities.

#ifndef WEBP_SHARPYUV_SHARPYUV_GAMMA_H_
#define WEBP_SHARPYUV_SHARPYUV_GAMMA_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initializes precomputed tables. Must be called once before calling
// SharpYuvGammaToLinear or SharpYuvLinearToGamma.
void SharpYuvInitGammaTables(void);

// Converts a gamma color value on 'bit_depth' bits to a 16 bit linear value.
uint32_t SharpYuvGammaToLinear(uint16_t v, int bit_depth);

// Converts a 16 bit linear color value to a gamma value on 'bit_depth' bits.
uint16_t SharpYuvLinearToGamma(uint32_t value, int bit_depth);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WEBP_SHARPYUV_SHARPYUV_GAMMA_H_
