// Copyright 2013 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Pseudo-random utilities
//
// Author: Skal (pascal.massimino@gmail.com)

#include <string.h>
#include "src/utils/random_utils.h"

//------------------------------------------------------------------------------

// 31b-range values
static const uint32_t kRandomTable[VP8_RANDOM_TABLE_SIZE] = {
  0x0de15230, 0x03b31886, 0x775faccb, 0x1c88626a, 0x68385c55, 0x14b3b828,
  0x4a85fef8, 0x49ddb84b, 0x64fcf397, 0x5c550289, 0x4a290000, 0x0d7ec1da,
  0x5940b7ab, 0x5492577d, 0x4e19ca72, 0x38d38c69, 0x0c01ee65, 0x32a1755f,
  0x5437f652, 0x5abb2c32, 0x0faa57b1, 0x73f533e7, 0x685feeda, 0x7563cce2,
  0x6e990e83, 0x4730a7ed, 0x4fc0d9c6, 0x496b153c, 0x4f1403fa, 0x541afb0c,
  0x73990b32, 0x26d7cb1c, 0x6fcc3706, 0x2cbb77d8, 0x75762f2a, 0x6425ccdd,
  0x24b35461, 0x0a7d8715, 0x220414a8, 0x141ebf67, 0x56b41583, 0x73e502e3,
  0x44cab16f, 0x28264d42, 0x73baaefb, 0x0a50ebed, 0x1d6ab6fb, 0x0d3ad40b,
  0x35db3b68, 0x2b081e83, 0x77ce6b95, 0x5181e5f0, 0x78853bbc, 0x009f9494,
  0x27e5ed3c
};

void VP8InitRandom(VP8Random* const rg, float dithering) {
  memcpy(rg->tab_, kRandomTable, sizeof(rg->tab_));
  rg->index1_ = 0;
  rg->index2_ = 31;
  rg->amp_ = (dithering < 0.0) ? 0
           : (dithering > 1.0) ? (1 << VP8_RANDOM_DITHER_FIX)
           : (uint32_t)((1 << VP8_RANDOM_DITHER_FIX) * dithering);
}

//------------------------------------------------------------------------------

