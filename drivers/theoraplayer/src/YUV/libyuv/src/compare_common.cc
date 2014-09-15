/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

uint32 SumSquareError_C(const uint8* src_a, const uint8* src_b, int count) {
  uint32 sse = 0u;
  int i;
  for (i = 0; i < count; ++i) {
    int diff = src_a[i] - src_b[i];
    sse += (uint32)(diff * diff);
  }
  return sse;
}

// hash seed of 5381 recommended.
// Internal C version of HashDjb2 with int sized count for efficiency.
uint32 HashDjb2_C(const uint8* src, int count, uint32 seed) {
  uint32 hash = seed;
  int i;
  for (i = 0; i < count; ++i) {
    hash += (hash << 5) + src[i];
  }
  return hash;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif
