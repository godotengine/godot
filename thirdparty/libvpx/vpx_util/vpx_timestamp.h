/*
 *  Copyright (c) 2019 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_UTIL_VPX_TIMESTAMP_H_
#define VPX_VPX_UTIL_VPX_TIMESTAMP_H_

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Rational Number with an int64 numerator
typedef struct vpx_rational64 {
  int64_t num;       // fraction numerator
  int den;           // fraction denominator
} vpx_rational64_t;  // alias for struct vpx_rational64_t

static INLINE int gcd(int64_t a, int b) {
  int r;  // remainder
  assert(a >= 0);
  assert(b > 0);
  while (b != 0) {
    r = (int)(a % b);
    a = b;
    b = r;
  }

  return (int)a;
}

static INLINE void reduce_ratio(vpx_rational64_t *ratio) {
  const int denom = gcd(ratio->num, ratio->den);
  ratio->num /= denom;
  ratio->den /= denom;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // VPX_VPX_UTIL_VPX_TIMESTAMP_H_
