/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_PPC_TRANSPOSE_VSX_H_
#define VPX_VPX_DSP_PPC_TRANSPOSE_VSX_H_

#include "./vpx_config.h"
#include "vpx_dsp/ppc/types_vsx.h"

static INLINE void vpx_transpose_s16_8x8(int16x8_t v[8]) {
  // d = vec_mergeh(a,b):
  // The even elements of the result are obtained left-to-right,
  // from the high elements of a.
  // The odd elements of the result are obtained left-to-right,
  // from the high elements of b.
  //
  // d = vec_mergel(a,b):
  // The even elements of the result are obtained left-to-right,
  // from the low elements of a.
  // The odd elements of the result are obtained left-to-right,
  // from the low elements of b.

  // Example, starting with:
  // v[0]: 00 01 02 03 04 05 06 07
  // v[1]: 10 11 12 13 14 15 16 17
  // v[2]: 20 21 22 23 24 25 26 27
  // v[3]: 30 31 32 33 34 35 36 37
  // v[4]: 40 41 42 43 44 45 46 47
  // v[5]: 50 51 52 53 54 55 56 57
  // v[6]: 60 61 62 63 64 65 66 67
  // v[7]: 70 71 72 73 74 75 76 77

  int16x8_t b0, b1, b2, b3, b4, b5, b6, b7;
  int16x8_t c0, c1, c2, c3, c4, c5, c6, c7;

  b0 = vec_mergeh(v[0], v[4]);
  b1 = vec_mergel(v[0], v[4]);
  b2 = vec_mergeh(v[1], v[5]);
  b3 = vec_mergel(v[1], v[5]);
  b4 = vec_mergeh(v[2], v[6]);
  b5 = vec_mergel(v[2], v[6]);
  b6 = vec_mergeh(v[3], v[7]);
  b7 = vec_mergel(v[3], v[7]);

  // After first merge operation
  // b0: 00 40 01 41 02 42 03 43
  // b1: 04 44 05 45 06 46 07 47
  // b2: 10 50 11 51 12 52 13 53
  // b3: 14 54 15 55 16 56 17 57
  // b4: 20 60 21 61 22 62 23 63
  // b5: 24 64 25 65 26 66 27 67
  // b6: 30 70 31 71 32 62 33 73
  // b7: 34 74 35 75 36 76 37 77

  c0 = vec_mergeh(b0, b4);
  c1 = vec_mergel(b0, b4);
  c2 = vec_mergeh(b1, b5);
  c3 = vec_mergel(b1, b5);
  c4 = vec_mergeh(b2, b6);
  c5 = vec_mergel(b2, b6);
  c6 = vec_mergeh(b3, b7);
  c7 = vec_mergel(b3, b7);

  // After second merge operation
  // c0: 00 20 40 60 01 21 41 61
  // c1: 02 22 42 62 03 23 43 63
  // c2: 04 24 44 64 05 25 45 65
  // c3: 06 26 46 66 07 27 47 67
  // c4: 10 30 50 70 11 31 51 71
  // c5: 12 32 52 72 13 33 53 73
  // c6: 14 34 54 74 15 35 55 75
  // c7: 16 36 56 76 17 37 57 77

  v[0] = vec_mergeh(c0, c4);
  v[1] = vec_mergel(c0, c4);
  v[2] = vec_mergeh(c1, c5);
  v[3] = vec_mergel(c1, c5);
  v[4] = vec_mergeh(c2, c6);
  v[5] = vec_mergel(c2, c6);
  v[6] = vec_mergeh(c3, c7);
  v[7] = vec_mergel(c3, c7);

  // After last merge operation
  // v[0]: 00 10 20 30 40 50 60 70
  // v[1]: 01 11 21 31 41 51 61 71
  // v[2]: 02 12 22 32 42 52 62 72
  // v[3]: 03 13 23 33 43 53 63 73
  // v[4]: 04 14 24 34 44 54 64 74
  // v[5]: 05 15 25 35 45 55 65 75
  // v[6]: 06 16 26 36 46 56 66 76
  // v[7]: 07 17 27 37 47 57 67 77
}

static INLINE void transpose_8x8(const int16x8_t *a, int16x8_t *b) {
  // Stage 1
  const int16x8_t s1_0 = vec_mergeh(a[0], a[4]);
  const int16x8_t s1_1 = vec_mergel(a[0], a[4]);
  const int16x8_t s1_2 = vec_mergeh(a[1], a[5]);
  const int16x8_t s1_3 = vec_mergel(a[1], a[5]);
  const int16x8_t s1_4 = vec_mergeh(a[2], a[6]);
  const int16x8_t s1_5 = vec_mergel(a[2], a[6]);
  const int16x8_t s1_6 = vec_mergeh(a[3], a[7]);
  const int16x8_t s1_7 = vec_mergel(a[3], a[7]);

  // Stage 2
  const int16x8_t s2_0 = vec_mergeh(s1_0, s1_4);
  const int16x8_t s2_1 = vec_mergel(s1_0, s1_4);
  const int16x8_t s2_2 = vec_mergeh(s1_1, s1_5);
  const int16x8_t s2_3 = vec_mergel(s1_1, s1_5);
  const int16x8_t s2_4 = vec_mergeh(s1_2, s1_6);
  const int16x8_t s2_5 = vec_mergel(s1_2, s1_6);
  const int16x8_t s2_6 = vec_mergeh(s1_3, s1_7);
  const int16x8_t s2_7 = vec_mergel(s1_3, s1_7);

  // Stage 2
  b[0] = vec_mergeh(s2_0, s2_4);
  b[1] = vec_mergel(s2_0, s2_4);
  b[2] = vec_mergeh(s2_1, s2_5);
  b[3] = vec_mergel(s2_1, s2_5);
  b[4] = vec_mergeh(s2_2, s2_6);
  b[5] = vec_mergel(s2_2, s2_6);
  b[6] = vec_mergeh(s2_3, s2_7);
  b[7] = vec_mergel(s2_3, s2_7);
}

#endif  // VPX_VPX_DSP_PPC_TRANSPOSE_VSX_H_
