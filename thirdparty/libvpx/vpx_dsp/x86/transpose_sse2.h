/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_X86_TRANSPOSE_SSE2_H_
#define VPX_VPX_DSP_X86_TRANSPOSE_SSE2_H_

#include <emmintrin.h>  // SSE2

#include "./vpx_config.h"

static INLINE __m128i transpose_8bit_4x4(const __m128i *const in) {
  // Unpack 8 bit elements. Goes from:
  // in[0]: 00 01 02 03
  // in[1]: 10 11 12 13
  // in[2]: 20 21 22 23
  // in[3]: 30 31 32 33
  // to:
  // a0:    00 10 01 11  02 12 03 13
  // a1:    20 30 21 31  22 32 23 33
  const __m128i a0 = _mm_unpacklo_epi8(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi8(in[2], in[3]);

  // Unpack 16 bit elements resulting in:
  // 00 10 20 30  01 11 21 31  02 12 22 32  03 13 23 33
  return _mm_unpacklo_epi16(a0, a1);
}

static INLINE void transpose_8bit_8x8(const __m128i *const in,
                                      __m128i *const out) {
  // Unpack 8 bit elements. Goes from:
  // in[0]: 00 01 02 03 04 05 06 07
  // in[1]: 10 11 12 13 14 15 16 17
  // in[2]: 20 21 22 23 24 25 26 27
  // in[3]: 30 31 32 33 34 35 36 37
  // in[4]: 40 41 42 43 44 45 46 47
  // in[5]: 50 51 52 53 54 55 56 57
  // in[6]: 60 61 62 63 64 65 66 67
  // in[7]: 70 71 72 73 74 75 76 77
  // to:
  // a0:    00 10 01 11 02 12 03 13  04 14 05 15 06 16 07 17
  // a1:    20 30 21 31 22 32 23 33  24 34 25 35 26 36 27 37
  // a2:    40 50 41 51 42 52 43 53  44 54 45 55 46 56 47 57
  // a3:    60 70 61 71 62 72 63 73  64 74 65 75 66 76 67 77
  const __m128i a0 = _mm_unpacklo_epi8(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi8(in[2], in[3]);
  const __m128i a2 = _mm_unpacklo_epi8(in[4], in[5]);
  const __m128i a3 = _mm_unpacklo_epi8(in[6], in[7]);

  // Unpack 16 bit elements resulting in:
  // b0: 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
  // b1: 40 50 60 70 41 51 61 71  42 52 62 72 43 53 63 73
  // b2: 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
  // b3: 44 54 64 74 45 55 65 75  46 56 66 76 47 57 67 77
  const __m128i b0 = _mm_unpacklo_epi16(a0, a1);
  const __m128i b1 = _mm_unpackhi_epi16(a0, a1);
  const __m128i b2 = _mm_unpacklo_epi16(a2, a3);
  const __m128i b3 = _mm_unpackhi_epi16(a2, a3);

  // Unpack 32 bit elements resulting in:
  // c0: 00 10 20 30 40 50 60 70  01 11 21 31 41 51 61 71
  // c1: 02 12 22 32 42 52 62 72  03 13 23 33 43 53 63 73
  // c2: 04 14 24 34 44 54 64 74  05 15 25 35 45 55 65 75
  // c3: 06 16 26 36 46 56 66 76  07 17 27 37 47 57 67 77
  const __m128i c0 = _mm_unpacklo_epi32(b0, b2);
  const __m128i c1 = _mm_unpackhi_epi32(b0, b2);
  const __m128i c2 = _mm_unpacklo_epi32(b1, b3);
  const __m128i c3 = _mm_unpackhi_epi32(b1, b3);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30 40 50 60 70
  // out[1]: 01 11 21 31 41 51 61 71
  // out[2]: 02 12 22 32 42 52 62 72
  // out[3]: 03 13 23 33 43 53 63 73
  // out[4]: 04 14 24 34 44 54 64 74
  // out[5]: 05 15 25 35 45 55 65 75
  // out[6]: 06 16 26 36 46 56 66 76
  // out[7]: 07 17 27 37 47 57 67 77
  out[0] = _mm_unpacklo_epi64(c0, c0);
  out[1] = _mm_unpackhi_epi64(c0, c0);
  out[2] = _mm_unpacklo_epi64(c1, c1);
  out[3] = _mm_unpackhi_epi64(c1, c1);
  out[4] = _mm_unpacklo_epi64(c2, c2);
  out[5] = _mm_unpackhi_epi64(c2, c2);
  out[6] = _mm_unpacklo_epi64(c3, c3);
  out[7] = _mm_unpackhi_epi64(c3, c3);
}

static INLINE void transpose_16bit_4x4(const __m128i *const in,
                                       __m128i *const out) {
  // Unpack 16 bit elements. Goes from:
  // in[0]: 00 01 02 03  XX XX XX XX
  // in[1]: 10 11 12 13  XX XX XX XX
  // in[2]: 20 21 22 23  XX XX XX XX
  // in[3]: 30 31 32 33  XX XX XX XX
  // to:
  // a0:    00 10 01 11  02 12 03 13
  // a1:    20 30 21 31  22 32 23 33
  const __m128i a0 = _mm_unpacklo_epi16(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi16(in[2], in[3]);

  // Unpack 32 bit elements resulting in:
  // out[0]: 00 10 20 30  01 11 21 31
  // out[1]: 02 12 22 32  03 13 23 33
  out[0] = _mm_unpacklo_epi32(a0, a1);
  out[1] = _mm_unpackhi_epi32(a0, a1);
}

static INLINE void transpose_16bit_4x8(const __m128i *const in,
                                       __m128i *const out) {
  // Unpack 16 bit elements. Goes from:
  // in[0]: 00 01 02 03  XX XX XX XX
  // in[1]: 10 11 12 13  XX XX XX XX
  // in[2]: 20 21 22 23  XX XX XX XX
  // in[3]: 30 31 32 33  XX XX XX XX
  // in[4]: 40 41 42 43  XX XX XX XX
  // in[5]: 50 51 52 53  XX XX XX XX
  // in[6]: 60 61 62 63  XX XX XX XX
  // in[7]: 70 71 72 73  XX XX XX XX
  // to:
  // a0:    00 10 01 11  02 12 03 13
  // a1:    20 30 21 31  22 32 23 33
  // a2:    40 50 41 51  42 52 43 53
  // a3:    60 70 61 71  62 72 63 73
  const __m128i a0 = _mm_unpacklo_epi16(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi16(in[2], in[3]);
  const __m128i a2 = _mm_unpacklo_epi16(in[4], in[5]);
  const __m128i a3 = _mm_unpacklo_epi16(in[6], in[7]);

  // Unpack 32 bit elements resulting in:
  // b0: 00 10 20 30  01 11 21 31
  // b1: 40 50 60 70  41 51 61 71
  // b2: 02 12 22 32  03 13 23 33
  // b3: 42 52 62 72  43 53 63 73
  const __m128i b0 = _mm_unpacklo_epi32(a0, a1);
  const __m128i b1 = _mm_unpacklo_epi32(a2, a3);
  const __m128i b2 = _mm_unpackhi_epi32(a0, a1);
  const __m128i b3 = _mm_unpackhi_epi32(a2, a3);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30  40 50 60 70
  // out[1]: 01 11 21 31  41 51 61 71
  // out[2]: 02 12 22 32  42 52 62 72
  // out[3]: 03 13 23 33  43 53 63 73
  out[0] = _mm_unpacklo_epi64(b0, b1);
  out[1] = _mm_unpackhi_epi64(b0, b1);
  out[2] = _mm_unpacklo_epi64(b2, b3);
  out[3] = _mm_unpackhi_epi64(b2, b3);
}

static INLINE void transpose_16bit_8x8(const __m128i *const in,
                                       __m128i *const out) {
  // Unpack 16 bit elements. Goes from:
  // in[0]: 00 01 02 03  04 05 06 07
  // in[1]: 10 11 12 13  14 15 16 17
  // in[2]: 20 21 22 23  24 25 26 27
  // in[3]: 30 31 32 33  34 35 36 37
  // in[4]: 40 41 42 43  44 45 46 47
  // in[5]: 50 51 52 53  54 55 56 57
  // in[6]: 60 61 62 63  64 65 66 67
  // in[7]: 70 71 72 73  74 75 76 77
  // to:
  // a0:    00 10 01 11  02 12 03 13
  // a1:    20 30 21 31  22 32 23 33
  // a2:    40 50 41 51  42 52 43 53
  // a3:    60 70 61 71  62 72 63 73
  // a4:    04 14 05 15  06 16 07 17
  // a5:    24 34 25 35  26 36 27 37
  // a6:    44 54 45 55  46 56 47 57
  // a7:    64 74 65 75  66 76 67 77
  const __m128i a0 = _mm_unpacklo_epi16(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi16(in[2], in[3]);
  const __m128i a2 = _mm_unpacklo_epi16(in[4], in[5]);
  const __m128i a3 = _mm_unpacklo_epi16(in[6], in[7]);
  const __m128i a4 = _mm_unpackhi_epi16(in[0], in[1]);
  const __m128i a5 = _mm_unpackhi_epi16(in[2], in[3]);
  const __m128i a6 = _mm_unpackhi_epi16(in[4], in[5]);
  const __m128i a7 = _mm_unpackhi_epi16(in[6], in[7]);

  // Unpack 32 bit elements resulting in:
  // b0: 00 10 20 30  01 11 21 31
  // b1: 40 50 60 70  41 51 61 71
  // b2: 04 14 24 34  05 15 25 35
  // b3: 44 54 64 74  45 55 65 75
  // b4: 02 12 22 32  03 13 23 33
  // b5: 42 52 62 72  43 53 63 73
  // b6: 06 16 26 36  07 17 27 37
  // b7: 46 56 66 76  47 57 67 77
  const __m128i b0 = _mm_unpacklo_epi32(a0, a1);
  const __m128i b1 = _mm_unpacklo_epi32(a2, a3);
  const __m128i b2 = _mm_unpacklo_epi32(a4, a5);
  const __m128i b3 = _mm_unpacklo_epi32(a6, a7);
  const __m128i b4 = _mm_unpackhi_epi32(a0, a1);
  const __m128i b5 = _mm_unpackhi_epi32(a2, a3);
  const __m128i b6 = _mm_unpackhi_epi32(a4, a5);
  const __m128i b7 = _mm_unpackhi_epi32(a6, a7);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30  40 50 60 70
  // out[1]: 01 11 21 31  41 51 61 71
  // out[2]: 02 12 22 32  42 52 62 72
  // out[3]: 03 13 23 33  43 53 63 73
  // out[4]: 04 14 24 34  44 54 64 74
  // out[5]: 05 15 25 35  45 55 65 75
  // out[6]: 06 16 26 36  46 56 66 76
  // out[7]: 07 17 27 37  47 57 67 77
  out[0] = _mm_unpacklo_epi64(b0, b1);
  out[1] = _mm_unpackhi_epi64(b0, b1);
  out[2] = _mm_unpacklo_epi64(b4, b5);
  out[3] = _mm_unpackhi_epi64(b4, b5);
  out[4] = _mm_unpacklo_epi64(b2, b3);
  out[5] = _mm_unpackhi_epi64(b2, b3);
  out[6] = _mm_unpacklo_epi64(b6, b7);
  out[7] = _mm_unpackhi_epi64(b6, b7);
}

// Transpose in-place
static INLINE void transpose_16bit_16x16(__m128i *const left,
                                         __m128i *const right) {
  __m128i tbuf[8];
  transpose_16bit_8x8(left, left);
  transpose_16bit_8x8(right, tbuf);
  transpose_16bit_8x8(left + 8, right);
  transpose_16bit_8x8(right + 8, right + 8);

  left[8] = tbuf[0];
  left[9] = tbuf[1];
  left[10] = tbuf[2];
  left[11] = tbuf[3];
  left[12] = tbuf[4];
  left[13] = tbuf[5];
  left[14] = tbuf[6];
  left[15] = tbuf[7];
}

static INLINE void transpose_32bit_4x4(const __m128i *const in,
                                       __m128i *const out) {
  // Unpack 32 bit elements. Goes from:
  // in[0]: 00 01 02 03
  // in[1]: 10 11 12 13
  // in[2]: 20 21 22 23
  // in[3]: 30 31 32 33
  // to:
  // a0:    00 10 01 11
  // a1:    20 30 21 31
  // a2:    02 12 03 13
  // a3:    22 32 23 33

  const __m128i a0 = _mm_unpacklo_epi32(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi32(in[2], in[3]);
  const __m128i a2 = _mm_unpackhi_epi32(in[0], in[1]);
  const __m128i a3 = _mm_unpackhi_epi32(in[2], in[3]);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30
  // out[1]: 01 11 21 31
  // out[2]: 02 12 22 32
  // out[3]: 03 13 23 33
  out[0] = _mm_unpacklo_epi64(a0, a1);
  out[1] = _mm_unpackhi_epi64(a0, a1);
  out[2] = _mm_unpacklo_epi64(a2, a3);
  out[3] = _mm_unpackhi_epi64(a2, a3);
}

static INLINE void transpose_32bit_4x4x2(const __m128i *const in,
                                         __m128i *const out) {
  // Unpack 32 bit elements. Goes from:
  // in[0]: 00 01 02 03
  // in[1]: 10 11 12 13
  // in[2]: 20 21 22 23
  // in[3]: 30 31 32 33
  // in[4]: 04 05 06 07
  // in[5]: 14 15 16 17
  // in[6]: 24 25 26 27
  // in[7]: 34 35 36 37
  // to:
  // a0:    00 10 01 11
  // a1:    20 30 21 31
  // a2:    02 12 03 13
  // a3:    22 32 23 33
  // a4:    04 14 05 15
  // a5:    24 34 25 35
  // a6:    06 16 07 17
  // a7:    26 36 27 37
  const __m128i a0 = _mm_unpacklo_epi32(in[0], in[1]);
  const __m128i a1 = _mm_unpacklo_epi32(in[2], in[3]);
  const __m128i a2 = _mm_unpackhi_epi32(in[0], in[1]);
  const __m128i a3 = _mm_unpackhi_epi32(in[2], in[3]);
  const __m128i a4 = _mm_unpacklo_epi32(in[4], in[5]);
  const __m128i a5 = _mm_unpacklo_epi32(in[6], in[7]);
  const __m128i a6 = _mm_unpackhi_epi32(in[4], in[5]);
  const __m128i a7 = _mm_unpackhi_epi32(in[6], in[7]);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30
  // out[1]: 01 11 21 31
  // out[2]: 02 12 22 32
  // out[3]: 03 13 23 33
  // out[4]: 04 14 24 34
  // out[5]: 05 15 25 35
  // out[6]: 06 16 26 36
  // out[7]: 07 17 27 37
  out[0] = _mm_unpacklo_epi64(a0, a1);
  out[1] = _mm_unpackhi_epi64(a0, a1);
  out[2] = _mm_unpacklo_epi64(a2, a3);
  out[3] = _mm_unpackhi_epi64(a2, a3);
  out[4] = _mm_unpacklo_epi64(a4, a5);
  out[5] = _mm_unpackhi_epi64(a4, a5);
  out[6] = _mm_unpacklo_epi64(a6, a7);
  out[7] = _mm_unpackhi_epi64(a6, a7);
}

static INLINE void transpose_32bit_8x4(const __m128i *const in,
                                       __m128i *const out) {
  // Unpack 32 bit elements. Goes from:
  // in[0]: 00 01 02 03
  // in[1]: 04 05 06 07
  // in[2]: 10 11 12 13
  // in[3]: 14 15 16 17
  // in[4]: 20 21 22 23
  // in[5]: 24 25 26 27
  // in[6]: 30 31 32 33
  // in[7]: 34 35 36 37
  // to:
  // a0: 00 10 01 11
  // a1: 20 30 21 31
  // a2: 02 12 03 13
  // a3: 22 32 23 33
  // a4: 04 14 05 15
  // a5: 24 34 25 35
  // a6: 06 16 07 17
  // a7: 26 36 27 37
  const __m128i a0 = _mm_unpacklo_epi32(in[0], in[2]);
  const __m128i a1 = _mm_unpacklo_epi32(in[4], in[6]);
  const __m128i a2 = _mm_unpackhi_epi32(in[0], in[2]);
  const __m128i a3 = _mm_unpackhi_epi32(in[4], in[6]);
  const __m128i a4 = _mm_unpacklo_epi32(in[1], in[3]);
  const __m128i a5 = _mm_unpacklo_epi32(in[5], in[7]);
  const __m128i a6 = _mm_unpackhi_epi32(in[1], in[3]);
  const __m128i a7 = _mm_unpackhi_epi32(in[5], in[7]);

  // Unpack 64 bit elements resulting in:
  // out[0]: 00 10 20 30
  // out[1]: 01 11 21 31
  // out[2]: 02 12 22 32
  // out[3]: 03 13 23 33
  // out[4]: 04 14 24 34
  // out[5]: 05 15 25 35
  // out[6]: 06 16 26 36
  // out[7]: 07 17 27 37
  out[0] = _mm_unpacklo_epi64(a0, a1);
  out[1] = _mm_unpackhi_epi64(a0, a1);
  out[2] = _mm_unpacklo_epi64(a2, a3);
  out[3] = _mm_unpackhi_epi64(a2, a3);
  out[4] = _mm_unpacklo_epi64(a4, a5);
  out[5] = _mm_unpackhi_epi64(a4, a5);
  out[6] = _mm_unpacklo_epi64(a6, a7);
  out[7] = _mm_unpackhi_epi64(a6, a7);
}

#endif  // VPX_VPX_DSP_X86_TRANSPOSE_SSE2_H_
