/*
 *  Copyright (c) 2016 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_TRANSPOSE_NEON_H_
#define VPX_VPX_DSP_ARM_TRANSPOSE_NEON_H_

#include <arm_neon.h>

#include "./vpx_config.h"

// Transpose 64 bit elements as follows:
// a0: 00 01 02 03 04 05 06 07
// a1: 16 17 18 19 20 21 22 23
//
// b0.val[0]: 00 01 02 03 16 17 18 19
// b0.val[1]: 04 05 06 07 20 21 22 23
static INLINE int16x8x2_t vpx_vtrnq_s64_to_s16(int32x4_t a0, int32x4_t a1) {
  int16x8x2_t b0;
#if VPX_ARCH_AARCH64
  b0.val[0] = vreinterpretq_s16_s64(
      vtrn1q_s64(vreinterpretq_s64_s32(a0), vreinterpretq_s64_s32(a1)));
  b0.val[1] = vreinterpretq_s16_s64(
      vtrn2q_s64(vreinterpretq_s64_s32(a0), vreinterpretq_s64_s32(a1)));
#else
  b0.val[0] = vcombine_s16(vreinterpret_s16_s32(vget_low_s32(a0)),
                           vreinterpret_s16_s32(vget_low_s32(a1)));
  b0.val[1] = vcombine_s16(vreinterpret_s16_s32(vget_high_s32(a0)),
                           vreinterpret_s16_s32(vget_high_s32(a1)));
#endif
  return b0;
}

static INLINE int32x4x2_t vpx_vtrnq_s64_to_s32(int32x4_t a0, int32x4_t a1) {
  int32x4x2_t b0;
#if VPX_ARCH_AARCH64
  b0.val[0] = vreinterpretq_s32_s64(
      vtrn1q_s64(vreinterpretq_s64_s32(a0), vreinterpretq_s64_s32(a1)));
  b0.val[1] = vreinterpretq_s32_s64(
      vtrn2q_s64(vreinterpretq_s64_s32(a0), vreinterpretq_s64_s32(a1)));
#else
  b0.val[0] = vcombine_s32(vget_low_s32(a0), vget_low_s32(a1));
  b0.val[1] = vcombine_s32(vget_high_s32(a0), vget_high_s32(a1));
#endif
  return b0;
}

static INLINE int64x2x2_t vpx_vtrnq_s64(int32x4_t a0, int32x4_t a1) {
  int64x2x2_t b0;
#if VPX_ARCH_AARCH64
  b0.val[0] = vtrn1q_s64(vreinterpretq_s64_s32(a0), vreinterpretq_s64_s32(a1));
  b0.val[1] = vtrn2q_s64(vreinterpretq_s64_s32(a0), vreinterpretq_s64_s32(a1));
#else
  b0.val[0] = vcombine_s64(vreinterpret_s64_s32(vget_low_s32(a0)),
                           vreinterpret_s64_s32(vget_low_s32(a1)));
  b0.val[1] = vcombine_s64(vreinterpret_s64_s32(vget_high_s32(a0)),
                           vreinterpret_s64_s32(vget_high_s32(a1)));
#endif
  return b0;
}

static INLINE uint8x16x2_t vpx_vtrnq_u64_to_u8(uint32x4_t a0, uint32x4_t a1) {
  uint8x16x2_t b0;
#if VPX_ARCH_AARCH64
  b0.val[0] = vreinterpretq_u8_u64(
      vtrn1q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a1)));
  b0.val[1] = vreinterpretq_u8_u64(
      vtrn2q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a1)));
#else
  b0.val[0] = vcombine_u8(vreinterpret_u8_u32(vget_low_u32(a0)),
                          vreinterpret_u8_u32(vget_low_u32(a1)));
  b0.val[1] = vcombine_u8(vreinterpret_u8_u32(vget_high_u32(a0)),
                          vreinterpret_u8_u32(vget_high_u32(a1)));
#endif
  return b0;
}

static INLINE uint16x8x2_t vpx_vtrnq_u64_to_u16(uint32x4_t a0, uint32x4_t a1) {
  uint16x8x2_t b0;
#if VPX_ARCH_AARCH64
  b0.val[0] = vreinterpretq_u16_u64(
      vtrn1q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a1)));
  b0.val[1] = vreinterpretq_u16_u64(
      vtrn2q_u64(vreinterpretq_u64_u32(a0), vreinterpretq_u64_u32(a1)));
#else
  b0.val[0] = vcombine_u16(vreinterpret_u16_u32(vget_low_u32(a0)),
                           vreinterpret_u16_u32(vget_low_u32(a1)));
  b0.val[1] = vcombine_u16(vreinterpret_u16_u32(vget_high_u32(a0)),
                           vreinterpret_u16_u32(vget_high_u32(a1)));
#endif
  return b0;
}

static INLINE void transpose_u8_4x4(uint8x8_t *a0, uint8x8_t *a1) {
  // Swap 16 bit elements. Goes from:
  // a0: 00 01 02 03  10 11 12 13
  // a1: 20 21 22 23  30 31 32 33
  // to:
  // b0.val[0]: 00 01 20 21  10 11 30 31
  // b0.val[1]: 02 03 22 23  12 13 32 33

  const uint16x4x2_t b0 =
      vtrn_u16(vreinterpret_u16_u8(*a0), vreinterpret_u16_u8(*a1));

  // Swap 32 bit elements resulting in:
  // c0.val[0]: 00 01 20 21  02 03 22 23
  // c0.val[1]: 10 11 30 31  12 13 32 33

  const uint32x2x2_t c0 = vtrn_u32(vreinterpret_u32_u16(b0.val[0]),
                                   vreinterpret_u32_u16(b0.val[1]));

  // Swap 8 bit elements resulting in:
  // d0.val[0]: 00 10 20 30  02 12 22 32
  // d0.val[1]: 01 11 21 31  03 13 23 33

  const uint8x8x2_t d0 =
      vtrn_u8(vreinterpret_u8_u32(c0.val[0]), vreinterpret_u8_u32(c0.val[1]));

  *a0 = d0.val[0];
  *a1 = d0.val[1];
}

static INLINE void transpose_s16_4x4d(int16x4_t *a0, int16x4_t *a1,
                                      int16x4_t *a2, int16x4_t *a3) {
  // Swap 16 bit elements. Goes from:
  // a0: 00 01 02 03
  // a1: 10 11 12 13
  // a2: 20 21 22 23
  // a3: 30 31 32 33
  // to:
  // b0.val[0]: 00 10 02 12
  // b0.val[1]: 01 11 03 13
  // b1.val[0]: 20 30 22 32
  // b1.val[1]: 21 31 23 33

  const int16x4x2_t b0 = vtrn_s16(*a0, *a1);
  const int16x4x2_t b1 = vtrn_s16(*a2, *a3);

  // Swap 32 bit elements resulting in:
  // c0.val[0]: 00 10 20 30
  // c0.val[1]: 02 12 22 32
  // c1.val[0]: 01 11 21 31
  // c1.val[1]: 03 13 23 33

  const int32x2x2_t c0 = vtrn_s32(vreinterpret_s32_s16(b0.val[0]),
                                  vreinterpret_s32_s16(b1.val[0]));
  const int32x2x2_t c1 = vtrn_s32(vreinterpret_s32_s16(b0.val[1]),
                                  vreinterpret_s32_s16(b1.val[1]));

  *a0 = vreinterpret_s16_s32(c0.val[0]);
  *a1 = vreinterpret_s16_s32(c1.val[0]);
  *a2 = vreinterpret_s16_s32(c0.val[1]);
  *a3 = vreinterpret_s16_s32(c1.val[1]);
}

static INLINE void transpose_s16_4x4q(int16x8_t *a0, int16x8_t *a1) {
  // Swap 32 bit elements. Goes from:
  // a0: 00 01 02 03  10 11 12 13
  // a1: 20 21 22 23  30 31 32 33
  // to:
  // b0.val[0]: 00 01 20 21  10 11 30 31
  // b0.val[1]: 02 03 22 23  12 13 32 33

  const int32x4x2_t b0 =
      vtrnq_s32(vreinterpretq_s32_s16(*a0), vreinterpretq_s32_s16(*a1));

  // Swap 64 bit elements resulting in:
  // c0: 00 01 20 21  02 03 22 23
  // c1: 10 11 30 31  12 13 32 33

  const int16x8x2_t c0 = vpx_vtrnq_s64_to_s16(b0.val[0], b0.val[1]);

  // Swap 16 bit elements resulting in:
  // d0.val[0]: 00 10 20 30  02 12 22 32
  // d0.val[1]: 01 11 21 31  03 13 23 33

  const int16x8x2_t d0 = vtrnq_s16(c0.val[0], c0.val[1]);

  *a0 = d0.val[0];
  *a1 = d0.val[1];
}

static INLINE void transpose_u16_4x4q(uint16x8_t *a0, uint16x8_t *a1) {
  // Swap 32 bit elements. Goes from:
  // a0: 00 01 02 03  10 11 12 13
  // a1: 20 21 22 23  30 31 32 33
  // to:
  // b0.val[0]: 00 01 20 21  10 11 30 31
  // b0.val[1]: 02 03 22 23  12 13 32 33

  const uint32x4x2_t b0 =
      vtrnq_u32(vreinterpretq_u32_u16(*a0), vreinterpretq_u32_u16(*a1));

  // Swap 64 bit elements resulting in:
  // c0: 00 01 20 21  02 03 22 23
  // c1: 10 11 30 31  12 13 32 33

  const uint16x8x2_t c0 = vpx_vtrnq_u64_to_u16(b0.val[0], b0.val[1]);

  // Swap 16 bit elements resulting in:
  // d0.val[0]: 00 10 20 30  02 12 22 32
  // d0.val[1]: 01 11 21 31  03 13 23 33

  const uint16x8x2_t d0 = vtrnq_u16(c0.val[0], c0.val[1]);

  *a0 = d0.val[0];
  *a1 = d0.val[1];
}

static INLINE void transpose_u8_4x8(uint8x8_t *a0, uint8x8_t *a1, uint8x8_t *a2,
                                    uint8x8_t *a3, const uint8x8_t a4,
                                    const uint8x8_t a5, const uint8x8_t a6,
                                    const uint8x8_t a7) {
  // Swap 32 bit elements. Goes from:
  // a0: 00 01 02 03 XX XX XX XX
  // a1: 10 11 12 13 XX XX XX XX
  // a2: 20 21 22 23 XX XX XX XX
  // a3; 30 31 32 33 XX XX XX XX
  // a4: 40 41 42 43 XX XX XX XX
  // a5: 50 51 52 53 XX XX XX XX
  // a6: 60 61 62 63 XX XX XX XX
  // a7: 70 71 72 73 XX XX XX XX
  // to:
  // b0.val[0]: 00 01 02 03 40 41 42 43
  // b1.val[0]: 10 11 12 13 50 51 52 53
  // b2.val[0]: 20 21 22 23 60 61 62 63
  // b3.val[0]: 30 31 32 33 70 71 72 73

  const uint32x2x2_t b0 =
      vtrn_u32(vreinterpret_u32_u8(*a0), vreinterpret_u32_u8(a4));
  const uint32x2x2_t b1 =
      vtrn_u32(vreinterpret_u32_u8(*a1), vreinterpret_u32_u8(a5));
  const uint32x2x2_t b2 =
      vtrn_u32(vreinterpret_u32_u8(*a2), vreinterpret_u32_u8(a6));
  const uint32x2x2_t b3 =
      vtrn_u32(vreinterpret_u32_u8(*a3), vreinterpret_u32_u8(a7));

  // Swap 16 bit elements resulting in:
  // c0.val[0]: 00 01 20 21 40 41 60 61
  // c0.val[1]: 02 03 22 23 42 43 62 63
  // c1.val[0]: 10 11 30 31 50 51 70 71
  // c1.val[1]: 12 13 32 33 52 53 72 73

  const uint16x4x2_t c0 = vtrn_u16(vreinterpret_u16_u32(b0.val[0]),
                                   vreinterpret_u16_u32(b2.val[0]));
  const uint16x4x2_t c1 = vtrn_u16(vreinterpret_u16_u32(b1.val[0]),
                                   vreinterpret_u16_u32(b3.val[0]));

  // Swap 8 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70
  // d0.val[1]: 01 11 21 31 41 51 61 71
  // d1.val[0]: 02 12 22 32 42 52 62 72
  // d1.val[1]: 03 13 23 33 43 53 63 73

  const uint8x8x2_t d0 =
      vtrn_u8(vreinterpret_u8_u16(c0.val[0]), vreinterpret_u8_u16(c1.val[0]));
  const uint8x8x2_t d1 =
      vtrn_u8(vreinterpret_u8_u16(c0.val[1]), vreinterpret_u8_u16(c1.val[1]));

  *a0 = d0.val[0];
  *a1 = d0.val[1];
  *a2 = d1.val[0];
  *a3 = d1.val[1];
}

static INLINE void transpose_s32_4x4(int32x4_t *a0, int32x4_t *a1,
                                     int32x4_t *a2, int32x4_t *a3) {
  // Swap 32 bit elements. Goes from:
  // a0: 00 01 02 03
  // a1: 10 11 12 13
  // a2: 20 21 22 23
  // a3: 30 31 32 33
  // to:
  // b0.val[0]: 00 10 02 12
  // b0.val[1]: 01 11 03 13
  // b1.val[0]: 20 30 22 32
  // b1.val[1]: 21 31 23 33

  const int32x4x2_t b0 = vtrnq_s32(*a0, *a1);
  const int32x4x2_t b1 = vtrnq_s32(*a2, *a3);

  // Swap 64 bit elements resulting in:
  // c0.val[0]: 00 10 20 30
  // c0.val[1]: 02 12 22 32
  // c1.val[0]: 01 11 21 31
  // c1.val[1]: 03 13 23 33

  const int32x4x2_t c0 = vpx_vtrnq_s64_to_s32(b0.val[0], b1.val[0]);
  const int32x4x2_t c1 = vpx_vtrnq_s64_to_s32(b0.val[1], b1.val[1]);

  *a0 = c0.val[0];
  *a1 = c1.val[0];
  *a2 = c0.val[1];
  *a3 = c1.val[1];
}

static INLINE void transpose_s16_4x8(const int16x4_t a0, const int16x4_t a1,
                                     const int16x4_t a2, const int16x4_t a3,
                                     const int16x4_t a4, const int16x4_t a5,
                                     const int16x4_t a6, const int16x4_t a7,
                                     int16x8_t *const o0, int16x8_t *const o1,
                                     int16x8_t *const o2, int16x8_t *const o3) {
  // Combine rows. Goes from:
  // a0: 00 01 02 03
  // a1: 10 11 12 13
  // a2: 20 21 22 23
  // a3: 30 31 32 33
  // a4: 40 41 42 43
  // a5: 50 51 52 53
  // a6: 60 61 62 63
  // a7: 70 71 72 73
  // to:
  // b0: 00 01 02 03 40 41 42 43
  // b1: 10 11 12 13 50 51 52 53
  // b2: 20 21 22 23 60 61 62 63
  // b3: 30 31 32 33 70 71 72 73

  const int16x8_t b0 = vcombine_s16(a0, a4);
  const int16x8_t b1 = vcombine_s16(a1, a5);
  const int16x8_t b2 = vcombine_s16(a2, a6);
  const int16x8_t b3 = vcombine_s16(a3, a7);

  // Swap 16 bit elements resulting in:
  // c0.val[0]: 00 10 02 12 40 50 42 52
  // c0.val[1]: 01 11 03 13 41 51 43 53
  // c1.val[0]: 20 30 22 32 60 70 62 72
  // c1.val[1]: 21 31 23 33 61 71 63 73

  const int16x8x2_t c0 = vtrnq_s16(b0, b1);
  const int16x8x2_t c1 = vtrnq_s16(b2, b3);

  // Swap 32 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70
  // d0.val[1]: 02 12 22 32 42 52 62 72
  // d1.val[0]: 01 11 21 31 41 51 61 71
  // d1.val[1]: 03 13 23 33 43 53 63 73

  const int32x4x2_t d0 = vtrnq_s32(vreinterpretq_s32_s16(c0.val[0]),
                                   vreinterpretq_s32_s16(c1.val[0]));
  const int32x4x2_t d1 = vtrnq_s32(vreinterpretq_s32_s16(c0.val[1]),
                                   vreinterpretq_s32_s16(c1.val[1]));

  *o0 = vreinterpretq_s16_s32(d0.val[0]);
  *o1 = vreinterpretq_s16_s32(d1.val[0]);
  *o2 = vreinterpretq_s16_s32(d0.val[1]);
  *o3 = vreinterpretq_s16_s32(d1.val[1]);
}

static INLINE void transpose_s32_4x8(int32x4_t *const a0, int32x4_t *const a1,
                                     int32x4_t *const a2, int32x4_t *const a3,
                                     int32x4_t *const a4, int32x4_t *const a5,
                                     int32x4_t *const a6, int32x4_t *const a7) {
  // Swap 32 bit elements. Goes from:
  // a0: 00 01 02 03
  // a1: 10 11 12 13
  // a2: 20 21 22 23
  // a3: 30 31 32 33
  // a4: 40 41 42 43
  // a5: 50 51 52 53
  // a6: 60 61 62 63
  // a7: 70 71 72 73
  // to:
  // b0.val[0]: 00 10 02 12
  // b0.val[1]: 01 11 03 13
  // b1.val[0]: 20 30 22 32
  // b1.val[1]: 21 31 23 33
  // b2.val[0]: 40 50 42 52
  // b2.val[1]: 41 51 43 53
  // b3.val[0]: 60 70 62 72
  // b3.val[1]: 61 71 63 73

  const int32x4x2_t b0 = vtrnq_s32(*a0, *a1);
  const int32x4x2_t b1 = vtrnq_s32(*a2, *a3);
  const int32x4x2_t b2 = vtrnq_s32(*a4, *a5);
  const int32x4x2_t b3 = vtrnq_s32(*a6, *a7);

  // Swap 64 bit elements resulting in:
  // c0.val[0]: 00 10 20 30
  // c0.val[1]: 02 12 22 32
  // c1.val[0]: 01 11 21 31
  // c1.val[1]: 03 13 23 33
  // c2.val[0]: 40 50 60 70
  // c2.val[1]: 42 52 62 72
  // c3.val[0]: 41 51 61 71
  // c3.val[1]: 43 53 63 73

  const int64x2x2_t c0 = vpx_vtrnq_s64(b0.val[0], b1.val[0]);
  const int64x2x2_t c1 = vpx_vtrnq_s64(b0.val[1], b1.val[1]);
  const int64x2x2_t c2 = vpx_vtrnq_s64(b2.val[0], b3.val[0]);
  const int64x2x2_t c3 = vpx_vtrnq_s64(b2.val[1], b3.val[1]);

  *a0 = vreinterpretq_s32_s64(c0.val[0]);
  *a1 = vreinterpretq_s32_s64(c2.val[0]);
  *a2 = vreinterpretq_s32_s64(c1.val[0]);
  *a3 = vreinterpretq_s32_s64(c3.val[0]);
  *a4 = vreinterpretq_s32_s64(c0.val[1]);
  *a5 = vreinterpretq_s32_s64(c2.val[1]);
  *a6 = vreinterpretq_s32_s64(c1.val[1]);
  *a7 = vreinterpretq_s32_s64(c3.val[1]);
}

static INLINE void transpose_u8_8x4(uint8x8_t *a0, uint8x8_t *a1, uint8x8_t *a2,
                                    uint8x8_t *a3) {
  // Swap 8 bit elements. Goes from:
  // a0: 00 01 02 03 04 05 06 07
  // a1: 10 11 12 13 14 15 16 17
  // a2: 20 21 22 23 24 25 26 27
  // a3: 30 31 32 33 34 35 36 37
  // to:
  // b0.val[0]: 00 10 02 12 04 14 06 16
  // b0.val[1]: 01 11 03 13 05 15 07 17
  // b1.val[0]: 20 30 22 32 24 34 26 36
  // b1.val[1]: 21 31 23 33 25 35 27 37

  const uint8x8x2_t b0 = vtrn_u8(*a0, *a1);
  const uint8x8x2_t b1 = vtrn_u8(*a2, *a3);

  // Swap 16 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34
  // c0.val[1]: 02 12 22 32 06 16 26 36
  // c1.val[0]: 01 11 21 31 05 15 25 35
  // c1.val[1]: 03 13 23 33 07 17 27 37

  const uint16x4x2_t c0 =
      vtrn_u16(vreinterpret_u16_u8(b0.val[0]), vreinterpret_u16_u8(b1.val[0]));
  const uint16x4x2_t c1 =
      vtrn_u16(vreinterpret_u16_u8(b0.val[1]), vreinterpret_u16_u8(b1.val[1]));

  *a0 = vreinterpret_u8_u16(c0.val[0]);
  *a1 = vreinterpret_u8_u16(c1.val[0]);
  *a2 = vreinterpret_u8_u16(c0.val[1]);
  *a3 = vreinterpret_u8_u16(c1.val[1]);
}

static INLINE void transpose_u16_8x4(uint16x8_t *a0, uint16x8_t *a1,
                                     uint16x8_t *a2, uint16x8_t *a3) {
  // Swap 16 bit elements. Goes from:
  // a0: 00 01 02 03 04 05 06 07
  // a1: 10 11 12 13 14 15 16 17
  // a2: 20 21 22 23 24 25 26 27
  // a3: 30 31 32 33 34 35 36 37
  // to:
  // b0.val[0]: 00 10 02 12 04 14 06 16
  // b0.val[1]: 01 11 03 13 05 15 07 17
  // b1.val[0]: 20 30 22 32 24 34 26 36
  // b1.val[1]: 21 31 23 33 25 35 27 37

  const uint16x8x2_t b0 = vtrnq_u16(*a0, *a1);
  const uint16x8x2_t b1 = vtrnq_u16(*a2, *a3);

  // Swap 32 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34
  // c0.val[1]: 02 12 22 32 06 16 26 36
  // c1.val[0]: 01 11 21 31 05 15 25 35
  // c1.val[1]: 03 13 23 33 07 17 27 37

  const uint32x4x2_t c0 = vtrnq_u32(vreinterpretq_u32_u16(b0.val[0]),
                                    vreinterpretq_u32_u16(b1.val[0]));
  const uint32x4x2_t c1 = vtrnq_u32(vreinterpretq_u32_u16(b0.val[1]),
                                    vreinterpretq_u32_u16(b1.val[1]));

  *a0 = vreinterpretq_u16_u32(c0.val[0]);
  *a1 = vreinterpretq_u16_u32(c1.val[0]);
  *a2 = vreinterpretq_u16_u32(c0.val[1]);
  *a3 = vreinterpretq_u16_u32(c1.val[1]);
}

static INLINE void transpose_s32_8x4(int32x4_t *const a0, int32x4_t *const a1,
                                     int32x4_t *const a2, int32x4_t *const a3,
                                     int32x4_t *const a4, int32x4_t *const a5,
                                     int32x4_t *const a6, int32x4_t *const a7) {
  // Swap 32 bit elements. Goes from:
  // a0: 00 01 02 03
  // a1: 04 05 06 07
  // a2: 10 11 12 13
  // a3: 14 15 16 17
  // a4: 20 21 22 23
  // a5: 24 25 26 27
  // a6: 30 31 32 33
  // a7: 34 35 36 37
  // to:
  // b0.val[0]: 00 10 02 12
  // b0.val[1]: 01 11 03 13
  // b1.val[0]: 04 14 06 16
  // b1.val[1]: 05 15 07 17
  // b2.val[0]: 20 30 22 32
  // b2.val[1]: 21 31 23 33
  // b3.val[0]: 24 34 26 36
  // b3.val[1]: 25 35 27 37

  const int32x4x2_t b0 = vtrnq_s32(*a0, *a2);
  const int32x4x2_t b1 = vtrnq_s32(*a1, *a3);
  const int32x4x2_t b2 = vtrnq_s32(*a4, *a6);
  const int32x4x2_t b3 = vtrnq_s32(*a5, *a7);

  // Swap 64 bit elements resulting in:
  // c0.val[0]: 00 10 20 30
  // c0.val[1]: 02 12 22 32
  // c1.val[0]: 01 11 21 31
  // c1.val[1]: 03 13 23 33
  // c2.val[0]: 04 14 24 34
  // c2.val[1]: 06 16 26 36
  // c3.val[0]: 05 15 25 35
  // c3.val[1]: 07 17 27 37

  const int64x2x2_t c0 = vpx_vtrnq_s64(b0.val[0], b2.val[0]);
  const int64x2x2_t c1 = vpx_vtrnq_s64(b0.val[1], b2.val[1]);
  const int64x2x2_t c2 = vpx_vtrnq_s64(b1.val[0], b3.val[0]);
  const int64x2x2_t c3 = vpx_vtrnq_s64(b1.val[1], b3.val[1]);

  *a0 = vreinterpretq_s32_s64(c0.val[0]);
  *a1 = vreinterpretq_s32_s64(c1.val[0]);
  *a2 = vreinterpretq_s32_s64(c0.val[1]);
  *a3 = vreinterpretq_s32_s64(c1.val[1]);
  *a4 = vreinterpretq_s32_s64(c2.val[0]);
  *a5 = vreinterpretq_s32_s64(c3.val[0]);
  *a6 = vreinterpretq_s32_s64(c2.val[1]);
  *a7 = vreinterpretq_s32_s64(c3.val[1]);
}

static INLINE void transpose_u8_8x8(uint8x8_t *a0, uint8x8_t *a1, uint8x8_t *a2,
                                    uint8x8_t *a3, uint8x8_t *a4, uint8x8_t *a5,
                                    uint8x8_t *a6, uint8x8_t *a7) {
  // Widen to 128-bit registers (usually a no-op once inlined.)
  const uint8x16_t a0q = vcombine_u8(*a0, vdup_n_u8(0));
  const uint8x16_t a1q = vcombine_u8(*a1, vdup_n_u8(0));
  const uint8x16_t a2q = vcombine_u8(*a2, vdup_n_u8(0));
  const uint8x16_t a3q = vcombine_u8(*a3, vdup_n_u8(0));
  const uint8x16_t a4q = vcombine_u8(*a4, vdup_n_u8(0));
  const uint8x16_t a5q = vcombine_u8(*a5, vdup_n_u8(0));
  const uint8x16_t a6q = vcombine_u8(*a6, vdup_n_u8(0));
  const uint8x16_t a7q = vcombine_u8(*a7, vdup_n_u8(0));

  // Zip 8 bit elements. Goes from:
  // a0: 00 01 02 03 04 05 06 07
  // a1: 10 11 12 13 14 15 16 17
  // a2: 20 21 22 23 24 25 26 27
  // a3: 30 31 32 33 34 35 36 37
  // a4: 40 41 42 43 44 45 46 47
  // a5: 50 51 52 53 54 55 56 57
  // a6: 60 61 62 63 64 65 66 67
  // a7: 70 71 72 73 74 75 76 77
  // to:
  // b0: 00 10 01 11 02 12 03 13  04 14 05 15 06 16 07 17
  // b1: 20 30 21 31 22 32 23 33  24 34 25 35 26 36 27 37
  // b2: 40 50 41 51 42 52 43 53  44 54 45 55 46 56 47 57
  // b3: 60 70 61 71 62 72 63 73  64 74 65 75 66 76 67 77
  const uint8x16_t b0 = vzipq_u8(a0q, a1q).val[0];
  const uint8x16_t b1 = vzipq_u8(a2q, a3q).val[0];
  const uint8x16_t b2 = vzipq_u8(a4q, a5q).val[0];
  const uint8x16_t b3 = vzipq_u8(a6q, a7q).val[0];

  // Zip 16 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
  // c0.val[1]: 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
  // c1.val[0]: 40 50 60 70 41 51 61 71  42 52 62 72 43 53 63 73
  // c1.val[1]: 44 54 64 74 45 55 65 75  46 66 56 76 47 67 57 77
  const uint16x8x2_t c0 =
      vzipq_u16(vreinterpretq_u16_u8(b0), vreinterpretq_u16_u8(b1));
  const uint16x8x2_t c1 =
      vzipq_u16(vreinterpretq_u16_u8(b2), vreinterpretq_u16_u8(b3));

  // Zip 32 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70  01 11 21 31 41 51 61 71
  // d0.val[1]: 02 12 22 32 42 52 62 72  03 13 23 33 43 53 63 73
  // d1.val[0]: 04 14 24 34 44 54 64 74  05 15 25 35 45 55 65 75
  // d1.val[1]: 06 16 26 36 46 56 66 76  07 17 27 37 47 57 67 77
  const uint32x4x2_t d0 = vzipq_u32(vreinterpretq_u32_u16(c0.val[0]),
                                    vreinterpretq_u32_u16(c1.val[0]));
  const uint32x4x2_t d1 = vzipq_u32(vreinterpretq_u32_u16(c0.val[1]),
                                    vreinterpretq_u32_u16(c1.val[1]));

  *a0 = vreinterpret_u8_u32(vget_low_u32(d0.val[0]));
  *a1 = vreinterpret_u8_u32(vget_high_u32(d0.val[0]));
  *a2 = vreinterpret_u8_u32(vget_low_u32(d0.val[1]));
  *a3 = vreinterpret_u8_u32(vget_high_u32(d0.val[1]));
  *a4 = vreinterpret_u8_u32(vget_low_u32(d1.val[0]));
  *a5 = vreinterpret_u8_u32(vget_high_u32(d1.val[0]));
  *a6 = vreinterpret_u8_u32(vget_low_u32(d1.val[1]));
  *a7 = vreinterpret_u8_u32(vget_high_u32(d1.val[1]));
}

// Transpose 8x8 to a new location.
static INLINE void transpose_s16_8x8q(int16x8_t *a, int16x8_t *out) {
  // Swap 16 bit elements. Goes from:
  // a0: 00 01 02 03 04 05 06 07
  // a1: 10 11 12 13 14 15 16 17
  // a2: 20 21 22 23 24 25 26 27
  // a3: 30 31 32 33 34 35 36 37
  // a4: 40 41 42 43 44 45 46 47
  // a5: 50 51 52 53 54 55 56 57
  // a6: 60 61 62 63 64 65 66 67
  // a7: 70 71 72 73 74 75 76 77
  // to:
  // b0.val[0]: 00 10 02 12 04 14 06 16
  // b0.val[1]: 01 11 03 13 05 15 07 17
  // b1.val[0]: 20 30 22 32 24 34 26 36
  // b1.val[1]: 21 31 23 33 25 35 27 37
  // b2.val[0]: 40 50 42 52 44 54 46 56
  // b2.val[1]: 41 51 43 53 45 55 47 57
  // b3.val[0]: 60 70 62 72 64 74 66 76
  // b3.val[1]: 61 71 63 73 65 75 67 77

  const int16x8x2_t b0 = vtrnq_s16(a[0], a[1]);
  const int16x8x2_t b1 = vtrnq_s16(a[2], a[3]);
  const int16x8x2_t b2 = vtrnq_s16(a[4], a[5]);
  const int16x8x2_t b3 = vtrnq_s16(a[6], a[7]);

  // Swap 32 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34
  // c0.val[1]: 02 12 22 32 06 16 26 36
  // c1.val[0]: 01 11 21 31 05 15 25 35
  // c1.val[1]: 03 13 23 33 07 17 27 37
  // c2.val[0]: 40 50 60 70 44 54 64 74
  // c2.val[1]: 42 52 62 72 46 56 66 76
  // c3.val[0]: 41 51 61 71 45 55 65 75
  // c3.val[1]: 43 53 63 73 47 57 67 77

  const int32x4x2_t c0 = vtrnq_s32(vreinterpretq_s32_s16(b0.val[0]),
                                   vreinterpretq_s32_s16(b1.val[0]));
  const int32x4x2_t c1 = vtrnq_s32(vreinterpretq_s32_s16(b0.val[1]),
                                   vreinterpretq_s32_s16(b1.val[1]));
  const int32x4x2_t c2 = vtrnq_s32(vreinterpretq_s32_s16(b2.val[0]),
                                   vreinterpretq_s32_s16(b3.val[0]));
  const int32x4x2_t c3 = vtrnq_s32(vreinterpretq_s32_s16(b2.val[1]),
                                   vreinterpretq_s32_s16(b3.val[1]));

  // Swap 64 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70
  // d0.val[1]: 04 14 24 34 44 54 64 74
  // d1.val[0]: 01 11 21 31 41 51 61 71
  // d1.val[1]: 05 15 25 35 45 55 65 75
  // d2.val[0]: 02 12 22 32 42 52 62 72
  // d2.val[1]: 06 16 26 36 46 56 66 76
  // d3.val[0]: 03 13 23 33 43 53 63 73
  // d3.val[1]: 07 17 27 37 47 57 67 77

  const int16x8x2_t d0 = vpx_vtrnq_s64_to_s16(c0.val[0], c2.val[0]);
  const int16x8x2_t d1 = vpx_vtrnq_s64_to_s16(c1.val[0], c3.val[0]);
  const int16x8x2_t d2 = vpx_vtrnq_s64_to_s16(c0.val[1], c2.val[1]);
  const int16x8x2_t d3 = vpx_vtrnq_s64_to_s16(c1.val[1], c3.val[1]);

  out[0] = d0.val[0];
  out[1] = d1.val[0];
  out[2] = d2.val[0];
  out[3] = d3.val[0];
  out[4] = d0.val[1];
  out[5] = d1.val[1];
  out[6] = d2.val[1];
  out[7] = d3.val[1];
}

static INLINE void transpose_s16_8x8(int16x8_t *a0, int16x8_t *a1,
                                     int16x8_t *a2, int16x8_t *a3,
                                     int16x8_t *a4, int16x8_t *a5,
                                     int16x8_t *a6, int16x8_t *a7) {
  // Swap 16 bit elements. Goes from:
  // a0: 00 01 02 03 04 05 06 07
  // a1: 10 11 12 13 14 15 16 17
  // a2: 20 21 22 23 24 25 26 27
  // a3: 30 31 32 33 34 35 36 37
  // a4: 40 41 42 43 44 45 46 47
  // a5: 50 51 52 53 54 55 56 57
  // a6: 60 61 62 63 64 65 66 67
  // a7: 70 71 72 73 74 75 76 77
  // to:
  // b0.val[0]: 00 10 02 12 04 14 06 16
  // b0.val[1]: 01 11 03 13 05 15 07 17
  // b1.val[0]: 20 30 22 32 24 34 26 36
  // b1.val[1]: 21 31 23 33 25 35 27 37
  // b2.val[0]: 40 50 42 52 44 54 46 56
  // b2.val[1]: 41 51 43 53 45 55 47 57
  // b3.val[0]: 60 70 62 72 64 74 66 76
  // b3.val[1]: 61 71 63 73 65 75 67 77

  const int16x8x2_t b0 = vtrnq_s16(*a0, *a1);
  const int16x8x2_t b1 = vtrnq_s16(*a2, *a3);
  const int16x8x2_t b2 = vtrnq_s16(*a4, *a5);
  const int16x8x2_t b3 = vtrnq_s16(*a6, *a7);

  // Swap 32 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34
  // c0.val[1]: 02 12 22 32 06 16 26 36
  // c1.val[0]: 01 11 21 31 05 15 25 35
  // c1.val[1]: 03 13 23 33 07 17 27 37
  // c2.val[0]: 40 50 60 70 44 54 64 74
  // c2.val[1]: 42 52 62 72 46 56 66 76
  // c3.val[0]: 41 51 61 71 45 55 65 75
  // c3.val[1]: 43 53 63 73 47 57 67 77

  const int32x4x2_t c0 = vtrnq_s32(vreinterpretq_s32_s16(b0.val[0]),
                                   vreinterpretq_s32_s16(b1.val[0]));
  const int32x4x2_t c1 = vtrnq_s32(vreinterpretq_s32_s16(b0.val[1]),
                                   vreinterpretq_s32_s16(b1.val[1]));
  const int32x4x2_t c2 = vtrnq_s32(vreinterpretq_s32_s16(b2.val[0]),
                                   vreinterpretq_s32_s16(b3.val[0]));
  const int32x4x2_t c3 = vtrnq_s32(vreinterpretq_s32_s16(b2.val[1]),
                                   vreinterpretq_s32_s16(b3.val[1]));

  // Swap 64 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70
  // d0.val[1]: 04 14 24 34 44 54 64 74
  // d1.val[0]: 01 11 21 31 41 51 61 71
  // d1.val[1]: 05 15 25 35 45 55 65 75
  // d2.val[0]: 02 12 22 32 42 52 62 72
  // d2.val[1]: 06 16 26 36 46 56 66 76
  // d3.val[0]: 03 13 23 33 43 53 63 73
  // d3.val[1]: 07 17 27 37 47 57 67 77

  const int16x8x2_t d0 = vpx_vtrnq_s64_to_s16(c0.val[0], c2.val[0]);
  const int16x8x2_t d1 = vpx_vtrnq_s64_to_s16(c1.val[0], c3.val[0]);
  const int16x8x2_t d2 = vpx_vtrnq_s64_to_s16(c0.val[1], c2.val[1]);
  const int16x8x2_t d3 = vpx_vtrnq_s64_to_s16(c1.val[1], c3.val[1]);

  *a0 = d0.val[0];
  *a1 = d1.val[0];
  *a2 = d2.val[0];
  *a3 = d3.val[0];
  *a4 = d0.val[1];
  *a5 = d1.val[1];
  *a6 = d2.val[1];
  *a7 = d3.val[1];
}

static INLINE void transpose_u16_8x8(uint16x8_t *a0, uint16x8_t *a1,
                                     uint16x8_t *a2, uint16x8_t *a3,
                                     uint16x8_t *a4, uint16x8_t *a5,
                                     uint16x8_t *a6, uint16x8_t *a7) {
  // Swap 16 bit elements. Goes from:
  // a0: 00 01 02 03 04 05 06 07
  // a1: 10 11 12 13 14 15 16 17
  // a2: 20 21 22 23 24 25 26 27
  // a3: 30 31 32 33 34 35 36 37
  // a4: 40 41 42 43 44 45 46 47
  // a5: 50 51 52 53 54 55 56 57
  // a6: 60 61 62 63 64 65 66 67
  // a7: 70 71 72 73 74 75 76 77
  // to:
  // b0.val[0]: 00 10 02 12 04 14 06 16
  // b0.val[1]: 01 11 03 13 05 15 07 17
  // b1.val[0]: 20 30 22 32 24 34 26 36
  // b1.val[1]: 21 31 23 33 25 35 27 37
  // b2.val[0]: 40 50 42 52 44 54 46 56
  // b2.val[1]: 41 51 43 53 45 55 47 57
  // b3.val[0]: 60 70 62 72 64 74 66 76
  // b3.val[1]: 61 71 63 73 65 75 67 77

  const uint16x8x2_t b0 = vtrnq_u16(*a0, *a1);
  const uint16x8x2_t b1 = vtrnq_u16(*a2, *a3);
  const uint16x8x2_t b2 = vtrnq_u16(*a4, *a5);
  const uint16x8x2_t b3 = vtrnq_u16(*a6, *a7);

  // Swap 32 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34
  // c0.val[1]: 02 12 22 32 06 16 26 36
  // c1.val[0]: 01 11 21 31 05 15 25 35
  // c1.val[1]: 03 13 23 33 07 17 27 37
  // c2.val[0]: 40 50 60 70 44 54 64 74
  // c2.val[1]: 42 52 62 72 46 56 66 76
  // c3.val[0]: 41 51 61 71 45 55 65 75
  // c3.val[1]: 43 53 63 73 47 57 67 77

  const uint32x4x2_t c0 = vtrnq_u32(vreinterpretq_u32_u16(b0.val[0]),
                                    vreinterpretq_u32_u16(b1.val[0]));
  const uint32x4x2_t c1 = vtrnq_u32(vreinterpretq_u32_u16(b0.val[1]),
                                    vreinterpretq_u32_u16(b1.val[1]));
  const uint32x4x2_t c2 = vtrnq_u32(vreinterpretq_u32_u16(b2.val[0]),
                                    vreinterpretq_u32_u16(b3.val[0]));
  const uint32x4x2_t c3 = vtrnq_u32(vreinterpretq_u32_u16(b2.val[1]),
                                    vreinterpretq_u32_u16(b3.val[1]));

  // Swap 64 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70
  // d0.val[1]: 04 14 24 34 44 54 64 74
  // d1.val[0]: 01 11 21 31 41 51 61 71
  // d1.val[1]: 05 15 25 35 45 55 65 75
  // d2.val[0]: 02 12 22 32 42 52 62 72
  // d2.val[1]: 06 16 26 36 46 56 66 76
  // d3.val[0]: 03 13 23 33 43 53 63 73
  // d3.val[1]: 07 17 27 37 47 57 67 77

  const uint16x8x2_t d0 = vpx_vtrnq_u64_to_u16(c0.val[0], c2.val[0]);
  const uint16x8x2_t d1 = vpx_vtrnq_u64_to_u16(c1.val[0], c3.val[0]);
  const uint16x8x2_t d2 = vpx_vtrnq_u64_to_u16(c0.val[1], c2.val[1]);
  const uint16x8x2_t d3 = vpx_vtrnq_u64_to_u16(c1.val[1], c3.val[1]);

  *a0 = d0.val[0];
  *a1 = d1.val[0];
  *a2 = d2.val[0];
  *a3 = d3.val[0];
  *a4 = d0.val[1];
  *a5 = d1.val[1];
  *a6 = d2.val[1];
  *a7 = d3.val[1];
}

static INLINE void transpose_s32_8x8(int32x4x2_t *a0, int32x4x2_t *a1,
                                     int32x4x2_t *a2, int32x4x2_t *a3,
                                     int32x4x2_t *a4, int32x4x2_t *a5,
                                     int32x4x2_t *a6, int32x4x2_t *a7) {
  // Swap 32 bit elements. Goes from:
  // a0: 00 01 02 03 04 05 06 07
  // a1: 10 11 12 13 14 15 16 17
  // a2: 20 21 22 23 24 25 26 27
  // a3: 30 31 32 33 34 35 36 37
  // a4: 40 41 42 43 44 45 46 47
  // a5: 50 51 52 53 54 55 56 57
  // a6: 60 61 62 63 64 65 66 67
  // a7: 70 71 72 73 74 75 76 77
  // to:
  // b0: 00 10 02 12 01 11 03 13
  // b1: 20 30 22 32 21 31 23 33
  // b2: 40 50 42 52 41 51 43 53
  // b3: 60 70 62 72 61 71 63 73
  // b4: 04 14 06 16 05 15 07 17
  // b5: 24 34 26 36 25 35 27 37
  // b6: 44 54 46 56 45 55 47 57
  // b7: 64 74 66 76 65 75 67 77

  const int32x4x2_t b0 = vtrnq_s32(a0->val[0], a1->val[0]);
  const int32x4x2_t b1 = vtrnq_s32(a2->val[0], a3->val[0]);
  const int32x4x2_t b2 = vtrnq_s32(a4->val[0], a5->val[0]);
  const int32x4x2_t b3 = vtrnq_s32(a6->val[0], a7->val[0]);
  const int32x4x2_t b4 = vtrnq_s32(a0->val[1], a1->val[1]);
  const int32x4x2_t b5 = vtrnq_s32(a2->val[1], a3->val[1]);
  const int32x4x2_t b6 = vtrnq_s32(a4->val[1], a5->val[1]);
  const int32x4x2_t b7 = vtrnq_s32(a6->val[1], a7->val[1]);

  // Swap 64 bit elements resulting in:
  // c0: 00 10 20 30 02 12 22 32
  // c1: 01 11 21 31 03 13 23 33
  // c2: 40 50 60 70 42 52 62 72
  // c3: 41 51 61 71 43 53 63 73
  // c4: 04 14 24 34 06 16 26 36
  // c5: 05 15 25 35 07 17 27 37
  // c6: 44 54 64 74 46 56 66 76
  // c7: 45 55 65 75 47 57 67 77
  const int32x4x2_t c0 = vpx_vtrnq_s64_to_s32(b0.val[0], b1.val[0]);
  const int32x4x2_t c1 = vpx_vtrnq_s64_to_s32(b0.val[1], b1.val[1]);
  const int32x4x2_t c2 = vpx_vtrnq_s64_to_s32(b2.val[0], b3.val[0]);
  const int32x4x2_t c3 = vpx_vtrnq_s64_to_s32(b2.val[1], b3.val[1]);
  const int32x4x2_t c4 = vpx_vtrnq_s64_to_s32(b4.val[0], b5.val[0]);
  const int32x4x2_t c5 = vpx_vtrnq_s64_to_s32(b4.val[1], b5.val[1]);
  const int32x4x2_t c6 = vpx_vtrnq_s64_to_s32(b6.val[0], b7.val[0]);
  const int32x4x2_t c7 = vpx_vtrnq_s64_to_s32(b6.val[1], b7.val[1]);

  // Swap 128 bit elements resulting in:
  // a0: 00 10 20 30 40 50 60 70
  // a1: 01 11 21 31 41 51 61 71
  // a2: 02 12 22 32 42 52 62 72
  // a3: 03 13 23 33 43 53 63 73
  // a4: 04 14 24 34 44 54 64 74
  // a5: 05 15 25 35 45 55 65 75
  // a6: 06 16 26 36 46 56 66 76
  // a7: 07 17 27 37 47 57 67 77
  a0->val[0] = c0.val[0];
  a0->val[1] = c2.val[0];
  a1->val[0] = c1.val[0];
  a1->val[1] = c3.val[0];
  a2->val[0] = c0.val[1];
  a2->val[1] = c2.val[1];
  a3->val[0] = c1.val[1];
  a3->val[1] = c3.val[1];
  a4->val[0] = c4.val[0];
  a4->val[1] = c6.val[0];
  a5->val[0] = c5.val[0];
  a5->val[1] = c7.val[0];
  a6->val[0] = c4.val[1];
  a6->val[1] = c6.val[1];
  a7->val[0] = c5.val[1];
  a7->val[1] = c7.val[1];
}

// Helper transpose function for highbd FDCT variants
static INLINE void transpose_s32_8x8_2(int32x4_t *left /*[8]*/,
                                       int32x4_t *right /*[8]*/,
                                       int32x4_t *out_left /*[8]*/,
                                       int32x4_t *out_right /*[8]*/) {
  int32x4x2_t out[8];

  out[0].val[0] = left[0];
  out[0].val[1] = right[0];
  out[1].val[0] = left[1];
  out[1].val[1] = right[1];
  out[2].val[0] = left[2];
  out[2].val[1] = right[2];
  out[3].val[0] = left[3];
  out[3].val[1] = right[3];
  out[4].val[0] = left[4];
  out[4].val[1] = right[4];
  out[5].val[0] = left[5];
  out[5].val[1] = right[5];
  out[6].val[0] = left[6];
  out[6].val[1] = right[6];
  out[7].val[0] = left[7];
  out[7].val[1] = right[7];

  transpose_s32_8x8(&out[0], &out[1], &out[2], &out[3], &out[4], &out[5],
                    &out[6], &out[7]);

  out_left[0] = out[0].val[0];
  out_left[1] = out[1].val[0];
  out_left[2] = out[2].val[0];
  out_left[3] = out[3].val[0];
  out_left[4] = out[4].val[0];
  out_left[5] = out[5].val[0];
  out_left[6] = out[6].val[0];
  out_left[7] = out[7].val[0];
  out_right[0] = out[0].val[1];
  out_right[1] = out[1].val[1];
  out_right[2] = out[2].val[1];
  out_right[3] = out[3].val[1];
  out_right[4] = out[4].val[1];
  out_right[5] = out[5].val[1];
  out_right[6] = out[6].val[1];
  out_right[7] = out[7].val[1];
}

static INLINE void transpose_s32_16x16(int32x4_t *left1, int32x4_t *right1,
                                       int32x4_t *left2, int32x4_t *right2) {
  int32x4_t tl[16], tr[16];

  // transpose the 4 8x8 quadrants separately but first swap quadrants 2 and 3.
  tl[0] = left1[8];
  tl[1] = left1[9];
  tl[2] = left1[10];
  tl[3] = left1[11];
  tl[4] = left1[12];
  tl[5] = left1[13];
  tl[6] = left1[14];
  tl[7] = left1[15];
  tr[0] = right1[8];
  tr[1] = right1[9];
  tr[2] = right1[10];
  tr[3] = right1[11];
  tr[4] = right1[12];
  tr[5] = right1[13];
  tr[6] = right1[14];
  tr[7] = right1[15];

  left1[8] = left2[0];
  left1[9] = left2[1];
  left1[10] = left2[2];
  left1[11] = left2[3];
  left1[12] = left2[4];
  left1[13] = left2[5];
  left1[14] = left2[6];
  left1[15] = left2[7];
  right1[8] = right2[0];
  right1[9] = right2[1];
  right1[10] = right2[2];
  right1[11] = right2[3];
  right1[12] = right2[4];
  right1[13] = right2[5];
  right1[14] = right2[6];
  right1[15] = right2[7];

  left2[0] = tl[0];
  left2[1] = tl[1];
  left2[2] = tl[2];
  left2[3] = tl[3];
  left2[4] = tl[4];
  left2[5] = tl[5];
  left2[6] = tl[6];
  left2[7] = tl[7];
  right2[0] = tr[0];
  right2[1] = tr[1];
  right2[2] = tr[2];
  right2[3] = tr[3];
  right2[4] = tr[4];
  right2[5] = tr[5];
  right2[6] = tr[6];
  right2[7] = tr[7];

  transpose_s32_8x8_2(left1, right1, left1, right1);
  transpose_s32_8x8_2(left2, right2, left2, right2);
  transpose_s32_8x8_2(left1 + 8, right1 + 8, left1 + 8, right1 + 8);
  transpose_s32_8x8_2(left2 + 8, right2 + 8, left2 + 8, right2 + 8);
}

static INLINE void transpose_u8_16x8(
    const uint8x16_t i0, const uint8x16_t i1, const uint8x16_t i2,
    const uint8x16_t i3, const uint8x16_t i4, const uint8x16_t i5,
    const uint8x16_t i6, const uint8x16_t i7, uint8x8_t *o0, uint8x8_t *o1,
    uint8x8_t *o2, uint8x8_t *o3, uint8x8_t *o4, uint8x8_t *o5, uint8x8_t *o6,
    uint8x8_t *o7, uint8x8_t *o8, uint8x8_t *o9, uint8x8_t *o10, uint8x8_t *o11,
    uint8x8_t *o12, uint8x8_t *o13, uint8x8_t *o14, uint8x8_t *o15) {
  // Swap 8 bit elements. Goes from:
  // i0: 00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F
  // i1: 10 11 12 13 14 15 16 17  18 19 1A 1B 1C 1D 1E 1F
  // i2: 20 21 22 23 24 25 26 27  28 29 2A 2B 2C 2D 2E 2F
  // i3: 30 31 32 33 34 35 36 37  38 39 3A 3B 3C 3D 3E 3F
  // i4: 40 41 42 43 44 45 46 47  48 49 4A 4B 4C 4D 4E 4F
  // i5: 50 51 52 53 54 55 56 57  58 59 5A 5B 5C 5D 5E 5F
  // i6: 60 61 62 63 64 65 66 67  68 69 6A 6B 6C 6D 6E 6F
  // i7: 70 71 72 73 74 75 76 77  78 79 7A 7B 7C 7D 7E 7F
  // to:
  // b0.val[0]: 00 10 02 12 04 14 06 16  08 18 0A 1A 0C 1C 0E 1E
  // b0.val[1]: 01 11 03 13 05 15 07 17  09 19 0B 1B 0D 1D 0F 1F
  // b1.val[0]: 20 30 22 32 24 34 26 36  28 38 2A 3A 2C 3C 2E 3E
  // b1.val[1]: 21 31 23 33 25 35 27 37  29 39 2B 3B 2D 3D 2F 3F
  // b2.val[0]: 40 50 42 52 44 54 46 56  48 58 4A 5A 4C 5C 4E 5E
  // b2.val[1]: 41 51 43 53 45 55 47 57  49 59 4B 5B 4D 5D 4F 5F
  // b3.val[0]: 60 70 62 72 64 74 66 76  68 78 6A 7A 6C 7C 6E 7E
  // b3.val[1]: 61 71 63 73 65 75 67 77  69 79 6B 7B 6D 7D 6F 7F
  const uint8x16x2_t b0 = vtrnq_u8(i0, i1);
  const uint8x16x2_t b1 = vtrnq_u8(i2, i3);
  const uint8x16x2_t b2 = vtrnq_u8(i4, i5);
  const uint8x16x2_t b3 = vtrnq_u8(i6, i7);

  // Swap 16 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34  08 18 28 38 0C 1C 2C 3C
  // c0.val[1]: 02 12 22 32 06 16 26 36  0A 1A 2A 3A 0E 1E 2E 3E
  // c1.val[0]: 01 11 21 31 05 15 25 35  09 19 29 39 0D 1D 2D 3D
  // c1.val[1]: 03 13 23 33 07 17 27 37  0B 1B 2B 3B 0F 1F 2F 3F
  // c2.val[0]: 40 50 60 70 44 54 64 74  48 58 68 78 4C 5C 6C 7C
  // c2.val[1]: 42 52 62 72 46 56 66 76  4A 5A 6A 7A 4E 5E 6E 7E
  // c3.val[0]: 41 51 61 71 45 55 65 75  49 59 69 79 4D 5D 6D 7D
  // c3.val[1]: 43 53 63 73 47 57 67 77  4B 5B 6B 7B 4F 5F 6F 7F
  const uint16x8x2_t c0 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[0]),
                                    vreinterpretq_u16_u8(b1.val[0]));
  const uint16x8x2_t c1 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[1]),
                                    vreinterpretq_u16_u8(b1.val[1]));
  const uint16x8x2_t c2 = vtrnq_u16(vreinterpretq_u16_u8(b2.val[0]),
                                    vreinterpretq_u16_u8(b3.val[0]));
  const uint16x8x2_t c3 = vtrnq_u16(vreinterpretq_u16_u8(b2.val[1]),
                                    vreinterpretq_u16_u8(b3.val[1]));

  // Swap 32 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70  08 18 28 38 48 58 68 78
  // d0.val[1]: 04 14 24 34 44 54 64 74  0C 1C 2C 3C 4C 5C 6C 7C
  // d1.val[0]: 02 12 22 32 42 52 62 72  0A 1A 2A 3A 4A 5A 6A 7A
  // d1.val[1]: 06 16 26 36 46 56 66 76  0E 1E 2E 3E 4E 5E 6E 7E
  // d2.val[0]: 01 11 21 31 41 51 61 71  09 19 29 39 49 59 69 79
  // d2.val[1]: 05 15 25 35 45 55 65 75  0D 1D 2D 3D 4D 5D 6D 7D
  // d3.val[0]: 03 13 23 33 43 53 63 73  0B 1B 2B 3B 4B 5B 6B 7B
  // d3.val[1]: 07 17 27 37 47 57 67 77  0F 1F 2F 3F 4F 5F 6F 7F
  const uint32x4x2_t d0 = vtrnq_u32(vreinterpretq_u32_u16(c0.val[0]),
                                    vreinterpretq_u32_u16(c2.val[0]));
  const uint32x4x2_t d1 = vtrnq_u32(vreinterpretq_u32_u16(c0.val[1]),
                                    vreinterpretq_u32_u16(c2.val[1]));
  const uint32x4x2_t d2 = vtrnq_u32(vreinterpretq_u32_u16(c1.val[0]),
                                    vreinterpretq_u32_u16(c3.val[0]));
  const uint32x4x2_t d3 = vtrnq_u32(vreinterpretq_u32_u16(c1.val[1]),
                                    vreinterpretq_u32_u16(c3.val[1]));

  // Output:
  // o0 : 00 10 20 30 40 50 60 70
  // o1 : 01 11 21 31 41 51 61 71
  // o2 : 02 12 22 32 42 52 62 72
  // o3 : 03 13 23 33 43 53 63 73
  // o4 : 04 14 24 34 44 54 64 74
  // o5 : 05 15 25 35 45 55 65 75
  // o6 : 06 16 26 36 46 56 66 76
  // o7 : 07 17 27 37 47 57 67 77
  // o8 : 08 18 28 38 48 58 68 78
  // o9 : 09 19 29 39 49 59 69 79
  // o10: 0A 1A 2A 3A 4A 5A 6A 7A
  // o11: 0B 1B 2B 3B 4B 5B 6B 7B
  // o12: 0C 1C 2C 3C 4C 5C 6C 7C
  // o13: 0D 1D 2D 3D 4D 5D 6D 7D
  // o14: 0E 1E 2E 3E 4E 5E 6E 7E
  // o15: 0F 1F 2F 3F 4F 5F 6F 7F
  *o0 = vget_low_u8(vreinterpretq_u8_u32(d0.val[0]));
  *o1 = vget_low_u8(vreinterpretq_u8_u32(d2.val[0]));
  *o2 = vget_low_u8(vreinterpretq_u8_u32(d1.val[0]));
  *o3 = vget_low_u8(vreinterpretq_u8_u32(d3.val[0]));
  *o4 = vget_low_u8(vreinterpretq_u8_u32(d0.val[1]));
  *o5 = vget_low_u8(vreinterpretq_u8_u32(d2.val[1]));
  *o6 = vget_low_u8(vreinterpretq_u8_u32(d1.val[1]));
  *o7 = vget_low_u8(vreinterpretq_u8_u32(d3.val[1]));
  *o8 = vget_high_u8(vreinterpretq_u8_u32(d0.val[0]));
  *o9 = vget_high_u8(vreinterpretq_u8_u32(d2.val[0]));
  *o10 = vget_high_u8(vreinterpretq_u8_u32(d1.val[0]));
  *o11 = vget_high_u8(vreinterpretq_u8_u32(d3.val[0]));
  *o12 = vget_high_u8(vreinterpretq_u8_u32(d0.val[1]));
  *o13 = vget_high_u8(vreinterpretq_u8_u32(d2.val[1]));
  *o14 = vget_high_u8(vreinterpretq_u8_u32(d1.val[1]));
  *o15 = vget_high_u8(vreinterpretq_u8_u32(d3.val[1]));
}

static INLINE void transpose_u8_8x16(
    const uint8x8_t i0, const uint8x8_t i1, const uint8x8_t i2,
    const uint8x8_t i3, const uint8x8_t i4, const uint8x8_t i5,
    const uint8x8_t i6, const uint8x8_t i7, const uint8x8_t i8,
    const uint8x8_t i9, const uint8x8_t i10, const uint8x8_t i11,
    const uint8x8_t i12, const uint8x8_t i13, const uint8x8_t i14,
    const uint8x8_t i15, uint8x16_t *o0, uint8x16_t *o1, uint8x16_t *o2,
    uint8x16_t *o3, uint8x16_t *o4, uint8x16_t *o5, uint8x16_t *o6,
    uint8x16_t *o7) {
  // Combine 8 bit elements. Goes from:
  // i0 : 00 01 02 03 04 05 06 07
  // i1 : 10 11 12 13 14 15 16 17
  // i2 : 20 21 22 23 24 25 26 27
  // i3 : 30 31 32 33 34 35 36 37
  // i4 : 40 41 42 43 44 45 46 47
  // i5 : 50 51 52 53 54 55 56 57
  // i6 : 60 61 62 63 64 65 66 67
  // i7 : 70 71 72 73 74 75 76 77
  // i8 : 80 81 82 83 84 85 86 87
  // i9 : 90 91 92 93 94 95 96 97
  // i10: A0 A1 A2 A3 A4 A5 A6 A7
  // i11: B0 B1 B2 B3 B4 B5 B6 B7
  // i12: C0 C1 C2 C3 C4 C5 C6 C7
  // i13: D0 D1 D2 D3 D4 D5 D6 D7
  // i14: E0 E1 E2 E3 E4 E5 E6 E7
  // i15: F0 F1 F2 F3 F4 F5 F6 F7
  // to:
  // a0: 00 01 02 03 04 05 06 07  80 81 82 83 84 85 86 87
  // a1: 10 11 12 13 14 15 16 17  90 91 92 93 94 95 96 97
  // a2: 20 21 22 23 24 25 26 27  A0 A1 A2 A3 A4 A5 A6 A7
  // a3: 30 31 32 33 34 35 36 37  B0 B1 B2 B3 B4 B5 B6 B7
  // a4: 40 41 42 43 44 45 46 47  C0 C1 C2 C3 C4 C5 C6 C7
  // a5: 50 51 52 53 54 55 56 57  D0 D1 D2 D3 D4 D5 D6 D7
  // a6: 60 61 62 63 64 65 66 67  E0 E1 E2 E3 E4 E5 E6 E7
  // a7: 70 71 72 73 74 75 76 77  F0 F1 F2 F3 F4 F5 F6 F7
  const uint8x16_t a0 = vcombine_u8(i0, i8);
  const uint8x16_t a1 = vcombine_u8(i1, i9);
  const uint8x16_t a2 = vcombine_u8(i2, i10);
  const uint8x16_t a3 = vcombine_u8(i3, i11);
  const uint8x16_t a4 = vcombine_u8(i4, i12);
  const uint8x16_t a5 = vcombine_u8(i5, i13);
  const uint8x16_t a6 = vcombine_u8(i6, i14);
  const uint8x16_t a7 = vcombine_u8(i7, i15);

  // Swap 8 bit elements resulting in:
  // b0.val[0]: 00 10 02 12 04 14 06 16  80 90 82 92 84 94 86 96
  // b0.val[1]: 01 11 03 13 05 15 07 17  81 91 83 93 85 95 87 97
  // b1.val[0]: 20 30 22 32 24 34 26 36  A0 B0 A2 B2 A4 B4 A6 B6
  // b1.val[1]: 21 31 23 33 25 35 27 37  A1 B1 A3 B3 A5 B5 A7 B7
  // b2.val[0]: 40 50 42 52 44 54 46 56  C0 D0 C2 D2 C4 D4 C6 D6
  // b2.val[1]: 41 51 43 53 45 55 47 57  C1 D1 C3 D3 C5 D5 C7 D7
  // b3.val[0]: 60 70 62 72 64 74 66 76  E0 F0 E2 F2 E4 F4 E6 F6
  // b3.val[1]: 61 71 63 73 65 75 67 77  E1 F1 E3 F3 E5 F5 E7 F7
  const uint8x16x2_t b0 = vtrnq_u8(a0, a1);
  const uint8x16x2_t b1 = vtrnq_u8(a2, a3);
  const uint8x16x2_t b2 = vtrnq_u8(a4, a5);
  const uint8x16x2_t b3 = vtrnq_u8(a6, a7);

  // Swap 16 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34  80 90 A0 B0 84 94 A4 B4
  // c0.val[1]: 02 12 22 32 06 16 26 36  82 92 A2 B2 86 96 A6 B6
  // c1.val[0]: 01 11 21 31 05 15 25 35  81 91 A1 B1 85 95 A5 B5
  // c1.val[1]: 03 13 23 33 07 17 27 37  83 93 A3 B3 87 97 A7 B7
  // c2.val[0]: 40 50 60 70 44 54 64 74  C0 D0 E0 F0 C4 D4 E4 F4
  // c2.val[1]: 42 52 62 72 46 56 66 76  C2 D2 E2 F2 C6 D6 E6 F6
  // c3.val[0]: 41 51 61 71 45 55 65 75  C1 D1 E1 F1 C5 D5 E5 F5
  // c3.val[1]: 43 53 63 73 47 57 67 77  C3 D3 E3 F3 C7 D7 E7 F7
  const uint16x8x2_t c0 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[0]),
                                    vreinterpretq_u16_u8(b1.val[0]));
  const uint16x8x2_t c1 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[1]),
                                    vreinterpretq_u16_u8(b1.val[1]));
  const uint16x8x2_t c2 = vtrnq_u16(vreinterpretq_u16_u8(b2.val[0]),
                                    vreinterpretq_u16_u8(b3.val[0]));
  const uint16x8x2_t c3 = vtrnq_u16(vreinterpretq_u16_u8(b2.val[1]),
                                    vreinterpretq_u16_u8(b3.val[1]));

  // Swap 32 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70  80 90 A0 B0 C0 D0 E0 F0
  // d0.val[1]: 04 14 24 34 44 54 64 74  84 94 A4 B4 C4 D4 E4 F4
  // d1.val[0]: 02 12 22 32 42 52 62 72  82 92 A2 B2 C2 D2 E2 F2
  // d1.val[1]: 06 16 26 36 46 56 66 76  86 96 A6 B6 C6 D6 E6 F6
  // d2.val[0]: 01 11 21 31 41 51 61 71  81 91 A1 B1 C1 D1 E1 F1
  // d2.val[1]: 05 15 25 35 45 55 65 75  85 95 A5 B5 C5 D5 E5 F5
  // d3.val[0]: 03 13 23 33 43 53 63 73  83 93 A3 B3 C3 D3 E3 F3
  // d3.val[1]: 07 17 27 37 47 57 67 77  87 97 A7 B7 C7 D7 E7 F7
  const uint32x4x2_t d0 = vtrnq_u32(vreinterpretq_u32_u16(c0.val[0]),
                                    vreinterpretq_u32_u16(c2.val[0]));
  const uint32x4x2_t d1 = vtrnq_u32(vreinterpretq_u32_u16(c0.val[1]),
                                    vreinterpretq_u32_u16(c2.val[1]));
  const uint32x4x2_t d2 = vtrnq_u32(vreinterpretq_u32_u16(c1.val[0]),
                                    vreinterpretq_u32_u16(c3.val[0]));
  const uint32x4x2_t d3 = vtrnq_u32(vreinterpretq_u32_u16(c1.val[1]),
                                    vreinterpretq_u32_u16(c3.val[1]));

  // Output:
  // o0: 00 10 20 30 40 50 60 70  80 90 A0 B0 C0 D0 E0 F0
  // o1: 01 11 21 31 41 51 61 71  81 91 A1 B1 C1 D1 E1 F1
  // o2: 02 12 22 32 42 52 62 72  82 92 A2 B2 C2 D2 E2 F2
  // o3: 03 13 23 33 43 53 63 73  83 93 A3 B3 C3 D3 E3 F3
  // o4: 04 14 24 34 44 54 64 74  84 94 A4 B4 C4 D4 E4 F4
  // o5: 05 15 25 35 45 55 65 75  85 95 A5 B5 C5 D5 E5 F5
  // o6: 06 16 26 36 46 56 66 76  86 96 A6 B6 C6 D6 E6 F6
  // o7: 07 17 27 37 47 57 67 77  87 97 A7 B7 C7 D7 E7 F7
  *o0 = vreinterpretq_u8_u32(d0.val[0]);
  *o1 = vreinterpretq_u8_u32(d2.val[0]);
  *o2 = vreinterpretq_u8_u32(d1.val[0]);
  *o3 = vreinterpretq_u8_u32(d3.val[0]);
  *o4 = vreinterpretq_u8_u32(d0.val[1]);
  *o5 = vreinterpretq_u8_u32(d2.val[1]);
  *o6 = vreinterpretq_u8_u32(d1.val[1]);
  *o7 = vreinterpretq_u8_u32(d3.val[1]);
}

static INLINE void transpose_u8_16x16(
    const uint8x16_t i0, const uint8x16_t i1, const uint8x16_t i2,
    const uint8x16_t i3, const uint8x16_t i4, const uint8x16_t i5,
    const uint8x16_t i6, const uint8x16_t i7, const uint8x16_t i8,
    const uint8x16_t i9, const uint8x16_t i10, const uint8x16_t i11,
    const uint8x16_t i12, const uint8x16_t i13, const uint8x16_t i14,
    const uint8x16_t i15, uint8x16_t *o0, uint8x16_t *o1, uint8x16_t *o2,
    uint8x16_t *o3, uint8x16_t *o4, uint8x16_t *o5, uint8x16_t *o6,
    uint8x16_t *o7, uint8x16_t *o8, uint8x16_t *o9, uint8x16_t *o10,
    uint8x16_t *o11, uint8x16_t *o12, uint8x16_t *o13, uint8x16_t *o14,
    uint8x16_t *o15) {
  // Swap 8 bit elements. Goes from:
  // i0:  00 01 02 03 04 05 06 07  08 09 0A 0B 0C 0D 0E 0F
  // i1:  10 11 12 13 14 15 16 17  18 19 1A 1B 1C 1D 1E 1F
  // i2:  20 21 22 23 24 25 26 27  28 29 2A 2B 2C 2D 2E 2F
  // i3:  30 31 32 33 34 35 36 37  38 39 3A 3B 3C 3D 3E 3F
  // i4:  40 41 42 43 44 45 46 47  48 49 4A 4B 4C 4D 4E 4F
  // i5:  50 51 52 53 54 55 56 57  58 59 5A 5B 5C 5D 5E 5F
  // i6:  60 61 62 63 64 65 66 67  68 69 6A 6B 6C 6D 6E 6F
  // i7:  70 71 72 73 74 75 76 77  78 79 7A 7B 7C 7D 7E 7F
  // i8:  80 81 82 83 84 85 86 87  88 89 8A 8B 8C 8D 8E 8F
  // i9:  90 91 92 93 94 95 96 97  98 99 9A 9B 9C 9D 9E 9F
  // i10: A0 A1 A2 A3 A4 A5 A6 A7  A8 A9 AA AB AC AD AE AF
  // i11: B0 B1 B2 B3 B4 B5 B6 B7  B8 B9 BA BB BC BD BE BF
  // i12: C0 C1 C2 C3 C4 C5 C6 C7  C8 C9 CA CB CC CD CE CF
  // i13: D0 D1 D2 D3 D4 D5 D6 D7  D8 D9 DA DB DC DD DE DF
  // i14: E0 E1 E2 E3 E4 E5 E6 E7  E8 E9 EA EB EC ED EE EF
  // i15: F0 F1 F2 F3 F4 F5 F6 F7  F8 F9 FA FB FC FD FE FF
  // to:
  // b0.val[0]: 00 10 02 12 04 14 06 16  08 18 0A 1A 0C 1C 0E 1E
  // b0.val[1]: 01 11 03 13 05 15 07 17  09 19 0B 1B 0D 1D 0F 1F
  // b1.val[0]: 20 30 22 32 24 34 26 36  28 38 2A 3A 2C 3C 2E 3E
  // b1.val[1]: 21 31 23 33 25 35 27 37  29 39 2B 3B 2D 3D 2F 3F
  // b2.val[0]: 40 50 42 52 44 54 46 56  48 58 4A 5A 4C 5C 4E 5E
  // b2.val[1]: 41 51 43 53 45 55 47 57  49 59 4B 5B 4D 5D 4F 5F
  // b3.val[0]: 60 70 62 72 64 74 66 76  68 78 6A 7A 6C 7C 6E 7E
  // b3.val[1]: 61 71 63 73 65 75 67 77  69 79 6B 7B 6D 7D 6F 7F
  // b4.val[0]: 80 90 82 92 84 94 86 96  88 98 8A 9A 8C 9C 8E 9E
  // b4.val[1]: 81 91 83 93 85 95 87 97  89 99 8B 9B 8D 9D 8F 9F
  // b5.val[0]: A0 B0 A2 B2 A4 B4 A6 B6  A8 B8 AA BA AC BC AE BE
  // b5.val[1]: A1 B1 A3 B3 A5 B5 A7 B7  A9 B9 AB BB AD BD AF BF
  // b6.val[0]: C0 D0 C2 D2 C4 D4 C6 D6  C8 D8 CA DA CC DC CE DE
  // b6.val[1]: C1 D1 C3 D3 C5 D5 C7 D7  C9 D9 CB DB CD DD CF DF
  // b7.val[0]: E0 F0 E2 F2 E4 F4 E6 F6  E8 F8 EA FA EC FC EE FE
  // b7.val[1]: E1 F1 E3 F3 E5 F5 E7 F7  E9 F9 EB FB ED FD EF FF
  const uint8x16x2_t b0 = vtrnq_u8(i0, i1);
  const uint8x16x2_t b1 = vtrnq_u8(i2, i3);
  const uint8x16x2_t b2 = vtrnq_u8(i4, i5);
  const uint8x16x2_t b3 = vtrnq_u8(i6, i7);
  const uint8x16x2_t b4 = vtrnq_u8(i8, i9);
  const uint8x16x2_t b5 = vtrnq_u8(i10, i11);
  const uint8x16x2_t b6 = vtrnq_u8(i12, i13);
  const uint8x16x2_t b7 = vtrnq_u8(i14, i15);

  // Swap 16 bit elements resulting in:
  // c0.val[0]: 00 10 20 30 04 14 24 34  08 18 28 38 0C 1C 2C 3C
  // c0.val[1]: 02 12 22 32 06 16 26 36  0A 1A 2A 3A 0E 1E 2E 3E
  // c1.val[0]: 01 11 21 31 05 15 25 35  09 19 29 39 0D 1D 2D 3D
  // c1.val[1]: 03 13 23 33 07 17 27 37  0B 1B 2B 3B 0F 1F 2F 3F
  // c2.val[0]: 40 50 60 70 44 54 64 74  48 58 68 78 4C 5C 6C 7C
  // c2.val[1]: 42 52 62 72 46 56 66 76  4A 5A 6A 7A 4E 5E 6E 7E
  // c3.val[0]: 41 51 61 71 45 55 65 75  49 59 69 79 4D 5D 6D 7D
  // c3.val[1]: 43 53 63 73 47 57 67 77  4B 5B 6B 7B 4F 5F 6F 7F
  // c4.val[0]: 80 90 A0 B0 84 94 A4 B4  88 98 A8 B8 8C 9C AC BC
  // c4.val[1]: 82 92 A2 B2 86 96 A6 B6  8A 9A AA BA 8E 9E AE BE
  // c5.val[0]: 81 91 A1 B1 85 95 A5 B5  89 99 A9 B9 8D 9D AD BD
  // c5.val[1]: 83 93 A3 B3 87 97 A7 B7  8B 9B AB BB 8F 9F AF BF
  // c6.val[0]: C0 D0 E0 F0 C4 D4 E4 F4  C8 D8 E8 F8 CC DC EC FC
  // c6.val[1]: C2 D2 E2 F2 C6 D6 E6 F6  CA DA EA FA CE DE EE FE
  // c7.val[0]: C1 D1 E1 F1 C5 D5 E5 F5  C9 D9 E9 F9 CD DD ED FD
  // c7.val[1]: C3 D3 E3 F3 C7 D7 E7 F7  CB DB EB FB CF DF EF FF
  const uint16x8x2_t c0 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[0]),
                                    vreinterpretq_u16_u8(b1.val[0]));
  const uint16x8x2_t c1 = vtrnq_u16(vreinterpretq_u16_u8(b0.val[1]),
                                    vreinterpretq_u16_u8(b1.val[1]));
  const uint16x8x2_t c2 = vtrnq_u16(vreinterpretq_u16_u8(b2.val[0]),
                                    vreinterpretq_u16_u8(b3.val[0]));
  const uint16x8x2_t c3 = vtrnq_u16(vreinterpretq_u16_u8(b2.val[1]),
                                    vreinterpretq_u16_u8(b3.val[1]));
  const uint16x8x2_t c4 = vtrnq_u16(vreinterpretq_u16_u8(b4.val[0]),
                                    vreinterpretq_u16_u8(b5.val[0]));
  const uint16x8x2_t c5 = vtrnq_u16(vreinterpretq_u16_u8(b4.val[1]),
                                    vreinterpretq_u16_u8(b5.val[1]));
  const uint16x8x2_t c6 = vtrnq_u16(vreinterpretq_u16_u8(b6.val[0]),
                                    vreinterpretq_u16_u8(b7.val[0]));
  const uint16x8x2_t c7 = vtrnq_u16(vreinterpretq_u16_u8(b6.val[1]),
                                    vreinterpretq_u16_u8(b7.val[1]));

  // Swap 32 bit elements resulting in:
  // d0.val[0]: 00 10 20 30 40 50 60 70  08 18 28 38 48 58 68 78
  // d0.val[1]: 04 14 24 34 44 54 64 74  0C 1C 2C 3C 4C 5C 6C 7C
  // d1.val[0]: 02 12 22 32 42 52 62 72  0A 1A 2A 3A 4A 5A 6A 7A
  // d1.val[1]: 06 16 26 36 46 56 66 76  0E 1E 2E 3E 4E 5E 6E 7E
  // d2.val[0]: 01 11 21 31 41 51 61 71  09 19 29 39 49 59 69 79
  // d2.val[1]: 05 15 25 35 45 55 65 75  0D 1D 2D 3D 4D 5D 6D 7D
  // d3.val[0]: 03 13 23 33 43 53 63 73  0B 1B 2B 3B 4B 5B 6B 7B
  // d3.val[1]: 07 17 27 37 47 57 67 77  0F 1F 2F 3F 4F 5F 6F 7F
  // d4.val[0]: 80 90 A0 B0 C0 D0 E0 F0  88 98 A8 B8 C8 D8 E8 F8
  // d4.val[1]: 84 94 A4 B4 C4 D4 E4 F4  8C 9C AC BC CC DC EC FC
  // d5.val[0]: 82 92 A2 B2 C2 D2 E2 F2  8A 9A AA BA CA DA EA FA
  // d5.val[1]: 86 96 A6 B6 C6 D6 E6 F6  8E 9E AE BE CE DE EE FE
  // d6.val[0]: 81 91 A1 B1 C1 D1 E1 F1  89 99 A9 B9 C9 D9 E9 F9
  // d6.val[1]: 85 95 A5 B5 C5 D5 E5 F5  8D 9D AD BD CD DD ED FD
  // d7.val[0]: 83 93 A3 B3 C3 D3 E3 F3  8B 9B AB BB CB DB EB FB
  // d7.val[1]: 87 97 A7 B7 C7 D7 E7 F7  8F 9F AF BF CF DF EF FF
  const uint32x4x2_t d0 = vtrnq_u32(vreinterpretq_u32_u16(c0.val[0]),
                                    vreinterpretq_u32_u16(c2.val[0]));
  const uint32x4x2_t d1 = vtrnq_u32(vreinterpretq_u32_u16(c0.val[1]),
                                    vreinterpretq_u32_u16(c2.val[1]));
  const uint32x4x2_t d2 = vtrnq_u32(vreinterpretq_u32_u16(c1.val[0]),
                                    vreinterpretq_u32_u16(c3.val[0]));
  const uint32x4x2_t d3 = vtrnq_u32(vreinterpretq_u32_u16(c1.val[1]),
                                    vreinterpretq_u32_u16(c3.val[1]));
  const uint32x4x2_t d4 = vtrnq_u32(vreinterpretq_u32_u16(c4.val[0]),
                                    vreinterpretq_u32_u16(c6.val[0]));
  const uint32x4x2_t d5 = vtrnq_u32(vreinterpretq_u32_u16(c4.val[1]),
                                    vreinterpretq_u32_u16(c6.val[1]));
  const uint32x4x2_t d6 = vtrnq_u32(vreinterpretq_u32_u16(c5.val[0]),
                                    vreinterpretq_u32_u16(c7.val[0]));
  const uint32x4x2_t d7 = vtrnq_u32(vreinterpretq_u32_u16(c5.val[1]),
                                    vreinterpretq_u32_u16(c7.val[1]));

  // Swap 64 bit elements resulting in:
  // e0.val[0]: 00 10 20 30 40 50 60 70  80 90 A0 B0 C0 D0 E0 F0
  // e0.val[1]: 08 18 28 38 48 58 68 78  88 98 A8 B8 C8 D8 E8 F8
  // e1.val[0]: 01 11 21 31 41 51 61 71  84 94 A4 B4 C4 D4 E4 F4
  // e1.val[1]: 09 19 29 39 49 59 69 79  89 99 A9 B9 C9 D9 E9 F9
  // e2.val[0]: 02 12 22 32 42 52 62 72  82 92 A2 B2 C2 D2 E2 F2
  // e2.val[1]: 0A 1A 2A 3A 4A 5A 6A 7A  8A 9A AA BA CA DA EA FA
  // e3.val[0]: 03 13 23 33 43 53 63 73  86 96 A6 B6 C6 D6 E6 F6
  // e3.val[1]: 0B 1B 2B 3B 4B 5B 6B 7B  8B 9B AB BB CB DB EB FB
  // e4.val[0]: 04 14 24 34 44 54 64 74  81 91 A1 B1 C1 D1 E1 F1
  // e4.val[1]: 0C 1C 2C 3C 4C 5C 6C 7C  8C 9C AC BC CC DC EC FC
  // e5.val[0]: 05 15 25 35 45 55 65 75  85 95 A5 B5 C5 D5 E5 F5
  // e5.val[1]: 0D 1D 2D 3D 4D 5D 6D 7D  8D 9D AD BD CD DD ED FD
  // e6.val[0]: 06 16 26 36 46 56 66 76  83 93 A3 B3 C3 D3 E3 F3
  // e6.val[1]: 0E 1E 2E 3E 4E 5E 6E 7E  8E 9E AE BE CE DE EE FE
  // e7.val[0]: 07 17 27 37 47 57 67 77  87 97 A7 B7 C7 D7 E7 F7
  // e7.val[1]: 0F 1F 2F 3F 4F 5F 6F 7F  8F 9F AF BF CF DF EF FF
  const uint8x16x2_t e0 = vpx_vtrnq_u64_to_u8(d0.val[0], d4.val[0]);
  const uint8x16x2_t e1 = vpx_vtrnq_u64_to_u8(d2.val[0], d6.val[0]);
  const uint8x16x2_t e2 = vpx_vtrnq_u64_to_u8(d1.val[0], d5.val[0]);
  const uint8x16x2_t e3 = vpx_vtrnq_u64_to_u8(d3.val[0], d7.val[0]);
  const uint8x16x2_t e4 = vpx_vtrnq_u64_to_u8(d0.val[1], d4.val[1]);
  const uint8x16x2_t e5 = vpx_vtrnq_u64_to_u8(d2.val[1], d6.val[1]);
  const uint8x16x2_t e6 = vpx_vtrnq_u64_to_u8(d1.val[1], d5.val[1]);
  const uint8x16x2_t e7 = vpx_vtrnq_u64_to_u8(d3.val[1], d7.val[1]);

  // Output:
  // o0 : 00 10 20 30 40 50 60 70  80 90 A0 B0 C0 D0 E0 F0
  // o1 : 01 11 21 31 41 51 61 71  84 94 A4 B4 C4 D4 E4 F4
  // o2 : 02 12 22 32 42 52 62 72  82 92 A2 B2 C2 D2 E2 F2
  // o3 : 03 13 23 33 43 53 63 73  86 96 A6 B6 C6 D6 E6 F6
  // o4 : 04 14 24 34 44 54 64 74  81 91 A1 B1 C1 D1 E1 F1
  // o5 : 05 15 25 35 45 55 65 75  85 95 A5 B5 C5 D5 E5 F5
  // o6 : 06 16 26 36 46 56 66 76  83 93 A3 B3 C3 D3 E3 F3
  // o7 : 07 17 27 37 47 57 67 77  87 97 A7 B7 C7 D7 E7 F7
  // o8 : 08 18 28 38 48 58 68 78  88 98 A8 B8 C8 D8 E8 F8
  // o9 : 09 19 29 39 49 59 69 79  89 99 A9 B9 C9 D9 E9 F9
  // o10: 0A 1A 2A 3A 4A 5A 6A 7A  8A 9A AA BA CA DA EA FA
  // o11: 0B 1B 2B 3B 4B 5B 6B 7B  8B 9B AB BB CB DB EB FB
  // o12: 0C 1C 2C 3C 4C 5C 6C 7C  8C 9C AC BC CC DC EC FC
  // o13: 0D 1D 2D 3D 4D 5D 6D 7D  8D 9D AD BD CD DD ED FD
  // o14: 0E 1E 2E 3E 4E 5E 6E 7E  8E 9E AE BE CE DE EE FE
  // o15: 0F 1F 2F 3F 4F 5F 6F 7F  8F 9F AF BF CF DF EF FF
  *o0 = e0.val[0];
  *o1 = e1.val[0];
  *o2 = e2.val[0];
  *o3 = e3.val[0];
  *o4 = e4.val[0];
  *o5 = e5.val[0];
  *o6 = e6.val[0];
  *o7 = e7.val[0];
  *o8 = e0.val[1];
  *o9 = e1.val[1];
  *o10 = e2.val[1];
  *o11 = e3.val[1];
  *o12 = e4.val[1];
  *o13 = e5.val[1];
  *o14 = e6.val[1];
  *o15 = e7.val[1];
}

static INLINE void transpose_s16_16x16(int16x8_t *in0, int16x8_t *in1) {
  int16x8_t t[8];

  // transpose the 4 8x8 quadrants separately but first swap quadrants 2 and 3.
  t[0] = in0[8];
  t[1] = in0[9];
  t[2] = in0[10];
  t[3] = in0[11];
  t[4] = in0[12];
  t[5] = in0[13];
  t[6] = in0[14];
  t[7] = in0[15];
  in0[8] = in1[0];
  in0[9] = in1[1];
  in0[10] = in1[2];
  in0[11] = in1[3];
  in0[12] = in1[4];
  in0[13] = in1[5];
  in0[14] = in1[6];
  in0[15] = in1[7];
  in1[0] = t[0];
  in1[1] = t[1];
  in1[2] = t[2];
  in1[3] = t[3];
  in1[4] = t[4];
  in1[5] = t[5];
  in1[6] = t[6];
  in1[7] = t[7];

  transpose_s16_8x8(&in0[0], &in0[1], &in0[2], &in0[3], &in0[4], &in0[5],
                    &in0[6], &in0[7]);
  transpose_s16_8x8(&in0[8], &in0[9], &in0[10], &in0[11], &in0[12], &in0[13],
                    &in0[14], &in0[15]);
  transpose_s16_8x8(&in1[0], &in1[1], &in1[2], &in1[3], &in1[4], &in1[5],
                    &in1[6], &in1[7]);
  transpose_s16_8x8(&in1[8], &in1[9], &in1[10], &in1[11], &in1[12], &in1[13],
                    &in1[14], &in1[15]);
}

static INLINE void load_and_transpose_u8_4x8(const uint8_t *a,
                                             const int a_stride, uint8x8_t *a0,
                                             uint8x8_t *a1, uint8x8_t *a2,
                                             uint8x8_t *a3) {
  uint8x8_t a4, a5, a6, a7;
  *a0 = vld1_u8(a);
  a += a_stride;
  *a1 = vld1_u8(a);
  a += a_stride;
  *a2 = vld1_u8(a);
  a += a_stride;
  *a3 = vld1_u8(a);
  a += a_stride;
  a4 = vld1_u8(a);
  a += a_stride;
  a5 = vld1_u8(a);
  a += a_stride;
  a6 = vld1_u8(a);
  a += a_stride;
  a7 = vld1_u8(a);

  transpose_u8_4x8(a0, a1, a2, a3, a4, a5, a6, a7);
}

static INLINE void load_and_transpose_u8_8x8(const uint8_t *a,
                                             const int a_stride, uint8x8_t *a0,
                                             uint8x8_t *a1, uint8x8_t *a2,
                                             uint8x8_t *a3, uint8x8_t *a4,
                                             uint8x8_t *a5, uint8x8_t *a6,
                                             uint8x8_t *a7) {
  *a0 = vld1_u8(a);
  a += a_stride;
  *a1 = vld1_u8(a);
  a += a_stride;
  *a2 = vld1_u8(a);
  a += a_stride;
  *a3 = vld1_u8(a);
  a += a_stride;
  *a4 = vld1_u8(a);
  a += a_stride;
  *a5 = vld1_u8(a);
  a += a_stride;
  *a6 = vld1_u8(a);
  a += a_stride;
  *a7 = vld1_u8(a);

  transpose_u8_8x8(a0, a1, a2, a3, a4, a5, a6, a7);
}

static INLINE void transpose_and_store_u8_8x8(uint8_t *a, const int a_stride,
                                              uint8x8_t a0, uint8x8_t a1,
                                              uint8x8_t a2, uint8x8_t a3,
                                              uint8x8_t a4, uint8x8_t a5,
                                              uint8x8_t a6, uint8x8_t a7) {
  transpose_u8_8x8(&a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7);

  vst1_u8(a, a0);
  a += a_stride;
  vst1_u8(a, a1);
  a += a_stride;
  vst1_u8(a, a2);
  a += a_stride;
  vst1_u8(a, a3);
  a += a_stride;
  vst1_u8(a, a4);
  a += a_stride;
  vst1_u8(a, a5);
  a += a_stride;
  vst1_u8(a, a6);
  a += a_stride;
  vst1_u8(a, a7);
}

static INLINE void load_and_transpose_s16_8x8(const int16_t *a,
                                              const int a_stride, int16x8_t *a0,
                                              int16x8_t *a1, int16x8_t *a2,
                                              int16x8_t *a3, int16x8_t *a4,
                                              int16x8_t *a5, int16x8_t *a6,
                                              int16x8_t *a7) {
  *a0 = vld1q_s16(a);
  a += a_stride;
  *a1 = vld1q_s16(a);
  a += a_stride;
  *a2 = vld1q_s16(a);
  a += a_stride;
  *a3 = vld1q_s16(a);
  a += a_stride;
  *a4 = vld1q_s16(a);
  a += a_stride;
  *a5 = vld1q_s16(a);
  a += a_stride;
  *a6 = vld1q_s16(a);
  a += a_stride;
  *a7 = vld1q_s16(a);

  transpose_s16_8x8(a0, a1, a2, a3, a4, a5, a6, a7);
}

static INLINE void load_and_transpose_s32_8x8(
    const int32_t *a, const int a_stride, int32x4x2_t *const a0,
    int32x4x2_t *const a1, int32x4x2_t *const a2, int32x4x2_t *const a3,
    int32x4x2_t *const a4, int32x4x2_t *const a5, int32x4x2_t *const a6,
    int32x4x2_t *const a7) {
  a0->val[0] = vld1q_s32(a);
  a0->val[1] = vld1q_s32(a + 4);
  a += a_stride;
  a1->val[0] = vld1q_s32(a);
  a1->val[1] = vld1q_s32(a + 4);
  a += a_stride;
  a2->val[0] = vld1q_s32(a);
  a2->val[1] = vld1q_s32(a + 4);
  a += a_stride;
  a3->val[0] = vld1q_s32(a);
  a3->val[1] = vld1q_s32(a + 4);
  a += a_stride;
  a4->val[0] = vld1q_s32(a);
  a4->val[1] = vld1q_s32(a + 4);
  a += a_stride;
  a5->val[0] = vld1q_s32(a);
  a5->val[1] = vld1q_s32(a + 4);
  a += a_stride;
  a6->val[0] = vld1q_s32(a);
  a6->val[1] = vld1q_s32(a + 4);
  a += a_stride;
  a7->val[0] = vld1q_s32(a);
  a7->val[1] = vld1q_s32(a + 4);

  transpose_s32_8x8(a0, a1, a2, a3, a4, a5, a6, a7);
}

static INLINE void transpose_concat_s16_4x4(const int16x4_t a0,
                                            const int16x4_t a1,
                                            const int16x4_t a2,
                                            const int16x4_t a3, int16x8_t *b0,
                                            int16x8_t *b1) {
  // Transpose 16-bit elements:
  // a0: 00, 01, 02, 03
  // a1: 10, 11, 12, 13
  // a2: 20, 21, 22, 23
  // a3: 30, 31, 32, 33
  //
  // b0: 00 10 20 30 01 11 21 31
  // b1: 02 12 22 32 03 13 23 33

  int16x8_t a0q = vcombine_s16(a0, vdup_n_s16(0));
  int16x8_t a1q = vcombine_s16(a1, vdup_n_s16(0));
  int16x8_t a2q = vcombine_s16(a2, vdup_n_s16(0));
  int16x8_t a3q = vcombine_s16(a3, vdup_n_s16(0));

  int16x8_t a02 = vzipq_s16(a0q, a2q).val[0];
  int16x8_t a13 = vzipq_s16(a1q, a3q).val[0];

  int16x8x2_t a0123 = vzipq_s16(a02, a13);

  *b0 = a0123.val[0];
  *b1 = a0123.val[1];
}

static INLINE void transpose_concat_s16_8x4(const int16x8_t a0,
                                            const int16x8_t a1,
                                            const int16x8_t a2,
                                            const int16x8_t a3, int16x8_t *b0,
                                            int16x8_t *b1, int16x8_t *b2,
                                            int16x8_t *b3) {
  // Transpose 16-bit elements:
  // a0: 00, 01, 02, 03, 04, 05, 06, 07
  // a1: 10, 11, 12, 13, 14, 15, 16, 17
  // a2: 20, 21, 22, 23, 24, 25, 26, 27
  // a3: 30, 31, 32, 33, 34, 35, 36, 37
  //
  // b0: 00 10 20 30 01 11 21 31
  // b1: 02 12 22 32 03 13 23 33
  // b2: 04 14 24 34 05 15 25 35
  // b3: 06 16 26 36 07 17 27 37

  int16x8x2_t a02 = vzipq_s16(a0, a2);
  int16x8x2_t a13 = vzipq_s16(a1, a3);

  int16x8x2_t a0123_lo = vzipq_s16(a02.val[0], a13.val[0]);
  int16x8x2_t a0123_hi = vzipq_s16(a02.val[1], a13.val[1]);

  *b0 = a0123_lo.val[0];
  *b1 = a0123_lo.val[1];
  *b2 = a0123_hi.val[0];
  *b3 = a0123_hi.val[1];
}

static INLINE void transpose_concat_s8_8x4(int8x8_t a0, int8x8_t a1,
                                           int8x8_t a2, int8x8_t a3,
                                           int8x16_t *b0, int8x16_t *b1) {
  // Transpose 8-bit elements and concatenate result rows as follows:
  // a0: 00, 01, 02, 03, 04, 05, 06, 07
  // a1: 10, 11, 12, 13, 14, 15, 16, 17
  // a2: 20, 21, 22, 23, 24, 25, 26, 27
  // a3: 30, 31, 32, 33, 34, 35, 36, 37
  //
  // b0: 00, 10, 20, 30, 01, 11, 21, 31, 02, 12, 22, 32, 03, 13, 23, 33
  // b1: 04, 14, 24, 34, 05, 15, 25, 35, 06, 16, 26, 36, 07, 17, 27, 37

  int8x16_t a0q = vcombine_s8(a0, vdup_n_s8(0));
  int8x16_t a1q = vcombine_s8(a1, vdup_n_s8(0));
  int8x16_t a2q = vcombine_s8(a2, vdup_n_s8(0));
  int8x16_t a3q = vcombine_s8(a3, vdup_n_s8(0));

  int8x16_t a02 = vzipq_s8(a0q, a2q).val[0];
  int8x16_t a13 = vzipq_s8(a1q, a3q).val[0];

  int8x16x2_t a0123 = vzipq_s8(a02, a13);

  *b0 = a0123.val[0];
  *b1 = a0123.val[1];
}

static INLINE void transpose_concat_u8_8x4(uint8x8_t a0, uint8x8_t a1,
                                           uint8x8_t a2, uint8x8_t a3,
                                           uint8x16_t *b0, uint8x16_t *b1) {
  // Transpose 8-bit elements and concatenate result rows as follows:
  // a0: 00, 01, 02, 03, 04, 05, 06, 07
  // a1: 10, 11, 12, 13, 14, 15, 16, 17
  // a2: 20, 21, 22, 23, 24, 25, 26, 27
  // a3: 30, 31, 32, 33, 34, 35, 36, 37
  //
  // b0: 00, 10, 20, 30, 01, 11, 21, 31, 02, 12, 22, 32, 03, 13, 23, 33
  // b1: 04, 14, 24, 34, 05, 15, 25, 35, 06, 16, 26, 36, 07, 17, 27, 37

  uint8x16_t a0q = vcombine_u8(a0, vdup_n_u8(0));
  uint8x16_t a1q = vcombine_u8(a1, vdup_n_u8(0));
  uint8x16_t a2q = vcombine_u8(a2, vdup_n_u8(0));
  uint8x16_t a3q = vcombine_u8(a3, vdup_n_u8(0));

  uint8x16_t a02 = vzipq_u8(a0q, a2q).val[0];
  uint8x16_t a13 = vzipq_u8(a1q, a3q).val[0];

  uint8x16x2_t a0123 = vzipq_u8(a02, a13);

  *b0 = a0123.val[0];
  *b1 = a0123.val[1];
}

static INLINE void transpose_concat_s8_4x4(int8x8_t a0, int8x8_t a1,
                                           int8x8_t a2, int8x8_t a3,
                                           int8x16_t *b) {
  // Transpose 8-bit elements and concatenate result rows as follows:
  // a0: 00, 01, 02, 03, XX, XX, XX, XX
  // a1: 10, 11, 12, 13, XX, XX, XX, XX
  // a2: 20, 21, 22, 23, XX, XX, XX, XX
  // a3: 30, 31, 32, 33, XX, XX, XX, XX
  //
  // b: 00, 10, 20, 30, 01, 11, 21, 31, 02, 12, 22, 32, 03, 13, 23, 33

  int8x16_t a0q = vcombine_s8(a0, vdup_n_s8(0));
  int8x16_t a1q = vcombine_s8(a1, vdup_n_s8(0));
  int8x16_t a2q = vcombine_s8(a2, vdup_n_s8(0));
  int8x16_t a3q = vcombine_s8(a3, vdup_n_s8(0));

  int8x16_t a02 = vzipq_s8(a0q, a2q).val[0];
  int8x16_t a13 = vzipq_s8(a1q, a3q).val[0];

  *b = vzipq_s8(a02, a13).val[0];
}

static INLINE void transpose_concat_u8_4x4(uint8x8_t a0, uint8x8_t a1,
                                           uint8x8_t a2, uint8x8_t a3,
                                           uint8x16_t *b) {
  // Transpose 8-bit elements and concatenate result rows as follows:
  // a0: 00, 01, 02, 03, XX, XX, XX, XX
  // a1: 10, 11, 12, 13, XX, XX, XX, XX
  // a2: 20, 21, 22, 23, XX, XX, XX, XX
  // a3: 30, 31, 32, 33, XX, XX, XX, XX
  //
  // b: 00, 10, 20, 30, 01, 11, 21, 31, 02, 12, 22, 32, 03, 13, 23, 33

  uint8x16_t a0q = vcombine_u8(a0, vdup_n_u8(0));
  uint8x16_t a1q = vcombine_u8(a1, vdup_n_u8(0));
  uint8x16_t a2q = vcombine_u8(a2, vdup_n_u8(0));
  uint8x16_t a3q = vcombine_u8(a3, vdup_n_u8(0));

  uint8x16_t a02 = vzipq_u8(a0q, a2q).val[0];
  uint8x16_t a13 = vzipq_u8(a1q, a3q).val[0];

  *b = vzipq_u8(a02, a13).val[0];
}

#endif  // VPX_VPX_DSP_ARM_TRANSPOSE_NEON_H_
