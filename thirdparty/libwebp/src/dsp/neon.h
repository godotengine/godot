// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//  NEON common code.

#ifndef WEBP_DSP_NEON_H_
#define WEBP_DSP_NEON_H_

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_NEON)

#include <arm_neon.h>

// Right now, some intrinsics functions seem slower, so we disable them
// everywhere except newer clang/gcc or aarch64 where the inline assembly is
// incompatible.
#if LOCAL_CLANG_PREREQ(3, 8) || LOCAL_GCC_PREREQ(4, 9) || WEBP_AARCH64
#define WEBP_USE_INTRINSICS   // use intrinsics when possible
#endif

#define INIT_VECTOR2(v, a, b) do {  \
  v.val[0] = a;                     \
  v.val[1] = b;                     \
} while (0)

#define INIT_VECTOR3(v, a, b, c) do {  \
  v.val[0] = a;                        \
  v.val[1] = b;                        \
  v.val[2] = c;                        \
} while (0)

#define INIT_VECTOR4(v, a, b, c, d) do {  \
  v.val[0] = a;                           \
  v.val[1] = b;                           \
  v.val[2] = c;                           \
  v.val[3] = d;                           \
} while (0)

// if using intrinsics, this flag avoids some functions that make gcc-4.6.3
// crash ("internal compiler error: in immed_double_const, at emit-rtl.").
// (probably similar to gcc.gnu.org/bugzilla/show_bug.cgi?id=48183)
#if !(LOCAL_CLANG_PREREQ(3, 8) || LOCAL_GCC_PREREQ(4, 8) || WEBP_AARCH64)
#define WORK_AROUND_GCC
#endif

static WEBP_INLINE int32x4x4_t Transpose4x4_NEON(const int32x4x4_t rows) {
  uint64x2x2_t row01, row23;

  row01.val[0] = vreinterpretq_u64_s32(rows.val[0]);
  row01.val[1] = vreinterpretq_u64_s32(rows.val[1]);
  row23.val[0] = vreinterpretq_u64_s32(rows.val[2]);
  row23.val[1] = vreinterpretq_u64_s32(rows.val[3]);
  // Transpose 64-bit values (there's no vswp equivalent)
  {
    const uint64x1_t row0h = vget_high_u64(row01.val[0]);
    const uint64x1_t row2l = vget_low_u64(row23.val[0]);
    const uint64x1_t row1h = vget_high_u64(row01.val[1]);
    const uint64x1_t row3l = vget_low_u64(row23.val[1]);
    row01.val[0] = vcombine_u64(vget_low_u64(row01.val[0]), row2l);
    row23.val[0] = vcombine_u64(row0h, vget_high_u64(row23.val[0]));
    row01.val[1] = vcombine_u64(vget_low_u64(row01.val[1]), row3l);
    row23.val[1] = vcombine_u64(row1h, vget_high_u64(row23.val[1]));
  }
  {
    const int32x4x2_t out01 = vtrnq_s32(vreinterpretq_s32_u64(row01.val[0]),
                                        vreinterpretq_s32_u64(row01.val[1]));
    const int32x4x2_t out23 = vtrnq_s32(vreinterpretq_s32_u64(row23.val[0]),
                                        vreinterpretq_s32_u64(row23.val[1]));
    int32x4x4_t out;
    out.val[0] = out01.val[0];
    out.val[1] = out01.val[1];
    out.val[2] = out23.val[0];
    out.val[3] = out23.val[1];
    return out;
  }
}

#if 0     // Useful debug macro.
#include <stdio.h>
#define PRINT_REG(REG, SIZE) do {                       \
  int i;                                                \
  printf("%s \t[%d]: 0x", #REG, SIZE);                  \
  if (SIZE == 8) {                                      \
    uint8_t _tmp[8];                                    \
    vst1_u8(_tmp, (REG));                               \
    for (i = 0; i < 8; ++i) printf("%.2x ", _tmp[i]);   \
  } else if (SIZE == 16) {                              \
    uint16_t _tmp[4];                                   \
    vst1_u16(_tmp, (REG));                              \
    for (i = 0; i < 4; ++i) printf("%.4x ", _tmp[i]);   \
  }                                                     \
  printf("\n");                                         \
} while (0)
#endif

#endif  // WEBP_USE_NEON
#endif  // WEBP_DSP_NEON_H_
