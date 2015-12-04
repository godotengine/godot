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

#include <arm_neon.h>

#include "./dsp.h"

// Right now, some intrinsics functions seem slower, so we disable them
// everywhere except aarch64 where the inline assembly is incompatible.
#if defined(__aarch64__)
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
#if !(LOCAL_GCC_PREREQ(4,8) || defined(__aarch64__))
#define WORK_AROUND_GCC
#endif

static WEBP_INLINE int32x4x4_t Transpose4x4(const int32x4x4_t rows) {
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

#endif  // WEBP_DSP_NEON_H_
