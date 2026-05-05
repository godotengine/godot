/*
 *  Copyright (c) 2024 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_VPX_NEON_SVE2_BRIDGE_H_
#define VPX_VPX_DSP_ARM_VPX_NEON_SVE2_BRIDGE_H_

#include <arm_neon.h>
#include <arm_sve.h>
#include <arm_neon_sve_bridge.h>

// Some very useful instructions are exclusive to the SVE2 instruction set.
// However, we can access these instructions from a predominantly Neon context
// by making use of the Neon-SVE bridge intrinsics to reinterpret Neon vectors
// as SVE vectors - with the high part of the SVE vector (if it's longer than
// 128 bits) being "don't care".

static INLINE int16x8_t vpx_tbl2_s16(int16x8_t s0, int16x8_t s1,
                                     uint16x8_t tbl) {
  svint16x2_t samples = svcreate2_s16(svset_neonq_s16(svundef_s16(), s0),
                                      svset_neonq_s16(svundef_s16(), s1));
  return svget_neonq_s16(
      svtbl2_s16(samples, svset_neonq_u16(svundef_u16(), tbl)));
}

static INLINE void vpx_tbl2x4_s16(int16x8_t s0[4], int16x8_t s1[4],
                                  int16x8_t res[4], uint16x8_t idx) {
  res[0] = vpx_tbl2_s16(s0[0], s1[0], idx);
  res[1] = vpx_tbl2_s16(s0[1], s1[1], idx);
  res[2] = vpx_tbl2_s16(s0[2], s1[2], idx);
  res[3] = vpx_tbl2_s16(s0[3], s1[3], idx);
}

static INLINE void vpx_tbl2x2_s16(int16x8_t s0[2], int16x8_t s1[2],
                                  int16x8_t res[2], uint16x8_t idx) {
  res[0] = vpx_tbl2_s16(s0[0], s1[0], idx);
  res[1] = vpx_tbl2_s16(s0[1], s1[1], idx);
}

#endif  // VPX_VPX_DSP_ARM_VPX_NEON_SVE2_BRIDGE_H_
