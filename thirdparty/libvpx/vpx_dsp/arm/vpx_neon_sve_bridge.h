/*
 *  Copyright (c) 2024 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_VPX_NEON_SVE_BRIDGE_H_
#define VPX_VPX_DSP_ARM_VPX_NEON_SVE_BRIDGE_H_

#include <arm_neon.h>
#include <arm_sve.h>
#include <arm_neon_sve_bridge.h>

// Dot product instructions operating on 16-bit input elements are exclusive to
// the SVE instruction set. However, we can access these instructions from a
// predominantly Neon context by making use of the Neon-SVE bridge intrinsics
// to reinterpret Neon vectors as SVE vectors - with the high part of the SVE
// vector (if it's longer than 128 bits) being "don't care".

// While sub-optimal on machines that have SVE vector length > 128-bit - as the
// remainder of the vector is unused - this approach is still beneficial when
// compared to a Neon-only solution.

static INLINE uint64x2_t vpx_dotq_u16(uint64x2_t acc, uint16x8_t x,
                                      uint16x8_t y) {
  return svget_neonq_u64(svdot_u64(svset_neonq_u64(svundef_u64(), acc),
                                   svset_neonq_u16(svundef_u16(), x),
                                   svset_neonq_u16(svundef_u16(), y)));
}

static INLINE int64x2_t vpx_dotq_s16(int64x2_t acc, int16x8_t x, int16x8_t y) {
  return svget_neonq_s64(svdot_s64(svset_neonq_s64(svundef_s64(), acc),
                                   svset_neonq_s16(svundef_s16(), x),
                                   svset_neonq_s16(svundef_s16(), y)));
}

#define vpx_dotq_lane_s16(acc, x, y, lane)                            \
  svget_neonq_s64(svdot_lane_s64(svset_neonq_s64(svundef_s64(), acc), \
                                 svset_neonq_s16(svundef_s16(), x),   \
                                 svset_neonq_s16(svundef_s16(), y), lane))

static INLINE uint16x8_t vpx_tbl_u16(uint16x8_t data, uint16x8_t indices) {
  return svget_neonq_u16(svtbl_u16(svset_neonq_u16(svundef_u16(), data),
                                   svset_neonq_u16(svundef_u16(), indices)));
}

static INLINE int16x8_t vpx_tbl_s16(int16x8_t data, uint16x8_t indices) {
  return svget_neonq_s16(svtbl_s16(svset_neonq_s16(svundef_s16(), data),
                                   svset_neonq_u16(svundef_u16(), indices)));
}

#endif  // VPX_VPX_DSP_ARM_VPX_NEON_SVE_BRIDGE_H_
