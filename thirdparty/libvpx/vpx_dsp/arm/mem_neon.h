/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_ARM_MEM_NEON_H_
#define VPX_VPX_DSP_ARM_MEM_NEON_H_

#include <arm_neon.h>
#include <assert.h>
#include <string.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"

// Support for these xN intrinsics is lacking in older versions of GCC.
#if defined(__GNUC__) && !defined(__clang__)
#if __GNUC__ < 8 || defined(__arm__)
static INLINE uint8x16x2_t vld1q_u8_x2(uint8_t const *ptr) {
  uint8x16x2_t res = { { vld1q_u8(ptr + 0 * 16), vld1q_u8(ptr + 1 * 16) } };
  return res;
}
#endif

#if __GNUC__ < 9 || defined(__arm__)
static INLINE uint8x16x3_t vld1q_u8_x3(uint8_t const *ptr) {
  uint8x16x3_t res = { { vld1q_u8(ptr + 0 * 16), vld1q_u8(ptr + 1 * 16),
                         vld1q_u8(ptr + 2 * 16) } };
  return res;
}
#endif
#endif

static INLINE int16x4_t create_s16x4_neon(const int16_t c0, const int16_t c1,
                                          const int16_t c2, const int16_t c3) {
  return vcreate_s16((uint16_t)c0 | ((uint32_t)c1 << 16) |
                     ((uint64_t)(uint16_t)c2 << 32) | ((uint64_t)c3 << 48));
}

static INLINE int32x2_t create_s32x2_neon(const int32_t c0, const int32_t c1) {
  return vcreate_s32((uint32_t)c0 | ((uint64_t)(uint32_t)c1 << 32));
}

static INLINE int32x4_t create_s32x4_neon(const int32_t c0, const int32_t c1,
                                          const int32_t c2, const int32_t c3) {
  return vcombine_s32(create_s32x2_neon(c0, c1), create_s32x2_neon(c2, c3));
}

// Helper functions used to load tran_low_t into int16, narrowing if necessary.
static INLINE int16x8x2_t load_tran_low_to_s16x2q(const tran_low_t *buf) {
#if CONFIG_VP9_HIGHBITDEPTH
  const int32x4x2_t v0 = vld2q_s32(buf);
  const int32x4x2_t v1 = vld2q_s32(buf + 8);
  const int16x4_t s0 = vmovn_s32(v0.val[0]);
  const int16x4_t s1 = vmovn_s32(v0.val[1]);
  const int16x4_t s2 = vmovn_s32(v1.val[0]);
  const int16x4_t s3 = vmovn_s32(v1.val[1]);
  int16x8x2_t res;
  res.val[0] = vcombine_s16(s0, s2);
  res.val[1] = vcombine_s16(s1, s3);
  return res;
#else
  return vld2q_s16(buf);
#endif
}

static INLINE int16x8_t load_tran_low_to_s16q(const tran_low_t *buf) {
#if CONFIG_VP9_HIGHBITDEPTH
  const int32x4_t v0 = vld1q_s32(buf);
  const int32x4_t v1 = vld1q_s32(buf + 4);
  const int16x4_t s0 = vmovn_s32(v0);
  const int16x4_t s1 = vmovn_s32(v1);
  return vcombine_s16(s0, s1);
#else
  return vld1q_s16(buf);
#endif
}

static INLINE int16x4_t load_tran_low_to_s16d(const tran_low_t *buf) {
#if CONFIG_VP9_HIGHBITDEPTH
  const int32x4_t v0 = vld1q_s32(buf);
  return vmovn_s32(v0);
#else
  return vld1_s16(buf);
#endif
}

static INLINE void store_s16q_to_tran_low(tran_low_t *buf, const int16x8_t a) {
#if CONFIG_VP9_HIGHBITDEPTH
  const int32x4_t v0 = vmovl_s16(vget_low_s16(a));
  const int32x4_t v1 = vmovl_s16(vget_high_s16(a));
  vst1q_s32(buf, v0);
  vst1q_s32(buf + 4, v1);
#else
  vst1q_s16(buf, a);
#endif
}

#if CONFIG_VP9_HIGHBITDEPTH
static INLINE void store_s32q_to_tran_low(tran_low_t *buf, const int32x4_t a) {
  vst1q_s32(buf, a);
}

static INLINE int32x4_t load_tran_low_to_s32q(const tran_low_t *buf) {
  return vld1q_s32(buf);
}
#endif

// Propagate type information to the compiler. Without this the compiler may
// assume the required alignment of uint32_t (4 bytes) and add alignment hints
// to the memory access.
//
// This is used for functions operating on uint8_t which wish to load or store 4
// values at a time but which may not be on 4 byte boundaries.
static INLINE void uint32_to_mem(uint8_t *buf, uint32_t a) {
  memcpy(buf, &a, 4);
}

// Load 4 contiguous bytes when alignment is not guaranteed.
static INLINE uint8x8_t load_unaligned_u8_4x1(const uint8_t *buf) {
  uint32_t a;
  uint32x2_t a_u32;
  memcpy(&a, buf, 4);
  a_u32 = vdup_n_u32(0);
  a_u32 = vset_lane_u32(a, a_u32, 0);
  return vreinterpret_u8_u32(a_u32);
}

// Load 4 contiguous bytes and replicate across a vector when alignment is not
// guaranteed.
static INLINE uint8x8_t load_replicate_u8_4x1(const uint8_t *buf) {
  uint32_t a;
  memcpy(&a, buf, 4);
  return vreinterpret_u8_u32(vdup_n_u32(a));
}

// Store 4 contiguous bytes from the low half of an 8x8 vector.
static INLINE void store_u8_4x1(uint8_t *buf, uint8x8_t a) {
  vst1_lane_u32((uint32_t *)buf, vreinterpret_u32_u8(a), 0);
}

// Store 4 contiguous bytes from the high half of an 8x8 vector.
static INLINE void store_u8_4x1_high(uint8_t *buf, uint8x8_t a) {
  vst1_lane_u32((uint32_t *)buf, vreinterpret_u32_u8(a), 1);
}

// Load 2 sets of 4 bytes when alignment is not guaranteed.
static INLINE uint8x8_t load_unaligned_u8(const uint8_t *buf,
                                          ptrdiff_t stride) {
  uint32_t a;
  uint32x2_t a_u32 = vdup_n_u32(0);
  memcpy(&a, buf, 4);
  buf += stride;
  a_u32 = vset_lane_u32(a, a_u32, 0);
  memcpy(&a, buf, 4);
  a_u32 = vset_lane_u32(a, a_u32, 1);
  return vreinterpret_u8_u32(a_u32);
}

// Load 8 bytes when alignment is not guaranteed.
static INLINE uint16x4_t load_unaligned_u16(const uint16_t *buf) {
  uint64_t a;
  uint64x1_t a_u64 = vdup_n_u64(0);
  memcpy(&a, buf, 8);
  a_u64 = vset_lane_u64(a, a_u64, 0);
  return vreinterpret_u16_u64(a_u64);
}

// Load 2 sets of 8 bytes when alignment is not guaranteed.
static INLINE uint16x8_t load_unaligned_u16q(const uint16_t *buf,
                                             ptrdiff_t stride) {
  uint64_t a;
  uint64x2_t a_u64 = vdupq_n_u64(0);
  memcpy(&a, buf, 8);
  buf += stride;
  a_u64 = vsetq_lane_u64(a, a_u64, 0);
  memcpy(&a, buf, 8);
  a_u64 = vsetq_lane_u64(a, a_u64, 1);
  return vreinterpretq_u16_u64(a_u64);
}

// Store 2 sets of 4 bytes when alignment is not guaranteed.
static INLINE void store_unaligned_u8(uint8_t *buf, ptrdiff_t stride,
                                      const uint8x8_t a) {
  const uint32x2_t a_u32 = vreinterpret_u32_u8(a);
  uint32_to_mem(buf, vget_lane_u32(a_u32, 0));
  buf += stride;
  uint32_to_mem(buf, vget_lane_u32(a_u32, 1));
}

// Load 4 sets of 4 bytes when alignment is not guaranteed.
static INLINE uint8x16_t load_unaligned_u8q(const uint8_t *buf,
                                            ptrdiff_t stride) {
  uint32_t a;
  uint32x4_t a_u32 = vdupq_n_u32(0);
  memcpy(&a, buf, 4);
  buf += stride;
  a_u32 = vsetq_lane_u32(a, a_u32, 0);
  memcpy(&a, buf, 4);
  buf += stride;
  a_u32 = vsetq_lane_u32(a, a_u32, 1);
  memcpy(&a, buf, 4);
  buf += stride;
  a_u32 = vsetq_lane_u32(a, a_u32, 2);
  memcpy(&a, buf, 4);
  buf += stride;
  a_u32 = vsetq_lane_u32(a, a_u32, 3);
  return vreinterpretq_u8_u32(a_u32);
}

// Store 4 sets of 4 bytes when alignment is not guaranteed.
static INLINE void store_unaligned_u8q(uint8_t *buf, ptrdiff_t stride,
                                       const uint8x16_t a) {
  const uint32x4_t a_u32 = vreinterpretq_u32_u8(a);
  uint32_to_mem(buf, vgetq_lane_u32(a_u32, 0));
  buf += stride;
  uint32_to_mem(buf, vgetq_lane_u32(a_u32, 1));
  buf += stride;
  uint32_to_mem(buf, vgetq_lane_u32(a_u32, 2));
  buf += stride;
  uint32_to_mem(buf, vgetq_lane_u32(a_u32, 3));
}

// Load 2 sets of 4 bytes when alignment is guaranteed.
static INLINE uint8x8_t load_u8(const uint8_t *buf, ptrdiff_t stride) {
  uint32x2_t a = vdup_n_u32(0);

  assert(!((intptr_t)buf % sizeof(uint32_t)));
  assert(!(stride % sizeof(uint32_t)));

  a = vld1_lane_u32((const uint32_t *)buf, a, 0);
  buf += stride;
  a = vld1_lane_u32((const uint32_t *)buf, a, 1);
  return vreinterpret_u8_u32(a);
}

// Store 2 sets of 4 bytes when alignment is guaranteed.
static INLINE void store_u8(uint8_t *buf, ptrdiff_t stride, const uint8x8_t a) {
  uint32x2_t a_u32 = vreinterpret_u32_u8(a);

  assert(!((intptr_t)buf % sizeof(uint32_t)));
  assert(!(stride % sizeof(uint32_t)));

  vst1_lane_u32((uint32_t *)buf, a_u32, 0);
  buf += stride;
  vst1_lane_u32((uint32_t *)buf, a_u32, 1);
}

static INLINE void store_u8_8x3(uint8_t *s, const ptrdiff_t p,
                                const uint8x8_t s0, const uint8x8_t s1,
                                const uint8x8_t s2) {
  vst1_u8(s, s0);
  s += p;
  vst1_u8(s, s1);
  s += p;
  vst1_u8(s, s2);
}

static INLINE void load_u8_8x3(const uint8_t *s, const ptrdiff_t p,
                               uint8x8_t *const s0, uint8x8_t *const s1,
                               uint8x8_t *const s2) {
  *s0 = vld1_u8(s);
  s += p;
  *s1 = vld1_u8(s);
  s += p;
  *s2 = vld1_u8(s);
}

static INLINE void load_u8_8x4(const uint8_t *s, const ptrdiff_t p,
                               uint8x8_t *const s0, uint8x8_t *const s1,
                               uint8x8_t *const s2, uint8x8_t *const s3) {
  *s0 = vld1_u8(s);
  s += p;
  *s1 = vld1_u8(s);
  s += p;
  *s2 = vld1_u8(s);
  s += p;
  *s3 = vld1_u8(s);
}

static INLINE void store_u8_8x4(uint8_t *s, const ptrdiff_t p,
                                const uint8x8_t s0, const uint8x8_t s1,
                                const uint8x8_t s2, const uint8x8_t s3) {
  vst1_u8(s, s0);
  s += p;
  vst1_u8(s, s1);
  s += p;
  vst1_u8(s, s2);
  s += p;
  vst1_u8(s, s3);
}

static INLINE void load_u8_16x3(const uint8_t *s, const ptrdiff_t p,
                                uint8x16_t *const s0, uint8x16_t *const s1,
                                uint8x16_t *const s2) {
  *s0 = vld1q_u8(s);
  s += p;
  *s1 = vld1q_u8(s);
  s += p;
  *s2 = vld1q_u8(s);
}

static INLINE void load_u8_16x4(const uint8_t *s, const ptrdiff_t p,
                                uint8x16_t *const s0, uint8x16_t *const s1,
                                uint8x16_t *const s2, uint8x16_t *const s3) {
  *s0 = vld1q_u8(s);
  s += p;
  *s1 = vld1q_u8(s);
  s += p;
  *s2 = vld1q_u8(s);
  s += p;
  *s3 = vld1q_u8(s);
}

static INLINE void store_u8_16x4(uint8_t *s, const ptrdiff_t p,
                                 const uint8x16_t s0, const uint8x16_t s1,
                                 const uint8x16_t s2, const uint8x16_t s3) {
  vst1q_u8(s, s0);
  s += p;
  vst1q_u8(s, s1);
  s += p;
  vst1q_u8(s, s2);
  s += p;
  vst1q_u8(s, s3);
}

static INLINE void load_u8_8x7(const uint8_t *s, const ptrdiff_t p,
                               uint8x8_t *const s0, uint8x8_t *const s1,
                               uint8x8_t *const s2, uint8x8_t *const s3,
                               uint8x8_t *const s4, uint8x8_t *const s5,
                               uint8x8_t *const s6) {
  *s0 = vld1_u8(s);
  s += p;
  *s1 = vld1_u8(s);
  s += p;
  *s2 = vld1_u8(s);
  s += p;
  *s3 = vld1_u8(s);
  s += p;
  *s4 = vld1_u8(s);
  s += p;
  *s5 = vld1_u8(s);
  s += p;
  *s6 = vld1_u8(s);
}

static INLINE void load_u8_8x8(const uint8_t *s, const ptrdiff_t p,
                               uint8x8_t *const s0, uint8x8_t *const s1,
                               uint8x8_t *const s2, uint8x8_t *const s3,
                               uint8x8_t *const s4, uint8x8_t *const s5,
                               uint8x8_t *const s6, uint8x8_t *const s7) {
  *s0 = vld1_u8(s);
  s += p;
  *s1 = vld1_u8(s);
  s += p;
  *s2 = vld1_u8(s);
  s += p;
  *s3 = vld1_u8(s);
  s += p;
  *s4 = vld1_u8(s);
  s += p;
  *s5 = vld1_u8(s);
  s += p;
  *s6 = vld1_u8(s);
  s += p;
  *s7 = vld1_u8(s);
}

static INLINE void load_u8_8x11(const uint8_t *s, ptrdiff_t p,
                                uint8x8_t *const s0, uint8x8_t *const s1,
                                uint8x8_t *const s2, uint8x8_t *const s3,
                                uint8x8_t *const s4, uint8x8_t *const s5,
                                uint8x8_t *const s6, uint8x8_t *const s7,
                                uint8x8_t *const s8, uint8x8_t *const s9,
                                uint8x8_t *const s10) {
  *s0 = vld1_u8(s);
  s += p;
  *s1 = vld1_u8(s);
  s += p;
  *s2 = vld1_u8(s);
  s += p;
  *s3 = vld1_u8(s);
  s += p;
  *s4 = vld1_u8(s);
  s += p;
  *s5 = vld1_u8(s);
  s += p;
  *s6 = vld1_u8(s);
  s += p;
  *s7 = vld1_u8(s);
  s += p;
  *s8 = vld1_u8(s);
  s += p;
  *s9 = vld1_u8(s);
  s += p;
  *s10 = vld1_u8(s);
}

static INLINE void store_u8_8x8(uint8_t *s, const ptrdiff_t p,
                                const uint8x8_t s0, const uint8x8_t s1,
                                const uint8x8_t s2, const uint8x8_t s3,
                                const uint8x8_t s4, const uint8x8_t s5,
                                const uint8x8_t s6, const uint8x8_t s7) {
  vst1_u8(s, s0);
  s += p;
  vst1_u8(s, s1);
  s += p;
  vst1_u8(s, s2);
  s += p;
  vst1_u8(s, s3);
  s += p;
  vst1_u8(s, s4);
  s += p;
  vst1_u8(s, s5);
  s += p;
  vst1_u8(s, s6);
  s += p;
  vst1_u8(s, s7);
}

static INLINE void load_u8_16x8(const uint8_t *s, const ptrdiff_t p,
                                uint8x16_t *const s0, uint8x16_t *const s1,
                                uint8x16_t *const s2, uint8x16_t *const s3,
                                uint8x16_t *const s4, uint8x16_t *const s5,
                                uint8x16_t *const s6, uint8x16_t *const s7) {
  *s0 = vld1q_u8(s);
  s += p;
  *s1 = vld1q_u8(s);
  s += p;
  *s2 = vld1q_u8(s);
  s += p;
  *s3 = vld1q_u8(s);
  s += p;
  *s4 = vld1q_u8(s);
  s += p;
  *s5 = vld1q_u8(s);
  s += p;
  *s6 = vld1q_u8(s);
  s += p;
  *s7 = vld1q_u8(s);
}

static INLINE void store_u8_16x8(uint8_t *s, const ptrdiff_t p,
                                 const uint8x16_t s0, const uint8x16_t s1,
                                 const uint8x16_t s2, const uint8x16_t s3,
                                 const uint8x16_t s4, const uint8x16_t s5,
                                 const uint8x16_t s6, const uint8x16_t s7) {
  vst1q_u8(s, s0);
  s += p;
  vst1q_u8(s, s1);
  s += p;
  vst1q_u8(s, s2);
  s += p;
  vst1q_u8(s, s3);
  s += p;
  vst1q_u8(s, s4);
  s += p;
  vst1q_u8(s, s5);
  s += p;
  vst1q_u8(s, s6);
  s += p;
  vst1q_u8(s, s7);
}

static INLINE void store_u16_4x3(uint16_t *s, const ptrdiff_t p,
                                 const uint16x4_t s0, const uint16x4_t s1,
                                 const uint16x4_t s2) {
  vst1_u16(s, s0);
  s += p;
  vst1_u16(s, s1);
  s += p;
  vst1_u16(s, s2);
}

static INLINE void load_s16_4x3(const int16_t *s, const ptrdiff_t p,
                                int16x4_t *s0, int16x4_t *s1, int16x4_t *s2) {
  *s0 = vld1_s16(s);
  s += p;
  *s1 = vld1_s16(s);
  s += p;
  *s2 = vld1_s16(s);
}

static INLINE void load_s16_4x4(const int16_t *s, const ptrdiff_t p,
                                int16x4_t *s0, int16x4_t *s1, int16x4_t *s2,
                                int16x4_t *s3) {
  *s0 = vld1_s16(s);
  s += p;
  *s1 = vld1_s16(s);
  s += p;
  *s2 = vld1_s16(s);
  s += p;
  *s3 = vld1_s16(s);
}

static INLINE void load_s16_4x11(const int16_t *s, const ptrdiff_t p,
                                 int16x4_t *s0, int16x4_t *s1, int16x4_t *s2,
                                 int16x4_t *s3, int16x4_t *s4, int16x4_t *s5,
                                 int16x4_t *s6, int16x4_t *s7, int16x4_t *s8,
                                 int16x4_t *s9, int16x4_t *s10) {
  *s0 = vld1_s16(s);
  s += p;
  *s1 = vld1_s16(s);
  s += p;
  *s2 = vld1_s16(s);
  s += p;
  *s3 = vld1_s16(s);
  s += p;
  *s4 = vld1_s16(s);
  s += p;
  *s5 = vld1_s16(s);
  s += p;
  *s6 = vld1_s16(s);
  s += p;
  *s7 = vld1_s16(s);
  s += p;
  *s8 = vld1_s16(s);
  s += p;
  *s9 = vld1_s16(s);
  s += p;
  *s10 = vld1_s16(s);
}

static INLINE void store_u16_4x4(uint16_t *s, const ptrdiff_t p,
                                 const uint16x4_t s0, const uint16x4_t s1,
                                 const uint16x4_t s2, const uint16x4_t s3) {
  vst1_u16(s, s0);
  s += p;
  vst1_u16(s, s1);
  s += p;
  vst1_u16(s, s2);
  s += p;
  vst1_u16(s, s3);
}

static INLINE void load_s16_4x7(const int16_t *s, const ptrdiff_t p,
                                int16x4_t *s0, int16x4_t *s1, int16x4_t *s2,
                                int16x4_t *s3, int16x4_t *s4, int16x4_t *s5,
                                int16x4_t *s6) {
  *s0 = vld1_s16(s);
  s += p;
  *s1 = vld1_s16(s);
  s += p;
  *s2 = vld1_s16(s);
  s += p;
  *s3 = vld1_s16(s);
  s += p;
  *s4 = vld1_s16(s);
  s += p;
  *s5 = vld1_s16(s);
  s += p;
  *s6 = vld1_s16(s);
}

static INLINE void load_s16_8x3(const int16_t *s, const ptrdiff_t p,
                                int16x8_t *s0, int16x8_t *s1, int16x8_t *s2) {
  *s0 = vld1q_s16(s);
  s += p;
  *s1 = vld1q_s16(s);
  s += p;
  *s2 = vld1q_s16(s);
}

static INLINE void load_s16_8x4(const int16_t *s, const ptrdiff_t p,
                                int16x8_t *s0, int16x8_t *s1, int16x8_t *s2,
                                int16x8_t *s3) {
  *s0 = vld1q_s16(s);
  s += p;
  *s1 = vld1q_s16(s);
  s += p;
  *s2 = vld1q_s16(s);
  s += p;
  *s3 = vld1q_s16(s);
}

static INLINE void load_u16_8x4(const uint16_t *s, const ptrdiff_t p,
                                uint16x8_t *s0, uint16x8_t *s1, uint16x8_t *s2,
                                uint16x8_t *s3) {
  *s0 = vld1q_u16(s);
  s += p;
  *s1 = vld1q_u16(s);
  s += p;
  *s2 = vld1q_u16(s);
  s += p;
  *s3 = vld1q_u16(s);
}

static INLINE void store_u16_8x4(uint16_t *s, const ptrdiff_t p,
                                 const uint16x8_t s0, const uint16x8_t s1,
                                 const uint16x8_t s2, const uint16x8_t s3) {
  vst1q_u16(s, s0);
  s += p;
  vst1q_u16(s, s1);
  s += p;
  vst1q_u16(s, s2);
  s += p;
  vst1q_u16(s, s3);
}

static INLINE void store_u16_8x3(uint16_t *s, const ptrdiff_t p,
                                 const uint16x8_t s0, const uint16x8_t s1,
                                 const uint16x8_t s2) {
  vst1q_u16(s, s0);
  s += p;
  vst1q_u16(s, s1);
  s += p;
  vst1q_u16(s, s2);
}

static INLINE void load_s16_8x7(const int16_t *s, const ptrdiff_t p,
                                int16x8_t *s0, int16x8_t *s1, int16x8_t *s2,
                                int16x8_t *s3, int16x8_t *s4, int16x8_t *s5,
                                int16x8_t *s6) {
  *s0 = vld1q_s16(s);
  s += p;
  *s1 = vld1q_s16(s);
  s += p;
  *s2 = vld1q_s16(s);
  s += p;
  *s3 = vld1q_s16(s);
  s += p;
  *s4 = vld1q_s16(s);
  s += p;
  *s5 = vld1q_s16(s);
  s += p;
  *s6 = vld1q_s16(s);
}

static INLINE void load_u16_8x8(const uint16_t *s, const ptrdiff_t p,
                                uint16x8_t *s0, uint16x8_t *s1, uint16x8_t *s2,
                                uint16x8_t *s3, uint16x8_t *s4, uint16x8_t *s5,
                                uint16x8_t *s6, uint16x8_t *s7) {
  *s0 = vld1q_u16(s);
  s += p;
  *s1 = vld1q_u16(s);
  s += p;
  *s2 = vld1q_u16(s);
  s += p;
  *s3 = vld1q_u16(s);
  s += p;
  *s4 = vld1q_u16(s);
  s += p;
  *s5 = vld1q_u16(s);
  s += p;
  *s6 = vld1q_u16(s);
  s += p;
  *s7 = vld1q_u16(s);
}

static INLINE void load_s16_4x8(const int16_t *s, const ptrdiff_t p,
                                int16x4_t *s0, int16x4_t *s1, int16x4_t *s2,
                                int16x4_t *s3, int16x4_t *s4, int16x4_t *s5,
                                int16x4_t *s6, int16x4_t *s7) {
  *s0 = vld1_s16(s);
  s += p;
  *s1 = vld1_s16(s);
  s += p;
  *s2 = vld1_s16(s);
  s += p;
  *s3 = vld1_s16(s);
  s += p;
  *s4 = vld1_s16(s);
  s += p;
  *s5 = vld1_s16(s);
  s += p;
  *s6 = vld1_s16(s);
  s += p;
  *s7 = vld1_s16(s);
}

static INLINE void load_s16_8x11(const int16_t *s, const ptrdiff_t p,
                                 int16x8_t *s0, int16x8_t *s1, int16x8_t *s2,
                                 int16x8_t *s3, int16x8_t *s4, int16x8_t *s5,
                                 int16x8_t *s6, int16x8_t *s7, int16x8_t *s8,
                                 int16x8_t *s9, int16x8_t *s10) {
  *s0 = vld1q_s16(s);
  s += p;
  *s1 = vld1q_s16(s);
  s += p;
  *s2 = vld1q_s16(s);
  s += p;
  *s3 = vld1q_s16(s);
  s += p;
  *s4 = vld1q_s16(s);
  s += p;
  *s5 = vld1q_s16(s);
  s += p;
  *s6 = vld1q_s16(s);
  s += p;
  *s7 = vld1q_s16(s);
  s += p;
  *s8 = vld1q_s16(s);
  s += p;
  *s9 = vld1q_s16(s);
  s += p;
  *s10 = vld1q_s16(s);
}

static INLINE void load_s16_8x12(const int16_t *s, const ptrdiff_t p,
                                 int16x8_t *s0, int16x8_t *s1, int16x8_t *s2,
                                 int16x8_t *s3, int16x8_t *s4, int16x8_t *s5,
                                 int16x8_t *s6, int16x8_t *s7, int16x8_t *s8,
                                 int16x8_t *s9, int16x8_t *s10,
                                 int16x8_t *s11) {
  *s0 = vld1q_s16(s);
  s += p;
  *s1 = vld1q_s16(s);
  s += p;
  *s2 = vld1q_s16(s);
  s += p;
  *s3 = vld1q_s16(s);
  s += p;
  *s4 = vld1q_s16(s);
  s += p;
  *s5 = vld1q_s16(s);
  s += p;
  *s6 = vld1q_s16(s);
  s += p;
  *s7 = vld1q_s16(s);
  s += p;
  *s8 = vld1q_s16(s);
  s += p;
  *s9 = vld1q_s16(s);
  s += p;
  *s10 = vld1q_s16(s);
  s += p;
  *s11 = vld1q_s16(s);
}

static INLINE void load_s16_8x8(const int16_t *s, const ptrdiff_t p,
                                int16x8_t *s0, int16x8_t *s1, int16x8_t *s2,
                                int16x8_t *s3, int16x8_t *s4, int16x8_t *s5,
                                int16x8_t *s6, int16x8_t *s7) {
  *s0 = vld1q_s16(s);
  s += p;
  *s1 = vld1q_s16(s);
  s += p;
  *s2 = vld1q_s16(s);
  s += p;
  *s3 = vld1q_s16(s);
  s += p;
  *s4 = vld1q_s16(s);
  s += p;
  *s5 = vld1q_s16(s);
  s += p;
  *s6 = vld1q_s16(s);
  s += p;
  *s7 = vld1q_s16(s);
}

#endif  // VPX_VPX_DSP_ARM_MEM_NEON_H_
