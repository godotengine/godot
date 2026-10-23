/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_PPC_TYPES_VSX_H_
#define VPX_VPX_DSP_PPC_TYPES_VSX_H_

#include <altivec.h>

typedef vector signed char int8x16_t;
typedef vector unsigned char uint8x16_t;
typedef vector signed short int16x8_t;
typedef vector unsigned short uint16x8_t;
typedef vector signed int int32x4_t;
typedef vector unsigned int uint32x4_t;
typedef vector bool char bool8x16_t;
typedef vector bool short bool16x8_t;
typedef vector bool int bool32x4_t;

#if defined(__clang__) && __clang_major__ < 6
static const uint8x16_t xxpermdi0_perm = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                                           0x06, 0x07, 0x10, 0x11, 0x12, 0x13,
                                           0x14, 0x15, 0x16, 0x17 };
static const uint8x16_t xxpermdi1_perm = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
                                           0x06, 0x07, 0x18, 0x19, 0x1A, 0x1B,
                                           0x1C, 0x1D, 0x1E, 0x1F };
static const uint8x16_t xxpermdi2_perm = { 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                           0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13,
                                           0x14, 0x15, 0x16, 0x17 };
static const uint8x16_t xxpermdi3_perm = { 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                           0x0E, 0x0F, 0x18, 0x19, 0x1A, 0x1B,
                                           0x1C, 0x1D, 0x1E, 0x1F };
#define xxpermdi(a, b, c) vec_perm(a, b, xxpermdi##c##_perm)
#elif defined(__GNUC__) && \
    (__GNUC__ > 6 || (__GNUC__ == 6 && __GNUC_MINOR__ >= 3))
#define xxpermdi(a, b, c) vec_xxpermdi(a, b, c)
#endif

#ifdef WORDS_BIGENDIAN
#define unpack_to_u16_h(v) \
  (uint16x8_t) vec_mergeh(vec_splat_u8(0), (uint8x16_t)v)
#define unpack_to_u16_l(v) \
  (uint16x8_t) vec_mergel(vec_splat_u8(0), (uint8x16_t)v)
#define unpack_to_s16_h(v) \
  (int16x8_t) vec_mergeh(vec_splat_u8(0), (uint8x16_t)v)
#define unpack_to_s16_l(v) \
  (int16x8_t) vec_mergel(vec_splat_u8(0), (uint8x16_t)v)
#ifndef xxpermdi
#define xxpermdi(a, b, c) vec_xxpermdi(a, b, c)
#endif
#else
#define unpack_to_u16_h(v) \
  (uint16x8_t) vec_mergeh((uint8x16_t)v, vec_splat_u8(0))
#define unpack_to_u16_l(v) \
  (uint16x8_t) vec_mergel((uint8x16_t)v, vec_splat_u8(0))
#define unpack_to_s16_h(v) \
  (int16x8_t) vec_mergeh((uint8x16_t)v, vec_splat_u8(0))
#define unpack_to_s16_l(v) \
  (int16x8_t) vec_mergel((uint8x16_t)v, vec_splat_u8(0))
#ifndef xxpermdi
#define xxpermdi(a, b, c) vec_xxpermdi(b, a, (((c) >> 1) | ((c) & 1) << 1) ^ 3)
#endif
#endif

static INLINE uint8x16_t read4x2(const uint8_t *a, int stride) {
  const uint32x4_t a0 = (uint32x4_t)vec_vsx_ld(0, a);
  const uint32x4_t a1 = (uint32x4_t)vec_vsx_ld(0, a + stride);

  return (uint8x16_t)vec_mergeh(a0, a1);
}

#ifndef __POWER9_VECTOR__
#define vec_absd(a, b) vec_sub(vec_max(a, b), vec_min(a, b))
#endif

static const uint8x16_t vec_zeros_u8 = { 0, 0, 0, 0, 0, 0, 0, 0,
                                         0, 0, 0, 0, 0, 0, 0, 0 };
static const int16x8_t vec_zeros_s16 = { 0, 0, 0, 0, 0, 0, 0, 0 };
static const int16x8_t vec_ones_s16 = { 1, 1, 1, 1, 1, 1, 1, 1 };
static const int16x8_t vec_twos_s16 = { 2, 2, 2, 2, 2, 2, 2, 2 };
static const uint16x8_t vec_ones_u16 = { 1, 1, 1, 1, 1, 1, 1, 1 };
static const uint32x4_t vec_ones_u32 = { 1, 1, 1, 1 };
static const int32x4_t vec_zeros_s32 = { 0, 0, 0, 0 };
static const uint32x4_t vec_zeros_u32 = { 0, 0, 0, 0 };
static const uint16x8_t vec_shift_sign_s16 = { 15, 15, 15, 15, 15, 15, 15, 15 };
static const uint32x4_t vec_shift_sign_s32 = { 31, 31, 31, 31 };
static const uint8x16_t vec_perm64 = { 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                       0x0E, 0x0F, 0x00, 0x01, 0x02, 0x03,
                                       0x04, 0x05, 0x06, 0x07 };
static const uint8x16_t vec_perm32 = { 0x04, 0x05, 0x06, 0x07, 0x08, 0x09,
                                       0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                                       0x00, 0x01, 0x02, 0x03 };
static const uint8x16_t vec_perm16 = { 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                       0x08, 0x09, 0x0A, 0x0B, 0x0E, 0x0D,
                                       0x0E, 0x0F, 0x00, 0x01 };

static const uint8x16_t vec_perm_odd_even_pack = { 0x00, 0x01, 0x10, 0x11,
                                                   0x04, 0x05, 0x14, 0x15,
                                                   0x08, 0x09, 0x18, 0x19,
                                                   0x0C, 0x0D, 0x1C, 0x1D };

#endif  // VPX_VPX_DSP_PPC_TYPES_VSX_H_
