/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/ppc/types_vsx.h"

void vpx_v_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x16_t d = vec_vsx_ld(0, above);
  int i;
  (void)left;

  for (i = 0; i < 16; i++, dst += stride) {
    vec_vsx_st(d, 0, dst);
  }
}

void vpx_v_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x16_t d0 = vec_vsx_ld(0, above);
  const uint8x16_t d1 = vec_vsx_ld(16, above);
  int i;
  (void)left;

  for (i = 0; i < 32; i++, dst += stride) {
    vec_vsx_st(d0, 0, dst);
    vec_vsx_st(d1, 16, dst);
  }
}

// TODO(crbug.com/webm/1522): Fix test failures.
#if 0
static const uint32x4_t mask4 = { 0, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF };

void vpx_h_predictor_4x4_vsx(uint8_t *dst, ptrdiff_t stride,
                             const uint8_t *above, const uint8_t *left) {
  const uint8x16_t d = vec_vsx_ld(0, left);
  const uint8x16_t v0 = vec_splat(d, 0);
  const uint8x16_t v1 = vec_splat(d, 1);
  const uint8x16_t v2 = vec_splat(d, 2);
  const uint8x16_t v3 = vec_splat(d, 3);

  (void)above;

  vec_vsx_st(vec_sel(v0, vec_vsx_ld(0, dst), (uint8x16_t)mask4), 0, dst);
  dst += stride;
  vec_vsx_st(vec_sel(v1, vec_vsx_ld(0, dst), (uint8x16_t)mask4), 0, dst);
  dst += stride;
  vec_vsx_st(vec_sel(v2, vec_vsx_ld(0, dst), (uint8x16_t)mask4), 0, dst);
  dst += stride;
  vec_vsx_st(vec_sel(v3, vec_vsx_ld(0, dst), (uint8x16_t)mask4), 0, dst);
}

void vpx_h_predictor_8x8_vsx(uint8_t *dst, ptrdiff_t stride,
                             const uint8_t *above, const uint8_t *left) {
  const uint8x16_t d = vec_vsx_ld(0, left);
  const uint8x16_t v0 = vec_splat(d, 0);
  const uint8x16_t v1 = vec_splat(d, 1);
  const uint8x16_t v2 = vec_splat(d, 2);
  const uint8x16_t v3 = vec_splat(d, 3);

  const uint8x16_t v4 = vec_splat(d, 4);
  const uint8x16_t v5 = vec_splat(d, 5);
  const uint8x16_t v6 = vec_splat(d, 6);
  const uint8x16_t v7 = vec_splat(d, 7);

  (void)above;

  vec_vsx_st(xxpermdi(v0, vec_vsx_ld(0, dst), 1), 0, dst);
  dst += stride;
  vec_vsx_st(xxpermdi(v1, vec_vsx_ld(0, dst), 1), 0, dst);
  dst += stride;
  vec_vsx_st(xxpermdi(v2, vec_vsx_ld(0, dst), 1), 0, dst);
  dst += stride;
  vec_vsx_st(xxpermdi(v3, vec_vsx_ld(0, dst), 1), 0, dst);
  dst += stride;
  vec_vsx_st(xxpermdi(v4, vec_vsx_ld(0, dst), 1), 0, dst);
  dst += stride;
  vec_vsx_st(xxpermdi(v5, vec_vsx_ld(0, dst), 1), 0, dst);
  dst += stride;
  vec_vsx_st(xxpermdi(v6, vec_vsx_ld(0, dst), 1), 0, dst);
  dst += stride;
  vec_vsx_st(xxpermdi(v7, vec_vsx_ld(0, dst), 1), 0, dst);
}
#endif

void vpx_h_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x16_t d = vec_vsx_ld(0, left);
  const uint8x16_t v0 = vec_splat(d, 0);
  const uint8x16_t v1 = vec_splat(d, 1);
  const uint8x16_t v2 = vec_splat(d, 2);
  const uint8x16_t v3 = vec_splat(d, 3);

  const uint8x16_t v4 = vec_splat(d, 4);
  const uint8x16_t v5 = vec_splat(d, 5);
  const uint8x16_t v6 = vec_splat(d, 6);
  const uint8x16_t v7 = vec_splat(d, 7);

  const uint8x16_t v8 = vec_splat(d, 8);
  const uint8x16_t v9 = vec_splat(d, 9);
  const uint8x16_t v10 = vec_splat(d, 10);
  const uint8x16_t v11 = vec_splat(d, 11);

  const uint8x16_t v12 = vec_splat(d, 12);
  const uint8x16_t v13 = vec_splat(d, 13);
  const uint8x16_t v14 = vec_splat(d, 14);
  const uint8x16_t v15 = vec_splat(d, 15);

  (void)above;

  vec_vsx_st(v0, 0, dst);
  dst += stride;
  vec_vsx_st(v1, 0, dst);
  dst += stride;
  vec_vsx_st(v2, 0, dst);
  dst += stride;
  vec_vsx_st(v3, 0, dst);
  dst += stride;
  vec_vsx_st(v4, 0, dst);
  dst += stride;
  vec_vsx_st(v5, 0, dst);
  dst += stride;
  vec_vsx_st(v6, 0, dst);
  dst += stride;
  vec_vsx_st(v7, 0, dst);
  dst += stride;
  vec_vsx_st(v8, 0, dst);
  dst += stride;
  vec_vsx_st(v9, 0, dst);
  dst += stride;
  vec_vsx_st(v10, 0, dst);
  dst += stride;
  vec_vsx_st(v11, 0, dst);
  dst += stride;
  vec_vsx_st(v12, 0, dst);
  dst += stride;
  vec_vsx_st(v13, 0, dst);
  dst += stride;
  vec_vsx_st(v14, 0, dst);
  dst += stride;
  vec_vsx_st(v15, 0, dst);
}

#define H_PREDICTOR_32(v) \
  vec_vsx_st(v, 0, dst);  \
  vec_vsx_st(v, 16, dst); \
  dst += stride

void vpx_h_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x16_t d0 = vec_vsx_ld(0, left);
  const uint8x16_t d1 = vec_vsx_ld(16, left);

  const uint8x16_t v0_0 = vec_splat(d0, 0);
  const uint8x16_t v1_0 = vec_splat(d0, 1);
  const uint8x16_t v2_0 = vec_splat(d0, 2);
  const uint8x16_t v3_0 = vec_splat(d0, 3);
  const uint8x16_t v4_0 = vec_splat(d0, 4);
  const uint8x16_t v5_0 = vec_splat(d0, 5);
  const uint8x16_t v6_0 = vec_splat(d0, 6);
  const uint8x16_t v7_0 = vec_splat(d0, 7);
  const uint8x16_t v8_0 = vec_splat(d0, 8);
  const uint8x16_t v9_0 = vec_splat(d0, 9);
  const uint8x16_t v10_0 = vec_splat(d0, 10);
  const uint8x16_t v11_0 = vec_splat(d0, 11);
  const uint8x16_t v12_0 = vec_splat(d0, 12);
  const uint8x16_t v13_0 = vec_splat(d0, 13);
  const uint8x16_t v14_0 = vec_splat(d0, 14);
  const uint8x16_t v15_0 = vec_splat(d0, 15);

  const uint8x16_t v0_1 = vec_splat(d1, 0);
  const uint8x16_t v1_1 = vec_splat(d1, 1);
  const uint8x16_t v2_1 = vec_splat(d1, 2);
  const uint8x16_t v3_1 = vec_splat(d1, 3);
  const uint8x16_t v4_1 = vec_splat(d1, 4);
  const uint8x16_t v5_1 = vec_splat(d1, 5);
  const uint8x16_t v6_1 = vec_splat(d1, 6);
  const uint8x16_t v7_1 = vec_splat(d1, 7);
  const uint8x16_t v8_1 = vec_splat(d1, 8);
  const uint8x16_t v9_1 = vec_splat(d1, 9);
  const uint8x16_t v10_1 = vec_splat(d1, 10);
  const uint8x16_t v11_1 = vec_splat(d1, 11);
  const uint8x16_t v12_1 = vec_splat(d1, 12);
  const uint8x16_t v13_1 = vec_splat(d1, 13);
  const uint8x16_t v14_1 = vec_splat(d1, 14);
  const uint8x16_t v15_1 = vec_splat(d1, 15);

  (void)above;

  H_PREDICTOR_32(v0_0);
  H_PREDICTOR_32(v1_0);
  H_PREDICTOR_32(v2_0);
  H_PREDICTOR_32(v3_0);

  H_PREDICTOR_32(v4_0);
  H_PREDICTOR_32(v5_0);
  H_PREDICTOR_32(v6_0);
  H_PREDICTOR_32(v7_0);

  H_PREDICTOR_32(v8_0);
  H_PREDICTOR_32(v9_0);
  H_PREDICTOR_32(v10_0);
  H_PREDICTOR_32(v11_0);

  H_PREDICTOR_32(v12_0);
  H_PREDICTOR_32(v13_0);
  H_PREDICTOR_32(v14_0);
  H_PREDICTOR_32(v15_0);

  H_PREDICTOR_32(v0_1);
  H_PREDICTOR_32(v1_1);
  H_PREDICTOR_32(v2_1);
  H_PREDICTOR_32(v3_1);

  H_PREDICTOR_32(v4_1);
  H_PREDICTOR_32(v5_1);
  H_PREDICTOR_32(v6_1);
  H_PREDICTOR_32(v7_1);

  H_PREDICTOR_32(v8_1);
  H_PREDICTOR_32(v9_1);
  H_PREDICTOR_32(v10_1);
  H_PREDICTOR_32(v11_1);

  H_PREDICTOR_32(v12_1);
  H_PREDICTOR_32(v13_1);
  H_PREDICTOR_32(v14_1);
  H_PREDICTOR_32(v15_1);
}

// TODO(crbug.com/webm/1522): Fix test failures.
#if 0
void vpx_tm_predictor_4x4_vsx(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  const int16x8_t tl = unpack_to_s16_h(vec_splat(vec_vsx_ld(-1, above), 0));
  const int16x8_t l = unpack_to_s16_h(vec_vsx_ld(0, left));
  const int16x8_t a = unpack_to_s16_h(vec_vsx_ld(0, above));
  int16x8_t tmp, val;
  uint8x16_t d;

  d = vec_vsx_ld(0, dst);
  tmp = unpack_to_s16_l(d);
  val = vec_sub(vec_add(vec_splat(l, 0), a), tl);
  vec_vsx_st(vec_sel(vec_packsu(val, tmp), d, (uint8x16_t)mask4), 0, dst);
  dst += stride;

  d = vec_vsx_ld(0, dst);
  tmp = unpack_to_s16_l(d);
  val = vec_sub(vec_add(vec_splat(l, 1), a), tl);
  vec_vsx_st(vec_sel(vec_packsu(val, tmp), d, (uint8x16_t)mask4), 0, dst);
  dst += stride;

  d = vec_vsx_ld(0, dst);
  tmp = unpack_to_s16_l(d);
  val = vec_sub(vec_add(vec_splat(l, 2), a), tl);
  vec_vsx_st(vec_sel(vec_packsu(val, tmp), d, (uint8x16_t)mask4), 0, dst);
  dst += stride;

  d = vec_vsx_ld(0, dst);
  tmp = unpack_to_s16_l(d);
  val = vec_sub(vec_add(vec_splat(l, 3), a), tl);
  vec_vsx_st(vec_sel(vec_packsu(val, tmp), d, (uint8x16_t)mask4), 0, dst);
}

void vpx_tm_predictor_8x8_vsx(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  const int16x8_t tl = unpack_to_s16_h(vec_splat(vec_vsx_ld(-1, above), 0));
  const int16x8_t l = unpack_to_s16_h(vec_vsx_ld(0, left));
  const int16x8_t a = unpack_to_s16_h(vec_vsx_ld(0, above));
  int16x8_t tmp, val;

  tmp = unpack_to_s16_l(vec_vsx_ld(0, dst));
  val = vec_sub(vec_add(vec_splat(l, 0), a), tl);
  vec_vsx_st(vec_packsu(val, tmp), 0, dst);
  dst += stride;

  tmp = unpack_to_s16_l(vec_vsx_ld(0, dst));
  val = vec_sub(vec_add(vec_splat(l, 1), a), tl);
  vec_vsx_st(vec_packsu(val, tmp), 0, dst);
  dst += stride;

  tmp = unpack_to_s16_l(vec_vsx_ld(0, dst));
  val = vec_sub(vec_add(vec_splat(l, 2), a), tl);
  vec_vsx_st(vec_packsu(val, tmp), 0, dst);
  dst += stride;

  tmp = unpack_to_s16_l(vec_vsx_ld(0, dst));
  val = vec_sub(vec_add(vec_splat(l, 3), a), tl);
  vec_vsx_st(vec_packsu(val, tmp), 0, dst);
  dst += stride;

  tmp = unpack_to_s16_l(vec_vsx_ld(0, dst));
  val = vec_sub(vec_add(vec_splat(l, 4), a), tl);
  vec_vsx_st(vec_packsu(val, tmp), 0, dst);
  dst += stride;

  tmp = unpack_to_s16_l(vec_vsx_ld(0, dst));
  val = vec_sub(vec_add(vec_splat(l, 5), a), tl);
  vec_vsx_st(vec_packsu(val, tmp), 0, dst);
  dst += stride;

  tmp = unpack_to_s16_l(vec_vsx_ld(0, dst));
  val = vec_sub(vec_add(vec_splat(l, 6), a), tl);
  vec_vsx_st(vec_packsu(val, tmp), 0, dst);
  dst += stride;

  tmp = unpack_to_s16_l(vec_vsx_ld(0, dst));
  val = vec_sub(vec_add(vec_splat(l, 7), a), tl);
  vec_vsx_st(vec_packsu(val, tmp), 0, dst);
}
#endif

static void tm_predictor_16x8(uint8_t *dst, const ptrdiff_t stride, int16x8_t l,
                              int16x8_t ah, int16x8_t al, int16x8_t tl) {
  int16x8_t vh, vl, ls;

  ls = vec_splat(l, 0);
  vh = vec_sub(vec_add(ls, ah), tl);
  vl = vec_sub(vec_add(ls, al), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
  dst += stride;

  ls = vec_splat(l, 1);
  vh = vec_sub(vec_add(ls, ah), tl);
  vl = vec_sub(vec_add(ls, al), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
  dst += stride;

  ls = vec_splat(l, 2);
  vh = vec_sub(vec_add(ls, ah), tl);
  vl = vec_sub(vec_add(ls, al), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
  dst += stride;

  ls = vec_splat(l, 3);
  vh = vec_sub(vec_add(ls, ah), tl);
  vl = vec_sub(vec_add(ls, al), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
  dst += stride;

  ls = vec_splat(l, 4);
  vh = vec_sub(vec_add(ls, ah), tl);
  vl = vec_sub(vec_add(ls, al), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
  dst += stride;

  ls = vec_splat(l, 5);
  vh = vec_sub(vec_add(ls, ah), tl);
  vl = vec_sub(vec_add(ls, al), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
  dst += stride;

  ls = vec_splat(l, 6);
  vh = vec_sub(vec_add(ls, ah), tl);
  vl = vec_sub(vec_add(ls, al), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
  dst += stride;

  ls = vec_splat(l, 7);
  vh = vec_sub(vec_add(ls, ah), tl);
  vl = vec_sub(vec_add(ls, al), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
}

void vpx_tm_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  const int16x8_t tl = unpack_to_s16_h(vec_splat(vec_vsx_ld(-1, above), 0));
  const uint8x16_t l = vec_vsx_ld(0, left);
  const int16x8_t lh = unpack_to_s16_h(l);
  const int16x8_t ll = unpack_to_s16_l(l);
  const uint8x16_t a = vec_vsx_ld(0, above);
  const int16x8_t ah = unpack_to_s16_h(a);
  const int16x8_t al = unpack_to_s16_l(a);

  tm_predictor_16x8(dst, stride, lh, ah, al, tl);

  dst += stride * 8;

  tm_predictor_16x8(dst, stride, ll, ah, al, tl);
}

static INLINE void tm_predictor_32x1(uint8_t *dst, const int16x8_t ls,
                                     const int16x8_t a0h, const int16x8_t a0l,
                                     const int16x8_t a1h, const int16x8_t a1l,
                                     const int16x8_t tl) {
  int16x8_t vh, vl;

  vh = vec_sub(vec_add(ls, a0h), tl);
  vl = vec_sub(vec_add(ls, a0l), tl);
  vec_vsx_st(vec_packsu(vh, vl), 0, dst);
  vh = vec_sub(vec_add(ls, a1h), tl);
  vl = vec_sub(vec_add(ls, a1l), tl);
  vec_vsx_st(vec_packsu(vh, vl), 16, dst);
}

static void tm_predictor_32x8(uint8_t *dst, const ptrdiff_t stride,
                              const int16x8_t l, const uint8x16_t a0,
                              const uint8x16_t a1, const int16x8_t tl) {
  const int16x8_t a0h = unpack_to_s16_h(a0);
  const int16x8_t a0l = unpack_to_s16_l(a0);
  const int16x8_t a1h = unpack_to_s16_h(a1);
  const int16x8_t a1l = unpack_to_s16_l(a1);

  tm_predictor_32x1(dst, vec_splat(l, 0), a0h, a0l, a1h, a1l, tl);
  dst += stride;

  tm_predictor_32x1(dst, vec_splat(l, 1), a0h, a0l, a1h, a1l, tl);
  dst += stride;

  tm_predictor_32x1(dst, vec_splat(l, 2), a0h, a0l, a1h, a1l, tl);
  dst += stride;

  tm_predictor_32x1(dst, vec_splat(l, 3), a0h, a0l, a1h, a1l, tl);
  dst += stride;

  tm_predictor_32x1(dst, vec_splat(l, 4), a0h, a0l, a1h, a1l, tl);
  dst += stride;

  tm_predictor_32x1(dst, vec_splat(l, 5), a0h, a0l, a1h, a1l, tl);
  dst += stride;

  tm_predictor_32x1(dst, vec_splat(l, 6), a0h, a0l, a1h, a1l, tl);
  dst += stride;

  tm_predictor_32x1(dst, vec_splat(l, 7), a0h, a0l, a1h, a1l, tl);
}

void vpx_tm_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  const int16x8_t tl = unpack_to_s16_h(vec_splat(vec_vsx_ld(-1, above), 0));
  const uint8x16_t l0 = vec_vsx_ld(0, left);
  const uint8x16_t l1 = vec_vsx_ld(16, left);
  const uint8x16_t a0 = vec_vsx_ld(0, above);
  const uint8x16_t a1 = vec_vsx_ld(16, above);

  tm_predictor_32x8(dst, stride, unpack_to_s16_h(l0), a0, a1, tl);
  dst += stride * 8;

  tm_predictor_32x8(dst, stride, unpack_to_s16_l(l0), a0, a1, tl);
  dst += stride * 8;

  tm_predictor_32x8(dst, stride, unpack_to_s16_h(l1), a0, a1, tl);
  dst += stride * 8;

  tm_predictor_32x8(dst, stride, unpack_to_s16_l(l1), a0, a1, tl);
}

static INLINE void dc_fill_predictor_8x8(uint8_t *dst, const ptrdiff_t stride,
                                         const uint8x16_t val) {
  int i;

  for (i = 0; i < 8; i++, dst += stride) {
    const uint8x16_t d = vec_vsx_ld(0, dst);
    vec_vsx_st(xxpermdi(val, d, 1), 0, dst);
  }
}

static INLINE void dc_fill_predictor_16x16(uint8_t *dst, const ptrdiff_t stride,
                                           const uint8x16_t val) {
  int i;

  for (i = 0; i < 16; i++, dst += stride) {
    vec_vsx_st(val, 0, dst);
  }
}

void vpx_dc_128_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above, const uint8_t *left) {
  const uint8x16_t v128 = vec_sl(vec_splat_u8(1), vec_splat_u8(7));
  (void)above;
  (void)left;

  dc_fill_predictor_16x16(dst, stride, v128);
}

static INLINE void dc_fill_predictor_32x32(uint8_t *dst, const ptrdiff_t stride,
                                           const uint8x16_t val) {
  int i;

  for (i = 0; i < 32; i++, dst += stride) {
    vec_vsx_st(val, 0, dst);
    vec_vsx_st(val, 16, dst);
  }
}

void vpx_dc_128_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above, const uint8_t *left) {
  const uint8x16_t v128 = vec_sl(vec_splat_u8(1), vec_splat_u8(7));
  (void)above;
  (void)left;

  dc_fill_predictor_32x32(dst, stride, v128);
}

static uint8x16_t avg16(const uint8_t *values) {
  const int32x4_t sum4s =
      (int32x4_t)vec_sum4s(vec_vsx_ld(0, values), vec_splat_u32(0));
  const uint32x4_t sum = (uint32x4_t)vec_sums(sum4s, vec_splat_s32(8));
  const uint32x4_t avg = (uint32x4_t)vec_sr(sum, vec_splat_u32(4));

  return vec_splat(vec_pack(vec_pack(avg, vec_splat_u32(0)), vec_splat_u16(0)),
                   3);
}

void vpx_dc_left_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  (void)above;

  dc_fill_predictor_16x16(dst, stride, avg16(left));
}

void vpx_dc_top_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above, const uint8_t *left) {
  (void)left;

  dc_fill_predictor_16x16(dst, stride, avg16(above));
}

static uint8x16_t avg32(const uint8_t *values) {
  const uint8x16_t v0 = vec_vsx_ld(0, values);
  const uint8x16_t v1 = vec_vsx_ld(16, values);
  const int32x4_t v16 = vec_sl(vec_splat_s32(1), vec_splat_u32(4));
  const int32x4_t sum4s =
      (int32x4_t)vec_sum4s(v0, vec_sum4s(v1, vec_splat_u32(0)));
  const uint32x4_t sum = (uint32x4_t)vec_sums(sum4s, v16);
  const uint32x4_t avg = (uint32x4_t)vec_sr(sum, vec_splat_u32(5));

  return vec_splat(vec_pack(vec_pack(avg, vec_splat_u32(0)), vec_splat_u16(0)),
                   3);
}

void vpx_dc_left_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                                     const uint8_t *above,
                                     const uint8_t *left) {
  (void)above;

  dc_fill_predictor_32x32(dst, stride, avg32(left));
}

void vpx_dc_top_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                                    const uint8_t *above, const uint8_t *left) {
  (void)left;

  dc_fill_predictor_32x32(dst, stride, avg32(above));
}

// TODO(crbug.com/webm/1522): Fix test failures.
#if 0
static uint8x16_t dc_avg8(const uint8_t *above, const uint8_t *left) {
  const uint8x16_t a0 = vec_vsx_ld(0, above);
  const uint8x16_t l0 = vec_vsx_ld(0, left);
  const int32x4_t sum4s =
      (int32x4_t)vec_sum4s(l0, vec_sum4s(a0, vec_splat_u32(0)));
  const int32x4_t sum4s8 = xxpermdi(sum4s, vec_splat_s32(0), 1);
  const uint32x4_t sum = (uint32x4_t)vec_sums(sum4s8, vec_splat_s32(8));
  const uint32x4_t avg = (uint32x4_t)vec_sr(sum, vec_splat_u32(4));

  return vec_splat(vec_pack(vec_pack(avg, vec_splat_u32(0)), vec_splat_u16(0)),
                   3);
}
#endif

static uint8x16_t dc_avg16(const uint8_t *above, const uint8_t *left) {
  const uint8x16_t a0 = vec_vsx_ld(0, above);
  const uint8x16_t l0 = vec_vsx_ld(0, left);
  const int32x4_t v16 = vec_sl(vec_splat_s32(1), vec_splat_u32(4));
  const int32x4_t sum4s =
      (int32x4_t)vec_sum4s(l0, vec_sum4s(a0, vec_splat_u32(0)));
  const uint32x4_t sum = (uint32x4_t)vec_sums(sum4s, v16);
  const uint32x4_t avg = (uint32x4_t)vec_sr(sum, vec_splat_u32(5));

  return vec_splat(vec_pack(vec_pack(avg, vec_splat_u32(0)), vec_splat_u16(0)),
                   3);
}

// TODO(crbug.com/webm/1522): Fix test failures.
#if 0
void vpx_dc_predictor_8x8_vsx(uint8_t *dst, ptrdiff_t stride,
                              const uint8_t *above, const uint8_t *left) {
  dc_fill_predictor_8x8(dst, stride, dc_avg8(above, left));
}
#endif

void vpx_dc_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  dc_fill_predictor_16x16(dst, stride, dc_avg16(above, left));
}

static uint8x16_t dc_avg32(const uint8_t *above, const uint8_t *left) {
  const uint8x16_t a0 = vec_vsx_ld(0, above);
  const uint8x16_t a1 = vec_vsx_ld(16, above);
  const uint8x16_t l0 = vec_vsx_ld(0, left);
  const uint8x16_t l1 = vec_vsx_ld(16, left);
  const int32x4_t v32 = vec_sl(vec_splat_s32(1), vec_splat_u32(5));
  const uint32x4_t a_sum = vec_sum4s(a0, vec_sum4s(a1, vec_splat_u32(0)));
  const int32x4_t sum4s = (int32x4_t)vec_sum4s(l0, vec_sum4s(l1, a_sum));
  const uint32x4_t sum = (uint32x4_t)vec_sums(sum4s, v32);
  const uint32x4_t avg = (uint32x4_t)vec_sr(sum, vec_splat_u32(6));

  return vec_splat(vec_pack(vec_pack(avg, vec_splat_u32(0)), vec_splat_u16(0)),
                   3);
}

void vpx_dc_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                                const uint8_t *above, const uint8_t *left) {
  dc_fill_predictor_32x32(dst, stride, dc_avg32(above, left));
}

static uint8x16_t avg3(const uint8x16_t a, const uint8x16_t b,
                       const uint8x16_t c) {
  const uint8x16_t ac =
      vec_adds(vec_and(a, c), vec_sr(vec_xor(a, c), vec_splat_u8(1)));

  return vec_avg(ac, b);
}

// Workaround vec_sld/vec_xxsldi/vec_lsdoi being missing or broken.
static const uint8x16_t sl1 = { 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8,
                                0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x10 };

// TODO(crbug.com/webm/1522): Fix test failures.
#if 0
void vpx_d45_predictor_8x8_vsx(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x16_t af = vec_vsx_ld(0, above);
  const uint8x16_t above_right = vec_splat(af, 7);
  const uint8x16_t a = xxpermdi(af, above_right, 1);
  const uint8x16_t b = vec_perm(a, above_right, sl1);
  const uint8x16_t c = vec_perm(b, above_right, sl1);
  uint8x16_t row = avg3(a, b, c);
  int i;
  (void)left;

  for (i = 0; i < 8; i++) {
    const uint8x16_t d = vec_vsx_ld(0, dst);
    vec_vsx_st(xxpermdi(row, d, 1), 0, dst);
    dst += stride;
    row = vec_perm(row, above_right, sl1);
  }
}
#endif

void vpx_d45_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x16_t a = vec_vsx_ld(0, above);
  const uint8x16_t above_right = vec_splat(a, 15);
  const uint8x16_t b = vec_perm(a, above_right, sl1);
  const uint8x16_t c = vec_perm(b, above_right, sl1);
  uint8x16_t row = avg3(a, b, c);
  int i;
  (void)left;

  for (i = 0; i < 16; i++) {
    vec_vsx_st(row, 0, dst);
    dst += stride;
    row = vec_perm(row, above_right, sl1);
  }
}

void vpx_d45_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x16_t a0 = vec_vsx_ld(0, above);
  const uint8x16_t a1 = vec_vsx_ld(16, above);
  const uint8x16_t above_right = vec_splat(a1, 15);
  const uint8x16_t b0 = vec_perm(a0, a1, sl1);
  const uint8x16_t b1 = vec_perm(a1, above_right, sl1);
  const uint8x16_t c0 = vec_perm(b0, b1, sl1);
  const uint8x16_t c1 = vec_perm(b1, above_right, sl1);
  uint8x16_t row0 = avg3(a0, b0, c0);
  uint8x16_t row1 = avg3(a1, b1, c1);
  int i;
  (void)left;

  for (i = 0; i < 32; i++) {
    vec_vsx_st(row0, 0, dst);
    vec_vsx_st(row1, 16, dst);
    dst += stride;
    row0 = vec_perm(row0, row1, sl1);
    row1 = vec_perm(row1, above_right, sl1);
  }
}

// TODO(crbug.com/webm/1522): Fix test failures.
#if 0
void vpx_d63_predictor_8x8_vsx(uint8_t *dst, ptrdiff_t stride,
                               const uint8_t *above, const uint8_t *left) {
  const uint8x16_t af = vec_vsx_ld(0, above);
  const uint8x16_t above_right = vec_splat(af, 9);
  const uint8x16_t a = xxpermdi(af, above_right, 1);
  const uint8x16_t b = vec_perm(a, above_right, sl1);
  const uint8x16_t c = vec_perm(b, above_right, sl1);
  uint8x16_t row0 = vec_avg(a, b);
  uint8x16_t row1 = avg3(a, b, c);
  int i;
  (void)left;

  for (i = 0; i < 4; i++) {
    const uint8x16_t d0 = vec_vsx_ld(0, dst);
    const uint8x16_t d1 = vec_vsx_ld(0, dst + stride);
    vec_vsx_st(xxpermdi(row0, d0, 1), 0, dst);
    vec_vsx_st(xxpermdi(row1, d1, 1), 0, dst + stride);
    dst += stride * 2;
    row0 = vec_perm(row0, above_right, sl1);
    row1 = vec_perm(row1, above_right, sl1);
  }
}
#endif

void vpx_d63_predictor_16x16_vsx(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x16_t a0 = vec_vsx_ld(0, above);
  const uint8x16_t a1 = vec_vsx_ld(16, above);
  const uint8x16_t above_right = vec_splat(a1, 0);
  const uint8x16_t b = vec_perm(a0, above_right, sl1);
  const uint8x16_t c = vec_perm(b, above_right, sl1);
  uint8x16_t row0 = vec_avg(a0, b);
  uint8x16_t row1 = avg3(a0, b, c);
  int i;
  (void)left;

  for (i = 0; i < 8; i++) {
    vec_vsx_st(row0, 0, dst);
    vec_vsx_st(row1, 0, dst + stride);
    dst += stride * 2;
    row0 = vec_perm(row0, above_right, sl1);
    row1 = vec_perm(row1, above_right, sl1);
  }
}

void vpx_d63_predictor_32x32_vsx(uint8_t *dst, ptrdiff_t stride,
                                 const uint8_t *above, const uint8_t *left) {
  const uint8x16_t a0 = vec_vsx_ld(0, above);
  const uint8x16_t a1 = vec_vsx_ld(16, above);
  const uint8x16_t a2 = vec_vsx_ld(32, above);
  const uint8x16_t above_right = vec_splat(a2, 0);
  const uint8x16_t b0 = vec_perm(a0, a1, sl1);
  const uint8x16_t b1 = vec_perm(a1, above_right, sl1);
  const uint8x16_t c0 = vec_perm(b0, b1, sl1);
  const uint8x16_t c1 = vec_perm(b1, above_right, sl1);
  uint8x16_t row0_0 = vec_avg(a0, b0);
  uint8x16_t row0_1 = vec_avg(a1, b1);
  uint8x16_t row1_0 = avg3(a0, b0, c0);
  uint8x16_t row1_1 = avg3(a1, b1, c1);
  int i;
  (void)left;

  for (i = 0; i < 16; i++) {
    vec_vsx_st(row0_0, 0, dst);
    vec_vsx_st(row0_1, 16, dst);
    vec_vsx_st(row1_0, 0, dst + stride);
    vec_vsx_st(row1_1, 16, dst + stride);
    dst += stride * 2;
    row0_0 = vec_perm(row0_0, row0_1, sl1);
    row0_1 = vec_perm(row0_1, above_right, sl1);
    row1_0 = vec_perm(row1_0, row1_1, sl1);
    row1_1 = vec_perm(row1_1, above_right, sl1);
  }
}
