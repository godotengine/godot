/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/bitdepth_conversion_sse2.h"
#include "vpx_ports/mem.h"

static INLINE void sign_extend_16bit_to_32bit_sse2(__m128i in, __m128i zero,
                                                   __m128i *out_lo,
                                                   __m128i *out_hi) {
  const __m128i sign_bits = _mm_cmplt_epi16(in, zero);
  *out_lo = _mm_unpacklo_epi16(in, sign_bits);
  *out_hi = _mm_unpackhi_epi16(in, sign_bits);
}

void vpx_minmax_8x8_sse2(const uint8_t *s, int p, const uint8_t *d, int dp,
                         int *min, int *max) {
  __m128i u0, s0, d0, diff, maxabsdiff, minabsdiff, negdiff, absdiff0, absdiff;
  u0 = _mm_setzero_si128();
  // Row 0
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s)), u0);
  d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(d)), u0);
  diff = _mm_subs_epi16(s0, d0);
  negdiff = _mm_subs_epi16(u0, diff);
  absdiff0 = _mm_max_epi16(diff, negdiff);
  // Row 1
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + p)), u0);
  d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(d + dp)), u0);
  diff = _mm_subs_epi16(s0, d0);
  negdiff = _mm_subs_epi16(u0, diff);
  absdiff = _mm_max_epi16(diff, negdiff);
  maxabsdiff = _mm_max_epi16(absdiff0, absdiff);
  minabsdiff = _mm_min_epi16(absdiff0, absdiff);
  // Row 2
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 2 * p)), u0);
  d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(d + 2 * dp)), u0);
  diff = _mm_subs_epi16(s0, d0);
  negdiff = _mm_subs_epi16(u0, diff);
  absdiff = _mm_max_epi16(diff, negdiff);
  maxabsdiff = _mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = _mm_min_epi16(minabsdiff, absdiff);
  // Row 3
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 3 * p)), u0);
  d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(d + 3 * dp)), u0);
  diff = _mm_subs_epi16(s0, d0);
  negdiff = _mm_subs_epi16(u0, diff);
  absdiff = _mm_max_epi16(diff, negdiff);
  maxabsdiff = _mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = _mm_min_epi16(minabsdiff, absdiff);
  // Row 4
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 4 * p)), u0);
  d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(d + 4 * dp)), u0);
  diff = _mm_subs_epi16(s0, d0);
  negdiff = _mm_subs_epi16(u0, diff);
  absdiff = _mm_max_epi16(diff, negdiff);
  maxabsdiff = _mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = _mm_min_epi16(minabsdiff, absdiff);
  // Row 5
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 5 * p)), u0);
  d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(d + 5 * dp)), u0);
  diff = _mm_subs_epi16(s0, d0);
  negdiff = _mm_subs_epi16(u0, diff);
  absdiff = _mm_max_epi16(diff, negdiff);
  maxabsdiff = _mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = _mm_min_epi16(minabsdiff, absdiff);
  // Row 6
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 6 * p)), u0);
  d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(d + 6 * dp)), u0);
  diff = _mm_subs_epi16(s0, d0);
  negdiff = _mm_subs_epi16(u0, diff);
  absdiff = _mm_max_epi16(diff, negdiff);
  maxabsdiff = _mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = _mm_min_epi16(minabsdiff, absdiff);
  // Row 7
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 7 * p)), u0);
  d0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(d + 7 * dp)), u0);
  diff = _mm_subs_epi16(s0, d0);
  negdiff = _mm_subs_epi16(u0, diff);
  absdiff = _mm_max_epi16(diff, negdiff);
  maxabsdiff = _mm_max_epi16(maxabsdiff, absdiff);
  minabsdiff = _mm_min_epi16(minabsdiff, absdiff);

  maxabsdiff = _mm_max_epi16(maxabsdiff, _mm_srli_si128(maxabsdiff, 8));
  maxabsdiff = _mm_max_epi16(maxabsdiff, _mm_srli_epi64(maxabsdiff, 32));
  maxabsdiff = _mm_max_epi16(maxabsdiff, _mm_srli_epi64(maxabsdiff, 16));
  *max = _mm_extract_epi16(maxabsdiff, 0);

  minabsdiff = _mm_min_epi16(minabsdiff, _mm_srli_si128(minabsdiff, 8));
  minabsdiff = _mm_min_epi16(minabsdiff, _mm_srli_epi64(minabsdiff, 32));
  minabsdiff = _mm_min_epi16(minabsdiff, _mm_srli_epi64(minabsdiff, 16));
  *min = _mm_extract_epi16(minabsdiff, 0);
}

unsigned int vpx_avg_8x8_sse2(const uint8_t *s, int p) {
  __m128i s0, s1, u0;
  unsigned int avg = 0;
  u0 = _mm_setzero_si128();
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s)), u0);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + p)), u0);
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 2 * p)), u0);
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 3 * p)), u0);
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 4 * p)), u0);
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 5 * p)), u0);
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 6 * p)), u0);
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 7 * p)), u0);
  s0 = _mm_adds_epu16(s0, s1);

  s0 = _mm_adds_epu16(s0, _mm_srli_si128(s0, 8));
  s0 = _mm_adds_epu16(s0, _mm_srli_epi64(s0, 32));
  s0 = _mm_adds_epu16(s0, _mm_srli_epi64(s0, 16));
  avg = _mm_extract_epi16(s0, 0);
  return (avg + 32) >> 6;
}

unsigned int vpx_avg_4x4_sse2(const uint8_t *s, int p) {
  __m128i s0, s1, u0;
  unsigned int avg = 0;
  u0 = _mm_setzero_si128();
  s0 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s)), u0);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + p)), u0);
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 2 * p)), u0);
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(s + 3 * p)), u0);
  s0 = _mm_adds_epu16(s0, s1);

  s0 = _mm_adds_epu16(s0, _mm_srli_si128(s0, 4));
  s0 = _mm_adds_epu16(s0, _mm_srli_epi64(s0, 16));
  avg = _mm_extract_epi16(s0, 0);
  return (avg + 8) >> 4;
}

#if CONFIG_VP9_HIGHBITDEPTH
unsigned int vpx_highbd_avg_8x8_sse2(const uint8_t *s8, int p) {
  __m128i s0, s1;
  unsigned int avg;
  const uint16_t *s = CONVERT_TO_SHORTPTR(s8);
  const __m128i zero = _mm_setzero_si128();
  s0 = _mm_loadu_si128((const __m128i *)(s));
  s1 = _mm_loadu_si128((const __m128i *)(s + p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_loadu_si128((const __m128i *)(s + 2 * p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_loadu_si128((const __m128i *)(s + 3 * p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_loadu_si128((const __m128i *)(s + 4 * p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_loadu_si128((const __m128i *)(s + 5 * p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_loadu_si128((const __m128i *)(s + 6 * p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_loadu_si128((const __m128i *)(s + 7 * p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_unpackhi_epi16(s0, zero);
  s0 = _mm_unpacklo_epi16(s0, zero);
  s0 = _mm_add_epi32(s0, s1);
  s0 = _mm_add_epi32(s0, _mm_srli_si128(s0, 8));
  s0 = _mm_add_epi32(s0, _mm_srli_si128(s0, 4));
  avg = (unsigned int)_mm_cvtsi128_si32(s0);

  return (avg + 32) >> 6;
}

unsigned int vpx_highbd_avg_4x4_sse2(const uint8_t *s8, int p) {
  __m128i s0, s1;
  unsigned int avg;
  const uint16_t *s = CONVERT_TO_SHORTPTR(s8);
  s0 = _mm_loadl_epi64((const __m128i *)(s));
  s1 = _mm_loadl_epi64((const __m128i *)(s + p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_loadl_epi64((const __m128i *)(s + 2 * p));
  s0 = _mm_adds_epu16(s0, s1);
  s1 = _mm_loadl_epi64((const __m128i *)(s + 3 * p));
  s0 = _mm_adds_epu16(s0, s1);
  s0 = _mm_add_epi16(s0, _mm_srli_si128(s0, 4));
  s0 = _mm_add_epi16(s0, _mm_srli_si128(s0, 2));
  avg = _mm_extract_epi16(s0, 0);

  return (avg + 8) >> 4;
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static void hadamard_col8_sse2(__m128i *in, int iter) {
  __m128i a0 = in[0];
  __m128i a1 = in[1];
  __m128i a2 = in[2];
  __m128i a3 = in[3];
  __m128i a4 = in[4];
  __m128i a5 = in[5];
  __m128i a6 = in[6];
  __m128i a7 = in[7];

  __m128i b0 = _mm_add_epi16(a0, a1);
  __m128i b1 = _mm_sub_epi16(a0, a1);
  __m128i b2 = _mm_add_epi16(a2, a3);
  __m128i b3 = _mm_sub_epi16(a2, a3);
  __m128i b4 = _mm_add_epi16(a4, a5);
  __m128i b5 = _mm_sub_epi16(a4, a5);
  __m128i b6 = _mm_add_epi16(a6, a7);
  __m128i b7 = _mm_sub_epi16(a6, a7);

  a0 = _mm_add_epi16(b0, b2);
  a1 = _mm_add_epi16(b1, b3);
  a2 = _mm_sub_epi16(b0, b2);
  a3 = _mm_sub_epi16(b1, b3);
  a4 = _mm_add_epi16(b4, b6);
  a5 = _mm_add_epi16(b5, b7);
  a6 = _mm_sub_epi16(b4, b6);
  a7 = _mm_sub_epi16(b5, b7);

  if (iter == 0) {
    b0 = _mm_add_epi16(a0, a4);
    b7 = _mm_add_epi16(a1, a5);
    b3 = _mm_add_epi16(a2, a6);
    b4 = _mm_add_epi16(a3, a7);
    b2 = _mm_sub_epi16(a0, a4);
    b6 = _mm_sub_epi16(a1, a5);
    b1 = _mm_sub_epi16(a2, a6);
    b5 = _mm_sub_epi16(a3, a7);

    a0 = _mm_unpacklo_epi16(b0, b1);
    a1 = _mm_unpacklo_epi16(b2, b3);
    a2 = _mm_unpackhi_epi16(b0, b1);
    a3 = _mm_unpackhi_epi16(b2, b3);
    a4 = _mm_unpacklo_epi16(b4, b5);
    a5 = _mm_unpacklo_epi16(b6, b7);
    a6 = _mm_unpackhi_epi16(b4, b5);
    a7 = _mm_unpackhi_epi16(b6, b7);

    b0 = _mm_unpacklo_epi32(a0, a1);
    b1 = _mm_unpacklo_epi32(a4, a5);
    b2 = _mm_unpackhi_epi32(a0, a1);
    b3 = _mm_unpackhi_epi32(a4, a5);
    b4 = _mm_unpacklo_epi32(a2, a3);
    b5 = _mm_unpacklo_epi32(a6, a7);
    b6 = _mm_unpackhi_epi32(a2, a3);
    b7 = _mm_unpackhi_epi32(a6, a7);

    in[0] = _mm_unpacklo_epi64(b0, b1);
    in[1] = _mm_unpackhi_epi64(b0, b1);
    in[2] = _mm_unpacklo_epi64(b2, b3);
    in[3] = _mm_unpackhi_epi64(b2, b3);
    in[4] = _mm_unpacklo_epi64(b4, b5);
    in[5] = _mm_unpackhi_epi64(b4, b5);
    in[6] = _mm_unpacklo_epi64(b6, b7);
    in[7] = _mm_unpackhi_epi64(b6, b7);
  } else {
    in[0] = _mm_add_epi16(a0, a4);
    in[7] = _mm_add_epi16(a1, a5);
    in[3] = _mm_add_epi16(a2, a6);
    in[4] = _mm_add_epi16(a3, a7);
    in[2] = _mm_sub_epi16(a0, a4);
    in[6] = _mm_sub_epi16(a1, a5);
    in[1] = _mm_sub_epi16(a2, a6);
    in[5] = _mm_sub_epi16(a3, a7);
  }
}

static INLINE void hadamard_8x8_sse2(const int16_t *src_diff,
                                     ptrdiff_t src_stride, tran_low_t *coeff,
                                     int is_final) {
  __m128i src[8];
  src[0] = _mm_load_si128((const __m128i *)src_diff);
  src[1] = _mm_load_si128((const __m128i *)(src_diff += src_stride));
  src[2] = _mm_load_si128((const __m128i *)(src_diff += src_stride));
  src[3] = _mm_load_si128((const __m128i *)(src_diff += src_stride));
  src[4] = _mm_load_si128((const __m128i *)(src_diff += src_stride));
  src[5] = _mm_load_si128((const __m128i *)(src_diff += src_stride));
  src[6] = _mm_load_si128((const __m128i *)(src_diff += src_stride));
  src[7] = _mm_load_si128((const __m128i *)(src_diff + src_stride));

  hadamard_col8_sse2(src, 0);
  hadamard_col8_sse2(src, 1);

  if (is_final) {
    store_tran_low(src[0], coeff);
    coeff += 8;
    store_tran_low(src[1], coeff);
    coeff += 8;
    store_tran_low(src[2], coeff);
    coeff += 8;
    store_tran_low(src[3], coeff);
    coeff += 8;
    store_tran_low(src[4], coeff);
    coeff += 8;
    store_tran_low(src[5], coeff);
    coeff += 8;
    store_tran_low(src[6], coeff);
    coeff += 8;
    store_tran_low(src[7], coeff);
  } else {
    int16_t *coeff16 = (int16_t *)coeff;
    _mm_store_si128((__m128i *)coeff16, src[0]);
    coeff16 += 8;
    _mm_store_si128((__m128i *)coeff16, src[1]);
    coeff16 += 8;
    _mm_store_si128((__m128i *)coeff16, src[2]);
    coeff16 += 8;
    _mm_store_si128((__m128i *)coeff16, src[3]);
    coeff16 += 8;
    _mm_store_si128((__m128i *)coeff16, src[4]);
    coeff16 += 8;
    _mm_store_si128((__m128i *)coeff16, src[5]);
    coeff16 += 8;
    _mm_store_si128((__m128i *)coeff16, src[6]);
    coeff16 += 8;
    _mm_store_si128((__m128i *)coeff16, src[7]);
  }
}

void vpx_hadamard_8x8_sse2(const int16_t *src_diff, ptrdiff_t src_stride,
                           tran_low_t *coeff) {
  hadamard_8x8_sse2(src_diff, src_stride, coeff, 1);
}

static INLINE void hadamard_16x16_sse2(const int16_t *src_diff,
                                       ptrdiff_t src_stride, tran_low_t *coeff,
                                       int is_final) {
#if CONFIG_VP9_HIGHBITDEPTH
  // For high bitdepths, it is unnecessary to store_tran_low
  // (mult/unpack/store), then load_tran_low (load/pack) the same memory in the
  // next stage.  Output to an intermediate buffer first, then store_tran_low()
  // in the final stage.
  DECLARE_ALIGNED(32, int16_t, temp_coeff[16 * 16]);
  int16_t *t_coeff = temp_coeff;
#else
  int16_t *t_coeff = coeff;
#endif
  int16_t *coeff16 = (int16_t *)coeff;
  int idx;
  for (idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 8 * src_stride + (idx & 0x01) * 8;
    hadamard_8x8_sse2(src_ptr, src_stride, (tran_low_t *)(t_coeff + idx * 64),
                      0);
  }

  for (idx = 0; idx < 64; idx += 8) {
    __m128i coeff0 = _mm_load_si128((const __m128i *)t_coeff);
    __m128i coeff1 = _mm_load_si128((const __m128i *)(t_coeff + 64));
    __m128i coeff2 = _mm_load_si128((const __m128i *)(t_coeff + 128));
    __m128i coeff3 = _mm_load_si128((const __m128i *)(t_coeff + 192));

    __m128i b0 = _mm_add_epi16(coeff0, coeff1);
    __m128i b1 = _mm_sub_epi16(coeff0, coeff1);
    __m128i b2 = _mm_add_epi16(coeff2, coeff3);
    __m128i b3 = _mm_sub_epi16(coeff2, coeff3);

    b0 = _mm_srai_epi16(b0, 1);
    b1 = _mm_srai_epi16(b1, 1);
    b2 = _mm_srai_epi16(b2, 1);
    b3 = _mm_srai_epi16(b3, 1);

    coeff0 = _mm_add_epi16(b0, b2);
    coeff1 = _mm_add_epi16(b1, b3);
    coeff2 = _mm_sub_epi16(b0, b2);
    coeff3 = _mm_sub_epi16(b1, b3);

    if (is_final) {
      store_tran_low(coeff0, coeff);
      store_tran_low(coeff1, coeff + 64);
      store_tran_low(coeff2, coeff + 128);
      store_tran_low(coeff3, coeff + 192);
      coeff += 8;
    } else {
      _mm_store_si128((__m128i *)coeff16, coeff0);
      _mm_store_si128((__m128i *)(coeff16 + 64), coeff1);
      _mm_store_si128((__m128i *)(coeff16 + 128), coeff2);
      _mm_store_si128((__m128i *)(coeff16 + 192), coeff3);
      coeff16 += 8;
    }

    t_coeff += 8;
  }
}

void vpx_hadamard_16x16_sse2(const int16_t *src_diff, ptrdiff_t src_stride,
                             tran_low_t *coeff) {
  hadamard_16x16_sse2(src_diff, src_stride, coeff, 1);
}

void vpx_hadamard_32x32_sse2(const int16_t *src_diff, ptrdiff_t src_stride,
                             tran_low_t *coeff) {
#if CONFIG_VP9_HIGHBITDEPTH
  // For high bitdepths, it is unnecessary to store_tran_low
  // (mult/unpack/store), then load_tran_low (load/pack) the same memory in the
  // next stage.  Output to an intermediate buffer first, then store_tran_low()
  // in the final stage.
  DECLARE_ALIGNED(32, int16_t, temp_coeff[32 * 32]);
  int16_t *t_coeff = temp_coeff;
#else
  int16_t *t_coeff = coeff;
#endif
  int idx;
  __m128i coeff0_lo, coeff1_lo, coeff2_lo, coeff3_lo, b0_lo, b1_lo, b2_lo,
      b3_lo;
  __m128i coeff0_hi, coeff1_hi, coeff2_hi, coeff3_hi, b0_hi, b1_hi, b2_hi,
      b3_hi;
  __m128i b0, b1, b2, b3;
  const __m128i zero = _mm_setzero_si128();
  for (idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    hadamard_16x16_sse2(src_ptr, src_stride,
                        (tran_low_t *)(t_coeff + idx * 256), 0);
  }

  for (idx = 0; idx < 256; idx += 8) {
    __m128i coeff0 = _mm_load_si128((const __m128i *)t_coeff);
    __m128i coeff1 = _mm_load_si128((const __m128i *)(t_coeff + 256));
    __m128i coeff2 = _mm_load_si128((const __m128i *)(t_coeff + 512));
    __m128i coeff3 = _mm_load_si128((const __m128i *)(t_coeff + 768));

    // Sign extend 16 bit to 32 bit.
    sign_extend_16bit_to_32bit_sse2(coeff0, zero, &coeff0_lo, &coeff0_hi);
    sign_extend_16bit_to_32bit_sse2(coeff1, zero, &coeff1_lo, &coeff1_hi);
    sign_extend_16bit_to_32bit_sse2(coeff2, zero, &coeff2_lo, &coeff2_hi);
    sign_extend_16bit_to_32bit_sse2(coeff3, zero, &coeff3_lo, &coeff3_hi);

    b0_lo = _mm_add_epi32(coeff0_lo, coeff1_lo);
    b0_hi = _mm_add_epi32(coeff0_hi, coeff1_hi);

    b1_lo = _mm_sub_epi32(coeff0_lo, coeff1_lo);
    b1_hi = _mm_sub_epi32(coeff0_hi, coeff1_hi);

    b2_lo = _mm_add_epi32(coeff2_lo, coeff3_lo);
    b2_hi = _mm_add_epi32(coeff2_hi, coeff3_hi);

    b3_lo = _mm_sub_epi32(coeff2_lo, coeff3_lo);
    b3_hi = _mm_sub_epi32(coeff2_hi, coeff3_hi);

    b0_lo = _mm_srai_epi32(b0_lo, 2);
    b1_lo = _mm_srai_epi32(b1_lo, 2);
    b2_lo = _mm_srai_epi32(b2_lo, 2);
    b3_lo = _mm_srai_epi32(b3_lo, 2);

    b0_hi = _mm_srai_epi32(b0_hi, 2);
    b1_hi = _mm_srai_epi32(b1_hi, 2);
    b2_hi = _mm_srai_epi32(b2_hi, 2);
    b3_hi = _mm_srai_epi32(b3_hi, 2);

    b0 = _mm_packs_epi32(b0_lo, b0_hi);
    b1 = _mm_packs_epi32(b1_lo, b1_hi);
    b2 = _mm_packs_epi32(b2_lo, b2_hi);
    b3 = _mm_packs_epi32(b3_lo, b3_hi);

    coeff0 = _mm_add_epi16(b0, b2);
    coeff1 = _mm_add_epi16(b1, b3);
    store_tran_low(coeff0, coeff);
    store_tran_low(coeff1, coeff + 256);

    coeff2 = _mm_sub_epi16(b0, b2);
    coeff3 = _mm_sub_epi16(b1, b3);
    store_tran_low(coeff2, coeff + 512);
    store_tran_low(coeff3, coeff + 768);

    coeff += 8;
    t_coeff += 8;
  }
}

int vpx_satd_sse2(const tran_low_t *coeff, int length) {
  int i;
  const __m128i zero = _mm_setzero_si128();
  __m128i accum = zero;

  for (i = 0; i < length; i += 8) {
    const __m128i src_line = load_tran_low(coeff);
    const __m128i inv = _mm_sub_epi16(zero, src_line);
    const __m128i abs = _mm_max_epi16(src_line, inv);  // abs(src_line)
    const __m128i abs_lo = _mm_unpacklo_epi16(abs, zero);
    const __m128i abs_hi = _mm_unpackhi_epi16(abs, zero);
    const __m128i sum = _mm_add_epi32(abs_lo, abs_hi);
    accum = _mm_add_epi32(accum, sum);
    coeff += 8;
  }

  {  // cascading summation of accum
    __m128i hi = _mm_srli_si128(accum, 8);
    accum = _mm_add_epi32(accum, hi);
    hi = _mm_srli_epi64(accum, 32);
    accum = _mm_add_epi32(accum, hi);
  }

  return _mm_cvtsi128_si32(accum);
}

void vpx_int_pro_row_sse2(int16_t hbuf[16], const uint8_t *ref,
                          const int ref_stride, const int height) {
  int idx;
  __m128i zero = _mm_setzero_si128();
  __m128i src_line = _mm_loadu_si128((const __m128i *)ref);
  __m128i s0 = _mm_unpacklo_epi8(src_line, zero);
  __m128i s1 = _mm_unpackhi_epi8(src_line, zero);
  __m128i t0, t1;
  int height_1 = height - 1;
  ref += ref_stride;

  for (idx = 1; idx < height_1; idx += 2) {
    src_line = _mm_loadu_si128((const __m128i *)ref);
    t0 = _mm_unpacklo_epi8(src_line, zero);
    t1 = _mm_unpackhi_epi8(src_line, zero);
    s0 = _mm_adds_epu16(s0, t0);
    s1 = _mm_adds_epu16(s1, t1);
    ref += ref_stride;

    src_line = _mm_loadu_si128((const __m128i *)ref);
    t0 = _mm_unpacklo_epi8(src_line, zero);
    t1 = _mm_unpackhi_epi8(src_line, zero);
    s0 = _mm_adds_epu16(s0, t0);
    s1 = _mm_adds_epu16(s1, t1);
    ref += ref_stride;
  }

  src_line = _mm_loadu_si128((const __m128i *)ref);
  t0 = _mm_unpacklo_epi8(src_line, zero);
  t1 = _mm_unpackhi_epi8(src_line, zero);
  s0 = _mm_adds_epu16(s0, t0);
  s1 = _mm_adds_epu16(s1, t1);

  if (height == 64) {
    s0 = _mm_srai_epi16(s0, 5);
    s1 = _mm_srai_epi16(s1, 5);
  } else if (height == 32) {
    s0 = _mm_srai_epi16(s0, 4);
    s1 = _mm_srai_epi16(s1, 4);
  } else {
    s0 = _mm_srai_epi16(s0, 3);
    s1 = _mm_srai_epi16(s1, 3);
  }

  _mm_storeu_si128((__m128i *)hbuf, s0);
  hbuf += 8;
  _mm_storeu_si128((__m128i *)hbuf, s1);
}

int16_t vpx_int_pro_col_sse2(const uint8_t *ref, const int width) {
  __m128i zero = _mm_setzero_si128();
  __m128i src_line = _mm_loadu_si128((const __m128i *)ref);
  __m128i s0 = _mm_sad_epu8(src_line, zero);
  __m128i s1;
  int i;

  for (i = 16; i < width; i += 16) {
    ref += 16;
    src_line = _mm_loadu_si128((const __m128i *)ref);
    s1 = _mm_sad_epu8(src_line, zero);
    s0 = _mm_adds_epu16(s0, s1);
  }

  s1 = _mm_srli_si128(s0, 8);
  s0 = _mm_adds_epu16(s0, s1);

  return _mm_extract_epi16(s0, 0);
}

int vpx_vector_var_sse2(const int16_t *ref, const int16_t *src, const int bwl) {
  int idx;
  int width = 4 << bwl;
  int16_t mean;
  __m128i v0 = _mm_loadu_si128((const __m128i *)ref);
  __m128i v1 = _mm_load_si128((const __m128i *)src);
  __m128i diff = _mm_subs_epi16(v0, v1);
  __m128i sum = diff;
  __m128i sse = _mm_madd_epi16(diff, diff);

  ref += 8;
  src += 8;

  for (idx = 8; idx < width; idx += 8) {
    v0 = _mm_loadu_si128((const __m128i *)ref);
    v1 = _mm_load_si128((const __m128i *)src);
    diff = _mm_subs_epi16(v0, v1);

    sum = _mm_add_epi16(sum, diff);
    v0 = _mm_madd_epi16(diff, diff);
    sse = _mm_add_epi32(sse, v0);

    ref += 8;
    src += 8;
  }

  v0 = _mm_srli_si128(sum, 8);
  sum = _mm_add_epi16(sum, v0);
  v0 = _mm_srli_epi64(sum, 32);
  sum = _mm_add_epi16(sum, v0);
  v0 = _mm_srli_epi32(sum, 16);
  sum = _mm_add_epi16(sum, v0);

  v1 = _mm_srli_si128(sse, 8);
  sse = _mm_add_epi32(sse, v1);
  v1 = _mm_srli_epi64(sse, 32);
  sse = _mm_add_epi32(sse, v1);

  mean = (int16_t)_mm_extract_epi16(sum, 0);

  return _mm_cvtsi128_si32(sse) - ((mean * mean) >> (bwl + 2));
}
