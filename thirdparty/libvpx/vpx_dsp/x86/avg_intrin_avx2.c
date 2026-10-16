/*
 *  Copyright (c) 2017 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/bitdepth_conversion_avx2.h"
#include "vpx_ports/mem.h"

#if CONFIG_VP9_HIGHBITDEPTH
static void highbd_hadamard_col8_avx2(__m256i *in, int iter) {
  __m256i a0 = in[0];
  __m256i a1 = in[1];
  __m256i a2 = in[2];
  __m256i a3 = in[3];
  __m256i a4 = in[4];
  __m256i a5 = in[5];
  __m256i a6 = in[6];
  __m256i a7 = in[7];

  __m256i b0 = _mm256_add_epi32(a0, a1);
  __m256i b1 = _mm256_sub_epi32(a0, a1);
  __m256i b2 = _mm256_add_epi32(a2, a3);
  __m256i b3 = _mm256_sub_epi32(a2, a3);
  __m256i b4 = _mm256_add_epi32(a4, a5);
  __m256i b5 = _mm256_sub_epi32(a4, a5);
  __m256i b6 = _mm256_add_epi32(a6, a7);
  __m256i b7 = _mm256_sub_epi32(a6, a7);

  a0 = _mm256_add_epi32(b0, b2);
  a1 = _mm256_add_epi32(b1, b3);
  a2 = _mm256_sub_epi32(b0, b2);
  a3 = _mm256_sub_epi32(b1, b3);
  a4 = _mm256_add_epi32(b4, b6);
  a5 = _mm256_add_epi32(b5, b7);
  a6 = _mm256_sub_epi32(b4, b6);
  a7 = _mm256_sub_epi32(b5, b7);

  if (iter == 0) {
    b0 = _mm256_add_epi32(a0, a4);
    b7 = _mm256_add_epi32(a1, a5);
    b3 = _mm256_add_epi32(a2, a6);
    b4 = _mm256_add_epi32(a3, a7);
    b2 = _mm256_sub_epi32(a0, a4);
    b6 = _mm256_sub_epi32(a1, a5);
    b1 = _mm256_sub_epi32(a2, a6);
    b5 = _mm256_sub_epi32(a3, a7);

    a0 = _mm256_unpacklo_epi32(b0, b1);
    a1 = _mm256_unpacklo_epi32(b2, b3);
    a2 = _mm256_unpackhi_epi32(b0, b1);
    a3 = _mm256_unpackhi_epi32(b2, b3);
    a4 = _mm256_unpacklo_epi32(b4, b5);
    a5 = _mm256_unpacklo_epi32(b6, b7);
    a6 = _mm256_unpackhi_epi32(b4, b5);
    a7 = _mm256_unpackhi_epi32(b6, b7);

    b0 = _mm256_unpacklo_epi64(a0, a1);
    b1 = _mm256_unpacklo_epi64(a4, a5);
    b2 = _mm256_unpackhi_epi64(a0, a1);
    b3 = _mm256_unpackhi_epi64(a4, a5);
    b4 = _mm256_unpacklo_epi64(a2, a3);
    b5 = _mm256_unpacklo_epi64(a6, a7);
    b6 = _mm256_unpackhi_epi64(a2, a3);
    b7 = _mm256_unpackhi_epi64(a6, a7);

    in[0] = _mm256_permute2x128_si256(b0, b1, 0x20);
    in[1] = _mm256_permute2x128_si256(b0, b1, 0x31);
    in[2] = _mm256_permute2x128_si256(b2, b3, 0x20);
    in[3] = _mm256_permute2x128_si256(b2, b3, 0x31);
    in[4] = _mm256_permute2x128_si256(b4, b5, 0x20);
    in[5] = _mm256_permute2x128_si256(b4, b5, 0x31);
    in[6] = _mm256_permute2x128_si256(b6, b7, 0x20);
    in[7] = _mm256_permute2x128_si256(b6, b7, 0x31);
  } else {
    in[0] = _mm256_add_epi32(a0, a4);
    in[7] = _mm256_add_epi32(a1, a5);
    in[3] = _mm256_add_epi32(a2, a6);
    in[4] = _mm256_add_epi32(a3, a7);
    in[2] = _mm256_sub_epi32(a0, a4);
    in[6] = _mm256_sub_epi32(a1, a5);
    in[1] = _mm256_sub_epi32(a2, a6);
    in[5] = _mm256_sub_epi32(a3, a7);
  }
}

void vpx_highbd_hadamard_8x8_avx2(const int16_t *src_diff, ptrdiff_t src_stride,
                                  tran_low_t *coeff) {
  __m128i src16[8];
  __m256i src32[8];

  src16[0] = _mm_loadu_si128((const __m128i *)src_diff);
  src16[1] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[2] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[3] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[4] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[5] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[6] = _mm_loadu_si128((const __m128i *)(src_diff += src_stride));
  src16[7] = _mm_loadu_si128((const __m128i *)(src_diff + src_stride));

  src32[0] = _mm256_cvtepi16_epi32(src16[0]);
  src32[1] = _mm256_cvtepi16_epi32(src16[1]);
  src32[2] = _mm256_cvtepi16_epi32(src16[2]);
  src32[3] = _mm256_cvtepi16_epi32(src16[3]);
  src32[4] = _mm256_cvtepi16_epi32(src16[4]);
  src32[5] = _mm256_cvtepi16_epi32(src16[5]);
  src32[6] = _mm256_cvtepi16_epi32(src16[6]);
  src32[7] = _mm256_cvtepi16_epi32(src16[7]);

  highbd_hadamard_col8_avx2(src32, 0);
  highbd_hadamard_col8_avx2(src32, 1);

  _mm256_storeu_si256((__m256i *)coeff, src32[0]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[1]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[2]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[3]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[4]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[5]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[6]);
  coeff += 8;
  _mm256_storeu_si256((__m256i *)coeff, src32[7]);
}

void vpx_highbd_hadamard_16x16_avx2(const int16_t *src_diff,
                                    ptrdiff_t src_stride, tran_low_t *coeff) {
  int idx;
  tran_low_t *t_coeff = coeff;
  for (idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 8 * src_stride + (idx & 0x01) * 8;
    vpx_highbd_hadamard_8x8_avx2(src_ptr, src_stride, t_coeff + idx * 64);
  }

  for (idx = 0; idx < 64; idx += 8) {
    __m256i coeff0 = _mm256_loadu_si256((const __m256i *)t_coeff);
    __m256i coeff1 = _mm256_loadu_si256((const __m256i *)(t_coeff + 64));
    __m256i coeff2 = _mm256_loadu_si256((const __m256i *)(t_coeff + 128));
    __m256i coeff3 = _mm256_loadu_si256((const __m256i *)(t_coeff + 192));

    __m256i b0 = _mm256_add_epi32(coeff0, coeff1);
    __m256i b1 = _mm256_sub_epi32(coeff0, coeff1);
    __m256i b2 = _mm256_add_epi32(coeff2, coeff3);
    __m256i b3 = _mm256_sub_epi32(coeff2, coeff3);

    b0 = _mm256_srai_epi32(b0, 1);
    b1 = _mm256_srai_epi32(b1, 1);
    b2 = _mm256_srai_epi32(b2, 1);
    b3 = _mm256_srai_epi32(b3, 1);

    coeff0 = _mm256_add_epi32(b0, b2);
    coeff1 = _mm256_add_epi32(b1, b3);
    coeff2 = _mm256_sub_epi32(b0, b2);
    coeff3 = _mm256_sub_epi32(b1, b3);

    _mm256_storeu_si256((__m256i *)coeff, coeff0);
    _mm256_storeu_si256((__m256i *)(coeff + 64), coeff1);
    _mm256_storeu_si256((__m256i *)(coeff + 128), coeff2);
    _mm256_storeu_si256((__m256i *)(coeff + 192), coeff3);

    coeff += 8;
    t_coeff += 8;
  }
}

void vpx_highbd_hadamard_32x32_avx2(const int16_t *src_diff,
                                    ptrdiff_t src_stride, tran_low_t *coeff) {
  int idx;
  tran_low_t *t_coeff = coeff;
  for (idx = 0; idx < 4; ++idx) {
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    vpx_highbd_hadamard_16x16_avx2(src_ptr, src_stride, t_coeff + idx * 256);
  }

  for (idx = 0; idx < 256; idx += 8) {
    __m256i coeff0 = _mm256_loadu_si256((const __m256i *)t_coeff);
    __m256i coeff1 = _mm256_loadu_si256((const __m256i *)(t_coeff + 256));
    __m256i coeff2 = _mm256_loadu_si256((const __m256i *)(t_coeff + 512));
    __m256i coeff3 = _mm256_loadu_si256((const __m256i *)(t_coeff + 768));

    __m256i b0 = _mm256_add_epi32(coeff0, coeff1);
    __m256i b1 = _mm256_sub_epi32(coeff0, coeff1);
    __m256i b2 = _mm256_add_epi32(coeff2, coeff3);
    __m256i b3 = _mm256_sub_epi32(coeff2, coeff3);

    b0 = _mm256_srai_epi32(b0, 2);
    b1 = _mm256_srai_epi32(b1, 2);
    b2 = _mm256_srai_epi32(b2, 2);
    b3 = _mm256_srai_epi32(b3, 2);

    coeff0 = _mm256_add_epi32(b0, b2);
    coeff1 = _mm256_add_epi32(b1, b3);
    coeff2 = _mm256_sub_epi32(b0, b2);
    coeff3 = _mm256_sub_epi32(b1, b3);

    _mm256_storeu_si256((__m256i *)coeff, coeff0);
    _mm256_storeu_si256((__m256i *)(coeff + 256), coeff1);
    _mm256_storeu_si256((__m256i *)(coeff + 512), coeff2);
    _mm256_storeu_si256((__m256i *)(coeff + 768), coeff3);

    coeff += 8;
    t_coeff += 8;
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

static INLINE void sign_extend_16bit_to_32bit_avx2(__m256i in, __m256i zero,
                                                   __m256i *out_lo,
                                                   __m256i *out_hi) {
  const __m256i sign_bits = _mm256_cmpgt_epi16(zero, in);
  *out_lo = _mm256_unpacklo_epi16(in, sign_bits);
  *out_hi = _mm256_unpackhi_epi16(in, sign_bits);
}

static void hadamard_col8x2_avx2(__m256i *in, int iter) {
  __m256i a0 = in[0];
  __m256i a1 = in[1];
  __m256i a2 = in[2];
  __m256i a3 = in[3];
  __m256i a4 = in[4];
  __m256i a5 = in[5];
  __m256i a6 = in[6];
  __m256i a7 = in[7];

  __m256i b0 = _mm256_add_epi16(a0, a1);
  __m256i b1 = _mm256_sub_epi16(a0, a1);
  __m256i b2 = _mm256_add_epi16(a2, a3);
  __m256i b3 = _mm256_sub_epi16(a2, a3);
  __m256i b4 = _mm256_add_epi16(a4, a5);
  __m256i b5 = _mm256_sub_epi16(a4, a5);
  __m256i b6 = _mm256_add_epi16(a6, a7);
  __m256i b7 = _mm256_sub_epi16(a6, a7);

  a0 = _mm256_add_epi16(b0, b2);
  a1 = _mm256_add_epi16(b1, b3);
  a2 = _mm256_sub_epi16(b0, b2);
  a3 = _mm256_sub_epi16(b1, b3);
  a4 = _mm256_add_epi16(b4, b6);
  a5 = _mm256_add_epi16(b5, b7);
  a6 = _mm256_sub_epi16(b4, b6);
  a7 = _mm256_sub_epi16(b5, b7);

  if (iter == 0) {
    b0 = _mm256_add_epi16(a0, a4);
    b7 = _mm256_add_epi16(a1, a5);
    b3 = _mm256_add_epi16(a2, a6);
    b4 = _mm256_add_epi16(a3, a7);
    b2 = _mm256_sub_epi16(a0, a4);
    b6 = _mm256_sub_epi16(a1, a5);
    b1 = _mm256_sub_epi16(a2, a6);
    b5 = _mm256_sub_epi16(a3, a7);

    a0 = _mm256_unpacklo_epi16(b0, b1);
    a1 = _mm256_unpacklo_epi16(b2, b3);
    a2 = _mm256_unpackhi_epi16(b0, b1);
    a3 = _mm256_unpackhi_epi16(b2, b3);
    a4 = _mm256_unpacklo_epi16(b4, b5);
    a5 = _mm256_unpacklo_epi16(b6, b7);
    a6 = _mm256_unpackhi_epi16(b4, b5);
    a7 = _mm256_unpackhi_epi16(b6, b7);

    b0 = _mm256_unpacklo_epi32(a0, a1);
    b1 = _mm256_unpacklo_epi32(a4, a5);
    b2 = _mm256_unpackhi_epi32(a0, a1);
    b3 = _mm256_unpackhi_epi32(a4, a5);
    b4 = _mm256_unpacklo_epi32(a2, a3);
    b5 = _mm256_unpacklo_epi32(a6, a7);
    b6 = _mm256_unpackhi_epi32(a2, a3);
    b7 = _mm256_unpackhi_epi32(a6, a7);

    in[0] = _mm256_unpacklo_epi64(b0, b1);
    in[1] = _mm256_unpackhi_epi64(b0, b1);
    in[2] = _mm256_unpacklo_epi64(b2, b3);
    in[3] = _mm256_unpackhi_epi64(b2, b3);
    in[4] = _mm256_unpacklo_epi64(b4, b5);
    in[5] = _mm256_unpackhi_epi64(b4, b5);
    in[6] = _mm256_unpacklo_epi64(b6, b7);
    in[7] = _mm256_unpackhi_epi64(b6, b7);
  } else {
    in[0] = _mm256_add_epi16(a0, a4);
    in[7] = _mm256_add_epi16(a1, a5);
    in[3] = _mm256_add_epi16(a2, a6);
    in[4] = _mm256_add_epi16(a3, a7);
    in[2] = _mm256_sub_epi16(a0, a4);
    in[6] = _mm256_sub_epi16(a1, a5);
    in[1] = _mm256_sub_epi16(a2, a6);
    in[5] = _mm256_sub_epi16(a3, a7);
  }
}

static void hadamard_8x8x2_avx2(const int16_t *src_diff, ptrdiff_t src_stride,
                                int16_t *coeff) {
  __m256i src[8];
  src[0] = _mm256_loadu_si256((const __m256i *)src_diff);
  src[1] = _mm256_loadu_si256((const __m256i *)(src_diff += src_stride));
  src[2] = _mm256_loadu_si256((const __m256i *)(src_diff += src_stride));
  src[3] = _mm256_loadu_si256((const __m256i *)(src_diff += src_stride));
  src[4] = _mm256_loadu_si256((const __m256i *)(src_diff += src_stride));
  src[5] = _mm256_loadu_si256((const __m256i *)(src_diff += src_stride));
  src[6] = _mm256_loadu_si256((const __m256i *)(src_diff += src_stride));
  src[7] = _mm256_loadu_si256((const __m256i *)(src_diff + src_stride));

  hadamard_col8x2_avx2(src, 0);
  hadamard_col8x2_avx2(src, 1);

  _mm256_storeu_si256((__m256i *)coeff,
                      _mm256_permute2x128_si256(src[0], src[1], 0x20));
  coeff += 16;
  _mm256_storeu_si256((__m256i *)coeff,
                      _mm256_permute2x128_si256(src[2], src[3], 0x20));
  coeff += 16;
  _mm256_storeu_si256((__m256i *)coeff,
                      _mm256_permute2x128_si256(src[4], src[5], 0x20));
  coeff += 16;
  _mm256_storeu_si256((__m256i *)coeff,
                      _mm256_permute2x128_si256(src[6], src[7], 0x20));
  coeff += 16;
  _mm256_storeu_si256((__m256i *)coeff,
                      _mm256_permute2x128_si256(src[0], src[1], 0x31));
  coeff += 16;
  _mm256_storeu_si256((__m256i *)coeff,
                      _mm256_permute2x128_si256(src[2], src[3], 0x31));
  coeff += 16;
  _mm256_storeu_si256((__m256i *)coeff,
                      _mm256_permute2x128_si256(src[4], src[5], 0x31));
  coeff += 16;
  _mm256_storeu_si256((__m256i *)coeff,
                      _mm256_permute2x128_si256(src[6], src[7], 0x31));
}

static INLINE void hadamard_16x16_avx2(const int16_t *src_diff,
                                       ptrdiff_t src_stride, tran_low_t *coeff,
                                       int is_final) {
#if CONFIG_VP9_HIGHBITDEPTH
  DECLARE_ALIGNED(32, int16_t, temp_coeff[16 * 16]);
  int16_t *t_coeff = temp_coeff;
#else
  int16_t *t_coeff = coeff;
#endif
  int16_t *coeff16 = (int16_t *)coeff;
  int idx;
  for (idx = 0; idx < 2; ++idx) {
    const int16_t *src_ptr = src_diff + idx * 8 * src_stride;
    hadamard_8x8x2_avx2(src_ptr, src_stride, t_coeff + (idx * 64 * 2));
  }

  for (idx = 0; idx < 64; idx += 16) {
    const __m256i coeff0 = _mm256_loadu_si256((const __m256i *)t_coeff);
    const __m256i coeff1 = _mm256_loadu_si256((const __m256i *)(t_coeff + 64));
    const __m256i coeff2 = _mm256_loadu_si256((const __m256i *)(t_coeff + 128));
    const __m256i coeff3 = _mm256_loadu_si256((const __m256i *)(t_coeff + 192));

    __m256i b0 = _mm256_add_epi16(coeff0, coeff1);
    __m256i b1 = _mm256_sub_epi16(coeff0, coeff1);
    __m256i b2 = _mm256_add_epi16(coeff2, coeff3);
    __m256i b3 = _mm256_sub_epi16(coeff2, coeff3);

    b0 = _mm256_srai_epi16(b0, 1);
    b1 = _mm256_srai_epi16(b1, 1);
    b2 = _mm256_srai_epi16(b2, 1);
    b3 = _mm256_srai_epi16(b3, 1);
    if (is_final) {
      store_tran_low(_mm256_add_epi16(b0, b2), coeff);
      store_tran_low(_mm256_add_epi16(b1, b3), coeff + 64);
      store_tran_low(_mm256_sub_epi16(b0, b2), coeff + 128);
      store_tran_low(_mm256_sub_epi16(b1, b3), coeff + 192);
      coeff += 16;
    } else {
      _mm256_storeu_si256((__m256i *)coeff16, _mm256_add_epi16(b0, b2));
      _mm256_storeu_si256((__m256i *)(coeff16 + 64), _mm256_add_epi16(b1, b3));
      _mm256_storeu_si256((__m256i *)(coeff16 + 128), _mm256_sub_epi16(b0, b2));
      _mm256_storeu_si256((__m256i *)(coeff16 + 192), _mm256_sub_epi16(b1, b3));
      coeff16 += 16;
    }
    t_coeff += 16;
  }
}

void vpx_hadamard_16x16_avx2(const int16_t *src_diff, ptrdiff_t src_stride,
                             tran_low_t *coeff) {
  hadamard_16x16_avx2(src_diff, src_stride, coeff, 1);
}

void vpx_hadamard_32x32_avx2(const int16_t *src_diff, ptrdiff_t src_stride,
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
  __m256i coeff0_lo, coeff1_lo, coeff2_lo, coeff3_lo, b0_lo, b1_lo, b2_lo,
      b3_lo;
  __m256i coeff0_hi, coeff1_hi, coeff2_hi, coeff3_hi, b0_hi, b1_hi, b2_hi,
      b3_hi;
  __m256i b0, b1, b2, b3;
  const __m256i zero = _mm256_setzero_si256();
  for (idx = 0; idx < 4; ++idx) {
    // src_diff: 9 bit, dynamic range [-255, 255]
    const int16_t *src_ptr =
        src_diff + (idx >> 1) * 16 * src_stride + (idx & 0x01) * 16;
    hadamard_16x16_avx2(src_ptr, src_stride,
                        (tran_low_t *)(t_coeff + idx * 256), 0);
  }

  for (idx = 0; idx < 256; idx += 16) {
    const __m256i coeff0 = _mm256_loadu_si256((const __m256i *)t_coeff);
    const __m256i coeff1 = _mm256_loadu_si256((const __m256i *)(t_coeff + 256));
    const __m256i coeff2 = _mm256_loadu_si256((const __m256i *)(t_coeff + 512));
    const __m256i coeff3 = _mm256_loadu_si256((const __m256i *)(t_coeff + 768));

    // Sign extend 16 bit to 32 bit.
    sign_extend_16bit_to_32bit_avx2(coeff0, zero, &coeff0_lo, &coeff0_hi);
    sign_extend_16bit_to_32bit_avx2(coeff1, zero, &coeff1_lo, &coeff1_hi);
    sign_extend_16bit_to_32bit_avx2(coeff2, zero, &coeff2_lo, &coeff2_hi);
    sign_extend_16bit_to_32bit_avx2(coeff3, zero, &coeff3_lo, &coeff3_hi);

    b0_lo = _mm256_add_epi32(coeff0_lo, coeff1_lo);
    b0_hi = _mm256_add_epi32(coeff0_hi, coeff1_hi);

    b1_lo = _mm256_sub_epi32(coeff0_lo, coeff1_lo);
    b1_hi = _mm256_sub_epi32(coeff0_hi, coeff1_hi);

    b2_lo = _mm256_add_epi32(coeff2_lo, coeff3_lo);
    b2_hi = _mm256_add_epi32(coeff2_hi, coeff3_hi);

    b3_lo = _mm256_sub_epi32(coeff2_lo, coeff3_lo);
    b3_hi = _mm256_sub_epi32(coeff2_hi, coeff3_hi);

    b0_lo = _mm256_srai_epi32(b0_lo, 2);
    b1_lo = _mm256_srai_epi32(b1_lo, 2);
    b2_lo = _mm256_srai_epi32(b2_lo, 2);
    b3_lo = _mm256_srai_epi32(b3_lo, 2);

    b0_hi = _mm256_srai_epi32(b0_hi, 2);
    b1_hi = _mm256_srai_epi32(b1_hi, 2);
    b2_hi = _mm256_srai_epi32(b2_hi, 2);
    b3_hi = _mm256_srai_epi32(b3_hi, 2);

    b0 = _mm256_packs_epi32(b0_lo, b0_hi);
    b1 = _mm256_packs_epi32(b1_lo, b1_hi);
    b2 = _mm256_packs_epi32(b2_lo, b2_hi);
    b3 = _mm256_packs_epi32(b3_lo, b3_hi);

    store_tran_low(_mm256_add_epi16(b0, b2), coeff);
    store_tran_low(_mm256_add_epi16(b1, b3), coeff + 256);
    store_tran_low(_mm256_sub_epi16(b0, b2), coeff + 512);
    store_tran_low(_mm256_sub_epi16(b1, b3), coeff + 768);

    coeff += 16;
    t_coeff += 16;
  }
}

int vpx_satd_avx2(const tran_low_t *coeff, int length) {
  const __m256i one = _mm256_set1_epi16(1);
  __m256i accum = _mm256_setzero_si256();
  int i;

  for (i = 0; i < length; i += 16) {
    const __m256i src_line = load_tran_low(coeff);
    const __m256i abs = _mm256_abs_epi16(src_line);
    const __m256i sum = _mm256_madd_epi16(abs, one);
    accum = _mm256_add_epi32(accum, sum);
    coeff += 16;
  }

  {  // 32 bit horizontal add
    const __m256i a = _mm256_srli_si256(accum, 8);
    const __m256i b = _mm256_add_epi32(accum, a);
    const __m256i c = _mm256_srli_epi64(b, 32);
    const __m256i d = _mm256_add_epi32(b, c);
    const __m128i accum_128 = _mm_add_epi32(_mm256_castsi256_si128(d),
                                            _mm256_extractf128_si256(d, 1));
    return _mm_cvtsi128_si32(accum_128);
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
int vpx_highbd_satd_avx2(const tran_low_t *coeff, int length) {
  __m256i accum = _mm256_setzero_si256();
  int i;

  for (i = 0; i < length; i += 8, coeff += 8) {
    const __m256i src_line = _mm256_loadu_si256((const __m256i *)coeff);
    const __m256i abs = _mm256_abs_epi32(src_line);
    accum = _mm256_add_epi32(accum, abs);
  }

  {  // 32 bit horizontal add
    const __m256i a = _mm256_srli_si256(accum, 8);
    const __m256i b = _mm256_add_epi32(accum, a);
    const __m256i c = _mm256_srli_epi64(b, 32);
    const __m256i d = _mm256_add_epi32(b, c);
    const __m128i accum_128 = _mm_add_epi32(_mm256_castsi256_si128(d),
                                            _mm256_extractf128_si256(d, 1));
    return _mm_cvtsi128_si32(accum_128);
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
