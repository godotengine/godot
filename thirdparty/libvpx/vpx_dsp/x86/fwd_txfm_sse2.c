/*
 *  Copyright (c) 2015 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>  // SSE2

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/x86/fwd_txfm_sse2.h"

void vpx_fdct4x4_1_sse2(const int16_t *input, tran_low_t *output, int stride) {
  __m128i in0, in1;
  __m128i tmp;
  const __m128i zero = _mm_setzero_si128();
  in0 = _mm_loadl_epi64((const __m128i *)(input + 0 * stride));
  in1 = _mm_loadl_epi64((const __m128i *)(input + 1 * stride));
  in1 = _mm_unpacklo_epi64(
      in1, _mm_loadl_epi64((const __m128i *)(input + 2 * stride)));
  in0 = _mm_unpacklo_epi64(
      in0, _mm_loadl_epi64((const __m128i *)(input + 3 * stride)));

  tmp = _mm_add_epi16(in0, in1);
  in0 = _mm_unpacklo_epi16(zero, tmp);
  in1 = _mm_unpackhi_epi16(zero, tmp);
  in0 = _mm_srai_epi32(in0, 16);
  in1 = _mm_srai_epi32(in1, 16);

  tmp = _mm_add_epi32(in0, in1);
  in0 = _mm_unpacklo_epi32(tmp, zero);
  in1 = _mm_unpackhi_epi32(tmp, zero);

  tmp = _mm_add_epi32(in0, in1);
  in0 = _mm_srli_si128(tmp, 8);

  in1 = _mm_add_epi32(tmp, in0);
  in0 = _mm_slli_epi32(in1, 1);
  output[0] = (tran_low_t)_mm_cvtsi128_si32(in0);
}

void vpx_fdct8x8_1_sse2(const int16_t *input, tran_low_t *output, int stride) {
  __m128i in0 = _mm_load_si128((const __m128i *)(input + 0 * stride));
  __m128i in1 = _mm_load_si128((const __m128i *)(input + 1 * stride));
  __m128i in2 = _mm_load_si128((const __m128i *)(input + 2 * stride));
  __m128i in3 = _mm_load_si128((const __m128i *)(input + 3 * stride));
  __m128i u0, u1, sum;

  u0 = _mm_add_epi16(in0, in1);
  u1 = _mm_add_epi16(in2, in3);

  in0 = _mm_load_si128((const __m128i *)(input + 4 * stride));
  in1 = _mm_load_si128((const __m128i *)(input + 5 * stride));
  in2 = _mm_load_si128((const __m128i *)(input + 6 * stride));
  in3 = _mm_load_si128((const __m128i *)(input + 7 * stride));

  sum = _mm_add_epi16(u0, u1);

  in0 = _mm_add_epi16(in0, in1);
  in2 = _mm_add_epi16(in2, in3);
  sum = _mm_add_epi16(sum, in0);

  u0 = _mm_setzero_si128();
  sum = _mm_add_epi16(sum, in2);

  in0 = _mm_unpacklo_epi16(u0, sum);
  in1 = _mm_unpackhi_epi16(u0, sum);
  in0 = _mm_srai_epi32(in0, 16);
  in1 = _mm_srai_epi32(in1, 16);

  sum = _mm_add_epi32(in0, in1);
  in0 = _mm_unpacklo_epi32(sum, u0);
  in1 = _mm_unpackhi_epi32(sum, u0);

  sum = _mm_add_epi32(in0, in1);
  in0 = _mm_srli_si128(sum, 8);

  in1 = _mm_add_epi32(sum, in0);
  output[0] = (tran_low_t)_mm_cvtsi128_si32(in1);
}

void vpx_fdct16x16_1_sse2(const int16_t *input, tran_low_t *output,
                          int stride) {
  __m128i in0, in1, in2, in3;
  __m128i u0, u1;
  __m128i sum = _mm_setzero_si128();
  int i;

  for (i = 0; i < 2; ++i) {
    in0 = _mm_load_si128((const __m128i *)(input + 0 * stride + 0));
    in1 = _mm_load_si128((const __m128i *)(input + 0 * stride + 8));
    in2 = _mm_load_si128((const __m128i *)(input + 1 * stride + 0));
    in3 = _mm_load_si128((const __m128i *)(input + 1 * stride + 8));

    u0 = _mm_add_epi16(in0, in1);
    u1 = _mm_add_epi16(in2, in3);
    sum = _mm_add_epi16(sum, u0);

    in0 = _mm_load_si128((const __m128i *)(input + 2 * stride + 0));
    in1 = _mm_load_si128((const __m128i *)(input + 2 * stride + 8));
    in2 = _mm_load_si128((const __m128i *)(input + 3 * stride + 0));
    in3 = _mm_load_si128((const __m128i *)(input + 3 * stride + 8));

    sum = _mm_add_epi16(sum, u1);
    u0 = _mm_add_epi16(in0, in1);
    u1 = _mm_add_epi16(in2, in3);
    sum = _mm_add_epi16(sum, u0);

    in0 = _mm_load_si128((const __m128i *)(input + 4 * stride + 0));
    in1 = _mm_load_si128((const __m128i *)(input + 4 * stride + 8));
    in2 = _mm_load_si128((const __m128i *)(input + 5 * stride + 0));
    in3 = _mm_load_si128((const __m128i *)(input + 5 * stride + 8));

    sum = _mm_add_epi16(sum, u1);
    u0 = _mm_add_epi16(in0, in1);
    u1 = _mm_add_epi16(in2, in3);
    sum = _mm_add_epi16(sum, u0);

    in0 = _mm_load_si128((const __m128i *)(input + 6 * stride + 0));
    in1 = _mm_load_si128((const __m128i *)(input + 6 * stride + 8));
    in2 = _mm_load_si128((const __m128i *)(input + 7 * stride + 0));
    in3 = _mm_load_si128((const __m128i *)(input + 7 * stride + 8));

    sum = _mm_add_epi16(sum, u1);
    u0 = _mm_add_epi16(in0, in1);
    u1 = _mm_add_epi16(in2, in3);
    sum = _mm_add_epi16(sum, u0);

    sum = _mm_add_epi16(sum, u1);
    input += 8 * stride;
  }

  u0 = _mm_setzero_si128();
  in0 = _mm_unpacklo_epi16(u0, sum);
  in1 = _mm_unpackhi_epi16(u0, sum);
  in0 = _mm_srai_epi32(in0, 16);
  in1 = _mm_srai_epi32(in1, 16);

  sum = _mm_add_epi32(in0, in1);
  in0 = _mm_unpacklo_epi32(sum, u0);
  in1 = _mm_unpackhi_epi32(sum, u0);

  sum = _mm_add_epi32(in0, in1);
  in0 = _mm_srli_si128(sum, 8);

  in1 = _mm_add_epi32(sum, in0);
  in1 = _mm_srai_epi32(in1, 1);
  output[0] = (tran_low_t)_mm_cvtsi128_si32(in1);
}

void vpx_fdct32x32_1_sse2(const int16_t *input, tran_low_t *output,
                          int stride) {
  __m128i in0, in1, in2, in3;
  __m128i u0, u1;
  __m128i sum = _mm_setzero_si128();
  int i;

  for (i = 0; i < 8; ++i) {
    in0 = _mm_load_si128((const __m128i *)(input + 0));
    in1 = _mm_load_si128((const __m128i *)(input + 8));
    in2 = _mm_load_si128((const __m128i *)(input + 16));
    in3 = _mm_load_si128((const __m128i *)(input + 24));

    input += stride;
    u0 = _mm_add_epi16(in0, in1);
    u1 = _mm_add_epi16(in2, in3);
    sum = _mm_add_epi16(sum, u0);

    in0 = _mm_load_si128((const __m128i *)(input + 0));
    in1 = _mm_load_si128((const __m128i *)(input + 8));
    in2 = _mm_load_si128((const __m128i *)(input + 16));
    in3 = _mm_load_si128((const __m128i *)(input + 24));

    input += stride;
    sum = _mm_add_epi16(sum, u1);
    u0 = _mm_add_epi16(in0, in1);
    u1 = _mm_add_epi16(in2, in3);
    sum = _mm_add_epi16(sum, u0);

    in0 = _mm_load_si128((const __m128i *)(input + 0));
    in1 = _mm_load_si128((const __m128i *)(input + 8));
    in2 = _mm_load_si128((const __m128i *)(input + 16));
    in3 = _mm_load_si128((const __m128i *)(input + 24));

    input += stride;
    sum = _mm_add_epi16(sum, u1);
    u0 = _mm_add_epi16(in0, in1);
    u1 = _mm_add_epi16(in2, in3);
    sum = _mm_add_epi16(sum, u0);

    in0 = _mm_load_si128((const __m128i *)(input + 0));
    in1 = _mm_load_si128((const __m128i *)(input + 8));
    in2 = _mm_load_si128((const __m128i *)(input + 16));
    in3 = _mm_load_si128((const __m128i *)(input + 24));

    input += stride;
    sum = _mm_add_epi16(sum, u1);
    u0 = _mm_add_epi16(in0, in1);
    u1 = _mm_add_epi16(in2, in3);
    sum = _mm_add_epi16(sum, u0);

    sum = _mm_add_epi16(sum, u1);
  }

  u0 = _mm_setzero_si128();
  in0 = _mm_unpacklo_epi16(u0, sum);
  in1 = _mm_unpackhi_epi16(u0, sum);
  in0 = _mm_srai_epi32(in0, 16);
  in1 = _mm_srai_epi32(in1, 16);

  sum = _mm_add_epi32(in0, in1);
  in0 = _mm_unpacklo_epi32(sum, u0);
  in1 = _mm_unpackhi_epi32(sum, u0);

  sum = _mm_add_epi32(in0, in1);
  in0 = _mm_srli_si128(sum, 8);

  in1 = _mm_add_epi32(sum, in0);
  in1 = _mm_srai_epi32(in1, 3);
  output[0] = (tran_low_t)_mm_cvtsi128_si32(in1);
}

#define DCT_HIGH_BIT_DEPTH 0
#define FDCT4x4_2D vpx_fdct4x4_sse2
#define FDCT8x8_2D vpx_fdct8x8_sse2
#define FDCT16x16_2D vpx_fdct16x16_sse2
#include "vpx_dsp/x86/fwd_txfm_impl_sse2.h"
#undef FDCT4x4_2D
#undef FDCT8x8_2D
#undef FDCT16x16_2D

#define FDCT32x32_2D vpx_fdct32x32_rd_sse2
#define FDCT32x32_HIGH_PRECISION 0
#include "vpx_dsp/x86/fwd_dct32x32_impl_sse2.h"
#undef FDCT32x32_2D
#undef FDCT32x32_HIGH_PRECISION

#define FDCT32x32_2D vpx_fdct32x32_sse2
#define FDCT32x32_HIGH_PRECISION 1
#include "vpx_dsp/x86/fwd_dct32x32_impl_sse2.h"  // NOLINT
#undef FDCT32x32_2D
#undef FDCT32x32_HIGH_PRECISION
#undef DCT_HIGH_BIT_DEPTH

#if CONFIG_VP9_HIGHBITDEPTH
#define DCT_HIGH_BIT_DEPTH 1
#define FDCT4x4_2D vpx_highbd_fdct4x4_sse2
#define FDCT8x8_2D vpx_highbd_fdct8x8_sse2
#define FDCT16x16_2D vpx_highbd_fdct16x16_sse2
#include "vpx_dsp/x86/fwd_txfm_impl_sse2.h"  // NOLINT
#undef FDCT4x4_2D
#undef FDCT8x8_2D
#undef FDCT16x16_2D

#define FDCT32x32_2D vpx_highbd_fdct32x32_rd_sse2
#define FDCT32x32_HIGH_PRECISION 0
#include "vpx_dsp/x86/fwd_dct32x32_impl_sse2.h"  // NOLINT
#undef FDCT32x32_2D
#undef FDCT32x32_HIGH_PRECISION

#define FDCT32x32_2D vpx_highbd_fdct32x32_sse2
#define FDCT32x32_HIGH_PRECISION 1
#include "vpx_dsp/x86/fwd_dct32x32_impl_sse2.h"  // NOLINT
#undef FDCT32x32_2D
#undef FDCT32x32_HIGH_PRECISION
#undef DCT_HIGH_BIT_DEPTH
#endif  // CONFIG_VP9_HIGHBITDEPTH
