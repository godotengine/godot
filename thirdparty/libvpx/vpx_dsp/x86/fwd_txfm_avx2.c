/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <immintrin.h>  // AVX2
#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx_dsp/txfm_common.h"
#define ADD256_EPI16 _mm256_add_epi16
#define SUB256_EPI16 _mm256_sub_epi16

static INLINE void load_buffer_16bit_to_16bit_avx2(const int16_t *in,
                                                   int stride, __m256i *out,
                                                   int out_size, int pass) {
  int i;
  const __m256i kOne = _mm256_set1_epi16(1);
  if (pass == 0) {
    for (i = 0; i < out_size; i++) {
      out[i] = _mm256_loadu_si256((const __m256i *)(in + i * stride));
      // x = x << 2
      out[i] = _mm256_slli_epi16(out[i], 2);
    }
  } else {
    for (i = 0; i < out_size; i++) {
      out[i] = _mm256_loadu_si256((const __m256i *)(in + i * 16));
      // x = (x + 1) >> 2
      out[i] = _mm256_add_epi16(out[i], kOne);
      out[i] = _mm256_srai_epi16(out[i], 2);
    }
  }
}

static INLINE void transpose2_8x8_avx2(const __m256i *const in,
                                       __m256i *const out) {
  int i;
  __m256i t[16], u[16];
  // (1st, 2nd) ==> (lo, hi)
  //   (0, 1)   ==>  (0, 1)
  //   (2, 3)   ==>  (2, 3)
  //   (4, 5)   ==>  (4, 5)
  //   (6, 7)   ==>  (6, 7)
  for (i = 0; i < 4; i++) {
    t[2 * i] = _mm256_unpacklo_epi16(in[2 * i], in[2 * i + 1]);
    t[2 * i + 1] = _mm256_unpackhi_epi16(in[2 * i], in[2 * i + 1]);
  }

  // (1st, 2nd) ==> (lo, hi)
  //   (0, 2)   ==>  (0, 2)
  //   (1, 3)   ==>  (1, 3)
  //   (4, 6)   ==>  (4, 6)
  //   (5, 7)   ==>  (5, 7)
  for (i = 0; i < 2; i++) {
    u[i] = _mm256_unpacklo_epi32(t[i], t[i + 2]);
    u[i + 2] = _mm256_unpackhi_epi32(t[i], t[i + 2]);

    u[i + 4] = _mm256_unpacklo_epi32(t[i + 4], t[i + 6]);
    u[i + 6] = _mm256_unpackhi_epi32(t[i + 4], t[i + 6]);
  }

  // (1st, 2nd) ==> (lo, hi)
  //   (0, 4)   ==>  (0, 1)
  //   (1, 5)   ==>  (4, 5)
  //   (2, 6)   ==>  (2, 3)
  //   (3, 7)   ==>  (6, 7)
  for (i = 0; i < 2; i++) {
    out[2 * i] = _mm256_unpacklo_epi64(u[2 * i], u[2 * i + 4]);
    out[2 * i + 1] = _mm256_unpackhi_epi64(u[2 * i], u[2 * i + 4]);

    out[2 * i + 4] = _mm256_unpacklo_epi64(u[2 * i + 1], u[2 * i + 5]);
    out[2 * i + 5] = _mm256_unpackhi_epi64(u[2 * i + 1], u[2 * i + 5]);
  }
}

static INLINE void transpose_16bit_16x16_avx2(const __m256i *const in,
                                              __m256i *const out) {
  __m256i t[16];

#define LOADL(idx)                                                            \
  t[idx] = _mm256_castsi128_si256(_mm_load_si128((__m128i const *)&in[idx])); \
  t[idx] = _mm256_inserti128_si256(                                           \
      t[idx], _mm_load_si128((__m128i const *)&in[idx + 8]), 1);

#define LOADR(idx)                                                           \
  t[8 + idx] =                                                               \
      _mm256_castsi128_si256(_mm_load_si128((__m128i const *)&in[idx] + 1)); \
  t[8 + idx] = _mm256_inserti128_si256(                                      \
      t[8 + idx], _mm_load_si128((__m128i const *)&in[idx + 8] + 1), 1);

  // load left 8x16
  LOADL(0)
  LOADL(1)
  LOADL(2)
  LOADL(3)
  LOADL(4)
  LOADL(5)
  LOADL(6)
  LOADL(7)

  // load right 8x16
  LOADR(0)
  LOADR(1)
  LOADR(2)
  LOADR(3)
  LOADR(4)
  LOADR(5)
  LOADR(6)
  LOADR(7)

  // get the top 16x8 result
  transpose2_8x8_avx2(t, out);
  // get the bottom 16x8 result
  transpose2_8x8_avx2(&t[8], &out[8]);
}

// Store 8 16-bit values. Sign extend the values.
static INLINE void store_buffer_16bit_to_32bit_w16_avx2(const __m256i *const in,
                                                        tran_low_t *out,
                                                        const int stride,
                                                        const int out_size) {
  int i;
  for (i = 0; i < out_size; ++i) {
    _mm256_storeu_si256((__m256i *)(out), in[i]);
    out += stride;
  }
}

#define PAIR256_SET_EPI16(a, b)                                            \
  _mm256_set_epi16((int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a), \
                   (int16_t)(b), (int16_t)(a), (int16_t)(b), (int16_t)(a))

static INLINE __m256i mult256_round_shift(const __m256i *pin0,
                                          const __m256i *pin1,
                                          const __m256i *pmultiplier,
                                          const __m256i *prounding,
                                          const int shift) {
  const __m256i u0 = _mm256_madd_epi16(*pin0, *pmultiplier);
  const __m256i u1 = _mm256_madd_epi16(*pin1, *pmultiplier);
  const __m256i v0 = _mm256_add_epi32(u0, *prounding);
  const __m256i v1 = _mm256_add_epi32(u1, *prounding);
  const __m256i w0 = _mm256_srai_epi32(v0, shift);
  const __m256i w1 = _mm256_srai_epi32(v1, shift);
  return _mm256_packs_epi32(w0, w1);
}

static INLINE void fdct16x16_1D_avx2(__m256i *input, __m256i *output) {
  int i;
  __m256i step2[4];
  __m256i in[8];
  __m256i step1[8];
  __m256i step3[8];

  const __m256i k__cospi_p16_p16 = _mm256_set1_epi16(cospi_16_64);
  const __m256i k__cospi_p16_m16 = PAIR256_SET_EPI16(cospi_16_64, -cospi_16_64);
  const __m256i k__cospi_p24_p08 = PAIR256_SET_EPI16(cospi_24_64, cospi_8_64);
  const __m256i k__cospi_p08_m24 = PAIR256_SET_EPI16(cospi_8_64, -cospi_24_64);
  const __m256i k__cospi_m08_p24 = PAIR256_SET_EPI16(-cospi_8_64, cospi_24_64);
  const __m256i k__cospi_p28_p04 = PAIR256_SET_EPI16(cospi_28_64, cospi_4_64);
  const __m256i k__cospi_m04_p28 = PAIR256_SET_EPI16(-cospi_4_64, cospi_28_64);
  const __m256i k__cospi_p12_p20 = PAIR256_SET_EPI16(cospi_12_64, cospi_20_64);
  const __m256i k__cospi_m20_p12 = PAIR256_SET_EPI16(-cospi_20_64, cospi_12_64);
  const __m256i k__cospi_p30_p02 = PAIR256_SET_EPI16(cospi_30_64, cospi_2_64);
  const __m256i k__cospi_p14_p18 = PAIR256_SET_EPI16(cospi_14_64, cospi_18_64);
  const __m256i k__cospi_m02_p30 = PAIR256_SET_EPI16(-cospi_2_64, cospi_30_64);
  const __m256i k__cospi_m18_p14 = PAIR256_SET_EPI16(-cospi_18_64, cospi_14_64);
  const __m256i k__cospi_p22_p10 = PAIR256_SET_EPI16(cospi_22_64, cospi_10_64);
  const __m256i k__cospi_p06_p26 = PAIR256_SET_EPI16(cospi_6_64, cospi_26_64);
  const __m256i k__cospi_m10_p22 = PAIR256_SET_EPI16(-cospi_10_64, cospi_22_64);
  const __m256i k__cospi_m26_p06 = PAIR256_SET_EPI16(-cospi_26_64, cospi_6_64);
  const __m256i k__DCT_CONST_ROUNDING = _mm256_set1_epi32(DCT_CONST_ROUNDING);

  // Calculate input for the first 8 results.
  for (i = 0; i < 8; i++) {
    in[i] = ADD256_EPI16(input[i], input[15 - i]);
  }

  // Calculate input for the next 8 results.
  for (i = 0; i < 8; i++) {
    step1[i] = SUB256_EPI16(input[7 - i], input[8 + i]);
  }

  // Work on the first eight values; fdct8(input, even_results);
  {
    // Add/subtract
    const __m256i q0 = ADD256_EPI16(in[0], in[7]);
    const __m256i q1 = ADD256_EPI16(in[1], in[6]);
    const __m256i q2 = ADD256_EPI16(in[2], in[5]);
    const __m256i q3 = ADD256_EPI16(in[3], in[4]);
    const __m256i q4 = SUB256_EPI16(in[3], in[4]);
    const __m256i q5 = SUB256_EPI16(in[2], in[5]);
    const __m256i q6 = SUB256_EPI16(in[1], in[6]);
    const __m256i q7 = SUB256_EPI16(in[0], in[7]);

    // Work on first four results
    {
      // Add/subtract
      const __m256i r0 = ADD256_EPI16(q0, q3);
      const __m256i r1 = ADD256_EPI16(q1, q2);
      const __m256i r2 = SUB256_EPI16(q1, q2);
      const __m256i r3 = SUB256_EPI16(q0, q3);

      // Interleave to do the multiply by constants which gets us
      // into 32 bits.
      {
        const __m256i t0 = _mm256_unpacklo_epi16(r0, r1);
        const __m256i t1 = _mm256_unpackhi_epi16(r0, r1);
        const __m256i t2 = _mm256_unpacklo_epi16(r2, r3);
        const __m256i t3 = _mm256_unpackhi_epi16(r2, r3);

        output[0] = mult256_round_shift(&t0, &t1, &k__cospi_p16_p16,
                                        &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
        output[8] = mult256_round_shift(&t0, &t1, &k__cospi_p16_m16,
                                        &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
        output[4] = mult256_round_shift(&t2, &t3, &k__cospi_p24_p08,
                                        &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
        output[12] =
            mult256_round_shift(&t2, &t3, &k__cospi_m08_p24,
                                &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      }
    }

    // Work on next four results
    {
      // Interleave to do the multiply by constants which gets us
      // into 32 bits.
      const __m256i d0 = _mm256_unpacklo_epi16(q6, q5);
      const __m256i d1 = _mm256_unpackhi_epi16(q6, q5);
      const __m256i r0 = mult256_round_shift(
          &d0, &d1, &k__cospi_p16_m16, &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      const __m256i r1 = mult256_round_shift(
          &d0, &d1, &k__cospi_p16_p16, &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);

      {
        // Add/subtract
        const __m256i x0 = ADD256_EPI16(q4, r0);
        const __m256i x1 = SUB256_EPI16(q4, r0);
        const __m256i x2 = SUB256_EPI16(q7, r1);
        const __m256i x3 = ADD256_EPI16(q7, r1);

        // Interleave to do the multiply by constants which gets us
        // into 32 bits.
        {
          const __m256i t0 = _mm256_unpacklo_epi16(x0, x3);
          const __m256i t1 = _mm256_unpackhi_epi16(x0, x3);
          const __m256i t2 = _mm256_unpacklo_epi16(x1, x2);
          const __m256i t3 = _mm256_unpackhi_epi16(x1, x2);
          output[2] =
              mult256_round_shift(&t0, &t1, &k__cospi_p28_p04,
                                  &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          output[14] =
              mult256_round_shift(&t0, &t1, &k__cospi_m04_p28,
                                  &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          output[10] =
              mult256_round_shift(&t2, &t3, &k__cospi_p12_p20,
                                  &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
          output[6] =
              mult256_round_shift(&t2, &t3, &k__cospi_m20_p12,
                                  &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
        }
      }
    }
  }
  // Work on the next eight values; step1 -> odd_results
  {  // step 2
    {
      const __m256i t0 = _mm256_unpacklo_epi16(step1[5], step1[2]);
      const __m256i t1 = _mm256_unpackhi_epi16(step1[5], step1[2]);
      const __m256i t2 = _mm256_unpacklo_epi16(step1[4], step1[3]);
      const __m256i t3 = _mm256_unpackhi_epi16(step1[4], step1[3]);
      step2[0] = mult256_round_shift(&t0, &t1, &k__cospi_p16_m16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      step2[1] = mult256_round_shift(&t2, &t3, &k__cospi_p16_m16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      step2[2] = mult256_round_shift(&t0, &t1, &k__cospi_p16_p16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      step2[3] = mult256_round_shift(&t2, &t3, &k__cospi_p16_p16,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
    }
    // step 3
    {
      step3[0] = ADD256_EPI16(step1[0], step2[1]);
      step3[1] = ADD256_EPI16(step1[1], step2[0]);
      step3[2] = SUB256_EPI16(step1[1], step2[0]);
      step3[3] = SUB256_EPI16(step1[0], step2[1]);
      step3[4] = SUB256_EPI16(step1[7], step2[3]);
      step3[5] = SUB256_EPI16(step1[6], step2[2]);
      step3[6] = ADD256_EPI16(step1[6], step2[2]);
      step3[7] = ADD256_EPI16(step1[7], step2[3]);
    }
    // step 4
    {
      const __m256i t0 = _mm256_unpacklo_epi16(step3[1], step3[6]);
      const __m256i t1 = _mm256_unpackhi_epi16(step3[1], step3[6]);
      const __m256i t2 = _mm256_unpacklo_epi16(step3[2], step3[5]);
      const __m256i t3 = _mm256_unpackhi_epi16(step3[2], step3[5]);
      step2[0] = mult256_round_shift(&t0, &t1, &k__cospi_m08_p24,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      step2[1] = mult256_round_shift(&t2, &t3, &k__cospi_p24_p08,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      step2[2] = mult256_round_shift(&t0, &t1, &k__cospi_p24_p08,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      step2[3] = mult256_round_shift(&t2, &t3, &k__cospi_p08_m24,
                                     &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
    }
    // step 5
    {
      step1[0] = ADD256_EPI16(step3[0], step2[0]);
      step1[1] = SUB256_EPI16(step3[0], step2[0]);
      step1[2] = ADD256_EPI16(step3[3], step2[1]);
      step1[3] = SUB256_EPI16(step3[3], step2[1]);
      step1[4] = SUB256_EPI16(step3[4], step2[3]);
      step1[5] = ADD256_EPI16(step3[4], step2[3]);
      step1[6] = SUB256_EPI16(step3[7], step2[2]);
      step1[7] = ADD256_EPI16(step3[7], step2[2]);
    }
    // step 6
    {
      const __m256i t0 = _mm256_unpacklo_epi16(step1[0], step1[7]);
      const __m256i t1 = _mm256_unpackhi_epi16(step1[0], step1[7]);
      const __m256i t2 = _mm256_unpacklo_epi16(step1[1], step1[6]);
      const __m256i t3 = _mm256_unpackhi_epi16(step1[1], step1[6]);
      output[1] = mult256_round_shift(&t0, &t1, &k__cospi_p30_p02,
                                      &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      output[9] = mult256_round_shift(&t2, &t3, &k__cospi_p14_p18,
                                      &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      output[15] = mult256_round_shift(&t0, &t1, &k__cospi_m02_p30,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      output[7] = mult256_round_shift(&t2, &t3, &k__cospi_m18_p14,
                                      &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
    }
    {
      const __m256i t0 = _mm256_unpacklo_epi16(step1[2], step1[5]);
      const __m256i t1 = _mm256_unpackhi_epi16(step1[2], step1[5]);
      const __m256i t2 = _mm256_unpacklo_epi16(step1[3], step1[4]);
      const __m256i t3 = _mm256_unpackhi_epi16(step1[3], step1[4]);
      output[5] = mult256_round_shift(&t0, &t1, &k__cospi_p22_p10,
                                      &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      output[13] = mult256_round_shift(&t2, &t3, &k__cospi_p06_p26,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      output[11] = mult256_round_shift(&t0, &t1, &k__cospi_m10_p22,
                                       &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
      output[3] = mult256_round_shift(&t2, &t3, &k__cospi_m26_p06,
                                      &k__DCT_CONST_ROUNDING, DCT_CONST_BITS);
    }
  }
}

void vpx_fdct16x16_avx2(const int16_t *input, tran_low_t *output, int stride) {
  int pass;
  DECLARE_ALIGNED(32, int16_t, intermediate[256]);
  int16_t *out0 = intermediate;
  tran_low_t *out1 = output;
  const int width = 16;
  const int height = 16;
  __m256i buf0[16], buf1[16];

  // Two transform and transpose passes
  // Process 16 columns (transposed rows in second pass) at a time.
  for (pass = 0; pass < 2; ++pass) {
    // Load and pre-condition input.
    load_buffer_16bit_to_16bit_avx2(input, stride, buf1, height, pass);

    // Calculate dct for 16x16 values
    fdct16x16_1D_avx2(buf1, buf0);

    // Transpose the results.
    transpose_16bit_16x16_avx2(buf0, buf1);

    if (pass == 0) {
      store_buffer_16bit_to_32bit_w16_avx2(buf1, (tran_low_t *)out0, width,
                                           height);
    } else {
      store_buffer_16bit_to_32bit_w16_avx2(buf1, out1, width, height);
    }
    // Setup in/out for next pass.
    input = intermediate;
  }
}

#if !CONFIG_VP9_HIGHBITDEPTH
#define FDCT32x32_2D_AVX2 vpx_fdct32x32_rd_avx2
#define FDCT32x32_HIGH_PRECISION 0
#include "vpx_dsp/x86/fwd_dct32x32_impl_avx2.h"
#undef FDCT32x32_2D_AVX2
#undef FDCT32x32_HIGH_PRECISION

#define FDCT32x32_2D_AVX2 vpx_fdct32x32_avx2
#define FDCT32x32_HIGH_PRECISION 1
#include "vpx_dsp/x86/fwd_dct32x32_impl_avx2.h"  // NOLINT
#undef FDCT32x32_2D_AVX2
#undef FDCT32x32_HIGH_PRECISION
#endif  // !CONFIG_VP9_HIGHBITDEPTH
