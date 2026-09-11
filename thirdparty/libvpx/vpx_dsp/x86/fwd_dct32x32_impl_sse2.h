/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <emmintrin.h>  // SSE2

#include "vpx_dsp/fwd_txfm.h"
#include "vpx_dsp/txfm_common.h"
#include "vpx_dsp/x86/txfm_common_sse2.h"

// TODO(jingning) The high bit-depth version needs re-work for performance.
// The current SSE2 implementation also causes cross reference to the static
// functions in the C implementation file.
#if DCT_HIGH_BIT_DEPTH
#define ADD_EPI16 _mm_adds_epi16
#define SUB_EPI16 _mm_subs_epi16
#if FDCT32x32_HIGH_PRECISION
static void vpx_fdct32x32_rows_c(const int16_t *intermediate, tran_low_t *out) {
  int i, j;
  for (i = 0; i < 32; ++i) {
    tran_high_t temp_in[32], temp_out[32];
    for (j = 0; j < 32; ++j) temp_in[j] = intermediate[j * 32 + i];
    vpx_fdct32(temp_in, temp_out, 0);
    for (j = 0; j < 32; ++j)
      out[j + i * 32] =
          (tran_low_t)((temp_out[j] + 1 + (temp_out[j] < 0)) >> 2);
  }
}
#define HIGH_FDCT32x32_2D_C vpx_highbd_fdct32x32_c
#define HIGH_FDCT32x32_2D_ROWS_C vpx_fdct32x32_rows_c
#else
static void vpx_fdct32x32_rd_rows_c(const int16_t *intermediate,
                                    tran_low_t *out) {
  int i, j;
  for (i = 0; i < 32; ++i) {
    tran_high_t temp_in[32], temp_out[32];
    for (j = 0; j < 32; ++j) temp_in[j] = intermediate[j * 32 + i];
    vpx_fdct32(temp_in, temp_out, 1);
    for (j = 0; j < 32; ++j) out[j + i * 32] = (tran_low_t)temp_out[j];
  }
}
#define HIGH_FDCT32x32_2D_C vpx_highbd_fdct32x32_rd_c
#define HIGH_FDCT32x32_2D_ROWS_C vpx_fdct32x32_rd_rows_c
#endif  // FDCT32x32_HIGH_PRECISION
#else
#define ADD_EPI16 _mm_add_epi16
#define SUB_EPI16 _mm_sub_epi16
#endif  // DCT_HIGH_BIT_DEPTH

void FDCT32x32_2D(const int16_t *input, tran_low_t *output_org, int stride) {
  // Calculate pre-multiplied strides
  const int str1 = stride;
  const int str2 = 2 * stride;
  const int str3 = 2 * stride + str1;
  // We need an intermediate buffer between passes.
  DECLARE_ALIGNED(16, int16_t, intermediate[32 * 32]);
  // Constants
  //    When we use them, in one case, they are all the same. In all others
  //    it's a pair of them that we need to repeat four times. This is done
  //    by constructing the 32 bit constant corresponding to that pair.
  const __m128i k__cospi_p16_p16 = _mm_set1_epi16(cospi_16_64);
  const __m128i k__cospi_p16_m16 = pair_set_epi16(+cospi_16_64, -cospi_16_64);
  const __m128i k__cospi_m08_p24 = pair_set_epi16(-cospi_8_64, cospi_24_64);
  const __m128i k__cospi_m24_m08 = pair_set_epi16(-cospi_24_64, -cospi_8_64);
  const __m128i k__cospi_p24_p08 = pair_set_epi16(+cospi_24_64, cospi_8_64);
  const __m128i k__cospi_p12_p20 = pair_set_epi16(+cospi_12_64, cospi_20_64);
  const __m128i k__cospi_m20_p12 = pair_set_epi16(-cospi_20_64, cospi_12_64);
  const __m128i k__cospi_m04_p28 = pair_set_epi16(-cospi_4_64, cospi_28_64);
  const __m128i k__cospi_p28_p04 = pair_set_epi16(+cospi_28_64, cospi_4_64);
  const __m128i k__cospi_m28_m04 = pair_set_epi16(-cospi_28_64, -cospi_4_64);
  const __m128i k__cospi_m12_m20 = pair_set_epi16(-cospi_12_64, -cospi_20_64);
  const __m128i k__cospi_p30_p02 = pair_set_epi16(+cospi_30_64, cospi_2_64);
  const __m128i k__cospi_p14_p18 = pair_set_epi16(+cospi_14_64, cospi_18_64);
  const __m128i k__cospi_p22_p10 = pair_set_epi16(+cospi_22_64, cospi_10_64);
  const __m128i k__cospi_p06_p26 = pair_set_epi16(+cospi_6_64, cospi_26_64);
  const __m128i k__cospi_m26_p06 = pair_set_epi16(-cospi_26_64, cospi_6_64);
  const __m128i k__cospi_m10_p22 = pair_set_epi16(-cospi_10_64, cospi_22_64);
  const __m128i k__cospi_m18_p14 = pair_set_epi16(-cospi_18_64, cospi_14_64);
  const __m128i k__cospi_m02_p30 = pair_set_epi16(-cospi_2_64, cospi_30_64);
  const __m128i k__cospi_p31_p01 = pair_set_epi16(+cospi_31_64, cospi_1_64);
  const __m128i k__cospi_p15_p17 = pair_set_epi16(+cospi_15_64, cospi_17_64);
  const __m128i k__cospi_p23_p09 = pair_set_epi16(+cospi_23_64, cospi_9_64);
  const __m128i k__cospi_p07_p25 = pair_set_epi16(+cospi_7_64, cospi_25_64);
  const __m128i k__cospi_m25_p07 = pair_set_epi16(-cospi_25_64, cospi_7_64);
  const __m128i k__cospi_m09_p23 = pair_set_epi16(-cospi_9_64, cospi_23_64);
  const __m128i k__cospi_m17_p15 = pair_set_epi16(-cospi_17_64, cospi_15_64);
  const __m128i k__cospi_m01_p31 = pair_set_epi16(-cospi_1_64, cospi_31_64);
  const __m128i k__cospi_p27_p05 = pair_set_epi16(+cospi_27_64, cospi_5_64);
  const __m128i k__cospi_p11_p21 = pair_set_epi16(+cospi_11_64, cospi_21_64);
  const __m128i k__cospi_p19_p13 = pair_set_epi16(+cospi_19_64, cospi_13_64);
  const __m128i k__cospi_p03_p29 = pair_set_epi16(+cospi_3_64, cospi_29_64);
  const __m128i k__cospi_m29_p03 = pair_set_epi16(-cospi_29_64, cospi_3_64);
  const __m128i k__cospi_m13_p19 = pair_set_epi16(-cospi_13_64, cospi_19_64);
  const __m128i k__cospi_m21_p11 = pair_set_epi16(-cospi_21_64, cospi_11_64);
  const __m128i k__cospi_m05_p27 = pair_set_epi16(-cospi_5_64, cospi_27_64);
  const __m128i k__DCT_CONST_ROUNDING = _mm_set1_epi32(DCT_CONST_ROUNDING);
  const __m128i kZero = _mm_setzero_si128();
  const __m128i kOne = _mm_set1_epi16(1);

  // Do the two transform/transpose passes
  int pass;
#if DCT_HIGH_BIT_DEPTH
  int overflow;
#endif
  for (pass = 0; pass < 2; ++pass) {
    // We process eight columns (transposed rows in second pass) at a time.
    int column_start;
    for (column_start = 0; column_start < 32; column_start += 8) {
      __m128i step1[32];
      __m128i step2[32];
      __m128i step3[32];
      __m128i out[32];
      // Stage 1
      // Note: even though all the loads below are aligned, using the aligned
      //       intrinsic make the code slightly slower.
      if (0 == pass) {
        const int16_t *in = &input[column_start];
        // step1[i] =  (in[ 0 * stride] + in[(32 -  1) * stride]) << 2;
        // Note: the next four blocks could be in a loop. That would help the
        //       instruction cache but is actually slower.
        {
          const int16_t *ina = in + 0 * str1;
          const int16_t *inb = in + 31 * str1;
          __m128i *step1a = &step1[0];
          __m128i *step1b = &step1[31];
          const __m128i ina0 = _mm_loadu_si128((const __m128i *)(ina));
          const __m128i ina1 = _mm_loadu_si128((const __m128i *)(ina + str1));
          const __m128i ina2 = _mm_loadu_si128((const __m128i *)(ina + str2));
          const __m128i ina3 = _mm_loadu_si128((const __m128i *)(ina + str3));
          const __m128i inb3 = _mm_loadu_si128((const __m128i *)(inb - str3));
          const __m128i inb2 = _mm_loadu_si128((const __m128i *)(inb - str2));
          const __m128i inb1 = _mm_loadu_si128((const __m128i *)(inb - str1));
          const __m128i inb0 = _mm_loadu_si128((const __m128i *)(inb));
          step1a[0] = _mm_add_epi16(ina0, inb0);
          step1a[1] = _mm_add_epi16(ina1, inb1);
          step1a[2] = _mm_add_epi16(ina2, inb2);
          step1a[3] = _mm_add_epi16(ina3, inb3);
          step1b[-3] = _mm_sub_epi16(ina3, inb3);
          step1b[-2] = _mm_sub_epi16(ina2, inb2);
          step1b[-1] = _mm_sub_epi16(ina1, inb1);
          step1b[-0] = _mm_sub_epi16(ina0, inb0);
          step1a[0] = _mm_slli_epi16(step1a[0], 2);
          step1a[1] = _mm_slli_epi16(step1a[1], 2);
          step1a[2] = _mm_slli_epi16(step1a[2], 2);
          step1a[3] = _mm_slli_epi16(step1a[3], 2);
          step1b[-3] = _mm_slli_epi16(step1b[-3], 2);
          step1b[-2] = _mm_slli_epi16(step1b[-2], 2);
          step1b[-1] = _mm_slli_epi16(step1b[-1], 2);
          step1b[-0] = _mm_slli_epi16(step1b[-0], 2);
        }
        {
          const int16_t *ina = in + 4 * str1;
          const int16_t *inb = in + 27 * str1;
          __m128i *step1a = &step1[4];
          __m128i *step1b = &step1[27];
          const __m128i ina0 = _mm_loadu_si128((const __m128i *)(ina));
          const __m128i ina1 = _mm_loadu_si128((const __m128i *)(ina + str1));
          const __m128i ina2 = _mm_loadu_si128((const __m128i *)(ina + str2));
          const __m128i ina3 = _mm_loadu_si128((const __m128i *)(ina + str3));
          const __m128i inb3 = _mm_loadu_si128((const __m128i *)(inb - str3));
          const __m128i inb2 = _mm_loadu_si128((const __m128i *)(inb - str2));
          const __m128i inb1 = _mm_loadu_si128((const __m128i *)(inb - str1));
          const __m128i inb0 = _mm_loadu_si128((const __m128i *)(inb));
          step1a[0] = _mm_add_epi16(ina0, inb0);
          step1a[1] = _mm_add_epi16(ina1, inb1);
          step1a[2] = _mm_add_epi16(ina2, inb2);
          step1a[3] = _mm_add_epi16(ina3, inb3);
          step1b[-3] = _mm_sub_epi16(ina3, inb3);
          step1b[-2] = _mm_sub_epi16(ina2, inb2);
          step1b[-1] = _mm_sub_epi16(ina1, inb1);
          step1b[-0] = _mm_sub_epi16(ina0, inb0);
          step1a[0] = _mm_slli_epi16(step1a[0], 2);
          step1a[1] = _mm_slli_epi16(step1a[1], 2);
          step1a[2] = _mm_slli_epi16(step1a[2], 2);
          step1a[3] = _mm_slli_epi16(step1a[3], 2);
          step1b[-3] = _mm_slli_epi16(step1b[-3], 2);
          step1b[-2] = _mm_slli_epi16(step1b[-2], 2);
          step1b[-1] = _mm_slli_epi16(step1b[-1], 2);
          step1b[-0] = _mm_slli_epi16(step1b[-0], 2);
        }
        {
          const int16_t *ina = in + 8 * str1;
          const int16_t *inb = in + 23 * str1;
          __m128i *step1a = &step1[8];
          __m128i *step1b = &step1[23];
          const __m128i ina0 = _mm_loadu_si128((const __m128i *)(ina));
          const __m128i ina1 = _mm_loadu_si128((const __m128i *)(ina + str1));
          const __m128i ina2 = _mm_loadu_si128((const __m128i *)(ina + str2));
          const __m128i ina3 = _mm_loadu_si128((const __m128i *)(ina + str3));
          const __m128i inb3 = _mm_loadu_si128((const __m128i *)(inb - str3));
          const __m128i inb2 = _mm_loadu_si128((const __m128i *)(inb - str2));
          const __m128i inb1 = _mm_loadu_si128((const __m128i *)(inb - str1));
          const __m128i inb0 = _mm_loadu_si128((const __m128i *)(inb));
          step1a[0] = _mm_add_epi16(ina0, inb0);
          step1a[1] = _mm_add_epi16(ina1, inb1);
          step1a[2] = _mm_add_epi16(ina2, inb2);
          step1a[3] = _mm_add_epi16(ina3, inb3);
          step1b[-3] = _mm_sub_epi16(ina3, inb3);
          step1b[-2] = _mm_sub_epi16(ina2, inb2);
          step1b[-1] = _mm_sub_epi16(ina1, inb1);
          step1b[-0] = _mm_sub_epi16(ina0, inb0);
          step1a[0] = _mm_slli_epi16(step1a[0], 2);
          step1a[1] = _mm_slli_epi16(step1a[1], 2);
          step1a[2] = _mm_slli_epi16(step1a[2], 2);
          step1a[3] = _mm_slli_epi16(step1a[3], 2);
          step1b[-3] = _mm_slli_epi16(step1b[-3], 2);
          step1b[-2] = _mm_slli_epi16(step1b[-2], 2);
          step1b[-1] = _mm_slli_epi16(step1b[-1], 2);
          step1b[-0] = _mm_slli_epi16(step1b[-0], 2);
        }
        {
          const int16_t *ina = in + 12 * str1;
          const int16_t *inb = in + 19 * str1;
          __m128i *step1a = &step1[12];
          __m128i *step1b = &step1[19];
          const __m128i ina0 = _mm_loadu_si128((const __m128i *)(ina));
          const __m128i ina1 = _mm_loadu_si128((const __m128i *)(ina + str1));
          const __m128i ina2 = _mm_loadu_si128((const __m128i *)(ina + str2));
          const __m128i ina3 = _mm_loadu_si128((const __m128i *)(ina + str3));
          const __m128i inb3 = _mm_loadu_si128((const __m128i *)(inb - str3));
          const __m128i inb2 = _mm_loadu_si128((const __m128i *)(inb - str2));
          const __m128i inb1 = _mm_loadu_si128((const __m128i *)(inb - str1));
          const __m128i inb0 = _mm_loadu_si128((const __m128i *)(inb));
          step1a[0] = _mm_add_epi16(ina0, inb0);
          step1a[1] = _mm_add_epi16(ina1, inb1);
          step1a[2] = _mm_add_epi16(ina2, inb2);
          step1a[3] = _mm_add_epi16(ina3, inb3);
          step1b[-3] = _mm_sub_epi16(ina3, inb3);
          step1b[-2] = _mm_sub_epi16(ina2, inb2);
          step1b[-1] = _mm_sub_epi16(ina1, inb1);
          step1b[-0] = _mm_sub_epi16(ina0, inb0);
          step1a[0] = _mm_slli_epi16(step1a[0], 2);
          step1a[1] = _mm_slli_epi16(step1a[1], 2);
          step1a[2] = _mm_slli_epi16(step1a[2], 2);
          step1a[3] = _mm_slli_epi16(step1a[3], 2);
          step1b[-3] = _mm_slli_epi16(step1b[-3], 2);
          step1b[-2] = _mm_slli_epi16(step1b[-2], 2);
          step1b[-1] = _mm_slli_epi16(step1b[-1], 2);
          step1b[-0] = _mm_slli_epi16(step1b[-0], 2);
        }
      } else {
        int16_t *in = &intermediate[column_start];
        // step1[i] =  in[ 0 * 32] + in[(32 -  1) * 32];
        // Note: using the same approach as above to have common offset is
        //       counter-productive as all offsets can be calculated at compile
        //       time.
        // Note: the next four blocks could be in a loop. That would help the
        //       instruction cache but is actually slower.
        {
          __m128i in00 = _mm_loadu_si128((const __m128i *)(in + 0 * 32));
          __m128i in01 = _mm_loadu_si128((const __m128i *)(in + 1 * 32));
          __m128i in02 = _mm_loadu_si128((const __m128i *)(in + 2 * 32));
          __m128i in03 = _mm_loadu_si128((const __m128i *)(in + 3 * 32));
          __m128i in28 = _mm_loadu_si128((const __m128i *)(in + 28 * 32));
          __m128i in29 = _mm_loadu_si128((const __m128i *)(in + 29 * 32));
          __m128i in30 = _mm_loadu_si128((const __m128i *)(in + 30 * 32));
          __m128i in31 = _mm_loadu_si128((const __m128i *)(in + 31 * 32));
          step1[0] = ADD_EPI16(in00, in31);
          step1[1] = ADD_EPI16(in01, in30);
          step1[2] = ADD_EPI16(in02, in29);
          step1[3] = ADD_EPI16(in03, in28);
          step1[28] = SUB_EPI16(in03, in28);
          step1[29] = SUB_EPI16(in02, in29);
          step1[30] = SUB_EPI16(in01, in30);
          step1[31] = SUB_EPI16(in00, in31);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x8(&step1[0], &step1[1], &step1[2],
                                             &step1[3], &step1[28], &step1[29],
                                             &step1[30], &step1[31]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          __m128i in04 = _mm_loadu_si128((const __m128i *)(in + 4 * 32));
          __m128i in05 = _mm_loadu_si128((const __m128i *)(in + 5 * 32));
          __m128i in06 = _mm_loadu_si128((const __m128i *)(in + 6 * 32));
          __m128i in07 = _mm_loadu_si128((const __m128i *)(in + 7 * 32));
          __m128i in24 = _mm_loadu_si128((const __m128i *)(in + 24 * 32));
          __m128i in25 = _mm_loadu_si128((const __m128i *)(in + 25 * 32));
          __m128i in26 = _mm_loadu_si128((const __m128i *)(in + 26 * 32));
          __m128i in27 = _mm_loadu_si128((const __m128i *)(in + 27 * 32));
          step1[4] = ADD_EPI16(in04, in27);
          step1[5] = ADD_EPI16(in05, in26);
          step1[6] = ADD_EPI16(in06, in25);
          step1[7] = ADD_EPI16(in07, in24);
          step1[24] = SUB_EPI16(in07, in24);
          step1[25] = SUB_EPI16(in06, in25);
          step1[26] = SUB_EPI16(in05, in26);
          step1[27] = SUB_EPI16(in04, in27);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x8(&step1[4], &step1[5], &step1[6],
                                             &step1[7], &step1[24], &step1[25],
                                             &step1[26], &step1[27]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          __m128i in08 = _mm_loadu_si128((const __m128i *)(in + 8 * 32));
          __m128i in09 = _mm_loadu_si128((const __m128i *)(in + 9 * 32));
          __m128i in10 = _mm_loadu_si128((const __m128i *)(in + 10 * 32));
          __m128i in11 = _mm_loadu_si128((const __m128i *)(in + 11 * 32));
          __m128i in20 = _mm_loadu_si128((const __m128i *)(in + 20 * 32));
          __m128i in21 = _mm_loadu_si128((const __m128i *)(in + 21 * 32));
          __m128i in22 = _mm_loadu_si128((const __m128i *)(in + 22 * 32));
          __m128i in23 = _mm_loadu_si128((const __m128i *)(in + 23 * 32));
          step1[8] = ADD_EPI16(in08, in23);
          step1[9] = ADD_EPI16(in09, in22);
          step1[10] = ADD_EPI16(in10, in21);
          step1[11] = ADD_EPI16(in11, in20);
          step1[20] = SUB_EPI16(in11, in20);
          step1[21] = SUB_EPI16(in10, in21);
          step1[22] = SUB_EPI16(in09, in22);
          step1[23] = SUB_EPI16(in08, in23);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x8(&step1[8], &step1[9], &step1[10],
                                             &step1[11], &step1[20], &step1[21],
                                             &step1[22], &step1[23]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          __m128i in12 = _mm_loadu_si128((const __m128i *)(in + 12 * 32));
          __m128i in13 = _mm_loadu_si128((const __m128i *)(in + 13 * 32));
          __m128i in14 = _mm_loadu_si128((const __m128i *)(in + 14 * 32));
          __m128i in15 = _mm_loadu_si128((const __m128i *)(in + 15 * 32));
          __m128i in16 = _mm_loadu_si128((const __m128i *)(in + 16 * 32));
          __m128i in17 = _mm_loadu_si128((const __m128i *)(in + 17 * 32));
          __m128i in18 = _mm_loadu_si128((const __m128i *)(in + 18 * 32));
          __m128i in19 = _mm_loadu_si128((const __m128i *)(in + 19 * 32));
          step1[12] = ADD_EPI16(in12, in19);
          step1[13] = ADD_EPI16(in13, in18);
          step1[14] = ADD_EPI16(in14, in17);
          step1[15] = ADD_EPI16(in15, in16);
          step1[16] = SUB_EPI16(in15, in16);
          step1[17] = SUB_EPI16(in14, in17);
          step1[18] = SUB_EPI16(in13, in18);
          step1[19] = SUB_EPI16(in12, in19);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x8(&step1[12], &step1[13], &step1[14],
                                             &step1[15], &step1[16], &step1[17],
                                             &step1[18], &step1[19]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
      }
      // Stage 2
      {
        step2[0] = ADD_EPI16(step1[0], step1[15]);
        step2[1] = ADD_EPI16(step1[1], step1[14]);
        step2[2] = ADD_EPI16(step1[2], step1[13]);
        step2[3] = ADD_EPI16(step1[3], step1[12]);
        step2[4] = ADD_EPI16(step1[4], step1[11]);
        step2[5] = ADD_EPI16(step1[5], step1[10]);
        step2[6] = ADD_EPI16(step1[6], step1[9]);
        step2[7] = ADD_EPI16(step1[7], step1[8]);
        step2[8] = SUB_EPI16(step1[7], step1[8]);
        step2[9] = SUB_EPI16(step1[6], step1[9]);
        step2[10] = SUB_EPI16(step1[5], step1[10]);
        step2[11] = SUB_EPI16(step1[4], step1[11]);
        step2[12] = SUB_EPI16(step1[3], step1[12]);
        step2[13] = SUB_EPI16(step1[2], step1[13]);
        step2[14] = SUB_EPI16(step1[1], step1[14]);
        step2[15] = SUB_EPI16(step1[0], step1[15]);
#if DCT_HIGH_BIT_DEPTH
        overflow = check_epi16_overflow_x16(
            &step2[0], &step2[1], &step2[2], &step2[3], &step2[4], &step2[5],
            &step2[6], &step2[7], &step2[8], &step2[9], &step2[10], &step2[11],
            &step2[12], &step2[13], &step2[14], &step2[15]);
        if (overflow) {
          if (pass == 0)
            HIGH_FDCT32x32_2D_C(input, output_org, stride);
          else
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
      }
      {
        const __m128i s2_20_0 = _mm_unpacklo_epi16(step1[27], step1[20]);
        const __m128i s2_20_1 = _mm_unpackhi_epi16(step1[27], step1[20]);
        const __m128i s2_21_0 = _mm_unpacklo_epi16(step1[26], step1[21]);
        const __m128i s2_21_1 = _mm_unpackhi_epi16(step1[26], step1[21]);
        const __m128i s2_22_0 = _mm_unpacklo_epi16(step1[25], step1[22]);
        const __m128i s2_22_1 = _mm_unpackhi_epi16(step1[25], step1[22]);
        const __m128i s2_23_0 = _mm_unpacklo_epi16(step1[24], step1[23]);
        const __m128i s2_23_1 = _mm_unpackhi_epi16(step1[24], step1[23]);
        const __m128i s2_20_2 = _mm_madd_epi16(s2_20_0, k__cospi_p16_m16);
        const __m128i s2_20_3 = _mm_madd_epi16(s2_20_1, k__cospi_p16_m16);
        const __m128i s2_21_2 = _mm_madd_epi16(s2_21_0, k__cospi_p16_m16);
        const __m128i s2_21_3 = _mm_madd_epi16(s2_21_1, k__cospi_p16_m16);
        const __m128i s2_22_2 = _mm_madd_epi16(s2_22_0, k__cospi_p16_m16);
        const __m128i s2_22_3 = _mm_madd_epi16(s2_22_1, k__cospi_p16_m16);
        const __m128i s2_23_2 = _mm_madd_epi16(s2_23_0, k__cospi_p16_m16);
        const __m128i s2_23_3 = _mm_madd_epi16(s2_23_1, k__cospi_p16_m16);
        const __m128i s2_24_2 = _mm_madd_epi16(s2_23_0, k__cospi_p16_p16);
        const __m128i s2_24_3 = _mm_madd_epi16(s2_23_1, k__cospi_p16_p16);
        const __m128i s2_25_2 = _mm_madd_epi16(s2_22_0, k__cospi_p16_p16);
        const __m128i s2_25_3 = _mm_madd_epi16(s2_22_1, k__cospi_p16_p16);
        const __m128i s2_26_2 = _mm_madd_epi16(s2_21_0, k__cospi_p16_p16);
        const __m128i s2_26_3 = _mm_madd_epi16(s2_21_1, k__cospi_p16_p16);
        const __m128i s2_27_2 = _mm_madd_epi16(s2_20_0, k__cospi_p16_p16);
        const __m128i s2_27_3 = _mm_madd_epi16(s2_20_1, k__cospi_p16_p16);
        // dct_const_round_shift
        const __m128i s2_20_4 = _mm_add_epi32(s2_20_2, k__DCT_CONST_ROUNDING);
        const __m128i s2_20_5 = _mm_add_epi32(s2_20_3, k__DCT_CONST_ROUNDING);
        const __m128i s2_21_4 = _mm_add_epi32(s2_21_2, k__DCT_CONST_ROUNDING);
        const __m128i s2_21_5 = _mm_add_epi32(s2_21_3, k__DCT_CONST_ROUNDING);
        const __m128i s2_22_4 = _mm_add_epi32(s2_22_2, k__DCT_CONST_ROUNDING);
        const __m128i s2_22_5 = _mm_add_epi32(s2_22_3, k__DCT_CONST_ROUNDING);
        const __m128i s2_23_4 = _mm_add_epi32(s2_23_2, k__DCT_CONST_ROUNDING);
        const __m128i s2_23_5 = _mm_add_epi32(s2_23_3, k__DCT_CONST_ROUNDING);
        const __m128i s2_24_4 = _mm_add_epi32(s2_24_2, k__DCT_CONST_ROUNDING);
        const __m128i s2_24_5 = _mm_add_epi32(s2_24_3, k__DCT_CONST_ROUNDING);
        const __m128i s2_25_4 = _mm_add_epi32(s2_25_2, k__DCT_CONST_ROUNDING);
        const __m128i s2_25_5 = _mm_add_epi32(s2_25_3, k__DCT_CONST_ROUNDING);
        const __m128i s2_26_4 = _mm_add_epi32(s2_26_2, k__DCT_CONST_ROUNDING);
        const __m128i s2_26_5 = _mm_add_epi32(s2_26_3, k__DCT_CONST_ROUNDING);
        const __m128i s2_27_4 = _mm_add_epi32(s2_27_2, k__DCT_CONST_ROUNDING);
        const __m128i s2_27_5 = _mm_add_epi32(s2_27_3, k__DCT_CONST_ROUNDING);
        const __m128i s2_20_6 = _mm_srai_epi32(s2_20_4, DCT_CONST_BITS);
        const __m128i s2_20_7 = _mm_srai_epi32(s2_20_5, DCT_CONST_BITS);
        const __m128i s2_21_6 = _mm_srai_epi32(s2_21_4, DCT_CONST_BITS);
        const __m128i s2_21_7 = _mm_srai_epi32(s2_21_5, DCT_CONST_BITS);
        const __m128i s2_22_6 = _mm_srai_epi32(s2_22_4, DCT_CONST_BITS);
        const __m128i s2_22_7 = _mm_srai_epi32(s2_22_5, DCT_CONST_BITS);
        const __m128i s2_23_6 = _mm_srai_epi32(s2_23_4, DCT_CONST_BITS);
        const __m128i s2_23_7 = _mm_srai_epi32(s2_23_5, DCT_CONST_BITS);
        const __m128i s2_24_6 = _mm_srai_epi32(s2_24_4, DCT_CONST_BITS);
        const __m128i s2_24_7 = _mm_srai_epi32(s2_24_5, DCT_CONST_BITS);
        const __m128i s2_25_6 = _mm_srai_epi32(s2_25_4, DCT_CONST_BITS);
        const __m128i s2_25_7 = _mm_srai_epi32(s2_25_5, DCT_CONST_BITS);
        const __m128i s2_26_6 = _mm_srai_epi32(s2_26_4, DCT_CONST_BITS);
        const __m128i s2_26_7 = _mm_srai_epi32(s2_26_5, DCT_CONST_BITS);
        const __m128i s2_27_6 = _mm_srai_epi32(s2_27_4, DCT_CONST_BITS);
        const __m128i s2_27_7 = _mm_srai_epi32(s2_27_5, DCT_CONST_BITS);
        // Combine
        step2[20] = _mm_packs_epi32(s2_20_6, s2_20_7);
        step2[21] = _mm_packs_epi32(s2_21_6, s2_21_7);
        step2[22] = _mm_packs_epi32(s2_22_6, s2_22_7);
        step2[23] = _mm_packs_epi32(s2_23_6, s2_23_7);
        step2[24] = _mm_packs_epi32(s2_24_6, s2_24_7);
        step2[25] = _mm_packs_epi32(s2_25_6, s2_25_7);
        step2[26] = _mm_packs_epi32(s2_26_6, s2_26_7);
        step2[27] = _mm_packs_epi32(s2_27_6, s2_27_7);
#if DCT_HIGH_BIT_DEPTH
        overflow = check_epi16_overflow_x8(&step2[20], &step2[21], &step2[22],
                                           &step2[23], &step2[24], &step2[25],
                                           &step2[26], &step2[27]);
        if (overflow) {
          if (pass == 0)
            HIGH_FDCT32x32_2D_C(input, output_org, stride);
          else
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
      }

#if !FDCT32x32_HIGH_PRECISION
      // dump the magnitude by half, hence the intermediate values are within
      // the range of 16 bits.
      if (1 == pass) {
        __m128i s3_00_0 = _mm_cmplt_epi16(step2[0], kZero);
        __m128i s3_01_0 = _mm_cmplt_epi16(step2[1], kZero);
        __m128i s3_02_0 = _mm_cmplt_epi16(step2[2], kZero);
        __m128i s3_03_0 = _mm_cmplt_epi16(step2[3], kZero);
        __m128i s3_04_0 = _mm_cmplt_epi16(step2[4], kZero);
        __m128i s3_05_0 = _mm_cmplt_epi16(step2[5], kZero);
        __m128i s3_06_0 = _mm_cmplt_epi16(step2[6], kZero);
        __m128i s3_07_0 = _mm_cmplt_epi16(step2[7], kZero);
        __m128i s2_08_0 = _mm_cmplt_epi16(step2[8], kZero);
        __m128i s2_09_0 = _mm_cmplt_epi16(step2[9], kZero);
        __m128i s3_10_0 = _mm_cmplt_epi16(step2[10], kZero);
        __m128i s3_11_0 = _mm_cmplt_epi16(step2[11], kZero);
        __m128i s3_12_0 = _mm_cmplt_epi16(step2[12], kZero);
        __m128i s3_13_0 = _mm_cmplt_epi16(step2[13], kZero);
        __m128i s2_14_0 = _mm_cmplt_epi16(step2[14], kZero);
        __m128i s2_15_0 = _mm_cmplt_epi16(step2[15], kZero);
        __m128i s3_16_0 = _mm_cmplt_epi16(step1[16], kZero);
        __m128i s3_17_0 = _mm_cmplt_epi16(step1[17], kZero);
        __m128i s3_18_0 = _mm_cmplt_epi16(step1[18], kZero);
        __m128i s3_19_0 = _mm_cmplt_epi16(step1[19], kZero);
        __m128i s3_20_0 = _mm_cmplt_epi16(step2[20], kZero);
        __m128i s3_21_0 = _mm_cmplt_epi16(step2[21], kZero);
        __m128i s3_22_0 = _mm_cmplt_epi16(step2[22], kZero);
        __m128i s3_23_0 = _mm_cmplt_epi16(step2[23], kZero);
        __m128i s3_24_0 = _mm_cmplt_epi16(step2[24], kZero);
        __m128i s3_25_0 = _mm_cmplt_epi16(step2[25], kZero);
        __m128i s3_26_0 = _mm_cmplt_epi16(step2[26], kZero);
        __m128i s3_27_0 = _mm_cmplt_epi16(step2[27], kZero);
        __m128i s3_28_0 = _mm_cmplt_epi16(step1[28], kZero);
        __m128i s3_29_0 = _mm_cmplt_epi16(step1[29], kZero);
        __m128i s3_30_0 = _mm_cmplt_epi16(step1[30], kZero);
        __m128i s3_31_0 = _mm_cmplt_epi16(step1[31], kZero);

        step2[0] = SUB_EPI16(step2[0], s3_00_0);
        step2[1] = SUB_EPI16(step2[1], s3_01_0);
        step2[2] = SUB_EPI16(step2[2], s3_02_0);
        step2[3] = SUB_EPI16(step2[3], s3_03_0);
        step2[4] = SUB_EPI16(step2[4], s3_04_0);
        step2[5] = SUB_EPI16(step2[5], s3_05_0);
        step2[6] = SUB_EPI16(step2[6], s3_06_0);
        step2[7] = SUB_EPI16(step2[7], s3_07_0);
        step2[8] = SUB_EPI16(step2[8], s2_08_0);
        step2[9] = SUB_EPI16(step2[9], s2_09_0);
        step2[10] = SUB_EPI16(step2[10], s3_10_0);
        step2[11] = SUB_EPI16(step2[11], s3_11_0);
        step2[12] = SUB_EPI16(step2[12], s3_12_0);
        step2[13] = SUB_EPI16(step2[13], s3_13_0);
        step2[14] = SUB_EPI16(step2[14], s2_14_0);
        step2[15] = SUB_EPI16(step2[15], s2_15_0);
        step1[16] = SUB_EPI16(step1[16], s3_16_0);
        step1[17] = SUB_EPI16(step1[17], s3_17_0);
        step1[18] = SUB_EPI16(step1[18], s3_18_0);
        step1[19] = SUB_EPI16(step1[19], s3_19_0);
        step2[20] = SUB_EPI16(step2[20], s3_20_0);
        step2[21] = SUB_EPI16(step2[21], s3_21_0);
        step2[22] = SUB_EPI16(step2[22], s3_22_0);
        step2[23] = SUB_EPI16(step2[23], s3_23_0);
        step2[24] = SUB_EPI16(step2[24], s3_24_0);
        step2[25] = SUB_EPI16(step2[25], s3_25_0);
        step2[26] = SUB_EPI16(step2[26], s3_26_0);
        step2[27] = SUB_EPI16(step2[27], s3_27_0);
        step1[28] = SUB_EPI16(step1[28], s3_28_0);
        step1[29] = SUB_EPI16(step1[29], s3_29_0);
        step1[30] = SUB_EPI16(step1[30], s3_30_0);
        step1[31] = SUB_EPI16(step1[31], s3_31_0);
#if DCT_HIGH_BIT_DEPTH
        overflow = check_epi16_overflow_x32(
            &step2[0], &step2[1], &step2[2], &step2[3], &step2[4], &step2[5],
            &step2[6], &step2[7], &step2[8], &step2[9], &step2[10], &step2[11],
            &step2[12], &step2[13], &step2[14], &step2[15], &step1[16],
            &step1[17], &step1[18], &step1[19], &step2[20], &step2[21],
            &step2[22], &step2[23], &step2[24], &step2[25], &step2[26],
            &step2[27], &step1[28], &step1[29], &step1[30], &step1[31]);
        if (overflow) {
          HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
          return;
        }
#endif  // DCT_HIGH_BIT_DEPTH
        step2[0] = _mm_add_epi16(step2[0], kOne);
        step2[1] = _mm_add_epi16(step2[1], kOne);
        step2[2] = _mm_add_epi16(step2[2], kOne);
        step2[3] = _mm_add_epi16(step2[3], kOne);
        step2[4] = _mm_add_epi16(step2[4], kOne);
        step2[5] = _mm_add_epi16(step2[5], kOne);
        step2[6] = _mm_add_epi16(step2[6], kOne);
        step2[7] = _mm_add_epi16(step2[7], kOne);
        step2[8] = _mm_add_epi16(step2[8], kOne);
        step2[9] = _mm_add_epi16(step2[9], kOne);
        step2[10] = _mm_add_epi16(step2[10], kOne);
        step2[11] = _mm_add_epi16(step2[11], kOne);
        step2[12] = _mm_add_epi16(step2[12], kOne);
        step2[13] = _mm_add_epi16(step2[13], kOne);
        step2[14] = _mm_add_epi16(step2[14], kOne);
        step2[15] = _mm_add_epi16(step2[15], kOne);
        step1[16] = _mm_add_epi16(step1[16], kOne);
        step1[17] = _mm_add_epi16(step1[17], kOne);
        step1[18] = _mm_add_epi16(step1[18], kOne);
        step1[19] = _mm_add_epi16(step1[19], kOne);
        step2[20] = _mm_add_epi16(step2[20], kOne);
        step2[21] = _mm_add_epi16(step2[21], kOne);
        step2[22] = _mm_add_epi16(step2[22], kOne);
        step2[23] = _mm_add_epi16(step2[23], kOne);
        step2[24] = _mm_add_epi16(step2[24], kOne);
        step2[25] = _mm_add_epi16(step2[25], kOne);
        step2[26] = _mm_add_epi16(step2[26], kOne);
        step2[27] = _mm_add_epi16(step2[27], kOne);
        step1[28] = _mm_add_epi16(step1[28], kOne);
        step1[29] = _mm_add_epi16(step1[29], kOne);
        step1[30] = _mm_add_epi16(step1[30], kOne);
        step1[31] = _mm_add_epi16(step1[31], kOne);

        step2[0] = _mm_srai_epi16(step2[0], 2);
        step2[1] = _mm_srai_epi16(step2[1], 2);
        step2[2] = _mm_srai_epi16(step2[2], 2);
        step2[3] = _mm_srai_epi16(step2[3], 2);
        step2[4] = _mm_srai_epi16(step2[4], 2);
        step2[5] = _mm_srai_epi16(step2[5], 2);
        step2[6] = _mm_srai_epi16(step2[6], 2);
        step2[7] = _mm_srai_epi16(step2[7], 2);
        step2[8] = _mm_srai_epi16(step2[8], 2);
        step2[9] = _mm_srai_epi16(step2[9], 2);
        step2[10] = _mm_srai_epi16(step2[10], 2);
        step2[11] = _mm_srai_epi16(step2[11], 2);
        step2[12] = _mm_srai_epi16(step2[12], 2);
        step2[13] = _mm_srai_epi16(step2[13], 2);
        step2[14] = _mm_srai_epi16(step2[14], 2);
        step2[15] = _mm_srai_epi16(step2[15], 2);
        step1[16] = _mm_srai_epi16(step1[16], 2);
        step1[17] = _mm_srai_epi16(step1[17], 2);
        step1[18] = _mm_srai_epi16(step1[18], 2);
        step1[19] = _mm_srai_epi16(step1[19], 2);
        step2[20] = _mm_srai_epi16(step2[20], 2);
        step2[21] = _mm_srai_epi16(step2[21], 2);
        step2[22] = _mm_srai_epi16(step2[22], 2);
        step2[23] = _mm_srai_epi16(step2[23], 2);
        step2[24] = _mm_srai_epi16(step2[24], 2);
        step2[25] = _mm_srai_epi16(step2[25], 2);
        step2[26] = _mm_srai_epi16(step2[26], 2);
        step2[27] = _mm_srai_epi16(step2[27], 2);
        step1[28] = _mm_srai_epi16(step1[28], 2);
        step1[29] = _mm_srai_epi16(step1[29], 2);
        step1[30] = _mm_srai_epi16(step1[30], 2);
        step1[31] = _mm_srai_epi16(step1[31], 2);
      }
#endif  // !FDCT32x32_HIGH_PRECISION

#if FDCT32x32_HIGH_PRECISION
      if (pass == 0) {
#endif
        // Stage 3
        {
          step3[0] = ADD_EPI16(step2[(8 - 1)], step2[0]);
          step3[1] = ADD_EPI16(step2[(8 - 2)], step2[1]);
          step3[2] = ADD_EPI16(step2[(8 - 3)], step2[2]);
          step3[3] = ADD_EPI16(step2[(8 - 4)], step2[3]);
          step3[4] = SUB_EPI16(step2[(8 - 5)], step2[4]);
          step3[5] = SUB_EPI16(step2[(8 - 6)], step2[5]);
          step3[6] = SUB_EPI16(step2[(8 - 7)], step2[6]);
          step3[7] = SUB_EPI16(step2[(8 - 8)], step2[7]);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x8(&step3[0], &step3[1], &step3[2],
                                             &step3[3], &step3[4], &step3[5],
                                             &step3[6], &step3[7]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i s3_10_0 = _mm_unpacklo_epi16(step2[13], step2[10]);
          const __m128i s3_10_1 = _mm_unpackhi_epi16(step2[13], step2[10]);
          const __m128i s3_11_0 = _mm_unpacklo_epi16(step2[12], step2[11]);
          const __m128i s3_11_1 = _mm_unpackhi_epi16(step2[12], step2[11]);
          const __m128i s3_10_2 = _mm_madd_epi16(s3_10_0, k__cospi_p16_m16);
          const __m128i s3_10_3 = _mm_madd_epi16(s3_10_1, k__cospi_p16_m16);
          const __m128i s3_11_2 = _mm_madd_epi16(s3_11_0, k__cospi_p16_m16);
          const __m128i s3_11_3 = _mm_madd_epi16(s3_11_1, k__cospi_p16_m16);
          const __m128i s3_12_2 = _mm_madd_epi16(s3_11_0, k__cospi_p16_p16);
          const __m128i s3_12_3 = _mm_madd_epi16(s3_11_1, k__cospi_p16_p16);
          const __m128i s3_13_2 = _mm_madd_epi16(s3_10_0, k__cospi_p16_p16);
          const __m128i s3_13_3 = _mm_madd_epi16(s3_10_1, k__cospi_p16_p16);
          // dct_const_round_shift
          const __m128i s3_10_4 = _mm_add_epi32(s3_10_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_10_5 = _mm_add_epi32(s3_10_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_11_4 = _mm_add_epi32(s3_11_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_11_5 = _mm_add_epi32(s3_11_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_12_4 = _mm_add_epi32(s3_12_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_12_5 = _mm_add_epi32(s3_12_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_13_4 = _mm_add_epi32(s3_13_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_13_5 = _mm_add_epi32(s3_13_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_10_6 = _mm_srai_epi32(s3_10_4, DCT_CONST_BITS);
          const __m128i s3_10_7 = _mm_srai_epi32(s3_10_5, DCT_CONST_BITS);
          const __m128i s3_11_6 = _mm_srai_epi32(s3_11_4, DCT_CONST_BITS);
          const __m128i s3_11_7 = _mm_srai_epi32(s3_11_5, DCT_CONST_BITS);
          const __m128i s3_12_6 = _mm_srai_epi32(s3_12_4, DCT_CONST_BITS);
          const __m128i s3_12_7 = _mm_srai_epi32(s3_12_5, DCT_CONST_BITS);
          const __m128i s3_13_6 = _mm_srai_epi32(s3_13_4, DCT_CONST_BITS);
          const __m128i s3_13_7 = _mm_srai_epi32(s3_13_5, DCT_CONST_BITS);
          // Combine
          step3[10] = _mm_packs_epi32(s3_10_6, s3_10_7);
          step3[11] = _mm_packs_epi32(s3_11_6, s3_11_7);
          step3[12] = _mm_packs_epi32(s3_12_6, s3_12_7);
          step3[13] = _mm_packs_epi32(s3_13_6, s3_13_7);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&step3[10], &step3[11], &step3[12],
                                             &step3[13]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          step3[16] = ADD_EPI16(step2[23], step1[16]);
          step3[17] = ADD_EPI16(step2[22], step1[17]);
          step3[18] = ADD_EPI16(step2[21], step1[18]);
          step3[19] = ADD_EPI16(step2[20], step1[19]);
          step3[20] = SUB_EPI16(step1[19], step2[20]);
          step3[21] = SUB_EPI16(step1[18], step2[21]);
          step3[22] = SUB_EPI16(step1[17], step2[22]);
          step3[23] = SUB_EPI16(step1[16], step2[23]);
          step3[24] = SUB_EPI16(step1[31], step2[24]);
          step3[25] = SUB_EPI16(step1[30], step2[25]);
          step3[26] = SUB_EPI16(step1[29], step2[26]);
          step3[27] = SUB_EPI16(step1[28], step2[27]);
          step3[28] = ADD_EPI16(step2[27], step1[28]);
          step3[29] = ADD_EPI16(step2[26], step1[29]);
          step3[30] = ADD_EPI16(step2[25], step1[30]);
          step3[31] = ADD_EPI16(step2[24], step1[31]);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x16(
              &step3[16], &step3[17], &step3[18], &step3[19], &step3[20],
              &step3[21], &step3[22], &step3[23], &step3[24], &step3[25],
              &step3[26], &step3[27], &step3[28], &step3[29], &step3[30],
              &step3[31]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }

        // Stage 4
        {
          step1[0] = ADD_EPI16(step3[3], step3[0]);
          step1[1] = ADD_EPI16(step3[2], step3[1]);
          step1[2] = SUB_EPI16(step3[1], step3[2]);
          step1[3] = SUB_EPI16(step3[0], step3[3]);
          step1[8] = ADD_EPI16(step3[11], step2[8]);
          step1[9] = ADD_EPI16(step3[10], step2[9]);
          step1[10] = SUB_EPI16(step2[9], step3[10]);
          step1[11] = SUB_EPI16(step2[8], step3[11]);
          step1[12] = SUB_EPI16(step2[15], step3[12]);
          step1[13] = SUB_EPI16(step2[14], step3[13]);
          step1[14] = ADD_EPI16(step3[13], step2[14]);
          step1[15] = ADD_EPI16(step3[12], step2[15]);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x16(
              &step1[0], &step1[1], &step1[2], &step1[3], &step1[4], &step1[5],
              &step1[6], &step1[7], &step1[8], &step1[9], &step1[10],
              &step1[11], &step1[12], &step1[13], &step1[14], &step1[15]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i s1_05_0 = _mm_unpacklo_epi16(step3[6], step3[5]);
          const __m128i s1_05_1 = _mm_unpackhi_epi16(step3[6], step3[5]);
          const __m128i s1_05_2 = _mm_madd_epi16(s1_05_0, k__cospi_p16_m16);
          const __m128i s1_05_3 = _mm_madd_epi16(s1_05_1, k__cospi_p16_m16);
          const __m128i s1_06_2 = _mm_madd_epi16(s1_05_0, k__cospi_p16_p16);
          const __m128i s1_06_3 = _mm_madd_epi16(s1_05_1, k__cospi_p16_p16);
          // dct_const_round_shift
          const __m128i s1_05_4 = _mm_add_epi32(s1_05_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_05_5 = _mm_add_epi32(s1_05_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_06_4 = _mm_add_epi32(s1_06_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_06_5 = _mm_add_epi32(s1_06_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_05_6 = _mm_srai_epi32(s1_05_4, DCT_CONST_BITS);
          const __m128i s1_05_7 = _mm_srai_epi32(s1_05_5, DCT_CONST_BITS);
          const __m128i s1_06_6 = _mm_srai_epi32(s1_06_4, DCT_CONST_BITS);
          const __m128i s1_06_7 = _mm_srai_epi32(s1_06_5, DCT_CONST_BITS);
          // Combine
          step1[5] = _mm_packs_epi32(s1_05_6, s1_05_7);
          step1[6] = _mm_packs_epi32(s1_06_6, s1_06_7);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x2(&step1[5], &step1[6]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i s1_18_0 = _mm_unpacklo_epi16(step3[18], step3[29]);
          const __m128i s1_18_1 = _mm_unpackhi_epi16(step3[18], step3[29]);
          const __m128i s1_19_0 = _mm_unpacklo_epi16(step3[19], step3[28]);
          const __m128i s1_19_1 = _mm_unpackhi_epi16(step3[19], step3[28]);
          const __m128i s1_20_0 = _mm_unpacklo_epi16(step3[20], step3[27]);
          const __m128i s1_20_1 = _mm_unpackhi_epi16(step3[20], step3[27]);
          const __m128i s1_21_0 = _mm_unpacklo_epi16(step3[21], step3[26]);
          const __m128i s1_21_1 = _mm_unpackhi_epi16(step3[21], step3[26]);
          const __m128i s1_18_2 = _mm_madd_epi16(s1_18_0, k__cospi_m08_p24);
          const __m128i s1_18_3 = _mm_madd_epi16(s1_18_1, k__cospi_m08_p24);
          const __m128i s1_19_2 = _mm_madd_epi16(s1_19_0, k__cospi_m08_p24);
          const __m128i s1_19_3 = _mm_madd_epi16(s1_19_1, k__cospi_m08_p24);
          const __m128i s1_20_2 = _mm_madd_epi16(s1_20_0, k__cospi_m24_m08);
          const __m128i s1_20_3 = _mm_madd_epi16(s1_20_1, k__cospi_m24_m08);
          const __m128i s1_21_2 = _mm_madd_epi16(s1_21_0, k__cospi_m24_m08);
          const __m128i s1_21_3 = _mm_madd_epi16(s1_21_1, k__cospi_m24_m08);
          const __m128i s1_26_2 = _mm_madd_epi16(s1_21_0, k__cospi_m08_p24);
          const __m128i s1_26_3 = _mm_madd_epi16(s1_21_1, k__cospi_m08_p24);
          const __m128i s1_27_2 = _mm_madd_epi16(s1_20_0, k__cospi_m08_p24);
          const __m128i s1_27_3 = _mm_madd_epi16(s1_20_1, k__cospi_m08_p24);
          const __m128i s1_28_2 = _mm_madd_epi16(s1_19_0, k__cospi_p24_p08);
          const __m128i s1_28_3 = _mm_madd_epi16(s1_19_1, k__cospi_p24_p08);
          const __m128i s1_29_2 = _mm_madd_epi16(s1_18_0, k__cospi_p24_p08);
          const __m128i s1_29_3 = _mm_madd_epi16(s1_18_1, k__cospi_p24_p08);
          // dct_const_round_shift
          const __m128i s1_18_4 = _mm_add_epi32(s1_18_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_18_5 = _mm_add_epi32(s1_18_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_19_4 = _mm_add_epi32(s1_19_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_19_5 = _mm_add_epi32(s1_19_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_20_4 = _mm_add_epi32(s1_20_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_20_5 = _mm_add_epi32(s1_20_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_21_4 = _mm_add_epi32(s1_21_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_21_5 = _mm_add_epi32(s1_21_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_26_4 = _mm_add_epi32(s1_26_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_26_5 = _mm_add_epi32(s1_26_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_27_4 = _mm_add_epi32(s1_27_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_27_5 = _mm_add_epi32(s1_27_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_28_4 = _mm_add_epi32(s1_28_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_28_5 = _mm_add_epi32(s1_28_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_29_4 = _mm_add_epi32(s1_29_2, k__DCT_CONST_ROUNDING);
          const __m128i s1_29_5 = _mm_add_epi32(s1_29_3, k__DCT_CONST_ROUNDING);
          const __m128i s1_18_6 = _mm_srai_epi32(s1_18_4, DCT_CONST_BITS);
          const __m128i s1_18_7 = _mm_srai_epi32(s1_18_5, DCT_CONST_BITS);
          const __m128i s1_19_6 = _mm_srai_epi32(s1_19_4, DCT_CONST_BITS);
          const __m128i s1_19_7 = _mm_srai_epi32(s1_19_5, DCT_CONST_BITS);
          const __m128i s1_20_6 = _mm_srai_epi32(s1_20_4, DCT_CONST_BITS);
          const __m128i s1_20_7 = _mm_srai_epi32(s1_20_5, DCT_CONST_BITS);
          const __m128i s1_21_6 = _mm_srai_epi32(s1_21_4, DCT_CONST_BITS);
          const __m128i s1_21_7 = _mm_srai_epi32(s1_21_5, DCT_CONST_BITS);
          const __m128i s1_26_6 = _mm_srai_epi32(s1_26_4, DCT_CONST_BITS);
          const __m128i s1_26_7 = _mm_srai_epi32(s1_26_5, DCT_CONST_BITS);
          const __m128i s1_27_6 = _mm_srai_epi32(s1_27_4, DCT_CONST_BITS);
          const __m128i s1_27_7 = _mm_srai_epi32(s1_27_5, DCT_CONST_BITS);
          const __m128i s1_28_6 = _mm_srai_epi32(s1_28_4, DCT_CONST_BITS);
          const __m128i s1_28_7 = _mm_srai_epi32(s1_28_5, DCT_CONST_BITS);
          const __m128i s1_29_6 = _mm_srai_epi32(s1_29_4, DCT_CONST_BITS);
          const __m128i s1_29_7 = _mm_srai_epi32(s1_29_5, DCT_CONST_BITS);
          // Combine
          step1[18] = _mm_packs_epi32(s1_18_6, s1_18_7);
          step1[19] = _mm_packs_epi32(s1_19_6, s1_19_7);
          step1[20] = _mm_packs_epi32(s1_20_6, s1_20_7);
          step1[21] = _mm_packs_epi32(s1_21_6, s1_21_7);
          step1[26] = _mm_packs_epi32(s1_26_6, s1_26_7);
          step1[27] = _mm_packs_epi32(s1_27_6, s1_27_7);
          step1[28] = _mm_packs_epi32(s1_28_6, s1_28_7);
          step1[29] = _mm_packs_epi32(s1_29_6, s1_29_7);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x8(&step1[18], &step1[19], &step1[20],
                                             &step1[21], &step1[26], &step1[27],
                                             &step1[28], &step1[29]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // Stage 5
        {
          step2[4] = ADD_EPI16(step1[5], step3[4]);
          step2[5] = SUB_EPI16(step3[4], step1[5]);
          step2[6] = SUB_EPI16(step3[7], step1[6]);
          step2[7] = ADD_EPI16(step1[6], step3[7]);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&step2[4], &step2[5], &step2[6],
                                             &step2[7]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i out_00_0 = _mm_unpacklo_epi16(step1[0], step1[1]);
          const __m128i out_00_1 = _mm_unpackhi_epi16(step1[0], step1[1]);
          const __m128i out_08_0 = _mm_unpacklo_epi16(step1[2], step1[3]);
          const __m128i out_08_1 = _mm_unpackhi_epi16(step1[2], step1[3]);
          const __m128i out_00_2 = _mm_madd_epi16(out_00_0, k__cospi_p16_p16);
          const __m128i out_00_3 = _mm_madd_epi16(out_00_1, k__cospi_p16_p16);
          const __m128i out_16_2 = _mm_madd_epi16(out_00_0, k__cospi_p16_m16);
          const __m128i out_16_3 = _mm_madd_epi16(out_00_1, k__cospi_p16_m16);
          const __m128i out_08_2 = _mm_madd_epi16(out_08_0, k__cospi_p24_p08);
          const __m128i out_08_3 = _mm_madd_epi16(out_08_1, k__cospi_p24_p08);
          const __m128i out_24_2 = _mm_madd_epi16(out_08_0, k__cospi_m08_p24);
          const __m128i out_24_3 = _mm_madd_epi16(out_08_1, k__cospi_m08_p24);
          // dct_const_round_shift
          const __m128i out_00_4 =
              _mm_add_epi32(out_00_2, k__DCT_CONST_ROUNDING);
          const __m128i out_00_5 =
              _mm_add_epi32(out_00_3, k__DCT_CONST_ROUNDING);
          const __m128i out_16_4 =
              _mm_add_epi32(out_16_2, k__DCT_CONST_ROUNDING);
          const __m128i out_16_5 =
              _mm_add_epi32(out_16_3, k__DCT_CONST_ROUNDING);
          const __m128i out_08_4 =
              _mm_add_epi32(out_08_2, k__DCT_CONST_ROUNDING);
          const __m128i out_08_5 =
              _mm_add_epi32(out_08_3, k__DCT_CONST_ROUNDING);
          const __m128i out_24_4 =
              _mm_add_epi32(out_24_2, k__DCT_CONST_ROUNDING);
          const __m128i out_24_5 =
              _mm_add_epi32(out_24_3, k__DCT_CONST_ROUNDING);
          const __m128i out_00_6 = _mm_srai_epi32(out_00_4, DCT_CONST_BITS);
          const __m128i out_00_7 = _mm_srai_epi32(out_00_5, DCT_CONST_BITS);
          const __m128i out_16_6 = _mm_srai_epi32(out_16_4, DCT_CONST_BITS);
          const __m128i out_16_7 = _mm_srai_epi32(out_16_5, DCT_CONST_BITS);
          const __m128i out_08_6 = _mm_srai_epi32(out_08_4, DCT_CONST_BITS);
          const __m128i out_08_7 = _mm_srai_epi32(out_08_5, DCT_CONST_BITS);
          const __m128i out_24_6 = _mm_srai_epi32(out_24_4, DCT_CONST_BITS);
          const __m128i out_24_7 = _mm_srai_epi32(out_24_5, DCT_CONST_BITS);
          // Combine
          out[0] = _mm_packs_epi32(out_00_6, out_00_7);
          out[16] = _mm_packs_epi32(out_16_6, out_16_7);
          out[8] = _mm_packs_epi32(out_08_6, out_08_7);
          out[24] = _mm_packs_epi32(out_24_6, out_24_7);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x4(&out[0], &out[16], &out[8], &out[24]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i s2_09_0 = _mm_unpacklo_epi16(step1[9], step1[14]);
          const __m128i s2_09_1 = _mm_unpackhi_epi16(step1[9], step1[14]);
          const __m128i s2_10_0 = _mm_unpacklo_epi16(step1[10], step1[13]);
          const __m128i s2_10_1 = _mm_unpackhi_epi16(step1[10], step1[13]);
          const __m128i s2_09_2 = _mm_madd_epi16(s2_09_0, k__cospi_m08_p24);
          const __m128i s2_09_3 = _mm_madd_epi16(s2_09_1, k__cospi_m08_p24);
          const __m128i s2_10_2 = _mm_madd_epi16(s2_10_0, k__cospi_m24_m08);
          const __m128i s2_10_3 = _mm_madd_epi16(s2_10_1, k__cospi_m24_m08);
          const __m128i s2_13_2 = _mm_madd_epi16(s2_10_0, k__cospi_m08_p24);
          const __m128i s2_13_3 = _mm_madd_epi16(s2_10_1, k__cospi_m08_p24);
          const __m128i s2_14_2 = _mm_madd_epi16(s2_09_0, k__cospi_p24_p08);
          const __m128i s2_14_3 = _mm_madd_epi16(s2_09_1, k__cospi_p24_p08);
          // dct_const_round_shift
          const __m128i s2_09_4 = _mm_add_epi32(s2_09_2, k__DCT_CONST_ROUNDING);
          const __m128i s2_09_5 = _mm_add_epi32(s2_09_3, k__DCT_CONST_ROUNDING);
          const __m128i s2_10_4 = _mm_add_epi32(s2_10_2, k__DCT_CONST_ROUNDING);
          const __m128i s2_10_5 = _mm_add_epi32(s2_10_3, k__DCT_CONST_ROUNDING);
          const __m128i s2_13_4 = _mm_add_epi32(s2_13_2, k__DCT_CONST_ROUNDING);
          const __m128i s2_13_5 = _mm_add_epi32(s2_13_3, k__DCT_CONST_ROUNDING);
          const __m128i s2_14_4 = _mm_add_epi32(s2_14_2, k__DCT_CONST_ROUNDING);
          const __m128i s2_14_5 = _mm_add_epi32(s2_14_3, k__DCT_CONST_ROUNDING);
          const __m128i s2_09_6 = _mm_srai_epi32(s2_09_4, DCT_CONST_BITS);
          const __m128i s2_09_7 = _mm_srai_epi32(s2_09_5, DCT_CONST_BITS);
          const __m128i s2_10_6 = _mm_srai_epi32(s2_10_4, DCT_CONST_BITS);
          const __m128i s2_10_7 = _mm_srai_epi32(s2_10_5, DCT_CONST_BITS);
          const __m128i s2_13_6 = _mm_srai_epi32(s2_13_4, DCT_CONST_BITS);
          const __m128i s2_13_7 = _mm_srai_epi32(s2_13_5, DCT_CONST_BITS);
          const __m128i s2_14_6 = _mm_srai_epi32(s2_14_4, DCT_CONST_BITS);
          const __m128i s2_14_7 = _mm_srai_epi32(s2_14_5, DCT_CONST_BITS);
          // Combine
          step2[9] = _mm_packs_epi32(s2_09_6, s2_09_7);
          step2[10] = _mm_packs_epi32(s2_10_6, s2_10_7);
          step2[13] = _mm_packs_epi32(s2_13_6, s2_13_7);
          step2[14] = _mm_packs_epi32(s2_14_6, s2_14_7);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x4(&step2[9], &step2[10], &step2[13],
                                             &step2[14]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          step2[16] = ADD_EPI16(step1[19], step3[16]);
          step2[17] = ADD_EPI16(step1[18], step3[17]);
          step2[18] = SUB_EPI16(step3[17], step1[18]);
          step2[19] = SUB_EPI16(step3[16], step1[19]);
          step2[20] = SUB_EPI16(step3[23], step1[20]);
          step2[21] = SUB_EPI16(step3[22], step1[21]);
          step2[22] = ADD_EPI16(step1[21], step3[22]);
          step2[23] = ADD_EPI16(step1[20], step3[23]);
          step2[24] = ADD_EPI16(step1[27], step3[24]);
          step2[25] = ADD_EPI16(step1[26], step3[25]);
          step2[26] = SUB_EPI16(step3[25], step1[26]);
          step2[27] = SUB_EPI16(step3[24], step1[27]);
          step2[28] = SUB_EPI16(step3[31], step1[28]);
          step2[29] = SUB_EPI16(step3[30], step1[29]);
          step2[30] = ADD_EPI16(step1[29], step3[30]);
          step2[31] = ADD_EPI16(step1[28], step3[31]);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x16(
              &step2[16], &step2[17], &step2[18], &step2[19], &step2[20],
              &step2[21], &step2[22], &step2[23], &step2[24], &step2[25],
              &step2[26], &step2[27], &step2[28], &step2[29], &step2[30],
              &step2[31]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // Stage 6
        {
          const __m128i out_04_0 = _mm_unpacklo_epi16(step2[4], step2[7]);
          const __m128i out_04_1 = _mm_unpackhi_epi16(step2[4], step2[7]);
          const __m128i out_20_0 = _mm_unpacklo_epi16(step2[5], step2[6]);
          const __m128i out_20_1 = _mm_unpackhi_epi16(step2[5], step2[6]);
          const __m128i out_12_0 = _mm_unpacklo_epi16(step2[5], step2[6]);
          const __m128i out_12_1 = _mm_unpackhi_epi16(step2[5], step2[6]);
          const __m128i out_28_0 = _mm_unpacklo_epi16(step2[4], step2[7]);
          const __m128i out_28_1 = _mm_unpackhi_epi16(step2[4], step2[7]);
          const __m128i out_04_2 = _mm_madd_epi16(out_04_0, k__cospi_p28_p04);
          const __m128i out_04_3 = _mm_madd_epi16(out_04_1, k__cospi_p28_p04);
          const __m128i out_20_2 = _mm_madd_epi16(out_20_0, k__cospi_p12_p20);
          const __m128i out_20_3 = _mm_madd_epi16(out_20_1, k__cospi_p12_p20);
          const __m128i out_12_2 = _mm_madd_epi16(out_12_0, k__cospi_m20_p12);
          const __m128i out_12_3 = _mm_madd_epi16(out_12_1, k__cospi_m20_p12);
          const __m128i out_28_2 = _mm_madd_epi16(out_28_0, k__cospi_m04_p28);
          const __m128i out_28_3 = _mm_madd_epi16(out_28_1, k__cospi_m04_p28);
          // dct_const_round_shift
          const __m128i out_04_4 =
              _mm_add_epi32(out_04_2, k__DCT_CONST_ROUNDING);
          const __m128i out_04_5 =
              _mm_add_epi32(out_04_3, k__DCT_CONST_ROUNDING);
          const __m128i out_20_4 =
              _mm_add_epi32(out_20_2, k__DCT_CONST_ROUNDING);
          const __m128i out_20_5 =
              _mm_add_epi32(out_20_3, k__DCT_CONST_ROUNDING);
          const __m128i out_12_4 =
              _mm_add_epi32(out_12_2, k__DCT_CONST_ROUNDING);
          const __m128i out_12_5 =
              _mm_add_epi32(out_12_3, k__DCT_CONST_ROUNDING);
          const __m128i out_28_4 =
              _mm_add_epi32(out_28_2, k__DCT_CONST_ROUNDING);
          const __m128i out_28_5 =
              _mm_add_epi32(out_28_3, k__DCT_CONST_ROUNDING);
          const __m128i out_04_6 = _mm_srai_epi32(out_04_4, DCT_CONST_BITS);
          const __m128i out_04_7 = _mm_srai_epi32(out_04_5, DCT_CONST_BITS);
          const __m128i out_20_6 = _mm_srai_epi32(out_20_4, DCT_CONST_BITS);
          const __m128i out_20_7 = _mm_srai_epi32(out_20_5, DCT_CONST_BITS);
          const __m128i out_12_6 = _mm_srai_epi32(out_12_4, DCT_CONST_BITS);
          const __m128i out_12_7 = _mm_srai_epi32(out_12_5, DCT_CONST_BITS);
          const __m128i out_28_6 = _mm_srai_epi32(out_28_4, DCT_CONST_BITS);
          const __m128i out_28_7 = _mm_srai_epi32(out_28_5, DCT_CONST_BITS);
          // Combine
          out[4] = _mm_packs_epi32(out_04_6, out_04_7);
          out[20] = _mm_packs_epi32(out_20_6, out_20_7);
          out[12] = _mm_packs_epi32(out_12_6, out_12_7);
          out[28] = _mm_packs_epi32(out_28_6, out_28_7);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x4(&out[4], &out[20], &out[12], &out[28]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          step3[8] = ADD_EPI16(step2[9], step1[8]);
          step3[9] = SUB_EPI16(step1[8], step2[9]);
          step3[10] = SUB_EPI16(step1[11], step2[10]);
          step3[11] = ADD_EPI16(step2[10], step1[11]);
          step3[12] = ADD_EPI16(step2[13], step1[12]);
          step3[13] = SUB_EPI16(step1[12], step2[13]);
          step3[14] = SUB_EPI16(step1[15], step2[14]);
          step3[15] = ADD_EPI16(step2[14], step1[15]);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x8(&step3[8], &step3[9], &step3[10],
                                             &step3[11], &step3[12], &step3[13],
                                             &step3[14], &step3[15]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i s3_17_0 = _mm_unpacklo_epi16(step2[17], step2[30]);
          const __m128i s3_17_1 = _mm_unpackhi_epi16(step2[17], step2[30]);
          const __m128i s3_18_0 = _mm_unpacklo_epi16(step2[18], step2[29]);
          const __m128i s3_18_1 = _mm_unpackhi_epi16(step2[18], step2[29]);
          const __m128i s3_21_0 = _mm_unpacklo_epi16(step2[21], step2[26]);
          const __m128i s3_21_1 = _mm_unpackhi_epi16(step2[21], step2[26]);
          const __m128i s3_22_0 = _mm_unpacklo_epi16(step2[22], step2[25]);
          const __m128i s3_22_1 = _mm_unpackhi_epi16(step2[22], step2[25]);
          const __m128i s3_17_2 = _mm_madd_epi16(s3_17_0, k__cospi_m04_p28);
          const __m128i s3_17_3 = _mm_madd_epi16(s3_17_1, k__cospi_m04_p28);
          const __m128i s3_18_2 = _mm_madd_epi16(s3_18_0, k__cospi_m28_m04);
          const __m128i s3_18_3 = _mm_madd_epi16(s3_18_1, k__cospi_m28_m04);
          const __m128i s3_21_2 = _mm_madd_epi16(s3_21_0, k__cospi_m20_p12);
          const __m128i s3_21_3 = _mm_madd_epi16(s3_21_1, k__cospi_m20_p12);
          const __m128i s3_22_2 = _mm_madd_epi16(s3_22_0, k__cospi_m12_m20);
          const __m128i s3_22_3 = _mm_madd_epi16(s3_22_1, k__cospi_m12_m20);
          const __m128i s3_25_2 = _mm_madd_epi16(s3_22_0, k__cospi_m20_p12);
          const __m128i s3_25_3 = _mm_madd_epi16(s3_22_1, k__cospi_m20_p12);
          const __m128i s3_26_2 = _mm_madd_epi16(s3_21_0, k__cospi_p12_p20);
          const __m128i s3_26_3 = _mm_madd_epi16(s3_21_1, k__cospi_p12_p20);
          const __m128i s3_29_2 = _mm_madd_epi16(s3_18_0, k__cospi_m04_p28);
          const __m128i s3_29_3 = _mm_madd_epi16(s3_18_1, k__cospi_m04_p28);
          const __m128i s3_30_2 = _mm_madd_epi16(s3_17_0, k__cospi_p28_p04);
          const __m128i s3_30_3 = _mm_madd_epi16(s3_17_1, k__cospi_p28_p04);
          // dct_const_round_shift
          const __m128i s3_17_4 = _mm_add_epi32(s3_17_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_17_5 = _mm_add_epi32(s3_17_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_18_4 = _mm_add_epi32(s3_18_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_18_5 = _mm_add_epi32(s3_18_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_21_4 = _mm_add_epi32(s3_21_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_21_5 = _mm_add_epi32(s3_21_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_22_4 = _mm_add_epi32(s3_22_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_22_5 = _mm_add_epi32(s3_22_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_17_6 = _mm_srai_epi32(s3_17_4, DCT_CONST_BITS);
          const __m128i s3_17_7 = _mm_srai_epi32(s3_17_5, DCT_CONST_BITS);
          const __m128i s3_18_6 = _mm_srai_epi32(s3_18_4, DCT_CONST_BITS);
          const __m128i s3_18_7 = _mm_srai_epi32(s3_18_5, DCT_CONST_BITS);
          const __m128i s3_21_6 = _mm_srai_epi32(s3_21_4, DCT_CONST_BITS);
          const __m128i s3_21_7 = _mm_srai_epi32(s3_21_5, DCT_CONST_BITS);
          const __m128i s3_22_6 = _mm_srai_epi32(s3_22_4, DCT_CONST_BITS);
          const __m128i s3_22_7 = _mm_srai_epi32(s3_22_5, DCT_CONST_BITS);
          const __m128i s3_25_4 = _mm_add_epi32(s3_25_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_25_5 = _mm_add_epi32(s3_25_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_26_4 = _mm_add_epi32(s3_26_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_26_5 = _mm_add_epi32(s3_26_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_29_4 = _mm_add_epi32(s3_29_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_29_5 = _mm_add_epi32(s3_29_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_30_4 = _mm_add_epi32(s3_30_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_30_5 = _mm_add_epi32(s3_30_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_25_6 = _mm_srai_epi32(s3_25_4, DCT_CONST_BITS);
          const __m128i s3_25_7 = _mm_srai_epi32(s3_25_5, DCT_CONST_BITS);
          const __m128i s3_26_6 = _mm_srai_epi32(s3_26_4, DCT_CONST_BITS);
          const __m128i s3_26_7 = _mm_srai_epi32(s3_26_5, DCT_CONST_BITS);
          const __m128i s3_29_6 = _mm_srai_epi32(s3_29_4, DCT_CONST_BITS);
          const __m128i s3_29_7 = _mm_srai_epi32(s3_29_5, DCT_CONST_BITS);
          const __m128i s3_30_6 = _mm_srai_epi32(s3_30_4, DCT_CONST_BITS);
          const __m128i s3_30_7 = _mm_srai_epi32(s3_30_5, DCT_CONST_BITS);
          // Combine
          step3[17] = _mm_packs_epi32(s3_17_6, s3_17_7);
          step3[18] = _mm_packs_epi32(s3_18_6, s3_18_7);
          step3[21] = _mm_packs_epi32(s3_21_6, s3_21_7);
          step3[22] = _mm_packs_epi32(s3_22_6, s3_22_7);
          // Combine
          step3[25] = _mm_packs_epi32(s3_25_6, s3_25_7);
          step3[26] = _mm_packs_epi32(s3_26_6, s3_26_7);
          step3[29] = _mm_packs_epi32(s3_29_6, s3_29_7);
          step3[30] = _mm_packs_epi32(s3_30_6, s3_30_7);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x8(&step3[17], &step3[18], &step3[21],
                                             &step3[22], &step3[25], &step3[26],
                                             &step3[29], &step3[30]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // Stage 7
        {
          const __m128i out_02_0 = _mm_unpacklo_epi16(step3[8], step3[15]);
          const __m128i out_02_1 = _mm_unpackhi_epi16(step3[8], step3[15]);
          const __m128i out_18_0 = _mm_unpacklo_epi16(step3[9], step3[14]);
          const __m128i out_18_1 = _mm_unpackhi_epi16(step3[9], step3[14]);
          const __m128i out_10_0 = _mm_unpacklo_epi16(step3[10], step3[13]);
          const __m128i out_10_1 = _mm_unpackhi_epi16(step3[10], step3[13]);
          const __m128i out_26_0 = _mm_unpacklo_epi16(step3[11], step3[12]);
          const __m128i out_26_1 = _mm_unpackhi_epi16(step3[11], step3[12]);
          const __m128i out_02_2 = _mm_madd_epi16(out_02_0, k__cospi_p30_p02);
          const __m128i out_02_3 = _mm_madd_epi16(out_02_1, k__cospi_p30_p02);
          const __m128i out_18_2 = _mm_madd_epi16(out_18_0, k__cospi_p14_p18);
          const __m128i out_18_3 = _mm_madd_epi16(out_18_1, k__cospi_p14_p18);
          const __m128i out_10_2 = _mm_madd_epi16(out_10_0, k__cospi_p22_p10);
          const __m128i out_10_3 = _mm_madd_epi16(out_10_1, k__cospi_p22_p10);
          const __m128i out_26_2 = _mm_madd_epi16(out_26_0, k__cospi_p06_p26);
          const __m128i out_26_3 = _mm_madd_epi16(out_26_1, k__cospi_p06_p26);
          const __m128i out_06_2 = _mm_madd_epi16(out_26_0, k__cospi_m26_p06);
          const __m128i out_06_3 = _mm_madd_epi16(out_26_1, k__cospi_m26_p06);
          const __m128i out_22_2 = _mm_madd_epi16(out_10_0, k__cospi_m10_p22);
          const __m128i out_22_3 = _mm_madd_epi16(out_10_1, k__cospi_m10_p22);
          const __m128i out_14_2 = _mm_madd_epi16(out_18_0, k__cospi_m18_p14);
          const __m128i out_14_3 = _mm_madd_epi16(out_18_1, k__cospi_m18_p14);
          const __m128i out_30_2 = _mm_madd_epi16(out_02_0, k__cospi_m02_p30);
          const __m128i out_30_3 = _mm_madd_epi16(out_02_1, k__cospi_m02_p30);
          // dct_const_round_shift
          const __m128i out_02_4 =
              _mm_add_epi32(out_02_2, k__DCT_CONST_ROUNDING);
          const __m128i out_02_5 =
              _mm_add_epi32(out_02_3, k__DCT_CONST_ROUNDING);
          const __m128i out_18_4 =
              _mm_add_epi32(out_18_2, k__DCT_CONST_ROUNDING);
          const __m128i out_18_5 =
              _mm_add_epi32(out_18_3, k__DCT_CONST_ROUNDING);
          const __m128i out_10_4 =
              _mm_add_epi32(out_10_2, k__DCT_CONST_ROUNDING);
          const __m128i out_10_5 =
              _mm_add_epi32(out_10_3, k__DCT_CONST_ROUNDING);
          const __m128i out_26_4 =
              _mm_add_epi32(out_26_2, k__DCT_CONST_ROUNDING);
          const __m128i out_26_5 =
              _mm_add_epi32(out_26_3, k__DCT_CONST_ROUNDING);
          const __m128i out_06_4 =
              _mm_add_epi32(out_06_2, k__DCT_CONST_ROUNDING);
          const __m128i out_06_5 =
              _mm_add_epi32(out_06_3, k__DCT_CONST_ROUNDING);
          const __m128i out_22_4 =
              _mm_add_epi32(out_22_2, k__DCT_CONST_ROUNDING);
          const __m128i out_22_5 =
              _mm_add_epi32(out_22_3, k__DCT_CONST_ROUNDING);
          const __m128i out_14_4 =
              _mm_add_epi32(out_14_2, k__DCT_CONST_ROUNDING);
          const __m128i out_14_5 =
              _mm_add_epi32(out_14_3, k__DCT_CONST_ROUNDING);
          const __m128i out_30_4 =
              _mm_add_epi32(out_30_2, k__DCT_CONST_ROUNDING);
          const __m128i out_30_5 =
              _mm_add_epi32(out_30_3, k__DCT_CONST_ROUNDING);
          const __m128i out_02_6 = _mm_srai_epi32(out_02_4, DCT_CONST_BITS);
          const __m128i out_02_7 = _mm_srai_epi32(out_02_5, DCT_CONST_BITS);
          const __m128i out_18_6 = _mm_srai_epi32(out_18_4, DCT_CONST_BITS);
          const __m128i out_18_7 = _mm_srai_epi32(out_18_5, DCT_CONST_BITS);
          const __m128i out_10_6 = _mm_srai_epi32(out_10_4, DCT_CONST_BITS);
          const __m128i out_10_7 = _mm_srai_epi32(out_10_5, DCT_CONST_BITS);
          const __m128i out_26_6 = _mm_srai_epi32(out_26_4, DCT_CONST_BITS);
          const __m128i out_26_7 = _mm_srai_epi32(out_26_5, DCT_CONST_BITS);
          const __m128i out_06_6 = _mm_srai_epi32(out_06_4, DCT_CONST_BITS);
          const __m128i out_06_7 = _mm_srai_epi32(out_06_5, DCT_CONST_BITS);
          const __m128i out_22_6 = _mm_srai_epi32(out_22_4, DCT_CONST_BITS);
          const __m128i out_22_7 = _mm_srai_epi32(out_22_5, DCT_CONST_BITS);
          const __m128i out_14_6 = _mm_srai_epi32(out_14_4, DCT_CONST_BITS);
          const __m128i out_14_7 = _mm_srai_epi32(out_14_5, DCT_CONST_BITS);
          const __m128i out_30_6 = _mm_srai_epi32(out_30_4, DCT_CONST_BITS);
          const __m128i out_30_7 = _mm_srai_epi32(out_30_5, DCT_CONST_BITS);
          // Combine
          out[2] = _mm_packs_epi32(out_02_6, out_02_7);
          out[18] = _mm_packs_epi32(out_18_6, out_18_7);
          out[10] = _mm_packs_epi32(out_10_6, out_10_7);
          out[26] = _mm_packs_epi32(out_26_6, out_26_7);
          out[6] = _mm_packs_epi32(out_06_6, out_06_7);
          out[22] = _mm_packs_epi32(out_22_6, out_22_7);
          out[14] = _mm_packs_epi32(out_14_6, out_14_7);
          out[30] = _mm_packs_epi32(out_30_6, out_30_7);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&out[2], &out[18], &out[10], &out[26],
                                      &out[6], &out[22], &out[14], &out[30]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          step1[16] = ADD_EPI16(step3[17], step2[16]);
          step1[17] = SUB_EPI16(step2[16], step3[17]);
          step1[18] = SUB_EPI16(step2[19], step3[18]);
          step1[19] = ADD_EPI16(step3[18], step2[19]);
          step1[20] = ADD_EPI16(step3[21], step2[20]);
          step1[21] = SUB_EPI16(step2[20], step3[21]);
          step1[22] = SUB_EPI16(step2[23], step3[22]);
          step1[23] = ADD_EPI16(step3[22], step2[23]);
          step1[24] = ADD_EPI16(step3[25], step2[24]);
          step1[25] = SUB_EPI16(step2[24], step3[25]);
          step1[26] = SUB_EPI16(step2[27], step3[26]);
          step1[27] = ADD_EPI16(step3[26], step2[27]);
          step1[28] = ADD_EPI16(step3[29], step2[28]);
          step1[29] = SUB_EPI16(step2[28], step3[29]);
          step1[30] = SUB_EPI16(step2[31], step3[30]);
          step1[31] = ADD_EPI16(step3[30], step2[31]);
#if DCT_HIGH_BIT_DEPTH
          overflow = check_epi16_overflow_x16(
              &step1[16], &step1[17], &step1[18], &step1[19], &step1[20],
              &step1[21], &step1[22], &step1[23], &step1[24], &step1[25],
              &step1[26], &step1[27], &step1[28], &step1[29], &step1[30],
              &step1[31]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        // Final stage --- outputs indices are bit-reversed.
        {
          const __m128i out_01_0 = _mm_unpacklo_epi16(step1[16], step1[31]);
          const __m128i out_01_1 = _mm_unpackhi_epi16(step1[16], step1[31]);
          const __m128i out_17_0 = _mm_unpacklo_epi16(step1[17], step1[30]);
          const __m128i out_17_1 = _mm_unpackhi_epi16(step1[17], step1[30]);
          const __m128i out_09_0 = _mm_unpacklo_epi16(step1[18], step1[29]);
          const __m128i out_09_1 = _mm_unpackhi_epi16(step1[18], step1[29]);
          const __m128i out_25_0 = _mm_unpacklo_epi16(step1[19], step1[28]);
          const __m128i out_25_1 = _mm_unpackhi_epi16(step1[19], step1[28]);
          const __m128i out_01_2 = _mm_madd_epi16(out_01_0, k__cospi_p31_p01);
          const __m128i out_01_3 = _mm_madd_epi16(out_01_1, k__cospi_p31_p01);
          const __m128i out_17_2 = _mm_madd_epi16(out_17_0, k__cospi_p15_p17);
          const __m128i out_17_3 = _mm_madd_epi16(out_17_1, k__cospi_p15_p17);
          const __m128i out_09_2 = _mm_madd_epi16(out_09_0, k__cospi_p23_p09);
          const __m128i out_09_3 = _mm_madd_epi16(out_09_1, k__cospi_p23_p09);
          const __m128i out_25_2 = _mm_madd_epi16(out_25_0, k__cospi_p07_p25);
          const __m128i out_25_3 = _mm_madd_epi16(out_25_1, k__cospi_p07_p25);
          const __m128i out_07_2 = _mm_madd_epi16(out_25_0, k__cospi_m25_p07);
          const __m128i out_07_3 = _mm_madd_epi16(out_25_1, k__cospi_m25_p07);
          const __m128i out_23_2 = _mm_madd_epi16(out_09_0, k__cospi_m09_p23);
          const __m128i out_23_3 = _mm_madd_epi16(out_09_1, k__cospi_m09_p23);
          const __m128i out_15_2 = _mm_madd_epi16(out_17_0, k__cospi_m17_p15);
          const __m128i out_15_3 = _mm_madd_epi16(out_17_1, k__cospi_m17_p15);
          const __m128i out_31_2 = _mm_madd_epi16(out_01_0, k__cospi_m01_p31);
          const __m128i out_31_3 = _mm_madd_epi16(out_01_1, k__cospi_m01_p31);
          // dct_const_round_shift
          const __m128i out_01_4 =
              _mm_add_epi32(out_01_2, k__DCT_CONST_ROUNDING);
          const __m128i out_01_5 =
              _mm_add_epi32(out_01_3, k__DCT_CONST_ROUNDING);
          const __m128i out_17_4 =
              _mm_add_epi32(out_17_2, k__DCT_CONST_ROUNDING);
          const __m128i out_17_5 =
              _mm_add_epi32(out_17_3, k__DCT_CONST_ROUNDING);
          const __m128i out_09_4 =
              _mm_add_epi32(out_09_2, k__DCT_CONST_ROUNDING);
          const __m128i out_09_5 =
              _mm_add_epi32(out_09_3, k__DCT_CONST_ROUNDING);
          const __m128i out_25_4 =
              _mm_add_epi32(out_25_2, k__DCT_CONST_ROUNDING);
          const __m128i out_25_5 =
              _mm_add_epi32(out_25_3, k__DCT_CONST_ROUNDING);
          const __m128i out_07_4 =
              _mm_add_epi32(out_07_2, k__DCT_CONST_ROUNDING);
          const __m128i out_07_5 =
              _mm_add_epi32(out_07_3, k__DCT_CONST_ROUNDING);
          const __m128i out_23_4 =
              _mm_add_epi32(out_23_2, k__DCT_CONST_ROUNDING);
          const __m128i out_23_5 =
              _mm_add_epi32(out_23_3, k__DCT_CONST_ROUNDING);
          const __m128i out_15_4 =
              _mm_add_epi32(out_15_2, k__DCT_CONST_ROUNDING);
          const __m128i out_15_5 =
              _mm_add_epi32(out_15_3, k__DCT_CONST_ROUNDING);
          const __m128i out_31_4 =
              _mm_add_epi32(out_31_2, k__DCT_CONST_ROUNDING);
          const __m128i out_31_5 =
              _mm_add_epi32(out_31_3, k__DCT_CONST_ROUNDING);
          const __m128i out_01_6 = _mm_srai_epi32(out_01_4, DCT_CONST_BITS);
          const __m128i out_01_7 = _mm_srai_epi32(out_01_5, DCT_CONST_BITS);
          const __m128i out_17_6 = _mm_srai_epi32(out_17_4, DCT_CONST_BITS);
          const __m128i out_17_7 = _mm_srai_epi32(out_17_5, DCT_CONST_BITS);
          const __m128i out_09_6 = _mm_srai_epi32(out_09_4, DCT_CONST_BITS);
          const __m128i out_09_7 = _mm_srai_epi32(out_09_5, DCT_CONST_BITS);
          const __m128i out_25_6 = _mm_srai_epi32(out_25_4, DCT_CONST_BITS);
          const __m128i out_25_7 = _mm_srai_epi32(out_25_5, DCT_CONST_BITS);
          const __m128i out_07_6 = _mm_srai_epi32(out_07_4, DCT_CONST_BITS);
          const __m128i out_07_7 = _mm_srai_epi32(out_07_5, DCT_CONST_BITS);
          const __m128i out_23_6 = _mm_srai_epi32(out_23_4, DCT_CONST_BITS);
          const __m128i out_23_7 = _mm_srai_epi32(out_23_5, DCT_CONST_BITS);
          const __m128i out_15_6 = _mm_srai_epi32(out_15_4, DCT_CONST_BITS);
          const __m128i out_15_7 = _mm_srai_epi32(out_15_5, DCT_CONST_BITS);
          const __m128i out_31_6 = _mm_srai_epi32(out_31_4, DCT_CONST_BITS);
          const __m128i out_31_7 = _mm_srai_epi32(out_31_5, DCT_CONST_BITS);
          // Combine
          out[1] = _mm_packs_epi32(out_01_6, out_01_7);
          out[17] = _mm_packs_epi32(out_17_6, out_17_7);
          out[9] = _mm_packs_epi32(out_09_6, out_09_7);
          out[25] = _mm_packs_epi32(out_25_6, out_25_7);
          out[7] = _mm_packs_epi32(out_07_6, out_07_7);
          out[23] = _mm_packs_epi32(out_23_6, out_23_7);
          out[15] = _mm_packs_epi32(out_15_6, out_15_7);
          out[31] = _mm_packs_epi32(out_31_6, out_31_7);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&out[1], &out[17], &out[9], &out[25],
                                      &out[7], &out[23], &out[15], &out[31]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i out_05_0 = _mm_unpacklo_epi16(step1[20], step1[27]);
          const __m128i out_05_1 = _mm_unpackhi_epi16(step1[20], step1[27]);
          const __m128i out_21_0 = _mm_unpacklo_epi16(step1[21], step1[26]);
          const __m128i out_21_1 = _mm_unpackhi_epi16(step1[21], step1[26]);
          const __m128i out_13_0 = _mm_unpacklo_epi16(step1[22], step1[25]);
          const __m128i out_13_1 = _mm_unpackhi_epi16(step1[22], step1[25]);
          const __m128i out_29_0 = _mm_unpacklo_epi16(step1[23], step1[24]);
          const __m128i out_29_1 = _mm_unpackhi_epi16(step1[23], step1[24]);
          const __m128i out_05_2 = _mm_madd_epi16(out_05_0, k__cospi_p27_p05);
          const __m128i out_05_3 = _mm_madd_epi16(out_05_1, k__cospi_p27_p05);
          const __m128i out_21_2 = _mm_madd_epi16(out_21_0, k__cospi_p11_p21);
          const __m128i out_21_3 = _mm_madd_epi16(out_21_1, k__cospi_p11_p21);
          const __m128i out_13_2 = _mm_madd_epi16(out_13_0, k__cospi_p19_p13);
          const __m128i out_13_3 = _mm_madd_epi16(out_13_1, k__cospi_p19_p13);
          const __m128i out_29_2 = _mm_madd_epi16(out_29_0, k__cospi_p03_p29);
          const __m128i out_29_3 = _mm_madd_epi16(out_29_1, k__cospi_p03_p29);
          const __m128i out_03_2 = _mm_madd_epi16(out_29_0, k__cospi_m29_p03);
          const __m128i out_03_3 = _mm_madd_epi16(out_29_1, k__cospi_m29_p03);
          const __m128i out_19_2 = _mm_madd_epi16(out_13_0, k__cospi_m13_p19);
          const __m128i out_19_3 = _mm_madd_epi16(out_13_1, k__cospi_m13_p19);
          const __m128i out_11_2 = _mm_madd_epi16(out_21_0, k__cospi_m21_p11);
          const __m128i out_11_3 = _mm_madd_epi16(out_21_1, k__cospi_m21_p11);
          const __m128i out_27_2 = _mm_madd_epi16(out_05_0, k__cospi_m05_p27);
          const __m128i out_27_3 = _mm_madd_epi16(out_05_1, k__cospi_m05_p27);
          // dct_const_round_shift
          const __m128i out_05_4 =
              _mm_add_epi32(out_05_2, k__DCT_CONST_ROUNDING);
          const __m128i out_05_5 =
              _mm_add_epi32(out_05_3, k__DCT_CONST_ROUNDING);
          const __m128i out_21_4 =
              _mm_add_epi32(out_21_2, k__DCT_CONST_ROUNDING);
          const __m128i out_21_5 =
              _mm_add_epi32(out_21_3, k__DCT_CONST_ROUNDING);
          const __m128i out_13_4 =
              _mm_add_epi32(out_13_2, k__DCT_CONST_ROUNDING);
          const __m128i out_13_5 =
              _mm_add_epi32(out_13_3, k__DCT_CONST_ROUNDING);
          const __m128i out_29_4 =
              _mm_add_epi32(out_29_2, k__DCT_CONST_ROUNDING);
          const __m128i out_29_5 =
              _mm_add_epi32(out_29_3, k__DCT_CONST_ROUNDING);
          const __m128i out_03_4 =
              _mm_add_epi32(out_03_2, k__DCT_CONST_ROUNDING);
          const __m128i out_03_5 =
              _mm_add_epi32(out_03_3, k__DCT_CONST_ROUNDING);
          const __m128i out_19_4 =
              _mm_add_epi32(out_19_2, k__DCT_CONST_ROUNDING);
          const __m128i out_19_5 =
              _mm_add_epi32(out_19_3, k__DCT_CONST_ROUNDING);
          const __m128i out_11_4 =
              _mm_add_epi32(out_11_2, k__DCT_CONST_ROUNDING);
          const __m128i out_11_5 =
              _mm_add_epi32(out_11_3, k__DCT_CONST_ROUNDING);
          const __m128i out_27_4 =
              _mm_add_epi32(out_27_2, k__DCT_CONST_ROUNDING);
          const __m128i out_27_5 =
              _mm_add_epi32(out_27_3, k__DCT_CONST_ROUNDING);
          const __m128i out_05_6 = _mm_srai_epi32(out_05_4, DCT_CONST_BITS);
          const __m128i out_05_7 = _mm_srai_epi32(out_05_5, DCT_CONST_BITS);
          const __m128i out_21_6 = _mm_srai_epi32(out_21_4, DCT_CONST_BITS);
          const __m128i out_21_7 = _mm_srai_epi32(out_21_5, DCT_CONST_BITS);
          const __m128i out_13_6 = _mm_srai_epi32(out_13_4, DCT_CONST_BITS);
          const __m128i out_13_7 = _mm_srai_epi32(out_13_5, DCT_CONST_BITS);
          const __m128i out_29_6 = _mm_srai_epi32(out_29_4, DCT_CONST_BITS);
          const __m128i out_29_7 = _mm_srai_epi32(out_29_5, DCT_CONST_BITS);
          const __m128i out_03_6 = _mm_srai_epi32(out_03_4, DCT_CONST_BITS);
          const __m128i out_03_7 = _mm_srai_epi32(out_03_5, DCT_CONST_BITS);
          const __m128i out_19_6 = _mm_srai_epi32(out_19_4, DCT_CONST_BITS);
          const __m128i out_19_7 = _mm_srai_epi32(out_19_5, DCT_CONST_BITS);
          const __m128i out_11_6 = _mm_srai_epi32(out_11_4, DCT_CONST_BITS);
          const __m128i out_11_7 = _mm_srai_epi32(out_11_5, DCT_CONST_BITS);
          const __m128i out_27_6 = _mm_srai_epi32(out_27_4, DCT_CONST_BITS);
          const __m128i out_27_7 = _mm_srai_epi32(out_27_5, DCT_CONST_BITS);
          // Combine
          out[5] = _mm_packs_epi32(out_05_6, out_05_7);
          out[21] = _mm_packs_epi32(out_21_6, out_21_7);
          out[13] = _mm_packs_epi32(out_13_6, out_13_7);
          out[29] = _mm_packs_epi32(out_29_6, out_29_7);
          out[3] = _mm_packs_epi32(out_03_6, out_03_7);
          out[19] = _mm_packs_epi32(out_19_6, out_19_7);
          out[11] = _mm_packs_epi32(out_11_6, out_11_7);
          out[27] = _mm_packs_epi32(out_27_6, out_27_7);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&out[5], &out[21], &out[13], &out[29],
                                      &out[3], &out[19], &out[11], &out[27]);
          if (overflow) {
            if (pass == 0)
              HIGH_FDCT32x32_2D_C(input, output_org, stride);
            else
              HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
#if FDCT32x32_HIGH_PRECISION
      } else {
        __m128i lstep1[64], lstep2[64], lstep3[64];
        __m128i u[32], v[32], sign[16];
        const __m128i K32One = _mm_set_epi32(1, 1, 1, 1);
        const __m128i k__pOne_mOne = pair_set_epi16(1, -1);
        // start using 32-bit operations
        // stage 3
        {
          // expanding to 32-bit length while adding and subtracting
          lstep2[0] = _mm_unpacklo_epi16(step2[0], step2[7]);
          lstep2[1] = _mm_unpackhi_epi16(step2[0], step2[7]);
          lstep2[2] = _mm_unpacklo_epi16(step2[1], step2[6]);
          lstep2[3] = _mm_unpackhi_epi16(step2[1], step2[6]);
          lstep2[4] = _mm_unpacklo_epi16(step2[2], step2[5]);
          lstep2[5] = _mm_unpackhi_epi16(step2[2], step2[5]);
          lstep2[6] = _mm_unpacklo_epi16(step2[3], step2[4]);
          lstep2[7] = _mm_unpackhi_epi16(step2[3], step2[4]);

          lstep3[0] = _mm_madd_epi16(lstep2[0], kOne);
          lstep3[1] = _mm_madd_epi16(lstep2[1], kOne);
          lstep3[2] = _mm_madd_epi16(lstep2[2], kOne);
          lstep3[3] = _mm_madd_epi16(lstep2[3], kOne);
          lstep3[4] = _mm_madd_epi16(lstep2[4], kOne);
          lstep3[5] = _mm_madd_epi16(lstep2[5], kOne);
          lstep3[6] = _mm_madd_epi16(lstep2[6], kOne);
          lstep3[7] = _mm_madd_epi16(lstep2[7], kOne);

          lstep3[8] = _mm_madd_epi16(lstep2[6], k__pOne_mOne);
          lstep3[9] = _mm_madd_epi16(lstep2[7], k__pOne_mOne);
          lstep3[10] = _mm_madd_epi16(lstep2[4], k__pOne_mOne);
          lstep3[11] = _mm_madd_epi16(lstep2[5], k__pOne_mOne);
          lstep3[12] = _mm_madd_epi16(lstep2[2], k__pOne_mOne);
          lstep3[13] = _mm_madd_epi16(lstep2[3], k__pOne_mOne);
          lstep3[14] = _mm_madd_epi16(lstep2[0], k__pOne_mOne);
          lstep3[15] = _mm_madd_epi16(lstep2[1], k__pOne_mOne);
        }
        {
          const __m128i s3_10_0 = _mm_unpacklo_epi16(step2[13], step2[10]);
          const __m128i s3_10_1 = _mm_unpackhi_epi16(step2[13], step2[10]);
          const __m128i s3_11_0 = _mm_unpacklo_epi16(step2[12], step2[11]);
          const __m128i s3_11_1 = _mm_unpackhi_epi16(step2[12], step2[11]);
          const __m128i s3_10_2 = _mm_madd_epi16(s3_10_0, k__cospi_p16_m16);
          const __m128i s3_10_3 = _mm_madd_epi16(s3_10_1, k__cospi_p16_m16);
          const __m128i s3_11_2 = _mm_madd_epi16(s3_11_0, k__cospi_p16_m16);
          const __m128i s3_11_3 = _mm_madd_epi16(s3_11_1, k__cospi_p16_m16);
          const __m128i s3_12_2 = _mm_madd_epi16(s3_11_0, k__cospi_p16_p16);
          const __m128i s3_12_3 = _mm_madd_epi16(s3_11_1, k__cospi_p16_p16);
          const __m128i s3_13_2 = _mm_madd_epi16(s3_10_0, k__cospi_p16_p16);
          const __m128i s3_13_3 = _mm_madd_epi16(s3_10_1, k__cospi_p16_p16);
          // dct_const_round_shift
          const __m128i s3_10_4 = _mm_add_epi32(s3_10_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_10_5 = _mm_add_epi32(s3_10_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_11_4 = _mm_add_epi32(s3_11_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_11_5 = _mm_add_epi32(s3_11_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_12_4 = _mm_add_epi32(s3_12_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_12_5 = _mm_add_epi32(s3_12_3, k__DCT_CONST_ROUNDING);
          const __m128i s3_13_4 = _mm_add_epi32(s3_13_2, k__DCT_CONST_ROUNDING);
          const __m128i s3_13_5 = _mm_add_epi32(s3_13_3, k__DCT_CONST_ROUNDING);
          lstep3[20] = _mm_srai_epi32(s3_10_4, DCT_CONST_BITS);
          lstep3[21] = _mm_srai_epi32(s3_10_5, DCT_CONST_BITS);
          lstep3[22] = _mm_srai_epi32(s3_11_4, DCT_CONST_BITS);
          lstep3[23] = _mm_srai_epi32(s3_11_5, DCT_CONST_BITS);
          lstep3[24] = _mm_srai_epi32(s3_12_4, DCT_CONST_BITS);
          lstep3[25] = _mm_srai_epi32(s3_12_5, DCT_CONST_BITS);
          lstep3[26] = _mm_srai_epi32(s3_13_4, DCT_CONST_BITS);
          lstep3[27] = _mm_srai_epi32(s3_13_5, DCT_CONST_BITS);
        }
        {
          lstep1[32] = _mm_unpacklo_epi16(step1[16], step2[23]);
          lstep1[33] = _mm_unpackhi_epi16(step1[16], step2[23]);
          lstep1[34] = _mm_unpacklo_epi16(step1[17], step2[22]);
          lstep1[35] = _mm_unpackhi_epi16(step1[17], step2[22]);
          lstep1[36] = _mm_unpacklo_epi16(step1[18], step2[21]);
          lstep1[37] = _mm_unpackhi_epi16(step1[18], step2[21]);
          lstep1[38] = _mm_unpacklo_epi16(step1[19], step2[20]);
          lstep1[39] = _mm_unpackhi_epi16(step1[19], step2[20]);

          lstep1[56] = _mm_unpacklo_epi16(step1[28], step2[27]);
          lstep1[57] = _mm_unpackhi_epi16(step1[28], step2[27]);
          lstep1[58] = _mm_unpacklo_epi16(step1[29], step2[26]);
          lstep1[59] = _mm_unpackhi_epi16(step1[29], step2[26]);
          lstep1[60] = _mm_unpacklo_epi16(step1[30], step2[25]);
          lstep1[61] = _mm_unpackhi_epi16(step1[30], step2[25]);
          lstep1[62] = _mm_unpacklo_epi16(step1[31], step2[24]);
          lstep1[63] = _mm_unpackhi_epi16(step1[31], step2[24]);

          lstep3[32] = _mm_madd_epi16(lstep1[32], kOne);
          lstep3[33] = _mm_madd_epi16(lstep1[33], kOne);
          lstep3[34] = _mm_madd_epi16(lstep1[34], kOne);
          lstep3[35] = _mm_madd_epi16(lstep1[35], kOne);
          lstep3[36] = _mm_madd_epi16(lstep1[36], kOne);
          lstep3[37] = _mm_madd_epi16(lstep1[37], kOne);
          lstep3[38] = _mm_madd_epi16(lstep1[38], kOne);
          lstep3[39] = _mm_madd_epi16(lstep1[39], kOne);

          lstep3[40] = _mm_madd_epi16(lstep1[38], k__pOne_mOne);
          lstep3[41] = _mm_madd_epi16(lstep1[39], k__pOne_mOne);
          lstep3[42] = _mm_madd_epi16(lstep1[36], k__pOne_mOne);
          lstep3[43] = _mm_madd_epi16(lstep1[37], k__pOne_mOne);
          lstep3[44] = _mm_madd_epi16(lstep1[34], k__pOne_mOne);
          lstep3[45] = _mm_madd_epi16(lstep1[35], k__pOne_mOne);
          lstep3[46] = _mm_madd_epi16(lstep1[32], k__pOne_mOne);
          lstep3[47] = _mm_madd_epi16(lstep1[33], k__pOne_mOne);

          lstep3[48] = _mm_madd_epi16(lstep1[62], k__pOne_mOne);
          lstep3[49] = _mm_madd_epi16(lstep1[63], k__pOne_mOne);
          lstep3[50] = _mm_madd_epi16(lstep1[60], k__pOne_mOne);
          lstep3[51] = _mm_madd_epi16(lstep1[61], k__pOne_mOne);
          lstep3[52] = _mm_madd_epi16(lstep1[58], k__pOne_mOne);
          lstep3[53] = _mm_madd_epi16(lstep1[59], k__pOne_mOne);
          lstep3[54] = _mm_madd_epi16(lstep1[56], k__pOne_mOne);
          lstep3[55] = _mm_madd_epi16(lstep1[57], k__pOne_mOne);

          lstep3[56] = _mm_madd_epi16(lstep1[56], kOne);
          lstep3[57] = _mm_madd_epi16(lstep1[57], kOne);
          lstep3[58] = _mm_madd_epi16(lstep1[58], kOne);
          lstep3[59] = _mm_madd_epi16(lstep1[59], kOne);
          lstep3[60] = _mm_madd_epi16(lstep1[60], kOne);
          lstep3[61] = _mm_madd_epi16(lstep1[61], kOne);
          lstep3[62] = _mm_madd_epi16(lstep1[62], kOne);
          lstep3[63] = _mm_madd_epi16(lstep1[63], kOne);
        }

        // stage 4
        {
          // expanding to 32-bit length prior to addition operations
          sign[0] = _mm_cmpgt_epi16(kZero, step2[8]);
          sign[1] = _mm_cmpgt_epi16(kZero, step2[9]);
          sign[2] = _mm_cmpgt_epi16(kZero, step2[14]);
          sign[3] = _mm_cmpgt_epi16(kZero, step2[15]);
          lstep2[16] = _mm_unpacklo_epi16(step2[8], sign[0]);
          lstep2[17] = _mm_unpackhi_epi16(step2[8], sign[0]);
          lstep2[18] = _mm_unpacklo_epi16(step2[9], sign[1]);
          lstep2[19] = _mm_unpackhi_epi16(step2[9], sign[1]);
          lstep2[28] = _mm_unpacklo_epi16(step2[14], sign[2]);
          lstep2[29] = _mm_unpackhi_epi16(step2[14], sign[2]);
          lstep2[30] = _mm_unpacklo_epi16(step2[15], sign[3]);
          lstep2[31] = _mm_unpackhi_epi16(step2[15], sign[3]);

          lstep1[0] = _mm_add_epi32(lstep3[6], lstep3[0]);
          lstep1[1] = _mm_add_epi32(lstep3[7], lstep3[1]);
          lstep1[2] = _mm_add_epi32(lstep3[4], lstep3[2]);
          lstep1[3] = _mm_add_epi32(lstep3[5], lstep3[3]);
          lstep1[4] = _mm_sub_epi32(lstep3[2], lstep3[4]);
          lstep1[5] = _mm_sub_epi32(lstep3[3], lstep3[5]);
          lstep1[6] = _mm_sub_epi32(lstep3[0], lstep3[6]);
          lstep1[7] = _mm_sub_epi32(lstep3[1], lstep3[7]);
          lstep1[16] = _mm_add_epi32(lstep3[22], lstep2[16]);
          lstep1[17] = _mm_add_epi32(lstep3[23], lstep2[17]);
          lstep1[18] = _mm_add_epi32(lstep3[20], lstep2[18]);
          lstep1[19] = _mm_add_epi32(lstep3[21], lstep2[19]);
          lstep1[20] = _mm_sub_epi32(lstep2[18], lstep3[20]);
          lstep1[21] = _mm_sub_epi32(lstep2[19], lstep3[21]);
          lstep1[22] = _mm_sub_epi32(lstep2[16], lstep3[22]);
          lstep1[23] = _mm_sub_epi32(lstep2[17], lstep3[23]);
          lstep1[24] = _mm_sub_epi32(lstep2[30], lstep3[24]);
          lstep1[25] = _mm_sub_epi32(lstep2[31], lstep3[25]);
          lstep1[26] = _mm_sub_epi32(lstep2[28], lstep3[26]);
          lstep1[27] = _mm_sub_epi32(lstep2[29], lstep3[27]);
          lstep1[28] = _mm_add_epi32(lstep3[26], lstep2[28]);
          lstep1[29] = _mm_add_epi32(lstep3[27], lstep2[29]);
          lstep1[30] = _mm_add_epi32(lstep3[24], lstep2[30]);
          lstep1[31] = _mm_add_epi32(lstep3[25], lstep2[31]);
        }
        {
          // to be continued...
          //
          const __m128i k32_p16_p16 = pair_set_epi32(cospi_16_64, cospi_16_64);
          const __m128i k32_p16_m16 = pair_set_epi32(cospi_16_64, -cospi_16_64);

          u[0] = _mm_unpacklo_epi32(lstep3[12], lstep3[10]);
          u[1] = _mm_unpackhi_epi32(lstep3[12], lstep3[10]);
          u[2] = _mm_unpacklo_epi32(lstep3[13], lstep3[11]);
          u[3] = _mm_unpackhi_epi32(lstep3[13], lstep3[11]);

          // TODO(jingning): manually inline k_madd_epi32_ to further hide
          // instruction latency.
          v[0] = k_madd_epi32(u[0], k32_p16_m16);
          v[1] = k_madd_epi32(u[1], k32_p16_m16);
          v[2] = k_madd_epi32(u[2], k32_p16_m16);
          v[3] = k_madd_epi32(u[3], k32_p16_m16);
          v[4] = k_madd_epi32(u[0], k32_p16_p16);
          v[5] = k_madd_epi32(u[1], k32_p16_p16);
          v[6] = k_madd_epi32(u[2], k32_p16_p16);
          v[7] = k_madd_epi32(u[3], k32_p16_p16);
#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_8(&v[0], &v[1], &v[2], &v[3], &v[4],
                                              &v[5], &v[6], &v[7], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);

          v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);

          lstep1[10] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
          lstep1[11] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
          lstep1[12] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
          lstep1[13] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
        }
        {
          const __m128i k32_m08_p24 = pair_set_epi32(-cospi_8_64, cospi_24_64);
          const __m128i k32_m24_m08 = pair_set_epi32(-cospi_24_64, -cospi_8_64);
          const __m128i k32_p24_p08 = pair_set_epi32(cospi_24_64, cospi_8_64);

          u[0] = _mm_unpacklo_epi32(lstep3[36], lstep3[58]);
          u[1] = _mm_unpackhi_epi32(lstep3[36], lstep3[58]);
          u[2] = _mm_unpacklo_epi32(lstep3[37], lstep3[59]);
          u[3] = _mm_unpackhi_epi32(lstep3[37], lstep3[59]);
          u[4] = _mm_unpacklo_epi32(lstep3[38], lstep3[56]);
          u[5] = _mm_unpackhi_epi32(lstep3[38], lstep3[56]);
          u[6] = _mm_unpacklo_epi32(lstep3[39], lstep3[57]);
          u[7] = _mm_unpackhi_epi32(lstep3[39], lstep3[57]);
          u[8] = _mm_unpacklo_epi32(lstep3[40], lstep3[54]);
          u[9] = _mm_unpackhi_epi32(lstep3[40], lstep3[54]);
          u[10] = _mm_unpacklo_epi32(lstep3[41], lstep3[55]);
          u[11] = _mm_unpackhi_epi32(lstep3[41], lstep3[55]);
          u[12] = _mm_unpacklo_epi32(lstep3[42], lstep3[52]);
          u[13] = _mm_unpackhi_epi32(lstep3[42], lstep3[52]);
          u[14] = _mm_unpacklo_epi32(lstep3[43], lstep3[53]);
          u[15] = _mm_unpackhi_epi32(lstep3[43], lstep3[53]);

          v[0] = k_madd_epi32(u[0], k32_m08_p24);
          v[1] = k_madd_epi32(u[1], k32_m08_p24);
          v[2] = k_madd_epi32(u[2], k32_m08_p24);
          v[3] = k_madd_epi32(u[3], k32_m08_p24);
          v[4] = k_madd_epi32(u[4], k32_m08_p24);
          v[5] = k_madd_epi32(u[5], k32_m08_p24);
          v[6] = k_madd_epi32(u[6], k32_m08_p24);
          v[7] = k_madd_epi32(u[7], k32_m08_p24);
          v[8] = k_madd_epi32(u[8], k32_m24_m08);
          v[9] = k_madd_epi32(u[9], k32_m24_m08);
          v[10] = k_madd_epi32(u[10], k32_m24_m08);
          v[11] = k_madd_epi32(u[11], k32_m24_m08);
          v[12] = k_madd_epi32(u[12], k32_m24_m08);
          v[13] = k_madd_epi32(u[13], k32_m24_m08);
          v[14] = k_madd_epi32(u[14], k32_m24_m08);
          v[15] = k_madd_epi32(u[15], k32_m24_m08);
          v[16] = k_madd_epi32(u[12], k32_m08_p24);
          v[17] = k_madd_epi32(u[13], k32_m08_p24);
          v[18] = k_madd_epi32(u[14], k32_m08_p24);
          v[19] = k_madd_epi32(u[15], k32_m08_p24);
          v[20] = k_madd_epi32(u[8], k32_m08_p24);
          v[21] = k_madd_epi32(u[9], k32_m08_p24);
          v[22] = k_madd_epi32(u[10], k32_m08_p24);
          v[23] = k_madd_epi32(u[11], k32_m08_p24);
          v[24] = k_madd_epi32(u[4], k32_p24_p08);
          v[25] = k_madd_epi32(u[5], k32_p24_p08);
          v[26] = k_madd_epi32(u[6], k32_p24_p08);
          v[27] = k_madd_epi32(u[7], k32_p24_p08);
          v[28] = k_madd_epi32(u[0], k32_p24_p08);
          v[29] = k_madd_epi32(u[1], k32_p24_p08);
          v[30] = k_madd_epi32(u[2], k32_p24_p08);
          v[31] = k_madd_epi32(u[3], k32_p24_p08);

#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_32(
              &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
              &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &v[16],
              &v[17], &v[18], &v[19], &v[20], &v[21], &v[22], &v[23], &v[24],
              &v[25], &v[26], &v[27], &v[28], &v[29], &v[30], &v[31], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);
          u[4] = k_packs_epi64(v[8], v[9]);
          u[5] = k_packs_epi64(v[10], v[11]);
          u[6] = k_packs_epi64(v[12], v[13]);
          u[7] = k_packs_epi64(v[14], v[15]);
          u[8] = k_packs_epi64(v[16], v[17]);
          u[9] = k_packs_epi64(v[18], v[19]);
          u[10] = k_packs_epi64(v[20], v[21]);
          u[11] = k_packs_epi64(v[22], v[23]);
          u[12] = k_packs_epi64(v[24], v[25]);
          u[13] = k_packs_epi64(v[26], v[27]);
          u[14] = k_packs_epi64(v[28], v[29]);
          u[15] = k_packs_epi64(v[30], v[31]);

          v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
          v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
          v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
          v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
          v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);
          v[8] = _mm_add_epi32(u[8], k__DCT_CONST_ROUNDING);
          v[9] = _mm_add_epi32(u[9], k__DCT_CONST_ROUNDING);
          v[10] = _mm_add_epi32(u[10], k__DCT_CONST_ROUNDING);
          v[11] = _mm_add_epi32(u[11], k__DCT_CONST_ROUNDING);
          v[12] = _mm_add_epi32(u[12], k__DCT_CONST_ROUNDING);
          v[13] = _mm_add_epi32(u[13], k__DCT_CONST_ROUNDING);
          v[14] = _mm_add_epi32(u[14], k__DCT_CONST_ROUNDING);
          v[15] = _mm_add_epi32(u[15], k__DCT_CONST_ROUNDING);

          lstep1[36] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
          lstep1[37] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
          lstep1[38] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
          lstep1[39] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
          lstep1[40] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
          lstep1[41] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
          lstep1[42] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
          lstep1[43] = _mm_srai_epi32(v[7], DCT_CONST_BITS);
          lstep1[52] = _mm_srai_epi32(v[8], DCT_CONST_BITS);
          lstep1[53] = _mm_srai_epi32(v[9], DCT_CONST_BITS);
          lstep1[54] = _mm_srai_epi32(v[10], DCT_CONST_BITS);
          lstep1[55] = _mm_srai_epi32(v[11], DCT_CONST_BITS);
          lstep1[56] = _mm_srai_epi32(v[12], DCT_CONST_BITS);
          lstep1[57] = _mm_srai_epi32(v[13], DCT_CONST_BITS);
          lstep1[58] = _mm_srai_epi32(v[14], DCT_CONST_BITS);
          lstep1[59] = _mm_srai_epi32(v[15], DCT_CONST_BITS);
        }
        // stage 5
        {
          lstep2[8] = _mm_add_epi32(lstep1[10], lstep3[8]);
          lstep2[9] = _mm_add_epi32(lstep1[11], lstep3[9]);
          lstep2[10] = _mm_sub_epi32(lstep3[8], lstep1[10]);
          lstep2[11] = _mm_sub_epi32(lstep3[9], lstep1[11]);
          lstep2[12] = _mm_sub_epi32(lstep3[14], lstep1[12]);
          lstep2[13] = _mm_sub_epi32(lstep3[15], lstep1[13]);
          lstep2[14] = _mm_add_epi32(lstep1[12], lstep3[14]);
          lstep2[15] = _mm_add_epi32(lstep1[13], lstep3[15]);
        }
        {
          const __m128i k32_p16_p16 = pair_set_epi32(cospi_16_64, cospi_16_64);
          const __m128i k32_p16_m16 = pair_set_epi32(cospi_16_64, -cospi_16_64);
          const __m128i k32_p24_p08 = pair_set_epi32(cospi_24_64, cospi_8_64);
          const __m128i k32_m08_p24 = pair_set_epi32(-cospi_8_64, cospi_24_64);

          u[0] = _mm_unpacklo_epi32(lstep1[0], lstep1[2]);
          u[1] = _mm_unpackhi_epi32(lstep1[0], lstep1[2]);
          u[2] = _mm_unpacklo_epi32(lstep1[1], lstep1[3]);
          u[3] = _mm_unpackhi_epi32(lstep1[1], lstep1[3]);
          u[4] = _mm_unpacklo_epi32(lstep1[4], lstep1[6]);
          u[5] = _mm_unpackhi_epi32(lstep1[4], lstep1[6]);
          u[6] = _mm_unpacklo_epi32(lstep1[5], lstep1[7]);
          u[7] = _mm_unpackhi_epi32(lstep1[5], lstep1[7]);

          // TODO(jingning): manually inline k_madd_epi32_ to further hide
          // instruction latency.
          v[0] = k_madd_epi32(u[0], k32_p16_p16);
          v[1] = k_madd_epi32(u[1], k32_p16_p16);
          v[2] = k_madd_epi32(u[2], k32_p16_p16);
          v[3] = k_madd_epi32(u[3], k32_p16_p16);
          v[4] = k_madd_epi32(u[0], k32_p16_m16);
          v[5] = k_madd_epi32(u[1], k32_p16_m16);
          v[6] = k_madd_epi32(u[2], k32_p16_m16);
          v[7] = k_madd_epi32(u[3], k32_p16_m16);
          v[8] = k_madd_epi32(u[4], k32_p24_p08);
          v[9] = k_madd_epi32(u[5], k32_p24_p08);
          v[10] = k_madd_epi32(u[6], k32_p24_p08);
          v[11] = k_madd_epi32(u[7], k32_p24_p08);
          v[12] = k_madd_epi32(u[4], k32_m08_p24);
          v[13] = k_madd_epi32(u[5], k32_m08_p24);
          v[14] = k_madd_epi32(u[6], k32_m08_p24);
          v[15] = k_madd_epi32(u[7], k32_m08_p24);

#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_16(
              &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
              &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);
          u[4] = k_packs_epi64(v[8], v[9]);
          u[5] = k_packs_epi64(v[10], v[11]);
          u[6] = k_packs_epi64(v[12], v[13]);
          u[7] = k_packs_epi64(v[14], v[15]);

          v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
          v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
          v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
          v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
          v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);

          u[0] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
          u[1] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
          u[2] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
          u[3] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
          u[4] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
          u[5] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
          u[6] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
          u[7] = _mm_srai_epi32(v[7], DCT_CONST_BITS);

          sign[0] = _mm_cmplt_epi32(u[0], kZero);
          sign[1] = _mm_cmplt_epi32(u[1], kZero);
          sign[2] = _mm_cmplt_epi32(u[2], kZero);
          sign[3] = _mm_cmplt_epi32(u[3], kZero);
          sign[4] = _mm_cmplt_epi32(u[4], kZero);
          sign[5] = _mm_cmplt_epi32(u[5], kZero);
          sign[6] = _mm_cmplt_epi32(u[6], kZero);
          sign[7] = _mm_cmplt_epi32(u[7], kZero);

          u[0] = _mm_sub_epi32(u[0], sign[0]);
          u[1] = _mm_sub_epi32(u[1], sign[1]);
          u[2] = _mm_sub_epi32(u[2], sign[2]);
          u[3] = _mm_sub_epi32(u[3], sign[3]);
          u[4] = _mm_sub_epi32(u[4], sign[4]);
          u[5] = _mm_sub_epi32(u[5], sign[5]);
          u[6] = _mm_sub_epi32(u[6], sign[6]);
          u[7] = _mm_sub_epi32(u[7], sign[7]);

          u[0] = _mm_add_epi32(u[0], K32One);
          u[1] = _mm_add_epi32(u[1], K32One);
          u[2] = _mm_add_epi32(u[2], K32One);
          u[3] = _mm_add_epi32(u[3], K32One);
          u[4] = _mm_add_epi32(u[4], K32One);
          u[5] = _mm_add_epi32(u[5], K32One);
          u[6] = _mm_add_epi32(u[6], K32One);
          u[7] = _mm_add_epi32(u[7], K32One);

          u[0] = _mm_srai_epi32(u[0], 2);
          u[1] = _mm_srai_epi32(u[1], 2);
          u[2] = _mm_srai_epi32(u[2], 2);
          u[3] = _mm_srai_epi32(u[3], 2);
          u[4] = _mm_srai_epi32(u[4], 2);
          u[5] = _mm_srai_epi32(u[5], 2);
          u[6] = _mm_srai_epi32(u[6], 2);
          u[7] = _mm_srai_epi32(u[7], 2);

          // Combine
          out[0] = _mm_packs_epi32(u[0], u[1]);
          out[16] = _mm_packs_epi32(u[2], u[3]);
          out[8] = _mm_packs_epi32(u[4], u[5]);
          out[24] = _mm_packs_epi32(u[6], u[7]);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x4(&out[0], &out[16], &out[8], &out[24]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i k32_m08_p24 = pair_set_epi32(-cospi_8_64, cospi_24_64);
          const __m128i k32_m24_m08 = pair_set_epi32(-cospi_24_64, -cospi_8_64);
          const __m128i k32_p24_p08 = pair_set_epi32(cospi_24_64, cospi_8_64);

          u[0] = _mm_unpacklo_epi32(lstep1[18], lstep1[28]);
          u[1] = _mm_unpackhi_epi32(lstep1[18], lstep1[28]);
          u[2] = _mm_unpacklo_epi32(lstep1[19], lstep1[29]);
          u[3] = _mm_unpackhi_epi32(lstep1[19], lstep1[29]);
          u[4] = _mm_unpacklo_epi32(lstep1[20], lstep1[26]);
          u[5] = _mm_unpackhi_epi32(lstep1[20], lstep1[26]);
          u[6] = _mm_unpacklo_epi32(lstep1[21], lstep1[27]);
          u[7] = _mm_unpackhi_epi32(lstep1[21], lstep1[27]);

          v[0] = k_madd_epi32(u[0], k32_m08_p24);
          v[1] = k_madd_epi32(u[1], k32_m08_p24);
          v[2] = k_madd_epi32(u[2], k32_m08_p24);
          v[3] = k_madd_epi32(u[3], k32_m08_p24);
          v[4] = k_madd_epi32(u[4], k32_m24_m08);
          v[5] = k_madd_epi32(u[5], k32_m24_m08);
          v[6] = k_madd_epi32(u[6], k32_m24_m08);
          v[7] = k_madd_epi32(u[7], k32_m24_m08);
          v[8] = k_madd_epi32(u[4], k32_m08_p24);
          v[9] = k_madd_epi32(u[5], k32_m08_p24);
          v[10] = k_madd_epi32(u[6], k32_m08_p24);
          v[11] = k_madd_epi32(u[7], k32_m08_p24);
          v[12] = k_madd_epi32(u[0], k32_p24_p08);
          v[13] = k_madd_epi32(u[1], k32_p24_p08);
          v[14] = k_madd_epi32(u[2], k32_p24_p08);
          v[15] = k_madd_epi32(u[3], k32_p24_p08);

#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_16(
              &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
              &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);
          u[4] = k_packs_epi64(v[8], v[9]);
          u[5] = k_packs_epi64(v[10], v[11]);
          u[6] = k_packs_epi64(v[12], v[13]);
          u[7] = k_packs_epi64(v[14], v[15]);

          u[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          u[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          u[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          u[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
          u[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
          u[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
          u[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
          u[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);

          lstep2[18] = _mm_srai_epi32(u[0], DCT_CONST_BITS);
          lstep2[19] = _mm_srai_epi32(u[1], DCT_CONST_BITS);
          lstep2[20] = _mm_srai_epi32(u[2], DCT_CONST_BITS);
          lstep2[21] = _mm_srai_epi32(u[3], DCT_CONST_BITS);
          lstep2[26] = _mm_srai_epi32(u[4], DCT_CONST_BITS);
          lstep2[27] = _mm_srai_epi32(u[5], DCT_CONST_BITS);
          lstep2[28] = _mm_srai_epi32(u[6], DCT_CONST_BITS);
          lstep2[29] = _mm_srai_epi32(u[7], DCT_CONST_BITS);
        }
        {
          lstep2[32] = _mm_add_epi32(lstep1[38], lstep3[32]);
          lstep2[33] = _mm_add_epi32(lstep1[39], lstep3[33]);
          lstep2[34] = _mm_add_epi32(lstep1[36], lstep3[34]);
          lstep2[35] = _mm_add_epi32(lstep1[37], lstep3[35]);
          lstep2[36] = _mm_sub_epi32(lstep3[34], lstep1[36]);
          lstep2[37] = _mm_sub_epi32(lstep3[35], lstep1[37]);
          lstep2[38] = _mm_sub_epi32(lstep3[32], lstep1[38]);
          lstep2[39] = _mm_sub_epi32(lstep3[33], lstep1[39]);
          lstep2[40] = _mm_sub_epi32(lstep3[46], lstep1[40]);
          lstep2[41] = _mm_sub_epi32(lstep3[47], lstep1[41]);
          lstep2[42] = _mm_sub_epi32(lstep3[44], lstep1[42]);
          lstep2[43] = _mm_sub_epi32(lstep3[45], lstep1[43]);
          lstep2[44] = _mm_add_epi32(lstep1[42], lstep3[44]);
          lstep2[45] = _mm_add_epi32(lstep1[43], lstep3[45]);
          lstep2[46] = _mm_add_epi32(lstep1[40], lstep3[46]);
          lstep2[47] = _mm_add_epi32(lstep1[41], lstep3[47]);
          lstep2[48] = _mm_add_epi32(lstep1[54], lstep3[48]);
          lstep2[49] = _mm_add_epi32(lstep1[55], lstep3[49]);
          lstep2[50] = _mm_add_epi32(lstep1[52], lstep3[50]);
          lstep2[51] = _mm_add_epi32(lstep1[53], lstep3[51]);
          lstep2[52] = _mm_sub_epi32(lstep3[50], lstep1[52]);
          lstep2[53] = _mm_sub_epi32(lstep3[51], lstep1[53]);
          lstep2[54] = _mm_sub_epi32(lstep3[48], lstep1[54]);
          lstep2[55] = _mm_sub_epi32(lstep3[49], lstep1[55]);
          lstep2[56] = _mm_sub_epi32(lstep3[62], lstep1[56]);
          lstep2[57] = _mm_sub_epi32(lstep3[63], lstep1[57]);
          lstep2[58] = _mm_sub_epi32(lstep3[60], lstep1[58]);
          lstep2[59] = _mm_sub_epi32(lstep3[61], lstep1[59]);
          lstep2[60] = _mm_add_epi32(lstep1[58], lstep3[60]);
          lstep2[61] = _mm_add_epi32(lstep1[59], lstep3[61]);
          lstep2[62] = _mm_add_epi32(lstep1[56], lstep3[62]);
          lstep2[63] = _mm_add_epi32(lstep1[57], lstep3[63]);
        }
        // stage 6
        {
          const __m128i k32_p28_p04 = pair_set_epi32(cospi_28_64, cospi_4_64);
          const __m128i k32_p12_p20 = pair_set_epi32(cospi_12_64, cospi_20_64);
          const __m128i k32_m20_p12 = pair_set_epi32(-cospi_20_64, cospi_12_64);
          const __m128i k32_m04_p28 = pair_set_epi32(-cospi_4_64, cospi_28_64);

          u[0] = _mm_unpacklo_epi32(lstep2[8], lstep2[14]);
          u[1] = _mm_unpackhi_epi32(lstep2[8], lstep2[14]);
          u[2] = _mm_unpacklo_epi32(lstep2[9], lstep2[15]);
          u[3] = _mm_unpackhi_epi32(lstep2[9], lstep2[15]);
          u[4] = _mm_unpacklo_epi32(lstep2[10], lstep2[12]);
          u[5] = _mm_unpackhi_epi32(lstep2[10], lstep2[12]);
          u[6] = _mm_unpacklo_epi32(lstep2[11], lstep2[13]);
          u[7] = _mm_unpackhi_epi32(lstep2[11], lstep2[13]);
          u[8] = _mm_unpacklo_epi32(lstep2[10], lstep2[12]);
          u[9] = _mm_unpackhi_epi32(lstep2[10], lstep2[12]);
          u[10] = _mm_unpacklo_epi32(lstep2[11], lstep2[13]);
          u[11] = _mm_unpackhi_epi32(lstep2[11], lstep2[13]);
          u[12] = _mm_unpacklo_epi32(lstep2[8], lstep2[14]);
          u[13] = _mm_unpackhi_epi32(lstep2[8], lstep2[14]);
          u[14] = _mm_unpacklo_epi32(lstep2[9], lstep2[15]);
          u[15] = _mm_unpackhi_epi32(lstep2[9], lstep2[15]);

          v[0] = k_madd_epi32(u[0], k32_p28_p04);
          v[1] = k_madd_epi32(u[1], k32_p28_p04);
          v[2] = k_madd_epi32(u[2], k32_p28_p04);
          v[3] = k_madd_epi32(u[3], k32_p28_p04);
          v[4] = k_madd_epi32(u[4], k32_p12_p20);
          v[5] = k_madd_epi32(u[5], k32_p12_p20);
          v[6] = k_madd_epi32(u[6], k32_p12_p20);
          v[7] = k_madd_epi32(u[7], k32_p12_p20);
          v[8] = k_madd_epi32(u[8], k32_m20_p12);
          v[9] = k_madd_epi32(u[9], k32_m20_p12);
          v[10] = k_madd_epi32(u[10], k32_m20_p12);
          v[11] = k_madd_epi32(u[11], k32_m20_p12);
          v[12] = k_madd_epi32(u[12], k32_m04_p28);
          v[13] = k_madd_epi32(u[13], k32_m04_p28);
          v[14] = k_madd_epi32(u[14], k32_m04_p28);
          v[15] = k_madd_epi32(u[15], k32_m04_p28);

#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_16(
              &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
              &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);
          u[4] = k_packs_epi64(v[8], v[9]);
          u[5] = k_packs_epi64(v[10], v[11]);
          u[6] = k_packs_epi64(v[12], v[13]);
          u[7] = k_packs_epi64(v[14], v[15]);

          v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
          v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
          v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
          v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
          v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);

          u[0] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
          u[1] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
          u[2] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
          u[3] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
          u[4] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
          u[5] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
          u[6] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
          u[7] = _mm_srai_epi32(v[7], DCT_CONST_BITS);

          sign[0] = _mm_cmplt_epi32(u[0], kZero);
          sign[1] = _mm_cmplt_epi32(u[1], kZero);
          sign[2] = _mm_cmplt_epi32(u[2], kZero);
          sign[3] = _mm_cmplt_epi32(u[3], kZero);
          sign[4] = _mm_cmplt_epi32(u[4], kZero);
          sign[5] = _mm_cmplt_epi32(u[5], kZero);
          sign[6] = _mm_cmplt_epi32(u[6], kZero);
          sign[7] = _mm_cmplt_epi32(u[7], kZero);

          u[0] = _mm_sub_epi32(u[0], sign[0]);
          u[1] = _mm_sub_epi32(u[1], sign[1]);
          u[2] = _mm_sub_epi32(u[2], sign[2]);
          u[3] = _mm_sub_epi32(u[3], sign[3]);
          u[4] = _mm_sub_epi32(u[4], sign[4]);
          u[5] = _mm_sub_epi32(u[5], sign[5]);
          u[6] = _mm_sub_epi32(u[6], sign[6]);
          u[7] = _mm_sub_epi32(u[7], sign[7]);

          u[0] = _mm_add_epi32(u[0], K32One);
          u[1] = _mm_add_epi32(u[1], K32One);
          u[2] = _mm_add_epi32(u[2], K32One);
          u[3] = _mm_add_epi32(u[3], K32One);
          u[4] = _mm_add_epi32(u[4], K32One);
          u[5] = _mm_add_epi32(u[5], K32One);
          u[6] = _mm_add_epi32(u[6], K32One);
          u[7] = _mm_add_epi32(u[7], K32One);

          u[0] = _mm_srai_epi32(u[0], 2);
          u[1] = _mm_srai_epi32(u[1], 2);
          u[2] = _mm_srai_epi32(u[2], 2);
          u[3] = _mm_srai_epi32(u[3], 2);
          u[4] = _mm_srai_epi32(u[4], 2);
          u[5] = _mm_srai_epi32(u[5], 2);
          u[6] = _mm_srai_epi32(u[6], 2);
          u[7] = _mm_srai_epi32(u[7], 2);

          out[4] = _mm_packs_epi32(u[0], u[1]);
          out[20] = _mm_packs_epi32(u[2], u[3]);
          out[12] = _mm_packs_epi32(u[4], u[5]);
          out[28] = _mm_packs_epi32(u[6], u[7]);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x4(&out[4], &out[20], &out[12], &out[28]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          lstep3[16] = _mm_add_epi32(lstep2[18], lstep1[16]);
          lstep3[17] = _mm_add_epi32(lstep2[19], lstep1[17]);
          lstep3[18] = _mm_sub_epi32(lstep1[16], lstep2[18]);
          lstep3[19] = _mm_sub_epi32(lstep1[17], lstep2[19]);
          lstep3[20] = _mm_sub_epi32(lstep1[22], lstep2[20]);
          lstep3[21] = _mm_sub_epi32(lstep1[23], lstep2[21]);
          lstep3[22] = _mm_add_epi32(lstep2[20], lstep1[22]);
          lstep3[23] = _mm_add_epi32(lstep2[21], lstep1[23]);
          lstep3[24] = _mm_add_epi32(lstep2[26], lstep1[24]);
          lstep3[25] = _mm_add_epi32(lstep2[27], lstep1[25]);
          lstep3[26] = _mm_sub_epi32(lstep1[24], lstep2[26]);
          lstep3[27] = _mm_sub_epi32(lstep1[25], lstep2[27]);
          lstep3[28] = _mm_sub_epi32(lstep1[30], lstep2[28]);
          lstep3[29] = _mm_sub_epi32(lstep1[31], lstep2[29]);
          lstep3[30] = _mm_add_epi32(lstep2[28], lstep1[30]);
          lstep3[31] = _mm_add_epi32(lstep2[29], lstep1[31]);
        }
        {
          const __m128i k32_m04_p28 = pair_set_epi32(-cospi_4_64, cospi_28_64);
          const __m128i k32_m28_m04 = pair_set_epi32(-cospi_28_64, -cospi_4_64);
          const __m128i k32_m20_p12 = pair_set_epi32(-cospi_20_64, cospi_12_64);
          const __m128i k32_m12_m20 =
              pair_set_epi32(-cospi_12_64, -cospi_20_64);
          const __m128i k32_p12_p20 = pair_set_epi32(cospi_12_64, cospi_20_64);
          const __m128i k32_p28_p04 = pair_set_epi32(cospi_28_64, cospi_4_64);

          u[0] = _mm_unpacklo_epi32(lstep2[34], lstep2[60]);
          u[1] = _mm_unpackhi_epi32(lstep2[34], lstep2[60]);
          u[2] = _mm_unpacklo_epi32(lstep2[35], lstep2[61]);
          u[3] = _mm_unpackhi_epi32(lstep2[35], lstep2[61]);
          u[4] = _mm_unpacklo_epi32(lstep2[36], lstep2[58]);
          u[5] = _mm_unpackhi_epi32(lstep2[36], lstep2[58]);
          u[6] = _mm_unpacklo_epi32(lstep2[37], lstep2[59]);
          u[7] = _mm_unpackhi_epi32(lstep2[37], lstep2[59]);
          u[8] = _mm_unpacklo_epi32(lstep2[42], lstep2[52]);
          u[9] = _mm_unpackhi_epi32(lstep2[42], lstep2[52]);
          u[10] = _mm_unpacklo_epi32(lstep2[43], lstep2[53]);
          u[11] = _mm_unpackhi_epi32(lstep2[43], lstep2[53]);
          u[12] = _mm_unpacklo_epi32(lstep2[44], lstep2[50]);
          u[13] = _mm_unpackhi_epi32(lstep2[44], lstep2[50]);
          u[14] = _mm_unpacklo_epi32(lstep2[45], lstep2[51]);
          u[15] = _mm_unpackhi_epi32(lstep2[45], lstep2[51]);

          v[0] = k_madd_epi32(u[0], k32_m04_p28);
          v[1] = k_madd_epi32(u[1], k32_m04_p28);
          v[2] = k_madd_epi32(u[2], k32_m04_p28);
          v[3] = k_madd_epi32(u[3], k32_m04_p28);
          v[4] = k_madd_epi32(u[4], k32_m28_m04);
          v[5] = k_madd_epi32(u[5], k32_m28_m04);
          v[6] = k_madd_epi32(u[6], k32_m28_m04);
          v[7] = k_madd_epi32(u[7], k32_m28_m04);
          v[8] = k_madd_epi32(u[8], k32_m20_p12);
          v[9] = k_madd_epi32(u[9], k32_m20_p12);
          v[10] = k_madd_epi32(u[10], k32_m20_p12);
          v[11] = k_madd_epi32(u[11], k32_m20_p12);
          v[12] = k_madd_epi32(u[12], k32_m12_m20);
          v[13] = k_madd_epi32(u[13], k32_m12_m20);
          v[14] = k_madd_epi32(u[14], k32_m12_m20);
          v[15] = k_madd_epi32(u[15], k32_m12_m20);
          v[16] = k_madd_epi32(u[12], k32_m20_p12);
          v[17] = k_madd_epi32(u[13], k32_m20_p12);
          v[18] = k_madd_epi32(u[14], k32_m20_p12);
          v[19] = k_madd_epi32(u[15], k32_m20_p12);
          v[20] = k_madd_epi32(u[8], k32_p12_p20);
          v[21] = k_madd_epi32(u[9], k32_p12_p20);
          v[22] = k_madd_epi32(u[10], k32_p12_p20);
          v[23] = k_madd_epi32(u[11], k32_p12_p20);
          v[24] = k_madd_epi32(u[4], k32_m04_p28);
          v[25] = k_madd_epi32(u[5], k32_m04_p28);
          v[26] = k_madd_epi32(u[6], k32_m04_p28);
          v[27] = k_madd_epi32(u[7], k32_m04_p28);
          v[28] = k_madd_epi32(u[0], k32_p28_p04);
          v[29] = k_madd_epi32(u[1], k32_p28_p04);
          v[30] = k_madd_epi32(u[2], k32_p28_p04);
          v[31] = k_madd_epi32(u[3], k32_p28_p04);

#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_32(
              &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
              &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &v[16],
              &v[17], &v[18], &v[19], &v[20], &v[21], &v[22], &v[23], &v[24],
              &v[25], &v[26], &v[27], &v[28], &v[29], &v[30], &v[31], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);
          u[4] = k_packs_epi64(v[8], v[9]);
          u[5] = k_packs_epi64(v[10], v[11]);
          u[6] = k_packs_epi64(v[12], v[13]);
          u[7] = k_packs_epi64(v[14], v[15]);
          u[8] = k_packs_epi64(v[16], v[17]);
          u[9] = k_packs_epi64(v[18], v[19]);
          u[10] = k_packs_epi64(v[20], v[21]);
          u[11] = k_packs_epi64(v[22], v[23]);
          u[12] = k_packs_epi64(v[24], v[25]);
          u[13] = k_packs_epi64(v[26], v[27]);
          u[14] = k_packs_epi64(v[28], v[29]);
          u[15] = k_packs_epi64(v[30], v[31]);

          v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
          v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
          v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
          v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
          v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);
          v[8] = _mm_add_epi32(u[8], k__DCT_CONST_ROUNDING);
          v[9] = _mm_add_epi32(u[9], k__DCT_CONST_ROUNDING);
          v[10] = _mm_add_epi32(u[10], k__DCT_CONST_ROUNDING);
          v[11] = _mm_add_epi32(u[11], k__DCT_CONST_ROUNDING);
          v[12] = _mm_add_epi32(u[12], k__DCT_CONST_ROUNDING);
          v[13] = _mm_add_epi32(u[13], k__DCT_CONST_ROUNDING);
          v[14] = _mm_add_epi32(u[14], k__DCT_CONST_ROUNDING);
          v[15] = _mm_add_epi32(u[15], k__DCT_CONST_ROUNDING);

          lstep3[34] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
          lstep3[35] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
          lstep3[36] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
          lstep3[37] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
          lstep3[42] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
          lstep3[43] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
          lstep3[44] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
          lstep3[45] = _mm_srai_epi32(v[7], DCT_CONST_BITS);
          lstep3[50] = _mm_srai_epi32(v[8], DCT_CONST_BITS);
          lstep3[51] = _mm_srai_epi32(v[9], DCT_CONST_BITS);
          lstep3[52] = _mm_srai_epi32(v[10], DCT_CONST_BITS);
          lstep3[53] = _mm_srai_epi32(v[11], DCT_CONST_BITS);
          lstep3[58] = _mm_srai_epi32(v[12], DCT_CONST_BITS);
          lstep3[59] = _mm_srai_epi32(v[13], DCT_CONST_BITS);
          lstep3[60] = _mm_srai_epi32(v[14], DCT_CONST_BITS);
          lstep3[61] = _mm_srai_epi32(v[15], DCT_CONST_BITS);
        }
        // stage 7
        {
          const __m128i k32_p30_p02 = pair_set_epi32(cospi_30_64, cospi_2_64);
          const __m128i k32_p14_p18 = pair_set_epi32(cospi_14_64, cospi_18_64);
          const __m128i k32_p22_p10 = pair_set_epi32(cospi_22_64, cospi_10_64);
          const __m128i k32_p06_p26 = pair_set_epi32(cospi_6_64, cospi_26_64);
          const __m128i k32_m26_p06 = pair_set_epi32(-cospi_26_64, cospi_6_64);
          const __m128i k32_m10_p22 = pair_set_epi32(-cospi_10_64, cospi_22_64);
          const __m128i k32_m18_p14 = pair_set_epi32(-cospi_18_64, cospi_14_64);
          const __m128i k32_m02_p30 = pair_set_epi32(-cospi_2_64, cospi_30_64);

          u[0] = _mm_unpacklo_epi32(lstep3[16], lstep3[30]);
          u[1] = _mm_unpackhi_epi32(lstep3[16], lstep3[30]);
          u[2] = _mm_unpacklo_epi32(lstep3[17], lstep3[31]);
          u[3] = _mm_unpackhi_epi32(lstep3[17], lstep3[31]);
          u[4] = _mm_unpacklo_epi32(lstep3[18], lstep3[28]);
          u[5] = _mm_unpackhi_epi32(lstep3[18], lstep3[28]);
          u[6] = _mm_unpacklo_epi32(lstep3[19], lstep3[29]);
          u[7] = _mm_unpackhi_epi32(lstep3[19], lstep3[29]);
          u[8] = _mm_unpacklo_epi32(lstep3[20], lstep3[26]);
          u[9] = _mm_unpackhi_epi32(lstep3[20], lstep3[26]);
          u[10] = _mm_unpacklo_epi32(lstep3[21], lstep3[27]);
          u[11] = _mm_unpackhi_epi32(lstep3[21], lstep3[27]);
          u[12] = _mm_unpacklo_epi32(lstep3[22], lstep3[24]);
          u[13] = _mm_unpackhi_epi32(lstep3[22], lstep3[24]);
          u[14] = _mm_unpacklo_epi32(lstep3[23], lstep3[25]);
          u[15] = _mm_unpackhi_epi32(lstep3[23], lstep3[25]);

          v[0] = k_madd_epi32(u[0], k32_p30_p02);
          v[1] = k_madd_epi32(u[1], k32_p30_p02);
          v[2] = k_madd_epi32(u[2], k32_p30_p02);
          v[3] = k_madd_epi32(u[3], k32_p30_p02);
          v[4] = k_madd_epi32(u[4], k32_p14_p18);
          v[5] = k_madd_epi32(u[5], k32_p14_p18);
          v[6] = k_madd_epi32(u[6], k32_p14_p18);
          v[7] = k_madd_epi32(u[7], k32_p14_p18);
          v[8] = k_madd_epi32(u[8], k32_p22_p10);
          v[9] = k_madd_epi32(u[9], k32_p22_p10);
          v[10] = k_madd_epi32(u[10], k32_p22_p10);
          v[11] = k_madd_epi32(u[11], k32_p22_p10);
          v[12] = k_madd_epi32(u[12], k32_p06_p26);
          v[13] = k_madd_epi32(u[13], k32_p06_p26);
          v[14] = k_madd_epi32(u[14], k32_p06_p26);
          v[15] = k_madd_epi32(u[15], k32_p06_p26);
          v[16] = k_madd_epi32(u[12], k32_m26_p06);
          v[17] = k_madd_epi32(u[13], k32_m26_p06);
          v[18] = k_madd_epi32(u[14], k32_m26_p06);
          v[19] = k_madd_epi32(u[15], k32_m26_p06);
          v[20] = k_madd_epi32(u[8], k32_m10_p22);
          v[21] = k_madd_epi32(u[9], k32_m10_p22);
          v[22] = k_madd_epi32(u[10], k32_m10_p22);
          v[23] = k_madd_epi32(u[11], k32_m10_p22);
          v[24] = k_madd_epi32(u[4], k32_m18_p14);
          v[25] = k_madd_epi32(u[5], k32_m18_p14);
          v[26] = k_madd_epi32(u[6], k32_m18_p14);
          v[27] = k_madd_epi32(u[7], k32_m18_p14);
          v[28] = k_madd_epi32(u[0], k32_m02_p30);
          v[29] = k_madd_epi32(u[1], k32_m02_p30);
          v[30] = k_madd_epi32(u[2], k32_m02_p30);
          v[31] = k_madd_epi32(u[3], k32_m02_p30);

#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_32(
              &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
              &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &v[16],
              &v[17], &v[18], &v[19], &v[20], &v[21], &v[22], &v[23], &v[24],
              &v[25], &v[26], &v[27], &v[28], &v[29], &v[30], &v[31], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);
          u[4] = k_packs_epi64(v[8], v[9]);
          u[5] = k_packs_epi64(v[10], v[11]);
          u[6] = k_packs_epi64(v[12], v[13]);
          u[7] = k_packs_epi64(v[14], v[15]);
          u[8] = k_packs_epi64(v[16], v[17]);
          u[9] = k_packs_epi64(v[18], v[19]);
          u[10] = k_packs_epi64(v[20], v[21]);
          u[11] = k_packs_epi64(v[22], v[23]);
          u[12] = k_packs_epi64(v[24], v[25]);
          u[13] = k_packs_epi64(v[26], v[27]);
          u[14] = k_packs_epi64(v[28], v[29]);
          u[15] = k_packs_epi64(v[30], v[31]);

          v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
          v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
          v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
          v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
          v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);
          v[8] = _mm_add_epi32(u[8], k__DCT_CONST_ROUNDING);
          v[9] = _mm_add_epi32(u[9], k__DCT_CONST_ROUNDING);
          v[10] = _mm_add_epi32(u[10], k__DCT_CONST_ROUNDING);
          v[11] = _mm_add_epi32(u[11], k__DCT_CONST_ROUNDING);
          v[12] = _mm_add_epi32(u[12], k__DCT_CONST_ROUNDING);
          v[13] = _mm_add_epi32(u[13], k__DCT_CONST_ROUNDING);
          v[14] = _mm_add_epi32(u[14], k__DCT_CONST_ROUNDING);
          v[15] = _mm_add_epi32(u[15], k__DCT_CONST_ROUNDING);

          u[0] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
          u[1] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
          u[2] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
          u[3] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
          u[4] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
          u[5] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
          u[6] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
          u[7] = _mm_srai_epi32(v[7], DCT_CONST_BITS);
          u[8] = _mm_srai_epi32(v[8], DCT_CONST_BITS);
          u[9] = _mm_srai_epi32(v[9], DCT_CONST_BITS);
          u[10] = _mm_srai_epi32(v[10], DCT_CONST_BITS);
          u[11] = _mm_srai_epi32(v[11], DCT_CONST_BITS);
          u[12] = _mm_srai_epi32(v[12], DCT_CONST_BITS);
          u[13] = _mm_srai_epi32(v[13], DCT_CONST_BITS);
          u[14] = _mm_srai_epi32(v[14], DCT_CONST_BITS);
          u[15] = _mm_srai_epi32(v[15], DCT_CONST_BITS);

          v[0] = _mm_cmplt_epi32(u[0], kZero);
          v[1] = _mm_cmplt_epi32(u[1], kZero);
          v[2] = _mm_cmplt_epi32(u[2], kZero);
          v[3] = _mm_cmplt_epi32(u[3], kZero);
          v[4] = _mm_cmplt_epi32(u[4], kZero);
          v[5] = _mm_cmplt_epi32(u[5], kZero);
          v[6] = _mm_cmplt_epi32(u[6], kZero);
          v[7] = _mm_cmplt_epi32(u[7], kZero);
          v[8] = _mm_cmplt_epi32(u[8], kZero);
          v[9] = _mm_cmplt_epi32(u[9], kZero);
          v[10] = _mm_cmplt_epi32(u[10], kZero);
          v[11] = _mm_cmplt_epi32(u[11], kZero);
          v[12] = _mm_cmplt_epi32(u[12], kZero);
          v[13] = _mm_cmplt_epi32(u[13], kZero);
          v[14] = _mm_cmplt_epi32(u[14], kZero);
          v[15] = _mm_cmplt_epi32(u[15], kZero);

          u[0] = _mm_sub_epi32(u[0], v[0]);
          u[1] = _mm_sub_epi32(u[1], v[1]);
          u[2] = _mm_sub_epi32(u[2], v[2]);
          u[3] = _mm_sub_epi32(u[3], v[3]);
          u[4] = _mm_sub_epi32(u[4], v[4]);
          u[5] = _mm_sub_epi32(u[5], v[5]);
          u[6] = _mm_sub_epi32(u[6], v[6]);
          u[7] = _mm_sub_epi32(u[7], v[7]);
          u[8] = _mm_sub_epi32(u[8], v[8]);
          u[9] = _mm_sub_epi32(u[9], v[9]);
          u[10] = _mm_sub_epi32(u[10], v[10]);
          u[11] = _mm_sub_epi32(u[11], v[11]);
          u[12] = _mm_sub_epi32(u[12], v[12]);
          u[13] = _mm_sub_epi32(u[13], v[13]);
          u[14] = _mm_sub_epi32(u[14], v[14]);
          u[15] = _mm_sub_epi32(u[15], v[15]);

          v[0] = _mm_add_epi32(u[0], K32One);
          v[1] = _mm_add_epi32(u[1], K32One);
          v[2] = _mm_add_epi32(u[2], K32One);
          v[3] = _mm_add_epi32(u[3], K32One);
          v[4] = _mm_add_epi32(u[4], K32One);
          v[5] = _mm_add_epi32(u[5], K32One);
          v[6] = _mm_add_epi32(u[6], K32One);
          v[7] = _mm_add_epi32(u[7], K32One);
          v[8] = _mm_add_epi32(u[8], K32One);
          v[9] = _mm_add_epi32(u[9], K32One);
          v[10] = _mm_add_epi32(u[10], K32One);
          v[11] = _mm_add_epi32(u[11], K32One);
          v[12] = _mm_add_epi32(u[12], K32One);
          v[13] = _mm_add_epi32(u[13], K32One);
          v[14] = _mm_add_epi32(u[14], K32One);
          v[15] = _mm_add_epi32(u[15], K32One);

          u[0] = _mm_srai_epi32(v[0], 2);
          u[1] = _mm_srai_epi32(v[1], 2);
          u[2] = _mm_srai_epi32(v[2], 2);
          u[3] = _mm_srai_epi32(v[3], 2);
          u[4] = _mm_srai_epi32(v[4], 2);
          u[5] = _mm_srai_epi32(v[5], 2);
          u[6] = _mm_srai_epi32(v[6], 2);
          u[7] = _mm_srai_epi32(v[7], 2);
          u[8] = _mm_srai_epi32(v[8], 2);
          u[9] = _mm_srai_epi32(v[9], 2);
          u[10] = _mm_srai_epi32(v[10], 2);
          u[11] = _mm_srai_epi32(v[11], 2);
          u[12] = _mm_srai_epi32(v[12], 2);
          u[13] = _mm_srai_epi32(v[13], 2);
          u[14] = _mm_srai_epi32(v[14], 2);
          u[15] = _mm_srai_epi32(v[15], 2);

          out[2] = _mm_packs_epi32(u[0], u[1]);
          out[18] = _mm_packs_epi32(u[2], u[3]);
          out[10] = _mm_packs_epi32(u[4], u[5]);
          out[26] = _mm_packs_epi32(u[6], u[7]);
          out[6] = _mm_packs_epi32(u[8], u[9]);
          out[22] = _mm_packs_epi32(u[10], u[11]);
          out[14] = _mm_packs_epi32(u[12], u[13]);
          out[30] = _mm_packs_epi32(u[14], u[15]);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&out[2], &out[18], &out[10], &out[26],
                                      &out[6], &out[22], &out[14], &out[30]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          lstep1[32] = _mm_add_epi32(lstep3[34], lstep2[32]);
          lstep1[33] = _mm_add_epi32(lstep3[35], lstep2[33]);
          lstep1[34] = _mm_sub_epi32(lstep2[32], lstep3[34]);
          lstep1[35] = _mm_sub_epi32(lstep2[33], lstep3[35]);
          lstep1[36] = _mm_sub_epi32(lstep2[38], lstep3[36]);
          lstep1[37] = _mm_sub_epi32(lstep2[39], lstep3[37]);
          lstep1[38] = _mm_add_epi32(lstep3[36], lstep2[38]);
          lstep1[39] = _mm_add_epi32(lstep3[37], lstep2[39]);
          lstep1[40] = _mm_add_epi32(lstep3[42], lstep2[40]);
          lstep1[41] = _mm_add_epi32(lstep3[43], lstep2[41]);
          lstep1[42] = _mm_sub_epi32(lstep2[40], lstep3[42]);
          lstep1[43] = _mm_sub_epi32(lstep2[41], lstep3[43]);
          lstep1[44] = _mm_sub_epi32(lstep2[46], lstep3[44]);
          lstep1[45] = _mm_sub_epi32(lstep2[47], lstep3[45]);
          lstep1[46] = _mm_add_epi32(lstep3[44], lstep2[46]);
          lstep1[47] = _mm_add_epi32(lstep3[45], lstep2[47]);
          lstep1[48] = _mm_add_epi32(lstep3[50], lstep2[48]);
          lstep1[49] = _mm_add_epi32(lstep3[51], lstep2[49]);
          lstep1[50] = _mm_sub_epi32(lstep2[48], lstep3[50]);
          lstep1[51] = _mm_sub_epi32(lstep2[49], lstep3[51]);
          lstep1[52] = _mm_sub_epi32(lstep2[54], lstep3[52]);
          lstep1[53] = _mm_sub_epi32(lstep2[55], lstep3[53]);
          lstep1[54] = _mm_add_epi32(lstep3[52], lstep2[54]);
          lstep1[55] = _mm_add_epi32(lstep3[53], lstep2[55]);
          lstep1[56] = _mm_add_epi32(lstep3[58], lstep2[56]);
          lstep1[57] = _mm_add_epi32(lstep3[59], lstep2[57]);
          lstep1[58] = _mm_sub_epi32(lstep2[56], lstep3[58]);
          lstep1[59] = _mm_sub_epi32(lstep2[57], lstep3[59]);
          lstep1[60] = _mm_sub_epi32(lstep2[62], lstep3[60]);
          lstep1[61] = _mm_sub_epi32(lstep2[63], lstep3[61]);
          lstep1[62] = _mm_add_epi32(lstep3[60], lstep2[62]);
          lstep1[63] = _mm_add_epi32(lstep3[61], lstep2[63]);
        }
        // stage 8
        {
          const __m128i k32_p31_p01 = pair_set_epi32(cospi_31_64, cospi_1_64);
          const __m128i k32_p15_p17 = pair_set_epi32(cospi_15_64, cospi_17_64);
          const __m128i k32_p23_p09 = pair_set_epi32(cospi_23_64, cospi_9_64);
          const __m128i k32_p07_p25 = pair_set_epi32(cospi_7_64, cospi_25_64);
          const __m128i k32_m25_p07 = pair_set_epi32(-cospi_25_64, cospi_7_64);
          const __m128i k32_m09_p23 = pair_set_epi32(-cospi_9_64, cospi_23_64);
          const __m128i k32_m17_p15 = pair_set_epi32(-cospi_17_64, cospi_15_64);
          const __m128i k32_m01_p31 = pair_set_epi32(-cospi_1_64, cospi_31_64);

          u[0] = _mm_unpacklo_epi32(lstep1[32], lstep1[62]);
          u[1] = _mm_unpackhi_epi32(lstep1[32], lstep1[62]);
          u[2] = _mm_unpacklo_epi32(lstep1[33], lstep1[63]);
          u[3] = _mm_unpackhi_epi32(lstep1[33], lstep1[63]);
          u[4] = _mm_unpacklo_epi32(lstep1[34], lstep1[60]);
          u[5] = _mm_unpackhi_epi32(lstep1[34], lstep1[60]);
          u[6] = _mm_unpacklo_epi32(lstep1[35], lstep1[61]);
          u[7] = _mm_unpackhi_epi32(lstep1[35], lstep1[61]);
          u[8] = _mm_unpacklo_epi32(lstep1[36], lstep1[58]);
          u[9] = _mm_unpackhi_epi32(lstep1[36], lstep1[58]);
          u[10] = _mm_unpacklo_epi32(lstep1[37], lstep1[59]);
          u[11] = _mm_unpackhi_epi32(lstep1[37], lstep1[59]);
          u[12] = _mm_unpacklo_epi32(lstep1[38], lstep1[56]);
          u[13] = _mm_unpackhi_epi32(lstep1[38], lstep1[56]);
          u[14] = _mm_unpacklo_epi32(lstep1[39], lstep1[57]);
          u[15] = _mm_unpackhi_epi32(lstep1[39], lstep1[57]);

          v[0] = k_madd_epi32(u[0], k32_p31_p01);
          v[1] = k_madd_epi32(u[1], k32_p31_p01);
          v[2] = k_madd_epi32(u[2], k32_p31_p01);
          v[3] = k_madd_epi32(u[3], k32_p31_p01);
          v[4] = k_madd_epi32(u[4], k32_p15_p17);
          v[5] = k_madd_epi32(u[5], k32_p15_p17);
          v[6] = k_madd_epi32(u[6], k32_p15_p17);
          v[7] = k_madd_epi32(u[7], k32_p15_p17);
          v[8] = k_madd_epi32(u[8], k32_p23_p09);
          v[9] = k_madd_epi32(u[9], k32_p23_p09);
          v[10] = k_madd_epi32(u[10], k32_p23_p09);
          v[11] = k_madd_epi32(u[11], k32_p23_p09);
          v[12] = k_madd_epi32(u[12], k32_p07_p25);
          v[13] = k_madd_epi32(u[13], k32_p07_p25);
          v[14] = k_madd_epi32(u[14], k32_p07_p25);
          v[15] = k_madd_epi32(u[15], k32_p07_p25);
          v[16] = k_madd_epi32(u[12], k32_m25_p07);
          v[17] = k_madd_epi32(u[13], k32_m25_p07);
          v[18] = k_madd_epi32(u[14], k32_m25_p07);
          v[19] = k_madd_epi32(u[15], k32_m25_p07);
          v[20] = k_madd_epi32(u[8], k32_m09_p23);
          v[21] = k_madd_epi32(u[9], k32_m09_p23);
          v[22] = k_madd_epi32(u[10], k32_m09_p23);
          v[23] = k_madd_epi32(u[11], k32_m09_p23);
          v[24] = k_madd_epi32(u[4], k32_m17_p15);
          v[25] = k_madd_epi32(u[5], k32_m17_p15);
          v[26] = k_madd_epi32(u[6], k32_m17_p15);
          v[27] = k_madd_epi32(u[7], k32_m17_p15);
          v[28] = k_madd_epi32(u[0], k32_m01_p31);
          v[29] = k_madd_epi32(u[1], k32_m01_p31);
          v[30] = k_madd_epi32(u[2], k32_m01_p31);
          v[31] = k_madd_epi32(u[3], k32_m01_p31);

#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_32(
              &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
              &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &v[16],
              &v[17], &v[18], &v[19], &v[20], &v[21], &v[22], &v[23], &v[24],
              &v[25], &v[26], &v[27], &v[28], &v[29], &v[30], &v[31], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);
          u[4] = k_packs_epi64(v[8], v[9]);
          u[5] = k_packs_epi64(v[10], v[11]);
          u[6] = k_packs_epi64(v[12], v[13]);
          u[7] = k_packs_epi64(v[14], v[15]);
          u[8] = k_packs_epi64(v[16], v[17]);
          u[9] = k_packs_epi64(v[18], v[19]);
          u[10] = k_packs_epi64(v[20], v[21]);
          u[11] = k_packs_epi64(v[22], v[23]);
          u[12] = k_packs_epi64(v[24], v[25]);
          u[13] = k_packs_epi64(v[26], v[27]);
          u[14] = k_packs_epi64(v[28], v[29]);
          u[15] = k_packs_epi64(v[30], v[31]);

          v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
          v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
          v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
          v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
          v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);
          v[8] = _mm_add_epi32(u[8], k__DCT_CONST_ROUNDING);
          v[9] = _mm_add_epi32(u[9], k__DCT_CONST_ROUNDING);
          v[10] = _mm_add_epi32(u[10], k__DCT_CONST_ROUNDING);
          v[11] = _mm_add_epi32(u[11], k__DCT_CONST_ROUNDING);
          v[12] = _mm_add_epi32(u[12], k__DCT_CONST_ROUNDING);
          v[13] = _mm_add_epi32(u[13], k__DCT_CONST_ROUNDING);
          v[14] = _mm_add_epi32(u[14], k__DCT_CONST_ROUNDING);
          v[15] = _mm_add_epi32(u[15], k__DCT_CONST_ROUNDING);

          u[0] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
          u[1] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
          u[2] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
          u[3] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
          u[4] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
          u[5] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
          u[6] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
          u[7] = _mm_srai_epi32(v[7], DCT_CONST_BITS);
          u[8] = _mm_srai_epi32(v[8], DCT_CONST_BITS);
          u[9] = _mm_srai_epi32(v[9], DCT_CONST_BITS);
          u[10] = _mm_srai_epi32(v[10], DCT_CONST_BITS);
          u[11] = _mm_srai_epi32(v[11], DCT_CONST_BITS);
          u[12] = _mm_srai_epi32(v[12], DCT_CONST_BITS);
          u[13] = _mm_srai_epi32(v[13], DCT_CONST_BITS);
          u[14] = _mm_srai_epi32(v[14], DCT_CONST_BITS);
          u[15] = _mm_srai_epi32(v[15], DCT_CONST_BITS);

          v[0] = _mm_cmplt_epi32(u[0], kZero);
          v[1] = _mm_cmplt_epi32(u[1], kZero);
          v[2] = _mm_cmplt_epi32(u[2], kZero);
          v[3] = _mm_cmplt_epi32(u[3], kZero);
          v[4] = _mm_cmplt_epi32(u[4], kZero);
          v[5] = _mm_cmplt_epi32(u[5], kZero);
          v[6] = _mm_cmplt_epi32(u[6], kZero);
          v[7] = _mm_cmplt_epi32(u[7], kZero);
          v[8] = _mm_cmplt_epi32(u[8], kZero);
          v[9] = _mm_cmplt_epi32(u[9], kZero);
          v[10] = _mm_cmplt_epi32(u[10], kZero);
          v[11] = _mm_cmplt_epi32(u[11], kZero);
          v[12] = _mm_cmplt_epi32(u[12], kZero);
          v[13] = _mm_cmplt_epi32(u[13], kZero);
          v[14] = _mm_cmplt_epi32(u[14], kZero);
          v[15] = _mm_cmplt_epi32(u[15], kZero);

          u[0] = _mm_sub_epi32(u[0], v[0]);
          u[1] = _mm_sub_epi32(u[1], v[1]);
          u[2] = _mm_sub_epi32(u[2], v[2]);
          u[3] = _mm_sub_epi32(u[3], v[3]);
          u[4] = _mm_sub_epi32(u[4], v[4]);
          u[5] = _mm_sub_epi32(u[5], v[5]);
          u[6] = _mm_sub_epi32(u[6], v[6]);
          u[7] = _mm_sub_epi32(u[7], v[7]);
          u[8] = _mm_sub_epi32(u[8], v[8]);
          u[9] = _mm_sub_epi32(u[9], v[9]);
          u[10] = _mm_sub_epi32(u[10], v[10]);
          u[11] = _mm_sub_epi32(u[11], v[11]);
          u[12] = _mm_sub_epi32(u[12], v[12]);
          u[13] = _mm_sub_epi32(u[13], v[13]);
          u[14] = _mm_sub_epi32(u[14], v[14]);
          u[15] = _mm_sub_epi32(u[15], v[15]);

          v[0] = _mm_add_epi32(u[0], K32One);
          v[1] = _mm_add_epi32(u[1], K32One);
          v[2] = _mm_add_epi32(u[2], K32One);
          v[3] = _mm_add_epi32(u[3], K32One);
          v[4] = _mm_add_epi32(u[4], K32One);
          v[5] = _mm_add_epi32(u[5], K32One);
          v[6] = _mm_add_epi32(u[6], K32One);
          v[7] = _mm_add_epi32(u[7], K32One);
          v[8] = _mm_add_epi32(u[8], K32One);
          v[9] = _mm_add_epi32(u[9], K32One);
          v[10] = _mm_add_epi32(u[10], K32One);
          v[11] = _mm_add_epi32(u[11], K32One);
          v[12] = _mm_add_epi32(u[12], K32One);
          v[13] = _mm_add_epi32(u[13], K32One);
          v[14] = _mm_add_epi32(u[14], K32One);
          v[15] = _mm_add_epi32(u[15], K32One);

          u[0] = _mm_srai_epi32(v[0], 2);
          u[1] = _mm_srai_epi32(v[1], 2);
          u[2] = _mm_srai_epi32(v[2], 2);
          u[3] = _mm_srai_epi32(v[3], 2);
          u[4] = _mm_srai_epi32(v[4], 2);
          u[5] = _mm_srai_epi32(v[5], 2);
          u[6] = _mm_srai_epi32(v[6], 2);
          u[7] = _mm_srai_epi32(v[7], 2);
          u[8] = _mm_srai_epi32(v[8], 2);
          u[9] = _mm_srai_epi32(v[9], 2);
          u[10] = _mm_srai_epi32(v[10], 2);
          u[11] = _mm_srai_epi32(v[11], 2);
          u[12] = _mm_srai_epi32(v[12], 2);
          u[13] = _mm_srai_epi32(v[13], 2);
          u[14] = _mm_srai_epi32(v[14], 2);
          u[15] = _mm_srai_epi32(v[15], 2);

          out[1] = _mm_packs_epi32(u[0], u[1]);
          out[17] = _mm_packs_epi32(u[2], u[3]);
          out[9] = _mm_packs_epi32(u[4], u[5]);
          out[25] = _mm_packs_epi32(u[6], u[7]);
          out[7] = _mm_packs_epi32(u[8], u[9]);
          out[23] = _mm_packs_epi32(u[10], u[11]);
          out[15] = _mm_packs_epi32(u[12], u[13]);
          out[31] = _mm_packs_epi32(u[14], u[15]);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&out[1], &out[17], &out[9], &out[25],
                                      &out[7], &out[23], &out[15], &out[31]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
        {
          const __m128i k32_p27_p05 = pair_set_epi32(cospi_27_64, cospi_5_64);
          const __m128i k32_p11_p21 = pair_set_epi32(cospi_11_64, cospi_21_64);
          const __m128i k32_p19_p13 = pair_set_epi32(cospi_19_64, cospi_13_64);
          const __m128i k32_p03_p29 = pair_set_epi32(cospi_3_64, cospi_29_64);
          const __m128i k32_m29_p03 = pair_set_epi32(-cospi_29_64, cospi_3_64);
          const __m128i k32_m13_p19 = pair_set_epi32(-cospi_13_64, cospi_19_64);
          const __m128i k32_m21_p11 = pair_set_epi32(-cospi_21_64, cospi_11_64);
          const __m128i k32_m05_p27 = pair_set_epi32(-cospi_5_64, cospi_27_64);

          u[0] = _mm_unpacklo_epi32(lstep1[40], lstep1[54]);
          u[1] = _mm_unpackhi_epi32(lstep1[40], lstep1[54]);
          u[2] = _mm_unpacklo_epi32(lstep1[41], lstep1[55]);
          u[3] = _mm_unpackhi_epi32(lstep1[41], lstep1[55]);
          u[4] = _mm_unpacklo_epi32(lstep1[42], lstep1[52]);
          u[5] = _mm_unpackhi_epi32(lstep1[42], lstep1[52]);
          u[6] = _mm_unpacklo_epi32(lstep1[43], lstep1[53]);
          u[7] = _mm_unpackhi_epi32(lstep1[43], lstep1[53]);
          u[8] = _mm_unpacklo_epi32(lstep1[44], lstep1[50]);
          u[9] = _mm_unpackhi_epi32(lstep1[44], lstep1[50]);
          u[10] = _mm_unpacklo_epi32(lstep1[45], lstep1[51]);
          u[11] = _mm_unpackhi_epi32(lstep1[45], lstep1[51]);
          u[12] = _mm_unpacklo_epi32(lstep1[46], lstep1[48]);
          u[13] = _mm_unpackhi_epi32(lstep1[46], lstep1[48]);
          u[14] = _mm_unpacklo_epi32(lstep1[47], lstep1[49]);
          u[15] = _mm_unpackhi_epi32(lstep1[47], lstep1[49]);

          v[0] = k_madd_epi32(u[0], k32_p27_p05);
          v[1] = k_madd_epi32(u[1], k32_p27_p05);
          v[2] = k_madd_epi32(u[2], k32_p27_p05);
          v[3] = k_madd_epi32(u[3], k32_p27_p05);
          v[4] = k_madd_epi32(u[4], k32_p11_p21);
          v[5] = k_madd_epi32(u[5], k32_p11_p21);
          v[6] = k_madd_epi32(u[6], k32_p11_p21);
          v[7] = k_madd_epi32(u[7], k32_p11_p21);
          v[8] = k_madd_epi32(u[8], k32_p19_p13);
          v[9] = k_madd_epi32(u[9], k32_p19_p13);
          v[10] = k_madd_epi32(u[10], k32_p19_p13);
          v[11] = k_madd_epi32(u[11], k32_p19_p13);
          v[12] = k_madd_epi32(u[12], k32_p03_p29);
          v[13] = k_madd_epi32(u[13], k32_p03_p29);
          v[14] = k_madd_epi32(u[14], k32_p03_p29);
          v[15] = k_madd_epi32(u[15], k32_p03_p29);
          v[16] = k_madd_epi32(u[12], k32_m29_p03);
          v[17] = k_madd_epi32(u[13], k32_m29_p03);
          v[18] = k_madd_epi32(u[14], k32_m29_p03);
          v[19] = k_madd_epi32(u[15], k32_m29_p03);
          v[20] = k_madd_epi32(u[8], k32_m13_p19);
          v[21] = k_madd_epi32(u[9], k32_m13_p19);
          v[22] = k_madd_epi32(u[10], k32_m13_p19);
          v[23] = k_madd_epi32(u[11], k32_m13_p19);
          v[24] = k_madd_epi32(u[4], k32_m21_p11);
          v[25] = k_madd_epi32(u[5], k32_m21_p11);
          v[26] = k_madd_epi32(u[6], k32_m21_p11);
          v[27] = k_madd_epi32(u[7], k32_m21_p11);
          v[28] = k_madd_epi32(u[0], k32_m05_p27);
          v[29] = k_madd_epi32(u[1], k32_m05_p27);
          v[30] = k_madd_epi32(u[2], k32_m05_p27);
          v[31] = k_madd_epi32(u[3], k32_m05_p27);

#if DCT_HIGH_BIT_DEPTH
          overflow = k_check_epi32_overflow_32(
              &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8],
              &v[9], &v[10], &v[11], &v[12], &v[13], &v[14], &v[15], &v[16],
              &v[17], &v[18], &v[19], &v[20], &v[21], &v[22], &v[23], &v[24],
              &v[25], &v[26], &v[27], &v[28], &v[29], &v[30], &v[31], &kZero);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
          u[0] = k_packs_epi64(v[0], v[1]);
          u[1] = k_packs_epi64(v[2], v[3]);
          u[2] = k_packs_epi64(v[4], v[5]);
          u[3] = k_packs_epi64(v[6], v[7]);
          u[4] = k_packs_epi64(v[8], v[9]);
          u[5] = k_packs_epi64(v[10], v[11]);
          u[6] = k_packs_epi64(v[12], v[13]);
          u[7] = k_packs_epi64(v[14], v[15]);
          u[8] = k_packs_epi64(v[16], v[17]);
          u[9] = k_packs_epi64(v[18], v[19]);
          u[10] = k_packs_epi64(v[20], v[21]);
          u[11] = k_packs_epi64(v[22], v[23]);
          u[12] = k_packs_epi64(v[24], v[25]);
          u[13] = k_packs_epi64(v[26], v[27]);
          u[14] = k_packs_epi64(v[28], v[29]);
          u[15] = k_packs_epi64(v[30], v[31]);

          v[0] = _mm_add_epi32(u[0], k__DCT_CONST_ROUNDING);
          v[1] = _mm_add_epi32(u[1], k__DCT_CONST_ROUNDING);
          v[2] = _mm_add_epi32(u[2], k__DCT_CONST_ROUNDING);
          v[3] = _mm_add_epi32(u[3], k__DCT_CONST_ROUNDING);
          v[4] = _mm_add_epi32(u[4], k__DCT_CONST_ROUNDING);
          v[5] = _mm_add_epi32(u[5], k__DCT_CONST_ROUNDING);
          v[6] = _mm_add_epi32(u[6], k__DCT_CONST_ROUNDING);
          v[7] = _mm_add_epi32(u[7], k__DCT_CONST_ROUNDING);
          v[8] = _mm_add_epi32(u[8], k__DCT_CONST_ROUNDING);
          v[9] = _mm_add_epi32(u[9], k__DCT_CONST_ROUNDING);
          v[10] = _mm_add_epi32(u[10], k__DCT_CONST_ROUNDING);
          v[11] = _mm_add_epi32(u[11], k__DCT_CONST_ROUNDING);
          v[12] = _mm_add_epi32(u[12], k__DCT_CONST_ROUNDING);
          v[13] = _mm_add_epi32(u[13], k__DCT_CONST_ROUNDING);
          v[14] = _mm_add_epi32(u[14], k__DCT_CONST_ROUNDING);
          v[15] = _mm_add_epi32(u[15], k__DCT_CONST_ROUNDING);

          u[0] = _mm_srai_epi32(v[0], DCT_CONST_BITS);
          u[1] = _mm_srai_epi32(v[1], DCT_CONST_BITS);
          u[2] = _mm_srai_epi32(v[2], DCT_CONST_BITS);
          u[3] = _mm_srai_epi32(v[3], DCT_CONST_BITS);
          u[4] = _mm_srai_epi32(v[4], DCT_CONST_BITS);
          u[5] = _mm_srai_epi32(v[5], DCT_CONST_BITS);
          u[6] = _mm_srai_epi32(v[6], DCT_CONST_BITS);
          u[7] = _mm_srai_epi32(v[7], DCT_CONST_BITS);
          u[8] = _mm_srai_epi32(v[8], DCT_CONST_BITS);
          u[9] = _mm_srai_epi32(v[9], DCT_CONST_BITS);
          u[10] = _mm_srai_epi32(v[10], DCT_CONST_BITS);
          u[11] = _mm_srai_epi32(v[11], DCT_CONST_BITS);
          u[12] = _mm_srai_epi32(v[12], DCT_CONST_BITS);
          u[13] = _mm_srai_epi32(v[13], DCT_CONST_BITS);
          u[14] = _mm_srai_epi32(v[14], DCT_CONST_BITS);
          u[15] = _mm_srai_epi32(v[15], DCT_CONST_BITS);

          v[0] = _mm_cmplt_epi32(u[0], kZero);
          v[1] = _mm_cmplt_epi32(u[1], kZero);
          v[2] = _mm_cmplt_epi32(u[2], kZero);
          v[3] = _mm_cmplt_epi32(u[3], kZero);
          v[4] = _mm_cmplt_epi32(u[4], kZero);
          v[5] = _mm_cmplt_epi32(u[5], kZero);
          v[6] = _mm_cmplt_epi32(u[6], kZero);
          v[7] = _mm_cmplt_epi32(u[7], kZero);
          v[8] = _mm_cmplt_epi32(u[8], kZero);
          v[9] = _mm_cmplt_epi32(u[9], kZero);
          v[10] = _mm_cmplt_epi32(u[10], kZero);
          v[11] = _mm_cmplt_epi32(u[11], kZero);
          v[12] = _mm_cmplt_epi32(u[12], kZero);
          v[13] = _mm_cmplt_epi32(u[13], kZero);
          v[14] = _mm_cmplt_epi32(u[14], kZero);
          v[15] = _mm_cmplt_epi32(u[15], kZero);

          u[0] = _mm_sub_epi32(u[0], v[0]);
          u[1] = _mm_sub_epi32(u[1], v[1]);
          u[2] = _mm_sub_epi32(u[2], v[2]);
          u[3] = _mm_sub_epi32(u[3], v[3]);
          u[4] = _mm_sub_epi32(u[4], v[4]);
          u[5] = _mm_sub_epi32(u[5], v[5]);
          u[6] = _mm_sub_epi32(u[6], v[6]);
          u[7] = _mm_sub_epi32(u[7], v[7]);
          u[8] = _mm_sub_epi32(u[8], v[8]);
          u[9] = _mm_sub_epi32(u[9], v[9]);
          u[10] = _mm_sub_epi32(u[10], v[10]);
          u[11] = _mm_sub_epi32(u[11], v[11]);
          u[12] = _mm_sub_epi32(u[12], v[12]);
          u[13] = _mm_sub_epi32(u[13], v[13]);
          u[14] = _mm_sub_epi32(u[14], v[14]);
          u[15] = _mm_sub_epi32(u[15], v[15]);

          v[0] = _mm_add_epi32(u[0], K32One);
          v[1] = _mm_add_epi32(u[1], K32One);
          v[2] = _mm_add_epi32(u[2], K32One);
          v[3] = _mm_add_epi32(u[3], K32One);
          v[4] = _mm_add_epi32(u[4], K32One);
          v[5] = _mm_add_epi32(u[5], K32One);
          v[6] = _mm_add_epi32(u[6], K32One);
          v[7] = _mm_add_epi32(u[7], K32One);
          v[8] = _mm_add_epi32(u[8], K32One);
          v[9] = _mm_add_epi32(u[9], K32One);
          v[10] = _mm_add_epi32(u[10], K32One);
          v[11] = _mm_add_epi32(u[11], K32One);
          v[12] = _mm_add_epi32(u[12], K32One);
          v[13] = _mm_add_epi32(u[13], K32One);
          v[14] = _mm_add_epi32(u[14], K32One);
          v[15] = _mm_add_epi32(u[15], K32One);

          u[0] = _mm_srai_epi32(v[0], 2);
          u[1] = _mm_srai_epi32(v[1], 2);
          u[2] = _mm_srai_epi32(v[2], 2);
          u[3] = _mm_srai_epi32(v[3], 2);
          u[4] = _mm_srai_epi32(v[4], 2);
          u[5] = _mm_srai_epi32(v[5], 2);
          u[6] = _mm_srai_epi32(v[6], 2);
          u[7] = _mm_srai_epi32(v[7], 2);
          u[8] = _mm_srai_epi32(v[8], 2);
          u[9] = _mm_srai_epi32(v[9], 2);
          u[10] = _mm_srai_epi32(v[10], 2);
          u[11] = _mm_srai_epi32(v[11], 2);
          u[12] = _mm_srai_epi32(v[12], 2);
          u[13] = _mm_srai_epi32(v[13], 2);
          u[14] = _mm_srai_epi32(v[14], 2);
          u[15] = _mm_srai_epi32(v[15], 2);

          out[5] = _mm_packs_epi32(u[0], u[1]);
          out[21] = _mm_packs_epi32(u[2], u[3]);
          out[13] = _mm_packs_epi32(u[4], u[5]);
          out[29] = _mm_packs_epi32(u[6], u[7]);
          out[3] = _mm_packs_epi32(u[8], u[9]);
          out[19] = _mm_packs_epi32(u[10], u[11]);
          out[11] = _mm_packs_epi32(u[12], u[13]);
          out[27] = _mm_packs_epi32(u[14], u[15]);
#if DCT_HIGH_BIT_DEPTH
          overflow =
              check_epi16_overflow_x8(&out[5], &out[21], &out[13], &out[29],
                                      &out[3], &out[19], &out[11], &out[27]);
          if (overflow) {
            HIGH_FDCT32x32_2D_ROWS_C(intermediate, output_org);
            return;
          }
#endif  // DCT_HIGH_BIT_DEPTH
        }
      }
#endif  // FDCT32x32_HIGH_PRECISION
      // Transpose the results, do it as four 8x8 transposes.
      {
        int transpose_block;
        int16_t *output0 = &intermediate[column_start * 32];
        tran_low_t *output1 = &output_org[column_start * 32];
        for (transpose_block = 0; transpose_block < 4; ++transpose_block) {
          __m128i *this_out = &out[8 * transpose_block];
          // 00 01 02 03 04 05 06 07
          // 10 11 12 13 14 15 16 17
          // 20 21 22 23 24 25 26 27
          // 30 31 32 33 34 35 36 37
          // 40 41 42 43 44 45 46 47
          // 50 51 52 53 54 55 56 57
          // 60 61 62 63 64 65 66 67
          // 70 71 72 73 74 75 76 77
          const __m128i tr0_0 = _mm_unpacklo_epi16(this_out[0], this_out[1]);
          const __m128i tr0_1 = _mm_unpacklo_epi16(this_out[2], this_out[3]);
          const __m128i tr0_2 = _mm_unpackhi_epi16(this_out[0], this_out[1]);
          const __m128i tr0_3 = _mm_unpackhi_epi16(this_out[2], this_out[3]);
          const __m128i tr0_4 = _mm_unpacklo_epi16(this_out[4], this_out[5]);
          const __m128i tr0_5 = _mm_unpacklo_epi16(this_out[6], this_out[7]);
          const __m128i tr0_6 = _mm_unpackhi_epi16(this_out[4], this_out[5]);
          const __m128i tr0_7 = _mm_unpackhi_epi16(this_out[6], this_out[7]);
          // 00 10 01 11 02 12 03 13
          // 20 30 21 31 22 32 23 33
          // 04 14 05 15 06 16 07 17
          // 24 34 25 35 26 36 27 37
          // 40 50 41 51 42 52 43 53
          // 60 70 61 71 62 72 63 73
          // 54 54 55 55 56 56 57 57
          // 64 74 65 75 66 76 67 77
          const __m128i tr1_0 = _mm_unpacklo_epi32(tr0_0, tr0_1);
          const __m128i tr1_1 = _mm_unpacklo_epi32(tr0_2, tr0_3);
          const __m128i tr1_2 = _mm_unpackhi_epi32(tr0_0, tr0_1);
          const __m128i tr1_3 = _mm_unpackhi_epi32(tr0_2, tr0_3);
          const __m128i tr1_4 = _mm_unpacklo_epi32(tr0_4, tr0_5);
          const __m128i tr1_5 = _mm_unpacklo_epi32(tr0_6, tr0_7);
          const __m128i tr1_6 = _mm_unpackhi_epi32(tr0_4, tr0_5);
          const __m128i tr1_7 = _mm_unpackhi_epi32(tr0_6, tr0_7);
          // 00 10 20 30 01 11 21 31
          // 40 50 60 70 41 51 61 71
          // 02 12 22 32 03 13 23 33
          // 42 52 62 72 43 53 63 73
          // 04 14 24 34 05 15 21 36
          // 44 54 64 74 45 55 61 76
          // 06 16 26 36 07 17 27 37
          // 46 56 66 76 47 57 67 77
          __m128i tr2_0 = _mm_unpacklo_epi64(tr1_0, tr1_4);
          __m128i tr2_1 = _mm_unpackhi_epi64(tr1_0, tr1_4);
          __m128i tr2_2 = _mm_unpacklo_epi64(tr1_2, tr1_6);
          __m128i tr2_3 = _mm_unpackhi_epi64(tr1_2, tr1_6);
          __m128i tr2_4 = _mm_unpacklo_epi64(tr1_1, tr1_5);
          __m128i tr2_5 = _mm_unpackhi_epi64(tr1_1, tr1_5);
          __m128i tr2_6 = _mm_unpacklo_epi64(tr1_3, tr1_7);
          __m128i tr2_7 = _mm_unpackhi_epi64(tr1_3, tr1_7);
          // 00 10 20 30 40 50 60 70
          // 01 11 21 31 41 51 61 71
          // 02 12 22 32 42 52 62 72
          // 03 13 23 33 43 53 63 73
          // 04 14 24 34 44 54 64 74
          // 05 15 25 35 45 55 65 75
          // 06 16 26 36 46 56 66 76
          // 07 17 27 37 47 57 67 77
          if (0 == pass) {
            // output[j] = (output[j] + 1 + (output[j] > 0)) >> 2;
            // TODO(cd): see quality impact of only doing
            //           output[j] = (output[j] + 1) >> 2;
            //           which would remove the code between here ...
            __m128i tr2_0_0 = _mm_cmpgt_epi16(tr2_0, kZero);
            __m128i tr2_1_0 = _mm_cmpgt_epi16(tr2_1, kZero);
            __m128i tr2_2_0 = _mm_cmpgt_epi16(tr2_2, kZero);
            __m128i tr2_3_0 = _mm_cmpgt_epi16(tr2_3, kZero);
            __m128i tr2_4_0 = _mm_cmpgt_epi16(tr2_4, kZero);
            __m128i tr2_5_0 = _mm_cmpgt_epi16(tr2_5, kZero);
            __m128i tr2_6_0 = _mm_cmpgt_epi16(tr2_6, kZero);
            __m128i tr2_7_0 = _mm_cmpgt_epi16(tr2_7, kZero);
            tr2_0 = _mm_sub_epi16(tr2_0, tr2_0_0);
            tr2_1 = _mm_sub_epi16(tr2_1, tr2_1_0);
            tr2_2 = _mm_sub_epi16(tr2_2, tr2_2_0);
            tr2_3 = _mm_sub_epi16(tr2_3, tr2_3_0);
            tr2_4 = _mm_sub_epi16(tr2_4, tr2_4_0);
            tr2_5 = _mm_sub_epi16(tr2_5, tr2_5_0);
            tr2_6 = _mm_sub_epi16(tr2_6, tr2_6_0);
            tr2_7 = _mm_sub_epi16(tr2_7, tr2_7_0);
            //           ... and here.
            //           PS: also change code in vp9/encoder/vp9_dct.c
            tr2_0 = _mm_add_epi16(tr2_0, kOne);
            tr2_1 = _mm_add_epi16(tr2_1, kOne);
            tr2_2 = _mm_add_epi16(tr2_2, kOne);
            tr2_3 = _mm_add_epi16(tr2_3, kOne);
            tr2_4 = _mm_add_epi16(tr2_4, kOne);
            tr2_5 = _mm_add_epi16(tr2_5, kOne);
            tr2_6 = _mm_add_epi16(tr2_6, kOne);
            tr2_7 = _mm_add_epi16(tr2_7, kOne);
            tr2_0 = _mm_srai_epi16(tr2_0, 2);
            tr2_1 = _mm_srai_epi16(tr2_1, 2);
            tr2_2 = _mm_srai_epi16(tr2_2, 2);
            tr2_3 = _mm_srai_epi16(tr2_3, 2);
            tr2_4 = _mm_srai_epi16(tr2_4, 2);
            tr2_5 = _mm_srai_epi16(tr2_5, 2);
            tr2_6 = _mm_srai_epi16(tr2_6, 2);
            tr2_7 = _mm_srai_epi16(tr2_7, 2);
          }
          // Note: even though all these stores are aligned, using the aligned
          //       intrinsic make the code slightly slower.
          if (pass == 0) {
            _mm_storeu_si128((__m128i *)(output0 + 0 * 32), tr2_0);
            _mm_storeu_si128((__m128i *)(output0 + 1 * 32), tr2_1);
            _mm_storeu_si128((__m128i *)(output0 + 2 * 32), tr2_2);
            _mm_storeu_si128((__m128i *)(output0 + 3 * 32), tr2_3);
            _mm_storeu_si128((__m128i *)(output0 + 4 * 32), tr2_4);
            _mm_storeu_si128((__m128i *)(output0 + 5 * 32), tr2_5);
            _mm_storeu_si128((__m128i *)(output0 + 6 * 32), tr2_6);
            _mm_storeu_si128((__m128i *)(output0 + 7 * 32), tr2_7);
            // Process next 8x8
            output0 += 8;
          } else {
            storeu_output(&tr2_0, (output1 + 0 * 32));
            storeu_output(&tr2_1, (output1 + 1 * 32));
            storeu_output(&tr2_2, (output1 + 2 * 32));
            storeu_output(&tr2_3, (output1 + 3 * 32));
            storeu_output(&tr2_4, (output1 + 4 * 32));
            storeu_output(&tr2_5, (output1 + 5 * 32));
            storeu_output(&tr2_6, (output1 + 6 * 32));
            storeu_output(&tr2_7, (output1 + 7 * 32));
            // Process next 8x8
            output1 += 8;
          }
        }
      }
    }
  }
}  // NOLINT

#undef ADD_EPI16
#undef SUB_EPI16
#undef HIGH_FDCT32x32_2D_C
#undef HIGH_FDCT32x32_2D_ROWS_C
