/*
 *  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp9_rtcd.h"
#include "vpx_dsp/x86/inv_txfm_sse2.h"

void vp9_iht4x4_16_add_sse2(const tran_low_t *input, uint8_t *dest, int stride,
                            int tx_type) {
  __m128i in[2];
  const __m128i eight = _mm_set1_epi16(8);

  in[0] = load_input_data8(input);
  in[1] = load_input_data8(input + 8);

  switch (tx_type) {
    case DCT_DCT:
      idct4_sse2(in);
      idct4_sse2(in);
      break;
    case ADST_DCT:
      idct4_sse2(in);
      iadst4_sse2(in);
      break;
    case DCT_ADST:
      iadst4_sse2(in);
      idct4_sse2(in);
      break;
    default:
      assert(tx_type == ADST_ADST);
      iadst4_sse2(in);
      iadst4_sse2(in);
      break;
  }

  // Final round and shift
  in[0] = _mm_add_epi16(in[0], eight);
  in[1] = _mm_add_epi16(in[1], eight);

  in[0] = _mm_srai_epi16(in[0], 4);
  in[1] = _mm_srai_epi16(in[1], 4);

  recon_and_store4x4_sse2(in, dest, stride);
}

void vp9_iht8x8_64_add_sse2(const tran_low_t *input, uint8_t *dest, int stride,
                            int tx_type) {
  __m128i in[8];
  const __m128i final_rounding = _mm_set1_epi16(1 << 4);

  // load input data
  in[0] = load_input_data8(input);
  in[1] = load_input_data8(input + 8 * 1);
  in[2] = load_input_data8(input + 8 * 2);
  in[3] = load_input_data8(input + 8 * 3);
  in[4] = load_input_data8(input + 8 * 4);
  in[5] = load_input_data8(input + 8 * 5);
  in[6] = load_input_data8(input + 8 * 6);
  in[7] = load_input_data8(input + 8 * 7);

  switch (tx_type) {
    case DCT_DCT:
      vpx_idct8_sse2(in);
      vpx_idct8_sse2(in);
      break;
    case ADST_DCT:
      vpx_idct8_sse2(in);
      iadst8_sse2(in);
      break;
    case DCT_ADST:
      iadst8_sse2(in);
      vpx_idct8_sse2(in);
      break;
    default:
      assert(tx_type == ADST_ADST);
      iadst8_sse2(in);
      iadst8_sse2(in);
      break;
  }

  // Final rounding and shift
  in[0] = _mm_adds_epi16(in[0], final_rounding);
  in[1] = _mm_adds_epi16(in[1], final_rounding);
  in[2] = _mm_adds_epi16(in[2], final_rounding);
  in[3] = _mm_adds_epi16(in[3], final_rounding);
  in[4] = _mm_adds_epi16(in[4], final_rounding);
  in[5] = _mm_adds_epi16(in[5], final_rounding);
  in[6] = _mm_adds_epi16(in[6], final_rounding);
  in[7] = _mm_adds_epi16(in[7], final_rounding);

  in[0] = _mm_srai_epi16(in[0], 5);
  in[1] = _mm_srai_epi16(in[1], 5);
  in[2] = _mm_srai_epi16(in[2], 5);
  in[3] = _mm_srai_epi16(in[3], 5);
  in[4] = _mm_srai_epi16(in[4], 5);
  in[5] = _mm_srai_epi16(in[5], 5);
  in[6] = _mm_srai_epi16(in[6], 5);
  in[7] = _mm_srai_epi16(in[7], 5);

  recon_and_store(dest + 0 * stride, in[0]);
  recon_and_store(dest + 1 * stride, in[1]);
  recon_and_store(dest + 2 * stride, in[2]);
  recon_and_store(dest + 3 * stride, in[3]);
  recon_and_store(dest + 4 * stride, in[4]);
  recon_and_store(dest + 5 * stride, in[5]);
  recon_and_store(dest + 6 * stride, in[6]);
  recon_and_store(dest + 7 * stride, in[7]);
}

static INLINE void load_buffer_8x16(const tran_low_t *const input,
                                    __m128i *const in) {
  in[0] = load_input_data8(input + 0 * 16);
  in[1] = load_input_data8(input + 1 * 16);
  in[2] = load_input_data8(input + 2 * 16);
  in[3] = load_input_data8(input + 3 * 16);
  in[4] = load_input_data8(input + 4 * 16);
  in[5] = load_input_data8(input + 5 * 16);
  in[6] = load_input_data8(input + 6 * 16);
  in[7] = load_input_data8(input + 7 * 16);

  in[8] = load_input_data8(input + 8 * 16);
  in[9] = load_input_data8(input + 9 * 16);
  in[10] = load_input_data8(input + 10 * 16);
  in[11] = load_input_data8(input + 11 * 16);
  in[12] = load_input_data8(input + 12 * 16);
  in[13] = load_input_data8(input + 13 * 16);
  in[14] = load_input_data8(input + 14 * 16);
  in[15] = load_input_data8(input + 15 * 16);
}

static INLINE void write_buffer_8x16(uint8_t *const dest, __m128i *const in,
                                     const int stride) {
  const __m128i final_rounding = _mm_set1_epi16(1 << 5);
  // Final rounding and shift
  in[0] = _mm_adds_epi16(in[0], final_rounding);
  in[1] = _mm_adds_epi16(in[1], final_rounding);
  in[2] = _mm_adds_epi16(in[2], final_rounding);
  in[3] = _mm_adds_epi16(in[3], final_rounding);
  in[4] = _mm_adds_epi16(in[4], final_rounding);
  in[5] = _mm_adds_epi16(in[5], final_rounding);
  in[6] = _mm_adds_epi16(in[6], final_rounding);
  in[7] = _mm_adds_epi16(in[7], final_rounding);
  in[8] = _mm_adds_epi16(in[8], final_rounding);
  in[9] = _mm_adds_epi16(in[9], final_rounding);
  in[10] = _mm_adds_epi16(in[10], final_rounding);
  in[11] = _mm_adds_epi16(in[11], final_rounding);
  in[12] = _mm_adds_epi16(in[12], final_rounding);
  in[13] = _mm_adds_epi16(in[13], final_rounding);
  in[14] = _mm_adds_epi16(in[14], final_rounding);
  in[15] = _mm_adds_epi16(in[15], final_rounding);

  in[0] = _mm_srai_epi16(in[0], 6);
  in[1] = _mm_srai_epi16(in[1], 6);
  in[2] = _mm_srai_epi16(in[2], 6);
  in[3] = _mm_srai_epi16(in[3], 6);
  in[4] = _mm_srai_epi16(in[4], 6);
  in[5] = _mm_srai_epi16(in[5], 6);
  in[6] = _mm_srai_epi16(in[6], 6);
  in[7] = _mm_srai_epi16(in[7], 6);
  in[8] = _mm_srai_epi16(in[8], 6);
  in[9] = _mm_srai_epi16(in[9], 6);
  in[10] = _mm_srai_epi16(in[10], 6);
  in[11] = _mm_srai_epi16(in[11], 6);
  in[12] = _mm_srai_epi16(in[12], 6);
  in[13] = _mm_srai_epi16(in[13], 6);
  in[14] = _mm_srai_epi16(in[14], 6);
  in[15] = _mm_srai_epi16(in[15], 6);

  recon_and_store(dest + 0 * stride, in[0]);
  recon_and_store(dest + 1 * stride, in[1]);
  recon_and_store(dest + 2 * stride, in[2]);
  recon_and_store(dest + 3 * stride, in[3]);
  recon_and_store(dest + 4 * stride, in[4]);
  recon_and_store(dest + 5 * stride, in[5]);
  recon_and_store(dest + 6 * stride, in[6]);
  recon_and_store(dest + 7 * stride, in[7]);
  recon_and_store(dest + 8 * stride, in[8]);
  recon_and_store(dest + 9 * stride, in[9]);
  recon_and_store(dest + 10 * stride, in[10]);
  recon_and_store(dest + 11 * stride, in[11]);
  recon_and_store(dest + 12 * stride, in[12]);
  recon_and_store(dest + 13 * stride, in[13]);
  recon_and_store(dest + 14 * stride, in[14]);
  recon_and_store(dest + 15 * stride, in[15]);
}

void vp9_iht16x16_256_add_sse2(const tran_low_t *input, uint8_t *dest,
                               int stride, int tx_type) {
  __m128i in0[16], in1[16];

  load_buffer_8x16(input, in0);
  input += 8;
  load_buffer_8x16(input, in1);

  switch (tx_type) {
    case DCT_DCT:
      idct16_sse2(in0, in1);
      idct16_sse2(in0, in1);
      break;
    case ADST_DCT:
      idct16_sse2(in0, in1);
      iadst16_sse2(in0, in1);
      break;
    case DCT_ADST:
      iadst16_sse2(in0, in1);
      idct16_sse2(in0, in1);
      break;
    default:
      assert(tx_type == ADST_ADST);
      iadst16_sse2(in0, in1);
      iadst16_sse2(in0, in1);
      break;
  }

  write_buffer_8x16(dest, in0, stride);
  dest += 8;
  write_buffer_8x16(dest, in1, stride);
}
