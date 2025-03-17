/*
 * jchuff-neon.c - Huffman entropy encoding (64-bit Arm Neon)
 *
 * Copyright (C) 2020-2021, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, 2022, 2024, D. R. Commander.  All Rights Reserved.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 * NOTE: All referenced figures are from
 * Recommendation ITU-T T.81 (1992) | ISO/IEC 10918-1:1994.
 */

#define JPEG_INTERNALS
#include "../../../src/jinclude.h"
#include "../../../src/jpeglib.h"
#include "../../../src/jsimd.h"
#include "../../../src/jdct.h"
#include "../../../src/jsimddct.h"
#include "../../jsimd.h"
#include "../align.h"
#include "../jchuff.h"
#include "neon-compat.h"

#include <limits.h>

#include <arm_neon.h>


ALIGN(16) static const uint8_t jsimd_huff_encode_one_block_consts[] = {
    0,   1,   2,   3,  16,  17,  32,  33,
   18,  19,   4,   5,   6,   7,  20,  21,
   34,  35,  48,  49, 255, 255,  50,  51,
   36,  37,  22,  23,   8,   9,  10,  11,
  255, 255,   6,   7,  20,  21,  34,  35,
   48,  49, 255, 255,  50,  51,  36,  37,
   54,  55,  40,  41,  26,  27,  12,  13,
   14,  15,  28,  29,  42,  43,  56,  57,
    6,   7,  20,  21,  34,  35,  48,  49,
   50,  51,  36,  37,  22,  23,   8,   9,
   26,  27,  12,  13, 255, 255,  14,  15,
   28,  29,  42,  43,  56,  57, 255, 255,
   52,  53,  54,  55,  40,  41,  26,  27,
   12,  13, 255, 255,  14,  15,  28,  29,
   26,  27,  40,  41,  42,  43,  28,  29,
   14,  15,  30,  31,  44,  45,  46,  47
};

/* The AArch64 implementation of the FLUSH() macro triggers a UBSan misaligned
 * address warning because the macro sometimes writes a 64-bit value to a
 * non-64-bit-aligned address.  That behavior is technically undefined per
 * the C specification, but it is supported by the AArch64 architecture and
 * compilers.
 */
#if defined(__has_feature)
#if __has_feature(undefined_behavior_sanitizer)
__attribute__((no_sanitize("alignment")))
#endif
#endif
JOCTET *jsimd_huff_encode_one_block_neon(void *state, JOCTET *buffer,
                                         JCOEFPTR block, int last_dc_val,
                                         c_derived_tbl *dctbl,
                                         c_derived_tbl *actbl)
{
  uint16_t block_diff[DCTSIZE2];

  /* Load lookup table indices for rows of zig-zag ordering. */
#ifdef HAVE_VLD1Q_U8_X4
  const uint8x16x4_t idx_rows_0123 =
    vld1q_u8_x4(jsimd_huff_encode_one_block_consts + 0 * DCTSIZE);
  const uint8x16x4_t idx_rows_4567 =
    vld1q_u8_x4(jsimd_huff_encode_one_block_consts + 8 * DCTSIZE);
#else
  /* GCC does not currently support intrinsics vl1dq_<type>_x4(). */
  const uint8x16x4_t idx_rows_0123 = { {
    vld1q_u8(jsimd_huff_encode_one_block_consts + 0 * DCTSIZE),
    vld1q_u8(jsimd_huff_encode_one_block_consts + 2 * DCTSIZE),
    vld1q_u8(jsimd_huff_encode_one_block_consts + 4 * DCTSIZE),
    vld1q_u8(jsimd_huff_encode_one_block_consts + 6 * DCTSIZE)
  } };
  const uint8x16x4_t idx_rows_4567 = { {
    vld1q_u8(jsimd_huff_encode_one_block_consts + 8 * DCTSIZE),
    vld1q_u8(jsimd_huff_encode_one_block_consts + 10 * DCTSIZE),
    vld1q_u8(jsimd_huff_encode_one_block_consts + 12 * DCTSIZE),
    vld1q_u8(jsimd_huff_encode_one_block_consts + 14 * DCTSIZE)
  } };
#endif

  /* Load 8x8 block of DCT coefficients. */
#ifdef HAVE_VLD1Q_U8_X4
  const int8x16x4_t tbl_rows_0123 =
    vld1q_s8_x4((int8_t *)(block + 0 * DCTSIZE));
  const int8x16x4_t tbl_rows_4567 =
    vld1q_s8_x4((int8_t *)(block + 4 * DCTSIZE));
#else
  const int8x16x4_t tbl_rows_0123 = { {
    vld1q_s8((int8_t *)(block + 0 * DCTSIZE)),
    vld1q_s8((int8_t *)(block + 1 * DCTSIZE)),
    vld1q_s8((int8_t *)(block + 2 * DCTSIZE)),
    vld1q_s8((int8_t *)(block + 3 * DCTSIZE))
  } };
  const int8x16x4_t tbl_rows_4567 = { {
    vld1q_s8((int8_t *)(block + 4 * DCTSIZE)),
    vld1q_s8((int8_t *)(block + 5 * DCTSIZE)),
    vld1q_s8((int8_t *)(block + 6 * DCTSIZE)),
    vld1q_s8((int8_t *)(block + 7 * DCTSIZE))
  } };
#endif

  /* Initialise extra lookup tables. */
  const int8x16x4_t tbl_rows_2345 = { {
    tbl_rows_0123.val[2], tbl_rows_0123.val[3],
    tbl_rows_4567.val[0], tbl_rows_4567.val[1]
  } };
  const int8x16x3_t tbl_rows_567 =
    { { tbl_rows_4567.val[1], tbl_rows_4567.val[2], tbl_rows_4567.val[3] } };

  /* Shuffle coefficients into zig-zag order. */
  int16x8_t row0 =
    vreinterpretq_s16_s8(vqtbl4q_s8(tbl_rows_0123, idx_rows_0123.val[0]));
  int16x8_t row1 =
    vreinterpretq_s16_s8(vqtbl4q_s8(tbl_rows_0123, idx_rows_0123.val[1]));
  int16x8_t row2 =
    vreinterpretq_s16_s8(vqtbl4q_s8(tbl_rows_2345, idx_rows_0123.val[2]));
  int16x8_t row3 =
    vreinterpretq_s16_s8(vqtbl4q_s8(tbl_rows_0123, idx_rows_0123.val[3]));
  int16x8_t row4 =
    vreinterpretq_s16_s8(vqtbl4q_s8(tbl_rows_4567, idx_rows_4567.val[0]));
  int16x8_t row5 =
    vreinterpretq_s16_s8(vqtbl4q_s8(tbl_rows_2345, idx_rows_4567.val[1]));
  int16x8_t row6 =
    vreinterpretq_s16_s8(vqtbl4q_s8(tbl_rows_4567, idx_rows_4567.val[2]));
  int16x8_t row7 =
    vreinterpretq_s16_s8(vqtbl3q_s8(tbl_rows_567, idx_rows_4567.val[3]));

  /* Compute DC coefficient difference value (F.1.1.5.1). */
  row0 = vsetq_lane_s16(block[0] - last_dc_val, row0, 0);
  /* Initialize AC coefficient lanes not reachable by lookup tables. */
  row1 =
    vsetq_lane_s16(vgetq_lane_s16(vreinterpretq_s16_s8(tbl_rows_4567.val[0]),
                                  0), row1, 2);
  row2 =
    vsetq_lane_s16(vgetq_lane_s16(vreinterpretq_s16_s8(tbl_rows_0123.val[1]),
                                  4), row2, 0);
  row2 =
    vsetq_lane_s16(vgetq_lane_s16(vreinterpretq_s16_s8(tbl_rows_4567.val[2]),
                                  0), row2, 5);
  row5 =
    vsetq_lane_s16(vgetq_lane_s16(vreinterpretq_s16_s8(tbl_rows_0123.val[1]),
                                  7), row5, 2);
  row5 =
    vsetq_lane_s16(vgetq_lane_s16(vreinterpretq_s16_s8(tbl_rows_4567.val[2]),
                                  3), row5, 7);
  row6 =
    vsetq_lane_s16(vgetq_lane_s16(vreinterpretq_s16_s8(tbl_rows_0123.val[3]),
                                  7), row6, 5);

  /* DCT block is now in zig-zag order; start Huffman encoding process. */

  /* Construct bitmap to accelerate encoding of AC coefficients.  A set bit
   * means that the corresponding coefficient != 0.
   */
  uint16x8_t row0_ne_0 = vtstq_s16(row0, row0);
  uint16x8_t row1_ne_0 = vtstq_s16(row1, row1);
  uint16x8_t row2_ne_0 = vtstq_s16(row2, row2);
  uint16x8_t row3_ne_0 = vtstq_s16(row3, row3);
  uint16x8_t row4_ne_0 = vtstq_s16(row4, row4);
  uint16x8_t row5_ne_0 = vtstq_s16(row5, row5);
  uint16x8_t row6_ne_0 = vtstq_s16(row6, row6);
  uint16x8_t row7_ne_0 = vtstq_s16(row7, row7);

  uint8x16_t row10_ne_0 = vuzp1q_u8(vreinterpretq_u8_u16(row1_ne_0),
                                    vreinterpretq_u8_u16(row0_ne_0));
  uint8x16_t row32_ne_0 = vuzp1q_u8(vreinterpretq_u8_u16(row3_ne_0),
                                    vreinterpretq_u8_u16(row2_ne_0));
  uint8x16_t row54_ne_0 = vuzp1q_u8(vreinterpretq_u8_u16(row5_ne_0),
                                    vreinterpretq_u8_u16(row4_ne_0));
  uint8x16_t row76_ne_0 = vuzp1q_u8(vreinterpretq_u8_u16(row7_ne_0),
                                    vreinterpretq_u8_u16(row6_ne_0));

  /* { 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 } */
  const uint8x16_t bitmap_mask =
    vreinterpretq_u8_u64(vdupq_n_u64(0x0102040810204080));

  uint8x16_t bitmap_rows_10 = vandq_u8(row10_ne_0, bitmap_mask);
  uint8x16_t bitmap_rows_32 = vandq_u8(row32_ne_0, bitmap_mask);
  uint8x16_t bitmap_rows_54 = vandq_u8(row54_ne_0, bitmap_mask);
  uint8x16_t bitmap_rows_76 = vandq_u8(row76_ne_0, bitmap_mask);

  uint8x16_t bitmap_rows_3210 = vpaddq_u8(bitmap_rows_32, bitmap_rows_10);
  uint8x16_t bitmap_rows_7654 = vpaddq_u8(bitmap_rows_76, bitmap_rows_54);
  uint8x16_t bitmap_rows_76543210 = vpaddq_u8(bitmap_rows_7654,
                                              bitmap_rows_3210);
  uint8x8_t bitmap_all = vpadd_u8(vget_low_u8(bitmap_rows_76543210),
                                  vget_high_u8(bitmap_rows_76543210));

  /* Shift left to remove DC bit. */
  bitmap_all =
    vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(bitmap_all), 1));
  /* Count bits set (number of non-zero coefficients) in bitmap. */
  unsigned int non_zero_coefficients = vaddv_u8(vcnt_u8(bitmap_all));
  /* Move bitmap to 64-bit scalar register. */
  uint64_t bitmap = vget_lane_u64(vreinterpret_u64_u8(bitmap_all), 0);

  /* Set up state and bit buffer for output bitstream. */
  working_state *state_ptr = (working_state *)state;
  int free_bits = state_ptr->cur.free_bits;
  size_t put_buffer = state_ptr->cur.put_buffer;

  /* Encode DC coefficient. */

  /* For negative coeffs: diff = abs(coeff) -1 = ~abs(coeff) */
  int16x8_t abs_row0 = vabsq_s16(row0);
  int16x8_t row0_lz = vclzq_s16(abs_row0);
  uint16x8_t row0_mask = vshlq_u16(vcltzq_s16(row0), vnegq_s16(row0_lz));
  uint16x8_t row0_diff = veorq_u16(vreinterpretq_u16_s16(abs_row0), row0_mask);
  /* Find nbits required to specify sign and amplitude of coefficient. */
  unsigned int lz = vgetq_lane_u16(vreinterpretq_u16_s16(row0_lz), 0);
  unsigned int nbits = 16 - lz;
  /* Emit Huffman-coded symbol and additional diff bits. */
  unsigned int diff = vgetq_lane_u16(row0_diff, 0);
  PUT_CODE(dctbl->ehufco[nbits], dctbl->ehufsi[nbits], diff)

  /* Encode AC coefficients. */

  unsigned int r = 0;  /* r = run length of zeros */
  unsigned int i = 1;  /* i = number of coefficients encoded */
  /* Code and size information for a run length of 16 zero coefficients */
  const unsigned int code_0xf0 = actbl->ehufco[0xf0];
  const unsigned int size_0xf0 = actbl->ehufsi[0xf0];

  /* The most efficient method of computing nbits and diff depends on the
   * number of non-zero coefficients.  If the bitmap is not too sparse (> 8
   * non-zero AC coefficients), it is beneficial to do all of the work using
   * Neon; else we do some of the work using Neon and the rest on demand using
   * scalar code.
   */
  if (non_zero_coefficients > 8) {
    uint8_t block_nbits[DCTSIZE2];

    int16x8_t abs_row1 = vabsq_s16(row1);
    int16x8_t abs_row2 = vabsq_s16(row2);
    int16x8_t abs_row3 = vabsq_s16(row3);
    int16x8_t abs_row4 = vabsq_s16(row4);
    int16x8_t abs_row5 = vabsq_s16(row5);
    int16x8_t abs_row6 = vabsq_s16(row6);
    int16x8_t abs_row7 = vabsq_s16(row7);
    int16x8_t row1_lz = vclzq_s16(abs_row1);
    int16x8_t row2_lz = vclzq_s16(abs_row2);
    int16x8_t row3_lz = vclzq_s16(abs_row3);
    int16x8_t row4_lz = vclzq_s16(abs_row4);
    int16x8_t row5_lz = vclzq_s16(abs_row5);
    int16x8_t row6_lz = vclzq_s16(abs_row6);
    int16x8_t row7_lz = vclzq_s16(abs_row7);
    /* Narrow leading zero count to 8 bits. */
    uint8x16_t row01_lz = vuzp1q_u8(vreinterpretq_u8_s16(row0_lz),
                                    vreinterpretq_u8_s16(row1_lz));
    uint8x16_t row23_lz = vuzp1q_u8(vreinterpretq_u8_s16(row2_lz),
                                    vreinterpretq_u8_s16(row3_lz));
    uint8x16_t row45_lz = vuzp1q_u8(vreinterpretq_u8_s16(row4_lz),
                                    vreinterpretq_u8_s16(row5_lz));
    uint8x16_t row67_lz = vuzp1q_u8(vreinterpretq_u8_s16(row6_lz),
                                    vreinterpretq_u8_s16(row7_lz));
    /* Compute nbits needed to specify magnitude of each coefficient. */
    uint8x16_t row01_nbits = vsubq_u8(vdupq_n_u8(16), row01_lz);
    uint8x16_t row23_nbits = vsubq_u8(vdupq_n_u8(16), row23_lz);
    uint8x16_t row45_nbits = vsubq_u8(vdupq_n_u8(16), row45_lz);
    uint8x16_t row67_nbits = vsubq_u8(vdupq_n_u8(16), row67_lz);
    /* Store nbits. */
    vst1q_u8(block_nbits + 0 * DCTSIZE, row01_nbits);
    vst1q_u8(block_nbits + 2 * DCTSIZE, row23_nbits);
    vst1q_u8(block_nbits + 4 * DCTSIZE, row45_nbits);
    vst1q_u8(block_nbits + 6 * DCTSIZE, row67_nbits);
    /* Mask bits not required to specify sign and amplitude of diff. */
    uint16x8_t row1_mask = vshlq_u16(vcltzq_s16(row1), vnegq_s16(row1_lz));
    uint16x8_t row2_mask = vshlq_u16(vcltzq_s16(row2), vnegq_s16(row2_lz));
    uint16x8_t row3_mask = vshlq_u16(vcltzq_s16(row3), vnegq_s16(row3_lz));
    uint16x8_t row4_mask = vshlq_u16(vcltzq_s16(row4), vnegq_s16(row4_lz));
    uint16x8_t row5_mask = vshlq_u16(vcltzq_s16(row5), vnegq_s16(row5_lz));
    uint16x8_t row6_mask = vshlq_u16(vcltzq_s16(row6), vnegq_s16(row6_lz));
    uint16x8_t row7_mask = vshlq_u16(vcltzq_s16(row7), vnegq_s16(row7_lz));
    /* diff = abs(coeff) ^ sign(coeff) [no-op for positive coefficients] */
    uint16x8_t row1_diff = veorq_u16(vreinterpretq_u16_s16(abs_row1),
                                     row1_mask);
    uint16x8_t row2_diff = veorq_u16(vreinterpretq_u16_s16(abs_row2),
                                     row2_mask);
    uint16x8_t row3_diff = veorq_u16(vreinterpretq_u16_s16(abs_row3),
                                     row3_mask);
    uint16x8_t row4_diff = veorq_u16(vreinterpretq_u16_s16(abs_row4),
                                     row4_mask);
    uint16x8_t row5_diff = veorq_u16(vreinterpretq_u16_s16(abs_row5),
                                     row5_mask);
    uint16x8_t row6_diff = veorq_u16(vreinterpretq_u16_s16(abs_row6),
                                     row6_mask);
    uint16x8_t row7_diff = veorq_u16(vreinterpretq_u16_s16(abs_row7),
                                     row7_mask);
    /* Store diff bits. */
    vst1q_u16(block_diff + 0 * DCTSIZE, row0_diff);
    vst1q_u16(block_diff + 1 * DCTSIZE, row1_diff);
    vst1q_u16(block_diff + 2 * DCTSIZE, row2_diff);
    vst1q_u16(block_diff + 3 * DCTSIZE, row3_diff);
    vst1q_u16(block_diff + 4 * DCTSIZE, row4_diff);
    vst1q_u16(block_diff + 5 * DCTSIZE, row5_diff);
    vst1q_u16(block_diff + 6 * DCTSIZE, row6_diff);
    vst1q_u16(block_diff + 7 * DCTSIZE, row7_diff);

    while (bitmap != 0) {
      r = BUILTIN_CLZLL(bitmap);
      i += r;
      bitmap <<= r;
      nbits = block_nbits[i];
      diff = block_diff[i];
      while (r > 15) {
        /* If run length > 15, emit special run-length-16 codes. */
        PUT_BITS(code_0xf0, size_0xf0)
        r -= 16;
      }
      /* Emit Huffman symbol for run length / number of bits. (F.1.2.2.1) */
      unsigned int rs = (r << 4) + nbits;
      PUT_CODE(actbl->ehufco[rs], actbl->ehufsi[rs], diff)
      i++;
      bitmap <<= 1;
    }
  } else if (bitmap != 0) {
    uint16_t block_abs[DCTSIZE2];
    /* Compute and store absolute value of coefficients. */
    int16x8_t abs_row1 = vabsq_s16(row1);
    int16x8_t abs_row2 = vabsq_s16(row2);
    int16x8_t abs_row3 = vabsq_s16(row3);
    int16x8_t abs_row4 = vabsq_s16(row4);
    int16x8_t abs_row5 = vabsq_s16(row5);
    int16x8_t abs_row6 = vabsq_s16(row6);
    int16x8_t abs_row7 = vabsq_s16(row7);
    vst1q_u16(block_abs + 0 * DCTSIZE, vreinterpretq_u16_s16(abs_row0));
    vst1q_u16(block_abs + 1 * DCTSIZE, vreinterpretq_u16_s16(abs_row1));
    vst1q_u16(block_abs + 2 * DCTSIZE, vreinterpretq_u16_s16(abs_row2));
    vst1q_u16(block_abs + 3 * DCTSIZE, vreinterpretq_u16_s16(abs_row3));
    vst1q_u16(block_abs + 4 * DCTSIZE, vreinterpretq_u16_s16(abs_row4));
    vst1q_u16(block_abs + 5 * DCTSIZE, vreinterpretq_u16_s16(abs_row5));
    vst1q_u16(block_abs + 6 * DCTSIZE, vreinterpretq_u16_s16(abs_row6));
    vst1q_u16(block_abs + 7 * DCTSIZE, vreinterpretq_u16_s16(abs_row7));
    /* Compute diff bits (without nbits mask) and store. */
    uint16x8_t row1_diff = veorq_u16(vreinterpretq_u16_s16(abs_row1),
                                     vcltzq_s16(row1));
    uint16x8_t row2_diff = veorq_u16(vreinterpretq_u16_s16(abs_row2),
                                     vcltzq_s16(row2));
    uint16x8_t row3_diff = veorq_u16(vreinterpretq_u16_s16(abs_row3),
                                     vcltzq_s16(row3));
    uint16x8_t row4_diff = veorq_u16(vreinterpretq_u16_s16(abs_row4),
                                     vcltzq_s16(row4));
    uint16x8_t row5_diff = veorq_u16(vreinterpretq_u16_s16(abs_row5),
                                     vcltzq_s16(row5));
    uint16x8_t row6_diff = veorq_u16(vreinterpretq_u16_s16(abs_row6),
                                     vcltzq_s16(row6));
    uint16x8_t row7_diff = veorq_u16(vreinterpretq_u16_s16(abs_row7),
                                     vcltzq_s16(row7));
    vst1q_u16(block_diff + 0 * DCTSIZE, row0_diff);
    vst1q_u16(block_diff + 1 * DCTSIZE, row1_diff);
    vst1q_u16(block_diff + 2 * DCTSIZE, row2_diff);
    vst1q_u16(block_diff + 3 * DCTSIZE, row3_diff);
    vst1q_u16(block_diff + 4 * DCTSIZE, row4_diff);
    vst1q_u16(block_diff + 5 * DCTSIZE, row5_diff);
    vst1q_u16(block_diff + 6 * DCTSIZE, row6_diff);
    vst1q_u16(block_diff + 7 * DCTSIZE, row7_diff);

    /* Same as above but must mask diff bits and compute nbits on demand. */
    while (bitmap != 0) {
      r = BUILTIN_CLZLL(bitmap);
      i += r;
      bitmap <<= r;
      lz = BUILTIN_CLZ(block_abs[i]);
      nbits = 32 - lz;
      diff = ((unsigned int)block_diff[i] << lz) >> lz;
      while (r > 15) {
        /* If run length > 15, emit special run-length-16 codes. */
        PUT_BITS(code_0xf0, size_0xf0)
        r -= 16;
      }
      /* Emit Huffman symbol for run length / number of bits. (F.1.2.2.1) */
      unsigned int rs = (r << 4) + nbits;
      PUT_CODE(actbl->ehufco[rs], actbl->ehufsi[rs], diff)
      i++;
      bitmap <<= 1;
    }
  }

  /* If the last coefficient(s) were zero, emit an end-of-block (EOB) code.
   * The value of RS for the EOB code is 0.
   */
  if (i != 64) {
    PUT_BITS(actbl->ehufco[0], actbl->ehufsi[0])
  }

  state_ptr->cur.put_buffer = put_buffer;
  state_ptr->cur.free_bits = free_bits;

  return buffer;
}
