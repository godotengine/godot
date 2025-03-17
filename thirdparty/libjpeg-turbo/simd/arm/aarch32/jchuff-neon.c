/*
 * jchuff-neon.c - Huffman entropy encoding (32-bit Arm Neon)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2024, D. R. Commander.  All Rights Reserved.
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
#include "../jchuff.h"
#include "neon-compat.h"

#include <limits.h>

#include <arm_neon.h>


JOCTET *jsimd_huff_encode_one_block_neon(void *state, JOCTET *buffer,
                                         JCOEFPTR block, int last_dc_val,
                                         c_derived_tbl *dctbl,
                                         c_derived_tbl *actbl)
{
  uint8_t block_nbits[DCTSIZE2];
  uint16_t block_diff[DCTSIZE2];

  /* Load rows of coefficients from DCT block in zig-zag order. */

  /* Compute DC coefficient difference value. (F.1.1.5.1) */
  int16x8_t row0 = vdupq_n_s16(block[0] - last_dc_val);
  row0 = vld1q_lane_s16(block +  1, row0, 1);
  row0 = vld1q_lane_s16(block +  8, row0, 2);
  row0 = vld1q_lane_s16(block + 16, row0, 3);
  row0 = vld1q_lane_s16(block +  9, row0, 4);
  row0 = vld1q_lane_s16(block +  2, row0, 5);
  row0 = vld1q_lane_s16(block +  3, row0, 6);
  row0 = vld1q_lane_s16(block + 10, row0, 7);

  int16x8_t row1 = vld1q_dup_s16(block + 17);
  row1 = vld1q_lane_s16(block + 24, row1, 1);
  row1 = vld1q_lane_s16(block + 32, row1, 2);
  row1 = vld1q_lane_s16(block + 25, row1, 3);
  row1 = vld1q_lane_s16(block + 18, row1, 4);
  row1 = vld1q_lane_s16(block + 11, row1, 5);
  row1 = vld1q_lane_s16(block +  4, row1, 6);
  row1 = vld1q_lane_s16(block +  5, row1, 7);

  int16x8_t row2 = vld1q_dup_s16(block + 12);
  row2 = vld1q_lane_s16(block + 19, row2, 1);
  row2 = vld1q_lane_s16(block + 26, row2, 2);
  row2 = vld1q_lane_s16(block + 33, row2, 3);
  row2 = vld1q_lane_s16(block + 40, row2, 4);
  row2 = vld1q_lane_s16(block + 48, row2, 5);
  row2 = vld1q_lane_s16(block + 41, row2, 6);
  row2 = vld1q_lane_s16(block + 34, row2, 7);

  int16x8_t row3 = vld1q_dup_s16(block + 27);
  row3 = vld1q_lane_s16(block + 20, row3, 1);
  row3 = vld1q_lane_s16(block + 13, row3, 2);
  row3 = vld1q_lane_s16(block +  6, row3, 3);
  row3 = vld1q_lane_s16(block +  7, row3, 4);
  row3 = vld1q_lane_s16(block + 14, row3, 5);
  row3 = vld1q_lane_s16(block + 21, row3, 6);
  row3 = vld1q_lane_s16(block + 28, row3, 7);

  int16x8_t abs_row0 = vabsq_s16(row0);
  int16x8_t abs_row1 = vabsq_s16(row1);
  int16x8_t abs_row2 = vabsq_s16(row2);
  int16x8_t abs_row3 = vabsq_s16(row3);

  int16x8_t row0_lz = vclzq_s16(abs_row0);
  int16x8_t row1_lz = vclzq_s16(abs_row1);
  int16x8_t row2_lz = vclzq_s16(abs_row2);
  int16x8_t row3_lz = vclzq_s16(abs_row3);

  /* Compute number of bits required to represent each coefficient. */
  uint8x8_t row0_nbits = vsub_u8(vdup_n_u8(16),
                                 vmovn_u16(vreinterpretq_u16_s16(row0_lz)));
  uint8x8_t row1_nbits = vsub_u8(vdup_n_u8(16),
                                 vmovn_u16(vreinterpretq_u16_s16(row1_lz)));
  uint8x8_t row2_nbits = vsub_u8(vdup_n_u8(16),
                                 vmovn_u16(vreinterpretq_u16_s16(row2_lz)));
  uint8x8_t row3_nbits = vsub_u8(vdup_n_u8(16),
                                 vmovn_u16(vreinterpretq_u16_s16(row3_lz)));

  vst1_u8(block_nbits + 0 * DCTSIZE, row0_nbits);
  vst1_u8(block_nbits + 1 * DCTSIZE, row1_nbits);
  vst1_u8(block_nbits + 2 * DCTSIZE, row2_nbits);
  vst1_u8(block_nbits + 3 * DCTSIZE, row3_nbits);

  uint16x8_t row0_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row0, 15)),
              vnegq_s16(row0_lz));
  uint16x8_t row1_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row1, 15)),
              vnegq_s16(row1_lz));
  uint16x8_t row2_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row2, 15)),
              vnegq_s16(row2_lz));
  uint16x8_t row3_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row3, 15)),
              vnegq_s16(row3_lz));

  uint16x8_t row0_diff = veorq_u16(vreinterpretq_u16_s16(abs_row0), row0_mask);
  uint16x8_t row1_diff = veorq_u16(vreinterpretq_u16_s16(abs_row1), row1_mask);
  uint16x8_t row2_diff = veorq_u16(vreinterpretq_u16_s16(abs_row2), row2_mask);
  uint16x8_t row3_diff = veorq_u16(vreinterpretq_u16_s16(abs_row3), row3_mask);

  /* Store diff values for rows 0, 1, 2, and 3. */
  vst1q_u16(block_diff + 0 * DCTSIZE, row0_diff);
  vst1q_u16(block_diff + 1 * DCTSIZE, row1_diff);
  vst1q_u16(block_diff + 2 * DCTSIZE, row2_diff);
  vst1q_u16(block_diff + 3 * DCTSIZE, row3_diff);

  /* Load last four rows of coefficients from DCT block in zig-zag order. */
  int16x8_t row4 = vld1q_dup_s16(block + 35);
  row4 = vld1q_lane_s16(block + 42, row4, 1);
  row4 = vld1q_lane_s16(block + 49, row4, 2);
  row4 = vld1q_lane_s16(block + 56, row4, 3);
  row4 = vld1q_lane_s16(block + 57, row4, 4);
  row4 = vld1q_lane_s16(block + 50, row4, 5);
  row4 = vld1q_lane_s16(block + 43, row4, 6);
  row4 = vld1q_lane_s16(block + 36, row4, 7);

  int16x8_t row5 = vld1q_dup_s16(block + 29);
  row5 = vld1q_lane_s16(block + 22, row5, 1);
  row5 = vld1q_lane_s16(block + 15, row5, 2);
  row5 = vld1q_lane_s16(block + 23, row5, 3);
  row5 = vld1q_lane_s16(block + 30, row5, 4);
  row5 = vld1q_lane_s16(block + 37, row5, 5);
  row5 = vld1q_lane_s16(block + 44, row5, 6);
  row5 = vld1q_lane_s16(block + 51, row5, 7);

  int16x8_t row6 = vld1q_dup_s16(block + 58);
  row6 = vld1q_lane_s16(block + 59, row6, 1);
  row6 = vld1q_lane_s16(block + 52, row6, 2);
  row6 = vld1q_lane_s16(block + 45, row6, 3);
  row6 = vld1q_lane_s16(block + 38, row6, 4);
  row6 = vld1q_lane_s16(block + 31, row6, 5);
  row6 = vld1q_lane_s16(block + 39, row6, 6);
  row6 = vld1q_lane_s16(block + 46, row6, 7);

  int16x8_t row7 = vld1q_dup_s16(block + 53);
  row7 = vld1q_lane_s16(block + 60, row7, 1);
  row7 = vld1q_lane_s16(block + 61, row7, 2);
  row7 = vld1q_lane_s16(block + 54, row7, 3);
  row7 = vld1q_lane_s16(block + 47, row7, 4);
  row7 = vld1q_lane_s16(block + 55, row7, 5);
  row7 = vld1q_lane_s16(block + 62, row7, 6);
  row7 = vld1q_lane_s16(block + 63, row7, 7);

  int16x8_t abs_row4 = vabsq_s16(row4);
  int16x8_t abs_row5 = vabsq_s16(row5);
  int16x8_t abs_row6 = vabsq_s16(row6);
  int16x8_t abs_row7 = vabsq_s16(row7);

  int16x8_t row4_lz = vclzq_s16(abs_row4);
  int16x8_t row5_lz = vclzq_s16(abs_row5);
  int16x8_t row6_lz = vclzq_s16(abs_row6);
  int16x8_t row7_lz = vclzq_s16(abs_row7);

  /* Compute number of bits required to represent each coefficient. */
  uint8x8_t row4_nbits = vsub_u8(vdup_n_u8(16),
                                 vmovn_u16(vreinterpretq_u16_s16(row4_lz)));
  uint8x8_t row5_nbits = vsub_u8(vdup_n_u8(16),
                                 vmovn_u16(vreinterpretq_u16_s16(row5_lz)));
  uint8x8_t row6_nbits = vsub_u8(vdup_n_u8(16),
                                 vmovn_u16(vreinterpretq_u16_s16(row6_lz)));
  uint8x8_t row7_nbits = vsub_u8(vdup_n_u8(16),
                                 vmovn_u16(vreinterpretq_u16_s16(row7_lz)));

  vst1_u8(block_nbits + 4 * DCTSIZE, row4_nbits);
  vst1_u8(block_nbits + 5 * DCTSIZE, row5_nbits);
  vst1_u8(block_nbits + 6 * DCTSIZE, row6_nbits);
  vst1_u8(block_nbits + 7 * DCTSIZE, row7_nbits);

  uint16x8_t row4_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row4, 15)),
              vnegq_s16(row4_lz));
  uint16x8_t row5_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row5, 15)),
              vnegq_s16(row5_lz));
  uint16x8_t row6_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row6, 15)),
              vnegq_s16(row6_lz));
  uint16x8_t row7_mask =
    vshlq_u16(vreinterpretq_u16_s16(vshrq_n_s16(row7, 15)),
              vnegq_s16(row7_lz));

  uint16x8_t row4_diff = veorq_u16(vreinterpretq_u16_s16(abs_row4), row4_mask);
  uint16x8_t row5_diff = veorq_u16(vreinterpretq_u16_s16(abs_row5), row5_mask);
  uint16x8_t row6_diff = veorq_u16(vreinterpretq_u16_s16(abs_row6), row6_mask);
  uint16x8_t row7_diff = veorq_u16(vreinterpretq_u16_s16(abs_row7), row7_mask);

  /* Store diff values for rows 4, 5, 6, and 7. */
  vst1q_u16(block_diff + 4 * DCTSIZE, row4_diff);
  vst1q_u16(block_diff + 5 * DCTSIZE, row5_diff);
  vst1q_u16(block_diff + 6 * DCTSIZE, row6_diff);
  vst1q_u16(block_diff + 7 * DCTSIZE, row7_diff);

  /* Construct bitmap to accelerate encoding of AC coefficients.  A set bit
   * means that the corresponding coefficient != 0.
   */
  uint8x8_t row0_nbits_gt0 = vcgt_u8(row0_nbits, vdup_n_u8(0));
  uint8x8_t row1_nbits_gt0 = vcgt_u8(row1_nbits, vdup_n_u8(0));
  uint8x8_t row2_nbits_gt0 = vcgt_u8(row2_nbits, vdup_n_u8(0));
  uint8x8_t row3_nbits_gt0 = vcgt_u8(row3_nbits, vdup_n_u8(0));
  uint8x8_t row4_nbits_gt0 = vcgt_u8(row4_nbits, vdup_n_u8(0));
  uint8x8_t row5_nbits_gt0 = vcgt_u8(row5_nbits, vdup_n_u8(0));
  uint8x8_t row6_nbits_gt0 = vcgt_u8(row6_nbits, vdup_n_u8(0));
  uint8x8_t row7_nbits_gt0 = vcgt_u8(row7_nbits, vdup_n_u8(0));

  /* { 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 } */
  const uint8x8_t bitmap_mask =
    vreinterpret_u8_u64(vmov_n_u64(0x0102040810204080));

  row0_nbits_gt0 = vand_u8(row0_nbits_gt0, bitmap_mask);
  row1_nbits_gt0 = vand_u8(row1_nbits_gt0, bitmap_mask);
  row2_nbits_gt0 = vand_u8(row2_nbits_gt0, bitmap_mask);
  row3_nbits_gt0 = vand_u8(row3_nbits_gt0, bitmap_mask);
  row4_nbits_gt0 = vand_u8(row4_nbits_gt0, bitmap_mask);
  row5_nbits_gt0 = vand_u8(row5_nbits_gt0, bitmap_mask);
  row6_nbits_gt0 = vand_u8(row6_nbits_gt0, bitmap_mask);
  row7_nbits_gt0 = vand_u8(row7_nbits_gt0, bitmap_mask);

  uint8x8_t bitmap_rows_10 = vpadd_u8(row1_nbits_gt0, row0_nbits_gt0);
  uint8x8_t bitmap_rows_32 = vpadd_u8(row3_nbits_gt0, row2_nbits_gt0);
  uint8x8_t bitmap_rows_54 = vpadd_u8(row5_nbits_gt0, row4_nbits_gt0);
  uint8x8_t bitmap_rows_76 = vpadd_u8(row7_nbits_gt0, row6_nbits_gt0);
  uint8x8_t bitmap_rows_3210 = vpadd_u8(bitmap_rows_32, bitmap_rows_10);
  uint8x8_t bitmap_rows_7654 = vpadd_u8(bitmap_rows_76, bitmap_rows_54);
  uint8x8_t bitmap = vpadd_u8(bitmap_rows_7654, bitmap_rows_3210);

  /* Shift left to remove DC bit. */
  bitmap = vreinterpret_u8_u64(vshl_n_u64(vreinterpret_u64_u8(bitmap), 1));
  /* Move bitmap to 32-bit scalar registers. */
  uint32_t bitmap_1_32 = vget_lane_u32(vreinterpret_u32_u8(bitmap), 1);
  uint32_t bitmap_33_63 = vget_lane_u32(vreinterpret_u32_u8(bitmap), 0);

  /* Set up state and bit buffer for output bitstream. */
  working_state *state_ptr = (working_state *)state;
  int free_bits = state_ptr->cur.free_bits;
  size_t put_buffer = state_ptr->cur.put_buffer;

  /* Encode DC coefficient. */

  unsigned int nbits = block_nbits[0];
  /* Emit Huffman-coded symbol and additional diff bits. */
  unsigned int diff = block_diff[0];
  PUT_CODE(dctbl->ehufco[nbits], dctbl->ehufsi[nbits], diff)

  /* Encode AC coefficients. */

  unsigned int r = 0;  /* r = run length of zeros */
  unsigned int i = 1;  /* i = number of coefficients encoded */
  /* Code and size information for a run length of 16 zero coefficients */
  const unsigned int code_0xf0 = actbl->ehufco[0xf0];
  const unsigned int size_0xf0 = actbl->ehufsi[0xf0];

  while (bitmap_1_32 != 0) {
    r = BUILTIN_CLZ(bitmap_1_32);
    i += r;
    bitmap_1_32 <<= r;
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
    bitmap_1_32 <<= 1;
  }

  r = 33 - i;
  i = 33;

  while (bitmap_33_63 != 0) {
    unsigned int leading_zeros = BUILTIN_CLZ(bitmap_33_63);
    r += leading_zeros;
    i += leading_zeros;
    bitmap_33_63 <<= leading_zeros;
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
    r = 0;
    i++;
    bitmap_33_63 <<= 1;
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
