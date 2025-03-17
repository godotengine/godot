/*
 * jdsample-neon.c - upsampling (Arm Neon)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, 2024, D. R. Commander.  All Rights Reserved.
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
 */

#define JPEG_INTERNALS
#include "../../src/jinclude.h"
#include "../../src/jpeglib.h"
#include "../../src/jsimd.h"
#include "../../src/jdct.h"
#include "../../src/jsimddct.h"
#include "../jsimd.h"
#include "neon-compat.h"

#include <arm_neon.h>


/* The diagram below shows a row of samples produced by h2v1 downsampling.
 *
 *                s0        s1        s2
 *            +---------+---------+---------+
 *            |         |         |         |
 *            | p0   p1 | p2   p3 | p4   p5 |
 *            |         |         |         |
 *            +---------+---------+---------+
 *
 * Samples s0-s2 were created by averaging the original pixel component values
 * centered at positions p0-p5 above.  To approximate those original pixel
 * component values, we proportionally blend the adjacent samples in each row.
 *
 * An upsampled pixel component value is computed by blending the sample
 * containing the pixel center with the nearest neighboring sample, in the
 * ratio 3:1.  For example:
 *     p1(upsampled) = 3/4 * s0 + 1/4 * s1
 *     p2(upsampled) = 3/4 * s1 + 1/4 * s0
 * When computing the first and last pixel component values in the row, there
 * is no adjacent sample to blend, so:
 *     p0(upsampled) = s0
 *     p5(upsampled) = s2
 */

void jsimd_h2v1_fancy_upsample_neon(int max_v_samp_factor,
                                    JDIMENSION downsampled_width,
                                    JSAMPARRAY input_data,
                                    JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow;
  unsigned colctr;
  /* Set up constants. */
  const uint16x8_t one_u16 = vdupq_n_u16(1);
  const uint8x8_t three_u8 = vdup_n_u8(3);

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];
    /* First pixel component value in this row of the original image */
    *outptr = (JSAMPLE)GETJSAMPLE(*inptr);

    /*    3/4 * containing sample + 1/4 * nearest neighboring sample
     * For p1: containing sample = s0, nearest neighboring sample = s1
     * For p2: containing sample = s1, nearest neighboring sample = s0
     */
    uint8x16_t s0 = vld1q_u8(inptr);
    uint8x16_t s1 = vld1q_u8(inptr + 1);
    /* Multiplication makes vectors twice as wide.  '_l' and '_h' suffixes
     * denote low half and high half respectively.
     */
    uint16x8_t s1_add_3s0_l =
      vmlal_u8(vmovl_u8(vget_low_u8(s1)), vget_low_u8(s0), three_u8);
    uint16x8_t s1_add_3s0_h =
      vmlal_u8(vmovl_u8(vget_high_u8(s1)), vget_high_u8(s0), three_u8);
    uint16x8_t s0_add_3s1_l =
      vmlal_u8(vmovl_u8(vget_low_u8(s0)), vget_low_u8(s1), three_u8);
    uint16x8_t s0_add_3s1_h =
      vmlal_u8(vmovl_u8(vget_high_u8(s0)), vget_high_u8(s1), three_u8);
    /* Add ordered dithering bias to odd pixel values. */
    s0_add_3s1_l = vaddq_u16(s0_add_3s1_l, one_u16);
    s0_add_3s1_h = vaddq_u16(s0_add_3s1_h, one_u16);

    /* The offset is initially 1, because the first pixel component has already
     * been stored.  However, in subsequent iterations of the SIMD loop, this
     * offset is (2 * colctr - 1) to stay within the bounds of the sample
     * buffers without having to resort to a slow scalar tail case for the last
     * (downsampled_width % 16) samples.  See "Creation of 2-D sample arrays"
     * in jmemmgr.c for more details.
     */
    unsigned outptr_offset = 1;
    uint8x16x2_t output_pixels;

    /* We use software pipelining to maximise performance.  The code indented
     * an extra two spaces begins the next iteration of the loop.
     */
    for (colctr = 16; colctr < downsampled_width; colctr += 16) {

        s0 = vld1q_u8(inptr + colctr - 1);
        s1 = vld1q_u8(inptr + colctr);

      /* Right-shift by 2 (divide by 4), narrow to 8-bit, and combine. */
      output_pixels.val[0] = vcombine_u8(vrshrn_n_u16(s1_add_3s0_l, 2),
                                         vrshrn_n_u16(s1_add_3s0_h, 2));
      output_pixels.val[1] = vcombine_u8(vshrn_n_u16(s0_add_3s1_l, 2),
                                         vshrn_n_u16(s0_add_3s1_h, 2));

        /* Multiplication makes vectors twice as wide.  '_l' and '_h' suffixes
         * denote low half and high half respectively.
         */
        s1_add_3s0_l =
          vmlal_u8(vmovl_u8(vget_low_u8(s1)), vget_low_u8(s0), three_u8);
        s1_add_3s0_h =
          vmlal_u8(vmovl_u8(vget_high_u8(s1)), vget_high_u8(s0), three_u8);
        s0_add_3s1_l =
          vmlal_u8(vmovl_u8(vget_low_u8(s0)), vget_low_u8(s1), three_u8);
        s0_add_3s1_h =
          vmlal_u8(vmovl_u8(vget_high_u8(s0)), vget_high_u8(s1), three_u8);
        /* Add ordered dithering bias to odd pixel values. */
        s0_add_3s1_l = vaddq_u16(s0_add_3s1_l, one_u16);
        s0_add_3s1_h = vaddq_u16(s0_add_3s1_h, one_u16);

      /* Store pixel component values to memory. */
      vst2q_u8(outptr + outptr_offset, output_pixels);
      outptr_offset = 2 * colctr - 1;
    }

    /* Complete the last iteration of the loop. */

    /* Right-shift by 2 (divide by 4), narrow to 8-bit, and combine. */
    output_pixels.val[0] = vcombine_u8(vrshrn_n_u16(s1_add_3s0_l, 2),
                                       vrshrn_n_u16(s1_add_3s0_h, 2));
    output_pixels.val[1] = vcombine_u8(vshrn_n_u16(s0_add_3s1_l, 2),
                                       vshrn_n_u16(s0_add_3s1_h, 2));
    /* Store pixel component values to memory. */
    vst2q_u8(outptr + outptr_offset, output_pixels);

    /* Last pixel component value in this row of the original image */
    outptr[2 * downsampled_width - 1] =
      GETJSAMPLE(inptr[downsampled_width - 1]);
  }
}


/* The diagram below shows an array of samples produced by h2v2 downsampling.
 *
 *                s0        s1        s2
 *            +---------+---------+---------+
 *            | p0   p1 | p2   p3 | p4   p5 |
 *       sA   |         |         |         |
 *            | p6   p7 | p8   p9 | p10  p11|
 *            +---------+---------+---------+
 *            | p12  p13| p14  p15| p16  p17|
 *       sB   |         |         |         |
 *            | p18  p19| p20  p21| p22  p23|
 *            +---------+---------+---------+
 *            | p24  p25| p26  p27| p28  p29|
 *       sC   |         |         |         |
 *            | p30  p31| p32  p33| p34  p35|
 *            +---------+---------+---------+
 *
 * Samples s0A-s2C were created by averaging the original pixel component
 * values centered at positions p0-p35 above.  To approximate one of those
 * original pixel component values, we proportionally blend the sample
 * containing the pixel center with the nearest neighboring samples in each
 * row, column, and diagonal.
 *
 * An upsampled pixel component value is computed by first blending the sample
 * containing the pixel center with the nearest neighboring samples in the
 * same column, in the ratio 3:1, and then blending each column sum with the
 * nearest neighboring column sum, in the ratio 3:1.  For example:
 *     p14(upsampled) = 3/4 * (3/4 * s1B + 1/4 * s1A) +
 *                      1/4 * (3/4 * s0B + 1/4 * s0A)
 *                    = 9/16 * s1B + 3/16 * s1A + 3/16 * s0B + 1/16 * s0A
 * When computing the first and last pixel component values in the row, there
 * is no horizontally adjacent sample to blend, so:
 *     p12(upsampled) = 3/4 * s0B + 1/4 * s0A
 *     p23(upsampled) = 3/4 * s2B + 1/4 * s2C
 * When computing the first and last pixel component values in the column,
 * there is no vertically adjacent sample to blend, so:
 *     p2(upsampled) = 3/4 * s1A + 1/4 * s0A
 *     p33(upsampled) = 3/4 * s1C + 1/4 * s2C
 * When computing the corner pixel component values, there is no adjacent
 * sample to blend, so:
 *     p0(upsampled) = s0A
 *     p35(upsampled) = s2C
 */

void jsimd_h2v2_fancy_upsample_neon(int max_v_samp_factor,
                                    JDIMENSION downsampled_width,
                                    JSAMPARRAY input_data,
                                    JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr0, inptr1, inptr2, outptr0, outptr1;
  int inrow, outrow;
  unsigned colctr;
  /* Set up constants. */
  const uint16x8_t seven_u16 = vdupq_n_u16(7);
  const uint8x8_t three_u8 = vdup_n_u8(3);
  const uint16x8_t three_u16 = vdupq_n_u16(3);

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    inptr0 = input_data[inrow - 1];
    inptr1 = input_data[inrow];
    inptr2 = input_data[inrow + 1];
    /* Suffixes 0 and 1 denote the upper and lower rows of output pixels,
     * respectively.
     */
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    /* First pixel component value in this row of the original image */
    int s0colsum0 = GETJSAMPLE(*inptr1) * 3 + GETJSAMPLE(*inptr0);
    *outptr0 = (JSAMPLE)((s0colsum0 * 4 + 8) >> 4);
    int s0colsum1 = GETJSAMPLE(*inptr1) * 3 + GETJSAMPLE(*inptr2);
    *outptr1 = (JSAMPLE)((s0colsum1 * 4 + 8) >> 4);

    /* Step 1: Blend samples vertically in columns s0 and s1.
     * Leave the divide by 4 until the end, when it can be done for both
     * dimensions at once, right-shifting by 4.
     */

    /* Load and compute s0colsum0 and s0colsum1. */
    uint8x16_t s0A = vld1q_u8(inptr0);
    uint8x16_t s0B = vld1q_u8(inptr1);
    uint8x16_t s0C = vld1q_u8(inptr2);
    /* Multiplication makes vectors twice as wide.  '_l' and '_h' suffixes
     * denote low half and high half respectively.
     */
    uint16x8_t s0colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(s0A)),
                                      vget_low_u8(s0B), three_u8);
    uint16x8_t s0colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(s0A)),
                                      vget_high_u8(s0B), three_u8);
    uint16x8_t s0colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(s0C)),
                                      vget_low_u8(s0B), three_u8);
    uint16x8_t s0colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(s0C)),
                                      vget_high_u8(s0B), three_u8);
    /* Load and compute s1colsum0 and s1colsum1. */
    uint8x16_t s1A = vld1q_u8(inptr0 + 1);
    uint8x16_t s1B = vld1q_u8(inptr1 + 1);
    uint8x16_t s1C = vld1q_u8(inptr2 + 1);
    uint16x8_t s1colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(s1A)),
                                      vget_low_u8(s1B), three_u8);
    uint16x8_t s1colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(s1A)),
                                      vget_high_u8(s1B), three_u8);
    uint16x8_t s1colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(s1C)),
                                      vget_low_u8(s1B), three_u8);
    uint16x8_t s1colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(s1C)),
                                      vget_high_u8(s1B), three_u8);

    /* Step 2: Blend the already-blended columns. */

    uint16x8_t output0_p1_l = vmlaq_u16(s1colsum0_l, s0colsum0_l, three_u16);
    uint16x8_t output0_p1_h = vmlaq_u16(s1colsum0_h, s0colsum0_h, three_u16);
    uint16x8_t output0_p2_l = vmlaq_u16(s0colsum0_l, s1colsum0_l, three_u16);
    uint16x8_t output0_p2_h = vmlaq_u16(s0colsum0_h, s1colsum0_h, three_u16);
    uint16x8_t output1_p1_l = vmlaq_u16(s1colsum1_l, s0colsum1_l, three_u16);
    uint16x8_t output1_p1_h = vmlaq_u16(s1colsum1_h, s0colsum1_h, three_u16);
    uint16x8_t output1_p2_l = vmlaq_u16(s0colsum1_l, s1colsum1_l, three_u16);
    uint16x8_t output1_p2_h = vmlaq_u16(s0colsum1_h, s1colsum1_h, three_u16);
    /* Add ordered dithering bias to odd pixel values. */
    output0_p1_l = vaddq_u16(output0_p1_l, seven_u16);
    output0_p1_h = vaddq_u16(output0_p1_h, seven_u16);
    output1_p1_l = vaddq_u16(output1_p1_l, seven_u16);
    output1_p1_h = vaddq_u16(output1_p1_h, seven_u16);
    /* Right-shift by 4 (divide by 16), narrow to 8-bit, and combine. */
    uint8x16x2_t output_pixels0 = { {
      vcombine_u8(vshrn_n_u16(output0_p1_l, 4), vshrn_n_u16(output0_p1_h, 4)),
      vcombine_u8(vrshrn_n_u16(output0_p2_l, 4), vrshrn_n_u16(output0_p2_h, 4))
    } };
    uint8x16x2_t output_pixels1 = { {
      vcombine_u8(vshrn_n_u16(output1_p1_l, 4), vshrn_n_u16(output1_p1_h, 4)),
      vcombine_u8(vrshrn_n_u16(output1_p2_l, 4), vrshrn_n_u16(output1_p2_h, 4))
    } };

    /* Store pixel component values to memory.
     * The minimum size of the output buffer for each row is 64 bytes => no
     * need to worry about buffer overflow here.  See "Creation of 2-D sample
     * arrays" in jmemmgr.c for more details.
     */
    vst2q_u8(outptr0 + 1, output_pixels0);
    vst2q_u8(outptr1 + 1, output_pixels1);

    /* The first pixel of the image shifted our loads and stores by one byte.
     * We have to re-align on a 32-byte boundary at some point before the end
     * of the row (we do it now on the 32/33 pixel boundary) to stay within the
     * bounds of the sample buffers without having to resort to a slow scalar
     * tail case for the last (downsampled_width % 16) samples.  See "Creation
     * of 2-D sample arrays" in jmemmgr.c for more details.
     */
    for (colctr = 16; colctr < downsampled_width; colctr += 16) {
      /* Step 1: Blend samples vertically in columns s0 and s1. */

      /* Load and compute s0colsum0 and s0colsum1. */
      s0A = vld1q_u8(inptr0 + colctr - 1);
      s0B = vld1q_u8(inptr1 + colctr - 1);
      s0C = vld1q_u8(inptr2 + colctr - 1);
      s0colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(s0A)), vget_low_u8(s0B),
                             three_u8);
      s0colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(s0A)), vget_high_u8(s0B),
                             three_u8);
      s0colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(s0C)), vget_low_u8(s0B),
                             three_u8);
      s0colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(s0C)), vget_high_u8(s0B),
                             three_u8);
      /* Load and compute s1colsum0 and s1colsum1. */
      s1A = vld1q_u8(inptr0 + colctr);
      s1B = vld1q_u8(inptr1 + colctr);
      s1C = vld1q_u8(inptr2 + colctr);
      s1colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(s1A)), vget_low_u8(s1B),
                             three_u8);
      s1colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(s1A)), vget_high_u8(s1B),
                             three_u8);
      s1colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(s1C)), vget_low_u8(s1B),
                             three_u8);
      s1colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(s1C)), vget_high_u8(s1B),
                             three_u8);

      /* Step 2: Blend the already-blended columns. */

      output0_p1_l = vmlaq_u16(s1colsum0_l, s0colsum0_l, three_u16);
      output0_p1_h = vmlaq_u16(s1colsum0_h, s0colsum0_h, three_u16);
      output0_p2_l = vmlaq_u16(s0colsum0_l, s1colsum0_l, three_u16);
      output0_p2_h = vmlaq_u16(s0colsum0_h, s1colsum0_h, three_u16);
      output1_p1_l = vmlaq_u16(s1colsum1_l, s0colsum1_l, three_u16);
      output1_p1_h = vmlaq_u16(s1colsum1_h, s0colsum1_h, three_u16);
      output1_p2_l = vmlaq_u16(s0colsum1_l, s1colsum1_l, three_u16);
      output1_p2_h = vmlaq_u16(s0colsum1_h, s1colsum1_h, three_u16);
      /* Add ordered dithering bias to odd pixel values. */
      output0_p1_l = vaddq_u16(output0_p1_l, seven_u16);
      output0_p1_h = vaddq_u16(output0_p1_h, seven_u16);
      output1_p1_l = vaddq_u16(output1_p1_l, seven_u16);
      output1_p1_h = vaddq_u16(output1_p1_h, seven_u16);
      /* Right-shift by 4 (divide by 16), narrow to 8-bit, and combine. */
      output_pixels0.val[0] = vcombine_u8(vshrn_n_u16(output0_p1_l, 4),
                                          vshrn_n_u16(output0_p1_h, 4));
      output_pixels0.val[1] = vcombine_u8(vrshrn_n_u16(output0_p2_l, 4),
                                          vrshrn_n_u16(output0_p2_h, 4));
      output_pixels1.val[0] = vcombine_u8(vshrn_n_u16(output1_p1_l, 4),
                                          vshrn_n_u16(output1_p1_h, 4));
      output_pixels1.val[1] = vcombine_u8(vrshrn_n_u16(output1_p2_l, 4),
                                          vrshrn_n_u16(output1_p2_h, 4));
      /* Store pixel component values to memory. */
      vst2q_u8(outptr0 + 2 * colctr - 1, output_pixels0);
      vst2q_u8(outptr1 + 2 * colctr - 1, output_pixels1);
    }

    /* Last pixel component value in this row of the original image */
    int s1colsum0 = GETJSAMPLE(inptr1[downsampled_width - 1]) * 3 +
                    GETJSAMPLE(inptr0[downsampled_width - 1]);
    outptr0[2 * downsampled_width - 1] = (JSAMPLE)((s1colsum0 * 4 + 7) >> 4);
    int s1colsum1 = GETJSAMPLE(inptr1[downsampled_width - 1]) * 3 +
                    GETJSAMPLE(inptr2[downsampled_width - 1]);
    outptr1[2 * downsampled_width - 1] = (JSAMPLE)((s1colsum1 * 4 + 7) >> 4);
    inrow++;
  }
}


/* The diagram below shows a column of samples produced by h1v2 downsampling
 * (or by losslessly rotating or transposing an h2v1-downsampled image.)
 *
 *            +---------+
 *            |   p0    |
 *     sA     |         |
 *            |   p1    |
 *            +---------+
 *            |   p2    |
 *     sB     |         |
 *            |   p3    |
 *            +---------+
 *            |   p4    |
 *     sC     |         |
 *            |   p5    |
 *            +---------+
 *
 * Samples sA-sC were created by averaging the original pixel component values
 * centered at positions p0-p5 above.  To approximate those original pixel
 * component values, we proportionally blend the adjacent samples in each
 * column.
 *
 * An upsampled pixel component value is computed by blending the sample
 * containing the pixel center with the nearest neighboring sample, in the
 * ratio 3:1.  For example:
 *     p1(upsampled) = 3/4 * sA + 1/4 * sB
 *     p2(upsampled) = 3/4 * sB + 1/4 * sA
 * When computing the first and last pixel component values in the column,
 * there is no adjacent sample to blend, so:
 *     p0(upsampled) = sA
 *     p5(upsampled) = sC
 */

void jsimd_h1v2_fancy_upsample_neon(int max_v_samp_factor,
                                    JDIMENSION downsampled_width,
                                    JSAMPARRAY input_data,
                                    JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr0, inptr1, inptr2, outptr0, outptr1;
  int inrow, outrow;
  unsigned colctr;
  /* Set up constants. */
  const uint16x8_t one_u16 = vdupq_n_u16(1);
  const uint8x8_t three_u8 = vdup_n_u8(3);

  inrow = outrow = 0;
  while (outrow < max_v_samp_factor) {
    inptr0 = input_data[inrow - 1];
    inptr1 = input_data[inrow];
    inptr2 = input_data[inrow + 1];
    /* Suffixes 0 and 1 denote the upper and lower rows of output pixels,
     * respectively.
     */
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];
    inrow++;

    /* The size of the input and output buffers is always a multiple of 32
     * bytes => no need to worry about buffer overflow when reading/writing
     * memory.  See "Creation of 2-D sample arrays" in jmemmgr.c for more
     * details.
     */
    for (colctr = 0; colctr < downsampled_width; colctr += 16) {
      /* Load samples. */
      uint8x16_t sA = vld1q_u8(inptr0 + colctr);
      uint8x16_t sB = vld1q_u8(inptr1 + colctr);
      uint8x16_t sC = vld1q_u8(inptr2 + colctr);
      /* Blend samples vertically. */
      uint16x8_t colsum0_l = vmlal_u8(vmovl_u8(vget_low_u8(sA)),
                                      vget_low_u8(sB), three_u8);
      uint16x8_t colsum0_h = vmlal_u8(vmovl_u8(vget_high_u8(sA)),
                                      vget_high_u8(sB), three_u8);
      uint16x8_t colsum1_l = vmlal_u8(vmovl_u8(vget_low_u8(sC)),
                                      vget_low_u8(sB), three_u8);
      uint16x8_t colsum1_h = vmlal_u8(vmovl_u8(vget_high_u8(sC)),
                                      vget_high_u8(sB), three_u8);
      /* Add ordered dithering bias to pixel values in even output rows. */
      colsum0_l = vaddq_u16(colsum0_l, one_u16);
      colsum0_h = vaddq_u16(colsum0_h, one_u16);
      /* Right-shift by 2 (divide by 4), narrow to 8-bit, and combine. */
      uint8x16_t output_pixels0 = vcombine_u8(vshrn_n_u16(colsum0_l, 2),
                                              vshrn_n_u16(colsum0_h, 2));
      uint8x16_t output_pixels1 = vcombine_u8(vrshrn_n_u16(colsum1_l, 2),
                                              vrshrn_n_u16(colsum1_h, 2));
      /* Store pixel component values to memory. */
      vst1q_u8(outptr0 + colctr, output_pixels0);
      vst1q_u8(outptr1 + colctr, output_pixels1);
    }
  }
}


/* The diagram below shows a row of samples produced by h2v1 downsampling.
 *
 *                s0        s1
 *            +---------+---------+
 *            |         |         |
 *            | p0   p1 | p2   p3 |
 *            |         |         |
 *            +---------+---------+
 *
 * Samples s0 and s1 were created by averaging the original pixel component
 * values centered at positions p0-p3 above.  To approximate those original
 * pixel component values, we duplicate the samples horizontally:
 *     p0(upsampled) = p1(upsampled) = s0
 *     p2(upsampled) = p3(upsampled) = s1
 */

void jsimd_h2v1_upsample_neon(int max_v_samp_factor, JDIMENSION output_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow;
  unsigned colctr;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];
    for (colctr = 0; 2 * colctr < output_width; colctr += 16) {
      uint8x16_t samples = vld1q_u8(inptr + colctr);
      /* Duplicate the samples.  The store operation below interleaves them so
       * that adjacent pixel component values take on the same sample value,
       * per above.
       */
      uint8x16x2_t output_pixels = { { samples, samples } };
      /* Store pixel component values to memory.
       * Due to the way sample buffers are allocated, we don't need to worry
       * about tail cases when output_width is not a multiple of 32.  See
       * "Creation of 2-D sample arrays" in jmemmgr.c for details.
       */
      vst2q_u8(outptr + 2 * colctr, output_pixels);
    }
  }
}


/* The diagram below shows an array of samples produced by h2v2 downsampling.
 *
 *                s0        s1
 *            +---------+---------+
 *            | p0   p1 | p2   p3 |
 *       sA   |         |         |
 *            | p4   p5 | p6   p7 |
 *            +---------+---------+
 *            | p8   p9 | p10  p11|
 *       sB   |         |         |
 *            | p12  p13| p14  p15|
 *            +---------+---------+
 *
 * Samples s0A-s1B were created by averaging the original pixel component
 * values centered at positions p0-p15 above.  To approximate those original
 * pixel component values, we duplicate the samples both horizontally and
 * vertically:
 *     p0(upsampled) = p1(upsampled) = p4(upsampled) = p5(upsampled) = s0A
 *     p2(upsampled) = p3(upsampled) = p6(upsampled) = p7(upsampled) = s1A
 *     p8(upsampled) = p9(upsampled) = p12(upsampled) = p13(upsampled) = s0B
 *     p10(upsampled) = p11(upsampled) = p14(upsampled) = p15(upsampled) = s1B
 */

void jsimd_h2v2_upsample_neon(int max_v_samp_factor, JDIMENSION output_width,
                              JSAMPARRAY input_data,
                              JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr0, outptr1;
  int inrow, outrow;
  unsigned colctr;

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    for (colctr = 0; 2 * colctr < output_width; colctr += 16) {
      uint8x16_t samples = vld1q_u8(inptr + colctr);
      /* Duplicate the samples.  The store operation below interleaves them so
       * that adjacent pixel component values take on the same sample value,
       * per above.
       */
      uint8x16x2_t output_pixels = { { samples, samples } };
      /* Store pixel component values for both output rows to memory.
       * Due to the way sample buffers are allocated, we don't need to worry
       * about tail cases when output_width is not a multiple of 32.  See
       * "Creation of 2-D sample arrays" in jmemmgr.c for details.
       */
      vst2q_u8(outptr0 + 2 * colctr, output_pixels);
      vst2q_u8(outptr1 + 2 * colctr, output_pixels);
    }
  }
}
