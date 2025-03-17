/*
 * jccolext-neon.c - colorspace conversion (64-bit Arm Neon)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
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

/* This file is included by jccolor-neon.c */


/* RGB -> YCbCr conversion is defined by the following equations:
 *    Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *    Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128
 *    Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + 128
 *
 * Avoid floating point arithmetic by using shifted integer constants:
 *    0.29899597 = 19595 * 2^-16
 *    0.58700561 = 38470 * 2^-16
 *    0.11399841 =  7471 * 2^-16
 *    0.16874695 = 11059 * 2^-16
 *    0.33125305 = 21709 * 2^-16
 *    0.50000000 = 32768 * 2^-16
 *    0.41868592 = 27439 * 2^-16
 *    0.08131409 =  5329 * 2^-16
 * These constants are defined in jccolor-neon.c
 *
 * We add the fixed-point equivalent of 0.5 to Cb and Cr, which effectively
 * rounds up or down the result via integer truncation.
 */

void jsimd_rgb_ycc_convert_neon(JDIMENSION image_width, JSAMPARRAY input_buf,
                                JSAMPIMAGE output_buf, JDIMENSION output_row,
                                int num_rows)
{
  /* Pointer to RGB(X/A) input data */
  JSAMPROW inptr;
  /* Pointers to Y, Cb, and Cr output data */
  JSAMPROW outptr0, outptr1, outptr2;
  /* Allocate temporary buffer for final (image_width % 16) pixels in row. */
  ALIGN(16) uint8_t tmp_buf[16 * RGB_PIXELSIZE];

  /* Set up conversion constants. */
  const uint16x8_t consts = vld1q_u16(jsimd_rgb_ycc_neon_consts);
  const uint32x4_t scaled_128_5 = vdupq_n_u32((128 << 16) + 32767);

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr0 = output_buf[0][output_row];
    outptr1 = output_buf[1][output_row];
    outptr2 = output_buf[2][output_row];
    output_row++;

    int cols_remaining = image_width;
    for (; cols_remaining >= 16; cols_remaining -= 16) {

#if RGB_PIXELSIZE == 4
      uint8x16x4_t input_pixels = vld4q_u8(inptr);
#else
      uint8x16x3_t input_pixels = vld3q_u8(inptr);
#endif
      uint16x8_t r_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_RED]));
      uint16x8_t g_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_GREEN]));
      uint16x8_t b_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_BLUE]));
      uint16x8_t r_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_RED]));
      uint16x8_t g_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_GREEN]));
      uint16x8_t b_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_BLUE]));

      /* Compute Y = 0.29900 * R + 0.58700 * G + 0.11400 * B */
      uint32x4_t y_ll = vmull_laneq_u16(vget_low_u16(r_l), consts, 0);
      y_ll = vmlal_laneq_u16(y_ll, vget_low_u16(g_l), consts, 1);
      y_ll = vmlal_laneq_u16(y_ll, vget_low_u16(b_l), consts, 2);
      uint32x4_t y_lh = vmull_laneq_u16(vget_high_u16(r_l), consts, 0);
      y_lh = vmlal_laneq_u16(y_lh, vget_high_u16(g_l), consts, 1);
      y_lh = vmlal_laneq_u16(y_lh, vget_high_u16(b_l), consts, 2);
      uint32x4_t y_hl = vmull_laneq_u16(vget_low_u16(r_h), consts, 0);
      y_hl = vmlal_laneq_u16(y_hl, vget_low_u16(g_h), consts, 1);
      y_hl = vmlal_laneq_u16(y_hl, vget_low_u16(b_h), consts, 2);
      uint32x4_t y_hh = vmull_laneq_u16(vget_high_u16(r_h), consts, 0);
      y_hh = vmlal_laneq_u16(y_hh, vget_high_u16(g_h), consts, 1);
      y_hh = vmlal_laneq_u16(y_hh, vget_high_u16(b_h), consts, 2);

      /* Compute Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128 */
      uint32x4_t cb_ll = scaled_128_5;
      cb_ll = vmlsl_laneq_u16(cb_ll, vget_low_u16(r_l), consts, 3);
      cb_ll = vmlsl_laneq_u16(cb_ll, vget_low_u16(g_l), consts, 4);
      cb_ll = vmlal_laneq_u16(cb_ll, vget_low_u16(b_l), consts, 5);
      uint32x4_t cb_lh = scaled_128_5;
      cb_lh = vmlsl_laneq_u16(cb_lh, vget_high_u16(r_l), consts, 3);
      cb_lh = vmlsl_laneq_u16(cb_lh, vget_high_u16(g_l), consts, 4);
      cb_lh = vmlal_laneq_u16(cb_lh, vget_high_u16(b_l), consts, 5);
      uint32x4_t cb_hl = scaled_128_5;
      cb_hl = vmlsl_laneq_u16(cb_hl, vget_low_u16(r_h), consts, 3);
      cb_hl = vmlsl_laneq_u16(cb_hl, vget_low_u16(g_h), consts, 4);
      cb_hl = vmlal_laneq_u16(cb_hl, vget_low_u16(b_h), consts, 5);
      uint32x4_t cb_hh = scaled_128_5;
      cb_hh = vmlsl_laneq_u16(cb_hh, vget_high_u16(r_h), consts, 3);
      cb_hh = vmlsl_laneq_u16(cb_hh, vget_high_u16(g_h), consts, 4);
      cb_hh = vmlal_laneq_u16(cb_hh, vget_high_u16(b_h), consts, 5);

      /* Compute Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B  + 128 */
      uint32x4_t cr_ll = scaled_128_5;
      cr_ll = vmlal_laneq_u16(cr_ll, vget_low_u16(r_l), consts, 5);
      cr_ll = vmlsl_laneq_u16(cr_ll, vget_low_u16(g_l), consts, 6);
      cr_ll = vmlsl_laneq_u16(cr_ll, vget_low_u16(b_l), consts, 7);
      uint32x4_t cr_lh = scaled_128_5;
      cr_lh = vmlal_laneq_u16(cr_lh, vget_high_u16(r_l), consts, 5);
      cr_lh = vmlsl_laneq_u16(cr_lh, vget_high_u16(g_l), consts, 6);
      cr_lh = vmlsl_laneq_u16(cr_lh, vget_high_u16(b_l), consts, 7);
      uint32x4_t cr_hl = scaled_128_5;
      cr_hl = vmlal_laneq_u16(cr_hl, vget_low_u16(r_h), consts, 5);
      cr_hl = vmlsl_laneq_u16(cr_hl, vget_low_u16(g_h), consts, 6);
      cr_hl = vmlsl_laneq_u16(cr_hl, vget_low_u16(b_h), consts, 7);
      uint32x4_t cr_hh = scaled_128_5;
      cr_hh = vmlal_laneq_u16(cr_hh, vget_high_u16(r_h), consts, 5);
      cr_hh = vmlsl_laneq_u16(cr_hh, vget_high_u16(g_h), consts, 6);
      cr_hh = vmlsl_laneq_u16(cr_hh, vget_high_u16(b_h), consts, 7);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y_l = vcombine_u16(vrshrn_n_u32(y_ll, 16),
                                    vrshrn_n_u32(y_lh, 16));
      uint16x8_t y_h = vcombine_u16(vrshrn_n_u32(y_hl, 16),
                                    vrshrn_n_u32(y_hh, 16));
      /* Descale Cb values (right shift) and narrow to 16-bit. */
      uint16x8_t cb_l = vcombine_u16(vshrn_n_u32(cb_ll, 16),
                                     vshrn_n_u32(cb_lh, 16));
      uint16x8_t cb_h = vcombine_u16(vshrn_n_u32(cb_hl, 16),
                                     vshrn_n_u32(cb_hh, 16));
      /* Descale Cr values (right shift) and narrow to 16-bit. */
      uint16x8_t cr_l = vcombine_u16(vshrn_n_u32(cr_ll, 16),
                                     vshrn_n_u32(cr_lh, 16));
      uint16x8_t cr_h = vcombine_u16(vshrn_n_u32(cr_hl, 16),
                                     vshrn_n_u32(cr_hh, 16));
      /* Narrow Y, Cb, and Cr values to 8-bit and store to memory.  Buffer
       * overwrite is permitted up to the next multiple of ALIGN_SIZE bytes.
       */
      vst1q_u8(outptr0, vcombine_u8(vmovn_u16(y_l), vmovn_u16(y_h)));
      vst1q_u8(outptr1, vcombine_u8(vmovn_u16(cb_l), vmovn_u16(cb_h)));
      vst1q_u8(outptr2, vcombine_u8(vmovn_u16(cr_l), vmovn_u16(cr_h)));

      /* Increment pointers. */
      inptr += (16 * RGB_PIXELSIZE);
      outptr0 += 16;
      outptr1 += 16;
      outptr2 += 16;
    }

    if (cols_remaining > 8) {
      /* To prevent buffer overread by the vector load instructions, the last
       * (image_width % 16) columns of data are first memcopied to a temporary
       * buffer large enough to accommodate the vector load.
       */
      memcpy(tmp_buf, inptr, cols_remaining * RGB_PIXELSIZE);
      inptr = tmp_buf;

#if RGB_PIXELSIZE == 4
      uint8x16x4_t input_pixels = vld4q_u8(inptr);
#else
      uint8x16x3_t input_pixels = vld3q_u8(inptr);
#endif
      uint16x8_t r_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_RED]));
      uint16x8_t g_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_GREEN]));
      uint16x8_t b_l = vmovl_u8(vget_low_u8(input_pixels.val[RGB_BLUE]));
      uint16x8_t r_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_RED]));
      uint16x8_t g_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_GREEN]));
      uint16x8_t b_h = vmovl_u8(vget_high_u8(input_pixels.val[RGB_BLUE]));

      /* Compute Y = 0.29900 * R + 0.58700 * G + 0.11400 * B */
      uint32x4_t y_ll = vmull_laneq_u16(vget_low_u16(r_l), consts, 0);
      y_ll = vmlal_laneq_u16(y_ll, vget_low_u16(g_l), consts, 1);
      y_ll = vmlal_laneq_u16(y_ll, vget_low_u16(b_l), consts, 2);
      uint32x4_t y_lh = vmull_laneq_u16(vget_high_u16(r_l), consts, 0);
      y_lh = vmlal_laneq_u16(y_lh, vget_high_u16(g_l), consts, 1);
      y_lh = vmlal_laneq_u16(y_lh, vget_high_u16(b_l), consts, 2);
      uint32x4_t y_hl = vmull_laneq_u16(vget_low_u16(r_h), consts, 0);
      y_hl = vmlal_laneq_u16(y_hl, vget_low_u16(g_h), consts, 1);
      y_hl = vmlal_laneq_u16(y_hl, vget_low_u16(b_h), consts, 2);
      uint32x4_t y_hh = vmull_laneq_u16(vget_high_u16(r_h), consts, 0);
      y_hh = vmlal_laneq_u16(y_hh, vget_high_u16(g_h), consts, 1);
      y_hh = vmlal_laneq_u16(y_hh, vget_high_u16(b_h), consts, 2);

      /* Compute Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128 */
      uint32x4_t cb_ll = scaled_128_5;
      cb_ll = vmlsl_laneq_u16(cb_ll, vget_low_u16(r_l), consts, 3);
      cb_ll = vmlsl_laneq_u16(cb_ll, vget_low_u16(g_l), consts, 4);
      cb_ll = vmlal_laneq_u16(cb_ll, vget_low_u16(b_l), consts, 5);
      uint32x4_t cb_lh = scaled_128_5;
      cb_lh = vmlsl_laneq_u16(cb_lh, vget_high_u16(r_l), consts, 3);
      cb_lh = vmlsl_laneq_u16(cb_lh, vget_high_u16(g_l), consts, 4);
      cb_lh = vmlal_laneq_u16(cb_lh, vget_high_u16(b_l), consts, 5);
      uint32x4_t cb_hl = scaled_128_5;
      cb_hl = vmlsl_laneq_u16(cb_hl, vget_low_u16(r_h), consts, 3);
      cb_hl = vmlsl_laneq_u16(cb_hl, vget_low_u16(g_h), consts, 4);
      cb_hl = vmlal_laneq_u16(cb_hl, vget_low_u16(b_h), consts, 5);
      uint32x4_t cb_hh = scaled_128_5;
      cb_hh = vmlsl_laneq_u16(cb_hh, vget_high_u16(r_h), consts, 3);
      cb_hh = vmlsl_laneq_u16(cb_hh, vget_high_u16(g_h), consts, 4);
      cb_hh = vmlal_laneq_u16(cb_hh, vget_high_u16(b_h), consts, 5);

      /* Compute Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B  + 128 */
      uint32x4_t cr_ll = scaled_128_5;
      cr_ll = vmlal_laneq_u16(cr_ll, vget_low_u16(r_l), consts, 5);
      cr_ll = vmlsl_laneq_u16(cr_ll, vget_low_u16(g_l), consts, 6);
      cr_ll = vmlsl_laneq_u16(cr_ll, vget_low_u16(b_l), consts, 7);
      uint32x4_t cr_lh = scaled_128_5;
      cr_lh = vmlal_laneq_u16(cr_lh, vget_high_u16(r_l), consts, 5);
      cr_lh = vmlsl_laneq_u16(cr_lh, vget_high_u16(g_l), consts, 6);
      cr_lh = vmlsl_laneq_u16(cr_lh, vget_high_u16(b_l), consts, 7);
      uint32x4_t cr_hl = scaled_128_5;
      cr_hl = vmlal_laneq_u16(cr_hl, vget_low_u16(r_h), consts, 5);
      cr_hl = vmlsl_laneq_u16(cr_hl, vget_low_u16(g_h), consts, 6);
      cr_hl = vmlsl_laneq_u16(cr_hl, vget_low_u16(b_h), consts, 7);
      uint32x4_t cr_hh = scaled_128_5;
      cr_hh = vmlal_laneq_u16(cr_hh, vget_high_u16(r_h), consts, 5);
      cr_hh = vmlsl_laneq_u16(cr_hh, vget_high_u16(g_h), consts, 6);
      cr_hh = vmlsl_laneq_u16(cr_hh, vget_high_u16(b_h), consts, 7);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y_l = vcombine_u16(vrshrn_n_u32(y_ll, 16),
                                    vrshrn_n_u32(y_lh, 16));
      uint16x8_t y_h = vcombine_u16(vrshrn_n_u32(y_hl, 16),
                                    vrshrn_n_u32(y_hh, 16));
      /* Descale Cb values (right shift) and narrow to 16-bit. */
      uint16x8_t cb_l = vcombine_u16(vshrn_n_u32(cb_ll, 16),
                                     vshrn_n_u32(cb_lh, 16));
      uint16x8_t cb_h = vcombine_u16(vshrn_n_u32(cb_hl, 16),
                                     vshrn_n_u32(cb_hh, 16));
      /* Descale Cr values (right shift) and narrow to 16-bit. */
      uint16x8_t cr_l = vcombine_u16(vshrn_n_u32(cr_ll, 16),
                                     vshrn_n_u32(cr_lh, 16));
      uint16x8_t cr_h = vcombine_u16(vshrn_n_u32(cr_hl, 16),
                                     vshrn_n_u32(cr_hh, 16));
      /* Narrow Y, Cb, and Cr values to 8-bit and store to memory.  Buffer
       * overwrite is permitted up to the next multiple of ALIGN_SIZE bytes.
       */
      vst1q_u8(outptr0, vcombine_u8(vmovn_u16(y_l), vmovn_u16(y_h)));
      vst1q_u8(outptr1, vcombine_u8(vmovn_u16(cb_l), vmovn_u16(cb_h)));
      vst1q_u8(outptr2, vcombine_u8(vmovn_u16(cr_l), vmovn_u16(cr_h)));

    } else if (cols_remaining > 0) {
      /* To prevent buffer overread by the vector load instructions, the last
       * (image_width % 8) columns of data are first memcopied to a temporary
       * buffer large enough to accommodate the vector load.
       */
      memcpy(tmp_buf, inptr, cols_remaining * RGB_PIXELSIZE);
      inptr = tmp_buf;

#if RGB_PIXELSIZE == 4
      uint8x8x4_t input_pixels = vld4_u8(inptr);
#else
      uint8x8x3_t input_pixels = vld3_u8(inptr);
#endif
      uint16x8_t r = vmovl_u8(input_pixels.val[RGB_RED]);
      uint16x8_t g = vmovl_u8(input_pixels.val[RGB_GREEN]);
      uint16x8_t b = vmovl_u8(input_pixels.val[RGB_BLUE]);

      /* Compute Y = 0.29900 * R + 0.58700 * G + 0.11400 * B */
      uint32x4_t y_l = vmull_laneq_u16(vget_low_u16(r), consts, 0);
      y_l = vmlal_laneq_u16(y_l, vget_low_u16(g), consts, 1);
      y_l = vmlal_laneq_u16(y_l, vget_low_u16(b), consts, 2);
      uint32x4_t y_h = vmull_laneq_u16(vget_high_u16(r), consts, 0);
      y_h = vmlal_laneq_u16(y_h, vget_high_u16(g), consts, 1);
      y_h = vmlal_laneq_u16(y_h, vget_high_u16(b), consts, 2);

      /* Compute Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128 */
      uint32x4_t cb_l = scaled_128_5;
      cb_l = vmlsl_laneq_u16(cb_l, vget_low_u16(r), consts, 3);
      cb_l = vmlsl_laneq_u16(cb_l, vget_low_u16(g), consts, 4);
      cb_l = vmlal_laneq_u16(cb_l, vget_low_u16(b), consts, 5);
      uint32x4_t cb_h = scaled_128_5;
      cb_h = vmlsl_laneq_u16(cb_h, vget_high_u16(r), consts, 3);
      cb_h = vmlsl_laneq_u16(cb_h, vget_high_u16(g), consts, 4);
      cb_h = vmlal_laneq_u16(cb_h, vget_high_u16(b), consts, 5);

      /* Compute Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B  + 128 */
      uint32x4_t cr_l = scaled_128_5;
      cr_l = vmlal_laneq_u16(cr_l, vget_low_u16(r), consts, 5);
      cr_l = vmlsl_laneq_u16(cr_l, vget_low_u16(g), consts, 6);
      cr_l = vmlsl_laneq_u16(cr_l, vget_low_u16(b), consts, 7);
      uint32x4_t cr_h = scaled_128_5;
      cr_h = vmlal_laneq_u16(cr_h, vget_high_u16(r), consts, 5);
      cr_h = vmlsl_laneq_u16(cr_h, vget_high_u16(g), consts, 6);
      cr_h = vmlsl_laneq_u16(cr_h, vget_high_u16(b), consts, 7);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y_u16 = vcombine_u16(vrshrn_n_u32(y_l, 16),
                                      vrshrn_n_u32(y_h, 16));
      /* Descale Cb values (right shift) and narrow to 16-bit. */
      uint16x8_t cb_u16 = vcombine_u16(vshrn_n_u32(cb_l, 16),
                                       vshrn_n_u32(cb_h, 16));
      /* Descale Cr values (right shift) and narrow to 16-bit. */
      uint16x8_t cr_u16 = vcombine_u16(vshrn_n_u32(cr_l, 16),
                                       vshrn_n_u32(cr_h, 16));
      /* Narrow Y, Cb, and Cr values to 8-bit and store to memory.  Buffer
       * overwrite is permitted up to the next multiple of ALIGN_SIZE bytes.
       */
      vst1_u8(outptr0, vmovn_u16(y_u16));
      vst1_u8(outptr1, vmovn_u16(cb_u16));
      vst1_u8(outptr2, vmovn_u16(cr_u16));
    }
  }
}
