/*
 * jccolext-neon.c - colorspace conversion (32-bit Arm Neon)
 *
 * Copyright (C) 2020, Arm Limited.  All Rights Reserved.
 * Copyright (C) 2020, D. R. Commander.  All Rights Reserved.
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
  /* Allocate temporary buffer for final (image_width % 8) pixels in row. */
  ALIGN(16) uint8_t tmp_buf[8 * RGB_PIXELSIZE];

  /* Set up conversion constants. */
#ifdef HAVE_VLD1_U16_X2
  const uint16x4x2_t consts = vld1_u16_x2(jsimd_rgb_ycc_neon_consts);
#else
  /* GCC does not currently support the intrinsic vld1_<type>_x2(). */
  const uint16x4_t consts1 = vld1_u16(jsimd_rgb_ycc_neon_consts);
  const uint16x4_t consts2 = vld1_u16(jsimd_rgb_ycc_neon_consts + 4);
  const uint16x4x2_t consts = { { consts1, consts2 } };
#endif
  const uint32x4_t scaled_128_5 = vdupq_n_u32((128 << 16) + 32767);

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr0 = output_buf[0][output_row];
    outptr1 = output_buf[1][output_row];
    outptr2 = output_buf[2][output_row];
    output_row++;

    int cols_remaining = image_width;
    for (; cols_remaining > 0; cols_remaining -= 8) {

      /* To prevent buffer overread by the vector load instructions, the last
       * (image_width % 8) columns of data are first memcopied to a temporary
       * buffer large enough to accommodate the vector load.
       */
      if (cols_remaining < 8) {
        memcpy(tmp_buf, inptr, cols_remaining * RGB_PIXELSIZE);
        inptr = tmp_buf;
      }

#if RGB_PIXELSIZE == 4
      uint8x8x4_t input_pixels = vld4_u8(inptr);
#else
      uint8x8x3_t input_pixels = vld3_u8(inptr);
#endif
      uint16x8_t r = vmovl_u8(input_pixels.val[RGB_RED]);
      uint16x8_t g = vmovl_u8(input_pixels.val[RGB_GREEN]);
      uint16x8_t b = vmovl_u8(input_pixels.val[RGB_BLUE]);

      /* Compute Y = 0.29900 * R + 0.58700 * G + 0.11400 * B */
      uint32x4_t y_low = vmull_lane_u16(vget_low_u16(r), consts.val[0], 0);
      y_low = vmlal_lane_u16(y_low, vget_low_u16(g), consts.val[0], 1);
      y_low = vmlal_lane_u16(y_low, vget_low_u16(b), consts.val[0], 2);
      uint32x4_t y_high = vmull_lane_u16(vget_high_u16(r), consts.val[0], 0);
      y_high = vmlal_lane_u16(y_high, vget_high_u16(g), consts.val[0], 1);
      y_high = vmlal_lane_u16(y_high, vget_high_u16(b), consts.val[0], 2);

      /* Compute Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + 128 */
      uint32x4_t cb_low = scaled_128_5;
      cb_low = vmlsl_lane_u16(cb_low, vget_low_u16(r), consts.val[0], 3);
      cb_low = vmlsl_lane_u16(cb_low, vget_low_u16(g), consts.val[1], 0);
      cb_low = vmlal_lane_u16(cb_low, vget_low_u16(b), consts.val[1], 1);
      uint32x4_t cb_high = scaled_128_5;
      cb_high = vmlsl_lane_u16(cb_high, vget_high_u16(r), consts.val[0], 3);
      cb_high = vmlsl_lane_u16(cb_high, vget_high_u16(g), consts.val[1], 0);
      cb_high = vmlal_lane_u16(cb_high, vget_high_u16(b), consts.val[1], 1);

      /* Compute Cr = 0.50000 * R - 0.41869 * G - 0.08131 * B  + 128 */
      uint32x4_t cr_low = scaled_128_5;
      cr_low = vmlal_lane_u16(cr_low, vget_low_u16(r), consts.val[1], 1);
      cr_low = vmlsl_lane_u16(cr_low, vget_low_u16(g), consts.val[1], 2);
      cr_low = vmlsl_lane_u16(cr_low, vget_low_u16(b), consts.val[1], 3);
      uint32x4_t cr_high = scaled_128_5;
      cr_high = vmlal_lane_u16(cr_high, vget_high_u16(r), consts.val[1], 1);
      cr_high = vmlsl_lane_u16(cr_high, vget_high_u16(g), consts.val[1], 2);
      cr_high = vmlsl_lane_u16(cr_high, vget_high_u16(b), consts.val[1], 3);

      /* Descale Y values (rounding right shift) and narrow to 16-bit. */
      uint16x8_t y_u16 = vcombine_u16(vrshrn_n_u32(y_low, 16),
                                      vrshrn_n_u32(y_high, 16));
      /* Descale Cb values (right shift) and narrow to 16-bit. */
      uint16x8_t cb_u16 = vcombine_u16(vshrn_n_u32(cb_low, 16),
                                       vshrn_n_u32(cb_high, 16));
      /* Descale Cr values (right shift) and narrow to 16-bit. */
      uint16x8_t cr_u16 = vcombine_u16(vshrn_n_u32(cr_low, 16),
                                       vshrn_n_u32(cr_high, 16));
      /* Narrow Y, Cb, and Cr values to 8-bit and store to memory.  Buffer
       * overwrite is permitted up to the next multiple of ALIGN_SIZE bytes.
       */
      vst1_u8(outptr0, vmovn_u16(y_u16));
      vst1_u8(outptr1, vmovn_u16(cb_u16));
      vst1_u8(outptr2, vmovn_u16(cr_u16));

      /* Increment pointers. */
      inptr += (8 * RGB_PIXELSIZE);
      outptr0 += 8;
      outptr1 += 8;
      outptr2 += 8;
    }
  }
}
