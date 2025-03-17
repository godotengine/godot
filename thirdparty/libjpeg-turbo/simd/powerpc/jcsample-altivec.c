/*
 * AltiVec optimizations for libjpeg-turbo
 *
 * Copyright (C) 2015, 2024, D. R. Commander.  All Rights Reserved.
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

/* CHROMA DOWNSAMPLING */

#include "jsimd_altivec.h"
#include "jcsample.h"


void jsimd_h2v1_downsample_altivec(JDIMENSION image_width,
                                   int max_v_samp_factor,
                                   JDIMENSION v_samp_factor,
                                   JDIMENSION width_in_blocks,
                                   JSAMPARRAY input_data,
                                   JSAMPARRAY output_data)
{
  int outrow, outcol;
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  JSAMPROW inptr, outptr;

  __vector unsigned char this0, next0, out;
  __vector unsigned short this0e, this0o, next0e, next0o, outl, outh;

  /* Constants */
  __vector unsigned short pw_bias = { __4X2(0, 1) },
    pw_one = { __8X(1) };
  __vector unsigned char even_odd_index =
    {  0,  2,  4,  6,  8, 10, 12, 14,  1,  3,  5,  7,  9, 11, 13, 15 },
    pb_zero = { __16X(0) };

  expand_right_edge(input_data, max_v_samp_factor, image_width,
                    output_cols * 2);

  for (outrow = 0; outrow < (int)v_samp_factor; outrow++) {
    outptr = output_data[outrow];
    inptr = input_data[outrow];

    for (outcol = output_cols; outcol > 0;
         outcol -= 16, inptr += 32, outptr += 16) {

      this0 = vec_ld(0, inptr);
      this0 = vec_perm(this0, this0, even_odd_index);
      this0e = (__vector unsigned short)VEC_UNPACKHU(this0);
      this0o = (__vector unsigned short)VEC_UNPACKLU(this0);
      outl = vec_add(this0e, this0o);
      outl = vec_add(outl, pw_bias);
      outl = vec_sr(outl, pw_one);

      if (outcol > 8) {
        next0 = vec_ld(16, inptr);
        next0 = vec_perm(next0, next0, even_odd_index);
        next0e = (__vector unsigned short)VEC_UNPACKHU(next0);
        next0o = (__vector unsigned short)VEC_UNPACKLU(next0);
        outh = vec_add(next0e, next0o);
        outh = vec_add(outh, pw_bias);
        outh = vec_sr(outh, pw_one);
      } else
        outh = vec_splat_u16(0);

      out = vec_pack(outl, outh);
      vec_st(out, 0, outptr);
    }
  }
}


void
jsimd_h2v2_downsample_altivec(JDIMENSION image_width, int max_v_samp_factor,
                              JDIMENSION v_samp_factor,
                              JDIMENSION width_in_blocks,
                              JSAMPARRAY input_data, JSAMPARRAY output_data)
{
  int inrow, outrow, outcol;
  JDIMENSION output_cols = width_in_blocks * DCTSIZE;
  JSAMPROW inptr0, inptr1, outptr;

  __vector unsigned char this0, next0, this1, next1, out;
  __vector unsigned short this0e, this0o, next0e, next0o, this1e, this1o,
    next1e, next1o, out0l, out0h, out1l, out1h, outl, outh;

  /* Constants */
  __vector unsigned short pw_bias = { __4X2(1, 2) },
    pw_two = { __8X(2) };
  __vector unsigned char even_odd_index =
    {  0,  2,  4,  6,  8, 10, 12, 14,  1,  3,  5,  7,  9, 11, 13, 15 },
    pb_zero = { __16X(0) };

  expand_right_edge(input_data, max_v_samp_factor, image_width,
                    output_cols * 2);

  for (inrow = 0, outrow = 0; outrow < (int)v_samp_factor;
       inrow += 2, outrow++) {

    inptr0 = input_data[inrow];
    inptr1 = input_data[inrow + 1];
    outptr = output_data[outrow];

    for (outcol = output_cols; outcol > 0;
         outcol -= 16, inptr0 += 32, inptr1 += 32, outptr += 16) {

      this0 = vec_ld(0, inptr0);
      this0 = vec_perm(this0, this0, even_odd_index);
      this0e = (__vector unsigned short)VEC_UNPACKHU(this0);
      this0o = (__vector unsigned short)VEC_UNPACKLU(this0);
      out0l = vec_add(this0e, this0o);

      this1 = vec_ld(0, inptr1);
      this1 = vec_perm(this1, this1, even_odd_index);
      this1e = (__vector unsigned short)VEC_UNPACKHU(this1);
      this1o = (__vector unsigned short)VEC_UNPACKLU(this1);
      out1l = vec_add(this1e, this1o);

      outl = vec_add(out0l, out1l);
      outl = vec_add(outl, pw_bias);
      outl = vec_sr(outl, pw_two);

      if (outcol > 8) {
        next0 = vec_ld(16, inptr0);
        next0 = vec_perm(next0, next0, even_odd_index);
        next0e = (__vector unsigned short)VEC_UNPACKHU(next0);
        next0o = (__vector unsigned short)VEC_UNPACKLU(next0);
        out0h = vec_add(next0e, next0o);

        next1 = vec_ld(16, inptr1);
        next1 = vec_perm(next1, next1, even_odd_index);
        next1e = (__vector unsigned short)VEC_UNPACKHU(next1);
        next1o = (__vector unsigned short)VEC_UNPACKLU(next1);
        out1h = vec_add(next1e, next1o);

        outh = vec_add(out0h, out1h);
        outh = vec_add(outh, pw_bias);
        outh = vec_sr(outh, pw_two);
      } else
        outh = vec_splat_u16(0);

      out = vec_pack(outl, outh);
      vec_st(out, 0, outptr);
    }
  }
}
