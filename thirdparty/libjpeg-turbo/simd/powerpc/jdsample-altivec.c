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

/* CHROMA UPSAMPLING */

#include "jsimd_altivec.h"


void jsimd_h2v1_fancy_upsample_altivec(int max_v_samp_factor,
                                       JDIMENSION downsampled_width,
                                       JSAMPARRAY input_data,
                                       JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow, incol;

  __vector unsigned char this0, last0, p_last0, next0 = { 0 }, p_next0,
    out;
  __vector short this0e, this0o, this0l, this0h, last0l, last0h,
    next0l, next0h, outle, outhe, outlo, outho;

  /* Constants */
  __vector unsigned char pb_zero = { __16X(0) }, pb_three = { __16X(3) },
    last_index_col0 =
      {  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14 },
    last_index =
      { 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 },
    next_index =
      {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16 },
    next_index_lastcol =
      {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15 },
#ifdef __BIG_ENDIAN__
    merge_pack_index =
      {  1, 17,  3, 19,  5, 21,  7, 23,  9, 25, 11, 27, 13, 29, 15, 31 };
#else
    merge_pack_index =
      {  0, 16,  2, 18,  4, 20,  6, 22,  8, 24, 10, 26, 12, 28, 14, 30 };
#endif
  __vector short pw_one = { __8X(1) }, pw_two = { __8X(2) };

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];

    if (downsampled_width & 15)
      inptr[downsampled_width] = inptr[downsampled_width - 1];

    this0 = vec_ld(0, inptr);
    p_last0 = vec_perm(this0, this0, last_index_col0);
    last0 = this0;

    for (incol = downsampled_width; incol > 0;
         incol -= 16, inptr += 16, outptr += 32) {

      if (downsampled_width - incol > 0) {
        p_last0 = vec_perm(last0, this0, last_index);
        last0 = this0;
      }

      if (incol <= 16)
        p_next0 = vec_perm(this0, this0, next_index_lastcol);
      else {
        next0 = vec_ld(16, inptr);
        p_next0 = vec_perm(this0, next0, next_index);
      }

      this0e = (__vector short)vec_mule(this0, pb_three);
      this0o = (__vector short)vec_mulo(this0, pb_three);
      this0l = vec_mergeh(this0e, this0o);
      this0h = vec_mergel(this0e, this0o);

      last0l = (__vector short)VEC_UNPACKHU(p_last0);
      last0h = (__vector short)VEC_UNPACKLU(p_last0);
      last0l = vec_add(last0l, pw_one);

      next0l = (__vector short)VEC_UNPACKHU(p_next0);
      next0h = (__vector short)VEC_UNPACKLU(p_next0);
      next0l = vec_add(next0l, pw_two);

      outle = vec_add(this0l, last0l);
      outlo = vec_add(this0l, next0l);
      outle = vec_sr(outle, (__vector unsigned short)pw_two);
      outlo = vec_sr(outlo, (__vector unsigned short)pw_two);

      out = vec_perm((__vector unsigned char)outle,
                     (__vector unsigned char)outlo, merge_pack_index);
      vec_st(out, 0, outptr);

      if (incol > 8) {
        last0h = vec_add(last0h, pw_one);
        next0h = vec_add(next0h, pw_two);

        outhe = vec_add(this0h, last0h);
        outho = vec_add(this0h, next0h);
        outhe = vec_sr(outhe, (__vector unsigned short)pw_two);
        outho = vec_sr(outho, (__vector unsigned short)pw_two);

        out = vec_perm((__vector unsigned char)outhe,
                       (__vector unsigned char)outho, merge_pack_index);
        vec_st(out, 16, outptr);
      }

      this0 = next0;
    }
  }
}


void jsimd_h2v2_fancy_upsample_altivec(int max_v_samp_factor,
                                       JDIMENSION downsampled_width,
                                       JSAMPARRAY input_data,
                                       JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr_1, inptr0, inptr1, outptr0, outptr1;
  int inrow, outrow, incol;

  __vector unsigned char this_1, this0, this1, out;
  __vector short this_1l, this_1h, this0l, this0h, this1l, this1h,
    lastcolsum_1h, lastcolsum1h,
    p_lastcolsum_1l, p_lastcolsum_1h, p_lastcolsum1l, p_lastcolsum1h,
    thiscolsum_1l, thiscolsum_1h, thiscolsum1l, thiscolsum1h,
    nextcolsum_1l = { 0 }, nextcolsum_1h = { 0 },
    nextcolsum1l = { 0 }, nextcolsum1h = { 0 },
    p_nextcolsum_1l, p_nextcolsum_1h, p_nextcolsum1l, p_nextcolsum1h,
    tmpl, tmph, outle, outhe, outlo, outho;

  /* Constants */
  __vector unsigned char pb_zero = { __16X(0) },
    last_index_col0 =
      {  0,  1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13 },
    last_index =
      { 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 },
    next_index =
      {  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17 },
    next_index_lastcol =
      {  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 14, 15 },
#ifdef __BIG_ENDIAN__
    merge_pack_index =
      {  1, 17,  3, 19,  5, 21,  7, 23,  9, 25, 11, 27, 13, 29, 15, 31 };
#else
    merge_pack_index =
      {  0, 16,  2, 18,  4, 20,  6, 22,  8, 24, 10, 26, 12, 28, 14, 30 };
#endif
  __vector short pw_zero = { __8X(0) }, pw_three = { __8X(3) },
    pw_seven = { __8X(7) }, pw_eight = { __8X(8) };
  __vector unsigned short pw_four = { __8X(4) };

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {

    inptr_1 = input_data[inrow - 1];
    inptr0 = input_data[inrow];
    inptr1 = input_data[inrow + 1];
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    if (downsampled_width & 15) {
      inptr_1[downsampled_width] = inptr_1[downsampled_width - 1];
      inptr0[downsampled_width] = inptr0[downsampled_width - 1];
      inptr1[downsampled_width] = inptr1[downsampled_width - 1];
    }

    this0 = vec_ld(0, inptr0);
    this0l = (__vector short)VEC_UNPACKHU(this0);
    this0h = (__vector short)VEC_UNPACKLU(this0);
    this0l = vec_mladd(this0l, pw_three, pw_zero);
    this0h = vec_mladd(this0h, pw_three, pw_zero);

    this_1 = vec_ld(0, inptr_1);
    this_1l = (__vector short)VEC_UNPACKHU(this_1);
    this_1h = (__vector short)VEC_UNPACKLU(this_1);
    thiscolsum_1l = vec_add(this0l, this_1l);
    thiscolsum_1h = vec_add(this0h, this_1h);
    lastcolsum_1h = thiscolsum_1h;
    p_lastcolsum_1l = vec_perm(thiscolsum_1l, thiscolsum_1l, last_index_col0);
    p_lastcolsum_1h = vec_perm(thiscolsum_1l, thiscolsum_1h, last_index);

    this1 = vec_ld(0, inptr1);
    this1l = (__vector short)VEC_UNPACKHU(this1);
    this1h = (__vector short)VEC_UNPACKLU(this1);
    thiscolsum1l = vec_add(this0l, this1l);
    thiscolsum1h = vec_add(this0h, this1h);
    lastcolsum1h = thiscolsum1h;
    p_lastcolsum1l = vec_perm(thiscolsum1l, thiscolsum1l, last_index_col0);
    p_lastcolsum1h = vec_perm(thiscolsum1l, thiscolsum1h, last_index);

    for (incol = downsampled_width; incol > 0;
         incol -= 16, inptr_1 += 16, inptr0 += 16, inptr1 += 16,
         outptr0 += 32, outptr1 += 32) {

      if (downsampled_width - incol > 0) {
        p_lastcolsum_1l = vec_perm(lastcolsum_1h, thiscolsum_1l, last_index);
        p_lastcolsum_1h = vec_perm(thiscolsum_1l, thiscolsum_1h, last_index);
        p_lastcolsum1l = vec_perm(lastcolsum1h, thiscolsum1l, last_index);
        p_lastcolsum1h = vec_perm(thiscolsum1l, thiscolsum1h, last_index);
        lastcolsum_1h = thiscolsum_1h;  lastcolsum1h = thiscolsum1h;
      }

      if (incol <= 16) {
        p_nextcolsum_1l = vec_perm(thiscolsum_1l, thiscolsum_1h, next_index);
        p_nextcolsum_1h = vec_perm(thiscolsum_1h, thiscolsum_1h,
                                   next_index_lastcol);
        p_nextcolsum1l = vec_perm(thiscolsum1l, thiscolsum1h, next_index);
        p_nextcolsum1h = vec_perm(thiscolsum1h, thiscolsum1h,
                                  next_index_lastcol);
      } else {
        this0 = vec_ld(16, inptr0);
        this0l = (__vector short)VEC_UNPACKHU(this0);
        this0h = (__vector short)VEC_UNPACKLU(this0);
        this0l = vec_mladd(this0l, pw_three, pw_zero);
        this0h = vec_mladd(this0h, pw_three, pw_zero);

        this_1 = vec_ld(16, inptr_1);
        this_1l = (__vector short)VEC_UNPACKHU(this_1);
        this_1h = (__vector short)VEC_UNPACKLU(this_1);
        nextcolsum_1l = vec_add(this0l, this_1l);
        nextcolsum_1h = vec_add(this0h, this_1h);
        p_nextcolsum_1l = vec_perm(thiscolsum_1l, thiscolsum_1h, next_index);
        p_nextcolsum_1h = vec_perm(thiscolsum_1h, nextcolsum_1l, next_index);

        this1 = vec_ld(16, inptr1);
        this1l = (__vector short)VEC_UNPACKHU(this1);
        this1h = (__vector short)VEC_UNPACKLU(this1);
        nextcolsum1l = vec_add(this0l, this1l);
        nextcolsum1h = vec_add(this0h, this1h);
        p_nextcolsum1l = vec_perm(thiscolsum1l, thiscolsum1h, next_index);
        p_nextcolsum1h = vec_perm(thiscolsum1h, nextcolsum1l, next_index);
      }

      /* Process the upper row */

      tmpl = vec_mladd(thiscolsum_1l, pw_three, pw_zero);
      outle = vec_add(tmpl, p_lastcolsum_1l);
      outle = vec_add(outle, pw_eight);
      outle = vec_sr(outle, pw_four);

      outlo = vec_add(tmpl, p_nextcolsum_1l);
      outlo = vec_add(outlo, pw_seven);
      outlo = vec_sr(outlo, pw_four);

      out = vec_perm((__vector unsigned char)outle,
                     (__vector unsigned char)outlo, merge_pack_index);
      vec_st(out, 0, outptr0);

      if (incol > 8) {
        tmph = vec_mladd(thiscolsum_1h, pw_three, pw_zero);
        outhe = vec_add(tmph, p_lastcolsum_1h);
        outhe = vec_add(outhe, pw_eight);
        outhe = vec_sr(outhe, pw_four);

        outho = vec_add(tmph, p_nextcolsum_1h);
        outho = vec_add(outho, pw_seven);
        outho = vec_sr(outho, pw_four);

        out = vec_perm((__vector unsigned char)outhe,
                       (__vector unsigned char)outho, merge_pack_index);
        vec_st(out, 16, outptr0);
      }

      /* Process the lower row */

      tmpl = vec_mladd(thiscolsum1l, pw_three, pw_zero);
      outle = vec_add(tmpl, p_lastcolsum1l);
      outle = vec_add(outle, pw_eight);
      outle = vec_sr(outle, pw_four);

      outlo = vec_add(tmpl, p_nextcolsum1l);
      outlo = vec_add(outlo, pw_seven);
      outlo = vec_sr(outlo, pw_four);

      out = vec_perm((__vector unsigned char)outle,
                     (__vector unsigned char)outlo, merge_pack_index);
      vec_st(out, 0, outptr1);

      if (incol > 8) {
        tmph = vec_mladd(thiscolsum1h, pw_three, pw_zero);
        outhe = vec_add(tmph, p_lastcolsum1h);
        outhe = vec_add(outhe, pw_eight);
        outhe = vec_sr(outhe, pw_four);

        outho = vec_add(tmph, p_nextcolsum1h);
        outho = vec_add(outho, pw_seven);
        outho = vec_sr(outho, pw_four);

        out = vec_perm((__vector unsigned char)outhe,
                       (__vector unsigned char)outho, merge_pack_index);
        vec_st(out, 16, outptr1);
      }

      thiscolsum_1l = nextcolsum_1l;  thiscolsum_1h = nextcolsum_1h;
      thiscolsum1l = nextcolsum1l;  thiscolsum1h = nextcolsum1h;
    }
  }
}


/* These are rarely used (mainly just for decompressing YCCK images) */

void jsimd_h2v1_upsample_altivec(int max_v_samp_factor,
                                 JDIMENSION output_width,
                                 JSAMPARRAY input_data,
                                 JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr;
  int inrow, incol;

  __vector unsigned char in, inl, inh;

  for (inrow = 0; inrow < max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];

    for (incol = (output_width + 31) & (~31); incol > 0;
         incol -= 64, inptr += 32, outptr += 64) {

      in = vec_ld(0, inptr);
      inl = vec_mergeh(in, in);
      inh = vec_mergel(in, in);

      vec_st(inl, 0, outptr);
      vec_st(inh, 16, outptr);

      if (incol > 32) {
        in = vec_ld(16, inptr);
        inl = vec_mergeh(in, in);
        inh = vec_mergel(in, in);

        vec_st(inl, 32, outptr);
        vec_st(inh, 48, outptr);
      }
    }
  }
}


void jsimd_h2v2_upsample_altivec(int max_v_samp_factor,
                                 JDIMENSION output_width,
                                 JSAMPARRAY input_data,
                                 JSAMPARRAY *output_data_ptr)
{
  JSAMPARRAY output_data = *output_data_ptr;
  JSAMPROW inptr, outptr0, outptr1;
  int inrow, outrow, incol;

  __vector unsigned char in, inl, inh;

  for (inrow = 0, outrow = 0; outrow < max_v_samp_factor; inrow++) {

    inptr = input_data[inrow];
    outptr0 = output_data[outrow++];
    outptr1 = output_data[outrow++];

    for (incol = (output_width + 31) & (~31); incol > 0;
         incol -= 64, inptr += 32, outptr0 += 64, outptr1 += 64) {

      in = vec_ld(0, inptr);
      inl = vec_mergeh(in, in);
      inh = vec_mergel(in, in);

      vec_st(inl, 0, outptr0);
      vec_st(inl, 0, outptr1);

      vec_st(inh, 16, outptr0);
      vec_st(inh, 16, outptr1);

      if (incol > 32) {
        in = vec_ld(16, inptr);
        inl = vec_mergeh(in, in);
        inh = vec_mergel(in, in);

        vec_st(inl, 32, outptr0);
        vec_st(inl, 32, outptr1);

        vec_st(inh, 48, outptr0);
        vec_st(inh, 48, outptr1);
      }
    }
  }
}
