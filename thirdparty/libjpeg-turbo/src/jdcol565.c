/*
 * jdcol565.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modifications:
 * Copyright (C) 2013, Linaro Limited.
 * Copyright (C) 2014-2015, 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains output colorspace conversion routines.
 */

/* This file is included by jdcolor.c */


INLINE
LOCAL(void)
ycc_rgb565_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                            JDIMENSION input_row, _JSAMPARRAY output_buf,
                            int num_rows)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  register int y, cb, cr;
  register _JSAMPROW outptr;
  register _JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;
  /* copy these pointers into registers if possible */
  register _JSAMPLE *range_limit = (_JSAMPLE *)cinfo->sample_range_limit;
  register int *Crrtab = cconvert->Cr_r_tab;
  register int *Cbbtab = cconvert->Cb_b_tab;
  register JLONG *Crgtab = cconvert->Cr_g_tab;
  register JLONG *Cbgtab = cconvert->Cb_g_tab;
  SHIFT_TEMPS

  while (--num_rows >= 0) {
    JLONG rgb;
    unsigned int r, g, b;
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;

    if (PACK_NEED_ALIGNMENT(outptr)) {
      y  = *inptr0++;
      cb = *inptr1++;
      cr = *inptr2++;
      r = range_limit[y + Crrtab[cr]];
      g = range_limit[y + ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                            SCALEBITS))];
      b = range_limit[y + Cbbtab[cb]];
      rgb = PACK_SHORT_565(r, g, b);
      *(INT16 *)outptr = (INT16)rgb;
      outptr += 2;
      num_cols--;
    }
    for (col = 0; col < (num_cols >> 1); col++) {
      y  = *inptr0++;
      cb = *inptr1++;
      cr = *inptr2++;
      r = range_limit[y + Crrtab[cr]];
      g = range_limit[y + ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                            SCALEBITS))];
      b = range_limit[y + Cbbtab[cb]];
      rgb = PACK_SHORT_565(r, g, b);

      y  = *inptr0++;
      cb = *inptr1++;
      cr = *inptr2++;
      r = range_limit[y + Crrtab[cr]];
      g = range_limit[y + ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                            SCALEBITS))];
      b = range_limit[y + Cbbtab[cb]];
      rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

      WRITE_TWO_ALIGNED_PIXELS(outptr, rgb);
      outptr += 4;
    }
    if (num_cols & 1) {
      y  = *inptr0;
      cb = *inptr1;
      cr = *inptr2;
      r = range_limit[y + Crrtab[cr]];
      g = range_limit[y + ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                            SCALEBITS))];
      b = range_limit[y + Cbbtab[cb]];
      rgb = PACK_SHORT_565(r, g, b);
      *(INT16 *)outptr = (INT16)rgb;
    }
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


INLINE
LOCAL(void)
ycc_rgb565D_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                             JDIMENSION input_row, _JSAMPARRAY output_buf,
                             int num_rows)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  register int y, cb, cr;
  register _JSAMPROW outptr;
  register _JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;
  /* copy these pointers into registers if possible */
  register _JSAMPLE *range_limit = (_JSAMPLE *)cinfo->sample_range_limit;
  register int *Crrtab = cconvert->Cr_r_tab;
  register int *Cbbtab = cconvert->Cb_b_tab;
  register JLONG *Crgtab = cconvert->Cr_g_tab;
  register JLONG *Cbgtab = cconvert->Cb_g_tab;
  JLONG d0 = dither_matrix[cinfo->output_scanline & DITHER_MASK];
  SHIFT_TEMPS

  while (--num_rows >= 0) {
    JLONG rgb;
    unsigned int r, g, b;

    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    if (PACK_NEED_ALIGNMENT(outptr)) {
      y  = *inptr0++;
      cb = *inptr1++;
      cr = *inptr2++;
      r = range_limit[DITHER_565_R(y + Crrtab[cr], d0)];
      g = range_limit[DITHER_565_G(y +
                                   ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                                     SCALEBITS)), d0)];
      b = range_limit[DITHER_565_B(y + Cbbtab[cb], d0)];
      rgb = PACK_SHORT_565(r, g, b);
      *(INT16 *)outptr = (INT16)rgb;
      outptr += 2;
      num_cols--;
    }
    for (col = 0; col < (num_cols >> 1); col++) {
      y  = *inptr0++;
      cb = *inptr1++;
      cr = *inptr2++;
      r = range_limit[DITHER_565_R(y + Crrtab[cr], d0)];
      g = range_limit[DITHER_565_G(y +
                                   ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                                     SCALEBITS)), d0)];
      b = range_limit[DITHER_565_B(y + Cbbtab[cb], d0)];
      d0 = DITHER_ROTATE(d0);
      rgb = PACK_SHORT_565(r, g, b);

      y  = *inptr0++;
      cb = *inptr1++;
      cr = *inptr2++;
      r = range_limit[DITHER_565_R(y + Crrtab[cr], d0)];
      g = range_limit[DITHER_565_G(y +
                                   ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                                     SCALEBITS)), d0)];
      b = range_limit[DITHER_565_B(y + Cbbtab[cb], d0)];
      d0 = DITHER_ROTATE(d0);
      rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

      WRITE_TWO_ALIGNED_PIXELS(outptr, rgb);
      outptr += 4;
    }
    if (num_cols & 1) {
      y  = *inptr0;
      cb = *inptr1;
      cr = *inptr2;
      r = range_limit[DITHER_565_R(y + Crrtab[cr], d0)];
      g = range_limit[DITHER_565_G(y +
                                   ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                                     SCALEBITS)), d0)];
      b = range_limit[DITHER_565_B(y + Cbbtab[cb], d0)];
      rgb = PACK_SHORT_565(r, g, b);
      *(INT16 *)outptr = (INT16)rgb;
    }
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


INLINE
LOCAL(void)
rgb_rgb565_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                            JDIMENSION input_row, _JSAMPARRAY output_buf,
                            int num_rows)
{
  register _JSAMPROW outptr;
  register _JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;
  SHIFT_TEMPS

  while (--num_rows >= 0) {
    JLONG rgb;
    unsigned int r, g, b;

    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    if (PACK_NEED_ALIGNMENT(outptr)) {
      r = *inptr0++;
      g = *inptr1++;
      b = *inptr2++;
      rgb = PACK_SHORT_565(r, g, b);
      *(INT16 *)outptr = (INT16)rgb;
      outptr += 2;
      num_cols--;
    }
    for (col = 0; col < (num_cols >> 1); col++) {
      r = *inptr0++;
      g = *inptr1++;
      b = *inptr2++;
      rgb = PACK_SHORT_565(r, g, b);

      r = *inptr0++;
      g = *inptr1++;
      b = *inptr2++;
      rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

      WRITE_TWO_ALIGNED_PIXELS(outptr, rgb);
      outptr += 4;
    }
    if (num_cols & 1) {
      r = *inptr0;
      g = *inptr1;
      b = *inptr2;
      rgb = PACK_SHORT_565(r, g, b);
      *(INT16 *)outptr = (INT16)rgb;
    }
  }
}


INLINE
LOCAL(void)
rgb_rgb565D_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                             JDIMENSION input_row, _JSAMPARRAY output_buf,
                             int num_rows)
{
  register _JSAMPROW outptr;
  register _JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  register _JSAMPLE *range_limit = (_JSAMPLE *)cinfo->sample_range_limit;
  JDIMENSION num_cols = cinfo->output_width;
  JLONG d0 = dither_matrix[cinfo->output_scanline & DITHER_MASK];
  SHIFT_TEMPS

  while (--num_rows >= 0) {
    JLONG rgb;
    unsigned int r, g, b;

    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    if (PACK_NEED_ALIGNMENT(outptr)) {
      r = range_limit[DITHER_565_R(*inptr0++, d0)];
      g = range_limit[DITHER_565_G(*inptr1++, d0)];
      b = range_limit[DITHER_565_B(*inptr2++, d0)];
      rgb = PACK_SHORT_565(r, g, b);
      *(INT16 *)outptr = (INT16)rgb;
      outptr += 2;
      num_cols--;
    }
    for (col = 0; col < (num_cols >> 1); col++) {
      r = range_limit[DITHER_565_R(*inptr0++, d0)];
      g = range_limit[DITHER_565_G(*inptr1++, d0)];
      b = range_limit[DITHER_565_B(*inptr2++, d0)];
      d0 = DITHER_ROTATE(d0);
      rgb = PACK_SHORT_565(r, g, b);

      r = range_limit[DITHER_565_R(*inptr0++, d0)];
      g = range_limit[DITHER_565_G(*inptr1++, d0)];
      b = range_limit[DITHER_565_B(*inptr2++, d0)];
      d0 = DITHER_ROTATE(d0);
      rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

      WRITE_TWO_ALIGNED_PIXELS(outptr, rgb);
      outptr += 4;
    }
    if (num_cols & 1) {
      r = range_limit[DITHER_565_R(*inptr0, d0)];
      g = range_limit[DITHER_565_G(*inptr1, d0)];
      b = range_limit[DITHER_565_B(*inptr2, d0)];
      rgb = PACK_SHORT_565(r, g, b);
      *(INT16 *)outptr = (INT16)rgb;
    }
  }
}


INLINE
LOCAL(void)
gray_rgb565_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                             JDIMENSION input_row, _JSAMPARRAY output_buf,
                             int num_rows)
{
  register _JSAMPROW inptr, outptr;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    JLONG rgb;
    unsigned int g;

    inptr = input_buf[0][input_row++];
    outptr = *output_buf++;
    if (PACK_NEED_ALIGNMENT(outptr)) {
      g = *inptr++;
      rgb = PACK_SHORT_565(g, g, g);
      *(INT16 *)outptr = (INT16)rgb;
      outptr += 2;
      num_cols--;
    }
    for (col = 0; col < (num_cols >> 1); col++) {
      g = *inptr++;
      rgb = PACK_SHORT_565(g, g, g);
      g = *inptr++;
      rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(g, g, g));
      WRITE_TWO_ALIGNED_PIXELS(outptr, rgb);
      outptr += 4;
    }
    if (num_cols & 1) {
      g = *inptr;
      rgb = PACK_SHORT_565(g, g, g);
      *(INT16 *)outptr = (INT16)rgb;
    }
  }
}


INLINE
LOCAL(void)
gray_rgb565D_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                              JDIMENSION input_row, _JSAMPARRAY output_buf,
                              int num_rows)
{
  register _JSAMPROW inptr, outptr;
  register JDIMENSION col;
  register _JSAMPLE *range_limit = (_JSAMPLE *)cinfo->sample_range_limit;
  JDIMENSION num_cols = cinfo->output_width;
  JLONG d0 = dither_matrix[cinfo->output_scanline & DITHER_MASK];

  while (--num_rows >= 0) {
    JLONG rgb;
    unsigned int g;

    inptr = input_buf[0][input_row++];
    outptr = *output_buf++;
    if (PACK_NEED_ALIGNMENT(outptr)) {
      g = *inptr++;
      g = range_limit[DITHER_565_R(g, d0)];
      rgb = PACK_SHORT_565(g, g, g);
      *(INT16 *)outptr = (INT16)rgb;
      outptr += 2;
      num_cols--;
    }
    for (col = 0; col < (num_cols >> 1); col++) {
      g = *inptr++;
      g = range_limit[DITHER_565_R(g, d0)];
      rgb = PACK_SHORT_565(g, g, g);
      d0 = DITHER_ROTATE(d0);

      g = *inptr++;
      g = range_limit[DITHER_565_R(g, d0)];
      rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(g, g, g));
      d0 = DITHER_ROTATE(d0);

      WRITE_TWO_ALIGNED_PIXELS(outptr, rgb);
      outptr += 4;
    }
    if (num_cols & 1) {
      g = *inptr;
      g = range_limit[DITHER_565_R(g, d0)];
      rgb = PACK_SHORT_565(g, g, g);
      *(INT16 *)outptr = (INT16)rgb;
    }
  }
}
