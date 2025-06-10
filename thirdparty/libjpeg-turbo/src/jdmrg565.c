/*
 * jdmrg565.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2013, Linaro Limited.
 * Copyright (C) 2014-2015, 2018, 2020, 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains code for merged upsampling/color conversion.
 */


INLINE
LOCAL(void)
h2v1_merged_upsample_565_internal(j_decompress_ptr cinfo,
                                  _JSAMPIMAGE input_buf,
                                  JDIMENSION in_row_group_ctr,
                                  _JSAMPARRAY output_buf)
{
  my_merged_upsample_ptr upsample = (my_merged_upsample_ptr)cinfo->upsample;
  register int y, cred, cgreen, cblue;
  int cb, cr;
  register _JSAMPROW outptr;
  _JSAMPROW inptr0, inptr1, inptr2;
  JDIMENSION col;
  /* copy these pointers into registers if possible */
  register _JSAMPLE *range_limit = (_JSAMPLE *)cinfo->sample_range_limit;
  int *Crrtab = upsample->Cr_r_tab;
  int *Cbbtab = upsample->Cb_b_tab;
  JLONG *Crgtab = upsample->Cr_g_tab;
  JLONG *Cbgtab = upsample->Cb_g_tab;
  unsigned int r, g, b;
  JLONG rgb;
  SHIFT_TEMPS

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  /* Loop for each pair of output pixels */
  for (col = cinfo->output_width >> 1; col > 0; col--) {
    /* Do the chroma part of the calculation */
    cb = *inptr1++;
    cr = *inptr2++;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];

    /* Fetch 2 Y values and emit 2 pixels */
    y  = *inptr0++;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_SHORT_565(r, g, b);

    y  = *inptr0++;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

    WRITE_TWO_PIXELS(outptr, rgb);
    outptr += 4;
  }

  /* If image width is odd, do the last output column separately */
  if (cinfo->output_width & 1) {
    cb = *inptr1;
    cr = *inptr2;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];
    y  = *inptr0;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_SHORT_565(r, g, b);
    *(INT16 *)outptr = (INT16)rgb;
  }
}


INLINE
LOCAL(void)
h2v1_merged_upsample_565D_internal(j_decompress_ptr cinfo,
                                   _JSAMPIMAGE input_buf,
                                   JDIMENSION in_row_group_ctr,
                                   _JSAMPARRAY output_buf)
{
  my_merged_upsample_ptr upsample = (my_merged_upsample_ptr)cinfo->upsample;
  register int y, cred, cgreen, cblue;
  int cb, cr;
  register _JSAMPROW outptr;
  _JSAMPROW inptr0, inptr1, inptr2;
  JDIMENSION col;
  /* copy these pointers into registers if possible */
  register _JSAMPLE *range_limit = (_JSAMPLE *)cinfo->sample_range_limit;
  int *Crrtab = upsample->Cr_r_tab;
  int *Cbbtab = upsample->Cb_b_tab;
  JLONG *Crgtab = upsample->Cr_g_tab;
  JLONG *Cbgtab = upsample->Cb_g_tab;
  JLONG d0 = dither_matrix[cinfo->output_scanline & DITHER_MASK];
  unsigned int r, g, b;
  JLONG rgb;
  SHIFT_TEMPS

  inptr0 = input_buf[0][in_row_group_ctr];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr = output_buf[0];

  /* Loop for each pair of output pixels */
  for (col = cinfo->output_width >> 1; col > 0; col--) {
    /* Do the chroma part of the calculation */
    cb = *inptr1++;
    cr = *inptr2++;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];

    /* Fetch 2 Y values and emit 2 pixels */
    y  = *inptr0++;
    r = range_limit[DITHER_565_R(y + cred, d0)];
    g = range_limit[DITHER_565_G(y + cgreen, d0)];
    b = range_limit[DITHER_565_B(y + cblue, d0)];
    d0 = DITHER_ROTATE(d0);
    rgb = PACK_SHORT_565(r, g, b);

    y  = *inptr0++;
    r = range_limit[DITHER_565_R(y + cred, d0)];
    g = range_limit[DITHER_565_G(y + cgreen, d0)];
    b = range_limit[DITHER_565_B(y + cblue, d0)];
    d0 = DITHER_ROTATE(d0);
    rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

    WRITE_TWO_PIXELS(outptr, rgb);
    outptr += 4;
  }

  /* If image width is odd, do the last output column separately */
  if (cinfo->output_width & 1) {
    cb = *inptr1;
    cr = *inptr2;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];
    y  = *inptr0;
    r = range_limit[DITHER_565_R(y + cred, d0)];
    g = range_limit[DITHER_565_G(y + cgreen, d0)];
    b = range_limit[DITHER_565_B(y + cblue, d0)];
    rgb = PACK_SHORT_565(r, g, b);
    *(INT16 *)outptr = (INT16)rgb;
  }
}


INLINE
LOCAL(void)
h2v2_merged_upsample_565_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                                  JDIMENSION in_row_group_ctr,
                                  _JSAMPARRAY output_buf)
{
  my_merged_upsample_ptr upsample = (my_merged_upsample_ptr)cinfo->upsample;
  register int y, cred, cgreen, cblue;
  int cb, cr;
  register _JSAMPROW outptr0, outptr1;
  _JSAMPROW inptr00, inptr01, inptr1, inptr2;
  JDIMENSION col;
  /* copy these pointers into registers if possible */
  register _JSAMPLE *range_limit = (_JSAMPLE *)cinfo->sample_range_limit;
  int *Crrtab = upsample->Cr_r_tab;
  int *Cbbtab = upsample->Cb_b_tab;
  JLONG *Crgtab = upsample->Cr_g_tab;
  JLONG *Cbgtab = upsample->Cb_g_tab;
  unsigned int r, g, b;
  JLONG rgb;
  SHIFT_TEMPS

  inptr00 = input_buf[0][in_row_group_ctr * 2];
  inptr01 = input_buf[0][in_row_group_ctr * 2 + 1];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr0 = output_buf[0];
  outptr1 = output_buf[1];

  /* Loop for each group of output pixels */
  for (col = cinfo->output_width >> 1; col > 0; col--) {
    /* Do the chroma part of the calculation */
    cb = *inptr1++;
    cr = *inptr2++;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];

    /* Fetch 4 Y values and emit 4 pixels */
    y  = *inptr00++;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_SHORT_565(r, g, b);

    y  = *inptr00++;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

    WRITE_TWO_PIXELS(outptr0, rgb);
    outptr0 += 4;

    y  = *inptr01++;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_SHORT_565(r, g, b);

    y  = *inptr01++;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

    WRITE_TWO_PIXELS(outptr1, rgb);
    outptr1 += 4;
  }

  /* If image width is odd, do the last output column separately */
  if (cinfo->output_width & 1) {
    cb = *inptr1;
    cr = *inptr2;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];

    y  = *inptr00;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_SHORT_565(r, g, b);
    *(INT16 *)outptr0 = (INT16)rgb;

    y  = *inptr01;
    r = range_limit[y + cred];
    g = range_limit[y + cgreen];
    b = range_limit[y + cblue];
    rgb = PACK_SHORT_565(r, g, b);
    *(INT16 *)outptr1 = (INT16)rgb;
  }
}


INLINE
LOCAL(void)
h2v2_merged_upsample_565D_internal(j_decompress_ptr cinfo,
                                   _JSAMPIMAGE input_buf,
                                   JDIMENSION in_row_group_ctr,
                                   _JSAMPARRAY output_buf)
{
  my_merged_upsample_ptr upsample = (my_merged_upsample_ptr)cinfo->upsample;
  register int y, cred, cgreen, cblue;
  int cb, cr;
  register _JSAMPROW outptr0, outptr1;
  _JSAMPROW inptr00, inptr01, inptr1, inptr2;
  JDIMENSION col;
  /* copy these pointers into registers if possible */
  register _JSAMPLE *range_limit = (_JSAMPLE *)cinfo->sample_range_limit;
  int *Crrtab = upsample->Cr_r_tab;
  int *Cbbtab = upsample->Cb_b_tab;
  JLONG *Crgtab = upsample->Cr_g_tab;
  JLONG *Cbgtab = upsample->Cb_g_tab;
  JLONG d0 = dither_matrix[cinfo->output_scanline & DITHER_MASK];
  JLONG d1 = dither_matrix[(cinfo->output_scanline + 1) & DITHER_MASK];
  unsigned int r, g, b;
  JLONG rgb;
  SHIFT_TEMPS

  inptr00 = input_buf[0][in_row_group_ctr * 2];
  inptr01 = input_buf[0][in_row_group_ctr * 2 + 1];
  inptr1 = input_buf[1][in_row_group_ctr];
  inptr2 = input_buf[2][in_row_group_ctr];
  outptr0 = output_buf[0];
  outptr1 = output_buf[1];

  /* Loop for each group of output pixels */
  for (col = cinfo->output_width >> 1; col > 0; col--) {
    /* Do the chroma part of the calculation */
    cb = *inptr1++;
    cr = *inptr2++;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];

    /* Fetch 4 Y values and emit 4 pixels */
    y  = *inptr00++;
    r = range_limit[DITHER_565_R(y + cred, d0)];
    g = range_limit[DITHER_565_G(y + cgreen, d0)];
    b = range_limit[DITHER_565_B(y + cblue, d0)];
    d0 = DITHER_ROTATE(d0);
    rgb = PACK_SHORT_565(r, g, b);

    y  = *inptr00++;
    r = range_limit[DITHER_565_R(y + cred, d0)];
    g = range_limit[DITHER_565_G(y + cgreen, d0)];
    b = range_limit[DITHER_565_B(y + cblue, d0)];
    d0 = DITHER_ROTATE(d0);
    rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

    WRITE_TWO_PIXELS(outptr0, rgb);
    outptr0 += 4;

    y  = *inptr01++;
    r = range_limit[DITHER_565_R(y + cred, d1)];
    g = range_limit[DITHER_565_G(y + cgreen, d1)];
    b = range_limit[DITHER_565_B(y + cblue, d1)];
    d1 = DITHER_ROTATE(d1);
    rgb = PACK_SHORT_565(r, g, b);

    y  = *inptr01++;
    r = range_limit[DITHER_565_R(y + cred, d1)];
    g = range_limit[DITHER_565_G(y + cgreen, d1)];
    b = range_limit[DITHER_565_B(y + cblue, d1)];
    d1 = DITHER_ROTATE(d1);
    rgb = PACK_TWO_PIXELS(rgb, PACK_SHORT_565(r, g, b));

    WRITE_TWO_PIXELS(outptr1, rgb);
    outptr1 += 4;
  }

  /* If image width is odd, do the last output column separately */
  if (cinfo->output_width & 1) {
    cb = *inptr1;
    cr = *inptr2;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];

    y  = *inptr00;
    r = range_limit[DITHER_565_R(y + cred, d0)];
    g = range_limit[DITHER_565_G(y + cgreen, d0)];
    b = range_limit[DITHER_565_B(y + cblue, d0)];
    rgb = PACK_SHORT_565(r, g, b);
    *(INT16 *)outptr0 = (INT16)rgb;

    y  = *inptr01;
    r = range_limit[DITHER_565_R(y + cred, d1)];
    g = range_limit[DITHER_565_G(y + cgreen, d1)];
    b = range_limit[DITHER_565_B(y + cblue, d1)];
    rgb = PACK_SHORT_565(r, g, b);
    *(INT16 *)outptr1 = (INT16)rgb;
  }
}
