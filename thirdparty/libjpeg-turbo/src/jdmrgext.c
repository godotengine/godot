/*
 * jdmrgext.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2011, 2015, 2020, 2022-2023, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains code for merged upsampling/color conversion.
 */


/* This file is included by jdmerge.c */


/*
 * Upsample and color convert for the case of 2:1 horizontal and 1:1 vertical.
 */

INLINE
LOCAL(void)
h2v1_merged_upsample_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
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
    outptr[RGB_RED] =   range_limit[y + cred];
    outptr[RGB_GREEN] = range_limit[y + cgreen];
    outptr[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr[RGB_ALPHA] = _MAXJSAMPLE;
#endif
    outptr += RGB_PIXELSIZE;
    y  = *inptr0++;
    outptr[RGB_RED] =   range_limit[y + cred];
    outptr[RGB_GREEN] = range_limit[y + cgreen];
    outptr[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr[RGB_ALPHA] = _MAXJSAMPLE;
#endif
    outptr += RGB_PIXELSIZE;
  }
  /* If image width is odd, do the last output column separately */
  if (cinfo->output_width & 1) {
    cb = *inptr1;
    cr = *inptr2;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];
    y  = *inptr0;
    outptr[RGB_RED] =   range_limit[y + cred];
    outptr[RGB_GREEN] = range_limit[y + cgreen];
    outptr[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr[RGB_ALPHA] = _MAXJSAMPLE;
#endif
  }
}


/*
 * Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
 */

INLINE
LOCAL(void)
h2v2_merged_upsample_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
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
    outptr0[RGB_RED] =   range_limit[y + cred];
    outptr0[RGB_GREEN] = range_limit[y + cgreen];
    outptr0[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr0[RGB_ALPHA] = _MAXJSAMPLE;
#endif
    outptr0 += RGB_PIXELSIZE;
    y  = *inptr00++;
    outptr0[RGB_RED] =   range_limit[y + cred];
    outptr0[RGB_GREEN] = range_limit[y + cgreen];
    outptr0[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr0[RGB_ALPHA] = _MAXJSAMPLE;
#endif
    outptr0 += RGB_PIXELSIZE;
    y  = *inptr01++;
    outptr1[RGB_RED] =   range_limit[y + cred];
    outptr1[RGB_GREEN] = range_limit[y + cgreen];
    outptr1[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr1[RGB_ALPHA] = _MAXJSAMPLE;
#endif
    outptr1 += RGB_PIXELSIZE;
    y  = *inptr01++;
    outptr1[RGB_RED] =   range_limit[y + cred];
    outptr1[RGB_GREEN] = range_limit[y + cgreen];
    outptr1[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr1[RGB_ALPHA] = _MAXJSAMPLE;
#endif
    outptr1 += RGB_PIXELSIZE;
  }
  /* If image width is odd, do the last output column separately */
  if (cinfo->output_width & 1) {
    cb = *inptr1;
    cr = *inptr2;
    cred = Crrtab[cr];
    cgreen = (int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr], SCALEBITS);
    cblue = Cbbtab[cb];
    y  = *inptr00;
    outptr0[RGB_RED] =   range_limit[y + cred];
    outptr0[RGB_GREEN] = range_limit[y + cgreen];
    outptr0[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr0[RGB_ALPHA] = _MAXJSAMPLE;
#endif
    y  = *inptr01;
    outptr1[RGB_RED] =   range_limit[y + cred];
    outptr1[RGB_GREEN] = range_limit[y + cgreen];
    outptr1[RGB_BLUE] =  range_limit[y + cblue];
#ifdef RGB_ALPHA
    outptr1[RGB_ALPHA] = _MAXJSAMPLE;
#endif
  }
}
