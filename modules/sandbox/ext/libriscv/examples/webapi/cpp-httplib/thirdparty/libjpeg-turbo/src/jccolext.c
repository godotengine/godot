/*
 * jccolext.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2009-2012, 2015, 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains input colorspace conversion routines.
 */


/* This file is included by jccolor.c */


/*
 * Convert some rows of samples to the JPEG colorspace.
 *
 * Note that we change from the application's interleaved-pixel format
 * to our internal noninterleaved, one-plane-per-component format.
 * The input buffer is therefore three times as wide as the output buffer.
 *
 * A starting row offset is provided only for the output buffer.  The caller
 * can easily adjust the passed input_buf value to accommodate any row
 * offset required on that side.
 */

INLINE
LOCAL(void)
rgb_ycc_convert_internal(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                         _JSAMPIMAGE output_buf, JDIMENSION output_row,
                         int num_rows)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  register int r, g, b;
  register JLONG *ctab = cconvert->rgb_ycc_tab;
  register _JSAMPROW inptr;
  register _JSAMPROW outptr0, outptr1, outptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->image_width;

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr0 = output_buf[0][output_row];
    outptr1 = output_buf[1][output_row];
    outptr2 = output_buf[2][output_row];
    output_row++;
    for (col = 0; col < num_cols; col++) {
      r = RANGE_LIMIT(inptr[RGB_RED]);
      g = RANGE_LIMIT(inptr[RGB_GREEN]);
      b = RANGE_LIMIT(inptr[RGB_BLUE]);
      inptr += RGB_PIXELSIZE;
      /* If the inputs are 0.._MAXJSAMPLE, the outputs of these equations
       * must be too; we do not need an explicit range-limiting operation.
       * Hence the value being shifted is never negative, and we don't
       * need the general RIGHT_SHIFT macro.
       */
      /* Y */
      outptr0[col] = (_JSAMPLE)((ctab[r + R_Y_OFF] + ctab[g + G_Y_OFF] +
                                 ctab[b + B_Y_OFF]) >> SCALEBITS);
      /* Cb */
      outptr1[col] = (_JSAMPLE)((ctab[r + R_CB_OFF] + ctab[g + G_CB_OFF] +
                                 ctab[b + B_CB_OFF]) >> SCALEBITS);
      /* Cr */
      outptr2[col] = (_JSAMPLE)((ctab[r + R_CR_OFF] + ctab[g + G_CR_OFF] +
                                 ctab[b + B_CR_OFF]) >> SCALEBITS);
    }
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/**************** Cases other than RGB -> YCbCr **************/


/*
 * Convert some rows of samples to the JPEG colorspace.
 * This version handles RGB->grayscale conversion, which is the same
 * as the RGB->Y portion of RGB->YCbCr.
 * We assume rgb_ycc_start has been called (we only use the Y tables).
 */

INLINE
LOCAL(void)
rgb_gray_convert_internal(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                          _JSAMPIMAGE output_buf, JDIMENSION output_row,
                          int num_rows)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  register int r, g, b;
  register JLONG *ctab = cconvert->rgb_ycc_tab;
  register _JSAMPROW inptr;
  register _JSAMPROW outptr;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->image_width;

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr = output_buf[0][output_row];
    output_row++;
    for (col = 0; col < num_cols; col++) {
      r = RANGE_LIMIT(inptr[RGB_RED]);
      g = RANGE_LIMIT(inptr[RGB_GREEN]);
      b = RANGE_LIMIT(inptr[RGB_BLUE]);
      inptr += RGB_PIXELSIZE;
      /* Y */
      outptr[col] = (_JSAMPLE)((ctab[r + R_Y_OFF] + ctab[g + G_Y_OFF] +
                                ctab[b + B_Y_OFF]) >> SCALEBITS);
    }
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/*
 * Convert some rows of samples to the JPEG colorspace.
 * This version handles extended RGB->plain RGB conversion
 */

INLINE
LOCAL(void)
rgb_rgb_convert_internal(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                         _JSAMPIMAGE output_buf, JDIMENSION output_row,
                         int num_rows)
{
  register _JSAMPROW inptr;
  register _JSAMPROW outptr0, outptr1, outptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->image_width;

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr0 = output_buf[0][output_row];
    outptr1 = output_buf[1][output_row];
    outptr2 = output_buf[2][output_row];
    output_row++;
    for (col = 0; col < num_cols; col++) {
      outptr0[col] = inptr[RGB_RED];
      outptr1[col] = inptr[RGB_GREEN];
      outptr2[col] = inptr[RGB_BLUE];
      inptr += RGB_PIXELSIZE;
    }
  }
}
