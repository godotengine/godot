/*
 * jdcolext.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2009, 2011, 2015, 2022-2023, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains output colorspace conversion routines.
 */


/* This file is included by jdcolor.c */


/*
 * Convert some rows of samples to the output colorspace.
 *
 * Note that we change from noninterleaved, one-plane-per-component format
 * to interleaved-pixel format.  The output buffer is therefore three times
 * as wide as the input buffer.
 * A starting row offset is provided only for the input buffer.  The caller
 * can easily adjust the passed output_buf value to accommodate any row
 * offset required on that side.
 */

INLINE
LOCAL(void)
ycc_rgb_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
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
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      y  = inptr0[col];
      cb = inptr1[col];
      cr = inptr2[col];
      /* Range-limiting is essential due to noise introduced by DCT losses. */
      outptr[RGB_RED] =   range_limit[y + Crrtab[cr]];
      outptr[RGB_GREEN] = range_limit[y +
                              ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                                SCALEBITS))];
      outptr[RGB_BLUE] =  range_limit[y + Cbbtab[cb]];
      /* Set unused byte to _MAXJSAMPLE so it can be interpreted as an */
      /* opaque alpha channel value */
#ifdef RGB_ALPHA
      outptr[RGB_ALPHA] = _MAXJSAMPLE;
#endif
      outptr += RGB_PIXELSIZE;
    }
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/*
 * Convert grayscale to RGB: just duplicate the graylevel three times.
 * This is provided to support applications that don't want to cope
 * with grayscale as a separate case.
 */

INLINE
LOCAL(void)
gray_rgb_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                          JDIMENSION input_row, _JSAMPARRAY output_buf,
                          int num_rows)
{
  register _JSAMPROW inptr, outptr;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    inptr = input_buf[0][input_row++];
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      outptr[RGB_RED] = outptr[RGB_GREEN] = outptr[RGB_BLUE] = inptr[col];
      /* Set unused byte to _MAXJSAMPLE so it can be interpreted as an */
      /* opaque alpha channel value */
#ifdef RGB_ALPHA
      outptr[RGB_ALPHA] = _MAXJSAMPLE;
#endif
      outptr += RGB_PIXELSIZE;
    }
  }
}


/*
 * Convert RGB to extended RGB: just swap the order of source pixels
 */

INLINE
LOCAL(void)
rgb_rgb_convert_internal(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                         JDIMENSION input_row, _JSAMPARRAY output_buf,
                         int num_rows)
{
  register _JSAMPROW inptr0, inptr1, inptr2;
  register _JSAMPROW outptr;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      outptr[RGB_RED] = inptr0[col];
      outptr[RGB_GREEN] = inptr1[col];
      outptr[RGB_BLUE] = inptr2[col];
      /* Set unused byte to _MAXJSAMPLE so it can be interpreted as an */
      /* opaque alpha channel value */
#ifdef RGB_ALPHA
      outptr[RGB_ALPHA] = _MAXJSAMPLE;
#endif
      outptr += RGB_PIXELSIZE;
    }
  }
}
