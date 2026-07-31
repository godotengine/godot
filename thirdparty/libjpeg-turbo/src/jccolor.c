/*
 * jccolor.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009-2012, 2015, 2022, 2024, D. R. Commander.
 * Copyright (C) 2014, MIPS Technologies, Inc., California.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains input colorspace conversion routines.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jsimd.h"
#include "jsamplecomp.h"


#if BITS_IN_JSAMPLE != 16 || defined(C_LOSSLESS_SUPPORTED)

/* Private subobject */

typedef struct {
  struct jpeg_color_converter pub; /* public fields */

#if BITS_IN_JSAMPLE != 16
  /* Private state for RGB->YCC conversion */
  JLONG *rgb_ycc_tab;           /* => table for RGB to YCbCr conversion */
#endif
} my_color_converter;

typedef my_color_converter *my_cconvert_ptr;


/**************** RGB -> YCbCr conversion: most common case **************/

/*
 * YCbCr is defined per CCIR 601-1, except that Cb and Cr are
 * normalized to the range 0.._MAXJSAMPLE rather than -0.5 .. 0.5.
 * The conversion equations to be implemented are therefore
 *      Y  =  0.29900 * R + 0.58700 * G + 0.11400 * B
 *      Cb = -0.16874 * R - 0.33126 * G + 0.50000 * B  + _CENTERJSAMPLE
 *      Cr =  0.50000 * R - 0.41869 * G - 0.08131 * B  + _CENTERJSAMPLE
 * (These numbers are derived from TIFF 6.0 section 21, dated 3-June-92.)
 * Note: older versions of the IJG code used a zero offset of _MAXJSAMPLE/2,
 * rather than _CENTERJSAMPLE, for Cb and Cr.  This gave equal positive and
 * negative swings for Cb/Cr, but meant that grayscale values (Cb=Cr=0)
 * were not represented exactly.  Now we sacrifice exact representation of
 * maximum red and maximum blue in order to get exact grayscales.
 *
 * To avoid floating-point arithmetic, we represent the fractional constants
 * as integers scaled up by 2^16 (about 4 digits precision); we have to divide
 * the products by 2^16, with appropriate rounding, to get the correct answer.
 *
 * For even more speed, we avoid doing any multiplications in the inner loop
 * by precalculating the constants times R,G,B for all possible values.
 * For 8-bit samples this is very reasonable (only 256 entries per table);
 * for 12-bit samples it is still acceptable.  It's not very reasonable for
 * 16-bit samples, but if you want lossless storage you shouldn't be changing
 * colorspace anyway.
 * The _CENTERJSAMPLE offsets and the rounding fudge-factor of 0.5 are included
 * in the tables to save adding them separately in the inner loop.
 */

#define SCALEBITS       16      /* speediest right-shift on some machines */
#define CBCR_OFFSET     ((JLONG)_CENTERJSAMPLE << SCALEBITS)
#define ONE_HALF        ((JLONG)1 << (SCALEBITS - 1))
#define FIX(x)          ((JLONG)((x) * (1L << SCALEBITS) + 0.5))

/* We allocate one big table and divide it up into eight parts, instead of
 * doing eight alloc_small requests.  This lets us use a single table base
 * address, which can be held in a register in the inner loops on many
 * machines (more than can hold all eight addresses, anyway).
 */

#define R_Y_OFF         0                       /* offset to R => Y section */
#define G_Y_OFF         (1 * (_MAXJSAMPLE + 1)) /* offset to G => Y section */
#define B_Y_OFF         (2 * (_MAXJSAMPLE + 1)) /* etc. */
#define R_CB_OFF        (3 * (_MAXJSAMPLE + 1))
#define G_CB_OFF        (4 * (_MAXJSAMPLE + 1))
#define B_CB_OFF        (5 * (_MAXJSAMPLE + 1))
#define R_CR_OFF        B_CB_OFF                /* B=>Cb, R=>Cr are the same */
#define G_CR_OFF        (6 * (_MAXJSAMPLE + 1))
#define B_CR_OFF        (7 * (_MAXJSAMPLE + 1))
#define TABLE_SIZE      (8 * (_MAXJSAMPLE + 1))

/* 12-bit samples use a 16-bit data type, so it is possible to pass
 * out-of-range sample values (< 0 or > 4095) to jpeg_write_scanlines().
 * Thus, we mask the incoming 12-bit samples to guard against overrunning
 * or underrunning the conversion tables.
 */

#if BITS_IN_JSAMPLE == 12
#define RANGE_LIMIT(value)  ((value) & 0xFFF)
#else
#define RANGE_LIMIT(value)  (value)
#endif


/* Include inline routines for colorspace extensions */

#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define rgb_ycc_convert_internal  extrgb_ycc_convert_internal
#define rgb_gray_convert_internal  extrgb_gray_convert_internal
#define rgb_rgb_convert_internal  extrgb_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define rgb_ycc_convert_internal  extrgbx_ycc_convert_internal
#define rgb_gray_convert_internal  extrgbx_gray_convert_internal
#define rgb_rgb_convert_internal  extrgbx_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define rgb_ycc_convert_internal  extbgr_ycc_convert_internal
#define rgb_gray_convert_internal  extbgr_gray_convert_internal
#define rgb_rgb_convert_internal  extbgr_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define rgb_ycc_convert_internal  extbgrx_ycc_convert_internal
#define rgb_gray_convert_internal  extbgrx_gray_convert_internal
#define rgb_rgb_convert_internal  extbgrx_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define rgb_ycc_convert_internal  extxbgr_ycc_convert_internal
#define rgb_gray_convert_internal  extxbgr_gray_convert_internal
#define rgb_rgb_convert_internal  extxbgr_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define rgb_ycc_convert_internal  extxrgb_ycc_convert_internal
#define rgb_gray_convert_internal  extxrgb_gray_convert_internal
#define rgb_rgb_convert_internal  extxrgb_rgb_convert_internal
#include "jccolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef rgb_ycc_convert_internal
#undef rgb_gray_convert_internal
#undef rgb_rgb_convert_internal


/*
 * Initialize for RGB->YCC colorspace conversion.
 */

METHODDEF(void)
rgb_ycc_start(j_compress_ptr cinfo)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  JLONG *rgb_ycc_tab;
  JLONG i;

  /* Allocate and fill in the conversion tables. */
  cconvert->rgb_ycc_tab = rgb_ycc_tab = (JLONG *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (TABLE_SIZE * sizeof(JLONG)));

  for (i = 0; i <= _MAXJSAMPLE; i++) {
    rgb_ycc_tab[i + R_Y_OFF] = FIX(0.29900) * i;
    rgb_ycc_tab[i + G_Y_OFF] = FIX(0.58700) * i;
    rgb_ycc_tab[i + B_Y_OFF] = FIX(0.11400) * i   + ONE_HALF;
    rgb_ycc_tab[i + R_CB_OFF] = (-FIX(0.16874)) * i;
    rgb_ycc_tab[i + G_CB_OFF] = (-FIX(0.33126)) * i;
    /* We use a rounding fudge-factor of 0.5-epsilon for Cb and Cr.
     * This ensures that the maximum output will round to _MAXJSAMPLE
     * not _MAXJSAMPLE+1, and thus that we don't have to range-limit.
     */
    rgb_ycc_tab[i + B_CB_OFF] = FIX(0.50000) * i  + CBCR_OFFSET + ONE_HALF - 1;
/*  B=>Cb and R=>Cr tables are the same
    rgb_ycc_tab[i + R_CR_OFF] = FIX(0.50000) * i  + CBCR_OFFSET + ONE_HALF - 1;
*/
    rgb_ycc_tab[i + G_CR_OFF] = (-FIX(0.41869)) * i;
    rgb_ycc_tab[i + B_CR_OFF] = (-FIX(0.08131)) * i;
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/*
 * Convert some rows of samples to the JPEG colorspace.
 */

METHODDEF(void)
rgb_ycc_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    extrgb_ycc_convert_internal(cinfo, input_buf, output_buf, output_row,
                                num_rows);
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    extrgbx_ycc_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  case JCS_EXT_BGR:
    extbgr_ycc_convert_internal(cinfo, input_buf, output_buf, output_row,
                                num_rows);
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    extbgrx_ycc_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    extxbgr_ycc_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    extxrgb_ycc_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  default:
    rgb_ycc_convert_internal(cinfo, input_buf, output_buf, output_row,
                             num_rows);
    break;
  }
}


/**************** Cases other than RGB -> YCbCr **************/


/*
 * Convert some rows of samples to the JPEG colorspace.
 */

METHODDEF(void)
rgb_gray_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                 _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    extrgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    extrgbx_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
    break;
  case JCS_EXT_BGR:
    extbgr_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    extbgrx_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    extxbgr_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    extxrgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                                  num_rows);
    break;
  default:
    rgb_gray_convert_internal(cinfo, input_buf, output_buf, output_row,
                              num_rows);
    break;
  }
}


/*
 * Extended RGB to plain RGB conversion
 */

METHODDEF(void)
rgb_rgb_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  switch (cinfo->in_color_space) {
  case JCS_EXT_RGB:
    extrgb_rgb_convert_internal(cinfo, input_buf, output_buf, output_row,
                                num_rows);
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    extrgbx_rgb_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  case JCS_EXT_BGR:
    extbgr_rgb_convert_internal(cinfo, input_buf, output_buf, output_row,
                                num_rows);
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    extbgrx_rgb_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    extxbgr_rgb_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    extxrgb_rgb_convert_internal(cinfo, input_buf, output_buf, output_row,
                                 num_rows);
    break;
  default:
    rgb_rgb_convert_internal(cinfo, input_buf, output_buf, output_row,
                             num_rows);
    break;
  }
}


/*
 * Convert some rows of samples to the JPEG colorspace.
 * This version handles Adobe-style CMYK->YCCK conversion,
 * where we convert R=1-C, G=1-M, and B=1-Y to YCbCr using the same
 * conversion as above, while passing K (black) unchanged.
 * We assume rgb_ycc_start has been called.
 */

METHODDEF(void)
cmyk_ycck_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                  _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  register int r, g, b;
  register JLONG *ctab = cconvert->rgb_ycc_tab;
  register _JSAMPROW inptr;
  register _JSAMPROW outptr0, outptr1, outptr2, outptr3;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->image_width;

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr0 = output_buf[0][output_row];
    outptr1 = output_buf[1][output_row];
    outptr2 = output_buf[2][output_row];
    outptr3 = output_buf[3][output_row];
    output_row++;
    for (col = 0; col < num_cols; col++) {
      r = _MAXJSAMPLE - RANGE_LIMIT(inptr[0]);
      g = _MAXJSAMPLE - RANGE_LIMIT(inptr[1]);
      b = _MAXJSAMPLE - RANGE_LIMIT(inptr[2]);
      /* K passes through as-is */
      outptr3[col] = inptr[3];
      inptr += 4;
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


/*
 * Convert some rows of samples to the JPEG colorspace.
 * This version handles grayscale output with no conversion.
 * The source can be either plain grayscale or YCbCr (since Y == gray).
 */

METHODDEF(void)
grayscale_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
                  _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  register _JSAMPROW inptr;
  register _JSAMPROW outptr;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->image_width;
  int instride = cinfo->input_components;

  while (--num_rows >= 0) {
    inptr = *input_buf++;
    outptr = output_buf[0][output_row];
    output_row++;
    for (col = 0; col < num_cols; col++) {
      outptr[col] = inptr[0];
      inptr += instride;
    }
  }
}


/*
 * Convert some rows of samples to the JPEG colorspace.
 * This version handles multi-component colorspaces without conversion.
 * We assume input_components == num_components.
 */

METHODDEF(void)
null_convert(j_compress_ptr cinfo, _JSAMPARRAY input_buf,
             _JSAMPIMAGE output_buf, JDIMENSION output_row, int num_rows)
{
  register _JSAMPROW inptr;
  register _JSAMPROW outptr, outptr0, outptr1, outptr2, outptr3;
  register JDIMENSION col;
  register int ci;
  int nc = cinfo->num_components;
  JDIMENSION num_cols = cinfo->image_width;

  if (nc == 3) {
    while (--num_rows >= 0) {
      inptr = *input_buf++;
      outptr0 = output_buf[0][output_row];
      outptr1 = output_buf[1][output_row];
      outptr2 = output_buf[2][output_row];
      output_row++;
      for (col = 0; col < num_cols; col++) {
        outptr0[col] = *inptr++;
        outptr1[col] = *inptr++;
        outptr2[col] = *inptr++;
      }
    }
  } else if (nc == 4) {
    while (--num_rows >= 0) {
      inptr = *input_buf++;
      outptr0 = output_buf[0][output_row];
      outptr1 = output_buf[1][output_row];
      outptr2 = output_buf[2][output_row];
      outptr3 = output_buf[3][output_row];
      output_row++;
      for (col = 0; col < num_cols; col++) {
        outptr0[col] = *inptr++;
        outptr1[col] = *inptr++;
        outptr2[col] = *inptr++;
        outptr3[col] = *inptr++;
      }
    }
  } else {
    while (--num_rows >= 0) {
      /* It seems fastest to make a separate pass for each component. */
      for (ci = 0; ci < nc; ci++) {
        inptr = *input_buf;
        outptr = output_buf[ci][output_row];
        for (col = 0; col < num_cols; col++) {
          outptr[col] = inptr[ci];
          inptr += nc;
        }
      }
      input_buf++;
      output_row++;
    }
  }
}


/*
 * Empty method for start_pass.
 */

METHODDEF(void)
null_method(j_compress_ptr cinfo)
{
  /* no work needed */
}


/*
 * Module initialization routine for input colorspace conversion.
 */

GLOBAL(void)
_jinit_color_converter(j_compress_ptr cinfo)
{
  my_cconvert_ptr cconvert;

#ifdef C_LOSSLESS_SUPPORTED
  if (cinfo->master->lossless) {
#if BITS_IN_JSAMPLE == 8
    if (cinfo->data_precision > BITS_IN_JSAMPLE || cinfo->data_precision < 2)
#else
    if (cinfo->data_precision > BITS_IN_JSAMPLE ||
        cinfo->data_precision < BITS_IN_JSAMPLE - 3)
#endif
      ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
  } else
#endif
  {
    if (cinfo->data_precision != BITS_IN_JSAMPLE)
      ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
  }

  cconvert = (my_cconvert_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(my_color_converter));
  cinfo->cconvert = (struct jpeg_color_converter *)cconvert;
  /* set start_pass to null method until we find out differently */
  cconvert->pub.start_pass = null_method;

  /* Make sure input_components agrees with in_color_space */
  switch (cinfo->in_color_space) {
  case JCS_GRAYSCALE:
    if (cinfo->input_components != 1)
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    break;

  case JCS_RGB:
  case JCS_EXT_RGB:
  case JCS_EXT_RGBX:
  case JCS_EXT_BGR:
  case JCS_EXT_BGRX:
  case JCS_EXT_XBGR:
  case JCS_EXT_XRGB:
  case JCS_EXT_RGBA:
  case JCS_EXT_BGRA:
  case JCS_EXT_ABGR:
  case JCS_EXT_ARGB:
    if (cinfo->input_components != rgb_pixelsize[cinfo->in_color_space])
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    break;

  case JCS_YCbCr:
    if (cinfo->input_components != 3)
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    break;

  case JCS_CMYK:
  case JCS_YCCK:
    if (cinfo->input_components != 4)
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    break;

  default:                      /* JCS_UNKNOWN can be anything */
    if (cinfo->input_components < 1)
      ERREXIT(cinfo, JERR_BAD_IN_COLORSPACE);
    break;
  }

  /* Check num_components, set conversion method based on requested space.
   * NOTE: We do not allow any lossy color conversion algorithms in lossless
   * mode.
   */
  switch (cinfo->jpeg_color_space) {
  case JCS_GRAYSCALE:
#ifdef C_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless &&
        cinfo->in_color_space != cinfo->jpeg_color_space)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    if (cinfo->num_components != 1)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    if (cinfo->in_color_space == JCS_GRAYSCALE)
      cconvert->pub._color_convert = grayscale_convert;
    else if (IsExtRGB(cinfo->in_color_space)) {
#ifdef WITH_SIMD
      if (jsimd_can_rgb_gray())
        cconvert->pub._color_convert = jsimd_rgb_gray_convert;
      else
#endif
      {
        cconvert->pub.start_pass = rgb_ycc_start;
        cconvert->pub._color_convert = rgb_gray_convert;
      }
    } else if (cinfo->in_color_space == JCS_YCbCr)
      cconvert->pub._color_convert = grayscale_convert;
    else
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    break;

  case JCS_RGB:
#ifdef C_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless && !IsExtRGB(cinfo->in_color_space))
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    if (cinfo->num_components != 3)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    if (rgb_red[cinfo->in_color_space] == 0 &&
        rgb_green[cinfo->in_color_space] == 1 &&
        rgb_blue[cinfo->in_color_space] == 2 &&
        rgb_pixelsize[cinfo->in_color_space] == 3) {
#if defined(WITH_SIMD) && defined(__mips__)
      if (jsimd_c_can_null_convert())
        cconvert->pub._color_convert = jsimd_c_null_convert;
      else
#endif
        cconvert->pub._color_convert = null_convert;
    } else if (IsExtRGB(cinfo->in_color_space))
      cconvert->pub._color_convert = rgb_rgb_convert;
    else
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    break;

  case JCS_YCbCr:
#ifdef C_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless &&
        cinfo->in_color_space != cinfo->jpeg_color_space)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    if (cinfo->num_components != 3)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    if (IsExtRGB(cinfo->in_color_space)) {
#ifdef WITH_SIMD
      if (jsimd_can_rgb_ycc())
        cconvert->pub._color_convert = jsimd_rgb_ycc_convert;
      else
#endif
      {
        cconvert->pub.start_pass = rgb_ycc_start;
        cconvert->pub._color_convert = rgb_ycc_convert;
      }
    } else if (cinfo->in_color_space == JCS_YCbCr) {
#if defined(WITH_SIMD) && defined(__mips__)
      if (jsimd_c_can_null_convert())
        cconvert->pub._color_convert = jsimd_c_null_convert;
      else
#endif
        cconvert->pub._color_convert = null_convert;
    } else
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    break;

  case JCS_CMYK:
#ifdef C_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless &&
        cinfo->in_color_space != cinfo->jpeg_color_space)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    if (cinfo->num_components != 4)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    if (cinfo->in_color_space == JCS_CMYK) {
#if defined(WITH_SIMD) && defined(__mips__)
      if (jsimd_c_can_null_convert())
        cconvert->pub._color_convert = jsimd_c_null_convert;
      else
#endif
        cconvert->pub._color_convert = null_convert;
    } else
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    break;

  case JCS_YCCK:
#ifdef C_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless &&
        cinfo->in_color_space != cinfo->jpeg_color_space)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    if (cinfo->num_components != 4)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    if (cinfo->in_color_space == JCS_CMYK) {
      cconvert->pub.start_pass = rgb_ycc_start;
      cconvert->pub._color_convert = cmyk_ycck_convert;
    } else if (cinfo->in_color_space == JCS_YCCK) {
#if defined(WITH_SIMD) && defined(__mips__)
      if (jsimd_c_can_null_convert())
        cconvert->pub._color_convert = jsimd_c_null_convert;
      else
#endif
        cconvert->pub._color_convert = null_convert;
    } else
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    break;

  default:                      /* allow null conversion of JCS_UNKNOWN */
    if (cinfo->jpeg_color_space != cinfo->in_color_space ||
        cinfo->num_components != cinfo->input_components)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#if defined(WITH_SIMD) && defined(__mips__)
    if (jsimd_c_can_null_convert())
      cconvert->pub._color_convert = jsimd_c_null_convert;
    else
#endif
      cconvert->pub._color_convert = null_convert;
    break;
  }
}

#endif /* BITS_IN_JSAMPLE != 16 || defined(C_LOSSLESS_SUPPORTED) */
