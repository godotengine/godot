/*
 * jdcolor.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2011 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009, 2011-2012, 2014-2015, 2022, 2024, D. R. Commander.
 * Copyright (C) 2013, Linaro Limited.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains output colorspace conversion routines.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jsimd.h"
#include "jsamplecomp.h"


#if BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED)

/* Private subobject */

typedef struct {
  struct jpeg_color_deconverter pub; /* public fields */

#if BITS_IN_JSAMPLE != 16
  /* Private state for YCC->RGB conversion */
  int *Cr_r_tab;                /* => table for Cr to R conversion */
  int *Cb_b_tab;                /* => table for Cb to B conversion */
  JLONG *Cr_g_tab;              /* => table for Cr to G conversion */
  JLONG *Cb_g_tab;              /* => table for Cb to G conversion */

  /* Private state for RGB->Y conversion */
  JLONG *rgb_y_tab;             /* => table for RGB to Y conversion */
#endif
} my_color_deconverter;

typedef my_color_deconverter *my_cconvert_ptr;


/**************** YCbCr -> RGB conversion: most common case **************/
/****************   RGB -> Y   conversion: less common case **************/

/*
 * YCbCr is defined per CCIR 601-1, except that Cb and Cr are
 * normalized to the range 0.._MAXJSAMPLE rather than -0.5 .. 0.5.
 * The conversion equations to be implemented are therefore
 *
 *      R = Y                + 1.40200 * Cr
 *      G = Y - 0.34414 * Cb - 0.71414 * Cr
 *      B = Y + 1.77200 * Cb
 *
 *      Y = 0.29900 * R + 0.58700 * G + 0.11400 * B
 *
 * where Cb and Cr represent the incoming values less _CENTERJSAMPLE.
 * (These numbers are derived from TIFF 6.0 section 21, dated 3-June-92.)
 *
 * To avoid floating-point arithmetic, we represent the fractional constants
 * as integers scaled up by 2^16 (about 4 digits precision); we have to divide
 * the products by 2^16, with appropriate rounding, to get the correct answer.
 * Notice that Y, being an integral input, does not contribute any fraction
 * so it need not participate in the rounding.
 *
 * For even more speed, we avoid doing any multiplications in the inner loop
 * by precalculating the constants times Cb and Cr for all possible values.
 * For 8-bit samples this is very reasonable (only 256 entries per table);
 * for 12-bit samples it is still acceptable.  It's not very reasonable for
 * 16-bit samples, but if you want lossless storage you shouldn't be changing
 * colorspace anyway.
 * The Cr=>R and Cb=>B values can be rounded to integers in advance; the
 * values for the G calculation are left scaled up, since we must add them
 * together before rounding.
 */

#define SCALEBITS       16      /* speediest right-shift on some machines */
#define ONE_HALF        ((JLONG)1 << (SCALEBITS - 1))
#define FIX(x)          ((JLONG)((x) * (1L << SCALEBITS) + 0.5))

/* We allocate one big table for RGB->Y conversion and divide it up into
 * three parts, instead of doing three alloc_small requests.  This lets us
 * use a single table base address, which can be held in a register in the
 * inner loops on many machines (more than can hold all three addresses,
 * anyway).
 */

#define R_Y_OFF         0                       /* offset to R => Y section */
#define G_Y_OFF         (1 * (_MAXJSAMPLE + 1)) /* offset to G => Y section */
#define B_Y_OFF         (2 * (_MAXJSAMPLE + 1)) /* etc. */
#define TABLE_SIZE      (3 * (_MAXJSAMPLE + 1))


/* Include inline routines for colorspace extensions */

#include "jdcolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define ycc_rgb_convert_internal  ycc_extrgb_convert_internal
#define gray_rgb_convert_internal  gray_extrgb_convert_internal
#define rgb_rgb_convert_internal  rgb_extrgb_convert_internal
#include "jdcolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef ycc_rgb_convert_internal
#undef gray_rgb_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_ALPHA  3
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define ycc_rgb_convert_internal  ycc_extrgbx_convert_internal
#define gray_rgb_convert_internal  gray_extrgbx_convert_internal
#define rgb_rgb_convert_internal  rgb_extrgbx_convert_internal
#include "jdcolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef ycc_rgb_convert_internal
#undef gray_rgb_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define ycc_rgb_convert_internal  ycc_extbgr_convert_internal
#define gray_rgb_convert_internal  gray_extbgr_convert_internal
#define rgb_rgb_convert_internal  rgb_extbgr_convert_internal
#include "jdcolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef ycc_rgb_convert_internal
#undef gray_rgb_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_ALPHA  3
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define ycc_rgb_convert_internal  ycc_extbgrx_convert_internal
#define gray_rgb_convert_internal  gray_extbgrx_convert_internal
#define rgb_rgb_convert_internal  rgb_extbgrx_convert_internal
#include "jdcolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef ycc_rgb_convert_internal
#undef gray_rgb_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_ALPHA  0
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define ycc_rgb_convert_internal  ycc_extxbgr_convert_internal
#define gray_rgb_convert_internal  gray_extxbgr_convert_internal
#define rgb_rgb_convert_internal  rgb_extxbgr_convert_internal
#include "jdcolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef ycc_rgb_convert_internal
#undef gray_rgb_convert_internal
#undef rgb_rgb_convert_internal

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_ALPHA  0
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define ycc_rgb_convert_internal  ycc_extxrgb_convert_internal
#define gray_rgb_convert_internal  gray_extxrgb_convert_internal
#define rgb_rgb_convert_internal  rgb_extxrgb_convert_internal
#include "jdcolext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef ycc_rgb_convert_internal
#undef gray_rgb_convert_internal
#undef rgb_rgb_convert_internal


/*
 * Initialize tables for YCC->RGB colorspace conversion.
 */

LOCAL(void)
build_ycc_rgb_table(j_decompress_ptr cinfo)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  int i;
  JLONG x;
  SHIFT_TEMPS

  cconvert->Cr_r_tab = (int *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (_MAXJSAMPLE + 1) * sizeof(int));
  cconvert->Cb_b_tab = (int *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (_MAXJSAMPLE + 1) * sizeof(int));
  cconvert->Cr_g_tab = (JLONG *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (_MAXJSAMPLE + 1) * sizeof(JLONG));
  cconvert->Cb_g_tab = (JLONG *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (_MAXJSAMPLE + 1) * sizeof(JLONG));

  for (i = 0, x = -_CENTERJSAMPLE; i <= _MAXJSAMPLE; i++, x++) {
    /* i is the actual input pixel value, in the range 0.._MAXJSAMPLE */
    /* The Cb or Cr value we are thinking of is x = i - _CENTERJSAMPLE */
    /* Cr=>R value is nearest int to 1.40200 * x */
    cconvert->Cr_r_tab[i] = (int)
                    RIGHT_SHIFT(FIX(1.40200) * x + ONE_HALF, SCALEBITS);
    /* Cb=>B value is nearest int to 1.77200 * x */
    cconvert->Cb_b_tab[i] = (int)
                    RIGHT_SHIFT(FIX(1.77200) * x + ONE_HALF, SCALEBITS);
    /* Cr=>G value is scaled-up -0.71414 * x */
    cconvert->Cr_g_tab[i] = (-FIX(0.71414)) * x;
    /* Cb=>G value is scaled-up -0.34414 * x */
    /* We also add in ONE_HALF so that need not do it in inner loop */
    cconvert->Cb_g_tab[i] = (-FIX(0.34414)) * x + ONE_HALF;
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/*
 * Convert some rows of samples to the output colorspace.
 */

METHODDEF(void)
ycc_rgb_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    ycc_extrgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                                num_rows);
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    ycc_extrgbx_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  case JCS_EXT_BGR:
    ycc_extbgr_convert_internal(cinfo, input_buf, input_row, output_buf,
                                num_rows);
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    ycc_extbgrx_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    ycc_extxbgr_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    ycc_extxrgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  default:
    ycc_rgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                             num_rows);
    break;
  }
}


/**************** Cases other than YCbCr -> RGB **************/


/*
 * Initialize for RGB->grayscale colorspace conversion.
 */

LOCAL(void)
build_rgb_y_table(j_decompress_ptr cinfo)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  JLONG *rgb_y_tab;
  JLONG i;

  /* Allocate and fill in the conversion tables. */
  cconvert->rgb_y_tab = rgb_y_tab = (JLONG *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (TABLE_SIZE * sizeof(JLONG)));

  for (i = 0; i <= _MAXJSAMPLE; i++) {
    rgb_y_tab[i + R_Y_OFF] = FIX(0.29900) * i;
    rgb_y_tab[i + G_Y_OFF] = FIX(0.58700) * i;
    rgb_y_tab[i + B_Y_OFF] = FIX(0.11400) * i + ONE_HALF;
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/*
 * Convert RGB to grayscale.
 */

METHODDEF(void)
rgb_gray_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                 JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  register int r, g, b;
  register JLONG *ctab = cconvert->rgb_y_tab;
  register _JSAMPROW outptr;
  register _JSAMPROW inptr0, inptr1, inptr2;
  register JDIMENSION col;
  JDIMENSION num_cols = cinfo->output_width;

  while (--num_rows >= 0) {
    inptr0 = input_buf[0][input_row];
    inptr1 = input_buf[1][input_row];
    inptr2 = input_buf[2][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      r = inptr0[col];
      g = inptr1[col];
      b = inptr2[col];
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
 * Color conversion for no colorspace change: just copy the data,
 * converting from separate-planes to interleaved representation.
 */

METHODDEF(void)
null_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
             JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  register _JSAMPROW inptr, inptr0, inptr1, inptr2, inptr3, outptr;
  register JDIMENSION col;
  register int num_components = cinfo->num_components;
  JDIMENSION num_cols = cinfo->output_width;
  int ci;

  if (num_components == 3) {
    while (--num_rows >= 0) {
      inptr0 = input_buf[0][input_row];
      inptr1 = input_buf[1][input_row];
      inptr2 = input_buf[2][input_row];
      input_row++;
      outptr = *output_buf++;
      for (col = 0; col < num_cols; col++) {
        *outptr++ = inptr0[col];
        *outptr++ = inptr1[col];
        *outptr++ = inptr2[col];
      }
    }
  } else if (num_components == 4) {
    while (--num_rows >= 0) {
      inptr0 = input_buf[0][input_row];
      inptr1 = input_buf[1][input_row];
      inptr2 = input_buf[2][input_row];
      inptr3 = input_buf[3][input_row];
      input_row++;
      outptr = *output_buf++;
      for (col = 0; col < num_cols; col++) {
        *outptr++ = inptr0[col];
        *outptr++ = inptr1[col];
        *outptr++ = inptr2[col];
        *outptr++ = inptr3[col];
      }
    }
  } else {
    while (--num_rows >= 0) {
      for (ci = 0; ci < num_components; ci++) {
        inptr = input_buf[ci][input_row];
        outptr = *output_buf;
        for (col = 0; col < num_cols; col++) {
          outptr[ci] = inptr[col];
          outptr += num_components;
        }
      }
      output_buf++;
      input_row++;
    }
  }
}


/*
 * Color conversion for grayscale: just copy the data.
 * This also works for YCbCr -> grayscale conversion, in which
 * we just copy the Y (luminance) component and ignore chrominance.
 */

METHODDEF(void)
grayscale_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                  JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  _jcopy_sample_rows(input_buf[0], (int)input_row, output_buf, 0, num_rows,
                     cinfo->output_width);
}


/*
 * Convert grayscale to RGB
 */

METHODDEF(void)
gray_rgb_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                 JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    gray_extrgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    gray_extrgbx_convert_internal(cinfo, input_buf, input_row, output_buf,
                                  num_rows);
    break;
  case JCS_EXT_BGR:
    gray_extbgr_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    gray_extbgrx_convert_internal(cinfo, input_buf, input_row, output_buf,
                                  num_rows);
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    gray_extxbgr_convert_internal(cinfo, input_buf, input_row, output_buf,
                                  num_rows);
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    gray_extxrgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                                  num_rows);
    break;
  default:
    gray_rgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                              num_rows);
    break;
  }
}


/*
 * Convert plain RGB to extended RGB
 */

METHODDEF(void)
rgb_rgb_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    rgb_extrgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                                num_rows);
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    rgb_extrgbx_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  case JCS_EXT_BGR:
    rgb_extbgr_convert_internal(cinfo, input_buf, input_row, output_buf,
                                num_rows);
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    rgb_extbgrx_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    rgb_extxbgr_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    rgb_extxrgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                                 num_rows);
    break;
  default:
    rgb_rgb_convert_internal(cinfo, input_buf, input_row, output_buf,
                             num_rows);
    break;
  }
}


/*
 * Adobe-style YCCK->CMYK conversion.
 * We convert YCbCr to R=1-C, G=1-M, and B=1-Y using the same
 * conversion as above, while passing K (black) unchanged.
 * We assume build_ycc_rgb_table has been called.
 */

METHODDEF(void)
ycck_cmyk_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                  JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
#if BITS_IN_JSAMPLE != 16
  my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
  register int y, cb, cr;
  register _JSAMPROW outptr;
  register _JSAMPROW inptr0, inptr1, inptr2, inptr3;
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
    inptr3 = input_buf[3][input_row];
    input_row++;
    outptr = *output_buf++;
    for (col = 0; col < num_cols; col++) {
      y  = inptr0[col];
      cb = inptr1[col];
      cr = inptr2[col];
      /* Range-limiting is essential due to noise introduced by DCT losses. */
      outptr[0] = range_limit[_MAXJSAMPLE - (y + Crrtab[cr])];  /* red */
      outptr[1] = range_limit[_MAXJSAMPLE - (y +                /* green */
                              ((int)RIGHT_SHIFT(Cbgtab[cb] + Crgtab[cr],
                                                 SCALEBITS)))];
      outptr[2] = range_limit[_MAXJSAMPLE - (y + Cbbtab[cb])];  /* blue */
      /* K passes through unchanged */
      outptr[3] = inptr3[col];
      outptr += 4;
    }
  }
#else
  ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
}


/*
 * RGB565 conversion
 */

#define PACK_SHORT_565_LE(r, g, b) \
  ((((r) << 8) & 0xF800) | (((g) << 3) & 0x7E0) | ((b) >> 3))
#define PACK_SHORT_565_BE(r, g, b) \
  (((r) & 0xF8) | ((g) >> 5) | (((g) << 11) & 0xE000) | (((b) << 5) & 0x1F00))

#define PACK_TWO_PIXELS_LE(l, r)    ((r << 16) | l)
#define PACK_TWO_PIXELS_BE(l, r)    ((l << 16) | r)

#define PACK_NEED_ALIGNMENT(ptr)    (((size_t)(ptr)) & 3)

#define WRITE_TWO_ALIGNED_PIXELS(addr, pixels)  ((*(int *)(addr)) = pixels)

#define DITHER_565_R(r, dither)  ((r) + ((dither) & 0xFF))
#define DITHER_565_G(g, dither)  ((g) + (((dither) & 0xFF) >> 1))
#define DITHER_565_B(b, dither)  ((b) + ((dither) & 0xFF))


/* Declarations for ordered dithering
 *
 * We use a 4x4 ordered dither array packed into 32 bits.  This array is
 * sufficient for dithering RGB888 to RGB565.
 */

#define DITHER_MASK       0x3
#define DITHER_ROTATE(x)  ((((x) & 0xFF) << 24) | (((x) >> 8) & 0x00FFFFFF))
static const JLONG dither_matrix[4] = {
  0x0008020A,
  0x0C040E06,
  0x030B0109,
  0x0F070D05
};


static INLINE boolean is_big_endian(void)
{
  int test_value = 1;
  if (*(char *)&test_value != 1)
    return TRUE;
  return FALSE;
}


/* Include inline routines for RGB565 conversion */

#define PACK_SHORT_565  PACK_SHORT_565_LE
#define PACK_TWO_PIXELS  PACK_TWO_PIXELS_LE
#define ycc_rgb565_convert_internal  ycc_rgb565_convert_le
#define ycc_rgb565D_convert_internal  ycc_rgb565D_convert_le
#define rgb_rgb565_convert_internal  rgb_rgb565_convert_le
#define rgb_rgb565D_convert_internal  rgb_rgb565D_convert_le
#define gray_rgb565_convert_internal  gray_rgb565_convert_le
#define gray_rgb565D_convert_internal  gray_rgb565D_convert_le
#include "jdcol565.c"
#undef PACK_SHORT_565
#undef PACK_TWO_PIXELS
#undef ycc_rgb565_convert_internal
#undef ycc_rgb565D_convert_internal
#undef rgb_rgb565_convert_internal
#undef rgb_rgb565D_convert_internal
#undef gray_rgb565_convert_internal
#undef gray_rgb565D_convert_internal

#define PACK_SHORT_565  PACK_SHORT_565_BE
#define PACK_TWO_PIXELS  PACK_TWO_PIXELS_BE
#define ycc_rgb565_convert_internal  ycc_rgb565_convert_be
#define ycc_rgb565D_convert_internal  ycc_rgb565D_convert_be
#define rgb_rgb565_convert_internal  rgb_rgb565_convert_be
#define rgb_rgb565D_convert_internal  rgb_rgb565D_convert_be
#define gray_rgb565_convert_internal  gray_rgb565_convert_be
#define gray_rgb565D_convert_internal  gray_rgb565D_convert_be
#include "jdcol565.c"
#undef PACK_SHORT_565
#undef PACK_TWO_PIXELS
#undef ycc_rgb565_convert_internal
#undef ycc_rgb565D_convert_internal
#undef rgb_rgb565_convert_internal
#undef rgb_rgb565D_convert_internal
#undef gray_rgb565_convert_internal
#undef gray_rgb565D_convert_internal


METHODDEF(void)
ycc_rgb565_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                   JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  if (is_big_endian())
    ycc_rgb565_convert_be(cinfo, input_buf, input_row, output_buf, num_rows);
  else
    ycc_rgb565_convert_le(cinfo, input_buf, input_row, output_buf, num_rows);
}


METHODDEF(void)
ycc_rgb565D_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                    JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  if (is_big_endian())
    ycc_rgb565D_convert_be(cinfo, input_buf, input_row, output_buf, num_rows);
  else
    ycc_rgb565D_convert_le(cinfo, input_buf, input_row, output_buf, num_rows);
}


METHODDEF(void)
rgb_rgb565_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                   JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  if (is_big_endian())
    rgb_rgb565_convert_be(cinfo, input_buf, input_row, output_buf, num_rows);
  else
    rgb_rgb565_convert_le(cinfo, input_buf, input_row, output_buf, num_rows);
}


METHODDEF(void)
rgb_rgb565D_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                    JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  if (is_big_endian())
    rgb_rgb565D_convert_be(cinfo, input_buf, input_row, output_buf, num_rows);
  else
    rgb_rgb565D_convert_le(cinfo, input_buf, input_row, output_buf, num_rows);
}


METHODDEF(void)
gray_rgb565_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                    JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  if (is_big_endian())
    gray_rgb565_convert_be(cinfo, input_buf, input_row, output_buf, num_rows);
  else
    gray_rgb565_convert_le(cinfo, input_buf, input_row, output_buf, num_rows);
}


METHODDEF(void)
gray_rgb565D_convert(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                     JDIMENSION input_row, _JSAMPARRAY output_buf, int num_rows)
{
  if (is_big_endian())
    gray_rgb565D_convert_be(cinfo, input_buf, input_row, output_buf, num_rows);
  else
    gray_rgb565D_convert_le(cinfo, input_buf, input_row, output_buf, num_rows);
}


/*
 * Empty method for start_pass.
 */

METHODDEF(void)
start_pass_dcolor(j_decompress_ptr cinfo)
{
  /* no work needed */
}


/*
 * Module initialization routine for output colorspace conversion.
 */

GLOBAL(void)
_jinit_color_deconverter(j_decompress_ptr cinfo)
{
  my_cconvert_ptr cconvert;
  int ci;

#ifdef D_LOSSLESS_SUPPORTED
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
                                sizeof(my_color_deconverter));
  cinfo->cconvert = (struct jpeg_color_deconverter *)cconvert;
  cconvert->pub.start_pass = start_pass_dcolor;

  /* Make sure num_components agrees with jpeg_color_space */
  switch (cinfo->jpeg_color_space) {
  case JCS_GRAYSCALE:
    if (cinfo->num_components != 1)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    break;

  case JCS_RGB:
  case JCS_YCbCr:
    if (cinfo->num_components != 3)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    break;

  case JCS_CMYK:
  case JCS_YCCK:
    if (cinfo->num_components != 4)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    break;

  default:                      /* JCS_UNKNOWN can be anything */
    if (cinfo->num_components < 1)
      ERREXIT(cinfo, JERR_BAD_J_COLORSPACE);
    break;
  }

  /* Set out_color_components and conversion method based on requested space.
   * Also clear the component_needed flags for any unused components,
   * so that earlier pipeline stages can avoid useless computation.
   * NOTE: We do not allow any lossy color conversion algorithms in lossless
   * mode.
   */

  switch (cinfo->out_color_space) {
  case JCS_GRAYSCALE:
#ifdef D_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless &&
        cinfo->jpeg_color_space != cinfo->out_color_space)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    cinfo->out_color_components = 1;
    if (cinfo->jpeg_color_space == JCS_GRAYSCALE ||
        cinfo->jpeg_color_space == JCS_YCbCr) {
      cconvert->pub._color_convert = grayscale_convert;
      /* For color->grayscale conversion, only the Y (0) component is needed */
      for (ci = 1; ci < cinfo->num_components; ci++)
        cinfo->comp_info[ci].component_needed = FALSE;
    } else if (cinfo->jpeg_color_space == JCS_RGB) {
      cconvert->pub._color_convert = rgb_gray_convert;
      build_rgb_y_table(cinfo);
    } else
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
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
#ifdef D_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless && cinfo->jpeg_color_space != JCS_RGB)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    cinfo->out_color_components = rgb_pixelsize[cinfo->out_color_space];
    if (cinfo->jpeg_color_space == JCS_YCbCr) {
#ifdef WITH_SIMD
      if (jsimd_can_ycc_rgb())
        cconvert->pub._color_convert = jsimd_ycc_rgb_convert;
      else
#endif
      {
        cconvert->pub._color_convert = ycc_rgb_convert;
        build_ycc_rgb_table(cinfo);
      }
    } else if (cinfo->jpeg_color_space == JCS_GRAYSCALE) {
      cconvert->pub._color_convert = gray_rgb_convert;
    } else if (cinfo->jpeg_color_space == JCS_RGB) {
      if (rgb_red[cinfo->out_color_space] == 0 &&
          rgb_green[cinfo->out_color_space] == 1 &&
          rgb_blue[cinfo->out_color_space] == 2 &&
          rgb_pixelsize[cinfo->out_color_space] == 3)
        cconvert->pub._color_convert = null_convert;
      else
        cconvert->pub._color_convert = rgb_rgb_convert;
    } else
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    break;

  case JCS_RGB565:
#ifdef D_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    cinfo->out_color_components = 3;
    if (cinfo->dither_mode == JDITHER_NONE) {
      if (cinfo->jpeg_color_space == JCS_YCbCr) {
#ifdef WITH_SIMD
        if (jsimd_can_ycc_rgb565())
          cconvert->pub._color_convert = jsimd_ycc_rgb565_convert;
        else
#endif
        {
          cconvert->pub._color_convert = ycc_rgb565_convert;
          build_ycc_rgb_table(cinfo);
        }
      } else if (cinfo->jpeg_color_space == JCS_GRAYSCALE) {
        cconvert->pub._color_convert = gray_rgb565_convert;
      } else if (cinfo->jpeg_color_space == JCS_RGB) {
        cconvert->pub._color_convert = rgb_rgb565_convert;
      } else
        ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    } else {
      /* only ordered dithering is supported */
      if (cinfo->jpeg_color_space == JCS_YCbCr) {
        cconvert->pub._color_convert = ycc_rgb565D_convert;
        build_ycc_rgb_table(cinfo);
      } else if (cinfo->jpeg_color_space == JCS_GRAYSCALE) {
        cconvert->pub._color_convert = gray_rgb565D_convert;
      } else if (cinfo->jpeg_color_space == JCS_RGB) {
        cconvert->pub._color_convert = rgb_rgb565D_convert;
      } else
        ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    }
    break;

  case JCS_CMYK:
#ifdef D_LOSSLESS_SUPPORTED
    if (cinfo->master->lossless &&
        cinfo->jpeg_color_space != cinfo->out_color_space)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
#endif
    cinfo->out_color_components = 4;
    if (cinfo->jpeg_color_space == JCS_YCCK) {
      cconvert->pub._color_convert = ycck_cmyk_convert;
      build_ycc_rgb_table(cinfo);
    } else if (cinfo->jpeg_color_space == JCS_CMYK) {
      cconvert->pub._color_convert = null_convert;
    } else
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    break;

  default:
    /* Permit null conversion to same output space */
    if (cinfo->out_color_space == cinfo->jpeg_color_space) {
      cinfo->out_color_components = cinfo->num_components;
      cconvert->pub._color_convert = null_convert;
    } else                      /* unsupported non-null conversion */
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    break;
  }

  if (cinfo->quantize_colors)
    cinfo->output_components = 1; /* single colormapped output component */
  else
    cinfo->output_components = cinfo->out_color_components;
}

#endif /* BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED) */
