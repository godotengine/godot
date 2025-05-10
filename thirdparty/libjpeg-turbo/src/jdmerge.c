/*
 * jdmerge.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2009, 2011, 2014-2015, 2020, 2022, D. R. Commander.
 * Copyright (C) 2013, Linaro Limited.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains code for merged upsampling/color conversion.
 *
 * This file combines functions from jdsample.c and jdcolor.c;
 * read those files first to understand what's going on.
 *
 * When the chroma components are to be upsampled by simple replication
 * (ie, box filtering), we can save some work in color conversion by
 * calculating all the output pixels corresponding to a pair of chroma
 * samples at one time.  In the conversion equations
 *      R = Y           + K1 * Cr
 *      G = Y + K2 * Cb + K3 * Cr
 *      B = Y + K4 * Cb
 * only the Y term varies among the group of pixels corresponding to a pair
 * of chroma samples, so the rest of the terms can be calculated just once.
 * At typical sampling ratios, this eliminates half or three-quarters of the
 * multiplications needed for color conversion.
 *
 * This file currently provides implementations for the following cases:
 *      YCbCr => RGB color conversion only.
 *      Sampling ratios of 2h1v or 2h2v.
 *      No scaling needed at upsample time.
 *      Corner-aligned (non-CCIR601) sampling alignment.
 * Other special cases could be added, but in most applications these are
 * the only common cases.  (For uncommon cases we fall back on the more
 * general code in jdsample.c and jdcolor.c.)
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdmerge.h"
#include "jsimd.h"

#ifdef UPSAMPLE_MERGING_SUPPORTED


#define SCALEBITS       16      /* speediest right-shift on some machines */
#define ONE_HALF        ((JLONG)1 << (SCALEBITS - 1))
#define FIX(x)          ((JLONG)((x) * (1L << SCALEBITS) + 0.5))


/* Include inline routines for colorspace extensions */

#include "jdmrgext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE

#define RGB_RED  EXT_RGB_RED
#define RGB_GREEN  EXT_RGB_GREEN
#define RGB_BLUE  EXT_RGB_BLUE
#define RGB_PIXELSIZE  EXT_RGB_PIXELSIZE
#define h2v1_merged_upsample_internal  extrgb_h2v1_merged_upsample_internal
#define h2v2_merged_upsample_internal  extrgb_h2v2_merged_upsample_internal
#include "jdmrgext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef h2v1_merged_upsample_internal
#undef h2v2_merged_upsample_internal

#define RGB_RED  EXT_RGBX_RED
#define RGB_GREEN  EXT_RGBX_GREEN
#define RGB_BLUE  EXT_RGBX_BLUE
#define RGB_ALPHA  3
#define RGB_PIXELSIZE  EXT_RGBX_PIXELSIZE
#define h2v1_merged_upsample_internal  extrgbx_h2v1_merged_upsample_internal
#define h2v2_merged_upsample_internal  extrgbx_h2v2_merged_upsample_internal
#include "jdmrgext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef h2v1_merged_upsample_internal
#undef h2v2_merged_upsample_internal

#define RGB_RED  EXT_BGR_RED
#define RGB_GREEN  EXT_BGR_GREEN
#define RGB_BLUE  EXT_BGR_BLUE
#define RGB_PIXELSIZE  EXT_BGR_PIXELSIZE
#define h2v1_merged_upsample_internal  extbgr_h2v1_merged_upsample_internal
#define h2v2_merged_upsample_internal  extbgr_h2v2_merged_upsample_internal
#include "jdmrgext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_PIXELSIZE
#undef h2v1_merged_upsample_internal
#undef h2v2_merged_upsample_internal

#define RGB_RED  EXT_BGRX_RED
#define RGB_GREEN  EXT_BGRX_GREEN
#define RGB_BLUE  EXT_BGRX_BLUE
#define RGB_ALPHA  3
#define RGB_PIXELSIZE  EXT_BGRX_PIXELSIZE
#define h2v1_merged_upsample_internal  extbgrx_h2v1_merged_upsample_internal
#define h2v2_merged_upsample_internal  extbgrx_h2v2_merged_upsample_internal
#include "jdmrgext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef h2v1_merged_upsample_internal
#undef h2v2_merged_upsample_internal

#define RGB_RED  EXT_XBGR_RED
#define RGB_GREEN  EXT_XBGR_GREEN
#define RGB_BLUE  EXT_XBGR_BLUE
#define RGB_ALPHA  0
#define RGB_PIXELSIZE  EXT_XBGR_PIXELSIZE
#define h2v1_merged_upsample_internal  extxbgr_h2v1_merged_upsample_internal
#define h2v2_merged_upsample_internal  extxbgr_h2v2_merged_upsample_internal
#include "jdmrgext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef h2v1_merged_upsample_internal
#undef h2v2_merged_upsample_internal

#define RGB_RED  EXT_XRGB_RED
#define RGB_GREEN  EXT_XRGB_GREEN
#define RGB_BLUE  EXT_XRGB_BLUE
#define RGB_ALPHA  0
#define RGB_PIXELSIZE  EXT_XRGB_PIXELSIZE
#define h2v1_merged_upsample_internal  extxrgb_h2v1_merged_upsample_internal
#define h2v2_merged_upsample_internal  extxrgb_h2v2_merged_upsample_internal
#include "jdmrgext.c"
#undef RGB_RED
#undef RGB_GREEN
#undef RGB_BLUE
#undef RGB_ALPHA
#undef RGB_PIXELSIZE
#undef h2v1_merged_upsample_internal
#undef h2v2_merged_upsample_internal


/*
 * Initialize tables for YCC->RGB colorspace conversion.
 * This is taken directly from jdcolor.c; see that file for more info.
 */

LOCAL(void)
build_ycc_rgb_table(j_decompress_ptr cinfo)
{
  my_merged_upsample_ptr upsample = (my_merged_upsample_ptr)cinfo->upsample;
  int i;
  JLONG x;
  SHIFT_TEMPS

  upsample->Cr_r_tab = (int *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (_MAXJSAMPLE + 1) * sizeof(int));
  upsample->Cb_b_tab = (int *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (_MAXJSAMPLE + 1) * sizeof(int));
  upsample->Cr_g_tab = (JLONG *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (_MAXJSAMPLE + 1) * sizeof(JLONG));
  upsample->Cb_g_tab = (JLONG *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                (_MAXJSAMPLE + 1) * sizeof(JLONG));

  for (i = 0, x = -_CENTERJSAMPLE; i <= _MAXJSAMPLE; i++, x++) {
    /* i is the actual input pixel value, in the range 0.._MAXJSAMPLE */
    /* The Cb or Cr value we are thinking of is x = i - _CENTERJSAMPLE */
    /* Cr=>R value is nearest int to 1.40200 * x */
    upsample->Cr_r_tab[i] = (int)
                    RIGHT_SHIFT(FIX(1.40200) * x + ONE_HALF, SCALEBITS);
    /* Cb=>B value is nearest int to 1.77200 * x */
    upsample->Cb_b_tab[i] = (int)
                    RIGHT_SHIFT(FIX(1.77200) * x + ONE_HALF, SCALEBITS);
    /* Cr=>G value is scaled-up -0.71414 * x */
    upsample->Cr_g_tab[i] = (-FIX(0.71414)) * x;
    /* Cb=>G value is scaled-up -0.34414 * x */
    /* We also add in ONE_HALF so that need not do it in inner loop */
    upsample->Cb_g_tab[i] = (-FIX(0.34414)) * x + ONE_HALF;
  }
}


/*
 * Initialize for an upsampling pass.
 */

METHODDEF(void)
start_pass_merged_upsample(j_decompress_ptr cinfo)
{
  my_merged_upsample_ptr upsample = (my_merged_upsample_ptr)cinfo->upsample;

  /* Mark the spare buffer empty */
  upsample->spare_full = FALSE;
  /* Initialize total-height counter for detecting bottom of image */
  upsample->rows_to_go = cinfo->output_height;
}


/*
 * Control routine to do upsampling (and color conversion).
 *
 * The control routine just handles the row buffering considerations.
 */

METHODDEF(void)
merged_2v_upsample(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                   JDIMENSION *in_row_group_ctr,
                   JDIMENSION in_row_groups_avail, _JSAMPARRAY output_buf,
                   JDIMENSION *out_row_ctr, JDIMENSION out_rows_avail)
/* 2:1 vertical sampling case: may need a spare row. */
{
  my_merged_upsample_ptr upsample = (my_merged_upsample_ptr)cinfo->upsample;
  _JSAMPROW work_ptrs[2];
  JDIMENSION num_rows;          /* number of rows returned to caller */

  if (upsample->spare_full) {
    /* If we have a spare row saved from a previous cycle, just return it. */
    JDIMENSION size = upsample->out_row_width;
    if (cinfo->out_color_space == JCS_RGB565)
      size = cinfo->output_width * 2;
    _jcopy_sample_rows(&upsample->spare_row, 0, output_buf + *out_row_ctr, 0,
                       1, size);
    num_rows = 1;
    upsample->spare_full = FALSE;
  } else {
    /* Figure number of rows to return to caller. */
    num_rows = 2;
    /* Not more than the distance to the end of the image. */
    if (num_rows > upsample->rows_to_go)
      num_rows = upsample->rows_to_go;
    /* And not more than what the client can accept: */
    out_rows_avail -= *out_row_ctr;
    if (num_rows > out_rows_avail)
      num_rows = out_rows_avail;
    /* Create output pointer array for upsampler. */
    work_ptrs[0] = output_buf[*out_row_ctr];
    if (num_rows > 1) {
      work_ptrs[1] = output_buf[*out_row_ctr + 1];
    } else {
      work_ptrs[1] = upsample->spare_row;
      upsample->spare_full = TRUE;
    }
    /* Now do the upsampling. */
    (*upsample->upmethod) (cinfo, input_buf, *in_row_group_ctr, work_ptrs);
  }

  /* Adjust counts */
  *out_row_ctr += num_rows;
  upsample->rows_to_go -= num_rows;
  /* When the buffer is emptied, declare this input row group consumed */
  if (!upsample->spare_full)
    (*in_row_group_ctr)++;
}


METHODDEF(void)
merged_1v_upsample(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                   JDIMENSION *in_row_group_ctr,
                   JDIMENSION in_row_groups_avail, _JSAMPARRAY output_buf,
                   JDIMENSION *out_row_ctr, JDIMENSION out_rows_avail)
/* 1:1 vertical sampling case: much easier, never need a spare row. */
{
  my_merged_upsample_ptr upsample = (my_merged_upsample_ptr)cinfo->upsample;

  /* Just do the upsampling. */
  (*upsample->upmethod) (cinfo, input_buf, *in_row_group_ctr,
                         output_buf + *out_row_ctr);
  /* Adjust counts */
  (*out_row_ctr)++;
  (*in_row_group_ctr)++;
}


/*
 * These are the routines invoked by the control routines to do
 * the actual upsampling/conversion.  One row group is processed per call.
 *
 * Note: since we may be writing directly into application-supplied buffers,
 * we have to be honest about the output width; we can't assume the buffer
 * has been rounded up to an even width.
 */


/*
 * Upsample and color convert for the case of 2:1 horizontal and 1:1 vertical.
 */

METHODDEF(void)
h2v1_merged_upsample(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                     JDIMENSION in_row_group_ctr, _JSAMPARRAY output_buf)
{
  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    extrgb_h2v1_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                         output_buf);
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    extrgbx_h2v1_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                          output_buf);
    break;
  case JCS_EXT_BGR:
    extbgr_h2v1_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                         output_buf);
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    extbgrx_h2v1_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                          output_buf);
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    extxbgr_h2v1_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                          output_buf);
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    extxrgb_h2v1_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                          output_buf);
    break;
  default:
    h2v1_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                  output_buf);
    break;
  }
}


/*
 * Upsample and color convert for the case of 2:1 horizontal and 2:1 vertical.
 */

METHODDEF(void)
h2v2_merged_upsample(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                     JDIMENSION in_row_group_ctr, _JSAMPARRAY output_buf)
{
  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
    extrgb_h2v2_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                         output_buf);
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    extrgbx_h2v2_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                          output_buf);
    break;
  case JCS_EXT_BGR:
    extbgr_h2v2_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                         output_buf);
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    extbgrx_h2v2_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                          output_buf);
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    extxbgr_h2v2_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                          output_buf);
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    extxrgb_h2v2_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                          output_buf);
    break;
  default:
    h2v2_merged_upsample_internal(cinfo, input_buf, in_row_group_ctr,
                                  output_buf);
    break;
  }
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

#define WRITE_TWO_PIXELS_LE(addr, pixels) { \
  ((INT16 *)(addr))[0] = (INT16)(pixels); \
  ((INT16 *)(addr))[1] = (INT16)((pixels) >> 16); \
}
#define WRITE_TWO_PIXELS_BE(addr, pixels) { \
  ((INT16 *)(addr))[1] = (INT16)(pixels); \
  ((INT16 *)(addr))[0] = (INT16)((pixels) >> 16); \
}

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


/* Include inline routines for RGB565 conversion */

#define PACK_SHORT_565  PACK_SHORT_565_LE
#define PACK_TWO_PIXELS  PACK_TWO_PIXELS_LE
#define WRITE_TWO_PIXELS  WRITE_TWO_PIXELS_LE
#define h2v1_merged_upsample_565_internal  h2v1_merged_upsample_565_le
#define h2v1_merged_upsample_565D_internal  h2v1_merged_upsample_565D_le
#define h2v2_merged_upsample_565_internal  h2v2_merged_upsample_565_le
#define h2v2_merged_upsample_565D_internal  h2v2_merged_upsample_565D_le
#include "jdmrg565.c"
#undef PACK_SHORT_565
#undef PACK_TWO_PIXELS
#undef WRITE_TWO_PIXELS
#undef h2v1_merged_upsample_565_internal
#undef h2v1_merged_upsample_565D_internal
#undef h2v2_merged_upsample_565_internal
#undef h2v2_merged_upsample_565D_internal

#define PACK_SHORT_565  PACK_SHORT_565_BE
#define PACK_TWO_PIXELS  PACK_TWO_PIXELS_BE
#define WRITE_TWO_PIXELS  WRITE_TWO_PIXELS_BE
#define h2v1_merged_upsample_565_internal  h2v1_merged_upsample_565_be
#define h2v1_merged_upsample_565D_internal  h2v1_merged_upsample_565D_be
#define h2v2_merged_upsample_565_internal  h2v2_merged_upsample_565_be
#define h2v2_merged_upsample_565D_internal  h2v2_merged_upsample_565D_be
#include "jdmrg565.c"
#undef PACK_SHORT_565
#undef PACK_TWO_PIXELS
#undef WRITE_TWO_PIXELS
#undef h2v1_merged_upsample_565_internal
#undef h2v1_merged_upsample_565D_internal
#undef h2v2_merged_upsample_565_internal
#undef h2v2_merged_upsample_565D_internal


static INLINE boolean is_big_endian(void)
{
  int test_value = 1;
  if (*(char *)&test_value != 1)
    return TRUE;
  return FALSE;
}


METHODDEF(void)
h2v1_merged_upsample_565(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                         JDIMENSION in_row_group_ctr, _JSAMPARRAY output_buf)
{
  if (is_big_endian())
    h2v1_merged_upsample_565_be(cinfo, input_buf, in_row_group_ctr,
                                output_buf);
  else
    h2v1_merged_upsample_565_le(cinfo, input_buf, in_row_group_ctr,
                                output_buf);
}


METHODDEF(void)
h2v1_merged_upsample_565D(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                          JDIMENSION in_row_group_ctr, _JSAMPARRAY output_buf)
{
  if (is_big_endian())
    h2v1_merged_upsample_565D_be(cinfo, input_buf, in_row_group_ctr,
                                 output_buf);
  else
    h2v1_merged_upsample_565D_le(cinfo, input_buf, in_row_group_ctr,
                                 output_buf);
}


METHODDEF(void)
h2v2_merged_upsample_565(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                         JDIMENSION in_row_group_ctr, _JSAMPARRAY output_buf)
{
  if (is_big_endian())
    h2v2_merged_upsample_565_be(cinfo, input_buf, in_row_group_ctr,
                                output_buf);
  else
    h2v2_merged_upsample_565_le(cinfo, input_buf, in_row_group_ctr,
                                output_buf);
}


METHODDEF(void)
h2v2_merged_upsample_565D(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                          JDIMENSION in_row_group_ctr, _JSAMPARRAY output_buf)
{
  if (is_big_endian())
    h2v2_merged_upsample_565D_be(cinfo, input_buf, in_row_group_ctr,
                                 output_buf);
  else
    h2v2_merged_upsample_565D_le(cinfo, input_buf, in_row_group_ctr,
                                 output_buf);
}


/*
 * Module initialization routine for merged upsampling/color conversion.
 *
 * NB: this is called under the conditions determined by use_merged_upsample()
 * in jdmaster.c.  That routine MUST correspond to the actual capabilities
 * of this module; no safety checks are made here.
 */

GLOBAL(void)
_jinit_merged_upsampler(j_decompress_ptr cinfo)
{
  my_merged_upsample_ptr upsample;

  if (cinfo->data_precision != BITS_IN_JSAMPLE)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  upsample = (my_merged_upsample_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(my_merged_upsampler));
  cinfo->upsample = (struct jpeg_upsampler *)upsample;
  upsample->pub.start_pass = start_pass_merged_upsample;
  upsample->pub.need_context_rows = FALSE;

  upsample->out_row_width = cinfo->output_width * cinfo->out_color_components;

  if (cinfo->max_v_samp_factor == 2) {
    upsample->pub._upsample = merged_2v_upsample;
#ifdef WITH_SIMD
    if (jsimd_can_h2v2_merged_upsample())
      upsample->upmethod = jsimd_h2v2_merged_upsample;
    else
#endif
      upsample->upmethod = h2v2_merged_upsample;
    if (cinfo->out_color_space == JCS_RGB565) {
      if (cinfo->dither_mode != JDITHER_NONE) {
        upsample->upmethod = h2v2_merged_upsample_565D;
      } else {
        upsample->upmethod = h2v2_merged_upsample_565;
      }
    }
    /* Allocate a spare row buffer */
    upsample->spare_row = (_JSAMPROW)
      (*cinfo->mem->alloc_large) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                (size_t)(upsample->out_row_width * sizeof(_JSAMPLE)));
  } else {
    upsample->pub._upsample = merged_1v_upsample;
#ifdef WITH_SIMD
    if (jsimd_can_h2v1_merged_upsample())
      upsample->upmethod = jsimd_h2v1_merged_upsample;
    else
#endif
      upsample->upmethod = h2v1_merged_upsample;
    if (cinfo->out_color_space == JCS_RGB565) {
      if (cinfo->dither_mode != JDITHER_NONE) {
        upsample->upmethod = h2v1_merged_upsample_565D;
      } else {
        upsample->upmethod = h2v1_merged_upsample_565;
      }
    }
    /* No spare row needed */
    upsample->spare_row = NULL;
  }

  build_ycc_rgb_table(cinfo);
}

#endif /* UPSAMPLE_MERGING_SUPPORTED */
