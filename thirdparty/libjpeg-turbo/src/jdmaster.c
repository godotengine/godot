/*
 * jdmaster.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2002-2009 by Guido Vollbeding.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2009-2011, 2016, 2019, 2022-2024, D. R. Commander.
 * Copyright (C) 2013, Linaro Limited.
 * Copyright (C) 2015, Google, Inc.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains master control logic for the JPEG decompressor.
 * These routines are concerned with selecting the modules to be executed
 * and with determining the number of passes and the work to be done in each
 * pass.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jpegapicomp.h"
#include "jdmaster.h"


/*
 * Determine whether merged upsample/color conversion should be used.
 * CRUCIAL: this must match the actual capabilities of jdmerge.c!
 */

LOCAL(boolean)
use_merged_upsample(j_decompress_ptr cinfo)
{
#ifdef UPSAMPLE_MERGING_SUPPORTED
  /* Colorspace conversion is not supported with lossless JPEG images */
  if (cinfo->master->lossless)
    return FALSE;
  /* Merging is the equivalent of plain box-filter upsampling */
  if (cinfo->do_fancy_upsampling || cinfo->CCIR601_sampling)
    return FALSE;
  /* jdmerge.c only supports YCC=>RGB and YCC=>RGB565 color conversion */
  if (cinfo->jpeg_color_space != JCS_YCbCr || cinfo->num_components != 3 ||
      (cinfo->out_color_space != JCS_RGB &&
       cinfo->out_color_space != JCS_RGB565 &&
       cinfo->out_color_space != JCS_EXT_RGB &&
       cinfo->out_color_space != JCS_EXT_RGBX &&
       cinfo->out_color_space != JCS_EXT_BGR &&
       cinfo->out_color_space != JCS_EXT_BGRX &&
       cinfo->out_color_space != JCS_EXT_XBGR &&
       cinfo->out_color_space != JCS_EXT_XRGB &&
       cinfo->out_color_space != JCS_EXT_RGBA &&
       cinfo->out_color_space != JCS_EXT_BGRA &&
       cinfo->out_color_space != JCS_EXT_ABGR &&
       cinfo->out_color_space != JCS_EXT_ARGB))
    return FALSE;
  if ((cinfo->out_color_space == JCS_RGB565 &&
       cinfo->out_color_components != 3) ||
      (cinfo->out_color_space != JCS_RGB565 &&
       cinfo->out_color_components != rgb_pixelsize[cinfo->out_color_space]))
    return FALSE;
  /* and it only handles 2h1v or 2h2v sampling ratios */
  if (cinfo->comp_info[0].h_samp_factor != 2 ||
      cinfo->comp_info[1].h_samp_factor != 1 ||
      cinfo->comp_info[2].h_samp_factor != 1 ||
      cinfo->comp_info[0].v_samp_factor >  2 ||
      cinfo->comp_info[1].v_samp_factor != 1 ||
      cinfo->comp_info[2].v_samp_factor != 1)
    return FALSE;
  /* furthermore, it doesn't work if we've scaled the IDCTs differently */
  if (cinfo->comp_info[0]._DCT_scaled_size != cinfo->_min_DCT_scaled_size ||
      cinfo->comp_info[1]._DCT_scaled_size != cinfo->_min_DCT_scaled_size ||
      cinfo->comp_info[2]._DCT_scaled_size != cinfo->_min_DCT_scaled_size)
    return FALSE;
  /* ??? also need to test for upsample-time rescaling, when & if supported */
  return TRUE;                  /* by golly, it'll work... */
#else
  return FALSE;
#endif
}


/*
 * Compute output image dimensions and related values.
 * NOTE: this is exported for possible use by application.
 * Hence it mustn't do anything that can't be done twice.
 */

#if JPEG_LIB_VERSION >= 80
GLOBAL(void)
#else
LOCAL(void)
#endif
jpeg_core_output_dimensions(j_decompress_ptr cinfo)
/* Do computations that are needed before master selection phase.
 * This function is used for transcoding and full decompression.
 */
{
#ifdef IDCT_SCALING_SUPPORTED
  int ci;
  jpeg_component_info *compptr;

  if (!cinfo->master->lossless) {
    /* Compute actual output image dimensions and DCT scaling choices. */
    if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom) {
      /* Provide 1/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 1;
      cinfo->_min_DCT_v_scaled_size = 1;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 2) {
      /* Provide 2/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 2L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 2L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 2;
      cinfo->_min_DCT_v_scaled_size = 2;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 3) {
      /* Provide 3/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 3L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 3L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 3;
      cinfo->_min_DCT_v_scaled_size = 3;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 4) {
      /* Provide 4/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 4L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 4L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 4;
      cinfo->_min_DCT_v_scaled_size = 4;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 5) {
      /* Provide 5/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 5L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 5L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 5;
      cinfo->_min_DCT_v_scaled_size = 5;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 6) {
      /* Provide 6/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 6L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 6L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 6;
      cinfo->_min_DCT_v_scaled_size = 6;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 7) {
      /* Provide 7/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 7L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 7L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 7;
      cinfo->_min_DCT_v_scaled_size = 7;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 8) {
      /* Provide 8/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 8L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 8L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 8;
      cinfo->_min_DCT_v_scaled_size = 8;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 9) {
      /* Provide 9/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 9L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 9L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 9;
      cinfo->_min_DCT_v_scaled_size = 9;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 10) {
      /* Provide 10/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 10L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 10L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 10;
      cinfo->_min_DCT_v_scaled_size = 10;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 11) {
      /* Provide 11/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 11L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 11L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 11;
      cinfo->_min_DCT_v_scaled_size = 11;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 12) {
      /* Provide 12/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 12L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 12L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 12;
      cinfo->_min_DCT_v_scaled_size = 12;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 13) {
      /* Provide 13/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 13L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 13L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 13;
      cinfo->_min_DCT_v_scaled_size = 13;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 14) {
      /* Provide 14/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 14L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 14L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 14;
      cinfo->_min_DCT_v_scaled_size = 14;
    } else if (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * 15) {
      /* Provide 15/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 15L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 15L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 15;
      cinfo->_min_DCT_v_scaled_size = 15;
    } else {
      /* Provide 16/block_size scaling */
      cinfo->output_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width * 16L, (long)DCTSIZE);
      cinfo->output_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height * 16L, (long)DCTSIZE);
      cinfo->_min_DCT_h_scaled_size = 16;
      cinfo->_min_DCT_v_scaled_size = 16;
    }

    /* Recompute dimensions of components */
    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
         ci++, compptr++) {
      compptr->_DCT_h_scaled_size = cinfo->_min_DCT_h_scaled_size;
      compptr->_DCT_v_scaled_size = cinfo->_min_DCT_v_scaled_size;
    }
  } else
#endif /* !IDCT_SCALING_SUPPORTED */
  {
    /* Hardwire it to "no scaling" */
    cinfo->output_width = cinfo->image_width;
    cinfo->output_height = cinfo->image_height;
    /* jdinput.c has already initialized DCT_scaled_size,
     * and has computed unscaled downsampled_width and downsampled_height.
     */
  }
}


/*
 * Compute output image dimensions and related values.
 * NOTE: this is exported for possible use by application.
 * Hence it mustn't do anything that can't be done twice.
 * Also note that it may be called before the master module is initialized!
 */

GLOBAL(void)
jpeg_calc_output_dimensions(j_decompress_ptr cinfo)
/* Do computations that are needed before master selection phase */
{
#ifdef IDCT_SCALING_SUPPORTED
  int ci;
  jpeg_component_info *compptr;
#endif

  /* Prevent application from calling me at wrong times */
  if (cinfo->global_state != DSTATE_READY)
    ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);

  /* Compute core output image dimensions and DCT scaling choices. */
  jpeg_core_output_dimensions(cinfo);

#ifdef IDCT_SCALING_SUPPORTED

  if (!cinfo->master->lossless) {
    /* In selecting the actual DCT scaling for each component, we try to
     * scale up the chroma components via IDCT scaling rather than upsampling.
     * This saves time if the upsampler gets to use 1:1 scaling.
     * Note this code adapts subsampling ratios which are powers of 2.
     */
    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
         ci++, compptr++) {
      int ssize = cinfo->_min_DCT_scaled_size;
      while (ssize < DCTSIZE &&
             ((cinfo->max_h_samp_factor * cinfo->_min_DCT_scaled_size) %
              (compptr->h_samp_factor * ssize * 2) == 0) &&
             ((cinfo->max_v_samp_factor * cinfo->_min_DCT_scaled_size) %
              (compptr->v_samp_factor * ssize * 2) == 0)) {
        ssize = ssize * 2;
      }
#if JPEG_LIB_VERSION >= 70
      compptr->DCT_h_scaled_size = compptr->DCT_v_scaled_size = ssize;
#else
      compptr->DCT_scaled_size = ssize;
#endif
    }

    /* Recompute downsampled dimensions of components;
     * application needs to know these if using raw downsampled data.
     */
    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
         ci++, compptr++) {
      /* Size in samples, after IDCT scaling */
      compptr->downsampled_width = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_width *
                      (long)(compptr->h_samp_factor *
                             compptr->_DCT_scaled_size),
                      (long)(cinfo->max_h_samp_factor * DCTSIZE));
      compptr->downsampled_height = (JDIMENSION)
        jdiv_round_up((long)cinfo->image_height *
                      (long)(compptr->v_samp_factor *
                             compptr->_DCT_scaled_size),
                      (long)(cinfo->max_v_samp_factor * DCTSIZE));
    }
  } else
#endif /* IDCT_SCALING_SUPPORTED */
  {
    /* Hardwire it to "no scaling" */
    cinfo->output_width = cinfo->image_width;
    cinfo->output_height = cinfo->image_height;
    /* jdinput.c has already initialized DCT_scaled_size to DCTSIZE,
     * and has computed unscaled downsampled_width and downsampled_height.
     */
  }

  /* Report number of components in selected colorspace. */
  /* Probably this should be in the color conversion module... */
  switch (cinfo->out_color_space) {
  case JCS_GRAYSCALE:
    cinfo->out_color_components = 1;
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
    cinfo->out_color_components = rgb_pixelsize[cinfo->out_color_space];
    break;
  case JCS_YCbCr:
  case JCS_RGB565:
    cinfo->out_color_components = 3;
    break;
  case JCS_CMYK:
  case JCS_YCCK:
    cinfo->out_color_components = 4;
    break;
  default:                      /* else must be same colorspace as in file */
    cinfo->out_color_components = cinfo->num_components;
    break;
  }
  cinfo->output_components = (cinfo->quantize_colors ? 1 :
                              cinfo->out_color_components);

  /* See if upsampler will want to emit more than one row at a time */
  if (use_merged_upsample(cinfo))
    cinfo->rec_outbuf_height = cinfo->max_v_samp_factor;
  else
    cinfo->rec_outbuf_height = 1;
}


/*
 * Several decompression processes need to range-limit values to the range
 * 0..MAXJSAMPLE; the input value may fall somewhat outside this range
 * due to noise introduced by quantization, roundoff error, etc.  These
 * processes are inner loops and need to be as fast as possible.  On most
 * machines, particularly CPUs with pipelines or instruction prefetch,
 * a (subscript-check-less) C table lookup
 *              x = sample_range_limit[x];
 * is faster than explicit tests
 *              if (x < 0)  x = 0;
 *              else if (x > MAXJSAMPLE)  x = MAXJSAMPLE;
 * These processes all use a common table prepared by the routine below.
 *
 * For most steps we can mathematically guarantee that the initial value
 * of x is within MAXJSAMPLE+1 of the legal range, so a table running from
 * -(MAXJSAMPLE+1) to 2*MAXJSAMPLE+1 is sufficient.  But for the initial
 * limiting step (just after the IDCT), a wildly out-of-range value is
 * possible if the input data is corrupt.  To avoid any chance of indexing
 * off the end of memory and getting a bad-pointer trap, we perform the
 * post-IDCT limiting thus:
 *              x = range_limit[x & MASK];
 * where MASK is 2 bits wider than legal sample data, ie 10 bits for 8-bit
 * samples.  Under normal circumstances this is more than enough range and
 * a correct output will be generated; with bogus input data the mask will
 * cause wraparound, and we will safely generate a bogus-but-in-range output.
 * For the post-IDCT step, we want to convert the data from signed to unsigned
 * representation by adding CENTERJSAMPLE at the same time that we limit it.
 * So the post-IDCT limiting table ends up looking like this:
 *   CENTERJSAMPLE,CENTERJSAMPLE+1,...,MAXJSAMPLE,
 *   MAXJSAMPLE (repeat 2*(MAXJSAMPLE+1)-CENTERJSAMPLE times),
 *   0          (repeat 2*(MAXJSAMPLE+1)-CENTERJSAMPLE times),
 *   0,1,...,CENTERJSAMPLE-1
 * Negative inputs select values from the upper half of the table after
 * masking.
 *
 * We can save some space by overlapping the start of the post-IDCT table
 * with the simpler range limiting table.  The post-IDCT table begins at
 * sample_range_limit + CENTERJSAMPLE.
 */

LOCAL(void)
prepare_range_limit_table(j_decompress_ptr cinfo)
/* Allocate and fill in the sample_range_limit table */
{
  JSAMPLE *table;
  J12SAMPLE *table12;
#ifdef D_LOSSLESS_SUPPORTED
  J16SAMPLE *table16;
#endif
  int i;

  if (cinfo->data_precision <= 8) {
    table = (JSAMPLE *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                  (5 * (MAXJSAMPLE + 1) + CENTERJSAMPLE) * sizeof(JSAMPLE));
    table += (MAXJSAMPLE + 1);  /* allow negative subscripts of simple table */
    cinfo->sample_range_limit = table;
    /* First segment of "simple" table: limit[x] = 0 for x < 0 */
    memset(table - (MAXJSAMPLE + 1), 0, (MAXJSAMPLE + 1) * sizeof(JSAMPLE));
    /* Main part of "simple" table: limit[x] = x */
    for (i = 0; i <= MAXJSAMPLE; i++)
      table[i] = (JSAMPLE)i;
    table += CENTERJSAMPLE;     /* Point to where post-IDCT table starts */
    /* End of simple table, rest of first half of post-IDCT table */
    for (i = CENTERJSAMPLE; i < 2 * (MAXJSAMPLE + 1); i++)
      table[i] = MAXJSAMPLE;
    /* Second half of post-IDCT table */
    memset(table + (2 * (MAXJSAMPLE + 1)), 0,
           (2 * (MAXJSAMPLE + 1) - CENTERJSAMPLE) * sizeof(JSAMPLE));
    memcpy(table + (4 * (MAXJSAMPLE + 1) - CENTERJSAMPLE),
           cinfo->sample_range_limit, CENTERJSAMPLE * sizeof(JSAMPLE));
  } else if (cinfo->data_precision <= 12) {
    table12 = (J12SAMPLE *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                  (5 * (MAXJ12SAMPLE + 1) + CENTERJ12SAMPLE) *
                  sizeof(J12SAMPLE));
    table12 += (MAXJ12SAMPLE + 1);  /* allow negative subscripts of simple
                                       table */
    cinfo->sample_range_limit = (JSAMPLE *)table12;
    /* First segment of "simple" table: limit[x] = 0 for x < 0 */
    memset(table12 - (MAXJ12SAMPLE + 1), 0,
           (MAXJ12SAMPLE + 1) * sizeof(J12SAMPLE));
    /* Main part of "simple" table: limit[x] = x */
    for (i = 0; i <= MAXJ12SAMPLE; i++)
      table12[i] = (J12SAMPLE)i;
    table12 += CENTERJ12SAMPLE; /* Point to where post-IDCT table starts */
    /* End of simple table, rest of first half of post-IDCT table */
    for (i = CENTERJ12SAMPLE; i < 2 * (MAXJ12SAMPLE + 1); i++)
      table12[i] = MAXJ12SAMPLE;
    /* Second half of post-IDCT table */
    memset(table12 + (2 * (MAXJ12SAMPLE + 1)), 0,
           (2 * (MAXJ12SAMPLE + 1) - CENTERJ12SAMPLE) * sizeof(J12SAMPLE));
    memcpy(table12 + (4 * (MAXJ12SAMPLE + 1) - CENTERJ12SAMPLE),
           cinfo->sample_range_limit, CENTERJ12SAMPLE * sizeof(J12SAMPLE));
  } else {
#ifdef D_LOSSLESS_SUPPORTED
    table16 = (J16SAMPLE *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                  (5 * (MAXJ16SAMPLE + 1) + CENTERJ16SAMPLE) *
                  sizeof(J16SAMPLE));
    table16 += (MAXJ16SAMPLE + 1);  /* allow negative subscripts of simple
                                       table */
    cinfo->sample_range_limit = (JSAMPLE *)table16;
    /* First segment of "simple" table: limit[x] = 0 for x < 0 */
    memset(table16 - (MAXJ16SAMPLE + 1), 0,
           (MAXJ16SAMPLE + 1) * sizeof(J16SAMPLE));
    /* Main part of "simple" table: limit[x] = x */
    for (i = 0; i <= MAXJ16SAMPLE; i++)
      table16[i] = (J16SAMPLE)i;
    table16 += CENTERJ16SAMPLE; /* Point to where post-IDCT table starts */
    /* End of simple table, rest of first half of post-IDCT table */
    for (i = CENTERJ16SAMPLE; i < 2 * (MAXJ16SAMPLE + 1); i++)
      table16[i] = MAXJ16SAMPLE;
    /* Second half of post-IDCT table */
    memset(table16 + (2 * (MAXJ16SAMPLE + 1)), 0,
           (2 * (MAXJ16SAMPLE + 1) - CENTERJ16SAMPLE) * sizeof(J16SAMPLE));
    memcpy(table16 + (4 * (MAXJ16SAMPLE + 1) - CENTERJ16SAMPLE),
           cinfo->sample_range_limit, CENTERJ16SAMPLE * sizeof(J16SAMPLE));
#else
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
#endif
  }
}


/*
 * Master selection of decompression modules.
 * This is done once at jpeg_start_decompress time.  We determine
 * which modules will be used and give them appropriate initialization calls.
 * We also initialize the decompressor input side to begin consuming data.
 *
 * Since jpeg_read_header has finished, we know what is in the SOF
 * and (first) SOS markers.  We also have all the application parameter
 * settings.
 */

LOCAL(void)
master_selection(j_decompress_ptr cinfo)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;
  boolean use_c_buffer;
  long samplesperrow;
  JDIMENSION jd_samplesperrow;

  /* Disable IDCT scaling and raw (downsampled) data output in lossless mode.
   * IDCT scaling is not useful in lossless mode, and it must be disabled in
   * order to properly calculate the output dimensions.  Raw data output isn't
   * particularly useful without subsampling and has not been tested in
   * lossless mode.
   */
#ifdef D_LOSSLESS_SUPPORTED
  if (cinfo->master->lossless) {
    cinfo->raw_data_out = FALSE;
    cinfo->scale_num = cinfo->scale_denom = 1;
  }
#endif

  /* Initialize dimensions and other stuff */
  jpeg_calc_output_dimensions(cinfo);
  prepare_range_limit_table(cinfo);

  /* Width of an output scanline must be representable as JDIMENSION. */
  samplesperrow = (long)cinfo->output_width *
                  (long)cinfo->out_color_components;
  jd_samplesperrow = (JDIMENSION)samplesperrow;
  if ((long)jd_samplesperrow != samplesperrow)
    ERREXIT(cinfo, JERR_WIDTH_OVERFLOW);

  /* Initialize my private state */
  master->pass_number = 0;
  master->using_merged_upsample = use_merged_upsample(cinfo);

  /* Color quantizer selection */
  master->quantizer_1pass = NULL;
  master->quantizer_2pass = NULL;
  /* No mode changes if not using buffered-image mode. */
  if (!cinfo->quantize_colors || !cinfo->buffered_image) {
    cinfo->enable_1pass_quant = FALSE;
    cinfo->enable_external_quant = FALSE;
    cinfo->enable_2pass_quant = FALSE;
  }
  if (cinfo->quantize_colors) {
    if (cinfo->raw_data_out)
      ERREXIT(cinfo, JERR_NOTIMPL);
    /* 2-pass quantizer only works in 3-component color space. */
    if (cinfo->out_color_components != 3 ||
        cinfo->out_color_space == JCS_RGB565) {
      cinfo->enable_1pass_quant = TRUE;
      cinfo->enable_external_quant = FALSE;
      cinfo->enable_2pass_quant = FALSE;
      cinfo->colormap = NULL;
    } else if (cinfo->colormap != NULL) {
      cinfo->enable_external_quant = TRUE;
    } else if (cinfo->two_pass_quantize) {
      cinfo->enable_2pass_quant = TRUE;
    } else {
      cinfo->enable_1pass_quant = TRUE;
    }

    if (cinfo->enable_1pass_quant) {
#ifdef QUANT_1PASS_SUPPORTED
      if (cinfo->data_precision == 8)
        jinit_1pass_quantizer(cinfo);
      else if (cinfo->data_precision == 12)
        j12init_1pass_quantizer(cinfo);
      else
        ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
      master->quantizer_1pass = cinfo->cquantize;
#else
      ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
    }

    /* We use the 2-pass code to map to external colormaps. */
    if (cinfo->enable_2pass_quant || cinfo->enable_external_quant) {
#ifdef QUANT_2PASS_SUPPORTED
      if (cinfo->data_precision == 8)
        jinit_2pass_quantizer(cinfo);
      else if (cinfo->data_precision == 12)
        j12init_2pass_quantizer(cinfo);
      else
        ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
      master->quantizer_2pass = cinfo->cquantize;
#else
      ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
    }
    /* If both quantizers are initialized, the 2-pass one is left active;
     * this is necessary for starting with quantization to an external map.
     */
  }

  /* Post-processing: in particular, color conversion first */
  if (!cinfo->raw_data_out) {
    if (master->using_merged_upsample) {
#ifdef UPSAMPLE_MERGING_SUPPORTED
      if (cinfo->data_precision == 8)
        jinit_merged_upsampler(cinfo); /* does color conversion too */
      else if (cinfo->data_precision == 12)
        j12init_merged_upsampler(cinfo); /* does color conversion too */
      else
        ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
#else
      ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
    } else {
      if (cinfo->data_precision <= 8) {
        jinit_color_deconverter(cinfo);
        jinit_upsampler(cinfo);
      } else if (cinfo->data_precision <= 12) {
        j12init_color_deconverter(cinfo);
        j12init_upsampler(cinfo);
      } else {
#ifdef D_LOSSLESS_SUPPORTED
        j16init_color_deconverter(cinfo);
        j16init_upsampler(cinfo);
#else
        ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
#endif
      }
    }
    if (cinfo->data_precision <= 8)
      jinit_d_post_controller(cinfo, cinfo->enable_2pass_quant);
    else if (cinfo->data_precision <= 12)
      j12init_d_post_controller(cinfo, cinfo->enable_2pass_quant);
    else
#ifdef D_LOSSLESS_SUPPORTED
      j16init_d_post_controller(cinfo, cinfo->enable_2pass_quant);
#else
      ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
#endif
  }

  if (cinfo->master->lossless) {
#ifdef D_LOSSLESS_SUPPORTED
    /* Prediction, sample undifferencing, point transform, and sample size
     * scaling
     */
    if (cinfo->data_precision <= 8)
      jinit_lossless_decompressor(cinfo);
    else if (cinfo->data_precision <= 12)
      j12init_lossless_decompressor(cinfo);
    else
      j16init_lossless_decompressor(cinfo);
    /* Entropy decoding: either Huffman or arithmetic coding. */
    if (cinfo->arith_code) {
      ERREXIT(cinfo, JERR_ARITH_NOTIMPL);
    } else {
      jinit_lhuff_decoder(cinfo);
    }

    /* Initialize principal buffer controllers. */
    use_c_buffer = cinfo->inputctl->has_multiple_scans ||
                   cinfo->buffered_image;
    if (cinfo->data_precision <= 8)
      jinit_d_diff_controller(cinfo, use_c_buffer);
    else if (cinfo->data_precision <= 12)
      j12init_d_diff_controller(cinfo, use_c_buffer);
    else
      j16init_d_diff_controller(cinfo, use_c_buffer);
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
  } else {
    /* Inverse DCT */
    if (cinfo->data_precision == 8)
      jinit_inverse_dct(cinfo);
    else if (cinfo->data_precision == 12)
      j12init_inverse_dct(cinfo);
    else
      ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
    /* Entropy decoding: either Huffman or arithmetic coding. */
    if (cinfo->arith_code) {
#ifdef D_ARITH_CODING_SUPPORTED
      jinit_arith_decoder(cinfo);
#else
      ERREXIT(cinfo, JERR_ARITH_NOTIMPL);
#endif
    } else {
      if (cinfo->progressive_mode) {
#ifdef D_PROGRESSIVE_SUPPORTED
        jinit_phuff_decoder(cinfo);
#else
        ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
      } else
        jinit_huff_decoder(cinfo);
    }

    /* Initialize principal buffer controllers. */
    use_c_buffer = cinfo->inputctl->has_multiple_scans ||
                   cinfo->buffered_image;
    if (cinfo->data_precision == 12)
      j12init_d_coef_controller(cinfo, use_c_buffer);
    else
      jinit_d_coef_controller(cinfo, use_c_buffer);
  }

  if (!cinfo->raw_data_out) {
    if (cinfo->data_precision <= 8)
      jinit_d_main_controller(cinfo, FALSE /* never need full buffer here */);
    else if (cinfo->data_precision <= 12)
      j12init_d_main_controller(cinfo,
                                FALSE /* never need full buffer here */);
    else
#ifdef D_LOSSLESS_SUPPORTED
      j16init_d_main_controller(cinfo,
                                FALSE /* never need full buffer here */);
#else
      ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
#endif
  }

  /* We can now tell the memory manager to allocate virtual arrays. */
  (*cinfo->mem->realize_virt_arrays) ((j_common_ptr)cinfo);

  /* Initialize input side of decompressor to consume first scan. */
  (*cinfo->inputctl->start_input_pass) (cinfo);

  /* Set the first and last iMCU columns to decompress from single-scan images.
   * By default, decompress all of the iMCU columns.
   */
  cinfo->master->first_iMCU_col = 0;
  cinfo->master->last_iMCU_col = cinfo->MCUs_per_row - 1;
  cinfo->master->last_good_iMCU_row = 0;

#ifdef D_MULTISCAN_FILES_SUPPORTED
  /* If jpeg_start_decompress will read the whole file, initialize
   * progress monitoring appropriately.  The input step is counted
   * as one pass.
   */
  if (cinfo->progress != NULL && !cinfo->buffered_image &&
      cinfo->inputctl->has_multiple_scans) {
    int nscans;
    /* Estimate number of scans to set pass_limit. */
    if (cinfo->progressive_mode) {
      /* Arbitrarily estimate 2 interleaved DC scans + 3 AC scans/component. */
      nscans = 2 + 3 * cinfo->num_components;
    } else {
      /* For a nonprogressive multiscan file, estimate 1 scan per component. */
      nscans = cinfo->num_components;
    }
    cinfo->progress->pass_counter = 0L;
    cinfo->progress->pass_limit = (long)cinfo->total_iMCU_rows * nscans;
    cinfo->progress->completed_passes = 0;
    cinfo->progress->total_passes = (cinfo->enable_2pass_quant ? 3 : 2);
    /* Count the input pass as done */
    master->pass_number++;
  }
#endif /* D_MULTISCAN_FILES_SUPPORTED */
}


/*
 * Per-pass setup.
 * This is called at the beginning of each output pass.  We determine which
 * modules will be active during this pass and give them appropriate
 * start_pass calls.  We also set is_dummy_pass to indicate whether this
 * is a "real" output pass or a dummy pass for color quantization.
 * (In the latter case, jdapistd.c will crank the pass to completion.)
 */

METHODDEF(void)
prepare_for_output_pass(j_decompress_ptr cinfo)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;

  if (master->pub.is_dummy_pass) {
#ifdef QUANT_2PASS_SUPPORTED
    /* Final pass of 2-pass quantization */
    master->pub.is_dummy_pass = FALSE;
    (*cinfo->cquantize->start_pass) (cinfo, FALSE);
    (*cinfo->post->start_pass) (cinfo, JBUF_CRANK_DEST);
    (*cinfo->main->start_pass) (cinfo, JBUF_CRANK_DEST);
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif /* QUANT_2PASS_SUPPORTED */
  } else {
    if (cinfo->quantize_colors && cinfo->colormap == NULL) {
      /* Select new quantization method */
      if (cinfo->two_pass_quantize && cinfo->enable_2pass_quant) {
        cinfo->cquantize = master->quantizer_2pass;
        master->pub.is_dummy_pass = TRUE;
      } else if (cinfo->enable_1pass_quant) {
        cinfo->cquantize = master->quantizer_1pass;
      } else {
        ERREXIT(cinfo, JERR_MODE_CHANGE);
      }
    }
    (*cinfo->idct->start_pass) (cinfo);
    (*cinfo->coef->start_output_pass) (cinfo);
    if (!cinfo->raw_data_out) {
      if (!master->using_merged_upsample)
        (*cinfo->cconvert->start_pass) (cinfo);
      (*cinfo->upsample->start_pass) (cinfo);
      if (cinfo->quantize_colors)
        (*cinfo->cquantize->start_pass) (cinfo, master->pub.is_dummy_pass);
      (*cinfo->post->start_pass) (cinfo,
            (master->pub.is_dummy_pass ? JBUF_SAVE_AND_PASS : JBUF_PASS_THRU));
      (*cinfo->main->start_pass) (cinfo, JBUF_PASS_THRU);
    }
  }

  /* Set up progress monitor's pass info if present */
  if (cinfo->progress != NULL) {
    cinfo->progress->completed_passes = master->pass_number;
    cinfo->progress->total_passes = master->pass_number +
                                    (master->pub.is_dummy_pass ? 2 : 1);
    /* In buffered-image mode, we assume one more output pass if EOI not
     * yet reached, but no more passes if EOI has been reached.
     */
    if (cinfo->buffered_image && !cinfo->inputctl->eoi_reached) {
      cinfo->progress->total_passes += (cinfo->enable_2pass_quant ? 2 : 1);
    }
  }
}


/*
 * Finish up at end of an output pass.
 */

METHODDEF(void)
finish_output_pass(j_decompress_ptr cinfo)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;

  if (cinfo->quantize_colors)
    (*cinfo->cquantize->finish_pass) (cinfo);
  master->pass_number++;
}


#ifdef D_MULTISCAN_FILES_SUPPORTED

/*
 * Switch to a new external colormap between output passes.
 */

GLOBAL(void)
jpeg_new_colormap(j_decompress_ptr cinfo)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;

  /* Prevent application from calling me at wrong times */
  if (cinfo->global_state != DSTATE_BUFIMAGE)
    ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);

  if (cinfo->quantize_colors && cinfo->enable_external_quant &&
      cinfo->colormap != NULL) {
    /* Select 2-pass quantizer for external colormap use */
    cinfo->cquantize = master->quantizer_2pass;
    /* Notify quantizer of colormap change */
    (*cinfo->cquantize->new_color_map) (cinfo);
    master->pub.is_dummy_pass = FALSE; /* just in case */
  } else
    ERREXIT(cinfo, JERR_MODE_CHANGE);
}

#endif /* D_MULTISCAN_FILES_SUPPORTED */


/*
 * Initialize master decompression control and select active modules.
 * This is performed at the start of jpeg_start_decompress.
 */

GLOBAL(void)
jinit_master_decompress(j_decompress_ptr cinfo)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;

  master->pub.prepare_for_output_pass = prepare_for_output_pass;
  master->pub.finish_output_pass = finish_output_pass;

  master->pub.is_dummy_pass = FALSE;
  master->pub.jinit_upsampler_no_alloc = FALSE;

  master_selection(cinfo);
}
