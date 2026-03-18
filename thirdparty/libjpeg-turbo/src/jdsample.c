/*
 * jdsample.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2010, 2015-2016, 2022, 2024, D. R. Commander.
 * Copyright (C) 2014, MIPS Technologies, Inc., California.
 * Copyright (C) 2015, Google, Inc.
 * Copyright (C) 2019-2020, Arm Limited.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains upsampling routines.
 *
 * Upsampling input data is counted in "row groups".  A row group
 * is defined to be (v_samp_factor * DCT_scaled_size / min_DCT_scaled_size)
 * sample rows of each component.  Upsampling will normally produce
 * max_v_samp_factor pixel rows from each row group (but this could vary
 * if the upsampler is applying a scale factor of its own).
 *
 * An excellent reference for image resampling is
 *   Digital Image Warping, George Wolberg, 1990.
 *   Pub. by IEEE Computer Society Press, Los Alamitos, CA. ISBN 0-8186-8944-7.
 */

#include "jinclude.h"
#include "jdsample.h"
#include "jsimd.h"
#include "jpegapicomp.h"



#if BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED)

/*
 * Initialize for an upsampling pass.
 */

METHODDEF(void)
start_pass_upsample(j_decompress_ptr cinfo)
{
  my_upsample_ptr upsample = (my_upsample_ptr)cinfo->upsample;

  /* Mark the conversion buffer empty */
  upsample->next_row_out = cinfo->max_v_samp_factor;
  /* Initialize total-height counter for detecting bottom of image */
  upsample->rows_to_go = cinfo->output_height;
}


/*
 * Control routine to do upsampling (and color conversion).
 *
 * In this version we upsample each component independently.
 * We upsample one row group into the conversion buffer, then apply
 * color conversion a row at a time.
 */

METHODDEF(void)
sep_upsample(j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
             JDIMENSION *in_row_group_ctr, JDIMENSION in_row_groups_avail,
             _JSAMPARRAY output_buf, JDIMENSION *out_row_ctr,
             JDIMENSION out_rows_avail)
{
  my_upsample_ptr upsample = (my_upsample_ptr)cinfo->upsample;
  int ci;
  jpeg_component_info *compptr;
  JDIMENSION num_rows;

  /* Fill the conversion buffer, if it's empty */
  if (upsample->next_row_out >= cinfo->max_v_samp_factor) {
    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
         ci++, compptr++) {
      /* Invoke per-component upsample method.  Notice we pass a POINTER
       * to color_buf[ci], so that fullsize_upsample can change it.
       */
      (*upsample->methods[ci]) (cinfo, compptr,
        input_buf[ci] + (*in_row_group_ctr * upsample->rowgroup_height[ci]),
        upsample->color_buf + ci);
    }
    upsample->next_row_out = 0;
  }

  /* Color-convert and emit rows */

  /* How many we have in the buffer: */
  num_rows = (JDIMENSION)(cinfo->max_v_samp_factor - upsample->next_row_out);
  /* Not more than the distance to the end of the image.  Need this test
   * in case the image height is not a multiple of max_v_samp_factor:
   */
  if (num_rows > upsample->rows_to_go)
    num_rows = upsample->rows_to_go;
  /* And not more than what the client can accept: */
  out_rows_avail -= *out_row_ctr;
  if (num_rows > out_rows_avail)
    num_rows = out_rows_avail;

  (*cinfo->cconvert->_color_convert) (cinfo, upsample->color_buf,
                                      (JDIMENSION)upsample->next_row_out,
                                      output_buf + *out_row_ctr,
                                      (int)num_rows);

  /* Adjust counts */
  *out_row_ctr += num_rows;
  upsample->rows_to_go -= num_rows;
  upsample->next_row_out += num_rows;
  /* When the buffer is emptied, declare this input row group consumed */
  if (upsample->next_row_out >= cinfo->max_v_samp_factor)
    (*in_row_group_ctr)++;
}


/*
 * These are the routines invoked by sep_upsample to upsample pixel values
 * of a single component.  One row group is processed per call.
 */


/*
 * For full-size components, we just make color_buf[ci] point at the
 * input buffer, and thus avoid copying any data.  Note that this is
 * safe only because sep_upsample doesn't declare the input row group
 * "consumed" until we are done color converting and emitting it.
 */

METHODDEF(void)
fullsize_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                  _JSAMPARRAY input_data, _JSAMPARRAY *output_data_ptr)
{
  *output_data_ptr = input_data;
}


/*
 * This is a no-op version used for "uninteresting" components.
 * These components will not be referenced by color conversion.
 */

METHODDEF(void)
noop_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
              _JSAMPARRAY input_data, _JSAMPARRAY *output_data_ptr)
{
  *output_data_ptr = NULL;      /* safety check */
}


/*
 * This version handles any integral sampling ratios.
 * This is not used for typical JPEG files, so it need not be fast.
 * Nor, for that matter, is it particularly accurate: the algorithm is
 * simple replication of the input pixel onto the corresponding output
 * pixels.  The hi-falutin sampling literature refers to this as a
 * "box filter".  A box filter tends to introduce visible artifacts,
 * so if you are actually going to use 3:1 or 4:1 sampling ratios
 * you would be well advised to improve this code.
 */

METHODDEF(void)
int_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
             _JSAMPARRAY input_data, _JSAMPARRAY *output_data_ptr)
{
  my_upsample_ptr upsample = (my_upsample_ptr)cinfo->upsample;
  _JSAMPARRAY output_data = *output_data_ptr;
  register _JSAMPROW inptr, outptr;
  register _JSAMPLE invalue;
  register int h;
  _JSAMPROW outend;
  int h_expand, v_expand;
  int inrow, outrow;

  h_expand = upsample->h_expand[compptr->component_index];
  v_expand = upsample->v_expand[compptr->component_index];

  inrow = outrow = 0;
  while (outrow < cinfo->max_v_samp_factor) {
    /* Generate one output row with proper horizontal expansion */
    inptr = input_data[inrow];
    outptr = output_data[outrow];
    outend = outptr + cinfo->output_width;
    while (outptr < outend) {
      invalue = *inptr++;
      for (h = h_expand; h > 0; h--) {
        *outptr++ = invalue;
      }
    }
    /* Generate any additional output rows by duplicating the first one */
    if (v_expand > 1) {
      _jcopy_sample_rows(output_data, outrow, output_data, outrow + 1,
                         v_expand - 1, cinfo->output_width);
    }
    inrow++;
    outrow += v_expand;
  }
}


/*
 * Fast processing for the common case of 2:1 horizontal and 1:1 vertical.
 * It's still a box filter.
 */

METHODDEF(void)
h2v1_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
              _JSAMPARRAY input_data, _JSAMPARRAY *output_data_ptr)
{
  _JSAMPARRAY output_data = *output_data_ptr;
  register _JSAMPROW inptr, outptr;
  register _JSAMPLE invalue;
  _JSAMPROW outend;
  int inrow;

  for (inrow = 0; inrow < cinfo->max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];
    outend = outptr + cinfo->output_width;
    while (outptr < outend) {
      invalue = *inptr++;
      *outptr++ = invalue;
      *outptr++ = invalue;
    }
  }
}


/*
 * Fast processing for the common case of 2:1 horizontal and 2:1 vertical.
 * It's still a box filter.
 */

METHODDEF(void)
h2v2_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
              _JSAMPARRAY input_data, _JSAMPARRAY *output_data_ptr)
{
  _JSAMPARRAY output_data = *output_data_ptr;
  register _JSAMPROW inptr, outptr;
  register _JSAMPLE invalue;
  _JSAMPROW outend;
  int inrow, outrow;

  inrow = outrow = 0;
  while (outrow < cinfo->max_v_samp_factor) {
    inptr = input_data[inrow];
    outptr = output_data[outrow];
    outend = outptr + cinfo->output_width;
    while (outptr < outend) {
      invalue = *inptr++;
      *outptr++ = invalue;
      *outptr++ = invalue;
    }
    _jcopy_sample_rows(output_data, outrow, output_data, outrow + 1, 1,
                       cinfo->output_width);
    inrow++;
    outrow += 2;
  }
}


/*
 * Fancy processing for the common case of 2:1 horizontal and 1:1 vertical.
 *
 * The upsampling algorithm is linear interpolation between pixel centers,
 * also known as a "triangle filter".  This is a good compromise between
 * speed and visual quality.  The centers of the output pixels are 1/4 and 3/4
 * of the way between input pixel centers.
 *
 * A note about the "bias" calculations: when rounding fractional values to
 * integer, we do not want to always round 0.5 up to the next integer.
 * If we did that, we'd introduce a noticeable bias towards larger values.
 * Instead, this code is arranged so that 0.5 will be rounded up or down at
 * alternate pixel locations (a simple ordered dither pattern).
 */

METHODDEF(void)
h2v1_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    _JSAMPARRAY input_data, _JSAMPARRAY *output_data_ptr)
{
  _JSAMPARRAY output_data = *output_data_ptr;
  register _JSAMPROW inptr, outptr;
  register int invalue;
  register JDIMENSION colctr;
  int inrow;

  for (inrow = 0; inrow < cinfo->max_v_samp_factor; inrow++) {
    inptr = input_data[inrow];
    outptr = output_data[inrow];
    /* Special case for first column */
    invalue = *inptr++;
    *outptr++ = (_JSAMPLE)invalue;
    *outptr++ = (_JSAMPLE)((invalue * 3 + inptr[0] + 2) >> 2);

    for (colctr = compptr->downsampled_width - 2; colctr > 0; colctr--) {
      /* General case: 3/4 * nearer pixel + 1/4 * further pixel */
      invalue = (*inptr++) * 3;
      *outptr++ = (_JSAMPLE)((invalue + inptr[-2] + 1) >> 2);
      *outptr++ = (_JSAMPLE)((invalue + inptr[0] + 2) >> 2);
    }

    /* Special case for last column */
    invalue = *inptr;
    *outptr++ = (_JSAMPLE)((invalue * 3 + inptr[-1] + 1) >> 2);
    *outptr++ = (_JSAMPLE)invalue;
  }
}


/*
 * Fancy processing for 1:1 horizontal and 2:1 vertical (4:4:0 subsampling).
 *
 * This is a less common case, but it can be encountered when losslessly
 * rotating/transposing a JPEG file that uses 4:2:2 chroma subsampling.
 */

METHODDEF(void)
h1v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    _JSAMPARRAY input_data, _JSAMPARRAY *output_data_ptr)
{
  _JSAMPARRAY output_data = *output_data_ptr;
  _JSAMPROW inptr0, inptr1, outptr;
#if BITS_IN_JSAMPLE == 8
  int thiscolsum, bias;
#else
  JLONG thiscolsum, bias;
#endif
  JDIMENSION colctr;
  int inrow, outrow, v;

  inrow = outrow = 0;
  while (outrow < cinfo->max_v_samp_factor) {
    for (v = 0; v < 2; v++) {
      /* inptr0 points to nearest input row, inptr1 points to next nearest */
      inptr0 = input_data[inrow];
      if (v == 0) {             /* next nearest is row above */
        inptr1 = input_data[inrow - 1];
        bias = 1;
      } else {                  /* next nearest is row below */
        inptr1 = input_data[inrow + 1];
        bias = 2;
      }
      outptr = output_data[outrow++];

      for (colctr = 0; colctr < compptr->downsampled_width; colctr++) {
        thiscolsum = (*inptr0++) * 3 + (*inptr1++);
        *outptr++ = (_JSAMPLE)((thiscolsum + bias) >> 2);
      }
    }
    inrow++;
  }
}


/*
 * Fancy processing for the common case of 2:1 horizontal and 2:1 vertical.
 * Again a triangle filter; see comments for h2v1 case, above.
 *
 * It is OK for us to reference the adjacent input rows because we demanded
 * context from the main buffer controller (see initialization code).
 */

METHODDEF(void)
h2v2_fancy_upsample(j_decompress_ptr cinfo, jpeg_component_info *compptr,
                    _JSAMPARRAY input_data, _JSAMPARRAY *output_data_ptr)
{
  _JSAMPARRAY output_data = *output_data_ptr;
  register _JSAMPROW inptr0, inptr1, outptr;
#if BITS_IN_JSAMPLE == 8
  register int thiscolsum, lastcolsum, nextcolsum;
#else
  register JLONG thiscolsum, lastcolsum, nextcolsum;
#endif
  register JDIMENSION colctr;
  int inrow, outrow, v;

  inrow = outrow = 0;
  while (outrow < cinfo->max_v_samp_factor) {
    for (v = 0; v < 2; v++) {
      /* inptr0 points to nearest input row, inptr1 points to next nearest */
      inptr0 = input_data[inrow];
      if (v == 0)               /* next nearest is row above */
        inptr1 = input_data[inrow - 1];
      else                      /* next nearest is row below */
        inptr1 = input_data[inrow + 1];
      outptr = output_data[outrow++];

      /* Special case for first column */
      thiscolsum = (*inptr0++) * 3 + (*inptr1++);
      nextcolsum = (*inptr0++) * 3 + (*inptr1++);
      *outptr++ = (_JSAMPLE)((thiscolsum * 4 + 8) >> 4);
      *outptr++ = (_JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
      lastcolsum = thiscolsum;  thiscolsum = nextcolsum;

      for (colctr = compptr->downsampled_width - 2; colctr > 0; colctr--) {
        /* General case: 3/4 * nearer pixel + 1/4 * further pixel in each */
        /* dimension, thus 9/16, 3/16, 3/16, 1/16 overall */
        nextcolsum = (*inptr0++) * 3 + (*inptr1++);
        *outptr++ = (_JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
        *outptr++ = (_JSAMPLE)((thiscolsum * 3 + nextcolsum + 7) >> 4);
        lastcolsum = thiscolsum;  thiscolsum = nextcolsum;
      }

      /* Special case for last column */
      *outptr++ = (_JSAMPLE)((thiscolsum * 3 + lastcolsum + 8) >> 4);
      *outptr++ = (_JSAMPLE)((thiscolsum * 4 + 7) >> 4);
    }
    inrow++;
  }
}


/*
 * Module initialization routine for upsampling.
 */

GLOBAL(void)
_jinit_upsampler(j_decompress_ptr cinfo)
{
  my_upsample_ptr upsample;
  int ci;
  jpeg_component_info *compptr;
  boolean need_buffer, do_fancy;
  int h_in_group, v_in_group, h_out_group, v_out_group;

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

  if (!cinfo->master->jinit_upsampler_no_alloc) {
    upsample = (my_upsample_ptr)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  sizeof(my_upsampler));
    cinfo->upsample = (struct jpeg_upsampler *)upsample;
    upsample->pub.start_pass = start_pass_upsample;
    upsample->pub._upsample = sep_upsample;
    upsample->pub.need_context_rows = FALSE; /* until we find out differently */
  } else
    upsample = (my_upsample_ptr)cinfo->upsample;

  if (cinfo->CCIR601_sampling)  /* this isn't supported */
    ERREXIT(cinfo, JERR_CCIR601_NOTIMPL);

  /* jdmainct.c doesn't support context rows when min_DCT_scaled_size = 1,
   * so don't ask for it.
   */
  do_fancy = cinfo->do_fancy_upsampling && cinfo->_min_DCT_scaled_size > 1;

  /* Verify we can handle the sampling factors, select per-component methods,
   * and create storage as needed.
   */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Compute size of an "input group" after IDCT scaling.  This many samples
     * are to be converted to max_h_samp_factor * max_v_samp_factor pixels.
     */
    h_in_group = (compptr->h_samp_factor * compptr->_DCT_scaled_size) /
                 cinfo->_min_DCT_scaled_size;
    v_in_group = (compptr->v_samp_factor * compptr->_DCT_scaled_size) /
                 cinfo->_min_DCT_scaled_size;
    h_out_group = cinfo->max_h_samp_factor;
    v_out_group = cinfo->max_v_samp_factor;
    upsample->rowgroup_height[ci] = v_in_group; /* save for use later */
    need_buffer = TRUE;
    if (!compptr->component_needed) {
      /* Don't bother to upsample an uninteresting component. */
      upsample->methods[ci] = noop_upsample;
      need_buffer = FALSE;
    } else if (h_in_group == h_out_group && v_in_group == v_out_group) {
      /* Fullsize components can be processed without any work. */
      upsample->methods[ci] = fullsize_upsample;
      need_buffer = FALSE;
    } else if (h_in_group * 2 == h_out_group && v_in_group == v_out_group) {
      /* Special cases for 2h1v upsampling */
      if (do_fancy && compptr->downsampled_width > 2) {
#ifdef WITH_SIMD
        if (jsimd_can_h2v1_fancy_upsample())
          upsample->methods[ci] = jsimd_h2v1_fancy_upsample;
        else
#endif
          upsample->methods[ci] = h2v1_fancy_upsample;
      } else {
#ifdef WITH_SIMD
        if (jsimd_can_h2v1_upsample())
          upsample->methods[ci] = jsimd_h2v1_upsample;
        else
#endif
          upsample->methods[ci] = h2v1_upsample;
      }
    } else if (h_in_group == h_out_group &&
               v_in_group * 2 == v_out_group && do_fancy) {
      /* Non-fancy upsampling is handled by the generic method */
#if defined(WITH_SIMD) && (defined(__arm__) || defined(__aarch64__) || \
                           defined(_M_ARM) || defined(_M_ARM64))
      if (jsimd_can_h1v2_fancy_upsample())
        upsample->methods[ci] = jsimd_h1v2_fancy_upsample;
      else
#endif
        upsample->methods[ci] = h1v2_fancy_upsample;
      upsample->pub.need_context_rows = TRUE;
    } else if (h_in_group * 2 == h_out_group &&
               v_in_group * 2 == v_out_group) {
      /* Special cases for 2h2v upsampling */
      if (do_fancy && compptr->downsampled_width > 2) {
#ifdef WITH_SIMD
        if (jsimd_can_h2v2_fancy_upsample())
          upsample->methods[ci] = jsimd_h2v2_fancy_upsample;
        else
#endif
          upsample->methods[ci] = h2v2_fancy_upsample;
        upsample->pub.need_context_rows = TRUE;
      } else {
#ifdef WITH_SIMD
        if (jsimd_can_h2v2_upsample())
          upsample->methods[ci] = jsimd_h2v2_upsample;
        else
#endif
          upsample->methods[ci] = h2v2_upsample;
      }
    } else if ((h_out_group % h_in_group) == 0 &&
               (v_out_group % v_in_group) == 0) {
      /* Generic integral-factors upsampling method */
#if defined(WITH_SIMD) && defined(__mips__)
      if (jsimd_can_int_upsample())
        upsample->methods[ci] = jsimd_int_upsample;
      else
#endif
        upsample->methods[ci] = int_upsample;
      upsample->h_expand[ci] = (UINT8)(h_out_group / h_in_group);
      upsample->v_expand[ci] = (UINT8)(v_out_group / v_in_group);
    } else
      ERREXIT(cinfo, JERR_FRACT_SAMPLE_NOTIMPL);
    if (need_buffer && !cinfo->master->jinit_upsampler_no_alloc) {
      upsample->color_buf[ci] = (_JSAMPARRAY)(*cinfo->mem->alloc_sarray)
        ((j_common_ptr)cinfo, JPOOL_IMAGE,
         (JDIMENSION)jround_up((long)cinfo->output_width,
                               (long)cinfo->max_h_samp_factor),
         (JDIMENSION)cinfo->max_v_samp_factor);
    }
  }
}

#endif /* BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED) */
