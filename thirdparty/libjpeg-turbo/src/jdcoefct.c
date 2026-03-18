/*
 * jdcoefct.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1997, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2010, 2015-2016, 2019-2020, 2022-2023, D. R. Commander.
 * Copyright (C) 2015, 2020, Google, Inc.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains the coefficient buffer controller for decompression.
 * This controller is the top level of the lossy JPEG decompressor proper.
 * The coefficient buffer lies between entropy decoding and inverse-DCT steps.
 *
 * In buffered-image mode, this controller is the interface between
 * input-oriented processing and output-oriented processing.
 * Also, the input side (only) is used when reading a file for transcoding.
 */

#include "jinclude.h"
#include "jdcoefct.h"
#include "jpegapicomp.h"
#include "jsamplecomp.h"


/* Forward declarations */
METHODDEF(int) decompress_onepass(j_decompress_ptr cinfo,
                                  _JSAMPIMAGE output_buf);
#ifdef D_MULTISCAN_FILES_SUPPORTED
METHODDEF(int) decompress_data(j_decompress_ptr cinfo, _JSAMPIMAGE output_buf);
#endif
#ifdef BLOCK_SMOOTHING_SUPPORTED
LOCAL(boolean) smoothing_ok(j_decompress_ptr cinfo);
METHODDEF(int) decompress_smooth_data(j_decompress_ptr cinfo,
                                      _JSAMPIMAGE output_buf);
#endif


/*
 * Initialize for an input processing pass.
 */

METHODDEF(void)
start_input_pass(j_decompress_ptr cinfo)
{
  cinfo->input_iMCU_row = 0;
  start_iMCU_row(cinfo);
}


/*
 * Initialize for an output processing pass.
 */

METHODDEF(void)
start_output_pass(j_decompress_ptr cinfo)
{
#ifdef BLOCK_SMOOTHING_SUPPORTED
  my_coef_ptr coef = (my_coef_ptr)cinfo->coef;

  /* If multipass, check to see whether to use block smoothing on this pass */
  if (coef->pub.coef_arrays != NULL) {
    if (cinfo->do_block_smoothing && smoothing_ok(cinfo))
      coef->pub._decompress_data = decompress_smooth_data;
    else
      coef->pub._decompress_data = decompress_data;
  }
#endif
  cinfo->output_iMCU_row = 0;
}


/*
 * Decompress and return some data in the single-pass case.
 * Always attempts to emit one fully interleaved MCU row ("iMCU" row).
 * Input and output must run in lockstep since we have only a one-MCU buffer.
 * Return value is JPEG_ROW_COMPLETED, JPEG_SCAN_COMPLETED, or JPEG_SUSPENDED.
 *
 * NB: output_buf contains a plane for each component in image,
 * which we index according to the component's SOF position.
 */

METHODDEF(int)
decompress_onepass(j_decompress_ptr cinfo, _JSAMPIMAGE output_buf)
{
  my_coef_ptr coef = (my_coef_ptr)cinfo->coef;
  JDIMENSION MCU_col_num;       /* index of current MCU within row */
  JDIMENSION last_MCU_col = cinfo->MCUs_per_row - 1;
  JDIMENSION last_iMCU_row = cinfo->total_iMCU_rows - 1;
  int blkn, ci, xindex, yindex, yoffset, useful_width;
  _JSAMPARRAY output_ptr;
  JDIMENSION start_col, output_col;
  jpeg_component_info *compptr;
  _inverse_DCT_method_ptr inverse_DCT;

  /* Loop to process as much as one whole iMCU row */
  for (yoffset = coef->MCU_vert_offset; yoffset < coef->MCU_rows_per_iMCU_row;
       yoffset++) {
    for (MCU_col_num = coef->MCU_ctr; MCU_col_num <= last_MCU_col;
         MCU_col_num++) {
      /* Try to fetch an MCU.  Entropy decoder expects buffer to be zeroed. */
      jzero_far((void *)coef->MCU_buffer[0],
                (size_t)(cinfo->blocks_in_MCU * sizeof(JBLOCK)));
      if (!cinfo->entropy->insufficient_data)
        cinfo->master->last_good_iMCU_row = cinfo->input_iMCU_row;
      if (!(*cinfo->entropy->decode_mcu) (cinfo, coef->MCU_buffer)) {
        /* Suspension forced; update state counters and exit */
        coef->MCU_vert_offset = yoffset;
        coef->MCU_ctr = MCU_col_num;
        return JPEG_SUSPENDED;
      }

      /* Only perform the IDCT on blocks that are contained within the desired
       * cropping region.
       */
      if (MCU_col_num >= cinfo->master->first_iMCU_col &&
          MCU_col_num <= cinfo->master->last_iMCU_col) {
        /* Determine where data should go in output_buf and do the IDCT thing.
         * We skip dummy blocks at the right and bottom edges (but blkn gets
         * incremented past them!).  Note the inner loop relies on having
         * allocated the MCU_buffer[] blocks sequentially.
         */
        blkn = 0;               /* index of current DCT block within MCU */
        for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
          compptr = cinfo->cur_comp_info[ci];
          /* Don't bother to IDCT an uninteresting component. */
          if (!compptr->component_needed) {
            blkn += compptr->MCU_blocks;
            continue;
          }
          inverse_DCT = cinfo->idct->_inverse_DCT[compptr->component_index];
          useful_width = (MCU_col_num < last_MCU_col) ?
                         compptr->MCU_width : compptr->last_col_width;
          output_ptr = output_buf[compptr->component_index] +
                       yoffset * compptr->_DCT_scaled_size;
          start_col = (MCU_col_num - cinfo->master->first_iMCU_col) *
                      compptr->MCU_sample_width;
          for (yindex = 0; yindex < compptr->MCU_height; yindex++) {
            if (cinfo->input_iMCU_row < last_iMCU_row ||
                yoffset + yindex < compptr->last_row_height) {
              output_col = start_col;
              for (xindex = 0; xindex < useful_width; xindex++) {
                (*inverse_DCT) (cinfo, compptr,
                                (JCOEFPTR)coef->MCU_buffer[blkn + xindex],
                                output_ptr, output_col);
                output_col += compptr->_DCT_scaled_size;
              }
            }
            blkn += compptr->MCU_width;
            output_ptr += compptr->_DCT_scaled_size;
          }
        }
      }
    }
    /* Completed an MCU row, but perhaps not an iMCU row */
    coef->MCU_ctr = 0;
  }
  /* Completed the iMCU row, advance counters for next one */
  cinfo->output_iMCU_row++;
  if (++(cinfo->input_iMCU_row) < cinfo->total_iMCU_rows) {
    start_iMCU_row(cinfo);
    return JPEG_ROW_COMPLETED;
  }
  /* Completed the scan */
  (*cinfo->inputctl->finish_input_pass) (cinfo);
  return JPEG_SCAN_COMPLETED;
}


/*
 * Dummy consume-input routine for single-pass operation.
 */

METHODDEF(int)
dummy_consume_data(j_decompress_ptr cinfo)
{
  return JPEG_SUSPENDED;        /* Always indicate nothing was done */
}


#ifdef D_MULTISCAN_FILES_SUPPORTED

/*
 * Consume input data and store it in the full-image coefficient buffer.
 * We read as much as one fully interleaved MCU row ("iMCU" row) per call,
 * ie, v_samp_factor block rows for each component in the scan.
 * Return value is JPEG_ROW_COMPLETED, JPEG_SCAN_COMPLETED, or JPEG_SUSPENDED.
 */

METHODDEF(int)
consume_data(j_decompress_ptr cinfo)
{
  my_coef_ptr coef = (my_coef_ptr)cinfo->coef;
  JDIMENSION MCU_col_num;       /* index of current MCU within row */
  int blkn, ci, xindex, yindex, yoffset;
  JDIMENSION start_col;
  JBLOCKARRAY buffer[MAX_COMPS_IN_SCAN];
  JBLOCKROW buffer_ptr;
  jpeg_component_info *compptr;

  /* Align the virtual buffers for the components used in this scan. */
  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    buffer[ci] = (*cinfo->mem->access_virt_barray)
      ((j_common_ptr)cinfo, coef->whole_image[compptr->component_index],
       cinfo->input_iMCU_row * compptr->v_samp_factor,
       (JDIMENSION)compptr->v_samp_factor, TRUE);
    /* Note: entropy decoder expects buffer to be zeroed,
     * but this is handled automatically by the memory manager
     * because we requested a pre-zeroed array.
     */
  }

  /* Loop to process one whole iMCU row */
  for (yoffset = coef->MCU_vert_offset; yoffset < coef->MCU_rows_per_iMCU_row;
       yoffset++) {
    for (MCU_col_num = coef->MCU_ctr; MCU_col_num < cinfo->MCUs_per_row;
         MCU_col_num++) {
      /* Construct list of pointers to DCT blocks belonging to this MCU */
      blkn = 0;                 /* index of current DCT block within MCU */
      for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
        compptr = cinfo->cur_comp_info[ci];
        start_col = MCU_col_num * compptr->MCU_width;
        for (yindex = 0; yindex < compptr->MCU_height; yindex++) {
          buffer_ptr = buffer[ci][yindex + yoffset] + start_col;
          for (xindex = 0; xindex < compptr->MCU_width; xindex++) {
            coef->MCU_buffer[blkn++] = buffer_ptr++;
          }
        }
      }
      if (!cinfo->entropy->insufficient_data)
        cinfo->master->last_good_iMCU_row = cinfo->input_iMCU_row;
      /* Try to fetch the MCU. */
      if (!(*cinfo->entropy->decode_mcu) (cinfo, coef->MCU_buffer)) {
        /* Suspension forced; update state counters and exit */
        coef->MCU_vert_offset = yoffset;
        coef->MCU_ctr = MCU_col_num;
        return JPEG_SUSPENDED;
      }
    }
    /* Completed an MCU row, but perhaps not an iMCU row */
    coef->MCU_ctr = 0;
  }
  /* Completed the iMCU row, advance counters for next one */
  if (++(cinfo->input_iMCU_row) < cinfo->total_iMCU_rows) {
    start_iMCU_row(cinfo);
    return JPEG_ROW_COMPLETED;
  }
  /* Completed the scan */
  (*cinfo->inputctl->finish_input_pass) (cinfo);
  return JPEG_SCAN_COMPLETED;
}


/*
 * Decompress and return some data in the multi-pass case.
 * Always attempts to emit one fully interleaved MCU row ("iMCU" row).
 * Return value is JPEG_ROW_COMPLETED, JPEG_SCAN_COMPLETED, or JPEG_SUSPENDED.
 *
 * NB: output_buf contains a plane for each component in image.
 */

METHODDEF(int)
decompress_data(j_decompress_ptr cinfo, _JSAMPIMAGE output_buf)
{
  my_coef_ptr coef = (my_coef_ptr)cinfo->coef;
  JDIMENSION last_iMCU_row = cinfo->total_iMCU_rows - 1;
  JDIMENSION block_num;
  int ci, block_row, block_rows;
  JBLOCKARRAY buffer;
  JBLOCKROW buffer_ptr;
  _JSAMPARRAY output_ptr;
  JDIMENSION output_col;
  jpeg_component_info *compptr;
  _inverse_DCT_method_ptr inverse_DCT;

  /* Force some input to be done if we are getting ahead of the input. */
  while (cinfo->input_scan_number < cinfo->output_scan_number ||
         (cinfo->input_scan_number == cinfo->output_scan_number &&
          cinfo->input_iMCU_row <= cinfo->output_iMCU_row)) {
    if ((*cinfo->inputctl->consume_input) (cinfo) == JPEG_SUSPENDED)
      return JPEG_SUSPENDED;
  }

  /* OK, output from the virtual arrays. */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Don't bother to IDCT an uninteresting component. */
    if (!compptr->component_needed)
      continue;
    /* Align the virtual buffer for this component. */
    buffer = (*cinfo->mem->access_virt_barray)
      ((j_common_ptr)cinfo, coef->whole_image[ci],
       cinfo->output_iMCU_row * compptr->v_samp_factor,
       (JDIMENSION)compptr->v_samp_factor, FALSE);
    /* Count non-dummy DCT block rows in this iMCU row. */
    if (cinfo->output_iMCU_row < last_iMCU_row)
      block_rows = compptr->v_samp_factor;
    else {
      /* NB: can't use last_row_height here; it is input-side-dependent! */
      block_rows = (int)(compptr->height_in_blocks % compptr->v_samp_factor);
      if (block_rows == 0) block_rows = compptr->v_samp_factor;
    }
    inverse_DCT = cinfo->idct->_inverse_DCT[ci];
    output_ptr = output_buf[ci];
    /* Loop over all DCT blocks to be processed. */
    for (block_row = 0; block_row < block_rows; block_row++) {
      buffer_ptr = buffer[block_row] + cinfo->master->first_MCU_col[ci];
      output_col = 0;
      for (block_num = cinfo->master->first_MCU_col[ci];
           block_num <= cinfo->master->last_MCU_col[ci]; block_num++) {
        (*inverse_DCT) (cinfo, compptr, (JCOEFPTR)buffer_ptr, output_ptr,
                        output_col);
        buffer_ptr++;
        output_col += compptr->_DCT_scaled_size;
      }
      output_ptr += compptr->_DCT_scaled_size;
    }
  }

  if (++(cinfo->output_iMCU_row) < cinfo->total_iMCU_rows)
    return JPEG_ROW_COMPLETED;
  return JPEG_SCAN_COMPLETED;
}

#endif /* D_MULTISCAN_FILES_SUPPORTED */


#ifdef BLOCK_SMOOTHING_SUPPORTED

/*
 * This code applies interblock smoothing; the first 9 AC coefficients are
 * estimated from the DC values of a DCT block and its 24 neighboring blocks.
 * We apply smoothing only for progressive JPEG decoding, and only if
 * the coefficients it can estimate are not yet known to full precision.
 */

/* Natural-order array positions of the first 9 zigzag-order coefficients */
#define Q01_POS  1
#define Q10_POS  8
#define Q20_POS  16
#define Q11_POS  9
#define Q02_POS  2
#define Q03_POS  3
#define Q12_POS  10
#define Q21_POS  17
#define Q30_POS  24

/*
 * Determine whether block smoothing is applicable and safe.
 * We also latch the current states of the coef_bits[] entries for the
 * AC coefficients; otherwise, if the input side of the decompressor
 * advances into a new scan, we might think the coefficients are known
 * more accurately than they really are.
 */

LOCAL(boolean)
smoothing_ok(j_decompress_ptr cinfo)
{
  my_coef_ptr coef = (my_coef_ptr)cinfo->coef;
  boolean smoothing_useful = FALSE;
  int ci, coefi;
  jpeg_component_info *compptr;
  JQUANT_TBL *qtable;
  int *coef_bits, *prev_coef_bits;
  int *coef_bits_latch, *prev_coef_bits_latch;

  if (!cinfo->progressive_mode || cinfo->coef_bits == NULL)
    return FALSE;

  /* Allocate latch area if not already done */
  if (coef->coef_bits_latch == NULL)
    coef->coef_bits_latch = (int *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  cinfo->num_components * 2 *
                                  (SAVED_COEFS * sizeof(int)));
  coef_bits_latch = coef->coef_bits_latch;
  prev_coef_bits_latch =
    &coef->coef_bits_latch[cinfo->num_components * SAVED_COEFS];

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* All components' quantization values must already be latched. */
    if ((qtable = compptr->quant_table) == NULL)
      return FALSE;
    /* Verify DC & first 9 AC quantizers are nonzero to avoid zero-divide. */
    if (qtable->quantval[0] == 0 ||
        qtable->quantval[Q01_POS] == 0 ||
        qtable->quantval[Q10_POS] == 0 ||
        qtable->quantval[Q20_POS] == 0 ||
        qtable->quantval[Q11_POS] == 0 ||
        qtable->quantval[Q02_POS] == 0 ||
        qtable->quantval[Q03_POS] == 0 ||
        qtable->quantval[Q12_POS] == 0 ||
        qtable->quantval[Q21_POS] == 0 ||
        qtable->quantval[Q30_POS] == 0)
      return FALSE;
    /* DC values must be at least partly known for all components. */
    coef_bits = cinfo->coef_bits[ci];
    prev_coef_bits = cinfo->coef_bits[ci + cinfo->num_components];
    if (coef_bits[0] < 0)
      return FALSE;
    coef_bits_latch[0] = coef_bits[0];
    /* Block smoothing is helpful if some AC coefficients remain inaccurate. */
    for (coefi = 1; coefi < SAVED_COEFS; coefi++) {
      if (cinfo->input_scan_number > 1)
        prev_coef_bits_latch[coefi] = prev_coef_bits[coefi];
      else
        prev_coef_bits_latch[coefi] = -1;
      coef_bits_latch[coefi] = coef_bits[coefi];
      if (coef_bits[coefi] != 0)
        smoothing_useful = TRUE;
    }
    coef_bits_latch += SAVED_COEFS;
    prev_coef_bits_latch += SAVED_COEFS;
  }

  return smoothing_useful;
}


/*
 * Variant of decompress_data for use when doing block smoothing.
 */

METHODDEF(int)
decompress_smooth_data(j_decompress_ptr cinfo, _JSAMPIMAGE output_buf)
{
  my_coef_ptr coef = (my_coef_ptr)cinfo->coef;
  JDIMENSION last_iMCU_row = cinfo->total_iMCU_rows - 1;
  JDIMENSION block_num, last_block_column;
  int ci, block_row, block_rows, access_rows, image_block_row,
    image_block_rows;
  JBLOCKARRAY buffer;
  JBLOCKROW buffer_ptr, prev_prev_block_row, prev_block_row;
  JBLOCKROW next_block_row, next_next_block_row;
  _JSAMPARRAY output_ptr;
  JDIMENSION output_col;
  jpeg_component_info *compptr;
  _inverse_DCT_method_ptr inverse_DCT;
  boolean change_dc;
  JCOEF *workspace;
  int *coef_bits;
  JQUANT_TBL *quanttbl;
  JLONG Q00, Q01, Q02, Q03 = 0, Q10, Q11, Q12 = 0, Q20, Q21 = 0, Q30 = 0, num;
  int DC01, DC02, DC03, DC04, DC05, DC06, DC07, DC08, DC09, DC10, DC11, DC12,
      DC13, DC14, DC15, DC16, DC17, DC18, DC19, DC20, DC21, DC22, DC23, DC24,
      DC25;
  int Al, pred;

  /* Keep a local variable to avoid looking it up more than once */
  workspace = coef->workspace;

  /* Force some input to be done if we are getting ahead of the input. */
  while (cinfo->input_scan_number <= cinfo->output_scan_number &&
         !cinfo->inputctl->eoi_reached) {
    if (cinfo->input_scan_number == cinfo->output_scan_number) {
      /* If input is working on current scan, we ordinarily want it to
       * have completed the current row.  But if input scan is DC,
       * we want it to keep two rows ahead so that next two block rows' DC
       * values are up to date.
       */
      JDIMENSION delta = (cinfo->Ss == 0) ? 2 : 0;
      if (cinfo->input_iMCU_row > cinfo->output_iMCU_row + delta)
        break;
    }
    if ((*cinfo->inputctl->consume_input) (cinfo) == JPEG_SUSPENDED)
      return JPEG_SUSPENDED;
  }

  /* OK, output from the virtual arrays. */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Don't bother to IDCT an uninteresting component. */
    if (!compptr->component_needed)
      continue;
    /* Count non-dummy DCT block rows in this iMCU row. */
    if (cinfo->output_iMCU_row + 1 < last_iMCU_row) {
      block_rows = compptr->v_samp_factor;
      access_rows = block_rows * 3; /* this and next two iMCU rows */
    } else if (cinfo->output_iMCU_row < last_iMCU_row) {
      block_rows = compptr->v_samp_factor;
      access_rows = block_rows * 2; /* this and next iMCU row */
    } else {
      /* NB: can't use last_row_height here; it is input-side-dependent! */
      block_rows = (int)(compptr->height_in_blocks % compptr->v_samp_factor);
      if (block_rows == 0) block_rows = compptr->v_samp_factor;
      access_rows = block_rows; /* this iMCU row only */
    }
    /* Align the virtual buffer for this component. */
    if (cinfo->output_iMCU_row > 1) {
      access_rows += 2 * compptr->v_samp_factor; /* prior two iMCU rows too */
      buffer = (*cinfo->mem->access_virt_barray)
        ((j_common_ptr)cinfo, coef->whole_image[ci],
         (cinfo->output_iMCU_row - 2) * compptr->v_samp_factor,
         (JDIMENSION)access_rows, FALSE);
      buffer += 2 * compptr->v_samp_factor; /* point to current iMCU row */
    } else if (cinfo->output_iMCU_row > 0) {
      access_rows += compptr->v_samp_factor; /* prior iMCU row too */
      buffer = (*cinfo->mem->access_virt_barray)
        ((j_common_ptr)cinfo, coef->whole_image[ci],
         (cinfo->output_iMCU_row - 1) * compptr->v_samp_factor,
         (JDIMENSION)access_rows, FALSE);
      buffer += compptr->v_samp_factor; /* point to current iMCU row */
    } else {
      buffer = (*cinfo->mem->access_virt_barray)
        ((j_common_ptr)cinfo, coef->whole_image[ci],
         (JDIMENSION)0, (JDIMENSION)access_rows, FALSE);
    }
    /* Fetch component-dependent info.
     * If the current scan is incomplete, then we use the component-dependent
     * info from the previous scan.
     */
    if (cinfo->output_iMCU_row > cinfo->master->last_good_iMCU_row)
      coef_bits =
        coef->coef_bits_latch + ((ci + cinfo->num_components) * SAVED_COEFS);
    else
      coef_bits = coef->coef_bits_latch + (ci * SAVED_COEFS);

    /* We only do DC interpolation if no AC coefficient data is available. */
    change_dc =
      coef_bits[1] == -1 && coef_bits[2] == -1 && coef_bits[3] == -1 &&
      coef_bits[4] == -1 && coef_bits[5] == -1 && coef_bits[6] == -1 &&
      coef_bits[7] == -1 && coef_bits[8] == -1 && coef_bits[9] == -1;

    quanttbl = compptr->quant_table;
    Q00 = quanttbl->quantval[0];
    Q01 = quanttbl->quantval[Q01_POS];
    Q10 = quanttbl->quantval[Q10_POS];
    Q20 = quanttbl->quantval[Q20_POS];
    Q11 = quanttbl->quantval[Q11_POS];
    Q02 = quanttbl->quantval[Q02_POS];
    if (change_dc) {
      Q03 = quanttbl->quantval[Q03_POS];
      Q12 = quanttbl->quantval[Q12_POS];
      Q21 = quanttbl->quantval[Q21_POS];
      Q30 = quanttbl->quantval[Q30_POS];
    }
    inverse_DCT = cinfo->idct->_inverse_DCT[ci];
    output_ptr = output_buf[ci];
    /* Loop over all DCT blocks to be processed. */
    image_block_rows = block_rows * cinfo->total_iMCU_rows;
    for (block_row = 0; block_row < block_rows; block_row++) {
      image_block_row = cinfo->output_iMCU_row * block_rows + block_row;
      buffer_ptr = buffer[block_row] + cinfo->master->first_MCU_col[ci];

      if (image_block_row > 0)
        prev_block_row =
          buffer[block_row - 1] + cinfo->master->first_MCU_col[ci];
      else
        prev_block_row = buffer_ptr;

      if (image_block_row > 1)
        prev_prev_block_row =
          buffer[block_row - 2] + cinfo->master->first_MCU_col[ci];
      else
        prev_prev_block_row = prev_block_row;

      if (image_block_row < image_block_rows - 1)
        next_block_row =
          buffer[block_row + 1] + cinfo->master->first_MCU_col[ci];
      else
        next_block_row = buffer_ptr;

      if (image_block_row < image_block_rows - 2)
        next_next_block_row =
          buffer[block_row + 2] + cinfo->master->first_MCU_col[ci];
      else
        next_next_block_row = next_block_row;

      /* We fetch the surrounding DC values using a sliding-register approach.
       * Initialize all 25 here so as to do the right thing on narrow pics.
       */
      DC01 = DC02 = DC03 = DC04 = DC05 = (int)prev_prev_block_row[0][0];
      DC06 = DC07 = DC08 = DC09 = DC10 = (int)prev_block_row[0][0];
      DC11 = DC12 = DC13 = DC14 = DC15 = (int)buffer_ptr[0][0];
      DC16 = DC17 = DC18 = DC19 = DC20 = (int)next_block_row[0][0];
      DC21 = DC22 = DC23 = DC24 = DC25 = (int)next_next_block_row[0][0];
      output_col = 0;
      last_block_column = compptr->width_in_blocks - 1;
      for (block_num = cinfo->master->first_MCU_col[ci];
           block_num <= cinfo->master->last_MCU_col[ci]; block_num++) {
        /* Fetch current DCT block into workspace so we can modify it. */
        jcopy_block_row(buffer_ptr, (JBLOCKROW)workspace, (JDIMENSION)1);
        /* Update DC values */
        if (block_num == cinfo->master->first_MCU_col[ci] &&
            block_num < last_block_column) {
          DC04 = DC05 = (int)prev_prev_block_row[1][0];
          DC09 = DC10 = (int)prev_block_row[1][0];
          DC14 = DC15 = (int)buffer_ptr[1][0];
          DC19 = DC20 = (int)next_block_row[1][0];
          DC24 = DC25 = (int)next_next_block_row[1][0];
        }
        if (block_num + 1 < last_block_column) {
          DC05 = (int)prev_prev_block_row[2][0];
          DC10 = (int)prev_block_row[2][0];
          DC15 = (int)buffer_ptr[2][0];
          DC20 = (int)next_block_row[2][0];
          DC25 = (int)next_next_block_row[2][0];
        }
        /* If DC interpolation is enabled, compute coefficient estimates using
         * a Gaussian-like kernel, keeping the averages of the DC values.
         *
         * If DC interpolation is disabled, compute coefficient estimates using
         * an algorithm similar to the one described in Section K.8 of the JPEG
         * standard, except applied to a 5x5 window rather than a 3x3 window.
         *
         * An estimate is applied only if the coefficient is still zero and is
         * not known to be fully accurate.
         */
        /* AC01 */
        if ((Al = coef_bits[1]) != 0 && workspace[1] == 0) {
          num = Q00 * (change_dc ?
                (-DC01 - DC02 + DC04 + DC05 - 3 * DC06 + 13 * DC07 -
                 13 * DC09 + 3 * DC10 - 3 * DC11 + 38 * DC12 - 38 * DC14 +
                 3 * DC15 - 3 * DC16 + 13 * DC17 - 13 * DC19 + 3 * DC20 -
                 DC21 - DC22 + DC24 + DC25) :
                (-7 * DC11 + 50 * DC12 - 50 * DC14 + 7 * DC15));
          if (num >= 0) {
            pred = (int)(((Q01 << 7) + num) / (Q01 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
          } else {
            pred = (int)(((Q01 << 7) - num) / (Q01 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
            pred = -pred;
          }
          workspace[1] = (JCOEF)pred;
        }
        /* AC10 */
        if ((Al = coef_bits[2]) != 0 && workspace[8] == 0) {
          num = Q00 * (change_dc ?
                (-DC01 - 3 * DC02 - 3 * DC03 - 3 * DC04 - DC05 - DC06 +
                 13 * DC07 + 38 * DC08 + 13 * DC09 - DC10 + DC16 -
                 13 * DC17 - 38 * DC18 - 13 * DC19 + DC20 + DC21 +
                 3 * DC22 + 3 * DC23 + 3 * DC24 + DC25) :
                (-7 * DC03 + 50 * DC08 - 50 * DC18 + 7 * DC23));
          if (num >= 0) {
            pred = (int)(((Q10 << 7) + num) / (Q10 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
          } else {
            pred = (int)(((Q10 << 7) - num) / (Q10 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
            pred = -pred;
          }
          workspace[8] = (JCOEF)pred;
        }
        /* AC20 */
        if ((Al = coef_bits[3]) != 0 && workspace[16] == 0) {
          num = Q00 * (change_dc ?
                (DC03 + 2 * DC07 + 7 * DC08 + 2 * DC09 - 5 * DC12 - 14 * DC13 -
                 5 * DC14 + 2 * DC17 + 7 * DC18 + 2 * DC19 + DC23) :
                (-DC03 + 13 * DC08 - 24 * DC13 + 13 * DC18 - DC23));
          if (num >= 0) {
            pred = (int)(((Q20 << 7) + num) / (Q20 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
          } else {
            pred = (int)(((Q20 << 7) - num) / (Q20 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
            pred = -pred;
          }
          workspace[16] = (JCOEF)pred;
        }
        /* AC11 */
        if ((Al = coef_bits[4]) != 0 && workspace[9] == 0) {
          num = Q00 * (change_dc ?
                (-DC01 + DC05 + 9 * DC07 - 9 * DC09 - 9 * DC17 +
                 9 * DC19 + DC21 - DC25) :
                (DC10 + DC16 - 10 * DC17 + 10 * DC19 - DC02 - DC20 + DC22 -
                 DC24 + DC04 - DC06 + 10 * DC07 - 10 * DC09));
          if (num >= 0) {
            pred = (int)(((Q11 << 7) + num) / (Q11 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
          } else {
            pred = (int)(((Q11 << 7) - num) / (Q11 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
            pred = -pred;
          }
          workspace[9] = (JCOEF)pred;
        }
        /* AC02 */
        if ((Al = coef_bits[5]) != 0 && workspace[2] == 0) {
          num = Q00 * (change_dc ?
                (2 * DC07 - 5 * DC08 + 2 * DC09 + DC11 + 7 * DC12 - 14 * DC13 +
                 7 * DC14 + DC15 + 2 * DC17 - 5 * DC18 + 2 * DC19) :
                (-DC11 + 13 * DC12 - 24 * DC13 + 13 * DC14 - DC15));
          if (num >= 0) {
            pred = (int)(((Q02 << 7) + num) / (Q02 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
          } else {
            pred = (int)(((Q02 << 7) - num) / (Q02 << 8));
            if (Al > 0 && pred >= (1 << Al))
              pred = (1 << Al) - 1;
            pred = -pred;
          }
          workspace[2] = (JCOEF)pred;
        }
        if (change_dc) {
          /* AC03 */
          if ((Al = coef_bits[6]) != 0 && workspace[3] == 0) {
            num = Q00 * (DC07 - DC09 + 2 * DC12 - 2 * DC14 + DC17 - DC19);
            if (num >= 0) {
              pred = (int)(((Q03 << 7) + num) / (Q03 << 8));
              if (Al > 0 && pred >= (1 << Al))
                pred = (1 << Al) - 1;
            } else {
              pred = (int)(((Q03 << 7) - num) / (Q03 << 8));
              if (Al > 0 && pred >= (1 << Al))
                pred = (1 << Al) - 1;
              pred = -pred;
            }
            workspace[3] = (JCOEF)pred;
          }
          /* AC12 */
          if ((Al = coef_bits[7]) != 0 && workspace[10] == 0) {
            num = Q00 * (DC07 - 3 * DC08 + DC09 - DC17 + 3 * DC18 - DC19);
            if (num >= 0) {
              pred = (int)(((Q12 << 7) + num) / (Q12 << 8));
              if (Al > 0 && pred >= (1 << Al))
                pred = (1 << Al) - 1;
            } else {
              pred = (int)(((Q12 << 7) - num) / (Q12 << 8));
              if (Al > 0 && pred >= (1 << Al))
                pred = (1 << Al) - 1;
              pred = -pred;
            }
            workspace[10] = (JCOEF)pred;
          }
          /* AC21 */
          if ((Al = coef_bits[8]) != 0 && workspace[17] == 0) {
            num = Q00 * (DC07 - DC09 - 3 * DC12 + 3 * DC14 + DC17 - DC19);
            if (num >= 0) {
              pred = (int)(((Q21 << 7) + num) / (Q21 << 8));
              if (Al > 0 && pred >= (1 << Al))
                pred = (1 << Al) - 1;
            } else {
              pred = (int)(((Q21 << 7) - num) / (Q21 << 8));
              if (Al > 0 && pred >= (1 << Al))
                pred = (1 << Al) - 1;
              pred = -pred;
            }
            workspace[17] = (JCOEF)pred;
          }
          /* AC30 */
          if ((Al = coef_bits[9]) != 0 && workspace[24] == 0) {
            num = Q00 * (DC07 + 2 * DC08 + DC09 - DC17 - 2 * DC18 - DC19);
            if (num >= 0) {
              pred = (int)(((Q30 << 7) + num) / (Q30 << 8));
              if (Al > 0 && pred >= (1 << Al))
                pred = (1 << Al) - 1;
            } else {
              pred = (int)(((Q30 << 7) - num) / (Q30 << 8));
              if (Al > 0 && pred >= (1 << Al))
                pred = (1 << Al) - 1;
              pred = -pred;
            }
            workspace[24] = (JCOEF)pred;
          }
          /* coef_bits[0] is non-negative.  Otherwise this function would not
           * be called.
           */
          num = Q00 *
                (-2 * DC01 - 6 * DC02 - 8 * DC03 - 6 * DC04 - 2 * DC05 -
                 6 * DC06 + 6 * DC07 + 42 * DC08 + 6 * DC09 - 6 * DC10 -
                 8 * DC11 + 42 * DC12 + 152 * DC13 + 42 * DC14 - 8 * DC15 -
                 6 * DC16 + 6 * DC17 + 42 * DC18 + 6 * DC19 - 6 * DC20 -
                 2 * DC21 - 6 * DC22 - 8 * DC23 - 6 * DC24 - 2 * DC25);
          if (num >= 0) {
            pred = (int)(((Q00 << 7) + num) / (Q00 << 8));
          } else {
            pred = (int)(((Q00 << 7) - num) / (Q00 << 8));
            pred = -pred;
          }
          workspace[0] = (JCOEF)pred;
        }  /* change_dc */

        /* OK, do the IDCT */
        (*inverse_DCT) (cinfo, compptr, (JCOEFPTR)workspace, output_ptr,
                        output_col);
        /* Advance for next column */
        DC01 = DC02;  DC02 = DC03;  DC03 = DC04;  DC04 = DC05;
        DC06 = DC07;  DC07 = DC08;  DC08 = DC09;  DC09 = DC10;
        DC11 = DC12;  DC12 = DC13;  DC13 = DC14;  DC14 = DC15;
        DC16 = DC17;  DC17 = DC18;  DC18 = DC19;  DC19 = DC20;
        DC21 = DC22;  DC22 = DC23;  DC23 = DC24;  DC24 = DC25;
        buffer_ptr++, prev_block_row++, next_block_row++,
          prev_prev_block_row++, next_next_block_row++;
        output_col += compptr->_DCT_scaled_size;
      }
      output_ptr += compptr->_DCT_scaled_size;
    }
  }

  if (++(cinfo->output_iMCU_row) < cinfo->total_iMCU_rows)
    return JPEG_ROW_COMPLETED;
  return JPEG_SCAN_COMPLETED;
}

#endif /* BLOCK_SMOOTHING_SUPPORTED */


/*
 * Initialize coefficient buffer controller.
 */

GLOBAL(void)
_jinit_d_coef_controller(j_decompress_ptr cinfo, boolean need_full_buffer)
{
  my_coef_ptr coef;

  if (cinfo->data_precision != BITS_IN_JSAMPLE)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  coef = (my_coef_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(my_coef_controller));
  cinfo->coef = (struct jpeg_d_coef_controller *)coef;
  coef->pub.start_input_pass = start_input_pass;
  coef->pub.start_output_pass = start_output_pass;
#ifdef BLOCK_SMOOTHING_SUPPORTED
  coef->coef_bits_latch = NULL;
#endif

  /* Create the coefficient buffer. */
  if (need_full_buffer) {
#ifdef D_MULTISCAN_FILES_SUPPORTED
    /* Allocate a full-image virtual array for each component, */
    /* padded to a multiple of samp_factor DCT blocks in each direction. */
    /* Note we ask for a pre-zeroed array. */
    int ci, access_rows;
    jpeg_component_info *compptr;

    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
         ci++, compptr++) {
      access_rows = compptr->v_samp_factor;
#ifdef BLOCK_SMOOTHING_SUPPORTED
      /* If block smoothing could be used, need a bigger window */
      if (cinfo->progressive_mode)
        access_rows *= 5;
#endif
      coef->whole_image[ci] = (*cinfo->mem->request_virt_barray)
        ((j_common_ptr)cinfo, JPOOL_IMAGE, TRUE,
         (JDIMENSION)jround_up((long)compptr->width_in_blocks,
                               (long)compptr->h_samp_factor),
         (JDIMENSION)jround_up((long)compptr->height_in_blocks,
                               (long)compptr->v_samp_factor),
         (JDIMENSION)access_rows);
    }
    coef->pub.consume_data = consume_data;
    coef->pub._decompress_data = decompress_data;
    coef->pub.coef_arrays = coef->whole_image; /* link to virtual arrays */
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
  } else {
    /* We only need a single-MCU buffer. */
    JBLOCKROW buffer;
    int i;

    buffer = (JBLOCKROW)
      (*cinfo->mem->alloc_large) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  D_MAX_BLOCKS_IN_MCU * sizeof(JBLOCK));
    for (i = 0; i < D_MAX_BLOCKS_IN_MCU; i++) {
      coef->MCU_buffer[i] = buffer + i;
    }
    coef->pub.consume_data = dummy_consume_data;
    coef->pub._decompress_data = decompress_onepass;
    coef->pub.coef_arrays = NULL; /* flag for no virtual arrays */
  }

  /* Allocate the workspace buffer */
  coef->workspace = (JCOEF *)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(JCOEF) * DCTSIZE2);
}
