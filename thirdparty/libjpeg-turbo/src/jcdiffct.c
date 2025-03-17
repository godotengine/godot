/*
 * jcdiffct.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1997, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, 2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains the difference buffer controller for compression.
 * This controller is the top level of the lossless JPEG compressor proper.
 * The difference buffer lies between the prediction/differencing and entropy
 * encoding steps.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"            /* Private declarations for lossless codec */


#ifdef C_LOSSLESS_SUPPORTED

/* We use a full-image sample buffer when doing Huffman optimization,
 * and also for writing multiple-scan JPEG files.  In all cases, the
 * full-image buffer is filled during the first pass, and the scaling,
 * prediction and differencing steps are run during subsequent passes.
 */
#ifdef ENTROPY_OPT_SUPPORTED
#define FULL_SAMP_BUFFER_SUPPORTED
#else
#ifdef C_MULTISCAN_FILES_SUPPORTED
#define FULL_SAMP_BUFFER_SUPPORTED
#endif
#endif


/* Private buffer controller object */

typedef struct {
  struct jpeg_c_coef_controller pub; /* public fields */

  JDIMENSION iMCU_row_num;      /* iMCU row # within image */
  JDIMENSION mcu_ctr;           /* counts MCUs processed in current row */
  int MCU_vert_offset;          /* counts MCU rows within iMCU row */
  int MCU_rows_per_iMCU_row;    /* number of such rows needed */

  _JSAMPROW cur_row[MAX_COMPONENTS];    /* row of point-transformed samples */
  _JSAMPROW prev_row[MAX_COMPONENTS];   /* previous row of Pt'd samples */
  JDIFFARRAY diff_buf[MAX_COMPONENTS];  /* iMCU row of differences */

  /* In multi-pass modes, we need a virtual sample array for each component. */
  jvirt_sarray_ptr whole_image[MAX_COMPONENTS];
} my_diff_controller;

typedef my_diff_controller *my_diff_ptr;


/* Forward declarations */
METHODDEF(boolean) compress_data(j_compress_ptr cinfo, _JSAMPIMAGE input_buf);
#ifdef FULL_SAMP_BUFFER_SUPPORTED
METHODDEF(boolean) compress_first_pass(j_compress_ptr cinfo,
                                       _JSAMPIMAGE input_buf);
METHODDEF(boolean) compress_output(j_compress_ptr cinfo,
                                   _JSAMPIMAGE input_buf);
#endif


LOCAL(void)
start_iMCU_row(j_compress_ptr cinfo)
/* Reset within-iMCU-row counters for a new row */
{
  my_diff_ptr diff = (my_diff_ptr)cinfo->coef;

  /* In an interleaved scan, an MCU row is the same as an iMCU row.
   * In a noninterleaved scan, an iMCU row has v_samp_factor MCU rows.
   * But at the bottom of the image, process only what's left.
   */
  if (cinfo->comps_in_scan > 1) {
    diff->MCU_rows_per_iMCU_row = 1;
  } else {
    if (diff->iMCU_row_num < (cinfo->total_iMCU_rows-1))
      diff->MCU_rows_per_iMCU_row = cinfo->cur_comp_info[0]->v_samp_factor;
    else
      diff->MCU_rows_per_iMCU_row = cinfo->cur_comp_info[0]->last_row_height;
  }

  diff->mcu_ctr = 0;
  diff->MCU_vert_offset = 0;
}


/*
 * Initialize for a processing pass.
 */

METHODDEF(void)
start_pass_diff(j_compress_ptr cinfo, J_BUF_MODE pass_mode)
{
  my_diff_ptr diff = (my_diff_ptr)cinfo->coef;

  /* Because it is hitching a ride on the jpeg_forward_dct struct,
   * start_pass_lossless() will be called at the start of the initial pass.
   * This ensures that it will be called at the start of the Huffman
   * optimization and output passes as well.
   */
  if (pass_mode == JBUF_CRANK_DEST)
    (*cinfo->fdct->start_pass) (cinfo);

  diff->iMCU_row_num = 0;
  start_iMCU_row(cinfo);

  switch (pass_mode) {
  case JBUF_PASS_THRU:
    if (diff->whole_image[0] != NULL)
      ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    diff->pub._compress_data = compress_data;
    break;
#ifdef FULL_SAMP_BUFFER_SUPPORTED
  case JBUF_SAVE_AND_PASS:
    if (diff->whole_image[0] == NULL)
      ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    diff->pub._compress_data = compress_first_pass;
    break;
  case JBUF_CRANK_DEST:
    if (diff->whole_image[0] == NULL)
      ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    diff->pub._compress_data = compress_output;
    break;
#endif
  default:
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    break;
  }
}


#define SWAP_ROWS(rowa, rowb) { \
  _JSAMPROW temp = rowa; \
  rowa = rowb;  rowb = temp; \
}

/*
 * Process some data in the single-pass case.
 * We process the equivalent of one fully interleaved MCU row ("iMCU" row)
 * per call, ie, v_samp_factor rows for each component in the image.
 * Returns TRUE if the iMCU row is completed, FALSE if suspended.
 *
 * NB: input_buf contains a plane for each component in image,
 * which we index according to the component's SOF position.
 */

METHODDEF(boolean)
compress_data(j_compress_ptr cinfo, _JSAMPIMAGE input_buf)
{
  my_diff_ptr diff = (my_diff_ptr)cinfo->coef;
  lossless_comp_ptr losslessc = (lossless_comp_ptr)cinfo->fdct;
  JDIMENSION MCU_col_num;       /* index of current MCU within row */
  JDIMENSION MCU_count;         /* number of MCUs encoded */
  JDIMENSION last_iMCU_row = cinfo->total_iMCU_rows - 1;
  int ci, compi, yoffset, samp_row, samp_rows, samps_across;
  jpeg_component_info *compptr;

  /* Loop to write as much as one whole iMCU row */
  for (yoffset = diff->MCU_vert_offset; yoffset < diff->MCU_rows_per_iMCU_row;
       yoffset++) {

    MCU_col_num = diff->mcu_ctr;

    /* Scale and predict each scanline of the MCU row separately.
     *
     * Note: We only do this if we are at the start of an MCU row, ie,
     * we don't want to reprocess a row suspended by the output.
     */
    if (MCU_col_num == 0) {
      for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
        compptr = cinfo->cur_comp_info[ci];
        compi = compptr->component_index;
        if (diff->iMCU_row_num < last_iMCU_row)
          samp_rows = compptr->v_samp_factor;
        else {
          /* NB: can't use last_row_height here, since may not be set! */
          samp_rows =
            (int)(compptr->height_in_blocks % compptr->v_samp_factor);
          if (samp_rows == 0) samp_rows = compptr->v_samp_factor;
          else {
            /* Fill dummy difference rows at the bottom edge with zeros, which
             * will encode to the smallest amount of data.
             */
            for (samp_row = samp_rows; samp_row < compptr->v_samp_factor;
                 samp_row++)
              memset(diff->diff_buf[compi][samp_row], 0,
                     jround_up((long)compptr->width_in_blocks,
                               (long)compptr->h_samp_factor) * sizeof(JDIFF));
          }
        }
        samps_across = compptr->width_in_blocks;

        for (samp_row = 0; samp_row < samp_rows; samp_row++) {
          (*losslessc->scaler_scale) (cinfo,
                                      input_buf[compi][samp_row],
                                      diff->cur_row[compi],
                                      samps_across);
          (*losslessc->predict_difference[compi])
            (cinfo, compi, diff->cur_row[compi], diff->prev_row[compi],
             diff->diff_buf[compi][samp_row], samps_across);
          SWAP_ROWS(diff->cur_row[compi], diff->prev_row[compi]);
        }
      }
    }
    /* Try to write the MCU row (or remaining portion of suspended MCU row). */
    MCU_count =
      (*cinfo->entropy->encode_mcus) (cinfo,
                                      diff->diff_buf, yoffset, MCU_col_num,
                                      cinfo->MCUs_per_row - MCU_col_num);
    if (MCU_count != cinfo->MCUs_per_row - MCU_col_num) {
      /* Suspension forced; update state counters and exit */
      diff->MCU_vert_offset = yoffset;
      diff->mcu_ctr += MCU_col_num;
      return FALSE;
    }
    /* Completed an MCU row, but perhaps not an iMCU row */
    diff->mcu_ctr = 0;
  }
  /* Completed the iMCU row, advance counters for next one */
  diff->iMCU_row_num++;
  start_iMCU_row(cinfo);
  return TRUE;
}


#ifdef FULL_SAMP_BUFFER_SUPPORTED

/*
 * Process some data in the first pass of a multi-pass case.
 * We process the equivalent of one fully interleaved MCU row ("iMCU" row)
 * per call, ie, v_samp_factor rows for each component in the image.
 * This amount of data is read from the source buffer and saved into the
 * virtual arrays.
 *
 * We must also emit the data to the compressor.  This is conveniently
 * done by calling compress_output() after we've loaded the current strip
 * of the virtual arrays.
 *
 * NB: input_buf contains a plane for each component in image.  All components
 * are loaded into the virtual arrays in this pass.  However, it may be that
 * only a subset of the components are emitted to the compressor during
 * this first pass; be careful about looking at the scan-dependent variables
 * (MCU dimensions, etc).
 */

METHODDEF(boolean)
compress_first_pass(j_compress_ptr cinfo, _JSAMPIMAGE input_buf)
{
  my_diff_ptr diff = (my_diff_ptr)cinfo->coef;
  JDIMENSION last_iMCU_row = cinfo->total_iMCU_rows - 1;
  JDIMENSION samps_across;
  int ci, samp_row, samp_rows;
  _JSAMPARRAY buffer;
  jpeg_component_info *compptr;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Align the virtual buffer for this component. */
    buffer = (_JSAMPARRAY)(*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, diff->whole_image[ci],
       diff->iMCU_row_num * compptr->v_samp_factor,
       (JDIMENSION)compptr->v_samp_factor, TRUE);

    /* Count non-dummy sample rows in this iMCU row. */
    if (diff->iMCU_row_num < last_iMCU_row)
      samp_rows = compptr->v_samp_factor;
    else {
      /* NB: can't use last_row_height here, since may not be set! */
      samp_rows = (int)(compptr->height_in_blocks % compptr->v_samp_factor);
      if (samp_rows == 0) samp_rows = compptr->v_samp_factor;
    }
    samps_across = compptr->width_in_blocks;

    /* Perform point transform scaling and prediction/differencing for all
     * non-dummy rows in this iMCU row.  Each call on these functions
     * processes a complete row of samples.
     */
    for (samp_row = 0; samp_row < samp_rows; samp_row++) {
      memcpy(buffer[samp_row], input_buf[ci][samp_row],
             samps_across * sizeof(_JSAMPLE));
    }
  }
  /* NB: compress_output will increment iMCU_row_num if successful.
   * A suspension return will result in redoing all the work above next time.
   */

  /* Emit data to the compressor, sharing code with subsequent passes */
  return compress_output(cinfo, input_buf);
}


/*
 * Process some data in subsequent passes of a multi-pass case.
 * We process the equivalent of one fully interleaved MCU row ("iMCU" row)
 * per call, ie, v_samp_factor rows for each component in the scan.
 * The data is obtained from the virtual arrays and fed to the compressor.
 * Returns TRUE if the iMCU row is completed, FALSE if suspended.
 *
 * NB: input_buf is ignored; it is likely to be a NULL pointer.
 */

METHODDEF(boolean)
compress_output(j_compress_ptr cinfo, _JSAMPIMAGE input_buf)
{
  my_diff_ptr diff = (my_diff_ptr)cinfo->coef;
  int ci, compi;
  _JSAMPARRAY buffer[MAX_COMPS_IN_SCAN];
  jpeg_component_info *compptr;

  /* Align the virtual buffers for the components used in this scan.
   * NB: during first pass, this is safe only because the buffers will
   * already be aligned properly, so jmemmgr.c won't need to do any I/O.
   */
  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    compi = compptr->component_index;
    buffer[compi] = (_JSAMPARRAY)(*cinfo->mem->access_virt_sarray)
      ((j_common_ptr)cinfo, diff->whole_image[compi],
       diff->iMCU_row_num * compptr->v_samp_factor,
       (JDIMENSION)compptr->v_samp_factor, FALSE);
  }

  return compress_data(cinfo, buffer);
}

#endif /* FULL_SAMP_BUFFER_SUPPORTED */


/*
 * Initialize difference buffer controller.
 */

GLOBAL(void)
_jinit_c_diff_controller(j_compress_ptr cinfo, boolean need_full_buffer)
{
  my_diff_ptr diff;
  int ci, row;
  jpeg_component_info *compptr;

#if BITS_IN_JSAMPLE == 8
  if (cinfo->data_precision > BITS_IN_JSAMPLE || cinfo->data_precision < 2)
#else
  if (cinfo->data_precision > BITS_IN_JSAMPLE ||
      cinfo->data_precision < BITS_IN_JSAMPLE - 3)
#endif
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  diff = (my_diff_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(my_diff_controller));
  cinfo->coef = (struct jpeg_c_coef_controller *)diff;
  diff->pub.start_pass = start_pass_diff;

  /* Create the prediction row buffers. */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    diff->cur_row[ci] = *(_JSAMPARRAY)(*cinfo->mem->alloc_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE,
       (JDIMENSION)jround_up((long)compptr->width_in_blocks,
                             (long)compptr->h_samp_factor),
       (JDIMENSION)1);
    diff->prev_row[ci] = *(_JSAMPARRAY)(*cinfo->mem->alloc_sarray)
      ((j_common_ptr)cinfo, JPOOL_IMAGE,
       (JDIMENSION)jround_up((long)compptr->width_in_blocks,
                             (long)compptr->h_samp_factor),
       (JDIMENSION)1);
  }

  /* Create the difference buffer. */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    diff->diff_buf[ci] =
      ALLOC_DARRAY(JPOOL_IMAGE,
                   (JDIMENSION)jround_up((long)compptr->width_in_blocks,
                                         (long)compptr->h_samp_factor),
                   (JDIMENSION)compptr->v_samp_factor);
    /* Prefill difference rows with zeros.  We do this because only actual
     * data is placed in the buffers during prediction/differencing, leaving
     * any dummy differences at the right edge as zeros, which will encode
     * to the smallest amount of data.
     */
    for (row = 0; row < compptr->v_samp_factor; row++)
      memset(diff->diff_buf[ci][row], 0,
             jround_up((long)compptr->width_in_blocks,
                       (long)compptr->h_samp_factor) * sizeof(JDIFF));
  }

  /* Create the sample buffer. */
  if (need_full_buffer) {
#ifdef FULL_SAMP_BUFFER_SUPPORTED
    /* Allocate a full-image virtual array for each component, */
    /* padded to a multiple of samp_factor differences in each direction. */
    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
         ci++, compptr++) {
      diff->whole_image[ci] = (*cinfo->mem->request_virt_sarray)
        ((j_common_ptr)cinfo, JPOOL_IMAGE, FALSE,
         (JDIMENSION)jround_up((long)compptr->width_in_blocks,
                               (long)compptr->h_samp_factor),
         (JDIMENSION)jround_up((long)compptr->height_in_blocks,
                               (long)compptr->v_samp_factor),
         (JDIMENSION)compptr->v_samp_factor);
    }
#else
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
#endif
  } else
    diff->whole_image[0] = NULL; /* flag for no virtual arrays */
}

#endif /* C_LOSSLESS_SUPPORTED */
