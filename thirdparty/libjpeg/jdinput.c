/*
 * jdinput.c
 *
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2002-2020 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains input control logic for the JPEG decompressor.
 * These routines are concerned with controlling the decompressor's input
 * processing (marker reading and coefficient decoding).  The actual input
 * reading is done in jdmarker.c, jdhuff.c, and jdarith.c.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


/* Private state */

typedef struct {
  struct jpeg_input_controller pub; /* public fields */

  int inheaders;		/* Nonzero until first SOS is reached */
} my_input_controller;

typedef my_input_controller * my_inputctl_ptr;


/* Forward declarations */
METHODDEF(int) consume_markers JPP((j_decompress_ptr cinfo));


/*
 * Routines to calculate various quantities related to the size of the image.
 */


/*
 * Compute output image dimensions and related values.
 * NOTE: this is exported for possible use by application.
 * Hence it mustn't do anything that can't be done twice.
 */

GLOBAL(void)
jpeg_core_output_dimensions (j_decompress_ptr cinfo)
/* Do computations that are needed before master selection phase.
 * This function is used for transcoding and full decompression.
 */
{
#ifdef IDCT_SCALING_SUPPORTED
  int ci;
  jpeg_component_info *compptr;

  /* Compute actual output image dimensions and DCT scaling choices. */
  if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom) {
    /* Provide 1/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 1;
    cinfo->min_DCT_v_scaled_size = 1;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 2) {
    /* Provide 2/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 2L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 2L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 2;
    cinfo->min_DCT_v_scaled_size = 2;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 3) {
    /* Provide 3/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 3L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 3L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 3;
    cinfo->min_DCT_v_scaled_size = 3;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 4) {
    /* Provide 4/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 4L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 4L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 4;
    cinfo->min_DCT_v_scaled_size = 4;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 5) {
    /* Provide 5/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 5L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 5L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 5;
    cinfo->min_DCT_v_scaled_size = 5;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 6) {
    /* Provide 6/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 6L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 6L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 6;
    cinfo->min_DCT_v_scaled_size = 6;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 7) {
    /* Provide 7/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 7L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 7L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 7;
    cinfo->min_DCT_v_scaled_size = 7;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 8) {
    /* Provide 8/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 8L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 8L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 8;
    cinfo->min_DCT_v_scaled_size = 8;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 9) {
    /* Provide 9/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 9L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 9L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 9;
    cinfo->min_DCT_v_scaled_size = 9;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 10) {
    /* Provide 10/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 10L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 10L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 10;
    cinfo->min_DCT_v_scaled_size = 10;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 11) {
    /* Provide 11/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 11L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 11L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 11;
    cinfo->min_DCT_v_scaled_size = 11;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 12) {
    /* Provide 12/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 12L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 12L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 12;
    cinfo->min_DCT_v_scaled_size = 12;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 13) {
    /* Provide 13/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 13L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 13L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 13;
    cinfo->min_DCT_v_scaled_size = 13;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 14) {
    /* Provide 14/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 14L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 14L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 14;
    cinfo->min_DCT_v_scaled_size = 14;
  } else if (cinfo->scale_num * cinfo->block_size <= cinfo->scale_denom * 15) {
    /* Provide 15/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 15L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 15L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 15;
    cinfo->min_DCT_v_scaled_size = 15;
  } else {
    /* Provide 16/block_size scaling */
    cinfo->output_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * 16L, (long) cinfo->block_size);
    cinfo->output_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * 16L, (long) cinfo->block_size);
    cinfo->min_DCT_h_scaled_size = 16;
    cinfo->min_DCT_v_scaled_size = 16;
  }

  /* Recompute dimensions of components */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    compptr->DCT_h_scaled_size = cinfo->min_DCT_h_scaled_size;
    compptr->DCT_v_scaled_size = cinfo->min_DCT_v_scaled_size;
  }

#else /* !IDCT_SCALING_SUPPORTED */

  /* Hardwire it to "no scaling" */
  cinfo->output_width = cinfo->image_width;
  cinfo->output_height = cinfo->image_height;
  /* initial_setup has already initialized DCT_scaled_size,
   * and has computed unscaled downsampled_width and downsampled_height.
   */

#endif /* IDCT_SCALING_SUPPORTED */
}


LOCAL(void)
initial_setup (j_decompress_ptr cinfo)
/* Called once, when first SOS marker is reached */
{
  int ci;
  jpeg_component_info *compptr;

  /* Make sure image isn't bigger than I can handle */
  if ((long) cinfo->image_height > (long) JPEG_MAX_DIMENSION ||
      (long) cinfo->image_width > (long) JPEG_MAX_DIMENSION)
    ERREXIT1(cinfo, JERR_IMAGE_TOO_BIG, (unsigned int) JPEG_MAX_DIMENSION);

  /* Only 8 to 12 bits data precision are supported for DCT based JPEG */
  if (cinfo->data_precision < 8 || cinfo->data_precision > 12)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Check that number of components won't exceed internal array sizes */
  if (cinfo->num_components > MAX_COMPONENTS)
    ERREXIT2(cinfo, JERR_COMPONENT_COUNT, cinfo->num_components,
	     MAX_COMPONENTS);

  /* Compute maximum sampling factors; check factor validity */
  cinfo->max_h_samp_factor = 1;
  cinfo->max_v_samp_factor = 1;
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    if (compptr->h_samp_factor<=0 || compptr->h_samp_factor>MAX_SAMP_FACTOR ||
	compptr->v_samp_factor<=0 || compptr->v_samp_factor>MAX_SAMP_FACTOR)
      ERREXIT(cinfo, JERR_BAD_SAMPLING);
    cinfo->max_h_samp_factor = MAX(cinfo->max_h_samp_factor,
				   compptr->h_samp_factor);
    cinfo->max_v_samp_factor = MAX(cinfo->max_v_samp_factor,
				   compptr->v_samp_factor);
  }

  /* Derive block_size, natural_order, and lim_Se */
  if (cinfo->is_baseline || (cinfo->progressive_mode &&
      cinfo->comps_in_scan)) { /* no pseudo SOS marker */
    cinfo->block_size = DCTSIZE;
    cinfo->natural_order = jpeg_natural_order;
    cinfo->lim_Se = DCTSIZE2-1;
  } else
    switch (cinfo->Se) {
    case (1*1-1):
      cinfo->block_size = 1;
      cinfo->natural_order = jpeg_natural_order; /* not needed */
      cinfo->lim_Se = cinfo->Se;
      break;
    case (2*2-1):
      cinfo->block_size = 2;
      cinfo->natural_order = jpeg_natural_order2;
      cinfo->lim_Se = cinfo->Se;
      break;
    case (3*3-1):
      cinfo->block_size = 3;
      cinfo->natural_order = jpeg_natural_order3;
      cinfo->lim_Se = cinfo->Se;
      break;
    case (4*4-1):
      cinfo->block_size = 4;
      cinfo->natural_order = jpeg_natural_order4;
      cinfo->lim_Se = cinfo->Se;
      break;
    case (5*5-1):
      cinfo->block_size = 5;
      cinfo->natural_order = jpeg_natural_order5;
      cinfo->lim_Se = cinfo->Se;
      break;
    case (6*6-1):
      cinfo->block_size = 6;
      cinfo->natural_order = jpeg_natural_order6;
      cinfo->lim_Se = cinfo->Se;
      break;
    case (7*7-1):
      cinfo->block_size = 7;
      cinfo->natural_order = jpeg_natural_order7;
      cinfo->lim_Se = cinfo->Se;
      break;
    case (8*8-1):
      cinfo->block_size = 8;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    case (9*9-1):
      cinfo->block_size = 9;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    case (10*10-1):
      cinfo->block_size = 10;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    case (11*11-1):
      cinfo->block_size = 11;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    case (12*12-1):
      cinfo->block_size = 12;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    case (13*13-1):
      cinfo->block_size = 13;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    case (14*14-1):
      cinfo->block_size = 14;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    case (15*15-1):
      cinfo->block_size = 15;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    case (16*16-1):
      cinfo->block_size = 16;
      cinfo->natural_order = jpeg_natural_order;
      cinfo->lim_Se = DCTSIZE2-1;
      break;
    default:
      ERREXIT4(cinfo, JERR_BAD_PROGRESSION,
	       cinfo->Ss, cinfo->Se, cinfo->Ah, cinfo->Al);
    }

  /* We initialize DCT_scaled_size and min_DCT_scaled_size to block_size.
   * In the full decompressor,
   * this will be overridden by jpeg_calc_output_dimensions in jdmaster.c;
   * but in the transcoder,
   * jpeg_calc_output_dimensions is not used, so we must do it here.
   */
  cinfo->min_DCT_h_scaled_size = cinfo->block_size;
  cinfo->min_DCT_v_scaled_size = cinfo->block_size;

  /* Compute dimensions of components */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    compptr->DCT_h_scaled_size = cinfo->block_size;
    compptr->DCT_v_scaled_size = cinfo->block_size;
    /* Size in DCT blocks */
    compptr->width_in_blocks = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * (long) compptr->h_samp_factor,
		    (long) (cinfo->max_h_samp_factor * cinfo->block_size));
    compptr->height_in_blocks = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * (long) compptr->v_samp_factor,
		    (long) (cinfo->max_v_samp_factor * cinfo->block_size));
    /* downsampled_width and downsampled_height will also be overridden by
     * jdmaster.c if we are doing full decompression.  The transcoder library
     * doesn't use these values, but the calling application might.
     */
    /* Size in samples */
    compptr->downsampled_width = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width * (long) compptr->h_samp_factor,
		    (long) cinfo->max_h_samp_factor);
    compptr->downsampled_height = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_height * (long) compptr->v_samp_factor,
		    (long) cinfo->max_v_samp_factor);
    /* Mark component needed, until color conversion says otherwise */
    compptr->component_needed = TRUE;
    /* Mark no quantization table yet saved for component */
    compptr->quant_table = NULL;
  }

  /* Compute number of fully interleaved MCU rows. */
  cinfo->total_iMCU_rows = (JDIMENSION)
    jdiv_round_up((long) cinfo->image_height,
	          (long) (cinfo->max_v_samp_factor * cinfo->block_size));

  /* Decide whether file contains multiple scans */
  if (cinfo->comps_in_scan < cinfo->num_components || cinfo->progressive_mode)
    cinfo->inputctl->has_multiple_scans = TRUE;
  else
    cinfo->inputctl->has_multiple_scans = FALSE;
}


LOCAL(void)
per_scan_setup (j_decompress_ptr cinfo)
/* Do computations that are needed before processing a JPEG scan */
/* cinfo->comps_in_scan and cinfo->cur_comp_info[] were set from SOS marker */
{
  int ci, mcublks, tmp;
  jpeg_component_info *compptr;

  if (cinfo->comps_in_scan == 1) {

    /* Noninterleaved (single-component) scan */
    compptr = cinfo->cur_comp_info[0];

    /* Overall image size in MCUs */
    cinfo->MCUs_per_row = compptr->width_in_blocks;
    cinfo->MCU_rows_in_scan = compptr->height_in_blocks;

    /* For noninterleaved scan, always one block per MCU */
    compptr->MCU_width = 1;
    compptr->MCU_height = 1;
    compptr->MCU_blocks = 1;
    compptr->MCU_sample_width = compptr->DCT_h_scaled_size;
    compptr->last_col_width = 1;
    /* For noninterleaved scans, it is convenient to define last_row_height
     * as the number of block rows present in the last iMCU row.
     */
    tmp = (int) (compptr->height_in_blocks % compptr->v_samp_factor);
    if (tmp == 0) tmp = compptr->v_samp_factor;
    compptr->last_row_height = tmp;

    /* Prepare array describing MCU composition */
    cinfo->blocks_in_MCU = 1;
    cinfo->MCU_membership[0] = 0;

  } else {

    /* Interleaved (multi-component) scan */
    if (cinfo->comps_in_scan <= 0 || cinfo->comps_in_scan > MAX_COMPS_IN_SCAN)
      ERREXIT2(cinfo, JERR_COMPONENT_COUNT, cinfo->comps_in_scan,
	       MAX_COMPS_IN_SCAN);

    /* Overall image size in MCUs */
    cinfo->MCUs_per_row = (JDIMENSION)
      jdiv_round_up((long) cinfo->image_width,
		    (long) (cinfo->max_h_samp_factor * cinfo->block_size));
    cinfo->MCU_rows_in_scan = cinfo->total_iMCU_rows;

    cinfo->blocks_in_MCU = 0;

    for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
      compptr = cinfo->cur_comp_info[ci];
      /* Sampling factors give # of blocks of component in each MCU */
      compptr->MCU_width = compptr->h_samp_factor;
      compptr->MCU_height = compptr->v_samp_factor;
      compptr->MCU_blocks = compptr->MCU_width * compptr->MCU_height;
      compptr->MCU_sample_width = compptr->MCU_width * compptr->DCT_h_scaled_size;
      /* Figure number of non-dummy blocks in last MCU column & row */
      tmp = (int) (compptr->width_in_blocks % compptr->MCU_width);
      if (tmp == 0) tmp = compptr->MCU_width;
      compptr->last_col_width = tmp;
      tmp = (int) (compptr->height_in_blocks % compptr->MCU_height);
      if (tmp == 0) tmp = compptr->MCU_height;
      compptr->last_row_height = tmp;
      /* Prepare array describing MCU composition */
      mcublks = compptr->MCU_blocks;
      if (cinfo->blocks_in_MCU + mcublks > D_MAX_BLOCKS_IN_MCU)
	ERREXIT(cinfo, JERR_BAD_MCU_SIZE);
      while (mcublks-- > 0) {
	cinfo->MCU_membership[cinfo->blocks_in_MCU++] = ci;
      }
    }

  }
}


/*
 * Save away a copy of the Q-table referenced by each component present
 * in the current scan, unless already saved during a prior scan.
 *
 * In a multiple-scan JPEG file, the encoder could assign different components
 * the same Q-table slot number, but change table definitions between scans
 * so that each component uses a different Q-table.  (The IJG encoder is not
 * currently capable of doing this, but other encoders might.)  Since we want
 * to be able to dequantize all the components at the end of the file, this
 * means that we have to save away the table actually used for each component.
 * We do this by copying the table at the start of the first scan containing
 * the component.
 * The JPEG spec prohibits the encoder from changing the contents of a Q-table
 * slot between scans of a component using that slot.  If the encoder does so
 * anyway, this decoder will simply use the Q-table values that were current
 * at the start of the first scan for the component.
 *
 * The decompressor output side looks only at the saved quant tables,
 * not at the current Q-table slots.
 */

LOCAL(void)
latch_quant_tables (j_decompress_ptr cinfo)
{
  int ci, qtblno;
  jpeg_component_info *compptr;
  JQUANT_TBL * qtbl;

  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    /* No work if we already saved Q-table for this component */
    if (compptr->quant_table != NULL)
      continue;
    /* Make sure specified quantization table is present */
    qtblno = compptr->quant_tbl_no;
    if (qtblno < 0 || qtblno >= NUM_QUANT_TBLS ||
	cinfo->quant_tbl_ptrs[qtblno] == NULL)
      ERREXIT1(cinfo, JERR_NO_QUANT_TABLE, qtblno);
    /* OK, save away the quantization table */
    qtbl = (JQUANT_TBL *) (*cinfo->mem->alloc_small)
      ((j_common_ptr) cinfo, JPOOL_IMAGE, SIZEOF(JQUANT_TBL));
    MEMCOPY(qtbl, cinfo->quant_tbl_ptrs[qtblno], SIZEOF(JQUANT_TBL));
    compptr->quant_table = qtbl;
  }
}


/*
 * Initialize the input modules to read a scan of compressed data.
 * The first call to this is done by jdmaster.c after initializing
 * the entire decompressor (during jpeg_start_decompress).
 * Subsequent calls come from consume_markers, below.
 */

METHODDEF(void)
start_input_pass (j_decompress_ptr cinfo)
{
  per_scan_setup(cinfo);
  latch_quant_tables(cinfo);
  (*cinfo->entropy->start_pass) (cinfo);
  (*cinfo->coef->start_input_pass) (cinfo);
  cinfo->inputctl->consume_input = cinfo->coef->consume_data;
}


/*
 * Finish up after inputting a compressed-data scan.
 * This is called by the coefficient controller after it's read all
 * the expected data of the scan.
 */

METHODDEF(void)
finish_input_pass (j_decompress_ptr cinfo)
{
  (*cinfo->entropy->finish_pass) (cinfo);
  cinfo->inputctl->consume_input = consume_markers;
}


/*
 * Read JPEG markers before, between, or after compressed-data scans.
 * Change state as necessary when a new scan is reached.
 * Return value is JPEG_SUSPENDED, JPEG_REACHED_SOS, or JPEG_REACHED_EOI.
 *
 * The consume_input method pointer points either here or to the
 * coefficient controller's consume_data routine, depending on whether
 * we are reading a compressed data segment or inter-segment markers.
 *
 * Note: This function should NOT return a pseudo SOS marker (with zero
 * component number) to the caller.  A pseudo marker received by
 * read_markers is processed and then skipped for other markers.
 */

METHODDEF(int)
consume_markers (j_decompress_ptr cinfo)
{
  my_inputctl_ptr inputctl = (my_inputctl_ptr) cinfo->inputctl;
  int val;

  if (inputctl->pub.eoi_reached) /* After hitting EOI, read no further */
    return JPEG_REACHED_EOI;

  for (;;) {			/* Loop to pass pseudo SOS marker */
    val = (*cinfo->marker->read_markers) (cinfo);

    switch (val) {
    case JPEG_REACHED_SOS:	/* Found SOS */
      if (inputctl->inheaders) { /* 1st SOS */
	if (inputctl->inheaders == 1)
	  initial_setup(cinfo);
	if (cinfo->comps_in_scan == 0) { /* pseudo SOS marker */
	  inputctl->inheaders = 2;
	  break;
	}
	inputctl->inheaders = 0;
	/* Note: start_input_pass must be called by jdmaster.c
	 * before any more input can be consumed.  jdapimin.c is
	 * responsible for enforcing this sequencing.
	 */
      } else {			/* 2nd or later SOS marker */
	if (! inputctl->pub.has_multiple_scans)
	  ERREXIT(cinfo, JERR_EOI_EXPECTED); /* Oops, I wasn't expecting this! */
	if (cinfo->comps_in_scan == 0) /* unexpected pseudo SOS marker */
	  break;
	start_input_pass(cinfo);
      }
      return val;
    case JPEG_REACHED_EOI:	/* Found EOI */
      inputctl->pub.eoi_reached = TRUE;
      if (inputctl->inheaders) { /* Tables-only datastream, apparently */
	if (cinfo->marker->saw_SOF)
	  ERREXIT(cinfo, JERR_SOF_NO_SOS);
      } else {
	/* Prevent infinite loop in coef ctlr's decompress_data routine
	 * if user set output_scan_number larger than number of scans.
	 */
	if (cinfo->output_scan_number > cinfo->input_scan_number)
	  cinfo->output_scan_number = cinfo->input_scan_number;
      }
      return val;
    case JPEG_SUSPENDED:
      return val;
    default:
      return val;
    }
  }
}


/*
 * Reset state to begin a fresh datastream.
 */

METHODDEF(void)
reset_input_controller (j_decompress_ptr cinfo)
{
  my_inputctl_ptr inputctl = (my_inputctl_ptr) cinfo->inputctl;

  inputctl->pub.consume_input = consume_markers;
  inputctl->pub.has_multiple_scans = FALSE; /* "unknown" would be better */
  inputctl->pub.eoi_reached = FALSE;
  inputctl->inheaders = 1;
  /* Reset other modules */
  (*cinfo->err->reset_error_mgr) ((j_common_ptr) cinfo);
  (*cinfo->marker->reset_marker_reader) (cinfo);
  /* Reset progression state -- would be cleaner if entropy decoder did this */
  cinfo->coef_bits = NULL;
}


/*
 * Initialize the input controller module.
 * This is called only once, when the decompression object is created.
 */

GLOBAL(void)
jinit_input_controller (j_decompress_ptr cinfo)
{
  my_inputctl_ptr inputctl;

  /* Create subobject in permanent pool */
  inputctl = (my_inputctl_ptr) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_PERMANENT, SIZEOF(my_input_controller));
  cinfo->inputctl = &inputctl->pub;
  /* Initialize method pointers */
  inputctl->pub.consume_input = consume_markers;
  inputctl->pub.reset_input_controller = reset_input_controller;
  inputctl->pub.start_input_pass = start_input_pass;
  inputctl->pub.finish_input_pass = finish_input_pass;
  /* Initialize state: can't use reset_input_controller since we don't
   * want to try to reset other modules yet.
   */
  inputctl->pub.has_multiple_scans = FALSE; /* "unknown" would be better */
  inputctl->pub.eoi_reached = FALSE;
  inputctl->inheaders = 1;
}
