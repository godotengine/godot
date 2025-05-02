/*
 * jdmainct.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2010, 2016, 2022, 2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains the main buffer controller for decompression.
 * The main buffer lies between the JPEG decompressor proper and the
 * post-processor; it holds downsampled data in the JPEG colorspace.
 *
 * Note that this code is bypassed in raw-data mode, since the application
 * supplies the equivalent of the main buffer in that case.
 */

#include "jinclude.h"
#include "jdmainct.h"


#if BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED)

/*
 * In the current system design, the main buffer need never be a full-image
 * buffer; any full-height buffers will be found inside the coefficient,
 * difference, or postprocessing controllers.  Nonetheless, the main controller
 * is not trivial.  Its responsibility is to provide context rows for
 * upsampling/rescaling, and doing this in an efficient fashion is a bit
 * tricky.
 *
 * Postprocessor input data is counted in "row groups".  A row group
 * is defined to be (v_samp_factor * DCT_scaled_size / min_DCT_scaled_size)
 * sample rows of each component.  (We require DCT_scaled_size values to be
 * chosen such that these numbers are integers.  In practice DCT_scaled_size
 * values will likely be powers of two, so we actually have the stronger
 * condition that DCT_scaled_size / min_DCT_scaled_size is an integer.)
 * Upsampling will typically produce max_v_samp_factor pixel rows from each
 * row group (times any additional scale factor that the upsampler is
 * applying).
 *
 * The coefficient or difference controller will deliver data to us one iMCU
 * row at a time; each iMCU row contains v_samp_factor * DCT_scaled_size sample
 * rows, or exactly min_DCT_scaled_size row groups.  (This amount of data
 * corresponds to one row of MCUs when the image is fully interleaved.)  Note
 * that the number of sample rows varies across components, but the number of
 * row groups does not.  Some garbage sample rows may be included in the last
 * iMCU row at the bottom of the image.
 *
 * Depending on the vertical scaling algorithm used, the upsampler may need
 * access to the sample row(s) above and below its current input row group.
 * The upsampler is required to set need_context_rows TRUE at global selection
 * time if so.  When need_context_rows is FALSE, this controller can simply
 * obtain one iMCU row at a time from the coefficient or difference controller
 * and dole it out as row groups to the postprocessor.
 *
 * When need_context_rows is TRUE, this controller guarantees that the buffer
 * passed to postprocessing contains at least one row group's worth of samples
 * above and below the row group(s) being processed.  Note that the context
 * rows "above" the first passed row group appear at negative row offsets in
 * the passed buffer.  At the top and bottom of the image, the required
 * context rows are manufactured by duplicating the first or last real sample
 * row; this avoids having special cases in the upsampling inner loops.
 *
 * The amount of context is fixed at one row group just because that's a
 * convenient number for this controller to work with.  The existing
 * upsamplers really only need one sample row of context.  An upsampler
 * supporting arbitrary output rescaling might wish for more than one row
 * group of context when shrinking the image; tough, we don't handle that.
 * (This is justified by the assumption that downsizing will be handled mostly
 * by adjusting the DCT_scaled_size values, so that the actual scale factor at
 * the upsample step needn't be much less than one.)
 *
 * To provide the desired context, we have to retain the last two row groups
 * of one iMCU row while reading in the next iMCU row.  (The last row group
 * can't be processed until we have another row group for its below-context,
 * and so we have to save the next-to-last group too for its above-context.)
 * We could do this most simply by copying data around in our buffer, but
 * that'd be very slow.  We can avoid copying any data by creating a rather
 * strange pointer structure.  Here's how it works.  We allocate a workspace
 * consisting of M+2 row groups (where M = min_DCT_scaled_size is the number
 * of row groups per iMCU row).  We create two sets of redundant pointers to
 * the workspace.  Labeling the physical row groups 0 to M+1, the synthesized
 * pointer lists look like this:
 *                   M+1                          M-1
 * master pointer --> 0         master pointer --> 0
 *                    1                            1
 *                   ...                          ...
 *                   M-3                          M-3
 *                   M-2                           M
 *                   M-1                          M+1
 *                    M                           M-2
 *                   M+1                          M-1
 *                    0                            0
 * We read alternate iMCU rows using each master pointer; thus the last two
 * row groups of the previous iMCU row remain un-overwritten in the workspace.
 * The pointer lists are set up so that the required context rows appear to
 * be adjacent to the proper places when we pass the pointer lists to the
 * upsampler.
 *
 * The above pictures describe the normal state of the pointer lists.
 * At top and bottom of the image, we diddle the pointer lists to duplicate
 * the first or last sample row as necessary (this is cheaper than copying
 * sample rows around).
 *
 * This scheme breaks down if M < 2, ie, min_DCT_scaled_size is 1.  In that
 * situation each iMCU row provides only one row group so the buffering logic
 * must be different (eg, we must read two iMCU rows before we can emit the
 * first row group).  For now, we simply do not support providing context
 * rows when min_DCT_scaled_size is 1.  That combination seems unlikely to
 * be worth providing --- if someone wants a 1/8th-size preview, they probably
 * want it quick and dirty, so a context-free upsampler is sufficient.
 */


/* Forward declarations */
METHODDEF(void) process_data_simple_main(j_decompress_ptr cinfo,
                                         _JSAMPARRAY output_buf,
                                         JDIMENSION *out_row_ctr,
                                         JDIMENSION out_rows_avail);
METHODDEF(void) process_data_context_main(j_decompress_ptr cinfo,
                                          _JSAMPARRAY output_buf,
                                          JDIMENSION *out_row_ctr,
                                          JDIMENSION out_rows_avail);
#ifdef QUANT_2PASS_SUPPORTED
METHODDEF(void) process_data_crank_post(j_decompress_ptr cinfo,
                                        _JSAMPARRAY output_buf,
                                        JDIMENSION *out_row_ctr,
                                        JDIMENSION out_rows_avail);
#endif


LOCAL(void)
alloc_funny_pointers(j_decompress_ptr cinfo)
/* Allocate space for the funny pointer lists.
 * This is done only once, not once per pass.
 */
{
  my_main_ptr main_ptr = (my_main_ptr)cinfo->main;
  int ci, rgroup;
  int M = cinfo->_min_DCT_scaled_size;
  jpeg_component_info *compptr;
  _JSAMPARRAY xbuf;

  /* Get top-level space for component array pointers.
   * We alloc both arrays with one call to save a few cycles.
   */
  main_ptr->xbuffer[0] = (_JSAMPIMAGE)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                cinfo->num_components * 2 *
                                sizeof(_JSAMPARRAY));
  main_ptr->xbuffer[1] = main_ptr->xbuffer[0] + cinfo->num_components;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    rgroup = (compptr->v_samp_factor * compptr->_DCT_scaled_size) /
      cinfo->_min_DCT_scaled_size; /* height of a row group of component */
    /* Get space for pointer lists --- M+4 row groups in each list.
     * We alloc both pointer lists with one call to save a few cycles.
     */
    xbuf = (_JSAMPARRAY)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  2 * (rgroup * (M + 4)) * sizeof(_JSAMPROW));
    xbuf += rgroup;             /* want one row group at negative offsets */
    main_ptr->xbuffer[0][ci] = xbuf;
    xbuf += rgroup * (M + 4);
    main_ptr->xbuffer[1][ci] = xbuf;
  }
}


LOCAL(void)
make_funny_pointers(j_decompress_ptr cinfo)
/* Create the funny pointer lists discussed in the comments above.
 * The actual workspace is already allocated (in main_ptr->buffer),
 * and the space for the pointer lists is allocated too.
 * This routine just fills in the curiously ordered lists.
 * This will be repeated at the beginning of each pass.
 */
{
  my_main_ptr main_ptr = (my_main_ptr)cinfo->main;
  int ci, i, rgroup;
  int M = cinfo->_min_DCT_scaled_size;
  jpeg_component_info *compptr;
  _JSAMPARRAY buf, xbuf0, xbuf1;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    rgroup = (compptr->v_samp_factor * compptr->_DCT_scaled_size) /
      cinfo->_min_DCT_scaled_size; /* height of a row group of component */
    xbuf0 = main_ptr->xbuffer[0][ci];
    xbuf1 = main_ptr->xbuffer[1][ci];
    /* First copy the workspace pointers as-is */
    buf = main_ptr->buffer[ci];
    for (i = 0; i < rgroup * (M + 2); i++) {
      xbuf0[i] = xbuf1[i] = buf[i];
    }
    /* In the second list, put the last four row groups in swapped order */
    for (i = 0; i < rgroup * 2; i++) {
      xbuf1[rgroup * (M - 2) + i] = buf[rgroup * M + i];
      xbuf1[rgroup * M + i] = buf[rgroup * (M - 2) + i];
    }
    /* The wraparound pointers at top and bottom will be filled later
     * (see set_wraparound_pointers, below).  Initially we want the "above"
     * pointers to duplicate the first actual data line.  This only needs
     * to happen in xbuffer[0].
     */
    for (i = 0; i < rgroup; i++) {
      xbuf0[i - rgroup] = xbuf0[0];
    }
  }
}


LOCAL(void)
set_bottom_pointers(j_decompress_ptr cinfo)
/* Change the pointer lists to duplicate the last sample row at the bottom
 * of the image.  whichptr indicates which xbuffer holds the final iMCU row.
 * Also sets rowgroups_avail to indicate number of nondummy row groups in row.
 */
{
  my_main_ptr main_ptr = (my_main_ptr)cinfo->main;
  int ci, i, rgroup, iMCUheight, rows_left;
  jpeg_component_info *compptr;
  _JSAMPARRAY xbuf;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Count sample rows in one iMCU row and in one row group */
    iMCUheight = compptr->v_samp_factor * compptr->_DCT_scaled_size;
    rgroup = iMCUheight / cinfo->_min_DCT_scaled_size;
    /* Count nondummy sample rows remaining for this component */
    rows_left = (int)(compptr->downsampled_height % (JDIMENSION)iMCUheight);
    if (rows_left == 0) rows_left = iMCUheight;
    /* Count nondummy row groups.  Should get same answer for each component,
     * so we need only do it once.
     */
    if (ci == 0) {
      main_ptr->rowgroups_avail = (JDIMENSION)((rows_left - 1) / rgroup + 1);
    }
    /* Duplicate the last real sample row rgroup*2 times; this pads out the
     * last partial rowgroup and ensures at least one full rowgroup of context.
     */
    xbuf = main_ptr->xbuffer[main_ptr->whichptr][ci];
    for (i = 0; i < rgroup * 2; i++) {
      xbuf[rows_left + i] = xbuf[rows_left - 1];
    }
  }
}


/*
 * Initialize for a processing pass.
 */

METHODDEF(void)
start_pass_main(j_decompress_ptr cinfo, J_BUF_MODE pass_mode)
{
  my_main_ptr main_ptr = (my_main_ptr)cinfo->main;

  switch (pass_mode) {
  case JBUF_PASS_THRU:
    if (cinfo->upsample->need_context_rows) {
      main_ptr->pub._process_data = process_data_context_main;
      make_funny_pointers(cinfo); /* Create the xbuffer[] lists */
      main_ptr->whichptr = 0;   /* Read first iMCU row into xbuffer[0] */
      main_ptr->context_state = CTX_PREPARE_FOR_IMCU;
      main_ptr->iMCU_row_ctr = 0;
    } else {
      /* Simple case with no context needed */
      main_ptr->pub._process_data = process_data_simple_main;
    }
    main_ptr->buffer_full = FALSE;      /* Mark buffer empty */
    main_ptr->rowgroup_ctr = 0;
    break;
#ifdef QUANT_2PASS_SUPPORTED
  case JBUF_CRANK_DEST:
    /* For last pass of 2-pass quantization, just crank the postprocessor */
    main_ptr->pub._process_data = process_data_crank_post;
    break;
#endif
  default:
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);
    break;
  }
}


/*
 * Process some data.
 * This handles the simple case where no context is required.
 */

METHODDEF(void)
process_data_simple_main(j_decompress_ptr cinfo, _JSAMPARRAY output_buf,
                         JDIMENSION *out_row_ctr, JDIMENSION out_rows_avail)
{
  my_main_ptr main_ptr = (my_main_ptr)cinfo->main;
  JDIMENSION rowgroups_avail;

  /* Read input data if we haven't filled the main buffer yet */
  if (!main_ptr->buffer_full) {
    if (!(*cinfo->coef->_decompress_data) (cinfo, main_ptr->buffer))
      return;                   /* suspension forced, can do nothing more */
    main_ptr->buffer_full = TRUE;       /* OK, we have an iMCU row to work with */
  }

  /* There are always min_DCT_scaled_size row groups in an iMCU row. */
  rowgroups_avail = (JDIMENSION)cinfo->_min_DCT_scaled_size;
  /* Note: at the bottom of the image, we may pass extra garbage row groups
   * to the postprocessor.  The postprocessor has to check for bottom
   * of image anyway (at row resolution), so no point in us doing it too.
   */

  /* Feed the postprocessor */
  (*cinfo->post->_post_process_data) (cinfo, main_ptr->buffer,
                                      &main_ptr->rowgroup_ctr, rowgroups_avail,
                                      output_buf, out_row_ctr, out_rows_avail);

  /* Has postprocessor consumed all the data yet? If so, mark buffer empty */
  if (main_ptr->rowgroup_ctr >= rowgroups_avail) {
    main_ptr->buffer_full = FALSE;
    main_ptr->rowgroup_ctr = 0;
  }
}


/*
 * Process some data.
 * This handles the case where context rows must be provided.
 */

METHODDEF(void)
process_data_context_main(j_decompress_ptr cinfo, _JSAMPARRAY output_buf,
                          JDIMENSION *out_row_ctr, JDIMENSION out_rows_avail)
{
  my_main_ptr main_ptr = (my_main_ptr)cinfo->main;

  /* Read input data if we haven't filled the main buffer yet */
  if (!main_ptr->buffer_full) {
    if (!(*cinfo->coef->_decompress_data) (cinfo,
                                           main_ptr->xbuffer[main_ptr->whichptr]))
      return;                   /* suspension forced, can do nothing more */
    main_ptr->buffer_full = TRUE;       /* OK, we have an iMCU row to work with */
    main_ptr->iMCU_row_ctr++;   /* count rows received */
  }

  /* Postprocessor typically will not swallow all the input data it is handed
   * in one call (due to filling the output buffer first).  Must be prepared
   * to exit and restart.  This switch lets us keep track of how far we got.
   * Note that each case falls through to the next on successful completion.
   */
  switch (main_ptr->context_state) {
  case CTX_POSTPONED_ROW:
    /* Call postprocessor using previously set pointers for postponed row */
    (*cinfo->post->_post_process_data) (cinfo,
                                        main_ptr->xbuffer[main_ptr->whichptr],
                                        &main_ptr->rowgroup_ctr,
                                        main_ptr->rowgroups_avail, output_buf,
                                        out_row_ctr, out_rows_avail);
    if (main_ptr->rowgroup_ctr < main_ptr->rowgroups_avail)
      return;                   /* Need to suspend */
    main_ptr->context_state = CTX_PREPARE_FOR_IMCU;
    if (*out_row_ctr >= out_rows_avail)
      return;                   /* Postprocessor exactly filled output buf */
    FALLTHROUGH                 /*FALLTHROUGH*/
  case CTX_PREPARE_FOR_IMCU:
    /* Prepare to process first M-1 row groups of this iMCU row */
    main_ptr->rowgroup_ctr = 0;
    main_ptr->rowgroups_avail = (JDIMENSION)(cinfo->_min_DCT_scaled_size - 1);
    /* Check for bottom of image: if so, tweak pointers to "duplicate"
     * the last sample row, and adjust rowgroups_avail to ignore padding rows.
     */
    if (main_ptr->iMCU_row_ctr == cinfo->total_iMCU_rows)
      set_bottom_pointers(cinfo);
    main_ptr->context_state = CTX_PROCESS_IMCU;
    FALLTHROUGH                 /*FALLTHROUGH*/
  case CTX_PROCESS_IMCU:
    /* Call postprocessor using previously set pointers */
    (*cinfo->post->_post_process_data) (cinfo,
                                        main_ptr->xbuffer[main_ptr->whichptr],
                                        &main_ptr->rowgroup_ctr,
                                        main_ptr->rowgroups_avail, output_buf,
                                        out_row_ctr, out_rows_avail);
    if (main_ptr->rowgroup_ctr < main_ptr->rowgroups_avail)
      return;                   /* Need to suspend */
    /* After the first iMCU, change wraparound pointers to normal state */
    if (main_ptr->iMCU_row_ctr == 1)
      set_wraparound_pointers(cinfo);
    /* Prepare to load new iMCU row using other xbuffer list */
    main_ptr->whichptr ^= 1;    /* 0=>1 or 1=>0 */
    main_ptr->buffer_full = FALSE;
    /* Still need to process last row group of this iMCU row, */
    /* which is saved at index M+1 of the other xbuffer */
    main_ptr->rowgroup_ctr = (JDIMENSION)(cinfo->_min_DCT_scaled_size + 1);
    main_ptr->rowgroups_avail = (JDIMENSION)(cinfo->_min_DCT_scaled_size + 2);
    main_ptr->context_state = CTX_POSTPONED_ROW;
  }
}


/*
 * Process some data.
 * Final pass of two-pass quantization: just call the postprocessor.
 * Source data will be the postprocessor controller's internal buffer.
 */

#ifdef QUANT_2PASS_SUPPORTED

METHODDEF(void)
process_data_crank_post(j_decompress_ptr cinfo, _JSAMPARRAY output_buf,
                        JDIMENSION *out_row_ctr, JDIMENSION out_rows_avail)
{
  (*cinfo->post->_post_process_data) (cinfo, (_JSAMPIMAGE)NULL,
                                      (JDIMENSION *)NULL, (JDIMENSION)0,
                                      output_buf, out_row_ctr, out_rows_avail);
}

#endif /* QUANT_2PASS_SUPPORTED */


/*
 * Initialize main buffer controller.
 */

GLOBAL(void)
_jinit_d_main_controller(j_decompress_ptr cinfo, boolean need_full_buffer)
{
  my_main_ptr main_ptr;
  int ci, rgroup, ngroups;
  jpeg_component_info *compptr;

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

  main_ptr = (my_main_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(my_main_controller));
  cinfo->main = (struct jpeg_d_main_controller *)main_ptr;
  main_ptr->pub.start_pass = start_pass_main;

  if (need_full_buffer)         /* shouldn't happen */
    ERREXIT(cinfo, JERR_BAD_BUFFER_MODE);

  /* Allocate the workspace.
   * ngroups is the number of row groups we need.
   */
  if (cinfo->upsample->need_context_rows) {
    if (cinfo->_min_DCT_scaled_size < 2) /* unsupported, see comments above */
      ERREXIT(cinfo, JERR_NOTIMPL);
    alloc_funny_pointers(cinfo); /* Alloc space for xbuffer[] lists */
    ngroups = cinfo->_min_DCT_scaled_size + 2;
  } else {
    ngroups = cinfo->_min_DCT_scaled_size;
  }

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    rgroup = (compptr->v_samp_factor * compptr->_DCT_scaled_size) /
      cinfo->_min_DCT_scaled_size; /* height of a row group of component */
    main_ptr->buffer[ci] = (_JSAMPARRAY)(*cinfo->mem->alloc_sarray)
                        ((j_common_ptr)cinfo, JPOOL_IMAGE,
                         compptr->width_in_blocks * compptr->_DCT_scaled_size,
                         (JDIMENSION)(rgroup * ngroups));
  }
}

#endif /* BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED) */
