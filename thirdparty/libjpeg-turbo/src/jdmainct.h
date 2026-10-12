/*
 * jdmainct.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#define JPEG_INTERNALS
#include "jpeglib.h"
#include "jpegapicomp.h"
#include "jsamplecomp.h"


#if BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED)

/* Private buffer controller object */

typedef struct {
  struct jpeg_d_main_controller pub; /* public fields */

  /* Pointer to allocated workspace (M or M+2 row groups). */
  _JSAMPARRAY buffer[MAX_COMPONENTS];

  boolean buffer_full;          /* Have we gotten an iMCU row from decoder? */
  JDIMENSION rowgroup_ctr;      /* counts row groups output to postprocessor */

  /* Remaining fields are only used in the context case. */

  /* These are the master pointers to the funny-order pointer lists. */
  _JSAMPIMAGE xbuffer[2];       /* pointers to weird pointer lists */

  int whichptr;                 /* indicates which pointer set is now in use */
  int context_state;            /* process_data state machine status */
  JDIMENSION rowgroups_avail;   /* row groups available to postprocessor */
  JDIMENSION iMCU_row_ctr;      /* counts iMCU rows to detect image top/bot */
} my_main_controller;

typedef my_main_controller *my_main_ptr;


/* context_state values: */
#define CTX_PREPARE_FOR_IMCU    0       /* need to prepare for MCU row */
#define CTX_PROCESS_IMCU        1       /* feeding iMCU to postprocessor */
#define CTX_POSTPONED_ROW       2       /* feeding postponed row group */


LOCAL(void)
set_wraparound_pointers(j_decompress_ptr cinfo)
/* Set up the "wraparound" pointers at top and bottom of the pointer lists.
 * This changes the pointer list state from top-of-image to the normal state.
 */
{
  my_main_ptr main_ptr = (my_main_ptr)cinfo->main;
  int ci, i, rgroup;
  int M = cinfo->_min_DCT_scaled_size;
  jpeg_component_info *compptr;
  _JSAMPARRAY xbuf0, xbuf1;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    rgroup = (compptr->v_samp_factor * compptr->_DCT_scaled_size) /
      cinfo->_min_DCT_scaled_size; /* height of a row group of component */
    xbuf0 = main_ptr->xbuffer[0][ci];
    xbuf1 = main_ptr->xbuffer[1][ci];
    for (i = 0; i < rgroup; i++) {
      xbuf0[i - rgroup] = xbuf0[rgroup * (M + 1) + i];
      xbuf1[i - rgroup] = xbuf1[rgroup * (M + 1) + i];
      xbuf0[rgroup * (M + 2) + i] = xbuf0[i];
      xbuf1[rgroup * (M + 2) + i] = xbuf1[i];
    }
  }
}

#endif /* BITS_IN_JSAMPLE != 16 || defined(D_LOSSLESS_SUPPORTED) */
