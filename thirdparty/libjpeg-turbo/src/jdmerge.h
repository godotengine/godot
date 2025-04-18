/*
 * jdmerge.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2020, 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#define JPEG_INTERNALS
#include "jpeglib.h"
#include "jsamplecomp.h"

#ifdef UPSAMPLE_MERGING_SUPPORTED


/* Private subobject */

typedef struct {
  struct jpeg_upsampler pub;    /* public fields */

  /* Pointer to routine to do actual upsampling/conversion of one row group */
  void (*upmethod) (j_decompress_ptr cinfo, _JSAMPIMAGE input_buf,
                    JDIMENSION in_row_group_ctr, _JSAMPARRAY output_buf);

  /* Private state for YCC->RGB conversion */
  int *Cr_r_tab;                /* => table for Cr to R conversion */
  int *Cb_b_tab;                /* => table for Cb to B conversion */
  JLONG *Cr_g_tab;              /* => table for Cr to G conversion */
  JLONG *Cb_g_tab;              /* => table for Cb to G conversion */

  /* For 2:1 vertical sampling, we produce two output rows at a time.
   * We need a "spare" row buffer to hold the second output row if the
   * application provides just a one-row buffer; we also use the spare
   * to discard the dummy last row if the image height is odd.
   */
  _JSAMPROW spare_row;
  boolean spare_full;           /* T if spare buffer is occupied */

  JDIMENSION out_row_width;     /* samples per output row */
  JDIMENSION rows_to_go;        /* counts rows remaining in image */
} my_merged_upsampler;

typedef my_merged_upsampler *my_merged_upsample_ptr;

#endif /* UPSAMPLE_MERGING_SUPPORTED */
