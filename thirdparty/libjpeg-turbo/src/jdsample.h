/*
 * jdsample.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

#define JPEG_INTERNALS
#include "jpeglib.h"
#include "jsamplecomp.h"


/* Pointer to routine to upsample a single component */
typedef void (*upsample1_ptr) (j_decompress_ptr cinfo,
                               jpeg_component_info *compptr,
                               _JSAMPARRAY input_data,
                               _JSAMPARRAY *output_data_ptr);

/* Private subobject */

typedef struct {
  struct jpeg_upsampler pub;    /* public fields */

  /* Color conversion buffer.  When using separate upsampling and color
   * conversion steps, this buffer holds one upsampled row group until it
   * has been color converted and output.
   * Note: we do not allocate any storage for component(s) which are full-size,
   * ie do not need rescaling.  The corresponding entry of color_buf[] is
   * simply set to point to the input data array, thereby avoiding copying.
   */
  _JSAMPARRAY color_buf[MAX_COMPONENTS];

  /* Per-component upsampling method pointers */
  upsample1_ptr methods[MAX_COMPONENTS];

  int next_row_out;             /* counts rows emitted from color_buf */
  JDIMENSION rows_to_go;        /* counts rows remaining in image */

  /* Height of an input row group for each component. */
  int rowgroup_height[MAX_COMPONENTS];

  /* These arrays save pixel expansion factors so that int_expand need not
   * recompute them each time.  They are unused for other upsampling methods.
   */
  UINT8 h_expand[MAX_COMPONENTS];
  UINT8 v_expand[MAX_COMPONENTS];
} my_upsampler;

typedef my_upsampler *my_upsample_ptr;
