/*
 * jdmaster.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1995, Thomas G. Lane.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains the master control structure for the JPEG decompressor.
 */

/* Private state */

typedef struct {
  struct jpeg_decomp_master pub; /* public fields */

  int pass_number;              /* # of passes completed */

  boolean using_merged_upsample; /* TRUE if using merged upsample/cconvert */

  /* Saved references to initialized quantizer modules,
   * in case we need to switch modes.
   */
  struct jpeg_color_quantizer *quantizer_1pass;
  struct jpeg_color_quantizer *quantizer_2pass;
} my_decomp_master;

typedef my_decomp_master *my_master_ptr;
