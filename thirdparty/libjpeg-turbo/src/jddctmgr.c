/*
 * jddctmgr.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1994-1996, Thomas G. Lane.
 * Modified 2002-2010 by Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
 * Copyright (C) 2010, 2015, 2022, D. R. Commander.
 * Copyright (C) 2013, MIPS Technologies, Inc., California.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains the inverse-DCT management logic.
 * This code selects a particular IDCT implementation to be used,
 * and it performs related housekeeping chores.  No code in this file
 * is executed per IDCT step, only during output pass setup.
 *
 * Note that the IDCT routines are responsible for performing coefficient
 * dequantization as well as the IDCT proper.  This module sets up the
 * dequantization multiplier table needed by the IDCT routine.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jdct.h"               /* Private declarations for DCT subsystem */
#include "jsimddct.h"
#include "jpegapicomp.h"


/*
 * The decompressor input side (jdinput.c) saves away the appropriate
 * quantization table for each component at the start of the first scan
 * involving that component.  (This is necessary in order to correctly
 * decode files that reuse Q-table slots.)
 * When we are ready to make an output pass, the saved Q-table is converted
 * to a multiplier table that will actually be used by the IDCT routine.
 * The multiplier table contents are IDCT-method-dependent.  To support
 * application changes in IDCT method between scans, we can remake the
 * multiplier tables if necessary.
 * In buffered-image mode, the first output pass may occur before any data
 * has been seen for some components, and thus before their Q-tables have
 * been saved away.  To handle this case, multiplier tables are preset
 * to zeroes; the result of the IDCT will be a neutral gray level.
 */


/* Private subobject for this module */

typedef struct {
  struct jpeg_inverse_dct pub;  /* public fields */

  /* This array contains the IDCT method code that each multiplier table
   * is currently set up for, or -1 if it's not yet set up.
   * The actual multiplier tables are pointed to by dct_table in the
   * per-component comp_info structures.
   */
  int cur_method[MAX_COMPONENTS];
} my_idct_controller;

typedef my_idct_controller *my_idct_ptr;


/* Allocated multiplier tables: big enough for any supported variant */

typedef union {
  ISLOW_MULT_TYPE islow_array[DCTSIZE2];
#ifdef DCT_IFAST_SUPPORTED
  IFAST_MULT_TYPE ifast_array[DCTSIZE2];
#endif
#ifdef DCT_FLOAT_SUPPORTED
  FLOAT_MULT_TYPE float_array[DCTSIZE2];
#endif
} multiplier_table;


/* The current scaled-IDCT routines require ISLOW-style multiplier tables,
 * so be sure to compile that code if either ISLOW or SCALING is requested.
 */
#ifdef DCT_ISLOW_SUPPORTED
#define PROVIDE_ISLOW_TABLES
#else
#ifdef IDCT_SCALING_SUPPORTED
#define PROVIDE_ISLOW_TABLES
#endif
#endif


/*
 * Prepare for an output pass.
 * Here we select the proper IDCT routine for each component and build
 * a matching multiplier table.
 */

METHODDEF(void)
start_pass(j_decompress_ptr cinfo)
{
  my_idct_ptr idct = (my_idct_ptr)cinfo->idct;
  int ci, i;
  jpeg_component_info *compptr;
  int method = 0;
  _inverse_DCT_method_ptr method_ptr = NULL;
  JQUANT_TBL *qtbl;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Select the proper IDCT routine for this component's scaling */
    switch (compptr->_DCT_scaled_size) {
#ifdef IDCT_SCALING_SUPPORTED
    case 1:
      method_ptr = _jpeg_idct_1x1;
      method = JDCT_ISLOW;      /* jidctred uses islow-style table */
      break;
    case 2:
#ifdef WITH_SIMD
      if (jsimd_can_idct_2x2())
        method_ptr = jsimd_idct_2x2;
      else
#endif
        method_ptr = _jpeg_idct_2x2;
      method = JDCT_ISLOW;      /* jidctred uses islow-style table */
      break;
    case 3:
      method_ptr = _jpeg_idct_3x3;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 4:
#ifdef WITH_SIMD
      if (jsimd_can_idct_4x4())
        method_ptr = jsimd_idct_4x4;
      else
#endif
        method_ptr = _jpeg_idct_4x4;
      method = JDCT_ISLOW;      /* jidctred uses islow-style table */
      break;
    case 5:
      method_ptr = _jpeg_idct_5x5;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 6:
#if defined(WITH_SIMD) && defined(__mips__)
      if (jsimd_can_idct_6x6())
        method_ptr = jsimd_idct_6x6;
      else
#endif
      method_ptr = _jpeg_idct_6x6;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 7:
      method_ptr = _jpeg_idct_7x7;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
#endif
    case DCTSIZE:
      switch (cinfo->dct_method) {
#ifdef DCT_ISLOW_SUPPORTED
      case JDCT_ISLOW:
#ifdef WITH_SIMD
        if (jsimd_can_idct_islow())
          method_ptr = jsimd_idct_islow;
        else
#endif
          method_ptr = _jpeg_idct_islow;
        method = JDCT_ISLOW;
        break;
#endif
#ifdef DCT_IFAST_SUPPORTED
      case JDCT_IFAST:
#ifdef WITH_SIMD
        if (jsimd_can_idct_ifast())
          method_ptr = jsimd_idct_ifast;
        else
#endif
          method_ptr = _jpeg_idct_ifast;
        method = JDCT_IFAST;
        break;
#endif
#ifdef DCT_FLOAT_SUPPORTED
      case JDCT_FLOAT:
#ifdef WITH_SIMD
        if (jsimd_can_idct_float())
          method_ptr = jsimd_idct_float;
        else
#endif
          method_ptr = _jpeg_idct_float;
        method = JDCT_FLOAT;
        break;
#endif
      default:
        ERREXIT(cinfo, JERR_NOT_COMPILED);
        break;
      }
      break;
#ifdef IDCT_SCALING_SUPPORTED
    case 9:
      method_ptr = _jpeg_idct_9x9;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 10:
      method_ptr = _jpeg_idct_10x10;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 11:
      method_ptr = _jpeg_idct_11x11;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 12:
#if defined(WITH_SIMD) && defined(__mips__)
      if (jsimd_can_idct_12x12())
        method_ptr = jsimd_idct_12x12;
      else
#endif
      method_ptr = _jpeg_idct_12x12;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 13:
      method_ptr = _jpeg_idct_13x13;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 14:
      method_ptr = _jpeg_idct_14x14;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 15:
      method_ptr = _jpeg_idct_15x15;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
    case 16:
      method_ptr = _jpeg_idct_16x16;
      method = JDCT_ISLOW;      /* jidctint uses islow-style table */
      break;
#endif
    default:
      ERREXIT1(cinfo, JERR_BAD_DCTSIZE, compptr->_DCT_scaled_size);
      break;
    }
    idct->pub._inverse_DCT[ci] = method_ptr;
    /* Create multiplier table from quant table.
     * However, we can skip this if the component is uninteresting
     * or if we already built the table.  Also, if no quant table
     * has yet been saved for the component, we leave the
     * multiplier table all-zero; we'll be reading zeroes from the
     * coefficient controller's buffer anyway.
     */
    if (!compptr->component_needed || idct->cur_method[ci] == method)
      continue;
    qtbl = compptr->quant_table;
    if (qtbl == NULL)           /* happens if no data yet for component */
      continue;
    idct->cur_method[ci] = method;
    switch (method) {
#ifdef PROVIDE_ISLOW_TABLES
    case JDCT_ISLOW:
      {
        /* For LL&M IDCT method, multipliers are equal to raw quantization
         * coefficients, but are stored as ints to ensure access efficiency.
         */
        ISLOW_MULT_TYPE *ismtbl = (ISLOW_MULT_TYPE *)compptr->dct_table;
        for (i = 0; i < DCTSIZE2; i++) {
          ismtbl[i] = (ISLOW_MULT_TYPE)qtbl->quantval[i];
        }
      }
      break;
#endif
#ifdef DCT_IFAST_SUPPORTED
    case JDCT_IFAST:
      {
        /* For AA&N IDCT method, multipliers are equal to quantization
         * coefficients scaled by scalefactor[row]*scalefactor[col], where
         *   scalefactor[0] = 1
         *   scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
         * For integer operation, the multiplier table is to be scaled by
         * IFAST_SCALE_BITS.
         */
        IFAST_MULT_TYPE *ifmtbl = (IFAST_MULT_TYPE *)compptr->dct_table;
#define CONST_BITS  14
        static const INT16 aanscales[DCTSIZE2] = {
          /* precomputed values scaled up by 14 bits */
          16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
          22725, 31521, 29692, 26722, 22725, 17855, 12299,  6270,
          21407, 29692, 27969, 25172, 21407, 16819, 11585,  5906,
          19266, 26722, 25172, 22654, 19266, 15137, 10426,  5315,
          16384, 22725, 21407, 19266, 16384, 12873,  8867,  4520,
          12873, 17855, 16819, 15137, 12873, 10114,  6967,  3552,
           8867, 12299, 11585, 10426,  8867,  6967,  4799,  2446,
           4520,  6270,  5906,  5315,  4520,  3552,  2446,  1247
        };
        SHIFT_TEMPS

        for (i = 0; i < DCTSIZE2; i++) {
          ifmtbl[i] = (IFAST_MULT_TYPE)
            DESCALE(MULTIPLY16V16((JLONG)qtbl->quantval[i],
                                  (JLONG)aanscales[i]),
                    CONST_BITS - IFAST_SCALE_BITS);
        }
      }
      break;
#endif
#ifdef DCT_FLOAT_SUPPORTED
    case JDCT_FLOAT:
      {
        /* For float AA&N IDCT method, multipliers are equal to quantization
         * coefficients scaled by scalefactor[row]*scalefactor[col], where
         *   scalefactor[0] = 1
         *   scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
         */
        FLOAT_MULT_TYPE *fmtbl = (FLOAT_MULT_TYPE *)compptr->dct_table;
        int row, col;
        static const double aanscalefactor[DCTSIZE] = {
          1.0, 1.387039845, 1.306562965, 1.175875602,
          1.0, 0.785694958, 0.541196100, 0.275899379
        };

        i = 0;
        for (row = 0; row < DCTSIZE; row++) {
          for (col = 0; col < DCTSIZE; col++) {
            fmtbl[i] = (FLOAT_MULT_TYPE)
              ((double)qtbl->quantval[i] *
               aanscalefactor[row] * aanscalefactor[col]);
            i++;
          }
        }
      }
      break;
#endif
    default:
      ERREXIT(cinfo, JERR_NOT_COMPILED);
      break;
    }
  }
}


/*
 * Initialize IDCT manager.
 */

GLOBAL(void)
_jinit_inverse_dct(j_decompress_ptr cinfo)
{
  my_idct_ptr idct;
  int ci;
  jpeg_component_info *compptr;

  if (cinfo->data_precision != BITS_IN_JSAMPLE)
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  idct = (my_idct_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(my_idct_controller));
  cinfo->idct = (struct jpeg_inverse_dct *)idct;
  idct->pub.start_pass = start_pass;

  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Allocate and pre-zero a multiplier table for each component */
    compptr->dct_table =
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  sizeof(multiplier_table));
    memset(compptr->dct_table, 0, sizeof(multiplier_table));
    /* Mark multiplier table not yet set up for any method */
    idct->cur_method[ci] = -1;
  }
}
