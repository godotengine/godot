/*
 * jlossls.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1998, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This include file contains common declarations for the lossless JPEG
 * codec modules.
 */

#ifndef JLOSSLS_H
#define JLOSSLS_H

#if defined(C_LOSSLESS_SUPPORTED) || defined(D_LOSSLESS_SUPPORTED)

#define JPEG_INTERNALS
#include "jpeglib.h"
#include "jsamplecomp.h"


#define ALLOC_DARRAY(pool_id, diffsperrow, numrows) \
  (JDIFFARRAY)(*cinfo->mem->alloc_sarray) \
    ((j_common_ptr)cinfo, pool_id, \
     (diffsperrow) * sizeof(JDIFF) / sizeof(_JSAMPLE), numrows)


/*
 * Table H.1: Predictors for lossless coding.
 */

#define PREDICTOR1  Ra
#define PREDICTOR2  Rb
#define PREDICTOR3  Rc
#define PREDICTOR4  (int)((JLONG)Ra + (JLONG)Rb - (JLONG)Rc)
#define PREDICTOR5  (int)((JLONG)Ra + RIGHT_SHIFT((JLONG)Rb - (JLONG)Rc, 1))
#define PREDICTOR6  (int)((JLONG)Rb + RIGHT_SHIFT((JLONG)Ra - (JLONG)Rc, 1))
#define PREDICTOR7  (int)RIGHT_SHIFT((JLONG)Ra + (JLONG)Rb, 1)

#endif


#ifdef C_LOSSLESS_SUPPORTED

typedef void (*predict_difference_method_ptr) (j_compress_ptr cinfo, int ci,
                                               _JSAMPROW input_buf,
                                               _JSAMPROW prev_row,
                                               JDIFFROW diff_buf,
                                               JDIMENSION width);

/* Lossless compressor */
typedef struct {
  struct jpeg_forward_dct pub;  /* public fields */

  /* It is useful to allow each component to have a separate diff method. */
  predict_difference_method_ptr predict_difference[MAX_COMPONENTS];

  /* MCU rows left in the restart interval for each component */
  unsigned int restart_rows_to_go[MAX_COMPONENTS];

  /* Sample scaling */
  void (*scaler_scale) (j_compress_ptr cinfo, _JSAMPROW input_buf,
                        _JSAMPROW output_buf, JDIMENSION width);
} jpeg_lossless_compressor;

typedef jpeg_lossless_compressor *lossless_comp_ptr;

#endif /* C_LOSSLESS_SUPPORTED */


#ifdef D_LOSSLESS_SUPPORTED

typedef void (*predict_undifference_method_ptr) (j_decompress_ptr cinfo,
                                                 int comp_index,
                                                 JDIFFROW diff_buf,
                                                 JDIFFROW prev_row,
                                                 JDIFFROW undiff_buf,
                                                 JDIMENSION width);

/* Lossless decompressor */
typedef struct {
  struct jpeg_inverse_dct pub;  /* public fields */

  /* It is useful to allow each component to have a separate undiff method. */
  predict_undifference_method_ptr predict_undifference[MAX_COMPONENTS];

  /* Sample scaling */
  void (*scaler_scale) (j_decompress_ptr cinfo, JDIFFROW diff_buf,
                        _JSAMPROW output_buf, JDIMENSION width);
} jpeg_lossless_decompressor;

typedef jpeg_lossless_decompressor *lossless_decomp_ptr;

#endif /* D_LOSSLESS_SUPPORTED */

#endif /* JLOSSLS_H */
