/*
 * jclossls.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1998, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, 2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains prediction, sample differencing, and point transform
 * routines for the lossless JPEG compressor.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"

#ifdef C_LOSSLESS_SUPPORTED


/************************** Sample differencing **************************/

/*
 * In order to avoid a performance penalty for checking which predictor is
 * being used and which row is being processed for each call of the
 * undifferencer, and to promote optimization, we have separate differencing
 * functions for each predictor selection value.
 *
 * We are able to avoid duplicating source code by implementing the predictors
 * and differencers as macros.  Each of the differencing functions is simply a
 * wrapper around a DIFFERENCE macro with the appropriate PREDICTOR macro
 * passed as an argument.
 */

/* Forward declarations */
LOCAL(void) reset_predictor(j_compress_ptr cinfo, int ci);


/* Predictor for the first column of the first row: 2^(P-Pt-1) */
#define INITIAL_PREDICTORx  (1 << (cinfo->data_precision - cinfo->Al - 1))

/* Predictor for the first column of the remaining rows: Rb */
#define INITIAL_PREDICTOR2  prev_row[0]


/*
 * 1-Dimensional differencer routine.
 *
 * This macro implements the 1-D horizontal predictor (1).  INITIAL_PREDICTOR
 * is used as the special case predictor for the first column, which must be
 * either INITIAL_PREDICTOR2 or INITIAL_PREDICTORx.  The remaining samples
 * use PREDICTOR1.
 */

#define DIFFERENCE_1D(INITIAL_PREDICTOR) \
  lossless_comp_ptr losslessc = (lossless_comp_ptr)cinfo->fdct; \
  boolean restart = FALSE; \
  int samp, Ra; \
  \
  samp = *input_buf++; \
  *diff_buf++ = samp - INITIAL_PREDICTOR; \
  \
  while (--width) { \
    Ra = samp; \
    samp = *input_buf++; \
    *diff_buf++ = samp - PREDICTOR1; \
  } \
  \
  /* Account for restart interval (no-op if not using restarts) */ \
  if (cinfo->restart_interval) { \
    if (--(losslessc->restart_rows_to_go[ci]) == 0) { \
      reset_predictor(cinfo, ci); \
      restart = TRUE; \
    } \
  }


/*
 * 2-Dimensional differencer routine.
 *
 * This macro implements the 2-D horizontal predictors (#2-7).  PREDICTOR2 is
 * used as the special case predictor for the first column.  The remaining
 * samples use PREDICTOR, which is a function of Ra, Rb, and Rc.
 *
 * Because prev_row and output_buf may point to the same storage area (in an
 * interleaved image with Vi=1, for example), we must take care to buffer Rb/Rc
 * before writing the current reconstructed sample value into output_buf.
 */

#define DIFFERENCE_2D(PREDICTOR) \
  lossless_comp_ptr losslessc = (lossless_comp_ptr)cinfo->fdct; \
  int samp, Ra, Rb, Rc; \
  \
  Rb = *prev_row++; \
  samp = *input_buf++; \
  *diff_buf++ = samp - PREDICTOR2; \
  \
  while (--width) { \
    Rc = Rb; \
    Rb = *prev_row++; \
    Ra = samp; \
    samp = *input_buf++; \
    *diff_buf++ = samp - PREDICTOR; \
  } \
  \
  /* Account for restart interval (no-op if not using restarts) */ \
  if (cinfo->restart_interval) { \
    if (--losslessc->restart_rows_to_go[ci] == 0) \
      reset_predictor(cinfo, ci); \
  }


/*
 * Differencers for the second and subsequent rows in a scan or restart
 * interval.  The first sample in the row is differenced using the vertical
 * predictor (2).  The rest of the samples are differenced using the predictor
 * specified in the scan header.
 */

METHODDEF(void)
jpeg_difference1(j_compress_ptr cinfo, int ci,
                 _JSAMPROW input_buf, _JSAMPROW prev_row,
                 JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_1D(INITIAL_PREDICTOR2);
  (void)(restart);
}

METHODDEF(void)
jpeg_difference2(j_compress_ptr cinfo, int ci,
                 _JSAMPROW input_buf, _JSAMPROW prev_row,
                 JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR2);
  (void)(Ra);
  (void)(Rc);
}

METHODDEF(void)
jpeg_difference3(j_compress_ptr cinfo, int ci,
                 _JSAMPROW input_buf, _JSAMPROW prev_row,
                 JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR3);
  (void)(Ra);
}

METHODDEF(void)
jpeg_difference4(j_compress_ptr cinfo, int ci,
                 _JSAMPROW input_buf, _JSAMPROW prev_row,
                 JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR4);
}

METHODDEF(void)
jpeg_difference5(j_compress_ptr cinfo, int ci,
                 _JSAMPROW input_buf, _JSAMPROW prev_row,
                 JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR5);
}

METHODDEF(void)
jpeg_difference6(j_compress_ptr cinfo, int ci,
                 _JSAMPROW input_buf, _JSAMPROW prev_row,
                 JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR6);
}

METHODDEF(void)
jpeg_difference7(j_compress_ptr cinfo, int ci,
                 _JSAMPROW input_buf, _JSAMPROW prev_row,
                 JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_2D(PREDICTOR7);
  (void)(Rc);
}


/*
 * Differencer for the first row in a scan or restart interval.  The first
 * sample in the row is differenced using the special predictor constant
 * x = 2 ^ (P-Pt-1).  The rest of the samples are differenced using the
 * 1-D horizontal predictor (1).
 */

METHODDEF(void)
jpeg_difference_first_row(j_compress_ptr cinfo, int ci,
                          _JSAMPROW input_buf, _JSAMPROW prev_row,
                          JDIFFROW diff_buf, JDIMENSION width)
{
  DIFFERENCE_1D(INITIAL_PREDICTORx);

  /*
   * Now that we have differenced the first row, we want to use the
   * differencer that corresponds to the predictor specified in the
   * scan header.
   *
   * Note that we don't do this if we have just reset the predictor
   * for a new restart interval.
   */
  if (!restart) {
    switch (cinfo->Ss) {
    case 1:
      losslessc->predict_difference[ci] = jpeg_difference1;
      break;
    case 2:
      losslessc->predict_difference[ci] = jpeg_difference2;
      break;
    case 3:
      losslessc->predict_difference[ci] = jpeg_difference3;
      break;
    case 4:
      losslessc->predict_difference[ci] = jpeg_difference4;
      break;
    case 5:
      losslessc->predict_difference[ci] = jpeg_difference5;
      break;
    case 6:
      losslessc->predict_difference[ci] = jpeg_difference6;
      break;
    case 7:
      losslessc->predict_difference[ci] = jpeg_difference7;
      break;
    }
  }
}

/*
 * Reset predictor at the start of a pass or restart interval.
 */

LOCAL(void)
reset_predictor(j_compress_ptr cinfo, int ci)
{
  lossless_comp_ptr losslessc = (lossless_comp_ptr)cinfo->fdct;

  /* Initialize restart counter */
  losslessc->restart_rows_to_go[ci] =
    cinfo->restart_interval / cinfo->MCUs_per_row;

  /* Set difference function to first row function */
  losslessc->predict_difference[ci] = jpeg_difference_first_row;
}


/********************** Sample downscaling by 2^Pt ***********************/

METHODDEF(void)
simple_downscale(j_compress_ptr cinfo,
                 _JSAMPROW input_buf, _JSAMPROW output_buf, JDIMENSION width)
{
  do {
    *output_buf++ = (_JSAMPLE)RIGHT_SHIFT(*input_buf++, cinfo->Al);
  } while (--width);
}


METHODDEF(void)
noscale(j_compress_ptr cinfo,
        _JSAMPROW input_buf, _JSAMPROW output_buf, JDIMENSION width)
{
  memcpy(output_buf, input_buf, width * sizeof(_JSAMPLE));
}


/*
 * Initialize for a processing pass.
 */

METHODDEF(void)
start_pass_lossless(j_compress_ptr cinfo)
{
  lossless_comp_ptr losslessc = (lossless_comp_ptr)cinfo->fdct;
  int ci;

  /* Set scaler function based on Pt */
  if (cinfo->Al)
    losslessc->scaler_scale = simple_downscale;
  else
    losslessc->scaler_scale = noscale;

  /* Check that the restart interval is an integer multiple of the number
   * of MCUs in an MCU row.
   */
  if (cinfo->restart_interval % cinfo->MCUs_per_row != 0)
    ERREXIT2(cinfo, JERR_BAD_RESTART,
             cinfo->restart_interval, cinfo->MCUs_per_row);

  /* Set predictors for start of pass */
  for (ci = 0; ci < cinfo->num_components; ci++)
    reset_predictor(cinfo, ci);
}


/*
 * Initialize the lossless compressor.
 */

GLOBAL(void)
_jinit_lossless_compressor(j_compress_ptr cinfo)
{
  lossless_comp_ptr losslessc;

#if BITS_IN_JSAMPLE == 8
  if (cinfo->data_precision > BITS_IN_JSAMPLE || cinfo->data_precision < 2)
#else
  if (cinfo->data_precision > BITS_IN_JSAMPLE ||
      cinfo->data_precision < BITS_IN_JSAMPLE - 3)
#endif
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Create subobject in permanent pool */
  losslessc = (lossless_comp_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_PERMANENT,
                                sizeof(jpeg_lossless_compressor));
  cinfo->fdct = (struct jpeg_forward_dct *)losslessc;
  losslessc->pub.start_pass = start_pass_lossless;
}

#endif /* C_LOSSLESS_SUPPORTED */
