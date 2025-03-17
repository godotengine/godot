/*
 * jdlossls.c
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
 * This file contains prediction, sample undifferencing, point transform, and
 * sample scaling routines for the lossless JPEG decompressor.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"

#ifdef D_LOSSLESS_SUPPORTED


/**************** Sample undifferencing (reconstruction) *****************/

/*
 * In order to avoid a performance penalty for checking which predictor is
 * being used and which row is being processed for each call of the
 * undifferencer, and to promote optimization, we have separate undifferencing
 * functions for each predictor selection value.
 *
 * We are able to avoid duplicating source code by implementing the predictors
 * and undifferencers as macros.  Each of the undifferencing functions is
 * simply a wrapper around an UNDIFFERENCE macro with the appropriate PREDICTOR
 * macro passed as an argument.
 */

/* Predictor for the first column of the first row: 2^(P-Pt-1) */
#define INITIAL_PREDICTORx  (1 << (cinfo->data_precision - cinfo->Al - 1))

/* Predictor for the first column of the remaining rows: Rb */
#define INITIAL_PREDICTOR2  prev_row[0]


/*
 * 1-Dimensional undifferencer routine.
 *
 * This macro implements the 1-D horizontal predictor (1).  INITIAL_PREDICTOR
 * is used as the special case predictor for the first column, which must be
 * either INITIAL_PREDICTOR2 or INITIAL_PREDICTORx.  The remaining samples
 * use PREDICTOR1.
 *
 * The reconstructed sample is supposed to be calculated modulo 2^16, so we
 * logically AND the result with 0xFFFF.
 */

#define UNDIFFERENCE_1D(INITIAL_PREDICTOR) \
  int Ra; \
  \
  Ra = (*diff_buf++ + INITIAL_PREDICTOR) & 0xFFFF; \
  *undiff_buf++ = Ra; \
  \
  while (--width) { \
    Ra = (*diff_buf++ + PREDICTOR1) & 0xFFFF; \
    *undiff_buf++ = Ra; \
  }


/*
 * 2-Dimensional undifferencer routine.
 *
 * This macro implements the 2-D horizontal predictors (#2-7).  PREDICTOR2 is
 * used as the special case predictor for the first column.  The remaining
 * samples use PREDICTOR, which is a function of Ra, Rb, and Rc.
 *
 * Because prev_row and output_buf may point to the same storage area (in an
 * interleaved image with Vi=1, for example), we must take care to buffer Rb/Rc
 * before writing the current reconstructed sample value into output_buf.
 *
 * The reconstructed sample is supposed to be calculated modulo 2^16, so we
 * logically AND the result with 0xFFFF.
 */

#define UNDIFFERENCE_2D(PREDICTOR) \
  int Ra, Rb, Rc; \
  \
  Rb = *prev_row++; \
  Ra = (*diff_buf++ + PREDICTOR2) & 0xFFFF; \
  *undiff_buf++ = Ra; \
  \
  while (--width) { \
    Rc = Rb; \
    Rb = *prev_row++; \
    Ra = (*diff_buf++ + PREDICTOR) & 0xFFFF; \
    *undiff_buf++ = Ra; \
  }


/*
 * Undifferencers for the second and subsequent rows in a scan or restart
 * interval.  The first sample in the row is undifferenced using the vertical
 * predictor (2).  The rest of the samples are undifferenced using the
 * predictor specified in the scan header.
 */

METHODDEF(void)
jpeg_undifference1(j_decompress_ptr cinfo, int comp_index,
                   JDIFFROW diff_buf, JDIFFROW prev_row,
                   JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_1D(INITIAL_PREDICTOR2);
}

METHODDEF(void)
jpeg_undifference2(j_decompress_ptr cinfo, int comp_index,
                   JDIFFROW diff_buf, JDIFFROW prev_row,
                   JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR2);
  (void)(Rc);
}

METHODDEF(void)
jpeg_undifference3(j_decompress_ptr cinfo, int comp_index,
                   JDIFFROW diff_buf, JDIFFROW prev_row,
                   JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR3);
}

METHODDEF(void)
jpeg_undifference4(j_decompress_ptr cinfo, int comp_index,
                   JDIFFROW diff_buf, JDIFFROW prev_row,
                   JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR4);
}

METHODDEF(void)
jpeg_undifference5(j_decompress_ptr cinfo, int comp_index,
                   JDIFFROW diff_buf, JDIFFROW prev_row,
                   JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR5);
}

METHODDEF(void)
jpeg_undifference6(j_decompress_ptr cinfo, int comp_index,
                   JDIFFROW diff_buf, JDIFFROW prev_row,
                   JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR6);
}

METHODDEF(void)
jpeg_undifference7(j_decompress_ptr cinfo, int comp_index,
                   JDIFFROW diff_buf, JDIFFROW prev_row,
                   JDIFFROW undiff_buf, JDIMENSION width)
{
  UNDIFFERENCE_2D(PREDICTOR7);
  (void)(Rc);
}


/*
 * Undifferencer for the first row in a scan or restart interval.  The first
 * sample in the row is undifferenced using the special predictor constant
 * x=2^(P-Pt-1).  The rest of the samples are undifferenced using the
 * 1-D horizontal predictor (1).
 */

METHODDEF(void)
jpeg_undifference_first_row(j_decompress_ptr cinfo, int comp_index,
                            JDIFFROW diff_buf, JDIFFROW prev_row,
                            JDIFFROW undiff_buf, JDIMENSION width)
{
  lossless_decomp_ptr losslessd = (lossless_decomp_ptr)cinfo->idct;

  UNDIFFERENCE_1D(INITIAL_PREDICTORx);

  /*
   * Now that we have undifferenced the first row, we want to use the
   * undifferencer that corresponds to the predictor specified in the
   * scan header.
   */
  switch (cinfo->Ss) {
  case 1:
    losslessd->predict_undifference[comp_index] = jpeg_undifference1;
    break;
  case 2:
    losslessd->predict_undifference[comp_index] = jpeg_undifference2;
    break;
  case 3:
    losslessd->predict_undifference[comp_index] = jpeg_undifference3;
    break;
  case 4:
    losslessd->predict_undifference[comp_index] = jpeg_undifference4;
    break;
  case 5:
    losslessd->predict_undifference[comp_index] = jpeg_undifference5;
    break;
  case 6:
    losslessd->predict_undifference[comp_index] = jpeg_undifference6;
    break;
  case 7:
    losslessd->predict_undifference[comp_index] = jpeg_undifference7;
    break;
  }
}


/*********************** Sample upscaling by 2^Pt ************************/

METHODDEF(void)
simple_upscale(j_decompress_ptr cinfo,
               JDIFFROW diff_buf, _JSAMPROW output_buf, JDIMENSION width)
{
  do {
    *output_buf++ = (_JSAMPLE)(*diff_buf++ << cinfo->Al);
  } while (--width);
}

METHODDEF(void)
noscale(j_decompress_ptr cinfo,
        JDIFFROW diff_buf, _JSAMPROW output_buf, JDIMENSION width)
{
  do {
    *output_buf++ = (_JSAMPLE)(*diff_buf++);
  } while (--width);
}


/*
 * Initialize for an input processing pass.
 */

METHODDEF(void)
start_pass_lossless(j_decompress_ptr cinfo)
{
  lossless_decomp_ptr losslessd = (lossless_decomp_ptr)cinfo->idct;
  int ci;

  /* Check that the scan parameters Ss, Se, Ah, Al are OK for lossless JPEG.
   *
   * Ss is the predictor selection value (psv).  Legal values for sequential
   * lossless JPEG are: 1 <= psv <= 7.
   *
   * Se and Ah are not used and should be zero.
   *
   * Al specifies the point transform (Pt).
   * Legal values are: 0 <= Pt <= (data precision - 1).
   */
  if (cinfo->Ss < 1 || cinfo->Ss > 7 ||
      cinfo->Se != 0 || cinfo->Ah != 0 ||
      cinfo->Al < 0 || cinfo->Al >= cinfo->data_precision)
    ERREXIT4(cinfo, JERR_BAD_PROGRESSION,
             cinfo->Ss, cinfo->Se, cinfo->Ah, cinfo->Al);

  /* Set undifference functions to first row function */
  for (ci = 0; ci < cinfo->num_components; ci++)
    losslessd->predict_undifference[ci] = jpeg_undifference_first_row;

  /* Set scaler function based on Pt */
  if (cinfo->Al)
    losslessd->scaler_scale = simple_upscale;
  else
    losslessd->scaler_scale = noscale;
}


/*
 * Initialize the lossless decompressor.
 */

GLOBAL(void)
_jinit_lossless_decompressor(j_decompress_ptr cinfo)
{
  lossless_decomp_ptr losslessd;

#if BITS_IN_JSAMPLE == 8
  if (cinfo->data_precision > BITS_IN_JSAMPLE || cinfo->data_precision < 2)
#else
  if (cinfo->data_precision > BITS_IN_JSAMPLE ||
      cinfo->data_precision < BITS_IN_JSAMPLE - 3)
#endif
    ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);

  /* Create subobject in permanent pool */
  losslessd = (lossless_decomp_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_PERMANENT,
                                sizeof(jpeg_lossless_decompressor));
  cinfo->idct = (struct jpeg_inverse_dct *)losslessd;
  losslessd->pub.start_pass = start_pass_lossless;
}

#endif /* D_LOSSLESS_SUPPORTED */
