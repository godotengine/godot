/*
 * jcmaster.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2003-2010 by Guido Vollbeding.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2010, 2016, 2018, 2022-2024, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains master control logic for the JPEG compressor.
 * These routines are concerned with parameter validation, initial setup,
 * and inter-pass control (determining the number of passes and the work
 * to be done in each pass).
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jpegapicomp.h"
#include "jcmaster.h"


/*
 * Support routines that do various essential calculations.
 */

#if JPEG_LIB_VERSION >= 70
/*
 * Compute JPEG image dimensions and related values.
 * NOTE: this is exported for possible use by application.
 * Hence it mustn't do anything that can't be done twice.
 */

GLOBAL(void)
jpeg_calc_jpeg_dimensions(j_compress_ptr cinfo)
/* Do computations that are needed before master selection phase */
{
  int data_unit = cinfo->master->lossless ? 1 : DCTSIZE;

  /* Hardwire it to "no scaling" */
  cinfo->jpeg_width = cinfo->image_width;
  cinfo->jpeg_height = cinfo->image_height;
  cinfo->min_DCT_h_scaled_size = data_unit;
  cinfo->min_DCT_v_scaled_size = data_unit;
}
#endif


LOCAL(boolean)
using_std_huff_tables(j_compress_ptr cinfo)
{
  int i;

  static const UINT8 bits_dc_luminance[17] = {
    /* 0-base */ 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
  };
  static const UINT8 val_dc_luminance[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
  };

  static const UINT8 bits_dc_chrominance[17] = {
    /* 0-base */ 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
  };
  static const UINT8 val_dc_chrominance[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
  };

  static const UINT8 bits_ac_luminance[17] = {
    /* 0-base */ 0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d
  };
  static const UINT8 val_ac_luminance[] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
  };

  static const UINT8 bits_ac_chrominance[17] = {
    /* 0-base */ 0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77
  };
  static const UINT8 val_ac_chrominance[] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
  };

  if (cinfo->dc_huff_tbl_ptrs[0] == NULL ||
      cinfo->ac_huff_tbl_ptrs[0] == NULL ||
      cinfo->dc_huff_tbl_ptrs[1] == NULL ||
      cinfo->ac_huff_tbl_ptrs[1] == NULL)
    return FALSE;

  for (i = 2; i < NUM_HUFF_TBLS; i++) {
    if (cinfo->dc_huff_tbl_ptrs[i] != NULL ||
        cinfo->ac_huff_tbl_ptrs[i] != NULL)
      return FALSE;
  }

  if (memcmp(cinfo->dc_huff_tbl_ptrs[0]->bits, bits_dc_luminance,
             sizeof(bits_dc_luminance)) ||
      memcmp(cinfo->dc_huff_tbl_ptrs[0]->huffval, val_dc_luminance,
             sizeof(val_dc_luminance)) ||
      memcmp(cinfo->ac_huff_tbl_ptrs[0]->bits, bits_ac_luminance,
             sizeof(bits_ac_luminance)) ||
      memcmp(cinfo->ac_huff_tbl_ptrs[0]->huffval, val_ac_luminance,
             sizeof(val_ac_luminance)) ||
      memcmp(cinfo->dc_huff_tbl_ptrs[1]->bits, bits_dc_chrominance,
             sizeof(bits_dc_chrominance)) ||
      memcmp(cinfo->dc_huff_tbl_ptrs[1]->huffval, val_dc_chrominance,
             sizeof(val_dc_chrominance)) ||
      memcmp(cinfo->ac_huff_tbl_ptrs[1]->bits, bits_ac_chrominance,
             sizeof(bits_ac_chrominance)) ||
      memcmp(cinfo->ac_huff_tbl_ptrs[1]->huffval, val_ac_chrominance,
             sizeof(val_ac_chrominance)))
    return FALSE;

  return TRUE;
}


LOCAL(void)
initial_setup(j_compress_ptr cinfo, boolean transcode_only)
/* Do computations that are needed before master selection phase */
{
  int ci;
  jpeg_component_info *compptr;
  long samplesperrow;
  JDIMENSION jd_samplesperrow;
  int data_unit = cinfo->master->lossless ? 1 : DCTSIZE;

#if JPEG_LIB_VERSION >= 70
#if JPEG_LIB_VERSION >= 80
  if (!transcode_only)
#endif
    jpeg_calc_jpeg_dimensions(cinfo);
#endif

  /* Sanity check on image dimensions */
  if (cinfo->_jpeg_height <= 0 || cinfo->_jpeg_width <= 0 ||
      cinfo->num_components <= 0 || cinfo->input_components <= 0)
    ERREXIT(cinfo, JERR_EMPTY_IMAGE);

  /* Make sure image isn't bigger than I can handle */
  if ((long)cinfo->_jpeg_height > (long)JPEG_MAX_DIMENSION ||
      (long)cinfo->_jpeg_width > (long)JPEG_MAX_DIMENSION)
    ERREXIT1(cinfo, JERR_IMAGE_TOO_BIG, (unsigned int)JPEG_MAX_DIMENSION);

  /* Width of an input scanline must be representable as JDIMENSION. */
  samplesperrow = (long)cinfo->image_width * (long)cinfo->input_components;
  jd_samplesperrow = (JDIMENSION)samplesperrow;
  if ((long)jd_samplesperrow != samplesperrow)
    ERREXIT(cinfo, JERR_WIDTH_OVERFLOW);

  /* Lossy JPEG images must have 8 or 12 bits per sample.  Lossless JPEG images
   * can have 2 to 16 bits per sample.
   */
#ifdef C_LOSSLESS_SUPPORTED
  if (cinfo->master->lossless) {
    if (cinfo->data_precision < 2 || cinfo->data_precision > 16)
      ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
  } else
#endif
  {
    if (cinfo->data_precision != 8 && cinfo->data_precision != 12)
      ERREXIT1(cinfo, JERR_BAD_PRECISION, cinfo->data_precision);
  }

  /* Check that number of components won't exceed internal array sizes */
  if (cinfo->num_components > MAX_COMPONENTS)
    ERREXIT2(cinfo, JERR_COMPONENT_COUNT, cinfo->num_components,
             MAX_COMPONENTS);

  /* Compute maximum sampling factors; check factor validity */
  cinfo->max_h_samp_factor = 1;
  cinfo->max_v_samp_factor = 1;
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    if (compptr->h_samp_factor <= 0 ||
        compptr->h_samp_factor > MAX_SAMP_FACTOR ||
        compptr->v_samp_factor <= 0 ||
        compptr->v_samp_factor > MAX_SAMP_FACTOR)
      ERREXIT(cinfo, JERR_BAD_SAMPLING);
    cinfo->max_h_samp_factor = MAX(cinfo->max_h_samp_factor,
                                   compptr->h_samp_factor);
    cinfo->max_v_samp_factor = MAX(cinfo->max_v_samp_factor,
                                   compptr->v_samp_factor);
  }

  /* Compute dimensions of components */
  for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
       ci++, compptr++) {
    /* Fill in the correct component_index value; don't rely on application */
    compptr->component_index = ci;
    /* For compression, we never do DCT scaling. */
#if JPEG_LIB_VERSION >= 70
    compptr->DCT_h_scaled_size = compptr->DCT_v_scaled_size = data_unit;
#else
    compptr->DCT_scaled_size = data_unit;
#endif
    /* Size in data units */
    compptr->width_in_blocks = (JDIMENSION)
      jdiv_round_up((long)cinfo->_jpeg_width * (long)compptr->h_samp_factor,
                    (long)(cinfo->max_h_samp_factor * data_unit));
    compptr->height_in_blocks = (JDIMENSION)
      jdiv_round_up((long)cinfo->_jpeg_height * (long)compptr->v_samp_factor,
                    (long)(cinfo->max_v_samp_factor * data_unit));
    /* Size in samples */
    compptr->downsampled_width = (JDIMENSION)
      jdiv_round_up((long)cinfo->_jpeg_width * (long)compptr->h_samp_factor,
                    (long)cinfo->max_h_samp_factor);
    compptr->downsampled_height = (JDIMENSION)
      jdiv_round_up((long)cinfo->_jpeg_height * (long)compptr->v_samp_factor,
                    (long)cinfo->max_v_samp_factor);
    /* Mark component needed (this flag isn't actually used for compression) */
    compptr->component_needed = TRUE;
  }

  /* Compute number of fully interleaved MCU rows (number of times that
   * main controller will call coefficient or difference controller).
   */
  cinfo->total_iMCU_rows = (JDIMENSION)
    jdiv_round_up((long)cinfo->_jpeg_height,
                  (long)(cinfo->max_v_samp_factor * data_unit));
}


#if defined(C_MULTISCAN_FILES_SUPPORTED) || defined(C_LOSSLESS_SUPPORTED)
#define NEED_SCAN_SCRIPT
#endif

#ifdef NEED_SCAN_SCRIPT

LOCAL(void)
validate_script(j_compress_ptr cinfo)
/* Verify that the scan script in cinfo->scan_info[] is valid; also
 * determine whether it uses progressive JPEG, and set cinfo->progressive_mode.
 */
{
  const jpeg_scan_info *scanptr;
  int scanno, ncomps, ci, coefi, thisi;
  int Ss, Se, Ah, Al;
  boolean component_sent[MAX_COMPONENTS];
#ifdef C_PROGRESSIVE_SUPPORTED
  int *last_bitpos_ptr;
  int last_bitpos[MAX_COMPONENTS][DCTSIZE2];
  /* -1 until that coefficient has been seen; then last Al for it */
#endif

  if (cinfo->num_scans <= 0)
    ERREXIT1(cinfo, JERR_BAD_SCAN_SCRIPT, 0);

#ifndef C_MULTISCAN_FILES_SUPPORTED
  if (cinfo->num_scans > 1)
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif

  scanptr = cinfo->scan_info;
  if (scanptr->Ss != 0 && scanptr->Se == 0) {
#ifdef C_LOSSLESS_SUPPORTED
    cinfo->master->lossless = TRUE;
    cinfo->progressive_mode = FALSE;
    for (ci = 0; ci < cinfo->num_components; ci++)
      component_sent[ci] = FALSE;
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
  }
  /* For sequential JPEG, all scans must have Ss=0, Se=DCTSIZE2-1;
   * for progressive JPEG, no scan can have this.
   */
  else if (scanptr->Ss != 0 || scanptr->Se != DCTSIZE2 - 1) {
#ifdef C_PROGRESSIVE_SUPPORTED
    cinfo->progressive_mode = TRUE;
    cinfo->master->lossless = FALSE;
    last_bitpos_ptr = &last_bitpos[0][0];
    for (ci = 0; ci < cinfo->num_components; ci++)
      for (coefi = 0; coefi < DCTSIZE2; coefi++)
        *last_bitpos_ptr++ = -1;
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
  } else {
    cinfo->progressive_mode = cinfo->master->lossless = FALSE;
    for (ci = 0; ci < cinfo->num_components; ci++)
      component_sent[ci] = FALSE;
  }

  for (scanno = 1; scanno <= cinfo->num_scans; scanptr++, scanno++) {
    /* Validate component indexes */
    ncomps = scanptr->comps_in_scan;
    if (ncomps <= 0 || ncomps > MAX_COMPS_IN_SCAN)
      ERREXIT2(cinfo, JERR_COMPONENT_COUNT, ncomps, MAX_COMPS_IN_SCAN);
    for (ci = 0; ci < ncomps; ci++) {
      thisi = scanptr->component_index[ci];
      if (thisi < 0 || thisi >= cinfo->num_components)
        ERREXIT1(cinfo, JERR_BAD_SCAN_SCRIPT, scanno);
      /* Components must appear in SOF order within each scan */
      if (ci > 0 && thisi <= scanptr->component_index[ci - 1])
        ERREXIT1(cinfo, JERR_BAD_SCAN_SCRIPT, scanno);
    }
    /* Validate progression parameters */
    Ss = scanptr->Ss;
    Se = scanptr->Se;
    Ah = scanptr->Ah;
    Al = scanptr->Al;
    if (cinfo->progressive_mode) {
#ifdef C_PROGRESSIVE_SUPPORTED
      /* Rec. ITU-T T.81 | ISO/IEC 10918-1 simply gives the ranges 0..13 for Ah
       * and Al, but that seems wrong: the upper bound ought to depend on data
       * precision.  Perhaps they really meant 0..N+1 for N-bit precision.
       * Here we allow 0..10 for 8-bit data; Al larger than 10 results in
       * out-of-range reconstructed DC values during the first DC scan,
       * which might cause problems for some decoders.
       */
      int max_Ah_Al = cinfo->data_precision == 12 ? 13 : 10;

      if (Ss < 0 || Ss >= DCTSIZE2 || Se < Ss || Se >= DCTSIZE2 ||
          Ah < 0 || Ah > max_Ah_Al || Al < 0 || Al > max_Ah_Al)
        ERREXIT1(cinfo, JERR_BAD_PROG_SCRIPT, scanno);
      if (Ss == 0) {
        if (Se != 0)            /* DC and AC together not OK */
          ERREXIT1(cinfo, JERR_BAD_PROG_SCRIPT, scanno);
      } else {
        if (ncomps != 1)        /* AC scans must be for only one component */
          ERREXIT1(cinfo, JERR_BAD_PROG_SCRIPT, scanno);
      }
      for (ci = 0; ci < ncomps; ci++) {
        last_bitpos_ptr = &last_bitpos[scanptr->component_index[ci]][0];
        if (Ss != 0 && last_bitpos_ptr[0] < 0) /* AC without prior DC scan */
          ERREXIT1(cinfo, JERR_BAD_PROG_SCRIPT, scanno);
        for (coefi = Ss; coefi <= Se; coefi++) {
          if (last_bitpos_ptr[coefi] < 0) {
            /* first scan of this coefficient */
            if (Ah != 0)
              ERREXIT1(cinfo, JERR_BAD_PROG_SCRIPT, scanno);
          } else {
            /* not first scan */
            if (Ah != last_bitpos_ptr[coefi] || Al != Ah - 1)
              ERREXIT1(cinfo, JERR_BAD_PROG_SCRIPT, scanno);
          }
          last_bitpos_ptr[coefi] = Al;
        }
      }
#endif
    } else {
#ifdef C_LOSSLESS_SUPPORTED
      if (cinfo->master->lossless) {
        /* The JPEG spec simply gives the range 0..15 for Al (Pt), but that
         * seems wrong: the upper bound ought to depend on data precision.
         * Perhaps they really meant 0..N-1 for N-bit precision, which is what
         * we allow here.  Values greater than or equal to the data precision
         * will result in a blank image.
         */
        if (Ss < 1 || Ss > 7 ||         /* predictor selection value */
            Se != 0 || Ah != 0 ||
            Al < 0 || Al >= cinfo->data_precision) /* point transform */
          ERREXIT1(cinfo, JERR_BAD_PROG_SCRIPT, scanno);
      } else
#endif
      {
        /* For sequential JPEG, all progression parameters must be these: */
        if (Ss != 0 || Se != DCTSIZE2 - 1 || Ah != 0 || Al != 0)
          ERREXIT1(cinfo, JERR_BAD_PROG_SCRIPT, scanno);
      }
      /* Make sure components are not sent twice */
      for (ci = 0; ci < ncomps; ci++) {
        thisi = scanptr->component_index[ci];
        if (component_sent[thisi])
          ERREXIT1(cinfo, JERR_BAD_SCAN_SCRIPT, scanno);
        component_sent[thisi] = TRUE;
      }
    }
  }

  /* Now verify that everything got sent. */
  if (cinfo->progressive_mode) {
#ifdef C_PROGRESSIVE_SUPPORTED
    /* For progressive mode, we only check that at least some DC data
     * got sent for each component; the spec does not require that all bits
     * of all coefficients be transmitted.  Would it be wiser to enforce
     * transmission of all coefficient bits??
     */
    for (ci = 0; ci < cinfo->num_components; ci++) {
      if (last_bitpos[ci][0] < 0)
        ERREXIT(cinfo, JERR_MISSING_DATA);
    }
#endif
  } else {
    for (ci = 0; ci < cinfo->num_components; ci++) {
      if (!component_sent[ci])
        ERREXIT(cinfo, JERR_MISSING_DATA);
    }
  }
}

#endif /* NEED_SCAN_SCRIPT */


LOCAL(void)
select_scan_parameters(j_compress_ptr cinfo)
/* Set up the scan parameters for the current scan */
{
  int ci;

#ifdef NEED_SCAN_SCRIPT
  if (cinfo->scan_info != NULL) {
    /* Prepare for current scan --- the script is already validated */
    my_master_ptr master = (my_master_ptr)cinfo->master;
    const jpeg_scan_info *scanptr = cinfo->scan_info + master->scan_number;

    cinfo->comps_in_scan = scanptr->comps_in_scan;
    for (ci = 0; ci < scanptr->comps_in_scan; ci++) {
      cinfo->cur_comp_info[ci] =
        &cinfo->comp_info[scanptr->component_index[ci]];
    }
    cinfo->Ss = scanptr->Ss;
    cinfo->Se = scanptr->Se;
    cinfo->Ah = scanptr->Ah;
    cinfo->Al = scanptr->Al;
  } else
#endif
  {
    /* Prepare for single sequential-JPEG scan containing all components */
    if (cinfo->num_components > MAX_COMPS_IN_SCAN)
      ERREXIT2(cinfo, JERR_COMPONENT_COUNT, cinfo->num_components,
               MAX_COMPS_IN_SCAN);
    cinfo->comps_in_scan = cinfo->num_components;
    for (ci = 0; ci < cinfo->num_components; ci++) {
      cinfo->cur_comp_info[ci] = &cinfo->comp_info[ci];
    }
    if (!cinfo->master->lossless) {
      cinfo->Ss = 0;
      cinfo->Se = DCTSIZE2 - 1;
      cinfo->Ah = 0;
      cinfo->Al = 0;
    }
  }
}


LOCAL(void)
per_scan_setup(j_compress_ptr cinfo)
/* Do computations that are needed before processing a JPEG scan */
/* cinfo->comps_in_scan and cinfo->cur_comp_info[] are already set */
{
  int ci, mcublks, tmp;
  jpeg_component_info *compptr;
  int data_unit = cinfo->master->lossless ? 1 : DCTSIZE;

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
    compptr->MCU_sample_width = data_unit;
    compptr->last_col_width = 1;
    /* For noninterleaved scans, it is convenient to define last_row_height
     * as the number of block rows present in the last iMCU row.
     */
    tmp = (int)(compptr->height_in_blocks % compptr->v_samp_factor);
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
      jdiv_round_up((long)cinfo->_jpeg_width,
                    (long)(cinfo->max_h_samp_factor * data_unit));
    cinfo->MCU_rows_in_scan = (JDIMENSION)
      jdiv_round_up((long)cinfo->_jpeg_height,
                    (long)(cinfo->max_v_samp_factor * data_unit));

    cinfo->blocks_in_MCU = 0;

    for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
      compptr = cinfo->cur_comp_info[ci];
      /* Sampling factors give # of blocks of component in each MCU */
      compptr->MCU_width = compptr->h_samp_factor;
      compptr->MCU_height = compptr->v_samp_factor;
      compptr->MCU_blocks = compptr->MCU_width * compptr->MCU_height;
      compptr->MCU_sample_width = compptr->MCU_width * data_unit;
      /* Figure number of non-dummy blocks in last MCU column & row */
      tmp = (int)(compptr->width_in_blocks % compptr->MCU_width);
      if (tmp == 0) tmp = compptr->MCU_width;
      compptr->last_col_width = tmp;
      tmp = (int)(compptr->height_in_blocks % compptr->MCU_height);
      if (tmp == 0) tmp = compptr->MCU_height;
      compptr->last_row_height = tmp;
      /* Prepare array describing MCU composition */
      mcublks = compptr->MCU_blocks;
      if (cinfo->blocks_in_MCU + mcublks > C_MAX_BLOCKS_IN_MCU)
        ERREXIT(cinfo, JERR_BAD_MCU_SIZE);
      while (mcublks-- > 0) {
        cinfo->MCU_membership[cinfo->blocks_in_MCU++] = ci;
      }
    }

  }

  /* Convert restart specified in rows to actual MCU count. */
  /* Note that count must fit in 16 bits, so we provide limiting. */
  if (cinfo->restart_in_rows > 0) {
    long nominal = (long)cinfo->restart_in_rows * (long)cinfo->MCUs_per_row;
    cinfo->restart_interval = (unsigned int)MIN(nominal, 65535L);
  }
}


/*
 * Per-pass setup.
 * This is called at the beginning of each pass.  We determine which modules
 * will be active during this pass and give them appropriate start_pass calls.
 * We also set is_last_pass to indicate whether any more passes will be
 * required.
 */

METHODDEF(void)
prepare_for_pass(j_compress_ptr cinfo)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;

  switch (master->pass_type) {
  case main_pass:
    /* Initial pass: will collect input data, and do either Huffman
     * optimization or data output for the first scan.
     */
    select_scan_parameters(cinfo);
    per_scan_setup(cinfo);
    if (!cinfo->raw_data_in) {
      (*cinfo->cconvert->start_pass) (cinfo);
      (*cinfo->downsample->start_pass) (cinfo);
      (*cinfo->prep->start_pass) (cinfo, JBUF_PASS_THRU);
    }
    (*cinfo->fdct->start_pass) (cinfo);
    (*cinfo->entropy->start_pass) (cinfo, cinfo->optimize_coding);
    (*cinfo->coef->start_pass) (cinfo,
                                (master->total_passes > 1 ?
                                 JBUF_SAVE_AND_PASS : JBUF_PASS_THRU));
    (*cinfo->main->start_pass) (cinfo, JBUF_PASS_THRU);
    if (cinfo->optimize_coding) {
      /* No immediate data output; postpone writing frame/scan headers */
      master->pub.call_pass_startup = FALSE;
    } else {
      /* Will write frame/scan headers at first jpeg_write_scanlines call */
      master->pub.call_pass_startup = TRUE;
    }
    break;
#ifdef ENTROPY_OPT_SUPPORTED
  case huff_opt_pass:
    /* Do Huffman optimization for a scan after the first one. */
    select_scan_parameters(cinfo);
    per_scan_setup(cinfo);
    if (cinfo->Ss != 0 || cinfo->Ah == 0 || cinfo->arith_code ||
        cinfo->master->lossless) {
      (*cinfo->entropy->start_pass) (cinfo, TRUE);
      (*cinfo->coef->start_pass) (cinfo, JBUF_CRANK_DEST);
      master->pub.call_pass_startup = FALSE;
      break;
    }
    /* Special case: Huffman DC refinement scans need no Huffman table
     * and therefore we can skip the optimization pass for them.
     */
    master->pass_type = output_pass;
    master->pass_number++;
#endif
    FALLTHROUGH                 /*FALLTHROUGH*/
  case output_pass:
    /* Do a data-output pass. */
    /* We need not repeat per-scan setup if prior optimization pass did it. */
    if (!cinfo->optimize_coding) {
      select_scan_parameters(cinfo);
      per_scan_setup(cinfo);
    }
    (*cinfo->entropy->start_pass) (cinfo, FALSE);
    (*cinfo->coef->start_pass) (cinfo, JBUF_CRANK_DEST);
    /* We emit frame/scan headers now */
    if (master->scan_number == 0)
      (*cinfo->marker->write_frame_header) (cinfo);
    (*cinfo->marker->write_scan_header) (cinfo);
    master->pub.call_pass_startup = FALSE;
    break;
  default:
    ERREXIT(cinfo, JERR_NOT_COMPILED);
  }

  master->pub.is_last_pass = (master->pass_number == master->total_passes - 1);

  /* Set up progress monitor's pass info if present */
  if (cinfo->progress != NULL) {
    cinfo->progress->completed_passes = master->pass_number;
    cinfo->progress->total_passes = master->total_passes;
  }
}


/*
 * Special start-of-pass hook.
 * This is called by jpeg_write_scanlines if call_pass_startup is TRUE.
 * In single-pass processing, we need this hook because we don't want to
 * write frame/scan headers during jpeg_start_compress; we want to let the
 * application write COM markers etc. between jpeg_start_compress and the
 * jpeg_write_scanlines loop.
 * In multi-pass processing, this routine is not used.
 */

METHODDEF(void)
pass_startup(j_compress_ptr cinfo)
{
  cinfo->master->call_pass_startup = FALSE; /* reset flag so call only once */

  (*cinfo->marker->write_frame_header) (cinfo);
  (*cinfo->marker->write_scan_header) (cinfo);
}


/*
 * Finish up at end of pass.
 */

METHODDEF(void)
finish_pass_master(j_compress_ptr cinfo)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;

  /* The entropy coder always needs an end-of-pass call,
   * either to analyze statistics or to flush its output buffer.
   */
  (*cinfo->entropy->finish_pass) (cinfo);

  /* Update state for next pass */
  switch (master->pass_type) {
  case main_pass:
    /* next pass is either output of scan 0 (after optimization)
     * or output of scan 1 (if no optimization).
     */
    master->pass_type = output_pass;
    if (!cinfo->optimize_coding)
      master->scan_number++;
    break;
  case huff_opt_pass:
    /* next pass is always output of current scan */
    master->pass_type = output_pass;
    break;
  case output_pass:
    /* next pass is either optimization or output of next scan */
    if (cinfo->optimize_coding)
      master->pass_type = huff_opt_pass;
    master->scan_number++;
    break;
  }

  master->pass_number++;
}


/*
 * Initialize master compression control.
 */

GLOBAL(void)
jinit_c_master_control(j_compress_ptr cinfo, boolean transcode_only)
{
  my_master_ptr master = (my_master_ptr)cinfo->master;
  boolean empty_huff_tables = TRUE;
  int i;

  master->pub.prepare_for_pass = prepare_for_pass;
  master->pub.pass_startup = pass_startup;
  master->pub.finish_pass = finish_pass_master;
  master->pub.is_last_pass = FALSE;

  if (cinfo->scan_info != NULL) {
#ifdef NEED_SCAN_SCRIPT
    validate_script(cinfo);
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
  } else {
    cinfo->progressive_mode = FALSE;
    cinfo->num_scans = 1;
  }

#ifdef C_LOSSLESS_SUPPORTED
  /* Disable smoothing and subsampling in lossless mode, since those are lossy
   * algorithms.  Set the JPEG colorspace to the input colorspace.  Disable raw
   * (downsampled) data input, because it isn't particularly useful without
   * subsampling and has not been tested in lossless mode.
   */
  if (cinfo->master->lossless) {
    int ci;
    jpeg_component_info *compptr;

    cinfo->raw_data_in = FALSE;
    cinfo->smoothing_factor = 0;
    jpeg_default_colorspace(cinfo);
    for (ci = 0, compptr = cinfo->comp_info; ci < cinfo->num_components;
         ci++, compptr++)
      compptr->h_samp_factor = compptr->v_samp_factor = 1;
  }
#endif

  /* Validate parameters, determine derived values */
  initial_setup(cinfo, transcode_only);

  if (cinfo->arith_code)
    cinfo->optimize_coding = FALSE;
  else {
    if (cinfo->master->lossless ||      /*  TEMPORARY HACK ??? */
        cinfo->progressive_mode)
      cinfo->optimize_coding = TRUE; /* assume default tables no good for
                                        progressive mode or lossless mode */
    for (i = 0; i < NUM_HUFF_TBLS; i++) {
      if (cinfo->dc_huff_tbl_ptrs[i] != NULL ||
          cinfo->ac_huff_tbl_ptrs[i] != NULL) {
        empty_huff_tables = FALSE;
        break;
      }
    }
    if (cinfo->data_precision == 12 && !cinfo->optimize_coding &&
        (empty_huff_tables || using_std_huff_tables(cinfo)))
      cinfo->optimize_coding = TRUE; /* assume default tables no good for
                                        12-bit data precision */
  }

  /* Initialize my private state */
  if (transcode_only) {
    /* no main pass in transcoding */
    if (cinfo->optimize_coding)
      master->pass_type = huff_opt_pass;
    else
      master->pass_type = output_pass;
  } else {
    /* for normal compression, first pass is always this type: */
    master->pass_type = main_pass;
  }
  master->scan_number = 0;
  master->pass_number = 0;
  if (cinfo->optimize_coding)
    master->total_passes = cinfo->num_scans * 2;
  else
    master->total_passes = cinfo->num_scans;

  master->jpeg_version = PACKAGE_NAME " version " VERSION " (build " BUILD ")";
}
