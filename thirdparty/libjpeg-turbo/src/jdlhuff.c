/*
 * jdlhuff.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2022, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains Huffman entropy decoding routines for lossless JPEG.
 *
 * Much of the complexity here has to do with supporting input suspension.
 * If the data source module demands suspension, we want to be able to back
 * up to the start of the current MCU.  To do this, we copy state variables
 * into local working storage, and update them back to the permanent
 * storage only upon successful completion of an MCU.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#include "jlossls.h"            /* Private declarations for lossless codec */
#include "jdhuff.h"             /* Declarations shared with jd*huff.c */


#ifdef D_LOSSLESS_SUPPORTED

typedef struct {
  int ci, yoffset, MCU_width;
} lhd_output_ptr_info;

/*
 * Expanded entropy decoder object for Huffman decoding in lossless mode.
 */

typedef struct {
  struct jpeg_entropy_decoder pub; /* public fields */

  /* These fields are loaded into local variables at start of each MCU.
   * In case of suspension, we exit WITHOUT updating them.
   */
  bitread_perm_state bitstate;  /* Bit buffer at start of MCU */

  /* Pointers to derived tables (these workspaces have image lifespan) */
  d_derived_tbl *derived_tbls[NUM_HUFF_TBLS];

  /* Precalculated info set up by start_pass for use in decode_mcus: */

  /* Pointers to derived tables to be used for each data unit within an MCU */
  d_derived_tbl *cur_tbls[D_MAX_BLOCKS_IN_MCU];

  /* Pointers to the proper output difference row for each group of data units
   * within an MCU.  For each component, there are Vi groups of Hi data units.
   */
  JDIFFROW output_ptr[D_MAX_BLOCKS_IN_MCU];

  /* Number of output pointers in use for the current MCU.  This is the sum
   * of all Vi in the MCU.
   */
  int num_output_ptrs;

  /* Information used for positioning the output pointers within the output
   * difference rows.
   */
  lhd_output_ptr_info output_ptr_info[D_MAX_BLOCKS_IN_MCU];

  /* Index of the proper output pointer for each data unit within an MCU */
  int output_ptr_index[D_MAX_BLOCKS_IN_MCU];

} lhuff_entropy_decoder;

typedef lhuff_entropy_decoder *lhuff_entropy_ptr;


/*
 * Initialize for a Huffman-compressed scan.
 */

METHODDEF(void)
start_pass_lhuff_decoder(j_decompress_ptr cinfo)
{
  lhuff_entropy_ptr entropy = (lhuff_entropy_ptr)cinfo->entropy;
  int ci, dctbl, sampn, ptrn, yoffset, xoffset;
  jpeg_component_info *compptr;

  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    dctbl = compptr->dc_tbl_no;
    /* Make sure requested tables are present */
    if (dctbl < 0 || dctbl >= NUM_HUFF_TBLS ||
        cinfo->dc_huff_tbl_ptrs[dctbl] == NULL)
      ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, dctbl);
    /* Compute derived values for Huffman tables */
    /* We may do this more than once for a table, but it's not expensive */
    jpeg_make_d_derived_tbl(cinfo, TRUE, dctbl,
                            &entropy->derived_tbls[dctbl]);
  }

  /* Precalculate decoding info for each sample in an MCU of this scan */
  for (sampn = 0, ptrn = 0; sampn < cinfo->blocks_in_MCU;) {
    compptr = cinfo->cur_comp_info[cinfo->MCU_membership[sampn]];
    ci = compptr->component_index;
    for (yoffset = 0; yoffset < compptr->MCU_height; yoffset++, ptrn++) {
      /* Precalculate the setup info for each output pointer */
      entropy->output_ptr_info[ptrn].ci = ci;
      entropy->output_ptr_info[ptrn].yoffset = yoffset;
      entropy->output_ptr_info[ptrn].MCU_width = compptr->MCU_width;
      for (xoffset = 0; xoffset < compptr->MCU_width; xoffset++, sampn++) {
        /* Precalculate the output pointer index for each sample */
        entropy->output_ptr_index[sampn] = ptrn;
        /* Precalculate which table to use for each sample */
        entropy->cur_tbls[sampn] = entropy->derived_tbls[compptr->dc_tbl_no];
      }
    }
  }
  entropy->num_output_ptrs = ptrn;

  /* Initialize bitread state variables */
  entropy->bitstate.bits_left = 0;
  entropy->bitstate.get_buffer = 0; /* unnecessary, but keeps Purify quiet */
  entropy->pub.insufficient_data = FALSE;
}


/*
 * Figure F.12: extend sign bit.
 * On some machines, a shift and add will be faster than a table lookup.
 */

#define AVOID_TABLES
#ifdef AVOID_TABLES

#define NEG_1  ((unsigned int)-1)
#define HUFF_EXTEND(x, s) \
  ((x) + ((((x) - (1 << ((s) - 1))) >> 31) & (((NEG_1) << (s)) + 1)))

#else

#define HUFF_EXTEND(x, s) \
  ((x) < extend_test[s] ? (x) + extend_offset[s] : (x))

static const int extend_test[16] = {   /* entry n is 2**(n-1) */
  0, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080,
  0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000
};

static const int extend_offset[16] = { /* entry n is (-1 << n) + 1 */
  0, ((-1) << 1) + 1, ((-1) << 2) + 1, ((-1) << 3) + 1, ((-1) << 4) + 1,
  ((-1) << 5) + 1, ((-1) << 6) + 1, ((-1) << 7) + 1, ((-1) << 8) + 1,
  ((-1) << 9) + 1, ((-1) << 10) + 1, ((-1) << 11) + 1, ((-1) << 12) + 1,
  ((-1) << 13) + 1, ((-1) << 14) + 1, ((-1) << 15) + 1
};

#endif /* AVOID_TABLES */


/*
 * Check for a restart marker & resynchronize decoder.
 * Returns FALSE if must suspend.
 */

LOCAL(boolean)
process_restart(j_decompress_ptr cinfo)
{
  lhuff_entropy_ptr entropy = (lhuff_entropy_ptr)cinfo->entropy;

  /* Throw away any unused bits remaining in bit buffer; */
  /* include any full bytes in next_marker's count of discarded bytes */
  cinfo->marker->discarded_bytes += entropy->bitstate.bits_left / 8;
  entropy->bitstate.bits_left = 0;

  /* Advance past the RSTn marker */
  if (!(*cinfo->marker->read_restart_marker) (cinfo))
    return FALSE;

  /* Reset out-of-data flag, unless read_restart_marker left us smack up
   * against a marker.  In that case we will end up treating the next data
   * segment as empty, and we can avoid producing bogus output pixels by
   * leaving the flag set.
   */
  if (cinfo->unread_marker == 0)
    entropy->pub.insufficient_data = FALSE;

  return TRUE;
}


/*
 * Decode and return nMCU MCUs' worth of Huffman-compressed differences.
 * Each MCU is also disassembled and placed accordingly in diff_buf.
 *
 * MCU_col_num specifies the column of the first MCU being requested within
 * the MCU row.  This tells us where to position the output row pointers in
 * diff_buf.
 *
 * Returns the number of MCUs decoded.  This may be less than nMCU MCUs if
 * data source requested suspension.  In that case no changes have been made
 * to permanent state.  (Exception: some output differences may already have
 * been assigned.  This is harmless for this module, since we'll just
 * re-assign them on the next call.)
 */

METHODDEF(JDIMENSION)
decode_mcus(j_decompress_ptr cinfo, JDIFFIMAGE diff_buf,
            JDIMENSION MCU_row_num, JDIMENSION MCU_col_num, JDIMENSION nMCU)
{
  lhuff_entropy_ptr entropy = (lhuff_entropy_ptr)cinfo->entropy;
  int sampn, ci, yoffset, MCU_width, ptrn;
  JDIMENSION mcu_num;
  BITREAD_STATE_VARS;

  /* Set output pointer locations based on MCU_col_num */
  for (ptrn = 0; ptrn < entropy->num_output_ptrs; ptrn++) {
    ci = entropy->output_ptr_info[ptrn].ci;
    yoffset = entropy->output_ptr_info[ptrn].yoffset;
    MCU_width = entropy->output_ptr_info[ptrn].MCU_width;
    entropy->output_ptr[ptrn] =
      diff_buf[ci][MCU_row_num + yoffset] + (MCU_col_num * MCU_width);
  }

  /*
   * If we've run out of data, zero out the buffers and return.
   * By resetting the undifferencer, the output samples will be CENTERJSAMPLE.
   *
   * NB: We should find a way to do this without interacting with the
   * undifferencer module directly.
   */
  if (entropy->pub.insufficient_data) {
    for (ptrn = 0; ptrn < entropy->num_output_ptrs; ptrn++)
      jzero_far((void FAR *)entropy->output_ptr[ptrn],
                nMCU * entropy->output_ptr_info[ptrn].MCU_width *
                sizeof(JDIFF));

    (*cinfo->idct->start_pass) (cinfo);

  } else {

    /* Load up working state */
    BITREAD_LOAD_STATE(cinfo, entropy->bitstate);

    /* Outer loop handles the number of MCUs requested */

    for (mcu_num = 0; mcu_num < nMCU; mcu_num++) {

      /* Inner loop handles the samples in the MCU */
      for (sampn = 0; sampn < cinfo->blocks_in_MCU; sampn++) {
        d_derived_tbl *dctbl = entropy->cur_tbls[sampn];
        register int s, r;

        /* Section H.2.2: decode the sample difference */
        HUFF_DECODE(s, br_state, dctbl, return mcu_num, label1);
        if (s) {
          if (s == 16)  /* special case: always output 32768 */
            s = 32768;
          else {        /* normal case: fetch subsequent bits */
            CHECK_BIT_BUFFER(br_state, s, return mcu_num);
            r = GET_BITS(s);
            s = HUFF_EXTEND(r, s);
          }
        }

        /* Output the sample difference */
        *entropy->output_ptr[entropy->output_ptr_index[sampn]]++ = (JDIFF)s;
      }

      /* Completed MCU, so update state */
      BITREAD_SAVE_STATE(cinfo, entropy->bitstate);
    }
  }

 return nMCU;
}


/*
 * Module initialization routine for lossless mode Huffman entropy decoding.
 */

GLOBAL(void)
jinit_lhuff_decoder(j_decompress_ptr cinfo)
{
  lhuff_entropy_ptr entropy;
  int i;

  entropy = (lhuff_entropy_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(lhuff_entropy_decoder));
  cinfo->entropy = (struct jpeg_entropy_decoder *)entropy;
  entropy->pub.start_pass = start_pass_lhuff_decoder;
  entropy->pub.decode_mcus = decode_mcus;
  entropy->pub.process_restart = process_restart;

  /* Mark tables unallocated */
  for (i = 0; i < NUM_HUFF_TBLS; i++) {
    entropy->derived_tbls[i] = NULL;
  }
}

#endif /* D_LOSSLESS_SUPPORTED */
