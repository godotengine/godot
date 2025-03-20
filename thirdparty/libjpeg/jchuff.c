/*
 * jchuff.c
 *
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Modified 2006-2023 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains Huffman entropy encoding routines.
 * Both sequential and progressive modes are supported in this single module.
 *
 * Much of the complexity here has to do with supporting output suspension.
 * If the data destination module demands suspension, we want to be able to
 * back up to the start of the current MCU.  To do this, we copy state
 * variables into local working storage, and update them back to the
 * permanent JPEG objects only upon successful completion of an MCU.
 *
 * We do not support output suspension for the progressive JPEG mode, since
 * the library currently does not allow multiple-scan files to be written
 * with output suspension.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


/* The legal range of a DCT coefficient is
 *  -1024 .. +1023  for 8-bit sample data precision;
 * -16384 .. +16383 for 12-bit sample data precision.
 * Hence the magnitude should always fit in sample data precision + 2 bits.
 */

/* Derived data constructed for each Huffman table */

typedef struct {
  unsigned int ehufco[256];	/* code for each symbol */
  char ehufsi[256];		/* length of code for each symbol */
  /* If no code has been allocated for a symbol S, ehufsi[S] contains 0 */
} c_derived_tbl;


/* Expanded entropy encoder object for Huffman encoding.
 *
 * The savable_state subrecord contains fields that change within an MCU,
 * but must not be updated permanently until we complete the MCU.
 */

typedef struct {
  INT32 put_buffer;		/* current bit-accumulation buffer */
  int put_bits;			/* # of bits now in it */
  int last_dc_val[MAX_COMPS_IN_SCAN]; /* last DC coef for each component */
} savable_state;

/* This macro is to work around compilers with missing or broken
 * structure assignment.  You'll need to fix this code if you have
 * such a compiler and you change MAX_COMPS_IN_SCAN.
 */

#ifndef NO_STRUCT_ASSIGN
#define ASSIGN_STATE(dest,src)  ((dest) = (src))
#else
#if MAX_COMPS_IN_SCAN == 4
#define ASSIGN_STATE(dest,src)  \
	((dest).put_buffer = (src).put_buffer, \
	 (dest).put_bits = (src).put_bits, \
	 (dest).last_dc_val[0] = (src).last_dc_val[0], \
	 (dest).last_dc_val[1] = (src).last_dc_val[1], \
	 (dest).last_dc_val[2] = (src).last_dc_val[2], \
	 (dest).last_dc_val[3] = (src).last_dc_val[3])
#endif
#endif


typedef struct {
  struct jpeg_entropy_encoder pub; /* public fields */

  savable_state saved;		/* Bit buffer & DC state at start of MCU */

  /* These fields are NOT loaded into local working state. */
  unsigned int restarts_to_go;	/* MCUs left in this restart interval */
  int next_restart_num;		/* next restart number to write (0-7) */

  /* Pointers to derived tables (these workspaces have image lifespan) */
  c_derived_tbl * dc_derived_tbls[NUM_HUFF_TBLS];
  c_derived_tbl * ac_derived_tbls[NUM_HUFF_TBLS];

  /* Statistics tables for optimization */
  long * dc_count_ptrs[NUM_HUFF_TBLS];
  long * ac_count_ptrs[NUM_HUFF_TBLS];

  /* Following fields used only in progressive mode */

  /* Mode flag: TRUE for optimization, FALSE for actual data output */
  boolean gather_statistics;

  /* next_output_byte/free_in_buffer are local copies of cinfo->dest fields.
   */
  JOCTET * next_output_byte;	/* => next byte to write in buffer */
  size_t free_in_buffer;	/* # of byte spaces remaining in buffer */
  j_compress_ptr cinfo;		/* link to cinfo (needed for dump_buffer) */

  /* Coding status for AC components */
  int ac_tbl_no;		/* the table number of the single component */
  unsigned int EOBRUN;		/* run length of EOBs */
  unsigned int BE;		/* # of buffered correction bits before MCU */
  char * bit_buffer;		/* buffer for correction bits (1 per char) */
  /* packing correction bits tightly would save some space but cost time... */
} huff_entropy_encoder;

typedef huff_entropy_encoder * huff_entropy_ptr;

/* Working state while writing an MCU (sequential mode).
 * This struct contains all the fields that are needed by subroutines.
 */

typedef struct {
  JOCTET * next_output_byte;	/* => next byte to write in buffer */
  size_t free_in_buffer;	/* # of byte spaces remaining in buffer */
  savable_state cur;		/* Current bit buffer & DC state */
  j_compress_ptr cinfo;		/* dump_buffer needs access to this */
} working_state;

/* MAX_CORR_BITS is the number of bits the AC refinement correction-bit
 * buffer can hold.  Larger sizes may slightly improve compression, but
 * 1000 is already well into the realm of overkill.
 * The minimum safe size is 64 bits.
 */

#define MAX_CORR_BITS  1000	/* Max # of correction bits I can buffer */

/* IRIGHT_SHIFT is like RIGHT_SHIFT, but works on int rather than INT32.
 * We assume that int right shift is unsigned if INT32 right shift is,
 * which should be safe.
 */

#ifdef RIGHT_SHIFT_IS_UNSIGNED
#define ISHIFT_TEMPS	int ishift_temp;
#define IRIGHT_SHIFT(x,shft)  \
	((ishift_temp = (x)) < 0 ? \
	 (ishift_temp >> (shft)) | ((~0) << (16-(shft))) : \
	 (ishift_temp >> (shft)))
#else
#define ISHIFT_TEMPS
#define IRIGHT_SHIFT(x,shft)	((x) >> (shft))
#endif


/*
 * Compute the derived values for a Huffman table.
 * This routine also performs some validation checks on the table.
 */

LOCAL(void)
jpeg_make_c_derived_tbl (j_compress_ptr cinfo, boolean isDC, int tblno,
			 c_derived_tbl ** pdtbl)
{
  JHUFF_TBL *htbl;
  c_derived_tbl *dtbl;
  int p, i, l, lastp, si, maxsymbol;
  char huffsize[257];
  unsigned int huffcode[257];
  unsigned int code;

  /* Note that huffsize[] and huffcode[] are filled in code-length order,
   * paralleling the order of the symbols themselves in htbl->huffval[].
   */

  /* Find the input Huffman table */
  if (tblno < 0 || tblno >= NUM_HUFF_TBLS)
    ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, tblno);
  htbl =
    isDC ? cinfo->dc_huff_tbl_ptrs[tblno] : cinfo->ac_huff_tbl_ptrs[tblno];
  if (htbl == NULL)
    htbl = jpeg_std_huff_table((j_common_ptr) cinfo, isDC, tblno);

  /* Allocate a workspace if we haven't already done so. */
  if (*pdtbl == NULL)
    *pdtbl = (c_derived_tbl *) (*cinfo->mem->alloc_small)
      ((j_common_ptr) cinfo, JPOOL_IMAGE, SIZEOF(c_derived_tbl));
  dtbl = *pdtbl;
  
  /* Figure C.1: make table of Huffman code length for each symbol */

  p = 0;
  for (l = 1; l <= 16; l++) {
    i = (int) htbl->bits[l];
    if (i < 0 || p + i > 256)	/* protect against table overrun */
      ERREXIT(cinfo, JERR_BAD_HUFF_TABLE);
    while (i--)
      huffsize[p++] = (char) l;
  }
  huffsize[p] = 0;
  lastp = p;
  
  /* Figure C.2: generate the codes themselves */
  /* We also validate that the counts represent a legal Huffman code tree. */

  code = 0;
  si = huffsize[0];
  p = 0;
  while (huffsize[p]) {
    while (((int) huffsize[p]) == si) {
      huffcode[p++] = code;
      code++;
    }
    /* code is now 1 more than the last code used for codelength si; but
     * it must still fit in si bits, since no code is allowed to be all ones.
     */
    if (((INT32) code) >= (((INT32) 1) << si))
      ERREXIT(cinfo, JERR_BAD_HUFF_TABLE);
    code <<= 1;
    si++;
  }
  
  /* Figure C.3: generate encoding tables */
  /* These are code and size indexed by symbol value */

  /* Set all codeless symbols to have code length 0;
   * this lets us detect duplicate VAL entries here, and later
   * allows emit_bits to detect any attempt to emit such symbols.
   */
  MEMZERO(dtbl->ehufsi, SIZEOF(dtbl->ehufsi));

  /* This is also a convenient place to check for out-of-range
   * and duplicated VAL entries.  We allow 0..255 for AC symbols
   * but only 0..15 for DC.  (We could constrain them further
   * based on data depth and mode, but this seems enough.)
   */
  maxsymbol = isDC ? 15 : 255;

  for (p = 0; p < lastp; p++) {
    i = htbl->huffval[p];
    if (i < 0 || i > maxsymbol || dtbl->ehufsi[i])
      ERREXIT(cinfo, JERR_BAD_HUFF_TABLE);
    dtbl->ehufco[i] = huffcode[p];
    dtbl->ehufsi[i] = huffsize[p];
  }
}


/* Outputting bytes to the file.
 * NB: these must be called only when actually outputting,
 * that is, entropy->gather_statistics == FALSE.
 */

/* Emit a byte, taking 'action' if must suspend. */
#define emit_byte_s(state,val,action)  \
	{ *(state)->next_output_byte++ = (JOCTET) (val);  \
	  if (--(state)->free_in_buffer == 0)  \
	    if (! dump_buffer_s(state))  \
	      { action; } }

/* Emit a byte */
#define emit_byte_e(entropy,val)  \
	{ *(entropy)->next_output_byte++ = (JOCTET) (val);  \
	  if (--(entropy)->free_in_buffer == 0)  \
	    dump_buffer_e(entropy); }


LOCAL(boolean)
dump_buffer_s (working_state * state)
/* Empty the output buffer; return TRUE if successful, FALSE if must suspend */
{
  struct jpeg_destination_mgr * dest = state->cinfo->dest;

  if (! (*dest->empty_output_buffer) (state->cinfo))
    return FALSE;
  /* After a successful buffer dump, must reset buffer pointers */
  state->next_output_byte = dest->next_output_byte;
  state->free_in_buffer = dest->free_in_buffer;
  return TRUE;
}


LOCAL(void)
dump_buffer_e (huff_entropy_ptr entropy)
/* Empty the output buffer; we do not support suspension in this case. */
{
  struct jpeg_destination_mgr * dest = entropy->cinfo->dest;

  if (! (*dest->empty_output_buffer) (entropy->cinfo))
    ERREXIT(entropy->cinfo, JERR_CANT_SUSPEND);
  /* After a successful buffer dump, must reset buffer pointers */
  entropy->next_output_byte = dest->next_output_byte;
  entropy->free_in_buffer = dest->free_in_buffer;
}


/* Outputting bits to the file */

/* Only the right 24 bits of put_buffer are used; the valid bits are
 * left-justified in this part.  At most 16 bits can be passed to emit_bits
 * in one call, and we never retain more than 7 bits in put_buffer
 * between calls, so 24 bits are sufficient.
 */

INLINE
LOCAL(boolean)
emit_bits_s (working_state * state, unsigned int code, int size)
/* Emit some bits; return TRUE if successful, FALSE if must suspend */
{
  /* This routine is heavily used, so it's worth coding tightly. */
  register INT32 put_buffer;
  register int put_bits;

  /* if size is 0, caller used an invalid Huffman table entry */
  if (size == 0)
    ERREXIT(state->cinfo, JERR_HUFF_MISSING_CODE);

  /* mask off any extra bits in code */
  put_buffer = ((INT32) code) & ((((INT32) 1) << size) - 1);

  /* new number of bits in buffer */
  put_bits = size + state->cur.put_bits;

  put_buffer <<= 24 - put_bits; /* align incoming bits */

  /* and merge with old buffer contents */
  put_buffer |= state->cur.put_buffer;

  while (put_bits >= 8) {
    int c = (int) ((put_buffer >> 16) & 0xFF);

    emit_byte_s(state, c, return FALSE);
    if (c == 0xFF) {		/* need to stuff a zero byte? */
      emit_byte_s(state, 0, return FALSE);
    }
    put_buffer <<= 8;
    put_bits -= 8;
  }

  state->cur.put_buffer = put_buffer; /* update state variables */
  state->cur.put_bits = put_bits;

  return TRUE;
}


INLINE
LOCAL(void)
emit_bits_e (huff_entropy_ptr entropy, unsigned int code, int size)
/* Emit some bits, unless we are in gather mode */
{
  /* This routine is heavily used, so it's worth coding tightly. */
  register INT32 put_buffer;
  register int put_bits;

  /* if size is 0, caller used an invalid Huffman table entry */
  if (size == 0)
    ERREXIT(entropy->cinfo, JERR_HUFF_MISSING_CODE);

  if (entropy->gather_statistics)
    return;			/* do nothing if we're only getting stats */

  /* mask off any extra bits in code */
  put_buffer = ((INT32) code) & ((((INT32) 1) << size) - 1);

  /* new number of bits in buffer */
  put_bits = size + entropy->saved.put_bits;

  put_buffer <<= 24 - put_bits; /* align incoming bits */

  /* and merge with old buffer contents */
  put_buffer |= entropy->saved.put_buffer;

  while (put_bits >= 8) {
    int c = (int) ((put_buffer >> 16) & 0xFF);

    emit_byte_e(entropy, c);
    if (c == 0xFF) {		/* need to stuff a zero byte? */
      emit_byte_e(entropy, 0);
    }
    put_buffer <<= 8;
    put_bits -= 8;
  }

  entropy->saved.put_buffer = put_buffer; /* update variables */
  entropy->saved.put_bits = put_bits;
}


LOCAL(boolean)
flush_bits_s (working_state * state)
{
  if (! emit_bits_s(state, 0x7F, 7)) /* fill any partial byte with ones */
    return FALSE;
  state->cur.put_buffer = 0;	     /* and reset bit-buffer to empty */
  state->cur.put_bits = 0;
  return TRUE;
}


LOCAL(void)
flush_bits_e (huff_entropy_ptr entropy)
{
  emit_bits_e(entropy, 0x7F, 7); /* fill any partial byte with ones */
  entropy->saved.put_buffer = 0; /* and reset bit-buffer to empty */
  entropy->saved.put_bits = 0;
}


/*
 * Emit (or just count) a Huffman symbol.
 */

INLINE
LOCAL(void)
emit_dc_symbol (huff_entropy_ptr entropy, int tbl_no, int symbol)
{
  if (entropy->gather_statistics)
    entropy->dc_count_ptrs[tbl_no][symbol]++;
  else {
    c_derived_tbl * tbl = entropy->dc_derived_tbls[tbl_no];
    emit_bits_e(entropy, tbl->ehufco[symbol], tbl->ehufsi[symbol]);
  }
}


INLINE
LOCAL(void)
emit_ac_symbol (huff_entropy_ptr entropy, int tbl_no, int symbol)
{
  if (entropy->gather_statistics)
    entropy->ac_count_ptrs[tbl_no][symbol]++;
  else {
    c_derived_tbl * tbl = entropy->ac_derived_tbls[tbl_no];
    emit_bits_e(entropy, tbl->ehufco[symbol], tbl->ehufsi[symbol]);
  }
}


/*
 * Emit bits from a correction bit buffer.
 */

LOCAL(void)
emit_buffered_bits (huff_entropy_ptr entropy, char * bufstart,
		    unsigned int nbits)
{
  if (entropy->gather_statistics)
    return;			/* no real work */

  while (nbits > 0) {
    emit_bits_e(entropy, (unsigned int) (*bufstart), 1);
    bufstart++;
    nbits--;
  }
}


/*
 * Emit any pending EOBRUN symbol.
 */

LOCAL(void)
emit_eobrun (huff_entropy_ptr entropy)
{
  register int temp, nbits;

  if (entropy->EOBRUN > 0) {	/* if there is any pending EOBRUN */
    temp = entropy->EOBRUN;
    nbits = 0;
    while ((temp >>= 1))
      nbits++;
    /* safety check: shouldn't happen given limited correction-bit buffer */
    if (nbits > 14)
      ERREXIT(entropy->cinfo, JERR_HUFF_MISSING_CODE);

    emit_ac_symbol(entropy, entropy->ac_tbl_no, nbits << 4);
    if (nbits)
      emit_bits_e(entropy, entropy->EOBRUN, nbits);

    entropy->EOBRUN = 0;

    /* Emit any buffered correction bits */
    emit_buffered_bits(entropy, entropy->bit_buffer, entropy->BE);
    entropy->BE = 0;
  }
}


/*
 * Emit a restart marker & resynchronize predictions.
 */

LOCAL(boolean)
emit_restart_s (working_state * state, int restart_num)
{
  int ci;

  if (! flush_bits_s(state))
    return FALSE;

  emit_byte_s(state, 0xFF, return FALSE);
  emit_byte_s(state, JPEG_RST0 + restart_num, return FALSE);

  /* Re-initialize DC predictions to 0 */
  for (ci = 0; ci < state->cinfo->comps_in_scan; ci++)
    state->cur.last_dc_val[ci] = 0;

  /* The restart counter is not updated until we successfully write the MCU. */

  return TRUE;
}


LOCAL(void)
emit_restart_e (huff_entropy_ptr entropy, int restart_num)
{
  int ci;

  emit_eobrun(entropy);

  if (! entropy->gather_statistics) {
    flush_bits_e(entropy);
    emit_byte_e(entropy, 0xFF);
    emit_byte_e(entropy, JPEG_RST0 + restart_num);
  }

  if (entropy->cinfo->Ss == 0) {
    /* Re-initialize DC predictions to 0 */
    for (ci = 0; ci < entropy->cinfo->comps_in_scan; ci++)
      entropy->saved.last_dc_val[ci] = 0;
  } else {
    /* Re-initialize all AC-related fields to 0 */
    entropy->EOBRUN = 0;
    entropy->BE = 0;
  }
}


/*
 * MCU encoding for DC initial scan (either spectral selection,
 * or first pass of successive approximation).
 */

METHODDEF(boolean)
encode_mcu_DC_first (j_compress_ptr cinfo, JBLOCKARRAY MCU_data)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  register int temp, temp2;
  register int nbits;
  int max_coef_bits;
  int blkn, ci, tbl;
  ISHIFT_TEMPS

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart_e(entropy, entropy->next_restart_num);

  /* Since we're encoding a difference, the range limit is twice as much. */
  max_coef_bits = cinfo->data_precision + 3;

  /* Encode the MCU data blocks */
  for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
    ci = cinfo->MCU_membership[blkn];
    tbl = cinfo->cur_comp_info[ci]->dc_tbl_no;

    /* Compute the DC value after the required point transform by Al.
     * This is simply an arithmetic right shift.
     */
    temp = IRIGHT_SHIFT((int) (MCU_data[blkn][0][0]), cinfo->Al);

    /* DC differences are figured on the point-transformed values. */
    if ((temp2 = temp - entropy->saved.last_dc_val[ci]) == 0) {
      /* Count/emit the Huffman-coded symbol for the number of bits */
      emit_dc_symbol(entropy, tbl, 0);

      continue;
    }

    entropy->saved.last_dc_val[ci] = temp;

    /* Encode the DC coefficient difference per section G.1.2.1 */
    if ((temp = temp2) < 0) {
      temp = -temp;		/* temp is abs value of input */
      /* For a negative input, want temp2 = bitwise complement of abs(input) */
      /* This code assumes we are on a two's complement machine */
      temp2--;
    }

    /* Find the number of bits needed for the magnitude of the coefficient */
    nbits = 0;
    do nbits++;			/* there must be at least one 1 bit */
    while ((temp >>= 1));
    /* Check for out-of-range coefficient values */
    if (nbits > max_coef_bits)
      ERREXIT(cinfo, JERR_BAD_DCT_COEF);

    /* Count/emit the Huffman-coded symbol for the number of bits */
    emit_dc_symbol(entropy, tbl, nbits);

    /* Emit that number of bits of the value, if positive, */
    /* or the complement of its magnitude, if negative. */
    emit_bits_e(entropy, (unsigned int) temp2, nbits);
  }

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/*
 * MCU encoding for AC initial scan (either spectral selection,
 * or first pass of successive approximation).
 */

METHODDEF(boolean)
encode_mcu_AC_first (j_compress_ptr cinfo, JBLOCKARRAY MCU_data)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  const int * natural_order;
  JBLOCKROW block;
  register int temp, temp2;
  register int nbits;
  register int r, k;
  int Se, Al, max_coef_bits;

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart_e(entropy, entropy->next_restart_num);

  Se = cinfo->Se;
  Al = cinfo->Al;
  natural_order = cinfo->natural_order;
  max_coef_bits = cinfo->data_precision + 2;

  /* Encode the MCU data block */
  block = MCU_data[0];

  /* Encode the AC coefficients per section G.1.2.2, fig. G.3 */
  
  r = 0;			/* r = run length of zeros */
   
  for (k = cinfo->Ss; k <= Se; k++) {
    if ((temp = (*block)[natural_order[k]]) == 0) {
      r++;
      continue;
    }
    /* We must apply the point transform by Al.  For AC coefficients this
     * is an integer division with rounding towards 0.  To do this portably
     * in C, we shift after obtaining the absolute value; so the code is
     * interwoven with finding the abs value (temp) and output bits (temp2).
     */
    if (temp < 0) {
      temp = -temp;		/* temp is abs value of input */
      /* Apply the point transform, and watch out for case */
      /* that nonzero coef is zero after point transform. */
      if ((temp >>= Al) == 0) {
	r++;
	continue;
      }
      /* For a negative coef, want temp2 = bitwise complement of abs(coef) */
      temp2 = ~temp;
    } else {
      /* Apply the point transform, and watch out for case */
      /* that nonzero coef is zero after point transform. */
      if ((temp >>= Al) == 0) {
	r++;
	continue;
      }
      temp2 = temp;
    }

    /* Emit any pending EOBRUN */
    if (entropy->EOBRUN > 0)
      emit_eobrun(entropy);
    /* if run length > 15, must emit special run-length-16 codes (0xF0) */
    while (r > 15) {
      emit_ac_symbol(entropy, entropy->ac_tbl_no, 0xF0);
      r -= 16;
    }

    /* Find the number of bits needed for the magnitude of the coefficient */
    nbits = 0;
    do nbits++;			/* there must be at least one 1 bit */
    while ((temp >>= 1));
    /* Check for out-of-range coefficient values */
    if (nbits > max_coef_bits)
      ERREXIT(cinfo, JERR_BAD_DCT_COEF);

    /* Count/emit Huffman symbol for run length / number of bits */
    emit_ac_symbol(entropy, entropy->ac_tbl_no, (r << 4) + nbits);

    /* Emit that number of bits of the value, if positive, */
    /* or the complement of its magnitude, if negative. */
    emit_bits_e(entropy, (unsigned int) temp2, nbits);

    r = 0;			/* reset zero run length */
  }

  if (r > 0) {			/* If there are trailing zeroes, */
    entropy->EOBRUN++;		/* count an EOB */
    if (entropy->EOBRUN == 0x7FFF)
      emit_eobrun(entropy);	/* force it out to avoid overflow */
  }

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/*
 * MCU encoding for DC successive approximation refinement scan.
 * Note: we assume such scans can be multi-component,
 * although the spec is not very clear on the point.
 */

METHODDEF(boolean)
encode_mcu_DC_refine (j_compress_ptr cinfo, JBLOCKARRAY MCU_data)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  int Al, blkn;

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart_e(entropy, entropy->next_restart_num);

  Al = cinfo->Al;

  /* Encode the MCU data blocks */
  for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
    /* We simply emit the Al'th bit of the DC coefficient value. */
    emit_bits_e(entropy, (unsigned int) (MCU_data[blkn][0][0] >> Al), 1);
  }

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/*
 * MCU encoding for AC successive approximation refinement scan.
 */

METHODDEF(boolean)
encode_mcu_AC_refine (j_compress_ptr cinfo, JBLOCKARRAY MCU_data)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  const int * natural_order;
  JBLOCKROW block;
  register int temp;
  register int r, k;
  int Se, Al;
  int EOB;
  char *BR_buffer;
  unsigned int BR;
  int absvalues[DCTSIZE2];

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart_e(entropy, entropy->next_restart_num);

  Se = cinfo->Se;
  Al = cinfo->Al;
  natural_order = cinfo->natural_order;

  /* Encode the MCU data block */
  block = MCU_data[0];

  /* It is convenient to make a pre-pass to determine the transformed
   * coefficients' absolute values and the EOB position.
   */
  EOB = 0;
  for (k = cinfo->Ss; k <= Se; k++) {
    temp = (*block)[natural_order[k]];
    /* We must apply the point transform by Al.  For AC coefficients this
     * is an integer division with rounding towards 0.  To do this portably
     * in C, we shift after obtaining the absolute value.
     */
    if (temp < 0)
      temp = -temp;		/* temp is abs value of input */
    temp >>= Al;		/* apply the point transform */
    absvalues[k] = temp;	/* save abs value for main pass */
    if (temp == 1)
      EOB = k;			/* EOB = index of last newly-nonzero coef */
  }

  /* Encode the AC coefficients per section G.1.2.3, fig. G.7 */
  
  r = 0;			/* r = run length of zeros */
  BR = 0;			/* BR = count of buffered bits added now */
  BR_buffer = entropy->bit_buffer + entropy->BE; /* Append bits to buffer */

  for (k = cinfo->Ss; k <= Se; k++) {
    if ((temp = absvalues[k]) == 0) {
      r++;
      continue;
    }

    /* Emit any required ZRLs, but not if they can be folded into EOB */
    while (r > 15 && k <= EOB) {
      /* emit any pending EOBRUN and the BE correction bits */
      emit_eobrun(entropy);
      /* Emit ZRL */
      emit_ac_symbol(entropy, entropy->ac_tbl_no, 0xF0);
      r -= 16;
      /* Emit buffered correction bits that must be associated with ZRL */
      emit_buffered_bits(entropy, BR_buffer, BR);
      BR_buffer = entropy->bit_buffer; /* BE bits are gone now */
      BR = 0;
    }

    /* If the coef was previously nonzero, it only needs a correction bit.
     * NOTE: a straight translation of the spec's figure G.7 would suggest
     * that we also need to test r > 15.  But if r > 15, we can only get here
     * if k > EOB, which implies that this coefficient is not 1.
     */
    if (temp > 1) {
      /* The correction bit is the next bit of the absolute value. */
      BR_buffer[BR++] = (char) (temp & 1);
      continue;
    }

    /* Emit any pending EOBRUN and the BE correction bits */
    emit_eobrun(entropy);

    /* Count/emit Huffman symbol for run length / number of bits */
    emit_ac_symbol(entropy, entropy->ac_tbl_no, (r << 4) + 1);

    /* Emit output bit for newly-nonzero coef */
    temp = ((*block)[natural_order[k]] < 0) ? 0 : 1;
    emit_bits_e(entropy, (unsigned int) temp, 1);

    /* Emit buffered correction bits that must be associated with this code */
    emit_buffered_bits(entropy, BR_buffer, BR);
    BR_buffer = entropy->bit_buffer; /* BE bits are gone now */
    BR = 0;
    r = 0;			/* reset zero run length */
  }

  if (r > 0 || BR > 0) {	/* If there are trailing zeroes, */
    entropy->EOBRUN++;		/* count an EOB */
    entropy->BE += BR;		/* concat my correction bits to older ones */
    /* We force out the EOB if we risk either:
     * 1. overflow of the EOB counter;
     * 2. overflow of the correction bit buffer during the next MCU.
     */
    if (entropy->EOBRUN == 0x7FFF || entropy->BE > (MAX_CORR_BITS-DCTSIZE2+1))
      emit_eobrun(entropy);
  }

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/* Encode a single block's worth of coefficients */

LOCAL(boolean)
encode_one_block (working_state * state, JCOEFPTR block, int last_dc_val,
		  c_derived_tbl *dctbl, c_derived_tbl *actbl)
{
  register int temp, temp2;
  register int nbits;
  register int r, k;
  int Se = state->cinfo->lim_Se;
  int max_coef_bits = state->cinfo->data_precision + 3;
  const int * natural_order = state->cinfo->natural_order;

  /* Encode the DC coefficient difference per section F.1.2.1 */

  if ((temp = block[0] - last_dc_val) == 0) {
    /* Emit the Huffman-coded symbol for the number of bits */
    if (! emit_bits_s(state, dctbl->ehufco[0], dctbl->ehufsi[0]))
      return FALSE;
  } else {
    if ((temp2 = temp) < 0) {
      temp = -temp;		/* temp is abs value of input */
      /* For a negative input, want temp2 = bitwise complement of abs(input) */
      /* This code assumes we are on a two's complement machine */
      temp2--;
    }

    /* Find the number of bits needed for the magnitude of the coefficient */
    nbits = 0;
    do nbits++;			/* there must be at least one 1 bit */
    while ((temp >>= 1));
    /* Check for out-of-range coefficient values.
     * Since we're encoding a difference, the range limit is twice as much.
     */
    if (nbits > max_coef_bits)
      ERREXIT(state->cinfo, JERR_BAD_DCT_COEF);

    /* Emit the Huffman-coded symbol for the number of bits */
    if (! emit_bits_s(state, dctbl->ehufco[nbits], dctbl->ehufsi[nbits]))
      return FALSE;

    /* Emit that number of bits of the value, if positive, */
    /* or the complement of its magnitude, if negative. */
    if (! emit_bits_s(state, (unsigned int) temp2, nbits))
      return FALSE;
  }

  /* Encode the AC coefficients per section F.1.2.2 */

  r = 0;			/* r = run length of zeros */

  for (k = 1; k <= Se; k++) {
    if ((temp = block[natural_order[k]]) == 0) {
      r++;
      continue;
    }

    /* if run length > 15, must emit special run-length-16 codes (0xF0) */
    while (r > 15) {
      if (! emit_bits_s(state, actbl->ehufco[0xF0], actbl->ehufsi[0xF0]))
	return FALSE;
      r -= 16;
    }

    if ((temp2 = temp) < 0) {
      temp = -temp;		/* temp is abs value of input */
      /* For a negative coef, want temp2 = bitwise complement of abs(coef) */
      /* This code assumes we are on a two's complement machine */
      temp2--;
    }

    /* Find the number of bits needed for the magnitude of the coefficient */
    nbits = 0;
    do nbits++;			/* there must be at least one 1 bit */
    while ((temp >>= 1));
    /* Check for out-of-range coefficient values.
     * Use ">=" instead of ">" so can use the
     * same one larger limit from DC check here.
     */
    if (nbits >= max_coef_bits)
      ERREXIT(state->cinfo, JERR_BAD_DCT_COEF);

    /* Emit Huffman symbol for run length / number of bits */
    temp = (r << 4) + nbits;
    if (! emit_bits_s(state, actbl->ehufco[temp], actbl->ehufsi[temp]))
      return FALSE;

    /* Emit that number of bits of the value, if positive, */
    /* or the complement of its magnitude, if negative. */
    if (! emit_bits_s(state, (unsigned int) temp2, nbits))
      return FALSE;

    r = 0;			/* reset zero run length */
  }

  /* If the last coef(s) were zero, emit an end-of-block code */
  if (r > 0)
    if (! emit_bits_s(state, actbl->ehufco[0], actbl->ehufsi[0]))
      return FALSE;

  return TRUE;
}


/*
 * Encode and output one MCU's worth of Huffman-compressed coefficients.
 */

METHODDEF(boolean)
encode_mcu_huff (j_compress_ptr cinfo, JBLOCKARRAY MCU_data)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  working_state state;
  int blkn, ci;
  jpeg_component_info * compptr;

  /* Load up working state */
  state.next_output_byte = cinfo->dest->next_output_byte;
  state.free_in_buffer = cinfo->dest->free_in_buffer;
  ASSIGN_STATE(state.cur, entropy->saved);
  state.cinfo = cinfo;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0)
      if (! emit_restart_s(&state, entropy->next_restart_num))
	return FALSE;
  }

  /* Encode the MCU data blocks */
  for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
    ci = cinfo->MCU_membership[blkn];
    compptr = cinfo->cur_comp_info[ci];
    if (! encode_one_block(&state,
			   MCU_data[blkn][0], state.cur.last_dc_val[ci],
			   entropy->dc_derived_tbls[compptr->dc_tbl_no],
			   entropy->ac_derived_tbls[compptr->ac_tbl_no]))
      return FALSE;
    /* Update last_dc_val */
    state.cur.last_dc_val[ci] = MCU_data[blkn][0][0];
  }

  /* Completed MCU, so update state */
  cinfo->dest->next_output_byte = state.next_output_byte;
  cinfo->dest->free_in_buffer = state.free_in_buffer;
  ASSIGN_STATE(entropy->saved, state.cur);

  /* Update restart-interval state too */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      entropy->restarts_to_go = cinfo->restart_interval;
      entropy->next_restart_num++;
      entropy->next_restart_num &= 7;
    }
    entropy->restarts_to_go--;
  }

  return TRUE;
}


/*
 * Finish up at the end of a Huffman-compressed scan.
 */

METHODDEF(void)
finish_pass_huff (j_compress_ptr cinfo)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  working_state state;

  if (cinfo->progressive_mode) {
    entropy->next_output_byte = cinfo->dest->next_output_byte;
    entropy->free_in_buffer = cinfo->dest->free_in_buffer;

    /* Flush out any buffered data */
    emit_eobrun(entropy);
    flush_bits_e(entropy);

    cinfo->dest->next_output_byte = entropy->next_output_byte;
    cinfo->dest->free_in_buffer = entropy->free_in_buffer;
  } else {
    /* Load up working state ... flush_bits needs it */
    state.next_output_byte = cinfo->dest->next_output_byte;
    state.free_in_buffer = cinfo->dest->free_in_buffer;
    ASSIGN_STATE(state.cur, entropy->saved);
    state.cinfo = cinfo;

    /* Flush out the last data */
    if (! flush_bits_s(&state))
      ERREXIT(cinfo, JERR_CANT_SUSPEND);

    /* Update state */
    cinfo->dest->next_output_byte = state.next_output_byte;
    cinfo->dest->free_in_buffer = state.free_in_buffer;
    ASSIGN_STATE(entropy->saved, state.cur);
  }
}


/*
 * Huffman coding optimization.
 *
 * We first scan the supplied data and count the number of uses of each symbol
 * that is to be Huffman-coded. (This process MUST agree with the code above.)
 * Then we build a Huffman coding tree for the observed counts.
 * Symbols which are not needed at all for the particular image are not
 * assigned any code, which saves space in the DHT marker as well as in
 * the compressed data.
 */


/* Process a single block's worth of coefficients */

LOCAL(void)
htest_one_block (j_compress_ptr cinfo, JCOEFPTR block, int last_dc_val,
		 long dc_counts[], long ac_counts[])
{
  register int temp;
  register int nbits;
  register int r, k;
  int Se = cinfo->lim_Se;
  int max_coef_bits = cinfo->data_precision + 3;
  const int * natural_order = cinfo->natural_order;

  /* Encode the DC coefficient difference per section F.1.2.1 */

  if ((temp = block[0] - last_dc_val) == 0) {
    /* Count the Huffman symbol for the number of bits */
    dc_counts[0]++;
  } else {
    if (temp < 0)
      temp = -temp;		/* temp is abs value of input */

    /* Find the number of bits needed for the magnitude of the coefficient */
    nbits = 0;
    do nbits++;			/* there must be at least one 1 bit */
    while ((temp >>= 1));
    /* Check for out-of-range coefficient values.
     * Since we're encoding a difference, the range limit is twice as much.
     */
    if (nbits > max_coef_bits)
      ERREXIT(cinfo, JERR_BAD_DCT_COEF);

    /* Count the Huffman symbol for the number of bits */
    dc_counts[nbits]++;
  }

  /* Encode the AC coefficients per section F.1.2.2 */

  r = 0;			/* r = run length of zeros */

  for (k = 1; k <= Se; k++) {
    if ((temp = block[natural_order[k]]) == 0) {
      r++;
      continue;
    }

    /* if run length > 15, must emit special run-length-16 codes (0xF0) */
    while (r > 15) {
      ac_counts[0xF0]++;
      r -= 16;
    }

    if (temp < 0)
      temp = -temp;		/* temp is abs value of input */

    /* Find the number of bits needed for the magnitude of the coefficient */
    nbits = 0;
    do nbits++;			/* there must be at least one 1 bit */
    while ((temp >>= 1));
    /* Check for out-of-range coefficient values.
     * Use ">=" instead of ">" so can use the
     * same one larger limit from DC check here.
     */
    if (nbits >= max_coef_bits)
      ERREXIT(cinfo, JERR_BAD_DCT_COEF);

    /* Count Huffman symbol for run length / number of bits */
    ac_counts[(r << 4) + nbits]++;

    r = 0;			/* reset zero run length */
  }

  /* If the last coef(s) were zero, emit an end-of-block code */
  if (r > 0)
    ac_counts[0]++;
}


/*
 * Trial-encode one MCU's worth of Huffman-compressed coefficients.
 * No data is actually output, so no suspension return is possible.
 */

METHODDEF(boolean)
encode_mcu_gather (j_compress_ptr cinfo, JBLOCKARRAY MCU_data)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  int blkn, ci;
  jpeg_component_info * compptr;

  /* Take care of restart intervals if needed */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0) {
      /* Re-initialize DC predictions to 0 */
      for (ci = 0; ci < cinfo->comps_in_scan; ci++)
	entropy->saved.last_dc_val[ci] = 0;
      /* Update restart state */
      entropy->restarts_to_go = cinfo->restart_interval;
    }
    entropy->restarts_to_go--;
  }

  for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
    ci = cinfo->MCU_membership[blkn];
    compptr = cinfo->cur_comp_info[ci];
    htest_one_block(cinfo, MCU_data[blkn][0], entropy->saved.last_dc_val[ci],
		    entropy->dc_count_ptrs[compptr->dc_tbl_no],
		    entropy->ac_count_ptrs[compptr->ac_tbl_no]);
    entropy->saved.last_dc_val[ci] = MCU_data[blkn][0][0];
  }

  return TRUE;
}


/*
 * Generate the best Huffman code table for the given counts, fill htbl.
 *
 * The JPEG standard requires that no symbol be assigned a codeword of all
 * one bits (so that padding bits added at the end of a compressed segment
 * can't look like a valid code).  Because of the canonical ordering of
 * codewords, this just means that there must be an unused slot in the
 * longest codeword length category.  Section K.2 of the JPEG spec suggests
 * reserving such a slot by pretending that symbol 256 is a valid symbol
 * with count 1.  In theory that's not optimal; giving it count zero but
 * including it in the symbol set anyway should give a better Huffman code.
 * But the theoretically better code actually seems to come out worse in
 * practice, because it produces more all-ones bytes (which incur stuffed
 * zero bytes in the final file).  In any case the difference is tiny.
 *
 * The JPEG standard requires Huffman codes to be no more than 16 bits long.
 * If some symbols have a very small but nonzero probability, the Huffman tree
 * must be adjusted to meet the code length restriction.  We currently use
 * the adjustment method suggested in JPEG section K.2.  This method is *not*
 * optimal; it may not choose the best possible limited-length code.  But
 * typically only very-low-frequency symbols will be given less-than-optimal
 * lengths, so the code is almost optimal.  Experimental comparisons against
 * an optimal limited-length-code algorithm indicate that the difference is
 * microscopic --- usually less than a hundredth of a percent of total size.
 * So the extra complexity of an optimal algorithm doesn't seem worthwhile.
 */

LOCAL(void)
jpeg_gen_optimal_table (j_compress_ptr cinfo, JHUFF_TBL * htbl, long freq[])
{
#define MAX_CLEN 32		/* assumed maximum initial code length */
  UINT8 bits[MAX_CLEN+1];	/* bits[k] = # of symbols with code length k */
  int codesize[257];		/* codesize[k] = code length of symbol k */
  int others[257];		/* next symbol in current branch of tree */
  int c1, c2, i, j;
  UINT8 *p;
  long v;

  freq[256] = 1;		/* make sure 256 has a nonzero count */
  /* Including the pseudo-symbol 256 in the Huffman procedure guarantees
   * that no real symbol is given code-value of all ones, because 256
   * will be placed last in the largest codeword category.
   * In the symbol list build procedure this element serves as sentinel
   * for the zero run loop.
   */

#ifndef DONT_USE_FANCY_HUFF_OPT

  /* Build list of symbols sorted in order of descending frequency */
  /* This approach has several benefits (thank to John Korejwa for the idea):
   *     1.
   * If a codelength category is split during the length limiting procedure
   * below, the feature that more frequent symbols are assigned shorter
   * codewords remains valid for the adjusted code.
   *     2.
   * To reduce consecutive ones in a Huffman data stream (thus reducing the
   * number of stuff bytes in JPEG) it is preferable to follow 0 branches
   * (and avoid 1 branches) as much as possible.  This is easily done by
   * assigning symbols to leaves of the Huffman tree in order of decreasing
   * frequency, with no secondary sort based on codelengths.
   *     3.
   * The symbol list can be built independently from the assignment of code
   * lengths by the Huffman procedure below.
   * Note: The symbol list build procedure must be performed first, because
   * the Huffman procedure assigning the codelengths clobbers the frequency
   * counts!
   */

  /* Here we use the others array as a linked list of nonzero frequencies
   * to be sorted.  Already sorted elements are removed from the list.
   */

  /* Building list */

  /* This item does not correspond to a valid symbol frequency and is used
   * as starting index.
   */
  j = 256;

  for (i = 0;; i++) {
    if (freq[i] == 0)		/* skip zero frequencies */
      continue;
    if (i > 255)
      break;
    others[j] = i;		/* this symbol value */
    j = i;			/* previous symbol value */
  }
  others[j] = -1;		/* mark end of list */

  /* Sorting list */

  p = htbl->huffval;
  while ((c1 = others[256]) >= 0) {
    v = freq[c1];
    i = c1;			/* first symbol value */
    j = 256;			/* pseudo symbol value for starting index */
    while ((c2 = others[c1]) >= 0) {
      if (freq[c2] > v) {
	v = freq[c2];
	i = c2;			/* this symbol value */
	j = c1;			/* previous symbol value */
      }
      c1 = c2;
    }
    others[j] = others[i];	/* remove this symbol i from list */
    *p++ = (UINT8) i;
  }

#endif /* DONT_USE_FANCY_HUFF_OPT */

  /* This algorithm is explained in section K.2 of the JPEG standard */

  MEMZERO(bits, SIZEOF(bits));
  MEMZERO(codesize, SIZEOF(codesize));
  for (i = 0; i < 257; i++)
    others[i] = -1;		/* init links to empty */

  /* Huffman's basic algorithm to assign optimal code lengths to symbols */

  for (;;) {
    /* Find the smallest nonzero frequency, set c1 = its symbol */
    /* In case of ties, take the larger symbol number */
    c1 = -1;
    v = 1000000000L;
    for (i = 0; i <= 256; i++) {
      if (freq[i] && freq[i] <= v) {
	v = freq[i];
	c1 = i;
      }
    }

    /* Find the next smallest nonzero frequency, set c2 = its symbol */
    /* In case of ties, take the larger symbol number */
    c2 = -1;
    v = 1000000000L;
    for (i = 0; i <= 256; i++) {
      if (freq[i] && freq[i] <= v && i != c1) {
	v = freq[i];
	c2 = i;
      }
    }

    /* Done if we've merged everything into one frequency */
    if (c2 < 0)
      break;

    /* Else merge the two counts/trees */
    freq[c1] += freq[c2];
    freq[c2] = 0;

    /* Increment the codesize of everything in c1's tree branch */
    codesize[c1]++;
    while (others[c1] >= 0) {
      c1 = others[c1];
      codesize[c1]++;
    }

    others[c1] = c2;		/* chain c2 onto c1's tree branch */

    /* Increment the codesize of everything in c2's tree branch */
    codesize[c2]++;
    while (others[c2] >= 0) {
      c2 = others[c2];
      codesize[c2]++;
    }
  }

  /* Now count the number of symbols of each code length */
  for (i = 0; i <= 256; i++) {
    if (codesize[i]) {
      /* The JPEG standard seems to think that this can't happen, */
      /* but I'm paranoid... */
      if (codesize[i] > MAX_CLEN)
	ERREXIT(cinfo, JERR_HUFF_CLEN_OUTOFBOUNDS);

      bits[codesize[i]]++;
    }
  }

  /* JPEG doesn't allow symbols with code lengths over 16 bits, so if the pure
   * Huffman procedure assigned any such lengths, we must adjust the coding.
   * Here is what the JPEG spec says about how this next bit works:
   * Since symbols are paired for the longest Huffman code, the symbols are
   * removed from this length category two at a time.  The prefix for the pair
   * (which is one bit shorter) is allocated to one of the pair; then,
   * skipping the BITS entry for that prefix length, a code word from the next
   * shortest nonzero BITS entry is converted into a prefix for two code words
   * one bit longer.
   */

  for (i = MAX_CLEN; i > 16; i--) {
    while (bits[i] > 0) {
      j = i - 2;		/* find length of new prefix to be used */
      while (bits[j] == 0) {
	if (j == 0)
	  ERREXIT(cinfo, JERR_HUFF_CLEN_OUTOFBOUNDS);
	j--;
      }

      bits[i] -= 2;		/* remove two symbols */
      bits[i-1]++;		/* one goes in this length */
      bits[j+1] += 2;		/* two new symbols in this length */
      bits[j]--;		/* symbol of this length is now a prefix */
    }
  }

  /* Remove the count for the pseudo-symbol 256 from the largest codelength */
  while (bits[i] == 0)		/* find largest codelength still in use */
    i--;
  bits[i]--;

  /* Return final symbol counts (only for lengths 0..16) */
  MEMCOPY(htbl->bits, bits, SIZEOF(htbl->bits));

#ifdef DONT_USE_FANCY_HUFF_OPT

  /* Return a list of the symbols sorted by code length */
  /* Note: Due to the codelength changes made above, it can happen
   * that more frequent symbols are assigned longer codewords.
   */
  p = htbl->huffval;
  for (i = 1; i <= MAX_CLEN; i++) {
    for (j = 0; j <= 255; j++) {
      if (codesize[j] == i) {
	*p++ = (UINT8) j;
      }
    }
  }

#endif /* DONT_USE_FANCY_HUFF_OPT */

  /* Set sent_table FALSE so updated table will be written to JPEG file. */
  htbl->sent_table = FALSE;
}


/*
 * Finish up a statistics-gathering pass and create the new Huffman tables.
 */

METHODDEF(void)
finish_pass_gather (j_compress_ptr cinfo)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  int ci, tbl;
  jpeg_component_info * compptr;
  JHUFF_TBL **htblptr;
  boolean did_dc[NUM_HUFF_TBLS];
  boolean did_ac[NUM_HUFF_TBLS];

  if (cinfo->progressive_mode)
    /* Flush out buffered data (all we care about is counting the EOB symbol) */
    emit_eobrun(entropy);

  /* It's important not to apply jpeg_gen_optimal_table more than once
   * per table, because it clobbers the input frequency counts!
   */
  MEMZERO(did_dc, SIZEOF(did_dc));
  MEMZERO(did_ac, SIZEOF(did_ac));

  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    /* DC needs no table for refinement scan */
    if (cinfo->Ss == 0 && cinfo->Ah == 0) {
      tbl = compptr->dc_tbl_no;
      if (! did_dc[tbl]) {
	htblptr = & cinfo->dc_huff_tbl_ptrs[tbl];
	if (*htblptr == NULL)
	  *htblptr = jpeg_alloc_huff_table((j_common_ptr) cinfo);
	jpeg_gen_optimal_table(cinfo, *htblptr, entropy->dc_count_ptrs[tbl]);
	did_dc[tbl] = TRUE;
      }
    }
    /* AC needs no table when not present */
    if (cinfo->Se) {
      tbl = compptr->ac_tbl_no;
      if (! did_ac[tbl]) {
	htblptr = & cinfo->ac_huff_tbl_ptrs[tbl];
	if (*htblptr == NULL)
	  *htblptr = jpeg_alloc_huff_table((j_common_ptr) cinfo);
	jpeg_gen_optimal_table(cinfo, *htblptr, entropy->ac_count_ptrs[tbl]);
	did_ac[tbl] = TRUE;
      }
    }
  }
}


/*
 * Initialize for a Huffman-compressed scan.
 * If gather_statistics is TRUE, we do not output anything during the scan,
 * just count the Huffman symbols used and generate Huffman code tables.
 */

METHODDEF(void)
start_pass_huff (j_compress_ptr cinfo, boolean gather_statistics)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr) cinfo->entropy;
  int ci, tbl;
  jpeg_component_info * compptr;

  if (gather_statistics)
    entropy->pub.finish_pass = finish_pass_gather;
  else
    entropy->pub.finish_pass = finish_pass_huff;

  if (cinfo->progressive_mode) {
    entropy->cinfo = cinfo;
    entropy->gather_statistics = gather_statistics;

    /* We assume jcmaster.c already validated the scan parameters. */

    /* Select execution routine */
    if (cinfo->Ah == 0) {
      if (cinfo->Ss == 0)
	entropy->pub.encode_mcu = encode_mcu_DC_first;
      else
	entropy->pub.encode_mcu = encode_mcu_AC_first;
    } else {
      if (cinfo->Ss == 0)
	entropy->pub.encode_mcu = encode_mcu_DC_refine;
      else {
	entropy->pub.encode_mcu = encode_mcu_AC_refine;
	/* AC refinement needs a correction bit buffer */
	if (entropy->bit_buffer == NULL)
	  entropy->bit_buffer = (char *) (*cinfo->mem->alloc_small)
	    ((j_common_ptr) cinfo, JPOOL_IMAGE, MAX_CORR_BITS * SIZEOF(char));
      }
    }

    /* Initialize AC stuff */
    entropy->ac_tbl_no = cinfo->cur_comp_info[0]->ac_tbl_no;
    entropy->EOBRUN = 0;
    entropy->BE = 0;
  } else {
    if (gather_statistics)
      entropy->pub.encode_mcu = encode_mcu_gather;
    else
      entropy->pub.encode_mcu = encode_mcu_huff;
  }

  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    /* DC needs no table for refinement scan */
    if (cinfo->Ss == 0 && cinfo->Ah == 0) {
      tbl = compptr->dc_tbl_no;
      if (gather_statistics) {
	/* Check for invalid table index */
	/* (make_c_derived_tbl does this in the other path) */
	if (tbl < 0 || tbl >= NUM_HUFF_TBLS)
	  ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, tbl);
	/* Allocate and zero the statistics tables */
	/* Note that jpeg_gen_optimal_table expects 257 entries in each table! */
	if (entropy->dc_count_ptrs[tbl] == NULL)
	  entropy->dc_count_ptrs[tbl] = (long *) (*cinfo->mem->alloc_small)
	    ((j_common_ptr) cinfo, JPOOL_IMAGE, 257 * SIZEOF(long));
	MEMZERO(entropy->dc_count_ptrs[tbl], 257 * SIZEOF(long));
      } else {
	/* Compute derived values for Huffman tables */
	/* We may do this more than once for a table, but it's not expensive */
	jpeg_make_c_derived_tbl(cinfo, TRUE, tbl,
				& entropy->dc_derived_tbls[tbl]);
      }
      /* Initialize DC predictions to 0 */
      entropy->saved.last_dc_val[ci] = 0;
    }
    /* AC needs no table when not present */
    if (cinfo->Se) {
      tbl = compptr->ac_tbl_no;
      if (gather_statistics) {
	if (tbl < 0 || tbl >= NUM_HUFF_TBLS)
	  ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, tbl);
	if (entropy->ac_count_ptrs[tbl] == NULL)
	  entropy->ac_count_ptrs[tbl] = (long *) (*cinfo->mem->alloc_small)
	    ((j_common_ptr) cinfo, JPOOL_IMAGE, 257 * SIZEOF(long));
	MEMZERO(entropy->ac_count_ptrs[tbl], 257 * SIZEOF(long));
      } else {
	jpeg_make_c_derived_tbl(cinfo, FALSE, tbl,
				& entropy->ac_derived_tbls[tbl]);
      }
    }
  }

  /* Initialize bit buffer to empty */
  entropy->saved.put_buffer = 0;
  entropy->saved.put_bits = 0;

  /* Initialize restart stuff */
  entropy->restarts_to_go = cinfo->restart_interval;
  entropy->next_restart_num = 0;
}


/*
 * Module initialization routine for Huffman entropy encoding.
 */

GLOBAL(void)
jinit_huff_encoder (j_compress_ptr cinfo)
{
  huff_entropy_ptr entropy;
  int i;

  entropy = (huff_entropy_ptr) (*cinfo->mem->alloc_small)
    ((j_common_ptr) cinfo, JPOOL_IMAGE, SIZEOF(huff_entropy_encoder));
  cinfo->entropy = &entropy->pub;
  entropy->pub.start_pass = start_pass_huff;

  /* Mark tables unallocated */
  for (i = 0; i < NUM_HUFF_TBLS; i++) {
    entropy->dc_derived_tbls[i] = entropy->ac_derived_tbls[i] = NULL;
    entropy->dc_count_ptrs[i] = entropy->ac_count_ptrs[i] = NULL;
  }

  if (cinfo->progressive_mode)
    entropy->bit_buffer = NULL;	/* needed only in AC refinement scan */
}
