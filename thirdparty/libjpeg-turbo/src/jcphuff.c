/*
 * jcphuff.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1995-1997, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2011, 2015, 2018, 2021-2022, 2024, D. R. Commander.
 * Copyright (C) 2016, 2018, 2022, Matthieu Darbois.
 * Copyright (C) 2020, Arm Limited.
 * Copyright (C) 2021, Alex Richardson.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains Huffman entropy encoding routines for progressive JPEG.
 *
 * We do not support output suspension in this module, since the library
 * currently does not allow multiple-scan files to be written with output
 * suspension.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"
#ifdef WITH_SIMD
#include "jsimd.h"
#else
#include "jchuff.h"             /* Declarations shared with jc*huff.c */
#endif
#include <limits.h>

#ifdef HAVE_INTRIN_H
#include <intrin.h>
#ifdef _MSC_VER
#ifdef HAVE_BITSCANFORWARD64
#pragma intrinsic(_BitScanForward64)
#endif
#ifdef HAVE_BITSCANFORWARD
#pragma intrinsic(_BitScanForward)
#endif
#endif
#endif

#ifdef C_PROGRESSIVE_SUPPORTED

#include "jpeg_nbits.h"


/* Expanded entropy encoder object for progressive Huffman encoding. */

typedef struct {
  struct jpeg_entropy_encoder pub; /* public fields */

  /* Pointer to routine to prepare data for encode_mcu_AC_first() */
  void (*AC_first_prepare) (const JCOEF *block,
                            const int *jpeg_natural_order_start, int Sl,
                            int Al, UJCOEF *values, size_t *zerobits);
  /* Pointer to routine to prepare data for encode_mcu_AC_refine() */
  int (*AC_refine_prepare) (const JCOEF *block,
                            const int *jpeg_natural_order_start, int Sl,
                            int Al, UJCOEF *absvalues, size_t *bits);

  /* Mode flag: TRUE for optimization, FALSE for actual data output */
  boolean gather_statistics;

  /* Bit-level coding status.
   * next_output_byte/free_in_buffer are local copies of cinfo->dest fields.
   */
  JOCTET *next_output_byte;     /* => next byte to write in buffer */
  size_t free_in_buffer;        /* # of byte spaces remaining in buffer */
  size_t put_buffer;            /* current bit-accumulation buffer */
  int put_bits;                 /* # of bits now in it */
  j_compress_ptr cinfo;         /* link to cinfo (needed for dump_buffer) */

  /* Coding status for DC components */
  int last_dc_val[MAX_COMPS_IN_SCAN]; /* last DC coef for each component */

  /* Coding status for AC components */
  int ac_tbl_no;                /* the table number of the single component */
  unsigned int EOBRUN;          /* run length of EOBs */
  unsigned int BE;              /* # of buffered correction bits before MCU */
  char *bit_buffer;             /* buffer for correction bits (1 per char) */
  /* packing correction bits tightly would save some space but cost time... */

  unsigned int restarts_to_go;  /* MCUs left in this restart interval */
  int next_restart_num;         /* next restart number to write (0-7) */

  /* Pointers to derived tables (these workspaces have image lifespan).
   * Since any one scan codes only DC or only AC, we only need one set
   * of tables, not one for DC and one for AC.
   */
  c_derived_tbl *derived_tbls[NUM_HUFF_TBLS];

  /* Statistics tables for optimization; again, one set is enough */
  long *count_ptrs[NUM_HUFF_TBLS];
} phuff_entropy_encoder;

typedef phuff_entropy_encoder *phuff_entropy_ptr;

/* MAX_CORR_BITS is the number of bits the AC refinement correction-bit
 * buffer can hold.  Larger sizes may slightly improve compression, but
 * 1000 is already well into the realm of overkill.
 * The minimum safe size is 64 bits.
 */

#define MAX_CORR_BITS  1000     /* Max # of correction bits I can buffer */

/* IRIGHT_SHIFT is like RIGHT_SHIFT, but works on int rather than JLONG.
 * We assume that int right shift is unsigned if JLONG right shift is,
 * which should be safe.
 */

#ifdef RIGHT_SHIFT_IS_UNSIGNED
#define ISHIFT_TEMPS    int ishift_temp;
#define IRIGHT_SHIFT(x, shft) \
  ((ishift_temp = (x)) < 0 ? \
   (ishift_temp >> (shft)) | ((~0) << (16 - (shft))) : \
   (ishift_temp >> (shft)))
#else
#define ISHIFT_TEMPS
#define IRIGHT_SHIFT(x, shft)   ((x) >> (shft))
#endif

#define PAD(v, p)  ((v + (p) - 1) & (~((p) - 1)))

/* Forward declarations */
METHODDEF(boolean) encode_mcu_DC_first(j_compress_ptr cinfo,
                                       JBLOCKROW *MCU_data);
METHODDEF(void) encode_mcu_AC_first_prepare
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *values, size_t *zerobits);
METHODDEF(boolean) encode_mcu_AC_first(j_compress_ptr cinfo,
                                       JBLOCKROW *MCU_data);
METHODDEF(boolean) encode_mcu_DC_refine(j_compress_ptr cinfo,
                                        JBLOCKROW *MCU_data);
METHODDEF(int) encode_mcu_AC_refine_prepare
  (const JCOEF *block, const int *jpeg_natural_order_start, int Sl, int Al,
   UJCOEF *absvalues, size_t *bits);
METHODDEF(boolean) encode_mcu_AC_refine(j_compress_ptr cinfo,
                                        JBLOCKROW *MCU_data);
METHODDEF(void) finish_pass_phuff(j_compress_ptr cinfo);
METHODDEF(void) finish_pass_gather_phuff(j_compress_ptr cinfo);


/* Count bit loop zeroes */
INLINE
METHODDEF(int)
count_zeroes(size_t *x)
{
#if defined(HAVE_BUILTIN_CTZL)
  int result;
  result = __builtin_ctzl(*x);
  *x >>= result;
#elif defined(HAVE_BITSCANFORWARD64)
  unsigned long result;
  _BitScanForward64(&result, *x);
  *x >>= result;
#elif defined(HAVE_BITSCANFORWARD)
  unsigned long result;
  _BitScanForward(&result, *x);
  *x >>= result;
#else
  int result = 0;
  while ((*x & 1) == 0) {
    ++result;
    *x >>= 1;
  }
#endif
  return (int)result;
}


/*
 * Initialize for a Huffman-compressed scan using progressive JPEG.
 */

METHODDEF(void)
start_pass_phuff(j_compress_ptr cinfo, boolean gather_statistics)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  boolean is_DC_band;
  int ci, tbl;
  jpeg_component_info *compptr;

  entropy->cinfo = cinfo;
  entropy->gather_statistics = gather_statistics;

  is_DC_band = (cinfo->Ss == 0);

  /* We assume jcmaster.c already validated the scan parameters. */

  /* Select execution routines */
  if (cinfo->Ah == 0) {
    if (is_DC_band)
      entropy->pub.encode_mcu = encode_mcu_DC_first;
    else
      entropy->pub.encode_mcu = encode_mcu_AC_first;
#ifdef WITH_SIMD
    if (jsimd_can_encode_mcu_AC_first_prepare())
      entropy->AC_first_prepare = jsimd_encode_mcu_AC_first_prepare;
    else
#endif
      entropy->AC_first_prepare = encode_mcu_AC_first_prepare;
  } else {
    if (is_DC_band)
      entropy->pub.encode_mcu = encode_mcu_DC_refine;
    else {
      entropy->pub.encode_mcu = encode_mcu_AC_refine;
#ifdef WITH_SIMD
      if (jsimd_can_encode_mcu_AC_refine_prepare())
        entropy->AC_refine_prepare = jsimd_encode_mcu_AC_refine_prepare;
      else
#endif
        entropy->AC_refine_prepare = encode_mcu_AC_refine_prepare;
      /* AC refinement needs a correction bit buffer */
      if (entropy->bit_buffer == NULL)
        entropy->bit_buffer = (char *)
          (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                      MAX_CORR_BITS * sizeof(char));
    }
  }
  if (gather_statistics)
    entropy->pub.finish_pass = finish_pass_gather_phuff;
  else
    entropy->pub.finish_pass = finish_pass_phuff;

  /* Only DC coefficients may be interleaved, so cinfo->comps_in_scan = 1
   * for AC coefficients.
   */
  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    /* Initialize DC predictions to 0 */
    entropy->last_dc_val[ci] = 0;
    /* Get table index */
    if (is_DC_band) {
      if (cinfo->Ah != 0)       /* DC refinement needs no table */
        continue;
      tbl = compptr->dc_tbl_no;
    } else {
      entropy->ac_tbl_no = tbl = compptr->ac_tbl_no;
    }
    if (gather_statistics) {
      /* Check for invalid table index */
      /* (make_c_derived_tbl does this in the other path) */
      if (tbl < 0 || tbl >= NUM_HUFF_TBLS)
        ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, tbl);
      /* Allocate and zero the statistics tables */
      /* Note that jpeg_gen_optimal_table expects 257 entries in each table! */
      if (entropy->count_ptrs[tbl] == NULL)
        entropy->count_ptrs[tbl] = (long *)
          (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                      257 * sizeof(long));
      memset(entropy->count_ptrs[tbl], 0, 257 * sizeof(long));
    } else {
      /* Compute derived values for Huffman table */
      /* We may do this more than once for a table, but it's not expensive */
      jpeg_make_c_derived_tbl(cinfo, is_DC_band, tbl,
                              &entropy->derived_tbls[tbl]);
    }
  }

  /* Initialize AC stuff */
  entropy->EOBRUN = 0;
  entropy->BE = 0;

  /* Initialize bit buffer to empty */
  entropy->put_buffer = 0;
  entropy->put_bits = 0;

  /* Initialize restart stuff */
  entropy->restarts_to_go = cinfo->restart_interval;
  entropy->next_restart_num = 0;
}


/* Outputting bytes to the file.
 * NB: these must be called only when actually outputting,
 * that is, entropy->gather_statistics == FALSE.
 */

/* Emit a byte */
#define emit_byte(entropy, val) { \
  *(entropy)->next_output_byte++ = (JOCTET)(val); \
  if (--(entropy)->free_in_buffer == 0) \
    dump_buffer(entropy); \
}


LOCAL(void)
dump_buffer(phuff_entropy_ptr entropy)
/* Empty the output buffer; we do not support suspension in this module. */
{
  struct jpeg_destination_mgr *dest = entropy->cinfo->dest;

  if (!(*dest->empty_output_buffer) (entropy->cinfo))
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

LOCAL(void)
emit_bits(phuff_entropy_ptr entropy, unsigned int code, int size)
/* Emit some bits, unless we are in gather mode */
{
  /* This routine is heavily used, so it's worth coding tightly. */
  register size_t put_buffer = (size_t)code;
  register int put_bits = entropy->put_bits;

  /* if size is 0, caller used an invalid Huffman table entry */
  if (size == 0)
    ERREXIT(entropy->cinfo, JERR_HUFF_MISSING_CODE);

  if (entropy->gather_statistics)
    return;                     /* do nothing if we're only getting stats */

  put_buffer &= (((size_t)1) << size) - 1; /* mask off any extra bits in code */

  put_bits += size;             /* new number of bits in buffer */

  put_buffer <<= 24 - put_bits; /* align incoming bits */

  put_buffer |= entropy->put_buffer; /* and merge with old buffer contents */

  while (put_bits >= 8) {
    int c = (int)((put_buffer >> 16) & 0xFF);

    emit_byte(entropy, c);
    if (c == 0xFF) {            /* need to stuff a zero byte? */
      emit_byte(entropy, 0);
    }
    put_buffer <<= 8;
    put_bits -= 8;
  }

  entropy->put_buffer = put_buffer; /* update variables */
  entropy->put_bits = put_bits;
}


LOCAL(void)
flush_bits(phuff_entropy_ptr entropy)
{
  emit_bits(entropy, 0x7F, 7); /* fill any partial byte with ones */
  entropy->put_buffer = 0;     /* and reset bit-buffer to empty */
  entropy->put_bits = 0;
}


/*
 * Emit (or just count) a Huffman symbol.
 */

LOCAL(void)
emit_symbol(phuff_entropy_ptr entropy, int tbl_no, int symbol)
{
  if (entropy->gather_statistics)
    entropy->count_ptrs[tbl_no][symbol]++;
  else {
    c_derived_tbl *tbl = entropy->derived_tbls[tbl_no];
    emit_bits(entropy, tbl->ehufco[symbol], tbl->ehufsi[symbol]);
  }
}


/*
 * Emit bits from a correction bit buffer.
 */

LOCAL(void)
emit_buffered_bits(phuff_entropy_ptr entropy, char *bufstart,
                   unsigned int nbits)
{
  if (entropy->gather_statistics)
    return;                     /* no real work */

  while (nbits > 0) {
    emit_bits(entropy, (unsigned int)(*bufstart), 1);
    bufstart++;
    nbits--;
  }
}


/*
 * Emit any pending EOBRUN symbol.
 */

LOCAL(void)
emit_eobrun(phuff_entropy_ptr entropy)
{
  register int temp, nbits;

  if (entropy->EOBRUN > 0) {    /* if there is any pending EOBRUN */
    temp = entropy->EOBRUN;
    nbits = JPEG_NBITS_NONZERO(temp) - 1;
    /* safety check: shouldn't happen given limited correction-bit buffer */
    if (nbits > 14)
      ERREXIT(entropy->cinfo, JERR_HUFF_MISSING_CODE);

    emit_symbol(entropy, entropy->ac_tbl_no, nbits << 4);
    if (nbits)
      emit_bits(entropy, entropy->EOBRUN, nbits);

    entropy->EOBRUN = 0;

    /* Emit any buffered correction bits */
    emit_buffered_bits(entropy, entropy->bit_buffer, entropy->BE);
    entropy->BE = 0;
  }
}


/*
 * Emit a restart marker & resynchronize predictions.
 */

LOCAL(void)
emit_restart(phuff_entropy_ptr entropy, int restart_num)
{
  int ci;

  emit_eobrun(entropy);

  if (!entropy->gather_statistics) {
    flush_bits(entropy);
    emit_byte(entropy, 0xFF);
    emit_byte(entropy, JPEG_RST0 + restart_num);
  }

  if (entropy->cinfo->Ss == 0) {
    /* Re-initialize DC predictions to 0 */
    for (ci = 0; ci < entropy->cinfo->comps_in_scan; ci++)
      entropy->last_dc_val[ci] = 0;
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
encode_mcu_DC_first(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  register int temp, temp2, temp3;
  register int nbits;
  int blkn, ci;
  int Al = cinfo->Al;
  JBLOCKROW block;
  jpeg_component_info *compptr;
  ISHIFT_TEMPS
  int max_coef_bits = cinfo->data_precision + 2;

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart(entropy, entropy->next_restart_num);

  /* Encode the MCU data blocks */
  for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
    block = MCU_data[blkn];
    ci = cinfo->MCU_membership[blkn];
    compptr = cinfo->cur_comp_info[ci];

    /* Compute the DC value after the required point transform by Al.
     * This is simply an arithmetic right shift.
     */
    temp2 = IRIGHT_SHIFT((int)((*block)[0]), Al);

    /* DC differences are figured on the point-transformed values. */
    temp = temp2 - entropy->last_dc_val[ci];
    entropy->last_dc_val[ci] = temp2;

    /* Encode the DC coefficient difference per section G.1.2.1 */

    /* This is a well-known technique for obtaining the absolute value without
     * a branch.  It is derived from an assembly language technique presented
     * in "How to Optimize for the Pentium Processors", Copyright (c) 1996,
     * 1997 by Agner Fog.
     */
    temp3 = temp >> (CHAR_BIT * sizeof(int) - 1);
    temp ^= temp3;
    temp -= temp3;              /* temp is abs value of input */
    /* For a negative input, want temp2 = bitwise complement of abs(input) */
    temp2 = temp ^ temp3;

    /* Find the number of bits needed for the magnitude of the coefficient */
    nbits = JPEG_NBITS(temp);
    /* Check for out-of-range coefficient values.
     * Since we're encoding a difference, the range limit is twice as much.
     */
    if (nbits > max_coef_bits + 1)
      ERREXIT(cinfo, JERR_BAD_DCT_COEF);

    /* Count/emit the Huffman-coded symbol for the number of bits */
    emit_symbol(entropy, compptr->dc_tbl_no, nbits);

    /* Emit that number of bits of the value, if positive, */
    /* or the complement of its magnitude, if negative. */
    if (nbits)                  /* emit_bits rejects calls with size 0 */
      emit_bits(entropy, (unsigned int)temp2, nbits);
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
 * Data preparation for encode_mcu_AC_first().
 */

#define COMPUTE_ABSVALUES_AC_FIRST(Sl) { \
  for (k = 0; k < Sl; k++) { \
    temp = block[jpeg_natural_order_start[k]]; \
    if (temp == 0) \
      continue; \
    /* We must apply the point transform by Al.  For AC coefficients this \
     * is an integer division with rounding towards 0.  To do this portably \
     * in C, we shift after obtaining the absolute value; so the code is \
     * interwoven with finding the abs value (temp) and output bits (temp2). \
     */ \
    temp2 = temp >> (CHAR_BIT * sizeof(int) - 1); \
    temp ^= temp2; \
    temp -= temp2;              /* temp is abs value of input */ \
    temp >>= Al;                /* apply the point transform */ \
    /* Watch out for case that nonzero coef is zero after point transform */ \
    if (temp == 0) \
      continue; \
    /* For a negative coef, want temp2 = bitwise complement of abs(coef) */ \
    temp2 ^= temp; \
    values[k] = (UJCOEF)temp; \
    values[k + DCTSIZE2] = (UJCOEF)temp2; \
    zerobits |= ((size_t)1U) << k; \
  } \
}

METHODDEF(void)
encode_mcu_AC_first_prepare(const JCOEF *block,
                            const int *jpeg_natural_order_start, int Sl,
                            int Al, UJCOEF *values, size_t *bits)
{
  register int k, temp, temp2;
  size_t zerobits = 0U;
  int Sl0 = Sl;

#if SIZEOF_SIZE_T == 4
  if (Sl0 > 32)
    Sl0 = 32;
#endif

  COMPUTE_ABSVALUES_AC_FIRST(Sl0);

  bits[0] = zerobits;
#if SIZEOF_SIZE_T == 4
  zerobits = 0U;

  if (Sl > 32) {
    Sl -= 32;
    jpeg_natural_order_start += 32;
    values += 32;

    COMPUTE_ABSVALUES_AC_FIRST(Sl);
  }
  bits[1] = zerobits;
#endif
}

/*
 * MCU encoding for AC initial scan (either spectral selection,
 * or first pass of successive approximation).
 */

#define ENCODE_COEFS_AC_FIRST(label) { \
  while (zerobits) { \
    r = count_zeroes(&zerobits); \
    cvalue += r; \
label \
    temp  = cvalue[0]; \
    temp2 = cvalue[DCTSIZE2]; \
    \
    /* if run length > 15, must emit special run-length-16 codes (0xF0) */ \
    while (r > 15) { \
      emit_symbol(entropy, entropy->ac_tbl_no, 0xF0); \
      r -= 16; \
    } \
    \
    /* Find the number of bits needed for the magnitude of the coefficient */ \
    nbits = JPEG_NBITS_NONZERO(temp);  /* there must be at least one 1 bit */ \
    /* Check for out-of-range coefficient values */ \
    if (nbits > max_coef_bits) \
      ERREXIT(cinfo, JERR_BAD_DCT_COEF); \
    \
    /* Count/emit Huffman symbol for run length / number of bits */ \
    emit_symbol(entropy, entropy->ac_tbl_no, (r << 4) + nbits); \
    \
    /* Emit that number of bits of the value, if positive, */ \
    /* or the complement of its magnitude, if negative. */ \
    emit_bits(entropy, (unsigned int)temp2, nbits); \
    \
    cvalue++; \
    zerobits >>= 1; \
  } \
}

METHODDEF(boolean)
encode_mcu_AC_first(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  register int temp, temp2;
  register int nbits, r;
  int Sl = cinfo->Se - cinfo->Ss + 1;
  int Al = cinfo->Al;
  UJCOEF values_unaligned[2 * DCTSIZE2 + 15];
  UJCOEF *values;
  const UJCOEF *cvalue;
  size_t zerobits;
  size_t bits[8 / SIZEOF_SIZE_T];
  int max_coef_bits = cinfo->data_precision + 2;

#ifdef ZERO_BUFFERS
  memset(values_unaligned, 0, sizeof(values_unaligned));
  memset(bits, 0, sizeof(bits));
#endif

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart(entropy, entropy->next_restart_num);

#ifdef WITH_SIMD
  cvalue = values = (UJCOEF *)PAD((JUINTPTR)values_unaligned, 16);
#else
  /* Not using SIMD, so alignment is not needed */
  cvalue = values = values_unaligned;
#endif

  /* Prepare data */
  entropy->AC_first_prepare(MCU_data[0][0], jpeg_natural_order + cinfo->Ss,
                            Sl, Al, values, bits);

  zerobits = bits[0];
#if SIZEOF_SIZE_T == 4
  zerobits |= bits[1];
#endif

  /* Emit any pending EOBRUN */
  if (zerobits && (entropy->EOBRUN > 0))
    emit_eobrun(entropy);

#if SIZEOF_SIZE_T == 4
  zerobits = bits[0];
#endif

  /* Encode the AC coefficients per section G.1.2.2, fig. G.3 */

  ENCODE_COEFS_AC_FIRST((void)0;);

#if SIZEOF_SIZE_T == 4
  zerobits = bits[1];
  if (zerobits) {
    int diff = ((values + DCTSIZE2 / 2) - cvalue);
    r = count_zeroes(&zerobits);
    r += diff;
    cvalue += r;
    goto first_iter_ac_first;
  }

  ENCODE_COEFS_AC_FIRST(first_iter_ac_first:);
#endif

  if (cvalue < (values + Sl)) { /* If there are trailing zeroes, */
    entropy->EOBRUN++;          /* count an EOB */
    if (entropy->EOBRUN == 0x7FFF)
      emit_eobrun(entropy);     /* force it out to avoid overflow */
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
 * Note: we assume such scans can be multi-component, although the spec
 * is not very clear on the point.
 */

METHODDEF(boolean)
encode_mcu_DC_refine(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  register int temp;
  int blkn;
  int Al = cinfo->Al;
  JBLOCKROW block;

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart(entropy, entropy->next_restart_num);

  /* Encode the MCU data blocks */
  for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
    block = MCU_data[blkn];

    /* We simply emit the Al'th bit of the DC coefficient value. */
    temp = (*block)[0];
    emit_bits(entropy, (unsigned int)(temp >> Al), 1);
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
 * Data preparation for encode_mcu_AC_refine().
 */

#define COMPUTE_ABSVALUES_AC_REFINE(Sl, koffset) { \
  /* It is convenient to make a pre-pass to determine the transformed \
   * coefficients' absolute values and the EOB position. \
   */ \
  for (k = 0; k < Sl; k++) { \
    temp = block[jpeg_natural_order_start[k]]; \
    /* We must apply the point transform by Al.  For AC coefficients this \
     * is an integer division with rounding towards 0.  To do this portably \
     * in C, we shift after obtaining the absolute value. \
     */ \
    temp2 = temp >> (CHAR_BIT * sizeof(int) - 1); \
    temp ^= temp2; \
    temp -= temp2;              /* temp is abs value of input */ \
    temp >>= Al;                /* apply the point transform */ \
    if (temp != 0) { \
      zerobits |= ((size_t)1U) << k; \
      signbits |= ((size_t)(temp2 + 1)) << k; \
    } \
    absvalues[k] = (UJCOEF)temp; /* save abs value for main pass */ \
    if (temp == 1) \
      EOB = k + koffset;        /* EOB = index of last newly-nonzero coef */ \
  } \
}

METHODDEF(int)
encode_mcu_AC_refine_prepare(const JCOEF *block,
                             const int *jpeg_natural_order_start, int Sl,
                             int Al, UJCOEF *absvalues, size_t *bits)
{
  register int k, temp, temp2;
  int EOB = 0;
  size_t zerobits = 0U, signbits = 0U;
  int Sl0 = Sl;

#if SIZEOF_SIZE_T == 4
  if (Sl0 > 32)
    Sl0 = 32;
#endif

  COMPUTE_ABSVALUES_AC_REFINE(Sl0, 0);

  bits[0] = zerobits;
#if SIZEOF_SIZE_T == 8
  bits[1] = signbits;
#else
  bits[2] = signbits;

  zerobits = 0U;
  signbits = 0U;

  if (Sl > 32) {
    Sl -= 32;
    jpeg_natural_order_start += 32;
    absvalues += 32;

    COMPUTE_ABSVALUES_AC_REFINE(Sl, 32);
  }

  bits[1] = zerobits;
  bits[3] = signbits;
#endif

  return EOB;
}


/*
 * MCU encoding for AC successive approximation refinement scan.
 */

#define ENCODE_COEFS_AC_REFINE(label) { \
  while (zerobits) { \
    idx = count_zeroes(&zerobits); \
    r += idx; \
    cabsvalue += idx; \
    signbits >>= idx; \
label \
    /* Emit any required ZRLs, but not if they can be folded into EOB */ \
    while (r > 15 && (cabsvalue <= EOBPTR)) { \
      /* emit any pending EOBRUN and the BE correction bits */ \
      emit_eobrun(entropy); \
      /* Emit ZRL */ \
      emit_symbol(entropy, entropy->ac_tbl_no, 0xF0); \
      r -= 16; \
      /* Emit buffered correction bits that must be associated with ZRL */ \
      emit_buffered_bits(entropy, BR_buffer, BR); \
      BR_buffer = entropy->bit_buffer; /* BE bits are gone now */ \
      BR = 0; \
    } \
    \
    temp = *cabsvalue++; \
    \
    /* If the coef was previously nonzero, it only needs a correction bit. \
     * NOTE: a straight translation of the spec's figure G.7 would suggest \
     * that we also need to test r > 15.  But if r > 15, we can only get here \
     * if k > EOB, which implies that this coefficient is not 1. \
     */ \
    if (temp > 1) { \
      /* The correction bit is the next bit of the absolute value. */ \
      BR_buffer[BR++] = (char)(temp & 1); \
      signbits >>= 1; \
      zerobits >>= 1; \
      continue; \
    } \
    \
    /* Emit any pending EOBRUN and the BE correction bits */ \
    emit_eobrun(entropy); \
    \
    /* Count/emit Huffman symbol for run length / number of bits */ \
    emit_symbol(entropy, entropy->ac_tbl_no, (r << 4) + 1); \
    \
    /* Emit output bit for newly-nonzero coef */ \
    temp = signbits & 1; /* ((*block)[jpeg_natural_order_start[k]] < 0) ? 0 : 1 */ \
    emit_bits(entropy, (unsigned int)temp, 1); \
    \
    /* Emit buffered correction bits that must be associated with this code */ \
    emit_buffered_bits(entropy, BR_buffer, BR); \
    BR_buffer = entropy->bit_buffer; /* BE bits are gone now */ \
    BR = 0; \
    r = 0;                      /* reset zero run length */ \
    signbits >>= 1; \
    zerobits >>= 1; \
  } \
}

METHODDEF(boolean)
encode_mcu_AC_refine(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  register int temp, r, idx;
  char *BR_buffer;
  unsigned int BR;
  int Sl = cinfo->Se - cinfo->Ss + 1;
  int Al = cinfo->Al;
  UJCOEF absvalues_unaligned[DCTSIZE2 + 15];
  UJCOEF *absvalues;
  const UJCOEF *cabsvalue, *EOBPTR;
  size_t zerobits, signbits;
  size_t bits[16 / SIZEOF_SIZE_T];

#ifdef ZERO_BUFFERS
  memset(absvalues_unaligned, 0, sizeof(absvalues_unaligned));
  memset(bits, 0, sizeof(bits));
#endif

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Emit restart marker if needed */
  if (cinfo->restart_interval)
    if (entropy->restarts_to_go == 0)
      emit_restart(entropy, entropy->next_restart_num);

#ifdef WITH_SIMD
  cabsvalue = absvalues = (UJCOEF *)PAD((JUINTPTR)absvalues_unaligned, 16);
#else
  /* Not using SIMD, so alignment is not needed */
  cabsvalue = absvalues = absvalues_unaligned;
#endif

  /* Prepare data */
  EOBPTR = absvalues +
    entropy->AC_refine_prepare(MCU_data[0][0], jpeg_natural_order + cinfo->Ss,
                               Sl, Al, absvalues, bits);

  /* Encode the AC coefficients per section G.1.2.3, fig. G.7 */

  r = 0;                        /* r = run length of zeros */
  BR = 0;                       /* BR = count of buffered bits added now */
  BR_buffer = entropy->bit_buffer + entropy->BE; /* Append bits to buffer */

  zerobits = bits[0];
#if SIZEOF_SIZE_T == 8
  signbits = bits[1];
#else
  signbits = bits[2];
#endif
  ENCODE_COEFS_AC_REFINE((void)0;);

#if SIZEOF_SIZE_T == 4
  zerobits = bits[1];
  signbits = bits[3];

  if (zerobits) {
    int diff = ((absvalues + DCTSIZE2 / 2) - cabsvalue);
    idx = count_zeroes(&zerobits);
    signbits >>= idx;
    idx += diff;
    r += idx;
    cabsvalue += idx;
    goto first_iter_ac_refine;
  }

  ENCODE_COEFS_AC_REFINE(first_iter_ac_refine:);
#endif

  r |= (int)((absvalues + Sl) - cabsvalue);

  if (r > 0 || BR > 0) {        /* If there are trailing zeroes, */
    entropy->EOBRUN++;          /* count an EOB */
    entropy->BE += BR;          /* concat my correction bits to older ones */
    /* We force out the EOB if we risk either:
     * 1. overflow of the EOB counter;
     * 2. overflow of the correction bit buffer during the next MCU.
     */
    if (entropy->EOBRUN == 0x7FFF ||
        entropy->BE > (MAX_CORR_BITS - DCTSIZE2 + 1))
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


/*
 * Finish up at the end of a Huffman-compressed progressive scan.
 */

METHODDEF(void)
finish_pass_phuff(j_compress_ptr cinfo)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;

  entropy->next_output_byte = cinfo->dest->next_output_byte;
  entropy->free_in_buffer = cinfo->dest->free_in_buffer;

  /* Flush out any buffered data */
  emit_eobrun(entropy);
  flush_bits(entropy);

  cinfo->dest->next_output_byte = entropy->next_output_byte;
  cinfo->dest->free_in_buffer = entropy->free_in_buffer;
}


/*
 * Finish up a statistics-gathering pass and create the new Huffman tables.
 */

METHODDEF(void)
finish_pass_gather_phuff(j_compress_ptr cinfo)
{
  phuff_entropy_ptr entropy = (phuff_entropy_ptr)cinfo->entropy;
  boolean is_DC_band;
  int ci, tbl;
  jpeg_component_info *compptr;
  JHUFF_TBL **htblptr;
  boolean did[NUM_HUFF_TBLS];

  /* Flush out buffered data (all we care about is counting the EOB symbol) */
  emit_eobrun(entropy);

  is_DC_band = (cinfo->Ss == 0);

  /* It's important not to apply jpeg_gen_optimal_table more than once
   * per table, because it clobbers the input frequency counts!
   */
  memset(did, 0, sizeof(did));

  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    if (is_DC_band) {
      if (cinfo->Ah != 0)       /* DC refinement needs no table */
        continue;
      tbl = compptr->dc_tbl_no;
    } else {
      tbl = compptr->ac_tbl_no;
    }
    if (!did[tbl]) {
      if (is_DC_band)
        htblptr = &cinfo->dc_huff_tbl_ptrs[tbl];
      else
        htblptr = &cinfo->ac_huff_tbl_ptrs[tbl];
      if (*htblptr == NULL)
        *htblptr = jpeg_alloc_huff_table((j_common_ptr)cinfo);
      jpeg_gen_optimal_table(cinfo, *htblptr, entropy->count_ptrs[tbl]);
      did[tbl] = TRUE;
    }
  }
}


/*
 * Module initialization routine for progressive Huffman entropy encoding.
 */

GLOBAL(void)
jinit_phuff_encoder(j_compress_ptr cinfo)
{
  phuff_entropy_ptr entropy;
  int i;

  entropy = (phuff_entropy_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(phuff_entropy_encoder));
  cinfo->entropy = (struct jpeg_entropy_encoder *)entropy;
  entropy->pub.start_pass = start_pass_phuff;

  /* Mark tables unallocated */
  for (i = 0; i < NUM_HUFF_TBLS; i++) {
    entropy->derived_tbls[i] = NULL;
    entropy->count_ptrs[i] = NULL;
  }
  entropy->bit_buffer = NULL;   /* needed only in AC refinement scan */
}

#endif /* C_PROGRESSIVE_SUPPORTED */
