/*
 * jchuff.c
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2009-2011, 2014-2016, 2018-2025, D. R. Commander.
 * Copyright (C) 2015, Matthieu Darbois.
 * Copyright (C) 2018, Matthias Räncker.
 * Copyright (C) 2020, Arm Limited.
 * Copyright (C) 2022, Felix Hanau.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains Huffman entropy encoding routines.
 *
 * Much of the complexity here has to do with supporting output suspension.
 * If the data destination module demands suspension, we want to be able to
 * back up to the start of the current MCU.  To do this, we copy state
 * variables into local working storage, and update them back to the
 * permanent JPEG objects only upon successful completion of an MCU.
 *
 * NOTE: All referenced figures are from
 * Recommendation ITU-T T.81 (1992) | ISO/IEC 10918-1:1994.
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
#include "jpeg_nbits.h"


/* Expanded entropy encoder object for Huffman encoding.
 *
 * The savable_state subrecord contains fields that change within an MCU,
 * but must not be updated permanently until we complete the MCU.
 */

#if defined(__x86_64__) && defined(__ILP32__)
typedef unsigned long long bit_buf_type;
#else
typedef size_t bit_buf_type;
#endif

/* NOTE: The more optimal Huffman encoding algorithm is only used by the
 * intrinsics implementation of the Arm Neon SIMD extensions, which is why we
 * retain the old Huffman encoder behavior when using the GAS implementation.
 */
#if defined(WITH_SIMD) && !(defined(__arm__) || defined(__aarch64__) || \
                            defined(_M_ARM) || defined(_M_ARM64) || \
                            defined(_M_ARM64EC))
typedef unsigned long long simd_bit_buf_type;
#else
typedef bit_buf_type simd_bit_buf_type;
#endif

#if (defined(SIZEOF_SIZE_T) && SIZEOF_SIZE_T == 8) || defined(_WIN64) || \
    (defined(__x86_64__) && defined(__ILP32__))
#define BIT_BUF_SIZE  64
#elif (defined(SIZEOF_SIZE_T) && SIZEOF_SIZE_T == 4) || defined(_WIN32)
#define BIT_BUF_SIZE  32
#else
#error Cannot determine word size
#endif
#define SIMD_BIT_BUF_SIZE  (sizeof(simd_bit_buf_type) * 8)

typedef struct {
  union {
    bit_buf_type c;
#ifdef WITH_SIMD
    simd_bit_buf_type simd;
#endif
  } put_buffer;                         /* current bit accumulation buffer */
  int free_bits;                        /* # of bits available in it */
                                        /* (Neon GAS: # of bits now in it) */
  int last_dc_val[MAX_COMPS_IN_SCAN];   /* last DC coef for each component */
} savable_state;

typedef struct {
  struct jpeg_entropy_encoder pub; /* public fields */

  savable_state saved;          /* Bit buffer & DC state at start of MCU */

  /* These fields are NOT loaded into local working state. */
  unsigned int restarts_to_go;  /* MCUs left in this restart interval */
  int next_restart_num;         /* next restart number to write (0-7) */

  /* Pointers to derived tables (these workspaces have image lifespan) */
  c_derived_tbl *dc_derived_tbls[NUM_HUFF_TBLS];
  c_derived_tbl *ac_derived_tbls[NUM_HUFF_TBLS];

#ifdef ENTROPY_OPT_SUPPORTED    /* Statistics tables for optimization */
  long *dc_count_ptrs[NUM_HUFF_TBLS];
  long *ac_count_ptrs[NUM_HUFF_TBLS];
#endif

#ifdef WITH_SIMD
  int simd;
#endif
} huff_entropy_encoder;

typedef huff_entropy_encoder *huff_entropy_ptr;

/* Working state while writing an MCU.
 * This struct contains all the fields that are needed by subroutines.
 */

typedef struct {
  JOCTET *next_output_byte;     /* => next byte to write in buffer */
  size_t free_in_buffer;        /* # of byte spaces remaining in buffer */
  savable_state cur;            /* Current bit buffer & DC state */
  j_compress_ptr cinfo;         /* dump_buffer needs access to this */
#ifdef WITH_SIMD
  int simd;
#endif
} working_state;


/* Forward declarations */
METHODDEF(boolean) encode_mcu_huff(j_compress_ptr cinfo, JBLOCKROW *MCU_data);
METHODDEF(void) finish_pass_huff(j_compress_ptr cinfo);
#ifdef ENTROPY_OPT_SUPPORTED
METHODDEF(boolean) encode_mcu_gather(j_compress_ptr cinfo,
                                     JBLOCKROW *MCU_data);
METHODDEF(void) finish_pass_gather(j_compress_ptr cinfo);
#endif


/*
 * Initialize for a Huffman-compressed scan.
 * If gather_statistics is TRUE, we do not output anything during the scan,
 * just count the Huffman symbols used and generate Huffman code tables.
 */

METHODDEF(void)
start_pass_huff(j_compress_ptr cinfo, boolean gather_statistics)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr)cinfo->entropy;
  int ci, dctbl, actbl;
  jpeg_component_info *compptr;

  if (gather_statistics) {
#ifdef ENTROPY_OPT_SUPPORTED
    entropy->pub.encode_mcu = encode_mcu_gather;
    entropy->pub.finish_pass = finish_pass_gather;
#else
    ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
  } else {
    entropy->pub.encode_mcu = encode_mcu_huff;
    entropy->pub.finish_pass = finish_pass_huff;
  }

#ifdef WITH_SIMD
  entropy->simd = jsimd_can_huff_encode_one_block();
#endif

  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    dctbl = compptr->dc_tbl_no;
    actbl = compptr->ac_tbl_no;
    if (gather_statistics) {
#ifdef ENTROPY_OPT_SUPPORTED
      /* Check for invalid table indexes */
      /* (make_c_derived_tbl does this in the other path) */
      if (dctbl < 0 || dctbl >= NUM_HUFF_TBLS)
        ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, dctbl);
      if (actbl < 0 || actbl >= NUM_HUFF_TBLS)
        ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, actbl);
      /* Allocate and zero the statistics tables */
      /* Note that jpeg_gen_optimal_table expects 257 entries in each table! */
      if (entropy->dc_count_ptrs[dctbl] == NULL)
        entropy->dc_count_ptrs[dctbl] = (long *)
          (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                      257 * sizeof(long));
      memset(entropy->dc_count_ptrs[dctbl], 0, 257 * sizeof(long));
      if (entropy->ac_count_ptrs[actbl] == NULL)
        entropy->ac_count_ptrs[actbl] = (long *)
          (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                      257 * sizeof(long));
      memset(entropy->ac_count_ptrs[actbl], 0, 257 * sizeof(long));
#endif
    } else {
      /* Compute derived values for Huffman tables */
      /* We may do this more than once for a table, but it's not expensive */
      jpeg_make_c_derived_tbl(cinfo, TRUE, dctbl,
                              &entropy->dc_derived_tbls[dctbl]);
      jpeg_make_c_derived_tbl(cinfo, FALSE, actbl,
                              &entropy->ac_derived_tbls[actbl]);
    }
    /* Initialize DC predictions to 0 */
    entropy->saved.last_dc_val[ci] = 0;
  }

  /* Initialize bit buffer to empty */
#ifdef WITH_SIMD
  if (entropy->simd) {
    entropy->saved.put_buffer.simd = 0;
#if defined(__aarch64__) && !defined(NEON_INTRINSICS)
    entropy->saved.free_bits = 0;
#else
    entropy->saved.free_bits = SIMD_BIT_BUF_SIZE;
#endif
  } else
#endif
  {
    entropy->saved.put_buffer.c = 0;
    entropy->saved.free_bits = BIT_BUF_SIZE;
  }

  /* Initialize restart stuff */
  entropy->restarts_to_go = cinfo->restart_interval;
  entropy->next_restart_num = 0;
}


/*
 * Compute the derived values for a Huffman table.
 * This routine also performs some validation checks on the table.
 *
 * Note this is also used by jcphuff.c and jclhuff.c.
 */

GLOBAL(void)
jpeg_make_c_derived_tbl(j_compress_ptr cinfo, boolean isDC, int tblno,
                        c_derived_tbl **pdtbl)
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
    ERREXIT1(cinfo, JERR_NO_HUFF_TABLE, tblno);

  /* Allocate a workspace if we haven't already done so. */
  if (*pdtbl == NULL)
    *pdtbl = (c_derived_tbl *)
      (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                  sizeof(c_derived_tbl));
  dtbl = *pdtbl;

  /* Figure C.1: make table of Huffman code length for each symbol */

  p = 0;
  for (l = 1; l <= 16; l++) {
    i = (int)htbl->bits[l];
    if (i < 0 || p + i > 256)   /* protect against table overrun */
      ERREXIT(cinfo, JERR_BAD_HUFF_TABLE);
    while (i--)
      huffsize[p++] = (char)l;
  }
  huffsize[p] = 0;
  lastp = p;

  /* Figure C.2: generate the codes themselves */
  /* We also validate that the counts represent a legal Huffman code tree. */

  code = 0;
  si = huffsize[0];
  p = 0;
  while (huffsize[p]) {
    while (((int)huffsize[p]) == si) {
      huffcode[p++] = code;
      code++;
    }
    /* code is now 1 more than the last code used for codelength si; but
     * it must still fit in si bits, since no code is allowed to be all ones.
     */
    if (((JLONG)code) >= (((JLONG)1) << si))
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
  memset(dtbl->ehufco, 0, sizeof(dtbl->ehufco));
  memset(dtbl->ehufsi, 0, sizeof(dtbl->ehufsi));

  /* This is also a convenient place to check for out-of-range and duplicated
   * VAL entries.  We allow 0..255 for AC symbols but only 0..15 for DC in
   * lossy mode and 0..16 for DC in lossless mode.  (We could constrain them
   * further based on data depth and mode, but this seems enough.)
   */
  maxsymbol = isDC ? (cinfo->master->lossless ? 16 : 15) : 255;

  for (p = 0; p < lastp; p++) {
    i = htbl->huffval[p];
    if (i < 0 || i > maxsymbol || dtbl->ehufsi[i])
      ERREXIT(cinfo, JERR_BAD_HUFF_TABLE);
    dtbl->ehufco[i] = huffcode[p];
    dtbl->ehufsi[i] = huffsize[p];
  }
}


/* Outputting bytes to the file */

/* Emit a byte, taking 'action' if must suspend. */
#define emit_byte(state, val, action) { \
  *(state)->next_output_byte++ = (JOCTET)(val); \
  if (--(state)->free_in_buffer == 0) \
    if (!dump_buffer(state)) \
      { action; } \
}


LOCAL(boolean)
dump_buffer(working_state *state)
/* Empty the output buffer; return TRUE if successful, FALSE if must suspend */
{
  struct jpeg_destination_mgr *dest = state->cinfo->dest;

  if (!(*dest->empty_output_buffer) (state->cinfo))
    return FALSE;
  /* After a successful buffer dump, must reset buffer pointers */
  state->next_output_byte = dest->next_output_byte;
  state->free_in_buffer = dest->free_in_buffer;
  return TRUE;
}


/* Outputting bits to the file */

/* Output byte b and, speculatively, an additional 0 byte.  0xFF must be
 * encoded as 0xFF 0x00, so the output buffer pointer is advanced by 2 if the
 * byte is 0xFF.  Otherwise, the output buffer pointer is advanced by 1, and
 * the speculative 0 byte will be overwritten by the next byte.
 */
#define EMIT_BYTE(b) { \
  buffer[0] = (JOCTET)(b); \
  buffer[1] = 0; \
  buffer -= -2 + ((JOCTET)(b) < 0xFF); \
}

/* Output the entire bit buffer.  If there are no 0xFF bytes in it, then write
 * directly to the output buffer.  Otherwise, use the EMIT_BYTE() macro to
 * encode 0xFF as 0xFF 0x00.
 */
#if BIT_BUF_SIZE == 64

#define FLUSH() { \
  if (put_buffer & 0x8080808080808080 & ~(put_buffer + 0x0101010101010101)) { \
    EMIT_BYTE(put_buffer >> 56) \
    EMIT_BYTE(put_buffer >> 48) \
    EMIT_BYTE(put_buffer >> 40) \
    EMIT_BYTE(put_buffer >> 32) \
    EMIT_BYTE(put_buffer >> 24) \
    EMIT_BYTE(put_buffer >> 16) \
    EMIT_BYTE(put_buffer >>  8) \
    EMIT_BYTE(put_buffer      ) \
  } else { \
    buffer[0] = (JOCTET)(put_buffer >> 56); \
    buffer[1] = (JOCTET)(put_buffer >> 48); \
    buffer[2] = (JOCTET)(put_buffer >> 40); \
    buffer[3] = (JOCTET)(put_buffer >> 32); \
    buffer[4] = (JOCTET)(put_buffer >> 24); \
    buffer[5] = (JOCTET)(put_buffer >> 16); \
    buffer[6] = (JOCTET)(put_buffer >> 8); \
    buffer[7] = (JOCTET)(put_buffer); \
    buffer += 8; \
  } \
}

#else

#define FLUSH() { \
  if (put_buffer & 0x80808080 & ~(put_buffer + 0x01010101)) { \
    EMIT_BYTE(put_buffer >> 24) \
    EMIT_BYTE(put_buffer >> 16) \
    EMIT_BYTE(put_buffer >>  8) \
    EMIT_BYTE(put_buffer      ) \
  } else { \
    buffer[0] = (JOCTET)(put_buffer >> 24); \
    buffer[1] = (JOCTET)(put_buffer >> 16); \
    buffer[2] = (JOCTET)(put_buffer >> 8); \
    buffer[3] = (JOCTET)(put_buffer); \
    buffer += 4; \
  } \
}

#endif

/* Fill the bit buffer to capacity with the leading bits from code, then output
 * the bit buffer and put the remaining bits from code into the bit buffer.
 */
#define PUT_AND_FLUSH(code, size) { \
  put_buffer = (put_buffer << (size + free_bits)) | (code >> -free_bits); \
  FLUSH() \
  free_bits += BIT_BUF_SIZE; \
  put_buffer = code; \
}

/* Insert code into the bit buffer and output the bit buffer if needed.
 * NOTE: We can't flush with free_bits == 0, since the left shift in
 * PUT_AND_FLUSH() would have undefined behavior.
 */
#define PUT_BITS(code, size) { \
  free_bits -= size; \
  if (free_bits < 0) \
    PUT_AND_FLUSH(code, size) \
  else \
    put_buffer = (put_buffer << size) | code; \
}

#define PUT_CODE(code, size) { \
  temp &= (((JLONG)1) << nbits) - 1; \
  temp |= code << nbits; \
  nbits += size; \
  PUT_BITS(temp, nbits) \
}


/* Although it is exceedingly rare, it is possible for a Huffman-encoded
 * coefficient block to be larger than the 128-byte unencoded block.  For each
 * of the 64 coefficients, PUT_BITS is invoked twice, and each invocation can
 * theoretically store 16 bits (for a maximum of 2048 bits or 256 bytes per
 * encoded block.)  If, for instance, one artificially sets the AC
 * coefficients to alternating values of 32767 and -32768 (using the JPEG
 * scanning order-- 1, 8, 16, etc.), then this will produce an encoded block
 * larger than 200 bytes.
 */
#define BUFSIZE  (DCTSIZE2 * 8)

#define LOAD_BUFFER() { \
  if (state->free_in_buffer < BUFSIZE) { \
    localbuf = 1; \
    buffer = _buffer; \
  } else \
    buffer = state->next_output_byte; \
}

#define STORE_BUFFER() { \
  if (localbuf) { \
    size_t bytes, bytestocopy; \
    bytes = buffer - _buffer; \
    buffer = _buffer; \
    while (bytes > 0) { \
      bytestocopy = MIN(bytes, state->free_in_buffer); \
      memcpy(state->next_output_byte, buffer, bytestocopy); \
      state->next_output_byte += bytestocopy; \
      buffer += bytestocopy; \
      state->free_in_buffer -= bytestocopy; \
      if (state->free_in_buffer == 0) \
        if (!dump_buffer(state)) return FALSE; \
      bytes -= bytestocopy; \
    } \
  } else { \
    state->free_in_buffer -= (buffer - state->next_output_byte); \
    state->next_output_byte = buffer; \
  } \
}


LOCAL(boolean)
flush_bits(working_state *state)
{
  JOCTET _buffer[BUFSIZE], *buffer, temp;
  simd_bit_buf_type put_buffer;  int put_bits;
  int localbuf = 0;

#ifdef WITH_SIMD
  if (state->simd) {
#if defined(__aarch64__) && !defined(NEON_INTRINSICS)
    put_bits = state->cur.free_bits;
#else
    put_bits = SIMD_BIT_BUF_SIZE - state->cur.free_bits;
#endif
    put_buffer = state->cur.put_buffer.simd;
  } else
#endif
  {
    put_bits = BIT_BUF_SIZE - state->cur.free_bits;
    put_buffer = state->cur.put_buffer.c;
  }

  LOAD_BUFFER()

  while (put_bits >= 8) {
    put_bits -= 8;
    temp = (JOCTET)(put_buffer >> put_bits);
    EMIT_BYTE(temp)
  }
  if (put_bits) {
    /* fill partial byte with ones */
    temp = (JOCTET)((put_buffer << (8 - put_bits)) | (0xFF >> put_bits));
    EMIT_BYTE(temp)
  }

#ifdef WITH_SIMD
  if (state->simd) {                    /* and reset bit buffer to empty */
    state->cur.put_buffer.simd = 0;
#if defined(__aarch64__) && !defined(NEON_INTRINSICS)
    state->cur.free_bits = 0;
#else
    state->cur.free_bits = SIMD_BIT_BUF_SIZE;
#endif
  } else
#endif
  {
    state->cur.put_buffer.c = 0;
    state->cur.free_bits = BIT_BUF_SIZE;
  }
  STORE_BUFFER()

  return TRUE;
}


#ifdef WITH_SIMD

/* Encode a single block's worth of coefficients */

LOCAL(boolean)
encode_one_block_simd(working_state *state, JCOEFPTR block, int last_dc_val,
                      c_derived_tbl *dctbl, c_derived_tbl *actbl)
{
  JOCTET _buffer[BUFSIZE], *buffer;
  int localbuf = 0;

#ifdef ZERO_BUFFERS
  memset(_buffer, 0, sizeof(_buffer));
#endif

  LOAD_BUFFER()

  buffer = jsimd_huff_encode_one_block(state, buffer, block, last_dc_val,
                                       dctbl, actbl);

  STORE_BUFFER()

  return TRUE;
}

#endif

LOCAL(boolean)
encode_one_block(working_state *state, JCOEFPTR block, int last_dc_val,
                 c_derived_tbl *dctbl, c_derived_tbl *actbl)
{
  int temp, nbits, free_bits;
  bit_buf_type put_buffer;
  JOCTET _buffer[BUFSIZE], *buffer;
  int localbuf = 0;
  int max_coef_bits = state->cinfo->data_precision + 2;

  free_bits = state->cur.free_bits;
  put_buffer = state->cur.put_buffer.c;
  LOAD_BUFFER()

  /* Encode the DC coefficient difference per section F.1.2.1 */

  temp = block[0] - last_dc_val;

  /* This is a well-known technique for obtaining the absolute value without a
   * branch.  It is derived from an assembly language technique presented in
   * "How to Optimize for the Pentium Processors", Copyright (c) 1996, 1997 by
   * Agner Fog.  This code assumes we are on a two's complement machine.
   */
  nbits = temp >> (CHAR_BIT * sizeof(int) - 1);
  temp += nbits;
  nbits ^= temp;

  /* Find the number of bits needed for the magnitude of the coefficient */
  nbits = JPEG_NBITS(nbits);
  /* Check for out-of-range coefficient values.
   * Since we're encoding a difference, the range limit is twice as much.
   */
  if (nbits > max_coef_bits + 1)
    ERREXIT(state->cinfo, JERR_BAD_DCT_COEF);

  /* Emit the Huffman-coded symbol for the number of bits.
   * Emit that number of bits of the value, if positive,
   * or the complement of its magnitude, if negative.
   */
  PUT_CODE(dctbl->ehufco[nbits], dctbl->ehufsi[nbits])

  /* Encode the AC coefficients per section F.1.2.2 */

  {
    int r = 0;                  /* r = run length of zeros */

/* Manually unroll the k loop to eliminate the counter variable.  This
 * improves performance greatly on systems with a limited number of
 * registers (such as x86.)
 */
#define kloop(jpeg_natural_order_of_k) { \
  if ((temp = block[jpeg_natural_order_of_k]) == 0) { \
    r += 16; \
  } else { \
    /* Branch-less absolute value, bitwise complement, etc., same as above */ \
    nbits = temp >> (CHAR_BIT * sizeof(int) - 1); \
    temp += nbits; \
    nbits ^= temp; \
    nbits = JPEG_NBITS_NONZERO(nbits); \
    /* Check for out-of-range coefficient values */ \
    if (nbits > max_coef_bits) \
      ERREXIT(state->cinfo, JERR_BAD_DCT_COEF); \
    /* if run length > 15, must emit special run-length-16 codes (0xF0) */ \
    while (r >= 16 * 16) { \
      r -= 16 * 16; \
      PUT_BITS(actbl->ehufco[0xf0], actbl->ehufsi[0xf0]) \
    } \
    /* Emit Huffman symbol for run length / number of bits */ \
    r += nbits; \
    PUT_CODE(actbl->ehufco[r], actbl->ehufsi[r]) \
    r = 0; \
  } \
}

    /* One iteration for each value in jpeg_natural_order[] */
    kloop(1);   kloop(8);   kloop(16);  kloop(9);   kloop(2);   kloop(3);
    kloop(10);  kloop(17);  kloop(24);  kloop(32);  kloop(25);  kloop(18);
    kloop(11);  kloop(4);   kloop(5);   kloop(12);  kloop(19);  kloop(26);
    kloop(33);  kloop(40);  kloop(48);  kloop(41);  kloop(34);  kloop(27);
    kloop(20);  kloop(13);  kloop(6);   kloop(7);   kloop(14);  kloop(21);
    kloop(28);  kloop(35);  kloop(42);  kloop(49);  kloop(56);  kloop(57);
    kloop(50);  kloop(43);  kloop(36);  kloop(29);  kloop(22);  kloop(15);
    kloop(23);  kloop(30);  kloop(37);  kloop(44);  kloop(51);  kloop(58);
    kloop(59);  kloop(52);  kloop(45);  kloop(38);  kloop(31);  kloop(39);
    kloop(46);  kloop(53);  kloop(60);  kloop(61);  kloop(54);  kloop(47);
    kloop(55);  kloop(62);  kloop(63);

    /* If the last coef(s) were zero, emit an end-of-block code */
    if (r > 0) {
      PUT_BITS(actbl->ehufco[0], actbl->ehufsi[0])
    }
  }

  state->cur.put_buffer.c = put_buffer;
  state->cur.free_bits = free_bits;
  STORE_BUFFER()

  return TRUE;
}


/*
 * Emit a restart marker & resynchronize predictions.
 */

LOCAL(boolean)
emit_restart(working_state *state, int restart_num)
{
  int ci;

  if (!flush_bits(state))
    return FALSE;

  emit_byte(state, 0xFF, return FALSE);
  emit_byte(state, JPEG_RST0 + restart_num, return FALSE);

  /* Re-initialize DC predictions to 0 */
  for (ci = 0; ci < state->cinfo->comps_in_scan; ci++)
    state->cur.last_dc_val[ci] = 0;

  /* The restart counter is not updated until we successfully write the MCU. */

  return TRUE;
}


/*
 * Encode and output one MCU's worth of Huffman-compressed coefficients.
 */

METHODDEF(boolean)
encode_mcu_huff(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr)cinfo->entropy;
  working_state state;
  int blkn, ci;
  jpeg_component_info *compptr;

  /* Load up working state */
  state.next_output_byte = cinfo->dest->next_output_byte;
  state.free_in_buffer = cinfo->dest->free_in_buffer;
  state.cur = entropy->saved;
  state.cinfo = cinfo;
#ifdef WITH_SIMD
  state.simd = entropy->simd;
#endif

  /* Emit restart marker if needed */
  if (cinfo->restart_interval) {
    if (entropy->restarts_to_go == 0)
      if (!emit_restart(&state, entropy->next_restart_num))
        return FALSE;
  }

  /* Encode the MCU data blocks */
#ifdef WITH_SIMD
  if (entropy->simd) {
    for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
      ci = cinfo->MCU_membership[blkn];
      compptr = cinfo->cur_comp_info[ci];
      if (!encode_one_block_simd(&state,
                                 MCU_data[blkn][0], state.cur.last_dc_val[ci],
                                 entropy->dc_derived_tbls[compptr->dc_tbl_no],
                                 entropy->ac_derived_tbls[compptr->ac_tbl_no]))
        return FALSE;
      /* Update last_dc_val */
      state.cur.last_dc_val[ci] = MCU_data[blkn][0][0];
    }
  } else
#endif
  {
    for (blkn = 0; blkn < cinfo->blocks_in_MCU; blkn++) {
      ci = cinfo->MCU_membership[blkn];
      compptr = cinfo->cur_comp_info[ci];
      if (!encode_one_block(&state,
                            MCU_data[blkn][0], state.cur.last_dc_val[ci],
                            entropy->dc_derived_tbls[compptr->dc_tbl_no],
                            entropy->ac_derived_tbls[compptr->ac_tbl_no]))
        return FALSE;
      /* Update last_dc_val */
      state.cur.last_dc_val[ci] = MCU_data[blkn][0][0];
    }
  }

  /* Completed MCU, so update state */
  cinfo->dest->next_output_byte = state.next_output_byte;
  cinfo->dest->free_in_buffer = state.free_in_buffer;
  entropy->saved = state.cur;

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
finish_pass_huff(j_compress_ptr cinfo)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr)cinfo->entropy;
  working_state state;

  /* Load up working state ... flush_bits needs it */
  state.next_output_byte = cinfo->dest->next_output_byte;
  state.free_in_buffer = cinfo->dest->free_in_buffer;
  state.cur = entropy->saved;
  state.cinfo = cinfo;
#ifdef WITH_SIMD
  state.simd = entropy->simd;
#endif

  /* Flush out the last data */
  if (!flush_bits(&state))
    ERREXIT(cinfo, JERR_CANT_SUSPEND);

  /* Update state */
  cinfo->dest->next_output_byte = state.next_output_byte;
  cinfo->dest->free_in_buffer = state.free_in_buffer;
  entropy->saved = state.cur;
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

#ifdef ENTROPY_OPT_SUPPORTED


/* Process a single block's worth of coefficients */

LOCAL(void)
htest_one_block(j_compress_ptr cinfo, JCOEFPTR block, int last_dc_val,
                long dc_counts[], long ac_counts[])
{
  register int temp;
  register int nbits;
  register int k, r;
  int max_coef_bits = cinfo->data_precision + 2;

  /* Encode the DC coefficient difference per section F.1.2.1 */

  temp = block[0] - last_dc_val;
  if (temp < 0)
    temp = -temp;

  /* Find the number of bits needed for the magnitude of the coefficient */
  nbits = 0;
  while (temp) {
    nbits++;
    temp >>= 1;
  }
  /* Check for out-of-range coefficient values.
   * Since we're encoding a difference, the range limit is twice as much.
   */
  if (nbits > max_coef_bits + 1)
    ERREXIT(cinfo, JERR_BAD_DCT_COEF);

  /* Count the Huffman symbol for the number of bits */
  dc_counts[nbits]++;

  /* Encode the AC coefficients per section F.1.2.2 */

  r = 0;                        /* r = run length of zeros */

  for (k = 1; k < DCTSIZE2; k++) {
    if ((temp = block[jpeg_natural_order[k]]) == 0) {
      r++;
    } else {
      /* if run length > 15, must emit special run-length-16 codes (0xF0) */
      while (r > 15) {
        ac_counts[0xF0]++;
        r -= 16;
      }

      /* Find the number of bits needed for the magnitude of the coefficient */
      if (temp < 0)
        temp = -temp;

      /* Find the number of bits needed for the magnitude of the coefficient */
      nbits = 1;                /* there must be at least one 1 bit */
      while ((temp >>= 1))
        nbits++;
      /* Check for out-of-range coefficient values */
      if (nbits > max_coef_bits)
        ERREXIT(cinfo, JERR_BAD_DCT_COEF);

      /* Count Huffman symbol for run length / number of bits */
      ac_counts[(r << 4) + nbits]++;

      r = 0;
    }
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
encode_mcu_gather(j_compress_ptr cinfo, JBLOCKROW *MCU_data)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr)cinfo->entropy;
  int blkn, ci;
  jpeg_component_info *compptr;

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
 * Note this is also used by jcphuff.c and jclhuff.c.
 *
 * The JPEG standard requires that no symbol be assigned a codeword of all
 * one bits (so that padding bits added at the end of a compressed segment
 * can't look like a valid code).  Because of the canonical ordering of
 * codewords, this just means that there must be an unused slot in the
 * longest codeword length category.  Annex K (Clause K.2) of
 * Rec. ITU-T T.81 (1992) | ISO/IEC 10918-1:1994 suggests reserving such a slot
 * by pretending that symbol 256 is a valid symbol with count 1.  In theory
 * that's not optimal; giving it count zero but including it in the symbol set
 * anyway should give a better Huffman code.  But the theoretically better code
 * actually seems to come out worse in practice, because it produces more
 * all-ones bytes (which incur stuffed zero bytes in the final file).  In any
 * case the difference is tiny.
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

GLOBAL(void)
jpeg_gen_optimal_table(j_compress_ptr cinfo, JHUFF_TBL *htbl, long freq[])
{
#define MAX_CLEN  32            /* assumed maximum initial code length */
  UINT8 bits[MAX_CLEN + 1];     /* bits[k] = # of symbols with code length k */
  int bit_pos[MAX_CLEN + 1];    /* # of symbols with smaller code length */
  int codesize[257];            /* codesize[k] = code length of symbol k */
  int nz_index[257];            /* index of nonzero symbol in the original freq
                                   array */
  int others[257];              /* next symbol in current branch of tree */
  int c1, c2;
  int p, i, j;
  int num_nz_symbols;
  long v, v2;

  /* This algorithm is explained in section K.2 of the JPEG standard */

  memset(bits, 0, sizeof(bits));
  memset(codesize, 0, sizeof(codesize));
  for (i = 0; i < 257; i++)
    others[i] = -1;             /* init links to empty */

  freq[256] = 1;                /* make sure 256 has a nonzero count */
  /* Including the pseudo-symbol 256 in the Huffman procedure guarantees
   * that no real symbol is given code-value of all ones, because 256
   * will be placed last in the largest codeword category.
   */

  /* Group nonzero frequencies together so we can more easily find the
   * smallest.
   */
  num_nz_symbols = 0;
  for (i = 0; i < 257; i++) {
    if (freq[i]) {
      nz_index[num_nz_symbols] = i;
      freq[num_nz_symbols] = freq[i];
      num_nz_symbols++;
    }
  }

  /* Huffman's basic algorithm to assign optimal code lengths to symbols */

  for (;;) {
    /* Find the two smallest nonzero frequencies; set c1, c2 = their symbols */
    /* In case of ties, take the larger symbol number.  Since we have grouped
     * the nonzero symbols together, checking for zero symbols is not
     * necessary.
     */
    c1 = -1;
    c2 = -1;
    v = 1000000000L;
    v2 = 1000000000L;
    for (i = 0; i < num_nz_symbols; i++) {
      if (freq[i] <= v2) {
        if (freq[i] <= v) {
          c2 = c1;
          v2 = v;
          v = freq[i];
          c1 = i;
        } else {
          v2 = freq[i];
          c2 = i;
        }
      }
    }

    /* Done if we've merged everything into one frequency */
    if (c2 < 0)
      break;

    /* Else merge the two counts/trees */
    freq[c1] += freq[c2];
    /* Set the frequency to a very high value instead of zero, so we don't have
     * to check for zero values.
     */
    freq[c2] = 1000000001L;

    /* Increment the codesize of everything in c1's tree branch */
    codesize[c1]++;
    while (others[c1] >= 0) {
      c1 = others[c1];
      codesize[c1]++;
    }

    others[c1] = c2;            /* chain c2 onto c1's tree branch */

    /* Increment the codesize of everything in c2's tree branch */
    codesize[c2]++;
    while (others[c2] >= 0) {
      c2 = others[c2];
      codesize[c2]++;
    }
  }

  /* Now count the number of symbols of each code length */
  for (i = 0; i < num_nz_symbols; i++) {
    /* The JPEG standard seems to think that this can't happen, */
    /* but I'm paranoid... */
    if (codesize[i] > MAX_CLEN)
      ERREXIT(cinfo, JERR_HUFF_CLEN_OVERFLOW);

    bits[codesize[i]]++;
  }

  /* Count the number of symbols with a length smaller than i bits, so we can
   * construct the symbol table more efficiently.  Note that this includes the
   * pseudo-symbol 256, but since it is the last symbol, it will not affect the
   * table.
   */
  p = 0;
  for (i = 1; i <= MAX_CLEN; i++) {
    bit_pos[i] = p;
    p += bits[i];
  }

  /* JPEG doesn't allow symbols with code lengths over 16 bits, so if the pure
   * Huffman procedure assigned any such lengths, we must adjust the coding.
   * Here is what Rec. ITU-T T.81 | ISO/IEC 10918-1 says about how this next
   * bit works: Since symbols are paired for the longest Huffman code, the
   * symbols are removed from this length category two at a time.  The prefix
   * for the pair (which is one bit shorter) is allocated to one of the pair;
   * then, skipping the BITS entry for that prefix length, a code word from the
   * next shortest nonzero BITS entry is converted into a prefix for two code
   * words one bit longer.
   */

  for (i = MAX_CLEN; i > 16; i--) {
    while (bits[i] > 0) {
      j = i - 2;                /* find length of new prefix to be used */
      while (bits[j] == 0)
        j--;

      bits[i] -= 2;             /* remove two symbols */
      bits[i - 1]++;            /* one goes in this length */
      bits[j + 1] += 2;         /* two new symbols in this length */
      bits[j]--;                /* symbol of this length is now a prefix */
    }
  }

  /* Remove the count for the pseudo-symbol 256 from the largest codelength */
  while (bits[i] == 0)          /* find largest codelength still in use */
    i--;
  bits[i]--;

  /* Return final symbol counts (only for lengths 0..16) */
  memcpy(htbl->bits, bits, sizeof(htbl->bits));

  /* Return a list of the symbols sorted by code length */
  /* It's not real clear to me why we don't need to consider the codelength
   * changes made above, but Rec. ITU-T T.81 | ISO/IEC 10918-1 seems to think
   * this works.
   */
  for (i = 0; i < num_nz_symbols - 1; i++) {
    htbl->huffval[bit_pos[codesize[i]]] = (UINT8)nz_index[i];
    bit_pos[codesize[i]]++;
  }

  /* Set sent_table FALSE so updated table will be written to JPEG file. */
  htbl->sent_table = FALSE;
}


/*
 * Finish up a statistics-gathering pass and create the new Huffman tables.
 */

METHODDEF(void)
finish_pass_gather(j_compress_ptr cinfo)
{
  huff_entropy_ptr entropy = (huff_entropy_ptr)cinfo->entropy;
  int ci, dctbl, actbl;
  jpeg_component_info *compptr;
  JHUFF_TBL **htblptr;
  boolean did_dc[NUM_HUFF_TBLS];
  boolean did_ac[NUM_HUFF_TBLS];

  /* It's important not to apply jpeg_gen_optimal_table more than once
   * per table, because it clobbers the input frequency counts!
   */
  memset(did_dc, 0, sizeof(did_dc));
  memset(did_ac, 0, sizeof(did_ac));

  for (ci = 0; ci < cinfo->comps_in_scan; ci++) {
    compptr = cinfo->cur_comp_info[ci];
    dctbl = compptr->dc_tbl_no;
    actbl = compptr->ac_tbl_no;
    if (!did_dc[dctbl]) {
      htblptr = &cinfo->dc_huff_tbl_ptrs[dctbl];
      if (*htblptr == NULL)
        *htblptr = jpeg_alloc_huff_table((j_common_ptr)cinfo);
      jpeg_gen_optimal_table(cinfo, *htblptr, entropy->dc_count_ptrs[dctbl]);
      did_dc[dctbl] = TRUE;
    }
    if (!did_ac[actbl]) {
      htblptr = &cinfo->ac_huff_tbl_ptrs[actbl];
      if (*htblptr == NULL)
        *htblptr = jpeg_alloc_huff_table((j_common_ptr)cinfo);
      jpeg_gen_optimal_table(cinfo, *htblptr, entropy->ac_count_ptrs[actbl]);
      did_ac[actbl] = TRUE;
    }
  }
}


#endif /* ENTROPY_OPT_SUPPORTED */


/*
 * Module initialization routine for Huffman entropy encoding.
 */

GLOBAL(void)
jinit_huff_encoder(j_compress_ptr cinfo)
{
  huff_entropy_ptr entropy;
  int i;

  entropy = (huff_entropy_ptr)
    (*cinfo->mem->alloc_small) ((j_common_ptr)cinfo, JPOOL_IMAGE,
                                sizeof(huff_entropy_encoder));
  cinfo->entropy = (struct jpeg_entropy_encoder *)entropy;
  entropy->pub.start_pass = start_pass_huff;

  /* Mark tables unallocated */
  for (i = 0; i < NUM_HUFF_TBLS; i++) {
    entropy->dc_derived_tbls[i] = entropy->ac_derived_tbls[i] = NULL;
#ifdef ENTROPY_OPT_SUPPORTED
    entropy->dc_count_ptrs[i] = entropy->ac_count_ptrs[i] = NULL;
#endif
  }
}
