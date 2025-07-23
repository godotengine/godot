/*
 * jdhuff.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * Lossless JPEG Modifications:
 * Copyright (C) 1999, Ken Murchison.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2010-2011, 2015-2016, 2021, D. R. Commander.
 * Copyright (C) 2018, Matthias Räncker.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains declarations for Huffman entropy decoding routines
 * that are shared between the sequential decoder (jdhuff.c), the progressive
 * decoder (jdphuff.c), and the lossless decoder (jdlhuff.c).  No other modules
 * need to see these.
 */

#include "jconfigint.h"


/* Derived data constructed for each Huffman table */

#define HUFF_LOOKAHEAD  8       /* # of bits of lookahead */

typedef struct {
  /* Basic tables: (element [0] of each array is unused) */
  JLONG maxcode[18];            /* largest code of length k (-1 if none) */
  /* (maxcode[17] is a sentinel to ensure jpeg_huff_decode terminates) */
  JLONG valoffset[18];          /* huffval[] offset for codes of length k */
  /* valoffset[k] = huffval[] index of 1st symbol of code length k, less
   * the smallest code of length k; so given a code of length k, the
   * corresponding symbol is huffval[code + valoffset[k]]
   */

  /* Link to public Huffman table (needed only in jpeg_huff_decode) */
  JHUFF_TBL *pub;

  /* Lookahead table: indexed by the next HUFF_LOOKAHEAD bits of
   * the input data stream.  If the next Huffman code is no more
   * than HUFF_LOOKAHEAD bits long, we can obtain its length and
   * the corresponding symbol directly from this tables.
   *
   * The lower 8 bits of each table entry contain the number of
   * bits in the corresponding Huffman code, or HUFF_LOOKAHEAD + 1
   * if too long.  The next 8 bits of each entry contain the
   * symbol.
   */
  int lookup[1 << HUFF_LOOKAHEAD];
} d_derived_tbl;

/* Expand a Huffman table definition into the derived format */
EXTERN(void) jpeg_make_d_derived_tbl(j_decompress_ptr cinfo, boolean isDC,
                                     int tblno, d_derived_tbl **pdtbl);


/*
 * Fetching the next N bits from the input stream is a time-critical operation
 * for the Huffman decoders.  We implement it with a combination of inline
 * macros and out-of-line subroutines.  Note that N (the number of bits
 * demanded at one time) never exceeds 15 for JPEG use.
 *
 * We read source bytes into get_buffer and dole out bits as needed.
 * If get_buffer already contains enough bits, they are fetched in-line
 * by the macros CHECK_BIT_BUFFER and GET_BITS.  When there aren't enough
 * bits, jpeg_fill_bit_buffer is called; it will attempt to fill get_buffer
 * as full as possible (not just to the number of bits needed; this
 * prefetching reduces the overhead cost of calling jpeg_fill_bit_buffer).
 * Note that jpeg_fill_bit_buffer may return FALSE to indicate suspension.
 * On TRUE return, jpeg_fill_bit_buffer guarantees that get_buffer contains
 * at least the requested number of bits --- dummy zeroes are inserted if
 * necessary.
 */

#if !defined(_WIN32) && !defined(SIZEOF_SIZE_T)
#error Cannot determine word size
#endif

#if SIZEOF_SIZE_T == 8 || defined(_WIN64)

typedef size_t bit_buf_type;            /* type of bit-extraction buffer */
#define BIT_BUF_SIZE  64                /* size of buffer in bits */

#elif defined(__x86_64__) && defined(__ILP32__)

typedef unsigned long long bit_buf_type; /* type of bit-extraction buffer */
#define BIT_BUF_SIZE  64                 /* size of buffer in bits */

#else

typedef unsigned long bit_buf_type;     /* type of bit-extraction buffer */
#define BIT_BUF_SIZE  32                /* size of buffer in bits */

#endif

/* If long is > 32 bits on your machine, and shifting/masking longs is
 * reasonably fast, making bit_buf_type be long and setting BIT_BUF_SIZE
 * appropriately should be a win.  Unfortunately we can't define the size
 * with something like  #define BIT_BUF_SIZE (sizeof(bit_buf_type)*8)
 * because not all machines measure sizeof in 8-bit bytes.
 */

typedef struct {                /* Bitreading state saved across MCUs */
  bit_buf_type get_buffer;      /* current bit-extraction buffer */
  int bits_left;                /* # of unused bits in it */
} bitread_perm_state;

typedef struct {                /* Bitreading working state within an MCU */
  /* Current data source location */
  /* We need a copy, rather than munging the original, in case of suspension */
  const JOCTET *next_input_byte; /* => next byte to read from source */
  size_t bytes_in_buffer;       /* # of bytes remaining in source buffer */
  /* Bit input buffer --- note these values are kept in register variables,
   * not in this struct, inside the inner loops.
   */
  bit_buf_type get_buffer;      /* current bit-extraction buffer */
  int bits_left;                /* # of unused bits in it */
  /* Pointer needed by jpeg_fill_bit_buffer. */
  j_decompress_ptr cinfo;       /* back link to decompress master record */
} bitread_working_state;

/* Macros to declare and load/save bitread local variables. */
#define BITREAD_STATE_VARS \
  register bit_buf_type get_buffer; \
  register int bits_left; \
  bitread_working_state br_state

#define BITREAD_LOAD_STATE(cinfop, permstate) \
  br_state.cinfo = cinfop; \
  br_state.next_input_byte = cinfop->src->next_input_byte; \
  br_state.bytes_in_buffer = cinfop->src->bytes_in_buffer; \
  get_buffer = permstate.get_buffer; \
  bits_left = permstate.bits_left;

#define BITREAD_SAVE_STATE(cinfop, permstate) \
  cinfop->src->next_input_byte = br_state.next_input_byte; \
  cinfop->src->bytes_in_buffer = br_state.bytes_in_buffer; \
  permstate.get_buffer = get_buffer; \
  permstate.bits_left = bits_left

/*
 * These macros provide the in-line portion of bit fetching.
 * Use CHECK_BIT_BUFFER to ensure there are N bits in get_buffer
 * before using GET_BITS, PEEK_BITS, or DROP_BITS.
 * The variables get_buffer and bits_left are assumed to be locals,
 * but the state struct might not be (jpeg_huff_decode needs this).
 *      CHECK_BIT_BUFFER(state, n, action);
 *              Ensure there are N bits in get_buffer; if suspend, take action.
 *      val = GET_BITS(n);
 *              Fetch next N bits.
 *      val = PEEK_BITS(n);
 *              Fetch next N bits without removing them from the buffer.
 *      DROP_BITS(n);
 *              Discard next N bits.
 * The value N should be a simple variable, not an expression, because it
 * is evaluated multiple times.
 */

#define CHECK_BIT_BUFFER(state, nbits, action) { \
  if (bits_left < (nbits)) { \
    if (!jpeg_fill_bit_buffer(&(state), get_buffer, bits_left, nbits)) \
      { action; } \
    get_buffer = (state).get_buffer;  bits_left = (state).bits_left; \
  } \
}

#define GET_BITS(nbits) \
  (((int)(get_buffer >> (bits_left -= (nbits)))) & ((1 << (nbits)) - 1))

#define PEEK_BITS(nbits) \
  (((int)(get_buffer >> (bits_left -  (nbits)))) & ((1 << (nbits)) - 1))

#define DROP_BITS(nbits) \
  (bits_left -= (nbits))

/* Load up the bit buffer to a depth of at least nbits */
EXTERN(boolean) jpeg_fill_bit_buffer(bitread_working_state *state,
                                     register bit_buf_type get_buffer,
                                     register int bits_left, int nbits);


/*
 * Code for extracting next Huffman-coded symbol from input bit stream.
 * Again, this is time-critical and we make the main paths be macros.
 *
 * We use a lookahead table to process codes of up to HUFF_LOOKAHEAD bits
 * without looping.  Usually, more than 95% of the Huffman codes will be 8
 * or fewer bits long.  The few overlength codes are handled with a loop,
 * which need not be inline code.
 *
 * Notes about the HUFF_DECODE macro:
 * 1. Near the end of the data segment, we may fail to get enough bits
 *    for a lookahead.  In that case, we do it the hard way.
 * 2. If the lookahead table contains no entry, the next code must be
 *    more than HUFF_LOOKAHEAD bits long.
 * 3. jpeg_huff_decode returns -1 if forced to suspend.
 */

#define HUFF_DECODE(result, state, htbl, failaction, slowlabel) { \
  register int nb, look; \
  if (bits_left < HUFF_LOOKAHEAD) { \
    if (!jpeg_fill_bit_buffer(&state, get_buffer, bits_left, 0)) \
      { failaction; } \
    get_buffer = state.get_buffer;  bits_left = state.bits_left; \
    if (bits_left < HUFF_LOOKAHEAD) { \
      nb = 1;  goto slowlabel; \
    } \
  } \
  look = PEEK_BITS(HUFF_LOOKAHEAD); \
  if ((nb = (htbl->lookup[look] >> HUFF_LOOKAHEAD)) <= HUFF_LOOKAHEAD) { \
    DROP_BITS(nb); \
    result = htbl->lookup[look] & ((1 << HUFF_LOOKAHEAD) - 1); \
  } else { \
slowlabel: \
    if ((result = \
         jpeg_huff_decode(&state, get_buffer, bits_left, htbl, nb)) < 0) \
      { failaction; } \
    get_buffer = state.get_buffer;  bits_left = state.bits_left; \
  } \
}

#define HUFF_DECODE_FAST(s, nb, htbl) \
  FILL_BIT_BUFFER_FAST; \
  s = PEEK_BITS(HUFF_LOOKAHEAD); \
  s = htbl->lookup[s]; \
  nb = s >> HUFF_LOOKAHEAD; \
  /* Pre-execute the common case of nb <= HUFF_LOOKAHEAD */ \
  DROP_BITS(nb); \
  s = s & ((1 << HUFF_LOOKAHEAD) - 1); \
  if (nb > HUFF_LOOKAHEAD) { \
    /* Equivalent of jpeg_huff_decode() */ \
    /* Don't use GET_BITS() here because we don't want to modify bits_left */ \
    s = (get_buffer >> bits_left) & ((1 << (nb)) - 1); \
    while (s > htbl->maxcode[nb]) { \
      s <<= 1; \
      s |= GET_BITS(1); \
      nb++; \
    } \
    if (nb > 16) \
      s = 0; \
    else \
      s = htbl->pub->huffval[(int)(s + htbl->valoffset[nb]) & 0xFF]; \
  }

/* Out-of-line case for Huffman code fetching */
EXTERN(int) jpeg_huff_decode(bitread_working_state *state,
                             register bit_buf_type get_buffer,
                             register int bits_left, d_derived_tbl *htbl,
                             int min_bits);
