/*
 * jchuff.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1997, Thomas G. Lane.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2009, 2018, 2021, D. R. Commander.
 * Copyright (C) 2018, Matthias Räncker.
 * Copyright (C) 2020-2021, Arm Limited.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

/* Expanded entropy encoder object for Huffman encoding.
 *
 * The savable_state subrecord contains fields that change within an MCU,
 * but must not be updated permanently until we complete the MCU.
 */

#if defined(__aarch64__) || defined(_M_ARM64)
#define BIT_BUF_SIZE  64
#else
#define BIT_BUF_SIZE  32
#endif

typedef struct {
  size_t put_buffer;                    /* current bit accumulation buffer */
  int free_bits;                        /* # of bits available in it */
  int last_dc_val[MAX_COMPS_IN_SCAN];   /* last DC coef for each component */
} savable_state;

typedef struct {
  JOCTET *next_output_byte;     /* => next byte to write in buffer */
  size_t free_in_buffer;        /* # of byte spaces remaining in buffer */
  savable_state cur;            /* Current bit buffer & DC state */
  j_compress_ptr cinfo;         /* dump_buffer needs access to this */
  int simd;
} working_state;

/* Outputting bits to the file */

/* Output byte b and, speculatively, an additional 0 byte. 0xFF must be encoded
 * as 0xFF 0x00, so the output buffer pointer is advanced by 2 if the byte is
 * 0xFF.  Otherwise, the output buffer pointer is advanced by 1, and the
 * speculative 0 byte will be overwritten by the next byte.
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
#if defined(__aarch64__) || defined(_M_ARM64)

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
    *((uint64_t *)buffer) = BUILTIN_BSWAP64(put_buffer); \
    buffer += 8; \
  } \
}

#else

#if defined(_MSC_VER) && !defined(__clang__)
#define SPLAT() { \
  buffer[0] = (JOCTET)(put_buffer >> 24); \
  buffer[1] = (JOCTET)(put_buffer >> 16); \
  buffer[2] = (JOCTET)(put_buffer >>  8); \
  buffer[3] = (JOCTET)(put_buffer      ); \
  buffer += 4; \
}
#else
#define SPLAT() { \
  put_buffer = __builtin_bswap32(put_buffer); \
  __asm__("str %1, [%0], #4" : "+r" (buffer) : "r" (put_buffer)); \
}
#endif

#define FLUSH() { \
  if (put_buffer & 0x80808080 & ~(put_buffer + 0x01010101)) { \
    EMIT_BYTE(put_buffer >> 24) \
    EMIT_BYTE(put_buffer >> 16) \
    EMIT_BYTE(put_buffer >>  8) \
    EMIT_BYTE(put_buffer      ) \
  } else { \
    SPLAT(); \
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

#define PUT_CODE(code, size, diff) { \
  diff |= code << nbits; \
  nbits += size; \
  PUT_BITS(diff, nbits) \
}
