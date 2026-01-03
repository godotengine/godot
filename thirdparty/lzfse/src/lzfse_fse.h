/*
Copyright (c) 2015-2016, Apple Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1.  Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the distribution.

3.  Neither the name of the copyright holder(s) nor the names of any contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Finite state entropy coding (FSE)
// This is an implementation of the tANS algorithm described by Jarek Duda,
// we use the more descriptive name "Finite State Entropy".

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

//  Select between 32/64-bit I/O streams for FSE. Note that the FSE stream
//  size need not match the word size of the machine, but in practice you
//  want to use 64b streams on 64b systems for better performance.
#if defined(_M_AMD64) || defined(__x86_64__) || defined(__arm64__)
#define FSE_IOSTREAM_64 1
#else
#define FSE_IOSTREAM_64 0
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#  define FSE_INLINE __forceinline
#  define inline __inline
#  pragma warning(disable : 4068) // warning C4068: unknown pragma
#else
#  define FSE_INLINE static inline __attribute__((__always_inline__))
#endif

// MARK: - Bit utils

/*! @abstract Signed type used to represent bit count. */
typedef int32_t fse_bit_count;

/*! @abstract Unsigned type used to represent FSE state. */
typedef uint16_t fse_state;

// Mask the NBITS lsb of X. 0 <= NBITS < 64
static inline uint64_t fse_mask_lsb64(uint64_t x, fse_bit_count nbits) {
  static const uint64_t mtable[65] = {
      0x0000000000000000LLU, 0x0000000000000001LLU, 0x0000000000000003LLU,
      0x0000000000000007LLU, 0x000000000000000fLLU, 0x000000000000001fLLU,
      0x000000000000003fLLU, 0x000000000000007fLLU, 0x00000000000000ffLLU,
      0x00000000000001ffLLU, 0x00000000000003ffLLU, 0x00000000000007ffLLU,
      0x0000000000000fffLLU, 0x0000000000001fffLLU, 0x0000000000003fffLLU,
      0x0000000000007fffLLU, 0x000000000000ffffLLU, 0x000000000001ffffLLU,
      0x000000000003ffffLLU, 0x000000000007ffffLLU, 0x00000000000fffffLLU,
      0x00000000001fffffLLU, 0x00000000003fffffLLU, 0x00000000007fffffLLU,
      0x0000000000ffffffLLU, 0x0000000001ffffffLLU, 0x0000000003ffffffLLU,
      0x0000000007ffffffLLU, 0x000000000fffffffLLU, 0x000000001fffffffLLU,
      0x000000003fffffffLLU, 0x000000007fffffffLLU, 0x00000000ffffffffLLU,
      0x00000001ffffffffLLU, 0x00000003ffffffffLLU, 0x00000007ffffffffLLU,
      0x0000000fffffffffLLU, 0x0000001fffffffffLLU, 0x0000003fffffffffLLU,
      0x0000007fffffffffLLU, 0x000000ffffffffffLLU, 0x000001ffffffffffLLU,
      0x000003ffffffffffLLU, 0x000007ffffffffffLLU, 0x00000fffffffffffLLU,
      0x00001fffffffffffLLU, 0x00003fffffffffffLLU, 0x00007fffffffffffLLU,
      0x0000ffffffffffffLLU, 0x0001ffffffffffffLLU, 0x0003ffffffffffffLLU,
      0x0007ffffffffffffLLU, 0x000fffffffffffffLLU, 0x001fffffffffffffLLU,
      0x003fffffffffffffLLU, 0x007fffffffffffffLLU, 0x00ffffffffffffffLLU,
      0x01ffffffffffffffLLU, 0x03ffffffffffffffLLU, 0x07ffffffffffffffLLU,
      0x0fffffffffffffffLLU, 0x1fffffffffffffffLLU, 0x3fffffffffffffffLLU,
      0x7fffffffffffffffLLU, 0xffffffffffffffffLLU,
  };
  return x & mtable[nbits];
}

// Mask the NBITS lsb of X. 0 <= NBITS < 32
static inline uint32_t fse_mask_lsb32(uint32_t x, fse_bit_count nbits) {
  static const uint32_t mtable[33] = {
      0x0000000000000000U, 0x0000000000000001U, 0x0000000000000003U,
      0x0000000000000007U, 0x000000000000000fU, 0x000000000000001fU,
      0x000000000000003fU, 0x000000000000007fU, 0x00000000000000ffU,
      0x00000000000001ffU, 0x00000000000003ffU, 0x00000000000007ffU,
      0x0000000000000fffU, 0x0000000000001fffU, 0x0000000000003fffU,
      0x0000000000007fffU, 0x000000000000ffffU, 0x000000000001ffffU,
      0x000000000003ffffU, 0x000000000007ffffU, 0x00000000000fffffU,
      0x00000000001fffffU, 0x00000000003fffffU, 0x00000000007fffffU,
      0x0000000000ffffffU, 0x0000000001ffffffU, 0x0000000003ffffffU,
      0x0000000007ffffffU, 0x000000000fffffffU, 0x000000001fffffffU,
      0x000000003fffffffU, 0x000000007fffffffU, 0x00000000ffffffffU,
  };
  return x & mtable[nbits];
}

/*! @abstract Select \c nbits at index \c start from \c x.
 *  0 <= start <= start+nbits <= 64 */
FSE_INLINE uint64_t fse_extract_bits64(uint64_t x, fse_bit_count start,
                                       fse_bit_count nbits) {
#if defined(__GNUC__)
  // If START and NBITS are constants, map to bit-field extraction instructions
  if (__builtin_constant_p(start) && __builtin_constant_p(nbits))
    return (x >> start) & ((1LLU << nbits) - 1LLU);
#endif

  // Otherwise, shift and mask
  return fse_mask_lsb64(x >> start, nbits);
}

/*! @abstract Select \c nbits at index \c start from \c x.
 *  0 <= start <= start+nbits <= 32 */
FSE_INLINE uint32_t fse_extract_bits32(uint32_t x, fse_bit_count start,
                                       fse_bit_count nbits) {
#if defined(__GNUC__)
  // If START and NBITS are constants, map to bit-field extraction instructions
  if (__builtin_constant_p(start) && __builtin_constant_p(nbits))
    return (x >> start) & ((1U << nbits) - 1U);
#endif

  // Otherwise, shift and mask
  return fse_mask_lsb32(x >> start, nbits);
}

// MARK: - Bit stream

// I/O streams
// The streams can be shared between several FSE encoders/decoders, which is why
// they are not in the state struct

/*! @abstract Output stream, 64-bit accum. */
typedef struct {
  uint64_t accum;            // Output bits
  fse_bit_count accum_nbits; // Number of valid bits in ACCUM, other bits are 0
} fse_out_stream64;

/*! @abstract Output stream, 32-bit accum. */
typedef struct {
  uint32_t accum;            // Output bits
  fse_bit_count accum_nbits; // Number of valid bits in ACCUM, other bits are 0
} fse_out_stream32;

/*! @abstract Object representing an input stream. */
typedef struct {
  uint64_t accum;            // Input bits
  fse_bit_count accum_nbits; // Number of valid bits in ACCUM, other bits are 0
} fse_in_stream64;

/*! @abstract Object representing an input stream. */
typedef struct {
  uint32_t accum;            // Input bits
  fse_bit_count accum_nbits; // Number of valid bits in ACCUM, other bits are 0
} fse_in_stream32;

/*! @abstract Initialize an output stream object. */
FSE_INLINE void fse_out_init64(fse_out_stream64 *s) {
  s->accum = 0;
  s->accum_nbits = 0;
}

/*! @abstract Initialize an output stream object. */
FSE_INLINE void fse_out_init32(fse_out_stream32 *s) {
  s->accum = 0;
  s->accum_nbits = 0;
}

/*! @abstract Write full bytes from the accumulator to output buffer, ensuring
 * accum_nbits is in [0, 7].
 * We assume we can write 8 bytes to the output buffer \c (*pbuf[0..7]) in all
 * cases.
 * @note *pbuf is incremented by the number of written bytes. */
FSE_INLINE void fse_out_flush64(fse_out_stream64 *s, uint8_t **pbuf) {
  fse_bit_count nbits =
      s->accum_nbits & -8; // number of bits written, multiple of 8

  // Write 8 bytes of current accumulator
  memcpy(*pbuf, &(s->accum), 8);
  *pbuf += (nbits >> 3); // bytes

  // Update state
  s->accum >>= nbits; // remove nbits
  s->accum_nbits -= nbits;

  assert(s->accum_nbits >= 0 && s->accum_nbits <= 7);
  assert(s->accum_nbits == 64 || (s->accum >> s->accum_nbits) == 0);
}

/*! @abstract Write full bytes from the accumulator to output buffer, ensuring
 * accum_nbits is in [0, 7].
 * We assume we can write 4 bytes to the output buffer \c (*pbuf[0..3]) in all
 * cases.
 * @note *pbuf is incremented by the number of written bytes. */
FSE_INLINE void fse_out_flush32(fse_out_stream32 *s, uint8_t **pbuf) {
  fse_bit_count nbits =
      s->accum_nbits & -8; // number of bits written, multiple of 8

  // Write 4 bytes of current accumulator
  memcpy(*pbuf, &(s->accum), 4);
  *pbuf += (nbits >> 3); // bytes

  // Update state
  s->accum >>= nbits; // remove nbits
  s->accum_nbits -= nbits;

  assert(s->accum_nbits >= 0 && s->accum_nbits <= 7);
  assert(s->accum_nbits == 32 || (s->accum >> s->accum_nbits) == 0);
}

/*! @abstract Write the last bytes from the accumulator to output buffer,
 * ensuring accum_nbits is in [-7, 0]. Bits are padded with 0 if needed.
 * We assume we can write 8 bytes to the output buffer \c (*pbuf[0..7]) in all
 * cases.
 * @note *pbuf is incremented by the number of written bytes. */
FSE_INLINE void fse_out_finish64(fse_out_stream64 *s, uint8_t **pbuf) {
  fse_bit_count nbits =
      (s->accum_nbits + 7) & -8; // number of bits written, multiple of 8

  // Write 8 bytes of current accumulator
  memcpy(*pbuf, &(s->accum), 8);
  *pbuf += (nbits >> 3); // bytes

  // Update state
  s->accum = 0; // remove nbits
  s->accum_nbits -= nbits;

  assert(s->accum_nbits >= -7 && s->accum_nbits <= 0);
}

/*! @abstract Write the last bytes from the accumulator to output buffer,
 * ensuring accum_nbits is in [-7, 0]. Bits are padded with 0 if needed.
 * We assume we can write 4 bytes to the output buffer \c (*pbuf[0..3]) in all
 * cases.
 * @note *pbuf is incremented by the number of written bytes. */
FSE_INLINE void fse_out_finish32(fse_out_stream32 *s, uint8_t **pbuf) {
  fse_bit_count nbits =
      (s->accum_nbits + 7) & -8; // number of bits written, multiple of 8

  // Write 8 bytes of current accumulator
  memcpy(*pbuf, &(s->accum), 4);
  *pbuf += (nbits >> 3); // bytes

  // Update state
  s->accum = 0; // remove nbits
  s->accum_nbits -= nbits;

  assert(s->accum_nbits >= -7 && s->accum_nbits <= 0);
}

/*! @abstract Accumulate \c n bits \c b to output stream \c s. We \b must have:
 * 0 <= b < 2^n, and N + s->accum_nbits <= 64.
 * @note The caller must ensure out_flush is called \b before the accumulator
 * overflows to more than 64 bits. */
FSE_INLINE void fse_out_push64(fse_out_stream64 *s, fse_bit_count n,
                               uint64_t b) {
  s->accum |= b << s->accum_nbits;
  s->accum_nbits += n;

  assert(s->accum_nbits >= 0 && s->accum_nbits <= 64);
  assert(s->accum_nbits == 64 || (s->accum >> s->accum_nbits) == 0);
}

/*! @abstract Accumulate \c n bits \c b to output stream \c s. We \b must have:
 * 0 <= n < 2^n, and n + s->accum_nbits <= 32.
 * @note The caller must ensure out_flush is called \b before the accumulator
 * overflows to more than 32 bits. */
FSE_INLINE void fse_out_push32(fse_out_stream32 *s, fse_bit_count n,
                               uint32_t b) {
  s->accum |= b << s->accum_nbits;
  s->accum_nbits += n;

  assert(s->accum_nbits >= 0 && s->accum_nbits <= 32);
  assert(s->accum_nbits == 32 || (s->accum >> s->accum_nbits) == 0);
}

#if FSE_IOSTREAM_64
#define DEBUG_CHECK_INPUT_STREAM_PARAMETERS                                    \
  assert(s->accum_nbits >= 56 && s->accum_nbits < 64);                         \
  assert((s->accum >> s->accum_nbits) == 0);
#else
#define DEBUG_CHECK_INPUT_STREAM_PARAMETERS                                    \
  assert(s->accum_nbits >= 24 && s->accum_nbits < 32);                         \
  assert((s->accum >> s->accum_nbits) == 0);
#endif

/*! @abstract   Initialize the fse input stream so that accum holds between 56
 *  and 63 bits. We never want to have 64 bits in the stream, because that allows
 *  us to avoid a special case in the fse_in_pull function (eliminating an
 *  unpredictable branch), while not requiring any additional fse_flush
 *  operations. This is why we have the special case for n == 0 (in which case
 *  we want to load only 7 bytes instead of 8). */
FSE_INLINE int fse_in_checked_init64(fse_in_stream64 *s, fse_bit_count n,
                                     const uint8_t **pbuf,
                                     const uint8_t *buf_start) {
  if (n) {
    if (*pbuf < buf_start + 8)
      return -1; // out of range
    *pbuf -= 8;
    memcpy(&(s->accum), *pbuf, 8);
    s->accum_nbits = n + 64;
  } else {
    if (*pbuf < buf_start + 7)
      return -1; // out of range
    *pbuf -= 7;
    memcpy(&(s->accum), *pbuf, 7);
    s->accum &= 0xffffffffffffff;
    s->accum_nbits = n + 56;
  }

  if ((s->accum_nbits < 56 || s->accum_nbits >= 64) ||
      ((s->accum >> s->accum_nbits) != 0)) {
    return -1; // the incoming input is wrong (encoder should have zeroed the
               // upper bits)
  }

  return 0; // OK
}

/*! @abstract Identical to previous function, but for 32-bit operation
 * (resulting bit count is between 24 and 31 bits). */
FSE_INLINE int fse_in_checked_init32(fse_in_stream32 *s, fse_bit_count n,
                                     const uint8_t **pbuf,
                                     const uint8_t *buf_start) {
  if (n) {
    if (*pbuf < buf_start + 4)
      return -1; // out of range
    *pbuf -= 4;
    memcpy(&(s->accum), *pbuf, 4);
    s->accum_nbits = n + 32;
  } else {
    if (*pbuf < buf_start + 3)
      return -1; // out of range
    *pbuf -= 3;
    memcpy(&(s->accum), *pbuf, 3);
    s->accum &= 0xffffff;
    s->accum_nbits = n + 24;
  }

  if ((s->accum_nbits < 24 || s->accum_nbits >= 32) ||
      ((s->accum >> s->accum_nbits) != 0)) {
    return -1; // the incoming input is wrong (encoder should have zeroed the
               // upper bits)
  }

  return 0; // OK
}

/*! @abstract  Read in new bytes from buffer to ensure that we have a full
 * complement of bits in the stream object (again, between 56 and 63 bits).
 * checking the new value of \c *pbuf remains >= \c buf_start.
 * @return 0 if OK.
 * @return -1 on failure. */
FSE_INLINE int fse_in_checked_flush64(fse_in_stream64 *s, const uint8_t **pbuf,
                                      const uint8_t *buf_start) {
  //  Get number of bits to add to bring us into the desired range.
  fse_bit_count nbits = (63 - s->accum_nbits) & -8;
  //  Convert bits to bytes and decrement buffer address, then load new data.
  const uint8_t *buf = (*pbuf) - (nbits >> 3);
  if (buf < buf_start) {
    return -1; // out of range
  }
  *pbuf = buf;
  uint64_t incoming;
  memcpy(&incoming, buf, 8);
  // Update the state object and verify its validity (in DEBUG).
  s->accum = (s->accum << nbits) | fse_mask_lsb64(incoming, nbits);
  s->accum_nbits += nbits;
  DEBUG_CHECK_INPUT_STREAM_PARAMETERS
  return 0; // OK
}

/*! @abstract Identical to previous function (but again, we're only filling
 * a 32-bit field with between 24 and 31 bits). */
FSE_INLINE int fse_in_checked_flush32(fse_in_stream32 *s, const uint8_t **pbuf,
                                      const uint8_t *buf_start) {
  //  Get number of bits to add to bring us into the desired range.
  fse_bit_count nbits = (31 - s->accum_nbits) & -8;

  if (nbits > 0) {
    //  Convert bits to bytes and decrement buffer address, then load new data.
    const uint8_t *buf = (*pbuf) - (nbits >> 3);
    if (buf < buf_start) {
      return -1; // out of range
    }

    *pbuf = buf;

    uint32_t incoming = *((uint32_t *)buf);

    // Update the state object and verify its validity (in DEBUG).
    s->accum = (s->accum << nbits) | fse_mask_lsb32(incoming, nbits);
    s->accum_nbits += nbits;
  }
  DEBUG_CHECK_INPUT_STREAM_PARAMETERS
  return 0; // OK
}

/*! @abstract Pull n bits out of the fse stream object. */
FSE_INLINE uint64_t fse_in_pull64(fse_in_stream64 *s, fse_bit_count n) {
  assert(n >= 0 && n <= s->accum_nbits);
  s->accum_nbits -= n;
  uint64_t result = s->accum >> s->accum_nbits;
  s->accum = fse_mask_lsb64(s->accum, s->accum_nbits);
  return result;
}

/*! @abstract Pull n bits out of the fse stream object. */
FSE_INLINE uint32_t fse_in_pull32(fse_in_stream32 *s, fse_bit_count n) {
  assert(n >= 0 && n <= s->accum_nbits);
  s->accum_nbits -= n;
  uint32_t result = s->accum >> s->accum_nbits;
  s->accum = fse_mask_lsb32(s->accum, s->accum_nbits);
  return result;
}

// MARK: - Encode/Decode

// Map to 32/64-bit implementations and types for I/O
#if FSE_IOSTREAM_64

typedef uint64_t fse_bits;
typedef fse_out_stream64 fse_out_stream;
typedef fse_in_stream64 fse_in_stream;
#define fse_mask_lsb fse_mask_lsb64
#define fse_extract_bits fse_extract_bits64
#define fse_out_init fse_out_init64
#define fse_out_flush fse_out_flush64
#define fse_out_finish fse_out_finish64
#define fse_out_push fse_out_push64
#define fse_in_init fse_in_checked_init64
#define fse_in_checked_init fse_in_checked_init64
#define fse_in_flush fse_in_checked_flush64
#define fse_in_checked_flush fse_in_checked_flush64
#define fse_in_flush2(_unused, _parameters, _unused2) 0 /* nothing */
#define fse_in_checked_flush2(_unused, _parameters)     /* nothing */
#define fse_in_pull fse_in_pull64

#else

typedef uint32_t fse_bits;
typedef fse_out_stream32 fse_out_stream;
typedef fse_in_stream32 fse_in_stream;
#define fse_mask_lsb fse_mask_lsb32
#define fse_extract_bits fse_extract_bits32
#define fse_out_init fse_out_init32
#define fse_out_flush fse_out_flush32
#define fse_out_finish fse_out_finish32
#define fse_out_push fse_out_push32
#define fse_in_init fse_in_checked_init32
#define fse_in_checked_init fse_in_checked_init32
#define fse_in_flush fse_in_checked_flush32
#define fse_in_checked_flush fse_in_checked_flush32
#define fse_in_flush2 fse_in_checked_flush32
#define fse_in_checked_flush2 fse_in_checked_flush32
#define fse_in_pull fse_in_pull32

#endif

/*! @abstract Entry for one symbol in the encoder table (64b). */
typedef struct {
  int16_t s0;     // First state requiring a K-bit shift
  int16_t k;      // States S >= S0 are shifted K bits. States S < S0 are
                  // shifted K-1 bits
  int16_t delta0; // Relative increment used to compute next state if S >= S0
  int16_t delta1; // Relative increment used to compute next state if S < S0
} fse_encoder_entry;

/*! @abstract  Entry for one state in the decoder table (32b). */
typedef struct {  // DO NOT REORDER THE FIELDS
  int8_t k;       // Number of bits to read
  uint8_t symbol; // Emitted symbol
  int16_t delta;  // Signed increment used to compute next state (+bias)
} fse_decoder_entry;

/*! @abstract  Entry for one state in the value decoder table (64b). */
typedef struct {      // DO NOT REORDER THE FIELDS
  uint8_t total_bits; // state bits + extra value bits = shift for next decode
  uint8_t value_bits; // extra value bits
  int16_t delta;      // state base (delta)
  int32_t vbase;      // value base
} fse_value_decoder_entry;

/*! @abstract Encode SYMBOL using the encoder table, and update \c *pstate,
 *  \c out.
 *  @note The caller must ensure we have enough bits available in the output
 *  stream accumulator. */
FSE_INLINE void fse_encode(fse_state *__restrict pstate,
                           const fse_encoder_entry *__restrict encoder_table,
                           fse_out_stream *__restrict out, uint8_t symbol) {
  int s = *pstate;
  fse_encoder_entry e = encoder_table[symbol];
  int s0 = e.s0;
  int k = e.k;
  int delta0 = e.delta0;
  int delta1 = e.delta1;

  // Number of bits to write
  int hi = s >= s0;
  fse_bit_count nbits = hi ? k : (k - 1);
  fse_state delta = hi ? delta0 : delta1;

  // Write lower NBITS of state
  fse_bits b = fse_mask_lsb(s, nbits);
  fse_out_push(out, nbits, b);

  // Update state with remaining bits and delta
  *pstate = delta + (s >> nbits);
}

/*! @abstract Decode and return symbol using the decoder table, and update
 *  \c *pstate, \c in.
 *  @note The caller must ensure we have enough bits available in the input
 *  stream accumulator. */
FSE_INLINE uint8_t fse_decode(fse_state *__restrict pstate,
                              const int32_t *__restrict decoder_table,
                              fse_in_stream *__restrict in) {
  int32_t e = decoder_table[*pstate];

  // Update state from K bits of input + DELTA
  *pstate = (fse_state)(e >> 16) + (fse_state)fse_in_pull(in, e & 0xff);

  // Return the symbol for this state
  return fse_extract_bits(e, 8, 8); // symbol
}

/*! @abstract Decode and return value using the decoder table, and update \c
 *  *pstate, \c in.
 * \c value_decoder_table[nstates]
 * @note The caller must ensure we have enough bits available in the input
 * stream accumulator. */
FSE_INLINE int32_t
fse_value_decode(fse_state *__restrict pstate,
                 const fse_value_decoder_entry *value_decoder_table,
                 fse_in_stream *__restrict in) {
  fse_value_decoder_entry entry = value_decoder_table[*pstate];
  uint32_t state_and_value_bits = (uint32_t)fse_in_pull(in, entry.total_bits);
  *pstate =
      (fse_state)(entry.delta + (state_and_value_bits >> entry.value_bits));
  return (int32_t)(entry.vbase +
                   fse_mask_lsb(state_and_value_bits, entry.value_bits));
}

// MARK: - Tables

// IMPORTANT: To properly decode an FSE encoded stream, both encoder/decoder
// tables shall be initialized with the same parameters, including the
// FREQ[NSYMBOL] array.
//

/*! @abstract Sanity check on frequency table, verify sum of \c freq
 *  is <= \c number_of_states. */
FSE_INLINE int fse_check_freq(const uint16_t *freq_table,
                              const size_t table_size,
                              const size_t number_of_states) {
  size_t sum_of_freq = 0;
  for (int i = 0; i < table_size; i++) {
    sum_of_freq += freq_table[i];
  }
  return (sum_of_freq > number_of_states) ? -1 : 0;
}

/*! @abstract Initialize encoder table \c t[nsymbols].
 *
 * @param nstates
 * sum \c freq[i]; the number of states (a power of 2).
 *
 * @param nsymbols
 * the number of symbols.
 *
 * @param freq[nsymbols]
 * is a normalized histogram of symbol frequencies, with \c freq[i] >= 0.
 * Some symbols may have a 0 frequency. In that case they should not be
 * present in the data.
 */
void fse_init_encoder_table(int nstates, int nsymbols,
                            const uint16_t *__restrict freq,
                            fse_encoder_entry *__restrict t);

/*! @abstract Initialize decoder table \c t[nstates].
 *
 * @param nstates
 * sum \c freq[i]; the number of states (a power of 2).
 *
 * @param nsymbols
 * the number of symbols.
 *
 * @param feq[nsymbols]
 * a normalized histogram of symbol frequencies, with \c freq[i] >= 0.
 * Some symbols may have a 0 frequency. In that case they should not be
 * present in the data.
 *
 * @return 0 if OK.
 * @return -1 on failure.
 */
int fse_init_decoder_table(int nstates, int nsymbols,
                           const uint16_t *__restrict freq,
                           int32_t *__restrict t);

/*! @abstract Initialize value decoder table \c t[nstates].
 *
 * @param nstates
 * sum \cfreq[i]; the number of states (a power of 2).
 *
 * @param nsymbols
 * the number of symbols.
 *
 * @param freq[nsymbols]
 * a normalized histogram of symbol frequencies, with \c freq[i] >= 0.
 * \c symbol_vbits[nsymbols] and \c symbol_vbase[nsymbols] are the number of
 * value bits to read and the base value for each symbol.
 * Some symbols may have a 0 frequency.  In that case they should not be
 * present in the data.
 */
void fse_init_value_decoder_table(int nstates, int nsymbols,
                                  const uint16_t *__restrict freq,
                                  const uint8_t *__restrict symbol_vbits,
                                  const int32_t *__restrict symbol_vbase,
                                  fse_value_decoder_entry *__restrict t);

/*! @abstract Normalize a table \c t[nsymbols] of occurrences to
 *  \c freq[nsymbols]. */
void fse_normalize_freq(int nstates, int nsymbols, const uint32_t *__restrict t,
                        uint16_t *__restrict freq);
