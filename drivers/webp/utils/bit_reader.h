// Copyright 2010 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Boolean decoder
//
// Author: Skal (pascal.massimino@gmail.com)
//         Vikas Arora (vikaas.arora@gmail.com)

#ifndef WEBP_UTILS_BIT_READER_H_
#define WEBP_UTILS_BIT_READER_H_

#include <assert.h>
#ifdef _MSC_VER
#include <stdlib.h>  // _byteswap_ulong
#endif
#include <string.h>  // For memcpy
#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define BITS 32     // can be 32, 16 or 8
#define MASK ((((bit_t)1) << (BITS)) - 1)
#if (BITS == 32)
typedef uint64_t bit_t;   // natural register type
typedef uint32_t lbit_t;  // natural type for memory I/O
#elif (BITS == 16)
typedef uint32_t bit_t;
typedef uint16_t lbit_t;
#else
typedef uint32_t bit_t;
typedef uint8_t lbit_t;
#endif

//------------------------------------------------------------------------------
// Bitreader and code-tree reader

typedef struct VP8BitReader VP8BitReader;
struct VP8BitReader {
  const uint8_t* buf_;        // next byte to be read
  const uint8_t* buf_end_;    // end of read buffer
  int eof_;                   // true if input is exhausted

  // boolean decoder
  bit_t range_;            // current range minus 1. In [127, 254] interval.
  bit_t value_;            // current value
  int missing_;            // number of missing bits in value_ (8bit)
};

// Initialize the bit reader and the boolean decoder.
void VP8InitBitReader(VP8BitReader* const br,
                      const uint8_t* const start, const uint8_t* const end);

// return the next value made of 'num_bits' bits
uint32_t VP8GetValue(VP8BitReader* const br, int num_bits);
static WEBP_INLINE uint32_t VP8Get(VP8BitReader* const br) {
  return VP8GetValue(br, 1);
}

// return the next value with sign-extension.
int32_t VP8GetSignedValue(VP8BitReader* const br, int num_bits);

// Read a bit with proba 'prob'. Speed-critical function!
extern const uint8_t kVP8Log2Range[128];
extern const bit_t kVP8NewRange[128];

void VP8LoadFinalBytes(VP8BitReader* const br);    // special case for the tail

static WEBP_INLINE void VP8LoadNewBytes(VP8BitReader* const br) {
  assert(br && br->buf_);
  // Read 'BITS' bits at a time if possible.
  if (br->buf_ + sizeof(lbit_t) <= br->buf_end_) {
    // convert memory type to register type (with some zero'ing!)
    bit_t bits;
    lbit_t in_bits = *(lbit_t*)br->buf_;
    br->buf_ += (BITS) >> 3;
#if !defined(__BIG_ENDIAN__)
#if (BITS == 32)
#if defined(__i386__) || defined(__x86_64__)
    __asm__ volatile("bswap %k0" : "=r"(in_bits) : "0"(in_bits));
    bits = (bit_t)in_bits;   // 32b -> 64b zero-extension
#elif defined(_MSC_VER)
    bits = _byteswap_ulong(in_bits);
#else
    bits = (bit_t)(in_bits >> 24) | ((in_bits >> 8) & 0xff00)
         | ((in_bits << 8) & 0xff0000)  | (in_bits << 24);
#endif  // x86
#elif (BITS == 16)
    // gcc will recognize a 'rorw $8, ...' here:
    bits = (bit_t)(in_bits >> 8) | ((in_bits & 0xff) << 8);
#endif
#else    // LITTLE_ENDIAN
    bits = (bit_t)in_bits;
#endif
    br->value_ |= bits << br->missing_;
    br->missing_ -= (BITS);
  } else {
    VP8LoadFinalBytes(br);    // no need to be inlined
  }
}

static WEBP_INLINE int VP8BitUpdate(VP8BitReader* const br, bit_t split) {
  const bit_t value_split = split | (MASK);
  if (br->missing_ > 0) {  // Make sure we have a least BITS bits in 'value_'
    VP8LoadNewBytes(br);
  }
  if (br->value_ > value_split) {
    br->range_ -= value_split + 1;
    br->value_ -= value_split + 1;
    return 1;
  } else {
    br->range_ = value_split;
    return 0;
  }
}

static WEBP_INLINE void VP8Shift(VP8BitReader* const br) {
  // range_ is in [0..127] interval here.
  const int idx = br->range_ >> (BITS);
  const int shift = kVP8Log2Range[idx];
  br->range_ = kVP8NewRange[idx];
  br->value_ <<= shift;
  br->missing_ += shift;
}

static WEBP_INLINE int VP8GetBit(VP8BitReader* const br, int prob) {
  // It's important to avoid generating a 64bit x 64bit multiply here.
  // We just need an 8b x 8b after all.
  const bit_t split =
      (bit_t)((uint32_t)(br->range_ >> (BITS)) * prob) << ((BITS) - 8);
  const int bit = VP8BitUpdate(br, split);
  if (br->range_ <= (((bit_t)0x7e << (BITS)) | (MASK))) {
    VP8Shift(br);
  }
  return bit;
}

static WEBP_INLINE int VP8GetSigned(VP8BitReader* const br, int v) {
  const bit_t split = (br->range_ >> 1);
  const int bit = VP8BitUpdate(br, split);
  VP8Shift(br);
  return bit ? -v : v;
}


// -----------------------------------------------------------------------------
// Bitreader

typedef struct {
  uint64_t       val_;
  const uint8_t* buf_;
  size_t         len_;
  size_t         pos_;
  int            bit_pos_;
  int            eos_;
  int            error_;
} VP8LBitReader;

void VP8LInitBitReader(VP8LBitReader* const br,
                       const uint8_t* const start,
                       size_t length);

//  Sets a new data buffer.
void VP8LBitReaderSetBuffer(VP8LBitReader* const br,
                            const uint8_t* const buffer, size_t length);

// Reads the specified number of bits from Read Buffer.
// Flags an error in case end_of_stream or n_bits is more than allowed limit.
// Flags eos if this read attempt is going to cross the read buffer.
uint32_t VP8LReadBits(VP8LBitReader* const br, int n_bits);

// Reads one bit from Read Buffer. Flags an error in case end_of_stream.
// Flags eos after reading last bit from the buffer.
uint32_t VP8LReadOneBit(VP8LBitReader* const br);

// VP8LReadOneBitUnsafe is faster than VP8LReadOneBit, but it can be called only
// 32 times after the last VP8LFillBitWindow. Any subsequent calls
// (without VP8LFillBitWindow) will return invalid data.
static WEBP_INLINE uint32_t VP8LReadOneBitUnsafe(VP8LBitReader* const br) {
  const uint32_t val = (br->val_ >> br->bit_pos_) & 1;
  ++br->bit_pos_;
  return val;
}

// Advances the Read buffer by 4 bytes to make room for reading next 32 bits.
void VP8LFillBitWindow(VP8LBitReader* const br);

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_BIT_READER_H_ */
