// Copyright 2010 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
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
#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// The Boolean decoder needs to maintain infinite precision on the value_ field.
// However, since range_ is only 8bit, we only need an active window of 8 bits
// for value_. Left bits (MSB) gets zeroed and shifted away when value_ falls
// below 128, range_ is updated, and fresh bits read from the bitstream are
// brought in as LSB.
// To avoid reading the fresh bits one by one (slow), we cache a few of them
// ahead (actually, we cache BITS of them ahead. See below). There's two
// strategies regarding how to shift these looked-ahead fresh bits into the
// 8bit window of value_: either we shift them in, while keeping the position of
// the window fixed. Or we slide the window to the right while keeping the cache
// bits at a fixed, right-justified, position.
//
//  Example, for BITS=16: here is the content of value_ for both strategies:
//
//          !USE_RIGHT_JUSTIFY            ||        USE_RIGHT_JUSTIFY
//                                        ||
//   <- 8b -><- 8b -><- BITS bits  ->     ||  <- 8b+3b -><- 8b -><- 13 bits ->
//   [unused][value_][cached bits][0]     ||  [unused...][value_][cached bits]
//  [........00vvvvvvBBBBBBBBBBBBB000]LSB || [...........00vvvvvvBBBBBBBBBBBBB]
//                                        ||
// After calling VP8Shift(), where we need to shift away two zeros:
//  [........vvvvvvvvBBBBBBBBBBB00000]LSB || [.............vvvvvvvvBBBBBBBBBBB]
//                                        ||
// Just before we need to call VP8LoadNewBytes(), the situation is:
//  [........vvvvvv000000000000000000]LSB || [..........................vvvvvv]
//                                        ||
// And just after calling VP8LoadNewBytes():
//  [........vvvvvvvvBBBBBBBBBBBBBBBB]LSB || [........vvvvvvvvBBBBBBBBBBBBBBBB]
//
// -> we're back to eight active 'value_' bits (marked 'v') and BITS cached
// bits (marked 'B')
//
// The right-justify strategy tends to use less shifts and is often faster.

//------------------------------------------------------------------------------
// BITS can be any multiple of 8 from 8 to 56 (inclusive).
// Pick values that fit natural register size.

#if !defined(WEBP_REFERENCE_IMPLEMENTATION)

#define USE_RIGHT_JUSTIFY

#if defined(__i386__) || defined(_M_IX86)      // x86 32bit
#define BITS 16
#elif defined(__x86_64__) || defined(_M_X64)   // x86 64bit
#define BITS 56
#elif defined(__arm__) || defined(_M_ARM)      // ARM
#define BITS 24
#else                      // reasonable default
#define BITS 24
#endif

#else     // reference choices

#define USE_RIGHT_JUSTIFY
#define BITS 8

#endif

//------------------------------------------------------------------------------
// Derived types and constants

// bit_t = natural register type
// lbit_t = natural type for memory I/O

#if (BITS > 32)
typedef uint64_t bit_t;
typedef uint64_t lbit_t;
#elif (BITS == 32)
typedef uint64_t bit_t;
typedef uint32_t lbit_t;
#elif (BITS == 24)
typedef uint32_t bit_t;
typedef uint32_t lbit_t;
#elif (BITS == 16)
typedef uint32_t bit_t;
typedef uint16_t lbit_t;
#else
typedef uint32_t bit_t;
typedef uint8_t lbit_t;
#endif

#ifndef USE_RIGHT_JUSTIFY
typedef bit_t range_t;     // type for storing range_
#define MASK ((((bit_t)1) << (BITS)) - 1)
#else
typedef uint32_t range_t;  // range_ only uses 8bits here. No need for bit_t.
#endif

//------------------------------------------------------------------------------
// Bitreader

typedef struct VP8BitReader VP8BitReader;
struct VP8BitReader {
  const uint8_t* buf_;        // next byte to be read
  const uint8_t* buf_end_;    // end of read buffer
  int eof_;                   // true if input is exhausted

  // boolean decoder
  range_t range_;            // current range minus 1. In [127, 254] interval.
  bit_t value_;              // current value
  int bits_;                 // number of valid bits left
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
extern const range_t kVP8NewRange[128];

void VP8LoadFinalBytes(VP8BitReader* const br);    // special case for the tail

static WEBP_INLINE void VP8LoadNewBytes(VP8BitReader* const br) {
  assert(br != NULL && br->buf_ != NULL);
  // Read 'BITS' bits at a time if possible.
  if (br->buf_ + sizeof(lbit_t) <= br->buf_end_) {
    // convert memory type to register type (with some zero'ing!)
    bit_t bits;
    const lbit_t in_bits = *(const lbit_t*)br->buf_;
    br->buf_ += (BITS) >> 3;
#if !defined(__BIG_ENDIAN__)
#if (BITS > 32)
// gcc 4.3 has builtin functions for swap32/swap64
#if defined(__GNUC__) && \
           (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))
    bits = (bit_t)__builtin_bswap64(in_bits);
#elif defined(_MSC_VER)
    bits = (bit_t)_byteswap_uint64(in_bits);
#elif defined(__x86_64__)
    __asm__ volatile("bswapq %0" : "=r"(bits) : "0"(in_bits));
#else  // generic code for swapping 64-bit values (suggested by bdb@)
    bits = (bit_t)in_bits;
    bits = ((bits & 0xffffffff00000000ull) >> 32) |
           ((bits & 0x00000000ffffffffull) << 32);
    bits = ((bits & 0xffff0000ffff0000ull) >> 16) |
           ((bits & 0x0000ffff0000ffffull) << 16);
    bits = ((bits & 0xff00ff00ff00ff00ull) >> 8) |
           ((bits & 0x00ff00ff00ff00ffull) << 8);
#endif
    bits >>= 64 - BITS;
#elif (BITS >= 24)
#if defined(__i386__) || defined(__x86_64__)
    {
      lbit_t swapped_in_bits;
      __asm__ volatile("bswap %k0" : "=r"(swapped_in_bits) : "0"(in_bits));
      bits = (bit_t)swapped_in_bits;   // 24b/32b -> 32b/64b zero-extension
    }
#elif defined(_MSC_VER)
    bits = (bit_t)_byteswap_ulong(in_bits);
#else
    bits = (bit_t)(in_bits >> 24) | ((in_bits >> 8) & 0xff00)
         | ((in_bits << 8) & 0xff0000)  | (in_bits << 24);
#endif  // x86
    bits >>= (32 - BITS);
#elif (BITS == 16)
    // gcc will recognize a 'rorw $8, ...' here:
    bits = (bit_t)(in_bits >> 8) | ((in_bits & 0xff) << 8);
#else   // BITS == 8
    bits = (bit_t)in_bits;
#endif
#else    // BIG_ENDIAN
    bits = (bit_t)in_bits;
    if (BITS != 8 * sizeof(bit_t)) bits >>= (8 * sizeof(bit_t) - BITS);
#endif
#ifndef USE_RIGHT_JUSTIFY
    br->value_ |= bits << (-br->bits_);
#else
    br->value_ = bits | (br->value_ << (BITS));
#endif
    br->bits_ += (BITS);
  } else {
    VP8LoadFinalBytes(br);    // no need to be inlined
  }
}

static WEBP_INLINE int VP8BitUpdate(VP8BitReader* const br, range_t split) {
  if (br->bits_ < 0) {  // Make sure we have a least BITS bits in 'value_'
    VP8LoadNewBytes(br);
  }
#ifndef USE_RIGHT_JUSTIFY
  split |= (MASK);
  if (br->value_ > split) {
    br->range_ -= split + 1;
    br->value_ -= split + 1;
    return 1;
  } else {
    br->range_ = split;
    return 0;
  }
#else
  {
    const int pos = br->bits_;
    const range_t value = (range_t)(br->value_ >> pos);
    if (value > split) {
      br->range_ -= split + 1;
      br->value_ -= (bit_t)(split + 1) << pos;
      return 1;
    } else {
      br->range_ = split;
      return 0;
    }
  }
#endif
}

static WEBP_INLINE void VP8Shift(VP8BitReader* const br) {
#ifndef USE_RIGHT_JUSTIFY
  // range_ is in [0..127] interval here.
  const bit_t idx = br->range_ >> (BITS);
  const int shift = kVP8Log2Range[idx];
  br->range_ = kVP8NewRange[idx];
  br->value_ <<= shift;
  br->bits_ -= shift;
#else
  const int shift = kVP8Log2Range[br->range_];
  assert(br->range_ < (range_t)128);
  br->range_ = kVP8NewRange[br->range_];
  br->bits_ -= shift;
#endif
}

static WEBP_INLINE int VP8GetBit(VP8BitReader* const br, int prob) {
#ifndef USE_RIGHT_JUSTIFY
  // It's important to avoid generating a 64bit x 64bit multiply here.
  // We just need an 8b x 8b after all.
  const range_t split =
      (range_t)((uint32_t)(br->range_ >> (BITS)) * prob) << ((BITS) - 8);
  const int bit = VP8BitUpdate(br, split);
  if (br->range_ <= (((range_t)0x7e << (BITS)) | (MASK))) {
    VP8Shift(br);
  }
  return bit;
#else
  const range_t split = (br->range_ * prob) >> 8;
  const int bit = VP8BitUpdate(br, split);
  if (br->range_ <= (range_t)0x7e) {
    VP8Shift(br);
  }
  return bit;
#endif
}

static WEBP_INLINE int VP8GetSigned(VP8BitReader* const br, int v) {
  const range_t split = (br->range_ >> 1);
  const int bit = VP8BitUpdate(br, split);
  VP8Shift(br);
  return bit ? -v : v;
}

// -----------------------------------------------------------------------------
// Bitreader for lossless format

typedef uint64_t vp8l_val_t;  // right now, this bit-reader can only use 64bit.

typedef struct {
  vp8l_val_t     val_;        // pre-fetched bits
  const uint8_t* buf_;        // input byte buffer
  size_t         len_;        // buffer length
  size_t         pos_;        // byte position in buf_
  int            bit_pos_;    // current bit-reading position in val_
  int            eos_;        // bitstream is finished
  int            error_;      // an error occurred (buffer overflow attempt...)
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

// Return the prefetched bits, so they can be looked up.
static WEBP_INLINE uint32_t VP8LPrefetchBits(VP8LBitReader* const br) {
  return (uint32_t)(br->val_ >> br->bit_pos_);
}

// For jumping over a number of bits in the bit stream when accessed with
// VP8LPrefetchBits and VP8LFillBitWindow.
static WEBP_INLINE void VP8LSetBitPos(VP8LBitReader* const br, int val) {
  br->bit_pos_ = val;
}

// Advances the read buffer by 4 bytes to make room for reading next 32 bits.
void VP8LFillBitWindow(VP8LBitReader* const br);

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_BIT_READER_H_ */
