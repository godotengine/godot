// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Specific inlined methods for boolean decoder [VP8GetBit() ...]
// This file should be included by the .c sources that actually need to call
// these methods.
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_UTILS_BIT_READER_INL_H_
#define WEBP_UTILS_BIT_READER_INL_H_

#ifdef HAVE_CONFIG_H
#include "../webp/config.h"
#endif

#ifdef WEBP_FORCE_ALIGNED
#include <string.h>  // memcpy
#endif

#include "../dsp/dsp.h"
#include "./bit_reader.h"
#include "./endian_inl.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Derived type lbit_t = natural type for memory I/O

#if   (BITS > 32)
typedef uint64_t lbit_t;
#elif (BITS > 16)
typedef uint32_t lbit_t;
#elif (BITS >  8)
typedef uint16_t lbit_t;
#else
typedef uint8_t lbit_t;
#endif

extern const uint8_t kVP8Log2Range[128];
extern const uint8_t kVP8NewRange[128];

// special case for the tail byte-reading
void VP8LoadFinalBytes(VP8BitReader* const br);

//------------------------------------------------------------------------------
// Inlined critical functions

// makes sure br->value_ has at least BITS bits worth of data
static WEBP_UBSAN_IGNORE_UNDEF WEBP_INLINE
void VP8LoadNewBytes(VP8BitReader* const br) {
  assert(br != NULL && br->buf_ != NULL);
  // Read 'BITS' bits at a time if possible.
  if (br->buf_ < br->buf_max_) {
    // convert memory type to register type (with some zero'ing!)
    bit_t bits;
#if defined(WEBP_FORCE_ALIGNED)
    lbit_t in_bits;
    memcpy(&in_bits, br->buf_, sizeof(in_bits));
#elif defined(WEBP_USE_MIPS32)
    // This is needed because of un-aligned read.
    lbit_t in_bits;
    lbit_t* p_buf_ = (lbit_t*)br->buf_;
    __asm__ volatile(
      ".set   push                             \n\t"
      ".set   at                               \n\t"
      ".set   macro                            \n\t"
      "ulw    %[in_bits], 0(%[p_buf_])         \n\t"
      ".set   pop                              \n\t"
      : [in_bits]"=r"(in_bits)
      : [p_buf_]"r"(p_buf_)
      : "memory", "at"
    );
#else
    const lbit_t in_bits = *(const lbit_t*)br->buf_;
#endif
    br->buf_ += BITS >> 3;
#if !defined(WORDS_BIGENDIAN)
#if (BITS > 32)
    bits = BSwap64(in_bits);
    bits >>= 64 - BITS;
#elif (BITS >= 24)
    bits = BSwap32(in_bits);
    bits >>= (32 - BITS);
#elif (BITS == 16)
    bits = BSwap16(in_bits);
#else   // BITS == 8
    bits = (bit_t)in_bits;
#endif  // BITS > 32
#else    // WORDS_BIGENDIAN
    bits = (bit_t)in_bits;
    if (BITS != 8 * sizeof(bit_t)) bits >>= (8 * sizeof(bit_t) - BITS);
#endif
    br->value_ = bits | (br->value_ << BITS);
    br->bits_ += BITS;
  } else {
    VP8LoadFinalBytes(br);    // no need to be inlined
  }
}

// Read a bit with proba 'prob'. Speed-critical function!
static WEBP_INLINE int VP8GetBit(VP8BitReader* const br, int prob) {
  // Don't move this declaration! It makes a big speed difference to store
  // 'range' *before* calling VP8LoadNewBytes(), even if this function doesn't
  // alter br->range_ value.
  range_t range = br->range_;
  if (br->bits_ < 0) {
    VP8LoadNewBytes(br);
  }
  {
    const int pos = br->bits_;
    const range_t split = (range * prob) >> 8;
    const range_t value = (range_t)(br->value_ >> pos);
#if defined(__arm__) || defined(_M_ARM)      // ARM-specific
    const int bit = ((int)(split - value) >> 31) & 1;
    if (value > split) {
      range -= split + 1;
      br->value_ -= (bit_t)(split + 1) << pos;
    } else {
      range = split;
    }
#else  // faster version on x86
    int bit;  // Don't use 'const int bit = (value > split);", it's slower.
    if (value > split) {
      range -= split + 1;
      br->value_ -= (bit_t)(split + 1) << pos;
      bit = 1;
    } else {
      range = split;
      bit = 0;
    }
#endif
    if (range <= (range_t)0x7e) {
      const int shift = kVP8Log2Range[range];
      range = kVP8NewRange[range];
      br->bits_ -= shift;
    }
    br->range_ = range;
    return bit;
  }
}

// simplified version of VP8GetBit() for prob=0x80 (note shift is always 1 here)
static WEBP_INLINE int VP8GetSigned(VP8BitReader* const br, int v) {
  if (br->bits_ < 0) {
    VP8LoadNewBytes(br);
  }
  {
    const int pos = br->bits_;
    const range_t split = br->range_ >> 1;
    const range_t value = (range_t)(br->value_ >> pos);
    const int32_t mask = (int32_t)(split - value) >> 31;  // -1 or 0
    br->bits_ -= 1;
    br->range_ += mask;
    br->range_ |= 1;
    br->value_ -= (bit_t)((split + 1) & mask) << pos;
    return (v ^ mask) - mask;
  }
}

#ifdef __cplusplus
}    // extern "C"
#endif

#endif   // WEBP_UTILS_BIT_READER_INL_H_
