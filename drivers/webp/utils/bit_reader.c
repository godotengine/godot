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

#include "./bit_reader.h"

#ifndef USE_RIGHT_JUSTIFY
#define MK(X) (((range_t)(X) << (BITS)) | (MASK))
#else
#define MK(X) ((range_t)(X))
#endif

//------------------------------------------------------------------------------
// VP8BitReader

void VP8InitBitReader(VP8BitReader* const br,
                      const uint8_t* const start, const uint8_t* const end) {
  assert(br != NULL);
  assert(start != NULL);
  assert(start <= end);
  br->range_   = MK(255 - 1);
  br->buf_     = start;
  br->buf_end_ = end;
  br->value_   = 0;
  br->bits_    = -8;   // to load the very first 8bits
  br->eof_     = 0;
}

const uint8_t kVP8Log2Range[128] = {
     7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  0
};

// range = (range << kVP8Log2Range[range]) + trailing 1's
const range_t kVP8NewRange[128] = {
  MK(127), MK(127), MK(191), MK(127), MK(159), MK(191), MK(223), MK(127),
  MK(143), MK(159), MK(175), MK(191), MK(207), MK(223), MK(239), MK(127),
  MK(135), MK(143), MK(151), MK(159), MK(167), MK(175), MK(183), MK(191),
  MK(199), MK(207), MK(215), MK(223), MK(231), MK(239), MK(247), MK(127),
  MK(131), MK(135), MK(139), MK(143), MK(147), MK(151), MK(155), MK(159),
  MK(163), MK(167), MK(171), MK(175), MK(179), MK(183), MK(187), MK(191),
  MK(195), MK(199), MK(203), MK(207), MK(211), MK(215), MK(219), MK(223),
  MK(227), MK(231), MK(235), MK(239), MK(243), MK(247), MK(251), MK(127),
  MK(129), MK(131), MK(133), MK(135), MK(137), MK(139), MK(141), MK(143),
  MK(145), MK(147), MK(149), MK(151), MK(153), MK(155), MK(157), MK(159),
  MK(161), MK(163), MK(165), MK(167), MK(169), MK(171), MK(173), MK(175),
  MK(177), MK(179), MK(181), MK(183), MK(185), MK(187), MK(189), MK(191),
  MK(193), MK(195), MK(197), MK(199), MK(201), MK(203), MK(205), MK(207),
  MK(209), MK(211), MK(213), MK(215), MK(217), MK(219), MK(221), MK(223),
  MK(225), MK(227), MK(229), MK(231), MK(233), MK(235), MK(237), MK(239),
  MK(241), MK(243), MK(245), MK(247), MK(249), MK(251), MK(253), MK(127)
};

#undef MK

void VP8LoadFinalBytes(VP8BitReader* const br) {
  assert(br != NULL && br->buf_ != NULL);
  // Only read 8bits at a time
  if (br->buf_ < br->buf_end_) {
#ifndef USE_RIGHT_JUSTIFY
    br->value_ |= (bit_t)(*br->buf_++) << ((BITS) - 8 - br->bits_);
#else
    br->value_ = (bit_t)(*br->buf_++) | (br->value_ << 8);
#endif
    br->bits_ += 8;
  } else if (!br->eof_) {
#ifdef USE_RIGHT_JUSTIFY
    // These are not strictly needed, but it makes the behaviour
    // consistent for both USE_RIGHT_JUSTIFY and !USE_RIGHT_JUSTIFY.
    br->value_ <<= 8;
    br->bits_ += 8;
#endif
    br->eof_ = 1;
  }
}

//------------------------------------------------------------------------------
// Higher-level calls

uint32_t VP8GetValue(VP8BitReader* const br, int bits) {
  uint32_t v = 0;
  while (bits-- > 0) {
    v |= VP8GetBit(br, 0x80) << bits;
  }
  return v;
}

int32_t VP8GetSignedValue(VP8BitReader* const br, int bits) {
  const int value = VP8GetValue(br, bits);
  return VP8Get(br) ? -value : value;
}

//------------------------------------------------------------------------------
// VP8LBitReader

#define MAX_NUM_BIT_READ 25

#define LBITS 64      // Number of bits prefetched.
#define WBITS 32      // Minimum number of bytes needed after VP8LFillBitWindow.
#define LOG8_WBITS 4  // Number of bytes needed to store WBITS bits.

static const uint32_t kBitMask[MAX_NUM_BIT_READ] = {
  0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767,
  65535, 131071, 262143, 524287, 1048575, 2097151, 4194303, 8388607, 16777215
};

void VP8LInitBitReader(VP8LBitReader* const br,
                       const uint8_t* const start,
                       size_t length) {
  size_t i;
  assert(br != NULL);
  assert(start != NULL);
  assert(length < 0xfffffff8u);   // can't happen with a RIFF chunk.

  br->buf_ = start;
  br->len_ = length;
  br->val_ = 0;
  br->pos_ = 0;
  br->bit_pos_ = 0;
  br->eos_ = 0;
  br->error_ = 0;
  for (i = 0; i < sizeof(br->val_) && i < br->len_; ++i) {
    br->val_ |= ((vp8l_val_t)br->buf_[br->pos_]) << (8 * i);
    ++br->pos_;
  }
}

void VP8LBitReaderSetBuffer(VP8LBitReader* const br,
                            const uint8_t* const buf, size_t len) {
  assert(br != NULL);
  assert(buf != NULL);
  assert(len < 0xfffffff8u);   // can't happen with a RIFF chunk.
  br->eos_ = (br->pos_ >= len);
  br->buf_ = buf;
  br->len_ = len;
}

// If not at EOS, reload up to LBITS byte-by-byte
static void ShiftBytes(VP8LBitReader* const br) {
  while (br->bit_pos_ >= 8 && br->pos_ < br->len_) {
    br->val_ >>= 8;
    br->val_ |= ((vp8l_val_t)br->buf_[br->pos_]) << (LBITS - 8);
    ++br->pos_;
    br->bit_pos_ -= 8;
  }
}

void VP8LFillBitWindow(VP8LBitReader* const br) {
  if (br->bit_pos_ >= WBITS) {
#if (defined(__x86_64__) || defined(_M_X64))
    if (br->pos_ + sizeof(br->val_) < br->len_) {
      br->val_ >>= WBITS;
      br->bit_pos_ -= WBITS;
      // The expression below needs a little-endian arch to work correctly.
      // This gives a large speedup for decoding speed.
      br->val_ |= *(const vp8l_val_t*)(br->buf_ + br->pos_) << (LBITS - WBITS);
      br->pos_ += LOG8_WBITS;
      return;
    }
#endif
    ShiftBytes(br);       // Slow path.
    if (br->pos_ == br->len_ && br->bit_pos_ >= LBITS) {
      br->eos_ = 1;
    }
  }
}

uint32_t VP8LReadBits(VP8LBitReader* const br, int n_bits) {
  assert(n_bits >= 0);
  // Flag an error if end_of_stream or n_bits is more than allowed limit.
  if (!br->eos_ && n_bits < MAX_NUM_BIT_READ) {
    const uint32_t val =
        (uint32_t)(br->val_ >> br->bit_pos_) & kBitMask[n_bits];
    const int new_bits = br->bit_pos_ + n_bits;
    br->bit_pos_ = new_bits;
    // If this read is going to cross the read buffer, set the eos flag.
    if (br->pos_ == br->len_) {
      if (new_bits >= LBITS) {
        br->eos_ = 1;
      }
    }
    ShiftBytes(br);
    return val;
  } else {
    br->error_ = 1;
    return 0;
  }
}

//------------------------------------------------------------------------------

