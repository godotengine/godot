// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Bit writing and boolean coder
//
// Author: Skal (pascal.massimino@gmail.com)

#ifndef WEBP_UTILS_BIT_WRITER_H_
#define WEBP_UTILS_BIT_WRITER_H_

#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Bit-writing

typedef struct VP8BitWriter VP8BitWriter;
struct VP8BitWriter {
  int32_t  range_;      // range-1
  int32_t  value_;
  int      run_;        // number of outstanding bits
  int      nb_bits_;    // number of pending bits
  uint8_t* buf_;        // internal buffer. Re-allocated regularly. Not owned.
  size_t   pos_;
  size_t   max_pos_;
  int      error_;      // true in case of error
};

// Initialize the object. Allocates some initial memory based on expected_size.
int VP8BitWriterInit(VP8BitWriter* const bw, size_t expected_size);
// Finalize the bitstream coding. Returns a pointer to the internal buffer.
uint8_t* VP8BitWriterFinish(VP8BitWriter* const bw);
// Release any pending memory and zeroes the object. Not a mandatory call.
// Only useful in case of error, when the internal buffer hasn't been grabbed!
void VP8BitWriterWipeOut(VP8BitWriter* const bw);

int VP8PutBit(VP8BitWriter* const bw, int bit, int prob);
int VP8PutBitUniform(VP8BitWriter* const bw, int bit);
void VP8PutValue(VP8BitWriter* const bw, int value, int nb_bits);
void VP8PutSignedValue(VP8BitWriter* const bw, int value, int nb_bits);

// Appends some bytes to the internal buffer. Data is copied.
int VP8BitWriterAppend(VP8BitWriter* const bw,
                       const uint8_t* data, size_t size);

// return approximate write position (in bits)
static WEBP_INLINE uint64_t VP8BitWriterPos(const VP8BitWriter* const bw) {
  return (uint64_t)(bw->pos_ + bw->run_) * 8 + 8 + bw->nb_bits_;
}

// Returns a pointer to the internal buffer.
static WEBP_INLINE uint8_t* VP8BitWriterBuf(const VP8BitWriter* const bw) {
  return bw->buf_;
}
// Returns the size of the internal buffer.
static WEBP_INLINE size_t VP8BitWriterSize(const VP8BitWriter* const bw) {
  return bw->pos_;
}

//------------------------------------------------------------------------------
// VP8LBitWriter
// TODO(vikasa): VP8LBitWriter is copied as-is from lossless code. There's scope
// of re-using VP8BitWriter. Will evaluate once basic lossless encoder is
// implemented.

typedef struct {
  uint8_t* buf_;
  size_t bit_pos_;
  size_t max_bytes_;

  // After all bits are written, the caller must observe the state of
  // error_. A value of 1 indicates that a memory allocation failure
  // has happened during bit writing. A value of 0 indicates successful
  // writing of bits.
  int error_;
} VP8LBitWriter;

static WEBP_INLINE size_t VP8LBitWriterNumBytes(VP8LBitWriter* const bw) {
  return (bw->bit_pos_ + 7) >> 3;
}

static WEBP_INLINE uint8_t* VP8LBitWriterFinish(VP8LBitWriter* const bw) {
  return bw->buf_;
}

// Returns 0 in case of memory allocation error.
int VP8LBitWriterInit(VP8LBitWriter* const bw, size_t expected_size);

void VP8LBitWriterDestroy(VP8LBitWriter* const bw);

// This function writes bits into bytes in increasing addresses, and within
// a byte least-significant-bit first.
//
// The function can write up to 16 bits in one go with WriteBits
// Example: let's assume that 3 bits (Rs below) have been written already:
//
// BYTE-0     BYTE+1       BYTE+2
//
// 0000 0RRR    0000 0000    0000 0000
//
// Now, we could write 5 or less bits in MSB by just sifting by 3
// and OR'ing to BYTE-0.
//
// For n bits, we take the last 5 bytes, OR that with high bits in BYTE-0,
// and locate the rest in BYTE+1 and BYTE+2.
//
// VP8LBitWriter's error_ flag is set in case of  memory allocation error.
void VP8LWriteBits(VP8LBitWriter* const bw, int n_bits, uint32_t bits);

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_BIT_WRITER_H_ */
