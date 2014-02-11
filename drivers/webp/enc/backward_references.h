// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Author: Jyrki Alakuijala (jyrki@google.com)
//

#ifndef WEBP_ENC_BACKWARD_REFERENCES_H_
#define WEBP_ENC_BACKWARD_REFERENCES_H_

#include <assert.h>
#include <stdlib.h>
#include "../webp/types.h"
#include "../webp/format_constants.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// The spec allows 11, we use 9 bits to reduce memory consumption in encoding.
// Having 9 instead of 11 only removes about 0.25 % of compression density.
#define MAX_COLOR_CACHE_BITS 9

// Max ever number of codes we'll use:
#define PIX_OR_COPY_CODES_MAX \
    (NUM_LITERAL_CODES + NUM_LENGTH_CODES + (1 << MAX_COLOR_CACHE_BITS))

// -----------------------------------------------------------------------------
// PrefixEncode()

// use GNU builtins where available.
#if defined(__GNUC__) && \
    ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4)
static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  return n == 0 ? -1 : 31 ^ __builtin_clz(n);
}
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)

static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  unsigned long first_set_bit;
  return _BitScanReverse(&first_set_bit, n) ? first_set_bit : -1;
}
#else
static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  int log = 0;
  uint32_t value = n;
  int i;

  if (value == 0) return -1;
  for (i = 4; i >= 0; --i) {
    const int shift = (1 << i);
    const uint32_t x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  return log;
}
#endif

static WEBP_INLINE int VP8LBitsLog2Ceiling(uint32_t n) {
  const int floor = BitsLog2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two.
    return floor;
  else
    return floor + 1;
}

// Splitting of distance and length codes into prefixes and
// extra bits. The prefixes are encoded with an entropy code
// while the extra bits are stored just as normal bits.
static WEBP_INLINE void PrefixEncode(int distance, int* const code,
                                     int* const extra_bits_count,
                                     int* const extra_bits_value) {
  // Collect the two most significant bits where the highest bit is 1.
  const int highest_bit = BitsLog2Floor(--distance);
  // & 0x3f is to make behavior well defined when highest_bit
  // does not exist or is the least significant bit.
  const int second_highest_bit =
      (distance >> ((highest_bit - 1) & 0x3f)) & 1;
  *extra_bits_count = (highest_bit > 0) ? (highest_bit - 1) : 0;
  *extra_bits_value = distance & ((1 << *extra_bits_count) - 1);
  *code = (highest_bit > 0) ? (2 * highest_bit + second_highest_bit)
                            : (highest_bit == 0) ? 1 : 0;
}

// -----------------------------------------------------------------------------
// PixOrCopy

enum Mode {
  kLiteral,
  kCacheIdx,
  kCopy,
  kNone
};

typedef struct {
  // mode as uint8_t to make the memory layout to be exactly 8 bytes.
  uint8_t mode;
  uint16_t len;
  uint32_t argb_or_distance;
} PixOrCopy;

static WEBP_INLINE PixOrCopy PixOrCopyCreateCopy(uint32_t distance,
                                                 uint16_t len) {
  PixOrCopy retval;
  retval.mode = kCopy;
  retval.argb_or_distance = distance;
  retval.len = len;
  return retval;
}

static WEBP_INLINE PixOrCopy PixOrCopyCreateCacheIdx(int idx) {
  PixOrCopy retval;
  assert(idx >= 0);
  assert(idx < (1 << MAX_COLOR_CACHE_BITS));
  retval.mode = kCacheIdx;
  retval.argb_or_distance = idx;
  retval.len = 1;
  return retval;
}

static WEBP_INLINE PixOrCopy PixOrCopyCreateLiteral(uint32_t argb) {
  PixOrCopy retval;
  retval.mode = kLiteral;
  retval.argb_or_distance = argb;
  retval.len = 1;
  return retval;
}

static WEBP_INLINE int PixOrCopyIsLiteral(const PixOrCopy* const p) {
  return (p->mode == kLiteral);
}

static WEBP_INLINE int PixOrCopyIsCacheIdx(const PixOrCopy* const p) {
  return (p->mode == kCacheIdx);
}

static WEBP_INLINE int PixOrCopyIsCopy(const PixOrCopy* const p) {
  return (p->mode == kCopy);
}

static WEBP_INLINE uint32_t PixOrCopyLiteral(const PixOrCopy* const p,
                                             int component) {
  assert(p->mode == kLiteral);
  return (p->argb_or_distance >> (component * 8)) & 0xff;
}

static WEBP_INLINE uint32_t PixOrCopyLength(const PixOrCopy* const p) {
  return p->len;
}

static WEBP_INLINE uint32_t PixOrCopyArgb(const PixOrCopy* const p) {
  assert(p->mode == kLiteral);
  return p->argb_or_distance;
}

static WEBP_INLINE uint32_t PixOrCopyCacheIdx(const PixOrCopy* const p) {
  assert(p->mode == kCacheIdx);
  assert(p->argb_or_distance < (1U << MAX_COLOR_CACHE_BITS));
  return p->argb_or_distance;
}

static WEBP_INLINE uint32_t PixOrCopyDistance(const PixOrCopy* const p) {
  assert(p->mode == kCopy);
  return p->argb_or_distance;
}

// -----------------------------------------------------------------------------
// VP8LBackwardRefs

typedef struct {
  PixOrCopy* refs;
  int size;      // currently used
  int max_size;  // maximum capacity
} VP8LBackwardRefs;

// Initialize the object. Must be called first. 'refs' can be NULL.
void VP8LInitBackwardRefs(VP8LBackwardRefs* const refs);

// Release memory and re-initialize the object. 'refs' can be NULL.
void VP8LClearBackwardRefs(VP8LBackwardRefs* const refs);

// Allocate 'max_size' references. Returns false in case of memory error.
int VP8LBackwardRefsAlloc(VP8LBackwardRefs* const refs, int max_size);

// -----------------------------------------------------------------------------
// Main entry points

// Evaluates best possible backward references for specified quality.
// Further optimize for 2D locality if use_2d_locality flag is set.
int VP8LGetBackwardReferences(int width, int height,
                              const uint32_t* const argb,
                              int quality, int cache_bits, int use_2d_locality,
                              VP8LBackwardRefs* const best);

// Produce an estimate for a good color cache size for the image.
int VP8LCalculateEstimateForCacheSize(const uint32_t* const argb,
                                      int xsize, int ysize,
                                      int* const best_cache_bits);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif  // WEBP_ENC_BACKWARD_REFERENCES_H_
