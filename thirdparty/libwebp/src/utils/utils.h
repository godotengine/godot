// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Misc. common utility functions
//
// Authors: Skal (pascal.massimino@gmail.com)
//          Urvang (urvang@google.com)

#ifndef WEBP_UTILS_UTILS_H_
#define WEBP_UTILS_UTILS_H_

#ifdef HAVE_CONFIG_H
#include "src/webp/config.h"
#endif

#include <assert.h>
#include <limits.h>

#include "src/dsp/dsp.h"
#include "src/webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Memory allocation

// This is the maximum memory amount that libwebp will ever try to allocate.
#ifndef WEBP_MAX_ALLOCABLE_MEMORY
#if SIZE_MAX > (1ULL << 34)
#define WEBP_MAX_ALLOCABLE_MEMORY (1ULL << 34)
#else
// For 32-bit targets keep this below INT_MAX to avoid valgrind warnings.
#define WEBP_MAX_ALLOCABLE_MEMORY ((1ULL << 31) - (1 << 16))
#endif
#endif  // WEBP_MAX_ALLOCABLE_MEMORY

// size-checking safe malloc/calloc: verify that the requested size is not too
// large, or return NULL. You don't need to call these for constructs like
// malloc(sizeof(foo)), but only if there's picture-dependent size involved
// somewhere (like: malloc(num_pixels * sizeof(*something))). That's why this
// safe malloc() borrows the signature from calloc(), pointing at the dangerous
// underlying multiply involved.
WEBP_EXTERN void* WebPSafeMalloc(uint64_t nmemb, size_t size);
// Note that WebPSafeCalloc() expects the second argument type to be 'size_t'
// in order to favor the "calloc(num_foo, sizeof(foo))" pattern.
WEBP_EXTERN void* WebPSafeCalloc(uint64_t nmemb, size_t size);

// Companion deallocation function to the above allocations.
WEBP_EXTERN void WebPSafeFree(void* const ptr);

//------------------------------------------------------------------------------
// Alignment

#define WEBP_ALIGN_CST 31
#define WEBP_ALIGN(PTR) (((uintptr_t)(PTR) + WEBP_ALIGN_CST) & ~WEBP_ALIGN_CST)

#include <string.h>
// memcpy() is the safe way of moving potentially unaligned 32b memory.
static WEBP_INLINE uint32_t WebPMemToUint32(const uint8_t* const ptr) {
  uint32_t A;
  memcpy(&A, ptr, sizeof(A));
  return A;
}
static WEBP_INLINE void WebPUint32ToMem(uint8_t* const ptr, uint32_t val) {
  memcpy(ptr, &val, sizeof(val));
}

//------------------------------------------------------------------------------
// Reading/writing data.

// Read 16, 24 or 32 bits stored in little-endian order.
static WEBP_INLINE int GetLE16(const uint8_t* const data) {
  return (int)(data[0] << 0) | (data[1] << 8);
}

static WEBP_INLINE int GetLE24(const uint8_t* const data) {
  return GetLE16(data) | (data[2] << 16);
}

static WEBP_INLINE uint32_t GetLE32(const uint8_t* const data) {
  return GetLE16(data) | ((uint32_t)GetLE16(data + 2) << 16);
}

// Store 16, 24 or 32 bits in little-endian order.
static WEBP_INLINE void PutLE16(uint8_t* const data, int val) {
  assert(val < (1 << 16));
  data[0] = (val >> 0);
  data[1] = (val >> 8);
}

static WEBP_INLINE void PutLE24(uint8_t* const data, int val) {
  assert(val < (1 << 24));
  PutLE16(data, val & 0xffff);
  data[2] = (val >> 16);
}

static WEBP_INLINE void PutLE32(uint8_t* const data, uint32_t val) {
  PutLE16(data, (int)(val & 0xffff));
  PutLE16(data + 2, (int)(val >> 16));
}

// Returns 31 ^ clz(n) = log2(n). This is the default C-implementation, either
// based on table or not. Can be used as fallback if clz() is not available.
#define WEBP_NEED_LOG_TABLE_8BIT
extern const uint8_t WebPLogTable8bit[256];
static WEBP_INLINE int WebPLog2FloorC(uint32_t n) {
  int log_value = 0;
  while (n >= 256) {
    log_value += 8;
    n >>= 8;
  }
  return log_value + WebPLogTable8bit[n];
}

// Returns (int)floor(log2(n)). n must be > 0.
// use GNU builtins where available.
#if defined(__GNUC__) && \
    ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4)
static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  return 31 ^ __builtin_clz(n);
}
#elif defined(_MSC_VER) && _MSC_VER > 1310 && \
      (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#pragma intrinsic(_BitScanReverse)

static WEBP_INLINE int BitsLog2Floor(uint32_t n) {
  unsigned long first_set_bit;
  _BitScanReverse(&first_set_bit, n);
  return first_set_bit;
}
#else   // default: use the C-version.
static WEBP_INLINE int BitsLog2Floor(uint32_t n) { return WebPLog2FloorC(n); }
#endif

//------------------------------------------------------------------------------
// Pixel copying.

struct WebPPicture;

// Copy width x height pixels from 'src' to 'dst' honoring the strides.
WEBP_EXTERN void WebPCopyPlane(const uint8_t* src, int src_stride,
                               uint8_t* dst, int dst_stride,
                               int width, int height);

// Copy ARGB pixels from 'src' to 'dst' honoring strides. 'src' and 'dst' are
// assumed to be already allocated and using ARGB data.
WEBP_EXTERN void WebPCopyPixels(const struct WebPPicture* const src,
                                struct WebPPicture* const dst);

//------------------------------------------------------------------------------
// Unique colors.

// Returns count of unique colors in 'pic', assuming pic->use_argb is true.
// If the unique color count is more than MAX_PALETTE_SIZE, returns
// MAX_PALETTE_SIZE+1.
// If 'palette' is not NULL and number of unique colors is less than or equal to
// MAX_PALETTE_SIZE, also outputs the actual unique colors into 'palette'.
// Note: 'palette' is assumed to be an array already allocated with at least
// MAX_PALETTE_SIZE elements.
WEBP_EXTERN int WebPGetColorPalette(const struct WebPPicture* const pic,
                                    uint32_t* const palette);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_UTILS_H_ */
