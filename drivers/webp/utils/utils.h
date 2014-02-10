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

#include <assert.h>

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Memory allocation

// This is the maximum memory amount that libwebp will ever try to allocate.
#define WEBP_MAX_ALLOCABLE_MEMORY (1ULL << 40)

// size-checking safe malloc/calloc: verify that the requested size is not too
// large, or return NULL. You don't need to call these for constructs like
// malloc(sizeof(foo)), but only if there's picture-dependent size involved
// somewhere (like: malloc(num_pixels * sizeof(*something))). That's why this
// safe malloc() borrows the signature from calloc(), pointing at the dangerous
// underlying multiply involved.
void* WebPSafeMalloc(uint64_t nmemb, size_t size);
// Note that WebPSafeCalloc() expects the second argument type to be 'size_t'
// in order to favor the "calloc(num_foo, sizeof(foo))" pattern.
void* WebPSafeCalloc(uint64_t nmemb, size_t size);

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
  return (uint32_t)GetLE16(data) | (GetLE16(data + 2) << 16);
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

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_UTILS_H_ */
