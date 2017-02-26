// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Color Cache for WebP Lossless
//
// Authors: Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#ifndef WEBP_UTILS_COLOR_CACHE_H_
#define WEBP_UTILS_COLOR_CACHE_H_

#include "../webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Main color cache struct.
typedef struct {
  uint32_t *colors_;  // color entries
  int hash_shift_;    // Hash shift: 32 - hash_bits_.
  int hash_bits_;
} VP8LColorCache;

static const uint64_t kHashMul = 0x1e35a7bdull;

static WEBP_INLINE int HashPix(uint32_t argb, int shift) {
  return (int)(((argb * kHashMul) & 0xffffffffu) >> shift);
}

static WEBP_INLINE uint32_t VP8LColorCacheLookup(
    const VP8LColorCache* const cc, uint32_t key) {
  assert((key >> cc->hash_bits_) == 0u);
  return cc->colors_[key];
}

static WEBP_INLINE void VP8LColorCacheSet(const VP8LColorCache* const cc,
                                          uint32_t key, uint32_t argb) {
  assert((key >> cc->hash_bits_) == 0u);
  cc->colors_[key] = argb;
}

static WEBP_INLINE void VP8LColorCacheInsert(const VP8LColorCache* const cc,
                                             uint32_t argb) {
  const int key = HashPix(argb, cc->hash_shift_);
  cc->colors_[key] = argb;
}

static WEBP_INLINE int VP8LColorCacheGetIndex(const VP8LColorCache* const cc,
                                              uint32_t argb) {
  return HashPix(argb, cc->hash_shift_);
}

// Return the key if cc contains argb, and -1 otherwise.
static WEBP_INLINE int VP8LColorCacheContains(const VP8LColorCache* const cc,
                                              uint32_t argb) {
  const int key = HashPix(argb, cc->hash_shift_);
  return (cc->colors_[key] == argb) ? key : -1;
}

//------------------------------------------------------------------------------

// Initializes the color cache with 'hash_bits' bits for the keys.
// Returns false in case of memory error.
int VP8LColorCacheInit(VP8LColorCache* const color_cache, int hash_bits);

void VP8LColorCacheCopy(const VP8LColorCache* const src,
                        VP8LColorCache* const dst);

// Delete the memory associated to color cache.
void VP8LColorCacheClear(VP8LColorCache* const color_cache);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}
#endif

#endif  // WEBP_UTILS_COLOR_CACHE_H_
