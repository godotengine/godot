// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Color Cache for WebP Lossless
//
// Authors: Jyrki Alakuijala (jyrki@google.com)
//          Urvang Joshi (urvang@google.com)

#ifndef WEBP_UTILS_COLOR_CACHE_H_
#define WEBP_UTILS_COLOR_CACHE_H_

#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

// Main color cache struct.
typedef struct {
  uint32_t *colors_;  // color entries
  int hash_shift_;    // Hash shift: 32 - hash_bits.
} VP8LColorCache;

static const uint32_t kHashMul = 0x1e35a7bd;

static WEBP_INLINE uint32_t VP8LColorCacheLookup(
    const VP8LColorCache* const cc, uint32_t key) {
  assert(key <= (~0U >> cc->hash_shift_));
  return cc->colors_[key];
}

static WEBP_INLINE void VP8LColorCacheInsert(const VP8LColorCache* const cc,
                                             uint32_t argb) {
  const uint32_t key = (kHashMul * argb) >> cc->hash_shift_;
  cc->colors_[key] = argb;
}

static WEBP_INLINE int VP8LColorCacheGetIndex(const VP8LColorCache* const cc,
                                              uint32_t argb) {
  return (kHashMul * argb) >> cc->hash_shift_;
}

static WEBP_INLINE int VP8LColorCacheContains(const VP8LColorCache* const cc,
                                              uint32_t argb) {
  const uint32_t key = (kHashMul * argb) >> cc->hash_shift_;
  return cc->colors_[key] == argb;
}

//------------------------------------------------------------------------------

// Initializes the color cache with 'hash_bits' bits for the keys.
// Returns false in case of memory error.
int VP8LColorCacheInit(VP8LColorCache* const color_cache, int hash_bits);

// Delete the memory associated to color cache.
void VP8LColorCacheClear(VP8LColorCache* const color_cache);

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif  // WEBP_UTILS_COLOR_CACHE_H_
